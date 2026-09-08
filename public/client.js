(() => {
  const micBtn = document.getElementById('micBtn');
  const textInput = document.getElementById('textInput');
  const sendTextBtn = document.getElementById('sendTextBtn');
  const clearBtn = document.getElementById('clearBtn');
  const statusEl = document.getElementById('status');
  const transcriptEl = document.getElementById('transcript');
  const replyEl = document.getElementById('reply');
  const latFirstAudio = document.getElementById('latFirstAudio');
  const latRag = document.getElementById('latRag');
  const latLlm = document.getElementById('latLlm');

  const MIC_SAMPLE_RATE = 16000;
  const PLAYBACK_SAMPLE_RATE = 24000;

  let ws = null;
  let micStream = null;
  let audioCtx = null;
  let sourceNode = null;
  let processorNode = null;
  let listening = false;
  let assistantSpeaking = false;
  let shouldReconnect = false;
  let reconnectAttempts = 0;
  const MAX_RECONNECT_ATTEMPTS = 5;
  
  // Current message display (not full history)
  let currentUserMessage = '';
  let currentAssistantReply = '';
  let isAssistantStreaming = false;

  // --- Gapless playback scheduling ---
  let playbackCtx = null;
  let nextStartTime = 0;

  function ensurePlaybackContext() {
    if (!playbackCtx) {
      playbackCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: PLAYBACK_SAMPLE_RATE });
      nextStartTime = playbackCtx.currentTime;
    }
    return playbackCtx;
  }

  // The server's Sarvam TTS config requests output_audio_codec: 'linear16'
  // (server/sarvamTts.js) — every audio_chunk is raw PCM16LE samples, NOT
  // a standalone WAV file. Treating it as audio/wav (old playWavChunk) made
  // the browser parse sample bytes as a RIFF header, hence the garbage
  // "sample rate 828715824 / channels 24375" readings. Decode as PCM and
  // schedule it on the Web Audio API timeline instead, so chunks also play
  // back-to-back with no gap between them.
  function base64ToInt16(base64) {
    const binary = atob(base64);
    const len = binary.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) bytes[i] = binary.charCodeAt(i);
    return new Int16Array(bytes.buffer);
  }

  function playPcmChunk(base64Audio) {
    const ctx = ensurePlaybackContext();
    const int16 = base64ToInt16(base64Audio);
    const float32 = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i++) float32[i] = int16[i] / 32768;

    const buffer = ctx.createBuffer(1, float32.length, PLAYBACK_SAMPLE_RATE);
    buffer.copyToChannel(float32, 0);

    const src = ctx.createBufferSource();
    src.buffer = buffer;

    // Each TTS chunk is scheduled as its own independent buffer, and raw
    // PCM chunks almost never start/end on a zero-crossing. Butting them
    // together directly produces an audible click/pop at every boundary,
    // which is what made the streamed voice sound choppy/robotic instead
    // of smooth. A few ms of gain ramp at each edge removes the
    // discontinuity without adding noticeable latency.
    const FADE_S = 0.004; // 4ms
    const startAt = Math.max(nextStartTime, ctx.currentTime);
    const duration = buffer.duration;
    const fade = Math.min(FADE_S, duration / 2); // never fade more than half a very short chunk

    const gainNode = ctx.createGain();
    gainNode.gain.setValueAtTime(0, startAt);
    gainNode.gain.linearRampToValueAtTime(1, startAt + fade);
    gainNode.gain.setValueAtTime(1, startAt + duration - fade);
    gainNode.gain.linearRampToValueAtTime(0, startAt + duration);

    src.connect(gainNode);
    gainNode.connect(ctx.destination);
    src.start(startAt);
    nextStartTime = startAt + duration;
  }

  function resetPlayback() {
    if (playbackCtx) nextStartTime = playbackCtx.currentTime;
  }

  function reconnectWebSocket() {
    if (!shouldReconnect || reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
      return;
    }

    reconnectAttempts++;
    const delay = Math.min(1000 * Math.pow(2, reconnectAttempts - 1), 10000);
    
    setTimeout(() => {
      if (!shouldReconnect) return;
      establishWebSocketConnection();
    }, delay);
  }

  function establishWebSocketConnection() {
    if (ws && (ws.readyState === WebSocket.CONNECTING || ws.readyState === WebSocket.OPEN)) {
      return;
    }

    ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/voice`);
    
    ws.onopen = () => {
      reconnectAttempts = 0;
      if (listening) {
        setStatus('Connected — listening…');
      } else {
        setStatus('Connected');
      }
    };
    
    ws.onclose = (event) => {
      if (shouldReconnect) {
        setStatus('Reconnecting…');
        reconnectWebSocket();
      } else {
        setStatus('Disconnected');
      }
    };
    
    ws.onerror = (err) => {
      if (!shouldReconnect) {
        setStatus('Connection error');
      }
    };
    
    ws.onmessage = (evt) => {
      const msg = JSON.parse(evt.data);
      handleServerMessage(msg);
    };
  }

  function arrayBufferToBase64(buf) {
    let binary = '';
    const bytes = new Uint8Array(buf);
    for (let i = 0; i < bytes.byteLength; i++) binary += String.fromCharCode(bytes[i]);
    return btoa(binary);
  }

  function floatTo16BitPCM(float32Array) {
    const buf = new ArrayBuffer(float32Array.length * 2);
    const view = new DataView(buf);
    for (let i = 0; i < float32Array.length; i++) {
      const s = Math.max(-1, Math.min(1, float32Array[i]));
      view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }
    return buf;
  }

  function pcmToWav(pcmData, sampleRate) {
    const numChannels = 1;
    const bitsPerSample = 16;
    const byteRate = sampleRate * numChannels * bitsPerSample / 8;
    const blockAlign = numChannels * bitsPerSample / 8;
    const dataSize = pcmData.byteLength;
    const buffer = new ArrayBuffer(44 + dataSize);
    const view = new DataView(buffer);

    // WAV header
    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + dataSize, true);
    writeString(view, 8, 'WAVE');
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); // PCM format
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, byteRate, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitsPerSample, true);
    writeString(view, 36, 'data');
    view.setUint32(40, dataSize, true);

    // Write PCM data
    new Uint8Array(buffer, 44).set(new Uint8Array(pcmData));

    return buffer;
  }

  function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  }

  function downsampleBuffer(buffer, inputRate, outputRate) {
    if (outputRate === inputRate) return buffer;
    const ratio = inputRate / outputRate;
    const newLength = Math.round(buffer.length / ratio);
    const result = new Float32Array(newLength);
    let offsetResult = 0;
    let offsetBuffer = 0;
    while (offsetResult < result.length) {
      const nextOffsetBuffer = Math.round((offsetResult + 1) * ratio);
      let accum = 0, count = 0;
      for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
        accum += buffer[i];
        count++;
      }
      result[offsetResult] = count ? accum / count : 0;
      offsetResult++;
      offsetBuffer = nextOffsetBuffer;
    }
    return result;
  }

  async function startListening() {
    shouldReconnect = true;
    establishWebSocketConnection();

    try {
      micStream = await navigator.mediaDevices.getUserMedia({ 
        audio: { 
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        } 
      });
    } catch (err) {
      setStatus('Microphone access denied');
      return;
    }

    audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: MIC_SAMPLE_RATE });
    
    if (audioCtx.state === 'suspended') {
      await audioCtx.resume();
    }
    
    sourceNode = audioCtx.createMediaStreamSource(micStream);

    processorNode = audioCtx.createScriptProcessor(2048, 1, 1);
    sourceNode.connect(processorNode);
    processorNode.connect(audioCtx.destination);

    processorNode.onaudioprocess = (e) => {
      if (!listening || ws.readyState !== WebSocket.OPEN) return;
      
      const input = e.inputBuffer.getChannelData(0);
      
      const pcm16 = floatTo16BitPCM(input);
      const wav = pcmToWav(pcm16, MIC_SAMPLE_RATE);
      const base64Audio = arrayBufferToBase64(wav);
      ws.send(JSON.stringify({ type: 'audio_chunk', audio: base64Audio }));
    };

    listening = true;
    micBtn.classList.add('listening');
    micBtn.textContent = 'Stop';
    setStatus('Listening…');
  }

  function stopListening() {
    listening = false;
    shouldReconnect = false;
    micBtn.classList.remove('listening');
    micBtn.textContent = 'Start talking';
    
    processorNode?.disconnect();
    sourceNode?.disconnect();
    micStream?.getTracks().forEach((t) => t.stop());
    
    if (ws) {
      ws.close();
    }
    
    setStatus('Idle');
  }

  function clearConversation() {
    currentUserMessage = '';
    currentAssistantReply = '';
    isAssistantStreaming = false;
    updateTranscriptDisplay();
    replyEl.textContent = '';
    transcriptEl.textContent = '';
    setStatus('Conversation cleared');
    setTimeout(() => setStatus('Idle'), 2000);
  }

  function setStatus(text) { statusEl.textContent = text; }

  function updateTranscriptDisplay() {
    let display = '';
    if (currentUserMessage) {
      display += `You: ${currentUserMessage}\n`;
    }
    if (currentAssistantReply) {
      display += `Assistant: ${currentAssistantReply}`;
    }
    transcriptEl.textContent = display;
    // Auto-scroll to bottom
    transcriptEl.scrollTop = transcriptEl.scrollHeight;
  }

  function handleServerMessage(msg) {
    switch (msg.type) {
      case 'transcript':
        // Update current user message
        currentUserMessage = msg.text;
        currentAssistantReply = ''; // Clear previous reply
        isAssistantStreaming = false; // Reset streaming state
        updateTranscriptDisplay();
        break;
      case 'vad':
        if (msg.signal === 'START_SPEECH') {
          setStatus('Listening…');
          // Allow user to speak even if assistant is speaking (barge-in)
          assistantSpeaking = false;
        }
        if (msg.signal === 'END_SPEECH') {
          setStatus('Thinking…');
          isAssistantStreaming = true; // Start of assistant response
        }
        break;
      case 'barge_in':
        resetPlayback();
        replyEl.textContent = '';
        assistantSpeaking = false;
        setStatus('Listening (interrupted)…');
        break;
      case 'reply_text':
        // Only update assistant reply on complete message (not during streaming)
        if (!isAssistantStreaming) {
          currentAssistantReply = msg.text;
          updateTranscriptDisplay();
        }
        replyEl.textContent = msg.text + (msg.cached ? '  ⚡ (cached)' : '');
        break;
      case 'audio_chunk':
        assistantSpeaking = true;
        playPcmChunk(msg.audio);
        setStatus('Speaking…');
        break;
      case 'turn_done':
        assistantSpeaking = false;
        isAssistantStreaming = false;
        setStatus('Listening…');
        break;
      case 'info':
        console.info('[server]', msg.message);
        break;
      case 'error':
        console.error(`[server:${msg.stage}]`, msg.message);
        setStatus(`Error: ${msg.message}`);
        break;
    }
    if (msg.latency) {
      latFirstAudio.textContent = msg.latency.first_audio_to_client_ms?.toFixed?.(0) ?? '–';
      latRag.textContent = msg.latency.rag_latency_ms?.toFixed?.(0) ?? '–';
      latLlm.textContent = msg.latency.llm_first_token_latency_ms?.toFixed?.(0) ?? '–';
    }
  }

  async function sendText() {
    const text = textInput.value.trim();
    if (!text) return;
    
    shouldReconnect = true;
    
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      establishWebSocketConnection();
    }
    
    // Wait for connection to be established
    const waitForConnection = () => {
      if (ws.readyState === WebSocket.OPEN) {
        setStatus('Connected — sending text…');
        ws.send(JSON.stringify({ type: 'text_input', text: text }));
        // Update current user message
        currentUserMessage = text;
        currentAssistantReply = '';
        isAssistantStreaming = false;
        updateTranscriptDisplay();
        textInput.value = '';
      } else if (ws.readyState === WebSocket.CONNECTING) {
        setTimeout(waitForConnection, 100);
      } else {
        establishWebSocketConnection();
        setTimeout(waitForConnection, 500);
      }
    };
    
    waitForConnection();
  }

  micBtn.addEventListener('click', () => {
    if (listening) stopListening();
    else startListening();
  });

  sendTextBtn.addEventListener('click', sendText);
  clearBtn.addEventListener('click', clearConversation);
  textInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendText();
  });
})();