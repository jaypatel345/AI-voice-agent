import 'dotenv/config';
import WebSocket from 'ws';

/**
 * Simple test to verify the pipeline works by sending text directly
 * instead of audio. This bypasses STT to test LLM + TTS + RAG.
 */

const WS_URL = `ws://localhost:${process.env.PORT || 8080}/ws/voice`;

function runTest() {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(WS_URL);
    let firstAudioAt = null;
    const t0 = Date.now();
    let latencySummary = null;

    ws.on('open', () => {
      console.log('[Test] Connected to WebSocket');
      
      // Send a text message directly (bypasses STT)
      // Use Hindi text to test TTS API
      const testText = 'नमस्ते, आप कैसे हैं?';
      console.log('[Test] Sending test text:', testText);
      ws.send(JSON.stringify({ type: 'text_input', text: testText }));
    });

    ws.on('message', (raw) => {
      const msg = JSON.parse(raw.toString());
      console.log('[Test] Received:', msg.type);
      
      if (msg.type === 'transcript') {
        console.log('[Test] Transcript:', msg.text);
      }
      if (msg.type === 'audio_chunk' && firstAudioAt == null) {
        firstAudioAt = Date.now() - t0;
        console.log('[Test] First audio received at:', firstAudioAt, 'ms');
      }
      if (msg.type === 'latency') {
        latencySummary = msg.latency;
        console.log('[Test] Latency:', latencySummary);
      }
      if (msg.type === 'reply_text') {
        console.log('[Test] Reply text received, waiting for audio...');
        // Don't close yet, wait for audio
      }
      if (msg.type === 'turn_done') {
        console.log('[Test] Turn done, closing connection');
        ws.close();
        resolve({ firstAudioMs: firstAudioAt, latencySummary });
      }
      if (msg.type === 'error') {
        console.error('[Test] Error:', msg.message);
        ws.close();
        reject(new Error(msg.message));
      }
    });

    ws.on('close', () => {
      console.log('[Test] WebSocket closed');
      resolve({ firstAudioMs: firstAudioAt, latencySummary });
    });

    ws.on('error', (err) => {
      console.error('[Test] WebSocket error:', err);
      reject(err);
    });

    // Safety timeout - increased to wait for TTS
    setTimeout(() => {
      try { ws.close(); } catch { /* noop */ }
      console.log('[Test] Timeout after 30s');
      resolve({ firstAudioMs: firstAudioAt, latencySummary });
    }, 30000);
  });
}

runTest()
  .then((result) => {
    console.log('[Test] Result:', result);
    process.exit(0);
  })
  .catch((err) => {
    console.error('[Test] Failed:', err);
    process.exit(1);
  });
