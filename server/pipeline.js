import { SarvamSTTStream } from './sarvamStt.js';
import { SarvamTTSStream } from './sarvamTts.js';
import { GeminiStreamingLLM } from './llmGemini.js';
import { OpenAIStreamingLLM } from './llmOpenai.js';
import { RagRetriever } from './rag/qdrantClient.js';
import { TurnTimer } from './utils/latencyLogger.js';

const BASE_SYSTEM_PROMPT =
  'You are a fast, concise voice assistant. Answer in 1-3 short spoken sentences. ' +
  'Use the provided context if relevant; if not relevant, answer from general knowledge. Never say you are an AI.';

const PARTIAL_DEBOUNCE_MS = 30; // Faster response - fire immediately after speech pauses
const MIN_TRIGGER_CHARS = 3;

/**
 * One instance per browser WebSocket connection. Owns the STT socket for
 * the whole session and spins up a fresh TTS+LLM turn per user utterance.
 */
class VoiceSession {
  constructor({ send, config }) {
    this.send = send; // (obj) => void, sends JSON to the browser
    this.config = config;
    this.shortTermTurns = []; // last N {role, text}
    this.userMemory = {}; // Store user information like name
    this.turnCounter = 0;
    this.debounceTimer = null;
    this.lastPartialText = '';
    this.activeTurn = null; // { llmAbort, ttsStream, ttsStarted }

    this.rag = new RagRetriever({
      qdrantUrl: config.qdrantUrl,
      qdrantApiKey: config.qdrantApiKey,
      collection: config.qdrantCollection,
      embeddingModel: config.embeddingModel,
      topK: config.ragTopK,
      contextTokenLimit: config.ragContextTokenLimit,
      timeoutMs: config.ragTimeoutMs,
    });

    // Select LLM based on provider
    if (config.llmProvider === 'openai') {
      this.llm = new OpenAIStreamingLLM({
        model: config.openaiModel,
        apiKey: config.openaiApiKey,
      });
    } else {
      this.llm = new GeminiStreamingLLM({
        model: config.geminiModel,
        apiKey: config.googleApiKey,
        useVertexAI: config.useVertexAI,
        project: config.gcpProject,
        location: config.gcpLocation,
      });
    }
    this.cache = config.cache; // shared RedisCache instance or null

    this.stt = new SarvamSTTStream({
      apiKey: config.sarvamApiKey,
      model: config.sttModel,
      languageCode: config.sttLanguageCode,
      sampleRate: 16000,
    }).connect();

    this.stt.on('partial', (text, isFinal) => this._onPartial(text, isFinal));
    this.stt.on('vad', (signal) => this._onVad(signal));
    this.stt.on('error', (err) => this.send({ type: 'error', stage: 'stt', message: err.message }));
  }

  /** Browser -> here: base64 PCM16LE mic chunk */
  pushAudio(base64Pcm) {
    this.stt.sendAudioChunk(base64Pcm);
  }

  _onVad(signal) {
    this.send({ type: 'vad', signal });
    if (signal === 'START_SPEECH' && this.activeTurn) {
      // Barge-in: the user started talking again while we were still
      // replying. Kill the in-flight turn immediately so playback stops.
      this._abortActiveTurn();
      this.send({ type: 'barge_in' });
    }
  }

  _onPartial(text, isFinal) {
    this.send({ type: 'transcript', text, isFinal });
    this.lastPartialText = text;
    console.log('[Pipeline] Partial text:', text, 'isFinal:', isFinal);
    if (!text || text.trim().length < MIN_TRIGGER_CHARS) return;

    clearTimeout(this.debounceTimer);
    // Debounce: wait a short beat after the last partial before committing
    // to a turn, so we don't fire on every single word -- but we still
    // never wait for STT's own "final" flag, per the low-latency spec.
    this.debounceTimer = setTimeout(() => {
      this._startTurn(this.lastPartialText.trim());
    }, PARTIAL_DEBOUNCE_MS);

    if (isFinal) {
      clearTimeout(this.debounceTimer);
      this._startTurn(text.trim());
    }
  }

  _abortActiveTurn() {
    if (!this.activeTurn) return;
    this.activeTurn.aborted = true;
    try { this.activeTurn.ttsStream?.close(); } catch { /* noop */ }
    this.activeTurn = null;
  }

  async _startTurn(userText) {
    console.log('[Pipeline] Starting turn with text:', userText);
    if (this.activeTurn) this._abortActiveTurn();
    const turnId = ++this.turnCounter;
    const timer = new TurnTimer(turnId);
    timer.mark('partial_transcript');
    const turn = { aborted: false, ttsStream: null, ttsStarted: false };
    this.activeTurn = turn;

    // --- Parallel: cache lookup + RAG retrieval + (lightweight) intent ---
    const cacheLookup = this.cache ? this.cache.get(userText).catch(() => null) : Promise.resolve(null);
    const ragLookup = this.rag.retrieve(userText).catch(() => ({ context: '', hits: [], timedOut: true }));

    const [cached, ragResult] = await Promise.all([cacheLookup, ragLookup]);
    if (turn.aborted) return;
    timer.mark('rag_retrieved');
    if (ragResult.timedOut) this.send({ type: 'info', message: 'RAG retrieval timed out, answering without context' });

    // --- Cache hit: skip LLM, go straight to TTS ---
    if (cached && cached.reply) {
      timer.marks.cache_hit = true;
      this._speak(turn, cached.reply, timer);
      this.send({ type: 'reply_text', text: cached.reply, cached: true });
      return;
    }

    // --- LLM streaming with clause-boundary TTS for lower latency ---
    let sentenceBuffer = '';
    let fullReply = '';
    let ttsStream = null;
    
    try {
      // Build system prompt with user memory
      const systemPrompt = this._buildSystemPrompt();
      
      // Initialize TTS stream early for clause-by-clause synthesis
      ttsStream = new SarvamTTSStream({
        apiKey: this.config.sarvamApiKey,
        model: this.config.ttsModel,
        speaker: this.config.ttsSpeaker,
        targetLanguageCode: this.config.ttsLanguageCode,
        sampleRate: this.config.ttsSampleRate,
      }).connect();
      turn.ttsStream = ttsStream;
      if (ttsStream) {
        this._wireTtsToClient(turn, timer);
      }
      
      let clauseBuffer = '';
      const CLAUSE_DELIMITERS = /[.!?;,\n]/;
      
      fullReply = await this.llm.streamReply({
        systemPrompt,
        shortTermTurns: this.shortTermTurns,
        ragContext: ragResult.context,
        userText,
        onToken: (delta) => {
          if (turn.aborted) return;
          if (timer.marks.llm_first_token == null) timer.mark('llm_first_token');
          sentenceBuffer += delta;
          clauseBuffer += delta;
          
          // Send partial text to client for display
          this.send({ type: 'reply_text', text: sentenceBuffer });
          
          // Flush clause to TTS when delimiter found
          if (CLAUSE_DELIMITERS.test(delta)) {
            const clause = clauseBuffer.trim();
            if (clause && ttsStream) {
              try {
                ttsStream.pushText(clause + ' ');
              } catch (e) {
                console.error('[Pipeline] TTS pushText error:', e.message);
              }
              clauseBuffer = '';
            }
          }
        },
      });

      // Flush remaining buffer
      if (clauseBuffer.trim() && ttsStream) {
        ttsStream.pushText(clauseBuffer);
      }
      if (ttsStream) ttsStream.flush();

      // Extract user information from conversation
      this._extractUserInfo(userText, fullReply);
    } catch (err) {
      this.send({ type: 'error', stage: 'llm', message: err.message });
      // Don't close ttsStream here - it will clean up itself on error
      return;
    }

    this._pushShortTermTurn(userText, fullReply);
    if (this.cache) this.cache.set(userText, { reply: fullReply }).catch(() => {});
  }

  _speak(turn, text, timer) {
    turn.ttsStream = new SarvamTTSStream({
      apiKey: this.config.sarvamApiKey,
      model: this.config.ttsModel,
      speaker: this.config.ttsSpeaker,
      targetLanguageCode: this.config.ttsLanguageCode,
      sampleRate: this.config.ttsSampleRate,
    }).connect();
    this._wireTtsToClient(turn, timer);
    turn.ttsStream.pushText(text);
    turn.ttsStream.flush();
  }

  _wireTtsToClient(turn, timer) {
    if (!turn.ttsStream) return;
    
    const stream = turn.ttsStream;
    stream.on('audio', (base64Audio, contentType) => {
      if (turn.aborted) return;
      if (!turn.ttsStarted) {
        turn.ttsStarted = true;
        timer.mark('tts_first_chunk');
        timer.mark('first_audio_to_client'); // headline metric: fired the instant we forward the first chunk
        const summary = timer.log();
        this.send({ type: 'latency', latency: summary });
      }
      this.send({ type: 'audio_chunk', audio: base64Audio, contentType });
    });
    stream.on('done', () => {
      if (!turn.aborted) {
        timer.mark('turn_end');
        this.send({ type: 'turn_done' });
      }
      if (this.activeTurn === turn) this.activeTurn = null;
    });
    stream.on('error', (err) => {
      this.send({ type: 'error', stage: 'tts', message: err.message });
    });
  }

  _pushShortTermTurn(userText, replyText) {
    this.shortTermTurns.push({ role: 'user', text: userText }, { role: 'model', text: replyText });
    // Keep only the last N turns (2-3), never replay the full conversation
    const maxMessages = (this.config.shortTermTurns || 3) * 2;
    if (this.shortTermTurns.length > maxMessages) {
      this.shortTermTurns = this.shortTermTurns.slice(-maxMessages);
    }
  }

  _buildSystemPrompt() {
    let prompt = BASE_SYSTEM_PROMPT;
    if (this.userMemory.name) {
      prompt += ` The user's name is ${this.userMemory.name}. Address them by name when appropriate.`;
    }
    return prompt;
  }

  _extractUserInfo(userText, assistantReply) {
    // Simple pattern matching for name introduction
    const namePatterns = [
      /(?:my name is|i am|i'm|i'm called|call me)\s+(\w+)/i,
      /(?:i'm|i am)\s+(\w+)/i,
    ];
    
    for (const pattern of namePatterns) {
      const match = userText.match(pattern);
      if (match && match[1]) {
        this.userMemory.name = match[1].charAt(0).toUpperCase() + match[1].slice(1);
        console.log('[Pipeline] Extracted user name:', this.userMemory.name);
        break;
      }
    }
  }

  close() {
    clearTimeout(this.debounceTimer);
    this._abortActiveTurn();
    this.stt.close();
  }
}

export { VoiceSession };
