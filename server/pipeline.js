import { SarvamSTTStream } from './sarvamStt.js';
import { SarvamTTSStream } from './sarvamTts.js';
import { GeminiStreamingLLM } from './llmGemini.js';
import { RagRetriever } from './rag/qdrantClient.js';
import { TurnTimer } from './utils/latencyLogger.js';

const BASE_SYSTEM_PROMPT =
  'You are a concise voice assistant. For greetings and small talk, reply naturally and briefly. ' +
  'For any question seeking facts or information, use ONLY the provided context and never outside ' +
  'knowledge; if such a question is not answered by the context, say you do not have that information. ' +
  'Reply in 1-2 short spoken sentences. Never say you are an AI.';

const PARTIAL_DEBOUNCE_MS = 30; // Faster response - fire immediately after speech pauses
const MIN_TRIGGER_CHARS = 3;
const UTTERANCE_GAP_MS = 8000; // silence long enough that new speech is a new question

// The Vertex (LLM + embeddings) and Qdrant clients are process-wide singletons
// (see llmGemini.js / embedder.js / qdrantClient.js) so their TLS session + ADC
// token are shared across every WebSocket connection instead of re-handshaked
// per session.
//   - warmClients(): a light, non-blocking warm on each new connection so the
//     sockets are hot by the time the user finishes their first sentence (a
//     keep-alive socket goes cold after a few seconds idle even though the
//     token stays cached). One ping each, not the old per-session burst.
//   - a single process-wide slow interval so a long-idle instance doesn't go
//     fully cold between callers.
let _keepaliveStarted = false;
function warmClients(llm, rag) {
  llm.warmup().catch(() => {});
  rag.warmup().catch(() => {});
  if (!_keepaliveStarted) {
    _keepaliveStarted = true;
    setInterval(() => { llm.warmup().catch(() => {}); rag.warmup().catch(() => {}); }, 90_000).unref();
  }
}

/** Merge a fresh STT segment `b` onto the accumulated utterance `a` without duplicating. */
function mergeSegment(a, b) {
  a = (a || '').trim();
  b = (b || '').trim();
  if (!a) return b;
  if (!b) return a;
  const al = a.toLowerCase();
  const bl = b.toLowerCase();
  if (bl.startsWith(al)) return b;  // b is the live extension of a
  if (al.endsWith(bl)) return a;    // b is already at the tail of a
  return `${a} ${b}`;
}

/** True when two STT fragments look like the same segment still growing / being corrected. */
function sameSegment(a, b) {
  a = (a || '').toLowerCase().trim();
  b = (b || '').toLowerCase().trim();
  if (!a || !b) return true;
  return a.startsWith(b) || b.startsWith(a) || a.includes(b) || b.includes(a);
}

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

    // --- Utterance accumulation: STT resets `transcript` per speech segment,
    // so we stitch segments together to keep the user's full question. ---
    this._uttPrefix = '';     // finished segments of the current utterance
    this._uttLiveSeg = '';    // the segment STT is still growing
    this._uttAt = 0;          // timestamp of the last partial
    this._uttConsumed = false; // set once the current utterance has been answered

    this.cache = config.cache; // shared RedisCache instance or null

    this.rag = new RagRetriever({
      qdrantUrl: config.qdrantUrl,
      qdrantApiKey: config.qdrantApiKey,
      collection: config.qdrantCollection,
      embeddingModel: config.embeddingModel,
      project: config.gcpProject,
      location: config.gcpLocation,
      topK: config.ragTopK,
      contextTokenLimit: config.ragContextTokenLimit,
      timeoutMs: config.ragTimeoutMs,
      cache: this.cache,
    });

    // LLM: Gemini on Vertex AI. Generation is served via `global` for this
    // project (regional asia-south1 endpoint 404s); RAG/embeddings stay regional.
    this.llm = new GeminiStreamingLLM({
      model: config.geminiModel,
      project: config.gcpProject,
      location: config.geminiLocation || config.gcpLocation,
    });

    // Warm the shared Vertex/Qdrant sockets so this connection's first turn
    // isn't cold; also arms the process-wide idle keepalive (once).
    warmClients(this.llm, this.rag);

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
      // replying. Kill the in-flight turn immediately (LLM + TTS) so
      // playback stops, and tell the client which turn to discard.
      const killedId = this.activeTurn.id;
      const wasSpeaking = this.activeTurn.ttsStarted;
      this._abortActiveTurn();
      this.send({ type: 'barge_in', turnId: killedId });
      // Talking over the assistant's spoken answer starts a brand-new question.
      if (wasSpeaking) this._resetUtterance();
    }
  }

  _resetUtterance() {
    this._uttPrefix = '';
    this._uttLiveSeg = '';
    this._uttConsumed = false;
  }

  _onPartial(text, isFinal) {
    const seg = (text || '').trim();
    const now = Date.now();

    // Start a fresh utterance once the previous one has been answered, or after
    // a long silence (the STT socket stays open across turns).
    if (this._uttConsumed || (this._uttAt && now - this._uttAt > UTTERANCE_GAP_MS)) {
      this._resetUtterance();
    }
    this._uttAt = now;

    if (seg) {
      // STT collapses `transcript` back to just the newest speech segment after
      // a pause -- fold the finished segment into the prefix so a pause in the
      // middle of a sentence doesn't drop the first half of the question.
      if (this._uttLiveSeg && !sameSegment(this._uttLiveSeg, seg)) {
        this._uttPrefix = mergeSegment(this._uttPrefix, this._uttLiveSeg);
      }
      this._uttLiveSeg = seg;
    }
    const full = mergeSegment(this._uttPrefix, this._uttLiveSeg).trim();

    this.send({ type: 'transcript', text: full, isFinal });
    this.lastPartialText = full;
    console.log('[Pipeline] Partial text:', full, 'isFinal:', isFinal);
    if (!full || full.length < MIN_TRIGGER_CHARS) return;

    clearTimeout(this.debounceTimer);
    // Debounce: wait a short beat after the last partial before committing
    // to a turn, so we don't fire on every single word -- but we still
    // never wait for STT's own "final" flag, per the low-latency spec.
    this.debounceTimer = setTimeout(() => {
      this._startTurn(this.lastPartialText.trim());
    }, PARTIAL_DEBOUNCE_MS);

    if (isFinal) {
      clearTimeout(this.debounceTimer);
      this._startTurn(full);
    }
  }

  _abortActiveTurn() {
    if (!this.activeTurn) return;
    this.activeTurn.aborted = true;
    try { this.activeTurn.abort?.abort(); } catch { /* noop */ }   // stops the Gemini stream
    try { this.activeTurn.ttsStream?.close(); } catch { /* noop */ } // stops TTS synthesis
    this.activeTurn = null;
  }

  async _startTurn(userText) {
    console.log('[Pipeline] Starting turn with text:', userText);
    // A new utterance always cancels whatever is in flight and starts a fresh
    // pipeline. Tell the client to drop the old turn's audio.
    if (this.activeTurn) {
      const killedId = this.activeTurn.id;
      this._abortActiveTurn();
      this.send({ type: 'barge_in', turnId: killedId });
    }
    const turnId = ++this.turnCounter;
    const timer = new TurnTimer(turnId);
    timer.mark('partial_transcript');
    const turn = { id: turnId, aborted: false, abort: new AbortController(), ttsStream: null, ttsStarted: false };
    this.activeTurn = turn;

    // --- Parallel: cache lookup + RAG retrieval (both fire now) ---
    const cacheLookup = this.cache ? this.cache.get(userText).catch(() => null) : Promise.resolve(null);
    const ragLookup = this.rag.retrieve(userText).catch(() => ({ context: '', hits: [], timedOut: true }));

    // --- Cache hit: skip LLM *and* the RAG wait, go straight to TTS ---
    const cached = await cacheLookup;
    if (turn.aborted) return;
    if (cached && cached.reply) {
      timer.mark('rag_retrieved');
      timer.marks.cache_hit = true;
      const ttsStream = this._openTts(turn, timer);
      this.send({ type: 'reply_text', text: cached.reply, cached: true, turnId: turn.id });
      ttsStream.pushText(cached.reply);
      ttsStream.flush();
      this._uttConsumed = true; // this question is answered; next speech is a new utterance
      return;
    }

    // --- Cache miss: now wait for RAG (already in flight) ---
    const ragResult = await ragLookup;
    if (turn.aborted) return;
    timer.mark('rag_retrieved');
    if (ragResult.timedOut) this.send({ type: 'info', message: 'RAG retrieval timed out, answering without context' });

    // --- LLM streaming with clause-boundary TTS for lower latency ---
    let sentenceBuffer = '';
    let clauseBuffer = '';
    let firstFlushDone = false;
    let fullReply = '';
    const CLAUSE_DELIMITERS = /[.!?;,\n]/;

    // Open the TTS socket now, before the LLM call, so its WebSocket connect +
    // config handshake (~200-300ms) overlaps the LLM's first-token latency
    // instead of adding to the critical path after it. The first clause is
    // pushed within ~1s (well inside Sarvam's idle-socket window), so opening
    // early carries no timeout risk. `onToken` keeps a lazy-open fallback.
    let ttsStream = this._openTts(turn, timer);

    try {
      const systemPrompt = this._buildSystemPrompt();

      fullReply = await this.llm.streamReply({
        systemPrompt,
        shortTermTurns: this.shortTermTurns,
        ragContext: ragResult.context,
        userText,
        signal: turn.abort.signal,
        onToken: (delta) => {
          if (turn.aborted) return;
          if (timer.marks.llm_first_token == null) timer.mark('llm_first_token');
          if (!ttsStream) ttsStream = this._openTts(turn, timer);
          sentenceBuffer += delta;
          clauseBuffer += delta;

          // Send partial text to client for display
          this.send({ type: 'reply_text', text: sentenceBuffer, turnId: turn.id });

          // Get the first words into TTS ASAP (don't wait for the first clause
          // delimiter, which can be 40+ chars in), then flush on clause bounds.
          const atClause = CLAUSE_DELIMITERS.test(delta);
          const earlyFlush = !firstFlushDone && clauseBuffer.trim().length >= 8 && /\s/.test(clauseBuffer);
          if ((atClause || earlyFlush) && clauseBuffer.trim()) {
            try {
              ttsStream.pushText(clauseBuffer.trim() + ' ');
            } catch (e) {
              console.error('[Pipeline] TTS pushText error:', e.message);
            }
            clauseBuffer = '';
            firstFlushDone = true;
          }
        },
      });
      if (turn.aborted) return;
      this._uttConsumed = true; // this question is answered; next speech is a new utterance

      // Flush remaining buffer
      if (clauseBuffer.trim() && ttsStream) ttsStream.pushText(clauseBuffer.trim());
      if (ttsStream) ttsStream.flush();
      // LLM produced nothing (rare) -> nothing to speak, end the turn cleanly.
      if (!ttsStream) this.send({ type: 'turn_done', turnId: turn.id });

      this._extractUserInfo(userText, fullReply);
    } catch (err) {
      try { turn.ttsStream?.close(); } catch { /* noop */ } // don't leak the pre-opened socket
      if (!turn.aborted) this.send({ type: 'error', stage: 'llm', message: err.message });
      return;
    }

    this._pushShortTermTurn(userText, fullReply);
    // Only cache the final response when RAG actually delivered context -- a
    // turn that answered without context (e.g. RAG timed out) must not poison
    // the cache with a degraded "no information" reply.
    if (this.cache && !ragResult.timedOut) {
      this.cache.set(userText, { reply: fullReply }).catch(() => {});
    }
  }

  _openTts(turn, timer) {
    const s = new SarvamTTSStream({
      apiKey: this.config.sarvamApiKey,
      model: this.config.ttsModel,
      speaker: this.config.ttsSpeaker,
      targetLanguageCode: this.config.ttsLanguageCode,
      sampleRate: this.config.ttsSampleRate,
    }).connect();
    turn.ttsStream = s;
    this._wireTtsToClient(turn, timer);
    return s;
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
      this.send({ type: 'audio_chunk', audio: base64Audio, contentType, turnId: turn.id });
    });
    stream.on('done', () => {
      if (!turn.aborted) {
        timer.mark('turn_end');
        this.send({ type: 'turn_done', turnId: turn.id });
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
