import { GoogleGenAI } from '@google/genai';

// One GoogleGenAI client per (project, location), shared across every
// GeminiStreamingLLM / VoiceSession so the TLS session + ADC token stay warm
// across WebSocket connections instead of paying a fresh cold handshake
// (~400-500ms of the first-token latency) on each new connection.
const _genaiClients = new Map();
function sharedGenAI(project, location) {
  const key = `${project}|${location}`;
  let client = _genaiClients.get(key);
  if (!client) {
    client = new GoogleGenAI({ vertexai: true, project, location });
    _genaiClients.set(key, client);
  }
  return client;
}

/**
 * Streaming Gemini calls on Vertex AI (asia-south1 / Mumbai).
 * Auth is Application Default Credentials only -- no API keys.
 */
class GeminiStreamingLLM {
  constructor({ model, project, location } = {}) {
    this.model = model || 'gemini-3.1-flash-lite';
    this.useVertexAI = true;
    // `global`: Vertex serves Gemini generation for this project only via the
    // global endpoint; the asia-south1 regional endpoint 404s. Embeddings/RAG
    // continue to run in asia-south1 (see server/rag/embedder.js).
    this.location = location || process.env.GEMINI_LOCATION || 'global';
    this.client = sharedGenAI(project || process.env.GOOGLE_CLOUD_PROJECT, this.location);
  }

  /**
   * Builds the minimal prompt required (system instructions kept tiny per
   * the latency directives) and streams the response.
   *
   * @param {string} systemPrompt - short, static instruction
   * @param {Array<{role:'user'|'model', text:string}>} shortTermTurns - last 2-3 turns only
   * @param {string} ragContext - <=1000 tokens of retrieved context
   * @param {string} userText - the (partial) transcript that triggered this turn
   * @param {(chunkText:string)=>void} onToken - called for every streamed text delta
   * @returns {Promise<string>} full response text
   */
  /** Fire tiny streaming requests so the TLS + ADC-token + HTTP/2 ramp is paid
   *  before the first real turn. Cold first-call TTFT is ~1.5s; it takes 2-3
   *  calls on the same client to settle to ~0.7s, so we do two. */
  async warmup() {
    for (let i = 0; i < 2; i++) {
      try {
        const stream = await this.client.models.generateContentStream({
          model: this.model,
          contents: 'ping',
          config: { maxOutputTokens: 1, temperature: 0 },
        });
        for await (const _ of stream) break; // consume just the first chunk
      } catch { /* best-effort */ }
    }
  }

  async streamReply({ systemPrompt, shortTermTurns = [], ragContext, userText, onToken, signal }) {
    const contents = [
      ...shortTermTurns.map((t) => ({ role: t.role, parts: [{ text: t.text }] })),
      {
        role: 'user',
        parts: [{
          text: ragContext
            ? `Context:\n${ragContext}\n\nUser: ${userText}`
            : userText,
        }],
      },
    ];
    const config = {
      systemInstruction: systemPrompt,
      maxOutputTokens: 80, // spec: minimal output -> shorter TTS queue -> lower perceived latency
      temperature: 0.2,
    };

    // Vertex `global` occasionally accepts the request then stalls before the
    // first token for many seconds. Guard only the pre-first-token window: if
    // nothing arrives in time, abort that attempt and retry once (same prompt,
    // same output). Once tokens are flowing we never restart -- that would
    // corrupt the streamed reply -- and a barge-in abort never retries.
    const FIRST_TOKEN_TIMEOUT_MS = 2200;
    const MAX_ATTEMPTS = 2;

    for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
      if (signal?.aborted) return '';
      const attemptCtrl = new AbortController();
      const relayAbort = () => attemptCtrl.abort();
      signal?.addEventListener('abort', relayAbort, { once: true });
      let firstToken = false;
      const watchdog = setTimeout(() => { if (!firstToken) attemptCtrl.abort(); }, FIRST_TOKEN_TIMEOUT_MS);
      let full = '';
      try {
        console.log('[LLM] Calling model:', this.model, 'attempt', attempt);
        const stream = await this.client.models.generateContentStream({
          model: this.model,
          contents,
          config: { ...config, abortSignal: attemptCtrl.signal },
        });
        for await (const chunk of stream) {
          if (signal?.aborted) { clearTimeout(watchdog); return ''; }
          const delta = chunk.text || '';
          if (delta) {
            if (!firstToken) { firstToken = true; clearTimeout(watchdog); }
            full += delta;
            onToken(delta);
          }
        }
        clearTimeout(watchdog);
        signal?.removeEventListener('abort', relayAbort);
        console.log('[LLM] Response complete, length:', full.length);
        return full;
      } catch (error) {
        clearTimeout(watchdog);
        signal?.removeEventListener('abort', relayAbort);
        if (signal?.aborted) return ''; // cancelled by barge-in
        if (!firstToken && attempt < MAX_ATTEMPTS) {
          console.warn(`[LLM] no first token in ${FIRST_TOKEN_TIMEOUT_MS}ms, retrying`);
          continue;
        }
        if (error?.name === 'AbortError') return '';
        console.error('[LLM] Error:', error.message);
        throw error;
      }
    }
    return '';
  }
}

export { GeminiStreamingLLM };
