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
    try {
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

      console.log('[LLM] Calling model:', this.model, 'with useVertexAI:', this.useVertexAI);
      const stream = await this.client.models.generateContentStream({
        model: this.model,
        contents,
        config: {
          systemInstruction: systemPrompt,
          maxOutputTokens: 80, // spec: minimal output -> shorter TTS queue -> lower perceived latency
          temperature: 0.2,
          abortSignal: signal, // barge-in: stop consuming tokens the instant the turn is cancelled
        },
      });

      let full = '';
      for await (const chunk of stream) {
        if (signal?.aborted) break;
        const delta = chunk.text || '';
        if (delta) {
          full += delta;
          onToken(delta);
        }
      }
      console.log('[LLM] Response complete, length:', full.length);
      return full;
    } catch (error) {
      if (signal?.aborted || error?.name === 'AbortError') return ''; // cancelled by barge-in
      console.error('[LLM] Error:', error.message);
      throw error;
    }
  }
}

export { GeminiStreamingLLM };
