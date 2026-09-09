import { GoogleGenAI } from '@google/genai';

/**
 * Google `text-embedding-004` embeddings via Vertex AI (Mumbai / asia-south1),
 * per the required tech stack. Used at ingest time (server/rag/ingest.js) and
 * at query time (server/rag/qdrantClient.js) so index and queries share the
 * same 768-dim vector space.
 */

// One GoogleGenAI client per (project, location), shared across every
// VertexEmbedder / VoiceSession. A per-connection client pays a fresh
// TLS + ADC-token handshake (~400-500ms) on its first call; reusing one
// process-wide keeps that warm across WebSocket connects, reconnects and tabs.
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

// In-process memo in front of the Redis embedding cache. An embedding for a
// fixed string never changes (model + dims are fixed), so this is safe with
// no TTL; a repeated query then skips both the Upstash round-trip and the
// embedContent call. Bounded FIFO so memory can't grow unbounded.
const _embedMemo = new Map();
const EMBED_MEMO_MAX = 500;

class VertexEmbedder {
  constructor({ model, project, location } = {}) {
    this.model = model || 'text-embedding-004';
    this.location = location || process.env.GOOGLE_CLOUD_LOCATION || 'asia-south1';
    this.project = project || process.env.GOOGLE_CLOUD_PROJECT;
    this.client = sharedGenAI(this.project, this.location);
  }

  /**
   * @param {string} text
   * @param {string} [taskType] - RETRIEVAL_DOCUMENT at ingest, RETRIEVAL_QUERY at query time
   * @returns {Promise<number[]>} embedding vector
   */
  async embed(text, taskType) {
    const memoKey = taskType === 'RETRIEVAL_QUERY' ? text.trim().toLowerCase() : null;
    if (memoKey && _embedMemo.has(memoKey)) return _embedMemo.get(memoKey);

    const res = await this.client.models.embedContent({
      model: this.model,
      contents: text,
      config: taskType ? { taskType } : undefined,
    });
    const values = res.embeddings?.[0]?.values;
    if (!values || !values.length) {
      throw new Error('empty embedding returned from Vertex AI');
    }
    if (memoKey) {
      if (_embedMemo.size >= EMBED_MEMO_MAX) _embedMemo.delete(_embedMemo.keys().next().value);
      _embedMemo.set(memoKey, values);
    }
    return values;
  }

  /** Non-blocking lookup of the in-process query-embedding memo. */
  peekMemo(text) {
    return _embedMemo.get(text.trim().toLowerCase());
  }

  /** Pay the TLS + ADC-token handshake before the first real retrieval.
   *  Bypasses the memo so it stays useful as an idle keepalive. */
  async warmup() {
    try {
      await this.client.models.embedContent({
        model: this.model,
        contents: 'ping',
        config: { taskType: 'RETRIEVAL_QUERY' },
      });
    } catch { /* best-effort */ }
  }
}

export { VertexEmbedder };
