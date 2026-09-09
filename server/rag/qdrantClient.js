import { QdrantClient } from '@qdrant/js-client-rest';
import { VertexEmbedder } from './embedder.js';

/**
 * RAG retrieval, bounded to RAG_RETRIEVAL_TIMEOUT_MS (default 150ms) per
 * the spec. Document embeddings are precomputed at ingest time
 * (server/rag/ingest.js) with Google `text-embedding-004`; at query time we
 * embed only the short user utterance (cached in Redis) then do a single
 * Qdrant top_k search.
 *
 * If retrieval doesn't land inside the budget, we abort and let the LLM
 * answer with no context rather than block the pipeline -- a slow RAG hit
 * must never blow the sub-2s latency budget.
 */
class RagRetriever {
  constructor({ qdrantUrl, qdrantApiKey, collection, embeddingModel, project, location, topK, contextTokenLimit, timeoutMs, cache }) {
    this.client = new QdrantClient({ url: qdrantUrl, apiKey: qdrantApiKey || undefined });
    this.collection = collection || 'voice_kb';
    this.topK = topK || 2;
    this.contextTokenLimit = contextTokenLimit || 1000;
    this.timeoutMs = timeoutMs || 150;
    this.cache = cache || null;
    this.embedder = new VertexEmbedder({ model: embeddingModel, project, location });
  }

  /** Warm the embedding + Qdrant connections so the first turn isn't slow. */
  async warmup() {
    await Promise.all([
      this.embedder.warmup(),
      this.client.getCollections().catch(() => {}),
    ]);
  }

  async _embedQuery(text) {
    if (this.cache) {
      const cached = await this.cache.getEmbedding(text).catch(() => null);
      if (cached) return cached;
    }
    const vector = await this.embedder.embed(text, 'RETRIEVAL_QUERY');
    if (this.cache) this.cache.setEmbedding(text, vector).catch(() => {});
    return vector;
  }

  /**
   * @returns {Promise<{context: string, hits: Array, timedOut: boolean}>}
   */
  async retrieve(queryText) {
    const withTimeout = (promise, ms) =>
      Promise.race([
        promise,
        new Promise((resolve) => setTimeout(() => resolve({ __timeout: true }), ms)),
      ]);

    const result = await withTimeout(this._doRetrieve(queryText), this.timeoutMs);
    if (result && result.__timeout) {
      return { context: '', hits: [], timedOut: true };
    }
    return result;
  }

  async _doRetrieve(queryText) {
    const vector = await this._embedQuery(queryText);
    const res = await this.client.query(this.collection, {
      query: vector,
      limit: this.topK,
      with_payload: true,
    });
    const hits = res.points || [];

    // Trim to the context token budget (rough estimate: ~4 chars/token)
    const maxChars = this.contextTokenLimit * 4;
    let used = 0;
    const pieces = [];
    for (const hit of hits) {
      const chunk = hit.payload?.text || '';
      if (used + chunk.length > maxChars) break;
      pieces.push(chunk);
      used += chunk.length;
    }
    return { context: pieces.join('\n---\n'), hits, timedOut: false };
  }
}

export { RagRetriever };
