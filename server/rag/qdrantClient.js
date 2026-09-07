import { QdrantClient } from '@qdrant/js-client-rest';
import { pipeline, env } from '@xenova/transformers';

// Disable local model downloads, use cache
env.allowLocalModels = false;
env.useBrowserCache = false;

// Global embedder instance (lazy loaded)
let embedder = null;
let embeddingModelName = null;

/**
 * RAG retrieval, bounded to RAG_RETRIEVAL_TIMEOUT_MS (default 150ms) per
 * the spec. Embeddings are precomputed at ingest time (server/rag/ingest.js);
 * at query time we only embed the short user utterance, which is fast, then
 * do a single Qdrant top_k search.
 *
 * If retrieval doesn't land inside the budget, we abort and let the LLM
 * answer with no context rather than block the pipeline -- a slow RAG hit
 * must never blow the sub-2s latency budget.
 */
class RagRetriever {
  constructor({ qdrantUrl, qdrantApiKey, collection, embeddingModel, topK, contextTokenLimit, timeoutMs }) {
    this.client = new QdrantClient({ url: qdrantUrl, apiKey: qdrantApiKey || undefined });
    this.collection = collection || 'voice_kb';
    this.topK = topK || 2;
    this.contextTokenLimit = contextTokenLimit || 1000;
    this.timeoutMs = timeoutMs || 150;
    this.embeddingModel = embeddingModel || 'Xenova/bge-small-en-v1.5';
    // Lazy load embedder on first use
  }

  async _getEmbedder() {
    if (!embedder || embeddingModelName !== this.embeddingModel) {
      console.log(`Loading local embedding model: ${this.embeddingModel}...`);
      embedder = await pipeline('feature-extraction', this.embeddingModel);
      embeddingModelName = this.embeddingModel;
      console.log('Embedding model loaded.');
    }
    return embedder;
  }

  async _embedQuery(text) {
    const model = await this._getEmbedder();
    const embedding = await model(text, { pooling: 'mean', normalize: true });
    return Array.from(embedding.data);
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
    const search = await this.client.search(this.collection, {
      vector,
      limit: this.topK,
      with_payload: true,
    });

    // Trim to the context token budget (rough estimate: ~4 chars/token)
    const maxChars = this.contextTokenLimit * 4;
    let used = 0;
    const pieces = [];
    for (const hit of search) {
      const chunk = hit.payload?.text || '';
      if (used + chunk.length > maxChars) break;
      pieces.push(chunk);
      used += chunk.length;
    }
    return { context: pieces.join('\n---\n'), hits: search, timedOut: false };
  }
}

export { RagRetriever };
