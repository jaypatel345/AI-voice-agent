import { GoogleGenAI } from '@google/genai';

/**
 * Google `text-embedding-004` embeddings via Vertex AI (Mumbai / asia-south1),
 * per the required tech stack. Used at ingest time (server/rag/ingest.js) and
 * at query time (server/rag/qdrantClient.js) so index and queries share the
 * same 768-dim vector space.
 *
 * One persistent GoogleGenAI client is reused across calls (connection reuse
 * is a latency rule).
 */
class VertexEmbedder {
  constructor({ model, project, location } = {}) {
    this.model = model || 'text-embedding-004';
    this.client = new GoogleGenAI({
      vertexai: true,
      project: project || process.env.GOOGLE_CLOUD_PROJECT,
      location: location || process.env.GOOGLE_CLOUD_LOCATION || 'asia-south1',
    });
  }

  /**
   * @param {string} text
   * @param {string} [taskType] - RETRIEVAL_DOCUMENT at ingest, RETRIEVAL_QUERY at query time
   * @returns {Promise<number[]>} embedding vector
   */
  async embed(text, taskType) {
    const res = await this.client.models.embedContent({
      model: this.model,
      contents: text,
      config: taskType ? { taskType } : undefined,
    });
    const values = res.embeddings?.[0]?.values;
    if (!values || !values.length) {
      throw new Error('empty embedding returned from Vertex AI');
    }
    return values;
  }

  /** Pay the TLS + ADC-token handshake before the first real retrieval. */
  async warmup() {
    try { await this.embed('ping', 'RETRIEVAL_QUERY'); } catch { /* best-effort */ }
  }
}

export { VertexEmbedder };
