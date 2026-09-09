import 'dotenv/config';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { QdrantClient } from '@qdrant/js-client-rest';
import { VertexEmbedder } from './embedder.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const CHUNK_MIN_TOKENS = 300;
const CHUNK_MAX_TOKENS = 400; // spec: 300-400 token chunks
const CHUNK_OVERLAP_TOKENS = 50; // spec: 50 token overlap
const CHARS_PER_TOKEN = 4; // rough heuristic, good enough for chunk sizing

/**
 * Splits long text into ~300-400 token chunks on sentence boundaries with a
 * ~50 token overlap between consecutive chunks (per the RAG spec).
 */
function chunkText(text) {
  const sentences = text.match(/[^.!?]+[.!?]+/g) || [text];
  const maxChars = CHUNK_MAX_TOKENS * CHARS_PER_TOKEN;
  const overlapChars = CHUNK_OVERLAP_TOKENS * CHARS_PER_TOKEN;

  const chunks = [];
  let current = '';
  for (const sentence of sentences) {
    const s = sentence.trim();
    const candidate = current ? `${current} ${s}` : s;
    if (candidate.length > maxChars && current) {
      chunks.push(current.trim());
      // start the next chunk with the tail of the previous one (overlap)
      const tail = current.slice(-overlapChars);
      current = `${tail} ${s}`.trim();
    } else {
      current = candidate;
    }
  }
  if (current) chunks.push(current.trim());

  // Merge any tiny trailing chunk below the min into the previous one
  if (chunks.length > 1 && chunks[chunks.length - 1].length / CHARS_PER_TOKEN < CHUNK_MIN_TOKENS) {
    const last = chunks.pop();
    chunks[chunks.length - 1] += ' ' + last;
  }
  return chunks;
}

async function main() {
  const qdrant = new QdrantClient({
    url: process.env.QDRANT_URL || 'http://localhost:6333',
    apiKey: process.env.QDRANT_API_KEY || undefined,
  });

  const embedder = new VertexEmbedder({
    model: process.env.EMBEDDING_MODEL || 'text-embedding-004',
    project: process.env.GOOGLE_CLOUD_PROJECT,
    location: process.env.GOOGLE_CLOUD_LOCATION || 'asia-south1',
  });

  const collection = process.env.QDRANT_COLLECTION || 'voice_kb';
  // text-embedding-004 produces 768-dimensional embeddings
  const dim = Number(process.env.EMBEDDING_DIM || 768);

  console.log(`Ensuring collection "${collection}" (dim=${dim})...`);
  const collections = await qdrant.getCollections();
  const existing = collections.collections.find((c) => c.name === collection);
  if (existing) {
    const info = await qdrant.getCollection(collection);
    const currentDim = info.config?.params?.vectors?.size;
    if (currentDim && currentDim !== dim) {
      console.log(`Collection dim ${currentDim} != ${dim}; recreating.`);
      await qdrant.deleteCollection(collection);
      await qdrant.createCollection(collection, { vectors: { size: dim, distance: 'Cosine' } });
      console.log('Recreated collection.');
    } else {
      console.log('Collection already exists.');
    }
  } else {
    await qdrant.createCollection(collection, { vectors: { size: dim, distance: 'Cosine' } });
    console.log('Created collection.');
  }

  const docsPath = path.join(__dirname, 'knowledge_base', 'sample_docs.json');
  const docs = JSON.parse(fs.readFileSync(docsPath, 'utf-8'));

  const points = [];
  let pointId = 1;
  for (const doc of docs) {
    const chunks = chunkText(doc.text);
    for (const chunk of chunks) {
      const vector = await embedder.embed(chunk, 'RETRIEVAL_DOCUMENT');
      points.push({
        id: pointId++,
        vector,
        payload: { doc_id: doc.id, title: doc.title, text: chunk },
      });
      console.log(`Embedded chunk from "${doc.title}" (${chunk.length} chars)`);
    }
  }

  await qdrant.upsert(collection, { wait: true, points });
  console.log(`Upserted ${points.length} chunks into "${collection}". RAG index ready.`);
}

main().catch((err) => {
  console.error('Ingest failed:', err);
  process.exit(1);
});
