import 'dotenv/config';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { QdrantClient } from '@qdrant/js-client-rest';
import { pipeline, env } from '@xenova/transformers';

// Disable local model downloads, use cache
env.allowLocalModels = false;
env.useBrowserCache = false;

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const CHUNK_MIN_TOKENS = 300;
const CHUNK_MAX_TOKENS = 500;
const CHARS_PER_TOKEN = 4; // rough heuristic, good enough for chunk sizing

/** Splits long text into ~300-500 token chunks on sentence boundaries. */
function chunkText(text) {
  const sentences = text.match(/[^.!?]+[.!?]+/g) || [text];
  const chunks = [];
  let current = '';
  for (const sentence of sentences) {
    const candidate = current ? `${current} ${sentence.trim()}` : sentence.trim();
    if (candidate.length / CHARS_PER_TOKEN > CHUNK_MAX_TOKENS && current) {
      chunks.push(current.trim());
      current = sentence.trim();
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

  // Load local embedding model
  const embeddingModelName = process.env.EMBEDDING_MODEL || 'Xenova/bge-small-en-v1.5';
  console.log(`Loading local embedding model: ${embeddingModelName}...`);
  const embedder = await pipeline('feature-extraction', embeddingModelName);
  console.log('Embedding model loaded.');

  const collection = process.env.QDRANT_COLLECTION || 'voice_kb';
  // BAAI/bge-small-en-v1.5 produces 384-dimensional embeddings
  const dim = Number(process.env.EMBEDDING_DIM || 384);

  console.log(`Ensuring collection "${collection}" (dim=${dim})...`);
  const collections = await qdrant.getCollections();
  if (!collections.collections.some((c) => c.name === collection)) {
    await qdrant.createCollection(collection, {
      vectors: { size: dim, distance: 'Cosine' },
    });
    console.log('Created collection.');
  } else {
    console.log('Collection already exists.');
  }

  const docsPath = path.join(__dirname, 'knowledge_base', 'sample_docs.json');
  const docs = JSON.parse(fs.readFileSync(docsPath, 'utf-8'));

  const points = [];
  let pointId = 1;
  for (const doc of docs) {
    const chunks = chunkText(doc.text);
    for (const chunk of chunks) {
      // Generate embedding using local model
      const embedding = await embedder(chunk, { pooling: 'mean', normalize: true });
      const vector = Array.from(embedding.data); // Convert to regular array
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
