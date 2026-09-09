import 'dotenv/config';
import path from 'path';
import { fileURLToPath } from 'url';
import { setGlobalDispatcher, Agent } from 'undici';
import express from 'express';
import { WebSocketServer } from 'ws';
import { VoiceSession } from './pipeline.js';
import { RedisCache } from './cache/redisCache.js';

// Node's global fetch (used by @google/genai and @qdrant/js-client-rest) drops
// an idle keep-alive socket after ~4s by default, so back-to-back turns kept
// re-paying the TLS + HTTP/2 handshake to Vertex and Qdrant -- ~100-200ms per
// leg on the RAG + LLM critical path. Hold idle sockets open for 60s instead.
setGlobalDispatcher(new Agent({
  keepAliveTimeout: 60_000,
  keepAliveMaxTimeout: 600_000,
  connections: 128,
}));

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const PORT = process.env.PORT || 8080;

const app = express();
app.use(express.static(path.join(__dirname, '..', 'public')));
app.get('/health', (_req, res) => res.json({ ok: true }));

const server = app.listen(PORT, () => {
  console.log(`Voice assistant server listening on http://localhost:${PORT}`);
});

// --- Shared Redis cache instance across all sessions ---
const cache = new RedisCache({
  url: process.env.REDIS_URL,
  ttlSeconds: Number(process.env.CACHE_TTL_SECONDS || 3600),
});
cache.connect().catch((err) => console.error('[redis] failed to connect, running without cache:', err.message));

function buildConfig() {
  return {
    sarvamApiKey: process.env.SARVAM_API_KEY,
    sttModel: process.env.STT_MODEL,
    sttLanguageCode: process.env.STT_LANGUAGE_CODE,
    ttsModel: process.env.TTS_MODEL,
    ttsSpeaker: process.env.TTS_SPEAKER,
    ttsSampleRate: Number(process.env.TTS_SAMPLE_RATE || 24000),
    ttsLanguageCode: process.env.TTS_LANGUAGE_CODE || process.env.STT_LANGUAGE_CODE,
    gcpProject: process.env.GOOGLE_CLOUD_PROJECT,
    gcpLocation: process.env.GOOGLE_CLOUD_LOCATION || 'asia-south1',
    // Vertex serves Gemini generation for this project only via `global`;
    // regional endpoints (asia-south1) 404. Embeddings/RAG stay in asia-south1.
    geminiLocation: process.env.GEMINI_LOCATION || 'global',
    geminiModel: process.env.GEMINI_MODEL || 'gemini-3.1-flash-lite',
    qdrantUrl: process.env.QDRANT_URL,
    qdrantApiKey: process.env.QDRANT_API_KEY,
    qdrantCollection: process.env.QDRANT_COLLECTION,
    embeddingModel: process.env.EMBEDDING_MODEL,
    ragTopK: Number(process.env.RAG_TOP_K || 2),
    ragContextTokenLimit: Number(process.env.RAG_CONTEXT_TOKEN_LIMIT || 1000),
    ragTimeoutMs: Number(process.env.RAG_RETRIEVAL_TIMEOUT_MS || 150),
    shortTermTurns: Number(process.env.SHORT_TERM_TURNS || 3),
    cache,
  };
}

// --- WebSocket: the whole speech-to-speech pipeline lives on one socket ---
const wss = new WebSocketServer({ server, path: '/ws/voice' });

wss.on('connection', (ws) => {
  console.log('[ws] client connected');
  const session = new VoiceSession({
    send: (obj) => {
      if (ws.readyState === ws.OPEN) ws.send(JSON.stringify(obj));
    },
    config: buildConfig(),
  });

  ws.on('message', (raw) => {
    let msg;
    try { msg = JSON.parse(raw.toString()); } catch { return; }
    if (msg.type === 'audio_chunk' && msg.audio) {
      session.pushAudio(msg.audio);
    } else if (msg.type === 'flush') {
      session.stt.flush();
    } else if (msg.type === 'text_input' && msg.text) {
      // Test-only entry point: bypasses STT so benchmark/latency_test.js can
      // drive the RAG→LLM→TTS pipeline deterministically. Not exposed in the UI.
      session._startTurn(msg.text);
    }
  });

  ws.on('close', () => {
    console.log('[ws] client disconnected');
    session.close();
  });
  ws.on('error', (err) => console.error('[ws] error', err.message));
});
