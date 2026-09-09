# Low-Latency AI Voice Assistant with RAG

A real-time, speech-to-speech voice assistant for the web: browser mic → streaming STT →
parallel RAG + prompt build → streaming LLM → streaming TTS → browser playback, engineered
to get audio back to the user in **well under 2 seconds**, targeting **≤1.5s**.

```text
Browser (mic)  --PCM16 chunks-->  Node/Express + WebSocket server
                                        │
                                Sarvam STT (streaming, partial transcripts)
                                        │  (debounced ~120ms after last partial —
                                        │   never waits for a full sentence)
                         ┌──────────────┴───────────────┐
                    Redis cache lookup            Qdrant RAG retrieval
                    (exact-match FAQ)              (top_k=2, ≤150ms budget,
                                                     precomputed embeddings)
                         └──────────────┬───────────────┘
                                        │
                        Gemini gemini-3.1-flash-lite (Vertex AI, asia-south1)
                              streaming LLM, short prompt (<1000 tokens)
                                        │  (flushed to TTS at clause boundaries)
                                 Sarvam TTS (streaming)
                                        │
                         audio chunks forwarded instantly
                                        │
                                 Browser playback
```

## Why this hits the latency budget

* **No sequential waiting anywhere on the hot path.** STT partials trigger RAG
  retrieval and cache lookup *in parallel* (`Promise.all` in `server/pipeline.js`).
* **RAG is timeout-bounded (150ms).** If Qdrant is slow, we drop context and answer
  from general knowledge rather than block the turn — see `RagRetriever.retrieve()`.
* **Cache-first.** Repeated/FAQ questions skip RAG + LLM entirely (Redis, keyed by
  normalized query hash) and jump straight to TTS.
* **Token-level streaming, twice.** Gemini streaming tokens are pushed toward the
  Sarvam TTS pipeline as they arrive and flushed at clause boundaries
  (`.`, `,`, `;`, newline) — audio synthesis starts before the LLM has finished
  the sentence.
* **Audio is forwarded the instant it's received** from Sarvam TTS — no server-side
  buffering, no waiting for the full clip.
* **Short prompts, short memory.** System prompt is one sentence; only the last
  2–3 turns are replayed, never the full conversation.
* **Barge-in.** `high_vad_sensitivity` + `vad_signals` on the STT socket surface a
  `START_SPEECH` event the instant the user starts talking again; the server
  aborts the in-flight LLM/TTS turn immediately (`_abortActiveTurn`).
* **n8n stays off the hot path.** See `n8n/voice-assistant-workflow.json` — it
  handles async admin work (KB re-indexing, Slack alerts on latency breaches),
  not the per-turn pipeline, because every extra webhook hop costs 50–150ms+
  that the budget can't absorb.

## Stack

| Component | Choice |
|---|---|
| STT | Sarvam AI `saaras:v4`, WebSocket streaming, raw PCM16, `high_vad_sensitivity` |
| LLM | Google Vertex AI `gemini-3.1-flash-lite`, streaming (`GEMINI_LOCATION`, default `global` — see note) |
| Embeddings | Google `text-embedding-004` (Vertex AI, `asia-south1`), 768-dim |
| TTS | Sarvam AI `bulbul:v3`, WebSocket streaming, `linear16` output |
| Vector DB | Qdrant, precomputed embeddings, `top_k=2` |
| Cache | Redis, SHA1-normalized query keys |
| Server | Node.js + Express + `ws` |
| Frontend | Vanilla JS, Web Audio API (mic capture + gapless scheduled playback) |

> **Region note — known model/region conflict.** The task brief mandates
> `asia-south1` (Mumbai) as non-negotiable *and* the model id
> `gemini-3.1-flash-lite`. These two cannot both be satisfied on this GCP
> project: `gemini-3.1-flash-lite` is not served from the `asia-south1`
> regional Vertex endpoint. Verified 2026-09-09 with a direct
> `generateContentStream` call:
>
> ```
> [asia-south1] HTTP 404 — Publisher model
>   `projects/<project>/locations/asia-south1/publishers/google/models/gemini-3.1-flash-lite`
>   was not found or your project does not have access to it.
> [global]      OK
> ```
>
> Resolution (no faked compliance): the model id is kept exactly as the brief
> requires (`GEMINI_MODEL=gemini-3.1-flash-lite`) and **only the Gemini
> generation call** falls back to the `global` endpoint via
> `GEMINI_LOCATION` (default `global`). Every other component —
> `text-embedding-004`, Qdrant, Redis, Sarvam, the Node server — runs in /
> targets `asia-south1`. Set `GEMINI_LOCATION=asia-south1` if/when Google
> enables regional Gemini for the project; the code needs no other change.

## Setup

### 1. Prerequisites

* Node.js 18+
* Docker (for local Qdrant + Redis), or your own hosted instances
* A Sarvam AI API key: https://dashboard.sarvam.ai
* A Google Cloud project with billing + Vertex AI API enabled, region `asia-south1`
  (Mumbai), authenticated via `gcloud auth application-default login` or a
  service-account JSON (`GOOGLE_APPLICATION_CREDENTIALS`)

### 2. Install

```bash
npm install
cp .env.example .env
# edit .env with your API keys
```

### 3. Environment variables

Your `.env` should contain:

```env
PORT=8080

SARVAM_API_KEY=your_sarvam_api_key

GOOGLE_CLOUD_PROJECT=your-gcp-project-id
GOOGLE_CLOUD_LOCATION=asia-south1
GEMINI_MODEL=gemini-3.1-flash-lite
USE_VERTEX_AI=true
EMBEDDING_MODEL=text-embedding-004
EMBEDDING_DIM=768

REDIS_URL=your_redis_url

QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION=voice_kb
```

**Never commit `.env` or API keys to GitHub.**

### 4. Start Qdrant + Redis locally

```bash
docker compose up -d
```

> For production, deploy Qdrant, Redis, and this server in regions that minimize
> network latency to the APIs you use. Cross-region network hops can increase
> latency on the real-time voice pipeline.

### 5. Build the RAG index

```bash
npm run ingest
```

This chunks `server/rag/knowledge_base/sample_docs.json` into 300–400 token
pieces (50-token overlap), generates `text-embedding-004` embeddings via
Vertex AI, and upserts them into Qdrant.

Swap in your own docs by editing that JSON file (or point the script at a folder).

### 6. Run the server

```bash
npm start
```

Open:

```text
http://localhost:8080
```

Click **Start talking** and speak.

The UI shows live transcript, reply text, and per-turn latency numbers as they land.

### 7. Benchmark

```bash
npm run benchmark -- --file ./benchmark/sample.wav --runs 10
```

Writes:

```text
benchmark/report.json
benchmark/report.md
```

with p50/p90/max time-to-first-audio-byte.

Without `--file`, it uses a synthetic tone to smoke-test the plumbing. This will not
produce meaningful transcripts, but verifies that the pipeline does not deadlock.

## Project layout

```text
server/
  index.js           Express + WebSocket server
  pipeline.js        Core orchestration: debounced trigger, parallel RAG/cache,
                     streaming LLM -> TTS, barge-in
  sarvamStt.js       Streaming STT client (raw PCM, VAD events)
  sarvamTts.js       Streaming TTS client (text-in, audio-out)
  llmGemini.js       Gemini (Vertex AI) streaming wrapper
  rag/
    qdrantClient.js  Timeout-bounded top-k retrieval
    embedder.js      Vertex AI text-embedding-004 client
    ingest.js        Chunk + embed + upsert script
    knowledge_base/  Sample docs
  cache/redisCache.js
  utils/latencyLogger.js

public/
  Browser client (mic capture, playback, UI)

n8n/
  Admin/async workflow (NOT the hot path)

benchmark/
  Latency test harness + generated reports
```

## Deployment

The recommended deployment architecture is:

```text
                    ┌─────────────────────┐
                    │       Browser       │
                    └──────────┬──────────┘
                               │
                         HTTPS / WSS
                               │
                    ┌──────────▼──────────┐
                    │       Railway       │
                    │                     │
                    │ Node.js + Express   │
                    │ WebSocket Server    │
                    └──────┬───┬───┬──────┘
                           │   │   │
             ┌─────────────┘   │   └──────────────┐
             ▼                 ▼                  ▼
        Sarvam AI       Vertex AI (Mumbai)     Qdrant
        STT + TTS    Gemini 3.1-flash-lite    Vector DB
                        + text-embedding-004
                                                 
                           │
                           ▼
                         Redis
                         Cache
```

### Recommended services

| Service | Platform |
|---|---|
| Node.js + WebSocket | Railway / GCP (asia-south1) |
| Redis | Upstash / Memorystore (asia-south1) |
| Qdrant | Qdrant Cloud / self-hosted (Mumbai) |
| LLM + Embeddings | Google Vertex AI (`asia-south1`) |
| STT + TTS | Sarvam AI |
| Source code | GitHub |

### Production environment variables

Add these variables to your hosting platform:

```env
SARVAM_API_KEY=...
GOOGLE_CLOUD_PROJECT=...
GOOGLE_CLOUD_LOCATION=asia-south1
GEMINI_MODEL=gemini-3.1-flash-lite
USE_VERTEX_AI=true
EMBEDDING_MODEL=text-embedding-004
EMBEDDING_DIM=768
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
REDIS_URL=...
QDRANT_URL=...
QDRANT_API_KEY=...
QDRANT_COLLECTION=voice_kb
```

Do not upload `.env` to GitHub.

## Known limitations / next steps

* Backend action capability (lead capture, CRM writes) was explicitly deprioritized
  per the task brief; `n8n/voice-assistant-workflow.json` shows where a CRM-write
  node would slot in without touching the hot path.
* `ScriptProcessorNode` is used for mic capture for broad browser support; swap
  for an `AudioWorkletNode` to shave a few more ms of jitter in production.
* Sample KB is 5 tiny FAQ docs for demo purposes — replace with your real
  knowledge base via `server/rag/ingest.js`.
* Realtime STT (`saaras:v3-realtime`/`v4-realtime`) is currently in beta and
  offers even lower latency partials than the `saaras:v4` legacy WebSocket used
  here; swap in once you have beta access.
* Gemini model selection is configurable via `GEMINI_MODEL` so the LLM can be
  changed without modifying the pipeline code (spec mandates `gemini-3.1-flash-lite`).