# Low-Latency AI Voice Assistant with RAG

A real-time, speech-to-speech voice assistant for the web: browser mic → streaming STT →
parallel RAG + prompt build → streaming LLM → streaming TTS → browser playback, engineered
to get audio back to the user in **well under 2 seconds**, targeting **≤1.5s**.

```
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
                          Gemini 3.1 Flash Lite (Vertex AI)
                          short prompt (<1000 tokens), streaming tokens
                                        │  (flushed to TTS at clause boundaries)
                                 Sarvam TTS (streaming)
                                        │  (audio chunks forwarded the instant
                                        │   they arrive — no buffering)
                                 Browser playback (gapless scheduling)
```

## Why this hits the latency budget

* **No sequential waiting anywhere on the hot path.** STT partials trigger RAG
  retrieval and cache lookup *in parallel* (`Promise.all` in `server/pipeline.js`).
* **RAG is timeout-bounded (150ms).** If Qdrant is slow, we drop context and answer
  from general knowledge rather than block the turn — see `RagRetriever.retrieve()`.
* **Cache-first.** Repeated/FAQ questions skip RAG + LLM entirely (Redis, keyed by
  normalized query hash) and jump straight to TTS.
* **Token-level streaming, twice.** Gemini tokens are pushed into the Sarvam TTS
  socket as they arrive and flushed at clause boundaries (`.`, `,`, `;`, newline) —
  audio synthesis starts before the LLM has finished the sentence.
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
| LLM | Gemini 3.1 Flash Lite via Vertex AI (`@google/genai`), streaming |
| TTS | Sarvam AI `bulbul:v3`, WebSocket streaming, `linear16` output |
| Vector DB | Qdrant, precomputed embeddings, `top_k=2` |
| Cache | Redis, SHA1-normalized query keys |
| Server | Node.js + Express + `ws` |
| Frontend | Vanilla JS, Web Audio API (mic capture + gapless scheduled playback) |

## Setup

### 1. Prerequisites
* Node.js 18+
* Docker (for local Qdrant + Redis), or your own hosted instances
* A Sarvam AI API key: https://dashboard.sarvam.ai
* A GCP project with the Vertex AI API enabled, and either:
  * `gcloud auth application-default login` (local dev), or
  * a service-account JSON, referenced via `GOOGLE_APPLICATION_CREDENTIALS`

### 2. Install
```bash
npm install
cp .env.example .env
# edit .env with your keys / project id
```

### 3. Start Qdrant + Redis locally
```bash
docker compose up -d
```
> **Deploy Qdrant, Redis, and this server in the same region as your Vertex AI
> endpoint and Sarvam's nearest PoP** — cross-region hops are the single easiest
> way to blow the latency budget.

### 4. Build the RAG index
```bash
npm run ingest
```
This chunks `server/rag/knowledge_base/sample_docs.json` into 300–500 token
pieces, embeds them (`text-embedding-004`), and upserts into Qdrant. Swap in
your own docs by editing that JSON file (or point the script at a folder).

### 5. Run the server
```bash
npm start
```
Open `http://localhost:8080`, click **Start talking**, and speak. The UI shows
live transcript, reply text, and per-turn latency numbers as they land.

### 6. Benchmark
```bash
npm run benchmark -- --file ./benchmark/sample.wav --runs 10
```
Writes `benchmark/report.json` and `benchmark/report.md` with p50/p90/max
time-to-first-audio-byte — the deliverable latency benchmark report. Without
`--file`, it uses a synthetic tone to smoke-test the plumbing (won't produce
meaningful transcripts, just verifies the pipeline doesn't deadlock).

## Project layout
```
server/
  index.js           Express + WebSocket server
  pipeline.js         Core orchestration: debounced trigger, parallel RAG/cache,
                       streaming LLM -> TTS, barge-in
  sarvamStt.js         Streaming STT client (raw PCM, VAD events)
  sarvamTts.js         Streaming TTS client (text-in, audio-out)
  llmGemini.js         Vertex AI Gemini streaming wrapper
  rag/
    qdrantClient.js    Timeout-bounded top-k retrieval
    ingest.js          Chunk + embed + upsert script
    knowledge_base/    Sample docs
  cache/redisCache.js  Query-response cache
  utils/latencyLogger.js  Per-turn timing + benchmark headline metric
public/                Browser client (mic capture, playback, UI)
n8n/                   Admin/async workflow (NOT the hot path)
benchmark/             Latency test harness + generated reports
```

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
# AI-voice-agent
