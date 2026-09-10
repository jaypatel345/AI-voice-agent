/**
 * Per-stage turn-latency benchmark.
 *
 * Drives the real /ws/voice WebSocket like a single conversation (one socket,
 * N sequential turns) and prints how long each stage of the pipeline took:
 *
 *   RAG        partial transcript -> retrieved context   (embed + Qdrant)
 *   LLM TTFT   retrieved context  -> first Gemini token
 *   TTS        first token        -> first audio chunk forwarded to client
 *   TOTAL      partial transcript -> first audio chunk    (what the user feels)
 *
 * Usage:
 *   node benchmark/turn_latency.mjs                       # prod, 10 default turns
 *   node benchmark/turn_latency.mjs --turns 5
 *   node benchmark/turn_latency.mjs --url ws://localhost:8080/ws/voice
 *   node benchmark/turn_latency.mjs --q "hello" --q "what are your hours"
 *   node benchmark/turn_latency.mjs --warm 3              # unmeasured warm-up turns first
 */
import WebSocket from 'ws';

const args = process.argv.slice(2);
const getAll = (name) => args.reduce((a, v, i) => (args[i - 1] === `--${name}` ? [...a, v] : a), []);
const get = (name, def) => { const i = args.indexOf(`--${name}`); return i !== -1 ? args[i + 1] : def; };

const URL = get('url', 'wss://voice-assistant-298912362502.asia-south1.run.app/ws/voice');
const WARM = Number(get('warm', 2));
const customQs = getAll('q');
const DEFAULT_QS = [
  'What are your business hours?',
  'Do you work on weekends?',
  'What is your refund policy?',
  'How long do refunds take?',
  'What pricing plans do you offer?',
  'Is there a free tier?',
  'How does onboarding work?',
  'When do I get production API keys?',
  'How is my data protected?',
  'Do you share data with third parties?',
];
let QS = customQs.length ? customQs : DEFAULT_QS;
const N = Number(get('turns', QS.length));
QS = Array.from({ length: N }, (_, i) => QS[i % QS.length]);

const ms = (x) => (x == null ? null : Math.round(x));

function runConversation(questions, collect) {
  return new Promise((resolve) => {
    const ws = new WebSocket(URL);
    let idx = -1, firstAudioAt = 0, sentAt = 0, srv = null, advanced = false;
    const rows = [];

    const next = () => {
      idx++;
      if (idx >= questions.length) { ws.close(); return; }
      sentAt = Date.now(); firstAudioAt = 0; srv = null; advanced = false;
      ws.send(JSON.stringify({ type: 'text_input', text: questions[idx] }));
    };
    // advance once we have the first audio for this turn (turn_done is not always sent)
    const advance = () => { if (advanced) return; advanced = true; setTimeout(next, 1400); };

    ws.on('open', () => setTimeout(next, 1200));
    ws.on('message', (raw) => {
      const m = JSON.parse(raw.toString());
      if (m.type === 'audio_chunk' && !firstAudioAt) firstAudioAt = Date.now() - sentAt;
      if (m.type === 'latency') {
        srv = m.latency;
        if (collect) {
          const rag = srv.rag_latency_ms;
          const llm = srv.llm_first_token_ms != null && srv.rag_retrieved_ms != null
            ? srv.llm_first_token_ms - srv.rag_retrieved_ms : null;
          const total = srv.first_audio_to_client_ms ?? firstAudioAt;
          const tts = total != null && srv.llm_first_token_ms != null
            ? total - srv.llm_first_token_ms : null;
          rows.push({ q: questions[idx], cache: !!srv.cache_hit, rag: ms(rag), llm: ms(llm), tts: ms(tts), total: ms(total) });
        }
        advance();
      }
    });
    ws.on('close', () => resolve(rows));
    ws.on('error', (e) => {
      const why = e.code === 'ECONNREFUSED'
        ? `cannot reach ${URL} — is the server running?  (start it with: npm start)`
        : (e.code || e.message || e);
      console.error('WS error:', why);
      resolve(rows);
    });
    setTimeout(() => ws.close(), 20000 + questions.length * 12000);
  });
}

function stats(arr) {
  const v = arr.filter((x) => x != null).sort((a, b) => a - b);
  if (!v.length) return { median: null, avg: null, p95: null };
  const p = (q) => v[Math.min(v.length - 1, Math.ceil(v.length * q) - 1)];
  return { median: p(0.5), avg: Math.round(v.reduce((a, b) => a + b, 0) / v.length), p95: p(0.95) };
}

(async () => {
  console.log(`\nURL: ${URL}\nTurns: ${N}  (warm-up: ${WARM})\n`);

  if (WARM > 0) {
    await runConversation(Array.from({ length: WARM }, (_, i) => `warm up ${i}`), false);
    await new Promise((r) => setTimeout(r, 800));
  }

  const rows = await runConversation(QS, true);

  const pad = (s, n) => String(s).padStart(n);
  console.log('  #   RAG    LLM    TTS   TOTAL  cache  question');
  console.log('  ─────────────────────────────────────────────────────────────');
  rows.forEach((r, i) => {
    console.log(
      `  ${pad(i + 1, 2)}  ${pad(r.rag ?? '–', 4)}  ${pad(r.llm ?? '–', 4)}  ${pad(r.tts ?? '–', 4)}  ${pad(r.total ?? '–', 5)}   ${r.cache ? '✓ ' : '  '}   ${r.q.slice(0, 40)}`,
    );
  });

  const cols = ['rag', 'llm', 'tts', 'total'];
  console.log('\n  stage    median    avg     p95   (ms)');
  console.log('  ────────────────────────────────────');
  for (const c of cols) {
    const s = stats(rows.map((r) => r[c]));
    console.log(`  ${c.toUpperCase().padEnd(6)}  ${pad(s.median ?? '–', 6)}  ${pad(s.avg ?? '–', 6)}  ${pad(s.p95 ?? '–', 6)}`);
  }
  const under1s = rows.filter((r) => r.total != null && r.total < 1000).length;
  console.log(`\n  turns under 1000 ms: ${under1s}/${rows.length}`);
})();
