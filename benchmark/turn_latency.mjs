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
    let idx = -1, firstAudioAt = 0, firstReplyAt = 0, sentAt = 0, advanced = false, turnErr = null;
    let perTurnTimer = null;
    const rows = [];

    const record = () => {
      if (!collect || idx < 0 || idx >= questions.length) return;
      // one row per turn, even if it errored
      if (rows[idx]) return;
      rows[idx] = { q: questions[idx], cache: false, rag: null, llm: null, tts: null, total: null, note: turnErr || 'no audio (client-timed)' };
    };
    const next = () => {
      clearTimeout(perTurnTimer);
      idx++;
      if (idx >= questions.length) { ws.close(); return; }
      sentAt = Date.now(); firstAudioAt = 0; firstReplyAt = 0; advanced = false; turnErr = null;
      ws.send(JSON.stringify({ type: 'text_input', text: questions[idx] }));
      // safety: if neither audio nor turn_done nor error lands, move on anyway
      perTurnTimer = setTimeout(() => { record(); advance(); }, 14000);
    };
    const advance = () => { if (advanced) return; advanced = true; clearTimeout(perTurnTimer); setTimeout(next, 1400); };

    ws.on('open', () => setTimeout(next, 1200));
    ws.on('message', (raw) => {
      const m = JSON.parse(raw.toString());
      if (m.type === 'reply_text' && !firstReplyAt) firstReplyAt = Date.now() - sentAt;
      if (m.type === 'audio_chunk' && !firstAudioAt) firstAudioAt = Date.now() - sentAt;
      if (m.type === 'error') { turnErr = `${m.stage || 'error'}: ${(m.message || '').slice(0, 60)}`; }
      if (m.type === 'latency') {
        const srv = m.latency;
        if (collect && !rows[idx]) {
          const rag = srv.rag_latency_ms;
          const llm = srv.llm_first_token_ms != null && srv.rag_retrieved_ms != null
            ? srv.llm_first_token_ms - srv.rag_retrieved_ms : null;
          const total = srv.first_audio_to_client_ms ?? firstAudioAt;
          const tts = total != null && srv.llm_first_token_ms != null
            ? total - srv.llm_first_token_ms : null;
          rows[idx] = { q: questions[idx], cache: !!srv.cache_hit, rag: ms(rag), llm: ms(llm), tts: ms(tts), total: ms(total), note: '' };
        }
        advance();
      }
      // TTS dead (no audio, no latency msg) -> fall back to client wall-clock at turn_done/error
      if (m.type === 'turn_done' || m.type === 'error') {
        if (collect && !rows[idx]) {
          rows[idx] = {
            q: questions[idx], cache: false,
            rag: null, llm: null, tts: null,
            total: firstAudioAt || null,
            note: turnErr || (firstReplyAt ? `~${firstReplyAt}ms to first reply text (RAG+LLM, no audio)` : 'no audio'),
          };
        }
        advance();
      }
    });
    ws.on('close', () => resolve(rows.filter(Boolean)));
    ws.on('error', (e) => {
      const why = e.code === 'ECONNREFUSED'
        ? `cannot reach ${URL} — is the server running?  (start it with: npm start)`
        : (e.code || e.message || e);
      console.error('WS error:', why);
      resolve(rows.filter(Boolean));
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

  if (!rows.length) {
    console.log('No turns completed. Check the server logs above for the cause.');
    return;
  }

  const pad = (s, n) => String(s).padStart(n);
  console.log('  #   RAG    LLM    TTS   TOTAL  cache  question');
  console.log('  ─────────────────────────────────────────────────────────────');
  rows.forEach((r, i) => {
    console.log(
      `  ${pad(i + 1, 2)}  ${pad(r.rag ?? '–', 4)}  ${pad(r.llm ?? '–', 4)}  ${pad(r.tts ?? '–', 4)}  ${pad(r.total ?? '–', 5)}   ${r.cache ? '✓ ' : '  '}   ${r.q.slice(0, 38)}`,
    );
    if (r.note) console.log(`        ↳ ${r.note}`);
  });

  const cols = ['rag', 'llm', 'tts', 'total'];
  const measured = rows.filter((r) => r.total != null);
  console.log(`\n  stage    median    avg     p95   (ms)   [${measured.length}/${rows.length} turns with full server timing]`);
  console.log('  ────────────────────────────────────');
  for (const c of cols) {
    const s = stats(rows.map((r) => r[c]));
    console.log(`  ${c.toUpperCase().padEnd(6)}  ${pad(s.median ?? '–', 6)}  ${pad(s.avg ?? '–', 6)}  ${pad(s.p95 ?? '–', 6)}`);
  }
  const under1s = measured.filter((r) => r.total < 1000).length;
  console.log(`\n  turns under 1000 ms: ${under1s}/${measured.length}`);
  if (measured.length < rows.length) {
    console.log('\n  Note: stage timings only arrive with the first audio chunk, so turns');
    console.log('  where TTS failed show no RAG/LLM/TTS split (see the ↳ note per turn).');
  }
})();
