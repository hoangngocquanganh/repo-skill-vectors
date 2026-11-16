// Builds the vector files for Repo Skill AI search.
//   d-<ver>.bin : EmbeddingGemma-2 (Workers AI REST), 256 dims, int8  → used with the rsa-embed Worker (option D)
//   b-<ver>.bin : multilingual MiniLM (run here on the Actions runner), 384 dims, int8 → used with the in-browser model (option B)
//   meta.json   : keys, text hashes, file names, worker URL
// Only new or changed texts are embedded again (vectors are reused by key + hash).
import fs from 'node:fs';
import crypto from 'node:crypto';

const SITE = process.env.SITE_EXPORT || 'https://hoangngocquanganh.com/wp-json/rsa/v1/export';
const ACC = process.env.CF_ACCOUNT_ID || '';
const TOK = process.env.CF_API_TOKEN || '';
const WORKER_URL = process.env.WORKER_URL || process.env.WORKER_URL_VAR || '';
const D_MODEL = '@cf/google/embeddinggemma-2', D_DIM = 256;
const B_MODEL = 'Xenova/paraphrase-multilingual-MiniLM-L12-v2', B_DIM = 384;
const DATA = new URL('../data/', import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1');
fs.mkdirSync(DATA, { recursive: true });

const hash = (s) => crypto.createHash('md5').update(s).digest('hex').slice(0, 10);
const old = fs.existsSync(DATA + 'meta.json') ? JSON.parse(fs.readFileSync(DATA + 'meta.json', 'utf8')) : null;

function loadOld(kind, dim) {
  // map key → {h, row Int8Array}
  const m = new Map();
  if (!old || !old[kind] || old[kind].dim !== dim) return m;
  const f = DATA + old[kind].file;
  if (!fs.existsSync(f)) return m;
  const buf = new Int8Array(fs.readFileSync(f).buffer.slice(0));
  old.keys.forEach((k, i) => { if (old[kind].ok?.[i] !== 0) m.set(k, { h: old.hashes[i], row: buf.subarray(i * dim, (i + 1) * dim) }); });
  return m;
}
function q8(v) {
  let n = 0; for (const x of v) n += x * x; n = Math.sqrt(n) || 1;
  const u = new Int8Array(v.length);
  for (let i = 0; i < v.length; i++) u[i] = Math.max(-127, Math.min(127, Math.round((v[i] / n) * 127)));
  return u;
}

const res = await fetch(SITE + '?t=' + Date.now(), { headers: { 'User-Agent': 'repo-skill-vectors-build' } });
if (!res.ok) throw new Error('export ' + res.status);
const items = (await res.json()).items.filter((x) => x.k && x.t);
const keys = items.map((x) => x.k);
const docD = items.map((x) => `title: ${x.ti || 'none'} | text: ${x.t}`);
const docB = items.map((x) => `${x.ti}. ${x.t}`);
const hashes = items.map((x, i) => hash(docD[i]));
console.log('items', items.length);

async function build(kind, dim, embedBatch, batchSize) {
  const prev = loadOld(kind, dim);
  const out = new Int8Array(items.length * dim), ok = new Array(items.length).fill(1);
  const todo = [];
  items.forEach((_, i) => {
    const p = prev.get(keys[i]);
    if (p && p.h === hashes[i]) out.set(p.row, i * dim); else todo.push(i);
  });
  console.log(kind, 'reuse', items.length - todo.length, 'embed', todo.length);
  for (let s = 0; s < todo.length; s += batchSize) {
    const idx = todo.slice(s, s + batchSize);
    try {
      const vecs = await embedBatch(idx);
      idx.forEach((i, j) => out.set(q8(vecs[j]), i * dim));
    } catch (e) {
      console.error(kind, 'batch failed', e.message);
      idx.forEach((i) => { ok[i] = 0; });
      if (/403|401|quota|limit/i.test(e.message)) { todo.slice(s + batchSize).forEach((i) => { ok[i] = 0; }); break; }
    }
    if (s % (batchSize * 10) === 0) console.log(kind, Math.min(todo.length, s + batchSize), '/', todo.length);
  }
  return { out, ok };
}

const ver = new Date().toISOString().replace(/[-:T]/g, '').slice(0, 12);
const meta = { ver, at: Date.now(), n: items.length, keys, hashes, worker: WORKER_URL || old?.worker || '' };

// ---- D via the rsa-embed Worker (no Cloudflare secret needed) ----
if (meta.worker) {
  const out = new Int8Array(items.length * D_DIM), ok = new Array(items.length).fill(0);
  const pos = new Map(keys.map((k, i) => [k, i]));
  let n = Infinity;
  for (let from = 0; from < n; from += 50) {
    let j;
    for (let t = 0; t < 3; t++) {
      try { const r = await fetch(`${meta.worker}/b?from=${from}`); if (r.ok) { j = await r.json(); break; } console.error('slice', from, r.status); }
      catch (e) { console.error('slice', from, e.message); }
      await new Promise((r) => setTimeout(r, 2000));
    }
    if (!j) break;
    n = j.n;
    const vs = j.v ? j.v.split(',') : [];
    j.keys.forEach((k, x) => {
      const i = pos.get(k);
      if (i === undefined || !vs[x]) return;
      out.set(new Int8Array(Buffer.from(vs[x], 'base64')), i * D_DIM); ok[i] = 1;
    });
    if (from % 1000 === 0) console.log('d', from, '/', n);
  }
  const good = ok.filter(Boolean).length;
  console.log('d ok', good, '/', items.length);
  if (good > items.length * 0.5) {
    fs.writeFileSync(DATA + `d-${ver}.bin`, Buffer.from(out.buffer));
    meta.d = { file: `d-${ver}.bin`, dim: D_DIM, model: D_MODEL, ok: good === items.length ? undefined : ok.map(Number) };
  }
} else if (ACC && TOK) {
// ---- D: Workers AI REST (fallback when no Worker URL) ----
  const d = await build('d', D_DIM, async (idx) => {
    const r = await fetch(`https://api.cloudflare.com/client/v4/accounts/${ACC}/ai/run/${D_MODEL}`, {
      method: 'POST', headers: { Authorization: 'Bearer ' + TOK, 'Content-Type': 'application/json' },
      body: JSON.stringify({ input: idx.map((i) => docD[i]), dimensions: D_DIM }),
    });
    const j = await r.json().catch(() => ({}));
    if (!r.ok || !j.result) throw new Error(r.status + ' ' + JSON.stringify(j.errors || j).slice(0, 200));
    return j.result.data.map((x) => x.embedding);
  }, 50);
  fs.writeFileSync(DATA + `d-${ver}.bin`, Buffer.from(d.out.buffer));
  meta.d = { file: `d-${ver}.bin`, dim: D_DIM, model: D_MODEL, ok: d.ok.every(Boolean) ? undefined : d.ok };
} else if (old?.d) {
  meta.d = null; // keys may have changed → old D file no longer lines up
  console.log('no CF secrets: skip D');
}

// ---- B: MiniLM on the runner ----
const { pipeline, env } = await import('@xenova/transformers');
env.cacheDir = './.models';
const ext = await pipeline('feature-extraction', B_MODEL, { quantized: true });
const b = await build('b', B_DIM, async (idx) => {
  const o = await ext(idx.map((i) => docB[i]), { pooling: 'mean', normalize: true });
  return idx.map((_, j) => o.data.slice(j * B_DIM, (j + 1) * B_DIM));
}, 32);
fs.writeFileSync(DATA + `b-${ver}.bin`, Buffer.from(b.out.buffer));
meta.b = { file: `b-${ver}.bin`, dim: B_DIM, model: B_MODEL, ok: b.ok.every(Boolean) ? undefined : b.ok };

fs.writeFileSync(DATA + 'meta.json', JSON.stringify(meta));
// keep only the files meta.json points to
for (const f of fs.readdirSync(DATA)) if (/\.bin$/.test(f) && f !== meta.b.file && f !== meta.d?.file) fs.unlinkSync(DATA + f);
console.log('done', ver, meta.d ? 'D+B' : 'B only');
