// rsa-embed: turns a search query into a vector for Repo Skill AI (hoangngocquanganh.com/repo-skill/).
// GET /e?q=<query>  →  {v: "<base64 int8, 256 values>", m: "<model>"}
// Results are cached at the edge for 30 days, so repeated queries cost no Workers AI neurons.
const MODEL = '@cf/google/embeddinggemma-2';
const DIM = 256;
const ALLOW = [/^https:\/\/(www\.)?hoangngocquanganh\.com$/, /^http:\/\/localhost(:\d+)?$/];

function cors(origin) {
  const ok = origin && ALLOW.some((r) => r.test(origin));
  return {
    'Access-Control-Allow-Origin': ok ? origin : 'https://hoangngocquanganh.com',
    'Access-Control-Allow-Methods': 'GET, OPTIONS',
    Vary: 'Origin',
  };
}

function toInt8B64(vec) {
  let n = 0;
  for (const x of vec) n += x * x;
  n = Math.sqrt(n) || 1;
  const u = new Int8Array(vec.length);
  for (let i = 0; i < vec.length; i++) u[i] = Math.max(-127, Math.min(127, Math.round((vec[i] / n) * 127)));
  let s = '';
  for (const b of new Uint8Array(u.buffer)) s += String.fromCharCode(b);
  return btoa(s);
}

export default {
  async fetch(req, env, ctx) {
    const url = new URL(req.url);
    const h = cors(req.headers.get('Origin'));
    if (req.method === 'OPTIONS') return new Response(null, { headers: h });
    if (url.pathname === '/health') return Response.json({ ok: true, model: MODEL, dim: DIM }, { headers: h });
    if (url.pathname !== '/e') return new Response('not found', { status: 404, headers: h });

    const q = (url.searchParams.get('q') || '').trim().toLowerCase().replace(/\s+/g, ' ').slice(0, 200);
    if (!q) return Response.json({ err: 'empty' }, { status: 400, headers: h });

    // Edge cache keyed by the normalised query (not by Origin)
    const key = new Request('https://rsa-embed.cache/e?q=' + encodeURIComponent(q));
    const cache = caches.default;
    let hit = await cache.match(key);
    if (hit) {
      const r = new Response(hit.body, hit);
      for (const [k, v] of Object.entries(h)) r.headers.set(k, v);
      return r;
    }
    try {
      const out = await env.AI.run(MODEL, { input: ['task: search result | query: ' + q], dimensions: DIM });
      const body = JSON.stringify({ v: toInt8B64(out.data[0].embedding), m: MODEL });
      const res = new Response(body, { headers: { 'Content-Type': 'application/json', 'Cache-Control': 'public, max-age=2592000' } });
      ctx.waitUntil(cache.put(key, res.clone()));
      for (const [k, v] of Object.entries(h)) res.headers.set(k, v);
      return res;
    } catch (e) {
      // e.g. daily free quota used up → the site falls back to in-browser AI
      return Response.json({ err: String(e && e.message || e).slice(0, 200) }, { status: 503, headers: h });
    }
  },
};
