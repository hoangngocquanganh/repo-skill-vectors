# repo-skill-vectors

Search vectors for [Repo Skill AI](https://hoangngocquanganh.com/repo-skill/). Everything here runs on free tiers, so the website's hosting never runs AI.

| Part | Where it runs | What it does |
|---|---|---|
| `build/build.mjs` | GitHub Actions, daily | Reads the site's `/wp-json/rsa/v1/export`, embeds new or changed texts, writes `data/` |
| `data/d-*.bin` | jsDelivr CDN | EmbeddingGemma-2 vectors (256 dims, int8) for option D |
| `data/b-*.bin` | jsDelivr CDN | Multilingual MiniLM vectors (384 dims, int8) for option B |
| `worker/` | Cloudflare Workers (free) | `GET /e?q=` embeds one search query with Workers AI, edge-cached |

The browser tries the options in order: **D** (Worker turns the query into a vector, the browser compares it with `d-*.bin`), then **B** (the browser loads MiniLM only for the query and compares it with `b-*.bin`), then **A** (everything in the browser), then keyword search.

## Secrets

- `CF_ACCOUNT_ID`: Cloudflare account ID
- `CF_API_TOKEN`: API token with **Workers Scripts: Edit** and **Workers AI: Read**

Without the secrets the workflow still builds the B vectors. It skips the Worker and the D vectors.
