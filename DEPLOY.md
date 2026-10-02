# Deploying MyAnalyst to Vercel

This is a **monorepo**. The deployable Next.js app lives in **`apps/web`**, not at the repo root.
That one fact is the cause of almost every "404: NOT_FOUND" you'll see — Vercel built from the root,
found no Next.js app, and served nothing.

## Fix the 404 — set the Root Directory

1. Go to your project on **vercel.com → Settings → Build and Deployment** (or **General**).
2. Find **Root Directory** and set it to:
   ```
   apps/web
   ```
   (Click **Edit**, type or browse to `apps/web`, **Save**.)
3. Make sure **Framework Preset** is **Next.js** (auto-detected once the root is correct).
4. Go to **Deployments → ⋯ on the latest → Redeploy** (uncheck "use existing build cache").

That's it. No environment variables are required for the default build.

### If you're importing the repo for the first time
On the **New Project** screen, click **Edit** next to *Root Directory* and choose `apps/web`
**before** clicking Deploy.

### Or deploy from the CLI (sets the root automatically)
```bash
cd apps/web
npx vercel        # first run links/creates the project
npx vercel --prod # production deploy
```
Running from inside `apps/web` makes that directory the project root.

## Optional — enable AI-narrated insights
Add these in **Settings → Environment Variables** (all optional; without them the local
templated narrator is used). See `apps/web/.env.example` for details.

| Variable | Example | Notes |
|---|---|---|
| `NEXT_PUBLIC_LLM_ENABLED` | `1` | Turns on the LLM path |
| `LLM_PROVIDER` | `groq` | `anthropic` \| `groq` \| `openai` \| `gemini` \| `openrouter` |
| `LLM_API_KEY` | `gsk_…` | Server-side only — never exposed to the browser |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Provider-specific model id |

## Sanity check after deploy
- `/` → landing page
- `/analyze` → the analyzer (try `/analyze?demo=1` for the sample)
- `/api/insights` → returns `{"insights":[],"provider":"none"}` on a POST with `{"kpis":[]}` when no key is set

## Python compute project and preview checks

Deploy the Python service separately with Root Directory `apps/pyapi`, then point the web project's
`NEXT_PUBLIC_PY_API` at that service. Its AI settings are independent of the web project's settings;
see `apps/pyapi/.env.example`. Configure secrets only on the server.

The API now allows exact origins using `API_ALLOWED_ORIGINS`. Its defaults include `myanalyst.net`,
`www.myanalyst.net` and local development on port 3000. To test a Vercel preview, add the **actual web
preview origin** to the Python project's list and redeploy that API preview. Do not use a wildcard
for all Vercel projects. CORS is not authentication; shared account quotas still require a gateway.

Before production rollout, test both projects as previews:

1. `GET /api/index` on the Python service returns health and AI availability.
2. POST a small dataset to `/api/analyze`; verify KPIs and `scope`.
3. POST `total Revenue by Region?` and the same dataset to `/api/ask`; verify the actual group totals.
4. Open the web preview, upload CSV and multi-sheet Excel, inspect the AI summary, and export a chart/report.
5. Verify malformed input returns 400, oversized input returns 413, and disallowed browser origins receive 403.

Roll out the API first, then the web app. Retain the previous deployments for rollback. The changes are
backward compatible with normal existing client payloads, except unknown routes, malformed data and
unlisted origins now fail explicitly. They do not require database migrations.

See `docs/07-backend-reliability.md` for implementation limits and the next backend priorities.

## SEO — refreshing the favicon & search snippet on Google

Google does **not** update the favicon or the search-result description the moment you deploy.
It only refreshes them when **Googlebot re-crawls** the site, which lags a deploy by **days to
weeks**. So right after changing the logo or the description, the old version keeps showing — this
is a cache/crawl-timing issue, not a bug, and there is nothing to "fix" in code once the new assets
are live.

**Where the assets live** (all should already be the coral logomark):
- Favicon / app icons: `apps/web/src/app/{favicon.ico, icon.png, icon.svg, apple-icon.png}`.
  `favicon.ico` exists specifically so the legacy **`/favicon.ico`** path Google fetches directly
  doesn't 404 (Next also emits `<link rel="icon">` from `icon.png`/`icon.svg`).
- Social/share image: `apps/web/src/app/opengraph-image.tsx` (embeds `public/logo.png`).
- Title + meta description: the `DESCRIPTION` constant + `metadata` export in
  `apps/web/src/app/layout.tsx` (the home page inherits these).

> Note: Google may **ignore the meta description** and auto-generate the snippet from visible page
> text (e.g. the hero copy in `apps/web/src/app/page.tsx`). A crisp, accurate `DESCRIPTION` makes it
> more likely Google uses ours, but the final snippet is Google's choice per query.

**To force a refresh (do this after any logo/description change):**
1. Open **[Google Search Console](https://search.google.com/search-console)** for `myanalyst.net`.
2. **URL Inspection** → enter `https://myanalyst.net` → **Request Indexing**.
3. Snippet typically updates within a few days; the favicon can take a couple of weeks.

Quick local check that the new assets are actually served (before blaming the cache):
```bash
cd apps/web && npx next build && npx next start
# then in another shell:
curl -sI http://localhost:3000/favicon.ico   # 200, image/x-icon
curl -s  http://localhost:3000/ | grep -i 'rel="icon"\|name="description"'
```
