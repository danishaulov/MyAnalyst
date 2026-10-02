# MyAnalyst — Python analysis engine (`apps/pyapi`)

The analysis engine: **Python (pandas/numpy/scipy/statsmodels)** does all the compute, and an LLM (Groq)
writes the conclusions from the computed facts. **Live** — `/analyze` on myanalyst.net runs on this engine,
with the in-browser TypeScript engine kept only as a fallback if the API is unreachable.

## Architecture

Deployed as its **own Vercel project** (`quantia-api` → `https://quantia-api.vercel.app`), separate from the
Next.js web app — Next shadows `/api/*` in the monorepo, so a same-origin Python function never routes. The
web app calls this API cross-origin via `NEXT_PUBLIC_PY_API`; CORS uses `API_ALLOWED_ORIGINS`, and the web CSP `connect-src` is
derived from that origin. See `docs/06-python-migration.md`.

Trade-off accepted: data leaves the browser to the server (the old "data never leaves the page" privacy moat
is dropped) in exchange for rigorous standard stats and big-data headroom.

## Layout

| File | Role |
|---|---|
| `index.py` | single entrypoint (`BaseHTTPRequestHandler`); a `vercel.json` rewrite maps `/api/{analyze,conclude,ask}` → `/api/index?fn=…`. Must `sys.path.insert` its own dir so sibling `_*.py` imports resolve on Vercel |
| `_engine.py` | the pandas engine — profiling, metric **semantics** (revenue vs cost vs attribute), structure-aware **domain**, revenue-first **KPIs**, best-sellers, Pareto |
| `_stats.py` | scipy/statsmodels inference — correlations (Fisher CIs + FDR), OLS drivers, ANOVA group gaps, chi-square associations |
| `_timeseries.py`, `_forecast.py` | monthly trend + significance, biggest-swing regime change, Holt-Winters forecast with a 95% band (clamped ≥0 for non-negative metrics) |
| `_outliers.py`, `_distribution.py`, `_segments.py`, `_rfm.py` | skew-vs-anomaly outliers, normality, k-means segments, RFM |
| `_currency.py` | detects the dataset's currency (header codes/symbols + cell values) so money isn't hardcoded to `$` |
| `_insights.py` | grounded **facts** + chart readings + the zero-API templated narrative |
| `_conclude.py`, `_ask.py`, `_groq.py` | LLM conclusions / ask-your-data + a stdlib Groq client (sends a real User-Agent — Cloudflare 403s the default `Python-urllib` UA from datacenter IPs) |
| `_server.py` | local dev server (ThreadingHTTPServer on :8000), reuses `index.handler` |
| `_test_*.py`, `_demo.py`, `_parity_one.py` | tests + demos (underscore = not routed) |

Vercel's request body is capped at **4.5 MB** — the frontend samples large files before POSTing.

## Run it (local)

Use `py`, not `python`, on Windows.

```bash
py _test_all.py     # all suites (engine + conclude); PYTHONUTF8=1 on a non-UTF8 console
py _demo.py         # smart KPIs + facts on synthetic car-sales data
py _server.py       # serve on http://127.0.0.1:8000 (web `dev:py` runs this)
```

Requires `pandas numpy scipy statsmodels` (`pip install -r requirements.txt`).

## Design notes

- **`_engine.analyze(df) -> dict`** returns a JSON-serializable spec (currency, domain, columns, kpis,
  bestSellers, trend, forecast, stats, outliers, segments, rfm, distributions, charts, facts, chartReadings,
  narrative, methodology) the web frontend renders with the existing ECharts/KPI cards.
- The engine computes deterministic FACTS; the LLM narrates them and a **numeric evidence check** rejects
  unsupported figures. One repair is allowed; failed verification falls back to templated facts. This is
  not proof of semantic correctness. Provider failures and missing keys also use the fallback.
- `/api/ask` validates a schema-only query plan before computing pandas aggregates and filtering. It
  returns clarification instead of guessing a metric, and includes a calculation method with its answer.
- See [backend reliability](../../../docs/07-backend-reliability.md) for limits, testing and next steps.
- The semantics layer is the heart of it: `revenue_metric` (top-line money, never cost), `is_additive`
  (sum flows, average attributes), `is_transaction_grain`, `detect_domain` (price/volume are only weak
  financial hints; a transaction stream is sales/ops, never a price series).
