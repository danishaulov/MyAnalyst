# Backend reliability and next steps

## What this change addresses

- Python conclusions previously returned unverified figures after a failed repair. Now only a fully passing numeric check returns AI prose; otherwise the response uses the supplied deterministic facts.
- Numeric checks now distinguish currencies, percentages and K/M/B scales, and check small numbers and years. They no longer accept arbitrary ratios formed from unrelated facts.
- The Python ask endpoint previously defaulted to the first numeric column and a sum when it could not understand the question. It now uses an exact simple grammar or an AI plan that sees only the schema, validates an allowlist, and executes pandas aggregations/filters. Unknown metrics and unsupported questions request clarification. No generated Python or SQL is executed.
- Ask narration receives only the computed result, not all column profiles, email values, or client-supplied facts. Grouping by common personal fields and high-cardinality identifiers is rejected. Responses include a calculation method and structured values.
- Missing values no longer cause a genuinely numeric column to be profiled as text. Completeness is still reported independently.
- HTTP requests have explicit shape, row, column, cell, text and byte limits. Invalid input returns a useful 4xx; unexpected failures return a generic 500 with a request ID. Nested NaN/infinity in computed output becomes JSON null.
- CORS permits exact configured origins. A semaphore rejects excess work on a single warm instance; response logs record IDs, endpoint, duration and status without dataset contents or exception messages.
- Python AI calls always set an output budget, hide Groq reasoning, reject truncated results, and apply short retry rules within a shared 40-second deadline. Supported compatible providers use their own defaults.
- The Next.js narrator validates and bounds input, times out provider calls, avoids logging provider response bodies, and applies a numeric gate before returning prose. Streamed answers are buffered until verification completes.
- Upload sampling measures actual UTF-8 bytes and verifies the serialized result after estimation. Scope metadata and sample notices survive the main dashboard, share links and report prose.
- Browser AI caches are scoped to a dataset object and analysis context, rather than filename and row count.
- The obsolete column-controls browser test is replaced with a regression for the current New analysis workflow, which clears the previous dataset before another upload. PDF report/deck downloads are also exercised after the library upgrade.
- Live Python engine, HTTP contract and AI failure tests are included and pass locally. The GitHub workflow update to run these tests and audit production dependencies is supplied as a separate patch: the current GitHub connection cannot push workflow changes without the `workflow` scope. Existing web CI remains unchanged.
- Vulnerable production dependencies were updated: Next.js, PDF import/export, chart rendering, SheetJS, and transitive packages. Sharp and PostCSS overrides keep their existing consumers on patched releases without migrating the app to a new Next.js major. The local production audit passes with zero vulnerabilities. Development-only test-server advisories remain separate.

## Validation

Run from the repository root:

```sh
python -m pip install -r apps/pyapi/api/requirements.txt
python apps/pyapi/api/_test_all.py
```

Run from `apps/web`:

```sh
npm ci
npm run typecheck
npm test
npx playwright install chromium
npm run test:e2e
```

Provider behavior is tested with mocked responses; no production keys or private datasets are required. Browser tests cover uploads, multi-sheet Excel, charts, questions, and the main dashboard's sample notice. A paid-model quality evaluation and deployment smoke test remain separate checks requiring configured access.

## Practical limits

Numeric evidence matching is a conservative filter, **not a proof of meaning, causation or correct query interpretation**. A model can still attach a real number to the wrong entity or choose the wrong valid query plan. Explicit evidence citations and an evaluation corpus are the next quality improvements. Numeric false rejections intentionally fall back to deterministic text. The Python conclude endpoint receives client aggregates; they are not signed server-attested facts.

The main `/analyze` page combines server KPIs/charts with browser analytics. Its Explore tools retain the existing richer browser query engine; the improved Python `/api/ask` is used by `/analyze-py` and direct API clients. Both narrator paths now have server checks. Browser query features such as regression and charts remain available.

Sampling is deterministic and evenly spaced, not guaranteed representative. Sample totals are **not extrapolated**. Scope reports the rows supplied to this backend; parsing may already have sampled an extremely large source file before this stage.

CORS and the two-workload instance guard do not provide account quotas, authentication or distributed rate limiting. A public direct client can bypass CORS. Before scaling or increasing paid AI usage, enforce authentication and per-user budgets at a shared gateway or datastore. Requests still send uploaded records to the Python server; only schema or computed aggregates go to the model. Aggregate labels can themselves be sensitive.

## Prioritized product and backend roadmap

1. **Private saved workspaces and reusable dataset handles.** Store datasets with explicit retention/deletion rules, authenticate users, isolate every object and query by tenant, and issue short-lived dataset handles. Repeated questions can then send a handle instead of uploading the same sampled file every time. Cache only computed aggregates under tenant + dataset version + normalized query, never a shared filename.
2. **Full-file background analysis.** Use private object storage uploads and an asynchronous worker with job progress, cancellation and idempotency. Move expensive analyses out of the synchronous request budget. This enables exact totals for large files, avoids repeated parsing, and reduces the current sample limitation. Retain Vercel for the web/gateway layer.
3. **Scheduled comparison and change alerts.** Reuse the existing comparison and anomaly engines, but add scheduled source refresh, versioned snapshots and alerts for meaningful KPI changes. Include data-quality changes and explain which segments account for the movement. Notify only on configured thresholds.
4. **Clickable evidence for each AI conclusion.** Link a sentence to the exact query, source columns, filters, aggregate value, dataset version and sample scope. Let users inspect the computation and provide feedback. Add bilingual evaluation cases, adversarial labels, ambiguous questions and incorrect-causation cases to CI.
5. **Cost and reliability dashboard.** Track provider latency/status, fallback rate, rejected claims and token usage by task without logging private content. Apply durable per-user limits and choose a smaller narration model where evaluations justify it. Do not hide failures behind a 200 alone; record the fallback reason.

Implement saved datasets and background compute before adding heavier AI agents or persistent conversations. These give larger accuracy, latency and cost benefits than rewriting prompts alone.

## References

- [Vercel function limits](https://vercel.com/docs/functions/limitations): payloads are limited to 4.5 MB; this implementation leaves a safety margin and preserves the project's configured 60-second duration.
- [Groq reasoning](https://console.groq.com/docs/reasoning) and [compatibility](https://console.groq.com/docs/openai): provider-specific reasoning and response options.
- [SheetJS installation](https://docs.sheetjs.com/docs/getting-started/installation/nodejs/): official updated package distribution rather than the stale npm registry release.
