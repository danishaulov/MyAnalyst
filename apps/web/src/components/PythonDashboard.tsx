"use client";

import { useState, type ReactNode } from "react";
import { KpiCard } from "./KpiCard";
import { PyConclusionsCard } from "./PyConclusionsCard";
import { Chart } from "./Chart";
import { pyChartsToSpecs } from "@/lib/py-charts";
import { runPythonAsk, type PyAnalysisSpec, type PyConclusions, type PyAnswer } from "@/lib/py-engine";
import { currencySymbol, setActiveCurrency } from "@/lib/currency";

// Renders a Python-engine analysis spec (Phase 5). Reuses the existing KPI cards + ECharts <Chart> via the
// chart adapter, so a Python-computed dashboard looks identical to the TS one.

export function PythonDashboard({
  spec,
  conclusions,
  table,
}: {
  spec: PyAnalysisSpec;
  conclusions?: PyConclusions | null;
  table?: { columns: string[]; rows: Record<string, unknown>[] } | null;
}) {
  setActiveCurrency(spec.currency);
  const charts = pyChartsToSpecs(spec.charts);
  return (
    <div className="space-y-6">
      {spec.scope?.sampled && (
        <p role="status" className="rounded-xl border border-amber-400/30 bg-amber-500/10 p-3 text-sm text-amber-200">
          Analyzed {spec.scope.analyzedRows.toLocaleString()} of {spec.scope.sourceRows.toLocaleString()} rows.
          {" "}Totals and findings describe this sample. Use a smaller file for full-dataset totals.
        </p>
      )}
      <header className="flex flex-wrap items-baseline justify-between gap-2">
        <h2 className="text-lg font-semibold text-slate-100">
          {spec.rowCount.toLocaleString()} rows ·{" "}
          <span className="text-slate-400">{spec.domain.domain.replace("-", " ")}</span>
        </h2>
        <div className="flex items-center gap-2">
          {spec.quality && (
            <span
              className={`rounded-full px-2.5 py-0.5 text-[11px] font-semibold ${
                spec.quality.rating === "good"
                  ? "bg-emerald-500/15 text-emerald-300"
                  : spec.quality.rating === "fair"
                    ? "bg-amber-500/15 text-amber-300"
                    : "bg-red-500/15 text-red-300"
              }`}
              title={spec.quality.issues.join(" · ") || "No quality issues found"}
            >
              Data quality {spec.quality.score}/100
            </span>
          )}
          <span className="rounded-full bg-[#ff5740]/15 px-2.5 py-0.5 text-[11px] font-semibold text-[#ff5740]">
            Python engine
          </span>
        </div>
      </header>
      {spec.quality && spec.quality.issues.length > 0 && (
        <p className="-mt-3 text-[12px] text-slate-500">
          Read with care: {spec.quality.issues.join(" · ")}.
        </p>
      )}

      {conclusions && <PyConclusionsCard c={conclusions} />}

      {table && <AskBox spec={spec} table={table} />}

      <section className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
        {spec.kpis.slice(0, 8).map((k, i) => (
          <KpiCard key={k.id} kpi={k} index={i} />
        ))}
      </section>

      {charts.length > 0 && (
        <section className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          {charts.map((c) => (
            <div key={c.id} className="card p-4">
              <h3 className="mb-1 text-sm font-semibold text-slate-100">{c.title}</h3>
              {c.subtitle && <p className="mb-2 text-[11px] text-slate-500">{c.subtitle}</p>}
              <Chart spec={c} />
            </div>
          ))}
        </section>
      )}

      {spec.bestSellers && (
        <Section title={`Best sellers by ${spec.bestSellers.dimension}`}>
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-3">
            {spec.bestSellers.byRevenue.map((p, i) => (
              <div key={p.name} className="flex items-center justify-between rounded-lg border border-[var(--line)] px-3 py-2">
                <span className="truncate text-[13px] text-slate-200">
                  <span className="text-slate-500">{i + 1}.</span> {p.name}
                </span>
                <span className="shrink-0 text-[12px] tabular-nums text-slate-400">
                  {money(p.revenue)} · {(p.revenueShare * 100).toFixed(0)}%
                </span>
              </div>
            ))}
          </div>
        </Section>
      )}

      {spec.rfm && spec.rfm.segments.length > 0 && (
        <Section title={`Customer value (RFM) · ${spec.rfm.customers.toLocaleString()} ${spec.rfm.entity}s`}>
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-3">
            {spec.rfm.segments.map((s) => (
              <div key={s.key} className="rounded-lg border border-[var(--line)] p-3">
                <div className="flex items-center justify-between">
                  <span className="text-[13px] font-semibold text-slate-100">{s.label}</span>
                  <span className="text-[11px] text-slate-400">{s.size} · {s.sharePct.toFixed(0)}%</span>
                </div>
                <div className="mt-1 text-[11px] text-slate-500">
                  {(s.monetaryShare * 100).toFixed(0)}% of revenue · avg {money(s.avgMonetary)}
                </div>
              </div>
            ))}
          </div>
        </Section>
      )}

      {spec.segments && spec.segments.segments.length > 1 && (
        <Section title={`Natural segments (clustered on ${spec.segments.features.join(", ")})`}>
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-3">
            {spec.segments.segments.map((s) => (
              <div key={s.id} className="rounded-lg border border-[var(--line)] p-3">
                <div className="flex items-center justify-between">
                  <span className="text-[13px] font-semibold text-slate-100">{s.label}</span>
                  <span className="text-[11px] text-slate-400">{s.sharePct.toFixed(0)}%</span>
                </div>
                <div className="mt-1 text-[11px] text-slate-500">{s.size.toLocaleString()} rows</div>
              </div>
            ))}
          </div>
        </Section>
      )}

      <section className="card p-5">
        <h3 className="mb-2 text-sm font-semibold text-slate-100">What the numbers say</h3>
        <ul className="space-y-1.5">
          {spec.facts.map((f) => (
            <li key={f.id} className="flex gap-2 text-[13px] leading-relaxed text-slate-300">
              <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-slate-600" aria-hidden />
              <span>{f.text}</span>
            </li>
          ))}
        </ul>
      </section>

      {spec.methodology && spec.methodology.length > 0 && (
        <details className="card p-5 text-slate-400">
          <summary className="cursor-pointer text-sm font-semibold text-slate-200">How this was computed</summary>
          <ul className="mt-3 space-y-1.5">
            {spec.methodology.map((m, i) => (
              <li key={i} className="flex gap-2 text-[12px] leading-relaxed">
                <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-slate-600" aria-hidden />
                <span>{m}</span>
              </li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}

function AskBox({ spec, table }: { spec: PyAnalysisSpec; table: { columns: string[]; rows: Record<string, unknown>[] } }) {
  const [q, setQ] = useState("");
  const [busy, setBusy] = useState(false);
  const [turns, setTurns] = useState<(PyAnswer & { q: string })[]>([]);

  async function ask() {
    const question = q.trim();
    if (!question || busy) return;
    setBusy(true);
    setQ("");
    try {
      const res = await runPythonAsk(question, table.columns, table.rows, spec.facts);
      setTurns((t) => [{ q: question, ...res }, ...t]);
    } catch (e) {
      setTurns((t) => [{ q: question, answer: e instanceof Error ? e.message : "Couldn't answer that.", provider: "error" }, ...t]);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="card p-5">
      <h3 className="mb-2 text-sm font-semibold text-slate-100">Ask your data</h3>
      <div className="flex gap-2">
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && ask()}
          placeholder="e.g. total revenue by region? average sale price?"
          className="flex-1 rounded-lg border border-slate-700 bg-slate-900/50 px-3 py-2 text-[13px] text-slate-200 placeholder:text-slate-600 focus:border-slate-500 focus:outline-none"
        />
        <button
          type="button"
          onClick={ask}
          disabled={busy || !q.trim()}
          className="rounded-lg border border-slate-700 px-3 py-2 text-[13px] text-slate-200 transition hover:bg-slate-800 disabled:opacity-40"
        >
          {busy ? "…" : "Ask"}
        </button>
      </div>
      {turns.length > 0 && (
        <div className="mt-3 space-y-2">
          {turns.map((t, i) => (
            <div key={i} className="rounded-lg border border-[var(--line)] p-3">
              <div className="text-[12px] font-medium text-slate-400">{t.q}</div>
              <div className="mt-1 text-[13px] text-slate-200">{t.answer}</div>
              {t.scope?.sampled && <p className="mt-1 text-xs text-amber-200">Computed from {t.scope.analyzedRows.toLocaleString()} sampled rows; totals describe the sample.</p>}
              {t.method && (
                <details className="mt-2 text-xs text-slate-400">
                  <summary className="cursor-pointer">How this was computed</summary>
                  <p className="mt-1">{t.method}</p>
                </details>
              )}
            </div>
          ))}
        </div>
      )}
      <p className="mt-2 text-[11px] text-slate-500">Answers are computed in pandas; the AI only phrases them.</p>
    </section>
  );
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="card p-5">
      <h3 className="mb-3 text-sm font-semibold text-slate-100">{title}</h3>
      {children}
    </section>
  );
}

function money(n: number): string {
  if (!Number.isFinite(n)) return "-";
  const sym = currencySymbol();
  const a = Math.abs(n);
  if (a >= 1e6) return sym + (n / 1e6).toFixed(1) + "M";
  if (a >= 1e3) return sym + (n / 1e3).toFixed(0) + "K";
  return sym + n.toFixed(0);
}
