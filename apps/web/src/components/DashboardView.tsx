"use client";

import { useState, type ReactNode, type Ref } from "react";
import type { DashboardSpec, Table } from "@/lib/types";
import { KpiCard } from "./KpiCard";
import { CaveatStrip } from "./CaveatStrip";
import { caveatForKpi } from "@/lib/caveats";
import { Chart } from "./Chart";
import { InsightCard } from "./InsightCard";
import { ChartBuilder } from "./ChartBuilder";
import { CleaningReport } from "./CleaningReport";
import { QualityCard } from "./QualityCard";
import { AnomalyCard } from "./AnomalyCard";
import { TimeTrendCard } from "./TimeTrendCard";
import { ContributionCard } from "./ContributionCard";
import { ThemesCard } from "./ThemesCard";
import { SegmentCard } from "./SegmentCard";
import { ConcentrationCard } from "./ConcentrationCard";
import { RfmCard } from "./RfmCard";
import { RelationshipCard } from "./RelationshipCard";
import { DriverCard } from "./DriverCard";
import { ScenarioCard } from "./ScenarioCard";
import { CohortCard } from "./CohortCard";
import { ActionPlanCard } from "./ActionPlanCard";
import { domainFocus } from "@/lib/domain-pack";
import { QueryBox } from "./QueryBox";
import { DataTable } from "./DataTable";
import { MethodologyCard } from "./MethodologyCard";
import { PyConclusionsCard } from "./PyConclusionsCard";
import type { PyConclusions } from "@/lib/py-engine";

// The dashboard body, rendered identically by the live analyzer (/analyze) and the read-only shared
// view (/view). Everything renders from the DashboardSpec alone; interactive tools (Ask / Build / Browse)
// only appear when the raw `table` is available.
//
// To keep ~20 findings from overwhelming the reader, the body is organised into tabs (Overview first).
// All panels stay mounted - hidden ones use the `hidden` attribute - so (a) interactive state survives
// tab switches and (b) the PNG/PDF export can reveal every panel for a complete report. Switching a tab
// dispatches a window resize so ECharts canvases that first laid out inside a hidden panel size up.

type TabId = "overview" | "insights" | "trends" | "quality" | "explore";

export function DashboardView({
  spec,
  table,
  conclusions,
  innerRef,
}: {
  spec: DashboardSpec;
  table?: Table | null;
  conclusions?: PyConclusions | null;
  innerRef?: Ref<HTMLDivElement>;
}) {
  const has = {
    insights:
      spec.insights.length > 0 ||
      !!spec.textAnalysis?.length ||
      (spec.segmentation?.segments.length ?? 0) > 1 ||
      !!spec.concentration?.length ||
      !!spec.rfm?.segments.length ||
      !!spec.anomalies?.length,
    trends:
      !!spec.timeAnalysis?.length ||
      !!spec.contributions?.length ||
      !!(spec.drivers && spec.drivers.drivers.length > 0) ||
      (spec.cohorts?.cohorts.length ?? 0) > 1 ||
      (spec.relationships?.columns.length ?? 0) >= 2 ||
      spec.charts.length > 0,
    explore: !!table,
  };

  const tabs: { id: TabId; label: string }[] = [
    { id: "overview", label: "Overview" },
    ...(has.insights ? [{ id: "insights" as const, label: "Insights" }] : []),
    ...(has.trends ? [{ id: "trends" as const, label: "Trends & drivers" }] : []),
    { id: "quality", label: "Data quality" },
    ...(has.explore ? [{ id: "explore" as const, label: "Explore" }] : []),
  ];

  const [active, setActive] = useState<TabId>("overview");

  const aiBadge =
    spec.narrator === "llm" ? (
      <span className="rounded-full bg-gradient-to-r from-[#ff5740]/20 to-[#ff8a4c]/20 px-2.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-[#ff5740]">
        ✨ AI-narrated
      </span>
    ) : undefined;

  return (
    <div className="space-y-6" ref={innerRef}>
      {spec.computeScope?.sampled && (
        <p role="status" className="rounded-xl border border-amber-400/30 bg-amber-500/10 p-3 text-sm text-amber-200">
          Server KPIs and charts use {spec.computeScope.analyzedRows.toLocaleString()} of {spec.computeScope.sourceRows.toLocaleString()} rows.
          {" "}Their totals describe the sample. Explore questions use the data available on your device.
        </p>
      )}
      {/* Persistent context - shown on every tab and kept in exports. */}
      <div className="card flex flex-wrap items-center justify-between gap-3 p-4">
        <div>
          <p className="text-sm font-semibold text-slate-100">{spec.datasetName}</p>
          <p className="text-xs text-slate-400">
            {spec.rowCount.toLocaleString()} rows · {spec.profiles.length} columns
          </p>
        </div>
        <div className="text-right">
          <span className="rounded-full bg-[#ff5740]/15 px-3 py-1 text-xs font-semibold text-[#ff5740]">
            {spec.domain.domain} · {(spec.domain.confidence * 100).toFixed(0)}% confidence
          </span>
          <p className="mt-1 max-w-md text-[11px] text-slate-500">{spec.domain.reason}</p>
        </div>
      </div>

      <CaveatStrip caveats={spec.caveats} smallSample={spec.smallSample} />

      {/* Tab bar - sticky, and excluded from the static export (the report shows all panels instead). */}
      <nav
        data-export-exclude
        aria-label="Dashboard sections"
        className="sticky top-2 z-20 flex gap-1 overflow-x-auto rounded-xl border border-[var(--line)] bg-slate-950/85 p-1 backdrop-blur"
      >
        {tabs.map((t) => {
          const on = active === t.id;
          return (
            <button
              key={t.id}
              type="button"
              aria-current={on ? "page" : undefined}
              onClick={() => setActive(t.id)}
              className={`whitespace-nowrap rounded-lg px-4 py-1.5 text-sm font-medium transition ${
                on ? "bg-[#ff5740] text-white shadow-sm shadow-[#ff5740]/30" : "text-slate-400 hover:bg-slate-800/60 hover:text-slate-200"
              }`}
            >
              {t.label}
            </button>
          );
        })}
      </nav>

      {/* ── Overview ───────────────────────────────────────────────────────── */}
      <Panel id="overview" active={active}>
        {/* AI conclusions (Groq reads the Python-computed KPIs + charts) - the decision-first read at the
            top of the Overview tab, and only there. Replaces the old auto-generated Executive summary. */}
        {conclusions && <PyConclusionsCard c={conclusions} />}

        {spec.actions && spec.actions.length > 0 && (
          <Section title="Your action plan" subtitle="What to do next - ranked by impact, grounded in the numbers.">
            <ActionPlanCard actions={spec.actions} />
          </Section>
        )}

        <Section title="Key metrics" subtitle="Auto-selected for this dataset's shape and domain.">
          <div className="grid grid-cols-2 gap-4 md:grid-cols-3 lg:grid-cols-4">
            {spec.kpis.slice(0, 8).map((kpi, i) => (
              <KpiCard key={kpi.id} kpi={kpi} index={i} caveat={spec.caveats ? caveatForKpi(kpi, spec.caveats) : undefined} />
            ))}
          </div>
        </Section>

        {/* The "About this data" story is intentionally NOT rendered: it's background context for the AI
            (orienting the Ask-your-data answers), not a panel for the reader. See `spec.story` passed into
            QueryBox's `analysis` below. */}
      </Panel>

      {/* ── Insights ───────────────────────────────────────────────────────── */}
      {has.insights && (
        <Panel id="insights" active={active}>
          {spec.insights.length > 0 && (
            <Section title="What the data is telling you" subtitle="The most important, statistically-backed findings - and what to do about them." badge={aiBadge}>
              <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
                {spec.insights.map((ins, i) => (
                  <InsightCard key={ins.id} insight={ins} index={i} />
                ))}
              </div>
            </Section>
          )}

          {spec.textAnalysis && spec.textAnalysis.length > 0 && (
            <Section title="Themes &amp; sentiment" subtitle="The recurring topics in your open-text responses, with sentiment and real quotes.">
              <ThemesCard analyses={spec.textAnalysis} />
            </Section>
          )}

          {spec.segmentation && spec.segmentation.segments.length > 1 && (
            <Section title="Natural segments" subtitle="Groups the data falls into on its own - and what defines each.">
              <SegmentCard segmentation={spec.segmentation} profiles={spec.profiles} table={table} />
            </Section>
          )}

          {spec.concentration && spec.concentration.length > 0 && (
            <Section title="The 80–20" subtitle="How concentrated each measure is - whether a vital few categories carry the whole number (a risk worth knowing).">
              <ConcentrationCard concentration={spec.concentration} profiles={spec.profiles} table={table} />
            </Section>
          )}

          {spec.rfm && spec.rfm.segments.length > 0 && (
            <Section title="Customer value (RFM)" subtitle="Customers grouped by how recently, how often, and how much they buy - your Champions through to those slipping away.">
              <RfmCard rfm={spec.rfm} table={table} profiles={spec.profiles} />
            </Section>
          )}

          {spec.anomalies && spec.anomalies.length > 0 && (
            <Section title="Anomalies &amp; outliers" subtitle="Unusual values that can skew averages - flagged so you can verify or exclude them.">
              <AnomalyCard anomalies={spec.anomalies} profiles={spec.profiles} />
            </Section>
          )}
        </Panel>
      )}

      {/* ── Trends & drivers ───────────────────────────────────────────────── */}
      {has.trends && (
        <Panel id="trends" active={active}>
          {spec.timeAnalysis && spec.timeAnalysis.length > 0 && (
            <Section title="Trends over time" subtitle="Period-over-period change with a moving-average trend line.">
              <TimeTrendCard analyses={spec.timeAnalysis} profiles={spec.profiles} />
            </Section>
          )}

          {spec.contributions && spec.contributions.length > 0 && (
            <Section title="What drove the change" subtitle="The latest period's move in the primary metric, attributed to the segments behind it.">
              <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
                {spec.contributions.map((c) => (
                  <ContributionCard key={`${c.metric}-${c.dimension}`} analysis={c} profiles={spec.profiles} />
                ))}
              </div>
            </Section>
          )}

          {spec.drivers && spec.drivers.drivers.length > 0 && (
            <Section title="What moves the needle" subtitle={`The factors that most influence ${spec.drivers.target}, each holding the others constant.`}>
              <DriverCard drivers={spec.drivers} />
            </Section>
          )}

          {spec.drivers?.model && spec.drivers.model.predictors.length > 0 && (
            <Section title="Scenario simulator" subtitle="Drag the factors to project the outcome, or set a target and see which levers reach it - modelled, not guaranteed." exclude>
              <ScenarioCard drivers={spec.drivers} profiles={spec.profiles} />
            </Section>
          )}

          {spec.cohorts && spec.cohorts.cohorts.length > 1 && (
            <Section title="Cohort retention" subtitle="How well each cohort sticks around over time - the heartbeat of recurring-revenue data.">
              <CohortCard cohorts={spec.cohorts} />
            </Section>
          )}

          {spec.relationships && spec.relationships.columns.length >= 2 && (
            <Section title="How your numbers relate" subtitle="A correlation heatmap of every numeric pair - click any cell to drill into the scatter, strength, and significance.">
              <RelationshipCard relationships={spec.relationships} table={table} profiles={spec.profiles} />
            </Section>
          )}

          {spec.charts.length > 0 && (
            <Section title="Automatic charts" subtitle="The engine picked these from your data shape.">
              <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
                {spec.charts.map((c) => (
                  <Chart key={c.id} spec={c} />
                ))}
              </div>
            </Section>
          )}
        </Panel>
      )}

      {/* ── Data quality ───────────────────────────────────────────────────── */}
      <Panel id="quality" active={active}>
        {spec.quality && (
          <Section title="Data quality" subtitle="An at-a-glance health check - what's solid and what to fix before you trust the numbers.">
            <QualityCard quality={spec.quality} />
          </Section>
        )}

        <Section title="Cleaning &amp; normalization" subtitle="The unglamorous core that makes everything else trustworthy.">
          <CleaningReport report={spec.cleaning} />
        </Section>

        <Section title="How this was computed" subtitle="Methods, assumptions, and limitations - plus a reproducible recipe.">
          <MethodologyCard spec={spec} />
        </Section>
      </Panel>

      {/* ── Explore (interactive; only with the raw table) ─────────────────── */}
      {has.explore && table && (
        <Panel id="explore" active={active}>
          <Section title="Ask your data" subtitle={`${domainFocus(spec.domain.domain)} Ask in plain English - grounded in your real numbers.`} exclude>
            <QueryBox
              table={table}
              profiles={spec.profiles}
              domain={spec.domain.domain}
              analysis={{
                // Prefer the AI conclusions' action plan (it read the Python KPIs + charts); fall back to
                // the templated actions. Details are included so advisory answers carry the rationale.
                actions: conclusions?.actions?.length
                  ? conclusions.actions.slice(0, 5).map((a) => ({ title: a.title, impact: "", detail: a.detail?.slice(0, 240) }))
                  : spec.actions?.slice(0, 5).map((a) => ({ title: a.title, impact: a.impact, detail: a.detail?.slice(0, 240) })),
                bottomLine: conclusions?.bottomLine,
                // The "About this data" story isn't shown anymore - it rides here as background so the AI
                // understands what the dataset is when answering questions.
                datasetDescription: spec.story ? `${spec.story.industry ? spec.story.industry + ". " : ""}${spec.story.summary}` : undefined,
                findings: (conclusions?.conclusions?.length
                  ? conclusions.conclusions
                  : spec.insights.filter((i) => i.kind !== "summary").map((i) => i.text)
                )
                  .slice(0, 6)
                  .map((f) => f.slice(0, 240)),
                drivers: spec.drivers
                  ? { target: spec.drivers.target, r2Pct: Math.round(spec.drivers.r2 * 100), factors: spec.drivers.drivers.slice(0, 4).map((d) => ({ name: d.name, beta: Math.round(d.beta * 100) / 100, significant: d.significant })) }
                  : undefined,
                trends: spec.timeAnalysis?.slice(0, 3).map((t) => ({ metric: t.metric, changePct: t.changePct ?? null })),
              }}
            />
          </Section>

          <Section title="Build your own" subtitle="Ask for any chart you want - in plain English or by picking columns." exclude>
            <ChartBuilder table={table} profiles={spec.profiles} />
          </Section>

          <Section title="Browse the data" subtitle="Search, sort, and page through every row." exclude>
            <DataTable table={table} profiles={spec.profiles} />
          </Section>
        </Panel>
      )}
    </div>
  );
}

/**
 * A tab panel. Always mounted and laid out (preserves interactive state, and lets ECharts initialise at
 * the correct width even while off-screen) - inactive panels are clipped to zero height and marked
 * `inert` rather than `display:none`, which would force charts to 0×0. The export step un-clips them so
 * the static report includes every section.
 */
function Panel({ id, active, children }: { id: TabId; active: TabId; children: ReactNode }) {
  const inactive = active !== id;
  return (
    <div
      data-tab-panel
      data-active={inactive ? "false" : "true"}
      inert={inactive}
      aria-hidden={inactive || undefined}
      className={inactive ? "pointer-events-none h-0 overflow-hidden opacity-0" : "space-y-8 stagger-children"}
    >
      {children}
    </div>
  );
}

function Section({
  title,
  subtitle,
  badge,
  exclude,
  children,
}: {
  title: string;
  subtitle?: string;
  badge?: React.ReactNode;
  exclude?: boolean;
  children: React.ReactNode;
}) {
  // `exclude` marks interactive sections (Ask / Build / Browse / Scenario) so PNG & PDF export skip them.
  return (
    <section {...(exclude ? { "data-export-exclude": "" } : {})}>
      <div className="mb-3">
        <div className="flex items-center gap-2">
          <span className="section-accent" aria-hidden />
          <h2 className="font-display text-[17px] font-semibold text-slate-100">{title}</h2>
          {badge}
        </div>
        {subtitle && <p className="ml-7 text-xs text-slate-400">{subtitle}</p>}
      </div>
      {children}
    </section>
  );
}
