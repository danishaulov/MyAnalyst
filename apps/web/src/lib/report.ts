import type { DashboardSpec } from "./types";
import { cadenceNoun } from "./timeseries";

// Executive summary: a few plain-language paragraphs that synthesize the whole analysis into the kind
// of opening a human analyst writes - what the data is and how trustworthy, the headline numbers, the
// key movement and its driver, and the standout finding. Grounded entirely in the computed spec (no
// fabrication). Rendered at the top of the dashboard, so it leads the exported PDF/PNG report too.

const pct = (n: number): string => `${n >= 0 ? "+" : ""}${(n * 100).toFixed(1)}%`;

export function buildExecutiveSummary(spec: DashboardSpec): string[] {
  const paras: string[] = [];
  const cols = spec.profiles.length;

  // 1. What it is + how trustworthy.
  const domainLabel = spec.domain.domain.replace(/-/g, " ");
  const quality = spec.quality ? ` Data quality grades ${spec.quality.grade} (${spec.quality.score}/100).` : "";
  paras.push(`This ${domainLabel} dataset has ${spec.rowCount.toLocaleString()} rows across ${cols} columns.${quality}`);
  if (spec.computeScope?.sampled) {
    paras.push(`Server KPIs and charts use a sample of ${spec.computeScope.analyzedRows.toLocaleString()} out of ${spec.computeScope.sourceRows.toLocaleString()} rows. Their totals describe the sample only.`);
  }

  // 2. Headline numbers.
  const kpis = spec.kpis.slice(0, 3);
  if (kpis.length) {
    const parts = kpis.map((k) => {
      const val = typeof k.value === "number" ? k.value.toLocaleString() : k.value;
      const unit = k.unit ? ` ${k.unit}` : "";
      const trend = typeof k.trend === "number" ? ` (${pct(k.trend)})` : "";
      return `${k.name} is ${val}${unit}${trend}`;
    });
    paras.push(`Key numbers - ${parts.join("; ")}.`);
  }

  // 3. The key movement + its driver.
  const moves: string[] = [];
  const ta = spec.timeAnalysis?.[0];
  if (ta && ta.changePct !== undefined) {
    moves.push(`${ta.metric} ${ta.changePct >= 0 ? "rose" : "fell"} ${Math.abs(ta.changePct * 100).toFixed(1)}% in the latest ${cadenceNoun(ta.cadence)}`);
  }
  if (spec.drivers?.drivers?.length) {
    const ranked = [...spec.drivers.drivers].sort((a, b) => Math.abs(b.beta) - Math.abs(a.beta));
    const top = ranked.find((d) => d.significant) ?? ranked[0];
    if (top) moves.push(`${top.name} is the strongest driver of ${spec.drivers.target}`);
  }
  if (moves.length) paras.push(`${moves.join(", and ")}.`);

  // 4. The standout finding(s).
  const finding: string[] = [];
  if (spec.insights[0]?.text) finding.push(spec.insights[0].text);
  if (spec.segmentation && spec.segmentation.segments.length > 1) {
    finding.push(`The data splits into ${spec.segmentation.segments.length} natural segments (e.g. ${spec.segmentation.segments[0].label.toLowerCase()}).`);
  }
  if (spec.concentration?.length) {
    const c = spec.concentration[0];
    finding.push(`${c.metricIsCount ? "Volume" : c.metric} is concentrated - the top ${c.paretoCount} ${c.dimension}${c.paretoCount === 1 ? "" : "s"} hold ${Math.round(c.paretoShare * 100)}% of it.`);
  }
  if (spec.rfm) {
    const champ = spec.rfm.segments.find((s) => s.key === "champions");
    if (champ) finding.push(`RFM puts ${champ.size} ${spec.rfm.entity}s in the Champions tier (${Math.round(champ.monetaryShare * 100)}% of revenue).`);
  }
  if (spec.anomalies?.length) {
    const total = spec.anomalies.reduce((s, a) => s + a.count, 0);
    finding.push(`${total.toLocaleString()} unusual value${total > 1 ? "s were" : " was"} flagged for review.`);
  }
  if (finding.length) paras.push(finding.join(" "));

  return paras.filter((p) => p.trim());
}
