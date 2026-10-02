import type { ChartSpec, ChartType, ColumnProfile, Table } from "./types";
import { numericColumn } from "./profile";
import { maxOf, minOf, pearson, isRedundantCorrelation, zOutliers } from "./stats";
import { welchTTest, multipleRegression } from "./inference";
import { analyzeTimeSeries, cadenceNoun } from "./timeseries";
import { aggregateCount, buildChart, buildComparisonChart, buildCrossTabChart, type ChartRequest } from "./charts";
import { parseChartRequest } from "./nl-chart";
import { sortByTime } from "./kpi";
import { currencySymbol, detectCurrency, setActiveCurrency } from "./currency";
import { concentrationFor } from "./concentration";
import { analyzeRfm } from "./rfm";
import { activeLlmConfig } from "./llm-settings";
import { verifyAnswerGrounding, type GroundingResult } from "./grounding";

// ── AI-enhanced answers ────────────────────────────────────────────────────────
// When the optional LLM is enabled (NEXT_PUBLIC_LLM_ENABLED=1), `answerQuestionAI` keeps the exact
// numbers from the deterministic heuristic below but lets the model narrate a thorough, professional
// analyst answer grounded in pre-computed aggregates. Privacy: only aggregates/metadata cross the wire
// (the same boundary as /api/insights) - never raw rows. Any failure falls back to the heuristic answer.

export interface RichAnswer extends QueryAnswer {
  /** Which narrator produced the prose. */
  source: "llm" | "heuristic";
  /** Suggested next questions (LLM only). */
  followups?: string[];
  /** Numeric grounding check of an LLM answer (W3.6) - present only when the LLM narrated and stated at
   *  least one salient number. Lets the UI show a "grounded in your data" / "unverified figure" signal. */
  grounding?: GroundingResult;
}

const CATEGORY_HINT = /(reason|category|type|status|segment|group|class|gender|channel|source|outcome|result|stage|priority|label|tag|product|region|country|state|city|department)/i;

// "What share of the total is this slice?" - triggers the share/percentage-of-total branch. Kept
// distinct from "percentile" (handled separately) so the two never collide.
const SHARE_SIGNAL = /\bwhat\s+(?:%|percent|percentage|fraction|proportion|share)\b|\b(?:percent|percentage|proportion|fraction|share)\s+of\b|\bhow much of\b/;
// Concentration / Pareto phrasing: "80–20", "how concentrated", "the vital few", "top N drive most of …".
const CONCENTRATION_SIGNAL = /(concentrat|pareto|\b80\s*[\/\- ]\s*20\b|vital few|lion'?s share|how (?:concentrated|spread out)|top \w+\b[^?]*\b(?:account|make up|drive|driving|represent|come from|comes from|generate)|\b(?:drive|drives|account for|accounts for|generate|generates|make up|makes up|represent|represents)\s+(?:most|the most|the bulk|the majority)\b|(?:most|majority|bulk) of [a-z %]+ (?:come|comes) from)/;
// RFM / customer-value phrasing: "best customers", "who's at risk", "customer segments", "rfm".
const RFM_SIGNAL = /\brfm\b|\b(?:best|top|most valuable|loyal|at[- ]?risk|churn(?:ing|ed)?|dormant|lapsed|new) customers\b|customer (?:segments?|value|tiers?)|who are my .*customers|champions/;

// "Ask your data" - a heuristic natural-language Q&A engine. No LLM, no key. It maps plain-English
// questions to exact computations over the local data (aggregates, group-bys, rankings, correlation,
// trends) and answers with the real numbers, optionally attaching a chart.

export interface QueryAnswer {
  ok: boolean;
  answer: string;
  chart?: ChartSpec;
  /** Plain-language account of HOW the number was computed - the rows, the aggregation, the filter.
   *  Surfaced as a collapsible "How I computed this" so users can trust (and audit) every figure. */
  method?: string;
}

type Agg = "sum" | "mean" | "max" | "min" | "median";

const AGG_WORDS: { agg: Agg; re: RegExp; label: string }[] = [
  // Median is listed first so "median X" wins outright (it's a distinct middle-value stat, not a mean).
  { agg: "median", re: /\bmedian\b/, label: "median" },
  { agg: "sum", re: /\b(total|sum|combined|altogether)\b/, label: "total" },
  { agg: "mean", re: /\b(average|avg|mean|typical)\b/, label: "average" },
  { agg: "max", re: /\b(max|maximum|highest|largest|biggest|most|peak)\b/, label: "maximum" },
  { agg: "min", re: /\b(min|minimum|lowest|smallest|least)\b/, label: "minimum" },
];

/** Human label for an aggregator - used in answer prose so a median is never mislabelled a "total". */
function labelForAgg(agg: Agg): string {
  return agg === "mean" ? "average" : agg === "sum" ? "total" : agg; // median/max/min read as their own name
}

/** Percentile via linear interpolation between order statistics (p in 0..100). */
function percentile(values: number[], p: number): number {
  const xs = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (xs.length === 0) return NaN;
  if (xs.length === 1) return xs[0];
  const idx = (Math.min(100, Math.max(0, p)) / 100) * (xs.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  return lo === hi ? xs[lo] : xs[lo] + (idx - lo) * (xs[hi] - xs[lo]);
}

/** "1st/2nd/3rd/Nth" ordinal for a whole number. */
function ordinal(n: number): string {
  const v = n % 100;
  const suffix = v >= 11 && v <= 13 ? "th" : ["th", "st", "nd", "rd"][n % 10] ?? "th";
  return `${n}${suffix}`;
}

/** Loose token match so a question word like "intense" resolves the column "Intensity", "calorie"
 *  resolves "Calories", "duration" resolves "Duration (min)", etc. */
/** Light suffix stripper: saved/saving/savings → sav, prices/priced/pricing → pric. Lets inflected
 *  question words find their column ("money we saved" → Cost_Savings_USD). */
function stem(w: string): string {
  const s = w.replace(/(ings|ing|ied|ies|ed|es|s)$/, "");
  return s.length >= 3 ? s : w;
}

function stemMatch(a: string, b: string): boolean {
  if (a === b) return true;
  if (a.length >= 4 && b.startsWith(a)) return true;
  if (b.length >= 4 && a.startsWith(b)) return true;
  if (stem(a) === stem(b)) return true; // saved ↔ savings, priced ↔ prices
  let p = 0;
  const n = Math.min(a.length, b.length);
  while (p < n && a[p] === b[p]) p++;
  return p >= 5; // share a long prefix (intens·ity ↔ intens·e)
}

const tokenize = (s: string): string[] => s.toLowerCase().split(/[^a-z0-9]+/).filter((w) => w.length >= 3);

/** Does any token of a column's name loosely match the given word? */
function columnMatchesWord(word: string, col: ColumnProfile): boolean {
  const cl = col.name.toLowerCase();
  if (cl.includes(word) || word.includes(cl)) return true;
  return tokenize(cl).some((ct) => stemMatch(ct, word));
}

/** Columns referenced in the text - by exact name, else by a loose (stemmed) word match. Ordered
 *  by where they appear so "spend vs revenue" keeps spend first. */
function resolveCols(text: string, cols: ColumnProfile[]): ColumnProfile[] {
  const lower = text.toLowerCase();
  const words = tokenize(lower);
  const scored: { c: ColumnProfile; pos: number }[] = [];
  for (const c of cols) {
    let pos = lower.indexOf(c.name.toLowerCase()); // exact substring wins
    if (pos < 0) {
      for (const w of words) {
        if (columnMatchesWord(w, c)) {
          pos = lower.indexOf(w);
          break;
        }
      }
    }
    if (pos >= 0) scored.push({ c, pos });
  }
  return scored.sort((a, b) => a.pos - b.pos).map((s) => s.c);
}

// Superlative cues, used to tell "rank a group" ("most intense workout") from "the overall extreme"
// ("maximum duration"). The distinguishing signal is a leftover subject noun or an explicit dimension.
const SUPER_HI = /\b(most|highest|largest|biggest|greatest|strongest|longest|top|best|maximum|max|peak)\b/;
const SUPER_LO = /\b(least|lowest|smallest|weakest|shortest|fewest|bottom|worst|minimum|min)\b/;
const RATE_LIKE = /(intensity|rating|score|rate|ratio|index|percent|pct|average|avg|age|temperature|temp|speed|level|satisfaction|nps|price|cost|efficiency|density)/i;
const RANK_STOP = new Set([
  "what", "whats", "which", "the", "is", "are", "most", "least", "highest", "lowest", "largest", "smallest",
  "biggest", "best", "worst", "top", "bottom", "greatest", "strongest", "weakest", "longest", "shortest",
  "maximum", "minimum", "max", "min", "peak", "average", "avg", "mean", "typical", "total", "sum", "combined",
  "overall", "aggregate", "for", "per", "each", "and", "value", "values", "how", "much", "many", "that",
  "this", "with", "does", "has", "have", "across", "over", "whole", "entire", "dataset", "data", "there",
  "get", "show", "give", "tell", "any", "all",
]);

/** Pick sum vs mean for a ranking: averages for rate/score-like metrics (intensity, price, rating),
 *  sums for additive quantities (revenue, units) - unless the question says otherwise. */
function chooseAgg(metric: ColumnProfile, lower: string): Agg {
  if (/\bmedian\b/.test(lower)) return "median";
  if (/\b(average|avg|mean|typical|per)\b/.test(lower)) return "mean";
  if (/\b(total|sum|combined|overall|aggregate)\b/.test(lower)) return "sum";
  return RATE_LIKE.test(metric.name) ? "mean" : "sum";
}

function fmt(n: number, profile?: ColumnProfile): string {
  if (!Number.isFinite(n)) return "-";
  if (profile?.type === "currency")
    return currencySymbol() + new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 }).format(n);
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 }).format(n);
}

function aggregate(values: number[], agg: Agg): number {
  const xs = values.filter(Number.isFinite);
  if (xs.length === 0) return NaN;
  switch (agg) {
    case "sum": return xs.reduce((a, b) => a + b, 0);
    case "mean": return xs.reduce((a, b) => a + b, 0) / xs.length;
    case "max": return maxOf(xs);
    case "min": return minOf(xs);
    case "median": {
      const s = [...xs].sort((a, b) => a - b);
      const mid = Math.floor(s.length / 2);
      return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
    }
  }
}

/** Metric values for the rows matching a predicate (finite only) - the raw arrays a t-test needs. */
function valuesWhere(table: Table, metric: string, pred: (row: Record<string, unknown>) => boolean): number[] {
  const vals = numericColumn(table, metric);
  const out: number[] = [];
  table.rows.forEach((r, i) => {
    if (pred(r) && Number.isFinite(vals[i])) out.push(vals[i]);
  });
  return out;
}

/** Format a p-value for prose: "< 0.001" for tiny values, else 3 decimals. */
function fmtP(p: number): string {
  return p < 0.001 ? "< 0.001" : `= ${p.toFixed(3)}`;
}

/** Group a metric by a dimension under the given aggregator → sorted [key, value] pairs (desc). */
function groupBy(table: Table, dim: string, metric: string, agg: Agg): [string, number][] {
  const buckets = new Map<string, number[]>();
  const vals = numericColumn(table, metric);
  table.rows.forEach((r, i) => {
    if (!Number.isFinite(vals[i])) return;
    const key = String(r[dim] ?? "-");
    if (!buckets.has(key)) buckets.set(key, []);
    buckets.get(key)!.push(vals[i]);
  });
  return [...buckets.entries()]
    .map(([k, v]) => [k, aggregate(v, agg)] as [string, number])
    .sort((a, b) => b[1] - a[1]);
}

interface CrossTabResult {
  cells: { a: string; b: string; value: number }[]; // every (dimA, dimB) cell, sorted desc by value
  aCount: number;
  bCount: number;
  xCats: string[]; // top dimA categories (for the x axis)
  seriesNames: string[]; // top dimB values (stacked series)
  matrix: number[][]; // matrix[seriesIndex][xIndex]
}

/** Cross-tabulate a metric (or row counts when `metric` is omitted) by two dimensions. Returns the
 *  ranked cells plus a capped, chart-ready matrix (top categories × top series). */
function crossTab(table: Table, dimA: string, dimB: string, metric: string | undefined, agg: Agg): CrossTabResult {
  const SEP = " ";
  const buckets = new Map<string, number[]>();
  const vals = metric ? numericColumn(table, metric) : null;
  const aSet = new Set<string>();
  const bSet = new Set<string>();
  table.rows.forEach((r, i) => {
    if (metric && !Number.isFinite(vals![i])) return;
    const a = String(r[dimA] ?? "-");
    const b = String(r[dimB] ?? "-");
    const key = a + SEP + b;
    if (!buckets.has(key)) buckets.set(key, []);
    buckets.get(key)!.push(metric ? vals![i] : 1);
    aSet.add(a);
    bSet.add(b);
  });
  const cells = [...buckets.entries()]
    .map(([k, arr]) => {
      const [a, b] = k.split(SEP);
      return { a, b, value: metric ? aggregate(arr, agg) : arr.length };
    })
    .filter((c) => Number.isFinite(c.value))
    .sort((x, y) => y.value - x.value);

  const aTotals = new Map<string, number>();
  const bTotals = new Map<string, number>();
  for (const c of cells) {
    aTotals.set(c.a, (aTotals.get(c.a) ?? 0) + c.value);
    bTotals.set(c.b, (bTotals.get(c.b) ?? 0) + c.value);
  }
  const topKeys = (m: Map<string, number>, n: number) => [...m.entries()].sort((p, q) => q[1] - p[1]).slice(0, n).map((e) => e[0]);
  const xCats = topKeys(aTotals, 8);
  const seriesNames = topKeys(bTotals, 6);
  const cellMap = new Map(cells.map((c) => [c.a + SEP + c.b, c.value]));
  const matrix = seriesNames.map((s) => xCats.map((x) => cellMap.get(x + SEP + s) ?? 0));
  return { cells, aCount: aSet.size, bCount: bSet.size, xCats, seriesNames, matrix };
}

/** "N of M rows (scope)" or "M rows" - the row basis quoted in every `method` string. */
function rowsNote(view: Table, table: Table, filter?: DataFilter): string {
  return filter ? `${view.rowCount.toLocaleString()} of ${table.rowCount.toLocaleString()} rows ${filter.phrase}` : `${table.rowCount.toLocaleString()} rows`;
}

export function answerQuestion(question: string, table: Table, profiles: ColumnProfile[]): QueryAnswer {
  setActiveCurrency(detectCurrency(table, profiles));
  const text = question.trim();
  if (!text) return { ok: false, answer: 'Ask something like "total revenue by region" or "correlation between spend and revenue".' };
  const lower = text.toLowerCase();

  const metrics = profiles.filter((p) => p.role === "metric" && p.numeric);
  const dims = profiles.filter((p) => p.role === "dimension");
  const time = profiles.find((p) => p.role === "time");

  const mMetrics = resolveCols(text, metrics);
  const mDims = resolveCols(text, dims);

  // "Most <quality> <thing>" intent: a superlative plus a leftover subject noun ("most intense workout")
  // or an explicit dimension means "rank a group", not "give the overall extreme". Subject nouns are the
  // question words that don't refer to any column and aren't filler - their presence flips us to ranking.
  const hasSuper = SUPER_HI.test(lower) || SUPER_LO.test(lower);
  const subjectNouns = tokenize(lower).filter((w) => !RANK_STOP.has(w) && !profiles.some((p) => columnMatchesWord(w, p)));
  const rankingIntent =
    hasSuper && (mDims.length > 0 || /\b(which|each|per|category|type|kind|group|segment)\b/.test(lower) || subjectNouns.length > 0);

  // W4.1 Significance testing: "is the difference in revenue between North and South significant / real?"
  // Runs Welch's t-test on the two slices and gives a plain verdict. Detected before the plain comparison
  // so a "significant?" question gets the test, not just the gap.
  if (/(significan|statistic|by chance|real difference|reliable|p-?value|meaningful difference)/.test(lower)) {
    const cmp = detectComparison(text, table, profiles);
    if (cmp) {
      const lv = valuesWhere(table, cmp.metric.name, cmp.left.predicate);
      const rv = valuesWhere(table, cmp.metric.name, cmp.right.predicate);
      const tt = welchTTest(lv, rv);
      if (tt) {
        const m1 = aggregate(lv, "mean");
        const m2 = aggregate(rv, "mean");
        const verdict = tt.significant
          ? `This difference **is** statistically significant (p ${fmtP(tt.p)}), so it's unlikely to be random chance.`
          : `This difference is **not** statistically significant (p ${fmtP(tt.p)}), so it could be random variation - treat it as suggestive, not proven.`;
        return {
          ok: true,
          answer: `Average ${cmp.metric.name}: "${cmp.left.label}" ${fmt(m1, cmp.metric)} vs "${cmp.right.label}" ${fmt(m2, cmp.metric)} - a gap of ${fmt(Math.abs(m1 - m2), cmp.metric)}. ${verdict} (Based on ${tt.n1} vs ${tt.n2} records.)`,
          chart: buildComparisonChart(cmp.metric.name, "mean", [[cmp.left.label, m1], [cmp.right.label, m2]]),
          method: `Welch's two-sample t-test on ${cmp.metric.name} for "${cmp.left.label}" (n=${tt.n1}) vs "${cmp.right.label}" (n=${tt.n2}): t=${tt.t.toFixed(2)}, df=${tt.df.toFixed(0)}, two-sided p ${fmtP(tt.p)}.`,
        };
      }
    }
  }

  // Comparison questions ("North vs South revenue", "how does 2023 compare to 2022") - two slices of
  // the same dimension or two periods, answered with the gap, ratio and % difference plus a paired bar.
  // Detected before the single filter so a "vs" question isn't collapsed to one slice.
  const comparison = detectComparison(text, table, profiles);
  if (comparison) return answerComparison(comparison, table, profiles);

  // Filtered & conditional questions: "total revenue in 2023", "average order value for the North
  // region", "how many orders where revenue is over 100". We answer over a filtered VIEW of the rows
  // and weave the scope into the prose. `view` is the (possibly) filtered table; `scope` is the natural
  // clause appended to answers. Charts are built from `view`, so they reflect the filter too.
  const filter = detectFilter(text, table, profiles);
  const view = filter ? applyFilter(table, filter) : table;
  if (filter && view.rowCount === 0) {
    return { ok: false, answer: `No records match ${filter.label}. Try a different value or range.` };
  }
  const scope = filter ? ` ${filter.phrase}` : "";

  // 0. Count distinct values of a column: "how many unique products", "number of distinct regions".
  // Only fires when a real column is named (so a plain "how many records" still falls to the row count).
  const wantsDistinct = /\b(distinct|unique)\b/.test(lower) || (/\bdifferent\b/.test(lower) && /\b(how many|number of|count)\b/.test(lower));
  if (wantsDistinct) {
    const target = resolveCols(text, profiles)[0];
    if (target) {
      const n = distinctCount(view, target.name);
      return {
        ok: true,
        answer: `There ${n === 1 ? "is" : "are"} ${n.toLocaleString()} distinct ${target.name} value${n === 1 ? "" : "s"}${scope}.`,
        method: `Counted unique non-empty values of ${target.name} across ${rowsNote(view, table, filter)}.`,
      };
    }
  }

  // 0.3 Percentile / quartile of a metric: "90th percentile of price", "top quartile of revenue".
  if (/\b(percentile|quartile|quantile)\b/.test(lower) || /\bp(\d{1,3})\b/.test(lower)) {
    const m = mMetrics[0] ?? metrics[0];
    if (m) {
      let p = 75;
      const pm =
        lower.match(/(\d{1,3})\s*(?:st|nd|rd|th)?\s*(?:percentile|quantile)/) ||
        lower.match(/\bp(\d{1,3})\b/);
      if (pm) p = Math.min(100, Math.max(0, Number(pm[1])));
      else if (/\bquartile\b/.test(lower)) p = /\b(bottom|lower|lowest|first|1st)\b/.test(lower) ? 25 : 75;
      const v = percentile(numericColumn(view, m.name), p);
      if (Number.isFinite(v)) {
        return {
          ok: true,
          answer: `The ${ordinal(p)} percentile of ${m.name}${scope} is ${fmt(v, m)} - ${p}% of values fall at or below it.`,
          method: `${ordinal(p)} percentile of ${m.name} across ${rowsNote(view, table, filter)} (linear interpolation between order statistics).`,
        };
      }
    }
  }

  // 0.5 Share / percentage of total for a slice: "what % of revenue comes from North", "what
  // percentage of orders are cancelled". Needs a slice (the filter); a named metric → metric share
  // (slice total ÷ grand total), otherwise a row-count share. The slice's own column never doubles as
  // the metric, so "% of revenue where revenue > 100" stays a count share, not a self-referential ratio.
  if (SHARE_SIGNAL.test(lower) && filter) {
    const shareMetric = mMetrics.find((m) => m.name !== filter.column);
    if (shareMetric) {
      const num = aggregate(numericColumn(view, shareMetric.name), "sum");
      const den = aggregate(numericColumn(table, shareMetric.name), "sum");
      if (Number.isFinite(num) && Number.isFinite(den) && den !== 0) {
        const pct = (num / den) * 100;
        let chart: ChartSpec | undefined;
        try {
          chart = buildChart(table, profiles, { type: "pie", x: filter.column, y: [shareMetric.name], aggregate: true });
        } catch {
          chart = undefined;
        }
        return {
          ok: true,
          answer: `${filter.label} accounts for ${pct.toFixed(1)}% of total ${shareMetric.name} - ${fmt(num, shareMetric)} of ${fmt(den, shareMetric)}.`,
          chart,
          method: `Divided ${shareMetric.name} ${filter.phrase} (${fmt(num, shareMetric)}) by the grand total (${fmt(den, shareMetric)}) across ${rowsNote(view, table, filter)}.`,
        };
      }
    }
    if (table.rowCount > 0) {
      const pct = (view.rowCount / table.rowCount) * 100;
      return {
        ok: true,
        answer: `Records ${filter.phrase} are ${pct.toFixed(1)}% of all rows - ${view.rowCount.toLocaleString()} of ${table.rowCount.toLocaleString()}.`,
        method: `Counted rows ${filter.phrase} (${view.rowCount.toLocaleString()}) ÷ all ${table.rowCount.toLocaleString()} rows.`,
      };
    }
  }

  // 0.6 Concentration / Pareto: "how concentrated is revenue across customers?", "do the top products
  // drive most of sales?", "what's the 80–20 on spend by channel?". Aggregates the measure by the named
  // category and reports the vital-few share + Gini. Exact and groundable; reuses the concentration lib.
  if (CONCENTRATION_SIGNAL.test(lower)) {
    const groupCols = profiles.filter((p) => p.role === "dimension" || p.role === "identifier");
    const dim = resolveCols(text, groupCols)[0] ?? mDims[0];
    if (dim) {
      const metric = mMetrics.find((m) => m.name !== dim.name);
      const conc = concentrationFor(view, dim.name, metric ? { name: metric.name, values: numericColumn(view, metric.name) } : null);
      if (conc) {
        const measure = conc.metricIsCount ? "the rows" : conc.metric;
        const biggest = conc.segments[0];
        const lvl =
          conc.level === "high"
            ? "That's heavy concentration - a few names carry the whole figure, which is a risk worth watching."
            : conc.level === "moderate"
              ? "That's moderate concentration."
              : "It's fairly evenly spread.";
        return {
          ok: true,
          answer: `The top ${conc.paretoCount} of ${conc.distinct} ${dim.name}${conc.paretoCount === 1 ? "" : "s"} (${(conc.paretoPctOfCategories * 100).toFixed(0)}% of them) account for ${(conc.paretoShare * 100).toFixed(0)}% of ${measure}${scope}. The single largest, "${biggest.name}", is ${(biggest.share * 100).toFixed(0)}% on its own. ${lvl} (Gini ${conc.gini.toFixed(2)}.)`,
          chart: metric
            ? buildChart(view, profiles, { type: "bar", x: dim.name, y: [metric.name], aggregate: true })
            : buildChart(view, profiles, { type: "bar", x: dim.name, y: [], count: true }),
          method: `Aggregated ${measure} by ${dim.name} across ${rowsNote(view, table, filter)}, sorted descending, then measured the cumulative (Pareto) share and the Gini coefficient (${conc.gini.toFixed(2)}).`,
        };
      }
    }
  }

  // 0.65 RFM / customer value: "who are my best customers?", "which customers are at risk?", "show me
  // the customer segments". Scores entities on Recency/Frequency/Monetary and reports the segment mix.
  // Needs transaction-shaped data (id + date + value); otherwise says so plainly rather than guessing.
  if (RFM_SIGNAL.test(lower)) {
    const rfm = analyzeRfm(view, profiles);
    if (rfm) {
      const champ = rfm.segments.find((s) => s.key === "champions");
      const risk = rfm.segments.find((s) => s.key === "at-risk");
      const top = rfm.segments[0];
      const parts: string[] = [
        `I scored your ${rfm.customers.toLocaleString()} ${rfm.entity}s on recency, frequency, and spend (${rfm.valueColumn}, as of ${rfm.asOf}).`,
      ];
      if (champ) parts.push(`Champions - recent, frequent, high-spend - are ${champ.size} ${rfm.entity}s (${champ.sharePct.toFixed(0)}%) and ${(champ.monetaryShare * 100).toFixed(0)}% of revenue.`);
      parts.push(`"${top.label}" is the most valuable segment by total spend.`);
      if (risk && risk.size > 0) parts.push(`Watch the ${risk.size} At Risk ${rfm.entity}s (${(risk.monetaryShare * 100).toFixed(0)}% of revenue) - they were valuable but have gone quiet; a win-back is the clearest opportunity.`);
      let chart: ChartSpec | undefined;
      try {
        chart = buildComparisonChart(`Revenue by RFM segment`, "sum", rfm.segments.map((s) => [s.label, s.totalMonetary] as [string, number]));
      } catch {
        chart = undefined;
      }
      return {
        ok: true,
        answer: parts.join(" "),
        chart,
        method: `RFM: scored each ${rfm.entity} on recency (days since last ${rfm.dateColumn}), frequency (transaction count), and monetary (${rfm.valueColumn} sum) into 1–5 quintiles across ${rowsNote(view, table, filter)}, then mapped scores to named value segments.`,
      };
    }
    // Intent was clearly RFM but the data isn't transaction-shaped - say why instead of misfiring.
    return {
      ok: false,
      answer: "RFM customer segmentation needs three things in the data: a customer/account id, a date, and a spend (or value) column. I couldn't find all three here, so I can't score recency, frequency, and monetary value.",
    };
  }

  // 0.7 Two-dimension breakdown: "revenue by region and product", "orders by status and channel".
  // Runs before the single-dimension ranking/aggregate so two named dimensions aren't collapsed to one.
  if (mDims.length >= 2 && mDims[0].name !== mDims[1].name && /\b(by|across|split|grouped|breakdown|broken down)\b/.test(lower)) {
    const [dimA, dimB] = mDims;
    const ctMetric = mMetrics[0];
    const agg = ctMetric ? chooseAgg(ctMetric, lower) : "sum";
    const ct = crossTab(view, dimA.name, dimB.name, ctMetric?.name, agg);
    if (ct.cells.length) {
      const top = ct.cells[0];
      const what = ctMetric ? `${labelForAgg(agg)} ${ctMetric.name}` : "row count";
      const valStr = ctMetric ? fmt(top.value, ctMetric) : top.value.toLocaleString();
      return {
        ok: true,
        answer: `By ${what}, the top ${dimA.name} × ${dimB.name} combination is "${top.a}" / "${top.b}" at ${valStr}${scope}. There ${ct.cells.length === 1 ? "is" : "are"} ${ct.cells.length} combination${ct.cells.length === 1 ? "" : "s"} across ${ct.aCount} ${dimA.name} × ${ct.bCount} ${dimB.name}.`,
        chart: buildCrossTabChart(`${cap(what)} by ${dimA.name} × ${dimB.name}`, ct.xCats, ct.seriesNames, ct.matrix, ctMetric ? ctMetric.name : "count"),
        method: `Cross-tabbed ${ctMetric ? ctMetric.name : "rows"} by ${dimA.name} and ${dimB.name} (${ctMetric ? labelForAgg(agg) : "count"}) across ${rowsNote(view, table, filter)}, then ranked all ${ct.cells.length} cells.`,
      };
    }
  }

  // W4.2 Drivers: "what drives revenue?", "what predicts price?" - multiple regression of the other
  // numeric columns on the target; report the strongest standardized driver, significance, and R².
  if (/\b(drives?|drivers?|predicts?|prediction|influences?|affects?|explains?|depends? on|biggest factor|key factor|what factor|drivers of)\b/.test(lower) && metrics.length >= 2) {
    const target = mMetrics[0] ?? metrics[0];
    const predictors = metrics.filter((m) => m.name !== target.name);
    if (predictors.length >= 1) {
      const X = predictors.map((p) => numericColumn(view, p.name));
      const y = numericColumn(view, target.name);
      const reg = multipleRegression(X, y, predictors.map((p) => p.name));
      if (reg && reg.coefficients.length) {
        const ranked = [...reg.coefficients].sort((a, b) => Math.abs(b.beta) - Math.abs(a.beta));
        const top = ranked[0];
        const sig = ranked.filter((c) => c.significant);
        const sigNote = sig.length
          ? `Statistically significant driver${sig.length > 1 ? "s" : ""}: ${sig.slice(0, 3).map((c) => c.name).join(", ")}.`
          : `None of the factors reach statistical significance, so read this as directional.`;
        const dir = top.beta >= 0 ? "higher" : "lower";
        return {
          ok: true,
          answer: `The strongest driver of ${target.name} is ${top.name} (standardized β = ${top.beta.toFixed(2)}) - ${dir} ${top.name} goes with ${top.beta >= 0 ? "higher" : "lower"} ${target.name}, even after accounting for the other factors. Together the factors explain ${Math.round(reg.r2 * 100)}% of the variation in ${target.name} (R²)${scope}. ${sigNote}`,
          chart: buildChart(view, profiles, { type: "scatter", x: top.name, y: [target.name] }),
          method: `Multiple linear regression of ${target.name} on ${predictors.map((p) => p.name).join(", ")} across ${rowsNote(view, table, filter)}; drivers ranked by |standardized β|, R² = ${reg.r2.toFixed(2)}, model p ${fmtP(reg.fP)}.`,
        };
      }
    }
  }

  // W4.3 Outliers / anomalies: "are there outliers in revenue?", "any unusual values?"
  if (/\b(outliers?|anomal(?:y|ies|ous)|unusual|abnormal|extreme values?)\b/.test(lower)) {
    const m = mMetrics[0] ?? metrics[0];
    if (m) {
      const outs = zOutliers(numericColumn(view, m.name), 3);
      if (outs.length) {
        const top = [...outs].sort((a, b) => Math.abs(b.z) - Math.abs(a.z))[0];
        return {
          ok: true,
          answer: `${m.name} has ${outs.length} outlier${outs.length === 1 ? "" : "s"} beyond 3 standard deviations${scope}. The most extreme is ${fmt(top.value, m)} (${top.z > 0 ? "+" : ""}${top.z.toFixed(1)}σ from the mean). Extreme values like these can quietly skew averages - check they're real before trusting ${m.name} summaries.`,
          chart: buildChart(view, profiles, { type: "histogram", x: m.name, y: [] }),
          method: `Z-score outlier scan of ${m.name} across ${rowsNote(view, table, filter)} (flagging |z| > 3).`,
        };
      }
      return {
        ok: true,
        answer: `No values in ${m.name} fall beyond 3 standard deviations${scope} - the distribution looks clean of extreme outliers.`,
        method: `Z-score outlier scan of ${m.name} across ${rowsNote(view, table, filter)} (none with |z| > 3).`,
      };
    }
  }

  // W4.4 Time-grain trend: "monthly revenue trend", "revenue year over year". Buckets by detected
  // cadence and reports latest / MoM / YoY / best & worst periods. Restricted to explicit grain words so
  // generic "over time" stays with the simpler trend branch below.
  if (time && /\b(monthly|quarterly|weekly|annual(?:ly)?|month over month|year over year|by month|by quarter|by year|by week|seasonal(?:ity)?|cadence|mom|yoy|per month|per quarter|each month)\b/.test(lower)) {
    const m = mMetrics[0] ?? metrics[0];
    if (m) {
      const ts = analyzeTimeSeries(view, time.name, m.name);
      if (ts) {
        const noun = cadenceNoun(ts.cadence);
        const mom = ts.changePct != null
          ? `${ts.changePct >= 0 ? "up" : "down"} ${Math.abs(ts.changePct * 100).toFixed(1)}% vs the prior ${noun}`
          : "with no prior period to compare";
        const yoy = ts.yoyChangePct != null ? ` Year over year it's ${ts.yoyChangePct >= 0 ? "up" : "down"} ${Math.abs(ts.yoyChangePct * 100).toFixed(1)}%.` : "";
        const seas = ts.seasonality
          ? ` Seasonally, it peaks in ${ts.seasonality.peak.label} (+${Math.round((ts.seasonality.peak.index - 1) * 100)}% vs the ${ts.seasonality.unit} average) and is lowest in ${ts.seasonality.trough.label} (${Math.round((ts.seasonality.trough.index - 1) * 100)}%).`
          : "";
        return {
          ok: true,
          answer: `${m.name} is tracked ${ts.cadence}${scope}. The latest period (${ts.latest.label}) totalled ${fmt(ts.latest.value, m)}, ${mom}.${yoy}${seas} Best ${noun}: ${ts.best.label} (${fmt(ts.best.value, m)}); weakest: ${ts.worst.label} (${fmt(ts.worst.value, m)}).`,
          chart: buildChart(view, profiles, { type: "line", x: time.name, y: [m.name] }),
          method: `Bucketed ${m.name} by ${ts.cadence} period across ${rowsNote(view, table, filter)} (${ts.periods.length} periods); reported latest, period-over-period and year-over-year change.`,
        };
      }
    }
  }

  // 1. Row count.
  if (/\b(how many|number of|count of|count)\b/.test(lower) && (/\b(row|record|entr|data point|observation)/.test(lower) || filter)) {
    return {
      ok: true,
      answer: `There are ${view.rowCount.toLocaleString()} records${filter ? scope : " in the dataset"}.`,
      method: `Counted rows - ${rowsNote(view, table, filter)}.`,
    };
  }

  // 1b. Most-common / distribution of a CATEGORICAL column - e.g. "the most common reason for not buying".
  // Explicit frequency phrasing always counts; the "which <category>" shortcut only counts when no
  // metric is involved (otherwise "which region has the highest revenue" is a ranking, handled below).
  const explicitFreq = /\b(most common|commonest|most frequent|most popular|main reason|top reason|biggest reason|usual|distribution|breakdown|how often|frequency)\b/.test(lower);
  const wantsFreq = explicitFreq || (/\bwhich\b/.test(lower) && CATEGORY_HINT.test(lower) && mMetrics.length === 0);
  if (wantsFreq) {
    // Pick the category column: one named in the question, else a hinted dimension, else the first dimension.
    const col =
      mDims[0] ??
      dims.find((d) => CATEGORY_HINT.test(d.name) && CATEGORY_HINT.test(lower) && lower.includes(d.name.toLowerCase().split(/\s+/)[0])) ??
      dims.find((d) => CATEGORY_HINT.test(d.name)) ??
      dims[0];
    if (col) {
      const counts = aggregateCount(view, col.name);
      const total = counts.reduce((s, [, c]) => s + c, 0);
      if (total > 0) {
        const top = counts[0];
        const second = counts[1];
        const tail = second ? `, then "${second[0]}" (${Math.round((second[1] / total) * 100)}%)` : "";
        return {
          ok: true,
          answer: `The most common ${col.name} is "${top[0]}" - ${top[1]} of ${total} (${Math.round((top[1] / total) * 100)}%)${tail}${scope}.`,
          chart: buildChart(view, profiles, { type: "bar", x: col.name, y: [], count: true }),
          method: `Tallied how often each ${col.name} occurs across ${rowsNote(view, table, filter)} - ${counts.length} distinct values.`,
        };
      }
    }
  }

  // 2. Correlation / relationship.
  if (/correlat|relationship|related|associat|vs\.?|versus/.test(lower)) {
    // "What's most correlated with revenue?" - one metric named → find its strongest (non-redundant) correlate.
    if (mMetrics.length === 1 && metrics.length >= 2) {
      const a = mMetrics[0];
      let best: { col: ColumnProfile; r: number } | undefined;
      for (const other of metrics) {
        if (other.name === a.name) continue;
        const r = pearson(numericColumn(view, a.name), numericColumn(view, other.name));
        if (Number.isFinite(r) && !isRedundantCorrelation(a.name, other.name, r) && (!best || Math.abs(r) > Math.abs(best.r))) best = { col: other, r };
      }
      if (best) {
        const strength = Math.abs(best.r) > 0.7 ? "strongly" : Math.abs(best.r) > 0.4 ? "moderately" : "weakly";
        const dir = best.r > 0 ? "positively" : "negatively";
        return {
          ok: true,
          answer: `Among your numbers, ${a.name} is most ${strength} ${dir} related to ${best.col.name} (r = ${best.r.toFixed(2)})${scope}. ${best.r > 0 ? "They tend to rise together" : "As one rises, the other tends to fall"} - but that's association, not proof of cause.`,
          chart: buildChart(view, profiles, { type: "scatter", x: a.name, y: [best.col.name] }),
          method: `Pearson correlation of ${a.name} against each other numeric column across ${rowsNote(view, table, filter)}; reported the strongest, excluding near-duplicate columns.`,
        };
      }
    }
    const ms = mMetrics.length >= 2 ? mMetrics : metrics;
    if (ms.length >= 2) {
      const [a, b] = ms;
      const r = pearson(numericColumn(view, a.name), numericColumn(view, b.name));
      if (Number.isFinite(r)) {
        const strength = Math.abs(r) > 0.7 ? "strong" : Math.abs(r) > 0.4 ? "moderate" : "weak";
        const dir = r > 0 ? "positive" : "negative";
        return {
          ok: true,
          answer: `${a.name} and ${b.name} have a ${strength} ${dir} correlation (r = ${r.toFixed(2)})${scope}. ${
            Math.abs(r) > 0.4 ? "They tend to move together." : "The link is loose."
          }`,
          chart: buildChart(view, profiles, { type: "scatter", x: a.name, y: [b.name] }),
          method: `Pearson correlation of ${a.name} and ${b.name} across ${rowsNote(view, table, filter)}.`,
        };
      }
    }
  }

  // 3. Trend over time.
  if (time && /(trend|over time|timeline|change over|grow|growth|increase|decrease|trajectory)/.test(lower)) {
    const m = mMetrics[0] ?? metrics[0];
    if (m) {
      const order = sortByTime(view, time.name);
      const mCol = numericColumn(view, m.name);
      const series = order.map((i) => mCol[i]).filter(Number.isFinite);
      if (series.length >= 2) {
        const first = series[0];
        const last = series[series.length - 1];
        const pct = first !== 0 ? ((last - first) / Math.abs(first)) * 100 : 0;
        const dir = pct > 1 ? "increased" : pct < -1 ? "decreased" : "stayed roughly flat";
        return {
          ok: true,
          answer: `${m.name} ${dir} ${Math.abs(pct).toFixed(1)}% over the period${scope} (${fmt(first, m)} → ${fmt(last, m)}).`,
          chart: buildChart(view, profiles, { type: "line", x: time.name, y: [m.name] }),
          method: `Compared first vs last ${m.name} along ${time.name} across ${rowsNote(view, table, filter)} (${series.length} points).`,
        };
      }
    }
  }

  // 4. Ranking / superlative-by-group: "which region has the highest revenue", "most intense workout",
  //    "top products by sales". Fires on explicit ranking words OR a superlative aimed at a group.
  if (/\b(which|top|rank|ranking)\b/.test(lower) || rankingIntent) {
    const dim = mDims[0] ?? dims[0];
    const metric = mMetrics[0] ?? metrics[0];
    // Only answer when the question actually referenced a column. With neither the metric nor the
    // dimension mentioned, defaulting both is a guess dressed up as an answer ("most money we saved"
    // once came back as usage-by-region) - fall through so the AI/open-ended path handles it instead.
    if (dim && metric && (mMetrics.length > 0 || mDims.length > 0)) {
      const wantLowest = SUPER_LO.test(lower);
      const agg = chooseAgg(metric, lower);
      const ranked = groupBy(view, dim.name, metric.name, agg);
      if (ranked.length) {
        const ordered = wantLowest ? [...ranked].reverse() : ranked;
        const top = ordered[0];
        const aggLabel = labelForAgg(agg);
        return {
          ok: true,
          answer: `By ${aggLabel} ${metric.name}, ${dim.name} "${top[0]}" is ${wantLowest ? "lowest" : "highest"} at ${fmt(top[1], metric)}${scope}.`,
          chart: agg === "sum" ? buildChart(view, profiles, { type: "bar", x: dim.name, y: [metric.name], aggregate: true }) : undefined,
          method: `Took the ${aggLabel} ${metric.name} by ${dim.name} across ${rowsNote(view, table, filter)}, then ranked ${wantLowest ? "ascending" : "descending"} (${ranked.length} groups).`,
        };
      }
    }
  }

  // 5. Aggregate (optionally grouped): "total revenue", "average units by region".
  const aggMatch = AGG_WORDS.find((w) => w.re.test(lower));
  if (aggMatch) {
    const metric = mMetrics[0] ?? metrics[0];
    // Same no-guessing rule as ranking: a superlative ("most/least …") aimed at words that resolve to
    // NO column ("most impressive thing") must not silently become max-of-the-first-metric. Explicit
    // aggregations ("total", "average") keep the headline-metric default - that's a useful shorthand.
    const guessing = (aggMatch.agg === "max" || aggMatch.agg === "min") && mMetrics.length === 0 && subjectNouns.length > 0;
    if (metric && !guessing) {
      const wantsGroup = /\b(by|per|across|for each|grouped)\b/.test(lower);
      const dim = wantsGroup ? mDims[0] ?? dims[0] : undefined;
      if (dim) {
        const ranked = groupBy(view, dim.name, metric.name, aggMatch.agg);
        const top = ranked[0];
        return {
          ok: true,
          answer: `${cap(aggMatch.label)} ${metric.name} by ${dim.name}${scope}: ${dim.name} "${top[0]}" leads at ${fmt(top[1], metric)} (${ranked.length} groups).`,
          chart: aggMatch.agg === "sum" ? buildChart(view, profiles, { type: "bar", x: dim.name, y: [metric.name], aggregate: true }) : undefined,
          method: `Took the ${aggMatch.label} of ${metric.name} within each ${dim.name} across ${rowsNote(view, table, filter)} (${ranked.length} groups).`,
        };
      }
      const v = aggregate(numericColumn(view, metric.name), aggMatch.agg);
      return {
        ok: true,
        answer: `The ${aggMatch.label} of ${metric.name}${scope} is ${fmt(v, metric)}.`,
        method: `${cap(aggMatch.label)} of ${metric.name} across ${rowsNote(view, table, filter)}.`,
      };
    }
  }

  // 6. A bare metric mention → give its headline stats.
  if (mMetrics.length) {
    const m = mMetrics[0];
    const vals = numericColumn(view, m.name);
    return {
      ok: true,
      answer: `${m.name}${scope}: total ${fmt(aggregate(vals, "sum"), m)}, average ${fmt(aggregate(vals, "mean"), m)}, range ${fmt(aggregate(vals, "min"), m)}–${fmt(aggregate(vals, "max"), m)}.`,
      method: `Summary stats (total, average, min/max) of ${m.name} across ${rowsNote(view, table, filter)}.`,
    };
  }

  return {
    ok: false,
    answer:
      'I couldn\'t match that to your data. Try: "total ' +
      (metrics[0]?.name ?? "value") +
      '", "average ' + (metrics[0]?.name ?? "value") + " by " + (dims[0]?.name ?? "category") +
      '", "what actions should I take?" - or ask "what can you do?" for the full list.',
  };
}

function cap(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

// ── Filters ──────────────────────────────────────────────────────────────────
// Turn a plain-English condition in the question into a row predicate so the engine can answer
// "total revenue in 2023", "average order value for the North region", or "how many orders where
// revenue is over 100". Three kinds, tried in order: time/year, numeric comparison on a named metric,
// and a categorical value of a dimension. Everything downstream runs on the filtered VIEW, so numbers
// and charts stay consistent with the stated scope.

export interface DataFilter {
  column: string;
  /** Natural clause woven into answers, e.g. `in 2023`, `for Region "North"`, `where Revenue is over 100`. */
  phrase: string;
  /** Short label for compact UI / error messages, e.g. `2023`, `North`, `Revenue > 100`. */
  label: string;
  predicate: (row: Record<string, unknown>) => boolean;
}

/** Return a filtered copy of the table (same columns, subset of rows). */
export function applyFilter(table: Table, filter: DataFilter): Table {
  const rows = table.rows.filter(filter.predicate);
  return { ...table, rows, rowCount: rows.length };
}

// Words that can coincide with a category value but are really question grammar - never treat as a filter value.
const FILTER_STOP = new Set([
  "total", "sum", "average", "avg", "mean", "median", "count", "number", "max", "min", "maximum", "minimum",
  "highest", "lowest", "top", "bottom", "most", "least", "data", "value", "values", "record", "records", "row",
  "rows", "the", "and", "or", "of", "in", "for", "by", "per", "is", "are", "with", "where", "over", "under",
  "between", "than", "all", "each", "show",
]);

function coerceNum(v: unknown): number {
  if (typeof v === "number") return v;
  return parseFloat(String(v ?? "").replace(/[^0-9.\-]/g, ""));
}

/** Exact count of distinct non-empty values of a column (uncapped - this is the answer, not a sample). */
function distinctCount(table: Table, col: string): number {
  const seen = new Set<string>();
  for (const r of table.rows) {
    const v = r[col];
    if (v === null || v === undefined || v === "") continue;
    seen.add(String(v));
  }
  return seen.size;
}

/** Distinct non-empty values of a column (capped, for matching against the question). */
function distinctValues(table: Table, col: string, cap = 500): string[] {
  const seen = new Set<string>();
  for (const r of table.rows) {
    const v = r[col];
    if (v === null || v === undefined || v === "") continue;
    seen.add(String(v));
    if (seen.size >= cap) break;
  }
  return [...seen];
}

/** Whole-word/phrase containment (so "north" doesn't match inside "northern" and survives punctuation). */
function containsPhrase(hayLower: string, needleLower: string): boolean {
  if (needleLower.length < 2) return false;
  const esc = needleLower.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return new RegExp(`(^|[^a-z0-9])${esc}([^a-z0-9]|$)`, "i").test(hayLower);
}

const NUMRE = "(-?\\d[\\d,]*(?:\\.\\d+)?)";
const parseNum = (s: string): number => parseFloat(s.replace(/,/g, ""));

export function detectFilter(question: string, table: Table, profiles: ColumnProfile[]): DataFilter | undefined {
  const lower = question.toLowerCase();

  // 1) Time / year on the time column: "in 2023", "after 2022", "between 2021 and 2023".
  const time = profiles.find((p) => p.role === "time");
  if (time) {
    const yearOf = (row: Record<string, unknown>): number | null => {
      const d = new Date(String(row[time.name]));
      return Number.isNaN(d.getTime()) ? null : d.getFullYear();
    };
    const between = lower.match(/between\s+((?:19|20)\d{2})\s+(?:and|to|-)\s+((?:19|20)\d{2})/);
    if (between) {
      const lo = Math.min(Number(between[1]), Number(between[2]));
      const hi = Math.max(Number(between[1]), Number(between[2]));
      return { column: time.name, phrase: `from ${lo} to ${hi}`, label: `${lo}–${hi}`, predicate: (r) => { const y = yearOf(r); return y !== null && y >= lo && y <= hi; } };
    }
    const years = [...lower.matchAll(/\b((?:19|20)\d{2})\b/g)].map((m) => Number(m[1]));
    if (years.length) {
      const y = years[0];
      if (/\b(after|since|from)\b/.test(lower)) return { column: time.name, phrase: `after ${y}`, label: `after ${y}`, predicate: (r) => { const yr = yearOf(r); return yr !== null && yr > y; } };
      if (/\b(before|until|up to|up until)\b/.test(lower)) return { column: time.name, phrase: `before ${y}`, label: `before ${y}`, predicate: (r) => { const yr = yearOf(r); return yr !== null && yr < y; } };
      return { column: time.name, phrase: `in ${y}`, label: `${y}`, predicate: (r) => yearOf(r) === y };
    }
  }

  // 2) Numeric comparison on a NAMED metric: "revenue over 100", "units between 10 and 20".
  const metrics = profiles.filter((p) => p.role === "metric" && p.numeric);
  const namedMetric = resolveCols(question, metrics)[0];
  if (namedMetric) {
    const m = namedMetric.name;
    const btw = lower.match(new RegExp(`${NUMRE}\\s+(?:and|to)\\s+${NUMRE}`));
    if (/\bbetween\b/.test(lower) && btw) {
      const lo = Math.min(parseNum(btw[1]), parseNum(btw[2]));
      const hi = Math.max(parseNum(btw[1]), parseNum(btw[2]));
      return { column: m, phrase: `where ${m} is between ${lo} and ${hi}`, label: `${m} ${lo}–${hi}`, predicate: (r) => { const v = coerceNum(r[m]); return Number.isFinite(v) && v >= lo && v <= hi; } };
    }
    const cmp = lower.match(new RegExp(`(>=|<=|>|<|over|above|greater than|more than|at least|under|below|less than|fewer than|at most)\\s*${NUMRE}`));
    if (cmp) {
      const op = cmp[1];
      const n = parseNum(cmp[2]);
      const gte = /^(>=|at least)$/.test(op);
      const lte = /^(<=|at most)$/.test(op);
      const gt = /^(>|over|above|greater than|more than)$/.test(op);
      const pred = gte ? (v: number) => v >= n : lte ? (v: number) => v <= n : gt ? (v: number) => v > n : (v: number) => v < n;
      const sym = gte ? "≥" : lte ? "≤" : gt ? ">" : "<";
      const word = gte ? "at least" : lte ? "at most" : gt ? "over" : "under";
      return { column: m, phrase: `where ${m} is ${word} ${n}`, label: `${m} ${sym} ${n}`, predicate: (r) => { const v = coerceNum(r[m]); return Number.isFinite(v) && pred(v); } };
    }
  }

  // 3) Categorical value of a dimension named in the question: "for the North region", "status cancelled".
  const dims = profiles.filter((p) => p.role === "dimension");
  let best: { col: string; value: string } | undefined;
  for (const d of dims) {
    for (const v of distinctValues(table, d.name)) {
      const vl = v.toLowerCase();
      if (vl.length < 2 || FILTER_STOP.has(vl)) continue;
      if (/^-?\d+(\.\d+)?$/.test(vl)) continue; // pure numbers are handled by the numeric filter
      if (vl === d.name.toLowerCase()) continue;
      if (containsPhrase(lower, vl) && (!best || vl.length > best.value.length)) best = { col: d.name, value: v };
    }
  }
  if (best) {
    const target = best.value.toLowerCase();
    const col = best.col;
    return { column: col, phrase: `for ${col} "${best.value}"`, label: best.value, predicate: (r) => String(r[col] ?? "").toLowerCase() === target };
  }

  return undefined;
}

// ── Comparisons ──────────────────────────────────────────────────────────────
// "X vs Y": compare a metric across two slices of one dimension (or two periods). Produces the gap,
// the % difference, the multiple (×), the winner, and a paired bar chart. Runs before the single
// filter so a "vs" question isn't collapsed to one side.

interface Slice {
  label: string;
  predicate: (row: Record<string, unknown>) => boolean;
}

export interface Comparison {
  metric: ColumnProfile;
  agg: "sum" | "mean"; // the other aggregators are nonsensical for a two-slice comparison
  column: string;
  kind: "category" | "time";
  left: Slice;
  right: Slice;
}

const COMPARE_SIGNAL = /\b(vs\.?|versus|compare[ds]?|comparison|compared to|compared with|against|difference between)\b/;

/** Aggregate a metric over the rows matching a predicate. */
function sliceAgg(table: Table, metricName: string, pred: (row: Record<string, unknown>) => boolean, agg: Agg): number {
  const vals = numericColumn(table, metricName);
  const xs: number[] = [];
  table.rows.forEach((row, i) => {
    if (Number.isFinite(vals[i]) && pred(row)) xs.push(vals[i]);
  });
  return aggregate(xs, agg);
}

export function detectComparison(question: string, table: Table, profiles: ColumnProfile[]): Comparison | undefined {
  const lower = question.toLowerCase();
  if (!COMPARE_SIGNAL.test(lower)) return undefined;

  const metrics = profiles.filter((p) => p.role === "metric" && p.numeric);
  if (!metrics.length) return undefined;
  const metric = resolveCols(question, metrics)[0] ?? metrics[0];
  const agg: "sum" | "mean" = /\b(average|avg|mean|per|typical)\b/.test(lower) ? "mean" : "sum";

  // Two distinct years on the time column → period comparison.
  const time = profiles.find((p) => p.role === "time");
  if (time) {
    const years = [...new Set([...lower.matchAll(/\b((?:19|20)\d{2})\b/g)].map((m) => Number(m[1])))];
    if (years.length >= 2) {
      const yearOf = (row: Record<string, unknown>): number | null => {
        const d = new Date(String(row[time.name]));
        return Number.isNaN(d.getTime()) ? null : d.getFullYear();
      };
      // Keep the order they appear in the question.
      const ordered = years.sort((a, b) => lower.indexOf(String(a)) - lower.indexOf(String(b))).slice(0, 2);
      const [y1, y2] = ordered;
      return {
        metric, agg, column: time.name, kind: "time",
        left: { label: `${y1}`, predicate: (r) => yearOf(r) === y1 },
        right: { label: `${y2}`, predicate: (r) => yearOf(r) === y2 },
      };
    }
  }

  // A dimension with ≥2 of its values named in the question → category comparison.
  let bestDim: { col: string; values: { value: string; pos: number }[] } | undefined;
  for (const d of profiles.filter((p) => p.role === "dimension")) {
    const seen = new Map<string, { value: string; pos: number }>();
    for (const v of distinctValues(table, d.name)) {
      const vl = v.toLowerCase();
      if (vl.length < 2 || FILTER_STOP.has(vl) || /^-?\d+(\.\d+)?$/.test(vl) || vl === d.name.toLowerCase()) continue;
      if (containsPhrase(lower, vl) && !seen.has(vl)) seen.set(vl, { value: v, pos: lower.indexOf(vl) });
    }
    const values = [...seen.values()];
    if (values.length >= 2 && (!bestDim || values.length > bestDim.values.length)) bestDim = { col: d.name, values };
  }
  if (bestDim) {
    const [a, b] = bestDim.values.sort((x, y) => x.pos - y.pos);
    const col = bestDim.col;
    const eq = (val: string) => (r: Record<string, unknown>) => String(r[col] ?? "").toLowerCase() === val.toLowerCase();
    return {
      metric, agg, column: col, kind: "category",
      left: { label: a.value, predicate: eq(a.value) },
      right: { label: b.value, predicate: eq(b.value) },
    };
  }

  return undefined;
}

function answerComparison(c: Comparison, table: Table, profiles: ColumnProfile[]): QueryAnswer {
  const l = sliceAgg(table, c.metric.name, c.left.predicate, c.agg);
  const r = sliceAgg(table, c.metric.name, c.right.predicate, c.agg);
  const aggLabel = c.agg === "mean" ? "average" : "total";
  if (!Number.isFinite(l) || !Number.isFinite(r)) {
    return { ok: false, answer: `I don't have enough data to compare ${aggLabel} ${c.metric.name} for both "${c.left.label}" and "${c.right.label}".` };
  }

  const chart = buildComparisonChart(c.metric.name, c.agg, [[c.left.label, l], [c.right.label, r]]);
  const head = `${cap(aggLabel)} ${c.metric.name}: ${c.left.label} ${fmt(l, c.metric)} vs ${c.right.label} ${fmt(r, c.metric)}.`;
  const method = `Computed the ${aggLabel} ${c.metric.name} for each ${c.kind === "time" ? "period" : c.column} slice - "${c.left.label}" and "${c.right.label}" - then took the difference and ratio.`;

  if (l === r) return { ok: true, answer: `${head} They're equal.`, chart, method };

  const higher = l > r ? c.left.label : c.right.label;
  const lowerLabel = l > r ? c.right.label : c.left.label;
  const gap = Math.abs(l - r);
  const hi = Math.max(l, r);
  const lo = Math.min(l, r);
  const pctAbove = lo > 0 ? (gap / lo) * 100 : NaN;
  const ratio = lo > 0 ? hi / lo : NaN;
  const detail =
    `${higher} is higher by ${fmt(gap, c.metric)}` +
    (Number.isFinite(pctAbove) ? ` (${pctAbove.toFixed(1)}% above ${lowerLabel})` : "") +
    (Number.isFinite(ratio) && ratio >= 1.1 ? `, about ${ratio.toFixed(2)}×` : "") +
    ".";
  return { ok: true, answer: `${head} ${detail}`, chart, method };
}

// ── LLM query plan ───────────────────────────────────────────────────────────
// For questions the deterministic engine can't parse, the LLM acts as a query PLANNER: given only the
// schema (column names/roles/types + small samples - never raw rows), it returns a structured plan. We
// validate that plan against the real columns and execute it LOCALLY here, so the numbers are always
// exact and grounded - the model chooses the method, this code computes the answer. Privacy intact.

export interface PlanFilter {
  column: string;
  op: "eq" | "gt" | "lt" | "gte" | "lte" | "between" | "year" | "contains";
  value: string | number;
  value2?: number | null;
}

export interface QueryPlan {
  intent: "metric" | "groupRank" | "groupAggregate" | "aggregate" | "compare" | "trend" | "correlation" | "count" | "distribution" | "describe";
  metric?: string | null;
  metric2?: string | null;
  dimension?: string | null;
  agg?: Agg | null;
  direction?: "top" | "bottom" | null;
  filter?: PlanFilter | null;
  compareValues?: [string, string] | null;
}

function colByName(name: unknown, profiles: ColumnProfile[]): ColumnProfile | undefined {
  if (typeof name !== "string" || !name) return undefined;
  return profiles.find((p) => p.name === name) ?? profiles.find((p) => p.name.toLowerCase() === name.toLowerCase());
}

/** Validate a raw LLM plan against the dataset; coerce column names, drop anything off-spec. */
export function validatePlan(raw: unknown, profiles: ColumnProfile[]): QueryPlan | undefined {
  if (!raw || typeof raw !== "object") return undefined;
  const r = raw as Record<string, unknown>;
  const INTENTS = new Set(["metric", "groupRank", "groupAggregate", "aggregate", "compare", "trend", "correlation", "count", "distribution", "describe"]);
  if (!INTENTS.has(r.intent as string)) return undefined;
  const name = (n: unknown): string | null => colByName(n, profiles)?.name ?? null;
  const AGG = new Set<Agg>(["sum", "mean", "max", "min", "median"]);
  const plan: QueryPlan = { intent: r.intent as QueryPlan["intent"] };
  plan.metric = name(r.metric);
  plan.metric2 = name(r.metric2);
  plan.dimension = name(r.dimension);
  plan.agg = AGG.has(r.agg as Agg) ? (r.agg as Agg) : null;
  plan.direction = r.direction === "bottom" ? "bottom" : r.direction === "top" ? "top" : null;
  if (r.filter && typeof r.filter === "object") {
    const f = r.filter as Record<string, unknown>;
    const col = name(f.column);
    const OPS = new Set(["eq", "gt", "lt", "gte", "lte", "between", "year", "contains"]);
    if (col && OPS.has(f.op as string) && (typeof f.value === "string" || typeof f.value === "number")) {
      plan.filter = { column: col, op: f.op as PlanFilter["op"], value: f.value as string | number, value2: typeof f.value2 === "number" ? f.value2 : null };
    }
  }
  if (Array.isArray(r.compareValues) && r.compareValues.length === 2) {
    plan.compareValues = [String(r.compareValues[0]), String(r.compareValues[1])];
  }
  return plan;
}

const PLAN_INTENTS = new Set(["metric", "groupRank", "groupAggregate", "aggregate", "compare", "trend", "correlation", "count", "distribution", "describe"]);

/** Why a raw LLM plan can't be used as-is - a human-readable reason fed back to the model for ONE repair
 *  attempt (W3.8). Returns null when the plan's intent is supported and every column it names exists.
 *  Pure + exported so it's unit-testable; the actual repair round-trip lives in planQuestion. */
export function planRejectionReason(raw: unknown, profiles: ColumnProfile[]): string | null {
  if (!raw || typeof raw !== "object") return "the response was not a plan object";
  const r = raw as Record<string, unknown>;
  if (!PLAN_INTENTS.has(r.intent as string)) return `intent "${String(r.intent)}" isn't supported`;
  const unknown: string[] = [];
  const check = (v: unknown) => {
    if (typeof v === "string" && v && !colByName(v, profiles)) unknown.push(v);
  };
  check(r.metric);
  check(r.metric2);
  check(r.dimension);
  if (r.filter && typeof r.filter === "object") check((r.filter as Record<string, unknown>).column);
  if (unknown.length) return `uses column${unknown.length > 1 ? "s" : ""} not in the dataset: ${[...new Set(unknown)].join(", ")}`;
  return null;
}

const planNum = (v: unknown): number => (typeof v === "number" ? v : parseFloat(String(v ?? "").replace(/[^0-9.\-]/g, "")));

/** Turn a validated plan filter into the same DataFilter shape the heuristic engine uses. */
function planFilterToDataFilter(f: PlanFilter, profiles: ColumnProfile[]): DataFilter | undefined {
  const col = colByName(f.column, profiles);
  if (!col) return undefined;
  const name = col.name;
  switch (f.op) {
    case "eq": {
      const t = String(f.value).toLowerCase();
      return { column: name, phrase: `for ${name} "${f.value}"`, label: String(f.value), predicate: (r) => String(r[name] ?? "").toLowerCase() === t };
    }
    case "contains": {
      const t = String(f.value).toLowerCase();
      return { column: name, phrase: `where ${name} contains "${f.value}"`, label: String(f.value), predicate: (r) => String(r[name] ?? "").toLowerCase().includes(t) };
    }
    case "year": {
      const y = planNum(f.value);
      return { column: name, phrase: `in ${y}`, label: `${y}`, predicate: (r) => { const d = new Date(String(r[name])); return !Number.isNaN(d.getTime()) && d.getFullYear() === y; } };
    }
    case "between": {
      const lo = Math.min(planNum(f.value), planNum(f.value2));
      const hi = Math.max(planNum(f.value), planNum(f.value2));
      return { column: name, phrase: `where ${name} is between ${lo} and ${hi}`, label: `${name} ${lo}–${hi}`, predicate: (r) => { const v = planNum(r[name]); return Number.isFinite(v) && v >= lo && v <= hi; } };
    }
    default: {
      const n = planNum(f.value);
      const cmp: Record<string, (v: number) => boolean> = { gt: (v) => v > n, lt: (v) => v < n, gte: (v) => v >= n, lte: (v) => v <= n };
      const pred = cmp[f.op];
      if (!pred) return undefined;
      const sym: Record<string, string> = { gt: ">", lt: "<", gte: "≥", lte: "≤" };
      return { column: name, phrase: `where ${name} ${sym[f.op]} ${n}`, label: `${name} ${sym[f.op]} ${n}`, predicate: (r) => { const v = planNum(r[name]); return Number.isFinite(v) && pred(v); } };
    }
  }
}

/** Execute a validated plan locally → an exact, grounded answer (same shape/format as the heuristic). */
export function executePlan(plan: QueryPlan, table: Table, profiles: ColumnProfile[]): QueryAnswer {
  const metrics = profiles.filter((p) => p.role === "metric" && p.numeric);
  const dims = profiles.filter((p) => p.role === "dimension");
  const time = profiles.find((p) => p.role === "time");
  const filter = plan.filter ? planFilterToDataFilter(plan.filter, profiles) : undefined;
  const view = filter ? applyFilter(table, filter) : table;
  if (filter && view.rowCount === 0) return { ok: false, answer: `No records match ${filter.label}. Try a different value or range.` };
  const scope = filter ? ` ${filter.phrase}` : "";
  const metric = colByName(plan.metric, profiles) ?? metrics[0];
  const dim = colByName(plan.dimension, profiles) ?? dims[0];
  const aggFor = (m: ColumnProfile): Agg => plan.agg ?? chooseAgg(m, "");

  switch (plan.intent) {
    case "count":
      return { ok: true, answer: `There are ${view.rowCount.toLocaleString()} records${filter ? scope : " in the dataset"}.`, method: `Counted rows - ${rowsNote(view, table, filter)}.` };
    case "metric": {
      if (!metric) break;
      const vals = numericColumn(view, metric.name);
      return { ok: true, answer: `${metric.name}${scope}: total ${fmt(aggregate(vals, "sum"), metric)}, average ${fmt(aggregate(vals, "mean"), metric)}, range ${fmt(aggregate(vals, "min"), metric)}–${fmt(aggregate(vals, "max"), metric)}.`, method: `Summary stats of ${metric.name} across ${rowsNote(view, table, filter)}.` };
    }
    case "aggregate": {
      if (!metric) break;
      const agg = plan.agg ?? "sum";
      const v = aggregate(numericColumn(view, metric.name), agg);
      const label = agg === "mean" ? "average" : agg === "sum" ? "total" : agg;
      return { ok: true, answer: `The ${label} of ${metric.name}${scope} is ${fmt(v, metric)}.`, method: `${cap(label)} of ${metric.name} across ${rowsNote(view, table, filter)}.` };
    }
    case "groupRank":
    case "groupAggregate": {
      if (!metric || !dim) break;
      const agg = aggFor(metric);
      const ranked = groupBy(view, dim.name, metric.name, agg);
      if (!ranked.length) break;
      const wantLow = plan.direction === "bottom";
      const top = (wantLow ? [...ranked].reverse() : ranked)[0];
      const aggLabel = agg === "mean" ? "average" : agg === "sum" ? "total" : agg;
      return {
        ok: true,
        answer: `By ${aggLabel} ${metric.name}, ${dim.name} "${top[0]}" is ${wantLow ? "lowest" : "highest"} at ${fmt(top[1], metric)}${scope}.`,
        chart: agg === "sum" ? buildChart(view, profiles, { type: "bar", x: dim.name, y: [metric.name], aggregate: true }) : undefined,
        method: `${cap(aggLabel)} ${metric.name} by ${dim.name} across ${rowsNote(view, table, filter)} (${ranked.length} groups).`,
      };
    }
    case "distribution": {
      if (!dim) break;
      const counts = aggregateCount(view, dim.name);
      const total = counts.reduce((s, [, c]) => s + c, 0);
      if (!total) break;
      const t0 = counts[0];
      const t1 = counts[1];
      return {
        ok: true,
        answer: `The most common ${dim.name} is "${t0[0]}" - ${t0[1]} of ${total} (${Math.round((t0[1] / total) * 100)}%)${t1 ? `, then "${t1[0]}" (${Math.round((t1[1] / total) * 100)}%)` : ""}${scope}.`,
        chart: buildChart(view, profiles, { type: "bar", x: dim.name, y: [], count: true }),
        method: `Tallied ${dim.name} across ${rowsNote(view, table, filter)} (${counts.length} distinct values).`,
      };
    }
    case "correlation": {
      const a = colByName(plan.metric, profiles) ?? metrics[0];
      const b = colByName(plan.metric2, profiles) ?? metrics.find((m) => m !== a);
      if (!a || !b) break;
      const r = pearson(numericColumn(view, a.name), numericColumn(view, b.name));
      if (!Number.isFinite(r)) break;
      const strength = Math.abs(r) > 0.7 ? "strong" : Math.abs(r) > 0.4 ? "moderate" : "weak";
      return { ok: true, answer: `${a.name} and ${b.name} have a ${strength} ${r > 0 ? "positive" : "negative"} correlation (r = ${r.toFixed(2)})${scope}.`, chart: buildChart(view, profiles, { type: "scatter", x: a.name, y: [b.name] }), method: `Pearson correlation of ${a.name} and ${b.name} across ${rowsNote(view, table, filter)}.` };
    }
    case "trend": {
      if (!time || !metric) break;
      const order = sortByTime(view, time.name);
      const col = numericColumn(view, metric.name);
      const series = order.map((i) => col[i]).filter(Number.isFinite);
      if (series.length < 2) break;
      const first = series[0];
      const last = series[series.length - 1];
      const pct = first !== 0 ? ((last - first) / Math.abs(first)) * 100 : 0;
      const d = pct > 1 ? "increased" : pct < -1 ? "decreased" : "stayed roughly flat";
      return { ok: true, answer: `${metric.name} ${d} ${Math.abs(pct).toFixed(1)}% over the period${scope} (${fmt(first, metric)} → ${fmt(last, metric)}).`, chart: buildChart(view, profiles, { type: "line", x: time.name, y: [metric.name] }), method: `First vs last ${metric.name} along ${time.name} across ${rowsNote(view, table, filter)}.` };
    }
    case "compare": {
      if (!metric || !dim || !plan.compareValues) break;
      const agg = aggFor(metric);
      const [lv, rv] = plan.compareValues;
      const mk = (val: string) => (r: Record<string, unknown>) => String(r[dim.name] ?? "").toLowerCase() === val.toLowerCase();
      const l = sliceAgg(view, metric.name, mk(lv), agg);
      const rr = sliceAgg(view, metric.name, mk(rv), agg);
      if (!Number.isFinite(l) || !Number.isFinite(rr)) break;
      const aggLabel = agg === "mean" ? "average" : "total";
      const higher = l > rr ? lv : rv;
      return {
        ok: true,
        answer: `${cap(aggLabel)} ${metric.name}: ${lv} ${fmt(l, metric)} vs ${rv} ${fmt(rr, metric)}. ${l === rr ? "They're equal." : `${higher} is higher by ${fmt(Math.abs(l - rr), metric)}.`}`,
        chart: buildComparisonChart(metric.name, agg === "mean" ? "mean" : "sum", [[lv, l], [rv, rr]]),
        method: `Compared ${aggLabel} ${metric.name} for "${lv}" vs "${rv}"${scope}.`,
      };
    }
  }
  return { ok: false, answer: "" };
}

// ── AI path ────────────────────────────────────────────────────────────────────

const round2 = (n: number): number | null => (Number.isFinite(n) ? Math.round(n * 100) / 100 : null);

// The work-context the user optionally typed on the analyzer ("what this data is about"), stored
// locally. Threaded into the AI prompt so answers are framed around the user's actual goal.
const CONTEXT_KEY = "quantia:context";

/** Whether the optional server-side LLM narrator is switched on (the key itself stays server-side). */
function llmOn(): boolean {
  // A user's own key (BYOK) enables the AI path regardless of the server flag - same rule as the
  // dashboard narrator. query.ts runs on the main thread, so localStorage (activeLlmConfig) is available.
  if (activeLlmConfig()) return true;
  return typeof process !== "undefined" && process.env.NEXT_PUBLIC_LLM_ENABLED === "1";
}

function readUserContext(): string | undefined {
  try {
    if (typeof window === "undefined" || !window.localStorage) return undefined;
    const v = window.localStorage.getItem(CONTEXT_KEY);
    return v && v.trim() ? v.trim().slice(0, 400) : undefined;
  } catch {
    return undefined;
  }
}

function strengthLabel(r: number): string {
  const a = Math.abs(r);
  return a > 0.7 ? "strong" : a > 0.4 ? "moderate" : a > 0.2 ? "weak" : "negligible";
}

/** One pass over the rows → per-group total, count and mean of a metric, sorted by total desc. */
function groupStats(table: Table, dim: string, metric: string) {
  const vals = numericColumn(table, metric);
  const m = new Map<string, { sum: number; count: number }>();
  table.rows.forEach((r, i) => {
    const v = vals[i];
    if (!Number.isFinite(v)) return;
    const k = String(r[dim] ?? "-");
    const e = m.get(k) ?? { sum: 0, count: 0 };
    e.sum += v;
    e.count++;
    m.set(k, e);
  });
  return [...m.entries()]
    .map(([key, e]) => ({ key, sum: e.sum, count: e.count, mean: e.count ? e.sum / e.count : NaN }))
    .sort((a, b) => b.sum - a.sum);
}

/** A prior turn in the Ask-your-data conversation (question + the answer the model gave). */
export interface QaTurn {
  q: string;
  a: string;
}

/** A rich, plain-number brief for one metric, reusing the already-computed NumericSummary. */
function metricBrief(p: ColumnProfile) {
  const n = p.numeric!;
  const cv = n.mean !== 0 ? Math.abs(n.std / n.mean) : null; // coefficient of variation (relative spread)
  const skew =
    n.median !== 0 && n.mean > n.median * 1.1
      ? "right-skewed (a few large values pull the average above the typical value)"
      : n.median !== 0 && n.mean < n.median * 0.9
      ? "left-skewed (a few small values pull the average down)"
      : "roughly symmetric";
  return {
    name: p.name,
    type: p.type,
    total: round2(n.sum),
    average: round2(n.mean),
    median: round2(n.median),
    min: round2(n.min),
    max: round2(n.max),
    stdDev: round2(n.std),
    spreadCV: cv === null ? null : round2(cv),
    shape: skew,
    fillRatePct: round2(p.fillRate * 100),
    count: n.count,
  };
}

/** Strongest pairwise correlations among the given metrics (capped for cost), ranked by |r|. */
function topCorrelations(table: Table, metrics: ColumnProfile[], limit: number) {
  const out: { a: string; b: string; r: number | null; r2: number | null; strength: string; direction: string }[] = [];
  for (let i = 0; i < metrics.length; i++) {
    for (let j = i + 1; j < metrics.length; j++) {
      const r = pearson(numericColumn(table, metrics[i].name), numericColumn(table, metrics[j].name));
      // Skip tautological pairs (a column derived from another) so the AI doesn't volunteer the obvious.
      if (Number.isFinite(r) && !isRedundantCorrelation(metrics[i].name, metrics[j].name, r)) {
        out.push({
          a: metrics[i].name,
          b: metrics[j].name,
          r: round2(r),
          r2: round2(r * r),
          strength: strengthLabel(r),
          direction: r > 0 ? "positive" : "negative",
        });
      }
    }
  }
  return out.sort((x, y) => Math.abs(y.r ?? 0) - Math.abs(x.r ?? 0)).slice(0, limit);
}

/** An always-on statistical brief of the WHOLE dataset, so even open-ended questions are grounded. */
function buildOverview(table: Table, profiles: ColumnProfile[]) {
  const metrics = profiles.filter((p) => p.role === "metric" && p.numeric);
  const dims = profiles.filter((p) => p.role === "dimension");
  const time = profiles.find((p) => p.role === "time");
  const o: Record<string, unknown> = {};

  if (metrics.length) o.metrics = metrics.slice(0, 8).map(metricBrief);

  const corr = topCorrelations(table, metrics.slice(0, 8), 6);
  if (corr.length) o.correlations = corr;

  if (dims.length) {
    o.categories = dims.slice(0, 4).map((d) => {
      const counts = aggregateCount(table, d.name);
      const total = counts.reduce((s, [, c]) => s + c, 0);
      return {
        column: d.name,
        distinct: counts.length,
        top: counts.slice(0, 5).map(([value, count]) => ({ value, count, pct: total ? round2((count / total) * 100) : null })),
      };
    });
  }

  if (time && metrics[0]) {
    const order = sortByTime(table, time.name);
    const m0Col = numericColumn(table, metrics[0].name);
    const series = order.map((i) => m0Col[i]).filter(Number.isFinite);
    if (series.length >= 2) {
      const first = series[0];
      const last = series[series.length - 1];
      const pct = first !== 0 ? ((last - first) / Math.abs(first)) * 100 : 0;
      o.overallTrend = { metric: metrics[0].name, over: time.name, first: round2(first), last: round2(last), changePct: round2(pct), periods: series.length };
    }
  }

  return o;
}

/** Question-specific numbers: the focal group breakdown, trend, distribution, and cited correlation. */
export function buildFocalFacts(question: string, table: Table, profiles: ColumnProfile[]) {
  const metrics = profiles.filter((p) => p.role === "metric" && p.numeric);
  const dims = profiles.filter((p) => p.role === "dimension");
  const time = profiles.find((p) => p.role === "time");
  const mMetrics = resolveCols(question, metrics);
  const mDims = resolveCols(question, dims);
  const metric = mMetrics[0] ?? metrics[0];
  const dim = mDims[0];

  const facts: Record<string, unknown> = {};

  // "X vs Y" comparison facts, so the AI narrates the gap/ratio with the exact slice numbers.
  const comparison = detectComparison(question, table, profiles);
  if (comparison) {
    const l = sliceAgg(table, comparison.metric.name, comparison.left.predicate, comparison.agg);
    const r = sliceAgg(table, comparison.metric.name, comparison.right.predicate, comparison.agg);
    if (Number.isFinite(l) && Number.isFinite(r)) {
      const lo = Math.min(l, r);
      facts.comparison = {
        metric: comparison.metric.name,
        basis: comparison.agg === "mean" ? "average" : "total",
        dimension: comparison.column,
        left: { label: comparison.left.label, value: round2(l) },
        right: { label: comparison.right.label, value: round2(r) },
        difference: round2(Math.abs(l - r)),
        pctDiff: r !== 0 ? round2(((l - r) / Math.abs(r)) * 100) : null,
        ratio: lo > 0 ? round2(Math.max(l, r) / lo) : null,
        higher: l === r ? "equal" : l > r ? comparison.left.label : comparison.right.label,
      };
    }
  }

  if (metric && dim) {
    const stats = groupStats(table, dim.name, metric.name); // sorted by total desc
    const grandTotal = stats.reduce((s, g) => s + g.sum, 0);
    const top3 = stats.slice(0, 3).reduce((s, g) => s + g.sum, 0);
    const byAvg = [...stats].sort((a, b) => b.mean - a.mean);
    facts.breakdown = {
      dimension: dim.name,
      metric: metric.name,
      groupCount: stats.length,
      total: round2(grandTotal),
      topGroupSharePct: grandTotal && stats[0] ? round2((stats[0].sum / grandTotal) * 100) : null,
      top3SharePct: grandTotal ? round2((top3 / grandTotal) * 100) : null,
      // Each group with BOTH its total and its average (+ row count) - so questions about per-unit
      // value ("highest average order", "most efficient channel") are answerable, not just totals.
      topGroups: stats.slice(0, 8).map((g) => ({
        key: g.key,
        total: round2(g.sum),
        average: round2(g.mean),
        count: g.count,
        sharePct: grandTotal ? round2((g.sum / grandTotal) * 100) : null,
      })),
      bottomByTotal: stats.length ? { key: stats[stats.length - 1].key, total: round2(stats[stats.length - 1].sum), average: round2(stats[stats.length - 1].mean) } : null,
      highestAverage: byAvg[0] ? { key: byAvg[0].key, average: round2(byAvg[0].mean), count: byAvg[0].count } : null,
    };
  }

  // Multi-facet breakdowns: pre-compute the focal metric across up to two relevant dimensions, so the
  // model can reason over several facets in a single grounded answer - a lightweight stand-in for
  // multi-step tool use, with no extra round-trips and no raw rows ever leaving the page.
  if (metric) {
    const dimsToBreak = (mDims.length ? mDims : dims).slice(0, 2);
    const breakdowns = dimsToBreak
      .map((dd) => {
        const stats = groupStats(table, dd.name, metric.name);
        const grandTotal = stats.reduce((s, g) => s + g.sum, 0);
        return {
          dimension: dd.name,
          metric: metric.name,
          groupCount: stats.length,
          topGroups: stats.slice(0, 6).map((g) => ({
            key: g.key,
            total: round2(g.sum),
            average: round2(g.mean),
            count: g.count,
            sharePct: grandTotal ? round2((g.sum / grandTotal) * 100) : null,
          })),
        };
      })
      .filter((b) => b.groupCount > 0);
    if (breakdowns.length) facts.breakdowns = breakdowns;
  }

  if (mMetrics.length >= 2) {
    const [a, b] = mMetrics;
    const r = pearson(numericColumn(table, a.name), numericColumn(table, b.name));
    if (Number.isFinite(r)) {
      facts.correlation = { a: a.name, b: b.name, r: round2(r), r2: round2(r * r), strength: strengthLabel(r), direction: r > 0 ? "positive" : "negative" };
    }
  }

  if (time && metric) {
    const order = sortByTime(table, time.name);
    const metricCol = numericColumn(table, metric.name);
    const series = order.map((i) => metricCol[i]).filter(Number.isFinite);
    if (series.length >= 2) {
      const first = series[0];
      const last = series[series.length - 1];
      const pct = first !== 0 ? ((last - first) / Math.abs(first)) * 100 : 0;
      facts.trend = { metric: metric.name, over: time.name, first: round2(first), last: round2(last), changePct: round2(pct), peak: round2(maxOf(series)), trough: round2(minOf(series)), periods: series.length };
    }
  }

  if (mDims[0]) {
    const counts = aggregateCount(table, mDims[0].name);
    const totalC = counts.reduce((s, [, c]) => s + c, 0);
    if (totalC > 0) {
      facts.distribution = {
        column: mDims[0].name,
        distinct: counts.length,
        top: counts.slice(0, 8).map(([value, count]) => ({ value, count, pct: round2((count / totalC) * 100) })),
      };
    }
  }

  return facts;
}

/**
 * Assemble the full aggregates-only evidence payload: dataset metadata (+ the user's goal), the
 * deterministic one-liner, question-specific facts, and a whole-dataset statistical overview. No raw
 * rows ever leave - this mirrors the /api/insights privacy boundary.
 */
/** The deep, pre-computed findings (from the full pipeline) handed to the narrator so "why / what
 *  should I do / what's driving X" questions are answered from the regression, ANOVA, trend and action
 *  analysis - not just raw aggregates. This is the multi-step reasoning, done up front. */
export interface AskAnalysis {
  actions?: { title: string; impact: string; detail?: string }[];
  drivers?: { target: string; r2Pct?: number | null; factors: { name: string; beta: number; significant: boolean }[] };
  trends?: { metric: string; changePct: number | null; direction?: string }[];
  /** The AI conclusions' one-line executive read, when available. */
  bottomLine?: string;
  /** Key findings (AI conclusions, falling back to the dashboard insights) - lets the box answer
   *  open/advisory questions ("what should I do?", "summarize this") like the rest of the app would. */
  findings?: string[];
  /** A plain-language "what this dataset is" description (the engine's story). Background ONLY - it
   *  orients the AI's answers; it is never shown to the user as its own panel. */
  datasetDescription?: string;
}

/** Advisory questions ("what should I do?", "give me recommendations") answered locally from the
 *  dashboard's ranked action plan + findings. This is the grounded base for such questions: with the LLM
 *  on it becomes the authoritative input the model narrates from; with it off it IS the answer - either
 *  way the box no longer shrugs at open questions. Returns undefined for anything non-advisory. */
const ADVISORY_RE =
  /(what\s+(should|can|do|would)\s+(i|we|you)\s+do|what\s+actions?|recommended?\s+actions?|recommendations?|next\s+steps?|how\s+(do|can)\s+(i|we)\s+improve|action\s+plan|advi[cs]e|key\s+takeaways?|main\s+conclusions?|summar(y|ize|ise))/i;

export function advisoryAnswer(question: string, analysis?: AskAnalysis): QueryAnswer | undefined {
  if (!analysis || !ADVISORY_RE.test(question)) return undefined;
  const actions = analysis.actions ?? [];
  const findings = analysis.findings ?? [];
  if (!actions.length && !findings.length && !analysis.bottomLine) return undefined;
  const parts: string[] = [];
  if (analysis.bottomLine) parts.push(analysis.bottomLine);
  if (actions.length) {
    parts.push(
      "Here's what I'd do, in priority order:\n" +
        actions
          .map((a, i) => `${i + 1}. ${a.title}${a.detail ? ` - ${a.detail}` : ""}${a.impact ? ` (${a.impact} impact)` : ""}`)
          .join("\n")
    );
  } else if (findings.length) {
    parts.push("The key findings:\n" + findings.map((f, i) => `${i + 1}. ${f}`).join("\n"));
  }
  return {
    ok: true,
    answer: parts.join("\n\n"),
    method:
      "From the dashboard's recommended-actions plan and key findings - each derived from the statistics computed on your data (not generic advice), ranked by impact.",
  };
}

// ── Meta questions ("what can you do?", greetings) ────────────────────────────
// Answered locally and instantly, like any chat assistant would - never the "couldn't map that to
// your columns" shrug. The capabilities answer is grounded in the dataset's real column names.

const GREETING_RE = /^\s*(hi+|hello|hey+|yo|howdy|good\s+(morning|afternoon|evening)|thanks?(\s+you)?|thank\s+you|thx|ty)[\s!.,?]*$/i;
// Note: must NOT match data questions like "what is this dataset about?" - only questions about the
// assistant itself ("you/u" subjects, help, capabilities).
const CAPABILITY_RE =
  /(what\s+(can|do)\s+(you|u)\s+do|what\s+can\s+i\s+ask|what\s+(are\s+)?(you|u)\s+able\s+to\s+do|what\s+are\s+(you|u)\s+capable|how\s+(do|does)\s+(you|this|it)\s+work|^\s*help\s*\??\s*$|help\s+me|who\s+are\s+(you|u)\b|what\s+are\s+(you|u)\??\s*$|your\s+capabilit|what\s+questions?\s+can|what\s+is\s+this\s+(tool|box|chat|assistant))/i;

export function metaAnswer(question: string, table: Table, profiles: ColumnProfile[], analysis?: AskAnalysis): QueryAnswer | undefined {
  const metrics = profiles.filter((p) => p.role === "metric" && p.numeric);
  const dims = profiles.filter((p) => p.role === "dimension");
  const time = profiles.find((p) => p.role === "time");
  const m = metrics[0]?.name ?? "a metric";
  const m2 = metrics[1]?.name;
  const d = dims[0]?.name ?? "a category";

  if (GREETING_RE.test(question)) {
    return {
      ok: true,
      answer: `Hi! I'm your data analyst for "${table.name}". Ask me anything about it - try "average ${m} by ${d}" or "what actions should I take?".`,
    };
  }
  if (!CAPABILITY_RE.test(question)) return undefined;

  const lines = [
    `I'm an AI data analyst for "${table.name}" (${table.rowCount.toLocaleString()} rows). Ask in plain English and I compute the answer from your actual data - every figure comes with a "How I computed this" note. I can:`,
    `• Totals, averages & medians - "total ${m}", "average ${m} by ${d}"`,
    `• Rankings - "which ${d} has the highest ${m}"`,
    `• Comparisons - "A vs B ${m}", and whether a gap is statistically real`,
    ...(m2 ? [`• Relationships - "correlation between ${m} and ${m2}"`] : []),
    ...(time ? [`• Time - "how did ${m} change over time"`] : []),
    `• Shares & spread - "what % of ${m} comes from …", "90th percentile of ${m}"`,
    ...((analysis?.actions?.length ?? 0) > 0 || (analysis?.findings?.length ?? 0) > 0
      ? [`• Advice - "what actions should I take?", "summarize the key takeaways"`]
      : []),
    `Follow-ups work too - I keep the thread's context.`,
  ];
  return { ok: true, answer: lines.join("\n") };
}

function buildEvidence(question: string, table: Table, profiles: ColumnProfile[], grounded: string, domain?: string, history?: QaTurn[], analysis?: AskAnalysis) {
  const conversation = history?.slice(-3).map((h) => ({ q: h.q, a: h.a.slice(0, 320) }));
  // If the question carries a filter, the focal facts are computed on the filtered subset (so the AI
  // narrates the scoped numbers). The overview stays whole-dataset, as broader context.
  const filter = detectFilter(question, table, profiles);
  const view = filter ? applyFilter(table, filter) : table;
  const hasAnalysis =
    analysis &&
    ((analysis.actions?.length ?? 0) > 0 ||
      (analysis.drivers?.factors?.length ?? 0) > 0 ||
      (analysis.trends?.length ?? 0) > 0 ||
      (analysis.findings?.length ?? 0) > 0 ||
      !!analysis.bottomLine);
  return {
    question,
    intent: grounded ? "specific" : "open-ended",
    conversation: conversation && conversation.length ? conversation : undefined,
    scope: filter ? { clause: filter.phrase, column: filter.column, matchedRows: view.rowCount, ofRows: table.rowCount } : undefined,
    dataset: {
      name: table.name,
      rowCount: table.rowCount,
      sampledFrom: table.sampledFrom ?? null,
      domain: domain || undefined,
      // Background orientation for the model (what this dataset is) - never shown to the user directly.
      description: analysis?.datasetDescription || undefined,
      userContext: readUserContext(),
      columns: profiles.map((p) => ({ name: p.name, role: p.role, type: p.type, fillRatePct: round2(p.fillRate * 100) })),
    },
    grounded: grounded || undefined,
    facts: buildFocalFacts(question, view, profiles),
    overview: buildOverview(table, profiles),
    analysis: hasAnalysis ? analysis : undefined,
  };
}

// ── Chart selection for AI answers ───────────────────────────────────────────
// Specific questions already attach a chart from the deterministic engine. Open-ended (and streamed)
// AI answers didn't get one. Here we give every AI answer a relevant chart: the LLM may emit a
// CONSTRAINED chart request (validated against real columns - it only chooses a type + columns, never
// raw ECharts), and when it doesn't (e.g. the streaming path carries prose only) we derive one locally
// from the question via the same NL→chart parser the chart builder uses. Privacy is untouched.

const CHART_TYPES: ChartType[] = ["line", "bar", "scatter", "area", "pie", "histogram"];

/** Validate an LLM-proposed chart request against the real columns; reject anything off-spec. */
export function sanitizeChartRequest(req: unknown, profiles: ColumnProfile[]): ChartRequest | undefined {
  if (!req || typeof req !== "object") return undefined;
  const r = req as Record<string, unknown>;
  if (!CHART_TYPES.includes(r.type as ChartType)) return undefined;
  const names = new Set(profiles.map((p) => p.name));
  if (typeof r.x !== "string" || !names.has(r.x)) return undefined;
  const y = Array.isArray(r.y) ? (r.y as unknown[]).filter((c): c is string => typeof c === "string" && names.has(c)) : [];
  return { type: r.type as ChartType, x: r.x, y, count: r.count === true, aggregate: r.aggregate === true };
}

/** Build a chart for an AI answer: the validated LLM choice if any, else a locally-parsed one from the
 *  question. Honors any filter in the question so the chart matches the scope. Never throws. */
function buildSuggestedChart(question: string, table: Table, profiles: ColumnProfile[], llmReq?: unknown): ChartSpec | undefined {
  const req = sanitizeChartRequest(llmReq, profiles) ?? parseChartRequest(question, profiles).request;
  if (!req) return undefined;
  const filter = detectFilter(question, table, profiles);
  const view = filter ? applyFilter(table, filter) : table;
  if (filter && view.rowCount === 0) return undefined;
  try {
    return buildChart(view, profiles, req);
  } catch {
    return undefined;
  }
}

/**
 * Answer a question with the optional LLM as a principal analyst, grounded in the exact numbers the
 * engine computed (plus a whole-dataset brief so open-ended questions work too). Falls back to the
 * heuristic answer when the LLM is off or fails.
 */
/** Privacy-safe schema brief for the LLM planner: column metadata + a few sample category values and
 *  numeric ranges - never raw rows. */
function buildSchemaBrief(table: Table, profiles: ColumnProfile[], domain?: string) {
  return {
    rowCount: table.rowCount,
    domain: domain || undefined,
    columns: profiles.map((p) => ({
      name: p.name,
      role: p.role,
      type: p.type,
      ...(p.numeric ? { min: round2(p.numeric.min), max: round2(p.numeric.max), mean: round2(p.numeric.mean) } : {}),
      ...(p.topValues && p.topValues.length ? { sampleValues: p.topValues.slice(0, 6).map((v) => v.value) } : {}),
    })),
  };
}

/** Validate a raw plan and execute it locally → a grounded answer, or undefined if it can't be used. */
function executeRawPlan(raw: unknown, table: Table, profiles: ColumnProfile[]): QueryAnswer | undefined {
  const plan = validatePlan(raw, profiles);
  if (!plan || plan.intent === "describe") return undefined;
  const result = executePlan(plan, table, profiles);
  return result.ok ? result : undefined;
}

/** LLM query planner: ask the model to map a hard question to a structured plan (schema only), validate
 *  it, and execute it locally for an exact answer. If the first plan can't be used, make ONE cheap
 *  repair attempt that feeds the rejection reason + valid column names back to the model (W3.8). Returns
 *  undefined on any failure (caller falls back to the heuristic). */
async function planQuestion(question: string, table: Table, profiles: ColumnProfile[], domain?: string, history?: QaTurn[]): Promise<QueryAnswer | undefined> {
  const schema = buildSchemaBrief(table, profiles, domain);
  const conversation = history?.filter((h) => h.a).slice(-2).map((h) => ({ q: h.q, a: h.a.slice(0, 200) }));

  const requestPlan = async (repair?: { rejected: unknown; reason: string }): Promise<unknown | undefined> => {
    try {
      const res = await fetch("/api/insights", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ task: "plan", question, schema, conversation, repair, byok: activeLlmConfig() ?? undefined }),
      });
      if (!res.ok) return undefined;
      return ((await res.json()) as { plan?: unknown }).plan;
    } catch {
      return undefined;
    }
  };

  const first = await requestPlan();
  const firstResult = executeRawPlan(first, table, profiles);
  if (firstResult) return firstResult;

  // One repair pass: tell the model exactly why its plan was rejected and the only valid column names.
  const reason = planRejectionReason(first, profiles) ?? "the plan could not be executed against the dataset";
  const repaired = await requestPlan({ rejected: first, reason });
  return executeRawPlan(repaired, table, profiles);
}

// In-memory cache of answered questions (per dataset), so asking the same thing twice is instant and
// doesn't spend another LLM call - cheaper, and far less likely to hit the provider's rate limit.
const answerCache = new WeakMap<Table, { domain?: string; analysis?: AskAnalysis; entries: Map<string, RichAnswer> }>();
const ANSWER_CACHE_MAX = 80;
const cacheKey = (question: string): string => question.trim().toLowerCase().replace(/\s+/g, " ");

/**
 * Cached entry point for the AI answer. A repeat of the same question (same dataset) returns the prior
 * answer instantly without another network/LLM round-trip. Only confident LLM answers are cached.
 */
export async function answerQuestionAI(
  question: string,
  table: Table,
  profiles: ColumnProfile[],
  domain?: string,
  history?: QaTurn[],
  onToken?: (delta: string) => void,
  analysis?: AskAnalysis
): Promise<RichAnswer> {
  // Self-sufficient currency: a history-loaded analysis may not have run through analyze-client, so detect
  // here too - so a "$" / "€" answer matches the dashboard regardless of how we got to this dataset.
  setActiveCurrency(detectCurrency(table, profiles));
  // Cache hits only apply to fresh questions (no prior conversation), so follow-ups still get context.
  // Two uploads may share a filename and row count while containing entirely different data.
  // Scope cached answers to the actual table and analysis context, and release them with the table.
  let cache = answerCache.get(table);
  if (!cache || cache.domain !== domain || cache.analysis !== analysis) {
    cache = { domain, analysis, entries: new Map() };
    answerCache.set(table, cache);
  }
  const key = cacheKey(question);
  if ((!history || history.length === 0) && cache.entries.has(key)) return cache.entries.get(key)!;

  const result = await runAnswerAI(question, table, profiles, domain, history, onToken, analysis);

  if (result.source === "llm" && result.ok && (!history || history.length === 0)) {
    if (cache.entries.size >= ANSWER_CACHE_MAX) cache.entries.delete(cache.entries.keys().next().value!);
    cache.entries.set(key, result);
  }
  return result;
}

async function runAnswerAI(
  question: string,
  table: Table,
  profiles: ColumnProfile[],
  domain?: string,
  history?: QaTurn[],
  onToken?: (delta: string) => void,
  analysis?: AskAnalysis
): Promise<RichAnswer> {
  // Meta questions ("what can you do?", "hi") get an instant local answer - no network, no LLM,
  // and never the "couldn't map that" failure.
  const meta = metaAnswer(question, table, profiles, analysis);
  if (meta) return { ...meta, source: "heuristic", followups: localFollowups(profiles) };

  let base = answerQuestion(question, table, profiles);

  // Advisory / open questions ("what should I do?", "summarize the takeaways") aren't computations -
  // answer them from the dashboard's ranked actions + findings. With the LLM on, this becomes the
  // authoritative grounded input the model narrates from; with it off, it's the answer itself.
  if (!base.ok) {
    const advisory = advisoryAnswer(question, analysis);
    if (advisory) base = advisory;
  }
  if (!llmOn()) return { ...base, source: "heuristic" };

  // Smart fallback: when the deterministic engine can't map the question, let the LLM PLAN the
  // computation (schema only), then execute it locally for an exact, grounded result. One extra small
  // LLM call, and only for the hard questions the heuristics miss.
  if (!base.ok) {
    const planned = await planQuestion(question, table, profiles, domain, history);
    if (planned?.ok) base = planned;
  }
  const evidence = buildEvidence(question, table, profiles, base.ok ? base.answer : "", domain, history, analysis);

  // Preferred path: stream the answer prose token-by-token for a live, responsive feel. Follow-ups
  // are generated locally in streaming mode (the stream carries prose only).
  if (onToken) {
    try {
      const res = await fetch("/api/insights", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ task: "answer", stream: true, ...evidence, byok: activeLlmConfig() ?? undefined }),
      });
      if (res.ok && res.body) {
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let text = "";
        for (;;) {
          const { value, done } = await reader.read();
          if (done) break;
          const chunk = decoder.decode(value, { stream: true });
          if (chunk) {
            text += chunk;
            onToken(chunk);
          }
        }
        if (text.trim()) {
          return {
            ok: true,
            answer: text.trim(),
            // Streaming carries prose only, so derive the chart locally from the question.
            chart: base.chart ?? buildSuggestedChart(question, table, profiles),
            source: "llm",
            method: base.method,
            followups: localFollowups(profiles),
            grounding: checkGrounding(text.trim(), evidence),
          };
        }
      }
    } catch {
      // streaming failed - fall through to the non-streaming JSON call below
    }
  }

  try {
    const res = await fetch("/api/insights", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ task: "answer", ...evidence, byok: activeLlmConfig() ?? undefined }),
    });
    if (res.ok) {
      const data = (await res.json()) as { answer?: string; followups?: string[]; chart?: unknown };
      if (data.answer && data.answer.trim()) {
        return {
          ok: true,
          answer: data.answer.trim(),
          // Specific questions keep the engine's chart; open-ended ones use the AI's chosen chart
          // (validated), falling back to a locally-parsed one.
          chart: base.chart ?? buildSuggestedChart(question, table, profiles, data.chart),
          source: "llm",
          method: base.method,
          followups: Array.isArray(data.followups)
            ? data.followups.filter((s) => typeof s === "string" && s.trim()).slice(0, 3)
            : undefined,
          grounding: checkGrounding(data.answer.trim(), evidence),
        };
      }
    }
  } catch {
    // fall through to the heuristic answer
  }
  return { ...base, source: "heuristic" };
}

/** Verify the LLM answer's numbers against the evidence it was given (W3.6). Returns undefined when the
 *  answer states no salient numbers (nothing to vouch for), so the UI only shows a signal when it means
 *  something. Never throws - grounding is a trust nicety, not a correctness gate. */
function checkGrounding(answer: string, evidence: unknown): GroundingResult | undefined {
  try {
    const g = verifyAnswerGrounding(answer, evidence);
    return g.salient > 0 ? g : undefined;
  } catch {
    return undefined;
  }
}

/** Relevant next questions, derived locally from the column roles (used for streamed answers). */
function localFollowups(profiles: ColumnProfile[]): string[] {
  const metrics = profiles.filter((p) => p.role === "metric" && p.numeric);
  const dims = profiles.filter((p) => p.role === "dimension");
  const time = profiles.find((p) => p.role === "time");
  const out: string[] = [];
  if (metrics[0] && dims[0]) out.push(`average ${metrics[0].name} by ${dims[0].name}`);
  if (metrics.length >= 2) out.push(`correlation between ${metrics[0].name} and ${metrics[1].name}`);
  if (time && metrics[0]) out.push(`how did ${metrics[0].name} change over time`);
  if (out.length < 3 && dims[0] && metrics[0]) out.push(`which ${dims[0].name} has the highest ${metrics[0].name}`);
  return out.slice(0, 3);
}
