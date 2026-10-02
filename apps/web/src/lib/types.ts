// Core contracts for MyAnalyst's analysis engine.
// Everything flows: raw file -> Table -> ColumnProfile[] -> Domain -> KPIs/Stats/Charts -> Insights -> DashboardSpec.
// These types are the seams between stages; keep them stable so stages can be swapped independently.

/** A parsed tabular dataset: header row + raw cell matrix, before typing. */
export interface Table {
  name: string;
  columns: string[];
  rows: Record<string, unknown>[];
  rowCount: number;
  /** When a very large file was down-sampled, the total number of rows actually
   *  scanned in the source file. `rows` then holds a representative random sample. */
  sampledFrom?: number;
}

export type SemanticType =
  | "number"
  | "currency"
  | "percent"
  | "integer"
  | "date"
  | "boolean"
  | "category"
  | "id"
  | "text";

/** Per-column profile: the statistical fingerprint that drives KPI/chart/domain choices. */
export interface ColumnProfile {
  name: string;
  type: SemanticType;
  /** Fraction of non-null values, 0..1 */
  fillRate: number;
  distinctCount: number;
  /** distinctCount / nonNull count, 0..1 - low = categorical-ish */
  cardinalityRatio: number;
  /** Numeric-only summary stats (undefined for non-numeric). */
  numeric?: NumericSummary;
  /** Date-only span (undefined for non-date). */
  dateRange?: { min: string; max: string };
  /** A few example values for display. */
  samples: string[];
  /** Heuristic role used by the KPI/chart engines. */
  role: "metric" | "dimension" | "time" | "identifier" | "other";
  /** Most frequent values (for categorical/dimension columns) - powers frequency charts & "most common" answers. */
  topValues?: ValueCount[];
}

export interface ValueCount {
  value: string;
  count: number;
  /** share of non-null rows, 0..1 */
  pct: number;
}

export interface NumericSummary {
  min: number;
  max: number;
  mean: number;
  median: number;
  std: number;
  sum: number;
  count: number;
}

export type Domain =
  | "financial-timeseries"
  | "sales-operational"
  | "marketing"
  | "survey"
  | "generic";

export interface DomainGuess {
  domain: Domain;
  confidence: number; // 0..1
  reason: string;
}

export interface Kpi {
  id: string;
  name: string;
  value: number | string;
  unit?: string;
  /** period-over-period delta as a fraction, e.g. 0.23 = +23% */
  trend?: number;
  howComputed: string;
  /** relevance score used to rank/limit cards */
  relevance: number;
  /** optional time-ordered series for a mini sparkline on the card. */
  spark?: number[];
}

export type ChartType =
  | "line"
  | "bar"
  | "scatter"
  | "area"
  | "pie"
  | "heatmap"
  | "histogram";

/** A self-contained chart: an ECharts option plus metadata. The renderer just hands `option` to ECharts. */
export interface ChartSpec {
  id: string;
  type: ChartType;
  title: string;
  subtitle?: string;
  /** Ready-to-render ECharts option object. */
  option: Record<string, unknown>;
  /** Why the engine picked this chart (shown as a tooltip / caption). */
  rationale: string;
}

export interface Insight {
  id: string;
  text: string;
  confidence: "high" | "medium" | "low";
  /** ids of KPIs/stats this claim is grounded in - every insight must cite at least one. */
  cites: string[];
  kind: "trend" | "correlation" | "regression" | "outlier" | "composition" | "summary";
}

/** Metadata-only context handed to an InsightProvider. NEVER contains raw rows. */
export interface InsightContext {
  domain: Domain;
  /** optional free-text description of the user's job/goal - sharpens AI wording & relevance. */
  userContext?: string;
  rowCount: number;
  columns: { name: string; type: SemanticType; role: ColumnProfile["role"] }[];
  kpis: Kpi[];
  correlations: CorrelationPair[];
  regression?: RegressionResult;
  trends: TrendFact[];
  outliers: OutlierFact[];
  forecast?: ForecastFact;
  categories: CategoryFact[];
  groupComparisons: GroupComparison[];
  associations: Association[];
  /** multiple-regression driver analysis (when ≥2 numeric predictors exist) */
  drivers?: DriverAnalysis;
  /** the most concentrated measure×category views ("80–20"), for a concentration-risk insight */
  concentration?: Concentration[];
  /** best-selling products/categories by revenue and by volume - the lead story for sales data */
  bestSellers?: BestSellers;
  /** group-comparison keys ("metric~dimension") whose "copy the leader" framing is a price/product
   *  tautology (a premium product just costs more) and must NOT become a "close the gap" recommendation. */
  suppressGroupComparisons?: string[];
  /** true when the dataset is small enough that estimates are unstable (n < 30) */
  smallSample: boolean;
}

export interface CorrelationPair {
  a: string;
  b: string;
  r: number;
  strength: "strong" | "moderate" | "weak";
  /** two-sided significance test of the correlation */
  p: number;
  significant: boolean;
  ciLow: number;
  ciHigh: number;
  n: number;
}

export interface RegressionResult {
  target: string;
  driver: string;
  slope: number;
  intercept: number;
  r2: number;
  adjR2: number;
  slopeP: number;
  slopeSE: number;
  ciLow: number; // 95% CI for the slope
  ciHigh: number;
  fP: number;
  n: number;
  significant: boolean;
}

export interface TrendFact {
  metric: string;
  changePct: number; // fraction
  direction: "up" | "down" | "flat";
  from: number;
  to: number;
  /** significance of the time trend (OLS of the metric on the time index) */
  slopeP?: number;
  significant?: boolean;
}

/** One-way ANOVA result: does a metric's mean differ across a categorical column's groups? */
export interface GroupComparison {
  metric: string;
  dimension: string;
  f: number;
  p: number;
  etaSq: number; // effect size
  significant: boolean;
  top: { name: string; mean: number; n: number };
  bottom: { name: string; mean: number; n: number };
}

/** Chi-square test of independence between two categorical columns. */
export interface Association {
  a: string;
  b: string;
  chi2: number;
  p: number;
  cramersV: number; // effect size
  significant: boolean;
}

/** Multiple-regression driver analysis: each predictor's independent effect on a target metric. */
export interface DriverAnalysis {
  target: string;
  r2: number;
  adjR2: number;
  fP: number;
  n: number;
  drivers: { name: string; coef: number; beta: number; p: number; significant: boolean }[];
  /** The fitted model + baselines, enough to run local what-if / goal-seek with no raw rows. */
  model?: DriverModel;
}

/** Coefficients + baselines for the driver regression, so the UI can simulate outcomes client-side. */
export interface DriverModel {
  intercept: number;
  targetMean: number;
  targetStd: number;
  /** per-predictor raw coefficient (Δtarget per +1 unit) and the observed distribution for slider bounds. */
  predictors: { name: string; coef: number; mean: number; std: number; min: number; max: number }[];
}

export interface OutlierFact {
  column: string;
  count: number;
  examples: { index: number; value: number; z: number }[];
  /** Root-cause hint: dimension values the anomalous rows concentrate in, vs their baseline share. */
  breakdown?: OutlierBreakdown[];
  /** "skew" = a heavy one-sided tail / real segment (use the median); "anomaly" = isolated suspect points. */
  kind?: "skew" | "anomaly";
  /** which side(s) the extreme values sit on. */
  direction?: "high" | "low" | "both";
  /** population skewness of the column (>0 right-tailed). */
  skew?: number;
  mean?: number;
  median?: number;
  /** extreme count ÷ finite count. */
  share?: number;
}

/** Full outlier classification for a numeric column. */
export interface OutlierAnalysis {
  column: string;
  count: number;
  share: number;
  kind: "skew" | "anomaly";
  direction: "high" | "low" | "both";
  skew: number;
  mean: number;
  median: number;
  std: number;
  examples: { index: number; value: number; z: number }[];
  /** all extreme row indices (for root-cause breakdown), not just the displayed examples. */
  indices: number[];
}

/** One over-represented segment among a metric's anomalous rows. */
export interface OutlierBreakdown {
  dimension: string;
  value: string;
  /** number of anomalous rows in this segment. */
  count: number;
  /** share of the anomalous rows in this segment, 0..1. */
  outlierShare: number;
  /** share of all rows in this segment, 0..1. */
  baseShare: number;
  /** outlierShare / baseShare - >1 means anomalies cluster here more than chance. */
  lift: number;
}

export interface ForecastFact {
  metric: string;
  horizon: number;
  lastValue: number;
  projected: number;
  changePct: number; // fraction, projected vs last observed
  seasonal?: boolean; // true when a seasonal (Holt-Winters) model carried the projection
  period?: number; // detected season length, when seasonal
}

/** Frequency breakdown of a categorical column. */
export interface CategoryFact {
  column: string;
  total: number; // non-null count
  distinct: number;
  top: ValueCount[];
}

/** One product/category's sales performance: how much revenue and volume it drives. */
export interface Performer {
  name: string;
  revenue: number;
  /** revenue / total revenue across the dimension, 0..1. */
  revenueShare: number;
  units: number;
  /** units / total units, 0..1. */
  unitShare: number;
}

/** The best-seller story: which products/categories drive the most revenue and the most volume. */
export interface BestSellers {
  /** the product/category dimension (e.g. "Model", "Brand"). */
  dimension: string;
  /** the revenue/value metric being ranked. */
  metric: string;
  /** number of distinct values in the dimension. */
  distinct: number;
  totalRevenue: number;
  totalUnits: number;
  /** top performers by revenue, largest first. */
  byRevenue: Performer[];
  /** top performers by volume (units / transaction count), largest first. */
  byUnits: Performer[];
  /** #1 by revenue and #1 by volume (may be the same or different products). */
  topRevenue: Performer;
  topUnits: Performer;
  /** true when units were summed from a real quantity column (vs. counting transactions). */
  hasQuantity: boolean;
}

/** An AI-derived, action-oriented interpretation of the data. Not professional advice. */
export interface Conclusion {
  id: string;
  /** plain-language takeaway, written for a non-statistician */
  text: string;
  /** optional "the numbers" line with the underlying statistics, for those who want them */
  detail?: string;
  /** what the conclusion is grounded in (e.g. "Reason frequency", "correlation X~Y"). */
  basis: string;
  confidence: "high" | "medium" | "low";
}

// ── Cleaning stage ──────────────────────────────────────────────────────────

/** Per-column account of what the cleaner did. */
export interface ColumnCleaning {
  name: string;
  detectedType: SemanticType;
  /** cells whose value was reformatted (e.g. "$1,200" → 1200, "1/3/23" → "2023-01-03"). */
  cellsNormalized: number;
  /** cells trimmed of surrounding whitespace. */
  trimmed: number;
  /** empty / null cells in the column. */
  missing: number;
}

export interface CleaningStep {
  label: string;
  detail: string;
  count: number;
}

/** A few rows shown raw vs cleaned, with per-cell changed flags, for the before/after preview. */
export interface CleaningPreview {
  columns: string[];
  rows: { before: string[]; after: string[]; changed: boolean[] }[];
}

export interface CleaningReport {
  rowsBefore: number;
  rowsAfter: number;
  duplicatesRemoved: number;
  emptyRowsRemoved: number;
  totalRowsRemoved: number;
  cellsNormalized: number;
  cellsTrimmed: number;
  columns: ColumnCleaning[];
  steps: CleaningStep[];
  preview: CleaningPreview;
}

// ── Time-series analysis ──────────────────────────────────────────────────────

export type Cadence = "daily" | "weekly" | "monthly" | "quarterly" | "yearly";

export interface PeriodPoint {
  label: string;
  value: number;
}

/** A metric aggregated into its natural periods, with period-over-period change + a moving average. */
export interface TimeSeriesAnalysis {
  metric: string;
  cadence: Cadence;
  /** the metric summed within each period, in chronological order. */
  periods: PeriodPoint[];
  latest: PeriodPoint;
  previous?: PeriodPoint;
  /** latest vs the immediately preceding period, as a fraction (0.2 = +20%). */
  changePct?: number;
  /** latest vs the same period one season (e.g. year) ago, when available. */
  yoyChangePct?: number;
  /** trailing moving average aligned to `periods` (null until the window fills). */
  movingAvg: (number | null)[];
  best: PeriodPoint;
  worst: PeriodPoint;
  /** Recurring within-cycle pattern (e.g. month-of-year, weekday), when ≥2 full cycles are present. */
  seasonality?: SeasonPattern;
}

/** One position in a seasonal cycle (e.g. "December", "Q4", "Monday") and its index vs the average. */
export interface SeasonIndex {
  label: string;
  /** average metric value at this cycle position. */
  avg: number;
  /** avg ÷ overall average (1 = on par, >1 = above-average season). */
  index: number;
}

/** A recurring seasonal pattern: which positions in the cycle run hot/cold, and how strongly. */
export interface SeasonPattern {
  /** the cycle unit - "month", "quarter", or "weekday". */
  unit: "month" | "quarter" | "weekday";
  /** every cycle position in order, with its seasonal index. */
  indices: SeasonIndex[];
  peak: SeasonIndex;
  trough: SeasonIndex;
  /** peak.index − trough.index - the amplitude of the swing. */
  strength: number;
}

// ── Contribution / mix-shift decomposition ────────────────────────────────────

/** One segment's part in a metric's period-over-period change. Deltas sum to the total change. */
export interface ContributionSegment {
  name: string;
  prev: number;
  latest: number;
  /** latest − prev. */
  delta: number;
  /** signed share of the total change (delta / totalDelta); can exceed 1 or be negative when segments offset. */
  contributionPct: number;
  /** share of the period total in the previous and latest periods (0..1) - the mix-shift view. */
  sharePrev: number;
  shareLatest: number;
  status: "grew" | "shrank" | "new" | "lost" | "flat";
}

/** Why a metric's total moved between the two most recent periods, attributed by a dimension. */
export interface ContributionAnalysis {
  metric: string;
  dimension: string;
  cadence: Cadence;
  prevLabel: string;
  latestLabel: string;
  prevTotal: number;
  latestTotal: number;
  totalDelta: number;
  totalDeltaPct: number | null;
  /** segments ranked by the absolute size of their contribution; small ones rolled into "Other". */
  segments: ContributionSegment[];
}

// ── Segmentation ──────────────────────────────────────────────────────────────

/** One natural group found by clustering, described by the features that set it apart. */
export interface Segment {
  id: number;
  /** plain-language descriptor, e.g. "High Revenue, low Tenure". */
  label: string;
  size: number;
  sharePct: number;
  /** the features that most distinguish this group from the overall average. */
  defining: { column: string; direction: "high" | "low"; z: number; mean: number }[];
}

export interface Segmentation {
  k: number;
  features: string[];
  segments: Segment[];
  /** number of rows clustered, when a sample was used instead of the full table. */
  sampled?: number;
}

/** A clustered row's assignment - `rowIndex` into the table, `cluster` matching a Segment.id. */
export interface SegmentMember {
  rowIndex: number;
  cluster: number;
}

// ── Concentration / Pareto (80–20) ──────────────────────────────────────────────

/** One category's slice of a measure, with its share and the running cumulative share. */
export interface ConcentrationSegment {
  name: string;
  /** the measure total (or row count) for this category. */
  value: number;
  /** value / grand total, 0..1. */
  share: number;
  /** cumulative share through this row (sorted largest-first), 0..1. */
  cumShare: number;
  /** 1-based position in the sorted ranking. */
  rank: number;
  /** true for the rolled-up long tail ("Other") row. */
  isOther?: boolean;
}

/** One "vital few" category behind a concentration result - the exportable membership detail. */
export interface ConcentrationMember {
  rank: number;
  name: string;
  value: number;
  share: number;
  cumShare: number;
}

/** How concentrated a measure is across the values of one categorical/identifier column. */
export interface Concentration {
  dimension: string;
  /** the measure being concentrated; "row count" when counting rows. */
  metric: string;
  metricIsCount: boolean;
  /** grand total of the measure across all categories. */
  total: number;
  /** number of distinct categories. */
  distinct: number;
  /** largest categories first, with the long tail rolled into a final "Other" row. */
  segments: ConcentrationSegment[];
  /** how many of the top categories it takes to reach 80% of the total. */
  paretoCount: number;
  /** the actual cumulative share at paretoCount (≥ 0.8 unless the whole set is below it). */
  paretoShare: number;
  /** paretoCount / distinct - the "vital few" as a fraction of all categories. */
  paretoPctOfCategories: number;
  /** share held by the single largest category, 0..1. */
  topShare: number;
  /** Gini coefficient across categories (0 = perfectly even, → 1 = one category holds everything). */
  gini: number;
  /** Herfindahl–Hirschman index = Σ share² (1/distinct = even, 1 = a single category). */
  hhi: number;
  level: "high" | "moderate" | "low";
}

// ── Relationship explorer (full correlation matrix) ──────────────────────────────

/** One numeric pair's relationship, with significance, CI, and a redundancy flag. */
export interface RelationshipPair {
  a: string;
  b: string;
  r: number;
  strength: "strong" | "moderate" | "weak";
  p: number;
  significant: boolean;
  ciLow: number;
  ciHigh: number;
  n: number;
  /** near-perfect or name-subset pairs that are almost certainly derived/duplicate, not findings. */
  redundant: boolean;
}

/** The full pairwise correlation matrix over the numeric columns, for an interactive heatmap. */
export interface RelationshipMatrix {
  /** numeric column names, in matrix row/column order. */
  columns: string[];
  /** symmetric r matrix; diagonal = 1. NaN where a pair had too few paired points. */
  matrix: number[][];
  /** every unique pair (i<j), strongest |r| first. */
  pairs: RelationshipPair[];
}

// ── RFM customer segmentation ────────────────────────────────────────────────────

/** One RFM segment (Champions, Loyal, At-Risk, …) with its averaged scores and value. */
export interface RfmSegment {
  key: string;
  label: string;
  /** short plain-language description of who these customers are. */
  blurb: string;
  size: number;
  sharePct: number;
  avgRecencyDays: number;
  avgFrequency: number;
  avgMonetary: number;
  totalMonetary: number;
  /** this segment's share of total monetary value, 0..1. */
  monetaryShare: number;
}

/** One scored entity (customer) behind the RFM segments - the exportable membership detail. */
export interface RfmMember {
  /** the entity id (value of the entity column). */
  id: string;
  recencyDays: number;
  frequency: number;
  monetary: number;
  /** 1–5 recency / frequency quintile scores. */
  rScore: number;
  fScore: number;
  segmentKey: string;
  segmentLabel: string;
}

/** Recency–Frequency–Monetary segmentation of an entity (customer) by transaction-shaped data. */
export interface RfmAnalysis {
  entity: string;
  dateColumn: string;
  valueColumn: string;
  /** the most recent date in the data - recency is measured back from here. */
  asOf: string;
  customers: number;
  /** segments ranked by total monetary value, largest first. */
  segments: RfmSegment[];
}

// ── Action report ─────────────────────────────────────────────────────────────

/** One prioritized, grounded "do this next" recommendation derived from the analysis. */
export interface ActionItem {
  id: string;
  /** imperative headline - "do this". */
  title: string;
  /** grounded rationale with the actual numbers. */
  detail: string;
  impact: "high" | "medium" | "low";
  /** what the recommendation is grounded in. */
  basis: string;
}

// ── Cohort & retention ────────────────────────────────────────────────────────

export interface CohortRow {
  /** the cohort's first period, e.g. "2023-01". */
  label: string;
  /** entities first seen in this cohort. */
  size: number;
  /** retention % at offset 0,1,2,… (offset 0 is always 100). */
  retention: (number | null)[];
}

export interface CohortAnalysis {
  entity: string;
  time: string;
  cadence: Cadence;
  cohorts: CohortRow[];
  /** widest retention row (number of offset columns). */
  periodCount: number;
}

// ── Open-text / survey analytics ───────────────────────────────────────────────

/** One recurring theme (keyword or phrase) in a free-text column, with a representative quote. */
export interface TextTerm {
  term: string;
  count: number;
  /** share of responses mentioning it, 0..1. */
  share: number;
  /** a representative verbatim containing the term (truncated). */
  sample?: string;
}

/** Themes + sentiment extracted from a free-text column (verbatims, feedback, reviews). */
export interface TextAnalysis {
  column: string;
  responseCount: number;
  avgWords: number;
  /** top themes (phrases preferred over their component words), most frequent first. */
  terms: TextTerm[];
  /** lexicon sentiment split (shares) + mean score in [-1, 1] (undefined if no sentiment words found). */
  sentiment?: { positive: number; neutral: number; negative: number; score: number };
}

// ── Data-quality scorecard ────────────────────────────────────────────────────

/** One dimension of the data-quality score (completeness, uniqueness, …). */
export interface QualityCheck {
  id: string;
  label: string;
  /** 0..1 sub-score for this dimension. */
  score: number;
  /** contribution weight in the overall score. */
  weight: number;
  status: "good" | "warn" | "bad";
  /** what we found, in plain language. */
  detail: string;
  /** a concrete suggested fix (present when the check isn't already good). */
  fix?: string;
}

/** A "read with care" flag on a column, propagated from the data-quality scorecard onto the analysis. */
export interface Caveat {
  column: string;
  reason: string;
  severity: "warn" | "bad";
}

/** An at-a-glance 0–100 health score for the dataset, with a letter grade and per-dimension breakdown. */
export interface DataQuality {
  score: number; // 0..100
  grade: "A" | "B" | "C" | "D" | "F";
  checks: QualityCheck[];
  summary: string;
}

/** The full declarative result the dashboard renders. */
export interface DashboardSpec {
  version: string;
  /** Scope of server-computed KPIs/charts; survives share links and exports. */
  computeScope?: { sourceRows: number; analyzedRows: number; sampled: boolean };
  datasetName: string;
  domain: DomainGuess;
  /** Detected currency for money formatting across the dashboard (defaults to USD/$). */
  currency?: { symbol: string; code: string };
  generatedAt: string;
  rowCount: number;
  cleaning: CleaningReport;
  profiles: ColumnProfile[];
  kpis: Kpi[];
  charts: ChartSpec[];
  insights: Insight[];
  /** AI-derived, action-oriented conclusions (with a "not professional advice" disclaimer in the UI). */
  conclusions: Conclusion[];
  /** which narrator wrote the insights - drives the "AI-narrated" badge. */
  narrator: "llm" | "templated";
  /** A short plain-language "what is this data" narrative: inferred industry,
   *  subject, and likely purpose - so findings stay connected to the subject. */
  story?: DataStory;
  /** A 0–100 data-quality health score with a per-dimension breakdown and fixes. */
  quality?: DataQuality;
  /** Per-metric unusual values (|z| > 3), surfaced so users can verify or exclude them. */
  anomalies?: OutlierFact[];
  /** Period-over-period analysis (cadence, MoM/YoY change, moving average) for the top metrics. */
  timeAnalysis?: TimeSeriesAnalysis[];
  /** Natural groups found by clustering the numeric columns. */
  segmentation?: Segmentation;
  /** Multiple-regression "what moves the primary metric" driver analysis (when ≥2 predictors exist). */
  drivers?: DriverAnalysis;
  /** Cohort retention grid, when the data has a recurring entity id + a time column. */
  cohorts?: CohortAnalysis;
  /** A ranked, quantified "what to do next" action plan derived from the analysis. */
  actions?: ActionItem[];
  /** "What drove the change" - period-over-period movement of the primary metric, attributed by dimension. */
  contributions?: ContributionAnalysis[];
  /** Themes + sentiment for free-text columns (open-ended feedback, reviews, notes). */
  textAnalysis?: TextAnalysis[];
  /** "Read with care" flags propagated from the data-quality scorecard onto the analysis. */
  caveats?: Caveat[];
  /** Pareto / concentration: how unevenly a measure is spread across its categories ("80–20"). */
  concentration?: Concentration[];
  /** Full pairwise correlation matrix over the numeric columns, for the interactive relationship explorer. */
  relationships?: RelationshipMatrix;
  /** RFM customer segmentation, when the data is transaction-shaped (entity id + date + value). */
  rfm?: RfmAnalysis;
  /** True when the dataset is small enough (n < 30) that estimates are unstable. */
  smallSample?: boolean;
}

export interface DataStory {
  /** e.g. "Sales / retail", "SaaS", "Marketing" - the inferred industry/subject. */
  industry: string;
  /** 2–3 sentence plain-language description of what the dataset is and is used for. */
  summary: string;
  /** Whether the description came from the local heuristic or was sharpened by the LLM. */
  source?: "heuristic" | "llm";
}

/** Pluggable insight generator. Templated impl now; an LLM-backed impl can replace it later. */
export interface InsightProvider {
  readonly name: string;
  /** set after generate(): which narrator actually produced the insights (LLM may fall back). */
  lastSource?: "llm" | "templated";
  generate(ctx: InsightContext): Promise<Insight[]>;
}
