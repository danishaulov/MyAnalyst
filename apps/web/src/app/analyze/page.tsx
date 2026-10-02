"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import type { DashboardSpec, Table } from "@/lib/types";
import { parseFile, type SourceInfo } from "@/lib/parse";
import { runAnalysis } from "@/lib/analyze-client";
import { downloadCsv } from "@/lib/csv";
import { sampleTable } from "@/lib/sample";
import { combinedContext } from "@/lib/industry-tags";
import { IndustryTagPicker } from "@/components/IndustryTagPicker";
import { exportPdf, exportPng } from "@/lib/export";
import { exportDeckPdf, exportReportPdf } from "@/lib/report-pdf";
import { loadBrand } from "@/lib/brand";
import { activeLlmConfig, localModelEnabled } from "@/lib/llm-settings";
import { localNarrateStory, webgpuAvailable } from "@/lib/local-llm";
import { BrandEditor } from "@/components/BrandEditor";
import { AiKeyEditor } from "@/components/AiKeyEditor";
import { encodeShare, MAX_LINK_CHARS } from "@/lib/share";
import { deleteAnalysis, getAnalysis, listHistory, saveAnalysis, type HistoryEntry } from "@/lib/history";
import { PresenterMode } from "@/components/PresenterMode";
import { Uploader } from "@/components/Uploader";
import { BrandMark } from "@/components/BrandMark";
import { DashboardView } from "@/components/DashboardView";
import { pyChartsToSpecs } from "@/lib/py-charts";
import { runPythonAnalysis, runPythonConclusions, type PyAnalysisSpec, type PyConclusions } from "@/lib/py-engine";
import { DISCLAIMER_TEXT } from "@/components/Disclaimer";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import { HistoryList } from "@/components/HistoryList";
import { PipelineProgress } from "@/components/PipelineProgress";

const CONTEXT_KEY = "quantia:context";
const INDUSTRY_KEY = "quantia:industry";

// Pipeline shown in the progress UI. The deep Python analysis is the authoritative engine and the
// long pole, so it sits right after the data is prepared (cleaned + profiled) and IS the headline step
// the user waits on. The in-browser TS engine's finer stages (domain, KPIs, stats, insights) now run
// quickly IN PARALLEL underneath the "Deep analysis" window, so they're not shown as separate steps.
const STAGES = ["Reading file", "Cleaning & normalizing", "Profiling columns", "Deep analysis (Python)", "Writing AI conclusions"];
// Worker stages that count as data prep (shown directly); everything else it emits is the fast,
// parallel TS detail that runs under the Deep-analysis window and is folded into it.
const PREP_STAGES = new Set(["Reading file", "Cleaning & normalizing", "Profiling columns"]);

/** A file that's been read and is waiting for the user to confirm before analysis starts - so nobody
 *  analyzes the wrong file, and they get a chance to pick the right sheet and add context first. */
interface PendingFile {
  file: File;
  table: Table;
  sources: SourceInfo[];
  sourceId: string;
  sourceKind?: "sheet" | "table";
}

function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`;
  return `${(n / (1024 * 1024)).toFixed(n < 10 * 1024 * 1024 ? 1 : 0)} MB`;
}

/** Build the spec the dashboard renders: when the Python engine has run, its KPIs + charts take over -
 *  but only when they're non-empty. Python's smart-KPI selection can legitimately return nothing for some
 *  datasets, and blindly swapping that in wiped the "Key metrics" panel; fall back to the TS engine's
 *  KPIs/charts in that case so the dashboard is never blank. */
function mergedDashboardSpec(spec: DashboardSpec, pySpec: PyAnalysisSpec | null): DashboardSpec {
  if (!pySpec) return spec;
  const pyCharts = pyChartsToSpecs(pySpec.charts);
  return {
    ...spec,
    computeScope: pySpec.scope,
    kpis: pySpec.kpis?.length ? pySpec.kpis : spec.kpis,
    charts: pyCharts.length ? pyCharts : spec.charts,
  };
}

export default function AnalyzePage() {
  const [busy, setBusy] = useState(false);
  const [stage, setStage] = useState<string | null>(null);
  const [readInfo, setReadInfo] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [table, setTable] = useState<Table | null>(null);
  const [spec, setSpec] = useState<DashboardSpec | null>(null);
  // The Python engine computes the dashboard KPIs/charts/conclusions (the primary view); the TS spec above
  // stays as a safety fallback if the Python backend is unreachable.
  const [pySpec, setPySpec] = useState<PyAnalysisSpec | null>(null);
  const [pyConclusions, setPyConclusions] = useState<PyConclusions | null>(null);
  // The full parsed source (all columns), kept so source/sheet switches can re-analyze from scratch.
  const [sourceTable, setSourceTable] = useState<Table | null>(null);
  // Multi-sheet (Excel) / multi-table (SQLite) support: the uploaded file + its selectable sources.
  const [sourceFile, setSourceFile] = useState<File | null>(null);
  const [fileSources, setFileSources] = useState<SourceInfo[]>([]);
  const [currentSourceId, setCurrentSourceId] = useState<string>("");
  const [sourceKind, setSourceKind] = useState<"sheet" | "table" | undefined>(undefined);
  const [joinId, setJoinId] = useState<string>("");
  const [exporting, setExporting] = useState<null | "png" | "pdf" | "report" | "deck">(null);
  const [presenting, setPresenting] = useState(false);
  const [branding, setBranding] = useState(false);
  const [aiKeyOpen, setAiKeyOpen] = useState(false);
  const [toast, setToast] = useState<{ text: string; tone: "info" | "error" } | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [jobDesc, setJobDesc] = useState("");
  const [industry, setIndustry] = useState<string | null>(null);
  // A read-but-not-yet-analyzed file, shown in a review step so the user can confirm it's the right
  // file (and sheet), add context, and only then start. `stagingSource` = re-reading a different sheet.
  const [pending, setPending] = useState<PendingFile | null>(null);
  const [stagingSource, setStagingSource] = useState(false);
  const dashboardRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    listHistory().then(setHistory).catch(() => {});
    // Work-context + industry tag live only in this browser (localStorage) - never sent anywhere.
    setJobDesc(localStorage.getItem(CONTEXT_KEY) ?? "");
    setIndustry(localStorage.getItem(INDUSTRY_KEY) || null);
  }, []);

  function updateIndustry(key: string | null) {
    setIndustry(key);
    try {
      if (key) localStorage.setItem(INDUSTRY_KEY, key);
      else localStorage.removeItem(INDUSTRY_KEY);
    } catch {
      /* ignore quota */
    }
  }

  // Transient toasts auto-dismiss; errors linger a little longer than info.
  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(() => setToast(null), toast.tone === "error" ? 8000 : 5000);
    return () => clearTimeout(t);
  }, [toast]);

  function updateContext(v: string) {
    setJobDesc(v);
    try {
      localStorage.setItem(CONTEXT_KEY, v);
    } catch {
      /* ignore quota */
    }
  }

  async function handleOpenHistory(id: string) {
    const loaded = await getAnalysis(id);
    if (!loaded) return;
    setError(null);
    setToast(null);
    // Chart formatters don't survive JSON serialization, so rebuild the charts from the stored
    // table + profiles - same as the live path - so reopened dashboards look identical to fresh ones.
    try {
      const { recommendCharts } = await import("@/lib/charts");
      loaded.spec.charts = recommendCharts(loaded.table, loaded.spec.profiles);
    } catch {
      /* fall back to the stored (functionless) charts */
    }
    setTable(loaded.table);
    setSpec(loaded.spec);
    // Restore the entry's own Python KPIs/charts + AI conclusions - and, crucially, CLEAR any left over
    // from the previously open dataset (they'd otherwise merge into this one's dashboard and share link).
    setPySpec(loaded.pySpec);
    setPyConclusions(loaded.conclusions);
    // Re-runnable from the loaded table.
    setSourceTable(loaded.table);
    // History entries have no underlying file, so there's no sheet/table picker.
    setSourceFile(null);
    setFileSources([]);
    setCurrentSourceId("");
    setSourceKind(undefined);
    if (typeof window !== "undefined") window.scrollTo({ top: 0 });
  }

  async function handleDeleteHistory(id: string) {
    await deleteAnalysis(id);
    setHistory(await listHistory());
  }

  async function handleShare() {
    if (!spec) return;
    setToast({ text: "Building link…", tone: "info" });
    try {
      const payload = await encodeShare(mergedDashboardSpec(spec, pySpec), pyConclusions);
      const url = `${window.location.origin}/view#${payload}`;
      if (url.length > MAX_LINK_CHARS) {
        setToast({ text: "Dataset too large for a link - use PNG/PDF export instead.", tone: "error" });
        return;
      }
      await navigator.clipboard.writeText(url);
      setToast({ text: `🔗 Read-only link copied (${(url.length / 1024).toFixed(0)} KB) - paste it anywhere.`, tone: "info" });
    } catch {
      setToast({ text: "Couldn't create the link in this browser.", tone: "error" });
    }
  }

  async function handleExport(kind: "png" | "pdf" | "report" | "deck") {
    if (!spec || exporting) return;
    if ((kind === "png" || kind === "pdf") && !dashboardRef.current) return;
    setExporting(kind);
    setToast(null);
    try {
      const meta = `${spec.rowCount.toLocaleString()} rows · ${spec.profiles.length} columns · ${spec.domain.domain}`;
      const brand = loadBrand();
      if (kind === "png") await exportPng(dashboardRef.current!, spec.datasetName, meta, brand);
      else if (kind === "pdf") await exportPdf(dashboardRef.current!, spec.datasetName, meta, brand);
      // Report/Deck render from the MERGED spec so their KPIs match what's on screen (the Python
      // engine's), not the TS fallback's.
      else if (kind === "report") await exportReportPdf(mergedDashboardSpec(spec, pySpec), brand);
      else await exportDeckPdf(mergedDashboardSpec(spec, pySpec), brand);
      const label = kind === "report" ? "Report PDF" : kind === "deck" ? "Deck PDF" : kind.toUpperCase();
      setToast({ text: `✓ ${label} downloaded.`, tone: "info" });
    } catch {
      // The export-error banner was previously routed to `error`, which only renders in the
      // pre-analysis (no-spec) view - so failures were silent. Surface it in the toast instead.
      setToast({ text: "Export failed - the dashboard may be too large. Try again or use PNG.", tone: "error" });
    } finally {
      setExporting(null);
    }
  }

  async function run(sourceTbl: Table) {
    // Fold the chosen industry tag into the analysis context so domain detection + the AI understand
    // the uploaded file. (The "or by industry" sample buttons are unrelated - they generate showcase data.)
    const ctx = combinedContext(industry, jobDesc);
    setError(null);
    setBusy(true);
    setSpec(null);
    setPySpec(null);
    setPyConclusions(null);
    setSourceTable(sourceTbl);
    try {
      // The whole pipeline (clean → profile → stats → charts → insights) runs in a Web Worker,
      // so the UI never freezes even on a 200k-row file - and the progress here reflects the
      // worker's REAL stage transitions instead of a timed animation.
      //
      // ── The deep PYTHON analysis (pandas/statsmodels) is the authoritative engine: it computes the
      // headline KPIs + charts and is the basis for every AI conclusion. It only needs the cleaned +
      // profiled table, so we kick it off IN PARALLEL the moment the worker has that ready (via the
      // `onPrepared` callback) rather than waiting for the entire TS pass to finish - its network
      // round-trip overlaps the rest of the in-browser dashboard build. We still reveal everything at
      // once (a single finished result, no confusing "re-analyzing" flash). If the backend is
      // unreachable (e.g. local `next dev`, or a transient error), we fall back to the TS spec.
      let pyPromise: Promise<PyAnalysisSpec> | null = null;

      setStage("Cleaning & normalizing");
      const { spec: result, table: cleaned } = await runAnalysis(
        sourceTbl,
        ctx,
        // Only surface the prep stages; the TS engine's finer stages run in parallel under the
        // "Deep analysis (Python)" step (set on onPrepared below), so we don't bounce the progress.
        (s) => { if (PREP_STAGES.has(s)) setStage(s); },
        {},
        activeLlmConfig() ?? undefined,
        (cleanedTbl, currency) => {
          // Data is cleaned & profiled - start the deep Python analysis now, in parallel, and show it
          // as the active step (it's the long pole everything else overlaps).
          setStage("Deep analysis (Python)");
          pyPromise = runPythonAnalysis(cleanedTbl.columns, cleanedTbl.rows, currency);
          pyPromise.catch(() => {}); // mark handled; the await below turns any failure into the TS fallback
        }
      );

      let py: PyAnalysisSpec | null = null;
      let conclusions: PyConclusions | null = null;
      try {
        setStage("Deep analysis (Python)"); // idempotent safety if onPrepared never fired
        // Started in parallel via onPrepared (so it's often already finished here). Fall back to
        // starting it now if, on some unusual path, onPrepared never fired.
        py = await (pyPromise ?? runPythonAnalysis(cleaned.columns, cleaned.rows, result.currency));
        setStage("Writing AI conclusions");
        conclusions = await runPythonConclusions(py, ctx).catch(() => null);
      } catch {
        py = null; // backend unreachable → render the TS dashboard so the page never goes blank
      }

      // Reveal the dashboard, Python KPIs/charts, and AI conclusions together - a single finished result.
      setTable(cleaned);
      setPySpec(py);
      setPyConclusions(conclusions);
      setSpec(result);

      // On-device narration (opt-in, WebGPU): sharpen the story locally with ZERO network, after the
      // dashboard is already up. Fire-and-forget so it never blocks rendering; patches the story in place.
      if (localModelEnabled() && webgpuAvailable() && result.story) {
        setToast({ text: "Warming the on-device model… (first run downloads it once)", tone: "info" });
        localNarrateStory(
          {
            datasetName: result.datasetName,
            domain: result.domain.domain,
            rowCount: result.rowCount,
            columns: result.profiles.map((p) => ({ name: p.name, role: p.role, type: p.type })),
            userContext: ctx,
          },
          result.story.summary
        )
          .then((text) => {
            if (text) {
              setSpec((prev) => (prev && prev.story ? { ...prev, story: { ...prev.story, summary: text, source: "llm" }, narrator: "llm" } : prev));
              setToast({ text: "✓ Story sharpened on-device - no network used.", tone: "info" });
            } else {
              setToast(null);
            }
          })
          .catch(() => setToast(null));
      }
      try {
        await saveAnalysis(result, cleaned, { pySpec: py, conclusions });
        setHistory(await listHistory());
      } catch {
        /* history is best-effort; never block the dashboard on it */
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong analyzing that file.");
    } finally {
      setBusy(false);
      setStage(null);
    }
  }

  async function handleFile(file: File) {
    try {
      setError(null);
      setBusy(true);
      setStage("Reading file");
      setReadInfo(file.size > 24 * 1024 * 1024 ? `Reading ${(file.size / 1048576).toFixed(0)} MB…` : null);
      const result = await parseFile(file, (p) => {
        const pct = p.totalBytes ? Math.min(100, Math.round((p.bytes / p.totalBytes) * 100)) : 0;
        setReadInfo(`Read ${p.rows.toLocaleString()} rows · ${pct}%`);
      });
      setReadInfo(null);
      if (!result.table.columns.length || !result.table.rowCount) {
        throw new Error("No tabular data found. Make sure the first row contains column headers.");
      }
      // Don't analyze yet - stage the file for a confirmation step so the user can check it's the right
      // file/sheet, add context, and start deliberately. Analysis begins from handleStartAnalysis().
      setPending({ file, table: result.table, sources: result.sources, sourceId: result.sourceId, sourceKind: result.sourceKind });
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not read that file.");
    } finally {
      setBusy(false);
      setStage(null);
      setReadInfo(null);
    }
  }

  // Re-read a different sheet/table of the staged file during the review step, so the user picks the
  // right one BEFORE analyzing (not after).
  async function handleStageSource(id: string) {
    if (!pending || id === pending.sourceId || stagingSource) return;
    setStagingSource(true);
    try {
      const result = await parseFile(pending.file, undefined, id);
      if (!result.table.columns.length || !result.table.rowCount) {
        throw new Error("That selection has no usable tabular data.");
      }
      setPending({ ...pending, table: result.table, sourceId: id, sourceKind: result.sourceKind, sources: result.sources });
    } catch (e) {
      setToast({ text: e instanceof Error ? e.message : "Couldn't read that selection.", tone: "error" });
    } finally {
      setStagingSource(false);
    }
  }

  // The user confirmed the staged file - commit it as the source and run the pipeline.
  async function handleStartAnalysis() {
    if (!pending || stagingSource) return;
    setSourceFile(pending.file);
    setFileSources(pending.sources);
    setCurrentSourceId(pending.sourceId);
    setSourceKind(pending.sourceKind);
    const tbl = pending.table;
    setPending(null);
    await run(tbl);
  }

  // Discard the staged file and return to the uploader to pick a different one.
  function cancelPending() {
    setPending(null);
    setError(null);
    setReadInfo(null);
  }

  // Re-parse a different sheet/table from the same file and analyze it.
  async function handleSelectSource(id: string) {
    if (!sourceFile || id === currentSourceId || busy) return;
    setError(null);
    setBusy(true);
    setStage("Reading file");
    try {
      const result = await parseFile(sourceFile, undefined, id);
      if (!result.table.columns.length || !result.table.rowCount) {
        throw new Error("That selection has no usable tabular data.");
      }
      setCurrentSourceId(id);
      setSourceKind(result.sourceKind);
      await run(result.table); // resets column choices and manages busy/stage
    } catch (e) {
      setError(e instanceof Error ? e.message : "Couldn't read that selection.");
      setBusy(false);
      setStage(null);
    }
  }

  // Enrich the current table by joining another table/sheet from the same file on an auto-detected key.
  async function handleJoin() {
    if (!sourceFile || !sourceTable || !joinId || busy) return;
    setError(null);
    setBusy(true);
    setStage("Reading file");
    try {
      const result = await parseFile(sourceFile, undefined, joinId);
      const right = result.table;
      if (!right.columns.length || !right.rowCount) throw new Error("That table has no usable data to join.");
      const { suggestJoinKeys, joinTables } = await import("@/lib/join");
      const keys = suggestJoinKeys(sourceTable, right);
      if (!keys.length) {
        throw new Error(`No shared key found between “${sourceTable.name}” and “${right.name}”. Tables need a common column (e.g. an id) to join.`);
      }
      const k = keys[0];
      const joined = joinTables(sourceTable, right, k.leftKey, k.rightKey, "left");
      // The result is a new combined dataset - clear the single-source picker to avoid confusion.
      setFileSources([]);
      setCurrentSourceId("");
      setSourceKind(undefined);
      setJoinId("");
      setToast({ text: `Joined on ${k.leftKey} = ${k.rightKey} · ${joined.rowCount.toLocaleString()} rows`, tone: "info" });
      await run(joined);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Couldn't join those tables.");
      setBusy(false);
      setStage(null);
    }
  }

  function startSample(kind?: string) {
    // Samples have no underlying file/sheets - clear any picker state from a prior upload.
    setSourceFile(null);
    setFileSources([]);
    setCurrentSourceId("");
    setSourceKind(undefined);
    run(sampleTable(kind));
  }

  // Auto-run the sample when arriving from the landing page's "Try a sample" CTA (/analyze?demo=1).
  // The ref guard makes this fire exactly once - React StrictMode double-invokes mount effects in dev,
  // which would otherwise kick off two concurrent analyses.
  const demoRan = useRef(false);
  useEffect(() => {
    if (demoRan.current) return;
    if (typeof window !== "undefined" && new URLSearchParams(window.location.search).get("demo") === "1") {
      demoRan.current = true;
      startSample();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function reset() {
    setSpec(null);
    setPySpec(null);
    setPyConclusions(null);
    setTable(null);
    setError(null);
    setSourceTable(null);
    setSourceFile(null);
    setFileSources([]);
    setCurrentSourceId("");
    setSourceKind(undefined);
    setPending(null);
  }

  return (
    <main id="main-content" className="glow app-bg min-h-screen">
      <div className="app-aurora" aria-hidden>
        <span className="a1" /><span className="a2" /><span className="a3" />
      </div>
      <div className="relative z-10 mx-auto max-w-6xl px-4 py-10 sm:px-6">
        <header className="mb-8 flex items-center justify-between">
          <Link href="/" className="flex items-center gap-3">
            <BrandMark className="h-10 w-10" />
            <div>
              <h1 className="text-xl font-bold tracking-tight text-slate-50">MyAnalyst</h1>
              <p className="text-xs text-slate-400">Analyzer</p>
            </div>
          </Link>
          <div className="flex flex-wrap items-center justify-end gap-2">
          {spec ? (
            <div className="flex flex-wrap items-center justify-end gap-2">
              <button
                onClick={() => table && downloadCsv(table, spec.datasetName)}
                className="rounded-xl border border-slate-700 px-3 py-2 text-sm font-medium text-slate-200 transition hover:bg-slate-800/60"
                title="Download the cleaned, normalized data as CSV"
              >
                ⬇ Data
              </button>
              <button
                onClick={handleShare}
                className="rounded-xl border border-slate-700 px-3 py-2 text-sm font-medium text-slate-200 transition hover:bg-slate-800/60"
                title="Copy a read-only link to this dashboard"
              >
                🔗 Share
              </button>
              <button
                onClick={() => setPresenting(true)}
                className="rounded-xl border border-slate-700 px-3 py-2 text-sm font-medium text-slate-200 transition hover:bg-slate-800/60"
                title="Full-screen presenter mode for walking a room through the findings"
              >
                ▶ Present
              </button>
              <button
                onClick={() => handleExport("png")}
                disabled={!!exporting}
                className="rounded-xl border border-slate-700 px-3 py-2 text-sm font-medium text-slate-200 transition hover:bg-slate-800/60 disabled:opacity-50"
                title="Download the dashboard as a PNG image"
              >
                {exporting === "png" ? "Exporting…" : "⬇ PNG"}
              </button>
              <button
                onClick={() => handleExport("pdf")}
                disabled={!!exporting}
                className="rounded-xl border border-slate-700 px-3 py-2 text-sm font-medium text-slate-200 transition hover:bg-slate-800/60 disabled:opacity-50"
                title="Download the dashboard as a PDF image"
              >
                {exporting === "pdf" ? "Exporting…" : "⬇ PDF"}
              </button>
              <button
                onClick={() => handleExport("report")}
                disabled={!!exporting}
                className="rounded-xl border border-slate-700 px-3 py-2 text-sm font-medium text-slate-200 transition hover:bg-slate-800/60 disabled:opacity-50"
                title="Download a text-first, multi-page consultant report (PDF)"
              >
                {exporting === "report" ? "Exporting…" : "⬇ Report"}
              </button>
              <button
                onClick={() => handleExport("deck")}
                disabled={!!exporting}
                className="rounded-xl border border-slate-700 px-3 py-2 text-sm font-medium text-slate-200 transition hover:bg-slate-800/60 disabled:opacity-50"
                title="Download a slide deck (PDF)"
              >
                {exporting === "deck" ? "Exporting…" : "⬇ Deck"}
              </button>
              <button
                onClick={() => setBranding(true)}
                disabled={!!exporting}
                className="rounded-xl border border-slate-700 px-3 py-2 text-sm font-medium text-slate-200 transition hover:bg-slate-800/60 disabled:opacity-50"
                title="Brand your exported report and deck"
              >
                ✦ Brand
              </button>
              <button
                onClick={() => setAiKeyOpen(true)}
                className="rounded-xl border border-slate-700 px-3 py-2 text-sm font-medium text-slate-200 transition hover:bg-slate-800/60"
                title="Use your own LLM key to sharpen the narration"
              >
                ✦ AI
              </button>
              <button
                onClick={reset}
                className="rounded-xl bg-[#ff5740] px-4 py-2 text-sm font-medium text-white transition hover:bg-[#ff3b30]"
              >
                New analysis
              </button>
            </div>
          ) : (
            <Link href="/" className="text-sm text-slate-400 transition hover:text-slate-200">
              ← Home
            </Link>
          )}
          </div>
        </header>

        {/* Upload view: shown until a file is staged for review (or analysis is running). */}
        {!spec && !pending && (
          <div className="space-y-4">
            <Uploader onFile={handleFile} onSample={startSample} busy={busy} industry={industry} onIndustry={updateIndustry} />
            {busy && stage && <PipelineProgress stages={STAGES} current={stage} detail={readInfo ?? undefined} />}
            {error && <div className="card border-rose-500/40 bg-rose-500/5 p-4 text-sm text-rose-300">{error}</div>}

            <details className="card p-4 text-sm">
              <summary className="cursor-pointer font-medium text-slate-200">
                ✨ Add context <span className="font-normal text-slate-500">(optional) - tell us what this data is about</span>
              </summary>
              <p className="mt-2 text-xs text-slate-400">
                A line about your data or goal sharpens the analysis: it steers domain detection and the metrics
                we pick, and frames the “About this data” summary and insights around what you care about. Stored
                only in this browser - never uploaded anywhere.
              </p>

              <textarea
                value={jobDesc}
                onChange={(e) => updateContext(e.target.value)}
                rows={2}
                placeholder="e.g. I run a used-car dealership and want to understand why customers don't buy."
                className="mt-3 w-full rounded-xl border border-slate-700 bg-slate-900/60 px-4 py-2.5 text-sm text-slate-100 placeholder:text-slate-500 focus:border-[#ff5740] focus:outline-none"
              />
            </details>

            <HistoryList entries={history} onOpen={handleOpenHistory} onDelete={handleDeleteHistory} />
          </div>
        )}

        {/* Review step: a file is read but NOT yet analyzed. Confirm it's the right file/sheet, add
            context to maximize the conclusions, then start deliberately. */}
        {!spec && pending && (
          <div className="space-y-4">
            <div className="fade-up card space-y-5 p-5 sm:p-6">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <h2 className="text-lg font-semibold text-slate-100">Ready to analyze - does this look right?</h2>
                  <p className="mt-0.5 text-sm text-slate-400">Check the file (and sheet), add a line of context, then start. Nothing is analyzed until you click the button.</p>
                </div>
                <button
                  onClick={cancelPending}
                  className="shrink-0 rounded-xl border border-slate-700 px-3 py-2 text-sm font-medium text-slate-200 transition hover:bg-slate-800/60"
                >
                  Change file
                </button>
              </div>

              {/* File summary */}
              <div className="rounded-xl border border-slate-700 bg-slate-900/50 p-4">
                <div className="flex items-center gap-3">
                  <div className="grid h-10 w-10 shrink-0 place-items-center rounded-xl bg-[#ff5740]/15 text-[#ff5740]">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" className="h-5 w-5">
                      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" /><path d="M14 2v6h6" />
                    </svg>
                  </div>
                  <div className="min-w-0">
                    <p className="truncate font-medium text-slate-100">{pending.file.name}</p>
                    <p className="text-xs text-slate-400">
                      {fmtBytes(pending.file.size)} · {pending.table.rowCount.toLocaleString()} rows × {pending.table.columns.length} columns
                    </p>
                  </div>
                </div>

                {/* Pick the right sheet/table BEFORE analyzing (multi-sheet workbooks / multi-table DBs). */}
                {pending.sources.length > 1 && (
                  <div className="mt-3 flex flex-wrap items-center gap-2 border-t border-slate-800 pt-3 text-sm">
                    <span className="font-medium text-slate-300">{pending.sourceKind === "table" ? "Table" : "Sheet"}:</span>
                    <select
                      value={pending.sourceId}
                      onChange={(e) => handleStageSource(e.target.value)}
                      disabled={stagingSource}
                      aria-label={`Choose which ${pending.sourceKind === "table" ? "table" : "sheet"} to analyze`}
                      className="rounded-lg border border-slate-700 bg-slate-900/60 px-3 py-1.5 text-sm text-slate-100 focus:border-[#ff5740] focus:outline-none disabled:opacity-50"
                    >
                      {pending.sources.map((s) => (
                        <option key={s.id} value={s.id}>
                          {s.label} ({s.rowCount.toLocaleString()} rows)
                        </option>
                      ))}
                    </select>
                    <span className="text-xs text-slate-500">
                      {stagingSource ? "Reading…" : `${pending.sources.length} ${pending.sourceKind === "table" ? "tables" : "sheets"} in this file`}
                    </span>
                  </div>
                )}

                {/* Column preview so the user can confirm at a glance that it's the right data. */}
                <div className="mt-3 flex flex-wrap gap-1.5 border-t border-slate-800 pt-3">
                  {pending.table.columns.slice(0, 14).map((c) => (
                    <span key={c} className="rounded-md bg-slate-800/70 px-2 py-0.5 text-xs text-slate-300">{c}</span>
                  ))}
                  {pending.table.columns.length > 14 && (
                    <span className="rounded-md px-2 py-0.5 text-xs text-slate-500">+{pending.table.columns.length - 14} more</span>
                  )}
                </div>
              </div>

              {/* Context - emphasized, because it markedly improves the conclusions. */}
              <div className="rounded-xl border border-[#ff5740]/30 bg-[#ff5740]/5 p-4">
                <h3 className="flex items-center gap-2 text-sm font-semibold text-slate-100">
                  <span aria-hidden>⭐</span> Add context to maximize your conclusions
                </h3>
                <p className="mt-1 text-xs leading-relaxed text-slate-300/90">
                  A sentence or two about what this data is and what you want to learn makes a <strong>real</strong> difference -
                  it steers domain detection and the metrics we surface, and frames every insight, the “About this data”
                  story, and the AI’s answers around <em>your</em> goal. Stored only in this browser; never uploaded.
                </p>

                {/* Optional industry tag - another fast way to sharpen the analysis. */}
                <div className="mt-3">
                  <IndustryTagPicker value={industry} onChange={updateIndustry} label="industry:" justify="start" size="sm" />
                </div>

                <textarea
                  value={jobDesc}
                  onChange={(e) => updateContext(e.target.value)}
                  rows={2}
                  placeholder="e.g. I run a used-car dealership and want to understand why customers don't buy."
                  className="mt-3 w-full rounded-xl border border-slate-700 bg-slate-900/60 px-4 py-2.5 text-sm text-slate-100 placeholder:text-slate-500 focus:border-[#ff5740] focus:outline-none"
                />
                {!jobDesc.trim() && (
                  <p className="mt-1.5 text-[11px] text-amber-300/80">Tip: even a short note noticeably sharpens the results - it’s worth the 10 seconds.</p>
                )}
              </div>

              {/* Start */}
              <div className="flex flex-col-reverse items-stretch gap-3 sm:flex-row sm:items-center sm:justify-between">
                <button onClick={cancelPending} className="text-sm text-slate-400 transition hover:text-slate-200">
                  ← Choose a different file
                </button>
                <button
                  onClick={handleStartAnalysis}
                  disabled={stagingSource}
                  className="btn-press btn-shine rounded-xl bg-[#ff5740] px-6 py-3 text-sm font-semibold text-white shadow-lg shadow-[#ff5740]/25 transition hover:bg-[#ff3b30] disabled:opacity-50"
                >
                  🚀 Start analyzing
                </button>
              </div>
            </div>
          </div>
        )}

        {toast && (
          <div
            role="status"
            aria-live="polite"
            className={`fade-up mb-4 flex items-start justify-between gap-3 rounded-xl border px-4 py-2.5 text-sm ${
              toast.tone === "error"
                ? "border-rose-500/40 bg-rose-500/10 text-rose-200"
                : "border-[#ff5740]/30 bg-[#ff5740]/10 text-[#ff5740]"
            }`}
          >
            <span>{toast.text}</span>
            <button
              onClick={() => setToast(null)}
              aria-label="Dismiss"
              className="-mr-1 shrink-0 rounded-md px-1.5 leading-none opacity-60 transition hover:opacity-100"
            >
              ✕
            </button>
          </div>
        )}

        {spec && table?.sampledFrom && (
          <div className="mb-4 rounded-xl border border-amber-500/30 bg-amber-500/10 px-4 py-2.5 text-sm text-amber-200">
            Large file detected - analyzed a representative random sample of{" "}
            <strong>{table.rowCount.toLocaleString()}</strong> rows out of{" "}
            <strong>{table.sampledFrom.toLocaleString()}</strong>. The statistics are valid for the full
            dataset; exact totals would need the complete file.
          </div>
        )}

        {spec && fileSources.length > 1 && (
          <div className="mb-4 card flex flex-wrap items-center gap-2 p-3 text-sm">
            <span className="font-medium text-slate-300">{sourceKind === "table" ? "Table" : "Sheet"}:</span>
            <select
              value={currentSourceId}
              onChange={(e) => handleSelectSource(e.target.value)}
              disabled={busy}
              aria-label={`Choose which ${sourceKind === "table" ? "table" : "sheet"} to analyze`}
              className="rounded-lg border border-slate-700 bg-slate-900/60 px-3 py-1.5 text-sm text-slate-100 focus:border-[#ff5740] focus:outline-none disabled:opacity-50"
            >
              {fileSources.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.label} ({s.rowCount.toLocaleString()} rows)
                </option>
              ))}
            </select>
            <span className="text-xs text-slate-500">
              {fileSources.length} {sourceKind === "table" ? "tables" : "sheets"} in this file
            </span>

            {/* Join the current table with another from the same file (auto-detected key). */}
            <span className="ml-1 border-l border-slate-700 pl-3 font-medium text-slate-300">Join with:</span>
            <select
              value={joinId}
              onChange={(e) => setJoinId(e.target.value)}
              disabled={busy}
              aria-label="Choose a table to join with the current one"
              className="rounded-lg border border-slate-700 bg-slate-900/60 px-3 py-1.5 text-sm text-slate-100 focus:border-[#ff5740] focus:outline-none disabled:opacity-50"
            >
              <option value="">{sourceKind === "table" ? "another table…" : "another sheet…"}</option>
              {fileSources
                .filter((s) => s.id !== currentSourceId)
                .map((s) => (
                  <option key={s.id} value={s.id}>
                    {s.label}
                  </option>
                ))}
            </select>
            <button
              onClick={handleJoin}
              disabled={busy || !joinId}
              className="rounded-lg bg-[#ff5740]/90 px-3 py-1.5 text-sm font-medium text-white transition hover:bg-[#ff3b30] disabled:opacity-50"
            >
              Join
            </button>
          </div>
        )}

        {presenting && spec && <PresenterMode spec={mergedDashboardSpec(spec, pySpec)} conclusions={pyConclusions} onClose={() => setPresenting(false)} />}
        {branding && <BrandEditor onClose={() => setBranding(false)} />}
        {aiKeyOpen && <AiKeyEditor onClose={() => setAiKeyOpen(false)} />}

        {spec && table && (
          <ErrorBoundary label="dashboard">
            <div className={busy ? "pointer-events-none opacity-50 transition-opacity" : "transition-opacity"}>
              {/* The rich dashboard (same UI as the demo). The file header + tab nav sit at the very top;
                  the Groq conclusions (which read the Python KPIs + charts) render just beneath the tabs.
                  When the Python engine has run, its KPIs and charts populate it; otherwise the TS engine does. */}
              <DashboardView
                spec={mergedDashboardSpec(spec, pySpec)}
                table={table}
                conclusions={pyConclusions}
                innerRef={dashboardRef}
              />
            </div>
          </ErrorBoundary>
        )}

        <footer className="mt-16 space-y-2 border-t border-slate-800 pt-6 text-center text-xs text-slate-600">
          <p>MyAnalyst · Your data is processed securely and never stored. Insight narration is pluggable.</p>
          {/* Not-financial-advice disclaimer: only after an analysis exists, in the page footer. */}
          {spec && <p className="text-slate-500">{DISCLAIMER_TEXT}</p>}
        </footer>
      </div>
    </main>
  );
}
