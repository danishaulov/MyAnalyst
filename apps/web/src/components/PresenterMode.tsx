"use client";

import { useEffect, useMemo, useState } from "react";
import type { DashboardSpec } from "@/lib/types";
import type { PyConclusions } from "@/lib/py-engine";
import { buildExecutiveSummary } from "@/lib/report";
import { Portal } from "./Portal";

// Full-screen presenter mode: the dashboard's key story as large, keyboard-navigable slides - for
// walking a room through the findings. ← / → / space to move, Esc to exit. Renders from the spec.

interface Slide {
  kind: "title" | "summary" | "action" | "insight";
  title: string;
  body: string[];
  badge?: string;
  eyebrow?: string;
}

function buildSlides(spec: DashboardSpec, conclusions?: PyConclusions | null): Slide[] {
  const slides: Slide[] = [];
  slides.push({
    kind: "title",
    eyebrow: spec.domain.domain.replace(/-/g, " "),
    title: spec.datasetName,
    body: [
      `${spec.rowCount.toLocaleString()} rows · ${spec.profiles.length} columns`,
      ...(spec.computeScope?.sampled ? [`Server KPIs and charts use ${spec.computeScope.analyzedRows.toLocaleString()} of ${spec.computeScope.sourceRows.toLocaleString()} rows; totals describe the sample.`] : []),
    ],
    badge: spec.quality ? `Data quality ${spec.quality.grade} · ${spec.quality.score}/100` : undefined,
  });

  // The headline read is the AI conclusions (the same "AI conclusions" panel shown on the dashboard):
  // a bottom-line slide, then one slide per conclusion. Fall back to the templated executive summary +
  // findings only when the AI engine hasn't produced conclusions (backend unreachable / LLM off).
  const aiPoints = conclusions?.conclusions?.filter((t) => t.trim()) ?? [];
  if (conclusions && (aiPoints.length || conclusions.bottomLine)) {
    const lead = [conclusions.bottomLine, conclusions.summary].filter((t): t is string => !!t && !!t.trim());
    if (lead.length) slides.push({ kind: "summary", eyebrow: "AI conclusions", title: "The big picture", body: lead });
    aiPoints.slice(0, 6).forEach((t, i) => slides.push({ kind: "insight", eyebrow: `Conclusion ${i + 1}`, title: t, body: [] }));
  } else {
    const summary = buildExecutiveSummary(spec);
    if (summary.length) slides.push({ kind: "summary", eyebrow: "Executive summary", title: "The big picture", body: summary });
    (spec.insights ?? []).slice(0, 4).forEach((ins) => slides.push({ kind: "insight", eyebrow: "Finding", title: ins.text, body: [] }));
  }

  // Action slides: prefer the AI engine's actions, else the templated ones.
  const actions = conclusions?.actions?.length ? conclusions.actions.map((a) => ({ title: a.title, detail: a.detail, impact: "" })) : (spec.actions ?? []);
  actions.forEach((a, i) => slides.push({ kind: "action", eyebrow: `Action ${i + 1}`, title: a.title, body: [a.detail], badge: a.impact ? `${a.impact} impact` : undefined }));
  return slides;
}

const IMPACT_TONE: Record<string, string> = {
  "high impact": "text-rose-300",
  "medium impact": "text-amber-300",
  "low impact": "text-slate-400",
};

export function PresenterMode({ spec, conclusions, onClose }: { spec: DashboardSpec; conclusions?: PyConclusions | null; onClose: () => void }) {
  const slides = useMemo(() => buildSlides(spec, conclusions), [spec, conclusions]);
  const [i, setI] = useState(0);
  const last = slides.length - 1;

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      else if (e.key === "ArrowRight" || e.key === " ") setI((v) => Math.min(last, v + 1));
      else if (e.key === "ArrowLeft") setI((v) => Math.max(0, v - 1));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose, last]);

  const s = slides[i] ?? slides[0];

  return (
    <Portal>
    <div className="fixed inset-0 z-50 flex flex-col bg-gradient-to-b from-slate-950 to-slate-900 text-slate-100" role="dialog" aria-modal="true" aria-label="Presenter mode">
      <div className="flex items-center justify-between px-6 py-4 text-xs text-slate-500">
        <span>
          {i + 1} / {slides.length}
        </span>
        <button onClick={onClose} className="rounded-lg px-3 py-1 text-slate-400 transition hover:bg-white/5 hover:text-slate-200" aria-label="Exit presenter mode">
          Esc to exit ✕
        </button>
      </div>

      <div key={i} className="fade-up flex flex-1 flex-col items-center justify-center px-8 text-center sm:px-16">
        <div className="w-full max-w-3xl">
          {s.eyebrow && <p className="mb-4 text-xs font-semibold uppercase tracking-[0.24em] text-[#ff5740]">{s.eyebrow}</p>}
          <h2 className={`font-bold tracking-tight ${s.kind === "title" ? "text-5xl sm:text-6xl" : "text-3xl leading-tight sm:text-4xl"}`}>{s.title}</h2>
          {s.badge && (
            <span className={`mt-4 inline-block rounded-full bg-white/5 px-3 py-1 text-sm font-semibold ${IMPACT_TONE[s.badge.toLowerCase()] ?? "text-[#ff5740]"}`}>{s.badge}</span>
          )}
          {s.body.map((p, j) => (
            <p key={j} className="mt-5 text-lg leading-relaxed text-slate-300 sm:text-xl">
              {p}
            </p>
          ))}
        </div>
      </div>

      <div className="flex items-center justify-between px-6 py-5">
        <button onClick={() => setI((v) => Math.max(0, v - 1))} disabled={i === 0} className="rounded-full border border-white/10 px-5 py-2 text-sm transition hover:bg-white/5 disabled:opacity-30">
          ← Prev
        </button>
        <div className="flex gap-1.5">
          {slides.map((_, j) => (
            <button key={j} onClick={() => setI(j)} aria-label={`Slide ${j + 1}`} className={`h-2 rounded-full transition-all ${j === i ? "w-6 bg-[#ff5740]" : "w-2 bg-white/20 hover:bg-white/40"}`} />
          ))}
        </div>
        <button onClick={() => setI((v) => Math.min(last, v + 1))} disabled={i === last} className="rounded-full border border-white/10 px-5 py-2 text-sm transition hover:bg-white/5 disabled:opacity-30">
          Next →
        </button>
      </div>
    </div>
    </Portal>
  );
}
