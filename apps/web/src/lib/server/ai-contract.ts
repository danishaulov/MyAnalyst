// Shared input and numeric guards for the server narrator. No provider credentials enter evidence.
export class AiInputError extends Error {
  constructor(message: string, public status = 400) { super(message); }
}

const MAX_BYTES = 120_000;
const PROVIDERS = new Set(["anthropic", "groq", "openai", "gemini", "openrouter", "openai-compat"]);
const object = (v: unknown): v is Record<string, unknown> => !!v && typeof v === "object" && !Array.isArray(v);

export async function readAiRequest(req: Request): Promise<Record<string, unknown>> {
  if (Number(req.headers.get("content-length")) > MAX_BYTES) throw new AiInputError("AI context is too large.", 413);
  if (!req.body) throw new AiInputError("Provide a JSON object.");
  const reader = req.body.getReader();
  const chunks: Uint8Array[] = [];
  let bytes = 0;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      bytes += value.byteLength;
      if (bytes > MAX_BYTES) {
        await reader.cancel();
        throw new AiInputError("AI context is too large.", 413);
      }
      chunks.push(value);
    }
  } finally { reader.releaseLock(); }
  const buffer = new Uint8Array(bytes);
  let offset = 0;
  for (const chunk of chunks) { buffer.set(chunk, offset); offset += chunk.byteLength; }
  let body: unknown;
  try { body = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(buffer)); }
  catch { throw new AiInputError("Invalid JSON body."); }
  return validateAiRequest(body);
}

export function validateAiRequest(body: unknown): Record<string, unknown> {
  if (!object(body)) throw new AiInputError("Body must be a JSON object.");
  const visit = (v: unknown, depth: number): void => {
    if (depth > 12) throw new AiInputError("AI context is too deeply nested.");
    if (typeof v === "string" && v.length > 16_000) throw new AiInputError("AI context contains oversized text.", 413);
    if (typeof v === "number" && !Number.isFinite(v)) throw new AiInputError("Numbers must be finite.");
    if (Array.isArray(v)) {
      if (v.length > 300) throw new AiInputError("AI context contains too many items.", 413);
      v.forEach((x) => visit(x, depth + 1));
    } else if (object(v)) {
      if (Object.keys(v).length > 200) throw new AiInputError("Too many context fields.", 413);
      if (typeof v.csv === "string" || Array.isArray(v.rows) || Array.isArray(v.records)) throw new AiInputError("Send computed aggregates to the AI endpoint, not raw records.");
      Object.values(v).forEach((x) => visit(x, depth + 1));
    }
  };
  visit(body, 0);
  const task = body.task;
  if (task !== undefined && !["answer", "plan", "story", "humanize"].includes(String(task))) throw new AiInputError("Unknown AI task.");
  if (["answer", "plan"].includes(String(task)) && (typeof body.question !== "string" || !body.question.trim() || body.question.length > 2000)) throw new AiInputError("Provide a question of at most 2000 characters.");
  if (body.userContext !== undefined && typeof body.userContext !== "string") throw new AiInputError("userContext must be text.");
  if (task === "humanize" && (!Array.isArray(body.conclusions) || body.conclusions.some((c) => !object(c) || typeof c.id !== "string" || typeof c.text !== "string" || (c.detail !== undefined && typeof c.detail !== "string")))) throw new AiInputError("Invalid conclusions.");
  if (task === "story" && ((body.meta !== undefined && !object(body.meta)) || (body.draft !== undefined && !object(body.draft)))) throw new AiInputError("Invalid story metadata.");
  if (task === undefined && !Array.isArray(body.kpis)) throw new AiInputError("Body must be an InsightContext.");
  if (task === undefined) {
    for (const key of ["kpis", "columns", "correlations", "trends", "outliers", "categories", "groupComparisons", "associations", "concentration"]) {
      if (body[key] !== undefined && (!Array.isArray(body[key]) || (body[key] as unknown[]).some((v) => !object(v)))) throw new AiInputError(`Invalid ${key} context.`);
    }
  }
  if (body.byok != null) {
    const key = body.byok;
    if (!object(key) || typeof key.provider !== "string" || !PROVIDERS.has(key.provider)
        || typeof key.apiKey !== "string" || !key.apiKey.trim() || key.apiKey.length > 512
        || (key.model !== undefined && (typeof key.model !== "string" || key.model.length > 200))) throw new AiInputError("Invalid AI provider configuration.");
  }
  return body;
}

interface Claim { value: number; unit: string; tolerance: number; }
const SCALES: Record<string, number> = { k: 1e3, thousand: 1e3, m: 1e6, million: 1e6, b: 1e9, billion: 1e9, t: 1e12, trillion: 1e12 };
function claims(text: string): Claim[] {
  const out: Claim[] = [];
  const cleaned = text.replace(/(^|[;\n])\s*\d{1,2}\s*[.):\-]\s+/gm, "$1 ");
  const regex = /(?<![\w.])([$€£₪])?\s*(-?\d+(?:,\d{3})*(?:\.\d+)?|-?\.\d+)\s*(trillion|billion|million|thousand|[kmbt]\b)?\s*(%|percent\b)?/gi;
  for (const m of cleaned.matchAll(regex)) {
    const number = m[2].replace(/,/g, "");
    const scale = SCALES[(m[3] || "").toLowerCase()] || 1;
    const value = Number(number) * scale;
    const precision = (number.split(".")[1] || "").length;
    out.push({ value, unit: m[4] ? "%" : m[1] || "number", tolerance: Math.min(0.5 * 10 ** -precision * scale, Math.max(Math.abs(value) * 0.02, 1e-9)) });
  }
  return out;
}

export function hasSupportedNumbers(answer: string, evidence: unknown): boolean {
  const source: Claim[] = [];
  const visit = (v: unknown): void => {
    if (typeof v === "string") source.push(...claims(v));
    else if (typeof v === "number" && Number.isFinite(v)) source.push({ value: v, unit: "number", tolerance: 1e-9 });
    else if (Array.isArray(v)) v.forEach(visit);
    else if (object(v)) Object.values(v).forEach(visit);
  };
  visit(evidence);
  return claims(answer).every((c) => source.some((s) =>
    (c.unit === "number" || c.unit === s.unit) &&
    (Math.abs(c.value - s.value) <= Math.max(c.tolerance, s.tolerance, 1e-9) ||
      (c.unit === "%" && c.value >= 0 && Math.abs(c.value + s.value) <= Math.max(c.tolerance, s.tolerance, 1e-9)))
  ));
}
