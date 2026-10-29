import { useEffect, useRef, useState } from "react";
import {
  getSystemMemory,
  type LoadState,
  type LoadedGenModel,
  type LoadedOllamaModel,
  type SystemMemory,
} from "../api";

const POLL_MS = 3000;
// While a model is loading or unloading, so the change shows as it happens.
const BUSY_POLL_MS = 1000;

interface Props {
  ollamaUrl: string;
}

const gb = (n: number) => n.toFixed(1);

/** Same thresholds as the context meter, so the two read the same way. */
function color(ratio: number): string {
  return ratio >= 0.85 ? "var(--red)" : ratio >= 0.6 ? "var(--amber)" : "var(--green)";
}

function Bar({ label, used, total, title }: {
  label: string;
  used: number;
  total: number;
  title: string;
}) {
  const ratio = Math.min(1, total > 0 ? used / total : 0);
  return (
    <div className="res-bar" title={title}>
      <span className="res-label">{label}</span>
      <span className="res-track">
        <span
          className="res-fill"
          style={{ width: `${ratio * 100}%`, background: color(ratio) }}
        />
      </span>
      <span className="res-num">
        {gb(used)}/{gb(total)}
      </span>
    </div>
  );
}

/** "in 24m" until Ollama unloads a model; null when it never will (or no time). */
function unloadsIn(expiresAt: string | null): string | null {
  if (!expiresAt) return null;
  const ms = Date.parse(expiresAt) - Date.now();
  // Ollama reports a far-future date for keep_alive -1 (never unload).
  if (Number.isNaN(ms) || ms > 365 * 24 * 3600e3) return null;
  if (ms <= 0) return "unloading";
  const min = Math.round(ms / 60e3);
  return min < 1 ? "unloads in <1m" : min < 60 ? `unloads in ${min}m` : `unloads in ${Math.round(min / 60)}h`;
}

/** Where a loaded model runs: all on the GPU, or split with the CPU (slow). */
function Placement({ pct }: { pct: number | null }) {
  if (pct == null) return null;
  if (pct >= 100) return <span className="res-badge ok">GPU</span>;
  if (pct <= 0) return <span className="res-badge bad">CPU</span>;
  return (
    <span
      className="res-badge warn"
      title="Didn't fit in VRAM, so part of it runs on the CPU — much slower."
    >
      {pct}% GPU · {100 - pct}% CPU
    </span>
  );
}

/** What a row is doing, when that's more than sitting loaded. */
function StateBadge({ state }: { state: LoadState }) {
  if (state === "loading") return <span className="res-badge loading">⏳ loading…</span>;
  if (state === "unloading") return <span className="res-badge unloading">unloading…</span>;
  if (state === "active") return <span className="res-badge active">in use</span>;
  return null;
}

function OllamaRow({ m }: { m: LoadedOllamaModel }) {
  const meta =
    m.state === "loading"
      ? ["reading from disk into memory"]
      : [
          m.size_gb != null ? `${gb(m.size_gb)} GB` : null,
          m.context ? `${m.context.toLocaleString()} ctx` : null,
          m.state === "unloading" ? null : unloadsIn(m.expires_at),
        ].filter(Boolean);
  return (
    <div className="res-model">
      <div className="res-model-head">
        <span className="res-model-name">{m.name}</span>
        <span className="res-badges">
          <StateBadge state={m.state} />
          {m.state !== "loading" && <Placement pct={m.gpu_pct} />}
        </span>
      </div>
      <div className="res-model-meta">{meta.join(" · ")}</div>
    </div>
  );
}

function GenRow({ m }: { m: LoadedGenModel }) {
  return (
    <div className="res-model">
      <div className="res-model-head">
        <span className="res-model-name">
          {m.video ? "🎬" : "🎨"} {m.label}
        </span>
        <span className="res-badges">
          <StateBadge state={m.state} />
        </span>
      </div>
      <div className="res-model-meta">{m.file}</div>
    </div>
  );
}

function PartRow({ m }: { m: LoadedGenModel }) {
  return (
    <div className="res-part">
      <span className="res-part-role">{m.role}</span> {m.file}
      {m.state === "loading" && <span className="res-part-state"> ⏳</span>}
    </div>
  );
}

function Section({ title, count, vram, children }: {
  title: string;
  count: number;
  vram?: number;
  children: React.ReactNode;
}) {
  return (
    <div className="res-section">
      <div className="res-section-title">
        <span className="res-section-name">
          {title} <span className="res-count">· {count}</span>
        </span>
        {vram != null && vram > 0 && <span>{gb(vram)} GB VRAM</span>}
      </div>
      {children}
    </div>
  );
}

const OLLAMA_SECTIONS = [
  ["vision", "Vision"],
  ["chat", "Chat"],
  ["embedding", "Embedding (documents)"],
] as const;

/** Every model either engine has loaded or is loading, grouped by what it's for:
 *  Ollama's by capability (vision / chat / embedding), ComfyUI's transformers by
 *  image or video, with the encoders, VAEs and LoRAs they share listed apart.
 *  The image side is the backend's memory of what it asked ComfyUI for, plus the
 *  run in progress, since ComfyUI can't be asked directly. */
function LoadedPanel({ mem, other }: { mem: SystemMemory; other: number }) {
  const ollama = mem.ollama_models ?? [];
  const gen = mem.gen_models;
  const genModels = gen?.models ?? [];
  const image = genModels.filter((m) => m.kind === "image");
  const video = genModels.filter((m) => m.kind === "video");
  const parts = genModels.filter((m) => m.role !== "transformer");
  const empty = ollama.length === 0 && genModels.length === 0;
  const notes = [
    mem.ollama_models == null && "Ollama unreachable — chat models unknown",
    gen == null && "Image/video engine not running",
    gen?.status === "starting" && "⏳ Image/video engine starting…",
    gen && gen.status === "ready" && !gen.known &&
      "Image/video engine running — contents unknown until the next generation",
    mem.flux_gb != null && mem.flux_gb > 0.05 &&
      `Image/video engine holds ${gb(mem.flux_gb)} GB VRAM`,
    other >= 0.5 && `Other GPU use (other processes, the desktop): ${gb(other)} GB`,
  ].filter(Boolean) as string[];
  return (
    <div className="res-pop" role="dialog" aria-label="Loaded models">
      {empty && <div className="res-empty">No models loaded</div>}
      {OLLAMA_SECTIONS.map(([kind, title]) => {
        const rows = ollama.filter((m) => m.kind === kind);
        if (!rows.length) return null;
        return (
          <Section
            key={kind}
            title={title}
            count={rows.length}
            vram={rows.reduce((t, m) => t + (m.vram_gb ?? 0), 0)}
          >
            {rows.map((m) => (
              <OllamaRow key={m.name} m={m} />
            ))}
          </Section>
        );
      })}
      {image.length > 0 && (
        <Section title="Image" count={image.length}>
          {image.map((m) => (
            <GenRow key={m.file} m={m} />
          ))}
        </Section>
      )}
      {video.length > 0 && (
        <Section title="Video" count={video.length}>
          {video.map((m) => (
            <GenRow key={m.file} m={m} />
          ))}
        </Section>
      )}
      {parts.length > 0 && (
        <Section title="Image/video components" count={parts.length}>
          {parts.map((m) => (
            <PartRow key={m.file} m={m} />
          ))}
        </Section>
      )}
      {notes.length > 0 && (
        <div className="res-notes">
          {notes.map((n) => (
            <div key={n}>{n}</div>
          ))}
        </div>
      )}
    </div>
  );
}

/** The line under the bars: what's changing right now, else how many are loaded. */
function summary(mem: SystemMemory): { text: string; tone: "" | "warn" | "busy" } {
  const ollama = mem.ollama_models ?? [];
  const gen = mem.gen_models?.models ?? [];
  const named = (ms: { name?: string; label?: string }[]) =>
    ms.length === 1 ? (ms[0].name ?? ms[0].label?.split(" — ")[0]) : `${ms.length} models`;
  const loading = [
    ...ollama.filter((m) => m.state === "loading"),
    ...gen.filter((m) => m.state === "loading" && m.role === "transformer"),
  ];
  if (loading.length) return { text: `⏳ loading ${named(loading)}…`, tone: "busy" };
  if (gen.some((m) => m.state === "loading"))
    return { text: "⏳ loading image/video components…", tone: "busy" };
  if (mem.gen_models?.status === "starting")
    return { text: "⏳ starting image engine…", tone: "busy" };
  const unloading = ollama.filter((m) => m.state === "unloading");
  if (unloading.length) return { text: `unloading ${named(unloading)}…`, tone: "busy" };
  const count = ollama.length + gen.filter((m) => m.role === "transformer").length;
  const offloaded = ollama.some((m) => m.gpu_pct != null && m.gpu_pct < 100);
  const text = count === 0 ? "no models loaded" : `${count} model${count === 1 ? "" : "s"} loaded`;
  return { text: `${offloaded ? "⚠ " : ""}${text} ▾`, tone: offloaded ? "warn" : "" };
}

/** Whether anything is mid-change, so the meter polls faster to show it. */
function busy(mem: SystemMemory | null): boolean {
  if (!mem) return false;
  return (
    mem.gen_models?.status === "starting" ||
    (mem.ollama_models ?? []).some((m) => m.state === "loading" || m.state === "unloading") ||
    (mem.gen_models?.models ?? []).some((m) => m.state === "loading")
  );
}

/** Live VRAM and RAM use, polled while the tab is visible. RAM matters as much
 *  as VRAM here: every model is staged through it on the way to the GPU, and
 *  running out gets the FLUX engine killed mid-load. Hidden when the backend
 *  can't read the GPU (no nvidia-smi), since half a meter would mislead.
 *  Clicking it lists the models behind those numbers. */
export default function ResourceMeter({ ollamaUrl }: Props) {
  const [mem, setMem] = useState<SystemMemory | null>(null);
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const fast = busy(mem);
  useEffect(() => {
    let alive = true;
    const poll = () => {
      if (document.visibilityState !== "visible") return;
      getSystemMemory(ollamaUrl)
        .then((m) => alive && setMem(m))
        .catch(() => {}); // a missed sample just keeps the last one
    };
    poll();
    const id = window.setInterval(poll, fast ? BUSY_POLL_MS : POLL_MS);
    document.addEventListener("visibilitychange", poll);
    return () => {
      alive = false;
      window.clearInterval(id);
      document.removeEventListener("visibilitychange", poll);
    };
  }, [ollamaUrl, fast]);

  if (!mem?.gpu || !mem.ram) return null;
  const { gpu, ram } = mem;
  const flux = mem.flux_gb ?? 0;
  const ollama = mem.ollama_gb ?? 0;
  const other = Math.max(0, gpu.used_gb - flux - ollama);
  const line = summary(mem);
  return (
    <div className="res-wrap" ref={rootRef}>
      <button
        type="button"
        className={`res-meter${open ? " open" : ""}`}
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
      >
        <Bar
          label="VRAM"
          used={gpu.used_gb}
          total={gpu.total_gb}
          title={
            `GPU memory: ${gb(gpu.used_gb)} of ${gb(gpu.total_gb)} GB\n` +
            `Image/video engine: ${mem.flux_gb == null ? "not running" : `${gb(flux)} GB`}\n` +
            `Vision model: ${gb(ollama)} GB\n` +
            `Other: ${gb(other)} GB`
          }
        />
        <Bar
          label="RAM"
          used={ram.used_gb}
          total={ram.total_gb}
          title={
            `System memory: ${gb(ram.used_gb)} of ${gb(ram.total_gb)} GB ` +
            `(${gb(ram.available_gb)} GB free)\n` +
            `Swap: ${ram.swap_gb ? `${gb(ram.swap_gb)} GB` : "none"}\n` +
            `If this fills up, the system kills the image/video engine.`
          }
        />
        <span className={`res-loaded${line.tone ? ` ${line.tone}` : ""}`}>{line.text}</span>
      </button>
      {open && <LoadedPanel mem={mem} other={other} />}
    </div>
  );
}
