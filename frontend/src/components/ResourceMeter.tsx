import { useEffect, useRef, useState } from "react";
import {
  getSystemMemory,
  type LoadedGenModel,
  type LoadedOllamaModel,
  type SystemMemory,
} from "../api";

const POLL_MS = 3000;

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

/** Where a model runs: all on the GPU, or split with the CPU (slow). */
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

function OllamaRow({ m }: { m: LoadedOllamaModel }) {
  const meta = [
    `${gb(m.size_gb)} GB`,
    m.context ? `${m.context.toLocaleString()} ctx` : null,
    unloadsIn(m.expires_at),
  ].filter(Boolean);
  return (
    <div className="res-model">
      <div className="res-model-head">
        <span className="res-model-name">{m.name}</span>
        <Placement pct={m.gpu_pct} />
      </div>
      <div className="res-model-meta">{meta.join(" · ")}</div>
    </div>
  );
}

function GenRows({ models }: { models: LoadedGenModel[] }) {
  const main = models.filter((m) => m.role === "transformer");
  const parts = models.filter((m) => m.role !== "transformer");
  return (
    <>
      {main.map((m) => (
        <div key={m.file} className="res-model">
          <div className="res-model-head">
            <span className="res-model-name">
              {m.video ? "🎬" : "🎨"} {m.label}
            </span>
            <span className={`res-badge ${m.video ? "video" : "ok"}`}>
              {m.video ? "video" : "image"}
            </span>
          </div>
          <div className="res-model-meta">{m.file}</div>
        </div>
      ))}
      {parts.map((m) => (
        <div key={m.file} className="res-part">
          <span className="res-part-role">{m.role}</span> {m.file}
        </div>
      ))}
    </>
  );
}

/** What each engine has loaded right now — the vision/chat models in Ollama and
 *  the image/video models in ComfyUI. The image side is the backend's memory of
 *  what it asked ComfyUI for, since ComfyUI can't be asked directly. */
function LoadedPanel({ mem, other }: { mem: SystemMemory; other: number }) {
  const ollama = mem.ollama_models;
  const gen = mem.gen_models;
  return (
    <div className="res-pop" role="dialog" aria-label="Loaded models">
      <div className="res-section">
        <div className="res-section-title">
          Vision / chat (Ollama)
          {mem.ollama_gb != null && <span>{gb(mem.ollama_gb)} GB VRAM</span>}
        </div>
        {ollama == null ? (
          <div className="res-empty">Ollama unreachable</div>
        ) : ollama.length === 0 ? (
          <div className="res-empty">None loaded</div>
        ) : (
          ollama.map((m) => <OllamaRow key={m.name} m={m} />)
        )}
      </div>
      <div className="res-section">
        <div className="res-section-title">
          Image / video (ComfyUI)
          {mem.flux_gb != null && <span>{gb(mem.flux_gb)} GB VRAM</span>}
        </div>
        {gen == null ? (
          <div className="res-empty">Engine not running</div>
        ) : !gen.known ? (
          <div className="res-empty">
            Engine running — contents unknown until the next generation
          </div>
        ) : gen.models.length === 0 ? (
          <div className="res-empty">None loaded</div>
        ) : (
          <GenRows models={gen.models} />
        )}
      </div>
      {other >= 0.5 && (
        <div className="res-section">
          <div className="res-section-title">
            Other GPU use<span>{gb(other)} GB VRAM</span>
          </div>
          <div className="res-empty">Other processes, or the desktop</div>
        </div>
      )}
    </div>
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

  useEffect(() => {
    let alive = true;
    const poll = () => {
      if (document.visibilityState !== "visible") return;
      getSystemMemory(ollamaUrl)
        .then((m) => alive && setMem(m))
        .catch(() => {}); // a missed sample just keeps the last one
    };
    poll();
    const id = window.setInterval(poll, POLL_MS);
    document.addEventListener("visibilitychange", poll);
    return () => {
      alive = false;
      window.clearInterval(id);
      document.removeEventListener("visibilitychange", poll);
    };
  }, [ollamaUrl]);

  if (!mem?.gpu || !mem.ram) return null;
  const { gpu, ram } = mem;
  const flux = mem.flux_gb ?? 0;
  const ollama = mem.ollama_gb ?? 0;
  const other = Math.max(0, gpu.used_gb - flux - ollama);
  const loaded =
    (mem.ollama_models?.length ?? 0) +
    (mem.gen_models?.models.filter((m) => m.role === "transformer").length ?? 0);
  const offloaded = mem.ollama_models?.some((m) => m.gpu_pct != null && m.gpu_pct < 100);
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
        <span className={`res-loaded${offloaded ? " warn" : ""}`}>
          {offloaded && "⚠ "}
          {loaded === 0 ? "no models loaded" : `${loaded} model${loaded === 1 ? "" : "s"} loaded`} ▾
        </span>
      </button>
      {open && <LoadedPanel mem={mem} other={other} />}
    </div>
  );
}
