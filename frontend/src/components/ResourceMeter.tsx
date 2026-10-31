import { useEffect, useRef, useState } from "react";
import {
  getSystemMemory,
  unloadGenModels,
  unloadModel,
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

/** "8s", "1m 05s" since `since` (epoch seconds); "" without one. The meter polls
 *  every second while anything loads, and each poll re-renders this. */
function elapsed(since: number | null | undefined): string {
  if (since == null) return "";
  const s = Math.max(0, Math.floor(Date.now() / 1000 - since));
  return s < 60 ? `${s}s` : `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, "0")}s`;
}

/** A small rotating ring — something is loading. Ollama can't say how far a load
 *  has got, so this shows that it's happening, with the elapsed time beside it,
 *  rather than a percentage it would have to invent. */
function Spinner() {
  return <span className="res-spin" aria-hidden="true" />;
}

/** "loading · 8s" with a spinner, for a row, a part, or the engine starting. */
function Loading({ label, since }: { label: string; since: number | null | undefined }) {
  const t = elapsed(since);
  return (
    <span className="res-badge loading" role="status">
      <Spinner />
      {label}
      {t && <span className="res-elapsed">· {t}</span>}
    </span>
  );
}

/** What a row is doing, when that's more than sitting loaded. */
function StateBadge({ state, since }: { state: LoadState; since: number | null }) {
  if (state === "loading") return <Loading label="loading" since={since} />;
  if (state === "unloading") return <span className="res-badge unloading">unloading…</span>;
  if (state === "active") return <span className="res-badge active">in use</span>;
  return null;
}

// How long an armed Offload button waits for the confirming click.
const ARM_MS = 4000;

/** Manual offload, on top of the automatic kind. Two clicks: the first arms it
 *  ("Confirm?"), the second within ARM_MS acts; waiting or leaving disarms it.
 *  `blocked` disables it, with the reason as its tooltip — nothing is offloaded
 *  mid-load or mid-request. */
function OffloadButton({ what, blocked, run }: {
  what: string;
  blocked: string | null;
  run: () => Promise<void>;
}) {
  const [armed, setArmed] = useState(false);
  const [pending, setPending] = useState(false);
  const [failed, setFailed] = useState<string | null>(null);

  useEffect(() => {
    if (!armed) return;
    const id = window.setTimeout(() => setArmed(false), ARM_MS);
    return () => window.clearTimeout(id);
  }, [armed]);

  const click = () => {
    if (!armed) {
      setFailed(null);
      setArmed(true);
      return;
    }
    setArmed(false);
    setPending(true);
    run()
      .catch((e) => setFailed(e instanceof Error ? e.message : String(e)))
      .finally(() => setPending(false));
  };

  return (
    <button
      type="button"
      className={`res-offload${armed ? " armed" : ""}`}
      disabled={blocked != null || pending}
      onClick={click}
      onMouseLeave={() => setArmed(false)}
      title={
        blocked ??
        (failed ? `Offload failed: ${failed}` : armed ? `Click again to confirm: ${what}` : what)
      }
    >
      {pending ? <Spinner /> : armed ? "Confirm?" : failed ? "failed" : "Offload"}
    </button>
  );
}

/** Why a row can't be offloaded right now, or null when it can. */
function offloadBlocked(state: LoadState): string | null {
  if (state === "loading") return "Still loading — offload once it's in memory";
  if (state === "active") return "In use — wait for it to finish";
  if (state === "unloading") return "Already unloading";
  return null;
}

function OllamaRow({ m, ollamaUrl, refresh }: {
  m: LoadedOllamaModel;
  ollamaUrl: string;
  refresh: () => void;
}) {
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
          <StateBadge state={m.state} since={m.since} />
          {m.state !== "loading" && <Placement pct={m.gpu_pct} />}
          <OffloadButton
            what={`Unload ${m.name} from memory now`}
            blocked={offloadBlocked(m.state)}
            run={() => unloadModel(ollamaUrl, m.name).then(refresh)}
          />
        </span>
      </div>
      <div className="res-model-meta">{meta.join(" · ")}</div>
    </div>
  );
}

/** `blocked` covers the whole engine: offloading one gen row frees them all, so
 *  any of them loading or in use holds every row's button. */
function GenRow({ m, blocked, refresh }: {
  m: LoadedGenModel;
  blocked: string | null;
  refresh: () => void;
}) {
  return (
    <div className="res-model">
      <div className="res-model-head">
        <span className="res-model-name">
          {m.video ? "🎬" : "🎨"} {m.label}
        </span>
        <span className="res-badges">
          <StateBadge state={m.state} since={m.since} />
          <OffloadButton
            what="Free all image/video models (ComfyUI can't unload just one)"
            blocked={blocked}
            run={() => unloadGenModels().then(refresh)}
          />
        </span>
      </div>
      <div className="res-model-meta">{m.file}</div>
    </div>
  );
}

function PartRow({ m }: { m: LoadedGenModel }) {
  return (
    <div className="res-part">
      <span className="res-part-name">
        <span className="res-part-role">{m.role}</span> {m.file}
      </span>
      {m.state === "loading" && <Loading label="loading" since={m.since} />}
    </div>
  );
}

function Section({ title, vram, children }: {
  title: string;
  vram?: number | null;
  children: React.ReactNode;
}) {
  return (
    <div className="res-section">
      <div className="res-section-title">
        {title}
        {vram != null && <span>{gb(vram)} GB VRAM</span>}
      </div>
      {children}
    </div>
  );
}

/** A type within an engine's section — "Vision · 1" — with its rows beneath. */
function Group({ title, count, children }: {
  title: string;
  count: number;
  children: React.ReactNode;
}) {
  return (
    <div className="res-group">
      <div className="res-group-title">
        {title} <span>· {count}</span>
      </div>
      {children}
    </div>
  );
}

const OLLAMA_GROUPS = [
  ["vision", "Vision"],
  ["chat", "Chat"],
  ["embedding", "Embedding (documents)"],
] as const;

/** Every model either engine has loaded or is loading. One section per engine,
 *  and inside it a group per kind of model: Ollama's by capability (vision /
 *  chat / embedding), ComfyUI's transformers by image or video, with the
 *  encoders, VAEs and LoRAs they share listed apart. The image side is the
 *  backend's memory of what it asked ComfyUI for, plus the run in progress,
 *  since ComfyUI can't be asked directly. */
function LoadedPanel({ mem, other, ollamaUrl, refresh }: {
  mem: SystemMemory;
  other: number;
  ollamaUrl: string;
  refresh: () => void;
}) {
  const ollama = mem.ollama_models;
  const gen = mem.gen_models;
  const genModels = gen?.models ?? [];
  const genBusy = genModels.find((m) => m.state !== "loaded");
  const genBlocked = genBusy ? offloadBlocked(genBusy.state) : null;
  const image = genModels.filter((m) => m.kind === "image");
  const video = genModels.filter((m) => m.kind === "video");
  const parts = genModels.filter((m) => m.role !== "transformer");
  return (
    <div className="res-pop" role="dialog" aria-label="Loaded models">
      <Section title="Vision / chat (Ollama)" vram={mem.ollama_gb}>
        {ollama == null ? (
          <div className="res-empty">Ollama unreachable</div>
        ) : ollama.length === 0 ? (
          <div className="res-empty">None loaded</div>
        ) : (
          OLLAMA_GROUPS.map(([kind, title]) => {
            const rows = ollama.filter((m) => m.kind === kind);
            return (
              rows.length > 0 && (
                <Group key={kind} title={title} count={rows.length}>
                  {rows.map((m) => (
                    <OllamaRow key={m.name} m={m} ollamaUrl={ollamaUrl} refresh={refresh} />
                  ))}
                </Group>
              )
            );
          })
        )}
      </Section>
      <Section title="Image / video (ComfyUI)" vram={mem.flux_gb}>
        {gen == null ? (
          <div className="res-empty">Engine not running</div>
        ) : gen.status === "starting" ? (
          <div className="res-empty">
            <Loading label="starting" since={gen.since} />
          </div>
        ) : !gen.known ? (
          <div className="res-empty">
            Engine running — contents unknown until the next generation
          </div>
        ) : genModels.length === 0 ? (
          <div className="res-empty">None loaded</div>
        ) : (
          <>
            {image.length > 0 && (
              <Group title="Image" count={image.length}>
                {image.map((m) => (
                  <GenRow key={m.file} m={m} blocked={genBlocked} refresh={refresh} />
                ))}
              </Group>
            )}
            {video.length > 0 && (
              <Group title="Video" count={video.length}>
                {video.map((m) => (
                  <GenRow key={m.file} m={m} blocked={genBlocked} refresh={refresh} />
                ))}
              </Group>
            )}
            {parts.length > 0 && (
              <Group title="Components" count={parts.length}>
                {parts.map((m) => (
                  <PartRow key={m.file} m={m} />
                ))}
              </Group>
            )}
          </>
        )}
      </Section>
      {other >= 0.5 && (
        <Section title="Other GPU use" vram={other}>
          <div className="res-empty">Other processes, or the desktop</div>
        </Section>
      )}
    </div>
  );
}

/** The line under the bars: what's changing right now, else how many are loaded. */
function summary(mem: SystemMemory): { text: string; tone: "" | "warn" | "busy"; spin?: boolean } {
  const ollama = mem.ollama_models ?? [];
  const gen = mem.gen_models?.models ?? [];
  const named = (ms: { name?: string; label?: string }[]) =>
    ms.length === 1 ? (ms[0].name ?? ms[0].label?.split(" — ")[0]) : `${ms.length} models`;
  // Timed from the load that has been going longest.
  const took = (ms: { since: number | null }[]) => {
    const first = Math.min(...ms.map((m) => m.since ?? Infinity));
    return Number.isFinite(first) ? ` · ${elapsed(first)}` : "";
  };
  const loading = [
    ...ollama.filter((m) => m.state === "loading"),
    ...gen.filter((m) => m.state === "loading" && m.role === "transformer"),
  ];
  if (loading.length)
    return { text: `loading ${named(loading)}${took(loading)}`, tone: "busy", spin: true };
  const parts = gen.filter((m) => m.state === "loading");
  if (parts.length)
    return { text: `loading image/video components${took(parts)}`, tone: "busy", spin: true };
  if (mem.gen_models?.status === "starting")
    return {
      text: `starting image engine${took([mem.gen_models])}`,
      tone: "busy",
      spin: true,
    };
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
  // An offload re-polls at once, so its "unloading" shows without waiting a tick.
  const refresh = () => {
    getSystemMemory(ollamaUrl)
      .then(setMem)
      .catch(() => {});
  };
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
        <span className={`res-loaded${line.tone ? ` ${line.tone}` : ""}`}>
          {line.spin && <Spinner />}
          {line.text}
        </span>
      </button>
      {open && <LoadedPanel mem={mem} other={other} ollamaUrl={ollamaUrl} refresh={refresh} />}
    </div>
  );
}
