import { useEffect, useState } from "react";
import { getSystemMemory, type SystemMemory } from "../api";

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

/** Live VRAM and RAM use, polled while the tab is visible. RAM matters as much
 *  as VRAM here: every model is staged through it on the way to the GPU, and
 *  running out gets the FLUX engine killed mid-load. Hidden when the backend
 *  can't read the GPU (no nvidia-smi), since half a meter would mislead. */
export default function ResourceMeter({ ollamaUrl }: Props) {
  const [mem, setMem] = useState<SystemMemory | null>(null);

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
  return (
    <div className="res-meter">
      <Bar
        label="VRAM"
        used={gpu.used_gb}
        total={gpu.total_gb}
        title={
          `GPU memory: ${gb(gpu.used_gb)} of ${gb(gpu.total_gb)} GB\n` +
          `FLUX engine: ${mem.flux_gb == null ? "not running" : `${gb(flux)} GB`}\n` +
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
          `If this fills up, the system kills the FLUX engine.`
        }
      />
    </div>
  );
}
