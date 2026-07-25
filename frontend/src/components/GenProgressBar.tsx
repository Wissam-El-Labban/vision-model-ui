import { useEffect, useState } from "react";
import type { GenProgress } from "../types";

/** How long the backend may stay silent before the bar says so.
 *
 * Loading is legitimately quiet: ComfyUI reads tens of GB off disk between
 * accepting the prompt and emitting step 1, and says nothing the whole time. So
 * the pre-sampling threshold is much looser than the between-steps one — a
 * 30-second gap while sampling is odd, while loading it is normal. */
const QUIET_LOADING_MS = 180_000;
const QUIET_SAMPLING_MS = 90_000;

function short(ms: number): string {
  const s = Math.max(0, Math.floor(ms / 1000));
  return s < 60 ? `${s}s` : `${Math.floor(s / 60)}m${String(s % 60).padStart(2, "0")}s`;
}

/** A thin liveness bar under the conversation.
 *
 * Determinate once the sampler reports steps, indeterminate before that. The
 * indeterminate sweep alone can't tell "loading" from "wedged" — both animate —
 * so the elapsed and quiet-for readouts carry that: only a real backend event
 * moves `updatedAt`, and once the gap passes the threshold the bar goes amber. */
export default function GenProgressBar({ progress }: { progress: GenProgress | null }) {
  const active = progress !== null;
  // Elapsed and quiet-for are read off the clock rather than off props, because
  // the stretch this bar exists to describe is precisely the one where no props
  // arrive. Without its own tick the display would freeze exactly when it matters.
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!active) return;
    setNow(Date.now());
    const id = setInterval(() => setNow(Date.now()), 500);
    return () => clearInterval(id);
  }, [active]);

  if (!progress) return null;

  const determinate = progress.total > 0;
  const pct = determinate
    ? Math.min(100, (progress.step / progress.total) * 100)
    : 0;
  const quiet = now - progress.updatedAt;
  const stalled = quiet > (determinate ? QUIET_SAMPLING_MS : QUIET_LOADING_MS);

  return (
    <div className={`genbar ${stalled ? "stalled" : ""}`}>
      <div className="genbar-track">
        <div
          className={`genbar-fill ${determinate ? "" : "indet"}`}
          style={determinate ? { width: `${pct}%` } : undefined}
        />
      </div>
      <div className="genbar-label">
        <span className="genbar-phase" title={progress.phase}>
          {progress.phase}
        </span>
        <span className="genbar-meta">
          {determinate && (
            <span className="genbar-step">
              {progress.step}/{progress.total}
            </span>
          )}
          <span className="genbar-time">{short(now - progress.startedAt)}</span>
          {stalled && (
            <span
              className="genbar-warn"
              title="How long since the backend last reported anything"
            >
              ⚠ quiet {short(quiet)}
            </span>
          )}
        </span>
      </div>
    </div>
  );
}
