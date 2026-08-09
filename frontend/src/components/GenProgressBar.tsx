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

/** A real progress bar for the in-flight generation.
 *
 * The fill is the backend's own measure of how much of the job is done — it prices
 * every node in the ComfyUI graph and reports the finished share (`_Progress` in
 * flux_client.py), so loading weights, sampling and decoding each move the bar by
 * what they actually cost. Nothing here animates on its own: a bar that swept while
 * the backend was wedged would be decoration, and the point of this one is to be
 * readable. The clock readouts carry what the fill can't — only a real backend event
 * moves `updatedAt`, and once the gap passes the threshold the bar goes amber.
 *
 * `frac === null` is the one honest gap: the text path (Ollama loading a model)
 * reports no progress at all, so the bar stays at zero and only the phase and the
 * elapsed clock move. */
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

  const measured = progress.frac !== null;
  const pct = measured ? Math.max(0, Math.min(100, progress.frac! * 100)) : 0;
  const elapsed = now - progress.startedAt;
  const quiet = now - progress.updatedAt;
  // `step > 0`, not `total > 0`: the step total is read off the graph before anything
  // runs, so it says nothing about whether sampling has started. A reported step does.
  const stalled = quiet > (progress.step > 0 ? QUIET_SAMPLING_MS : QUIET_LOADING_MS);
  // Elapsed against the share done. Held back until the bar has enough of the job
  // behind it to divide by — at 1% the same arithmetic says anything at all.
  const eta =
    measured && progress.frac! >= 0.04 && progress.frac! < 1
      ? (elapsed * (1 - progress.frac!)) / progress.frac!
      : null;

  return (
    <div className={`genbar ${stalled ? "stalled" : ""}`}>
      <div
        className="genbar-track"
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={measured ? Math.round(pct) : undefined}
        aria-valuetext={measured ? `${Math.round(pct)}%` : "working"}
      >
        <div className="genbar-fill" style={{ width: `${pct}%` }} />
      </div>
      <div className="genbar-label">
        <span className="genbar-phase" title={progress.phase}>
          {progress.stage || progress.phase}
        </span>
        <span className="genbar-meta">
          {progress.step > 0 && (
            <span className="genbar-step">
              {progress.step}/{progress.total}
            </span>
          )}
          {measured && <span className="genbar-pct">{Math.round(pct)}%</span>}
          <span className="genbar-time">{short(elapsed)}</span>
          {eta !== null && !stalled && (
            <span className="genbar-eta" title="Estimated from the share done so far">
              ~{short(eta)} left
            </span>
          )}
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
