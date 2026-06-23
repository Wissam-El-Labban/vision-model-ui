interface Props {
  used: number;
  numCtx: number;
}

/** Circular meter showing how full the context window is, based on the exact
 *  token counts Ollama reports for the last turn. */
export default function ContextMeter({ used, numCtx }: Props) {
  const ratio = Math.min(1, numCtx > 0 ? used / numCtx : 0);
  const pct = Math.round(ratio * 100);
  const R = 9;
  const C = 2 * Math.PI * R;
  const color =
    ratio >= 0.85 ? "var(--red)" : ratio >= 0.6 ? "#d29922" : "var(--green)";

  return (
    <div
      className="ctx-meter"
      title={`Context: ${used.toLocaleString()} / ${numCtx.toLocaleString()} tokens (${pct}%)${
        ratio >= 0.8 ? " — older turns are being trimmed" : ""
      }`}
    >
      <svg width="22" height="22" viewBox="0 0 22 22">
        <circle cx="11" cy="11" r={R} className="ctx-track" />
        <circle
          cx="11"
          cy="11"
          r={R}
          className="ctx-fill"
          stroke={color}
          strokeDasharray={C}
          strokeDashoffset={C * (1 - ratio)}
          transform="rotate(-90 11 11)"
        />
      </svg>
      <span className="ctx-pct" style={{ color }}>
        {pct}%
      </span>
    </div>
  );
}
