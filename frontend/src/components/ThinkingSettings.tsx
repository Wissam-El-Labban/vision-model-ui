import { useState } from "react";
import type { ThinkingDisplay } from "../types";

interface Props {
  thinking: boolean;
  setThinking: (v: boolean) => void;
  display: ThinkingDisplay;
  setDisplay: (v: ThinkingDisplay) => void;
}

const HINTS: Record<ThinkingDisplay, string> = {
  quiet: "Quiet: you only see “Thinking…” until the answer starts.",
  verbose:
    "Verbose: the reasoning streams into a collapsible block above the answer, and under a prompt the enhancer rewrote.",
};

/** One switch for thinking in Analyze, prompt enhancement and agent mode, plus
 *  how the reasoning is shown. The header carries the state so it reads at a
 *  glance with the section collapsed. */
export default function ThinkingSettings({ thinking, setThinking, display, setDisplay }: Props) {
  const [open, setOpen] = useState(false);
  const status = thinking ? `On · ${display === "verbose" ? "Verbose" : "Quiet"}` : "Off";

  return (
    <div className="section">
      <button className="section-head" onClick={() => setOpen(!open)}>
        💭 Thinking
        <span className="section-head-right">
          <span className={`section-status ${thinking ? "on" : ""}`}>{status}</span>
          <span className="chev">{open ? "▾" : "▸"}</span>
        </span>
      </button>
      {open && (
        <div className="section-body">
          <label className="enhance-row">
            <input
              type="checkbox"
              checked={thinking}
              onChange={(e) => setThinking(e.target.checked)}
            />
            Let the model think before answering
          </label>
          <p className="hint muted small">
            Applies to Analyze chats, prompt enhancement and agent mode. Slower, and
            ignored by models that can't think.
          </p>

          <label className="lbl">Show reasoning</label>
          <select
            value={display}
            disabled={!thinking}
            onChange={(e) => setDisplay(e.target.value as ThinkingDisplay)}
          >
            <option value="quiet">Quiet</option>
            <option value="verbose">Verbose</option>
          </select>
          <p className="hint muted small">{HINTS[display]}</p>
        </div>
      )}
    </div>
  );
}
