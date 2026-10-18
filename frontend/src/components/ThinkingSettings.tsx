import { useState } from "react";
import type { ThinkingDisplay } from "../types";

interface Props {
  display: ThinkingDisplay;
  setDisplay: (v: ThinkingDisplay) => void;
}

const HINTS: Record<ThinkingDisplay, string> = {
  quiet: "Quiet: you only see “Thinking…” until the answer starts.",
  verbose:
    "Verbose: the reasoning streams into a collapsible block above the answer, and under a prompt the enhancer rewrote.",
};

/** How thinking is shown. Switching it on and off is the composer's 💭 button;
 *  this is the set-once part, so it lives here rather than in the toolbar. */
export default function ThinkingSettings({ display, setDisplay }: Props) {
  const [open, setOpen] = useState(false);

  return (
    <div className="section">
      <button className="section-head" onClick={() => setOpen(!open)}>
        💭 Thinking <span className="chev">{open ? "▾" : "▸"}</span>
      </button>
      {open && (
        <div className="section-body">
          <label className="lbl">Show reasoning</label>
          <select
            value={display}
            onChange={(e) => setDisplay(e.target.value as ThinkingDisplay)}
          >
            <option value="quiet">Quiet</option>
            <option value="verbose">Verbose</option>
          </select>
          <p className="hint muted small">{HINTS[display]}</p>
          <p className="hint muted small">
            Turn thinking on or off with the 💭 button in the composer. It applies to
            Analyze chats and prompt enhancement, and is ignored by models that can't
            think.
          </p>
        </div>
      )}
    </div>
  );
}
