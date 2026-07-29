import { useState } from "react";
import type { EnhancerMode } from "../types";

interface Props {
  visionModels: string[];
  enhancerModel: string;
  setEnhancerModel: (v: string) => void;
  enhancerMode: EnhancerMode;
  setEnhancerMode: (v: EnhancerMode) => void;
}

const HINTS: Record<EnhancerMode, string> = {
  off: "Off: the prompt is only rewritten when you click ✨ Improve prompt in the composer.",
  on: "On: the picked model silently rewrites your prompt before every generation.",
  verbose:
    "Verbose: rewrites automatically and shows the result in the composer before it's sent.",
};

export default function PromptEnhancer({
  visionModels,
  enhancerModel,
  setEnhancerModel,
  enhancerMode,
  setEnhancerMode,
}: Props) {
  const [open, setOpen] = useState(false);

  return (
    <div className="section">
      <button className="section-head" onClick={() => setOpen(!open)}>
        ✨ Prompt Enhancer <span className="chev">{open ? "▾" : "▸"}</span>
      </button>
      {open && (
        <div className="section-body">
          <label className="lbl">Vision model (sees your reference images)</label>
          <select
            value={enhancerModel}
            onChange={(e) => setEnhancerModel(e.target.value)}
          >
            <option value="">Auto (chat model, else first vision model)</option>
            {visionModels.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>

          <label className="lbl">Enhancement</label>
          <select
            value={enhancerMode}
            onChange={(e) => setEnhancerMode(e.target.value as EnhancerMode)}
          >
            <option value="off">Off</option>
            <option value="on">On</option>
            <option value="verbose">Verbose</option>
          </select>

          <p className="hint muted small">{HINTS[enhancerMode]}</p>
        </div>
      )}
    </div>
  );
}
