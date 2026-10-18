import { useState } from "react";
import type { EnhancerMode } from "../types";

interface Props {
  visionModels: string[];
  enhancerModel: string;
  setEnhancerModel: (v: string) => void;
  /** The model that will actually do the rewriting — the pick, or what Auto
   *  resolves to. "" when there's none to use. */
  effectiveModel: string;
  enhancerMode: EnhancerMode;
  setEnhancerMode: (v: EnhancerMode) => void;
  /** Independent of the model/mode above: a simpler server-side template wrap
   *  (adds boilerplate camera + lighting phrasing) for create/img2img prompts,
   *  skipped whenever this same request was already rewritten by the picks
   *  above. */
  enhanceTemplate: boolean;
  setEnhanceTemplate: (v: boolean) => void;
}

const STATUS: Record<EnhancerMode, string> = { off: "Off", on: "On", verbose: "Verbose" };

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
  effectiveModel,
  enhancerMode,
  setEnhancerMode,
  enhanceTemplate,
  setEnhanceTemplate,
}: Props) {
  const [open, setOpen] = useState(false);

  return (
    <div className="section">
      <button className="section-head" onClick={() => setOpen(!open)}>
        <span className="section-head-left">
          ✨ Prompt Enhancer
          {/* The full name, readable with the section collapsed — Auto included,
              since "Auto" alone doesn't say which model you're getting. */}
          <span className="section-sub" title={effectiveModel}>
            {effectiveModel
              ? `${effectiveModel}${enhancerModel ? "" : " (auto)"}`
              : "No vision model installed"}
          </span>
        </span>
        <span className="section-head-right">
          <span className={`section-status ${enhancerMode !== "off" ? "on" : ""}`}>
            {STATUS[enhancerMode]}
          </span>
          <span className="chev">{open ? "▾" : "▸"}</span>
        </span>
      </button>
      {open && (
        <div className="section-body">
          <label className="lbl">Vision model (sees your reference images)</label>
          <select
            value={enhancerModel}
            onChange={(e) => setEnhancerModel(e.target.value)}
          >
            <option value="">
              Auto (chat model, else first vision model)
              {!enhancerModel && effectiveModel ? ` → ${effectiveModel}` : ""}
            </option>
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

          <label className="enhance-row">
            <input
              type="checkbox"
              checked={enhanceTemplate}
              onChange={(e) => setEnhanceTemplate(e.target.checked)}
            />
            Enhance photoreal prompt (adds camera + lighting detail)
          </label>
          <p className="hint muted small">
            A simpler, always-local template wrap for create/img2img prompts —
            independent of the vision model above. Skipped whenever that model
            already rewrote this prompt.
          </p>
        </div>
      )}
    </div>
  );
}
