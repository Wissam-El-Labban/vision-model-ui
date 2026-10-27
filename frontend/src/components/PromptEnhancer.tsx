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

const HINTS: Record<EnhancerMode, string> = {
  off: "Off: your prompt goes to the image model exactly as you wrote it.",
  on: "Normal: the picked model silently rewrites your prompt before every generation.",
  verbose:
    "Verbose: rewrites automatically and shows the result under your message in the chat.",
};

/** The prompt enhancer's settings, shown in the composer's ✨ popover. It lives
 *  there, beside ⚙️, and only in Generate mode: that's the only mode it touches —
 *  agent mode writes its own prompts — and in the sidebar it read as if it
 *  applied everywhere. */
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
  return (
    <>
      <div className="popover-title">✨ Prompt enhancer</div>
      {/* The model line the sidebar header used to show — Auto included, since
          "Auto" alone doesn't say which model you're getting. */}
      <div className="section-sub" title={effectiveModel}>
        {effectiveModel
          ? `${effectiveModel}${enhancerModel ? "" : " (auto)"}`
          : "No vision model installed"}
      </div>
      <label className="lbl">Vision model (sees your reference images)</label>
      {visionModels.length > 0 ? (
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
      ) : (
        <p className="hint muted">
          No vision model installed — pull one from the sidebar's Model Manager.
        </p>
      )}

      <label className="lbl">Enhancement</label>
      <select
        value={enhancerMode}
        onChange={(e) => setEnhancerMode(e.target.value as EnhancerMode)}
      >
        <option value="off">Off</option>
        <option value="on">Normal</option>
        <option value="verbose">Verbose</option>
      </select>

      <p className="hint muted">{HINTS[enhancerMode]}</p>

      <label className="enhance-row">
        <input
          type="checkbox"
          checked={enhanceTemplate}
          onChange={(e) => setEnhanceTemplate(e.target.checked)}
        />
        Enhance photoreal prompt (adds camera + lighting detail)
      </label>
      <p className="hint muted">
        A simpler, always-local template wrap for create/img2img prompts —
        independent of the vision model above. Skipped whenever that model
        already rewrote this prompt.
      </p>
    </>
  );
}
