import { useEffect, useRef, useState } from "react";
import { fileToDataUrl } from "../fileUtils";
import { resolveFlux, roleFor } from "../flux";
import type { GenSettings, GenOp } from "../types";
import type { FluxModel } from "../api";

interface Props {
  text: string;
  setText: (v: string) => void;
  images: string[];
  onAddFiles: (files: FileList | File[]) => void;
  onRemoveImage: (i: number) => void;
  onRotateImage: (i: number) => void;
  onSubmit: () => void;
  onStop: () => void;
  streaming: boolean;
  disabled: boolean;
  // Moved here from the sidebar: system prompt (left) + model selector (right).
  models: { vision: string[]; all: string[] };
  model: string;
  setModel: (v: string) => void;
  systemPrompt: string;
  setSystemPrompt: (v: string) => void;
  systemImage: string | null;
  setSystemImage: (v: string | null) => void;
  // Image generation (FLUX).
  genMode: boolean;
  setGenMode: (v: boolean) => void;
  /** Which generate workflow (create / edit / compose). */
  genOp: GenOp;
  setGenOp: (v: GenOp) => void;
  /** Engine + at least one model installed on the backend. */
  fluxAvailable: boolean;
  /** Installed image models. Filtered by role per mode; installing and removing
   *  them lives in the sidebar's Image Models panel. */
  fluxModels: FluxModel[];
  gen: GenSettings;
  setGen: (v: GenSettings) => void;
  /** Rewrite the prompt with a local vision model that can see the attached
   *  images. Replaces the composer text in place — what's shown is what's sent. */
  onEnhance: () => void;
  /** Restore the pre-enhance prompt. `canUndoEnhance` gates the button. */
  onUndoEnhance: () => void;
  enhancing: boolean;
  canUndoEnhance: boolean;
  /** "" when no vision model is installed — the rewrite falls back to a template. */
  enhanceModel: string;
  /** How many images are pinned in the panel (compose reference count). */
  pinnedCount: number;
  /** First pinned-panel image, used as the img2img source when nothing is
   *  attached to the message. `null` when the panel is empty. */
  pinnedInit: string | null;
}

export default function Composer({
  text,
  setText,
  images,
  onAddFiles,
  onRemoveImage,
  onRotateImage,
  onSubmit,
  onStop,
  streaming,
  disabled,
  models,
  model,
  setModel,
  systemPrompt,
  setSystemPrompt,
  systemImage,
  setSystemImage,
  genMode,
  setGenMode,
  genOp,
  setGenOp,
  fluxAvailable,
  fluxModels,
  gen,
  setGen,
  onEnhance,
  onUndoEnhance,
  enhancing,
  canUndoEnhance,
  enhanceModel,
  pinnedCount,
  pinnedInit,
}: Props) {
  const fileRef = useRef<HTMLInputElement>(null);
  const sysRef = useRef<HTMLDivElement>(null);
  const settingsRef = useRef<HTMLDivElement>(null);
  const [sysOpen, setSysOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  // Generate doesn't need an Ollama model, so `disabled` only bites in chat mode
  // — same rule the attach button uses.
  const dropDisabled = disabled && !genMode;

  const hasSystem = systemPrompt.trim().length > 0 || !!systemImage;
  const patchGen = (p: Partial<typeof gen>) => setGen({ ...gen, ...p });

  const isEdit = genOp === "edit";
  const isCompose = genOp === "compose";
  const isAnimate = genOp === "animate";
  // A FLUX.2 model serves both roles; on FLUX.1 the sets are disjoint, because
  // edit/compose need a Kontext transformer that conditions on the source image.
  const role = roleFor(genOp);
  const roleModels = fluxModels.filter((m) => m.roles.includes(role));

  // What this mode will actually run on. `resolveFlux` is the one place that
  // decides that — App sends the same value, so the picker can't show one model
  // while another does the work.
  const activeFlux = resolveFlux(gen.fluxModel, fluxModels, role);
  // Bundled models carry a real label ("FLUX.2 [klein] 9B — …"); for a user-added
  // one the label *is* the filename, so keep stripping the extension.
  const prettyFlux = (name: string) => name.replace(/\.(gguf|safetensors|sft)$/i, "");
  const pickFlux = (m: FluxModel) => patchGen({ fluxModel: m.name });

  // In create/edit the source is the attached image, else the first pinned-panel
  // image. create infers txt2img vs img2img from whether one is present.
  const initSource = images.length > 0 ? "attached" : pinnedInit ? "pinned" : null;
  // Only `create` has a submode. animate is excluded because it always takes a
  // source image: left in, it would read as img2img and offer the strength slider,
  // which Wan has no equivalent of.
  const genSubmode =
    genMode && !isEdit && !isCompose && !isAnimate && initSource ? "img2img" : "txt2img";
  const initPreview = images.length > 0 ? images[0] : pinnedInit;
  // compose blends every attached image, else every pinned one.
  const composeCount = images.length > 0 ? images.length : pinnedCount;
  // edit: everything after the first source image is a subject reference.
  const editRefCount = Math.max((images.length > 0 ? images.length : pinnedCount) - 1, 0);

  // Close the system-prompt popover on any click outside it (parity with the
  // native model <select>, which closes itself).
  useEffect(() => {
    if (!sysOpen) return;
    const onDown = (e: MouseEvent) => {
      if (sysRef.current && !sysRef.current.contains(e.target as Node)) {
        setSysOpen(false);
      }
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [sysOpen]);

  useEffect(() => {
    if (!settingsOpen) return;
    const onDown = (e: MouseEvent) => {
      if (settingsRef.current && !settingsRef.current.contains(e.target as Node)) {
        setSettingsOpen(false);
      }
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [settingsOpen]);

  function onKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!streaming) onSubmit();
    }
  }

  // Dropping images anywhere on the composer attaches them — in generate mode
  // those are FLUX's reference images, so this is the natural place to drop them.
  // Without a handler the browser treats a dropped image as a navigation and
  // replaces the page, losing the conversation.
  const dropTarget = {
    onDragOver: (e: React.DragEvent) => {
      // Always cancel, even when we won't accept the files: a dragover that isn't
      // cancelled makes the composer an invalid drop target, so `drop` never fires
      // and the browser navigates to the image instead. Only the *affordance* is
      // conditional.
      e.preventDefault();
      if (!dropDisabled) setDragOver(true);
    },
    onDragLeave: (e: React.DragEvent) => {
      // dragleave also fires when the cursor crosses onto a child, and the
      // composer is full of them — ignore those or the outline strobes.
      if (e.currentTarget.contains(e.relatedTarget as Node | null)) return;
      setDragOver(false);
    },
    onDrop: (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      if (!dropDisabled && e.dataTransfer.files.length) onAddFiles(e.dataTransfer.files);
    },
  };

  return (
    <div className={`composer ${dragOver ? "drag" : ""}`} {...dropTarget}>
      {dragOver && (
        <div className="drop-overlay">
          {genMode
            ? isEdit
              ? "🖼️ Drop to add — first image is edited, the rest are references"
              : isCompose
                ? "🖼️ Drop to add reference images to combine"
                : "🖼️ Drop an image to generate from"
            : "📎 Drop to attach to your message"}
        </div>
      )}
      {fluxAvailable && (
        <div className="mode-toggle" role="tablist" aria-label="Composer mode">
          <button
            role="tab"
            aria-selected={!genMode}
            className={`mode-tab ${!genMode ? "active" : ""}`}
            onClick={() => setGenMode(false)}
          >
            🔍 Analyze
          </button>
          <button
            role="tab"
            aria-selected={genMode}
            className={`mode-tab ${genMode ? "active" : ""}`}
            onClick={() => setGenMode(true)}
          >
            🎨 Generate
          </button>
        </div>
      )}

      {genMode && (
        <div className="mode-toggle sub" role="tablist" aria-label="Generate workflow">
          <button
            role="tab"
            aria-selected={genOp === "create"}
            className={`mode-tab ${genOp === "create" ? "active" : ""}`}
            onClick={() => setGenOp("create")}
            title="Text-to-image, or transform one attached image"
          >
            🖼️ Create
          </button>
          <button
            role="tab"
            aria-selected={genOp === "edit"}
            className={`mode-tab ${genOp === "edit" ? "active" : ""}`}
            onClick={() => setGenOp("edit")}
            title="Instruction edit — 'make the cat eat the broccoli'"
          >
            ✏️ Edit
          </button>
          <button
            role="tab"
            aria-selected={genOp === "compose"}
            className={`mode-tab ${genOp === "compose" ? "active" : ""}`}
            onClick={() => setGenOp("compose")}
            title="Blend several reference images into one new image"
          >
            🧩 Combine
          </button>
          <button
            role="tab"
            aria-selected={genOp === "animate"}
            className={`mode-tab ${genOp === "animate" ? "active" : ""}`}
            onClick={() => setGenOp("animate")}
            title="Bring one image to life — a 5-second video. Describe the motion, not the scene."
          >
            🎬 Animate
          </button>
        </div>
      )}

      {genMode && genOp === "create" && genSubmode === "img2img" && (
        <div className="init-hint">
          {initPreview && (
            <img className="init-thumb" src={initPreview} alt="img2img source" />
          )}
          <span>
            Starting image ({initSource === "attached" ? "attached" : "from panel"}) — img2img
            transforms this <em>one</em> image toward your prompt.
          </span>
        </div>
      )}

      {genMode && isEdit && (
        <div className="init-hint">
          {initPreview && (
            <img className="init-thumb" src={initPreview} alt="edit source" />
          )}
          <span>
            {initPreview ? (
              <>
                Editing this image ({initSource === "attached" ? "attached" : "from panel"})
                {editRefCount > 0 ? (
                  <>, using the other {editRefCount} as reference{editRefCount > 1 ? "s" : ""} — e.g.{" "}
                  <em>“add the man in the black suit from the reference photo, keeping everyone else
                  unchanged”</em>.</>
                ) : (
                  <> — write an instruction like <em>“make the cat eat the broccoli”</em>. Attach more
                  images to pull subjects from them.</>
                )}
              </>
            ) : (
              <>Attach or pin an image to edit (the <em>first</em> one is the image that changes; any
              others are references), then write an instruction.</>
            )}
          </span>
        </div>
      )}

      {genMode && isCompose && (
        <div className="init-hint">
          <span>
            {composeCount > 0 ? (
              <>Blending <em>{composeCount}</em> reference image{composeCount > 1 ? "s" : ""}{" "}
              ({images.length > 0 ? "attached" : "from panel"}) into one new image guided by your prompt.</>
            ) : (
              <>Attach or pin the images you want to combine, then describe the result.</>
            )}
          </span>
        </div>
      )}

      {images.length > 0 && (
        <>
          <div className="thumbs">
            {images.map((src, i) => (
              <div className="thumb" key={i}>
                <img src={src} alt={`attachment ${i + 1}`} />
                <div className="thumb-actions">
                  <button title="Rotate" onClick={() => onRotateImage(i)}>
                    ↻
                  </button>
                  <button title="Remove" onClick={() => onRemoveImage(i)}>
                    ✕
                  </button>
                </div>
              </div>
            ))}
          </div>
        </>
      )}

      <div className="composer-row">
        <div className="sys-control" ref={sysRef}>
          <button
            className={`btn ghost icon ${hasSystem ? "has-dot" : ""}`}
            onClick={() => setSysOpen((v) => !v)}
            title="System prompt"
          >
            💬{hasSystem && <span className="dot" />}
          </button>
          {sysOpen && (
            <div className="system-popover">
              <div className="popover-title">💬 System prompt</div>
              <textarea
                className="block"
                rows={4}
                value={systemPrompt}
                onChange={(e) => setSystemPrompt(e.target.value)}
                placeholder="Guide the model's behavior across the chat…"
              />
              <label className="lbl">📎 Persistent context image (optional)</label>
              {systemImage ? (
                <div className="sys-image">
                  <img src={systemImage} alt="system" />
                  <button
                    className="btn danger block"
                    onClick={() => setSystemImage(null)}
                  >
                    Remove image
                  </button>
                </div>
              ) : (
                <input
                  type="file"
                  accept="image/*"
                  onChange={async (e) => {
                    // Store the original: the backend caps and downscales once,
                    // with a better filter, and hands each consumer its own copy.
                    const f = e.target.files?.[0];
                    if (f) setSystemImage(await fileToDataUrl(f));
                  }}
                />
              )}
            </div>
          )}
        </div>

        {genMode && (
          <div className="sys-control" ref={settingsRef}>
            <button
              className="btn ghost icon"
              onClick={() => setSettingsOpen((v) => !v)}
              title="Generation settings"
            >
              ⚙️
            </button>
            {settingsOpen && (
              <div className="system-popover gen-popover">
                <div className="popover-title">🎨 Generation settings</div>
                {!fluxAvailable && (
                  <div className="dl-box">
                    <p className="hint muted">
                      ⚠️ No image model is installed, so image generation is unavailable.
                      Install one under <strong>🖼️ Image Models</strong> in the sidebar.
                    </p>
                  </div>
                )}
                {fluxAvailable && (
                  <div className="flux-models">
                    <label className="lbl">
                      Model ({role === "edit" ? "edit / combine" : "create"})
                    </label>
                    <ul className="flux-model-list">
                      {roleModels.map((m) => (
                        <li key={m.name} className={activeFlux === m.name ? "active" : ""}>
                          <button
                            type="button"
                            className="flux-model-pick"
                            title="Use this model"
                            onClick={() => pickFlux(m)}
                          >
                            <span className="flux-radio">{activeFlux === m.name ? "●" : "○"}</span>
                            {prettyFlux(m.label)}
                          </button>
                          <span className="flux-model-size">{m.size_gb} GB</span>
                        </li>
                      ))}
                    </ul>
                    <p className="hint muted">
                      Local, and fully offline once downloaded — expect about a minute per
                      image. Add or remove models under <strong>🖼️ Image Models</strong> in
                      the sidebar.
                    </p>
                  </div>
                )}
                <div className="gen-grid">
                  <label>Steps
                    <input type="number" min={8} max={40} value={gen.steps}
                      onChange={(e) => patchGen({ steps: +e.target.value })} />
                  </label>
                  <label>Guidance
                    <input type="number" min={0.5} max={10} step={0.5} value={gen.guidance}
                      onChange={(e) => patchGen({ guidance: +e.target.value })} />
                  </label>
                  {genOp === "create" && (
                    <>
                      <label>Width
                        <input type="number" min={256} max={1536} step={64} value={gen.width}
                          onChange={(e) => patchGen({ width: +e.target.value })} />
                      </label>
                      <label>Height
                        <input type="number" min={256} max={1536} step={64} value={gen.height}
                          onChange={(e) => patchGen({ height: +e.target.value })} />
                      </label>
                      <label className={genSubmode === "img2img" ? "" : "muted-field"}>
                        Strength
                        <input type="number" min={0} max={1} step={0.05} value={gen.strength}
                          disabled={genSubmode !== "img2img"}
                          onChange={(e) => patchGen({ strength: +e.target.value })} />
                      </label>
                    </>
                  )}
                  <label>Seed
                    <input type="text" inputMode="numeric" value={gen.seed}
                      placeholder="random"
                      onChange={(e) => patchGen({ seed: e.target.value.replace(/[^0-9]/g, "") })} />
                  </label>
                </div>
                {/* The static template only fits a create prompt — it describes a
                    photograph, which is the wrong shape for an edit instruction. The
                    ✨ rewrite below works in every mode, so edit/combine get their
                    prompt help from there. */}
                {genOp === "create" && (
                  <label className="enhance-row">
                    <input type="checkbox" checked={gen.enhance}
                      onChange={(e) => patchGen({ enhance: e.target.checked })} />
                    Enhance photoreal prompt (adds camera + lighting detail)
                  </label>
                )}
                <div className="enhance-row">
                  <button
                    type="button"
                    className="btn small"
                    disabled={enhancing || !text.trim()}
                    onClick={onEnhance}
                    title={
                      enhanceModel
                        ? `Rewrite the prompt with ${enhanceModel}, which can see the attached images`
                        : "No vision model installed — falls back to a text template"
                    }
                  >
                    {enhancing ? "✨ Rewriting…" : "✨ Improve prompt"}
                  </button>
                  {canUndoEnhance && (
                    <button
                      type="button"
                      className="btn small ghost"
                      onClick={onUndoEnhance}
                      title="Restore the prompt you wrote"
                    >
                      ↩ Undo
                    </button>
                  )}
                </div>
                <p className="hint muted">
                  {enhanceModel
                    ? "✨ rewrites your prompt in place, reading the attached images. Edit the result before generating."
                    : "✨ uses a text template — install a vision model in Ollama for a rewrite that reads your images."}
                </p>
                <p className="hint muted">
                  {isAnimate
                    ? "Attach one image — it becomes the first frame. Describe the motion, not the scene; the frame already fixes that. Takes a few minutes."
                    : role === "edit"
                      ? "Guidance ~2.5 follows the instruction closely; raise it if the subject isn't changing enough."
                      : genSubmode === "img2img"
                        ? "Image-to-image: strength controls how far from the attached image."
                        : "Text-to-image: guidance ~3.5. Attach an image above to switch to image-to-image."}
                </p>
              </div>
            )}
          </div>
        )}

        <button
          className="btn icon"
          title={genMode ? "Attach a starting image (img2img)" : "Attach images"}
          onClick={() => fileRef.current?.click()}
          disabled={disabled && !genMode}
        >
          📎
        </button>
        <input
          ref={fileRef}
          type="file"
          accept="image/*"
          multiple
          hidden
          onChange={(e) => {
            if (e.target.files) onAddFiles(e.target.files);
            e.target.value = "";
          }}
        />
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder={
            genMode
              ? isAnimate
                ? "Describe the motion… e.g. “she turns to look at the camera, slow push in”"
                : isEdit
                ? "Instruction to apply… e.g. “make the cat eat the broccoli”"
                : isCompose
                  ? "Describe the combined image to create from the references…"
                  : genSubmode === "img2img"
                    ? "Describe how to transform the attached image…"
                    : "Describe the image to generate… (attach an image for img2img)"
              : disabled
                ? "Select a vision model to start…"
                : "Ask about your image(s)… (drop or paste images anywhere, Enter to send)"
          }
          rows={1}
          disabled={genMode ? false : disabled}
        />

        {streaming ? (
          <button className="btn stop" onClick={onStop}>
            ■ Stop
          </button>
        ) : (
          <button
            className={`btn send ${genMode ? "gen" : ""}`}
            onClick={onSubmit}
            title={genMode ? "Generate image" : "Send"}
            disabled={
              genMode ? !text.trim() : disabled || (!text.trim() && images.length === 0)
            }
          >
            {genMode ? "🎨" : "➤"}
          </button>
        )}

        {genMode ? (
          <div className="model-control" title="Image model">
            <span aria-hidden>✨</span>
            {roleModels.length > 0 ? (
              <select
                value={activeFlux}
                onChange={(e) => patchGen({ fluxModel: e.target.value })}
              >
                {roleModels.map((m) => (
                  <option key={m.name} value={m.name}>
                    {prettyFlux(m.label)}
                  </option>
                ))}
              </select>
            ) : (
              <span className="flux-engine">no model</span>
            )}
          </div>
        ) : (
          <div className="model-control" title="Vision model">
            <span aria-hidden>🤖</span>
            {models.vision.length > 0 ? (
              <select value={model} onChange={(e) => setModel(e.target.value)}>
                {models.vision.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
            ) : (
              <input
                value={model}
                onChange={(e) => setModel(e.target.value)}
                placeholder="No vision models — type one"
              />
            )}
          </div>
        )}
      </div>
    </div>
  );
}
