import { useEffect, useRef, useState } from "react";
import { fileToResizedDataUrl } from "../fileUtils";
import type { GenSettings, GenOp } from "../types";
import { pullSdModel, type SdModel } from "../api";

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
  // Image generation (diffusers).
  genMode: boolean;
  setGenMode: (v: boolean) => void;
  /** Which generate workflow (create / edit / compose). */
  genOp: GenOp;
  setGenOp: (v: GenOp) => void;
  sdAvailable: boolean;
  /** FLUX Kontext (edit + compose) installed on the backend. */
  fluxAvailable: boolean;
  sdModels: SdModel[];
  gen: GenSettings;
  setGen: (v: GenSettings) => void;
  /** How many images are pinned in the panel (compose reference count). */
  pinnedCount: number;
  /** First pinned-panel image, used as the img2img source when nothing is
   *  attached to the message. `null` when the panel is empty. */
  pinnedInit: string | null;
  /** Refresh SD model info after a model is downloaded. */
  onModelPulled: () => void;
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
  sdAvailable,
  fluxAvailable,
  sdModels,
  gen,
  setGen,
  pinnedCount,
  pinnedInit,
  onModelPulled,
}: Props) {
  const fileRef = useRef<HTMLInputElement>(null);
  const sysRef = useRef<HTMLDivElement>(null);
  const settingsRef = useRef<HTMLDivElement>(null);
  const [sysOpen, setSysOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [dlStatus, setDlStatus] = useState<string | null>(null);
  const selectedModel = sdModels.find((m) => m.id === gen.model);

  async function downloadModel() {
    if (!selectedModel) return;
    setDlStatus("Starting…");
    try {
      await pullSdModel(selectedModel.id, setDlStatus);
      onModelPulled();
      setDlStatus(null);
    } catch (e) {
      setDlStatus(`⚠️ ${(e as Error).message}`);
    }
  }
  const hasSystem = systemPrompt.trim().length > 0 || !!systemImage;
  const patchGen = (p: Partial<typeof gen>) => setGen({ ...gen, ...p });
  // Switching model applies its preset: Turbo is a few-step, no-guidance model;
  // standard SD wants ~25 steps and CFG ~7.5.
  const onModelChange = (id: string) => {
    const turbo = sdModels.find((m) => m.id === id)?.turbo;
    setGen({
      ...gen,
      model: id,
      steps: turbo ? 2 : 25,
      guidance: turbo ? 0 : 7.5,
    });
  };
  const isEdit = genOp === "edit";
  const isCompose = genOp === "compose";
  // In create/edit the source is the attached image, else the first pinned-panel
  // image. create infers txt2img vs img2img from whether one is present.
  const initSource = images.length > 0 ? "attached" : pinnedInit ? "pinned" : null;
  const genSubmode = genMode && !isEdit && !isCompose && initSource ? "img2img" : "txt2img";
  const initPreview = images.length > 0 ? images[0] : pinnedInit;
  // compose blends every attached image, else every pinned one.
  const composeCount = images.length > 0 ? images.length : pinnedCount;
  // create runs on a selected SD checkpoint; edit/compose run on FLUX Kontext
  // (a fixed engine — no model dropdown).
  const isFlux = isEdit || isCompose;

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

  return (
    <div className="composer">
      {sdAvailable && (
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
              <>Editing this image ({initSource === "attached" ? "attached" : "from panel"}) — write an
              instruction like <em>“make the cat eat the broccoli”</em>.</>
            ) : (
              <>Attach or pin <em>one</em> image to edit, then write an instruction.</>
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
                    const f = e.target.files?.[0];
                    if (f) setSystemImage(await fileToResizedDataUrl(f));
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
                {genOp === "create" && selectedModel && !selectedModel.downloaded && (
                  <div className="dl-box">
                    <div className="dl-row">
                      <span>
                        <strong>{selectedModel.label}</strong> isn't downloaded
                        {" "}(~{selectedModel.size_gb} GB, one-time).
                      </span>
                    </div>
                    {dlStatus ? (
                      <div className="dl-status">{dlStatus}</div>
                    ) : (
                      <button className="btn block" onClick={downloadModel}>
                        ⬇ Download {selectedModel.label}
                      </button>
                    )}
                    <p className="hint muted">
                      The only network call this app makes. After download,
                      generation is fully offline.
                    </p>
                  </div>
                )}
                {isFlux && !fluxAvailable && (
                  <div className="dl-box">
                    <p className="hint muted">
                      ⚠️ FLUX Kontext isn't installed on this machine, so Edit and
                      Combine are unavailable.
                    </p>
                  </div>
                )}
                {isFlux && (
                  <p className="hint muted" style={{ marginTop: 0 }}>
                    ✨ Powered by <strong>FLUX Kontext</strong> (local). High quality —
                    expect a few minutes per image.
                  </p>
                )}
                <label className="lbl">Negative prompt</label>
                <textarea
                  className="block"
                  rows={2}
                  value={gen.negativePrompt}
                  onChange={(e) => patchGen({ negativePrompt: e.target.value })}
                  placeholder="What to avoid (optional)…"
                />
                <div className="gen-grid">
                  <label>Steps
                    <input type="number" min={1} max={50} value={gen.steps}
                      onChange={(e) => patchGen({ steps: +e.target.value })} />
                  </label>
                  <label>Guidance
                    <input type="number" min={0} max={20} step={0.5} value={gen.guidance}
                      onChange={(e) => patchGen({ guidance: +e.target.value })} />
                  </label>
                  {genOp === "create" && (
                    <>
                      <label>Width
                        <input type="number" min={256} max={1024} step={64} value={gen.width}
                          onChange={(e) => patchGen({ width: +e.target.value })} />
                      </label>
                      <label>Height
                        <input type="number" min={256} max={1024} step={64} value={gen.height}
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
                {genOp === "create" && selectedModel?.photoreal && (
                  <label className="enhance-row">
                    <input type="checkbox" checked={gen.enhance}
                      onChange={(e) => patchGen({ enhance: e.target.checked })} />
                    Enhance photoreal prompt (adds quality tags)
                  </label>
                )}
                <p className="hint muted">
                  {isFlux
                    ? "Guidance ~2.5 follows the instruction closely; lower it for looser, more creative edits."
                    : genSubmode === "img2img"
                      ? "Image-to-image: strength controls how far from the attached image."
                      : "Text-to-image: attach an image above to switch to image-to-image."}
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
              ? isEdit
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

        {genMode && isFlux ? (
          <div className="model-control" title="Image engine">
            <span aria-hidden>✨</span>
            <span className="flux-engine">FLUX Kontext</span>
          </div>
        ) : genMode ? (
          <div className="model-control" title="Image model">
            <span aria-hidden>🎨</span>
            <select value={gen.model} onChange={(e) => onModelChange(e.target.value)}>
              {sdModels.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.label}
                  {m.downloaded ? "" : " ⬇"}
                </option>
              ))}
            </select>
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
