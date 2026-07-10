import { useEffect, useRef, useState } from "react";
import { fileToResizedDataUrl } from "../fileUtils";
import type { GenSettings, GenOp } from "../types";
import { pullFluxModel, deleteFluxModel, type FluxModel } from "../api";

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
  /** FLUX sidecar + weights installed on the backend. */
  fluxAvailable: boolean;
  /** Installed FLUX UNets, defaults first. Filtered by role per mode. */
  fluxModels: FluxModel[];
  gen: GenSettings;
  setGen: (v: GenSettings) => void;
  /** How many images are pinned in the panel (compose reference count). */
  pinnedCount: number;
  /** First pinned-panel image, used as the img2img source when nothing is
   *  attached to the message. `null` when the panel is empty. */
  pinnedInit: string | null;
  /** Refresh the FLUX model list after an add/remove. */
  onFluxModelsChanged: () => void;
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
  pinnedCount,
  pinnedInit,
  onFluxModelsChanged,
}: Props) {
  const fileRef = useRef<HTMLInputElement>(null);
  const sysRef = useRef<HTMLDivElement>(null);
  const settingsRef = useRef<HTMLDivElement>(null);
  const [sysOpen, setSysOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [fluxRepo, setFluxRepo] = useState("");
  const [fluxStatus, setFluxStatus] = useState<string | null>(null);
  const [fluxBusy, setFluxBusy] = useState(false);

  const hasSystem = systemPrompt.trim().length > 0 || !!systemImage;
  const patchGen = (p: Partial<typeof gen>) => setGen({ ...gen, ...p });

  const isEdit = genOp === "edit";
  const isCompose = genOp === "compose";
  // create runs on a plain FLUX dev UNet; edit/compose need a Kontext UNet, which
  // additionally conditions on the source image. The two sets are disjoint.
  const role = isEdit || isCompose ? "edit" : "create";
  const roleModels = fluxModels.filter((m) => m.role === role);

  // "" in gen.fluxModel means "this mode's default", so the active selection
  // resolves to the default model's filename within the current role.
  const defaultFluxName = roleModels.find((m) => m.default)?.name ?? "";
  const activeFlux = gen.fluxModel || defaultFluxName;
  const prettyFlux = (name: string) => name.replace(/\.(gguf|safetensors|sft)$/i, "");
  const pickFlux = (m: FluxModel) => patchGen({ fluxModel: m.default ? "" : m.name });

  async function addFluxModel() {
    const repo = fluxRepo.trim();
    if (!repo || fluxBusy) return;
    setFluxBusy(true);
    setFluxStatus("Starting…");
    try {
      await pullFluxModel(repo, setFluxStatus);
      setFluxRepo("");
      setFluxStatus(null);
      onFluxModelsChanged();
    } catch (e) {
      setFluxStatus(`⚠️ ${(e as Error).message}`);
    } finally {
      setFluxBusy(false);
    }
  }

  async function removeFluxModel(name: string) {
    if (fluxBusy) return;
    setFluxBusy(true);
    setFluxStatus(null);
    try {
      await deleteFluxModel(name);
      if (gen.fluxModel === name) patchGen({ fluxModel: "" }); // fell back to default
      onFluxModelsChanged();
    } catch (e) {
      setFluxStatus(`⚠️ ${(e as Error).message}`);
    } finally {
      setFluxBusy(false);
    }
  }
  // In create/edit the source is the attached image, else the first pinned-panel
  // image. create infers txt2img vs img2img from whether one is present.
  const initSource = images.length > 0 ? "attached" : pinnedInit ? "pinned" : null;
  const genSubmode = genMode && !isEdit && !isCompose && initSource ? "img2img" : "txt2img";
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

  return (
    <div className="composer">
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
                {!fluxAvailable && (
                  <div className="dl-box">
                    <p className="hint muted">
                      ⚠️ FLUX isn't installed on this machine, so image generation is
                      unavailable. Run <code>./run.sh</code> to fetch the weights.
                    </p>
                  </div>
                )}
                <p className="hint muted" style={{ marginTop: 0 }}>
                  ✨ Powered by <strong>{role === "edit" ? "FLUX Kontext" : "FLUX.1-dev"}</strong>{" "}
                  (local). High quality — expect about a minute per image.
                </p>
                {fluxAvailable && (
                  <div className="flux-models">
                    <label className="lbl">
                      {role === "edit" ? "FLUX Kontext models" : "FLUX create models"}
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
                            {prettyFlux(m.name)}
                          </button>
                          <span className="flux-model-size">{m.size_gb} GB</span>
                          {m.default ? (
                            <span className="flux-model-tag">default</span>
                          ) : (
                            <button
                              type="button"
                              className="flux-model-del"
                              title="Remove this model"
                              disabled={fluxBusy}
                              onClick={() => removeFluxModel(m.name)}
                            >
                              🗑
                            </button>
                          )}
                        </li>
                      ))}
                    </ul>
                    <div className="flux-add">
                      <input
                        type="text"
                        value={fluxRepo}
                        placeholder="owner/model (HuggingFace repo)"
                        disabled={fluxBusy}
                        onChange={(e) => setFluxRepo(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") {
                            e.preventDefault();
                            addFluxModel();
                          }
                        }}
                      />
                      <button
                        type="button"
                        className="btn"
                        disabled={fluxBusy || !fluxRepo.trim()}
                        onClick={addFluxModel}
                      >
                        ⬇ Add
                      </button>
                    </div>
                    {fluxStatus && <div className="dl-status">{fluxStatus}</div>}
                    <p className="hint muted">
                      Paste a HuggingFace FLUX repo (e.g.{" "}
                      <code>{role === "edit"
                        ? "QuantStack/FLUX.1-Kontext-dev-GGUF"
                        : "city96/FLUX.1-dev-gguf"}</code>), or{" "}
                      <code>owner/repo:file</code> for a specific <code>.gguf</code>/
                      <code>.safetensors</code>. A bare repo picks the highest quant
                      (Q8_0), or the largest checkpoint if the repo ships no GGUF;
                      unquantized models load in fp8 to fit this GPU. Models with{" "}
                      <code>kontext</code> in the name serve Edit/Combine; the rest serve
                      Create. Downloaded once, then fully offline.
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
                {genOp === "create" && (
                  <label className="enhance-row">
                    <input type="checkbox" checked={gen.enhance}
                      onChange={(e) => patchGen({ enhance: e.target.checked })} />
                    Enhance photoreal prompt (adds camera + lighting detail)
                  </label>
                )}
                <p className="hint muted">
                  {role === "edit"
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

        {genMode ? (
          <div className="model-control" title="FLUX model">
            <span aria-hidden>✨</span>
            {roleModels.length > 0 ? (
              <select
                value={activeFlux}
                onChange={(e) =>
                  patchGen({
                    fluxModel: e.target.value === defaultFluxName ? "" : e.target.value,
                  })
                }
              >
                {roleModels.map((m) => (
                  <option key={m.name} value={m.name}>
                    {prettyFlux(m.name)}
                    {m.default ? " (default)" : ""}
                  </option>
                ))}
              </select>
            ) : (
              <span className="flux-engine">
                {role === "edit" ? "FLUX Kontext" : "FLUX.1-dev"}
              </span>
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
