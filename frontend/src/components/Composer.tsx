import { useEffect, useRef, useState } from "react";
import { fileToDataUrl } from "../fileUtils";
import { guidanceFor, resolveFlux, roleFor, stepsFor } from "../flux";
import type { ControlKind, GenSettings, GenOp } from "../types";
import type { FluxModel, FluxPreprocessor } from "../api";

/** The control maps the Control tab offers, in the order they're worth reaching for.
 *
 * Depth first because it is the one that answers the question a skeleton can't: a
 * pose that interacts with something — sitting on a chair, standing on a closet, a
 * hand pressed into a mat — needs the *scene* in the control signal, and only depth
 * carries it. Pose is listed last not because it's weak but because alone it is
 * ambiguous exactly where hard poses break: it can't say which arm is in front. */
const CONTROL_KINDS: { kind: ControlKind; label: string; hint: string }[] = [
  { kind: "depth", label: "Depth",
    hint: "3-D shape of the whole scene — the chair, the floor, which limb is in front. Best for a pose that touches something." },
  { kind: "canny", label: "Edges",
    hint: "Every outline in the source. Tightest hold on layout, but it carries the source's style across too." },
  { kind: "pose", label: "Pose",
    hint: "An OpenPose skeleton: limb positions only, nothing about the scene. Stack it with Depth." },
];

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
  /** True while the settings-level auto-enhancer (sidebar) is rewriting the
   *  prompt for this submit — guards against a double-send during that gap.
   *  Verbose mode's rewrite itself isn't shown here: it's attached to the chat
   *  message it produced and rendered underneath it, not in the composer. */
  enhancing: boolean;
  /** How many images are pinned in the panel (compose reference count). */
  pinnedCount: number;
  /** First pinned-panel image, used as the img2img source when nothing is
   *  attached to the message. `null` when the panel is empty. */
  pinnedInit: string | null;
  /** Which control maps can be built right now. A kind whose model isn't installed
   *  is offered but not selectable — hiding it would leave the user wondering why
   *  the app can't do the thing every ControlNet guide says it should. */
  preprocessors: FluxPreprocessor[];
  /** Maps posed in the studio, waiting to be generated from. */
  studioMaps: { kind: ControlKind; url: string }[];
  /** What those maps contain. The enhancer is off by default, so for most runs
   *  this is only useful said out loud to the user. */
  studioMeta: { subjects: number; contact: boolean } | null;
  onOpenStudio: () => void;
  onClearStudioMaps: () => void;
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
  enhancing,
  pinnedCount,
  pinnedInit,
  preprocessors,
  studioMaps,
  studioMeta,
  onOpenStudio,
  onClearStudioMaps,
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
  const isControl = genOp === "control";
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
  // Picking a model explicitly retunes it to that model's own defaults — steps
  // and guidance both vary by model (see `stepsFor`/`guidanceFor`), and a seed
  // hand-tuned for one model's output isn't meaningful for another's.
  const pickFlux = (m: FluxModel) =>
    patchGen({
      fluxModel: m.name,
      steps: stepsFor(genOp, fluxModels, m.name),
      guidance: guidanceFor(genOp, fluxModels, m.name),
      seed: "",
    });

  // In create/edit the source is the attached image, else the first pinned-panel
  // image. create infers txt2img vs img2img from whether one is present.
  const initSource = images.length > 0 ? "attached" : pinnedInit ? "pinned" : null;
  // Only `create` has a submode. animate is excluded because it always takes a
  // source image: left in, it would read as img2img and offer the strength slider,
  // which Wan has no equivalent of.
  const genSubmode =
    genMode && !isEdit && !isCompose && !isAnimate && !isControl && initSource
      ? "img2img"
      : "txt2img";
  const initPreview = images.length > 0 ? images[0] : pinnedInit;
  // compose blends every attached image, else every pinned one.
  const composeCount = images.length > 0 ? images.length : pinnedCount;
  // edit: everything after the first source image is a subject reference.
  const editRefCount = Math.max((images.length > 0 ? images.length : pinnedCount) - 1, 0);

  // control: the same first-is-the-source split edit uses.
  const controlRefCount = editRefCount;
  const readyKinds = new Set(preprocessors.filter((p) => p.installed).map((p) => p.kind));
  const toggleKind = (kind: ControlKind) =>
    patchGen({
      controlKinds: gen.controlKinds.includes(kind)
        ? gen.controlKinds.filter((k) => k !== kind)
        : [...gen.controlKinds, kind],
    });
  // Without a source there is nothing to derive a map from and nothing to lock to.
  // The maps the user attached directly still work — that's the re-roll path.
  const hasControlSource = !!initPreview;
  // A studio pose changes what an attachment *means*: by default it's a subject
  // reference and there is no source at all, so the lock has nothing to hold onto
  // even though an image is present. Ticking "use as the scene" is what promotes
  // it back to a source. Everything that keys off "is there a source" has to ask
  // this rather than `hasControlSource`, or the lock offers itself for a run the
  // backend will reject.
  const hasStudioMaps = studioMaps.length > 0;
  const studioScene = hasStudioMaps && gen.studioSource && !!initPreview;
  const controlSourceActive = hasStudioMaps ? studioScene : hasControlSource;

  /** What the backend will do with attachment `i`, in one word. Mirrors the split
   *  in `generateImage`; "" where the mode gives every image the same job. */
  function imageRole(i: number): string {
    if (isCompose) return "";
    if (isControl) {
      if (hasStudioMaps) return studioScene && i === 0 ? "scene" : "subject ref";
      if (i > 0) return "subject ref";
      return gen.controlKinds.length === 0 ? "control map" : "structure";
    }
    if (isEdit) return i === 0 ? "scene" : "subject ref";
    return "";
  }
  // Whether the model that will run has an adapter flagged as its control adapter.
  // Without one, FLUX.2 treats a control map as an image to *emulate* rather than a
  // structure to follow — the classic symptom being a result that comes back looking
  // like the greyscale depth map itself. Worth saying in the tab, not in a status
  // line that scrolls away before the image lands.
  const controlAdapter = fluxModels.find((m) => m.name === activeFlux)?.loras?.some((l) => l.control);
  // Instruction-shaped prompts are the single most common way this mode is misused,
  // and the failure is silent: the image comes back looking like the control map and
  // nothing says why. A crude opener test catches nearly all of them and can only
  // ever produce a hint, never a block.
  const instructionPrompt =
    /^\s*(make|have|turn|change|put|move|give|let|pose|set|adjust|fix|edit|redo|add|remove)\b/i.test(text) &&
    text.trim().length > 0;

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
            aria-selected={genOp === "control"}
            className={`mode-tab ${genOp === "control" ? "active" : ""}`}
            onClick={() => setGenOp("control")}
            title="Copy the pose or layout of a reference image into a new one"
          >
            🕹️ Control
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

      {/* Two independent notices. They were one, which was a mistake: flagging a
          control adapter made the prompt advice disappear along with it, and the
          prompt is the part that actually decides whether you get a photograph or a
          greyscale copy of the map. */}
      {genMode && isControl && instructionPrompt && (
        <div className="init-hint warn">
          <span>
            ⚠️ That reads like an <strong>edit instruction</strong>. Control mode can't
            change a pose — it copies the one in the map — and an instruction gives the
            model nothing to render, so it falls back on imitating the map and returns a
            greyscale image. Describe the <em>finished picture</em> instead: “a man in a
            black shirt standing with both arms straight out, aviation museum,
            photorealistic”. To change a pose in your own photo, use{" "}
            <button className="link-btn" onClick={() => setGenOp("edit")}>✏️ Edit</button>.
          </span>
        </div>
      )}

      {genMode && isControl && !controlAdapter && (
        <div className="init-hint warn">
          <span>
            ⚠️ <strong>{prettyFlux(activeFlux) || "This model"}</strong> has no control
            adapter flagged, so the map will guide only loosely. Tick{" "}
            <em>control adapter</em> on a pose/control LoRA under{" "}
            <strong>🖼️ Image Models</strong>.
          </span>
        </div>
      )}

      {genMode && isControl && hasStudioMaps && (
        <div className="init-hint">
          <div className="studio-maps">
            {studioMaps.map((m) => (
              <figure key={m.kind}>
                <img src={m.url} alt={`${m.kind} map`} />
                <figcaption>{m.kind}</figcaption>
              </figure>
            ))}
            {/* The scene photo sits in the same strip once it's been promoted to
                a source, because at that point it is part of the same signal —
                keeping it in the attachment row below implied it was a reference. */}
            {studioScene && initPreview && (
              <figure>
                <img src={initPreview} alt="scene" />
                <figcaption>scene</figcaption>
              </figure>
            )}
          </div>
          <span>
            Posed in the studio — generating on {studioMaps.map((m) => m.kind).join(" + ")}
            {studioMeta && studioMeta.subjects > 1 ? (
              <>
                , <strong>{studioMeta.subjects} figures</strong>
                {studioMeta.contact ? " in contact" : ""}
              </>
            ) : null}
            . Describe the image you want built on this pose
            {studioMeta && studioMeta.subjects > 1
              ? " — say there are " +
                studioMeta.subjects +
                " people and what each is doing, or the model will render one."
              : "."}{" "}
            <button className="link-btn" onClick={onOpenStudio}>Edit pose</button>
            {" · "}
            <button className="link-btn" onClick={onClearStudioMaps}>Clear</button>
            {initPreview && (
              <>
                <br />
                <label className="studio-scene-toggle">
                  <input
                    type="checkbox"
                    checked={gen.studioSource}
                    onChange={(e) => patchGen({ studioSource: e.target.checked })}
                  />
                  Use the {initSource === "attached" ? "attached" : "pinned"} image as the{" "}
                  <em>scene</em> to pose into
                </label>{" "}
                {studioScene
                  ? "— it's the source now, so the structure lock is live and any control types you pick are derived from it and stacked with the studio's maps."
                  : "— off, it's a subject reference: it lends a face and clothing, not a place."}
              </>
            )}
          </span>
        </div>
      )}

      {genMode && isControl && !hasStudioMaps && (
        <div className="init-hint">
          {initPreview && (
            <img className="init-thumb" src={initPreview} alt="control source" />
          )}
          <span>
            {hasControlSource ? (
              <>
                Copying the structure of this image ({initSource === "attached" ? "attached" : "from panel"})
                {controlRefCount > 0 ? (
                  <>, with the other {controlRefCount} as subject reference{controlRefCount > 1 ? "s" : ""}</>
                ) : null}
                . Describe the image you want <em>built on that structure</em> — the pose comes
                from the picture, everything else from your prompt. This copies the pose that
                is already in the photo; to build a different one,{" "}
                <button className="link-btn" onClick={onOpenStudio}>
                  🧍 open the Pose Studio
                </button>.
              </>
            ) : (
              <>
                <button className="link-btn strong" onClick={onOpenStudio}>
                  🧍 Open the Pose Studio
                </button>{" "}
                to build the pose in 3-D — that's the way to get a pose you can't
                photograph. Or attach an image whose <em>pose or layout</em> you want copied;
                it has to already contain whatever the subject is touching, since a bare
                skeleton can't say “on a chair”.
              </>
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
                {/* An attachment's job has always been decided by its position in
                    this list, and nothing said so. It matters most here, where a
                    photo is either the place the scene happens or the person in
                    it, and the two produce completely different images. */}
                {genMode && imageRole(i) && <span className="thumb-role">{imageRole(i)}</span>}
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
                      Model ({role === "edit" ? "edit / combine / control" : "create"})
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
                    <input type="number" min={4} max={60} value={gen.steps}
                      onChange={(e) => patchGen({ steps: +e.target.value })} />
                  </label>
                  <label>Guidance
                    <input type="number" min={0.5} max={10} step={0.5} value={gen.guidance}
                      onChange={(e) => patchGen({ guidance: +e.target.value })} />
                  </label>
                  {/* Size is only the user's to set when nothing else fixes it. A
                      locked control run starts from the source's latent, which decides
                      the shape the way img2img's does. */}
                  {(genOp === "create" || (isControl && gen.structureLock >= 1)) && (
                    <>
                      <label>Width
                        <input type="number" min={256} max={1536} step={64} value={gen.width}
                          onChange={(e) => patchGen({ width: +e.target.value })} />
                      </label>
                      <label>Height
                        <input type="number" min={256} max={1536} step={64} value={gen.height}
                          onChange={(e) => patchGen({ height: +e.target.value })} />
                      </label>
                    </>
                  )}
                  {genOp === "create" && (
                    <label className={genSubmode === "img2img" ? "" : "muted-field"}>
                      Strength
                      <input type="number" min={0} max={1} step={0.05} value={gen.strength}
                        disabled={genSubmode !== "img2img"}
                        onChange={(e) => patchGen({ strength: +e.target.value })} />
                    </label>
                  )}
                  <label>Seed
                    <input type="text" inputMode="numeric" value={gen.seed}
                      placeholder="random"
                      onChange={(e) => patchGen({ seed: e.target.value.replace(/[^0-9]/g, "") })} />
                  </label>
                </div>

                {isControl && (
                  <div className="control-settings">
                    <button className="btn studio-open" onClick={onOpenStudio}>
                      🧍 Pose Studio — build a pose in 3-D
                    </button>
                    <p className="hint muted">
                      Pose a figure, put a box where the chair is, and it renders its own
                      depth and OpenPose maps. This is the route for a pose you can't find a
                      photo of.
                    </p>

                    <label className="lbl">Or derive maps from a source image</label>
                    <div className="control-kinds">
                      {CONTROL_KINDS.map(({ kind, label, hint }) => {
                        const ready = readyKinds.has(kind);
                        const on = gen.controlKinds.includes(kind);
                        return (
                          <button
                            key={kind}
                            type="button"
                            className={`control-chip ${on ? "on" : ""} ${ready ? "" : "unavailable"}`}
                            disabled={!ready}
                            title={ready ? hint : `${hint}\n\nNot installed — add it under 🖼️ Image Models → Control preprocessors.`}
                            onClick={() => toggleKind(kind)}
                          >
                            {on ? "✓ " : ""}{label}
                            {!ready && <span className="chip-badge">install</span>}
                          </button>
                        );
                      })}
                    </div>
                    <p className="hint muted">
                      Depth is the one to reach for first, and Depth + Pose together is the
                      pair that holds a hard pose: depth carries the scene and the contact,
                      pose pins which limb is which. Select none to use a control map you
                      attached as-is.
                    </p>

                    <label className="lbl">
                      Structure lock — {gen.structureLock >= 1
                        ? "off (composition is free)"
                        : gen.structureLock >= 0.8
                          ? `${gen.structureLock.toFixed(2)} · loose`
                          : gen.structureLock >= 0.6
                            ? `${gen.structureLock.toFixed(2)} · firm`
                            : `${gen.structureLock.toFixed(2)} · pinned`}
                    </label>
                    <input
                      className="control-slider"
                      type="range" min={0.4} max={1} step={0.05}
                      value={gen.structureLock}
                      disabled={!controlSourceActive}
                      onChange={(e) => patchGen({ structureLock: +e.target.value })}
                    />
                    <p className="hint muted">
                      {controlSourceActive
                        ? "This is the dial that decides whether the pose is suggested or held. At 1 the maps only guide; lower it and generation starts from the source image itself, so its geometry survives. Below ~0.6 the source's own appearance starts coming through as well — sweep down until the pose lands, then back off."
                        : hasStudioMaps
                          ? "A studio pose is the whole signal on its own — there's no source image to lock onto. Attach the photo you want the figures posed into and tick “use it as the scene”."
                          : "Needs a source image — there's nothing to lock onto yet."}
                    </p>

                    <label className="lbl">
                      Control adapter strength — {gen.controlStrength.toFixed(2)}
                    </label>
                    <input
                      className="control-slider"
                      type="range" min={0} max={1.5} step={0.05}
                      value={gen.controlStrength}
                      onChange={(e) => patchGen({ controlStrength: +e.target.value })}
                    />
                    <p className="hint muted">
                      Weight of the LoRA you flagged as this model's control adapter, for this
                      generation only. Does nothing if none is flagged.
                    </p>

                    {gen.controlKinds.includes("canny") && (
                      <div className="gen-grid">
                        <label>Edge low
                          <input type="number" min={0.01} max={0.99} step={0.05}
                            value={gen.cannyLow}
                            onChange={(e) => patchGen({ cannyLow: +e.target.value })} />
                        </label>
                        <label>Edge high
                          <input type="number" min={0.01} max={0.99} step={0.05}
                            value={gen.cannyHigh}
                            onChange={(e) => patchGen({ cannyHigh: +e.target.value })} />
                        </label>
                      </div>
                    )}
                  </div>
                )}
                <p className="hint muted">
                  {isAnimate
                    ? "Attach one image — it becomes the first frame. Describe the motion, not the scene; the frame already fixes that. Takes a few minutes."
                    : isControl
                    ? "The control maps are built first and shown with the result, so you can see whether a miss was the map's fault or the model's."
                    : role === "edit"
                      ? "Guidance ~2.5 follows the instruction closely. If the source comes back unchanged, lower it — raising it makes the model cling to the reference instead of editing harder."
                      : genSubmode === "img2img"
                        ? "Image-to-image: strength controls how far from the attached image."
                        : "Text-to-image: guidance ~3.5. Attach an image above to switch to image-to-image."}
                </p>
              </div>
            )}
          </div>
        )}

        {/* Sits in the composer row, beside the attach button, for every control
            generation — not only when nothing is attached. It used to appear solely
            in the empty-state hint and inside the ⚙️ popover, so attaching a photo
            hid the studio entirely and there was no visible way back to it. */}
        {genMode && isControl && (
          <button
            className={`btn icon studio-btn ${studioMaps.length ? "has-dot" : ""}`}
            title="Pose Studio — build a pose in 3-D"
            onClick={onOpenStudio}
          >
            🧍{studioMaps.length > 0 && <span className="dot" />}
          </button>
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
                : isControl
                ? "Describe the image to build on this pose… e.g. “a woman in a red sari, on a beach at sunrise”"
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
            title={genMode && enhancing ? "Enhancing prompt…" : genMode ? "Generate image" : "Send"}
            disabled={
              genMode
                ? !text.trim() || enhancing
                : disabled || (!text.trim() && images.length === 0)
            }
          >
            {genMode ? (enhancing ? "✨" : "🎨") : "➤"}
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
