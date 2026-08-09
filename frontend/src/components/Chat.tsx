import { Fragment, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { AgentStep, ChatMessage, GenProgress } from "../types";
import GenProgressBar from "./GenProgressBar";

/** Stable color per model name, for the per-chunk indicator. */
function modelColor(model?: string): string {
  if (!model) return "var(--border)";
  let h = 0;
  for (const c of model) h = (h * 31 + c.charCodeAt(0)) % 360;
  return `hsl(${h} 60% 58%)`;
}

const ORDINALS: Record<string, number> = {
  first: 1, second: 2, third: 3, fourth: 4, fifth: 5,
  sixth: 6, seventh: 7, eighth: 8, ninth: 9, tenth: 10,
};

/** Save an image to disk. Works for both data URLs and same-origin
 *  /api/images/<hash>.<ext> URLs; the filename extension follows the source. */
function downloadImage(src: string): void {
  let name = "generated-image.png";
  const mime = src.match(/^data:image\/(\w+)/);
  if (mime) {
    name = `generated-image.${mime[1] === "jpeg" ? "jpg" : mime[1]}`;
  } else {
    const last = src.split("/").pop() || "";
    if (last.includes(".")) name = last;
  }
  const a = document.createElement("a");
  a.href = src;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
}

/** Inject a placeholder thumbnail (`![](ctx://idx)`) after the model's manifest
 *  references — "image N", "the Nth image", "pinned reference image" — so the
 *  <img> renderer below can turn each into a clickable thumbnail. Every mention
 *  gets its own thumbnail; only indices that actually exist in this turn's
 *  context list are annotated. */
function annotateImageRefs(content: string, count: number): string {
  if (count <= 0) return content;
  const inject = (whole: string, n: number): string =>
    n < 1 || n > count ? whole : `${whole} ![](ctx://${n - 1})`;
  return content
    .replace(/\bimages?\s*#?\s*(\d{1,2})\b/gi, (m, n) => inject(m, parseInt(n, 10)))
    .replace(
      /\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+image\b/gi,
      (m, ord) => inject(m, ORDINALS[ord.toLowerCase()])
    )
    .replace(/\bpinned reference images?\b/gi, (m) => inject(m, 1));
}

/** How the chip labels one tool call, image numbers included.
 *
 * The numbers are the point: "Edit image 2" is the difference between the agent
 * having understood which photo was meant and having guessed. Reading them off
 * the arguments rather than out of a separate field keeps this honest — it says
 * what was actually sent. */
function stepLabel(step: AgentStep): string {
  const args = step.args ?? {};
  const nums = (v: unknown): string =>
    Array.isArray(v) ? v.join(", ") : v === undefined || v === null ? "" : String(v);
  switch (step.name) {
    case "create_image":
      return args.source_image
        ? `🖼️ Create from image ${nums(args.source_image)}`
        : "🖼️ Create image";
    case "edit_image":
      return `✏️ Edit image ${nums(args.image)}`;
    case "combine_images":
      return `🧩 Combine images ${nums(args.images)}`;
    case "animate_image":
      return `🎬 Animate image ${nums(args.image)}`;
    default:
      return step.name;
  }
}

/** The prompt the agent wrote for a step — the one argument worth reading. */
function stepPrompt(step: AgentStep): string {
  const args = step.args ?? {};
  const text = args.prompt ?? args.instruction ?? args.motion;
  return typeof text === "string" ? text : "";
}

/** What agent mode decided to do, above the images it produced.
 *
 * Collapsed by default: the answer to "why does this image look like this" is a
 * paragraph of generated prompt, which is worth having but not worth reading
 * every turn. */
function AgentSteps({ steps }: { steps: AgentStep[] }) {
  const [open, setOpen] = useState<number | null>(null);
  return (
    <div className="agent-steps">
      {steps.map((step, i) => {
        const prompt = stepPrompt(step);
        return (
          <div key={i} className={`agent-step ${step.state}`}>
            <button
              type="button"
              className="agent-step-head"
              onClick={() => setOpen(open === i ? null : i)}
              disabled={!prompt && !step.message}
              title={prompt || step.message || ""}
            >
              <span className="agent-step-state" aria-hidden>
                {step.state === "running" ? "⏳" : step.state === "error" ? "⚠️" : "✓"}
              </span>
              <span className="agent-step-label">{stepLabel(step)}</span>
              {(prompt || step.message) && (
                <span className="chev" aria-hidden>{open === i ? "▾" : "▸"}</span>
              )}
            </button>
            {step.message && <div className="agent-step-error">{step.message}</div>}
            {open === i && prompt && <div className="agent-step-prompt">{prompt}</div>}
          </div>
        );
      })}
    </div>
  );
}

interface Props {
  messages: ChatMessage[];
  streaming: boolean;
  disabled: boolean;
  /** Liveness of the in-flight turn; null when idle. Drives the bar below. */
  progress: GenProgress | null;
  onDropFiles: (files: FileList | File[]) => void;
}

/** The scrolling conversation (the composer lives full-width below it). */
export default function Chat({
  messages,
  streaming,
  disabled,
  progress,
  onDropFiles,
}: Props) {
  const endRef = useRef<HTMLDivElement>(null);
  const [dragOver, setDragOver] = useState(false);
  const [zoom, setZoom] = useState<string | null>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Close the expanded image on Escape.
  useEffect(() => {
    if (!zoom) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setZoom(null);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [zoom]);

  return (
    <div className="chat">
      <div
        className={`messages ${dragOver ? "drag" : ""}`}
        onDragOver={(e) => {
          if (disabled) return;
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          if (!disabled) onDropFiles(e.dataTransfer.files);
        }}
      >
        {dragOver && (
          <div className="drop-overlay">📎 Drop to attach to your next message</div>
        )}
        {messages.length === 0 && (
          <div className="empty">
            <p className="empty-emoji">💬</p>
            <p>Add image(s) to the panel on the left, then ask away.</p>
            <p className="muted">
              Need more context mid-chat? Drop images here (or use 📎) — they'll
              appear inline below.
            </p>
          </div>
        )}
        {messages.map((m, i) => {
          const prevModel = i > 0 ? messages[i - 1].model : undefined;
          const showModel = !!m.model && m.model !== prevModel;
          const color = modelColor(m.model);
          return (
            <Fragment key={i}>
              {showModel && (
                <div className="model-divider">
                  <span className="model-chip" style={{ borderColor: color }}>
                    <span className="model-dot" style={{ background: color }} />
                    {m.model}
                  </span>
                </div>
              )}
              <div className={`msg ${m.role}`}>
                <div className="avatar">{m.role === "user" ? "🧑" : "🤖"}</div>
                <div className="bubble" style={{ ["--mc" as string]: color }}>
                  {/* What the agent decided, above what it produced: the chips
                      explain the images below them, so they have to be read
                      first. */}
                  {m.agentSteps && m.agentSteps.length > 0 && (
                    <AgentSteps steps={m.agentSteps} />
                  )}
                  {m.videos && m.videos.length > 0 && (
                    <div className="msg-images">
                      {m.videos.map((src, j) => (
                        // Not zoomable like an image: <video> owns its own click
                        // (play/pause), and the zoom overlay renders an <img>.
                        <div key={j} className="msg-video">
                          {/* muted is required or Chrome refuses to autoplay;
                              playsInline or iOS takes it fullscreen. */}
                          <video src={src} autoPlay loop muted playsInline controls />
                          <a
                            className="video-dl"
                            href={src}
                            download={`animate-${src.slice(-11, -5)}.webm`}
                            title="Download video"
                          >
                            ⬇
                          </a>
                        </div>
                      ))}
                    </div>
                  )}
                  {m.images && m.images.length > 0 && (
                    <div className="msg-images">
                      {m.images.map((src, j) => (
                        <button
                          key={j}
                          type="button"
                          className="msg-image-btn"
                          title="Click to expand"
                          onClick={() => setZoom(src)}
                        >
                          <img src={src} alt={`image ${j + 1}`} />
                        </button>
                      ))}
                    </div>
                  )}
                  {/* The control maps this turn conditioned on. Shown small and
                      after the result, because they're diagnostic rather than the
                      output: when a pose comes out wrong, the image alone can't tell
                      you whether the map was bad or the model ignored a good one. */}
                  {m.controlMaps && m.controlMaps.length > 0 && (
                    <div className="control-maps">
                      {m.controlMaps.map((cm, j) => (
                        <button
                          key={j}
                          type="button"
                          className="control-map"
                          title={`${cm.kind} map — click to expand`}
                          onClick={() => setZoom(cm.url)}
                        >
                          <img src={cm.url} alt={`${cm.kind} control map`} />
                          <span className="control-map-kind">{cm.kind}</span>
                        </button>
                      ))}
                    </div>
                  )}
                  {m.content ? (
                    m.role === "assistant" ? (
                      <div className="content markdown">
                        <ReactMarkdown
                          remarkPlugins={[remarkGfm]}
                          // Keep our ctx:// placeholder scheme (default strips it).
                          urlTransform={(url) => url}
                          components={{
                            img: ({ src, alt }) => {
                              const ctx = m.contextImages;
                              if (typeof src === "string" && src.startsWith("ctx://") && ctx) {
                                const idx = parseInt(src.slice(6), 10);
                                const img = ctx[idx];
                                if (!img) return null;
                                return (
                                  <button
                                    type="button"
                                    className="ctx-thumb"
                                    title="Referenced image — click to expand"
                                    onClick={() => setZoom(img)}
                                  >
                                    <img src={img} alt={`referenced image ${idx + 1}`} />
                                  </button>
                                );
                              }
                              return <img src={src} alt={alt} />;
                            },
                          }}
                        >
                          {annotateImageRefs(m.content, m.contextImages?.length ?? 0)}
                        </ReactMarkdown>
                      </div>
                    ) : (
                      <div className="content">{m.content}</div>
                    )
                  ) : (
                    streaming &&
                    i === messages.length - 1 && <span className="cursor">▋</span>
                  )}
                  {m.role === "user" && m.enhancedPrompt && (
                    <div className="content enhanced-prompt muted small">
                      ✨ Sent as: {m.enhancedPrompt}
                    </div>
                  )}
                </div>
              </div>
            </Fragment>
          );
        })}
        <div ref={endRef} />
      </div>
      <GenProgressBar progress={progress} />
      {zoom && (
        <div className="lightbox" onClick={() => setZoom(null)}>
          <div className="lightbox-body" onClick={(e) => e.stopPropagation()}>
            <img src={zoom} alt="full size" />
            <div className="lightbox-bar">
              <button type="button" className="btn" onClick={() => downloadImage(zoom)}>
                ⬇ Download
              </button>
              <button type="button" className="btn ghost" onClick={() => setZoom(null)}>
                ✕ Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
