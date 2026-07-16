import { Fragment, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { ChatMessage } from "../types";

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

interface Props {
  messages: ChatMessage[];
  streaming: boolean;
  disabled: boolean;
  onDropFiles: (files: FileList | File[]) => void;
}

/** The scrolling conversation (the composer lives full-width below it). */
export default function Chat({ messages, streaming, disabled, onDropFiles }: Props) {
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
                </div>
              </div>
            </Fragment>
          );
        })}
        <div ref={endRef} />
      </div>
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
