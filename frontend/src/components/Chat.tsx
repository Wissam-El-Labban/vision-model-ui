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

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

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
                        <img key={j} src={src} alt={`image ${j + 1}`} />
                      ))}
                    </div>
                  )}
                  {m.content ? (
                    m.role === "assistant" ? (
                      <div className="content markdown">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                          {m.content}
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
    </div>
  );
}
