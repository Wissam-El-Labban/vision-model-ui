import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import Composer from "./Composer";
import { fileToResizedDataUrl, rotateDataUrl } from "../fileUtils";
import type { ChatMessage } from "../types";

interface Props {
  messages: ChatMessage[];
  streaming: boolean;
  onSend: (text: string, images: string[]) => void;
  onStop: () => void;
  disabled: boolean;
}

export default function Chat({ messages, streaming, onSend, onStop, disabled }: Props) {
  const endRef = useRef<HTMLDivElement>(null);
  const [text, setText] = useState("");
  const [attach, setAttach] = useState<string[]>([]);
  const [dragOver, setDragOver] = useState(false);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function addFiles(files: FileList | File[]) {
    const list = Array.from(files).filter((f) => f.type.startsWith("image/"));
    if (list.length === 0) return;
    const urls = await Promise.all(list.map((f) => fileToResizedDataUrl(f)));
    setAttach((prev) => [...prev, ...urls]);
  }
  function removeImage(i: number) {
    setAttach((prev) => prev.filter((_, idx) => idx !== i));
  }
  async function rotateImage(i: number) {
    const rotated = await rotateDataUrl(attach[i], 90);
    setAttach((prev) => prev.map((img, idx) => (idx === i ? rotated : img)));
  }
  function submit() {
    const trimmed = text.trim();
    if (!trimmed && attach.length === 0) return;
    onSend(trimmed, attach);
    setText("");
    setAttach([]);
  }

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
          if (!disabled) addFiles(e.dataTransfer.files);
        }}
      >
        {dragOver && (
          <div className="drop-overlay">📎 Drop to attach to your next message</div>
        )}
        {messages.length === 0 && (
          <div className="empty">
            <p className="empty-emoji">💬</p>
            <p>Add your image(s) to the bar above, then ask away.</p>
            <p className="muted">
              Need to add more context mid-chat? Drop images right here (or use
              📎) — they'll appear inline below.
            </p>
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`msg ${m.role}`}>
            <div className="avatar">{m.role === "user" ? "🧑" : "🤖"}</div>
            <div className="bubble">
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
        ))}
        <div ref={endRef} />
      </div>
      <Composer
        text={text}
        setText={setText}
        images={attach}
        onAddFiles={addFiles}
        onRemoveImage={removeImage}
        onRotateImage={rotateImage}
        onSubmit={submit}
        onStop={onStop}
        streaming={streaming}
        disabled={disabled}
      />
    </div>
  );
}
