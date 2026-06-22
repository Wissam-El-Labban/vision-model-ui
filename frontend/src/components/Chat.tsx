import { useEffect, useRef } from "react";
import Composer from "./Composer";
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

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div className="chat">
      <div className="messages">
        {messages.length === 0 && (
          <div className="empty">
            <p className="empty-emoji">🖼️</p>
            <p>Drop one or more images and ask a question.</p>
            <p className="muted">
              Attach as many images as you like — in your first question or any
              follow-up.
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
                <div className="content">{m.content}</div>
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
        onSend={onSend}
        onStop={onStop}
        streaming={streaming}
        disabled={disabled}
      />
    </div>
  );
}
