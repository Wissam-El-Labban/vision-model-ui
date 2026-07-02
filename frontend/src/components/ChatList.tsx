import { useEffect, useState } from "react";
import type { ChatSummary } from "../types";

interface Props {
  chats: ChatSummary[];
  currentChatId: string;
  onNewChat: () => void;
  onOpenChat: (id: string) => void;
  onDeleteChat: (id: string) => void;
}

/** List of saved conversations. Each row shows the chat's first pinned image(s)
 *  as small icons plus its LLM-generated title. Lives in the sidebar under the
 *  Ollama URL. */
export default function ChatList({
  chats,
  currentChatId,
  onNewChat,
  onOpenChat,
  onDeleteChat,
}: Props) {
  // Two-step delete guardrail: first click on ✕ arms "Delete?"; a second click
  // deletes. Auto-reverts after a few seconds (and on leaving the row).
  const [confirmId, setConfirmId] = useState<string | null>(null);
  useEffect(() => {
    if (!confirmId) return;
    const t = setTimeout(() => setConfirmId(null), 5000);
    return () => clearTimeout(t);
  }, [confirmId]);

  return (
    <div className="chat-list">
      <div className="chat-list-head">
        <label className="lbl">💬 Chats</label>
        <button className="btn ghost small" onClick={onNewChat}>
          ＋ New
        </button>
      </div>

      {chats.length === 0 ? (
        <p className="muted small chat-empty">No saved chats yet.</p>
      ) : (
        <ul className="chat-items">
          {chats.map((c) => (
            <li
              key={c.id}
              className={`chat-item ${c.id === currentChatId ? "active" : ""}`}
              onClick={() => onOpenChat(c.id)}
              title={c.title ?? "Untitled chat"}
            >
              <div className="chat-icons">
                {c.icons.length > 0 ? (
                  c.icons.map((src, i) => (
                    <img key={i} src={src} alt="" />
                  ))
                ) : (
                  <span className="chat-icon-blank">🗨️</span>
                )}
              </div>
              <span className="chat-title">
                {c.title ?? <span className="muted">Untitled…</span>}
              </span>
              <button
                className={`chat-del ${confirmId === c.id ? "confirming" : ""}`}
                title={confirmId === c.id ? "Click again to delete" : "Delete chat"}
                onClick={(e) => {
                  e.stopPropagation();
                  if (confirmId === c.id) {
                    onDeleteChat(c.id);
                    setConfirmId(null);
                  } else {
                    setConfirmId(c.id);
                  }
                }}
              >
                {confirmId === c.id ? "Delete?" : "✕"}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
