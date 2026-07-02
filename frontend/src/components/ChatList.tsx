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
                className="chat-del"
                title="Delete chat"
                onClick={(e) => {
                  e.stopPropagation();
                  if (confirm("Delete this chat?")) onDeleteChat(c.id);
                }}
              >
                ✕
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
