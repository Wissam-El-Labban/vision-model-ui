import { useState } from "react";
import ModelManager from "./ModelManager";
import UpdateBanner from "./UpdateBanner";
import ChatList from "./ChatList";
import type { ChatSummary } from "../types";

interface Props {
  ollamaUrl: string;
  setOllamaUrl: (v: string) => void;
  models: { vision: string[]; all: string[] };
  refreshModels: () => void;
  chats: ChatSummary[];
  currentChatId: string;
  onNewChat: () => void;
  onOpenChat: (id: string) => void;
  onDeleteChat: (id: string) => void;
}

export default function Sidebar(props: Props) {
  const {
    ollamaUrl,
    setOllamaUrl,
    models,
    refreshModels,
    chats,
    currentChatId,
    onNewChat,
    onOpenChat,
    onDeleteChat,
  } = props;
  const [urlDraft, setUrlDraft] = useState(ollamaUrl);

  return (
    <aside className="sidebar">
      <h2 className="brand">⚙️ Settings</h2>

      <UpdateBanner ollamaUrl={ollamaUrl} />

      <label className="lbl">🌐 Ollama URL</label>
      <div className="row">
        <input
          value={urlDraft}
          onChange={(e) => setUrlDraft(e.target.value)}
          onBlur={() => setOllamaUrl(urlDraft)}
          onKeyDown={(e) => e.key === "Enter" && setOllamaUrl(urlDraft)}
        />
        <button className="btn" title="Reconnect" onClick={refreshModels}>
          ⟳
        </button>
      </div>

      <ChatList
        chats={chats}
        currentChatId={currentChatId}
        onNewChat={onNewChat}
        onOpenChat={onOpenChat}
        onDeleteChat={onDeleteChat}
      />

      <ModelManager
        ollamaUrl={ollamaUrl}
        allModels={models.all}
        onChanged={refreshModels}
      />

      <div className="spacer" />
      <p className="footer-note muted small">
        Powered by Ollama · images sent per message
      </p>
    </aside>
  );
}
