import { useState } from "react";
import ImageModels from "./ImageModels";
import ModelManager from "./ModelManager";
import ThinkingSettings from "./ThinkingSettings";
import UpdateBanner from "./UpdateBanner";
import ChatList from "./ChatList";
import type { ChatSummary, ThinkingDisplay } from "../types";
import type { FluxModel } from "../api";

interface Props {
  ollamaUrl: string;
  setOllamaUrl: (v: string) => void;
  models: { vision: string[]; all: string[] };
  refreshModels: () => void;
  fluxModels: FluxModel[];
  refreshFlux: () => void;
  chats: ChatSummary[];
  currentChatId: string;
  onNewChat: () => void;
  onOpenChat: (id: string) => void;
  onDeleteChat: (id: string) => void;
  thinking: boolean;
  setThinking: (v: boolean) => void;
  thinkingDisplay: ThinkingDisplay;
  setThinkingDisplay: (v: ThinkingDisplay) => void;
}

export default function Sidebar(props: Props) {
  const {
    ollamaUrl,
    setOllamaUrl,
    models,
    refreshModels,
    fluxModels,
    refreshFlux,
    chats,
    currentChatId,
    onNewChat,
    onOpenChat,
    onDeleteChat,
    thinking,
    setThinking,
    thinkingDisplay,
    setThinkingDisplay,
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

      <ThinkingSettings
        thinking={thinking}
        setThinking={setThinking}
        display={thinkingDisplay}
        setDisplay={setThinkingDisplay}
      />

      <ImageModels models={fluxModels} onChanged={refreshFlux} />

      <div className="spacer" />
      <p className="footer-note muted small">
        Powered by Ollama · images sent per message
      </p>
    </aside>
  );
}
