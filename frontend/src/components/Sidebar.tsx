import { useState } from "react";
import ImageModels from "./ImageModels";
import ModelManager from "./ModelManager";
import PromptEnhancer from "./PromptEnhancer";
import UpdateBanner from "./UpdateBanner";
import ChatList from "./ChatList";
import type { ChatSummary, EnhancerMode } from "../types";
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
  enhancerModel: string;
  setEnhancerModel: (v: string) => void;
  enhancerMode: EnhancerMode;
  setEnhancerMode: (v: EnhancerMode) => void;
  enhanceTemplate: boolean;
  setEnhanceTemplate: (v: boolean) => void;
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
    enhancerModel,
    setEnhancerModel,
    enhancerMode,
    setEnhancerMode,
    enhanceTemplate,
    setEnhanceTemplate,
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

      <PromptEnhancer
        visionModels={models.vision}
        enhancerModel={enhancerModel}
        setEnhancerModel={setEnhancerModel}
        enhancerMode={enhancerMode}
        setEnhancerMode={setEnhancerMode}
        enhanceTemplate={enhanceTemplate}
        setEnhanceTemplate={setEnhanceTemplate}
      />

      <ImageModels models={fluxModels} onChanged={refreshFlux} />

      <div className="spacer" />
      <p className="footer-note muted small">
        Powered by Ollama · images sent per message
      </p>
    </aside>
  );
}
