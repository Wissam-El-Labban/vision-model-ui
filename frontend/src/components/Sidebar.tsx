import { useRef, useState } from "react";
import ImageModels from "./ImageModels";
import ModelManager from "./ModelManager";
import ThinkingSettings from "./ThinkingSettings";
import UpdateBanner from "./UpdateBanner";
import ChatList from "./ChatList";
import Documents from "./Documents";
import type { DocumentsApi } from "../useDocuments";
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
  runningChats: Set<string>;
  thinking: boolean;
  setThinking: (v: boolean) => void;
  thinkingDisplay: ThinkingDisplay;
  setThinkingDisplay: (v: ThinkingDisplay) => void;
  docs: DocumentsApi;
  activeDocIds: string[];
  setActiveDocIds: (ids: string[]) => void;
  docPrompts: boolean;
  setDocPrompts: (v: boolean) => void;
  docsOpen: boolean;
  setDocsOpen: (v: boolean) => void;
  ocrModel: string;
}

// Drag range for the sidebar's width. The upper bound also caps at 70% of the window
// so the chat can never be squeezed out entirely.
const DEFAULT_WIDTH = 320;
const MIN_WIDTH = 260;
const MAX_WIDTH = 900;
const WIDTH_KEY = "sidebarWidth";

const clampWidth = (w: number) =>
  Math.round(Math.max(MIN_WIDTH, Math.min(w, MAX_WIDTH, window.innerWidth * 0.7)));

function savedWidth(): number {
  try {
    const n = Number(localStorage.getItem(WIDTH_KEY));
    return n ? clampWidth(n) : DEFAULT_WIDTH;
  } catch {
    return DEFAULT_WIDTH;
  }
}

function saveWidth(w: number) {
  try {
    localStorage.setItem(WIDTH_KEY, String(w));
  } catch {
    /* a private window: the width just doesn't persist */
  }
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
    runningChats,
    thinking,
    setThinking,
    thinkingDisplay,
    setThinkingDisplay,
    docs,
    activeDocIds,
    setActiveDocIds,
    docPrompts,
    setDocPrompts,
    docsOpen,
    setDocsOpen,
    ocrModel,
  } = props;
  const [urlDraft, setUrlDraft] = useState(ollamaUrl);
  const [width, setWidth] = useState(savedWidth);
  const drag = useRef<{ x: number; w: number } | null>(null);

  // Drag the edge facing the chat. Pointer capture keeps the drag alive when the
  // cursor outruns the 6 px handle; the width is saved once, on release.
  function startDrag(e: React.PointerEvent<HTMLDivElement>) {
    e.preventDefault();
    e.currentTarget.setPointerCapture(e.pointerId);
    drag.current = { x: e.clientX, w: width };
    document.body.classList.add("resizing");
  }
  function onDrag(e: React.PointerEvent<HTMLDivElement>) {
    if (!drag.current) return;
    setWidth(clampWidth(drag.current.w + e.clientX - drag.current.x));
  }
  function endDrag() {
    if (!drag.current) return;
    drag.current = null;
    document.body.classList.remove("resizing");
    saveWidth(width);
  }
  function resetWidth() {
    setWidth(DEFAULT_WIDTH);
    saveWidth(DEFAULT_WIDTH);
  }

  return (
    <>
      <aside className="sidebar" style={{ width }}>
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
          runningChats={runningChats}
        />

        <Documents
          docs={docs}
          activeIds={activeDocIds}
          setActiveIds={setActiveDocIds}
          docPrompts={docPrompts}
          setDocPrompts={setDocPrompts}
          ollamaUrl={ollamaUrl}
          ocrModel={ocrModel}
          open={docsOpen}
          setOpen={setDocsOpen}
        />

        <ModelManager ollamaUrl={ollamaUrl} allModels={models.all} onChanged={refreshModels} />

        <ThinkingSettings
          thinking={thinking}
          setThinking={setThinking}
          display={thinkingDisplay}
          setDisplay={setThinkingDisplay}
        />

        <ImageModels models={fluxModels} onChanged={refreshFlux} />

        <div className="spacer" />
        <p className="footer-note muted small">Powered by Ollama · images sent per message</p>
      </aside>
      {/* A sibling of the sidebar, not a child: inside it, the handle would scroll
        away with the sidebar's content. */}
      <div
        className="sidebar-resizer"
        role="separator"
        aria-orientation="vertical"
        title="Drag to resize · double-click to reset"
        onPointerDown={startDrag}
        onPointerMove={onDrag}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
        onDoubleClick={resetWidth}
      />
    </>
  );
}
