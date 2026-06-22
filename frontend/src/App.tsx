import { useCallback, useEffect, useRef, useState } from "react";
import Sidebar from "./components/Sidebar";
import Chat from "./components/Chat";
import { getModels, streamChat } from "./api";
import type { ChatMessage } from "./types";

const DEFAULT_URL = "http://localhost:11434";

export default function App() {
  const [ollamaUrl, setOllamaUrl] = useState(
    () => localStorage.getItem("ollamaUrl") || DEFAULT_URL
  );
  const [models, setModels] = useState<{ vision: string[]; all: string[] }>({
    vision: [],
    all: [],
  });
  const [model, setModel] = useState(() => localStorage.getItem("model") || "");
  const [systemPrompt, setSystemPrompt] = useState(
    () => localStorage.getItem("systemPrompt") || ""
  );
  const [systemImage, setSystemImage] = useState<string | null>(null);

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Persist settings.
  useEffect(() => localStorage.setItem("ollamaUrl", ollamaUrl), [ollamaUrl]);
  useEffect(() => localStorage.setItem("model", model), [model]);
  useEffect(
    () => localStorage.setItem("systemPrompt", systemPrompt),
    [systemPrompt]
  );

  const refreshModels = useCallback(async () => {
    try {
      const m = await getModels(ollamaUrl);
      setModels(m);
      setModel((cur) =>
        cur && m.vision.includes(cur) ? cur : m.vision[0] ?? cur
      );
      setError(null);
    } catch {
      setError(`Could not reach Ollama at ${ollamaUrl}`);
      setModels({ vision: [], all: [] });
    }
  }, [ollamaUrl]);

  useEffect(() => {
    refreshModels();
  }, [refreshModels]);

  const send = useCallback(
    async (text: string, images: string[]) => {
      if (!model) {
        setError("Select a vision model first.");
        return;
      }
      setError(null);

      const userMsg: ChatMessage = { role: "user", content: text, images };
      const history = [...messages, userMsg];
      setMessages([...history, { role: "assistant", content: "" }]);
      setStreaming(true);

      // Build the request: optional system message (with persistent image) + history.
      const payload: ChatMessage[] = [];
      if (systemPrompt.trim()) {
        payload.push({
          role: "system",
          content: systemPrompt.trim(),
          images: systemImage ? [systemImage] : undefined,
        });
      }
      payload.push(...history);

      const controller = new AbortController();
      abortRef.current = controller;
      try {
        await streamChat(
          ollamaUrl,
          model,
          payload,
          (token) =>
            setMessages((prev) => {
              const next = [...prev];
              next[next.length - 1] = {
                ...next[next.length - 1],
                content: next[next.length - 1].content + token,
              };
              return next;
            }),
          controller.signal
        );
      } catch (e) {
        if ((e as Error).name !== "AbortError") {
          setError(`Could not reach Ollama at ${ollamaUrl}`);
        }
      } finally {
        setStreaming(false);
        abortRef.current = null;
      }
    },
    [messages, model, ollamaUrl, systemPrompt, systemImage]
  );

  const stop = useCallback(() => abortRef.current?.abort(), []);
  const clearChat = useCallback(() => setMessages([]), []);

  return (
    <div className="app">
      <Sidebar
        ollamaUrl={ollamaUrl}
        setOllamaUrl={setOllamaUrl}
        models={models}
        model={model}
        setModel={setModel}
        systemPrompt={systemPrompt}
        setSystemPrompt={setSystemPrompt}
        systemImage={systemImage}
        setSystemImage={setSystemImage}
        refreshModels={refreshModels}
      />
      <main className="main">
        <header className="topbar">
          <h1>👁️ Vision Model Chat</h1>
          <div className="topbar-actions">
            {model && <span className="model-pill">{model}</span>}
            {messages.length > 0 && (
              <button className="btn ghost" onClick={clearChat}>
                🗑️ Clear
              </button>
            )}
          </div>
        </header>
        {error && <div className="banner error">{error}</div>}
        <Chat
          messages={messages}
          streaming={streaming}
          onSend={send}
          onStop={stop}
          disabled={!model}
        />
      </main>
    </div>
  );
}
