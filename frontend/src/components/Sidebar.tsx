import { useState } from "react";
import ModelManager from "./ModelManager";
import UpdateBanner from "./UpdateBanner";
import { fileToResizedDataUrl } from "../fileUtils";

interface Props {
  ollamaUrl: string;
  setOllamaUrl: (v: string) => void;
  models: { vision: string[]; all: string[] };
  model: string;
  setModel: (v: string) => void;
  systemPrompt: string;
  setSystemPrompt: (v: string) => void;
  systemImage: string | null;
  setSystemImage: (v: string | null) => void;
  refreshModels: () => void;
}

export default function Sidebar(props: Props) {
  const {
    ollamaUrl,
    setOllamaUrl,
    models,
    model,
    setModel,
    systemPrompt,
    setSystemPrompt,
    systemImage,
    setSystemImage,
    refreshModels,
  } = props;
  const [urlDraft, setUrlDraft] = useState(ollamaUrl);
  const [sysOpen, setSysOpen] = useState(false);

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

      <label className="lbl">🤖 Vision model</label>
      {models.vision.length > 0 ? (
        <select
          className="block"
          value={model}
          onChange={(e) => setModel(e.target.value)}
        >
          {models.vision.map((m) => (
            <option key={m} value={m}>
              {m}
            </option>
          ))}
        </select>
      ) : (
        <input
          className="block"
          value={model}
          onChange={(e) => setModel(e.target.value)}
          placeholder="No vision models found — type one"
        />
      )}

      <div className="section">
        <button className="section-head" onClick={() => setSysOpen(!sysOpen)}>
          💬 System Prompt <span className="chev">{sysOpen ? "▾" : "▸"}</span>
        </button>
        {sysOpen && (
          <div className="section-body">
            <textarea
              className="block"
              rows={4}
              value={systemPrompt}
              onChange={(e) => setSystemPrompt(e.target.value)}
              placeholder="Guide the model's behavior across the chat…"
            />
            <label className="lbl">📎 Persistent context image (optional)</label>
            {systemImage ? (
              <div className="sys-image">
                <img src={systemImage} alt="system" />
                <button
                  className="btn danger block"
                  onClick={() => setSystemImage(null)}
                >
                  Remove image
                </button>
              </div>
            ) : (
              <input
                type="file"
                accept="image/*"
                onChange={async (e) => {
                  const f = e.target.files?.[0];
                  if (f) setSystemImage(await fileToResizedDataUrl(f));
                }}
              />
            )}
          </div>
        )}
      </div>

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
