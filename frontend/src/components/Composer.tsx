import { useRef, useState } from "react";
import { fileToResizedDataUrl } from "../fileUtils";

interface Props {
  text: string;
  setText: (v: string) => void;
  images: string[];
  onAddFiles: (files: FileList | File[]) => void;
  onRemoveImage: (i: number) => void;
  onRotateImage: (i: number) => void;
  onSubmit: () => void;
  onStop: () => void;
  streaming: boolean;
  disabled: boolean;
  // Moved here from the sidebar: system prompt (left) + model selector (right).
  models: { vision: string[]; all: string[] };
  model: string;
  setModel: (v: string) => void;
  systemPrompt: string;
  setSystemPrompt: (v: string) => void;
  systemImage: string | null;
  setSystemImage: (v: string | null) => void;
}

export default function Composer({
  text,
  setText,
  images,
  onAddFiles,
  onRemoveImage,
  onRotateImage,
  onSubmit,
  onStop,
  streaming,
  disabled,
  models,
  model,
  setModel,
  systemPrompt,
  setSystemPrompt,
  systemImage,
  setSystemImage,
}: Props) {
  const fileRef = useRef<HTMLInputElement>(null);
  const [sysOpen, setSysOpen] = useState(false);
  const hasSystem = systemPrompt.trim().length > 0 || !!systemImage;

  function onKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!streaming) onSubmit();
    }
  }

  return (
    <div className="composer">
      {images.length > 0 && (
        <div className="thumbs">
          {images.map((src, i) => (
            <div className="thumb" key={i}>
              <img src={src} alt={`attachment ${i + 1}`} />
              <div className="thumb-actions">
                <button title="Rotate" onClick={() => onRotateImage(i)}>
                  ↻
                </button>
                <button title="Remove" onClick={() => onRemoveImage(i)}>
                  ✕
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      <div className="composer-controls">
        <div className="sys-control">
          <button
            className={`btn ghost small ${hasSystem ? "has-dot" : ""}`}
            onClick={() => setSysOpen((v) => !v)}
            title="System prompt"
          >
            💬 System {hasSystem && <span className="dot" />}
            <span className="chev">{sysOpen ? "▾" : "▸"}</span>
          </button>
          {sysOpen && (
            <div className="system-popover">
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

        <div className="model-control">
          <span className="lbl inline">🤖 Model</span>
          {models.vision.length > 0 ? (
            <select value={model} onChange={(e) => setModel(e.target.value)}>
              {models.vision.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          ) : (
            <input
              value={model}
              onChange={(e) => setModel(e.target.value)}
              placeholder="No vision models — type one"
            />
          )}
        </div>
      </div>

      <div className="composer-row">
        <button
          className="btn icon"
          title="Attach images"
          onClick={() => fileRef.current?.click()}
          disabled={disabled}
        >
          📎
        </button>
        <input
          ref={fileRef}
          type="file"
          accept="image/*"
          multiple
          hidden
          onChange={(e) => {
            if (e.target.files) onAddFiles(e.target.files);
            e.target.value = "";
          }}
        />
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder={
            disabled
              ? "Select a vision model to start…"
              : "Ask about your image(s)… (drop images anywhere in the chat, Enter to send)"
          }
          rows={1}
          disabled={disabled}
        />
        {streaming ? (
          <button className="btn stop" onClick={onStop}>
            ■ Stop
          </button>
        ) : (
          <button
            className="btn send"
            onClick={onSubmit}
            disabled={disabled || (!text.trim() && images.length === 0)}
          >
            ➤
          </button>
        )}
      </div>
    </div>
  );
}
