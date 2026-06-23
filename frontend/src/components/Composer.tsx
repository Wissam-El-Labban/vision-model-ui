import { useRef } from "react";

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
}: Props) {
  const fileRef = useRef<HTMLInputElement>(null);

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
