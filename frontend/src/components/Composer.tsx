import { useRef, useState } from "react";
import { fileToResizedDataUrl, rotateDataUrl } from "../fileUtils";

interface Props {
  onSend: (text: string, images: string[]) => void;
  onStop: () => void;
  streaming: boolean;
  disabled: boolean;
}

export default function Composer({ onSend, onStop, streaming, disabled }: Props) {
  const [text, setText] = useState("");
  const [images, setImages] = useState<string[]>([]);
  const [dragOver, setDragOver] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  async function addFiles(files: FileList | File[]) {
    const list = Array.from(files).filter((f) => f.type.startsWith("image/"));
    const urls = await Promise.all(list.map((f) => fileToResizedDataUrl(f)));
    setImages((prev) => [...prev, ...urls]);
  }

  function removeImage(i: number) {
    setImages((prev) => prev.filter((_, idx) => idx !== i));
  }

  async function rotateImage(i: number) {
    setImages((prev) => prev.slice());
    const rotated = await rotateDataUrl(images[i], 90);
    setImages((prev) => prev.map((img, idx) => (idx === i ? rotated : img)));
  }

  function submit() {
    const trimmed = text.trim();
    if (!trimmed && images.length === 0) return;
    onSend(trimmed, images);
    setText("");
    setImages([]);
  }

  function onKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!streaming) submit();
    }
  }

  return (
    <div
      className={`composer ${dragOver ? "drag" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragOver(false);
        addFiles(e.dataTransfer.files);
      }}
    >
      {images.length > 0 && (
        <div className="thumbs">
          {images.map((src, i) => (
            <div className="thumb" key={i}>
              <img src={src} alt={`attachment ${i + 1}`} />
              <div className="thumb-actions">
                <button title="Rotate" onClick={() => rotateImage(i)}>
                  ↻
                </button>
                <button title="Remove" onClick={() => removeImage(i)}>
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
            if (e.target.files) addFiles(e.target.files);
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
              : "Ask about your image(s)… (drop or 📎 to attach, Enter to send)"
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
            onClick={submit}
            disabled={disabled || (!text.trim() && images.length === 0)}
          >
            ➤
          </button>
        )}
      </div>
    </div>
  );
}
