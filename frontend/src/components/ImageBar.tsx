import { useEffect, useRef, useState } from "react";

interface Props {
  images: string[];
  onAdd: (files: FileList | File[]) => void;
  onRemove: (i: number) => void;
  onRotate: (i: number) => void;
}

const MIN_SIZE = 100;
const MAX_SIZE = 600;
const DEFAULT_SIZE = 200;

/** A single filled box (card, no borders) anchored top-left. It grows to fit
 *  the images added; drag the bottom-right corner grip to resize it and the
 *  images scale with it. Double-click the grip to reset, click an image to
 *  view full size. */
export default function ImageBar({ images, onAdd, onRemove, onRotate }: Props) {
  const [dragOver, setDragOver] = useState(false);
  const [zoom, setZoom] = useState<string | null>(null);
  const [size, setSize] = useState(
    () => Number(localStorage.getItem("imgSize")) || DEFAULT_SIZE
  );
  const fileRef = useRef<HTMLInputElement>(null);
  const drag = useRef<{ y: number; start: number } | null>(null);

  useEffect(() => localStorage.setItem("imgSize", String(size)), [size]);

  useEffect(() => {
    function move(e: MouseEvent) {
      if (!drag.current) return;
      const next = drag.current.start + (e.clientY - drag.current.y);
      setSize(Math.min(MAX_SIZE, Math.max(MIN_SIZE, next)));
    }
    function up() {
      drag.current = null;
      document.body.style.cursor = "";
    }
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
    return () => {
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
    };
  }, []);

  const browse = () => fileRef.current?.click();

  return (
    <div className="image-bar">
      <div
        className={`image-box ${dragOver ? "drag" : ""}`}
        style={{ ["--img-h" as string]: `${size}px` } as React.CSSProperties}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          onAdd(e.dataTransfer.files);
        }}
      >
        <div className="image-box-head">
          <span>🖼️ Images</span>
          {images.length > 0 && (
            <button className="btn ghost small" onClick={browse}>
              + Add
            </button>
          )}
        </div>

        {images.length === 0 ? (
          <div className="box-empty" onClick={browse}>
            <span className="dz-emoji">🖼️⬇️</span>
            <span>
              Drag &amp; drop image(s) here <span className="muted">or click</span>
            </span>
          </div>
        ) : (
          <>
            <div className="bar-strip">
              {images.map((src, i) => (
                <div className="bar-img" key={i}>
                  <img
                    src={src}
                    alt={`image ${i + 1}`}
                    onClick={() => setZoom(src)}
                  />
                  <div className="bar-img-actions">
                    <button title="Rotate" onClick={() => onRotate(i)}>
                      ↻
                    </button>
                    <button title="Remove" onClick={() => onRemove(i)}>
                      ✕
                    </button>
                  </div>
                </div>
              ))}
            </div>
            <div
              className="corner-grip"
              title="Drag to resize · double-click to reset"
              onMouseDown={(e) => {
                e.preventDefault();
                drag.current = { y: e.clientY, start: size };
                document.body.style.cursor = "nwse-resize";
              }}
              onDoubleClick={() => setSize(DEFAULT_SIZE)}
            />
          </>
        )}
      </div>

      <input
        ref={fileRef}
        type="file"
        accept="image/*"
        multiple
        hidden
        onChange={(e) => {
          if (e.target.files) onAdd(e.target.files);
          e.target.value = "";
        }}
      />

      {zoom && (
        <div className="lightbox" onClick={() => setZoom(null)}>
          <img src={zoom} alt="full size" />
        </div>
      )}
    </div>
  );
}
