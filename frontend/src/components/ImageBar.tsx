import { useEffect, useRef, useState } from "react";

interface Props {
  images: string[];
  onAdd: (files: FileList | File[]) => void;
  onRemove: (i: number) => void;
  onRotate: (i: number) => void;
}

const MIN_H = 120;

/** Horizontal image bar across the top of the chat. Images fill the bar's
 *  height; drag the bottom edge up/down to resize, and the images scale with
 *  it. Click any image to view it full size. */
export default function ImageBar({ images, onAdd, onRemove, onRotate }: Props) {
  const [dragOver, setDragOver] = useState(false);
  const [zoom, setZoom] = useState<string | null>(null);
  const [height, setHeight] = useState(
    () => Number(localStorage.getItem("imgBarHeight")) || 200
  );
  const fileRef = useRef<HTMLInputElement>(null);
  const barRef = useRef<HTMLDivElement>(null);
  const resizing = useRef(false);

  useEffect(() => localStorage.setItem("imgBarHeight", String(height)), [height]);

  useEffect(() => {
    function move(e: MouseEvent) {
      if (!resizing.current || !barRef.current) return;
      const top = barRef.current.getBoundingClientRect().top;
      const max = window.innerHeight * 0.7;
      setHeight(Math.min(max, Math.max(MIN_H, e.clientY - top)));
    }
    function up() {
      resizing.current = false;
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
  const dropProps = {
    onDragOver: (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(true);
    },
    onDragLeave: () => setDragOver(false),
    onDrop: (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      onAdd(e.dataTransfer.files);
    },
  };

  return (
    <div className="image-bar" ref={barRef} style={{ height }}>
      <div className="image-bar-head">
        <span>🖼️ Images</span>
        {images.length > 0 && (
          <button className="btn ghost small" onClick={browse}>
            + Add
          </button>
        )}
      </div>

      <div className="image-bar-body" {...dropProps}>
        {images.length === 0 ? (
          <div
            className={`bar-dropzone ${dragOver ? "drag" : ""}`}
            onClick={browse}
          >
            <span className="dz-emoji">🖼️⬇️</span>
            <span>
              Drag &amp; drop image(s) here <span className="muted">or click</span>
            </span>
          </div>
        ) : (
          <div className={`bar-strip ${dragOver ? "drag" : ""}`}>
            {images.map((src, i) => (
              <div className="bar-img" key={i}>
                <img src={src} alt={`image ${i + 1}`} onClick={() => setZoom(src)} />
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

      <div
        className="bar-resize-handle"
        title="Drag to resize"
        onMouseDown={(e) => {
          e.preventDefault();
          resizing.current = true;
          document.body.style.cursor = "row-resize";
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
