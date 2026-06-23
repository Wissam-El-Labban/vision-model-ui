import { useEffect, useRef, useState } from "react";

interface Props {
  images: string[];
  onAdd: (files: FileList | File[]) => void;
  onRemove: (i: number) => void;
  onRotate: (i: number) => void;
}

const MIN_SIZE = 100;
const DEFAULT_SIZE = 200;
const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

/** A single filled box (card, no borders) anchored top-left. It grows to fit
 *  the images added; resize it from the right edge (horizontal), bottom edge
 *  (vertical), or the corner (both). Images scale with it. Double-click a handle
 *  to reset, click an image to view full size. The card stays in flow so the
 *  chat below always resizes to fit and is never covered. */
export default function ImageBar({ images, onAdd, onRemove, onRotate }: Props) {
  const [dragOver, setDragOver] = useState(false);
  const [zoom, setZoom] = useState<string | null>(null);
  const [size, setSize] = useState(
    () => Number(localStorage.getItem("imgSize")) || DEFAULT_SIZE
  );
  const fileRef = useRef<HTMLInputElement>(null);
  const drag = useRef<{ x: number; y: number; start: number; dx: boolean; dy: boolean } | null>(
    null
  );

  useEffect(() => localStorage.setItem("imgSize", String(size)), [size]);

  useEffect(() => {
    function move(e: MouseEvent) {
      const d = drag.current;
      if (!d) return;
      let delta = 0;
      if (d.dx && d.dy) delta = Math.max(e.clientX - d.x, e.clientY - d.y);
      else if (d.dx) delta = e.clientX - d.x;
      else delta = e.clientY - d.y;
      // Cap so the card can't grow large enough to crowd out the chat.
      const max = Math.max(MIN_SIZE, window.innerHeight * 0.6);
      setSize(clamp(d.start + delta, MIN_SIZE, max));
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

  function startResize(e: React.MouseEvent, dx: boolean, dy: boolean) {
    e.preventDefault();
    e.stopPropagation();
    drag.current = { x: e.clientX, y: e.clientY, start: size, dx, dy };
    document.body.style.cursor = dx && dy ? "nwse-resize" : dx ? "ew-resize" : "ns-resize";
  }

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
              className="rh rh-right"
              title="Drag to resize width"
              onMouseDown={(e) => startResize(e, true, false)}
              onDoubleClick={() => setSize(DEFAULT_SIZE)}
            />
            <div
              className="rh rh-bottom"
              title="Drag to resize height"
              onMouseDown={(e) => startResize(e, false, true)}
              onDoubleClick={() => setSize(DEFAULT_SIZE)}
            />
            <div
              className="rh rh-corner"
              title="Drag to resize · double-click to reset"
              onMouseDown={(e) => startResize(e, true, true)}
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
