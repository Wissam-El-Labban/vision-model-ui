import { useEffect, useRef, useState } from "react";

interface Props {
  images: string[];
  onAdd: (files: FileList | File[]) => void;
  onRemove: (i: number) => void;
  onRotate: (i: number) => void;
}

const MIN_W = 160;
const MIN_H = 130;
const DEF_W = 420;
const DEF_H = 300;
const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

/** A separate card stuck to the top-left corner. Resize it horizontally (right
 *  edge), vertically (bottom edge), or both (corner) — the images always scale
 *  to fit inside, so they're never clipped no matter how small you make it.
 *  Double-click a handle to reset; click an image to view full size. */
export default function ImageBar({ images, onAdd, onRemove, onRotate }: Props) {
  const [dragOver, setDragOver] = useState(false);
  const [zoom, setZoom] = useState<string | null>(null);
  const [w, setW] = useState(() => Number(localStorage.getItem("imgW")) || DEF_W);
  const [h, setH] = useState(() => Number(localStorage.getItem("imgH")) || DEF_H);
  const fileRef = useRef<HTMLInputElement>(null);
  const drag = useRef<{ x: number; y: number; w: number; h: number; dx: boolean; dy: boolean } | null>(
    null
  );

  useEffect(() => localStorage.setItem("imgW", String(w)), [w]);
  useEffect(() => localStorage.setItem("imgH", String(h)), [h]);

  useEffect(() => {
    function move(e: MouseEvent) {
      const d = drag.current;
      if (!d) return;
      if (d.dx) setW(clamp(d.w + (e.clientX - d.x), MIN_W, window.innerWidth * 0.6));
      if (d.dy) setH(clamp(d.h + (e.clientY - d.y), MIN_H, window.innerHeight * 0.6));
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
    drag.current = { x: e.clientX, y: e.clientY, w, h, dx, dy };
    document.body.style.cursor = dx && dy ? "nwse-resize" : dx ? "ew-resize" : "ns-resize";
  }
  function reset() {
    setW(DEF_W);
    setH(DEF_H);
  }

  const browse = () => fileRef.current?.click();

  return (
    <div className="image-bar">
      <div
        className={`image-box ${dragOver ? "drag" : ""}`}
        style={{ width: w, height: h }}
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
              Drag &amp; drop image(s) here
              <br />
              <span className="muted">or click to browse</span>
            </span>
          </div>
        ) : (
          <div className="bar-strip">
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

        {images.length > 0 && (
          <>
            <div
              className="rh rh-right"
              title="Drag to resize width"
              onMouseDown={(e) => startResize(e, true, false)}
              onDoubleClick={reset}
            />
            <div
              className="rh rh-bottom"
              title="Drag to resize height"
              onMouseDown={(e) => startResize(e, false, true)}
              onDoubleClick={reset}
            />
            <div
              className="rh rh-corner"
              title="Drag to resize · double-click to reset"
              onMouseDown={(e) => startResize(e, true, true)}
              onDoubleClick={reset}
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
