import { useEffect, useRef, useState } from "react";

interface Props {
  images: string[];
  onAdd: (files: FileList | File[]) => void;
  onRemove: (i: number) => void;
  onRotate: (i: number) => void;
  /** Notifies the parent when the panel is "focused" (clicked). While focused, a
   *  Ctrl+V paste routes to the pinned images instead of the message composer. */
  onLockChange: (locked: boolean) => void;
}

const MIN_W = 200;
const DEF_W = 420;
const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

/** Pinned image panel down the left side. It stays put while the chat scrolls
 *  on the right, never covering it. Drag the right edge to resize its width; the
 *  images always scale to fit inside (never clipped). Click an image to zoom. */
export default function ImageBar({ images, onAdd, onRemove, onRotate, onLockChange }: Props) {
  const [dragOver, setDragOver] = useState(false);
  const [zoom, setZoom] = useState<string | null>(null);
  const [locked, setLocked] = useState(false);
  const [w, setW] = useState(() => Number(localStorage.getItem("imgW")) || DEF_W);
  const fileRef = useRef<HTMLInputElement>(null);
  const rootRef = useRef<HTMLDivElement>(null);
  const drag = useRef<{ x: number; w: number; maxW: number } | null>(null);

  useEffect(() => localStorage.setItem("imgW", String(w)), [w]);

  // Report the focus/lock state up so App can route pastes to pinned.
  useEffect(() => onLockChange(locked), [locked, onLockChange]);

  // Clicking anywhere outside the panel unfocuses it (removes the glow).
  useEffect(() => {
    if (!locked) return;
    const onDown = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) {
        setLocked(false);
      }
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [locked]);

  useEffect(() => {
    function move(e: MouseEvent) {
      const d = drag.current;
      if (!d) return;
      setW(clamp(d.w + (e.clientX - d.x), MIN_W, d.maxW));
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

  function startResize(e: React.MouseEvent) {
    e.preventDefault();
    e.stopPropagation();
    const pw = rootRef.current?.parentElement?.clientWidth ?? window.innerWidth;
    drag.current = { x: e.clientX, w, maxW: Math.max(MIN_W, Math.min(pw * 0.55, pw - 360)) };
    document.body.style.cursor = "col-resize";
  }

  const browse = () => fileRef.current?.click();
  const has = images.length > 0;

  return (
    <div
      className={`image-bar ${has ? "filled" : ""} ${locked ? "locked" : ""}`}
      ref={rootRef}
      style={has ? { width: w } : undefined}
      onMouseDown={() => setLocked(true)}
      title={locked ? "Focused — pasted images will pin here" : undefined}
    >
      <div
        className={`image-box ${dragOver ? "drag" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          e.stopPropagation();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          e.stopPropagation();
          setDragOver(false);
          onAdd(e.dataTransfer.files);
        }}
      >
        <div className="image-box-head">
          <span>🖼️ Images</span>
          {has && (
            <button className="btn ghost small" onClick={browse}>
              + Add
            </button>
          )}
        </div>

        {!has ? (
          <div className="box-empty" onClick={browse}>
            <span className="dz-emoji">🖼️⬇️</span>
            <span>
              Drag &amp; drop image(s) here
              <br />
              <span className="muted">or click to browse · click here then paste to pin</span>
            </span>
          </div>
        ) : (
          <div className="bar-strip">
            {images.map((src, i) => (
              <div className="bar-img" key={i}>
                <div className="bar-img-inner">
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
              </div>
            ))}
          </div>
        )}
      </div>

      {has && (
        <div className="dock-resize rz-right" title="Drag to resize" onMouseDown={startResize} />
      )}

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
