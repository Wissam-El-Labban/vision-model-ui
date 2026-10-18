import { useEffect, useRef, useState } from "react";
import { DOC_ACCEPT, isDocFile, pullModel } from "../api";
import type { DocumentsApi } from "../useDocuments";
import type { DocInfo } from "../types";

interface Props {
  docs: DocumentsApi;
  /** Ids ticked for the current chat. */
  activeIds: string[];
  setActiveIds: (ids: string[]) => void;
  ollamaUrl: string;
  /** The vision model scanned pages are read with ("" = none installed). */
  ocrModel: string;
  /** Lifted so the topbar's docs pill can open the panel. */
  open: boolean;
  setOpen: (v: boolean) => void;
}

export function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(0)} KB`;
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  return `${(n / 1024 ** 3).toFixed(2)} GB`;
}

function docMeta(d: DocInfo): string {
  const parts = [];
  if (d.pages) parts.push(`${d.pages} page${d.pages === 1 ? "" : "s"}`);
  if (d.n_chunks) parts.push(`${d.n_chunks} chunk${d.n_chunks === 1 ? "" : "s"}`);
  parts.push(formatBytes(d.disk_bytes));
  return parts.join(" · ");
}

/** The document library: add files, pick which ones this chat answers from, and
 *  delete what's no longer wanted (which frees its disk space outright). */
export default function Documents({
  docs,
  activeIds,
  setActiveIds,
  ollamaUrl,
  ocrModel,
  open,
  setOpen,
}: Props) {
  const { library, uploads, reindexing } = docs;
  const fileRef = useRef<HTMLInputElement>(null);
  const rootRef = useRef<HTMLDivElement>(null);
  const [dragOver, setDragOver] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);
  const [modelDraft, setModelDraft] = useState("");
  const [pull, setPull] = useState<{ status: string; pct: number | null } | null>(null);

  useEffect(() => setModelDraft(library.embed_model), [library.embed_model]);
  // Opened from the topbar pill: bring the panel into view, since the sidebar
  // scrolls and it may be well below the fold.
  useEffect(() => {
    if (open) rootRef.current?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [open]);

  const ready = library.documents.filter((d) => d.status === "ready");
  const activeCount = activeIds.filter((id) => ready.some((d) => d.id === id)).length;

  function addFiles(list: FileList | File[]) {
    const files = Array.from(list).filter(isDocFile);
    if (files.length) docs.add(files);
  }

  function toggle(id: string) {
    setActiveIds(
      activeIds.includes(id) ? activeIds.filter((x) => x !== id) : [...activeIds, id]
    );
  }

  async function pullEmbedModel() {
    setPull({ status: "starting…", pct: null });
    try {
      await pullModel(ollamaUrl, library.embed_model, (status, pct) =>
        setPull({ status, pct })
      );
      setPull(null);
      docs.refresh();
    } catch {
      setPull({ status: "✗ download failed", pct: null });
    }
  }

  return (
    <div className="section" ref={rootRef}>
      <button className="section-head" onClick={() => setOpen(!open)}>
        <span className="section-head-left">
          📚 Documents
          <span className="section-sub">
            {library.documents.length
              ? `${library.documents.length} in library · ${formatBytes(library.disk_bytes)}`
              : "Drop PDFs or text files to answer from"}
          </span>
        </span>
        <span className="section-head-right">
          <span className={`section-status ${activeCount ? "on" : ""}`}>
            {activeCount ? `${activeCount} in chat` : "None"}
          </span>
          <span className="chev">{open ? "▾" : "▸"}</span>
        </span>
      </button>
      {open && (
        <div className="section-body">
          {!library.embed_installed && (
            <div className="doc-warn small">
              The embedding model <b>{library.embed_model}</b> isn't installed — documents
              can't be indexed or searched until it is.
              <button className="btn" onClick={pullEmbedModel} disabled={!!pull}>
                ⬇ Pull {library.embed_model}
              </button>
              {pull && (
                <div className="pull-progress">
                  <div className="muted small">{pull.status}</div>
                  {pull.pct !== null && (
                    <div className="progress">
                      <div
                        className="progress-bar"
                        style={{ width: `${Math.round(pull.pct * 100)}%` }}
                      />
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          <div
            className={`doc-drop ${dragOver ? "drag" : ""}`}
            role="button"
            tabIndex={0}
            onClick={() => fileRef.current?.click()}
            onKeyDown={(e) => e.key === "Enter" && fileRef.current?.click()}
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
            📄 Drop PDFs or text files here, or click to browse
          </div>
          <input
            ref={fileRef}
            type="file"
            accept={DOC_ACCEPT}
            multiple
            hidden
            onChange={(e) => {
              if (e.target.files) addFiles(e.target.files);
              e.target.value = "";
            }}
          />

          {uploads.map((u) => (
            <div key={u.key} className={`doc-row ${u.error ? "error" : ""}`}>
              <div className="doc-main">
                <span className="doc-name" title={u.name}>
                  {u.name}
                </span>
                {u.error ? (
                  <span className="doc-meta doc-error">{u.error}</span>
                ) : (
                  <>
                    <span className="doc-meta">{u.progress.message}</span>
                    <div className="progress">
                      <div
                        className={`progress-bar ${u.progress.frac === null ? "indeterminate" : ""}`}
                        style={{ width: `${Math.round((u.progress.frac ?? 0.3) * 100)}%` }}
                      />
                    </div>
                  </>
                )}
              </div>
              {u.error && (
                <button
                  className="btn ghost doc-btn"
                  title="Dismiss"
                  onClick={() => docs.dismissUpload(u.key)}
                >
                  ✕
                </button>
              )}
            </div>
          ))}

          {library.documents.length === 0 && uploads.length === 0 && (
            <p className="hint muted small">
              Nothing here yet. Tick a document to have this chat answer from it — in
              Analyze, Agent mode, and the prompt enhancer.
            </p>
          )}

          {library.documents.map((d) => {
            const progress = reindexing[d.id];
            const stale =
              d.status === "ready" && !!d.embed_model && d.embed_model !== library.embed_model;
            const usable = d.status === "ready" && !progress;
            return (
              <div key={d.id} className={`doc-row ${d.status === "error" ? "error" : ""}`}>
                <input
                  type="checkbox"
                  checked={activeIds.includes(d.id)}
                  disabled={!usable}
                  onChange={() => toggle(d.id)}
                  title={usable ? "Use in this chat" : "Not indexed"}
                />
                <div className="doc-main">
                  <span className="doc-name" title={d.name}>
                    {d.kind === "pdf" ? "📕" : "📄"} {d.name}
                  </span>
                  {progress ? (
                    <>
                      <span className="doc-meta">{progress.message}</span>
                      <div className="progress">
                        <div
                          className={`progress-bar ${progress.frac === null ? "indeterminate" : ""}`}
                          style={{ width: `${Math.round((progress.frac ?? 0.3) * 100)}%` }}
                        />
                      </div>
                    </>
                  ) : d.status === "error" ? (
                    <span className="doc-meta doc-error">{d.error || "Indexing failed."}</span>
                  ) : d.status === "indexing" ? (
                    <span className="doc-meta">Indexing…</span>
                  ) : (
                    <span className="doc-meta">
                      {docMeta(d)}
                      {stale && (
                        <span
                          className="doc-badge"
                          title={`Indexed with ${d.embed_model}; new documents use ${library.embed_model}. It still works — re-index to switch it over.`}
                        >
                          {d.embed_model}
                        </span>
                      )}
                    </span>
                  )}
                </div>
                {(d.status === "error" || stale) && !progress && (
                  <button
                    className="btn ghost doc-btn"
                    title={`Re-index${ocrModel ? ` (scanned pages read with ${ocrModel})` : ""}`}
                    onClick={() => docs.reindex(d.id)}
                  >
                    ↻
                  </button>
                )}
                {confirmDelete === d.id ? (
                  <span className="doc-confirm">
                    <button
                      className="btn danger doc-btn"
                      onClick={() => {
                        setConfirmDelete(null);
                        docs.remove(d.id);
                      }}
                    >
                      Delete
                    </button>
                    <button className="btn ghost doc-btn" onClick={() => setConfirmDelete(null)}>
                      Keep
                    </button>
                  </span>
                ) : (
                  <button
                    className="btn ghost doc-btn"
                    title={`Delete — removes it from every chat and frees ${formatBytes(d.disk_bytes)}`}
                    disabled={!!progress}
                    onClick={() => setConfirmDelete(d.id)}
                  >
                    🗑
                  </button>
                )}
              </div>
            );
          })}

          {ready.length > 1 && (
            <div className="row doc-select">
              <button className="btn ghost" onClick={() => setActiveIds(ready.map((d) => d.id))}>
                Use all
              </button>
              <button className="btn ghost" onClick={() => setActiveIds([])}>
                Use none
              </button>
            </div>
          )}

          <label className="lbl">Embedding model</label>
          <div className="row">
            <input
              value={modelDraft}
              onChange={(e) => setModelDraft(e.target.value)}
              placeholder="nomic-embed-text"
            />
            <button
              className="btn"
              disabled={!modelDraft.trim() || modelDraft.trim() === library.embed_model}
              onClick={() => docs.changeEmbedModel(modelDraft.trim())}
            >
              Save
            </button>
          </div>
          <p className="hint muted small">
            Runs in Ollama, locally. Changing it applies to documents added or
            re-indexed from then on.{" "}
            {ocrModel
              ? `Scanned PDF pages are read with ${ocrModel}.`
              : "Install a vision model to read scanned PDFs."}
          </p>
        </div>
      )}
    </div>
  );
}
