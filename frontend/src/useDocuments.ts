import { useCallback, useEffect, useState } from "react";
import {
  attachTask,
  deleteDoc,
  docIndexEvents,
  listDocs,
  listTasks,
  reindexDoc,
  setEmbedModel,
  uploadDoc,
  type DocLibrary,
  type DocProgress,
} from "./api";
import type { DocInfo } from "./types";

/** A file being added to the library, or one that failed to be. Keyed by a local
 *  id because it has no document id until the backend has indexed it. */
export interface DocUpload {
  key: string;
  name: string;
  progress: DocProgress;
  error?: string;
}

const EMPTY: DocLibrary = {
  documents: [],
  embed_model: "",
  embed_installed: true,
  disk_bytes: 0,
};

/** The document library: what's in it, what's being added, and the actions on it.
 *
 * Which documents a *chat* uses isn't here — that's per-chat state App owns and
 * saves with the chat. `onIndexed` is how a finished upload gets ticked for the
 * chat it was dropped into; `onDeleted` is how a deleted one leaves the selection.
 *
 * `ocrModel` is the vision model that reads scanned PDF pages: the chat model,
 * since it's the one most likely to be loaded already.
 */
export function useDocuments(
  ollamaUrl: string,
  ocrModel: string,
  onIndexed: (doc: DocInfo) => void,
  onDeleted: (id: string) => void
) {
  const [library, setLibrary] = useState<DocLibrary>(EMPTY);
  const [uploads, setUploads] = useState<DocUpload[]>([]);
  // Re-indexes in flight, by document id, with their progress.
  const [reindexing, setReindexing] = useState<Record<string, DocProgress>>({});

  const refresh = useCallback(async () => {
    try {
      setLibrary(await listDocs(ollamaUrl));
    } catch {
      /* backend predates documents, or is down — the panel shows an empty library */
    }
  }, [ollamaUrl]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  // Indexing runs on the backend and outlives a reload. Show each one still going
  // on its row again, and refresh the library as each finishes.
  useEffect(() => {
    let cancelled = false;
    listTasks()
      .then((tasks) => {
        for (const t of tasks) {
          const id = t.meta.doc_id;
          if (cancelled || t.kind !== "doc" || !id) continue;
          setReindexing((r) => ({ ...r, [id]: { message: "Indexing…", frac: null } }));
          attachTask(
            t.id,
            docIndexEvents(
              (p) => setReindexing((r) => ({ ...r, [id]: p })),
              () => {}
            )
          )
            .catch(() => {
              /* the row now carries the error; refresh shows it */
            })
            .finally(() => {
              setReindexing(({ [id]: _, ...rest }) => rest);
              refresh();
            });
        }
      })
      .catch(() => {
        /* nothing to resume */
      });
    return () => {
      cancelled = true;
    };
  }, [refresh]);

  const patchUpload = (key: string, patch: Partial<DocUpload>) =>
    setUploads((prev) => prev.map((u) => (u.key === key ? { ...u, ...patch } : u)));

  /** Send every file at once. The backend still indexes them one at a time (each
   *  embeds on the same Ollama), but holding the queue there rather than here is
   *  what lets the files waiting their turn survive a reload. */
  const add = useCallback(
    async (files: File[]) => {
      const queued = files.map((f) => ({
        file: f,
        key: crypto.randomUUID(),
      }));
      setUploads((prev) => [
        ...prev,
        ...queued.map(({ file, key }) => ({
          key,
          name: file.name,
          progress: { message: "Waiting…", frac: null },
        })),
      ]);
      await Promise.all(
        queued.map(async ({ file, key }) => {
          try {
            const { doc } = await uploadDoc(file, ollamaUrl, ocrModel, (p) =>
              patchUpload(key, { progress: p })
            );
            setUploads((prev) => prev.filter((u) => u.key !== key));
            onIndexed(doc);
          } catch (e) {
            patchUpload(key, { error: (e as Error).message });
          }
          await refresh();
        })
      );
    },
    [ollamaUrl, ocrModel, onIndexed, refresh]
  );

  const dismissUpload = useCallback(
    (key: string) => setUploads((prev) => prev.filter((u) => u.key !== key)),
    []
  );

  const reindex = useCallback(
    async (id: string) => {
      setReindexing((r) => ({ ...r, [id]: { message: "Starting…", frac: null } }));
      try {
        await reindexDoc(id, ollamaUrl, ocrModel, (p) =>
          setReindexing((r) => ({ ...r, [id]: p }))
        );
      } catch {
        /* the row now carries the error; refresh shows it */
      } finally {
        setReindexing(({ [id]: _, ...rest }) => rest);
        await refresh();
      }
    },
    [ollamaUrl, ocrModel, refresh]
  );

  const remove = useCallback(
    async (id: string) => {
      try {
        await deleteDoc(id);
        onDeleted(id);
      } finally {
        await refresh();
      }
    },
    [onDeleted, refresh]
  );

  const changeEmbedModel = useCallback(
    async (name: string) => {
      await setEmbedModel(name, ollamaUrl);
      await refresh();
    },
    [ollamaUrl, refresh]
  );

  return {
    library,
    uploads,
    reindexing,
    refresh,
    add,
    dismissUpload,
    reindex,
    remove,
    changeEmbedModel,
  };
}

export type DocumentsApi = ReturnType<typeof useDocuments>;
