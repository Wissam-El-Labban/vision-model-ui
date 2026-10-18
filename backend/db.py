"""SQLite persistence for chats, messages, and images.

The app used to keep every conversation in browser RAM; this module gives it
durable, browsable history. Chat/message rows live in a SQLite file; image
*bytes* live as files on disk (deduped by sha256 content hash) with a `hash`
registry row. Thumbnails (generated client-side) are stored under the same hash.

Full images are kept in whatever format they arrived in — `images.mime` records
which, and the file's extension follows it. Thumbnails are always JPEG. Encoding
policy lives in `images.py`; this module only stores what that hands it.
"""
import base64
import hashlib
import json
import re
import sqlite3
import time
import uuid
from pathlib import Path

from . import images

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
DATA_DIR = Path(__file__).resolve().parent / "data"
IMAGES_DIR = DATA_DIR / "images"
THUMBS_DIR = DATA_DIR / "thumbs"
DB_PATH = DATA_DIR / "app.db"

for _d in (DATA_DIR, IMAGES_DIR, THUMBS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Connection / schema
# --------------------------------------------------------------------------- #
def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


_SCHEMA = """
CREATE TABLE IF NOT EXISTS chats (
    id                TEXT PRIMARY KEY,
    title             TEXT,
    model             TEXT,
    system_prompt     TEXT NOT NULL DEFAULT '',
    system_image_hash TEXT,
    created_at        INTEGER NOT NULL,
    updated_at        INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id         TEXT PRIMARY KEY,
    chat_id    TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    ordinal    INTEGER NOT NULL,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL DEFAULT '',
    model      TEXT,
    created_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS images (
    hash       TEXT PRIMARY KEY,
    mime       TEXT NOT NULL DEFAULT 'image/jpeg',
    created_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS message_images (
    message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    image_hash TEXT NOT NULL,
    ordinal    INTEGER NOT NULL
);
-- The ordered set of images that were in the model's context when a message
-- was generated (pinned + in-chat, in manifest order). Used to resolve the
-- model's "image N" references back to a thumbnail. Distinct from
-- message_images (images actually attached to/displayed on the message).
CREATE TABLE IF NOT EXISTS message_context_images (
    message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    image_hash TEXT NOT NULL,
    ordinal    INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS chat_pinned_images (
    chat_id    TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    image_hash TEXT NOT NULL,
    ordinal    INTEGER NOT NULL
);
-- Documents for retrieval. The bytes and the vector index live in data/rag/<id>/
-- (see rag.py); this row is what the library lists and what a search is scoped by.
CREATE TABLE IF NOT EXISTS documents (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    sha256      TEXT NOT NULL UNIQUE,
    kind        TEXT NOT NULL,             -- pdf | text
    size_bytes  INTEGER NOT NULL,
    pages       INTEGER,
    n_chunks    INTEGER NOT NULL DEFAULT 0,
    embed_model TEXT,
    dim         INTEGER,
    status      TEXT NOT NULL,             -- indexing | ready | error
    error       TEXT,
    created_at  INTEGER NOT NULL
);
-- Which documents each chat answers from. Cascades both ways: deleting a document
-- takes it out of every chat, deleting a chat drops its selection.
CREATE TABLE IF NOT EXISTS chat_documents (
    chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    doc_id  TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    PRIMARY KEY (chat_id, doc_id)
);
CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id, ordinal);
CREATE INDEX IF NOT EXISTS idx_pinned_chat ON chat_pinned_images(chat_id, ordinal);
CREATE INDEX IF NOT EXISTS idx_msgimg_msg ON message_images(message_id, ordinal);
CREATE INDEX IF NOT EXISTS idx_msgctx_msg ON message_context_images(message_id, ordinal);
"""


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(_SCHEMA)
        # Columns added after the first release. CREATE TABLE IF NOT EXISTS won't add
        # them to an existing database, so add whichever are missing.
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(messages)")}
        if "sources_json" not in cols:
            conn.execute("ALTER TABLE messages ADD COLUMN sources_json TEXT")
        # A server that died mid-index left the row saying so forever.
        conn.execute(
            "UPDATE documents SET status = 'error', "
            "error = 'Indexing was interrupted — re-index to finish it.' "
            "WHERE status = 'indexing'"
        )


init_db()


def _now() -> int:
    return int(time.time())


# --------------------------------------------------------------------------- #
# Images (deduped files on disk)
# --------------------------------------------------------------------------- #
def _strip_data_url(data_url: str) -> bytes:
    """Decode a `data:image/...;base64,XXXX` string (or raw base64) to bytes."""
    comma = data_url.find(",")
    b64 = data_url[comma + 1 :] if comma >= 0 else data_url
    return base64.b64decode(b64)


_HEX64 = re.compile(r"[0-9a-f]{64}")


def image_path(h: str) -> Path | None:
    """The stored file for a hash, whatever format it's in — or None if absent.

    Globs rather than reading `images.mime`, so it resolves a file the registry
    disagrees with (or predates) and needs no migration for images stored back when
    everything was `.jpg`. The hex guard matters: hashes arrive straight off request
    bodies, and this builds a path from one.
    """
    if not _HEX64.fullmatch(h or ""):
        return None
    return next(IMAGES_DIR.glob(f"{h}.*"), None)


def image_url(h: str) -> str:
    """The URL that serves a stored image, extension included."""
    p = image_path(h)
    return f"/api/images/{h}{p.suffix if p else '.png'}"


def _ext_map(conn: sqlite3.Connection) -> dict[str, str]:
    """hash -> file extension, for every registered image.

    One query, built once per request that needs to name several images. The
    alternative — globbing per hash — is a filesystem hit per image per chat load.
    A hash with no row here isn't guessed at; callers fall back to `image_path`,
    which reads the answer off the disk rather than inventing a second one.
    """
    return {
        r["hash"]: images.MIME_TO_EXT[r["mime"]]
        for r in conn.execute("SELECT hash, mime FROM images")
        if r["mime"] in images.MIME_TO_EXT
    }


def save_image(full_data_url: str, thumb_data_url: str | None = None) -> str:
    """Persist one image (full + optional thumbnail) deduped by content hash.

    Returns the sha256 hash. The full image keeps its own format (see
    `images.normalize_for_store`); the thumbnail is always JPEG, so the sidebar can
    always request `/api/thumbs/<hash>.jpg`.

    The hash is taken over the *stored* bytes, not the uploaded ones. Hashing the
    upload instead would leave the hash naming something that isn't on disk, and
    dedupe comparing images we no longer have.
    """
    raw = images.normalize_for_store(_strip_data_url(full_data_url))
    h = hashlib.sha256(raw).hexdigest()
    mime, ext = images.sniff(raw)

    full_path = IMAGES_DIR / f"{h}.{ext}"
    if not full_path.exists():
        full_path.write_bytes(raw)

    if thumb_data_url:
        thumb_path = THUMBS_DIR / f"{h}.jpg"
        if not thumb_path.exists():
            thumb_path.write_bytes(_strip_data_url(thumb_data_url))

    with _connect() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO images(hash, mime, created_at) VALUES (?, ?, ?)",
            (h, mime, _now()),
        )
    return h


def _gc_orphan_images(conn: sqlite3.Connection) -> None:
    """Delete image files/rows no longer referenced by any chat or message."""
    referenced = set()
    for tbl, col in (
        ("chat_pinned_images", "image_hash"),
        ("message_images", "image_hash"),
        ("message_context_images", "image_hash"),
    ):
        for row in conn.execute(f"SELECT DISTINCT {col} AS h FROM {tbl}"):
            referenced.add(row["h"])
    for row in conn.execute(
        "SELECT system_image_hash AS h FROM chats WHERE system_image_hash IS NOT NULL"
    ):
        referenced.add(row["h"])

    for row in conn.execute("SELECT hash FROM images"):
        h = row["hash"]
        if h in referenced:
            continue
        p = image_path(h)
        if p:
            p.unlink(missing_ok=True)
        (THUMBS_DIR / f"{h}.jpg").unlink(missing_ok=True)
        conn.execute("DELETE FROM images WHERE hash = ?", (h,))


# --------------------------------------------------------------------------- #
# Chats
# --------------------------------------------------------------------------- #
def _pinned_hashes(conn: sqlite3.Connection, chat_id: str) -> list[str]:
    rows = conn.execute(
        "SELECT image_hash FROM chat_pinned_images WHERE chat_id = ? ORDER BY ordinal",
        (chat_id,),
    )
    return [r["image_hash"] for r in rows]


def list_chats() -> list[dict]:
    """Sidebar list: newest first, with up to 3 pinned-image thumbnails."""
    with _connect() as conn:
        chats = conn.execute(
            "SELECT id, title, model, updated_at FROM chats ORDER BY updated_at DESC"
        ).fetchall()
        out = []
        for c in chats:
            icons = [
                f"/api/thumbs/{h}.jpg" for h in _pinned_hashes(conn, c["id"])[:3]
            ]
            out.append(
                {
                    "id": c["id"],
                    "title": c["title"],
                    "model": c["model"],
                    "updated_at": c["updated_at"],
                    "icons": icons,
                }
            )
        return out


def get_chat(chat_id: str) -> dict | None:
    """Full chat detail with image URLs (not bytes)."""
    with _connect() as conn:
        c = conn.execute("SELECT * FROM chats WHERE id = ?", (chat_id,)).fetchone()
        if not c:
            return None

        # Images keep their own format, so every URL here needs its extension.
        # Read them once rather than globbing the store per image; anything the
        # registry doesn't know falls back to the disk rather than to a guess.
        exts = _ext_map(conn)
        url = lambda h: f"/api/images/{h}.{exts[h]}" if h in exts else image_url(h)  # noqa: E731

        messages = []
        for m in conn.execute(
            "SELECT * FROM messages WHERE chat_id = ? ORDER BY ordinal", (chat_id,)
        ):
            imgs = conn.execute(
                "SELECT image_hash FROM message_images WHERE message_id = ? ORDER BY ordinal",
                (m["id"],),
            )
            ctx = conn.execute(
                "SELECT image_hash FROM message_context_images WHERE message_id = ? ORDER BY ordinal",
                (m["id"],),
            )
            messages.append(
                {
                    "role": m["role"],
                    "content": m["content"],
                    "model": m["model"],
                    "images": [url(r["image_hash"]) for r in imgs],
                    "context_images": [url(r["image_hash"]) for r in ctx],
                    "sources": json.loads(m["sources_json"]) if m["sources_json"] else [],
                }
            )

        pinned = [url(h) for h in _pinned_hashes(conn, chat_id)]
        sys_img = url(c["system_image_hash"]) if c["system_image_hash"] else None
        return {
            "id": c["id"],
            "title": c["title"],
            "model": c["model"],
            "system_prompt": c["system_prompt"],
            "system_image": sys_img,
            "pinned": pinned,
            "documents": _chat_doc_ids(conn, chat_id),
            "messages": messages,
        }


def upsert_chat(
    chat_id: str,
    model: str | None,
    system_prompt: str,
    pinned_hashes: list[str],
    system_image_hash: str | None,
    doc_ids: list[str] | None = None,
) -> None:
    """Create the chat row if new, else update its metadata + pinned images.

    Title is never overwritten here (it's set once by the LLM titler). `doc_ids`
    replaces the chat's document selection; None leaves it as it is, so a caller
    that doesn't know about documents can't clear them.
    """
    now = _now()
    with _connect() as conn:
        exists = conn.execute(
            "SELECT 1 FROM chats WHERE id = ?", (chat_id,)
        ).fetchone()
        if exists:
            conn.execute(
                "UPDATE chats SET model = ?, system_prompt = ?, "
                "system_image_hash = ?, updated_at = ? WHERE id = ?",
                (model, system_prompt, system_image_hash, now, chat_id),
            )
        else:
            conn.execute(
                "INSERT INTO chats(id, title, model, system_prompt, "
                "system_image_hash, created_at, updated_at) "
                "VALUES (?, NULL, ?, ?, ?, ?, ?)",
                (chat_id, model, system_prompt, system_image_hash, now, now),
            )

        conn.execute(
            "DELETE FROM chat_pinned_images WHERE chat_id = ?", (chat_id,)
        )
        conn.executemany(
            "INSERT INTO chat_pinned_images(chat_id, image_hash, ordinal) VALUES (?, ?, ?)",
            [(chat_id, h, i) for i, h in enumerate(pinned_hashes)],
        )
        if doc_ids is not None:
            conn.execute("DELETE FROM chat_documents WHERE chat_id = ?", (chat_id,))
            # INSERT ... SELECT so an id that's since been deleted is skipped rather
            # than failing the foreign key and with it the whole save.
            conn.executemany(
                "INSERT OR IGNORE INTO chat_documents(chat_id, doc_id) "
                "SELECT ?, id FROM documents WHERE id = ?",
                [(chat_id, d) for d in doc_ids],
            )


def append_message(
    chat_id: str,
    role: str,
    content: str,
    model: str | None,
    image_hashes: list[str],
    context_hashes: list[str] | None = None,
    sources: list[dict] | None = None,
) -> str:
    """Append a message to a chat. Returns the new message id.

    `context_hashes` records the ordered images that were in the model's context
    when this message was produced (used to resolve "image N" references to a
    thumbnail); it is separate from `image_hashes` (images shown on the message).
    """
    msg_id = str(uuid.uuid4())
    now = _now()
    with _connect() as conn:
        row = conn.execute(
            "SELECT COALESCE(MAX(ordinal), -1) + 1 AS nxt FROM messages WHERE chat_id = ?",
            (chat_id,),
        ).fetchone()
        ordinal = row["nxt"]
        conn.execute(
            "INSERT INTO messages(id, chat_id, ordinal, role, content, model, "
            "created_at, sources_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (msg_id, chat_id, ordinal, role, content, model, now,
             json.dumps(sources) if sources else None),
        )
        conn.executemany(
            "INSERT INTO message_images(message_id, image_hash, ordinal) VALUES (?, ?, ?)",
            [(msg_id, h, i) for i, h in enumerate(image_hashes)],
        )
        conn.executemany(
            "INSERT INTO message_context_images(message_id, image_hash, ordinal) VALUES (?, ?, ?)",
            [(msg_id, h, i) for i, h in enumerate(context_hashes or [])],
        )
        conn.execute(
            "UPDATE chats SET updated_at = ? WHERE id = ?", (now, chat_id)
        )
    return msg_id


def set_title(chat_id: str, title: str) -> None:
    with _connect() as conn:
        conn.execute("UPDATE chats SET title = ? WHERE id = ?", (title, chat_id))


def get_first_exchange(chat_id: str) -> tuple[str, str] | None:
    """Return (first_user_content, first_assistant_content) for titling."""
    with _connect() as conn:
        user = conn.execute(
            "SELECT content FROM messages WHERE chat_id = ? AND role = 'user' "
            "ORDER BY ordinal LIMIT 1",
            (chat_id,),
        ).fetchone()
        assistant = conn.execute(
            "SELECT content FROM messages WHERE chat_id = ? AND role = 'assistant' "
            "ORDER BY ordinal LIMIT 1",
            (chat_id,),
        ).fetchone()
    if not user:
        return None
    return (user["content"], assistant["content"] if assistant else "")


def delete_chat(chat_id: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
        # NOTE: we intentionally do NOT garbage-collect image files here. Content-
        # addressed bytes can still be live in an open browser session (e.g. a
        # pinned image not yet committed to another chat), and deleting them out
        # from under the frontend's hash cache produced dangling references /
        # broken images. Orphaned image files are cheap on a local disk; reclaim
        # them explicitly via gc_orphan_images() if it ever matters.


def gc_orphan_images() -> None:
    """Explicitly reclaim image files/rows not referenced by any chat or message.

    Deliberately NOT run automatically (see delete_chat) — only safe to call when
    no browser session might still hold an uncommitted image's bytes.
    """
    with _connect() as conn:
        _gc_orphan_images(conn)


# --------------------------------------------------------------------------- #
# Documents (RAG). Files and indexes live in rag.DOCS_DIR; these are the rows.
# --------------------------------------------------------------------------- #
def _chat_doc_ids(conn: sqlite3.Connection, chat_id: str) -> list[str]:
    rows = conn.execute(
        "SELECT doc_id FROM chat_documents WHERE chat_id = ?", (chat_id,)
    )
    return [r["doc_id"] for r in rows]


def list_documents() -> list[dict]:
    with _connect() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM documents ORDER BY created_at DESC"
        )]


def get_document(doc_id: str) -> dict | None:
    with _connect() as conn:
        r = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
    return dict(r) if r else None


def get_documents(doc_ids: list[str]) -> list[dict]:
    """Rows for these ids that exist and are searchable, in the order asked for."""
    if not doc_ids:
        return []
    with _connect() as conn:
        rows = {r["id"]: dict(r) for r in conn.execute(
            f"SELECT * FROM documents WHERE status = 'ready' AND id IN "
            f"({','.join('?' * len(doc_ids))})", list(doc_ids),
        )}
    return [rows[d] for d in doc_ids if d in rows]


def document_by_hash(sha: str) -> dict | None:
    with _connect() as conn:
        r = conn.execute("SELECT * FROM documents WHERE sha256 = ?", (sha,)).fetchone()
    return dict(r) if r else None


def create_document(name: str, sha: str, kind: str, size_bytes: int) -> str:
    doc_id = uuid.uuid4().hex
    with _connect() as conn:
        conn.execute(
            "INSERT INTO documents(id, name, sha256, kind, size_bytes, status, "
            "created_at) VALUES (?, ?, ?, ?, ?, 'indexing', ?)",
            (doc_id, name, sha, kind, size_bytes, _now()),
        )
    return doc_id


_DOC_FIELDS = {"name", "pages", "n_chunks", "embed_model", "dim", "status", "error"}


def update_document(doc_id: str, **fields) -> None:
    fields = {k: v for k, v in fields.items() if k in _DOC_FIELDS}
    if not fields:
        return
    with _connect() as conn:
        conn.execute(
            f"UPDATE documents SET {', '.join(f'{k} = ?' for k in fields)} WHERE id = ?",
            [*fields.values(), doc_id],
        )


def delete_document(doc_id: str) -> None:
    """Drop the row; chat selections go with it (ON DELETE CASCADE)."""
    with _connect() as conn:
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))


def document_ids() -> set[str]:
    with _connect() as conn:
        return {r["id"] for r in conn.execute("SELECT id FROM documents")}
