"""SQLite persistence for chats, messages, and images.

The app used to keep every conversation in browser RAM; this module gives it
durable, browsable history. Chat/message rows live in a SQLite file; image
*bytes* live as files on disk (deduped by sha256 content hash) with a `hash`
registry row. Thumbnails (generated client-side) are stored under the same hash.

`sqlite3` and `hashlib` are stdlib — no new dependencies.
"""
import base64
import hashlib
import sqlite3
import time
import uuid
from pathlib import Path

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
CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id, ordinal);
CREATE INDEX IF NOT EXISTS idx_pinned_chat ON chat_pinned_images(chat_id, ordinal);
CREATE INDEX IF NOT EXISTS idx_msgimg_msg ON message_images(message_id, ordinal);
CREATE INDEX IF NOT EXISTS idx_msgctx_msg ON message_context_images(message_id, ordinal);
"""


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(_SCHEMA)


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


def save_image(full_data_url: str, thumb_data_url: str | None = None) -> str:
    """Persist one image (full + optional thumbnail) deduped by content hash.

    Returns the sha256 hash. Both the full image and its thumbnail are stored
    under the same hash so the sidebar can request `/api/thumbs/<hash>.jpg`.
    """
    raw = _strip_data_url(full_data_url)
    h = hashlib.sha256(raw).hexdigest()

    full_path = IMAGES_DIR / f"{h}.jpg"
    if not full_path.exists():
        full_path.write_bytes(raw)

    if thumb_data_url:
        thumb_path = THUMBS_DIR / f"{h}.jpg"
        if not thumb_path.exists():
            thumb_path.write_bytes(_strip_data_url(thumb_data_url))

    with _connect() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO images(hash, mime, created_at) VALUES (?, 'image/jpeg', ?)",
            (h, _now()),
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
        (IMAGES_DIR / f"{h}.jpg").unlink(missing_ok=True)
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
                    "images": [f"/api/images/{r['image_hash']}.jpg" for r in imgs],
                    "context_images": [f"/api/images/{r['image_hash']}.jpg" for r in ctx],
                }
            )

        pinned = [f"/api/images/{h}.jpg" for h in _pinned_hashes(conn, chat_id)]
        sys_img = (
            f"/api/images/{c['system_image_hash']}.jpg"
            if c["system_image_hash"]
            else None
        )
        return {
            "id": c["id"],
            "title": c["title"],
            "model": c["model"],
            "system_prompt": c["system_prompt"],
            "system_image": sys_img,
            "pinned": pinned,
            "messages": messages,
        }


def upsert_chat(
    chat_id: str,
    model: str | None,
    system_prompt: str,
    pinned_hashes: list[str],
    system_image_hash: str | None,
) -> None:
    """Create the chat row if new, else update its metadata + pinned images.

    Title is never overwritten here (it's set once by the LLM titler).
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


def append_message(
    chat_id: str,
    role: str,
    content: str,
    model: str | None,
    image_hashes: list[str],
    context_hashes: list[str] | None = None,
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
            "INSERT INTO messages(id, chat_id, ordinal, role, content, model, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (msg_id, chat_id, ordinal, role, content, model, now),
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
