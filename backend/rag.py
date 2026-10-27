"""Document retrieval (RAG): text files and PDFs the model can answer from.

Every document is its own little index, not a row in one shared vector database:

    data/rag/<doc_id>/
        source.<ext>          the upload, kept so the document can be re-indexed
        index/chunks.jsonl    one {"text", "page"} per chunk, in document order
        index/vectors.npy     float32, L2-normalised, one row per chunk

That layout is the whole design. Deleting a document is one `rmtree` — the disk space
comes back immediately, with no tombstones, no VACUUM and no shared index to rebuild,
and nothing another document depends on can be touched by it. Choosing which documents
a chat uses is just which folders a search opens. And a broken or interrupted ingest
can only ever damage its own folder: the index is built in `index.tmp/` and swapped in
with a rename, so a document either has a complete index or none.

Search is exact brute force: a dot product over the selected documents' memory-mapped
vectors. A local library of a few hundred documents is ~10^5 chunks, which numpy scores
in milliseconds — an approximate index would add a dependency and a rebuild step to buy
nothing. If that ever stops being true, this module's interface is what to keep.

Embeddings come from Ollama's /api/embed, so the app stays offline once the embedding
model is pulled. Each document records which model built it; a query is embedded with
that same model, so switching the default never mixes vectors from two models.

PDFs are read with PDFium. A page with no text layer (a scan, a photographed page) is
rendered and transcribed by the local vision model instead, so a scanned PDF indexes
like any other — just slower.
"""
import base64
import hashlib
import io
import json
import logging
import os
import re
import shutil
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np
import requests

from . import db
from . import ollama_client as oc

DOCS_DIR = db.DATA_DIR / "rag"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_EMBED_MODEL = "nomic-embed-text"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024

# ~300 tokens: small enough that a hit is about one thing, big enough to carry the
# sentence around the fact. The overlap keeps a fact that straddles a boundary whole
# in at least one chunk.
CHUNK_CHARS = 1200
CHUNK_OVERLAP = 200
EMBED_BATCH = 32
# Short, unlike the chat models' 30m: the embedder is used in bursts, and every MB it
# holds is one FLUX can't have.
EMBED_KEEP_ALIVE = "5m"
EMBED_TIMEOUT = (10, 300)

# A page with less text than this is treated as having no text layer worth keeping —
# a scan with a stray page number, say — and goes to the vision model instead.
OCR_MIN_CHARS = 25
# Long side of a page rendered for OCR. Enough for body text to be legible to a
# vision model without paying for a poster-sized image in vision tokens.
OCR_RENDER_PX = 1600
OCR_TIMEOUT = (10, 600)
_OCR_SYSTEM = (
    "You transcribe document pages. Write out all of the text on the page exactly, "
    "in reading order, keeping paragraphs and list items on their own lines and "
    "tables as rows of text. Output only the transcription — no commentary, no "
    "description of the page. If the page has no text, output nothing."
)

# How much retrieved text goes into one prompt. ~2k tokens — enough for six chunks,
# and small enough to sit inside the extra window `ollama_client` adds for it.
CONTEXT_BUDGET_CHARS = 8000
DEFAULT_TOP_K = 6
# Cosine similarity below which a chunk is noise. Deliberately low: the user picked
# these documents for this chat, so a weak match is still more often wanted than not.
MIN_SCORE = 0.25

TEXT_EXTS = {
    "txt", "md", "markdown", "rst", "csv", "tsv", "json", "jsonl", "log", "xml",
    "html", "htm", "yaml", "yml", "ini", "toml", "cfg", "conf", "tex", "srt", "vtt",
    "py", "js", "ts", "tsx", "jsx", "java", "c", "h", "cpp", "hpp", "cs", "go", "rs",
    "rb", "php", "sh", "bash", "sql", "css", "scss", "swift", "kt", "lua", "r",
}

# Marks a message that carries retrieved context, so `ollama_client.context_size_for`
# can give it room without every chat call having to thread a size through.
CONTEXT_OPEN = oc.DOCS_TAG
CONTEXT_CLOSE = "</documents>"


class NoText(ValueError):
    """The file parsed but holds no text we could read."""


# --------------------------------------------------------------------------- #
# Files
# --------------------------------------------------------------------------- #
def kind_of(name: str, mime: str | None) -> str | None:
    """"pdf", "text", or None if this isn't something we can index."""
    ext = Path(name).suffix.lower().lstrip(".")
    if ext == "pdf" or mime == "application/pdf":
        return "pdf"
    if ext in TEXT_EXTS or (mime or "").startswith("text/"):
        return "text"
    return None


def doc_dir(doc_id: str) -> Path:
    # doc ids are minted by db as hex; anything else arriving off a URL is refused
    # rather than turned into a path.
    if not re.fullmatch(r"[0-9a-f]{32}", doc_id or ""):
        raise ValueError("bad document id")
    return DOCS_DIR / doc_id


def source_path(doc_id: str) -> Path | None:
    return next(doc_dir(doc_id).glob("source.*"), None)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def disk_bytes(doc_id: str) -> int:
    """Everything the document occupies on disk — source and index together."""
    root = doc_dir(doc_id)
    if not root.exists():
        return 0
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
_TAGS = re.compile(r"<(script|style)\b.*?</\1>|<[^>]+>", re.S | re.I)


def _ocr(url: str, model: str, image) -> str:
    """Transcribe one rendered page with the vision model. "" on any failure."""
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=90)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _OCR_SYSTEM},
            {"role": "user", "content": "Transcribe this page.",
             "images": [base64.b64encode(buf.getvalue()).decode()]},
        ],
        "stream": False,
        "think": False,
        "keep_alive": "30m",  # same as chat: it's the chat model, wanted again next
        "options": {"num_ctx": 8192, "temperature": 0},
    }
    try:
        r = requests.post(f"{url}/api/chat", json=payload, timeout=OCR_TIMEOUT)
        if r.status_code != 200:
            logging.warning("OCR with %s failed: %s", model, r.text[:200])
            return ""
        return ((r.json().get("message") or {}).get("content") or "").strip()
    except (requests.RequestException, ValueError) as exc:
        logging.warning("OCR with %s failed: %s", model, exc)
        return ""


def _blank(image) -> bool:
    """A page with nothing on it — not worth a vision-model call."""
    lo, hi = image.convert("L").getextrema()
    return lo > 235 or hi - lo < 12


def _extract_pdf(path: Path, url: str, ocr_model: str | None, emit) -> list:
    import pypdfium2 as pdfium

    try:
        pdf = pdfium.PdfDocument(str(path))
    except pdfium.PdfiumError as exc:
        if "password" in str(exc).lower():
            raise ValueError("This PDF is password-protected.")
        raise ValueError(f"This PDF couldn't be opened: {exc}")
    try:
        n = len(pdf)
        pages, scanned = [], []
        for i in range(n):
            page = pdf[i]
            try:
                text = page.get_textpage().get_text_bounded() or ""
            except Exception:
                text = ""
            pages.append([i + 1, text])
            if len(text.strip()) < OCR_MIN_CHARS:
                scanned.append(i)

        if scanned and not ocr_model:
            emit({"type": "status", "stage": "extract", "message":
                  f"{len(scanned)} page(s) have no text layer and no vision model is "
                  "selected to read them — skipping those."})
        elif scanned:
            for done, i in enumerate(scanned, 1):
                emit({"type": "progress", "stage": "ocr", "done": done - 1,
                      "total": len(scanned),
                      "message": f"Reading scanned page {i + 1} with {ocr_model} "
                                 f"({done}/{len(scanned)})…"})
                page = pdf[i]
                w, h = page.get_size()
                scale = OCR_RENDER_PX / max(w, h, 1)
                image = page.render(scale=scale).to_pil()
                if _blank(image):
                    continue
                text = _ocr(url, ocr_model, image)
                if text:
                    pages[i][1] = text
            emit({"type": "progress", "stage": "ocr", "done": len(scanned),
                  "total": len(scanned), "message": "Scanned pages read."})
        return [(p, t) for p, t in pages]
    finally:
        pdf.close()


def extract(path: Path, kind: str, url: str = "", ocr_model: str | None = None,
            emit=lambda e: None) -> list[tuple[int | None, str]]:
    """The document's text as (page, text) pairs. Page is None for text files."""
    if kind == "pdf":
        return _extract_pdf(path, url, ocr_model, emit)

    raw = path.read_bytes()
    bom16 = raw.startswith((b"\xff\xfe", b"\xfe\xff"))
    if b"\x00" in raw[:4096] and not bom16:
        raise ValueError("That file looks binary, not text.")
    text = raw.decode("utf-16" if bom16 else "utf-8-sig", errors="replace")
    if path.suffix.lower() in (".html", ".htm"):
        text = _TAGS.sub(" ", text)
    return [(None, text)]


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _pieces(text: str) -> list[str]:
    """Paragraphs, with any paragraph too long for one chunk split into sentences,
    and any sentence still too long cut at the limit."""
    out = []
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        if len(para) <= CHUNK_CHARS:
            out.append(para)
            continue
        for sent in _SENTENCE.split(para):
            while len(sent) > CHUNK_CHARS:
                out.append(sent[:CHUNK_CHARS])
                sent = sent[CHUNK_CHARS - CHUNK_OVERLAP:]
            if sent.strip():
                out.append(sent)
    return out


def chunk(pages: list[tuple[int | None, str]]) -> list[dict]:
    """Pack each page's paragraphs into ~CHUNK_CHARS chunks with a little overlap.

    Chunks never cross a page, so every hit can cite exactly one page.
    """
    chunks = []
    for page, text in pages:
        buf = ""
        for piece in _pieces(_normalize(text)):
            if buf and len(buf) + 2 + len(piece) > CHUNK_CHARS:
                chunks.append({"text": buf, "page": page})
                # Carry the end of this chunk into the next one, from a word boundary.
                tail = buf[-CHUNK_OVERLAP:]
                buf = tail[tail.find(" ") + 1:] if " " in tail else tail
            buf = f"{buf}\n\n{piece}" if buf else piece
        if buf.strip():
            chunks.append({"text": buf, "page": page})
    return chunks


# --------------------------------------------------------------------------- #
# Embedding
# --------------------------------------------------------------------------- #
def _prefix(model: str, role: str) -> str:
    """Task prefixes some embedders are trained with and quietly degrade without."""
    base = model.split(":")[0].split("/")[-1]
    if base.startswith("nomic-embed"):
        return "search_query: " if role == "query" else "search_document: "
    if base.startswith(("mxbai-embed", "snowflake-arctic-embed")) and role == "query":
        return "Represent this sentence for searching relevant passages: "
    return ""


def embed(url: str, model: str, texts: list[str], role: str = "document") -> np.ndarray:
    """Embed texts with an Ollama embedding model. Rows come back L2-normalised."""
    payload = {
        "model": model,
        "input": [_prefix(model, role) + t for t in texts],
        "keep_alive": EMBED_KEEP_ALIVE,
        "truncate": True,
    }
    try:
        r = requests.post(f"{url}/api/embed", json=payload, timeout=EMBED_TIMEOUT)
    except requests.RequestException as exc:
        raise RuntimeError(f"Couldn't reach Ollama to embed: {exc}")
    if r.status_code == 404:
        raise RuntimeError(
            f"The embedding model {model} isn't installed. Pull it from the "
            f"Documents panel, or run `ollama pull {model}`."
        )
    if r.status_code != 200:
        raise RuntimeError(f"Ollama couldn't embed with {model}: {r.text[:300]}")
    vecs = np.asarray(r.json().get("embeddings") or [], dtype=np.float32)
    if vecs.ndim != 2 or len(vecs) != len(texts):
        raise RuntimeError(f"{model} didn't return embeddings — is it an embedding model?")
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.maximum(norms, 1e-12)


def model_installed(url: str, model: str) -> bool:
    try:
        r = requests.post(f"{url}/api/show", json={"name": model}, timeout=10)
        return r.status_code == 200
    except requests.RequestException:
        return False


# --------------------------------------------------------------------------- #
# Ingest
# --------------------------------------------------------------------------- #
def store_source(doc_id: str, name: str, data: bytes) -> Path:
    """Write the upload into the document's folder. Done before indexing, so a
    failed or interrupted index can always be retried from what's on disk."""
    root = doc_dir(doc_id)
    root.mkdir(parents=True, exist_ok=True)
    ext = Path(name).suffix.lower()
    if not re.fullmatch(r"\.[a-z0-9]{1,10}", ext):
        ext = ".txt"
    path = root / f"source{ext}"
    path.write_bytes(data)
    return path


def build_index(doc_id: str, kind: str, url: str, model: str,
                ocr_model: str | None, emit) -> dict:
    """Extract, chunk and embed a stored source; swap the new index in atomically.

    Used for the first index and for every re-index alike. Returns the facts the
    metadata row records. Raises `NoText` for a file with nothing to index.
    """
    src = source_path(doc_id)
    if src is None:
        raise ValueError("The original file is missing; upload it again.")
    root = doc_dir(doc_id)

    emit({"type": "status", "stage": "extract", "message": "Reading the file…"})
    pages = extract(src, kind, url, ocr_model, emit)
    chunks = chunk(pages)
    if not chunks:
        raise NoText(
            "No readable text found. If this is a scanned PDF, pick a vision model "
            "in the composer and re-index — scanned pages are read with it."
            if kind == "pdf" else "That file has no text in it."
        )

    tmp = root / "index.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    total = len(chunks)
    vectors = []
    for start in range(0, total, EMBED_BATCH):
        batch = chunks[start:start + EMBED_BATCH]
        vectors.append(embed(url, model, [c["text"] for c in batch]))
        done = min(start + EMBED_BATCH, total)
        emit({"type": "progress", "stage": "embed", "done": done, "total": total,
              "message": f"Embedding {done}/{total} chunks…"})
    matrix = np.vstack(vectors).astype(np.float32)
    np.save(tmp / "vectors.npy", matrix)
    with open(tmp / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # Swap: the old index (if this is a re-index) steps aside, the new one takes its
    # name, then the old one goes. Each rename is atomic, so a reader — or a crash —
    # sees a whole index or none.
    final, old = root / "index", root / "index.old"
    shutil.rmtree(old, ignore_errors=True)
    with _cache_lock:
        _cache.pop(doc_id, None)
        if final.exists():
            os.replace(final, old)
        os.replace(tmp, final)
    shutil.rmtree(old, ignore_errors=True)

    n_pages = sum(1 for p, _ in pages if p is not None)
    return {"pages": n_pages or None, "n_chunks": total, "dim": int(matrix.shape[1]),
            "embed_model": model}


def delete(doc_id: str) -> None:
    """Remove a document's folder — source, index, everything — from disk."""
    with _cache_lock:
        _cache.pop(doc_id, None)
    shutil.rmtree(doc_dir(doc_id), ignore_errors=True)


def gc(known_ids: set[str]) -> None:
    """Startup sweep: folders with no metadata row, and half-built indexes, go."""
    for p in DOCS_DIR.iterdir():
        if not p.is_dir():
            continue
        if p.name not in known_ids:
            shutil.rmtree(p, ignore_errors=True)
            continue
        for stale in ("index.tmp", "index.old"):
            shutil.rmtree(p / stale, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #
# doc_id -> (vectors memmap, chunks). Bounded: memmaps cost address space, not RAM,
# but the chunk text is held in memory.
_cache: "OrderedDict[str, tuple[np.ndarray, list[dict]]]" = OrderedDict()
_cache_lock = threading.Lock()
_CACHE_MAX = 32


def _load(doc_id: str) -> tuple[np.ndarray, list[dict]] | None:
    with _cache_lock:
        if doc_id in _cache:
            _cache.move_to_end(doc_id)
            return _cache[doc_id]
        index = doc_dir(doc_id) / "index"
        try:
            vecs = np.load(index / "vectors.npy", mmap_mode="r")
            with open(index / "chunks.jsonl", encoding="utf-8") as f:
                chunks = [json.loads(line) for line in f if line.strip()]
        except (OSError, ValueError):
            return None
        _cache[doc_id] = (vecs, chunks)
        while len(_cache) > _CACHE_MAX:
            _cache.popitem(last=False)
        return vecs, chunks


def search(url: str, docs: list[dict], query: str, k: int = DEFAULT_TOP_K) -> list[dict]:
    """The best chunks for `query` across `docs` (metadata rows from db).

    Documents small enough to fit the context budget whole are returned whole, in
    order — retrieval can only lose information there, and "summarise this" has no
    query to retrieve with. Otherwise it's the top `k` by cosine similarity, with
    each document capped so one long file can't crowd out the rest.
    """
    loaded = [(d, _load(d["id"])) for d in docs]
    loaded = [(d, ix) for d, ix in loaded if ix is not None]
    if not loaded:
        return []

    total_chars = sum(len(c["text"]) for _, (_, chunks) in loaded for c in chunks)
    if total_chars <= CONTEXT_BUDGET_CHARS:
        return [
            {"doc_id": d["id"], "name": d["name"], "page": c.get("page"),
             "text": c["text"], "score": None}
            for d, (_, chunks) in loaded for c in chunks
        ]

    # One query embedding per model the selected documents were built with.
    by_model: dict[str, np.ndarray] = {}
    hits = []
    for d, (vecs, chunks) in loaded:
        m = d.get("embed_model") or DEFAULT_EMBED_MODEL
        if m not in by_model:
            by_model[m] = embed(url, m, [query.strip() or "summary"], role="query")[0]
        q = by_model[m]
        if vecs.shape[1] != q.shape[0]:
            continue  # built by a model of another width; needs a re-index
        scores = np.asarray(vecs @ q)
        top = np.argsort(-scores)[:k]
        hits.extend(
            {"doc_id": d["id"], "name": d["name"], "page": chunks[i].get("page"),
             "text": chunks[i]["text"], "score": float(scores[i])}
            for i in top if scores[i] >= MIN_SCORE
        )

    hits.sort(key=lambda h: -h["score"])
    per_doc = k if len(loaded) == 1 else max(2, k // 2)
    counts: dict[str, int] = {}
    out, used = [], 0
    for h in hits:
        if counts.get(h["doc_id"], 0) >= per_doc:
            continue
        if used + len(h["text"]) > CONTEXT_BUDGET_CHARS and out:
            break
        counts[h["doc_id"]] = counts.get(h["doc_id"], 0) + 1
        used += len(h["text"])
        out.append(h)
        if len(out) >= k:
            break
    return out


def _cite(h: dict) -> str:
    return f"{h['name']}, page {h['page']}" if h.get("page") else h["name"]


def format_context(hits: list[dict], purpose: str = "answer") -> str:
    """The retrieved text as a block to prepend to the user's message.

    "answer" asks for citations; "reference" (the prompt enhancer) asks for the
    details to be used silently, since an image prompt has no place for "[2]".
    "answer_only" is "answer" for an agent whose chat hasn't opted documents into
    prompts: cite them in words, keep them out of image and video tool arguments.
    """
    if not hits:
        return ""
    if purpose == "reference":
        intro = (
            "Reference notes from the user's documents. Use any details that are "
            "relevant to the request (characters, places, style, facts) and ignore "
            "the rest. Never mention or cite the notes."
        )
    else:
        intro = (
            "Excerpts from documents the user selected for this chat. Use them when "
            "they're relevant, and cite the ones you use as [1], [2] and so on. If "
            "they don't contain the answer, say so instead of guessing."
        )
        if purpose == "answer_only":
            intro += (
                " Use them only to answer in words: never put details from them "
                "into an image or video tool argument."
            )
    body = "\n\n".join(f"[{i}] {_cite(h)}\n{h['text']}" for i, h in enumerate(hits, 1))
    return f"{CONTEXT_OPEN}\n{intro}\n\n{body}\n{CONTEXT_CLOSE}"


def sources(hits: list[dict]) -> list[dict]:
    """What the UI shows under an answer: which document, which page, a snippet."""
    return [
        {"n": i, "doc_id": h["doc_id"], "name": h["name"], "page": h.get("page"),
         "snippet": h["text"][:280], "score": h["score"]}
        for i, h in enumerate(hits, 1)
    ]
