"""FastAPI backend for the Vision Model Chat UI.

Serves the built React frontend and proxies all Ollama interaction so the
browser never talks to Ollama directly.
"""
import json
import queue
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import db
from . import ollama_client as oc
from . import sd_client as sd

app = FastAPI(title="Vision Model Chat")

# Allow the Vite dev server (5173) to call the API during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve persisted image bytes / thumbnails as static files. Registered before
# the catch-all frontend mount so `/api/*` always wins.
app.mount("/api/images", StaticFiles(directory=str(db.IMAGES_DIR)), name="images")
app.mount("/api/thumbs", StaticFiles(directory=str(db.THUMBS_DIR)), name="thumbs")


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class Message(BaseModel):
    role: str
    content: str
    images: list[str] | None = None


class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    ollama_url: str = oc.DEFAULT_URL


class PullRequest(BaseModel):
    name: str
    ollama_url: str = oc.DEFAULT_URL


class ImageItem(BaseModel):
    full: str
    thumb: str | None = None


class ImagesRequest(BaseModel):
    items: list[ImageItem]


class ChatUpsert(BaseModel):
    model: str | None = None
    system_prompt: str = ""
    pinned_hashes: list[str] = []
    system_image_hash: str | None = None


class MessageAppend(BaseModel):
    role: str
    content: str = ""
    model: str | None = None
    image_hashes: list[str] = []
    context_hashes: list[str] = []


class TitleRequest(BaseModel):
    model: str
    ollama_url: str = oc.DEFAULT_URL


class GenerateRequest(BaseModel):
    mode: str = "txt2img"  # txt2img | img2img
    model: str | None = None
    prompt: str = ""
    negative_prompt: str = ""
    init_image_hash: str | None = None
    steps: int | None = None
    guidance: float | None = None
    strength: float | None = None
    width: int = 512
    height: int = 512
    seed: int | None = None
    ollama_url: str = oc.DEFAULT_URL


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
@app.get("/api/models")
def get_models(ollama_url: str = oc.DEFAULT_URL):
    return {
        "vision": oc.list_vision_models(ollama_url),
        "all": oc.list_all_models(ollama_url),
    }


@app.post("/api/chat")
def chat(req: ChatRequest):
    messages = [m.model_dump(exclude_none=True) for m in req.messages]

    def gen():
        try:
            for event in oc.stream_chat(req.ollama_url, req.model, messages):
                yield json.dumps(event) + "\n"
        except Exception as exc:  # surfaced inline so the client can show it
            yield json.dumps({"type": "error", "message": str(exc)}) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.post("/api/models/pull")
def pull_model(req: PullRequest):
    return StreamingResponse(
        oc.pull(req.ollama_url, req.name), media_type="application/x-ndjson"
    )


@app.delete("/api/models/{name:path}")
def delete_model(name: str, ollama_url: str = oc.DEFAULT_URL):
    try:
        oc.delete(ollama_url, name)
        return {"ok": True}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@app.post("/api/models/unload")
def unload_models(ollama_url: str = oc.DEFAULT_URL):
    try:
        return {"unloaded": oc.unload_all(ollama_url)}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@app.get("/api/ps")
def ps(ollama_url: str = oc.DEFAULT_URL):
    try:
        return {"models": oc.running(ollama_url)}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))


# --------------------------------------------------------------------------- #
# Local image generation (diffusers, in-process)
# --------------------------------------------------------------------------- #
@app.get("/api/generate/models")
def generate_models():
    return {"available": sd.available(), "device": sd.device(), "models": sd.list_models()}


@app.post("/api/generate/unload")
def generate_unload():
    sd.unload()
    return {"ok": True}


class SdPullRequest(BaseModel):
    model: str


@app.post("/api/generate/pull")
def generate_pull(req: SdPullRequest):
    """Explicitly download an image model's weights (opt-in, streams status)."""
    if not sd.available():
        raise HTTPException(status_code=503, detail="Image generation deps not installed.")

    def gen():
        events: queue.Queue = queue.Queue()

        def worker():
            try:
                sd.pull(req.model, on_status=lambda m: events.put({"type": "status", "message": m}))
                events.put({"type": "done"})
            except Exception as exc:
                events.put({"type": "error", "message": str(exc)})
            finally:
                events.put(None)

        threading.Thread(target=worker, daemon=True).start()
        while True:
            item = events.get()
            if item is None:
                break
            yield json.dumps(item) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.post("/api/generate")
def generate(req: GenerateRequest):
    """Run a local diffusion generation, streaming progress then the final image.

    Frees VRAM from any resident Ollama vision model first (the 6 GB can't hold
    both). The diffusion call is synchronous, so it runs in a worker thread that
    pushes events onto a queue the response generator drains.
    """
    if not sd.available():
        raise HTTPException(
            status_code=503,
            detail="Image generation deps (torch/diffusers) are not installed.",
        )

    # Resolve the init image (img2img) from the content-addressed store to a PIL.
    init_image = None
    if req.init_image_hash:
        from PIL import Image

        path = db.IMAGES_DIR / f"{req.init_image_hash}.jpg"
        if not path.exists():
            raise HTTPException(status_code=404, detail="init image not found")
        init_image = Image.open(path).convert("RGB")

    def gen():
        events: queue.Queue = queue.Queue()

        def worker():
            try:
                events.put({"type": "status", "message": "Freeing VRAM (unloading vision model)…"})
                try:
                    oc.unload_all(req.ollama_url)
                except Exception:
                    pass  # best-effort; Ollama may be remote or already free

                params = {
                    "mode": req.mode,
                    "model": req.model,
                    "prompt": req.prompt,
                    "negative_prompt": req.negative_prompt,
                    "steps": req.steps,
                    "guidance": req.guidance,
                    "strength": req.strength,
                    "width": req.width,
                    "height": req.height,
                    "seed": req.seed,
                    "init_image": init_image,
                }
                image, seed = sd.generate(
                    params,
                    on_step=lambda s, t: events.put({"type": "progress", "step": s, "total": t}),
                    on_status=lambda m: events.put({"type": "status", "message": m}),
                )
                full = sd.pil_to_data_url(image)
                thumb = sd.pil_to_data_url(image, max_size=64)
                h = db.save_image(full, thumb)
                events.put(
                    {
                        "type": "image",
                        "hash": h,
                        "seed": seed,
                        "width": image.size[0],
                        "height": image.size[1],
                    }
                )
            except Exception as exc:
                events.put({"type": "error", "message": str(exc)})
            finally:
                events.put(None)  # sentinel: worker done

        threading.Thread(target=worker, daemon=True).start()
        while True:
            item = events.get()
            if item is None:
                break
            yield json.dumps(item) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")


# --------------------------------------------------------------------------- #
# Ollama version / upgrade
# --------------------------------------------------------------------------- #
@app.get("/api/ollama/version")
def ollama_version(ollama_url: str = oc.DEFAULT_URL):
    return oc.version_info(ollama_url)


@app.post("/api/ollama/upgrade")
def ollama_upgrade(ollama_url: str = oc.DEFAULT_URL):
    if not oc.is_local(ollama_url):
        raise HTTPException(
            status_code=400,
            detail="Upgrade is only available for a local Ollama. Upgrade on the host instead.",
        )
    return StreamingResponse(oc.upgrade(), media_type="text/plain; charset=utf-8")


# --------------------------------------------------------------------------- #
# Chat history persistence
# --------------------------------------------------------------------------- #
@app.post("/api/images")
def upload_images(req: ImagesRequest):
    """Store images (full + optional thumbnail), deduped. Returns their hashes."""
    return {"hashes": [db.save_image(it.full, it.thumb) for it in req.items]}


@app.get("/api/chats")
def get_chats():
    return {"chats": db.list_chats()}


@app.get("/api/chats/{chat_id}")
def get_chat(chat_id: str):
    chat = db.get_chat(chat_id)
    if chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat


@app.put("/api/chats/{chat_id}")
def put_chat(chat_id: str, req: ChatUpsert):
    db.upsert_chat(
        chat_id,
        req.model,
        req.system_prompt,
        req.pinned_hashes,
        req.system_image_hash,
    )
    return {"ok": True}


@app.post("/api/chats/{chat_id}/messages")
def add_message(chat_id: str, req: MessageAppend):
    msg_id = db.append_message(
        chat_id, req.role, req.content, req.model, req.image_hashes, req.context_hashes
    )
    return {"id": msg_id}


@app.post("/api/chats/{chat_id}/title")
def make_title(chat_id: str, req: TitleRequest):
    exchange = db.get_first_exchange(chat_id)
    if not exchange:
        raise HTTPException(status_code=404, detail="Chat has no messages")
    first_user, first_assistant = exchange
    title = oc.generate_title(req.ollama_url, req.model, first_user, first_assistant)
    if title:
        db.set_title(chat_id, title)
    return {"title": title}


@app.delete("/api/chats/{chat_id}")
def remove_chat(chat_id: str):
    db.delete_chat(chat_id)
    return {"ok": True}


@app.on_event("shutdown")
def _unload_on_shutdown():
    try:
        oc.unload_all(oc.DEFAULT_URL)
    except Exception:
        pass
    try:
        sd.unload()
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Serve the built frontend (production). In dev, Vite serves on :5173.
# --------------------------------------------------------------------------- #
_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if _DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(_DIST), html=True), name="frontend")
