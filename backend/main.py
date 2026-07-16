"""FastAPI backend for the Vision Model Chat UI.

Serves the built React frontend and proxies all Ollama interaction so the
browser never talks to Ollama directly.
"""
import base64
import json
import os
import queue
import random
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import images  # noqa: F401  (imported first: pins HF offline env)
from . import db
from . import flux_catalog as cat
from . import flux_client as fx
from . import ollama_client as oc
from . import settings

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


# More than a handful of references costs vision tokens without telling the model
# much more about what the user wants.
MAX_ENHANCE_IMAGES = 4


class EnhanceRequest(BaseModel):
    prompt: str
    mode: str = "txt2img"  # txt2img | img2img | edit | compose
    model: str  # the Ollama vision model to rewrite with
    image_hashes: list[str] = []
    ollama_url: str = oc.DEFAULT_URL


class GenerateRequest(BaseModel):
    # No negative prompt is exposed. On FLUX that's because there's nothing to expose:
    # FLUX.1 samples at cfg=1.0 (the negative branch has no effect) and FLUX.2 has no
    # negative branch at all. `animate` runs on Wan, which does use a real negative at
    # cfg~3.5 — but it's a fixed quality string the model was tuned against, not a knob
    # (see flux_client.WAN_NEGATIVE), so it stays out of the API.
    mode: str = "txt2img"  # txt2img | img2img | edit | compose | animate
    prompt: str = ""
    init_image_hash: str | None = None  # img2img / edit / animate: the source image
    ref_image_hashes: list[str] = []  # compose: reference images to fuse
    flux_model: str | None = None  # which UNet (None = that mode's default)
    steps: int | None = None
    guidance: float | None = None
    strength: float | None = None  # img2img: how far to drift from the source
    enhance: bool = True  # wrap create prompts in a photoreal template
    width: int = 1024  # FLUX is trained at ~1 megapixel
    height: int = 1024
    seconds: float | None = None  # animate: clip length (capped at 5s)
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
# Image generation (FLUX, via the ComfyUI sidecar)
# --------------------------------------------------------------------------- #
@app.post("/api/generate/unload")
def generate_unload():
    fx.free()
    return {"ok": True}


# --------------------------------------------------------------------------- #
# Image models: install a catalog bundle, add a UNet from any HF repo, remove either
# --------------------------------------------------------------------------- #
def _ndjson(work) -> StreamingResponse:
    """Run `work(emit)` on a thread and stream whatever it emits as NDJSON.

    Downloads are tens of GB, so the browser gets progress as it happens rather than
    one response an hour later.
    """
    def gen():
        events: queue.Queue = queue.Queue()

        def worker():
            try:
                work(events.put)
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


@app.get("/api/flux/models")
def flux_models():
    """Installed transformers, each tagged with the roles (create / edit) it can serve."""
    return {"available": fx.available(), "models": fx.list_unets()}


@app.post("/api/flux/enhance")
def flux_enhance(req: EnhanceRequest):
    """Rewrite a prompt with a local vision model that can see the attached images.

    Separate from /api/generate on purpose. Generate unloads Ollama to free VRAM, so
    enhancing inside it would reload the vision model for every run; and the whole
    point is that the user sees the rewrite and can edit it before sampling. Ordering
    then works out: enhance loads the VLM, the user reviews, generate frees it.

    Never fails: a rewrite the user can't get is a rewrite they type themselves, so
    an unreachable Ollama falls back to the static template rather than 500ing.
    """
    if req.model:
        imgs = []
        for h in req.image_hashes[:MAX_ENHANCE_IMAGES]:
            p = db.image_path(h)
            if not p:
                continue
            try:
                imgs.append(images.image_to_b64(p))
            except Exception:
                # Not decodable as an image — a stored video, most likely. This
                # endpoint's contract is that it never fails, so drop it and let the
                # model rewrite from whatever else it was given.
                pass
        text = oc.enhance_prompt(req.ollama_url, req.model, req.prompt, req.mode, imgs)
        if text:
            return {"prompt": text, "source": "vlm"}
    # No vision model installed, or Ollama couldn't answer.
    return {"prompt": fx.static_enhance(req.prompt, req.mode), "source": "template"}


@app.get("/api/flux/catalog")
def flux_catalog():
    """The installable models and what it would take to install them."""
    return {
        "runtime_ready": fx.runtime_ready(),
        "available": fx.available(),
        "disk_free_gb": cat.free_gb(),
        "bundles": fx.catalog(),
        "hf_token": settings.hf_token_source(),
        # A download survives the request that started it, so a reloaded page can find
        # it again here instead of assuming nothing is happening.
        "installing": fx.install_state(),
    }


class FluxInstallRequest(BaseModel):
    id: str


@app.post("/api/flux/install")
def flux_install(req: FluxInstallRequest):
    """Download a catalog model — weights, text encoder and VAE together."""
    if not fx.runtime_ready():
        raise HTTPException(status_code=503, detail="The image engine isn't installed.")

    def work(emit):
        fx.install_bundle(
            req.id,
            on_status=lambda m: emit({"type": "status", "message": m}),
            on_progress=lambda p: emit({"type": "progress", **p}),
        )

    return _ndjson(work)


@app.delete("/api/flux/bundles/{bundle_id}")
def flux_bundle_delete(bundle_id: str):
    try:
        fx.delete_bundle(bundle_id)
        return {"ok": True}
    except ValueError as exc:  # unknown id
        raise HTTPException(status_code=400, detail=str(exc))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="That model isn't installed.")


class FluxPullRequest(BaseModel):
    repo: str


@app.post("/api/flux/pull")
def flux_pull(req: FluxPullRequest):
    """Download an extra FLUX.1 UNet from any HuggingFace repo (opt-in, streams status)."""
    if not fx.runtime_ready():
        raise HTTPException(status_code=503, detail="The image engine isn't installed.")

    def work(emit):
        fx.pull_unet(
            req.repo,
            on_status=lambda m: emit({"type": "status", "message": m}),
            on_progress=lambda p: emit({"type": "progress", **p}),
        )

    return _ndjson(work)


@app.delete("/api/flux/models/{name}")
def flux_delete(name: str):
    try:
        fx.delete_unet(name)
        return {"ok": True}
    except ValueError as exc:  # part of a bundle, or not a model file
        raise HTTPException(status_code=400, detail=str(exc))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model not found.")


# --------------------------------------------------------------------------- #
# Text encoders — a FLUX.2 model's conditioning half. Swappable: the bundled one is
# only a default, and a lighter quant of the same encoder is the usual reason to change.
# --------------------------------------------------------------------------- #
@app.get("/api/flux/text-encoders")
def flux_text_encoders():
    return {"encoders": fx.list_text_encoders(), "selected": fx.selected_text_encoders()}


class TextEncoderPullRequest(BaseModel):
    repo: str


@app.post("/api/flux/text-encoders/pull")
def flux_text_encoder_pull(req: TextEncoderPullRequest):
    """Add a text encoder from HuggingFace. Needs owner/repo:file — see pull_text_encoder."""
    if not fx.runtime_ready():
        raise HTTPException(status_code=503, detail="The image engine isn't installed.")

    def work(emit):
        fx.pull_text_encoder(
            req.repo,
            on_status=lambda m: emit({"type": "status", "message": m}),
            on_progress=lambda p: emit({"type": "progress", **p}),
        )

    return _ndjson(work)


class TextEncoderSelectRequest(BaseModel):
    bundle_id: str
    name: str  # "" restores the model's default


@app.put("/api/flux/text-encoders/select")
def flux_text_encoder_select(req: TextEncoderSelectRequest):
    try:
        fx.set_text_encoder(req.bundle_id, req.name)
        return {"ok": True, "selected": fx.selected_text_encoders()}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="That text encoder isn't installed.")


@app.delete("/api/flux/text-encoders/{name}")
def flux_text_encoder_delete(name: str):
    try:
        fx.delete_text_encoder(name)
        return {"ok": True}
    except ValueError as exc:  # a model is currently using it
        raise HTTPException(status_code=400, detail=str(exc))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Text encoder not found.")


# --------------------------------------------------------------------------- #
# HuggingFace token — needed only for gated repos. Stored server-side, 0600, and
# never sent back to the browser: the UI only ever learns whether one is set.
# --------------------------------------------------------------------------- #
class HfTokenRequest(BaseModel):
    token: str


@app.get("/api/settings/hf-token")
def hf_token_get():
    return {"source": settings.hf_token_source()}


@app.put("/api/settings/hf-token")
def hf_token_put(req: HfTokenRequest):
    try:
        user = fx.verify_token(req.token)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    settings.set_hf_token(req.token)
    return {"source": "saved", "user": user}


@app.delete("/api/settings/hf-token")
def hf_token_delete():
    settings.clear_hf_token()
    return {"source": settings.hf_token_source()}


@app.post("/api/generate")
def generate(req: GenerateRequest):
    """Run a local image generation, streaming progress then the final image.

    Every mode runs on FLUX in the ComfyUI sidecar: create (txt2img/img2img) on
    FLUX dev, edit/compose on FLUX Kontext. The other GPU tenant (the Ollama vision
    model) is freed first, since the GPU can't hold both. The synchronous call runs
    in a worker thread that pushes events onto a queue the response generator drains.
    """
    if not fx.available():
        raise HTTPException(
            status_code=503,
            detail="FLUX isn't installed on this machine. Run ./run.sh to fetch the weights.",
        )

    # Resolve stored image hashes to PIL objects (init image for img2img/edit,
    # reference images for compose).
    from PIL import Image, ImageOps

    def _resolve(h: str):
        path = db.image_path(h)
        if path is None:
            raise HTTPException(status_code=404, detail=f"image {h} not found")
        # Orientation is applied here, not at write time — the store keeps a phone
        # photo's original bytes and its EXIF Orientation tag with them, rather than
        # paying a lossy re-encode to bake the rotation in. Uploads used to be
        # laundered through a browser canvas, which applied it silently; without
        # this, that removal would land every phone photo in FLUX sideways.
        try:
            return ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        except Exception:
            # The store holds video too now, and this runs in the request body —
            # outside `gen()`'s try — so an unreadable file here is a bare 500 rather
            # than an error in the stream. A generated video is a plausible thing to
            # drag back in, so say what's wrong instead of failing opaquely.
            raise HTTPException(
                status_code=400,
                detail=f"{path.suffix.lstrip('.') or 'that file'} isn't an image "
                       "that can be used as a source.",
            )

    init_image = _resolve(req.init_image_hash) if req.init_image_hash else None
    ref_images = [_resolve(h) for h in req.ref_image_hashes]

    def gen():
        events: queue.Queue = queue.Queue()

        def worker():
            try:
                events.put({"type": "status", "message": "Freeing VRAM (unloading vision model)…"})
                try:
                    oc.unload_all(req.ollama_url)
                except Exception:
                    pass  # best-effort; Ollama may be remote or already free

                status_cb = lambda m: events.put({"type": "status", "message": m})
                step_cb = lambda s, t: events.put({"type": "progress", "step": s, "total": t})

                # Resolve the transformer once, here, and pass the resolved name down
                # (resolving is idempotent). A request naming a model that can't serve
                # this mode still falls back rather than failing — a stale model list
                # is a legitimate reason to arrive here — but it says so instead of
                # quietly running something else.
                role = fx.role_for_mode(req.mode)
                unet = fx.resolve_unet(req.flux_model, role)
                if req.flux_model and unet != os.path.basename(str(req.flux_model)):
                    events.put({"type": "status", "message":
                                f"{os.path.basename(str(req.flux_model))} can't {req.mode} "
                                f"here — using {fx.label(unet)} instead"})

                seed = req.seed if req.seed is not None else random.randint(0, 2**31 - 1)
                common = dict(steps=req.steps, guidance=req.guidance, seed=seed,
                              model=unet, on_step=step_cb, on_status=status_cb)

                if req.mode == "animate":
                    if init_image is None:
                        raise ValueError("Animate needs a source image.")
                    # Returns bytes + a first frame, not a PIL image, so it doesn't
                    # join the shared tail below.
                    data, frame = fx.animate(init_image, req.prompt, seconds=req.seconds,
                                             **common)
                    h = db.save_image(
                        "data:video/webm;base64," + base64.b64encode(data).decode(),
                        images.pil_to_data_url(frame, max_size=64, fmt="JPEG") if frame else None,
                    )
                    events.put({
                        "type": "image", "kind": "video",
                        "hash": h, "url": db.image_url(h), "seed": seed,
                        "width": frame.size[0] if frame else 0,
                        "height": frame.size[1] if frame else 0,
                        "model": unet, "model_label": fx.label(unet),
                    })
                    return

                if req.mode == "edit":
                    if init_image is None:
                        raise ValueError("Edit needs a source image.")
                    # Extra images are references the instruction can draw subjects
                    # from; init_image stays the thing being edited.
                    image = fx.edit(init_image, req.prompt, refs=ref_images, **common)
                elif req.mode == "compose":
                    if not ref_images:
                        raise ValueError("Combine needs at least one reference image.")
                    image = fx.compose(ref_images, req.prompt, **common)
                elif req.mode == "img2img":
                    if init_image is None:
                        raise ValueError("Image-to-image needs a source image.")
                    image = fx.img2img(init_image, req.prompt, strength=req.strength,
                                       enhance=req.enhance, **common)
                else:  # txt2img
                    image = fx.create(req.prompt, width=req.width, height=req.height,
                                      enhance=req.enhance, **common)

                # PNG: a generated image is often the input to the next edit, and a
                # JPEG round-trip per generation compounds. Thumbs stay JPEG — 64px
                # of sidebar icon has nothing to preserve.
                full = images.pil_to_data_url(image)
                thumb = images.pil_to_data_url(image, max_size=64, fmt="JPEG")
                h = db.save_image(full, thumb)
                events.put(
                    {
                        "type": "image",
                        # Stated rather than left to default, so the client never has to
                        # read an absent field as "image" — `animate` sends "video" here
                        # and both go down the same terminal path.
                        "kind": "image",
                        "hash": h,
                        "url": db.image_url(h),
                        "seed": seed,
                        "width": image.size[0],
                        "height": image.size[1],
                        # What actually ran, so the UI never has to guess.
                        "model": unet,
                        "model_label": fx.label(unet),
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
        fx.free()
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Serve the built frontend (production). In dev, Vite serves on :5173.
# --------------------------------------------------------------------------- #
_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if _DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(_DIST), html=True), name="frontend")
