"""FastAPI backend for the Vision Model Chat UI.

Serves the built React frontend and proxies all Ollama interaction so the
browser never talks to Ollama directly.
"""
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import ollama_client as oc

app = FastAPI(title="Vision Model Chat")

# Allow the Vite dev server (5173) to call the API during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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
            yield from oc.stream_chat(req.ollama_url, req.model, messages)
        except Exception as exc:  # surfaced inline so the client can show it
            yield f"\n\n⚠️ {exc}"

    return StreamingResponse(gen(), media_type="text/plain; charset=utf-8")


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


@app.on_event("shutdown")
def _unload_on_shutdown():
    try:
        oc.unload_all(oc.DEFAULT_URL)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Serve the built frontend (production). In dev, Vite serves on :5173.
# --------------------------------------------------------------------------- #
_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if _DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(_DIST), html=True), name="frontend")
