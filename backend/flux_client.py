"""FLUX.1 Kontext image editing / composition via a local ComfyUI sidecar.

Parallels `sd_client.py` and `ollama_client.py`: the FastAPI backend owns the
model runtime and the browser never talks to it directly. FLUX Kontext is a 12B
model that can't run in this backend's torch 2.3 / diffusers 0.31 environment, so
it runs inside ComfyUI (its own venv, GGUF-quantized) which we drive over HTTP.

ComfyUI is started on demand (first edit/compose) and left resident. GPU is shared
with Ollama + the SD stack, so callers unload those first and can call `free()` to
release FLUX's VRAM when switching back to chat / create.

Everything is local: ComfyUI listens only on loopback and the weights were fetched
once (see the one-time setup); generation makes no outbound calls.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

# --------------------------------------------------------------------------- #
# Layout + model files (all non-gated, GGUF-quantized to fit the GPU)
# --------------------------------------------------------------------------- #
_RUNTIME = Path(__file__).resolve().parent.parent / "flux_runtime"
COMFY_DIR = _RUNTIME / "ComfyUI"
CVENV_PY = _RUNTIME / "cvenv" / "bin" / "python"
COMFY_URL = "http://127.0.0.1:8188"

UNET = "flux1-kontext-dev-Q4_K_S.gguf"
T5 = "t5-v1_1-xxl-encoder-Q4_K_M.gguf"
CLIP_L = "clip_l.safetensors"
VAE = "ae.safetensors"

# Quality-first defaults: 20 steps + Q4 quant is the sweet spot on this GPU;
# fewer steps / smaller quant visibly degrade output, so they aren't the knobs
# we turn for speed (keeping models resident is).
DEFAULT_STEPS = 20
DEFAULT_GUIDANCE = 2.5


def _guidance(v) -> float:
    """Kontext wants a low guidance (~2.5). Guard against a stray SD-scale value
    (e.g. 7.5) arriving from the shared settings and blowing out the edit."""
    try:
        g = float(v)
    except (TypeError, ValueError):
        return DEFAULT_GUIDANCE
    return g if 0.5 <= g <= 5.0 else DEFAULT_GUIDANCE


def _steps(v) -> int:
    try:
        s = int(v)
    except (TypeError, ValueError):
        return DEFAULT_STEPS
    return s if 8 <= s <= 40 else DEFAULT_STEPS

_proc: subprocess.Popen | None = None  # the ComfyUI child, if we started it


# --------------------------------------------------------------------------- #
# Availability / server lifecycle
# --------------------------------------------------------------------------- #
def available() -> bool:
    """True if the FLUX runtime is installed (venv + all weights present)."""
    if not CVENV_PY.exists():
        return False
    need = [
        COMFY_DIR / "models" / "unet" / UNET,
        COMFY_DIR / "models" / "clip" / T5,
        COMFY_DIR / "models" / "clip" / CLIP_L,
        COMFY_DIR / "models" / "vae" / VAE,
    ]
    return all(p.exists() for p in need)


def _server_up() -> bool:
    try:
        with urllib.request.urlopen(COMFY_URL + "/system_stats", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def ensure_server(on_status=None) -> None:
    """Start the ComfyUI sidecar if it isn't already listening, and wait for it."""
    global _proc
    if _server_up():
        return
    if not available():
        raise RuntimeError("FLUX runtime isn't installed (missing venv or weights).")
    if on_status:
        on_status("starting FLUX engine…")
    # Default (normalvram) memory management: ComfyUI keeps the model resident and
    # offloads only as needed — the right balance on this 15 GB GPU. (--lowvram
    # would over-offload and slow things down for no quality gain.) Loopback-only
    # for privacy; logs kept for troubleshooting.
    log = open(_RUNTIME / "comfyui.log", "w")  # noqa: SIM115
    # Once installed the sidecar must never reach out: the GGUF loaders read local
    # weight files, and these flags stop transformers / HF-hub from making metadata
    # calls to huggingface.co behind our back. (The backend process already sets
    # these, but pin them on the child explicitly so it holds regardless.)
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "HF_HUB_DISABLE_TELEMETRY": "1"}
    _proc = subprocess.Popen(
        [str(CVENV_PY), "main.py", "--listen", "127.0.0.1", "--port", "8188"],
        cwd=str(COMFY_DIR),
        stdout=log,
        stderr=subprocess.STDOUT,
        env=env,
    )
    for _ in range(120):  # up to ~60s for boot
        if _server_up():
            return
        time.sleep(0.5)
    raise RuntimeError("FLUX engine failed to start in time.")


def free() -> None:
    """Release FLUX's VRAM (call before handing the GPU back to chat / SD)."""
    if not _server_up():
        return
    try:
        _post("/free", {"unload_models": True, "free_memory": True})
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# HTTP helpers
# --------------------------------------------------------------------------- #
def _post(path: str, data: dict) -> dict:
    req = urllib.request.Request(
        COMFY_URL + path,
        data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as r:
        return json.load(r)


def _get(path: str) -> dict:
    with urllib.request.urlopen(COMFY_URL + path) as r:
        return json.load(r)


def _upload_image(pil, name: str) -> str:
    """Upload a PIL image to ComfyUI's input store; return its filename."""
    buf = io.BytesIO()
    pil.convert("RGB").save(buf, format="PNG")
    body, boundary = _multipart(name, buf.getvalue())
    req = urllib.request.Request(
        COMFY_URL + "/upload/image",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req) as r:
        return json.load(r)["name"]


def _multipart(filename: str, content: bytes):
    boundary = uuid.uuid4().hex
    pre = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
        "Content-Type: image/png\r\n\r\n"
    ).encode()
    post = (
        f"\r\n--{boundary}\r\n"
        'Content-Disposition: form-data; name="overwrite"\r\n\r\ntrue\r\n'
        f"--{boundary}--\r\n"
    ).encode()
    return pre + content + post, boundary


def _fetch_output(filename: str, subfolder: str):
    from PIL import Image  # local import; Pillow is a backend dep

    q = urllib.parse.urlencode({"filename": filename, "subfolder": subfolder, "type": "output"})
    with urllib.request.urlopen(COMFY_URL + "/view?" + q) as r:
        return Image.open(io.BytesIO(r.read())).convert("RGB")


# --------------------------------------------------------------------------- #
# Workflow graphs
# --------------------------------------------------------------------------- #
def _loaders() -> dict:
    return {
        "unet": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": UNET}},
        "clip": {"class_type": "DualCLIPLoaderGGUF",
                 "inputs": {"clip_name1": CLIP_L, "clip_name2": T5, "type": "flux"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
    }


def _edit_graph(image_name, prompt, steps, guidance, seed, prefix):
    g = _loaders()
    g.update({
        "img": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "scale": {"class_type": "FluxKontextImageScale", "inputs": {"image": ["img", 0]}},
        "enc": {"class_type": "VAEEncode", "inputs": {"pixels": ["scale", 0], "vae": ["vae", 0]}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["clip", 0]}},
        "ref": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["pos", 0], "latent": ["enc", 0]}},
        "guide": {"class_type": "FluxGuidance", "inputs": {"conditioning": ["ref", 0], "guidance": guidance}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["clip", 0]}},
        "sampler": {"class_type": "KSampler",
                    "inputs": {"model": ["unet", 0], "positive": ["guide", 0], "negative": ["neg", 0],
                               "latent_image": ["enc", 0], "seed": seed, "steps": steps, "cfg": 1.0,
                               "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "dec": {"class_type": "VAEDecode", "inputs": {"samples": ["sampler", 0], "vae": ["vae", 0]}},
        "save": {"class_type": "SaveImage", "inputs": {"images": ["dec", 0], "filename_prefix": prefix}},
    })
    return g


def _compose_graph(image_names, prompt, steps, guidance, seed, prefix):
    """Multi-image: stitch the inputs side-by-side into one canvas, then edit that
    single image. FLUX Kontext *dev* is a single-image model — stacking a
    ReferenceLatent per input just makes it echo the first image and ignore the
    rest. ImageStitch puts every subject in one latent the model can actually fuse
    (the community-standard multi-image recipe for Kontext dev)."""
    g = _loaders()
    # Load each input and join them left-to-right into a single image.
    g["img0"] = {"class_type": "LoadImage", "inputs": {"image": image_names[0]}}
    stitched = ("img0", 0)
    for i, name in enumerate(image_names[1:], start=1):
        g[f"img{i}"] = {"class_type": "LoadImage", "inputs": {"image": name}}
        g[f"stitch{i}"] = {"class_type": "ImageStitch",
                           "inputs": {"image1": list(stitched), "image2": [f"img{i}", 0],
                                      "direction": "right", "match_image_size": True,
                                      "spacing_width": 0, "spacing_color": "white"}}
        stitched = (f"stitch{i}", 0)
    # Scale the combined canvas to a Kontext-friendly resolution, then edit it.
    g["scale"] = {"class_type": "FluxKontextImageScale", "inputs": {"image": list(stitched)}}
    g["enc"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["scale", 0], "vae": ["vae", 0]}}
    g["pos"] = {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["clip", 0]}}
    g["ref"] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["pos", 0], "latent": ["enc", 0]}}
    g["guide"] = {"class_type": "FluxGuidance", "inputs": {"conditioning": ["ref", 0], "guidance": guidance}}
    g["neg"] = {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["clip", 0]}}
    g["sampler"] = {"class_type": "KSampler",
                    "inputs": {"model": ["unet", 0], "positive": ["guide", 0], "negative": ["neg", 0],
                               "latent_image": ["enc", 0], "seed": seed, "steps": steps, "cfg": 1.0,
                               "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}}
    g["dec"] = {"class_type": "VAEDecode", "inputs": {"samples": ["sampler", 0], "vae": ["vae", 0]}}
    g["save"] = {"class_type": "SaveImage", "inputs": {"images": ["dec", 0], "filename_prefix": prefix}}
    return g


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
def _run(graph, on_step=None, on_status=None):
    """Submit a graph, relay step progress via the ComfyUI websocket, return PIL."""
    client_id = uuid.uuid4().hex
    pid = _post("/prompt", {"prompt": graph, "client_id": client_id})["prompt_id"]

    # Progress via websocket if the client lib is available; otherwise just wait.
    try:
        import websocket  # noqa: PLC0415  (websocket-client; optional)

        ws = websocket.create_connection(
            COMFY_URL.replace("http", "ws") + "/ws?clientId=" + client_id, timeout=5
        )
        ws.settimeout(600)
        while True:
            msg = ws.recv()
            if not isinstance(msg, str):
                continue
            ev = json.loads(msg)
            if ev.get("type") == "progress":
                d = ev["data"]
                if on_step:
                    on_step(d.get("value", 0), d.get("max", DEFAULT_STEPS))
            elif ev.get("type") == "executing" and ev["data"].get("node") is None \
                    and ev["data"].get("prompt_id") == pid:
                break
        ws.close()
    except Exception:
        # No websocket lib / connection: poll history until the prompt completes.
        t0 = time.time()
        while time.time() - t0 < 900:
            h = _get("/history/" + pid)
            if h.get(pid, {}).get("outputs"):
                break
            time.sleep(2)

    hist = _get("/history/" + pid).get(pid, {})
    for node in hist.get("outputs", {}).values():
        for im in node.get("images", []):
            return _fetch_output(im["filename"], im.get("subfolder", ""))
    status = hist.get("status", {})
    raise RuntimeError(f"FLUX generation produced no image ({status.get('status_str', 'unknown')}).")


def edit(pil, prompt, steps=None, guidance=None, seed=0, on_step=None, on_status=None):
    """Instruction-edit a single image ('make the cat eat the cauliflower')."""
    ensure_server(on_status=on_status)
    name = _upload_image(pil, f"edit_{uuid.uuid4().hex}.png")
    g = _edit_graph(name, prompt or "", _steps(steps), _guidance(guidance), int(seed), "flux_edit")
    if on_status:
        on_status("editing with FLUX Kontext…")
    return _run(g, on_step=on_step, on_status=on_status)


def compose(pils, prompt, steps=None, guidance=None, seed=0, on_step=None, on_status=None):
    """Combine multiple reference images into one new image."""
    ensure_server(on_status=on_status)
    if not pils:
        raise ValueError("compose requires at least one reference image")
    names = [_upload_image(p, f"ref{i}_{uuid.uuid4().hex}.png") for i, p in enumerate(pils)]
    g = _compose_graph(names, prompt or "", _steps(steps), _guidance(guidance), int(seed), "flux_compose")
    if on_status:
        on_status("composing with FLUX Kontext…")
    return _run(g, on_step=on_step, on_status=on_status)
