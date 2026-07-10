"""All image generation — create, edit, compose — via a local ComfyUI sidecar.

Parallels `ollama_client.py`: the FastAPI backend owns the model runtime and the
browser never talks to it directly. FLUX is a 12B model that needs torch >= 2.4,
so it runs inside ComfyUI (its own venv, GGUF-quantized) which we drive over HTTP.
The backend process itself holds no torch at all.

Two UNets share the runtime, because the architectures differ in what they
condition on:
  - **FLUX.1-dev** for `create` (txt2img / img2img). Pure text conditioning.
  - **FLUX.1 Kontext** for `edit` / `compose`. Additionally takes a `ReferenceLatent`
    of the source image, which is what makes it preserve identity while editing.
Using one for the other's job produces bad output, so `_resolve_unet` keys on role.

ComfyUI is started on demand and left resident. The GPU is shared with Ollama, so
callers unload it first and can call `free()` to release FLUX's VRAM. Only one
UNet fits alongside the text encoders at a time; ComfyUI evicts as needed when a
graph names a different one.

Everything is local: ComfyUI listens only on loopback and the weights were fetched
once (see the one-time setup); generation makes no outbound calls.
"""
from __future__ import annotations

import io
import json
import os
import re
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

# The UNets are swappable: the two defaults below ship with the runtime and can't
# be removed, but users can add extra UNets from any HuggingFace repo (see
# `pull_unet`) and pick which one a mode runs on. The text encoders + VAE are
# shared across all of them.
#
# Q8_0 is the quality tier this 24 GB card is sized for: ~12.6 GB of UNet plus
# activations at 1024x1024 peaks around 16 GB. (The T5 encoder is only resident
# while conditioning runs; ComfyUI offloads it before sampling.)
DEFAULT_CREATE_UNET = "flux1-dev-Q8_0.gguf"  # txt2img / img2img
DEFAULT_KONTEXT_UNET = "flux1-kontext-dev-Q8_0.gguf"  # edit / compose
DEFAULT_UNETS = (DEFAULT_CREATE_UNET, DEFAULT_KONTEXT_UNET)

UNET_DIR = COMFY_DIR / "models" / "unet"
# Loadable UNet formats: GGUF (via the GGUF node) and plain diffusion checkpoints
# (via ComfyUI's built-in UNETLoader). Users can add either.
UNET_EXTS = (".gguf", ".safetensors", ".sft")
T5 = "t5-v1_1-xxl-encoder-Q8_0.gguf"
CLIP_L = "clip_l.safetensors"
VAE = "ae.safetensors"

# Roles. A Kontext UNet takes a ReferenceLatent of the source image; a plain dev
# UNet doesn't. Filename is the only signal we have for a user-added model.
ROLE_CREATE = "create"
ROLE_EDIT = "edit"


def _role_of(unet: str) -> str:
    return ROLE_EDIT if "kontext" in unet.lower() else ROLE_CREATE


# Quality-first defaults: 20 steps at Q8 is the sweet spot on this GPU; fewer
# steps visibly degrades output, so that isn't the knob we turn for speed
# (keeping models resident is).
DEFAULT_STEPS = 20
# Guidance is mode-specific. Kontext follows an instruction at ~2.5; dev needs a
# higher ~3.5 to bind a text-only prompt. Feeding either the other's value (or a
# stray SD-scale 7.5 from shared settings) blows out the image.
#
# Do not raise KONTEXT_GUIDANCE to "make edits stronger" — it does the opposite.
# At 3.5 the model clings to the reference image and silently ignores the
# instruction, returning the source unchanged (measured on a two-reference edit).
KONTEXT_GUIDANCE = 2.5
CREATE_GUIDANCE = 3.5
GUIDANCE_MIN, GUIDANCE_MAX = 0.5, 10.0

# How the model lays out multiple reference images. See `_conditioning`.
REF_METHOD = "offset"

# FLUX responds to natural photographic language, not SD 1.5's comma-separated
# quality tags — a tag salad actively hurts it. Applied when the caller opts in.
#
# Deliberately says nothing about lighting, depth of field, or framing: those are
# the user's to set, and baking in "natural lighting" would fight a prompt like
# "a neon-lit alley at night". Only medium and surface realism are asserted.
PHOTOREAL_TEMPLATE = (
    "A photorealistic, high-resolution photograph. {prompt}. Shot on a full-frame "
    "DSLR, realistic skin texture and pores, fine detail, sharp focus."
)


def _guidance(v, default: float) -> float:
    try:
        g = float(v)
    except (TypeError, ValueError):
        return default
    return g if GUIDANCE_MIN <= g <= GUIDANCE_MAX else default


def _steps(v) -> int:
    try:
        s = int(v)
    except (TypeError, ValueError):
        return DEFAULT_STEPS
    return s if 8 <= s <= 40 else DEFAULT_STEPS


def _strength(v) -> float:
    """img2img denoise: 0 = return the input, 1 = ignore it."""
    try:
        s = float(v)
    except (TypeError, ValueError):
        return 0.6
    return min(max(s, 0.05), 1.0)


def _dim(v, default: int = 1024) -> int:
    """FLUX is trained at ~1 megapixel. Snap to the multiple of 16 the VAE needs."""
    try:
        d = int(v)
    except (TypeError, ValueError):
        return default
    d = min(max(d, 256), 1536)
    return d - (d % 16)


# The ~1 MP shapes Kontext was trained on. Mirrors ComfyUI's
# PREFERRED_KONTEXT_RESOLUTIONS (comfy_extras/nodes_flux.py); we need them in
# Python because compose sizes an EmptySD3LatentImage rather than snapping a real
# image through FluxKontextImageScale.
PREFERRED_KONTEXT_RESOLUTIONS = [
    (672, 1568), (688, 1504), (720, 1456), (752, 1392), (800, 1328), (832, 1248),
    (880, 1184), (944, 1104), (1024, 1024), (1104, 944), (1184, 880), (1248, 832),
    (1328, 800), (1392, 752), (1456, 720), (1504, 688), (1568, 672),
]


def _kontext_resolution(pil) -> tuple[int, int]:
    """Nearest Kontext resolution by aspect ratio — the same rule the scale node uses."""
    w, h = pil.size
    aspect = w / h if h else 1.0
    _, bw, bh = min((abs(aspect - rw / rh), rw, rh) for rw, rh in PREFERRED_KONTEXT_RESOLUTIONS)
    return bw, bh

_proc: subprocess.Popen | None = None  # the ComfyUI child, if we started it


# --------------------------------------------------------------------------- #
# Availability / server lifecycle
# --------------------------------------------------------------------------- #
def available() -> bool:
    """True if the FLUX runtime is installed (venv + all weights present)."""
    if not CVENV_PY.exists():
        return False
    need = [
        *(UNET_DIR / u for u in DEFAULT_UNETS),
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
    """Release FLUX's VRAM (call before handing the GPU back to chat)."""
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
def _unet_node(unet: str) -> dict:
    """Pick the loader for the UNet's format: GGUF quantized vs. a plain
    safetensors diffusion checkpoint (ComfyUI's built-in UNETLoader)."""
    if unet.lower().endswith(".gguf"):
        return {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": unet}}
    return {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}}


def _loaders(unet: str) -> dict:
    return {
        "unet": _unet_node(unet),
        "clip": {"class_type": "DualCLIPLoaderGGUF",
                 "inputs": {"clip_name1": CLIP_L, "clip_name2": T5, "type": "flux"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
    }


def _sampler(latent_src, steps, seed, denoise=1.0) -> dict:
    """FLUX samples at cfg=1.0 — the negative branch is unused, which is why this
    app exposes no negative prompt. Guidance is carried by FluxGuidance instead."""
    return {"class_type": "KSampler",
            "inputs": {"model": ["unet", 0], "positive": ["guide", 0], "negative": ["neg", 0],
                       "latent_image": list(latent_src), "seed": seed, "steps": steps, "cfg": 1.0,
                       "sampler_name": "euler", "scheduler": "simple", "denoise": denoise}}


def _conditioning(prompt: str, guidance: float, ref_latents=()) -> dict:
    """Text conditioning shared by every graph. `guide` is what the sampler reads.

    `ref_latents` chains one Kontext `ReferenceLatent` node per encoded reference
    image. The node *appends* to the conditioning, and `Flux.extra_conds` forwards
    the whole list, so the model sees each reference as its own token block with its
    own RoPE offsets — the images stay distinct. Only Kontext UNets understand this;
    pass nothing for plain dev (create) graphs.
    """
    g = {
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["clip", 0]}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["clip", 0]}},
    }
    src = "pos"
    for i, lat in enumerate(ref_latents):
        node = f"ref{i}"
        g[node] = {"class_type": "ReferenceLatent",
                   "inputs": {"conditioning": [src, 0], "latent": list(lat)}}
        src = node
    # With several references the model needs to know how to lay them out. "offset"
    # (also ComfyUI's default) packs each into its own region of a shared coordinate
    # frame and preserved composition best in testing; "index" gives each its own
    # RoPE index but makes the last subject dominate the frame. Pin it explicitly so
    # a ComfyUI upgrade can't silently change the default under us.
    if len(ref_latents) > 1:
        g["refmethod"] = {"class_type": "FluxKontextMultiReferenceLatentMethod",
                          "inputs": {"conditioning": [src, 0], "reference_latents_method": REF_METHOD}}
        src = "refmethod"
    g["guide"] = {"class_type": "FluxGuidance",
                  "inputs": {"conditioning": [src, 0], "guidance": guidance}}
    return g


def _tail(prefix: str) -> dict:
    return {
        "dec": {"class_type": "VAEDecode", "inputs": {"samples": ["sampler", 0], "vae": ["vae", 0]}},
        "save": {"class_type": "SaveImage", "inputs": {"images": ["dec", 0], "filename_prefix": prefix}},
    }


def _txt2img_graph(prompt, width, height, steps, guidance, seed, prefix, unet):
    """Text-to-image on a plain FLUX dev UNet. No ReferenceLatent — that node is
    Kontext's identity-preserving path and would pin the output to a source image.

    EmptySD3LatentImage (not EmptyLatentImage): FLUX's VAE is 16-channel, and the
    4-channel SD latent would decode to noise.
    """
    g = _loaders(unet)
    g.update(_conditioning(prompt, guidance))
    g["latent"] = {"class_type": "EmptySD3LatentImage",
                   "inputs": {"width": width, "height": height, "batch_size": 1}}
    g["sampler"] = _sampler(("latent", 0), steps, seed)
    g.update(_tail(prefix))
    return g


def _img2img_graph(image_name, prompt, strength, steps, guidance, seed, prefix, unet):
    """Image-to-image: encode the input and partially denoise it. `strength` is the
    denoise fraction — how far the result may drift from the attached image."""
    g = _loaders(unet)
    g.update(_conditioning(prompt, guidance))
    g["img"] = {"class_type": "LoadImage", "inputs": {"image": image_name}}
    g["scale"] = {"class_type": "FluxKontextImageScale", "inputs": {"image": ["img", 0]}}
    g["enc"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["scale", 0], "vae": ["vae", 0]}}
    g["sampler"] = _sampler(("enc", 0), steps, seed, denoise=strength)
    g.update(_tail(prefix))
    return g


def _encode_image(g, name, key):
    """LoadImage -> snap to a Kontext-native resolution -> VAEEncode. Returns the
    latent output ref. Every reference goes through the resolution snap: it keeps
    the token count (and VRAM) bounded and matches how Kontext was trained."""
    g[f"{key}_img"] = {"class_type": "LoadImage", "inputs": {"image": name}}
    g[f"{key}_scale"] = {"class_type": "FluxKontextImageScale", "inputs": {"image": [f"{key}_img", 0]}}
    g[key] = {"class_type": "VAEEncode", "inputs": {"pixels": [f"{key}_scale", 0], "vae": ["vae", 0]}}
    return (key, 0)


def _edit_graph(scene_name, ref_names, prompt, steps, guidance, seed, prefix, unet):
    """Instruction-edit `scene_name` on a Kontext UNet.

    The scene is encoded once and used twice: as the latent being denoised, and as
    the first ReferenceLatent (which is what preserves its identity and background).
    Any `ref_names` are chained on as additional references — that's how a subject
    from another photo gets carried into this one.
    """
    g = _loaders(unet)
    scene = _encode_image(g, scene_name, "enc")
    refs = [scene] + [_encode_image(g, n, f"src{i}") for i, n in enumerate(ref_names)]
    g.update(_conditioning(prompt, guidance, ref_latents=refs))
    g["sampler"] = _sampler(scene, steps, seed)
    g.update(_tail(prefix))
    return g


def _compose_graph(image_names, prompt, width, height, steps, guidance, seed, prefix, unet):
    """Multi-image: build a new scene from every input, each kept as its own
    reference image.

    Earlier versions stitched the inputs side-by-side into one canvas. That is
    strictly worse: it fuses everything into one coordinate frame, downscales each
    subject, and leaves the model free to treat the result as a diptych (it did —
    it edited only the left half). Chaining a ReferenceLatent per input keeps each
    image in its own token block with its own position offsets, which is what
    actually transfers a subject between photos.
    """
    g = _loaders(unet)
    refs = [_encode_image(g, n, f"src{i}") for i, n in enumerate(image_names)]
    g.update(_conditioning(prompt, guidance, ref_latents=refs))
    g["latent"] = {"class_type": "EmptySD3LatentImage",
                   "inputs": {"width": width, "height": height, "batch_size": 1}}
    g["sampler"] = _sampler(("latent", 0), steps, seed)
    g.update(_tail(prefix))
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


def _prompt_for(prompt: str, enhance: bool) -> str:
    p = (prompt or "").strip()
    return PHOTOREAL_TEMPLATE.format(prompt=p.rstrip(".")) if enhance and p else p


def create(prompt, width=None, height=None, steps=None, guidance=None, seed=0,
           model=None, enhance=True, on_step=None, on_status=None):
    """Text-to-image on FLUX dev ('a candid photo of a woman laughing')."""
    ensure_server(on_status=on_status)
    unet = _resolve_unet(model, ROLE_CREATE)
    g = _txt2img_graph(_prompt_for(prompt, enhance), _dim(width), _dim(height), _steps(steps),
                       _guidance(guidance, CREATE_GUIDANCE), int(seed), "flux_create", unet)
    if on_status:
        on_status("generating with FLUX…")
    return _run(g, on_step=on_step, on_status=on_status)


def img2img(pil, prompt, strength=None, steps=None, guidance=None, seed=0,
            model=None, enhance=True, on_step=None, on_status=None):
    """Transform an attached image on FLUX dev, keeping its composition."""
    ensure_server(on_status=on_status)
    unet = _resolve_unet(model, ROLE_CREATE)
    name = _upload_image(pil, f"init_{uuid.uuid4().hex}.png")
    g = _img2img_graph(name, _prompt_for(prompt, enhance), _strength(strength), _steps(steps),
                       _guidance(guidance, CREATE_GUIDANCE), int(seed), "flux_img2img", unet)
    if on_status:
        on_status("transforming with FLUX…")
    return _run(g, on_step=on_step, on_status=on_status)


def edit(pil, prompt, refs=(), steps=None, guidance=None, seed=0, model=None,
         on_step=None, on_status=None):
    """Instruction-edit an image ('make the cat eat the cauliflower').

    `refs` are optional extra images the instruction may draw subjects from, e.g.
    'add the man from the reference photo'. `pil` is always the image being edited:
    its composition and background are what survive.
    """
    ensure_server(on_status=on_status)
    unet = _resolve_unet(model, ROLE_EDIT)
    tag = uuid.uuid4().hex
    scene = _upload_image(pil, f"edit_{tag}.png")
    ref_names = [_upload_image(p, f"editref{i}_{tag}.png") for i, p in enumerate(refs)]
    g = _edit_graph(scene, ref_names, prompt or "", _steps(steps),
                    _guidance(guidance, KONTEXT_GUIDANCE), int(seed), "flux_edit", unet)
    if on_status:
        on_status("editing with FLUX Kontext…")
    return _run(g, on_step=on_step, on_status=on_status)


def compose(pils, prompt, steps=None, guidance=None, seed=0, model=None, on_step=None, on_status=None):
    """Combine multiple reference images into one new image."""
    ensure_server(on_status=on_status)
    if not pils:
        raise ValueError("compose requires at least one reference image")
    unet = _resolve_unet(model, ROLE_EDIT)
    tag = uuid.uuid4().hex
    names = [_upload_image(p, f"ref{i}_{tag}.png") for i, p in enumerate(pils)]
    # The new scene takes its shape from the first reference.
    width, height = _kontext_resolution(pils[0])
    g = _compose_graph(names, prompt or "", width, height, _steps(steps),
                       _guidance(guidance, KONTEXT_GUIDANCE), int(seed), "flux_compose", unet)
    if on_status:
        on_status("composing with FLUX Kontext…")
    return _run(g, on_step=on_step, on_status=on_status)


# --------------------------------------------------------------------------- #
# Model management: list / add (from any HF repo) / remove extra UNets
# --------------------------------------------------------------------------- #
def list_unets() -> list[dict]:
    """Installed FLUX UNets, defaults first. Each: name, role, default flag, size_gb.

    `role` tells the UI which mode a model can serve: create models appear in the
    Create dropdown, edit models in Edit/Combine.
    """
    out = []
    if UNET_DIR.is_dir():
        for p in sorted(UNET_DIR.iterdir()):
            if p.is_file() and p.suffix.lower() in UNET_EXTS:
                out.append({
                    "name": p.name,
                    "role": _role_of(p.name),
                    "default": p.name in DEFAULT_UNETS,
                    "size_gb": round(p.stat().st_size / 1e9, 2),
                })
    out.sort(key=lambda m: (not m["default"], m["name"].lower()))
    return out


def _default_for(role: str) -> str:
    return DEFAULT_KONTEXT_UNET if role == ROLE_EDIT else DEFAULT_CREATE_UNET


def _resolve_unet(model, role: str) -> str:
    """Map a requested model name to an installed UNet file, guarding against path
    traversal. Unknown, empty, or wrong-role → that role's default.

    The role check matters: a Kontext graph feeds a ReferenceLatent that a plain dev
    UNet ignores, and a dev UNet asked to edit would just regenerate from scratch.
    """
    if not model:
        return _default_for(role)
    safe = os.path.basename(str(model))
    if (safe.lower().endswith(UNET_EXTS)
            and (UNET_DIR / safe).exists()
            and _role_of(safe) == role):
        return safe
    return _default_for(role)


def delete_unet(name: str) -> None:
    """Remove an extra UNet. The bundled defaults can never be deleted."""
    safe = os.path.basename(name or "")
    if not safe.lower().endswith(UNET_EXTS):
        raise ValueError("Not a model file.")
    if safe in DEFAULT_UNETS:
        raise ValueError("The default FLUX models can't be removed.")
    p = UNET_DIR / safe
    if not p.exists():
        raise FileNotFoundError(safe)
    p.unlink()


_REPO_RE = re.compile(r"^[\w.-]+/[\w.-]+$")


def _parse_repo(spec: str) -> tuple[str, str]:
    """Parse a paste into (repo_id, filename). Accepts:
      owner/repo                      -> auto-pick a UNet from the repo
      owner/repo:file.safetensors     -> that exact file (any supported format)
      owner/repo/sub/dir/file.gguf    -> that exact file (path after the repo id)
    """
    spec = (spec or "").strip()
    filename = ""
    if ":" in spec:
        spec, filename = spec.split(":", 1)
        spec, filename = spec.strip(), filename.strip()
    elif spec.lower().endswith(UNET_EXTS):
        parts = spec.split("/")
        if len(parts) > 2:
            spec, filename = "/".join(parts[:2]), "/".join(parts[2:])
    if not _REPO_RE.match(spec):
        raise ValueError(f"'{spec}' isn't a valid HuggingFace repo (expected owner/name).")
    if filename and not filename.lower().endswith(UNET_EXTS):
        raise ValueError("Unsupported file — use a .gguf or .safetensors UNet.")
    return spec, filename


# Runs in a child process (network allowed) so the serving process stays offline.
# argv: repo, filename ("" = auto-pick a UNet), dest_dir. Downloads straight into
# the UNet dir (no HF-cache copy) and streams coarse progress on stdout.
_DOWNLOAD_UNET = r"""
import os, sys, time, urllib.request
from urllib.error import HTTPError, URLError
EXTS = (".gguf", ".safetensors", ".sft")
repo, filename, dest_dir = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    if not filename:
        from huggingface_hub import HfApi
        files = [f for f in HfApi().list_repo_files(repo) if f.lower().endswith(EXTS)]
        if not files:
            print("ERROR: no .gguf or .safetensors UNet files found in " + repo, flush=True)
            sys.exit(1)
        ggufs = [f for f in files if f.lower().endswith(".gguf")]
        # Prefer a GGUF (fits this GPU); pick the highest quant this 24 GB card can
        # hold. Otherwise fall back to a top-level safetensors, skipping obvious
        # multi-file shards.
        others = [f for f in files if not f.lower().endswith(".gguf")
                  and "/" not in f and "-of-" not in f.lower()] or \
                 [f for f in files if not f.lower().endswith(".gguf")]
        filename = (next((f for f in ggufs if "Q8_0" in f), None)
                    or next((f for f in ggufs if "Q6_K" in f), None)
                    or next((f for f in ggufs if "Q4_K" in f), None)
                    or (ggufs[0] if ggufs else None)
                    or others[0])
    base = os.path.basename(filename)
    out = os.path.join(dest_dir, base)
    if os.path.exists(out):
        print("ERROR: a model named " + base + " already exists.", flush=True); sys.exit(1)
    url = "https://huggingface.co/%s/resolve/main/%s?download=true" % (repo, filename)
    print("Downloading %s from %s…" % (base, repo), flush=True)
    tmp = out + ".part"
    with urllib.request.urlopen(url) as r:
        total = int(r.headers.get("Content-Length", 0))
        done = last = 0
        with open(tmp, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk); done += len(chunk)
                if time.time() - last > 0.5:
                    if total:
                        print("Downloading %s: %.2f/%.2f GB (%d%%)"
                              % (base, done/1e9, total/1e9, 100*done//total), flush=True)
                    else:
                        print("Downloading %s: %.2f GB" % (base, done/1e9), flush=True)
                    last = time.time()
    os.replace(tmp, out)
    print("DONE " + base, flush=True)
except (HTTPError, URLError) as e:
    print("ERROR: could not fetch from %s (%s)" % (repo, e), flush=True); sys.exit(1)
"""


def pull_unet(repo: str, on_status=None) -> None:
    """Download an extra FLUX UNet from a HuggingFace repo (opt-in, streams status).

    The fetch runs in a subprocess with the offline flags dropped so the serving
    process itself never gains network access.
    """
    import sys  # PLC0415

    repo_id, filename = _parse_repo(repo)
    if not available():
        raise RuntimeError("FLUX runtime isn't installed.")
    UNET_DIR.mkdir(parents=True, exist_ok=True)
    if on_status:
        on_status(f"Resolving {repo_id}…")

    env = {**os.environ}
    env.pop("HF_HUB_OFFLINE", None)
    env.pop("TRANSFORMERS_OFFLINE", None)
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"

    proc = subprocess.Popen(
        [sys.executable, "-c", _DOWNLOAD_UNET, repo_id, filename, str(UNET_DIR)],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0,
    )
    buf = b""
    last = 0.0
    err = None
    while True:
        ch = proc.stdout.read(1)
        if not ch:
            break
        if ch in (b"\r", b"\n"):
            line = buf.decode("utf-8", "replace").strip()
            buf = b""
            if line.startswith("ERROR:"):
                err = line[6:].strip()
            elif line and on_status and (time.time() - last) > 0.4:
                on_status(line)
                last = time.time()
        else:
            buf += ch
    if proc.wait() != 0 or err:
        raise RuntimeError(err or "Download failed. Check the repo id and your connection.")
    if on_status:
        on_status("Download complete.")
