"""All image generation — create, edit, compose — via a local ComfyUI sidecar.

Parallels `ollama_client.py`: the FastAPI backend owns the model runtime and the
browser never talks to it directly. FLUX needs torch >= 2.4, so it runs inside
ComfyUI (its own venv) which we drive over HTTP. The backend process itself holds
no torch at all.

No weights ship with the runtime — the user installs a *bundle* from the UI (see
`flux_catalog`). A bundle names a **family**, and the family decides the graph:

  - **flux2** — FLUX.2 [dev]. One 32B transformer serves every mode: text
    conditioning through a Mistral-3 encoder, plus optional `ReferenceLatent`s for
    edit/compose. Samples through `SamplerCustomAdvanced` with no negative branch.
  - **flux1** — FLUX.1, two transformers split by role: dev for `create` (pure text
    conditioning), Kontext for `edit`/`compose` (additionally takes a
    `ReferenceLatent` of the source image, which is what preserves identity). Using
    one for the other's job produces bad output, so `_resolve_unet` keys on role.

ComfyUI is started on demand and left resident. The GPU is shared with Ollama, so
callers unload it first and can call `free()` to release FLUX's VRAM. Only one
transformer fits alongside the text encoder at a time; ComfyUI evicts as needed
when a graph names a different one.

Generation is entirely local: ComfyUI listens only on loopback and never reaches
the network (the offline flags in `ensure_server`). The one exception is the model
installer, which runs in a separate child process — see `_HF_CHILD`.
"""
from __future__ import annotations

import io
import json
import math
import os
import re
import shutil
import struct
import subprocess
import threading
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

from . import flux_catalog as cat
from . import settings

# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
_RUNTIME = Path(__file__).resolve().parent.parent / "flux_runtime"
COMFY_DIR = cat.COMFY_DIR
CVENV_PY = _RUNTIME / "cvenv" / "bin" / "python"
COMFY_URL = "http://127.0.0.1:8188"

UNET_DIR = cat.UNET_DIR
# Loadable transformer formats: GGUF (via the GGUF node) and plain diffusion
# checkpoints (via ComfyUI's built-in UNETLoader). Users can add either.
UNET_EXTS = (".gguf", ".safetensors", ".sft")

# FLUX.1's shared encoders + VAE. FLUX.2 brings its own, named by its bundle.
T5 = "t5-v1_1-xxl-encoder-Q8_0.gguf"
CLIP_L = "clip_l.safetensors"
FLUX1_VAE = "ae.safetensors"

ROLE_CREATE = cat.ROLE_CREATE
ROLE_EDIT = cat.ROLE_EDIT


# Quality-first defaults: 20 steps at Q8 is the sweet spot on this GPU; fewer
# steps visibly degrades output, so that isn't the knob we turn for speed
# (keeping models resident is).
DEFAULT_STEPS = 20
# Guidance is mode- and family-specific. On FLUX.1, Kontext follows an instruction at
# ~2.5 while dev needs a higher ~3.5 to bind a text-only prompt; feeding either the
# other's value (or a stray SD-scale 7.5 from shared settings) blows out the image.
#
# Do not raise KONTEXT_GUIDANCE to "make edits stronger" — it does the opposite.
# At 3.5 the model clings to the reference image and silently ignores the
# instruction, returning the source unchanged (measured on a two-reference edit).
#
# FLUX.2 uses one value for both jobs (ComfyUI's own template ships 4.0).
KONTEXT_GUIDANCE = 2.5
CREATE_GUIDANCE = 3.5
FLUX2_GUIDANCE = 4.0
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


def _flux2_resolution(pil) -> tuple[int, int]:
    """The shape ImageScaleToTotalPixels(area, 1 MP, steps=16) will produce. Mirrored
    here because Flux2Scheduler needs the sampled resolution up front: its sigma
    schedule is shifted by sequence length, so feeding it the *unscaled* size would
    shift the whole schedule wrong."""
    w, h = pil.size
    scale = math.sqrt((1024 * 1024) / float(w * h or 1))
    return max(round(w * scale / 16) * 16, 16), max(round(h * scale / 16) * 16, 16)


def _source_resolution(unet, pil) -> tuple[int, int]:
    """The resolution an input image will be sampled at, per family."""
    if cat.family_of(unet) == cat.FAMILY_FLUX2:
        return _flux2_resolution(pil)
    return _kontext_resolution(pil)


def _default_guidance(unet, role: str) -> float:
    if cat.family_of(unet) == cat.FAMILY_FLUX2:
        return FLUX2_GUIDANCE
    return KONTEXT_GUIDANCE if role == ROLE_EDIT else CREATE_GUIDANCE

_proc: subprocess.Popen | None = None  # the ComfyUI child, if we started it


# --------------------------------------------------------------------------- #
# Availability / server lifecycle
# --------------------------------------------------------------------------- #
def runtime_ready() -> bool:
    """True if the engine is installed — enough to download models, not to generate."""
    return CVENV_PY.exists() and (COMFY_DIR / "main.py").exists()


def available() -> bool:
    """True if the app can actually generate: engine installed *and* a model with it.

    run.sh no longer downloads weights, so a fresh install is runtime_ready() but not
    available() until the user installs a bundle from the Models panel.
    """
    return runtime_ready() and bool(cat.installed_bundles())


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
    if not runtime_ready():
        raise RuntimeError("The image engine isn't installed. Re-run ./run.sh.")
    if not cat.installed_bundles():
        raise RuntimeError("No image model is installed — install one in the Models panel.")
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
    try:
        with urllib.request.urlopen(req) as r:
            return json.load(r)
    except urllib.error.HTTPError as exc:
        # ComfyUI rejects a graph it can't run with a 400 whose *body* says why — which
        # node, which input, what it expected. Letting urllib raise the bare status threw
        # that away and left the UI showing "HTTP Error 400: Bad Request".
        raise RuntimeError(_comfy_error(exc)) from exc


def _comfy_error(exc) -> str:
    """The human-readable complaint out of a ComfyUI error body."""
    try:
        body = json.loads(exc.read())
    except Exception:
        return f"the image engine rejected the request (HTTP {exc.code})."
    bits = []
    for node in (body.get("node_errors") or {}).values():
        for e in node.get("errors") or []:
            detail = e.get("details") or ""
            bits.append(f"{e.get('message', 'error')}{f' ({detail})' if detail else ''}")
    if not bits:
        err = body.get("error") or {}
        msg = err.get("message") if isinstance(err, dict) else str(err)
        detail = err.get("details") if isinstance(err, dict) else ""
        bits = [f"{msg}{f' ({detail})' if detail else ''}"] if msg else []
    return "; ".join(bits) or f"the image engine rejected the request (HTTP {exc.code})."


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
    """Pick the loader for the transformer's format: GGUF quantized (the GGUF custom
    node) vs. a plain safetensors checkpoint (ComfyUI's built-in UNETLoader)."""
    if unet.lower().endswith(".gguf"):
        return {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": unet}}
    b = cat.bundle_of_unet(unet)
    if b:
        # A catalog bundle states its own dtype. FLUX.2's checkpoint is fp8mixed —
        # already quantized — so it loads as "default"; casting it again would throw
        # away the higher-precision layers the mix deliberately keeps.
        dtype = b["weight_dtype"]
    else:
        # A user-added model: all we have is its size. An unquantized FLUX.1
        # transformer is ~23.8 GB in bf16, which won't fit a 24 GB card alongside the
        # text encoders, so cast it on load rather than refusing it.
        p = UNET_DIR / unet
        dtype = "fp8_e4m3fn" if p.exists() and p.stat().st_size > 16e9 else "default"
    return {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": dtype}}


def _clip_node(name: str, kind: str) -> dict:
    """Pick the loader for a text encoder's format, the way `_unet_node` does.

    ComfyUI's own CLIPLoader can't take a GGUF — `.gguf` isn't in its folder's allowed
    extensions, so naming one gets the whole graph rejected at validation (a bare 400).
    The GGUF node registers a parallel `clip_gguf` folder over the same directory and
    reads the quantized file; both loaders take the same `type`.
    """
    cls = "CLIPLoaderGGUF" if name.lower().endswith(".gguf") else "CLIPLoader"
    return {"class_type": cls, "inputs": {"clip_name": name, "type": kind}}


def _loaders(unet: str) -> dict:
    """The transformer + its text encoder + its VAE. All three are family-specific:
    a FLUX.2 transformer decodes 128-channel latents through its own VAE and reads
    conditioning from a Mistral-3 encoder, none of which FLUX.1's parts can supply."""
    if cat.family_of(unet) == cat.FAMILY_FLUX2:
        b = cat.bundle_of_unet(unet)
        return {
            "unet": _unet_node(unet),
            "clip": _clip_node(clip_for(b), "flux2"),
            "vae": {"class_type": "VAELoader", "inputs": {"vae_name": b["vae"]}},
        }
    return {
        "unet": _unet_node(unet),
        "clip": {"class_type": "DualCLIPLoaderGGUF",
                 "inputs": {"clip_name1": CLIP_L, "clip_name2": T5, "type": "flux"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": FLUX1_VAE}},
    }


def _sampler(g: dict, unet, latent_src, steps, seed, width, height, denoise=1.0) -> None:
    """Add the sampling nodes, writing the output latent to `g["sampler"]` output 0.

    FLUX.1 samples through KSampler at cfg=1.0 — the negative branch is unused, which
    is why this app exposes no negative prompt; guidance rides on FluxGuidance instead.

    FLUX.2 has no negative branch at all (a BasicGuider, not a CFG pair) and needs a
    sequence-length-aware sigma schedule, which only Flux2Scheduler computes — the
    stock "simple" schedule is wrong for it. That combination only exists on the
    custom-sampler path, so it assembles RandomNoise + BasicGuider + KSamplerSelect +
    Flux2Scheduler into SamplerCustomAdvanced.
    """
    if cat.family_of(unet) != cat.FAMILY_FLUX2:
        g["sampler"] = {
            "class_type": "KSampler",
            "inputs": {"model": ["unet", 0], "positive": ["guide", 0], "negative": ["neg", 0],
                       "latent_image": list(latent_src), "seed": seed, "steps": steps, "cfg": 1.0,
                       "sampler_name": "euler", "scheduler": "simple", "denoise": denoise},
        }
        return

    g["noise"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}}
    g["guider"] = {"class_type": "BasicGuider",
                   "inputs": {"model": ["unet", 0], "conditioning": ["guide", 0]}}
    g["sampler_sel"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}}
    g["sigmas"] = {"class_type": "Flux2Scheduler",
                   "inputs": {"steps": steps, "width": width, "height": height}}
    sigmas = ("sigmas", 0)
    if denoise < 1.0:
        # SamplerCustomAdvanced has no denoise input: partial denoise means starting
        # part-way down the schedule. SplitSigmas output 1 is the tail (`sigmas[step:]`),
        # so dropping the first (1 - denoise) of the steps leaves exactly that.
        g["split"] = {"class_type": "SplitSigmas",
                      "inputs": {"sigmas": ["sigmas", 0], "step": round(steps * (1.0 - denoise))}}
        sigmas = ("split", 1)
    g["sampler"] = {
        "class_type": "SamplerCustomAdvanced",
        "inputs": {"noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["sampler_sel", 0],
                   "sigmas": list(sigmas), "latent_image": list(latent_src)},
    }


def _conditioning(unet, prompt: str, guidance: float, ref_latents=()) -> dict:
    """Text conditioning shared by every graph. `guide` is what the sampler reads.

    `ref_latents` chains one `ReferenceLatent` node per encoded reference image. The
    node *appends* to the conditioning and the model forwards the whole list, so each
    reference arrives as its own token block with its own RoPE offsets — the images
    stay distinct. FLUX.2 and Kontext both understand this; a plain FLUX.1 dev
    transformer ignores it, so create graphs pass nothing.
    """
    flux2 = cat.family_of(unet) == cat.FAMILY_FLUX2
    g = {"pos": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["clip", 0]}}}
    if not flux2:
        g["neg"] = {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["clip", 0]}}
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
    # a ComfyUI upgrade can't silently change the default under us. Kontext-only —
    # FLUX.2 lays its references out itself.
    if len(ref_latents) > 1 and not flux2:
        g["refmethod"] = {"class_type": "FluxKontextMultiReferenceLatentMethod",
                          "inputs": {"conditioning": [src, 0], "reference_latents_method": REF_METHOD}}
        src = "refmethod"
    g["guide"] = {"class_type": "FluxGuidance",
                  "inputs": {"conditioning": [src, 0], "guidance": guidance}}
    return g


def _empty_latent(unet, width, height) -> dict:
    """FLUX.1's VAE is 16-channel (EmptySD3LatentImage — the 4-channel SD latent would
    decode to noise); FLUX.2's is 128-channel and has its own node."""
    cls = ("EmptyFlux2LatentImage" if cat.family_of(unet) == cat.FAMILY_FLUX2
           else "EmptySD3LatentImage")
    return {"class_type": cls, "inputs": {"width": width, "height": height, "batch_size": 1}}


def _scale_node(unet, src) -> dict:
    """Snap an input image to a resolution its family was trained on. Kontext has a
    fixed table of ~1 MP shapes; FLUX.2 just wants ~1 MP on a multiple of 16 (its VAE
    downscale), which keeps the token count — and the VRAM — bounded either way."""
    if cat.family_of(unet) == cat.FAMILY_FLUX2:
        return {"class_type": "ImageScaleToTotalPixels",
                "inputs": {"image": list(src), "upscale_method": "area",
                           "megapixels": 1.0, "resolution_steps": 16}}
    return {"class_type": "FluxKontextImageScale", "inputs": {"image": list(src)}}


def _tail(prefix: str) -> dict:
    return {
        "dec": {"class_type": "VAEDecode", "inputs": {"samples": ["sampler", 0], "vae": ["vae", 0]}},
        "save": {"class_type": "SaveImage", "inputs": {"images": ["dec", 0], "filename_prefix": prefix}},
    }


def _txt2img_graph(prompt, width, height, steps, guidance, seed, prefix, unet):
    """Text-to-image. No ReferenceLatent — on FLUX.1 that node is Kontext's
    identity-preserving path and would pin the output to a source image; on FLUX.2
    there simply is no source image to reference."""
    g = _loaders(unet)
    g.update(_conditioning(unet, prompt, guidance))
    g["latent"] = _empty_latent(unet, width, height)
    _sampler(g, unet, ("latent", 0), steps, seed, width, height)
    g.update(_tail(prefix))
    return g


def _img2img_graph(image_name, prompt, strength, steps, guidance, seed, width, height,
                   prefix, unet):
    """Image-to-image: encode the input and partially denoise it. `strength` is the
    denoise fraction — how far the result may drift from the attached image."""
    g = _loaders(unet)
    g.update(_conditioning(unet, prompt, guidance))
    g["img"] = {"class_type": "LoadImage", "inputs": {"image": image_name}}
    g["scale"] = _scale_node(unet, ("img", 0))
    g["enc"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["scale", 0], "vae": ["vae", 0]}}
    _sampler(g, unet, ("enc", 0), steps, seed, width, height, denoise=strength)
    g.update(_tail(prefix))
    return g


def _encode_image(g, unet, name, key):
    """LoadImage -> snap to a resolution the family was trained on -> VAEEncode.
    Returns the latent output ref."""
    g[f"{key}_img"] = {"class_type": "LoadImage", "inputs": {"image": name}}
    g[f"{key}_scale"] = _scale_node(unet, (f"{key}_img", 0))
    g[key] = {"class_type": "VAEEncode", "inputs": {"pixels": [f"{key}_scale", 0], "vae": ["vae", 0]}}
    return (key, 0)


def _edit_graph(scene_name, ref_names, prompt, steps, guidance, seed, width, height,
                prefix, unet):
    """Instruction-edit `scene_name` (FLUX.2, or FLUX.1 Kontext).

    The scene is encoded once and used twice: as the latent being denoised, and as
    the first ReferenceLatent (which is what preserves its identity and background).
    Any `ref_names` are chained on as additional references — that's how a subject
    from another photo gets carried into this one.
    """
    g = _loaders(unet)
    scene = _encode_image(g, unet, scene_name, "enc")
    refs = [scene] + [_encode_image(g, unet, n, f"src{i}") for i, n in enumerate(ref_names)]
    g.update(_conditioning(unet, prompt, guidance, ref_latents=refs))
    _sampler(g, unet, scene, steps, seed, width, height)
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
    refs = [_encode_image(g, unet, n, f"src{i}") for i, n in enumerate(image_names)]
    g.update(_conditioning(unet, prompt, guidance, ref_latents=refs))
    g["latent"] = _empty_latent(unet, width, height)
    _sampler(g, unet, ("latent", 0), steps, seed, width, height)
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


def _label(unet: str) -> str:
    b = cat.bundle_of_unet(unet)
    return b["label"] if b else "FLUX"


def create(prompt, width=None, height=None, steps=None, guidance=None, seed=0,
           model=None, enhance=True, on_step=None, on_status=None):
    """Text-to-image ('a candid photo of a woman laughing')."""
    ensure_server(on_status=on_status)
    unet = _resolve_unet(model, ROLE_CREATE)
    w, h = _dim(width), _dim(height)
    g = _txt2img_graph(_prompt_for(prompt, enhance), w, h, _steps(steps),
                       _guidance(guidance, _default_guidance(unet, ROLE_CREATE)),
                       int(seed), "flux_create", unet)
    if on_status:
        on_status(f"generating with {_label(unet)}…")
    return _run(g, on_step=on_step, on_status=on_status)


def img2img(pil, prompt, strength=None, steps=None, guidance=None, seed=0,
            model=None, enhance=True, on_step=None, on_status=None):
    """Transform an attached image, keeping its composition."""
    ensure_server(on_status=on_status)
    unet = _resolve_unet(model, ROLE_CREATE)
    name = _upload_image(pil, f"init_{uuid.uuid4().hex}.png")
    w, h = _source_resolution(unet, pil)
    g = _img2img_graph(name, _prompt_for(prompt, enhance), _strength(strength), _steps(steps),
                       _guidance(guidance, _default_guidance(unet, ROLE_CREATE)),
                       int(seed), w, h, "flux_img2img", unet)
    if on_status:
        on_status(f"transforming with {_label(unet)}…")
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
    w, h = _source_resolution(unet, pil)
    g = _edit_graph(scene, ref_names, prompt or "", _steps(steps),
                    _guidance(guidance, _default_guidance(unet, ROLE_EDIT)),
                    int(seed), w, h, "flux_edit", unet)
    if on_status:
        on_status(f"editing with {_label(unet)}…")
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
    width, height = _source_resolution(unet, pils[0])
    g = _compose_graph(names, prompt or "", width, height, _steps(steps),
                       _guidance(guidance, _default_guidance(unet, ROLE_EDIT)),
                       int(seed), "flux_compose", unet)
    if on_status:
        on_status(f"composing with {_label(unet)}…")
    return _run(g, on_step=on_step, on_status=on_status)


# --------------------------------------------------------------------------- #
# Model management: list / add (from any HF repo) / remove extra UNets
# --------------------------------------------------------------------------- #
def list_unets() -> list[dict]:
    """Installed transformers. Each: name, roles, bundle id (None if user-added), size_gb.

    `roles` tells the UI which modes a model can serve: a FLUX.2 transformer serves
    both, a FLUX.1 one serves exactly one. Catalog models sort first.
    """
    out = []
    if UNET_DIR.is_dir():
        for p in sorted(UNET_DIR.iterdir()):
            if p.is_file() and p.suffix.lower() in UNET_EXTS:
                b = cat.bundle_of_unet(p.name)
                out.append({
                    "name": p.name,
                    "label": (b["label"] if b else p.name),
                    "roles": cat.roles_of(p.name),
                    "bundle": b["id"] if b else None,
                    "family": cat.family_of(p.name),
                    "size_gb": round(p.stat().st_size / 1e9, 2),
                })
    out.sort(key=lambda m: (m["bundle"] is None, m["name"].lower()))
    return out


def _default_for(role: str) -> str:
    """The transformer a mode runs on when the user hasn't picked one.

    Catalog order is quality order, so the first installed bundle that can serve the
    role wins — FLUX.2 over FLUX.1 when both are installed.
    """
    for b in cat.installed_bundles():
        if role not in b["roles"]:
            continue
        name = b["unet_edit"] if (role == ROLE_EDIT and b.get("unet_edit")) else b["unet"]
        if (UNET_DIR / name).exists():
            return name
    # No bundle can serve this role (e.g. only a user-added create model is present).
    for m in list_unets():
        if role in m["roles"]:
            return m["name"]
    raise RuntimeError("No image model is installed — install one in the Models panel.")


def _resolve_unet(model, role: str) -> str:
    """Map a requested model name to an installed transformer, guarding against path
    traversal. Unknown, empty, or wrong-role → that role's default.

    The role check matters on FLUX.1: an edit graph feeds a ReferenceLatent that a
    plain dev transformer ignores, and a dev one asked to edit would just regenerate
    from scratch. FLUX.2 serves both roles, so it passes either way.
    """
    if not model:
        return _default_for(role)
    safe = os.path.basename(str(model))
    if (safe.lower().endswith(UNET_EXTS)
            and (UNET_DIR / safe).exists()
            and role in cat.roles_of(safe)):
        return safe
    return _default_for(role)


def delete_unet(name: str) -> None:
    """Remove a user-added transformer. Catalog models are removed as a bundle
    (`delete_bundle`) — deleting just their UNet would strand their encoder and VAE."""
    safe = os.path.basename(name or "")
    if not safe.lower().endswith(UNET_EXTS):
        raise ValueError("Not a model file.")
    if cat.bundle_of_unet(safe):
        raise ValueError("This model is part of an installed bundle — remove it in the "
                         "Models panel instead.")
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


# Companion files a FLUX repo ships next to the transformer — adapters, the VAE, the
# text encoders. They're the same extension as the UNet, so name is the only signal.
_AUX_RE = re.compile(
    r"lora|vae|\bae\b|text_encoder|tokenizer|clip|t5|controlnet|embed|encoder|decoder",
    re.I,
)
# Task-specific FLUX variants. They're real transformers, but they condition on extra
# inputs (a mask, a control image) that our graphs don't feed, so they lose to a base
# model of the same size — a repo shipping both means the base one is what was wanted.
_VARIANT_RE = re.compile(r"fill|canny|depth|redux|inpaint|outpaint", re.I)


def _pick_unet(files: list[dict], repo: str) -> str:
    """Choose the transformer from a repo's tensor files (each: name, size).

    Auto-pick has nothing to go on but names and sizes, so: drop the companion files,
    drop shards (UNETLoader takes a single file), then prefer a GGUF — it's the cheaper
    tier — and otherwise take the largest checkpoint, which in a FLUX repo is always
    the transformer. Picking by list order instead is what made an unquantized repo
    resolve to its alphabetically-first file, `ae.safetensors`.
    """
    if not files:
        raise ValueError(f"No .gguf or .safetensors files found in {repo}.")
    real = [f for f in files if not _AUX_RE.search(f["name"])]
    cand = [f for f in real if "-of-" not in f["name"].lower()]
    if not cand:
        if real:
            raise ValueError(
                f"{repo} only ships a sharded checkpoint, which ComfyUI can't load. "
                "It needs a single-file .gguf or .safetensors UNet."
            )
        listed = ", ".join(sorted(os.path.basename(f["name"]) for f in files)[:6])
        raise ValueError(
            f"No UNet found in {repo} — it only holds companion files ({listed}). "
            "Pass owner/repo:file to name a UNet yourself."
        )
    # Sorting by name first makes `max` break size ties deterministically.
    cand.sort(key=lambda f: f["name"])
    cand = [f for f in cand if not _VARIANT_RE.search(f["name"])] or cand
    ggufs = [f for f in cand if f["name"].lower().endswith(".gguf")]
    if ggufs:
        for quant in ("Q8_0", "Q6_K", "Q4_K"):
            hit = next((f for f in ggufs if quant in f["name"]), None)
            if hit:
                return hit["name"]
        return max(ggufs, key=lambda f: f["size"])["name"]
    return max(cand, key=lambda f: f["size"])["name"]


# A FLUX transformer names its blocks `double_blocks`/`single_blocks` (original layout)
# or `transformer_blocks` (diffusers). A LoRA names its blocks the same way and only
# differs in the adapter tensors hung off them — and every naming convention in the
# wild (`lora_A`, `lora_down`, `proj_lora1.down`, `lora_unet_…`) spells out "lora",
# while no transformer tensor does. So test for that first, before the block names.
_UNET_KEYS = ("double_blocks.", "single_blocks.", "transformer_blocks.")
# An all-in-one checkpoint bundles the text encoders and VAE alongside the transformer.
# It also carries the block names above, so look for the bundled parts first.
_BUNDLE_KEYS = ("text_encoders.", "conditioner.", "vae.", "first_stage_model.")


def _looks_like_unet(keys: list[str]) -> bool:
    return any(m in k for k in keys for m in _UNET_KEYS)


def _reject_reason(keys: list[str]) -> str | None:
    """Why this safetensors checkpoint can't serve as a UNet, or None if it can.

    An unrecognized layout is allowed through: this catches the known-wrong files a
    repo might hand us, it isn't a whitelist of blessed architectures.
    """
    lowered = [k.lower() for k in keys]
    if any("lora" in k for k in lowered):
        return "it's a LoRA adapter, not a full UNet"
    if any(k.startswith(_BUNDLE_KEYS) for k in lowered):
        return ("it bundles the text encoders and VAE, and this runtime loads a bare "
                "diffusion model — look for one under split_files/diffusion_models/")
    if _looks_like_unet(keys):
        return None
    if any(k.startswith(("encoder.", "decoder.")) for k in lowered):
        return "it's a VAE, not a UNet"
    return None


# Runs in a child process (network allowed) so the serving process stays offline. The
# child only does I/O — probe (list files + sizes), inspect (read a safetensors header),
# fetch (stream into the UNet dir, no HF-cache copy) — so the rules for picking and
# validating a UNet stay in the parent, offline and testable. Progress goes to stdout.
_HF_CHILD = r'''
import json, os, struct, sys, time, urllib.request
from urllib.error import HTTPError, URLError

EXTS = (".gguf", ".safetensors", ".sft")
MAX_HEADER = 64 << 20  # a safetensors header is KBs; larger means it isn't one
TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""

def _api():
    # `token=False` (not None) so huggingface_hub uses the token the parent handed us
    # and nothing else. With None it silently falls back to a token cached in
    # ~/.cache/huggingface/token — a credential this app never wrote and can't replace,
    # which is exactly the stale-token trap we're avoiding.
    from huggingface_hub import HfApi
    return HfApi(token=TOKEN or False)

def _url(repo, filename):
    return "https://huggingface.co/%s/resolve/main/%s?download=true" % (repo, filename)

def _open(url, headers=None):
    headers = dict(headers or {})
    if TOKEN:  # gated repos serve weights only to an accepted license
        headers["Authorization"] = "Bearer " + TOKEN
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers))

def _ranged(url, first, last):
    # Refuse a server that ignores Range and would hand back the whole multi-GB file.
    with _open(url, {"Range": "bytes=%d-%d" % (first, last)}) as r:
        if r.status != 206:
            raise RuntimeError("huggingface.co ignored a range request")
        return r.read(last - first + 1)

def whoami():
    # Validates a pasted token before it's saved. Without one this 401s, which is the
    # answer too: "no valid token".
    with _open("https://huggingface.co/api/whoami-v2") as r:
        info = json.load(r)
    print(json.dumps([info.get("name") or "?"]), flush=True)

def probe(repo):
    # files_metadata gives sizes; list_repo_files doesn't. The token matters here too:
    # a gated repo won't even list its files to an anonymous caller.
    info = _api().model_info(repo, files_metadata=True)
    print(json.dumps([{"name": s.rfilename, "size": s.size or 0} for s in info.siblings
                      if s.rfilename.lower().endswith(EXTS)]), flush=True)

def listing(repo):
    # Every file, not just the tensors probe() reports — a transformers-layout encoder
    # keeps its tokenizer in there too, and we need to see it.
    print(json.dumps(sorted(_api().list_repo_files(repo))), flush=True)

def inspect(repo, filename):
    url = _url(repo, filename)
    n = struct.unpack("<Q", _ranged(url, 0, 7))[0]
    if n > MAX_HEADER:
        raise RuntimeError("%s has no readable safetensors header" % filename)
    header = json.loads(_ranged(url, 8, 8 + n - 1))
    print(json.dumps([k for k in header if k != "__metadata__"]), flush=True)

def fetch(repo, filename, dest_dir, out_name=""):
    # out_name renames on the way in. Upstream file names collide across models —
    # FLUX.2's VAE is `ae.safetensors`, the same name FLUX.1 gives its very different one.
    base = out_name or os.path.basename(filename)
    out = os.path.join(dest_dir, base)
    if os.path.exists(out):
        raise RuntimeError("a model named %s already exists." % base)
    if not os.path.isdir(dest_dir):
        os.makedirs(dest_dir)
    tmp = out + ".part"
    # Resume a part-file from an interrupted run. These are tens of GB — restarting a
    # 33 GB download because the connection dropped at 90% is not an option.
    have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    r = _open(_url(repo, filename), {"Range": "bytes=%d-" % have} if have else None)
    if have and r.status != 206:  # server ignored the range; start clean
        r.close()
        have = 0
        r = _open(_url(repo, filename))
    print("Downloading %s from %s%s…"
          % (base, repo, (" (resuming at %.2f GB)" % (have/1e9)) if have else ""), flush=True)
    total = int(r.headers.get("Content-Length", 0)) + have
    done, last = have, 0
    with r, open(tmp, "ab" if have else "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk); done += len(chunk)
            if time.time() - last > 0.5:
                print("PROGRESS " + json.dumps({"file": base, "done": done, "total": total,
                      "pct": (100 * done // total) if total else 0}), flush=True)
                last = time.time()
    if total and done < total:
        raise RuntimeError("%s: connection dropped at %d%% — retry to resume."
                           % (base, 100 * done // total))
    os.replace(tmp, out)
    print("PROGRESS " + json.dumps({"file": base, "done": done, "total": total or done,
                                    "pct": 100}), flush=True)
    print("DONE " + base, flush=True)

try:
    {"whoami": whoami, "probe": probe, "listing": listing, "inspect": inspect,
     "fetch": fetch}[sys.argv[1]](*sys.argv[2:])
except HTTPError as e:
    # A gated repo (black-forest-labs' own among them) lists its files to anyone but
    # serves the weights only to an accepted license, so this is the common failure,
    # not a typo.
    if e.code in (401, 403):
        if sys.argv[1] == "whoami":
            print("ERROR: HuggingFace rejected that token.", flush=True)
        else:
            print("ERROR: %s is gated — accept its license on huggingface.co, then paste "
                  "a HuggingFace token in the Models panel." % sys.argv[2], flush=True)
    else:
        print("ERROR: huggingface.co returned %s for %s" % (e.code, sys.argv[-1]), flush=True)
    sys.exit(1)
except URLError as e:
    print("ERROR: could not reach huggingface.co (%s)" % e, flush=True); sys.exit(1)
except Exception as e:
    print("ERROR: %s" % e, flush=True); sys.exit(1)
'''


def _run_child(args: list[str], on_status=None, on_progress=None, token=None) -> list[str]:
    """Run one `_HF_CHILD` mode, streaming its stdout, and return the lines it printed.

    The offline flags are dropped for the child alone, so the serving process itself
    never gains network access. The HuggingFace token is handed over the same way —
    only this child ever sees it.

    `PROGRESS {...}` lines carry structured download progress; everything else is a
    human status line.
    """
    import sys  # PLC0415

    # `token` is an unsaved one being validated; otherwise use whatever is configured
    # (the saved token, else one from the environment). Resolved before the scrub below,
    # which only touches the child's copy of the environment.
    tok = (token or settings.hf_token()).strip()

    env = {**os.environ}
    env.pop("HF_HUB_OFFLINE", None)
    env.pop("TRANSFORMERS_OFFLINE", None)
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    # Exactly one token reaches the child. The child inherits the server's environment,
    # so a token exported there (a stale HUGGING_FACE_HUB_TOKEN, say) would otherwise
    # travel alongside the configured one — and still be used after the saved token was
    # replaced or cleared. Drop every token variable first, then set only the one we
    # resolved above.
    for var in settings.ENV_VARS:
        env.pop(var, None)
    if tok:
        env["HF_TOKEN"] = tok

    proc = subprocess.Popen(
        [sys.executable, "-c", _HF_CHILD, *args],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0,
    )
    lines: list[str] = []
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
            elif line.startswith("PROGRESS "):
                ev = json.loads(line[9:])
                # Throttle: the child speaks every 0.5s per file, but a bundle install
                # runs several of them and the browser doesn't need every tick.
                if (time.time() - last) > 0.4 or ev["pct"] == 100:
                    if on_progress:
                        on_progress(ev)
                    elif on_status:
                        on_status("Downloading %s: %.2f/%.2f GB (%d%%)"
                                  % (ev["file"], ev["done"] / 1e9, ev["total"] / 1e9, ev["pct"]))
                    last = time.time()
            elif line:
                lines.append(line)
                if on_status:
                    on_status(line)
        else:
            buf += ch
    if proc.wait() != 0 or err:
        raise RuntimeError(err or "Download failed. Check the repo id and your connection.")
    return lines


def _json_line(lines: list[str]) -> list:
    """The child's JSON payload, ignoring any warning HuggingFace wrote to stderr."""
    for line in reversed(lines):
        if line.startswith("["):
            return json.loads(line)
    raise RuntimeError("HuggingFace returned nothing usable. Check the repo id.")


def pull_unet(repo: str, on_status=None, on_progress=None) -> None:
    """Download an extra FLUX UNet from a HuggingFace repo (opt-in, streams status).

    A bare `owner/repo` auto-picks the transformer; `owner/repo:file` names it outright.
    Either way a safetensors checkpoint is checked against its header before the bytes
    are spent, so a LoRA or VAE can't land in the UNet dir and be loaded as a model.

    These are extras, and they run on FLUX.1's encoders (see `flux_catalog.family_of`)
    — so that bundle has to be installed for them to load.
    """
    repo_id, filename = _parse_repo(repo)
    if not runtime_ready():
        raise RuntimeError("The image engine isn't installed. Re-run ./run.sh.")
    UNET_DIR.mkdir(parents=True, exist_ok=True)
    say = on_status or (lambda _msg: None)

    say(f"Resolving {repo_id}…")
    if not filename:
        filename = _pick_unet(_json_line(_run_child(["probe", repo_id])), repo_id)
        say(f"Selected {os.path.basename(filename)}.")

    if filename.lower().endswith((".safetensors", ".sft")):
        say(f"Checking {os.path.basename(filename)}…")
        keys = _json_line(_run_child(["inspect", repo_id, filename]))
        reason = _reject_reason(keys)
        if reason:
            raise RuntimeError(
                f"{os.path.basename(filename)} can't be used — {reason}. "
                "Pass owner/repo:file to name the UNet yourself."
            )
        if not _looks_like_unet(keys):
            say(f"{os.path.basename(filename)} doesn't look like a FLUX UNet — loading anyway.")

    _run_child(["fetch", repo_id, filename, str(UNET_DIR)], say, on_progress=on_progress)
    say("Download complete.")


# --------------------------------------------------------------------------- #
# Bundles: install / remove a catalog model (weights, encoder and VAE together)
# --------------------------------------------------------------------------- #
def catalog() -> list[dict]:
    """The installable models, each with its install state. Drives the Models panel."""
    return [
        {
            "id": b["id"],
            "label": b["label"],
            "blurb": b["blurb"],
            "family": b["family"],
            "roles": b["roles"],
            "size_gb": b["size_gb"],
            "vram_gb": b["vram_gb"],
            "gated": b["gated"],
            "installed": cat.installed(b),
            "needed_gb": cat.needed_gb(b),
        }
        for b in cat.BUNDLES
    ]


def _st_header(path) -> tuple[dict, int]:
    """A safetensors header and the absolute offset its data buffer starts at."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if n > 256 << 20:
            raise RuntimeError(f"{os.path.basename(path)} isn't a safetensors file.")
        return json.loads(f.read(n)), 8 + n


def _merge_shards(shards: list, out: Path, embed: tuple | None = None,
                  on_progress=None) -> None:
    """Stitch a sharded safetensors checkpoint back into the single file ComfyUI loads.

    Copies tensor *bytes* between files instead of materializing them: [dev]'s encoder
    is 48 GB and loading it into RAM to re-save would need more memory than the machine
    has. So we rebuild the container — read every shard's header, lay the tensors out
    back-to-back, and stream each one across.

    `embed` is a (local-file, tensor-name) blob to carry in alongside the weights.
    Mistral's tokenizer lives outside the checkpoint upstream, but ComfyUI expects to
    find it *inside*, as a uint8 tensor. It's fetched by the download child like any
    other file — this runs in the serving process, which has no network access.
    """
    plan, offset = [], 0
    for shard in shards:
        header, data_start = _st_header(shard)
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            begin, end = meta["data_offsets"]
            size = end - begin
            plan.append((name, meta["dtype"], meta["shape"], shard,
                         data_start + begin, size, offset))
            offset += size

    names = [p[0] for p in plan]
    if len(names) != len(set(names)):
        raise RuntimeError("These shards overlap — they aren't one checkpoint.")

    blob = b""
    if embed:
        path, tensor = embed
        blob = Path(path).read_bytes()
        plan.append((tensor, "U8", [len(blob)], None, 0, len(blob), offset))
        offset += len(blob)

    header = {n: {"dtype": d, "shape": s, "data_offsets": [o, o + sz]}
              for n, d, s, _f, _st, sz, o in plan}
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)  # keep the data buffer 8-byte aligned

    total = offset
    done = 0
    last = 0.0  # throttle: chunks land every few ms and the browser doesn't need each one
    tmp = out.with_name(out.name + ".part")
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(tmp, "wb") as w:
            w.write(struct.pack("<Q", len(raw)))
            w.write(raw)
            for name, _d, _s, shard, start, size, _o in plan:
                if shard is None:  # the embedded blob is already in memory
                    w.write(blob)
                    done += size
                    continue
                with open(shard, "rb") as r:
                    r.seek(start)
                    left = size
                    while left:
                        chunk = r.read(min(1 << 22, left))
                        if not chunk:
                            raise RuntimeError(f"{os.path.basename(shard)} is truncated.")
                        w.write(chunk)
                        left -= len(chunk)
                        done += len(chunk)
                        if on_progress and time.time() - last > 0.4:
                            on_progress({"file": out.name, "done": done, "total": total,
                                         "pct": 100 * done // max(total, 1)})
                            last = time.time()
        if on_progress:  # the throttle can swallow the last chunk; don't end at 97%
            on_progress({"file": out.name, "done": total, "total": total, "pct": 100})
        os.replace(tmp, out)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


# An install outlives the request that started it: these are 50 GB downloads, and the
# browser *will* reload or wander off. So it runs under a process-wide lock, and its
# progress is kept here rather than only streamed — otherwise a reload would leave the
# user staring at a panel that can't see the download still running underneath it, and
# a second click would start a rival writer appending to the same .part file.
_install_mu = threading.Lock()
_install_state: dict | None = None


def install_state() -> dict | None:
    """The install running right now, if any: {id, label, file, pct, done, total}."""
    return dict(_install_state) if _install_state else None


def install_bundle(bundle_id: str, on_status=None, on_progress=None) -> None:
    """Download every file a catalog model needs, skipping the ones already there.

    Files are fetched one at a time and land atomically (`.part` → rename), so an
    interrupted install resumes where it stopped rather than starting over.
    """
    global _install_state
    b = cat.get(bundle_id)
    if not runtime_ready():
        raise RuntimeError("The image engine isn't installed. Re-run ./run.sh.")
    if not _install_mu.acquire(blocking=False):
        running = install_state() or {}
        raise RuntimeError(f"An install is already running ({running.get('label', '?')}). "
                           "Wait for it to finish.")
    try:
        say = on_status or (lambda _msg: None)

        todo = cat.missing_files(b)
        pending = cat.pending_merges(b)
        if not todo and not pending:
            say(f"{b['label']} is already installed.")
            return

        # Check the disk before spending an hour on a download that can't land. A merge
        # briefly stores its encoder twice (shards + stitched file), so check the peak.
        need = cat.peak_gb(b)
        free = cat.free_gb()
        if free < need + 2:  # a couple of GB of headroom for the .part → rename
            raise RuntimeError(
                f"Not enough disk space for {b['label']}: it needs {need:.0f} GB "
                f"(peak, while the text encoder is being stitched together) and only "
                f"{free:.0f} GB is free."
            )

        # Every download, whether it lands in a model dir or in a merge's staging area.
        # A merge's shards are fetched like anything else and collapsed afterwards; the
        # tokenizer blob rides along as one more download.
        steps = [{"repo": f[0], "path": f[1], "dest": cat.dest_dir(f[2]),
                  "save_as": f[3] if len(f) > 3 else ""} for f in todo]
        for m in pending:
            paths = list(m["shards"]) + ([m["embed"][1]] if m["embed"] else [])
            repos = [m["repo"]] * len(m["shards"]) + ([m["embed"][0]] if m["embed"] else [])
            steps += [{"repo": r, "path": p, "dest": cat.staging_dir(m), "save_as": ""}
                      for r, p in zip(repos, paths)]

        token = settings.hf_token()
        total_steps = len(steps) + len(pending)
        _install_state = {"id": b["id"], "label": b["label"], "file": "", "pct": 0,
                          "done": 0, "total": 0, "index": 1, "count": total_steps}

        def progress(p):
            _install_state.update(p)
            if on_progress:
                on_progress(p)

        for i, s in enumerate(steps, 1):
            s["dest"].mkdir(parents=True, exist_ok=True)
            name = s["save_as"] or os.path.basename(s["path"])
            _install_state.update({"index": i, "file": name, "pct": 0})
            say(f"[{i}/{total_steps}] {name}")
            _run_child(["fetch", s["repo"], s["path"], str(s["dest"]), s["save_as"]],
                       on_status=say, on_progress=progress, token=token)

        for j, m in enumerate(pending, len(steps) + 1):
            out = cat.merge_out(m)
            _install_state.update({"index": j, "file": out.name, "pct": 0})
            say(f"[{j}/{total_steps}] stitching {out.name} from {len(m['shards'])} shards…")
            embed = None
            if m["embed"]:
                embed = (cat.shard_path(m, m["embed"][1]), m["embed"][2])
            _merge_shards([cat.shard_path(m, s) for s in m["shards"]], out,
                          embed=embed, on_progress=progress)
            shutil.rmtree(cat.staging_dir(m), ignore_errors=True)  # shards served their purpose
        say(f"{b['label']} installed.")
    finally:
        _install_state = None
        _install_mu.release()


def delete_bundle(bundle_id: str) -> None:
    """Remove a model's files — but not any it shares with another installed bundle
    (FLUX.1's VAE and encoders would otherwise be pulled out from under it)."""
    b = cat.get(bundle_id)
    running = install_state()
    if running:
        raise ValueError(f"Can't remove anything while {running['label']} is downloading.")
    keep = {cat.file_path(f)
            for other in cat.BUNDLES if other["id"] != bundle_id and cat.installed(other)
            for f in other["files"]}
    keep |= {cat.merge_out(m)
             for other in cat.BUNDLES if other["id"] != bundle_id and cat.installed(other)
             for m in cat.merges(other)}
    removed = 0
    for p in [cat.file_path(f) for f in b["files"]] + [cat.merge_out(m) for m in cat.merges(b)]:
        if p.exists() and p not in keep:
            p.unlink()
            removed += 1
        p.with_name(p.name + ".part").unlink(missing_ok=True)
    for m in cat.merges(b):
        shutil.rmtree(cat.staging_dir(m), ignore_errors=True)
    if not removed:
        raise FileNotFoundError(bundle_id)


# --------------------------------------------------------------------------- #
# Text encoders
# --------------------------------------------------------------------------- #
# A bundle names the encoder it was trained against, but that file is separable: any
# checkpoint of the same architecture conditions the transformer just as well, and a
# lighter quant of it is the usual reason to swap (FLUX.2 [dev]'s Mistral is 48 GB in
# bf16). ComfyUI identifies the architecture from the checkpoint itself, so a wrong one
# fails at load rather than generating quietly-broken images.
def clip_for(bundle: dict) -> str:
    """The text encoder this model will actually load — the user's pick, or its default.

    An override that's been deleted off disk falls back to the default rather than
    failing the graph.
    """
    chosen = settings.text_encoders().get(bundle["id"], "")
    if chosen and (cat.TE_DIR / os.path.basename(chosen)).exists():
        return os.path.basename(chosen)
    return bundle["clip"]


def list_text_encoders() -> list[dict]:
    """Every text encoder on disk, with the models each one is the default for."""
    defaults: dict[str, list[str]] = {}
    for b in cat.BUNDLES:
        if b["family"] == cat.FAMILY_FLUX2:
            defaults.setdefault(b["clip"], []).append(b["label"])
    out = []
    if cat.TE_DIR.is_dir():
        for p in sorted(cat.TE_DIR.iterdir()):
            if p.is_file() and p.suffix.lower() in (".safetensors", ".sft", ".gguf"):
                out.append({
                    "name": p.name,
                    "size_gb": round(p.stat().st_size / 1e9, 2),
                    "default_for": defaults.get(p.name, []),
                })
    return out


def selected_text_encoders() -> dict[str, str]:
    """What each FLUX.2 model is currently set to load."""
    return {b["id"]: clip_for(b) for b in cat.BUNDLES if b["family"] == cat.FAMILY_FLUX2}


def set_text_encoder(bundle_id: str, name: str) -> None:
    b = cat.get(bundle_id)
    if b["family"] != cat.FAMILY_FLUX2:
        raise ValueError(f"{b['label']} doesn't take a swappable text encoder.")
    safe = os.path.basename(name or "")
    if safe and not (cat.TE_DIR / safe).exists():
        raise FileNotFoundError(safe)
    settings.set_text_encoder(bundle_id, safe)


_SHARD_RE = re.compile(r"^(?P<stem>.*?)-\d{5}-of-\d{5}\.safetensors$", re.I)

# What is *not* a text encoder. The mirror image of `_AUX_RE`, which lists what isn't a
# UNet — and which must not be reused here: it rejects anything named `clip`, `t5` or
# `encoder`, i.e. exactly the files this function exists to find.
_NOT_TE_RE = re.compile(r"lora|vae|\bae\b|unet|transformer|diffusion_model|controlnet", re.I)


def _shard_group(files: list[str]) -> list[str]:
    """The one sharded checkpoint in a repo, in order — or [] if there isn't exactly one.

    A transformers-layout repo (what `AutoModel.from_pretrained` reads) keeps its weights
    as `model-00001-of-00004.safetensors` under `text_encoder/`. ComfyUI can't load that,
    so we pull the set and stitch it back into one file.
    """
    groups: dict[str, list[str]] = {}
    for f in files:
        m = _SHARD_RE.match(os.path.basename(f))
        if m:
            groups.setdefault(f"{os.path.dirname(f)}/{m.group('stem')}", []).append(f)
    if not groups:
        return []
    # A model repo ships the transformer sharded too; the encoder is the one under
    # text_encoder/. Anything else ambiguous, we'd rather ask than guess.
    te = [k for k in groups if os.path.dirname(k).endswith("text_encoder")]
    if len(groups) > 1 and len(te) != 1:
        return []
    return sorted(groups[te[0] if te else next(iter(groups))])


def pull_text_encoder(repo: str, on_status=None, on_progress=None) -> None:
    """Add a text encoder from any HuggingFace repo, gated ones included.

    Three shapes of repo, because that's what people actually paste:
      owner/repo:file    — that exact checkpoint
      owner/repo         — one holding a single-file encoder (a ComfyUI-style release)
      owner/repo         — transformers layout, sharded: downloaded and stitched into the
                           single file ComfyUI's CLIPLoader takes. This is the
                           `AutoModel.from_pretrained` case, which ComfyUI can't read
                           directly because it builds the encoder itself rather than
                           handing the job to transformers.

    ComfyUI names the architecture from the checkpoint's own tensors, so the encoder gets
    hooked to whichever model you point at it — and one it can't use fails at load rather
    than quietly conditioning on nonsense.
    """
    repo_id, filename = _parse_repo(repo)
    token = settings.hf_token()
    cat.TE_DIR.mkdir(parents=True, exist_ok=True)
    say = on_status or (lambda _m: None)
    tick = on_progress or (lambda _p: None)

    if filename:
        _run_child(["fetch", repo_id, filename, str(cat.TE_DIR)],
                   on_status=say, on_progress=tick, token=token)
        say("Download complete.")
        return

    say(f"Resolving {repo_id}…")
    # Named after the repo, not the file inside it: a transformers repo calls its weights
    # `model.safetensors`, and every one of them would land on top of the last.
    out = cat.TE_DIR / f"{repo_id.split('/')[-1].lower().replace('.', '_')}.safetensors"
    if out.exists():
        raise ValueError(f"{out.name} is already installed.")

    # Shards first. A full model repo holds the transformer *and* the encoder, and only
    # the encoder is sharded under text_encoder/ — checking single files first would pick
    # FLUX.2 [dev]'s 64 GB transformer and call it a text encoder.
    listing = _json_line(_run_child(["listing", repo_id], token=token))
    shards = _shard_group(listing)
    if not shards:
        tensors = _json_line(_run_child(["probe", repo_id], token=token))
        whole = [f for f in tensors
                 if "-of-" not in f["name"].lower() and not _NOT_TE_RE.search(f["name"])]
        if len(whole) == 1:
            _run_child(["fetch", repo_id, whole[0]["name"], str(cat.TE_DIR), out.name],
                       on_status=say, on_progress=tick, token=token)
            say("Download complete.")
            return
        if len(whole) > 1:
            listed = "\n".join(f"  {repo_id}:{f['name']}  ({f['size'] / 1e9:.1f} GB)"
                               for f in sorted(whole, key=lambda f: -f["size"])[:8])
            raise ValueError(
                f"{repo_id} holds several encoders — name the one you want:\n{listed}")
        raise ValueError(f"No text encoder found in {repo_id}. Use owner/repo:file to name one.")
    staging = cat.STAGING_DIR / out.stem

    # Mistral keeps its tokenizer outside the checkpoint, but ComfyUI reads it from
    # *inside*, as a `tekken_model` tensor. If the repo carries one, bring it along —
    # without it a Mistral encoder loads and then fails at the first prompt.
    tekken = next((f for f in listing if os.path.basename(f) == "tekken.json"), None)
    # The stitch is a step of its own, not a pause at 100%: it moves 16-48 GB of bytes and
    # takes long enough that a bar frozen on the last shard reads as a hang.
    total_steps = len(shards) + (1 if tekken else 0) + 1

    def step(i: int):
        return lambda p: tick({**p, "index": i, "count": total_steps})

    try:
        for i, s in enumerate(shards, 1):
            say(f"[{i}/{total_steps}] {os.path.basename(s)}")
            _run_child(["fetch", repo_id, s, str(staging)],
                       on_status=say, on_progress=step(i), token=token)
        if tekken:
            say(f"[{len(shards) + 1}/{total_steps}] {os.path.basename(tekken)}")
            _run_child(["fetch", repo_id, tekken, str(staging)],
                       on_status=say, on_progress=step(len(shards) + 1), token=token)
        say(f"[{total_steps}/{total_steps}] stitching {out.name} from {len(shards)} shards…")
        _merge_shards([staging / os.path.basename(s) for s in shards], out,
                      embed=(staging / "tekken.json", "tekken_model") if tekken else None,
                      on_progress=step(total_steps))
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    say("Download complete.")


def delete_text_encoder(name: str) -> None:
    """Remove a text encoder — unless a model is currently set to load it."""
    safe = os.path.basename(name or "")
    p = cat.TE_DIR / safe
    if not p.exists():
        raise FileNotFoundError(safe)
    for b in cat.BUNDLES:
        if b["family"] == cat.FAMILY_FLUX2 and cat.installed(b) and clip_for(b) == safe:
            raise ValueError(f"{b['label']} is using {safe}. Point it at another encoder first.")
    p.unlink()


def verify_token(token: str) -> str:
    """Ask HuggingFace who a token belongs to. Raises if it doesn't belong to anyone.

    Checked before saving, so a typo'd token fails at the paste rather than an hour
    into a download.
    """
    if not (token or "").strip():
        raise ValueError("Paste a token first.")
    name = _json_line(_run_child(["whoami"], token=token.strip()))
    return name[0] if name else "?"
