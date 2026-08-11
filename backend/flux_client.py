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
T5 = "t5xxl_fp16.safetensors"
CLIP_L = "clip_l.safetensors"
FLUX1_VAE = "ae.safetensors"

ROLE_CREATE = cat.ROLE_CREATE
ROLE_EDIT = cat.ROLE_EDIT
ROLE_ANIMATE = cat.ROLE_ANIMATE


# Quality-first defaults: 20 steps at Q8 is the sweet spot on this GPU; fewer
# steps visibly degrades output, so that isn't the knob we turn for speed
# (keeping models resident is). [klein] is the exception — a distilled model,
# like Wan's Lightning LoRA path, converges in far fewer steps, so it defaults
# lower and the floor drops to let it go lower still.
DEFAULT_STEPS = 20
FLUX2_STEPS = 35
KLEIN_STEPS = 8
STEPS_MIN, STEPS_MAX = 4, 60
# Guidance is mode- and family-specific. On FLUX.1, Kontext follows an instruction at
# ~2.5 while dev needs a higher ~3.5 to bind a text-only prompt; feeding either the
# other's value (or a stray SD-scale 7.5 from shared settings) blows out the image.
#
# Do not raise KONTEXT_GUIDANCE to "make edits stronger" — it does the opposite.
# At 3.5 the model clings to the reference image and silently ignores the
# instruction, returning the source unchanged (measured on a two-reference edit).
#
# FLUX.2 [dev] uses one value for both jobs (ComfyUI's own template ships 4.0;
# 2.5 is tuned lower for this GPU/workflow). [klein] is a distilled 9B variant
# sharing the family, so it gets its own value rather than inheriting FLUX2_GUIDANCE.
KONTEXT_GUIDANCE = 2.5
CREATE_GUIDANCE = 3.5
FLUX2_GUIDANCE = 2.5
KLEIN_GUIDANCE = 3.5
GUIDANCE_MIN, GUIDANCE_MAX = 0.5, 10.0

# Qwen-Image, from ComfyUI's own templates for it (image_qwen_Image_2512.json and
# image_qwen_image.json, shipped in comfyui_workflow_templates).
#
# QWEN_CFG is a real CFG scale over a real negative branch — the same quantity Wan's
# is, and not the distilled guidance embedding FLUX rides on FluxGuidance. That costs
# two forward passes per step, which is why the step counts here are the models' own
# reference settings rather than something tuned down: at cfg 1 the negative branch
# does nothing and the output falls apart.
QWEN_STEPS = 50           # 2512's reference setting
QWEN_BASE_STEPS = 20      # the original's
QWEN_CFG = 4.0
# The edit models are separate checkpoints with their own published settings, and they
# are not the create ones' — 2511 runs longer and lower, the original shorter and lower
# still. Straight from ComfyUI's image_qwen_image_edit_2511.json / _edit.json.
QWEN_EDIT_STEPS, QWEN_EDIT_CFG = 40, 3.0
QWEN_EDIT1_STEPS, QWEN_EDIT1_CFG = 20, 2.5
# CFGNorm rescales the CFG result back to the conditional's norm. Every Qwen *edit*
# template ships it at 1.0 and none of the create ones do: at cfg 3-4 over a real
# uncond branch the combined prediction drifts hot, and an edit shows it as a colour
# shift against the source that a from-scratch generation has nothing to be judged
# against. Not applied to the create models, to stay with their published graphs.
QWEN_CFG_NORM = 1.0
# Qwen samples on a shifted sigma schedule that only ModelSamplingAuraFlow applies;
# without the patch the sampler runs on the wrong schedule and returns mush.
QWEN_SHIFT = 3.1
# Qwen's native resolution is 1328x1328 — ~1.76 MP, not FLUX's 1 MP.
QWEN_PIXELS = 1328 * 1328
# ComfyUI's 2512 template ships this negative prompt, and at cfg 4 the uncond branch
# is live so it is actually read (the base Qwen template ships an empty one). Roughly:
# "low resolution, low quality, deformed limbs, deformed fingers, oversaturated, waxy,
# featureless faces, over-smoothed, AI-looking, muddled composition, blurry or
# distorted text". This app exposes no negative prompt field, so it is a constant.
QWEN_NEGATIVE = ("低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，"
                 "人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲")
# Per checkpoint, because within this one family the published settings differ by more
# than the families do from each other — and since a bundle now ships two transformers
# with different answers, the key is the *weight file*, not the bundle. Anything not
# listed (a future Qwen checkpoint, or a user-added one) falls back to the create
# defaults, the safer of the two: too many steps costs time, too few costs the image.
# `flux.ts` mirrors both tables off the same filenames; they must not drift apart.
_QWEN_STEPS_BY_UNET = {
    "qwen_image_2512_fp8_e4m3fn.safetensors": QWEN_STEPS,
    "qwen_image_fp8_e4m3fn.safetensors": QWEN_BASE_STEPS,
    "qwen_image_edit_2511_fp8mixed.safetensors": QWEN_EDIT_STEPS,
    "qwen_image_edit_fp8_e4m3fn.safetensors": QWEN_EDIT1_STEPS,
}
_QWEN_CFG_BY_UNET = {
    "qwen_image_2512_fp8_e4m3fn.safetensors": QWEN_CFG,
    "qwen_image_fp8_e4m3fn.safetensors": QWEN_CFG,
    "qwen_image_edit_2511_fp8mixed.safetensors": QWEN_EDIT_CFG,
    "qwen_image_edit_fp8_e4m3fn.safetensors": QWEN_EDIT1_CFG,
}

# How the model lays out multiple reference images. See `_conditioning`.
REF_METHOD = "offset"

# --------------------------------------------------------------------------- #
# Structural control
# --------------------------------------------------------------------------- #
# The control map defaults, taken from ComfyUI's own blueprints for these nodes rather
# than picked here: `blueprints/Image Depth Estimation (Depth Anything 3).json` and
# `blueprints/Image to Pose Map (SDPose-OOD).json`.
DA3_MODEL = "depth_anything_3_mono_large.safetensors"
DA3_RESOLUTION = 504          # longest side the estimator runs at; must be a multiple of 14
SDPOSE_CKPT = "sdpose_wholebody_fp16.safetensors"
SDPOSE_BATCH = 16
SDPOSE_THRESHOLD = 0.5        # keypoint confidence below which a limb isn't drawn
CANNY_LOW, CANNY_HIGH = 0.3, 0.4

# How far down the sigma schedule a control generation starts. 1.0 means "start from
# noise": the control maps guide through ReferenceLatent and nothing constrains the
# geometry. Below 1.0 the sampler starts from the *source image* instead, so its limb
# geometry survives — the only hard structural constraint FLUX.2 offers, and the dial
# that matters for a pose the model won't otherwise hit. See `_control_graph`.
#
# The floor is 0.4 rather than `_strength`'s 0.05 because a control generation is meant
# to replace the subject and the setting: below ~0.4 the source's own appearance is
# still there and the prompt has stopped mattering, which is img2img with extra steps.
LOCK_MIN, LOCK_MAX = 0.4, 1.0
DEFAULT_LOCK = 1.0

# Fraction of the progress bar the preprocess pass owns. It runs as its own ComfyUI
# prompt (see `control`), so without a split the bar would fill once for the maps and
# then restart for the generation.
PREPROCESS_FRACTION = 0.2

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

# --------------------------------------------------------------------------- #
# Wan 2.2 I2V
# --------------------------------------------------------------------------- #
# These are lifted from ComfyUI's own template for this exact model
# (comfyui_workflow_templates_json/templates/video_wan2_2_14B_i2v.json), not from
# folklore. That workflow ships *two* sampling paths behind a switch: a 4-step, cfg-1
# one for the Lightning distillation LoRAs, and a 20-step, cfg-3.5 one for the base
# weights. The bundle doesn't include the LoRAs, so these are the base path's values.
WAN_STEPS = 20
# The step where the high-noise expert hands over to the low-noise one. This is the
# whole point of the MoE: the first expert lays down motion and composition out of
# noise, the second refines detail. It is a position in one shared 20-step schedule,
# not two separate jobs — see `_wan_i2v_graph`.
WAN_BOUNDARY = 10
# Wan has a real negative branch, unlike FLUX (cfg=1.0 + FluxGuidance), so this is a
# true CFG scale rather than a distilled guidance embedding.
WAN_CFG = 3.5
# Sigma shift. The template's value; the number most often quoted online is 8.0, which
# is Wan *2.1*'s and visibly over-smooths motion here.
WAN_SHIFT = 5.0
WAN_FPS = 16
WAN_SECONDS = 5.0

# Wan 2.2 I2V A14B was trained at 480p and 720p; this is the 720p budget. `vram_gb` on
# the bundle assumes it — raising it raises VRAM by roughly the square, times 81 frames.
WAN_PIXELS = 1280 * 720
WAN_MAX_EDGE = 1280

# SaveWEBM's own default is 32, which is visibly blocky on faces. 26 roughly doubles
# the file (~4 MB for 5s at 720p) and is worth it: the point of a 28 GB transformer is
# detail, and throwing it away in the encoder is the cheapest possible mistake.
WAN_CRF = 26.0

# A video job's timeouts are not an image job's. `_await`'s 600s recv default assumes a
# step every few seconds; the risk here is `sampler_lo`, which starts, then evicts the
# 28 GB high-noise expert and loads the 28 GB low-noise one *inside* the node while
# emitting nothing. The poll fallback needs to outlast the whole job, not one silence.
WAN_RECV_TIMEOUT = 1800
WAN_POLL_TIMEOUT = 5400

# Verbatim from the template's negative CLIPTextEncode. It's Chinese because Wan's
# umt5 conditioning was trained that way and this exact string is what the model was
# tuned against — a translation or a retyping is a different prompt, and measurably
# worse. Roughly: garish colour, overexposure, static, blurred detail, subtitles,
# worst/low quality, JPEG artifacts, malformed limbs, fused fingers, cluttered
# background, three legs, walking backwards.
WAN_NEGATIVE = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def _guidance(v, default: float) -> float:
    try:
        g = float(v)
    except (TypeError, ValueError):
        return default
    return g if GUIDANCE_MIN <= g <= GUIDANCE_MAX else default


def _steps(v, default: int = DEFAULT_STEPS) -> int:
    try:
        s = int(v)
    except (TypeError, ValueError):
        return default
    return s if STEPS_MIN <= s <= STEPS_MAX else default


def _strength(v) -> float:
    """img2img denoise: 0 = return the input, 1 = ignore it."""
    try:
        s = float(v)
    except (TypeError, ValueError):
        return 0.6
    return min(max(s, 0.05), 1.0)


def _lock(v) -> float:
    """Clamp a structure lock. Out-of-range or unparseable means "off" (1.0).

    Deliberately not `_strength`: that one floors at 0.05, which is right for img2img
    (where returning the input nearly unchanged is a legitimate ask) and wrong here.
    """
    try:
        s = float(v)
    except (TypeError, ValueError):
        return DEFAULT_LOCK
    return min(max(s, LOCK_MIN), LOCK_MAX)


def _control_scale(v) -> float | None:
    """Strength for the model's control adapter, or None to keep its saved weight.

    None rather than 1.0 for "unset": a control adapter's saved strength is a choice
    the user already made in the Image Models panel, and a request that says nothing
    about strength should not quietly overwrite it with a default.
    """
    try:
        s = float(v)
    except (TypeError, ValueError):
        return None
    return min(max(s, 0.0), 2.0)


def _canny_edge(v, default: float) -> float:
    """A Canny threshold. The node's own range; either side of it is a hard error there."""
    try:
        t = float(v)
    except (TypeError, ValueError):
        return default
    return t if 0.01 <= t <= 0.99 else default


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


def _area_resolution(pil, pixels: int) -> tuple[int, int]:
    """Fit an image's aspect ratio into a fixed pixel budget, on a multiple of 16.

    The shape ImageScaleToTotalPixels(area, pixels, steps=16) will produce. Mirrored
    here because the caller generally needs the sampled resolution up front — for
    FLUX.2, Flux2Scheduler shifts its sigma schedule by sequence length, so feeding it
    the *unscaled* size would shift the whole schedule wrong.
    """
    w, h = pil.size
    scale = math.sqrt(pixels / float(w * h or 1))
    return max(round(w * scale / 16) * 16, 16), max(round(h * scale / 16) * 16, 16)


def _flux2_resolution(pil) -> tuple[int, int]:
    return _area_resolution(pil, 1024 * 1024)


def _qwen_resolution(pil) -> tuple[int, int]:
    """The same area fit, at Qwen's larger native budget. Mirrors `_scale_node`'s
    ImageScaleToTotalPixels for the family, so the size reported up front is the size
    that is actually sampled."""
    return _area_resolution(pil, QWEN_PIXELS)


def _wan_resolution(pil) -> tuple[int, int]:
    """The 720p-budget shape a Wan I2V start frame is sampled at.

    An area fit rather than a nearest-aspect table like Kontext's, because Wan's
    published buckets straddle two tiers — 1280x720 and 832x480 — whose areas differ by
    2.4x. Nearest-by-aspect across both would let a 4:3 photo pick its *pixel budget* by
    accident, and on a video that budget is multiplied by 81 frames: the difference
    between fitting the card and not. Fixing the area and fitting the aspect into it
    keeps the token count, and the VRAM, the same whatever shape comes in.

    The clamp is for panoramas: a 3:1 image area-fits to ~1470x490, wider than anything
    Wan was trained on, so scale the long edge back to 1280 and let the area fall.
    """
    w, h = _area_resolution(pil, WAN_PIXELS)
    if max(w, h) > WAN_MAX_EDGE:
        s = WAN_MAX_EDGE / max(w, h)
        w, h = max(round(w * s / 16) * 16, 16), max(round(h * s / 16) * 16, 16)
    return w, h


def _source_resolution(unet, pil) -> tuple[int, int]:
    """The resolution an input image will be sampled at, per family."""
    fam = cat.family_of(unet)
    if fam == cat.FAMILY_WAN:
        return _wan_resolution(pil)
    if fam == cat.FAMILY_FLUX2:
        return _flux2_resolution(pil)
    # Only the create models get Qwen's larger budget; the edit ones run Kontext's
    # table, exactly as `_scale_node` wires them.
    if fam == cat.FAMILY_QWEN and not _qwen_edits(unet):
        return _qwen_resolution(pil)
    return _kontext_resolution(pil)


def _default_guidance(unet, role: str) -> float:
    fam = cat.family_of(unet)
    if fam == cat.FAMILY_WAN:
        return WAN_CFG
    if fam == cat.FAMILY_QWEN:
        return _QWEN_CFG_BY_UNET.get(os.path.basename(unet or ""), QWEN_CFG)
    if fam == cat.FAMILY_FLUX2:
        bundle = cat.bundle_of_unet(unet)
        if bundle and bundle["id"] == "flux2-klein-9b":
            return KLEIN_GUIDANCE
        return FLUX2_GUIDANCE
    return KONTEXT_GUIDANCE if role == ROLE_EDIT else CREATE_GUIDANCE


def _default_steps(unet) -> int:
    bundle = cat.bundle_of_unet(unet)
    if bundle and bundle["id"] == "flux2-klein-9b":
        return KLEIN_STEPS
    if cat.family_of(unet) == cat.FAMILY_QWEN:
        # Every Qwen checkpoint publishes its own settings and they are all different.
        # None of them is distilled, so cutting the count costs quality rather than
        # only time — that trade is the Lightning LoRAs' job, not the default's.
        return _QWEN_STEPS_BY_UNET.get(os.path.basename(unet or ""), QWEN_STEPS)
    if cat.family_of(unet) == cat.FAMILY_FLUX2:
        return FLUX2_STEPS
    return DEFAULT_STEPS


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
    # A sidecar we are about to start holds nothing, whatever an earlier one held —
    # which also means residency is knowable again from here on.
    global _residency_known
    _resident.clear()
    _residency_known = True
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
    # Nothing is loaded after this, so the next run pays the cold price again and the
    # progress bar should expect it (see `_resident`).
    global _residency_known
    _resident.clear()
    _residency_known = True
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


def _fetch_bytes(filename: str, subfolder: str) -> bytes:
    """An output file off the sidecar, as bytes. ComfyUI serves every output through
    /view regardless of what it is — a PNG and a WebM come back the same way."""
    q = urllib.parse.urlencode({"filename": filename, "subfolder": subfolder, "type": "output"})
    with urllib.request.urlopen(COMFY_URL + "/view?" + q) as r:
        return r.read()


def _fetch_output(filename: str, subfolder: str):
    from PIL import Image  # local import; Pillow is a backend dep

    return Image.open(io.BytesIO(_fetch_bytes(filename, subfolder))).convert("RGB")


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


def _check_encoder_layout(name: str) -> None:
    """Reject a FLUX.2 text encoder ComfyUI won't recognise, before sampling starts.

    ComfyUI identifies an encoder from its tensor *names* (`detect_te_model` in
    comfy/sd.py), and every FLUX.2 candidate — Mistral-3, Qwen3 — is keyed off
    `model.layers.0.…`. HuggingFace's multimodal layout names the same tensors
    `language_model.model.layers.0.…`; ComfyUI's only prefix remap is for
    `model.language_model.`, the other ordering, so nothing matches.

    Nothing errors when that happens. Detection falls through, a default CLIP-L is
    built from config, the 48 GB of real weights are discarded as unmatched, and the
    only symptom arrives minutes later inside the sampler as "mat1 and mat2 shapes
    cannot be multiplied (512x768 and 15360x6144)" — 768 being CLIP-L's width. This
    reads the header (tens of KB, not the weights) and says so up front instead.
    """
    if name.lower().endswith(".gguf"):
        return  # GGUF carries its architecture in metadata; detection can't misfire
    path = cat.TE_DIR / name
    if not path.exists():
        path = cat.CLIP_DIR / name
    if not path.exists():
        return  # missing entirely is ComfyUI's error to report, and it does it clearly
    try:
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            keys = json.loads(fh.read(n)).keys()
    except (OSError, ValueError, struct.error):
        return  # unreadable header: let ComfyUI be the one to complain
    if any(k.startswith("model.layers.0.") for k in keys):
        return
    nested = next((k for k in keys if k.startswith("language_model.model.layers.0.")), None)
    if nested:
        raise RuntimeError(
            f"{name} is in HuggingFace's multimodal layout (its tensors are named "
            f"'language_model.model.…'), which ComfyUI's FLUX.2 loader can't identify — "
            f"it would silently fall back to CLIP-L and fail mid-sample. Install the "
            f"ComfyUI-packaged encoder instead: in the Models panel, under Text "
            f"encoders, add "
            f"Comfy-Org/flux2-dev:split_files/text_encoders/{name}")
    raise RuntimeError(
        f"{name} doesn't look like a FLUX.2 text encoder — ComfyUI identifies one by "
        f"its tensor names and this file has none it recognises.")


def _dual_clip_node(name1: str, name2: str, kind: str) -> dict:
    """The same choice as `_clip_node`, for FLUX.1's two-encoder pair (CLIP-L + T5).

    The GGUF loader can read a plain safetensors too, but it forces GGML custom ops and
    a GGUF patcher onto a model with nothing quantized in it — so it's used only when
    one of the pair actually is a GGUF. Which one that is depends on the installed
    bundle, hence the check rather than a fixed class.
    """
    gguf = name1.lower().endswith(".gguf") or name2.lower().endswith(".gguf")
    cls = "DualCLIPLoaderGGUF" if gguf else "DualCLIPLoader"
    return {"class_type": cls,
            "inputs": {"clip_name1": name1, "clip_name2": name2, "type": kind}}


def _qwen_edits(unet: str) -> bool:
    """Whether this file is a Qwen *edit* transformer.

    Qwen splits the two jobs across separately fine-tuned checkpoints that now ship in
    one bundle, so the fork is per *file*, not per bundle or per role: the edit weights
    condition through TextEncodeQwenImageEdit* and take CFGNorm, the create weights use
    a plain CLIPTextEncode and don't.

    Deliberately not asked of `roles_of` — the edit half carries `create` too (it can
    generate from a bare prompt), so roles no longer identify which weights these are.
    Only the filename does.
    """
    b = cat.bundle_of_unet(unet)
    if not b or b["family"] != cat.FAMILY_QWEN:
        return False
    return os.path.basename(unet or "") == b.get("unet_edit")


def _qwen_edit_encode(unet: str, text: str, ref_images) -> dict:
    """One Qwen edit conditioning node: the instruction plus its reference images.

    This single node replaces the whole CLIPTextEncode -> ReferenceLatent chain the
    other families build. It tokenizes each image for the VL encoder at 384x384, VAE-
    encodes it at 1 MP as a reference latent, and prefixes the prompt with a
    "Picture N:" marker per image — none of which ReferenceLatent does, which is why
    feeding a Qwen edit model that chain instead produces an image that ignores its
    references.

    With no images at all this is still a valid text-to-image encode — every image
    input on the node is optional — which is what lets the edit half serve `create`.
    """
    inputs = {"clip": ["clip", 0], "prompt": text, "vae": ["vae", 0]}
    node = _qwen_edit_node(unet)
    if node == "TextEncodeQwenImageEdit":
        if ref_images:
            inputs["image"] = list(ref_images[0])
    else:
        for i, ref in enumerate(ref_images):
            inputs[f"image{i + 1}"] = list(ref)
    return {"class_type": node, "inputs": inputs}


def _qwen_edit_node(unet: str) -> str:
    """Which encode node this bundle's edit half uses."""
    return (cat.bundle_of_unet(unet) or {}).get("edit_encode") or "TextEncodeQwenImageEditPlus"


# How many reference images each encode node actually has inputs for. The original
# edit model takes one; 2511's "Plus" takes three. Hard limits of the node, not a
# tuning choice.
_QWEN_REF_LIMIT = {"TextEncodeQwenImageEdit": 1, "TextEncodeQwenImageEditPlus": 3}


def _check_qwen_refs(unet: str, ref_images) -> None:
    """Refuse more references than the encode node has inputs for.

    Compose and control both hand over as many images as the user attached, and this
    family is the first with a hard ceiling on that. Silently keeping the first few
    would be the worst outcome: the run succeeds, looks normal, and quietly ignores
    pictures the user deliberately attached — so say it up front instead.
    """
    limit = _QWEN_REF_LIMIT.get(_qwen_edit_node(unet), 3)
    if len(ref_images) <= limit:
        return
    raise RuntimeError(
        f"{_label(unet)} takes at most {limit} image{'s' if limit > 1 else ''} at once "
        f"and {len(ref_images)} were attached. Remove "
        f"{len(ref_images) - limit} of them, or switch to a FLUX.2 model, which has no "
        f"such limit.")


def _with_lora(unet: str, loaders: dict, control_scale: float | None = None) -> dict:
    """Chain the model's selected LoRAs onto its transformer, one node per adapter.

    Every graph builder here refers to the model as `["unet", 0]`. Rather than teach
    each of them about adapters, the raw loader moves to `unet_base` and `unet`
    becomes the *patched* output — so the whole pipeline picks the LoRAs up with no
    other change, and a graph with no LoRA is byte-for-byte what it was before.
    Multiple adapters chain: each `LoraLoaderModelOnly` patches the previous stage's
    output, so stacking two is just two nodes in a row.

    LoraLoaderModelOnly rather than LoraLoader: FLUX adapters patch the transformer,
    and the CLIP-patching variant demands a `clip` input that would also have to be
    rewired into every text-encode node for no gain.

    `control_scale` overrides the saved strength of adapters flagged `control`, and
    only those, for this one graph. A control generation needs a strength dial on the
    reference path — `ReferenceLatent` has no input for one — and the adapter's weight
    is the only lever there is. Scoped to the flagged adapters because scaling the
    whole stack would drag a character or style LoRA along with it, changing who is in
    the picture when the user only asked to change how hard the pose is held.
    """
    picks = loras_for(unet)
    if not picks:
        return loaders
    base = dict(loaders)
    base["unet_base"] = base.pop("unet")
    prev = ["unet_base", 0]
    for i, pick in enumerate(picks):
        key = "unet" if i == len(picks) - 1 else f"unet_lora_{i}"
        strength = pick["strength"]
        if control_scale is not None and pick.get("control"):
            strength = round(float(control_scale), 3)
        base[key] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": prev,
                       "lora_name": pick["name"], "strength_model": strength},
        }
        prev = [key, 0]
    return base


def _loaders(unet: str, control_scale: float | None = None) -> dict:
    """The transformer + its text encoder + its VAE. All three are family-specific:
    a FLUX.2 transformer decodes 128-channel latents through its own VAE and reads
    conditioning from a Mistral-3 encoder, none of which FLUX.1's parts can supply.

    `control_scale` is passed straight through to `_with_lora`; only `_control_graph`
    sets it.
    """
    if cat.family_of(unet) == cat.FAMILY_FLUX2:
        b = cat.bundle_of_unet(unet)
        clip = clip_for(b)
        _check_encoder_layout(clip)
        return _with_lora(unet, {
            "unet": _unet_node(unet),
            "clip": _clip_node(clip, "flux2"),
            "vae": {"class_type": "VAELoader", "inputs": {"vae_name": b["vae"]}},
        }, control_scale)
    if cat.family_of(unet) == cat.FAMILY_QWEN:
        b = cat.bundle_of_unet(unet)
        # The model chain upstream of any LoRA, in ComfyUI's own order for this family:
        # loader -> ModelSamplingAuraFlow -> (edit only) CFGNorm. `_with_lora` renames
        # whatever is called `unet` to `unet_base` and chains the adapters onto it, so
        # naming the last patch `unet` puts the LoRAs after these — which is where the
        # Qwen templates put them.
        g = {
            "unet_raw": _unet_node(unet),
            # One encoder, not a pair: ComfyUI's `qwen_image` CLIP type reads
            # Qwen2.5-VL directly. No `_check_encoder_layout` — that check is keyed to
            # FLUX.2's detection path and would reject a perfectly good Qwen encoder.
            "clip": _clip_node(b["clip"], "qwen_image"),
            "vae": {"class_type": "VAELoader", "inputs": {"vae_name": b["vae"]}},
        }
        shift = {"class_type": "ModelSamplingAuraFlow",
                 "inputs": {"model": ["unet_raw", 0], "shift": QWEN_SHIFT}}
        if _qwen_edits(unet):
            g["unet_shift"] = shift
            g["unet"] = {"class_type": "CFGNorm",
                         "inputs": {"model": ["unet_shift", 0], "strength": QWEN_CFG_NORM}}
        else:
            g["unet"] = shift
        return _with_lora(unet, g, control_scale)
    return _with_lora(unet, {
        "unet": _unet_node(unet),
        "clip": _dual_clip_node(CLIP_L, T5, "flux"),
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": FLUX1_VAE}},
    }, control_scale)


def _sampler(g: dict, unet, latent_src, steps, seed, width, height, denoise=1.0,
             guidance=1.0) -> None:
    """Add the sampling nodes, writing the output latent to `g["sampler"]` output 0.

    FLUX.1 samples through KSampler at cfg=1.0 — the negative branch is unused, which
    is why this app exposes no negative prompt; guidance rides on FluxGuidance instead.

    Qwen shares that KSampler but not that arrangement: it reads no guidance embedding,
    so the scale has to be a real cfg over a real negative branch. Hence `guidance`,
    which every other family ignores.

    FLUX.2 has no negative branch at all (a BasicGuider, not a CFG pair) and needs a
    sequence-length-aware sigma schedule, which only Flux2Scheduler computes — the
    stock "simple" schedule is wrong for it. That combination only exists on the
    custom-sampler path, so it assembles RandomNoise + BasicGuider + KSamplerSelect +
    Flux2Scheduler into SamplerCustomAdvanced.
    """
    if cat.family_of(unet) != cat.FAMILY_FLUX2:
        cfg = guidance if cat.family_of(unet) == cat.FAMILY_QWEN else 1.0
        g["sampler"] = {
            "class_type": "KSampler",
            "inputs": {"model": ["unet", 0], "positive": ["guide", 0], "negative": ["neg", 0],
                       "latent_image": list(latent_src), "seed": seed, "steps": steps, "cfg": cfg,
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


def _qwen_edit_conditioning(unet, prompt: str, ref_images) -> dict:
    """Qwen's edit conditioning: one encode node per branch, positive and negative.

    The negative is the same node over the same images with an empty instruction,
    which is what every Qwen edit template ships — the uncond branch has to see the
    references too, or CFG pulls the result away from the image it is meant to be
    editing.

    2511 adds a layout choice on top, for the same reason Kontext has one: with more
    than one reference the model needs to know how they are arranged. Its own template
    pins "index_timestep_zero" and so does the bundle.
    """
    _check_qwen_refs(unet, ref_images)
    b = cat.bundle_of_unet(unet)
    g = {"pos": _qwen_edit_encode(unet, prompt, ref_images),
         "neg_enc": _qwen_edit_encode(unet, "", ref_images)}
    method = (b or {}).get("ref_method")
    if method and len(ref_images) > 1:
        g["guide"] = {"class_type": "FluxKontextMultiReferenceLatentMethod",
                      "inputs": {"conditioning": ["pos", 0], "reference_latents_method": method}}
        g["neg"] = {"class_type": "FluxKontextMultiReferenceLatentMethod",
                    "inputs": {"conditioning": ["neg_enc", 0], "reference_latents_method": method}}
        return g
    g["guide"] = g.pop("pos")
    g["neg"] = g.pop("neg_enc")
    return g


def _conditioning(unet, prompt: str, guidance: float, ref_latents=(), ref_images=()) -> dict:
    """Text conditioning shared by every graph. `guide` is what the sampler reads.

    `ref_latents` chains one `ReferenceLatent` node per encoded reference image. The
    node *appends* to the conditioning and the model forwards the whole list, so each
    reference arrives as its own token block with its own RoPE offsets — the images
    stay distinct. FLUX.2 and Kontext both understand this; a plain FLUX.1 dev
    transformer ignores it, so create graphs pass nothing.

    `ref_images` is the same set of references *before* they were encoded, and only
    Qwen's edit models use it: their encode node takes pixels, tokenizes them for the
    VL encoder and does its own VAE pass, so it needs the image, not the latent. Every
    caller that builds references through `_encode_image` has both to hand.

    Qwen's create models return early instead: their negative branch is live (see
    `_sampler`) so it gets real negative text, and there is no FluxGuidance node to end
    on — `guide` is simply the positive encode.
    """
    if _qwen_edits(unet):
        return _qwen_edit_conditioning(unet, prompt, ref_images)
    flux2 = cat.family_of(unet) == cat.FAMILY_FLUX2
    g = {"pos": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["clip", 0]}}}
    if not flux2:
        g["neg"] = {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["clip", 0]}}
    if cat.family_of(unet) == cat.FAMILY_QWEN:
        g["neg"]["inputs"]["text"] = QWEN_NEGATIVE
        g["guide"] = g.pop("pos")
        return g
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
    decode to noise); FLUX.2's is 128-channel and has its own node.

    Qwen falls in with FLUX.1 here, and does so correctly rather than by accident: its
    latent format is Wan21 (comfy/supported_models.py), which is also 16-channel.
    """
    cls = ("EmptyFlux2LatentImage" if cat.family_of(unet) == cat.FAMILY_FLUX2
           else "EmptySD3LatentImage")
    return {"class_type": cls, "inputs": {"width": width, "height": height, "batch_size": 1}}


def _scale_node(unet, src) -> dict:
    """Snap an input image to a resolution its family was trained on. Kontext has a
    fixed table of ~1 MP shapes; FLUX.2 just wants ~1 MP on a multiple of 16 (its VAE
    downscale), which keeps the token count — and the VRAM — bounded either way.

    Qwen's *create* models take the same area fit at their own, larger budget: they are
    trained at 1328x1328 and squeezing a source into Kontext's 1 MP table throws away
    resolution they can use. Its *edit* models go the other way and use Kontext's table
    outright, which is what their own templates do — TextEncodeQwenImageEdit* rescales
    every reference to 1 MP internally anyway, so a larger input buys nothing and only
    puts the sampled latent out of step with the references.
    """
    fam = cat.family_of(unet)
    if fam == cat.FAMILY_QWEN and not _qwen_edits(unet):
        return {"class_type": "ImageScaleToTotalPixels",
                "inputs": {"image": list(src), "upscale_method": "area",
                           "megapixels": round(QWEN_PIXELS / 1e6, 2), "resolution_steps": 16}}
    if fam == cat.FAMILY_FLUX2:
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
    _sampler(g, unet, ("latent", 0), steps, seed, width, height, guidance=guidance)
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
    _sampler(g, unet, ("enc", 0), steps, seed, width, height, denoise=strength,
             guidance=guidance)
    g.update(_tail(prefix))
    return g


def _encode_image(g, unet, name, key):
    """LoadImage -> snap to a resolution the family was trained on -> VAEEncode.
    Returns the latent output ref."""
    g[f"{key}_img"] = {"class_type": "LoadImage", "inputs": {"image": name}}
    g[f"{key}_scale"] = _scale_node(unet, (f"{key}_img", 0))
    g[key] = {"class_type": "VAEEncode", "inputs": {"pixels": [f"{key}_scale", 0], "vae": ["vae", 0]}}
    return (key, 0)


def _scaled_image(latent_ref):
    """The scaled *pixels* behind an `_encode_image` latent, for the consumers that
    want the image rather than its encoding — Qwen's edit nodes, which VAE-encode
    their references themselves. Derived here so only one place knows the key
    layout `_encode_image` writes."""
    return (f"{latent_ref[0]}_scale", 0)


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
    g.update(_conditioning(unet, prompt, guidance, ref_latents=refs,
                           ref_images=[_scaled_image(r) for r in refs]))
    _sampler(g, unet, scene, steps, seed, width, height, guidance=guidance)
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
    g.update(_conditioning(unet, prompt, guidance, ref_latents=refs,
                           ref_images=[_scaled_image(r) for r in refs]))
    g["latent"] = _empty_latent(unet, width, height)
    _sampler(g, unet, ("latent", 0), steps, seed, width, height, guidance=guidance)
    g.update(_tail(prefix))
    return g


# --------------------------------------------------------------------------- #
# Control: preprocess graphs
# --------------------------------------------------------------------------- #
# These turn a source image into a control map. They run as their own ComfyUI prompt,
# separate from the generation that consumes the map — see `control` for why.
#
# All three snap the source through `_scale_node` first, for the same reason every
# other image path here does: the map is about to be VAE-encoded as a reference at that
# family's resolution, and preprocessing at some other size only to rescale afterwards
# throws away detail in the one signal the whole mode depends on.
#
# Each ends in SaveImage, so `_run` reads the result back with no special casing.


def _canny_graph(image_name: str, low: float, high: float, unet: str) -> dict:
    """Edge map. Needs no weights at all — ComfyUI's `Canny` is a kornia filter.

    The hardest structural signal of the three and the only one that is always
    available, but it carries style as well as structure: every edge in the source,
    including the ones that describe its clothing and its background, is handed to the
    model as something to reproduce.
    """
    return {
        "img": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "scale": _scale_node(unet, ("img", 0)),
        "map": {"class_type": "Canny",
                "inputs": {"image": ["scale", 0],
                           "low_threshold": low, "high_threshold": high}},
        "save": {"class_type": "SaveImage",
                 "inputs": {"images": ["map", 0], "filename_prefix": "control_canny"}},
    }


def _depth_graph(image_name: str, unet: str) -> dict:
    """Depth map, via Depth Anything 3.

    The one that answers the question a skeleton can't: an OpenPose figure says where
    the limbs are and nothing about what they are resting on, so a subject asked to sit
    on a chair or stand on top of a closet floats. A depth map carries the chair, the
    closet, the subject and the contact between them in one signal — and it resolves
    limb ordering, which is what breaks first on a hard pose.

    `mode` and `output` are DynamicCombo inputs: in ComfyUI's API format their nested
    options are flat, dot-prefixed sibling keys (`output.normalization`), not a nested
    object. `mode: "mono"` has no nested options of its own.
    """
    return {
        "img": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "scale": _scale_node(unet, ("img", 0)),
        "da3": {"class_type": "LoadDA3Model",
                "inputs": {"model_name": DA3_MODEL, "weight_dtype": "default"}},
        "geo": {"class_type": "DA3Inference",
                "inputs": {"da3_model": ["da3", 0], "image": ["scale", 0],
                           "resolution": DA3_RESOLUTION,
                           "resize_method": "upper_bound_resize", "mode": "mono"}},
        "map": {"class_type": "DA3Render",
                "inputs": {"da3_geometry": ["geo", 0], "output": "depth",
                           "output.normalization": "v2_style",
                           "output.apply_sky_clip": False}},
        "save": {"class_type": "SaveImage",
                 "inputs": {"images": ["map", 0], "filename_prefix": "control_depth"}},
    }


def _pose_graph(image_name: str, unet: str) -> dict:
    """OpenPose skeleton, via SDPose.

    Limb configuration only — pair it with depth rather than using it alone, because on
    its own it says nothing about the scene the pose happens in.

    SDPose is an SD-architecture checkpoint whose keypoints are read out of a UNet
    feature map, so it needs a MODEL *and* the matching VAE: `CheckpointLoaderSimple`
    returns (MODEL, CLIP, VAE) and outputs 0 and 2 are the two this wants.

    Single-person. Multi-person detection needs the optional `bboxes` input fed from
    an RT-DETR detector (`rt_detr_v4-x-hgnet_fp16.safetensors` + `RTDETR_detect`) —
    a second model to install and a second thing to fail, left out until one control
    map per person is something the UI can express.
    """
    return {
        "img": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "scale": _scale_node(unet, ("img", 0)),
        "ck": {"class_type": "CheckpointLoaderSimple",
               "inputs": {"ckpt_name": SDPOSE_CKPT}},
        "kp": {"class_type": "SDPoseKeypointExtractor",
               "inputs": {"model": ["ck", 0], "vae": ["ck", 2], "image": ["scale", 0],
                          "batch_size": SDPOSE_BATCH}},
        "map": {"class_type": "SDPoseDrawKeypoints",
                "inputs": {"keypoints": ["kp", 0],
                           "draw_body": True, "draw_hands": True, "draw_face": True,
                           "draw_feet": True, "draw_head": True,
                           "stick_width": 4, "face_point_size": 2,
                           "score_threshold": SDPOSE_THRESHOLD}},
        "save": {"class_type": "SaveImage",
                 "inputs": {"images": ["map", 0], "filename_prefix": "control_pose"}},
    }


def _preprocess_graph(kind: str, image_name: str, unet: str,
                      canny_low: float = CANNY_LOW, canny_high: float = CANNY_HIGH) -> dict:
    if kind == "canny":
        return _canny_graph(image_name, canny_low, canny_high, unet)
    if kind == "depth":
        return _depth_graph(image_name, unet)
    if kind == "pose":
        return _pose_graph(image_name, unet)
    raise ValueError(f"Unknown control type '{kind}'.")


def _control_graph(control_names, ref_names, source_name, lock, control_scale, prompt,
                   width, height, steps, guidance, seed, prefix, unet):
    """Generate from one or more control maps.

    Structural control here is *not* a ControlNet, and shouldn't be turned into one
    without checking what family is running. ComfyUI has ControlNet loaders for FLUX.1
    only (`comfy/controlnet.py`: xlabs, mistoline, InstantX) — there is no FLUX.2
    branch, no `comfy/ldm/flux2/controlnet.py`, and no published FLUX.2 ControlNet
    weights to load into one. Wiring `ControlNetApplyAdvanced` into this graph on a
    FLUX.2 model does not degrade, it fails.

    What FLUX.2 does understand is a reference image, so each control map is chained in
    as its own `ReferenceLatent` — the same mechanism `_edit_graph` and `_compose_graph`
    use for subject references. A control-adapter LoRA (flagged in the model's LoRA
    picks) is what turns "here is an image" into "match this structure"; without one
    the maps still bias the composition, just more loosely.

    Maps come before subject references in the chain: the structural signal should
    anchor the layout, and whatever arrives later reads as an addition to it.

    `lock` is where the sampler starts:

      1.0  — an empty latent. The maps guide, nothing constrains; composition is free.
      <1.0 — the *source image's* latent, denoised from `lock` down. Its geometry
             survives while the prompt repaints subject, style and setting over it.
             This is `_img2img_graph`'s mechanism with the maps riding along, and it
             is the only hard geometric constraint available on this family.

    The cost of a low lock is that the source's appearance survives too, not only its
    geometry — which is why it is a dial the user sweeps rather than a fixed value, and
    why depth is the map to pair it with: a depth map has no appearance to leak.
    """
    g = _loaders(unet, control_scale=control_scale)
    refs = ([_encode_image(g, unet, n, f"ctl{i}") for i, n in enumerate(control_names)]
            + [_encode_image(g, unet, n, f"src{i}") for i, n in enumerate(ref_names)])
    if not refs:
        raise ValueError("A control generation needs at least one control map.")
    g.update(_conditioning(unet, prompt, guidance, ref_latents=refs,
                           ref_images=[_scaled_image(r) for r in refs]))
    if lock < 1.0:
        if not source_name:
            raise ValueError("Structure lock needs the source image it locks to.")
        latent = _encode_image(g, unet, source_name, "lock")
    else:
        g["latent"] = _empty_latent(unet, width, height)
        latent = ("latent", 0)
    _sampler(g, unet, latent, steps, seed, width, height, denoise=min(lock, 1.0),
             guidance=guidance)
    g.update(_tail(prefix))
    return g


def _wan_length(seconds, fps: int = WAN_FPS) -> int:
    """Frames to sample for a clip of `seconds`, on the lattice Wan requires.

    Wan's VAE packs 4 frames into each latent step, plus the start frame, so the frame
    count must be 4n+1 — `WanImageToVideo` builds a latent of ((length-1)//4)+1 and
    anything off the lattice is silently truncated rather than rejected. ComfyUI's own
    template computes `floor(seconds * fps + 1)`, which lands on it exactly for whole
    seconds; this rounds explicitly so a fractional request can't drift off.

    Capped at 5s: past that Wan loses the thread — the motion stops matching the prompt
    and the subject drifts — and the attention cost is quadratic in frames.
    """
    try:
        s = float(seconds)
    except (TypeError, ValueError):
        s = WAN_SECONDS
    if not 0 < s <= WAN_SECONDS:
        s = WAN_SECONDS
    frames = int(s * fps) + 1
    return max(((frames - 1) // 4) * 4 + 1, 5)


def _wan_boundary(steps: int) -> int:
    """The step the high-noise expert hands over on, for a schedule of `steps`.

    Half, which is what both of the template's presets are (20→10 and 4→2). Derived
    rather than pinned at WAN_BOUNDARY because the step count is user-settable down to
    8: a fixed 10 would sit past the end of a short schedule, so the high-noise expert
    would run the whole thing and the low-noise one — the expert that puts the detail
    in — would silently never run at all.
    """
    return max(1, min(steps - 1, steps // 2))


def _wan_i2v_graph(image_name, prompt, width, height, length, steps, boundary, cfg,
                   seed, prefix, b):
    """Wan 2.2 image-to-video: one start frame plus a prompt, out to a webm.

    Built here rather than through `_loaders`/`_sampler`/`_conditioning` because it
    shares no structure with them, only vocabulary. Three things differ in kind:

    * Two transformers, not one. Wan 2.2 A14B is a mixture of experts that splits by
      *noise level*, not by role — both run in this one graph, the high-noise one over
      the first half of the schedule and the low-noise one over the rest. Every FLUX
      builder hard-codes a single `["unet", 0]` as the model.
    * The samplers can't read the text encoders. `WanImageToVideo` rewrites both
      conditionings (it attaches the encoded start frame as `concat_latent_image` plus a
      mask), so the sampler must read *its* outputs — the conditioning is data flowing
      through a node, not a chain assembled around one.
    * Real CFG with a real negative branch, where FLUX.1 runs cfg=1.0 and FLUX.2 has no
      negative branch at all.

    Node ids are load-bearing: `_run_video` reads outputs by id and `_wan_progress`
    tells the two samplers apart by id.
    """
    g = {
        "unet_hi": _unet_node(b["unet"]),
        "unet_lo": _unet_node(b["unet_low"]),
        "clip": _clip_node(b["clip"], "wan"),
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": b["vae"]}},
    }

    # Shift the sigma schedule toward high noise. Wan is trained with it; without this
    # node the motion comes out mushy and the first frames barely move.
    for tag in ("hi", "lo"):
        g[f"shift_{tag}"] = {"class_type": "ModelSamplingSD3",
                             "inputs": {"model": [f"unet_{tag}", 0], "shift": WAN_SHIFT}}

    g["pos"] = {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["clip", 0]}}
    g["neg"] = {"class_type": "CLIPTextEncode",
                "inputs": {"text": WAN_NEGATIVE, "clip": ["clip", 0]}}

    g["img"] = {"class_type": "LoadImage", "inputs": {"image": image_name}}
    # WanImageToVideo will resize to width/height itself, but with a bilinear filter,
    # which aliases badly coming down from a 2048px stored original. Prescaling by area
    # to the same budget hands it something it only has to nudge — the same reasoning
    # `_scale_node` applies for FLUX.2. Not `_scale_node` itself: that falls through to
    # FluxKontextImageScale, whose resolution table is meaningless here.
    g["scale"] = {"class_type": "ImageScaleToTotalPixels",
                  "inputs": {"image": ["img", 0], "upscale_method": "area",
                             "megapixels": round(width * height / 1e6, 4),
                             "resolution_steps": 16}}

    # Encodes the start frame into both conditionings and sizes the empty latent.
    # `clip_vision_output` is left off deliberately: it's optional, and Wan 2.2 I2V
    # doesn't use it (Wan 2.1 I2V did).
    g["i2v"] = {"class_type": "WanImageToVideo",
                "inputs": {"positive": ["pos", 0], "negative": ["neg", 0], "vae": ["vae", 0],
                           "width": width, "height": height, "length": length,
                           "batch_size": 1, "start_image": ["scale", 0]}}

    # One 20-step schedule run by two models. `steps` is the *shared* total in both
    # nodes — that's what makes start/end_at_step slice one schedule rather than
    # describe two. The first keeps its leftover noise for the second to finish, and
    # the second must not re-noise what it's handed.
    g["sampler_hi"] = {
        "class_type": "KSamplerAdvanced",
        "inputs": {"model": ["shift_hi", 0], "add_noise": "enable", "noise_seed": seed,
                   "steps": steps, "cfg": cfg, "sampler_name": "euler", "scheduler": "simple",
                   "positive": ["i2v", 0], "negative": ["i2v", 1], "latent_image": ["i2v", 2],
                   "start_at_step": 0, "end_at_step": boundary,
                   "return_with_leftover_noise": "enable"},
    }
    g["sampler_lo"] = {
        "class_type": "KSamplerAdvanced",
        "inputs": {"model": ["shift_lo", 0], "add_noise": "disable", "noise_seed": seed,
                   "steps": steps, "cfg": cfg, "sampler_name": "euler", "scheduler": "simple",
                   "positive": ["i2v", 0], "negative": ["i2v", 1],
                   "latent_image": ["sampler_hi", 0],
                   "start_at_step": boundary, "end_at_step": 10000,
                   "return_with_leftover_noise": "disable"},
    }

    g["dec"] = {"class_type": "VAEDecode", "inputs": {"samples": ["sampler_lo", 0],
                                                      "vae": ["vae", 0]}}
    g["video"] = {"class_type": "SaveWEBM",
                  "inputs": {"images": ["dec", 0], "filename_prefix": prefix,
                             "codec": "vp9", "fps": float(WAN_FPS), "crf": WAN_CRF}}
    # The sidebar wants a JPEG thumbnail and there's no PIL image in a video. Pull the
    # first frame out here, where it's already decoded in VRAM — the alternative is
    # decoding webm in the backend, which has no media stack and shouldn't grow one.
    g["first"] = {"class_type": "ImageFromBatch",
                  "inputs": {"image": ["dec", 0], "batch_index": 0, "length": 1}}
    g["thumb"] = {"class_type": "SaveImage",
                  "inputs": {"images": ["first", 0], "filename_prefix": prefix + "_thumb"}}
    return g


# --------------------------------------------------------------------------- #
# Progress
# --------------------------------------------------------------------------- #
# What a generation spends its time on is the *graph*, not the sampler. ComfyUI
# executes a graph node by node and says which node it is on; sampling is one of
# those nodes, and on a cold run it is not the slow one — reading a 32 GB
# transformer off disk is. So each node is priced in seconds, and progress is the
# share of the graph's total price that is finished: nodes ComfyUI reports as
# executed (or cached, which is what a warm re-run reports for the loaders) count
# in full, the running node counts for as much of itself as it has reported.
#
# The prices below are only a cold start. Every node is timed as it runs and the
# result is folded into `data/gen_timing.json`, so after a run or two the weights
# are measurements from this machine rather than these guesses.
#
# They are not what the node names suggest, because ComfyUI's loaders are lazy: the
# *Loader nodes hand back a patcher in milliseconds and the weights are read and
# staged by the first node that actually uses them. Measured on a cold FLUX.2 Klein
# run here: CLIPLoader 0.3s but CLIPTextEncode 115s, and the transformer's own load
# lands inside the sampler. The one exception is the GGUF loader, which dequantizes
# in the node.
_NODE_SECONDS = {
    "UNETLoader": 3.0, "UnetLoaderGGUF": 40.0,
    "CLIPLoader": 3.0, "DualCLIPLoader": 3.0, "CLIPLoaderGGUF": 20.0,
    "VAELoader": 1.0,
    "LoraLoaderModelOnly": 5.0, "LoraLoader": 5.0,
    "ModelSamplingSD3": 0.5, "ModelSamplingAuraFlow": 0.5,
    "CFGNorm": 0.3,
    # Where the text encoder is really loaded, on the run that loads it.
    "CLIPTextEncode": 60.0,
    # Qwen's edit encode does the same encoder load, and a VAE pass per reference on
    # top of it — it is the whole conditioning chain in one node.
    "TextEncodeQwenImageEdit": 65.0, "TextEncodeQwenImageEditPlus": 65.0,
    "LoadImage": 0.5, "VAEEncode": 2.0, "WanImageToVideo": 4.0,
    "VAEDecode": 3.0, "SaveImage": 0.5, "SaveWEBM": 10.0,
    # Control preprocessors. Priced here rather than left to `_OTHER_NODE_SECONDS`
    # (0.3s) because a preprocess pass is a whole graph made of these: costed at the
    # default the bar would jump to full and then sit there for the ten seconds the
    # pose extractor actually takes. Unlike the *Loader nodes above, these two do read
    # their weights in the node — nothing downstream of them touches a patcher — so
    # the load lands on the loader for once.
    "LoadDA3Model": 6.0, "DA3Inference": 8.0, "DA3Render": 0.5,
    "CheckpointLoaderSimple": 8.0,
    "SDPoseKeypointExtractor": 10.0, "SDPoseDrawKeypoints": 0.5,
    "Canny": 1.0,
}
_OTHER_NODE_SECONDS = 0.3
# Per sampler step, multiplied by the steps that node actually runs. High for a step
# because the transformer's load is inside the first one on a cold run.
_STEP_SECONDS = 6.0
_SAMPLERS = ("KSampler", "KSamplerAdvanced", "SamplerCustomAdvanced")

# The file names that identify *which* model a node loads. A 6 GB Klein and a 32 GB
# FLUX.2 both load through UNETLoader and are a minute apart, so timings are keyed by
# the file, not by the node class.
_MODEL_INPUTS = ("unet_name", "clip_name", "clip_name1", "vae_name", "lora_name",
                 # The control preprocessors' equivalents. Both load in the node, so
                 # they benefit from the same per-file timing as the transformers.
                 "ckpt_name", "model_name")

# How often the bar advances while ComfyUI is silent (seconds).
_TICK = 1.0

_STAGE_LABELS = {
    "UNETLoader": "Loading the model", "UnetLoaderGGUF": "Loading the model",
    "CLIPLoader": "Loading the text encoder", "DualCLIPLoader": "Loading the text encoder",
    "CLIPLoaderGGUF": "Loading the text encoder",
    "VAELoader": "Loading the VAE",
    "LoraLoaderModelOnly": "Applying the LoRA", "LoraLoader": "Applying the LoRA",
    "CLIPTextEncode": "Reading the prompt", "FluxGuidance": "Reading the prompt",
    "TextEncodeQwenImageEdit": "Reading the prompt and images",
    "TextEncodeQwenImageEditPlus": "Reading the prompt and images",
    "ReferenceLatent": "Reading the reference images",
    "FluxKontextMultiReferenceLatentMethod": "Reading the reference images",
    "LoadImage": "Reading the source image", "VAEEncode": "Encoding the source image",
    "FluxKontextImageScale": "Sizing the source image",
    "ImageScaleToTotalPixels": "Sizing the source image",
    "EmptySD3LatentImage": "Preparing the canvas",
    "EmptyFlux2LatentImage": "Preparing the canvas",
    "Flux2Scheduler": "Preparing the schedule", "SplitSigmas": "Preparing the schedule",
    "RandomNoise": "Preparing the noise", "BasicGuider": "Preparing the sampler",
    "KSamplerSelect": "Preparing the sampler", "ModelSamplingSD3": "Preparing the sampler",
    "ModelSamplingAuraFlow": "Preparing the sampler", "CFGNorm": "Preparing the sampler",
    "ModelSamplingAuraFlow": "Preparing the sampler",
    "WanImageToVideo": "Preparing the frames",
    "KSampler": "Generating", "KSamplerAdvanced": "Generating",
    "SamplerCustomAdvanced": "Generating",
    "VAEDecode": "Decoding the image",
    "SaveImage": "Saving", "SaveWEBM": "Encoding the video",
    "LoadDA3Model": "Loading the depth model", "DA3Inference": "Reading the scene depth",
    "DA3Render": "Drawing the depth map",
    "CheckpointLoaderSimple": "Loading the pose model",
    "SDPoseKeypointExtractor": "Finding the pose",
    "SDPoseDrawKeypoints": "Drawing the skeleton",
    "Canny": "Tracing the edges",
}

_TIMING_PATH = Path(__file__).resolve().parent / "data" / "gen_timing.json"
_timing_mu = threading.Lock()
_timing_cache: dict | None = None

# Model files ComfyUI is known to have loaded already. The same node is two orders of
# magnitude apart on either side of this — 115s to stage a 15 GB encoder, 0.2s to reuse
# the staged one — so a single average of the two would describe neither, and every
# timing is filed as cold or warm accordingly.
#
# Best-effort by construction: it is this process's memory of what it asked for, while
# residency is ComfyUI's to decide (it evicts under VRAM pressure). Wrong here costs
# pacing, never correctness — the bar is still driven by what the graph reports.
_resident: set[str] = set()

# Whether that memory means anything yet. A backend that restarts against a sidecar
# left running has no idea what it is holding, and a warm run filed as cold would
# teach the store that a 15 GB encoder loads in 200ms. Such a run is still measured —
# it just isn't allowed to write anything down.
_residency_known = False


def _model_inputs(graph: dict) -> set[str]:
    """Every model file this graph names."""
    return {v for node in graph.values()
            for k, v in (node.get("inputs") or {}).items()
            if k in _MODEL_INPUTS and isinstance(v, str)}


def _timings() -> dict:
    """Measured node durations from previous runs. Seconds, keyed by `_cost_key`."""
    global _timing_cache
    if _timing_cache is None:
        try:
            loaded = json.loads(_TIMING_PATH.read_text())
            _timing_cache = loaded if isinstance(loaded, dict) else {}
        except (OSError, ValueError):
            _timing_cache = {}
    return _timing_cache


def _remember_timings(measured: dict[str, float]) -> None:
    """Fold one run's durations into the store, as an EMA so a cold outlier fades.

    Best-effort: a progress bar is not worth failing a finished generation over.
    """
    if not measured:
        return
    with _timing_mu:
        store = _timings()
        for key, seconds in measured.items():
            prev = store.get(key)
            store[key] = seconds if not isinstance(prev, (int, float)) else prev * 0.6 + seconds * 0.4
        try:
            _TIMING_PATH.parent.mkdir(parents=True, exist_ok=True)
            _TIMING_PATH.write_text(json.dumps(store))
        except OSError:
            pass


def _cost_key(graph: dict, nid: str, depth: int = 0) -> str:
    """The identity a node's duration is remembered under.

    Keyed by the model it works on, not just its class: a 9B Klein and a 32B FLUX.2
    both encode through `CLIPTextEncode` and are two minutes apart, and that node is
    where the encoder is loaded (see `_NODE_SECONDS`). A node that names no model
    inherits the one it reads from, which is how the encode and the decode end up
    filed under the encoder and the transformer they really wait on.
    """
    node = graph.get(nid) or {}
    cls = node.get("class_type", "?")
    ins = node.get("inputs") or {}
    for k in _MODEL_INPUTS:
        if isinstance(ins.get(k), str):
            return _keyed(cls, ins[k])
    if cls in _SAMPLERS:
        # Per *step*, not per node: the step count is the user's to change, so a
        # 40-step run must not teach the bar that sampling takes twice as long.
        return _keyed("step", _graph_unet(graph))
    if depth < 3:
        for v in ins.values():
            if isinstance(v, (list, tuple)) and len(v) == 2 and v[0] in graph:
                upstream = _cost_key(graph, v[0], depth + 1)
                if ":" in upstream:
                    return f"{cls}:{upstream.split(':', 1)[1]}"
    return cls


def _keyed(prefix: str, model: str) -> str:
    return f"{prefix}:{model}#{'warm' if model in _resident else 'cold'}"


def _graph_unet(graph: dict) -> str:
    for node in graph.values():
        name = (node.get("inputs") or {}).get("unet_name")
        if isinstance(name, str):
            return name
    return "?"


def _sigma_steps(graph: dict, ref) -> int:
    """How many steps a sigma schedule holds, following SplitSigmas back to its source.

    SamplerCustomAdvanced takes no step count of its own — the schedule it is handed
    is what decides how long it runs (see `_sampler`), so the count is read from there.
    """
    if not (isinstance(ref, (list, tuple)) and len(ref) == 2 and ref[0] in graph):
        return DEFAULT_STEPS
    node = graph[ref[0]]
    ins = node.get("inputs") or {}
    if node.get("class_type") == "SplitSigmas":
        base = _sigma_steps(graph, ins.get("sigmas"))
        cut = int(ins.get("step", 0) or 0)
        # Output 1 is the tail the partial-denoise path samples; output 0 is the head.
        return max(1, base - cut if ref[1] == 1 else cut)
    try:
        return max(1, int(ins.get("steps", DEFAULT_STEPS)))
    except (TypeError, ValueError):
        return DEFAULT_STEPS


def _sampler_steps(graph: dict, nid: str) -> int:
    """Steps this sampler node runs — 0 if it isn't a sampler.

    Not simply `inputs["steps"]`: KSampler at denoise < 1 starts part-way down the
    schedule, and the two Wan experts each run a *slice* of one shared schedule.
    """
    node = graph.get(nid) or {}
    cls = node.get("class_type")
    ins = node.get("inputs") or {}
    try:
        if cls == "KSampler":
            return max(1, round(int(ins.get("steps", DEFAULT_STEPS))
                                * float(ins.get("denoise", 1.0))))
        if cls == "KSamplerAdvanced":
            steps = int(ins.get("steps", DEFAULT_STEPS))
            first = int(ins.get("start_at_step", 0))
            last = min(int(ins.get("end_at_step", steps)), steps)
            return max(1, last - first)
    except (TypeError, ValueError):
        return DEFAULT_STEPS
    if cls == "SamplerCustomAdvanced":
        return _sigma_steps(graph, ins.get("sigmas"))
    return 0


class _Progress:
    """One graph's execution, as a single monotonic fraction.

    `on_progress` receives `{"frac", "stage", "step", "total"}` — the overall share
    of the job that is done, what is happening right now, and the sampler counters
    (0 outside sampling). Every field is a snapshot, so a client can render straight
    from the last event it saw.
    """

    def __init__(self, graph: dict, on_progress=None):
        self.graph = graph
        self.on_progress = on_progress
        self.steps = {nid: _sampler_steps(graph, nid) for nid in graph}
        self.steps = {nid: n for nid, n in self.steps.items() if n}
        self.cost = {nid: self._cost(nid) for nid in graph}
        self.done: set[str] = set()
        self.measured: dict[str, float] = {}
        self.trusted = _residency_known   # may this run teach the timing store?
        self.node: str | None = None
        self.started = 0.0
        self.reported = 0.0     # within-node fraction, as ComfyUI reported it
        self.pre: float | None = None   # where the clock had the node when it spoke
        self.steps_done = 0     # steps credited by samplers that have finished
        self.step = 0
        self.stage = "Queued"
        self.sent = -1.0
        self.sent_at = 0.0

    # -- cost model -------------------------------------------------------- #
    def _cost(self, nid: str) -> float:
        cls = (self.graph.get(nid) or {}).get("class_type", "")
        known = _timings().get(_cost_key(self.graph, nid))
        known = float(known) if isinstance(known, (int, float)) and known > 0 else None
        if nid in self.steps:
            return max(0.05, known or _STEP_SECONDS) * self.steps[nid]
        return max(0.05, known or _NODE_SECONDS.get(cls, _OTHER_NODE_SECONDS))

    @property
    def total_steps(self) -> int:
        return sum(self.steps.values())

    # -- events ------------------------------------------------------------ #
    def begin(self) -> None:
        self._emit(force=True)

    def cached(self, ids) -> None:
        """Nodes ComfyUI is reusing from its cache — a warm re-run's loaders, which are
        most of a cold run's cost.

        They are priced at zero rather than credited as done: work that will not happen
        this run does not belong in the total either. Otherwise a second generation with
        the same model would open at 40% and spend the whole job in the last stretch,
        when what is actually left to do is all of it. ComfyUI sends this before the
        first node executes, so the denominator is settled before the bar moves.
        """
        for i in ids:
            if i in self.cost:
                self.cost[i] = 0.0
                self.done.add(i)
                self.steps.pop(i, None)
        self._emit(force=True)

    def executing(self, nid) -> None:
        self._close()
        if nid in self.cost:
            self.node, self.started = nid, time.time()
            self.reported, self.pre = 0.0, None
            self.stage = _STAGE_LABELS.get(self.graph[nid].get("class_type", ""), "Working")
        self._emit(force=True)

    def node_progress(self, nid, value: float, maximum: float) -> None:
        """A node reporting its own progress: sampler steps, or VAE frames."""
        if nid not in self.cost:
            nid = self.node          # older payloads omit the node id
        if nid is None:
            return
        if nid != self.node:
            self.executing(nid)
        if maximum > 0:
            # The clock's estimate for this node stops here and the node's own count
            # takes over the rest of it. A sampler spends its first minute loading the
            # transformer and says nothing; step 1 of 8 does not mean the node is an
            # eighth done, it means the silent part is behind us.
            if self.pre is None:
                self.pre = self._elapsed_frac(self.cost[nid])
            self.reported = max(self.reported, min(1.0, value / maximum))
        if nid in self.steps:
            # ComfyUI knows what it is actually running; our count came from reading
            # the graph, so let its number correct ours.
            if maximum > 0 and int(maximum) != self.steps[nid]:
                self.steps[nid] = int(maximum)
                self.cost[nid] = self._cost(nid)
            self.step = min(self.steps_done + int(value), self.total_steps)
        self._emit()

    def tick(self) -> None:
        """Advance the estimate while ComfyUI says nothing (see `_frac`).

        Marked `live: False`, because it is this module's arithmetic rather than news
        from the sidecar. A client that used these to decide the job is alive would
        watch a wedged ComfyUI creep forward forever.
        """
        self._emit(live=False)

    def finish(self) -> None:
        self._close()
        self.done = set(self.cost)
        self.step = self.total_steps
        self.stage = "Finishing"
        self._emit(force=True)
        # Order matters: the durations were measured against this run's cold/warm
        # keys, and it is only now that these models count as loaded — which is also
        # what makes residency knowable from here on, whatever it was at the start.
        global _residency_known
        if self.trusted:
            _remember_timings(self.measured)
        _resident.update(_model_inputs(self.graph))
        _residency_known = True

    # -- internals --------------------------------------------------------- #
    def _close(self) -> None:
        """Credit the running node in full and time it for the next run."""
        nid, self.node = self.node, None
        if nid is None:
            return
        self.done.add(nid)
        if nid in self.steps:
            self.steps_done = min(self.steps_done + self.steps[nid], self.total_steps)
            self.step = self.steps_done
        elapsed = time.time() - self.started
        if elapsed > 0.05:
            per = elapsed / self.steps[nid] if nid in self.steps else elapsed
            self.measured[_cost_key(self.graph, nid)] = per

    def _frac(self) -> float:
        total = sum(self.cost.values()) or 1.0
        done = sum(c for nid, c in self.cost.items() if nid in self.done)
        nid = self.node
        if nid is not None and nid not in self.done:
            share = self.cost[nid]
            within = (self.pre + (1.0 - self.pre) * self.reported
                      if self.pre is not None else self._elapsed_frac(share))
            done += share * within
        return min(1.0, done / total)

    def _elapsed_frac(self, share: float) -> float:
        """How much of the running node the clock says is behind us.

        A node that loads a 15 GB text encoder reports nothing for the minute it takes,
        so its share is filled against how long that same node took last time. Linear
        for as long as it was expected to take, then an exponential tail: an estimate
        that is running late must keep moving — a bar parked at 80% is the useless kind
        — but it must never arrive, because the only thing allowed to say a node is
        finished is ComfyUI saying so.
        """
        if share <= 0:
            return 0.0
        ratio = (time.time() - self.started) / share
        if ratio <= 1.0:
            return 0.8 * ratio
        return 0.8 + 0.2 * (1.0 - math.exp(-(ratio - 1.0)))

    def _emit(self, force: bool = False, live: bool = True) -> None:
        if not self.on_progress:
            return
        frac = max(self._frac(), self.sent)   # a progress bar never goes backwards
        now = time.time()
        if not force and (frac - self.sent < 0.002 or now - self.sent_at < 0.15):
            return
        self.sent, self.sent_at = frac, now
        self.on_progress({"frac": round(frac, 4), "stage": self.stage, "live": live,
                          "step": self.step, "total": self.total_steps})


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
def _await(graph, on_progress=None, on_status=None,
           recv_timeout=600, poll_timeout=900) -> dict:
    """Submit a graph, relay progress, and return its history entry once it finishes.

    `on_progress` gets `_Progress`'s snapshots — one dict per update, covering the
    whole graph rather than only the sampler. `live` on a snapshot separates the ones
    ComfyUI prompted from the ones the clock did, which is what lets a client keep a
    stall detector while the bar still moves through a silent load.

    The timeouts are parameters because they describe the *job*, not this function.
    `recv_timeout` is how long ComfyUI may stay silent before we give up on the
    websocket: ample at 600s for a FLUX step every few seconds, but a Wan graph goes
    quiet for minutes while ComfyUI swaps a 28 GB expert *inside* a node. It is now
    spent a second at a time, because the bar has to keep moving through exactly that
    silence. `poll_timeout` is the fallback path's hard wall, and too short is worse
    than useless there — it stops waiting and lets the caller report "no output" for a
    job still running fine.
    """
    client_id = uuid.uuid4().hex
    pid = _post("/prompt", {"prompt": graph, "client_id": client_id})["prompt_id"]
    prog = _Progress(graph, on_progress=on_progress)
    prog.begin()

    # Progress via websocket if the client lib is available; otherwise just wait.
    try:
        import websocket  # noqa: PLC0415  (websocket-client; optional)

        ws = websocket.create_connection(
            COMFY_URL.replace("http", "ws") + "/ws?clientId=" + client_id, timeout=5
        )
        ws.settimeout(_TICK)
        try:
            silent = 0.0
            while True:
                try:
                    msg = ws.recv()
                except websocket.WebSocketTimeoutException:
                    silent += _TICK
                    if silent >= recv_timeout:
                        raise TimeoutError(f"ComfyUI sent nothing for {recv_timeout}s")
                    prog.tick()
                    continue
                silent = 0.0
                if not isinstance(msg, str):
                    continue
                ev = json.loads(msg)
                data = ev.get("data") or {}
                # Our own client id already filters most of it; this drops anything
                # left over from a different prompt on the same socket.
                if data.get("prompt_id") not in (None, pid):
                    continue
                kind = ev.get("type")
                if kind == "progress":
                    prog.node_progress(data.get("node"), data.get("value", 0),
                                       data.get("max", 0))
                elif kind == "execution_cached":
                    prog.cached(data.get("nodes") or [])
                elif kind == "executing":
                    if data.get("node") is None and data.get("prompt_id") == pid:
                        break
                    prog.executing(data.get("node"))
                elif kind == "executed":
                    prog.executing(None)
        finally:
            ws.close()
    except Exception:
        # No websocket lib / connection: poll history until the prompt completes.
        t0 = time.time()
        while time.time() - t0 < poll_timeout:
            h = _get("/history/" + pid)
            if h.get(pid, {}).get("outputs"):
                break
            prog.tick()
            time.sleep(2)

    prog.finish()
    return _get("/history/" + pid).get(pid, {})


def _run(graph, on_progress=None, on_status=None):
    """Submit a graph, relay progress via the ComfyUI websocket, return PIL."""
    hist = _await(graph, on_progress=on_progress, on_status=on_status)
    for node in hist.get("outputs", {}).values():
        for im in node.get("images", []):
            return _fetch_output(im["filename"], im.get("subfolder", ""))
    status = hist.get("status", {})
    raise RuntimeError(f"FLUX generation produced no image ({status.get('status_str', 'unknown')}).")


def _run_video(graph, on_progress=None, on_status=None):
    """Run a Wan graph and return (webm bytes, PIL first frame | None).

    Separate from `_run` rather than a flag on it, because the two disagree about what
    an output *is*. ComfyUI files a saved video under the same `images` key a saved
    image uses (PreviewVideo.as_dict → {"images": [...], "animated": (True,)}), so
    `_run` would happily find the webm, hand it to Pillow, and raise. Keeping them apart
    means the four image ops never touch this path and `_run`'s contract — "the first
    output anywhere, as PIL" — stays true, because only a Wan graph emits a video.

    Outputs are read by node id, not by scanning for `animated`: this module builds the
    graph and names the nodes, so "the video is at `video`" is a fact about our own
    code, while `animated` is an inference about someone else's UI payload.

    The two samplers need no special handling: each KSamplerAdvanced runs a slice of
    one shared schedule (`start_at_step`/`end_at_step`), and `_Progress` prices and
    counts them from exactly those inputs, so the bar reads as the one 20-step job it is.
    """
    hist = _await(graph, on_progress=on_progress, on_status=on_status,
                  recv_timeout=WAN_RECV_TIMEOUT, poll_timeout=WAN_POLL_TIMEOUT)
    outs = hist.get("outputs", {})
    vid = (outs.get("video") or {}).get("images") or []
    if not vid:
        status = hist.get("status", {})
        raise RuntimeError(f"Video generation produced no video "
                           f"({status.get('status_str', 'unknown')}).")
    data = _fetch_bytes(vid[0]["filename"], vid[0].get("subfolder", ""))

    # The first frame, for the sidebar thumbnail. Best-effort: a missing thumb is a
    # placeholder icon, which is not worth failing a five-minute generation over.
    thumb = (outs.get("thumb") or {}).get("images") or []
    frame = _fetch_output(thumb[0]["filename"], thumb[0].get("subfolder", "")) if thumb else None
    return data, frame


def _prompt_for(prompt: str, enhance: bool) -> str:
    p = (prompt or "").strip()
    return PHOTOREAL_TEMPLATE.format(prompt=p.rstrip(".")) if enhance and p else p


def static_enhance(prompt: str, mode: str) -> str:
    """The template fallback for when no vision model is available to rewrite.

    Create-only, deliberately. PHOTOREAL_TEMPLATE describes a photograph; wrapping an
    edit instruction in it ("A photorealistic photograph. Make the jacket red. Shot on
    a full-frame DSLR…") reads as a scene to describe rather than a change to make,
    and the edit becomes a regeneration. Edit and compose get their prompt back
    unchanged — which is what the graphs have always sent them.
    """
    return _prompt_for(prompt, mode in ("txt2img", "img2img"))


def _label(unet: str) -> str:
    """This transformer's name, or the filename for a user-added model — the same rule
    `list_unets` labels by, so the UI and the status line agree on a model's name."""
    return cat.label_of_unet(unet)


def create(prompt, width=None, height=None, steps=None, guidance=None, seed=0,
           model=None, enhance=True, on_progress=None, on_status=None):
    """Text-to-image ('a candid photo of a woman laughing')."""
    ensure_server(on_status=on_status)
    unet = _resolve_unet(model, ROLE_CREATE)
    w, h = _dim(width), _dim(height)
    g = _txt2img_graph(_prompt_for(prompt, enhance), w, h, _steps(steps, _default_steps(unet)),
                       _guidance(guidance, _default_guidance(unet, ROLE_CREATE)),
                       int(seed), "flux_create", unet)
    if on_status:
        on_status(f"generating with {_label(unet)}…")
    return _run(g, on_progress=on_progress, on_status=on_status)


def img2img(pil, prompt, strength=None, steps=None, guidance=None, seed=0,
            model=None, enhance=True, on_progress=None, on_status=None):
    """Transform an attached image, keeping its composition."""
    ensure_server(on_status=on_status)
    unet = _resolve_unet(model, ROLE_CREATE)
    name = _upload_image(pil, f"init_{uuid.uuid4().hex}.png")
    w, h = _source_resolution(unet, pil)
    g = _img2img_graph(name, _prompt_for(prompt, enhance), _strength(strength),
                       _steps(steps, _default_steps(unet)),
                       _guidance(guidance, _default_guidance(unet, ROLE_CREATE)),
                       int(seed), w, h, "flux_img2img", unet)
    if on_status:
        on_status(f"transforming with {_label(unet)}…")
    return _run(g, on_progress=on_progress, on_status=on_status)


def edit(pil, prompt, refs=(), steps=None, guidance=None, seed=0, model=None,
         on_progress=None, on_status=None):
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
    g = _edit_graph(scene, ref_names, prompt or "", _steps(steps, _default_steps(unet)),
                    _guidance(guidance, _default_guidance(unet, ROLE_EDIT)),
                    int(seed), w, h, "flux_edit", unet)
    if on_status:
        on_status(f"editing with {_label(unet)}…")
    return _run(g, on_progress=on_progress, on_status=on_status)


def compose(pils, prompt, steps=None, guidance=None, seed=0, model=None, on_progress=None, on_status=None):
    """Combine multiple reference images into one new image."""
    ensure_server(on_status=on_status)
    if not pils:
        raise ValueError("compose requires at least one reference image")
    unet = _resolve_unet(model, ROLE_EDIT)
    tag = uuid.uuid4().hex
    names = [_upload_image(p, f"ref{i}_{tag}.png") for i, p in enumerate(pils)]
    # The new scene takes its shape from the first reference.
    width, height = _source_resolution(unet, pils[0])
    g = _compose_graph(names, prompt or "", width, height, _steps(steps, _default_steps(unet)),
                       _guidance(guidance, _default_guidance(unet, ROLE_EDIT)),
                       int(seed), "flux_compose", unet)
    if on_status:
        on_status(f"composing with {_label(unet)}…")
    return _run(g, on_progress=on_progress, on_status=on_status)


def _scaled_progress(cb, lo: float, hi: float):
    """Squeeze one graph's progress into the [lo, hi] slice of an overall bar.

    `_Progress` costs a single graph and always runs 0 -> 1 over it. A control job is
    two or three graphs, so without this the bar fills for the depth map, resets, fills
    again for the pose map, resets, and fills a third time for the generation.
    """
    if cb is None:
        return None

    def relay(p):
        frac = p.get("frac")
        if isinstance(frac, (int, float)):
            p = {**p, "frac": lo + (hi - lo) * min(max(frac, 0.0), 1.0)}
        cb(p)

    return relay


def control(pil, kinds=(), prompt="", refs=(), maps=(), lock=None, control_strength=None,
            canny_low=None, canny_high=None, width=None, height=None, steps=None,
            guidance=None, seed=0, model=None, on_progress=None, on_status=None):
    """Generate an image that follows the structure of a source image.

    This is the answer to a pose words can't specify. `pil` is the source the structure
    comes from — a photo of the pose, or a posed-mannequin render — and `kinds` names
    the control maps to derive from it ("depth", "canny", "pose"; stackable, and depth
    plus pose is the strong pair). `maps` are control maps the caller already has, which
    skips deriving them; that is both the re-roll path and how a map drawn somewhere
    else gets in.

    Returns `(image, [(kind, map_pil), ...])` — the maps come back so the caller can
    show and store them. Seeing the map is most of the value when a pose is failing: it
    is the difference between "the model ignored me" and "the map didn't have the pose
    in it either".

    The preprocessors run as their own ComfyUI prompts rather than as extra nodes on the
    generation graph. Two reasons, both practical: they are whole models (SDPose is an
    SD-architecture checkpoint plus its VAE), and co-loading them with an 18 GB
    transformer is a VRAM failure waiting to happen — as separate prompts, ComfyUI is
    free to evict one before the other loads. And a map that comes back as its own
    result can be looked at, kept, and re-fed.
    """
    ensure_server(on_status=on_status)
    # ROLE_EDIT, not ROLE_CREATE: the whole mode rides on ReferenceLatent, which a plain
    # FLUX.1 dev transformer silently ignores. The edit role is exactly the set of
    # models that read a reference image.
    unet = _resolve_unet(model, ROLE_EDIT)
    lock = _lock(lock)
    tag = uuid.uuid4().hex
    say = on_status or (lambda _m: None)

    kinds = [k for k in kinds if k]
    unknown = [k for k in kinds if k not in cat.CONTROL_KINDS]
    if unknown:
        raise ValueError(f"Unknown control type '{unknown[0]}'.")
    missing = [k for k in kinds if not cat.preprocessor_installed(k)]
    if missing:
        raise ValueError(
            f"The {missing[0]} preprocessor isn't installed. Add it under Control "
            "preprocessors in the Image Models panel.")
    if pil is None:
        if not maps:
            raise ValueError("Control needs a source image to take its structure from.")
        if kinds:
            raise ValueError("Deriving a control map needs a source image.")
        if lock < 1.0:
            raise ValueError("Structure lock needs the source image it locks to.")
    if not kinds and not maps:
        raise ValueError("Pick at least one control type.")

    # Say so rather than refuse. Without an adapter the maps still bias composition
    # through the reference chain — weakly, but a weak result the user can see beats a
    # refusal, and the structure lock works regardless of what LoRAs are attached.
    if not any(p.get("control") for p in loras_for(unet)):
        say(f"{_label(unet)} has no control LoRA attached — the maps will guide loosely. "
            "Flag one as the control adapter in the Image Models panel.")

    # Derive the missing maps, each as its own pass over the leading slice of the bar.
    built: list[tuple[str, object]] = []
    if kinds:
        source = _upload_image(pil, f"ctlsrc_{tag}.png")
        span = PREPROCESS_FRACTION / len(kinds)
        for i, kind in enumerate(kinds):
            say(f"building the {kind} map…")
            g = _preprocess_graph(kind, source, unet,
                                  _canny_edge(canny_low, CANNY_LOW),
                                  _canny_edge(canny_high, CANNY_HIGH))
            built.append((kind, _run(g, on_progress=_scaled_progress(
                on_progress, i * span, (i + 1) * span))))

    map_names = [_upload_image(p, f"ctlmap{i}_{tag}.png") for i, (_, p) in enumerate(built)]
    map_names += [_upload_image(p, f"ctlgiven{i}_{tag}.png") for i, p in enumerate(maps)]
    ref_names = [_upload_image(p, f"ctlref{i}_{tag}.png") for i, p in enumerate(refs)]
    source_name = _upload_image(pil, f"ctllock_{tag}.png") if (pil and lock < 1.0) else ""

    # The starting latent fixes the output shape, so a locked run takes its resolution
    # from the source and an unlocked one from the request — the rule `img2img` and
    # `create` already follow. With neither, the first map decides, the way `compose`
    # sizes itself off its first reference.
    if lock < 1.0:
        w, h = _source_resolution(unet, pil)
    elif width or height:
        w, h = _dim(width), _dim(height)
    else:
        w, h = _source_resolution(unet, (built[0][1] if built else maps[0]))

    g = _control_graph(map_names, ref_names, source_name, lock,
                       _control_scale(control_strength), prompt or "", w, h,
                       _steps(steps, _default_steps(unet)),
                       _guidance(guidance, _default_guidance(unet, ROLE_EDIT)),
                       int(seed), "flux_control", unet)
    say(f"generating with {_label(unet)}…")
    lo = PREPROCESS_FRACTION if built else 0.0
    image = _run(g, on_progress=_scaled_progress(on_progress, lo, 1.0),
                 on_status=on_status)
    return image, built


def animate(pil, prompt, seconds=None, steps=None, guidance=None, seed=0, model=None,
            on_progress=None, on_status=None):
    """Animate an image into a short video ('she turns to look at the camera').

    Returns `(webm bytes, PIL first frame | None)` — not a PIL image like every other
    op here, which is why it goes through `_run_video`.

    `prompt` should describe *motion*, not the scene: the start frame already fixes
    subject, setting and lighting, and re-describing them fights it. See
    `ollama_client`'s animate enhance mode.
    """
    ensure_server(on_status=on_status)
    unet = _resolve_unet(model, ROLE_ANIMATE)
    b = cat.bundle_of_unet(unet)
    # `roles_of` gives an uncatalogued file [ROLE_CREATE], so _resolve_unet can't hand
    # one back for this role — but assert rather than trust that, because the failure
    # mode is a graph naming b["unet_low"] on a bundle that hasn't got one.
    if not b or b["family"] != cat.FAMILY_WAN:
        raise RuntimeError(f"{_label(unet)} can't make video. Install a video model "
                           "in the Models panel.")

    name = _upload_image(pil, f"animate_{uuid.uuid4().hex}.png")
    w, h = _source_resolution(unet, pil)
    n_steps = _steps(steps)
    boundary = _wan_boundary(n_steps)
    length = _wan_length(seconds)
    g = _wan_i2v_graph(name, prompt or "", w, h, length, n_steps, boundary,
                       _guidance(guidance, _default_guidance(unet, ROLE_ANIMATE)),
                       int(seed), "wan_animate", b)
    if on_status:
        on_status(f"animating with {_label(unet)} — {length} frames "
                  f"({length / WAN_FPS:.0f}s at {w}x{h}). This takes a few minutes.")
    return _run_video(g, on_progress=on_progress, on_status=on_status)


# --------------------------------------------------------------------------- #
# Model management: list / add (from any HF repo) / remove extra UNets
# --------------------------------------------------------------------------- #
def list_unets() -> list[dict]:
    """Installed transformers. Each: name, roles, bundle id (None if user-added), size_gb.

    `roles` tells the UI which modes a model can serve: a FLUX.2 transformer serves
    both, a FLUX.1 one serves exactly one.

    Sorted in catalog order (quality order), which is the order `_default_for` picks
    in — so the head of this list is that role's default. Sorting alphabetically
    instead put klein at the head while dev was the default, and the UI believed the
    list.
    """
    out = []
    if UNET_DIR.is_dir():
        for p in sorted(UNET_DIR.iterdir()):
            if p.is_file() and p.suffix.lower() in UNET_EXTS:
                b = cat.bundle_of_unet(p.name)
                # Half of a model is not a model. Wan's two experts both live here, but
                # they run together in one graph off the bundle — so listing the second
                # would offer the user a pick that means nothing.
                if not cat.is_primary(p.name, b):
                    continue
                out.append({
                    "name": p.name,
                    # Through `_label`, not `b["label"]`: the FLUX.1 bundle's two
                    # transformers are separate entries here and need separate names.
                    "label": _label(p.name),
                    "roles": cat.roles_of(p.name),
                    "bundle": b["id"] if b else None,
                    "family": cat.family_of(p.name),
                    "size_gb": round(p.stat().st_size / 1e9, 2),
                    "encoder": encoder_for(p.name),
                    "loras": loras_for(p.name),  # [] when nothing is attached
                })
    out.sort(key=lambda m: (cat.bundle_rank(m["name"]), m["name"].lower()))
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
    # Say what's actually missing. Roles are no longer all served by the same kind of
    # model: someone with FLUX installed who clicks Animate has plenty of image models
    # and none that can animate, and telling them "no image model is installed" sends
    # them looking for a problem that isn't there.
    if role == ROLE_ANIMATE:
        raise RuntimeError("No video model is installed — install one in the Models panel.")
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


# `_resolve_unet` is idempotent — a name it returns resolves to itself — so the API
# layer can resolve once, up front, report what it picked, and pass the result down
# to a graph builder that will resolve it again to the same thing.
def resolve_unet(model, role: str) -> str:
    """The transformer a request will actually run on. See `_resolve_unet`."""
    return _resolve_unet(model, role)


def role_for_mode(mode: str) -> str:
    """The role a generate mode draws its transformer from."""
    if mode == "animate":
        return ROLE_ANIMATE
    # "control" is an edit-role mode: it conditions on ReferenceLatent, which is the
    # thing that role means, even though what comes out is a new image rather than an
    # edited one.
    return ROLE_EDIT if mode in ("edit", "compose", "control") else ROLE_CREATE


def label(unet: str) -> str:
    """A human name for a transformer, for status lines and the chat's model tag."""
    return _label(unet)


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
# fetch (download a file into a model dir) — so the rules for picking and validating a
# UNet stay in the parent, offline and testable. Progress goes to stdout.
_HF_CHILD = r'''
import json, os, struct, sys, time, urllib.request
from urllib.error import HTTPError, URLError

EXTS = (".gguf", ".safetensors", ".sft")
ADAPTER_EXTS = EXTS + (".pt",)
MAX_HEADER = 64 << 20  # a safetensors header is KBs; larger means it isn't one
TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""
CIVITAI_TOKEN = os.environ.get("CIVITAI_TOKEN") or ""

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

    # huggingface_hub owns the transfer so Xet-backed repos work. Black Forest Labs'
    # weights (klein, [dev]) live in HuggingFace's Xet content-addressed store, which
    # only the Xet protocol (hf_xet) can reconstruct — a plain ranged GET on the resolve
    # URL 403s, no matter the token. The library also resumes an interrupted download
    # (via a .cache marker in the staging dir) — these are tens of GB, and restarting a
    # 33 GB pull because the connection dropped at 90% is not an option.
    #
    # It builds a tqdm in one place for both the classic and Xet paths; subclassing that
    # class turns each update into a PROGRESS line. The bar is disabled in this non-tty
    # child, so its own counter never moves — we sum the byte increments passed to
    # update() ourselves.
    import importlib, shutil  # PLC0415
    tqdm_mod = importlib.import_module("huggingface_hub.utils.tqdm")
    _Base = tqdm_mod.tqdm
    st = {"done": 0, "total": 0, "last": 0.0}

    def emit(done, total):
        print("PROGRESS " + json.dumps({"file": base, "done": done, "total": total,
              "pct": (100 * done // total) if total else 0}), flush=True)

    class _Progress(_Base):
        def update(self, n=1):
            if n:
                st["done"] += int(n)
            if self.total:
                st["total"] = int(self.total)
            if time.time() - st["last"] > 0.5:
                emit(st["done"], st["total"])
                st["last"] = time.time()
            return super().update(n)

    from huggingface_hub import hf_hub_download
    print("Downloading %s from %s…" % (base, repo), flush=True)
    # Stage under dest_dir so the finished file is a rename away — same filesystem, no
    # second full-size copy (the weights are the largest files on the disk).
    staging = os.path.join(dest_dir, ".hf-download")
    tqdm_mod.tqdm = _Progress
    try:
        got = hf_hub_download(repo, filename, local_dir=staging, token=TOKEN or False)
    finally:
        tqdm_mod.tqdm = _Base
    os.replace(got, out)
    shutil.rmtree(staging, ignore_errors=True)
    size = os.path.getsize(out)
    emit(size, size)
    print("DONE " + base, flush=True)

def _civitai_open(url):
    headers = {"User-Agent": "vision-model-ui"}
    if CIVITAI_TOKEN:
        headers["Authorization"] = "Bearer " + CIVITAI_TOKEN
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers))

def _civitai_version(ref):
    # A bare id is ambiguous — people paste both the model id and the version id out of
    # the same URL — so try it as a version, then as a model whose newest version we
    # take. Resolving it here rather than in the parent keeps every CivitAI call in the
    # one process that is allowed to make them.
    try:
        with _civitai_open("https://civitai.com/api/v1/model-versions/%s" % ref) as r:
            return json.load(r)
    except HTTPError as e:
        if e.code != 404:
            raise
    with _civitai_open("https://civitai.com/api/v1/models/%s" % ref) as r:
        versions = (json.load(r) or {}).get("modelVersions") or []
    if not versions:
        raise RuntimeError("CivitAI model %s has no downloadable versions." % ref)
    with _civitai_open("https://civitai.com/api/v1/model-versions/%s"
                       % versions[0]["id"]) as r:
        return json.load(r)

def civitai(ref, dest_dir, out_name=""):
    meta = _civitai_version(ref)
    files = [f for f in (meta.get("files") or [])
             if str(f.get("name") or "").lower().endswith(ADAPTER_EXTS)]
    if not files:
        raise RuntimeError("that CivitAI version has no adapter file attached.")
    # `primary` is CivitAI's own answer to "which file is the model" — a version often
    # also carries a config or a VAE, and picking by order would sometimes take those.
    pick = ([f for f in files if f.get("primary")] or files)[0]
    base = out_name or os.path.basename(str(pick.get("name") or "lora.safetensors"))
    out = os.path.join(dest_dir, base)
    if os.path.exists(out):
        raise RuntimeError("a LoRA named %s already exists." % base)
    if not os.path.isdir(dest_dir):
        os.makedirs(dest_dir)
    trained = meta.get("baseModel") or "?"
    name = (meta.get("model") or {}).get("name") or ref
    print("Found %s (%s), trained on %s." % (name, meta.get("name") or "?", trained),
          flush=True)

    url = pick.get("downloadUrl") or ("https://civitai.com/api/download/models/%s"
                                      % meta.get("id"))
    tmp = out + ".part"
    print("Downloading %s from CivitAI…" % base, flush=True)
    with _civitai_open(url) as r:
        # Without a key CivitAI does not 401 — it 200s with its sign-in page, which
        # would otherwise land on disk as a .safetensors full of HTML and fail much
        # later as an unreadable checkpoint.
        ctype = (r.headers.get("Content-Type") or "").lower()
        if "text/html" in ctype:
            raise RuntimeError(
                "CivitAI returned its login page instead of the file. That adapter "
                "needs an API key — add one in the Models panel (civitai.com > "
                "Account settings > API Keys).")
        total = int(r.headers.get("Content-Length") or 0) or int(
            float(pick.get("sizeKB") or 0) * 1024)
        done = 0
        last = 0.0
        with open(tmp, "wb") as fh:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                if time.time() - last > 0.5:
                    print("PROGRESS " + json.dumps(
                        {"file": base, "done": done, "total": total,
                         "pct": (100 * done // total) if total else 0}), flush=True)
                    last = time.time()
    os.replace(tmp, out)
    size = os.path.getsize(out)
    print("PROGRESS " + json.dumps({"file": base, "done": size, "total": size,
                                    "pct": 100}), flush=True)
    print("DONE " + base, flush=True)

try:
    {"whoami": whoami, "probe": probe, "listing": listing, "inspect": inspect,
     "fetch": fetch, "civitai": civitai}[sys.argv[1]](*sys.argv[2:])
except HTTPError as e:
    if sys.argv[1] == "civitai":
        if e.code in (401, 403):
            print("ERROR: CivitAI refused that download. Most adapters there need an "
                  "API key — add one in the Models panel, and check the key hasn't "
                  "expired.", flush=True)
        elif e.code in (400, 404):
            # 400 is what a malformed id gets, 404 a well-formed one that doesn't
            # exist. Same cause from where the user is standing: wrong number.
            print("ERROR: CivitAI has no model or version %s — paste the page URL "
                  "(civitai.com/models/…) or the version id." % sys.argv[2], flush=True)
        else:
            print("ERROR: civitai.com returned %s for %s" % (e.code, sys.argv[2]),
                  flush=True)
        sys.exit(1)
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
except Exception as e:
    # huggingface_hub raises its own requests-based errors, which are NOT
    # urllib.error.HTTPError and so sail past the handler above. Uncaught, they reached
    # the UI as a raw "404 Client Error. (Request ID: ...)" — technically the truth and
    # practically useless, since the actionable part (which repo, which file, whose
    # fault) is exactly what that string omits.
    name = type(e).__name__
    repo = sys.argv[2] if len(sys.argv) > 2 else "?"
    if name == "RepositoryNotFoundError":
        print("ERROR: No such repo on HuggingFace: %s — check the owner/name spelling "
              "(it is case-sensitive), or the repo is private." % repo, flush=True)
    elif name == "GatedRepoError":
        print("ERROR: %s is gated — accept its license on huggingface.co while signed in "
              "as the account your token belongs to." % repo, flush=True)
    elif name == "EntryNotFoundError":
        print("ERROR: %s has no file named '%s'." % (repo, sys.argv[3] if len(sys.argv) > 3
                                                     else "?"), flush=True)
    elif name in ("LocalEntryNotFoundError", "ConnectionError", "URLError"):
        print("ERROR: Could not reach huggingface.co — check the connection.", flush=True)
    else:
        print("ERROR: %s while fetching from %s: %s" % (name, repo, e), flush=True)
    sys.exit(1)
except URLError as e:
    host = "civitai.com" if sys.argv[1] == "civitai" else "huggingface.co"
    print("ERROR: could not reach %s (%s)" % (host, e), flush=True); sys.exit(1)
except Exception as e:
    # huggingface_hub raises its own requests-based errors (GatedRepoError and the like),
    # not urllib's, so the gated case reaches this branch on the fetch path. Read the
    # HTTP status off the attached response and give the same license hint.
    status = getattr(getattr(e, "response", None), "status_code", None)
    if status in (401, 403):
        print("ERROR: %s is gated — accept its license on huggingface.co, then paste "
              "a HuggingFace token in the Models panel." % sys.argv[2], flush=True)
    else:
        print("ERROR: %s" % e, flush=True)
    sys.exit(1)
'''


def _run_child(args: list[str], on_status=None, on_progress=None, token=None) -> list[str]:
    """Run one `_HF_CHILD` mode, streaming its stdout, and return the lines it printed.

    The offline flags are dropped for the child alone, so the serving process itself
    never gains network access. A token is handed over the same way — only this child
    ever sees it — and only the one its mode actually needs: a CivitAI download talks
    to a third-party host, so the HuggingFace credential stays out of that process
    entirely, and vice versa.

    `PROGRESS {...}` lines carry structured download progress; everything else is a
    human status line.
    """
    import sys  # PLC0415

    mode = args[0] if args else ""
    # `token` is an unsaved one being validated; otherwise use whatever is configured
    # (the saved token, else one from the environment). Resolved before the scrub below,
    # which only touches the child's copy of the environment.
    tok = "" if mode == "civitai" else (token or settings.hf_token()).strip()

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
    # The CivitAI key travels the same way and under the same rule: scrub whatever the
    # server inherited, then set only what's configured now, so a stale key exported in
    # the shell can't outlive the one the user saved. Set only for the mode that uses
    # it, so an HF download never carries it either.
    for var in settings.CIVITAI_ENV_VARS:
        env.pop(var, None)
    if mode == "civitai":
        civitai = settings.civitai_token().strip()
        if civitai:
            env["CIVITAI_TOKEN"] = civitai

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
    code = proc.wait()
    if err:
        raise RuntimeError(err)
    if code != 0:
        # A child that dies without printing an ERROR line was killed rather than failing:
        # a negative code is the signal that did it (-9 is the OOM killer, the likely one
        # on a box this size — these downloads are the biggest thing running). Say which,
        # so the next reader isn't left guessing at "check your connection".
        how = f"killed by signal {-code}" if code < 0 else f"exited {code}"
        raise RuntimeError(f"Download failed — the download process {how}.")
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

    def _vet(keys):
        reason = _reject_reason(keys)
        if reason:
            raise RuntimeError(
                f"{os.path.basename(filename)} can't be used — {reason}. "
                "Pass owner/repo:file to name the UNet yourself."
            )
        if not _looks_like_unet(keys):
            say(f"{os.path.basename(filename)} doesn't look like a FLUX UNet — loading anyway.")

    is_st = filename.lower().endswith((".safetensors", ".sft"))
    checked = False
    if is_st:
        say(f"Checking {os.path.basename(filename)}…")
        try:
            _vet(_json_line(_run_child(["inspect", repo_id, filename])))
            checked = True
        except RuntimeError as e:
            # A Xet-backed repo can't serve a header-only ranged read (it 403s), so the
            # pre-download check is impossible there — vet the file once it's on disk
            # instead. Any other failure is a real problem: surface it.
            if "gated" not in str(e).lower() and "403" not in str(e):
                raise

    _run_child(["fetch", repo_id, filename, str(UNET_DIR)], say, on_progress=on_progress)

    if is_st and not checked:
        dest = UNET_DIR / os.path.basename(filename)
        try:
            header, _ = _st_header(dest)
            _vet([k for k in header if k != "__metadata__"])
        except RuntimeError:
            dest.unlink(missing_ok=True)  # don't leave a rejected file behind to load
            raise
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
            # Only a bundle with a merge has a peak above its download size, and only
            # for that bundle does the explanation make sense. Wan has no merges — it
            # would have claimed a text encoder was being stitched together when the
            # number is just the download.
            why = " (peak, while the text encoder is being stitched together)" if pending else ""
            raise RuntimeError(
                f"Not enough disk space for {b['label']}: it needs {need:.0f} GB{why} "
                f"and only {free:.0f} GB is free."
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

        # A shard already sitting in the staging area is one an interrupted install
        # finished. Skip it: `fetch` refuses to overwrite, so without this a resume trips
        # over its own completed downloads instead of picking up where it stopped.
        # (`missing_files` already does this for the files that land in a model dir.)
        steps = [s for s in steps
                 if not (s["dest"] / (s["save_as"] or os.path.basename(s["path"]))).exists()]

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
    (FLUX.1's VAE and encoders would otherwise be pulled out from under it).

    Its add-on choices go with it. Both are keyed by a filename, so leaving them behind
    left a removed model's LoRA stack and encoder override lying in wait: reinstalling
    it later restored those picks silently, and the model came back subtly not being
    the model it shipped as. Uninstalling is the one moment where "back to how it
    arrived" is unambiguously what was meant, so a reinstall now starts clean and any
    adapters are re-attached deliberately.
    """
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
    # After the files are gone, so a bundle that turned out not to be installed raises
    # above and leaves the settings it isn't removing alone.
    for name in cat.unets_of(b):
        settings.set_loras(name, [])
    if b["family"] == cat.FAMILY_FLUX2:
        settings.set_text_encoder(bundle_id, "")


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


def encoder_for(unet: str) -> str:
    """The text encoder(s) `_loaders` will attach to this transformer, for display.

    Mirrors `_loaders`'s four cases exactly — FLUX.2's swappable pick, Qwen's and Wan's
    bundled single encoder, FLUX.1's fixed CLIP-L + T5 pair — so the header can't name
    one encoder while the graph loads another. `family_of` returns FLUX.1 for a
    user-added UNet with no bundle, which is the same fallback `_loaders` takes.
    """
    family = cat.family_of(unet)
    if family == cat.FAMILY_FLUX2:
        return clip_for(cat.bundle_of_unet(unet))
    if family in (cat.FAMILY_QWEN, cat.FAMILY_WAN):
        b = cat.bundle_of_unet(unet)
        return b["clip"] if b else ""
    return f"{CLIP_L} + {T5}"


# --------------------------------------------------------------------------- #
# Add-on compatibility: which encoders and LoRAs a given transformer can load
# --------------------------------------------------------------------------- #
# Both kinds of add-on are silently base-specific, and "silently" is the problem: the
# wrong encoder fails deep in the sampler (see `_check_encoder_layout`) and the wrong
# LoRA binds to nothing and quietly does approximately nothing. Neither says so at the
# point where the user picks it.
#
# Both are settled by a hidden width, but not the *same* width, which is the trap here.
# A LoRA patches the transformer, so its down-projection has to take the transformer's
# width: 6144 on FLUX.2 [dev], 4096 on [klein] 9B, 3072 on FLUX.1. An encoder never
# touches those layers — its output is projected on the way in — so it is measured
# against the encoder the bundle ships with instead. [dev] reads a 5120-wide Mistral
# into a 6144-wide transformer; comparing an encoder to the transformer would reject
# [dev]'s own default. Either way the number survives quantization, so a lighter quant
# of the right checkpoint still fits — which is the swap the encoder panel exists for —
# and the filter keys on architecture rather than on a filename or a metadata string
# the author may never have set.
#
# Every reader below returns None for "couldn't tell", and `_fits` treats that as
# compatible. Hiding a working add-on is worse than offering a broken one: the broken
# one is still caught at load, while the hidden one leaves the user with no way to
# choose it and no explanation.
def _safetensors_header(path) -> dict | None:
    """A safetensors file's header dict — tens of KB off the front, not the weights."""
    try:
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            return json.loads(fh.read(n))
    except (OSError, ValueError, struct.error):
        return None


# GGUF's value types, by the type tag that precedes each metadata value. Only the fixed
# widths need naming; strings and arrays are length-prefixed and handled inline.
_GGUF_SCALARS = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def _gguf_metadata(path) -> dict | None:
    """A GGUF file's metadata key/values, or None if it isn't one / can't be read.

    Walks the key-value block at the head of the file. Values are read only when they
    are the scalars and strings this needs; arrays are skipped by computing their
    length, since the tokenizer's vocabulary is one of them and is enormous.
    """
    try:
        with open(path, "rb") as fh:
            if fh.read(4) != b"GGUF":
                return None
            struct.unpack("<I", fh.read(4))  # format version
            fh.read(8)                       # tensor count
            n_kv = struct.unpack("<Q", fh.read(8))[0]

            def read_str() -> str:
                ln = struct.unpack("<Q", fh.read(8))[0]
                return fh.read(ln).decode("utf-8", "replace")

            def read_value(vtype: int):
                if vtype == 8:                       # string
                    return read_str()
                if vtype == 9:                       # array: skip it
                    itype = struct.unpack("<I", fh.read(4))[0]
                    count = struct.unpack("<Q", fh.read(8))[0]
                    if itype == 8:
                        for _ in range(count):
                            read_str()
                    elif itype == 9:
                        return None                  # nested arrays: give up cleanly
                    else:
                        fh.seek(_GGUF_SCALARS.get(itype, 0) * count, os.SEEK_CUR)
                    return None
                width = _GGUF_SCALARS.get(vtype)
                if width is None:
                    return None
                raw = fh.read(width)
                fmt = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
                       6: "<f", 7: "<B", 10: "<Q", 11: "<q", 12: "<d"}[vtype]
                return struct.unpack(fmt, raw)[0]

            out = {}
            for _ in range(n_kv):
                key = read_str()
                vtype = struct.unpack("<I", fh.read(4))[0]
                out[key] = read_value(vtype)
            return out
    except (OSError, ValueError, struct.error, KeyError, UnicodeDecodeError):
        return None


def _encoder_width(name: str) -> int | None:
    """The hidden width a text encoder emits — what the transformer has to consume.

    Mistral-3 Small is 5120, Qwen3-8B is 4096. Read from the token embedding for a
    safetensors checkpoint, and from `<arch>.embedding_length` for a GGUF, which is
    where llama.cpp records the same number.
    """
    path = cat.TE_DIR / name
    if not path.exists():
        path = cat.CLIP_DIR / name
    if not path.exists():
        return None
    if name.lower().endswith(".gguf"):
        meta = _gguf_metadata(path)
        if not meta:
            return None
        arch = meta.get("general.architecture")
        width = meta.get(f"{arch}.embedding_length") if arch else None
        return int(width) if isinstance(width, int) else None
    header = _safetensors_header(path) or {}
    for key in ("model.embed_tokens.weight", "language_model.model.embed_tokens.weight"):
        shape = (header.get(key) or {}).get("shape")
        if shape and len(shape) == 2:
            return int(shape[1])
    return None


def _unet_width(unet: str) -> int | None:
    """A transformer's hidden width, off its own header — the number both kinds of
    add-on have to match."""
    path = UNET_DIR / os.path.basename(unet or "")
    if not path.exists() or path.suffix.lower() == ".gguf":
        return None
    header = _safetensors_header(path) or {}
    for key, dim in (("double_blocks.0.img_attn.proj.weight", 1), ("img_in.weight", 0)):
        shape = (header.get(key) or {}).get("shape")
        if shape and len(shape) > dim:
            return int(shape[dim])
    return None


def _lora_width(name: str) -> int | None:
    """The hidden width a LoRA was trained against — the input dim of its
    down-projection.

    Covers both naming conventions in the wild: the `diffusion_model.…lora_A` layout
    ai-toolkit writes, and kohya's `lora_unet_…lora_down`. A LoRA that patches only the
    text encoder (`lora_te…`) has no transformer width to report and comes back None.
    """
    path = cat.LORA_DIR / os.path.basename(name or "")
    if not path.exists() or path.suffix.lower() == ".pt":
        return None
    header = _safetensors_header(path) or {}
    for key, info in header.items():
        if key == "__metadata__" or "double_blocks" not in key:
            continue
        if not (key.endswith("lora_A.weight") or key.endswith("lora_down.weight")):
            continue
        shape = (info or {}).get("shape")
        if shape and len(shape) == 2:
            return int(shape[1])
    return None


def _fits(add_on: int | None, model: int | None) -> bool:
    """Whether an add-on of this width can serve a model of that width. Fail-open on
    either being unknown — see the note above `_safetensors_header`."""
    return add_on is None or model is None or add_on == model


def list_text_encoders() -> list[dict]:
    """Every text encoder on disk, with the models each one is the default for and the
    ones it can actually serve.

    `fits` is the bundle ids whose transformer this encoder's architecture matches. The
    picker offers only those: a FLUX.2 model is trained against one encoder
    architecture, and handing it another doesn't degrade the image, it fails deep in
    the sampler (or silently builds a default CLIP-L — see `_check_encoder_layout`).
    Both live FLUX.2 bundles take a *different* architecture from each other, so
    offering every encoder for every model made a wrong pick the default-looking case.
    """
    defaults: dict[str, list[str]] = {}
    wanted: dict[str, int | None] = {}
    for b in cat.BUNDLES:
        if b["family"] == cat.FAMILY_FLUX2:
            defaults.setdefault(b["clip"], []).append(b["label"])
            # Measured against the encoder the bundle ships with, *not* against the
            # transformer: a projection sits between them and the two widths are not
            # the same number. [dev] reads a 5120-wide Mistral into a 6144-wide
            # transformer; [klein]'s 4096 matching on both sides is a coincidence.
            wanted[b["id"]] = _encoder_width(b["clip"])
    out = []
    if cat.TE_DIR.is_dir():
        for p in sorted(cat.TE_DIR.iterdir()):
            if p.is_file() and p.suffix.lower() in (".safetensors", ".sft", ".gguf"):
                width = _encoder_width(p.name)
                out.append({
                    "name": p.name,
                    "size_gb": round(p.stat().st_size / 1e9, 2),
                    "default_for": defaults.get(p.name, []),
                    "fits": [bid for bid, want in wanted.items() if _fits(width, want)],
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
    # Checked here as well as filtered in the picker: the list the browser is choosing
    # from can be stale (an encoder added or removed since it loaded), and this is the
    # only path that writes the override. Fail-open on an undetectable architecture,
    # like `list_text_encoders` — see `_encoder_arch`.
    if safe and not _fits(_encoder_width(safe), _encoder_width(b["clip"])):
        raise ValueError(
            f"{safe} is a different text-encoder architecture than {b['label']} was "
            f"trained against ({b['clip']}). It would fail at load rather than "
            "generate badly — pick an encoder built for this model, or a lighter "
            "quant of its own."
        )
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

    # A file already in the staging area is one an earlier attempt finished: skip it
    # rather than trip `fetch`'s refusal to overwrite. These encoders run to 48 GB, so a
    # second attempt has to pick up where the first stopped instead of starting over —
    # and the staging area therefore survives a failure, and is swept only once the
    # stitched file has landed.
    def done_already(path) -> bool:
        return (staging / os.path.basename(path)).exists()

    for i, s in enumerate(shards, 1):
        if done_already(s):
            continue
        say(f"[{i}/{total_steps}] {os.path.basename(s)}")
        _run_child(["fetch", repo_id, s, str(staging)],
                   on_status=say, on_progress=step(i), token=token)
    if tekken and not done_already(tekken):
        say(f"[{len(shards) + 1}/{total_steps}] {os.path.basename(tekken)}")
        _run_child(["fetch", repo_id, tekken, str(staging)],
                   on_status=say, on_progress=step(len(shards) + 1), token=token)
    say(f"[{total_steps}/{total_steps}] stitching {out.name} from {len(shards)} shards…")
    _merge_shards([staging / os.path.basename(s) for s in shards], out,
                  embed=(staging / "tekken.json", "tekken_model") if tekken else None,
                  on_progress=step(total_steps))
    shutil.rmtree(staging, ignore_errors=True)  # the shards have served their purpose
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


# --------------------------------------------------------------------------- #
# LoRA adapters
# --------------------------------------------------------------------------- #
# A LoRA is a low-rank patch applied over the transformer's weights at load time. It
# steers style or subject matter without replacing the 64 GB checkpoint, which is what
# makes it the practical way to change what a model will render — the base weights are
# far too expensive to retrain.
#
# Unlike a text encoder, none of this is required: no bundle has a default LoRA, and
# "none" is both the starting state and always reachable again. And unlike an encoder,
# a LoRA is silently base-specific — a FLUX.2 [dev] adapter loaded onto klein doesn't
# error, it just fails to bind to most of the layers it names. Hence the per-bundle
# keying: the pick follows the model it was trained for.
LORA_EXTS = (".safetensors", ".sft", ".pt")


def _takes_lora(unet: str) -> bool:
    """Whether this transformer's graph can actually apply an adapter.

    Wan is excluded: `_wan_i2v_graph` assembles its own two-expert graph rather than
    going through `_loaders`, so a LoRA chained there would never be reached. Better to
    not offer the choice than to accept one that silently does nothing.
    """
    return cat.family_of(unet) != cat.FAMILY_WAN


def _lora_picks(model: str) -> list[dict]:
    """The stored LoRA choices for one transformer, in attach order.

    A pick whose file has since been deleted is dropped rather than failing the
    graph — the same fallback shape `clip_for` takes for a missing encoder.

    The rebuild is deliberate (it sanitises the name and coerces the strength), but
    it must carry every field a pick can hold: `control` used to be dropped here,
    so the flag was written to settings.json and then lost by every reader. The
    checkbox appeared to do nothing and `_with_lora` never saw an adapter to scale.
    """
    out = []
    for pick in settings.loras().get(model) or []:
        name = os.path.basename(pick.get("name") or "")
        if name and (cat.LORA_DIR / name).exists():
            entry = {"name": name, "strength": float(pick.get("strength", 1.0))}
            if pick.get("control"):
                entry["control"] = True
            out.append(entry)
    return out


def loras_for(unet: str) -> list[dict]:
    """The LoRAs `_with_lora` will chain onto this transformer, if any."""
    name = os.path.basename(unet)
    return _lora_picks(name) if _takes_lora(name) else []


def list_loras() -> list[dict]:
    """Every LoRA on disk, with the transformers each one can actually patch.

    Sizes are MB — these are patches, not checkpoints. `fits` is the transformer
    filenames whose hidden width this adapter was trained against; the picker offers
    only those, because attaching a klein adapter to [dev] binds to nothing and looks
    like a LoRA that simply does very little rather than like a mistake.
    """
    widths = {m["name"]: _unet_width(m["name"]) for m in list_unets() if _takes_lora(m["name"])}
    out = []
    if cat.LORA_DIR.is_dir():
        for p in sorted(cat.LORA_DIR.iterdir()):
            if p.is_file() and p.suffix.lower() in LORA_EXTS:
                width = _lora_width(p.name)
                out.append({
                    "name": p.name,
                    "size_mb": round(p.stat().st_size / 1e6, 1),
                    "fits": [u for u, w in widths.items() if _fits(width, w)],
                })
    return out


def selected_loras() -> dict[str, list[dict]]:
    """What each installed transformer is set to load. An empty list means no LoRA.

    Keyed the same way the picker is: one entry per selectable transformer, so the
    FLUX.1 bundle's dev and Kontext halves get a row each.
    """
    return {m["name"]: _lora_picks(m["name"])
            for m in list_unets() if _takes_lora(m["name"])}


def set_loras(model: str, picks: list[dict]) -> None:
    """Replace the set of LoRAs attached to one transformer. An empty list detaches
    everything."""
    target = os.path.basename(model or "")
    if not target or not (UNET_DIR / target).exists():
        raise ValueError(f"'{target}' isn't an installed model.")
    if picks and not _takes_lora(target):
        raise ValueError(f"{label(target)} doesn't take a LoRA adapter.")
    seen: dict[str, dict] = {}
    for pick in picks:
        safe = os.path.basename(pick.get("name") or "")
        if not safe:
            continue
        if not (cat.LORA_DIR / safe).exists():
            raise FileNotFoundError(safe)
        # Same guard, and for the same reason, as `set_text_encoder`'s: the browser's
        # list can be stale, and this is the only path that writes the attachment.
        if not _fits(_lora_width(safe), _unet_width(target)):
            raise ValueError(
                f"{safe} was trained against a different transformer than "
                f"{label(target)}. It would bind to almost none of the layers it "
                "names — attach an adapter built for this model."
            )
        # ComfyUI accepts any float, but outside this range a LoRA either does nothing
        # or overwhelms the base weights into noise. Clamp rather than reject: the
        # slider can't produce an out-of-range value, so anything here came from a
        # hand-made request and silently doing the sane thing beats a 400.
        strength = max(-2.0, min(2.0, float(pick.get("strength", 1.0))))
        # Last write for a repeated name wins, but keeps its original position —
        # attaching the same adapter twice is a no-op, not a stack of itself.
        seen[safe] = {"name": safe, "strength": strength}
        # The control-adapter flag rides along. Stored only when set, so a settings
        # file written before this existed reads back identically.
        if pick.get("control"):
            seen[safe]["control"] = True
    settings.set_loras(target, list(seen.values()))


# --------------------------------------------------------------------------- #
# Control preprocessors
# --------------------------------------------------------------------------- #
def list_preprocessors() -> list[dict]:
    """Every control map the app can build, installed or not.

    Includes the ones that need no weights (canny), because the UI's question is "can I
    use this control type", not "is there a file". Those report `installed: True` and no
    size, and the panel renders them as always-available rather than as a download.
    """
    out = []
    for kind in cat.CONTROL_KINDS:
        p = cat.preprocessor(kind)
        if not p:
            out.append({"kind": kind, "id": None, "installed": True, "builtin": True,
                        "label": "Edge detection (Canny)", "size_gb": 0.0,
                        "note": "Traces every edge in the source. No download — it's "
                                "an image filter, not a model. The tightest lock on "
                                "layout, but it carries the source's style across too."})
            continue
        path = cat.preprocessor_file(kind)
        out.append({
            "kind": kind, "id": p["id"], "installed": path.exists(), "builtin": False,
            "label": p["label"], "note": p["note"],
            "size_gb": round(path.stat().st_size / 1e9, 2) if path.exists() else p["size_gb"],
        })
    return out


def install_preprocessor(pid: str, on_status=None, on_progress=None) -> None:
    """Download a control preprocessor's weights.

    One file each, so this is `pull_lora`'s shape rather than `install_bundle`'s: no
    shard merging, no encoder or VAE to fetch alongside, nothing to resolve. The
    destination is whichever ComfyUI folder the loading node searches — the node looks
    models up by folder name, so it is the file's location that makes it findable.
    """
    p = cat.get_preprocessor(pid)
    say = on_status or (lambda _m: None)
    tick = on_progress or (lambda _p: None)
    token = settings.hf_token()
    for spec in p["files"]:
        dest = cat.dest_dir(spec[2])
        dest.mkdir(parents=True, exist_ok=True)
        if cat.file_path(spec).exists():
            continue
        say(f"Downloading {p['label']}…")
        _run_child(["fetch", spec[0], spec[1], str(dest), os.path.basename(spec[1])],
                   on_status=say, on_progress=tick, token=token)
    say("Download complete.")


def delete_preprocessor(pid: str) -> None:
    """Remove a control preprocessor's weights.

    No detaching to do, unlike `delete_lora`: nothing points at a preprocessor. It is
    named by the control kind a request asks for, and a request naming a kind whose
    weights are gone is refused up front in `control` with something the user can act
    on, rather than failing inside a graph.
    """
    p = cat.get_preprocessor(pid)
    removed = False
    for spec in p["files"]:
        path = cat.file_path(spec)
        if path.exists():
            path.unlink()
            removed = True
    if not removed:
        raise FileNotFoundError(p["label"])


# Filenames that name the format rather than the adapter. Several popular repos call
# their only file exactly this, which both collides with the next such repo and leaves a
# picker full of entries called "lora" — so these get renamed after the repo instead.
_GENERIC_LORA_NAMES = frozenset((
    "lora.safetensors", "lora.sft", "lora.pt",
    "pytorch_lora_weights.safetensors", "adapter_model.safetensors",
    "model.safetensors", "lora_weights.safetensors",
    "diffusion_pytorch_model.safetensors",
))

# Trainers publish intermediate snapshots beside the finished weights, named with the
# step count: `style_000000500.safetensors` next to `style.safetensors`.
_LORA_SNAPSHOT_RE = re.compile(r"_\d{6,}\.(safetensors|sft|pt)$", re.I)

# An adapter is a low-rank patch. Anything this size is a merged checkpoint that has
# been filed under "LoRA" — downloading it into models/loras/ would waste tens of GB and
# then fail to load, since LoraLoaderModelOnly expects lora_A/lora_B pairs. FLUX.2's
# Turbo adapter is a legitimate 2.8 GB, so the line sits well above that.
_LORA_MAX_BYTES = 10e9


def _lora_out_name(repo_id: str, filename: str) -> str:
    """What to call the downloaded file on disk.

    A descriptive upstream name is kept — `aidmaNSFWunlock-FLUX-V0.2.safetensors` says
    what it is. A generic one is replaced by the repo's own name, which does.
    """
    base = os.path.basename(filename)
    if base.lower() not in _GENERIC_LORA_NAMES:
        return base
    return repo_id.split("/")[-1] + os.path.splitext(base)[1]


def _drop_snapshots(files: list[dict]) -> list[dict]:
    """Prefer finished weights over training snapshots.

    A repo publishing `style.safetensors` alongside four `style_0000NNNN.safetensors`
    isn't ambiguous to a person — they want the final one. Only applied when it leaves
    something behind: a repo of nothing but snapshots still needs a choice made.
    """
    finals = [f for f in files if not _LORA_SNAPSHOT_RE.search(f["name"])]
    return finals or files


def pull_lora(repo: str, on_status=None, on_progress=None) -> None:
    """Add a LoRA from a HuggingFace repo — `owner/repo:file` or `owner/repo`.

    Single-file only, with none of `pull_text_encoder`'s shard-stitching: a LoRA is
    tens to hundreds of MB and is published as one file. A repo holding several
    genuinely different adapters is an ambiguity we hand back rather than guess at,
    since the names carry the meaning — but training snapshots and generic filenames
    are resolved here rather than pushed onto the user.

    Files downloaded from elsewhere (Civitai hosts most of the FLUX.2 adapters) can be
    dropped straight into `models/loras/` — `list_loras` reads the directory, so
    nothing has to come through here.
    """
    repo_id, filename = _parse_repo(repo)
    token = settings.hf_token()
    cat.LORA_DIR.mkdir(parents=True, exist_ok=True)
    say = on_status or (lambda _m: None)
    tick = on_progress or (lambda _p: None)

    def grab(name: str):
        _run_child(["fetch", repo_id, name, str(cat.LORA_DIR),
                    _lora_out_name(repo_id, name)],
                   on_status=say, on_progress=tick, token=token)
        say("Download complete.")

    if filename:
        grab(filename)
        return

    say(f"Resolving {repo_id}…")
    tensors = _json_line(_run_child(["probe", repo_id], token=token))
    whole = _drop_snapshots([f for f in tensors
                             if f["name"].lower().endswith(LORA_EXTS)])
    if len(whole) == 1:
        if whole[0]["size"] > _LORA_MAX_BYTES:
            raise ValueError(
                f"{repo_id}'s only weights file is {whole[0]['size'] / 1e9:.1f} GB — "
                "that's a merged checkpoint, not a LoRA adapter. Add it under 'Add a "
                "model from any repo' instead.")
        grab(whole[0]["name"])
        return
    if len(whole) > 1:
        # Sorted by name, not size: a collection's adapters are all the same rank and
        # therefore all the same size, so a size sort returns an arbitrary slice.
        shown = sorted(f["name"] for f in whole)
        listed = "\n".join(f"  {repo_id}:{n}" for n in shown[:15])
        more = f"\n  …and {len(shown) - 15} more" if len(shown) > 15 else ""
        raise ValueError(
            f"{repo_id} holds {len(shown)} LoRAs — name the one you want:\n{listed}{more}")
    raise ValueError(f"No LoRA found in {repo_id}. Use owner/repo:file to name one.")


def _parse_civitai(ref: str) -> str:
    """The id in a CivitAI reference, however it was pasted.

    Every shape the site hands out reduces to one number: the page URL, the API
    download URL, the AIR identifier its API returns, or a bare id copied out of any of
    them. Which *kind* of number it is can't be told by looking, so the child tries it
    as a version and falls back to treating it as a model — see `_civitai_version`.
    """
    text = (ref or "").strip()
    if not text:
        raise ValueError("Paste a CivitAI model URL or id.")
    # An explicit version wins over the model id in the same URL: someone who picked a
    # specific version on the page means that one, and both numbers are present there.
    m = re.search(r"modelVersionId=(\d+)", text, re.I) or re.search(r"@(\d+)", text)
    if m:
        return m.group(1)
    m = re.search(r"/(?:api/download/)?models/(\d+)", text, re.I)
    if m:
        return m.group(1)
    if text.isdigit():
        return text
    raise ValueError(
        f"'{text}' doesn't look like a CivitAI model. Paste the page URL "
        "(civitai.com/models/…) or the id from it.")


def pull_lora_civitai(ref: str, on_status=None, on_progress=None) -> None:
    """Add a LoRA from CivitAI, given a page URL, a download URL, an AIR, or an id.

    CivitAI is where most FLUX adapters actually live, and unlike HuggingFace it serves
    the file directly rather than through a repo listing — so there's no probe step and
    nothing to disambiguate: a version names its own primary file. What it does need is
    an API key for most downloads, and it signals a missing one by serving its login
    page with a 200 rather than refusing, which the child checks for explicitly.

    The adapter lands in the same directory as a HuggingFace one and is read back by
    `list_loras`, so its base model is detected from the file itself — CivitAI's own
    `baseModel` string is reported in the status line but never trusted for that.
    """
    version = _parse_civitai(ref)
    cat.LORA_DIR.mkdir(parents=True, exist_ok=True)
    say = on_status or (lambda _m: None)
    say(f"Resolving CivitAI {version}…")
    _run_child(["civitai", version, str(cat.LORA_DIR)],
               on_status=say, on_progress=on_progress or (lambda _p: None))
    say("Download complete.")


def delete_lora(name: str) -> None:
    """Remove a LoRA, detaching it from any model that had it selected.

    Detach rather than refuse (which is what `delete_text_encoder` does): a model
    without its encoder can't run at all, while a model without its LoRA is just the
    base model — the state it shipped in.
    """
    safe = os.path.basename(name or "")
    p = cat.LORA_DIR / safe
    if not p.exists():
        raise FileNotFoundError(safe)
    for model, picks in list(settings.loras().items()):
        kept = [pick for pick in picks
                if os.path.basename((pick or {}).get("name") or "") != safe]
        if len(kept) != len(picks):
            settings.set_loras(model, kept)
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
