"""Local image generation via HuggingFace `diffusers`, in-process.

Parallels `ollama_client.py`: the FastAPI backend owns every diffusion call so
the browser never talks to a model runtime directly. This runs entirely offline
after a one-time weight download (like `ollama pull`).

Sized for a 6 GB GPU: one pipeline resident at a time, fp16 + model CPU offload
so the weights stream to the GPU submodule-by-submodule instead of all at once.
The heavy libs (torch/diffusers) are imported lazily so the chat backend still
boots if they're missing or mid-install.
"""
from __future__ import annotations

import gc
import io
import os
import random
import threading
from pathlib import Path

# Privacy: the serving process is forced FULLY OFFLINE for HuggingFace. These are
# set before diffusers/transformers/huggingface_hub are ever imported (all heavy
# imports in this module are lazy), so the offline flags take effect. Result: no
# staleness checks, no telemetry, no outbound calls of any kind from generation.
#
# The ONE exception is a deliberate model download, which runs in a separate
# subprocess (see `pull`) whose environment has these flags removed — so network
# access is confined to the moment the user explicitly clicks "Download", and the
# long-lived serving process itself never has network-enabled HF access.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# --------------------------------------------------------------------------- #
# Known models. SD 1.5 covers text2img + img2img from one download; SD-Turbo is
# the fast few-step option; Realistic Vision is an opt-in photoreal checkpoint
# (never auto-downloaded — the user pulls it explicitly). Inpaint / upscale
# models come in later phases.
# --------------------------------------------------------------------------- #
BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
TURBO_MODEL = "stabilityai/sd-turbo"
PHOTOREAL_MODEL = "SG161222/Realistic_Vision_V5.1_noVAE"

# id -> display metadata for the UI. This module only handles create-mode
# generation (txt2img / img2img). Instruction editing and multi-image
# composition are handled by FLUX Kontext (see `flux_client.py`).
MODELS = [
    {"id": TURBO_MODEL, "label": "SD-Turbo (fast)", "turbo": True, "photoreal": False, "size_gb": 2.5},
    {"id": BASE_MODEL, "label": "Stable Diffusion 1.5", "turbo": False, "photoreal": False, "size_gb": 4.0},
    {"id": PHOTOREAL_MODEL, "label": "Realistic Vision 5.1 (photoreal)", "turbo": False, "photoreal": True, "size_gb": 4.0},
]

# The external VAE Realistic Vision expects (it ships as `_noVAE`). Without it the
# bundled fallback VAE washes out color/detail. Downloaded alongside the photoreal
# checkpoint (see `pull`) and assigned in `_load`.
VAE_MODEL = "stabilityai/sd-vae-ft-mse"

# Baseline anti-deformity negative prompt, always merged in (SD 1.5 mutates hands,
# faces and anatomy badly without one — the single biggest cause of "horror"
# output). The user's own negative prompt, if any, is prepended to this.
DEFAULT_NEGATIVE = (
    "deformed, distorted, disfigured, mutated, mutation, extra limbs, extra arms, "
    "extra legs, missing limbs, fused fingers, too many fingers, mutated hands, "
    "poorly drawn hands, poorly drawn face, bad anatomy, bad proportions, "
    "malformed, cloned face, gross proportions, long neck, blurry, lowres, "
    "worst quality, low quality, jpeg artifacts, watermark, signature, text, "
    "cropped, out of frame, ugly, duplicate, morbid, mutilated"
)

# Positive quality wrapper for photoreal checkpoints (Realistic Vision responds
# strongly to these tags). Applied only when the request opts into enhancement
# and the model is flagged photoreal. `{prompt}` is the user's text.
PHOTOREAL_TEMPLATE = (
    "RAW photo, {prompt}, (high detailed skin:1.2), 8k uhd, dslr, soft lighting, "
    "high quality, film grain, Fujifilm XT3, sharp focus"
)


def _model_meta(model_id: str) -> dict:
    """UI metadata dict for a model id (empty dict if unknown)."""
    return next((m for m in MODELS if m["id"] == model_id), {})

# Generation is serialized: there's no VRAM to run two at once, and swapping the
# resident pipeline mid-run would corrupt both.
_LOCK = threading.Lock()

# Currently-resident pipeline + which model it holds. `_img2img` is derived from
# `_txt2img` via `from_pipe` (shares weights, no extra VRAM).
_txt2img = None
_img2img = None
_loaded_model: str | None = None


# --------------------------------------------------------------------------- #
# Device / capability probing (lazy — never import torch at module load)
# --------------------------------------------------------------------------- #
def _torch():
    import torch  # noqa: PLC0415  (lazy on purpose)

    return torch


def device() -> str:
    try:
        return "cuda" if _torch().cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def available() -> bool:
    """True if the diffusion stack is importable (deps installed)."""
    try:
        import diffusers  # noqa: F401, PLC0415
        import torch  # noqa: F401, PLC0415

        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Which weights are already on disk (for the models list / UI)
# --------------------------------------------------------------------------- #
def is_downloaded(model_id: str) -> bool:
    """True if the model's weights are in the HuggingFace cache (offline-ready)."""
    try:
        from huggingface_hub.constants import HF_HUB_CACHE  # PLC0415

        folder = "models--" + model_id.replace("/", "--")
        return (Path(HF_HUB_CACHE) / folder).is_dir()
    except Exception:
        return False


def list_models() -> list[dict]:
    return [{**m, "downloaded": is_downloaded(m["id"])} for m in MODELS]


# The download runs in a child process so the network access is fully isolated
# from the always-offline serving process. It reads a JSON list of asset specs on
# argv (`{"repo","kind","patterns"}`): full diffusers pipelines vs. bare repo
# snapshots (the VAE) that aren't loadable as a pipeline.
_DOWNLOAD_SCRIPT = """
import json, sys
specs = json.loads(sys.argv[1])
for s in specs:
    repo, kind = s["repo"], s["kind"]
    if kind == "pipeline":
        from diffusers import DiffusionPipeline
        # Prefer safetensors (smaller); fall back to the full set if a
        # component lacks them.
        try:
            DiffusionPipeline.download(repo, use_safetensors=True)
        except Exception:
            DiffusionPipeline.download(repo)
    else:  # snapshot: fetch only the files we need from a plain repo
        from huggingface_hub import snapshot_download
        snapshot_download(repo, allow_patterns=s.get("patterns") or None)
"""


def _download_specs(model_id: str) -> list[dict]:
    """Everything to fetch for `model_id`: the checkpoint plus its dependencies.

    Photoreal checkpoints additionally need the external VAE (they ship `_noVAE`).
    """
    meta = _model_meta(model_id)
    specs = [{"repo": model_id, "kind": "pipeline", "safetensors": True}]
    if meta.get("photoreal"):
        specs.append({"repo": VAE_MODEL, "kind": "snapshot",
                      "patterns": ["*.json", "*.safetensors"]})
    return specs


def pull(model_id: str, on_status=None):
    """Download a model's weights (and dependencies) to the cache (opt-in).

    Runs in a subprocess whose environment has the offline flags removed, so this
    is the ONLY moment the app touches the network — and the long-lived serving
    process never has network-enabled HuggingFace access. Streams coarse progress
    from the child's output.
    """
    import json  # PLC0415
    import subprocess  # PLC0415
    import sys  # PLC0415
    import time  # PLC0415

    if on_status:
        on_status(f"Downloading {model_id}… (one-time, may take several minutes)")

    # Child env: allow network (drop the offline flags) but keep telemetry off.
    env = {**os.environ}
    env.pop("HF_HUB_OFFLINE", None)
    env.pop("TRANSFORMERS_OFFLINE", None)
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"

    proc = subprocess.Popen(
        [sys.executable, "-c", _DOWNLOAD_SCRIPT, json.dumps(_download_specs(model_id))],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )
    # Read byte-wise and split on \r or \n so tqdm's carriage-return progress
    # lines surface; forward the most recent progress-ish line, throttled.
    buf = b""
    last = 0.0
    while True:
        ch = proc.stdout.read(1)
        if not ch:
            break
        if ch in (b"\r", b"\n"):
            line = buf.decode("utf-8", "replace").strip()
            buf = b""
            if line and on_status and (time.time() - last) > 0.5:
                if "%" in line or "Fetching" in line or "Downloading" in line:
                    on_status(line)
                    last = time.time()
        else:
            buf += ch
    if proc.wait() != 0:
        raise RuntimeError(f"Download of {model_id} failed. Check your connection.")
    if on_status:
        on_status("Download complete.")


# --------------------------------------------------------------------------- #
# Pipeline lifecycle
# --------------------------------------------------------------------------- #
def _apply_memory_opts(pipe, dev: str) -> None:
    """Fit-for-VRAM optimizations, shared by every pipeline we load."""
    if dev == "cuda":
        # Stream submodules to the GPU only while they run — the decisive fit for 6 GB.
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()
        pipe.enable_attention_slicing()
    else:
        pipe.to("cpu")


def _load(model_id: str, on_status=None) -> None:
    """Ensure `model_id` is the resident pipeline. Evicts any other first."""
    global _txt2img, _img2img, _loaded_model
    if _loaded_model == model_id and _txt2img is not None:
        return

    unload()  # free any previously-resident model before loading a new one

    torch = _torch()
    dev = device()
    dtype = torch.float16 if dev == "cuda" else torch.float32
    meta = _model_meta(model_id)

    if on_status:
        on_status(f"loading {model_id}…")

    from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image  # PLC0415

    # Belt-and-suspenders with the process-level HF_HUB_OFFLINE: load purely from
    # the local cache, never the network. (generate() guarantees the model is
    # already downloaded before we get here.)
    pipe = AutoPipelineForText2Image.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,  # local & offline; avoids an extra model + false positives
        use_safetensors=True,
        local_files_only=True,
    )

    # DPM++ 2M SDE Karras: far fewer artifacts than the default PNDM scheduler at
    # the same step count. Turbo has its own few-step schedule — leave it alone.
    if not meta.get("turbo"):
        from diffusers import DPMSolverMultistepScheduler  # PLC0415

        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            use_karras_sigmas=True,
            algorithm_type="sde-dpmsolver++",
        )

    # Realistic Vision ships without a VAE (`_noVAE`); load the external one it
    # expects. Fail soft — a missing VAE only degrades color, it shouldn't block.
    if meta.get("photoreal"):
        try:
            from diffusers import AutoencoderKL  # PLC0415

            pipe.vae = AutoencoderKL.from_pretrained(
                VAE_MODEL, torch_dtype=dtype, local_files_only=True
            )
        except Exception as exc:  # noqa: BLE001
            if on_status:
                on_status(f"(VAE {VAE_MODEL} unavailable, using fallback: {exc})")

    _apply_memory_opts(pipe, dev)

    # img2img shares all components with txt2img — zero extra VRAM/weights.
    img2img = AutoPipelineForImage2Image.from_pipe(pipe)

    _txt2img, _img2img, _loaded_model = pipe, img2img, model_id


def unload() -> None:
    """Free the resident pipeline and reclaim VRAM."""
    global _txt2img, _img2img, _loaded_model
    _txt2img = None
    _img2img = None
    _loaded_model = None
    gc.collect()
    try:
        torch = _torch()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def generate(params: dict, on_step=None, on_status=None):
    """Run one generation. Returns (PIL.Image, seed_used).

    `params`: mode (txt2img|img2img), model, prompt, negative_prompt, steps,
    guidance, strength, width, height, seed (int|None), init_image (PIL|None).
    `on_step(step, total)` fires after each denoising step. Serialized by _LOCK.
    """
    torch = _torch()
    with _LOCK:
        model_id = params.get("model") or BASE_MODEL
        if not is_downloaded(model_id):
            raise RuntimeError(
                f"Model '{model_id}' isn't downloaded yet. Download it first "
                "(Generation settings ▸ Download)."
            )
        _load(model_id, on_status=on_status)

        mode = params.get("mode", "txt2img")
        seed = params.get("seed")
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        generator = torch.Generator(device=device()).manual_seed(int(seed))

        steps = int(params.get("steps") or (2 if model_id == TURBO_MODEL else 25))
        guidance = params.get("guidance")
        if guidance is None:
            guidance = 0.0 if model_id == TURBO_MODEL else 7.5

        def _cb(pipe, step_index, timestep, cb_kwargs):
            if on_step:
                total = getattr(pipe, "num_timesteps", steps) or steps
                on_step(step_index + 1, total)
            return cb_kwargs

        meta = _model_meta(model_id)

        # Prompt: optionally wrap photoreal models in a quality template.
        prompt = params.get("prompt") or ""
        if params.get("enhance", True) and meta.get("photoreal") and prompt.strip():
            prompt = PHOTOREAL_TEMPLATE.format(prompt=prompt)

        # Negative: always fold in the baseline anti-deformity list (user's first).
        user_neg = (params.get("negative_prompt") or "").strip()
        negative = ", ".join(p for p in (user_neg, DEFAULT_NEGATIVE) if p)

        common = dict(
            prompt=prompt,
            negative_prompt=negative,
            num_inference_steps=steps,
            guidance_scale=float(guidance),
            generator=generator,
            callback_on_step_end=_cb,
        )

        if mode == "img2img":
            init = params.get("init_image")
            if init is None:
                raise ValueError("img2img requires an init image")
            result = _img2img(
                image=init,
                strength=float(params.get("strength") or 0.6),
                **common,
            )
        else:  # txt2img
            result = _txt2img(
                width=int(params.get("width") or 512),
                height=int(params.get("height") or 512),
                **common,
            )

        return result.images[0], int(seed)


def pil_to_data_url(image, max_size: int | None = None) -> str:
    """Encode a PIL image as a `data:image/jpeg;base64,...` URL for the store.

    JPEG matches the content-addressed store's `.jpg` convention (see db.py).
    `max_size` produces a downscaled thumbnail (longest edge) when set.
    """
    import base64  # PLC0415

    img = image.convert("RGB")
    if max_size:
        img = img.copy()
        img.thumbnail((max_size, max_size))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
