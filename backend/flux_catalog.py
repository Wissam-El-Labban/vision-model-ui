"""The catalog of installable image models.

A model is not just a transformer file: it comes with a text encoder, a VAE, and a
graph shape that only fits that architecture. So the unit the user installs is a
*bundle*, and every bundle declares a *family* — the thing `flux_client` dispatches
its workflow graphs on.

Nothing here is downloaded at startup any more (it used to be, in run.sh). The user
picks a bundle in the UI and the app fetches it; see `flux_client.install_bundle`.
"""
import os
import shutil
from pathlib import Path

_RUNTIME = Path(__file__).resolve().parent.parent / "flux_runtime"
COMFY_DIR = _RUNTIME / "ComfyUI"
MODELS_DIR = COMFY_DIR / "models"

# Where each kind of file lives. These are ComfyUI's own search paths: it looks for
# text encoders in both models/text_encoders and models/clip, and for diffusion
# models in both models/unet and models/diffusion_models (folder_paths.py:26-27).
UNET_DIR = MODELS_DIR / "unet"
CLIP_DIR = MODELS_DIR / "clip"
TE_DIR = MODELS_DIR / "text_encoders"
VAE_DIR = MODELS_DIR / "vae"

_DIRS = {"unet": UNET_DIR, "clip": CLIP_DIR, "text_encoders": TE_DIR, "vae": VAE_DIR}

FAMILY_FLUX1 = "flux1"
FAMILY_FLUX2 = "flux2"

ROLE_CREATE = "create"
ROLE_EDIT = "edit"

# --------------------------------------------------------------------------- #
# Bundles
# --------------------------------------------------------------------------- #
# `files` are (repo, path-in-repo, dest-dir-key). `unet`/`clip`/`vae` name the files
# the graphs load. `weight_dtype` is what UNETLoader is told: FLUX.2's fp8mixed
# checkpoint is *already* quantized, so it loads as "default" — casting it again to
# fp8_e4m3fn (which the size heuristic for user-added models would do) degrades it.
BUNDLES = [
    {
        "id": "flux2-dev-fp8",
        "label": "FLUX.2 [dev] — fp8",
        "family": FAMILY_FLUX2,
        "roles": [ROLE_CREATE, ROLE_EDIT],
        "blurb": ("Best photorealism that fits 48 GB. One model does create, edit and "
                  "combine. Ungated — no HuggingFace token needed."),
        "size_gb": 50.1,
        "vram_gb": 48,
        "gated": False,
        "unet": "flux2_dev_fp8mixed.safetensors",
        "weight_dtype": "default",
        "clip": "mistral_3_small_flux2_fp8.safetensors",
        "vae": "flux2-vae.safetensors",
        "files": [
            ("Comfy-Org/flux2-dev",
             "split_files/diffusion_models/flux2_dev_fp8mixed.safetensors", "unet"),
            ("Comfy-Org/flux2-dev",
             "split_files/text_encoders/mistral_3_small_flux2_fp8.safetensors", "text_encoders"),
            ("Comfy-Org/flux2-dev", "split_files/vae/flux2-vae.safetensors", "vae"),
        ],
    },
    {
        "id": "flux1-q8-gguf",
        "label": "FLUX.1 dev + Kontext — Q8 GGUF",
        "family": FAMILY_FLUX1,
        "roles": [ROLE_CREATE, ROLE_EDIT],
        "blurb": ("Quantized FLUX.1. Two transformers (dev creates, Kontext edits). "
                  "Lower quality than FLUX.2, but runs on a 24 GB card."),
        "size_gb": 30.9,
        "vram_gb": 24,
        "gated": False,
        # Two UNets, split by role — the only bundle where that's true.
        "unet": "flux1-dev-Q8_0.gguf",
        "unet_edit": "flux1-kontext-dev-Q8_0.gguf",
        "weight_dtype": "default",
        "clip": "t5-v1_1-xxl-encoder-Q8_0.gguf",
        "clip_l": "clip_l.safetensors",
        "vae": "ae.safetensors",
        "files": [
            ("city96/FLUX.1-dev-gguf", "flux1-dev-Q8_0.gguf", "unet"),
            ("QuantStack/FLUX.1-Kontext-dev-GGUF", "flux1-kontext-dev-Q8_0.gguf", "unet"),
            ("city96/t5-v1_1-xxl-encoder-gguf", "t5-v1_1-xxl-encoder-Q8_0.gguf", "clip"),
            ("comfyanonymous/flux_text_encoders", "clip_l.safetensors", "clip"),
            ("ffxvs/vae-flux", "ae.safetensors", "vae"),
        ],
    },
]

_BY_ID = {b["id"]: b for b in BUNDLES}


def get(bundle_id: str) -> dict:
    b = _BY_ID.get(bundle_id or "")
    if not b:
        raise ValueError(f"Unknown model '{bundle_id}'.")
    return b


def dest_dir(key: str) -> Path:
    return _DIRS[key]


def file_path(spec: tuple) -> Path:
    """Where a (repo, path-in-repo, dest-key) file lands on disk."""
    _repo, path, key = spec
    return _DIRS[key] / os.path.basename(path)


def installed(bundle: dict) -> bool:
    return all(file_path(f).exists() for f in bundle["files"])


def installed_bundles() -> list[dict]:
    return [b for b in BUNDLES if installed(b)]


def missing_files(bundle: dict) -> list[tuple]:
    return [f for f in bundle["files"] if not file_path(f).exists()]


def bundle_of_unet(name: str) -> dict | None:
    """The bundle a UNet filename belongs to, or None for a user-added model."""
    base = os.path.basename(name or "")
    for b in BUNDLES:
        if base in (b.get("unet"), b.get("unet_edit")):
            return b
    return None


def unets() -> dict[str, dict]:
    """Every UNet filename the catalog knows about -> its bundle."""
    out = {}
    for b in BUNDLES:
        for key in ("unet", "unet_edit"):
            if b.get(key):
                out[b[key]] = b
    return out


def family_of(unet: str) -> str:
    """The graph family a UNet loads under.

    User-added models (`flux_client.pull_unet`) aren't in the catalog: they land in
    models/unet and share FLUX.1's encoders, which is the only family we can build a
    graph for without knowing what they are.
    """
    b = bundle_of_unet(unet)
    return b["family"] if b else FAMILY_FLUX1


def roles_of(unet: str) -> list[str]:
    """Which modes a UNet can serve. FLUX.2 does both; a FLUX.1 UNet does one.

    For a user-added model the filename is the only signal we have — a Kontext
    transformer takes a ReferenceLatent and a plain dev one ignores it.
    """
    b = bundle_of_unet(unet)
    if b and b["family"] == FAMILY_FLUX2:
        return list(b["roles"])
    return [ROLE_EDIT] if "kontext" in os.path.basename(unet or "").lower() else [ROLE_CREATE]


def free_gb() -> float:
    """Free space where the weights land (created lazily, so walk up to a real dir)."""
    p = MODELS_DIR
    while not p.exists() and p != p.parent:
        p = p.parent
    return round(shutil.disk_usage(p).free / 1e9, 1)


def needed_gb(bundle: dict) -> float:
    """Download size still outstanding for this bundle."""
    if installed(bundle):
        return 0.0
    total = bundle["size_gb"]
    have = sum(file_path(f).stat().st_size for f in bundle["files"] if file_path(f).exists())
    return round(max(total - have / 1e9, 0.0), 1)
