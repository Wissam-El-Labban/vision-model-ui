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

# Shards land here while a merge runs, not in the model dirs — ComfyUI scans those and
# would offer a half-written encoder as a loadable file.
STAGING_DIR = MODELS_DIR / ".shards"

# --------------------------------------------------------------------------- #
# Bundles
# --------------------------------------------------------------------------- #
# `files` are (repo, path-in-repo, dest-dir-key) with an optional 4th element: the name
# to save it under. That rename is not cosmetic — FLUX.2's VAE is also called
# `ae.safetensors` upstream and would overwrite FLUX.1's, which is a different VAE.
#
# `merges` rebuild a single loadable file from a repo that only ships a sharded one.
# Black Forest Labs publishes its text encoders in diffusers layout (2-10 shards) and
# ComfyUI's CLIPLoader takes one file, so we stitch the shards back together after the
# download; see `flux_client._merge_shards`. `embed` adds a non-weight tensor the
# encoder needs at load: Mistral's tokenizer, which ComfyUI reads out of the checkpoint
# as `tekken_model` and which BFL's own repo doesn't carry.
#
# `unet`/`clip`/`vae` name the files the graphs load. `weight_dtype` is what UNETLoader
# is told: FLUX.2's fp8mixed checkpoint is *already* quantized, so it loads as "default"
# — casting it again to fp8_e4m3fn (which the size heuristic for user-added models would
# do) degrades it.
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
        "id": "flux2-dev-bfl",
        "label": "FLUX.2 [dev] — from Black Forest Labs",
        "family": FAMILY_FLUX2,
        "roles": [ROLE_CREATE, ROLE_EDIT],
        "blurb": ("The full bf16 release, straight from Black Forest Labs. Gated: accept "
                  "the licence on HuggingFace and save a token first. 113 GB on disk — "
                  "the weights are cast to fp8 at load to fit the GPU."),
        "size_gb": 112.8,
        "vram_gb": 46,
        "gated": True,
        "unet": "flux2-dev.safetensors",
        # 64 GB of bf16 will not fit a 46 GB card. fp8 is native on Ada/Hopper, so cast
        # on load rather than spilling the transformer to system RAM.
        "weight_dtype": "fp8_e4m3fn",
        "clip": "mistral_3_small_flux2_bf16.safetensors",
        "vae": "flux2-dev-vae.safetensors",
        "files": [
            ("black-forest-labs/FLUX.2-dev", "flux2-dev.safetensors", "unet"),
            # Renamed: upstream calls it ae.safetensors, same as FLUX.1's very different VAE.
            ("black-forest-labs/FLUX.2-dev", "ae.safetensors", "vae", "flux2-dev-vae.safetensors"),
        ],
        "merges": [
            {
                "repo": "black-forest-labs/FLUX.2-dev",
                "shards": [f"text_encoder/model-{i:05d}-of-00010.safetensors"
                           for i in range(1, 11)],
                "key": "text_encoders",
                "out": "mistral_3_small_flux2_bf16.safetensors",
                "shards_gb": 48.0,
                # BFL ships an HF-format tokenizer.json; ComfyUI wants Mistral's tekken
                # vocab, which only Mistral publishes. Same tokenizer, first-party source.
                "embed": ("mistralai/Mistral-Small-3.2-24B-Instruct-2506",
                          "tekken.json", "tekken_model"),
            },
        ],
    },
    {
        "id": "flux2-klein-9b",
        "label": "FLUX.2 [klein] 9B — from Black Forest Labs",
        "family": FAMILY_FLUX2,
        "roles": [ROLE_CREATE, ROLE_EDIT],
        "blurb": ("The distilled 9B FLUX.2, straight from Black Forest Labs. Gated: accept "
                  "the licence on HuggingFace and save a token first. Far lighter than "
                  "[dev] and runs in bf16 with no quantization."),
        "size_gb": 34.7,
        "vram_gb": 32,
        "gated": True,
        "unet": "flux-2-klein-9b.safetensors",
        "weight_dtype": "default",
        # klein conditions on Qwen3-8B, not [dev]'s Mistral. ComfyUI's flux2 CLIP type
        # detects which from the checkpoint, so the loader needs no help — but the two
        # encoders are not interchangeable.
        "clip": "qwen_3_8b_flux2.safetensors",
        "vae": "flux2-klein-vae.safetensors",
        "files": [
            ("black-forest-labs/FLUX.2-klein-9B", "flux-2-klein-9b.safetensors", "unet"),
            ("black-forest-labs/FLUX.2-klein-9B", "vae/diffusion_pytorch_model.safetensors",
             "vae", "flux2-klein-vae.safetensors"),
        ],
        "merges": [
            {
                "repo": "black-forest-labs/FLUX.2-klein-9B",
                "shards": [f"text_encoder/model-{i:05d}-of-00004.safetensors"
                           for i in range(1, 5)],
                "key": "text_encoders",
                "out": "qwen_3_8b_flux2.safetensors",
                "shards_gb": 16.4,
                # Qwen3's tokenizer ships inside ComfyUI, so nothing to embed.
                "embed": None,
            },
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
    """Where a (repo, path-in-repo, dest-key[, save-as]) file lands on disk."""
    path, key = spec[1], spec[2]
    name = spec[3] if len(spec) > 3 else os.path.basename(path)
    return _DIRS[key] / name


def merges(bundle: dict) -> list[dict]:
    return bundle.get("merges") or []


def merge_out(m: dict) -> Path:
    """The single file a merge produces — what the graph actually loads."""
    return _DIRS[m["key"]] / m["out"]


def staging_dir(m: dict) -> Path:
    """Where this merge's shards sit until they've been stitched together."""
    return STAGING_DIR / m["out"].rsplit(".", 1)[0]


def shard_path(m: dict, path: str) -> Path:
    return staging_dir(m) / os.path.basename(path)


def pending_merges(bundle: dict) -> list[dict]:
    return [m for m in merges(bundle) if not merge_out(m).exists()]


def installed(bundle: dict) -> bool:
    return (all(file_path(f).exists() for f in bundle["files"])
            and not pending_merges(bundle))


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


def part_path(spec: tuple) -> Path:
    """The .part file an interrupted download of this file left behind."""
    p = file_path(spec)
    return p.with_name(p.name + ".part")


def needed_gb(bundle: dict) -> float:
    """Download size still outstanding for this bundle.

    Counts bytes already sitting in a half-finished `.part` too — they don't need to
    be fetched again (the download resumes), and they're already on the disk. Ignoring
    them made the disk check refuse a resumed install that would in fact have fit.
    """
    if installed(bundle):
        return 0.0
    have = 0
    for f in bundle["files"]:
        for p in (file_path(f), part_path(f)):
            if p.exists():
                have += p.stat().st_size
    for m in merges(bundle):
        out = merge_out(m)
        if out.exists():
            have += out.stat().st_size
            continue
        for s in m["shards"]:
            sp = shard_path(m, s)
            for p in (sp, sp.with_name(sp.name + ".part")):
                if p.exists():
                    have += p.stat().st_size
    return round(max(bundle["size_gb"] - have / 1e9, 0.0), 1)


def peak_gb(bundle: dict) -> float:
    """Free space the install needs at its high-water mark.

    A merge is the peak: its shards are still on disk while the stitched file is being
    written beside them, so that one encoder is briefly stored twice. Checking only the
    final footprint would green-light an install that runs the disk dry mid-merge.
    """
    extra = max((m["shards_gb"] for m in pending_merges(bundle)), default=0.0)
    return round(needed_gb(bundle) + extra, 1)
