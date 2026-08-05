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
LORA_DIR = MODELS_DIR / "loras"

_DIRS = {"unet": UNET_DIR, "clip": CLIP_DIR, "text_encoders": TE_DIR, "vae": VAE_DIR,
         "loras": LORA_DIR}

FAMILY_FLUX1 = "flux1"
FAMILY_FLUX2 = "flux2"
FAMILY_WAN = "wan"

ROLE_CREATE = "create"
ROLE_EDIT = "edit"
ROLE_ANIMATE = "animate"

# Every bundle key that names a transformer file. Wan 2.2 is a mixture of experts: two
# transformers that run in *one* graph, one after the other, rather than FLUX.1's two
# that split by role. So the list is not "unet + unet_edit" any more, and the lookups
# below must agree on it — they used to spell the pair out separately, which is how a
# third key could be invisible to one of them and not the other.
_EXPERT_KEYS = ("unet", "unet_edit", "unet_low")

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
# is told, and it is per-bundle because the right answer differs: "default" keeps a
# checkpoint at the precision it shipped in, which is what an already-quantized or
# small-enough one wants — casting it again (as the size heuristic for user-added models
# would) only degrades it. A checkpoint too big for the card says so explicitly instead.
BUNDLES = [
    {
        "id": "flux2-dev-bfl",
        "label": "FLUX.2 [dev] — from Black Forest Labs",
        "family": FAMILY_FLUX2,
        "roles": [ROLE_CREATE, ROLE_EDIT],
        "blurb": ("The full bf16 release, straight from Black Forest Labs. Gated: accept "
                  "the licence on HuggingFace and save a token first. 100 GB on disk — "
                  "the weights are cast to fp8 at load to fit the GPU."),
        "size_gb": 100.4,
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
            # The encoder comes from ComfyUI's packaging, not BFL's own text_encoder/
            # shards, even though the transformer beside it is first-party.
            #
            # BFL publishes the encoder in HuggingFace's multimodal layout, naming its
            # tensors `language_model.model.…` / `vision_tower.…`. ComfyUI identifies an
            # encoder purely by tensor name (`detect_te_model`) and every FLUX.2 case is
            # keyed off a flattened `model.layers.0.…`, with a prefix remap only for
            # `model.language_model.` — the other ordering. So a faithful stitch of the
            # shards produces a file ComfyUI cannot recognise: it silently falls back to
            # a default CLIP-L and the run dies deep in the sampler with a 768-vs-15360
            # matmul. This build is the same weights, re-keyed (and pruned to the 30
            # layers FLUX.2 actually taps), which is what the loader expects.
            ("Comfy-Org/flux2-dev",
             "split_files/text_encoders/mistral_3_small_flux2_bf16.safetensors",
             "text_encoders"),
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
        "id": "wan22-i2v-a14b-fp16",
        "label": "Wan 2.2 I2V A14B — fp16",
        "family": FAMILY_WAN,
        "roles": [ROLE_ANIMATE],
        "blurb": ("Turns one image into a 5-second 720p video. Two 14B expert "
                  "transformers: the first lays down motion, the second sharpens it. "
                  "Ungated — no HuggingFace token needed. Needs a very large card."),
        "size_gb": 68.8,
        # Estimate: one 28.6 GB expert resident at a time (ComfyUI evicts the high-noise
        # one before loading the low-noise one) + the 11.4 GB encoder + activations for
        # 81 frames at 720p. Measure on real hardware and correct this.
        "vram_gb": 80,
        "gated": False,
        # The two experts run in one graph, so `unet_low` is not `unet_edit`: that key
        # means "the transformer for the *edit role*" and `_default_for` reads it as a
        # role split. Wan's split is temporal and both experts serve `animate`. Naming
        # the high-noise one `unet` keeps it the bundle's canonical handle, so every
        # lookup that resolves a model by filename works unmodified; `unet_low` stays a
        # detail the graph builder reads back off the bundle.
        "unet": "wan2.2_i2v_high_noise_14B_fp16.safetensors",
        "unet_low": "wan2.2_i2v_low_noise_14B_fp16.safetensors",
        # fp16 is native on this hardware and the checkpoint is already fp16 — the size
        # heuristic for user-added models would cast it to fp8, which is a real loss
        # here and unnecessary on a card this big.
        "weight_dtype": "default",
        "clip": "umt5_xxl_fp16.safetensors",
        "vae": "wan_2.1_vae.safetensors",
        "files": [
            ("Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
             "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors", "unet"),
            ("Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
             "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors", "unet"),
            ("Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
             "split_files/text_encoders/umt5_xxl_fp16.safetensors", "text_encoders"),
            ("Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
             "split_files/vae/wan_2.1_vae.safetensors", "vae"),
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
        if base in {b.get(k) for k in _EXPERT_KEYS}:
            return b
    return None


def label_of_unet(name: str) -> str:
    """The name to show for one transformer *file*.

    A bundle's label names the install rather than a file, which is the same thing
    only while a bundle ships one pickable transformer — as all of them currently do,
    so the label is normally the answer. A bundle that ships two (FLUX.1 was one: dev
    created, Kontext edited, and the picker listed them separately) can set
    `unet_labels` to name them apart. Without it both entries rendered the same
    string, and since the picker filters by role only one was ever on screen — so it
    read as a single model that behaved differently per tab, and a status line naming
    it said nothing about which file had actually run.
    """
    base = os.path.basename(name or "")
    b = bundle_of_unet(base)
    if not b:
        return base or "FLUX"
    return (b.get("unet_labels") or {}).get(base) or b["label"]


def is_primary(name: str, bundle: dict | None = None) -> bool:
    """Whether a UNet filename is a model a user can *pick*, rather than half of one.

    True for anything uncatalogued: a user-added transformer is a model in its own
    right. Only a bundle's second expert is not — Wan's low-noise half sits in
    models/unet like any other transformer, so a directory scan finds it and would
    offer it as a model. Picking it isn't meaningful; the graph loads both experts off
    the bundle regardless of which one was named.

    `bundle` is this name's bundle when the caller already has it, to save the lookup.
    """
    b = bundle if bundle is not None else bundle_of_unet(name)
    return not b or os.path.basename(name or "") != b.get("unet_low")


def bundle_rank(unet: str) -> int:
    """Where a UNet's bundle sits in BUNDLES, for sorting. User-added models sort
    last.

    BUNDLES is written in quality order, which is the order `flux_client._default_for`
    picks a default in. Sorting the model list by it too keeps "first in the list" and
    "what runs by default" the same model — they used to disagree (the list sorted
    alphabetically, so klein led it while dev was the default), which is how the UI
    ended up showing one model while another did the work.
    """
    b = bundle_of_unet(unet)
    return BUNDLES.index(b) if b else len(BUNDLES)


def unets_of(bundle: dict) -> list[str]:
    """Every transformer filename one bundle ships, in `_EXPERT_KEYS` order.

    One name for most bundles; Wan's two experts, or a role-split pair, come back
    together. Callers that need to act on a bundle's transformers — clearing the
    add-ons attached to them, say — go through this rather than spelling the keys out,
    which is how a third key once became visible to one lookup and not another.
    """
    return [bundle[k] for k in _EXPERT_KEYS if bundle.get(k)]


def unets() -> dict[str, dict]:
    """Every UNet filename the catalog knows about -> its bundle."""
    out = {}
    for b in BUNDLES:
        for key in _EXPERT_KEYS:
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
    """Which modes a UNet can serve. FLUX.2 creates and edits; Wan only animates; a
    FLUX.1 UNet does one or the other.

    A catalogued bundle states its own roles, and that answer is authoritative — the
    filename heuristic below is scoped to FLUX.1 deliberately. It is the only family
    whose transformers genuinely split by role, and it's the only family a user-added
    model can land in (`family_of` says so). Letting the heuristic see every family
    meant a Wan file fell through it and came back as a *create* model.
    """
    b = bundle_of_unet(unet)
    if b and b["family"] != FAMILY_FLUX1:
        return list(b["roles"])
    # A user-added model: a Kontext transformer takes a ReferenceLatent and a plain dev
    # one ignores it, and the filename is the only signal we have for which it is.
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
