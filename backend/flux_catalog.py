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
# The two directories the control preprocessors load from. Both are ComfyUI search
# paths already (folder_paths.py:57 and :61) — the nodes look models up by folder name,
# so these are the only places their files can go.
#
# Neither is scanned by `flux_client.list_unets`, which reads UNET_DIR alone. That is
# what keeps a depth estimator or a pose checkpoint out of the model picker: they are
# not transformers and must never be offered as one.
GEOM_DIR = MODELS_DIR / "geometry_estimation"
CKPT_DIR = MODELS_DIR / "checkpoints"

_DIRS = {"unet": UNET_DIR, "clip": CLIP_DIR, "text_encoders": TE_DIR, "vae": VAE_DIR,
         "loras": LORA_DIR, "geometry_estimation": GEOM_DIR, "checkpoints": CKPT_DIR}

FAMILY_FLUX1 = "flux1"
FAMILY_FLUX2 = "flux2"
FAMILY_FLUX3 = "flux3"
FAMILY_QWEN = "qwen"
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
# Groups
# --------------------------------------------------------------------------- #
# How the Image Models panel organises the catalog: one section per model line, in
# display order. Every bundle names its `group`. A bundle with `quant_of` is a
# quantized build of another bundle in the same group (its *parent*, the unquantized
# release) and is listed in a dropdown under that parent rather than beside it.
GROUPS = [
    {"id": "flux3-klein", "title": "FLUX 3 Image Klein",
     "blurb": "Black Forest Labs' newest distilled model."},
    {"id": "flux2-dev", "title": "FLUX.2 [dev]",
     "blurb": "The full-size FLUX.2 — highest fidelity, heaviest to run."},
    {"id": "flux2-klein-9b", "title": "FLUX.2 [klein] 9B",
     "blurb": "Distilled 9B FLUX.2: 8 steps, conditions on Qwen3-8B. Every quantized "
              "build takes the same LoRAs as the unquantized one."},
    {"id": "flux2-klein-4b", "title": "FLUX.2 [klein] 4B",
     "blurb": "The smallest FLUX.2: distilled 4B, conditions on Qwen3-4B. Its LoRAs are "
              "not interchangeable with 9B's, but work on every 4B build."},
    {"id": "qwen-image", "title": "Qwen-Image",
     "blurb": "Alibaba's 20B create + edit pairs — best at legible text in images."},
    {"id": "wan22", "title": "Wan 2.2 (video)",
     "blurb": "Image-to-video."},
]

KLEIN_GROUPS = ("flux2-klein-9b", "flux2-klein-4b")

# --------------------------------------------------------------------------- #
# [klein] quantized builds
# --------------------------------------------------------------------------- #
# Every quantized FLUX.2 [klein] (9B and 4B distilled) published as a single file:
# unsloth's GGUF conversions and BFL's own fp8. NVFP4 is left out — it only runs fast
# on Blackwell — and so is the F16 GGUF, which is BF16's size with a narrower range.
#
# All of them are straight conversions: tensor names are the bf16 checkpoint's own,
# so a LoRA trained for unquantized [klein] binds to every build of the same size
# (`flux_client._unet_width`/`_unet_blocks` read the same numbers off a GGUF header as
# off a safetensors one). The VAE is byte-identical across 4B and 9B (sha256
# ca70d220…), which is why every klein bundle saves it under the one filename.
#
# The encoder is paired by tier: down to Q6_K the transformer gets Qwen3 at Q8_0, and
# Q5 and below gets Q4_K_M so the pair keeps shrinking together. Either can be swapped
# in the encoder picker afterwards.
#
# (quant, file in the repo, transformer GB). Quality-descending: this is also the
# order `_default_for` picks in, so the best installed build is the default.
_KLEIN_QUANTS = {
    "9b": [
        ("BF16", "flux-2-klein-9b-BF16.gguf", 18.16),
        ("Q8_0", "flux-2-klein-9b-Q8_0.gguf", 10.25),
        ("fp8", "flux-2-klein-9b-fp8.safetensors", 9.43),
        ("Q6_K", "flux-2-klein-9b-Q6_K.gguf", 7.87),
        ("Q5_K_M", "flux-2-klein-9b-Q5_K_M.gguf", 7.02),
        ("Q5_K_S", "flux-2-klein-9b-Q5_K_S.gguf", 6.94),
        ("Q5_1", "flux-2-klein-9b-Q5_1.gguf", 7.25),
        ("Q5_0", "flux-2-klein-9b-Q5_0.gguf", 6.71),
        ("Q4_K_M", "flux-2-klein-9b-Q4_K_M.gguf", 5.67),
        ("Q4_K_S", "flux-2-klein-9b-Q4_K_S.gguf", 5.83),
        ("Q4_1", "flux-2-klein-9b-Q4_1.gguf", 6.16),
        ("Q4_0", "flux-2-klein-9b-Q4_0.gguf", 5.62),
        ("Q3_K_M", "flux-2-klein-9b-Q3_K_M.gguf", 4.77),
        ("Q3_K_S", "flux-2-klein-9b-Q3_K_S.gguf", 4.69),
        ("Q2_K", "flux-2-klein-9b-Q2_K.gguf", 3.40),
    ],
    "4b": [
        ("BF16", "flux-2-klein-4b-BF16.gguf", 7.75),
        ("Q8_0", "flux-2-klein-4b-Q8_0.gguf", 4.30),
        ("fp8", "flux-2-klein-4b-fp8.safetensors", 4.07),
        ("Q6_K", "flux-2-klein-4b-Q6_K.gguf", 3.41),
        ("Q5_K_M", "flux-2-klein-4b-Q5_K_M.gguf", 3.07),
        ("Q5_K_S", "flux-2-klein-4b-Q5_K_S.gguf", 3.05),
        ("Q5_1", "flux-2-klein-4b-Q5_1.gguf", 3.15),
        ("Q5_0", "flux-2-klein-4b-Q5_0.gguf", 2.92),
        ("Q4_K_M", "flux-2-klein-4b-Q4_K_M.gguf", 2.60),
        ("Q4_K_S", "flux-2-klein-4b-Q4_K_S.gguf", 2.58),
        ("Q4_1", "flux-2-klein-4b-Q4_1.gguf", 2.69),
        ("Q4_0", "flux-2-klein-4b-Q4_0.gguf", 2.46),
        ("Q3_K_M", "flux-2-klein-4b-Q3_K_M.gguf", 2.12),
        ("Q3_K_S", "flux-2-klein-4b-Q3_K_S.gguf", 2.10),
        ("Q2_K", "flux-2-klein-4b-Q2_K.gguf", 1.83),
    ],
}

# Qwen3 encoder GGUFs per [klein] size: (high tier, low tier) as (file, GB).
_KLEIN_ENCODERS = {
    "9b": ("unsloth/Qwen3-8B-GGUF", ("Qwen3-8B-Q8_0.gguf", 8.71), ("Qwen3-8B-Q4_K_M.gguf", 5.03)),
    "4b": ("unsloth/Qwen3-4B-GGUF", ("Qwen3-4B-Q8_0.gguf", 4.28), ("Qwen3-4B-Q4_K_M.gguf", 2.50)),
}
_KLEIN_VAE_GB = 0.17
_HIGH_TIER = ("BF16", "Q8_0", "fp8", "Q6_K")

_QUANT_NOTES = {
    "BF16": "GGUF at bf16 — the unquantized weights, but memory-mapped read-only, so it "
            "loads on machines with less RAM than the safetensors release needs.",
    "Q8_0": "8-bit GGUF (blocks of 32 with a scale each).",
    "fp8": "Black Forest Labs' own fp8 build (safetensors). Stored in 8 bits; Ampere "
           "and older GPUs compute it in bf16.",
    "Q6_K": "6-bit K-quant GGUF.",
    "Q5_K_M": "5-bit K-quant GGUF, medium.",
    "Q5_K_S": "5-bit K-quant GGUF, small.",
    "Q5_1": "5-bit legacy GGUF (with offset).",
    "Q5_0": "5-bit legacy GGUF.",
    "Q4_K_M": "4-bit K-quant GGUF, medium.",
    "Q4_K_S": "4-bit K-quant GGUF, small.",
    "Q4_1": "4-bit legacy GGUF (with offset).",
    "Q4_0": "4-bit legacy GGUF.",
    "Q3_K_M": "3-bit K-quant GGUF, medium.",
    "Q3_K_S": "3-bit K-quant GGUF, small.",
    "Q2_K": "2-bit K-quant GGUF — the smallest build, for very small cards.",
}


def _resources(t_gb: float, e_gb: float, mmapped: bool) -> dict:
    """Recommended system resources for a transformer + encoder pair. Estimates.

    `vram_gb` keeps both resident (fastest); `min_vram_gb` relies on ComfyUI offloading
    one while the other runs. A GGUF is mapped read-only into page cache, so system RAM
    only has to hold whichever model is parked; a safetensors file is mapped privately
    and counts in full (see the ENOMEM note on [klein] 9B Q8_0's origin, git a738e3f).
    """
    ram = max(t_gb, e_gb) + 6 if mmapped else t_gb + e_gb + 6
    return {"vram_gb": round(t_gb + e_gb + 3), "min_vram_gb": round(max(t_gb, e_gb) + 2),
            "ram_gb": round(ram)}


def _klein_variants(size: str) -> list[dict]:
    """Every quantized build of [klein] `size` ("9b"/"4b") as a catalog bundle."""
    enc_repo, enc_hi, enc_lo = _KLEIN_ENCODERS[size]
    name = size.upper()
    out = []
    for quant, fname, t_gb in _KLEIN_QUANTS[size]:
        enc_file, e_gb = enc_hi if quant in _HIGH_TIER else enc_lo
        fp8 = quant == "fp8"
        # Q8_0 keeps the id it shipped with; settings (encoder, LoRAs) are keyed on it.
        suffix = {"Q8_0": "q8", "BF16": "bf16-gguf"}.get(quant, quant.lower())
        repo = (f"black-forest-labs/FLUX.2-klein-{size}-fp8" if fp8
                else f"unsloth/FLUX.2-klein-{name}-GGUF")
        out.append({
            "id": f"flux2-klein-{size}-{suffix}",
            "label": f"FLUX.2 [klein] {name} — {quant}",
            "family": FAMILY_FLUX2,
            "group": f"flux2-klein-{size}",
            "quant_of": f"flux2-klein-{size}",
            "quant": quant,
            "roles": [ROLE_CREATE, ROLE_EDIT],
            "blurb": _QUANT_NOTES[quant],
            "size_gb": round(t_gb + e_gb + _KLEIN_VAE_GB, 1),
            **_resources(t_gb, e_gb, mmapped=not fp8),
            # BFL gates its 9B repos (the fp8 one included); its 4B and unsloth's
            # mirrors are open.
            "gated": fp8 and size == "9b",
            "unet": fname,
            # A .gguf loads through UnetLoaderGGUF and ignores this; the fp8 file is
            # already quantized and must not be cast again.
            "weight_dtype": "default",
            "clip": enc_file,
            "vae": "flux2-klein-vae.safetensors",
            "files": [
                (repo, fname, "unet"),
                (enc_repo, enc_file, "text_encoders"),
                (f"unsloth/FLUX.2-klein-{name}", "vae/diffusion_pytorch_model.safetensors",
                 "vae", "flux2-klein-vae.safetensors"),
            ],
        })
    return out


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
    # FLUX 3 is its own family, not a FLUX.2 variant, although ComfyUI derives its model
    # class from Flux2 and it decodes through the same 128-channel VAE. The graph is
    # different where it matters: a plain KSampler on euler/simple (the timestep shift
    # is baked into ComfyUI's Flux3 config, so there is no Flux2Scheduler), no
    # FluxGuidance (the distilled checkpoint has no guidance embedding), and a
    # Qwen3-VL-4B encoder that loads only under CLIP type `flux3`.
    #
    # From BFL's ComfyUI packaging, not the reference repo (flux-3-image-klein). Those
    # are the same distilled weights, but the reference repo ships its encoder in
    # diffusers layout, which is the re-keying trap FLUX.2 [dev]'s encoder hit (below).
    # The packaging's files load as they are. bf16 throughout: the repo's int8/nvfp4
    # variants exist for 8 GB cards and Blackwell, and on this GPU they only cost quality.
    {
        "id": "flux3-klein",
        "group": "flux3-klein",
        "label": "FLUX 3 Image Klein — from Black Forest Labs",
        "family": FAMILY_FLUX3,
        "roles": [ROLE_CREATE, ROLE_EDIT],
        "blurb": ("BFL's newest distilled model: 16 steps, no guidance, and edits or "
                  "combines up to 10 reference images. Gated: accept the licence (FLUX "
                  "Kommunity — non-commercial, or under $5M revenue) on HuggingFace and "
                  "save a token first. Full bf16."),
        "size_gb": 20.3,
        "vram_gb": 31,
        "min_vram_gb": 13,
        "ram_gb": 26,
        "gated": True,
        "unet": "flux3-image-klein-bf16.safetensors",
        "weight_dtype": "default",
        "clip": "qwen3vl_4b_bf16.safetensors",
        "vae": "flux2-vae.safetensors",
        "files": [
            ("black-forest-labs/flux-3-image-klein-comfy",
             "diffusion_models/flux3-image-klein-bf16.safetensors", "unet"),
            ("black-forest-labs/flux-3-image-klein-comfy",
             "text_encoders/qwen3vl_4b_bf16.safetensors", "text_encoders"),
            # Not the same file as either FLUX.2 bundle's VAE, which are saved under their
            # own names, so this one keeps the upstream name without colliding.
            ("black-forest-labs/flux-3-image-klein-comfy",
             "vae/flux2-vae.safetensors", "vae"),
        ],
    },
    {
        "id": "flux2-dev-bfl",
        "group": "flux2-dev",
        "label": "FLUX.2 [dev] — from Black Forest Labs",
        "family": FAMILY_FLUX2,
        "roles": [ROLE_CREATE, ROLE_EDIT],
        "blurb": ("The full bf16 release, straight from Black Forest Labs. Gated: accept "
                  "the licence on HuggingFace and save a token first. 100 GB on disk — "
                  "the weights are cast to fp8 at load to fit the GPU."),
        "size_gb": 100.4,
        "vram_gb": 46,
        # Estimate: the 64 GB transformer is mapped in full before the fp8 cast.
        "min_vram_gb": 34,
        "ram_gb": 96,
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
        "group": "flux2-klein-9b",
        "label": "FLUX.2 [klein] 9B — from Black Forest Labs",
        "family": FAMILY_FLUX2,
        "roles": [ROLE_CREATE, ROLE_EDIT],
        "blurb": ("The distilled 9B FLUX.2, straight from Black Forest Labs. Gated: accept "
                  "the licence on HuggingFace and save a token first. Far lighter than "
                  "[dev] and runs in bf16 with no quantization."),
        "size_gb": 34.7,
        "vram_gb": 32,
        "min_vram_gb": 20,
        "ram_gb": 40,
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
    # The bf16 [klein] 9B above loads only on a machine with more RAM than its 16.9 GiB
    # transformer: safetensors maps it privately, and with no swap the kernel refuses
    # the mapping outright (ENOMEM). Its GGUF builds below are mapped read-only, so they
    # live in page cache and don't count against RAM — Q8_0 runs on 16 GB.
    *_klein_variants("9b"),
    # [klein] 4B, unquantized. ComfyUI's single-file packaging: the transformer and a
    # ready-to-load Qwen3-4B encoder, so unlike 9B nothing needs stitching. Ungated.
    {
        "id": "flux2-klein-4b",
        "label": "FLUX.2 [klein] 4B — from Black Forest Labs",
        "family": FAMILY_FLUX2,
        "group": "flux2-klein-4b",
        "roles": [ROLE_CREATE, ROLE_EDIT],
        "blurb": ("The distilled 4B FLUX.2 in full bf16 — the lightest FLUX.2, still "
                  "creating and editing with multiple reference images. Ungated — no "
                  "HuggingFace token needed."),
        "size_gb": 16.0,
        **_resources(7.75, 8.04, mmapped=False),
        "gated": False,
        "unet": "flux-2-klein-4b.safetensors",
        "weight_dtype": "default",
        "clip": "qwen_3_4b.safetensors",
        "vae": "flux2-klein-vae.safetensors",
        "files": [
            ("Comfy-Org/Flux-2-Klein-4B",
             "split_files/diffusion_models/flux-2-klein-4b.safetensors", "unet"),
            ("Comfy-Org/Flux-2-Klein-4B",
             "split_files/text_encoders/qwen_3_4b.safetensors", "text_encoders"),
            ("unsloth/FLUX.2-klein-4B", "vae/diffusion_pytorch_model.safetensors",
             "vae", "flux2-klein-vae.safetensors"),
        ],
    },
    *_klein_variants("4b"),
    # Qwen splits create and edit across two separately fine-tuned checkpoints of one
    # 20B architecture — the same shape FLUX.1 had, where dev created and Kontext
    # edited. So a Qwen bundle ships both: `unet` creates, `unet_edit` edits, and
    # `_default_for` picks between them by role. One install covers every image mode.
    #
    # They are paired by generation, not mixed: the December text-to-image refresh
    # (2512) goes with the December edit model (2511, the newest edit release — there
    # is no "Edit 2512"), and the August originals go together. Both transformers in a
    # bundle share its text encoder and VAE, which is why a 50 GB bundle is two 20 GB
    # transformers plus ~10 GB of parts, not two 30 GB models.
    #
    # `unet_labels` names them apart in the picker. Without it both rows render the
    # bundle's label and read as one model behaving differently per tab.
    #
    # `edit_encode` names the node that turns the instruction *and* the reference images
    # into conditioning, and applies to `unet_edit` alone. Qwen does not use
    # ReferenceLatent: the encode node takes the images directly, tokenizes them for the
    # VL encoder and VAE-encodes them as references itself, so the whole `_conditioning`
    # chain collapses into one node. The generations differ — 2511 takes up to three
    # images through the "Plus" node, the original takes one — which is why the class is
    # named per bundle rather than assumed.
    #
    # `ref_method` is how multiple references are laid out; 2511 wants
    # "index_timestep_zero" and ships that in its own template. Pinned rather than left
    # to ComfyUI's checkpoint-sniffed default (model_detection.py:823) for the same
    # reason `flux_client.REF_METHOD` is: an upgrade must not silently change it.
    {
        "id": "qwen-image-2512",
        "group": "qwen-image",
        "label": "Qwen-Image 2512 + Edit 2511 — fp8",
        "family": FAMILY_QWEN,
        "roles": [ROLE_CREATE, ROLE_EDIT],
        "blurb": ("Alibaba's 20B model, December refresh — both halves. The best there is "
                  "at rendering legible text inside an image, English and Chinese alike, "
                  "and the edit half takes up to three reference images at once. Ungated — "
                  "no HuggingFace token needed. Two 20 GB transformers: one loads at a "
                  "time, so it needs no more VRAM than either alone."),
        "size_gb": 50.6,
        # Estimate: one fp8 transformer resident (20.4 GB) plus activations at Qwen's
        # native 1328x1328. Only ever one at a time — ComfyUI evicts the create half
        # before loading the edit half — and the 9.4 GB encoder is a separate load it
        # evicts too. Measure on real hardware and correct this.
        "vram_gb": 28,
        "min_vram_gb": 23,
        "ram_gb": 36,
        "gated": False,
        "unet": "qwen_image_2512_fp8_e4m3fn.safetensors",
        "unet_edit": "qwen_image_edit_2511_fp8mixed.safetensors",
        "unet_labels": {
            "qwen_image_2512_fp8_e4m3fn.safetensors": "Qwen-Image 2512",
            "qwen_image_edit_2511_fp8mixed.safetensors": "Qwen-Image Edit 2511",
        },
        # Both are already quantized — the create half plain fp8, the edit half fp8mixed,
        # which deliberately keeps some layers higher. Casting either again would only
        # degrade it, the same reasoning FLUX.2's checkpoint gets.
        "weight_dtype": "default",
        # Qwen conditions on Qwen2.5-VL-7B, and both halves share the one encoder.
        # ComfyUI's `qwen_image` CLIP type reads it directly, so unlike FLUX.2's
        # encoders there is nothing to stitch or re-key.
        "clip": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
        "vae": "qwen_image_vae.safetensors",
        "edit_encode": "TextEncodeQwenImageEditPlus",
        "ref_method": "index_timestep_zero",
        # Qwen publishes diffusers-sharded weights (Qwen/Qwen-Image-2512 and
        # Qwen/Qwen-Image-Edit-2511); these are ComfyUI's single-file repackages of the
        # same releases, which is what CLIPLoader and UNETLoader take. Same arrangement
        # as the Wan bundle below.
        "files": [
            ("Comfy-Org/Qwen-Image_ComfyUI",
             "split_files/diffusion_models/qwen_image_2512_fp8_e4m3fn.safetensors", "unet"),
            ("Comfy-Org/Qwen-Image-Edit_ComfyUI",
             "split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors", "unet"),
            ("Comfy-Org/Qwen-Image_ComfyUI",
             "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors", "text_encoders"),
            ("Comfy-Org/Qwen-Image_ComfyUI",
             "split_files/vae/qwen_image_vae.safetensors", "vae"),
        ],
    },
    {
        "id": "qwen-image",
        "group": "qwen-image",
        "label": "Qwen-Image + Edit — fp8",
        "family": FAMILY_QWEN,
        "roles": [ROLE_CREATE, ROLE_EDIT],
        "blurb": ("The original August 2025 pair, superseded by the December one above — "
                  "the same architecture, retrained. Worth installing only to compare the "
                  "two. Its edit half takes one reference image rather than three. Shares "
                  "the encoder and VAE with 2512, so it costs ~41 GB once that is in."),
        # The whole bundle. `needed_gb` subtracts whatever 2512 already put on disk, so
        # the button shows only what is actually outstanding.
        "size_gb": 50.5,
        "vram_gb": 28,
        "min_vram_gb": 23,
        "ram_gb": 36,
        "gated": False,
        "unet": "qwen_image_fp8_e4m3fn.safetensors",
        "unet_edit": "qwen_image_edit_fp8_e4m3fn.safetensors",
        "unet_labels": {
            "qwen_image_fp8_e4m3fn.safetensors": "Qwen-Image",
            "qwen_image_edit_fp8_e4m3fn.safetensors": "Qwen-Image Edit",
        },
        "weight_dtype": "default",
        "clip": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
        "vae": "qwen_image_vae.safetensors",
        # The original edit model predates the "Plus" node: one image input, and no
        # multi-reference layout to choose because there is only ever one reference.
        "edit_encode": "TextEncodeQwenImageEdit",
        "ref_method": None,
        "files": [
            ("Comfy-Org/Qwen-Image_ComfyUI",
             "split_files/diffusion_models/qwen_image_fp8_e4m3fn.safetensors", "unet"),
            ("Comfy-Org/Qwen-Image-Edit_ComfyUI",
             "split_files/diffusion_models/qwen_image_edit_fp8_e4m3fn.safetensors", "unet"),
            ("Comfy-Org/Qwen-Image_ComfyUI",
             "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors", "text_encoders"),
            ("Comfy-Org/Qwen-Image_ComfyUI",
             "split_files/vae/qwen_image_vae.safetensors", "vae"),
        ],
    },
    {
        "id": "wan22-i2v-a14b-fp16",
        "group": "wan22",
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
        "min_vram_gb": 31,
        "ram_gb": 75,
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

# --------------------------------------------------------------------------- #
# Control preprocessors
# --------------------------------------------------------------------------- #
# The models that turn a source image into a *control map* — the depth map or pose
# skeleton a control generation conditions on. Deliberately not bundles: a bundle is a
# transformer plus the encoder and VAE that make it samplable, and dispatches a graph
# family. These have no family, serve no role, and are never picked as a model; they
# are one file each, loaded by one preprocess graph. Giving them their own registry
# keeps them out of `BUNDLES`, which is what every model-facing lookup iterates.
#
# `kind` is the control map they produce and is the key the rest of the app uses. Canny
# has no entry here on purpose — ComfyUI's `Canny` node is pure kornia, so edge control
# needs no weights at all and is always available.
PREPROCESSORS = [
    {
        "id": "depth-anything-3",
        "kind": "depth",
        "label": "Depth Anything 3 (Mono, Large)",
        "note": "Encodes the whole scene in 3-D — a chair, a closet top, a subject's "
                "contact with them, and which limb is in front. The strongest single "
                "control for poses that interact with something.",
        "size_gb": 1.4,
        "files": [("Comfy-Org/Depth-Anything-3",
                   "geometry_estimation/depth_anything_3_mono_large.safetensors",
                   "geometry_estimation")],
    },
    {
        "id": "sdpose",
        "kind": "pose",
        "label": "SDPose (whole body)",
        "note": "An OpenPose skeleton: limb configuration only, nothing about the "
                "scene. Pairs with depth rather than replacing it.",
        "size_gb": 2.6,
        "files": [("Comfy-Org/SDPose",
                   "checkpoints/sdpose_wholebody_fp16.safetensors", "checkpoints")],
    },
]

_BY_KIND = {p["kind"]: p for p in PREPROCESSORS}

# Every control map the app can build, in the order the UI offers them. Depth leads
# because it is the one that solves the hard cases; canny needs no download and so has
# no PREPROCESSORS entry to be ordered by.
CONTROL_KINDS = ("depth", "canny", "pose")


def preprocessor(kind: str) -> dict | None:
    """The model a control kind needs, or None if it needs no weights (canny)."""
    return _BY_KIND.get(kind or "")


def preprocessor_file(kind: str) -> Path | None:
    """Where that model's single file lives on disk."""
    p = preprocessor(kind)
    return file_path(p["files"][0]) if p else None


def preprocessor_installed(kind: str) -> bool:
    """Whether a control kind can run right now. Kinds with no model are always ready."""
    path = preprocessor_file(kind)
    return True if path is None else path.exists()


def get_preprocessor(pid: str) -> dict:
    for p in PREPROCESSORS:
        if p["id"] == pid:
            return p
    raise ValueError(f"Unknown control preprocessor '{pid}'.")


def is_klein(bundle: dict | None) -> bool:
    """Whether a bundle is any build of FLUX.2 [klein] (9B or 4B, any precision) —
    they share the distilled step count and guidance."""
    return bool(bundle) and bundle.get("group") in KLEIN_GROUPS


def lora_base(bundle: dict | None) -> str | None:
    """The bundle id LoRAs are made for: a quantized build's parent, else itself.
    Every build sharing one `lora_base` takes the same adapters."""
    if not bundle:
        return None
    return bundle.get("quant_of") or bundle["id"]


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
        edit_unet = b.get("unet_edit")
        if not edit_unet:
            return list(b["roles"])
        # A bundle whose transformers split by role states the *union* in `roles` —
        # that is what the install offers — but the two files are not interchangeable
        # and the picker has to say so per file, or a create-only transformer gets
        # offered for an edit it cannot do.
        #
        # The split is deliberately lopsided. The edit half keeps every role: its
        # encode node takes its images optionally, so with none attached it is a
        # perfectly good text-to-image model, and control (an edit-role op here) is
        # its job anyway. The create half loses `edit` — a plain CLIPTextEncode has
        # nowhere to put a source image, so an edit asked of it silently ignores the
        # picture and answers the prompt alone.
        if os.path.basename(unet or "") == edit_unet:
            return list(b["roles"])
        return [r for r in b["roles"] if r != ROLE_EDIT]
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
