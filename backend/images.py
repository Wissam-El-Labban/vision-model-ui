"""Image encoding helpers shared by the API layer and the content-addressed store.

Also the single place that forces HuggingFace fully offline for the serving
process. These must be set before `huggingface_hub` is imported anywhere, so this
module is imported early by `main.py`.

The ONE exception is a deliberate model download, which runs in a separate
subprocess (see `flux_client.pull_unet`) whose environment has these flags
removed — network access is confined to the moment the user clicks "Add", and the
long-lived serving process never has network-enabled HF access.
"""
from __future__ import annotations

import io
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


# The formats the store keeps as-is. `normalize_for_store` re-encodes anything else
# to PNG on the way in, so `sniff` only ever has to answer for these — a format
# missing here must not reach the store, or it lands under the wrong extension.
_FMT_TO_MIME_EXT = {
    "PNG": ("image/png", "png"),
    "JPEG": ("image/jpeg", "jpg"),
    "WEBP": ("image/webp", "webp"),
}
MIME_TO_EXT = {mime: ext for mime, ext in _FMT_TO_MIME_EXT.values()}

# The longest edge the store will keep. FLUX samples at ~1 MP (see
# `flux_client._flux2_resolution`), so 2048 leaves better than 2x linear headroom
# over the sampled resolution while still capping a 12 MP phone photo. Above this
# we resample once, here, with a good filter — rather than letting the browser do
# it badly and JPEG the result, which is what used to happen.
MAX_STORE_DIM = 2048


def sniff(raw: bytes) -> tuple[str, str]:
    """(mime, ext) read from the bytes themselves.

    Authoritative on purpose: a client's `data:` prefix is a claim, not evidence.
    `Image.open` only reads the header — pixels load lazily — so this costs a few
    bytes, not a decode.
    """
    from PIL import Image  # PLC0415

    try:
        fmt = Image.open(io.BytesIO(raw)).format
    except Exception:
        return ("image/jpeg", "jpg")
    return _FMT_TO_MIME_EXT.get(fmt or "", ("image/jpeg", "jpg"))


def normalize_for_store(raw: bytes) -> bytes:
    """The bytes the store will actually hold.

    The rule, in one place: **bytes are stored verbatim if they are already a
    format the store names (PNG/JPEG/WEBP) and no larger than MAX_STORE_DIM.
    Otherwise exactly one re-encode to PNG — LANCZOS-downscaled if oversized.**
    Every consumer resamples down from this, so this is the last copy that still
    has the original's detail — for a face, that detail (pores, eyelashes, edges)
    is what survives into the VAE.

    The format check is what keeps `sniff` honest: without it a GIF would sail
    through verbatim, sniff would fail to name it, and the file would be stored as
    `.jpg` and served as image/jpeg while holding GIF bytes.

    Deliberately does *not* apply EXIF rotation on the verbatim path: that would
    force a lossy re-encode of every phone photo, which is the exact cost this
    function exists to avoid. Orientation is applied on read instead (see
    `main.py:_resolve`).

    Deterministic, so re-uploading a file still dedupes to the same hash.
    """
    from PIL import Image  # PLC0415

    try:
        img = Image.open(io.BytesIO(raw))
        if img.format in _FMT_TO_MIME_EXT and max(img.size) <= MAX_STORE_DIM:
            return raw
        img = img.convert("RGB")
        if max(img.size) > MAX_STORE_DIM:
            img.thumbnail((MAX_STORE_DIM, MAX_STORE_DIM), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return raw  # not decodable as an image; store what we were given


def pil_to_data_url(image, max_size: int | None = None, fmt: str = "PNG") -> str:
    """Encode a PIL image as a `data:image/...;base64,...` URL for the store.

    PNG by default: a generated image is frequently the source of the next edit,
    and a JPEG round-trip per generation compounds over a chain of them. Pass
    fmt="JPEG" for thumbnails, where there's no detail worth keeping.
    `max_size` downscales (longest edge) when set.
    """
    import base64  # PLC0415

    img = image.convert("RGB")
    if max_size:
        img = img.copy()
        img.thumbnail((max_size, max_size))
    buf = io.BytesIO()
    if fmt == "JPEG":
        img.save(buf, format="JPEG", quality=92)
    else:
        img.save(buf, format="PNG")
    mime = "image/jpeg" if fmt == "JPEG" else "image/png"
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode()


def image_to_b64(path, max_dim: int = 1280) -> str:
    """A downscaled JPEG copy of a stored image, for a vision model.

    Vision models tokenize by resolution, so they want a small image — but that's
    a property of *this consumer*, not of the image, so the downscale happens here
    at send time rather than by degrading what we store. LANCZOS from the stored
    original beats the browser's bilinear from whatever it had.

    Returns bare base64: Ollama wants the payload, not a `data:` URL.
    """
    import base64  # PLC0415

    from PIL import Image, ImageOps  # PLC0415

    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    if max(img.size) > max_dim:
        img.thumbnail((max_dim, max_dim), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()
