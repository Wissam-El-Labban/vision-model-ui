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
