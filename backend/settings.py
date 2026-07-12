"""Server-side settings that outlive a browser session.

Right now that's just the HuggingFace token, which the model installer needs for
gated repos. It's a credential, so: written 0600, never returned to the browser (the
API reports only whether one is present), and read from the environment if the user
would rather not store it at all.
"""
import json
import os
from pathlib import Path

SETTINGS_PATH = Path(__file__).resolve().parent / "data" / "settings.json"

ENV_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")


def _read() -> dict:
    try:
        return json.loads(SETTINGS_PATH.read_text())
    except (OSError, ValueError):
        return {}


def _write(data: dict) -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Create it 0600 up front rather than chmod-ing after the write, so the token is
    # never briefly world-readable on disk.
    fd = os.open(SETTINGS_PATH, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(data, f)


def hf_token() -> str:
    """The saved token, or one from the environment. Empty string if neither."""
    saved = _read().get("hf_token") or ""
    if saved:
        return saved
    for var in ENV_VARS:
        if os.environ.get(var):
            return os.environ[var]
    return ""


def hf_token_source() -> str | None:
    """Where the token came from — for the UI, which never sees the value itself."""
    if _read().get("hf_token"):
        return "saved"
    if any(os.environ.get(v) for v in ENV_VARS):
        return "env"
    return None


def set_hf_token(token: str) -> None:
    data = _read()
    data["hf_token"] = token.strip()
    _write(data)


def clear_hf_token() -> None:
    data = _read()
    data.pop("hf_token", None)
    _write(data)


def text_encoders() -> dict:
    """Per-model text-encoder overrides: {bundle_id: filename}.

    A bundle ships with the encoder it was trained against, but the file is separable
    and interchangeable within an architecture — a smaller quant of the same encoder,
    say. So the choice is remembered here rather than baked into the catalog.
    """
    return _read().get("text_encoders") or {}


def set_text_encoder(bundle_id: str, name: str) -> None:
    """Point a model at a different text encoder. Empty name restores its default."""
    data = _read()
    tes = data.get("text_encoders") or {}
    if name:
        tes[bundle_id] = name
    else:
        tes.pop(bundle_id, None)
    data["text_encoders"] = tes
    _write(data)
