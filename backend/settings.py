"""Server-side settings that outlive a browser session.

Two of them are credentials — the HuggingFace token the model installer needs for
gated repos, and the CivitAI key most LoRA downloads there now require. Both are
handled the same way: written 0600, never returned to the browser (the API reports
only whether one is present), and read from the environment if the user would rather
not store them at all.
"""
import json
import os
import tempfile
import threading
from pathlib import Path

SETTINGS_PATH = Path(__file__).resolve().parent / "data" / "settings.json"

# Every setting lives in one file, so every save is a read-modify-write of the whole
# thing — and the API endpoints that call them are sync `def`, which FastAPI runs in a
# threadpool. Two of those overlapping used to lose one update outright; dragging the
# LoRA strength slider fires one save per notch, so "overlapping" was the normal case
# rather than the rare one. A process-wide lock makes the cycle indivisible.
#
# A thread lock and not a file lock because the app is one uvicorn process (see
# run.sh, which starts it with no --workers). Under multiple workers this would have
# to become an flock on the settings file.
_LOCK = threading.RLock()

ENV_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
# CivitAI's own docs call it an API key; both spellings are in circulation in people's
# shells, so accept either rather than silently ignoring the one they exported.
CIVITAI_ENV_VARS = ("CIVITAI_TOKEN", "CIVITAI_API_KEY")


def _read() -> dict:
    """The whole settings file, or `{}` if there isn't one yet.

    An unreadable file is *not* quietly treated as an empty one. That was the second
    half of a data-loss bug: a caller that read `{}` went on to write back only the
    key it happened to know about, so a settings file that couldn't be parsed for a
    moment came back permanently missing the token and the encoder pick. A file that
    exists but won't parse is preserved beside itself instead, so the contents are
    recoverable rather than overwritten on the next save.
    """
    try:
        raw = SETTINGS_PATH.read_text()
    except FileNotFoundError:
        return {}
    except OSError:
        # Can't read it *and* can't rule out that it's fine — refuse rather than
        # hand back an empty dict a writer would then make true.
        raise
    if not raw.strip():
        return {}
    try:
        return json.loads(raw)
    except ValueError:
        keep = SETTINGS_PATH.with_suffix(".corrupt.json")
        try:
            os.replace(SETTINGS_PATH, keep)
        except OSError:
            pass
        return {}


def _write(data: dict) -> None:
    """Replace the settings file atomically.

    Write-in-place was the first half of the data-loss bug: `O_TRUNC` emptied the file
    and *then* filled it, so a concurrent reader could catch it at zero bytes and
    conclude there were no settings at all — and two concurrent writers could splice
    each other into JSON that parsed as nothing. A sibling file plus `os.replace`
    (atomic on POSIX) means a reader always sees one whole version or the other.
    """
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    # mkstemp creates 0600, so the token is never briefly world-readable on disk.
    # Same directory as the target, because os.replace is only atomic within a
    # filesystem.
    fd, tmp = tempfile.mkstemp(
        dir=str(SETTINGS_PATH.parent), prefix=".settings-", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
            f.flush()
            # Get the bytes down before the rename, so a crash can't leave the new
            # name pointing at an empty file.
            os.fsync(f.fileno())
        os.replace(tmp, SETTINGS_PATH)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


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
    """Save a token, replacing any previous one.

    Exactly one token is ever stored: `_write` replaces the whole settings file, so
    the old value is gone from disk rather than kept beside the new one. Nothing
    caches it in memory either — `hf_token()` re-reads the file on every call, so a
    replacement takes effect on the very next download. An empty token clears the
    setting instead of storing a blank one.
    """
    with _LOCK:
        data = _read()
        tok = (token or "").strip()
        if tok:
            data["hf_token"] = tok
        else:
            data.pop("hf_token", None)
        _write(data)


def clear_hf_token() -> None:
    with _LOCK:
        data = _read()
        data.pop("hf_token", None)
        _write(data)


def civitai_token() -> str:
    """The saved CivitAI key, or one from the environment. Empty string if neither.

    Optional in a way `hf_token` isn't for gated repos: some adapters download
    anonymously. CivitAI requires a key for most of them, and answers a missing one
    with a redirect to its login page rather than a 401 — see `pull_lora_civitai`.
    """
    saved = _read().get("civitai_token") or ""
    if saved:
        return saved
    for var in CIVITAI_ENV_VARS:
        if os.environ.get(var):
            return os.environ[var]
    return ""


def civitai_token_source() -> str | None:
    """Where the CivitAI key came from — for the UI, which never sees the value."""
    if _read().get("civitai_token"):
        return "saved"
    if any(os.environ.get(v) for v in CIVITAI_ENV_VARS):
        return "env"
    return None


def set_civitai_token(token: str) -> None:
    """Save a CivitAI key, replacing any previous one. Empty clears it.

    Same one-token-only guarantee as `set_hf_token`: `_write` replaces the whole
    file, so the old value leaves the disk rather than sitting beside the new one.
    """
    with _LOCK:
        data = _read()
        tok = (token or "").strip()
        if tok:
            data["civitai_token"] = tok
        else:
            data.pop("civitai_token", None)
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
    with _LOCK:
        data = _read()
        tes = data.get("text_encoders") or {}
        if name:
            tes[bundle_id] = name
        else:
            tes.pop(bundle_id, None)
        data["text_encoders"] = tes
        _write(data)


def loras() -> dict:
    """Per-transformer LoRA choices: {unet_filename: [{"name": ..., "strength": ...}, ...]}.

    Unlike a text encoder, no model has a default LoRA — an absent or empty list means
    "none", which is the state the user starts in and can always return to. Several
    adapters can be stacked on the same transformer at once.

    Keyed by the transformer file, not the bundle that shipped it. A LoRA is trained
    against one specific base, and one bundle can carry two: FLUX.1 pairs dev with
    Kontext, whose adapter ecosystems are entirely disjoint. Keying by bundle would
    chain a dev adapter onto Kontext whenever you edited.

    A pick may also carry `"control": True`, marking it as this transformer's *control
    adapter* — the one that teaches the model to obey a control map, and therefore the
    one the Control tab's strength dial scales. Flagged rather than detected from the
    filename: an adapter's name is a guess and its role is a fact the user knows. An
    absent flag means False, so nothing needs migrating.
    """
    raw = _read().get("loras") or {}
    # Pre-multi-LoRA settings.json files store one {"name", "strength"} object per
    # model rather than a list — normalize on read so an existing pick keeps working
    # instead of vanishing the first time this runs after the upgrade.
    return {model: [pick] if isinstance(pick, dict) else pick for model, pick in raw.items()}


def set_loras(model: str, picks: list[dict]) -> None:
    """Replace the whole set of LoRAs attached to one transformer. An empty list
    detaches everything (the "None" option)."""
    with _LOCK:
        data = _read()
        all_picks = data.get("loras") or {}
        if picks:
            all_picks[model] = picks
        else:
            all_picks.pop(model, None)
        data["loras"] = all_picks
        _write(data)


