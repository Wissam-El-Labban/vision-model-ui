"""Thin proxy helpers over the Ollama HTTP API.

The browser never talks to Ollama directly; the FastAPI backend owns every
Ollama call. Logic here is ported from the previous Streamlit app
(utils.py / app.py) and keeps the two load-bearing fixes from this branch:
`"think": False` and a (connect, read) timeout split.
"""
import json
import re
import subprocess
from urllib.parse import urlparse

import requests

DEFAULT_URL = "http://localhost:11434"

# Connect quickly (fail fast if Ollama is down) but allow a generous read
# timeout so a cold model load on the first request doesn't look like a hang.
CHAT_TIMEOUT = (10, 300)

GITHUB_LATEST = "https://api.github.com/repos/ollama/ollama/releases/latest"


# --------------------------------------------------------------------------- #
# Chat
# --------------------------------------------------------------------------- #
def context_size_for(messages):
    """Pick a context window large enough for the images in the request.

    The model default (often 4096) is too small once a real image is encoded
    into vision tokens, but a huge fixed value (e.g. 32768) makes Ollama
    allocate a big KV cache and load slowly. So scale modestly with the number
    of images and cap it.
    """
    n_images = sum(len(m.get("images") or []) for m in messages)
    return min(32768, max(8192, 4096 + 2048 * n_images))


def stream_chat(url, model, messages):
    """POST to Ollama's /api/chat and yield streaming events.

    Yields dicts: {"type": "token", "text": ...} for content, and a final
    {"type": "usage", ...} carrying the exact token counts Ollama reports
    (prompt_eval_count + eval_count) plus the num_ctx we used — the ground
    truth for the context-usage meter.

    Sends `think: False` so reasoning-capable models answer directly, and a
    `num_ctx` sized to fit the request's images (see context_size_for).
    """
    num_ctx = context_size_for(messages)
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "think": False,
        "options": {"num_ctx": num_ctx},
    }
    response = requests.post(
        f"{url}/api/chat",
        json=payload,
        stream=True,
        timeout=CHAT_TIMEOUT,
    )
    if response.status_code != 200:
        raise RuntimeError(f"Ollama returned {response.status_code}: {response.text}")

    for line in response.iter_lines():
        if not line:
            continue
        try:
            chunk = json.loads(line)
        except json.JSONDecodeError:
            continue
        content = (chunk.get("message") or {}).get("content")
        if content:
            yield {"type": "token", "text": content}
        if chunk.get("done"):
            prompt_tokens = chunk.get("prompt_eval_count") or 0
            eval_tokens = chunk.get("eval_count") or 0
            yield {
                "type": "usage",
                "used": prompt_tokens + eval_tokens,
                "prompt_tokens": prompt_tokens,
                "eval_tokens": eval_tokens,
                "num_ctx": num_ctx,
            }


def generate_title(url, model, first_user, first_assistant):
    """Ask the model for a short conversation title from the first exchange.

    Text-only (no images) so the vision context doesn't have to reload just to
    name a chat. Returns a cleaned 3-6 word title, capped in length. Reuses the
    same `think: False` behavior as chat and fails soft (returns "" on error).
    """
    system = (
        "You write short conversation titles. Reply with ONLY a 3 to 6 word "
        "title. No quotes, no trailing punctuation, no preamble."
    )
    user = f"User asked: {first_user}\n\nAssistant replied: {first_assistant[:500]}"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "think": False,
        "options": {"num_ctx": 4096},
    }
    try:
        response = requests.post(
            f"{url}/api/chat", json=payload, timeout=CHAT_TIMEOUT
        )
        if response.status_code != 200:
            return ""
        text = (response.json().get("message") or {}).get("content", "")
    except (requests.RequestException, ValueError):
        return ""

    # Clean: first line, strip surrounding quotes/whitespace, cap length.
    title = text.strip().splitlines()[0].strip() if text.strip() else ""
    title = title.strip('"').strip("'").rstrip(".").strip()
    return title[:50]


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def is_vision_model(url, model_name):
    """Check if a model supports vision by inspecting its details."""
    try:
        response = requests.post(
            f"{url}/api/show",
            json={"name": model_name},
            timeout=10,
        )
        if response.status_code != 200:
            return False
        info = response.json()

        # Capabilities are authoritative on modern Ollama — trust them. (The old
        # keyword heuristic gave false positives, e.g. qwen2.5 whose modelfile
        # text merely mentions "vision" but has no vision capability.)
        capabilities = info.get("capabilities")
        if capabilities is not None:
            return "vision" in capabilities

        # Fallback only for older Ollama that doesn't report capabilities.
        modelfile = (info.get("modelfile") or "").lower()
        template = (info.get("template") or "").lower()
        indicators = [
            "vision" in modelfile,
            "visual" in modelfile,
            "[img" in template,
            "clip" in modelfile,
            "mm_projector" in modelfile,
            "vision_tower" in modelfile,
            "image_processor" in modelfile,
        ]
        return any(indicators)
    except requests.RequestException:
        return False


def list_vision_models(url):
    """Return sorted names of installed vision-capable models."""
    try:
        response = requests.get(f"{url}/api/tags", timeout=5)
        if response.status_code != 200:
            return []
        models = response.json().get("models", [])
        vision = [m["name"] for m in models if is_vision_model(url, m["name"])]
        return sorted(vision)
    except requests.RequestException:
        return []


def list_all_models(url):
    """Return names of every installed model (for the remove dropdown)."""
    try:
        response = requests.get(f"{url}/api/tags", timeout=5)
        if response.status_code != 200:
            return []
        return [m["name"] for m in response.json().get("models", [])]
    except requests.RequestException:
        return []


def pull(url, name):
    """Stream `ollama pull` progress as raw JSON status lines."""
    response = requests.post(
        f"{url}/api/pull",
        json={"name": name},
        stream=True,
        timeout=600,
    )
    if response.status_code != 200:
        raise RuntimeError(f"Ollama returned {response.status_code}: {response.text}")
    for line in response.iter_lines():
        if line:
            yield line.decode("utf-8") + "\n"


def delete(url, name):
    response = requests.delete(f"{url}/api/delete", json={"name": name}, timeout=30)
    response.raise_for_status()
    return True


def running(url):
    response = requests.get(f"{url}/api/ps", timeout=5)
    response.raise_for_status()
    return response.json().get("models", [])


def unload_all(url):
    """Unload every loaded model to free VRAM. Returns the names unloaded."""
    unloaded = []
    for model in running(url):
        name = model.get("name", "")
        if not name:
            continue
        requests.post(
            f"{url}/api/generate",
            json={"model": name, "keep_alive": 0},
            timeout=10,
        )
        unloaded.append(name)
    return unloaded


# --------------------------------------------------------------------------- #
# Version / upgrade
# --------------------------------------------------------------------------- #
def _parse_version(text):
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", text or "")
    return tuple(int(x) for x in match.groups()) if match else None


def is_local(url):
    host = (urlparse(url).hostname or "").lower()
    return host in ("localhost", "127.0.0.1", "0.0.0.0", "::1", "")


def installed_version(url):
    try:
        response = requests.get(f"{url}/api/version", timeout=5)
        if response.status_code == 200:
            return response.json().get("version")
    except requests.RequestException:
        pass
    return None


def latest_version():
    try:
        response = requests.get(GITHUB_LATEST, timeout=5)
        if response.status_code == 200:
            tag = response.json().get("tag_name", "")
            return tag.lstrip("v") or None
    except requests.RequestException:
        pass
    return None


def version_info(url):
    installed = installed_version(url)
    latest = latest_version()
    update_available = False
    if installed and latest:
        pi, pl = _parse_version(installed), _parse_version(latest)
        if pi and pl:
            update_available = pl > pi
    return {
        "installed": installed,
        "latest": latest,
        "update_available": update_available,
        "is_local": is_local(url),
    }


def upgrade():
    """Run the official Ollama installer, streaming combined output lines.

    Linux only. The installer may need sudo; if it fails, the caller surfaces
    the error and the manual fallback command.
    """
    process = subprocess.Popen(
        ["bash", "-c", "curl -fsSL https://ollama.com/install.sh | sh"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in iter(process.stdout.readline, ""):
        yield line
    process.stdout.close()
    code = process.wait()
    if code != 0:
        yield (
            f"\n✗ Upgrade failed (exit {code}). Run manually:\n"
            "curl -fsSL https://ollama.com/install.sh | sh\n"
        )
    else:
        yield "\n✓ Ollama upgraded. Restart the Ollama service to use the new version.\n"
