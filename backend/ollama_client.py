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


def stream_chat_tools(url, model, messages, tools):
    """Like `stream_chat`, but offers the model tools and reports what it called.

    Yields the same {"type": "token"} and {"type": "usage"} events, plus one
    {"type": "tool_calls", "calls": [{"name": ..., "arguments": {...}}, ...]} if
    the model asked for any. Tool calls arrive whole inside a chunk's message
    rather than a token at a time, but a model may emit them across several
    chunks, so they're accumulated and reported once at the end — the caller
    wants the whole plan before it starts unloading GPUs on the strength of it.

    Ollama returns `arguments` already parsed as an object; older builds send it
    as a JSON string, so both are accepted.
    """
    num_ctx = context_size_for(messages)
    payload = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "stream": True,
        "think": False,
        "options": {"num_ctx": num_ctx},
    }
    response = requests.post(
        f"{url}/api/chat", json=payload, stream=True, timeout=CHAT_TIMEOUT
    )
    if response.status_code != 200:
        raise RuntimeError(f"Ollama returned {response.status_code}: {response.text}")

    calls = []
    for line in response.iter_lines():
        if not line:
            continue
        try:
            chunk = json.loads(line)
        except json.JSONDecodeError:
            continue
        message = chunk.get("message") or {}
        content = message.get("content")
        if content:
            yield {"type": "token", "text": content}
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            name = function.get("name")
            if not name:
                continue
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            calls.append({"name": name, "arguments": arguments or {}})
        if chunk.get("done"):
            if calls:
                yield {"type": "tool_calls", "calls": calls}
            prompt_tokens = chunk.get("prompt_eval_count") or 0
            eval_tokens = chunk.get("eval_count") or 0
            yield {
                "type": "usage",
                "used": prompt_tokens + eval_tokens,
                "prompt_tokens": prompt_tokens,
                "eval_tokens": eval_tokens,
                "num_ctx": num_ctx,
            }


def ask_once(url, model, system, user, num_ctx=2048):
    """One short text-only question, answered in full. "" on any failure.

    No images and a small context on purpose: this is for classification-shaped
    questions about what the user *asked for*, which the text answers on its own.
    Skipping the vision tokens is what keeps it cheap enough to run before the
    real turn.
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "think": False,
        "options": {"num_ctx": num_ctx},
    }
    try:
        response = requests.post(f"{url}/api/chat", json=payload, timeout=CHAT_TIMEOUT)
        if response.status_code != 200:
            return ""
        return ((response.json().get("message") or {}).get("content") or "").strip()
    except (requests.RequestException, ValueError):
        return ""


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
# Prompt enhancement
# --------------------------------------------------------------------------- #
# FLUX.2 is conditioned by a large LLM text encoder (Mistral-3 for [dev], Qwen3 for
# [klein]) and follows long natural prose — see the note above `PHOTOREAL_TEMPLATE`
# in flux_client.py for why tag salad hurts it. These prompts are written for that,
# and are split by mode because an edit is not a description: rewriting "make the
# jacket red" into a scene paragraph turns an edit into a regeneration, which is the
# one way an enhancer can make things actively worse.
_ENHANCE_SYSTEM = {
    "create": (
        "You are a prompt engineer for the FLUX.2 image model. It is conditioned by a "
        "large language model and follows long, natural prose. Never write "
        "comma-separated tag lists — they degrade this model.\n"
        "Rewrite the user's idea as a single vivid paragraph describing the finished "
        "photograph: the subject, what they are doing, the setting, camera and lens, "
        "lighting, composition, mood and style.\n"
        "If reference images are attached, describe their subjects accurately — "
        "appearance, clothing, distinguishing features — so the model reproduces them "
        "rather than inventing new ones.\n"
        "Keep every concrete detail the user specified. Invent only what they left "
        "open.\n"
        "Return the prompt only: no preamble, no quotes, no commentary."
    ),
    "edit": (
        "You are a prompt engineer for the FLUX.2 image editing model.\n"
        "The user gives an INSTRUCTION describing a change to the attached image. "
        "Rewrite it as a clearer, more specific instruction. It must REMAIN an "
        "instruction in the imperative. Never turn it into a description of a scene.\n"
        "Use the attached image to name exactly what to change and where it is.\n"
        "State explicitly what must stay unchanged: background, other subjects, pose, "
        "lighting, framing.\n"
        "Do not add camera, lens or style language unless the user asked to change "
        "those.\n"
        "Return the instruction only: no preamble, no quotes, no commentary."
    ),
    "compose": (
        "You are a prompt engineer for the FLUX.2 image model. It is conditioned by a "
        "large language model and follows long, natural prose. Never write "
        "comma-separated tag lists — they degrade this model.\n"
        "Several reference images are attached. The user wants a new image that "
        "combines them. Describe the finished image as a single vivid paragraph.\n"
        "Identify each reference's subject explicitly and say what it contributes, so "
        "the model knows which is which and reproduces each faithfully rather than "
        "blending them into someone new.\n"
        "Keep every concrete detail the user specified. Invent only what they left "
        "open.\n"
        "Return the prompt only: no preamble, no quotes, no commentary."
    ),
    "control": (
        "You are a prompt engineer for the FLUX.2 image model. It is conditioned by a "
        "large language model and follows long, natural prose. Never write "
        "comma-separated tag lists — they degrade this model.\n"
        "The attached image is a CONTROL MAP — a depth render, an edge trace or an "
        "OpenPose skeleton. It is not content and must never be described. It exists "
        "only to fix the pose and the layout.\n"
        "The user often writes an instruction ('make him clap his hands', 'do a "
        "T-pose'). That is the wrong shape here and produces a greyscale copy of the "
        "map, because an instruction gives the model nothing to render and imitating "
        "the reference is all that is left. Convert it into a description of the "
        "FINISHED PHOTOGRAPH, in which the subject is already in that pose.\n"
        "Describe the subject, their clothing, the setting, the lighting and the "
        "style, as a single vivid paragraph. Say it is a photograph.\n"
        "Never mention depth maps, skeletons, greyscale, silhouettes, poses maps or "
        "the control image itself.\n"
        "Keep every concrete detail the user specified. Invent only what they left "
        "open.\n"
        "Return the prompt only: no preamble, no quotes, no commentary."
    ),
    "animate": (
        "You are a prompt engineer for the Wan 2.2 image-to-video model.\n"
        "The attached image is the video's FIRST FRAME. It already fixes the subject, "
        "the setting, the lighting and the framing — do not describe them. Describing "
        "the scene again wastes the prompt and fights the frame the model is starting "
        "from.\n"
        "Describe only what HAPPENS over the next few seconds: how the subject moves, "
        "and how the camera moves (a slow push in, a pan left, a locked-off static "
        "shot). Name the camera move explicitly — it is the strongest control there "
        "is.\n"
        "It is one continuous shot. Never describe a cut, a new angle, or a second "
        "scene.\n"
        "Keep it to what five seconds can hold: one action, not a sequence of them.\n"
        "Write a short paragraph of plain prose in the present tense.\n"
        "Return the prompt only: no preamble, no quotes, no commentary."
    ),
}

# Which system prompt a generate mode gets. img2img is a partial denoise toward a
# described scene, so it reads as a description, not an instruction. animate is the
# odd one out: its image isn't a reference to describe, it's the frame the video
# starts from, so the brief is about motion rather than about the picture. control
# is the opposite of edit and gets its own brief for that reason: its attachment is
# a control map to be obeyed and never described, and an instruction-shaped prompt
# there returns a greyscale copy of the map (measured, not theorised).
_ENHANCE_MODE = {"txt2img": "create", "img2img": "create", "edit": "edit",
                 "compose": "compose", "control": "control", "animate": "animate"}

# What the control brief cannot work out for itself.
#
# The maps are never sent to the vision model — a control map is the one image the
# control brief explicitly forbids describing, so showing it invites the exact
# failure the brief exists to prevent. That leaves the model with no way to know
# how many people the pose holds, and a two-figure pose described as one person
# comes back with one person and a spare set of limbs. The studio counts them and
# says so in words instead.
_SUBJECT_BRIEF = (
    "The pose contains {n} people. Describe all {n} of them individually — each "
    "one's appearance and clothing — and keep them in the exact arrangement and "
    "relative position the pose specifies. Never describe fewer than {n}."
)
_CONTACT_BRIEF = (
    "They are in physical contact. Say so explicitly and describe how they touch "
    "(holding, leaning on, carrying, hands clasped) — the contact is the point of "
    "the picture, and a prompt that omits it produces two people standing apart."
)


def _subject_note(subjects: int, contact: bool) -> str:
    if subjects < 2:
        return ""
    note = "\n" + _SUBJECT_BRIEF.format(n=subjects)
    if contact:
        note += "\n" + _CONTACT_BRIEF
    return note


def enhance_system(template: str) -> str:
    """The brief that teaches a model to write for one generate mode.

    Public so agent mode can reuse them. The agent picks the mode *and* writes the
    prompt in a single turn, so it needs the same instruction the standalone
    enhancer gets — and there should be exactly one copy of it, or the two paths
    drift and the same request produces differently-styled prompts depending on
    which tab it went through.
    """
    return _ENHANCE_SYSTEM[template]


_PREAMBLE = re.compile(r"^\s*(here'?s|here is|sure[,!]?|prompt:)[^\n]*:\s*", re.I)


def _clean_prompt(text: str) -> str:
    """Strip the wrapping a chat model adds despite being told not to."""
    t = (text or "").strip()
    t = _PREAMBLE.sub("", t).strip()
    if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'":
        t = t[1:-1].strip()
    return t


def enhance_prompt(url, model, prompt, mode, images_b64=(), subjects=1, contact=False):
    """Rewrite a FLUX prompt with a vision model that can see the references.

    The identity-preserving work is done by the reference latents, not by this text
    — no description reproduces a face. What this buys is prompt adherence,
    composition and phrasing the text encoder actually responds to, so the user
    isn't rewriting the same prompt five times by hand.

    `subjects`/`contact` describe a pose built in the studio. They are appended to
    the control brief and nowhere else: no other mode conditions on a pose, so a
    figure count would be noise in them.

    Fails soft (returns "") exactly like `generate_title`; the caller falls back to
    the static template.
    """
    template = _ENHANCE_MODE.get(mode, "create")
    system = _ENHANCE_SYSTEM[template]
    if template == "control":
        system += _subject_note(subjects, contact)
    user = {"role": "user", "content": prompt}
    if images_b64:
        user["images"] = list(images_b64)
    messages = [{"role": "system", "content": system}, user]
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {"num_ctx": context_size_for(messages)},
    }
    try:
        response = requests.post(f"{url}/api/chat", json=payload, timeout=CHAT_TIMEOUT)
        if response.status_code != 200:
            return ""
        text = (response.json().get("message") or {}).get("content", "")
    except (requests.RequestException, ValueError):
        return ""
    return _clean_prompt(text)


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


# Agent mode drives tools instead of answering in prose, and below roughly this
# size a model stops choosing tools reliably: it describes the tool it would call,
# calls one with the arguments of another, or loops on the same call. Ollama draws
# the same line for its own agent features. Sized in billions of parameters.
AGENT_MIN_PARAMS = 7.0


def _parse_params(text) -> float:
    """Ollama's `parameter_size` ("9.7B", "700M") as a number of billions."""
    match = re.search(r"([\d.]+)\s*([BbMm])", str(text or ""))
    if not match:
        return 0.0
    value = float(match.group(1))
    return value / 1000 if match.group(2).lower() == "m" else value


def model_details(url):
    """Every installed model with its size and capabilities.

    One `/api/tags` request: modern Ollama reports `details` and `capabilities`
    inline there, so the per-model `/api/show` round trips this used to make
    (one per model, on every model-list refresh) are only needed as a fallback
    for older builds that omit `capabilities`.
    """
    try:
        response = requests.get(f"{url}/api/tags", timeout=5)
        if response.status_code != 200:
            return []
        models = response.json().get("models", [])
    except requests.RequestException:
        return []

    out = []
    for m in models:
        name = m.get("name")
        if not name:
            continue
        details = m.get("details") or {}
        capabilities = m.get("capabilities")
        if capabilities is None:
            # Old Ollama. `is_vision_model` has its own heuristic fallback, and
            # a build this old has no tool support to report anyway.
            capabilities = ["vision"] if is_vision_model(url, name) else []
        parameter_size = details.get("parameter_size") or ""
        out.append({
            "name": name,
            "parameter_size": parameter_size,
            "params_b": _parse_params(parameter_size),
            "capabilities": list(capabilities),
        })
    return out


def agent_eligible(detail) -> bool:
    """Whether a model can run agent mode.

    Three requirements, all load-bearing. `tools` because function calling needs a
    tool-aware chat template — without one the model answers in prose and never
    calls anything. `vision` because the agent is also the prompt enhancer, and it
    can't write "make his jacket red" into a specific instruction without seeing
    the jacket. And the size floor above.
    """
    caps = set(detail.get("capabilities") or [])
    return (
        detail.get("params_b", 0.0) >= AGENT_MIN_PARAMS
        and "tools" in caps
        and "vision" in caps
    )


def agent_shortfall(detail) -> str:
    """Why a model can't run agent mode, as a sentence. "" if it can."""
    caps = set(detail.get("capabilities") or [])
    missing = [c for c in ("tools", "vision") if c not in caps]
    if missing:
        return f"{detail.get('name')} has no {' or '.join(missing)} support."
    if detail.get("params_b", 0.0) < AGENT_MIN_PARAMS:
        return (
            f"{detail.get('name')} is {detail.get('parameter_size') or 'too small'}; "
            f"agent mode needs at least {AGENT_MIN_PARAMS:g}B parameters."
        )
    return ""


def list_agent_models(url):
    """Return sorted names of models that can run agent mode."""
    return sorted(d["name"] for d in model_details(url) if agent_eligible(d))


def list_vision_models(url):
    """Return sorted names of installed vision-capable models."""
    return sorted(
        d["name"] for d in model_details(url) if "vision" in d["capabilities"]
    )




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
