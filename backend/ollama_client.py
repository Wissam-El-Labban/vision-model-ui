"""Thin proxy helpers over the Ollama HTTP API.

The browser never talks to Ollama directly; the FastAPI backend owns every
Ollama call. Logic here is ported from the previous Streamlit app
(utils.py / app.py) and keeps the two load-bearing fixes from this branch:
`"think": False` and a (connect, read) timeout split. Thinking is opt-in: chat and
prompt enhancement take a `think` flag the user switches on in the UI, and every
other call keeps it off.
"""
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from contextlib import contextmanager
from urllib.parse import urlparse

import requests

from . import system

DEFAULT_URL = "http://localhost:11434"

# Connect quickly (fail fast if Ollama is down) but allow a generous read
# timeout so a cold model load on the first request doesn't look like a hang.
CHAT_TIMEOUT = (10, 300)

# Loading a model means reading its whole blob from disk (~45 s for a 9B model on
# this box's EBS volume), so keep it resident well past Ollama's 5 minute default.
KEEP_ALIVE = "30m"

# The smallest window `context_size_for` hands out. Every call uses at least this,
# because Ollama reloads the model whenever `num_ctx` changes between requests.
MIN_CTX = 8192

# Extra window for a request carrying retrieved document text (see rag.py), which
# marks its block with this tag. One fixed step rather than a size measured per
# message, so a chat that keeps using its documents keeps one `num_ctx` — and one
# loaded runner — from turn to turn.
DOCS_TAG = "<documents>"
DOCS_EXTRA_CTX = 4096

GITHUB_LATEST = "https://api.github.com/repos/ollama/ollama/releases/latest"


# --------------------------------------------------------------------------- #
# Activity — what the resource meter shows as loading / in use / unloading
# --------------------------------------------------------------------------- #
# Ollama's /api/ps only lists a model once its load has finished, so a 45 s disk
# read looks like nothing at all. What this process has asked for is the only
# other signal: a model with a request in flight that ps doesn't list yet is
# loading; one that ps does list is in use.
_activity_mu = threading.Lock()
_inflight: dict[str, int] = {}
_inflight_since: dict[str, float] = {}  # name -> when its first open request began
_unloading: dict[str, float] = {}   # name -> when the unload was asked for
_UNLOAD_MARK_S = 60                 # give up on a mark Ollama never acted on


def canonical(name):
    """`nomic-embed-text` as ps spells it: `nomic-embed-text:latest`."""
    return name if ":" in name.rsplit("/", 1)[-1] else f"{name}:latest"


@contextmanager
def using(model):
    """Mark `model` as having a request in flight for the duration."""
    key = canonical(model or "")
    with _activity_mu:
        _inflight[key] = _inflight.get(key, 0) + 1
        _inflight_since.setdefault(key, time.time())
    try:
        yield
    finally:
        with _activity_mu:
            _inflight[key] -= 1
            if not _inflight[key]:
                del _inflight[key]
                del _inflight_since[key]


def activity(loaded_names=()):
    """`{"inflight": names, "since": {name: epoch s}, "unloading": names}` right now.

    `since` is when a model's oldest open request began — for one ps doesn't list
    yet, when its load started.

    `loaded_names` is what ps lists; an unloading mark whose model is gone from
    it has done its job and is dropped, as is one older than `_UNLOAD_MARK_S`.
    """
    now, loaded = time.time(), set(loaded_names)
    with _activity_mu:
        for name, t in list(_unloading.items()):
            if name not in loaded or now - t > _UNLOAD_MARK_S:
                del _unloading[name]
        return {"inflight": set(_inflight), "since": dict(_inflight_since),
                "unloading": set(_unloading)}


_KIND_CACHE: dict[tuple, str] = {}


def model_kind(url, name):
    """"embedding", "vision" or "chat", from the model's capabilities.

    Cached per name; /api/tags is only read again when a name turns up that the
    cache doesn't know, so the meter's poll doesn't re-read it every time.
    """
    key = (url, canonical(name))
    if key not in _KIND_CACHE:
        for d in model_details(url):
            caps = set(d["capabilities"])
            _KIND_CACHE[(url, canonical(d["name"]))] = (
                "embedding" if "embedding" in caps else
                "vision" if "vision" in caps else "chat")
    return _KIND_CACHE.get(key, "chat")


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
    docs = any(DOCS_TAG in (m.get("content") or "") for m in messages)
    return min(32768, max(MIN_CTX, 4096 + 2048 * n_images) + (DOCS_EXTRA_CTX if docs else 0))


def supports_thinking(url, model):
    """Whether a model can reason before answering (Ollama's `thinking` capability).

    Asked before sending `think: True`, because Ollama rejects the whole request
    with a 400 for a model that can't think — and the toggle is global, so it is
    routinely on while a model without the capability is picked. Only a definite
    answer is cached; a failed lookup says False this time and asks again next.
    """
    key = (url, model)
    if key in _THINKING_CACHE:
        return _THINKING_CACHE[key]
    try:
        response = requests.post(f"{url}/api/show", json={"name": model}, timeout=10)
        if response.status_code != 200:
            return False
        caps = response.json().get("capabilities") or []
    except (requests.RequestException, ValueError):
        return False
    _THINKING_CACHE[key] = "thinking" in caps
    return _THINKING_CACHE[key]


_THINKING_CACHE = {}


def _lines(response):
    """A streamed response's lines, closing it however the reading ends.

    A caller that stops reading early — a cancelled job — closes the generator
    above this one, and the close lands here: dropping the connection is what tells
    Ollama to stop generating, rather than finish an answer nobody will read.
    """
    try:
        yield from response.iter_lines()
    finally:
        response.close()


def stream_chat(url, model, messages, think=False):
    with using(model):
        yield from _stream_chat(url, model, messages, think)


def _stream_chat(url, model, messages, think=False):
    """POST to Ollama's /api/chat and yield streaming events.

    Yields dicts: {"type": "token", "text": ...} for content,
    {"type": "thinking", "text": ...} for reasoning when `think` is on, and a
    final {"type": "usage", ...} carrying the exact token counts Ollama reports
    (prompt_eval_count + eval_count) plus the num_ctx we used — the ground
    truth for the context-usage meter.

    `think` is off unless the user asked for it, so reasoning-capable models
    answer directly by default; the caller checks `supports_thinking` first. The
    `num_ctx` is sized to fit the request's images (see context_size_for).
    """
    num_ctx = context_size_for(messages)
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "think": bool(think),
        "keep_alive": KEEP_ALIVE,
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

    for line in _lines(response):
        if not line:
            continue
        try:
            chunk = json.loads(line)
        except json.JSONDecodeError:
            continue
        message = chunk.get("message") or {}
        thinking = message.get("thinking")
        if thinking:
            yield {"type": "thinking", "text": thinking}
        content = message.get("content")
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


def stream_chat_tools(url, model, messages, tools, think=False):
    with using(model):
        yield from _stream_chat_tools(url, model, messages, tools, think)


def _stream_chat_tools(url, model, messages, tools, think=False):
    """Like `stream_chat`, but offers the model tools and reports what it called.

    Yields the same {"type": "token"} and {"type": "usage"} events, plus one
    {"type": "tool_calls", "calls": [{"name": ..., "arguments": {...}}, ...]} if
    the model asked for any. Tool calls arrive whole inside a chunk's message
    rather than a token at a time, but a model may emit them across several
    chunks, so they're accumulated and reported once at the end — the caller
    wants the whole plan before it starts unloading GPUs on the strength of it.

    Ollama returns `arguments` already parsed as an object; older builds send it
    as a JSON string, so both are accepted. `think` works as in `stream_chat`.
    """
    num_ctx = context_size_for(messages)
    payload = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "stream": True,
        "think": bool(think),
        "keep_alive": KEEP_ALIVE,
        "options": {"num_ctx": num_ctx},
    }
    response = requests.post(
        f"{url}/api/chat", json=payload, stream=True, timeout=CHAT_TIMEOUT
    )
    if response.status_code != 200:
        raise RuntimeError(f"Ollama returned {response.status_code}: {response.text}")

    calls = []
    for line in _lines(response):
        if not line:
            continue
        try:
            chunk = json.loads(line)
        except json.JSONDecodeError:
            continue
        message = chunk.get("message") or {}
        thinking = message.get("thinking")
        if thinking:
            yield {"type": "thinking", "text": thinking}
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


def ask_once(url, model, system, user, num_ctx=MIN_CTX):
    """One short text-only question, answered in full. "" on any failure.

    No images on purpose: this is for classification-shaped questions about what
    the user *asked for*, which the text answers on its own. Skipping the vision
    tokens is what keeps it cheap enough to run before the real turn. The context
    is the same as the turn that follows, so that turn reuses the loaded runner.
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "think": False,
        "keep_alive": KEEP_ALIVE,
        "options": {"num_ctx": num_ctx},
    }
    try:
        with using(model):
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
        "keep_alive": KEEP_ALIVE,
        "options": {"num_ctx": MIN_CTX},
    }
    try:
        with using(model):
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
        "If the image shows people, say that their face, identity and natural skin "
        "texture must stay unchanged.\n"
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


def enhance_prompt(url, model, prompt, mode, images_b64=(), subjects=1, contact=False,
                   think=False, reference=""):
    """Rewrite a FLUX prompt with a vision model that can see the references.

    The identity-preserving work is done by the reference latents, not by this text
    — no description reproduces a face. What this buys is prompt adherence,
    composition and phrasing the text encoder actually responds to, so the user
    isn't rewriting the same prompt five times by hand.

    `subjects`/`contact` describe a pose built in the studio. They are appended to
    the control brief and nowhere else: no other mode conditions on a pose, so a
    figure count would be noise in them.

    Returns `(prompt, thinking)`. Ollama keeps the reasoning out of `content`, so
    `thinking` is "" unless `think` is on, and the prompt is clean either way.

    `reference` is retrieved document text (`rag.format_context(..., "reference")`)
    put ahead of the prompt, for a character sheet or style guide to inform the
    rewrite. The brief's output rules are untouched, so the result is still a prompt
    the image model's text encoder can take.

    Fails soft (returns ("", "")) exactly like `generate_title`; the caller falls
    back to the static template.
    """
    template = _ENHANCE_MODE.get(mode, "create")
    system = _ENHANCE_SYSTEM[template]
    if template == "control":
        system += _subject_note(subjects, contact)
    content = f"{reference}\n\nPrompt to rewrite:\n{prompt}" if reference else prompt
    user = {"role": "user", "content": content}
    if images_b64:
        user["images"] = list(images_b64)
    messages = [{"role": "system", "content": system}, user]
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": bool(think),
        "keep_alive": KEEP_ALIVE,
        "options": {"num_ctx": context_size_for(messages)},
    }
    try:
        with using(model):
            response = requests.post(f"{url}/api/chat", json=payload, timeout=CHAT_TIMEOUT)
        if response.status_code != 200:
            return "", ""
        message = response.json().get("message") or {}
    except (requests.RequestException, ValueError):
        return "", ""
    return _clean_prompt(message.get("content") or ""), (message.get("thinking") or "").strip()


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

    Two requirements, both load-bearing. `tools` because function calling needs a
    tool-aware chat template — without one the model answers in prose and never
    calls anything. `vision` because the agent is also the prompt enhancer, and it
    can't write "make his jacket red" into a specific instruction without seeing
    the jacket.

    No size floor: small models pick tools less reliably, but whether that's good
    enough is the user's call to make, not ours.
    """
    caps = set(detail.get("capabilities") or [])
    return "tools" in caps and "vision" in caps


def agent_shortfall(detail) -> str:
    """Why a model can't run agent mode, as a sentence. "" if it can."""
    caps = set(detail.get("capabilities") or [])
    missing = [c for c in ("tools", "vision") if c not in caps]
    if missing:
        return f"{detail.get('name')} has no {' or '.join(missing)} support."
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


def unload(url, name):
    """Ask Ollama to unload `name` now, and mark it unloading for the meter."""
    with _activity_mu:
        _unloading[canonical(name)] = time.time()
    requests.post(
        f"{url}/api/generate",
        json={"model": name, "keep_alive": 0},
        timeout=10,
    )


def unload_all(url):
    """Unload every loaded model to free VRAM. Returns the names unloaded."""
    unloaded = []
    for model in running(url):
        name = model.get("name", "")
        if not name:
            continue
        unload(url, name)
        unloaded.append(name)
    return unloaded


def gpu_total_gb():
    """Total VRAM of the first GPU in GB, or None if it can't be read.

    Only meaningful when Ollama runs on this machine; callers treat None as
    "unknown" and fall back to unloading.
    """
    g = system.gpu()
    return g and g["total_gb"]


def loaded_vram_gb(url):
    """VRAM held by Ollama's loaded models, in GB."""
    return sum(m.get("size_vram") or 0 for m in running(url)) / 1e9


def _model_size_gb(url, name):
    """Size on disk of an installed model, in GB (None if not installed)."""
    try:
        models = requests.get(f"{url}/api/tags", timeout=5).json().get("models", [])
    except (requests.RequestException, ValueError):
        return None
    for m in models:
        if m.get("name") == name:
            return (m.get("size") or 0) / 1e9
    return None


def warm(url, model, reserved_gb):
    """Load `model` ahead of time so the next request doesn't pay the disk read.

    Skipped when it's already loaded, or when it wouldn't fit beside `reserved_gb`
    of VRAM held elsewhere (FLUX): a model that doesn't fit gets split onto the
    CPU, and a slow model is worse than a cold one. The 1.2 covers the KV cache
    and vision projector on top of the weights. Fails soft — this is only ever
    an optimisation.
    """
    try:
        loaded = running(url)
        if any(m.get("name") == model for m in loaded):
            return False
        size = _model_size_gb(url, model)
        total = gpu_total_gb()
        if size is None or total is None:
            return False
        held = sum(m.get("size_vram") or 0 for m in loaded) / 1e9
        fits = held + size * 1.2 + reserved_gb <= total
        logging.getLogger("uvicorn.error").info(
            "warm %s: ollama %.1f GB + model %.1f GB + flux %.1f GB vs %.1f GB -> %s",
            model, held, size * 1.2, reserved_gb, total, "warm" if fits else "skip")
        if not fits:
            return False
        with using(model):
            requests.post(
                f"{url}/api/chat",
                json={"model": model, "messages": [], "keep_alive": KEEP_ALIVE,
                      "options": {"num_ctx": MIN_CTX}},
                timeout=CHAT_TIMEOUT,
            )
        return True
    except requests.RequestException:
        return False


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
        "needs_password": bool(update_available and is_local(url) and needs_password()),
    }


INSTALL_SH = "https://ollama.com/install.sh"
MANUAL_UPGRADE = f"curl -fsSL {INSTALL_SH} | sh"
_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\(B")


def needs_password() -> bool:
    """Whether running the installer means asking for a sudo password."""
    if os.geteuid() == 0:
        return False
    try:
        return subprocess.run(["sudo", "-n", "true"], stdin=subprocess.DEVNULL,
                              capture_output=True, timeout=10).returncode != 0
    except (OSError, subprocess.TimeoutExpired):
        return True


def _relay_output(stream, emit) -> None:
    """Emit a process's output as lines, splitting on carriage returns too.

    The installer's download is a curl progress bar that redraws itself with "\r"
    and never ends a line until it's finished — read by line, the log looked frozen
    on "Downloading…" for the whole download. A "\r" segment is sent as a
    replacement for the line before it, and at most a few times a second.
    """
    buf = b""
    last_replace = 0.0
    while True:
        chunk = stream.read1(4096) if hasattr(stream, "read1") else stream.read(4096)
        if not chunk:
            break
        buf += chunk
        while True:
            cut = min((i for i in (buf.find(b"\n"), buf.find(b"\r")) if i >= 0),
                      default=-1)
            if cut < 0:
                break
            piece, sep, buf = buf[:cut], buf[cut:cut + 1], buf[cut + 1:]
            text = _ANSI.sub("", piece.decode("utf-8", "replace")).rstrip()
            if sep == b"\n":
                if text:
                    emit({"type": "line", "text": text})
            elif text and time.time() - last_replace >= 0.25:
                last_replace = time.time()
                emit({"type": "line", "text": text, "replace": True})
    text = _ANSI.sub("", buf.decode("utf-8", "replace")).rstrip()
    if text:
        emit({"type": "line", "text": text})


def upgrade(url, emit, password: str | None = None) -> None:
    """Run the official Ollama installer, emitting its output as it goes.

    Linux only, and the installer needs root: it replaces /usr/local/bin/ollama and
    restarts the systemd service. It calls plain `sudo`, which used to prompt for a
    password on the terminal the backend was started from — invisible to the page,
    so the upgrade sat on its last line forever. Now nothing can prompt on a tty
    (no stdin, its own session), and when a password is needed it comes from the
    page: a `sudo` on PATH that adds `-A`, and an askpass helper that answers with
    the password from this process tree's environment. It is checked before the
    installer starts, and never written to disk.
    """
    tmp = tempfile.mkdtemp(prefix="ollama-upgrade-")
    env = {**os.environ}
    try:
        if needs_password():
            if not password:
                raise RuntimeError("Your sudo password is needed to upgrade Ollama.")
            real_sudo = shutil.which("sudo")
            if not real_sudo:
                raise RuntimeError("sudo isn't installed, so the upgrade can't get root.")
            wrapper = os.path.join(tmp, "sudo")
            askpass = os.path.join(tmp, "askpass")
            with open(wrapper, "w") as f:
                f.write(f'#!/bin/sh\nexec {real_sudo} -A "$@"\n')
            with open(askpass, "w") as f:
                f.write('#!/bin/sh\nprintf \'%s\\n\' "$OLLAMA_UPGRADE_PW"\n')
            os.chmod(wrapper, 0o700)
            os.chmod(askpass, 0o700)
            env.update(PATH=f"{tmp}:{env.get('PATH', '')}", SUDO_ASKPASS=askpass,
                       OLLAMA_UPGRADE_PW=password)
            emit({"type": "line", "text": "Checking your sudo password…"})
            try:
                check = subprocess.run(
                    [real_sudo, "-A", "-v"], env=env, stdin=subprocess.DEVNULL,
                    capture_output=True, timeout=30, start_new_session=True)
            except subprocess.TimeoutExpired:
                raise RuntimeError("sudo didn't answer within 30 seconds.")
            if check.returncode != 0:
                raise RuntimeError("That password was rejected by sudo.")

        emit({"type": "line", "text": f"Downloading the installer from {INSTALL_SH}…"})
        script = os.path.join(tmp, "install.sh")
        try:
            response = requests.get(INSTALL_SH, timeout=30)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"Couldn't download the installer: {exc}")
        with open(script, "wb") as f:
            f.write(response.content)

        process = subprocess.Popen(
            ["sh", script], env=env, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        _relay_output(process.stdout, emit)
        code = process.wait()
        if code != 0:
            raise RuntimeError(
                f"The installer failed (exit {code}). To run it yourself: {MANUAL_UPGRADE}")

        # The installer restarts the service itself; wait for it to answer again.
        emit({"type": "line", "text": "Waiting for Ollama to come back…"})
        for _ in range(30):
            version = installed_version(url)
            if version:
                emit({"type": "line", "text": f"✓ Ollama {version} is running."})
                emit({"type": "upgraded", "version": version})
                return
            time.sleep(1)
        emit({"type": "line", "text":
              "✓ Installed, but Ollama isn't answering yet. Restart the Ollama "
              "service if it doesn't come back on its own."})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
