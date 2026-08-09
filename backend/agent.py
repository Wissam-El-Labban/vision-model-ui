"""Agent mode: the tools a local model may call, and the brief that teaches it to.

Generate mode asks the user to know things: which of five tabs does what, how many
images each one wants, and which position in the attachment list means "the thing
being edited" rather than "a reference". Agent mode moves that knowledge into the
model — it picks the workflow, picks the images, and writes the prompt.

Two design points worth stating up front, because everything here follows from them.

**Images are addressed by number.** The conversation carries every image the chat has
seen (pinned first, then in-chat attachments, deduped) and the last user message
carries a manifest numbering them 1..N — the same manifest analyze mode already
builds so a vision model can tell "this image" from "the initial one". Tool arguments
reference those numbers, and the caller maps number N back to a stored image. Without
it the model has no vocabulary for *which* picture it means, which is most of what a
multi-image request is about.

**The agent is also the prompt enhancer.** The text it passes as a tool argument is
what goes to FLUX; there is no second rewriting pass. So the system prompt carries the
same per-mode briefs `/api/flux/enhance` uses, pulled from `ollama_client` rather than
restated (see `oc.enhance_system`), and only the briefs for tools that are actually
offered — a model told how to write an animate prompt when animate is switched off has
been given one more way to pick wrong.
"""
from __future__ import annotations

from . import ollama_client as oc

# Aspect ratios, not raw pixels. FLUX is trained at ~1 megapixel and drifts badly
# away from it, and a model handed a free `width` will eventually ask for 4096 —
# which on FLUX.2 is a several-minute run that produces a worse image than the
# square would have. Three named shapes cover what anyone actually wants, and the
# request's own width/height stay the default when the model says nothing.
ASPECTS = {
    "square": (1024, 1024),
    "portrait": (832, 1216),
    "landscape": (1216, 832),
}

# Wan I2V is trained for short clips and the backend caps it anyway; stated here so
# the model doesn't ask for a thirty-second film.
MAX_SECONDS = 5.0

_IMAGE_ARG = "The number of an image from the manifest in the user's message."

TOOLS: dict[str, dict] = {
    "create_image": {
        "label": "Create image",
        "icon": "🖼️",
        "hint": "Make a new image from a description, optionally starting from one image.",
        "available": True,
        "default": True,
        "brief": "create",
        # txt2img, or img2img when the model names a source image.
        "mode": "txt2img",
        "schema": {
            "name": "create_image",
            "description": (
                "Generate a brand new image from a text description. Use this when the "
                "user wants something made rather than an existing image changed. "
                "Optionally start from one existing image, which the result will "
                "loosely follow in colour and composition."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": (
                            "A vivid paragraph describing the finished photograph, "
                            "written to the CREATE brief in the system prompt."
                        ),
                    },
                    "aspect": {
                        "type": "string",
                        "enum": sorted(ASPECTS),
                        "description": "Shape of the image. Omit for the user's default.",
                    },
                    "source_image": {
                        "type": "integer",
                        "description": (
                            _IMAGE_ARG + " Give this ONLY when the new image should be "
                            "a variation of that one. To change an existing image, use "
                            "edit_image instead."
                        ),
                    },
                },
                "required": ["prompt"],
            },
        },
    },
    "edit_image": {
        "label": "Edit image",
        "icon": "✏️",
        "hint": "Change one existing image while keeping the rest of it intact.",
        "available": True,
        "default": True,
        "brief": "edit",
        "mode": "edit",
        "min_images": 1,
        "schema": {
            "name": "edit_image",
            "description": (
                "Change one existing image, keeping everything the user didn't ask to "
                "change. Use this for 'make the jacket red', 'remove the car', 'give "
                "her sunglasses'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "instruction": {
                        "type": "string",
                        "description": (
                            "An imperative instruction describing the change, written "
                            "to the EDIT brief in the system prompt. Never a "
                            "description of a scene."
                        ),
                    },
                    "image": {
                        "type": "integer",
                        "description": _IMAGE_ARG + " This is the image being changed.",
                    },
                    "reference_images": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": (
                            "Other image numbers the instruction draws from, e.g. the "
                            "photo of the person being added. Usually empty."
                        ),
                    },
                },
                "required": ["instruction", "image"],
            },
        },
    },
    "combine_images": {
        "label": "Combine images",
        "icon": "🧩",
        "hint": "Fuse two or more images into one new image.",
        "available": True,
        "default": True,
        "brief": "compose",
        "mode": "compose",
        "min_images": 2,
        "schema": {
            "name": "combine_images",
            "description": (
                "Build one new image out of two or more existing ones — put these two "
                "people in the same photo, dress this person in that outfit, place "
                "this object in that room."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": (
                            "A vivid paragraph describing the combined result, written "
                            "to the COMBINE brief in the system prompt."
                        ),
                    },
                    "images": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": (
                            "Two or more image numbers to combine, in the order the "
                            "prompt refers to them."
                        ),
                    },
                },
                "required": ["prompt", "images"],
            },
        },
    },
    "animate_image": {
        "label": "Animate image",
        "icon": "🎬",
        "hint": "Turn one image into a short video. Slow — off by default.",
        "available": True,
        "default": False,
        "brief": "animate",
        "mode": "animate",
        "min_images": 1,
        "schema": {
            "name": "animate_image",
            "description": (
                "Bring one image to life as a short video clip. Only when the user asks "
                "for motion, video or animation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "motion": {
                        "type": "string",
                        "description": (
                            "What happens over the next few seconds and how the camera "
                            "moves, written to the ANIMATE brief in the system prompt. "
                            "Never a description of the scene."
                        ),
                    },
                    "image": {
                        "type": "integer",
                        "description": _IMAGE_ARG + " This becomes the first frame.",
                    },
                    "seconds": {
                        "type": "number",
                        "description": f"Clip length, up to {MAX_SECONDS:g} seconds.",
                    },
                },
                "required": ["motion", "image"],
            },
        },
    },
    # Listed so the Tools menu can show it greyed out rather than leaving the user
    # wondering why the mode they can reach from the Generate tab is missing here.
    # Control needs a preprocessor choice (depth / canny / pose), a structure lock
    # and often a studio-built pose — decisions the agent has no way to make well
    # yet, and a control run set up wrong returns a greyscale copy of its own map.
    "control_image": {
        "label": "Control (pose / layout)",
        "icon": "🕹️",
        "hint": "Not available to the agent yet — use the Generate tab's Control workflow.",
        "available": False,
        "default": False,
        "mode": "control",
    },
}

#: Tools the UI enables when the user has never touched the Tools menu.
DEFAULT_TOOLS = [n for n, t in TOOLS.items() if t.get("default")]


def catalog() -> list[dict]:
    """The Tools menu, unavailable entries included."""
    return [
        {
            "id": name,
            "label": tool["label"],
            "icon": tool["icon"],
            "hint": tool["hint"],
            "available": tool["available"],
            "default": tool.get("default", False),
        }
        for name, tool in TOOLS.items()
    ]


def _offered(enabled, n_images: int) -> list[str]:
    """Tool names that are switched on, available, and runnable right now.

    The image floor is applied here rather than left to validation on purpose: a
    tool the conversation can't satisfy is never shown to the model, so it can't
    pick it. Offering `combine_images` with one image and rejecting the call
    afterwards wastes a whole planning turn to arrive at an error.
    """
    wanted = set(enabled)
    return [
        name
        for name, tool in TOOLS.items()
        if name in wanted
        and tool["available"]
        and n_images >= tool.get("min_images", 0)
    ]


def tool_schemas(enabled, n_images: int) -> list[dict]:
    return [
        {"type": "function", "function": TOOLS[name]["schema"]}
        for name in _offered(enabled, n_images)
    ]


_ROUTER = (
    "You are the image assistant in a local chat app. You do two things, and they "
    "matter equally: you ANSWER QUESTIONS about the images in the conversation, and "
    "you MAKE OR CHANGE images by calling a tool.\n"
    "\n"
    "Answering is the default. Call a tool only when the user asked for an image to "
    "be produced or altered — 'make', 'draw', 'edit', 'combine', 'turn this into'. "
    "If they did not ask for that, reply in words and call nothing.\n"
    "\n"
    "These are questions. Answer them in text, with no tool call:\n"
    "  describing what is in an image, or in several\n"
    "  comparing images, or saying how they differ\n"
    "  counting images or things inside them\n"
    "  reading text, signs or numbers in an image\n"
    "  judging quality, sharpness, lighting, composition\n"
    "  anything phrased as who / what / where / when / why / how many / which\n"
    "Answer those fully and concretely, from what you can actually see. Look at every "
    "image before you answer, and describe only what is there — never guess at an "
    "image you cannot see, and never say an image is still being generated.\n"
    "\n"
    "Images being attached is not a request to generate. A question about a picture "
    "is still a question.\n"
    "\n"
    "When the request IS to make or change an image: usually call exactly one tool, "
    "several only when the user clearly asked for several images. Then your reply is "
    "at most one short sentence saying what you are making — often nothing at all. "
    "Never repeat the prompt in it: the user sees the tool's prompt and the finished "
    "image already, so restating them shows the same paragraph twice.\n"
    "\n"
    "Never do both for one request. Answering and generating are alternatives — a "
    "description of an image the user asked about is a complete reply on its own, and "
    "generating on top of it produces a picture nobody asked for.\n"
    "\n"
    "If a request is too vague to act on, ask which image or what change they mean. "
    "Asking is better than guessing."
)

#: For the fallback turn when the planning turn produced nothing at all — no text
#: and no tool call. Same job, none of the tool vocabulary that caused it.
_ANSWER_ONLY = (
    "You are a vision assistant in a local chat app. Answer the user's question about "
    "the images in the conversation, fully and concretely, from what you can actually "
    "see in them. Describe only what is there. If several images are attached, they "
    "are numbered in the user's message — refer to them by those numbers."
)


def answer_system() -> str:
    return _ANSWER_ONLY


# --------------------------------------------------------------------------- #
# Intent
# --------------------------------------------------------------------------- #
# Deciding whether to generate is settled *before* the real turn, on the text
# alone, and the tools are only put in front of the model if the answer is yes.
#
# This is a structural fix for a measured failure, not belt-and-braces. Asked to
# route and to write a prompt in one turn, a 9B model handed four tool schemas and
# the prompt-writing briefs would answer a plain question ("what's the difference
# between them?") *and* call create_image alongside the answer — no instruction to
# the contrary reliably stopped it, because the tools were still sitting there. A
# spurious call is the expensive kind of wrong: it unloads Ollama and spends
# minutes of GPU producing an image nobody asked for. With no tools passed, that
# outcome is unreachable rather than merely discouraged, and the analysis turn
# also gets a system prompt about looking at images instead of one about writing
# prompts.
#
# Text-only and 2k of context, so it costs a round trip rather than a re-encode of
# every image in the conversation.
_INTENT_SYSTEM = (
    "Decide what the user's message asks for. Reply with ONE word.\n"
    "\n"
    "GENERATE — they want an image produced or altered:\n"
    "  making, drawing, creating, rendering something new\n"
    "  editing, changing, fixing, adding, removing, recolouring\n"
    "  combining or merging pictures, or putting subjects together in one\n"
    "  animating, or turning a picture into a video\n"
    "  a bare description of a picture, with no question in it — naming a scene is "
    "how people ask for one\n"
    "\n"
    "ANSWER — anything else:\n"
    "  questions about an image: what, which, where, how many, why\n"
    "  describing, comparing, counting, reading text in it, judging quality\n"
    "  ordinary conversation\n"
    "\n"
    "Examples:\n"
    "  a lighthouse at dusk -> GENERATE\n"
    "  give her sunglasses -> GENERATE\n"
    "  merge these into one picture -> GENERATE\n"
    "  make it look like winter -> GENERATE\n"
    "  bring this photo to life -> GENERATE\n"
    "  what breed is this dog? -> ANSWER\n"
    "  compare the two photos -> ANSWER\n"
    "  is this in focus? -> ANSWER\n"
    "  how many people are in it? -> ANSWER\n"
    "  describe what you see -> ANSWER\n"
    "\n"
    "Reply with GENERATE or ANSWER and nothing else."
)


def wants_image(url: str, model: str, prompt: str) -> bool:
    """Whether this request asks for an image to be produced or altered.

    Fails open: an unreachable or confused classifier returns True, so the turn
    proceeds exactly as it would without this step — the router brief still says
    to answer questions in words, and a false yes costs a tool the model can
    decline, while a false no would make generating impossible.
    """
    verdict = oc.ask_once(url, model, _INTENT_SYSTEM, prompt).upper()
    return "ANSWER" not in verdict

_MANIFEST_NOTE = (
    "The user's message lists the {n} images in this conversation, numbered 1 to {n}. "
    "Every image argument you pass is one of those numbers. Pick the number the user "
    "means — 'the first one', 'the pinned photo', 'that one' all refer to entries in "
    "that list."
)

_NO_IMAGES = (
    "There are no images in this conversation yet, so only tools that need none are "
    "available. If the user is asking to change an image, tell them to attach it."
)

_BRIEF_HEADER = (
    "--- HOW TO WRITE A TOOL ARGUMENT ---\n"
    "Everything below applies ONLY when you have decided to call a tool. It is not "
    "guidance for replying to the user, and it is not a reason to call one: if the "
    "request was a question, ignore all of it and answer in words.\n"
    "The text you pass as a tool argument is the final prompt — nothing rewrites it "
    "afterwards. Write it to the brief for the tool you are calling. Where a brief "
    "says to return only the prompt, that means the argument holds only the prompt."
)

#: Which brief heads which tool, for the labelled sections below.
_BRIEF_TITLE = {
    "create": "CREATE brief — for create_image's `prompt`",
    "edit": "EDIT brief — for edit_image's `instruction`",
    "compose": "COMBINE brief — for combine_images's `prompt`",
    "animate": "ANIMATE brief — for animate_image's `motion`",
}


def system_prompt(enabled, n_images: int) -> str:
    """Router brief + the writing brief for each tool actually on offer."""
    offered = _offered(enabled, n_images)
    parts = [_ROUTER]
    parts.append(_MANIFEST_NOTE.format(n=n_images) if n_images else _NO_IMAGES)

    briefs = []
    seen = set()
    for name in offered:
        brief = TOOLS[name].get("brief")
        if not brief or brief in seen:
            continue
        seen.add(brief)
        briefs.append(f"--- {_BRIEF_TITLE[brief]} ---\n{oc.enhance_system(brief)}")
    if briefs:
        parts.append(_BRIEF_HEADER)
        parts.extend(briefs)
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def _index(value, n_images: int, what: str) -> int:
    """One image number -> a 0-based offset into the manifest."""
    try:
        i = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{what} isn't an image number.")
    if not 1 <= i <= n_images:
        raise ValueError(
            f"{what} is image {i}, but this conversation has "
            + (f"{n_images} image{'s' if n_images != 1 else ''}." if n_images
               else "no images.")
        )
    return i - 1


def _indices(values, n_images: int, what: str) -> list[int]:
    if values is None:
        return []
    if not isinstance(values, (list, tuple)):
        values = [values]
    out = []
    for v in values:
        i = _index(v, n_images, what)
        if i not in out:  # a model that lists the same image twice means it once
            out.append(i)
    return out


def _text(args, key: str, what: str) -> str:
    value = (args.get(key) or "").strip() if isinstance(args.get(key), str) else ""
    if not value:
        raise ValueError(f"The agent called {what} without a prompt.")
    return value


def validate(name: str, args: dict, n_images: int) -> dict:
    """Turn one tool call into a job spec, or raise with a sentence to show the user.

    The spec speaks in 0-based offsets into the manifest (`init`, `refs`); the
    caller resolves those to stored images. Raising rather than silently repairing
    is deliberate: a call the agent got wrong should be visible as a failed step,
    not quietly turned into a different image than anyone asked for.
    """
    tool = TOOLS.get(name)
    if tool is None or not tool["available"]:
        raise ValueError(f"{name} isn't a tool this app can run.")
    args = args if isinstance(args, dict) else {}

    if name == "create_image":
        width = height = None
        aspect = args.get("aspect")
        if isinstance(aspect, str) and aspect.lower() in ASPECTS:
            width, height = ASPECTS[aspect.lower()]
        init = None
        if args.get("source_image") is not None:
            init = _index(args["source_image"], n_images, "The source image")
        return {
            "mode": "img2img" if init is not None else "txt2img",
            "prompt": _text(args, "prompt", "create_image"),
            "init": init,
            "refs": [],
            "width": width,
            "height": height,
        }

    if name == "edit_image":
        init = _index(args.get("image"), n_images, "The image to edit")
        refs = [
            i for i in _indices(args.get("reference_images"), n_images,
                                "A reference image")
            if i != init
        ]
        return {
            "mode": "edit",
            "prompt": _text(args, "instruction", "edit_image"),
            "init": init,
            "refs": refs,
        }

    if name == "combine_images":
        refs = _indices(args.get("images"), n_images, "An image to combine")
        if len(refs) < 2:
            raise ValueError(
                "Combining needs at least two different images, and the agent named "
                + ("one." if len(refs) == 1 else "none.")
            )
        return {
            "mode": "compose",
            "prompt": _text(args, "prompt", "combine_images"),
            "init": None,
            "refs": refs,
        }

    if name == "animate_image":
        seconds = args.get("seconds")
        try:
            seconds = min(float(seconds), MAX_SECONDS) if seconds is not None else None
        except (TypeError, ValueError):
            seconds = None
        return {
            "mode": "animate",
            "prompt": _text(args, "motion", "animate_image"),
            "init": _index(args.get("image"), n_images, "The image to animate"),
            "refs": [],
            "seconds": seconds,
        }

    raise ValueError(f"{name} isn't a tool this app can run.")
