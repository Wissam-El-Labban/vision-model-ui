"""What this machine has, what is using it, and whether a pick of models fits.

Two different shortages, and the second is the one that actually bit: VRAM decides
whether the vision model and FLUX can stay loaded side by side (if not, the app just
swaps them — slower, but it works), while system RAM decides whether ComfyUI
survives a load at all. Every model is staged through RAM on its way to the GPU,
and on a box with less RAM than the weights the kernel's OOM killer ends the
sidecar mid-load — which looks, from the UI, like the engine restarting for no
reason and a "connection refused" in chat.

Estimates, not measurements: sizes come from the files on disk. Anything that
can't be read comes back as None, and `fit` then stays quiet rather than guessing.
"GB" throughout is GiB — what nvidia-smi and `free -g` show the user.
"""

from __future__ import annotations

import subprocess

# What the vision model needs beyond its weights — KV cache and the vision
# projector. The same factor `ollama_client.warm` sizes a load with.
OLLAMA_OVERHEAD = 1.2

# Headroom kept free on the GPU, matching main.py's `_VRAM_MARGIN_GB`.
VRAM_MARGIN_GB = 2.0

# RAM that isn't a model's weights: ComfyUI's own process, the backend, the OS.
RAM_OVERHEAD_GB = 3.0


def ram() -> dict | None:
    """System RAM and swap in GB, from /proc/meminfo; None off Linux."""
    try:
        with open("/proc/meminfo") as f:
            info = {k: int(v.split()[0]) for k, v in
                    (line.split(":", 1) for line in f if ":" in line)}
        total = info["MemTotal"] / 2**20  # kB here means KiB
        available = info["MemAvailable"] / 2**20
        return {
            "total_gb": round(total, 2),
            "available_gb": round(available, 2),
            "used_gb": round(total - available, 2),
            "swap_gb": round(info.get("SwapTotal", 0) / 2**20, 2),
        }
    except (OSError, KeyError, ValueError):
        return None


def gpu() -> dict | None:
    """Total and used VRAM of the first GPU in GB, or None without nvidia-smi."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout
        total, used = (int(x) / 1024 for x in out.splitlines()[0].split(","))
        return {"total_gb": round(total, 2), "used_gb": round(used, 2)}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def fit(flux_gb: float | None, ollama_gb: float | None) -> dict:
    """Whether FLUX weights of `flux_gb` and a vision model of `ollama_gb` suit this box.

    `level` is "ok", "amber" (works, but slower or close to the edge), "red" (likely
    to crash or crawl), or "unknown" when the GPU or a size couldn't be read. Each
    rule that trips adds one sentence to `messages`, worst first.
    """
    g, r = gpu(), ram()
    have = {"vram_gb": g and g["total_gb"], "ram_gb": r and r["total_gb"],
            "swap_gb": r and r["swap_gb"]}
    sizes = [s for s in (flux_gb, ollama_gb) if s]
    if not g or not r or not sizes:
        return {"level": "unknown", "messages": [], "need": {}, "have": have}

    ollama_need = (ollama_gb or 0) * OLLAMA_OVERHEAD
    flux_need = (flux_gb or 0) * 1.1
    memory = r["total_gb"] + r["swap_gb"]
    vram = g["total_gb"]
    red, amber = [], []

    if flux_need > vram:
        red.append(f"The image model needs about {flux_need:.0f} GB of VRAM but this GPU "
                   f"has {vram:.0f} GB — it will spill into system RAM and run very "
                   f"slowly, if at all. Pick a smaller image model.")
    if ollama_need > vram:
        red.append(f"The chat model needs about {ollama_need:.0f} GB of VRAM but this "
                   f"GPU has {vram:.0f} GB — it will run partly on the CPU and be slow. "
                   f"Pick a smaller chat model.")
    biggest = max(flux_gb or 0, ollama_gb or 0)
    if biggest + RAM_OVERHEAD_GB > memory:
        red.append(f"Loading the {biggest:.0f} GB model needs about "
                   f"{biggest + RAM_OVERHEAD_GB:.0f} GB of system RAM, and this machine "
                   f"has {memory:.0f} GB. The system will likely kill the engine "
                   f"mid-load (it restarts, and chat shows connection errors). Add RAM "
                   f"or pick a smaller model.")
    elif flux_gb and ollama_gb and flux_gb + ollama_need + RAM_OVERHEAD_GB > memory:
        amber.append(f"Both models together need about "
                     f"{flux_gb + ollama_need + RAM_OVERHEAD_GB:.0f} GB of system RAM "
                     f"and this machine has {memory:.0f} GB — switching between chat and "
                     f"image generation may run out of memory.")
    if (flux_gb and ollama_gb and not red
            and flux_need + ollama_need > vram - VRAM_MARGIN_GB):
        amber.append(f"The chat model ({ollama_need:.0f} GB) and image model "
                     f"({flux_need:.0f} GB) can't both stay on the {vram:.0f} GB GPU, so "
                     f"the chat model is unloaded for each image and reloaded after. "
                     f"That works, it's just slower.")

    return {
        "level": "red" if red else "amber" if amber else "ok",
        "messages": red + amber,
        "need": {"vram_gb": round(flux_need + ollama_need, 1),
                 "ram_gb": round(biggest + RAM_OVERHEAD_GB, 1)},
        "have": have,
    }
