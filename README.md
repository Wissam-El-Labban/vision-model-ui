# 👁️ Vision Model Chat

A polished local web UI for chatting with **vision models via Ollama**. Drop in any number of
images, ask questions, and stream the model's response — all from a single app.

Built with a **React (Vite + TypeScript)** frontend and a **FastAPI** backend that proxies Ollama.

## Features

- **Unified multi-image chat** — attach as many images as you want to your first question *or any
  follow-up*. Images are sent as separate images (not combined into one), so the model sees each one
  distinctly. This replaces the old Single/Dual/Triple tabs.
- **Streaming responses** — tokens appear live as the model generates them.
- **System prompt** — optional custom system prompt, with an optional persistent context image sent
  with every message.
- **Model management** — list installed vision models, download new ones (with live progress),
  remove, unload from VRAM, and view running models.
- **Image generation** — create, edit and combine images locally on FLUX. Models are installed
  from the app, not at setup; see below.
- **Agent mode** (**beta**) — say what you want in plain words and a local model picks the workflow,
  picks which images it applies to, and writes the prompt. See below.
- **In-app Ollama updates** — the UI tells you when a newer Ollama is available and lets you choose
  to upgrade with one click (local installs only). Updates are **opt-in**, never forced.
- **Per-image rotate** in the composer (client-side, via canvas).

## Quick Start

```bash
./run.sh
```

This creates the Python venv, installs backend deps, builds the frontend, ensures Ollama is installed
and running, and serves everything at **http://127.0.0.1:8000**.

> Requires Python 3.10+, Node.js 18+, and (on first run) internet access to install Ollama.

`run.sh` also installs the image-generation *engine* (a ComfyUI sidecar in its own venv) but
downloads **no image weights** — which model to run is your choice, and they are large. Skip the
engine entirely with `SKIP_FLUX=1 ./run.sh` if you only want chat.

## Image generation

Install a model from the sidebar's **🖼️ Image Models** panel. Nothing generates until you do.

| Model | Download | VRAM | Does | Notes |
| --- | --- | --- | --- | --- |
| **FLUX.2 [dev]** | ~100 GB | 46 GB | create + edit | Best photorealism. Gated. bf16 on disk, cast to fp8 at load. |
| **FLUX.2 [klein] 9B** | ~35 GB | 32 GB | create + edit | Distilled 9B, runs in bf16 with no quantization, 8 steps. Gated. |
| **Qwen-Image 2512** — fp8 | ~30 GB | 28 GB | create | The best there is at legible text inside an image, English and Chinese. Ungated. |
| **Qwen-Image** — fp8 | ~30 GB | 28 GB | create | The original; superseded by 2512. Shares its encoder and VAE, so ~20 GB if 2512 is installed. |
| **Qwen-Image Edit 2511** — fp8 | ~30 GB | 28 GB | edit + combine | Instruction editing on Qwen, up to three reference images at once. The newest edit release. Ungated. |
| **Qwen-Image Edit** — fp8 | ~30 GB | 28 GB | edit + combine | The original edit model; one reference image, weaker on multi-step instructions. |
| **Wan 2.2 I2V A14B** — fp16 | ~69 GB | 80 GB | animate | One image → a 5-second 720p video. Ungated. |

Qwen splits create and edit across separate checkpoints the way FLUX.1 split dev and Kontext, so
covering both jobs means installing two — but all four Qwen bundles share one text encoder and one
VAE, so the second costs only its transformer (~20 GB). There is no "Qwen-Image Edit 2512": 2512 was
a text-to-image refresh, and 2511 is the current edit model.

The list is in quality order, and the first installed model that can serve a mode is what that mode
runs on by default — so installing Qwen does not displace FLUX.2 for Create or Edit; pick it in the
model picker.

Downloads resume if interrupted, and the weights are used entirely offline afterwards. The panel
checks free disk space before starting and refuses rather than filling the disk mid-download.

A **HuggingFace token** is only needed for the gated models (Black Forest Labs' own repos): paste
one into the same panel — it is validated on save, stored `0600` on the server, and never sent back
to the browser. `HF_TOKEN` in the environment works too.

You can also add any single-file FLUX.1 transformer from a HuggingFace repo (`owner/model`, or
`owner/model:file.safetensors`) from that panel. Those extras run on FLUX.1's text encoder, so the
FLUX.1 model has to be installed alongside them.

## Agent mode (beta)

> **Beta.** This one is not finished. The agent picks the workflow, the images and the prompt for
> you, and it gets those calls wrong often enough to notice — and when it does, the result still
> looks like a normal answer rather than an error. Use **🎨 Generate** when you already know what
> you want; Agent is for when you'd rather describe it and see.

The **🤖 Agent** tab is the Generate tab without the homework. Instead of choosing Create / Edit /
Combine yourself and arranging the attachments so the right one lands first, you describe what you
want and a local Ollama model works it out — which workflow to run, which images it applies to, and
the prompt to run it with.

**It analyzes too.** Ask a question instead — *"what's different about these two?"*, *"how many
people are in it?"*, *"does this look sharp to you?"* — and it answers in words, the way the Analyze
tab does, without generating anything. Whether a message is a request for a picture or a question
about one is decided before the turn runs, and on a question the model is never shown a tool at
all, so a question can't cost you a generation you didn't ask for.

Images come in the same way they do in **Analyze**: everything pinned in the left panel and
everything attached anywhere in the conversation is visible to the agent, numbered, on every turn.
That is what makes follow-ups work — *"now combine that one with the first photo"* refers to
pictures from three messages ago.

The chosen model is also the prompt enhancer. It writes the final FLUX prompt itself, to the same
per-mode briefs the standalone enhancer uses, so there is no second rewriting pass — and it is
shown, under the step in the chat, so you can see exactly what generated the image.

**Requirements.** A model qualifies only if Ollama reports it with **tool support**, **vision**, and
**at least 7B parameters**. All three matter: without tools it answers in prose and never calls
anything, without vision it can't write *"make his jacket red"* into a specific instruction, and
below ~7B it picks the wrong tool often enough to be worse than choosing by hand. With no eligible
model installed the tab is disabled and says so.

**Tools.** The 🧰 menu in the composer controls what the agent may reach for — Create, Edit and
Combine on by default, Animate off (a video run is slow), Control listed but unavailable for now.
A tool that is switched off is never shown to the model at all, so it cannot pick it. Neither can
it pick one the conversation can't satisfy: Combine isn't offered until two images are in view.

It plans once per turn, then runs everything it decided on. The GPU can't hold Ollama and FLUX at
the same time, so a reason-act loop would pay a full model swap per step — minutes, for a second
opinion the model can't form anyway, since it never sees the finished image.

## Development

Run the backend and the Vite dev server separately for hot-reload:

```bash
# Terminal 1 — backend (auto-reload)
./venv/bin/uvicorn backend.main:app --reload --port 8000

# Terminal 2 — frontend (proxies /api to :8000)
cd frontend && npm install && npm run dev
```

Then open the Vite URL (http://localhost:5173).

## Architecture

```
vision-model-ui/
├── backend/
│   ├── main.py            # FastAPI routes + serves built frontend
│   ├── ollama_client.py   # all Ollama HTTP calls (chat, models, version, upgrade)
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── App.tsx            # state + chat orchestration
│       ├── api.ts             # fetch + streaming client
│       └── components/        # Sidebar, Chat, Composer, ModelManager, UpdateBanner
└── run.sh                 # build + serve
```

- The browser never talks to Ollama directly — every call goes through the FastAPI backend.
- Chat, model-pull progress, and the Ollama upgrade all **stream** to the client.
- Chat requests send only `model` + `messages` with `think: false`, and use a `(10s connect, 300s
  read)` timeout (fast failure when Ollama is down, headroom for cold model loads).

## API endpoints

| Method | Path                     | Purpose                              |
| ------ | ------------------------ | ------------------------------------ |
| GET    | `/api/models`            | List vision (and all) models         |
| POST   | `/api/chat`              | Stream a chat response               |
| POST   | `/api/models/pull`       | Download a model (streams progress)  |
| DELETE | `/api/models/{name}`     | Remove a model                       |
| POST   | `/api/models/unload`     | Unload all models from VRAM          |
| GET    | `/api/ps`                | Running models                       |
| GET    | `/api/ollama/version`    | Installed vs latest + update flag    |
| POST   | `/api/ollama/upgrade`    | Run the Ollama installer (streams)   |

## Troubleshooting

- **Can't reach Ollama** — make sure it's running (`ollama serve`). The chat fails fast (~10s) with a
  clear message if it's down.
- **No vision models listed** — pull one (e.g. `qwen2.5-vl:7b`, `llava:latest`) via the sidebar or
  `ollama pull`.
- **Upgrade button missing** — it only appears when Ollama is local and a newer version exists. For a
  remote Ollama, upgrade it on its host.
- **Upgrade fails** — the installer may need `sudo`; the streamed log shows the error and the manual
  command (`curl -fsSL https://ollama.com/install.sh | sh`).
