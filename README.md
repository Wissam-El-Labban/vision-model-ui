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
