#!/bin/bash

# Vision Model Chat - Startup Script
# Sets up the Python backend + React frontend, ensures Ollama is running,
# then serves the app from a single FastAPI server.

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

VENV_DIR="venv"
REQUIREMENTS_FILE="backend/requirements.txt"
HOST="127.0.0.1"
PORT="8000"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Vision Model Chat Interface${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# --- Python backend deps ---------------------------------------------------- #
PYTHON="${PYTHON:-python3}"

if ! command -v "$PYTHON" &> /dev/null; then
    echo -e "${RED}✗ '$PYTHON' not found. Install Python 3.9+ (or set PYTHON=/path/to/python).${NC}"
    exit 1
fi

# git + curl are required to fetch ComfyUI and install Ollama. Vanilla
# Debian/Ubuntu images often ship without them, so install them up front (before
# the FLUX section, which may be skipped, and Ollama, which also needs curl).
for _tool in git curl; do
    if ! command -v "$_tool" &> /dev/null; then
        echo -e "${YELLOW}$_tool not found. Installing...${NC}"
        if command -v apt-get &> /dev/null; then
            sudo apt-get update -qq && sudo apt-get install -y "$_tool"
        fi
        if ! command -v "$_tool" &> /dev/null; then
            echo -e "${RED}✗ '$_tool' is required but couldn't be installed. Install it and re-run.${NC}"
            exit 1
        fi
    fi
done

# Vanilla Debian/Ubuntu ship python3 without ensurepip, so `python3 -m venv`
# fails ("ensurepip is not available"). pyenv-built Pythons bundle it, so this
# check passes untouched on pyenv machines and only the vanilla path installs
# the missing OS package.
ensure_venv_support() {
    "$PYTHON" -m ensurepip --version &> /dev/null && "$PYTHON" -c "import venv" &> /dev/null
}

if ! ensure_venv_support; then
    PYVER="$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    PKG="python${PYVER}-venv"
    echo -e "${YELLOW}Python venv support (ensurepip) is missing for $PYTHON.${NC}"
    if command -v apt-get &> /dev/null; then
        echo -e "${YELLOW}Installing $PKG (may prompt for sudo)...${NC}"
        sudo apt-get update -qq && sudo apt-get install -y "$PKG" python3-pip
    fi
    if ! ensure_venv_support; then
        echo -e "${RED}✗ Could not enable venv support. Install it manually, e.g.:${NC}"
        echo -e "${RED}    sudo apt install $PKG python3-pip${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ venv support enabled${NC}"
fi

# A venv left over from a failed run can exist as a directory without a working
# activate script — check for the activate script, not just the directory, and
# rebuild it if it's incomplete.
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    if [ -d "$VENV_DIR" ]; then
        echo -e "${YELLOW}Removing incomplete virtual environment...${NC}"
        rm -rf "$VENV_DIR"
    fi
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    "$PYTHON" -m venv "$VENV_DIR"
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi
source "$VENV_DIR/bin/activate"
echo -e "${BLUE}Installing backend dependencies...${NC}"
pip install --upgrade pip --quiet
pip install -r "$REQUIREMENTS_FILE" --quiet
echo -e "${GREEN}✓ Backend dependencies ready${NC}"

# --- Frontend build --------------------------------------------------------- #
echo ""
echo -e "${BLUE}Building frontend...${NC}"
# Vanilla machines often have no Node.js. Vite 5 needs Node 18+, so install the
# Node 20 LTS via NodeSource (falls back to distro apt package if that fails).
if ! command -v npm &> /dev/null; then
    echo -e "${YELLOW}Node.js/npm not found. Installing Node 20 LTS...${NC}"
    if command -v apt-get &> /dev/null; then
        if curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash - ; then
            sudo apt-get install -y nodejs
        else
            echo -e "${YELLOW}NodeSource unavailable — falling back to distro nodejs/npm...${NC}"
            sudo apt-get update -qq && sudo apt-get install -y nodejs npm
        fi
    fi
    if ! command -v npm &> /dev/null; then
        echo -e "${RED}✗ npm still not found. Install Node.js 18+ manually to build the UI.${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Node.js $(node --version) installed${NC}"
fi
pushd frontend > /dev/null
if [ ! -d node_modules ]; then
    npm install
fi
npm run build
popd > /dev/null
echo -e "${GREEN}✓ Frontend built${NC}"

# --- Image generation: the FLUX engine (ComfyUI sidecar) -------------------- #
# ALL image generation runs here — the backend venv above holds no torch at all.
# This installs the *engine* only. No weights: which model to run is the user's
# choice (a 48 GB card wants FLUX.2, a 24 GB one wants quantized FLUX.1), and it's
# 30-50 GB either way, so models are installed from the app's Models panel instead
# of behind a startup script. See backend/flux_catalog.py.
#
# Idempotent and local. Chat-only users can skip the engine too: SKIP_FLUX=1 ./run.sh
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Setting up the image engine...${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

FLUX_DIR="flux_runtime"
COMFY_DIR="$FLUX_DIR/ComfyUI"
CVENV="$FLUX_DIR/cvenv"
# Pinned to the exact commits this workflow was validated against.
COMFY_COMMIT="51bf508a0b1bde9416a0c221b0f33f8325305229"
GGUF_COMMIT="6ea2651e7df66d7585f6ffee804b20e92fb38b8a"

# Mirror flux_client.runtime_ready(): the venv and ComfyUI, not the weights.
flux_installed() {
    [ -x "$CVENV/bin/python" ] || return 1
    [ -f "$COMFY_DIR/main.py" ] || return 1
    return 0
}

# Report what's installed, and tell a fresh machine where to get a model.
models_hint() {
    local unets
    unets="$(ls "$COMFY_DIR/models/unet"/*.gguf "$COMFY_DIR/models/unet"/*.safetensors 2>/dev/null | wc -l)"
    if [ "$unets" -gt 0 ]; then
        echo -e "${GREEN}✓ $unets image model file(s) installed${NC}"
    else
        echo -e "${YELLOW}No image model installed yet — open the app and install one in the${NC}"
        echo -e "${YELLOW}Models panel (FLUX.2 [dev] recommended; needs ~50 GB and a 48 GB GPU).${NC}"
    fi
}

if [ "${SKIP_FLUX:-0}" = "1" ]; then
    echo -e "${YELLOW}SKIP_FLUX=1 — skipping the image engine (image generation will be unavailable).${NC}"
elif flux_installed; then
    echo -e "${GREEN}✓ Image engine already installed${NC}"
    models_hint
else
    # 1. ComfyUI + the GGUF loader node, pinned to known-good commits.
    if [ ! -f "$COMFY_DIR/main.py" ]; then
        echo -e "${YELLOW}Cloning ComfyUI...${NC}"
        git clone --quiet https://github.com/comfyanonymous/ComfyUI "$COMFY_DIR"
        git -C "$COMFY_DIR" checkout --quiet "$COMFY_COMMIT"
    fi
    if [ ! -d "$COMFY_DIR/custom_nodes/ComfyUI-GGUF" ]; then
        echo -e "${YELLOW}Cloning ComfyUI-GGUF loader node...${NC}"
        git clone --quiet https://github.com/city96/ComfyUI-GGUF \
            "$COMFY_DIR/custom_nodes/ComfyUI-GGUF"
        git -C "$COMFY_DIR/custom_nodes/ComfyUI-GGUF" checkout --quiet "$GGUF_COMMIT"
    fi

    # 2. ComfyUI's own venv — CUDA 12.1 torch (like the backend) + node deps.
    if [ ! -x "$CVENV/bin/python" ]; then
        echo -e "${YELLOW}Creating ComfyUI virtual environment...${NC}"
        rm -rf "$CVENV"
        "$PYTHON" -m venv "$CVENV"
    fi
    echo -e "${BLUE}Installing ComfyUI dependencies (torch + runtime, this is slow)...${NC}"
    "$CVENV/bin/pip" install --upgrade pip --quiet
    "$CVENV/bin/pip" install --quiet --extra-index-url https://download.pytorch.org/whl/cu121 \
        -r "$COMFY_DIR/requirements.txt"
    "$CVENV/bin/pip" install --quiet -r "$COMFY_DIR/custom_nodes/ComfyUI-GGUF/requirements.txt"

    if flux_installed; then
        echo -e "${GREEN}✓ Image engine ready${NC}"
        models_hint
    else
        echo -e "${RED}✗ Image engine setup incomplete — image generation unavailable.${NC}"
    fi
fi

# --- Ollama ----------------------------------------------------------------- #
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Checking Ollama...${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

check_internet() {
    curl -s --connect-timeout 3 https://ollama.com > /dev/null 2>&1
    return $?
}

if ! command -v ollama &> /dev/null; then
    echo -e "${YELLOW}Ollama not found. Installing...${NC}"
    if ! check_internet; then
        echo -e "${RED}✗ No internet connection — cannot install Ollama.${NC}"
        exit 1
    fi
    OS="$(uname -s)"
    case "${OS}" in
        Linux*)
            curl -fsSL https://ollama.com/install.sh | sh
            echo -e "${GREEN}✓ Ollama installed${NC}"
            ;;
        *)
            echo -e "${YELLOW}Install Ollama from https://ollama.com/download, then re-run.${NC}"
            exit 1
            ;;
    esac
else
    echo -e "${GREEN}✓ Ollama is installed${NC}"
    # Update checks are now handled in the UI (opt-in), not here.
fi

# Keep Ollama local-only: force any instance we start to listen on loopback,
# so the model server is never reachable from the network. (Ollama also reads
# OLLAMA_HOST for CLI/serve; 127.0.0.1 is its default, but we set it explicitly
# so this holds even if the environment had it overridden to 0.0.0.0.)
export OLLAMA_HOST="127.0.0.1:11434"

# Start Ollama service if not already running.
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${YELLOW}Starting Ollama service (local-only, ${OLLAMA_HOST})...${NC}"
    nohup ollama serve > /dev/null 2>&1 &
    for _ in {1..10}; do
        curl -s http://localhost:11434/api/tags > /dev/null 2>&1 && break
        sleep 1
    done
fi
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Ollama service is running${NC}"
    # If Ollama was ALREADY running, our OLLAMA_HOST can't rebind it — check that
    # it isn't listening on a non-loopback (network-exposed) address and warn.
    listen_addrs="$(ss -tlnH 2>/dev/null | awk '{print $4}' | grep ':11434' || true)"
    if [ -n "$listen_addrs" ] && echo "$listen_addrs" | grep -qvE '^(127\.0\.0\.1|\[::1\]):11434$'; then
        echo -e "${YELLOW}⚠️  Ollama is listening on a non-local address — it's exposed to the network.${NC}"
        echo -e "${YELLOW}   Restart it local-only: pkill ollama && OLLAMA_HOST=127.0.0.1 ollama serve${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  Could not reach Ollama. Start it with: ollama serve${NC}"
fi

# --- Serve ------------------------------------------------------------------ #
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Serving app at http://${HOST}:${PORT}${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

uvicorn backend.main:app --host "$HOST" --port "$PORT"

deactivate 2>/dev/null || true
