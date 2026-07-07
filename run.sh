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
