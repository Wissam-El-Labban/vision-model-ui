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
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv "$VENV_DIR"
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
if ! command -v npm &> /dev/null; then
    echo -e "${RED}✗ npm not found. Install Node.js 18+ to build the UI.${NC}"
    exit 1
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
