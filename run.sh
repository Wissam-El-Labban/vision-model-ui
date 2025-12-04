#!/bin/bash

# Vision Model Chat - Startup Script
# This script creates a virtual environment, installs dependencies, and runs the app

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

VENV_DIR="venv"
REQUIREMENTS_FILE="requirements.txt"
APP_FILE="app.py"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Vision Model Chat Interface${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Virtual environment not found. Creating one...${NC}"
    
    # Create virtual environment
    python3 -m venv "$VENV_DIR"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Virtual environment created successfully${NC}"
    else
        echo -e "${RED}✗ Failed to create virtual environment${NC}"
        exit 1
    fi
    
    echo ""
    echo -e "${YELLOW}Installing dependencies...${NC}"
    
    # Activate virtual environment and install requirements
    source "$VENV_DIR/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip --quiet
    
    # Install requirements
    if [ -f "$REQUIREMENTS_FILE" ]; then
        pip install -r "$REQUIREMENTS_FILE"
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ Dependencies installed successfully${NC}"
        else
            echo -e "${RED}✗ Failed to install dependencies${NC}"
            exit 1
        fi
    else
        echo -e "${RED}✗ $REQUIREMENTS_FILE not found${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✓ Virtual environment found${NC}"
    
    # Activate existing virtual environment
    source "$VENV_DIR/bin/activate"
    
    # Check if requirements have changed and reinstall if needed
    echo -e "${BLUE}Checking dependencies...${NC}"
    pip install -r "$REQUIREMENTS_FILE" --quiet
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Checking Ollama...${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Function to check internet connectivity
check_internet() {
    curl -s --connect-timeout 3 https://ollama.com > /dev/null 2>&1
    return $?
}

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo -e "${YELLOW}Ollama not found. Installing Ollama...${NC}"
    
    # Check internet connectivity
    if ! check_internet; then
        echo -e "${RED}✗ No internet connection detected${NC}"
        echo -e "${YELLOW}Cannot install Ollama without internet.${NC}"
        echo -e "${YELLOW}Please connect to the internet and run this script again.${NC}"
        exit 1
    fi
    
    # Detect OS
    OS="$(uname -s)"
    case "${OS}" in
        Linux*)
            # Install Ollama on Linux
            curl -fsSL https://ollama.com/install.sh | sh
            
            if [ $? -eq 0 ]; then
                echo -e "${GREEN}✓ Ollama installed successfully${NC}"
            else
                echo -e "${RED}✗ Failed to install Ollama${NC}"
                echo -e "${YELLOW}Please visit https://ollama.com/download for manual installation${NC}"
                exit 1
            fi
            ;;
        Darwin*)
            # macOS
            echo -e "${YELLOW}Please install Ollama from: https://ollama.com/download${NC}"
            echo -e "${YELLOW}After installation, run this script again.${NC}"
            exit 1
            ;;
        *)
            echo -e "${RED}✗ Unsupported OS: ${OS}${NC}"
            echo -e "${YELLOW}Please visit https://ollama.com/download for manual installation${NC}"
            exit 1
            ;;
    esac
else
    echo -e "${GREEN}✓ Ollama is installed${NC}"
    
    # Check for updates if internet is available
    if check_internet; then
        echo -e "${BLUE}Checking for Ollama updates...${NC}"
        
        # Get current version (use -oE for macOS compatibility instead of -oP)
        CURRENT_VERSION=$(ollama --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -n1 || echo "unknown")
        
        # Get latest version from GitHub API with timeout and better error handling
        LATEST_VERSION=$(curl -s --max-time 5 https://api.github.com/repos/ollama/ollama/releases/latest 2>/dev/null | grep -oE '"tag_name": "v[0-9.]+"' | grep -oE '[0-9.]+' || echo "unknown")
        
        if [ "$CURRENT_VERSION" != "unknown" ] && [ "$LATEST_VERSION" != "unknown" ]; then
            # Both versions successfully retrieved - compare them
            if [ "$CURRENT_VERSION" != "$LATEST_VERSION" ]; then
                echo -e "${YELLOW}Update available: v$CURRENT_VERSION → v$LATEST_VERSION${NC}"
                
                # Only update on Linux (automatic updates supported)
                OS="$(uname -s)"
                if [ "${OS}" = "Linux" ]; then
                    echo -e "${YELLOW}Updating Ollama...${NC}"
                    curl -fsSL https://ollama.com/install.sh | sh
                    
                    if [ $? -eq 0 ]; then
                        echo -e "${GREEN}✓ Ollama updated to v$LATEST_VERSION${NC}"
                    else
                        echo -e "${YELLOW}⚠️  Update failed, continuing with current version${NC}"
                    fi
                else
                    echo -e "${YELLOW}⚠️  New version available. Please update manually from https://ollama.com/download${NC}"
                fi
            else
                echo -e "${GREEN}✓ Ollama is up to date (v$CURRENT_VERSION)${NC}"
            fi
        elif [ "$LATEST_VERSION" = "unknown" ] && [ "$CURRENT_VERSION" != "unknown" ]; then
            # Failed to get latest version from GitHub
            echo -e "${YELLOW}⚠️  Could not check for updates (GitHub API unavailable)${NC}"
            echo -e "${GREEN}✓ Continuing with current version (v$CURRENT_VERSION)${NC}"
        elif [ "$CURRENT_VERSION" = "unknown" ] && [ "$LATEST_VERSION" != "unknown" ]; then
            # Failed to get current version but got latest
            echo -e "${YELLOW}⚠️  Could not determine current Ollama version${NC}"
            echo -e "${BLUE}Latest version available: v$LATEST_VERSION${NC}"
        else
            # Both failed
            echo -e "${YELLOW}⚠️  Could not verify version information${NC}"
        fi
    else
        echo -e "${BLUE}No internet connection - skipping update check${NC}"
    fi
fi

# Check if Ollama is running, start if not
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${YELLOW}Starting Ollama service...${NC}"
    
    # Start Ollama in background
    nohup ollama serve > /dev/null 2>&1 &
    
    # Wait for Ollama to start (max 10 seconds)
    for i in {1..10}; do
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            echo -e "${GREEN}✓ Ollama service started${NC}"
            break
        fi
        sleep 1
    done
    
    if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo -e "${YELLOW}⚠️  Ollama service may not have started properly${NC}"
        echo -e "${YELLOW}   You may need to start it manually with: ollama serve${NC}"
    fi
else
    echo -e "${GREEN}✓ Ollama service is running${NC}"
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Starting Streamlit app...${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Run the Streamlit app
streamlit run "$APP_FILE"

# Deactivate virtual environment on exit (if script is interrupted)
deactivate 2>/dev/null || true
