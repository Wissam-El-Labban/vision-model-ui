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
echo -e "${GREEN}Starting Streamlit app...${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if Ollama is running (optional check)
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  Warning: Ollama doesn't seem to be running on localhost:11434${NC}"
    echo -e "${YELLOW}   Make sure to start Ollama with: ollama serve${NC}"
    echo ""
fi

# Run the Streamlit app
streamlit run "$APP_FILE"

# Deactivate virtual environment on exit (if script is interrupted)
deactivate 2>/dev/null || true
