#!/bin/bash
# =============================================================================
# CLV Prediction System - Unix/Linux/Mac Setup Script
# Author: Ali Abbass (OTE22)
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo ""
echo -e "${BLUE}====================================================${NC}"
echo -e "${BLUE}   CLV Prediction System - Setup Script${NC}"
echo -e "${BLUE}   Author: Ali Abbass (OTE22)${NC}"
echo -e "${BLUE}====================================================${NC}"
echo ""

# Function to check command existence
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python
echo -e "${YELLOW}[CHECK]${NC} Checking Python installation..."
if command_exists python3; then
    PYTHON_CMD="python3"
elif command_exists python; then
    PYTHON_CMD="python"
else
    echo -e "${RED}[ERROR]${NC} Python is not installed"
    echo "Please install Python 3.9+ from https://python.org"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
echo -e "${GREEN}[SUCCESS]${NC} Python $PYTHON_VERSION found"

# Check Python version (need 3.9+)
PYTHON_MAJOR=$($PYTHON_CMD -c 'import sys; print(sys.version_info.major)')
PYTHON_MINOR=$($PYTHON_CMD -c 'import sys; print(sys.version_info.minor)')

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]); then
    echo -e "${RED}[ERROR]${NC} Python 3.9+ is required (found $PYTHON_VERSION)"
    exit 1
fi

# Step 1: Create virtual environment
echo ""
echo -e "${YELLOW}[STEP 1/5]${NC} Creating virtual environment..."
if [ -d "venv" ]; then
    echo -e "${BLUE}[INFO]${NC} Virtual environment already exists"
else
    $PYTHON_CMD -m venv venv
    echo -e "${GREEN}[SUCCESS]${NC} Virtual environment created"
fi

# Step 2: Activate virtual environment
echo ""
echo -e "${YELLOW}[STEP 2/5]${NC} Activating virtual environment..."
source venv/bin/activate
echo -e "${GREEN}[SUCCESS]${NC} Virtual environment activated"

# Step 3: Upgrade pip
echo ""
echo -e "${YELLOW}[STEP 3/5]${NC} Upgrading pip..."
pip install --upgrade pip --quiet

# Step 4: Install dependencies
echo ""
echo -e "${YELLOW}[STEP 4/5]${NC} Installing dependencies..."
pip install -r requirements.txt --quiet
echo -e "${GREEN}[SUCCESS]${NC} Dependencies installed"

# Step 5: Create directories
echo ""
echo -e "${YELLOW}[STEP 5/5]${NC} Creating directories..."
mkdir -p logs models data
echo -e "${GREEN}[SUCCESS]${NC} Directories created"

# Generate data if not exists
if [ ! -f "data/customers.csv" ]; then
    echo ""
    echo -e "${BLUE}[INFO]${NC} Generating sample data..."
    $PYTHON_CMD data/generate_data.py
fi

# Make scripts executable
chmod +x setup.sh 2>/dev/null || true

echo ""
echo -e "${GREEN}====================================================${NC}"
echo -e "${GREEN}   Setup Complete!${NC}"
echo -e "${GREEN}====================================================${NC}"
echo ""
echo "  To start the API server:"
echo "    1. Activate venv:  source venv/bin/activate"
echo "    2. Run server:     python -m uvicorn backend.api_enhanced:app --reload"
echo ""
echo "  Or use Docker:"
echo "    docker-compose up -d"
echo ""
echo "  API will be available at: http://localhost:8000"
echo "  API Docs at: http://localhost:8000/api/docs"
echo "  Frontend at: http://localhost:3000 (with Docker)"
echo ""
