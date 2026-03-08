#!/bin/bash
# Start the ADHD environment FastAPI server locally
# Usage: ./scripts/run_local_server.sh

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Starting ADHD Environment Server Locally${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Check if venv exists
if [ ! -d ".venv" ]; then
    echo -e "${RED}Error: .venv not found. Run 'uv sync' first.${NC}"
    exit 1
fi

echo -e "${GREEN}Starting server on http://localhost:8001${NC}\n"

# Set PYTHONPATH and run server
PYTHONPATH=/workspaces/adhd-coach .venv/bin/python -m uvicorn \
    src.environment.server:app \
    --host 0.0.0.0 \
    --port 8001 \
    --reload
