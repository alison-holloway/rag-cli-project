#!/bin/bash
#
# Stop the RAG CLI Web Application
# Cleanly shuts down both backend and frontend services
#

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/.web-pids"

echo -e "${BLUE}Stopping RAG CLI Web Application...${NC}"
echo ""

# Check if PID file exists
if [ ! -f "$PID_FILE" ]; then
    echo -e "${YELLOW}No PID file found. Services may not be running.${NC}"
    echo ""
    echo "Attempting to find and stop any running services..."

    # Try to find and kill any running uvicorn or vite processes for this project
    BACKEND_PIDS=$(pgrep -f "uvicorn backend.main:app" 2>/dev/null || true)
    FRONTEND_PIDS=$(pgrep -f "vite.*frontend" 2>/dev/null || true)

    if [ -n "$BACKEND_PIDS" ]; then
        echo -e "  Stopping backend (PIDs: $BACKEND_PIDS)..."
        echo "$BACKEND_PIDS" | xargs kill 2>/dev/null || true
    fi

    if [ -n "$FRONTEND_PIDS" ]; then
        echo -e "  Stopping frontend (PIDs: $FRONTEND_PIDS)..."
        echo "$FRONTEND_PIDS" | xargs kill 2>/dev/null || true
    fi

    if [ -z "$BACKEND_PIDS" ] && [ -z "$FRONTEND_PIDS" ]; then
        echo -e "${GREEN}No running services found.${NC}"
    else
        echo -e "${GREEN}Done.${NC}"
    fi
    exit 0
fi

# Read PIDs from file
PIDS=($(cat "$PID_FILE"))
BACKEND_PID="${PIDS[0]}"
FRONTEND_PID="${PIDS[1]}"

STOPPED=0

# Stop backend
if [ -n "$BACKEND_PID" ]; then
    if kill -0 "$BACKEND_PID" 2>/dev/null; then
        echo -e "  Stopping backend (PID: $BACKEND_PID)..."
        kill "$BACKEND_PID" 2>/dev/null
        STOPPED=$((STOPPED + 1))
    else
        echo -e "  Backend (PID: $BACKEND_PID) already stopped."
    fi
fi

# Stop frontend
if [ -n "$FRONTEND_PID" ]; then
    if kill -0 "$FRONTEND_PID" 2>/dev/null; then
        echo -e "  Stopping frontend (PID: $FRONTEND_PID)..."
        kill "$FRONTEND_PID" 2>/dev/null
        STOPPED=$((STOPPED + 1))
    else
        echo -e "  Frontend (PID: $FRONTEND_PID) already stopped."
    fi
fi

# Clean up PID file
rm -f "$PID_FILE"

echo ""
if [ $STOPPED -gt 0 ]; then
    echo -e "${GREEN}RAG CLI Web Application stopped.${NC}"
else
    echo -e "${YELLOW}No running services were found.${NC}"
fi
echo ""
