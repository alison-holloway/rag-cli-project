#!/bin/bash
#
# Start the RAG CLI Web Application
# Launches both the FastAPI backend and React frontend
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/.web-pids"
LOG_DIR="$SCRIPT_DIR/logs"

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

echo -e "${BLUE}Starting RAG CLI Web Application...${NC}"
echo ""

# Check if already running
if [ -f "$PID_FILE" ]; then
    echo -e "${YELLOW}Warning: PID file exists. Services may already be running.${NC}"
    echo -e "Run ${GREEN}./stop-web.sh${NC} first, or delete ${PID_FILE} if stale."
    exit 1
fi

# Check for required dependencies
if ! command -v node &> /dev/null; then
    echo -e "${RED}Error: Node.js is not installed.${NC}"
    echo "Install with: brew install node (macOS) or apt install nodejs (Linux)"
    exit 1
fi

if [ ! -d "$SCRIPT_DIR/venv" ]; then
    echo -e "${RED}Error: Python virtual environment not found.${NC}"
    echo "Run: python -m venv venv && pip install -r requirements.txt"
    exit 1
fi

if [ ! -d "$SCRIPT_DIR/frontend/node_modules" ]; then
    echo -e "${YELLOW}Installing frontend dependencies...${NC}"
    cd "$SCRIPT_DIR/frontend"
    npm install
    cd "$SCRIPT_DIR"
fi

# Start the backend
echo -e "${BLUE}[1/2]${NC} Starting FastAPI backend on port 8000..."
source "$SCRIPT_DIR/venv/bin/activate"
uvicorn backend.main:app --port 8000 --host 127.0.0.1 > "$LOG_DIR/backend.log" 2>&1 &
BACKEND_PID=$!
echo "      Backend PID: $BACKEND_PID"

# Wait a moment for backend to start
sleep 2

# Check if backend started successfully
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo -e "${RED}Error: Backend failed to start. Check $LOG_DIR/backend.log${NC}"
    exit 1
fi

# Start the frontend
echo -e "${BLUE}[2/2]${NC} Starting React frontend on port 5173..."
cd "$SCRIPT_DIR/frontend"
npm run dev > "$LOG_DIR/frontend.log" 2>&1 &
FRONTEND_PID=$!
cd "$SCRIPT_DIR"
echo "      Frontend PID: $FRONTEND_PID"

# Wait a moment for frontend to start
sleep 2

# Check if frontend started successfully
if ! kill -0 $FRONTEND_PID 2>/dev/null; then
    echo -e "${RED}Error: Frontend failed to start. Check $LOG_DIR/frontend.log${NC}"
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

# Save PIDs for stop script
echo "$BACKEND_PID" > "$PID_FILE"
echo "$FRONTEND_PID" >> "$PID_FILE"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  RAG CLI Web Application Started!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "  Web UI:    ${BLUE}http://localhost:5173${NC}"
echo -e "  API:       ${BLUE}http://localhost:8000${NC}"
echo -e "  API Docs:  ${BLUE}http://localhost:8000/docs${NC}"
echo ""
echo -e "  View logs:"
echo -e "    Backend:  ${YELLOW}tail -f logs/backend.log${NC}"
echo -e "    Frontend: ${YELLOW}tail -f logs/frontend.log${NC}"
echo ""
echo -e "  Stop with: ${GREEN}./stop-web.sh${NC}"
echo ""
