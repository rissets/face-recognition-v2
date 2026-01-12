#!/bin/bash
# =============================================================================
# Gunicorn Server Launcher for Face Recognition Application
# Uses multiple worker processes for better CPU utilization
# =============================================================================
#
# Usage:
#   ./run_gunicorn.sh                     # Default 4 workers
#   ./run_gunicorn.sh --workers 8         # Specify workers
#   ./run_gunicorn.sh --production        # Production settings
#   ./run_gunicorn.sh --help              # Show help
#
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default settings
HOST="${GUNICORN_HOST:-0.0.0.0}"
PORT="${GUNICORN_PORT:-8000}"
WORKERS="${GUNICORN_WORKERS:-4}"
FACE_MESH_POOL="${FACE_MESH_POOL_SIZE:-8}"
LOG_LEVEL="${GUNICORN_LOG_LEVEL:-info}"
MODE="production"

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="${SCRIPT_DIR}/face_recognition_app"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --workers|-w)
            WORKERS="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --port|-p)
            PORT="$2"
            shift 2
            ;;
        --pool-size)
            FACE_MESH_POOL="$2"
            shift 2
            ;;
        --log-level|-l)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --production)
            MODE="production"
            shift
            ;;
        --development|-d)
            MODE="development"
            shift
            ;;
        --reload)
            RELOAD=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Multi-Worker Gunicorn Server for Face Recognition"
            echo ""
            echo "Options:"
            echo "  --workers, -w NUM     Number of worker processes (default: 4)"
            echo "  --host HOST           Bind to host (default: 0.0.0.0)"
            echo "  --port, -p PORT       Bind to port (default: 8000)"
            echo "  --pool-size NUM       FaceMesh pool size per worker (default: 8)"
            echo "  --log-level, -l LVL   Log level (debug, info, warning, error)"
            echo "  --production          Production mode (default)"
            echo "  --development, -d     Development mode"
            echo "  --reload              Enable auto-reload (development only)"
            echo "  --help, -h            Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --workers 8                    # 8 workers, default pool"
            echo "  $0 --workers 4 --pool-size 4     # 4 workers, 4 FaceMesh per worker"
            echo "  $0 -w 2 --reload                 # 2 workers with reload"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Print banner
echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Face Recognition - Gunicorn + Uvicorn Multi-Worker Server     ║"
echo "╠════════════════════════════════════════════════════════════════╣"
echo "║  High-performance multi-process server with WebSocket support  ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Change to app directory
cd "$APP_DIR"

# Export environment variables
export GUNICORN_WORKERS="$WORKERS"
export GUNICORN_BIND="${HOST}:${PORT}"
export GUNICORN_LOG_LEVEL="$LOG_LEVEL"
export FACE_MESH_POOL_SIZE="$FACE_MESH_POOL"

# Calculate resources
TOTAL_FACE_MESH=$((WORKERS * FACE_MESH_POOL))

echo -e "${GREEN}Configuration:${NC}"
echo -e "  Mode:           $MODE"
echo -e "  Workers:        $WORKERS processes"
echo -e "  Bind:           ${HOST}:${PORT}"
echo -e "  FaceMesh Pool:  $FACE_MESH_POOL per worker ($TOTAL_FACE_MESH total)"
echo -e "  Log Level:      $LOG_LEVEL"
echo ""

# Build gunicorn command
GUNICORN_CMD="gunicorn face_app.asgi:application"
GUNICORN_CMD+=" -c gunicorn_config.py"
GUNICORN_CMD+=" -w $WORKERS"
GUNICORN_CMD+=" -b ${HOST}:${PORT}"
GUNICORN_CMD+=" --log-level $LOG_LEVEL"

# Add reload for development
if [ "$RELOAD" = true ] || [ "$MODE" = "development" ]; then
    GUNICORN_CMD+=" --reload"
    echo -e "${YELLOW}Auto-reload enabled${NC}"
fi

echo -e "${BLUE}Starting server...${NC}"
echo ""
echo -e "${GREEN}Server URL: http://${HOST}:${PORT}${NC}"
echo -e "${GREEN}WebSocket:  ws://${HOST}:${PORT}/ws/${NC}"
echo ""

# Run gunicorn
exec $GUNICORN_CMD
