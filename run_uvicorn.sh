#!/bin/bash
# =============================================================================
# Uvicorn Server Launcher for Face Recognition Application
# =============================================================================
#
# Usage:
#   ./run_uvicorn.sh                    # Development mode (auto-reload)
#   ./run_uvicorn.sh --production       # Production mode (multi-worker)
#   ./run_uvicorn.sh --ssl              # With SSL/HTTPS
#   ./run_uvicorn.sh --help             # Show help
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
HOST="${UVICORN_HOST:-0.0.0.0}"
PORT="${UVICORN_PORT:-8000}"
WORKERS="${UVICORN_WORKERS:-4}"
LOG_LEVEL="${UVICORN_LOG_LEVEL:-info}"
MODE="development"
USE_SSL=false
SSL_KEYFILE="${SSL_KEYFILE:-}"
SSL_CERTFILE="${SSL_CERTFILE:-}"

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="${SCRIPT_DIR}/face_recognition_app"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --production|-p)
            MODE="production"
            shift
            ;;
        --development|-d)
            MODE="development"
            shift
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --workers|-w)
            WORKERS="$2"
            shift 2
            ;;
        --ssl)
            USE_SSL=true
            shift
            ;;
        --ssl-key)
            SSL_KEYFILE="$2"
            shift 2
            ;;
        --ssl-cert)
            SSL_CERTFILE="$2"
            shift 2
            ;;
        --log-level|-l)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --production, -p     Run in production mode (multi-worker)"
            echo "  --development, -d    Run in development mode (auto-reload)"
            echo "  --host HOST          Bind to host (default: 0.0.0.0)"
            echo "  --port PORT          Bind to port (default: 8000)"
            echo "  --workers, -w NUM    Number of workers (production only, default: 4)"
            echo "  --ssl                Enable SSL/HTTPS"
            echo "  --ssl-key FILE       Path to SSL key file"
            echo "  --ssl-cert FILE      Path to SSL certificate file"
            echo "  --log-level, -l LVL  Log level (debug, info, warning, error, critical)"
            echo "  --help, -h           Show this help message"
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
echo "║  Face Recognition - Uvicorn ASGI Server                        ║"
echo "╠════════════════════════════════════════════════════════════════╣"
echo "║  High-performance server with WebSocket & HTTPS support        ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Change to app directory
cd "$APP_DIR"

# Build uvicorn command
UVICORN_CMD="uvicorn face_app.asgi:application"
UVICORN_CMD+=" --host $HOST"
UVICORN_CMD+=" --port $PORT"
UVICORN_CMD+=" --log-level $LOG_LEVEL"

# Add performance options
UVICORN_CMD+=" --loop uvloop"
UVICORN_CMD+=" --http httptools"
UVICORN_CMD+=" --ws websockets"
UVICORN_CMD+=" --lifespan off"
UVICORN_CMD+=" --timeout-keep-alive 120"

if [ "$MODE" == "production" ]; then
    echo -e "${GREEN}Mode: Production${NC}"
    echo -e "Workers: $WORKERS"
    UVICORN_CMD+=" --workers $WORKERS"
else
    echo -e "${YELLOW}Mode: Development${NC}"
    UVICORN_CMD+=" --reload"
    UVICORN_CMD+=" --reload-dir ."
fi

# Add SSL options
if [ "$USE_SSL" = true ]; then
    if [ -z "$SSL_KEYFILE" ] || [ -z "$SSL_CERTFILE" ]; then
        # Check for default SSL files
        if [ -f "../nginx/ssl/privkey.pem" ] && [ -f "../nginx/ssl/fullchain.pem" ]; then
            SSL_KEYFILE="../nginx/ssl/privkey.pem"
            SSL_CERTFILE="../nginx/ssl/fullchain.pem"
        elif [ -f "../nginx/ssl/server.key" ] && [ -f "../nginx/ssl/server.crt" ]; then
            SSL_KEYFILE="../nginx/ssl/server.key"
            SSL_CERTFILE="../nginx/ssl/server.crt"
        else
            echo -e "${RED}Error: SSL enabled but no certificate files found.${NC}"
            echo "Please specify --ssl-key and --ssl-cert options,"
            echo "or place certificate files in nginx/ssl/ directory."
            exit 1
        fi
    fi
    echo -e "${GREEN}SSL: Enabled${NC}"
    echo -e "  Key: $SSL_KEYFILE"
    echo -e "  Cert: $SSL_CERTFILE"
    UVICORN_CMD+=" --ssl-keyfile $SSL_KEYFILE"
    UVICORN_CMD+=" --ssl-certfile $SSL_CERTFILE"
    PROTOCOL="https"
else
    PROTOCOL="http"
fi

echo ""
echo -e "${GREEN}Server URL: ${PROTOCOL}://${HOST}:${PORT}${NC}"
echo -e "${GREEN}WebSocket URL: wss://${HOST}:${PORT}/ws/${NC}" 
echo ""
echo -e "${BLUE}Starting server...${NC}"
echo ""

# Run uvicorn
exec $UVICORN_CMD
