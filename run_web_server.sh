#!/bin/bash
# Helper script untuk menjalankan web server

# Color codes
BLUE='\033[0;34m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default values
HOST="127.0.0.1"
PORT="8080"
PUBLIC=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --public)
            PUBLIC=true
            HOST="0.0.0.0"
            shift
            ;;
        -h|--help)
            echo "Usage: ./run_web_server.sh [options]"
            echo ""
            echo "Options:"
            echo "  --port PORT    Port to run on (default: 8080)"
            echo "  --host HOST    Host to bind to (default: 127.0.0.1)"
            echo "  --public       Bind to 0.0.0.0 for network access"
            echo "  -h, --help     Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Face Recognition WebSocket Client - Web Server                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo -e "${GREEN}Starting server on ${HOST}:${PORT}...${NC}"
echo ""

# Run the Python server
python3 web_server.py --host "$HOST" --port "$PORT"
