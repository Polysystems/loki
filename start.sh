#!/bin/bash

# Loki AI Startup Script
# This script helps you get Loki running quickly

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "🚀 Loki AI Startup Script"
echo "========================"

# Check for .env file
if [ ! -f .env ]; then
    echo -e "${RED}❌ Error: .env file not found!${NC}"
    echo "Please create a .env file with your API keys."
    echo "See QUICKSTART.md for the template."
    exit 1
fi

# Check for Rust
if ! command -v cargo &> /dev/null; then
    echo -e "${RED}❌ Error: Rust not installed!${NC}"
    echo "Install Rust from: https://rustup.rs"
    exit 1
fi

# Check for Ollama
if ! command -v ollama &> /dev/null; then
    echo -e "${YELLOW}⚠️  Warning: Ollama not installed!${NC}"
    echo "Loki will use API models instead of local ones."
    echo "Install Ollama for better performance: brew install ollama"
else
    # Check if Ollama is running
    if ! pgrep -x "ollama" > /dev/null; then
        echo "Starting Ollama service..."
        ollama serve &
        sleep 2
    fi
fi

# Create data directory
mkdir -p data/logs

# Build the project
echo -e "${GREEN}Building Loki...${NC}"
cargo build

# Check API configuration
echo ""
echo -e "${GREEN}Checking API configuration...${NC}"
cargo run --bin loki -- check-apis || true
echo ""

# Check what the user wants to do
if [ "$1" == "daemon" ]; then
    echo -e "${GREEN}Starting Loki daemon...${NC}"
    cargo run --bin loki-daemon
elif [ "$1" == "tui" ]; then
    echo -e "${GREEN}Starting Loki TUI...${NC}"
    cargo run --bin loki-tui
elif [ "$1" == "interact" ]; then
    echo -e "${GREEN}Starting interactive mode...${NC}"
    cargo run --bin loki -- interact
elif [ "$1" == "check-apis" ]; then
    echo -e "${GREEN}Checking API configuration...${NC}"
    cargo run --bin loki -- check-apis
else
    echo -e "${GREEN}✅ Loki is ready!${NC}"
    echo ""
    echo "Usage:"
    echo "  ./start.sh          - Show this help"
    echo "  ./start.sh daemon   - Start Loki daemon (background service)"
    echo "  ./start.sh tui      - Start Terminal UI"
    echo "  ./start.sh interact - Start interactive CLI"
    echo "  ./start.sh check-apis - Check API configuration"
    echo ""
    echo "Or run directly:"
    echo "  cargo run --bin loki -- --help"
fi 