#!/bin/bash
set -e

# Loki AI Installation Script
# This script automatically downloads and installs the latest version of Loki

REPO="polysystems/loki"
INSTALL_DIR="${INSTALL_DIR:-$HOME/.local/bin}"
BINARY_NAME="loki"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Detect OS and Architecture
detect_platform() {
    OS="$(uname -s)"
    ARCH="$(uname -m)"
    
    case "$OS" in
        Linux*)
            PLATFORM="linux"
            ;;
        Darwin*)
            PLATFORM="macos"
            ;;
        MINGW*|MSYS*|CYGWIN*)
            PLATFORM="windows"
            ;;
        *)
            echo -e "${RED}Unsupported operating system: $OS${NC}"
            exit 1
            ;;
    esac
    
    case "$ARCH" in
        x86_64|amd64)
            ARCH="x64"
            ;;
        aarch64|arm64)
            ARCH="arm64"
            ;;
        *)
            echo -e "${RED}Unsupported architecture: $ARCH${NC}"
            exit 1
            ;;
    esac
    
    # Construct artifact name
    if [ "$PLATFORM" = "windows" ]; then
        ARTIFACT="${BINARY_NAME}-${PLATFORM}-${ARCH}.exe"
    else
        ARTIFACT="${BINARY_NAME}-${PLATFORM}-${ARCH}"
    fi
    
    echo -e "${GREEN}Detected platform:${NC} $PLATFORM-$ARCH"
}

# Get latest release URL
get_latest_release() {
    echo -e "${YELLOW}Fetching latest release...${NC}"
    
    # Try to get the latest release URL
    LATEST_RELEASE=$(curl -sL "https://api.github.com/repos/${REPO}/releases/latest" | grep '"browser_download_url"' | grep "$ARTIFACT" | cut -d '"' -f 4)
    
    if [ -z "$LATEST_RELEASE" ]; then
        echo -e "${RED}Could not find a release for ${ARTIFACT}${NC}"
        echo -e "${YELLOW}Please check: https://github.com/${REPO}/releases${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Found release:${NC} $LATEST_RELEASE"
}

# Download and install
install_loki() {
    echo -e "${YELLOW}Downloading Loki...${NC}"
    
    # Create install directory if it doesn't exist
    mkdir -p "$INSTALL_DIR"
    
    # Download binary
    TEMP_FILE=$(mktemp)
    if curl -L -o "$TEMP_FILE" "$LATEST_RELEASE"; then
        echo -e "${GREEN}Download complete${NC}"
    else
        echo -e "${RED}Download failed${NC}"
        rm -f "$TEMP_FILE"
        exit 1
    fi
    
    # Move to install directory
    mv "$TEMP_FILE" "$INSTALL_DIR/$BINARY_NAME"
    chmod +x "$INSTALL_DIR/$BINARY_NAME"
    
    echo -e "${GREEN}Loki installed to: ${INSTALL_DIR}/${BINARY_NAME}${NC}"
}

# Add to PATH if needed
update_path() {
    if [[ ":$PATH:" != *":$INSTALL_DIR:"* ]]; then
        echo -e "${YELLOW}Adding $INSTALL_DIR to PATH...${NC}"
        
        # Detect shell and update appropriate config file
        SHELL_NAME=$(basename "$SHELL")
        case "$SHELL_NAME" in
            bash)
                RC_FILE="$HOME/.bashrc"
                ;;
            zsh)
                RC_FILE="$HOME/.zshrc"
                ;;
            fish)
                RC_FILE="$HOME/.config/fish/config.fish"
                ;;
            *)
                RC_FILE="$HOME/.profile"
                ;;
        esac
        
        echo "" >> "$RC_FILE"
        echo "# Added by Loki installer" >> "$RC_FILE"
        echo "export PATH=\"\$PATH:$INSTALL_DIR\"" >> "$RC_FILE"
        
        echo -e "${GREEN}PATH updated in $RC_FILE${NC}"
        echo -e "${YELLOW}Please run: source $RC_FILE${NC}"
        echo -e "${YELLOW}Or restart your terminal${NC}"
    else
        echo -e "${GREEN}$INSTALL_DIR is already in PATH${NC}"
    fi
}

# Verify installation
verify_installation() {
    if [ -x "$INSTALL_DIR/$BINARY_NAME" ]; then
        echo -e "${GREEN}✅ Loki installation successful!${NC}"
        echo ""
        echo "To get started, run:"
        echo "  ${BINARY_NAME} --help"
        echo ""
        echo "Setup your API keys:"
        echo "  ${BINARY_NAME} setup"
        echo ""
        echo "Start the TUI:"
        echo "  ${BINARY_NAME} tui"
    else
        echo -e "${RED}Installation verification failed${NC}"
        exit 1
    fi
}

# Main installation flow
main() {
    echo "🌀 Loki AI Installer"
    echo "===================="
    echo ""
    
    detect_platform
    get_latest_release
    install_loki
    update_path
    verify_installation
}

# Run main function
main "$@"