#!/bin/bash
set -e

# Build release binaries for all platforms
echo "🔨 Building Loki release binaries for all platforms..."

# Create releases directory
mkdir -p releases

# Build for macOS ARM64 (Apple Silicon)
echo "📦 Building for macOS ARM64..."
cargo build --release --target aarch64-apple-darwin
cp target/aarch64-apple-darwin/release/loki releases/loki-macos-arm64

# Build for macOS x86_64 (Intel)
echo "📦 Building for macOS x86_64..."
cargo build --release --target x86_64-apple-darwin
cp target/x86_64-apple-darwin/release/loki releases/loki-macos-x64

# Build for Linux x86_64
echo "📦 Building for Linux x86_64..."
cargo build --release --target x86_64-unknown-linux-gnu
cp target/x86_64-unknown-linux-gnu/release/loki releases/loki-linux-x64

# Build for Windows x86_64
echo "📦 Building for Windows x86_64..."
cargo build --release --target x86_64-pc-windows-gnu
cp target/x86_64-pc-windows-gnu/release/loki.exe releases/loki-windows-x64.exe

# Create checksums
echo "🔐 Generating checksums..."
cd releases
shasum -a 256 * > checksums.txt
cd ..

echo "✅ Build complete! Binaries available in ./releases/"
ls -lh releases/