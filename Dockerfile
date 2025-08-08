# === Build Stage ===
FROM rust:1.75-slim as builder

# Install dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    libpq-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy manifests first for better caching
COPY Cargo.toml Cargo.lock ./

# Copy source code
COPY src ./src
COPY benches ./benches
COPY tests ./tests

# Build optimized binary
RUN cargo build --release --bin loki-daemon

# === Runtime Stage ===
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create loki user for security
RUN useradd -r -s /bin/false loki

# Copy binary and set permissions
COPY --from=builder /app/target/release/loki-daemon /usr/local/bin/loki-daemon
RUN chmod +x /usr/local/bin/loki-daemon

# Create data directory
RUN mkdir -p /data && chown loki:loki /data

# Switch to non-root user
USER loki

# Set working directory
WORKDIR /data

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose ports
EXPOSE 8080 8081

# Start Loki daemon
CMD ["loki-daemon"] 