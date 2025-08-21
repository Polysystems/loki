# 🐳 Docker Deployment

## Overview

Deploy Loki using Docker for consistent, isolated, and scalable environments. This guide covers everything from basic single-container deployment to production-ready multi-container orchestration.

## Quick Start

### Pull and Run

```bash
# Latest stable version
docker run -it --rm \
  -v ~/.loki:/root/.loki \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  polysystems/loki:latest

# Specific version
docker run -it --rm \
  -v ~/.loki:/root/.loki \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  polysystems/loki:0.2.0
```

## Building Custom Image

### Dockerfile

```dockerfile
# Dockerfile
FROM rust:1.83 as builder

WORKDIR /app
COPY . .

# Build with all features
RUN cargo build --release --features all

# Runtime image
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Copy binary
COPY --from=builder /app/target/release/loki /usr/local/bin/loki

# Create data directory
RUN mkdir -p /data/loki

# Set environment
ENV LOKI_DATA_DIR=/data/loki
ENV RUST_LOG=info

EXPOSE 8080

CMD ["loki", "tui"]
```

### Build and Run

```bash
# Build image
docker build -t loki:custom .

# Run custom image
docker run -it --rm \
  -v ./data:/data/loki \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  loki:custom
```

## Docker Compose

### Basic Setup

```yaml
# docker-compose.yml
version: '3.8'

services:
  loki:
    image: polysystems/loki:latest
    container_name: loki-ai
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - RUST_LOG=info
      - LOKI_DATA_DIR=/data
    volumes:
      - loki-data:/data
      - ./config:/config:ro
    ports:
      - "8080:8080"
    restart: unless-stopped
    
volumes:
  loki-data:
```

### Production Setup

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  loki:
    image: polysystems/loki:latest
    container_name: loki-ai
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - RUST_LOG=warn
      - LOKI_DATA_DIR=/data
    volumes:
      - loki-data:/data
      - ./config:/config:ro
    ports:
      - "8080:8080"
    restart: always
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
    healthcheck:
      test: ["CMD", "loki", "health"]
      interval: 30s
      timeout: 10s
      retries: 3
    
  postgres:
    image: postgres:16
    environment:
      - POSTGRES_DB=loki
      - POSTGRES_USER=loki
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres-data:/var/lib/postgresql/data
    restart: always
    
  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis-data:/data
    restart: always
    
volumes:
  loki-data:
  postgres-data:
  redis-data:
```

### Run with Compose

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f loki

# Stop services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

## Environment Variables

### Required
```bash
# At least one LLM provider
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

### Optional
```bash
# System
LOKI_DATA_DIR=/data/loki
LOKI_CONFIG=/config/config.yaml
RUST_LOG=info

# Performance
LOKI_MAX_WORKERS=8
LOKI_CACHE_SIZE=2GB

# Features
LOKI_COGNITIVE_ENABLED=true
LOKI_TOOLS_ENABLED=true
LOKI_MEMORY_PERSISTENT=true

# Network
LOKI_SERVER_HOST=0.0.0.0
LOKI_SERVER_PORT=8080
```

## Volume Management

### Persistent Data

```bash
# Create named volume
docker volume create loki-data

# Inspect volume
docker volume inspect loki-data

# Backup volume
docker run --rm \
  -v loki-data:/data \
  -v $(pwd):/backup \
  alpine tar czf /backup/loki-backup.tar.gz -C /data .

# Restore volume
docker run --rm \
  -v loki-data:/data \
  -v $(pwd):/backup \
  alpine tar xzf /backup/loki-backup.tar.gz -C /data
```

### Bind Mounts

```bash
# Use local directories
docker run -it \
  -v ./data:/data/loki \
  -v ./config:/config \
  -v ./plugins:/plugins \
  polysystems/loki:latest
```

## Networking

### Bridge Network

```bash
# Create custom network
docker network create loki-net

# Run with network
docker run -d \
  --network loki-net \
  --name loki \
  polysystems/loki:latest
```

### Host Network

```bash
# Use host network (Linux only)
docker run -d \
  --network host \
  polysystems/loki:latest
```

## Resource Limits

### CPU and Memory

```bash
# Limit resources
docker run -d \
  --cpus="2.0" \
  --memory="4g" \
  --memory-swap="4g" \
  polysystems/loki:latest
```

### GPU Support

```bash
# NVIDIA GPU (requires nvidia-docker)
docker run -d \
  --gpus all \
  polysystems/loki:latest

# Specific GPU
docker run -d \
  --gpus device=0 \
  polysystems/loki:latest
```

## Health Checks

### Dockerfile Health Check

```dockerfile
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD loki health || exit 1
```

### Runtime Health Check

```bash
docker run -d \
  --health-cmd="loki health" \
  --health-interval=30s \
  --health-timeout=3s \
  --health-retries=3 \
  polysystems/loki:latest
```

## Security

### Non-root User

```dockerfile
# In Dockerfile
RUN useradd -m -u 1000 loki
USER loki
```

### Read-only Filesystem

```bash
docker run -d \
  --read-only \
  --tmpfs /tmp \
  -v loki-data:/data \
  polysystems/loki:latest
```

### Secrets Management

```bash
# Create secret
echo "sk-..." | docker secret create openai_key -

# Use in service
docker service create \
  --secret openai_key \
  polysystems/loki:latest
```

## Monitoring

### Logs

```bash
# View logs
docker logs loki

# Follow logs
docker logs -f loki

# Last 100 lines
docker logs --tail 100 loki
```

### Stats

```bash
# Real-time stats
docker stats loki

# One-time stats
docker stats --no-stream loki
```

### Prometheus Metrics

```yaml
# docker-compose with monitoring
services:
  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"
```

## Orchestration

### Docker Swarm

```bash
# Initialize swarm
docker swarm init

# Create service
docker service create \
  --name loki \
  --replicas 3 \
  --env OPENAI_API_KEY=$OPENAI_API_KEY \
  polysystems/loki:latest

# Scale service
docker service scale loki=5
```

### Kubernetes

See [Kubernetes Deployment](kubernetes.md) for detailed K8s setup.

## Troubleshooting

### Common Issues

**Container exits immediately:**
```bash
# Check logs
docker logs loki

# Run interactively
docker run -it --rm polysystems/loki:latest /bin/bash
```

**Permission denied:**
```bash
# Fix volume permissions
docker run --rm \
  -v loki-data:/data \
  alpine chown -R 1000:1000 /data
```

**Out of memory:**
```bash
# Increase memory limit
docker run -d \
  --memory="8g" \
  polysystems/loki:latest
```

**Cannot connect:**
```bash
# Check port mapping
docker port loki

# Check network
docker network inspect bridge
```

## Best Practices

1. **Use specific tags**: Avoid `:latest` in production
2. **Set resource limits**: Prevent resource exhaustion
3. **Use volumes**: Persist important data
4. **Health checks**: Monitor container health
5. **Security**: Run as non-root, use secrets
6. **Logging**: Configure appropriate log levels
7. **Backup**: Regular volume backups
8. **Monitoring**: Track resource usage

## Multi-Stage Builds

### Optimized Production Image

```dockerfile
# Build stage
FROM rust:1.83 as builder
WORKDIR /app
COPY Cargo.* ./
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo build --release
COPY . .
RUN touch src/main.rs && cargo build --release --features all

# Runtime stage
FROM gcr.io/distroless/cc-debian12
COPY --from=builder /app/target/release/loki /loki
ENTRYPOINT ["/loki"]
```

## Development Setup

### Hot Reload

```yaml
# docker-compose.dev.yml
services:
  loki-dev:
    build:
      context: .
      dockerfile: Dockerfile.dev
    volumes:
      - .:/app
      - cargo-cache:/usr/local/cargo
    command: cargo watch -x run
    environment:
      - RUST_LOG=debug
```

---

Next: [Kubernetes Deployment](kubernetes.md) | [Quick Start](quick_start.md)