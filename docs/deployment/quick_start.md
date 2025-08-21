# 🚀 Quick Start Deployment Guide

## 5-Minute Setup

Get Loki running in under 5 minutes with this streamlined deployment guide.

## Prerequisites

- **OS**: macOS, Linux, or Windows 10+
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 20GB free space
- **Internet**: Required for API access

## Option 1: One-Line Install (Fastest)

```bash
# macOS/Linux
curl -sSL https://raw.githubusercontent.com/polysystems/loki/main/install.sh | bash

# Windows (PowerShell as Admin)
iwr -useb https://raw.githubusercontent.com/polysystems/loki/main/install.ps1 | iex
```

This automatically:
- Downloads the correct binary for your platform
- Installs to `/usr/local/bin` (or Windows equivalent)
- Creates initial configuration
- Verifies installation

## Option 2: Docker (Isolated)

```bash
# Pull and run
docker run -it \
  -v ~/.loki:/root/.loki \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  polysystems/loki:latest
```

## Option 3: Binary Download

1. Download for your platform:
   - [macOS (Apple Silicon)](https://github.com/polysystems/loki/releases/latest/download/loki-macos-arm64)
   - [macOS (Intel)](https://github.com/polysystems/loki/releases/latest/download/loki-macos-x64)
   - [Linux](https://github.com/polysystems/loki/releases/latest/download/loki-linux-x64)
   - [Windows](https://github.com/polysystems/loki/releases/latest/download/loki-windows-x64.exe)

2. Make executable and move to PATH:
```bash
chmod +x loki-*
sudo mv loki-* /usr/local/bin/loki
```

## Initial Configuration

### Step 1: Set API Keys

Create `.env` file:
```bash
cat > .env << EOF
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
EOF
```

Or use interactive setup:
```bash
loki setup
```

### Step 2: Verify Installation

```bash
# Check version
loki --version

# Test API connections
loki check-apis

# Run diagnostics
loki diagnostic
```

### Step 3: First Run

```bash
# Launch TUI
loki tui

# Or direct chat
loki chat "Hello, Loki!"
```

## Quick Configuration Examples

### Minimal Setup
```yaml
# ~/.loki/config.yaml
llm:
  default_provider: openai
  default_model: gpt-3.5-turbo
```

### Production Setup
```yaml
# ~/.loki/config.yaml
llm:
  default_provider: openai
  default_model: gpt-4-turbo-preview
  
cognitive:
  enabled: true
  
memory:
  persistent: true
  
tools:
  enabled: true
```

## Common Deployment Scenarios

### Personal Development
```bash
# Install
curl -sSL https://raw.githubusercontent.com/polysystems/loki/main/install.sh | bash

# Configure
echo "OPENAI_API_KEY=sk-..." > .env

# Run
loki tui
```

### Team Server
```bash
# Install with Docker
docker-compose up -d

# Configure for team
loki config set network.server.enabled true
loki config set network.server.port 8080

# Start daemon
loki daemon start
```

### CI/CD Integration
```yaml
# .github/workflows/loki.yml
name: Loki AI Assistant
on: [pull_request]

jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Install Loki
        run: |
          curl -sSL https://raw.githubusercontent.com/polysystems/loki/main/install.sh | bash
      - name: Code Review
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          loki agent deploy code-reviewer
          loki tool execute github review-pr --pr ${{ github.event.pull_request.number }}
```

## Cloud Deployments

### AWS EC2
```bash
# Launch instance (t3.large recommended)
aws ec2 run-instances --image-id ami-0c55b159cbfafe1f0 --instance-type t3.large

# SSH and install
ssh ec2-user@instance
curl -sSL https://raw.githubusercontent.com/polysystems/loki/main/install.sh | bash
```

### Google Cloud
```bash
# Create instance
gcloud compute instances create loki-ai \
  --machine-type=n1-standard-2 \
  --image-family=ubuntu-2204-lts

# Install
gcloud compute ssh loki-ai --command="curl -sSL https://raw.githubusercontent.com/polysystems/loki/main/install.sh | bash"
```

### Azure
```bash
# Create VM
az vm create \
  --resource-group loki-rg \
  --name loki-vm \
  --image Ubuntu2204 \
  --size Standard_B2s

# Install
az vm run-command invoke \
  --resource-group loki-rg \
  --name loki-vm \
  --command-id RunShellScript \
  --scripts "curl -sSL https://raw.githubusercontent.com/polysystems/loki/main/install.sh | bash"
```

## Kubernetes Deployment

```yaml
# loki-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: loki-ai
spec:
  replicas: 1
  selector:
    matchLabels:
      app: loki
  template:
    metadata:
      labels:
        app: loki
    spec:
      containers:
      - name: loki
        image: polysystems/loki:latest
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: loki-secrets
              key: openai-key
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
```

Deploy:
```bash
kubectl apply -f loki-deployment.yaml
```

## Verification Checklist

✅ **Installation**
```bash
loki --version  # Should show version
```

✅ **API Keys**
```bash
loki check-apis  # All should be green
```

✅ **Basic Function**
```bash
loki chat "test"  # Should respond
```

✅ **Memory**
```bash
loki memory stats  # Should show stats
```

✅ **Tools**
```bash
loki tool list  # Should list tools
```

## Quick Troubleshooting

### "Command not found"
```bash
export PATH=$PATH:/usr/local/bin
```

### "API key not found"
```bash
export OPENAI_API_KEY="sk-..."
```

### "Permission denied"
```bash
chmod +x /usr/local/bin/loki
```

### "Out of memory"
```bash
export LOKI_CACHE_SIZE=512MB
export LOKI_MAX_WORKERS=4
```

## Next Steps

1. **Configure**: See [Configuration Guide](configuration.md)
2. **Learn**: Read [Getting Started Tutorial](../tutorials/getting_started.md)
3. **Explore**: Try the [TUI Interface](../api/tui_interface.md)
4. **Extend**: Check [Plugin Development](../api/plugin_api.md)
5. **Scale**: Read [Docker Deployment](docker.md)

## Getting Help

- **Documentation**: [Full Docs](../README.md)
- **Discord**: [Join Community](https://discord.gg/eigencode)
- **Issues**: [GitHub Issues](https://github.com/polysystems/loki/issues)

---

**Ready to go!** You should now have Loki running. Try `loki tui` to explore the full interface.