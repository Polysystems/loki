# 📟 CLI Reference

## Overview

The Loki command-line interface provides comprehensive access to all system features, from basic chat interactions to advanced cognitive operations and system management.

## Installation

```bash
# Quick install
curl -sSL https://raw.githubusercontent.com/polysystems/loki/main/install.sh | bash

# Or download binary directly
wget https://github.com/polysystems/loki/releases/latest/download/loki-$(uname -s)-$(uname -m)
chmod +x loki-*
sudo mv loki-* /usr/local/bin/loki
```

## Global Options

```bash
loki [GLOBAL OPTIONS] <COMMAND> [ARGS]
```

### Global Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--version`, `-V` | Print version information | - |
| `--help`, `-h` | Display help message | - |
| `--config`, `-c` | Specify config file path | `~/.loki/config.yaml` |
| `--env` | Environment file path | `.env` |
| `--log-level` | Set log level (trace/debug/info/warn/error) | `info` |
| `--quiet`, `-q` | Suppress output | `false` |
| `--verbose`, `-v` | Verbose output | `false` |
| `--json` | Output in JSON format | `false` |

## Core Commands

### `tui` - Terminal User Interface

Launch the interactive terminal interface:

```bash
loki tui [OPTIONS]
```

**Options:**
- `--theme <THEME>` - Color theme (dark/light/auto)
- `--layout <LAYOUT>` - Layout preset (default/compact/wide)
- `--tabs <TABS>` - Initial tabs to open (comma-separated)

**Examples:**
```bash
# Launch with default settings
loki tui

# Dark theme with specific tabs
loki tui --theme dark --tabs chat,memory,tools

# Compact layout
loki tui --layout compact
```

### `chat` - Direct Chat Interface

Start a direct chat session:

```bash
loki chat [OPTIONS] [MESSAGE]
```

**Options:**
- `--model <MODEL>` - Specify model (gpt-4, claude-3, etc.)
- `--cognitive` - Enable cognitive enhancements
- `--memory` - Enable memory system
- `--tools` - Enable tool usage
- `--stream` - Stream responses
- `--context <FILE>` - Load context from file

**Examples:**
```bash
# Simple chat
loki chat "Hello, how are you?"

# Chat with cognitive features
loki chat --cognitive --memory "Let's solve a complex problem"

# Chat with specific model
loki chat --model claude-3-opus "Explain quantum computing"

# Interactive session
loki chat
> What's the weather like?
> Tell me a joke
> exit
```

### `cognitive` - Cognitive System Management

Manage cognitive subsystems:

```bash
loki cognitive <SUBCOMMAND>
```

**Subcommands:**

#### `start` - Start cognitive systems
```bash
loki cognitive start [OPTIONS]
```
- `--all` - Start all systems
- `--reasoning` - Start reasoning engines
- `--creativity` - Start creativity engine
- `--empathy` - Start empathy system
- `--consciousness` - Start consciousness stream

#### `stop` - Stop cognitive systems
```bash
loki cognitive stop [OPTIONS]
```

#### `status` - Check system status
```bash
loki cognitive status [--json]
```

#### `configure` - Configure cognitive systems
```bash
loki cognitive configure <SYSTEM> <SETTING> <VALUE>
```

**Examples:**
```bash
# Start all cognitive systems
loki cognitive start --all

# Check status
loki cognitive status

# Configure reasoning engine
loki cognitive configure reasoning max_depth 10

# Stop specific system
loki cognitive stop --creativity
```

### `memory` - Memory System Operations

Manage the hierarchical memory system:

```bash
loki memory <SUBCOMMAND>
```

**Subcommands:**

#### `stats` - Memory statistics
```bash
loki memory stats [--detailed]
```

#### `search` - Search memories
```bash
loki memory search <QUERY> [OPTIONS]
```
- `--limit <N>` - Maximum results (default: 10)
- `--type <TYPE>` - Memory type (episodic/semantic/procedural)
- `--since <DATE>` - Memories since date
- `--context <CTX>` - Context filter

#### `store` - Store a memory
```bash
loki memory store <CONTENT> [OPTIONS]
```
- `--type <TYPE>` - Memory type
- `--importance <0-1>` - Importance score
- `--tags <TAGS>` - Comma-separated tags

#### `forget` - Remove memories
```bash
loki memory forget <ID|QUERY> [OPTIONS]
```
- `--confirm` - Skip confirmation
- `--older-than <DAYS>` - Forget memories older than N days

#### `export` - Export memories
```bash
loki memory export [OPTIONS]
```
- `--format <FORMAT>` - Export format (json/csv/sqlite)
- `--output <FILE>` - Output file path
- `--filter <FILTER>` - Filter expression

#### `import` - Import memories
```bash
loki memory import <FILE> [OPTIONS]
```
- `--format <FORMAT>` - Import format
- `--merge` - Merge with existing memories

**Examples:**
```bash
# View memory statistics
loki memory stats --detailed

# Search for specific memories
loki memory search "rust programming" --limit 20

# Store important memory
loki memory store "Learned new Rust optimization technique" \
  --type semantic --importance 0.9

# Export memories to JSON
loki memory export --format json --output memories_backup.json

# Forget old memories
loki memory forget --older-than 30 --confirm
```

### `agent` - Multi-Agent Management

Deploy and manage specialized agents:

```bash
loki agent <SUBCOMMAND>
```

**Subcommands:**

#### `deploy` - Deploy an agent
```bash
loki agent deploy <AGENT_TYPE> [OPTIONS]
```
- `--name <NAME>` - Agent name
- `--config <FILE>` - Configuration file
- `--background` - Run in background

**Agent Types:**
- `code-reviewer` - Code review specialist
- `researcher` - Research and analysis
- `writer` - Content creation
- `debugger` - Debugging assistant
- `planner` - Task planning
- `custom` - Custom agent from config

#### `list` - List active agents
```bash
loki agent list [--all]
```

#### `stop` - Stop an agent
```bash
loki agent stop <AGENT_ID|NAME>
```

#### `communicate` - Send message to agent
```bash
loki agent communicate <AGENT_ID> <MESSAGE>
```

**Examples:**
```bash
# Deploy code review agent
loki agent deploy code-reviewer --name "PR-Reviewer"

# List all agents
loki agent list --all

# Communicate with agent
loki agent communicate PR-Reviewer "Review the latest commit"

# Stop agent
loki agent stop PR-Reviewer
```

### `tool` - Tool Management

Manage and execute tools:

```bash
loki tool <SUBCOMMAND>
```

**Subcommands:**

#### `list` - List available tools
```bash
loki tool list [--category <CATEGORY>]
```

#### `execute` - Execute a tool
```bash
loki tool execute <TOOL> [PARAMS]
```

#### `install` - Install a tool plugin
```bash
loki tool install <PLUGIN_PATH|URL>
```

#### `config` - Configure tool settings
```bash
loki tool config <TOOL> <SETTING> <VALUE>
```

**Examples:**
```bash
# List all tools
loki tool list

# List web tools
loki tool list --category web

# Execute GitHub tool
loki tool execute github create-pr \
  --repo "myorg/myrepo" \
  --title "Fix bug" \
  --branch "fix-bug"

# Install custom tool
loki tool install https://example.com/my-tool.wasm

# Configure tool
loki tool config github rate_limit 5000
```

### `setup` - Initial Setup

Configure Loki for first use:

```bash
loki setup [OPTIONS]
```

**Options:**
- `--interactive` - Interactive setup wizard
- `--minimal` - Minimal setup
- `--import <FILE>` - Import configuration

**Examples:**
```bash
# Interactive setup
loki setup --interactive

# Minimal setup
loki setup --minimal

# Import existing config
loki setup --import old_config.yaml
```

### `check-apis` - Verify API Connections

Check all configured API connections:

```bash
loki check-apis [OPTIONS]
```

**Options:**
- `--provider <PROVIDER>` - Check specific provider
- `--verbose` - Show detailed results

**Examples:**
```bash
# Check all APIs
loki check-apis

# Check specific provider
loki check-apis --provider openai --verbose
```

## Advanced Commands

### `story` - Story-Driven Operations

Manage story-driven autonomy:

```bash
loki story <SUBCOMMAND>
```

**Subcommands:**
- `create` - Create a new story
- `continue` - Continue existing story
- `list` - List active stories
- `export` - Export story

**Examples:**
```bash
# Create new story
loki story create "Refactor authentication system"

# Continue story
loki story continue story-123

# Export story
loki story export story-123 --format markdown
```

### `safety` - Safety System

Manage safety constraints:

```bash
loki safety <SUBCOMMAND>
```

**Subcommands:**
- `status` - Check safety status
- `configure` - Configure safety settings
- `audit` - View audit log
- `test` - Run safety tests

**Examples:**
```bash
# Check safety status
loki safety status

# Configure limits
loki safety configure max_tokens 4000

# View audit log
loki safety audit --last 100

# Run safety tests
loki safety test --comprehensive
```

### `plugin` - Plugin Management

Manage Loki plugins:

```bash
loki plugin <SUBCOMMAND>
```

**Subcommands:**
- `list` - List installed plugins
- `install` - Install a plugin
- `remove` - Remove a plugin
- `update` - Update plugins
- `create` - Create new plugin

**Examples:**
```bash
# List plugins
loki plugin list

# Install plugin
loki plugin install loki-plugin-aws

# Update all plugins
loki plugin update --all

# Create plugin template
loki plugin create my-plugin --template rust
```

### `x` - X/Twitter Integration

Manage X/Twitter integration:

```bash
loki x <SUBCOMMAND>
```

**Subcommands:**
- `auth` - Authenticate with X
- `post` - Create a post
- `monitor` - Monitor timeline
- `engage` - Auto-engage mode

**Examples:**
```bash
# Authenticate
loki x auth

# Post tweet
loki x post "Hello from Loki AI!"

# Monitor and engage
loki x engage --filter "AI" --reply-rate 0.1
```

## Development Commands

### `build` - Build from Source

Build Loki from source:

```bash
loki build [OPTIONS]
```

**Options:**
- `--release` - Build in release mode
- `--features <FEATURES>` - Enable features
- `--target <TARGET>` - Build target

### `test` - Run Tests

Run test suite:

```bash
loki test [OPTIONS]
```

**Options:**
- `--filter <PATTERN>` - Test filter
- `--verbose` - Verbose output
- `--coverage` - Generate coverage report

### `bench` - Run Benchmarks

Run performance benchmarks:

```bash
loki bench [OPTIONS]
```

**Options:**
- `--filter <PATTERN>` - Benchmark filter
- `--baseline <NAME>` - Compare with baseline

## System Commands

### `daemon` - Daemon Mode

Run Loki as a daemon:

```bash
loki daemon <SUBCOMMAND>
```

**Subcommands:**
- `start` - Start daemon
- `stop` - Stop daemon
- `restart` - Restart daemon
- `status` - Check daemon status

**Examples:**
```bash
# Start daemon
loki daemon start --port 8080

# Check status
loki daemon status

# Stop daemon
loki daemon stop
```

### `diagnostic` - System Diagnostics

Run system diagnostics:

```bash
loki diagnostic [OPTIONS]
```

**Options:**
- `--full` - Full diagnostic scan
- `--fix` - Attempt to fix issues

### `backup` - Backup System

Create system backup:

```bash
loki backup <OUTPUT_PATH> [OPTIONS]
```

**Options:**
- `--include-memory` - Include memory data
- `--include-plugins` - Include plugins
- `--compress` - Compress backup

### `restore` - Restore System

Restore from backup:

```bash
loki restore <BACKUP_PATH> [OPTIONS]
```

**Options:**
- `--memory-only` - Restore only memory
- `--config-only` - Restore only configuration

## Configuration

### Configuration File

Default location: `~/.loki/config.yaml`

```yaml
# Example configuration
version: 1.0

llm:
  default_provider: openai
  default_model: gpt-4
  
cognitive:
  reasoning:
    enabled: true
    max_depth: 10
  consciousness:
    enabled: true
    stream_interval: 100ms
    
memory:
  cache_size: 1GB
  persistence: true
  path: ~/.loki/memory
  
tools:
  parallel_execution: 5
  timeout: 30s
  
safety:
  max_tokens: 4000
  rate_limit: 100
  audit: true
```

### Environment Variables

```bash
# API Keys
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...

# System Settings
export LOKI_HOME=~/.loki
export LOKI_LOG_LEVEL=info
export LOKI_CONFIG=~/.loki/config.yaml

# Performance
export LOKI_MAX_WORKERS=8
export LOKI_CACHE_SIZE=2GB

# Features
export LOKI_COGNITIVE_ENABLED=true
export LOKI_TOOLS_ENABLED=true
```

## Output Formats

### JSON Output

Use `--json` flag for JSON output:

```bash
loki cognitive status --json
```

```json
{
  "status": "active",
  "systems": {
    "reasoning": "running",
    "consciousness": "running",
    "creativity": "stopped",
    "empathy": "running"
  },
  "uptime": 3600,
  "memory_usage": "512MB"
}
```

### Quiet Mode

Use `-q` for minimal output:

```bash
loki memory search "test" -q
# Returns only IDs
mem_123
mem_456
mem_789
```

### Verbose Mode

Use `-v` for detailed output:

```bash
loki check-apis -v
# Detailed connection information
Checking OpenAI API...
  Endpoint: https://api.openai.com/v1
  Response: 200 OK
  Latency: 245ms
  Models available: 15
  Rate limit: 10000/hour
  ✓ Connection successful
```

## Exit Codes

| Code | Description |
|------|-------------|
| 0 | Success |
| 1 | General error |
| 2 | Misuse of command |
| 3 | Configuration error |
| 4 | API connection error |
| 5 | Permission denied |
| 6 | Resource not found |
| 7 | Operation cancelled |
| 8 | Timeout |
| 9 | Invalid input |
| 10 | System error |

## Examples

### Complete Workflow Example

```bash
# Initial setup
loki setup --interactive

# Check connections
loki check-apis

# Start cognitive systems
loki cognitive start --all

# Launch TUI for interaction
loki tui

# Or use direct chat
loki chat --cognitive --tools "Help me analyze my codebase"

# Deploy specialized agent
loki agent deploy code-reviewer --name reviewer

# Search memories
loki memory search "important decisions" --limit 5

# Backup system
loki backup ~/loki-backup.tar.gz --compress

# Check system status
loki diagnostic --full
```

### Scripting Example

```bash
#!/bin/bash
# Automated daily tasks

# Start Loki daemon
loki daemon start

# Run code review on new PRs
loki agent deploy code-reviewer --background
loki tool execute github review-prs --repo "myorg/myrepo"

# Generate daily summary
SUMMARY=$(loki chat "Summarize today's completed tasks" --json | jq -r '.response')

# Post to Slack
loki tool execute slack post \
  --channel "#daily-updates" \
  --message "$SUMMARY"

# Backup memories
loki memory export --format json --output "memories_$(date +%Y%m%d).json"

# Stop daemon
loki daemon stop
```

## Troubleshooting

### Common Issues

**Command not found:**
```bash
# Ensure loki is in PATH
export PATH=$PATH:/usr/local/bin

# Or use full path
/usr/local/bin/loki --version
```

**API connection failed:**
```bash
# Check API keys
loki check-apis --verbose

# Verify environment variables
env | grep LOKI
```

**Permission denied:**
```bash
# Check file permissions
ls -la ~/.loki

# Fix permissions
chmod 755 ~/.loki
chmod 600 ~/.loki/config.yaml
```

**Out of memory:**
```bash
# Reduce cache size
export LOKI_CACHE_SIZE=512MB

# Limit workers
export LOKI_MAX_WORKERS=4
```

## Getting Help

```bash
# General help
loki --help

# Command-specific help
loki cognitive --help
loki memory search --help

# Version information
loki --version

# Full diagnostic
loki diagnostic --full
```

---

Next: [TUI Interface](tui_interface.md) | [Plugin API](plugin_api.md)