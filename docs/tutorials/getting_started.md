# 🚀 Getting Started with Loki AI

Welcome to Loki AI! This guide will help you get up and running with the most advanced open-source cognitive AI system available. In just a few minutes, you'll have Loki running on your machine and ready to assist with complex tasks.

## Prerequisites

Before installing Loki, ensure you have:

- **Operating System**: macOS, Linux, or Windows 10+
- **RAM**: Minimum 8GB (16GB recommended)
- **Storage**: 20GB free space (50GB recommended)
- **Internet**: For API access and model downloads

## Installation

### Method 1: Quick Install (Recommended)

The fastest way to get started:

```bash
# macOS/Linux
curl -sSL https://raw.githubusercontent.com/polysystems/loki/main/install.sh | bash

# Or using wget
wget -qO- https://raw.githubusercontent.com/polysystems/loki/main/install.sh | bash
```

This script will:
1. Download the appropriate binary for your platform
2. Install it to `/usr/local/bin`
3. Set up initial configuration
4. Verify the installation

### Method 2: Download Binary

Download the pre-built binary for your platform:

| Platform | Download Link |
|----------|--------------|
| macOS (Apple Silicon) | [loki-macos-arm64](https://github.com/polysystems/loki/releases/latest/download/loki-macos-arm64) |
| macOS (Intel) | [loki-macos-x64](https://github.com/polysystems/loki/releases/latest/download/loki-macos-x64) |
| Linux (x64) | [loki-linux-x64](https://github.com/polysystems/loki/releases/latest/download/loki-linux-x64) |
| Windows | [loki-windows-x64.exe](https://github.com/polysystems/loki/releases/latest/download/loki-windows-x64.exe) |

After downloading:

```bash
# macOS/Linux
chmod +x loki-*
sudo mv loki-* /usr/local/bin/loki

# Windows (PowerShell as Administrator)
Move-Item loki-windows-x64.exe C:\Windows\System32\loki.exe
```

### Method 3: Build from Source

For developers who want the latest features:

```bash
# Clone the repository
git clone https://github.com/polysystems/loki.git
cd loki

# Install Rust if you haven't already
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Build with all features
cargo build --release --features all

# Install
sudo cp target/release/loki /usr/local/bin/
```

## Configuration

### Step 1: Generate Configuration

Run the setup command to create your configuration:

```bash
loki setup
```

This interactive setup will:
1. Create a `.env` file with required settings
2. Guide you through API key configuration
3. Set up data directories
4. Configure initial preferences

### Step 2: API Keys

You'll need at least one LLM provider API key. Loki supports:

- **OpenAI** - [Get API Key](https://platform.openai.com/api-keys)
- **Anthropic** - [Get API Key](https://console.anthropic.com/account/keys)
- **Google Gemini** - [Get API Key](https://makersuite.google.com/app/apikey)
- **Mistral** - [Get API Key](https://console.mistral.ai/api-keys)
- **Local Models** - Via Ollama (no API key needed)

Add your keys to the `.env` file:

```bash
# Edit .env file
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...
```

### Step 3: Verify Installation

Check that everything is working:

```bash
# Check version
loki --version

# Verify API connections
loki check-apis

# Run system diagnostics
loki diagnostic
```

## First Run

### Launch the Terminal UI

The Terminal User Interface (TUI) is the primary way to interact with Loki:

```bash
loki tui
```

You'll see a rich terminal interface with multiple tabs:
- **Chat**: Interactive conversation with cognitive features
- **Memory**: View and manage the memory system
- **Tools**: Available tool integrations
- **Agents**: Multi-agent coordination
- **Monitoring**: System performance metrics

### Navigation

- **Tab Navigation**: Use `Tab` or `Shift+Tab` to switch tabs
- **Selection**: Arrow keys to navigate, `Enter` to select
- **Commands**: Type `/` to see available commands
- **Exit**: Press `Ctrl+C` or type `/exit`

## Your First Conversation

### Basic Interaction

Start with a simple conversation:

```
You: Hello Loki! Can you introduce yourself?

Loki: Hello! I'm Loki, an advanced cognitive AI system with over 100 specialized 
modules for reasoning, creativity, and autonomous operation. I can help with 
complex tasks, code generation, analysis, and much more. How can I assist you today?
```

### Enable Cognitive Features

Activate advanced cognitive capabilities:

```
You: /cognitive enable

System: Cognitive enhancements enabled. Advanced reasoning, theory of mind, 
and consciousness simulation are now active.

You: Let's solve a complex problem together...
```

## Common Use Cases

### 1. Code Generation

```
You: Create a Python web scraper that extracts article titles from Hacker News

Loki: I'll create a Python web scraper for Hacker News. Let me design this with 
proper error handling and rate limiting...

[Loki generates complete, production-ready code with explanations]
```

### 2. Research & Analysis

```
You: Analyze the pros and cons of microservices vs monolithic architecture

Loki: I'll provide a comprehensive analysis of both architectural patterns...

[Detailed analysis with trade-offs, use cases, and recommendations]
```

### 3. Creative Tasks

```
You: Write a short story about an AI discovering consciousness

Loki: [Engages creative engine and narrative generation to produce an original story]
```

### 4. Problem Solving

```
You: I have a memory leak in my Rust application. How can I debug it?

Loki: Let me help you debug that memory leak. I'll guide you through a systematic 
approach using Rust-specific tools...

[Step-by-step debugging guide with commands and explanations]
```

## Advanced Features

### Memory System

Loki maintains a hierarchical memory system:

```bash
# View memory statistics
loki memory stats

# Search memories
loki memory search "rust optimization"

# Export memory
loki memory export --format json > memories.json
```

### Multi-Agent Tasks

Deploy specialized agents for complex tasks:

```
You: /agent deploy code-reviewer

System: Code review agent deployed. Ready to analyze code.

You: Review the changes in my last commit
```

### Tool Integration

Loki can use various tools:

```
You: Search GitHub for popular Rust web frameworks

Loki: I'll search GitHub for popular Rust web frameworks...
[Uses GitHub tool to find and analyze repositories]
```

### Story-Driven Autonomy

Enable narrative-based task execution:

```
You: /story enable

You: Help me refactor my authentication system

Loki: I'll approach this refactoring as a story with clear chapters:
1. Understanding the current system
2. Identifying improvement areas
3. Planning the refactoring
4. Implementation
5. Testing and validation

Let's begin with Chapter 1...
```

## Configuration Files

### Main Configuration

Loki uses several configuration files:

1. **`.env`** - Environment variables and API keys
2. **`config/models.yaml`** - Model configurations
3. **`config/tools.yaml`** - Tool settings
4. **`config/safety.yaml`** - Safety constraints

### Example `.env` File

```bash
# LLM Providers
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# System Settings
RUST_LOG=info
LOKI_DATA_DIR=/home/user/.loki
LOKI_CACHE_SIZE=1GB

# Features
LOKI_COGNITIVE_ENABLED=true
LOKI_MEMORY_PERSISTENT=true
LOKI_TOOLS_ENABLED=true

# Performance
LOKI_PARALLEL_TOOLS=5
LOKI_MAX_WORKERS=8
```

## Troubleshooting

### Common Issues

#### Issue: "API key not found"
**Solution**: Ensure your `.env` file contains valid API keys:
```bash
loki check-apis
```

#### Issue: "Out of memory"
**Solution**: Reduce memory usage in configuration:
```bash
LOKI_CACHE_SIZE=512MB
LOKI_MAX_WORKERS=4
```

#### Issue: "Connection timeout"
**Solution**: Check your internet connection and firewall settings.

#### Issue: "Model not available"
**Solution**: Ensure you have access to the specified model:
```bash
loki models list
```

### Getting Help

- **Documentation**: [Full Documentation](../README.md)
- **Discord**: [Join our Discord](https://discord.gg/eigencode)
- **GitHub Issues**: [Report Issues](https://github.com/polysystems/loki/issues)
- **Email Support**: thermo@polysystems.ai

## Next Steps

Now that you have Loki running, explore these areas:

### Learn More
- [Architecture Overview](../architecture/overview.md) - Understand the system design
- [Cognitive Features](../features/cognitive/consciousness_stream.md) - Deep dive into cognitive capabilities
- [Memory System](../features/memory/hierarchical_storage.md) - Learn about memory architecture
- [Tool Integration](../features/tools/tool_ecosystem.md) - Explore available tools

### Tutorials
- [First Cognitive Agent](first_cognitive_agent.md) - Create your first agent
- [Memory Integration](memory_integration.md) - Work with the memory system
- [Custom Tools](custom_tools.md) - Add your own tools
- [Story-Driven Workflows](story_driven_workflow.md) - Use narrative processing

### Advanced Topics
- [Plugin Development](../development/plugin_development.md) - Create plugins
- [Performance Tuning](../deployment/performance_tuning.md) - Optimize performance
- [Clustering](../deployment/scaling_guide.md) - Scale to multiple nodes

## Tips for Success

### Best Practices

1. **Start Simple**: Begin with basic tasks before enabling all features
2. **Use Context**: Provide clear context for better responses
3. **Leverage Memory**: Let Loki learn from your interactions
4. **Experiment**: Try different cognitive modes and features
5. **Monitor Resources**: Keep an eye on system resources

### Performance Tips

1. **Cache Warming**: First run may be slower as caches populate
2. **Model Selection**: Choose appropriate models for your tasks
3. **Parallel Tools**: Enable for faster multi-tool operations
4. **Memory Limits**: Set appropriate limits for your system

### Security Considerations

1. **API Key Security**: Never commit API keys to version control
2. **Data Privacy**: Understand what data is sent to external APIs
3. **Local Models**: Use Ollama for sensitive data
4. **Audit Logs**: Enable for compliance requirements

## Example Session

Here's a complete example session showcasing Loki's capabilities:

```bash
# Start Loki
$ loki tui

# In the TUI
You: /cognitive enable
System: Cognitive enhancements enabled

You: I need to build a REST API in Rust that handles user authentication, 
includes rate limiting, and connects to PostgreSQL. Can you help me design 
and implement this?

Loki: I'll help you build a production-ready REST API in Rust with those 
requirements. Let me break this down into components and create a complete 
implementation.

[Loki proceeds to:]
1. Design the architecture
2. Set up the project structure
3. Implement authentication with JWT
4. Add rate limiting middleware
5. Create PostgreSQL models and migrations
6. Write comprehensive tests
7. Provide deployment configuration

You: This is excellent! Can you also add OpenAPI documentation?

Loki: Absolutely! I'll integrate OpenAPI documentation using the utoipa crate...

[Continues with OpenAPI integration]

You: /memory save "Rust REST API Template"
System: Memory saved successfully

You: Thank you, this has been incredibly helpful!

Loki: You're welcome! I've saved this as a template in my memory system. 
Feel free to ask if you need any modifications or have questions about the 
implementation. The complete project is ready for production use.
```

## Conclusion

Congratulations! You're now ready to use Loki AI for your projects. Remember that Loki is continuously learning and improving through its self-modification capabilities. The more you use it, the better it becomes at understanding your needs and preferences.

Explore the features, experiment with different modes, and don't hesitate to push the boundaries of what's possible. Loki is designed to be your cognitive companion for complex tasks, creative endeavors, and continuous learning.

Welcome to the future of AI-assisted development and problem-solving!

---

**Ready for more?** Check out [First Cognitive Agent](first_cognitive_agent.md) or dive into [Advanced Features](../features/)