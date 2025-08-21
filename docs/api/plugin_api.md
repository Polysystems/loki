# 🔌 Plugin API Documentation

## Overview

Loki's plugin system allows developers to extend functionality through WebAssembly (WASM) and native plugins. Plugins can add new tools, cognitive modules, UI components, and more.

## Plugin Architecture

```
┌─────────────────────────────────────────┐
│            Plugin Manager                │
│         Registration & Lifecycle         │
└────────────────┬────────────────────────┘
                 ▼
┌─────────────────────────────────────────┐
│            Plugin Sandbox                │
│      WASM Runtime / Native Isolation    │
└────────────────┬────────────────────────┘
                 ▼
┌──────────┬──────────┬──────────┬────────┐
│   Tool   │Cognitive │    UI    │ Custom │
│  Plugin  │  Plugin  │  Plugin  │ Plugin │
└──────────┴──────────┴──────────┴────────┘
```

## Creating Plugins

### Plugin Structure

```rust
// src/lib.rs
use loki_plugin_api::*;

#[derive(Plugin)]
pub struct MyPlugin {
    name: String,
    version: String,
}

#[plugin_main]
impl LokiPlugin for MyPlugin {
    fn init(&mut self, config: Config) -> Result<()> {
        // Initialize plugin
        Ok(())
    }
    
    fn execute(&self, input: Input) -> Result<Output> {
        // Plugin logic
        Ok(Output::new("Result"))
    }
    
    fn shutdown(&mut self) -> Result<()> {
        // Cleanup
        Ok(())
    }
}
```

### Plugin Manifest

```toml
# plugin.toml
[plugin]
name = "my-plugin"
version = "0.1.0"
author = "Your Name"
description = "Plugin description"

[capabilities]
tools = ["web_search", "file_access"]
permissions = ["network", "filesystem"]

[dependencies]
loki-plugin-api = "0.2"
```

## Plugin Types

### Tool Plugin
```rust
#[tool_plugin]
impl ToolPlugin for MyTool {
    fn execute(&self, params: ToolParams) -> ToolResult {
        // Tool implementation
    }
    
    fn metadata(&self) -> ToolMetadata {
        ToolMetadata {
            name: "my_tool",
            description: "Custom tool",
            parameters: vec![...],
        }
    }
}
```

### Cognitive Plugin
```rust
#[cognitive_plugin]
impl CognitivePlugin for MyCognitive {
    fn process(&self, thought: Thought) -> ProcessedThought {
        // Cognitive processing
    }
}
```

### UI Plugin
```rust
#[ui_plugin]
impl UIPlugin for MyUI {
    fn render(&self, state: UIState) -> Widget {
        // UI rendering
    }
}
```

## Plugin API

### Core Interfaces
```rust
pub trait LokiPlugin: Send + Sync {
    fn init(&mut self, config: Config) -> Result<()>;
    fn execute(&self, input: Input) -> Result<Output>;
    fn shutdown(&mut self) -> Result<()>;
}

pub trait EventHandler {
    fn on_event(&self, event: Event) -> Result<()>;
}
```

### Available APIs
- **Memory API**: Access memory system
- **Tool API**: Execute tools
- **Bridge API**: Cross-component communication
- **UI API**: Create UI elements

## Building Plugins

### WASM Plugin
```bash
# Install wasm-pack
cargo install wasm-pack

# Build plugin
wasm-pack build --target web

# Output: pkg/my_plugin.wasm
```

### Native Plugin
```bash
# Build as dynamic library
cargo build --release

# Output: target/release/libmy_plugin.so
```

## Installing Plugins

### CLI Installation
```bash
# Install from file
loki plugin install ./my_plugin.wasm

# Install from URL
loki plugin install https://example.com/plugin.wasm

# Install from marketplace
loki plugin install loki-plugin-name
```

### Manual Installation
```bash
# Copy to plugins directory
cp my_plugin.wasm ~/.loki/plugins/

# Register plugin
loki plugin register my_plugin
```

## Plugin Configuration

```yaml
plugins:
  my_plugin:
    enabled: true
    config:
      api_key: "..."
      timeout: 30s
    permissions:
      - network
      - filesystem
```

## Security

### Sandboxing
- WASM plugins run in isolated sandbox
- Limited system access
- Resource quotas enforced

### Permissions
```rust
pub enum Permission {
    Network,     // Network access
    FileSystem,  // File access
    Memory,      // Memory system
    Tools,       // Tool execution
    UI,          // UI modification
}
```

## Plugin Lifecycle

1. **Discovery**: Plugin found in directory
2. **Loading**: Plugin loaded into memory
3. **Initialization**: `init()` called
4. **Registration**: Plugin registered with system
5. **Execution**: Plugin responds to events
6. **Shutdown**: `shutdown()` called

## Examples

### Simple Tool Plugin
```rust
use loki_plugin_api::*;

#[tool_plugin]
pub struct WebScraperPlugin;

impl ToolPlugin for WebScraperPlugin {
    fn execute(&self, params: ToolParams) -> ToolResult {
        let url = params.get_string("url")?;
        let content = fetch_url(&url)?;
        Ok(ToolResult::success(content))
    }
}
```

### Event Handler Plugin
```rust
#[event_handler]
impl EventHandler for MyEventPlugin {
    fn on_event(&self, event: Event) -> Result<()> {
        match event {
            Event::MessageReceived(msg) => {
                // Handle message
            },
            Event::ToolExecuted(result) => {
                // Handle tool result
            },
            _ => {}
        }
        Ok(())
    }
}
```

## Best Practices

1. **Error Handling**: Always return proper errors
2. **Resource Management**: Clean up resources
3. **Performance**: Minimize overhead
4. **Documentation**: Document plugin capabilities
5. **Testing**: Include tests with plugin

---

Next: [TUI Interface](tui_interface.md) | [CLI Reference](cli_reference.md)