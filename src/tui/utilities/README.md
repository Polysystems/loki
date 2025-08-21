# Loki Utilities Module

## Overview

The Utilities module provides a comprehensive management interface for various system components in the Loki AI system. It features a modular, tab-based architecture with five main subsystems:

1. **Tools Management** - AI tool configuration and monitoring
2. **MCP (Model Context Protocol)** - Server management and marketplace
3. **Plugins** - Plugin installation, configuration, and lifecycle management
4. **Daemon Control** - System service monitoring and control
5. **Monitoring** - Real-time system metrics and performance monitoring

## Architecture

### Module Structure

```
utilities/
├── mod.rs                 # Main module definition and public API
├── state.rs              # Shared state management
├── types.rs              # Common type definitions
├── config.rs             # Configuration management
├── metrics.rs            # System metrics collection
├── integration.rs        # Subtab manager and coordination
├── subtabs/              # Individual tab implementations
│   ├── mod.rs
│   ├── tools_tab.rs     # Tools management tab
│   ├── mcp_tab.rs       # MCP server management tab
│   ├── plugins_tab.rs   # Plugin management tab
│   ├── daemon_tab.rs    # Daemon control tab
│   └── monitoring_tab.rs # System monitoring tab
├── components/           # Reusable UI components
│   ├── mod.rs
│   ├── command_palette.rs
│   └── search_overlay.rs
├── bridges/              # Backend service integration
│   └── mod.rs           # Service bridges for tools, MCP, plugins, daemons
├── rendering/            # Rendering utilities
│   └── mod.rs
├── handlers/             # Input and event handlers
│   └── mod.rs
└── tests.rs             # Module tests

```

### Key Components

#### State Management
- **UtilitiesState**: Central state container shared across all tabs
- **CachedMetrics**: Performance metrics cache
- Thread-safe with `Arc<RwLock>` for concurrent access

#### Subtab System
- **UtilitiesSubtabController**: Trait defining tab interface
- Each tab implements async key handling and rendering
- Supports hot-swapping and dynamic updates

#### Configuration
- **ConfigManager**: Persistent configuration storage
- JSON-based configuration files
- Located in `~/.loki/utilities/config.json`

#### Backend Integration
- **Bridge Pattern**: Abstraction layer for backend services
- Supports both connected and disconnected modes
- Graceful fallback to demo data when services unavailable

## Features

### Tools Tab
- View and manage AI tools
- Real-time status monitoring
- Configuration editing with JSON editor
- Tool creation wizard
- Search and filter capabilities
- Categories: Automation, AI Generation, 3D Modeling, Analysis, Development

### MCP Tab
- Server connection management
- Marketplace for discovering new servers
- Configuration editing
- Real-time connection status
- Protocol version tracking
- Environment variable management

### Plugins Tab
- Plugin marketplace browser
- Installation/uninstallation management
- Enable/disable functionality
- Configuration editor
- Capability viewer
- Auto-update settings

### Daemon Tab
- Service status monitoring
- Start/Stop/Restart controls
- Log viewer
- Resource usage tracking
- Auto-restart policies
- Service dependency management

### Monitoring Tab
- Real-time CPU usage (overall and per-core)
- Memory and swap monitoring
- Disk I/O statistics
- Network traffic monitoring
- Process management
- System uptime and load averages
- Alert notifications

## Usage

### Basic Integration

```rust
use loki::tui::utilities::ModularUtilities;

// Create utilities instance
let utilities = ModularUtilities::new();

// Connect backend services (optional)
utilities.connect_systems(
    mcp_client,
    tool_manager,
    monitoring_system,
    // ... other services
);

// Render in TUI
utilities.render(frame, area);

// Handle key events
utilities.handle_key_event(key_event).await?;
```

### Configuration Management

```rust
use loki::tui::utilities::config::{ConfigManager, ToolConfig};

// Create config manager
let config_manager = ConfigManager::new()?;

// Save tool configuration
let tool_config = ToolConfig {
    id: "my-tool".to_string(),
    enabled: true,
    settings: serde_json::json!({
        "api_key": "secret",
        "timeout": 30
    }),
};
config_manager.update_tool_config(tool_config)?;

// Load configuration
if let Some(config) = config_manager.get_tool_config("my-tool") {
    println!("Tool settings: {:?}", config.settings);
}
```

### System Metrics

```rust
use loki::tui::utilities::metrics::SystemMetrics;

let metrics = SystemMetrics::new();

// Get CPU usage
let cpu_usage = metrics.get_cpu_usage().await;
println!("CPU: {:.1}%", cpu_usage);

// Get memory usage
let (used, total) = metrics.get_memory_usage().await;
println!("Memory: {} / {}", format_bytes(used), format_bytes(total));

// Get top processes
let top_processes = metrics.get_top_processes_by_cpu(10).await;
```

## Keyboard Shortcuts

### Global
- `Tab` - Switch between tabs
- `q` - Exit/Back
- `/` - Open search
- `Esc` - Cancel/Close overlay

### Tools Tab
- `↑↓/jk` - Navigate tools
- `Enter` - Configure tool
- `c` - Create new tool
- `d` - Toggle enabled/disabled
- `r` - Refresh

### MCP Tab
- `↑↓/jk` - Navigate servers
- `Enter` - View details
- `c` - Connect to server
- `d` - Disconnect
- `i` - Install from marketplace

### Plugins Tab
- `Tab` - Switch views (Installed/Marketplace/Config)
- `i` - Install plugin
- `u` - Uninstall plugin
- `e` - Enable/Disable
- `c` - Configure

### Daemon Tab
- `1-4` - Switch view modes
- `s` - Start daemon
- `S` - Stop daemon
- `r` - Restart daemon
- `l` - View logs

### Monitoring Tab
- `1-5` - Switch metric views
- `r` - Refresh metrics
- `c` - Clear alerts
- `p` - Pause updates

## Configuration Files

### Main Configuration
`~/.loki/utilities/config.json`

```json
{
  "tools": [...],
  "mcp_servers": [...],
  "plugins": [...],
  "daemons": [...]
}
```

### Tool Configuration Example
```json
{
  "id": "computer-use",
  "enabled": true,
  "settings": {
    "api_key": "...",
    "timeout": 30,
    "retry_count": 3
  }
}
```

### MCP Server Configuration Example
```json
{
  "name": "local-mcp",
  "command": "mcp-server",
  "args": ["--port", "7890"],
  "auto_connect": true,
  "env_vars": {
    "MCP_LOG_LEVEL": "info"
  }
}
```

## Testing

Run tests with:
```bash
cargo test -p loki --lib tui::utilities::tests
```

## Performance Considerations

- **Lazy Loading**: Data is loaded on-demand to reduce memory usage
- **Caching**: Metrics are cached with configurable TTL
- **Async Operations**: All I/O operations are async to prevent blocking
- **Debouncing**: Search and refresh operations are debounced
- **Connection Pooling**: Reuses backend connections

## Error Handling

- Graceful degradation when backend services unavailable
- Fallback to cached or demo data
- User-friendly error messages
- Automatic retry with exponential backoff
- Comprehensive logging via `tracing`

## Future Enhancements

- [ ] Plugin dependency resolution
- [ ] Tool composition and chaining
- [ ] Advanced metric analytics
- [ ] Custom alert rules
- [ ] Service orchestration
- [ ] Configuration templates
- [ ] Backup and restore
- [ ] Multi-profile support

## Contributing

When adding new features:
1. Implement the `UtilitiesSubtabController` trait for new tabs
2. Add types to `types.rs`
3. Update `UtilitiesAction` enum for cross-tab communication
4. Add bridge methods for backend integration
5. Include tests in `tests.rs`
6. Update this README

## License

Part of the Loki AI system. See main LICENSE file.