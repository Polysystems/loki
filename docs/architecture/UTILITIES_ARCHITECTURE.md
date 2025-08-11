# Utilities Module Architecture

## Overview
The utilities module provides a comprehensive management interface for tools, MCP servers, plugins, daemons, and system monitoring within the Loki TUI application. It follows a modular, trait-based architecture that promotes code reusability and maintainability.

## Directory Structure
```
src/tui/utilities/
├── mod.rs                 # Main module coordinator
├── bridges/              # Backend system integration
│   └── mod.rs           # Tool, MCP, Plugin, Daemon bridges
├── components/           # Reusable UI components
│   ├── mod.rs           # Common UI helpers
│   └── command_palette.rs # Command palette widget
├── config.rs            # Configuration management
├── handlers/            # Input and command handling
│   └── mod.rs          # Keyboard, command, search handlers
├── integration.rs       # Subtab manager
├── rendering/           # Rendering utilities
│   └── mod.rs
├── state/              # State management
│   └── mod.rs
├── subtabs/            # Individual tab implementations
│   ├── mod.rs
│   ├── tools_tab.rs    # Tools management
│   ├── mcp_tab.rs      # MCP server management
│   ├── plugins_tab.rs  # Plugin management
│   ├── daemon_tab.rs   # Daemon control
│   └── monitoring_tab.rs # System monitoring
└── types.rs            # Shared type definitions
```

## Core Components

### 1. ModularUtilities (`mod.rs`)
The main coordinator that manages all utilities functionality:
- **State Management**: Arc<RwLock<UtilitiesState>> for thread-safe shared state
- **Backend Connections**: Optional connections to tool, MCP, plugin, and daemon managers
- **Configuration**: Integrated ConfigManager for persistent settings
- **Metrics Cache**: For monitoring tab compatibility

### 2. Subtab System
Each tab implements the `UtilitiesSubtabController` trait:
```rust
#[async_trait]
pub trait UtilitiesSubtabController: Send {
    fn name(&self) -> &str;
    fn render(&mut self, f: &mut Frame, area: Rect);
    async fn handle_key_event(&mut self, event: KeyEvent) -> Result<bool>;
    async fn handle_action(&mut self, action: UtilitiesAction) -> Result<()>;
    async fn refresh(&mut self) -> Result<()>;
    fn is_editing(&self) -> bool;
}
```

#### Tools Tab (`tools_tab.rs`)
- **Features**: Tool discovery, execution, configuration, metrics tracking
- **Views**: List view, detail view, configuration editor
- **Integration**: IntelligentToolManager via ToolBridge

#### MCP Tab (`mcp_tab.rs`)
- **Features**: Server discovery, connection management, marketplace
- **Views**: Server list, server details, marketplace browser, config editor
- **Integration**: McpManager via McpBridge

#### Plugins Tab (`plugins_tab.rs`)
- **Features**: Plugin installation, enable/disable, configuration, marketplace
- **Views**: Installed list, marketplace, plugin details, config editor
- **Integration**: PluginManager via PluginBridge

#### Daemon Tab (`daemon_tab.rs`)
- **Features**: Service control, log viewing, resource monitoring
- **Views**: Service list, logs viewer, system overview
- **Integration**: DaemonClient via DaemonBridge

#### Monitoring Tab (`monitoring_tab.rs`)
- **Features**: Real-time metrics, performance graphs, alerts, logs
- **Views**: Overview (gauges/sparklines), detailed metrics, logs, alerts, performance
- **Data**: CPU, memory, network, disk I/O tracking with history

### 3. Backend Bridges (`bridges/mod.rs`)
Provides abstraction layer between UI and backend systems:
- **ToolBridge**: Tool execution, configuration management
- **McpBridge**: Server connection, status monitoring
- **PluginBridge**: Plugin lifecycle management
- **DaemonBridge**: Service control, log retrieval

### 4. State Management (`state/mod.rs`)
Centralized state with caching:
```rust
pub struct UtilitiesState {
    pub cache: UtilitiesCache,
    pub view_mode: UtilitiesViewMode,
    pub selected_tab: usize,
    // View-specific state...
}
```

### 5. Input Handling (`handlers/mod.rs`)
Modular input processing:
- **KeyboardHandler**: Context-aware keyboard shortcuts
- **CommandHandler**: Action execution with history
- **SearchHandler**: Cross-tab search functionality
- **HandlerCoordinator**: Central coordination

### 6. Configuration (`config.rs`)
Persistent configuration management:
- **Storage**: JSON files in `~/.loki/utilities/`
- **Per-component**: Tool, MCP, Plugin, Daemon configs
- **Import/Export**: Configuration portability

### 7. UI Components (`components/`)
Reusable UI elements:
- **CommandPalette**: Quick command access with fuzzy search
- **Helper Functions**: Status indicators, progress gauges, info panels
- **Formatters**: Bytes, duration, threshold-based coloring

## Data Flow

### 1. Initialization
```
App::new() 
  → ModularUtilities::new()
    → UtilitiesSubtabManager::new()
      → Create all tabs (Tools, MCP, Plugins, Daemon, Monitoring)
    → ConfigManager::new()
      → Load saved configurations
```

### 2. Backend Connection
```
App::connect_systems()
  → ModularUtilities::connect_systems()
    → Update backend managers
    → Initialize bridges
```

### 3. User Interaction
```
KeyEvent
  → UtilitiesSubtabManager::handle_key_event()
    → Active Tab::handle_key_event()
      → Update local state
      → Trigger actions
```

### 4. Data Refresh
```
Tab::refresh()
  → Bridge::fetch_data()
    → Backend System
  → Update UtilitiesState cache
  → Trigger re-render
```

### 5. Action Execution
```
UtilitiesAction
  → HandlerCoordinator::execute_action()
    → CommandHandler::execute()
      → Bridge::perform_action()
        → Backend System
```

## Key Features

### 1. Modular Architecture
- **Trait-based design**: Easy to add new tabs
- **Separation of concerns**: UI, state, backend logic separated
- **Reusable components**: Shared UI elements and utilities

### 2. Real-time Updates
- **Async refresh**: Non-blocking data updates
- **Caching**: Reduce backend calls
- **Demo mode**: Works without backend connections

### 3. Rich UI
- **Multiple view modes**: List, detail, edit views per tab
- **Interactive elements**: Tables, gauges, sparklines, charts
- **Keyboard navigation**: Full keyboard control

### 4. Search & Discovery
- **Cross-tab search**: Find items across all utilities
- **Command palette**: Quick action access
- **Marketplace integration**: Discover new tools/plugins/servers

### 5. Configuration Management
- **Persistent settings**: Save user preferences
- **Import/Export**: Share configurations
- **Per-component config**: Fine-grained control

## Usage Examples

### Adding a New Tab
1. Create new file in `subtabs/`
2. Implement `UtilitiesSubtabController` trait
3. Add to `UtilitiesSubtabManager::new()`
4. Create bridge if needed in `bridges/`

### Adding a New Action
1. Add variant to `UtilitiesAction` enum in `types.rs`
2. Handle in relevant tab's `handle_action()`
3. Add to command palette if user-facing

### Extending Search
1. Add search logic in `SearchHandler::search()`
2. Map results to appropriate `UtilitiesAction`
3. Results appear in search overlay

## Testing

### Unit Tests
```bash
cargo test utilities
```

### Integration Tests
```bash
cargo test --test utilities_integration
```

### Manual Testing
1. Launch TUI: `cargo run`
2. Navigate to Utilities (Tab 3)
3. Test each subtab:
   - Tab/Shift+Tab: Switch between subtabs
   - Arrow keys: Navigate lists
   - Enter: Select/execute
   - Various keys: Tab-specific actions

## Performance Considerations

### Caching Strategy
- **State cache**: Reduces backend calls
- **Refresh on demand**: User-triggered updates
- **Background refresh**: Optional periodic updates

### Memory Management
- **Bounded history**: Limited metric history (60 points)
- **Lazy loading**: Load details only when needed
- **Resource cleanup**: Proper drop implementations

### Concurrency
- **Thread-safe state**: Arc<RwLock> for shared access
- **Async operations**: Non-blocking I/O
- **Parallel refresh**: Multiple tabs can refresh simultaneously

## Future Enhancements

### Planned Features
- [ ] Advanced filtering and sorting
- [ ] Batch operations (multi-select)
- [ ] Export metrics/logs
- [ ] Custom dashboard layouts
- [ ] Plugin development kit
- [ ] Automated workflows
- [ ] Remote daemon management
- [ ] Performance profiling tools

### Architecture Improvements
- [ ] Event-driven updates (pub/sub)
- [ ] Incremental rendering
- [ ] Virtual scrolling for large lists
- [ ] Undo/redo system
- [ ] Macro recording
- [ ] Theme customization

## Troubleshooting

### Common Issues

#### 1. Backend Connection Failed
- **Symptom**: Demo data shown instead of real data
- **Solution**: Check backend service is running, verify connection settings

#### 2. Configuration Not Saving
- **Symptom**: Settings lost on restart
- **Solution**: Check write permissions for `~/.loki/utilities/`

#### 3. Slow Performance
- **Symptom**: UI lag when switching tabs
- **Solution**: Reduce refresh frequency, check backend response times

#### 4. Missing Features
- **Symptom**: Some actions don't work
- **Solution**: Check if backend service supports the feature

## Contributing

### Guidelines
1. Follow existing patterns and conventions
2. Add tests for new functionality
3. Update documentation
4. Ensure backward compatibility
5. Consider performance impact

### Code Style
- Use rustfmt for formatting
- Follow Rust API guidelines
- Add descriptive comments
- Use meaningful variable names
- Handle errors properly (avoid unwrap)

## License
Part of the Loki project. See main LICENSE file.