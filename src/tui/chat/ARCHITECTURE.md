# Modular Chat Architecture

## Overview

The Loki TUI chat system has been completely refactored from a monolithic 14,151-line file into a clean, modular architecture. This document describes the new structure and design decisions.

## Migration Summary

- **Before**: Single `chat.rs` file with 14,151 lines
- **After**: Modular architecture with clear separation of concerns
- **Code Reduction**: 31.1% (4,405 lines removed)
- **Completion Date**: January 2025

## Core Architecture

### 1. Module Structure

```
src/tui/chat/
├── mod.rs                    # Main module definition and ModularChat struct
├── types.rs                  # Shared types (ActiveModel, etc.)
├── state/                    # State management
│   ├── mod.rs               # ChatState and related types
│   ├── persistence.rs       # Save/load functionality
│   ├── navigation.rs        # Message navigation
│   └── state_manager.rs     # State transitions
├── subtabs/                  # UI tabs
│   ├── mod.rs               # SubtabController trait
│   ├── chat_tab.rs          # Main chat interface
│   ├── history_tab.rs       # Message history browser
│   ├── models_tab.rs        # Model management
│   ├── settings_tab.rs      # Settings management
│   └── agents_tab.rs        # Agent configuration
├── integration.rs            # SubtabManager - coordinates all tabs
├── processing/               # Message processing pipeline
│   └── message_processor.rs  # Core processing logic
├── handlers/                 # Input and command handling
│   ├── commands.rs          # Command processing
│   ├── input.rs             # Keyboard input
│   └── message_handler.rs   # Message operations
├── rendering/                # UI rendering
│   ├── modular_renderer.rs  # Main render function
│   └── chat_renderer.rs     # Chat-specific rendering
└── integrations/             # External system integration
    ├── cognitive.rs         # Cognitive system integration
    └── nlp.rs              # NLP processing
```

### 2. Key Components

#### ModularChat
The main entry point replacing the old ChatManager:

```rust
pub struct ModularChat {
    pub subtab_manager: RefCell<SubtabManager>,
    pub chat_state: Arc<RwLock<ChatState>>,
    pub orchestration: Arc<RwLock<OrchestrationManager>>,
    pub agent_manager: Arc<RwLock<AgentManager>>,
    pub available_models: Vec<ActiveModel>,
    pub active_chat: usize,
}
```

#### SubtabManager
Coordinates all UI tabs and message processing:

```rust
pub struct SubtabManager {
    tabs: Vec<Box<dyn SubtabController>>,
    current_index: usize,
    message_processor: Option<MessageProcessor>,
    // ... channels and state references
}
```

#### SubtabController Trait
Interface for all tabs:

```rust
pub trait SubtabController: Send {
    fn name(&self) -> &str;
    fn title(&self) -> String;
    fn render(&mut self, f: &mut Frame, area: Rect);
    fn handle_key_event(&mut self, event: KeyEvent) -> bool;
    fn handle_char_input(&mut self, chars: Vec<char>);
    fn is_typing(&self) -> bool;
}
```

### 3. Message Processing Pipeline

1. **User Input** → SubtabManager → ChatTab
2. **Command Detection** → CommandHandler → Execution
3. **Message Processing**:
   - Tool detection and execution
   - Cognitive enhancement (if enabled)
   - NLP orchestration
   - Model orchestration fallback
4. **Response** → Chat state update → UI refresh

### 4. State Management

- **ChatState**: Core chat data (messages, settings, etc.)
- **OrchestrationManager**: Model routing and strategy
- **AgentManager**: Multi-agent configuration
- **Persistence**: Auto-save and manual save/load

### 5. Integration Points

#### Cognitive System
```rust
// In MessageProcessor
if let Some(ref cognitive_enhancement) = self.cognitive_enhancement {
    match cognitive_enhancement.process_message(...).await {
        Ok(response) => { /* Handle cognitive response */ }
        Err(e) => { /* Fallback to standard processing */ }
    }
}
```

#### Tool Execution
```rust
// In MessageProcessor
if self.looks_like_tool_request(content) {
    if let Some(tool_result) = self.try_direct_tool_execution(...).await? {
        self.response_tx.send(tool_result).await?;
        return Ok(());
    }
}
```

## Design Decisions

### 1. RefCell for SubtabManager
Used `RefCell` to allow mutable access in the TUI rendering context while maintaining a clean API.

### 2. Arc<RwLock<T>> for Shared State
Enables safe concurrent access across async boundaries and between tabs.

### 3. Message Channel Architecture
- Internal channel for message processing
- Response channel for AI responses
- Separates UI events from processing logic

### 4. Trait-based Tab System
Allows easy addition of new tabs without modifying core logic.

## Migration Path

1. Created modular structure alongside old system
2. Implemented bridge pattern for gradual migration
3. Replaced all external references
4. Removed old monolithic file

## Future Enhancements

1. **Plugin System**: Allow external tab implementations
2. **Advanced Rendering**: GPU-accelerated rendering for large chats
3. **Distributed Processing**: Multi-node chat processing
4. **Enhanced Persistence**: Database-backed chat storage

## Testing

Comprehensive test suite in `src/tui/chat/tests/`:
- Integration tests
- Cognitive system tests
- Tool execution tests
- State management tests

## Performance Improvements

- **Lazy Loading**: Messages loaded on demand
- **Efficient Rendering**: Only visible content rendered
- **Async Processing**: Non-blocking message handling
- **Memory Efficiency**: Reduced memory footprint through modular loading

## Conclusion

The modular chat architecture provides a solid foundation for future enhancements while maintaining compatibility with existing features. The clean separation of concerns makes the codebase more maintainable and extensible.