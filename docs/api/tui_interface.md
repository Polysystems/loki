# 🖥️ TUI Interface Documentation

## Overview

Loki's Terminal User Interface (TUI) provides a rich, interactive experience in the terminal with multiple tabs, real-time updates, and comprehensive system control.

## Launching the TUI

```bash
# Default launch
loki tui

# With options
loki tui --theme dark --layout wide
```

## Interface Layout

```
┌─────────────────────────────────────────────────────────┐
│  Loki AI v0.2.0  │ Chat │ Memory │ Tools │ Agents │ ⚙️  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│                    Main Content Area                    │
│                                                         │
│                                                         │
├─────────────────────────────────────────────────────────┤
│ Status: Ready | Memory: 512MB | CPU: 12% | Agents: 2   │
└─────────────────────────────────────────────────────────┘
```

## Tabs

### Chat Tab
Primary interaction interface:
- Message input/output
- Context display
- Cognitive features toggle
- Conversation history

### Memory Tab
Memory system management:
- Memory statistics
- Search interface
- Memory browser
- Consolidation controls

### Tools Tab
Tool management:
- Available tools list
- Tool execution
- Configuration
- Results display

### Agents Tab
Multi-agent control:
- Agent deployment
- Status monitoring
- Communication
- Task assignment

### Settings Tab (⚙️)
System configuration:
- API keys
- Model selection
- Feature toggles
- Performance settings

## Navigation

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Tab` | Next tab |
| `Shift+Tab` | Previous tab |
| `↑/↓` | Navigate items |
| `Enter` | Select/Activate |
| `Esc` | Cancel/Back |
| `Ctrl+C` | Exit |
| `/` | Command mode |
| `?` | Help |
| `F1-F5` | Jump to tab |

### Commands

Type `/` to enter command mode:

- `/help` - Show help
- `/clear` - Clear screen
- `/cognitive` - Toggle cognitive features
- `/memory` - Memory commands
- `/agent` - Agent commands
- `/tool` - Tool commands
- `/exit` - Exit TUI

## Features

### Real-time Updates
- Live message streaming
- Progress indicators
- Status updates
- Metric displays

### Multi-pane Views
- Split screens
- Resizable panes
- Focus switching
- Synchronized scrolling

### Rich Text
- Syntax highlighting
- Markdown rendering
- Color coding
- Unicode support

## Chat Interface

### Input Modes

**Normal Mode**: Standard text input
```
> Hello, can you help me with Python?
```

**Multiline Mode**: `Ctrl+M` to toggle
```
>>> def complex_function():
...     # Multiple lines
...     return result
```

**Command Mode**: Start with `/`
```
> /cognitive enable
```

### Message Display
- User messages (right-aligned)
- AI responses (left-aligned)
- System messages (centered)
- Timestamps
- Confidence indicators

## Memory Interface

### Memory Browser
```
┌─ Memory Browser ──────────────────────┐
│ ▼ Long-term (1,234 items)            │
│   ▼ Semantic (567)                   │
│     • Programming concepts           │
│     • Rust patterns                  │
│   ▼ Episodic (432)                  │
│     • Recent conversations           │
│   ▼ Procedural (235)                │
│     • Learned procedures             │
└───────────────────────────────────────┘
```

### Search Interface
```
Search: [rust async patterns    ] 🔍
Results: 23 memories found
```

## Tool Interface

### Tool List
```
┌─ Available Tools ─────────────────────┐
│ ✓ GitHub         [Active]            │
│ ✓ Web Search     [Active]            │
│ ○ Database       [Inactive]          │
│ ✓ File System    [Active]            │
└───────────────────────────────────────┘
```

### Tool Execution
```
Tool: GitHub
Action: Create PR
Parameters:
  Repository: [owner/repo        ]
  Title: [Fix async bug          ]
  Branch: [fix-async             ]
[Execute]
```

## Agent Interface

### Agent Status
```
┌─ Active Agents ───────────────────────┐
│ 📊 Researcher    | Idle    | GPT-4    │
│ 💻 Developer     | Working | DeepSeek │
│ ✍️ Writer        | Idle    | Claude   │
└───────────────────────────────────────┘
```

### Task Assignment
```
Task: Research WebRTC implementation
Assign to: [Researcher ▼]
Priority: [High ▼]
[Deploy]
```

## Customization

### Themes

```yaml
# ~/.loki/tui_theme.yaml
theme:
  name: custom
  colors:
    background: "#1e1e1e"
    foreground: "#d4d4d4"
    primary: "#007acc"
    success: "#4ec9b0"
    warning: "#ce9178"
    error: "#f48771"
```

### Layouts

```yaml
# ~/.loki/tui_layout.yaml
layout:
  tabs:
    - chat
    - memory
    - tools
    - agents
  default_tab: chat
  pane_sizes:
    sidebar: 30%
    main: 70%
```

## Configuration

```yaml
tui:
  theme: dark
  layout: default
  refresh_rate: 60
  
  chat:
    history_size: 1000
    show_timestamps: true
    show_confidence: true
    
  memory:
    auto_refresh: true
    search_limit: 50
    
  tools:
    show_inactive: false
    auto_execute: false
    
  agents:
    show_metrics: true
    update_interval: 1s
```

## Advanced Features

### Split Views
- `Ctrl+S` - Split horizontally
- `Ctrl+V` - Split vertically
- `Ctrl+W` - Close pane
- `Ctrl+Arrow` - Navigate panes

### Session Management
- Sessions auto-save
- Resume with `loki tui --resume`
- Multiple sessions supported
- Session history

### Scripting
```bash
# Send commands to TUI
echo "/cognitive enable" | loki tui --batch

# Script multiple commands
cat commands.txt | loki tui --batch
```

## Troubleshooting

### Display Issues
```bash
# Reset terminal
reset

# Force redraw
loki tui --force-redraw

# Use compatibility mode
loki tui --compat
```

### Performance
```bash
# Reduce refresh rate
loki tui --refresh-rate 30

# Disable animations
loki tui --no-animations
```

## Best Practices

1. **Learn shortcuts**: Faster navigation
2. **Use commands**: Powerful operations
3. **Customize layout**: Optimize for workflow
4. **Monitor metrics**: Track performance
5. **Save sessions**: Preserve work

---

Next: [CLI Reference](cli_reference.md) | [Plugin API](plugin_api.md)