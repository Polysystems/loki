use ratatui::Frame;
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::prelude::{Color, Line, Modifier, Span, Style};
use ratatui::widgets::{Block, Borders, List, ListItem, Paragraph, Wrap};

use crate::tui::App;

/// Configuration for a single keybinding
#[derive(Debug, Clone)]
pub struct KeyBinding {
    pub action: String,
    pub key: String,
    pub modifiers: Vec<String>,
    pub description: String,
    pub category: KeyBindingCategory,
}

#[derive(Debug, Clone, PartialEq)]
pub enum KeyBindingCategory {
    Navigation,
    Editing,
    Commands,
    Panels,
    System,
}

impl KeyBindingCategory {
    fn display_name(&self) -> &str {
        match self {
            Self::Navigation => "Navigation",
            Self::Editing => "Editing",
            Self::Commands => "Commands",
            Self::Panels => "Panel Controls",
            Self::System => "System",
        }
    }
    
    fn color(&self) -> Color {
        match self {
            Self::Navigation => Color::Cyan,
            Self::Editing => Color::Green,
            Self::Commands => Color::Yellow,
            Self::Panels => Color::Magenta,
            Self::System => Color::Red,
        }
    }
}

/// Draw the keybindings configuration interface
pub fn draw_keybindings(f: &mut Frame, app: &App, area: Rect) {
    // Create main layout
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);
    
    // Draw left panel - Current keybindings
    draw_current_keybindings(f, app, chunks[0]);
    
    // Draw right panel - Available shortcuts and customization
    draw_keybinding_customization(f, app, chunks[1]);
}

fn draw_current_keybindings(f: &mut Frame, _app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(0)])
        .split(area);
    
    // Header
    let header = Paragraph::new("⌨️ Current Keybindings")
        .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
        .block(Block::default().borders(Borders::ALL))
        .alignment(Alignment::Center);
    f.render_widget(header, chunks[0]);
    
    // Get all keybindings organized by category
    let keybindings = get_default_keybindings();
    
    // Create sections for each category
    let mut lines = Vec::new();
    
    for category in [
        KeyBindingCategory::Navigation,
        KeyBindingCategory::Editing,
        KeyBindingCategory::Commands,
        KeyBindingCategory::Panels,
        KeyBindingCategory::System,
    ] {
        // Category header
        lines.push(Line::from(vec![
            Span::styled(
                format!("━━━ {} ━━━", category.display_name()),
                Style::default().fg(category.color()).add_modifier(Modifier::BOLD)
            ),
        ]));
        lines.push(Line::from(""));
        
        // Filter bindings for this category
        let category_bindings: Vec<_> = keybindings.iter()
            .filter(|kb| kb.category == category)
            .collect();
        
        for binding in category_bindings {
            // Format the key combination
            let mut key_str = String::new();
            for modifier in &binding.modifiers {
                key_str.push_str(modifier);
                key_str.push('+');
            }
            key_str.push_str(&binding.key);
            
            lines.push(Line::from(vec![
                Span::styled(
                    format!("  {:<15}", key_str),
                    Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
                ),
                Span::raw(" → "),
                Span::styled(
                    &binding.description,
                    Style::default().fg(Color::White)
                ),
            ]));
        }
        
        lines.push(Line::from(""));
    }
    
    let keybindings_widget = Paragraph::new(lines)
        .block(Block::default().borders(Borders::ALL))
        .wrap(Wrap { trim: true });
    
    f.render_widget(keybindings_widget, chunks[1]);
}

fn draw_keybinding_customization(f: &mut Frame, _app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // Header
            Constraint::Min(10),    // Content
            Constraint::Length(4),  // Instructions
        ])
        .split(area);
    
    // Header
    let header = Paragraph::new("🎨 Customize Keybindings")
        .style(Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD))
        .block(Block::default().borders(Borders::ALL))
        .alignment(Alignment::Center);
    f.render_widget(header, chunks[0]);
    
    // Content - Show customization options
    let mut items = vec![
        ListItem::new("📝 Edit keybinding configuration"),
        ListItem::new("💾 Export current keybindings"),
        ListItem::new("📥 Import keybinding preset"),
        ListItem::new("🔄 Reset to defaults"),
        ListItem::new(""),
        ListItem::new("Preset Configurations:"),
        ListItem::new("  • Vim-style navigation"),
        ListItem::new("  • Emacs-style editing"),
        ListItem::new("  • VS Code compatibility"),
        ListItem::new("  • Minimal (essential keys only)"),
        ListItem::new(""),
        ListItem::new("Recently Used:"),
    ];
    
    // Add recently used keybindings if available
    if let Some(recent) = get_recent_keybindings() {
        for action in recent {
            items.push(ListItem::new(format!("  • {}", action)));
        }
    } else {
        items.push(ListItem::new("  • No recent actions"));
    }
    
    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL))
        .style(Style::default().fg(Color::White))
        .highlight_style(Style::default().add_modifier(Modifier::BOLD));
    
    f.render_widget(list, chunks[1]);
    
    // Instructions footer
    let instructions = vec![
        Line::from(""),
        Line::from(vec![
            Span::styled("[Enter]", Style::default().fg(Color::Yellow)),
            Span::raw(" Select/Edit  "),
            Span::styled("[R]", Style::default().fg(Color::Yellow)),
            Span::raw(" Reset  "),
            Span::styled("[E]", Style::default().fg(Color::Yellow)),
            Span::raw(" Export  "),
            Span::styled("[I]", Style::default().fg(Color::Yellow)),
            Span::raw(" Import"),
        ]),
        Line::from(vec![
            Span::styled("[↑↓]", Style::default().fg(Color::Yellow)),
            Span::raw(" Navigate  "),
            Span::styled("[Tab]", Style::default().fg(Color::Yellow)),
            Span::raw(" Switch panels  "),
            Span::styled("[Esc]", Style::default().fg(Color::Yellow)),
            Span::raw(" Back"),
        ]),
    ];
    
    let footer = Paragraph::new(instructions)
        .block(Block::default().borders(Borders::ALL))
        .alignment(Alignment::Center);
    
    f.render_widget(footer, chunks[2]);
}

/// Get the default keybindings configuration
fn get_default_keybindings() -> Vec<KeyBinding> {
    vec![
        // Navigation
        KeyBinding {
            action: "next_tab".to_string(),
            key: "Tab".to_string(),
            modifiers: vec![],
            description: "Next tab".to_string(),
            category: KeyBindingCategory::Navigation,
        },
        KeyBinding {
            action: "prev_tab".to_string(),
            key: "Tab".to_string(),
            modifiers: vec!["Shift".to_string()],
            description: "Previous tab".to_string(),
            category: KeyBindingCategory::Navigation,
        },
        KeyBinding {
            action: "go_to_home".to_string(),
            key: "1".to_string(),
            modifiers: vec![],
            description: "Go to Home tab".to_string(),
            category: KeyBindingCategory::Navigation,
        },
        KeyBinding {
            action: "go_to_chat".to_string(),
            key: "2".to_string(),
            modifiers: vec![],
            description: "Go to Chat tab".to_string(),
            category: KeyBindingCategory::Navigation,
        },
        KeyBinding {
            action: "go_to_utilities".to_string(),
            key: "3".to_string(),
            modifiers: vec![],
            description: "Go to Utilities tab".to_string(),
            category: KeyBindingCategory::Navigation,
        },
        KeyBinding {
            action: "go_to_memory".to_string(),
            key: "4".to_string(),
            modifiers: vec![],
            description: "Go to Memory tab".to_string(),
            category: KeyBindingCategory::Navigation,
        },
        KeyBinding {
            action: "go_to_cognitive".to_string(),
            key: "5".to_string(),
            modifiers: vec![],
            description: "Go to Cognitive tab".to_string(),
            category: KeyBindingCategory::Navigation,
        },
        KeyBinding {
            action: "go_to_social".to_string(),
            key: "6".to_string(),
            modifiers: vec![],
            description: "Go to Social tab".to_string(),
            category: KeyBindingCategory::Navigation,
        },
        KeyBinding {
            action: "go_to_settings".to_string(),
            key: "7".to_string(),
            modifiers: vec![],
            description: "Go to Settings tab".to_string(),
            category: KeyBindingCategory::Navigation,
        },
        KeyBinding {
            action: "next_subtab".to_string(),
            key: "J".to_string(),
            modifiers: vec!["Ctrl".to_string()],
            description: "Next sub-tab".to_string(),
            category: KeyBindingCategory::Navigation,
        },
        KeyBinding {
            action: "prev_subtab".to_string(),
            key: "K".to_string(),
            modifiers: vec!["Ctrl".to_string()],
            description: "Previous sub-tab".to_string(),
            category: KeyBindingCategory::Navigation,
        },
        
        // Editing
        KeyBinding {
            action: "copy".to_string(),
            key: "C".to_string(),
            modifiers: vec!["Ctrl".to_string()],
            description: "Copy selected text".to_string(),
            category: KeyBindingCategory::Editing,
        },
        KeyBinding {
            action: "paste".to_string(),
            key: "V".to_string(),
            modifiers: vec!["Ctrl".to_string()],
            description: "Paste from clipboard".to_string(),
            category: KeyBindingCategory::Editing,
        },
        KeyBinding {
            action: "undo".to_string(),
            key: "Z".to_string(),
            modifiers: vec!["Ctrl".to_string()],
            description: "Undo last action".to_string(),
            category: KeyBindingCategory::Editing,
        },
        KeyBinding {
            action: "clear_input".to_string(),
            key: "U".to_string(),
            modifiers: vec!["Ctrl".to_string()],
            description: "Clear input field".to_string(),
            category: KeyBindingCategory::Editing,
        },
        
        // Commands
        KeyBinding {
            action: "execute_command".to_string(),
            key: "Enter".to_string(),
            modifiers: vec![],
            description: "Execute command/send message".to_string(),
            category: KeyBindingCategory::Commands,
        },
        KeyBinding {
            action: "command_history_up".to_string(),
            key: "↑".to_string(),
            modifiers: vec![],
            description: "Previous command in history".to_string(),
            category: KeyBindingCategory::Commands,
        },
        KeyBinding {
            action: "command_history_down".to_string(),
            key: "↓".to_string(),
            modifiers: vec![],
            description: "Next command in history".to_string(),
            category: KeyBindingCategory::Commands,
        },
        KeyBinding {
            action: "autocomplete".to_string(),
            key: "Tab".to_string(),
            modifiers: vec![],
            description: "Autocomplete suggestion".to_string(),
            category: KeyBindingCategory::Commands,
        },
        
        // Panel Controls
        KeyBinding {
            action: "toggle_context_panel".to_string(),
            key: "F2".to_string(),
            modifiers: vec![],
            description: "Toggle Context/Reasoning panel".to_string(),
            category: KeyBindingCategory::Panels,
        },
        KeyBinding {
            action: "toggle_tool_panel".to_string(),
            key: "F3".to_string(),
            modifiers: vec![],
            description: "Toggle Tool Output panel".to_string(),
            category: KeyBindingCategory::Panels,
        },
        KeyBinding {
            action: "toggle_workflow_panel".to_string(),
            key: "F4".to_string(),
            modifiers: vec![],
            description: "Toggle Workflow panel".to_string(),
            category: KeyBindingCategory::Panels,
        },
        KeyBinding {
            action: "toggle_preview_panel".to_string(),
            key: "F5".to_string(),
            modifiers: vec![],
            description: "Toggle Preview panel".to_string(),
            category: KeyBindingCategory::Panels,
        },
        KeyBinding {
            action: "toggle_story_panel".to_string(),
            key: "S".to_string(),
            modifiers: vec!["Ctrl".to_string()],
            description: "Toggle Story visualization".to_string(),
            category: KeyBindingCategory::Panels,
        },
        KeyBinding {
            action: "cycle_panel_focus".to_string(),
            key: "Tab".to_string(),
            modifiers: vec!["Ctrl".to_string()],
            description: "Cycle panel focus".to_string(),
            category: KeyBindingCategory::Panels,
        },
        
        // System
        KeyBinding {
            action: "show_help".to_string(),
            key: "F1".to_string(),
            modifiers: vec![],
            description: "Show help overlay".to_string(),
            category: KeyBindingCategory::System,
        },
        KeyBinding {
            action: "quit".to_string(),
            key: "Q".to_string(),
            modifiers: vec!["Ctrl".to_string()],
            description: "Quit application".to_string(),
            category: KeyBindingCategory::System,
        },
        KeyBinding {
            action: "refresh".to_string(),
            key: "R".to_string(),
            modifiers: vec!["Ctrl".to_string()],
            description: "Refresh current view".to_string(),
            category: KeyBindingCategory::System,
        },
        KeyBinding {
            action: "save_state".to_string(),
            key: "S".to_string(),
            modifiers: vec!["Ctrl".to_string(), "Shift".to_string()],
            description: "Save application state".to_string(),
            category: KeyBindingCategory::System,
        },
    ]
}

/// Get recently used keybindings (placeholder - would connect to actual tracking)
fn get_recent_keybindings() -> Option<Vec<String>> {
    Some(vec![
        "Toggle Context Panel (F2)".to_string(),
        "Send Message (Enter)".to_string(),
        "Next Tab (Tab)".to_string(),
    ])
}