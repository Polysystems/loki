//! Keyboard shortcuts overlay system

use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Clear, List, ListItem, Paragraph, Wrap},
    Frame,
};
use std::collections::HashMap;

/// Keyboard shortcut definition
#[derive(Debug, Clone)]
pub struct KeyboardShortcut {
    /// Key combination (e.g., "Ctrl+S", "F1", "Alt+Enter")
    pub keys: String,
    /// Description of what the shortcut does
    pub description: String,
    /// Category for grouping
    pub category: ShortcutCategory,
    /// Whether this shortcut is context-sensitive
    pub context_sensitive: bool,
}

/// Categories for organizing shortcuts
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ShortcutCategory {
    General,
    Navigation,
    Chat,
    Commands,
    Editing,
    View,
    Help,
}

impl ShortcutCategory {
    fn display_name(&self) -> &'static str {
        match self {
            Self::General => "General",
            Self::Navigation => "Navigation",
            Self::Chat => "Chat",
            Self::Commands => "Commands",
            Self::Editing => "Editing",
            Self::View => "View",
            Self::Help => "Help",
        }
    }
    
    fn icon(&self) -> &'static str {
        match self {
            Self::General => "⚙️",
            Self::Navigation => "🧭",
            Self::Chat => "💬",
            Self::Commands => "⌨️",
            Self::Editing => "✏️",
            Self::View => "👁️",
            Self::Help => "❓",
        }
    }
}

/// Keyboard shortcuts overlay
pub struct KeyboardShortcutsOverlay {
    /// All registered shortcuts
    shortcuts: Vec<KeyboardShortcut>,
    /// Whether the overlay is visible
    visible: bool,
    /// Current filter/search
    filter: String,
    /// Selected category (None means all)
    selected_category: Option<ShortcutCategory>,
    /// Scroll offset
    scroll_offset: usize,
    /// Selected shortcut index
    selected_index: usize,
}

impl KeyboardShortcutsOverlay {
    /// Create a new keyboard shortcuts overlay
    pub fn new() -> Self {
        let mut overlay = Self {
            shortcuts: Vec::new(),
            visible: false,
            filter: String::new(),
            selected_category: None,
            scroll_offset: 0,
            selected_index: 0,
        };
        
        // Register default shortcuts
        overlay.register_default_shortcuts();
        overlay
    }
    
    /// Register default keyboard shortcuts
    fn register_default_shortcuts(&mut self) {
        // General shortcuts
        self.add_shortcut("Ctrl+Q", "Quit application", ShortcutCategory::General, false);
        self.add_shortcut("Ctrl+S", "Save chat", ShortcutCategory::General, false);
        self.add_shortcut("Ctrl+O", "Open/Load chat", ShortcutCategory::General, false);
        self.add_shortcut("F1", "Show this help", ShortcutCategory::General, false);
        self.add_shortcut("Esc", "Close dialog/Cancel", ShortcutCategory::General, true);
        
        // Navigation shortcuts
        self.add_shortcut("Tab", "Switch between input modes", ShortcutCategory::Navigation, true);
        self.add_shortcut("Ctrl+Tab", "Next subtab", ShortcutCategory::Navigation, false);
        self.add_shortcut("Ctrl+Shift+Tab", "Previous subtab", ShortcutCategory::Navigation, false);
        self.add_shortcut("↑/↓", "Navigate messages/items", ShortcutCategory::Navigation, true);
        self.add_shortcut("PgUp/PgDn", "Page up/down", ShortcutCategory::Navigation, true);
        self.add_shortcut("Home/End", "Jump to start/end", ShortcutCategory::Navigation, true);
        
        // Chat shortcuts
        self.add_shortcut("Enter", "Send message", ShortcutCategory::Chat, true);
        self.add_shortcut("Shift+Enter", "New line in message", ShortcutCategory::Chat, true);
        self.add_shortcut("Ctrl+L", "Clear chat", ShortcutCategory::Chat, false);
        self.add_shortcut("Ctrl+E", "Export chat", ShortcutCategory::Chat, false);
        self.add_shortcut("Ctrl+F", "Find in chat", ShortcutCategory::Chat, false);
        self.add_shortcut("Ctrl+R", "Regenerate last response", ShortcutCategory::Chat, false);
        self.add_shortcut("Ctrl+Z", "Undo last message", ShortcutCategory::Chat, false);
        
        // Command shortcuts
        self.add_shortcut("/", "Open command palette", ShortcutCategory::Commands, true);
        self.add_shortcut(":", "Quick command", ShortcutCategory::Commands, true);
        self.add_shortcut("Ctrl+P", "Command history", ShortcutCategory::Commands, false);
        self.add_shortcut("Ctrl+Space", "Command completion", ShortcutCategory::Commands, true);
        
        // Editing shortcuts
        self.add_shortcut("Ctrl+A", "Select all", ShortcutCategory::Editing, true);
        self.add_shortcut("Ctrl+C", "Copy", ShortcutCategory::Editing, true);
        self.add_shortcut("Ctrl+V", "Paste", ShortcutCategory::Editing, true);
        self.add_shortcut("Ctrl+X", "Cut", ShortcutCategory::Editing, true);
        self.add_shortcut("Ctrl+←/→", "Move by word", ShortcutCategory::Editing, true);
        self.add_shortcut("Alt+Backspace", "Delete word", ShortcutCategory::Editing, true);
        
        // View shortcuts
        self.add_shortcut("Ctrl+Plus", "Increase font size", ShortcutCategory::View, false);
        self.add_shortcut("Ctrl+Minus", "Decrease font size", ShortcutCategory::View, false);
        self.add_shortcut("Ctrl+0", "Reset font size", ShortcutCategory::View, false);
        self.add_shortcut("F11", "Toggle fullscreen", ShortcutCategory::View, false);
        self.add_shortcut("Ctrl+B", "Toggle sidebar", ShortcutCategory::View, false);
        self.add_shortcut("Ctrl+I", "Toggle info panel", ShortcutCategory::View, false);
        
        // Help shortcuts
        self.add_shortcut("?", "Show help", ShortcutCategory::Help, true);
        self.add_shortcut("Ctrl+?", "Context help", ShortcutCategory::Help, false);
        self.add_shortcut("F2", "Show tips", ShortcutCategory::Help, false);
    }
    
    /// Add a keyboard shortcut
    pub fn add_shortcut(&mut self, keys: &str, description: &str, category: ShortcutCategory, context_sensitive: bool) {
        self.shortcuts.push(KeyboardShortcut {
            keys: keys.to_string(),
            description: description.to_string(),
            category,
            context_sensitive,
        });
    }
    
    /// Toggle overlay visibility
    pub fn toggle(&mut self) {
        self.visible = !self.visible;
        if self.visible {
            self.reset_selection();
        }
    }
    
    /// Show the overlay
    pub fn show(&mut self) {
        self.visible = true;
        self.reset_selection();
    }
    
    /// Hide the overlay
    pub fn hide(&mut self) {
        self.visible = false;
    }
    
    /// Check if overlay is visible
    pub fn is_visible(&self) -> bool {
        self.visible
    }
    
    /// Set filter
    pub fn set_filter(&mut self, filter: String) {
        self.filter = filter;
        self.reset_selection();
    }
    
    /// Set category filter
    pub fn set_category(&mut self, category: Option<ShortcutCategory>) {
        self.selected_category = category;
        self.reset_selection();
    }
    
    /// Reset selection
    fn reset_selection(&mut self) {
        self.selected_index = 0;
        self.scroll_offset = 0;
    }
    
    /// Get filtered shortcuts
    fn get_filtered_shortcuts(&self) -> Vec<&KeyboardShortcut> {
        self.shortcuts
            .iter()
            .filter(|s| {
                // Category filter
                if let Some(cat) = &self.selected_category {
                    if s.category != *cat {
                        return false;
                    }
                }
                
                // Text filter
                if !self.filter.is_empty() {
                    let filter_lower = self.filter.to_lowercase();
                    let keys_match = s.keys.to_lowercase().contains(&filter_lower);
                    let desc_match = s.description.to_lowercase().contains(&filter_lower);
                    if !keys_match && !desc_match {
                        return false;
                    }
                }
                
                true
            })
            .collect()
    }
    
    /// Navigate up
    pub fn navigate_up(&mut self) {
        if self.selected_index > 0 {
            self.selected_index -= 1;
            
            // Adjust scroll if needed
            if self.selected_index < self.scroll_offset {
                self.scroll_offset = self.selected_index;
            }
        }
    }
    
    /// Navigate down
    pub fn navigate_down(&mut self) {
        let filtered = self.get_filtered_shortcuts();
        if self.selected_index < filtered.len().saturating_sub(1) {
            self.selected_index += 1;
            
            // Adjust scroll if needed
            let visible_height = 15; // Approximate visible items
            if self.selected_index >= self.scroll_offset + visible_height {
                self.scroll_offset = self.selected_index - visible_height + 1;
            }
        }
    }
    
    /// Render the overlay
    pub fn render(&self, f: &mut Frame, area: Rect) {
        if !self.visible {
            return;
        }
        
        // Calculate overlay size (centered, 80% of screen)
        let overlay_width = (area.width as f32 * 0.8) as u16;
        let overlay_height = (area.height as f32 * 0.8) as u16;
        let overlay_x = (area.width - overlay_width) / 2;
        let overlay_y = (area.height - overlay_height) / 2;
        
        let overlay_area = Rect {
            x: area.x + overlay_x,
            y: area.y + overlay_y,
            width: overlay_width,
            height: overlay_height,
        };
        
        // Clear background
        f.render_widget(Clear, overlay_area);
        
        // Main block
        let block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(Color::Cyan))
            .title(" ⌨️  Keyboard Shortcuts ")
            .title_alignment(Alignment::Center);
            
        let inner = block.inner(overlay_area);
        f.render_widget(block, overlay_area);
        
        // Layout
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),  // Header
                Constraint::Min(10),    // Shortcuts list
                Constraint::Length(3),  // Footer
            ])
            .split(inner);
        
        // Header with categories
        self.render_header(f, chunks[0]);
        
        // Shortcuts list
        self.render_shortcuts_list(f, chunks[1]);
        
        // Footer with help
        self.render_footer(f, chunks[2]);
    }
    
    /// Render header with category tabs
    fn render_header(&self, f: &mut Frame, area: Rect) {
        let categories = vec![
            ShortcutCategory::General,
            ShortcutCategory::Navigation,
            ShortcutCategory::Chat,
            ShortcutCategory::Commands,
            ShortcutCategory::Editing,
            ShortcutCategory::View,
            ShortcutCategory::Help,
        ];
        
        let mut spans = vec![Span::raw("Filter: ")];
        
        // All category
        let all_style = if self.selected_category.is_none() {
            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::Gray)
        };
        spans.push(Span::styled("[All]", all_style));
        spans.push(Span::raw(" "));
        
        // Individual categories
        for cat in categories {
            let style = if self.selected_category.as_ref() == Some(&cat) {
                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::Gray)
            };
            
            spans.push(Span::styled(
                format!("[{} {}]", cat.icon(), cat.display_name()),
                style
            ));
            spans.push(Span::raw(" "));
        }
        
        let header = Paragraph::new(Line::from(spans))
            .block(Block::default().borders(Borders::BOTTOM));
            
        f.render_widget(header, area);
    }
    
    /// Render shortcuts list
    fn render_shortcuts_list(&self, f: &mut Frame, area: Rect) {
        let filtered = self.get_filtered_shortcuts();
        
        if filtered.is_empty() {
            let empty_msg = if !self.filter.is_empty() {
                "No shortcuts match your filter"
            } else {
                "No shortcuts available"
            };
            
            let msg = Paragraph::new(empty_msg)
                .style(Style::default().fg(Color::DarkGray))
                .alignment(Alignment::Center);
                
            f.render_widget(msg, area);
            return;
        }
        
        // Create list items grouped by category
        let mut items = Vec::new();
        let mut last_category = None;
        
        for (i, shortcut) in filtered.iter().enumerate() {
            // Add category header if changed
            if last_category != Some(&shortcut.category) {
                if !items.is_empty() {
                    items.push(ListItem::new(Line::from(""))); // Spacing
                }
                
                items.push(ListItem::new(Line::from(vec![
                    Span::styled(
                        format!("{} {}", shortcut.category.icon(), shortcut.category.display_name()),
                        Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
                    ),
                ])));
                
                last_category = Some(&shortcut.category);
            }
            
            // Add shortcut item
            let is_selected = i == self.selected_index;
            let key_style = if is_selected {
                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::Green)
            };
            
            let desc_style = if is_selected {
                Style::default().fg(Color::White)
            } else {
                Style::default().fg(Color::Gray)
            };
            
            let context_indicator = if shortcut.context_sensitive {
                Span::styled(" ⚡", Style::default().fg(Color::Blue))
            } else {
                Span::raw("")
            };
            
            items.push(ListItem::new(Line::from(vec![
                Span::raw("  "),
                Span::styled(format!("{:<15}", shortcut.keys), key_style),
                Span::raw(" → "),
                Span::styled(&shortcut.description, desc_style),
                context_indicator,
            ])));
        }
        
        // Create scrollable list
        let list = List::new(items)
            .block(Block::default())
            .style(Style::default());
            
        f.render_widget(list, area);
    }
    
    /// Render footer with help text
    fn render_footer(&self, f: &mut Frame, area: Rect) {
        let help_text = vec![
            Span::raw("Press "),
            Span::styled("Esc", Style::default().fg(Color::Yellow)),
            Span::raw(" to close | "),
            Span::styled("↑↓", Style::default().fg(Color::Yellow)),
            Span::raw(" Navigate | "),
            Span::styled("Tab", Style::default().fg(Color::Yellow)),
            Span::raw(" Switch category | "),
            Span::styled("/", Style::default().fg(Color::Yellow)),
            Span::raw(" Search | "),
            Span::styled("⚡", Style::default().fg(Color::Blue)),
            Span::raw(" = Context sensitive"),
        ];
        
        let footer = Paragraph::new(Line::from(help_text))
            .block(Block::default().borders(Borders::TOP))
            .alignment(Alignment::Center);
            
        f.render_widget(footer, area);
    }
}

impl Default for KeyboardShortcutsOverlay {
    fn default() -> Self {
        Self::new()
    }
}

/// Quick reference card for common shortcuts
pub struct QuickReferenceCard;

impl QuickReferenceCard {
    /// Render a compact quick reference card
    pub fn render(f: &mut Frame, area: Rect) {
        let shortcuts = vec![
            ("Essential", vec![
                ("Enter", "Send message"),
                ("Tab", "Switch input mode"),
                ("Esc", "Cancel/Close"),
                ("F1", "Show all shortcuts"),
            ]),
            ("Navigation", vec![
                ("↑/↓", "Navigate"),
                ("PgUp/PgDn", "Page scroll"),
                ("Ctrl+Tab", "Next tab"),
            ]),
            ("Commands", vec![
                ("/", "Command palette"),
                ("Ctrl+L", "Clear chat"),
                ("Ctrl+E", "Export chat"),
            ]),
        ];
        
        let mut lines = vec![
            Line::from(vec![
                Span::styled("Quick Reference", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
            ]),
            Line::from(""),
        ];
        
        for (category, items) in shortcuts {
            lines.push(Line::from(vec![
                Span::styled(category, Style::default().fg(Color::Yellow))
            ]));
            
            for (key, desc) in items {
                lines.push(Line::from(vec![
                    Span::raw("  "),
                    Span::styled(format!("{:<10}", key), Style::default().fg(Color::Green)),
                    Span::raw(" "),
                    Span::styled(desc, Style::default().fg(Color::Gray)),
                ]));
            }
            
            lines.push(Line::from(""));
        }
        
        let card = Paragraph::new(lines)
            .block(Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .title(" ? "))
            .wrap(Wrap { trim: true });
            
        f.render_widget(card, area);
    }
}