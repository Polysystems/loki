//! Interactive widgets for the chat interface
//! 
//! Provides clickable buttons, context menus, quick actions, and other
//! interactive elements for the terminal UI.

use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph},
    Frame,
};
use crossterm::event::{MouseButton, MouseEvent, MouseEventKind};
use std::collections::HashMap;

/// Interactive widget manager
pub struct InteractiveWidgetManager {
    /// Registered widgets
    widgets: HashMap<String, Box<dyn InteractiveWidget>>,
    
    /// Widget positions for hit testing
    widget_areas: HashMap<String, Rect>,
    
    /// Currently focused widget
    focused_widget: Option<String>,
    
    /// Hover state
    hovered_widget: Option<String>,
    
    /// Context menus
    context_menus: Vec<ContextMenu>,
}

/// Base trait for interactive widgets
pub trait InteractiveWidget: Send + Sync {
    /// Get widget ID
    fn id(&self) -> &str;
    
    /// Render the widget
    fn render(&self, f: &mut Frame, area: Rect, theme: &super::theme_engine::ChatTheme);
    
    /// Handle mouse events
    fn handle_mouse(&mut self, event: MouseEvent, area: Rect) -> Option<WidgetAction>;
    
    /// Handle keyboard events
    fn handle_key(&mut self, key: crossterm::event::KeyEvent) -> Option<WidgetAction>;
    
    /// Check if point is within widget
    fn contains_point(&self, x: u16, y: u16, area: Rect) -> bool {
        x >= area.x && x < area.x + area.width && y >= area.y && y < area.y + area.height
    }
}

/// Widget actions
#[derive(Debug, Clone)]
pub enum WidgetAction {
    Click(String),
    Submit(String),
    Cancel,
    OpenContextMenu(Vec<MenuItem>),
    Navigate(String),
    Custom(String, String),
}

/// Button widget
pub struct Button {
    id: String,
    label: String,
    icon: Option<String>,
    style: ButtonStyle,
    enabled: bool,
    pressed: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum ButtonStyle {
    Primary,
    Secondary,
    Text,
    Icon,
}

impl Button {
    pub fn new(id: String, label: String) -> Self {
        Self {
            id,
            label,
            icon: None,
            style: ButtonStyle::Primary,
            enabled: true,
            pressed: false,
        }
    }
    
    pub fn with_icon(mut self, icon: String) -> Self {
        self.icon = Some(icon);
        self
    }
    
    pub fn with_style(mut self, style: ButtonStyle) -> Self {
        self.style = style;
        self
    }
}

impl InteractiveWidget for Button {
    fn id(&self) -> &str {
        &self.id
    }
    
    fn render(&self, f: &mut Frame, area: Rect, theme: &super::theme_engine::ChatTheme) {
        let style = if !self.enabled {
            theme.component_styles.button.disabled
        } else if self.pressed {
            theme.component_styles.button.pressed
        } else {
            match self.style {
                ButtonStyle::Primary => theme.component_styles.button.focused,
                ButtonStyle::Secondary => theme.component_styles.button.normal,
                ButtonStyle::Text => Style::default().fg(theme.colors.primary),
                ButtonStyle::Icon => Style::default(),
            }
        };
        
        let content = if let Some(icon) = &self.icon {
            format!("{} {}", icon, self.label)
        } else {
            self.label.clone()
        };
        
        let button = match self.style {
            ButtonStyle::Text | ButtonStyle::Icon => {
                Paragraph::new(content).style(style)
            }
            _ => {
                Paragraph::new(content)
                    .style(style)
                    .block(
                        Block::default()
                            .borders(Borders::ALL)
                            .border_style(style)
                    )
            }
        };
        
        f.render_widget(button, area);
    }
    
    fn handle_mouse(&mut self, event: MouseEvent, area: Rect) -> Option<WidgetAction> {
        if !self.enabled {
            return None;
        }
        
        match event.kind {
            MouseEventKind::Down(MouseButton::Left) => {
                if self.contains_point(event.column, event.row, area) {
                    self.pressed = true;
                }
            }
            MouseEventKind::Up(MouseButton::Left) => {
                if self.pressed && self.contains_point(event.column, event.row, area) {
                    self.pressed = false;
                    return Some(WidgetAction::Click(self.id.clone()));
                }
                self.pressed = false;
            }
            _ => {}
        }
        
        None
    }
    
    fn handle_key(&mut self, _key: crossterm::event::KeyEvent) -> Option<WidgetAction> {
        None
    }
}

/// Quick action bar
pub struct QuickActionBar {
    id: String,
    actions: Vec<QuickAction>,
    selected: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct QuickAction {
    pub id: String,
    pub icon: String,
    pub label: String,
    pub shortcut: Option<String>,
    pub enabled: bool,
}

impl QuickActionBar {
    pub fn new(id: String) -> Self {
        Self {
            id,
            actions: Vec::new(),
            selected: None,
        }
    }
    
    pub fn add_action(&mut self, action: QuickAction) {
        self.actions.push(action);
    }
}

impl InteractiveWidget for QuickActionBar {
    fn id(&self) -> &str {
        &self.id
    }
    
    fn render(&self, f: &mut Frame, area: Rect, theme: &super::theme_engine::ChatTheme) {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints(
                self.actions
                    .iter()
                    .map(|_| Constraint::Min(10))
                    .collect::<Vec<_>>()
            )
            .split(area);
        
        for (i, (action, chunk)) in self.actions.iter().zip(chunks.iter()).enumerate() {
            let is_selected = self.selected == Some(i);
            let style = if !action.enabled {
                theme.component_styles.button.disabled
            } else if is_selected {
                theme.component_styles.button.focused
            } else {
                theme.component_styles.button.normal
            };
            
            let content = vec![
                Line::from(vec![Span::styled(&action.icon, style.add_modifier(Modifier::BOLD))]),
                Line::from(vec![Span::styled(&action.label, style)]),
            ];
            
            if let Some(shortcut) = &action.shortcut {
                let shortcut_line = Line::from(vec![
                    Span::styled(shortcut, style.fg(Color::Gray).add_modifier(Modifier::DIM))
                ]);
                let paragraph = Paragraph::new(vec![content[0].clone(), content[1].clone(), shortcut_line])
                    .alignment(Alignment::Center);
                f.render_widget(paragraph, *chunk);
            } else {
                let paragraph = Paragraph::new(content)
                    .alignment(Alignment::Center);
                f.render_widget(paragraph, *chunk);
            }
        }
    }
    
    fn handle_mouse(&mut self, event: MouseEvent, area: Rect) -> Option<WidgetAction> {
        match event.kind {
            MouseEventKind::Up(MouseButton::Left) => {
                // Calculate which action was clicked
                let action_width = area.width / self.actions.len() as u16;
                let index = ((event.column - area.x) / action_width) as usize;
                
                if index < self.actions.len() && self.actions[index].enabled {
                    return Some(WidgetAction::Click(self.actions[index].id.clone()));
                }
            }
            MouseEventKind::Moved => {
                // Update hover state
                let action_width = area.width / self.actions.len() as u16;
                let index = ((event.column - area.x) / action_width) as usize;
                self.selected = if index < self.actions.len() {
                    Some(index)
                } else {
                    None
                };
            }
            _ => {}
        }
        
        None
    }
    
    fn handle_key(&mut self, key: crossterm::event::KeyEvent) -> Option<WidgetAction> {
        use crossterm::event::KeyCode;
        
        match key.code {
            KeyCode::Left => {
                if let Some(selected) = self.selected {
                    if selected > 0 {
                        self.selected = Some(selected - 1);
                    }
                } else {
                    self.selected = Some(0);
                }
            }
            KeyCode::Right => {
                if let Some(selected) = self.selected {
                    if selected < self.actions.len() - 1 {
                        self.selected = Some(selected + 1);
                    }
                } else {
                    self.selected = Some(0);
                }
            }
            KeyCode::Enter => {
                if let Some(selected) = self.selected {
                    if self.actions[selected].enabled {
                        return Some(WidgetAction::Click(self.actions[selected].id.clone()));
                    }
                }
            }
            _ => {}
        }
        
        None
    }
}

/// Context menu
#[derive(Debug, Clone)]
pub struct ContextMenu {
    pub id: String,
    pub position: (u16, u16),
    pub items: Vec<MenuItem>,
    pub selected: Option<usize>,
    pub visible: bool,
}

#[derive(Debug, Clone)]
pub struct MenuItem {
    pub id: String,
    pub label: String,
    pub icon: Option<String>,
    pub shortcut: Option<String>,
    pub enabled: bool,
    pub separator: bool,
}

impl ContextMenu {
    pub fn new(id: String, position: (u16, u16)) -> Self {
        Self {
            id,
            position,
            items: Vec::new(),
            selected: Some(0),
            visible: true,
        }
    }
    
    pub fn add_item(&mut self, item: MenuItem) {
        self.items.push(item);
    }
    
    pub fn render(&self, f: &mut Frame, theme: &super::theme_engine::ChatTheme) {
        if !self.visible {
            return;
        }
        
        // Calculate menu size
        let max_width = self.items
            .iter()
            .map(|item| {
                let base_len = item.label.len();
                let icon_len = item.icon.as_ref().map(|i| i.len() + 2).unwrap_or(0);
                let shortcut_len = item.shortcut.as_ref().map(|s| s.len() + 4).unwrap_or(0);
                base_len + icon_len + shortcut_len + 4
            })
            .max()
            .unwrap_or(20) as u16;
        
        let height = (self.items.len() + 2) as u16;
        
        // Ensure menu fits on screen
        let menu_area = Rect {
            x: self.position.0.min(f.area().width.saturating_sub(max_width)),
            y: self.position.1.min(f.area().height.saturating_sub(height)),
            width: max_width,
            height,
        };
        
        // Clear background
        f.render_widget(ratatui::widgets::Clear, menu_area);
        
        // Create menu items
        let items: Vec<ListItem> = self.items
            .iter()
            .enumerate()
            .map(|(i, item)| {
                if item.separator {
                    ListItem::new(Line::from("─".repeat(max_width as usize - 2)))
                        .style(theme.component_styles.menu.separator)
                } else {
                    let mut spans = Vec::new();
                    
                    if let Some(icon) = &item.icon {
                        spans.push(Span::raw(format!("{} ", icon)));
                    }
                    
                    spans.push(Span::raw(&item.label));
                    
                    if let Some(shortcut) = &item.shortcut {
                        let padding = max_width as usize - item.label.len() - 
                            item.icon.as_ref().map(|i| i.len() + 2).unwrap_or(0) - 
                            shortcut.len() - 4;
                        spans.push(Span::raw(" ".repeat(padding)));
                        spans.push(Span::styled(
                            shortcut,
                            Style::default().fg(Color::Gray).add_modifier(Modifier::DIM)
                        ));
                    }
                    
                    let style = if !item.enabled {
                        theme.component_styles.menu.item_disabled
                    } else if self.selected == Some(i) {
                        theme.component_styles.menu.item_selected
                    } else {
                        theme.component_styles.menu.item
                    };
                    
                    ListItem::new(Line::from(spans)).style(style)
                }
            })
            .collect();
        
        let menu = List::new(items)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(theme.component_styles.panel.border)
                    .style(theme.component_styles.panel.background)
            );
        
        f.render_widget(menu, menu_area);
    }
    
    pub fn handle_mouse(&mut self, event: MouseEvent) -> Option<WidgetAction> {
        match event.kind {
            MouseEventKind::Up(MouseButton::Left) => {
                if let Some(selected) = self.selected {
                    if selected < self.items.len() && self.items[selected].enabled && !self.items[selected].separator {
                        self.visible = false;
                        return Some(WidgetAction::Click(self.items[selected].id.clone()));
                    }
                }
            }
            MouseEventKind::Moved => {
                // Update selection based on mouse position
                // This would require calculating the actual menu position
            }
            _ => {}
        }
        
        None
    }
    
    pub fn handle_key(&mut self, key: crossterm::event::KeyEvent) -> Option<WidgetAction> {
        use crossterm::event::KeyCode;
        
        match key.code {
            KeyCode::Up => {
                if let Some(selected) = self.selected {
                    // Find previous non-separator item
                    for i in (0..selected).rev() {
                        if !self.items[i].separator {
                            self.selected = Some(i);
                            break;
                        }
                    }
                }
            }
            KeyCode::Down => {
                if let Some(selected) = self.selected {
                    // Find next non-separator item
                    for i in selected + 1..self.items.len() {
                        if !self.items[i].separator {
                            self.selected = Some(i);
                            break;
                        }
                    }
                }
            }
            KeyCode::Enter => {
                if let Some(selected) = self.selected {
                    if self.items[selected].enabled && !self.items[selected].separator {
                        self.visible = false;
                        return Some(WidgetAction::Click(self.items[selected].id.clone()));
                    }
                }
            }
            KeyCode::Esc => {
                self.visible = false;
                return Some(WidgetAction::Cancel);
            }
            _ => {}
        }
        
        None
    }
}

/// Tag selector widget
pub struct TagSelector {
    id: String,
    tags: Vec<Tag>,
    selected_tags: Vec<String>,
    allow_multiple: bool,
}

#[derive(Debug, Clone)]
pub struct Tag {
    pub id: String,
    pub label: String,
    pub color: Color,
}

impl TagSelector {
    pub fn new(id: String) -> Self {
        Self {
            id,
            tags: Vec::new(),
            selected_tags: Vec::new(),
            allow_multiple: true,
        }
    }
    
    pub fn add_tag(&mut self, tag: Tag) {
        self.tags.push(tag);
    }
    
    pub fn toggle_tag(&mut self, tag_id: &str) {
        if self.selected_tags.contains(&tag_id.to_string()) {
            self.selected_tags.retain(|t| t != tag_id);
        } else {
            if !self.allow_multiple {
                self.selected_tags.clear();
            }
            self.selected_tags.push(tag_id.to_string());
        }
    }
}

impl InteractiveWidget for TagSelector {
    fn id(&self) -> &str {
        &self.id
    }
    
    fn render(&self, f: &mut Frame, area: Rect, theme: &super::theme_engine::ChatTheme) {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints(
                self.tags
                    .iter()
                    .map(|tag| Constraint::Length(tag.label.len() as u16 + 4))
                    .collect::<Vec<_>>()
            )
            .split(area);
        
        for (tag, chunk) in self.tags.iter().zip(chunks.iter()) {
            let is_selected = self.selected_tags.contains(&tag.id);
            let style = if is_selected {
                Style::default().bg(tag.color).fg(Color::Black)
            } else {
                Style::default().fg(tag.color)
            };
            
            let tag_text = if is_selected {
                format!(" ✓ {} ", tag.label)
            } else {
                format!("   {} ", tag.label)
            };
            
            let tag_widget = Paragraph::new(tag_text)
                .style(style)
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .border_style(Style::default().fg(tag.color))
                );
            
            f.render_widget(tag_widget, *chunk);
        }
    }
    
    fn handle_mouse(&mut self, event: MouseEvent, area: Rect) -> Option<WidgetAction> {
        if let MouseEventKind::Up(MouseButton::Left) = event.kind {
            // Calculate which tag was clicked
            let mut x_offset = area.x;
            for tag in &self.tags {
                let tag_width = tag.label.len() as u16 + 4;
                if event.column >= x_offset && event.column < x_offset + tag_width {
                    let tag_id = tag.id.clone();
                    self.toggle_tag(&tag_id);
                    return Some(WidgetAction::Custom("tag_toggled".to_string(), tag_id));
                }
                x_offset += tag_width;
            }
        }
        
        None
    }
    
    fn handle_key(&mut self, _key: crossterm::event::KeyEvent) -> Option<WidgetAction> {
        None
    }
}

impl InteractiveWidgetManager {
    pub fn new() -> Self {
        Self {
            widgets: HashMap::new(),
            widget_areas: HashMap::new(),
            focused_widget: None,
            hovered_widget: None,
            context_menus: Vec::new(),
        }
    }
    
    /// Register a widget
    pub fn register_widget(&mut self, widget: Box<dyn InteractiveWidget>) {
        let id = widget.id().to_string();
        self.widgets.insert(id, widget);
    }
    
    /// Update widget area for hit testing
    pub fn update_widget_area(&mut self, id: &str, area: Rect) {
        self.widget_areas.insert(id.to_string(), area);
    }
    
    /// Handle mouse event
    pub fn handle_mouse(&mut self, event: MouseEvent) -> Option<WidgetAction> {
        // Check context menus first
        for menu in &mut self.context_menus {
            if let Some(action) = menu.handle_mouse(event) {
                return Some(action);
            }
        }
        
        // Find widget under mouse
        let widget_id = self.widget_areas
            .iter()
            .find(|(_, area)| {
                event.column >= area.x && 
                event.column < area.x + area.width &&
                event.row >= area.y && 
                event.row < area.y + area.height
            })
            .map(|(id, _)| id.clone());
        
        // Update hover state
        if let MouseEventKind::Moved = event.kind {
            self.hovered_widget = widget_id.clone();
        }
        
        // Forward to widget
        if let Some(id) = widget_id {
            if let Some(area) = self.widget_areas.get(&id).cloned() {
                if let Some(widget) = self.widgets.get_mut(&id) {
                    return widget.handle_mouse(event, area);
                }
            }
        }
        
        None
    }
    
    /// Handle keyboard event
    pub fn handle_key(&mut self, key: crossterm::event::KeyEvent) -> Option<WidgetAction> {
        // Check context menus first
        for menu in &mut self.context_menus {
            if menu.visible {
                return menu.handle_key(key);
            }
        }
        
        // Forward to focused widget
        if let Some(id) = &self.focused_widget {
            if let Some(widget) = self.widgets.get_mut(id) {
                return widget.handle_key(key);
            }
        }
        
        None
    }
    
    /// Open context menu
    pub fn open_context_menu(&mut self, menu: ContextMenu) {
        self.context_menus.push(menu);
    }
    
    /// Close all context menus
    pub fn close_context_menus(&mut self) {
        self.context_menus.clear();
    }
    
    /// Render all widgets
    pub fn render(&self, f: &mut Frame, theme: &super::theme_engine::ChatTheme) {
        // Render widgets
        for (id, widget) in &self.widgets {
            if let Some(area) = self.widget_areas.get(id) {
                widget.render(f, *area, theme);
            }
        }
        
        // Render context menus on top
        for menu in &self.context_menus {
            menu.render(f, theme);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_button_creation() {
        let button = Button::new("test".to_string(), "Click Me".to_string())
            .with_icon("🔵".to_string())
            .with_style(ButtonStyle::Primary);
        
        assert_eq!(button.id(), "test");
        assert_eq!(button.label, "Click Me");
        assert_eq!(button.icon, Some("🔵".to_string()));
    }
    
    #[test]
    fn test_tag_selector() {
        let mut selector = TagSelector::new("tags".to_string());
        selector.add_tag(Tag {
            id: "rust".to_string(),
            label: "Rust".to_string(),
            color: Color::Red,
        });
        
        selector.toggle_tag("rust");
        assert!(selector.selected_tags.contains(&"rust".to_string()));
        
        selector.toggle_tag("rust");
        assert!(!selector.selected_tags.contains(&"rust".to_string()));
    }
}