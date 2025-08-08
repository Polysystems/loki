//! Modular chat renderer that uses the SubtabManager directly

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use crate::tui::App;

/// Render the modular chat system
pub fn render_modular_chat(f: &mut Frame, app: &mut App, area: Rect) {
    // Split area for subtab navigation and content
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // Subtab navigation
            Constraint::Min(0),     // Content area
        ])
        .split(area);
        
    // Draw subtab navigation
    crate::tui::ui::draw_sub_tab_navigation(
        f,
        &app.state.chat_tabs,
        chunks[0]
    );
    
    // Use the modular system's subtab manager to render content
    app.state.chat.subtab_manager.borrow_mut().render(f, chunks[1]);
}

/// Handle input for the modular chat system
pub fn handle_modular_chat_input(app: &mut App, key: crossterm::event::KeyEvent) -> anyhow::Result<()> {
    // Delegate to the subtab manager
    app.state.chat.subtab_manager.borrow_mut().handle_input(key)
}