//! Main entry point for chat rendering

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};

use crate::tui::App;
use crate::tui::chat::subtabs::SubtabController;

use super::{
    chat_renderer::ChatContentRenderer,
    orchestration_renderer::OrchestrationRenderer,
    agent_renderer::AgentRenderer,
};

/// Main renderer that coordinates all chat rendering
pub struct MainChatRenderer {
    chat_renderer: ChatContentRenderer,
    orchestration_renderer: OrchestrationRenderer,
    agent_renderer: AgentRenderer,
}

impl MainChatRenderer {
    pub fn new() -> Self {
        Self {
            chat_renderer: ChatContentRenderer::new(),
            orchestration_renderer: OrchestrationRenderer::new(),
            agent_renderer: AgentRenderer::new(),
        }
    }
    
    /// Main entry point - will replace draw_tab_chat
    pub fn render(&self, f: &mut Frame, app: &mut App, area: Rect) {
        let chat_manager = &app.state.chat;
        
        // Split area for subtab navigation and content
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),  // Subtab navigation
                Constraint::Min(0),     // Content area
            ])
            .split(area);
            
        // Draw subtab navigation (using existing function for now)
        crate::tui::ui::draw_sub_tab_navigation(
            f,
            &app.state.chat_tabs,
            chunks[0]
        );
        
        // Render content based on current subtab
        self.render_subtab_content(f, app, chunks[1]);
    }
    
    fn render_subtab_content(&self, f: &mut Frame, app: &mut App, area: Rect) {
        match app.state.chat_tabs.current_index {
            0 => self.render_chat_subtab(f, app, area),
            1 => self.render_editor_subtab(f, app, area),
            2 => self.render_models_subtab(f, app, area),
            3 => self.render_history_subtab(f, app, area),
            4 => self.render_settings_subtab(f, app, area),
            5 => self.render_orchestration_subtab(f, app, area),
            6 => self.render_agents_subtab(f, app, area),
            7 => self.render_cli_subtab(f, app, area),
            8 => self.render_statistics_subtab(f, app, area),
            _ => {}
        }
    }
    
    fn render_chat_subtab(&self, f: &mut Frame, app: &mut App, area: Rect) {
        // Use new modular implementation
        super::chat_content_impl::render_chat_content(f, app, area);
    }
    
    fn render_models_subtab(&self, f: &mut Frame, app: &mut App, area: Rect) {
        // Use new modular implementation
        super::chat_content_impl::render_models_content(f, app, area);
    }
    
    fn render_history_subtab(&self, f: &mut Frame, app: &mut App, area: Rect) {
        // Use new modular implementation
        super::chat_content_impl::render_history_content(f, app, area);
    }
    
    fn render_settings_subtab(&self, f: &mut Frame, app: &mut App, area: Rect) {
        // Use new modular implementation
        super::settings_impl::render_settings_content(f, app, area);
    }
    
    fn render_orchestration_subtab(&self, f: &mut Frame, app: &mut App, area: Rect) {
        // Use new modular implementation
        super::settings_impl::render_orchestration_content(f, app, area);
    }
    
    fn render_agents_subtab(&self, f: &mut Frame, app: &mut App, area: Rect) {
        // Use new modular implementation
        super::settings_impl::render_agents_content(f, app, area);
    }
    
    fn render_cli_subtab(&self, f: &mut Frame, app: &mut App, area: Rect) {
        // Use new modular implementation
        super::settings_impl::render_cli_content(f, app, area);
    }
    
    fn render_editor_subtab(&self, f: &mut Frame, app: &mut App, area: Rect) {
        // Use the actual EditorTab controller if available
        if let Some(ref editor_tab) = app.state.chat.editor_tab {
            // Use block_in_place to safely access the async lock
            tokio::task::block_in_place(|| {
                let rt = tokio::runtime::Handle::current();
                rt.block_on(async {
                    let mut editor = editor_tab.write().await;
                    editor.render(f, area);
                })
            });
        } else {
            // Fallback to static welcome screen if controller not available
            super::settings_impl::render_editor_content(f, app, area);
        }
    }
    
    fn render_statistics_subtab(&self, f: &mut Frame, app: &mut App, area: Rect) {
        // Use new modular implementation
        super::settings_impl::render_statistics_content(f, app, area);
    }
}

/// Public function that will replace draw_tab_chat
pub fn draw_modular_chat_tab(f: &mut Frame, app: &mut App, area: Rect) {
    let renderer = MainChatRenderer::new();
    renderer.render(f, app, area);
}