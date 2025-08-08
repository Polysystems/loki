//! Modular rendering system for chat UI

pub mod chat_renderer;
pub mod orchestration_renderer;
pub mod agent_renderer;
pub mod agent_thread_view;
pub mod tools_sidebar;
pub mod cognitive_insights;
pub mod main_renderer;
pub mod chat_content_impl;
pub mod settings_impl;
pub mod modular_renderer;
pub mod render_state;

// Re-export the main rendering function
pub use main_renderer::draw_modular_chat_tab;
// Re-export render state utilities
pub use render_state::{
    RenderState, RenderStateManager, CachedOrchestration, CachedMessages,
    CachedAgents, CachedTools, get_render_state_manager, get_current_render_state,
    initialize_render_state
};

use ratatui::Frame;
use ratatui::layout::Rect;

/// Main trait for rendering chat components
pub trait ChatRenderer {
    /// Render the component to the given area
    fn render(&self, f: &mut Frame, area: Rect);
}

/// Coordinator for all chat rendering
pub struct RenderingCoordinator {
    pub chat_renderer: chat_renderer::ChatContentRenderer,
    pub orchestration_renderer: orchestration_renderer::OrchestrationRenderer,
    pub agent_renderer: agent_renderer::AgentRenderer,
}

impl RenderingCoordinator {
    pub fn new() -> Self {
        Self {
            chat_renderer: chat_renderer::ChatContentRenderer::new(),
            orchestration_renderer: orchestration_renderer::OrchestrationRenderer::new(),
            agent_renderer: agent_renderer::AgentRenderer::new(),
        }
    }
    
    /// Main entry point for rendering the chat tab
    pub fn render_chat_tab(&self, f: &mut Frame, area: Rect, chat_manager: &crate::tui::chat::ChatManager) {
        // This will coordinate rendering of all chat components
        // Note: Removed the circular dependency on App::default()
        // The actual rendering is handled by the main_renderer module
    }
}