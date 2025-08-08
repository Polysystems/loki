//! Enhanced chat interface module for Loki TUI
//! 
//! This module provides a sophisticated terminal chat experience with rich
//! message rendering, multi-panel layouts, and interactive components.

pub mod message_renderer;
pub mod layout_manager;
pub mod input_handler;
pub mod stream_renderer;
pub mod interactive_widgets;
pub mod markdown_parser;
pub mod syntax_highlighter;
pub mod workflow_visualizer;
pub mod theme_engine;
pub mod file_handler;
pub mod code_completion_renderer;
pub mod thread_renderer;
pub mod rich_media_renderer;
pub mod chat_exporter;
pub mod template_manager;
pub mod template_ui;
pub mod collaboration;
pub mod collaboration_renderer;
pub mod agent_stream_manager;
pub mod agent_panel_renderer;
pub mod agent_panel;
pub mod cognitive_panel;

// Re-export commonly used types
pub use message_renderer::{MessageRenderer, RichMessage, MessageType, MessageMetadata, ToolStatus, WorkflowStep, StepStatus, DataFormat, StreamingInfo};
pub use layout_manager::{ChatLayout, PanelType, LayoutConfig, LayoutPreset, LayoutManager};
pub use input_handler::{InputHandler, InputMode, CompletionSuggestion, InputEvent};
pub use stream_renderer::{StreamRenderer, StreamState, StreamUpdate};
pub use markdown_parser::{MarkdownParser, FormattedText};
pub use syntax_highlighter::{SyntaxHighlighter, Language};
pub use theme_engine::{ChatTheme, ThemeVariant};
pub use file_handler::{FileHandler, FileAttachment, AttachmentState, AttachmentManager};
pub use code_completion_renderer::CodeCompletionRenderer;
pub use thread_renderer::ThreadRenderer;
pub use rich_media_renderer::{RichMediaRenderer, RichMediaContent, MediaType, MediaData};
pub use chat_exporter::{ChatExporter, ChatImporter, ExportFormat, ExportOptions};
pub use template_manager::{TemplateManager, ChatTemplate, CodeSnippet, TemplateCategory, TemplateInputState};
pub use template_ui::{TemplatePicker, TemplateInputDialog, SnippetPicker};
pub use collaboration::{CollaborationManager, CollaborationSession, UserPresence, CollaborationEvent, EditOperation};
pub use collaboration_renderer::CollaborationRenderer;
pub use agent_stream_manager::{
    AgentStreamManager, AgentStream, AgentMessage, AgentMessageType, 
    MessagePriority, AgentStatus, StreamRoutingRules, AgentStreamUpdate, StreamTarget
};
pub use agent_panel::{render_agent_panel, render_agent_panels_grid, AgentPanelConfig};
pub use cognitive_panel::CognitivePanel;

use ratatui::layout::Rect;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Main chat interface coordinator
pub struct EnhancedChatInterface {
    /// Message rendering engine
    pub message_renderer: MessageRenderer,
    
    /// Layout management
    pub layout_manager: LayoutManager,
    
    /// Input handling
    pub input_handler: InputHandler,
    
    /// Stream rendering
    pub stream_renderer: StreamRenderer,
    
    /// Theme engine
    pub theme: ChatTheme,
    
    /// Current view state
    pub view_state: Arc<RwLock<ChatViewState>>,
}

/// Chat view state
#[derive(Debug, Clone)]
pub struct ChatViewState {
    /// Active panel
    pub active_panel: PanelType,
    
    /// Scroll positions for each panel
    pub scroll_positions: std::collections::HashMap<PanelType, usize>,
    
    /// Expanded/collapsed sections
    pub expanded_sections: std::collections::HashSet<String>,
    
    /// Current search query
    pub search_query: Option<String>,
    
    /// Selected message index
    pub selected_message: Option<usize>,
}

impl EnhancedChatInterface {
    /// Create a new enhanced chat interface
    pub fn new() -> Self {
        Self {
            message_renderer: MessageRenderer::new(),
            layout_manager: LayoutManager::new(),
            input_handler: InputHandler::new(),
            stream_renderer: StreamRenderer::new(),
            theme: ChatTheme::default(),
            view_state: Arc::new(RwLock::new(ChatViewState {
                active_panel: PanelType::Chat,
                scroll_positions: std::collections::HashMap::new(),
                expanded_sections: std::collections::HashSet::new(),
                search_query: None,
                selected_message: None,
            })),
        }
    }
    
    /// Update the layout based on terminal size
    pub fn update_layout(&mut self, area: Rect) {
        // The layout manager will calculate layout when needed
        let _ = self.layout_manager.calculate_layout(area);
    }
    
    /// Handle keyboard input with external buffer
    pub async fn handle_input(&mut self, key: crossterm::event::KeyEvent, buffer: &mut String) -> Result<(), Box<dyn std::error::Error>> {
        self.input_handler.handle_key(key, buffer).await
    }
    
    /// Get current input from handler
    pub fn get_current_input(&self, buffer: &String) -> String {
        self.input_handler.get_current_input(buffer)
    }
    
    /// Render a message with rich formatting
    pub fn render_message(&self, message: &RichMessage) -> Vec<String> {
        self.message_renderer.render(message, &self.theme)
    }
}


