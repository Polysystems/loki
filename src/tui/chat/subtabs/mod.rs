//! Subtab controllers module
//! 
//! Individual subtab implementations for the chat interface

pub mod chat_tab;
pub mod editor_tab;
pub mod models_tab;
pub mod history_tab;
pub mod settings_tab;
pub mod orchestration_tab;
pub mod agents_tab;
pub mod cli_tab;
pub mod statistics_tab;

// Re-export subtab trait and implementations
pub use chat_tab::ChatTab;
pub use editor_tab::EditorTab;
pub use models_tab::ModelsTab;
pub use history_tab::HistoryTab;
pub use settings_tab::SettingsTab;
pub use orchestration_tab::OrchestrationTab;
pub use agents_tab::AgentsTab;
pub use cli_tab::CliTab;
pub use statistics_tab::StatisticsTab;

/// Trait for subtab implementations
pub trait SubtabController {
    /// Render the subtab
    fn render(&mut self, f: &mut ratatui::Frame, area: ratatui::layout::Rect);
    
    /// Handle input for the subtab
    fn handle_input(&mut self, key: crossterm::event::KeyEvent) -> anyhow::Result<()>;
    
    /// Update the subtab state
    fn update(&mut self) -> anyhow::Result<()>;
    
    /// Get the subtab name
    fn name(&self) -> &str;
    
    /// Get the subtab title (defaults to name)
    fn title(&self) -> String {
        self.name().to_string()
    }
}