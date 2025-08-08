//! Command processing for chat interface

use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use anyhow::{Result};

// Define command types locally until we can import from the main chat module
#[derive(Debug, Clone)]
pub enum ChatCommand {
    Clear,
    Save(Option<String>),
    Load(String),
    Export(String),
    SetModel(String),
    ListModels,
    EnableOrchestration,
    DisableOrchestration,
    SetStrategy(String),
    Status,
    Help,
    Statistics,
}

#[derive(Debug, Clone)]
pub enum CommandResult {
    Success(String),
    Error(String),
}
use crate::tui::run::AssistantResponseType;
use crate::tui::chat::ChatState;
use super::super::state::StateManagementExt;
use super::super::orchestration::OrchestrationManager;

/// Handles command processing for the chat interface
pub struct CommandProcessor {
    /// Current chat state
    state: Arc<RwLock<ChatState>>,
    
    /// Orchestration manager
    orchestration: Arc<RwLock<OrchestrationManager>>,
    
    /// Channel for sending responses
    response_tx: mpsc::Sender<AssistantResponseType>,
}

impl CommandProcessor {
    /// Create a new command processor
    pub fn new(
        state: Arc<RwLock<ChatState>>,
        orchestration: Arc<RwLock<OrchestrationManager>>,
        response_tx: mpsc::Sender<AssistantResponseType>,
    ) -> Self {
        Self {
            state,
            orchestration,
            response_tx,
        }
    }
    
    /// Process a command
    pub async fn process_command(&self, command: ChatCommand) -> Result<CommandResult> {
        match command {
            ChatCommand::Clear => {
                self.handle_clear().await
            }
            ChatCommand::Save(path) => {
                self.handle_save(path).await
            }
            ChatCommand::Load(path) => {
                self.handle_load(path).await
            }
            ChatCommand::Export(format) => {
                self.handle_export(format).await
            }
            ChatCommand::SetModel(model) => {
                self.handle_set_model(model).await
            }
            ChatCommand::ListModels => {
                self.handle_list_models().await
            }
            ChatCommand::EnableOrchestration => {
                self.handle_enable_orchestration().await
            }
            ChatCommand::DisableOrchestration => {
                self.handle_disable_orchestration().await
            }
            ChatCommand::SetStrategy(strategy) => {
                self.handle_set_strategy(strategy).await
            }
            ChatCommand::Status => {
                self.handle_status().await
            }
            ChatCommand::Help => {
                self.handle_help().await
            }
            ChatCommand::Statistics => {
                self.handle_statistics().await
            }
        }
    }
    
    /// Handle clear command
    async fn handle_clear(&self) -> Result<CommandResult> {
        let mut state = self.state.write().await;
        state.messages.clear();
        Ok(CommandResult::Success("Chat cleared".to_string()))
    }
    
    /// Handle save command
    async fn handle_save(&self, path: Option<String>) -> Result<CommandResult> {
        let state = self.state.read().await;
        let path = path.unwrap_or_else(|| format!("chat_{}.json", chrono::Utc::now().format("%Y%m%d_%H%M%S")));
        
        // Use the persistence module to save
        use crate::tui::chat::state::persistence::ChatPersistence;
        let persistence = ChatPersistence::with_default_dir()?;
        
        match persistence.save_chat_with_filename(&*state, &path).await {
            Ok(_) => Ok(CommandResult::Success(format!("Chat saved to {}", path))),
            Err(e) => Ok(CommandResult::Error(format!("Failed to save chat: {}", e)))
        }
    }
    
    /// Handle load command
    async fn handle_load(&self, path: String) -> Result<CommandResult> {
        use crate::tui::chat::state::persistence::ChatPersistence;
        let persistence = ChatPersistence::with_default_dir()?;
        
        match persistence.load_chat_by_filename(&path).await {
            Ok(loaded_state) => {
                let mut state = self.state.write().await;
                *state = loaded_state;
                Ok(CommandResult::Success(format!("Chat loaded from {}", path)))
            }
            Err(e) => Ok(CommandResult::Error(format!("Failed to load chat: {}", e)))
        }
    }
    
    /// Handle export command
    async fn handle_export(&self, format: String) -> Result<CommandResult> {
        use crate::tui::chat::export::{ChatExporter, ExportFormat, ExportOptions};
        use std::path::Path;
        
        let state = self.state.read().await;
        
        // Parse format
        let export_format = match format.to_lowercase().as_str() {
            "json" => ExportFormat::Json,
            "markdown" | "md" => ExportFormat::Markdown,
            "text" | "txt" => ExportFormat::Text,
            "html" => ExportFormat::Html,
            "pdf" => ExportFormat::Pdf,
            "csv" => ExportFormat::Csv,
            _ => {
                return Ok(CommandResult::Error(format!(
                    "Unknown format: {}. Supported formats: json, markdown, text, html, pdf, csv", 
                    format
                )))
            }
        };
        
        // Create exporter
        let mut exporter = ChatExporter::new();
        
        // Generate filename
        let filename = ChatExporter::suggest_filename(export_format, &state.title);
        let path = Path::new(&filename);
        
        // Export with default options
        let options = ExportOptions::default();
        
        match exporter.export_chat(&state, export_format, path, options).await {
            Ok(_) => {
                // Get file size for info
                let size = match std::fs::metadata(&filename) {
                    Ok(metadata) => {
                        let bytes = metadata.len();
                        if bytes > 1_000_000 {
                            format!(" ({:.1} MB)", bytes as f64 / 1_000_000.0)
                        } else if bytes > 1_000 {
                            format!(" ({:.1} KB)", bytes as f64 / 1_000.0)
                        } else {
                            format!(" ({} bytes)", bytes)
                        }
                    }
                    Err(_) => String::new(),
                };
                
                Ok(CommandResult::Success(format!(
                    "✅ Chat exported successfully!\n📁 File: {}{}\n📊 Format: {}\n💬 Messages: {}",
                    filename,
                    size,
                    export_format.name(),
                    state.messages.len()
                )))
            }
            Err(e) => Ok(CommandResult::Error(format!("Export failed: {}", e)))
        }
    }
    
    /// Handle set model command
    async fn handle_set_model(&self, model: String) -> Result<CommandResult> {
        let mut state = self.state.write().await;
        state.update_model(model.clone());
        Ok(CommandResult::Success(format!("Model set to {}", model)))
    }
    
    /// Handle list models command
    async fn handle_list_models(&self) -> Result<CommandResult> {
        let orchestration = self.orchestration.read().await;
        let models = orchestration.get_available_models().await;
        
        let model_list = models.join("\n");
        Ok(CommandResult::Success(format!("Available models:\n{}", model_list)))
    }
    
    /// Handle enable orchestration command
    async fn handle_enable_orchestration(&self) -> Result<CommandResult> {
        let mut orchestration = self.orchestration.write().await;
        orchestration.orchestration_enabled = true;
        Ok(CommandResult::Success("Orchestration enabled".to_string()))
    }
    
    /// Handle disable orchestration command
    async fn handle_disable_orchestration(&self) -> Result<CommandResult> {
        let mut orchestration = self.orchestration.write().await;
        orchestration.orchestration_enabled = false;
        Ok(CommandResult::Success("Orchestration disabled".to_string()))
    }
    
    /// Handle set strategy command
    async fn handle_set_strategy(&self, strategy: String) -> Result<CommandResult> {
        let mut orchestration = self.orchestration.write().await;
        
        match strategy.as_str() {
            "roundrobin" => orchestration.preferred_strategy = crate::tui::chat::orchestration::RoutingStrategy::RoundRobin,
            "leastlatency" => orchestration.preferred_strategy = crate::tui::chat::orchestration::RoutingStrategy::LeastLatency,
            "contextaware" => orchestration.preferred_strategy = crate::tui::chat::orchestration::RoutingStrategy::ContextAware,
            _ => return Ok(CommandResult::Error(format!("Unknown strategy: {}", strategy))),
        }
        
        Ok(CommandResult::Success(format!("Strategy set to {}", strategy)))
    }
    
    /// Handle status command
    async fn handle_status(&self) -> Result<CommandResult> {
        let state = self.state.read().await;
        let orchestration = self.orchestration.read().await;
        
        let status = format!(
            "Chat: {}\nMessages: {}\nModel: {}\nOrchestration: {}\nStrategy: {:?}",
            state.name,
            state.messages.len(),
            state.active_model.as_ref().map(|m| m.as_str()).unwrap_or("default"),
            if orchestration.orchestration_enabled { "enabled" } else { "disabled" },
            orchestration.preferred_strategy
        );
        
        Ok(CommandResult::Success(status))
    }
    
    /// Handle help command
    async fn handle_help(&self) -> Result<CommandResult> {
        let help_text = r#"Available commands:
/clear - Clear chat history
/save [path] - Save chat to file
/load <path> - Load chat from file
/export <format> - Export chat (markdown, json)
/model <name> - Set active model
/models - List available models
/orchestration on|off - Toggle orchestration
/strategy <name> - Set routing strategy
/status - Show current status
/stats - Show chat statistics
/help - Show this help message"#;
        
        Ok(CommandResult::Success(help_text.to_string()))
    }
    
    /// Handle statistics command
    async fn handle_statistics(&self) -> Result<CommandResult> {
        use crate::tui::chat::statistics::{MetricsCalculator, TimeRange};
        
        let state = self.state.read().await;
        let metrics = MetricsCalculator::calculate(&state, TimeRange::AllTime);
        
        let stats = format!(
            "📊 Chat Statistics (All Time)\n\n\
            Total Messages: {}\n\
            User Messages: {}\n\
            AI Messages: {}\n\
            Total Tokens: {}\n\
            Avg Response Time: {:.0}ms\n\
            Error Rate: {:.1}%\n\
            Success Rate: {:.1}%\n\
            Avg Conversation Length: {:.1} messages\n\n\
            💡 Tip: Switch to Statistics tab for detailed analytics",
            metrics.total_messages,
            metrics.messages_by_type.get("User").unwrap_or(&0),
            metrics.messages_by_type.get("Assistant").unwrap_or(&0),
            metrics.total_tokens,
            metrics.avg_response_time,
            if metrics.total_messages > 0 {
                metrics.error_count as f64 / metrics.total_messages as f64 * 100.0
            } else {
                0.0
            },
            metrics.quality_metrics.success_rate * 100.0,
            metrics.avg_conversation_length
        );
        
        Ok(CommandResult::Success(stats))
    }
}