//! Main chat exporter implementation

use std::path::{Path, PathBuf};
use anyhow::{Result, Context};
use chrono::Local;

use crate::tui::chat::{ChatState, context::indexer::ConversationIndexer};
use crate::tui::run::AssistantResponseType;
use super::{
    ExportedChat, ExportedMessage, ExportMetadata, ChatStatistics,
    ExportFormat, ExportOptions,
    formats::{Exporter, JsonExporter, MarkdownExporter, TextExporter, HtmlExporter, CsvExporter},
};

/// Main chat exporter
pub struct ChatExporter {
    /// Conversation indexer for topic/entity extraction
    indexer: ConversationIndexer,
}

impl ChatExporter {
    /// Create a new chat exporter
    pub fn new() -> Self {
        Self {
            indexer: ConversationIndexer::new(),
        }
    }
    
    /// Export chat state to file
    pub async fn export_chat(
        &mut self,
        state: &ChatState,
        format: ExportFormat,
        path: &Path,
        options: ExportOptions,
    ) -> Result<()> {
        // Convert chat state to exported format
        let exported_chat = self.prepare_export(state, format).await?;
        
        // Export based on format
        match format {
            ExportFormat::Json => {
                JsonExporter.export_to_file(&exported_chat, path, &options)?;
            }
            ExportFormat::Markdown => {
                MarkdownExporter.export_to_file(&exported_chat, path, &options)?;
            }
            ExportFormat::Text => {
                TextExporter.export_to_file(&exported_chat, path, &options)?;
            }
            ExportFormat::Html => {
                HtmlExporter.export_to_file(&exported_chat, path, &options)?;
            }
            ExportFormat::Pdf => {
                // PDF export requires additional dependencies
                self.export_pdf(&exported_chat, path, &options)?;
            }
            ExportFormat::Csv => {
                CsvExporter.export_to_file(&exported_chat, path, &options)?;
            }
        }
        
        Ok(())
    }
    
    /// Export chat to string
    pub async fn export_to_string(
        &mut self,
        state: &ChatState,
        format: ExportFormat,
        options: ExportOptions,
    ) -> Result<String> {
        let exported_chat = self.prepare_export(state, format).await?;
        
        match format {
            ExportFormat::Json => JsonExporter.export_to_string(&exported_chat, &options),
            ExportFormat::Markdown => MarkdownExporter.export_to_string(&exported_chat, &options),
            ExportFormat::Text => TextExporter.export_to_string(&exported_chat, &options),
            ExportFormat::Html => HtmlExporter.export_to_string(&exported_chat, &options),
            ExportFormat::Csv => CsvExporter.export_to_string(&exported_chat, &options),
            ExportFormat::Pdf => Err(anyhow::anyhow!("PDF export to string not supported")),
        }
    }
    
    /// Prepare chat for export
    async fn prepare_export(&mut self, state: &ChatState, format: ExportFormat) -> Result<ExportedChat> {
        // Clear and rebuild index
        self.indexer.clear();
        
        // Convert messages and build index
        let mut exported_messages = Vec::new();
        let mut first_timestamp = None;
        let mut last_timestamp = None;
        
        for (i, msg) in state.messages.iter().enumerate() {
            let exported_msg = ExportedMessage::from(msg);
            
            // Index message for topic/entity extraction
            self.indexer.index_message(
                i,
                &exported_msg.content,
                exported_msg.timestamp.with_timezone(&chrono::Utc),
            );
            
            // Track date range
            if first_timestamp.is_none() {
                first_timestamp = Some(exported_msg.timestamp);
            }
            last_timestamp = Some(exported_msg.timestamp);
            
            exported_messages.push(exported_msg);
        }
        
        // Calculate statistics
        let statistics = self.calculate_statistics(&state.messages);
        
        // Create metadata
        let metadata = ExportMetadata {
            exported_at: Local::now(),
            chat_title: state.title.clone(),
            format: format.name().to_string(),
            loki_version: env!("CARGO_PKG_VERSION").to_string(),
            message_count: exported_messages.len(),
            date_range: match (first_timestamp, last_timestamp) {
                (Some(first), Some(last)) => Some((first, last)),
                _ => None,
            },
        };
        
        Ok(ExportedChat {
            metadata,
            messages: exported_messages,
            statistics,
        })
    }
    
    /// Calculate chat statistics
    fn calculate_statistics(&self, messages: &[AssistantResponseType]) -> ChatStatistics {
        let mut total_length = 0;
        let mut messages_by_author = std::collections::HashMap::new();
        let mut total_tokens = 0;
        let mut hourly_activity = std::collections::HashMap::new();
        
        for msg in messages {
            match msg {
                AssistantResponseType::Message { author, message, timestamp, metadata, .. } => {
                    *messages_by_author.entry(author.clone()).or_insert(0) += 1;
                    total_length += message.len();
                    
                    if let Some(tokens) = metadata.tokens_used {
                        total_tokens += tokens as usize;
                    } else {
                        // Estimate tokens (rough approximation)
                        total_tokens += message.split_whitespace().count();
                    }
                    
                    // Track hourly activity
                    if let Ok(dt) = DateTime::parse_from_rfc3339(timestamp) {
                        let hour = dt.hour();
                        *hourly_activity.entry(hour).or_insert(0) += 1;
                    }
                }
                AssistantResponseType::UserMessage { content, .. } => {
                    *messages_by_author.entry("user".to_string()).or_insert(0) += 1;
                    total_length += content.len();
                    total_tokens += content.split_whitespace().count();
                }
                _ => {}
            }
        }
        
        // Calculate duration
        let duration_minutes = if messages.len() >= 2 {
            let first_time = messages.first().and_then(|m| match m {
                AssistantResponseType::Message { timestamp, .. } |
                AssistantResponseType::UserMessage { timestamp, .. } => {
                    DateTime::parse_from_rfc3339(timestamp).ok()
                }
                _ => None,
            });
            
            let last_time = messages.last().and_then(|m| match m {
                AssistantResponseType::Message { timestamp, .. } |
                AssistantResponseType::UserMessage { timestamp, .. } => {
                    DateTime::parse_from_rfc3339(timestamp).ok()
                }
                _ => None,
            });
            
            match (first_time, last_time) {
                (Some(first), Some(last)) => {
                    let duration = last.signed_duration_since(first);
                    Some(duration.num_minutes() as f64)
                }
                _ => None,
            }
        } else {
            None
        };
        
        // Find most active hour
        let most_active_hour = hourly_activity
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(hour, _)| *hour);
        
        // Get topics and entities from indexer
        let topics: Vec<String> = self.indexer.get_all_topics()
            .into_iter()
            .take(10) // Top 10 topics
            .map(|(topic, _)| topic)
            .collect();
        
        let entities: Vec<String> = vec!["url", "file", "email", "github", "function", "class"]
            .into_iter()
            .flat_map(|entity_type| self.indexer.get_entities_by_type(entity_type))
            .take(20) // Top 20 entities
            .collect();
        
        ChatStatistics {
            total_messages: messages.len(),
            messages_by_author,
            avg_message_length: if messages.is_empty() { 0 } else { total_length / messages.len() },
            total_tokens,
            duration_minutes,
            most_active_hour,
            topics,
            entities,
        }
    }
    
    /// Export to PDF (requires additional setup)
    fn export_pdf(&self, chat: &ExportedChat, path: &Path, options: &ExportOptions) -> Result<()> {
        // For now, we'll convert HTML to PDF using a simple approach
        // In production, you might want to use a proper PDF library like printpdf or wkhtmltopdf
        
        // First generate HTML
        let html_content = HtmlExporter.export_to_string(chat, options)?;
        
        // Save as HTML with .pdf extension notice
        let html_path = path.with_extension("html");
        std::fs::write(&html_path, &html_content)
            .with_context(|| format!("Failed to write HTML for PDF conversion to {:?}", html_path))?;
        
        // Note for user
        tracing::info!(
            "PDF export requires external tools. HTML version saved to {:?}. \
            You can convert it to PDF using your browser's print function or tools like wkhtmltopdf.",
            html_path
        );
        
        Ok(())
    }
    
    /// Get suggested filename for export
    pub fn suggest_filename(format: ExportFormat, chat_title: &str) -> String {
        let sanitized_title = chat_title
            .chars()
            .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
            .collect::<String>();
        
        let timestamp = Local::now().format("%Y%m%d_%H%M%S");
        
        format!("loki_chat_{}_{}.{}", 
            sanitized_title.to_lowercase(),
            timestamp,
            format.extension()
        )
    }
}

use chrono::{DateTime, Datelike, Timelike};

impl Default for ChatExporter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_suggest_filename() {
        let filename = ChatExporter::suggest_filename(ExportFormat::Json, "Test Chat!");
        assert!(filename.starts_with("loki_chat_test_chat_"));
        assert!(filename.ends_with(".json"));
    }
}