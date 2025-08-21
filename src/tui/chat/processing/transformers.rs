//! Message transformation utilities

use anyhow::{Result, Context};
use regex::Regex;
use serde::{Serialize, Deserialize};

use crate::tui::run::AssistantResponseType;

/// Message transformer for various formats and transformations
pub struct MessageTransformer {
    /// Code block regex
    code_block_regex: Regex,
    
    /// Command regex
    command_regex: Regex,
    
    /// Mention regex
    mention_regex: Regex,
    
    /// URL regex
    url_regex: Regex,
}

impl Default for MessageTransformer {
    fn default() -> Self {
        // These regex patterns are known to be valid, but we'll handle errors gracefully
        let code_block_regex = Regex::new(r"```(\w+)?\n([\s\S]*?)```")
            .expect("Valid code block regex pattern");
        let command_regex = Regex::new(r"^/(\w+)(?:\s+(.*))?$")
            .expect("Valid command regex pattern");
        let mention_regex = Regex::new(r"@(\w+)")
            .expect("Valid mention regex pattern");
        let url_regex = Regex::new(r"https?://[^\s]+")
            .expect("Valid URL regex pattern");
        
        Self {
            code_block_regex,
            command_regex,
            mention_regex,
            url_regex,
        }
    }
}

impl MessageTransformer {
    /// Create a new message transformer
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Transform message for display
    pub fn transform_for_display(&self, message: &AssistantResponseType) -> Result<DisplayMessage> {
        match message {
            AssistantResponseType::Message { id, author, message, timestamp, .. } => {
                let formatted_content = self.format_content(message)?;
                let metadata = self.extract_metadata(message)?;
                
                Ok(DisplayMessage {
                    content: formatted_content,
                    model: author.clone(),
                    timestamp: chrono::DateTime::parse_from_rfc3339(timestamp)
                        .map(|dt| dt.with_timezone(&chrono::Utc))
                        .unwrap_or_else(|_| chrono::Utc::now()),
                    metadata,
                    message_type: MessageType::Regular,
                })
            }
            AssistantResponseType::Stream { id, author, partial_content, .. } => {
                Ok(DisplayMessage {
                    content: partial_content.clone(),
                    model: author.clone(),
                    timestamp: chrono::Utc::now(),
                    metadata: MessageMetadata::default(),
                    message_type: MessageType::Stream { done: false },
                })
            }
            AssistantResponseType::Error { id, error_type, message, timestamp, .. } => {
                Ok(DisplayMessage {
                    content: format!("❌ {}: {}", error_type, message),
                    model: "system".to_string(),
                    timestamp: chrono::DateTime::parse_from_rfc3339(timestamp)
                        .map(|dt| dt.with_timezone(&chrono::Utc))
                        .unwrap_or_else(|_| chrono::Utc::now()),
                    metadata: MessageMetadata::default(),
                    message_type: MessageType::Error,
                })
            }
            _ => {
                Ok(DisplayMessage {
                    content: "Unsupported message type".to_string(),
                    model: "system".to_string(),
                    timestamp: chrono::Utc::now(),
                    metadata: MessageMetadata::default(),
                    message_type: MessageType::System,
                })
            }
        }
    }
    
    /// Format content with syntax highlighting and formatting
    fn format_content(&self, content: &str) -> Result<String> {
        let mut formatted = content.to_string();
        
        // Process code blocks
        formatted = self.format_code_blocks(&formatted)?;
        
        // Process URLs
        formatted = self.format_urls(&formatted)?;
        
        // Process mentions
        formatted = self.format_mentions(&formatted)?;
        
        Ok(formatted)
    }
    
    /// Format code blocks
    fn format_code_blocks(&self, content: &str) -> Result<String> {
        let result = self.code_block_regex.replace_all(content, |caps: &regex::Captures| {
            let language = caps.get(1).map_or("", |m| m.as_str());
            let code = caps.get(2).map_or("", |m| m.as_str());
            
            format!("\n[CODE:{}]\n{}\n[/CODE]\n", language, code)
        });
        
        Ok(result.to_string())
    }
    
    /// Format URLs
    fn format_urls(&self, content: &str) -> Result<String> {
        let result = self.url_regex.replace_all(content, |caps: &regex::Captures| {
            let url = caps.get(0).map_or("", |m| m.as_str());
            format!("[LINK:{}]", url)
        });
        
        Ok(result.to_string())
    }
    
    /// Format mentions
    fn format_mentions(&self, content: &str) -> Result<String> {
        let result = self.mention_regex.replace_all(content, |caps: &regex::Captures| {
            let mention = caps.get(1).map_or("", |m| m.as_str());
            format!("[MENTION:{}]", mention)
        });
        
        Ok(result.to_string())
    }
    
    /// Extract metadata from content
    fn extract_metadata(&self, content: &str) -> Result<MessageMetadata> {
        let code_blocks: Vec<String> = self.code_block_regex
            .captures_iter(content)
            .filter_map(|cap| cap.get(1).map(|m| m.as_str().to_string()))
            .collect();
        
        let mentions: Vec<String> = self.mention_regex
            .captures_iter(content)
            .filter_map(|cap| cap.get(1).map(|m| m.as_str().to_string()))
            .collect();
        
        let urls: Vec<String> = self.url_regex
            .find_iter(content)
            .map(|m| m.as_str().to_string())
            .collect();
        
        let has_command = self.command_regex.is_match(content);
        
        Ok(MessageMetadata {
            has_code: !code_blocks.is_empty(),
            code_languages: code_blocks,
            has_mentions: !mentions.is_empty(),
            mentions,
            has_urls: !urls.is_empty(),
            urls,
            has_command,
            word_count: content.split_whitespace().count(),
        })
    }
    
    /// Convert to markdown
    pub fn to_markdown(&self, messages: &[AssistantResponseType]) -> Result<String> {
        let mut markdown = String::new();
        
        for message in messages {
            match message {
                AssistantResponseType::Message { id, author, message, timestamp, .. } => {
                    markdown.push_str(&format!(
                        "### {} - {}\n\n{}\n\n---\n\n",
                        author,
                        timestamp,
                        message
                    ));
                }
                _ => {}
            }
        }
        
        Ok(markdown)
    }
    
    /// Convert to JSON
    pub fn to_json(&self, messages: &[AssistantResponseType]) -> Result<String> {
        let json_messages: Vec<JsonMessage> = messages
            .iter()
            .filter_map(|msg| {
                match msg {
                    AssistantResponseType::Message { id, author, message, timestamp, .. } => {
                        Some(JsonMessage {
                            content: message.clone(),
                            model: author.clone(),
                            timestamp: timestamp.clone(),
                            role: "assistant".to_string(),
                        })
                    }
                    _ => None,
                }
            })
            .collect();
        
        serde_json::to_string_pretty(&json_messages)
            .context("Failed to serialize messages to JSON")
    }
}

/// Display message format
#[derive(Debug, Clone)]
pub struct DisplayMessage {
    pub content: String,
    pub model: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub metadata: MessageMetadata,
    pub message_type: MessageType,
}

/// Message metadata
#[derive(Debug, Clone, Default)]
pub struct MessageMetadata {
    pub has_code: bool,
    pub code_languages: Vec<String>,
    pub has_mentions: bool,
    pub mentions: Vec<String>,
    pub has_urls: bool,
    pub urls: Vec<String>,
    pub has_command: bool,
    pub word_count: usize,
}

/// Message type
#[derive(Debug, Clone)]
pub enum MessageType {
    Regular,
    Stream { done: bool },
    Error,
    System,
}

/// JSON message format
#[derive(Debug, Clone, Serialize, Deserialize)]
struct JsonMessage {
    content: String,
    model: String,
    timestamp: String,
    role: String,
}