//! Chat export and import functionality
//! 
//! Provides support for exporting chat conversations to various formats
//! including Markdown, JSON, and PDF (via HTML).

use std::path::Path;
use std::collections::HashMap;
use anyhow::{Result, anyhow};
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};

use crate::tui::run::AssistantResponseType;

// Re-export ChatState type
pub use crate::tui::chat::ChatState;

/// Export format options
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExportFormat {
    Markdown,
    Json,
    Html,
    Pdf, // Generated from HTML
    PlainText,
}

/// Export options
#[derive(Debug, Clone)]
pub struct ExportOptions {
    /// Include timestamps in export
    pub include_timestamps: bool,
    
    /// Include metadata (model info, settings, etc.)
    pub include_metadata: bool,
    
    /// Include attachments
    pub include_attachments: bool,
    
    /// Include thread structure
    pub preserve_threads: bool,
    
    /// Include code syntax highlighting (for formats that support it)
    pub syntax_highlighting: bool,
    
    /// Custom CSS for HTML/PDF export
    pub custom_css: Option<String>,
    
    /// Page size for PDF
    pub pdf_page_size: String,
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            include_timestamps: true,
            include_metadata: true,
            include_attachments: true,
            preserve_threads: true,
            syntax_highlighting: true,
            custom_css: None,
            pdf_page_size: "A4".to_string(),
        }
    }
}

/// Chat exporter
pub struct ChatExporter {
    options: ExportOptions,
}

impl ChatExporter {
    pub fn new(options: ExportOptions) -> Self {
        Self { options }
    }
    
    /// Export a chat to the specified format
    pub async fn export_chat(
        &self,
        chat: &ChatState,
        format: ExportFormat,
        output_path: &Path,
    ) -> Result<()> {
        match format {
            ExportFormat::Markdown => self.export_markdown(chat, output_path).await,
            ExportFormat::Json => self.export_json(chat, output_path).await,
            ExportFormat::Html => self.export_html(chat, output_path).await,
            ExportFormat::Pdf => self.export_pdf(chat, output_path).await,
            ExportFormat::PlainText => self.export_plain_text(chat, output_path).await,
        }
    }
    
    /// Export to Markdown format
    async fn export_markdown(&self, chat: &ChatState, output_path: &Path) -> Result<()> {
        let mut content = String::new();
        
        // Header
        content.push_str(&format!("# {}\n\n", chat.title));
        
        if self.options.include_metadata {
            content.push_str("## Metadata\n\n");
            content.push_str(&format!("- **Created**: {}\n", chat.created_at.format("%Y-%m-%d %H:%M:%S UTC")));
            content.push_str(&format!("- **Last Activity**: {}\n", chat.last_activity.format("%Y-%m-%d %H:%M:%S UTC")));
            content.push_str(&format!("- **Messages**: {}\n", chat.messages.len()));
            content.push_str("\n");
        }
        
        // Messages
        content.push_str("## Conversation\n\n");
        
        for (idx, message) in chat.messages.iter().enumerate() {
            match message {
                AssistantResponseType::Message { message: text, .. } => {
                    content.push_str(&format!("### Message {}\n", idx + 1));
                    if self.options.include_timestamps {
                        content.push_str(&format!("*{}*\n\n", chrono::Utc::now().format("%H:%M:%S")));
                    }
                    content.push_str(&format!("{}\n\n", text));
                }
                AssistantResponseType::Code { language, code, .. } => {
                    content.push_str(&format!("### Code Block {}\n", idx + 1));
                    content.push_str(&format!("```{}\n{}\n```\n\n", language, code));
                }
                AssistantResponseType::Error { message, .. } => {
                    content.push_str(&format!("### Error {}\n", idx + 1));
                    content.push_str(&format!("> ⚠️ {}\n\n", message));
                }
                AssistantResponseType::ToolUse { tool_name, parameters, result, .. } => {
                    content.push_str(&format!("### Tool Use: {}\n", tool_name));
                    content.push_str(&format!("**Parameters**: `{}`\n", parameters));
                    if let Some(res) = result {
                        content.push_str(&format!("**Result**: {}\n", res));
                    }
                    content.push_str("\n");
                }
                AssistantResponseType::Stream { partial_content, .. } => {
                    content.push_str(&format!("### Streaming Response {}\n", idx + 1));
                    content.push_str(&format!("{}\n\n", partial_content));
                }
                _ => {}
            }
        }
        
        // Write to file
        tokio::fs::write(output_path, content).await?;
        Ok(())
    }
    
    /// Export to JSON format
    async fn export_json(&self, chat: &ChatState, output_path: &Path) -> Result<()> {
        #[derive(Serialize)]
        struct ExportedChat {
            id: String,
            title: String,
            created_at: DateTime<Utc>,
            last_activity: DateTime<Utc>,
            messages: Vec<ExportedMessage>,
            metadata: Option<HashMap<String, String>>,
        }
        
        #[derive(Serialize)]
        struct ExportedMessage {
            #[serde(rename = "type")]
            message_type: String,
            content: serde_json::Value,
            timestamp: Option<DateTime<Utc>>,
        }
        
        let mut messages = Vec::new();
        
        for message in &chat.messages {
            let (message_type, content) = match message {
                AssistantResponseType::Message { message: text, .. } => {
                    ("message".to_string(), serde_json::json!({ "text": text }))
                }
                AssistantResponseType::Code { language, code, .. } => {
                    ("code".to_string(), serde_json::json!({
                        "language": language,
                        "code": code
                    }))
                }
                AssistantResponseType::Error { message: error, .. } => {
                    ("error".to_string(), serde_json::json!({ "error": error }))
                }
                AssistantResponseType::ToolUse { tool_name, parameters, result, .. } => {
                    ("tool_use".to_string(), serde_json::json!({
                        "tool": tool_name,
                        "parameters": parameters,
                        "result": result
                    }))
                }
                _ => continue,
            };
            
            messages.push(ExportedMessage {
                message_type,
                content,
                timestamp: if self.options.include_timestamps {
                    Some(chrono::Utc::now())
                } else {
                    None
                },
            });
        }
        
        let exported = ExportedChat {
            id: chat.id.clone(),
            title: chat.title.clone(),
            created_at: chat.created_at,
            last_activity: chat.last_activity,
            messages,
            metadata: if self.options.include_metadata {
                Some(chat.session_context.clone())
            } else {
                None
            },
        };
        
        let json = serde_json::to_string_pretty(&exported)?;
        tokio::fs::write(output_path, json).await?;
        Ok(())
    }
    
    /// Export to HTML format
    async fn export_html(&self, chat: &ChatState, output_path: &Path) -> Result<()> {
        let mut html = String::new();
        
        // HTML header
        html.push_str("<!DOCTYPE html>\n<html>\n<head>\n");
        html.push_str(&format!("<title>{}</title>\n", html_escape(&chat.title)));
        html.push_str("<meta charset=\"utf-8\">\n");
        html.push_str("<style>\n");
        
        // Default CSS
        if let Some(custom_css) = &self.options.custom_css {
            html.push_str(custom_css);
        } else {
            html.push_str(include_str!("default_export_styles.css"));
        }
        
        html.push_str("</style>\n</head>\n<body>\n");
        
        // Content
        html.push_str(&format!("<h1>{}</h1>\n", html_escape(&chat.title)));
        
        if self.options.include_metadata {
            html.push_str("<div class=\"metadata\">\n");
            html.push_str(&format!("<p><strong>Created:</strong> {}</p>\n", chat.created_at.format("%Y-%m-%d %H:%M:%S UTC")));
            html.push_str(&format!("<p><strong>Last Activity:</strong> {}</p>\n", chat.last_activity.format("%Y-%m-%d %H:%M:%S UTC")));
            html.push_str(&format!("<p><strong>Messages:</strong> {}</p>\n", chat.messages.len()));
            html.push_str("</div>\n");
        }
        
        html.push_str("<div class=\"conversation\">\n");
        
        for message in &chat.messages {
            match message {
                AssistantResponseType::Message { message: text, .. } => {
                    html.push_str("<div class=\"message\">\n");
                    if self.options.include_timestamps {
                        html.push_str(&format!("<div class=\"timestamp\">{}</div>\n", chrono::Utc::now().format("%H:%M:%S")));
                    }
                    html.push_str(&format!("<div class=\"content\">{}</div>\n", markdown_to_html(text)));
                    html.push_str("</div>\n");
                }
                AssistantResponseType::Code { language, code, .. } => {
                    html.push_str("<div class=\"code-block\">\n");
                    html.push_str(&format!("<pre><code class=\"language-{}\">{}</code></pre>\n", 
                        language,
                        html_escape(code)
                    ));
                    html.push_str("</div>\n");
                }
                AssistantResponseType::Error { message: error, .. } => {
                    html.push_str("<div class=\"error\">\n");
                    html.push_str(&format!("<div class=\"content\">⚠️ {}</div>\n", html_escape(error)));
                    html.push_str("</div>\n");
                }
                _ => {}
            }
        }
        
        html.push_str("</div>\n</body>\n</html>");
        
        tokio::fs::write(output_path, html).await?;
        Ok(())
    }
    
    /// Export to PDF format (via HTML)
    async fn export_pdf(&self, chat: &ChatState, output_path: &Path) -> Result<()> {
        // First generate HTML
        let html_path = output_path.with_extension("html");
        self.export_html(chat, &html_path).await?;
        
        // Note: PDF generation would require a library like wkhtmltopdf or headless chrome
        // For now, we just indicate that HTML was generated
        Err(anyhow!("PDF export requires external tool. HTML file generated at: {:?}", html_path))
    }
    
    /// Export to plain text format
    async fn export_plain_text(&self, chat: &ChatState, output_path: &Path) -> Result<()> {
        let mut content = String::new();
        
        content.push_str(&format!("{}\n", chat.title));
        content.push_str(&format!("{}\n\n", "=".repeat(chat.title.len())));
        
        for message in &chat.messages {
            match message {
                AssistantResponseType::Message { message: text, .. } => {
                    content.push_str(&format!("{}\n\n", text));
                }
                AssistantResponseType::Code { code, .. } => {
                    content.push_str(&format!("{}\n\n", code));
                }
                AssistantResponseType::Error { message: error, .. } => {
                    content.push_str(&format!("Error: {}\n\n", error));
                }
                _ => {}
            }
        }
        
        tokio::fs::write(output_path, content).await?;
        Ok(())
    }
}

/// Chat importer
pub struct ChatImporter;

impl ChatImporter {
    /// Import a chat from JSON
    pub async fn import_json(path: &Path) -> Result<ChatState> {
        let content = tokio::fs::read_to_string(path).await?;
        let imported: ImportedChat = serde_json::from_str(&content)?;
        
        // Generate a numeric ID from the imported string ID or use a default
        let numeric_id = imported.id.parse::<usize>().unwrap_or_else(|_| {
            // If not a number, use hash of the ID as a fallback
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut hasher = DefaultHasher::new();
            imported.id.hash(&mut hasher);
            hasher.finish() as usize
        });
        let mut chat = ChatState::new(numeric_id, imported.title);
        chat.id = imported.id;
        chat.created_at = imported.created_at;
        chat.last_activity = imported.last_activity;
        
        // Convert messages
        for msg in imported.messages {
            let message = match msg.message_type.as_str() {
                "message" => {
                    if let Some(text) = msg.content.get("text").and_then(|v| v.as_str()) {
                        AssistantResponseType::Message {
                            id: uuid::Uuid::new_v4().to_string(),
                            author: "Assistant".to_string(),
                            message: text.to_string(),
                            timestamp: msg.timestamp.map(|t| t.format("%H:%M:%S").to_string()).unwrap_or_else(|| chrono::Utc::now().format("%H:%M:%S").to_string()),
                            is_editing: false,
                            edit_history: Vec::new(),
                            streaming_state: crate::tui::run::StreamingState::Complete,
                            metadata: crate::tui::run::MessageMetadata::default(),
                        }
                    } else {
                        continue;
                    }
                }
                "code" => {
                    let code = msg.content.get("code").and_then(|v| v.as_str()).unwrap_or("").to_string();
                    let language = msg.content.get("language").and_then(|v| v.as_str()).unwrap_or("").to_string();
                    
                    AssistantResponseType::Code {
                        id: uuid::Uuid::new_v4().to_string(),
                        author: "Assistant".to_string(),
                        language,
                        code,
                        timestamp: msg.timestamp.map(|t| t.format("%H:%M:%S").to_string()).unwrap_or_else(|| chrono::Utc::now().format("%H:%M:%S").to_string()),
                        is_editing: false,
                        edit_history: Vec::new(),
                        metadata: crate::tui::run::MessageMetadata::default(),
                    }
                }
                "error" => {
                    if let Some(error) = msg.content.get("error").and_then(|v| v.as_str()) {
                        AssistantResponseType::Error {
                            id: uuid::Uuid::new_v4().to_string(),
                            error_type: "Error".to_string(),
                            message: error.to_string(),
                            timestamp: msg.timestamp.map(|t| t.format("%H:%M:%S").to_string()).unwrap_or_else(|| chrono::Utc::now().format("%H:%M:%S").to_string()),
                            metadata: crate::tui::run::MessageMetadata::default(),
                        }
                    } else {
                        continue;
                    }
                }
                _ => continue,
            };
            
            chat.messages.push(message);
        }
        
        if let Some(metadata) = imported.metadata {
            chat.session_context = metadata;
        }
        
        Ok(chat)
    }
    
    /// Import a chat from Markdown (basic support)
    pub async fn import_markdown(path: &Path) -> Result<ChatState> {
        let content = tokio::fs::read_to_string(path).await?;
        let lines: Vec<&str> = content.lines().collect();
        
        if lines.is_empty() {
            return Err(anyhow!("Empty markdown file"));
        }
        
        // Extract title from first heading
        let title = lines[0]
            .trim_start_matches('#')
            .trim()
            .to_string();
        
        // Generate a unique ID for the imported chat
        let numeric_id = chrono::Utc::now().timestamp() as usize;
        let mut chat = ChatState::new(numeric_id, title);
        let mut current_message = String::new();
        let mut in_code_block = false;
        let mut code_language = None;
        
        for line in lines.iter().skip(1) {
            if line.starts_with("```") {
                if in_code_block {
                    // End of code block
                    let lang = code_language.take();
                    chat.messages.push(AssistantResponseType::Code {
                        id: uuid::Uuid::new_v4().to_string(),
                        author: "Assistant".to_string(),
                        language: lang.unwrap_or_default(),
                        code: current_message.clone(),
                        timestamp: chrono::Utc::now().format("%H:%M:%S").to_string(),
                        is_editing: false,
                        edit_history: Vec::new(),
                        metadata: crate::tui::run::MessageMetadata::default(),
                    });
                    current_message.clear();
                    in_code_block = false;
                } else {
                    // Start of code block
                    if !current_message.is_empty() {
                        chat.messages.push(AssistantResponseType::Message {
                            id: uuid::Uuid::new_v4().to_string(),
                            author: "Assistant".to_string(),
                            message: current_message.clone(),
                            timestamp: chrono::Utc::now().format("%H:%M:%S").to_string(),
                            is_editing: false,
                            edit_history: Vec::new(),
                            streaming_state: crate::tui::run::StreamingState::Complete,
                            metadata: crate::tui::run::MessageMetadata::default(),
                        });
                        current_message.clear();
                    }
                    code_language = Some(line[3..].trim().to_string());
                    in_code_block = true;
                }
            } else if in_code_block {
                if !current_message.is_empty() {
                    current_message.push('\n');
                }
                current_message.push_str(line);
            } else if line.starts_with("###") || line.starts_with("##") {
                // New section - save current message
                if !current_message.is_empty() {
                    chat.messages.push(AssistantResponseType::Message {
                        id: uuid::Uuid::new_v4().to_string(),
                        author: "Assistant".to_string(),
                        message: current_message.clone(),
                        timestamp: chrono::Utc::now().format("%H:%M:%S").to_string(),
                        is_editing: false,
                        edit_history: Vec::new(),
                        streaming_state: crate::tui::run::StreamingState::Complete,
                        metadata: crate::tui::run::MessageMetadata::default(),
                    });
                    current_message.clear();
                }
            } else if !line.trim().is_empty() {
                if !current_message.is_empty() {
                    current_message.push('\n');
                }
                current_message.push_str(line);
            }
        }
        
        // Add any remaining content
        if !current_message.is_empty() {
            chat.messages.push(AssistantResponseType::Message {
                id: uuid::Uuid::new_v4().to_string(),
                author: "Assistant".to_string(),
                message: current_message,
                timestamp: chrono::Utc::now().format("%H:%M:%S").to_string(),
                is_editing: false,
                edit_history: Vec::new(),
                streaming_state: crate::tui::run::StreamingState::Complete,
                metadata: crate::tui::run::MessageMetadata::default(),
            });
        }
        
        Ok(chat)
    }
}

#[derive(Deserialize)]
struct ImportedChat {
    id: String,
    title: String,
    created_at: DateTime<Utc>,
    last_activity: DateTime<Utc>,
    messages: Vec<ImportedMessage>,
    metadata: Option<HashMap<String, String>>,
}

#[derive(Deserialize)]
struct ImportedMessage {
    #[serde(rename = "type")]
    message_type: String,
    content: serde_json::Value,
    timestamp: Option<DateTime<Utc>>,
}

/// HTML escape helper
fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

/// Simple markdown to HTML converter
fn markdown_to_html(markdown: &str) -> String {
    // This is a very basic implementation
    // In production, use a proper markdown parser
    let mut html = String::new();
    
    for line in markdown.lines() {
        if line.starts_with("# ") {
            html.push_str(&format!("<h1>{}</h1>\n", html_escape(&line[2..])));
        } else if line.starts_with("## ") {
            html.push_str(&format!("<h2>{}</h2>\n", html_escape(&line[3..])));
        } else if line.starts_with("- ") {
            html.push_str(&format!("<li>{}</li>\n", html_escape(&line[2..])));
        } else if line.starts_with("**") && line.ends_with("**") && line.len() > 4 {
            html.push_str(&format!("<strong>{}</strong>\n", html_escape(&line[2..line.len()-2])));
        } else if !line.trim().is_empty() {
            html.push_str(&format!("<p>{}</p>\n", html_escape(line)));
        }
    }
    
    html
}