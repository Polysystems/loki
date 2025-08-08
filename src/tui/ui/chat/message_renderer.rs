//! Rich message rendering system for the chat interface
//! 
//! Supports various message types with specialized rendering including
//! markdown formatting, code highlighting, and interactive elements.

use ratatui::{
    style::{Color, Style},
};
use std::collections::HashMap;
use chrono::{DateTime, Utc};

use super::{
    markdown_parser::{MarkdownParser},
    syntax_highlighter::{SyntaxHighlighter, Language},
    theme_engine::ChatTheme,
};

/// Rich message type with metadata and formatting
#[derive(Debug, Clone)]
pub struct RichMessage {
    /// Unique message ID
    pub id: String,
    
    /// Message author
    pub author: String,
    
    /// Message content
    pub content: String,
    
    /// Message type for specialized rendering
    pub message_type: MessageType,
    
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    
    /// Metadata for rendering hints
    pub metadata: MessageMetadata,
    
    /// Whether message is being edited
    pub is_editing: bool,
    
    /// Whether message is selected
    pub is_selected: bool,
    
    /// Streaming state
    pub streaming_state: Option<StreamingInfo>,
}

/// Message types for specialized rendering
#[derive(Debug, Clone)]
pub enum MessageType {
    Text,
    Code { language: Option<Language> },
    ToolExecution { tool_name: String, status: ToolStatus },
    Workflow { workflow_id: String, steps: Vec<WorkflowStep> },
    Error { error_type: String },
    Data { format: DataFormat },
    System,
    Attachment { file_name: String, file_size: u64, mime_type: String },
}

/// Tool execution status
#[derive(Debug, Clone, PartialEq)]
pub enum ToolStatus {
    Pending,
    Running { progress: f32 },
    Success,
    Failed { error: String },
}

/// Workflow step information
#[derive(Debug, Clone)]
pub struct WorkflowStep {
    pub name: String,
    pub status: StepStatus,
    pub description: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum StepStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Skipped,
}

/// Data format for structured data messages
#[derive(Debug, Clone, PartialEq)]
pub enum DataFormat {
    Json,
    Table,
    List,
    Tree,
}

/// Message metadata for rendering hints
#[derive(Debug, Clone)]
pub struct MessageMetadata {
    /// Collapsed sections
    pub collapsed_sections: Vec<String>,
    
    /// Highlighted lines (for code)
    pub highlighted_lines: Vec<usize>,
    
    /// Custom styles
    pub custom_styles: HashMap<String, Style>,
    
    /// Action buttons
    pub actions: Vec<MessageAction>,
    
    /// Associated media content IDs
    pub media_ids: Vec<String>,
}

/// Interactive message actions
#[derive(Debug, Clone)]
pub struct MessageAction {
    pub id: String,
    pub label: String,
    pub icon: String,
    pub enabled: bool,
}

/// Streaming information
#[derive(Debug, Clone)]
pub struct StreamingInfo {
    pub started_at: DateTime<Utc>,
    pub chunks_received: usize,
    pub is_complete: bool,
    pub typing_indicator: bool,
}

/// Message renderer
#[derive(Clone)]
pub struct MessageRenderer {
    markdown_parser: MarkdownParser,
    syntax_highlighter: SyntaxHighlighter,
}

impl MessageRenderer {
    pub fn new() -> Self {
        Self {
            markdown_parser: MarkdownParser::new(),
            syntax_highlighter: SyntaxHighlighter::new(),
        }
    }
    
    /// Render a rich message to lines of text
    pub fn render(&self, message: &RichMessage, theme: &ChatTheme) -> Vec<String> {
        self.render_message(message)
    }
    
    /// Render a rich message to lines of text (new method name for compatibility)
    pub fn render_message(&self, message: &RichMessage) -> Vec<String> {
        let mut lines = Vec::new();
        
        // Add message header
        lines.extend(self.render_header(message));
        
        // Render content based on message type
        match &message.message_type {
            MessageType::Text => {
                lines.extend(self.render_text_message(&message.content));
            }
            MessageType::Code { language } => {
                lines.extend(self.render_code_message(&message.content, language.as_ref()));
            }
            MessageType::ToolExecution { tool_name, status } => {
                lines.extend(self.render_tool_message(tool_name, status, &message.content));
            }
            MessageType::Workflow { workflow_id, steps } => {
                lines.extend(self.render_workflow_message(workflow_id, steps));
            }
            MessageType::Error { error_type } => {
                lines.extend(self.render_error_message(error_type, &message.content));
            }
            MessageType::Data { format } => {
                lines.extend(self.render_data_message(&message.content, format));
            }
            MessageType::System => {
                lines.extend(self.render_system_message(&message.content));
            }
            MessageType::Attachment { file_name, file_size, mime_type } => {
                lines.extend(self.render_attachment_message(file_name, *file_size, mime_type, &message.content));
            }
        }
        
        // Add streaming indicator if applicable
        if let Some(streaming) = &message.streaming_state {
            lines.push(self.render_streaming_indicator(streaming));
        }
        
        // Add action buttons if any
        if !message.metadata.actions.is_empty() {
            lines.push(self.render_actions(&message.metadata.actions));
        }
        
        lines
    }
    
    /// Render message header with author and timestamp
    fn render_header(&self, message: &RichMessage) -> Vec<String> {
        let timestamp = message.timestamp.format("%H:%M:%S").to_string();
        
        let mut header = format!("{} ", message.author);
        if message.is_editing {
            header.push_str("(editing) ");
        }
        
        vec![
            format!("┌─ {} • {} {}",
                header,
                timestamp,
                if message.is_selected { "◆" } else { "" }
            )
        ]
    }
    
    /// Render text message with markdown formatting
    fn render_text_message(&self, content: &str) -> Vec<String> {
        let formatted = self.markdown_parser.parse(content);
        let mut lines = Vec::new();
        
        for line in formatted.lines {
            let mut rendered_line = String::new();
            for span in line.spans {
                rendered_line.push_str(&span.content);
            }
            lines.push(format!("│ {}", rendered_line));
        }
        
        lines.push("└─".to_string());
        lines
    }
    
    /// Render code message with syntax highlighting
    fn render_code_message(&self, code: &str, language: Option<&Language>) -> Vec<String> {
        let mut lines = vec![
            format!("│ ```{}", language.map(|l| format!("{:?}", l)).unwrap_or_default()),
        ];
        
        let highlighted = if let Some(lang) = language {
            self.syntax_highlighter.highlight(code, lang)
        } else {
            code.lines().map(|l| l.to_string()).collect()
        };
        
        for line in highlighted {
            lines.push(format!("│ {}", line));
        }
        
        lines.push("│ ```".to_string());
        lines.push("└─".to_string());
        lines
    }
    
    /// Render tool execution message
    fn render_tool_message(&self, tool_name: &str, status: &ToolStatus, output: &str) -> Vec<String> {
        let mut lines = Vec::new();
        
        let status_icon = match status {
            ToolStatus::Pending => "⏳",
            ToolStatus::Running { progress } => {
                let bar = self.render_progress_bar(*progress, 20);
                lines.push(format!("│ 🔧 {} {}", tool_name, bar));
                return lines;
            }
            ToolStatus::Success => "✅",
            ToolStatus::Failed { .. } => "❌",
        };
        
        lines.push(format!("│ {} {}", status_icon, tool_name));
        
        // Add output
        for line in output.lines() {
            lines.push(format!("│   {}", line));
        }
        
        lines.push("└─".to_string());
        lines
    }
    
    /// Render workflow message
    fn render_workflow_message(&self, workflow_id: &str, steps: &[WorkflowStep]) -> Vec<String> {
        let mut lines = vec![
            format!("│ 📋 Workflow: {}", workflow_id),
            "│".to_string(),
        ];
        
        for (i, step) in steps.iter().enumerate() {
            let status_icon = match step.status {
                StepStatus::Pending => "⏳",
                StepStatus::Running => "🔄",
                StepStatus::Completed => "✅",
                StepStatus::Failed => "❌",
                StepStatus::Skipped => "⏭️",
            };
            
            let connector = if i == steps.len() - 1 { "└" } else { "├" };
            lines.push(format!("│ {}─ {} {}", connector, status_icon, step.name));
            
            if let Some(desc) = &step.description {
                let prefix = if i == steps.len() - 1 { " " } else { "│" };
                lines.push(format!("│ {}   {}", prefix, desc));
            }
        }
        
        lines.push("└─".to_string());
        lines
    }
    
    /// Render error message
    fn render_error_message(&self, error_type: &str, content: &str) -> Vec<String> {
        let mut lines = vec![
            format!("│ ❌ Error: {}", error_type),
            "│".to_string(),
        ];
        
        for line in content.lines() {
            lines.push(format!("│ {}", line));
        }
        
        lines.push("└─".to_string());
        lines
    }
    
    /// Render data message
    fn render_data_message(&self, content: &str, format: &DataFormat) -> Vec<String> {
        match format {
            DataFormat::Json => self.render_json_data(content),
            DataFormat::Table => self.render_table_data(content),
            DataFormat::List => self.render_list_data(content),
            DataFormat::Tree => self.render_tree_data(content),
        }
    }
    
    /// Render JSON data with syntax highlighting
    fn render_json_data(&self, json: &str) -> Vec<String> {
        let mut lines = vec!["│ 📊 JSON Data:".to_string()];
        
        // Pretty print and highlight JSON
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(json) {
            if let Ok(pretty) = serde_json::to_string_pretty(&value) {
                for line in pretty.lines() {
                    lines.push(format!("│   {}", line));
                }
            }
        } else {
            lines.push(format!("│   {}", json));
        }
        
        lines.push("└─".to_string());
        lines
    }
    
    /// Render table data
    fn render_table_data(&self, content: &str) -> Vec<String> {
        // Implementation for table rendering
        vec![
            "│ 📊 Table:".to_string(),
            format!("│ {}", content),
            "└─".to_string(),
        ]
    }
    
    /// Render list data
    fn render_list_data(&self, content: &str) -> Vec<String> {
        let mut lines = vec!["│ 📋 List:".to_string()];
        
        for item in content.lines() {
            lines.push(format!("│ • {}", item));
        }
        
        lines.push("└─".to_string());
        lines
    }
    
    /// Render tree data
    fn render_tree_data(&self, content: &str) -> Vec<String> {
        let mut lines = vec!["│ 🌳 Tree:".to_string()];
        
        for line in content.lines() {
            lines.push(format!("│ {}", line));
        }
        
        lines.push("└─".to_string());
        lines
    }
    
    /// Render system message
    fn render_system_message(&self, content: &str) -> Vec<String> {
        let mut lines = vec!["│ 🔧 System:".to_string()];
        
        for line in content.lines() {
            lines.push(format!("│ {}", line));
        }
        
        lines.push("└─".to_string());
        lines
    }
    
    /// Render attachment message
    fn render_attachment_message(&self, file_name: &str, file_size: u64, mime_type: &str, content: &str) -> Vec<String> {
        let mut lines = vec!["│ 📎 File Attachment:".to_string()];
        
        // Format file size
        let size_str = self.format_file_size(file_size);
        
        // Determine file icon based on mime type
        let icon = self.get_file_icon(mime_type);
        
        lines.push(format!("│ {} {} ({})", icon, file_name, size_str));
        lines.push(format!("│ Type: {}", mime_type));
        
        // If there's preview content, show it
        if !content.is_empty() {
            lines.push("│".to_string());
            lines.push("│ Preview:".to_string());
            let preview_lines: Vec<&str> = content.lines().take(5).collect();
            for line in preview_lines {
                lines.push(format!("│   {}", line));
            }
            if content.lines().count() > 5 {
                lines.push("│   ...".to_string());
            }
        }
        
        lines.push("└─".to_string());
        lines
    }
    
    /// Format file size for display
    fn format_file_size(&self, size: u64) -> String {
        const UNITS: &[&str] = &["B", "KB", "MB", "GB"];
        let mut size = size as f64;
        let mut unit_index = 0;
        
        while size >= 1024.0 && unit_index < UNITS.len() - 1 {
            size /= 1024.0;
            unit_index += 1;
        }
        
        if unit_index == 0 {
            format!("{} {}", size as u64, UNITS[unit_index])
        } else {
            format!("{:.1} {}", size, UNITS[unit_index])
        }
    }
    
    /// Get file icon based on mime type
    fn get_file_icon(&self, mime_type: &str) -> &'static str {
        if mime_type.starts_with("image/") {
            "🖼️"
        } else if mime_type.starts_with("text/") {
            "📄"
        } else if mime_type.starts_with("application/pdf") {
            "📑"
        } else if mime_type.starts_with("application/zip") || mime_type.contains("compressed") {
            "📦"
        } else if mime_type.starts_with("video/") {
            "🎬"
        } else if mime_type.starts_with("audio/") {
            "🎵"
        } else {
            "📎"
        }
    }
    
    /// Render streaming indicator
    fn render_streaming_indicator(&self, streaming: &StreamingInfo) -> String {
        if streaming.typing_indicator {
            let dots = ".".repeat((streaming.chunks_received % 4) as usize);
            format!("│ 💭 Typing{:<3}", dots)
        } else {
            format!("│ ⏳ Streaming... ({} chunks)", streaming.chunks_received)
        }
    }
    
    /// Render action buttons
    fn render_actions(&self, actions: &[MessageAction]) -> String {
        let buttons: Vec<String> = actions.iter()
            .map(|action| {
                if action.enabled {
                    format!("[{} {}]", action.icon, action.label)
                } else {
                    format!("({} {})", action.icon, action.label)
                }
            })
            .collect();
        
        format!("│ {}", buttons.join(" "))
    }
    
    /// Render progress bar
    fn render_progress_bar(&self, progress: f32, width: usize) -> String {
        let filled = (progress * width as f32) as usize;
        let empty = width.saturating_sub(filled);
        
        format!("[{}{}] {:.0}%",
            "█".repeat(filled),
            "░".repeat(empty),
            progress * 100.0
        )
    }
    
    /// Get author color
    fn get_author_color(&self, author: &str) -> Color {
        match author.to_lowercase().as_str() {
            "you" | "user" => Color::Cyan,
            "loki" | "assistant" => Color::Green,
            "system" => Color::Yellow,
            _ => Color::White,
        }
    }
}

impl Default for MessageMetadata {
    fn default() -> Self {
        Self {
            collapsed_sections: Vec::new(),
            highlighted_lines: Vec::new(),
            custom_styles: HashMap::new(),
            actions: Vec::new(),
            media_ids: Vec::new(),
        }
    }
}