//! Export format definitions and implementations

use std::path::Path;
use anyhow::{Result, Context};
use serde::{Serialize, Deserialize};

use super::{ExportedChat, ExportedMessage};

/// Supported export formats
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExportFormat {
    /// JSON format (structured data)
    Json,
    /// Markdown format (human-readable)
    Markdown,
    /// Plain text format
    Text,
    /// HTML format (web-viewable)
    Html,
    /// PDF format (printable)
    Pdf,
    /// CSV format (spreadsheet-compatible)
    Csv,
}

impl ExportFormat {
    /// Get file extension for format
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Json => "json",
            Self::Markdown => "md",
            Self::Text => "txt",
            Self::Html => "html",
            Self::Pdf => "pdf",
            Self::Csv => "csv",
        }
    }
    
    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Json => "JSON",
            Self::Markdown => "Markdown",
            Self::Text => "Plain Text",
            Self::Html => "HTML",
            Self::Pdf => "PDF",
            Self::Csv => "CSV",
        }
    }
}

/// Export options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportOptions {
    /// Include metadata in export
    pub include_metadata: bool,
    
    /// Include timestamps
    pub include_timestamps: bool,
    
    /// Include message IDs
    pub include_ids: bool,
    
    /// Include statistics
    pub include_statistics: bool,
    
    /// Pretty print (for JSON)
    pub pretty_print: bool,
    
    /// Theme for HTML/PDF
    pub theme: ExportTheme,
    
    /// Date format
    pub date_format: String,
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            include_metadata: true,
            include_timestamps: true,
            include_ids: false,
            include_statistics: true,
            pretty_print: true,
            theme: ExportTheme::default(),
            date_format: "%Y-%m-%d %H:%M:%S".to_string(),
        }
    }
}

/// Export theme for HTML/PDF
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportTheme {
    pub font_family: String,
    pub font_size: String,
    pub user_color: String,
    pub assistant_color: String,
    pub system_color: String,
    pub background_color: String,
    pub text_color: String,
}

impl Default for ExportTheme {
    fn default() -> Self {
        Self {
            font_family: "system-ui, -apple-system, sans-serif".to_string(),
            font_size: "14px".to_string(),
            user_color: "#0066cc".to_string(),
            assistant_color: "#00aa00".to_string(),
            system_color: "#666666".to_string(),
            background_color: "#ffffff".to_string(),
            text_color: "#333333".to_string(),
        }
    }
}

/// Format-specific exporters
pub trait Exporter {
    /// Export chat to string
    fn export_to_string(&self, chat: &ExportedChat, options: &ExportOptions) -> Result<String>;
    
    /// Export chat to file
    fn export_to_file(&self, chat: &ExportedChat, path: &Path, options: &ExportOptions) -> Result<()> {
        let content = self.export_to_string(chat, options)?;
        std::fs::write(path, content)
            .with_context(|| format!("Failed to write export to {:?}", path))
    }
}

/// JSON exporter
pub struct JsonExporter;

impl Exporter for JsonExporter {
    fn export_to_string(&self, chat: &ExportedChat, options: &ExportOptions) -> Result<String> {
        if options.pretty_print {
            serde_json::to_string_pretty(chat)
        } else {
            serde_json::to_string(chat)
        }
        .context("Failed to serialize chat to JSON")
    }
}

/// Markdown exporter
pub struct MarkdownExporter;

impl Exporter for MarkdownExporter {
    fn export_to_string(&self, chat: &ExportedChat, options: &ExportOptions) -> Result<String> {
        let mut output = String::new();
        
        // Header
        output.push_str(&format!("# {}\n\n", chat.metadata.chat_title));
        
        // Metadata
        if options.include_metadata {
            output.push_str("## Metadata\n\n");
            output.push_str(&format!("- **Exported**: {}\n", chat.metadata.exported_at.format(&options.date_format)));
            output.push_str(&format!("- **Messages**: {}\n", chat.metadata.message_count));
            if let Some((start, end)) = &chat.metadata.date_range {
                output.push_str(&format!("- **Date Range**: {} to {}\n", 
                    start.format(&options.date_format),
                    end.format(&options.date_format)
                ));
            }
            output.push_str("\n---\n\n");
        }
        
        // Messages
        output.push_str("## Conversation\n\n");
        
        for msg in &chat.messages {
            // Author and timestamp
            if options.include_timestamps {
                output.push_str(&format!("**{}** - *{}*\n\n", 
                    msg.author, 
                    msg.timestamp.format(&options.date_format)
                ));
            } else {
                output.push_str(&format!("**{}**\n\n", msg.author));
            }
            
            // Message content
            output.push_str(&msg.content);
            output.push_str("\n\n");
            
            // Metadata
            if let Some(metadata) = &msg.metadata {
                if let Some(model) = &metadata.model {
                    output.push_str(&format!("> Model: {}\n", model));
                }
                if let Some(tokens) = metadata.tokens {
                    output.push_str(&format!("> Tokens: {}\n", tokens));
                }
                if let Some(tools) = &metadata.tools_used {
                    output.push_str(&format!("> Tools: {}\n", tools.join(", ")));
                }
                output.push('\n');
            }
            
            output.push_str("---\n\n");
        }
        
        // Statistics
        if options.include_statistics {
            output.push_str("## Statistics\n\n");
            output.push_str(&format!("- **Total Messages**: {}\n", chat.statistics.total_messages));
            output.push_str(&format!("- **Average Message Length**: {} characters\n", chat.statistics.avg_message_length));
            output.push_str(&format!("- **Estimated Tokens**: {}\n", chat.statistics.total_tokens));
            
            if let Some(duration) = chat.statistics.duration_minutes {
                output.push_str(&format!("- **Duration**: {:.1} minutes\n", duration));
            }
            
            if !chat.statistics.topics.is_empty() {
                output.push_str(&format!("- **Topics**: {}\n", chat.statistics.topics.join(", ")));
            }
        }
        
        Ok(output)
    }
}

/// Plain text exporter
pub struct TextExporter;

impl Exporter for TextExporter {
    fn export_to_string(&self, chat: &ExportedChat, options: &ExportOptions) -> Result<String> {
        let mut output = String::new();
        
        // Header
        output.push_str(&format!("{}\n", chat.metadata.chat_title));
        output.push_str(&format!("{}\n\n", "=".repeat(chat.metadata.chat_title.len())));
        
        // Messages
        for msg in &chat.messages {
            if options.include_timestamps {
                output.push_str(&format!("[{}] ", msg.timestamp.format(&options.date_format)));
            }
            output.push_str(&format!("{}: {}\n\n", msg.author.to_uppercase(), msg.content));
        }
        
        Ok(output)
    }
}

/// HTML exporter
pub struct HtmlExporter;

impl Exporter for HtmlExporter {
    fn export_to_string(&self, chat: &ExportedChat, options: &ExportOptions) -> Result<String> {
        let mut output = String::new();
        
        // HTML header
        output.push_str("<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n");
        output.push_str("    <meta charset=\"UTF-8\">\n");
        output.push_str("    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n");
        output.push_str(&format!("    <title>{}</title>\n", html_escape(&chat.metadata.chat_title)));
        
        // CSS styles
        output.push_str("    <style>\n");
        output.push_str(&format!("        body {{ font-family: {}; font-size: {}; background-color: {}; color: {}; max-width: 800px; margin: 0 auto; padding: 20px; }}\n", 
            options.theme.font_family, options.theme.font_size, options.theme.background_color, options.theme.text_color));
        output.push_str("        .message { margin-bottom: 20px; padding: 10px; border-radius: 8px; }\n");
        output.push_str(&format!("        .user {{ background-color: {}22; border-left: 3px solid {}; }}\n", 
            options.theme.user_color, options.theme.user_color));
        output.push_str(&format!("        .assistant {{ background-color: {}22; border-left: 3px solid {}; }}\n", 
            options.theme.assistant_color, options.theme.assistant_color));
        output.push_str(&format!("        .system {{ background-color: {}22; border-left: 3px solid {}; }}\n", 
            options.theme.system_color, options.theme.system_color));
        output.push_str("        .author { font-weight: bold; margin-bottom: 5px; }\n");
        output.push_str("        .timestamp { font-size: 0.9em; color: #666; }\n");
        output.push_str("        .content { white-space: pre-wrap; }\n");
        output.push_str("        .metadata { font-size: 0.85em; color: #888; margin-top: 5px; }\n");
        output.push_str("        .statistics { background-color: #f5f5f5; padding: 15px; border-radius: 8px; margin-top: 30px; }\n");
        output.push_str("    </style>\n");
        output.push_str("</head>\n<body>\n");
        
        // Header
        output.push_str(&format!("    <h1>{}</h1>\n", html_escape(&chat.metadata.chat_title)));
        
        // Metadata
        if options.include_metadata {
            output.push_str("    <div class=\"metadata\">\n");
            output.push_str(&format!("        <p>Exported: {}</p>\n", chat.metadata.exported_at.format(&options.date_format)));
            output.push_str(&format!("        <p>Total messages: {}</p>\n", chat.metadata.message_count));
            output.push_str("    </div>\n");
            output.push_str("    <hr>\n");
        }
        
        // Messages
        output.push_str("    <div class=\"conversation\">\n");
        for msg in &chat.messages {
            let class = match msg.author.as_str() {
                "user" | "User" => "user",
                "assistant" | "Assistant" => "assistant",
                _ => "system",
            };
            
            output.push_str(&format!("        <div class=\"message {}\">\n", class));
            output.push_str(&format!("            <div class=\"author\">{}</div>\n", html_escape(&msg.author)));
            
            if options.include_timestamps {
                output.push_str(&format!("            <div class=\"timestamp\">{}</div>\n", 
                    msg.timestamp.format(&options.date_format)));
            }
            
            output.push_str(&format!("            <div class=\"content\">{}</div>\n", html_escape(&msg.content)));
            
            if let Some(metadata) = &msg.metadata {
                output.push_str("            <div class=\"metadata\">\n");
                if let Some(model) = &metadata.model {
                    output.push_str(&format!("                Model: {} | ", html_escape(model)));
                }
                if let Some(tokens) = metadata.tokens {
                    output.push_str(&format!("                Tokens: {} | ", tokens));
                }
                if let Some(time) = metadata.processing_time_ms {
                    output.push_str(&format!("                Time: {}ms", time));
                }
                output.push_str("\n            </div>\n");
            }
            
            output.push_str("        </div>\n");
        }
        output.push_str("    </div>\n");
        
        // Statistics
        if options.include_statistics {
            output.push_str("    <div class=\"statistics\">\n");
            output.push_str("        <h2>Statistics</h2>\n");
            output.push_str(&format!("        <p>Total messages: {}</p>\n", chat.statistics.total_messages));
            output.push_str(&format!("        <p>Average message length: {} characters</p>\n", chat.statistics.avg_message_length));
            output.push_str(&format!("        <p>Estimated tokens: {}</p>\n", chat.statistics.total_tokens));
            
            if let Some(duration) = chat.statistics.duration_minutes {
                output.push_str(&format!("        <p>Duration: {:.1} minutes</p>\n", duration));
            }
            
            if !chat.statistics.topics.is_empty() {
                output.push_str(&format!("        <p>Topics: {}</p>\n", html_escape(&chat.statistics.topics.join(", "))));
            }
            
            output.push_str("    </div>\n");
        }
        
        // Footer
        output.push_str("</body>\n</html>");
        
        Ok(output)
    }
}

/// CSV exporter
pub struct CsvExporter;

impl Exporter for CsvExporter {
    fn export_to_string(&self, chat: &ExportedChat, options: &ExportOptions) -> Result<String> {
        let mut csv = String::new();
        
        // Header
        let mut headers = vec!["Author", "Content"];
        if options.include_timestamps {
            headers.insert(0, "Timestamp");
        }
        if options.include_ids {
            headers.insert(0, "ID");
        }
        csv.push_str(&headers.join(","));
        csv.push('\n');
        
        // Messages
        for msg in &chat.messages {
            let mut record = vec![];
            
            if options.include_ids {
                record.push(csv_escape(&msg.id));
            }
            if options.include_timestamps {
                record.push(csv_escape(&msg.timestamp.format(&options.date_format).to_string()));
            }
            record.push(csv_escape(&msg.author));
            record.push(csv_escape(&msg.content));
            
            csv.push_str(&record.join(","));
            csv.push('\n');
        }
        
        Ok(csv)
    }
}

/// CSV escape helper
fn csv_escape(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

/// HTML escape helper
fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}