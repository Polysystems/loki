//! File attachment and handling for the chat interface
//! 
//! Provides support for attaching files, displaying file metadata,
//! and integrating file content into chat messages.

use std::path::{Path, PathBuf};
use std::fs;
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};

/// File attachment information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileAttachment {
    /// Unique ID for the attachment
    pub id: String,
    
    /// File path
    pub path: PathBuf,
    
    /// File name
    pub name: String,
    
    /// File size in bytes
    pub size: u64,
    
    /// MIME type
    pub mime_type: String,
    
    /// File preview (first few lines for text files)
    pub preview: Option<String>,
    
    /// Attachment state
    pub state: AttachmentState,
}

/// Attachment state
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AttachmentState {
    Pending,
    Loaded,
    Failed(String),
}

/// Supported file types
#[derive(Debug, Clone, PartialEq)]
pub enum FileType {
    Text,
    Code(String), // language
    Image,
    Binary,
    Archive,
    Document,
}

/// File handler for chat attachments
#[derive(Clone)]
pub struct FileHandler {
    /// Maximum file size in bytes (default: 10MB)
    max_file_size: u64,
    
    /// Supported text extensions
    text_extensions: Vec<String>,
    
    /// Supported code extensions
    code_extensions: Vec<(String, String)>, // (extension, language)
}

impl FileHandler {
    pub fn new() -> Self {
        Self {
            max_file_size: 10 * 1024 * 1024, // 10MB
            text_extensions: vec![
                "txt".to_string(),
                "md".to_string(),
                "log".to_string(),
                "csv".to_string(),
                "json".to_string(),
                "yaml".to_string(),
                "yml".to_string(),
                "toml".to_string(),
                "xml".to_string(),
            ],
            code_extensions: vec![
                ("rs".to_string(), "rust".to_string()),
                ("py".to_string(), "python".to_string()),
                ("js".to_string(), "javascript".to_string()),
                ("ts".to_string(), "typescript".to_string()),
                ("go".to_string(), "go".to_string()),
                ("java".to_string(), "java".to_string()),
                ("cpp".to_string(), "cpp".to_string()),
                ("c".to_string(), "c".to_string()),
                ("h".to_string(), "c".to_string()),
                ("hpp".to_string(), "cpp".to_string()),
                ("sh".to_string(), "bash".to_string()),
                ("bash".to_string(), "bash".to_string()),
                ("sql".to_string(), "sql".to_string()),
                ("html".to_string(), "html".to_string()),
                ("css".to_string(), "css".to_string()),
            ],
        }
    }
    
    /// Attach a file
    pub async fn attach_file(&self, path: &Path) -> Result<FileAttachment> {
        // Check if file exists
        if !path.exists() {
            return Err(anyhow!("File does not exist: {:?}", path));
        }
        
        // Get file metadata
        let metadata = fs::metadata(path)?;
        
        // Check file size
        if metadata.len() > self.max_file_size {
            return Err(anyhow!(
                "File too large: {} bytes (max: {} bytes)",
                metadata.len(),
                self.max_file_size
            ));
        }
        
        // Determine file type
        let file_type = self.detect_file_type(path);
        let mime_type = self.get_mime_type(path, &file_type);
        
        // Generate preview for text/code files
        let preview = match &file_type {
            FileType::Text | FileType::Code(_) => {
                self.generate_preview(path).ok()
            }
            _ => None,
        };
        
        // Create attachment
        let attachment = FileAttachment {
            id: uuid::Uuid::new_v4().to_string(),
            path: path.to_path_buf(),
            name: path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown")
                .to_string(),
            size: metadata.len(),
            mime_type,
            preview,
            state: AttachmentState::Loaded,
        };
        
        Ok(attachment)
    }
    
    /// Detect file type from extension
    fn detect_file_type(&self, path: &Path) -> FileType {
        let extension = path.extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();
        
        // Check code extensions
        for (ext, lang) in &self.code_extensions {
            if extension == *ext {
                return FileType::Code(lang.clone());
            }
        }
        
        // Check text extensions
        if self.text_extensions.contains(&extension) {
            return FileType::Text;
        }
        
        // Check image extensions
        match extension.as_str() {
            "png" | "jpg" | "jpeg" | "gif" | "bmp" | "svg" | "webp" => FileType::Image,
            "zip" | "tar" | "gz" | "rar" | "7z" => FileType::Archive,
            "pdf" | "doc" | "docx" | "odt" => FileType::Document,
            _ => FileType::Binary,
        }
    }
    
    /// Get MIME type for file
    fn get_mime_type(&self, path: &Path, file_type: &FileType) -> String {
        match file_type {
            FileType::Text => "text/plain".to_string(),
            FileType::Code(_) => "text/plain".to_string(),
            FileType::Image => {
                let ext = path.extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("");
                match ext {
                    "png" => "image/png",
                    "jpg" | "jpeg" => "image/jpeg",
                    "gif" => "image/gif",
                    "svg" => "image/svg+xml",
                    _ => "image/unknown",
                }.to_string()
            }
            FileType::Binary => "application/octet-stream".to_string(),
            FileType::Archive => "application/zip".to_string(),
            FileType::Document => "application/pdf".to_string(),
        }
    }
    
    /// Generate preview for text files
    fn generate_preview(&self, path: &Path) -> Result<String> {
        let content = fs::read_to_string(path)?;
        let lines: Vec<&str> = content.lines().take(5).collect();
        
        if lines.len() < 5 && content.len() < 500 {
            // Small file, show all content
            Ok(content)
        } else {
            // Show first 5 lines with ellipsis
            let preview = lines.join("\n");
            if content.lines().count() > 5 {
                Ok(format!("{}\n...", preview))
            } else {
                Ok(preview)
            }
        }
    }
    
    /// Read file content
    pub async fn read_file_content(&self, attachment: &FileAttachment) -> Result<String> {
        let file_type = self.detect_file_type(&attachment.path);
        
        match file_type {
            FileType::Text | FileType::Code(_) => {
                Ok(fs::read_to_string(&attachment.path)?)
            }
            FileType::Image => {
                Ok(format!("[Image: {}]", attachment.name))
            }
            _ => {
                Ok(format!("[Binary file: {} ({} bytes)]", attachment.name, attachment.size))
            }
        }
    }
    
    /// Format attachment for display
    pub fn format_attachment(&self, attachment: &FileAttachment) -> String {
        let size_str = self.format_file_size(attachment.size);
        let icon = self.get_file_icon(&attachment.mime_type);
        
        let mut result = format!("{} {} ({})", icon, attachment.name, size_str);
        
        if let Some(preview) = &attachment.preview {
            result.push_str(&format!("\n{}", preview));
        }
        
        result
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
    
    /// Get file icon based on type
    pub fn get_file_icon(&self, mime_type: &str) -> &'static str {
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
    
    /// Get file icon based on path
    fn get_file_icon_from_path(&self, path: &Path) -> &'static str {
        let file_type = self.detect_file_type(path);
        
        match file_type {
            FileType::Text => "📄",
            FileType::Code(_) => "📝",
            FileType::Image => "🖼️",
            FileType::Archive => "📦",
            FileType::Document => "📑",
            FileType::Binary => "📎",
        }
    }
}

impl Default for FileHandler {
    fn default() -> Self {
        Self::new()
    }
}

/// Attachment manager for tracking attachments in a chat
#[derive(Debug, Clone, Default)]
pub struct AttachmentManager {
    /// Active attachments by message ID
    attachments: std::collections::HashMap<String, Vec<FileAttachment>>,
}

impl AttachmentManager {
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Add attachment to a message
    pub fn add_attachment(&mut self, message_id: String, attachment: FileAttachment) {
        self.attachments
            .entry(message_id)
            .or_insert_with(Vec::new)
            .push(attachment);
    }
    
    /// Get attachments for a message
    pub fn get_attachments(&self, message_id: &str) -> Option<&Vec<FileAttachment>> {
        self.attachments.get(message_id)
    }
    
    /// Remove attachment
    pub fn remove_attachment(&mut self, message_id: &str, attachment_id: &str) {
        if let Some(attachments) = self.attachments.get_mut(message_id) {
            attachments.retain(|a| a.id != attachment_id);
        }
    }
    
    /// Clear all attachments for a message
    pub fn clear_attachments(&mut self, message_id: &str) {
        self.attachments.remove(message_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;
    
    #[tokio::test]
    async fn test_file_attachment() {
        let handler = FileHandler::new();
        
        // Create a temporary file
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "Hello, world!").unwrap();
        writeln!(temp_file, "This is a test file.").unwrap();
        
        // Attach the file
        let attachment = handler.attach_file(temp_file.path()).await.unwrap();
        
        assert_eq!(attachment.state, AttachmentState::Loaded);
        assert!(attachment.preview.is_some());
        assert_eq!(attachment.mime_type, "text/plain");
    }
    
    #[test]
    fn test_file_type_detection() {
        let handler = FileHandler::new();
        
        assert!(matches!(
            handler.detect_file_type(Path::new("test.rs")),
            FileType::Code(ref lang) if lang == "rust"
        ));
        
        assert!(matches!(
            handler.detect_file_type(Path::new("image.png")),
            FileType::Image
        ));
        
        assert!(matches!(
            handler.detect_file_type(Path::new("document.txt")),
            FileType::Text
        ));
    }
    
    #[test]
    fn test_file_size_formatting() {
        let handler = FileHandler::new();
        
        assert_eq!(handler.format_file_size(512), "512 B");
        assert_eq!(handler.format_file_size(1024), "1.0 KB");
        assert_eq!(handler.format_file_size(1536), "1.5 KB");
        assert_eq!(handler.format_file_size(1048576), "1.0 MB");
    }
}