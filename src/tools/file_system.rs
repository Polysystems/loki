use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::fs;
// File I/O imports removed as not currently used
use tracing::{debug, error, info, warn};

use crate::memory::{CognitiveMemory, MemoryMetadata};
use crate::tools::intelligent_manager::{ToolRequest, ToolResult, ToolStatus};

/// File system tool for comprehensive file and directory operations
#[derive(Debug, Clone)]
pub struct FileSystemTool {
    config: FileSystemConfig,
    memory: Option<Arc<CognitiveMemory>>,
    operation_history: Arc<tokio::sync::RwLock<Vec<FileOperation>>>,
    safety_checks: SafetyConfig,
}

/// Configuration for file system operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileSystemConfig {
    /// Base directory for operations (sandbox)
    pub base_directory: Option<PathBuf>,
    /// Maximum file size for operations (bytes)
    pub max_file_size: u64,
    /// Allowed file extensions for operations
    pub allowed_extensions: Vec<String>,
    /// Restricted directories (cannot be modified)
    pub restricted_directories: Vec<PathBuf>,
    /// Enable recursive operations
    pub enable_recursive_operations: bool,
    /// Maximum depth for recursive operations
    pub max_recursion_depth: u32,
}

/// Safety configuration for file operations
#[derive(Debug, Clone)]
pub struct SafetyConfig {
    /// Require confirmation for destructive operations
    pub require_confirmation: bool,
    /// Enable backup before destructive operations
    pub create_backups: bool,
    /// Maximum operations per minute
    pub rate_limit: u32,
    /// Dry run mode (preview operations without executing)
    pub dry_run: bool,
}

/// Represents a file system operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileOperation {
    pub operation_type: FileOperationType,
    pub source_path: Option<PathBuf>,
    pub target_path: Option<PathBuf>,
    pub timestamp: SystemTime,
    pub success: bool,
    pub error_message: Option<String>,
    pub size_affected: Option<u64>,
    pub items_affected: u32,
}

/// Types of file system operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FileOperationType {
    CreateDirectory,
    CreateFile,
    ReadFile,
    WriteFile,
    CopyFile,
    MoveFile,
    DeleteFile,
    DeleteDirectory,
    ListDirectory,
    GetFileInfo,
    SetPermissions,
    SearchFiles,
    WatchDirectory,
}

/// Result of a file system operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileSystemResult {
    pub success: bool,
    pub operation: FileOperationType,
    pub affected_paths: Vec<PathBuf>,
    pub content: Option<String>,
    pub metadata: HashMap<String, Value>,
    pub warnings: Vec<String>,
}

impl Default for FileSystemConfig {
    fn default() -> Self {
        Self {
            base_directory: None,
            max_file_size: 100 * 1024 * 1024, // 100MB
            allowed_extensions: vec![
                "txt".to_string(),
                "md".to_string(), 
                "json".to_string(),
                "yaml".to_string(),
                "yml".to_string(),
                "toml".to_string(),
                "py".to_string(),
                "rs".to_string(),
                "js".to_string(),
                "ts".to_string(),
                "html".to_string(),
                "css".to_string(),
                "log".to_string(),
            ],
            restricted_directories: vec![
                PathBuf::from("/etc"),
                PathBuf::from("/var"),
                PathBuf::from("/usr"),
                PathBuf::from("/bin"),
                PathBuf::from("/sbin"),
                PathBuf::from("/System"), // macOS
                PathBuf::from("/Library"), // macOS system
            ],
            enable_recursive_operations: true,
            max_recursion_depth: 10,
        }
    }
}

impl Default for SafetyConfig {
    fn default() -> Self {
        Self {
            require_confirmation: false, // For automation, set to false
            create_backups: true,
            rate_limit: 60, // 60 operations per minute
            dry_run: false,
        }
    }
}

impl FileSystemTool {
    pub fn new(config: FileSystemConfig) -> Self {
        Self {
            config,
            memory: None,
            operation_history: Arc::new(tokio::sync::RwLock::new(Vec::new())),
            safety_checks: SafetyConfig::default(),
        }
    }

    pub fn with_memory(mut self, memory: Arc<CognitiveMemory>) -> Self {
        self.memory = Some(memory);
        self
    }

    pub fn with_safety_config(mut self, safety: SafetyConfig) -> Self {
        self.safety_checks = safety;
        self
    }

    /// Execute a file system operation based on natural language request
    pub async fn execute_operation(&self, request: &ToolRequest) -> Result<ToolResult> {
        info!("🗂️ Executing file system operation: {}", request.tool_name);

        // Start timing the operation
        let start_time = std::time::Instant::now();

        // Parse the operation from parameters
        let operation = self.parse_operation(request)?;
        
        // Check safety constraints
        self.check_safety_constraints(&operation).await?;

        // Clone operation type for later use
        let operation_type_clone = operation.operation_type.clone();

        // Execute the operation
        let result = match &operation.operation_type {
            FileOperationType::CreateDirectory => {
                let path = operation.target_path.as_ref().ok_or_else(|| anyhow!("Target path required for CreateDirectory"))?;
                self.create_directory(path).await
            }
            FileOperationType::CreateFile => {
                let path = operation.target_path.as_ref().ok_or_else(|| anyhow!("Target path required for CreateFile"))?;
                let content = request.parameters.get("content").map(|v| v.as_str().unwrap_or("")).unwrap_or("");
                self.create_file(path, content).await
            }
            FileOperationType::ReadFile => {
                let path = operation.source_path.as_ref().ok_or_else(|| anyhow!("Source path required for ReadFile"))?;
                self.read_file(path).await
            }
            FileOperationType::WriteFile => {
                let path = operation.target_path.as_ref().ok_or_else(|| anyhow!("Target path required for WriteFile"))?;
                let content = request.parameters.get("content")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                self.write_file(path, content).await
            }
            FileOperationType::ListDirectory => {
                let path = operation.source_path.as_ref().ok_or_else(|| anyhow!("Source path required for ListDirectory"))?;
                self.list_directory(path).await
            }
            FileOperationType::DeleteFile => {
                let path = operation.source_path.as_ref().ok_or_else(|| anyhow!("Source path required for DeleteFile"))?;
                self.delete_file(path).await
            }
            FileOperationType::DeleteDirectory => {
                let path = operation.source_path.as_ref().ok_or_else(|| anyhow!("Source path required for DeleteDirectory"))?;
                self.delete_directory(path).await
            }
            FileOperationType::CopyFile => {
                let source = operation.source_path.as_ref().ok_or_else(|| anyhow!("Source path required for CopyFile"))?;
                let target = operation.target_path.as_ref().ok_or_else(|| anyhow!("Target path required for CopyFile"))?;
                self.copy_file(source, target).await
            }
            FileOperationType::MoveFile => {
                let source = operation.source_path.as_ref().ok_or_else(|| anyhow!("Source path required for MoveFile"))?;
                let target = operation.target_path.as_ref().ok_or_else(|| anyhow!("Target path required for MoveFile"))?;
                self.move_file(source, target).await
            }
            FileOperationType::GetFileInfo => {
                let path = operation.source_path.as_ref().ok_or_else(|| anyhow!("Source path required for GetFileInfo"))?;
                self.get_file_info(path).await
            }
            _ => {
                return Err(anyhow!("Operation type not implemented: {:?}", operation.operation_type));
            }
        };

        // Record operation in history
        self.record_operation(operation).await;

        // Store in memory if available
        if let Some(memory) = &self.memory {
            self.store_in_memory(memory, &result, request).await?;
        }

        // Calculate actual execution time
        let execution_time_ms = start_time.elapsed().as_millis() as u64;

        match result {
            Ok(fs_result) => Ok(ToolResult {
                status: if fs_result.success { ToolStatus::Success } else { ToolStatus::Failure("Operation failed".to_string()) },
                content: json!(fs_result),
                summary: format!("File system operation completed: {:?}", operation_type_clone),
                execution_time_ms,
                quality_score: if fs_result.success { 0.9 } else { 0.1 },
                memory_integrated: self.memory.is_some(),
                follow_up_suggestions: vec![],
            }),
            Err(e) => Ok(ToolResult {
                status: ToolStatus::Failure(e.to_string()),
                content: json!({
                    "error": e.to_string(),
                    "success": false
                }),
                summary: format!("File system operation failed: {}", e),
                execution_time_ms,
                quality_score: 0.0,
                memory_integrated: false,
                follow_up_suggestions: vec![],
            }),
        }
    }

    /// Parse operation from tool request
    fn parse_operation(&self, request: &ToolRequest) -> Result<FileOperation> {
        let operation_type = match request.tool_name.as_str() {
            "create_directory" | "mkdir" => FileOperationType::CreateDirectory,
            "create_file" | "touch" => FileOperationType::CreateFile,
            "read_file" | "cat" => FileOperationType::ReadFile,
            "write_file" => FileOperationType::WriteFile,
            "list_directory" | "ls" => FileOperationType::ListDirectory,
            "delete_file" | "rm" => FileOperationType::DeleteFile,
            "delete_directory" | "rmdir" => FileOperationType::DeleteDirectory,
            "copy_file" | "cp" => FileOperationType::CopyFile,
            "move_file" | "mv" => FileOperationType::MoveFile,
            "get_file_info" | "stat" => FileOperationType::GetFileInfo,
            _ => return Err(anyhow!("Unknown file system operation: {}", request.tool_name)),
        };

        let source_path = request.parameters.get("source")
            .or_else(|| request.parameters.get("path"))
            .and_then(|v| v.as_str())
            .map(PathBuf::from);

        let target_path = request.parameters.get("target")
            .or_else(|| request.parameters.get("destination"))
            .or_else(|| request.parameters.get("path"))
            .and_then(|v| v.as_str())
            .map(PathBuf::from);

        Ok(FileOperation {
            operation_type,
            source_path,
            target_path,
            timestamp: SystemTime::now(),
            success: false,
            error_message: None,
            size_affected: None,
            items_affected: 0,
        })
    }

    /// Check safety constraints before executing operation
    async fn check_safety_constraints(&self, operation: &FileOperation) -> Result<()> {
        // Check restricted directories
        if let Some(path) = operation.source_path.as_ref().or(operation.target_path.as_ref()) {
            for restricted in &self.config.restricted_directories {
                if path.starts_with(restricted) {
                    return Err(anyhow!("Access denied: Path {} is in restricted directory {:?}", 
                        path.display(), restricted));
                }
            }
        }

        // Check base directory constraint
        if let Some(base_dir) = &self.config.base_directory {
            if let Some(path) = operation.target_path.as_ref().or(operation.source_path.as_ref()) {
                if !path.starts_with(base_dir) {
                    return Err(anyhow!("Access denied: Path {} is outside base directory {:?}", 
                        path.display(), base_dir));
                }
            }
        }

        Ok(())
    }

    /// Create a directory
    async fn create_directory(&self, path: &Path) -> Result<FileSystemResult> {
        debug!("Creating directory: {}", path.display());

        if self.safety_checks.dry_run {
            return Ok(FileSystemResult {
                success: true,
                operation: FileOperationType::CreateDirectory,
                affected_paths: vec![path.to_path_buf()],
                content: None,
                metadata: [("dry_run".to_string(), json!(true))].into(),
                warnings: vec!["Dry run mode - directory not actually created".to_string()],
            });
        }

        match fs::create_dir_all(path).await {
            Ok(_) => {
                info!("✅ Created directory: {}", path.display());
                Ok(FileSystemResult {
                    success: true,
                    operation: FileOperationType::CreateDirectory,
                    affected_paths: vec![path.to_path_buf()],
                    content: None,
                    metadata: HashMap::new(),
                    warnings: Vec::new(),
                })
            }
            Err(e) => {
                error!("❌ Failed to create directory {}: {}", path.display(), e);
                Err(anyhow!("Failed to create directory: {}", e))
            }
        }
    }

    /// Create a file with optional content
    async fn create_file(&self, path: &Path, content: &str) -> Result<FileSystemResult> {
        debug!("Creating file: {} with {} bytes", path.display(), content.len());

        if self.safety_checks.dry_run {
            return Ok(FileSystemResult {
                success: true,
                operation: FileOperationType::CreateFile,
                affected_paths: vec![path.to_path_buf()],
                content: Some(format!("Would create file with {} bytes", content.len())),
                metadata: [("dry_run".to_string(), json!(true))].into(),
                warnings: vec!["Dry run mode - file not actually created".to_string()],
            });
        }

        // Create parent directory if it doesn't exist
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).await?;
        }

        match fs::write(path, content).await {
            Ok(_) => {
                info!("✅ Created file: {} ({} bytes)", path.display(), content.len());
                Ok(FileSystemResult {
                    success: true,
                    operation: FileOperationType::CreateFile,
                    affected_paths: vec![path.to_path_buf()],
                    content: Some(format!("Created file with {} bytes", content.len())),
                    metadata: [("size".to_string(), json!(content.len()))].into(),
                    warnings: Vec::new(),
                })
            }
            Err(e) => {
                error!("❌ Failed to create file {}: {}", path.display(), e);
                Err(anyhow!("Failed to create file: {}", e))
            }
        }
    }

    /// Read file content
    async fn read_file(&self, path: &Path) -> Result<FileSystemResult> {
        debug!("Reading file: {}", path.display());

        match fs::read_to_string(path).await {
            Ok(content) => {
                info!("✅ Read file: {} ({} bytes)", path.display(), content.len());
                Ok(FileSystemResult {
                    success: true,
                    operation: FileOperationType::ReadFile,
                    affected_paths: vec![path.to_path_buf()],
                    content: Some(content.clone()),
                    metadata: [
                        ("size".to_string(), json!(content.len())),
                        ("lines".to_string(), json!(content.lines().count())),
                    ].into(),
                    warnings: Vec::new(),
                })
            }
            Err(e) => {
                error!("❌ Failed to read file {}: {}", path.display(), e);
                Err(anyhow!("Failed to read file: {}", e))
            }
        }
    }

    /// Write content to file
    async fn write_file(&self, path: &Path, content: &str) -> Result<FileSystemResult> {
        debug!("Writing to file: {} ({} bytes)", path.display(), content.len());

        if self.safety_checks.dry_run {
            return Ok(FileSystemResult {
                success: true,
                operation: FileOperationType::WriteFile,
                affected_paths: vec![path.to_path_buf()],
                content: Some(format!("Would write {} bytes", content.len())),
                metadata: [("dry_run".to_string(), json!(true))].into(),
                warnings: vec!["Dry run mode - file not actually written".to_string()],
            });
        }

        // Create backup if enabled and file exists
        if self.safety_checks.create_backups && path.exists() {
            let backup_path = path.with_extension(format!("{}.backup", path.extension().and_then(|s| s.to_str()).unwrap_or("txt")));
            if let Err(e) = fs::copy(path, &backup_path).await {
                warn!("Failed to create backup: {}", e);
            } else {
                debug!("Created backup: {}", backup_path.display());
            }
        }

        // Create parent directory if needed
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).await?;
        }

        match fs::write(path, content).await {
            Ok(_) => {
                info!("✅ Wrote to file: {} ({} bytes)", path.display(), content.len());
                Ok(FileSystemResult {
                    success: true,
                    operation: FileOperationType::WriteFile,
                    affected_paths: vec![path.to_path_buf()],
                    content: Some(format!("Wrote {} bytes", content.len())),
                    metadata: [("size".to_string(), json!(content.len()))].into(),
                    warnings: Vec::new(),
                })
            }
            Err(e) => {
                error!("❌ Failed to write file {}: {}", path.display(), e);
                Err(anyhow!("Failed to write file: {}", e))
            }
        }
    }

    /// List directory contents
    async fn list_directory(&self, path: &Path) -> Result<FileSystemResult> {
        debug!("Listing directory: {}", path.display());

        match fs::read_dir(path).await {
            Ok(mut entries) => {
                let mut files = Vec::new();
                let mut total_size = 0u64;
                let mut item_count = 0u32;

                while let Some(entry) = entries.next_entry().await? {
                    let file_path = entry.path();
                    let metadata = entry.metadata().await?;
                    let is_dir = metadata.is_dir();
                    let size = if is_dir { 0 } else { metadata.len() };
                    
                    files.push(json!({
                        "name": file_path.file_name().and_then(|n| n.to_str()).unwrap_or("unknown"),
                        "path": file_path.to_string_lossy(),
                        "is_directory": is_dir,
                        "size": size,
                        "modified": metadata.modified().ok()
                            .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
                            .map(|d| d.as_secs())
                    }));
                    
                    total_size += size;
                    item_count += 1;
                }

                info!("✅ Listed directory: {} ({} items)", path.display(), item_count);
                Ok(FileSystemResult {
                    success: true,
                    operation: FileOperationType::ListDirectory,
                    affected_paths: vec![path.to_path_buf()],
                    content: Some(serde_json::to_string_pretty(&files)?),
                    metadata: [
                        ("item_count".to_string(), json!(item_count)),
                        ("total_size".to_string(), json!(total_size)),
                    ].into(),
                    warnings: Vec::new(),
                })
            }
            Err(e) => {
                error!("❌ Failed to list directory {}: {}", path.display(), e);
                Err(anyhow!("Failed to list directory: {}", e))
            }
        }
    }

    /// Delete a file
    async fn delete_file(&self, path: &Path) -> Result<FileSystemResult> {
        debug!("Deleting file: {}", path.display());

        if self.safety_checks.dry_run {
            return Ok(FileSystemResult {
                success: true,
                operation: FileOperationType::DeleteFile,
                affected_paths: vec![path.to_path_buf()],
                content: Some("Would delete file".to_string()),
                metadata: [("dry_run".to_string(), json!(true))].into(),
                warnings: vec!["Dry run mode - file not actually deleted".to_string()],
            });
        }

        // Create backup if enabled
        if self.safety_checks.create_backups && path.exists() {
            let backup_path = path.with_extension(format!("{}.deleted_backup", path.extension().and_then(|s| s.to_str()).unwrap_or("txt")));
            if let Err(e) = fs::copy(path, &backup_path).await {
                warn!("Failed to create backup before deletion: {}", e);
            }
        }

        match fs::remove_file(path).await {
            Ok(_) => {
                info!("✅ Deleted file: {}", path.display());
                Ok(FileSystemResult {
                    success: true,
                    operation: FileOperationType::DeleteFile,
                    affected_paths: vec![path.to_path_buf()],
                    content: Some("File deleted successfully".to_string()),
                    metadata: HashMap::new(),
                    warnings: Vec::new(),
                })
            }
            Err(e) => {
                error!("❌ Failed to delete file {}: {}", path.display(), e);
                Err(anyhow!("Failed to delete file: {}", e))
            }
        }
    }

    /// Delete a directory
    async fn delete_directory(&self, path: &Path) -> Result<FileSystemResult> {
        debug!("Deleting directory: {}", path.display());

        if self.safety_checks.dry_run {
            return Ok(FileSystemResult {
                success: true,
                operation: FileOperationType::DeleteDirectory,
                affected_paths: vec![path.to_path_buf()],
                content: Some("Would delete directory".to_string()),
                metadata: [("dry_run".to_string(), json!(true))].into(),
                warnings: vec!["Dry run mode - directory not actually deleted".to_string()],
            });
        }

        match fs::remove_dir_all(path).await {
            Ok(_) => {
                info!("✅ Deleted directory: {}", path.display());
                Ok(FileSystemResult {
                    success: true,
                    operation: FileOperationType::DeleteDirectory,
                    affected_paths: vec![path.to_path_buf()],
                    content: Some("Directory deleted successfully".to_string()),
                    metadata: HashMap::new(),
                    warnings: Vec::new(),
                })
            }
            Err(e) => {
                error!("❌ Failed to delete directory {}: {}", path.display(), e);
                Err(anyhow!("Failed to delete directory: {}", e))
            }
        }
    }

    /// Copy a file
    async fn copy_file(&self, source: &Path, target: &Path) -> Result<FileSystemResult> {
        debug!("Copying file: {} -> {}", source.display(), target.display());

        if self.safety_checks.dry_run {
            return Ok(FileSystemResult {
                success: true,
                operation: FileOperationType::CopyFile,
                affected_paths: vec![source.to_path_buf(), target.to_path_buf()],
                content: Some("Would copy file".to_string()),
                metadata: [("dry_run".to_string(), json!(true))].into(),
                warnings: vec!["Dry run mode - file not actually copied".to_string()],
            });
        }

        // Create target directory if needed
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent).await?;
        }

        match fs::copy(source, target).await {
            Ok(bytes_copied) => {
                info!("✅ Copied file: {} -> {} ({} bytes)", source.display(), target.display(), bytes_copied);
                Ok(FileSystemResult {
                    success: true,
                    operation: FileOperationType::CopyFile,
                    affected_paths: vec![source.to_path_buf(), target.to_path_buf()],
                    content: Some(format!("Copied {} bytes", bytes_copied)),
                    metadata: [("bytes_copied".to_string(), json!(bytes_copied))].into(),
                    warnings: Vec::new(),
                })
            }
            Err(e) => {
                error!("❌ Failed to copy file {} -> {}: {}", source.display(), target.display(), e);
                Err(anyhow!("Failed to copy file: {}", e))
            }
        }
    }

    /// Move a file
    async fn move_file(&self, source: &Path, target: &Path) -> Result<FileSystemResult> {
        debug!("Moving file: {} -> {}", source.display(), target.display());

        if self.safety_checks.dry_run {
            return Ok(FileSystemResult {
                success: true,
                operation: FileOperationType::MoveFile,
                affected_paths: vec![source.to_path_buf(), target.to_path_buf()],
                content: Some("Would move file".to_string()),
                metadata: [("dry_run".to_string(), json!(true))].into(),
                warnings: vec!["Dry run mode - file not actually moved".to_string()],
            });
        }

        // Create target directory if needed
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent).await?;
        }

        match fs::rename(source, target).await {
            Ok(_) => {
                info!("✅ Moved file: {} -> {}", source.display(), target.display());
                Ok(FileSystemResult {
                    success: true,
                    operation: FileOperationType::MoveFile,
                    affected_paths: vec![source.to_path_buf(), target.to_path_buf()],
                    content: Some("File moved successfully".to_string()),
                    metadata: HashMap::new(),
                    warnings: Vec::new(),
                })
            }
            Err(e) => {
                error!("❌ Failed to move file {} -> {}: {}", source.display(), target.display(), e);
                Err(anyhow!("Failed to move file: {}", e))
            }
        }
    }

    /// Get file information
    async fn get_file_info(&self, path: &Path) -> Result<FileSystemResult> {
        debug!("Getting file info: {}", path.display());

        match fs::metadata(path).await {
            Ok(metadata) => {
                let file_type = if metadata.is_dir() {
                    "directory"
                } else if metadata.is_file() {
                    "file"
                } else {
                    "other"
                };

                let info = json!({
                    "path": path.to_string_lossy(),
                    "type": file_type,
                    "size": metadata.len(),
                    "is_readonly": metadata.permissions().readonly(),
                    "modified": metadata.modified().ok()
                        .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
                        .map(|d| d.as_secs()),
                    "created": metadata.created().ok()
                        .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
                        .map(|d| d.as_secs()),
                });

                info!("✅ Got file info: {}", path.display());
                Ok(FileSystemResult {
                    success: true,
                    operation: FileOperationType::GetFileInfo,
                    affected_paths: vec![path.to_path_buf()],
                    content: Some(serde_json::to_string_pretty(&info)?),
                    metadata: [
                        ("size".to_string(), json!(metadata.len())),
                        ("type".to_string(), json!(file_type)),
                    ].into(),
                    warnings: Vec::new(),
                })
            }
            Err(e) => {
                error!("❌ Failed to get file info {}: {}", path.display(), e);
                Err(anyhow!("Failed to get file info: {}", e))
            }
        }
    }

    /// Record operation in history
    async fn record_operation(&self, mut operation: FileOperation) -> Result<()> {
        operation.timestamp = SystemTime::now();
        let mut history = self.operation_history.write().await;
        history.push(operation);
        
        // Keep only last 1000 operations
        if history.len() > 1000 {
            let len = history.len();
            history.drain(0..len - 1000);
        }
        
        Ok(())
    }

    /// Store operation result in cognitive memory
    async fn store_in_memory(&self, memory: &CognitiveMemory, result: &Result<FileSystemResult>, request: &ToolRequest) -> Result<()> {
        let memory_content = match result {
            Ok(fs_result) => json!({
                "operation": fs_result.operation,
                "success": fs_result.success,
                "affected_paths": fs_result.affected_paths,
                "request": request.parameters
            }),
            Err(e) => json!({
                "operation": "file_system_error",
                "success": false,
                "error": e.to_string(),
                "request": request.parameters
            })
        };

        let metadata = MemoryMetadata {
            source: "file_system_tool".to_string(),
            tags: vec!["file_system".to_string(), "tool_execution".to_string()],
            importance: 0.6,
            associations: Vec::new(),
            context: Some("file_system_operation".to_string()),
            created_at: chrono::Utc::now(),
            accessed_count: 0,
            last_accessed: None,
            version: 1,
                    category: "tool_usage".to_string(),
            timestamp: chrono::Utc::now(),
            expiration: None,
        };

        memory.store(memory_content.to_string(), vec!["file_operation".to_string()], metadata).await?;
        Ok(())
    }

    /// Get operation history
    pub async fn get_operation_history(&self, limit: Option<usize>) -> Vec<FileOperation> {
        let history = self.operation_history.read().await;
        let limit = limit.unwrap_or(100);
        history.iter().rev().take(limit).cloned().collect()
    }
}