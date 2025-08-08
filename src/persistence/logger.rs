use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::fs::{File, OpenOptions};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::RwLock;
use tracing::{debug, error, info};

use super::PersistenceConfig;

/// Log levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LogLevel {
    Debug,
    Info,
    Warn,
    Error,
    Critical,
}

/// Log categories for different aspects of Loki
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LogCategory {
    Consciousness, // Thought stream
    Learning,      // Learning events
    SelfModify,    // Code changes
    GitHub,        // GitHub interactions
    Social,        // Twitter/social media
    Memory,        // Memory operations
    System,        // System health
    Error,         // Errors
    Debug,         // Debug info
}

/// A single log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    pub timestamp: DateTime<Utc>,
    pub level: LogLevel,
    pub category: LogCategory,
    pub message: String,
    pub metadata: serde_json::Value,
    pub context: HashMap<String, String>,
}

/// Logger implementation with rotation
pub struct LokiLogger {
    config: PersistenceConfig,
    current_file: Arc<RwLock<Option<File>>>,
    current_path: Arc<RwLock<PathBuf>>,
    current_size: Arc<RwLock<u64>>,
}

impl LokiLogger {
    /// Create new logger
    pub async fn new(config: PersistenceConfig) -> Result<Self> {
        let log_path =
            config.log_dir.join(format!("loki_{}.jsonl", Utc::now().format("%Y%m%d_%H%M%S")));

        let file = OpenOptions::new().create(true).append(true).open(&log_path).await?;

        Ok(Self {
            config,
            current_file: Arc::new(RwLock::new(Some(file))),
            current_path: Arc::new(RwLock::new(log_path)),
            current_size: Arc::new(RwLock::new(0)),
        })
    }

    /// Write a log entry
    pub async fn write(&self, entry: LogEntry) -> Result<()> {
        let json = serde_json::to_string(&entry)?;
        let bytes = format!("{}\n", json).into_bytes();

        // Check if rotation needed
        {
            let size = self.current_size.read().await;
            if *size + bytes.len() as u64 > (self.config.max_log_size_mb as u64 * 1024 * 1024) {
                self.rotate().await?;
            }
        }

        // Write to file
        {
            let mut file_opt = self.current_file.write().await;
            if let Some(file) = file_opt.as_mut() {
                file.write_all(&bytes).await?;
                file.flush().await?;
            }
        }

        // Update size
        {
            let mut size = self.current_size.write().await;
            *size += bytes.len() as u64;
        }

        // Also log to tracing based on level
        match entry.level {
            LogLevel::Debug => debug!("[{}] {}", entry.category_str(), entry.message),
            LogLevel::Info => info!("[{}] {}", entry.category_str(), entry.message),
            LogLevel::Warn => tracing::warn!("[{}] {}", entry.category_str(), entry.message),
            LogLevel::Error => error!("[{}] {}", entry.category_str(), entry.message),
            LogLevel::Critical => error!("[CRITICAL][{}] {}", entry.category_str(), entry.message),
        }

        Ok(())
    }

    /// Rotate log file
    pub async fn rotate(&self) -> Result<()> {
        info!("Rotating log file");

        // Close current file
        {
            let mut file_opt = self.current_file.write().await;
            *file_opt = None;
        }

        // Create new file
        let new_path =
            self.config.log_dir.join(format!("loki_{}.jsonl", Utc::now().format("%Y%m%d_%H%M%S")));

        let new_file = OpenOptions::new().create(true).append(true).open(&new_path).await?;

        // Update references
        {
            let mut file_opt = self.current_file.write().await;
            *file_opt = Some(new_file);
        }
        {
            let mut path = self.current_path.write().await;
            *path = new_path;
        }
        {
            let mut size = self.current_size.write().await;
            *size = 0;
        }

        Ok(())
    }

    /// Rotate if needed based on size
    pub async fn rotate_if_needed(&self) -> Result<()> {
        let size = *self.current_size.read().await;
        if size > (self.config.max_log_size_mb as u64 * 1024 * 1024) {
            self.rotate().await?;
        }
        Ok(())
    }

    /// Read logs since a timestamp
    pub async fn read_since(
        &self,
        since: DateTime<Utc>,
        category: Option<LogCategory>,
    ) -> Result<Vec<LogEntry>> {
        let mut entries = Vec::new();

        // Read all log files in directory
        let mut dir = tokio::fs::read_dir(&self.config.log_dir).await?;
        let mut log_files = Vec::new();

        while let Some(entry) = dir.next_entry().await? {
            let path = entry.path();
            if path.extension() == Some(std::ffi::OsStr::new("jsonl")) {
                log_files.push(path);
            }
        }

        // Sort by name (which includes timestamp)
        log_files.sort();

        // Read each file
        for path in log_files {
            let file = File::open(&path).await?;
            let reader = BufReader::new(file);
            let mut lines = reader.lines();

            while let Some(line) = lines.next_line().await? {
                if let Ok(entry) = serde_json::from_str::<LogEntry>(&line) {
                    if entry.timestamp >= since {
                        if category.is_none() || category.as_ref() == Some(&entry.category) {
                            entries.push(entry);
                        }
                    }
                }
            }
        }

        Ok(entries)
    }

    /// Get all log files
    pub async fn get_log_files(&self) -> Result<Vec<LogFileInfo>> {
        let mut files = Vec::new();
        let mut dir = tokio::fs::read_dir(&self.config.log_dir).await?;

        while let Some(entry) = dir.next_entry().await? {
            let path = entry.path();
            if path.extension() == Some(std::ffi::OsStr::new("jsonl")) {
                let metadata = entry.metadata().await?;
                files.push(LogFileInfo {
                    path,
                    size: metadata.len(),
                    modified: metadata.modified()?.into(),
                });
            }
        }

        files.sort_by_key(|f| f.modified);
        Ok(files)
    }
}

impl LogEntry {
    /// Get category as string
    pub fn category_str(&self) -> &str {
        match self.category {
            LogCategory::Consciousness => "CONSCIOUS",
            LogCategory::Learning => "LEARNING",
            LogCategory::SelfModify => "SELFMOD",
            LogCategory::GitHub => "GITHUB",
            LogCategory::Social => "SOCIAL",
            LogCategory::Memory => "MEMORY",
            LogCategory::System => "SYSTEM",
            LogCategory::Error => "ERROR",
            LogCategory::Debug => "DEBUG",
        }
    }

    /// Check if entry is important for archival
    pub fn is_important(&self) -> bool {
        match self.level {
            LogLevel::Critical | LogLevel::Error => true,
            LogLevel::Warn => {
                self.category == LogCategory::Learning || self.category == LogCategory::SelfModify
            }
            LogLevel::Info => {
                matches!(
                    self.category,
                    LogCategory::Consciousness
                        | LogCategory::Learning
                        | LogCategory::SelfModify
                        | LogCategory::GitHub
                )
            }
            LogLevel::Debug => false,
        }
    }
}

/// Information about a log file
#[derive(Debug, Clone)]
pub struct LogFileInfo {
    pub path: PathBuf,
    pub size: u64,
    pub modified: std::time::SystemTime,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_logger() {
        let config = PersistenceConfig {
            log_dir: PathBuf::from("/tmp/loki_logger_test"),
            max_log_size_mb: 1,
            ..Default::default()
        };

        tokio::fs::create_dir_all(&config.log_dir).await.unwrap();
        let logger = LokiLogger::new(config).await.unwrap();

        // Write some entries
        for i in 0..10 {
            let entry = LogEntry {
                timestamp: Utc::now(),
                level: LogLevel::Info,
                category: LogCategory::Consciousness,
                message: format!("Test thought #{}", i),
                metadata: serde_json::json!({"index": i}),
                context: HashMap::new(),
            };

            logger.write(entry).await.unwrap();
        }

        // Read back
        let entries = logger
            .read_since(Utc::now() - chrono::Duration::hours(1), Some(LogCategory::Consciousness))
            .await
            .unwrap();

        assert_eq!(entries.len(), 10);
    }
}
