use std::path::PathBuf;

use anyhow::Result;
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use tokio::fs;
use tracing::info;

use super::PersistenceConfig;
use super::archive::ArchiveManager;
use super::logger::{LogCategory, LogEntry, LogLevel};

/// Retention policy for logs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionPolicy {
    /// Hot logs (immediate access)
    pub hot_hours: u32,

    /// Warm logs (quick access)
    pub warm_days: u32,

    /// Cold logs (archived)
    pub cold_days: u32,

    /// Categories to always keep
    pub preserve_categories: Vec<LogCategory>,

    /// Minimum importance to preserve
    pub preserve_importance: f32,
}

impl Default for RetentionPolicy {
    fn default() -> Self {
        Self {
            hot_hours: 24,
            warm_days: 7,
            cold_days: 30,
            preserve_categories: vec![
                LogCategory::SelfModify,
                LogCategory::Learning,
                LogCategory::GitHub,
            ],
            preserve_importance: 0.8,
        }
    }
}

/// Garbage collector for log management
pub struct GarbageCollector {
    config: PersistenceConfig,
    policy: RetentionPolicy,
    stats: GcStats,
}

impl GarbageCollector {
    /// Create new garbage collector
    pub async fn new(config: PersistenceConfig) -> Result<Self> {
        Ok(Self { config, policy: RetentionPolicy::default(), stats: GcStats::default() })
    }

    /// Run garbage collection
    pub async fn run(&self, archive: &ArchiveManager) -> Result<GcStats> {
        info!("Starting garbage collection");
        let start_time = Utc::now();

        let mut stats = GcStats::default();
        stats.start_time = start_time;

        // Get all log files
        let log_files = self.get_log_files().await?;

        for file_info in log_files {
            let age = start_time - DateTime::<Utc>::from(file_info.modified);

            // Determine file category
            let category = if age < Duration::hours(self.policy.hot_hours as i64) {
                FileCategory::Hot
            } else if age < Duration::days(self.policy.warm_days as i64) {
                FileCategory::Warm
            } else if age < Duration::days(self.policy.cold_days as i64) {
                FileCategory::Cold
            } else {
                FileCategory::Expired
            };

            match category {
                FileCategory::Hot => {
                    // Keep as is
                    stats.hot_files += 1;
                    stats.hot_size += file_info.size;
                }
                FileCategory::Warm => {
                    // Move to warm storage if needed
                    stats.warm_files += 1;
                    stats.warm_size += file_info.size;
                }
                FileCategory::Cold => {
                    // Archive important logs
                    let archived = self.process_cold_file(&file_info, archive).await?;
                    if archived {
                        stats.archived_files += 1;
                        stats.archived_size += file_info.size;
                    } else {
                        stats.deleted_files += 1;
                        stats.deleted_size += file_info.size;
                    }
                }
                FileCategory::Expired => {
                    // Delete unless it contains important logs
                    let should_preserve = self.check_preserve_file(&file_info).await?;
                    if should_preserve {
                        // Archive before deletion
                        archive.archive_file(&file_info.path).await?;
                        stats.archived_files += 1;
                        stats.archived_size += file_info.size;
                    }

                    // Delete the file
                    fs::remove_file(&file_info.path).await?;
                    stats.deleted_files += 1;
                    stats.deleted_size += file_info.size;
                }
            }
        }

        stats.end_time = Utc::now();
        stats.duration = (stats.end_time - stats.start_time).num_seconds() as u64;

        info!("Garbage collection completed: {:?}", stats);
        Ok(stats)
    }

    /// Process a cold file for archival
    async fn process_cold_file(
        &self,
        file_info: &LogFileInfo,
        archive: &ArchiveManager,
    ) -> Result<bool> {
        // Read file and check for important logs
        let important_count = self.count_important_logs(&file_info.path).await?;

        if important_count > 0 {
            // Archive the file
            archive.archive_file(&file_info.path).await?;

            // Delete original after successful archive
            fs::remove_file(&file_info.path).await?;

            Ok(true)
        } else {
            // Delete without archiving
            fs::remove_file(&file_info.path).await?;
            Ok(false)
        }
    }

    /// Check if file should be preserved
    async fn check_preserve_file(&self, file_info: &LogFileInfo) -> Result<bool> {
        let important_count = self.count_important_logs(&file_info.path).await?;
        Ok(important_count > 10) // Preserve if more than 10 important logs
    }

    /// Count important logs in a file
    async fn count_important_logs(&self, path: &PathBuf) -> Result<usize> {
        use tokio::io::{AsyncBufReadExt, BufReader};

        let file = fs::File::open(path).await?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();
        let mut count = 0;

        while let Some(line) = lines.next_line().await? {
            if let Ok(entry) = serde_json::from_str::<LogEntry>(&line) {
                if self.is_important_log(&entry) {
                    count += 1;
                }
            }
        }

        Ok(count)
    }

    /// Check if a log entry is important
    fn is_important_log(&self, entry: &LogEntry) -> bool {
        // Check category
        if self.policy.preserve_categories.contains(&entry.category) {
            return true;
        }

        // Check level
        if matches!(entry.level, LogLevel::Critical | LogLevel::Error) {
            return true;
        }

        // Check metadata importance
        if let Some(importance) = entry.metadata.get("importance").and_then(|v| v.as_f64()) {
            if importance >= self.policy.preserve_importance as f64 {
                return true;
            }
        }

        false
    }

    /// Get all log files
    async fn get_log_files(&self) -> Result<Vec<LogFileInfo>> {
        let mut files = Vec::new();
        let mut dir = fs::read_dir(&self.config.log_dir).await?;

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

    /// Update retention policy
    pub fn set_policy(&mut self, policy: RetentionPolicy) {
        self.policy = policy;
    }

    /// Get current stats
    pub fn stats(&self) -> &GcStats {
        &self.stats
    }
}

/// File categories based on age
#[derive(Debug, Clone, Copy)]
enum FileCategory {
    Hot,     // Recent, keep in place
    Warm,    // Medium age, may compress
    Cold,    // Old, archive important
    Expired, // Very old, delete
}

/// Log file information
#[derive(Debug, Clone)]
struct LogFileInfo {
    path: PathBuf,
    size: u64,
    modified: std::time::SystemTime,
}

/// Garbage collection statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GcStats {
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub duration: u64,
    pub hot_files: usize,
    pub hot_size: u64,
    pub warm_files: usize,
    pub warm_size: u64,
    pub archived_files: usize,
    pub archived_size: u64,
    pub deleted_files: usize,
    pub deleted_size: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_retention_policy() {
        let policy = RetentionPolicy::default();
        assert_eq!(policy.hot_hours, 24);
        assert_eq!(policy.warm_days, 7);
        assert!(policy.preserve_categories.contains(&LogCategory::SelfModify));
    }
}
