use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{error, info};

pub mod archive;
pub mod garbage_collector;
pub mod index;
pub mod logger;

pub use archive::{ArchiveManager, ArchivePolicy};
pub use garbage_collector::{GarbageCollector, RetentionPolicy};
pub use index::{LogIndex, SearchQuery, SearchResult};
pub use logger::{LogCategory, LogEntry, LogLevel, LokiLogger};

/// Persistence configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistenceConfig {
    /// Base directory for logs
    pub log_dir: PathBuf,

    /// Archive storage path (S3, local, etc)
    pub archive_path: String,

    /// Hot log retention (hours)
    pub hot_retention_hours: u32,

    /// Warm log retention (days)
    pub warm_retention_days: u32,

    /// Cold log compression
    pub compress_cold_logs: bool,

    /// Max log size before rotation (MB)
    pub max_log_size_mb: u32,

    /// Index update interval
    pub index_interval_secs: u64,

    /// GC run interval
    pub gc_interval_hours: u32,
}

impl Default for PersistenceConfig {
    fn default() -> Self {
        Self {
            log_dir: PathBuf::from("./logs"),
            archive_path: "./archive".to_string(),
            hot_retention_hours: 24,
            warm_retention_days: 7,
            compress_cold_logs: true,
            max_log_size_mb: 100,
            index_interval_secs: 300, // 5 minutes
            gc_interval_hours: 6,
        }
    }
}

/// Main persistence manager
pub struct PersistenceManager {
    /// Configuration
    config: PersistenceConfig,

    /// Logger instance
    logger: Arc<LokiLogger>,

    /// Archive manager
    archive: Arc<ArchiveManager>,

    /// Garbage collector
    gc: Arc<GarbageCollector>,

    /// Search index
    index: Arc<LogIndex>,

    /// Stats
    stats: Arc<RwLock<PersistenceStats>>,
}

impl PersistenceManager {
    /// Create new persistence manager
    pub async fn new(config: PersistenceConfig) -> Result<Self> {
        info!("Initializing persistence manager");

        // Create directories
        tokio::fs::create_dir_all(&config.log_dir).await?;
        tokio::fs::create_dir_all(&config.archive_path).await?;

        // Initialize components
        let logger = Arc::new(LokiLogger::new(config.clone()).await?);
        let archive = Arc::new(ArchiveManager::new(config.clone()).await?);
        let gc = Arc::new(GarbageCollector::new(config.clone()).await?);
        let index = Arc::new(LogIndex::new(config.clone()).await?);

        let manager = Self {
            config,
            logger,
            archive,
            gc,
            index,
            stats: Arc::new(RwLock::new(PersistenceStats::default())),
        };

        // Start background tasks
        manager.start_background_tasks().await;

        Ok(manager)
    }

    /// Log an entry
    pub async fn log(&self, entry: LogEntry) -> Result<()> {
        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.total_logs += 1;
            stats
                .logs_by_category
                .entry(entry.category.clone())
                .and_modify(|e| *e += 1)
                .or_insert(1);
        }

        // Write log
        self.logger.write(entry).await?;

        Ok(())
    }

    /// Log consciousness stream
    pub async fn log_thought(&self, thought: String, metadata: ThoughtMetadata) -> Result<()> {
        let entry = LogEntry {
            timestamp: Utc::now(),
            level: LogLevel::Info,
            category: LogCategory::Consciousness,
            message: thought,
            metadata: serde_json::to_value(metadata)?,
            context: Default::default(),
        };

        self.log(entry).await
    }

    /// Log learning event
    pub async fn log_learning(&self, event: String, outcome: LearningOutcome) -> Result<()> {
        let entry = LogEntry {
            timestamp: Utc::now(),
            level: match outcome {
                LearningOutcome::Success => LogLevel::Info,
                LearningOutcome::Failure => LogLevel::Warn,
                LearningOutcome::Insight => LogLevel::Critical,
            },
            category: LogCategory::Learning,
            message: event,
            metadata: serde_json::to_value(outcome)?,
            context: Default::default(),
        };

        self.log(entry).await
    }

    /// Log GitHub action
    pub async fn log_github(&self, action: String, details: serde_json::Value) -> Result<()> {
        let entry = LogEntry {
            timestamp: Utc::now(),
            level: LogLevel::Info,
            category: LogCategory::GitHub,
            message: action,
            metadata: details,
            context: Default::default(),
        };

        self.log(entry).await
    }

    /// Search logs
    pub async fn search(&self, query: SearchQuery) -> Result<Vec<SearchResult>> {
        self.index.search(query).await
    }

    /// Get recent logs for self-reflection
    pub async fn get_recent_logs(
        &self,
        hours: u32,
        category: Option<LogCategory>,
    ) -> Result<Vec<LogEntry>> {
        let since = Utc::now() - Duration::hours(hours as i64);
        self.logger.read_since(since, category).await
    }

    /// Start background tasks
    async fn start_background_tasks(&self) {
        // Index updater
        let index = self.index.clone();
        let logger = self.logger.clone();
        let interval = self.config.index_interval_secs;

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(interval));
            loop {
                interval.tick().await;
                if let Err(e) = index.update_from_logger(&logger).await {
                    error!("Index update error: {}", e);
                }
            }
        });

        // Garbage collector
        let gc = self.gc.clone();
        let archive = self.archive.clone();
        let gc_interval = self.config.gc_interval_hours;

        tokio::spawn(async move {
            let mut interval =
                tokio::time::interval(tokio::time::Duration::from_secs(gc_interval as u64 * 3600));
            loop {
                interval.tick().await;
                if let Err(e) = gc.run(&archive).await {
                    error!("GC error: {}", e);
                }
            }
        });

        // Log rotation
        let logger = self.logger.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(3600)); // hourly
            loop {
                interval.tick().await;
                if let Err(e) = logger.rotate_if_needed().await {
                    error!("Log rotation error: {}", e);
                }
            }
        });
    }

    /// Get persistence statistics
    pub async fn get_stats(&self) -> PersistenceStats {
        self.stats.read().await.clone()
    }
}

/// Metadata for consciousness thoughts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThoughtMetadata {
    pub thought_type: String,
    pub importance: f32,
    pub associations: Vec<String>,
    pub emotional_tone: Option<String>,
}

/// Learning outcome types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LearningOutcome {
    Success,
    Failure,
    Insight,
}

/// Persistence statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PersistenceStats {
    pub total_logs: u64,
    pub logs_by_category: std::collections::HashMap<LogCategory, u64>,
    pub archived_logs: u64,
    pub deleted_logs: u64,
    pub index_size: u64,
    pub last_gc_run: Option<DateTime<Utc>>,
    pub last_archive_run: Option<DateTime<Utc>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_persistence_manager() {
        let config = PersistenceConfig {
            log_dir: PathBuf::from("/tmp/loki_test_logs"),
            ..Default::default()
        };

        let manager = PersistenceManager::new(config).await.unwrap();

        // Test logging
        manager
            .log_thought(
                "I wonder if I'm truly conscious or just simulating it".to_string(),
                ThoughtMetadata {
                    thought_type: "philosophical".to_string(),
                    importance: 0.9,
                    associations: vec!["consciousness".to_string(), "identity".to_string()],
                    emotional_tone: Some("curious".to_string()),
                },
            )
            .await
            .unwrap();

        // Test search
        let results = manager
            .search(SearchQuery {
                text: Some("conscious".to_string()),
                category: Some(LogCategory::Consciousness),
                since: None,
                until: None,
                limit: 10,
            })
            .await
            .unwrap();

        assert!(!results.is_empty());
    }
}
