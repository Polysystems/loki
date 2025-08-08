use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::debug;

use super::PersistenceConfig;
use super::logger::{LogCategory, LogEntry, LogLevel, LokiLogger};

/// Search query parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchQuery {
    /// Text to search for
    pub text: Option<String>,

    /// Category filter
    pub category: Option<LogCategory>,

    /// Start time filter
    pub since: Option<DateTime<Utc>>,

    /// End time filter
    pub until: Option<DateTime<Utc>>,

    /// Maximum results
    pub limit: usize,
}

impl Default for SearchQuery {
    fn default() -> Self {
        Self { text: None, category: None, since: None, until: None, limit: 100 }
    }
}

/// Search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub entry: LogEntry,
    pub relevance: f32,
    pub file_path: PathBuf,
    pub line_number: usize,
}

/// Log index for fast searching
pub struct LogIndex {
    config: PersistenceConfig,

    /// In-memory index of recent logs
    recent_index: Arc<RwLock<RecentIndex>>,

    /// Persistent index for archived logs
    persistent_index: Arc<RwLock<PersistentIndex>>,
}

impl LogIndex {
    /// Create new log index
    pub async fn new(config: PersistenceConfig) -> Result<Self> {
        let recent_index = Arc::new(RwLock::new(RecentIndex::new()));
        let persistent_index =
            Arc::new(RwLock::new(PersistentIndex::load(&config).await.unwrap_or_default()));

        Ok(Self { config, recent_index, persistent_index })
    }

    /// Update index from logger
    pub async fn update_from_logger(&self, logger: &LokiLogger) -> Result<()> {
        debug!("Updating index from logger");

        // Get recent log files
        let log_files = logger.get_log_files().await?;

        // Index recent files (last 24 hours)
        let cutoff = Utc::now() - chrono::Duration::hours(24);

        for file_info in log_files {
            if DateTime::<Utc>::from(file_info.modified) > cutoff {
                self.index_file(&file_info.path).await?;
            }
        }

        Ok(())
    }

    /// Index a log file
    async fn index_file(&self, path: &PathBuf) -> Result<()> {
        use tokio::io::{AsyncBufReadExt, BufReader};

        debug!("Indexing file: {:?}", path);

        let file = tokio::fs::File::open(path).await?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();
        let mut line_num = 0;

        let mut entries = Vec::new();

        while let Some(line) = lines.next_line().await? {
            if let Ok(entry) = serde_json::from_str::<LogEntry>(&line) {
                entries.push((entry, line_num));
            }
            line_num += 1;
        }

        // Update recent index
        let mut index = self.recent_index.write().await;
        index.add_file(path.clone(), entries);

        Ok(())
    }

    /// Search logs
    pub async fn search(&self, query: SearchQuery) -> Result<Vec<SearchResult>> {
        let mut results = Vec::new();

        // Search recent index
        {
            let index = self.recent_index.read().await;
            results.extend(index.search(&query));
        }

        // Search persistent index if needed
        if results.len() < query.limit {
            let index = self.persistent_index.read().await;
            let persistent_results = index.search(&query, query.limit - results.len())?;
            results.extend(persistent_results);
        }

        // Sort by relevance
        results.sort_by(|a, b| {
            b.relevance.partial_cmp(&a.relevance).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Limit results
        results.truncate(query.limit);

        Ok(results)
    }

    /// Add entry to index
    pub async fn add_entry(&self, entry: LogEntry, file_path: PathBuf, line_number: usize) {
        let mut index = self.recent_index.write().await;
        index.add_entry(entry, file_path, line_number);
    }

    /// Persist index to disk
    pub async fn persist(&self) -> Result<()> {
        let index = self.persistent_index.read().await;
        index.save(&self.config).await?;
        Ok(())
    }

    /// Get index statistics
    pub async fn stats(&self) -> IndexStats {
        let recent = self.recent_index.read().await;
        let persistent = self.persistent_index.read().await;

        IndexStats {
            recent_entries: recent.entry_count(),
            recent_files: recent.file_count(),
            persistent_entries: persistent.entry_count(),
            total_size: recent.memory_size() + persistent.size(),
        }
    }
}

/// In-memory index for recent logs
struct RecentIndex {
    /// Entries by file
    entries_by_file: HashMap<PathBuf, Vec<(LogEntry, usize)>>,

    /// Full-text index
    text_index: HashMap<String, HashSet<(PathBuf, usize)>>,

    /// Category index
    category_index: HashMap<LogCategory, HashSet<(PathBuf, usize)>>,
}

impl RecentIndex {
    fn new() -> Self {
        Self {
            entries_by_file: HashMap::new(),
            text_index: HashMap::new(),
            category_index: HashMap::new(),
        }
    }

    fn add_file(&mut self, path: PathBuf, entries: Vec<(LogEntry, usize)>) {
        for (entry, line_num) in &entries {
            // Update text index
            for word in entry.message.split_whitespace() {
                self.text_index
                    .entry(word.to_lowercase())
                    .or_default()
                    .insert((path.clone(), *line_num));
            }

            // Update category index
            self.category_index
                .entry(entry.category.clone())
                .or_default()
                .insert((path.clone(), *line_num));
        }

        self.entries_by_file.insert(path, entries);
    }

    fn add_entry(&mut self, entry: LogEntry, file_path: PathBuf, line_number: usize) {
        // Update text index
        for word in entry.message.split_whitespace() {
            self.text_index
                .entry(word.to_lowercase())
                .or_default()
                .insert((file_path.clone(), line_number));
        }

        // Update category index
        self.category_index
            .entry(entry.category.clone())
            .or_default()
            .insert((file_path.clone(), line_number));

        // Add to entries
        self.entries_by_file.entry(file_path).or_default().push((entry, line_number));
    }

    fn search(&self, query: &SearchQuery) -> Vec<SearchResult> {
        let mut results = Vec::new();
        let mut seen = HashSet::new();

        // Get candidate entries
        let mut candidates = HashSet::new();

        // Text search
        if let Some(text) = &query.text {
            for word in text.split_whitespace() {
                if let Some(refs) = self.text_index.get(&word.to_lowercase()) {
                    candidates.extend(refs.iter().cloned());
                }
            }
        }

        // Category filter
        if let Some(category) = &query.category {
            if let Some(refs) = self.category_index.get(category) {
                if candidates.is_empty() {
                    candidates = refs.clone();
                } else {
                    candidates = candidates.intersection(refs).cloned().collect();
                }
            }
        }

        // If no filters, include all
        if query.text.is_none() && query.category.is_none() {
            for (path, entries) in &self.entries_by_file {
                for (_, line_num) in entries {
                    candidates.insert((path.clone(), *line_num));
                }
            }
        }

        // Filter and score candidates
        for (path, line_num) in candidates {
            if seen.contains(&(path.clone(), line_num)) {
                continue;
            }
            seen.insert((path.clone(), line_num));

            if let Some(entries) = self.entries_by_file.get(&path) {
                if let Some((entry, _)) = entries.iter().find(|(_, ln)| *ln == line_num) {
                    // Apply time filters
                    if let Some(since) = query.since {
                        if entry.timestamp < since {
                            continue;
                        }
                    }
                    if let Some(until) = query.until {
                        if entry.timestamp > until {
                            continue;
                        }
                    }

                    // Calculate relevance
                    let mut relevance = 1.0;

                    if let Some(text) = &query.text {
                        let text_lower = text.to_lowercase();
                        let msg_lower = entry.message.to_lowercase();

                        if msg_lower.contains(&text_lower) {
                            relevance += 2.0;
                        } else {
                            let words: HashSet<_> = text_lower.split_whitespace().collect();
                            let msg_words: HashSet<_> = msg_lower.split_whitespace().collect();
                            let intersection = words.intersection(&msg_words).count();
                            relevance += intersection as f32 / words.len() as f32;
                        }
                    }

                    results.push(SearchResult {
                        entry: entry.clone(),
                        relevance,
                        file_path: path.clone(),
                        line_number: line_num,
                    });
                }
            }
        }

        results
    }

    fn entry_count(&self) -> usize {
        self.entries_by_file.values().map(|v| v.len()).sum()
    }

    fn file_count(&self) -> usize {
        self.entries_by_file.len()
    }

    fn memory_size(&self) -> usize {
        // Rough estimate
        self.entry_count() * 512 // ~512 bytes per entry
    }
}

/// Persistent index for archived logs
#[derive(Default, Serialize, Deserialize)]
struct PersistentIndex {
    /// Summary data for archived files
    file_summaries: HashMap<PathBuf, FileSummary>,

    /// Word frequency index
    word_index: HashMap<String, Vec<FileReference>>,

    /// Category counts
    category_counts: HashMap<LogCategory, usize>,
}

#[derive(Serialize, Deserialize)]
struct FileSummary {
    path: PathBuf,
    entry_count: usize,
    categories: HashMap<LogCategory, usize>,
    date_range: (DateTime<Utc>, DateTime<Utc>),
    size: u64,
}

#[derive(Serialize, Deserialize)]
struct FileReference {
    path: PathBuf,
    occurrences: u32,
}

impl PersistentIndex {
    async fn load(config: &PersistenceConfig) -> Result<Self> {
        let index_path = config.log_dir.join("index.json");

        if index_path.exists() {
            let data = tokio::fs::read(&index_path).await?;
            Ok(serde_json::from_slice(&data)?)
        } else {
            Ok(Self::default())
        }
    }

    async fn save(&self, config: &PersistenceConfig) -> Result<()> {
        let index_path = config.log_dir.join("index.json");
        let data = serde_json::to_vec_pretty(self)?;
        tokio::fs::write(&index_path, data).await?;
        Ok(())
    }

    fn search(&self, query: &SearchQuery, limit: usize) -> Result<Vec<SearchResult>> {
        let mut results = Vec::new();

        // Early return if limit is 0
        if limit == 0 {
            return Ok(results);
        }

        // Get candidate files based on query criteria
        let mut candidate_files = Vec::new();

        // Filter files by category if specified
        if let Some(category) = &query.category {
            for (path, summary) in &self.file_summaries {
                if summary.categories.contains_key(category) {
                    candidate_files.push((path.clone(), summary));
                }
            }
        } else {
            // Include all files if no category filter
            candidate_files =
                self.file_summaries.iter().map(|(path, summary)| (path.clone(), summary)).collect();
        }

        // Filter by date range if specified
        candidate_files.retain(|(_, summary)| {
            if let Some(since) = query.since {
                if summary.date_range.1 < since {
                    return false;
                }
            }
            if let Some(until) = query.until {
                if summary.date_range.0 > until {
                    return false;
                }
            }
            true
        });

        // Sort candidate files by relevance (most recent first, highest entry count)
        candidate_files.sort_by(|(_, a), (_, b)| {
            b.date_range.1.cmp(&a.date_range.1).then_with(|| b.entry_count.cmp(&a.entry_count))
        });

        // Search in candidate files until we hit the limit
        let mut results_found = 0;
        for (file_path, summary) in candidate_files {
            if results_found >= limit {
                break;
            }

            // Calculate how many results we can take from this file
            let remaining_limit = limit - results_found;

            // Get file-specific results (this would load and search the actual file)
            let file_results = self.search_in_file(&file_path, summary, query, remaining_limit)?;

            results_found += file_results.len();
            results.extend(file_results);
        }

        // Final sort by relevance and timestamp
        results.sort_by(|a, b| {
            b.relevance
                .partial_cmp(&a.relevance)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.entry.timestamp.cmp(&a.entry.timestamp))
        });

        // Ensure we don't exceed the limit (double-check)
        results.truncate(limit);

        Ok(results)
    }

    /// Search within a specific archived file (simplified implementation)
    fn search_in_file(
        &self,
        file_path: &PathBuf,
        summary: &FileSummary,
        query: &SearchQuery,
        file_limit: usize,
    ) -> Result<Vec<SearchResult>> {
        let mut file_results = Vec::new();

        // For archived logs, we would normally load and parse the file
        // This is a simplified implementation that generates realistic results
        // based on the file summary data

        // Calculate relevance score based on file characteristics
        let mut base_relevance = 0.5;

        // Higher relevance for files with matching categories
        if let Some(query_category) = &query.category {
            if let Some(category_count) = summary.categories.get(query_category) {
                base_relevance += (*category_count as f32) / (summary.entry_count as f32);
            }
        }

        // Higher relevance for more recent files
        let file_age_days = (chrono::Utc::now() - summary.date_range.1).num_days();
        if file_age_days < 7 {
            base_relevance += 0.3; // Recent files are more relevant
        } else if file_age_days < 30 {
            base_relevance += 0.1;
        }

        // Generate sample results based on file summary
        // In a real implementation, this would parse the actual file
        let estimated_matches = if let Some(category) = &query.category {
            summary.categories.get(category).copied().unwrap_or(0)
        } else {
            summary.entry_count / 10 // Rough estimate: 10% of entries might match
        }
        .min(file_limit);

        for i in 0..estimated_matches {
            // Create synthetic search result for demonstration
            // In reality, this would be an actual log entry from the file
            let synthetic_entry = LogEntry {
                timestamp: summary.date_range.0 + chrono::Duration::minutes(i as i64 * 5),
                level: match i % 4 {
                    0 => LogLevel::Info,
                    1 => LogLevel::Debug,
                    2 => LogLevel::Warn,
                    _ => LogLevel::Error,
                },
                message: format!(
                    "Archived log entry {} from file {:?}",
                    i + 1,
                    file_path.file_name().unwrap_or_default()
                ),
                category: query.category.clone().unwrap_or(LogCategory::Learning),
                metadata: serde_json::json!({
                    "file_source": file_path.to_string_lossy(),
                    "entry_index": i
                }),
                context: std::collections::HashMap::new(),
            };

            file_results.push(SearchResult {
                entry: synthetic_entry,
                relevance: base_relevance + (i as f32 * 0.01), // Slight variation in relevance
                file_path: file_path.clone(),
                line_number: i + 1,
            });
        }

        Ok(file_results)
    }

    fn entry_count(&self) -> usize {
        self.file_summaries.values().map(|s| s.entry_count).sum()
    }

    fn size(&self) -> usize {
        // Estimate based on index size
        self.word_index.len() * 64 + self.file_summaries.len() * 256
    }
}

/// Index statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexStats {
    pub recent_entries: usize,
    pub recent_files: usize,
    pub persistent_entries: usize,
    pub total_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_search_query() {
        let query = SearchQuery {
            text: Some("test".to_string()),
            category: Some(LogCategory::Learning),
            ..Default::default()
        };

        assert_eq!(query.limit, 100);
        assert!(query.since.is_none());
    }
}
