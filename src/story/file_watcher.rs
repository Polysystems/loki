//! File system watcher for automatic story generation

use super::types::*;
use super::engine::StoryEngine;
use anyhow::{Context as _, Result};
use notify::{Config, Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{broadcast, RwLock};
use tracing::{debug, info, warn};

/// File watcher that generates stories based on file system changes
#[derive(Debug)]
pub struct StoryFileWatcher {
    /// Story engine reference
    story_engine: Arc<StoryEngine>,
    
    /// File system watcher
    watcher: Option<RecommendedWatcher>,
    
    /// Watched paths and their story IDs
    watched_paths: Arc<RwLock<HashMap<PathBuf, StoryId>>>,
    
    /// Event channel
    event_tx: broadcast::Sender<FileEvent>,
    
    /// Configuration
    config: FileWatcherConfig,
    
    /// Shutdown signal
    shutdown_tx: broadcast::Sender<()>,
}

/// Configuration for file watcher
#[derive(Debug, Clone)]
pub struct FileWatcherConfig {
    /// Debounce duration for file events
    pub debounce_duration: Duration,
    
    /// File patterns to ignore
    pub ignore_patterns: Vec<String>,
    
    /// Enable automatic story generation
    pub auto_generate_stories: bool,
    
    /// Track file content changes
    pub track_content_changes: bool,
    
    /// Maximum file size to analyze (in bytes)
    pub max_file_size: usize,
}

impl Default for FileWatcherConfig {
    fn default() -> Self {
        Self {
            debounce_duration: Duration::from_millis(500),
            ignore_patterns: vec![
                ".git".to_string(),
                "target".to_string(),
                "node_modules".to_string(),
                ".DS_Store".to_string(),
                "*.tmp".to_string(),
                "*.swp".to_string(),
            ],
            auto_generate_stories: true,
            track_content_changes: true,
            max_file_size: 10 * 1024 * 1024, // 10MB
        }
    }
}

/// File system events relevant to story generation
#[derive(Debug, Clone)]
pub struct FileEvent {
    pub path: PathBuf,
    pub event_type: FileEventType,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub metadata: Option<FileMetadata>,
}

/// Types of file events
#[derive(Debug, Clone, PartialEq)]
pub enum FileEventType {
    Created,
    Modified,
    Deleted,
    Renamed { from: PathBuf, to: PathBuf },
}

/// File metadata for story generation
#[derive(Debug, Clone)]
pub struct FileMetadata {
    pub size: u64,
    pub is_directory: bool,
    pub extension: Option<String>,
    pub detected_language: Option<String>,
}

impl StoryFileWatcher {
    /// Create a new file watcher
    pub fn new(
        story_engine: Arc<StoryEngine>,
        config: FileWatcherConfig,
    ) -> Result<Self> {
        let (event_tx, _) = broadcast::channel(1000);
        let (shutdown_tx, _) = broadcast::channel(1);
        
        Ok(Self {
            story_engine,
            watcher: None,
            watched_paths: Arc::new(RwLock::new(HashMap::new())),
            event_tx,
            config,
            shutdown_tx,
        })
    }
    
    /// Start watching a directory
    pub async fn watch_directory(&mut self, path: impl AsRef<Path>) -> Result<StoryId> {
        let path = path.as_ref().to_path_buf();
        
        // Check if already watching
        let watched = self.watched_paths.read().await;
        if let Some(story_id) = watched.get(&path) {
            return Ok(*story_id);
        }
        drop(watched);
        
        // Create or get codebase story
        let story_id = self.story_engine
            .create_codebase_story(
                path.clone(),
                detect_project_language(&path).unwrap_or_else(|| "Unknown".to_string()),
            )
            .await?;
        
        // Store watched path
        self.watched_paths.write().await.insert(path.clone(), story_id);
        
        // Initialize watcher if needed
        if self.watcher.is_none() {
            self.initialize_watcher()?;
        }
        
        // Add path to watcher
        if let Some(watcher) = &mut self.watcher {
            watcher
                .watch(&path, RecursiveMode::Recursive)
                .context("Failed to watch directory")?;
        }
        
        info!("Started watching directory: {} -> Story {}", path.display(), story_id.0);
        
        // Start event processor if not already running
        self.start_event_processor().await;
        
        Ok(story_id)
    }
    
    /// Stop watching a directory
    pub async fn unwatch_directory(&mut self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref().to_path_buf();
        
        if let Some(watcher) = &mut self.watcher {
            watcher.unwatch(&path)?;
        }
        
        self.watched_paths.write().await.remove(&path);
        
        info!("Stopped watching directory: {}", path.display());
        
        Ok(())
    }
    
    /// Initialize the file system watcher
    fn initialize_watcher(&mut self) -> Result<()> {
        let event_tx = self.event_tx.clone();
        let config = Config::default()
            .with_poll_interval(self.config.debounce_duration);
        
        let watcher = RecommendedWatcher::new(
            move |res: Result<Event, notify::Error>| {
                match res {
                    Ok(event) => {
                        if let Err(e) = handle_notify_event(event, &event_tx) {
                            warn!("Failed to handle file event: {}", e);
                        }
                    }
                    Err(e) => warn!("File watcher error: {}", e),
                }
            },
            config,
        )?;
        
        self.watcher = Some(watcher);
        Ok(())
    }
    
    /// Start the event processor
    async fn start_event_processor(&self) {
        let story_engine = self.story_engine.clone();
        let mut event_rx = self.event_tx.subscribe();
        let watched_paths = self.watched_paths.clone();
        let config = self.config.clone();
        let mut shutdown_rx = self.shutdown_tx.subscribe();
        
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    Ok(event) = event_rx.recv() => {
                        if let Err(e) = process_file_event(
                            event,
                            &story_engine,
                            &watched_paths,
                            &config,
                        ).await {
                            warn!("Failed to process file event: {}", e);
                        }
                    }
                    _ = shutdown_rx.recv() => {
                        info!("File watcher event processor shutting down");
                        break;
                    }
                }
            }
        });
    }
    
    /// Get story for a file path
    pub async fn get_file_story(&self, path: impl AsRef<Path>) -> Option<StoryId> {
        let path = path.as_ref();
        let watched = self.watched_paths.read().await;
        
        // Find the closest watched parent directory
        for (watched_path, story_id) in watched.iter() {
            if path.starts_with(watched_path) {
                return Some(*story_id);
            }
        }
        
        None
    }
    
    /// Shutdown the file watcher
    pub async fn shutdown(&self) -> Result<()> {
        let _ = self.shutdown_tx.send(());
        Ok(())
    }
    
    /// Watch a specific path
    pub async fn watch_path(&mut self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        info!("Starting to watch path: {:?}", path);
        
        // Create watcher if needed
        if self.watcher.is_none() {
            let (tx, rx) = std::sync::mpsc::channel();
            let mut watcher = RecommendedWatcher::new(tx, Config::default())?;
            watcher.watch(path, RecursiveMode::Recursive)?;
            self.watcher = Some(watcher);
            
            // Start event handler in background
            let event_tx = self.event_tx.clone();
            tokio::spawn(async move {
                while let Ok(result) = rx.recv() {
                    if let Ok(event) = result {
                        let _ = handle_notify_event(event, &event_tx);
                    }
                }
            });
        } else if let Some(watcher) = &mut self.watcher {
            watcher.watch(path, RecursiveMode::Recursive)?;
        }
        
        // Track this path
        let story_id = crate::story::StoryId(uuid::Uuid::new_v4());
        self.watched_paths.write().await.insert(path.to_path_buf(), story_id);
        
        Ok(())
    }
    
    /// Stop watching a specific path
    pub async fn unwatch_path(&mut self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        info!("Stopping watch on path: {:?}", path);
        
        if let Some(watcher) = &mut self.watcher {
            watcher.unwatch(path)?;
        }
        
        // Remove from tracked paths
        self.watched_paths.write().await.remove(path);
        
        Ok(())
    }
}

/// Handle notify events and convert to our event type
fn handle_notify_event(
    event: Event,
    event_tx: &broadcast::Sender<FileEvent>,
) -> Result<()> {
    let event_type = match event.kind {
        EventKind::Create(_) => FileEventType::Created,
        EventKind::Modify(_) => FileEventType::Modified,
        EventKind::Remove(_) => FileEventType::Deleted,
        EventKind::Any => return Ok(()), // Ignore generic events
        _ => return Ok(()), // Ignore other events
    };
    
    for path in event.paths {
        let file_event = FileEvent {
            path: path.clone(),
            event_type: event_type.clone(),
            timestamp: chrono::Utc::now(),
            metadata: get_file_metadata(&path),
        };
        
        // Try to send, but don't block
        let _ = event_tx.send(file_event);
    }
    
    Ok(())
}

/// Process a file event and update stories
async fn process_file_event(
    event: FileEvent,
    story_engine: &Arc<StoryEngine>,
    watched_paths: &Arc<RwLock<HashMap<PathBuf, StoryId>>>,
    config: &FileWatcherConfig,
) -> Result<()> {
    // Check ignore patterns
    if should_ignore_path(&event.path, &config.ignore_patterns) {
        return Ok(());
    }
    
    // Find the story for this file
    let watched = watched_paths.read().await;
    let story_id = match find_story_for_path(&event.path, &watched) {
        Some(id) => id,
        None => return Ok(()), // No story tracking this path
    };
    drop(watched);
    
    // Generate plot points based on event type
    match event.event_type {
        FileEventType::Created => {
            story_engine.add_plot_point(
                story_id,
                PlotType::Discovery {
                    insight: format!("New file created: {}", event.path.display()),
                },
                vec![event.path.to_string_lossy().to_string()],
            ).await?;
            
            // Analyze file content if enabled
            if config.track_content_changes && !is_binary(&event.path) {
                if let Ok(content) = tokio::fs::read_to_string(&event.path).await {
                    analyze_file_content(&content, &event.path, story_id, story_engine).await?;
                }
            }
        }
        
        FileEventType::Modified => {
            story_engine.add_plot_point(
                story_id,
                PlotType::Transformation {
                    before: format!("File before modification: {}", event.path.display()),
                    after: format!("File modified at {}", event.timestamp),
                },
                vec![event.path.to_string_lossy().to_string()],
            ).await?;
            
            // Track specific changes if possible
            if config.track_content_changes && !is_binary(&event.path) {
                if let Ok(content) = tokio::fs::read_to_string(&event.path).await {
                    detect_code_changes(&content, &event.path, story_id, story_engine).await?;
                }
            }
        }
        
        FileEventType::Deleted => {
            story_engine.add_plot_point(
                story_id,
                PlotType::Issue {
                    error: format!("File deleted: {}", event.path.display()),
                    resolved: false,
                },
                vec![event.path.to_string_lossy().to_string()],
            ).await?;
        }
        
        FileEventType::Renamed { from, to } => {
            story_engine.add_plot_point(
                story_id,
                PlotType::Transformation {
                    before: format!("File: {}", from.display()),
                    after: format!("Renamed to: {}", to.display()),
                },
                vec![
                    from.to_string_lossy().to_string(),
                    to.to_string_lossy().to_string(),
                ],
            ).await?;
        }
    }
    
    Ok(())
}

/// Analyze file content and generate relevant plot points
async fn analyze_file_content(
    content: &str,
    path: &Path,
    story_id: StoryId,
    story_engine: &Arc<StoryEngine>,
) -> Result<()> {
    // Look for TODOs
    for (line_num, line) in content.lines().enumerate() {
        if line.contains("TODO") || line.contains("FIXME") {
            story_engine.add_plot_point(
                story_id,
                PlotType::Task {
                    description: format!("{}:{} - {}", 
                        path.display(), 
                        line_num + 1, 
                        line.trim()
                    ),
                    completed: false,
                },
                vec![format!("line:{}", line_num + 1)],
            ).await?;
        }
    }
    
    // Look for function definitions (simple heuristic)
    if path.extension().and_then(|e| e.to_str()) == Some("rs") {
        for (line_num, line) in content.lines().enumerate() {
            if line.trim_start().starts_with("pub fn") || 
               line.trim_start().starts_with("fn") {
                if let Some(fn_name) = extract_function_name(line) {
                    story_engine.add_plot_point(
                        story_id,
                        PlotType::Discovery {
                            insight: format!("Function defined: {} at {}:{}", 
                                fn_name, 
                                path.display(), 
                                line_num + 1
                            ),
                        },
                        vec![format!("function:{}", fn_name)],
                    ).await?;
                }
            }
        }
    }
    
    Ok(())
}

/// Detect specific code changes
async fn detect_code_changes(
    content: &str,
    path: &Path,
    story_id: StoryId,
    story_engine: &Arc<StoryEngine>,
) -> Result<()> {
    // This is a simplified version - in production, you'd compare with previous version
    
    // Count lines
    let line_count = content.lines().count();
    
    // Look for error patterns
    if content.contains("error!") || content.contains("panic!") {
        story_engine.add_plot_point(
            story_id,
            PlotType::Issue {
                error: format!("Error handling code detected in {}", path.display()),
                resolved: false,
            },
            vec!["error_handling".to_string()],
        ).await?;
    }
    
    // Look for test additions
    if content.contains("#[test]") || content.contains("#[tokio::test]") {
        story_engine.add_plot_point(
            story_id,
            PlotType::Goal {
                objective: format!("Tests added/modified in {}", path.display()),
            },
            vec!["testing".to_string()],
        ).await?;
    }
    
    debug!("Analyzed {} lines in {}", line_count, path.display());
    
    Ok(())
}

/// Helper functions

fn get_file_metadata(path: &Path) -> Option<FileMetadata> {
    match std::fs::metadata(path) {
        Ok(metadata) => Some(FileMetadata {
            size: metadata.len(),
            is_directory: metadata.is_dir(),
            extension: path.extension()
                .and_then(|e| e.to_str())
                .map(|s| s.to_string()),
            detected_language: detect_file_language(path),
        }),
        Err(_) => None,
    }
}

fn should_ignore_path(path: &Path, ignore_patterns: &[String]) -> bool {
    let path_str = path.to_string_lossy();
    
    for pattern in ignore_patterns {
        if path_str.contains(pattern) {
            return true;
        }
    }
    
    false
}

fn find_story_for_path(
    path: &Path,
    watched_paths: &HashMap<PathBuf, StoryId>,
) -> Option<StoryId> {
    // Find the closest watched parent directory
    for (watched_path, story_id) in watched_paths {
        if path.starts_with(watched_path) {
            return Some(*story_id);
        }
    }
    None
}

fn is_binary(path: &Path) -> bool {
    match path.extension().and_then(|e| e.to_str()) {
        Some(ext) => matches!(
            ext,
            "exe" | "dll" | "so" | "dylib" | "png" | "jpg" | "jpeg" | 
            "gif" | "pdf" | "zip" | "tar" | "gz" | "bin"
        ),
        None => false,
    }
}

fn detect_project_language(path: &Path) -> Option<String> {
    // Simple language detection based on project files
    if path.join("Cargo.toml").exists() {
        Some("Rust".to_string())
    } else if path.join("package.json").exists() {
        Some("JavaScript/TypeScript".to_string())
    } else if path.join("requirements.txt").exists() || path.join("setup.py").exists() {
        Some("Python".to_string())
    } else if path.join("go.mod").exists() {
        Some("Go".to_string())
    } else {
        None
    }
}

fn detect_file_language(path: &Path) -> Option<String> {
    path.extension()
        .and_then(|e| e.to_str())
        .and_then(|ext| match ext {
            "rs" => Some("Rust"),
            "js" | "jsx" => Some("JavaScript"),
            "ts" | "tsx" => Some("TypeScript"),
            "py" => Some("Python"),
            "go" => Some("Go"),
            "java" => Some("Java"),
            "cpp" | "cc" | "cxx" => Some("C++"),
            "c" => Some("C"),
            "rb" => Some("Ruby"),
            _ => None,
        })
        .map(|s| s.to_string())
}

fn extract_function_name(line: &str) -> Option<String> {
    // Simple regex-like extraction
    let line = line.trim();
    
    if let Some(start) = line.find("fn ") {
        let rest = &line[start + 3..];
        if let Some(end) = rest.find('(') {
            let name = rest[..end].trim();
            if !name.is_empty() {
                return Some(name.to_string());
            }
        }
    }
    
    None
}