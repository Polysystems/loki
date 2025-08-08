//! Story export and import functionality

use super::types::*;
use super::engine::StoryEngine;
use crate::story::context_chain::{ContextSegment, SegmentId, ContextLink, ChainMetadata};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tracing::{debug, info, warn};

/// Story export format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoryExport {
    pub version: String,
    pub exported_at: chrono::DateTime<chrono::Utc>,
    pub stories: Vec<ExportedStory>,
    pub context_chains: Vec<ExportedContextChain>,
    pub metadata: ExportMetadata,
}

/// Represents a relationship between stories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoryRelationship {
    pub from_story: StoryId,
    pub to_story: StoryId,
    pub relationship_type: String,
    pub metadata: HashMap<String, String>,
}

/// Exported story data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportedStory {
    pub id: StoryId,
    pub title: String,
    pub summary: String,
    pub story_type: StoryType,
    pub plot_points: Vec<PlotPoint>,
    pub arcs: Vec<StoryArc>,
    pub metadata: StoryMetadata,
    pub context: HashMap<String, String>,
    pub relationships: Vec<StoryRelationship>,
}

/// Exported context chain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportedContextChain {
    pub id: ChainId,
    pub story_id: StoryId,
    pub segments: Vec<ContextSegment>,
    pub links: HashMap<SegmentId, Vec<ContextLink>>,
    pub metadata: ChainMetadata,
}

/// Export metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportMetadata {
    pub total_stories: usize,
    pub total_plot_points: usize,
    pub total_context_chains: usize,
    pub export_options: ExportOptions,
}

/// Export options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportOptions {
    pub include_completed: bool,
    pub include_context_chains: bool,
    pub compress: bool,
    pub format: ExportFormat,
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            include_completed: true,
            include_context_chains: true,
            compress: false,
            format: ExportFormat::Json,
        }
    }
}

/// Export format
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ExportFormat {
    Json,
    Yaml,
    MessagePack,
}

/// Story importer/exporter
pub struct StoryPorter {
    story_engine: Arc<StoryEngine>,
}

impl StoryPorter {
    /// Create a new story porter
    pub fn new(story_engine: Arc<StoryEngine>) -> Self {
        Self { story_engine }
    }

    /// Export stories to file
    pub async fn export_to_file(
        &self,
        path: &Path,
        options: ExportOptions,
    ) -> Result<()> {
        let export_data = self.export_stories(options.clone()).await?;

        match options.format {
            ExportFormat::Json => {
                let json = serde_json::to_string_pretty(&export_data)?;
                if options.compress {
                    self.write_compressed(path, json.as_bytes()).await?;
                } else {
                    fs::write(path, json)?;
                }
            }
            ExportFormat::Yaml => {
                let yaml = serde_yaml::to_string(&export_data)?;
                if options.compress {
                    self.write_compressed(path, yaml.as_bytes()).await?;
                } else {
                    fs::write(path, yaml)?;
                }
            }
            ExportFormat::MessagePack => {
                let msgpack = rmp_serde::to_vec(&export_data)?;
                if options.compress {
                    self.write_compressed(path, &msgpack).await?;
                } else {
                    fs::write(path, msgpack)?;
                }
            }
        }

        info!("Exported {} stories to {}", export_data.stories.len(), path.display());
        Ok(())
    }

    /// Import stories from file
    pub async fn import_from_file(
        &self,
        path: &Path,
        merge_strategy: MergeStrategy,
    ) -> Result<ImportResult> {
        let data = fs::read(path)?;

        // Try to detect format
        let export_data: StoryExport = if data.starts_with(b"{") {
            // JSON
            serde_json::from_slice(&data)?
        } else if data.starts_with(b"version:") || data.starts_with(b"---") {
            // YAML
            serde_yaml::from_slice(&data)?
        } else {
            // Try MessagePack or compressed
            match rmp_serde::from_slice(&data) {
                Ok(export) => export,
                Err(_) => {
                    // Try decompressing
                    let decompressed = self.read_compressed(path).await?;
                    serde_json::from_slice(&decompressed)?
                }
            }
        };

        self.import_stories(export_data, merge_strategy).await
    }

    /// Export stories
    pub async fn export_stories(&self, options: ExportOptions) -> Result<StoryExport> {
        let mut exported_stories = Vec::new();
        let mut exported_chains = Vec::new();
        let mut total_plot_points = 0;

        // Export all stories
        for story_ref in self.story_engine.stories.iter() {
            let story = story_ref.value();

            // Skip completed stories if not included
            if !options.include_completed && self.is_story_completed(story) {
                continue;
            }

            total_plot_points += story.arcs.iter()
                .map(|arc| arc.plot_points.len())
                .sum::<usize>();

            exported_stories.push(ExportedStory {
                id: story.id,
                title: story.title.clone(),
                summary: story.summary.clone(), // Added missing summary field
                story_type: story.story_type.clone(),
                plot_points: story.arcs.iter()
                    .flat_map(|arc| arc.plot_points.clone())
                    .collect(),
                arcs: story.arcs.clone(),
                metadata: story.metadata.clone(),
                context: HashMap::new(), // Story no longer has context field
                relationships: vec![], // Story no longer has relationships field
            });
        }

        // Export context chains if requested
        if options.include_context_chains {
            for chain_ref in self.story_engine.context_chains.iter() {
                let chain = chain_ref.value();

                let segments = chain.segments.read().await.clone();
                let links = chain.links.read().await.clone();
                let metadata = chain.metadata.read().await.clone();

                exported_chains.push(ExportedContextChain {
                    id: chain.id,
                    story_id: chain.story_id,
                    segments: segments.into_iter().collect(),
                    links,
                    metadata,
                });
            }
        }

        let total_stories = exported_stories.len();
        let total_context_chains = exported_chains.len();

        Ok(StoryExport {
            version: "1.0.0".to_string(),
            exported_at: chrono::Utc::now(),
            stories: exported_stories,
            context_chains: exported_chains,
            metadata: ExportMetadata {
                total_stories,
                total_plot_points,
                total_context_chains,
                export_options: options,
            },
        })
    }

    /// Import stories
    pub async fn import_stories(
        &self,
        export_data: StoryExport,
        merge_strategy: MergeStrategy,
    ) -> Result<ImportResult> {
        let mut imported_stories = 0;
        let mut skipped_stories = 0;
        let mut imported_chains = 0;
        let mut errors = Vec::new();

        // Import stories
        for exported_story in export_data.stories {
            match self.import_story(exported_story, &merge_strategy).await {
                Ok(imported) => {
                    if imported {
                        imported_stories += 1;
                    } else {
                        skipped_stories += 1;
                    }
                }
                Err(e) => {
                    errors.push(format!("Failed to import story: {}", e));
                    warn!("Failed to import story: {}", e);
                }
            }
        }

        // Import context chains
        for exported_chain in export_data.context_chains {
            match self.import_context_chain(exported_chain).await {
                Ok(_) => imported_chains += 1,
                Err(e) => {
                    errors.push(format!("Failed to import context chain: {}", e));
                    warn!("Failed to import context chain: {}", e);
                }
            }
        }

        info!(
            "Import complete: {} stories imported, {} skipped, {} chains imported",
            imported_stories, skipped_stories, imported_chains
        );

        Ok(ImportResult {
            imported_stories,
            skipped_stories,
            imported_chains,
            conflicts: vec![],
            errors,
        })
    }

    /// Export selected stories
    pub async fn export_selected_stories(
        &self,
        story_ids: Vec<StoryId>,
        options: ExportOptions,
    ) -> Result<StoryExport> {
        let mut exported_stories = Vec::new();
        let mut exported_chains = Vec::new();
        let mut total_plot_points = 0;

        for story_id in story_ids {
            if let Some(story) = self.story_engine.get_story(&story_id) {
                total_plot_points += story.arcs.iter()
                .map(|arc| arc.plot_points.len())
                .sum::<usize>();

                exported_stories.push(ExportedStory {
                    id: story.id,
                    title: story.title.clone(),
                    summary: story.summary.clone(), // Added missing summary field
                    story_type: story.story_type.clone(),
                    plot_points: story.arcs.iter()
                    .flat_map(|arc| arc.plot_points.clone())
                    .collect(),
                    arcs: story.arcs.clone(),
                    metadata: story.metadata.clone(),
                    context: HashMap::new(), // Story no longer has context field
                    relationships: vec![], // Story no longer has relationships field
                });

                // Export associated context chain
                if options.include_context_chains {
                    // Find chains for this story
                    for chain_ref in self.story_engine.context_chains.iter() {
                        let chain = chain_ref.value();
                        if chain.story_id == story_id {
                            let segments = chain.segments.read().await.clone();
                            let links = chain.links.read().await.clone();
                            let metadata = chain.metadata.read().await.clone();

                            exported_chains.push(ExportedContextChain {
                                id: chain.id,
                                story_id: chain.story_id,
                                segments: segments.into_iter().collect(),
                                links,
                                metadata,
                            });
                        }
                    }
                }
            }
        }

        let total_stories = exported_stories.len();
        let total_context_chains = exported_chains.len();

        Ok(StoryExport {
            version: "1.0.0".to_string(),
            exported_at: chrono::Utc::now(),
            stories: exported_stories,
            context_chains: exported_chains,
            metadata: ExportMetadata {
                total_stories,
                total_plot_points,
                total_context_chains,
                export_options: options,
            },
        })
    }

    /// Export stories by type
    pub async fn export_by_type(
        &self,
        story_type: StoryType,
        options: ExportOptions,
    ) -> Result<StoryExport> {
        let story_ids: Vec<StoryId> = self.story_engine.stories
            .iter()
            .filter(|s| s.value().story_type == story_type)
            .map(|s| s.value().id)
            .collect();

        self.export_selected_stories(story_ids, options).await
    }

    // Helper methods

    async fn import_story(
        &self,
        exported: ExportedStory,
        merge_strategy: &MergeStrategy,
    ) -> Result<bool> {
        // Check if story already exists
        if self.story_engine.stories.contains_key(&exported.id) {
            match merge_strategy {
                MergeStrategy::Skip => return Ok(false),
                MergeStrategy::Replace => {
                    self.story_engine.stories.remove(&exported.id);
                }
                MergeStrategy::Merge => {
                    // Merge plot points and context
                    let mut existing = self.story_engine.stories
                        .get_mut(&exported.id)
                        .unwrap();

                    // Merge plot points by merging arcs
                    // For simplicity, we'll merge all plot points into the current arc
                    if let Some(current_arc_id) = existing.current_arc {
                        if let Some(arc) = existing.arcs.iter_mut().find(|a| a.id == current_arc_id) {
                            for plot_point in exported.plot_points {
                                if !arc.plot_points.iter().any(|p| p.id == plot_point.id) {
                                    arc.plot_points.push(plot_point);
                                }
                            }
                        }
                    }

                    // Merge context
                    // Context field no longer exists on Story struct

                    return Ok(true);
                }
                MergeStrategy::RenameNew => {
                    // Create with new ID
                    let new_story = Story {
                        id: StoryId(uuid::Uuid::new_v4()),
                        title: format!("{} (imported)", exported.title),
                        description: String::new(),
                        story_type: exported.story_type,
                        summary: String::new(),
                        status: StoryStatus::Active,
                        arcs: exported.arcs,
                        current_arc: None,
                        metadata: exported.metadata,
                        context_chain: ChainId(uuid::Uuid::new_v4()),
                        created_at: chrono::Utc::now(),
                        updated_at: chrono::Utc::now(),
                        segments: vec![],
                        context: HashMap::new(),
                    };

                    self.story_engine.stories.insert(new_story.id, new_story);
                    return Ok(true);
                }
            }
        }

        // Import new story
        let story = Story {
            id: exported.id,
            title: exported.title.clone(),
            description: String::new(),
            story_type: exported.story_type,
            summary: String::new(),
            status: StoryStatus::Active,
            arcs: exported.arcs,
            current_arc: None,
            metadata: exported.metadata,
            context_chain: ChainId(uuid::Uuid::new_v4()),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            segments: vec![],
            context: HashMap::new(),
        };

        self.story_engine.stories.insert(story.id, story);
        Ok(true)
    }

    async fn import_context_chain(&self, exported: ExportedContextChain) -> Result<()> {
        use super::context_chain::ContextChain;
        use std::collections::VecDeque;
        use tokio::sync::RwLock;

        let chain = ContextChain {
            id: exported.id,
            story_id: exported.story_id,
            segments: RwLock::new(VecDeque::from(exported.segments)),
            links: RwLock::new(exported.links),
            metadata: RwLock::new(exported.metadata),
        };

        self.story_engine.context_chains.insert(chain.id, chain);
        Ok(())
    }

    fn is_story_completed(&self, story: &Story) -> bool {
        story.arcs.iter()
            .flat_map(|arc| &arc.plot_points)
            .all(|p| match &p.plot_type {
                PlotType::Task { completed, .. } => *completed,
                PlotType::Issue { resolved, .. } => *resolved,
                _ => true,
            })
    }

    async fn write_compressed(&self, path: &Path, data: &[u8]) -> Result<()> {
        use flate2::write::GzEncoder;
        use flate2::Compression;
        use std::io::Write;

        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(data)?;
        let compressed = encoder.finish()?;

        let mut file = tokio::fs::File::create(path).await?;
        file.write_all(&compressed).await?;

        Ok(())
    }

    async fn read_compressed(&self, path: &Path) -> Result<Vec<u8>> {
        use flate2::read::GzDecoder;
        use std::io::Read;

        let mut file = tokio::fs::File::open(path).await?;
        let mut compressed = Vec::new();
        file.read_to_end(&mut compressed).await?;

        let mut decoder = GzDecoder::new(&compressed[..]);
        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed)?;

        Ok(decompressed)
    }
}

/// Merge strategy for imports
#[derive(Debug, Clone, PartialEq)]
pub enum MergeStrategy {
    /// Skip existing stories
    Skip,
    /// Replace existing stories
    Replace,
    /// Merge plot points and context
    Merge,
    /// Create new stories with different IDs
    RenameNew,
}

/// Import conflict information
#[derive(Debug, Clone)]
pub struct ImportConflict {
    pub item_type: String,
    pub item_id: String,
    pub existing_title: Option<String>,
    pub imported_title: Option<String>,
    pub conflict_type: String,
}

/// Import result
#[derive(Debug, Clone)]
pub struct ImportResult {
    pub imported_stories: usize,
    pub skipped_stories: usize,
    pub imported_chains: usize,
    pub conflicts: Vec<ImportConflict>,
    pub errors: Vec<String>,
}

/// Story archiver for long-term storage
pub struct StoryArchiver {
    story_engine: Arc<StoryEngine>,
    archive_dir: PathBuf,
}

impl StoryArchiver {
    /// Create a new archiver
    pub fn new(story_engine: Arc<StoryEngine>, archive_dir: PathBuf) -> Self {
        Self {
            story_engine,
            archive_dir,
        }
    }

    /// Archive completed stories
    pub async fn archive_completed_stories(&self) -> Result<usize> {
        let mut archived_count = 0;

        // Create archive directory
        fs::create_dir_all(&self.archive_dir)?;

        // Find completed stories
        let completed_stories: Vec<_> = self.story_engine.stories
            .iter()
            .filter(|s| {
                let story = s.value();
                story.arcs.iter()
                    .flat_map(|arc| &arc.plot_points)
                    .all(|p| match &p.plot_type {
                        PlotType::Task { completed, .. } => *completed,
                        PlotType::Issue { resolved, .. } => *resolved,
                        _ => true,
                    })
            })
            .map(|s| s.value().clone())
            .collect();

        // Archive each completed story
        for story in completed_stories {
            let filename = format!("story_{}__{}.json",
                story.created_at.format("%Y%m%d_%H%M%S"),
                story.id.0
            );
            let path = self.archive_dir.join(&filename);

            // Export single story
            let porter = StoryPorter::new(self.story_engine.clone());
            let export_data = porter.export_selected_stories(
                vec![story.id],
                ExportOptions {
                    include_completed: true,
                    include_context_chains: true,
                    compress: true,
                    format: ExportFormat::Json,
                },
            ).await?;

            // Write to archive
            porter.export_to_file(&path, ExportOptions {
                compress: true,
                format: ExportFormat::Json,
                ..Default::default()
            }).await?;

            // Remove from active stories
            self.story_engine.stories.remove(&story.id);
            archived_count += 1;

            debug!("Archived story {} to {}", story.id.0, filename);
        }

        info!("Archived {} completed stories", archived_count);
        Ok(archived_count)
    }

    /// List archived stories
    pub fn list_archives(&self) -> Result<Vec<ArchiveInfo>> {
        let mut archives = Vec::new();

        for entry in fs::read_dir(&self.archive_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                let metadata = entry.metadata()?;
                let filename = path.file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown")
                    .to_string();

                archives.push(ArchiveInfo {
                    filename,
                    path,
                    size: metadata.len(),
                    created: metadata.created()?,
                });
            }
        }

        archives.sort_by(|a, b| b.created.cmp(&a.created));
        Ok(archives)
    }
}

/// Archive information
#[derive(Debug, Clone)]
pub struct ArchiveInfo {
    pub filename: String,
    pub path: PathBuf,
    pub size: u64,
    pub created: std::time::SystemTime,
}
