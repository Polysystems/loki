//! Story synchronization for maintaining consistency across agents and components

use super::types::*;
use crate::story::engine::StoryEvent;
use anyhow::Result;
use std::collections::HashMap;
use tokio::sync::broadcast;
use tracing::{debug, info};
use uuid::Uuid;

/// Synchronizes stories across different parts of the system
#[derive(Debug)]
pub struct StorySynchronizer {
    event_tx: broadcast::Sender<StoryEvent>,
}

impl StorySynchronizer {
    /// Create a new synchronizer
    pub fn new(event_tx: broadcast::Sender<StoryEvent>) -> Self {
        Self { event_tx }
    }
    
    /// Synchronize a single story
    pub async fn sync_story(&self, story: &Story) -> Result<SyncEvent> {
        let sync_event = SyncEvent {
            id: Uuid::new_v4(),
            timestamp: chrono::Utc::now(),
            source_story: story.id,
            target_stories: vec![],
            sync_type: SyncType::Full,
            payload: self.create_sync_payload(story),
        };
        
        debug!("Synchronized story {}", story.id.0);
        
        Ok(sync_event)
    }
    
    /// Synchronize multiple stories
    pub async fn sync_multiple(&self, stories: Vec<Story>) -> Result<SyncEvent> {
        if stories.is_empty() {
            return Err(anyhow::anyhow!("No stories to sync"));
        }
        
        let source = stories[0].id;
        let targets: Vec<StoryId> = stories.iter().skip(1).map(|s| s.id).collect();
        
        // Merge all story data
        let mut merged_plot_points = Vec::new();
        let mut merged_context = HashMap::new();
        let mut merged_metadata = HashMap::new();
        
        for story in &stories {
            // Collect all plot points
            for arc in &story.arcs {
                merged_plot_points.extend(arc.plot_points.clone());
            }
            
            // Merge context updates
            merged_context.insert(
                story.id.0.to_string(),
                story.summary.clone(),
            );
            
            // Merge metadata
            for (key, value) in &story.metadata.custom_data {
                merged_metadata.insert(
                    format!("{}:{}", story.id.0, key),
                    value.clone(),
                );
            }
        }
        
        // Sort plot points by timestamp
        merged_plot_points.sort_by_key(|p| p.timestamp);
        
        let sync_event = SyncEvent {
            id: Uuid::new_v4(),
            timestamp: chrono::Utc::now(),
            source_story: source,
            target_stories: targets,
            sync_type: SyncType::Merge,
            payload: SyncPayload {
                plot_points: merged_plot_points,
                context_updates: merged_context,
                metadata_changes: merged_metadata,
            },
        };
        
        info!(
            "Synchronized {} stories with {} plot points",
            stories.len(),
            sync_event.payload.plot_points.len()
        );
        
        Ok(sync_event)
    }
    
    /// Create incremental sync between stories
    pub async fn sync_delta(
        &self,
        source: &Story,
        target: &Story,
        since: chrono::DateTime<chrono::Utc>,
    ) -> Result<SyncEvent> {
        let mut delta_plot_points = Vec::new();
        
        // Find plot points added since timestamp
        for arc in &source.arcs {
            for plot_point in &arc.plot_points {
                if plot_point.timestamp > since {
                    delta_plot_points.push(plot_point.clone());
                }
            }
        }
        
        let sync_event = SyncEvent {
            id: Uuid::new_v4(),
            timestamp: chrono::Utc::now(),
            source_story: source.id,
            target_stories: vec![target.id],
            sync_type: SyncType::Delta,
            payload: SyncPayload {
                plot_points: delta_plot_points,
                context_updates: HashMap::new(),
                metadata_changes: HashMap::new(),
            },
        };
        
        Ok(sync_event)
    }
    
    /// Broadcast sync to all related stories
    pub async fn broadcast_sync(
        &self,
        source: &Story,
        message: String,
        plot_type: PlotType,
    ) -> Result<SyncEvent> {
        let plot_point = PlotPoint {
            id: PlotPointId(Uuid::new_v4()),
            title: String::from("Sync Event"),
            description: message.clone(),
            sequence_number: 0,
            timestamp: chrono::Utc::now(),
            plot_type,
            status: crate::story::PlotPointStatus::Pending,
            estimated_duration: None,
            actual_duration: None,
            context_tokens: vec![],
            importance: 0.7,
            metadata: PlotMetadata::default(),
            tags: vec![],
            consequences: vec![],
        };
        
        let mut context_updates = HashMap::new();
        context_updates.insert("broadcast_message".to_string(), message);
        
        let sync_event = SyncEvent {
            id: Uuid::new_v4(),
            timestamp: chrono::Utc::now(),
            source_story: source.id,
            target_stories: source.metadata.related_stories.clone(),
            sync_type: SyncType::Broadcast,
            payload: SyncPayload {
                plot_points: vec![plot_point],
                context_updates,
                metadata_changes: HashMap::new(),
            },
        };
        
        Ok(sync_event)
    }
    
    /// Apply a sync event to a story
    pub async fn apply_sync(&self, story: &mut Story, sync_event: &SyncEvent) -> Result<()> {
        // Add plot points to current arc
        if let Some(arc_id) = story.current_arc {
            if let Some(arc) = story.arcs.iter_mut().find(|a| a.id == arc_id) {
                for plot_point in &sync_event.payload.plot_points {
                    // Check if plot point already exists
                    if !arc.plot_points.iter().any(|p| p.id == plot_point.id) {
                        arc.plot_points.push(plot_point.clone());
                    }
                }
            }
        }
        
        // Apply metadata changes
        for (key, value) in &sync_event.payload.metadata_changes {
            story.metadata.custom_data.insert(key.clone(), value.clone());
        }
        
        story.updated_at = chrono::Utc::now();
        
        Ok(())
    }
    
    /// Create sync payload from a story
    fn create_sync_payload(&self, story: &Story) -> SyncPayload {
        let mut plot_points = Vec::new();
        
        for arc in &story.arcs {
            plot_points.extend(arc.plot_points.clone());
        }
        
        let mut context_updates = HashMap::new();
        context_updates.insert("story_summary".to_string(), story.summary.clone());
        context_updates.insert("story_title".to_string(), story.title.clone());
        
        SyncPayload {
            plot_points,
            context_updates,
            metadata_changes: story.metadata.custom_data.clone(),
        }
    }
}

/// Conflict resolution strategies
#[derive(Debug, Clone, PartialEq)]
pub enum ConflictResolution {
    /// Use the most recent version
    MostRecent,
    /// Merge both versions
    Merge,
    /// Use the version with higher importance
    HighestImportance,
    /// Manual resolution required
    Manual,
}

/// Resolve conflicts between plot points
pub fn resolve_plot_point_conflict(
    a: &PlotPoint,
    b: &PlotPoint,
    strategy: ConflictResolution,
) -> PlotPoint {
    match strategy {
        ConflictResolution::MostRecent => {
            if a.timestamp > b.timestamp {
                a.clone()
            } else {
                b.clone()
            }
        }
        ConflictResolution::HighestImportance => {
            if a.importance > b.importance {
                a.clone()
            } else {
                b.clone()
            }
        }
        ConflictResolution::Merge => {
            // Create merged plot point
            PlotPoint {
                id: PlotPointId(Uuid::new_v4()),
                title: format!("Merged: {} / {}", a.title, b.title),
                description: format!("{} | {}", a.description, b.description),
                sequence_number: a.sequence_number.max(b.sequence_number),
                timestamp: chrono::Utc::now(),
                plot_type: a.plot_type.clone(), // Use first plot type
                status: crate::story::PlotPointStatus::Pending,
                estimated_duration: a.estimated_duration.or(b.estimated_duration),
                actual_duration: a.actual_duration.or(b.actual_duration),
                context_tokens: [a.context_tokens.clone(), b.context_tokens.clone()].concat(),
                importance: (a.importance + b.importance) / 2.0,
                metadata: a.metadata.clone(), // Use first metadata
                tags: [a.tags.clone(), b.tags.clone()].concat(),
                consequences: [a.consequences.clone(), b.consequences.clone()].concat(),
            }
        }
        ConflictResolution::Manual => {
            // Return the first one, but this should trigger manual review
            a.clone()
        }
    }
}