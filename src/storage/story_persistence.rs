//! Story persistence for TUI system
//!
//! This module provides persistent storage for stories, story arcs,
//! and story execution history.

use anyhow::{Result, Context};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::{Pool, Sqlite, SqlitePool};
use std::collections::HashMap;
use tracing::{debug, info};

use crate::story::{Story, StoryArc, PlotPoint, StoryType, StoryStatus};

/// Story persistence storage
pub struct StoryPersistence {
    pool: Pool<Sqlite>,
    cache: Vec<StorySummary>,
}

/// Story summary for quick access
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorySummary {
    pub id: String,
    pub title: String,
    pub story_type: String,
    pub status: String,
    pub created_at: DateTime<Utc>,
    pub last_updated: DateTime<Utc>,
    pub completion_percentage: f32,
    pub arc_count: i32,
    pub plot_point_count: i32,
}

/// Story execution record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoryExecution {
    pub id: String,
    pub story_id: String,
    pub arc_id: Option<String>,
    pub plot_point_id: Option<String>,
    pub action: String,
    pub result: String,
    pub success: bool,
    pub duration_ms: i64,
    pub timestamp: DateTime<Utc>,
    pub metadata: Option<serde_json::Value>,
}

impl StoryPersistence {
    /// Create a new story persistence instance
    pub async fn new(database_url: Option<String>) -> Result<Self> {
        let db_url = database_url.unwrap_or_else(|| {
            let home = dirs::home_dir().unwrap();
            format!("sqlite://{}/loki/stories.db", home.display())
        });

        // Ensure directory exists
        if let Some(path) = db_url.strip_prefix("sqlite://") {
            if let Some(parent) = std::path::Path::new(path).parent() {
                std::fs::create_dir_all(parent)?;
            }
        }

        let pool = SqlitePool::connect(&db_url).await?;
        
        // Initialize schema
        Self::initialize_schema(&pool).await?;
        
        let mut instance = Self {
            pool,
            cache: Vec::new(),
        };
        
        // Load initial cache
        instance.refresh_cache().await?;
        
        Ok(instance)
    }
    
    /// Initialize database schema
    async fn initialize_schema(pool: &Pool<Sqlite>) -> Result<()> {
        // Stories table
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS stories (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT,
                story_type TEXT NOT NULL,
                status TEXT NOT NULL,
                context TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completion_percentage REAL DEFAULT 0,
                metadata TEXT
            )
            "#,
        )
        .execute(pool)
        .await?;
        
        // Story arcs table
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS story_arcs (
                id TEXT PRIMARY KEY,
                story_id TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT,
                arc_type TEXT NOT NULL,
                sequence_number INTEGER NOT NULL,
                status TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT,
                FOREIGN KEY (story_id) REFERENCES stories(id) ON DELETE CASCADE
            )
            "#,
        )
        .execute(pool)
        .await?;
        
        // Plot points table
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS plot_points (
                id TEXT PRIMARY KEY,
                arc_id TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT,
                plot_type TEXT NOT NULL,
                sequence_number INTEGER NOT NULL,
                status TEXT NOT NULL,
                estimated_duration_minutes INTEGER,
                actual_duration_minutes INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                metadata TEXT,
                FOREIGN KEY (arc_id) REFERENCES story_arcs(id) ON DELETE CASCADE
            )
            "#,
        )
        .execute(pool)
        .await?;
        
        // Story executions table
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS story_executions (
                id TEXT PRIMARY KEY,
                story_id TEXT NOT NULL,
                arc_id TEXT,
                plot_point_id TEXT,
                action TEXT NOT NULL,
                result TEXT NOT NULL,
                success BOOLEAN NOT NULL,
                duration_ms INTEGER NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT,
                FOREIGN KEY (story_id) REFERENCES stories(id) ON DELETE CASCADE
            )
            "#,
        )
        .execute(pool)
        .await?;
        
        // Create indexes
        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_stories_status ON stories(status)"
        )
        .execute(pool)
        .await?;
        
        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_story_arcs_story_id ON story_arcs(story_id)"
        )
        .execute(pool)
        .await?;
        
        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_plot_points_arc_id ON plot_points(arc_id)"
        )
        .execute(pool)
        .await?;
        
        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_executions_story_id ON story_executions(story_id)"
        )
        .execute(pool)
        .await?;
        
        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_executions_timestamp ON story_executions(timestamp)"
        )
        .execute(pool)
        .await?;
        
        info!("Story persistence schema initialized");
        Ok(())
    }
    
    /// Save a story to persistent storage
    pub async fn save_story(&mut self, story: &Story) -> Result<()> {
        // Convert story to JSON for metadata
        let metadata = serde_json::to_string(&story.context)?;
        
        // Insert or update story
        sqlx::query(
            r#"
            INSERT INTO stories (id, title, description, story_type, status, context, metadata, completion_percentage)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
            ON CONFLICT(id) DO UPDATE SET
                title = excluded.title,
                description = excluded.description,
                status = excluded.status,
                context = excluded.context,
                metadata = excluded.metadata,
                completion_percentage = excluded.completion_percentage,
                last_updated = CURRENT_TIMESTAMP
            "#,
        )
        .bind(&story.id.0.to_string())
        .bind(&story.title)
        .bind(&story.description)
        .bind(format!("{:?}", story.story_type))
        .bind(format!("{:?}", story.status))
        .bind(&metadata)
        .bind(&metadata)
        .bind(story.calculate_completion_percentage())
        .execute(&self.pool)
        .await?;
        
        // Save arcs
        for arc in &story.arcs {
            self.save_arc(&story.id.0.to_string(), arc).await?;
        }
        
        // Refresh cache
        self.refresh_cache().await?;
        
        info!("Saved story: {}", story.title);
        Ok(())
    }
    
    /// Save a story arc
    async fn save_arc(&self, story_id: &str, arc: &StoryArc) -> Result<()> {
        let metadata = serde_json::to_string(&arc)?;
        
        sqlx::query(
            r#"
            INSERT INTO story_arcs (id, story_id, title, description, arc_type, sequence_number, status, metadata)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
            ON CONFLICT(id) DO UPDATE SET
                title = excluded.title,
                description = excluded.description,
                status = excluded.status,
                metadata = excluded.metadata
            "#,
        )
        .bind(&arc.id.0.to_string())
        .bind(story_id)
        .bind(&arc.title)
        .bind(&arc.description)
        .bind("standard") // Arc type as string
        .bind(arc.sequence_number as i64)
        .bind(format!("{:?}", arc.status))
        .bind(&metadata)
        .execute(&self.pool)
        .await?;
        
        // Save plot points
        for plot_point in &arc.plot_points {
            self.save_plot_point(&arc.id.0.to_string(), plot_point).await?;
        }
        
        Ok(())
    }
    
    /// Save a plot point
    async fn save_plot_point(&self, arc_id: &str, plot_point: &PlotPoint) -> Result<()> {
        let metadata = serde_json::to_string(&plot_point)?;
        
        sqlx::query(
            r#"
            INSERT INTO plot_points (
                id, arc_id, title, description, plot_type, 
                sequence_number, status, estimated_duration_minutes, 
                actual_duration_minutes, metadata
            )
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
            ON CONFLICT(id) DO UPDATE SET
                title = excluded.title,
                description = excluded.description,
                status = excluded.status,
                actual_duration_minutes = excluded.actual_duration_minutes,
                metadata = excluded.metadata
            "#,
        )
        .bind(&plot_point.id.0.to_string())
        .bind(arc_id)
        .bind(&plot_point.title)
        .bind(&plot_point.description)
        .bind(format!("{:?}", plot_point.plot_type))
        .bind(plot_point.sequence_number as i64)
        .bind(format!("{:?}", plot_point.status))
        .bind(plot_point.estimated_duration.as_ref().map(|d| d.num_minutes() as i64))
        .bind(plot_point.actual_duration.as_ref().map(|d| d.num_minutes() as i64))
        .bind(&metadata)
        .execute(&self.pool)
        .await?;
        
        Ok(())
    }
    
    /// Load a story from storage
    pub async fn load_story(&self, story_id: &str) -> Result<Option<Story>> {
        // Load story data
        let story_row = sqlx::query_as::<_, (String, String, String, String, String, Option<String>, String)>(
            "SELECT id, title, description, story_type, status, context, metadata FROM stories WHERE id = ?1"
        )
        .bind(story_id)
        .fetch_optional(&self.pool)
        .await?;
        
        if let Some((id, title, description, _story_type, _status, _context, metadata)) = story_row {
            // Parse metadata to reconstruct story
            let _metadata: serde_json::Value = serde_json::from_str(&metadata)?;
            
            // For now, return a basic story structure
            // In production, you'd fully reconstruct the story with arcs and plot points
            info!("Loaded story: {} ({})", title, id);
            
            // Note: Full reconstruction would require loading arcs and plot points
            // This is a simplified version for demonstration
            let story_id = uuid::Uuid::parse_str(&id).unwrap_or_else(|_| uuid::Uuid::new_v4());
            Ok(Some(Story {
                id: crate::story::StoryId(story_id),
                title,
                description,
                story_type: StoryType::Feature {
                    feature_name: String::from("Unknown"),
                    description: String::from("Loaded from storage"),
                },
                status: StoryStatus::NotStarted,
                summary: String::new(),
                arcs: Vec::new(),
                current_arc: None,
                metadata: crate::story::StoryMetadata::default(),
                context_chain: crate::story::ChainId::new(),
                segments: Vec::new(),
                context: HashMap::new(),
                created_at: Utc::now(),
                updated_at: Utc::now(),
            }))
        } else {
            Ok(None)
        }
    }
    
    /// List all stories
    pub async fn list_stories(&self, limit: i64) -> Result<Vec<StorySummary>> {
        let stories = sqlx::query_as::<_, (String, String, String, String, DateTime<Utc>, DateTime<Utc>, f32)>(
            r#"
            SELECT 
                s.id,
                s.title,
                s.story_type,
                s.status,
                s.created_at,
                s.last_updated,
                s.completion_percentage
            FROM stories s
            ORDER BY s.last_updated DESC
            LIMIT ?1
            "#,
        )
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;
        
        let mut summaries = Vec::new();
        for (id, title, story_type, status, created_at, last_updated, completion) in stories {
            // Count arcs and plot points
            let arc_count = sqlx::query_scalar::<_, i32>(
                "SELECT COUNT(*) FROM story_arcs WHERE story_id = ?1"
            )
            .bind(&id)
            .fetch_one(&self.pool)
            .await?;
            
            let plot_count = sqlx::query_scalar::<_, i32>(
                r#"
                SELECT COUNT(*) 
                FROM plot_points p
                JOIN story_arcs a ON p.arc_id = a.id
                WHERE a.story_id = ?1
                "#
            )
            .bind(&id)
            .fetch_one(&self.pool)
            .await?;
            
            summaries.push(StorySummary {
                id,
                title,
                story_type,
                status,
                created_at,
                last_updated,
                completion_percentage: completion,
                arc_count,
                plot_point_count: plot_count,
            });
        }
        
        Ok(summaries)
    }
    
    /// Record a story execution
    pub async fn record_execution(&self, execution: StoryExecution) -> Result<()> {
        let metadata = execution.metadata.as_ref()
            .map(|m| serde_json::to_string(m))
            .transpose()?;
        
        sqlx::query(
            r#"
            INSERT INTO story_executions (
                id, story_id, arc_id, plot_point_id, action, 
                result, success, duration_ms, metadata
            )
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
            "#,
        )
        .bind(&execution.id)
        .bind(&execution.story_id)
        .bind(&execution.arc_id)
        .bind(&execution.plot_point_id)
        .bind(&execution.action)
        .bind(&execution.result)
        .bind(execution.success)
        .bind(execution.duration_ms)
        .bind(metadata)
        .execute(&self.pool)
        .await?;
        
        debug!("Recorded execution for story {}: {}", execution.story_id, execution.action);
        Ok(())
    }
    
    /// Get execution history for a story
    pub async fn get_execution_history(&self, story_id: &str, limit: i64) -> Result<Vec<StoryExecution>> {
        let executions = sqlx::query_as::<_, (String, String, Option<String>, Option<String>, String, String, bool, i64, DateTime<Utc>, Option<String>)>(
            r#"
            SELECT 
                id, story_id, arc_id, plot_point_id, action,
                result, success, duration_ms, timestamp, metadata
            FROM story_executions
            WHERE story_id = ?1
            ORDER BY timestamp DESC
            LIMIT ?2
            "#,
        )
        .bind(story_id)
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;
        
        let mut history = Vec::new();
        for (id, story_id, arc_id, plot_point_id, action, result, success, duration_ms, timestamp, metadata) in executions {
            let metadata = metadata
                .map(|m| serde_json::from_str(&m))
                .transpose()?;
            
            history.push(StoryExecution {
                id,
                story_id,
                arc_id,
                plot_point_id,
                action,
                result,
                success,
                duration_ms,
                timestamp,
                metadata,
            });
        }
        
        Ok(history)
    }
    
    /// Delete a story and all related data
    pub async fn delete_story(&mut self, story_id: &str) -> Result<()> {
        sqlx::query("DELETE FROM stories WHERE id = ?1")
            .bind(story_id)
            .execute(&self.pool)
            .await?;
        
        self.refresh_cache().await?;
        info!("Deleted story: {}", story_id);
        Ok(())
    }
    
    /// Refresh the cache
    async fn refresh_cache(&mut self) -> Result<()> {
        self.cache = self.list_stories(100).await?;
        Ok(())
    }
    
    /// Get cached summaries
    pub fn get_cached_summaries(&self) -> &[StorySummary] {
        &self.cache
    }
    
    /// Export stories to JSON
    pub async fn export_stories(&self) -> Result<serde_json::Value> {
        let stories = self.list_stories(1000).await?;
        Ok(serde_json::to_value(stories)?)
    }
    
    /// Import stories from JSON
    pub async fn import_stories(&mut self, data: serde_json::Value) -> Result<()> {
        if let Ok(summaries) = serde_json::from_value::<Vec<StorySummary>>(data) {
            for summary in summaries {
                info!("Importing story: {}", summary.title);
                // Note: Full import would require reconstructing complete Story objects
                // This is simplified for demonstration
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_story_persistence() {
        let persistence = StoryPersistence::new(Some("sqlite::memory:".to_string()))
            .await
            .unwrap();
        
        // Test basic operations
        let summaries = persistence.list_stories(10).await.unwrap();
        assert_eq!(summaries.len(), 0);
    }
}