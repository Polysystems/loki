//! Cognitive State Persistence for Chat Sessions
//!
//! This module handles saving and loading cognitive state across sessions,
//! allowing Loki to remember context, insights, and learning between conversations.

use std::sync::Arc;
use std::path::{PathBuf};
use anyhow::{Result, Context};
use serde::{Serialize, Deserialize};
use tokio::fs;
use tracing::{info, debug, warn};
use chrono::{DateTime, Utc};

use crate::cognitive::{
    CognitiveSystem,
};
use crate::memory::{CognitiveMemory, MemoryMetadata};
use crate::tui::{
    cognitive_stream_integration::{CognitiveActivity},
};

/// Persisted cognitive state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedCognitiveState {
    /// Session ID
    pub session_id: String,
    
    /// Timestamp of last save
    pub last_saved: DateTime<Utc>,
    
    /// Consciousness state
    pub consciousness: PersistedConsciousnessState,
    
    /// Recent insights
    pub insights: Vec<PersistedInsight>,
    
    /// Active modalities and their usage
    pub modality_usage: Vec<ModalityUsage>,
    
    /// Session context
    pub context: SessionContext,
    
    /// Learning outcomes
    pub learnings: Vec<LearningOutcome>,
    
    /// Conversation summary
    pub summary: ConversationSummary,
}

/// Persisted consciousness state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedConsciousnessState {
    pub awareness_level: f64,
    pub gradient_coherence: f64,
    pub free_energy: f64,
    pub current_focus: String,
    pub consciousness_mode: String,
    pub narrative: String,
}

/// Persisted insight
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedInsight {
    pub timestamp: DateTime<Utc>,
    pub content: String,
    pub category: String,
    pub relevance: f64,
    pub source_modalities: Vec<String>,
}

/// Modality usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModalityUsage {
    pub modality: String,
    pub activation_count: u32,
    pub total_duration_ms: u64,
    pub average_confidence: f64,
}

/// Session context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionContext {
    pub user_preferences: serde_json::Value,
    pub discussion_topics: Vec<String>,
    pub emotional_tone: String,
    pub interaction_style: String,
}

/// Learning outcome
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningOutcome {
    pub timestamp: DateTime<Utc>,
    pub learning_type: LearningType,
    pub content: String,
    pub confidence: f64,
    pub applied: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LearningType {
    ConceptualUnderstanding,
    PatternRecognition,
    UserPreference,
    ProblemSolvingStrategy,
    EmotionalIntelligence,
}

/// Conversation summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationSummary {
    pub message_count: usize,
    pub duration_seconds: u64,
    pub key_topics: Vec<String>,
    pub user_satisfaction: Option<f64>,
    pub cognitive_depth_achieved: f64,
}

/// Cognitive state manager for persistence
pub struct CognitiveStateManager {
    /// Base directory for state files
    state_dir: PathBuf,
    
    /// Reference to cognitive system
    cognitive_system: Arc<CognitiveSystem>,
    
    /// Reference to memory
    memory: Arc<CognitiveMemory>,
    
    /// Auto-save interval in seconds
    auto_save_interval: u64,
}

impl CognitiveStateManager {
    /// Create new state manager
    pub fn new(
        state_dir: PathBuf,
        cognitive_system: Arc<CognitiveSystem>,
        memory: Arc<CognitiveMemory>,
    ) -> Result<Self> {
        // Ensure state directory exists
        std::fs::create_dir_all(&state_dir)
            .context("Failed to create cognitive state directory")?;
        
        Ok(Self {
            state_dir,
            cognitive_system,
            memory,
            auto_save_interval: 300, // 5 minutes default
        })
    }
    
    /// Save current cognitive state
    pub async fn save_state(
        &self,
        session_id: &str,
        activity: &CognitiveActivity,
        insights: Vec<PersistedInsight>,
        modality_usage: Vec<ModalityUsage>,
        context: SessionContext,
        learnings: Vec<LearningOutcome>,
        summary: ConversationSummary,
    ) -> Result<()> {
        info!("💾 Saving cognitive state for session: {}", session_id);
        
        let state = PersistedCognitiveState {
            session_id: session_id.to_string(),
            last_saved: Utc::now(),
            consciousness: PersistedConsciousnessState {
                awareness_level: activity.awareness_level,
                gradient_coherence: activity.gradient_coherence,
                free_energy: activity.free_energy,
                current_focus: activity.current_focus.clone(),
                consciousness_mode: "Standard".to_string(), // Default mode
                narrative: String::new(), // Narrative would be retrieved from consciousness stream
            },
            insights,
            modality_usage,
            context,
            learnings,
            summary,
        };
        
        // Serialize to JSON
        let json = serde_json::to_string_pretty(&state)?;
        
        // Write to file
        let file_path = self.get_state_file_path(session_id);
        fs::write(&file_path, json).await
            .context("Failed to write cognitive state file")?;
        
        debug!("Cognitive state saved to: {:?}", file_path);
        
        // Also save to long-term memory
        self.save_to_memory(&state).await?;
        
        Ok(())
    }
    
    /// Load cognitive state
    pub async fn load_state(&self, session_id: &str) -> Result<Option<PersistedCognitiveState>> {
        let file_path = self.get_state_file_path(session_id);
        
        if !file_path.exists() {
            debug!("No saved state found for session: {}", session_id);
            return Ok(None);
        }
        
        info!("📂 Loading cognitive state for session: {}", session_id);
        
        let json = fs::read_to_string(&file_path).await
            .context("Failed to read cognitive state file")?;
        
        let state: PersistedCognitiveState = serde_json::from_str(&json)
            .context("Failed to deserialize cognitive state")?;
        
        Ok(Some(state))
    }
    
    /// List all saved sessions
    pub async fn list_sessions(&self) -> Result<Vec<SessionInfo>> {
        let mut sessions = Vec::new();
        
        let mut entries = fs::read_dir(&self.state_dir).await?;
        while let Some(entry) = entries.next_entry().await? {
            if let Some(file_name) = entry.file_name().to_str() {
                if file_name.ends_with(".cognitive.json") {
                    let session_id = file_name.replace(".cognitive.json", "");
                    
                    // Try to load basic info
                    if let Ok(Some(state)) = self.load_state(&session_id).await {
                        sessions.push(SessionInfo {
                            session_id: state.session_id,
                            last_saved: state.last_saved,
                            message_count: state.summary.message_count,
                            key_topics: state.summary.key_topics,
                        });
                    }
                }
            }
        }
        
        // Sort by last saved (most recent first)
        sessions.sort_by(|a, b| b.last_saved.cmp(&a.last_saved));
        
        Ok(sessions)
    }
    
    /// Delete old sessions
    pub async fn cleanup_old_sessions(&self, days_to_keep: i64) -> Result<usize> {
        let cutoff = Utc::now() - chrono::Duration::days(days_to_keep);
        let mut deleted = 0;
        
        let sessions = self.list_sessions().await?;
        for session in sessions {
            if session.last_saved < cutoff {
                if let Ok(_) = self.delete_session(&session.session_id).await {
                    deleted += 1;
                }
            }
        }
        
        info!("🗑️ Cleaned up {} old cognitive state files", deleted);
        Ok(deleted)
    }
    
    /// Delete a specific session
    pub async fn delete_session(&self, session_id: &str) -> Result<()> {
        let file_path = self.get_state_file_path(session_id);
        if file_path.exists() {
            fs::remove_file(&file_path).await?;
        }
        Ok(())
    }
    
    /// Get state file path
    fn get_state_file_path(&self, session_id: &str) -> PathBuf {
        self.state_dir.join(format!("{}.cognitive.json", session_id))
    }
    
    /// Save important learnings to long-term memory
    async fn save_to_memory(&self, state: &PersistedCognitiveState) -> Result<()> {
        // Save key insights
        for insight in &state.insights {
            if insight.relevance > 0.8 {
                let memory_item = serde_json::json!({
                    "type": "cognitive_insight",
                    "session_id": state.session_id,
                    "timestamp": insight.timestamp,
                    "content": insight.content,
                    "category": insight.category,
                    "relevance": insight.relevance,
                });
                
                self.memory.store(
                    memory_item.to_string(),
                    vec!["cognitive".to_string(), "insight".to_string(), insight.category.clone()],
                    MemoryMetadata {
                        source: "cognitive_state".to_string(),
                        tags: vec!["insight".to_string()],
                        timestamp: insight.timestamp,
                        expiration: None,
                        importance: insight.relevance as f32,
                        associations: vec![],
                        context: Some(state.session_id.clone()),
                        created_at: chrono::Utc::now(),
                        accessed_count: 0,
                        last_accessed: None,
                        version: 1,
                        category: "cognitive_insight".to_string(),
                    },
                ).await?;
            }
        }
        
        // Save important learnings
        for learning in &state.learnings {
            if learning.confidence > 0.7 {
                let memory_item = serde_json::json!({
                    "type": "learning_outcome",
                    "session_id": state.session_id,
                    "timestamp": learning.timestamp,
                    "learning_type": learning.learning_type,
                    "content": learning.content,
                    "confidence": learning.confidence,
                });
                
                self.memory.store(
                    memory_item.to_string(),
                    vec!["cognitive".to_string(), "learning".to_string()],
                    MemoryMetadata {
                        source: "cognitive_state".to_string(),
                        tags: vec!["learning".to_string()],
                        timestamp: learning.timestamp,
                        expiration: None,
                        importance: learning.confidence as f32,
                        associations: vec![],
                        context: Some(state.session_id.clone()),
                        created_at: chrono::Utc::now(),
                        accessed_count: 0,
                        last_accessed: None,
                        version: 1,
                        category: "learning_outcome".to_string(),
                    },
                ).await?;
            }
        }
        
        Ok(())
    }
}

/// Session info for listing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionInfo {
    pub session_id: String,
    pub last_saved: DateTime<Utc>,
    pub message_count: usize,
    pub key_topics: Vec<String>,
}

/// State restoration helper
pub struct CognitiveStateRestorer;

impl CognitiveStateRestorer {
    /// Restore cognitive state into active systems
    pub async fn restore_state(
        state: &PersistedCognitiveState,
        cognitive_system: &CognitiveSystem,
        memory: &CognitiveMemory,
    ) -> Result<()> {
        info!("🔄 Restoring cognitive state from session: {}", state.session_id);
        
        // Restore consciousness parameters
        // This would need methods on CognitiveSystem to set internal state
        
        // Restore recent insights to working memory
        for insight in &state.insights {
            let insight_data = serde_json::json!({
                "content": insight.content,
                "category": insight.category,
                "relevance": insight.relevance,
                "timestamp": insight.timestamp,
            });
            
            memory.store(
                insight_data.to_string(),
                vec!["restored".to_string(), "insight".to_string()],
                MemoryMetadata {
                    source: "cognitive_restoration".to_string(),
                    tags: vec!["restored".to_string(), "insight".to_string()],
                    timestamp: insight.timestamp,
                    expiration: None,
                    importance: insight.relevance as f32,
                    associations: vec![],
                    context: Some(state.session_id.clone()),
                    created_at: chrono::Utc::now(),
                    accessed_count: 0,
                    last_accessed: None,
                    version: 1,
                    category: insight.category.clone(),
                },
            ).await?;
        }
        
        // Restore learnings
        for learning in &state.learnings {
            if learning.applied {
                debug!("Restoring applied learning: {}", learning.content);
                // Apply learning to cognitive system
                // This would need specific methods on CognitiveSystem
            }
        }
        
        info!("✅ Cognitive state restoration complete");
        Ok(())
    }
    
    /// Create a restoration summary
    pub fn create_restoration_summary(state: &PersistedCognitiveState) -> String {
        format!(
            "🧠 Restored Session: {}\n\
            📅 Last Active: {}\n\
            💬 Messages: {}\n\
            💡 Insights: {}\n\
            📚 Learnings: {}\n\
            🎯 Topics: {}\n\
            🌊 Awareness: {:.0}%\n\
            🔗 Coherence: {:.0}%",
            state.session_id,
            state.last_saved.format("%Y-%m-%d %H:%M UTC"),
            state.summary.message_count,
            state.insights.len(),
            state.learnings.len(),
            state.summary.key_topics.join(", "),
            state.consciousness.awareness_level * 100.0,
            state.consciousness.gradient_coherence * 100.0,
        )
    }
}

/// Auto-save manager for periodic state saves
pub struct CognitiveAutoSave {
    state_manager: Arc<CognitiveStateManager>,
    active: Arc<tokio::sync::RwLock<bool>>,
}

impl CognitiveAutoSave {
    pub fn new(state_manager: Arc<CognitiveStateManager>) -> Self {
        Self {
            state_manager,
            active: Arc::new(tokio::sync::RwLock::new(false)),
        }
    }
    
    /// Start auto-save task
    pub async fn start(
        &self,
        session_id: String,
        get_state: impl Fn() -> CognitiveStateSnapshot + Send + Sync + 'static,
    ) -> tokio::task::JoinHandle<()> {
        let state_manager = self.state_manager.clone();
        let active = self.active.clone();
        let interval = self.state_manager.auto_save_interval;
        
        *active.write().await = true;
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(interval));
            
            while *active.read().await {
                interval.tick().await;
                
                let snapshot = get_state();
                
                if let Err(e) = state_manager.save_state(
                    &session_id,
                    &snapshot.activity,
                    snapshot.insights,
                    snapshot.modality_usage,
                    snapshot.context,
                    snapshot.learnings,
                    snapshot.summary,
                ).await {
                    warn!("Auto-save failed: {}", e);
                }
            }
        })
    }
    
    /// Stop auto-save
    pub async fn stop(&self) {
        *self.active.write().await = false;
    }
}

/// Snapshot of current cognitive state
pub struct CognitiveStateSnapshot {
    pub activity: CognitiveActivity,
    pub insights: Vec<PersistedInsight>,
    pub modality_usage: Vec<ModalityUsage>,
    pub context: SessionContext,
    pub learnings: Vec<LearningOutcome>,
    pub summary: ConversationSummary,
}