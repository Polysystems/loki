//! Cognitive Session Manager for Chat
//!
//! Manages cognitive sessions with persistence, restoration, and continuity
//! across chat interactions.

use std::sync::Arc;
use std::collections::HashMap;
use anyhow::{Result};
use tokio::sync::RwLock;
use tracing::info;
use chrono::{DateTime, Utc};

use crate::tui::{
    cognitive::persistence::state::{
        CognitiveStateManager, PersistedCognitiveState, SessionInfo,
        CognitiveStateRestorer, CognitiveAutoSave, CognitiveStateSnapshot,
        PersistedInsight, ModalityUsage, SessionContext, LearningOutcome,
        ConversationSummary, LearningType,
    },
    chat::integrations::cognitive::{CognitiveChatEnhancement, CognitiveResponse},
    cognitive_stream_integration::CognitiveActivity,
};
use crate::cognitive::CognitiveSystem;
use crate::memory::CognitiveMemory;

/// Active cognitive session
pub struct CognitiveSession {
    /// Session ID
    pub id: String,
    
    /// Session start time
    pub started: DateTime<Utc>,
    
    /// Message count
    pub message_count: usize,
    
    /// Collected insights
    pub insights: Vec<PersistedInsight>,
    
    /// Modality usage tracking
    pub modality_usage: HashMap<String, ModalityUsageTracker>,
    
    /// Session context
    pub context: SessionContext,
    
    /// Learning outcomes
    pub learnings: Vec<LearningOutcome>,
    
    /// Key topics discussed
    pub key_topics: Vec<String>,
    
    /// Auto-save handle
    auto_save_handle: Option<tokio::task::JoinHandle<()>>,
}

/// Tracks modality usage
#[derive(Debug, Clone)]
pub struct ModalityUsageTracker {
    pub activation_count: u32,
    pub total_duration_ms: u64,
    pub confidence_sum: f64,
    pub last_activated: DateTime<Utc>,
}

/// Manages all cognitive sessions
pub struct CognitiveSessionManager {
    /// State persistence manager
    state_manager: Arc<CognitiveStateManager>,
    
    /// Active sessions
    active_sessions: Arc<RwLock<HashMap<String, CognitiveSession>>>,
    
    /// Cognitive system reference
    cognitive_system: Arc<CognitiveSystem>,
    
    /// Memory reference
    memory: Arc<CognitiveMemory>,
    
    /// Auto-save manager
    auto_save: Arc<CognitiveAutoSave>,
}

impl CognitiveSessionManager {
    /// Create new session manager
    pub fn new(
        state_dir: std::path::PathBuf,
        cognitive_system: Arc<CognitiveSystem>,
        memory: Arc<CognitiveMemory>,
    ) -> Result<Self> {
        let state_manager = Arc::new(
            CognitiveStateManager::new(state_dir, cognitive_system.clone(), memory.clone())?
        );
        
        let auto_save = Arc::new(CognitiveAutoSave::new(state_manager.clone()));
        
        Ok(Self {
            state_manager,
            active_sessions: Arc::new(RwLock::new(HashMap::new())),
            cognitive_system,
            memory,
            auto_save,
        })
    }
    
    /// Start a new session or resume existing
    pub async fn start_session(
        &self,
        session_id: &str,
        enhancement: Arc<CognitiveChatEnhancement>,
    ) -> Result<SessionStartResult> {
        info!("🚀 Starting cognitive session: {}", session_id);
        
        // Check if we can restore a previous session
        if let Some(restored_state) = self.state_manager.load_state(session_id).await? {
            // Restore the session
            let result = self.restore_session(session_id, restored_state, enhancement).await?;
            Ok(result)
        } else {
            // Create new session
            let result = self.create_new_session(session_id, enhancement).await?;
            Ok(result)
        }
    }
    
    /// Create a new session
    async fn create_new_session(
        &self,
        session_id: &str,
        enhancement: Arc<CognitiveChatEnhancement>,
    ) -> Result<SessionStartResult> {
        let session = CognitiveSession {
            id: session_id.to_string(),
            started: Utc::now(),
            message_count: 0,
            insights: Vec::new(),
            modality_usage: HashMap::new(),
            context: SessionContext {
                user_preferences: serde_json::json!({}),
                discussion_topics: Vec::new(),
                emotional_tone: "neutral".to_string(),
                interaction_style: "conversational".to_string(),
            },
            learnings: Vec::new(),
            key_topics: Vec::new(),
            auto_save_handle: None,
        };
        
        // Start auto-save
        let auto_save_handle = self.start_auto_save(session_id, enhancement).await?;
        
        let mut sessions = self.active_sessions.write().await;
        sessions.get_mut(session_id).unwrap().auto_save_handle = Some(auto_save_handle);
        sessions.insert(session_id.to_string(), session);
        
        Ok(SessionStartResult {
            session_id: session_id.to_string(),
            is_restored: false,
            restoration_summary: None,
        })
    }
    
    /// Restore an existing session
    async fn restore_session(
        &self,
        session_id: &str,
        state: PersistedCognitiveState,
        enhancement: Arc<CognitiveChatEnhancement>,
    ) -> Result<SessionStartResult> {
        // Restore cognitive state
        CognitiveStateRestorer::restore_state(
            &state,
            &self.cognitive_system,
            &self.memory,
        ).await?;
        
        // Create restoration summary
        let restoration_summary = CognitiveStateRestorer::create_restoration_summary(&state);
        
        // Convert persisted state to active session
        let mut modality_usage = HashMap::new();
        for usage in state.modality_usage {
            modality_usage.insert(
                usage.modality.clone(),
                ModalityUsageTracker {
                    activation_count: usage.activation_count,
                    total_duration_ms: usage.total_duration_ms,
                    confidence_sum: usage.average_confidence * usage.activation_count as f64,
                    last_activated: state.last_saved,
                },
            );
        }
        
        let session = CognitiveSession {
            id: session_id.to_string(),
            started: state.last_saved, // Continue from last save
            message_count: state.summary.message_count,
            insights: state.insights,
            modality_usage,
            context: state.context,
            learnings: state.learnings,
            key_topics: state.summary.key_topics,
            auto_save_handle: None,
        };
        
        // Start auto-save
        let auto_save_handle = self.start_auto_save(session_id, enhancement).await?;
        
        let mut sessions = self.active_sessions.write().await;
        sessions.insert(session_id.to_string(), session);
        sessions.get_mut(session_id).unwrap().auto_save_handle = Some(auto_save_handle);
        
        Ok(SessionStartResult {
            session_id: session_id.to_string(),
            is_restored: true,
            restoration_summary: Some(restoration_summary),
        })
    }
    
    /// Update session with cognitive response
    pub async fn update_session(
        &self,
        session_id: &str,
        user_message: &str,
        response: &CognitiveResponse,
    ) -> Result<()> {
        let mut sessions = self.active_sessions.write().await;
        if let Some(session) = sessions.get_mut(session_id) {
            // Increment message count
            session.message_count += 2; // User + AI
            
            // Extract and store insights
            for insight in &response.cognitive_insights {
                session.insights.push(PersistedInsight {
                    timestamp: Utc::now(),
                    content: insight.clone(),
                    category: "conversation".to_string(),
                    relevance: response.confidence as f64,
                    source_modalities: response.modalities_used.iter()
                        .map(|m| format!("{:?}", m))
                        .collect(),
                });
            }
            
            // Update modality usage
            let now = Utc::now();
            for modality in &response.modalities_used {
                let modality_str = format!("{:?}", modality);
                let tracker = session.modality_usage
                    .entry(modality_str)
                    .or_insert_with(|| ModalityUsageTracker {
                        activation_count: 0,
                        total_duration_ms: 0,
                        confidence_sum: 0.0,
                        last_activated: now,
                    });
                
                tracker.activation_count += 1;
                tracker.total_duration_ms += 100; // Approximate
                tracker.confidence_sum += response.confidence as f64;
                tracker.last_activated = now;
            }
            
            // Extract topics from user message
            self.extract_topics(session, user_message);
        }
        
        Ok(())
    }
    
    /// End a session
    pub async fn end_session(&self, session_id: &str) -> Result<SessionEndResult> {
        let mut sessions = self.active_sessions.write().await;
        
        if let Some(mut session) = sessions.remove(session_id) {
            // Stop auto-save
            if let Some(handle) = session.auto_save_handle.take() {
                self.auto_save.stop().await;
                handle.abort();
            }
            
            // Calculate final summary
            let duration = (Utc::now() - session.started).num_seconds() as u64;
            let summary = ConversationSummary {
                message_count: session.message_count,
                duration_seconds: duration,
                key_topics: session.key_topics.clone(),
                user_satisfaction: None, // Could be calculated from sentiment
                cognitive_depth_achieved: self.calculate_cognitive_depth(&session),
            };
            
            // Convert modality usage
            let modality_usage: Vec<ModalityUsage> = session.modality_usage.into_iter()
                .map(|(modality, tracker)| ModalityUsage {
                    modality,
                    activation_count: tracker.activation_count,
                    total_duration_ms: tracker.total_duration_ms,
                    average_confidence: if tracker.activation_count > 0 {
                        tracker.confidence_sum / tracker.activation_count as f64
                    } else {
                        0.0
                    },
                })
                .collect();
            
            // Final save
            let activity = CognitiveActivity {
                awareness_level: 0.0,
                active_insights: Vec::new(),
                gradient_coherence: 0.0,
                free_energy: 1.0,
                current_focus: "Session ended".to_string(),
                background_thoughts: 0,
            };
            
            // Capture counts before moving
            let insights_count = session.insights.len();
            let learnings_count = session.learnings.len();
            
            self.state_manager.save_state(
                session_id,
                &activity,
                session.insights,
                modality_usage,
                session.context,
                session.learnings,
                summary.clone(),
            ).await?;
            
            Ok(SessionEndResult {
                session_id: session_id.to_string(),
                duration_seconds: duration,
                message_count: session.message_count,
                insights_generated: insights_count,
                learnings_captured: learnings_count,
                summary,
            })
        } else {
            Err(anyhow::anyhow!("Session not found: {}", session_id))
        }
    }
    
    /// Start auto-save for a session
    async fn start_auto_save(
        &self,
        session_id: &str,
        enhancement: Arc<CognitiveChatEnhancement>,
    ) -> Result<tokio::task::JoinHandle<()>> {
        let session_id_clone = session_id.to_string();
        let sessions = self.active_sessions.clone();
        
        let get_state = move || -> CognitiveStateSnapshot {
            // This runs in the auto-save task
            let sessions_lock = futures::executor::block_on(sessions.read());
            if let Some(session) = sessions_lock.get(&session_id_clone) {
                // Get current consciousness activity
                let activity = enhancement.get_cognitive_activity();
                
                // Convert modality usage
                let modality_usage: Vec<ModalityUsage> = session.modality_usage.iter()
                    .map(|(modality, tracker)| ModalityUsage {
                        modality: modality.clone(),
                        activation_count: tracker.activation_count,
                        total_duration_ms: tracker.total_duration_ms,
                        average_confidence: if tracker.activation_count > 0 {
                            tracker.confidence_sum / tracker.activation_count as f64
                        } else {
                            0.0
                        },
                    })
                    .collect();
                
                // Create summary
                let duration = (Utc::now() - session.started).num_seconds() as u64;
                let summary = ConversationSummary {
                    message_count: session.message_count,
                    duration_seconds: duration,
                    key_topics: session.key_topics.clone(),
                    user_satisfaction: None,
                    cognitive_depth_achieved: 0.7, // Placeholder
                };
                
                CognitiveStateSnapshot {
                    activity,
                    insights: session.insights.clone(),
                    modality_usage,
                    context: session.context.clone(),
                    learnings: session.learnings.clone(),
                    summary,
                }
            } else {
                // Fallback empty state
                CognitiveStateSnapshot {
                    activity: CognitiveActivity {
                        awareness_level: 0.0,
                        active_insights: Vec::new(),
                        gradient_coherence: 0.0,
                        free_energy: 1.0,
                        current_focus: "Unknown".to_string(),
                        background_thoughts: 0,
                    },
                    insights: Vec::new(),
                    modality_usage: Vec::new(),
                    context: SessionContext {
                        user_preferences: serde_json::json!({}),
                        discussion_topics: Vec::new(),
                        emotional_tone: "neutral".to_string(),
                        interaction_style: "conversational".to_string(),
                    },
                    learnings: Vec::new(),
                    summary: ConversationSummary {
                        message_count: 0,
                        duration_seconds: 0,
                        key_topics: Vec::new(),
                        user_satisfaction: None,
                        cognitive_depth_achieved: 0.0,
                    },
                }
            }
        };
        
        Ok(self.auto_save.start(session_id.to_string(), get_state).await)
    }
    
    /// Extract topics from message
    fn extract_topics(&self, session: &mut CognitiveSession, message: &str) {
        // Simple topic extraction - could be enhanced with NLP
        let words: Vec<&str> = message.split_whitespace()
            .filter(|w| w.len() > 4)
            .collect();
        
        for word in words {
            let topic = word.to_lowercase();
            if !session.key_topics.contains(&topic) && session.key_topics.len() < 10 {
                session.key_topics.push(topic);
            }
        }
    }
    
    
    /// Calculate cognitive depth achieved
    fn calculate_cognitive_depth(&self, session: &CognitiveSession) -> f64 {
        let modality_diversity = session.modality_usage.len() as f64 / 8.0; // 8 total modalities
        let insight_quality = session.insights.iter()
            .map(|i| i.relevance)
            .sum::<f64>() / session.insights.len().max(1) as f64;
        let learning_depth = session.learnings.len() as f64 / 10.0; // Normalize to 10
        
        (modality_diversity + insight_quality + learning_depth) / 3.0
    }
    
    /// List all sessions (active and saved)
    pub async fn list_all_sessions(&self) -> Result<Vec<SessionInfo>> {
        let mut all_sessions = self.state_manager.list_sessions().await?;
        
        // Add active sessions
        let active = self.active_sessions.read().await;
        for (id, session) in active.iter() {
            all_sessions.push(SessionInfo {
                session_id: id.clone(),
                last_saved: session.started,
                message_count: session.message_count,
                key_topics: session.key_topics.clone(),
            });
        }
        
        // Sort by date
        all_sessions.sort_by(|a, b| b.last_saved.cmp(&a.last_saved));
        all_sessions.dedup_by(|a, b| a.session_id == b.session_id);
        
        Ok(all_sessions)
    }
}

/// Result of starting a session
#[derive(Debug, Clone)]
pub struct SessionStartResult {
    pub session_id: String,
    pub is_restored: bool,
    pub restoration_summary: Option<String>,
}

/// Result of ending a session
#[derive(Debug, Clone)]
pub struct SessionEndResult {
    pub session_id: String,
    pub duration_seconds: u64,
    pub message_count: usize,
    pub insights_generated: usize,
    pub learnings_captured: usize,
    pub summary: ConversationSummary,
}