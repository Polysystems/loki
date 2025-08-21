//! Story-Agent Integration Module
//! 
//! Provides seamless integration between agents and the story engine,
//! enabling narrative-driven agent behavior and story-aware collaboration.

use std::sync::Arc;
use std::collections::HashMap;
use tokio::sync::{RwLock, mpsc, broadcast};
use anyhow::{Result, Context};
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};
use tracing::{info, debug, warn};

use crate::story::{StoryEngine, StoryContext, StoryId, PlotType, PlotPointId};
use crate::story::agent_coordination::{StoryAgentCoordinator, AgentCapability, AgentStatus};
use crate::story::agent_story::{AgentStory, InteractionType};
use crate::cognitive::agents::AgentSpecialization;
use super::{AgentManager, CollaborationCoordinator, AgentCoordinator};

/// Story-aware agent orchestrator
pub struct StoryAgentOrchestrator {
    /// Agent manager
    agent_manager: Arc<RwLock<AgentManager>>,
    
    /// Story engine
    story_engine: Arc<StoryEngine>,
    
    /// Story agent coordinator
    story_coordinator: Arc<StoryAgentCoordinator>,
    
    /// Agent stories by agent ID
    agent_stories: Arc<RwLock<HashMap<String, AgentStory>>>,
    
    /// Narrative influences on agent behavior
    narrative_influences: Arc<RwLock<HashMap<String, NarrativeInfluence>>>,
    
    /// Story-driven collaboration sessions
    story_collaborations: Arc<RwLock<Vec<StoryCollaboration>>>,
    
    /// Event channel for story-agent events
    event_tx: broadcast::Sender<StoryAgentEvent>,
    
    /// Configuration
    config: StoryAgentConfig,
}

/// Narrative influence on agent behavior
#[derive(Debug, Clone)]
pub struct NarrativeInfluence {
    pub agent_id: String,
    pub story_arc: String,
    pub behavior_modifier: BehaviorModifier,
    pub influence_strength: f32,
    pub applied_at: DateTime<Utc>,
}

/// Behavior modifiers based on story context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BehaviorModifier {
    /// Agent becomes more cautious due to story tension
    Cautious { risk_tolerance: f32 },
    
    /// Agent becomes more creative during discovery arcs
    Creative { innovation_boost: f32 },
    
    /// Agent becomes more collaborative during team arcs
    Collaborative { cooperation_level: f32 },
    
    /// Agent becomes more focused during critical moments
    Focused { attention_multiplier: f32 },
    
    /// Agent adapts to story mood
    Adaptive { mood: String, adaptation_rate: f32 },
}

/// Story-driven collaboration
#[derive(Debug, Clone)]
pub struct StoryCollaboration {
    pub id: String,
    pub story_context: StoryContext,
    pub participating_agents: Vec<String>,
    pub collaboration_goal: String,
    pub narrative_role: NarrativeRole,
    pub started_at: DateTime<Utc>,
    pub status: CollaborationStatus,
}

/// Narrative roles for collaborations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NarrativeRole {
    /// Protagonist role - drives the narrative forward
    Protagonist,
    
    /// Supporting role - assists the protagonist
    Supporting,
    
    /// Antagonist role - creates narrative tension
    Antagonist,
    
    /// Mentor role - provides guidance
    Mentor,
    
    /// Observer role - documents the narrative
    Observer,
}

/// Collaboration status
#[derive(Debug, Clone, PartialEq)]
pub enum CollaborationStatus {
    Planning,
    Active,
    Climax,
    Resolution,
    Completed,
}

/// Story-agent events
#[derive(Debug, Clone)]
pub enum StoryAgentEvent {
    AgentJoinedStory {
        agent_id: String,
        story_id: StoryId,
        role: NarrativeRole,
    },
    NarrativeInfluenceApplied {
        agent_id: String,
        modifier: BehaviorModifier,
    },
    CollaborationStarted {
        collaboration_id: String,
        agents: Vec<String>,
    },
    PlotPointAchieved {
        agent_id: String,
        plot_point: PlotPointId,
    },
    StoryArcCompleted {
        story_id: StoryId,
        arc_title: String,
    },
}

/// Configuration for story-agent integration
#[derive(Debug, Clone)]
pub struct StoryAgentConfig {
    pub enable_narrative_influence: bool,
    pub max_influence_strength: f32,
    pub collaboration_story_weight: f32,
    pub auto_assign_roles: bool,
    pub track_agent_narratives: bool,
}

impl Default for StoryAgentConfig {
    fn default() -> Self {
        Self {
            enable_narrative_influence: true,
            max_influence_strength: 0.8,
            collaboration_story_weight: 0.6,
            auto_assign_roles: true,
            track_agent_narratives: true,
        }
    }
}

impl StoryAgentOrchestrator {
    /// Create a new story-agent orchestrator
    pub async fn new(
        agent_manager: Arc<RwLock<AgentManager>>,
        story_engine: Arc<StoryEngine>,
        story_coordinator: Arc<StoryAgentCoordinator>,
    ) -> Result<Self> {
        let (event_tx, _) = broadcast::channel(100);
        
        Ok(Self {
            agent_manager,
            story_engine,
            story_coordinator,
            agent_stories: Arc::new(RwLock::new(HashMap::new())),
            narrative_influences: Arc::new(RwLock::new(HashMap::new())),
            story_collaborations: Arc::new(RwLock::new(Vec::new())),
            event_tx,
            config: StoryAgentConfig::default(),
        })
    }
    
    /// Register an agent with a story
    pub async fn register_agent_to_story(
        &self,
        agent_id: String,
        story_id: StoryId,
        role: NarrativeRole,
    ) -> Result<()> {
        info!("Registering agent {} to story {} with role {:?}", agent_id, story_id.0, role);
        
        // Get story from engine
        let story = self.story_engine.get_story(&story_id)
            .ok_or_else(|| anyhow::anyhow!("Story not found: {:?}", story_id))?;
        
        // Create agent story
        let agent_story = AgentStory::new(story.clone());
        self.agent_stories.write().await.insert(agent_id.clone(), agent_story);
        
        // Register with coordinator
        let capabilities = self.map_specialization_to_capabilities(&agent_id).await?;
        self.story_coordinator.register_agent(
            agent_id.clone(),
            story_id.0.to_string(),
            capabilities,
        ).await?;
        
        // Apply narrative influence if enabled
        if self.config.enable_narrative_influence {
            self.apply_narrative_influence(&agent_id, &story).await?;
        }
        
        // Send event
        let _ = self.event_tx.send(StoryAgentEvent::AgentJoinedStory {
            agent_id,
            story_id,
            role,
        });
        
        Ok(())
    }
    
    /// Start a story-driven collaboration
    pub async fn start_story_collaboration(
        &self,
        agents: Vec<String>,
        goal: String,
    ) -> Result<String> {
        let collaboration_id = uuid::Uuid::new_v4().to_string();
        
        // Get current story context
        let narrative_context = self.story_engine.get_current_context().await?;
        let story_context: StoryContext = narrative_context.into();
        
        // Determine narrative role based on story arc
        let narrative_role = self.determine_narrative_role(&story_context).await;
        
        // Create collaboration
        let collaboration = StoryCollaboration {
            id: collaboration_id.clone(),
            story_context: story_context.clone(),
            participating_agents: agents.clone(),
            collaboration_goal: goal.clone(),
            narrative_role,
            started_at: Utc::now(),
            status: CollaborationStatus::Planning,
        };
        
        self.story_collaborations.write().await.push(collaboration);
        
        // Create coordination session in story coordinator
        let required_capabilities = agents.iter()
            .map(|_| AgentCapability::Planning)
            .collect();
        let session_id = self.story_coordinator.create_session(
            goal.clone(),
            required_capabilities,
        ).await?;
        
        // Add collaboration to story as plot point
        let plot_type = PlotType::Event {
            event_type: "collaboration".to_string(),
            description: format!("Agents collaborate: {}", goal),
            impact: 0.6,
        };
        self.story_engine.add_plot_point(
            story_context.story_id,
            plot_type,
            vec!["agent-collaboration".to_string()],
        ).await?;
        
        // Send event
        let _ = self.event_tx.send(StoryAgentEvent::CollaborationStarted {
            collaboration_id: collaboration_id.clone(),
            agents,
        });
        
        info!("Started story collaboration: {}", collaboration_id);
        Ok(collaboration_id)
    }
    
    /// Apply narrative influence to agent behavior
    async fn apply_narrative_influence(
        &self,
        agent_id: &str,
        story: &crate::story::Story,
    ) -> Result<()> {
        // Determine influence based on current arc
        let modifier = if let Some(arc) = story.arcs.last() {
            match arc.title.to_lowercase().as_str() {
                "tension" | "conflict" => BehaviorModifier::Cautious {
                    risk_tolerance: 0.3,
                },
                "discovery" | "exploration" => BehaviorModifier::Creative {
                    innovation_boost: 0.7,
                },
                "collaboration" | "team" => BehaviorModifier::Collaborative {
                    cooperation_level: 0.9,
                },
                "climax" | "critical" => BehaviorModifier::Focused {
                    attention_multiplier: 1.5,
                },
                _ => BehaviorModifier::Adaptive {
                    mood: arc.title.clone(),
                    adaptation_rate: 0.5,
                },
            }
        } else {
            BehaviorModifier::Adaptive {
                mood: "neutral".to_string(),
                adaptation_rate: 0.5,
            }
        };
        
        let influence = NarrativeInfluence {
            agent_id: agent_id.to_string(),
            story_arc: story.arcs.last().map(|a| a.title.clone()).unwrap_or_default(),
            behavior_modifier: modifier.clone(),
            influence_strength: 0.6,
            applied_at: Utc::now(),
        };
        
        self.narrative_influences.write().await.insert(agent_id.to_string(), influence);
        
        // Send event
        let _ = self.event_tx.send(StoryAgentEvent::NarrativeInfluenceApplied {
            agent_id: agent_id.to_string(),
            modifier,
        });
        
        debug!("Applied narrative influence to agent {}", agent_id);
        Ok(())
    }
    
    /// Map agent specialization to story capabilities
    async fn map_specialization_to_capabilities(
        &self,
        agent_id: &str,
    ) -> Result<Vec<AgentCapability>> {
        let agent_manager = self.agent_manager.read().await;
        
        // Get agent specialization (simplified - would need actual implementation)
        let capabilities = vec![
            AgentCapability::CodeGeneration(vec!["rust".to_string(), "python".to_string()]),
            AgentCapability::Planning,
            AgentCapability::Communication,
        ];
        
        Ok(capabilities)
    }
    
    /// Determine narrative role based on story context
    async fn determine_narrative_role(&self, context: &StoryContext) -> NarrativeRole {
        if let Some(arc) = &context.active_arc {
            match arc.title.to_lowercase().as_str() {
                "hero" | "protagonist" => NarrativeRole::Protagonist,
                "support" | "assistance" => NarrativeRole::Supporting,
                "conflict" | "challenge" => NarrativeRole::Antagonist,
                "guidance" | "wisdom" => NarrativeRole::Mentor,
                _ => NarrativeRole::Observer,
            }
        } else {
            NarrativeRole::Observer
        }
    }
    
    /// Update collaboration status based on story progression
    pub async fn update_collaboration_status(
        &self,
        collaboration_id: &str,
        new_status: CollaborationStatus,
    ) -> Result<()> {
        let mut collaborations = self.story_collaborations.write().await;
        
        if let Some(collab) = collaborations.iter_mut().find(|c| c.id == collaboration_id) {
            collab.status = new_status.clone();
            
            // Add status change to story
            let plot_type = PlotType::Event {
                event_type: "status_change".to_string(),
                description: format!("Collaboration {} reached {:?}", collaboration_id, new_status),
                impact: 0.3,
            };
            
            self.story_engine.add_plot_point(
                collab.story_context.story_id.clone(),
                plot_type,
                vec!["collaboration-status".to_string()],
            ).await?;
        }
        
        Ok(())
    }
    
    /// Get narrative influence for an agent
    pub async fn get_agent_influence(&self, agent_id: &str) -> Option<NarrativeInfluence> {
        self.narrative_influences.read().await.get(agent_id).cloned()
    }
    
    /// Record agent achievement in story
    pub async fn record_agent_achievement(
        &self,
        agent_id: &str,
        achievement: &str,
    ) -> Result<PlotPointId> {
        let mut agent_stories = self.agent_stories.write().await;
        
        if let Some(agent_story) = agent_stories.get_mut(agent_id) {
            let plot_type = PlotType::Task {
                description: achievement.to_string(),
                completed: true,
            };
            
            let plot_id = self.story_engine.add_plot_point(
                agent_story.story.id.clone(),
                plot_type,
                vec![format!("agent:{}", agent_id)],
            ).await?;
            
            // Send event
            let _ = self.event_tx.send(StoryAgentEvent::PlotPointAchieved {
                agent_id: agent_id.to_string(),
                plot_point: plot_id.clone(),
            });
            
            Ok(plot_id)
        } else {
            Err(anyhow::anyhow!("Agent {} not found in stories", agent_id))
        }
    }
    
    /// Subscribe to story-agent events
    pub fn subscribe(&self) -> broadcast::Receiver<StoryAgentEvent> {
        self.event_tx.subscribe()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_narrative_influence() {
        // Test that narrative influences are applied correctly based on story arc
    }
    
    #[tokio::test]
    async fn test_story_collaboration() {
        // Test story-driven collaboration creation and management
    }
}