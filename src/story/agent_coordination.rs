//! Story-driven agent coordination

use super::types::*;
use super::engine::StoryEngine;
use super::context_retrieval::{StoryContextRetriever, RetrievedContext};
use anyhow::Result;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use tokio::sync::broadcast;
use tracing::info;
use uuid::Uuid;

/// Story-driven agent coordinator
pub struct StoryAgentCoordinator {
    /// Story engine reference
    story_engine: Arc<StoryEngine>,

    /// Context retriever
    context_retriever: Arc<StoryContextRetriever>,

    /// Active agent registrations
    active_agents: Arc<DashMap<String, AgentRegistration>>,

    /// Coordination sessions
    coordination_sessions: Arc<DashMap<CoordinationSessionId, CoordinationSession>>,

    /// Event broadcaster
    event_tx: broadcast::Sender<CoordinationEvent>,

    /// Configuration
    config: CoordinationConfig,
}

/// Agent registration information
#[derive(Debug, Clone)]
pub struct AgentRegistration {
    pub agent_id: String,
    pub agent_type: String,
    pub story_id: StoryId,
    pub capabilities: Vec<AgentCapability>,
    pub status: AgentStatus,
    pub last_activity: chrono::DateTime<chrono::Utc>,
}

/// Agent capability
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AgentCapability {
    CodeGeneration(Vec<String>), // Languages
    CodeReview,
    Documentation,
    Testing,
    Debugging,
    Research,
    Planning,
    Communication,
    Custom(String),
}

/// Agent status
#[derive(Debug, Clone, PartialEq)]
pub enum AgentStatus {
    Available,
    Busy,
    Collaborating,
    Offline,
}

/// Coordination session ID
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CoordinationSessionId(pub Uuid);

impl fmt::Display for CoordinationSessionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Coordination session for multi-agent collaboration
#[derive(Debug, Clone)]
pub struct CoordinationSession {
    pub id: CoordinationSessionId,
    pub goal: String,
    pub participating_agents: Vec<String>,
    pub lead_agent: Option<String>,
    pub shared_context: SharedContext,
    pub status: SessionStatus,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub completed_at: Option<chrono::DateTime<chrono::Utc>>,
}

/// Shared context for coordination
#[derive(Debug, Clone)]
pub struct SharedContext {
    pub primary_story: StoryId,
    pub shared_goals: Vec<String>,
    pub shared_tasks: Vec<MappedTask>,
    pub shared_insights: Vec<String>,
    pub coordination_points: Vec<CoordinationPoint>,
}

/// Point where agents need to coordinate
#[derive(Debug, Clone)]
pub struct CoordinationPoint {
    pub id: Uuid,
    pub description: String,
    pub required_capabilities: Vec<AgentCapability>,
    pub assigned_agents: Vec<String>,
    pub status: CoordinationStatus,
}

/// Coordination status
#[derive(Debug, Clone, PartialEq)]
pub enum CoordinationStatus {
    Pending,
    InProgress,
    WaitingForInput,
    Completed,
    Failed,
}

/// Session status
#[derive(Debug, Clone, PartialEq)]
pub enum SessionStatus {
    Planning,
    Active,
    Executing,
    Reviewing,
    Completed,
    Cancelled,
}

/// Coordination event
#[derive(Debug, Clone)]
pub enum CoordinationEvent {
    AgentJoined(String, CoordinationSessionId),
    AgentLeft(String, CoordinationSessionId),
    TaskAssigned(String, MappedTask),
    TaskCompleted(String, MappedTask),
    ContextShared(String, RetrievedContext),
    SessionCompleted(CoordinationSessionId),
}

/// Configuration for coordination
#[derive(Debug, Clone)]
pub struct CoordinationConfig {
    /// Maximum agents per session
    pub max_agents_per_session: usize,

    /// Session timeout
    pub session_timeout: chrono::Duration,

    /// Enable automatic agent matching
    pub auto_match_agents: bool,

    /// Enable story-based planning
    pub story_based_planning: bool,

    /// Consensus threshold for decisions
    pub consensus_threshold: f32,
}

impl Default for CoordinationConfig {
    fn default() -> Self {
        Self {
            max_agents_per_session: 5,
            session_timeout: chrono::Duration::hours(1),
            auto_match_agents: true,
            story_based_planning: true,
            consensus_threshold: 0.7,
        }
    }
}

impl StoryAgentCoordinator {
    /// Create a new coordinator
    pub fn new(
        story_engine: Arc<StoryEngine>,
        context_retriever: Arc<StoryContextRetriever>,
        config: CoordinationConfig,
    ) -> Self {
        let (event_tx, _) = broadcast::channel(1000);

        Self {
            story_engine,
            context_retriever,
            active_agents: Arc::new(DashMap::new()),
            coordination_sessions: Arc::new(DashMap::new()),
            event_tx,
            config,
        }
    }

    /// Register an agent
    pub async fn register_agent(
        &self,
        agent_id: String,
        agent_type: String,
        capabilities: Vec<AgentCapability>,
    ) -> Result<StoryId> {
        // Create or get agent story
        let story_id = self.story_engine
            .create_agent_story(agent_id.clone(), agent_type.clone())
            .await?;

        // Register agent
        let registration = AgentRegistration {
            agent_id: agent_id.clone(),
            agent_type,
            story_id,
            capabilities,
            status: AgentStatus::Available,
            last_activity: chrono::Utc::now(),
        };

        self.active_agents.insert(agent_id.clone(), registration);

        // Add registration plot point
        self.story_engine.add_plot_point(
            story_id,
            PlotType::Discovery {
                insight: format!("Agent {} registered with coordinator", agent_id),
            },
            vec!["registration".to_string()],
        ).await?;

        info!("Registered agent {} with story {}", agent_id, story_id.0);

        Ok(story_id)
    }

    /// Create a coordination session
    pub async fn create_session(
        &self,
        goal: String,
        required_capabilities: Vec<AgentCapability>,
    ) -> Result<CoordinationSessionId> {
        let session_id = CoordinationSessionId(Uuid::new_v4());

        // Find suitable agents
        let suitable_agents = if self.config.auto_match_agents {
            self.find_suitable_agents(&required_capabilities).await?
        } else {
            vec![]
        };

        // Create primary story for session
        let primary_story = self.story_engine.create_agent_story(
            format!("session-{}", session_id.0),
            "Coordination Session".to_string(),
        ).await?;

        // Add goal
        self.story_engine.add_plot_point(
            primary_story,
            PlotType::Goal {
                objective: goal.clone(),
            },
            required_capabilities.iter()
                .map(|c| format!("{:?}", c))
                .collect(),
        ).await?;

        // Create session
        let session = CoordinationSession {
            id: session_id,
            goal: goal.clone(),
            participating_agents: suitable_agents.clone(),
            lead_agent: suitable_agents.first().cloned(),
            shared_context: SharedContext {
                primary_story,
                shared_goals: vec![goal.clone()],
                shared_tasks: vec![],
                shared_insights: vec![],
                coordination_points: vec![],
            },
            status: SessionStatus::Planning,
            created_at: chrono::Utc::now(),
            completed_at: None,
        };

        self.coordination_sessions.insert(session_id, session);

        // Notify agents
        for agent_id in suitable_agents {
            let _ = self.event_tx.send(CoordinationEvent::AgentJoined(
                agent_id.clone(),
                session_id,
            ));
        }

        info!("Created coordination session {} for goal: {}", session_id.0, goal);

        Ok(session_id)
    }

    /// Plan session using stories
    pub async fn plan_session(
        &self,
        session_id: CoordinationSessionId,
    ) -> Result<Vec<CoordinationPoint>> {
        let mut session = self.coordination_sessions
            .get_mut(&session_id)
            .ok_or_else(|| anyhow::anyhow!("Session not found"))?;

        if !self.config.story_based_planning {
            return Ok(vec![]);
        }

        // Analyze goal and create coordination points
        let coordination_points = self.analyze_goal_for_coordination(&session.goal).await?;

        // Add to session
        session.shared_context.coordination_points = coordination_points.clone();
        session.status = SessionStatus::Active;

        // Create plot points for planning
        for point in &coordination_points {
            self.story_engine.add_plot_point(
                session.shared_context.primary_story,
                PlotType::Task {
                    description: point.description.clone(),
                    completed: false,
                },
                vec![format!("coordination_point:{}", point.id)],
            ).await?;
        }

        Ok(coordination_points)
    }

    /// Assign agent to coordination point
    pub async fn assign_agent_to_point(
        &self,
        session_id: CoordinationSessionId,
        point_id: Uuid,
        agent_id: String,
    ) -> Result<()> {
        let mut session = self.coordination_sessions
            .get_mut(&session_id)
            .ok_or_else(|| anyhow::anyhow!("Session not found"))?;

        // Extract needed values before mutable borrow
        let primary_story = session.shared_context.primary_story;
        
        // Find and update coordination point
        for point in &mut session.shared_context.coordination_points {
            if point.id == point_id {
                if !point.assigned_agents.contains(&agent_id) {
                    let point_description = point.description.clone();
                    point.assigned_agents.push(agent_id.clone());
                    point.status = CoordinationStatus::InProgress;

                    // Update agent status
                    if let Some(mut agent) = self.active_agents.get_mut(&agent_id) {
                        agent.status = AgentStatus::Collaborating;
                        agent.last_activity = chrono::Utc::now();
                    }

                    // Add assignment plot point
                    self.story_engine.add_plot_point(
                        primary_story,
                        PlotType::Interaction {
                            with: agent_id.clone(),
                            action: format!("Assigned to: {}", point_description),
                        },
                        vec![],
                    ).await?;

                    break;
                }
            }
        }

        Ok(())
    }

    /// Share context between agents
    pub async fn share_context(
        &self,
        session_id: CoordinationSessionId,
        from_agent: String,
        context_summary: String,
    ) -> Result<()> {
        let session = self.coordination_sessions
            .get(&session_id)
            .ok_or_else(|| anyhow::anyhow!("Session not found"))?;

        // Get agent's context
        let context = self.context_retriever
            .retrieve_context_for_agent(&from_agent, Some(&context_summary))
            .await?;

        // Add insights to shared context
        let mut session = self.coordination_sessions
            .get_mut(&session_id)
            .unwrap();

        session.shared_context.shared_insights.extend(context.insights.clone());
        session.shared_context.shared_tasks.extend(context.relevant_tasks.clone());

        // Create synchronization event
        let sync_payload = crate::story::SyncPayload {
            plot_points: vec![],
            context_updates: HashMap::from([
                ("shared_by".to_string(), from_agent.clone()),
                ("summary".to_string(), context_summary),
            ]),
            metadata_changes: HashMap::new(),
        };

        // Sync with all participating agents
        let agent_stories: Vec<StoryId> = session.participating_agents
            .iter()
            .filter_map(|aid| {
                self.active_agents.get(aid).map(|a| a.story_id)
            })
            .collect();

        self.story_engine.sync_stories(agent_stories).await?;

        // Broadcast event
        let _ = self.event_tx.send(CoordinationEvent::ContextShared(
            from_agent,
            context,
        ));

        Ok(())
    }

    /// Complete a coordination point
    pub async fn complete_coordination_point(
        &self,
        session_id: CoordinationSessionId,
        point_id: Uuid,
        agent_id: String,
        result: String,
    ) -> Result<()> {
        let mut session = self.coordination_sessions
            .get_mut(&session_id)
            .ok_or_else(|| anyhow::anyhow!("Session not found"))?;

        // Extract needed values before mutable borrow
        let primary_story = session.shared_context.primary_story;
        
        // Update coordination point
        for point in &mut session.shared_context.coordination_points {
            if point.id == point_id {
                let point_description = point.description.clone();
                point.status = CoordinationStatus::Completed;

                // Add completion plot point
                self.story_engine.add_plot_point(
                    primary_story,
                    PlotType::Task {
                        description: format!("{} - Completed by {}", point_description, agent_id),
                        completed: true,
                    },
                    vec![result],
                ).await?;

                break;
            }
        }

        // Check if all points are completed
        let all_completed = session.shared_context.coordination_points
            .iter()
            .all(|p| p.status == CoordinationStatus::Completed);

        if all_completed {
            session.status = SessionStatus::Reviewing;
            info!("All coordination points completed for session {}", session_id.0);
        }

        Ok(())
    }

    /// Complete a session
    pub async fn complete_session(
        &self,
        session_id: CoordinationSessionId,
        summary: String,
    ) -> Result<()> {
        let mut session = self.coordination_sessions
            .get_mut(&session_id)
            .ok_or_else(|| anyhow::anyhow!("Session not found"))?;

        session.status = SessionStatus::Completed;
        session.completed_at = Some(chrono::Utc::now());

        // Add completion plot point
        self.story_engine.add_plot_point(
            session.shared_context.primary_story,
            PlotType::Goal {
                objective: format!("Session completed: {}", summary),
            },
            vec![],
        ).await?;

        // Update agent statuses
        for agent_id in &session.participating_agents {
            if let Some(mut agent) = self.active_agents.get_mut(agent_id) {
                agent.status = AgentStatus::Available;
            }
        }

        // Broadcast completion
        let _ = self.event_tx.send(CoordinationEvent::SessionCompleted(session_id));

        info!("Completed coordination session {}", session_id.0);

        Ok(())
    }

    /// Get agent recommendations for a task
    pub async fn recommend_agents_for_task(
        &self,
        task_description: &str,
        required_capabilities: Vec<AgentCapability>,
    ) -> Result<Vec<AgentRecommendation>> {
        let mut recommendations = Vec::new();

        for agent_ref in self.active_agents.iter() {
            let agent = agent_ref.value();

            // Check capabilities
            let capability_match = required_capabilities
                .iter()
                .filter(|req| agent.capabilities.contains(req))
                .count() as f32 / required_capabilities.len() as f32;

            if capability_match > 0.0 {
                // Get agent's recent performance
                let performance = self.get_agent_performance(&agent.agent_id).await?;

                // Calculate recommendation score
                let score = capability_match * 0.6 +
                           performance.success_rate * 0.3 +
                           (if agent.status == AgentStatus::Available { 0.1 } else { 0.0 });

                recommendations.push(AgentRecommendation {
                    agent_id: agent.agent_id.clone(),
                    agent_type: agent.agent_type.clone(),
                    score,
                    capability_match,
                    availability: agent.status == AgentStatus::Available,
                    recent_performance: performance,
                });
            }
        }

        // Sort by score
        recommendations.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        Ok(recommendations)
    }

    /// Subscribe to coordination events
    pub fn subscribe(&self) -> broadcast::Receiver<CoordinationEvent> {
        self.event_tx.subscribe()
    }

    // Helper methods

    async fn find_suitable_agents(&self, capabilities: &[AgentCapability]) -> Result<Vec<String>> {
        let mut suitable = Vec::new();

        for agent_ref in self.active_agents.iter() {
            let agent = agent_ref.value();

            // Check if agent has required capabilities
            let has_capabilities = capabilities.iter().any(|cap| {
                agent.capabilities.contains(cap)
            });

            if has_capabilities && agent.status == AgentStatus::Available {
                suitable.push(agent.agent_id.clone());

                if suitable.len() >= self.config.max_agents_per_session {
                    break;
                }
            }
        }

        Ok(suitable)
    }

    async fn analyze_goal_for_coordination(&self, goal: &str) -> Result<Vec<CoordinationPoint>> {
        let mut points = Vec::new();

        // Simple heuristic analysis - in production, use NLP
        let goal_lower = goal.to_lowercase();

        if goal_lower.contains("api") || goal_lower.contains("endpoint") {
            points.push(CoordinationPoint {
                id: Uuid::new_v4(),
                description: "Design API specification".to_string(),
                required_capabilities: vec![AgentCapability::Planning, AgentCapability::Documentation],
                assigned_agents: vec![],
                status: CoordinationStatus::Pending,
            });

            points.push(CoordinationPoint {
                id: Uuid::new_v4(),
                description: "Implement API endpoints".to_string(),
                required_capabilities: vec![AgentCapability::CodeGeneration(vec!["rust".to_string()])],
                assigned_agents: vec![],
                status: CoordinationStatus::Pending,
            });

            points.push(CoordinationPoint {
                id: Uuid::new_v4(),
                description: "Write API tests".to_string(),
                required_capabilities: vec![AgentCapability::Testing],
                assigned_agents: vec![],
                status: CoordinationStatus::Pending,
            });
        }

        if goal_lower.contains("debug") || goal_lower.contains("fix") {
            points.push(CoordinationPoint {
                id: Uuid::new_v4(),
                description: "Analyze and identify issue".to_string(),
                required_capabilities: vec![AgentCapability::Debugging, AgentCapability::Research],
                assigned_agents: vec![],
                status: CoordinationStatus::Pending,
            });

            points.push(CoordinationPoint {
                id: Uuid::new_v4(),
                description: "Implement fix".to_string(),
                required_capabilities: vec![AgentCapability::CodeGeneration(vec![])],
                assigned_agents: vec![],
                status: CoordinationStatus::Pending,
            });
        }

        // Default coordination point if none found
        if points.is_empty() {
            points.push(CoordinationPoint {
                id: Uuid::new_v4(),
                description: goal.to_string(),
                required_capabilities: vec![AgentCapability::Planning],
                assigned_agents: vec![],
                status: CoordinationStatus::Pending,
            });
        }

        Ok(points)
    }

    async fn get_agent_performance(&self, agent_id: &str) -> Result<AgentPerformance> {
        if let Some(agent) = self.active_agents.get(agent_id) {
            if let Some(story) = self.story_engine.get_story(&agent.story_id) {
                let agent_story = crate::story::AgentStory::new(story);
                let metrics = crate::story::agent_story::AgentStoryAnalyzer::analyze_performance(&agent_story);

                Ok(AgentPerformance {
                    total_tasks: metrics.total_tasks,
                    completed_tasks: metrics.completed_tasks,
                    success_rate: metrics.task_completion_rate,
                    collaboration_count: metrics.interaction_count,
                })
            } else {
                Ok(AgentPerformance::default())
            }
        } else {
            Ok(AgentPerformance::default())
        }
    }
}

/// Agent recommendation
#[derive(Debug, Clone)]
pub struct AgentRecommendation {
    pub agent_id: String,
    pub agent_type: String,
    pub score: f32,
    pub capability_match: f32,
    pub availability: bool,
    pub recent_performance: AgentPerformance,
}

/// Agent performance metrics
#[derive(Debug, Clone, Default)]
pub struct AgentPerformance {
    pub total_tasks: usize,
    pub completed_tasks: usize,
    pub success_rate: f32,
    pub collaboration_count: usize,
}
