//! Agent-specific story functionality

use super::types::*;
use anyhow::Result;
use std::collections::HashMap;
use tracing::info;

/// Represents a story for an agent
pub struct AgentStory {
    pub story: Story,
    pub current_goal: Option<PlotPointId>,
    pub interaction_history: Vec<AgentInteraction>,
}

/// Record of agent interactions
#[derive(Debug, Clone)]
pub struct AgentInteraction {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub other_agent: String,
    pub interaction_type: InteractionType,
    pub outcome: Option<String>,
}

/// Types of agent interactions
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum InteractionType {
    Collaboration,
    InformationExchange,
    TaskDelegation,
    ConflictResolution,
    StatusUpdate,
}

impl AgentStory {
    /// Create a new agent story
    pub fn new(story: Story) -> Self {
        Self {
            story,
            current_goal: None,
            interaction_history: Vec::new(),
        }
    }
    
    /// Set the agent's current goal
    pub async fn set_goal(
        &mut self,
        goal: String,
        story_engine: &crate::story::StoryEngine,
    ) -> Result<PlotPointId> {
        let plot_id = story_engine.add_plot_point(
            self.story.id,
            PlotType::Goal { objective: goal },
            vec![],
        ).await?;
        
        self.current_goal = Some(plot_id);
        info!("Agent {} set new goal: {}", self.story.id.0, plot_id.0);
        
        Ok(plot_id)
    }
    
    /// Record a decision made by the agent
    pub async fn record_decision(
        &mut self,
        question: String,
        choice: String,
        reasoning: Vec<String>,
        story_engine: &crate::story::StoryEngine,
    ) -> Result<PlotPointId> {
        story_engine.add_plot_point(
            self.story.id,
            PlotType::Decision { question, choice },
            reasoning,
        ).await
    }
    
    /// Record an interaction with another agent
    pub async fn record_interaction(
        &mut self,
        other_agent: String,
        interaction_type: InteractionType,
        details: String,
        story_engine: &crate::story::StoryEngine,
    ) -> Result<PlotPointId> {
        let interaction = AgentInteraction {
            timestamp: chrono::Utc::now(),
            other_agent: other_agent.clone(),
            interaction_type: interaction_type.clone(),
            outcome: None,
        };
        
        self.interaction_history.push(interaction);
        
        let action = match interaction_type {
            InteractionType::Collaboration => "collaborated",
            InteractionType::InformationExchange => "exchanged information",
            InteractionType::TaskDelegation => "delegated task",
            InteractionType::ConflictResolution => "resolved conflict",
            InteractionType::StatusUpdate => "provided status update",
        };
        
        story_engine.add_plot_point(
            self.story.id,
            PlotType::Interaction {
                with: other_agent,
                action: format!("{}: {}", action, details),
            },
            vec![],
        ).await
    }
    
    /// Complete the current goal
    pub async fn complete_goal(
        &mut self,
        outcome: String,
        story_engine: &crate::story::StoryEngine,
    ) -> Result<()> {
        if let Some(goal_id) = self.current_goal.take() {
            // Add completion plot point
            story_engine.add_plot_point(
                self.story.id,
                PlotType::Task {
                    description: format!("Completed goal: {}", outcome),
                    completed: true,
                },
                vec![format!("Goal ID: {}", goal_id.0)],
            ).await?;
            
            info!("Agent {} completed goal {}", self.story.id.0, goal_id.0);
        }
        
        Ok(())
    }
    
    /// Get agent's current status
    pub fn get_status(&self) -> AgentStatus {
        let has_goal = self.current_goal.is_some();
        let recent_interactions = self.interaction_history
            .iter()
            .filter(|i| {
                let age = chrono::Utc::now() - i.timestamp;
                age.num_minutes() < 30
            })
            .count();
        
        if has_goal && recent_interactions > 0 {
            AgentStatus::Active
        } else if has_goal {
            AgentStatus::Working
        } else if recent_interactions > 0 {
            AgentStatus::Available
        } else {
            AgentStatus::Idle
        }
    }
    
    /// Get collaboration suggestions based on story analysis
    pub async fn get_collaboration_suggestions(
        &self,
        all_agents: &[AgentStory],
    ) -> Vec<CollaborationSuggestion> {
        let mut suggestions = Vec::new();
        
        // Find agents with complementary goals
        if let Some(my_goal) = &self.current_goal {
            for other_agent in all_agents {
                if other_agent.story.id == self.story.id {
                    continue;
                }
                
                if let Some(their_goal) = &other_agent.current_goal {
                    // Simple similarity check (in production, use embeddings)
                    if self.goals_are_related(my_goal, their_goal) {
                        suggestions.push(CollaborationSuggestion {
                            agent_id: other_agent.story.id,
                            reason: "Related goals detected".to_string(),
                            suggested_action: InteractionType::Collaboration,
                            priority: 0.8,
                        });
                    }
                }
            }
        }
        
        // Find agents that haven't interacted recently
        let one_hour_ago = chrono::Utc::now() - chrono::Duration::hours(1);
        let recent_interactions: HashMap<String, _> = self.interaction_history
            .iter()
            .filter(|i| i.timestamp > one_hour_ago)
            .map(|i| (i.other_agent.clone(), i))
            .collect();
        
        for other_agent in all_agents {
            if other_agent.story.id == self.story.id {
                continue;
            }
            
            let agent_id = format!("{}", other_agent.story.id.0);
            if !recent_interactions.contains_key(&agent_id) {
                suggestions.push(CollaborationSuggestion {
                    agent_id: other_agent.story.id,
                    reason: "Haven't interacted recently".to_string(),
                    suggested_action: InteractionType::StatusUpdate,
                    priority: 0.3,
                });
            }
        }
        
        suggestions.sort_by(|a, b| b.priority.partial_cmp(&a.priority).unwrap());
        suggestions
    }
    
    // Helper methods
    
    fn goals_are_related(&self, _goal1: &PlotPointId, _goal2: &PlotPointId) -> bool {
        // Simplified - in production, compare actual goal content
        true
    }
}

/// Agent status
#[derive(Debug, Clone, PartialEq)]
pub enum AgentStatus {
    Idle,
    Available,
    Working,
    Active,
}

/// Collaboration suggestion
#[derive(Debug, Clone)]
pub struct CollaborationSuggestion {
    pub agent_id: StoryId,
    pub reason: String,
    pub suggested_action: InteractionType,
    pub priority: f32,
}

/// Agent story analysis utilities
pub struct AgentStoryAnalyzer;

impl AgentStoryAnalyzer {
    /// Analyze agent performance from story
    pub fn analyze_performance(agent_story: &AgentStory) -> PerformanceMetrics {
        let mut completed_tasks = 0;
        let mut total_tasks = 0;
        let mut decisions_made = 0;
        let mut issues_resolved = 0;
        
        for arc in &agent_story.story.arcs {
            for plot_point in &arc.plot_points {
                match &plot_point.plot_type {
                    PlotType::Task { completed, .. } => {
                        total_tasks += 1;
                        if *completed {
                            completed_tasks += 1;
                        }
                    }
                    PlotType::Decision { .. } => {
                        decisions_made += 1;
                    }
                    PlotType::Issue { resolved, .. } => {
                        if *resolved {
                            issues_resolved += 1;
                        }
                    }
                    _ => {}
                }
            }
        }
        
        let task_completion_rate = if total_tasks > 0 {
            completed_tasks as f32 / total_tasks as f32
        } else {
            0.0
        };
        
        PerformanceMetrics {
            task_completion_rate,
            total_tasks,
            completed_tasks,
            decisions_made,
            issues_resolved,
            interaction_count: agent_story.interaction_history.len(),
        }
    }
    
    /// Find patterns in agent behavior
    pub fn find_behavior_patterns(agent_story: &AgentStory) -> Vec<BehaviorPattern> {
        let mut patterns = Vec::new();
        
        // Analyze interaction patterns
        let mut interaction_counts: HashMap<InteractionType, usize> = HashMap::new();
        for interaction in &agent_story.interaction_history {
            *interaction_counts.entry(interaction.interaction_type.clone())
                .or_insert(0) += 1;
        }
        
        if let Some((dominant_type, count)) = interaction_counts.iter()
            .max_by_key(|(_, count)| *count)
        {
            if *count > agent_story.interaction_history.len() / 2 {
                patterns.push(BehaviorPattern::DominantInteractionType(
                    dominant_type.clone()
                ));
            }
        }
        
        // More pattern detection would go here...
        
        patterns
    }
}

/// Performance metrics for an agent
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub task_completion_rate: f32,
    pub total_tasks: usize,
    pub completed_tasks: usize,
    pub decisions_made: usize,
    pub issues_resolved: usize,
    pub interaction_count: usize,
}

/// Detected behavior patterns
#[derive(Debug, Clone)]
pub enum BehaviorPattern {
    DominantInteractionType(InteractionType),
    FrequentCollaborator(String),
    TaskFocused,
    ProblemSolver,
}