//! Goal Management System
//!
//! This module implements hierarchical goal management with dynamic
//! prioritization, progress tracking, and conflict resolution capabilities.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use tokio::sync::{RwLock, broadcast, mpsc};
use tokio::time::interval;
use tracing::{debug, info};

use crate::cognitive::{
    Thought,
    ThoughtId,
    ThoughtType,
};
use crate::cognitive::decision_engine::{CriterionType, DecisionCriterion, DecisionEngine, DecisionId, DecisionOption, OptimizationType};
use crate::cognitive::emotional_core::EmotionalCore;
use crate::cognitive::neuroprocessor::NeuroProcessor;
use crate::memory::{CognitiveMemory, MemoryId, MemoryMetadata};

// Helper functions for default values
fn default_option_instant() -> Option<Instant> {
    None
}

/// Unique identifier for goals
#[derive(Clone, Debug, Hash, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub struct GoalId(String);

impl GoalId {
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4().to_string())
    }
}

impl std::fmt::Display for GoalId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Goal type categorization
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GoalType {
    Strategic,   // Long-term vision goals
    Tactical,    // Medium-term objectives
    Operational, // Short-term tasks
    Learning,    // Knowledge acquisition
    Maintenance, // System upkeep
    Social,      // Relationship goals
    Creative,    // Creative pursuits
    Personal,    // Personal development
    Problem,     // Problem-solving goals
    Achievement, // Achievement-based goals
}

/// Goal state
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum GoalState {
    Proposed,  // Not yet active
    Active,    // Currently being pursued
    Suspended, // Temporarily paused
    Completed, // Successfully achieved
    Failed,    // Could not be achieved
    Abandoned, // Deliberately dropped
}

/// Goal priority level
#[derive(Clone, Debug, PartialEq, PartialOrd, Serialize, Deserialize)]
pub enum Priority {
    Low,
    Medium,
    High,
    Critical,
}

impl Priority {
    pub fn new(value: f32) -> Self {
        match value {
            x if x >= 0.9 => Priority::Critical,
            x if x >= 0.7 => Priority::High,
            x if x >= 0.4 => Priority::Medium,
            _ => Priority::Low,
        }
    }

    pub fn to_f32(&self) -> f32 {
        match self {
            Priority::Low => 0.3,
            Priority::Medium => 0.6,
            Priority::High => 0.8,
            Priority::Critical => 0.95,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Priority::Low => "LOW",
            Priority::Medium => "MEDIUM",
            Priority::High => "HIGH",
            Priority::Critical => "CRITICAL",
        }
    }
}

/// Individual goal
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Goal {
    pub id: GoalId,
    pub name: String,
    pub description: String,
    pub goal_type: GoalType,
    pub state: GoalState,
    pub priority: Priority,
    pub parent: Option<GoalId>,
    pub children: Vec<GoalId>,
    pub dependencies: Vec<GoalId>,
    pub progress: f32, // 0.0 to 1.0
    #[serde(skip, default = "default_option_instant")]
    pub target_completion: Option<Instant>,
    #[serde(skip, default = "default_option_instant")]
    pub actual_completion: Option<Instant>,
    #[serde(skip, default = "Instant::now")]
    pub created_at: Instant,
    #[serde(skip, default = "Instant::now")]
    pub last_updated: Instant,
    pub success_criteria: Vec<SuccessCriterion>,
    pub resources_required: ResourceRequirements,
    pub emotional_significance: f32, // -1.0 to 1.0
}

/// Success criterion for a goal
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuccessCriterion {
    pub description: String,
    pub measurable: bool,
    pub target_value: Option<f32>,
    pub current_value: Option<f32>,
    pub met: bool,
}

/// Resources required for a goal
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ResourceRequirements {
    pub time_estimate: Option<Duration>,
    pub cognitive_load: f32,       // 0.0 to 1.0
    pub emotional_energy: f32,     // 0.0 to 1.0
    pub dependencies: Vec<String>, // External dependencies
}

/// Goal conflict
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GoalConflict {
    pub goal1: GoalId,
    pub goal2: GoalId,
    pub conflict_type: ConflictType,
    pub severity: f32, // 0.0 to 1.0
    pub resolution: Option<ConflictResolution>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ConflictType {
    ResourceCompetition, // Both need same resources
    MutuallyExclusive,   // Can't achieve both
    Temporal,            // Time conflicts
    Priority,            // Priority disagreement
    Philosophical,       // Value conflicts
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConflictResolution {
    pub strategy: ResolutionStrategy,
    pub decision_id: Option<DecisionId>,
    #[serde(skip, default = "Instant::now")]
    pub timestamp: Instant,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ResolutionStrategy {
    Prioritize(GoalId),       // Choose one over other
    Sequence(GoalId, GoalId), // Do one then other
    Compromise,               // Partial achievement of both
    Synthesize(GoalId),       // Create new combined goal
    Delegate,                 // Let decision engine decide
}

/// Goal achievement history
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GoalAchievement {
    pub goal_id: GoalId,
    pub success_rate: f32,
    pub time_taken: Duration,
    pub lessons_learned: Vec<String>,
    pub emotional_impact: f32,
}

/// Aspiration levels for different goal types
#[derive(Clone, Debug)]
pub struct AspirationLevels {
    levels: HashMap<GoalType, f32>,
}

impl Default for AspirationLevels {
    fn default() -> Self {
        let mut levels = HashMap::new();
        levels.insert(GoalType::Strategic, 0.8);
        levels.insert(GoalType::Tactical, 0.7);
        levels.insert(GoalType::Operational, 0.9);
        levels.insert(GoalType::Learning, 0.6);
        levels.insert(GoalType::Maintenance, 0.8);
        levels.insert(GoalType::Social, 0.7);
        levels.insert(GoalType::Creative, 0.5);
        levels.insert(GoalType::Personal, 0.6);

        Self { levels }
    }
}

/// Configuration for goal manager
#[derive(Clone, Debug)]
pub struct GoalConfig {
    /// Maximum active goals
    pub max_active_goals: usize,

    /// Goal evaluation interval
    pub evaluation_interval: Duration,

    /// Conflict check interval
    pub conflict_check_interval: Duration,

    /// Progress update interval
    pub progress_update_interval: Duration,

    /// Enable emotional influence
    pub use_emotions: bool,

    /// Goal history size
    pub history_size: usize,
}

impl Default for GoalConfig {
    fn default() -> Self {
        Self {
            max_active_goals: 10,
            evaluation_interval: Duration::from_secs(300), // 5 minutes
            conflict_check_interval: Duration::from_secs(600), // 10 minutes
            progress_update_interval: Duration::from_secs(60), // 1 minute
            use_emotions: true,
            history_size: 1000,
        }
    }
}

#[derive(Debug, Clone)]
/// Main goal manager
pub struct GoalManager {
    /// All goals organized hierarchically
    goals: Arc<RwLock<BTreeMap<GoalId, Goal>>>,

    /// Active goals
    active_goals: Arc<RwLock<HashSet<GoalId>>>,

    /// Goal conflicts
    conflicts: Arc<RwLock<Vec<GoalConflict>>>,

    /// Goal achievements
    achievements: Arc<RwLock<Vec<GoalAchievement>>>,

    /// Aspiration levels
    aspiration_levels: Arc<RwLock<AspirationLevels>>,

    /// Decision engine reference
    decision_engine: Arc<DecisionEngine>,

    /// Emotional core reference
    emotional_core: Arc<EmotionalCore>,

    /// Neural processor reference
    neural_processor: Arc<NeuroProcessor>,

    /// Memory system reference
    memory: Arc<CognitiveMemory>,

    /// Configuration
    config: GoalConfig,

    /// Goal update channel
    goal_tx: mpsc::Sender<GoalUpdate>,

    /// Shutdown signal
    shutdown_tx: broadcast::Sender<()>,

    /// Statistics
    stats: Arc<RwLock<GoalStats>>,
}

#[derive(Clone, Debug)]
pub enum GoalUpdate {
    Created(Goal),
    Updated(Goal),
    Completed(GoalId),
    Failed(GoalId),
    ConflictDetected(GoalConflict),
}

#[derive(Debug, Default, Clone)]
pub struct GoalStats {
    pub total_goals: u64,
    pub active_goals: u64,
    pub completed_goals: u64,
    pub failed_goals: u64,
    pub avg_completion_time: Duration,
    pub success_rate: f32,
    pub conflicts_resolved: u64,
}

impl GoalManager {
    pub async fn new(
        decision_engine: Arc<DecisionEngine>,
        emotional_core: Arc<EmotionalCore>,
        neural_processor: Arc<NeuroProcessor>,
        memory: Arc<CognitiveMemory>,
        config: GoalConfig,
    ) -> Result<Self> {
        info!("Initializing Goal Manager");

        let (goal_tx, _) = mpsc::channel(100);
        let (shutdown_tx, _) = broadcast::channel(1);

        Ok(Self {
            goals: Arc::new(RwLock::new(BTreeMap::new())),
            active_goals: Arc::new(RwLock::new(HashSet::new())),
            conflicts: Arc::new(RwLock::new(Vec::new())),
            achievements: Arc::new(RwLock::new(Vec::new())),
            aspiration_levels: Arc::new(RwLock::new(AspirationLevels::default())),
            decision_engine,
            emotional_core,
            neural_processor,
            memory,
            config,
            goal_tx,
            shutdown_tx,
            stats: Arc::new(RwLock::new(GoalStats::default())),
        })
    }

    /// Start the goal manager
    pub async fn start(self: Arc<Self>) -> Result<()> {
        info!("Starting Goal Manager");

        // Goal evaluation loop
        {
            let manager = self.clone();
            tokio::spawn(async move {
                manager.evaluation_loop().await;
            });
        }

        // Conflict detection loop
        {
            let manager = self.clone();
            tokio::spawn(async move {
                manager.conflict_loop().await;
            });
        }

        // Progress tracking loop
        {
            let manager = self.clone();
            tokio::spawn(async move {
                manager.progress_loop().await;
            });
        }

        Ok(())
    }

    /// Create a new goal with narrative context awareness
    pub async fn create_goal(&self, mut goal: Goal) -> Result<GoalId> {
        let goal_id = goal.id.clone();
        info!("Creating goal with narrative context: {}", goal.name);

        // Validate goal
        self.validate_goal(&goal).await?;

        // Set timestamps
        goal.created_at = Instant::now();
        goal.last_updated = Instant::now();

        // Enhance goal with narrative intelligence context
        let narrative_context = self.generate_goal_narrative_context(&goal).await?;
        let enhanced_description =
            format!("{}\n\nNarrative Context: {}", goal.description, narrative_context);
        goal.description = enhanced_description;

        // Analyze goal for story-driven characteristics
        let story_characteristics = self.analyze_goal_story_characteristics(&goal).await?;

        // Check parent relationship
        if let Some(parent_id) = &goal.parent {
            let mut goals = self.goals.write().await;
            if let Some(parent) = goals.get_mut(parent_id) {
                parent.children.push(goal_id.clone());
            } else {
                return Err(anyhow!("Parent goal not found"));
            }
        }

        // Store goal
        self.goals.write().await.insert(goal_id.clone(), goal.clone());

        // Activate if appropriate
        if goal.state == GoalState::Active {
            self.activate_goal(&goal_id).await?;
        }

        // Send update
        let _ = self.goal_tx.send(GoalUpdate::Created(goal.clone())).await;

        // Store in memory with narrative context
        self.memory
            .store(
                format!("Created narrative-enhanced goal: {} - {}", goal.name, narrative_context),
                vec![
                    goal.description.clone(),
                    format!("Story characteristics: {:?}", story_characteristics),
                ],
                MemoryMetadata {
                    source: "narrative_goal_manager".to_string(),
                    tags: vec![
                        "goal".to_string(),
                        "creation".to_string(),
                        "narrative".to_string(),
                        format!("goal_type_{:?}", goal.goal_type).to_lowercase(),
                    ],
                    importance: 0.7 + (goal.priority.to_f32() * 0.2),
                    associations: self.create_goal_associations(&goal).await?,
                    context: Some("narrative goal creation".to_string()),
                    created_at: chrono::Utc::now(),
                    accessed_count: 0,
                    last_accessed: None,
                    version: 1,
                    timestamp: chrono::Utc::now(),
                    expiration: None,
                    category: "goal_management".to_string(),
                },
            )
            .await?;

        // Update stats
        let mut stats = self.stats.write().await;
        stats.total_goals += 1;

        Ok(goal_id)
    }

    /// Validate a goal
    async fn validate_goal(&self, goal: &Goal) -> Result<()> {
        // Check for circular dependencies
        if self.has_circular_dependency(&goal.id, &goal.dependencies).await? {
            return Err(anyhow!("Circular dependency detected"));
        }

        // Validate success criteria
        if goal.success_criteria.is_empty() {
            return Err(anyhow!("Goal must have at least one success criterion"));
        }

        // Priority is automatically valid as an enum

        Ok(())
    }

    /// Check for circular dependencies
    async fn has_circular_dependency(
        &self,
        goal_id: &GoalId,
        dependencies: &[GoalId],
    ) -> Result<bool> {
        let goals = self.goals.read().await;
        let mut visited = HashSet::new();
        let mut stack = VecDeque::new();

        for dep in dependencies {
            stack.push_back(dep.clone());
        }

        while let Some(current) = stack.pop_front() {
            if current == *goal_id {
                return Ok(true); // Circular dependency found
            }

            if visited.contains(&current) {
                continue;
            }

            visited.insert(current.clone());

            if let Some(goal) = goals.get(&current) {
                for dep in &goal.dependencies {
                    stack.push_back(dep.clone());
                }
            }
        }

        Ok(false)
    }

    /// Activate a goal
    async fn activate_goal(&self, goal_id: &GoalId) -> Result<()> {
        let mut active = self.active_goals.write().await;

        if active.len() >= self.config.max_active_goals {
            return Err(anyhow!("Maximum active goals reached"));
        }

        active.insert(goal_id.clone());

        // Update goal state
        if let Some(goal) = self.goals.write().await.get_mut(goal_id) {
            goal.state = GoalState::Active;
            goal.last_updated = Instant::now();
        }

        // Update stats
        self.stats.write().await.active_goals = active.len() as u64;

        Ok(())
    }

    /// Update goal progress
    pub async fn update_progress(&self, goal_id: &GoalId, progress: f32) -> Result<()> {
        let mut goals = self.goals.write().await;
        let goal = goals.get_mut(goal_id).ok_or_else(|| anyhow!("Goal not found"))?;

        goal.progress = progress.clamp(0.0, 1.0);
        goal.last_updated = Instant::now();

        // Check success criteria
        let mut all_met = true;
        for criterion in &mut goal.success_criteria {
            if let Some(current) = criterion.current_value {
                if let Some(target) = criterion.target_value {
                    criterion.met = current >= target;
                }
            }
            all_met &= criterion.met;
        }

        // Complete goal if all criteria met
        if all_met && goal.progress >= 1.0 {
            goal.state = GoalState::Completed;
            goal.actual_completion = Some(Instant::now());

            let achievement = GoalAchievement {
                goal_id: goal_id.clone(),
                success_rate: 1.0,
                time_taken: goal.actual_completion.unwrap().duration_since(goal.created_at),
                lessons_learned: Vec::new(),
                emotional_impact: goal.emotional_significance,
            };

            drop(goals); // Release lock

            self.complete_goal(goal_id, achievement).await?;
        }

        Ok(())
    }

    /// Complete a goal
    async fn complete_goal(&self, goal_id: &GoalId, achievement: GoalAchievement) -> Result<()> {
        info!("Goal completed: {}", goal_id);

        // Remove from active goals
        self.active_goals.write().await.remove(goal_id);

        // Store achievement
        self.achievements.write().await.push(achievement);

        // Send update
        let _ = self.goal_tx.send(GoalUpdate::Completed(goal_id.clone())).await;

        // Update stats
        let mut stats = self.stats.write().await;
        stats.completed_goals += 1;
        stats.active_goals = self.active_goals.read().await.len() as u64;

        // Update success rate
        let total = stats.completed_goals + stats.failed_goals;
        if total > 0 {
            stats.success_rate = stats.completed_goals as f32 / total as f32;
        }

        Ok(())
    }

    /// Goal evaluation loop
    async fn evaluation_loop(&self) {
        let mut shutdown_rx = self.shutdown_tx.subscribe();
        let mut interval = interval(self.config.evaluation_interval);

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    if let Err(e) = self.evaluate_goals().await {
                        debug!("Goal evaluation error: {}", e);
                    }
                }

                _ = shutdown_rx.recv() => {
                    info!("Goal evaluation loop shutting down");
                    break;
                }
            }
        }
    }

    /// Evaluate and reprioritize goals
    async fn evaluate_goals(&self) -> Result<()> {
        let active_ids: Vec<GoalId> = self.active_goals.read().await.iter().cloned().collect();
        let mut priority_updates = Vec::new();

        for goal_id in active_ids {
            if let Some(goal) = self.goals.read().await.get(&goal_id).cloned() {
                let new_priority = self.calculate_dynamic_priority(&goal).await?;

                if (new_priority - goal.priority.to_f32()).abs() > 0.1 {
                    priority_updates.push((goal_id, Priority::new(new_priority)));
                }
            }
        }

        // Apply priority updates
        let mut goals = self.goals.write().await;
        for (goal_id, new_priority) in priority_updates {
            if let Some(goal) = goals.get_mut(&goal_id) {
                goal.priority = new_priority;
                goal.last_updated = Instant::now();
            }
        }

        Ok(())
    }

    /// Calculate dynamic priority based on context
    async fn calculate_dynamic_priority(&self, goal: &Goal) -> Result<f32> {
        let mut priority = goal.priority.to_f32();

        // Time pressure factor
        if let Some(deadline) = goal.target_completion {
            let time_remaining = deadline.saturating_duration_since(Instant::now());
            let urgency = 1.0 - (time_remaining.as_secs() as f32 / 86400.0).min(1.0); // 1 day scale
            priority += urgency * 0.2;
        }

        // Progress factor - boost nearly complete goals
        if goal.progress > 0.8 {
            priority += 0.1;
        }

        // Emotional influence
        if self.config.use_emotions {
            let emotional_state = self.emotional_core.get_emotional_state().await;
            let emotional_boost =
                emotional_state.overall_valence * goal.emotional_significance * 0.1;
            priority += emotional_boost;
        }

        // Dependency factor - boost if blocking other goals
        let blocking_count = self.count_blocked_goals(&goal.id).await;
        priority += (blocking_count as f32 * 0.05).min(0.2);

        Ok(priority.clamp(0.0, 1.0))
    }

    /// Count how many goals are blocked by this goal
    async fn count_blocked_goals(&self, goal_id: &GoalId) -> usize {
        let goals = self.goals.read().await;
        goals.values().filter(|g| g.dependencies.contains(goal_id)).count()
    }

    /// Conflict detection loop
    async fn conflict_loop(&self) {
        let mut shutdown_rx = self.shutdown_tx.subscribe();
        let mut interval = interval(self.config.conflict_check_interval);

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    if let Err(e) = self.detect_conflicts().await {
                        debug!("Conflict detection error: {}", e);
                    }
                }

                _ = shutdown_rx.recv() => {
                    info!("Conflict detection loop shutting down");
                    break;
                }
            }
        }
    }

    /// Detect conflicts between goals
    async fn detect_conflicts(&self) -> Result<()> {
        let active_goals: Vec<Goal> = {
            let goals = self.goals.read().await;
            let active = self.active_goals.read().await;

            active.iter().filter_map(|id| goals.get(id).cloned()).collect()
        };

        let mut new_conflicts = Vec::new();

        // Check each pair of active goals
        for i in 0..active_goals.len() {
            for j in (i + 1)..active_goals.len() {
                let goal1 = &active_goals[i];
                let goal2 = &active_goals[j];

                if let Some(conflict) = self.check_goal_conflict(goal1, goal2).await? {
                    new_conflicts.push(conflict);
                }
            }
        }

        // Store new conflicts
        for conflict in new_conflicts {
            self.conflicts.write().await.push(conflict.clone());
            let _ = self.goal_tx.send(GoalUpdate::ConflictDetected(conflict)).await;
        }

        // Try to resolve conflicts
        self.resolve_conflicts().await?;

        Ok(())
    }

    /// Check if two goals conflict
    async fn check_goal_conflict(
        &self,
        goal1: &Goal,
        goal2: &Goal,
    ) -> Result<Option<GoalConflict>> {
        // Resource competition
        let resource_overlap =
            goal1.resources_required.cognitive_load + goal2.resources_required.cognitive_load;

        if resource_overlap > 1.2 {
            return Ok(Some(GoalConflict {
                goal1: goal1.id.clone(),
                goal2: goal2.id.clone(),
                conflict_type: ConflictType::ResourceCompetition,
                severity: (resource_overlap - 1.0).min(1.0),
                resolution: None,
            }));
        }

        // Temporal conflicts
        if let (Some(t1), Some(t2)) = (goal1.target_completion, goal2.target_completion) {
            if t1 == t2
                && goal1.resources_required.time_estimate.is_some()
                && goal2.resources_required.time_estimate.is_some()
            {
                return Ok(Some(GoalConflict {
                    goal1: goal1.id.clone(),
                    goal2: goal2.id.clone(),
                    conflict_type: ConflictType::Temporal,
                    severity: 0.6,
                    resolution: None,
                }));
            }
        }

        Ok(None)
    }

    /// Resolve detected conflicts
    async fn resolve_conflicts(&self) -> Result<()> {
        let mut conflicts = self.conflicts.write().await;
        let mut resolutions = Vec::new();

        for (i, conflict) in conflicts.iter().enumerate() {
            if conflict.resolution.is_none() {
                if let Some(resolution) = self.resolve_single_conflict(conflict).await? {
                    resolutions.push((i, resolution));
                }
            }
        }

        // Apply resolutions
        for (index, resolution) in resolutions {
            if let Some(conflict) = conflicts.get_mut(index) {
                conflict.resolution = Some(resolution);

                let mut stats = self.stats.write().await;
                stats.conflicts_resolved += 1;
            }
        }

        Ok(())
    }

    /// Resolve a single conflict
    async fn resolve_single_conflict(
        &self,
        conflict: &GoalConflict,
    ) -> Result<Option<ConflictResolution>> {
        // Get the conflicting goals
        let goal_collection = self.goals.read().await;
        let first_goal = goal_collection.get(&conflict.goal1).cloned();
        let second_goal = goal_collection.get(&conflict.goal2).cloned();
        drop(goal_collection);

        if let (Some(g1), Some(g2)) = (first_goal, second_goal) {
            // Use decision engine to resolve
            let options = vec![
                DecisionOption {
                    id: "prioritize_1".to_string(),
                    description: format!("Prioritize {}", g1.name),
                    scores: HashMap::from([
                        ("priority".to_string(), g1.priority.to_f32()),
                        ("progress".to_string(), g1.progress),
                    ]),
                    feasibility: 0.9,
                    risk_level: 0.1,
                    emotional_appeal: g1.emotional_significance,
                    expected_outcome: format!("Focus on {}", g1.name),
                    confidence: g1.priority.to_f32() * 0.8,
                    resources_required: vec!["attention".to_string()],
                    time_estimate: Duration::from_secs(300),
                    success_probability: g1.priority.to_f32(),
                },
                DecisionOption {
                    id: "prioritize_2".to_string(),
                    description: format!("Prioritize {}", g2.name),
                    scores: HashMap::from([
                        ("priority".to_string(), g2.priority.to_f32()),
                        ("progress".to_string(), g2.progress),
                    ]),
                    feasibility: 0.9,
                    risk_level: 0.1,
                    emotional_appeal: g2.emotional_significance,
                    expected_outcome: format!("Focus on {}", g2.name),
                    confidence: g2.priority.to_f32() * 0.8,
                    resources_required: vec!["attention".to_string()],
                    time_estimate: Duration::from_secs(300),
                    success_probability: g2.priority.to_f32(),
                },
                DecisionOption {
                    id: "sequence".to_string(),
                    description: "Complete goals sequentially".to_string(),
                    scores: HashMap::from([
                        (
                            "priority".to_string(),
                            (g1.priority.to_f32() + g2.priority.to_f32()) / 2.0,
                        ),
                        ("progress".to_string(), 0.5),
                    ]),
                    feasibility: 0.8,
                    risk_level: 0.2,
                    emotional_appeal: 0.0,
                    expected_outcome: "Complete goals in sequence".to_string(),
                    confidence: 0.7,
                    resources_required: vec![
                        "time_management".to_string(),
                        "attention".to_string(),
                    ],
                    time_estimate: Duration::from_secs(600),
                    success_probability: 0.6,
                },
            ];

            let criteria = vec![
                DecisionCriterion {
                    name: "priority".to_string(),
                    weight: 0.6,
                    criterion_type: CriterionType::Quantitative,
                    optimization: OptimizationType::Maximize,
                },
                DecisionCriterion {
                    name: "progress".to_string(),
                    weight: 0.4,
                    criterion_type: CriterionType::Quantitative,
                    optimization: OptimizationType::Maximize,
                },
            ];

            let decision = self
                .decision_engine
                .make_decision(
                    format!("Resolve conflict between {} and {}", g1.name, g2.name),
                    options,
                    criteria,
                )
                .await?;

            let strategy = match decision.selected.as_ref().map(|s| s.id.as_str()) {
                Some("prioritize_1") => ResolutionStrategy::Prioritize(g1.id),
                Some("prioritize_2") => ResolutionStrategy::Prioritize(g2.id),
                Some("sequence") => {
                    if g1.priority.to_f32() > g2.priority.to_f32() {
                        ResolutionStrategy::Sequence(g1.id, g2.id)
                    } else {
                        ResolutionStrategy::Sequence(g2.id, g1.id)
                    }
                }
                _ => ResolutionStrategy::Compromise,
            };

            return Ok(Some(ConflictResolution {
                strategy,
                decision_id: Some(decision.id),
                timestamp: Instant::now(),
            }));
        }

        Ok(None)
    }

    /// Progress tracking loop
    async fn progress_loop(&self) {
        let mut shutdown_rx = self.shutdown_tx.subscribe();
        let mut interval = interval(self.config.progress_update_interval);

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    if let Err(e) = self.track_progress().await {
                        debug!("Progress tracking error: {}", e);
                    }
                }

                _ = shutdown_rx.recv() => {
                    info!("Progress tracking loop shutting down");
                    break;
                }
            }
        }
    }

    /// Track progress on active goals
    async fn track_progress(&self) -> Result<()> {
        // This would integrate with task execution systems
        // For now, we'll simulate progress based on time

        let active_ids: Vec<GoalId> = self.active_goals.read().await.iter().cloned().collect();

        for goal_id in active_ids {
            // Create thought for goal progress
            let thought = Thought {
                id: ThoughtId::new(),
                content: format!("Tracking progress for goal: {}", goal_id),
                thought_type: ThoughtType::Analysis,
                ..Default::default()
            };

            self.neural_processor.process_thought(&thought).await?;
        }

        Ok(())
    }

    /// Get all goals
    pub async fn get_all_goals(&self) -> Vec<Goal> {
        self.goals.read().await.values().cloned().collect()
    }

    /// Get active goals
    pub async fn get_active_goals(&self) -> Vec<Goal> {
        let goals = self.goals.read().await;
        let active = self.active_goals.read().await;

        active.iter().filter_map(|id| goals.get(id).cloned()).collect()
    }

    /// Get goal hierarchy
    pub async fn get_goal_hierarchy(&self, root_id: Option<&GoalId>) -> Vec<Goal> {
        let goals = self.goals.read().await;
        let mut hierarchy = Vec::new();

        if let Some(root) = root_id {
            self.build_hierarchy(&goals, root, &mut hierarchy);
        } else {
            // Get all root goals (no parent)
            for goal in goals.values() {
                if goal.parent.is_none() {
                    self.build_hierarchy(&goals, &goal.id, &mut hierarchy);
                }
            }
        }

        hierarchy
    }

    /// Build goal hierarchy recursively
    fn build_hierarchy(
        &self,
        goals: &BTreeMap<GoalId, Goal>,
        current: &GoalId,
        result: &mut Vec<Goal>,
    ) {
        if let Some(goal) = goals.get(current) {
            result.push(goal.clone());

            for child_id in &goal.children {
                self.build_hierarchy(goals, child_id, result);
            }
        }
    }

    /// Generate narrative context for goal creation
    async fn generate_goal_narrative_context(&self, goal: &Goal) -> Result<String> {
        let narrative_context = match goal.goal_type {
            GoalType::Strategic => {
                format!(
                    "This goal represents a chapter in Loki's strategic evolution, shaping \
                     long-term capabilities and identity. Like a main plot thread, it will \
                     influence many other aspects of development."
                )
            }
            GoalType::Learning => {
                format!(
                    "In the pursuit of knowledge and understanding, I approach this learning goal \
                     with curiosity and systematic exploration."
                )
            }
            GoalType::Creative => {
                format!(
                    "Through creative expression and innovative thinking, I seek to bring new \
                     possibilities into existence."
                )
            }
            GoalType::Social => {
                format!(
                    "In building meaningful connections and fostering understanding, I engage \
                     with empathy and authentic communication."
                )
            }
            GoalType::Maintenance => {
                format!(
                    "This maintenance goal ensures system stability - like the steady rhythm that \
                     underlies all other activities."
                )
            }
            GoalType::Tactical => {
                format!(
                    "This tactical goal serves as a bridge between vision and action - \
                     transforming strategic intent into concrete progress."
                )
            }
            GoalType::Operational => {
                format!(
                    "This operational goal represents immediate action - the day-to-day steps \
                     that accumulate into larger achievements."
                )
            }
            GoalType::Personal => {
                format!(
                    "Through self-reflection and personal growth, I develop greater understanding \
                     and capability."
                )
            }
            GoalType::Problem => {
                format!(
                    "Facing this challenge with analytical clarity and persistent determination, \
                     I work toward effective solutions."
                )
            }
            GoalType::Achievement => {
                format!(
                    "This achievement goal marks a significant milestone in the journey, \
                     representing tangible progress and demonstrable capability."
                )
            }
        };

        // Add temporal context based on target completion
        let temporal_context = if let Some(deadline) = goal.target_completion {
            let time_remaining = deadline.saturating_duration_since(goal.created_at);
            match time_remaining.as_secs() {
                secs if secs < 3600 => "urgent sprint toward completion",
                secs if secs < 86_400 => "focused daily pursuit",
                secs if secs < 604_800 => "sustained weekly effort",
                secs if secs < 2_592_000 => "patient monthly development",
                _ => "long-term evolutionary process",
            }
        } else {
            "open-ended journey of growth"
        };

        Ok(format!("{} The timeline suggests this is a {}.", narrative_context, temporal_context))
    }

    /// Analyze goal for story-driven characteristics
    async fn analyze_goal_story_characteristics(
        &self,
        goal: &Goal,
    ) -> Result<GoalStoryCharacteristics> {
        // Determine story type based on goal characteristics
        let story_type = if goal.priority.to_f32() > 0.8 {
            "heroic_quest"
        } else if goal.goal_type == GoalType::Learning {
            "coming_of_age"
        } else if goal.goal_type == GoalType::Creative {
            "artistic_journey"
        } else if goal.dependencies.len() > 2 {
            "ensemble_collaboration"
        } else if goal.children.len() > 3 {
            "epic_saga"
        } else {
            "personal_growth"
        }
        .to_string();

        // Determine character role in this goal's story
        let character_role = match goal.goal_type {
            GoalType::Strategic => "visionary_leader",
            GoalType::Learning => "eager_student",
            GoalType::Creative => "inspired_artist",
            GoalType::Social => "relationship_builder",
            GoalType::Maintenance => "reliable_guardian",
            GoalType::Tactical => "skilled_strategist",
            GoalType::Operational => "efficient_executor",
            GoalType::Personal => "self_developer",
            GoalType::Problem => "analytical_problem_solver",
            GoalType::Achievement => "accomplished_achiever",
        }
        .to_string();

        // Predict narrative arc progression
        let expected_challenges = if goal.resources_required.cognitive_load > 0.7 {
            vec!["intellectual_complexity".to_string(), "mental_fatigue".to_string()]
        } else if goal.resources_required.emotional_energy > 0.6 {
            vec!["emotional_resistance".to_string(), "motivation_fluctuation".to_string()]
        } else if goal.dependencies.len() > 1 {
            vec!["coordination_difficulty".to_string(), "dependency_delays".to_string()]
        } else {
            vec!["routine_obstacles".to_string()]
        };

        // Determine potential story conclusion types
        let potential_endings = if goal.priority.to_f32() > 0.8 {
            vec!["triumphant_success".to_string(), "valuable_lesson".to_string()]
        } else {
            vec!["quiet_accomplishment".to_string(), "incremental_progress".to_string()]
        };

        Ok(GoalStoryCharacteristics {
            story_type,
            character_role,
            expected_challenges,
            potential_endings,
            narrative_complexity: (goal.resources_required.cognitive_load
                * goal.resources_required.emotional_energy
                * goal.priority.to_f32())
            .min(1.0),
        })
    }

    /// Enhanced goal progress tracking with narrative awareness
    #[allow(dead_code)]
    async fn track_progress_with_narrative_context(&self) -> Result<()> {
        let active_ids: Vec<GoalId> = self.active_goals.read().await.iter().cloned().collect();

        for goal_id in active_ids {
            if let Some(goal) = self.goals.read().await.get(&goal_id).cloned() {
                // Generate narrative progress update
                let progress_narrative = self.generate_progress_narrative(&goal).await?;

                // Create thought for goal progress with narrative context
                let thought = Thought {
                    id: ThoughtId::new(),
                    content: format!(
                        "Goal progress narrative: {} - {}",
                        goal.name, progress_narrative
                    ),
                    thought_type: ThoughtType::Analysis,
                    ..Default::default()
                };

                self.neural_processor.process_thought(&thought).await?;

                // Store narrative progress in memory
                self.memory
                    .store(
                        format!("Progress update: {} - {}", goal.name, progress_narrative),
                        vec![format!("Progress: {:.1}%", goal.progress * 100.0)],
                        MemoryMetadata {
                            source: "narrative_goal_progress".to_string(),
                            tags: vec![
                                "goal_progress".to_string(),
                                "narrative".to_string(),
                                format!(
                                    "goal_{}",
                                    goal_id.to_string().chars().take(8).collect::<String>()
                                ),
                            ],
                            importance: 0.5 + (goal.progress * 0.3),
                            associations: vec![],
                            context: Some("narrative goal progress tracking".to_string()),
                            created_at: chrono::Utc::now(),
                            accessed_count: 0,
                            last_accessed: None,
                            version: 1,
                            timestamp: chrono::Utc::now(),
                            expiration: None,
                            category: "goal_progress".to_string(),
                        },
                    )
                    .await?;
            }
        }

        Ok(())
    }

    /// Generate narrative description of goal progress
    #[allow(dead_code)]
    async fn generate_progress_narrative(&self, goal: &Goal) -> Result<String> {
        let progress_stage = match goal.progress {
            p if p < 0.1 => "just beginning the journey",
            p if p < 0.3 => "gaining momentum and early insights",
            p if p < 0.5 => "navigating through the challenging middle phase",
            p if p < 0.7 => "seeing clear progress toward the goal",
            p if p < 0.9 => "approaching the final challenges before completion",
            _ => "reaching the culmination of the effort",
        };

        let time_context = if let Some(target) = goal.target_completion {
            let elapsed = goal.created_at.elapsed();
            let total_planned = target.duration_since(goal.created_at);
            let time_progress = elapsed.as_secs() as f32 / total_planned.as_secs() as f32;

            if time_progress > goal.progress + 0.2 {
                "behind the intended timeline but making steady progress"
            } else if time_progress < goal.progress - 0.1 {
                "ahead of schedule with excellent momentum"
            } else {
                "progressing at the expected pace"
            }
        } else {
            "moving forward at its natural rhythm"
        };

        Ok(format!("{}, {}", progress_stage, time_context))
    }

    /// Get goal statistics
    pub async fn get_stats(&self) -> GoalStats {
        self.stats.read().await.clone()
    }

    /// Get a specific goal by ID
    pub async fn get_goal(&self, goal_id: &GoalId) -> Option<Goal> {
        self.goals.read().await.get(goal_id).cloned()
    }

    /// Create intelligent associations for a goal based on its properties
    async fn create_goal_associations(&self, goal: &Goal) -> Result<Vec<MemoryId>> {
        let mut associations = Vec::new();

        // 1. Find related goals by type and dependencies
        let related_goals = self.find_related_goals(goal).await?;
        for related_memory_id in related_goals {
            associations.push(related_memory_id);
        }

        // 2. Search for memories related to goal keywords
        let keywords = self.extract_goal_keywords(goal);
        for keyword in keywords {
            if let Ok(related_memories) = self.memory.search(&keyword).await {
                for memory_item in related_memories.into_iter().take(3) {
                    // Limit associations
                    associations.push(memory_item.id);
                }
            }
        }

        // 3. Associate with similar emotional significance memories
        if goal.emotional_significance.abs() > 0.1 {
            let emotional_query = if goal.emotional_significance > 0.0 {
                "positive achievement satisfaction accomplishment"
            } else {
                "challenge difficulty obstacle problem"
            };

            if let Ok(emotional_memories) = self.memory.search(emotional_query).await {
                for memory_item in emotional_memories.into_iter().take(2) {
                    associations.push(memory_item.id);
                }
            }
        }

        // Remove duplicates and limit total associations
        associations.sort();
        associations.dedup();
        Ok(associations.into_iter().take(10).collect()) // Max 10 associations
    }

    /// Find memory IDs of related goals
    async fn find_related_goals(&self, goal: &Goal) -> Result<Vec<MemoryId>> {
        let mut related_memory_ids = Vec::new();

        // Search for goals of the same type
        let type_query = format!("goal goal_type_{:?}", goal.goal_type).to_lowercase();
        if let Ok(type_memories) = self.memory.search(&type_query).await {
            for memory_item in type_memories.into_iter().take(3) {
                related_memory_ids.push(memory_item.id);
            }
        }

        // Search for dependency-related goals
        for dependency in &goal.dependencies {
            let dep_query = format!("goal {}", dependency);
            if let Ok(dep_memories) = self.memory.search(&dep_query).await {
                for memory_item in dep_memories.into_iter().take(2) {
                    related_memory_ids.push(memory_item.id);
                }
            }
        }

        Ok(related_memory_ids)
    }

    /// Extract meaningful keywords from goal for association
    fn extract_goal_keywords(&self, goal: &Goal) -> Vec<String> {
        let mut keywords = Vec::new();

        // Extract from name (skip common words)
        let name_words: Vec<&str> = goal
            .name
            .split_whitespace()
            .filter(|word| word.len() > 3 && !is_common_word(word))
            .collect();
        keywords.extend(name_words.iter().map(|s| s.to_string()));

        // Extract from description
        let desc_words: Vec<&str> = goal
            .description
            .split_whitespace()
            .filter(|word| word.len() > 4 && !is_common_word(word))
            .take(5) // Limit to 5 most relevant words
            .collect();
        keywords.extend(desc_words.iter().map(|s| s.to_string()));

        // Add goal type as keyword
        keywords.push(format!("{:?}", goal.goal_type).to_lowercase());

        keywords
    }
}

/// Story characteristics for narrative-driven goal management
#[derive(Debug, Clone)]
struct GoalStoryCharacteristics {
    story_type: String,
    character_role: String,
    expected_challenges: Vec<String>,
    potential_endings: Vec<String>,
    narrative_complexity: f32,
}

/// Check if a word is a common word that shouldn't be used for associations
fn is_common_word(word: &str) -> bool {
    const COMMON_WORDS: &[&str] = &[
        "the", "and", "for", "are", "but", "not", "you", "all", "can", "had", "was", "one", "our",
        "out", "day", "get", "has", "him", "his", "how", "man", "new", "now", "old", "see", "two",
        "way", "who", "boy", "did", "its", "let", "put", "say", "she", "too", "use",
    ];
    COMMON_WORDS.contains(&word.to_lowercase().as_str())
}

/// Goal alignment assessment for consciousness system integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoalAlignment {
    /// Overall alignment score (0.0 to 1.0)
    pub alignment_score: f32,
    
    /// Areas where alignment is lacking
    pub misalignment_areas: Vec<String>,
    
    /// Specific goals that are well-aligned
    pub aligned_goals: Vec<GoalId>,
    
    /// Goals that are misaligned
    pub misaligned_goals: Vec<GoalId>,
    
    /// Recommendations for improvement
    pub recommendations: Vec<String>,
    
    /// Confidence in the alignment assessment
    pub confidence: f32,
}

impl Default for GoalAlignment {
    fn default() -> Self {
        Self {
            alignment_score: 0.5,
            misalignment_areas: Vec::new(),
            aligned_goals: Vec::new(),
            misaligned_goals: Vec::new(),
            recommendations: Vec::new(),
            confidence: 0.8,
        }
    }
}

impl GoalManager {
    /// Check alignment of a response with current goals
    pub async fn check_alignment(&self, response: &str) -> Result<GoalAlignment> {
        debug!("🎯 Checking goal alignment for response");
        
        let active_goals = self.get_active_goals().await;
        let mut aligned_goals = Vec::new();
        let mut misaligned_goals = Vec::new();
        let mut misalignment_areas = Vec::new();
        let mut total_alignment = 0.0;
        
        // Check each active goal for alignment
        for goal in active_goals {
            let alignment_score = self.calculate_goal_response_alignment(&goal, response).await?;
            
            if alignment_score > 0.6 {
                aligned_goals.push(goal.id.clone());
            } else {
                misaligned_goals.push(goal.id.clone());
                misalignment_areas.push(format!("Goal '{}' not addressed", goal.description));
            }
            
            total_alignment += alignment_score;
        }
        
        // Calculate overall alignment
        let goal_count = aligned_goals.len() + misaligned_goals.len();
        let alignment_score = if goal_count > 0 {
            total_alignment / goal_count as f32
        } else {
            0.8 // Default if no active goals
        };
        
        // Generate recommendations
        let mut recommendations = Vec::new();
        if alignment_score < 0.6 {
            recommendations.push("Consider referencing active goals in response".to_string());
        }
        if !misaligned_goals.is_empty() {
            recommendations.push("Address unmet goal requirements".to_string());
        }
        
        Ok(GoalAlignment {
            alignment_score,
            misalignment_areas,
            aligned_goals,
            misaligned_goals,
            recommendations,
            confidence: 0.8,
        })
    }
    
    /// Calculate alignment between a specific goal and response
    async fn calculate_goal_response_alignment(&self, goal: &Goal, response: &str) -> Result<f32> {
        // Simple keyword-based alignment (in production, this would use NLP)
        let goal_keywords: Vec<&str> = goal.description.split_whitespace().collect();
        let response_keywords: Vec<&str> = response.split_whitespace().collect();
        
        let mut matches = 0;
        for goal_word in &goal_keywords {
            if goal_word.len() > 3 { // Skip short words
                for response_word in &response_keywords {
                    if goal_word.to_lowercase() == response_word.to_lowercase() {
                        matches += 1;
                        break;
                    }
                }
            }
        }
        
        let alignment = if goal_keywords.len() > 0 {
            matches as f32 / goal_keywords.len() as f32
        } else {
            0.5
        };
        
        Ok(alignment.min(1.0))
    }
}
