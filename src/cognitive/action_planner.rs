//! Action Planning System
//!
//! This module implements STRIPS-like action planning with parallel
//! coordination, resource allocation, and temporal constraints.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use tokio::sync::{RwLock, mpsc};
use tracing::{info, warn};

use crate::cognitive::{
    CriterionType,
    DecisionCriterion,
    DecisionEngine,
    DecisionOption,
    DecisionOptimizationType as OptimizationType,
    Goal,
    GoalId,
    GoalManager,
    NeuroProcessor,
    SuccessCriterion,
};
use crate::memory::{CognitiveMemory, MemoryMetadata};

/// Unique identifier for actions
#[derive(Clone, Debug, Hash, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub struct ActionId(String);

impl ActionId {
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4().to_string())
    }
}

impl std::fmt::Display for ActionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// State representation for planning
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StateVariable {
    pub name: String,
    pub value: StateValue,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StateValue {
    Boolean(bool),
    Integer(i64),
    String(String),
    Enum(String, String), // (type, value)
}

/// Action representation (STRIPS-like)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Action {
    pub id: ActionId,
    pub name: String,
    pub description: String,
    pub preconditions: Vec<Condition>,
    pub effects: Vec<Effect>,
    pub duration: Duration,
    pub resources: ResourceRequirements,
    pub cost: f32,
    pub reliability: f32, // 0.0 to 1.0
    pub parallelizable: bool,
}

/// Condition for preconditions
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Condition {
    pub variable: String,
    pub operator: ConditionOperator,
    pub value: StateValue,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ConditionOperator {
    Equals,
    NotEquals,
    GreaterThan,
    LessThan,
    GreaterOrEqual,
    LessOrEqual,
    Contains,
    NotContains,
}

/// Effect of an action
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Effect {
    pub variable: String,
    pub operation: EffectOperation,
    pub value: StateValue,
    pub probability: f32, // Probabilistic effects
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum EffectOperation {
    Assign,
    Increment,
    Decrement,
    Append,
    Remove,
}

/// Resource requirements for actions
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ResourceRequirements {
    pub cognitive_load: f32,              // 0.0 to 1.0
    pub memory_usage: f32,                // 0.0 to 1.0
    pub energy: f32,                      // 0.0 to 1.0
    pub exclusive_resources: Vec<String>, // Named resources that can't be shared
}

/// A plan consisting of ordered actions
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Plan {
    pub id: String,
    pub goal_id: GoalId,
    pub steps: Vec<PlanStep>,
    pub total_duration: Duration,
    pub total_cost: f32,
    pub success_probability: f32,
    pub parallel_tracks: Vec<ParallelTrack>,
}

/// Individual step in a plan
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlanStep {
    pub step_id: usize,
    pub action: Action,
    pub start_time: Duration,     // Relative to plan start
    pub dependencies: Vec<usize>, // Step IDs this depends on
    pub track_id: Option<usize>,  // Parallel track assignment
}

/// Parallel execution track
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ParallelTrack {
    pub track_id: usize,
    pub resource_usage: ResourceRequirements,
    pub steps: Vec<usize>, // Step IDs in this track
}

/// Plan execution state
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum PlanState {
    Planning,
    Ready,
    Executing,
    Paused,
    Completed,
    Failed,
    Repairing,
}

/// Plan execution context
#[derive(Clone, Debug)]
pub struct ExecutionContext {
    pub plan: Plan,
    pub state: PlanState,
    pub current_steps: HashSet<usize>,
    pub completed_steps: HashSet<usize>,
    pub failed_steps: HashSet<usize>,
    pub world_state: HashMap<String, StateValue>,
    pub start_time: Option<Instant>,
}

#[derive(Debug)]
/// Action repository for available actions
pub struct ActionRepository {
    actions: Arc<RwLock<HashMap<ActionId, Action>>>,
    action_index: Arc<RwLock<HashMap<String, Vec<ActionId>>>>, // Effect variable -> Actions
}

impl ActionRepository {
    pub fn new() -> Self {
        Self {
            actions: Arc::new(RwLock::new(HashMap::new())),
            action_index: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a new action
    pub async fn register_action(&self, action: Action) -> Result<()> {
        let action_id = action.id.clone();

        // Index by effects
        let mut index = self.action_index.write().await;
        for effect in &action.effects {
            index.entry(effect.variable.clone()).or_insert_with(Vec::new).push(action_id.clone());
        }

        // Store action
        self.actions.write().await.insert(action_id, action);

        Ok(())
    }

    /// Find actions that can achieve a specific effect
    pub async fn find_actions_for_effect(&self, variable: &str, value: &StateValue) -> Vec<Action> {
        let index = self.action_index.read().await;
        let actions = self.actions.read().await;

        if let Some(action_ids) = index.get(variable) {
            action_ids
                .iter()
                .filter_map(|id| actions.get(id))
                .filter(|action| {
                    action.effects.iter().any(|eff| {
                        eff.variable == variable
                            && Self::effect_achieves_value(&eff.operation, &eff.value, value)
                    })
                })
                .cloned()
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Check if an effect operation achieves the desired value
    fn effect_achieves_value(
        op: &EffectOperation,
        effect_val: &StateValue,
        target_val: &StateValue,
    ) -> bool {
        match op {
            EffectOperation::Assign => effect_val == target_val,
            _ => true, // Simplified for now
        }
    }
}

#[derive(Debug, Clone)]
/// Main action planner
pub struct ActionPlanner {
    /// Action repository
    repository: Arc<ActionRepository>,

    /// Goal manager reference
    #[allow(dead_code)] // Infrastructure component for goal-oriented planning
    goal_manager: Arc<GoalManager>,

    /// Decision engine reference
    decision_engine: Arc<DecisionEngine>,

    /// Neural processor reference
    #[allow(dead_code)] // Infrastructure component for neural computation
    neural_processor: Arc<NeuroProcessor>,

    /// Memory system reference
    memory: Arc<CognitiveMemory>,

    /// Active plans
    active_plans: Arc<RwLock<HashMap<String, ExecutionContext>>>,

    /// Plan execution channel
    execution_tx: mpsc::Sender<PlanUpdate>,

    /// Configuration
    config: PlannerConfig,

    /// Statistics
    stats: Arc<RwLock<PlannerStats>>,
}

#[derive(Clone, Debug)]
pub struct PlannerConfig {
    /// Maximum planning time
    pub max_planning_time: Duration,

    /// Maximum plan depth
    pub max_plan_depth: usize,

    /// Enable parallel planning
    pub enable_parallel: bool,

    /// Resource oversubscription factor
    pub resource_oversubscription: f32,

    /// Plan repair threshold
    pub repair_threshold: f32,
}

impl Default for PlannerConfig {
    fn default() -> Self {
        Self {
            max_planning_time: Duration::from_secs(10),
            max_plan_depth: 20,
            enable_parallel: true,
            resource_oversubscription: 1.2,
            repair_threshold: 0.3,
        }
    }
}

#[derive(Clone, Debug)]
pub enum PlanUpdate {
    PlanCreated(Plan),
    StepStarted(String, usize), // plan_id, step_id
    StepCompleted(String, usize),
    StepFailed(String, usize, String), // plan_id, step_id, reason
    PlanCompleted(String),
    PlanFailed(String, String), // plan_id, reason
}

#[derive(Debug, Default, Clone)]
pub struct PlannerStats {
    pub plans_created: u64,
    pub plans_executed: u64,
    pub plans_succeeded: u64,
    pub plans_failed: u64,
    pub repairs_attempted: u64,
    pub repairs_succeeded: u64,
    pub avg_plan_length: f32,
    pub avg_planning_time_ms: f64,
}

impl ActionPlanner {
    pub async fn new(
        repository: Arc<ActionRepository>,
        goal_manager: Arc<GoalManager>,
        decision_engine: Arc<DecisionEngine>,
        neural_processor: Arc<NeuroProcessor>,
        memory: Arc<CognitiveMemory>,
        config: PlannerConfig,
    ) -> Result<Self> {
        info!("Initializing Action Planner");

        let (execution_tx, _) = mpsc::channel(100);

        Ok(Self {
            repository,
            goal_manager,
            decision_engine,
            neural_processor,
            memory,
            active_plans: Arc::new(RwLock::new(HashMap::new())),
            execution_tx,
            config,
            stats: Arc::new(RwLock::new(PlannerStats::default())),
        })
    }

    /// Create a plan to achieve a goal
    pub async fn create_plan(&self, goal: &Goal) -> Result<Plan> {
        let start = Instant::now();
        info!("Creating plan for goal: {}", goal.name);

        // Extract goal conditions from success criteria
        let goal_conditions = self.extract_goal_conditions(&goal.success_criteria);

        // Get current world state
        let current_state = self.get_current_state().await?;

        // Use backward chaining to find actions
        let plan_steps = self
            .backward_chain(&current_state, &goal_conditions, self.config.max_plan_depth)
            .await?;

        // Optimize plan for parallel execution
        let (ordered_steps, parallel_tracks) = if self.config.enable_parallel {
            self.optimize_parallel_execution(&plan_steps).await?
        } else {
            (self.create_sequential_plan(&plan_steps), Vec::new())
        };

        // Calculate plan metrics
        let total_duration = self.calculate_total_duration(&ordered_steps);
        let total_cost = ordered_steps.iter().map(|s| s.action.cost).sum();
        let success_probability = self.calculate_success_probability(&ordered_steps);

        let plan = Plan {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            steps: ordered_steps,
            total_duration,
            total_cost,
            success_probability,
            parallel_tracks,
        };

        // Update statistics
        let planning_time = start.elapsed();
        self.update_stats(&plan, planning_time).await;

        // Store plan in memory
        self.memory
            .store(
                format!("Created plan for goal: {}", goal.name),
                vec![format!("{} steps, duration: {:?}", plan.steps.len(), total_duration)],
                MemoryMetadata {
                    source: "action_planner".to_string(),
                    tags: vec!["plan".to_string(), "goal".to_string()],
                    importance: 0.7,
                    associations: vec![],
                    context: Some("action plan creation".to_string()),
                    created_at: chrono::Utc::now(),
                    accessed_count: 0,
                    last_accessed: None,
                    version: 1,
                    category: "cognitive".to_string(),
                    timestamp: chrono::Utc::now(),
                    expiration: None,
                },
            )
            .await?;

        // Send update
        let _ = self.execution_tx.send(PlanUpdate::PlanCreated(plan.clone())).await;

        Ok(plan)
    }

    /// Extract goal conditions from success criteria
    fn extract_goal_conditions(&self, criteria: &[SuccessCriterion]) -> Vec<Condition> {
        criteria
            .iter()
            .filter_map(|criterion| {
                if let (Some(target), true) = (criterion.target_value, criterion.measurable) {
                    Some(Condition {
                        variable: criterion.description.clone(),
                        operator: ConditionOperator::GreaterOrEqual,
                        value: StateValue::Integer(target as i64),
                    })
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get current world state
    async fn get_current_state(&self) -> Result<HashMap<String, StateValue>> {
        // This would integrate with sensors and memory
        // For now, return a simplified state
        let mut state = HashMap::new();

        // Add some default state variables
        state.insert("energy_level".to_string(), StateValue::Integer(80));
        state.insert("knowledge_base".to_string(), StateValue::Integer(50));
        state.insert("active_tasks".to_string(), StateValue::Integer(0));

        Ok(state)
    }

    /// Backward chaining planning algorithm
    async fn backward_chain(
        &self,
        current_state: &HashMap<String, StateValue>,
        goals: &[Condition],
        max_depth: usize,
    ) -> Result<Vec<Action>> {
        let mut plan = Vec::new();
        let mut open_goals = VecDeque::from(goals.to_vec());
        let mut achieved_effects = HashSet::new();
        let mut depth = 0;

        while !open_goals.is_empty() && depth < max_depth {
            let goal = match open_goals.pop_front() {
                Some(goal) => goal,
                None => {
                    // No more open goals to process
                    break;
                }
            };

            // Check if goal is already satisfied
            if self.condition_satisfied(&goal, current_state) {
                continue;
            }

            // Find actions that can achieve this goal
            let candidate_actions =
                self.repository.find_actions_for_effect(&goal.variable, &goal.value).await;

            if candidate_actions.is_empty() {
                return Err(anyhow!("No actions found to achieve: {}", goal.variable));
            }

            // Select best action using decision engine
            let selected_action = self.select_best_action(&candidate_actions, &goal).await?;

            // Add preconditions as new goals
            for precondition in &selected_action.preconditions {
                if !self.condition_satisfied(precondition, current_state)
                    && !achieved_effects.contains(&precondition.variable)
                {
                    open_goals.push_back(precondition.clone());
                }
            }

            // Track achieved effects
            for effect in &selected_action.effects {
                achieved_effects.insert(effect.variable.clone());
            }

            plan.push(selected_action);
            depth += 1;
        }

        // Reverse plan since we built it backward
        plan.reverse();
        Ok(plan)
    }

    /// Check if a condition is satisfied in the current state
    fn condition_satisfied(
        &self,
        condition: &Condition,
        state: &HashMap<String, StateValue>,
    ) -> bool {
        if let Some(value) = state.get(&condition.variable) {
            match (&condition.operator, value, &condition.value) {
                (ConditionOperator::Equals, v1, v2) => v1 == v2,
                (ConditionOperator::NotEquals, v1, v2) => v1 != v2,
                (
                    ConditionOperator::GreaterThan,
                    StateValue::Integer(v1),
                    StateValue::Integer(v2),
                ) => v1 > v2,
                (ConditionOperator::LessThan, StateValue::Integer(v1), StateValue::Integer(v2)) => {
                    v1 < v2
                }
                _ => false,
            }
        } else {
            false
        }
    }

    /// Select best action using decision engine
    async fn select_best_action(&self, actions: &[Action], goal: &Condition) -> Result<Action> {
        if actions.len() == 1 {
            return Ok(actions[0].clone());
        }

        // Convert actions to decision options
        let options: Vec<DecisionOption> = actions
            .iter()
            .map(|action| DecisionOption {
                id: action.id.to_string(),
                description: action.description.clone(),
                scores: HashMap::from([
                    ("cost".to_string(), 1.0 - action.cost),
                    ("reliability".to_string(), action.reliability),
                    (
                        "duration".to_string(),
                        1.0 - (action.duration.as_secs() as f32 / 3600.0).min(1.0),
                    ),
                ]),
                feasibility: action.reliability,
                risk_level: 1.0 - action.reliability,
                emotional_appeal: 0.0,
                confidence: action.reliability,
                expected_outcome: format!("Execute action: {}", action.description),
                resources_required: vec!["cognitive_processing".to_string()],
                time_estimate: action.duration,
                success_probability: action.reliability,
            })
            .collect();

        let criteria = vec![
            DecisionCriterion {
                name: "cost".to_string(),
                weight: 0.3,
                criterion_type: CriterionType::Quantitative,
                optimization: OptimizationType::Maximize,
            },
            DecisionCriterion {
                name: "reliability".to_string(),
                weight: 0.5,
                criterion_type: CriterionType::Quantitative,
                optimization: OptimizationType::Maximize,
            },
            DecisionCriterion {
                name: "duration".to_string(),
                weight: 0.2,
                criterion_type: CriterionType::Quantitative,
                optimization: OptimizationType::Maximize,
            },
        ];

        let decision = self
            .decision_engine
            .make_decision(
                format!("Select action to achieve: {}", goal.variable),
                options,
                criteria,
            )
            .await?;

        // Find the selected action
        let selected_id = decision.selected.ok_or_else(|| anyhow!("No action selected"))?.id;

        actions
            .iter()
            .find(|a| a.id.to_string() == selected_id)
            .cloned()
            .ok_or_else(|| anyhow!("Selected action not found"))
    }

    /// Optimize plan for parallel execution
    async fn optimize_parallel_execution(
        &self,
        actions: &[Action],
    ) -> Result<(Vec<PlanStep>, Vec<ParallelTrack>)> {
        let mut steps = Vec::new();
        let mut tracks = Vec::new();

        // Build dependency graph
        let dependencies = self.build_dependency_graph(actions);

        // Assign actions to parallel tracks
        let mut track_resources: Vec<ResourceRequirements> = Vec::new();
        let mut action_tracks: HashMap<usize, usize> = HashMap::new();

        for (i, action) in actions.iter().enumerate() {
            // Find a track that can accommodate this action
            let mut assigned_track = None;

            for (track_id, track_res) in track_resources.iter_mut().enumerate() {
                if self.can_merge_resources(track_res, &action.resources) {
                    self.merge_resources(track_res, &action.resources);
                    assigned_track = Some(track_id);
                    break;
                }
            }

            // Create new track if needed
            if assigned_track.is_none() && action.parallelizable {
                track_resources.push(action.resources.clone());
                assigned_track = Some(track_resources.len() - 1);
            }

            if let Some(track_id) = assigned_track {
                action_tracks.insert(i, track_id);
            }

            // Create plan step
            let step = PlanStep {
                step_id: i,
                action: action.clone(),
                start_time: Duration::from_secs(0), // Will be calculated
                dependencies: dependencies.get(&i).cloned().unwrap_or_default(),
                track_id: assigned_track,
            };

            steps.push(step);
        }

        // Calculate start times
        self.calculate_start_times(&mut steps);

        // Create parallel tracks
        for (track_id, resources) in track_resources.into_iter().enumerate() {
            let track_steps: Vec<usize> = action_tracks
                .iter()
                .filter_map(
                    |(step_id, track)| {
                        if *track == track_id { Some(*step_id) } else { None }
                    },
                )
                .collect();

            tracks.push(ParallelTrack { track_id, resource_usage: resources, steps: track_steps });
        }

        Ok((steps, tracks))
    }

    /// Build dependency graph for actions
    fn build_dependency_graph(&self, actions: &[Action]) -> HashMap<usize, Vec<usize>> {
        let mut dependencies = HashMap::new();

        for (i, action) in actions.iter().enumerate() {
            let mut deps = Vec::new();

            // Check which earlier actions this depends on
            for (j, earlier) in actions.iter().enumerate().take(i) {
                // Check if any precondition of current action depends on effects of earlier
                // action
                for precondition in &action.preconditions {
                    for effect in &earlier.effects {
                        if precondition.variable == effect.variable {
                            deps.push(j);
                            break;
                        }
                    }
                }
            }

            dependencies.insert(i, deps);
        }

        dependencies
    }

    /// Check if resources can be merged
    fn can_merge_resources(&self, r1: &ResourceRequirements, r2: &ResourceRequirements) -> bool {
        let total_cognitive = r1.cognitive_load + r2.cognitive_load;
        let total_memory = r1.memory_usage + r2.memory_usage;
        let total_energy = r1.energy + r2.energy;

        // Check if any exclusive resources conflict
        let exclusive_conflict =
            r1.exclusive_resources.iter().any(|res| r2.exclusive_resources.contains(res));

        total_cognitive <= self.config.resource_oversubscription
            && total_memory <= self.config.resource_oversubscription
            && total_energy <= self.config.resource_oversubscription
            && !exclusive_conflict
    }

    /// Merge resources
    fn merge_resources(&self, r1: &mut ResourceRequirements, r2: &ResourceRequirements) {
        r1.cognitive_load += r2.cognitive_load;
        r1.memory_usage += r2.memory_usage;
        r1.energy += r2.energy;
        r1.exclusive_resources.extend(r2.exclusive_resources.clone());
    }

    /// Create sequential plan
    fn create_sequential_plan(&self, actions: &[Action]) -> Vec<PlanStep> {
        let mut steps = Vec::new();
        let mut current_time = Duration::from_secs(0);

        for (i, action) in actions.iter().enumerate() {
            steps.push(PlanStep {
                step_id: i,
                action: action.clone(),
                start_time: current_time,
                dependencies: if i > 0 { vec![i - 1] } else { vec![] },
                track_id: None,
            });

            current_time += action.duration;
        }

        steps
    }

    /// Calculate start times for plan steps
    fn calculate_start_times(&self, steps: &mut [PlanStep]) {
        let mut step_end_times: HashMap<usize, Duration> = HashMap::new();

        for step in steps.iter_mut() {
            // Find latest end time of dependencies
            let mut start_time = Duration::from_secs(0);

            for &dep_id in &step.dependencies {
                if let Some(&dep_end) = step_end_times.get(&dep_id) {
                    start_time = start_time.max(dep_end);
                }
            }

            step.start_time = start_time;
            step_end_times.insert(step.step_id, start_time + step.action.duration);
        }
    }

    /// Calculate total duration of plan
    fn calculate_total_duration(&self, steps: &[PlanStep]) -> Duration {
        steps
            .iter()
            .map(|s| s.start_time + s.action.duration)
            .max()
            .unwrap_or_else(|| Duration::from_secs(0))
    }

    /// Calculate success probability of plan
    fn calculate_success_probability(&self, steps: &[PlanStep]) -> f32 {
        steps.iter().map(|s| s.action.reliability).product()
    }

    /// Update planner statistics
    async fn update_stats(&self, plan: &Plan, planning_time: Duration) {
        let mut stats = self.stats.write().await;
        stats.plans_created += 1;

        let n = stats.plans_created as f32;
        stats.avg_plan_length = (stats.avg_plan_length * (n - 1.0) + plan.steps.len() as f32) / n;

        let planning_ms = planning_time.as_millis() as f64;
        stats.avg_planning_time_ms =
            (stats.avg_planning_time_ms * (n as f64 - 1.0) + planning_ms) / n as f64;
    }

    /// Execute a plan
    pub async fn execute_plan(&self, plan: Plan) -> Result<()> {
        info!("Executing plan: {}", plan.id);

        let context = ExecutionContext {
            plan: plan.clone(),
            state: PlanState::Executing,
            current_steps: HashSet::new(),
            completed_steps: HashSet::new(),
            failed_steps: HashSet::new(),
            world_state: self.get_current_state().await?,
            start_time: Some(Instant::now()),
        };

        self.active_plans.write().await.insert(plan.id.clone(), context);

        // This would spawn execution tasks
        // For now, we'll just mark it as stored

        Ok(())
    }

    /// Repair a failed plan
    pub async fn repair_plan(&self, plan_id: &str, failed_step: usize) -> Result<Plan> {
        let plans = self.active_plans.read().await;
        let context = plans.get(plan_id).ok_or_else(|| anyhow!("Plan not found"))?;

        info!("Attempting to repair plan {} at step {}", plan_id, failed_step);

        // Extract remaining goals
        let remaining_effects: Vec<Condition> = context
            .plan
            .steps
            .iter()
            .skip(failed_step)
            .flat_map(|step| {
                step.action.effects.iter().map(|eff| Condition {
                    variable: eff.variable.clone(),
                    operator: ConditionOperator::Equals,
                    value: eff.value.clone(),
                })
            })
            .collect();

        // Create new plan from current state
        let repair_steps = self
            .backward_chain(&context.world_state, &remaining_effects, self.config.max_plan_depth)
            .await?;

        // Merge with completed steps
        let mut new_steps = Vec::new();

        // Add completed steps
        for step in &context.plan.steps {
            if context.completed_steps.contains(&step.step_id) {
                new_steps.push(step.action.clone());
            }
        }

        // Add repair steps
        new_steps.extend(repair_steps);

        // Create new plan
        let (ordered_steps, parallel_tracks) = if self.config.enable_parallel {
            self.optimize_parallel_execution(&new_steps).await?
        } else {
            (self.create_sequential_plan(&new_steps), Vec::new())
        };

        // Calculate metrics before creating the Plan struct
        let total_duration = self.calculate_total_duration(&ordered_steps);
        let total_cost = ordered_steps.iter().map(|s| s.action.cost).sum();
        let success_probability = self.calculate_success_probability(&ordered_steps);

        let repaired_plan = Plan {
            id: format!("{}_repair_{}", plan_id, uuid::Uuid::new_v4()),
            goal_id: context.plan.goal_id.clone(),
            steps: ordered_steps,
            total_duration,
            total_cost,
            success_probability,
            parallel_tracks,
        };

        // Update stats
        let mut stats = self.stats.write().await;
        stats.repairs_attempted += 1;

        Ok(repaired_plan)
    }

    /// Get planner statistics
    pub async fn get_stats(&self) -> PlannerStats {
        self.stats.read().await.clone()
    }
    
    /// Get planned actions across all active plans
    pub async fn get_planned_actions(&self) -> Result<Vec<Action>> {
        let plans = self.active_plans.read().await;
        let mut actions = Vec::new();
        
        for context in plans.values() {
            for step in &context.plan.steps {
                actions.push(step.action.clone());
            }
        }
        
        Ok(actions)
    }
}

/// Plan monitor for tracking execution
pub struct PlanMonitor {
    planner: Arc<ActionPlanner>,
    update_rx: mpsc::Receiver<PlanUpdate>,
}

impl PlanMonitor {
    pub async fn monitor_loop(&mut self) {
        while let Some(update) = self.update_rx.recv().await {
            match update {
                PlanUpdate::StepFailed(plan_id, step_id, reason) => {
                    warn!("Plan {} step {} failed: {}", plan_id, step_id, reason);

                    // Attempt repair if threshold met
                    if let Ok(plan) = self.planner.repair_plan(&plan_id, step_id).await {
                        info!("Successfully repaired plan: {}", plan.id);
                        let _ = self.planner.execute_plan(plan).await;
                    }
                }

                PlanUpdate::PlanCompleted(plan_id) => {
                    info!("Plan {} completed successfully", plan_id);

                    let mut stats = self.planner.stats.write().await;
                    stats.plans_succeeded += 1;
                }

                PlanUpdate::PlanFailed(plan_id, reason) => {
                    warn!("Plan {} failed: {}", plan_id, reason);

                    let mut stats = self.planner.stats.write().await;
                    stats.plans_failed += 1;
                }

                _ => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_action_registration() {
        let repo = ActionRepository::new();

        let action = Action {
            id: ActionId::new(),
            name: "learn_topic".to_string(),
            description: "Learn a new topic".to_string(),
            preconditions: vec![Condition {
                variable: "energy_level".to_string(),
                operator: ConditionOperator::GreaterThan,
                value: StateValue::Integer(30),
            }],
            effects: vec![Effect {
                variable: "knowledge".to_string(),
                operation: EffectOperation::Increment,
                value: StateValue::Integer(10),
                probability: 0.9,
            }],
            duration: Duration::from_secs(3600),
            resources: ResourceRequirements {
                cognitive_load: 0.7,
                memory_usage: 0.4,
                energy: 0.3,
                exclusive_resources: vec!["focus".to_string()],
            },
            cost: 0.5,
            reliability: 0.85,
            parallelizable: false,
        };

        repo.register_action(action).await.unwrap();

        let found = repo.find_actions_for_effect("knowledge", &StateValue::Integer(10)).await;
        assert_eq!(found.len(), 1);
    }
}
