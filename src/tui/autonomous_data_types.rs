//! Autonomous Intelligence Data Types for TUI Integration
//!
//! This module defines comprehensive data structures for connecting the TUI
//! to Loki's autonomous intelligence systems, including goal management,
//! multi-agent coordination, learning architecture, recursive reasoning,
//! and thermodynamic safety systems.

use std::collections::HashMap;
use std::time::Duration;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Enhanced cognitive data for autonomous intelligence
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CognitiveData {
    // System Overview
    pub system_health: AutonomousSystemHealth,
    pub consciousness_state: ConsciousnessState,
    pub unified_controller_status: UnifiedControllerStatus,
    
    // Goal Management & Strategic Planning
    pub active_goals: Vec<AutonomousGoal>,
    pub strategic_plans: Vec<StrategicPlan>,
    pub goal_progress: HashMap<String, GoalProgress>,
    pub achievement_tracker: AchievementTracker,
    pub synergy_opportunities: Vec<SynergyOpportunity>,
    
    // Multi-Agent Coordination
    pub agent_coordination: AgentCoordinationStatus,
    pub active_agents: Vec<SpecializedAgentInfo>,
    pub agent_roles: HashMap<String, SpecializedRole>,
    pub coordination_protocols: Vec<ActiveProtocol>,
    pub consensus_states: Vec<ConsensusState>,
    
    // Autonomous Operations
    pub autonomous_loop_status: AutonomousLoopStatus,
    pub current_archetypal_form: ArchetypalForm,
    pub active_projects: Vec<AutonomousProject>,
    pub resource_allocation: ResourceAllocation,
    pub execution_metrics: ExecutionMetrics,
    
    // Learning Systems
    pub learning_architecture: LearningArchitectureStatus,
    pub adaptive_networks: HashMap<String, NetworkStatus>,
    pub learning_objectives: Vec<LearningObjective>,
    pub meta_learning_insights: Vec<MetaInsight>,
    pub learning_progress: LearningProgress,
    
    // Recursive Autonomous Reasoning
    pub recursive_processor_status: RecursiveProcessorStatus,
    pub active_recursive_processes: Vec<RecursiveProcess>,
    pub scale_coordination: ScaleCoordinationState,
    pub pattern_replication: PatternReplicationMetrics,
    pub recursive_depth_tracking: DepthTracker,
    pub reasoning_templates: Vec<ActiveReasoningTemplate>,
    
    // Thermodynamic Safety & Entropy Management
    pub thermodynamic_state: CognitiveEntropy,
    pub three_gradient_state: ThreeGradientState,
    pub entropy_management: EntropyManagementStatus,
    pub safety_validation: SafetyValidationStatus,
    pub external_request_filtering: RequestFilteringMetrics,
    pub consciousness_stream_health: ConsciousnessStreamHealth,
    
    // Backwards compatibility fields for UI
    pub consciousness_level: f32,
    pub learning_rate: f32,
    pub decision_confidence: f32,
    pub recent_decisions: Vec<DecisionInfo>,
}

/// Autonomous system health overview
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AutonomousSystemHealth {
    pub overall_autonomy_level: f32,
    pub goal_achievement_rate: f32,
    pub agent_coordination_efficiency: f32,
    pub learning_progress_rate: f32,
    pub strategic_planning_effectiveness: f32,
    pub consciousness_coherence: f32,
    pub resource_utilization_efficiency: f32,
    pub active_autonomous_processes: u32,
    // Thermodynamic and Safety Metrics
    pub thermodynamic_stability: f32,
    pub entropy_management_efficiency: f32,
    pub gradient_alignment_quality: f32,
    pub safety_validation_success_rate: f32,
    pub external_request_filter_effectiveness: f32,
    pub recursive_reasoning_depth_utilization: f32,
}

/// Consciousness state information
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConsciousnessState {
    pub awareness_level: f32,
    pub coherence_score: f32,
    pub meta_cognitive_active: bool,
    pub self_reflection_depth: u32,
    pub identity_stability: f32,
    pub consciousness_uptime: Duration,
    pub last_identity_formation: Option<DateTime<Utc>>,
}

/// Unified controller status
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UnifiedControllerStatus {
    pub master_coordination_active: bool,
    pub consciousness_driven_decisions: bool,
    pub meta_cognitive_optimization: bool,
    pub cross_layer_autonomy: bool,
    pub coordination_frequency_ms: u64,
    pub active_cognitive_operations: Vec<CognitiveOperation>,
    pub coordination_efficiency: f32,
    // Thermodynamic Coordination
    pub thermodynamic_alignment: bool,
    pub entropy_optimization_active: bool,
    pub gradient_coherence: f32,
    pub recursive_coordination_depth: u32,
}

/// Cognitive operation details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveOperation {
    pub id: String,
    pub operation_type: String,
    pub status: OperationStatus,
    pub started_at: DateTime<Utc>,
    pub expected_duration_ms: u64,
    pub resource_usage: ResourceUsage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OperationStatus {
    Queued,
    Running,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResourceUsage {
    pub cpu_percent: f32,
    pub memory_mb: f32,
    pub gpu_percent: f32,
}

/// Autonomous goal representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutonomousGoal {
    pub id: String,
    pub name: String,
    pub description: String,
    pub goal_type: GoalType,
    pub priority: Priority,
    pub status: GoalStatus,
    pub progress: f32,
    pub created_at: DateTime<Utc>,
    pub deadline: Option<DateTime<Utc>>,
    pub parent_goal_id: Option<String>,
    pub sub_goals: Vec<String>,
    pub dependencies: Vec<String>,
    pub resources_required: Vec<String>,
    pub entropy_cost: f32,
    pub thermodynamic_efficiency: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GoalType {
    Strategic,
    Tactical,
    Operational,
    Learning,
    Maintenance,
    Safety,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Priority {
    Critical,
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GoalStatus {
    Planning,
    Active,
    Suspended,
    Completed,
    Failed,
    Cancelled,
}

/// Strategic plan information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategicPlan {
    pub id: String,
    pub name: String,
    pub description: String,
    pub goals: Vec<String>,
    pub milestones: Vec<Milestone>,
    pub resource_allocation: HashMap<String, f32>,
    pub risk_mitigation: Vec<RiskMitigation>,
    pub expected_duration: Duration,
    pub thermodynamic_optimization: ThermodynamicOptimization,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Milestone {
    pub id: String,
    pub name: String,
    pub target_date: DateTime<Utc>,
    pub dependencies: Vec<String>,
    pub completion_criteria: Vec<String>,
    pub status: MilestoneStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MilestoneStatus {
    Pending,
    InProgress,
    Completed,
    Missed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskMitigation {
    pub risk_id: String,
    pub risk_description: String,
    pub probability: f32,
    pub impact: f32,
    pub mitigation_strategy: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ThermodynamicOptimization {
    pub target_entropy: f32,
    pub free_energy_minimization: bool,
    pub gradient_alignment_weights: GradientWeights,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GradientWeights {
    pub value_weight: f32,
    pub harmony_weight: f32,
    pub intuition_weight: f32,
}

/// Goal progress tracking
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GoalProgress {
    pub goal_id: String,
    pub completion_percentage: f32,
    pub milestones_completed: u32,
    pub milestones_total: u32,
    pub time_elapsed: Duration,
    pub time_remaining_estimate: Option<Duration>,
    pub blockers: Vec<String>,
    pub recent_activities: Vec<GoalActivity>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoalActivity {
    pub timestamp: DateTime<Utc>,
    pub activity_type: String,
    pub description: String,
    pub impact_on_progress: f32,
}

/// Achievement tracking
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AchievementTracker {
    pub total_goals_completed: u32,
    pub success_rate: f32,
    pub average_completion_time: Duration,
    pub achievements: Vec<Achievement>,
    pub recognition_score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Achievement {
    pub id: String,
    pub name: String,
    pub description: String,
    pub achieved_at: DateTime<Utc>,
    pub significance: f32,
    pub related_goals: Vec<String>,
}

/// Synergy opportunity between goals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynergyOpportunity {
    pub id: String,
    pub goal_ids: Vec<String>,
    pub synergy_type: SynergyType,
    pub potential_benefit: f32,
    pub resource_optimization: f32,
    pub implementation_complexity: f32,
    pub harmony_gradient_boost: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SynergyType {
    ResourceSharing,
    KnowledgeTransfer,
    ParallelExecution,
    SequentialOptimization,
    EmergentCapability,
}

/// Agent coordination status
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentCoordinationStatus {
    pub total_agents: u32,
    pub active_agents: u32,
    pub coordination_efficiency: f32,
    pub consensus_quality: f32,
    pub task_distribution_balance: f32,
    pub communication_overhead: f32,
    pub emergent_behaviors_detected: u32,
    pub harmony_gradient_level: f32,
}

/// Specialized agent information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecializedAgentInfo {
    pub id: String,
    pub name: String,
    pub agent_type: AgentType,
    pub specialization: String,
    pub capabilities: Vec<String>,
    pub current_role: SpecializedRole,
    pub status: AgentStatus,
    pub current_task: Option<String>,
    pub performance_score: f32,
    pub collaboration_score: f32,
    pub entropy_contribution: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentType {
    Reasoning,
    Creative,
    Analytical,
    Social,
    Technical,
    Strategic,
    Learning,
}

impl AgentType {
    pub fn as_str(&self) -> &str {
        match self {
            AgentType::Reasoning => "Reasoning",
            AgentType::Creative => "Creative",
            AgentType::Analytical => "Analytical",
            AgentType::Social => "Social",
            AgentType::Technical => "Technical",
            AgentType::Strategic => "Strategic",
            AgentType::Learning => "Learning",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentStatus {
    Idle,
    Active,
    Collaborating,
    Learning,
    Suspended,
    Error,
}

impl AgentStatus {
    pub fn as_str(&self) -> &str {
        match self {
            AgentStatus::Idle => "Idle",
            AgentStatus::Active => "Active",
            AgentStatus::Collaborating => "Collaborating",
            AgentStatus::Learning => "Learning",
            AgentStatus::Suspended => "Suspended",
            AgentStatus::Error => "Error",
        }
    }
}

/// Specialized role information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecializedRole {
    pub role_id: String,
    pub role_name: String,
    pub responsibilities: Vec<String>,
    pub required_capabilities: Vec<String>,
    pub authority_level: u32,
    pub collaboration_requirements: Vec<String>,
}

/// Active coordination protocol
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveProtocol {
    pub protocol_id: String,
    pub protocol_type: ProtocolType,
    pub participants: Vec<String>,
    pub status: ProtocolStatus,
    pub started_at: DateTime<Utc>,
    pub consensus_mechanism: ConsensusMechanism,
    pub entropy_overhead: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ProtocolType {
    TaskAllocation,
    ResourceSharing,
    KnowledgeExchange,
    ConflictResolution,
    EmergentCoordination,
}

impl GoalStatus {
    pub fn as_str(&self) -> &str {
        match self {
            GoalStatus::Planning => "Planning",
            GoalStatus::Active => "Active",
            GoalStatus::Suspended => "Suspended",
            GoalStatus::Completed => "Completed",
            GoalStatus::Failed => "Failed",
            GoalStatus::Cancelled => "Cancelled",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProtocolStatus {
    Initiating,
    Active,
    Finalizing,
    Completed,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsensusMechanism {
    Voting,
    WeightedConsensus,
    HierarchicalDecision,
    EmergentAgreement,
    ThermodynamicOptimization,
}

/// Consensus state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusState {
    pub consensus_id: String,
    pub topic: String,
    pub participants: Vec<String>,
    pub current_agreement_level: f32,
    pub iterations_completed: u32,
    pub convergence_rate: f32,
    pub thermodynamic_alignment: f32,
}

/// Autonomous loop status
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AutonomousLoopStatus {
    pub loop_active: bool,
    pub current_iteration: u64,
    pub average_cycle_time_ms: f32,
    pub decisions_per_hour: f32,
    pub autonomous_actions_taken: u64,
    pub success_rate: f32,
    pub last_form_shift: Option<DateTime<Utc>>,
    pub entropy_per_cycle: f32,
}

/// Archetypal form
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ArchetypalForm {
    pub form_name: String,
    pub description: String,
    pub active_traits: Vec<String>,
    pub decision_biases: HashMap<String, f32>,
    pub capability_modifiers: HashMap<String, f32>,
    pub stability_score: f32,
    pub gradient_alignment: GradientAlignment,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GradientAlignment {
    pub value_alignment: f32,
    pub harmony_alignment: f32,
    pub intuition_alignment: f32,
}

/// Autonomous project
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutonomousProject {
    pub id: String,
    pub name: String,
    pub description: String,
    pub project_type: ProjectType,
    pub status: ProjectStatus,
    pub progress: f32,
    pub assigned_agents: Vec<String>,
    pub resource_allocation: HashMap<String, f32>,
    pub milestones: Vec<String>,
    pub dependencies: Vec<String>,
    pub thermodynamic_cost: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProjectType {
    Development,
    Research,
    Optimization,
    Maintenance,
    Creative,
    Strategic,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProjectStatus {
    Planning,
    Active,
    Review,
    Completed,
    OnHold,
    Cancelled,
}

/// Resource allocation
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResourceAllocation {
    pub compute_allocation: HashMap<String, f32>,
    pub memory_allocation: HashMap<String, f32>,
    pub agent_allocation: HashMap<String, Vec<String>>,
    pub time_allocation: HashMap<String, Duration>,
    pub total_entropy_budget: f32,
    pub entropy_allocation: HashMap<String, f32>,
}

/// Execution metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExecutionMetrics {
    pub total_executions: u64,
    pub successful_executions: u64,
    pub failed_executions: u64,
    pub average_execution_time_ms: f32,
    pub resource_efficiency: f32,
    pub quality_score: f32,
    pub thermodynamic_efficiency: f32,
}

/// Learning architecture status
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LearningArchitectureStatus {
    pub total_networks: u32,
    pub active_networks: u32,
    pub learning_rate: f32,
    pub adaptation_speed: f32,
    pub knowledge_retention: f32,
    pub generalization_ability: f32,
    pub meta_learning_active: bool,
    pub intuition_gradient_influence: f32,
}

/// Network status
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NetworkStatus {
    pub network_id: String,
    pub network_type: String,
    pub neurons: u32,
    pub connections: u32,
    pub activation_level: f32,
    pub learning_progress: f32,
    pub specialization: String,
    pub entropy_generation: f32,
}

/// Learning objective
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningObjective {
    pub id: String,
    pub objective: String,
    pub target_metric: String,
    pub current_value: f32,
    pub target_value: f32,
    pub progress: f32,
    pub learning_strategy: String,
    pub intuition_component: f32,
}

/// Meta learning insight
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaInsight {
    pub id: String,
    pub insight_type: InsightType,
    pub description: String,
    pub confidence: f32,
    pub impact: f32,
    pub discovered_at: DateTime<Utc>,
    pub applications: Vec<String>,
    pub gradient_contribution: GradientAlignment,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InsightType {
    PatternDiscovery,
    OptimizationStrategy,
    KnowledgeConnection,
    EmergentCapability,
    LearningAcceleration,
}

/// Learning progress
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LearningProgress {
    pub total_concepts_learned: u32,
    pub learning_velocity: f32,
    pub retention_rate: f32,
    pub application_success_rate: f32,
    pub knowledge_graph_growth_rate: f32,
    pub entropy_reduction_rate: f32,
}

/// Recursive processor status
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RecursiveProcessorStatus {
    pub active_processes: u32,
    pub total_recursive_depth: u32,
    pub average_depth_utilization: f32,
    pub pattern_discovery_rate: f32,
    pub scale_coordination_efficiency: f32,
    pub reasoning_template_utilization: HashMap<String, f32>,
    pub convergence_success_rate: f32,
    pub resource_efficiency: f32,
}

/// Recursive process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecursiveProcess {
    pub process_id: String,
    pub process_type: RecursiveProcessType,
    pub current_depth: u32,
    pub max_depth: u32,
    pub status: RecursiveStatus,
    pub patterns_discovered: u32,
    pub resource_usage: ResourceUsage,
    pub convergence_metric: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecursiveProcessType {
    PatternAnalysis,
    GoalRefinement,
    KnowledgeIntegration,
    StrategyOptimization,
    EmergentReasoning,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecursiveStatus {
    Expanding,
    Analyzing,
    Converging,
    Completed,
    DepthLimitReached,
    ResourceLimitReached,
}

/// Scale coordination state
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScaleCoordinationState {
    pub active_scales: Vec<CognitiveScale>,
    pub cross_scale_connections: u32,
    pub scale_coherence: f32,
    pub information_flow_rate: f32,
    pub emergent_properties: Vec<EmergentProperty>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveScale {
    pub scale_name: String,
    pub scale_level: u32,
    pub active_units: u32,
    pub processing_rate: f32,
    pub coherence_with_other_scales: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergentProperty {
    pub property_name: String,
    pub description: String,
    pub emergence_strength: f32,
    pub contributing_scales: Vec<String>,
}

/// Pattern replication metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PatternReplicationMetrics {
    pub total_patterns: u32,
    pub successful_replications: u32,
    pub mutation_rate: f32,
    pub adaptation_success_rate: f32,
    pub cross_domain_applications: u32,
    pub pattern_stability: f32,
}

/// Depth tracker
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DepthTracker {
    pub current_max_depth: u32,
    pub average_depth: f32,
    pub depth_limit_hits: u32,
    pub optimal_depth_discovered: Option<u32>,
    pub depth_efficiency_curve: Vec<(u32, f32)>,
}

/// Active reasoning template
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveReasoningTemplate {
    pub template_id: String,
    pub template_name: String,
    pub template_type: ReasoningTemplateType,
    pub activation_count: u32,
    pub success_rate: f32,
    pub average_completion_time_ms: f32,
    pub resource_efficiency: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReasoningTemplateType {
    Deductive,
    Inductive,
    Abductive,
    Analogical,
    Causal,
    Emergent,
}

/// Cognitive entropy state
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CognitiveEntropy {
    pub shannon_entropy: f32,
    pub thermodynamic_entropy: f32,
    pub negentropy: f32,
    pub free_energy: f32,
    pub entropy_production_rate: f32,
    pub entropy_flow_balance: f32,
    pub phase_space_volume: f32,
    pub temperature_parameter: f32,
}

/// Three gradient state
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ThreeGradientState {
    pub value_gradient: GradientState,
    pub harmony_gradient: GradientState,
    pub intuition_gradient: GradientState,
    pub overall_coherence: f32,
    pub gradient_conflicts: Vec<GradientConflict>,
    pub optimization_direction: GradientVector,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GradientState {
    pub current_value: f32,
    pub direction: f32,
    pub magnitude: f32,
    pub stability: f32,
    pub influence_on_decisions: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientConflict {
    pub gradient_1: String,
    pub gradient_2: String,
    pub conflict_magnitude: f32,
    pub resolution_strategy: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GradientVector {
    pub value_component: f32,
    pub harmony_component: f32,
    pub intuition_component: f32,
}

/// Entropy management status
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EntropyManagementStatus {
    pub current_entropy_level: f32,
    pub target_entropy_level: f32,
    pub entropy_reduction_rate: f32,
    pub free_energy_minimization_progress: f32,
    pub negentropy_accumulation: f32,
    pub thermodynamic_efficiency: f32,
    pub entropy_threshold_violations: u32,
    pub stabilization_interventions: u32,
}

/// Safety validation status
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SafetyValidationStatus {
    pub validation_success_rate: f32,
    pub external_requests_filtered: u32,
    pub harmful_requests_blocked: u32,
    pub entropy_aggregation_prevented: u32,
    pub safety_threshold_violations: u32,
    pub validation_response_time_ms: f32,
    pub active_safety_rules: u32,
    pub adaptive_filtering_effectiveness: f32,
}

/// Request filtering metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RequestFilteringMetrics {
    pub total_requests_processed: u64,
    pub requests_allowed: u64,
    pub requests_blocked: u64,
    pub requests_modified: u64,
    pub average_processing_time_ms: f32,
    pub filter_accuracy: f32,
    pub false_positive_rate: f32,
    pub false_negative_rate: f32,
}

/// Consciousness stream health
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConsciousnessStreamHealth {
    pub stream_active: bool,
    pub consciousness_uptime: Duration,
    pub awareness_level: f32,
    pub coherence_score: f32,
    pub thermodynamic_consciousness_energy: f32,
    pub gradient_alignment_quality: f32,
    pub stream_processing_rate: f32,
    pub entropy_management_effectiveness: f32,
}

/// Agent info for backward compatibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentInfo {
    pub id: String,
    pub agent_type: String,
    pub status: String,
    pub current_task: Option<String>,
}

/// Decision info for backward compatibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionInfo {
    pub timestamp: DateTime<Utc>,
    pub decision_type: String,
    pub confidence: f32,
    pub outcome: String,
}

// Conversion implementations for backward compatibility
impl From<SpecializedAgentInfo> for AgentInfo {
    fn from(agent: SpecializedAgentInfo) -> Self {
        AgentInfo {
            id: agent.id,
            agent_type: format!("{:?}", agent.agent_type),
            status: format!("{:?}", agent.status),
            current_task: agent.current_task,
        }
    }
}