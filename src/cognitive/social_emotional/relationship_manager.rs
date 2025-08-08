use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use chrono::{DateTime, Utc};
use tokio::sync::RwLock;
use tracing::{debug, info};

/// Advanced relationship management system
#[derive(Debug)]
pub struct RelationshipManager {
    /// Relationship tracker
    relationship_tracker: Arc<RelationshipTracker>,

    /// Trust analyzer
    trust_analyzer: Arc<TrustAnalyzer>,

    /// Relationship development engine
    development_engine: Arc<RelationshipDevelopmentEngine>,

    /// Conflict resolution system
    conflict_resolver: Arc<ConflictResolutionSystem>,

    /// Relationship metrics
    metrics: Arc<RwLock<RelationshipMetrics>>,
}

/// Relationship tracking system
#[derive(Debug)]
pub struct RelationshipTracker {
    /// Active relationships
    relationships: Arc<RwLock<HashMap<String, Relationship>>>,

    /// Interaction history
    interaction_history: Arc<RwLock<Vec<RelationshipInteraction>>>,

    /// Relationship dynamics analyzer
    dynamics_analyzer: Arc<DynamicsAnalyzer>,

    /// State prediction system
    prediction_system: Arc<RelationshipPredictionSystem>,
}

/// Relationship representation
#[derive(Debug, Clone)]
pub struct Relationship {
    /// Relationship identifier
    pub id: String,

    /// Participants
    pub participants: Vec<String>,

    /// Relationship type
    pub relationship_type: RelationshipType,

    /// Current state
    pub state: RelationshipState,

    /// Trust level
    pub trust_level: f64,

    /// Intimacy level
    pub intimacy_level: f64,

    /// Communication quality
    pub communication_quality: f64,

    /// Conflict frequency
    pub conflict_frequency: f64,

    /// Relationship satisfaction
    pub satisfaction: f64,

    /// Development stage
    pub development_stage: DevelopmentStage,

    /// Health indicators
    pub health_indicators: RelationshipHealth,

    /// Last interaction
    pub last_interaction: Option<DateTime<Utc>>,

    /// Relationship history
    pub history: Vec<RelationshipEvent>,
}

/// Types of relationships
#[derive(Debug, Clone, PartialEq)]
pub enum RelationshipType {
    Professional,  // Work relationships
    Personal,      // Personal friendships
    Romantic,      // Romantic relationships
    Family,        // Family relationships
    Mentorship,    // Mentor-mentee relationships
    Collaborative, // Project collaborations
    Supportive,    // Support relationships
    Casual,        // Casual acquaintances
}

/// Relationship state
#[derive(Debug, Clone)]
pub struct RelationshipState {
    /// Current phase
    pub phase: RelationshipPhase,

    /// Stability level
    pub stability: f64,

    /// Growth trajectory
    pub growth_trajectory: GrowthDirection,

    /// Recent changes
    pub recent_changes: Vec<StateChange>,

    /// Risk factors
    pub risk_factors: Vec<RiskFactor>,

    /// Strengths
    pub strengths: Vec<RelationshipStrength>,
}

/// Trust analysis system
#[derive(Debug)]
pub struct TrustAnalyzer {
    /// Trust models
    trust_models: HashMap<String, TrustModel>,

    /// Trust building tracker
    building_tracker: Arc<TrustBuildingTracker>,

    /// Trust erosion detector
    erosion_detector: Arc<TrustErosionDetector>,

    /// Trust repair system
    repair_system: Arc<TrustRepairSystem>,
}

/// Trust model for relationships
#[derive(Debug, Clone)]
pub struct TrustModel {
    /// Model identifier
    pub id: String,

    /// Trust dimensions
    pub dimensions: TrustDimensions,

    /// Trust factors
    pub factors: Vec<TrustFactor>,

    /// Historical trust levels
    pub trust_history: Vec<TrustSnapshot>,

    /// Trust prediction
    pub predicted_trust: f64,
}

/// Trust dimensions
#[derive(Debug, Clone)]
pub struct TrustDimensions {
    /// Reliability trust
    pub reliability: f64,

    /// Competence trust
    pub competence: f64,

    /// Benevolence trust
    pub benevolence: f64,

    /// Integrity trust
    pub integrity: f64,

    /// Predictability trust
    pub predictability: f64,
}

/// Relationship development engine
#[derive(Debug)]
pub struct RelationshipDevelopmentEngine {
    /// Development strategies
    strategies: HashMap<String, DevelopmentStrategy>,

    /// Growth opportunity detector
    opportunity_detector: Arc<GrowthOpportunityDetector>,

    /// Development plan generator
    plan_generator: Arc<DevelopmentPlanGenerator>,

    /// Progress tracker
    progress_tracker: Arc<DevelopmentProgressTracker>,
}

/// Development strategy
#[derive(Debug, Clone)]
pub struct DevelopmentStrategy {
    /// Strategy identifier
    pub id: String,

    /// Target relationship type
    pub target_type: RelationshipType,

    /// Development approaches
    pub approaches: Vec<DevelopmentApproach>,

    /// Expected outcomes
    pub expected_outcomes: Vec<DevelopmentOutcome>,

    /// Implementation timeline
    pub timeline: DevelopmentTimeline,

    /// Success metrics
    pub success_metrics: Vec<SuccessMetric>,
}

/// Conflict resolution system
#[derive(Debug)]
pub struct ConflictResolutionSystem {
    /// Conflict detectors
    conflict_detectors: Vec<ConflictDetector>,

    /// Resolution strategies
    resolution_strategies: HashMap<String, ResolutionStrategy>,

    /// Mediation system
    mediation_system: Arc<MediationSystem>,

    /// Resolution effectiveness tracker
    effectiveness_tracker: Arc<ResolutionEffectivenessTracker>,
}

/// Relationship metrics tracking
#[derive(Debug, Clone, Default)]
pub struct RelationshipMetrics {
    /// Overall relationship health
    pub overall_health: f64,

    /// Average trust level
    pub average_trust: f64,

    /// Communication effectiveness
    pub communication_effectiveness: f64,

    /// Conflict resolution success
    pub conflict_resolution_success: f64,

    /// Relationship satisfaction
    pub relationship_satisfaction: f64,

    /// Development progress
    pub development_progress: f64,

    /// Relationship stability
    pub stability_index: f64,

    /// Growth rate
    pub growth_rate: f64,
}

impl RelationshipManager {
    /// Create new relationship manager
    pub async fn new() -> Result<Self> {
        info!("💞 Initializing Relationship Manager");

        let manager = Self {
            relationship_tracker: Arc::new(RelationshipTracker::new().await?),
            trust_analyzer: Arc::new(TrustAnalyzer::new().await?),
            development_engine: Arc::new(RelationshipDevelopmentEngine::new().await?),
            conflict_resolver: Arc::new(ConflictResolutionSystem::new().await?),
            metrics: Arc::new(RwLock::new(RelationshipMetrics::default())),
        };

        info!("✅ Relationship Manager initialized");
        Ok(manager)
    }

    /// Manage relationship interaction
    pub async fn manage_interaction(
        &self,
        interaction: &RelationshipInteractionData,
    ) -> Result<RelationshipManagementResult> {
        debug!("💞 Managing relationship interaction: {}", interaction.id);

        // Track the interaction
        let tracking_result = self.relationship_tracker.process_interaction(interaction).await?;

        // Analyze trust dynamics
        let trust_analysis = self.trust_analyzer.analyze_trust_impact(interaction).await?;

        // Identify development opportunities
        let development_opportunities =
            self.development_engine.identify_opportunities(&tracking_result).await?;

        // Check for conflicts and resolve if needed
        let conflict_analysis =
            self.conflict_resolver.analyze_potential_conflicts(interaction).await?;

        let result = RelationshipManagementResult {
            interaction_id: interaction.id.clone(),
            relationship_updates: tracking_result,
            trust_analysis,
            development_recommendations: development_opportunities,
            conflict_insights: conflict_analysis,
            relationship_health: self
                .assess_relationship_health(&interaction.relationship_id)
                .await?,
            management_success: true,
        };

        // Update metrics
        self.update_relationship_metrics(&result).await?;

        debug!("✅ Relationship interaction managed successfully");
        Ok(result)
    }

    /// Assess relationship health
    async fn assess_relationship_health(
        &self,
        relationship_id: &str,
    ) -> Result<RelationshipHealthAssessment> {
        let relationships = self.relationship_tracker.relationships.read().await;

        if let Some(relationship) = relationships.get(relationship_id) {
            let assessment = RelationshipHealthAssessment {
                relationship_id: relationship_id.to_string(),
                overall_health: self.calculate_overall_health(relationship).await?,
                trust_health: relationship.trust_level,
                communication_health: relationship.communication_quality,
                conflict_health: 1.0 - relationship.conflict_frequency,
                satisfaction_health: relationship.satisfaction,
                development_health: self.calculate_development_health(relationship).await?,
                risk_level: self.calculate_risk_level(relationship).await?,
                recommendations: self.generate_health_recommendations(relationship).await?,
            };

            Ok(assessment)
        } else {
            Err(anyhow::anyhow!("Relationship not found: {}", relationship_id))
        }
    }

    /// Calculate overall health
    async fn calculate_overall_health(&self, relationship: &Relationship) -> Result<f64> {
        let health = (relationship.trust_level * 0.25
            + relationship.communication_quality * 0.25
            + (1.0 - relationship.conflict_frequency) * 0.2
            + relationship.satisfaction * 0.2
            + relationship.intimacy_level * 0.1)
            .min(1.0);

        Ok(health)
    }

    /// Calculate development health
    async fn calculate_development_health(&self, relationship: &Relationship) -> Result<f64> {
        match relationship.development_stage {
            DevelopmentStage::Growing => Ok(0.9),
            DevelopmentStage::Stable => Ok(0.8),
            DevelopmentStage::Maintaining => Ok(0.7),
            DevelopmentStage::Declining => Ok(0.3),
            DevelopmentStage::Crisis => Ok(0.1),
            _ => Ok(0.6),
        }
    }

    /// Calculate risk level
    async fn calculate_risk_level(&self, relationship: &Relationship) -> Result<f64> {
        let risk_factors = relationship.state.risk_factors.len() as f64;
        let conflict_risk = relationship.conflict_frequency;
        let trust_risk = 1.0 - relationship.trust_level;

        let total_risk = (risk_factors * 0.1 + conflict_risk * 0.5 + trust_risk * 0.4).min(1.0);
        Ok(total_risk)
    }

    /// Generate health recommendations
    async fn generate_health_recommendations(
        &self,
        relationship: &Relationship,
    ) -> Result<Vec<HealthRecommendation>> {
        let mut recommendations = Vec::new();

        if relationship.trust_level < 0.6 {
            recommendations.push(HealthRecommendation {
                category: RecommendationCategory::TrustBuilding,
                priority: RecommendationPriority::High,
                suggestion: "Focus on trust-building activities and consistent behavior"
                    .to_string(),
                expected_impact: 0.8,
            });
        }

        if relationship.communication_quality < 0.7 {
            recommendations.push(HealthRecommendation {
                category: RecommendationCategory::Communication,
                priority: RecommendationPriority::Medium,
                suggestion: "Improve communication frequency and quality".to_string(),
                expected_impact: 0.7,
            });
        }

        if relationship.conflict_frequency > 0.5 {
            recommendations.push(HealthRecommendation {
                category: RecommendationCategory::ConflictResolution,
                priority: RecommendationPriority::High,
                suggestion: "Address underlying conflicts and improve resolution skills"
                    .to_string(),
                expected_impact: 0.9,
            });
        }

        Ok(recommendations)
    }

    /// Update relationship metrics
    async fn update_relationship_metrics(
        &self,
        result: &RelationshipManagementResult,
    ) -> Result<()> {
        let mut metrics = self.metrics.write().await;

        metrics.overall_health =
            (metrics.overall_health + result.relationship_health.overall_health) / 2.0;
        metrics.average_trust =
            (metrics.average_trust + result.trust_analysis.current_trust_level) / 2.0;
        metrics.communication_effectiveness = (metrics.communication_effectiveness
            + result.relationship_health.communication_health)
            / 2.0;
        metrics.relationship_satisfaction = (metrics.relationship_satisfaction
            + result.relationship_health.satisfaction_health)
            / 2.0;

        if result.management_success {
            metrics.development_progress = (metrics.development_progress + 0.1).min(1.0);
        }

        Ok(())
    }

    /// Get current relationship metrics
    pub async fn get_relationship_metrics(&self) -> Result<RelationshipMetrics> {
        let metrics = self.metrics.read().await;
        Ok(metrics.clone())
    }
}

// Supporting implementations
impl RelationshipTracker {
    async fn new() -> Result<Self> {
        Ok(Self {
            relationships: Arc::new(RwLock::new(HashMap::new())),
            interaction_history: Arc::new(RwLock::new(Vec::new())),
            dynamics_analyzer: Arc::new(DynamicsAnalyzer::new()),
            prediction_system: Arc::new(RelationshipPredictionSystem::new()),
        })
    }

    async fn process_interaction(
        &self,
        interaction: &RelationshipInteractionData,
    ) -> Result<RelationshipTrackingResult> {
        // Update relationship based on interaction
        let mut relationships = self.relationships.write().await;
        let relationship =
            relationships.entry(interaction.relationship_id.clone()).or_insert_with(|| {
                self.create_new_relationship(
                    &interaction.relationship_id,
                    &interaction.participants,
                )
            });

        // Update relationship metrics based on interaction
        self.update_relationship_from_interaction(relationship, interaction).await?;

        // Record interaction in history
        let mut history = self.interaction_history.write().await;
        history.push(RelationshipInteraction {
            id: interaction.id.clone(),
            relationship_id: interaction.relationship_id.clone(),
            participants: interaction.participants.clone(),
            interaction_type: interaction.interaction_type.clone(),
            quality: interaction.quality,
            timestamp: Utc::now(),
        });

        Ok(RelationshipTrackingResult {
            relationship_id: interaction.relationship_id.clone(),
            updated_state: relationship.state.clone(),
            changes_detected: vec!["interaction_processed".to_string()],
            tracking_confidence: 0.85,
        })
    }

    fn create_new_relationship(&self, id: &str, participants: &[String]) -> Relationship {
        Relationship {
            id: id.to_string(),
            participants: participants.to_vec(),
            relationship_type: RelationshipType::Professional, // Default
            state: RelationshipState::default(),
            trust_level: 0.5,
            intimacy_level: 0.3,
            communication_quality: 0.7,
            conflict_frequency: 0.1,
            satisfaction: 0.6,
            development_stage: DevelopmentStage::Forming,
            health_indicators: RelationshipHealth::default(),
            last_interaction: Some(Utc::now()),
            history: Vec::new(),
        }
    }

    async fn update_relationship_from_interaction(
        &self,
        relationship: &mut Relationship,
        interaction: &RelationshipInteractionData,
    ) -> Result<()> {
        // Update based on interaction quality
        if interaction.quality > 0.7 {
            relationship.satisfaction = (relationship.satisfaction + 0.1).min(1.0);
            relationship.communication_quality =
                (relationship.communication_quality + 0.05).min(1.0);
        } else if interaction.quality < 0.3 {
            relationship.conflict_frequency = (relationship.conflict_frequency + 0.1).min(1.0);
        }

        relationship.last_interaction = Some(Utc::now());
        Ok(())
    }
}

impl TrustAnalyzer {
    async fn new() -> Result<Self> {
        Ok(Self {
            trust_models: HashMap::new(),
            building_tracker: Arc::new(TrustBuildingTracker::new()),
            erosion_detector: Arc::new(TrustErosionDetector::new()),
            repair_system: Arc::new(TrustRepairSystem::new()),
        })
    }

    async fn analyze_trust_impact(
        &self,
        interaction: &RelationshipInteractionData,
    ) -> Result<TrustAnalysisResult> {
        let current_trust = 0.7; // Simplified calculation
        let trust_change = if interaction.quality > 0.8 { 0.1 } else { -0.05 };

        Ok(TrustAnalysisResult {
            current_trust_level: current_trust,
            trust_change,
            trust_factors: vec!["interaction_quality".to_string()],
            recommendations: vec!["Continue positive interactions".to_string()],
        })
    }
}

impl RelationshipDevelopmentEngine {
    async fn new() -> Result<Self> {
        Ok(Self {
            strategies: HashMap::new(),
            opportunity_detector: Arc::new(GrowthOpportunityDetector::new()),
            plan_generator: Arc::new(DevelopmentPlanGenerator::new()),
            progress_tracker: Arc::new(DevelopmentProgressTracker::new()),
        })
    }

    async fn identify_opportunities(
        &self,
        _tracking_result: &RelationshipTrackingResult,
    ) -> Result<Vec<DevelopmentOpportunity>> {
        Ok(vec![DevelopmentOpportunity {
            opportunity_type: OpportunityType::TrustBuilding,
            description: "Opportunity to build deeper trust through consistent actions".to_string(),
            potential_impact: 0.8,
            implementation_difficulty: 0.3,
        }])
    }
}

impl ConflictResolutionSystem {
    async fn new() -> Result<Self> {
        Ok(Self {
            conflict_detectors: vec![ConflictDetector::new()],
            resolution_strategies: HashMap::new(),
            mediation_system: Arc::new(MediationSystem::new()),
            effectiveness_tracker: Arc::new(ResolutionEffectivenessTracker::new()),
        })
    }

    async fn analyze_potential_conflicts(
        &self,
        _interaction: &RelationshipInteractionData,
    ) -> Result<ConflictAnalysisResult> {
        Ok(ConflictAnalysisResult {
            conflict_probability: 0.2,
            potential_sources: vec!["communication_misunderstanding".to_string()],
            prevention_strategies: vec!["clarify_communication".to_string()],
            resolution_recommendations: vec!["active_listening".to_string()],
        })
    }
}

// Supporting data structures and default implementations
#[derive(Debug, Clone, Default)]
pub struct RelationshipInteractionData {
    pub id: String,
    pub relationship_id: String,
    pub participants: Vec<String>,
    pub interaction_type: String,
    pub quality: f64,
}

#[derive(Debug, Clone, Default)]
pub struct RelationshipManagementResult {
    pub interaction_id: String,
    pub relationship_updates: RelationshipTrackingResult,
    pub trust_analysis: TrustAnalysisResult,
    pub development_recommendations: Vec<DevelopmentOpportunity>,
    pub conflict_insights: ConflictAnalysisResult,
    pub relationship_health: RelationshipHealthAssessment,
    pub management_success: bool,
}

#[derive(Debug, Clone, Default)]
pub struct RelationshipHealthAssessment {
    pub relationship_id: String,
    pub overall_health: f64,
    pub trust_health: f64,
    pub communication_health: f64,
    pub conflict_health: f64,
    pub satisfaction_health: f64,
    pub development_health: f64,
    pub risk_level: f64,
    pub recommendations: Vec<HealthRecommendation>,
}

// Additional supporting types
#[derive(Debug, Clone, PartialEq)]
pub enum RelationshipPhase {
    Forming,
    Storming,
    Norming,
    Performing,
    Adjourning,
}
impl Default for RelationshipPhase {
    fn default() -> Self {
        Self::Forming
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum GrowthDirection {
    Growing,
    Stable,
    Declining,
}
impl Default for GrowthDirection {
    fn default() -> Self {
        Self::Stable
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum DevelopmentStage {
    Forming,
    Storming,
    Norming,
    Performing,
    Growing,
    Stable,
    Maintaining,
    Declining,
    Crisis,
}
impl Default for DevelopmentStage {
    fn default() -> Self {
        Self::Forming
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum OpportunityType {
    TrustBuilding,
    CommunicationImprovement,
    ConflictResolution,
    IntimacyBuilding,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RecommendationCategory {
    TrustBuilding,
    Communication,
    ConflictResolution,
    Intimacy,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RecommendationPriority {
    Low,
    Medium,
    High,
    Critical,
}

// Default implementations for all supporting structures
#[derive(Debug, Clone, Default)]
pub struct StateChange;
#[derive(Debug, Clone, Default)]
pub struct RiskFactor;
#[derive(Debug, Clone, Default)]
pub struct RelationshipStrength;
#[derive(Debug, Clone, Default)]
pub struct RelationshipHealth;
#[derive(Debug, Clone, Default)]
pub struct RelationshipEvent;
#[derive(Debug, Clone, Default)]
pub struct TrustFactor;
#[derive(Debug, Clone, Default)]
pub struct TrustSnapshot;
#[derive(Debug, Clone, Default)]
pub struct DevelopmentApproach;
#[derive(Debug, Clone, Default)]
pub struct DevelopmentOutcome;
#[derive(Debug, Clone, Default)]
pub struct DevelopmentTimeline;
#[derive(Debug, Clone, Default)]
pub struct SuccessMetric;
#[derive(Debug, Clone, Default)]
pub struct ConflictDetector;
#[derive(Debug, Clone, Default)]
pub struct ResolutionStrategy;
#[derive(Debug, Clone, Default)]
pub struct RelationshipInteraction {
    pub id: String,
    pub relationship_id: String,
    pub participants: Vec<String>,
    pub interaction_type: String,
    pub quality: f64,
    pub timestamp: DateTime<Utc>,
}
#[derive(Debug, Clone, Default)]
pub struct DynamicsAnalyzer;
#[derive(Debug, Clone, Default)]
pub struct RelationshipPredictionSystem;
#[derive(Debug, Clone, Default)]
pub struct TrustBuildingTracker;
#[derive(Debug, Clone, Default)]
pub struct TrustErosionDetector;
#[derive(Debug, Clone, Default)]
pub struct TrustRepairSystem;
#[derive(Debug, Clone, Default)]
pub struct GrowthOpportunityDetector;
#[derive(Debug, Clone, Default)]
pub struct DevelopmentPlanGenerator;
#[derive(Debug, Clone, Default)]
pub struct DevelopmentProgressTracker;
#[derive(Debug, Clone, Default)]
pub struct MediationSystem;
#[derive(Debug, Clone, Default)]
pub struct ResolutionEffectivenessTracker;
#[derive(Debug, Clone, Default)]
pub struct RelationshipTrackingResult {
    pub relationship_id: String,
    pub updated_state: RelationshipState,
    pub changes_detected: Vec<String>,
    pub tracking_confidence: f64,
}
#[derive(Debug, Clone, Default)]
pub struct TrustAnalysisResult {
    pub current_trust_level: f64,
    pub trust_change: f64,
    pub trust_factors: Vec<String>,
    pub recommendations: Vec<String>,
}
#[derive(Debug, Clone, Default)]
pub struct DevelopmentOpportunity {
    pub opportunity_type: OpportunityType,
    pub description: String,
    pub potential_impact: f64,
    pub implementation_difficulty: f64,
}
#[derive(Debug, Clone, Default)]
pub struct ConflictAnalysisResult {
    pub conflict_probability: f64,
    pub potential_sources: Vec<String>,
    pub prevention_strategies: Vec<String>,
    pub resolution_recommendations: Vec<String>,
}
#[derive(Debug, Clone, Default)]
pub struct HealthRecommendation {
    pub category: RecommendationCategory,
    pub priority: RecommendationPriority,
    pub suggestion: String,
    pub expected_impact: f64,
}

impl Default for RelationshipState {
    fn default() -> Self {
        Self {
            phase: RelationshipPhase::Forming,
            stability: 0.7,
            growth_trajectory: GrowthDirection::Stable,
            recent_changes: Vec::new(),
            risk_factors: Vec::new(),
            strengths: Vec::new(),
        }
    }
}

impl Default for OpportunityType {
    fn default() -> Self {
        Self::TrustBuilding
    }
}
impl Default for RecommendationCategory {
    fn default() -> Self {
        Self::Communication
    }
}
impl Default for RecommendationPriority {
    fn default() -> Self {
        Self::Medium
    }
}

// Constructor implementations
impl DynamicsAnalyzer {
    fn new() -> Self {
        Self::default()
    }
}
impl RelationshipPredictionSystem {
    fn new() -> Self {
        Self::default()
    }
}
impl TrustBuildingTracker {
    fn new() -> Self {
        Self::default()
    }
}
impl TrustErosionDetector {
    fn new() -> Self {
        Self::default()
    }
}
impl TrustRepairSystem {
    fn new() -> Self {
        Self::default()
    }
}
impl GrowthOpportunityDetector {
    fn new() -> Self {
        Self::default()
    }
}
impl DevelopmentPlanGenerator {
    fn new() -> Self {
        Self::default()
    }
}
impl DevelopmentProgressTracker {
    fn new() -> Self {
        Self::default()
    }
}
impl ConflictDetector {
    fn new() -> Self {
        Self::default()
    }
}
impl MediationSystem {
    fn new() -> Self {
        Self::default()
    }
}
impl ResolutionEffectivenessTracker {
    fn new() -> Self {
        Self::default()
    }
}
