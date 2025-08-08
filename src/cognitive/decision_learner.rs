//! Decision Learning System
//!
//! This module implements learning mechanisms to improve decision-making over
//! time through experience replay, pattern recognition, skill development, and
//! meta-learning.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::sync::{RwLock, mpsc};
use tokio::time::interval;
use tracing::{debug, info, warn};

use crate::cognitive::{
    ActionId,
    ActualOutcome,
    Decision,
    DecisionCriterion,
    DecisionEngine,
    DecisionId,
    DecisionOption,
    GoalId,
    NeuroProcessor,
    Thought,
    ThoughtType,
};
use crate::memory::CognitiveMemory;

/// Experience from a decision, goal, or action
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Experience {
    pub id: String,
    pub experience_type: ExperienceType,
    pub context: String,
    pub decision: Option<DecisionSnapshot>,
    pub outcome: ExperienceOutcome,
    pub lessons: Vec<LessonLearned>,
    pub patterns: Vec<PatternObservation>,
    #[serde(skip, default = "Instant::now")]
    pub timestamp: Instant,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ExperienceType {
    Decision(DecisionId),
    Goal(GoalId),
    Plan(String), // Plan ID
    Action(ActionId),
}

/// Snapshot of a decision for learning
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DecisionSnapshot {
    pub options: Vec<DecisionOption>,
    pub criteria: Vec<DecisionCriterion>,
    pub selected: String, // Option ID
    pub confidence: f32,
    pub emotional_state: String, // Serialized emotional state
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExperienceOutcome {
    pub success: bool,
    pub score: f32, // -1.0 to 1.0
    pub unexpected_effects: Vec<String>,
    pub time_taken: Duration,
    pub resource_usage: f32, // 0.0 to 1.0
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LessonLearned {
    pub lesson_type: LessonType,
    pub description: String,
    pub importance: f32, // 0.0 to 1.0
    pub applicable_contexts: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum LessonType {
    StrategySuccess,
    StrategyFailure,
    ResourceOptimization,
    TimingInsight,
    ContextualFactor,
    EmotionalInfluence,
}

/// Pattern observed in experiences
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PatternObservation {
    pub pattern_type: PatternType,
    pub description: String,
    pub frequency: u32,
    pub confidence: f32,
    pub examples: Vec<String>, // Experience IDs
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum PatternType {
    SuccessPattern,
    FailurePattern,
    ContextPattern,
    TemporalPattern,
    ResourcePattern,
    EmotionalPattern,
}

/// Skill in the skill tree
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Skill {
    pub id: SkillId,
    pub name: String,
    pub description: String,
    pub category: SkillCategory,
    pub level: SkillLevel,
    pub experience_points: u64,
    pub prerequisites: Vec<SkillId>,
    pub unlocks: Vec<SkillId>,
    pub bonuses: SkillBonuses,
    #[serde(skip)]
    pub acquired_at: Option<Instant>,
    #[serde(skip, default = "Instant::now")]
    pub last_improved: Instant,
}

#[derive(Clone, Debug, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct SkillId(String);

impl SkillId {
    pub fn new(name: &str) -> Self {
        Self(name.to_string())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SkillCategory {
    DecisionMaking,
    Planning,
    ResourceManagement,
    PatternRecognition,
    EmotionalIntelligence,
    Adaptation,
    MetaCognition,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SkillLevel {
    pub current: u32,  // 0-10
    pub progress: f32, // 0.0-1.0 to next level
    pub mastery: bool,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SkillBonuses {
    pub decision_speed: f32,      // Multiplier
    pub confidence_boost: f32,    // Added confidence
    pub resource_efficiency: f32, // Resource usage reduction
    pub pattern_detection: f32,   // Pattern recognition boost
    pub adaptation_rate: f32,     // Learning speed multiplier
}

/// Transfer learning domain
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LearningDomain {
    pub name: String,
    pub description: String,
    pub experiences: Vec<String>, // Experience IDs
    pub patterns: Vec<PatternObservation>,
    pub transferable_skills: Vec<SkillId>,
    pub similarity_map: HashMap<String, f32>, // Domain -> similarity
}

/// Meta-learning strategy
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MetaStrategy {
    pub id: String,
    pub name: String,
    pub strategy_type: StrategyType,
    pub effectiveness: f32, // 0.0 to 1.0
    pub applicable_contexts: Vec<String>,
    pub parameters: HashMap<String, f32>,
    pub adaptation_history: Vec<StrategyAdaptation>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum StrategyType {
    ExplorationVsExploitation,
    RiskManagement,
    ResourceAllocation,
    TimeOptimization,
    LearningRateAdjustment,
    ContextSwitching,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StrategyAdaptation {
    pub parameter: String,
    pub old_value: f32,
    pub new_value: f32,
    pub reason: String,
    #[serde(skip, default = "Instant::now")]
    pub timestamp: Instant,
}

#[derive(Debug)]
/// Experience replay buffer
pub struct ExperienceBuffer {
    /// All experiences
    experiences: Arc<RwLock<VecDeque<Experience>>>,

    /// Experiences by type
    by_type: Arc<RwLock<HashMap<String, Vec<String>>>>, // Type -> Experience IDs

    /// Pattern index
    patterns: Arc<RwLock<HashMap<PatternType, Vec<PatternObservation>>>>,

    /// Configuration
    max_size: usize,
    prioritized: bool,
}

impl ExperienceBuffer {
    pub fn new(max_size: usize, prioritized: bool) -> Self {
        Self {
            experiences: Arc::new(RwLock::new(VecDeque::with_capacity(max_size))),
            by_type: Arc::new(RwLock::new(HashMap::new())),
            patterns: Arc::new(RwLock::new(HashMap::new())),
            max_size,
            prioritized,
        }
    }

    /// Add an experience
    pub async fn add_experience(&self, experience: Experience) -> Result<()> {
        let mut experiences = self.experiences.write().await;

        // Remove oldest if at capacity
        if experiences.len() >= self.max_size {
            experiences.pop_front();
        }

        // Index by type
        let type_key = match &experience.experience_type {
            ExperienceType::Decision(_) => "decision",
            ExperienceType::Goal(_) => "goal",
            ExperienceType::Plan(_) => "plan",
            ExperienceType::Action(_) => "action",
        };

        self.by_type
            .write()
            .await
            .entry(type_key.to_string())
            .or_insert_with(Vec::new)
            .push(experience.id.clone());

        // Extract patterns
        for pattern in &experience.patterns {
            self.patterns
                .write()
                .await
                .entry(pattern.pattern_type.clone())
                .or_insert_with(Vec::new)
                .push(pattern.clone());
        }

        experiences.push_back(experience);

        Ok(())
    }

    /// Sample experiences for replay
    pub async fn sample_batch(&self, batch_size: usize) -> Vec<Experience> {
        let experiences = self.experiences.read().await;

        if self.prioritized {
            // Prioritize recent and high-importance experiences
            let mut scored: Vec<(f32, &Experience)> = experiences
                .iter()
                .map(|exp| {
                    let recency =
                        1.0 - (exp.timestamp.elapsed().as_secs() as f32 / 86400.0).min(1.0);
                    let importance = exp
                        .lessons
                        .iter()
                        .map(|l| l.importance)
                        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                        .unwrap_or(0.5);
                    let score = recency * 0.3 + importance * 0.7;
                    (score, exp)
                })
                .collect();

            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

            scored.into_iter().take(batch_size).map(|(_, exp)| exp.clone()).collect()
        } else {
            // Random sampling
            experiences.iter().take(batch_size).cloned().collect()
        }
    }
}

#[derive(Debug)]
/// Skill tree manager
pub struct SkillTree {
    /// All skills
    skills: Arc<RwLock<HashMap<SkillId, Skill>>>,

    /// Skill relationships
    dependencies: Arc<RwLock<HashMap<SkillId, Vec<SkillId>>>>,

    /// Active skills
    active_skills: Arc<RwLock<HashSet<SkillId>>>,

    /// Skill progress tracking
    progress: Arc<RwLock<HashMap<SkillId, SkillProgress>>>,
}

#[derive(Clone, Debug)]
struct SkillProgress {
    experiences_used: Vec<String>,
    total_xp: u64,
    level_ups: Vec<Instant>,
}

impl SkillTree {
    pub fn new() -> Self {
        Self {
            skills: Arc::new(RwLock::new(HashMap::new())),
            dependencies: Arc::new(RwLock::new(HashMap::new())),
            active_skills: Arc::new(RwLock::new(HashSet::new())),
            progress: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Initialize default skill tree
    pub async fn initialize_defaults(&self) -> Result<()> {
        // Decision-making skills
        self.add_skill(Skill {
            id: SkillId::new("quick_decisions"),
            name: "Quick Decision Making".to_string(),
            description: "Make faster decisions without losing quality".to_string(),
            category: SkillCategory::DecisionMaking,
            level: SkillLevel { current: 0, progress: 0.0, mastery: false },
            experience_points: 0,
            prerequisites: vec![],
            unlocks: vec![SkillId::new("parallel_evaluation")],
            bonuses: SkillBonuses { decision_speed: 1.2, ..Default::default() },
            acquired_at: None,
            last_improved: Instant::now(),
        })
        .await?;

        // Planning skills
        self.add_skill(Skill {
            id: SkillId::new("efficient_planning"),
            name: "Efficient Planning".to_string(),
            description: "Create more efficient plans with fewer resources".to_string(),
            category: SkillCategory::Planning,
            level: SkillLevel { current: 0, progress: 0.0, mastery: false },
            experience_points: 0,
            prerequisites: vec![],
            unlocks: vec![SkillId::new("adaptive_planning")],
            bonuses: SkillBonuses { resource_efficiency: 0.15, ..Default::default() },
            acquired_at: None,
            last_improved: Instant::now(),
        })
        .await?;

        // Pattern recognition skills
        self.add_skill(Skill {
            id: SkillId::new("pattern_mastery"),
            name: "Pattern Recognition Mastery".to_string(),
            description: "Identify complex patterns in experiences".to_string(),
            category: SkillCategory::PatternRecognition,
            level: SkillLevel { current: 0, progress: 0.0, mastery: false },
            experience_points: 0,
            prerequisites: vec![],
            unlocks: vec![SkillId::new("predictive_modeling")],
            bonuses: SkillBonuses { pattern_detection: 1.5, ..Default::default() },
            acquired_at: None,
            last_improved: Instant::now(),
        })
        .await?;

        Ok(())
    }

    /// Add a skill to the tree
    async fn add_skill(&self, skill: Skill) -> Result<()> {
        let skill_id = skill.id.clone();
        let dependencies = skill.prerequisites.clone();

        self.skills.write().await.insert(skill_id.clone(), skill);
        self.dependencies.write().await.insert(skill_id.clone(), dependencies);

        self.progress.write().await.insert(
            skill_id,
            SkillProgress { experiences_used: Vec::new(), total_xp: 0, level_ups: Vec::new() },
        );

        Ok(())
    }

    /// Gain experience for a skill
    pub async fn gain_experience(
        &self,
        skill_id: &SkillId,
        xp: u64,
        experience_id: String,
    ) -> Result<()> {
        let mut skills = self.skills.write().await;
        let mut progress = self.progress.write().await;

        if let Some(skill) = skills.get_mut(skill_id) {
            skill.experience_points += xp;
            skill.last_improved = Instant::now();

            // Check for level up
            let xp_for_next_level = 100 * (skill.level.current + 1) as u64;
            if skill.experience_points >= xp_for_next_level {
                skill.level.current += 1;
                skill.level.progress = 0.0;

                if skill.acquired_at.is_none() {
                    skill.acquired_at = Some(Instant::now());
                    self.active_skills.write().await.insert(skill_id.clone());
                }

                // Check for mastery
                if skill.level.current >= 10 {
                    skill.level.mastery = true;
                }

                info!("Skill {} leveled up to {}", skill.name, skill.level.current);
            } else {
                skill.level.progress = (skill.experience_points % 100) as f32 / 100.0;
            }

            // Update progress tracking
            if let Some(prog) = progress.get_mut(skill_id) {
                prog.experiences_used.push(experience_id);
                prog.total_xp += xp;
                if skill.experience_points >= xp_for_next_level {
                    prog.level_ups.push(Instant::now());
                }
            }
        }

        Ok(())
    }

    /// Get total skill bonuses
    pub async fn get_active_bonuses(&self) -> SkillBonuses {
        let skills = self.skills.read().await;
        let active = self.active_skills.read().await;

        // Initialize with proper base values for multipliers and additive bonuses
        let mut total = SkillBonuses {
            decision_speed: 1.0,      // Start at 1.0 for multipliers
            confidence_boost: 0.0,    // Start at 0.0 for additive bonuses
            resource_efficiency: 0.0, // Start at 0.0 for additive bonuses
            pattern_detection: 1.0,   // Start at 1.0 for multipliers
            adaptation_rate: 1.0,     // Start at 1.0 for multipliers
        };

        for skill_id in active.iter() {
            if let Some(skill) = skills.get(skill_id) {
                let level_mult = 1.0 + (skill.level.current as f32 * 0.1);

                total.decision_speed *= skill.bonuses.decision_speed * level_mult;
                total.confidence_boost += skill.bonuses.confidence_boost * level_mult;
                total.resource_efficiency += skill.bonuses.resource_efficiency * level_mult;
                total.pattern_detection *= skill.bonuses.pattern_detection * level_mult;
                total.adaptation_rate *= skill.bonuses.adaptation_rate * level_mult;
            }
        }

        // Normalize multipliers
        total.decision_speed = total.decision_speed.max(1.0);
        total.pattern_detection = total.pattern_detection.max(1.0);
        total.adaptation_rate = total.adaptation_rate.max(1.0);

        total
    }
}

#[derive(Debug)]
/// Transfer learning manager
pub struct TransferLearner {
    /// Learning domains
    domains: Arc<RwLock<HashMap<String, LearningDomain>>>,

    /// Domain similarity matrix
    similarity_matrix: Arc<RwLock<HashMap<(String, String), f32>>>,

    /// Transfer history
    transfers: Arc<RwLock<Vec<TransferEvent>>>,
}

#[derive(Clone, Debug)]
struct TransferEvent {
    from_domain: String,
    to_domain: String,
    skills_transferred: Vec<SkillId>,
    effectiveness: f32,
    timestamp: Instant,
}

impl TransferLearner {
    pub fn new() -> Self {
        Self {
            domains: Arc::new(RwLock::new(HashMap::new())),
            similarity_matrix: Arc::new(RwLock::new(HashMap::new())),
            transfers: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Identify transferable knowledge
    pub async fn identify_transfers(&self, from: &str, to: &str) -> Vec<(SkillId, f32)> {
        let domains = self.domains.read().await;
        let similarity = self.get_domain_similarity(from, to).await;

        if let (Some(from_domain), Some(_to_domain)) = (domains.get(from), domains.get(to)) {
            from_domain
                .transferable_skills
                .iter()
                .map(|skill| (skill.clone(), similarity))
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Calculate domain similarity
    async fn get_domain_similarity(&self, domain1: &str, domain2: &str) -> f32 {
        let key = if domain1 < domain2 {
            (domain1.to_string(), domain2.to_string())
        } else {
            (domain2.to_string(), domain1.to_string())
        };

        self.similarity_matrix.read().await.get(&key).cloned().unwrap_or(0.0)
    }
}

#[derive(Debug)]
/// Meta-learning optimizer
pub struct MetaLearner {
    /// Active strategies
    strategies: Arc<RwLock<HashMap<String, MetaStrategy>>>,

    /// Strategy performance history
    performance: Arc<RwLock<HashMap<String, StrategyPerformance>>>,

    /// Current meta-parameters
    meta_params: Arc<RwLock<MetaParameters>>,
}

impl MetaLearner {
    pub fn new() -> Self {
        Self {
            strategies: Arc::new(RwLock::new(HashMap::new())),
            performance: Arc::new(RwLock::new(HashMap::new())),
            meta_params: Arc::new(RwLock::new(MetaParameters::default())),
        }
    }
}

#[derive(Clone, Debug)]
struct StrategyPerformance {
    successes: u32,
    failures: u32,
    avg_effectiveness: f32,
    contexts_used: HashSet<String>,
}

#[derive(Clone, Debug)]
struct MetaParameters {
    exploration_rate: f32,      // 0.0 to 1.0
    learning_rate: f32,         // 0.0 to 1.0
    adaptation_threshold: f32,  // When to adapt strategies
    strategy_timeout: Duration, // How long to try a strategy
}

impl Default for MetaParameters {
    fn default() -> Self {
        Self {
            exploration_rate: 0.2,
            learning_rate: 0.1,
            adaptation_threshold: 0.3,
            strategy_timeout: Duration::from_secs(3600),
        }
    }
}

/// Configuration for decision learner
#[derive(Clone, Debug)]
pub struct LearnerConfig {
    /// Experience buffer size
    pub buffer_size: usize,

    /// Use prioritized replay
    pub prioritized_replay: bool,

    /// Learning update interval
    pub update_interval: Duration,

    /// Pattern detection threshold
    pub pattern_threshold: f32,

    /// Skill XP multiplier
    pub xp_multiplier: f32,

    /// Enable transfer learning
    pub enable_transfer: bool,

    /// Enable meta-learning
    pub enable_meta: bool,
}

impl Default for LearnerConfig {
    fn default() -> Self {
        Self {
            buffer_size: 10000,
            prioritized_replay: true,
            update_interval: Duration::from_secs(300), // 5 minutes
            pattern_threshold: 0.7,
            xp_multiplier: 1.0,
            enable_transfer: true,
            enable_meta: true,
        }
    }
}

#[derive(Debug)]
/// Main decision learner
pub struct DecisionLearner {
    /// Experience buffer
    experience_buffer: Arc<ExperienceBuffer>,

    /// Skill tree
    skill_tree: Arc<SkillTree>,

    /// Transfer learner
    transfer_learner: Arc<TransferLearner>,

    /// Meta-learner
    meta_learner: Arc<MetaLearner>,

    /// Neural processor reference
    neural_processor: Arc<NeuroProcessor>,

    /// Memory system reference
    memory: Arc<CognitiveMemory>,

    /// Decision engine reference
    decision_engine: Arc<DecisionEngine>,

    /// Configuration
    config: LearnerConfig,

    /// Update channel
    update_tx: mpsc::Sender<LearningUpdate>,

    /// Statistics
    stats: Arc<RwLock<LearnerStats>>,
}

#[derive(Clone, Debug)]
pub enum LearningUpdate {
    ExperienceAdded(Experience),
    PatternDetected(PatternObservation),
    SkillLevelUp(SkillId, u32),
    TransferCompleted(String, String), // from, to
    StrategyAdapted(String),           // strategy_id
}

#[derive(Debug, Default, Clone)]
pub struct LearnerStats {
    pub total_experiences: u64,
    pub patterns_detected: u64,
    pub skills_acquired: u64,
    pub transfers_completed: u64,
    pub strategies_adapted: u64,
    pub avg_learning_rate: f32,
}

impl DecisionLearner {
    pub async fn new(
        neural_processor: Arc<NeuroProcessor>,
        memory: Arc<CognitiveMemory>,
        decision_engine: Arc<DecisionEngine>,
        config: LearnerConfig,
    ) -> Result<Self> {
        info!("Initializing Decision Learner");

        let (update_tx, _) = mpsc::channel(100);

        let experience_buffer =
            Arc::new(ExperienceBuffer::new(config.buffer_size, config.prioritized_replay));

        let skill_tree = Arc::new(SkillTree::new());
        skill_tree.initialize_defaults().await?;

        let transfer_learner = Arc::new(TransferLearner::new());
        let meta_learner = Arc::new(MetaLearner::new());

        Ok(Self {
            experience_buffer,
            skill_tree,
            transfer_learner,
            meta_learner,
            neural_processor,
            memory,
            decision_engine,
            config,
            update_tx,
            stats: Arc::new(RwLock::new(LearnerStats::default())),
        })
    }

    /// Create a minimal decision learner for bootstrapping
    pub async fn new_minimal() -> Result<Self> {
        let memory = crate::memory::CognitiveMemory::new_minimal().await?;
        let neural_processor = Arc::new(
            crate::cognitive::NeuroProcessor::new(Arc::new(
                crate::memory::simd_cache::SimdSmartCache::new(
                    crate::memory::SimdCacheConfig::default(),
                ),
            ))
            .await?,
        );
        let emotional_core = Arc::new(
            crate::cognitive::EmotionalCore::new(
                memory.clone(),
                crate::cognitive::EmotionalConfig::default(),
            )
            .await?,
        );
        let character = Arc::new(crate::cognitive::LokiCharacter::new_minimal().await?);
        let tool_manager = Arc::new(crate::tools::IntelligentToolManager::new_minimal().await?);
        let safety_validator = Arc::new(crate::safety::ActionValidator::new_minimal().await?);
        let decision_engine = Arc::new(
            crate::cognitive::DecisionEngine::new(
                neural_processor.clone(),
                emotional_core.clone(),
                memory.clone(),
                character.clone(),
                tool_manager.clone(),
                safety_validator.clone(),
                crate::cognitive::DecisionConfig::default(),
            )
            .await?,
        );
        let config = LearnerConfig::default();

        Self::new(neural_processor, memory, decision_engine, config).await
    }

    /// Start the learning loop
    pub async fn start(self: Arc<Self>) -> Result<()> {
        info!("Starting Decision Learner");

        // Experience replay loop
        {
            let learner = self.clone();
            tokio::spawn(async move {
                learner.replay_loop().await;
            });
        }

        // Pattern detection loop
        {
            let learner = self.clone();
            tokio::spawn(async move {
                learner.pattern_detection_loop().await;
            });
        }

        // Meta-learning loop
        if self.config.enable_meta {
            let learner = self.clone();
            tokio::spawn(async move {
                learner.meta_learning_loop().await;
            });
        }

        Ok(())
    }

    /// Add experience from a decision
    pub async fn add_decision_experience(
        &self,
        decision: &Decision,
        outcome: ActualOutcome,
    ) -> Result<()> {
        let experience = self.create_decision_experience(decision, outcome).await?;

        // Extract lessons
        let lessons = self.extract_lessons(&experience).await?;

        // Detect patterns
        let patterns = self.detect_patterns(&experience).await?;

        let mut complete_experience = experience;
        complete_experience.lessons = lessons;
        complete_experience.patterns = patterns;

        // Add to buffer
        self.experience_buffer.add_experience(complete_experience.clone()).await?;

        // Update skills
        self.update_skills_from_experience(&complete_experience).await?;

        // Send update
        let _ = self.update_tx.send(LearningUpdate::ExperienceAdded(complete_experience)).await;

        // Update stats
        self.stats.write().await.total_experiences += 1;

        Ok(())
    }

    /// Create experience from decision
    async fn create_decision_experience(
        &self,
        decision: &Decision,
        outcome: ActualOutcome,
    ) -> Result<Experience> {
        let decision_snapshot = DecisionSnapshot {
            options: decision.options.clone(),
            criteria: Vec::new(), // Would need to get from decision engine
            selected: decision.selected.as_ref().map(|o| o.id.clone()).unwrap_or_default(),
            confidence: decision.confidence,
            emotional_state: "neutral".to_string(), // Would get from emotional core
        };

        let experience_outcome = ExperienceOutcome {
            success: outcome.success_rate > 0.7,
            score: outcome.success_rate * 2.0 - 1.0, // Convert to -1 to 1
            unexpected_effects: outcome.unexpected_consequences,
            time_taken: decision.decision_time,
            resource_usage: 0.5, // Would calculate actual usage
        };

        Ok(Experience {
            id: uuid::Uuid::new_v4().to_string(),
            experience_type: ExperienceType::Decision(decision.id.clone()),
            context: decision.context.clone(),
            decision: Some(decision_snapshot),
            outcome: experience_outcome,
            lessons: Vec::new(),  // Will be filled
            patterns: Vec::new(), // Will be filled
            timestamp: Instant::now(),
        })
    }

    /// Extract lessons from experience
    async fn extract_lessons(&self, experience: &Experience) -> Result<Vec<LessonLearned>> {
        let mut lessons = Vec::new();

        // Success/failure lessons
        if experience.outcome.success {
            lessons.push(LessonLearned {
                lesson_type: LessonType::StrategySuccess,
                description: format!("Strategy worked well in context: {}", experience.context),
                importance: experience.outcome.score.abs(),
                applicable_contexts: vec![experience.context.clone()],
            });
        } else {
            lessons.push(LessonLearned {
                lesson_type: LessonType::StrategyFailure,
                description: format!("Strategy failed in context: {}", experience.context),
                importance: 1.0 - experience.outcome.score,
                applicable_contexts: vec![experience.context.clone()],
            });
        }

        // Resource optimization lessons
        if experience.outcome.resource_usage < 0.5 {
            lessons.push(LessonLearned {
                lesson_type: LessonType::ResourceOptimization,
                description: "Efficient resource usage".to_string(),
                importance: 0.6,
                applicable_contexts: vec![experience.context.clone()],
            });
        }

        Ok(lessons)
    }

    /// Detect patterns in experience
    async fn detect_patterns(&self, experience: &Experience) -> Result<Vec<PatternObservation>> {
        let mut patterns = Vec::new();

        // Use neural processor for thought activation
        let thought = Thought {
            id: crate::cognitive::ThoughtId::new(),
            content: format!("Analyzing patterns in experience: {}", experience.context),
            thought_type: ThoughtType::Analysis,
            ..Default::default()
        };

        let activation_strength = self.neural_processor.process_thought(&thought).await?;

        // If activation is strong, we likely have a pattern
        if activation_strength > 0.5 {
            let pattern_type = if experience.outcome.success {
                PatternType::SuccessPattern
            } else {
                PatternType::FailurePattern
            };

            patterns.push(PatternObservation {
                pattern_type,
                description: format!(
                    "Pattern detected in {}: activation={:.2}",
                    experience.context, activation_strength
                ),
                frequency: 1,
                confidence: activation_strength,
                examples: vec![experience.id.clone()],
            });
        }

        // Check for resource patterns
        if experience.outcome.resource_usage < 0.3 || experience.outcome.resource_usage > 0.8 {
            patterns.push(PatternObservation {
                pattern_type: PatternType::ResourcePattern,
                description: format!(
                    "Resource usage pattern: {:.1}%",
                    experience.outcome.resource_usage * 100.0
                ),
                frequency: 1,
                confidence: 0.7,
                examples: vec![experience.id.clone()],
            });
        }

        // Check for temporal patterns
        if experience.outcome.time_taken < Duration::from_secs(1) {
            patterns.push(PatternObservation {
                pattern_type: PatternType::TemporalPattern,
                description: "Quick decision pattern".to_string(),
                frequency: 1,
                confidence: 0.8,
                examples: vec![experience.id.clone()],
            });
        }

        Ok(patterns)
    }

    /// Update skills based on experience
    async fn update_skills_from_experience(&self, experience: &Experience) -> Result<()> {
        let xp_gain = (experience.outcome.score.abs() * 100.0 * self.config.xp_multiplier) as u64;

        // Determine which skills to update
        match &experience.experience_type {
            ExperienceType::Decision(_) => {
                self.skill_tree
                    .gain_experience(
                        &SkillId::new("quick_decisions"),
                        xp_gain,
                        experience.id.clone(),
                    )
                    .await?;
            }
            ExperienceType::Plan(_) => {
                self.skill_tree
                    .gain_experience(
                        &SkillId::new("efficient_planning"),
                        xp_gain,
                        experience.id.clone(),
                    )
                    .await?;
            }
            _ => {}
        }

        // Pattern recognition XP for all experiences
        if !experience.patterns.is_empty() {
            self.skill_tree
                .gain_experience(
                    &SkillId::new("pattern_mastery"),
                    xp_gain / 2,
                    experience.id.clone(),
                )
                .await?;
        }

        Ok(())
    }

    /// Experience replay loop
    async fn replay_loop(&self) {
        let mut interval = interval(self.config.update_interval);

        loop {
            interval.tick().await;

            if let Err(e) = self.replay_experiences().await {
                warn!("Experience replay error: {}", e);
            }
        }
    }

    /// Replay experiences for learning
    async fn replay_experiences(&self) -> Result<()> {
        let batch = self.experience_buffer.sample_batch(32).await;

        if batch.is_empty() {
            return Ok(());
        }

        debug!("Replaying {} experiences", batch.len());

        // Group by pattern types
        let mut pattern_groups: HashMap<PatternType, Vec<&Experience>> = HashMap::new();

        for exp in &batch {
            for pattern in &exp.patterns {
                pattern_groups
                    .entry(pattern.pattern_type.clone())
                    .or_insert_with(Vec::new)
                    .push(exp);
            }
        }

        // Learn from patterns
        for (pattern_type, experiences) in pattern_groups {
            self.learn_from_pattern_group(pattern_type, experiences).await?;
        }

        Ok(())
    }

    /// Learn from a group of similar patterns
    async fn learn_from_pattern_group(
        &self,
        pattern_type: PatternType,
        experiences: Vec<&Experience>,
    ) -> Result<()> {
        // This would implement specific learning algorithms
        // For now, we just update stats

        if experiences.len() >= 3 {
            self.stats.write().await.patterns_detected += 1;

            let pattern = PatternObservation {
                pattern_type,
                description: format!("Recurring pattern in {} experiences", experiences.len()),
                frequency: experiences.len() as u32,
                confidence: 0.8,
                examples: experiences.iter().map(|e| e.id.clone()).collect(),
            };

            let _ = self.update_tx.send(LearningUpdate::PatternDetected(pattern)).await;
        }

        Ok(())
    }

    /// Pattern detection loop
    async fn pattern_detection_loop(&self) {
        let mut interval = interval(Duration::from_secs(600)); // 10 minutes

        loop {
            interval.tick().await;

            if let Err(e) = self.detect_global_patterns().await {
                warn!("Pattern detection error: {}", e);
            }
        }
    }

    /// Detect patterns across all experiences
    async fn detect_global_patterns(&self) -> Result<()> {
        // This would implement sophisticated pattern detection
        // For now, it's a placeholder
        debug!("Running global pattern detection");

        Ok(())
    }

    /// Meta-learning loop
    async fn meta_learning_loop(&self) {
        let mut interval = interval(Duration::from_secs(1800)); // 30 minutes

        loop {
            interval.tick().await;

            if let Err(e) = self.optimize_meta_parameters().await {
                warn!("Meta-learning error: {}", e);
            }
        }
    }

    /// Optimize meta-learning parameters
    async fn optimize_meta_parameters(&self) -> Result<()> {
        let mut meta_params = self.meta_learner.meta_params.write().await;

        // Adaptive exploration rate
        let stats = self.stats.read().await;
        if stats.total_experiences > 1000 {
            // Reduce exploration as we gain experience
            meta_params.exploration_rate *= 0.99;
            meta_params.exploration_rate = meta_params.exploration_rate.max(0.05);
        }

        // Adaptive learning rate
        if stats.avg_learning_rate > 0.8 {
            // Learning well, can increase rate
            meta_params.learning_rate *= 1.01;
            meta_params.learning_rate = meta_params.learning_rate.min(0.5);
        }

        debug!(
            "Meta-parameters updated: exploration={:.3}, learning={:.3}",
            meta_params.exploration_rate, meta_params.learning_rate
        );

        Ok(())
    }

    /// Get current skill bonuses
    pub async fn get_skill_bonuses(&self) -> SkillBonuses {
        self.skill_tree.get_active_bonuses().await
    }

    /// Get learner statistics
    pub async fn get_stats(&self) -> LearnerStats {
        self.stats.read().await.clone()
    }

    /// Evaluate decision outcome
    pub async fn evaluate_decision_outcome(&self, decision_id: &DecisionId) -> Result<()> {
        // This would normally look up the original decision and evaluate its outcome
        // For now, we'll create a simulated evaluation

        info!("Evaluating outcome for decision: {}", decision_id);

        // Create a simulated outcome based on random success
        let success_rate = 0.7; // In reality, would calculate from actual results

        let outcome = ActualOutcome {
            decision_id: decision_id.clone(),
            success_rate,
            unexpected_consequences: if success_rate < 0.5 {
                vec!["Outcome was not as expected".to_string()]
            } else {
                vec![]
            },
            learning_points: vec![format!(
                "Decision {} evaluated with {:.0}% success",
                decision_id,
                success_rate * 100.0
            )],
        };

        // Store evaluation in memory
        self.memory
            .store(
                format!(
                    "Decision {} outcome evaluated: {:.0}% success",
                    decision_id,
                    success_rate * 100.0
                ),
                outcome.learning_points.clone(),
                crate::memory::MemoryMetadata {
                    source: "decision_evaluation".to_string(),
                    tags: vec![
                        "learning".to_string(),
                        "decision".to_string(),
                        "evaluation".to_string(),
                    ],
                    importance: success_rate,
                    associations: vec![],
                    context: Some("decision outcome evaluation".to_string()),
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

        Ok(())
    }

    /// Get recent experiences for cognitive synthesis
    /// Returns the most recent experiences sorted by timestamp
    pub async fn get_recent_experiences(&self, limit: usize) -> Vec<Experience> {
        let experiences = self.experience_buffer.experiences.read().await;

        // Get the most recent experiences
        let mut recent: Vec<Experience> = experiences
            .iter()
            .rev() // Reverse to get most recent first
            .take(limit)
            .cloned()
            .collect();

        // Sort by timestamp (most recent first)
        recent.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));

        recent
    }

    /// Get recent decision experiences specifically
    /// Filters for only decision-type experiences
    pub async fn get_recent_decision_experiences(&self, limit: usize) -> Vec<Experience> {
        let experiences = self.experience_buffer.experiences.read().await;

        let mut recent_decisions: Vec<Experience> = experiences
            .iter()
            .filter(|exp| matches!(exp.experience_type, ExperienceType::Decision(_)))
            .rev()
            .take(limit)
            .cloned()
            .collect();

        recent_decisions.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));

        recent_decisions
    }

    /// Get experiences by type with time filtering
    pub async fn get_experiences_by_type_recent(
        &self,
        experience_type: &str,
        limit: usize,
        max_age: Duration,
    ) -> Vec<Experience> {
        let experiences = self.experience_buffer.experiences.read().await;
        let cutoff_time = Instant::now() - max_age;

        let mut filtered: Vec<Experience> = experiences
            .iter()
            .filter(|exp| {
                exp.timestamp >= cutoff_time
                    && match (experience_type, &exp.experience_type) {
                        ("decision", ExperienceType::Decision(_)) => true,
                        ("goal", ExperienceType::Goal(_)) => true,
                        ("plan", ExperienceType::Plan(_)) => true,
                        ("action", ExperienceType::Action(_)) => true,
                        _ => false,
                    }
            })
            .rev()
            .take(limit)
            .cloned()
            .collect();

        filtered.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));

        filtered
    }

    /// Get experience statistics for reflection
    pub async fn get_experience_stats(&self) -> HashMap<String, u64> {
        let experiences = self.experience_buffer.experiences.read().await;
        let mut stats = HashMap::new();

        for exp in experiences.iter() {
            let type_key = match &exp.experience_type {
                ExperienceType::Decision(_) => "decisions",
                ExperienceType::Goal(_) => "goals",
                ExperienceType::Plan(_) => "plans",
                ExperienceType::Action(_) => "actions",
            };

            *stats.entry(type_key.to_string()).or_insert(0) += 1;
        }

        stats.insert("total".to_string(), experiences.len() as u64);
        stats
    }
    
    /// Get total number of experiences
    pub async fn get_total_experiences(&self) -> usize {
        let experiences = self.experience_buffer.experiences.read().await;
        experiences.len()
    }
}

impl PartialEq for PatternType {
    fn eq(&self, other: &Self) -> bool {
        matches!(
            (self, other),
            (PatternType::SuccessPattern, PatternType::SuccessPattern)
                | (PatternType::FailurePattern, PatternType::FailurePattern)
                | (PatternType::ContextPattern, PatternType::ContextPattern)
                | (PatternType::TemporalPattern, PatternType::TemporalPattern)
                | (PatternType::ResourcePattern, PatternType::ResourcePattern)
                | (PatternType::EmotionalPattern, PatternType::EmotionalPattern)
        )
    }
}

impl Eq for PatternType {}

impl std::hash::Hash for PatternType {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            PatternType::SuccessPattern => 0.hash(state),
            PatternType::FailurePattern => 1.hash(state),
            PatternType::ContextPattern => 2.hash(state),
            PatternType::TemporalPattern => 3.hash(state),
            PatternType::ResourcePattern => 4.hash(state),
            PatternType::EmotionalPattern => 5.hash(state),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_skill_progression() {
        let skill_tree = SkillTree::new();
        skill_tree.initialize_defaults().await.unwrap();

        // Test XP gain
        let skill_id = SkillId::new("quick_decisions");
        skill_tree.gain_experience(&skill_id, 150, "test_exp_1".to_string()).await.unwrap();

        let bonuses = skill_tree.get_active_bonuses().await;
        assert!(bonuses.decision_speed > 1.0);
    }
}
