//! Cognitive Modules
//!
//! Implements the cognitive modules that form the building blocks of the
//! adaptive architecture. Modules can be dynamically loaded, configured,
//! and specialized based on task requirements.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};
use super::CognitiveFunction;

/// Types of cognitive modules
#[derive(Clone, Debug, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum ModuleType {
    Analyzer,
    Synthesizer,
    PatternRecognizer,
    DecisionMaker,
    MemoryInterface,
    LearningModule,
    MetaMonitor,
    CreativeProcessor,
    Coordinator,
    RecursiveProcessor,
    Specialized(String),
}

/// Capabilities that modules can provide
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ModuleCapability {
    /// Analysis capabilities
    StructuralAnalysis,
    FunctionalAnalysis,
    CausalAnalysis,
    SemanticAnalysis,

    /// Pattern recognition
    PatternRecognition,
    AnalogyFormation,
    HierarchicalDecomposition,

    /// Synthesis and integration
    InformationIntegration,
    ConceptualSynthesis,
    ConceptualBlending,

    /// Decision making
    DecisionMaking,
    LogicalReasoning,
    StrategySelection,

    /// Learning capabilities
    KnowledgeAcquisition,
    TransferLearning,
    MetaLearning,

    /// Creative capabilities
    CreativeGeneration,
    Brainstorming,
    Innovation,

    /// Meta-cognitive capabilities
    SelfReflection,
    PerformanceMonitoring,
    SelfModification,

    /// Planning and coordination
    PlanGeneration,
    TaskCoordination,
    ResourceManagement,

    /// Memory operations
    MemoryRetrieval,
    MemoryStorage,
    MemoryOrganization,
}

/// Current state of a cognitive module
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ModuleState {
    /// Module is inactive
    Inactive,
    /// Module is initializing
    Initializing,
    /// Module is active and processing
    Active,
    /// Module is being reconfigured
    Reconfiguring,
    /// Module has encountered an error
    Error(String),
    /// Module is being shut down
    ShuttingDown,
}

/// Performance metrics for a module
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct ModulePerformance {
    /// Total processing time (milliseconds)
    pub total_processing_time: u64,

    /// Number of operations completed
    pub operations_completed: u64,

    /// Average processing time per operation
    pub avg_processing_time: f64,

    /// Success rate (0.0 to 1.0)
    pub success_rate: f64,

    /// Error count
    pub error_count: u64,

    /// Resource utilization (0.0 to 1.0)
    pub resource_utilization: f64,

    /// Quality score (0.0 to 1.0)
    pub quality_score: f64,
}

/// Resource requirements for a module
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResourceRequirements {
    /// Memory requirement (MB)
    pub memory_mb: u64,

    /// CPU utilization (0.0 to 1.0)
    pub cpu_utilization: f64,

    /// Network bandwidth (MB/s)
    pub network_bandwidth: f64,

    /// Storage requirement (MB)
    pub storage_mb: u64,

    /// GPU requirement (optional)
    pub gpu_memory_mb: Option<u64>,
}

impl Default for ResourceRequirements {
    fn default() -> Self {
        Self {
            memory_mb: 64,
            cpu_utilization: 0.1,
            network_bandwidth: 1.0,
            storage_mb: 10,
            gpu_memory_mb: None,
        }
    }
}

/// Configuration for a cognitive module
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModuleConfiguration {
    /// Module parameters
    pub parameters: HashMap<String, String>,

    /// Performance thresholds
    pub performance_thresholds: PerformanceThresholds,

    /// Resource limits
    pub resource_limits: ResourceRequirements,

    /// Adaptation settings
    pub adaptation_enabled: bool,

    /// Learning rate for adaptive modules
    pub learning_rate: f64,
}

impl Default for ModuleConfiguration {
    fn default() -> Self {
        Self {
            parameters: HashMap::new(),
            performance_thresholds: PerformanceThresholds::default(),
            resource_limits: ResourceRequirements::default(),
            adaptation_enabled: true,
            learning_rate: 0.01,
        }
    }
}

/// Performance thresholds for module operation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PerformanceThresholds {
    /// Minimum success rate
    pub min_success_rate: f64,

    /// Maximum processing time (milliseconds)
    pub max_processing_time: u64,

    /// Maximum error rate
    pub max_error_rate: f64,

    /// Minimum quality score
    pub min_quality_score: f64,
}

impl Default for PerformanceThresholds {
    fn default() -> Self {
        Self {
            min_success_rate: 0.8,
            max_processing_time: 1000,
            max_error_rate: 0.1,
            min_quality_score: 0.7,
        }
    }
}

/// A cognitive module in the adaptive architecture
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CognitiveModule {
    /// Module identifier
    pub id: String,

    /// Module type
    pub module_type: ModuleType,

    /// Current state
    pub state: ModuleState,

    /// Capabilities provided by this module
    pub capabilities: Vec<ModuleCapability>,

    /// Current cognitive function
    pub cognitive_function: CognitiveFunction,

    /// Module configuration
    pub configuration: ModuleConfiguration,

    /// Performance metrics
    pub performance: ModulePerformance,

    /// Creation timestamp
    pub created_at: DateTime<Utc>,

    /// Last activation timestamp
    pub last_activated: Option<DateTime<Utc>>,

    /// Specialization data (for emergent specialized modules)
    pub specialization_data: Option<SpecializationData>,
}

/// Data for specialized modules that emerged from usage patterns
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SpecializationData {
    /// Pattern that led to specialization
    pub source_pattern: String,

    /// Performance improvement achieved
    pub performance_improvement: f64,

    /// Usage frequency that triggered specialization
    pub trigger_frequency: f64,

    /// Specialization timestamp
    pub specialized_at: DateTime<Utc>,
}

impl CognitiveModule {
    /// Create a new analyzer module
    pub async fn create_analyzer() -> Result<Self> {
        Ok(Self {
            id: uuid::Uuid::new_v4().to_string(),
            module_type: ModuleType::Analyzer,
            state: ModuleState::Inactive,
            capabilities: vec![
                ModuleCapability::StructuralAnalysis,
                ModuleCapability::FunctionalAnalysis,
                ModuleCapability::CausalAnalysis,
            ],
            cognitive_function: CognitiveFunction::Analyzer {
                analysis_type: super::AnalysisType::Structural,
                complexity_threshold: 0.5,
            },
            configuration: ModuleConfiguration::default(),
            performance: ModulePerformance::default(),
            created_at: chrono::Utc::now(),
            last_activated: None,
            specialization_data: None,
        })
    }

    /// Create a new synthesizer module
    pub async fn create_synthesizer() -> Result<Self> {
        Ok(Self {
            id: uuid::Uuid::new_v4().to_string(),
            module_type: ModuleType::Synthesizer,
            state: ModuleState::Inactive,
            capabilities: vec![
                ModuleCapability::InformationIntegration,
                ModuleCapability::ConceptualSynthesis,
                ModuleCapability::ConceptualBlending,
            ],
            cognitive_function: CognitiveFunction::Synthesizer {
                synthesis_method: super::SynthesisMethod::Hierarchical,
                integration_strength: 0.8,
            },
            configuration: ModuleConfiguration::default(),
            performance: ModulePerformance::default(),
            created_at: chrono::Utc::now(),
            last_activated: None,
            specialization_data: None,
        })
    }

    /// Create a new pattern recognizer module
    pub async fn create_pattern_recognizer() -> Result<Self> {
        Ok(Self {
            id: uuid::Uuid::new_v4().to_string(),
            module_type: ModuleType::PatternRecognizer,
            state: ModuleState::Inactive,
            capabilities: vec![
                ModuleCapability::PatternRecognition,
                ModuleCapability::AnalogyFormation,
                ModuleCapability::HierarchicalDecomposition,
            ],
            cognitive_function: CognitiveFunction::PatternRecognizer {
                pattern_types: vec![super::PatternType::Structural, super::PatternType::Semantic],
                recognition_threshold: 0.7,
            },
            configuration: ModuleConfiguration::default(),
            performance: ModulePerformance::default(),
            created_at: chrono::Utc::now(),
            last_activated: None,
            specialization_data: None,
        })
    }

    /// Create a new decision maker module
    pub async fn create_decision_maker() -> Result<Self> {
        Ok(Self {
            id: uuid::Uuid::new_v4().to_string(),
            module_type: ModuleType::DecisionMaker,
            state: ModuleState::Inactive,
            capabilities: vec![
                ModuleCapability::DecisionMaking,
                ModuleCapability::LogicalReasoning,
                ModuleCapability::StrategySelection,
            ],
            cognitive_function: CognitiveFunction::DecisionMaker {
                decision_strategy: super::DecisionStrategy::Hybrid,
                confidence_threshold: 0.7,
            },
            configuration: ModuleConfiguration::default(),
            performance: ModulePerformance::default(),
            created_at: chrono::Utc::now(),
            last_activated: None,
            specialization_data: None,
        })
    }

    /// Create a new memory interface module
    pub async fn create_memory_interface() -> Result<Self> {
        Ok(Self {
            id: uuid::Uuid::new_v4().to_string(),
            module_type: ModuleType::MemoryInterface,
            state: ModuleState::Inactive,
            capabilities: vec![
                ModuleCapability::MemoryRetrieval,
                ModuleCapability::MemoryStorage,
                ModuleCapability::MemoryOrganization,
            ],
            cognitive_function: CognitiveFunction::MemoryInterface {
                memory_type: super::MemoryType::Working,
                access_pattern: super::AccessPattern::Associative,
            },
            configuration: ModuleConfiguration::default(),
            performance: ModulePerformance::default(),
            created_at: chrono::Utc::now(),
            last_activated: None,
            specialization_data: None,
        })
    }

    /// Create a new learning module
    pub async fn create_learning_module() -> Result<Self> {
        Ok(Self {
            id: uuid::Uuid::new_v4().to_string(),
            module_type: ModuleType::LearningModule,
            state: ModuleState::Inactive,
            capabilities: vec![
                ModuleCapability::KnowledgeAcquisition,
                ModuleCapability::TransferLearning,
                ModuleCapability::MetaLearning,
            ],
            cognitive_function: CognitiveFunction::LearningModule {
                learning_type: super::LearningType::Transfer,
                adaptation_rate: 0.1,
            },
            configuration: ModuleConfiguration::default(),
            performance: ModulePerformance::default(),
            created_at: chrono::Utc::now(),
            last_activated: None,
            specialization_data: None,
        })
    }

    /// Create a new meta monitor module
    pub async fn create_meta_monitor() -> Result<Self> {
        Ok(Self {
            id: uuid::Uuid::new_v4().to_string(),
            module_type: ModuleType::MetaMonitor,
            state: ModuleState::Inactive,
            capabilities: vec![
                ModuleCapability::SelfReflection,
                ModuleCapability::PerformanceMonitoring,
                ModuleCapability::SelfModification,
            ],
            cognitive_function: CognitiveFunction::MetaMonitor {
                monitoring_scope: super::MonitoringScope::Global,
                reflection_depth: 3,
            },
            configuration: ModuleConfiguration::default(),
            performance: ModulePerformance::default(),
            created_at: chrono::Utc::now(),
            last_activated: None,
            specialization_data: None,
        })
    }

    /// Create a new creative processor module
    pub async fn create_creative_processor() -> Result<Self> {
        Ok(Self {
            id: uuid::Uuid::new_v4().to_string(),
            module_type: ModuleType::CreativeProcessor,
            state: ModuleState::Inactive,
            capabilities: vec![
                ModuleCapability::CreativeGeneration,
                ModuleCapability::Brainstorming,
                ModuleCapability::Innovation,
            ],
            cognitive_function: CognitiveFunction::CreativeProcessor {
                creativity_domain: super::CreativityDomain::Conceptual,
                innovation_bias: 0.6,
            },
            configuration: ModuleConfiguration::default(),
            performance: ModulePerformance::default(),
            created_at: chrono::Utc::now(),
            last_activated: None,
            specialization_data: None,
        })
    }

    /// Create a default module (analyzer)
    pub async fn create_default() -> Result<Self> {
        Self::create_analyzer().await
    }

    /// Activate the module
    pub async fn activate(&mut self) -> Result<()> {
        self.state = ModuleState::Active;
        self.last_activated = Some(chrono::Utc::now());
        Ok(())
    }

    /// Deactivate the module
    pub async fn deactivate(&mut self) -> Result<()> {
        self.state = ModuleState::Inactive;
        Ok(())
    }

    /// Check if module meets performance requirements
    pub fn meets_performance_requirements(&self) -> bool {
        let thresholds = &self.configuration.performance_thresholds;

        self.performance.success_rate >= thresholds.min_success_rate &&
        self.performance.avg_processing_time <= thresholds.max_processing_time as f64 &&
        self.performance.quality_score >= thresholds.min_quality_score
    }

    /// Update performance metrics
    pub async fn update_performance(&mut self, processing_time: u64, success: bool, quality: f64) -> Result<()> {
        self.performance.operations_completed += 1;
        self.performance.total_processing_time += processing_time;

        if success {
            self.performance.success_rate =
                (self.performance.success_rate * (self.performance.operations_completed - 1) as f64 + 1.0) /
                self.performance.operations_completed as f64;
        } else {
            self.performance.error_count += 1;
            self.performance.success_rate =
                (self.performance.success_rate * (self.performance.operations_completed - 1) as f64) /
                self.performance.operations_completed as f64;
        }

        self.performance.avg_processing_time =
            self.performance.total_processing_time as f64 / self.performance.operations_completed as f64;

        self.performance.quality_score =
            (self.performance.quality_score * (self.performance.operations_completed - 1) as f64 + quality) /
            self.performance.operations_completed as f64;

        Ok(())
    }

    /// Get module efficiency score
    pub fn get_efficiency_score(&self) -> f64 {
        let success_weight = 0.4;
        let quality_weight = 0.3;
        let speed_weight = 0.2;
        let resource_weight = 0.1;

        let speed_score = if self.performance.avg_processing_time > 0.0 {
            (1000.0 / self.performance.avg_processing_time).min(1.0)
        } else {
            1.0
        };

        success_weight * self.performance.success_rate +
        quality_weight * self.performance.quality_score +
        speed_weight * speed_score +
        resource_weight * (1.0 - self.performance.resource_utilization)
    }

    /// Check if module should be specialized
    pub fn should_specialize(&self, usage_frequency: f64, performance_improvement: f64) -> bool {
        usage_frequency > 0.8 && performance_improvement > 0.2 && self.specialization_data.is_none()
    }

    /// Mark module as specialized
    pub async fn mark_as_specialized(&mut self, pattern: String, improvement: f64, frequency: f64) -> Result<()> {
        self.specialization_data = Some(SpecializationData {
            source_pattern: pattern,
            performance_improvement: improvement,
            trigger_frequency: frequency,
            specialized_at: chrono::Utc::now(),
        });

        // Update module type to specialized
        self.module_type = ModuleType::Specialized(self.id.clone());

        Ok(())
    }
}
