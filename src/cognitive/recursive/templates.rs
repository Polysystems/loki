//! Reasoning Templates for Recursive Cognitive Processing
//!
//! Implements reusable reasoning patterns that can be instantiated across
//! different scales and contexts to enable consistent cognitive processing.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use uuid::Uuid;

use super::thoughts::{AtomicOperation, AtomicThoughtUnit, ThoughtUnit};
use super::{RecursionType, RecursiveContext};
use crate::memory::fractal::ScaleLevel;

/// Unique identifier for reasoning templates
#[derive(Clone, Debug, Hash, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub struct TemplateId(String);

impl TemplateId {
    pub fn new() -> Self {
        Self(Uuid::new_v4().to_string())
    }

    pub fn from_name(name: &str) -> Self {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(name.as_bytes());
        let hash = format!("{:?}", hasher.finalize());
        Self(format!("template_{}", &hash[..12]))
    }
}

impl std::fmt::Display for TemplateId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A reusable reasoning pattern template
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReasoningTemplate {
    /// Unique identifier
    pub id: TemplateId,

    /// Human-readable name
    pub name: String,

    /// Description of the reasoning pattern
    pub description: String,

    /// Category of reasoning
    pub category: ReasoningCategory,

    /// Scale levels where this template is applicable
    pub applicable_scales: Vec<ScaleLevel>,

    /// Pattern specification
    pub pattern: ReasoningPattern,

    /// Input requirements
    pub input_requirements: InputRequirements,

    /// Expected output characteristics
    pub output_specification: OutputSpecification,

    /// Performance metrics
    pub metrics: TemplateMetrics,

    /// Metadata
    pub metadata: TemplateMetadata,
}

/// Categories of reasoning patterns
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ReasoningCategory {
    /// Analytical decomposition
    Analysis,
    /// Creative synthesis
    Synthesis,
    /// Logical deduction
    Deduction,
    /// Pattern induction
    Induction,
    /// Abductive reasoning
    Abduction,
    /// Causal reasoning
    Causal,
    /// Analogical reasoning
    Analogical,
    /// Meta-reasoning
    MetaReasoning,
    /// Problem solving
    ProblemSolving,
    /// Decision making
    DecisionMaking,
}

/// Structured reasoning pattern definition
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReasoningPattern {
    /// Pattern type
    pub pattern_type: PatternType,

    /// Steps in the reasoning process
    pub steps: Vec<ReasoningStep>,

    /// Branching conditions
    pub branches: Vec<BranchCondition>,

    /// Loop conditions
    pub loops: Vec<LoopCondition>,

    /// Termination criteria
    pub termination: TerminationCriteria,

    /// Quality checks
    pub quality_checks: Vec<QualityCheck>,
}

/// Types of reasoning patterns
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum PatternType {
    /// Linear sequence of steps
    Linear,
    /// Branching decision tree
    Branching,
    /// Iterative refinement
    Iterative,
    /// Recursive self-application
    Recursive,
    /// Parallel processing
    Parallel,
    /// Hierarchical decomposition
    Hierarchical,
}

/// Single step in a reasoning pattern
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReasoningStep {
    /// Step identifier
    pub id: String,

    /// Step name
    pub name: String,

    /// Operation to perform
    pub operation: StepOperation,

    /// Input transformations
    pub input_transform: Option<String>,

    /// Output transformations
    pub output_transform: Option<String>,

    /// Dependencies on other steps
    pub dependencies: Vec<String>,

    /// Estimated processing time
    pub estimated_time: std::time::Duration,

    /// Resource requirements
    pub resource_requirements: StepResources,
}

/// Operations that can be performed in reasoning steps
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum StepOperation {
    /// Apply transformation function
    Transform(String),
    /// Filter based on criteria
    Filter(String),
    /// Classify or categorize
    Classify(Vec<String>),
    /// Extract specific information
    Extract(String),
    /// Compare multiple inputs
    Compare(String),
    /// Aggregate information
    Aggregate(String),
    /// Generate new content
    Generate(String),
    /// Validate against criteria
    Validate(String),
    /// Recursive application
    RecursiveApply(RecursionType),
}

/// Resource requirements for a reasoning step
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StepResources {
    /// Memory requirement (bytes)
    pub memory: u64,

    /// CPU cycles estimate
    pub cpu_cycles: u64,

    /// Network calls needed
    pub network_calls: u32,

    /// Required external tools
    pub tools: Vec<String>,
}

impl Default for StepResources {
    fn default() -> Self {
        Self {
            memory: 1024 * 1024, // 1MB
            cpu_cycles: 1000,
            network_calls: 0,
            tools: Vec::new(),
        }
    }
}

/// Branching condition in reasoning patterns
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BranchCondition {
    /// Condition expression
    pub condition: String,

    /// Steps to execute if true
    pub true_steps: Vec<String>,

    /// Steps to execute if false
    pub false_steps: Vec<String>,

    /// Default branch if condition is indeterminate
    pub default_branch: Option<String>,
}

/// Loop condition in reasoning patterns
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LoopCondition {
    /// Loop type
    pub loop_type: LoopType,

    /// Continuation condition
    pub condition: String,

    /// Steps to repeat
    pub loop_steps: Vec<String>,

    /// Maximum iterations
    pub max_iterations: u32,

    /// Convergence criteria
    pub convergence: Option<String>,
}

/// Types of loops in reasoning patterns
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum LoopType {
    /// While condition is true
    While,
    /// Until condition is met
    Until,
    /// Fixed number of iterations
    For,
    /// Iterative refinement
    Refinement,
}

/// Termination criteria for reasoning patterns
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TerminationCriteria {
    /// Maximum execution time
    pub max_time: std::time::Duration,

    /// Maximum number of steps
    pub max_steps: u32,

    /// Quality threshold
    pub quality_threshold: f32,

    /// Convergence criteria
    pub convergence_criteria: Vec<String>,

    /// Resource limits
    pub resource_limits: StepResources,
}

/// Quality check for reasoning patterns
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QualityCheck {
    /// Check name
    pub name: String,

    /// Check expression
    pub expression: String,

    /// Minimum acceptable score
    pub threshold: f32,

    /// Whether this check is critical
    pub critical: bool,
}

/// Input requirements for a template
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InputRequirements {
    /// Minimum input length
    pub min_length: usize,

    /// Maximum input length
    pub max_length: usize,

    /// Required input format
    pub format: InputFormat,

    /// Required input qualities
    pub quality_requirements: QualityRequirements,

    /// Required context keys
    pub required_context: Vec<String>,

    /// Optional context keys
    pub optional_context: Vec<String>,
}

/// Input format specifications
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum InputFormat {
    /// Plain text
    Text,
    /// Structured data
    Structured,
    /// JSON format
    Json,
    /// XML format
    Xml,
    /// Custom format
    Custom(String),
}

/// Quality requirements for input
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QualityRequirements {
    /// Minimum overall quality
    pub min_quality: f32,

    /// Minimum coherence
    pub min_coherence: f32,

    /// Minimum relevance
    pub min_relevance: f32,

    /// Required clarity level
    pub min_clarity: f32,
}

impl Default for QualityRequirements {
    fn default() -> Self {
        Self { min_quality: 0.3, min_coherence: 0.3, min_relevance: 0.3, min_clarity: 0.3 }
    }
}

/// Output specification for templates
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OutputSpecification {
    /// Expected output format
    pub format: InputFormat,

    /// Expected length range
    pub length_range: (usize, usize),

    /// Expected quality levels
    pub quality_expectations: QualityRequirements,

    /// Output structure requirements
    pub structure: OutputStructure,
}

/// Output structure requirements
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OutputStructure {
    /// Required sections
    pub required_sections: Vec<String>,

    /// Optional sections
    pub optional_sections: Vec<String>,

    /// Section ordering requirements
    pub ordering: Vec<String>,

    /// Formatting requirements
    pub formatting: HashMap<String, String>,
}

/// Performance metrics for templates
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TemplateMetrics {
    /// Number of times used
    pub usage_count: u64,

    /// Success rate
    pub success_rate: f32,

    /// Average execution time
    pub average_time: std::time::Duration,

    /// Average quality score
    pub average_quality: f32,

    /// Resource efficiency
    pub efficiency: f32,

    /// User satisfaction rating
    pub satisfaction: f32,

    /// Last updated timestamp
    pub last_updated: DateTime<Utc>,
}

impl Default for TemplateMetrics {
    fn default() -> Self {
        Self {
            usage_count: 0,
            success_rate: 0.0,
            average_time: std::time::Duration::from_secs(0),
            average_quality: 0.0,
            efficiency: 0.0,
            satisfaction: 0.0,
            last_updated: Utc::now(),
        }
    }
}

/// Template metadata
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TemplateMetadata {
    /// Author information
    pub author: String,

    /// Version number
    pub version: String,

    /// Creation timestamp
    pub created: DateTime<Utc>,

    /// Last modified timestamp
    pub modified: DateTime<Utc>,

    /// Tags for categorization
    pub tags: Vec<String>,

    /// Dependencies on other templates
    pub dependencies: Vec<TemplateId>,

    /// Related templates
    pub related: Vec<TemplateId>,

    /// Usage examples
    pub examples: Vec<String>,
}

/// Template instantiation parameters
#[derive(Clone, Debug)]
pub struct InstantiationParams {
    /// Scale level for instantiation
    pub scale_level: ScaleLevel,

    /// Context for instantiation
    pub context: RecursiveContext,

    /// Parameter overrides
    pub parameter_overrides: HashMap<String, String>,

    /// Resource constraints
    pub resource_constraints: StepResources,

    /// Quality requirements
    pub quality_requirements: QualityRequirements,
}

/// Result of template instantiation
pub struct InstantiationResult {
    /// Instantiated thought units
    pub thought_units: Vec<Box<dyn ThoughtUnit + Send + Sync>>,

    /// Execution plan
    pub execution_plan: ExecutionPlan,

    /// Resource requirements
    pub resource_requirements: StepResources,

    /// Expected quality
    pub expected_quality: f32,

    /// Estimated execution time
    pub estimated_time: std::time::Duration,
}

// Manual Clone implementation since trait objects don't auto-derive Clone
impl Clone for InstantiationResult {
    fn clone(&self) -> Self {
        let cloned_units: Vec<Box<dyn ThoughtUnit + Send + Sync>> =
            self.thought_units.iter().map(|unit| unit.clone_unit()).collect();

        Self {
            thought_units: cloned_units,
            execution_plan: self.execution_plan.clone(),
            resource_requirements: self.resource_requirements.clone(),
            expected_quality: self.expected_quality,
            estimated_time: self.estimated_time,
        }
    }
}

// Manual Debug implementation since trait objects don't auto-derive Debug
impl std::fmt::Debug for InstantiationResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InstantiationResult")
            .field("thought_units_count", &self.thought_units.len())
            .field("execution_plan", &self.execution_plan)
            .field("resource_requirements", &self.resource_requirements)
            .field("expected_quality", &self.expected_quality)
            .field("estimated_time", &self.estimated_time)
            .finish()
    }
}

/// Execution plan for instantiated template
#[derive(Clone, Debug)]
pub struct ExecutionPlan {
    /// Ordered steps
    pub steps: Vec<ExecutionStep>,

    /// Parallel execution groups
    pub parallel_groups: Vec<Vec<usize>>,

    /// Critical path
    pub critical_path: Vec<usize>,

    /// Checkpoints for monitoring
    pub checkpoints: Vec<usize>,
}

/// Single step in execution plan
#[derive(Clone, Debug)]
pub struct ExecutionStep {
    /// Step index
    pub index: usize,

    /// Thought unit to execute
    pub unit_index: usize,

    /// Input sources
    pub input_sources: Vec<usize>,

    /// Output targets
    pub output_targets: Vec<usize>,

    /// Estimated start time
    pub estimated_start: std::time::Duration,

    /// Estimated duration
    pub estimated_duration: std::time::Duration,
}

/// Template instantiator - converts templates to executable thought units
pub struct TemplateInstantiator {
    /// Template library
    template_library: Arc<TemplateLibrary>,

    /// Instantiation cache
    instantiation_cache: Arc<RwLock<HashMap<(TemplateId, String), InstantiationResult>>>,

    /// Performance tracker
    performance_tracker: Arc<RwLock<HashMap<TemplateId, TemplateMetrics>>>,
}

impl TemplateInstantiator {
    pub async fn new(library: Arc<TemplateLibrary>) -> Result<Self> {
        Ok(Self {
            template_library: library,
            instantiation_cache: Arc::new(RwLock::new(HashMap::new())),
            performance_tracker: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Instantiate a template with given parameters
    pub async fn instantiate(
        &self,
        template_id: &TemplateId,
        params: InstantiationParams,
    ) -> Result<InstantiationResult> {
        // Generate cache key
        let cache_key = (template_id.clone(), format!("{:?}", params.scale_level));

        // Check cache first
        {
            let cache = self.instantiation_cache.read().await;
            if let Some(cached_result) = cache.get(&cache_key) {
                return Ok(cached_result.clone());
            }
        }

        // Get template from library
        let template = self
            .template_library
            .get_template(template_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Template not found: {}", template_id))?;

        // Check if template is applicable to the scale level
        if !template.applicable_scales.contains(&params.scale_level) {
            return Err(anyhow::anyhow!(
                "Template {} not applicable to scale {:?}",
                template_id,
                params.scale_level
            ));
        }

        // Instantiate thought units
        let thought_units = self.create_thought_units(&template, &params).await?;

        // Create execution plan
        let execution_plan = self.create_execution_plan(&template, &thought_units).await?;

        // Calculate resource requirements
        let resource_requirements = self.calculate_resource_requirements(&template, &params);

        // Estimate quality and time
        let expected_quality = self.estimate_quality(&template, &params);
        let estimated_time = self.estimate_execution_time(&template, &params);

        let result = InstantiationResult {
            thought_units,
            execution_plan,
            resource_requirements,
            expected_quality,
            estimated_time,
        };

        // Cache the result
        {
            let mut cache = self.instantiation_cache.write().await;
            cache.insert(cache_key, result.clone());
        }

        Ok(result)
    }

    /// Create thought units from template
    async fn create_thought_units(
        &self,
        template: &ReasoningTemplate,
        params: &InstantiationParams,
    ) -> Result<Vec<Box<dyn ThoughtUnit + Send + Sync>>> {
        let mut units: Vec<Box<dyn ThoughtUnit + Send + Sync>> = Vec::new();

        for step in &template.pattern.steps {
            let operation = match &step.operation {
                StepOperation::Transform(pattern) => AtomicOperation::Transform(pattern.clone()),
                StepOperation::Extract(target) => AtomicOperation::Extract(target.clone()),
                StepOperation::Classify(_) => AtomicOperation::Classify,
                StepOperation::Validate(_) => AtomicOperation::Validate,
                StepOperation::Compare(_) => AtomicOperation::Compare,
                StepOperation::Aggregate(_) => AtomicOperation::Synthesize,
                StepOperation::Generate(_) => AtomicOperation::Synthesize,
                StepOperation::Filter(_) => AtomicOperation::Match("filter".to_string()),
                StepOperation::RecursiveApply(_) => {
                    AtomicOperation::Transform("recursive".to_string())
                }
            };

            let unit = AtomicThoughtUnit::new(operation, params.scale_level);
            units.push(Box::new(unit));
        }

        Ok(units)
    }

    /// Create execution plan from template
    async fn create_execution_plan(
        &self,
        template: &ReasoningTemplate,
        _thought_units: &[Box<dyn ThoughtUnit + Send + Sync>],
    ) -> Result<ExecutionPlan> {
        let mut steps = Vec::new();
        let mut critical_path = Vec::new();
        let mut checkpoints = Vec::new();

        // Create execution steps
        for (index, step) in template.pattern.steps.iter().enumerate() {
            let execution_step = ExecutionStep {
                index,
                unit_index: index,
                input_sources: self
                    .resolve_dependencies(&step.dependencies, &template.pattern.steps),
                output_targets: vec![index + 1], // Simplified
                estimated_start: std::time::Duration::from_secs(0),
                estimated_duration: step.estimated_time,
            };
            steps.push(execution_step);

            // Add to critical path if no dependencies
            if step.dependencies.is_empty() {
                critical_path.push(index);
            }

            // Add checkpoint every 5 steps
            if index % 5 == 0 {
                checkpoints.push(index);
            }
        }

        // Identify parallel execution opportunities
        let parallel_groups = self.identify_parallel_groups(&template.pattern.steps);

        Ok(ExecutionPlan { steps, parallel_groups, critical_path, checkpoints })
    }

    /// Resolve step dependencies to indices
    fn resolve_dependencies(
        &self,
        dependencies: &[String],
        all_steps: &[ReasoningStep],
    ) -> Vec<usize> {
        dependencies
            .iter()
            .filter_map(|dep| all_steps.iter().position(|step| step.id == *dep))
            .collect()
    }

    /// Identify groups of steps that can run in parallel
    fn identify_parallel_groups(&self, steps: &[ReasoningStep]) -> Vec<Vec<usize>> {
        let mut groups = Vec::new();
        let mut current_group = Vec::new();

        for (index, step) in steps.iter().enumerate() {
            if step.dependencies.is_empty() {
                current_group.push(index);
            } else if !current_group.is_empty() {
                groups.push(current_group.clone());
                current_group.clear();
                current_group.push(index);
            }
        }

        if !current_group.is_empty() {
            groups.push(current_group);
        }

        groups
    }

    /// Calculate total resource requirements
    fn calculate_resource_requirements(
        &self,
        template: &ReasoningTemplate,
        _params: &InstantiationParams,
    ) -> StepResources {
        let mut total_memory = 0;
        let mut total_cpu = 0;
        let mut total_network = 0;
        let mut all_tools = Vec::new();

        for step in &template.pattern.steps {
            total_memory += step.resource_requirements.memory;
            total_cpu += step.resource_requirements.cpu_cycles;
            total_network += step.resource_requirements.network_calls;
            all_tools.extend(step.resource_requirements.tools.clone());
        }

        all_tools.sort();
        all_tools.dedup();

        StepResources {
            memory: total_memory,
            cpu_cycles: total_cpu,
            network_calls: total_network,
            tools: all_tools,
        }
    }

    /// Estimate quality of instantiated template
    fn estimate_quality(&self, template: &ReasoningTemplate, _params: &InstantiationParams) -> f32 {
        // Use historical metrics if available
        template.metrics.average_quality.max(0.5)
    }

    /// Estimate execution time
    fn estimate_execution_time(
        &self,
        template: &ReasoningTemplate,
        _params: &InstantiationParams,
    ) -> std::time::Duration {
        // Sum up all step times (simplified)
        template.pattern.steps.iter().map(|step| step.estimated_time).sum()
    }

    /// Record performance metrics
    pub async fn record_performance(
        &self,
        template_id: &TemplateId,
        execution_time: std::time::Duration,
        quality_score: f32,
        success: bool,
    ) -> Result<()> {
        let mut tracker = self.performance_tracker.write().await;
        let metrics = tracker.entry(template_id.clone()).or_insert_with(TemplateMetrics::default);

        metrics.usage_count += 1;

        if success {
            let success_count =
                (metrics.success_rate * (metrics.usage_count - 1) as f32) as u64 + 1;
            metrics.success_rate = success_count as f32 / metrics.usage_count as f32;
        } else {
            let success_count = (metrics.success_rate * (metrics.usage_count - 1) as f32) as u64;
            metrics.success_rate = success_count as f32 / metrics.usage_count as f32;
        }

        // Update averages
        let old_weight = (metrics.usage_count - 1) as f32 / metrics.usage_count as f32;
        let new_weight = 1.0 / metrics.usage_count as f32;

        metrics.average_time = std::time::Duration::from_millis(
            (metrics.average_time.as_millis() as f32 * old_weight
                + execution_time.as_millis() as f32 * new_weight) as u64,
        );

        metrics.average_quality = metrics.average_quality * old_weight + quality_score * new_weight;
        metrics.last_updated = Utc::now();

        Ok(())
    }
}

/// Template library - manages collections of reasoning templates
pub struct TemplateLibrary {
    /// Stored templates
    templates: Arc<RwLock<HashMap<TemplateId, ReasoningTemplate>>>,

    /// Category index
    category_index: Arc<RwLock<HashMap<ReasoningCategory, Vec<TemplateId>>>>,

    /// Scale index
    scale_index: Arc<RwLock<HashMap<ScaleLevel, Vec<TemplateId>>>>,

    /// Tag index
    tag_index: Arc<RwLock<HashMap<String, Vec<TemplateId>>>>,

    /// Usage statistics
    usage_stats: Arc<RwLock<BTreeMap<TemplateId, u64>>>,
}

impl TemplateLibrary {
    pub async fn new() -> Result<Self> {
        let library = Self {
            templates: Arc::new(RwLock::new(HashMap::new())),
            category_index: Arc::new(RwLock::new(HashMap::new())),
            scale_index: Arc::new(RwLock::new(HashMap::new())),
            tag_index: Arc::new(RwLock::new(HashMap::new())),
            usage_stats: Arc::new(RwLock::new(BTreeMap::new())),
        };

        // Initialize with built-in templates
        library.initialize_builtin_templates().await?;

        Ok(library)
    }

    /// Initialize built-in reasoning templates
    async fn initialize_builtin_templates(&self) -> Result<()> {
        // Analytical decomposition template
        let analysis_template = ReasoningTemplate {
            id: TemplateId::from_name("analytical_decomposition"),
            name: "Analytical Decomposition".to_string(),
            description: "Break down complex problems into smaller components".to_string(),
            category: ReasoningCategory::Analysis,
            applicable_scales: vec![ScaleLevel::Concept, ScaleLevel::Schema, ScaleLevel::Worldview],
            pattern: ReasoningPattern {
                pattern_type: PatternType::Hierarchical,
                steps: vec![
                    ReasoningStep {
                        id: "identify".to_string(),
                        name: "Identify Components".to_string(),
                        operation: StepOperation::Extract("components".to_string()),
                        input_transform: None,
                        output_transform: None,
                        dependencies: vec![],
                        estimated_time: std::time::Duration::from_secs(5),
                        resource_requirements: StepResources::default(),
                    },
                    ReasoningStep {
                        id: "analyze".to_string(),
                        name: "Analyze Relationships".to_string(),
                        operation: StepOperation::Compare("relationships".to_string()),
                        input_transform: None,
                        output_transform: None,
                        dependencies: vec!["identify".to_string()],
                        estimated_time: std::time::Duration::from_secs(10),
                        resource_requirements: StepResources::default(),
                    },
                    ReasoningStep {
                        id: "synthesize".to_string(),
                        name: "Synthesize Insights".to_string(),
                        operation: StepOperation::Aggregate("insights".to_string()),
                        input_transform: None,
                        output_transform: None,
                        dependencies: vec!["analyze".to_string()],
                        estimated_time: std::time::Duration::from_secs(8),
                        resource_requirements: StepResources::default(),
                    },
                ],
                branches: vec![],
                loops: vec![],
                termination: TerminationCriteria {
                    max_time: std::time::Duration::from_secs(60),
                    max_steps: 10,
                    quality_threshold: 0.7,
                    convergence_criteria: vec!["stable_output".to_string()],
                    resource_limits: StepResources {
                        memory: 10 * 1024 * 1024, // 10MB
                        cpu_cycles: 100_000,
                        network_calls: 0,
                        tools: vec![],
                    },
                },
                quality_checks: vec![QualityCheck {
                    name: "Completeness".to_string(),
                    expression: "all_components_identified".to_string(),
                    threshold: 0.8,
                    critical: true,
                }],
            },
            input_requirements: InputRequirements {
                min_length: 10,
                max_length: 1000,
                format: InputFormat::Text,
                quality_requirements: QualityRequirements::default(),
                required_context: vec![],
                optional_context: vec!["domain".to_string(), "complexity".to_string()],
            },
            output_specification: OutputSpecification {
                format: InputFormat::Structured,
                length_range: (50, 500),
                quality_expectations: QualityRequirements {
                    min_quality: 0.7,
                    min_coherence: 0.8,
                    min_relevance: 0.9,
                    min_clarity: 0.8,
                },
                structure: OutputStructure {
                    required_sections: vec![
                        "components".to_string(),
                        "relationships".to_string(),
                        "insights".to_string(),
                    ],
                    optional_sections: vec!["recommendations".to_string()],
                    ordering: vec![
                        "components".to_string(),
                        "relationships".to_string(),
                        "insights".to_string(),
                    ],
                    formatting: HashMap::new(),
                },
            },
            metrics: TemplateMetrics::default(),
            metadata: TemplateMetadata {
                author: "Loki System".to_string(),
                version: "1.0".to_string(),
                created: Utc::now(),
                modified: Utc::now(),
                tags: vec![
                    "analysis".to_string(),
                    "decomposition".to_string(),
                    "structure".to_string(),
                ],
                dependencies: vec![],
                related: vec![],
                examples: vec![
                    "Analyze the structure of a complex argument".to_string(),
                    "Break down a system into its components".to_string(),
                ],
            },
        };

        self.add_template(analysis_template).await?;

        Ok(())
    }

    /// Add a new template to the library
    pub async fn add_template(&self, template: ReasoningTemplate) -> Result<()> {
        let template_id = template.id.clone();

        // Store template
        {
            let mut templates = self.templates.write().await;
            templates.insert(template_id.clone(), template.clone());
        }

        // Update category index
        {
            let mut category_index = self.category_index.write().await;
            category_index
                .entry(template.category)
                .or_insert_with(Vec::new)
                .push(template_id.clone());
        }

        // Update scale index
        {
            let mut scale_index = self.scale_index.write().await;
            for scale in &template.applicable_scales {
                scale_index.entry(*scale).or_insert_with(Vec::new).push(template_id.clone());
            }
        }

        // Update tag index
        {
            let mut tag_index = self.tag_index.write().await;
            for tag in &template.metadata.tags {
                tag_index.entry(tag.clone()).or_insert_with(Vec::new).push(template_id.clone());
            }
        }

        Ok(())
    }

    /// Get a template by ID
    pub async fn get_template(&self, id: &TemplateId) -> Result<Option<ReasoningTemplate>> {
        let templates = self.templates.read().await;
        Ok(templates.get(id).cloned())
    }

    /// Find templates by category
    pub async fn find_by_category(
        &self,
        category: ReasoningCategory,
    ) -> Result<Vec<ReasoningTemplate>> {
        let category_index = self.category_index.read().await;
        let templates = self.templates.read().await;

        let empty_vec = vec![];
        let template_ids = category_index.get(&category).unwrap_or(&empty_vec);
        let results = template_ids.iter().filter_map(|id| templates.get(id).cloned()).collect();

        Ok(results)
    }

    /// Find templates by scale level
    pub async fn find_by_scale(&self, scale: ScaleLevel) -> Result<Vec<ReasoningTemplate>> {
        let scale_index = self.scale_index.read().await;
        let templates = self.templates.read().await;

        let empty_vec = vec![];
        let template_ids = scale_index.get(&scale).unwrap_or(&empty_vec);
        let results = template_ids.iter().filter_map(|id| templates.get(id).cloned()).collect();

        Ok(results)
    }

    /// Find templates by tag
    pub async fn find_by_tag(&self, tag: &str) -> Result<Vec<ReasoningTemplate>> {
        let tag_index = self.tag_index.read().await;
        let templates = self.templates.read().await;

        let empty_vec = vec![];
        let template_ids = tag_index.get(tag).unwrap_or(&empty_vec);
        let results = template_ids.iter().filter_map(|id| templates.get(id).cloned()).collect();

        Ok(results)
    }

    /// Get most popular templates
    pub async fn get_popular_templates(&self, limit: usize) -> Result<Vec<ReasoningTemplate>> {
        let usage_stats = self.usage_stats.read().await;
        let templates = self.templates.read().await;

        let mut sorted_ids: Vec<_> =
            usage_stats.iter().map(|(id, count)| (id.clone(), *count)).collect();
        sorted_ids.sort_by(|a, b| b.1.cmp(&a.1));

        let results = sorted_ids
            .into_iter()
            .take(limit)
            .filter_map(|(id, _)| templates.get(&id).cloned())
            .collect();

        Ok(results)
    }

    /// Record template usage
    pub async fn record_usage(&self, template_id: &TemplateId) -> Result<()> {
        let mut usage_stats = self.usage_stats.write().await;
        *usage_stats.entry(template_id.clone()).or_insert(0) += 1;
        Ok(())
    }

    /// Get library statistics
    pub async fn get_statistics(&self) -> Result<LibraryStatistics> {
        let templates = self.templates.read().await;
        let usage_stats = self.usage_stats.read().await;

        let total_templates = templates.len();
        let total_usage = usage_stats.values().sum();
        let categories = templates
            .values()
            .map(|t| t.category.clone())
            .collect::<std::collections::HashSet<_>>()
            .len();

        Ok(LibraryStatistics {
            total_templates,
            total_usage,
            category_count: categories,
            average_usage: if total_templates > 0 {
                total_usage as f64 / total_templates as f64
            } else {
                0.0
            },
        })
    }
}

/// Library statistics
#[derive(Clone, Debug)]
pub struct LibraryStatistics {
    pub total_templates: usize,
    pub total_usage: u64,
    pub category_count: usize,
    pub average_usage: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_template_library_creation() {
        let library = TemplateLibrary::new().await.unwrap();
        let stats = library.get_statistics().await.unwrap();

        assert!(stats.total_templates > 0);
    }

    #[tokio::test]
    async fn test_template_instantiation() {
        let library = Arc::new(TemplateLibrary::new().await.unwrap());
        let instantiator = TemplateInstantiator::new(library).await.unwrap();

        let template_id = TemplateId::from_name("analytical_decomposition");
        let params = InstantiationParams {
            scale_level: ScaleLevel::Concept,
            context: RecursiveContext::default(),
            parameter_overrides: HashMap::new(),
            resource_constraints: StepResources::default(),
            quality_requirements: QualityRequirements::default(),
        };

        let result = instantiator.instantiate(&template_id, params).await.unwrap();
        assert!(!result.thought_units.is_empty());
        assert!(!result.execution_plan.steps.is_empty());
    }

    #[test]
    fn test_template_id_generation() {
        let id1 = TemplateId::new();
        let id2 = TemplateId::new();
        assert_ne!(id1, id2);

        let id3 = TemplateId::from_name("test");
        let id4 = TemplateId::from_name("test");
        assert_eq!(id3, id4);
    }
}
