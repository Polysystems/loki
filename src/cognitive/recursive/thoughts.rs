//! Thought Units for Recursive Cognitive Processing
//!
//! Implements different types of thought units that can be combined recursively
//! to create complex cognitive processes.

use std::collections::HashMap;

use anyhow::Result;
use async_trait::async_trait;
// RwLock imports removed - not used in current implementation
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::RecursiveContext;
use crate::memory::fractal::ScaleLevel;

/// Unique identifier for thought units
#[derive(Clone, Debug, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct ThoughtUnitId(String);

impl ThoughtUnitId {
    pub fn new() -> Self {
        Self(Uuid::new_v4().to_string())
    }
}

/// Input data for thought processing
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ThoughtInput {
    /// Raw content to process
    pub content: String,

    /// Context information
    pub context: HashMap<String, String>,

    /// Quality score of input
    pub quality: f32,

    /// Source of this input
    pub source: InputSource,

    /// Timestamp of input creation
    pub timestamp: DateTime<Utc>,
}

/// Source of thought input
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum InputSource {
    External,
    Internal,
    Recursive,
    Memory,
    Sensor,
    Communication,
}

impl std::fmt::Display for InputSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InputSource::External => write!(f, "External"),
            InputSource::Internal => write!(f, "Internal"),
            InputSource::Recursive => write!(f, "Recursive"),
            InputSource::Memory => write!(f, "Memory"),
            InputSource::Sensor => write!(f, "Sensor"),
            InputSource::Communication => write!(f, "Communication"),
        }
    }
}

/// Output from thought processing
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ThoughtOutput {
    /// Processed content
    pub content: String,

    /// Confidence in this output
    pub confidence: f32,

    /// Processing metadata
    pub metadata: HashMap<String, String>,

    /// Quality assessment
    pub quality: OutputQuality,

    /// Timestamp of output creation
    pub timestamp: DateTime<Utc>,

    /// Whether this output triggers further recursion
    pub triggers_recursion: bool,
}

/// Quality assessment of thought output
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OutputQuality {
    /// Logical coherence
    pub coherence: f32,

    /// Creativity/novelty
    pub creativity: f32,

    /// Relevance to input
    pub relevance: f32,

    /// Depth of processing
    pub depth: f32,

    /// Overall quality score
    pub overall: f32,
}

impl Default for OutputQuality {
    fn default() -> Self {
        Self { coherence: 0.5, creativity: 0.5, relevance: 0.5, depth: 0.5, overall: 0.5 }
    }
}

/// Trait for all thought units
#[async_trait]
pub trait ThoughtUnit: Send + Sync {
    /// Get unique identifier for this thought unit
    fn get_id(&self) -> ThoughtUnitId;

    /// Get the type name of this thought unit
    fn get_type_name(&self) -> &'static str;

    /// Get the scale level this unit operates at
    fn get_scale_level(&self) -> ScaleLevel;

    /// Process input and produce output
    async fn process(
        &self,
        input: ThoughtInput,
        context: &RecursiveContext,
    ) -> Result<ThoughtOutput>;

    /// Check if this unit can handle the given input
    async fn can_handle(&self, input: &ThoughtInput) -> bool;

    /// Get processing complexity (for resource planning)
    fn get_complexity(&self) -> ProcessingComplexity;

    /// Clone this thought unit
    fn clone_unit(&self) -> Box<dyn ThoughtUnit + Send + Sync>;
}

/// Processing complexity levels
#[derive(Clone, Debug, PartialEq)]
pub enum ProcessingComplexity {
    Low,
    Medium,
    High,
    VeryHigh,
}

/// Atomic thought unit - simplest processing element
#[derive(Clone, Debug)]
pub struct AtomicThoughtUnit {
    id: ThoughtUnitId,
    operation: AtomicOperation,
    scale_level: ScaleLevel,
    parameters: AtomicParameters,
}

/// Types of atomic operations
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum AtomicOperation {
    /// Simple text transformation
    Transform(String),
    /// Pattern matching
    Match(String),
    /// Classification
    Classify,
    /// Extraction
    Extract(String),
    /// Validation
    Validate,
    /// Comparison
    Compare,
    /// Synthesis
    Synthesize,
}

/// Parameters for atomic operations
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AtomicParameters {
    /// Processing strength (0.0 to 1.0)
    pub strength: f32,

    /// Sensitivity threshold
    pub threshold: f32,

    /// Custom parameters
    pub custom: HashMap<String, String>,
}

impl Default for AtomicParameters {
    fn default() -> Self {
        Self { strength: 0.5, threshold: 0.5, custom: HashMap::new() }
    }
}

impl AtomicThoughtUnit {
    pub fn new(operation: AtomicOperation, scale_level: ScaleLevel) -> Self {
        Self {
            id: ThoughtUnitId::new(),
            operation,
            scale_level,
            parameters: AtomicParameters::default(),
        }
    }

    pub fn with_parameters(mut self, parameters: AtomicParameters) -> Self {
        self.parameters = parameters;
        self
    }
}

#[async_trait]
impl ThoughtUnit for AtomicThoughtUnit {
    fn get_id(&self) -> ThoughtUnitId {
        self.id.clone()
    }

    fn get_type_name(&self) -> &'static str {
        "AtomicThoughtUnit"
    }

    fn get_scale_level(&self) -> ScaleLevel {
        self.scale_level
    }

    async fn process(
        &self,
        input: ThoughtInput,
        _context: &RecursiveContext,
    ) -> Result<ThoughtOutput> {
        let processed_content = match &self.operation {
            AtomicOperation::Transform(pattern) => {
                format!("transformed[{}]({})", pattern, input.content)
            }
            AtomicOperation::Match(pattern) => {
                let matches = input.content.contains(pattern);
                format!("match_result: {} for pattern '{}'", matches, pattern)
            }
            AtomicOperation::Classify => {
                format!("classified: {}", self.classify_content(&input.content))
            }
            AtomicOperation::Extract(target) => {
                format!("extracted[{}] from: {}", target, input.content)
            }
            AtomicOperation::Validate => {
                let valid = self.validate_content(&input.content);
                format!("validation_result: {}", valid)
            }
            AtomicOperation::Compare => {
                format!("comparison_analysis: {}", input.content)
            }
            AtomicOperation::Synthesize => {
                format!("synthesized: {}", self.synthesize_content(&input.content))
            }
        };

        let quality = OutputQuality {
            coherence: 0.8,
            creativity: 0.3,
            relevance: 0.9,
            depth: 0.4,
            overall: 0.6,
        };

        Ok(ThoughtOutput {
            content: processed_content,
            confidence: 0.7,
            metadata: HashMap::new(),
            quality,
            timestamp: Utc::now(),
            triggers_recursion: false,
        })
    }

    async fn can_handle(&self, input: &ThoughtInput) -> bool {
        // Atomic units can handle most simple inputs
        !input.content.is_empty() && input.content.len() < 1000
    }

    fn get_complexity(&self) -> ProcessingComplexity {
        ProcessingComplexity::Low
    }

    fn clone_unit(&self) -> Box<dyn ThoughtUnit + Send + Sync> {
        Box::new(self.clone())
    }
}

impl AtomicThoughtUnit {
    fn classify_content(&self, content: &str) -> String {
        if content.contains("question") || content.ends_with('?') {
            "question".to_string()
        } else if content.contains("urgent") || content.contains('!') {
            "urgent".to_string()
        } else {
            "statement".to_string()
        }
    }

    fn validate_content(&self, content: &str) -> bool {
        !content.trim().is_empty() && content.len() > 3
    }

    fn synthesize_content(&self, content: &str) -> String {
        format!("synthesis of: {}", content.chars().take(50).collect::<String>())
    }
}

/// Composite thought unit - combines multiple thought units
pub struct CompositeThoughtUnit {
    id: ThoughtUnitId,
    child_units: Vec<Box<dyn ThoughtUnit + Send + Sync>>,
    composition_strategy: CompositionStrategy,
    scale_level: ScaleLevel,
    orchestration: OrchestrationMode,
}

/// Strategies for combining thought units
#[derive(Clone, Debug, PartialEq)]
pub enum CompositionStrategy {
    /// Execute all units in sequence
    Sequential,
    /// Execute all units in parallel
    Parallel,
    /// Execute units conditionally
    Conditional,
    /// Execute units in a pipeline
    Pipeline,
    /// Execute units in a tree structure
    Tree,
    /// Execute units based on voting
    Voting,
}

/// Orchestration modes for composite units
#[derive(Clone, Debug, PartialEq)]
pub enum OrchestrationMode {
    /// Simple aggregation
    Aggregate,
    /// Weighted combination
    Weighted,
    /// Best result selection
    BestOf,
    /// Consensus building
    Consensus,
    /// Competitive selection
    Competitive,
}

impl CompositeThoughtUnit {
    pub fn new(
        child_units: Vec<Box<dyn ThoughtUnit + Send + Sync>>,
        strategy: CompositionStrategy,
        scale_level: ScaleLevel,
    ) -> Self {
        Self {
            id: ThoughtUnitId::new(),
            child_units,
            composition_strategy: strategy,
            scale_level,
            orchestration: OrchestrationMode::Aggregate,
        }
    }

    pub fn with_orchestration(mut self, mode: OrchestrationMode) -> Self {
        self.orchestration = mode;
        self
    }
}

// Manual Clone implementation since trait objects don't auto-derive Clone
impl Clone for CompositeThoughtUnit {
    fn clone(&self) -> Self {
        let cloned_children: Vec<Box<dyn ThoughtUnit + Send + Sync>> =
            self.child_units.iter().map(|unit| unit.clone_unit()).collect();

        Self {
            id: ThoughtUnitId::new(),
            child_units: cloned_children,
            composition_strategy: self.composition_strategy.clone(),
            scale_level: self.scale_level,
            orchestration: self.orchestration.clone(),
        }
    }
}

#[async_trait]
impl ThoughtUnit for CompositeThoughtUnit {
    fn get_id(&self) -> ThoughtUnitId {
        self.id.clone()
    }

    fn get_type_name(&self) -> &'static str {
        "CompositeThoughtUnit"
    }

    fn get_scale_level(&self) -> ScaleLevel {
        self.scale_level
    }

    async fn process(
        &self,
        input: ThoughtInput,
        context: &RecursiveContext,
    ) -> Result<ThoughtOutput> {
        match self.composition_strategy {
            CompositionStrategy::Sequential => self.process_sequential(input, context).await,
            CompositionStrategy::Parallel => self.process_parallel(input, context).await,
            CompositionStrategy::Pipeline => self.process_pipeline(input, context).await,
            _ => {
                // Simplified implementation for other strategies
                self.process_sequential(input, context).await
            }
        }
    }

    async fn can_handle(&self, input: &ThoughtInput) -> bool {
        // Check if any child unit can handle the input
        for unit in &self.child_units {
            if unit.can_handle(input).await {
                return true;
            }
        }
        false
    }

    fn get_complexity(&self) -> ProcessingComplexity {
        ProcessingComplexity::Medium
    }

    fn clone_unit(&self) -> Box<dyn ThoughtUnit + Send + Sync> {
        Box::new(self.clone())
    }
}

impl CompositeThoughtUnit {
    async fn process_sequential(
        &self,
        mut input: ThoughtInput,
        context: &RecursiveContext,
    ) -> Result<ThoughtOutput> {
        let mut combined_content = String::new();
        let mut overall_confidence = 1.0f32;

        for unit in &self.child_units {
            if unit.can_handle(&input).await {
                let output = unit.process(input.clone(), context).await?;
                combined_content.push_str(&format!("{} -> ", output.content));
                overall_confidence *= output.confidence;

                // Use output as input for next unit
                input.content = output.content;
            }
        }

        Ok(ThoughtOutput {
            content: combined_content,
            confidence: overall_confidence,
            metadata: HashMap::new(),
            quality: OutputQuality::default(),
            timestamp: Utc::now(),
            triggers_recursion: true,
        })
    }

    async fn process_parallel(
        &self,
        input: ThoughtInput,
        context: &RecursiveContext,
    ) -> Result<ThoughtOutput> {
        let mut results = Vec::new();

        // Process all units in parallel
        let futures: Vec<_> = self
            .child_units
            .iter()
            .filter_map(|unit| {
                if tokio::runtime::Handle::try_current().is_ok() {
                    // We're in an async context, create a future
                    let unit_clone = unit.clone_unit();
                    let input_clone = input.clone();
                    let context_clone = context.clone();
                    Some(async move {
                        if unit_clone.can_handle(&input_clone).await {
                            unit_clone.process(input_clone, &context_clone).await
                        } else {
                            Err(anyhow::anyhow!("Unit cannot handle input"))
                        }
                    })
                } else {
                    None
                }
            })
            .collect();

        // Execute all futures
        for future in futures {
            if let Ok(output) = future.await {
                results.push(output);
            }
        }

        // Combine results based on orchestration mode
        self.combine_results(results)
    }

    async fn process_pipeline(
        &self,
        input: ThoughtInput,
        context: &RecursiveContext,
    ) -> Result<ThoughtOutput> {
        let mut pipeline_output = input.clone();

        for unit in &self.child_units {
            let unit_input = ThoughtInput {
                content: pipeline_output.content.clone(),
                context: pipeline_output.context.clone(),
                quality: pipeline_output.quality,
                source: InputSource::Internal,
                timestamp: Utc::now(),
            };

            if unit.can_handle(&unit_input).await {
                let output = unit.process(unit_input, context).await?;
                pipeline_output.content = output.content;
                pipeline_output.quality = output.confidence;
            }
        }

        Ok(ThoughtOutput {
            content: pipeline_output.content,
            confidence: pipeline_output.quality,
            metadata: HashMap::new(),
            quality: OutputQuality::default(),
            timestamp: Utc::now(),
            triggers_recursion: false,
        })
    }

    fn combine_results(&self, results: Vec<ThoughtOutput>) -> Result<ThoughtOutput> {
        if results.is_empty() {
            return Ok(ThoughtOutput {
                content: "No results produced".to_string(),
                confidence: 0.0,
                metadata: HashMap::new(),
                quality: OutputQuality::default(),
                timestamp: Utc::now(),
                triggers_recursion: false,
            });
        }

        match self.orchestration {
            OrchestrationMode::Aggregate => {
                let combined_content =
                    results.iter().map(|r| r.content.clone()).collect::<Vec<_>>().join(" | ");

                let avg_confidence =
                    results.iter().map(|r| r.confidence).sum::<f32>() / results.len() as f32;

                Ok(ThoughtOutput {
                    content: combined_content,
                    confidence: avg_confidence,
                    metadata: HashMap::new(),
                    quality: OutputQuality::default(),
                    timestamp: Utc::now(),
                    triggers_recursion: false,
                })
            }
            OrchestrationMode::BestOf => {
                let best_result = results
                    .into_iter()
                    .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap())
                    .unwrap();

                Ok(best_result)
            }
            _ => {
                // Default to aggregate for other modes
                self.combine_results(results)
            }
        }
    }
}

/// Meta thought unit - thinks about thinking
pub struct MetaThoughtUnit {
    id: ThoughtUnitId,
    target_unit: Option<Box<dyn ThoughtUnit + Send + Sync>>,
    meta_operation: MetaOperation,
    scale_level: ScaleLevel,
    reflection_depth: u32,
}

/// Types of meta-cognitive operations
#[derive(Clone, Debug, PartialEq)]
pub enum MetaOperation {
    /// Analyze thinking process
    ProcessAnalysis,
    /// Monitor performance
    PerformanceMonitoring,
    /// Strategic planning
    StrategyPlanning,
    /// Self-improvement
    SelfImprovement,
    /// Bias detection
    BiasDetection,
    /// Confidence calibration
    ConfidenceCalibration,
}

impl MetaThoughtUnit {
    pub fn new(meta_operation: MetaOperation, scale_level: ScaleLevel) -> Self {
        Self {
            id: ThoughtUnitId::new(),
            target_unit: None,
            meta_operation,
            scale_level,
            reflection_depth: 1,
        }
    }

    pub fn with_target(mut self, target: Box<dyn ThoughtUnit + Send + Sync>) -> Self {
        self.target_unit = Some(target);
        self
    }

    pub fn with_reflection_depth(mut self, depth: u32) -> Self {
        self.reflection_depth = depth;
        self
    }
}

// Manual Clone implementation since trait objects don't auto-derive Clone
impl Clone for MetaThoughtUnit {
    fn clone(&self) -> Self {
        Self {
            id: ThoughtUnitId::new(),
            target_unit: self.target_unit.as_ref().map(|unit| unit.clone_unit()),
            meta_operation: self.meta_operation.clone(),
            scale_level: self.scale_level,
            reflection_depth: self.reflection_depth,
        }
    }
}

#[async_trait]
impl ThoughtUnit for MetaThoughtUnit {
    fn get_id(&self) -> ThoughtUnitId {
        self.id.clone()
    }

    fn get_type_name(&self) -> &'static str {
        "MetaThoughtUnit"
    }

    fn get_scale_level(&self) -> ScaleLevel {
        self.scale_level
    }

    async fn process(
        &self,
        input: ThoughtInput,
        context: &RecursiveContext,
    ) -> Result<ThoughtOutput> {
        let meta_content = match &self.meta_operation {
            MetaOperation::ProcessAnalysis => self.analyze_process(&input, context).await?,
            MetaOperation::PerformanceMonitoring => {
                self.monitor_performance(&input, context).await?
            }
            MetaOperation::StrategyPlanning => self.plan_strategy(&input, context).await?,
            MetaOperation::SelfImprovement => self.identify_improvements(&input, context).await?,
            MetaOperation::BiasDetection => self.detect_biases(&input, context).await?,
            MetaOperation::ConfidenceCalibration => {
                self.calibrate_confidence(&input, context).await?
            }
        };

        Ok(ThoughtOutput {
            content: meta_content,
            confidence: 0.8,
            metadata: HashMap::new(),
            quality: OutputQuality {
                coherence: 0.9,
                creativity: 0.7,
                relevance: 0.8,
                depth: 0.9,
                overall: 0.85,
            },
            timestamp: Utc::now(),
            triggers_recursion: true,
        })
    }

    async fn can_handle(&self, input: &ThoughtInput) -> bool {
        // Meta units can handle any input for reflection
        !input.content.is_empty()
    }

    fn get_complexity(&self) -> ProcessingComplexity {
        ProcessingComplexity::High
    }

    fn clone_unit(&self) -> Box<dyn ThoughtUnit + Send + Sync> {
        Box::new(self.clone())
    }
}

impl MetaThoughtUnit {
    async fn analyze_process(
        &self,
        input: &ThoughtInput,
        context: &RecursiveContext,
    ) -> Result<String> {
        Ok(format!(
            "Process analysis: Input '{}' at depth {} using {:?} recursion. Quality assessment: {}",
            input.content,
            context.depth.as_u32(),
            context.recursion_type,
            input.quality
        ))
    }

    async fn monitor_performance(
        &self,
        input: &ThoughtInput,
        context: &RecursiveContext,
    ) -> Result<String> {
        let efficiency =
            context.resource_usage.compute_cycles as f64 / (input.content.len() as f64 + 1.0);
        Ok(format!(
            "Performance monitoring: Efficiency {:.2}, Memory usage {} bytes, Step count {}",
            efficiency,
            context.resource_usage.memory_bytes,
            context.recursion_trail.len()
        ))
    }

    async fn plan_strategy(
        &self,
        input: &ThoughtInput,
        _context: &RecursiveContext,
    ) -> Result<String> {
        Ok(format!(
            "Strategic planning: For input type '{}', recommend {} approach with {} priority",
            input.source,
            if input.content.len() > 100 { "complex" } else { "simple" },
            if input.quality > 0.7 { "high" } else { "medium" }
        ))
    }

    async fn identify_improvements(
        &self,
        input: &ThoughtInput,
        context: &RecursiveContext,
    ) -> Result<String> {
        let suggestions = if context.depth.as_u32() > 5 {
            "Consider depth limiting"
        } else if input.quality < 0.5 {
            "Improve input preprocessing"
        } else {
            "Optimize resource usage"
        };

        Ok(format!("Self-improvement analysis: {}", suggestions))
    }

    async fn detect_biases(
        &self,
        input: &ThoughtInput,
        _context: &RecursiveContext,
    ) -> Result<String> {
        let potential_biases =
            if input.content.contains("always") || input.content.contains("never") {
                vec!["Absolute thinking"]
            } else if input.content.len() < 10 {
                vec!["Superficial processing"]
            } else {
                vec![]
            };

        Ok(format!("Bias detection: {:?}", potential_biases))
    }

    async fn calibrate_confidence(
        &self,
        input: &ThoughtInput,
        context: &RecursiveContext,
    ) -> Result<String> {
        let adjusted_confidence = if context.depth.as_u32() > 3 {
            input.quality * 0.8 // Reduce confidence for deep recursion
        } else {
            input.quality * 1.1 // Slightly increase for shallow processing
        };

        Ok(format!(
            "Confidence calibration: Original {:.2}, Adjusted {:.2}",
            input.quality,
            adjusted_confidence.min(1.0)
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_atomic_thought_unit() {
        let unit = AtomicThoughtUnit::new(
            AtomicOperation::Transform("test".to_string()),
            ScaleLevel::Atomic,
        );

        let input = ThoughtInput {
            content: "hello world".to_string(),
            context: HashMap::new(),
            quality: 0.8,
            source: InputSource::External,
            timestamp: Utc::now(),
        };

        let context = RecursiveContext::default();
        let output = unit.process(input, &context).await.unwrap();

        assert!(output.content.contains("transformed"));
        assert!(output.confidence > 0.0);
    }

    #[tokio::test]
    async fn test_meta_thought_unit() {
        let unit = MetaThoughtUnit::new(MetaOperation::ProcessAnalysis, ScaleLevel::Meta);

        let input = ThoughtInput {
            content: "analyze this".to_string(),
            context: HashMap::new(),
            quality: 0.7,
            source: InputSource::Internal,
            timestamp: Utc::now(),
        };

        let context = RecursiveContext::default();
        let output = unit.process(input, &context).await.unwrap();

        assert!(output.content.contains("Process analysis"));
        assert!(output.quality.overall > 0.5);
    }

    #[test]
    fn test_thought_unit_id_generation() {
        let id1 = ThoughtUnitId::new();
        let id2 = ThoughtUnitId::new();

        assert_ne!(id1, id2);
    }
}
