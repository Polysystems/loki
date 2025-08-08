//! Story-Driven Code Generation
//!
//! This module implements intelligent code generation that understands context
//! through the story system, learning from patterns and generating code that
//! fits naturally with the existing codebase narrative.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{info, warn};

use crate::cognitive::self_modify::{ChangeType, CodeChange, RiskLevel, SelfModificationPipeline};
use crate::cognitive::test_generator::{TestCase, TestGenerator, TestGeneratorConfig};
use crate::memory::{CognitiveMemory, MemoryMetadata};
use crate::models::{InferenceEngine, InferenceRequest};
use crate::story::{
    PlotType, StoryEngine, StoryId, StoryType, StorySegment, MappedTask, TaskStatus,
};
use crate::tools::code_analysis::CodeAnalyzer;

/// Configuration for story-driven code generation
#[derive(Debug, Clone)]
pub struct StoryDrivenCodeGenConfig {
    /// Enable function generation from story context
    pub enable_function_generation: bool,

    /// Enable test generation from story context
    pub enable_test_generation: bool,

    /// Enable documentation generation
    pub enable_doc_generation: bool,

    /// Enable refactoring suggestions
    pub enable_refactoring: bool,

    /// Enable API endpoint generation
    pub enable_api_generation: bool,

    /// Enable data structure generation
    pub enable_struct_generation: bool,

    /// Maximum risk level for autonomous generation
    pub max_risk_level: RiskLevel,

    /// Use learned patterns for generation
    pub use_learned_patterns: bool,

    /// Confidence threshold for generation
    pub confidence_threshold: f32,
}

impl Default for StoryDrivenCodeGenConfig {
    fn default() -> Self {
        Self {
            enable_function_generation: true,
            enable_test_generation: true,
            enable_doc_generation: true,
            enable_refactoring: true,
            enable_api_generation: true,
            enable_struct_generation: true,
            max_risk_level: RiskLevel::Medium,
            use_learned_patterns: true,
            confidence_threshold: 0.7,
        }
    }
}

/// Code generation context from story
#[derive(Debug, Clone)]
pub struct GenerationContext {
    /// Story segment providing context
    pub story_segment: StorySegment,

    /// Related code files
    pub related_files: Vec<PathBuf>,

    /// Learned patterns applicable
    pub applicable_patterns: Vec<CodePattern>,

    /// Current task if any
    pub task: Option<MappedTask>,

    /// Confidence score
    pub confidence: f32,
}

/// Learned code pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodePattern {
    pub pattern_id: String,
    pub pattern_type: CodePatternType,
    pub description: String,
    pub template: String,
    pub usage_count: usize,
    pub success_rate: f32,
    pub applicable_contexts: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CodePatternType {
    FunctionStructure,
    ErrorHandling,
    TestStructure,
    APIEndpoint,
    DataStructure,
    Documentation,
    Refactoring,
}

/// Generated code artifact
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedCode {
    pub artifact_type: GeneratedArtifactType,
    pub file_path: PathBuf,
    pub content: String,
    pub description: String,
    pub confidence: f32,
    pub risk_level: RiskLevel,
    pub dependencies: Vec<String>,
    pub tests: Vec<TestCase>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GeneratedArtifactType {
    Function,
    Test,
    Documentation,
    Refactoring,
    API,
    DataStructure,
    Module,
}

/// Story-driven code generation system
pub struct StoryDrivenCodeGenerator {
    config: StoryDrivenCodeGenConfig,
    story_engine: Arc<StoryEngine>,
    self_modify: Arc<SelfModificationPipeline>,
    test_generator: Arc<TestGenerator>,
    code_analyzer: Arc<CodeAnalyzer>,
    memory: Arc<CognitiveMemory>,
    inference_engine: Option<Arc<dyn InferenceEngine>>,

    /// Learned code patterns
    learned_patterns: Arc<RwLock<HashMap<String, CodePattern>>>,

    /// Generation history for learning
    generation_history: Arc<RwLock<Vec<GenerationRecord>>>,
}

#[derive(Debug, Clone)]
struct GenerationRecord {
    pub artifact: GeneratedCode,
    pub context: GenerationContext,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub success: bool,
    pub feedback: Option<String>,
}

impl StoryDrivenCodeGenerator {
    /// Create a new story-driven code generator
    pub async fn new(
        config: StoryDrivenCodeGenConfig,
        story_engine: Arc<StoryEngine>,
        self_modify: Arc<SelfModificationPipeline>,
        memory: Arc<CognitiveMemory>,
        inference_engine: Option<Arc<dyn InferenceEngine>>,
    ) -> Result<Self> {
        info!("🎨 Initializing Story-Driven Code Generator");

        // Create test generator
        let test_generator = Arc::new(
            TestGenerator::new(TestGeneratorConfig::default(), memory.clone()).await?
        );

        // Create code analyzer
        let code_analyzer = Arc::new(CodeAnalyzer::new(memory.clone()).await?);

        // Load learned patterns from memory
        let patterns = Self::load_patterns_from_memory(&memory).await?;

        Ok(Self {
            config,
            story_engine,
            self_modify,
            test_generator,
            code_analyzer,
            memory,
            inference_engine,
            learned_patterns: Arc::new(RwLock::new(patterns)),
            generation_history: Arc::new(RwLock::new(Vec::new())),
        })
    }

    /// Generate code based on story context
    pub async fn generate_from_story_context(
        &self,
        story_id: &StoryId,
        segment_id: Option<String>,
    ) -> Result<Vec<GeneratedCode>> {
        info!("📝 Generating code from story context");

        // Get story context
        let context = self.build_generation_context(story_id, segment_id).await?;

        if context.confidence < self.config.confidence_threshold {
            warn!(
                "Context confidence {} below threshold {}",
                context.confidence, self.config.confidence_threshold
            );
            return Ok(Vec::new());
        }

        let mut generated_artifacts = Vec::new();

        // Get plot points from the story if available
        let segment = &context.story_segment;
        let story_id = segment.story_id;
        let plot_points = if let Some(story) = self.story_engine.get_story(&story_id) {
            story.arcs.iter()
                .flat_map(|arc| arc.plot_points.clone())
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };

        // Check segment context for type and generate accordingly
        if let Some(segment_type) = segment.context.get("type") {
            match segment_type.as_str() {
                "goal" => {
                    // Extract objective from content
                    let objective = segment.content.lines().next().unwrap_or(&segment.content);
                    if self.config.enable_function_generation {
                        if let Ok(code) = self.generate_goal_implementation(objective, &context).await {
                            generated_artifacts.push(code);
                        }
                    }
                }
                "task" => {
                    // Extract task description from content
                    let description = segment.content.lines().next().unwrap_or(&segment.content);
                    let completed = segment.context.get("completed").map(|s| s == "true").unwrap_or(false);
                    if !completed && self.config.enable_function_generation {
                        if let Ok(code) = self.generate_task_implementation(description, &context).await {
                            generated_artifacts.push(code);
                        }
                    }
                }
                "issue" | "error" => {
                    // Extract error from content
                    let error = segment.content.lines().next().unwrap_or(&segment.content);
                    let resolved = segment.context.get("resolved").map(|s| s == "true").unwrap_or(false);
                    if !resolved && self.config.enable_refactoring {
                        if let Ok(code) = self.generate_issue_fix(error, &context).await {
                            generated_artifacts.push(code);
                        }
                    }
                }
                "discovery" => {
                    // Extract insight from content
                    let insight = segment.content.lines().next().unwrap_or(&segment.content);
                    if self.config.use_learned_patterns {
                        if let Ok(codes) = self.generate_from_insight(insight, &context).await {
                            generated_artifacts.extend(codes);
                        }
                    }
                }
                _ => {}
            }
        }

        // Generate tests for new code
        if self.config.enable_test_generation {
            let mut tests_to_generate = Vec::new();
            for artifact in &generated_artifacts {
                if matches!(artifact.artifact_type, GeneratedArtifactType::Function) {
                    tests_to_generate.push(artifact.clone());
                }
            }

            for artifact in tests_to_generate {
                if let Ok(test_code) = self.generate_tests_for_artifact(&artifact).await {
                    generated_artifacts.push(test_code);
                }
            }
        }

        // Record generation in story
        for artifact in &generated_artifacts {
            let _ = self.story_engine
                .add_plot_point(
                    story_id.clone(),
                    PlotType::Transformation {
                            before: "No implementation".to_string(),
                            after: format!("Generated {}: {}",
                                match artifact.artifact_type {
                                    GeneratedArtifactType::Function => "function",
                                    GeneratedArtifactType::Test => "test",
                                    GeneratedArtifactType::Documentation => "documentation",
                                    GeneratedArtifactType::Refactoring => "refactoring",
                                    GeneratedArtifactType::API => "API",
                                    GeneratedArtifactType::DataStructure => "data structure",
                                    GeneratedArtifactType::Module => "module",
                                },
                                artifact.description
                            ),
                        },
                    vec!["code_generation".to_string(), "autonomous".to_string()],
                )
                .await;
        }

        // Store in generation history
        let mut history = self.generation_history.write().await;
        for artifact in &generated_artifacts {
            history.push(GenerationRecord {
                artifact: artifact.clone(),
                context: context.clone(),
                timestamp: Utc::now(),
                success: true,
                feedback: None,
            });
        }

        Ok(generated_artifacts)
    }

    /// Build generation context from story
    async fn build_generation_context(
        &self,
        story_id: &StoryId,
        segment_id: Option<String>,
    ) -> Result<GenerationContext> {
        // Get story segment
        let segment = if let Some(seg_id) = segment_id {
            self.story_engine.get_segment(story_id, &seg_id).await?
        } else {
            // Get latest segment
            let story = self.story_engine.get_story(story_id)
                .ok_or_else(|| anyhow::anyhow!("Story not found"))?;
            // Create a segment from the story's current context
            let segment_content = story.summary.clone();
            crate::story::StorySegment {
                id: uuid::Uuid::new_v4().to_string(),
                story_id: story.id,
                content: segment_content,
                context: std::collections::HashMap::new(),
                created_at: chrono::Utc::now(),
                segment_type: crate::story::SegmentType::Development,
                tags: Vec::new(),
            }
        };

        // Find related files from story context
        let mut related_files = Vec::new();
        // Extract file paths from plot points and segment context
        if let Some(story) = self.story_engine.get_story(&segment.story_id) {
            for arc in &story.arcs {
                for plot_point in &arc.plot_points {
                    // Extract file paths from plot point context tokens
                    for token in &plot_point.context_tokens {
                        if token.ends_with(".rs") || token.contains("src/") {
                            related_files.push(PathBuf::from(token));
                        }
                    }
                }
            }
        }
        if let Some(files) = segment.context.get("files") {
            for file in files.split(',') {
                related_files.push(PathBuf::from(file.trim()));
            }
        }
        // Also check for file paths in content
        for line in segment.content.lines() {
            if line.ends_with(".rs") || line.contains("src/") {
                if let Some(path) = line.split_whitespace().find(|s| s.ends_with(".rs") || s.contains("src/")) {
                    related_files.push(PathBuf::from(path));
                }
            }
        }

        // Find applicable patterns
        let patterns = self.learned_patterns.read().await;
        let applicable_patterns: Vec<CodePattern> = patterns
            .values()
            .filter(|p| {
                // Check if pattern applies to this context
                // Check pattern against plot points and segment content
                let story_has_context = if let Some(story) = self.story_engine.get_story(&segment.story_id) {
                    p.applicable_contexts.iter().any(|ctx| {
                        story.arcs.iter().any(|arc| {
                            arc.plot_points.iter().any(|pp| {
                                pp.context_tokens.iter().any(|token| token.contains(ctx)) ||
                                pp.description.contains(ctx)
                            })
                        })
                    })
                } else {
                    false
                };
                
                story_has_context ||
                p.applicable_contexts.iter().any(|ctx| {
                    segment.content.contains(ctx) ||
                    segment.context.values().any(|v| v.contains(ctx))
                })
            })
            .cloned()
            .collect();

        // Extract current task if any
        let task: Option<MappedTask> = if let Some(story) = self.story_engine.get_story(&segment.story_id) {
            // Look for task plot points in the story arcs
            story.arcs.iter()
                .flat_map(|arc| &arc.plot_points)
                .find_map(|pp| {
                    if let PlotType::Task { description, completed } = &pp.plot_type {
                        if !completed {
                            Some(MappedTask {
                                id: pp.id.0.to_string(),
                                description: description.clone(),
                                story_context: format!("Story: {}", story.title),
                                status: TaskStatus::Pending,
                                created_at: pp.timestamp,
                                updated_at: pp.timestamp,
                                plot_point: Some(pp.id),
                                assigned_to: None,
                            })
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
        } else {
            None
        };

        // Calculate confidence based on context richness
        let confidence = Self::calculate_context_confidence(&segment, &applicable_patterns);

        Ok(GenerationContext {
            story_segment: segment,
            related_files,
            applicable_patterns,
            task,
            confidence,
        })
    }

    /// Generate implementation for a goal
    async fn generate_goal_implementation(
        &self,
        objective: &str,
        context: &GenerationContext,
    ) -> Result<GeneratedCode> {
        info!("🎯 Generating implementation for goal: {}", objective);

        // Use AI to generate implementation
        let prompt = self.build_generation_prompt(objective, context, "function");

        let generated_content = if let Some(engine) = &self.inference_engine {
            let request = InferenceRequest {
                prompt,
                max_tokens: 1024,
                temperature: 0.7,
                top_p: 0.9,
                stop_sequences: vec![],
            };

            engine.infer(request).await?.text
        } else {
            // Fallback to template-based generation
            self.generate_from_template(objective, context, "function")?
        };

        // Determine file path
        let file_path = self.determine_file_path(objective, &context.related_files);

        // Create code change
        let code_change = CodeChange {
            file_path: file_path.clone(),
            change_type: ChangeType::Feature,
            description: format!("Implement: {}", objective),
            reasoning: "Story-driven goal implementation".to_string(),
            old_content: None,
            new_content: generated_content.clone(),
            line_range: None,
            risk_level: self.assess_risk_level(&generated_content),
            attribution: None,
        };

        // Validate with self-modification pipeline
        if code_change.risk_level <= self.config.max_risk_level {
            Ok(GeneratedCode {
                artifact_type: GeneratedArtifactType::Function,
                file_path,
                content: generated_content.clone(),
                description: objective.to_string(),
                confidence: context.confidence,
                risk_level: code_change.risk_level,
                dependencies: self.extract_dependencies(&generated_content),
                tests: Vec::new(),
            })
        } else {
            Err(anyhow::anyhow!("Generated code exceeds risk threshold"))
        }
    }

    /// Generate implementation for a task
    async fn generate_task_implementation(
        &self,
        description: &str,
        context: &GenerationContext,
    ) -> Result<GeneratedCode> {
        info!("📋 Generating implementation for task: {}", description);

        // Similar to goal implementation but task-focused
        self.generate_goal_implementation(description, context).await
    }

    /// Generate fix for an issue
    async fn generate_issue_fix(
        &self,
        error: &str,
        context: &GenerationContext,
    ) -> Result<GeneratedCode> {
        info!("🔧 Generating fix for issue: {}", error);

        // Analyze the error to understand what needs fixing
        let error_analysis = self.analyze_error(error, context).await?;

        // Generate fix based on analysis
        let fix_content = if let Some(pattern) = context.applicable_patterns
            .iter()
            .find(|p| matches!(p.pattern_type, CodePatternType::ErrorHandling))
        {
            // Use learned error handling pattern
            self.apply_pattern_template(&pattern.template, &error_analysis)?
        } else {
            // Generate new fix
            self.generate_error_fix_code(error, &error_analysis)?
        };

        Ok(GeneratedCode {
            artifact_type: GeneratedArtifactType::Refactoring,
            file_path: error_analysis.file_path,
            content: fix_content,
            description: format!("Fix: {}", error),
            confidence: context.confidence * 0.9, // Slightly lower confidence for fixes
            risk_level: RiskLevel::Low,
            dependencies: Vec::new(),
            tests: Vec::new(),
        })
    }

    /// Generate code from discovered insight
    async fn generate_from_insight(
        &self,
        insight: &str,
        context: &GenerationContext,
    ) -> Result<Vec<GeneratedCode>> {
        info!("💡 Generating code from insight: {}", insight);

        let mut generated = Vec::new();

        // Analyze insight to determine what to generate
        if insight.contains("missing test") && self.config.enable_test_generation {
            // Generate missing tests
            if let Ok(test) = self.generate_missing_tests(insight, context).await {
                generated.push(test);
            }
        }

        if insight.contains("could be refactored") && self.config.enable_refactoring {
            // Generate refactoring
            if let Ok(refactor) = self.generate_refactoring(insight, context).await {
                generated.push(refactor);
            }
        }

        if insight.contains("needs documentation") && self.config.enable_doc_generation {
            // Generate documentation
            if let Ok(docs) = self.generate_documentation(insight, context).await {
                generated.push(docs);
            }
        }

        Ok(generated)
    }

    /// Generate tests for an artifact
    async fn generate_tests_for_artifact(
        &self,
        artifact: &GeneratedCode,
    ) -> Result<GeneratedCode> {
        info!("🧪 Generating tests for: {}", artifact.description);

        // Parse the generated code to understand what to test
        let temp_file = PathBuf::from("/tmp").join(&artifact.file_path.file_name().unwrap());
        tokio::fs::write(&temp_file, &artifact.content).await?;

        // Use test generator
        let test_suite = self.test_generator.generate_tests_for_file(&temp_file).await?;

        // Clean up temp file
        let _ = tokio::fs::remove_file(&temp_file).await;

        // Format tests
        let mut test_content = String::new();
        test_content.push_str("// Story-driven generated tests\n");
        test_content.push_str("use super::*;\n\n");
        test_content.push_str("#[cfg(test)]\n");
        test_content.push_str("mod tests {\n");
        test_content.push_str("    use super::*;\n\n");

        for test_case in &test_suite.test_cases {
            test_content.push_str(&format!("    {}\n\n", test_case.code));
        }

        test_content.push_str("}\n");

        // Determine test file path
        let test_file_path = artifact.file_path.with_extension("test.rs");

        Ok(GeneratedCode {
            artifact_type: GeneratedArtifactType::Test,
            file_path: test_file_path,
            content: test_content,
            description: format!("Tests for {}", artifact.description),
            confidence: artifact.confidence * 0.95,
            risk_level: RiskLevel::Low,
            dependencies: vec!["super::*".to_string()],
            tests: test_suite.test_cases,
        })
    }

    /// Build generation prompt for AI
    fn build_generation_prompt(
        &self,
        objective: &str,
        context: &GenerationContext,
        artifact_type: &str,
    ) -> String {
        let mut prompt = format!(
            "Generate a Rust {} to achieve the following objective:\n{}\n\n",
            artifact_type, objective
        );

        // Add story context
        prompt.push_str("Story Context:\n");
        // Use plot points and segment content as context
        if let Some(story) = self.story_engine.get_story(&context.story_segment.story_id) {
            for arc in &story.arcs {
                for plot_point in &arc.plot_points {
                    if !plot_point.description.is_empty() {
                        prompt.push_str(&format!("- {}: {}\n", 
                            chrono::DateTime::<chrono::Utc>::from(plot_point.timestamp).format("%Y-%m-%d"),
                            plot_point.description
                        ));
                    }
                }
            }
        }
        let content_lines: Vec<&str> = context.story_segment.content.lines().take(3).collect();
        for line in content_lines {
            prompt.push_str(&format!("- {}\n", line));
        }
        prompt.push_str("\n");

        // Add applicable patterns
        if !context.applicable_patterns.is_empty() {
            prompt.push_str("Follow these patterns from the codebase:\n");
            for pattern in context.applicable_patterns.iter().take(2) {
                prompt.push_str(&format!("- {:?}: {}\n", pattern.pattern_type, pattern.description));
            }
            prompt.push_str("\n");
        }

        // Add specific requirements
        prompt.push_str("Requirements:\n");
        prompt.push_str("- Follow Rust best practices and idioms\n");
        prompt.push_str("- Include proper error handling\n");
        prompt.push_str("- Add documentation comments\n");
        prompt.push_str("- Make the code testable and maintainable\n");

        prompt
    }

    /// Generate from template when AI is not available
    fn generate_from_template(
        &self,
        objective: &str,
        context: &GenerationContext,
        artifact_type: &str,
    ) -> Result<String> {
        // Find best matching pattern
        let pattern = context.applicable_patterns
            .iter()
            .find(|p| match artifact_type {
                "function" => matches!(p.pattern_type, CodePatternType::FunctionStructure),
                "test" => matches!(p.pattern_type, CodePatternType::TestStructure),
                _ => false,
            });

        if let Some(pattern) = pattern {
            // Apply pattern template
            let mut generated = pattern.template.clone();
            generated = generated.replace("{{objective}}", objective);
            generated = generated.replace("{{description}}", &format!("Story-driven: {}", objective));
            Ok(generated)
        } else {
            // Use enhanced template with real implementation structure
            match artifact_type {
                "function" => {
                    // Parse the objective to determine function structure
                    let function_name = self.extract_function_name(objective);
                    let (params, return_type) = self.infer_function_signature(objective, context);
                    
                    Ok(format!(
                        r#"/// {}
/// 
/// Generated from story context: {}
pub fn {}({}) -> {} {{
    // Initialize with default implementation
    let mut result = Default::default();
    
    // Process based on objective: {}
    {}
    
    // Return result
    Ok(result)
}}"#,
                        objective,
                        context.story_segment.content.lines().next().unwrap_or("No context"),
                        function_name,
                        params,
                        return_type,
                        objective,
                        self.generate_function_body(objective, context)
                    ))
                },
                "test" => {
                    let test_name = self.extract_test_name(objective);
                    let test_setup = self.generate_test_setup(objective, context);
                    
                    Ok(format!(
                        r#"#[test]
fn {}() {{
    // Setup test environment
    {}
    
    // Test objective: {}
    let result = {}();
    
    // Verify expectations
    assert!(result.is_ok(), "Function should succeed");
    {}
}}"#,
                        test_name,
                        test_setup,
                        objective,
                        self.extract_function_name(objective),
                        self.generate_test_assertions(objective, context)
                    ))
                },
                "module" => {
                    let module_name = self.extract_module_name(objective);
                    Ok(format!(
                        r#"//! {}
//!
//! Generated module for story-driven functionality

use anyhow::Result;
use std::collections::HashMap;

/// Main structure for {}
#[derive(Debug, Clone, Default)]
pub struct {} {{
    /// Internal state
    state: HashMap<String, String>,
}}

impl {} {{
    /// Create a new instance
    pub fn new() -> Self {{
        Self::default()
    }}
    
    /// Process according to objective: {}
    pub fn process(&mut self) -> Result<()> {{
        // Implementation based on story context
        Ok(())
    }}
}}"#,
                        objective,
                        module_name,
                        self.to_pascal_case(&module_name),
                        self.to_pascal_case(&module_name),
                        objective
                    ))
                },
                _ => Err(anyhow::anyhow!("No template for artifact type: {}", artifact_type)),
            }
        }
    }

    /// Determine appropriate file path for generated code
    fn determine_file_path(&self, objective: &str, related_files: &[PathBuf]) -> PathBuf {
        // Try to find the most relevant file
        if let Some(file) = related_files.first() {
            file.clone()
        } else {
            // Generate based on objective
            let module_name = objective
                .to_lowercase()
                .chars()
                .filter(|c| c.is_alphanumeric() || *c == '_')
                .collect::<String>()
                .replace(' ', "_");

            PathBuf::from("src").join(format!("{}.rs", module_name))
        }
    }

    /// Assess risk level of generated code
    fn assess_risk_level(&self, code: &str) -> RiskLevel {
        // Simple heuristics for risk assessment
        if code.contains("unsafe") || code.contains("mem::") {
            RiskLevel::High
        } else if code.contains("unwrap()") || code.contains("panic!") {
            RiskLevel::Medium
        } else {
            RiskLevel::Low
        }
    }

    /// Extract dependencies from code
    fn extract_dependencies(&self, code: &str) -> Vec<String> {
        let mut deps = Vec::new();

        for line in code.lines() {
            if line.trim().starts_with("use ") {
                deps.push(line.trim().to_string());
            }
        }

        deps
    }

    /// Calculate confidence score for context
    fn calculate_context_confidence(
        segment: &StorySegment,
        patterns: &[CodePattern],
    ) -> f32 {
        let mut confidence = 0.5; // Base confidence

        // More context in segment = better context
        // Use content length as proxy for context richness
        confidence += ((segment.content.len() / 100) as f32 * 0.05).min(0.2);

        // Applicable patterns increase confidence
        confidence += (patterns.len() as f32 * 0.1).min(0.2);

        // Recent segment = higher confidence
        let age = Utc::now().signed_duration_since(segment.created_at);
        if age.num_hours() < 24 {
            confidence += 0.1;
        }

        confidence.min(1.0)
    }

    /// Analyze error for fix generation
    async fn analyze_error(
        &self,
        error: &str,
        context: &GenerationContext,
    ) -> Result<ErrorAnalysis> {
        // Simple error analysis - in real implementation would be more sophisticated
        let file_path = context.related_files.first()
            .cloned()
            .unwrap_or_else(|| PathBuf::from("src/lib.rs"));

        Ok(ErrorAnalysis {
            error_type: self.classify_error(error),
            file_path,
            line_number: None,
            suggested_fix: None,
        })
    }

    /// Classify error type
    fn classify_error(&self, error: &str) -> String {
        if error.contains("borrow") {
            "borrow_checker".to_string()
        } else if error.contains("type") {
            "type_error".to_string()
        } else if error.contains("trait") {
            "trait_error".to_string()
        } else {
            "generic_error".to_string()
        }
    }

    /// Apply pattern template with substitutions
    fn apply_pattern_template(&self, template: &str, analysis: &ErrorAnalysis) -> Result<String> {
        let mut result = template.to_string();
        result = result.replace("{{error_type}}", &analysis.error_type);
        result = result.replace("{{file_path}}", &analysis.file_path.display().to_string());
        Ok(result)
    }

    /// Generate error fix code
    fn generate_error_fix_code(&self, error: &str, analysis: &ErrorAnalysis) -> Result<String> {
        // Simple template-based fix generation
        match analysis.error_type.as_str() {
            "borrow_checker" => Ok(format!(
                "// Fix for borrow checker error: {}\n// Consider using Arc<RwLock<T>> or cloning",
                error
            )),
            "type_error" => Ok(format!(
                "// Fix for type error: {}\n// Ensure types match or add type conversions",
                error
            )),
            _ => Ok(format!("// TODO: Fix error: {}", error)),
        }
    }

    /// Generate missing tests based on insight
    async fn generate_missing_tests(
        &self,
        insight: &str,
        context: &GenerationContext,
    ) -> Result<GeneratedCode> {
        // Extract what needs testing from insight
        let test_target = insight.split("missing test").nth(1)
            .unwrap_or("unknown")
            .trim();

        let test_content = format!(
            r#"#[test]
fn test_{}() {{
    // Generated from insight: {}
    // TODO: Implement test
    assert!(true);
}}"#,
            test_target.replace(' ', "_"),
            insight
        );

        Ok(GeneratedCode {
            artifact_type: GeneratedArtifactType::Test,
            file_path: PathBuf::from("tests/generated_tests.rs"),
            content: test_content,
            description: format!("Test for {}", test_target),
            confidence: context.confidence * 0.8,
            risk_level: RiskLevel::Low,
            dependencies: vec![],
            tests: vec![],
        })
    }

    /// Generate refactoring based on insight
    async fn generate_refactoring(
        &self,
        insight: &str,
        context: &GenerationContext,
    ) -> Result<GeneratedCode> {
        let refactor_content = format!(
            "// Refactoring suggestion based on: {}\n// TODO: Implement refactoring",
            insight
        );

        Ok(GeneratedCode {
            artifact_type: GeneratedArtifactType::Refactoring,
            file_path: context.related_files.first()
                .cloned()
                .unwrap_or_else(|| PathBuf::from("src/lib.rs")),
            content: refactor_content,
            description: "Refactoring suggestion".to_string(),
            confidence: context.confidence * 0.7,
            risk_level: RiskLevel::Medium,
            dependencies: vec![],
            tests: vec![],
        })
    }

    /// Generate documentation based on insight
    async fn generate_documentation(
        &self,
        insight: &str,
        context: &GenerationContext,
    ) -> Result<GeneratedCode> {
        let doc_content = format!(
            r#"//! Documentation generated from story context
//!
//! {}
//!
//! ## Story Context
//! {}
"#,
            insight,
            context.story_segment.content
                .lines()
                .take(3)
                .map(|line| format!("//! - {}", line))
                .collect::<Vec<_>>()
                .join("\n")
        );

        Ok(GeneratedCode {
            artifact_type: GeneratedArtifactType::Documentation,
            file_path: PathBuf::from("docs/generated_docs.md"),
            content: doc_content,
            description: "Generated documentation".to_string(),
            confidence: context.confidence,
            risk_level: RiskLevel::Low,
            dependencies: vec![],
            tests: vec![],
        })
    }

    /// Load learned patterns from memory
    async fn load_patterns_from_memory(memory: &CognitiveMemory) -> Result<HashMap<String, CodePattern>> {
        let mut patterns = HashMap::new();

        // Search for code patterns in memory
        let pattern_memories = memory.retrieve_similar("code pattern template", 20).await?;

        for mem in pattern_memories {
            // Try to parse as code pattern
            if let Ok(pattern) = serde_json::from_str::<CodePattern>(&mem.content) {
                patterns.insert(pattern.pattern_id.clone(), pattern);
            }
        }

        // Add some default patterns if none found
        if patterns.is_empty() {
            patterns.insert(
                "default_function".to_string(),
                CodePattern {
                    pattern_id: "default_function".to_string(),
                    pattern_type: CodePatternType::FunctionStructure,
                    description: "Default function structure".to_string(),
                    template: r#"/// {{description}}
pub fn {{name}}({{params}}) -> Result<{{return_type}}> {
    {{body}}
}"#.to_string(),
                    usage_count: 0,
                    success_rate: 0.0,
                    applicable_contexts: vec!["function".to_string()],
                },
            );
        }

        Ok(patterns)
    }

    /// Learn from generation outcome
    pub async fn learn_from_outcome(
        &self,
        artifact: &GeneratedCode,
        success: bool,
        feedback: Option<String>,
    ) -> Result<()> {
        // Update generation history
        let mut history = self.generation_history.write().await;
        if let Some(record) = history.iter_mut().find(|r| {
            r.artifact.file_path == artifact.file_path &&
            r.artifact.description == artifact.description
        }) {
            record.success = success;
            record.feedback = feedback.clone();
        }

        // Update pattern success rates if applicable
        if success {
            let mut patterns = self.learned_patterns.write().await;
            for pattern in patterns.values_mut() {
                // Simple heuristic - could be more sophisticated
                if artifact.content.contains(&pattern.description) {
                    pattern.usage_count += 1;
                    pattern.success_rate = (pattern.success_rate * (pattern.usage_count - 1) as f32
                        + 1.0) / pattern.usage_count as f32;
                }
            }
        }

        // Store learning in memory
        self.memory
            .store(
                format!(
                    "Code generation outcome: {} - {}",
                    artifact.description,
                    if success { "successful" } else { "failed" }
                ),
                vec![feedback.unwrap_or_default()],
                MemoryMetadata {
                    source: "story_driven_code_gen".to_string(),
                    tags: vec!["learning".to_string(), "code_generation".to_string()],
                    timestamp: chrono::Utc::now(),
                    expiration: None,
                    importance: if success { 0.7 } else { 0.9 },
                    associations: vec![],
                    context: Some("code generation learning".to_string()),
                    created_at: chrono::Utc::now(),
                    accessed_count: 0,
                    last_accessed: None,
                    version: 1,
                    category: "code_generation".to_string(),
                },
            )
            .await?;

        Ok(())
    }
    
    /// Generate code based on a prompt (public interface)
    pub async fn generate_code(
        &self,
        prompt: &str,
        context: Option<String>,
        requirements: Vec<String>,
    ) -> Result<GeneratedCode> {
        info!("🎨 Generating code for prompt: {}", prompt);
        
        // Create a basic story segment from the prompt
        let story_id = self.story_engine.create_story(
            StoryType::Task {
                task_id: uuid::Uuid::new_v4().to_string(),
                parent_story: None,
            },
            format!("Code Generation: {}", prompt),
            prompt.to_string(),
            vec!["code-generation".to_string()],
            crate::story::Priority::Medium,
        ).await?;
        
        // Generate with story context
        let results = self.generate_from_story_context(&story_id, None).await?;
        
        // Return the first generated code or create a default one
        Ok(results.into_iter().next().unwrap_or_else(|| {
            GeneratedCode {
                artifact_type: GeneratedArtifactType::Function,
                file_path: PathBuf::from("src/generated.rs"),
                content: format!("// Generated code for: {}\n// TODO: Implement", prompt),
                description: prompt.to_string(),
                confidence: 0.5,
                risk_level: RiskLevel::Low,
                dependencies: vec![],
                tests: vec![],
            }
        }))
    }
    
    /// Create a new story-driven code generator with minimal dependencies
    /// This is useful for initialization in contexts where not all dependencies are available
    pub async fn new_with_defaults(
        story_engine: Arc<StoryEngine>,
        memory: Arc<CognitiveMemory>,
        tool_manager: Arc<crate::tools::IntelligentToolManager>,
    ) -> Result<Self> {
        info!("🎨 Initializing Story-Driven Code Generator with defaults");
        
        let config = StoryDrivenCodeGenConfig::default();
        
        // Create minimal self-modification pipeline
        let self_modify = Arc::new(SelfModificationPipeline::new(
            PathBuf::from("."),
            memory.clone(),
        ).await?);
        
        // Create test generator
        let test_generator = Arc::new(
            TestGenerator::new(TestGeneratorConfig::default(), memory.clone()).await?
        );
        
        // Create code analyzer
        let code_analyzer = Arc::new(CodeAnalyzer::new(memory.clone()).await?);
        
        // Load learned patterns from memory
        let patterns = Self::load_patterns_from_memory(&memory).await?;
        
        Ok(Self {
            config,
            story_engine,
            self_modify,
            test_generator,
            code_analyzer,
            memory,
            inference_engine: None, // No inference engine by default
            learned_patterns: Arc::new(RwLock::new(patterns)),
            generation_history: Arc::new(RwLock::new(Vec::new())),
        })
    }
    
    /// Extract function name from objective
    fn extract_function_name(&self, objective: &str) -> String {
        // Convert objective to snake_case function name
        let words: Vec<&str> = objective.split_whitespace()
            .filter(|w| !["the", "a", "an", "to", "for", "of", "in", "on", "at", "by"].contains(w))
            .take(3)
            .collect();
        
        if words.is_empty() {
            "generated_function".to_string()
        } else {
            words.join("_").to_lowercase()
                .chars()
                .filter(|c| c.is_alphanumeric() || *c == '_')
                .collect()
        }
    }
    
    /// Extract test name from objective
    fn extract_test_name(&self, objective: &str) -> String {
        format!("test_{}", self.extract_function_name(objective))
    }
    
    /// Extract module name from objective
    fn extract_module_name(&self, objective: &str) -> String {
        self.extract_function_name(objective)
    }
    
    /// Infer function signature from objective and context
    fn infer_function_signature(&self, objective: &str, _context: &GenerationContext) -> (String, String) {
        // Simple heuristics for common patterns
        let objective_lower = objective.to_lowercase();
        
        let params = if objective_lower.contains("process") || objective_lower.contains("handle") {
            "input: &str"
        } else if objective_lower.contains("create") || objective_lower.contains("new") {
            ""
        } else if objective_lower.contains("update") || objective_lower.contains("modify") {
            "&mut self, value: String"
        } else {
            "&self"
        }.to_string();
        
        let return_type = if objective_lower.contains("check") || objective_lower.contains("is") {
            "bool"
        } else if objective_lower.contains("get") || objective_lower.contains("find") {
            "Option<String>"
        } else {
            "Result<()>"
        }.to_string();
        
        (params, return_type)
    }
    
    /// Generate function body based on objective
    fn generate_function_body(&self, objective: &str, context: &GenerationContext) -> String {
        let mut body = String::new();
        
        // Add pattern-based logic if available
        if let Some(pattern) = context.applicable_patterns.first() {
            body.push_str(&format!("    // Applying pattern: {:?}\n", pattern.pattern_type));
        }
        
        // Add basic implementation based on objective keywords
        let objective_lower = objective.to_lowercase();
        
        if objective_lower.contains("validate") {
            body.push_str("    // Validation logic\n");
            body.push_str("    if input.is_empty() {\n");
            body.push_str("        return Err(anyhow::anyhow!(\"Input cannot be empty\"));\n");
            body.push_str("    }\n");
        } else if objective_lower.contains("process") {
            body.push_str("    // Processing logic\n");
            body.push_str("    let processed = input.trim().to_string();\n");
            body.push_str("    result = processed;\n");
        } else if objective_lower.contains("calculate") {
            body.push_str("    // Calculation logic\n");
            body.push_str("    let calculated_value = 42; // Placeholder calculation\n");
            body.push_str("    result = calculated_value;\n");
        } else {
            body.push_str("    // Core implementation\n");
            body.push_str("    // Add specific logic based on requirements\n");
        }
        
        body
    }
    
    /// Generate test setup code
    fn generate_test_setup(&self, objective: &str, _context: &GenerationContext) -> String {
        let objective_lower = objective.to_lowercase();
        
        if objective_lower.contains("database") || objective_lower.contains("storage") {
            "let temp_dir = tempfile::tempdir()?;\n    let db_path = temp_dir.path().join(\"test.db\");"
        } else if objective_lower.contains("network") || objective_lower.contains("http") {
            "let mock_server = mockito::Server::new();"
        } else if objective_lower.contains("file") || objective_lower.contains("path") {
            "let test_file = PathBuf::from(\"/tmp/test_file.txt\");\n    std::fs::write(&test_file, \"test content\")?;"
        } else {
            "// No special setup required"
        }.to_string()
    }
    
    /// Generate test assertions
    fn generate_test_assertions(&self, objective: &str, _context: &GenerationContext) -> String {
        let objective_lower = objective.to_lowercase();
        
        if objective_lower.contains("error") || objective_lower.contains("fail") {
            "assert!(result.is_err(), \"Should return an error\");"
        } else if objective_lower.contains("empty") || objective_lower.contains("none") {
            "assert!(result.unwrap().is_none(), \"Should return None\");"
        } else if objective_lower.contains("equal") || objective_lower.contains("match") {
            "assert_eq!(result.unwrap(), expected_value, \"Values should match\");"
        } else {
            "assert!(result.unwrap().is_some(), \"Should return a value\");"
        }.to_string()
    }
    
    /// Convert snake_case to PascalCase
    fn to_pascal_case(&self, s: &str) -> String {
        s.split('_')
            .map(|word| {
                let mut chars = word.chars();
                match chars.next() {
                    None => String::new(),
                    Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
                }
            })
            .collect()
    }
}

/// Error analysis for fix generation
#[derive(Debug)]
struct ErrorAnalysis {
    error_type: String,
    file_path: PathBuf,
    line_number: Option<usize>,
    suggested_fix: Option<String>,
}
