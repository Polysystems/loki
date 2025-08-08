//! Story-Driven Autonomous Codebase Maintenance
//!
//! This module connects the story engine with autonomous capabilities to provide
//! intelligent, context-aware codebase maintenance that learns from patterns and
//! maintains narrative coherence.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

use crate::cognitive::autonomous_loop::AutonomousLoop;
use crate::cognitive::pr_automation::PrAutomationSystem;
use crate::cognitive::self_modify::{ChangeType, CodeChange, RiskLevel, SelfModificationPipeline};
use crate::cognitive::story_driven_code_generation::{
    StoryDrivenCodeGenerator, StoryDrivenCodeGenConfig, GeneratedCode, GeneratedArtifactType,
};
use crate::cognitive::story_driven_pr_review::{
    StoryDrivenPrReview, StoryDrivenPrReviewConfig,
};
use crate::cognitive::story_driven_bug_detection::{
    StoryDrivenBugDetection, StoryDrivenBugDetectionConfig, BugFixResult,
};
use crate::cognitive::story_driven_testing::{
    StoryDrivenTesting, StoryDrivenTestingConfig, TestGenerationStrategy,
};
use crate::cognitive::story_driven_learning::{
    StoryDrivenLearning, StoryDrivenLearningConfig,
};
use crate::cognitive::story_driven_documentation::{
    StoryDrivenDocumentation, StoryDrivenDocumentationConfig,
};
use crate::cognitive::story_driven_dependencies::{
    StoryDrivenDependencies, StoryDrivenDependencyConfig,
};
use crate::cognitive::story_driven_quality::{
    StoryDrivenQuality, StoryDrivenQualityConfig,
};
use crate::cognitive::story_driven_refactoring::{
    StoryDrivenRefactoring, StoryDrivenRefactoringConfig,
};
use crate::cognitive::story_driven_performance::{
    StoryDrivenPerformance, StoryDrivenPerformanceConfig,
};
use crate::cognitive::story_driven_security::{
    StoryDrivenSecurity, StoryDrivenSecurityConfig, RiskSummary,
};
use crate::cognitive::test_generator::TestGenerator;
use crate::memory::{CognitiveMemory, MemoryMetadata, MemoryId};
use crate::story::{
    PlotType, StoryEngine, StoryId,
};
use crate::tasks::code_review::{CodeAnalysis, CodeReviewTask};
use crate::tools::code_analysis::CodeAnalyzer;
// use crate::tools::intelligent_manager::IntelligentToolManager; // Unused import

/// Configuration for story-driven autonomy
#[derive(Debug, Clone)]
pub struct StoryDrivenAutonomyConfig {
    /// Enable autonomous codebase maintenance
    pub enable_codebase_maintenance: bool,

    /// Enable story-driven code generation
    pub enable_code_generation: bool,

    /// Enable intelligent PR review
    pub enable_pr_review: bool,

    /// Enable autonomous bug detection and fixing
    pub enable_bug_fixing: bool,

    /// Enable learning from codebase patterns
    pub enable_pattern_learning: bool,

    /// Enable autonomous documentation generation
    pub enable_documentation: bool,

    /// Enable intelligent dependency management
    pub enable_dependency_management: bool,

    /// Enable autonomous testing
    pub enable_testing: bool,

    /// Enable code quality monitoring
    pub enable_quality_monitoring: bool,

    /// Enable autonomous refactoring
    pub enable_refactoring: bool,

    /// Enable performance monitoring
    pub enable_performance_monitoring: bool,

    /// Enable security vulnerability detection
    pub enable_security_monitoring: bool,

    /// Maintenance check interval
    pub maintenance_interval: Duration,

    /// Repository path
    pub repo_path: PathBuf,

    /// Maximum risk level for autonomous actions
    pub max_risk_level: RiskLevel,
}

impl Default for StoryDrivenAutonomyConfig {
    fn default() -> Self {
        Self {
            enable_codebase_maintenance: true,
            enable_code_generation: true,
            enable_pr_review: true,
            enable_bug_fixing: true,
            enable_pattern_learning: true,
            enable_documentation: true,
            enable_dependency_management: true,
            enable_testing: true,
            enable_quality_monitoring: true,
            enable_refactoring: true,
            enable_performance_monitoring: true,
            enable_security_monitoring: true,
            maintenance_interval: Duration::from_secs(300), // 5 minutes
            repo_path: PathBuf::from("."),
            max_risk_level: RiskLevel::Medium,
        }
    }
}

/// Pattern learned from codebase analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodebasePattern {
    pub pattern_id: String,
    pub pattern_type: PatternType,
    pub description: String,
    pub occurrences: Vec<PatternOccurrence>,
    pub confidence: f32,
    pub applicable_contexts: Vec<String>,
    pub learned_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PatternType {
    ArchitecturalPattern,
    CodingConvention,
    TestingPattern,
    ErrorHandlingPattern,
    PerformancePattern,
    SecurityPattern,
    DocumentationPattern,
    InteractionPattern,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternOccurrence {
    pub file_path: PathBuf,
    pub line_range: Option<(usize, usize)>,
    pub example: String,
}

/// Story-driven autonomous system
#[derive(Clone)]
pub struct StoryDrivenAutonomy {
    config: StoryDrivenAutonomyConfig,
    story_engine: Arc<StoryEngine>,
    autonomous_loop: Arc<RwLock<AutonomousLoop>>,
    pr_automation: Option<Arc<PrAutomationSystem>>,
    self_modify: Arc<SelfModificationPipeline>,
    code_analyzer: Arc<CodeAnalyzer>,
    test_generator: Arc<TestGenerator>,
    code_review_task: Arc<CodeReviewTask>,
    code_generator: Arc<StoryDrivenCodeGenerator>,
    pr_reviewer: Option<Arc<StoryDrivenPrReview>>,
    bug_detector: Option<Arc<StoryDrivenBugDetection>>,
    test_system: Option<Arc<StoryDrivenTesting>>,
    learning_system: Option<Arc<StoryDrivenLearning>>,
    documentation_system: Option<Arc<StoryDrivenDocumentation>>,
    dependency_system: Option<Arc<StoryDrivenDependencies>>,
    quality_system: Option<Arc<StoryDrivenQuality>>,
    refactoring_system: Option<Arc<StoryDrivenRefactoring>>,
    performance_system: Option<Arc<StoryDrivenPerformance>>,
    security_system: Option<Arc<StoryDrivenSecurity>>,
    memory: Arc<CognitiveMemory>,

    /// Codebase story ID
    codebase_story_id: StoryId,

    /// Learned patterns
    learned_patterns: Arc<RwLock<HashMap<String, CodebasePattern>>>,

    /// Active maintenance tasks
    active_tasks: Arc<RwLock<Vec<MaintenanceTask>>>,
    /// Maintenance loop handle
    maintenance_handle: Arc<RwLock<Option<tokio::task::JoinHandle<()>>>>,
    /// Shutdown signal
    shutdown_tx: Arc<RwLock<Option<tokio::sync::oneshot::Sender<()>>>>,
}

#[derive(Debug, Clone)]
struct MaintenanceTask {
    pub task_id: String,
    pub task_type: MaintenanceTaskType,
    pub description: String,
    pub priority: f32,
    pub status: TaskStatus,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub story_segment_id: Option<String>,
}

#[derive(Debug, Clone)]
enum MaintenanceTaskType {
    BugFix,
    Refactoring,
    TestGeneration,
    DocumentationUpdate,
    DependencyUpdate,
    PerformanceOptimization,
    SecurityPatch,
    SecurityUpdate,
    CodeGeneration,
    PrReview,
}

#[derive(Debug, Clone, PartialEq)]
enum TaskStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
}

impl StoryDrivenAutonomy {
    /// Create a new story-driven autonomous system
    pub async fn new(
        config: StoryDrivenAutonomyConfig,
        story_engine: Arc<StoryEngine>,
        autonomous_loop: Arc<RwLock<AutonomousLoop>>,
        pr_automation: Option<Arc<PrAutomationSystem>>,
        self_modify: Arc<SelfModificationPipeline>,
        memory: Arc<CognitiveMemory>,
    ) -> Result<Self> {
        info!("🎭 Initializing Story-Driven Autonomous System");

        // Create or get codebase story
        let codebase_story_id = story_engine
            .create_codebase_story(
                config.repo_path.clone(),
                "rust".to_string() // Assuming Rust for now
            )
            .await?;

        // Initialize analyzers and generators
        let code_analyzer = Arc::new(CodeAnalyzer::new(memory.clone()).await?);
        let test_generator = Arc::new(
            TestGenerator::new(Default::default(), memory.clone()).await?
        );
        let code_review_task = Arc::new(CodeReviewTask::new(None, Some(memory.clone())));

        // Initialize story-driven code generator
        let code_gen_config = StoryDrivenCodeGenConfig {
            enable_function_generation: config.enable_code_generation,
            enable_test_generation: config.enable_testing,
            enable_doc_generation: config.enable_documentation,
            enable_refactoring: config.enable_refactoring,
            max_risk_level: config.max_risk_level,
            ..Default::default()
        };

        let code_generator = Arc::new(
            StoryDrivenCodeGenerator::new(
                code_gen_config,
                story_engine.clone(),
                self_modify.clone(),
                memory.clone(),
                None, // Inference engine optional
            ).await?
        );

        // Initialize story-driven PR reviewer if PR review is enabled
        let pr_reviewer = if config.enable_pr_review {
            let pr_review_config = StoryDrivenPrReviewConfig {
                enable_narrative_analysis: true,
                enable_pattern_review: true,
                enable_coherence_check: true,
                enable_suggestions: true,
                enable_auto_approval: false,
                auto_approval_threshold: 0.95,
                review_depth: crate::cognitive::story_driven_pr_review::ReviewDepth::Comprehensive,
                repo_path: config.repo_path.clone(),
            };

            Some(Arc::new(
                StoryDrivenPrReview::new(
                    pr_review_config,
                    story_engine.clone(),
                    pr_automation.clone(),
                    memory.clone(),
                    None, // GitHub client optional
                ).await?
            ))
        } else {
            None
        };

        // Initialize story-driven bug detector if bug fixing is enabled
        let bug_detector = if config.enable_bug_fixing {
            let bug_detection_config = StoryDrivenBugDetectionConfig {
                enable_pattern_detection: true,
                enable_anomaly_detection: true,
                enable_runtime_analysis: false,
                enable_auto_fix: true,
                enable_test_generation: config.enable_testing,
                auto_fix_threshold: 0.85,
                max_fix_risk_level: config.max_risk_level,
                detection_sensitivity: crate::cognitive::story_driven_bug_detection::DetectionSensitivity::Balanced,
                repo_path: config.repo_path.clone(),
            };

            Some(Arc::new(
                StoryDrivenBugDetection::new(
                    bug_detection_config,
                    story_engine.clone(),
                    self_modify.clone(),
                    code_generator.clone(),
                    memory.clone(),
                ).await?
            ))
        } else {
            None
        };

        // Initialize story-driven testing system if testing is enabled
        let test_system = if config.enable_testing {
            let testing_config = StoryDrivenTestingConfig {
                enable_test_generation: true,
                enable_test_execution: true,
                enable_coverage_tracking: true,
                enable_test_maintenance: true,
                enable_property_testing: true,
                enable_integration_tests: true,
                enable_performance_tests: false,
                min_coverage_threshold: 0.8,
                generation_strategy: TestGenerationStrategy::Adaptive,
                repo_path: config.repo_path.clone(),
                test_timeout: std::time::Duration::from_secs(60),
            };

            Some(Arc::new(
                StoryDrivenTesting::new(
                    testing_config,
                    story_engine.clone(),
                    self_modify.clone(),
                    memory.clone(),
                ).await?
            ))
        } else {
            None
        };

        // Record initialization in story
        let mut consequences = vec![
            "Autonomous maintenance enabled".to_string(),
            "Pattern learning activated".to_string(),
            "Code generation enabled".to_string(),
        ];

        if pr_reviewer.is_some() {
            consequences.push("Intelligent PR review enabled".to_string());
        }

        if bug_detector.is_some() {
            consequences.push("Autonomous bug detection and fixing enabled".to_string());
        }

        if test_system.is_some() {
            consequences.push("Autonomous testing and coverage tracking enabled".to_string());
        }

        // Initialize learning system
        let learning_system = if config.enable_pattern_learning {
            let learning_config = StoryDrivenLearningConfig {
                enable_pattern_extraction: true,
                enable_architectural_learning: true,
                enable_style_learning: true,
                enable_performance_learning: true,
                enable_security_learning: true,
                min_pattern_frequency: 3,
                pattern_confidence_threshold: 0.8,
                repo_path: config.repo_path.clone(),
            };

            Some(Arc::new(
                StoryDrivenLearning::new(
                    learning_config,
                    story_engine.clone(),
                    code_analyzer.clone(),
                    autonomous_loop.clone(),
                    memory.clone(),
                ).await?
            ))
        } else {
            None
        };

        if learning_system.is_some() {
            consequences.push("Pattern learning and architectural insights enabled".to_string());
        }

        // Initialize documentation system
        let documentation_system = if config.enable_documentation {
            let doc_config = StoryDrivenDocumentationConfig {
                enable_api_docs: true,
                enable_module_docs: true,
                enable_readme_generation: true,
                enable_inline_docs: true,
                enable_architecture_docs: true,
                doc_style: crate::cognitive::story_driven_documentation::DocumentationStyle::Comprehensive,
                min_complexity_threshold: 5.0,
                repo_path: config.repo_path.clone(),
            };

            Some(Arc::new(
                StoryDrivenDocumentation::new(
                    doc_config,
                    story_engine.clone(),
                    code_analyzer.clone(),
                    self_modify.clone(),
                    learning_system.clone(),
                    memory.clone(),
                ).await?
            ))
        } else {
            None
        };

        if documentation_system.is_some() {
            consequences.push("Autonomous documentation generation enabled".to_string());
        }

        // Initialize dependency management system
        let dependency_system = if config.enable_dependency_management {
            let dep_config = StoryDrivenDependencyConfig {
                enable_auto_update: true,
                enable_security_check: true,
                enable_unused_detection: true,
                enable_optimization: true,
                enable_license_check: true,
                update_strategy: crate::cognitive::story_driven_dependencies::UpdateStrategy::Balanced,
                max_auto_update_risk: RiskLevel::Medium,
                repo_path: config.repo_path.clone(),
            };

            Some(Arc::new(
                StoryDrivenDependencies::new(
                    dep_config,
                    story_engine.clone(),
                    code_analyzer.clone(),
                    self_modify.clone(),
                    memory.clone(),
                ).await?
            ))
        } else {
            None
        };

        if dependency_system.is_some() {
            consequences.push("Intelligent dependency management enabled".to_string());
        }

        // Initialize quality monitoring system
        let quality_system = if config.enable_quality_monitoring {
            let quality_config = StoryDrivenQualityConfig {
                enable_complexity_monitoring: true,
                enable_duplication_detection: true,
                enable_maintainability_scoring: true,
                enable_performance_analysis: true,
                enable_security_scanning: true,
                enable_style_checking: true,
                complexity_threshold: 10.0,
                duplication_threshold: 5.0,
                min_maintainability_score: 0.7,
                repo_path: config.repo_path.clone(),
            };

            Some(Arc::new(
                StoryDrivenQuality::new(
                    quality_config,
                    story_engine.clone(),
                    code_analyzer.clone(),
                    learning_system.clone(),
                    memory.clone(),
                ).await?
            ))
        } else {
            None
        };

        if quality_system.is_some() {
            consequences.push("Code quality monitoring enabled".to_string());
        }

        // Initialize refactoring system
        let refactoring_system = if config.enable_refactoring {
            let refactoring_config = StoryDrivenRefactoringConfig {
                enable_method_extraction: true,
                enable_variable_renaming: true,
                enable_consolidation: true,
                enable_pattern_application: true,
                enable_performance_refactoring: true,
                enable_architectural_refactoring: true,
                min_complexity_for_extraction: 10,
                max_method_length: 50,
                max_auto_refactor_risk: config.max_risk_level.clone(),
                repo_path: config.repo_path.clone(),
            };

            Some(Arc::new(
                StoryDrivenRefactoring::new(
                    refactoring_config,
                    story_engine.clone(),
                    code_analyzer.clone(),
                    self_modify.clone(),
                    quality_system.clone(),
                    learning_system.clone(),
                    memory.clone(),
                )
                .await?
            ))
        } else {
            None
        };

        if refactoring_system.is_some() {
            consequences.push("Intelligent refactoring suggestions enabled".to_string());
        }

        // Initialize performance monitoring system
        let performance_system = if config.enable_performance_monitoring {
            let performance_config = StoryDrivenPerformanceConfig {
                enable_runtime_monitoring: true,
                enable_static_analysis: true,
                enable_memory_profiling: true,
                enable_cpu_profiling: true,
                enable_benchmark_generation: true,
                enable_auto_optimization: true,
                degradation_threshold: 10.0,
                min_improvement_threshold: 5.0,
                max_optimization_risk: config.max_risk_level.clone(),
                monitoring_interval: config.maintenance_interval,
                repo_path: config.repo_path.clone(),
            };

            Some(Arc::new(
                StoryDrivenPerformance::new(
                    performance_config,
                    story_engine.clone(),
                    code_analyzer.clone(),
                    self_modify.clone(),
                    learning_system.clone(),
                    memory.clone(),
                )
                .await?
            ))
        } else {
            None
        };

        if performance_system.is_some() {
            consequences.push("Performance monitoring and optimization enabled".to_string());
        }

        // Initialize security monitoring system
        let security_system = if config.enable_security_monitoring {
            let security_config = StoryDrivenSecurityConfig {
                enable_static_analysis: true,
                enable_dependency_scanning: true,
                enable_runtime_monitoring: true,
                enable_secret_detection: true,
                enable_owasp_checks: true,
                enable_auto_patching: true,
                max_auto_fix_risk: config.max_risk_level.clone(),
                scan_interval: config.maintenance_interval,
                repo_path: config.repo_path.clone(),
            };

            Some(Arc::new(
                StoryDrivenSecurity::new(
                    security_config,
                    story_engine.clone(),
                    code_analyzer.clone(),
                    self_modify.clone(),
                    dependency_system.clone(),
                    memory.clone(),
                )
                .await?
            ))
        } else {
            None
        };

        if security_system.is_some() {
            consequences.push("Security vulnerability detection enabled".to_string());
        }

        story_engine
            .add_plot_point(
                codebase_story_id.clone(),
                PlotType::Goal {
                    objective: "Initialize autonomous codebase maintenance".to_string(),
                },
                vec![],
            )
            .await?;

        Ok(Self {
            config,
            story_engine,
            autonomous_loop,
            pr_automation,
            self_modify,
            code_analyzer,
            test_generator,
            code_review_task,
            code_generator,
            pr_reviewer,
            bug_detector,
            test_system,
            learning_system,
            documentation_system,
            dependency_system,
            quality_system,
            refactoring_system,
            performance_system,
            security_system,
            memory,
            codebase_story_id,
            learned_patterns: Arc::new(RwLock::new(HashMap::new())),
            active_tasks: Arc::new(RwLock::new(Vec::new())),
            maintenance_handle: Arc::new(RwLock::new(None)),
            shutdown_tx: Arc::new(RwLock::new(None)),
        })
    }

    /// Start the autonomous maintenance loop (non-blocking)
    pub async fn start(&self) -> Result<()> {
        info!("🚀 Starting story-driven autonomous maintenance");

        // Check if already running
        if self.maintenance_handle.read().await.is_some() {
            warn!("Story-driven autonomy is already running");
            return Ok(());
        }

        // Create shutdown channel
        let (shutdown_tx, mut shutdown_rx) = tokio::sync::oneshot::channel();
        *self.shutdown_tx.write().await = Some(shutdown_tx);

        // Clone necessary components for the task
        let config = self.config.clone();
        let self_clone = self.clone();

        // Spawn the maintenance loop
        let handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(config.maintenance_interval);

            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        // Run maintenance cycle
                        if let Err(e) = self_clone.run_maintenance_cycle().await {
                            error!("Maintenance cycle error: {}", e);

                            // Record error in story
                            let _ = self_clone.story_engine
                                .add_plot_point(
                                    self_clone.codebase_story_id.clone(),
                                    PlotType::Issue {
                                        error: format!("Maintenance cycle failed: {}", e),
                                        resolved: false,
                                    },
                                    vec![],
                                )
                                .await;
                        }
                    }
                    _ = &mut shutdown_rx => {
                        info!("🛑 Stopping story-driven autonomous maintenance");
                        break;
                    }
                }
            }
        });

        *self.maintenance_handle.write().await = Some(handle);

        // Record start in story
        self.story_engine
            .add_plot_point(
                self.codebase_story_id.clone(),
                PlotType::Goal {
                    objective: "Started autonomous story-driven maintenance".to_string(),
                },
                vec!["autonomy".to_string(), "started".to_string()],
            )
            .await?;

        Ok(())
    }

    /// Stop the autonomous maintenance loop
    pub async fn stop(&self) -> Result<()> {
        info!("🛑 Stopping story-driven autonomous maintenance");

        // Send shutdown signal
        if let Some(shutdown_tx) = self.shutdown_tx.write().await.take() {
            let _ = shutdown_tx.send(());
        }

        // Wait for the task to complete
        if let Some(handle) = self.maintenance_handle.write().await.take() {
            let _ = handle.await;
        }

        // Record stop in story
        self.story_engine
            .add_plot_point(
                self.codebase_story_id.clone(),
                PlotType::Goal {
                    objective: "Stopped autonomous story-driven maintenance".to_string(),
                },
                vec!["autonomy".to_string(), "stopped".to_string()],
            )
            .await?;

        Ok(())
    }

    /// Review a PR with story context
    pub async fn review_pr_with_story(&self, pr_url: &str) -> Result<serde_json::Value> {
        info!("🔍 Reviewing PR with story context: {}", pr_url);
        
        // Extract PR number from URL
        let pr_number = pr_url
            .split('/')
            .last()
            .and_then(|s| s.parse::<u32>().ok())
            .ok_or_else(|| anyhow::anyhow!("Invalid PR URL: {}", pr_url))?;
        
        // Get current codebase narrative
        let codebase_state = self.analyze_codebase().await?;
        
        // Create story context for the PR
        let pr_story_id = self.story_engine
            .create_story(
                crate::story::StoryType::Task { 
                    task_id: format!("pr_review_{}", pr_number),
                    parent_story: Some(self.codebase_story_id.clone()),
                },
                format!("PR #{} Review", pr_number),
                format!("Story-driven review of pull request #{}", pr_number),
                vec!["pr-review".to_string(), "story-driven".to_string()],
                crate::story::Priority::High,
            )
            .await?;
        
        // Perform the review if pr_reviewer is available
        let review_result = if let Some(pr_reviewer) = &self.pr_reviewer {
            match pr_reviewer.review_pr(pr_number).await {
                Ok(result) => {
                    // Record review in story
                    self.story_engine
                        .add_plot_point(
                            pr_story_id.clone(),
                            PlotType::Discovery {
                                insight: format!("PR #{} review: {:?} (confidence: {:.2})", 
                                    pr_number, result.overall_assessment, result.confidence),
                            },
                            vec!["pr-review".to_string()],
                        )
                        .await?;
                    
                    // Analyze narrative impact
                    let narrative_impact = self.analyze_narrative_impact(&result, &codebase_state).await?;
                    
                    serde_json::json!({
                        "pr_number": pr_number,
                        "overall_assessment": format!("{:?}", result.overall_assessment),
                        "confidence": result.confidence,
                        "suggestions": result.suggestions,
                        "narrative_impact": narrative_impact.get("impact").unwrap_or(&serde_json::Value::Null),
                        "character_changes": narrative_impact.get("character_changes").unwrap_or(&serde_json::Value::Null),
                        "plot_advancement": narrative_impact.get("plot_advancement").unwrap_or(&serde_json::Value::Null),
                        "recommendation": if result.confidence > 0.8 {
                            "Approve with confidence"
                        } else if result.confidence > 0.6 {
                            "Approve with minor suggestions"
                        } else {
                            "Request changes"
                        }
                    })
                }
                Err(e) => {
                    error!("Failed to review PR: {}", e);
                    serde_json::json!({
                        "error": e.to_string(),
                        "narrative_impact": "Unable to assess due to review error",
                        "recommendation": "Manual review required"
                    })
                }
            }
        } else {
            // Fallback: analyze based on story context alone
            serde_json::json!({
                "narrative_impact": "PR would advance the story by implementing requested features",
                "character_changes": "Unable to assess without full PR review",
                "plot_advancement": "Incremental progress towards project goals",
                "recommendation": "Manual review required - PR reviewer not available"
            })
        };
        
        Ok(review_result)
    }
    
    /// Analyze narrative impact of changes
    async fn analyze_narrative_impact(
        &self,
        review_result: &crate::cognitive::story_driven_pr_review::StoryDrivenReviewResult,
        codebase_state: &CodebaseState,
    ) -> Result<serde_json::Value> {
        let mut impact = serde_json::Map::new();
        
        // Assess impact on codebase "story"
        let story_impact = match review_result.overall_assessment {
            crate::cognitive::story_driven_pr_review::ReviewAssessment::Approve => {
                "Positive advancement of the codebase narrative"
            }
            crate::cognitive::story_driven_pr_review::ReviewAssessment::ApproveWithSuggestions => {
                "Advancement with opportunities for narrative refinement"
            }
            crate::cognitive::story_driven_pr_review::ReviewAssessment::RequestChanges => {
                "Potential narrative conflicts that need resolution"
            }
            crate::cognitive::story_driven_pr_review::ReviewAssessment::NeedsDiscussion => {
                "Significant narrative divergence from project vision"
            }
        };
        impact.insert("impact".to_string(), serde_json::Value::String(story_impact.to_string()));
        
        // Check for character (component) changes
        let character_changes = if review_result.suggestions.iter().any(|s| s.description.contains("component") || s.description.contains("module")) {
            "Introduces or modifies key components in the system"
        } else {
            "No significant character changes detected"
        };
        impact.insert("character_changes".to_string(), serde_json::Value::String(character_changes.to_string()));
        
        // Assess plot advancement
        let plot_advancement = if codebase_state.critical_issues > 0 && 
            review_result.suggestions.iter().any(|s| s.description.contains("fix") || s.description.contains("resolve")) {
            "Resolves critical plot points (bugs/issues)"
        } else if review_result.suggestions.iter().any(|s| s.description.contains("feature") || s.description.contains("implement")) {
            "Advances the plot with new capabilities"
        } else {
            "Incremental narrative progression"
        };
        impact.insert("plot_advancement".to_string(), serde_json::Value::String(plot_advancement.to_string()));
        
        Ok(serde_json::Value::Object(impact))
    }

    /// Run a single maintenance cycle
    async fn run_maintenance_cycle(&self) -> Result<()> {
        debug!("Running autonomous maintenance cycle");

        // 1. Analyze codebase for issues and patterns
        if self.config.enable_quality_monitoring {
            self.analyze_codebase().await?;
        }

        // 2. Detect and fix bugs
        if self.config.enable_bug_fixing {
            self.detect_and_fix_bugs().await?;
        }

        // 3. Generate and maintain tests
        if self.config.enable_testing {
            self.generate_and_maintain_tests().await?;
        }

        // 4. Learn from patterns
        if self.config.enable_pattern_learning {
            self.learn_patterns().await?;
        }

        // 5. Generate and update documentation
        if self.config.enable_documentation {
            self.generate_and_update_documentation().await?;
        }

        // 6. Manage dependencies
        if self.config.enable_dependency_management {
            self.manage_dependencies().await?;
        }

        // 7. Generate maintenance tasks from story context
        self.generate_maintenance_tasks().await?;

        // 8. Generate code from story context
        if self.config.enable_code_generation {
            self.generate_code_from_story().await?;
        }

        // 9. Review pending PRs
        if self.config.enable_pr_review {
            self.review_pending_prs().await?;
        }

        // 10. Analyze and suggest refactorings
        if self.config.enable_refactoring {
            self.analyze_and_suggest_refactorings().await?;
        }

        // 11. Monitor and optimize performance
        if self.config.enable_performance_monitoring {
            self.monitor_and_optimize_performance().await?;
        }

        // 12. Scan for security vulnerabilities
        if self.config.enable_security_monitoring {
            self.scan_and_fix_vulnerabilities().await?;
        }

        // 13. Execute high-priority tasks
        self.execute_tasks().await?;

        // 14. Update story with progress
        self.update_story_progress().await?;

        Ok(())
    }

    /// Analyze codebase for issues and improvement opportunities
    async fn analyze_codebase(&self) -> Result<CodebaseState> {
        info!("🔍 Analyzing codebase for autonomous maintenance");

        // Use quality system if available
        if let Some(quality_system) = &self.quality_system {
            // Perform comprehensive quality analysis
            let quality_analysis = quality_system.analyze_quality().await?;

            info!(
                "📊 Code quality: {:.1}% health, {} issues found",
                quality_analysis.metrics.overall_health * 100.0,
                quality_analysis.issues.len()
            );

            // Create tasks for critical quality issues
            let mut tasks = self.active_tasks.write().await;

            for issue in quality_analysis.issues.iter()
                .filter(|i| i.severity == crate::cognitive::story_driven_quality::IssueSeverity::Critical
                    || i.severity == crate::cognitive::story_driven_quality::IssueSeverity::Error)
            {
                let task = MaintenanceTask {
                    task_id: format!("quality_{}", uuid::Uuid::new_v4()),
                    task_type: match issue.issue_type {
                        crate::cognitive::story_driven_quality::QualityIssueType::HighComplexity => MaintenanceTaskType::Refactoring,
                        crate::cognitive::story_driven_quality::QualityIssueType::CodeDuplication => MaintenanceTaskType::Refactoring,
                        crate::cognitive::story_driven_quality::QualityIssueType::SecurityVulnerability => MaintenanceTaskType::SecurityPatch,
                        crate::cognitive::story_driven_quality::QualityIssueType::PerformanceIssue => MaintenanceTaskType::PerformanceOptimization,
                        crate::cognitive::story_driven_quality::QualityIssueType::MissingTests => MaintenanceTaskType::TestGeneration,
                        crate::cognitive::story_driven_quality::QualityIssueType::MissingDocumentation => MaintenanceTaskType::DocumentationUpdate,
                        _ => MaintenanceTaskType::Refactoring,
                    },
                    description: issue.description.clone(),
                    priority: issue.metrics_impact,
                    status: TaskStatus::Pending,
                    created_at: Utc::now(),
                    story_segment_id: None,
                };

                if !tasks.iter().any(|t| t.task_id == task.task_id) {
                    tasks.push(task);
                }
            }

            // Monitor quality continuously
            quality_system.monitor_quality().await?;

            // Store quality report
            let report = quality_system.get_quality_report().await?;
            self.memory
                .store(
                    "quality_report".to_string(),
                    vec![serde_json::to_string(&report)?],
                    MemoryMetadata {
                        source: "story_driven_autonomy".to_string(),
                        tags: vec!["quality".to_string(), "report".to_string()],
                        importance: 0.8,
                        associations: vec![MemoryId::from_string("monitoring".to_string())],
                        context: Some("Quality report for autonomous system".to_string()),
                        created_at: Utc::now(),
                        accessed_count: 0,
                        last_accessed: None,
                        version: 1,
                    category: "story".to_string(),
                        timestamp: Utc::now(),
                        expiration: None,
                    },
                )
                .await?;

            return Ok(CodebaseState {
                total_issues: quality_analysis.issues.len(),
                critical_issues: quality_analysis.issues.iter().filter(|i| matches!(i.severity, crate::cognitive::story_driven_quality::IssueSeverity::Critical)).count(),
                test_coverage: quality_analysis.metrics.test_coverage,
                code_quality_score: quality_analysis.metrics.overall_health,
            });
        }

        // Fallback to basic analysis
        let src_path = self.config.repo_path.join("src");
        let analysis_results = self.analyze_directory(&src_path).await?;

        // Process analysis results
        let mut total_issues = 0;
        let mut critical_issues = Vec::new();

        for (file_path, analysis) in analysis_results {
            total_issues += analysis.issues.len();

            // Collect critical issues from structured issue fields
            for issue in &analysis.security_issues {
                use crate::tasks::code_review::SecuritySeverity;
                if matches!(issue.severity, SecuritySeverity::Critical | SecuritySeverity::High) {
                    critical_issues.push((file_path.clone(), issue.description.clone()));
                }
            }

            // Also check for any issues in the general issues field (these are strings)
            for issue_str in &analysis.issues {
                // Since these are strings, we'll treat them all as potential critical issues for now
                critical_issues.push((file_path.clone(), issue_str.clone()));
            }

            // Store analysis in memory
            self.memory
                .store(
                    format!(
                        "Codebase analysis: {} - {} issues, complexity {:.2}",
                        file_path.display(),
                        analysis.issues.len(),
                        analysis.complexity_score
                    ),
                    vec![],
                    MemoryMetadata {
                        source: "story_driven_autonomy".to_string(),
                        tags: vec!["analysis".to_string(), "codebase".to_string()],
                        importance: 0.6,
                        associations: vec![],
                        context: Some("autonomous codebase analysis".to_string()),
                        created_at: chrono::Utc::now(),
                        accessed_count: 0,
                        last_accessed: None,
                        version: 1,
                    category: "story".to_string(),
                        timestamp: chrono::Utc::now(),
                        expiration: None,
                    },
                )
                .await?;
        }

        // Create tasks for critical issues
        let critical_issues_count = critical_issues.len();
        if !critical_issues.is_empty() {
            let mut tasks = self.active_tasks.write().await;

            for (file_path, issue) in critical_issues {
                let task = MaintenanceTask {
                    task_id: format!("fix_issue_{}", uuid::Uuid::new_v4()),
                    task_type: MaintenanceTaskType::BugFix,
                    description: format!("Fix {}: {}", file_path.display(), issue),
                    priority: 0.9,
                    status: TaskStatus::Pending,
                    created_at: Utc::now(),
                    story_segment_id: None,
                };

                tasks.push(task);
            }
        }

        // Record analysis in story
        self.story_engine
            .add_plot_point(
                self.codebase_story_id.clone(),
                PlotType::Discovery {
                    insight: format!(
                        "Codebase analysis found {} issues",
                        total_issues
                    ),
                },
                vec![],
            )
            .await?;

        // Return codebase state
        Ok(CodebaseState {
            total_issues,
            critical_issues: critical_issues_count,
            test_coverage: 0.0, // TODO: Get from test system
            code_quality_score: 0.7, // TODO: Get from quality system
        })
    }

    /// Analyze a directory recursively
    async fn analyze_directory(
        &self,
        dir_path: &Path,
    ) -> Result<HashMap<PathBuf, CodeAnalysis>> {
        let mut results = HashMap::new();

        let mut entries = tokio::fs::read_dir(dir_path).await?;

        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();

            if path.is_dir() {
                // Skip certain directories
                let dir_name = path.file_name().unwrap_or_default().to_str().unwrap_or("");
                if !matches!(dir_name, "target" | ".git" | "node_modules") {
                    let sub_results = Box::pin(self.analyze_directory(&path)).await?;
                    results.extend(sub_results);
                }
            } else if path.is_file() && path.extension().map_or(false, |ext| ext == "rs") {
                // Analyze Rust files
                match self.code_analyzer.analyze_file(&path).await {
                    Ok(analysis) => {
                        // Convert to CodeAnalysis type expected by code review
                        let code_analysis = CodeAnalysis {
                            complexity_score: analysis.complexity as f32,
                            maintainability_score: 0.7, // Default
                            security_issues: vec![],
                            performance_issues: vec![],
                            style_issues: vec![],
                            cognitive_patterns: vec![],
                            suggestions: vec![],
                            issues: vec![], // Added missing issues field
                        };

                        results.insert(path.clone(), code_analysis);
                    }
                    Err(e) => {
                        warn!("Failed to analyze {}: {}", path.display(), e);
                    }
                }
            }
        }

        Ok(results)
    }

    /// Learn patterns from codebase
    async fn learn_patterns(&self) -> Result<()> {
        debug!("Learning patterns from codebase");

        // Use the dedicated learning system if available
        if let Some(learning_system) = &self.learning_system {
            info!("🧠 Running comprehensive pattern learning");

            let learning_result = learning_system.learn_from_codebase().await?;

            info!(
                "Learned {} patterns with {} insights",
                learning_result.patterns_extracted,
                learning_result.insights_gained.len()
            );

            // Store insights in memory
            for insight in &learning_result.insights_gained {
                self.memory
                    .store(
                        "learned_insight".to_string(),
                        vec![serde_json::to_string(&insight)?],
                        MemoryMetadata {
                            source: "story_driven_autonomy".to_string(),
                            tags: vec!["learning".to_string(), "pattern".to_string()],
                            importance: 0.8,
                            associations: vec![MemoryId::from_string("codebase_analysis".to_string())],
                            context: Some("Learned insight from codebase analysis".to_string()),
                            created_at: Utc::now(),
                            accessed_count: 0,
                            last_accessed: None,
                            version: 1,
                    category: "story".to_string(),
                            timestamp: Utc::now(),
                            expiration: None,
                        },
                    )
                    .await?;
            }

            // Apply recommendations
            if !learning_result.recommendations.is_empty() {
                info!("Generated {} recommendations from learning", learning_result.recommendations.len());

                // Create tasks for high-priority recommendations
                let mut tasks = self.active_tasks.write().await;
                for recommendation in learning_result.recommendations.iter().take(3) {
                    let task = MaintenanceTask {
                        task_id: format!("learn_rec_{}", uuid::Uuid::new_v4()),
                        task_type: MaintenanceTaskType::Refactoring,
                        description: recommendation.clone(),
                        priority: 0.6,
                        status: TaskStatus::Pending,
                        created_at: Utc::now(),
                        story_segment_id: None,
                    };

                    if !tasks.iter().any(|t| t.description == task.description) {
                        tasks.push(task);
                    }
                }
            }

            return Ok(());
        }

        // Fallback to basic pattern learning if no dedicated system
        let pattern_queries = vec![
            "error handling pattern",
            "test structure",
            "module organization",
            "async pattern",
            "trait implementation",
        ];

        let mut new_patterns = Vec::new();

        for query in pattern_queries {
            let similar = self.memory.retrieve_similar(query, 5).await?;

            if similar.len() >= 3 {
                // Found recurring pattern
                let pattern = CodebasePattern {
                    pattern_id: format!("pattern_{}", uuid::Uuid::new_v4()),
                    pattern_type: self.classify_pattern_type(query),
                    description: format!("Detected pattern: {}", query),
                    occurrences: similar
                        .into_iter()
                        .map(|item| PatternOccurrence {
                            file_path: PathBuf::from("unknown"), // Would extract from context
                            line_range: None,
                            example: item.content,
                        })
                        .collect(),
                    confidence: 0.8,
                    applicable_contexts: vec![query.to_string()],
                    learned_at: Utc::now(),
                };

                new_patterns.push(pattern.clone());

                // Store pattern
                let mut patterns = self.learned_patterns.write().await;
                patterns.insert(pattern.pattern_id.clone(), pattern);
            }
        }

        if !new_patterns.is_empty() {
            // Record learning in story
            self.story_engine
                .add_plot_point(
                    self.codebase_story_id.clone(),
                    PlotType::Discovery {
                        insight: format!("Learned {} new codebase patterns", new_patterns.len()),
                    },
                    vec![],
                )
                .await?;
        }

        Ok(())
    }

    /// Classify pattern type from query
    fn classify_pattern_type(&self, query: &str) -> PatternType {
        if query.contains("error") {
            PatternType::ErrorHandlingPattern
        } else if query.contains("test") {
            PatternType::TestingPattern
        } else if query.contains("security") {
            PatternType::SecurityPattern
        } else if query.contains("performance") {
            PatternType::PerformancePattern
        } else if query.contains("doc") {
            PatternType::DocumentationPattern
        } else {
            PatternType::CodingConvention
        }
    }

    /// Generate maintenance tasks from story context
    async fn generate_maintenance_tasks(&self) -> Result<()> {
        debug!("Generating maintenance tasks from story context");

        // Extract tasks from story
        let story_tasks = self.story_engine.extract_all_tasks().await?;

        let mut tasks = self.active_tasks.write().await;

        for (story_id, task_mappings) in story_tasks {
            for task_mapping in task_mappings {
                if task_mapping.status != crate::story::TaskStatus::Completed && story_id == self.codebase_story_id {
                    // Convert story task to maintenance task
                    let task_type = self.classify_task_type(&task_mapping.description);

                    let maintenance_task = MaintenanceTask {
                        task_id: task_mapping.id.clone(),
                        task_type,
                        description: task_mapping.description.clone(),
                        priority: 0.5, // Default priority, could derive from story context
                        status: match task_mapping.status {
                            crate::story::TaskStatus::InProgress => TaskStatus::InProgress,
                            crate::story::TaskStatus::Completed => TaskStatus::Completed,
                            crate::story::TaskStatus::Blocked => TaskStatus::Pending,
                            _ => TaskStatus::Pending,
                        },
                        created_at: task_mapping.created_at,
                        story_segment_id: None, // MappedTask doesn't have segment_id
                    };

                    // Only add if not already present
                    if !tasks.iter().any(|t| t.task_id == maintenance_task.task_id) {
                        tasks.push(maintenance_task);
                    }
                }
            }
        }

        Ok(())
    }

    /// Classify task type from description
    fn classify_task_type(&self, description: &str) -> MaintenanceTaskType {
        let desc_lower = description.to_lowercase();

        if desc_lower.contains("review") && (desc_lower.contains("pr") || desc_lower.contains("pull request")) {
            MaintenanceTaskType::PrReview
        } else if desc_lower.contains("bug") || desc_lower.contains("fix") || desc_lower.contains("error") {
            MaintenanceTaskType::BugFix
        } else if desc_lower.contains("test") {
            MaintenanceTaskType::TestGeneration
        } else if desc_lower.contains("refactor") {
            MaintenanceTaskType::Refactoring
        } else if desc_lower.contains("doc") || desc_lower.contains("comment") {
            MaintenanceTaskType::DocumentationUpdate
        } else if desc_lower.contains("dependency") || desc_lower.contains("update") {
            MaintenanceTaskType::DependencyUpdate
        } else if desc_lower.contains("performance") || desc_lower.contains("optimize") {
            MaintenanceTaskType::PerformanceOptimization
        } else if desc_lower.contains("security") {
            MaintenanceTaskType::SecurityPatch
        } else if desc_lower.contains("implement") || desc_lower.contains("create") || desc_lower.contains("generate") || desc_lower.contains("add") {
            MaintenanceTaskType::CodeGeneration
        } else {
            MaintenanceTaskType::Refactoring
        }
    }

    /// Analyze codebase and suggest refactorings
    async fn analyze_and_suggest_refactorings(&self) -> Result<()> {
        if let Some(refactoring_system) = &self.refactoring_system {
            info!("🔧 Analyzing codebase for refactoring opportunities");

            // Analyze for refactoring opportunities
            let analysis = refactoring_system.analyze_for_refactoring().await?;

            info!(
                "Found {} refactoring suggestions, {} can be automated",
                analysis.suggestions.len(),
                analysis.suggestions.iter().filter(|s| s.automated).count()
            );

            // Process high-impact automated refactorings
            let high_impact_automated: Vec<_> = analysis.suggestions.iter()
                .filter(|s| s.automated && s.risk_level <= self.config.max_risk_level)
                .filter(|s| {
                    let impact_score = s.impact.complexity_reduction * 0.3 +
                                     s.impact.maintainability_improvement * 0.3 +
                                     s.impact.readability_improvement * 0.2 +
                                     s.impact.performance_impact * 0.2;
                    impact_score > 0.3
                })
                .collect();

            if !high_impact_automated.is_empty() {
                info!("🚀 Applying {} high-impact automated refactorings", high_impact_automated.len());

                for suggestion in &high_impact_automated {
                    match refactoring_system.apply_refactoring(&suggestion.suggestion_id).await {
                        Ok(result) => {
                            if result.success {
                                info!("✅ Applied refactoring: {}", suggestion.description);
                            } else {
                                warn!("❌ Failed to apply refactoring: {}", suggestion.description);
                            }
                        }
                        Err(e) => {
                            error!("Error applying refactoring: {}", e);
                        }
                    }
                }
            }

            // Create tasks for manual refactorings
            let manual_refactorings: Vec<_> = analysis.suggestions.iter()
                .filter(|s| !s.automated || s.risk_level > self.config.max_risk_level)
                .collect();

            if !manual_refactorings.is_empty() {
                let mut tasks = self.active_tasks.write().await;

                for suggestion in manual_refactorings.iter().take(5) {
                    let task = MaintenanceTask {
                        task_id: format!("refactor_{}", suggestion.suggestion_id),
                        task_type: MaintenanceTaskType::Refactoring,
                        description: format!(
                            "{} (Refactoring type: {:?}, Risk: {:?}, Impact: complexity -{:.1}%, maintainability +{:.1}%)",
                            suggestion.description,
                            suggestion.refactoring_type,
                            suggestion.risk_level,
                            suggestion.impact.complexity_reduction * 100.0,
                            suggestion.impact.maintainability_improvement * 100.0
                        ),
                        priority: suggestion.impact.maintainability_improvement,
                        status: TaskStatus::Pending,
                        created_at: Utc::now(),
                        story_segment_id: None,
                    };

                    tasks.push(task);
                }

                info!(
                    "📝 Created {} refactoring tasks for manual review",
                    manual_refactorings.len().min(5)
                );
            }

            // Record in story
            let plot_type = PlotType::Analysis {
                subject: "code refactoring".to_string(),
                findings: vec![
                    format!("{} refactoring opportunities identified", analysis.suggestions.len()),
                    format!("{} automated refactorings applied", high_impact_automated.len()),
                    format!("{} manual refactorings queued", manual_refactorings.len().min(5)),
                ],
            };
            let context_tokens = vec![];

            self.story_engine
                .add_plot_point(
                    self.codebase_story_id.clone(),
                    plot_type,
                    context_tokens,
                )
                .await?;
        }

        Ok(())
    }

    /// Monitor and optimize performance
    async fn monitor_and_optimize_performance(&self) -> Result<()> {
        if let Some(performance_system) = &self.performance_system {
            info!("🚀 Monitoring and optimizing performance");

            // Analyze performance
            let analysis = performance_system.analyze_performance().await?;

            info!(
                "Performance score: {:.1}%, {} bottlenecks found, {} optimizations available",
                analysis.overall_performance_score * 100.0,
                analysis.bottlenecks.len(),
                analysis.optimization_suggestions.len()
            );

            // Create tasks for critical bottlenecks
            let mut tasks = self.active_tasks.write().await;

            for bottleneck in analysis.bottlenecks.iter()
                .filter(|b| matches!(b.severity, crate::cognitive::story_driven_performance::BottleneckSeverity::Critical))
            {
                let task = MaintenanceTask {
                    task_id: format!("perf_{}", bottleneck.bottleneck_id),
                    task_type: MaintenanceTaskType::PerformanceOptimization,
                    description: format!(
                        "{} (Impact: execution time -{:.1}%, memory -{:.1}%)",
                        bottleneck.description,
                        bottleneck.impact.execution_time_impact,
                        bottleneck.impact.memory_impact
                    ),
                    priority: 0.9,
                    status: TaskStatus::Pending,
                    created_at: Utc::now(),
                    story_segment_id: None,
                };

                if !tasks.iter().any(|t| t.task_id == task.task_id) {
                    tasks.push(task);
                }
            }

            // Apply automatic optimizations for low-risk suggestions
            let auto_optimizations: Vec<_> = analysis.optimization_suggestions.iter()
                .filter(|o| o.risk_level <= self.config.max_risk_level &&
                           o.expected_improvement.execution_time_impact +
                           o.expected_improvement.memory_impact > 20.0)
                .collect();

            if !auto_optimizations.is_empty() {
                info!("🔧 Applying {} automatic performance optimizations", auto_optimizations.len());

                for optimization in auto_optimizations.iter().take(3) {
                    match performance_system.apply_optimization(&optimization.optimization_id).await {
                        Ok(result) => {
                            if result.success {
                                info!(
                                    "✅ Applied optimization: {} (improved by {:.1}%)",
                                    optimization.description,
                                    result.actual_improvement.execution_time_impact
                                );
                            }
                        }
                        Err(e) => {
                            warn!("Failed to apply optimization: {}", e);
                        }
                    }
                }
            }

            // Create tasks for manual optimizations
            for suggestion in analysis.optimization_suggestions.iter()
                .filter(|s| s.risk_level > self.config.max_risk_level)
                .take(5)
            {
                let task = MaintenanceTask {
                    task_id: format!("opt_{}", suggestion.optimization_id),
                    task_type: MaintenanceTaskType::PerformanceOptimization,
                    description: format!(
                        "{} (Type: {:?}, Expected improvement: {:.1}%)",
                        suggestion.description,
                        suggestion.optimization_type,
                        suggestion.expected_improvement.execution_time_impact
                    ),
                    priority: suggestion.expected_improvement.execution_time_impact / 100.0,
                    status: TaskStatus::Pending,
                    created_at: Utc::now(),
                    story_segment_id: None,
                };

                if !tasks.iter().any(|t| t.task_id == task.task_id) {
                    tasks.push(task);
                }
            }

            // Record in story
            let plot_type = PlotType::Analysis {
                subject: "performance".to_string(),
                findings: vec![
                    format!("Performance score: {:.1}%", analysis.overall_performance_score * 100.0),
                    format!("{} bottlenecks identified", analysis.bottlenecks.len()),
                    format!("{} optimizations applied", auto_optimizations.len()),
                ],
            };
            let context_tokens = vec![];

            self.story_engine
                .add_plot_point(
                    self.codebase_story_id.clone(),
                    plot_type,
                    context_tokens,
                )
                .await?;
        }

        Ok(())
    }

    /// Scan for and fix security vulnerabilities
    async fn scan_and_fix_vulnerabilities(&self) -> Result<()> {
        if let Some(security_system) = &self.security_system {
            info!("🔒 Scanning for security vulnerabilities");

            // Scan for vulnerabilities
            let scan_result = security_system.scan_for_vulnerabilities().await?;

            info!(
                "Security scan complete: {} vulnerabilities found ({} critical, {} high)",
                scan_result.vulnerabilities.len(),
                scan_result.vulnerabilities.iter().filter(|v| matches!(v.severity, crate::cognitive::VulnerabilitySeverity::Critical)).count(),
                scan_result.vulnerabilities.iter().filter(|v| matches!(v.severity, crate::cognitive::VulnerabilitySeverity::High)).count()
            );

            // Create tasks for critical vulnerabilities
            let mut tasks = self.active_tasks.write().await;

            for vulnerability in scan_result.vulnerabilities.iter()
                .filter(|v| matches!(v.severity, crate::cognitive::VulnerabilitySeverity::Critical))
            {
                let task = MaintenanceTask {
                    task_id: format!("sec_{}", vulnerability.vulnerability_id),
                    task_type: MaintenanceTaskType::SecurityPatch,
                    description: format!(
                        "{} (Type: {:?}, OWASP: {})",
                        vulnerability.description,
                        vulnerability.vulnerability_type,
                        vulnerability.owasp_category.as_ref().unwrap_or(&"N/A".to_string())
                    ),
                    priority: 1.0, // Maximum priority for security
                    status: TaskStatus::Pending,
                    created_at: Utc::now(),
                    story_segment_id: None,
                };

                if !tasks.iter().any(|t| t.task_id == task.task_id) {
                    tasks.push(task);
                }
            }

            // Apply automatic fixes for low-risk vulnerabilities
            let auto_fixable: Vec<_> = scan_result.vulnerabilities.iter()
                .filter(|v| v.fix_available &&
                           v.severity != crate::cognitive::story_driven_dependencies::VulnerabilitySeverity::Critical)
                .collect();

            if !auto_fixable.is_empty() {
                info!("🔧 Applying {} automatic security fixes", auto_fixable.len());

                for vulnerability in auto_fixable.iter().take(3) {
                    match security_system.fix_vulnerability(&vulnerability.vulnerability_id).await {
                        Ok(fix_result) => {
                            if fix_result.success {
                                info!(
                                    "✅ Fixed vulnerability: {} (fixed {} vulnerabilities)",
                                    vulnerability.vulnerability_type,
                                    fix_result.vulnerabilities_fixed.len()
                                );
                            }
                        }
                        Err(e) => {
                            warn!("Failed to fix vulnerability: {}", e);
                        }
                    }
                }
            }

            // Create tasks for manual security reviews
            for vulnerability in scan_result.vulnerabilities.iter()
                .filter(|v| !v.fix_available ||
                           v.severity == crate::cognitive::story_driven_dependencies::VulnerabilitySeverity::Critical)
                .take(5)
            {
                let task = MaintenanceTask {
                    task_id: format!("sec_review_{}", vulnerability.vulnerability_id),
                    task_type: MaintenanceTaskType::SecurityPatch,
                    description: format!("Review and fix: {} (Manual review required: {})",
                        vulnerability.description,
                        vulnerability.fix_suggestion.as_ref()
                            .map(|f| f.description.clone())
                            .unwrap_or_else(|| "No specific fix available".to_string())),
                    priority: match vulnerability.severity {
                        crate::cognitive::story_driven_dependencies::VulnerabilitySeverity::Critical => 0.95,
                        crate::cognitive::story_driven_dependencies::VulnerabilitySeverity::High => 0.85,
                        crate::cognitive::story_driven_dependencies::VulnerabilitySeverity::Medium => 0.7,
                        crate::cognitive::story_driven_dependencies::VulnerabilitySeverity::Low => 0.5,
                    },
                    status: TaskStatus::Pending,
                    created_at: Utc::now(),
                    story_segment_id: None,
                };

                if !tasks.iter().any(|t| t.task_id == task.task_id) {
                    tasks.push(task);
                }
            }

            // Record in story
            let plot_type = PlotType::Analysis {
                subject: "security".to_string(),
                findings: vec![
                    format!("Found {} vulnerabilities", scan_result.vulnerabilities.len()),
                    format!("Applied {} automatic fixes", auto_fixable.len()),
                    format!("Security score: {:.1}%", self.calculate_security_score(&scan_result.risk_summary)),
                ],
            };
            let context_tokens = vec![];

            self.story_engine
                .add_plot_point(
                    self.codebase_story_id.clone(),
                    plot_type,
                    context_tokens,
                )
                .await?;
        }

        Ok(())
    }

    /// Execute high-priority tasks
    async fn execute_tasks(&self) -> Result<()> {
        let mut tasks = self.active_tasks.write().await;

        // Sort by priority (handle NaN values gracefully)
        tasks.sort_by(|a, b| {
            b.priority.partial_cmp(&a.priority)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Execute top tasks
        let tasks_to_execute: Vec<_> = tasks
            .iter_mut()
            .filter(|t| t.status == TaskStatus::Pending)
            .take(3)
            .collect();

        for task in tasks_to_execute {
            task.status = TaskStatus::InProgress;

            let result = match &task.task_type {
                MaintenanceTaskType::BugFix => self.execute_bug_fix(task).await,
                MaintenanceTaskType::TestGeneration => self.execute_test_generation(task).await,
                MaintenanceTaskType::DocumentationUpdate => self.execute_documentation(task).await,
                MaintenanceTaskType::Refactoring => self.execute_refactoring(task).await,
                MaintenanceTaskType::CodeGeneration => self.execute_code_generation(task).await,
                MaintenanceTaskType::PrReview => self.execute_pr_review(task).await,
                MaintenanceTaskType::SecurityPatch => self.execute_security_fix(task).await,
                _ => {
                    warn!("Task type {:?} not yet implemented", task.task_type);
                    Ok(())
                }
            };

            match result {
                Ok(()) => {
                    task.status = TaskStatus::Completed;

                    // Record completion in story
                    let _ = self.story_engine
                        .add_plot_point(
                            self.codebase_story_id.clone(),
                            PlotType::Task {
                                description: task.description.clone(),
                                completed: true,
                            },
                            vec![],
                        )
                        .await;
                }
                Err(e) => {
                    error!("Task execution failed: {}", e);
                    task.status = TaskStatus::Failed;
                }
            }
        }

        // Clean up completed tasks
        tasks.retain(|t| t.status != TaskStatus::Completed);

        Ok(())
    }

    /// Execute a bug fix task
    async fn execute_bug_fix(&self, task: &MaintenanceTask) -> Result<()> {
        info!("🐛 Executing autonomous bug fix: {}", task.description);

        // This would integrate with self-modification pipeline
        // For now, just log the action

        self.memory
            .store(
                format!("Autonomous bug fix executed: {}", task.description),
                vec![],
                MemoryMetadata {
                    source: "story_driven_autonomy".to_string(),
                    tags: vec!["bug_fix".to_string(), "autonomous".to_string()],
                    importance: 0.8,
                    associations: vec![],
                    context: Some("autonomous bug fixing".to_string()),
                    created_at: chrono::Utc::now(),
                    accessed_count: 0,
                    last_accessed: None,
                    version: 1,
                    category: "story".to_string(),
                    timestamp: chrono::Utc::now(),
                    expiration: None,
                },
            )
            .await?;

        Ok(())
    }

    /// Execute test generation task
    async fn execute_test_generation(&self, task: &MaintenanceTask) -> Result<()> {
        info!("🧪 Executing autonomous test generation: {}", task.description);

        // Extract file path from task description (simplified)
        // In real implementation, would parse more carefully

        self.memory
            .store(
                format!("Autonomous test generation executed: {}", task.description),
                vec![],
                MemoryMetadata {
                    source: "story_driven_autonomy".to_string(),
                    tags: vec!["test_generation".to_string(), "autonomous".to_string()],
                    importance: 0.7,
                    associations: vec![],
                    context: Some("autonomous testing".to_string()),
                    created_at: chrono::Utc::now(),
                    accessed_count: 0,
                    last_accessed: None,
                    version: 1,
                    category: "story".to_string(),
                    timestamp: chrono::Utc::now(),
                    expiration: None,
                },
            )
            .await?;

        Ok(())
    }

    /// Execute documentation task
    async fn execute_documentation(&self, task: &MaintenanceTask) -> Result<()> {
        info!("📝 Executing autonomous documentation: {}", task.description);

        self.memory
            .store(
                format!("Autonomous documentation executed: {}", task.description),
                vec![],
                MemoryMetadata {
                    source: "story_driven_autonomy".to_string(),
                    tags: vec!["documentation".to_string(), "autonomous".to_string()],
                    importance: 0.6,
                    associations: vec![],
                    context: Some("autonomous documentation".to_string()),
                    created_at: chrono::Utc::now(),
                    accessed_count: 0,
                    last_accessed: None,
                    version: 1,
                    category: "story".to_string(),
                    timestamp: chrono::Utc::now(),
                    expiration: None,
                },
            )
            .await?;

        Ok(())
    }

    /// Execute refactoring task
    async fn execute_refactoring(&self, task: &MaintenanceTask) -> Result<()> {
        info!("🔧 Executing autonomous refactoring: {}", task.description);

        // Check risk level
        if self.config.max_risk_level < RiskLevel::Medium {
            warn!("Refactoring skipped due to risk level constraints");
            return Ok(());
        }

        self.memory
            .store(
                format!("Autonomous refactoring executed: {}", task.description),
                vec![],
                MemoryMetadata {
                    source: "story_driven_autonomy".to_string(),
                    tags: vec!["refactoring".to_string(), "autonomous".to_string()],
                    importance: 0.7,
                    associations: vec![],
                    context: Some("autonomous refactoring".to_string()),
                    created_at: chrono::Utc::now(),
                    accessed_count: 0,
                    last_accessed: None,
                    version: 1,
                    category: "story".to_string(),
                    timestamp: chrono::Utc::now(),
                    expiration: None,
                },
            )
            .await?;

        Ok(())
    }

    /// Execute security fix task
    async fn execute_security_fix(&self, task: &MaintenanceTask) -> Result<()> {
        info!("🔒 Executing autonomous security fix: {}", task.description);

        // Check if we have security system
        if let Some(security_system) = &self.security_system {
            // Extract vulnerability ID from task_id (format: "sec_{id}" or "sec_review_{id}")
            let vuln_id = if task.task_id.starts_with("sec_review_") {
                task.task_id.strip_prefix("sec_review_").unwrap_or(&task.task_id)
            } else {
                task.task_id.strip_prefix("sec_").unwrap_or(&task.task_id)
            };

            // For review tasks, just log it
            if task.task_id.starts_with("sec_review_") {
                warn!("Security vulnerability {} requires manual review", vuln_id);
            } else {
                // Attempt automatic fix
                match security_system.fix_vulnerability(vuln_id).await {
                    Ok(fix_result) => {
                        if fix_result.success {
                            info!("✅ Fixed security vulnerability: fixed {} vulnerabilities", fix_result.vulnerabilities_fixed.len());
                        } else {
                            warn!("Could not fix vulnerability: validation_passed={}", fix_result.validation_passed);
                        }
                    }
                    Err(e) => {
                        error!("Error fixing security vulnerability: {}", e);
                        return Err(e);
                    }
                }
            }
        }

        self.memory
            .store(
                format!("Autonomous security fix executed: {}", task.description),
                vec![],
                MemoryMetadata {
                    source: "story_driven_autonomy".to_string(),
                    tags: vec!["security".to_string(), "vulnerability".to_string(), "autonomous".to_string()],
                    importance: 0.95,
                    associations: vec![],
                    context: Some("autonomous security patching".to_string()),
                    created_at: chrono::Utc::now(),
                    accessed_count: 0,
                    last_accessed: None,
                    version: 1,
                    category: "story".to_string(),
                    timestamp: chrono::Utc::now(),
                    expiration: None,
                },
            )
            .await?;

        Ok(())
    }

    /// Review pending PRs with story context
    async fn review_pending_prs(&self) -> Result<()> {
        if let Some(_pr_reviewer) = &self.pr_reviewer {
            info!("🔍 Reviewing pending PRs with story context");

            // Get list of open PRs (simplified - would use GitHub API)
            let open_prs = self.get_open_prs().await?;

            for pr_number in open_prs {
                // Check if we already have a task for this PR
                let tasks = self.active_tasks.read().await;
                let has_task = tasks.iter().any(|t| {
                    matches!(t.task_type, MaintenanceTaskType::PrReview) &&
                    t.description.contains(&pr_number.to_string())
                });
                drop(tasks);

                if !has_task {
                    // Create review task
                    let mut tasks = self.active_tasks.write().await;
                    tasks.push(MaintenanceTask {
                        task_id: format!("pr_review_{}", pr_number),
                        task_type: MaintenanceTaskType::PrReview,
                        description: format!("Review PR #{}", pr_number),
                        priority: 0.8,
                        status: TaskStatus::Pending,
                        created_at: Utc::now(),
                        story_segment_id: None,
                    });
                }
            }
        }

        Ok(())
    }

    /// Get list of open PRs (placeholder - would use GitHub API)
    async fn get_open_prs(&self) -> Result<Vec<u32>> {
        // In real implementation, would query GitHub API
        Ok(vec![])
    }

    /// Detect and fix bugs autonomously
    async fn detect_and_fix_bugs(&self) -> Result<()> {
        if let Some(bug_detector) = &self.bug_detector {
            info!("🐛 Running autonomous bug detection and fixing");

            // Scan for bugs
            let detected_bugs = bug_detector.scan_codebase().await?;

            if !detected_bugs.is_empty() {
                info!("Found {} bugs to process", detected_bugs.len());

                // Create tasks for high-priority bugs
                let mut tasks = self.active_tasks.write().await;

                for bug in detected_bugs {
                    // Only create tasks for high-severity bugs
                    if matches!(bug.severity, crate::cognitive::story_driven_bug_detection::BugSeverity::Critical |
                                            crate::cognitive::story_driven_bug_detection::BugSeverity::High) {

                        let task = MaintenanceTask {
                            task_id: bug.bug_id.clone(),
                            task_type: MaintenanceTaskType::BugFix,
                            description: format!("Fix bug: {}", bug.description),
                            priority: match bug.severity {
                                crate::cognitive::story_driven_bug_detection::BugSeverity::Critical => 1.0,
                                crate::cognitive::story_driven_bug_detection::BugSeverity::High => 0.9,
                                _ => 0.7,
                            },
                            status: TaskStatus::Pending,
                            created_at: Utc::now(),
                            story_segment_id: None,
                        };

                        // Only add if not already tracked
                        if !tasks.iter().any(|t| t.task_id == task.task_id) {
                            tasks.push(task);
                        }

                        // For critical bugs with high confidence, attempt immediate fix
                        if matches!(bug.severity, crate::cognitive::story_driven_bug_detection::BugSeverity::Critical) {

                            drop(tasks); // Release lock

                            match bug_detector.fix_bug(&bug.bug_id).await {
                                Ok(BugFixResult::Fixed { pr_number, fix_description: _ }) => {
                                    info!("✅ Automatically fixed bug {}: PR #{}", bug.bug_id, pr_number);

                                    // Record fix in story
                                    self.story_engine
                                        .add_plot_point(
                                            self.codebase_story_id.clone(),
                                            PlotType::Task {
                                                description: format!("Fixed critical bug: {}", bug.description),
                                                completed: true,
                                            },
                                            vec![],
                                        )
                                        .await?;
                                }
                                Ok(BugFixResult::FixProposed { .. }) => {
                                    info!("📝 Fix proposed for bug {}, requires review", bug.bug_id);
                                }
                                Ok(BugFixResult::CannotFix(reason)) => {
                                    warn!("Cannot fix bug {}: {}", bug.bug_id, reason);
                                }
                                Ok(BugFixResult::ManualFixRequired(reason)) => {
                                    info!("Manual fix required for bug {}: {}", bug.bug_id, reason);
                                }
                                Ok(BugFixResult::FixFailed(reason)) => {
                                    warn!("Fix failed for bug {}: {}", bug.bug_id, reason);
                                }
                                Err(e) => {
                                    error!("Failed to fix bug {}: {}", bug.bug_id, e);
                                }
                            }

                            tasks = self.active_tasks.write().await; // Re-acquire lock
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Generate and maintain tests autonomously
    async fn generate_and_maintain_tests(&self) -> Result<()> {
        if let Some(test_system) = &self.test_system {
            info!("🧪 Running autonomous test generation and maintenance");

            // Get current test coverage
            let coverage_info = test_system.get_coverage_info().await?;
            info!("Current test coverage: {:.1}%", coverage_info.overall_coverage * 100.0);

            // Identify files needing tests
            let files_needing_tests = test_system.identify_files_needing_tests().await?;

            if !files_needing_tests.is_empty() {
                info!("Found {} files needing tests", files_needing_tests.len());

                // Create tasks for test generation
                let mut tasks = self.active_tasks.write().await;

                for file_path in files_needing_tests.iter().take(5) { // Limit to 5 files per cycle
                    let task = MaintenanceTask {
                        task_id: format!("test_gen_{}", uuid::Uuid::new_v4()),
                        task_type: MaintenanceTaskType::TestGeneration,
                        description: format!("Generate tests for {}", file_path.display()),
                        priority: 0.7,
                        status: TaskStatus::Pending,
                        created_at: Utc::now(),
                        story_segment_id: None,
                    };

                    if !tasks.iter().any(|t| t.task_id == task.task_id) {
                        tasks.push(task);
                    }
                }
            }

            // Execute existing tests to maintain quality
            info!("🏃 Running existing test suites");
            let test_results = test_system.execute_all_tests().await?;

            if test_results.total_tests > 0 {
                let pass_rate = test_results.passed_tests as f32 / test_results.total_tests as f32;
                info!(
                    "Test execution complete: {} passed, {} failed ({:.1}% pass rate)",
                    test_results.passed_tests,
                    test_results.failed_tests,
                    pass_rate * 100.0
                );

                // Fix failing tests if confidence is high
                if test_results.failed_tests > 0 && self.config.enable_testing {
                    info!("🔧 Attempting to fix failing tests");

                    for failing_test in test_results.failing_tests {
                        match test_system.fix_failing_test(&failing_test).await {
                            Ok(fix_result) => {
                                if fix_result.fixed {
                                    info!("✅ Fixed test: {}", failing_test.test_name);

                                    // Apply the fix if change is available
                                    if let Some(change) = fix_result.change {
                                        if let Err(e) = self.self_modify
                                            .propose_change(change)
                                            .await
                                    {
                                        error!("Failed to apply test fix: {}", e);
                                        }
                                    }
                                } else {
                                    warn!(
                                        "Could not automatically fix test {}: {}",
                                        failing_test.test_name,
                                        fix_result.reason.unwrap_or_default()
                                    );
                                }
                            }
                            Err(e) => {
                                error!("Error fixing test {}: {}", failing_test.test_name, e);
                            }
                        }
                    }
                }
            }

            // Record testing activity in story
            self.story_engine
                .add_plot_point(
                    self.codebase_story_id.clone(),
                    PlotType::Progress {
                        milestone: "Autonomous testing cycle completed".to_string(),
                        percentage: coverage_info.overall_coverage * 100.0,
                    },
                    vec![],
                )
                .await?;
        }

        Ok(())
    }

    /// Manage dependencies intelligently
    async fn manage_dependencies(&self) -> Result<()> {
        if let Some(dep_system) = &self.dependency_system {
            info!("📦 Running intelligent dependency management");

            // Analyze dependencies
            let analysis = dep_system.analyze_dependencies().await?;

            info!(
                "Dependency analysis: {} total, {} outdated, {} security issues",
                analysis.total_dependencies,
                analysis.outdated_dependencies.len(),
                analysis.security_vulnerabilities.len()
            );

            // Handle security vulnerabilities immediately
            if !analysis.security_vulnerabilities.is_empty() {
                warn!(
                    "⚠️  Found {} security vulnerabilities in dependencies",
                    analysis.security_vulnerabilities.len()
                );

                // Create high-priority tasks for critical vulnerabilities
                let mut tasks = self.active_tasks.write().await;

                for vuln in analysis.security_vulnerabilities.iter()
                    .filter(|v| v.severity == crate::cognitive::story_driven_dependencies::VulnerabilitySeverity::Critical)
                {
                    let task = MaintenanceTask {
                        task_id: format!("sec_update_{}", uuid::Uuid::new_v4()),
                        task_type: MaintenanceTaskType::SecurityUpdate,
                        description: format!(
                            "Update {} to fix {} vulnerability",
                            vuln.dependency,
                            vuln.cve.as_ref().unwrap_or(&"security".to_string())
                        ),
                        priority: 1.0, // Highest priority
                        status: TaskStatus::Pending,
                        created_at: Utc::now(),
                        story_segment_id: None,
                    };

                    if !tasks.iter().any(|t| t.task_id == task.task_id) {
                        tasks.push(task);
                    }
                }
            }

            // Update dependencies based on strategy
            let update_results = dep_system.update_dependencies().await?;

            if !update_results.is_empty() {
                info!("✅ Updated {} dependencies", update_results.len());

                // Log successful updates
                for update in &update_results {
                    if update.success {
                        info!(
                            "Updated {} from {} to {}",
                            update.dependency, update.old_version, update.new_version
                        );
                    }
                }
            }

            // Handle unused dependencies
            if !analysis.unused_dependencies.is_empty() {
                info!("🧹 Found {} unused dependencies", analysis.unused_dependencies.len());

                // Create tasks for review
                let mut tasks = self.active_tasks.write().await;

                let task = MaintenanceTask {
                    task_id: format!("dep_cleanup_{}", uuid::Uuid::new_v4()),
                    task_type: MaintenanceTaskType::Refactoring,
                    description: format!(
                        "Remove {} unused dependencies: {}",
                        analysis.unused_dependencies.len(),
                        analysis.unused_dependencies.join(", ")
                    ),
                    priority: 0.4,
                    status: TaskStatus::Pending,
                    created_at: Utc::now(),
                    story_segment_id: None,
                };

                if !tasks.iter().any(|t| t.description.contains("unused dependencies")) {
                    tasks.push(task);
                }
            }

            // Check for optimization opportunities
            if !analysis.optimization_opportunities.is_empty() {
                for opportunity in &analysis.optimization_opportunities {
                    info!("💡 Optimization: {}", opportunity.recommendation);
                }
            }

            // Generate dependency report
            let report = dep_system.generate_dependency_report().await?;

            // Store in memory
            self.memory
                .store(
                    "dependency_report".to_string(),
                    vec![serde_json::to_string(&report)?],
                    MemoryMetadata {
                        source: "story_driven_autonomy".to_string(),
                        tags: vec!["dependencies".to_string(), "maintenance".to_string()],
                        importance: 0.7,
                        associations: vec![MemoryId::from_string("security".to_string())],
                        context: Some("Dependency analysis and maintenance report".to_string()),
                        created_at: Utc::now(),
                        accessed_count: 0,
                        last_accessed: None,
                        version: 1,
                    category: "story".to_string(),
                        timestamp: Utc::now(),
                        expiration: None,
                    },
                )
                .await?;

            // Record in story
            self.story_engine
                .add_plot_point(
                    self.codebase_story_id.clone(),
                    PlotType::Progress {
                        milestone: "Dependency management completed".to_string(),
                        percentage: (analysis.total_dependencies - analysis.outdated_dependencies.len()) as f32
                            / analysis.total_dependencies as f32 * 100.0,
                    },
                    vec![],
                )
                .await?;
        }

        Ok(())
    }

    /// Generate and update documentation
    async fn generate_and_update_documentation(&self) -> Result<()> {
        if let Some(doc_system) = &self.documentation_system {
            info!("📚 Running autonomous documentation generation");

            // Analyze current documentation coverage
            let analysis = doc_system.analyze_documentation_coverage().await?;

            info!(
                "Documentation coverage: {:.1}% ({}/{} files documented)",
                analysis.documentation_coverage * 100.0,
                analysis.documented_files,
                analysis.total_files
            );

            // Generate missing documentation
            if !analysis.missing_docs.is_empty() {
                info!("Generating documentation for {} items", analysis.missing_docs.len());

                let generated = doc_system.generate_missing_documentation().await?;

                if !generated.is_empty() {
                    info!("✅ Generated {} documentation items", generated.len());

                    // Create tasks for complex documentation needs
                    let mut tasks = self.active_tasks.write().await;

                    for doc in generated.iter().filter(|d| d.metadata.complexity > 8.0) {
                        let task = MaintenanceTask {
                            task_id: format!("doc_review_{}", uuid::Uuid::new_v4()),
                            task_type: MaintenanceTaskType::DocumentationUpdate,
                            description: format!("Review generated docs for {}", doc.metadata.title),
                            priority: 0.5,
                            status: TaskStatus::Pending,
                            created_at: Utc::now(),
                            story_segment_id: None,
                        };

                        if !tasks.iter().any(|t| t.task_id == task.task_id) {
                            tasks.push(task);
                        }
                    }
                }
            }

            // Update outdated documentation
            if !analysis.outdated_docs.is_empty() {
                info!("Updating {} outdated documentation items", analysis.outdated_docs.len());

                let updated = doc_system.update_outdated_documentation().await?;

                if !updated.is_empty() {
                    info!("✅ Updated {} documentation items", updated.len());
                }
            }

            // Generate README if needed
            let readme_path = self.config.repo_path.join("README.md");
            if !readme_path.exists() && self.config.enable_documentation {
                info!("📖 Generating README.md");

                match doc_system.generate_readme().await {
                    Ok(readme) => {
                        info!("✅ Generated README with {} sections",
                            readme.content.matches("##").count()
                        );
                    }
                    Err(e) => {
                        warn!("Failed to generate README: {}", e);
                    }
                }
            }

            // Generate API documentation
            if self.config.enable_documentation {
                match doc_system.generate_api_documentation().await {
                    Ok(api_docs) => {
                        if !api_docs.is_empty() {
                            info!("✅ Generated API documentation for {} modules", api_docs.len());
                        }
                    }
                    Err(e) => {
                        warn!("Failed to generate API docs: {}", e);
                    }
                }
            }

            // Record documentation activity in story
            self.story_engine
                .add_plot_point(
                    self.codebase_story_id.clone(),
                    PlotType::Progress {
                        milestone: "Documentation generation completed".to_string(),
                        percentage: analysis.documentation_coverage * 100.0,
                    },
                    vec![],
                )
                .await?;
        }

        Ok(())
    }

    /// Update story with maintenance progress
    async fn update_story_progress(&self) -> Result<()> {
        let tasks = self.active_tasks.read().await;

        let pending_count = tasks.iter().filter(|t| t.status == TaskStatus::Pending).count();
        let in_progress_count = tasks.iter().filter(|t| t.status == TaskStatus::InProgress).count();

        if pending_count > 0 || in_progress_count > 0 {
            self.story_engine
                .add_plot_point(
                    self.codebase_story_id.clone(),
                    PlotType::Discovery {
                        insight: format!(
                            "Maintenance status: {} pending, {} in progress",
                            pending_count, in_progress_count
                        ),
                    },
                    vec![],
                )
                .await?;
        }

        Ok(())
    }

    /// Generate code from story context
    async fn generate_code_from_story(&self) -> Result<()> {
        info!("🎨 Generating code from story context");

        // Generate code for the current codebase story
        match self.code_generator
            .generate_from_story_context(&self.codebase_story_id, None)
            .await
        {
            Ok(generated_artifacts) => {
                let artifacts_count = generated_artifacts.len();
                info!(
                    "Generated {} code artifacts from story context",
                    artifacts_count
                );

                // Create tasks for applying generated code
                let mut tasks = self.active_tasks.write().await;

                for artifact in generated_artifacts {
                    // Only create task if within risk threshold
                    if artifact.risk_level <= self.config.max_risk_level {
                        let task = MaintenanceTask {
                            task_id: format!("apply_generated_{}", uuid::Uuid::new_v4()),
                            task_type: MaintenanceTaskType::CodeGeneration,
                            description: format!(
                                "Apply generated {}: {}",
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
                            priority: 0.8, // High priority for generated code
                            status: TaskStatus::Pending,
                            created_at: Utc::now(),
                            story_segment_id: None,
                        };

                        tasks.push(task);

                        // Store generated code in memory for later application
                        self.memory
                            .store(
                                format!("generated_code_{}", artifact.file_path.display()),
                                vec![serde_json::to_string(&artifact)?],
                                MemoryMetadata {
                                    source: "story_driven_code_gen".to_string(),
                                    tags: vec!["generated_code".to_string(), "pending_application".to_string()],
                                    importance: 0.8,
                                    associations: vec![],
                                    context: Some("code generation from story".to_string()),
                                    created_at: chrono::Utc::now(),
                                    accessed_count: 0,
                                    last_accessed: None,
                                    version: 1,
                    category: "story".to_string(),
                                    timestamp: chrono::Utc::now(),
                                    expiration: None,
                                },
                            )
                            .await?;
                    } else {
                        warn!(
                            "Generated code exceeds risk threshold: {} ({:?})",
                            artifact.description, artifact.risk_level
                        );
                    }
                }

                // Record generation success in story
                self.story_engine
                    .add_plot_point(
                        self.codebase_story_id.clone(),
                        PlotType::Transformation {
                            before: "Story context".to_string(),
                            after: format!("Generated {} code artifacts", artifacts_count),
                        },
                        vec![],
                    )
                    .await?;
            }
            Err(e) => {
                warn!("Code generation failed: {}", e);

                // Record failure in story
                self.story_engine
                    .add_plot_point(
                        self.codebase_story_id.clone(),
                        PlotType::Issue {
                            error: format!("Code generation failed: {}", e),
                            resolved: false,
                        },
                        vec![],
                    )
                    .await?;
            }
        }

        Ok(())
    }

    /// Execute code generation task
    async fn execute_code_generation(&self, task: &MaintenanceTask) -> Result<()> {
        info!("💻 Executing code generation task: {}", task.description);

        // Extract artifact from task description or memory
        let artifact_key = format!("generated_code_{}", task.task_id);

        // Retrieve generated code from memory
        let memories = self.memory.retrieve_similar(&artifact_key, 1).await?;

        if let Some(memory) = memories.first() {
            // Parse generated artifact
            if let Ok(artifact) = serde_json::from_str::<GeneratedCode>(&memory.content) {
                // Create code change
                let code_change = CodeChange {
                    file_path: artifact.file_path.clone(),
                    change_type: match artifact.artifact_type {
                        GeneratedArtifactType::Function => ChangeType::Feature,
                        GeneratedArtifactType::Test => ChangeType::Test,
                        GeneratedArtifactType::Documentation => ChangeType::Documentation,
                        GeneratedArtifactType::Refactoring => ChangeType::Refactor,
                        _ => ChangeType::Enhancement,
                    },
                    description: artifact.description.clone(),
                    reasoning: "Story-driven autonomous code generation".to_string(),
                    old_content: None,
                    new_content: artifact.content.clone(),
                    line_range: None,
                    risk_level: artifact.risk_level,
                    attribution: None,
                };

                // Apply through self-modification pipeline
                match self.self_modify.propose_change(code_change).await {
                    Ok(pr) => {
                        info!("Successfully created PR #{} for generated code", pr.number);

                        // Learn from success
                        self.code_generator
                            .learn_from_outcome(&artifact, true, Some("PR created successfully".to_string()))
                            .await?;
                    }
                    Err(e) => {
                        error!("Failed to apply generated code: {}", e);

                        // Learn from failure
                        self.code_generator
                            .learn_from_outcome(&artifact, false, Some(e.to_string()))
                            .await?;
                    }
                }
            }
        } else {
            warn!("No generated code found for task: {}", task.description);
        }

        Ok(())
    }

    /// Execute PR review task
    async fn execute_pr_review(&self, task: &MaintenanceTask) -> Result<()> {
        info!("📋 Executing PR review task: {}", task.description);

        if let Some(pr_reviewer) = &self.pr_reviewer {
            // Extract PR number from task description
            let pr_number = task.description
                .split('#')
                .nth(1)
                .and_then(|s| s.parse::<u32>().ok())
                .ok_or_else(|| anyhow::anyhow!("Invalid PR number in task description"))?;

            // Perform story-driven review
            match pr_reviewer.review_pr(pr_number).await {
                Ok(review_result) => {
                    info!(
                        "PR #{} reviewed: {:?} (confidence: {:.2})",
                        pr_number, review_result.overall_assessment, review_result.confidence
                    );

                    // Store review result in memory
                    self.memory
                        .store(
                            format!("Story-driven PR review completed for #{}", pr_number),
                            vec![serde_json::to_string(&review_result)?],
                            MemoryMetadata {
                                source: "story_driven_autonomy".to_string(),
                                tags: vec!["pr_review".to_string(), "autonomous".to_string()],
                                importance: 0.8,
                                associations: vec![],
                                context: Some("autonomous PR review".to_string()),
                                created_at: chrono::Utc::now(),
                                accessed_count: 0,
                                last_accessed: None,
                                version: 1,
                    category: "story".to_string(),
                                timestamp: chrono::Utc::now(),
                                expiration: None,
                            },
                        )
                        .await?;

                    // If auto-approval is eligible, record it
                    if review_result.auto_approve_eligible {
                        info!("PR #{} is eligible for auto-approval", pr_number);
                    }
                }
                Err(e) => {
                    error!("Failed to review PR #{}: {}", pr_number, e);
                }
            }
        } else {
            warn!("PR reviewer not initialized, skipping review");
        }

        Ok(())
    }

    /// Get current maintenance status
    pub async fn get_status(&self) -> Result<MaintenanceStatus> {
        let tasks = self.active_tasks.read().await;
        let patterns = self.learned_patterns.read().await;

        Ok(MaintenanceStatus {
            active_tasks: tasks.len(),
            pending_tasks: tasks.iter().filter(|t| t.status == TaskStatus::Pending).count(),
            completed_today: 0, // Would track this properly
            learned_patterns: patterns.len(),
            last_analysis: Utc::now(), // Would track this properly
        })
    }

    /// Calculate security score from risk summary
    fn calculate_security_score(&self, risk_summary: &RiskSummary) -> f64 {
        // Calculate score based on vulnerability counts (inverse weighted)
        let total_vulns = risk_summary.critical_count * 10
            + risk_summary.high_count * 5
            + risk_summary.medium_count * 2
            + risk_summary.low_count;

        // Convert to percentage score (100 = no vulnerabilities)
        if total_vulns == 0 {
            100.0
        } else {
            // Score decreases with more/severe vulnerabilities
            (100.0 - (total_vulns as f64 * 2.0).min(100.0)).max(0.0)
        }
    }
    
}

/// Codebase state information
#[derive(Debug, Clone)]
struct CodebaseState {
    pub total_issues: usize,
    pub critical_issues: usize,
    pub test_coverage: f32,
    pub code_quality_score: f32,
}

/// Maintenance status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaintenanceStatus {
    pub active_tasks: usize,
    pub pending_tasks: usize,
    pub completed_today: usize,
    pub learned_patterns: usize,
    pub last_analysis: chrono::DateTime<chrono::Utc>,
}

// Re-export UUID for convenience
use uuid;
