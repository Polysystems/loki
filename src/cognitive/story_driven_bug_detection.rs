//! Story-Driven Bug Detection and Fixing
//!
//! This module implements intelligent bug detection that understands context
//! through the story system, learning from patterns to detect, analyze, and
//! autonomously fix bugs while maintaining narrative coherence.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{info, warn};

use crate::cognitive::anomaly_detection::AnomalyDetector;
use crate::cognitive::self_modify::{ChangeType, CodeChange, RiskLevel, SelfModificationPipeline};
use crate::cognitive::story_driven_code_generation::StoryDrivenCodeGenerator;
use crate::cognitive::test_generator::TestGenerator;
use crate::memory::{CognitiveMemory, MemoryMetadata};
use crate::story::{
    PlotType, StoryEngine, StoryId, StorySegment,
};
use crate::tools::code_analysis::{CodeAnalyzer, CodeIssue, IssueSeverity};

/// Configuration for story-driven bug detection
#[derive(Debug, Clone)]
pub struct StoryDrivenBugDetectionConfig {
    /// Enable pattern-based bug detection
    pub enable_pattern_detection: bool,

    /// Enable anomaly-based bug detection
    pub enable_anomaly_detection: bool,

    /// Enable runtime analysis
    pub enable_runtime_analysis: bool,

    /// Enable automated fixing
    pub enable_auto_fix: bool,

    /// Enable test generation for fixes
    pub enable_test_generation: bool,

    /// Confidence threshold for auto-fixing
    pub auto_fix_threshold: f32,

    /// Maximum risk level for auto-fixes
    pub max_fix_risk_level: RiskLevel,

    /// Bug detection sensitivity
    pub detection_sensitivity: DetectionSensitivity,

    /// Repository path
    pub repo_path: PathBuf,
}

impl Default for StoryDrivenBugDetectionConfig {
    fn default() -> Self {
        Self {
            enable_pattern_detection: true,
            enable_anomaly_detection: true,
            enable_runtime_analysis: false,
            enable_auto_fix: true,
            enable_test_generation: true,
            auto_fix_threshold: 0.85,
            max_fix_risk_level: RiskLevel::Medium,
            detection_sensitivity: DetectionSensitivity::Balanced,
            repo_path: PathBuf::from("."),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum DetectionSensitivity {
    High,      // More false positives, fewer missed bugs
    Balanced,  // Balance between precision and recall
    Low,       // Fewer false positives, may miss subtle bugs
}

/// Story-driven bug detection system
pub struct StoryDrivenBugDetection {
    config: StoryDrivenBugDetectionConfig,
    story_engine: Arc<StoryEngine>,
    code_analyzer: Arc<CodeAnalyzer>,
    anomaly_detector: Arc<dyn AnomalyDetector>,
    self_modify: Arc<SelfModificationPipeline>,
    code_generator: Arc<StoryDrivenCodeGenerator>,
    test_generator: Arc<TestGenerator>,
    memory: Arc<CognitiveMemory>,

    /// Codebase story ID
    codebase_story_id: StoryId,

    /// Learned bug patterns
    bug_patterns: Arc<RwLock<HashMap<String, BugPattern>>>,

    /// Detection history for learning
    detection_history: Arc<RwLock<Vec<DetectionRecord>>>,

    /// Active bugs being tracked
    active_bugs: Arc<RwLock<HashMap<String, TrackedBug>>>,
}

/// Learned bug pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BugPattern {
    pub pattern_id: String,
    pub pattern_type: BugPatternType,
    pub description: String,
    pub detection_signature: Vec<String>,
    pub common_fixes: Vec<FixTemplate>,
    pub severity: BugSeverity,
    pub confidence: f32,
    pub occurrences: usize,
    pub successful_fixes: usize,
    pub false_positive_rate: f32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum BugPatternType {
    NullPointer,
    ResourceLeak,
    RaceCondition,
    LogicError,
    BoundsViolation,
    TypeMismatch,
    UnhandledError,
    PerformanceBug,
    SecurityVulnerability,
    MemoryCorruption,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum BugSeverity {
    Critical,  // System crash, data loss
    High,      // Major functionality broken
    Medium,    // Minor functionality affected
    Low,       // Cosmetic or edge case
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixTemplate {
    pub template_id: String,
    pub description: String,
    pub code_pattern: String,
    pub replacement_pattern: String,
    pub preconditions: Vec<String>,
    pub postconditions: Vec<String>,
    pub success_rate: f32,
}

/// Tracked bug information
#[derive(Debug, Clone)]
pub struct TrackedBug {
    pub bug_id: String,
    pub detected_at: chrono::DateTime<chrono::Utc>,
    pub location: BugLocation,
    pub bug_type: DetectedBugType,
    pub severity: BugSeverity,
    pub confidence: f32,
    pub description: String,
    pub potential_fixes: Vec<PotentialFix>,
    pub status: BugStatus,
    pub story_context: Option<StoryContext>,
}

#[derive(Debug, Clone)]
pub struct BugLocation {
    pub file_path: PathBuf,
    pub line_range: Option<(usize, usize)>,
    pub function_name: Option<String>,
    pub code_snippet: String,
}

#[derive(Debug, Clone)]
pub enum DetectedBugType {
    PatternMatch(BugPattern),
    AnomalyDetection(AnomalyInfo),
    RuntimeError(RuntimeErrorInfo),
    StaticAnalysis(CodeIssue),
}

#[derive(Debug, Clone)]
pub struct AnomalyInfo {
    pub anomaly_type: String,
    pub deviation_score: f32,
    pub expected_behavior: String,
    pub actual_behavior: String,
}

#[derive(Debug, Clone)]
pub struct RuntimeErrorInfo {
    pub error_type: String,
    pub stack_trace: Vec<String>,
    pub frequency: usize,
    pub last_occurrence: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone)]
pub struct PotentialFix {
    pub fix_id: String,
    pub description: String,
    pub code_change: ProposedCodeChange,
    pub confidence: f32,
    pub risk_level: RiskLevel,
    pub requires_testing: bool,
}

#[derive(Debug, Clone)]
pub struct ProposedCodeChange {
    pub old_code: String,
    pub new_code: String,
    pub explanation: String,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BugStatus {
    Detected,
    Analyzing,
    FixProposed,
    FixApplied,
    TestingFix,
    Resolved,
    CannotFix,
}

#[derive(Debug, Clone)]
struct StoryContext {
    pub related_goals: Vec<String>,
    pub related_issues: Vec<String>,
    pub narrative_impact: f32,
}

#[derive(Debug, Clone)]
struct DetectionRecord {
    pub bug_id: String,
    pub detection_time: chrono::DateTime<chrono::Utc>,
    pub was_real_bug: Option<bool>,
    pub fix_successful: Option<bool>,
    pub feedback: Option<String>,
}

impl StoryDrivenBugDetection {
    /// Create a new story-driven bug detection system
    pub async fn new(
        config: StoryDrivenBugDetectionConfig,
        story_engine: Arc<StoryEngine>,
        self_modify: Arc<SelfModificationPipeline>,
        code_generator: Arc<StoryDrivenCodeGenerator>,
        memory: Arc<CognitiveMemory>,
    ) -> Result<Self> {
        info!("🐛 Initializing Story-Driven Bug Detection System");

        // Create or get codebase story
        let codebase_story_id = story_engine
            .create_codebase_story(
                config.repo_path.clone(),
                "rust".to_string()
            )
            .await?;

        // Initialize components
        let code_analyzer = Arc::new(CodeAnalyzer::new(memory.clone()).await?);
        let anomaly_detector = Arc::new(
            crate::cognitive::anomaly_detection::StatisticalAnomalyDetector::new()
        );
        let test_generator = Arc::new(
            TestGenerator::new(Default::default(), memory.clone()).await?
        );

        // Load bug patterns
        let patterns = Self::load_bug_patterns(&memory).await?;

        // Record initialization in story
        story_engine
            .add_plot_point(
                codebase_story_id.clone(),
                PlotType::Goal {
                    objective: "Initialize autonomous bug detection".to_string(),
                },
                vec!["bug_detection", "initialization"].iter().map(|s| s.to_string()).collect(), // context_tokens
            )
            .await?;

        Ok(Self {
            config,
            story_engine,
            code_analyzer,
            anomaly_detector,
            self_modify,
            code_generator,
            test_generator,
            memory,
            codebase_story_id,
            bug_patterns: Arc::new(RwLock::new(patterns)),
            detection_history: Arc::new(RwLock::new(Vec::new())),
            active_bugs: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Scan codebase for bugs
    pub async fn scan_codebase(&self) -> Result<Vec<TrackedBug>> {
        info!("🔍 Scanning codebase for bugs with story context");

        let mut detected_bugs = Vec::new();

        // Get story context
        let story_context = self.build_story_context().await?;

        // 1. Pattern-based detection
        if self.config.enable_pattern_detection {
            let pattern_bugs = self.detect_pattern_bugs(&story_context).await?;
            detected_bugs.extend(pattern_bugs);
        }

        // 2. Anomaly-based detection
        if self.config.enable_anomaly_detection {
            let anomaly_bugs = self.detect_anomaly_bugs(&story_context).await?;
            detected_bugs.extend(anomaly_bugs);
        }

        // 3. Static analysis
        let static_bugs = self.detect_static_analysis_bugs().await?;
        detected_bugs.extend(static_bugs);

        // Store detected bugs
        let mut active_bugs = self.active_bugs.write().await;
        for bug in &detected_bugs {
            active_bugs.insert(bug.bug_id.clone(), bug.clone());
        }

        // Record detection in story
        if !detected_bugs.is_empty() {
            self.story_engine
                .add_plot_point(
                    self.codebase_story_id.clone(),
                    PlotType::Discovery {
                        insight: format!("Detected {} bugs in codebase scan", detected_bugs.len()),
                    },
                    vec![],
                )
                .await?;
        }

        Ok(detected_bugs)
    }

    /// Detect bugs using learned patterns
    async fn detect_pattern_bugs(&self, context: &StoryContext) -> Result<Vec<TrackedBug>> {
        info!("🎯 Detecting bugs using learned patterns");

        let mut bugs = Vec::new();
        let patterns = self.bug_patterns.read().await;

        // Scan source files
        let src_path = self.config.repo_path.join("src");
        if src_path.exists() {
            bugs.extend(self.scan_directory_for_patterns(&src_path, &patterns).await?);
        }

        // Apply story context to adjust confidence
        for bug in &mut bugs {
            for issue in &context.related_issues {
                if bug.description.contains(issue) {
                    bug.confidence *= 1.2; // Boost confidence for story-related bugs
                }
            }
        }

        Ok(bugs)
    }

    /// Scan directory for pattern matches
    async fn scan_directory_for_patterns(
        &self,
        dir: &Path,
        patterns: &HashMap<String, BugPattern>,
    ) -> Result<Vec<TrackedBug>> {
        let mut bugs = Vec::new();

        let mut entries = tokio::fs::read_dir(dir).await?;

        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();

            if path.is_dir() {
                // Skip certain directories
                let dir_name = path.file_name().unwrap_or_default().to_str().unwrap_or("");
                if !matches!(dir_name, "target" | ".git" | "node_modules") {
                    let sub_bugs = Box::pin(self.scan_directory_for_patterns(&path, patterns)).await?;
                    bugs.extend(sub_bugs);
                }
            } else if path.is_file() && path.extension().map_or(false, |ext| ext == "rs") {
                // Scan Rust files
                if let Ok(content) = tokio::fs::read_to_string(&path).await {
                    bugs.extend(self.check_file_for_patterns(&path, &content, patterns).await?);
                }
            }
        }

        Ok(bugs)
    }

    /// Check file content for bug patterns
    async fn check_file_for_patterns(
        &self,
        file_path: &Path,
        content: &str,
        patterns: &HashMap<String, BugPattern>,
    ) -> Result<Vec<TrackedBug>> {
        let mut bugs = Vec::new();

        for (pattern_id, pattern) in patterns {
            for signature in &pattern.detection_signature {
                if content.contains(signature) {
                    // Find specific location
                    let location = self.find_pattern_location(content, signature, file_path);

                    // Generate potential fixes
                    let potential_fixes = self.generate_fixes_for_pattern(pattern, &location).await?;

                    let bug = TrackedBug {
                        bug_id: format!("bug_{}", uuid::Uuid::new_v4()),
                        detected_at: Utc::now(),
                        location,
                        bug_type: DetectedBugType::PatternMatch(pattern.clone()),
                        severity: pattern.severity.clone(),
                        confidence: pattern.confidence * self.sensitivity_multiplier(),
                        description: format!(
                            "{} detected: {}",
                            pattern.pattern_type as u8,
                            pattern.description
                        ),
                        potential_fixes,
                        status: BugStatus::Detected,
                        story_context: None,
                    };

                    bugs.push(bug);
                }
            }
        }

        Ok(bugs)
    }

    /// Detect anomalies that might indicate bugs
    async fn detect_anomaly_bugs(&self, _context: &StoryContext) -> Result<Vec<TrackedBug>> {
        info!("🔮 Detecting anomalies that might indicate bugs");

        let mut bugs = Vec::new();

        // For now, return empty as anomaly detection needs proper integration
        // TODO: Integrate with AnomalyDetectionSystem instead of trait object
        // Define a simple anomaly type for this context
        #[derive(Debug)]
        struct SimpleAnomaly {
            severity: f64,
            anomaly_type: String,
            details: String,
        }
        let anomalies: Vec<SimpleAnomaly> = Vec::new();

        for anomaly in anomalies {
            if anomaly.severity > 0.7 {
                let bug = TrackedBug {
                    bug_id: format!("anomaly_{}", uuid::Uuid::new_v4()),
                    detected_at: Utc::now(),
                    location: BugLocation {
                        file_path: PathBuf::from("unknown"),
                        line_range: None,
                        function_name: None,
                        code_snippet: String::new(),
                    },
                    bug_type: DetectedBugType::AnomalyDetection(AnomalyInfo {
                        anomaly_type: anomaly.anomaly_type.to_string(),
                        deviation_score: anomaly.severity as f32,
                        expected_behavior: "Normal system behavior".to_string(),
                        actual_behavior: anomaly.details.clone(),
                    }),
                    severity: if anomaly.severity > 0.9 {
                        BugSeverity::Critical
                    } else if anomaly.severity > 0.7 {
                        BugSeverity::High
                    } else {
                        BugSeverity::Medium
                    },
                    confidence: anomaly.severity as f32,
                    description: format!("Anomaly detected: {:?}", anomaly),
                    potential_fixes: vec![],
                    status: BugStatus::Detected,
                    story_context: None,
                };

                bugs.push(bug);
            }
        }

        Ok(bugs)
    }

    /// Detect bugs using static analysis
    async fn detect_static_analysis_bugs(&self) -> Result<Vec<TrackedBug>> {
        info!("🔧 Running static analysis for bug detection");

        let mut bugs = Vec::new();

        // Analyze source files
        let src_path = self.config.repo_path.join("src");
        if src_path.exists() {
            let analysis_results = self.analyze_directory(&src_path).await?;

            for (file_path, analysis) in analysis_results {
                for issue in analysis.issues {
                    if matches!(issue.severity, IssueSeverity::Error) {
                        let bug = TrackedBug {
                            bug_id: format!("static_{}", uuid::Uuid::new_v4()),
                            detected_at: Utc::now(),
                            location: BugLocation {
                                file_path: file_path.clone(),
                                line_range: Some((issue.line, issue.line)),
                                function_name: None,
                                code_snippet: String::new(), // CodeIssue doesn't have code_snippet field
                            },
                            bug_type: DetectedBugType::StaticAnalysis(issue.clone()),
                            severity: BugSeverity::High,
                            confidence: 0.9,
                            description: issue.message,
                            potential_fixes: vec![],
                            status: BugStatus::Detected,
                            story_context: None,
                        };

                        bugs.push(bug);
                    }
                }
            }
        }

        Ok(bugs)
    }

    /// Fix a detected bug autonomously
    pub async fn fix_bug(&self, bug_id: &str) -> Result<BugFixResult> {
        info!("🔧 Attempting to fix bug: {}", bug_id);

        let bugs = self.active_bugs.read().await;
        let bug = bugs.get(bug_id)
            .ok_or_else(|| anyhow::anyhow!("Bug not found: {}", bug_id))?
            .clone();
        drop(bugs);

        // Check if we should auto-fix
        if !self.config.enable_auto_fix {
            return Ok(BugFixResult::ManualFixRequired("Auto-fix disabled".to_string()));
        }

        if bug.confidence < self.config.auto_fix_threshold {
            return Ok(BugFixResult::ManualFixRequired(
                format!("Confidence {} below threshold {}", bug.confidence, self.config.auto_fix_threshold)
            ));
        }

        // Update status
        self.update_bug_status(bug_id, BugStatus::Analyzing).await?;

        // Generate fix based on bug type
        let fix_result = match &bug.bug_type {
            DetectedBugType::PatternMatch(pattern) => {
                self.fix_pattern_bug(&bug, pattern).await?
            }
            DetectedBugType::StaticAnalysis(_) => {
                self.fix_static_analysis_bug(&bug).await?
            }
            _ => {
                BugFixResult::CannotFix("Bug type not supported for auto-fix".to_string())
            }
        };

        // Update bug status based on result
        match &fix_result {
            BugFixResult::Fixed { .. } => {
                self.update_bug_status(bug_id, BugStatus::Resolved).await?;
            }
            BugFixResult::FixProposed { .. } => {
                self.update_bug_status(bug_id, BugStatus::FixProposed).await?;
            }
            BugFixResult::CannotFix(_) => {
                self.update_bug_status(bug_id, BugStatus::CannotFix).await?;
            }
            _ => {}
        }

        // Record fix attempt in story
        let plot_type = match &fix_result {
            BugFixResult::Fixed { .. } => PlotType::Task {
                description: format!("Fixed bug: {}", bug.description),
                completed: true,
            },
            _ => PlotType::Issue {
                error: format!("Could not fix: {}", bug.description),
                resolved: false,
            },
        };

        self.story_engine
            .add_plot_point(
                self.codebase_story_id.clone(),
                plot_type,
                vec![format!("bug_fix_{}", bug_id), "autonomous".to_string()],
            )
            .await?;

        Ok(fix_result)
    }

    /// Fix a pattern-matched bug
    async fn fix_pattern_bug(&self, bug: &TrackedBug, pattern: &BugPattern) -> Result<BugFixResult> {
        // Find best fix template
        let best_fix = pattern.common_fixes
            .iter()
            .max_by(|a, b| a.success_rate.partial_cmp(&b.success_rate).unwrap());

        if let Some(fix_template) = best_fix {
            // Read the file
            let content = tokio::fs::read_to_string(&bug.location.file_path).await?;

            // Apply fix template
            let fixed_content = content.replace(
                &fix_template.code_pattern,
                &fix_template.replacement_pattern
            );

            // Create code change
            let code_change = CodeChange {
                file_path: bug.location.file_path.clone(),
                change_type: ChangeType::BugFix,
                description: format!("Fix {}: {}", pattern.pattern_type as u8, bug.description),
                reasoning: format!("Applied fix template: {}", fix_template.description),
                old_content: Some(content),
                new_content: fixed_content,
                line_range: bug.location.line_range,
                risk_level: self.assess_fix_risk(pattern),
                attribution: None,
            };

            // Check risk level
            if code_change.risk_level > self.config.max_fix_risk_level {
                return Ok(BugFixResult::FixProposed {
                    proposed_change: code_change,
                    confidence: fix_template.success_rate,
                    requires_review: true,
                });
            }

            // Apply fix
            match self.self_modify.propose_change(code_change.clone()).await {
                Ok(pr) => {
                    // Generate tests if enabled
                    if self.config.enable_test_generation {
                        let _ = self.generate_test_for_fix(&bug, &code_change).await;
                    }

                    Ok(BugFixResult::Fixed {
                        pr_number: pr.number,
                        fix_description: fix_template.description.clone(),
                    })
                }
                Err(e) => {
                    Ok(BugFixResult::FixFailed(format!("Failed to apply fix: {}", e)))
                }
            }
        } else {
            // No fix template available, try to generate one
            self.generate_intelligent_fix(bug).await
        }
    }

    /// Fix a static analysis bug
    async fn fix_static_analysis_bug(&self, bug: &TrackedBug) -> Result<BugFixResult> {
        if let DetectedBugType::StaticAnalysis(issue) = &bug.bug_type {
            // Use code generator to create a fix
            let fix_prompt = format!(
                "Fix this {} issue: {}",
                issue.severity as u8,
                issue.message
            );

            // Generate fix using story context
            let generated_fix = self.code_generator
                .generate_from_story_context(&self.codebase_story_id, None)
                .await?;

            if let Some(fix) = generated_fix.first() {
                let code_change = CodeChange {
                    file_path: bug.location.file_path.clone(),
                    change_type: ChangeType::BugFix,
                    description: format!("Fix static analysis issue: {}", issue.message),
                    reasoning: "Generated fix for static analysis issue".to_string(),
                    old_content: None,
                    new_content: fix.content.clone(),
                    line_range: bug.location.line_range,
                    risk_level: fix.risk_level,
                    attribution: None,
                };

                // Apply fix
                match self.self_modify.propose_change(code_change).await {
                    Ok(pr) => Ok(BugFixResult::Fixed {
                        pr_number: pr.number,
                        fix_description: "Static analysis issue fixed".to_string(),
                    }),
                    Err(e) => Ok(BugFixResult::FixFailed(format!("Failed to apply fix: {}", e))),
                }
            } else {
                Ok(BugFixResult::CannotFix("Could not generate fix".to_string()))
            }
        } else {
            Ok(BugFixResult::CannotFix("Not a static analysis bug".to_string()))
        }
    }

    /// Generate an intelligent fix when no template is available
    async fn generate_intelligent_fix(&self, bug: &TrackedBug) -> Result<BugFixResult> {
        // Use code generator with bug context
        let bug_context = format!(
            "Fix this bug: {}\nLocation: {}:{:?}\nSeverity: {:?}",
            bug.description,
            bug.location.file_path.display(),
            bug.location.line_range,
            bug.severity
        );

        // Store bug context in memory for code generator
        self.memory
            .store(
                bug_context,
                vec![bug.location.code_snippet.clone()],
                MemoryMetadata {
                    source: "bug_detection".to_string(),
                    tags: vec!["bug_fix".to_string(), "generation".to_string()],
                    importance: 0.9,
                    associations: vec![],
                    context: Some("bug fix generation".to_string()),
                    created_at: chrono::Utc::now(),
                    accessed_count: 0,
                    last_accessed: None,
                    version: 1,
                    timestamp: chrono::Utc::now(),
                    expiration: None,
                    category: "bug_detection".to_string(),
                },
            )
            .await?;

        // Generate fix
        let generated_fixes = self.code_generator
            .generate_from_story_context(&self.codebase_story_id, None)
            .await?;

        if let Some(fix) = generated_fixes.first() {
            let code_change = CodeChange {
                file_path: bug.location.file_path.clone(),
                change_type: ChangeType::BugFix,
                description: format!("AI-generated fix for: {}", bug.description),
                reasoning: "Intelligently generated fix based on bug context".to_string(),
                old_content: None,
                new_content: fix.content.clone(),
                line_range: bug.location.line_range,
                risk_level: fix.risk_level,
                attribution: None,
            };

            Ok(BugFixResult::FixProposed {
                proposed_change: code_change,
                confidence: fix.confidence,
                requires_review: true,
            })
        } else {
            Ok(BugFixResult::CannotFix("Could not generate intelligent fix".to_string()))
        }
    }

    /// Generate tests for a bug fix
    async fn generate_test_for_fix(&self, bug: &TrackedBug, fix: &CodeChange) -> Result<()> {
        info!("🧪 Generating test for bug fix");

        // Create test description
        let test_description = format!(
            "Test that verifies fix for: {}",
            bug.description
        );

        // Generate test using test generator
        let test_file = fix.file_path.with_extension("test.rs");
        let test_suite = self.test_generator.generate_tests_for_file(&test_file).await?;

        if !test_suite.test_cases.is_empty() {
            // Create test file change
            let test_content = format!(
                "// Test for bug fix: {}\n\n{}",
                bug.bug_id,
                test_suite.test_cases.iter()
                    .map(|t| t.code.as_str())
                    .collect::<Vec<_>>()
                    .join("\n\n")
            );

            let test_change = CodeChange {
                file_path: test_file,
                change_type: ChangeType::Test,
                description: test_description,
                reasoning: "Ensure bug fix is properly tested".to_string(),
                old_content: None,
                new_content: test_content,
                line_range: None,
                risk_level: RiskLevel::Low,
                attribution: None,
            };

            // Apply test
            let _ = self.self_modify.propose_change(test_change).await;
        }

        Ok(())
    }

    /// Helper methods
    fn find_pattern_location(&self, content: &str, pattern: &str, file_path: &Path) -> BugLocation {
        let lines: Vec<&str> = content.lines().collect();
        let mut line_number = None;
        let mut snippet = String::new();

        for (i, line) in lines.iter().enumerate() {
            if line.contains(pattern) {
                line_number = Some(i + 1);
                // Get surrounding context
                let start = i.saturating_sub(2);
                let end = (i + 3).min(lines.len());
                snippet = lines[start..end].join("\n");
                break;
            }
        }

        BugLocation {
            file_path: file_path.to_path_buf(),
            line_range: line_number.map(|n| (n, n)),
            function_name: None,
            code_snippet: snippet,
        }
    }

    async fn generate_fixes_for_pattern(
        &self,
        pattern: &BugPattern,
        _location: &BugLocation,
    ) -> Result<Vec<PotentialFix>> {
        let mut fixes = Vec::new();

        for fix_template in &pattern.common_fixes {
            fixes.push(PotentialFix {
                fix_id: format!("fix_{}", uuid::Uuid::new_v4()),
                description: fix_template.description.clone(),
                code_change: ProposedCodeChange {
                    old_code: fix_template.code_pattern.clone(),
                    new_code: fix_template.replacement_pattern.clone(),
                    explanation: format!(
                        "Apply fix template with {:.0}% success rate",
                        fix_template.success_rate * 100.0
                    ),
                },
                confidence: fix_template.success_rate,
                risk_level: RiskLevel::Low,
                requires_testing: true,
            });
        }

        Ok(fixes)
    }

    fn sensitivity_multiplier(&self) -> f32 {
        match self.config.detection_sensitivity {
            DetectionSensitivity::High => 1.2,
            DetectionSensitivity::Balanced => 1.0,
            DetectionSensitivity::Low => 0.8,
        }
    }

    fn assess_fix_risk(&self, pattern: &BugPattern) -> RiskLevel {
        match pattern.severity {
            BugSeverity::Critical => RiskLevel::High,
            BugSeverity::High => RiskLevel::Medium,
            _ => RiskLevel::Low,
        }
    }

    async fn build_story_context(&self) -> Result<StoryContext> {
        let story = self.story_engine.get_story(&self.codebase_story_id)
            .ok_or_else(|| anyhow::anyhow!("Story not found"))?;
        let recent_segments: Vec<StorySegment> = vec![
            StorySegment {
                id: uuid::Uuid::new_v4().to_string(),
                story_id: story.id,
                content: story.summary.clone(),
                context: std::collections::HashMap::new(),
                created_at: story.updated_at,
                segment_type: crate::story::SegmentType::Development,
                tags: Vec::new(),
            }
        ]
            .into_iter()
            .rev()
            .take(5)
            .collect();

        let mut related_goals = Vec::new();
        let mut related_issues = Vec::new();

        // Note: Plot points are stored in Story.arcs, segments provide context
        // Extract goals and issues from segment content and context
        for segment in recent_segments {
            // Check segment context for type information
            if let Some(segment_type) = segment.context.get("type") {
                match segment_type.as_str() {
                    "goal" => {
                        // Extract goal from content
                        let goal = segment.content.lines().next().unwrap_or(&segment.content).to_string();
                        related_goals.push(goal);
                    },
                    "issue" | "error" => {
                        // Extract issue from content
                        let issue = segment.content.lines().next().unwrap_or(&segment.content).to_string();
                        related_issues.push(issue);
                    },
                    _ => {}
                }
            }

            // Also check for keywords in content
            if segment.content.contains("goal:") || segment.content.contains("objective:") {
                if let Some(goal_text) = segment.content.split("goal:").nth(1) {
                    related_goals.push(goal_text.lines().next().unwrap_or("").trim().to_string());
                }
            }
            if segment.content.contains("error:") || segment.content.contains("issue:") {
                if let Some(issue_text) = segment.content.split("error:").nth(1) {
                    related_issues.push(issue_text.lines().next().unwrap_or("").trim().to_string());
                }
            }
        }

        Ok(StoryContext {
            related_goals,
            related_issues,
            narrative_impact: 0.8, // Default impact
        })
    }

    async fn update_bug_status(&self, bug_id: &str, status: BugStatus) -> Result<()> {
        let mut bugs = self.active_bugs.write().await;
        if let Some(bug) = bugs.get_mut(bug_id) {
            bug.status = status;
        }
        Ok(())
    }

    async fn analyze_directory(&self, dir: &Path) -> Result<HashMap<PathBuf, crate::tools::code_analysis::AnalysisResult>> {
        let mut results = HashMap::new();

        let mut entries = tokio::fs::read_dir(dir).await?;

        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();

            if path.is_dir() {
                let dir_name = path.file_name().unwrap_or_default().to_str().unwrap_or("");
                if !matches!(dir_name, "target" | ".git" | "node_modules") {
                    let sub_results = Box::pin(self.analyze_directory(&path)).await?;
                    results.extend(sub_results);
                }
            } else if path.is_file() && path.extension().map_or(false, |ext| ext == "rs") {
                match self.code_analyzer.analyze_file(&path).await {
                    Ok(analysis) => {
                        results.insert(path, analysis);
                    }
                    Err(e) => {
                        warn!("Failed to analyze {}: {}", path.display(), e);
                    }
                }
            }
        }

        Ok(results)
    }

    /// Load bug patterns from memory
    async fn load_bug_patterns(memory: &CognitiveMemory) -> Result<HashMap<String, BugPattern>> {
        let mut patterns = HashMap::new();

        // Add default patterns
        patterns.insert(
            "unwrap_panic".to_string(),
            BugPattern {
                pattern_id: "unwrap_panic".to_string(),
                pattern_type: BugPatternType::UnhandledError,
                description: "Unsafe unwrap() that could panic".to_string(),
                detection_signature: vec![".unwrap()".to_string()],
                common_fixes: vec![
                    FixTemplate {
                        template_id: "use_question_mark".to_string(),
                        description: "Replace with ? operator".to_string(),
                        code_pattern: ".unwrap()".to_string(),
                        replacement_pattern: "?".to_string(),
                        preconditions: vec!["Function returns Result".to_string()],
                        postconditions: vec!["No panic possible".to_string()],
                        success_rate: 0.95,
                    },
                ],
                severity: BugSeverity::High,
                confidence: 0.9,
                occurrences: 0,
                successful_fixes: 0,
                false_positive_rate: 0.1,
            },
        );

        patterns.insert(
            "todo_macro".to_string(),
            BugPattern {
                pattern_id: "todo_macro".to_string(),
                pattern_type: BugPatternType::LogicError,
                description: "Unimplemented code using todo!()".to_string(),
                detection_signature: vec!["todo!()".to_string()],
                common_fixes: vec![],
                severity: BugSeverity::Medium,
                confidence: 1.0,
                occurrences: 0,
                successful_fixes: 0,
                false_positive_rate: 0.0,
            },
        );

        patterns.insert(
            "sql_injection".to_string(),
            BugPattern {
                pattern_id: "sql_injection".to_string(),
                pattern_type: BugPatternType::SecurityVulnerability,
                description: "Potential SQL injection vulnerability".to_string(),
                detection_signature: vec![
                    "format!(\"SELECT * FROM".to_string(),
                    "format!(\"INSERT INTO".to_string(),
                    "format!(\"UPDATE".to_string(),
                    "format!(\"DELETE FROM".to_string(),
                ],
                common_fixes: vec![
                    FixTemplate {
                        template_id: "use_prepared_statement".to_string(),
                        description: "Use prepared statements".to_string(),
                        code_pattern: "format!(\"SELECT".to_string(),
                        replacement_pattern: "sqlx::query!(\"SELECT".to_string(),
                        preconditions: vec!["Using sqlx".to_string()],
                        postconditions: vec!["SQL injection prevented".to_string()],
                        success_rate: 0.98,
                    },
                ],
                severity: BugSeverity::Critical,
                confidence: 0.85,
                occurrences: 0,
                successful_fixes: 0,
                false_positive_rate: 0.15,
            },
        );

        Ok(patterns)
    }

    /// Learn from bug fix outcome
    pub async fn learn_from_outcome(
        &self,
        bug_id: &str,
        was_real_bug: bool,
        fix_successful: bool,
        feedback: Option<String>,
    ) -> Result<()> {
        // Update detection history
        let mut history = self.detection_history.write().await;
        history.push(DetectionRecord {
            bug_id: bug_id.to_string(),
            detection_time: Utc::now(),
            was_real_bug: Some(was_real_bug),
            fix_successful: Some(fix_successful),
            feedback: feedback.clone(),
        });

        // Update pattern confidence if applicable
        let bugs = self.active_bugs.read().await;
        if let Some(bug) = bugs.get(bug_id) {
            if let DetectedBugType::PatternMatch(pattern) = &bug.bug_type {
                let mut patterns = self.bug_patterns.write().await;
                if let Some(stored_pattern) = patterns.get_mut(&pattern.pattern_id) {
                    stored_pattern.occurrences += 1;

                    if !was_real_bug {
                        stored_pattern.false_positive_rate =
                            (stored_pattern.false_positive_rate * stored_pattern.occurrences as f32 + 1.0) /
                            (stored_pattern.occurrences + 1) as f32;
                    } else if fix_successful {
                        stored_pattern.successful_fixes += 1;
                    }

                    // Adjust confidence based on accuracy
                    if was_real_bug {
                        stored_pattern.confidence = (stored_pattern.confidence * 0.9 + 0.1).min(1.0);
                    } else {
                        stored_pattern.confidence = (stored_pattern.confidence * 0.9).max(0.1);
                    }
                }
            }
        }

        // Store learning in memory
        self.memory
            .store(
                format!("Bug detection learning: {} (real: {}, fixed: {})", bug_id, was_real_bug, fix_successful),
                vec![feedback.unwrap_or_default()],
                MemoryMetadata {
                    source: "bug_detection".to_string(),
                    tags: vec!["learning".to_string(), "bug_fix".to_string()],
                    importance: 0.8,
                    associations: vec![],
                    context: Some("bug detection learning".to_string()),
                    created_at: chrono::Utc::now(),
                    accessed_count: 0,
                    last_accessed: None,
                    version: 1,
                    timestamp: chrono::Utc::now(),
                    expiration: None,
                    category: "bug_detection".to_string(),
                },
            )
            .await?;

        Ok(())
    }
}

/// Result of a bug fix attempt
#[derive(Debug, Clone)]
pub enum BugFixResult {
    Fixed {
        pr_number: u32,
        fix_description: String,
    },
    FixProposed {
        proposed_change: CodeChange,
        confidence: f32,
        requires_review: bool,
    },
    CannotFix(String),
    ManualFixRequired(String),
    FixFailed(String),
}

// Re-export UUID for convenience
use uuid;
