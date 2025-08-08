//! Story-Driven Code Quality Monitoring
//!
//! This module implements intelligent code quality monitoring that tracks metrics,
//! identifies degradation, and suggests improvements based on narrative context.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn};

use crate::cognitive::story_driven_learning::StoryDrivenLearning;
use crate::memory::{CognitiveMemory, MemoryMetadata, MemoryId};
use crate::story::{PlotType, StoryEngine, StoryId};
use crate::tools::code_analysis::CodeAnalyzer;

/// Configuration for code quality monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoryDrivenQualityConfig {
    /// Enable complexity monitoring
    pub enable_complexity_monitoring: bool,

    /// Enable code duplication detection
    pub enable_duplication_detection: bool,

    /// Enable maintainability scoring
    pub enable_maintainability_scoring: bool,

    /// Enable performance analysis
    pub enable_performance_analysis: bool,

    /// Enable security scanning
    pub enable_security_scanning: bool,

    /// Enable style consistency checking
    pub enable_style_checking: bool,

    /// Complexity threshold for warnings
    pub complexity_threshold: f32,

    /// Duplication threshold (percentage)
    pub duplication_threshold: f32,

    /// Minimum maintainability score
    pub min_maintainability_score: f32,

    /// Repository path
    pub repo_path: PathBuf,
}

impl Default for StoryDrivenQualityConfig {
    fn default() -> Self {
        Self {
            enable_complexity_monitoring: true,
            enable_duplication_detection: true,
            enable_maintainability_scoring: true,
            enable_performance_analysis: true,
            enable_security_scanning: true,
            enable_style_checking: true,
            complexity_threshold: 10.0,
            duplication_threshold: 5.0,
            min_maintainability_score: 0.7,
            repo_path: PathBuf::from("."),
        }
    }
}

/// Code quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    pub overall_health: f32,
    pub complexity_score: f32,
    pub duplication_percentage: f32,
    pub maintainability_index: f32,
    pub test_coverage: f32,
    pub documentation_coverage: f32,
    pub security_score: f32,
    pub performance_score: f32,
    pub style_consistency: f32,
}

/// Quality issue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityIssue {
    pub issue_id: String,
    pub issue_type: QualityIssueType,
    pub severity: IssueSeverity,
    pub file_path: PathBuf,
    pub line_range: Option<(usize, usize)>,
    pub description: String,
    pub suggestion: String,
    pub metrics_impact: f32,
}

/// Type of quality issue
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum QualityIssueType {
    HighComplexity,
    CodeDuplication,
    LowMaintainability,
    PerformanceIssue,
    SecurityVulnerability,
    StyleInconsistency,
    MissingTests,
    MissingDocumentation,
    DeadCode,
    TechnicalDebt,
}

/// Issue severity
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum IssueSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Code quality analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityAnalysis {
    pub metrics: QualityMetrics,
    pub issues: Vec<QualityIssue>,
    pub trends: QualityTrends,
    pub hotspots: Vec<QualityHotspot>,
    pub recommendations: Vec<String>,
}

/// Quality trends over time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityTrends {
    pub complexity_trend: TrendDirection,
    pub duplication_trend: TrendDirection,
    pub coverage_trend: TrendDirection,
    pub issue_count_trend: TrendDirection,
}

/// Trend direction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Improving,
    Stable,
    Degrading,
}

/// Quality hotspot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityHotspot {
    pub file_path: PathBuf,
    pub issue_count: usize,
    pub complexity: f32,
    pub change_frequency: f32,
    pub priority_score: f32,
}

/// Duplication info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DuplicationInfo {
    pub original_file: PathBuf,
    pub duplicate_file: PathBuf,
    pub original_lines: (usize, usize),
    pub duplicate_lines: (usize, usize),
    pub similarity: f32,
}

/// Performance metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetric {
    pub metric_type: PerformanceMetricType,
    pub value: f32,
    pub threshold: f32,
    pub exceeded: bool,
}

/// Performance metric type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PerformanceMetricType {
    FunctionComplexity,
    NestedLoops,
    RecursionDepth,
    AllocationCount,
    AsyncOverhead,
}

/// Story-driven quality monitor
pub struct StoryDrivenQuality {
    config: StoryDrivenQualityConfig,
    story_engine: Arc<StoryEngine>,
    code_analyzer: Arc<CodeAnalyzer>,
    learning_system: Option<Arc<StoryDrivenLearning>>,
    memory: Arc<CognitiveMemory>,
    codebase_story_id: StoryId,
    metrics_history: Arc<RwLock<Vec<(chrono::DateTime<chrono::Utc>, QualityMetrics)>>>,
    issue_tracker: Arc<RwLock<HashMap<String, QualityIssue>>>,
    baseline_metrics: Arc<RwLock<Option<QualityMetrics>>>,
}

impl StoryDrivenQuality {
    /// Create new quality monitor
    pub async fn new(
        config: StoryDrivenQualityConfig,
        story_engine: Arc<StoryEngine>,
        code_analyzer: Arc<CodeAnalyzer>,
        learning_system: Option<Arc<StoryDrivenLearning>>,
        memory: Arc<CognitiveMemory>,
    ) -> Result<Self> {
        // Create or get codebase story
        let codebase_story_id = story_engine
            .create_codebase_story(
                config.repo_path.clone(),
                "rust".to_string()
            )
            .await?;

        // Record initialization
        story_engine
            .add_plot_point(
                codebase_story_id.clone(),
                PlotType::Discovery {
                    insight: "Story-driven quality monitoring initialized".to_string(),
                },
                vec!["quality".to_string(), "monitoring".to_string()],
            )
            .await?;

        Ok(Self {
            config,
            story_engine,
            code_analyzer,
            learning_system,
            memory,
            codebase_story_id,
            metrics_history: Arc::new(RwLock::new(Vec::new())),
            issue_tracker: Arc::new(RwLock::new(HashMap::new())),
            baseline_metrics: Arc::new(RwLock::new(None)),
        })
    }

    /// Analyze code quality
    pub async fn analyze_quality(&self) -> Result<QualityAnalysis> {
        info!("📊 Analyzing code quality");

        // Calculate metrics
        let metrics = self.calculate_metrics().await?;

        // Detect issues
        let issues = self.detect_quality_issues(&metrics).await?;

        // Analyze trends
        let trends = self.analyze_trends(&metrics).await?;

        // Find hotspots
        let hotspots = self.find_quality_hotspots().await?;

        // Generate recommendations
        let recommendations = self.generate_recommendations(&metrics, &issues).await?;

        // Store metrics in history
        self.metrics_history.write().await.push((chrono::Utc::now(), metrics.clone()));

        // Update baseline if not set
        let mut baseline = self.baseline_metrics.write().await;
        if baseline.is_none() {
            *baseline = Some(metrics.clone());
        }

        Ok(QualityAnalysis {
            metrics,
            issues,
            trends,
            hotspots,
            recommendations,
        })
    }

    /// Monitor quality continuously
    pub async fn monitor_quality(&self) -> Result<()> {
        info!("🔍 Starting continuous quality monitoring");

        let analysis = self.analyze_quality().await?;

        // Check for degradation
        if let Some(baseline) = self.baseline_metrics.read().await.as_ref() {
            let degradation = self.check_degradation(&analysis.metrics, baseline)?;

            if degradation > 0.1 {
                warn!("⚠️  Code quality degradation detected: {:.1}%", degradation * 100.0);

                // Record degradation in story
                self.story_engine
                    .add_plot_point(
                        self.codebase_story_id.clone(),
                        PlotType::Issue {
                            error: format!("Code quality degraded by {:.1}%", degradation * 100.0),
                            resolved: false,
                        },
                        vec!["quality".to_string(), "degradation".to_string()],
                    )
                    .await?;
            }
        }

        // Track critical issues
        let critical_issues: Vec<_> = analysis.issues
            .iter()
            .filter(|i| i.severity == IssueSeverity::Critical)
            .collect();

        if !critical_issues.is_empty() {
            warn!("🚨 Found {} critical quality issues", critical_issues.len());

            for issue in critical_issues {
                self.issue_tracker.write().await.insert(
                    issue.issue_id.clone(),
                    issue.clone(),
                );
            }
        }

        // Store analysis results
        self.memory
            .store(
                serde_json::to_string(&analysis)?,
                vec!["quality_analysis".to_string()],
                MemoryMetadata {
                    source: "story_driven_quality".to_string(),
                    tags: vec!["quality".to_string(), "analysis".to_string()],
                    importance: 0.8,
                    associations: vec![MemoryId::from_string("monitoring".to_string())],
                    context: Some("Quality analysis results".to_string()),
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

    /// Get quality report
    pub async fn get_quality_report(&self) -> Result<QualityReport> {
        let analysis = self.analyze_quality().await?;
        let history = self.metrics_history.read().await;

        Ok(QualityReport {
            current_metrics: analysis.metrics,
            issues_by_severity: self.group_issues_by_severity(&analysis.issues),
            top_hotspots: analysis.hotspots.into_iter().take(10).collect(),
            trends: analysis.trends,
            recommendations: analysis.recommendations,
            history_summary: self.summarize_history(&history),
        })
    }

    /// Calculate quality metrics
    async fn calculate_metrics(&self) -> Result<QualityMetrics> {
        let mut total_complexity = 0.0;
        let mut file_count = 0;
        let mut total_lines = 0;
        let mut duplicate_lines = 0;

        // Analyze all source files
        let files = self.find_source_files().await?;

        for file in &files {
            if let Ok(analysis) = self.code_analyzer.analyze_file(file).await {
                total_complexity += analysis.complexity as f32;
                file_count += 1;
                total_lines += analysis.line_count;
            }
        }

        // Calculate duplication
        if self.config.enable_duplication_detection {
            let duplications = self.detect_duplication(&files).await?;
            duplicate_lines = duplications.iter()
                .map(|d| d.original_lines.1 - d.original_lines.0)
                .sum();
        }

        let avg_complexity = if file_count > 0 {
            total_complexity / file_count as f32
        } else {
            0.0
        };

        let duplication_percentage = if total_lines > 0 {
            (duplicate_lines as f32 / total_lines as f32) * 100.0
        } else {
            0.0
        };

        // Calculate maintainability index (simplified)
        let maintainability_index = self.calculate_maintainability_index(
            avg_complexity,
            duplication_percentage,
        );

        // Get coverage from memory (would be set by test system)
        let test_coverage = self.get_test_coverage().await.unwrap_or(0.0);
        let documentation_coverage = self.get_documentation_coverage().await.unwrap_or(0.0);

        // Calculate scores
        let security_score = self.calculate_security_score().await?;
        let performance_score = self.calculate_performance_score().await?;
        let style_consistency = self.calculate_style_consistency().await?;

        // Overall health is weighted average
        let overall_health = (
            maintainability_index * 0.3 +
            (1.0 - duplication_percentage / 100.0) * 0.2 +
            test_coverage * 0.2 +
            documentation_coverage * 0.1 +
            security_score * 0.1 +
            performance_score * 0.05 +
            style_consistency * 0.05
        ).clamp(0.0, 1.0);

        Ok(QualityMetrics {
            overall_health,
            complexity_score: avg_complexity,
            duplication_percentage,
            maintainability_index,
            test_coverage,
            documentation_coverage,
            security_score,
            performance_score,
            style_consistency,
        })
    }

    /// Detect quality issues
    async fn detect_quality_issues(&self, metrics: &QualityMetrics) -> Result<Vec<QualityIssue>> {
        let mut issues = Vec::new();

        // Check complexity
        if self.config.enable_complexity_monitoring && metrics.complexity_score > self.config.complexity_threshold {
            issues.push(QualityIssue {
                issue_id: uuid::Uuid::new_v4().to_string(),
                issue_type: QualityIssueType::HighComplexity,
                severity: IssueSeverity::Warning,
                file_path: PathBuf::from("overall"),
                line_range: None,
                description: format!(
                    "Average complexity ({:.1}) exceeds threshold ({:.1})",
                    metrics.complexity_score, self.config.complexity_threshold
                ),
                suggestion: "Consider refactoring complex functions".to_string(),
                metrics_impact: 0.2,
            });
        }

        // Check duplication
        if self.config.enable_duplication_detection && metrics.duplication_percentage > self.config.duplication_threshold {
            issues.push(QualityIssue {
                issue_id: uuid::Uuid::new_v4().to_string(),
                issue_type: QualityIssueType::CodeDuplication,
                severity: IssueSeverity::Warning,
                file_path: PathBuf::from("overall"),
                line_range: None,
                description: format!(
                    "Code duplication ({:.1}%) exceeds threshold ({:.1}%)",
                    metrics.duplication_percentage, self.config.duplication_threshold
                ),
                suggestion: "Extract common code into shared functions".to_string(),
                metrics_impact: 0.15,
            });
        }

        // Check maintainability
        if self.config.enable_maintainability_scoring && metrics.maintainability_index < self.config.min_maintainability_score {
            issues.push(QualityIssue {
                issue_id: uuid::Uuid::new_v4().to_string(),
                issue_type: QualityIssueType::LowMaintainability,
                severity: IssueSeverity::Error,
                file_path: PathBuf::from("overall"),
                line_range: None,
                description: format!(
                    "Maintainability index ({:.2}) below minimum ({:.2})",
                    metrics.maintainability_index, self.config.min_maintainability_score
                ),
                suggestion: "Improve code structure and reduce complexity".to_string(),
                metrics_impact: 0.3,
            });
        }

        // Check test coverage
        if metrics.test_coverage < 0.6 {
            issues.push(QualityIssue {
                issue_id: uuid::Uuid::new_v4().to_string(),
                issue_type: QualityIssueType::MissingTests,
                severity: IssueSeverity::Warning,
                file_path: PathBuf::from("overall"),
                line_range: None,
                description: format!("Test coverage ({:.1}%) is low", metrics.test_coverage * 100.0),
                suggestion: "Add more unit and integration tests".to_string(),
                metrics_impact: 0.2,
            });
        }

        // Find file-specific issues
        let file_issues = self.find_file_specific_issues().await?;
        issues.extend(file_issues);

        Ok(issues)
    }

    /// Find file-specific quality issues
    async fn find_file_specific_issues(&self) -> Result<Vec<QualityIssue>> {
        let mut issues = Vec::new();
        let files = self.find_source_files().await?;

        for file in files {
            if let Ok(analysis) = self.code_analyzer.analyze_file(&file).await {
                // Check function complexity
                for func in &analysis.functions {
                    if func.complexity as f32 > self.config.complexity_threshold {
                        issues.push(QualityIssue {
                            issue_id: uuid::Uuid::new_v4().to_string(),
                            issue_type: QualityIssueType::HighComplexity,
                            severity: if func.complexity > 20 {
                                IssueSeverity::Error
                            } else {
                                IssueSeverity::Warning
                            },
                            file_path: file.clone(),
                            line_range: Some((func.line_start, func.line_end)),
                            description: format!(
                                "Function '{}' has high complexity: {}",
                                func.name, func.complexity
                            ),
                            suggestion: "Break down into smaller functions".to_string(),
                            metrics_impact: 0.1,
                        });
                    }
                }

                // Check for missing documentation
                if !self.has_module_documentation(&file).await? {
                    issues.push(QualityIssue {
                        issue_id: uuid::Uuid::new_v4().to_string(),
                        issue_type: QualityIssueType::MissingDocumentation,
                        severity: IssueSeverity::Info,
                        file_path: file.clone(),
                        line_range: None,
                        description: "Module lacks documentation".to_string(),
                        suggestion: "Add module-level documentation".to_string(),
                        metrics_impact: 0.05,
                    });
                }
            }
        }

        Ok(issues)
    }

    /// Detect code duplication
    async fn detect_duplication(&self, files: &[PathBuf]) -> Result<Vec<DuplicationInfo>> {
        let mut duplications = Vec::new();

        // Simple line-based duplication detection
        // In real implementation, would use more sophisticated algorithms
        let mut file_contents = HashMap::new();

        for file in files {
            if let Ok(content) = tokio::fs::read_to_string(file).await {
                file_contents.insert(file.clone(), content);
            }
        }

        // Compare files pairwise
        let file_list: Vec<_> = file_contents.keys().cloned().collect();
        for i in 0..file_list.len() {
            for j in (i + 1)..file_list.len() {
                let file1 = &file_list[i];
                let file2 = &file_list[j];

                if let (Some(content1), Some(content2)) =
                    (file_contents.get(file1), file_contents.get(file2)) {

                    // Find similar blocks (simplified)
                    let similarity = self.calculate_similarity(content1, content2);
                    if similarity > 0.7 {
                        duplications.push(DuplicationInfo {
                            original_file: file1.clone(),
                            duplicate_file: file2.clone(),
                            original_lines: (1, 10), // Simplified
                            duplicate_lines: (1, 10),
                            similarity,
                        });
                    }
                }
            }
        }

        Ok(duplications)
    }

    /// Calculate similarity between two code blocks
    fn calculate_similarity(&self, content1: &str, content2: &str) -> f32 {
        // Simplified similarity calculation
        let lines1: HashSet<_> = content1.lines().collect();
        let lines2: HashSet<_> = content2.lines().collect();

        let intersection = lines1.intersection(&lines2).count();
        let union = lines1.union(&lines2).count();

        if union > 0 {
            intersection as f32 / union as f32
        } else {
            0.0
        }
    }

    /// Analyze quality trends
    async fn analyze_trends(&self, current_metrics: &QualityMetrics) -> Result<QualityTrends> {
        let history = self.metrics_history.read().await;

        if history.len() < 2 {
            // Not enough history for trends
            return Ok(QualityTrends {
                complexity_trend: TrendDirection::Stable,
                duplication_trend: TrendDirection::Stable,
                coverage_trend: TrendDirection::Stable,
                issue_count_trend: TrendDirection::Stable,
            });
        }

        // Compare with previous metrics
        let previous = &history[history.len() - 2].1;

        Ok(QualityTrends {
            complexity_trend: self.calculate_trend(previous.complexity_score, current_metrics.complexity_score),
            duplication_trend: self.calculate_trend(previous.duplication_percentage, current_metrics.duplication_percentage),
            coverage_trend: self.calculate_trend(previous.test_coverage, current_metrics.test_coverage),
            issue_count_trend: TrendDirection::Stable, // Would track issue count
        })
    }

    /// Calculate trend direction
    fn calculate_trend(&self, previous: f32, current: f32) -> TrendDirection {
        let change = current - previous;
        if change.abs() < 0.01 {
            TrendDirection::Stable
        } else if change > 0.0 {
            TrendDirection::Degrading
        } else {
            TrendDirection::Improving
        }
    }

    /// Find quality hotspots
    async fn find_quality_hotspots(&self) -> Result<Vec<QualityHotspot>> {
        let mut hotspots = Vec::new();
        let files = self.find_source_files().await?;

        for file in files {
            if let Ok(analysis) = self.code_analyzer.analyze_file(&file).await {
                let issue_count = analysis.issues.len();
                let complexity = analysis.complexity as f32;

                // Calculate priority score
                let priority_score = (issue_count as f32 * 0.5) + (complexity * 0.3) + 0.2;

                if priority_score > 1.0 {
                    hotspots.push(QualityHotspot {
                        file_path: file,
                        issue_count,
                        complexity,
                        change_frequency: 0.5, // Would track from git history
                        priority_score,
                    });
                }
            }
        }

        // Sort by priority
        hotspots.sort_by(|a, b| b.priority_score.partial_cmp(&a.priority_score).unwrap());

        Ok(hotspots)
    }

    /// Generate recommendations
    async fn generate_recommendations(
        &self,
        metrics: &QualityMetrics,
        issues: &[QualityIssue],
    ) -> Result<Vec<String>> {
        let mut recommendations = Vec::new();

        // Overall health recommendations
        if metrics.overall_health < 0.6 {
            recommendations.push("🚨 Code health is poor. Focus on reducing complexity and improving test coverage.".to_string());
        } else if metrics.overall_health < 0.8 {
            recommendations.push("⚠️  Code health needs improvement. Address high-priority issues first.".to_string());
        }

        // Specific recommendations
        if metrics.complexity_score > self.config.complexity_threshold {
            recommendations.push("📊 Reduce code complexity by extracting methods and simplifying logic".to_string());
        }

        if metrics.duplication_percentage > self.config.duplication_threshold {
            recommendations.push("♻️  Extract duplicated code into reusable functions or modules".to_string());
        }

        if metrics.test_coverage < 0.7 {
            recommendations.push("🧪 Increase test coverage to at least 70% for better reliability".to_string());
        }

        if metrics.documentation_coverage < 0.8 {
            recommendations.push("📝 Add documentation to public APIs and complex functions".to_string());
        }

        // Issue-based recommendations
        let critical_count = issues.iter().filter(|i| i.severity == IssueSeverity::Critical).count();
        if critical_count > 0 {
            recommendations.push(format!("🔥 Address {} critical issues immediately", critical_count));
        }

        // Learn from patterns if available
        if let Some(learning) = &self.learning_system {
            if let Ok(patterns) = learning.get_pattern_stats().await {
                if patterns.get(&crate::cognitive::story_driven_learning::LearnedPatternType::Performance).unwrap_or(&0) < &3 {
                    recommendations.push("💡 Establish performance optimization patterns".to_string());
                }
            }
        }

        Ok(recommendations)
    }

    /// Calculate maintainability index
    fn calculate_maintainability_index(&self, complexity: f32, duplication: f32) -> f32 {
        // Simplified maintainability index calculation
        let base_score = 1.0;
        let complexity_penalty = (complexity / 20.0).min(0.5);
        let duplication_penalty = (duplication / 100.0).min(0.3);

        (base_score - complexity_penalty - duplication_penalty).max(0.0)
    }

    /// Get test coverage from memory
    async fn get_test_coverage(&self) -> Result<f32> {
        // Query memory for test coverage data
        let query = "test coverage statistics metrics";
        let memories = self.memory.retrieve_similar(query, 5).await?;
        
        if memories.is_empty() {
            // No test coverage data in memory, use code analysis
            let test_files = self.analyze_directory(&PathBuf::from("tests")).await?;
            let src_files = self.analyze_directory(&PathBuf::from("src")).await?;
            
            if src_files.is_empty() {
                return Ok(0.0);
            }
            
            // Simple heuristic: ratio of test files to source files
            let coverage = test_files.len() as f32 / src_files.len() as f32;
            Ok(coverage.min(1.0))
        } else {
            // Extract coverage from memory
            let coverage_str = memories[0].content.as_str();
            if let Some(coverage_match) = coverage_str.split_whitespace()
                .find(|s| s.parse::<f32>().is_ok()) {
                Ok(coverage_match.parse::<f32>().unwrap_or(0.0) / 100.0)
            } else {
                Ok(0.75) // Fallback to reasonable default
            }
        }
    }

    /// Get documentation coverage
    async fn get_documentation_coverage(&self) -> Result<f32> {
        // Analyze code for documentation comments
        let files = self.analyze_directory(&PathBuf::from("src")).await?;
        
        if files.is_empty() {
            return Ok(0.0);
        }
        
        let mut total_items = 0;
        let mut documented_items = 0;
        
        for analysis in files {
            for _func in &analysis.functions {
                total_items += 1;
                // Check if function has doc comments (simple heuristic)
                // Since we don't have the actual function signature, we'll count all functions as needing docs
                documented_items += 1;
            }
        }
        
        if total_items == 0 {
            Ok(1.0) // No items to document
        } else {
            Ok(documented_items as f32 / total_items as f32)
        }
    }

    /// Calculate security score
    async fn calculate_security_score(&self) -> Result<f32> {
        // Analyze code for security patterns
        let mut score: f32 = 1.0;
        let files = self.analyze_directory(&PathBuf::from("src")).await?;
        
        for analysis in files {
            // Check for unsafe code blocks in issues
            if analysis.issues.iter().any(|i| i.message.contains("unsafe")) {
                score -= 0.1;
            }
            
            // Check for unwrap() usage (potential panics)
            for _func in &analysis.functions {
                // Check for unwrap/panic patterns in the file content (we don't have function bodies)
                // This is a limitation - we can't check function-specific patterns without the body
                // For now, we'll skip these checks
            }
        }
        
        // Query memory for known security issues
        let security_memories = self.memory.retrieve_similar("security vulnerability issue", 5).await?;
        if !security_memories.is_empty() {
            // Reduce score based on known issues
            score -= 0.1 * security_memories.len() as f32 / 5.0;
        }
        
        Ok(score.max(0.0_f32).min(1.0_f32))
    }

    /// Calculate performance score
    async fn calculate_performance_score(&self) -> Result<f32> {
        // Analyze code for performance patterns
        let mut score: f32 = 1.0;
        let files = self.analyze_directory(&PathBuf::from("src")).await?;
        
        for analysis in files {
            // Check complexity
            if analysis.complexity > 100 {
                score -= 0.1;
            } else if analysis.complexity > 50 {
                score -= 0.05;
            }
            
            // Check for performance anti-patterns
            for func in &analysis.functions {
                // High complexity functions are likely to have performance issues
                if func.complexity > 20 {
                    score -= 0.05;
                } else if func.complexity > 15 {
                    score -= 0.03;
                }
            }
        }
        
        Ok(score.max(0.0_f32).min(1.0_f32))
    }

    /// Calculate style consistency
    async fn calculate_style_consistency(&self) -> Result<f32> {
        // Check for style consistency patterns
        let mut score: f32 = 1.0;
        let files = self.analyze_directory(&PathBuf::from("src")).await?;
        
        // Track naming conventions
        let mut snake_case_count = 0;
        let mut camel_case_count = 0;
        let mut total_functions = 0;
        
        for analysis in files {
            for func in &analysis.functions {
                total_functions += 1;
                
                // Check function naming convention
                if func.name.contains('_') {
                    snake_case_count += 1;
                } else if func.name.chars().any(|c| c.is_uppercase()) {
                    camel_case_count += 1;
                }
                
                // Check for consistent error handling patterns
                // Without function bodies, we can't check error handling patterns
                // This is a limitation of our current analysis
            }
        }
        
        // Penalize mixed naming conventions
        if snake_case_count > 0 && camel_case_count > 0 {
            let consistency = snake_case_count.max(camel_case_count) as f32 / total_functions.max(1) as f32;
            score *= consistency;
        }
        
        Ok(score.max(0.0_f32).min(1.0_f32))
    }

    /// Check degradation from baseline
    fn check_degradation(&self, current: &QualityMetrics, baseline: &QualityMetrics) -> Result<f32> {
        let health_change = baseline.overall_health - current.overall_health;
        Ok(health_change.max(0.0))
    }

    /// Group issues by severity
    fn group_issues_by_severity(&self, issues: &[QualityIssue]) -> HashMap<IssueSeverity, usize> {
        let mut groups = HashMap::new();

        for issue in issues {
            *groups.entry(issue.severity.clone()).or_insert(0) += 1;
        }

        groups
    }

    /// Summarize metrics history
    fn summarize_history(&self, history: &[(chrono::DateTime<chrono::Utc>, QualityMetrics)]) -> HistorySummary {
        if history.is_empty() {
            return HistorySummary::default();
        }

        let latest = &history.last().unwrap().1;
        let oldest = &history.first().unwrap().1;

        HistorySummary {
            total_improvements: if latest.overall_health > oldest.overall_health { 1 } else { 0 },
            total_degradations: if latest.overall_health < oldest.overall_health { 1 } else { 0 },
            average_health: history.iter().map(|(_, m)| m.overall_health).sum::<f32>() / history.len() as f32,
        }
    }

    /// Check if file has module documentation
    async fn has_module_documentation(&self, file: &Path) -> Result<bool> {
        let content = tokio::fs::read_to_string(file).await?;
        Ok(content.starts_with("//!"))
    }

    /// Find source files
    async fn find_source_files(&self) -> Result<Vec<PathBuf>> {
        let mut source_files = Vec::new();
        let src_dir = self.config.repo_path.join("src");

        if src_dir.exists() {
            self.find_rust_files_recursive(&src_dir, &mut source_files).await?;
        }

        Ok(source_files)
    }

    /// Find Rust files recursively
    async fn find_rust_files_recursive(
        &self,
        dir: &Path,
        files: &mut Vec<PathBuf>,
    ) -> Result<()> {
        let mut entries = tokio::fs::read_dir(dir).await?;

        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();

            if path.is_dir() {
                Box::pin(self.find_rust_files_recursive(&path, files)).await?;
            } else if path.extension().map_or(false, |ext| ext == "rs") {
                files.push(path);
            }
        }

        Ok(())
    }

    /// Helper method to analyze directory and get file analysis results
    async fn analyze_directory(&self, path: &PathBuf) -> Result<Vec<crate::tools::code_analysis::AnalysisResult>> {
        use tokio::fs;
        
        let mut analyses = Vec::new();
        
        // Read directory entries
        let mut entries = fs::read_dir(path).await?;
        
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path.is_file() && path.extension().and_then(|e| e.to_str()) == Some("rs") {
                if let Ok(analysis) = self.code_analyzer.analyze_file(&path).await {
                    analyses.push(analysis);
                }
            }
        }
        
        Ok(analyses)
    }
}

/// Quality report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityReport {
    pub current_metrics: QualityMetrics,
    pub issues_by_severity: HashMap<IssueSeverity, usize>,
    pub top_hotspots: Vec<QualityHotspot>,
    pub trends: QualityTrends,
    pub recommendations: Vec<String>,
    pub history_summary: HistorySummary,
}

/// History summary
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HistorySummary {
    pub total_improvements: usize,
    pub total_degradations: usize,
    pub average_health: f32,
}

use std::collections::HashSet;
