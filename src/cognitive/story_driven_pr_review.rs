//! Story-Driven PR Review System
//!
//! This module implements intelligent PR review that understands context through
//! the story system, providing narrative-aware code reviews with learned patterns
//! and maintaining coherence across the codebase evolution.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::info;

use crate::cognitive::pr_automation::PrAutomationSystem;
use crate::memory::{CognitiveMemory, MemoryMetadata};
use crate::story::{
    PlotType, StoryEngine, StoryId, StorySegment, MappedTask, TaskStatus,
};
use crate::tasks::code_review::CodeReviewTask;
use crate::tools::code_analysis::{CodeAnalyzer, IssueSeverity};
use crate::tools::github::{GitHubClient, PullRequestDetails};

/// Configuration for story-driven PR review
#[derive(Debug, Clone)]
pub struct StoryDrivenPrReviewConfig {
    /// Enable narrative context analysis
    pub enable_narrative_analysis: bool,

    /// Enable pattern-based review
    pub enable_pattern_review: bool,

    /// Enable story coherence checks
    pub enable_coherence_check: bool,

    /// Enable intelligent suggestions
    pub enable_suggestions: bool,

    /// Enable automated approval for low-risk PRs
    pub enable_auto_approval: bool,

    /// Minimum confidence for auto-approval
    pub auto_approval_threshold: f32,

    /// Review depth level
    pub review_depth: ReviewDepth,

    /// Repository path
    pub repo_path: PathBuf,
}

impl Default for StoryDrivenPrReviewConfig {
    fn default() -> Self {
        Self {
            enable_narrative_analysis: true,
            enable_pattern_review: true,
            enable_coherence_check: true,
            enable_suggestions: true,
            enable_auto_approval: false,
            auto_approval_threshold: 0.95,
            review_depth: ReviewDepth::Comprehensive,
            repo_path: PathBuf::from("."),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ReviewDepth {
    Quick,         // Basic checks only
    Standard,      // Normal review
    Comprehensive, // Deep analysis with all features
}

/// Story-driven PR review system
pub struct StoryDrivenPrReview {
    config: StoryDrivenPrReviewConfig,
    story_engine: Arc<StoryEngine>,
    pr_automation: Option<Arc<PrAutomationSystem>>,
    code_analyzer: Arc<CodeAnalyzer>,
    code_review_task: Arc<CodeReviewTask>,
    memory: Arc<CognitiveMemory>,
    github_client: Option<Arc<GitHubClient>>,

    /// Codebase story ID
    codebase_story_id: StoryId,

    /// Review patterns learned
    review_patterns: Arc<RwLock<HashMap<String, ReviewPattern>>>,

    /// Review history for learning
    review_history: Arc<RwLock<Vec<ReviewRecord>>>,
}

/// Learned review pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewPattern {
    pub pattern_id: String,
    pub pattern_type: ReviewPatternType,
    pub description: String,
    pub detection_rules: Vec<String>,
    pub severity: IssueSeverity,
    pub suggested_fix: Option<String>,
    pub occurrences: usize,
    pub false_positive_rate: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReviewPatternType {
    SecurityVulnerability,
    PerformanceIssue,
    CodeSmell,
    ArchitecturalViolation,
    TestingGap,
    DocumentationMissing,
    DependencyRisk,
}

/// Review record for learning
#[derive(Debug, Clone)]
struct ReviewRecord {
    pub pr_number: u32,
    pub review_result: StoryDrivenReviewResult,
    pub reviewer_feedback: Option<String>,
    pub was_accurate: Option<bool>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Enhanced review result with story context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoryDrivenReviewResult {
    pub pr_number: u32,
    pub overall_assessment: ReviewAssessment,
    pub confidence: f32,
    pub narrative_analysis: NarrativeAnalysis,
    pub pattern_matches: Vec<PatternMatch>,
    pub coherence_score: f32,
    pub suggestions: Vec<ReviewSuggestion>,
    pub risk_assessment: RiskAssessment,
    pub auto_approve_eligible: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReviewAssessment {
    Approve,
    ApproveWithSuggestions,
    RequestChanges,
    NeedsDiscussion,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NarrativeAnalysis {
    pub aligns_with_story: bool,
    pub story_context: String,
    pub narrative_fit_score: f32,
    pub potential_consequences: Vec<String>,
    pub future_implications: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternMatch {
    pub pattern: ReviewPattern,
    pub location: String,
    pub code_snippet: String,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewSuggestion {
    pub suggestion_type: SuggestionType,
    pub description: String,
    pub code_change: Option<String>,
    pub priority: f32,
    pub rationale: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SuggestionType {
    Improvement,
    Refactoring,
    TestAddition,
    DocumentationUpdate,
    SecurityFix,
    PerformanceOptimization,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAssessment {
    pub risk_level: String,
    pub risk_factors: Vec<String>,
    pub mitigation_suggestions: Vec<String>,
}

impl StoryDrivenPrReview {
    /// Create a new story-driven PR review system
    pub async fn new(
        config: StoryDrivenPrReviewConfig,
        story_engine: Arc<StoryEngine>,
        pr_automation: Option<Arc<PrAutomationSystem>>,
        memory: Arc<CognitiveMemory>,
        github_client: Option<Arc<GitHubClient>>,
    ) -> Result<Self> {
        info!("🔍 Initializing Story-Driven PR Review System");

        // Create or get codebase story
        let codebase_story_id = story_engine
            .create_codebase_story(
                config.repo_path.clone(),
                "rust".to_string()
            )
            .await?;

        // Initialize components
        let code_analyzer = Arc::new(CodeAnalyzer::new(memory.clone()).await?);
        let code_review_task = Arc::new(CodeReviewTask::new(None, Some(memory.clone())));

        // Load review patterns
        let patterns = Self::load_review_patterns(&memory).await?;

        // Record initialization in story
        story_engine
            .add_plot_point(
                codebase_story_id.clone(),
                PlotType::Goal {
                    objective: "Initialize intelligent PR review system".to_string(),
                },
                vec!["pr_review", "initialization"].iter().map(|s| s.to_string()).collect(), // context_tokens
            )
            .await?;

        Ok(Self {
            config,
            story_engine,
            pr_automation,
            code_analyzer,
            code_review_task,
            memory,
            github_client,
            codebase_story_id,
            review_patterns: Arc::new(RwLock::new(patterns)),
            review_history: Arc::new(RwLock::new(Vec::new())),
        })
    }

    /// Review a pull request with story context
    pub async fn review_pr(&self, pr_number: u32) -> Result<StoryDrivenReviewResult> {
        info!("📖 Reviewing PR #{} with story context", pr_number);

        // Get PR details
        let pr_details = self.fetch_pr_details(pr_number).await?;


        // Build review context from story
        let review_context = self.build_review_context(&pr_details, &[]).await?;

        // Perform narrative analysis
        let narrative_analysis = if self.config.enable_narrative_analysis {
            self.analyze_narrative_fit(&pr_details, &review_context).await?
        } else {
            NarrativeAnalysis {
                aligns_with_story: true,
                story_context: "Narrative analysis disabled".to_string(),
                narrative_fit_score: 1.0,
                potential_consequences: vec![],
                future_implications: vec![],
            }
        };

        // Check patterns
        let pattern_matches = if self.config.enable_pattern_review {
            self.check_patterns(&[]).await?
        } else {
            vec![]
        };

        // Check coherence
        let coherence_score = if self.config.enable_coherence_check {
            self.check_coherence(&pr_details, &[], &review_context).await?
        } else {
            1.0
        };

        // Generate suggestions
        let suggestions = if self.config.enable_suggestions {
            self.generate_suggestions(&pr_details, &pattern_matches, &narrative_analysis).await?
        } else {
            vec![]
        };

        // Assess risk
        let risk_assessment = self.assess_risk(&pr_details, &pattern_matches).await?;

        // Determine overall assessment
        let (overall_assessment, confidence) = self.determine_assessment(
            &narrative_analysis,
            &pattern_matches,
            coherence_score,
            &risk_assessment,
        );

        // Check auto-approval eligibility
        let auto_approve_eligible = self.config.enable_auto_approval
            && confidence >= self.config.auto_approval_threshold
            && matches!(overall_assessment, ReviewAssessment::Approve)
            && risk_assessment.risk_level == "low";

        let review_result = StoryDrivenReviewResult {
            pr_number,
            overall_assessment,
            confidence,
            narrative_analysis,
            pattern_matches,
            coherence_score,
            suggestions,
            risk_assessment,
            auto_approve_eligible,
        };

        // Store review in history
        let mut history = self.review_history.write().await;
        history.push(ReviewRecord {
            pr_number,
            review_result: review_result.clone(),
            reviewer_feedback: None,
            was_accurate: None,
            timestamp: Utc::now(),
        });

        // Record review in story
        self.story_engine
            .add_plot_point(
                self.codebase_story_id.clone(),
                PlotType::Discovery {
                    insight: format!(
                        "PR #{} reviewed: {:?} (confidence: {:.2})",
                        pr_number, review_result.overall_assessment, confidence
                    ),
                },
                vec!["pr_review".to_string(), format!("pr_{}", pr_number)], // context_tokens
            )
            .await?;

        // Post review comment if GitHub client available
        if let Some(github) = &self.github_client {
            self.post_review_comment(github, pr_number, &review_result).await?;
        }

        Ok(review_result)
    }

    /// Build review context from story
    async fn build_review_context(
        &self,
        pr_details: &PullRequestDetails,
        changed_files: &[ChangedFile],
    ) -> Result<ReviewContext> {
        // Get recent story segments
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

        // Extract relevant patterns
        let patterns = self.review_patterns.read().await;
        let relevant_patterns: Vec<ReviewPattern> = patterns
            .values()
            .filter(|p| {
                // Check if pattern applies to changed files
                changed_files.iter().any(|f| {
                    p.detection_rules.iter().any(|rule| {
                        f.filename.contains(rule) || f.patch.contains(rule)
                    })
                })
            })
            .cloned()
            .collect();

        // Get related tasks from story
        let tasks = self.story_engine.extract_all_tasks().await?;
        let related_tasks: Vec<MappedTask> = tasks
            .get(&self.codebase_story_id)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .filter(|t| {
                pr_details.title.contains(&t.description) ||
                pr_details.body.contains(&t.description)
            })
            .collect();

        // Load architectural rules from configuration or story context
        let mut architectural_rules = HashMap::new();

        // Default architectural rules
        architectural_rules.insert("error_handling".to_string(),
            "All Results must be properly handled with context".to_string());
        architectural_rules.insert("async_patterns".to_string(),
            "Use tokio for async runtime, avoid blocking in async contexts".to_string());
        architectural_rules.insert("memory_safety".to_string(),
            "Prefer Arc<RwLock<T>> for shared state, document unsafe code thoroughly".to_string());
        architectural_rules.insert("testing".to_string(),
            "All public APIs must have unit tests".to_string());

        // Extract custom rules from story segments
        for segment in &recent_segments {
            if segment.tags.contains(&"architecture".to_string()) ||
               segment.tags.contains(&"design_rule".to_string()) {
                // Extract rule from segment content
                if let Some(rule_name) = segment.content.lines()
                    .find(|l| l.starts_with("rule:"))
                    .and_then(|l| l.strip_prefix("rule:"))
                    .map(|s| s.trim().to_string()) {
                    architectural_rules.insert(rule_name, segment.content.clone());
                }
            }
        }

        Ok(ReviewContext {
            recent_story_segments: recent_segments,
            relevant_patterns: relevant_patterns,
            related_tasks: related_tasks,
            pr_metadata: PrMetadata {
                author: "unknown".to_string(), // Would get from GitHub API
                created_at: chrono::Utc::now(),
                labels: vec![],
                is_draft: false,
            },
            architectural_rules,
        })
    }

    /// Analyze narrative fit of the PR
    async fn analyze_narrative_fit(
        &self,
        pr_details: &PullRequestDetails,
        context: &ReviewContext,
    ) -> Result<NarrativeAnalysis> {
        let mut fit_score: f64 = 0.5; // Base score
        let mut consequences = Vec::new();
        let mut implications = Vec::new();

        // Check if PR aligns with recent story developments
        // Note: Plot points are stored in Story.arcs, segments provide context
        // Analyze based on segment content
        for segment in &context.recent_story_segments {
            // Check segment content for relevant keywords
            if segment.content.contains(&pr_details.title) ||
               pr_details.body.contains(&segment.content) {
                fit_score += 0.1;
                consequences.push(format!("Related to story segment: {}", segment.id));
            }

            // Check segment context for goal/task information
            if let Some(segment_type) = segment.context.get("type") {
                if segment_type == "goal" && segment.content.contains(&pr_details.title) {
                    fit_score += 0.2;
                    consequences.push(format!("Advances goal from segment"));
                } else if segment_type == "task" && pr_details.title.contains(&segment.content) {
                    fit_score += 0.15;
                    consequences.push(format!("Completes task from segment"));
                } else if segment_type == "issue" && pr_details.title.contains("fix") {
                    fit_score += 0.25;
                    consequences.push(format!("Fixes issue from segment"));
                }
            }
        }

        // Check for related tasks
        for task in &context.related_tasks {
            if task.status != TaskStatus::Completed {
                fit_score += 0.1;
                implications.push(format!("Progresses task: {}", task.description));
            }
        }

        // Determine story context
        let story_context = if !context.recent_story_segments.is_empty() {
            format!(
                "Recent narrative: {}",
                context.recent_story_segments[0].content.chars().take(200).collect::<String>()
            )
        } else {
            "No recent story context available".to_string()
        };

        fit_score = fit_score.min(1.0);

        Ok(NarrativeAnalysis {
            aligns_with_story: fit_score > 0.6,
            story_context,
            narrative_fit_score: fit_score as f32,
            potential_consequences: consequences,
            future_implications: implications,
        })
    }

    /// Check patterns in changed files
    async fn check_patterns(&self, changed_files: &[ChangedFile]) -> Result<Vec<PatternMatch>> {
        let mut matches = Vec::new();
        let patterns = self.review_patterns.read().await;

        for file in changed_files {
            for pattern in patterns.values() {
                for rule in &pattern.detection_rules {
                    if file.patch.contains(rule) {
                        // Find the specific location
                        let location = self.find_pattern_location(&file.patch, rule);

                        matches.push(PatternMatch {
                            pattern: pattern.clone(),
                            location: format!("{}:{}", file.filename, location),
                            code_snippet: self.extract_snippet(&file.patch, rule),
                            confidence: 0.85, // Base confidence, could be refined
                        });
                    }
                }
            }
        }

        Ok(matches)
    }

    /// Check coherence with existing codebase
    async fn check_coherence(
        &self,
        pr_details: &PullRequestDetails,
        changed_files: &[ChangedFile],
        context: &ReviewContext,
    ) -> Result<f32> {
        let mut coherence_score: f32 = 1.0;

        // Factor in PR size - larger PRs tend to have lower coherence
        let pr_size_penalty = match changed_files.len() {
            0..=5 => 0.0,
            6..=15 => 0.05,
            16..=30 => 0.1,
            _ => 0.15,
        };
        coherence_score -= pr_size_penalty;

        // Check if PR description aligns with changes
        if pr_details.body.len() < 50 {
            coherence_score -= 0.1; // Penalty for insufficient description
        }

        // Check naming conventions
        for file in changed_files {
            if !self.check_naming_conventions(&file.patch) {
                coherence_score -= 0.1;
            }
        }

        // Check architectural patterns
        if !self.check_architectural_patterns(changed_files, context) {
            coherence_score -= 0.2;
        }

        // Check test coverage
        if self.lacks_tests(changed_files) {
            coherence_score -= 0.15;
        }

        // Check documentation
        if self.lacks_documentation(changed_files) {
            coherence_score -= 0.1;
        }

        Ok(coherence_score.max(0.0f32))
    }

    /// Generate intelligent suggestions
    async fn generate_suggestions(
        &self,
        pr_details: &PullRequestDetails,
        pattern_matches: &[PatternMatch],
        narrative_analysis: &NarrativeAnalysis,
    ) -> Result<Vec<ReviewSuggestion>> {
        let mut suggestions = Vec::new();

        // Add suggestion for PR description if missing or inadequate
        if pr_details.body.len() < 100 {
            suggestions.push(ReviewSuggestion {
                suggestion_type: SuggestionType::DocumentationUpdate,
                description: "Consider adding a more detailed PR description explaining the motivation, approach, and impact of these changes.".to_string(),
                code_change: None,
                priority: 0.3,
                rationale: "A detailed PR description helps reviewers understand the context and purpose of the changes.".to_string(),
            });
        }

        // Check if PR title follows conventional commit format
        if !pr_details.title.contains(':') || pr_details.title.len() < 10 {
            suggestions.push(ReviewSuggestion {
                suggestion_type: SuggestionType::Improvement,
                description: format!("Consider using conventional commit format for the PR title. Example: 'feat: {}' or 'fix: {}'",
                    pr_details.title.to_lowercase(), pr_details.title.to_lowercase()),
                code_change: Some(format!("feat: {}", pr_details.title.to_lowercase())),
                priority: 0.2,
                rationale: "Conventional commit format improves commit history readability and enables automated tooling.".to_string(),
            });
        }

        // Suggestions based on pattern matches
        for pattern_match in pattern_matches {
            if let Some(fix) = &pattern_match.pattern.suggested_fix {
                suggestions.push(ReviewSuggestion {
                    suggestion_type: match pattern_match.pattern.pattern_type {
                        ReviewPatternType::SecurityVulnerability => SuggestionType::SecurityFix,
                        ReviewPatternType::PerformanceIssue => SuggestionType::PerformanceOptimization,
                        _ => SuggestionType::Improvement,
                    },
                    description: pattern_match.pattern.description.clone(),
                    code_change: Some(fix.clone()),
                    priority: match pattern_match.pattern.severity {
                        IssueSeverity::Error => 1.0,
                        IssueSeverity::Warning => 0.7,
                        IssueSeverity::Info => 0.4,
                        IssueSeverity::Hint => 0.2,
                    },
                    rationale: format!(
                        "Pattern '{}' detected at {}",
                        pattern_match.pattern.pattern_id,
                        pattern_match.location
                    ),
                });
            }
        }

        // Suggestions based on narrative analysis
        if narrative_analysis.narrative_fit_score < 0.7 {
            suggestions.push(ReviewSuggestion {
                suggestion_type: SuggestionType::DocumentationUpdate,
                description: "Consider updating PR description to better align with project narrative".to_string(),
                code_change: None,
                priority: 0.5,
                rationale: format!(
                    "Narrative fit score is {:.2}, which suggests the PR could better align with ongoing story",
                    narrative_analysis.narrative_fit_score
                ),
            });
        }

        // Sort by priority
        suggestions.sort_by(|a, b| b.priority.partial_cmp(&a.priority).unwrap());

        Ok(suggestions)
    }

    /// Assess risk of the PR
    async fn assess_risk(
        &self,
        pr_details: &PullRequestDetails,
        pattern_matches: &[PatternMatch],
    ) -> Result<RiskAssessment> {
        let mut risk_factors = Vec::new();
        let mut mitigation_suggestions = Vec::new();

        // Check for security patterns
        let security_issues = pattern_matches.iter()
            .filter(|m| matches!(m.pattern.pattern_type, ReviewPatternType::SecurityVulnerability))
            .count();

        if security_issues > 0 {
            risk_factors.push(format!("{} security vulnerabilities detected", security_issues));
            mitigation_suggestions.push("Address all security issues before merging".to_string());
        }

        // Check PR size (would get from GitHub API)
        // For now, use pattern matches as proxy
        if pattern_matches.len() > 10 {
            risk_factors.push("Large PR with many issues detected".to_string());
            mitigation_suggestions.push("Consider breaking into smaller PRs".to_string());
        }

        // Check for breaking changes
        if pr_details.title.contains("BREAKING") ||
           pr_details.body.contains("BREAKING") {
            risk_factors.push("Contains breaking changes".to_string());
            mitigation_suggestions.push("Ensure proper versioning and migration guide".to_string());
        }

        // Determine overall risk level
        let risk_level = if security_issues > 0 || risk_factors.len() > 2 {
            "high"
        } else if risk_factors.len() > 0 {
            "medium"
        } else {
            "low"
        }.to_string();

        Ok(RiskAssessment {
            risk_level,
            risk_factors,
            mitigation_suggestions,
        })
    }

    /// Determine overall assessment
    fn determine_assessment(
        &self,
        narrative_analysis: &NarrativeAnalysis,
        pattern_matches: &[PatternMatch],
        coherence_score: f32,
        risk_assessment: &RiskAssessment,
    ) -> (ReviewAssessment, f32) {
        let mut confidence = 0.5;

        // Factor in narrative fit
        confidence += narrative_analysis.narrative_fit_score * 0.2;

        // Factor in coherence
        confidence += coherence_score * 0.2;

        // Reduce confidence for pattern matches
        confidence -= (pattern_matches.len() as f32 * 0.05).min(0.3);

        // Determine assessment
        let assessment = if risk_assessment.risk_level == "high" {
            ReviewAssessment::RequestChanges
        } else if pattern_matches.iter().any(|m| matches!(m.pattern.severity, IssueSeverity::Error)) {
            ReviewAssessment::RequestChanges
        } else if !pattern_matches.is_empty() || coherence_score < 0.7 {
            ReviewAssessment::ApproveWithSuggestions
        } else if narrative_analysis.narrative_fit_score < 0.5 {
            ReviewAssessment::NeedsDiscussion
        } else {
            ReviewAssessment::Approve
        };

        (assessment, confidence.min(1.0).max(0.0))
    }

    /// Post review comment to GitHub
    async fn post_review_comment(
        &self,
        github: &GitHubClient,
        pr_number: u32,
        review_result: &StoryDrivenReviewResult,
    ) -> Result<()> {
        let mut comment_body = format!(
            "## 🤖 Story-Driven PR Review\n\n\
            **Assessment**: {:?}\n\
            **Confidence**: {:.2}\n\
            **Narrative Fit**: {:.2}\n\
            **Coherence Score**: {:.2}\n\n",
            review_result.overall_assessment,
            review_result.confidence,
            review_result.narrative_analysis.narrative_fit_score,
            review_result.coherence_score
        );

        // Add narrative analysis
        if review_result.narrative_analysis.aligns_with_story {
            comment_body.push_str("✅ **This PR aligns well with the project narrative**\n\n");
        } else {
            comment_body.push_str("⚠️ **This PR may not align with the current project narrative**\n\n");
        }

        // Add consequences
        if !review_result.narrative_analysis.potential_consequences.is_empty() {
            comment_body.push_str("### Potential Consequences\n");
            for consequence in &review_result.narrative_analysis.potential_consequences {
                comment_body.push_str(&format!("- {}\n", consequence));
            }
            comment_body.push_str("\n");
        }

        // Add pattern matches
        if !review_result.pattern_matches.is_empty() {
            comment_body.push_str("### Pattern Analysis\n");
            for pattern_match in &review_result.pattern_matches {
                comment_body.push_str(&format!(
                    "- **{}** at `{}` ({:?})\n",
                    pattern_match.pattern.description,
                    pattern_match.location,
                    pattern_match.pattern.severity
                ));
            }
            comment_body.push_str("\n");
        }

        // Add suggestions
        if !review_result.suggestions.is_empty() {
            comment_body.push_str("### Suggestions\n");
            for suggestion in &review_result.suggestions {
                comment_body.push_str(&format!(
                    "- **{:?}**: {}\n  _Rationale_: {}\n",
                    suggestion.suggestion_type,
                    suggestion.description,
                    suggestion.rationale
                ));
            }
            comment_body.push_str("\n");
        }

        // Add risk assessment
        comment_body.push_str(&format!(
            "### Risk Assessment: {}\n",
            review_result.risk_assessment.risk_level.to_uppercase()
        ));

        if !review_result.risk_assessment.risk_factors.is_empty() {
            comment_body.push_str("**Risk Factors**:\n");
            for factor in &review_result.risk_assessment.risk_factors {
                comment_body.push_str(&format!("- {}\n", factor));
            }
        }

        // Add footer
        comment_body.push_str("\n---\n_Generated by Loki's Story-Driven PR Review System_");

        // Post the comment
        github.comment_on_pr(pr_number, &comment_body).await?;

        Ok(())
    }

    /// Helper methods
    async fn fetch_pr_details(&self, pr_number: u32) -> Result<PullRequestDetails> {
        if let Some(github) = &self.github_client {
            github.get_pull_request(pr_number).await
        } else {
            Err(anyhow::anyhow!("GitHub client not available"))
        }
    }

    fn find_pattern_location(&self, patch: &str, pattern: &str) -> String {
        // Find line number where pattern occurs
        for (i, line) in patch.lines().enumerate() {
            if line.contains(pattern) {
                return format!("L{}", i + 1);
            }
        }
        "unknown".to_string()
    }

    fn extract_snippet(&self, patch: &str, pattern: &str) -> String {
        // Extract a few lines around the pattern
        let lines: Vec<&str> = patch.lines().collect();
        for (i, line) in lines.iter().enumerate() {
            if line.contains(pattern) {
                let start = i.saturating_sub(2);
                let end = (i + 3).min(lines.len());
                return lines[start..end].join("\n");
            }
        }
        pattern.to_string()
    }

    fn check_naming_conventions(&self, patch: &str) -> bool {
        // Check Rust naming conventions
        !patch.contains("CamelCase") || !patch.contains("mixedCase")
    }

    fn check_architectural_patterns(&self, files: &[ChangedFile], context: &ReviewContext) -> bool {
        // Check for common architectural violations
        let mut follows_patterns = true;

        for file in files {
            // Check module organization
            if file.filename.starts_with("src/") {
                // Ensure cognitive modules don't directly access low-level implementations
                if file.filename.contains("cognitive/") && file.patch.contains("use crate::memory::simd") {
                    follows_patterns = false;
                }

                // Ensure safety layer is used for external operations
                if !file.filename.contains("safety/") && file.patch.contains("Command::new") {
                    // Check if it's wrapped in safety validation
                    if !file.patch.contains("safety_validator") && !file.patch.contains("validate_action") {
                        follows_patterns = false;
                    }
                }
            }
        }

        // Check against architectural rules from context
        for (rule_name, rule_description) in &context.architectural_rules {
                match rule_name.as_str() {
                    "error_handling" => {
                        // Check for proper error handling
                        let has_unwrap = files.iter().any(|f|
                            f.patch.contains(".unwrap()")
                        );
                        let has_expect_without_context = files.iter().any(|f|
                            f.patch.contains(".expect(\"") && !f.patch.contains("context")
                        );
                        if has_unwrap || has_expect_without_context {
                            follows_patterns = false;
                        }
                    },
                    "async_patterns" => {
                        // Check for blocking calls in async contexts
                        let has_blocking_in_async = files.iter().any(|f|
                            f.patch.contains("async") && (f.patch.contains("std::thread::sleep") ||
                                                         f.patch.contains(".wait()") ||
                                                         f.patch.contains("std::fs::read"))
                        );
                        if has_blocking_in_async {
                            follows_patterns = false;
                        }
                    },
                    "testing" => {
                        // This is checked in lacks_tests method
                    },
                    _ => {
                        // Custom rules can be added here
                    }
                }
        }

        follows_patterns
    }

    fn lacks_tests(&self, files: &[ChangedFile]) -> bool {
        // Check if PR adds code without tests
        let has_src_changes = files.iter().any(|f| f.filename.starts_with("src/"));
        let has_test_changes = files.iter().any(|f|
            f.filename.contains("test") || f.filename.contains("tests/")
        );

        has_src_changes && !has_test_changes
    }

    fn lacks_documentation(&self, files: &[ChangedFile]) -> bool {
        // Check if PR adds public APIs without docs
        files.iter().any(|f|
            f.patch.contains("pub fn") && !f.patch.contains("///")
        )
    }

    /// Load review patterns from memory
    async fn load_review_patterns(memory: &CognitiveMemory) -> Result<HashMap<String, ReviewPattern>> {
        let mut patterns = HashMap::new();

        // Try to load custom patterns from memory
        if let Some(stored_pattern) = memory.retrieve_by_key("review_patterns_code_review").await.ok().flatten() {
            if let Ok(pattern_data) = serde_json::from_str::<HashMap<String, ReviewPattern>>(&stored_pattern.content) {
                patterns = pattern_data;
            }
        }

        // Add default patterns
        patterns.insert(
            "unsafe_unwrap".to_string(),
            ReviewPattern {
                pattern_id: "unsafe_unwrap".to_string(),
                pattern_type: ReviewPatternType::CodeSmell,
                description: "Unsafe use of unwrap()".to_string(),
                detection_rules: vec![".unwrap()".to_string()],
                severity: IssueSeverity::Warning,
                suggested_fix: Some("Consider using ? operator or proper error handling".to_string()),
                occurrences: 0,
                false_positive_rate: 0.1,
            },
        );

        patterns.insert(
            "missing_error_handling".to_string(),
            ReviewPattern {
                pattern_id: "missing_error_handling".to_string(),
                pattern_type: ReviewPatternType::CodeSmell,
                description: "Missing error handling".to_string(),
                detection_rules: vec!["todo!()".to_string(), "unimplemented!()".to_string()],
                severity: IssueSeverity::Warning,
                suggested_fix: Some("Implement proper error handling".to_string()),
                occurrences: 0,
                false_positive_rate: 0.05,
            },
        );

        Ok(patterns)
    }

    /// Learn from review feedback
    pub async fn learn_from_feedback(
        &self,
        pr_number: u32,
        feedback: &str,
        was_accurate: bool,
    ) -> Result<()> {
        let mut history = self.review_history.write().await;

        if let Some(record) = history.iter_mut().find(|r| r.pr_number == pr_number) {
            record.reviewer_feedback = Some(feedback.to_string());
            record.was_accurate = Some(was_accurate);

            // Update pattern accuracy if not accurate
            if !was_accurate {
                let mut patterns = self.review_patterns.write().await;
                for pattern_match in &record.review_result.pattern_matches {
                    if let Some(pattern) = patterns.get_mut(&pattern_match.pattern.pattern_id) {
                        pattern.false_positive_rate =
                            (pattern.false_positive_rate * pattern.occurrences as f32 + 1.0) /
                            (pattern.occurrences + 1) as f32;
                        pattern.occurrences += 1;
                    }
                }
            }

            // Store learning in memory
            self.memory
                .store(
                    format!(
                        "PR review feedback for #{}: {} (accurate: {})",
                        pr_number, feedback, was_accurate
                    ),
                    vec![],
                    MemoryMetadata {
                        source: "story_driven_pr_review".to_string(),
                        tags: vec!["learning".to_string(), "pr_review".to_string()],
                        timestamp: chrono::Utc::now(),
                        expiration: None,
                        importance: if was_accurate { 0.7 } else { 0.9 },
                        associations: vec![],
                        context: Some("PR review learning".to_string()),
                        created_at: chrono::Utc::now(),
                        accessed_count: 0,
                        last_accessed: None,
                        version: 1,
                        category: "pr_review".to_string(),
                    },
                )
                .await?;
        }

        Ok(())
    }
}

/// Supporting structures
#[derive(Debug, Clone)]
struct ReviewContext {
    recent_story_segments: Vec<StorySegment>,
    relevant_patterns: Vec<ReviewPattern>,
    related_tasks: Vec<MappedTask>,
    pr_metadata: PrMetadata,
    architectural_rules: HashMap<String, String>,  // Custom architectural rules metadata
}

#[derive(Debug, Clone)]
struct PrMetadata {
    author: String,
    created_at: chrono::DateTime<chrono::Utc>,
    labels: Vec<String>,
    is_draft: bool,
}

#[derive(Debug, Clone)]
pub struct ChangedFile {
    pub filename: String,
    pub patch: String,
    pub additions: usize,
    pub deletions: usize,
}
