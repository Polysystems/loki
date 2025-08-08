use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use rayon::prelude::*;
use regex::Regex;
use serde_json::json;
use tokio::fs;
use tracing::{info, warn};

use crate::memory::CognitiveMemory;
use crate::models::{InferenceEngine, InferenceRequest};
use crate::tasks::{Task, TaskArgs, TaskContext, TaskResult};

/// Code review patterns and analysis
#[derive(Debug, Clone)]
pub struct CodeAnalysis {
    pub complexity_score: f32,
    pub maintainability_score: f32,
    pub security_issues: Vec<SecurityIssue>,
    pub performance_issues: Vec<PerformanceIssue>,
    pub style_issues: Vec<StyleIssue>,
    pub cognitive_patterns: Vec<String>,
    pub suggestions: Vec<ReviewSuggestion>,

    // General issues field being accessed in the codebase
    pub issues: Vec<String>,
}

/// Security vulnerability detection
#[derive(Debug, Clone)]
pub struct SecurityIssue {
    pub severity: SecuritySeverity,
    pub description: String,
    #[allow(dead_code)]
    pub line_number: Option<usize>,
    pub suggestion: String,
}

/// Performance optimization opportunities
#[derive(Debug, Clone)]
pub struct PerformanceIssue {
    pub issue_type: PerformanceIssueType,
    pub description: String,
    #[allow(dead_code)]
    pub line_number: Option<usize>,
    pub impact: PerformanceImpact,
    pub suggestion: String,
}

/// Code style and convention issues
#[derive(Debug, Clone)]
pub struct StyleIssue {
    #[allow(dead_code)]
    pub issue_type: StyleIssueType,
    #[allow(dead_code)]
    pub description: String,
    #[allow(dead_code)]
    pub line_number: Option<usize>,
    #[allow(dead_code)]
    pub severity: StyleSeverity,
}

/// Review suggestions with confidence scores
#[derive(Debug, Clone)]
pub struct ReviewSuggestion {
    pub category: SuggestionCategory,
    pub description: String,
    pub confidence: f32,
    pub priority: Priority,
    #[allow(dead_code)]
    pub code_example: Option<String>,
}

/// Security severity levels
#[derive(Debug, Clone)]
pub enum SecuritySeverity {
    Critical,
    High,
    Medium,
    Low,
    Info,
}

/// Performance issue types
#[derive(Debug, Clone)]
pub enum PerformanceIssueType {
    AlgorithmicComplexity,
    MemoryUsage,
    Concurrency,
    DatabaseQuery,
    NetworkCall,
    CacheUtilization,
}

/// Performance impact assessment
#[derive(Debug, Clone)]
pub enum PerformanceImpact {
    Critical,
    High,
    Medium,
    Low,
}

/// Style issue categories
#[derive(Debug, Clone)]
pub enum StyleIssueType {
    Naming,
    Documentation,
    Formatting,
    Organization,
    Comments,
}

/// Style severity levels
#[derive(Debug, Clone)]
pub enum StyleSeverity {
    Error,
    Warning,
    Info,
}

/// Suggestion categories
#[derive(Debug, Clone)]
pub enum SuggestionCategory {
    Architecture,
    Security,
    Performance,
    Maintainability,
    Testing,
    Documentation,
    ErrorHandling,
    Concurrency,
}

/// Review priority levels
#[derive(Debug, Clone)]
pub enum Priority {
    Critical,
    High,
    Medium,
    Low,
}

/// Enhanced code review task with cognitive analysis
pub struct CodeReviewTask {
    inference_engine: Option<Arc<dyn InferenceEngine>>,
    memory: Option<Arc<CognitiveMemory>>,
}

impl CodeReviewTask {
    /// Create a new code review task
    pub fn new(
        inference_engine: Option<Arc<dyn InferenceEngine>>,
        memory: Option<Arc<CognitiveMemory>>,
    ) -> Self {
        Self { inference_engine, memory }
    }

    /// Perform comprehensive code analysis
    async fn analyze_code(&self, file_path: &Path, content: &str) -> Result<CodeAnalysis> {
        
        // Parallel analysis using structured concurrency
        let (complexity_tx, complexity_rx) = tokio::sync::oneshot::channel();
        let (security_tx, security_rx) = tokio::sync::oneshot::channel();
        let (performance_tx, performance_rx) = tokio::sync::oneshot::channel();
        let (style_tx, style_rx) = tokio::sync::oneshot::channel();
        let (cognitive_tx, cognitive_rx) = tokio::sync::oneshot::channel();

        // Complexity analysis
        let content_for_complexity = content.to_string();
        let complexity_task = tokio::spawn(async move {
            let complexity = Self::analyze_complexity(&content_for_complexity);
            let maintainability = Self::calculate_maintainability(&content_for_complexity);
            let _ = complexity_tx.send((complexity, maintainability));
        });

        // Security analysis
        let content_for_security = content.to_string();
        let security_task = tokio::spawn(async move {
            let security_issues = Self::detect_security_issues(&content_for_security);
            let _ = security_tx.send(security_issues);
        });

        // Performance analysis
        let content_for_performance = content.to_string();
        let performance_task = tokio::spawn(async move {
            let performance_issues = Self::analyze_performance(&content_for_performance);
            let _ = performance_tx.send(performance_issues);
        });

        // Style analysis
        let content_for_style = content.to_string();
        let file_extension = file_path
            .extension()
            .and_then(|s| s.to_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| "".to_string());
        let style_task = tokio::spawn(async move {
            let style_issues = Self::check_style(&content_for_style, &file_extension);
            let _ = style_tx.send(style_issues);
        });

        // Cognitive pattern analysis
        let content_for_cognitive = content.to_string();
        let cognitive_task = tokio::spawn(async move {
            let patterns = Self::identify_cognitive_patterns(&content_for_cognitive);
            let _ = cognitive_tx.send(patterns);
        });

        // Wait for all analyses to complete with timeout
        let timeout_duration = std::time::Duration::from_secs(30);

        let (complexity_score, maintainability_score) =
            tokio::time::timeout(timeout_duration, complexity_rx)
                .await
                .unwrap_or(Ok((0.5, 0.5)))
                .unwrap_or((0.5, 0.5));
        let security_issues = tokio::time::timeout(timeout_duration, security_rx)
            .await
            .unwrap_or(Ok(Vec::new()))
            .unwrap_or_default();
        let performance_issues = tokio::time::timeout(timeout_duration, performance_rx)
            .await
            .unwrap_or(Ok(Vec::new()))
            .unwrap_or_default();
        let style_issues = tokio::time::timeout(timeout_duration, style_rx)
            .await
            .unwrap_or(Ok(Vec::new()))
            .unwrap_or_default();
        let cognitive_patterns = tokio::time::timeout(timeout_duration, cognitive_rx)
            .await
            .unwrap_or(Ok(Vec::new()))
            .unwrap_or_default();

        // Cleanup tasks
        complexity_task.abort();
        security_task.abort();
        performance_task.abort();
        style_task.abort();
        cognitive_task.abort();

        // Generate AI-powered suggestions if model is available
        let suggestions = if let Some(ref engine) = self.inference_engine {
            self.generate_ai_suggestions(
                engine,
                file_path,
                content,
                &security_issues,
                &performance_issues,
            )
            .await
            .unwrap_or_else(|e| {
                warn!("Failed to generate AI suggestions: {}", e);
                Vec::new()
            })
        } else {
            self.generate_rule_based_suggestions(
                &security_issues,
                &performance_issues,
                &style_issues,
            )
        };

        Ok(CodeAnalysis {
            complexity_score,
            maintainability_score,
            security_issues,
            performance_issues,
            style_issues,
            cognitive_patterns,
            suggestions,
            issues: Vec::new(), // Initialize empty issues list
        })
    }

    /// Analyze cyclomatic complexity
    fn analyze_complexity(content: &str) -> f32 {
        // Simple complexity analysis based on control flow statements
        let complexity_patterns = [
            r"\bif\b",
            r"\belse\b",
            r"\bwhile\b",
            r"\bfor\b",
            r"\bmatch\b",
            r"\bloop\b",
            r"\?\?",
            r"\?",
            r"&&",
            r"\|\|",
            r"\bcase\b",
        ];

        let total_complexity: usize = complexity_patterns
            .par_iter()
            .map(|pattern| Regex::new(pattern).map(|re| re.find_iter(content).count()).unwrap_or(0))
            .sum();

        // Normalize complexity score (0.0 to 1.0)
        let lines = content.lines().count().max(1);
        (total_complexity as f32 / lines as f32).min(1.0)
    }

    /// Calculate maintainability index
    fn calculate_maintainability(content: &str) -> f32 {
        let lines = content.lines().count() as f32;
        let comment_lines = content
            .lines()
            .filter(|line| line.trim().starts_with("//") || line.trim().starts_with("/*"))
            .count() as f32;

        let comment_ratio = if lines > 0.0 { comment_lines / lines } else { 0.0 };
        let avg_line_length =
            content.lines().map(|line| line.len()).sum::<usize>() as f32 / lines.max(1.0);

        // Simplified maintainability calculation
        let base_score = 1.0 - (avg_line_length / 200.0).min(0.5);
        let comment_bonus = comment_ratio * 0.3;
        let size_penalty = (lines / 1000.0).min(0.3);

        (base_score + comment_bonus - size_penalty).max(0.0).min(1.0)
    }

    /// Detect security vulnerabilities
    fn detect_security_issues(content: &str) -> Vec<SecurityIssue> {
        let mut issues = Vec::new();

        // SQL injection patterns
        if Regex::new(r#"format!\s*\(\s*"[^"]*\{\}[^"]*"\s*,.*\)"#).unwrap().is_match(content) {
            issues.push(SecurityIssue {
                severity: SecuritySeverity::High,
                description: "Potential SQL injection vulnerability in string formatting"
                    .to_string(),
                line_number: None,
                suggestion: "Use parameterized queries or prepared statements".to_string(),
            });
        }

        // Unsafe code blocks
        if content.contains("unsafe {") {
            issues.push(SecurityIssue {
                severity: SecuritySeverity::Medium,
                description: "Unsafe code block detected".to_string(),
                line_number: None,
                suggestion: "Review unsafe code for memory safety guarantees".to_string(),
            });
        }

        // Unwrap usage that could panic
        if Regex::new(r"\.unwrap\(\)").unwrap().find_iter(content).count() > 3 {
            issues.push(SecurityIssue {
                severity: SecuritySeverity::Medium,
                description: "Excessive use of unwrap() which can cause panics".to_string(),
                line_number: None,
                suggestion: "Use proper error handling with Result<T, E> or Option<T>".to_string(),
            });
        }

        // Hardcoded secrets patterns
        let secret_patterns = [
            r#"password\s*=\s*['"][^'"]+['"]"#,
            r#"api_key\s*=\s*['"][^'"]+['"]"#,
            r#"secret\s*=\s*['"][^'"]+['"]"#,
            r#"token\s*=\s*['"][^'"]+['"]"#,
        ];

        for pattern in &secret_patterns {
            if Regex::new(pattern).unwrap().is_match(content) {
                issues.push(SecurityIssue {
                    severity: SecuritySeverity::Critical,
                    description: "Hardcoded secrets detected".to_string(),
                    line_number: None,
                    suggestion: "Use environment variables or secure credential storage"
                        .to_string(),
                });
                break; // Only report once per file
            }
        }

        issues
    }

    /// Analyze performance issues
    fn analyze_performance(content: &str) -> Vec<PerformanceIssue> {
        let mut issues = Vec::new();

        // Clone in loop detection
        if Regex::new(r"for\s+.*\{[^}]*\.clone\(\)").unwrap().is_match(content) {
            issues.push(PerformanceIssue {
                issue_type: PerformanceIssueType::MemoryUsage,
                description: "Potential inefficient cloning in loop".to_string(),
                line_number: None,
                impact: PerformanceImpact::Medium,
                suggestion: "Consider borrowing instead of cloning, or move clone outside loop"
                    .to_string(),
            });
        }

        // Nested loops detection
        let nested_loop_pattern = r"for\s+[^{]*\{[^}]*for\s+[^{]*\{";
        if Regex::new(nested_loop_pattern).unwrap().is_match(content) {
            issues.push(PerformanceIssue {
                issue_type: PerformanceIssueType::AlgorithmicComplexity,
                description: "Nested loops detected - potential O(n²) complexity".to_string(),
                line_number: None,
                impact: PerformanceImpact::High,
                suggestion: "Consider using hash maps, iterators, or parallel processing"
                    .to_string(),
            });
        }

        // Blocking operations in async context
        if content.contains("async") && content.contains("std::thread::sleep") {
            issues.push(PerformanceIssue {
                issue_type: PerformanceIssueType::Concurrency,
                description: "Blocking operation in async function".to_string(),
                line_number: None,
                impact: PerformanceImpact::High,
                suggestion: "Use tokio::time::sleep instead of std::thread::sleep".to_string(),
            });
        }

        // Large string concatenation
        if Regex::new(r"string\s*\+=").unwrap().find_iter(content).count() > 5 {
            issues.push(PerformanceIssue {
                issue_type: PerformanceIssueType::MemoryUsage,
                description: "Multiple string concatenations detected".to_string(),
                line_number: None,
                impact: PerformanceImpact::Medium,
                suggestion: "Use format! macro or String::with_capacity for better performance"
                    .to_string(),
            });
        }

        issues
    }

    /// Check code style and conventions
    fn check_style(content: &str, file_extension: &str) -> Vec<StyleIssue> {
        let mut issues = Vec::new();

        // Language-specific style checks
        match file_extension {
            "rs" => {
                // Rust-specific style checks
                if !Regex::new(r"^//!|^///").unwrap().is_match(content) && content.len() > 1000 {
                    issues.push(StyleIssue {
                        issue_type: StyleIssueType::Documentation,
                        description: "Missing module or crate-level documentation".to_string(),
                        line_number: None,
                        severity: StyleSeverity::Warning,
                    });
                }

                // Check for proper function naming (snake_case)
                if Regex::new(r"\bfn\s+[A-Z]").unwrap().is_match(content) {
                    issues.push(StyleIssue {
                        issue_type: StyleIssueType::Naming,
                        description: "Function names should use snake_case".to_string(),
                        line_number: None,
                        severity: StyleSeverity::Error,
                    });
                }
            }
            _ => {
                // Generic style checks
                if content.lines().any(|line| line.len() > 120) {
                    issues.push(StyleIssue {
                        issue_type: StyleIssueType::Formatting,
                        description: "Lines exceed 120 character limit".to_string(),
                        line_number: None,
                        severity: StyleSeverity::Warning,
                    });
                }
            }
        }

        // Generic issues
        if content.lines().filter(|line| line.trim().is_empty()).count()
            > content.lines().count() / 3
        {
            issues.push(StyleIssue {
                issue_type: StyleIssueType::Formatting,
                description: "Excessive blank lines detected".to_string(),
                line_number: None,
                severity: StyleSeverity::Info,
            });
        }

        issues
    }

    /// Identify cognitive patterns in code
    fn identify_cognitive_patterns(content: &str) -> Vec<String> {
        let mut patterns = Vec::new();

        // Pattern matching usage
        if content.contains("match ") {
            patterns.push(
                "Pattern Matching: Code uses Rust's pattern matching for control flow".to_string(),
            );
        }

        // Error handling patterns
        if content.contains("Result<") && content.contains("?") {
            patterns.push("Error Propagation: Code uses idiomatic Rust error handling".to_string());
        }

        // Iterator patterns
        if content.contains(".iter()")
            && (content.contains(".map(") || content.contains(".filter("))
        {
            patterns.push(
                "Functional Programming: Code uses iterator chains for data processing".to_string(),
            );
        }

        // Ownership patterns
        if content.contains("&mut ") && content.contains("&") {
            patterns.push(
                "Ownership Management: Code demonstrates Rust's borrowing system".to_string(),
            );
        }

        // Concurrency patterns
        if content.contains("Arc<") && content.contains("Mutex<") {
            patterns.push(
                "Concurrency Safety: Code uses thread-safe shared state patterns".to_string(),
            );
        }

        // Async patterns
        if content.contains("async ") && content.contains(".await") {
            patterns.push(
                "Asynchronous Programming: Code uses async/await for non-blocking operations"
                    .to_string(),
            );
        }

        patterns
    }

    /// Generate AI-powered suggestions using the inference engine
    async fn generate_ai_suggestions(
        &self,
        engine: &Arc<dyn InferenceEngine>,
        file_path: &Path,
        content: &str,
        security_issues: &[SecurityIssue],
        performance_issues: &[PerformanceIssue],
    ) -> Result<Vec<ReviewSuggestion>> {
        let analysis_summary = format!(
            "File: {:?}\nSecurity Issues: {}\nPerformance Issues: {}\nCode Length: {} lines",
            file_path,
            security_issues.len(),
            performance_issues.len(),
            content.lines().count()
        );

        let prompt = format!(
            "As an expert code reviewer, analyze this code and provide specific, actionable \
             suggestions:\n\n{}\n\nCode snippet (first 500 chars):\n{}\n\nFocus on architecture, \
             maintainability, and Rust best practices.",
            analysis_summary,
            content.chars().take(500).collect::<String>()
        );

        let request = InferenceRequest {
            prompt,
            max_tokens: 512,
            temperature: 0.3, // Lower temperature for more focused suggestions
            top_p: 0.9,
            stop_sequences: vec![],
        };

        let response = engine.infer(request).await?;

        // Parse AI response into structured suggestions
        let suggestions = self.parse_ai_suggestions(&response.text);

        Ok(suggestions)
    }

    /// Parse AI response into structured suggestions
    fn parse_ai_suggestions(&self, ai_response: &str) -> Vec<ReviewSuggestion> {
        let mut suggestions = Vec::new();

        // Simple parsing - in a real implementation, this would be more sophisticated
        let lines: Vec<&str> = ai_response.lines().collect();

        for line in lines {
            if line.contains("Architecture:") {
                suggestions.push(ReviewSuggestion {
                    category: SuggestionCategory::Architecture,
                    description: line.to_string(),
                    confidence: 0.8,
                    priority: Priority::Medium,
                    code_example: None,
                });
            } else if line.contains("Performance:") {
                suggestions.push(ReviewSuggestion {
                    category: SuggestionCategory::Performance,
                    description: line.to_string(),
                    confidence: 0.7,
                    priority: Priority::High,
                    code_example: None,
                });
            } else if line.contains("Security:") {
                suggestions.push(ReviewSuggestion {
                    category: SuggestionCategory::Security,
                    description: line.to_string(),
                    confidence: 0.9,
                    priority: Priority::Critical,
                    code_example: None,
                });
            }
        }

        // Add a general AI insight
        suggestions.push(ReviewSuggestion {
            category: SuggestionCategory::Maintainability,
            description: format!(
                "AI Analysis: {}",
                ai_response.chars().take(200).collect::<String>()
            ),
            confidence: 0.6,
            priority: Priority::Medium,
            code_example: None,
        });

        suggestions
    }

    /// Generate rule-based suggestions as fallback
    fn generate_rule_based_suggestions(
        &self,
        security_issues: &[SecurityIssue],
        performance_issues: &[PerformanceIssue],
        style_issues: &[StyleIssue],
    ) -> Vec<ReviewSuggestion> {
        let mut suggestions = Vec::new();

        // Convert issues to suggestions
        for issue in security_issues {
            suggestions.push(ReviewSuggestion {
                category: SuggestionCategory::Security,
                description: format!("Security: {}", issue.description),
                confidence: 0.9,
                priority: match issue.severity {
                    SecuritySeverity::Critical => Priority::Critical,
                    SecuritySeverity::High => Priority::High,
                    SecuritySeverity::Medium => Priority::Medium,
                    _ => Priority::Low,
                },
                code_example: None,
            });
        }

        for issue in performance_issues {
            suggestions.push(ReviewSuggestion {
                category: SuggestionCategory::Performance,
                description: format!("Performance: {}", issue.description),
                confidence: 0.8,
                priority: match issue.impact {
                    PerformanceImpact::Critical => Priority::Critical,
                    PerformanceImpact::High => Priority::High,
                    PerformanceImpact::Medium => Priority::Medium,
                    PerformanceImpact::Low => Priority::Low,
                },
                code_example: None,
            });
        }

        if !style_issues.is_empty() {
            suggestions.push(ReviewSuggestion {
                category: SuggestionCategory::Maintainability,
                description: format!("Style: {} style issues detected", style_issues.len()),
                confidence: 0.7,
                priority: Priority::Low,
                code_example: None,
            });
        }

        // Add general suggestions
        suggestions.push(ReviewSuggestion {
            category: SuggestionCategory::Testing,
            description: "Consider adding unit tests if not present".to_string(),
            confidence: 0.6,
            priority: Priority::Medium,
            code_example: None,
        });

        suggestions.push(ReviewSuggestion {
            category: SuggestionCategory::Documentation,
            description: "Ensure all public functions have documentation".to_string(),
            confidence: 0.7,
            priority: Priority::Medium,
            code_example: None,
        });

        suggestions
    }

    /// Store review results in cognitive memory
    async fn store_review_results(&self, file_path: &Path, analysis: &CodeAnalysis) -> Result<()> {
        if let Some(ref memory) = self.memory {
            let memory_content = format!(
                "Code review completed for {:?}. Complexity: {:.2}, Maintainability: {:.2}, \
                 Security issues: {}, Performance issues: {}, Suggestions: {}",
                file_path,
                analysis.complexity_score,
                analysis.maintainability_score,
                analysis.security_issues.len(),
                analysis.performance_issues.len(),
                analysis.suggestions.len()
            );

            memory
                .store(
                    memory_content,
                    vec!["code_review".to_string(), file_path.to_string_lossy().to_string()],
                    crate::memory::MemoryMetadata {
                        source: "code_review_task".to_string(),
                        tags: vec![
                            "code_review".to_string(),
                            "analysis".to_string(),
                            "quality".to_string(),
                        ],
                        importance: 0.7,
                        associations: vec![],
                        context: Some(format!(
                            "Code review for file: {}",
                            file_path.to_string_lossy()
                        )),
                        timestamp: chrono::Utc::now(),
                        expiration: None,
                        created_at: chrono::Utc::now(),
                        accessed_count: 0,
                        last_accessed: None,
                        version: 1,
                        category: "code_review".to_string(),
                    },
                )
                .await?;
        }
        Ok(())
    }
}

#[async_trait]
impl Task for CodeReviewTask {
    fn name(&self) -> &str {
        "review"
    }

    fn description(&self) -> &str {
        "Perform comprehensive code review with AI-powered analysis and cognitive pattern \
         recognition"
    }

    async fn execute(&self, args: TaskArgs, _context: TaskContext) -> Result<TaskResult> {
        let input_path =
            args.input.ok_or_else(|| anyhow::anyhow!("Input path required for code review"))?;

        info!("🔍 Performing comprehensive code review for: {:?}", input_path);

        // Read file content
        let content = fs::read_to_string(&input_path).await.context("Failed to read input file")?;

        // Perform comprehensive analysis
        let analysis = self
            .analyze_code(Path::new(&input_path), &content)
            .await
            .context("Code analysis failed")?;

        // Store results in memory
        if let Err(e) = self.store_review_results(Path::new(&input_path), &analysis).await {
            warn!("Failed to store review results in memory: {}", e);
        }

        // Format results as JSON
        let review_data = json!({
            "reviewed_path": input_path,
            "analysis": {
                "complexity_score": analysis.complexity_score,
                "maintainability_score": analysis.maintainability_score,
                "security_issues_count": analysis.security_issues.len(),
                "performance_issues_count": analysis.performance_issues.len(),
                "style_issues_count": analysis.style_issues.len(),
                "cognitive_patterns": analysis.cognitive_patterns
            },
            "suggestions": analysis.suggestions.iter().map(|s| json!({
                "category": format!("{:?}", s.category),
                "description": s.description,
                "confidence": s.confidence,
                "priority": format!("{:?}", s.priority)
            })).collect::<Vec<_>>(),
            "detailed_issues": {
                "security": analysis.security_issues.iter().map(|issue| json!({
                    "severity": format!("{:?}", issue.severity),
                    "description": issue.description,
                    "suggestion": issue.suggestion
                })).collect::<Vec<_>>(),
                "performance": analysis.performance_issues.iter().map(|issue| json!({
                    "type": format!("{:?}", issue.issue_type),
                    "description": issue.description,
                    "impact": format!("{:?}", issue.impact),
                    "suggestion": issue.suggestion
                })).collect::<Vec<_>>()
            }
        });

        info!("✅ Code review completed successfully");

        Ok(TaskResult {
            success: true,
            message: format!(
                "Code review completed. Complexity: {:.2}, {} security issues, {} performance \
                 issues, {} suggestions",
                analysis.complexity_score,
                analysis.security_issues.len(),
                analysis.performance_issues.len(),
                analysis.suggestions.len()
            ),
            data: Some(review_data),
        })
    }
}
