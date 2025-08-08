use std::path::Path;

use anyhow::Result;
use async_trait::async_trait;
use rayon::prelude::*;
use serde_json::json;
use tokio::fs;
use tracing::{debug, info};

use crate::tasks::{Task, TaskArgs, TaskContext, TaskResult};

/// Refactoring suggestions task for code quality improvements
pub struct RefactoringTask;

#[async_trait]
impl Task for RefactoringTask {
    fn name(&self) -> &str {
        "refactor"
    }

    fn description(&self) -> &str {
        "Analyze code and suggest refactoring improvements for better maintainability, \
         performance, and Rust idioms"
    }

    async fn execute(&self, args: TaskArgs, _context: TaskContext) -> Result<TaskResult> {
        let input_path = args
            .input
            .ok_or_else(|| anyhow::anyhow!("Input path required for refactoring suggestions"))?;

        info!("Analyzing code for refactoring suggestions: {:?}", input_path);

        // Analyze code structure and patterns
        let code_analysis = self.analyze_code_quality(&input_path).await?;

        // Generate refactoring suggestions
        let suggestions = self.generate_refactoring_suggestions(&code_analysis).await?;

        // Rank suggestions by impact and difficulty
        let ranked_suggestions = self.rank_suggestions(suggestions);

        info!("Generated {} refactoring suggestions", ranked_suggestions.len());

        Ok(TaskResult {
            success: true,
            message: format!("Generated {} refactoring suggestions", ranked_suggestions.len()),
            data: Some(json!({
                "source_path": input_path,
                "suggestions": ranked_suggestions.iter().map(|s| json!({
                    "type": s.suggestion_type,
                    "priority": s.priority,
                    "line": s.line_number,
                    "description": s.description,
                    "rationale": s.rationale,
                    "difficulty": s.difficulty,
                    "impact": s.impact,
                    "code_before": s.code_before,
                    "code_after": s.code_after,
                })).collect::<Vec<_>>(),
                "analysis_summary": {
                    "total_lines": code_analysis.total_lines,
                    "functions_analyzed": code_analysis.functions.len(),
                    "complexity_score": code_analysis.avg_complexity,
                    "issues_found": code_analysis.code_issues.len(),
                }
            })),
        })
    }
}

impl RefactoringTask {
    /// Analyze code quality and structure
    async fn analyze_code_quality(&self, path: &Path) -> Result<CodeAnalysis> {
        debug!("Analyzing code quality for: {:?}", path);

        let content = fs::read_to_string(path).await?;
        let lines: Vec<&str> = content.lines().collect();

        let mut analysis = CodeAnalysis {
            total_lines: lines.len(),
            functions: Vec::new(),
            code_issues: Vec::new(),
            avg_complexity: 0.0,
        };

        // Analyze functions in parallel
        let function_analyses: Vec<_> = lines
            .par_iter()
            .enumerate()
            .filter_map(|(line_num, line)| self.analyze_function_line(line, line_num, &lines))
            .collect();

        analysis.functions = function_analyses;

        // Detect code issues in parallel
        let code_issues: Vec<_> = lines
            .par_iter()
            .enumerate()
            .flat_map(|(line_num, line)| self.detect_code_issues(line, line_num))
            .collect();

        analysis.code_issues = code_issues;

        // Calculate average complexity
        if !analysis.functions.is_empty() {
            analysis.avg_complexity =
                analysis.functions.iter().map(|f| f.complexity_score).sum::<f64>()
                    / analysis.functions.len() as f64;
        }

        Ok(analysis)
    }

    /// Analyze individual function
    fn analyze_function_line(
        &self,
        line: &str,
        line_num: usize,
        all_lines: &[&str],
    ) -> Option<FunctionAnalysis> {
        let trimmed = line.trim();

        if trimmed.contains("fn ") && !trimmed.contains("//") {
            let name = self.extract_function_name(trimmed)?;
            let complexity = self.calculate_function_complexity(line_num, all_lines);
            let line_count = self.count_function_lines(line_num, all_lines);

            Some(FunctionAnalysis {
                name,
                line_number: line_num,
                complexity_score: complexity,
                line_count,
                #[allow(dead_code)]
                has_error_handling: self.has_error_handling(line_num, all_lines),
                parameter_count: self.count_parameters(trimmed),
                #[allow(dead_code)]
                return_type: self.extract_return_type(trimmed),
            })
        } else {
            None
        }
    }

    /// Extract function name from line
    fn extract_function_name(&self, line: &str) -> Option<String> {
        if let Some(fn_start) = line.find("fn ") {
            let after_fn = &line[fn_start + 3..];
            if let Some(paren_pos) = after_fn.find('(') {
                return Some(after_fn[..paren_pos].trim().to_string());
            }
        }
        None
    }

    /// Calculate function complexity (simplified cyclomatic complexity)
    fn calculate_function_complexity(&self, start_line: usize, lines: &[&str]) -> f64 {
        let mut complexity = 1.0; // Base complexity
        let function_end = self.find_function_end(start_line, lines);

        for line in &lines[start_line..function_end.min(lines.len())] {
            let trimmed = line.trim();

            // Control flow increases complexity
            if trimmed.contains("if ") || trimmed.contains("else if ") {
                complexity += 1.0;
            }
            if trimmed.contains("match ") {
                complexity += 1.0;
            }
            if trimmed.contains("for ") || trimmed.contains("while ") {
                complexity += 1.0;
            }
            if trimmed.contains("loop ") {
                complexity += 1.0;
            }
            // Nested complexity
            if trimmed.starts_with("        ") {
                // Deep nesting
                complexity += 0.5;
            }
        }

        complexity
    }

    /// Find function end line
    fn find_function_end(&self, start_line: usize, lines: &[&str]) -> usize {
        let mut brace_count = 0;
        let mut found_opening = false;

        for (i, line) in lines[start_line..].iter().enumerate() {
            for ch in line.chars() {
                match ch {
                    '{' => {
                        brace_count += 1;
                        found_opening = true;
                    }
                    '}' => {
                        brace_count -= 1;
                        if found_opening && brace_count == 0 {
                            return start_line + i + 1;
                        }
                    }
                    _ => {}
                }
            }
        }

        start_line + 20 // Fallback
    }

    /// Count function lines
    fn count_function_lines(&self, start_line: usize, lines: &[&str]) -> usize {
        let end_line = self.find_function_end(start_line, lines);
        end_line - start_line
    }

    /// Check if function has error handling
    fn has_error_handling(&self, start_line: usize, lines: &[&str]) -> bool {
        let end_line = self.find_function_end(start_line, lines);

        for line in &lines[start_line..end_line.min(lines.len())] {
            if line.contains("Result<")
                || line.contains("?")
                || line.contains("match ")
                || line.contains("if let Err")
            {
                return true;
            }
        }
        false
    }

    /// Count function parameters
    fn count_parameters(&self, line: &str) -> usize {
        if let Some(start) = line.find('(') {
            if let Some(end) = line.find(')') {
                let params_str = &line[start + 1..end];
                if params_str.trim().is_empty() {
                    return 0;
                }
                return params_str.split(',').count();
            }
        }
        0
    }

    /// Extract return type
    fn extract_return_type(&self, line: &str) -> Option<String> {
        if let Some(arrow_pos) = line.find(" -> ") {
            let after_arrow = &line[arrow_pos + 4..];
            let return_type = after_arrow.split_whitespace().next().unwrap_or("").replace("{", "");
            if !return_type.is_empty() {
                return Some(return_type);
            }
        }
        None
    }

    /// Detect various code issues
    fn detect_code_issues(&self, line: &str, line_num: usize) -> Vec<CodeIssue> {
        let mut issues = Vec::new();
        let trimmed = line.trim();

        // Long line
        if line.len() > 100 {
            issues.push(CodeIssue {
                issue_type: "long_line".to_string(),
                line_number: line_num,
                description: "Line exceeds 100 characters".to_string(),
                severity: "low".to_string(),
            });
        }

        // Unwrap usage
        if trimmed.contains(".unwrap()") && !trimmed.contains("//") {
            issues.push(CodeIssue {
                issue_type: "unwrap_usage".to_string(),
                line_number: line_num,
                description: "Consider using proper error handling instead of unwrap()".to_string(),
                severity: "medium".to_string(),
            });
        }

        // Clone usage
        if trimmed.contains(".clone()") && !trimmed.contains("//") {
            issues.push(CodeIssue {
                issue_type: "unnecessary_clone".to_string(),
                line_number: line_num,
                description: "Consider borrowing instead of cloning".to_string(),
                severity: "low".to_string(),
            });
        }

        // String concatenation with +
        if trimmed.contains(" + \"") || trimmed.contains("\" + ") {
            issues.push(CodeIssue {
                issue_type: "string_concatenation".to_string(),
                line_number: line_num,
                description: "Consider using format! macro for string concatenation".to_string(),
                severity: "low".to_string(),
            });
        }

        // Nested if statements
        if trimmed.starts_with("        if ") {
            // 8 spaces = deeply nested
            issues.push(CodeIssue {
                issue_type: "deep_nesting".to_string(),
                line_number: line_num,
                description: "Consider extracting nested logic into separate function".to_string(),
                severity: "medium".to_string(),
            });
        }

        issues
    }

    /// Generate refactoring suggestions based on analysis
    async fn generate_refactoring_suggestions(
        &self,
        analysis: &CodeAnalysis,
    ) -> Result<Vec<RefactoringSuggestion>> {
        let mut suggestions = Vec::new();

        // Function-based suggestions
        for func in &analysis.functions {
            // Large function suggestion
            if func.line_count > 50 {
                suggestions.push(RefactoringSuggestion {
                    suggestion_type: "extract_function".to_string(),
                    line_number: func.line_number,
                    description: format!(
                        "Function '{}' is {} lines long. Consider breaking it into smaller \
                         functions.",
                        func.name, func.line_count
                    ),
                    rationale: "Smaller functions are easier to test, understand, and maintain."
                        .to_string(),
                    priority: "high".to_string(),
                    difficulty: "medium".to_string(),
                    impact: "high".to_string(),
                    code_before: format!(
                        "fn {}() {{ /* {} lines of code */ }}",
                        func.name, func.line_count
                    ),
                    code_after: format!(
                        "fn {}() {{\n    helper_function_1();\n    helper_function_2();\n}}",
                        func.name
                    ),
                });
            }

            // High complexity suggestion
            if func.complexity_score > 10.0 {
                suggestions.push(RefactoringSuggestion {
                    suggestion_type: "reduce_complexity".to_string(),
                    line_number: func.line_number,
                    description: format!(
                        "Function '{}' has high cyclomatic complexity ({}). Consider simplifying.",
                        func.name, func.complexity_score
                    ),
                    rationale: "Lower complexity improves readability and reduces bug potential."
                        .to_string(),
                    priority: "high".to_string(),
                    difficulty: "high".to_string(),
                    impact: "high".to_string(),
                    code_before: "Complex nested conditions and loops".to_string(),
                    code_after: "Simplified logic with early returns and helper functions"
                        .to_string(),
                });
            }

            // Too many parameters
            if func.parameter_count > 5 {
                suggestions.push(RefactoringSuggestion {
                    suggestion_type: "parameter_object".to_string(),
                    line_number: func.line_number,
                    description: format!(
                        "Function '{}' has {} parameters. Consider using a parameter object.",
                        func.name, func.parameter_count
                    ),
                    rationale: "Parameter objects make function calls clearer and more \
                                maintainable."
                        .to_string(),
                    priority: "medium".to_string(),
                    difficulty: "medium".to_string(),
                    impact: "medium".to_string(),
                    code_before: format!(
                        "fn {}(a: T1, b: T2, c: T3, d: T4, e: T5, f: T6)",
                        func.name
                    ),
                    code_after: format!("fn {}(params: {}Params)", func.name, func.name),
                });
            }
        }

        // Issue-based suggestions
        for issue in &analysis.code_issues {
            match issue.issue_type.as_str() {
                "unwrap_usage" => {
                    suggestions.push(RefactoringSuggestion {
                        suggestion_type: "replace_unwrap".to_string(),
                        line_number: issue.line_number,
                        description: "Replace unwrap() with proper error handling".to_string(),
                        rationale: "Unwrap can cause panics. Use ? operator or match for better \
                                    error handling."
                            .to_string(),
                        priority: "high".to_string(),
                        difficulty: "low".to_string(),
                        impact: "high".to_string(),
                        code_before: "value.unwrap()".to_string(),
                        code_after: "value?".to_string(),
                    });
                }
                "unnecessary_clone" => {
                    suggestions.push(RefactoringSuggestion {
                        suggestion_type: "avoid_clone".to_string(),
                        line_number: issue.line_number,
                        description: "Consider borrowing instead of cloning".to_string(),
                        rationale: "Borrowing is more efficient and follows Rust ownership \
                                    principles."
                            .to_string(),
                        priority: "low".to_string(),
                        difficulty: "low".to_string(),
                        impact: "medium".to_string(),
                        code_before: "process(data.clone())".to_string(),
                        code_after: "process(&data)".to_string(),
                    });
                }
                "deep_nesting" => {
                    suggestions.push(RefactoringSuggestion {
                        suggestion_type: "reduce_nesting".to_string(),
                        line_number: issue.line_number,
                        description: "Extract deeply nested logic into separate function"
                            .to_string(),
                        rationale: "Reduced nesting improves readability and maintainability."
                            .to_string(),
                        priority: "medium".to_string(),
                        difficulty: "medium".to_string(),
                        impact: "medium".to_string(),
                        code_before: "if { if { if { /* deep nesting */ } } }".to_string(),
                        code_after: "Use early returns or helper functions".to_string(),
                    });
                }
                _ => {}
            }
        }

        Ok(suggestions)
    }

    /// Rank suggestions by priority and impact
    fn rank_suggestions(
        &self,
        mut suggestions: Vec<RefactoringSuggestion>,
    ) -> Vec<RefactoringSuggestion> {
        suggestions.sort_by(|a, b| {
            // First by priority (high > medium > low)
            let priority_order = |p: &str| match p {
                "high" => 3,
                "medium" => 2,
                "low" => 1,
                _ => 0,
            };

            let priority_cmp = priority_order(&b.priority).cmp(&priority_order(&a.priority));
            if priority_cmp != std::cmp::Ordering::Equal {
                return priority_cmp;
            }

            // Then by impact
            let impact_cmp = priority_order(&b.impact).cmp(&priority_order(&a.impact));
            if impact_cmp != std::cmp::Ordering::Equal {
                return impact_cmp;
            }

            // Then by difficulty (easier first)
            let difficulty_order = |d: &str| match d {
                "low" => 3,
                "medium" => 2,
                "high" => 1,
                _ => 0,
            };
            difficulty_order(&b.difficulty).cmp(&difficulty_order(&a.difficulty))
        });

        suggestions
    }
}

/// Code analysis result
struct CodeAnalysis {
    total_lines: usize,
    functions: Vec<FunctionAnalysis>,
    code_issues: Vec<CodeIssue>,
    avg_complexity: f64,
}

/// Function analysis details
struct FunctionAnalysis {
    name: String,
    line_number: usize,
    complexity_score: f64,
    line_count: usize,
    #[allow(dead_code)]
    has_error_handling: bool,
    parameter_count: usize,
    #[allow(dead_code)]
    return_type: Option<String>,
}

/// Code issue detection
#[allow(dead_code)]
struct CodeIssue {
    issue_type: String,
    line_number: usize,
    description: String,
    severity: String,
}

/// Refactoring suggestion
struct RefactoringSuggestion {
    suggestion_type: String,
    line_number: usize,
    description: String,
    rationale: String,
    priority: String,
    difficulty: String,
    impact: String,
    code_before: String,
    code_after: String,
}
