use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::memory::{CognitiveMemory, MemoryMetadata};

/// Code analyzer with AST parsing and various analysis capabilities
pub struct CodeAnalyzer {
    /// Memory system
    memory: Arc<CognitiveMemory>,

    /// Language analyzers
    analyzers: HashMap<String, Box<dyn LanguageAnalyzer>>,
}

impl CodeAnalyzer {
    /// Create a new code analyzer
    pub async fn new(memory: Arc<CognitiveMemory>) -> Result<Self> {
        info!("Initializing code analyzer");

        let mut analyzers: HashMap<String, Box<dyn LanguageAnalyzer>> = HashMap::new();

        // Add language-specific analyzers
        analyzers.insert("rs".to_string(), Box::new(RustAnalyzer::new()));
        analyzers.insert("py".to_string(), Box::new(PythonAnalyzer::new()));
        analyzers.insert("js".to_string(), Box::new(JavaScriptAnalyzer::new()));
        analyzers.insert("ts".to_string(), Box::new(TypeScriptAnalyzer::new()));
        analyzers.insert("go".to_string(), Box::new(GoAnalyzer::new()));

        Ok(Self { memory, analyzers })
    }

    /// Analyze a file
    pub async fn analyze_file(&self, path: &Path) -> Result<AnalysisResult> {
        let extension = path.extension().and_then(|e| e.to_str()).unwrap_or("");

        let content = tokio::fs::read_to_string(path).await?;

        let analysis = if let Some(analyzer) = self.analyzers.get(extension) {
            analyzer.analyze(&content)?
        } else {
            // Generic analysis for unknown file types
            self.generic_analysis(&content)
        };

        // Store analysis in memory
        self.memory
            .store(
                format!(
                    "Code analysis of {}: {} functions, {} complexity",
                    path.display(),
                    analysis.functions.len(),
                    analysis.complexity
                ),
                vec![format!("{:?}", analysis)],
                MemoryMetadata {
                    source: "code_analysis".to_string(),
                    tags: vec!["analysis".to_string(), extension.to_string()],
                    importance: 0.6,
                    associations: vec![],

                    context: Some("Generated from automated fix".to_string()),
                    created_at: chrono::Utc::now(),
                    accessed_count: 0,
                    last_accessed: None,
                    version: 1,
                    category: "tool_usage".to_string(),
                    timestamp: chrono::Utc::now(),
                    expiration: None,
                },
            )
            .await?;

        Ok(analysis)
    }

    /// Analyze a project directory
    pub async fn analyze_project(&self, dir: &Path) -> Result<ProjectAnalysis> {
        info!("Analyzing project: {:?}", dir);

        let mut project_analysis = ProjectAnalysis {
            total_files: 0,
            total_lines: 0,
            language_stats: HashMap::new(),
            complexity_score: 0.0,
            test_coverage: None,
            dependencies: Vec::new(),
            issues: Vec::new(),
        };

        // Walk through directory
        self.analyze_directory_recursive(dir, &mut project_analysis).await?;

        // Calculate overall complexity
        project_analysis.complexity_score /= project_analysis.total_files as f32;

        Ok(project_analysis)
    }

    /// Recursively analyze directory
    fn analyze_directory_recursive<'a>(
        &'a self,
        dir: &'a Path,
        project: &'a mut ProjectAnalysis,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            let mut entries = tokio::fs::read_dir(dir).await?;

            while let Some(entry) = entries.next_entry().await? {
                let path = entry.path();

                if path.is_dir() {
                    // Skip common directories
                    let dir_name = path.file_name().unwrap_or_default().to_str().unwrap_or("");
                    if !matches!(dir_name, "node_modules" | "target" | ".git" | "__pycache__") {
                        self.analyze_directory_recursive(&path, project).await?;
                    }
                } else if path.is_file() {
                    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                        if self.analyzers.contains_key(ext) {
                            match self.analyze_file(&path).await {
                                Ok(analysis) => {
                                    project.total_files += 1;
                                    project.total_lines += analysis.line_count;
                                    project.complexity_score += analysis.complexity as f32;

                                    *project.language_stats.entry(ext.to_string()).or_insert(0) +=
                                        1;

                                    // Collect issues
                                    if !analysis.issues.is_empty() {
                                        project.issues.push(FileIssues {
                                            file: path.to_path_buf(),
                                            issues: analysis.issues,
                                        });
                                    }
                                }
                                Err(e) => {
                                    warn!("Failed to analyze {}: {}", path.display(), e);
                                }
                            }
                        }
                    }
                }
            }

            Ok(())
        })
    }

    /// Generic analysis for unknown file types
    fn generic_analysis(&self, content: &str) -> AnalysisResult {
        let lines: Vec<&str> = content.lines().collect();
        let line_count = lines.len();

        AnalysisResult {
            line_count,
            functions: Vec::new(),
            complexity: 1,
            dependencies: Vec::new(),
            issues: Vec::new(),
            test_coverage: None,
        }
    }

    /// Find code patterns
    pub async fn find_patterns(
        &self,
        path: &Path,
        patterns: Vec<Pattern>,
    ) -> Result<Vec<PatternMatch>> {
        let content = tokio::fs::read_to_string(path).await?;
        let mut matches = Vec::new();

        for pattern in patterns {
            matches.extend(self.search_pattern(&content, &pattern));
        }

        Ok(matches)
    }

    /// Search for a pattern in content
    fn search_pattern(&self, content: &str, pattern: &Pattern) -> Vec<PatternMatch> {
        let mut matches = Vec::new();

        for (line_num, line) in content.lines().enumerate() {
            if pattern.matches(line) {
                matches.push(PatternMatch {
                    pattern: pattern.clone(),
                    line: line_num + 1,
                    content: line.to_string(),
                });
            }
        }

        matches
    }
}

/// Analysis result for a single file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResult {
    pub line_count: usize,
    pub functions: Vec<FunctionInfo>,
    pub complexity: u32,
    pub dependencies: Vec<String>,
    pub issues: Vec<CodeIssue>,
    pub test_coverage: Option<f32>,
}

/// Function information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionInfo {
    pub name: String,
    pub line_start: usize,
    pub line_end: usize,
    pub complexity: u32,
    pub parameters: Vec<String>,
    pub is_async: bool,
    pub is_test: bool,
}

/// Code issue found during analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeIssue {
    pub severity: IssueSeverity,
    pub message: String,
    pub line: usize,
    pub column: Option<usize>,
    pub suggestion: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum IssueSeverity {
    Error,
    Warning,
    Info,
    Hint,
}

/// Project-wide analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectAnalysis {
    pub total_files: usize,
    pub total_lines: usize,
    pub language_stats: HashMap<String, usize>,
    pub complexity_score: f32,
    pub test_coverage: Option<f32>,
    pub dependencies: Vec<Dependency>,
    pub issues: Vec<FileIssues>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileIssues {
    pub file: PathBuf,
    pub issues: Vec<CodeIssue>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dependency {
    pub name: String,
    pub version: String,
    pub is_dev: bool,
}

/// Pattern for searching
#[derive(Debug, Clone)]
pub struct Pattern {
    pub name: String,
    pub regex: regex::Regex,
    pub description: String,
}

impl Pattern {
    pub fn new(name: &str, pattern: &str, description: &str) -> Result<Self> {
        Ok(Self {
            name: name.to_string(),
            regex: regex::Regex::new(pattern)?,
            description: description.to_string(),
        })
    }

    pub fn matches(&self, text: &str) -> bool {
        self.regex.is_match(text)
    }
}

/// Pattern match result
#[derive(Debug, Clone)]
pub struct PatternMatch {
    pub pattern: Pattern,
    pub line: usize,
    pub content: String,
}

/// Trait for language-specific analyzers
trait LanguageAnalyzer: Send + Sync {
    fn analyze(&self, content: &str) -> Result<AnalysisResult>;
}

/// Rust analyzer
struct RustAnalyzer;

impl RustAnalyzer {
    fn new() -> Self {
        Self
    }
}

impl LanguageAnalyzer for RustAnalyzer {
    fn analyze(&self, content: &str) -> Result<AnalysisResult> {
        let syntax = syn::parse_file(content)?;

        let mut functions = Vec::new();
        let mut complexity = 1u32;
        let mut dependencies = Vec::new();
        let mut issues = Vec::new();

        // Extract function information
        for item in &syntax.items {
            if let syn::Item::Fn(func) = item {
                let is_test = func.attrs.iter().any(|attr| {
                    attr.path().is_ident("test") || attr.path().is_ident("tokio::test")
                });

                functions.push(FunctionInfo {
                    name: func.sig.ident.to_string(),
                    line_start: 0, // Would need to track spans
                    line_end: 0,
                    complexity: estimate_complexity(&func.block),
                    parameters: func
                        .sig
                        .inputs
                        .iter()
                        .filter_map(|arg| {
                            if let syn::FnArg::Typed(pat) = arg {
                                Some(quote::quote!(#pat).to_string())
                            } else {
                                None
                            }
                        })
                        .collect(),
                    is_async: func.sig.asyncness.is_some(),
                    is_test,
                });

                complexity += estimate_complexity(&func.block);
            }
        }

        // Extract use statements (dependencies)
        for item in &syntax.items {
            if let syn::Item::Use(use_item) = item {
                dependencies.push(quote::quote!(#use_item).to_string());
            }
        }

        // Basic issue detection
        if content.contains("unwrap()") {
            issues.push(CodeIssue {
                severity: IssueSeverity::Warning,
                message: "Use of unwrap() detected - consider using ? or proper error handling"
                    .to_string(),
                line: 0,
                column: None,
                suggestion: Some("Replace unwrap() with ?".to_string()),
            });
        }

        Ok(AnalysisResult {
            line_count: content.lines().count(),
            functions,
            complexity,
            dependencies,
            issues,
            test_coverage: None,
        })
    }
}

/// Estimate cyclomatic complexity
fn estimate_complexity(block: &syn::Block) -> u32 {
    let mut complexity = 1;

    for stmt in &block.stmts {
        if let syn::Stmt::Expr(expr, _) = stmt {
            complexity += count_branches(expr);
        }
    }

    complexity
}

/// Count branches in expression
fn count_branches(expr: &syn::Expr) -> u32 {
    match expr {
        syn::Expr::If(_) => 1,
        syn::Expr::Match(m) => m.arms.len() as u32,
        syn::Expr::While(_) | syn::Expr::ForLoop(_) | syn::Expr::Loop(_) => 1,
        _ => 0,
    }
}

/// Python analyzer (placeholder)
struct PythonAnalyzer;

impl PythonAnalyzer {
    fn new() -> Self {
        Self
    }
}

impl LanguageAnalyzer for PythonAnalyzer {
    fn analyze(&self, content: &str) -> Result<AnalysisResult> {
        let mut functions = Vec::new();
        let mut dependencies = Vec::new();
        let mut issues = Vec::new();
        let mut complexity = 1u32;
        let lines: Vec<&str> = content.lines().collect();

        // Track indentation to find function end
        let mut in_function = false;
        let mut function_indent = 0;
        let mut current_function: Option<FunctionInfo> = None;

        for (i, line) in lines.iter().enumerate() {
            let trimmed = line.trim_start();
            let indent = line.len() - trimmed.len();

            // Function detection
            if trimmed.starts_with("def ") || trimmed.starts_with("async def ") {
                // Save previous function if exists
                if let Some(mut func) = current_function.take() {
                    func.line_end = i;
                    functions.push(func);
                }

                let is_async = trimmed.starts_with("async def ");
                let def_start = if is_async { 10 } else { 4 };

                let func_signature = &trimmed[def_start..];
                let func_name = func_signature.split('(').next().unwrap_or("unknown").trim();

                // Extract parameters
                let params = if let Some(params_start) = func_signature.find('(') {
                    if let Some(params_end) = func_signature.find(')') {
                        let params_str = &func_signature[params_start + 1..params_end];
                        params_str
                            .split(',')
                            .map(|p| p.trim().split(':').next().unwrap_or(p.trim()).to_string())
                            .filter(|p| !p.is_empty() && p != "self" && p != "cls")
                            .collect()
                    } else {
                        Vec::new()
                    }
                } else {
                    Vec::new()
                };

                current_function = Some(FunctionInfo {
                    name: func_name.to_string(),
                    line_start: i + 1,
                    line_end: i + 1,
                    complexity: 1,
                    parameters: params,
                    is_async,
                    is_test: func_name.starts_with("test_") || func_name.contains("_test"),
                });

                in_function = true;
                function_indent = indent;
            } else if in_function
                && !trimmed.is_empty()
                && indent <= function_indent
                && !trimmed.starts_with("#")
            {
                // Function ended
                if let Some(mut func) = current_function.take() {
                    func.line_end = i;
                    functions.push(func);
                }
                in_function = false;
            }

            // Import detection
            if trimmed.starts_with("import ") || trimmed.starts_with("from ") {
                dependencies.push(trimmed.to_string());
            }

            // Complexity analysis
            if trimmed.contains("if ") || trimmed.contains("elif ") {
                complexity += 1;
                if let Some(ref mut func) = current_function {
                    func.complexity += 1;
                }
            }
            if trimmed.contains("for ") || trimmed.contains("while ") {
                complexity += 1;
                if let Some(ref mut func) = current_function {
                    func.complexity += 1;
                }
            }
            if trimmed.contains("try:") || trimmed.contains("except") {
                complexity += 1;
                if let Some(ref mut func) = current_function {
                    func.complexity += 1;
                }
            }

            // Issue detection
            if trimmed.contains("print(") && !trimmed.contains("#") {
                issues.push(CodeIssue {
                    severity: IssueSeverity::Info,
                    message: "Print statement found - consider using logging".to_string(),
                    line: i + 1,
                    column: None,
                    suggestion: Some("Use logging.info() instead of print()".to_string()),
                });
            }

            if trimmed.contains("TODO") || trimmed.contains("FIXME") || trimmed.contains("XXX") {
                issues.push(CodeIssue {
                    severity: IssueSeverity::Info,
                    message: format!("TODO/FIXME found: {}", trimmed),
                    line: i + 1,
                    column: None,
                    suggestion: None,
                });
            }

            if line.len() > 100 {
                issues.push(CodeIssue {
                    severity: IssueSeverity::Warning,
                    message: "Line too long (>100 characters)".to_string(),
                    line: i + 1,
                    column: Some(100),
                    suggestion: Some("Consider breaking into multiple lines".to_string()),
                });
            }
        }

        // Handle last function
        if let Some(mut func) = current_function.take() {
            func.line_end = lines.len();
            functions.push(func);
        }

        Ok(AnalysisResult {
            line_count: lines.len(),
            functions,
            complexity,
            dependencies,
            issues,
            test_coverage: None,
        })
    }
}

/// JavaScript analyzer (placeholder)
struct JavaScriptAnalyzer;

impl JavaScriptAnalyzer {
    fn new() -> Self {
        Self
    }
}

impl LanguageAnalyzer for JavaScriptAnalyzer {
    fn analyze(&self, content: &str) -> Result<AnalysisResult> {
        let mut functions = Vec::new();
        let mut dependencies = Vec::new();
        let mut issues = Vec::new();
        let mut complexity = 1u32;
        let lines: Vec<&str> = content.lines().collect();

        for (i, line) in lines.iter().enumerate() {
            let trimmed = line.trim();

            // Function detection
            if let Some(func_info) = self.parse_js_function(trimmed, i + 1) {
                functions.push(func_info);
            }

            // Import/require detection
            if trimmed.starts_with("import ")
                || trimmed.starts_with("const ") && trimmed.contains("require(")
                || trimmed.starts_with("let ") && trimmed.contains("require(")
                || trimmed.starts_with("var ") && trimmed.contains("require(")
            {
                dependencies.push(trimmed.to_string());
            }

            // Complexity analysis
            if trimmed.contains("if ") || trimmed.contains("else if ") || trimmed.contains("? ") {
                complexity += 1;
            }
            if trimmed.contains("for ") || trimmed.contains("while ") || trimmed.contains("do ") {
                complexity += 1;
            }
            if trimmed.contains("try ") || trimmed.contains("catch ") {
                complexity += 1;
            }
            if trimmed.contains("switch ") {
                complexity += 1;
            }

            // Issue detection
            if trimmed.contains("console.log") && !line.contains("//") {
                issues.push(CodeIssue {
                    severity: IssueSeverity::Info,
                    message: "Console.log found - consider using proper logging".to_string(),
                    line: i + 1,
                    column: None,
                    suggestion: Some(
                        "Use a logging library or remove before production".to_string(),
                    ),
                });
            }

            if trimmed.contains("eval(") {
                issues.push(CodeIssue {
                    severity: IssueSeverity::Error,
                    message: "Use of eval() is dangerous and should be avoided".to_string(),
                    line: i + 1,
                    column: None,
                    suggestion: Some("Find alternative to eval()".to_string()),
                });
            }

            if trimmed.contains("==") && !trimmed.contains("===") && !trimmed.contains("!=") {
                issues.push(CodeIssue {
                    severity: IssueSeverity::Warning,
                    message: "Use strict equality (===) instead of ==".to_string(),
                    line: i + 1,
                    column: None,
                    suggestion: Some("Replace == with ===".to_string()),
                });
            }

            if trimmed.contains("var ") {
                issues.push(CodeIssue {
                    severity: IssueSeverity::Info,
                    message: "Use let or const instead of var".to_string(),
                    line: i + 1,
                    column: None,
                    suggestion: Some("Replace var with let or const".to_string()),
                });
            }

            if line.len() > 120 {
                issues.push(CodeIssue {
                    severity: IssueSeverity::Warning,
                    message: "Line too long (>120 characters)".to_string(),
                    line: i + 1,
                    column: Some(120),
                    suggestion: Some("Consider breaking into multiple lines".to_string()),
                });
            }
        }

        Ok(AnalysisResult {
            line_count: lines.len(),
            functions,
            complexity,
            dependencies,
            issues,
            test_coverage: None,
        })
    }
}

impl JavaScriptAnalyzer {
    fn parse_js_function(&self, line: &str, line_num: usize) -> Option<FunctionInfo> {
        // Function patterns: function name(), const name = function(), const name = ()
        // =>, async function
        if line.contains("function ") {
            if let Some(func_start) = line.find("function ") {
                let after_function = &line[func_start + 9..];
                let is_async = line.contains("async ");

                // Extract function name
                let func_name = if let Some(paren_pos) = after_function.find('(') {
                    after_function[..paren_pos].trim()
                } else {
                    "anonymous"
                };

                // Extract parameters
                let params = if let Some(start) = after_function.find('(') {
                    if let Some(end) = after_function.find(')') {
                        let params_str = &after_function[start + 1..end];
                        params_str
                            .split(',')
                            .map(|p| p.trim().split('=').next().unwrap_or(p.trim()).to_string())
                            .filter(|p| !p.is_empty())
                            .collect()
                    } else {
                        Vec::new()
                    }
                } else {
                    Vec::new()
                };

                return Some(FunctionInfo {
                    name: if func_name.is_empty() {
                        "anonymous".to_string()
                    } else {
                        func_name.to_string()
                    },
                    line_start: line_num,
                    line_end: line_num,
                    complexity: 1,
                    parameters: params,
                    is_async,
                    is_test: func_name.contains("test") || func_name.contains("spec"),
                });
            }
        }

        // Arrow functions: const name = () =>
        if line.contains(" => ") {
            if let Some(equals_pos) = line.find(" = ") {
                let before_equals = &line[..equals_pos];
                let func_name = before_equals.split_whitespace().last().unwrap_or("anonymous");

                let arrow_part = &line[equals_pos + 3..];
                let is_async = arrow_part.trim().starts_with("async ");

                // Extract parameters
                let params = if let Some(start) = arrow_part.find('(') {
                    if let Some(end) = arrow_part.find(')') {
                        let params_str = &arrow_part[start + 1..end];
                        params_str
                            .split(',')
                            .map(|p| p.trim().split('=').next().unwrap_or(p.trim()).to_string())
                            .filter(|p| !p.is_empty())
                            .collect()
                    } else {
                        Vec::new()
                    }
                } else {
                    Vec::new()
                };

                return Some(FunctionInfo {
                    name: func_name.to_string(),
                    line_start: line_num,
                    line_end: line_num,
                    complexity: 1,
                    parameters: params,
                    is_async,
                    is_test: func_name.contains("test") || func_name.contains("spec"),
                });
            }
        }

        None
    }
}

/// TypeScript analyzer (placeholder)
struct TypeScriptAnalyzer;

impl TypeScriptAnalyzer {
    fn new() -> Self {
        Self
    }
}

impl LanguageAnalyzer for TypeScriptAnalyzer {
    fn analyze(&self, content: &str) -> Result<AnalysisResult> {
        let mut functions = Vec::new();
        let mut dependencies = Vec::new();
        let mut issues = Vec::new();
        let mut complexity = 1u32;
        let lines: Vec<&str> = content.lines().collect();

        for (i, line) in lines.iter().enumerate() {
            let trimmed = line.trim();

            // Function detection (includes TypeScript-specific features)
            if let Some(func_info) = self.parse_ts_function(trimmed, i + 1) {
                functions.push(func_info);
            }

            // Import detection
            if trimmed.starts_with("import ")
                || trimmed.starts_with("export ")
                || trimmed.contains("from ") && (trimmed.contains("'") || trimmed.contains("\""))
            {
                dependencies.push(trimmed.to_string());
            }

            // Complexity analysis (same as JavaScript)
            if trimmed.contains("if ") || trimmed.contains("else if ") || trimmed.contains("? ") {
                complexity += 1;
            }
            if trimmed.contains("for ") || trimmed.contains("while ") || trimmed.contains("do ") {
                complexity += 1;
            }
            if trimmed.contains("try ") || trimmed.contains("catch ") {
                complexity += 1;
            }
            if trimmed.contains("switch ") {
                complexity += 1;
            }

            // TypeScript-specific issues
            if trimmed.contains("any") && !trimmed.contains("//") {
                issues.push(CodeIssue {
                    severity: IssueSeverity::Warning,
                    message: "Use of 'any' type reduces type safety".to_string(),
                    line: i + 1,
                    column: None,
                    suggestion: Some("Use more specific types instead of 'any'".to_string()),
                });
            }

            if trimmed.contains("@ts-ignore") {
                issues.push(CodeIssue {
                    severity: IssueSeverity::Warning,
                    message: "@ts-ignore found - consider fixing the underlying issue".to_string(),
                    line: i + 1,
                    column: None,
                    suggestion: Some(
                        "Address the TypeScript error instead of ignoring it".to_string(),
                    ),
                });
            }

            if trimmed.contains("console.log") && !line.contains("//") {
                issues.push(CodeIssue {
                    severity: IssueSeverity::Info,
                    message: "Console.log found - consider using proper logging".to_string(),
                    line: i + 1,
                    column: None,
                    suggestion: Some(
                        "Use a logging library or remove before production".to_string(),
                    ),
                });
            }

            if trimmed.contains("==") && !trimmed.contains("===") && !trimmed.contains("!=") {
                issues.push(CodeIssue {
                    severity: IssueSeverity::Warning,
                    message: "Use strict equality (===) instead of ==".to_string(),
                    line: i + 1,
                    column: None,
                    suggestion: Some("Replace == with ===".to_string()),
                });
            }

            // Check for missing type annotations on public methods
            if (trimmed.contains("public ") || trimmed.contains("export "))
                && trimmed.contains("function ")
                && !trimmed.contains(": ")
            {
                issues.push(CodeIssue {
                    severity: IssueSeverity::Info,
                    message: "Consider adding type annotations to public functions".to_string(),
                    line: i + 1,
                    column: None,
                    suggestion: Some("Add return type annotation".to_string()),
                });
            }

            if line.len() > 120 {
                issues.push(CodeIssue {
                    severity: IssueSeverity::Warning,
                    message: "Line too long (>120 characters)".to_string(),
                    line: i + 1,
                    column: Some(120),
                    suggestion: Some("Consider breaking into multiple lines".to_string()),
                });
            }
        }

        Ok(AnalysisResult {
            line_count: lines.len(),
            functions,
            complexity,
            dependencies,
            issues,
            test_coverage: None,
        })
    }
}

impl TypeScriptAnalyzer {
    fn parse_ts_function(&self, line: &str, line_num: usize) -> Option<FunctionInfo> {
        // TypeScript function patterns include types
        if line.contains("function ") {
            if let Some(func_start) = line.find("function ") {
                let after_function = &line[func_start + 9..];
                let is_async = line.contains("async ");

                // Extract function name
                let func_name = if let Some(paren_pos) = after_function.find('(') {
                    after_function[..paren_pos].trim()
                } else {
                    "anonymous"
                };

                // Extract parameters (with types)
                let params = if let Some(start) = after_function.find('(') {
                    if let Some(end) = after_function.find(')') {
                        let params_str = &after_function[start + 1..end];
                        params_str
                            .split(',')
                            .map(|p| {
                                // Extract parameter name, ignoring type annotations
                                let param_part = p.trim().split(':').next().unwrap_or(p.trim());
                                param_part
                                    .split('=')
                                    .next()
                                    .unwrap_or(param_part)
                                    .trim()
                                    .to_string()
                            })
                            .filter(|p| !p.is_empty())
                            .collect()
                    } else {
                        Vec::new()
                    }
                } else {
                    Vec::new()
                };

                return Some(FunctionInfo {
                    name: if func_name.is_empty() {
                        "anonymous".to_string()
                    } else {
                        func_name.to_string()
                    },
                    line_start: line_num,
                    line_end: line_num,
                    complexity: 1,
                    parameters: params,
                    is_async,
                    is_test: func_name.contains("test") || func_name.contains("spec"),
                });
            }
        }

        // Arrow functions with types
        if line.contains(" => ") {
            if let Some(equals_pos) = line.find(" = ") {
                let before_equals = &line[..equals_pos];
                let func_name = before_equals.split_whitespace().last().unwrap_or("anonymous");

                let arrow_part = &line[equals_pos + 3..];
                let is_async = arrow_part.trim().starts_with("async ");

                // Extract parameters
                let params = if let Some(start) = arrow_part.find('(') {
                    if let Some(end) = arrow_part.find(')') {
                        let params_str = &arrow_part[start + 1..end];
                        params_str
                            .split(',')
                            .map(|p| {
                                let param_part = p.trim().split(':').next().unwrap_or(p.trim());
                                param_part
                                    .split('=')
                                    .next()
                                    .unwrap_or(param_part)
                                    .trim()
                                    .to_string()
                            })
                            .filter(|p| !p.is_empty())
                            .collect()
                    } else {
                        Vec::new()
                    }
                } else {
                    Vec::new()
                };

                return Some(FunctionInfo {
                    name: func_name.to_string(),
                    line_start: line_num,
                    line_end: line_num,
                    complexity: 1,
                    parameters: params,
                    is_async,
                    is_test: func_name.contains("test") || func_name.contains("spec"),
                });
            }
        }

        // Method definitions in classes/interfaces
        if (line.contains("public ") || line.contains("private ") || line.contains("protected "))
            && line.contains("(")
            && line.contains(")")
        {
            let method_part = if let Some(access_pos) = line
                .find("public ")
                .or_else(|| line.find("private "))
                .or_else(|| line.find("protected "))
            {
                let access_keyword_end = if line.contains("public ") {
                    access_pos + 7
                } else if line.contains("private ") {
                    access_pos + 8
                } else {
                    access_pos + 10
                };
                &line[access_keyword_end..]
            } else {
                line
            };

            let is_async = method_part.contains("async ");
            let method_name = method_part.trim().split('(').next().unwrap_or("unknown").trim();

            if !method_name.is_empty() && method_name != "constructor" {
                return Some(FunctionInfo {
                    name: method_name.to_string(),
                    line_start: line_num,
                    line_end: line_num,
                    complexity: 1,
                    parameters: Vec::new(), // Could extract parameters here too
                    is_async,
                    is_test: method_name.contains("test") || method_name.contains("spec"),
                });
            }
        }

        None
    }
}

/// Go analyzer (placeholder)
struct GoAnalyzer;

impl GoAnalyzer {
    fn new() -> Self {
        Self
    }
}

impl LanguageAnalyzer for GoAnalyzer {
    fn analyze(&self, content: &str) -> Result<AnalysisResult> {
        let mut functions = Vec::new();
        let mut dependencies = Vec::new();
        let mut issues = Vec::new();
        let mut complexity = 1u32;
        let lines: Vec<&str> = content.lines().collect();

        for (i, line) in lines.iter().enumerate() {
            let trimmed = line.trim();

            // Function detection
            if let Some(func_info) = self.parse_go_function(trimmed, i + 1) {
                functions.push(func_info);
            }

            // Import detection
            if trimmed.starts_with("import ")
                || (trimmed.starts_with("import") && trimmed.contains("("))
                || trimmed.contains("\"") && !trimmed.starts_with("//")
            {
                dependencies.push(trimmed.to_string());
            }

            // Complexity analysis
            if trimmed.contains("if ") || trimmed.contains("else if ") {
                complexity += 1;
            }
            if trimmed.contains("for ") || trimmed.contains("range ") {
                complexity += 1;
            }
            if trimmed.contains("switch ") || trimmed.contains("case ") {
                complexity += 1;
            }
            if trimmed.contains("select ") {
                complexity += 1;
            }

            // Go-specific issues
            if trimmed.contains("panic(") && !trimmed.contains("//") {
                issues.push(CodeIssue {
                    severity: IssueSeverity::Warning,
                    message: "Use of panic() - consider returning an error instead".to_string(),
                    line: i + 1,
                    column: None,
                    suggestion: Some("Return an error instead of panicking".to_string()),
                });
            }

            if trimmed.contains("fmt.Print")
                && !trimmed.contains("//")
                && !func_name_from_line(trimmed).map_or(false, |name| name.contains("test"))
            {
                issues.push(CodeIssue {
                    severity: IssueSeverity::Info,
                    message: "Print statement found - consider using logging".to_string(),
                    line: i + 1,
                    column: None,
                    suggestion: Some("Use log package or remove debug prints".to_string()),
                });
            }

            if trimmed.contains("_ = ") && !trimmed.contains("//") {
                issues.push(CodeIssue {
                    severity: IssueSeverity::Info,
                    message: "Blank identifier assignment - ensure this is intentional".to_string(),
                    line: i + 1,
                    column: None,
                    suggestion: Some("Consider handling the ignored value".to_string()),
                });
            }

            // Check for unhandled errors
            if trimmed.contains("err")
                && !trimmed.contains("if err")
                && !trimmed.contains("return")
                && !trimmed.contains("//")
                && !trimmed.contains("nil")
            {
                issues.push(CodeIssue {
                    severity: IssueSeverity::Warning,
                    message: "Potential unhandled error".to_string(),
                    line: i + 1,
                    column: None,
                    suggestion: Some("Check for errors with 'if err != nil'".to_string()),
                });
            }

            if line.len() > 120 {
                issues.push(CodeIssue {
                    severity: IssueSeverity::Warning,
                    message: "Line too long (>120 characters)".to_string(),
                    line: i + 1,
                    column: Some(120),
                    suggestion: Some("Consider breaking into multiple lines".to_string()),
                });
            }
        }

        Ok(AnalysisResult {
            line_count: lines.len(),
            functions,
            complexity,
            dependencies,
            issues,
            test_coverage: None,
        })
    }
}

impl GoAnalyzer {
    fn parse_go_function(&self, line: &str, line_num: usize) -> Option<FunctionInfo> {
        // Go function pattern: func name(params) returnType {
        if line.starts_with("func ") {
            let after_func = &line[5..];

            // Handle method receivers: func (r Receiver) method()
            let (func_part, _has_receiver) = if after_func.trim_start().starts_with("(") {
                if let Some(receiver_end) = after_func.find(") ") {
                    (&after_func[receiver_end + 2..], true)
                } else {
                    (after_func, false)
                }
            } else {
                (after_func, false)
            };

            // Extract function name
            let func_name = if let Some(paren_pos) = func_part.find('(') {
                func_part[..paren_pos].trim()
            } else {
                "unknown"
            };

            // Extract parameters
            let params = if let Some(start) = func_part.find('(') {
                if let Some(end) = func_part.find(')') {
                    let params_str = &func_part[start + 1..end];
                    params_str
                        .split(',')
                        .map(|p| {
                            // Go parameters can have types: name type
                            let parts: Vec<&str> = p.trim().split_whitespace().collect();
                            if parts.len() >= 2 {
                                parts[0].to_string() // parameter name
                            } else {
                                p.trim().to_string()
                            }
                        })
                        .filter(|p| !p.is_empty())
                        .collect()
                } else {
                    Vec::new()
                }
            } else {
                Vec::new()
            };

            return Some(FunctionInfo {
                name: func_name.to_string(),
                line_start: line_num,
                line_end: line_num,
                complexity: 1,
                parameters: params,
                is_async: false, // Go doesn't have async/await
                is_test: func_name.starts_with("Test")
                    || func_name.starts_with("Benchmark")
                    || func_name.starts_with("Example"),
            });
        }

        None
    }
}

// Helper function to extract function name from a line (for Go)
fn func_name_from_line(line: &str) -> Option<String> {
    if line.trim().starts_with("func ") {
        let after_func = &line.trim()[5..];
        let func_part = if after_func.trim_start().starts_with("(") {
            if let Some(receiver_end) = after_func.find(") ") {
                &after_func[receiver_end + 2..]
            } else {
                after_func
            }
        } else {
            after_func
        };

        if let Some(paren_pos) = func_part.find('(') {
            Some(func_part[..paren_pos].trim().to_string())
        } else {
            None
        }
    } else {
        None
    }
}
