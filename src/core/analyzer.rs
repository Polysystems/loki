use std::path::Path;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tracing::{debug, info};
use walkdir::WalkDir;

use crate::config::Config;

/// Analysis result structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Analysis {
    pub file_count: usize,
    pub total_lines: usize,
    pub languages: Vec<String>,
    pub issues: Vec<Issue>,
    pub statistics: Statistics,
}

/// Issue found during analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Issue {
    pub severity: Severity,
    pub message: String,
    pub file: String,
    pub line: Option<usize>,
}

/// Issue severity levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Severity {
    Error,
    Warning,
    Info,
}

/// Code statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Statistics {
    pub functions: usize,
    pub classes: usize,
    pub imports: usize,
    pub comments: usize,
}

/// Code analyzer
pub struct Analyzer {
    #[allow(dead_code)] // TODO: Implement configuration usage
    config: Config,
}

impl Analyzer {
    /// Create a new analyzer
    pub fn new(config: Config) -> Self {
        Self { config }
    }

    /// Analyze a path (file or directory)
    pub async fn analyze(&self, path: &Path) -> Result<Analysis> {
        info!("Starting analysis of: {:?}", path);

        let mut analysis = Analysis {
            file_count: 0,
            total_lines: 0,
            languages: Vec::new(),
            issues: Vec::new(),
            statistics: Statistics { functions: 0, classes: 0, imports: 0, comments: 0 },
        };

        if path.is_file() {
            self.analyze_file(path, &mut analysis).await?;
        } else if path.is_dir() {
            self.analyze_directory(path, &mut analysis).await?;
        } else {
            anyhow::bail!("Path is neither a file nor a directory: {:?}", path);
        }

        // Deduplicate languages
        analysis.languages.sort();
        analysis.languages.dedup();

        debug!("Analysis complete: {:?}", analysis);
        Ok(analysis)
    }

    /// Analyze a single file
    async fn analyze_file(&self, path: &Path, analysis: &mut Analysis) -> Result<()> {
        debug!("Analyzing file: {:?}", path);

        // Skip non-text files
        if !self.is_text_file(path) {
            return Ok(());
        }

        // Read file content
        let content = tokio::fs::read_to_string(path).await?;
        let lines: Vec<&str> = content.lines().collect();

        analysis.file_count += 1;
        analysis.total_lines += lines.len();

        // Detect language
        if let Some(lang) = self.detect_language(path) {
            analysis.languages.push(lang);
        }

        // Basic analysis
        for (line_num, line) in lines.iter().enumerate() {
            let trimmed = line.trim();

            // Count comments
            if self.is_comment(trimmed) {
                analysis.statistics.comments += 1;
            }

            // Count imports
            if self.is_import(trimmed) {
                analysis.statistics.imports += 1;
            }

            // Count functions
            if self.is_function_def(trimmed) {
                analysis.statistics.functions += 1;
            }

            // Check for basic issues
            if let Some(issue) = self.check_line_issues(trimmed, line_num + 1, path) {
                analysis.issues.push(issue);
            }
        }

        Ok(())
    }

    /// Analyze a directory recursively
    async fn analyze_directory(&self, path: &Path, analysis: &mut Analysis) -> Result<()> {
        debug!("Analyzing directory: {:?}", path);

        for entry in WalkDir::new(path).follow_links(false).into_iter().filter_map(|e| e.ok()) {
            if entry.file_type().is_file() {
                // Skip hidden files and common ignore patterns
                let file_name = entry.file_name().to_string_lossy();
                if file_name.starts_with('.')
                    || file_name == "Cargo.lock"
                    || entry.path().components().any(|c| {
                        matches!(c.as_os_str().to_str(), Some("target" | "node_modules" | ".git"))
                    })
                {
                    continue;
                }

                if let Err(e) = self.analyze_file(entry.path(), analysis).await {
                    debug!("Error analyzing file {:?}: {}", entry.path(), e);
                }
            }
        }

        Ok(())
    }

    /// Check if a file is a text file
    fn is_text_file(&self, path: &Path) -> bool {
        match path.extension() {
            Some(ext) => {
                let ext = ext.to_string_lossy().to_lowercase();
                matches!(
                    ext.as_str(),
                    "rs" | "py"
                        | "js"
                        | "ts"
                        | "jsx"
                        | "tsx"
                        | "java"
                        | "c"
                        | "cpp"
                        | "h"
                        | "hpp"
                        | "go"
                        | "rb"
                        | "php"
                        | "cs"
                        | "swift"
                        | "kt"
                        | "scala"
                        | "r"
                        | "jl"
                        | "lua"
                        | "vim"
                        | "sh"
                        | "bash"
                        | "zsh"
                        | "fish"
                        | "ps1"
                        | "bat"
                        | "toml"
                        | "yaml"
                        | "yml"
                        | "json"
                        | "xml"
                        | "html"
                        | "css"
                        | "scss"
                        | "md"
                        | "txt"
                        | "conf"
                        | "cfg"
                        | "ini"
                )
            }
            None => false,
        }
    }

    /// Detect programming language from file extension
    fn detect_language(&self, path: &Path) -> Option<String> {
        path.extension()
            .and_then(|ext| ext.to_str())
            .and_then(|ext| match ext.to_lowercase().as_str() {
                "rs" => Some("Rust"),
                "py" => Some("Python"),
                "js" | "jsx" => Some("JavaScript"),
                "ts" | "tsx" => Some("TypeScript"),
                "java" => Some("Java"),
                "c" | "h" => Some("C"),
                "cpp" | "cc" | "cxx" | "hpp" => Some("C++"),
                "go" => Some("Go"),
                "rb" => Some("Ruby"),
                "php" => Some("PHP"),
                "cs" => Some("C#"),
                "swift" => Some("Swift"),
                "kt" | "kts" => Some("Kotlin"),
                "scala" => Some("Scala"),
                "r" => Some("R"),
                "jl" => Some("Julia"),
                "lua" => Some("Lua"),
                "sh" | "bash" | "zsh" | "fish" => Some("Shell"),
                "ps1" => Some("PowerShell"),
                "bat" | "cmd" => Some("Batch"),
                _ => None,
            })
            .map(|s| s.to_string())
    }

    /// Check if a line is a comment
    fn is_comment(&self, line: &str) -> bool {
        line.starts_with("//")
            || line.starts_with('#')
            || line.starts_with("/*")
            || line.starts_with("*")
            || line.starts_with("--")
    }

    /// Check if a line is an import statement
    fn is_import(&self, line: &str) -> bool {
        line.starts_with("use ")
            || line.starts_with("import ")
            || line.starts_with("from ")
            || line.starts_with("require(")
            || line.starts_with("#include")
    }

    /// Check if a line is a function definition
    fn is_function_def(&self, line: &str) -> bool {
        line.contains("fn ")
            || line.contains("def ")
            || line.contains("function ")
            || line.contains("func ")
            || (line.contains("public ") && line.contains("("))
            || (line.contains("private ") && line.contains("("))
    }

    /// Check for common issues in a line
    fn check_line_issues(&self, line: &str, line_num: usize, path: &Path) -> Option<Issue> {
        let file = path.display().to_string();

        // Check for TODO/FIXME comments
        if line.contains("TODO") || line.contains("FIXME") {
            return Some(Issue {
                severity: Severity::Info,
                message: format!(
                    "Found {} comment",
                    if line.contains("TODO") { "TODO" } else { "FIXME" }
                ),
                file,
                line: Some(line_num),
            });
        }

        // Check for potential issues
        if line.contains("unwrap()") && !line.contains("// ") {
            return Some(Issue {
                severity: Severity::Warning,
                message: "Using unwrap() without error handling".to_string(),
                file,
                line: Some(line_num),
            });
        }

        if line.contains("panic!") {
            return Some(Issue {
                severity: Severity::Warning,
                message: "Using panic! macro".to_string(),
                file,
                line: Some(line_num),
            });
        }

        if line.len() > 120 {
            return Some(Issue {
                severity: Severity::Info,
                message: "Line exceeds 120 characters".to_string(),
                file,
                line: Some(line_num),
            });
        }

        None
    }
}
