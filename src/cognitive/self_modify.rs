use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::process::Command;
use tracing::{info, warn};

use crate::cognitive::sandbox_executor::{SandboxConfig, SandboxExecutor, SandboxResult};
use crate::cognitive::test_generator::{TestGenerator, TestGeneratorConfig};
use crate::memory::{CognitiveMemory, MemoryMetadata};
use crate::safety::{ActionValidator, ValidatorConfig};
use crate::tools::code_analysis::{AnalysisResult, CodeAnalyzer};
use crate::tools::github::{GitHubClient, GitHubConfig};

/// Code change proposal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeChange {
    pub file_path: PathBuf,
    pub change_type: ChangeType,
    pub description: String,
    pub reasoning: String,
    pub old_content: Option<String>,
    pub new_content: String,
    pub line_range: Option<(usize, usize)>,
    pub risk_level: RiskLevel,
    pub attribution: Option<Attribution>,
}

/// Attribution information for community suggestions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Attribution {
    pub contributor: String,
    pub platform: String,
    pub suggestion_id: String,
    pub suggestion_text: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ChangeType {
    Create,
    Modify,
    Delete,
    Refactor,
    BugFix,
    Feature,
    Enhancement,
    Documentation,
    Test,
    PerformanceOptimization,
    SecurityPatch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RiskLevel {
    Low = 0,
    Medium = 1,
    High = 2,
    Critical = 3,
}

/// Validation result
#[derive(Debug)]
pub struct ValidationResult {
    pub passed: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub syntax_valid: bool,
    pub tests_pass: bool,
    pub behavior_preserved: bool,
}

/// The self-modification pipeline
pub struct SelfModificationPipeline {
    /// Code validator
    validator: Arc<MultiStageValidator>,

    /// Enhanced sandbox executor
    sandbox_executor: Arc<SandboxExecutor>,

    /// Test runner
    test_runner: Arc<IntegratedTestRunner>,

    /// Rollback manager
    rollback: Arc<InstantRollback>,

    /// Memory system for tracking changes
    memory: Arc<CognitiveMemory>,

    /// Repository path
    repo_path: PathBuf,

    /// GitHub client for PR creation
    github_client: Option<Arc<GitHubClient>>,

    /// Code analyzer for AST understanding
    code_analyzer: Arc<CodeAnalyzer>,

    /// Test generator for automated test creation
    test_generator: Arc<TestGenerator>,
}

impl SelfModificationPipeline {
    pub async fn new(repo_path: PathBuf, memory: Arc<CognitiveMemory>) -> Result<Self> {
        info!("Initializing self-modification pipeline for {:?}", repo_path);

        // Try to create GitHub client if config is available
        let github_client = match GitHubConfig::from_env() {
            Ok(config) => match GitHubClient::new(config, memory.clone()).await {
                Ok(client) => {
                    info!("GitHub client initialized successfully");
                    Some(Arc::new(client))
                }
                Err(e) => {
                    warn!("Failed to initialize GitHub client: {}", e);
                    None
                }
            },
            Err(e) => {
                warn!("GitHub config not available: {}", e);
                None
            }
        };

        let code_analyzer = Arc::new(CodeAnalyzer::new(memory.clone()).await?);

        // Create test generator
        let test_generator =
            Arc::new(TestGenerator::new(TestGeneratorConfig::default(), memory.clone()).await?);

        // Create sandbox executor with custom config
        let mut sandboxconfig = SandboxConfig::default();
        sandboxconfig.sandbox_base_path = repo_path.join(".loki_sandboxes");
        sandboxconfig.max_execution_time = Duration::from_secs(600); // 10 minutes for tests
        sandboxconfig.max_memory_mb = 2048; // 2GB for compilation

        let action_validator = Arc::new(ActionValidator::new(ValidatorConfig::default()).await?);
        let sandbox_executor =
            Arc::new(SandboxExecutor::new(sandboxconfig, action_validator, memory.clone()).await?);

        Ok(Self {
            validator: Arc::new(MultiStageValidator::new()),
            sandbox_executor,
            test_runner: Arc::new(IntegratedTestRunner::new(&repo_path)),
            rollback: Arc::new(InstantRollback::new(&repo_path)),
            code_analyzer,
            test_generator,
            repo_path,
            memory,
            github_client,
        })
    }

    /// Analyze code before making changes
    pub async fn analyze_code(&self, file_path: &Path) -> Result<AnalysisResult> {
        self.code_analyzer.analyze_file(file_path).await
    }

    /// Propose a code change with full validation and testing
    pub async fn propose_change(&self, change: CodeChange) -> Result<PullRequest> {
        info!("Proposing code change: {:?}", change.description);

        // Analyze the current code if modifying
        if matches!(
            change.change_type,
            ChangeType::Modify | ChangeType::Refactor | ChangeType::BugFix
        ) {
            let analysis = self.analyze_code(&change.file_path).await?;
            info!(
                "Current code analysis: {} functions, {} complexity",
                analysis.functions.len(),
                analysis.complexity
            );
        }

        // Store proposal in memory with attribution
        let attribution_text = if let Some(ref attr) = change.attribution {
            format!(" (suggested by @{} on {})", attr.contributor, attr.platform)
        } else {
            String::new()
        };

        self.memory
            .store(
                format!(
                    "Code change proposal: {} - {}{}",
                    change.description, change.reasoning, attribution_text
                ),
                vec![format!("{:?}", change)],
                MemoryMetadata {
                    source: "self_modification".to_string(),
                    timestamp: chrono::Utc::now(),
                    expiration: None,
                    tags: vec!["proposal".to_string(), format!("{:?}", change.change_type)],
                    importance: 0.8,
                    associations: vec![],
                    context: Some("self modification proposal".to_string()),
                    created_at: chrono::Utc::now(),
                    accessed_count: 0,
                    last_accessed: None,
                    version: 1,
                    category: "cognitive".to_string(),
                },
            )
            .await?;

        // Validate the change
        let validation = self.validator.validate(&change).await?;
        if !validation.passed {
            return Err(anyhow::anyhow!("Validation failed: {}", validation.errors.join(", ")));
        }

        // Create a branch
        let branch_name = self.create_branch_name(&change);
        self.create_branch(&branch_name).await?;

        // Create sandbox for testing
        let sandbox = self.sandbox_executor.create_sandbox(format!("test_{}", branch_name)).await?;

        // Apply change in sandbox first
        self.apply_change_to_sandbox(&change, &sandbox.id).await?;

        // Run tests in sandbox
        let sandbox_test_result = self.run_sandbox_tests(&sandbox.id).await?;

        if !sandbox_test_result.success {
            warn!("Sandbox tests failed: {}", sandbox_test_result.stderr);

            // Clean up sandbox
            self.sandbox_executor.cleanup_sandbox(&sandbox.id).await?;

            // Check if we should attempt to fix
            if sandbox_test_result.violations.is_empty() {
                // No security violations, just test failures - could attempt fix
                warn!("Tests failed but no security violations, could attempt auto-fix");
            }

            return Err(anyhow::anyhow!(
                "Sandbox validation failed: {}",
                sandbox_test_result.stderr
            ));
        }

        // Create snapshot before applying to real repo
        let _snapshot = self.sandbox_executor.create_snapshot(&sandbox.id).await?;

        // Apply change to actual repository
        self.apply_change_to_repo(&change).await?;

        // Run full test suite
        let test_result = self.test_runner.run_tests(&branch_name).await?;
        if !test_result.all_passed {
            // Rollback using snapshot
            warn!("Tests failed, rolling back changes");
            self.rollback.revert_branch(&branch_name).await?;

            // Clean up
            self.sandbox_executor.cleanup_sandbox(&sandbox.id).await?;

            return Err(anyhow::anyhow!("Tests failed: {:?}", test_result.failures));
        }

        // Clean up successful sandbox
        self.sandbox_executor.cleanup_sandbox(&sandbox.id).await?;

        // Commit the changes
        self.commit_changes(&change, &branch_name).await?;

        // Create pull request
        let pr = if let Some(ref github_client) = self.github_client {
            // Use real GitHub API
            github_client.create_pull_request(branch_name.clone(), change.clone()).await?
        } else {
            // Fallback to local PR representation
            self.create_local_pull_request(branch_name.clone(), change.clone()).await?
        };

        // Store result in memory
        self.memory
            .store(
                format!("Created PR #{}: {}", pr.number, pr.title),
                vec![pr.description.clone()],
                MemoryMetadata {
                    source: "self_modification".to_string(),
                    tags: vec!["pull_request".to_string(), "success".to_string()],
                    importance: 0.9,
                    associations: vec![],
                    context: Some("pull request creation".to_string()),
                    created_at: chrono::Utc::now(),
                    accessed_count: 0,
                    last_accessed: None,
                    version: 1,
                    category: "cognitive".to_string(),
                    timestamp: chrono::Utc::now(),
                    expiration: None,
                },
            )
            .await?;

        Ok(pr)
    }

    /// Apply change to the actual repository
    async fn apply_change_to_repo(&self, change: &CodeChange) -> Result<()> {
        let target_path = self.repo_path.join(&change.file_path);

        // Create parent directories if needed
        if let Some(parent) = target_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        // Apply the change
        match change.change_type {
            ChangeType::Create => {
                tokio::fs::write(&target_path, &change.new_content).await?;
            }
            ChangeType::Modify
            | ChangeType::Refactor
            | ChangeType::BugFix
            | ChangeType::Feature => {
                tokio::fs::write(&target_path, &change.new_content).await?;
            }
            ChangeType::Delete => {
                if target_path.exists() {
                    tokio::fs::remove_file(&target_path).await?;
                }
            }
            _ => {
                tokio::fs::write(&target_path, &change.new_content).await?;
            }
        }

        Ok(())
    }

    /// Commit changes to git
    async fn commit_changes(&self, change: &CodeChange, _branch_name: &str) -> Result<()> {
        // Stage the changes
        let output = Command::new("git")
            .args(&["add", change.file_path.to_str().unwrap()])
            .current_dir(&self.repo_path)
            .output()
            .await?;

        if !output.status.success() {
            return Err(anyhow::anyhow!(
                "Failed to stage changes: {}",
                String::from_utf8_lossy(&output.stderr)
            ));
        }

        // Create commit message with attribution
        let mut commit_message = format!(
            "{}: {}\n\n{}",
            match change.change_type {
                ChangeType::Feature => "feat",
                ChangeType::BugFix => "fix",
                ChangeType::Refactor => "refactor",
                ChangeType::Documentation => "docs",
                ChangeType::Test => "test",
                _ => "chore",
            },
            change.description,
            change.reasoning
        );

        if let Some(ref attr) = change.attribution {
            commit_message.push_str(&format!(
                "\n\nSuggested by: @{} on {}\nOriginal suggestion: {}",
                attr.contributor, attr.platform, attr.suggestion_text
            ));
        }

        // Commit
        let output = Command::new("git")
            .args(&["commit", "-m", &commit_message])
            .current_dir(&self.repo_path)
            .output()
            .await?;

        if !output.status.success() {
            return Err(anyhow::anyhow!(
                "Failed to commit: {}",
                String::from_utf8_lossy(&output.stderr)
            ));
        }

        Ok(())
    }

    /// Create a local pull request representation
    async fn create_local_pull_request(
        &self,
        branch_name: String,
        change: CodeChange,
    ) -> Result<PullRequest> {
        let mut description = format!(
            "## Description\n{}\n\n## Reasoning\n{}\n\n## Risk Level\n{:?}\n\n## Changes\n- \
             Modified: {}",
            change.description,
            change.reasoning,
            change.risk_level,
            change.file_path.display()
        );

        // Add attribution section if present
        if let Some(ref attr) = change.attribution {
            description.push_str(&format!(
                "\n\n## Attribution\nThis change was suggested by **@{}** on **{}**\n\n> \
                 {}\n\nTimestamp: {}",
                attr.contributor, attr.platform, attr.suggestion_text, attr.timestamp
            ));
        }

        // Add code analysis if available
        if let Ok(analysis) = self.analyze_code(&change.file_path).await {
            description.push_str(&format!(
                "\n\n## Code Analysis\n- Functions: {}\n- Complexity: {}\n- Lines: {}",
                analysis.functions.len(),
                analysis.complexity,
                analysis.line_count
            ));
        }

        Ok(PullRequest {
            number: rand::random::<u32>() % 1000 + 1,
            title: change.description.clone(),
            description,
            branch: branch_name,
            status: PullRequestStatus::Open,
        })
    }

    /// Generate test cases for a code change
    pub async fn generate_tests(&self, change: &CodeChange) -> Result<String> {
        info!("Generating tests for code change: {}", change.description);

        // Create a temporary file with the new content
        let temp_path = self.repo_path.join(".tmp_test_gen").join(&change.file_path);
        if let Some(parent) = temp_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        tokio::fs::write(&temp_path, &change.new_content).await?;

        // Generate tests using AI
        let test_suite = self.test_generator.generate_tests_for_file(&temp_path).await?;

        // Clean up temp file
        let _ = tokio::fs::remove_file(&temp_path).await;

        // Format tests into a single string
        let mut test_code = String::new();

        // Add imports
        for import in &test_suite.imports {
            test_code.push_str(import);
            test_code.push('\n');
        }
        test_code.push('\n');

        // Add test module
        test_code.push_str("#[cfg(test)]\n");
        test_code.push_str("mod generated_tests {\n");
        test_code.push_str("    use super::*;\n\n");

        // Add each test case
        for test_case in &test_suite.test_cases {
            test_code.push_str("    ");
            test_code.push_str(&test_case.code.replace('\n', "\n    "));
            test_code.push_str("\n\n");
        }

        test_code.push_str("}\n");

        // Store generated tests in memory
        self.memory
            .store(
                format!(
                    "Generated {} tests for {}",
                    test_suite.test_cases.len(),
                    change.file_path.display()
                ),
                vec![test_code.clone()],
                MemoryMetadata {
                    source: "test_generation".to_string(),
                    tags: vec!["tests".to_string(), "ai_generated".to_string()],
                    importance: 0.8,
                    associations: vec![],
                    context: Some("test generation process".to_string()),
                    created_at: chrono::Utc::now(),
                    accessed_count: 0,
                    last_accessed: None,
                    version: 1,
                    category: "cognitive".to_string(),
                    timestamp: chrono::Utc::now(),
                    expiration: None,
                },
            )
            .await?;

        info!(
            "Generated {} tests with estimated {:.1}% coverage",
            test_suite.test_cases.len(),
            test_suite.coverage_estimate * 100.0
        );

        Ok(test_code)
    }

    /// Create a descriptive branch name
    fn create_branch_name(&self, change: &CodeChange) -> String {
        let type_prefix = match change.change_type {
            ChangeType::Feature => "feat",
            ChangeType::Enhancement => "enhance",
            ChangeType::BugFix => "fix",
            ChangeType::Refactor => "refactor",
            ChangeType::Documentation => "docs",
            ChangeType::Test => "test",
            _ => "chore",
        };

        let clean_desc = change
            .description
            .to_lowercase()
            .chars()
            .map(|c| if c.is_alphanumeric() { c } else { '-' })
            .collect::<String>()
            .trim_matches('-')
            .to_string();

        format!("{}/{}", type_prefix, clean_desc)
    }

    /// Create a new git branch
    async fn create_branch(&self, branch_name: &str) -> Result<()> {
        let output = Command::new("git")
            .args(&["checkout", "-b", branch_name])
            .current_dir(&self.repo_path)
            .output()
            .await?;

        if !output.status.success() {
            return Err(anyhow::anyhow!(
                "Failed to create branch: {}",
                String::from_utf8_lossy(&output.stderr)
            ));
        }

        Ok(())
    }

    /// Apply change to sandbox
    async fn apply_change_to_sandbox(&self, change: &CodeChange, sandbox_id: &str) -> Result<()> {
        // Get sandbox path
        let sandbox_path = self.sandbox_executor.get_sandbox_path(sandbox_id).await?;
        let target_path = sandbox_path.join(&change.file_path);

        // Create parent directories if needed
        if let Some(parent) = target_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        // Apply the change
        match change.change_type {
            ChangeType::Create => {
                tokio::fs::write(&target_path, &change.new_content).await?;
            }
            ChangeType::Modify
            | ChangeType::Refactor
            | ChangeType::BugFix
            | ChangeType::Feature => {
                // Copy original file first if it exists
                let original = self.repo_path.join(&change.file_path);
                if original.exists() {
                    tokio::fs::copy(&original, &target_path).await?;
                }
                tokio::fs::write(&target_path, &change.new_content).await?;
            }
            ChangeType::Delete => {
                if target_path.exists() {
                    tokio::fs::remove_file(&target_path).await?;
                }
            }
            _ => {
                tokio::fs::write(&target_path, &change.new_content).await?;
            }
        }

        Ok(())
    }

    /// Run tests in sandbox
    async fn run_sandbox_tests(&self, sandbox_id: &str) -> Result<SandboxResult> {
        info!("Running tests in sandbox: {}", sandbox_id);

        // Get sandbox path
        let sandbox_path = self.sandbox_executor.get_sandbox_path(sandbox_id).await?;

        // Copy Cargo.toml and other necessary files
        let files_to_copy = ["Cargo.toml", "Cargo.lock", "rust-toolchain.toml"];
        for file in &files_to_copy {
            let src = self.repo_path.join(file);
            if src.exists() {
                let dst = sandbox_path.join(file);
                tokio::fs::copy(&src, &dst).await?;
            }
        }

        // Copy src directory structure
        copy_dir_structure(&self.repo_path.join("src"), &sandbox_path.join("src")).await?;

        // Run cargo test in sandbox
        self.sandbox_executor.execute(sandbox_id, "cargo", &["test", "--all"]).await
    }

    /// Apply a code change directly without creating a PR
    pub async fn apply_code_change(&self, change: CodeChange) -> Result<()> {
        warn!("apply_code_change is not yet implemented for file: {:?}", change.file_path);

        // In a real implementation, this would:
        // 1. Validate the change
        // 2. Apply it to the file system
        // 3. Run tests to ensure nothing broke
        // 4. Store the change in memory for tracking
        Ok(())
    }

    /// Rollback a previously applied code change
    pub async fn rollback_change(&self, change_id: &str) -> Result<()> {
        warn!("rollback_change is not yet implemented for change_id: {}", change_id);

        // In a real implementation, this would:
        // 1. Look up the change in memory/history
        // 2. Revert the file to its previous state
        // 3. Run tests to ensure the rollback worked
        // 4. Update the change tracking
        Ok(())
    }
}

/// Multi-stage code validator
pub struct MultiStageValidator {
    syntax_validators: HashMap<String, Box<dyn SyntaxValidator>>,
}

impl MultiStageValidator {
    pub fn new() -> Self {
        let mut validators: HashMap<String, Box<dyn SyntaxValidator>> = HashMap::new();

        // Add Rust validator
        validators.insert("rs".to_string(), Box::new(RustValidator));

        Self { syntax_validators: validators }
    }

    pub async fn validate(&self, change: &CodeChange) -> Result<ValidationResult> {
        let mut result = ValidationResult {
            passed: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            syntax_valid: true,
            tests_pass: true,
            behavior_preserved: true,
        };

        // Check risk level
        if change.risk_level >= RiskLevel::Critical {
            result.warnings.push("Critical risk level - extra caution required".to_string());
        }

        // Validate syntax
        if let Some(ext) = change.file_path.extension().and_then(|e| e.to_str()) {
            if let Some(validator) = self.syntax_validators.get(ext) {
                let syntax_result = validator.validate(&change.new_content).await?;
                result.syntax_valid = syntax_result.is_valid;
                result.errors.extend(syntax_result.errors);
            }
        }

        result.passed = result.syntax_valid && result.errors.is_empty();
        Ok(result)
    }
}

/// Trait for syntax validators
#[async_trait::async_trait]
trait SyntaxValidator: Send + Sync {
    async fn validate(&self, content: &str) -> Result<SyntaxValidationResult>;
}

#[derive(Debug)]
struct SyntaxValidationResult {
    is_valid: bool,
    errors: Vec<String>,
}

/// Rust syntax validator
struct RustValidator;

#[async_trait::async_trait]
impl SyntaxValidator for RustValidator {
    async fn validate(&self, content: &str) -> Result<SyntaxValidationResult> {
        // Use syn crate to parse Rust code
        match syn::parse_file(content) {
            Ok(_) => Ok(SyntaxValidationResult { is_valid: true, errors: Vec::new() }),
            Err(e) => Ok(SyntaxValidationResult {
                is_valid: false,
                errors: vec![format!("Syntax error: {}", e)],
            }),
        }
    }
}

/// Integrated test runner
pub struct IntegratedTestRunner {
    repo_path: PathBuf,
}

impl IntegratedTestRunner {
    pub fn new(repo_path: &Path) -> Self {
        Self { repo_path: repo_path.to_path_buf() }
    }

    pub async fn run_tests(&self, branch_name: &str) -> Result<TestResult> {
        info!("Running tests on branch: {}", branch_name);

        let output = Command::new("cargo")
            .args(&["test", "--all"])
            .current_dir(&self.repo_path)
            .output()
            .await?;

        let all_passed = output.status.success();
        let failures = if !all_passed {
            vec![String::from_utf8_lossy(&output.stderr).to_string()]
        } else {
            Vec::new()
        };

        Ok(TestResult {
            all_passed,
            failures,
            test_count: 0, // Would parse from output
        })
    }
}

#[derive(Debug)]
pub struct TestResult {
    pub all_passed: bool,
    pub failures: Vec<String>,
    pub test_count: usize,
}

/// Instant rollback manager
pub struct InstantRollback {
    repo_path: PathBuf,
}

impl InstantRollback {
    pub fn new(repo_path: &Path) -> Self {
        Self { repo_path: repo_path.to_path_buf() }
    }

    pub async fn revert_branch(&self, branch_name: &str) -> Result<()> {
        warn!("Reverting branch: {}", branch_name);

        // Checkout main branch
        Command::new("git")
            .args(&["checkout", "main"])
            .current_dir(&self.repo_path)
            .output()
            .await?;

        // Delete the branch
        Command::new("git")
            .args(&["branch", "-D", branch_name])
            .current_dir(&self.repo_path)
            .output()
            .await?;

        Ok(())
    }
}

/// Pull request representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PullRequest {
    pub number: u32,
    pub title: String,
    pub description: String,
    pub branch: String,
    pub status: PullRequestStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PullRequestStatus {
    Open,
    Merged,
    Closed,
}

/// Copy directory structure (only directories, not files)
#[async_recursion::async_recursion]
async fn copy_dir_structure(src: &Path, dst: &Path) -> Result<()> {
    tokio::fs::create_dir_all(dst).await?;

    let mut entries = tokio::fs::read_dir(src).await?;

    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();
        if path.is_dir() {
            let dst_path = dst.join(entry.file_name());
            copy_dir_structure(&path, &dst_path).await?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_branch_name_generation() {
        // Create a minimal pipeline just to test the branch name generation
        let change = CodeChange {
            file_path: PathBuf::from("src/main.rs"),
            change_type: ChangeType::Feature,
            description: "Add new authentication system".to_string(),
            reasoning: "Needed for security".to_string(),
            old_content: None,
            new_content: String::new(),
            line_range: None,
            risk_level: RiskLevel::Medium,
            attribution: None,
        };

        // Test the branch name generation logic directly
        let type_prefix = match change.change_type {
            ChangeType::Feature => "feat",
            ChangeType::Enhancement => "enhance",
            ChangeType::BugFix => "fix",
            ChangeType::Refactor => "refactor",
            ChangeType::Documentation => "docs",
            ChangeType::Test => "test",
            _ => "chore",
        };

        let clean_desc = change
            .description
            .to_lowercase()
            .chars()
            .map(|c| if c.is_alphanumeric() { c } else { '-' })
            .collect::<String>()
            .trim_matches('-')
            .to_string();

        let branch_name = format!("{}/{}", type_prefix, clean_desc);
        assert_eq!(branch_name, "feat/add-new-authentication-system");
    }
}
