//! Integration tests for Story-Driven Autonomy System
//!
//! These tests verify the complete functionality of the autonomous codebase
//! maintenance system, including all integrated subsystems.

use anyhow::Result;
use loki::cognitive::{
    StoryDrivenAutonomy, StoryDrivenAutonomyConfig,
    StoryDrivenCodeGenerator, GeneratedCode, GeneratedArtifactType,
    StoryDrivenPrReview, StoryDrivenReviewResult,
    StoryDrivenBugDetection, TrackedBug, BugSeverity,
    StoryDrivenTesting, TestGenerationStrategy,
    StoryDrivenLearning, LearnedPattern,
    StoryDrivenDocumentation, DocumentationType,
    StoryDrivenDependencies, DependencyAnalysis,
    StoryDrivenQuality, QualityMetrics,
    StoryDrivenRefactoring, RefactoringType,
    MaintenanceStatus, PatternType,
};
use loki::cognitive::autonomous_loop::AutonomousLoop;
use loki::cognitive::self_modify::{SelfModificationPipeline, RiskLevel};
use loki::cognitive::test_generator::TestGenerator;
use loki::cognitive::task_automation::CodeReviewTask;
use loki::memory::{CognitiveMemory, MemoryConfig};
use loki::story::{StoryEngine, PlotType};
use loki::tools::code_analysis::CodeAnalyzer;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tempfile::TempDir;
use tokio::sync::RwLock;

/// Test fixture for story-driven autonomy tests
struct TestFixture {
    autonomy: Arc<StoryDrivenAutonomy>,
    story_engine: Arc<StoryEngine>,
    memory: Arc<CognitiveMemory>,
    temp_dir: TempDir,
}

impl TestFixture {
    async fn new() -> Result<Self> {
        let temp_dir = TempDir::new()?;
        
        // Initialize memory
        let memory_config = MemoryConfig {
            persist_to_disk: false,
            ..Default::default()
        };
        let memory = Arc::new(CognitiveMemory::new(memory_config).await?);
        
        // Initialize story engine
        let story_engine = Arc::new(StoryEngine::new(memory.clone()).await?);
        
        // Initialize autonomous loop
        let autonomous_loop = Arc::new(RwLock::new(
            AutonomousLoop::new(story_engine.clone(), memory.clone()).await?
        ));
        
        // Initialize other components
        let self_modify = Arc::new(SelfModificationPipeline::new(
            story_engine.clone(),
            memory.clone(),
        ));
        let code_analyzer = Arc::new(CodeAnalyzer::new(None));
        let test_generator = Arc::new(TestGenerator::new(code_analyzer.clone()));
        let code_review = Arc::new(CodeReviewTask::new(code_analyzer.clone()));
        
        // Configure autonomy
        let config = StoryDrivenAutonomyConfig {
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
            maintenance_interval: Duration::from_secs(60),
            repo_path: temp_dir.path().to_path_buf(),
            max_risk_level: RiskLevel::Medium,
        };
        
        // Create autonomy system
        let autonomy = Arc::new(
            StoryDrivenAutonomy::new(
                config,
                story_engine.clone(),
                autonomous_loop,
                None, // PR automation
                self_modify,
                code_analyzer,
                test_generator,
                code_review,
                memory.clone(),
            )
            .await?
        );
        
        Ok(Self {
            autonomy,
            story_engine,
            memory,
            temp_dir,
        })
    }
    
    /// Create a test file in the temp directory
    async fn create_test_file(&self, path: &str, content: &str) -> Result<PathBuf> {
        let file_path = self.temp_dir.path().join(path);
        if let Some(parent) = file_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        tokio::fs::write(&file_path, content).await?;
        Ok(file_path)
    }
}

#[tokio::test]
async fn test_story_driven_autonomy_initialization() -> Result<()> {
    let fixture = TestFixture::new().await?;
    
    // Verify story was created
    let stories = fixture.story_engine.list_stories().await?;
    assert!(!stories.is_empty(), "Should have created at least one story");
    
    // Verify codebase story exists
    let codebase_story = stories.iter()
        .find(|s| s.story_type == loki::story::StoryType::Codebase)
        .expect("Should have a codebase story");
    
    assert_eq!(codebase_story.language, "rust");
    
    Ok(())
}

#[tokio::test]
async fn test_code_generation_from_story() -> Result<()> {
    let fixture = TestFixture::new().await?;
    
    // Create a story context for code generation
    let story_id = fixture.story_engine
        .create_codebase_story(
            fixture.temp_dir.path().to_path_buf(),
            "rust".to_string()
        )
        .await?;
    
    // Add a plot point requesting feature implementation
    fixture.story_engine
        .add_plot_point(
            story_id.clone(),
            PlotType::Goal {
                objective: "Implement a configuration parser".to_string(),
            },
            vec!["feature".to_string(), "config".to_string()]
        )
        .await?;
    
    // Trigger code generation
    let generated = fixture.autonomy.generate_code_from_story_context(&story_id).await?;
    
    assert!(!generated.is_empty(), "Should generate some code");
    
    // Verify generated code contains expected elements
    let config_parser = generated.iter()
        .find(|g| g.artifact_type == GeneratedArtifactType::Implementation)
        .expect("Should have generated implementation");
    
    assert!(config_parser.content.contains("fn parse"), "Should contain parse function");
    assert!(config_parser.file_path.to_str().unwrap().contains("config"), "Should be in config module");
    
    Ok(())
}

#[tokio::test]
async fn test_bug_detection_and_fixing() -> Result<()> {
    let fixture = TestFixture::new().await?;
    
    // Create a file with a known bug pattern
    let buggy_code = r#"
fn divide(a: i32, b: i32) -> i32 {
    a / b  // Bug: No zero check
}

fn get_item(items: &Vec<String>, index: usize) -> &String {
    &items[index]  // Bug: No bounds check
}

fn parse_number(s: &str) -> i32 {
    s.parse().unwrap()  // Bug: Unwrap can panic
}
"#;
    
    fixture.create_test_file("src/buggy.rs", buggy_code).await?;
    
    // Run bug detection
    let bugs = fixture.autonomy.detect_bugs().await?;
    
    assert!(bugs.len() >= 2, "Should detect at least 2 bugs");
    
    // Check bug types
    let has_zero_check = bugs.iter().any(|b| b.description.contains("zero") || b.description.contains("division"));
    let has_bounds_check = bugs.iter().any(|b| b.description.contains("bounds") || b.description.contains("index"));
    
    assert!(has_zero_check, "Should detect division by zero bug");
    assert!(has_bounds_check, "Should detect bounds check bug");
    
    // Test bug fixing
    if let Some(bug) = bugs.first() {
        let fix_result = fixture.autonomy.fix_bug(&bug.bug_id).await?;
        assert!(fix_result.success, "Should successfully fix bug");
        assert!(!fix_result.changes.is_empty(), "Should have made changes");
    }
    
    Ok(())
}

#[tokio::test]
async fn test_test_generation() -> Result<()> {
    let fixture = TestFixture::new().await?;
    
    // Create a simple function to test
    let code = r#"
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

pub fn is_even(n: i32) -> bool {
    n % 2 == 0
}

pub struct Calculator {
    value: i32,
}

impl Calculator {
    pub fn new(initial: i32) -> Self {
        Self { value: initial }
    }
    
    pub fn add(&mut self, n: i32) {
        self.value += n;
    }
    
    pub fn get_value(&self) -> i32 {
        self.value
    }
}
"#;
    
    let file_path = fixture.create_test_file("src/lib.rs", code).await?;
    
    // Generate tests
    let test_result = fixture.autonomy.generate_tests_for_file(&file_path).await?;
    
    assert!(test_result.tests_generated > 0, "Should generate tests");
    assert!(test_result.success, "Test generation should succeed");
    
    // Verify test content
    if let Some(test_content) = test_result.test_file_content {
        assert!(test_content.contains("fn test_add"), "Should have test for add function");
        assert!(test_content.contains("fn test_is_even"), "Should have test for is_even");
        assert!(test_content.contains("Calculator::new"), "Should test Calculator");
    }
    
    Ok(())
}

#[tokio::test]
async fn test_pattern_learning() -> Result<()> {
    let fixture = TestFixture::new().await?;
    
    // Create files with common patterns
    let error_pattern = r#"
use anyhow::{Result, Context};

pub fn read_config(path: &str) -> Result<Config> {
    let content = std::fs::read_to_string(path)
        .context("Failed to read config file")?;
    let config = serde_json::from_str(&content)
        .context("Failed to parse config")?;
    Ok(config)
}
"#;
    
    let builder_pattern = r#"
pub struct ServerBuilder {
    host: String,
    port: u16,
    threads: usize,
}

impl ServerBuilder {
    pub fn new() -> Self {
        Self {
            host: "localhost".to_string(),
            port: 8080,
            threads: 4,
        }
    }
    
    pub fn host(mut self, host: String) -> Self {
        self.host = host;
        self
    }
    
    pub fn port(mut self, port: u16) -> Self {
        self.port = port;
        self
    }
    
    pub fn build(self) -> Server {
        Server {
            host: self.host,
            port: self.port,
            threads: self.threads,
        }
    }
}
"#;
    
    fixture.create_test_file("src/config.rs", error_pattern).await?;
    fixture.create_test_file("src/server.rs", builder_pattern).await?;
    
    // Learn patterns
    let learning_result = fixture.autonomy.learn_from_codebase().await?;
    
    assert!(learning_result.patterns_found > 0, "Should find patterns");
    assert!(learning_result.new_patterns > 0, "Should learn new patterns");
    
    // Check learned patterns
    let patterns = fixture.autonomy.get_learned_patterns().await?;
    
    let has_error_handling = patterns.iter().any(|p| 
        matches!(p.pattern_type, PatternType::ErrorHandling)
    );
    let has_builder = patterns.iter().any(|p| 
        matches!(p.pattern_type, PatternType::ArchitecturalPattern)
    );
    
    assert!(has_error_handling, "Should learn error handling pattern");
    assert!(has_builder, "Should learn builder pattern");
    
    Ok(())
}

#[tokio::test]
async fn test_documentation_generation() -> Result<()> {
    let fixture = TestFixture::new().await?;
    
    // Create undocumented code
    let code = r#"
pub struct User {
    id: u64,
    name: String,
    email: String,
}

impl User {
    pub fn new(id: u64, name: String, email: String) -> Self {
        Self { id, name, email }
    }
    
    pub fn validate_email(&self) -> bool {
        self.email.contains('@') && self.email.contains('.')
    }
    
    pub async fn save(&self) -> Result<()> {
        // Save to database
        Ok(())
    }
}

pub fn hash_password(password: &str) -> String {
    // Simple hash for demo
    format!("hashed_{}", password)
}
"#;
    
    fixture.create_test_file("src/user.rs", code).await?;
    
    // Generate documentation
    let docs = fixture.autonomy.generate_missing_documentation().await?;
    
    assert!(!docs.is_empty(), "Should generate documentation");
    
    // Check documentation content
    let user_doc = docs.iter()
        .find(|d| d.file_path.to_str().unwrap().contains("user.rs"))
        .expect("Should have documentation for user.rs");
    
    assert!(user_doc.content.contains("/// User"), "Should document User struct");
    assert!(user_doc.content.contains("validate_email"), "Should document methods");
    assert!(user_doc.content.contains("# Examples"), "Should include examples");
    
    Ok(())
}

#[tokio::test]
async fn test_quality_monitoring() -> Result<()> {
    let fixture = TestFixture::new().await?;
    
    // Create files with varying quality
    let high_quality = r#"
//! High quality module with good practices

use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};

/// Configuration for the application
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    /// Server host
    pub host: String,
    /// Server port
    pub port: u16,
}

impl AppConfig {
    /// Create new configuration with defaults
    pub fn new() -> Self {
        Self {
            host: "localhost".to_string(),
            port: 8080,
        }
    }
    
    /// Load configuration from file
    pub fn from_file(path: &str) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .context("Failed to read config file")?;
        serde_json::from_str(&content)
            .context("Failed to parse config")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_default_config() {
        let config = AppConfig::new();
        assert_eq!(config.host, "localhost");
        assert_eq!(config.port, 8080);
    }
}
"#;
    
    let low_quality = r#"
fn do_stuff(x: i32, y: i32, z: i32, a: i32, b: i32) -> i32 {
    let mut result = 0;
    if x > 0 {
        if y > 0 {
            if z > 0 {
                if a > 0 {
                    if b > 0 {
                        result = x + y + z + a + b;
                    } else {
                        result = x + y + z + a;
                    }
                } else {
                    result = x + y + z;
                }
            } else {
                result = x + y;
            }
        } else {
            result = x;
        }
    }
    result
}

fn another_long_function() {
    println!("1");
    println!("2");
    println!("3");
    // ... imagine 100 more lines
}
"#;
    
    fixture.create_test_file("src/good.rs", high_quality).await?;
    fixture.create_test_file("src/bad.rs", low_quality).await?;
    
    // Analyze quality
    let quality_analysis = fixture.autonomy.analyze_code_quality().await?;
    
    assert!(quality_analysis.overall_health > 0.0, "Should have health score");
    assert!(!quality_analysis.issues.is_empty(), "Should find quality issues");
    assert!(!quality_analysis.hotspots.is_empty(), "Should identify hotspots");
    
    // Check for specific issues
    let has_complexity_issue = quality_analysis.issues.iter()
        .any(|i| i.issue_type == loki::cognitive::story_driven_quality::QualityIssueType::HighComplexity);
    
    assert!(has_complexity_issue, "Should detect high complexity");
    
    Ok(())
}

#[tokio::test]
async fn test_refactoring_suggestions() -> Result<()> {
    let fixture = TestFixture::new().await?;
    
    // Create code that needs refactoring
    let code = r#"
fn process_data(input: Vec<String>) -> Vec<String> {
    let mut results = Vec::new();
    for item in input {
        // Long processing logic that should be extracted
        let trimmed = item.trim();
        let uppercase = trimmed.to_uppercase();
        let with_prefix = format!("PROCESSED_{}", uppercase);
        let with_suffix = format!("{}_DONE", with_prefix);
        
        // Validation that should be extracted
        if with_suffix.len() > 100 {
            continue;
        }
        if with_suffix.contains("INVALID") {
            continue;
        }
        
        results.push(with_suffix);
    }
    
    // Duplicate logic
    for i in 0..results.len() {
        let item = results[i].clone();
        results[i] = item.replace("_", "-");
    }
    
    results
}

fn another_function() {
    // Similar duplicate logic
    let mut data = vec!["test_one".to_string(), "test_two".to_string()];
    for i in 0..data.len() {
        let item = data[i].clone();
        data[i] = item.replace("_", "-");
    }
}
"#;
    
    fixture.create_test_file("src/refactor_me.rs", code).await?;
    
    // Get refactoring suggestions
    let refactoring_analysis = fixture.autonomy.analyze_for_refactoring().await?;
    
    assert!(!refactoring_analysis.suggestions.is_empty(), "Should have refactoring suggestions");
    
    // Check suggestion types
    let has_extract_method = refactoring_analysis.suggestions.iter()
        .any(|s| s.refactoring_type == RefactoringType::ExtractMethod);
    let has_consolidation = refactoring_analysis.suggestions.iter()
        .any(|s| s.refactoring_type == RefactoringType::ConsolidateDuplication);
    
    assert!(has_extract_method, "Should suggest method extraction");
    assert!(has_consolidation || has_extract_method, "Should suggest consolidation or extraction");
    
    Ok(())
}

#[tokio::test]
async fn test_maintenance_cycle() -> Result<()> {
    let fixture = TestFixture::new().await?;
    
    // Create a small codebase
    let lib_code = r#"
pub fn calculate(x: i32, y: i32) -> i32 {
    x + y
}
"#;
    
    fixture.create_test_file("src/lib.rs", lib_code).await?;
    
    // Run a maintenance cycle
    fixture.autonomy.run_maintenance_cycle().await?;
    
    // Check that tasks were created
    let tasks = fixture.autonomy.get_active_tasks().await?;
    
    // Should have created some maintenance tasks
    assert!(!tasks.is_empty(), "Should create maintenance tasks");
    
    // Check story was updated
    let plot_points = fixture.story_engine
        .get_plot_points(&fixture.autonomy.get_codebase_story_id())
        .await?;
    
    // Should have recorded maintenance activities
    let has_maintenance_activity = plot_points.iter()
        .any(|p| matches!(p.plot_type, PlotType::Analysis { .. }) || 
                 matches!(p.plot_type, PlotType::Action { .. }));
    
    assert!(has_maintenance_activity, "Should record maintenance in story");
    
    Ok(())
}

#[tokio::test]
async fn test_dependency_management() -> Result<()> {
    let fixture = TestFixture::new().await?;
    
    // Create a Cargo.toml with dependencies
    let cargo_toml = r#"
[package]
name = "test-project"
version = "0.1.0"

[dependencies]
serde = "1.0.100"  # Outdated
tokio = { version = "0.2", features = ["full"] }  # Very outdated
anyhow = "1.0"
"#;
    
    fixture.create_test_file("Cargo.toml", cargo_toml).await?;
    
    // Analyze dependencies
    let dep_analysis = fixture.autonomy.analyze_dependencies().await?;
    
    assert!(dep_analysis.total_dependencies > 0, "Should find dependencies");
    assert!(!dep_analysis.outdated_dependencies.is_empty(), "Should find outdated deps");
    
    // Check for outdated deps
    let has_serde = dep_analysis.outdated_dependencies.iter()
        .any(|d| d.name == "serde");
    let has_tokio = dep_analysis.outdated_dependencies.iter()
        .any(|d| d.name == "tokio");
    
    assert!(has_serde || has_tokio, "Should detect outdated dependencies");
    
    Ok(())
}

#[tokio::test]
async fn test_pr_review() -> Result<()> {
    let fixture = TestFixture::new().await?;
    
    // Create a simulated PR diff
    let pr_diff = r#"
diff --git a/src/api.rs b/src/api.rs
index 1234567..abcdefg 100644
--- a/src/api.rs
+++ b/src/api.rs
@@ -10,7 +10,7 @@ pub fn handle_request(req: Request) -> Response {
-    let data = req.body.unwrap();
+    let data = req.body.unwrap();  // TODO: handle None case
     
     Response::ok()
 }
"#;
    
    // Create PR review context
    let review_result = fixture.autonomy.review_pr_diff(pr_diff, "Add TODO comment").await?;
    
    assert_eq!(review_result.overall_assessment, loki::cognitive::story_driven_pr_review::ReviewAssessment::NeedsWork);
    assert!(!review_result.issues.is_empty(), "Should find issues");
    
    // Should detect the unwrap issue
    let has_unwrap_issue = review_result.issues.iter()
        .any(|i| i.description.contains("unwrap") || i.description.contains("error"));
    
    assert!(has_unwrap_issue, "Should detect unwrap issue");
    
    Ok(())
}

#[tokio::test]
async fn test_end_to_end_autonomy() -> Result<()> {
    let fixture = TestFixture::new().await?;
    
    // Create a complete mini project
    let main_rs = r#"
fn main() {
    println!("Hello, world!");
    let result = calculate(5, 3);
    println!("Result: {}", result);
}

fn calculate(a: i32, b: i32) -> i32 {
    a / b  // Bug: no zero check
}
"#;
    
    let lib_rs = r#"
pub struct Config {
    pub name: String,
    pub value: i32,
}

pub fn process(config: &Config) -> String {
    format!("{}: {}", config.name, config.value)
}
"#;
    
    fixture.create_test_file("src/main.rs", main_rs).await?;
    fixture.create_test_file("src/lib.rs", lib_rs).await?;
    
    // Run full autonomy cycle
    fixture.autonomy.run_maintenance_cycle().await?;
    
    // Execute some tasks
    let executed = fixture.autonomy.execute_pending_tasks(3).await?;
    assert!(executed > 0, "Should execute some tasks");
    
    // Verify improvements were made
    let final_quality = fixture.autonomy.analyze_code_quality().await?;
    
    // Check story progression
    let story_points = fixture.story_engine
        .get_plot_points(&fixture.autonomy.get_codebase_story_id())
        .await?;
    
    // Should have multiple types of activities
    let activity_types: std::collections::HashSet<_> = story_points.iter()
        .filter_map(|p| match &p.plot_type {
            PlotType::Action { action_type, .. } => Some(action_type.clone()),
            PlotType::Analysis { subject, .. } => Some(subject.clone()),
            _ => None,
        })
        .collect();
    
    assert!(activity_types.len() >= 2, "Should have multiple activity types");
    
    Ok(())
}

/// Test helper to verify task execution
async fn verify_task_execution(fixture: &TestFixture, task_type: &str) -> Result<bool> {
    let tasks = fixture.autonomy.get_active_tasks().await?;
    let has_task = tasks.iter().any(|t| t.description.contains(task_type));
    Ok(has_task)
}

/// Test helper to create a realistic codebase
async fn create_realistic_codebase(fixture: &TestFixture) -> Result<()> {
    // Create multiple modules with various issues
    let api_module = r#"
use std::collections::HashMap;

pub struct ApiHandler {
    routes: HashMap<String, fn() -> String>,
}

impl ApiHandler {
    pub fn new() -> Self {
        Self {
            routes: HashMap::new(),
        }
    }
    
    pub fn handle(&self, path: &str) -> String {
        self.routes[path]()  // Bug: panics if path not found
    }
}
"#;
    
    let db_module = r#"
pub struct Database {
    connection: String,
}

impl Database {
    pub fn connect(url: &str) -> Self {
        Self {
            connection: url.to_string(),
        }
    }
    
    pub fn query(&self, sql: &str) -> Vec<String> {
        // TODO: implement actual query
        vec![]
    }
}
"#;
    
    fixture.create_test_file("src/api/mod.rs", api_module).await?;
    fixture.create_test_file("src/db/mod.rs", db_module).await?;
    
    Ok(())
}