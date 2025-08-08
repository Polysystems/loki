//! Integration tests for complete Story-Driven workflows
//!
//! These tests verify end-to-end scenarios and complex workflows
//! that involve multiple subsystems working together.

use anyhow::Result;
use loki::cognitive::{
    StoryDrivenAutonomy, StoryDrivenAutonomyConfig,
    MaintenanceStatus, MaintenanceTaskType,
};
use loki::cognitive::autonomous_loop::AutonomousLoop;
use loki::cognitive::self_modify::{SelfModificationPipeline, RiskLevel};
use loki::cognitive::test_generator::TestGenerator;
use loki::cognitive::task_automation::CodeReviewTask;
use loki::memory::{CognitiveMemory, MemoryConfig};
use loki::story::{StoryEngine, PlotType, PlotPoint};
use loki::tools::code_analysis::CodeAnalyzer;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tempfile::TempDir;
use tokio::sync::RwLock;
use tokio::time::timeout;

/// Create a complete test project
async fn create_test_project(temp_dir: &TempDir) -> Result<()> {
    // Cargo.toml
    let cargo_toml = r#"
[package]
name = "test-project"
version = "0.1.0"
edition = "2021"

[dependencies]
anyhow = "1.0"
serde = { version = "1.0", features = ["derive"] }
tokio = { version = "1.0", features = ["full"] }
reqwest = "0.11"

[dev-dependencies]
mockito = "0.31"
"#;
    
    // Main library file
    let lib_rs = r#"
//! Test project library

use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub id: u64,
    pub username: String,
    pub email: String,
}

pub struct UserService {
    api_url: String,
}

impl UserService {
    pub fn new(api_url: String) -> Self {
        Self { api_url }
    }
    
    pub async fn get_user(&self, id: u64) -> Result<User> {
        let url = format!("{}/users/{}", self.api_url, id);
        let response = reqwest::get(&url).await?;
        let user = response.json().await?;
        Ok(user)
    }
    
    pub fn validate_email(email: &str) -> bool {
        email.contains('@') && email.contains('.')
    }
}

pub fn calculate_discount(amount: f64, user_type: &str) -> f64 {
    match user_type {
        "premium" => amount * 0.2,
        "regular" => amount * 0.1,
        _ => 0.0,
    }
}
"#;
    
    // API module with issues
    let api_rs = r#"
use crate::User;
use std::collections::HashMap;

pub struct ApiHandler {
    users: HashMap<u64, User>,
}

impl ApiHandler {
    pub fn new() -> Self {
        Self {
            users: HashMap::new(),
        }
    }
    
    pub fn handle_request(&self, path: &str) -> String {
        if path.starts_with("/users/") {
            let id_str = path.trim_start_matches("/users/");
            let id = id_str.parse::<u64>().unwrap(); // Bug: unwrap
            
            if let Some(user) = self.users.get(&id) {
                format!("User: {} ({})", user.username, user.email)
            } else {
                "User not found".to_string()
            }
        } else {
            "Invalid path".to_string()
        }
    }
    
    pub fn add_user(&mut self, user: User) {
        self.users.insert(user.id, user);
    }
}

// Duplicate code that should be refactored
pub fn format_user_info(user: &User) -> String {
    format!("User: {} ({})", user.username, user.email)
}
"#;
    
    // Utils module with performance issues
    let utils_rs = r#"
pub fn process_data(items: Vec<String>) -> Vec<String> {
    let mut results = Vec::new();
    for item in items {
        let processed = item.clone(); // Performance: unnecessary clone
        results.push(processed.to_uppercase());
    }
    results
}

pub fn find_duplicates(items: &[String]) -> Vec<String> {
    let mut duplicates = Vec::new();
    for i in 0..items.len() {
        for j in i+1..items.len() {
            if items[i] == items[j] && !duplicates.contains(&items[i]) {
                duplicates.push(items[i].clone());
            }
        }
    }
    duplicates  // O(n²) complexity
}
"#;
    
    // Create directory structure
    tokio::fs::create_dir_all(temp_dir.path().join("src")).await?;
    
    // Write files
    tokio::fs::write(temp_dir.path().join("Cargo.toml"), cargo_toml).await?;
    tokio::fs::write(temp_dir.path().join("src/lib.rs"), lib_rs).await?;
    tokio::fs::write(temp_dir.path().join("src/api.rs"), api_rs).await?;
    tokio::fs::write(temp_dir.path().join("src/utils.rs"), utils_rs).await?;
    
    Ok(())
}

#[tokio::test]
async fn test_complete_maintenance_workflow() -> Result<()> {
    let temp_dir = TempDir::new()?;
    create_test_project(&temp_dir).await?;
    
    // Initialize components
    let memory_config = MemoryConfig {
        persist_to_disk: false,
        ..Default::default()
    };
    let memory = Arc::new(CognitiveMemory::new(memory_config).await?);
    let story_engine = Arc::new(StoryEngine::new(memory.clone()).await?);
    let autonomous_loop = Arc::new(RwLock::new(
        AutonomousLoop::new(story_engine.clone(), memory.clone()).await?
    ));
    
    let self_modify = Arc::new(SelfModificationPipeline::new(
        story_engine.clone(),
        memory.clone(),
    ));
    let code_analyzer = Arc::new(CodeAnalyzer::new(None));
    let test_generator = Arc::new(TestGenerator::new(code_analyzer.clone()));
    let code_review = Arc::new(CodeReviewTask::new(code_analyzer.clone()));
    
    // Configure autonomy for comprehensive maintenance
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
    
    let autonomy = Arc::new(
        StoryDrivenAutonomy::new(
            config,
            story_engine.clone(),
            autonomous_loop,
            None,
            self_modify,
            code_analyzer,
            test_generator,
            code_review,
            memory.clone(),
        )
        .await?
    );
    
    // Run complete maintenance cycle
    autonomy.run_maintenance_cycle().await?;
    
    // Wait for tasks to be created
    tokio::time::sleep(Duration::from_millis(100)).await;
    
    // Get generated tasks
    let tasks = autonomy.get_active_tasks().await?;
    println!("Generated {} maintenance tasks", tasks.len());
    
    // Verify various task types were created
    let task_types: std::collections::HashSet<_> = tasks.iter()
        .map(|t| t.task_type.clone())
        .collect();
    
    assert!(!tasks.is_empty(), "Should generate maintenance tasks");
    
    // Should detect the unwrap bug
    let has_bug_fix = tasks.iter().any(|t| 
        matches!(t.task_type, MaintenanceTaskType::BugFix) &&
        (t.description.contains("unwrap") || t.description.contains("error handling"))
    );
    assert!(has_bug_fix, "Should detect unwrap bug");
    
    // Should suggest refactoring for duplicate code
    let has_refactoring = tasks.iter().any(|t|
        matches!(t.task_type, MaintenanceTaskType::Refactoring)
    );
    assert!(has_refactoring, "Should suggest refactoring");
    
    // Should suggest test generation
    let has_test_gen = tasks.iter().any(|t|
        matches!(t.task_type, MaintenanceTaskType::TestGeneration)
    );
    assert!(has_test_gen, "Should suggest test generation");
    
    // Execute high-priority tasks
    let executed = autonomy.execute_pending_tasks(5).await?;
    assert!(executed > 0, "Should execute some tasks");
    
    // Verify story progression
    let plot_points = story_engine
        .get_plot_points(&autonomy.get_codebase_story_id())
        .await?;
    
    // Should have various types of plot points
    let has_analysis = plot_points.iter().any(|p| 
        matches!(p.plot_type, PlotType::Analysis { .. })
    );
    let has_action = plot_points.iter().any(|p|
        matches!(p.plot_type, PlotType::Action { .. })
    );
    
    assert!(has_analysis, "Should record analysis activities");
    assert!(has_action, "Should record actions taken");
    
    Ok(())
}

#[tokio::test]
async fn test_incremental_improvement_workflow() -> Result<()> {
    let temp_dir = TempDir::new()?;
    create_test_project(&temp_dir).await?;
    
    // Initialize autonomy system
    let memory = Arc::new(CognitiveMemory::new(MemoryConfig::default()).await?);
    let story_engine = Arc::new(StoryEngine::new(memory.clone()).await?);
    let autonomous_loop = Arc::new(RwLock::new(
        AutonomousLoop::new(story_engine.clone(), memory.clone()).await?
    ));
    
    let config = StoryDrivenAutonomyConfig {
        enable_codebase_maintenance: true,
        enable_quality_monitoring: true,
        enable_refactoring: true,
        enable_testing: true,
        maintenance_interval: Duration::from_secs(60),
        repo_path: temp_dir.path().to_path_buf(),
        max_risk_level: RiskLevel::Low, // Conservative approach
        ..Default::default()
    };
    
    let autonomy = Arc::new(
        StoryDrivenAutonomy::new(
            config,
            story_engine.clone(),
            autonomous_loop,
            None,
            Arc::new(SelfModificationPipeline::new(story_engine.clone(), memory.clone())),
            Arc::new(CodeAnalyzer::new(None)),
            Arc::new(TestGenerator::new(Arc::new(CodeAnalyzer::new(None)))),
            Arc::new(CodeReviewTask::new(Arc::new(CodeAnalyzer::new(None)))),
            memory.clone(),
        )
        .await?
    );
    
    // Measure initial quality
    let initial_quality = autonomy.analyze_code_quality().await?;
    println!("Initial quality score: {:.2}", initial_quality.overall_health);
    
    // Run multiple improvement cycles
    for cycle in 1..=3 {
        println!("\n--- Improvement Cycle {} ---", cycle);
        
        // Run maintenance
        autonomy.run_maintenance_cycle().await?;
        
        // Execute tasks
        let executed = autonomy.execute_pending_tasks(3).await?;
        println!("Executed {} tasks", executed);
        
        // Check quality improvement
        let current_quality = autonomy.analyze_code_quality().await?;
        println!("Quality score after cycle {}: {:.2}", cycle, current_quality.overall_health);
        
        // Quality should improve or maintain
        assert!(
            current_quality.overall_health >= initial_quality.overall_health * 0.95,
            "Quality should not degrade significantly"
        );
    }
    
    // Final quality check
    let final_quality = autonomy.analyze_code_quality().await?;
    
    // Should show improvement
    assert!(
        final_quality.overall_health >= initial_quality.overall_health,
        "Quality should improve after multiple cycles"
    );
    
    // Check specific improvements
    assert!(
        final_quality.issues.len() <= initial_quality.issues.len(),
        "Should have fewer or equal issues"
    );
    
    Ok(())
}

#[tokio::test]
async fn test_emergency_response_workflow() -> Result<()> {
    let temp_dir = TempDir::new()?;
    
    // Create project with critical security issue
    let vulnerable_code = r#"
use std::process::Command;

pub fn execute_command(user_input: &str) -> String {
    // CRITICAL: Command injection vulnerability
    let output = Command::new("sh")
        .arg("-c")
        .arg(user_input)  // Direct user input!
        .output()
        .expect("Failed to execute command");
    
    String::from_utf8_lossy(&output.stdout).to_string()
}

pub fn read_file(path: &str) -> String {
    // Path traversal vulnerability
    std::fs::read_to_string(path).unwrap()
}
"#;
    
    tokio::fs::create_dir_all(temp_dir.path().join("src")).await?;
    tokio::fs::write(temp_dir.path().join("src/lib.rs"), vulnerable_code).await?;
    
    // Initialize autonomy with security focus
    let memory = Arc::new(CognitiveMemory::new(MemoryConfig::default()).await?);
    let story_engine = Arc::new(StoryEngine::new(memory.clone()).await?);
    
    let config = StoryDrivenAutonomyConfig {
        enable_bug_fixing: true,
        enable_quality_monitoring: true,
        max_risk_level: RiskLevel::High, // Allow high-risk fixes for security
        repo_path: temp_dir.path().to_path_buf(),
        ..Default::default()
    };
    
    let autonomy = Arc::new(
        StoryDrivenAutonomy::new(
            config,
            story_engine.clone(),
            Arc::new(RwLock::new(AutonomousLoop::new(story_engine.clone(), memory.clone()).await?)),
            None,
            Arc::new(SelfModificationPipeline::new(story_engine.clone(), memory.clone())),
            Arc::new(CodeAnalyzer::new(None)),
            Arc::new(TestGenerator::new(Arc::new(CodeAnalyzer::new(None)))),
            Arc::new(CodeReviewTask::new(Arc::new(CodeAnalyzer::new(None)))),
            memory.clone(),
        )
        .await?
    );
    
    // Run emergency security scan
    autonomy.run_maintenance_cycle().await?;
    
    // Get tasks
    let tasks = autonomy.get_active_tasks().await?;
    
    // Should create high-priority security tasks
    let security_tasks: Vec<_> = tasks.iter()
        .filter(|t| matches!(t.task_type, MaintenanceTaskType::SecurityPatch))
        .collect();
    
    assert!(!security_tasks.is_empty(), "Should create security tasks");
    
    // All security tasks should be high priority
    for task in &security_tasks {
        assert!(task.priority > 0.8, "Security tasks should be high priority");
    }
    
    // Execute security fixes immediately
    let executed = autonomy.execute_pending_tasks(10).await?;
    assert!(executed > 0, "Should execute security fixes");
    
    // Verify story records emergency response
    let plot_points = story_engine
        .get_plot_points(&autonomy.get_codebase_story_id())
        .await?;
    
    let has_security_action = plot_points.iter().any(|p|
        match &p.plot_type {
            PlotType::Action { action_type, .. } => action_type.contains("security"),
            _ => false,
        }
    );
    
    assert!(has_security_action, "Should record security actions");
    
    Ok(())
}

#[tokio::test]
async fn test_continuous_learning_workflow() -> Result<()> {
    let temp_dir = TempDir::new()?;
    
    // Create evolving codebase
    let phase1_code = r#"
pub fn calculate(a: i32, b: i32) -> i32 {
    a + b
}
"#;
    
    tokio::fs::create_dir_all(temp_dir.path().join("src")).await?;
    tokio::fs::write(temp_dir.path().join("src/lib.rs"), phase1_code).await?;
    
    // Initialize autonomy with learning focus
    let memory = Arc::new(CognitiveMemory::new(MemoryConfig::default()).await?);
    let story_engine = Arc::new(StoryEngine::new(memory.clone()).await?);
    
    let config = StoryDrivenAutonomyConfig {
        enable_pattern_learning: true,
        enable_code_generation: true,
        repo_path: temp_dir.path().to_path_buf(),
        ..Default::default()
    };
    
    let autonomy = Arc::new(
        StoryDrivenAutonomy::new(
            config,
            story_engine.clone(),
            Arc::new(RwLock::new(AutonomousLoop::new(story_engine.clone(), memory.clone()).await?)),
            None,
            Arc::new(SelfModificationPipeline::new(story_engine.clone(), memory.clone())),
            Arc::new(CodeAnalyzer::new(None)),
            Arc::new(TestGenerator::new(Arc::new(CodeAnalyzer::new(None)))),
            Arc::new(CodeReviewTask::new(Arc::new(CodeAnalyzer::new(None)))),
            memory.clone(),
        )
        .await?
    );
    
    // Phase 1: Learn from simple code
    autonomy.learn_from_codebase().await?;
    let initial_patterns = autonomy.get_learned_patterns().await?;
    
    // Phase 2: Add more complex patterns
    let phase2_code = r#"
use anyhow::{Result, Context};

pub fn calculate(a: i32, b: i32) -> i32 {
    a + b
}

pub fn safe_divide(a: i32, b: i32) -> Result<i32> {
    if b == 0 {
        anyhow::bail!("Division by zero");
    }
    Ok(a / b)
}

pub struct Calculator {
    history: Vec<i32>,
}

impl Calculator {
    pub fn new() -> Self {
        Self {
            history: Vec::new(),
        }
    }
    
    pub fn calculate(&mut self, a: i32, b: i32) -> i32 {
        let result = a + b;
        self.history.push(result);
        result
    }
}
"#;
    
    tokio::fs::write(temp_dir.path().join("src/lib.rs"), phase2_code).await?;
    
    // Learn from enhanced code
    autonomy.learn_from_codebase().await?;
    let evolved_patterns = autonomy.get_learned_patterns().await?;
    
    // Should learn new patterns
    assert!(
        evolved_patterns.len() > initial_patterns.len(),
        "Should learn additional patterns"
    );
    
    // Request new feature using learned patterns
    let story_id = autonomy.get_codebase_story_id();
    story_engine.add_plot_point(
        story_id.clone(),
        PlotType::Goal {
            objective: "Add multiply function following existing patterns".to_string(),
        },
        vec!["feature".to_string()]
    ).await?;
    
    // Generate code using learned patterns
    let generated = autonomy.generate_code_from_story_context(&story_id).await?;
    
    assert!(!generated.is_empty(), "Should generate code");
    
    // Generated code should follow learned patterns
    let has_error_handling = generated.iter().any(|g|
        g.content.contains("Result") && g.content.contains("Context")
    );
    
    assert!(has_error_handling, "Should apply learned error handling pattern");
    
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_parallel_maintenance_workflow() -> Result<()> {
    let temp_dir = TempDir::new()?;
    create_test_project(&temp_dir).await?;
    
    // Initialize autonomy
    let memory = Arc::new(CognitiveMemory::new(MemoryConfig::default()).await?);
    let story_engine = Arc::new(StoryEngine::new(memory.clone()).await?);
    
    let config = StoryDrivenAutonomyConfig {
        enable_codebase_maintenance: true,
        enable_bug_fixing: true,
        enable_testing: true,
        enable_documentation: true,
        enable_refactoring: true,
        repo_path: temp_dir.path().to_path_buf(),
        ..Default::default()
    };
    
    let autonomy = Arc::new(
        StoryDrivenAutonomy::new(
            config,
            story_engine.clone(),
            Arc::new(RwLock::new(AutonomousLoop::new(story_engine.clone(), memory.clone()).await?)),
            None,
            Arc::new(SelfModificationPipeline::new(story_engine.clone(), memory.clone())),
            Arc::new(CodeAnalyzer::new(None)),
            Arc::new(TestGenerator::new(Arc::new(CodeAnalyzer::new(None)))),
            Arc::new(CodeReviewTask::new(Arc::new(CodeAnalyzer::new(None)))),
            memory.clone(),
        )
        .await?
    );
    
    // Run maintenance to generate tasks
    autonomy.run_maintenance_cycle().await?;
    
    // Execute multiple tasks in parallel
    let autonomy_clone = autonomy.clone();
    let handle1 = tokio::spawn(async move {
        autonomy_clone.execute_pending_tasks(2).await
    });
    
    let autonomy_clone = autonomy.clone();
    let handle2 = tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(50)).await;
        autonomy_clone.execute_pending_tasks(2).await
    });
    
    // Wait for both to complete
    let (result1, result2) = tokio::join!(handle1, handle2);
    let executed1 = result1??;
    let executed2 = result2??;
    
    println!("Thread 1 executed: {}, Thread 2 executed: {}", executed1, executed2);
    
    // Verify no conflicts
    let tasks = autonomy.get_active_tasks().await?;
    
    // Check for duplicate executions
    let completed_tasks: Vec<_> = tasks.iter()
        .filter(|t| matches!(t.status, MaintenanceStatus::Completed))
        .collect();
    
    // Each task should be executed only once
    let task_ids: std::collections::HashSet<_> = completed_tasks.iter()
        .map(|t| &t.task_id)
        .collect();
    
    assert_eq!(
        task_ids.len(),
        completed_tasks.len(),
        "No duplicate task executions"
    );
    
    Ok(())
}

/// Helper to wait for condition with timeout
async fn wait_for_condition<F, Fut>(
    condition: F,
    timeout_duration: Duration,
) -> Result<()>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = bool>,
{
    timeout(timeout_duration, async {
        loop {
            if condition().await {
                return Ok(());
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    })
    .await
    .map_err(|_| anyhow::anyhow!("Timeout waiting for condition"))
}