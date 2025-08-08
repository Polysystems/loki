//! Integration tests for individual Story-Driven subsystems
//!
//! These tests focus on specific subsystems in isolation to ensure
//! each component works correctly before integration.

use anyhow::Result;
use loki::cognitive::story_driven_code_generation::{
    StoryDrivenCodeGenerator, StoryDrivenCodeGenConfig,
    GeneratedCode, GeneratedArtifactType, CodePattern,
};
use loki::cognitive::story_driven_bug_detection::{
    StoryDrivenBugDetection, StoryDrivenBugDetectionConfig,
    BugPattern, BugPatternType, BugSeverity, TrackedBug,
};
use loki::cognitive::story_driven_quality::{
    StoryDrivenQuality, StoryDrivenQualityConfig,
    QualityMetrics, QualityIssueType, IssueSeverity,
};
use loki::cognitive::story_driven_refactoring::{
    StoryDrivenRefactoring, StoryDrivenRefactoringConfig,
    RefactoringType, RefactoringSuggestion, RefactoringImpact,
};
use loki::cognitive::story_driven_learning::{
    StoryDrivenLearning, StoryDrivenLearningConfig,
    LearnedPattern, LearnedPatternType,
};
use loki::cognitive::self_modify::{SelfModificationPipeline, RiskLevel};
use loki::memory::{CognitiveMemory, MemoryConfig};
use loki::story::{StoryEngine, PlotType, PlotPoint};
use loki::tools::code_analysis::CodeAnalyzer;
use std::path::PathBuf;
use std::sync::Arc;
use tempfile::TempDir;

/// Common test setup
async fn setup() -> Result<(Arc<StoryEngine>, Arc<CognitiveMemory>, TempDir)> {
    let temp_dir = TempDir::new()?;
    let memory_config = MemoryConfig {
        persist_to_disk: false,
        ..Default::default()
    };
    let memory = Arc::new(CognitiveMemory::new(memory_config).await?);
    let story_engine = Arc::new(StoryEngine::new(memory.clone()).await?);
    
    Ok((story_engine, memory, temp_dir))
}

#[tokio::test]
async fn test_code_generation_patterns() -> Result<()> {
    let (story_engine, memory, temp_dir) = setup().await?;
    
    let config = StoryDrivenCodeGenConfig {
        enable_pattern_learning: true,
        enable_style_matching: true,
        enable_test_generation: true,
        max_generation_attempts: 3,
        confidence_threshold: 0.7,
        repo_path: temp_dir.path().to_path_buf(),
    };
    
    let code_analyzer = Arc::new(CodeAnalyzer::new(None));
    let generator = StoryDrivenCodeGenerator::new(
        config,
        story_engine.clone(),
        code_analyzer,
        None, // task mapper
        memory,
    ).await?;
    
    // Create a story with specific requirements
    let story_id = story_engine.create_codebase_story(
        temp_dir.path().to_path_buf(),
        "rust".to_string()
    ).await?;
    
    // Add detailed requirements
    story_engine.add_plot_point(
        story_id.clone(),
        PlotType::Goal {
            objective: "Create user authentication endpoint".to_string(),
        },
        vec!["api".to_string(), "auth".to_string(), "rest".to_string()]
    ).await?;
    
    // Generate code
    let generated = generator.generate_code_from_story(&story_id).await?;
    
    assert!(!generated.is_empty(), "Should generate code");
    
    // Verify generated artifacts
    let has_handler = generated.iter().any(|g| 
        g.artifact_type == GeneratedArtifactType::Implementation &&
        g.content.contains("login") &&
        g.content.contains("jwt")
    );
    
    let has_types = generated.iter().any(|g|
        g.artifact_type == GeneratedArtifactType::Types &&
        (g.content.contains("LoginRequest") || g.content.contains("username"))
    );
    
    assert!(has_handler, "Should generate login handler");
    assert!(has_types, "Should generate request/response types");
    
    Ok(())
}

#[tokio::test]
async fn test_bug_detection_patterns() -> Result<()> {
    let (story_engine, memory, temp_dir) = setup().await?;
    
    let config = StoryDrivenBugDetectionConfig {
        enable_pattern_detection: true,
        enable_static_analysis: true,
        enable_anomaly_detection: true,
        enable_auto_fix: true,
        max_auto_fix_risk: RiskLevel::Low,
        scan_interval: std::time::Duration::from_secs(60),
        repo_path: temp_dir.path().to_path_buf(),
    };
    
    let code_analyzer = Arc::new(CodeAnalyzer::new(None));
    let self_modify = Arc::new(SelfModificationPipeline::new(
        story_engine.clone(),
        memory.clone(),
    ));
    
    let bug_detector = StoryDrivenBugDetection::new(
        config,
        story_engine.clone(),
        code_analyzer,
        self_modify,
        None, // test generator
        memory,
    ).await?;
    
    // Add known bug patterns
    bug_detector.add_bug_pattern(BugPattern {
        pattern_id: "unchecked_div".to_string(),
        pattern_type: BugPatternType::LogicError,
        description: "Division without zero check".to_string(),
        detection_regex: r"(\w+)\s*/\s*(\w+)".to_string(),
        severity: BugSeverity::High,
        auto_fixable: true,
        fix_template: Some("if {divisor} != 0 { {dividend} / {divisor} } else { 0 }".to_string()),
    }).await?;
    
    // Create file with bugs
    let buggy_code = r#"
fn calculate_average(numbers: &[i32]) -> i32 {
    let sum: i32 = numbers.iter().sum();
    let count = numbers.len() as i32;
    sum / count  // Bug: division by zero when empty
}

fn get_first(items: &Vec<String>) -> &String {
    &items[0]  // Bug: panic on empty vector
}
"#;
    
    tokio::fs::create_dir_all(temp_dir.path().join("src")).await?;
    tokio::fs::write(
        temp_dir.path().join("src/math.rs"),
        buggy_code
    ).await?;
    
    // Scan for bugs
    let bugs = bug_detector.scan_codebase().await?;
    
    assert!(bugs.len() >= 2, "Should detect multiple bugs");
    
    // Verify bug types
    let has_div_zero = bugs.iter().any(|b| 
        b.bug_type == BugPatternType::LogicError &&
        b.description.contains("division")
    );
    
    let has_bounds = bugs.iter().any(|b|
        b.bug_type == BugPatternType::MemorySafety &&
        b.description.contains("bound")
    );
    
    assert!(has_div_zero, "Should detect division by zero");
    assert!(has_bounds, "Should detect bounds check issue");
    
    // Test auto-fix
    let fixable_bugs: Vec<_> = bugs.iter()
        .filter(|b| b.auto_fixable)
        .collect();
    
    if let Some(bug) = fixable_bugs.first() {
        let fix_result = bug_detector.fix_bug(&bug.bug_id).await?;
        assert!(fix_result.success, "Should fix bug successfully");
    }
    
    Ok(())
}

#[tokio::test]
async fn test_quality_metrics_calculation() -> Result<()> {
    let (story_engine, memory, temp_dir) = setup().await?;
    
    let config = StoryDrivenQualityConfig {
        enable_complexity_monitoring: true,
        enable_duplication_detection: true,
        enable_maintainability_scoring: true,
        enable_performance_analysis: true,
        enable_security_scanning: true,
        enable_style_checking: true,
        complexity_threshold: 10.0,
        duplication_threshold: 5.0,
        min_maintainability_score: 0.7,
        repo_path: temp_dir.path().to_path_buf(),
    };
    
    let code_analyzer = Arc::new(CodeAnalyzer::new(None));
    let quality_monitor = StoryDrivenQuality::new(
        config,
        story_engine.clone(),
        code_analyzer,
        None, // learning system
        memory,
    ).await?;
    
    // Create files with different quality levels
    let high_quality_code = r#"
//! Well-documented module

use std::error::Error;

/// Represents a user in the system
#[derive(Debug, Clone)]
pub struct User {
    id: u64,
    name: String,
}

impl User {
    /// Creates a new user
    pub fn new(id: u64, name: String) -> Self {
        Self { id, name }
    }
    
    /// Gets the user's ID
    pub fn id(&self) -> u64 {
        self.id
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_user_creation() {
        let user = User::new(1, "Test".to_string());
        assert_eq!(user.id(), 1);
    }
}
"#;
    
    let low_quality_code = r#"
fn x(a: i32, b: i32, c: i32, d: i32, e: i32, f: i32) -> i32 {
    if a > 0 {
        if b > 0 {
            if c > 0 {
                if d > 0 {
                    if e > 0 {
                        if f > 0 {
                            return a + b + c + d + e + f;
                        }
                    }
                }
            }
        }
    }
    0
}

fn copy1() { println!("duplicate"); }
fn copy2() { println!("duplicate"); }
fn copy3() { println!("duplicate"); }
"#;
    
    tokio::fs::create_dir_all(temp_dir.path().join("src")).await?;
    tokio::fs::write(
        temp_dir.path().join("src/good.rs"),
        high_quality_code
    ).await?;
    tokio::fs::write(
        temp_dir.path().join("src/bad.rs"),
        low_quality_code
    ).await?;
    
    // Analyze quality
    let analysis = quality_monitor.analyze_quality().await?;
    
    // Verify metrics
    assert!(analysis.current_metrics.overall_health > 0.0);
    assert!(analysis.current_metrics.overall_health <= 1.0);
    
    // Should detect issues
    assert!(!analysis.issues.is_empty(), "Should find quality issues");
    
    // Check for specific issue types
    let has_complexity = analysis.issues.iter().any(|i|
        i.issue_type == QualityIssueType::HighComplexity
    );
    let has_duplication = analysis.issues.iter().any(|i|
        i.issue_type == QualityIssueType::CodeDuplication
    );
    
    assert!(has_complexity, "Should detect high complexity");
    assert!(has_duplication, "Should detect code duplication");
    
    // Verify hotspots
    assert!(!analysis.hotspots.is_empty(), "Should identify hotspots");
    
    let bad_file_hotspot = analysis.hotspots.iter()
        .find(|h| h.file_path.to_str().unwrap().contains("bad.rs"));
    
    assert!(bad_file_hotspot.is_some(), "bad.rs should be a hotspot");
    
    Ok(())
}

#[tokio::test]
async fn test_refactoring_impact_analysis() -> Result<()> {
    let (story_engine, memory, temp_dir) = setup().await?;
    
    let config = StoryDrivenRefactoringConfig {
        enable_method_extraction: true,
        enable_variable_renaming: true,
        enable_consolidation: true,
        enable_pattern_application: true,
        enable_performance_refactoring: true,
        enable_architectural_refactoring: true,
        min_complexity_for_extraction: 8,
        max_method_length: 30,
        max_auto_refactor_risk: RiskLevel::Low,
        repo_path: temp_dir.path().to_path_buf(),
    };
    
    let code_analyzer = Arc::new(CodeAnalyzer::new(None));
    let self_modify = Arc::new(SelfModificationPipeline::new(
        story_engine.clone(),
        memory.clone(),
    ));
    
    let refactoring_system = StoryDrivenRefactoring::new(
        config,
        story_engine.clone(),
        code_analyzer,
        self_modify,
        None, // quality monitor
        None, // learning system
        memory,
    ).await?;
    
    // Create code needing refactoring
    let complex_code = r#"
pub fn process_order(order: Order) -> Result<ProcessedOrder, OrderError> {
    // Validation logic
    if order.items.is_empty() {
        return Err(OrderError::NoItems);
    }
    if order.customer_id == 0 {
        return Err(OrderError::InvalidCustomer);
    }
    
    // Calculate totals
    let mut subtotal = 0.0;
    let mut tax_total = 0.0;
    for item in &order.items {
        let item_total = item.price * item.quantity as f64;
        subtotal += item_total;
        let item_tax = item_total * 0.08; // 8% tax
        tax_total += item_tax;
    }
    
    // Apply discounts
    let mut discount = 0.0;
    if subtotal > 100.0 {
        discount = subtotal * 0.1; // 10% discount
    }
    if order.customer_type == CustomerType::Premium {
        discount += subtotal * 0.05; // Extra 5% for premium
    }
    
    // Calculate final total
    let final_total = subtotal - discount + tax_total;
    
    // Create processed order
    Ok(ProcessedOrder {
        order_id: generate_order_id(),
        customer_id: order.customer_id,
        items: order.items,
        subtotal,
        tax: tax_total,
        discount,
        total: final_total,
        status: OrderStatus::Processed,
    })
}

fn calculate_shipping(weight: f64, distance: f64) -> f64 {
    weight * 0.5 + distance * 0.1
}
"#;
    
    tokio::fs::create_dir_all(temp_dir.path().join("src")).await?;
    tokio::fs::write(
        temp_dir.path().join("src/order.rs"),
        complex_code
    ).await?;
    
    // Analyze for refactoring
    let analysis = refactoring_system.analyze_for_refactoring().await?;
    
    assert!(!analysis.suggestions.is_empty(), "Should have refactoring suggestions");
    
    // Check for method extraction suggestions
    let extraction_suggestions: Vec<_> = analysis.suggestions.iter()
        .filter(|s| s.refactoring_type == RefactoringType::ExtractMethod)
        .collect();
    
    assert!(!extraction_suggestions.is_empty(), "Should suggest method extraction");
    
    // Verify impact metrics
    for suggestion in &analysis.suggestions {
        assert!(suggestion.impact.complexity_reduction >= 0.0);
        assert!(suggestion.impact.complexity_reduction <= 1.0);
        assert!(suggestion.impact.maintainability_improvement >= 0.0);
        assert!(suggestion.impact.maintainability_improvement <= 1.0);
    }
    
    // Check priority ordering
    assert_eq!(
        analysis.priority_order.len(),
        analysis.suggestions.len(),
        "Should prioritize all suggestions"
    );
    
    // Verify total impact calculation
    assert!(analysis.total_impact.complexity_reduction >= 0.0);
    assert!(analysis.total_impact.maintainability_improvement >= 0.0);
    
    Ok(())
}

#[tokio::test]
async fn test_pattern_learning_and_application() -> Result<()> {
    let (story_engine, memory, temp_dir) = setup().await?;
    
    let config = StoryDrivenLearningConfig {
        enable_pattern_extraction: true,
        enable_style_learning: true,
        enable_architecture_learning: true,
        enable_best_practices_learning: true,
        min_pattern_frequency: 2,
        confidence_threshold: 0.7,
        max_patterns_per_type: 50,
        repo_path: temp_dir.path().to_path_buf(),
    };
    
    let code_analyzer = Arc::new(CodeAnalyzer::new(None));
    let learning_system = StoryDrivenLearning::new(
        config,
        story_engine.clone(),
        code_analyzer,
        memory,
    ).await?;
    
    // Create files with patterns to learn
    let pattern1 = r#"
use anyhow::{Result, Context};

pub fn read_file(path: &str) -> Result<String> {
    std::fs::read_to_string(path)
        .context(format!("Failed to read file: {}", path))
}

pub fn write_file(path: &str, content: &str) -> Result<()> {
    std::fs::write(path, content)
        .context(format!("Failed to write file: {}", path))
}
"#;
    
    let pattern2 = r#"
use anyhow::{Result, Context};

pub fn parse_config(data: &str) -> Result<Config> {
    serde_json::from_str(data)
        .context("Failed to parse config")
}

pub fn load_settings(path: &str) -> Result<Settings> {
    let content = std::fs::read_to_string(path)
        .context("Failed to read settings file")?;
    serde_json::from_str(&content)
        .context("Failed to parse settings")
}
"#;
    
    tokio::fs::create_dir_all(temp_dir.path().join("src")).await?;
    tokio::fs::write(
        temp_dir.path().join("src/io.rs"),
        pattern1
    ).await?;
    tokio::fs::write(
        temp_dir.path().join("src/config.rs"),
        pattern2
    ).await?;
    
    // Learn patterns
    let result = learning_system.learn_from_codebase().await?;
    
    assert!(result.patterns_found > 0, "Should find patterns");
    assert!(result.new_patterns > 0, "Should learn new patterns");
    
    // Get learned patterns
    let patterns = learning_system.get_pattern_stats().await?;
    
    // Should learn error handling pattern
    let error_patterns = patterns.get(&LearnedPatternType::ErrorHandling);
    assert!(error_patterns.is_some(), "Should learn error handling patterns");
    
    // Apply patterns to new code
    let new_code = r#"
pub fn process_data(input: &str) -> String {
    let parsed = input.parse::<i32>().unwrap();
    format!("Processed: {}", parsed)
}
"#;
    
    let suggestions = learning_system.suggest_pattern_applications(new_code).await?;
    assert!(!suggestions.is_empty(), "Should suggest pattern applications");
    
    Ok(())
}

#[tokio::test]
async fn test_subsystem_integration() -> Result<()> {
    let (story_engine, memory, temp_dir) = setup().await?;
    
    // Create all subsystems
    let code_analyzer = Arc::new(CodeAnalyzer::new(None));
    let self_modify = Arc::new(SelfModificationPipeline::new(
        story_engine.clone(),
        memory.clone(),
    ));
    
    // Quality monitor
    let quality_config = StoryDrivenQualityConfig::default();
    let quality_monitor = Arc::new(StoryDrivenQuality::new(
        quality_config,
        story_engine.clone(),
        code_analyzer.clone(),
        None,
        memory.clone(),
    ).await?);
    
    // Learning system
    let learning_config = StoryDrivenLearningConfig::default();
    let learning_system = Arc::new(StoryDrivenLearning::new(
        learning_config,
        story_engine.clone(),
        code_analyzer.clone(),
        memory.clone(),
    ).await?);
    
    // Refactoring with quality and learning integration
    let refactoring_config = StoryDrivenRefactoringConfig {
        repo_path: temp_dir.path().to_path_buf(),
        ..Default::default()
    };
    
    let refactoring_system = StoryDrivenRefactoring::new(
        refactoring_config,
        story_engine.clone(),
        code_analyzer.clone(),
        self_modify,
        Some(quality_monitor.clone()),
        Some(learning_system.clone()),
        memory.clone(),
    ).await?;
    
    // Create test code
    let code = r#"
pub fn complex_function(data: Vec<i32>) -> i32 {
    let mut result = 0;
    for i in 0..data.len() {
        if data[i] > 0 {
            result += data[i];
        }
    }
    result
}
"#;
    
    tokio::fs::create_dir_all(temp_dir.path().join("src")).await?;
    tokio::fs::write(
        temp_dir.path().join("src/lib.rs"),
        code
    ).await?;
    
    // Run integrated analysis
    let quality_analysis = quality_monitor.analyze_quality().await?;
    let learning_result = learning_system.learn_from_codebase().await?;
    let refactoring_analysis = refactoring_system.analyze_for_refactoring().await?;
    
    // Verify integration
    assert!(quality_analysis.current_metrics.overall_health > 0.0);
    assert!(learning_result.patterns_analyzed > 0);
    assert!(!refactoring_analysis.suggestions.is_empty());
    
    // Check that refactoring uses quality data
    if !quality_analysis.hotspots.is_empty() {
        let hotspot_files: std::collections::HashSet<_> = quality_analysis.hotspots.iter()
            .map(|h| h.file_path.clone())
            .collect();
        
        let refactoring_files: std::collections::HashSet<_> = refactoring_analysis.suggestions.iter()
            .map(|s| s.file_path.clone())
            .collect();
        
        // Refactoring should target hotspots
        let overlap = hotspot_files.intersection(&refactoring_files).count();
        assert!(overlap > 0 || hotspot_files.is_empty(), "Should target quality hotspots");
    }
    
    Ok(())
}