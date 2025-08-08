//! Tests for the new memory hierarchy implementations

use loki::memory::fractal::hierarchy::{
    DynamicHierarchyManager, HierarchyConfig, FormationStrategy, StrategySelectionCriteria,
    OptimizationSettings, MonitoringConfig, PerformanceThresholds, ResourceConstraints,
};
use loki::memory::fractal::nodes::FractalMemoryNode;
use std::sync::Arc;
use std::collections::HashMap;

#[tokio::test]
async fn test_evolve_hierarchy() {
    // Create a test configuration
    let config = create_test_config();
    
    // Initialize the hierarchy manager
    let manager = DynamicHierarchyManager::new(config).await
        .expect("Failed to create hierarchy manager");
    
    // Create a test root node
    let root = Arc::new(FractalMemoryNode::new(
        "test_root".to_string(),
        "Root node content".to_string(),
        HashMap::new(),
    ));
    
    // Form a hierarchy
    let hierarchy_result = manager.form_adaptive_hierarchy_with_leadership(&root).await
        .expect("Failed to form hierarchy");
    
    // Test the evolve method
    let evolution_result = manager.evolve(&hierarchy_result.hierarchy_id).await
        .expect("Failed to evolve hierarchy");
    
    assert!(evolution_result.evolution_success);
    assert!(evolution_result.evolution_quality_score > 0.0);
    println!("Evolution completed with score: {}", evolution_result.evolution_quality_score);
}

#[tokio::test]
async fn test_balance_layers() {
    let config = create_test_config();
    let manager = DynamicHierarchyManager::new(config).await
        .expect("Failed to create hierarchy manager");
    
    let root = Arc::new(FractalMemoryNode::new(
        "test_root".to_string(),
        "Root node content".to_string(),
        HashMap::new(),
    ));
    
    let hierarchy_result = manager.form_adaptive_hierarchy_with_leadership(&root).await
        .expect("Failed to form hierarchy");
    
    // Test balance_layers method
    let balance_result = manager.balance_layers(&hierarchy_result.hierarchy_id).await
        .expect("Failed to balance layers");
    
    assert!(balance_result.balance_score > 0.0);
    println!("Balance score: {}", balance_result.balance_score);
}

#[tokio::test]
async fn test_memory_pressure() {
    let config = create_test_config();
    let manager = DynamicHierarchyManager::new(config).await
        .expect("Failed to create hierarchy manager");
    
    let root = Arc::new(FractalMemoryNode::new(
        "test_root".to_string(),
        "Root node content".to_string(),
        HashMap::new(),
    ));
    
    let hierarchy_result = manager.form_adaptive_hierarchy_with_leadership(&root).await
        .expect("Failed to form hierarchy");
    
    // Test get_memory_pressure method
    let pressure_report = manager.get_memory_pressure(&hierarchy_result.hierarchy_id).await
        .expect("Failed to get memory pressure");
    
    assert!(pressure_report.overall_pressure >= 0.0);
    assert!(pressure_report.overall_pressure <= 1.0);
    println!("Memory pressure: {}", pressure_report.overall_pressure);
}

#[tokio::test]
async fn test_fractal_dimension() {
    let config = create_test_config();
    let manager = DynamicHierarchyManager::new(config).await
        .expect("Failed to create hierarchy manager");
    
    let root = Arc::new(FractalMemoryNode::new(
        "test_root".to_string(),
        "Root node content".to_string(),
        HashMap::new(),
    ));
    
    let hierarchy_result = manager.form_adaptive_hierarchy_with_leadership(&root).await
        .expect("Failed to form hierarchy");
    
    // Test calculate_fractal_dimension method
    let dimension_analysis = manager.calculate_fractal_dimension(&hierarchy_result.hierarchy_id).await
        .expect("Failed to calculate fractal dimension");
    
    assert!(dimension_analysis.hausdorff_dimension > 0.0);
    assert!(dimension_analysis.complexity_score > 0.0);
    println!("Fractal dimension: {}", dimension_analysis.hausdorff_dimension);
    println!("Complexity score: {}", dimension_analysis.complexity_score);
}

#[tokio::test]
async fn test_anomaly_detection() {
    let config = create_test_config();
    let manager = DynamicHierarchyManager::new(config).await
        .expect("Failed to create hierarchy manager");
    
    let root = Arc::new(FractalMemoryNode::new(
        "test_root".to_string(),
        "Root node content".to_string(),
        HashMap::new(),
    ));
    
    let hierarchy_result = manager.form_adaptive_hierarchy_with_leadership(&root).await
        .expect("Failed to form hierarchy");
    
    // Test detect_anomalies method
    let anomaly_report = manager.detect_anomalies(&hierarchy_result.hierarchy_id).await
        .expect("Failed to detect anomalies");
    
    assert!(anomaly_report.health_score >= 0.0);
    assert!(anomaly_report.health_score <= 1.0);
    println!("Health score: {}", anomaly_report.health_score);
    println!("Critical anomalies: {}", anomaly_report.critical_count);
}

// Helper function to create test configuration
fn create_test_config() -> HierarchyConfig {
    HierarchyConfig {
        max_depth: 5,
        target_branching_factor: 3,
        reorganization_threshold: 0.7,
        enable_adaptive_formation: true,
        enable_emergent_leadership: true,
        enable_self_organization: true,
        formation_strategies: vec![
            FormationStrategy::AdaptiveWithLeadership,
        ],
        strategy_selection: StrategySelectionCriteria {
            performance_weight: 0.3,
            memory_efficiency_weight: 0.3,
            adaptability_weight: 0.2,
            complexity_tolerance: 0.8,
            context_preferences: HashMap::new(),
        },
        optimization_settings: OptimizationSettings {
            enable_realtime_optimization: true,
            optimization_frequency: std::time::Duration::from_secs(60),
            performance_thresholds: PerformanceThresholds {
                min_balance_score: 0.7,
                max_access_latency_ms: 100.0,
                min_semantic_coherence: 0.6,
                max_memory_overhead: 0.2,
            },
            optimization_algorithms: vec![],
            resource_constraints: ResourceConstraints {
                max_memory_mb: 1024,
                max_cpu_percentage: 80.0,
                max_io_operations_per_sec: 1000,
                max_network_bandwidth_mbps: 100.0,
            },
        },
        monitoringconfig: MonitoringConfig {
            enable_detailed_metrics: true,
            metric_collection_interval: std::time::Duration::from_secs(10),
            metric_retention_period: std::time::Duration::from_secs(3600),
            alert_thresholds: loki::memory::fractal::hierarchy::AlertThresholds {
                critical_performance_threshold: 0.3,
                warning_performance_threshold: 0.5,
                structural_imbalance_threshold: 0.4,
                access_pattern_anomaly_threshold: 0.6,
            },
            reporting_frequency: std::time::Duration::from_secs(300),
        },
    }
}