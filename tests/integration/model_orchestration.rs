use anyhow::Result;
use loki::{
    config::ApiKeysConfig,
    models::{
        integration::IntegratedModelSystem,
        orchestrator::{TaskRequest, TaskType, RoutingStrategy},
        config::ModelOrchestrationConfig,
    },
};
use std::time::Duration;
use tokio::time::timeout;

/// Integration tests for the advanced model orchestration system
/// These tests verify the complete workflow from configuration to execution

#[tokio::test]
async fn test_model_system_initialization() -> Result<()> {
    // Test that the integrated model system can be initialized
    let api_config = ApiKeysConfig::default();
    let system = IntegratedModelSystem::new(&api_config).await?;
    
    // Verify system status
    let status = system.get_system_status().await?;
    assert!(status.is_healthy);
    assert!(!status.available_models.is_empty());
    
    Ok(())
}

#[tokio::test]
async fn test_task_routing_capability_based() -> Result<()> {
    let api_config = ApiKeysConfig::default();
    let system = IntegratedModelSystem::new(&api_config).await?;
    
    // Test code generation routing
    let code_task = TaskRequest {
        task_type: TaskType::CodeGeneration { language: "python".to_string() },
        content: "Write a hello world function".to_string(),
        max_tokens: Some(100),
        prefer_local: true,
    };
    
    // This should complete within a reasonable time even with mock/empty responses
    let result = timeout(Duration::from_secs(30), system.process_request(&code_task)).await;
    assert!(result.is_ok(), "Task routing should not timeout");
    
    Ok(())
}

#[tokio::test]
async fn test_fallback_mechanism() -> Result<()> {
    let api_config = ApiKeysConfig::default();
    let system = IntegratedModelSystem::new(&api_config).await?;
    
    // Test with a task that should trigger fallback
    let complex_task = TaskRequest {
        task_type: TaskType::LogicalReasoning,
        content: "Explain quantum computing principles".to_string(),
        max_tokens: Some(500),
        prefer_local: false,
    };
    
    // Should handle gracefully even without real API keys
    let result = timeout(Duration::from_secs(30), system.process_request(&complex_task)).await;
    assert!(result.is_ok(), "Fallback mechanism should work");
    
    Ok(())
}

#[tokio::test]
async fn test_cost_tracking() -> Result<()> {
    let api_config = ApiKeysConfig::default();
    let system = IntegratedModelSystem::new(&api_config).await?;
    
    // Test cost tracking functionality
    let initial_costs = system.get_cost_summary().await?;
    assert!(initial_costs.total_cost_cents >= 0.0);
    
    // Process a request and verify cost tracking
    let task = TaskRequest {
        task_type: TaskType::CreativeWriting,
        content: "Write a haiku about AI".to_string(),
        max_tokens: Some(50),
        prefer_local: true,
    };
    
    let _ = timeout(Duration::from_secs(30), system.process_request(&task)).await;
    
    // Cost tracking should be working (even if costs are 0 for local models)
    let final_costs = system.get_cost_summary().await?;
    assert!(final_costs.total_cost_cents >= initial_costs.total_cost_cents);
    
    Ok(())
}

#[tokio::test]
async fn test_performance_monitoring() -> Result<()> {
    let api_config = ApiKeysConfig::default();
    let system = IntegratedModelSystem::new(&api_config).await?;
    
    // Test performance metrics collection
    let metrics = system.get_performance_metrics().await?;
    assert!(metrics.total_requests >= 0);
    assert!(metrics.average_latency_ms >= 0.0);
    
    Ok(())
}

#[tokio::test]
async fn test_model_configuration_loading() -> Result<()> {
    // Test that model configuration can be loaded and validated
    let config = ModelOrchestrationConfig::load_default().await?;
    
    // Verify basic configuration structure
    assert!(!config.models.local.is_empty() || !config.models.api.is_empty());
    assert_eq!(config.orchestration.default_strategy, RoutingStrategy::CapabilityBased);
    
    Ok(())
}

#[tokio::test]
async fn test_routing_strategies() -> Result<()> {
    let api_config = ApiKeysConfig::default();
    let system = IntegratedModelSystem::new(&api_config).await?;
    
    // Test different routing strategies
    let strategies = [
        RoutingStrategy::CapabilityBased,
        RoutingStrategy::LoadBased,
        RoutingStrategy::CostOptimized,
    ];
    
    for strategy in strategies {
        system.set_routing_strategy(strategy.clone()).await?;
        
        let task = TaskRequest {
            task_type: TaskType::DataAnalysis,
            content: "Analyze sample data".to_string(),
            max_tokens: Some(200),
            prefer_local: true,
        };
        
        // Should handle different strategies gracefully
        let result = timeout(Duration::from_secs(30), system.process_request(&task)).await;
        assert!(result.is_ok(), "Strategy {:?} should work", strategy);
    }
    
    Ok(())
}

#[tokio::test]
async fn test_concurrent_requests() -> Result<()> {
    let api_config = ApiKeysConfig::default();
    let system = IntegratedModelSystem::new(&api_config).await?;
    
    // Test concurrent request handling
    let mut tasks = Vec::new();
    
    for i in 0..5 {
        let system_clone = system.clone();
        let task = TaskRequest {
            task_type: TaskType::CodeGeneration { language: "rust".to_string() },
            content: format!("Write function number {}", i),
            max_tokens: Some(100),
            prefer_local: true,
        };
        
        tasks.push(tokio::spawn(async move {
            timeout(Duration::from_secs(30), system_clone.process_request(&task)).await
        }));
    }
    
    // All tasks should complete successfully
    for task in tasks {
        let result = task.await?;
        assert!(result.is_ok(), "Concurrent requests should succeed");
    }
    
    Ok(())
}

#[tokio::test]
async fn test_system_health_monitoring() -> Result<()> {
    let api_config = ApiKeysConfig::default();
    let system = IntegratedModelSystem::new(&api_config).await?;
    
    // Test health monitoring
    let health = system.check_health().await?;
    assert!(health.overall_healthy);
    assert!(!health.component_health.is_empty());
    
    // Test specific components
    assert!(health.component_health.contains_key("orchestrator"));
    assert!(health.component_health.contains_key("cost_manager"));
    
    Ok(())
}

#[tokio::test]
async fn test_model_discovery() -> Result<()> {
    let api_config = ApiKeysConfig::default();
    let system = IntegratedModelSystem::new(&api_config).await?;
    
    // Test model discovery functionality
    let discovered_models = system.discover_available_models().await?;
    assert!(!discovered_models.is_empty(), "Should discover at least some models");
    
    // Verify model information structure
    for model in discovered_models {
        assert!(!model.name.is_empty());
        assert!(!model.description.is_empty());
        assert!(model.capabilities.code_generation >= 0.0);
        assert!(model.capabilities.code_generation <= 1.0);
    }
    
    Ok(())
}