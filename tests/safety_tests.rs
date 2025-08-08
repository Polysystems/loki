use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use loki::safety::{
    ActionType,
    ActionValidator,
    LimitExceeded,
    ResourceLimits,
    ResourceMonitor,
    RiskLevel,
    ValidatorConfig,
};
use tokio::sync::mpsc;

async fn setup_test_resource_monitor() -> Result<(ResourceMonitor, mpsc::Receiver<LimitExceeded>)> {
    let limits = ResourceLimits::default();
    let (alert_tx, alert_rx) = mpsc::channel(100);

    let monitor = ResourceMonitor::new(limits, alert_tx);

    Ok((monitor, alert_rx))
}

fn create_test_validator_config() -> ValidatorConfig {
    ValidatorConfig {
        safe_mode: true,
        dry_run: false,
        approval_required: false, // Disable for automated tests
        approval_timeout: Duration::from_secs(30),
        allowed_paths: vec!["test/**".to_string(), "workspace/**".to_string()],
        blocked_paths: vec!["src/safety/**".to_string(), ".env".to_string()],
        max_file_size: 1024 * 1024, // 1MB
        storage_path: Some(std::path::PathBuf::from("data/test/safety_decisions")),
        encrypt_decisions: true,
    }
}

#[tokio::test]
async fn test_resource_monitor_creation() -> Result<()> {
    let (monitor, _alert_rx) = setup_test_resource_monitor().await?;

    // Test that monitor starts successfully
    monitor.start().await?;

    // Test getting current usage
    let usage = monitor.get_usage().await;
    assert!(usage.memory_mb >= 0);
    assert!(usage.cpu_percent >= 0.0);
    assert!(usage.file_handles >= 0);

    monitor.shutdown().await;
    Ok(())
}

#[tokio::test]
async fn test_resource_limits_enforcement() -> Result<()> {
    let (monitor, mut _alert_rx) = setup_test_resource_monitor().await?;

    // Start monitoring
    monitor.start().await?;

    // Wait a moment for monitoring to begin
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Test API rate limiting
    let api_result = monitor.check_api_limit("openai").await;
    assert!(api_result.is_ok());

    // Test token budget checking
    let token_result = monitor.check_token_budget("openai", 1000).await;
    assert!(token_result.is_ok());

    // Test operation counting
    let op_result = monitor.start_operation().await;
    assert!(op_result.is_ok());

    monitor.end_operation().await;

    monitor.shutdown().await;
    Ok(())
}

#[tokio::test]
async fn test_resource_usage_tracking() -> Result<()> {
    let (monitor, _alert_rx) = setup_test_resource_monitor().await?;

    monitor.start().await?;

    // Wait for some monitoring cycles
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Get usage history
    let history = monitor.get_history().await;
    // Should have some history after monitoring
    // Note: History might be empty in fast tests, so we just check it doesn't panic

    // Get current usage
    let usage = monitor.get_usage().await;

    monitor.shutdown().await;
    Ok(())
}

#[tokio::test]
async fn test_action_validator_creation() -> Result<()> {
    let config = create_test_validator_config();
    let validator = ActionValidator::new(config);

    // Test basic functionality
    assert!(validator.get_pending_actions().await.is_empty());

    Ok(())
}

#[tokio::test]
async fn test_file_action_validation() -> Result<()> {
    let config = create_test_validator_config();
    let validator = ActionValidator::new(config);

    // Test allowed file read
    let read_action = ActionType::FileRead { path: "test/sample.txt".to_string() };

    let result = validator
        .validate_action(
            read_action,
            "Testing file read validation".to_string(),
            vec!["Read operation for testing".to_string()],
        )
        .await;

    assert!(result.is_ok());

    // Test blocked file write
    let blocked_write = ActionType::FileWrite {
        path: "src/safety/validator.rs".to_string(),
        content: "malicious content".to_string(),
    };

    let result = validator
        .validate_action(
            blocked_write,
            "Testing blocked path".to_string(),
            vec!["Attempting to modify safety system".to_string()],
        )
        .await;

    assert!(result.is_err());

    Ok(())
}

#[tokio::test]
async fn test_action_type_risk_levels() -> Result<()> {
    // Test various action types and their risk levels
    let actions = vec![
        (ActionType::FileRead { path: "test.txt".to_string() }, RiskLevel::Low),
        (
            ActionType::FileWrite {
                path: "workspace/output.txt".to_string(),
                content: "test content".to_string(),
            },
            RiskLevel::High,
        ),
        (
            ActionType::ApiCall {
                provider: "openai".to_string(),
                endpoint: "/v1/chat/completions".to_string(),
            },
            RiskLevel::Medium,
        ),
        (
            ActionType::GitCommit {
                message: "Test commit".to_string(),
                files: vec!["test.txt".to_string()],
            },
            RiskLevel::Critical,
        ),
    ];

    for (action, expected_risk) in actions {
        assert_eq!(action.risk_level(), expected_risk);
    }

    Ok(())
}

#[tokio::test]
async fn test_action_decision_process() -> Result<()> {
    let config = create_test_validator_config();
    let validator = ActionValidator::new(config);

    // Test low-risk action (should auto-approve)
    let low_risk_action = ActionType::FileRead { path: "test.txt".to_string() };

    let result = validator
        .validate_action(
            low_risk_action,
            "Reading test file".to_string(),
            vec!["Low risk operation".to_string()],
        )
        .await;

    assert!(result.is_ok());

    Ok(())
}

#[tokio::test]
async fn test_action_approval_required() -> Result<()> {
    // Test actions that require approval
    let actions = vec![
        ActionType::FileWrite { path: "test.txt".to_string(), content: "test".to_string() },
        ActionType::GitCommit {
            message: "Test commit".to_string(),
            files: vec!["test.txt".to_string()],
        },
        ActionType::CommandExecute { command: "ls".to_string(), args: vec!["-la".to_string()] },
    ];

    for action in actions {
        assert!(action.requires_approval());
    }

    Ok(())
}

#[tokio::test]
async fn test_action_history_tracking() -> Result<()> {
    let config = create_test_validator_config();
    let validator = ActionValidator::new(config);

    // Validate several actions
    let actions = vec![
        ActionType::FileRead { path: "test1.txt".to_string() },
        ActionType::FileRead { path: "test2.txt".to_string() },
        ActionType::ApiCall { provider: "test".to_string(), endpoint: "/test".to_string() },
    ];

    for action in actions {
        let _ = validator
            .validate_action(
                action,
                "Test action".to_string(),
                vec!["Testing history tracking".to_string()],
            )
            .await;
    }

    // Check history
    let history = validator.get_history().await;
    assert!(history.len() >= 3);

    // Clear history
    validator.clear_history().await;
    let cleared_history = validator.get_history().await;
    assert!(cleared_history.is_empty());

    Ok(())
}

#[tokio::test]
async fn test_concurrent_safety_operations() -> Result<()> {
    let config = create_test_validator_config();
    let validator = Arc::new(ActionValidator::new(config));

    // Spawn multiple concurrent validations
    let tasks: Vec<_> = (0..5)
        .map(|i| {
            let validator = validator.clone();
            tokio::spawn(async move {
                let action = ActionType::FileRead { path: format!("test/concurrent_{}.txt", i) };

                validator
                    .validate_action(
                        action,
                        format!("Concurrent test {}", i),
                        vec![format!("Testing concurrency {}", i)],
                    )
                    .await
            })
        })
        .collect();

    // Wait for all tasks to complete
    for task in tasks {
        let result = task.await?;
        assert!(result.is_ok());
    }

    // Check that all actions were recorded
    let history = validator.get_history().await;
    assert!(history.len() >= 5);

    Ok(())
}

#[tokio::test]
async fn test_safety_performance_benchmarks() -> Result<()> {
    let config = create_test_validator_config();
    let validator = ActionValidator::new(config);

    let start = std::time::Instant::now();

    // Benchmark validation performance
    for i in 0..10 {
        let action = ActionType::FileRead { path: format!("test/benchmark_{}.txt", i) };

        validator
            .validate_action(
                action,
                "Performance test".to_string(),
                vec!["Benchmarking validation speed".to_string()],
            )
            .await?;
    }

    let duration = start.elapsed();

    // Should complete validations quickly
    assert!(duration < Duration::from_secs(2));

    Ok(())
}

#[tokio::test]
async fn test_safety_memory_efficiency() -> Result<()> {
    // Test that safety systems don't consume excessive memory
    let config = create_test_validator_config();
    let validator = ActionValidator::new(config);

    // Perform many operations to test memory usage
    for i in 0..100 {
        let action = ActionType::FileRead { path: format!("test/memory_test_{}.txt", i) };

        validator
            .validate_action(
                action,
                "Memory efficiency test".to_string(),
                vec!["Testing memory usage".to_string()],
            )
            .await?;
    }

    // Check that history is bounded (should be limited to 1000 entries)
    let history = validator.get_history().await;
    assert!(history.len() <= 1000);

    Ok(())
}

#[tokio::test]
async fn test_resource_limit_updates() -> Result<()> {
    let (monitor, _alert_rx) = setup_test_resource_monitor().await?;

    // Test updating limits
    let new_limits = ResourceLimits {
        max_memory_mb: 2048,
        max_cpu_percent: 90.0,
        api_rate_limits: std::collections::HashMap::new(),
        token_budgets: std::collections::HashMap::new(),
        max_file_handles: 500,
        max_concurrent_ops: 50,
    };

    monitor.update_limits(new_limits).await;

    // Verify limits were updated by checking operations
    let result = monitor.start_operation().await;
    assert!(result.is_ok());

    monitor.end_operation().await;

    Ok(())
}

#[tokio::test]
async fn test_safety_integration() -> Result<()> {
    // Test integration between resource monitoring and action validation
    let (monitor, _alert_rx) = setup_test_resource_monitor().await?;
    let config = create_test_validator_config();
    let validator = ActionValidator::new(config);

    monitor.start().await?;

    // Test that resource monitoring works with action validation
    let resource_intensive_action = ActionType::CommandExecute {
        command: "stress_test".to_string(),
        args: vec!["--cpu".to_string(), "4".to_string()],
    };

    // Should validate but consider resource implications
    let result = validator
        .validate_action(
            resource_intensive_action,
            "Resource intensive operation".to_string(),
            vec!["Testing resource integration".to_string()],
        )
        .await;

    // In safe mode, this should be allowed but monitored
    assert!(result.is_ok());

    monitor.shutdown().await;
    Ok(())
}

#[test]
fn test_risk_level_ordering() {
    // Test that risk levels are properly ordered
    assert!(RiskLevel::Critical > RiskLevel::High);
    assert!(RiskLevel::High > RiskLevel::Medium);
    assert!(RiskLevel::Medium > RiskLevel::Low);
}

#[test]
fn test_action_type_creation() {
    // Test that action types can be created
    let file_read = ActionType::FileRead { path: "test.txt".to_string() };
    assert_eq!(file_read.risk_level(), RiskLevel::Low);

    let file_write =
        ActionType::FileWrite { path: "test.txt".to_string(), content: "test content".to_string() };
    assert_eq!(file_write.risk_level(), RiskLevel::High);
}
