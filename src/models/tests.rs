#[cfg(test)]
mod tests {
    // Removed unused super::* import
    use tokio;

    use crate::models::{AllocationPriority, ComponentType, ModelCapabilities, ModelConfigManager, ModelInfo, ModelRegistry, ModelSpecialization, PerformanceMetrics, ProviderType, QuantizationType, ResourceMonitor, ResourceRequirements, RoutingStrategy, TaskType, parse_task_type, RegistryPerformanceMetrics};

    #[tokio::test]
    async fn test_model_registry_capabilities() {
        let mut registry = ModelRegistry::new();

        let model = ModelInfo {
            name: "test-model".to_string(),
            description: "Test model".to_string(),
            size: 1000,
            file_name: "test.bin".to_string(),
            quantization: "Q4_K_M".to_string(),
            parameters: 7_000_000_000,
            license: "MIT".to_string(),
            url: None,
            version: Some("1.0".to_string()),
            provider_type: ProviderType::Local,
            capabilities: ModelCapabilities {
                code_generation: 0.9,
                code_review: 0.8,
                reasoning: 0.7,
                creative_writing: 0.5,
                data_analysis: 0.6,
                mathematical_computation: 0.7,
                language_translation: 0.4,
                context_window: 8192,
                max_tokens_per_second: 50.0,
                supports_streaming: true,
                supports_function_calling: false,
            },
            specializations: vec![
                ModelSpecialization::CodeGeneration,
                ModelSpecialization::CodeReview,
            ],
            resource_requirements: Some(ResourceRequirements {
                min_memory_gb: 4.0,
                recommended_memory_gb: 6.0,
                min_gpu_memory_gb: None,
                recommended_gpu_memory_gb: None,
                cpu_cores: 4,
                gpu_layers: None,
                quantization: QuantizationType::Q4KM,
            }),
            performance_metrics: RegistryPerformanceMetrics::default(),
        };

        registry.add_model(model).expect("Should add model successfully");

        // Test specialization filtering
        let code_models = registry.find_by_specialization(&ModelSpecialization::CodeGeneration);
        assert_eq!(code_models.len(), 1);

        // Test provider type filtering
        let local_models = registry.find_by_provider_type(&ProviderType::Local);
        assert_eq!(local_models.len(), 1);
    }

    #[tokio::test]
    async fn testconfig_manager_default() {
        let config_manager = ModelConfigManager::create_default();
        let config = config_manager.getconfig();

        // Should have some default models
        assert!(!config.models.local.is_empty());
        assert!(!config.models.api.is_empty());

        // Should have orchestration settings
        assert_eq!(config.orchestration.default_strategy, RoutingStrategy::CapabilityBased);
        assert!(config.orchestration.fallback_enabled);
        assert!(config.orchestration.prefer_local);
    }

    #[tokio::test]
    async fn test_resource_monitor_creation() {
        let monitor = ResourceMonitor::new().await;
        assert!(monitor.is_ok());

        let monitor = monitor.unwrap();
        let system_info = monitor.get_system_info();

        // Should detect some system resources
        assert!(system_info.total_ram_gb > 0.0);
        assert!(system_info.cpu_cores > 0);
    }

    #[tokio::test]
    async fn test_resource_allocation() {
        let monitor = ResourceMonitor::new().await.expect("Should create monitor");

        // Test allocation
        let result = monitor
            .allocate_resources(
                "test-component".to_string(),
                ComponentType::LocalModel,
                1024, // 1GB
                None,
                2,
                AllocationPriority::Medium,
            )
            .await;

        assert!(result.is_ok());

        // Test deallocation
        let result = monitor.deallocate_resources("test-component").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_task_type_parsing() {
        let task_type = parse_task_type("code_generation");
        match task_type {
            TaskType::CodeGeneration { language } => {
                assert_eq!(language, "python");
            }
            _ => panic!("Wrong task type parsed"),
        }

        let task_type = parse_task_type("logical_reasoning");
        match task_type {
            TaskType::LogicalReasoning => {}
            _ => panic!("Wrong task type parsed"),
        }
    }

    #[test]
    fn test_provider_type_serialization() {
        // Test that our enums can be serialized/deserialized
        let provider_type = ProviderType::Local;
        assert_eq!(provider_type.as_str(), "local");

        let api_type = ProviderType::API;
        assert_eq!(api_type.as_str(), "api");
    }

    #[test]
    fn test_capability_defaults() {
        let caps = ModelCapabilities::default();
        assert_eq!(caps.code_generation, 0.5);
        assert_eq!(caps.context_window, 2048);
        assert!(!caps.supports_streaming);
    }

    #[test]
    fn test_quantization_types() {
        let q4 = QuantizationType::Q4KM;
        let q5 = QuantizationType::Q5KM;
        let custom = QuantizationType::Custom("GPTQ".to_string());

        // These should all be different variants
        assert_ne!(std::mem::discriminant(&q4), std::mem::discriminant(&q5));
        assert_ne!(std::mem::discriminant(&q4), std::mem::discriminant(&custom));
    }
}
