//! Unit tests for trigger analysis in emergence patterns

use loki::memory::fractal::hierarchy::{
    AdaptationTriggerSystem, EmergencePattern, EmergencePatternType,
};
use std::collections::HashMap;

#[tokio::test]
async fn test_trigger_analysis_basic() {
    let trigger_system = AdaptationTriggerSystem::new();
    
    let patterns = vec![
        EmergencePattern {
            pattern_id: "test_pattern_1".to_string(),
            pattern_type: EmergencePatternType::CoherentBehavior,
            description: "Test pattern".to_string(),
            confidence: 0.9,
            strength: 0.8,
            impact_metrics: HashMap::new(),
            ..Default::default()
        },
    ];

    let triggers = trigger_system.analyze_triggers(&patterns).await.unwrap();
    
    assert_eq!(triggers.len(), 1);
    assert_eq!(triggers[0].pattern_id, "test_pattern_1");
    assert_eq!(triggers[0].trigger_type, "threshold_crossing");
    assert!(triggers[0].urgency > 0.0 && triggers[0].urgency <= 1.0);
}

#[tokio::test]
async fn test_trigger_analysis_multiple_patterns() {
    let trigger_system = AdaptationTriggerSystem::new();
    
    let patterns = vec![
        EmergencePattern {
            pattern_id: "pattern_1".to_string(),
            pattern_type: EmergencePatternType::ResourceOptimization,
            confidence: 0.8,
            strength: 0.9,
            ..Default::default()
        },
        EmergencePattern {
            pattern_id: "pattern_2".to_string(),
            pattern_type: EmergencePatternType::NoveltyGeneration,
            confidence: 0.7,
            strength: 0.6,
            ..Default::default()
        },
    ];

    let triggers = trigger_system.analyze_triggers(&patterns).await.unwrap();
    
    assert_eq!(triggers.len(), 2);
    
    // Check first trigger
    let trigger1 = triggers.iter().find(|t| t.pattern_id == "pattern_1").unwrap();
    assert_eq!(trigger1.trigger_type, "resource_pressure");
    
    // Check second trigger
    let trigger2 = triggers.iter().find(|t| t.pattern_id == "pattern_2").unwrap();
    assert_eq!(trigger2.trigger_type, "anomaly_detection");
}

#[tokio::test]
async fn test_anomaly_detection_trigger() {
    let trigger_system = AdaptationTriggerSystem::new();
    
    // Pattern with low confidence but high strength should trigger anomaly detection
    let patterns = vec![
        EmergencePattern {
            pattern_id: "anomaly_pattern".to_string(),
            pattern_type: EmergencePatternType::InformationFlow,
            confidence: 0.2,  // Low confidence
            strength: 0.9,    // High strength
            ..Default::default()
        },
    ];

    let triggers = trigger_system.analyze_triggers(&patterns).await.unwrap();
    
    assert_eq!(triggers.len(), 1);
    assert_eq!(triggers[0].trigger_type, "anomaly_detection");
}

#[tokio::test]
async fn test_urgency_calculation() {
    let trigger_system = AdaptationTriggerSystem::new();
    
    let patterns = vec![
        EmergencePattern {
            pattern_id: "high_urgency".to_string(),
            pattern_type: EmergencePatternType::ResourceOptimization,
            confidence: 1.0,
            strength: 1.0,
            ..Default::default()
        },
        EmergencePattern {
            pattern_id: "low_urgency".to_string(),
            pattern_type: EmergencePatternType::DynamicEquilibrium,
            confidence: 0.5,
            strength: 0.5,
            ..Default::default()
        },
    ];

    let triggers = trigger_system.analyze_triggers(&patterns).await.unwrap();
    
    let high_urgency_trigger = triggers.iter().find(|t| t.pattern_id == "high_urgency").unwrap();
    let low_urgency_trigger = triggers.iter().find(|t| t.pattern_id == "low_urgency").unwrap();
    
    // High urgency should be greater than low urgency
    assert!(high_urgency_trigger.urgency > low_urgency_trigger.urgency);
    
    // Check urgency bounds
    assert!(high_urgency_trigger.urgency <= 1.0);
    assert!(low_urgency_trigger.urgency >= 0.1);
}

#[tokio::test]
async fn test_all_pattern_types() {
    let trigger_system = AdaptationTriggerSystem::new();
    
    let pattern_types = vec![
        (EmergencePatternType::CoherentBehavior, "threshold_crossing"),
        (EmergencePatternType::InformationFlow, "temporal_correlation"),
        (EmergencePatternType::StructuralFormation, "critical_mass"),
        (EmergencePatternType::ResourceOptimization, "resource_pressure"),
        (EmergencePatternType::CommunicationProtocol, "pattern_detection"),
        (EmergencePatternType::MemoryConsolidation, "temporal_trigger"),
        (EmergencePatternType::SynergyAmplification, "multi_factor_convergence"),
        (EmergencePatternType::NoveltyGeneration, "anomaly_detection"),
        (EmergencePatternType::CollectiveIntelligence, "distributed_consensus"),
        (EmergencePatternType::DynamicEquilibrium, "balance_threshold"),
    ];
    
    for (pattern_type, expected_trigger) in pattern_types {
        let patterns = vec![
            EmergencePattern {
                pattern_id: format!("test_{:?}", pattern_type),
                pattern_type,
                confidence: 0.7,
                strength: 0.7,
                ..Default::default()
            },
        ];
        
        let triggers = trigger_system.analyze_triggers(&patterns).await.unwrap();
        assert_eq!(triggers.len(), 1);
        assert_eq!(triggers[0].trigger_type, expected_trigger);
    }
}