//! Type conversions between modular and core systems

use crate::models;
use crate::tui::chat::orchestration::manager::{RoutingStrategy as ModularRoutingStrategy};

/// Convert from modular RoutingStrategy to models::RoutingStrategy
impl From<ModularRoutingStrategy> for models::RoutingStrategy {
    fn from(modular: ModularRoutingStrategy) -> Self {
        match modular {
            ModularRoutingStrategy::RoundRobin => models::RoutingStrategy::LoadBased,
            ModularRoutingStrategy::LeastLatency => models::RoutingStrategy::LatencyOptimized,
            ModularRoutingStrategy::ContextAware => models::RoutingStrategy::CapabilityBased,
            ModularRoutingStrategy::CapabilityBased => models::RoutingStrategy::CapabilityBased,
            ModularRoutingStrategy::CostOptimized => models::RoutingStrategy::CostOptimized,
            ModularRoutingStrategy::Custom(_) => models::RoutingStrategy::CapabilityBased,
            ModularRoutingStrategy::Capability => models::RoutingStrategy::CapabilityBased,
            ModularRoutingStrategy::Cost => models::RoutingStrategy::CostOptimized,
            ModularRoutingStrategy::Speed => models::RoutingStrategy::LatencyOptimized,
            ModularRoutingStrategy::Quality => models::RoutingStrategy::CapabilityBased,
            ModularRoutingStrategy::Availability => models::RoutingStrategy::LoadBased,
            ModularRoutingStrategy::Hybrid => models::RoutingStrategy::LoadBased,
        }
    }
}

/// Convert from models::RoutingStrategy to modular RoutingStrategy
impl From<models::RoutingStrategy> for ModularRoutingStrategy {
    fn from(core: models::RoutingStrategy) -> Self {
        match core {
            models::RoutingStrategy::CapabilityBased => ModularRoutingStrategy::CapabilityBased,
            models::RoutingStrategy::LoadBased => ModularRoutingStrategy::RoundRobin,
            models::RoutingStrategy::CostOptimized => ModularRoutingStrategy::CostOptimized,
            models::RoutingStrategy::LatencyOptimized => ModularRoutingStrategy::LeastLatency,
        }
    }
}

/// Conversion utilities for OrchestrationManager
pub mod orchestration {
    use crate::tui::chat::orchestration::manager::OrchestrationManager as ModularOrchestrationManager;
    
    /// Helper to convert orchestration configuration
    pub fn sync_orchestration_config(
        modular: &ModularOrchestrationManager,
        core_orchestrator: &mut crate::models::ModelOrchestrator,
    ) {
        // Apply routing strategy
        if let Ok(strategy) = serde_json::to_value(&modular.preferred_strategy) {
            // ModelOrchestrator might need updates to accept runtime config
            tracing::debug!("Would apply routing strategy: {:?}", strategy);
        }
        
        // Apply other settings as they become available in ModelOrchestrator
        if modular.ensemble_enabled {
            tracing::debug!("Would enable ensemble mode with {} models", modular.parallel_models);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_routing_strategy_conversions() {
        // Test modular to core
        assert_eq!(
            models::RoutingStrategy::from(ModularRoutingStrategy::RoundRobin),
            models::RoutingStrategy::LoadBased
        );
        assert_eq!(
            models::RoutingStrategy::from(ModularRoutingStrategy::LeastLatency),
            models::RoutingStrategy::LatencyOptimized
        );
        
        // Test core to modular
        assert_eq!(
            ModularRoutingStrategy::from(models::RoutingStrategy::LoadBased),
            ModularRoutingStrategy::RoundRobin
        );
        assert_eq!(
            ModularRoutingStrategy::from(models::RoutingStrategy::LatencyOptimized),
            ModularRoutingStrategy::LeastLatency
        );
    }
}