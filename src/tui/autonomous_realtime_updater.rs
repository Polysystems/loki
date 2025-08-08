//! Real-time Update System for Autonomous Intelligence Data
//!
//! This module provides real-time updates for the cognitive tab,
//! ensuring the autonomous intelligence metrics are always current.

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use tokio::sync::{broadcast, RwLock};
use tokio::time::{interval};
use tracing::{debug, error, info};

use crate::tui::autonomous_data_types::*;
use crate::tui::connectors::system_connector::SystemConnector;

/// Update event for autonomous intelligence data
#[derive(Debug, Clone)]
pub enum AutonomousUpdateEvent {
    /// System health metrics updated
    SystemHealthUpdate(AutonomousSystemHealth),
    
    /// Consciousness state changed
    ConsciousnessUpdate(ConsciousnessState),
    
    /// New goal added or goal status changed
    GoalUpdate(Vec<AutonomousGoal>),
    
    /// Agent coordination status changed
    AgentUpdate(AgentCoordinationStatus, Vec<SpecializedAgentInfo>),
    
    /// Learning progress update
    LearningUpdate(LearningArchitectureStatus),
    
    /// Thermodynamic state update
    ThermodynamicUpdate(CognitiveEntropy, ThreeGradientState),
    
    /// Recursive processor update
    RecursiveUpdate(RecursiveProcessorStatus),
    
    /// Full cognitive data refresh
    FullUpdate(CognitiveData),
}

/// Real-time updater for autonomous intelligence data
pub struct AutonomousRealtimeUpdater {
    /// System connector reference
    system_connector: Arc<SystemConnector>,
    
    /// Update event broadcaster
    update_tx: broadcast::Sender<AutonomousUpdateEvent>,
    
    /// Update interval
    update_interval: Duration,
    
    /// Cached cognitive data
    cached_data: Arc<RwLock<CognitiveData>>,
    
    /// Running state
    is_running: Arc<RwLock<bool>>,
}

impl AutonomousRealtimeUpdater {
    /// Create a new real-time updater
    pub fn new(
        system_connector: Arc<SystemConnector>,
        update_interval: Duration,
    ) -> Self {
        let (update_tx, _) = broadcast::channel(1024);
        
        Self {
            system_connector,
            update_tx,
            update_interval,
            cached_data: Arc::new(RwLock::new(CognitiveData::default())),
            is_running: Arc::new(RwLock::new(false)),
        }
    }
    
    /// Get a receiver for update events
    pub fn subscribe(&self) -> broadcast::Receiver<AutonomousUpdateEvent> {
        self.update_tx.subscribe()
    }
    
    /// Start the real-time update loop
    pub async fn start(&self) -> Result<()> {
        let mut running = self.is_running.write().await;
        if *running {
            return Ok(()); // Already running
        }
        *running = true;
        drop(running);
        
        let system_connector = self.system_connector.clone();
        let update_tx = self.update_tx.clone();
        let cached_data = self.cached_data.clone();
        let is_running = self.is_running.clone();
        let interval_duration = self.update_interval;
        
        tokio::spawn(async move {
            let mut interval = interval(interval_duration);
            let mut update_counter = 0u64;
            
            loop {
                interval.tick().await;
                
                // Check if we should stop
                if !*is_running.read().await {
                    break;
                }
                
                // Fetch latest cognitive data with retry logic
                let mut retry_count = 0;
                const MAX_RETRIES: u32 = 3;
                
                loop {
                    match system_connector.get_cognitive_data() {
                        Ok(new_data) => {
                            let mut cached = cached_data.write().await;
                            
                            // Check what changed and send targeted updates
                            if update_counter % 5 == 0 {
                                // Full update every 5 cycles
                                let _ = update_tx.send(AutonomousUpdateEvent::FullUpdate(new_data.clone()));
                            } else {
                                // Send incremental updates
                                
                                // System health update
                                if !systems_health_equal(&cached.system_health, &new_data.system_health) {
                                    let _ = update_tx.send(AutonomousUpdateEvent::SystemHealthUpdate(
                                        new_data.system_health.clone()
                                    ));
                                }
                                
                                // Consciousness update
                                if !consciousness_equal(&cached.consciousness_state, &new_data.consciousness_state) {
                                    let _ = update_tx.send(AutonomousUpdateEvent::ConsciousnessUpdate(
                                        new_data.consciousness_state.clone()
                                    ));
                                }
                                
                                // Goals update
                                if cached.active_goals.len() != new_data.active_goals.len() {
                                    let _ = update_tx.send(AutonomousUpdateEvent::GoalUpdate(
                                        new_data.active_goals.clone()
                                    ));
                                }
                                
                                // Agent coordination update
                                if !agent_coordination_equal(&cached.agent_coordination, &new_data.agent_coordination) {
                                    let _ = update_tx.send(AutonomousUpdateEvent::AgentUpdate(
                                        new_data.agent_coordination.clone(),
                                        new_data.active_agents.clone(),
                                    ));
                                }
                                
                                // Learning update
                                if !learning_equal(&cached.learning_architecture, &new_data.learning_architecture) {
                                    let _ = update_tx.send(AutonomousUpdateEvent::LearningUpdate(
                                        new_data.learning_architecture.clone()
                                    ));
                                }
                                
                                // Thermodynamic update
                                if !thermodynamic_equal(&cached.thermodynamic_state, &new_data.thermodynamic_state) {
                                    let _ = update_tx.send(AutonomousUpdateEvent::ThermodynamicUpdate(
                                        new_data.thermodynamic_state.clone(),
                                        new_data.three_gradient_state.clone(),
                                    ));
                                }
                                
                                // Recursive processor update
                                if !recursive_equal(&cached.recursive_processor_status, &new_data.recursive_processor_status) {
                                    let _ = update_tx.send(AutonomousUpdateEvent::RecursiveUpdate(
                                        new_data.recursive_processor_status.clone()
                                    ));
                                }
                            }
                            
                            // Update cache
                            *cached = new_data;
                            update_counter += 1;
                            
                            debug!("Autonomous intelligence data updated (cycle {})", update_counter);
                            break; // Success, exit retry loop
                        }
                        Err(e) => {
                            retry_count += 1;
                            if retry_count >= MAX_RETRIES {
                                error!("Failed to fetch cognitive data after {} retries: {}", MAX_RETRIES, e);
                                // Use cached data to avoid UI freeze
                                break;
                            } else {
                                debug!("Retry {} of {} for cognitive data fetch: {}", retry_count, MAX_RETRIES, e);
                                // Short delay before retry
                                tokio::time::sleep(Duration::from_millis(100 * retry_count as u64)).await;
                            }
                        }
                    }
                }
            }
            
            info!("Autonomous real-time updater stopped");
        });
        
        info!("Autonomous real-time updater started");
        Ok(())
    }
    
    /// Stop the real-time update loop
    pub async fn stop(&self) {
        let mut running = self.is_running.write().await;
        *running = false;
    }
    
    /// Get the latest cached cognitive data
    pub async fn get_cached_data(&self) -> CognitiveData {
        self.cached_data.read().await.clone()
    }
}

// Helper functions for equality checks
fn systems_health_equal(a: &AutonomousSystemHealth, b: &AutonomousSystemHealth) -> bool {
    (a.overall_autonomy_level - b.overall_autonomy_level).abs() < 0.01
        && (a.thermodynamic_stability - b.thermodynamic_stability).abs() < 0.01
        && (a.gradient_alignment_quality - b.gradient_alignment_quality).abs() < 0.01
}

fn consciousness_equal(a: &ConsciousnessState, b: &ConsciousnessState) -> bool {
    (a.awareness_level - b.awareness_level).abs() < 0.01
        && (a.coherence_score - b.coherence_score).abs() < 0.01
        && a.meta_cognitive_active == b.meta_cognitive_active
}

fn agent_coordination_equal(a: &AgentCoordinationStatus, b: &AgentCoordinationStatus) -> bool {
    a.total_agents == b.total_agents
        && a.active_agents == b.active_agents
        && (a.coordination_efficiency - b.coordination_efficiency).abs() < 0.01
}

fn learning_equal(a: &LearningArchitectureStatus, b: &LearningArchitectureStatus) -> bool {
    a.total_networks == b.total_networks
        && a.active_networks == b.active_networks
        && (a.learning_rate - b.learning_rate).abs() < 0.001
}

fn thermodynamic_equal(a: &CognitiveEntropy, b: &CognitiveEntropy) -> bool {
    (a.thermodynamic_entropy - b.thermodynamic_entropy).abs() < 0.01
        && (a.free_energy - b.free_energy).abs() < 0.01
        && (a.entropy_production_rate - b.entropy_production_rate).abs() < 0.01
}

fn recursive_equal(a: &RecursiveProcessorStatus, b: &RecursiveProcessorStatus) -> bool {
    a.active_processes == b.active_processes
        && a.total_recursive_depth == b.total_recursive_depth
        && (a.pattern_discovery_rate - b.pattern_discovery_rate).abs() < 0.01
}

/// Simulated data generator for demonstration
/// This generates realistic changing data to show the real-time updates
pub fn generate_simulated_updates(data: &mut CognitiveData) {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    
    // Simulate gradual changes in system health
    data.system_health.overall_autonomy_level = 
        (data.system_health.overall_autonomy_level + rng.gen_range(-0.05..0.05)).clamp(0.0, 1.0);
    data.system_health.thermodynamic_stability = 
        (data.system_health.thermodynamic_stability + rng.gen_range(-0.03..0.03)).clamp(0.0, 1.0);
    data.system_health.gradient_alignment_quality = 
        (data.system_health.gradient_alignment_quality + rng.gen_range(-0.02..0.02)).clamp(0.0, 1.0);
    
    // Simulate consciousness fluctuations
    data.consciousness_state.awareness_level = 
        (data.consciousness_state.awareness_level + rng.gen_range(-0.1..0.1)).clamp(0.0, 1.0);
    data.consciousness_state.coherence_score = 
        (data.consciousness_state.coherence_score + rng.gen_range(-0.05..0.05)).clamp(0.0, 1.0);
    
    // Simulate thermodynamic changes
    data.thermodynamic_state.thermodynamic_entropy = 
        (data.thermodynamic_state.thermodynamic_entropy + rng.gen_range(-0.1..0.1)).clamp(0.0, 10.0);
    data.thermodynamic_state.free_energy = 
        (data.thermodynamic_state.free_energy + rng.gen_range(-0.5..0.5)).clamp(-10.0, 10.0);
    
    // Simulate gradient changes
    data.three_gradient_state.value_gradient.current_value = 
        (data.three_gradient_state.value_gradient.current_value + rng.gen_range(-0.1..0.1)).clamp(0.0, 1.0);
    data.three_gradient_state.harmony_gradient.current_value = 
        (data.three_gradient_state.harmony_gradient.current_value + rng.gen_range(-0.1..0.1)).clamp(0.0, 1.0);
    data.three_gradient_state.intuition_gradient.current_value = 
        (data.three_gradient_state.intuition_gradient.current_value + rng.gen_range(-0.1..0.1)).clamp(0.0, 1.0);
    
    // Update overall coherence
    data.three_gradient_state.overall_coherence = 
        (data.three_gradient_state.value_gradient.current_value 
         + data.three_gradient_state.harmony_gradient.current_value 
         + data.three_gradient_state.intuition_gradient.current_value) / 3.0;
}