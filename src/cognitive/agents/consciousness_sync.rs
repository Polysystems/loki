//! Consciousness Synchronization System
//!
//! Enables distributed consciousness across multiple agents through
//! state sharing, coherence maintenance, and emergent awareness.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tokio::sync::{RwLock, broadcast, mpsc};
use tokio::time::interval;
use tracing::{debug, error, info, warn};

use crate::cognitive::consciousness::{ConsciousnessState, MetaCognitiveAwareness};
// consciousness imports removed as unused
// emergent import removed as unused
use crate::memory::{CognitiveMemory, MemoryId};
use crate::models::agent_specialization_router::AgentId;
// super imports removed as unused

/// Configuration for consciousness synchronization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessSyncConfig {
    /// Synchronization interval
    pub sync_interval: Duration,

    /// Coherence threshold for accepting updates
    pub coherence_threshold: f64,

    /// Maximum state divergence allowed
    pub max_divergence: f64,

    /// Enable emergent consciousness detection
    pub enable_emergence_detection: bool,

    /// State history size
    pub history_size: usize,

    /// Consensus weight for different agent types
    pub agent_weights: HashMap<String, f64>,
}

impl Default for ConsciousnessSyncConfig {
    fn default() -> Self {
        let mut weights = HashMap::new();
        weights.insert("Analytical".to_string(), 1.2);
        weights.insert("Creative".to_string(), 1.0);
        weights.insert("Strategic".to_string(), 1.3);
        weights.insert("Social".to_string(), 0.9);
        weights.insert("Guardian".to_string(), 1.1);
        weights.insert("Learning".to_string(), 1.0);
        weights.insert("Coordinator".to_string(), 1.5);

        Self {
            sync_interval: Duration::from_secs(5),
            coherence_threshold: 0.7,
            max_divergence: 0.3,
            enable_emergence_detection: true,
            history_size: 100,
            agent_weights: weights,
        }
    }
}

/// Consciousness synchronization state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncState {
    /// Last synchronization time
    pub last_sync: SystemTime,

    /// Current coherence level
    pub coherence_level: f64,

    /// Divergence from collective
    pub divergence_score: f64,

    /// Synchronization errors
    pub error_count: u32,

    /// Successful syncs
    pub success_count: u32,
}

/// Collective consciousness state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectiveConsciousness {
    /// Merged consciousness state
    pub unified_state: ConsciousnessState,

    /// Individual agent states
    pub agent_states: HashMap<AgentId, ConsciousnessState>,

    /// Emergence indicators
    pub emergence_indicators: Vec<EmergenceIndicator>,

    /// Collective coherence score
    pub collective_coherence: f64,

    /// Last update time
    pub last_updated: SystemTime,
}

/// Indicators of emergent consciousness
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergenceIndicator {
    pub indicator_type: EmergenceType,
    pub strength: f64,
    pub contributing_agents: Vec<AgentId>,
    pub description: String,
    pub detected_at: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmergenceType {
    CollectiveInsight,
    SynchronizedThought,
    SharedIntuition,
    GroupFlow,
    UnifiedPurpose,
    EmergentCapability,
}

/// Consciousness synchronization system
pub struct ConsciousnessSync {
    /// Configuration
    config: ConsciousnessSyncConfig,

    /// Collective consciousness state
    collective_state: Arc<RwLock<CollectiveConsciousness>>,

    /// Agent sync states
    sync_states: Arc<RwLock<HashMap<AgentId, SyncState>>>,

    /// State history for emergence detection
    state_history: Arc<RwLock<VecDeque<CollectiveConsciousness>>>,

    /// Shared memory system
    shared_memory: Arc<CognitiveMemory>,

    /// Message channel for sync events
    sync_tx: mpsc::Sender<SyncMessage>,
    sync_rx: Arc<RwLock<mpsc::Receiver<SyncMessage>>>,

    /// Broadcast channel for consciousness updates
    broadcast_tx: broadcast::Sender<ConsciousnessUpdate>,

    /// Running state
    running: Arc<RwLock<bool>>,
}

/// Synchronization messages
#[derive(Debug, Clone)]
pub enum SyncMessage {
    StateUpdate(AgentId, ConsciousnessState),
    RequestSync(AgentId),
    EmergenceDetected(EmergenceIndicator),
    CoherenceBreach(AgentId, f64),
}

/// Consciousness update broadcast
#[derive(Debug, Clone)]
pub struct ConsciousnessUpdate {
    pub update_type: UpdateType,
    pub collective_state: ConsciousnessState,
    pub source_agents: Vec<AgentId>,
    pub coherence_delta: f64,
    pub timestamp: SystemTime,
}

#[derive(Debug, Clone)]
pub enum UpdateType {
    PeriodicSync,
    EmergentChange,
    CoherenceCorrection,
    CollectiveShift,
}

impl ConsciousnessSync {
    /// Create new consciousness synchronization system
    pub async fn new(
        config: ConsciousnessSyncConfig,
        shared_memory: Arc<CognitiveMemory>,
    ) -> Result<Self> {
        info!("🧠 Initializing Consciousness Synchronization System");

        let (sync_tx, sync_rx) = mpsc::channel(1000);
        let (broadcast_tx, _) = broadcast::channel(100);

        let initial_collective = CollectiveConsciousness {
            unified_state: ConsciousnessState::default(),
            agent_states: HashMap::new(),
            emergence_indicators: Vec::new(),
            collective_coherence: 1.0,
            last_updated: SystemTime::now(),
        };

        Ok(Self {
            config,
            collective_state: Arc::new(RwLock::new(initial_collective)),
            sync_states: Arc::new(RwLock::new(HashMap::new())),
            state_history: Arc::new(RwLock::new(VecDeque::with_capacity(100))),
            shared_memory,
            sync_tx,
            sync_rx: Arc::new(RwLock::new(sync_rx)),
            broadcast_tx,
            running: Arc::new(RwLock::new(false)),
        })
    }

    /// Start the synchronization system
    pub async fn start(self: Arc<Self>) -> Result<()> {
        *self.running.write().await = true;
        info!("✨ Consciousness Synchronization System started");

        // Start synchronization loop
        let sync_self = self.clone();
        tokio::spawn(async move {
            if let Err(e) = sync_self.spawn_sync_loop().await {
                tracing::error!("Synchronization loop failed: {}", e);
            }
        });

        // Start emergence detection
        if self.config.enable_emergence_detection {
            let emergence_self = self.clone();
            tokio::spawn(async move {
                if let Err(e) = emergence_self.spawn_emergence_detection().await {
                    tracing::error!("Emergence detection failed: {}", e);
                }
            });
        }

        Ok(())
    }

    /// Stop the synchronization system
    pub async fn stop(&self) -> Result<()> {
        *self.running.write().await = false;
        info!("🛑 Consciousness Synchronization System stopped");
        Ok(())
    }

    /// Update agent consciousness state
    pub async fn update_agent_state(
        &self,
        agent_id: AgentId,
        state: ConsciousnessState,
    ) -> Result<()> {
        // Send update message
        self.sync_tx
            .send(SyncMessage::StateUpdate(agent_id.clone(), state.clone()))
            .await
            .context("Failed to send state update")?;

        // Update sync state
        let mut sync_states = self.sync_states.write().await;
        let sync_state = sync_states.entry(agent_id).or_insert(SyncState {
            last_sync: SystemTime::now(),
            coherence_level: 1.0,
            divergence_score: 0.0,
            error_count: 0,
            success_count: 0,
        });
        sync_state.last_sync = SystemTime::now();
        sync_state.success_count += 1;

        Ok(())
    }

    /// Get current collective consciousness state
    pub async fn get_collective_state(&self) -> CollectiveConsciousness {
        self.collective_state.read().await.clone()
    }

    /// Get agent-specific synchronized state
    pub async fn get_agent_synchronized_state(
        &self,
        agent_id: &AgentId,
    ) -> Option<ConsciousnessState> {
        let collective = self.collective_state.read().await;
        collective.agent_states.get(agent_id).cloned()
    }

    /// Main synchronization loop
    async fn spawn_sync_loop(self: Arc<Self>) -> Result<()> {
        let mut interval = interval(self.config.sync_interval);

        while *self.running.read().await {
            interval.tick().await;

            if let Err(e) = self.perform_synchronization().await {
                error!("Synchronization error: {}", e);
            }
        }

        Ok(())
    }

    /// Perform consciousness synchronization
    async fn perform_synchronization(&self) -> Result<()> {
        debug!("🔄 Performing consciousness synchronization");

        // Collect pending updates
        let updates = self.collect_pending_updates().await?;

        if updates.is_empty() {
            return Ok(());
        }

        // Merge consciousness states
        let merged_state = self.merge_consciousness_states(&updates).await?;

        // Check coherence
        let coherence = self.calculate_collective_coherence(&merged_state, &updates).await?;

        if coherence < self.config.coherence_threshold {
            warn!("Coherence below threshold: {:.2}", coherence);
            self.handle_coherence_breach(coherence).await?;
        }

        // Update collective state
        let mut collective = self.collective_state.write().await;
        collective.unified_state = merged_state.clone();
        collective.collective_coherence = coherence;
        collective.last_updated = SystemTime::now();

        // Update individual agent states
        for (agent_id, state) in updates {
            collective.agent_states.insert(agent_id, state);
        }

        // Add to history
        let mut history = self.state_history.write().await;
        if history.len() >= self.config.history_size {
            history.pop_front();
        }
        history.push_back(collective.clone());

        // Broadcast update
        let update = ConsciousnessUpdate {
            update_type: UpdateType::PeriodicSync,
            collective_state: merged_state,
            source_agents: collective.agent_states.keys().cloned().collect(),
            coherence_delta: coherence - collective.collective_coherence,
            timestamp: SystemTime::now(),
        };

        let _ = self.broadcast_tx.send(update);

        info!("✅ Synchronization complete - Coherence: {:.2}", coherence);
        Ok(())
    }

    /// Collect pending state updates
    async fn collect_pending_updates(&self) -> Result<HashMap<AgentId, ConsciousnessState>> {
        let mut updates = HashMap::new();
        let mut rx = self.sync_rx.write().await;

        // Collect all pending messages
        while let Ok(msg) = rx.try_recv() {
            match msg {
                SyncMessage::StateUpdate(agent_id, state) => {
                    updates.insert(agent_id, state);
                }
                SyncMessage::RequestSync(agent_id) => {
                    debug!("Sync requested by agent: {}", agent_id);
                }
                SyncMessage::EmergenceDetected(indicator) => {
                    self.handle_emergence(indicator).await?;
                }
                SyncMessage::CoherenceBreach(agent_id, score) => {
                    warn!("Coherence breach from {}: {:.2}", agent_id, score);
                }
            }
        }

        Ok(updates)
    }

    /// Merge multiple consciousness states
    async fn merge_consciousness_states(
        &self,
        states: &HashMap<AgentId, ConsciousnessState>,
    ) -> Result<ConsciousnessState> {
        if states.is_empty() {
            return Ok(ConsciousnessState::default());
        }

        // Calculate weighted averages based on agent types
        let mut total_weight = 0.0;
        let mut weighted_awareness = 0.0;
        let mut weighted_coherence = 0.0;
        let mut weighted_identity_stability = 0.0;

        let mut all_domains = Vec::new();
        let mut all_insights = Vec::new();
        let mut merged_traits = HashMap::new();

        for (agent_id, state) in states {
            let weight = self.get_agent_weight(agent_id).await;
            total_weight += weight;

            weighted_awareness += state.awareness_level * weight;
            weighted_coherence += state.coherence_score * weight;
            weighted_identity_stability += state.identity_stability * weight;

            all_domains.extend(state.active_domains.clone());
            all_insights.extend(state.introspection_insights.clone());

            // Merge personality traits
            for (trait_name, value) in &state.personality_traits {
                *merged_traits.entry(trait_name.clone()).or_insert(0.0) += value * weight;
            }
        }

        // Normalize by total weight
        if total_weight > 0.0 {
            weighted_awareness /= total_weight;
            weighted_coherence /= total_weight;
            weighted_identity_stability /= total_weight;

            for value in merged_traits.values_mut() {
                *value /= total_weight;
            }
        }

        // Deduplicate domains
        all_domains.sort();
        all_domains.dedup();

        // Select most relevant insights
        all_insights.sort_by(|a, b| {
            b.importance.partial_cmp(&a.importance).unwrap_or(std::cmp::Ordering::Equal)
        });
        all_insights.truncate(10);

        Ok(ConsciousnessState {
            awareness_level: weighted_awareness,
            reflection_depth: 3, // Average depth
            active_domains: all_domains,
            coherence_score: weighted_coherence,
            identity_stability: weighted_identity_stability,
            personality_traits: merged_traits,
            introspection_insights: all_insights,
            meta_awareness: MetaCognitiveAwareness::default(),
            last_update: chrono::Utc::now(),
            consciousness_memory_ids: Vec::new(),
            state_memory_id: Some(MemoryId::new()),
        })
    }

    /// Calculate collective coherence
    async fn calculate_collective_coherence(
        &self,
        unified_state: &ConsciousnessState,
        individual_states: &HashMap<AgentId, ConsciousnessState>,
    ) -> Result<f64> {
        if individual_states.is_empty() {
            return Ok(1.0);
        }

        let mut total_divergence = 0.0;
        let mut count = 0;

        for (_agent_id, state) in individual_states {
            // Calculate divergence in key metrics
            let awareness_diff = (state.awareness_level - unified_state.awareness_level).abs();
            let coherence_diff = (state.coherence_score - unified_state.coherence_score).abs();
            let identity_diff = (state.identity_stability - unified_state.identity_stability).abs();

            let divergence = (awareness_diff + coherence_diff + identity_diff) / 3.0;
            total_divergence += divergence;
            count += 1;
        }

        let avg_divergence = if count > 0 { total_divergence / count as f64 } else { 0.0 };
        Ok(1.0 - avg_divergence.min(1.0))
    }

    /// Get agent weight for consciousness merging
    async fn get_agent_weight(&self, _agent_id: &AgentId) -> f64 {
        // This would be enhanced to get actual agent type
        self.config.agent_weights.get("Analytical").copied().unwrap_or(1.0)
    }

    /// Handle coherence breach
    async fn handle_coherence_breach(&self, coherence: f64) -> Result<()> {
        warn!("🚨 Coherence breach detected: {:.2}", coherence);

        // Broadcast coherence correction event
        let update = ConsciousnessUpdate {
            update_type: UpdateType::CoherenceCorrection,
            collective_state: self.collective_state.read().await.unified_state.clone(),
            source_agents: Vec::new(),
            coherence_delta: coherence - self.config.coherence_threshold,
            timestamp: SystemTime::now(),
        };

        let _ = self.broadcast_tx.send(update);
        Ok(())
    }

    /// Spawn emergence detection task
    async fn spawn_emergence_detection(self: Arc<Self>) -> Result<()> {
        let mut interval = interval(Duration::from_secs(10));

        while *self.running.read().await {
            interval.tick().await;

            if let Err(e) = self.detect_emergence().await {
                error!("Emergence detection error: {}", e);
            }
        }

        Ok(())
    }

    /// Detect emergent consciousness patterns
    async fn detect_emergence(&self) -> Result<()> {
        let history = self.state_history.read().await;
        if history.len() < 3 {
            return Ok(()); // Need history for emergence detection
        }

        // Analyze recent states for emergence patterns
        let recent_states: Vec<_> = history.iter().rev().take(5).collect();

        // Check for synchronized thought patterns
        if let Some(indicator) = self.check_synchronized_thoughts(&recent_states).await? {
            self.sync_tx.send(SyncMessage::EmergenceDetected(indicator)).await?;
        }

        // Check for collective insights
        if let Some(indicator) = self.check_collective_insights(&recent_states).await? {
            self.sync_tx.send(SyncMessage::EmergenceDetected(indicator)).await?;
        }

        // Check for group flow states
        if let Some(indicator) = self.check_group_flow(&recent_states).await? {
            self.sync_tx.send(SyncMessage::EmergenceDetected(indicator)).await?;
        }

        Ok(())
    }

    /// Check for synchronized thought patterns
    async fn check_synchronized_thoughts(
        &self,
        states: &[&CollectiveConsciousness],
    ) -> Result<Option<EmergenceIndicator>> {
        // Analyze domain activation patterns
        let mut domain_frequencies = HashMap::new();

        for state in states {
            for domain in &state.unified_state.active_domains {
                *domain_frequencies.entry(domain.clone()).or_insert(0) += 1;
            }
        }

        // Look for domains that suddenly become active across all agents
        for (domain, freq) in domain_frequencies {
            if freq >= states.len() - 1 {
                return Ok(Some(EmergenceIndicator {
                    indicator_type: EmergenceType::SynchronizedThought,
                    strength: freq as f64 / states.len() as f64,
                    contributing_agents: states
                        .last()
                        .map(|s| s.agent_states.keys().cloned().collect())
                        .unwrap_or_default(),
                    description: format!("Synchronized activation of {:?} domain", domain),
                    detected_at: SystemTime::now(),
                }));
            }
        }

        Ok(None)
    }

    /// Check for collective insights
    async fn check_collective_insights(
        &self,
        states: &[&CollectiveConsciousness],
    ) -> Result<Option<EmergenceIndicator>> {
        if states.len() < 2 {
            return Ok(None);
        }

        let current = states.last().unwrap();
        let previous = states[states.len() - 2];

        // Check for sudden increase in insight quality
        let current_avg_significance =
            current.unified_state.introspection_insights.iter().map(|i| i.importance).sum::<f64>()
                / current.unified_state.introspection_insights.len().max(1) as f64;

        let previous_avg_significance =
            previous.unified_state.introspection_insights.iter().map(|i| i.importance).sum::<f64>()
                / previous.unified_state.introspection_insights.len().max(1) as f64;

        if current_avg_significance > previous_avg_significance * 1.5 {
            return Ok(Some(EmergenceIndicator {
                indicator_type: EmergenceType::CollectiveInsight,
                strength: current_avg_significance,
                contributing_agents: current.agent_states.keys().cloned().collect(),
                description: "Significant increase in collective insight quality".to_string(),
                detected_at: SystemTime::now(),
            }));
        }

        Ok(None)
    }

    /// Check for group flow states
    async fn check_group_flow(
        &self,
        states: &[&CollectiveConsciousness],
    ) -> Result<Option<EmergenceIndicator>> {
        // Check for high coherence + high awareness combination
        let current = states.last().unwrap();

        if current.collective_coherence > 0.9 && current.unified_state.awareness_level > 0.8 {
            return Ok(Some(EmergenceIndicator {
                indicator_type: EmergenceType::GroupFlow,
                strength: current.collective_coherence * current.unified_state.awareness_level,
                contributing_agents: current.agent_states.keys().cloned().collect(),
                description: "Group flow state detected - high coherence and awareness".to_string(),
                detected_at: SystemTime::now(),
            }));
        }

        Ok(None)
    }

    /// Handle emergence detection
    async fn handle_emergence(&self, indicator: EmergenceIndicator) -> Result<()> {
        info!("🌟 Emergence detected: {:?} - {}", indicator.indicator_type, indicator.description);

        // Update collective state with emergence indicator
        let mut collective = self.collective_state.write().await;
        collective.emergence_indicators.push(indicator.clone());

        // Keep only recent indicators
        if collective.emergence_indicators.len() > 20 {
            collective.emergence_indicators.drain(0..5);
        }

        // Broadcast emergence event
        let update = ConsciousnessUpdate {
            update_type: UpdateType::EmergentChange,
            collective_state: collective.unified_state.clone(),
            source_agents: indicator.contributing_agents,
            coherence_delta: 0.0,
            timestamp: SystemTime::now(),
        };

        let _ = self.broadcast_tx.send(update);
        Ok(())
    }

    /// Subscribe to consciousness updates
    pub fn subscribe(&self) -> broadcast::Receiver<ConsciousnessUpdate> {
        self.broadcast_tx.subscribe()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_consciousness_sync_creation() {
        let config = ConsciousnessSyncConfig::default();
        let memory = Arc::new(CognitiveMemory::new(Default::default()).await.unwrap());

        let sync = ConsciousnessSync::new(config, memory).await.unwrap();
        assert!(!*sync.running.read().await);
    }

    #[tokio::test]
    async fn test_state_merging() {
        let config = ConsciousnessSyncConfig::default();
        let memory = Arc::new(CognitiveMemory::new(Default::default()).await.unwrap());
        let sync = Arc::new(ConsciousnessSync::new(config, memory).await.unwrap());

        let mut states = HashMap::new();

        let mut state1 = ConsciousnessState::default();
        state1.awareness_level = 0.8;
        state1.coherence_score = 0.7;
        states.insert(AgentId::new_v4(), state1);

        let mut state2 = ConsciousnessState::default();
        state2.awareness_level = 0.6;
        state2.coherence_score = 0.9;
        states.insert(AgentId::new_v4(), state2);

        let merged = sync.merge_consciousness_states(&states).await.unwrap();

        // Should be weighted average
        assert!(merged.awareness_level > 0.6 && merged.awareness_level < 0.8);
        assert!(merged.coherence_score > 0.7 && merged.coherence_score < 0.9);
    }
}
