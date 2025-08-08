//! Coordination Protocol
//!
//! Defines the communication protocol for multi-agent coordination,
//! including message formats, priorities, and synchronization mechanisms.

use std::sync::Arc;
use std::time::{Duration, SystemTime};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, info};
use crate::cognitive::consciousness::ConsciousnessState;
use crate::models::agent_specialization_router::AgentId;

/// Coordination protocol for agent communication
pub struct CoordinationProtocol {
    /// Protocol version
    version: String,

    /// Message history
    message_history: Arc<RwLock<Vec<CoordinationMessage>>>,

    /// Synchronization interval
    sync_interval: Duration,

    /// Protocol metrics
    metrics: Arc<RwLock<ProtocolMetrics>>,
}

/// Coordination message structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinationMessage {
    /// Unique message ID
    pub id: String,

    /// Sender agent ID
    pub sender: AgentId,

    /// Message type
    pub message_type: MessageType,

    /// Message priority
    pub priority: MessagePriority,

    /// Message payload
    pub payload: serde_json::Value,

    /// Timestamp
    pub timestamp: SystemTime,
}

/// Message types for coordination
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MessageType {
    /// Request for consensus
    ConsensusRequest,

    /// Vote on consensus
    ConsensusVote,

    /// Task assignment
    TaskAssignment,

    /// Task completion notification
    TaskCompletion,

    /// Status update
    StatusUpdate,

    /// Knowledge sharing
    KnowledgeShare,

    /// Synchronization message
    Synchronization,

    /// Emergency signal
    Emergency,

    /// Emergence detection signal
    EmergenceSignal,

    /// Heartbeat
    Heartbeat,

    /// Consciousness state update
    ConsciousnessUpdate,
}

/// Message priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MessagePriority {
    Low,
    Normal,
    High,
    Critical,
}

/// Protocol metrics
#[derive(Debug, Clone, Default)]
pub struct ProtocolMetrics {
    pub messages_sent: u64,
    pub messages_received: u64,
    pub sync_events: u64,
    pub failed_deliveries: u64,
    pub average_latency: Duration,
}

impl CoordinationProtocol {
    /// Create a new coordination protocol
    pub async fn new(sync_interval: Duration) -> Result<Self> {
        Ok(Self {
            version: "1.0.0".to_string(),
            message_history: Arc::new(RwLock::new(Vec::with_capacity(1000))),
            sync_interval,
            metrics: Arc::new(RwLock::new(ProtocolMetrics::default())),
        })
    }

    /// Create a synchronization message
    pub async fn create_sync_message(&self, state: &ConsciousnessState) -> Result<CoordinationMessage> {
        let sync_data = SynchronizationData {
            consciousness_state: state.clone(),
            timestamp: SystemTime::now(),
            protocol_version: self.version.clone(),
        };

        Ok(CoordinationMessage {
            id: uuid::Uuid::new_v4().to_string(),
            sender: AgentId::new_v4(),
            message_type: MessageType::Synchronization,
            priority: MessagePriority::Normal,
            payload: serde_json::to_value(sync_data)?,
            timestamp: SystemTime::now(),
        })
    }

    /// Send a message
    pub async fn send_message(
        &self,
        sender: AgentId,
        message_type: MessageType,
        priority: MessagePriority,
        payload: serde_json::Value,
    ) -> Result<CoordinationMessage> {
        let message = CoordinationMessage {
            id: uuid::Uuid::new_v4().to_string(),
            sender,
            message_type,
            priority,
            payload,
            timestamp: SystemTime::now(),
        };

        // Store in history
        self.message_history.write().await.push(message.clone());

        // Update metrics
        let mut metrics = self.metrics.write().await;
        metrics.messages_sent += 1;

        debug!("Sent coordination message: {:?}", message.message_type);

        Ok(message)
    }

    /// Process received message
    pub async fn process_message(&self, message: &CoordinationMessage) -> Result<()> {
        debug!("Processing message: {:?} from {}", message.message_type, message.sender);

        // Update metrics
        let mut metrics = self.metrics.write().await;
        metrics.messages_received += 1;

        // Store in history
        self.message_history.write().await.push(message.clone());

        // Process based on type
        match &message.message_type {
            MessageType::Emergency => {
                info!("Emergency message received from {}", message.sender);
                // Handle emergency protocols
            }
            MessageType::Synchronization => {
                metrics.sync_events += 1;
            }
            _ => {}
        }

        Ok(())
    }

    /// Get message history
    pub async fn get_message_history(&self, limit: usize) -> Result<Vec<CoordinationMessage>> {
        let history = self.message_history.read().await;
        let start = history.len().saturating_sub(limit);
        Ok(history[start..].to_vec())
    }

    /// Get protocol metrics
    pub async fn get_metrics(&self) -> Result<ProtocolMetrics> {
        Ok(self.metrics.read().await.clone())
    }

    /// Clear old messages from history
    pub async fn cleanup_history(&self, retention: Duration) -> Result<usize> {
        let cutoff = SystemTime::now() - retention;
        let mut history = self.message_history.write().await;
        let original_len = history.len();

        history.retain(|msg| msg.timestamp > cutoff);

        let removed = original_len - history.len();
        debug!("Cleaned up {} old messages", removed);

        Ok(removed)
    }
}

/// Synchronization data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynchronizationData {
    pub consciousness_state: ConsciousnessState,
    pub timestamp: SystemTime,
    pub protocol_version: String,
}

/// Message routing information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageRoute {
    pub from: AgentId,
    pub to: Vec<AgentId>,
    pub via: Vec<AgentId>,
    pub hops: u32,
}

/// Message acknowledgment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageAck {
    pub message_id: String,
    pub agent_id: AgentId,
    pub received_at: SystemTime,
    pub processed: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_coordination_protocol() {
        let protocol = CoordinationProtocol::new(Duration::from_secs(60))
            .await
            .unwrap();

        // Test message sending
        let sender = AgentId::new_v4();
        let message = protocol.send_message(
            sender.clone(),
            MessageType::StatusUpdate,
            MessagePriority::Normal,
            serde_json::json!({"status": "active"}),
        ).await.unwrap();

        assert_eq!(message.sender, sender);
        assert_eq!(message.message_type, MessageType::StatusUpdate);

        // Test metrics
        let metrics = protocol.get_metrics().await.unwrap();
        assert_eq!(metrics.messages_sent, 1);
    }
}
