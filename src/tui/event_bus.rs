//! Unified Event Bus System for TUI
//! 
//! This module provides the central event-driven communication system that connects
//! all tabs and components in the TUI, enabling seamless cross-tab interaction.

use std::collections::{HashMap, VecDeque, BinaryHeap};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, AtomicU64, Ordering};
use std::time::{Duration, Instant};
use dashmap::DashMap;
use crossbeam_channel::{unbounded, Sender, Receiver};
use tokio::sync::mpsc;
use bytes::Bytes;
use serde::{Serialize, Deserialize};
use serde_json::Value;
use tracing::{debug, info, error};
use anyhow::Result;

use crate::infrastructure::lockfree::{LockFreeEventQueue, EventRouter, Event as LockFreeEvent, EventPriority as LockFreePriority, IndexedRingBuffer, HasTimestamp};

use crate::tools::{ToolResult, ToolStatus};
use crate::cognitive::agents::AgentSpecialization;

/// Wrapper for SystemEvent with timestamp for lock-free storage
#[derive(Debug, Clone)]
pub struct EventHistoryEntry {
    pub event: SystemEvent,
    pub timestamp: Instant,
}

impl HasTimestamp for EventHistoryEntry {
    fn timestamp(&self) -> Instant {
        self.timestamp
    }
}

/// Unique identifier for tabs
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TabId {
    Home,
    Chat,
    Utilities,
    Memory,
    Cognitive,
    Settings,
    System, // For system-level events
    Custom(String), // For custom/dynamic tabs like editor
}

/// Event priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EventPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// System-wide events that can be published and subscribed to
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SystemEvent {
    // Tool events
    ToolConfigured {
        tool_id: String,
        config: Value,
        source: TabId,
    },
    ToolExecuted {
        tool_id: String,
        params: Value,
        result: ToolResult,
        duration: Duration,
        source: TabId,
    },
    ToolStatusChanged {
        tool_id: String,
        status: ToolStatus,
    },
    
    // Model events
    ModelDiscovered {
        provider: String,
        models: Vec<String>,
        source: TabId,
    },
    ModelSelected {
        model_id: String,
        source: TabId,
    },
    ModelBenchmarked {
        model_id: String,
        metrics: BenchmarkMetrics,
    },
    
    // Agent events
    AgentCreated {
        agent_id: String,
        specialization: AgentSpecialization,
        config: Value,
    },
    AgentTaskAssigned {
        agent_id: String,
        task_id: String,
        task_description: String,
    },
    AgentStatusChanged {
        agent_id: String,
        status: String,
    },
    
    // Memory events
    MemoryStored {
        key: String,
        value_type: String,
        source: TabId,
    },
    ContextRetrieved {
        query: String,
        result_count: usize,
        source: TabId,
    },
    KnowledgeGraphUpdated {
        nodes_added: usize,
        edges_added: usize,
    },
    
    // Cognitive events
    ReasoningCompleted {
        chain_id: String,
        duration: Duration,
        insights_count: usize,
    },
    InsightGenerated {
        insight_id: String,
        category: String,
        confidence: f32,
    },
    GoalAchieved {
        goal_id: String,
        actions_taken: usize,
    },
    
    // Chat events
    MessageReceived {
        message_id: String,
        content: String,
        source: TabId,
    },
    ResponseGenerated {
        response_id: String,
        model_used: String,
        tokens_used: usize,
    },
    
    // Cross-tab communication
    CrossTabMessage {
        from: TabId,
        to: TabId,
        message: Value,
    },
    StateChanged {
        key: String,
        value: Value,
        source: TabId,
    },
    
    // Orchestration events
    OrchestrationRequested {
        request_id: String,
        config: Value,
    },
    OrchestrationCompleted {
        request_id: String,
        result: Value,
    },
    
    // Code editing events
    CodeEdited {
        file: String,
        changes: Value,
    },
    
    // Agent configuration
    AgentConfig {
        agent_id: String,
        config: Value,
    },
    
    // System events
    TabSwitched {
        from: TabId,
        to: TabId,
    },
    ConfigurationChanged {
        setting: String,
        old_value: Value,
        new_value: Value,
    },
    ErrorOccurred {
        source: TabId,
        error: String,
        severity: ErrorSeverity,
    },
    MetricsUpdated {
        metric_type: String,
        value: f64,
    },
    
    // Story events
    StoryProgressed {
        story_id: String,
        plot_point: String,
        source: TabId,
    },
    StoryArcChanged {
        story_id: String,
        from_arc: String,
        to_arc: String,
        source: TabId,
    },
    StoryTaskCreated {
        story_id: String,
        task_id: String,
        task_description: String,
        source: TabId,
    },
    StoryNarrativeUpdated {
        story_id: String,
        narrative: String,
        confidence: f32,
        source: TabId,
    },
    StoryTodoMapped {
        story_id: String,
        todo_id: String,
        mapping_type: String,
        source: TabId,
    },
    
    // Storage events
    StorageUnlocked,
    StorageLocked,
    ApiKeyStored {
        provider: String,
        source: TabId,
    },
    DatabaseConfigSaved {
        backend: String,
        source: TabId,
    },
    ChatHistoryUpdated {
        conversation_id: String,
        message_count: usize,
    },
    
    // Custom events for flexible communication
    CustomEvent {
        name: String,
        data: Value,
        source: TabId,
        target: Option<TabId>,
    },
}

/// Error severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Benchmark metrics for models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkMetrics {
    pub latency_ms: f64,
    pub tokens_per_second: f64,
    pub accuracy_score: f64,
    pub cost_per_token: f64,
}

/// Event with priority and metadata
#[derive(Debug, Clone)]
pub struct PriorityEvent {
    pub event: SystemEvent,
    pub priority: EventPriority,
    pub timestamp: Instant,
    pub event_id: String,
}

impl PartialEq for PriorityEvent {
    fn eq(&self, other: &Self) -> bool {
        self.event_id == other.event_id
    }
}

impl Eq for PriorityEvent {}

impl PartialOrd for PriorityEvent {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PriorityEvent {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Higher priority first, then earlier timestamp
        match self.priority.cmp(&other.priority) {
            std::cmp::Ordering::Equal => other.timestamp.cmp(&self.timestamp),
            other => other,
        }
    }
}

/// Subscription handler for events
pub type EventHandler = Arc<dyn Fn(SystemEvent) -> Result<()> + Send + Sync>;

/// Subscriber information
pub struct Subscriber {
    pub id: String,
    pub tab_id: TabId,
    pub handler: EventHandler,
    pub filter: Option<EventFilter>,
}

/// Event filter for selective subscription
#[derive(Clone)]
pub enum EventFilter {
    ByTab(TabId),
    ByEventType(String),
    Custom(Arc<dyn Fn(&SystemEvent) -> bool + Send + Sync>),
}

impl std::fmt::Debug for EventFilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EventFilter::ByTab(tab) => write!(f, "ByTab({:?})", tab),
            EventFilter::ByEventType(event_type) => write!(f, "ByEventType({})", event_type),
            EventFilter::Custom(_) => write!(f, "Custom(<function>)"),
        }
    }
}


/// Main event bus implementation (lock-free)
pub struct EventBus {
    /// Subscribers organized by event type (lock-free)
    subscribers: Arc<DashMap<String, Vec<Subscriber>>>,
    
    /// Global subscribers that receive all events (lock-free)
    global_subscribers: Arc<DashMap<usize, Subscriber>>,
    
    /// Lock-free event queue
    event_queue: Arc<LockFreeEventQueue>,
    
    /// Event router for topic-based routing
    event_router: Arc<EventRouter>,
    
    /// Event history for replay and debugging (lock-free)
    event_history: Arc<IndexedRingBuffer<EventHistoryEntry>>,
    
    /// Event processing channel (lock-free)
    event_tx: Sender<PriorityEvent>,
    event_rx: Receiver<PriorityEvent>,
    
    /// Statistics (atomic)
    events_published: Arc<AtomicUsize>,
    events_processed: Arc<AtomicUsize>,
    events_failed: Arc<AtomicUsize>,
    subscribers_count: Arc<AtomicUsize>,
}

/// Event bus statistics
#[derive(Debug, Default, Clone)]
pub struct EventBusStats {
    pub events_published: usize,
    pub events_processed: usize,
    pub events_failed: usize,
    pub average_latency_ms: f64,
    pub subscribers_count: usize,
}

impl EventBus {
    /// Create a new event bus
    pub fn new(history_capacity: usize) -> Self {
        let (tx, rx) = unbounded();
        
        Self {
            subscribers: Arc::new(DashMap::new()),
            global_subscribers: Arc::new(DashMap::new()),
            event_queue: Arc::new(LockFreeEventQueue::new(10000)),
            event_router: Arc::new(EventRouter::new(10000)),
            event_history: Arc::new(IndexedRingBuffer::new(history_capacity)),
            event_tx: tx,
            event_rx: rx,
            events_published: Arc::new(AtomicUsize::new(0)),
            events_processed: Arc::new(AtomicUsize::new(0)),
            events_failed: Arc::new(AtomicUsize::new(0)),
            subscribers_count: Arc::new(AtomicUsize::new(0)),
        }
    }
    
    /// Subscribe to specific event types
    pub async fn subscribe(
        &self,
        event_type: String,
        subscriber: Subscriber,
    ) -> Result<String> {
        let subscriber_id = subscriber.id.clone();
        
        self.subscribers
            .entry(event_type.clone())
            .or_insert_with(Vec::new)
            .push(subscriber);
        
        self.subscribers_count.fetch_add(1, Ordering::Relaxed);
        
        info!("Subscriber {} registered for event type: {}", subscriber_id, event_type);
        Ok(subscriber_id)
    }
    
    /// Subscribe to all events
    pub async fn subscribe_global(&self, subscriber: Subscriber) -> Result<String> {
        let subscriber_id = subscriber.id.clone();
        let subscriber_key = self.subscribers_count.fetch_add(1, Ordering::Relaxed);
        self.global_subscribers.insert(subscriber_key, subscriber);
        
        info!("Global subscriber {} registered", subscriber_id);
        Ok(subscriber_id)
    }
    
    /// Unsubscribe from events
    pub async fn unsubscribe(&self, subscriber_id: &str) -> Result<()> {
        // Remove from specific subscriptions
        for mut entry in self.subscribers.iter_mut() {
            entry.value_mut().retain(|s| s.id != subscriber_id);
        }
        
        // Remove from global subscriptions
        self.global_subscribers.retain(|_, subscriber| subscriber.id != subscriber_id);
        
        let current = self.subscribers_count.load(Ordering::Relaxed);
        if current > 0 {
            self.subscribers_count.store(current - 1, Ordering::Relaxed);
        }
        
        info!("Subscriber {} unregistered", subscriber_id);
        Ok(())
    }
    
    /// Publish an event with normal priority
    pub async fn publish(&self, event: SystemEvent) -> Result<()> {
        self.publish_with_priority(event, EventPriority::Normal).await
    }
    
    /// Publish an event with specific priority
    pub async fn publish_with_priority(
        &self,
        event: SystemEvent,
        priority: EventPriority,
    ) -> Result<()> {
        let event_id = uuid::Uuid::new_v4().to_string();
        let priority_event = PriorityEvent {
            event: event.clone(),
            priority,
            timestamp: Instant::now(),
            event_id: event_id.clone(),
        };
        
        // Add to history
        let history_entry = EventHistoryEntry {
            event: event.clone(),
            timestamp: Instant::now(),
        };
        self.event_history.push(history_entry);
        
        // Send through channel for processing
        if let Err(e) = self.event_tx.send(priority_event) {
            error!("Failed to send event {} through channel: {}", event_id, e);
            return Err(anyhow::anyhow!("Event channel send failed: {}", e));
        }
        
        // Update stats
        self.events_published.fetch_add(1, Ordering::Relaxed);
        
        debug!("Event {} published with priority {:?}", event_id, priority);
        Ok(())
    }
    
    /// Process events from the queue
    pub async fn process_events(&self) -> Result<()> {
        // Process events from the receiver
        
        while let Ok(priority_event) = self.event_rx.recv() {
            let start = Instant::now();
            let event_type = self.get_event_type(&priority_event.event);
            
            // Process specific subscribers (DashMap doesn't need .read())
            if let Some(subs) = self.subscribers.get(&event_type) {
                for subscriber in subs.iter() {
                    // Apply filter if present
                    if let Some(filter) = &subscriber.filter {
                        if !self.apply_filter(filter, &priority_event.event) {
                            continue;
                        }
                    }
                    
                    // Call handler
                    if let Err(e) = (subscriber.handler)(priority_event.event.clone()) {
                        error!("Error in subscriber {}: {:?}", subscriber.id, e);
                        self.events_failed.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
            
            // Process global subscribers
            for subscriber_entry in self.global_subscribers.iter() {
                let subscriber = subscriber_entry.value();
                if let Some(filter) = &subscriber.filter {
                    if !self.apply_filter(filter, &priority_event.event) {
                        continue;
                    }
                }
                
                if let Err(e) = (subscriber.handler)(priority_event.event.clone()) {
                    error!("Error in global subscriber {}: {:?}", subscriber.id, e);
                    self.events_failed.fetch_add(1, Ordering::Relaxed);
                }
            }
            
            // Update stats
            self.events_processed.fetch_add(1, Ordering::Relaxed);
            let _latency = start.elapsed().as_millis() as f64;
            // TODO: Update average latency tracking with atomic operations
        }
        
        Ok(())
    }
    
    /// Get event type as string for routing
    fn get_event_type(&self, event: &SystemEvent) -> String {
        match event {
            SystemEvent::ToolConfigured { .. } => "ToolConfigured".to_string(),
            SystemEvent::ToolExecuted { .. } => "ToolExecuted".to_string(),
            SystemEvent::ToolStatusChanged { .. } => "ToolStatusChanged".to_string(),
            SystemEvent::ModelDiscovered { .. } => "ModelDiscovered".to_string(),
            SystemEvent::ModelSelected { .. } => "ModelSelected".to_string(),
            SystemEvent::ModelBenchmarked { .. } => "ModelBenchmarked".to_string(),
            SystemEvent::AgentCreated { .. } => "AgentCreated".to_string(),
            SystemEvent::AgentTaskAssigned { .. } => "AgentTaskAssigned".to_string(),
            SystemEvent::AgentStatusChanged { .. } => "AgentStatusChanged".to_string(),
            SystemEvent::MemoryStored { .. } => "MemoryStored".to_string(),
            SystemEvent::ContextRetrieved { .. } => "ContextRetrieved".to_string(),
            SystemEvent::KnowledgeGraphUpdated { .. } => "KnowledgeGraphUpdated".to_string(),
            SystemEvent::ReasoningCompleted { .. } => "ReasoningCompleted".to_string(),
            SystemEvent::InsightGenerated { .. } => "InsightGenerated".to_string(),
            SystemEvent::GoalAchieved { .. } => "GoalAchieved".to_string(),
            SystemEvent::MessageReceived { .. } => "MessageReceived".to_string(),
            SystemEvent::ResponseGenerated { .. } => "ResponseGenerated".to_string(),
            SystemEvent::CrossTabMessage { .. } => "CrossTabMessage".to_string(),
            SystemEvent::StateChanged { .. } => "StateChanged".to_string(),
            SystemEvent::OrchestrationRequested { .. } => "OrchestrationRequested".to_string(),
            SystemEvent::OrchestrationCompleted { .. } => "OrchestrationCompleted".to_string(),
            SystemEvent::CodeEdited { .. } => "CodeEdited".to_string(),
            SystemEvent::AgentConfig { .. } => "AgentConfig".to_string(),
            SystemEvent::TabSwitched { .. } => "TabSwitched".to_string(),
            SystemEvent::ConfigurationChanged { .. } => "ConfigurationChanged".to_string(),
            SystemEvent::ErrorOccurred { .. } => "ErrorOccurred".to_string(),
            SystemEvent::MetricsUpdated { .. } => "MetricsUpdated".to_string(),
            SystemEvent::StoryProgressed { .. } => "StoryProgressed".to_string(),
            SystemEvent::StoryArcChanged { .. } => "StoryArcChanged".to_string(),
            SystemEvent::StoryTaskCreated { .. } => "StoryTaskCreated".to_string(),
            SystemEvent::StoryNarrativeUpdated { .. } => "StoryNarrativeUpdated".to_string(),
            SystemEvent::StoryTodoMapped { .. } => "StoryTodoMapped".to_string(),
            SystemEvent::CustomEvent { name, .. } => name.clone(),
            SystemEvent::StorageUnlocked => "StorageUnlocked".to_string(),
            SystemEvent::StorageLocked => "StorageLocked".to_string(),
            SystemEvent::ApiKeyStored { .. } => "ApiKeyStored".to_string(),
            SystemEvent::DatabaseConfigSaved { .. } => "DatabaseConfigSaved".to_string(),
            SystemEvent::ChatHistoryUpdated { .. } => "ChatHistoryUpdated".to_string(),
        }
    }
    
    /// Apply event filter
    fn apply_filter(&self, filter: &EventFilter, event: &SystemEvent) -> bool {
        match filter {
            EventFilter::ByTab(tab_id) => {
                match event {
                    SystemEvent::ToolConfigured { source, .. } |
                    SystemEvent::ToolExecuted { source, .. } |
                    SystemEvent::ModelDiscovered { source, .. } |
                    SystemEvent::ModelSelected { source, .. } |
                    SystemEvent::MemoryStored { source, .. } |
                    SystemEvent::ContextRetrieved { source, .. } |
                    SystemEvent::MessageReceived { source, .. } |
                    SystemEvent::ErrorOccurred { source, .. } |
                    SystemEvent::StoryProgressed { source, .. } |
                    SystemEvent::StoryArcChanged { source, .. } |
                    SystemEvent::StoryTaskCreated { source, .. } |
                    SystemEvent::StoryNarrativeUpdated { source, .. } |
                    SystemEvent::StoryTodoMapped { source, .. } |
                    SystemEvent::ApiKeyStored { source, .. } |
                    SystemEvent::DatabaseConfigSaved { source, .. } |
                    SystemEvent::CustomEvent { source, .. } => source == tab_id,
                    _ => true,
                }
            }
            EventFilter::ByEventType(event_type) => {
                &self.get_event_type(event) == event_type
            }
            EventFilter::Custom(filter_fn) => filter_fn(event),
        }
    }
    
    /// Get event history
    pub async fn get_history(&self) -> Vec<SystemEvent> {
        let history = self.event_history.get_all();
        history.into_iter().map(|entry| entry.event).collect()
    }
    
    /// Get statistics
    pub async fn get_stats(&self) -> EventBusStats {
        EventBusStats {
            events_published: self.events_published.load(Ordering::Relaxed),
            events_processed: self.events_processed.load(Ordering::Relaxed),
            events_failed: self.events_failed.load(Ordering::Relaxed),
            average_latency_ms: 0.0, // TODO: Track this separately if needed
            subscribers_count: self.subscribers_count.load(Ordering::Relaxed),
        }
    }
    
    /// Subscribe to all events with a channel
    pub async fn subscribe_all(&self) -> mpsc::UnboundedReceiver<SystemEvent> {
        let (tx, rx) = mpsc::unbounded_channel();
        
        let subscriber = Subscriber {
            id: uuid::Uuid::new_v4().to_string(),
            tab_id: TabId::System,
            handler: create_handler(move |event| {
                let _ = tx.send(event);
                Ok(())
            }),
            filter: None,
        };
        
        self.subscribe_global(subscriber).await.unwrap();
        rx
    }
    
    /// Clear event history
    pub async fn clear_history(&self) {
        self.event_history.clear();
    }
    
    /// Broadcast an event to all subscribers (high priority)
    pub async fn broadcast(&self, event: SystemEvent) -> Result<()> {
        self.publish_with_priority(event, EventPriority::High).await
    }
    
    /// Broadcast a critical system event  
    pub async fn broadcast_critical(&self, event: SystemEvent) -> Result<()> {
        self.publish_with_priority(event, EventPriority::Critical).await
    }
    
    /// Emit an event (alias for publish for compatibility)
    pub async fn emit(&self, event: SystemEvent) -> Result<()> {
        self.publish(event).await
    }
    
    /// Start the event processing loop
    pub fn start_processing(self: Arc<Self>) {
        tokio::spawn(async move {
            if let Err(e) = self.process_events().await {
                error!("Event processing error: {:?}", e);
            }
        });
    }
}

/// Helper function to create a standard event handler
pub fn create_handler<F>(handler: F) -> EventHandler
where
    F: Fn(SystemEvent) -> Result<()> + Send + Sync + 'static,
{
    Arc::new(handler)
}

#[cfg(test)]
mod tests {
    use super::*;
    use parking_lot::RwLock;
    
    #[tokio::test]
    async fn test_event_bus_basic() {
        let bus = Arc::new(EventBus::new(100));
        
        // Create a test subscriber
        let received = Arc::new(RwLock::new(Vec::new()));
        let received_clone = received.clone();
        
        let subscriber = Subscriber {
            id: "test_subscriber".to_string(),
            tab_id: TabId::Chat,
            handler: create_handler(move |event| {
                let received = received_clone.clone();
                tokio::spawn(async move {
                    let mut r = received.write().await;
                    r.push(event);
                });
                Ok(())
            }),
            filter: None,
        };
        
        // Subscribe to model events
        bus.subscribe("ModelSelected".to_string(), subscriber).await.unwrap();
        
        // Start processing
        let bus_clone = bus.clone();
        bus_clone.start_processing();
        
        // Publish an event
        bus.publish(SystemEvent::ModelSelected {
            model_id: "gpt-4".to_string(),
            source: TabId::Chat,
        }).await.unwrap();
        
        // Wait for processing
        tokio::time::sleep(Duration::from_millis(100)).await;
        
        // Check if event was received
        let received_events = received.read().await;
        assert_eq!(received_events.len(), 1);
    }
    
    #[tokio::test]
    async fn test_event_priority() {
        let bus = Arc::new(EventBus::new(100));
        
        // Publish events with different priorities
        bus.publish_with_priority(
            SystemEvent::MetricsUpdated {
                metric_type: "low".to_string(),
                value: 1.0,
            },
            EventPriority::Low
        ).await.unwrap();
        
        bus.publish_with_priority(
            SystemEvent::ErrorOccurred {
                source: TabId::Chat,
                error: "critical".to_string(),
                severity: ErrorSeverity::Critical,
            },
            EventPriority::Critical
        ).await.unwrap();
        
        // Critical event should be processed first despite being published second
        let stats = bus.get_stats().await;
        assert_eq!(stats.events_published, 2);
    }
}