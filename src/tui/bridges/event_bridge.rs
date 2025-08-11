//! Event Bridge - Central routing for all cross-tab events
//! 
//! This bridge manages the flow of events between tabs, ensuring proper
//! routing, filtering, and delivery of system events.

use std::collections::HashMap;
use std::sync::Arc;
use std::pin::Pin;
use std::future::Future;
use tokio::sync::{RwLock, mpsc};
use anyhow::Result;

use crate::tui::event_bus::{SystemEvent, TabId, EventBus};

/// Type for async event handlers
pub type EventHandler = Box<
    dyn Fn(SystemEvent) -> Pin<Box<dyn Future<Output = Result<()>> + Send>> + Send + Sync
>;

/// Central event routing bridge
pub struct EventBridge {
    event_bus: Arc<EventBus>,
    
    /// Custom event handlers by event type
    handlers: Arc<RwLock<HashMap<String, Vec<EventHandler>>>>,
    
    /// Event routing rules
    routing_rules: Arc<RwLock<Vec<RoutingRule>>>,
    
    /// Event statistics
    stats: Arc<RwLock<EventStats>>,
    
    /// Control channel for routing
    control_tx: mpsc::UnboundedSender<RoutingControl>,
}

/// Routing rule for events
#[derive(Debug, Clone)]
pub struct RoutingRule {
    pub event_type: String,
    pub source_tab: Option<TabId>,
    pub target_tabs: Vec<TabId>,
    pub enabled: bool,
}

/// Routing control messages
#[derive(Debug)]
enum RoutingControl {
    Start,
    Stop,
    UpdateRules(Vec<RoutingRule>),
}

/// Event statistics
#[derive(Debug, Default)]
pub struct EventStats {
    pub events_routed: usize,
    pub events_by_type: HashMap<String, usize>,
    pub events_by_source: HashMap<TabId, usize>,
    pub events_by_target: HashMap<TabId, usize>,
    pub routing_errors: usize,
}

impl EventBridge {
    /// Create a new event bridge
    pub fn new(event_bus: Arc<EventBus>) -> Self {
        let (control_tx, _control_rx) = mpsc::unbounded_channel();
        
        Self {
            event_bus,
            handlers: Arc::new(RwLock::new(HashMap::new())),
            routing_rules: Arc::new(RwLock::new(Self::default_routing_rules())),
            stats: Arc::new(RwLock::new(EventStats::default())),
            control_tx,
        }
    }
    
    /// Get default routing rules
    fn default_routing_rules() -> Vec<RoutingRule> {
        vec![
            // Tool events route to Home and Chat
            RoutingRule {
                event_type: "ToolExecuted".to_string(),
                source_tab: None,
                target_tabs: vec![TabId::Home, TabId::Chat],
                enabled: true,
            },
            // Model events route to Chat and Settings
            RoutingRule {
                event_type: "ModelSelected".to_string(),
                source_tab: None,
                target_tabs: vec![TabId::Chat, TabId::Settings],
                enabled: true,
            },
            // Memory events route to Chat
            RoutingRule {
                event_type: "MemoryStored".to_string(),
                source_tab: None,
                target_tabs: vec![TabId::Chat],
                enabled: true,
            },
            // Cognitive events route to Chat and Home
            RoutingRule {
                event_type: "ReasoningCompleted".to_string(),
                source_tab: None,
                target_tabs: vec![TabId::Chat, TabId::Home],
                enabled: true,
            },
            // Story events route to all relevant tabs
            RoutingRule {
                event_type: "StoryProgressed".to_string(),
                source_tab: None,
                target_tabs: vec![TabId::Chat, TabId::Home, TabId::Cognitive],
                enabled: true,
            },
            RoutingRule {
                event_type: "StoryArcChanged".to_string(),
                source_tab: None,
                target_tabs: vec![TabId::Chat, TabId::Home, TabId::Cognitive],
                enabled: true,
            },
            RoutingRule {
                event_type: "StoryTaskCreated".to_string(),
                source_tab: None,
                target_tabs: vec![TabId::Chat, TabId::Utilities],
                enabled: true,
            },
            // Error events route to all tabs
            RoutingRule {
                event_type: "ErrorOccurred".to_string(),
                source_tab: None,
                target_tabs: vec![
                    TabId::Home,
                    TabId::Chat,
                    TabId::Utilities,
                    TabId::Memory,
                    TabId::Cognitive,
                    TabId::Settings,
                ],
                enabled: true,
            },
        ]
    }
    
    /// Start event routing
    pub async fn start_routing(&self) -> Result<()> {
        tracing::info!("Starting event routing");
        
        // For now, just log that routing has started
        // The actual routing is handled through the event bus subscriptions
        tracing::debug!("Event routing started - handlers are active");
        
        Ok(())
    }
    
    /// Stop event routing
    pub async fn stop_routing(&self) -> Result<()> {
        tracing::info!("Stopping event routing");
        self.control_tx.send(RoutingControl::Stop)?;
        Ok(())
    }
    
    /// Subscribe a handler for a specific event type
    pub async fn subscribe_handler<F>(&self, event_type: &str, handler: F) -> Result<()>
    where
        F: Fn(SystemEvent) -> Pin<Box<dyn Future<Output = Result<()>> + Send>> + Send + Sync + 'static,
    {
        let mut handlers = self.handlers.write().await;
        handlers.entry(event_type.to_string())
            .or_insert_with(Vec::new)
            .push(Box::new(handler));
        
        tracing::debug!("Handler subscribed for event type: {}", event_type);
        Ok(())
    }
    
    /// Publish an event to the event bus
    pub async fn publish(&self, event: SystemEvent) -> Result<()> {
        // Update statistics
        let mut stats = self.stats.write().await;
        stats.events_routed += 1;
        
        let event_type = self.get_event_type(&event);
        *stats.events_by_type.entry(event_type.clone()).or_insert(0) += 1;
        
        if let Some(source) = self.get_event_source(&event) {
            *stats.events_by_source.entry(source).or_insert(0) += 1;
        }
        
        // Publish to event bus
        self.event_bus.publish(event.clone()).await?;
        
        // Call registered handlers
        let handlers = self.handlers.read().await;
        if let Some(event_handlers) = handlers.get(&event_type) {
            for handler in event_handlers {
                if let Err(e) = handler(event.clone()).await {
                    tracing::error!("Handler error for {}: {}", event_type, e);
                    stats.routing_errors += 1;
                }
            }
        }
        
        Ok(())
    }
    
    /// Route an event based on routing rules
    pub async fn route_event(&self, event: SystemEvent) -> Result<()> {
        let event_type = self.get_event_type(&event);
        let source_tab = self.get_event_source(&event);
        
        let rules = self.routing_rules.read().await;
        
        for rule in rules.iter() {
            if !rule.enabled {
                continue;
            }
            
            if rule.event_type != event_type {
                continue;
            }
            
            if let Some(ref rule_source) = rule.source_tab {
                if Some(rule_source) != source_tab.as_ref() {
                    continue;
                }
            }
            
            // Route to target tabs
            for target in &rule.target_tabs {
                tracing::debug!("Routing {} to {:?}", event_type, target);
                
                // Update stats
                let mut stats = self.stats.write().await;
                *stats.events_by_target.entry(target.clone()).or_insert(0) += 1;
                
                // In a real implementation, this would send to tab-specific handlers
            }
        }
        
        Ok(())
    }
    
    /// Get event type as string
    fn get_event_type(&self, event: &SystemEvent) -> String {
        match event {
            SystemEvent::ToolConfigured { .. } => "ToolConfigured",
            SystemEvent::ToolExecuted { .. } => "ToolExecuted",
            SystemEvent::ToolStatusChanged { .. } => "ToolStatusChanged",
            SystemEvent::ModelDiscovered { .. } => "ModelDiscovered",
            SystemEvent::ModelSelected { .. } => "ModelSelected",
            SystemEvent::ModelBenchmarked { .. } => "ModelBenchmarked",
            SystemEvent::AgentCreated { .. } => "AgentCreated",
            SystemEvent::AgentTaskAssigned { .. } => "AgentTaskAssigned",
            SystemEvent::AgentStatusChanged { .. } => "AgentStatusChanged",
            SystemEvent::MemoryStored { .. } => "MemoryStored",
            SystemEvent::ContextRetrieved { .. } => "ContextRetrieved",
            SystemEvent::ReasoningCompleted { .. } => "ReasoningCompleted",
            SystemEvent::GoalAchieved { .. } => "GoalAchieved",
            SystemEvent::InsightGenerated { .. } => "InsightGenerated",
            SystemEvent::MessageReceived { .. } => "MessageReceived",
            SystemEvent::ResponseGenerated { .. } => "ResponseGenerated",
            SystemEvent::CrossTabMessage { .. } => "CrossTabMessage",
            SystemEvent::StateChanged { .. } => "StateChanged",
            SystemEvent::TabSwitched { .. } => "TabSwitched",
            SystemEvent::ErrorOccurred { .. } => "ErrorOccurred",
            SystemEvent::MetricsUpdated { .. } => "MetricsUpdated",
            SystemEvent::OrchestrationRequested { .. } => "OrchestrationRequested",
            SystemEvent::StoryProgressed { .. } => "StoryProgressed",
            SystemEvent::StoryArcChanged { .. } => "StoryArcChanged",
            SystemEvent::StoryTaskCreated { .. } => "StoryTaskCreated",
            SystemEvent::StoryNarrativeUpdated { .. } => "StoryNarrativeUpdated",
            SystemEvent::StoryTodoMapped { .. } => "StoryTodoMapped",
            SystemEvent::OrchestrationCompleted { .. } => "OrchestrationCompleted",
            SystemEvent::CodeEdited { .. } => "CodeEdited",
            SystemEvent::AgentConfig { .. } => "AgentConfig",
            SystemEvent::ConfigurationChanged { .. } => "ConfigurationChanged",
            SystemEvent::KnowledgeGraphUpdated { .. } => "KnowledgeGraphUpdated",
            SystemEvent::CustomEvent { name, .. } => name.as_str(),
            SystemEvent::StorageUnlocked => "StorageUnlocked",
            SystemEvent::StorageLocked => "StorageLocked",
            SystemEvent::ApiKeyStored { .. } => "ApiKeyStored",
            SystemEvent::DatabaseConfigSaved { .. } => "DatabaseConfigSaved",
            SystemEvent::ChatHistoryUpdated { .. } => "ChatHistoryUpdated",
        }.to_string()
    }
    
    /// Get event source tab
    fn get_event_source(&self, event: &SystemEvent) -> Option<TabId> {
        match event {
            SystemEvent::ToolConfigured { source, .. } |
            SystemEvent::ToolExecuted { source, .. } |
            SystemEvent::ModelDiscovered { source, .. } |
            SystemEvent::ModelSelected { source, .. } |
            SystemEvent::MemoryStored { source, .. } |
            SystemEvent::ContextRetrieved { source, .. } |
            SystemEvent::MessageReceived { source, .. } |
            SystemEvent::StateChanged { source, .. } |
            SystemEvent::ErrorOccurred { source, .. } |
            SystemEvent::CustomEvent { source, .. } => Some(source.clone()),
            
            SystemEvent::TabSwitched { from, .. } => Some(from.clone()),
            SystemEvent::CrossTabMessage { from, .. } => Some(from.clone()),
            
            _ => None,
        }
    }
    
    /// Add a new routing rule
    pub async fn add_routing_rule(&self, rule: RoutingRule) -> Result<()> {
        let mut rules = self.routing_rules.write().await;
        rules.push(rule);
        
        tracing::info!("New routing rule added");
        Ok(())
    }
    
    /// Update routing rules
    pub async fn update_routing_rules(&self, rules: Vec<RoutingRule>) -> Result<()> {
        self.control_tx.send(RoutingControl::UpdateRules(rules))?;
        Ok(())
    }
    
    /// Get event statistics
    pub async fn get_stats(&self) -> EventStats {
        let stats = self.stats.read().await;
        EventStats {
            events_routed: stats.events_routed,
            events_by_type: stats.events_by_type.clone(),
            events_by_source: stats.events_by_source.clone(),
            events_by_target: stats.events_by_target.clone(),
            routing_errors: stats.routing_errors,
        }
    }
    
    /// Reset statistics
    pub async fn reset_stats(&self) {
        let mut stats = self.stats.write().await;
        *stats = EventStats::default();
        
        tracing::debug!("Event statistics reset");
    }
    
    /// Subscribe to a specific event type with a callback
    /// This is a convenience method that wraps subscribe_handler
    pub async fn on_event<F>(&self, event_type: &str, handler: F) -> Result<()>
    where
        F: Fn(SystemEvent) -> Pin<Box<dyn Future<Output = Result<()>> + Send>> + Send + Sync + 'static,
    {
        self.subscribe_handler(event_type, handler).await
    }
    
    /// Emit an event to the event bus
    /// This is a convenience method that wraps publish
    pub async fn emit(&self, event: SystemEvent) -> Result<()> {
        self.publish(event).await
    }
}