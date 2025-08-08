//! Tab Registry for TUI
//! 
//! This module provides tab discovery, registration, and capability management
//! for all tabs in the TUI system.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};
use tracing::{info, debug};
use anyhow::Result;

use crate::tui::event_bus::{EventBus, SystemEvent, TabId};

/// Tab capability types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TabCapability {
    ExecuteTools,
    ManageModels,
    StoreMemory,
    RetrieveContext,
    ProcessReasoning,
    GenerateInsights,
    ConfigureSettings,
    MonitorMetrics,
    ManageAgents,
    OrchestrateModels,
}

/// Tab information and metadata
#[derive(Debug, Clone)]
pub struct TabInfo {
    pub id: TabId,
    pub name: String,
    pub description: String,
    pub capabilities: Vec<TabCapability>,
    pub shortcuts: Vec<String>,
    pub active: bool,
    pub dependencies: Vec<TabId>,
}

/// Tab communication channel
pub struct TabChannel {
    pub id: TabId,
    pub sender: tokio::sync::mpsc::UnboundedSender<TabMessage>,
    pub receiver: Arc<RwLock<tokio::sync::mpsc::UnboundedReceiver<TabMessage>>>,
}

/// Messages between tabs
#[derive(Debug, Clone)]
pub enum TabMessage {
    Request {
        from: TabId,
        request_id: String,
        payload: serde_json::Value,
    },
    Response {
        to: TabId,
        request_id: String,
        payload: serde_json::Value,
    },
    Notification {
        from: TabId,
        message: String,
    },
}

/// Tab registry for managing all tabs
pub struct TabRegistry {
    /// Registered tabs
    tabs: Arc<RwLock<HashMap<TabId, TabInfo>>>,
    
    /// Tab communication channels
    channels: Arc<RwLock<HashMap<TabId, TabChannel>>>,
    
    /// Capability index for quick lookup
    capability_index: Arc<RwLock<HashMap<TabCapability, Vec<TabId>>>>,
    
    /// Event bus for notifications
    event_bus: Arc<EventBus>,
}

impl TabRegistry {
    /// Create a new tab registry
    pub fn new(event_bus: Arc<EventBus>) -> Self {
        let registry = Self {
            tabs: Arc::new(RwLock::new(HashMap::new())),
            channels: Arc::new(RwLock::new(HashMap::new())),
            capability_index: Arc::new(RwLock::new(HashMap::new())),
            event_bus,
        };
        
        // Register default tabs
        tokio::spawn({
            let registry = registry.clone();
            async move {
                registry.register_default_tabs().await;
            }
        });
        
        registry
    }
    
    /// Register all default tabs
    async fn register_default_tabs(&self) {
        // Home tab
        self.register_tab(TabInfo {
            id: TabId::Home,
            name: "Home".to_string(),
            description: "System monitoring and control center".to_string(),
            capabilities: vec![
                TabCapability::MonitorMetrics,
                TabCapability::ConfigureSettings,
            ],
            shortcuts: vec!["1".to_string(), "h".to_string()],
            active: true,
            dependencies: vec![],
        }).await.unwrap();
        
        // Chat tab
        self.register_tab(TabInfo {
            id: TabId::Chat,
            name: "Chat".to_string(),
            description: "AI conversation and orchestration hub".to_string(),
            capabilities: vec![
                TabCapability::ExecuteTools,
                TabCapability::ManageModels,
                TabCapability::ProcessReasoning,
                TabCapability::ManageAgents,
                TabCapability::OrchestrateModels,
            ],
            shortcuts: vec!["2".to_string(), "c".to_string()],
            active: true,
            dependencies: vec![TabId::Utilities, TabId::Memory, TabId::Cognitive],
        }).await.unwrap();
        
        // Utilities tab
        self.register_tab(TabInfo {
            id: TabId::Utilities,
            name: "Utilities".to_string(),
            description: "Tool and utility configuration".to_string(),
            capabilities: vec![
                TabCapability::ExecuteTools,
                TabCapability::ConfigureSettings,
            ],
            shortcuts: vec!["3".to_string(), "u".to_string()],
            active: true,
            dependencies: vec![],
        }).await.unwrap();
        
        // Memory tab
        self.register_tab(TabInfo {
            id: TabId::Memory,
            name: "Memory".to_string(),
            description: "Knowledge and memory management".to_string(),
            capabilities: vec![
                TabCapability::StoreMemory,
                TabCapability::RetrieveContext,
            ],
            shortcuts: vec!["4".to_string(), "m".to_string()],
            active: true,
            dependencies: vec![],
        }).await.unwrap();
        
        // Cognitive tab
        self.register_tab(TabInfo {
            id: TabId::Cognitive,
            name: "Cognitive".to_string(),
            description: "Cognitive processing and reasoning".to_string(),
            capabilities: vec![
                TabCapability::ProcessReasoning,
                TabCapability::GenerateInsights,
            ],
            shortcuts: vec!["5".to_string(), "g".to_string()],
            active: true,
            dependencies: vec![TabId::Memory],
        }).await.unwrap();
        
        // Settings tab
        self.register_tab(TabInfo {
            id: TabId::Settings,
            name: "Settings".to_string(),
            description: "Global configuration and preferences".to_string(),
            capabilities: vec![
                TabCapability::ConfigureSettings,
            ],
            shortcuts: vec!["6".to_string(), "s".to_string()],
            active: true,
            dependencies: vec![],
        }).await.unwrap();
        
        info!("Default tabs registered");
    }
    
    /// Register a new tab
    pub async fn register_tab(&self, tab_info: TabInfo) -> Result<()> {
        let tab_id = tab_info.id.clone();
        
        // Create communication channel
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let channel = TabChannel {
            id: tab_id.clone(),
            sender: tx,
            receiver: Arc::new(RwLock::new(rx)),
        };
        
        // Update capability index
        {
            let mut index = self.capability_index.write().await;
            for capability in &tab_info.capabilities {
                index.entry(capability.clone())
                    .or_insert_with(Vec::new)
                    .push(tab_id.clone());
            }
        }
        
        // Register tab
        {
            let mut tabs = self.tabs.write().await;
            tabs.insert(tab_id.clone(), tab_info.clone());
        }
        
        // Register channel
        {
            let mut channels = self.channels.write().await;
            channels.insert(tab_id.clone(), channel);
        }
        
        // Notify via event bus
        self.event_bus.publish(SystemEvent::TabSwitched {
            from: tab_id.clone(), // Using same ID to indicate registration
            to: tab_id.clone(),
        }).await?;
        
        info!("Tab registered: {:?} - {}", tab_id, tab_info.name);
        Ok(())
    }
    
    /// Unregister a tab
    pub async fn unregister_tab(&self, tab_id: TabId) -> Result<()> {
        // Remove from tabs
        let tab_info = {
            let mut tabs = self.tabs.write().await;
            tabs.remove(&tab_id)
        };
        
        if let Some(info) = tab_info {
            // Update capability index
            let mut index = self.capability_index.write().await;
            for capability in &info.capabilities {
                if let Some(tabs) = index.get_mut(capability) {
                    tabs.retain(|id| *id != tab_id);
                }
            }
            
            // Remove channel
            let mut channels = self.channels.write().await;
            channels.remove(&tab_id);
            
            info!("Tab unregistered: {:?}", tab_id);
        }
        
        Ok(())
    }
    
    /// Get tab information
    pub async fn get_tab(&self, tab_id: TabId) -> Option<TabInfo> {
        let tabs = self.tabs.read().await;
        tabs.get(&tab_id).cloned()
    }
    
    /// Get all registered tabs
    pub async fn get_all_tabs(&self) -> Vec<TabInfo> {
        let tabs = self.tabs.read().await;
        tabs.values().cloned().collect()
    }
    
    /// Get tabs with specific capability
    pub async fn get_tabs_with_capability(&self, capability: TabCapability) -> Vec<TabId> {
        let index = self.capability_index.read().await;
        index.get(&capability).cloned().unwrap_or_default()
    }
    
    /// Check if a tab has a specific capability
    pub async fn has_capability(&self, tab_id: TabId, capability: TabCapability) -> bool {
        if let Some(tab) = self.get_tab(tab_id).await {
            tab.capabilities.contains(&capability)
        } else {
            false
        }
    }
    
    /// Send a message to a tab
    pub async fn send_message(&self, to: TabId, message: TabMessage) -> Result<()> {
        let channels = self.channels.read().await;
        if let Some(channel) = channels.get(&to) {
            channel.sender.send(message)?;
            Ok(())
        } else {
            Err(anyhow::anyhow!("Tab {:?} not found", to))
        }
    }
    
    /// Request a service from a tab with specific capability
    pub async fn request_service(
        &self,
        from: TabId,
        capability: TabCapability,
        request_id: String,
        payload: serde_json::Value,
    ) -> Result<()> {
        // Find tabs with the capability
        let tabs = self.get_tabs_with_capability(capability.clone()).await;
        
        if tabs.is_empty() {
            return Err(anyhow::anyhow!("No tab found with capability {:?}", capability));
        }
        
        // Send to first available tab (could implement load balancing here)
        let target = tabs[0].clone();
        
        let message = TabMessage::Request {
            from: from.clone(),
            request_id,
            payload,
        };
        
        self.send_message(target.clone(), message).await?;
        debug!("Service request sent from {:?} to {:?} for {:?}", from, target, capability);
        
        Ok(())
    }
    
    /// Get tab dependencies
    pub async fn get_dependencies(&self, tab_id: TabId) -> Vec<TabId> {
        if let Some(tab) = self.get_tab(tab_id).await {
            tab.dependencies
        } else {
            Vec::new()
        }
    }
    
    /// Check if all dependencies are active
    pub async fn check_dependencies(&self, tab_id: TabId) -> Result<bool> {
        let dependencies = self.get_dependencies(tab_id).await;
        let tabs = self.tabs.read().await;
        
        for dep in dependencies {
            if let Some(tab) = tabs.get(&dep) {
                if !tab.active {
                    return Ok(false);
                }
            } else {
                return Ok(false);
            }
        }
        
        Ok(true)
    }
    
    /// Activate a tab
    pub async fn activate_tab(&self, tab_id: TabId) -> Result<()> {
        // Check dependencies
        if !self.check_dependencies(tab_id.clone()).await? {
            return Err(anyhow::anyhow!("Tab dependencies not satisfied"));
        }
        
        let mut tabs = self.tabs.write().await;
        if let Some(tab) = tabs.get_mut(&tab_id) {
            tab.active = true;
            info!("Tab activated: {:?}", tab_id);
        }
        
        Ok(())
    }
    
    /// Deactivate a tab
    pub async fn deactivate_tab(&self, tab_id: TabId) -> Result<()> {
        // Check if other tabs depend on this
        let all_tabs = self.get_all_tabs().await;
        for tab in all_tabs {
            if tab.dependencies.contains(&tab_id) && tab.active {
                return Err(anyhow::anyhow!(
                    "Cannot deactivate {:?}: {:?} depends on it",
                    tab_id,
                    tab.id
                ));
            }
        }
        
        let mut tabs = self.tabs.write().await;
        if let Some(tab) = tabs.get_mut(&tab_id) {
            tab.active = false;
            info!("Tab deactivated: {:?}", tab_id);
        }
        
        Ok(())
    }
    
    /// Get tab shortcuts
    pub async fn get_shortcuts(&self) -> HashMap<String, TabId> {
        let tabs = self.tabs.read().await;
        let mut shortcuts = HashMap::new();
        
        for (id, info) in tabs.iter() {
            for shortcut in &info.shortcuts {
                shortcuts.insert(shortcut.clone(), id.clone());
            }
        }
        
        shortcuts
    }
}

// Implement Clone for TabRegistry to use in SharedSystemState
impl Clone for TabRegistry {
    fn clone(&self) -> Self {
        Self {
            tabs: self.tabs.clone(),
            channels: self.channels.clone(),
            capability_index: self.capability_index.clone(),
            event_bus: self.event_bus.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_tab_registration() {
        let event_bus = Arc::new(EventBus::new(100));
        let registry = TabRegistry::new(event_bus);
        
        // Wait for default tabs to register
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        
        // Check that default tabs are registered
        let all_tabs = registry.get_all_tabs().await;
        assert_eq!(all_tabs.len(), 6);
        
        // Check specific tab
        let chat_tab = registry.get_tab(TabId::Chat).await;
        assert!(chat_tab.is_some());
        
        let chat_info = chat_tab.unwrap();
        assert_eq!(chat_info.name, "Chat");
        assert!(chat_info.capabilities.contains(&TabCapability::ExecuteTools));
    }
    
    #[tokio::test]
    async fn test_capability_lookup() {
        let event_bus = Arc::new(EventBus::new(100));
        let registry = TabRegistry::new(event_bus);
        
        // Wait for default tabs to register
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        
        // Find tabs with ExecuteTools capability
        let tabs = registry.get_tabs_with_capability(TabCapability::ExecuteTools).await;
        assert!(tabs.contains(&TabId::Chat));
        assert!(tabs.contains(&TabId::Utilities));
    }
    
    #[tokio::test]
    async fn test_dependencies() {
        let event_bus = Arc::new(EventBus::new(100));
        let registry = TabRegistry::new(event_bus);
        
        // Wait for default tabs to register
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        
        // Check Chat tab dependencies
        let deps = registry.get_dependencies(TabId::Chat).await;
        assert!(deps.contains(&TabId::Utilities));
        assert!(deps.contains(&TabId::Memory));
        assert!(deps.contains(&TabId::Cognitive));
    }
}