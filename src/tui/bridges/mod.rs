//! Bridge modules for cross-tab communication and integration
//! 
//! This module provides the infrastructure for seamless communication
//! between different tabs in the TUI, enabling features to work together
//! as a unified system.

pub mod tool_bridge;
pub mod memory_bridge;
pub mod cognitive_bridge;
pub mod event_bridge;
pub mod context_sync;
pub mod editor_bridge;
pub mod storage_bridge;

// Re-export key types
pub use tool_bridge::ToolBridge;
pub use memory_bridge::MemoryBridge;
pub use cognitive_bridge::CognitiveBridge;
pub use event_bridge::EventBridge;
pub use context_sync::ContextSync;
pub use editor_bridge::EditorBridge;
pub use storage_bridge::{StorageBridge, StorageStatus};

use std::sync::Arc;
use tokio::sync::RwLock;
use anyhow::Result;

/// Unified bridge system that coordinates all cross-tab communication
pub struct UnifiedBridge {
    pub tool_bridge: Arc<ToolBridge>,
    pub memory_bridge: Arc<MemoryBridge>,
    pub cognitive_bridge: Arc<CognitiveBridge>,
    pub event_bridge: Arc<EventBridge>,
    pub context_sync: Arc<RwLock<ContextSync>>,
    pub editor_bridge: Arc<EditorBridge>,
    pub storage_bridge: Arc<StorageBridge>,
}

impl UnifiedBridge {
    /// Create a new unified bridge system
    pub fn new(event_bus: Arc<crate::tui::event_bus::EventBus>) -> Self {
        let event_bridge = Arc::new(EventBridge::new(event_bus.clone()));
        let context_sync = Arc::new(RwLock::new(ContextSync::new()));
        
        Self {
            tool_bridge: Arc::new(ToolBridge::new(event_bridge.clone())),
            memory_bridge: Arc::new(MemoryBridge::new(event_bridge.clone(), context_sync.clone())),
            cognitive_bridge: Arc::new(CognitiveBridge::new(event_bridge.clone())),
            editor_bridge: Arc::new(EditorBridge::new(event_bridge.clone())),
            storage_bridge: Arc::new(StorageBridge::new(event_bridge.clone())),
            event_bridge,
            context_sync,
        }
    }
    
    /// Initialize all bridges
    pub async fn initialize(&self) -> Result<()> {
        tracing::info!("Initializing unified bridge system");
        
        // Initialize individual bridges
        self.tool_bridge.initialize().await?;
        self.memory_bridge.initialize().await?;
        self.cognitive_bridge.initialize().await?;
        self.editor_bridge.initialize().await?;
        
        // Start event routing
        self.event_bridge.start_routing().await?;
        
        tracing::info!("✅ Unified bridge system initialized");
        Ok(())
    }
    
    /// Shutdown all bridges
    pub async fn shutdown(&self) -> Result<()> {
        tracing::info!("Shutting down unified bridge system");
        
        self.event_bridge.stop_routing().await?;
        
        tracing::info!("Bridge system shutdown complete");
        Ok(())
    }
}