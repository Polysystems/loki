//! Context Synchronization - Manages context sharing across tabs
//! 
//! This module ensures that context is properly synchronized and available
//! to all tabs that need it, with priority and relevance management.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde_json::Value;
use anyhow::Result;

use crate::tui::event_bus::TabId;
use crate::tui::bridges::memory_bridge::ContextItem;

/// Context synchronization manager
pub struct ContextSync {
    /// Context storage per tab
    tab_contexts: Arc<RwLock<HashMap<TabId, TabContext>>>,
    
    /// Global shared context
    global_context: Arc<RwLock<GlobalContext>>,
    
    /// Context priority rules
    priority_rules: Arc<RwLock<Vec<PriorityRule>>>,
    
    /// Context window limits
    window_limits: ContextWindowLimits,
}

/// Context for a specific tab
#[derive(Debug, Clone)]
pub struct TabContext {
    pub tab_id: TabId,
    pub items: Vec<ContextItem>,
    pub last_updated: std::time::Instant,
    pub active: bool,
}

/// Global shared context
#[derive(Debug, Clone)]
pub struct GlobalContext {
    pub shared_items: Vec<ContextItem>,
    pub conversation_history: Vec<ConversationTurn>,
    pub system_state: Value,
    pub last_sync: std::time::Instant,
}

/// Conversation turn for history
#[derive(Debug, Clone)]
pub struct ConversationTurn {
    pub id: String,
    pub role: String,
    pub content: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub metadata: Value,
}

/// Priority rule for context items
#[derive(Debug, Clone)]
pub struct PriorityRule {
    pub source: ContextSource,
    pub priority: f32,
    pub max_age_seconds: u64,
}

/// Source of context items
#[derive(Debug, Clone, PartialEq)]
pub enum ContextSource {
    UserInput,
    SystemResponse,
    Memory,
    Cognitive,
    Tool,
}

/// Limits for context windows
#[derive(Debug, Clone)]
pub struct ContextWindowLimits {
    pub max_items_per_tab: usize,
    pub max_global_items: usize,
    pub max_history_turns: usize,
    pub max_total_tokens: usize,
}

impl Default for ContextWindowLimits {
    fn default() -> Self {
        Self {
            max_items_per_tab: 20,
            max_global_items: 50,
            max_history_turns: 10,
            max_total_tokens: 8000,
        }
    }
}

impl ContextSync {
    /// Create a new context sync manager
    pub fn new() -> Self {
        Self {
            tab_contexts: Arc::new(RwLock::new(HashMap::new())),
            global_context: Arc::new(RwLock::new(GlobalContext {
                shared_items: Vec::new(),
                conversation_history: Vec::new(),
                system_state: serde_json::json!({}),
                last_sync: std::time::Instant::now(),
            })),
            priority_rules: Arc::new(RwLock::new(Self::default_priority_rules())),
            window_limits: ContextWindowLimits::default(),
        }
    }
    
    /// Get default priority rules
    fn default_priority_rules() -> Vec<PriorityRule> {
        vec![
            PriorityRule {
                source: ContextSource::UserInput,
                priority: 1.0,
                max_age_seconds: 300, // 5 minutes
            },
            PriorityRule {
                source: ContextSource::SystemResponse,
                priority: 0.9,
                max_age_seconds: 300,
            },
            PriorityRule {
                source: ContextSource::Cognitive,
                priority: 0.8,
                max_age_seconds: 600, // 10 minutes
            },
            PriorityRule {
                source: ContextSource::Memory,
                priority: 0.7,
                max_age_seconds: 1800, // 30 minutes
            },
            PriorityRule {
                source: ContextSource::Tool,
                priority: 0.6,
                max_age_seconds: 600,
            },
        ]
    }
    
    /// Update context for a specific tab
    pub async fn update_context(
        &self,
        tab_id: TabId,
        items: Vec<ContextItem>,
    ) -> Result<()> {
        let mut tab_contexts = self.tab_contexts.write().await;
        
        let tab_context = tab_contexts.entry(tab_id.clone()).or_insert_with(|| TabContext {
            tab_id: tab_id.clone(),
            items: Vec::new(),
            last_updated: std::time::Instant::now(),
            active: true,
        });
        
        // Update items with priority sorting
        tab_context.items = self.prioritize_items(items, self.window_limits.max_items_per_tab);
        tab_context.last_updated = std::time::Instant::now();
        
        // Trigger global context sync
        self.sync_global_context().await?;
        
        tracing::debug!("Context updated for tab {:?} with {} items", tab_id, tab_context.items.len());
        
        Ok(())
    }
    
    /// Get context for a specific tab
    pub async fn get_tab_context(&self, tab_id: TabId) -> Option<TabContext> {
        let tab_contexts = self.tab_contexts.read().await;
        tab_contexts.get(&tab_id).cloned()
    }
    
    /// Get merged context for a tab (tab-specific + global)
    pub async fn get_merged_context(&self, tab_id: TabId) -> Vec<ContextItem> {
        let mut merged = Vec::new();
        
        // Add tab-specific context
        if let Some(tab_context) = self.get_tab_context(tab_id).await {
            merged.extend(tab_context.items);
        }
        
        // Add relevant global context
        let global = self.global_context.read().await;
        for item in &global.shared_items {
            if !merged.iter().any(|i| i.id == item.id) {
                merged.push(item.clone());
            }
        }
        
        // Prioritize and limit
        self.prioritize_items(merged, self.window_limits.max_items_per_tab)
    }
    
    /// Sync global context from all tabs
    async fn sync_global_context(&self) -> Result<()> {
        let tab_contexts = self.tab_contexts.read().await;
        let mut global = self.global_context.write().await;
        
        // Clear old shared items
        global.shared_items.clear();
        
        // Collect high-priority items from all active tabs
        for (_, tab_context) in tab_contexts.iter() {
            if !tab_context.active {
                continue;
            }
            
            for item in &tab_context.items {
                if item.relevance_score > 0.7 { // Only share highly relevant items
                    if !global.shared_items.iter().any(|i| i.id == item.id) {
                        global.shared_items.push(item.clone());
                    }
                }
            }
        }
        
        // Limit global items
        global.shared_items = self.prioritize_items(
            global.shared_items.clone(),
            self.window_limits.max_global_items
        );
        
        global.last_sync = std::time::Instant::now();
        
        tracing::debug!("Global context synced with {} shared items", global.shared_items.len());
        
        Ok(())
    }
    
    /// Add conversation turn to history
    pub async fn add_conversation_turn(
        &self,
        role: String,
        content: String,
        metadata: Value,
    ) -> Result<()> {
        let mut global = self.global_context.write().await;
        
        let turn = ConversationTurn {
            id: uuid::Uuid::new_v4().to_string(),
            role,
            content,
            timestamp: chrono::Utc::now(),
            metadata,
        };
        
        global.conversation_history.push(turn);
        
        // Limit history size
        if global.conversation_history.len() > self.window_limits.max_history_turns {
            let excess = global.conversation_history.len() - self.window_limits.max_history_turns;
            global.conversation_history.drain(0..excess);
        }
        
        Ok(())
    }
    
    /// Get conversation history
    pub async fn get_conversation_history(&self) -> Vec<ConversationTurn> {
        let global = self.global_context.read().await;
        global.conversation_history.clone()
    }
    
    /// Update system state in global context
    pub async fn update_system_state(&self, state: Value) -> Result<()> {
        let mut global = self.global_context.write().await;
        global.system_state = state;
        global.last_sync = std::time::Instant::now();
        
        Ok(())
    }
    
    /// Prioritize and limit context items
    fn prioritize_items(&self, mut items: Vec<ContextItem>, max_items: usize) -> Vec<ContextItem> {
        // Sort by relevance score (highest first)
        items.sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score).unwrap());
        
        // Take only the top items
        items.truncate(max_items);
        
        items
    }
    
    /// Clear context for a tab
    pub async fn clear_tab_context(&self, tab_id: TabId) -> Result<()> {
        let mut tab_contexts = self.tab_contexts.write().await;
        
        if let Some(context) = tab_contexts.get_mut(&tab_id) {
            context.items.clear();
            context.last_updated = std::time::Instant::now();
            context.active = false;
        }
        
        Ok(())
    }
    
    /// Activate a tab's context
    pub async fn activate_tab(&self, tab_id: TabId) -> Result<()> {
        let mut tab_contexts = self.tab_contexts.write().await;
        
        // Deactivate all other tabs
        for (_, context) in tab_contexts.iter_mut() {
            context.active = false;
        }
        
        // Activate the specified tab
        if let Some(context) = tab_contexts.get_mut(&tab_id) {
            context.active = true;
        } else {
            // Create new context if it doesn't exist
            tab_contexts.insert(tab_id.clone(), TabContext {
                tab_id: tab_id.clone(),
                items: Vec::new(),
                last_updated: std::time::Instant::now(),
                active: true,
            });
        }
        
        // Sync global context
        drop(tab_contexts); // Release lock before async call
        self.sync_global_context().await?;
        
        Ok(())
    }
    
    /// Get context statistics
    pub async fn get_stats(&self) -> ContextStats {
        let tab_contexts = self.tab_contexts.read().await;
        let global = self.global_context.read().await;
        
        let total_items: usize = tab_contexts.values()
            .map(|c| c.items.len())
            .sum();
        
        let active_tabs = tab_contexts.values()
            .filter(|c| c.active)
            .count();
        
        ContextStats {
            total_context_items: total_items,
            global_shared_items: global.shared_items.len(),
            conversation_turns: global.conversation_history.len(),
            active_tabs,
            last_sync: global.last_sync,
        }
    }
}

/// Context synchronization statistics
#[derive(Debug, Clone)]
pub struct ContextStats {
    pub total_context_items: usize,
    pub global_shared_items: usize,
    pub conversation_turns: usize,
    pub active_tabs: usize,
    pub last_sync: std::time::Instant,
}