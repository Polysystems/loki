//! Individual message thread

use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};

/// A conversation thread
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageThread {
    pub id: String,
    pub name: String,
    pub messages: Vec<crate::tui::run::AssistantResponseType>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub parent_thread: Option<String>,
    pub branch_point: Option<usize>,
}

impl MessageThread {
    /// Create a new thread
    pub fn new(id: String, name: String) -> Self {
        let now = Utc::now();
        Self {
            id,
            name,
            messages: Vec::new(),
            created_at: now,
            updated_at: now,
            parent_thread: None,
            branch_point: None,
        }
    }
    
    /// Add a message to the thread
    pub fn add_message(&mut self, message: crate::tui::run::AssistantResponseType) {
        self.messages.push(message);
        self.updated_at = Utc::now();
    }
    
    /// Create a branch from this thread
    pub fn branch_at(&self, branch_point: usize, new_name: String) -> Self {
        let mut branch = Self::new(
            uuid::Uuid::new_v4().to_string(),
            new_name,
        );
        
        // Copy messages up to branch point
        branch.messages = self.messages[..=branch_point.min(self.messages.len() - 1)].to_vec();
        branch.parent_thread = Some(self.id.clone());
        branch.branch_point = Some(branch_point);
        
        branch
    }
}