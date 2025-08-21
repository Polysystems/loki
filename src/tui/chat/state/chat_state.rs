//! Core chat state management
//! 
//! Contains the main ChatState struct and related types

use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};

use crate::memory::MemoryId;
use crate::tui::run::AssistantResponseType;

/// Chat initialization state for UI feedback
#[derive(Debug, Clone, PartialEq)]
pub enum ChatInitState {
    Uninitialized,
    InitializingUI,
    InitializingCognitive,
    InitializingMemory,
    Ready,
    Error(String),
    Degraded(String), // Partial functionality available
}

/// Interactive workflow state for multi-step processes like API setup
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum InteractiveWorkflowState {
    None,
    ApiSetup {
        current_provider: Option<String>,
        step: ApiSetupStep,
        collected_keys: HashMap<String, String>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ApiSetupStep {
    SelectProvider,
    EnterKey { provider: String },
    Confirm { provider: String, key: String },
    Complete,
}

/// Enhanced chat state with persistent memory and editing capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatState {
    pub id: String,
    pub name: String,
    pub title: String,
    pub messages: Vec<AssistantResponseType>,
    pub messages_1: Vec<AssistantResponseType>,
    pub active_model: Option<String>,
    pub messages_2: Vec<AssistantResponseType>,
    pub selected_message_index: Option<usize>,
    pub edit_buffer: String,
    pub scroll_offset: usize,
    pub created_at: DateTime<Utc>,
    pub last_activity: DateTime<Utc>,
    pub is_persistent: bool,
    pub memory_associations: Vec<MemoryId>,
    pub session_context: HashMap<String, String>,
    pub is_modified: bool,
    pub interactive_workflow: InteractiveWorkflowState,
    
    // File attachments
    pub file_attachments: Vec<crate::tui::ui::chat::FileAttachment>,
    
    // Additional fields for shared state integration
    pub current_model: Option<String>,
    pub orchestration_enabled: bool,
    pub active_agents: Vec<String>,
    pub tool_executions: Vec<ToolExecution>,
    
    // UI state fields
    pub active_chat: usize,
    pub show_timestamps: bool,
    pub show_context_panel: bool,
}

/// Tool execution record for chat state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolExecution {
    pub tool_id: String,
    pub timestamp: DateTime<Utc>,
    pub success: bool,
    pub result_summary: String,
}

impl ChatState {
    /// Create a new chat state
    pub fn new(id: usize, name: String) -> Self {
        let id_str = id.to_string();
        Self {
            id: id_str.clone(),
            name: name.clone(),
            title: format!("Chat #{}", id),
            messages: Vec::new(),
            messages_1: Vec::new(),
            messages_2: Vec::new(),
            selected_message_index: None,
            edit_buffer: String::new(),
            scroll_offset: 0,
            created_at: Utc::now(),
            last_activity: Utc::now(),
            is_persistent: false,
            memory_associations: Vec::new(),
            session_context: HashMap::new(),
            is_modified: false,
            interactive_workflow: InteractiveWorkflowState::None,
            file_attachments: Vec::new(),
            active_model: None,
            current_model: None,
            orchestration_enabled: false,
            active_agents: Vec::new(),
            tool_executions: Vec::new(),
            active_chat: id,
            show_timestamps: false,
            show_context_panel: false,
        }
    }
    
    /// Add a message to the chat
    pub fn add_message_to_chat(&mut self, message: AssistantResponseType, thread: usize) {
        self.last_activity = Utc::now();
        self.is_modified = true;
        
        match thread {
            0 => self.messages.push(message),
            1 => self.messages_1.push(message),
            2 => self.messages_2.push(message),
            _ => self.messages.push(message),
        }
    }
    
    /// Make this chat persistent
    pub fn make_persistent(&mut self) {
        self.is_persistent = true;
        self.is_modified = true;
    }
    
    /// Clear all messages
    pub fn clear_messages(&mut self) {
        self.messages.clear();
        self.messages_1.clear();
        self.messages_2.clear();
        self.selected_message_index = None;
        self.scroll_offset = 0;
        self.is_modified = true;
        self.last_activity = Utc::now();
    }
    
    /// Get total message count across all threads
    pub fn total_message_count(&self) -> usize {
        self.messages.len() + self.messages_1.len() + self.messages_2.len()
    }
    


    /// Check if chat has unsaved changes
    pub fn has_unsaved_changes(&self) -> bool {
        self.is_modified && self.is_persistent
    }
    
    /// Mark chat as saved
    pub fn mark_as_saved(&mut self) {
        self.is_modified = false;
    }

  
}