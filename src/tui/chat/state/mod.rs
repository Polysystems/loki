//! Chat state management module
//! 
//! This module contains all state-related functionality extracted from chat.rs

pub mod chat_state;
pub mod session;
pub mod persistence;
pub mod history;
pub mod settings;
pub mod navigation;
pub mod state_manager;

// Re-export commonly used types
pub use chat_state::{ChatState, ChatInitState};
pub use session::SessionManager;
pub use persistence::StatePersistence;
pub use history::HistoryManager;
pub use settings::ChatSettings;
pub use navigation::{NavigationHandler, NavigationExt};
pub use state_manager::{StateManager, StateManagementExt};