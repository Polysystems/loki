//\! Chat tab modules

pub mod state;
pub mod models;
pub mod agents;
pub mod context;
pub mod search;
pub mod threads;
pub mod rendering;

// Re-export commonly used items
pub use state::{ChatState, ChatManager};
pub use models::{ModelManager, ActiveModel};
pub use agents::{AgentManager, OrchestrationManager};
pub use context::{SmartContextManager, TokenEstimator};
pub use search::{ChatSearchFilters, ChatSearchResult};
pub use threads::{ThreadManager, MessageThread};
