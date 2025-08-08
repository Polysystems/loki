//! Thread management module
//! 
//! Message threading, branching, and thread-based conversations

pub mod manager;
pub mod thread;
pub mod branching;

// Re-export commonly used types
pub use manager::ThreadManager;
pub use thread::MessageThread;
pub use branching::ThreadBranch;