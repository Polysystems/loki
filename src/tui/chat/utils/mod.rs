//! Utility modules for the chat system

pub mod timeout;
pub mod telemetry;

// Re-export commonly used items
pub use timeout::{
    with_timeout,
    with_timeout_retry,
    TimeoutConfig,
    AdaptiveTimeout,
    DEFAULT_TIMEOUT,
    QUICK_TIMEOUT,
    LONG_TIMEOUT,
};