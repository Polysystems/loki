//! Search functionality module
//! 
//! Chat message search, filtering, and result management

pub mod filters;
pub mod results;
pub mod engine;

// Re-export commonly used types
pub use filters::ChatSearchFilters;
pub use results::ChatSearchResult;
pub use engine::SearchEngine;