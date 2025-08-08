//! Model management module
//! 
//! This module provides model discovery, cataloging, benchmarking, and management
//! capabilities for the chat system.

pub mod discovery;
pub mod catalog;
pub mod benchmark;

pub use discovery::{ModelDiscoveryEngine, ProviderStatus};
pub use catalog::{ModelCatalog, ModelEntry, ModelCategory, PricingInfo, PerformanceMetrics};
pub use benchmark::{BenchmarkSuite, BenchmarkTest, BenchmarkResult};