//! Stub tracking and management utilities for the Loki codebase
//! 
//! This module provides tools for tracking, categorizing, and managing
//! stub implementations across the codebase.

use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::RwLock;
use chrono::{DateTime, Utc};
use serde::{Serialize, Deserialize};

/// Priority levels for stub implementations
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum StubPriority {
    /// Blocks core functionality or creates security risks
    Critical,
    /// Required for advertised features
    High,
    /// Would improve performance or user experience
    Medium,
    /// Extension points or future enhancements
    Low,
}

/// Categories of stub implementations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StubCategory {
    /// Safety and security critical functions
    Safety,
    /// Core feature implementations
    Feature,
    /// Performance or UX enhancements
    Enhancement,
    /// Intentional extension points
    Extension,
    /// Test-specific implementations
    Test,
}

/// Information about a stub implementation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StubInfo {
    /// Unique identifier for the stub
    pub id: String,
    /// Source file containing the stub
    pub file: String,
    /// Line number where stub is defined
    pub line: u32,
    /// Implementation priority
    pub priority: StubPriority,
    /// Stub category
    pub category: StubCategory,
    /// Human-readable description
    pub description: String,
    /// Dependencies on other stubs
    pub dependencies: Vec<String>,
    /// Date when stub was created
    pub created_at: DateTime<Utc>,
    /// Whether this is an intentional stub
    pub intentional: bool,
    /// Risk assessment notes
    pub risks: Option<String>,
}

/// Runtime tracking of stub usage
#[derive(Debug, Default)]
pub struct StubUsageStats {
    /// Number of times each stub has been called
    pub call_counts: HashMap<String, AtomicUsize>,
    /// First call timestamp for each stub
    pub first_calls: HashMap<String, DateTime<Utc>>,
    /// Last call timestamp for each stub
    pub last_calls: HashMap<String, DateTime<Utc>>,
}

/// Global stub registry
pub static STUB_REGISTRY: Lazy<RwLock<HashMap<String, StubInfo>>> = Lazy::new(|| {
    RwLock::new(HashMap::new())
});

/// Global usage statistics
pub static STUB_USAGE: Lazy<RwLock<StubUsageStats>> = Lazy::new(|| {
    RwLock::new(StubUsageStats::default())
});

/// Register a new stub in the global registry
pub fn register_stub(info: StubInfo) {
    let mut registry = STUB_REGISTRY.write().unwrap();
    registry.insert(info.id.clone(), info);
}

/// Track usage of a stub
pub fn track_stub_usage(stub_id: &str) {
    let mut usage = STUB_USAGE.write().unwrap();
    let now = Utc::now();
    
    // Update call count
    usage.call_counts
        .entry(stub_id.to_string())
        .or_insert_with(|| AtomicUsize::new(0))
        .fetch_add(1, Ordering::Relaxed);
    
    // Update timestamps
    usage.first_calls.entry(stub_id.to_string()).or_insert(now);
    usage.last_calls.insert(stub_id.to_string(), now);
}

/// Generate a report of all stubs
pub fn generate_stub_report() -> StubReport {
    let registry = STUB_REGISTRY.read().unwrap();
    let usage = STUB_USAGE.read().unwrap();
    
    let mut stubs_by_priority = HashMap::new();
    let mut stubs_by_category = HashMap::new();
    let mut critical_stubs = Vec::new();
    let mut frequently_used = Vec::new();
    
    for (id, info) in registry.iter() {
        // Group by priority
        stubs_by_priority
            .entry(info.priority)
            .or_insert_with(Vec::new)
            .push(id.clone());
        
        // Group by category
        stubs_by_category
            .entry(info.category)
            .or_insert_with(Vec::new)
            .push(id.clone());
        
        // Identify critical stubs
        if info.priority == StubPriority::Critical {
            critical_stubs.push(id.clone());
        }
        
        // Find frequently used stubs
        if let Some(counter) = usage.call_counts.get(id) {
            let count = counter.load(Ordering::Relaxed);
            if count > 100 {
                frequently_used.push((id.clone(), count));
            }
        }
    }
    
    frequently_used.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
    
    StubReport {
        total_stubs: registry.len(),
        stubs_by_priority,
        stubs_by_category,
        critical_stubs,
        frequently_used: frequently_used.into_iter().take(10).collect(),
        report_generated: Utc::now(),
    }
}

/// Report structure for stub analysis
#[derive(Debug, Serialize)]
pub struct StubReport {
    pub total_stubs: usize,
    pub stubs_by_priority: HashMap<StubPriority, Vec<String>>,
    pub stubs_by_category: HashMap<StubCategory, Vec<String>>,
    pub critical_stubs: Vec<String>,
    pub frequently_used: Vec<(String, usize)>,
    pub report_generated: DateTime<Utc>,
}

/// Macro for marking stub usage
#[macro_export]
macro_rules! stub_used {
    ($name:expr) => {
        #[cfg(feature = "stub_tracking")]
        {
            $crate::utils::stub_tracking::track_stub_usage($name);
            tracing::warn!(stub = $name, "Stub implementation used");
        }
    };
}

/// Macro for marking intentional stubs
#[macro_export]
macro_rules! stub_intentional {
    ($reason:expr) => {
        #[cfg(feature = "stub_tracking")]
        tracing::debug!(reason = $reason, "Intentional stub (extension point)");
    };
}

/// Macro for marking implemented stubs
#[macro_export]
macro_rules! stub_implemented {
    ($name:expr, $date:expr, $description:expr) => {
        #[cfg(feature = "stub_tracking")]
        tracing::info!(
            stub = $name,
            implemented_date = $date,
            description = $description,
            "Stub has been implemented"
        );
    };
}

/// Builder for creating stub info entries
pub struct StubInfoBuilder {
    id: String,
    file: String,
    line: u32,
    priority: StubPriority,
    category: StubCategory,
    description: String,
    dependencies: Vec<String>,
    intentional: bool,
    risks: Option<String>,
}

impl StubInfoBuilder {
    pub fn new(id: impl Into<String>, file: impl Into<String>, line: u32) -> Self {
        Self {
            id: id.into(),
            file: file.into(),
            line,
            priority: StubPriority::Medium,
            category: StubCategory::Feature,
            description: String::new(),
            dependencies: Vec::new(),
            intentional: false,
            risks: None,
        }
    }
    
    pub fn priority(mut self, priority: StubPriority) -> Self {
        self.priority = priority;
        self
    }
    
    pub fn category(mut self, category: StubCategory) -> Self {
        self.category = category;
        self
    }
    
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }
    
    pub fn depends_on(mut self, stub_id: impl Into<String>) -> Self {
        self.dependencies.push(stub_id.into());
        self
    }
    
    pub fn intentional(mut self, is_intentional: bool) -> Self {
        self.intentional = is_intentional;
        self
    }
    
    pub fn risks(mut self, risks: impl Into<String>) -> Self {
        self.risks = Some(risks.into());
        self
    }
    
    pub fn build(self) -> StubInfo {
        StubInfo {
            id: self.id,
            file: self.file,
            line: self.line,
            priority: self.priority,
            category: self.category,
            description: self.description,
            dependencies: self.dependencies,
            created_at: Utc::now(),
            intentional: self.intentional,
            risks: self.risks,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_stub_registration() {
        let stub = StubInfoBuilder::new("TEST-001", "src/test.rs", 42)
            .priority(StubPriority::High)
            .category(StubCategory::Feature)
            .description("Test stub for unit testing")
            .build();
        
        register_stub(stub.clone());
        
        let registry = STUB_REGISTRY.read().unwrap();
        assert!(registry.contains_key("TEST-001"));
        assert_eq!(registry.get("TEST-001").unwrap().priority, StubPriority::High);
    }
    
    #[test]
    fn test_stub_usage_tracking() {
        track_stub_usage("TEST-USAGE");
        track_stub_usage("TEST-USAGE");
        track_stub_usage("TEST-USAGE");
        
        let usage = STUB_USAGE.read().unwrap();
        let counter = usage.call_counts.get("TEST-USAGE").unwrap();
        assert_eq!(counter.load(Ordering::Relaxed), 3);
    }
    
    #[test]
    fn test_stub_report_generation() {
        // Register some test stubs
        register_stub(
            StubInfoBuilder::new("CRIT-001", "src/safety.rs", 10)
                .priority(StubPriority::Critical)
                .category(StubCategory::Safety)
                .description("Critical safety validation")
                .build()
        );
        
        register_stub(
            StubInfoBuilder::new("LOW-001", "src/plugins.rs", 20)
                .priority(StubPriority::Low)
                .category(StubCategory::Extension)
                .description("Plugin extension point")
                .intentional(true)
                .build()
        );
        
        let report = generate_stub_report();
        assert!(report.critical_stubs.contains(&"CRIT-001".to_string()));
        assert_eq!(
            report.stubs_by_priority.get(&StubPriority::Critical).unwrap().len(),
            1
        );
    }
}