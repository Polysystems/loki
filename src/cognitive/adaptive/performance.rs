//! Performance monitoring for adaptive architecture

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct CognitivePerformanceMetrics {
    pub throughput: f64,
    pub latency: f64,
    pub accuracy: f64,
    pub resource_usage: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct TaskPerformance {
    pub task_id: String,
    pub metrics: CognitivePerformanceMetrics,
    pub completed: bool,
}

pub struct CognitivePerformanceMonitor {
    pub metrics: HashMap<String, CognitivePerformanceMetrics>,
}

impl CognitivePerformanceMonitor {
    pub async fn new() -> Result<Self> {
        Ok(Self {
            metrics: HashMap::new(),
        })
    }
}
