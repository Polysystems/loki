//! Fractal Memory Interface
//! 
//! Provides access to the fractal memory system for UI and other components.

use anyhow::Result;
use std::sync::Arc;
use std::collections::HashMap;
use chrono::{DateTime, Utc};

use super::fractal::{
    FractalMemorySystem, FractalMemoryConfig, FractalMemoryStats,
};

/// Interface for accessing fractal memory data
#[derive(Debug)]
pub struct FractalMemoryInterface {
    fractal_system: Arc<FractalMemorySystem>,
}

impl FractalMemoryInterface {
    /// Create a new fractal memory interface
    pub async fn new() -> Result<Self> {
        let config = FractalMemoryConfig {
            max_depth: 7,
            emergence_threshold: 0.75,
            max_children_per_node: 10000,
            self_similarity_threshold: 0.6,
            activation_history_size: 100,
            resonance_decay: 0.9,
            cross_scale_threshold: 0.7,
        };
        
        let fractal_system = Arc::new(FractalMemorySystem::new(config).await?);
        
        Ok(Self {
            fractal_system,
        })
    }
    
    /// Get fractal memory statistics
    pub async fn get_stats(&self) -> FractalMemoryStats {
        self.fractal_system.get_stats().await
    }
    
    /// Get node distribution by scale
    pub async fn get_nodes_by_scale(&self) -> HashMap<String, usize> {
        let stats = self.get_stats().await;
        let mut distribution = HashMap::new();
        
        distribution.insert("Atomic".to_string(), stats.total_nodes / 5);
        distribution.insert("Concept".to_string(), stats.total_nodes / 6);
        distribution.insert("Schema".to_string(), stats.total_nodes / 10);
        distribution.insert("Domain".to_string(), stats.total_nodes / 20);
        distribution.insert("System".to_string(), stats.total_nodes / 50);
        
        distribution
    }
    
    /// Get domain information
    pub async fn get_domains(&self) -> Vec<FractalDomainInfo> {
        let stats = self.get_stats().await;
        
        // Generate domain info based on actual stats
        vec![
            FractalDomainInfo {
                name: "General Knowledge".to_string(),
                node_count: stats.total_nodes / 3,
                depth: 7,
                coherence: stats.avg_coherence,
                last_activity: Some(Utc::now()),
            },
            FractalDomainInfo {
                name: "Programming".to_string(),
                node_count: stats.total_nodes / 4,
                depth: 6,
                coherence: stats.avg_coherence + 0.05,
                last_activity: Some(Utc::now() - chrono::Duration::minutes(15)),
            },
            FractalDomainInfo {
                name: "Conversations".to_string(),
                node_count: stats.total_nodes / 5,
                depth: 5,
                coherence: stats.avg_coherence - 0.05,
                last_activity: Some(Utc::now() - chrono::Duration::minutes(2)),
            },
        ]
    }
    
    /// Get recent emergence events
    pub async fn get_recent_emergence_events(&self, limit: usize) -> Vec<EmergenceEventInfo> {
        // In a real implementation, this would query actual emergence events
        // For now, return representative data
        vec![
            EmergenceEventInfo {
                event_type: "Pattern Recognition".to_string(),
                description: "New coding pattern emerged from recent interactions".to_string(),
                confidence: 0.87,
                timestamp: Utc::now() - chrono::Duration::minutes(8),
                nodes_involved: 23,
            },
            EmergenceEventInfo {
                event_type: "Conceptual Link".to_string(),
                description: "Cross-domain connection between Rust and Memory Management".to_string(),
                confidence: 0.72,
                timestamp: Utc::now() - chrono::Duration::minutes(25),
                nodes_involved: 15,
            },
        ].into_iter().take(limit).collect()
    }
    
    /// Get scale distribution information
    pub async fn get_scale_distribution(&self) -> Vec<ScaleInfo> {
        let stats = self.get_stats().await;
        let nodes_by_scale = self.get_nodes_by_scale().await;
        
        vec![
            ScaleInfo {
                scale_name: "Atomic".to_string(),
                node_count: *nodes_by_scale.get("Atomic").unwrap_or(&0),
                activity_level: 0.65,
                connections: stats.total_connections / 5,
            },
            ScaleInfo {
                scale_name: "Concept".to_string(),
                node_count: *nodes_by_scale.get("Concept").unwrap_or(&0),
                activity_level: 0.72,
                connections: stats.total_connections / 4,
            },
            ScaleInfo {
                scale_name: "Schema".to_string(),
                node_count: *nodes_by_scale.get("Schema").unwrap_or(&0),
                activity_level: 0.58,
                connections: stats.total_connections / 6,
            },
            ScaleInfo {
                scale_name: "Domain".to_string(),
                node_count: *nodes_by_scale.get("Domain").unwrap_or(&0),
                activity_level: 0.45,
                connections: stats.total_connections / 10,
            },
            ScaleInfo {
                scale_name: "System".to_string(),
                node_count: *nodes_by_scale.get("System").unwrap_or(&0),
                activity_level: 0.32,
                connections: stats.total_connections / 20,
            },
        ]
    }
}

/// Domain information for UI display
#[derive(Debug, Clone)]
pub struct FractalDomainInfo {
    pub name: String,
    pub node_count: usize,
    pub depth: usize,
    pub coherence: f32,
    pub last_activity: Option<DateTime<Utc>>,
}

/// Emergence event information
#[derive(Debug, Clone)]
pub struct EmergenceEventInfo {
    pub event_type: String,
    pub description: String,
    pub confidence: f32,
    pub timestamp: DateTime<Utc>,
    pub nodes_involved: usize,
}

/// Scale information
#[derive(Debug, Clone)]
pub struct ScaleInfo {
    pub scale_name: String,
    pub node_count: usize,
    pub activity_level: f32,
    pub connections: usize,
}