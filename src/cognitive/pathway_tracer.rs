//! Neural Pathway Tracer
//!
//! This module tracks neural activations and analyzes pathway effectiveness,
//! enabling Loki to understand which thought patterns lead to success.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, RwLock};

use crate::cognitive::{Insight, InsightCategory, ThoughtId};
use crate::memory::SimdSmartCache;

/// Neural pathway representing connected thoughts
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct NeuralPathway {
    /// Unique pathway identifier
    pub id: PathwayId,

    /// Thoughts in this pathway
    pub thoughts: Vec<ThoughtId>,

    /// Activation strengths between thoughts
    pub connections: Vec<Connection>,

    /// Overall pathway strength (0.0-1.0)
    pub strength: f32,

    /// Number of activations
    pub activation_count: u32,

    /// Success rate when this pathway activates
    pub success_rate: f32,

    /// Average activation time
    pub avg_activation_time: Duration,

    /// Last activation
    #[serde(skip)]
    pub last_activation: Instant,

    /// Pathway type
    pub pathway_type: PathwayType,
}

impl Default for NeuralPathway {
    fn default() -> Self {
        Self {
            id: PathwayId::new(),
            thoughts: Vec::new(),
            connections: Vec::new(),
            strength: 0.0,
            activation_count: 0,
            success_rate: 0.5,
            avg_activation_time: Duration::from_millis(100),
            last_activation: Instant::now(),
            pathway_type: PathwayType::Direct,
        }
    }
}

#[derive(Clone, Debug, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct PathwayId(pub String);

impl PathwayId {
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4().to_string())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Connection {
    pub from: ThoughtId,
    pub to: ThoughtId,
    pub strength: f32,
    pub activation_count: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum PathwayType {
    Direct,       // Single connection
    Sequential,   // Chain of thoughts
    Branching,    // Multiple outputs
    Convergent,   // Multiple inputs
    Recurrent,    // Contains loops
    Hierarchical, // Parent-child structure
}

/// Activation record for tracking
#[derive(Clone, Debug)]
struct ActivationRecord {
    thought_id: ThoughtId,
    activation_strength: f32,
    timestamp: Instant,
    success: Option<bool>,
}

/// Pathway effectiveness metrics
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PathwayMetrics {
    pub total_activations: u64,
    pub successful_activations: u64,
    pub avg_activation_strength: f32,
    pub avg_time_between_activations: Duration,
    pub decay_rate: f32,
}

/// Main pathway tracer
pub struct PathwayTracer {
    /// All known pathways
    pathways: Arc<RwLock<HashMap<PathwayId, NeuralPathway>>>,

    /// Activation history
    activation_history: Arc<Mutex<VecDeque<ActivationRecord>>>,

    /// Connection strength map
    connections: Arc<RwLock<HashMap<(ThoughtId, ThoughtId), f32>>>,

    /// Pathway metrics
    metrics: Arc<RwLock<HashMap<PathwayId, PathwayMetrics>>>,

    /// Cache for fast access
    cache: Arc<SimdSmartCache>,

    /// Configuration
    config: TracerConfig,
}

#[derive(Clone, Debug)]
pub struct TracerConfig {
    /// Minimum strength to form a connection
    pub min_connection_strength: f32,

    /// Decay rate for unused connections
    pub decay_rate: f32,

    /// Maximum history size
    pub max_history_size: usize,

    /// Pathway detection window
    pub detection_window: Duration,
}

impl Default for TracerConfig {
    fn default() -> Self {
        Self {
            min_connection_strength: 0.3,
            decay_rate: 0.95,
            max_history_size: 10000,
            detection_window: Duration::from_secs(60),
        }
    }
}

impl PathwayTracer {
    pub fn new(cache: Arc<SimdSmartCache>) -> Self {
        Self {
            pathways: Arc::new(RwLock::new(HashMap::new())),
            activation_history: Arc::new(Mutex::new(VecDeque::with_capacity(10000))),
            connections: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(RwLock::new(HashMap::new())),
            cache,
            config: TracerConfig::default(),
        }
    }

    /// Record a thought activation
    pub async fn record_activation(&self, thought_id: &ThoughtId, strength: f32) -> Result<()> {
        let record = ActivationRecord {
            thought_id: thought_id.clone(),
            activation_strength: strength,
            timestamp: Instant::now(),
            success: None,
        };

        // Add to history
        {
            let mut history = self.activation_history.lock().await;
            history.push_back(record.clone());

            // Maintain history size
            if history.len() > self.config.max_history_size {
                history.pop_front();
            }
        }

        // Update connections based on recent history
        self.update_connections(thought_id, strength).await?;

        // Detect pathways
        self.detect_pathways().await?;

        Ok(())
    }

    /// Update connection strengths
    async fn update_connections(&self, current: &ThoughtId, strength: f32) -> Result<()> {
        let history = self.activation_history.lock().await;

        // Look for recent activations to form connections
        let now = Instant::now();
        let mut connections = self.connections.write().await;

        for record in history.iter().rev().skip(1).take(10) {
            let time_diff = now.duration_since(record.timestamp);

            if time_diff < self.config.detection_window {
                // Calculate connection strength based on time and activation strengths
                let time_factor =
                    1.0 - (time_diff.as_secs_f32() / self.config.detection_window.as_secs_f32());
                let conn_strength = strength * record.activation_strength * time_factor;

                if conn_strength >= self.config.min_connection_strength {
                    let key = (record.thought_id.clone(), current.clone());
                    let existing = connections.get(&key).copied().unwrap_or(0.0);
                    connections.insert(key, (existing + conn_strength).min(1.0));
                }
            }
        }

        Ok(())
    }

    /// Detect pathways from activation patterns
    async fn detect_pathways(&self) -> Result<()> {
        let connections = self.connections.read().await;
        let mut pathways = self.pathways.write().await;

        // Group connected thoughts
        let mut thought_groups: HashMap<ThoughtId, Vec<ThoughtId>> = HashMap::new();

        for ((from, to), strength) in connections.iter() {
            if *strength >= self.config.min_connection_strength {
                thought_groups.entry(from.clone()).or_insert_with(Vec::new).push(to.clone());
            }
        }

        // Build pathways from groups
        for (start, _) in &thought_groups {
            let pathway = self.build_pathway(start, &thought_groups, &connections).await?;

            if pathway.thoughts.len() >= 2 {
                pathways.insert(pathway.id.clone(), pathway);
            }
        }

        Ok(())
    }

    /// Build a pathway from a starting thought
    async fn build_pathway(
        &self,
        start: &ThoughtId,
        groups: &HashMap<ThoughtId, Vec<ThoughtId>>,
        connections: &HashMap<(ThoughtId, ThoughtId), f32>,
    ) -> Result<NeuralPathway> {
        let mut thoughts = vec![start.clone()];
        let mut visited = std::collections::HashSet::new();
        visited.insert(start.clone());

        let mut current = start;
        let mut pathway_connections = Vec::new();
        let mut total_strength = 0.0;

        // Follow strongest connections
        while let Some(next_thoughts) = groups.get(current) {
            let mut best_next = None;
            let mut best_strength = 0.0;

            for next in next_thoughts {
                if !visited.contains(next) {
                    if let Some(&strength) = connections.get(&(current.clone(), next.clone())) {
                        if strength > best_strength {
                            best_strength = strength;
                            best_next = Some(next);
                        }
                    }
                }
            }

            if let Some(next) = best_next {
                thoughts.push(next.clone());
                visited.insert(next.clone());

                pathway_connections.push(Connection {
                    from: current.clone(),
                    to: next.clone(),
                    strength: best_strength,
                    activation_count: 1,
                });

                total_strength += best_strength;
                current = next;
            } else {
                break;
            }
        }

        let pathway_type = self.determine_pathway_type(&thoughts, groups);

        let connection_count = pathway_connections.len();

        Ok(NeuralPathway {
            id: PathwayId::new(),
            thoughts,
            connections: pathway_connections,
            strength: if connection_count > 0 {
                total_strength / connection_count as f32
            } else {
                0.0
            },
            activation_count: 1,
            success_rate: 0.5, // Initial neutral rate
            avg_activation_time: Duration::from_millis(100),
            last_activation: Instant::now(),
            pathway_type,
        })
    }

    /// Determine pathway type
    fn determine_pathway_type(
        &self,
        thoughts: &[ThoughtId],
        groups: &HashMap<ThoughtId, Vec<ThoughtId>>,
    ) -> PathwayType {
        if thoughts.len() == 2 {
            PathwayType::Direct
        } else if thoughts
            .windows(2)
            .all(|w| groups.get(&w[0]).map_or(false, |next| next.contains(&w[1])))
        {
            PathwayType::Sequential
        } else {
            // Check for branching
            let has_branches =
                thoughts.iter().any(|t| groups.get(t).map_or(false, |next| next.len() > 1));

            if has_branches { PathwayType::Branching } else { PathwayType::Convergent }
        }
    }

    /// Get active pathways
    pub async fn get_active_pathways(&self) -> Result<Vec<NeuralPathway>> {
        let pathways = self.pathways.read().await;
        let now = Instant::now();

        let active: Vec<_> = pathways
            .values()
            .filter(|p| {
                now.duration_since(p.last_activation) < Duration::from_secs(300) && p.strength > 0.3
            })
            .cloned()
            .collect();

        Ok(active)
    }

    /// Mark pathway outcome
    pub async fn mark_pathway_outcome(&self, pathway_id: &PathwayId, success: bool) -> Result<()> {
        let mut pathways = self.pathways.write().await;

        if let Some(pathway) = pathways.get_mut(pathway_id) {
            let total = pathway.activation_count as f32;
            let successful = pathway.success_rate * total + if success { 1.0 } else { 0.0 };
            pathway.success_rate = successful / (total + 1.0);
            pathway.activation_count += 1;

            // Update metrics
            let mut metrics = self.metrics.write().await;
            let metric = metrics.entry(pathway_id.clone()).or_insert(PathwayMetrics {
                total_activations: 0,
                successful_activations: 0,
                avg_activation_strength: pathway.strength,
                avg_time_between_activations: Duration::from_secs(0),
                decay_rate: self.config.decay_rate,
            });

            metric.total_activations += 1;
            if success {
                metric.successful_activations += 1;
            }
        }

        Ok(())
    }

    /// Analyze pathway effectiveness
    pub async fn analyze_effectiveness(&self) -> Result<Vec<Insight>> {
        let pathways = self.pathways.read().await;
        let metrics = self.metrics.read().await;
        let mut insights = Vec::new();

        // Find highly effective pathways
        for (pathway_id, pathway) in pathways.iter() {
            if let Some(metric) = metrics.get(pathway_id) {
                let effectiveness =
                    metric.successful_activations as f32 / metric.total_activations.max(1) as f32;

                if effectiveness > 0.8 && metric.total_activations > 10 {
                    insights.push(Insight {
                        content: format!(
                            "Highly effective pathway detected: {:?} with {:.1}% success rate",
                            pathway.pathway_type,
                            effectiveness * 100.0
                        ),
                        confidence: effectiveness,
                        category: InsightCategory::Pattern,
                        timestamp: Instant::now(),
                    });
                } else if effectiveness < 0.2 && metric.total_activations > 10 {
                    insights.push(Insight {
                        content: format!(
                            "Ineffective pathway detected: {:?} with only {:.1}% success rate",
                            pathway.pathway_type,
                            effectiveness * 100.0
                        ),
                        confidence: 0.8,
                        category: InsightCategory::Warning,
                        timestamp: Instant::now(),
                    });
                }
            }
        }

        // Analyze pathway diversity
        let pathway_types: HashMap<_, _> =
            pathways.values().map(|p| &p.pathway_type).fold(HashMap::new(), |mut map, t| {
                *map.entry(format!("{:?}", t)).or_insert(0) += 1;
                map
            });

        if pathway_types.len() < 3 {
            insights.push(Insight {
                content: "Limited pathway diversity detected. Consider exploring new thinking \
                          patterns."
                    .to_string(),
                confidence: 0.7,
                category: InsightCategory::Improvement,
                timestamp: Instant::now(),
            });
        }

        Ok(insights)
    }

    /// Apply decay to unused connections
    pub async fn decay_connections(&self) -> Result<()> {
        let mut connections = self.connections.write().await;
        let mut to_remove = Vec::new();

        for (key, strength) in connections.iter_mut() {
            *strength *= self.config.decay_rate;

            if *strength < 0.01 {
                to_remove.push(key.clone());
            }
        }

        for key in to_remove {
            connections.remove(&key);
        }

        Ok(())
    }

    /// Restore a pathway
    pub async fn restore_pathway(&self, pathway: NeuralPathway) -> Result<()> {
        self.pathways.write().await.insert(pathway.id.clone(), pathway);
        Ok(())
    }

    /// Get pathway statistics
    pub async fn get_stats(&self) -> PathwayStats {
        let pathways = self.pathways.read().await;
        let connections = self.connections.read().await;
        let history = self.activation_history.lock().await;

        let active_pathways = pathways.values().filter(|p| p.strength > 0.3).count();

        let avg_pathway_length = if !pathways.is_empty() {
            pathways.values().map(|p| p.thoughts.len()).sum::<usize>() as f32
                / pathways.len() as f32
        } else {
            0.0
        };

        PathwayStats {
            total_pathways: pathways.len(),
            active_pathways,
            total_connections: connections.len(),
            avg_connection_strength: if !connections.is_empty() {
                connections.values().sum::<f32>() / connections.len() as f32
            } else {
                0.0
            },
            avg_pathway_length,
            history_size: history.len(),
        }
    }
}

/// Pathway statistics
#[derive(Debug, Clone)]
pub struct PathwayStats {
    pub total_pathways: usize,
    pub active_pathways: usize,
    pub total_connections: usize,
    pub avg_connection_strength: f32,
    pub avg_pathway_length: f32,
    pub history_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::SimdCacheConfig;

    #[tokio::test]
    async fn test_pathway_detection() {
        let cache = Arc::new(SimdSmartCache::new(SimdCacheConfig::default()));
        let tracer = PathwayTracer::new(cache);

        // Simulate a sequence of activations
        let thoughts = vec![ThoughtId::new(), ThoughtId::new(), ThoughtId::new()];

        // Record activations in sequence
        for (i, thought) in thoughts.iter().enumerate() {
            tracer.record_activation(thought, 0.8 - i as f32 * 0.1).await.unwrap();
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        // Check pathways were detected
        let pathways = tracer.get_active_pathways().await.unwrap();
        assert!(!pathways.is_empty());
    }

    #[tokio::test]
    async fn test_connection_decay() {
        let cache = Arc::new(SimdSmartCache::new(SimdCacheConfig::default()));
        let tracer = PathwayTracer::new(cache);

        let thought1 = ThoughtId::new();
        let thought2 = ThoughtId::new();

        // Create connection
        tracer.record_activation(&thought1, 0.9).await.unwrap();
        tracer.record_activation(&thought2, 0.9).await.unwrap();

        let initial_strength = {
            let connections = tracer.connections.read().await;
            connections.get(&(thought1.clone(), thought2.clone())).copied()
        };

        assert!(initial_strength.is_some());

        // Apply decay
        tracer.decay_connections().await.unwrap();

        let decayed_strength = {
            let connections = tracer.connections.read().await;
            connections.get(&(thought1, thought2)).copied()
        };

        assert!(decayed_strength.unwrap_or(0.0) < initial_strength.unwrap());
    }
}
