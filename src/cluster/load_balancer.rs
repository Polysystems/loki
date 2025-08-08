use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::Result;
use parking_lot::Mutex;

/// Load balancing strategy
#[derive(Debug, Clone, Copy)]
pub enum BalancingStrategy {
    RoundRobin,
    LeastConnections,
    WeightedRoundRobin,
    ResponseTime,
}

/// Load balancer for distributing requests
pub struct LoadBalancer {
    strategy: BalancingStrategy,
    round_robin_counter: AtomicUsize,
    node_weights: Arc<Mutex<HashMap<String, f32>>>,
    response_times: Arc<Mutex<HashMap<String, Vec<f64>>>>,
}

impl LoadBalancer {
    /// Create a new load balancer
    pub fn new(strategy: BalancingStrategy) -> Self {
        Self {
            strategy,
            round_robin_counter: AtomicUsize::new(0),
            node_weights: Arc::new(Mutex::new(HashMap::new())),
            response_times: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Select a node based on the configured strategy
    pub fn select_node(&self, available_nodes: &[(String, u32)]) -> Result<String> {
        if available_nodes.is_empty() {
            anyhow::bail!("No available nodes");
        }

        match self.strategy {
            BalancingStrategy::RoundRobin => self.round_robin(available_nodes),
            BalancingStrategy::LeastConnections => self.least_connections(available_nodes),
            BalancingStrategy::WeightedRoundRobin => self.weighted_round_robin(available_nodes),
            BalancingStrategy::ResponseTime => self.response_time_based(available_nodes),
        }
    }

    /// Round-robin selection
    fn round_robin(&self, nodes: &[(String, u32)]) -> Result<String> {
        let index = self.round_robin_counter.fetch_add(1, Ordering::Relaxed) % nodes.len();
        Ok(nodes[index].0.clone())
    }

    /// Least connections selection
    fn least_connections(&self, nodes: &[(String, u32)]) -> Result<String> {
        nodes
            .iter()
            .min_by_key(|(_, connections)| connections)
            .map(|(id, _)| id.clone())
            .ok_or_else(|| anyhow::anyhow!("Failed to select node"))
    }

    /// Weighted round-robin selection
    fn weighted_round_robin(&self, nodes: &[(String, u32)]) -> Result<String> {
        let weights = self.node_weights.lock();

        // Calculate total weight
        let total_weight: f32 =
            nodes.iter().map(|(id, _)| weights.get(id).copied().unwrap_or(1.0)).sum();

        if total_weight == 0.0 {
            return self.round_robin(nodes);
        }

        // Select based on weight
        let mut random = rand::random::<f32>() * total_weight;

        for (id, _) in nodes {
            let weight = weights.get(id).copied().unwrap_or(1.0);
            if random < weight {
                return Ok(id.clone());
            }
            random -= weight;
        }

        // Fallback to first node
        Ok(nodes[0].0.clone())
    }

    /// Response time based selection
    fn response_time_based(&self, nodes: &[(String, u32)]) -> Result<String> {
        let response_times = self.response_times.lock();

        // Find node with lowest average response time
        let mut best_node = None;
        let mut best_avg = f64::MAX;

        for (id, _) in nodes {
            if let Some(times) = response_times.get(id) {
                if !times.is_empty() {
                    let avg = times.iter().sum::<f64>() / times.len() as f64;
                    if avg < best_avg {
                        best_avg = avg;
                        best_node = Some(id.clone());
                    }
                }
            } else {
                // No history, consider it as best
                return Ok(id.clone());
            }
        }

        best_node.ok_or_else(|| anyhow::anyhow!("Failed to select node"))
    }

    /// Update node weight
    pub fn set_node_weight(&self, node_id: &str, weight: f32) {
        self.node_weights.lock().insert(node_id.to_string(), weight);
    }

    /// Record response time for a node
    pub fn record_response_time(&self, node_id: &str, time_ms: f64) {
        let mut response_times = self.response_times.lock();
        let times = response_times.entry(node_id.to_string()).or_insert_with(Vec::new);

        times.push(time_ms);

        // Keep only last 100 samples
        if times.len() > 100 {
            times.remove(0);
        }
    }

    /// Get current strategy
    pub fn strategy(&self) -> BalancingStrategy {
        self.strategy
    }

    /// Change strategy
    pub fn set_strategy(&mut self, strategy: BalancingStrategy) {
        self.strategy = strategy;
    }
}
