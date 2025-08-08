use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tokio::time::{Duration, sleep};
use tracing::{debug, error, info, warn};

use super::ClusterConfig;

/// Node information for coordination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    pub id: String,
    pub capacity: usize,
    pub current_load: usize,
    pub available_memory_gb: f64,
    pub cpu_cores: usize,
    pub gpu_memory_gb: Option<f64>,
    pub network_bandwidth_gbps: f64,
    pub health_status: NodeHealth,
    pub last_heartbeat: DateTime<Utc>,
    pub supported_model_types: Vec<String>,
}

/// Node health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeHealth {
    Healthy,
    Degraded,
    Unhealthy,
    Critical,
    Offline,
}

/// Model deployment information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDeployment {
    pub model_id: String,
    pub node_id: String,
    pub memory_usage_gb: f64,
    pub cpu_usage_percent: f64,
    pub requests_per_second: f64,
    pub deployed_at: DateTime<Utc>,
    pub deployment_status: DeploymentStatus,
    pub migration_cost: Option<f64>,
}

/// Deployment status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeploymentStatus {
    Deploying,
    Running,
    Migrating,
    Stopping,
    Failed,
}

/// Migration plan for rebalancing
#[derive(Debug, Clone)]
pub struct MigrationPlan {
    pub migrations: Vec<Migration>,
    pub estimated_duration_minutes: f64,
    pub total_transfer_gb: f64,
    pub risk_score: f64,
    pub priority: MigrationPriority,
}

/// Individual migration operation
#[derive(Debug, Clone)]
pub struct Migration {
    pub model_id: String,
    pub from_node: String,
    pub to_node: String,
    pub migration_type: MigrationType,
    pub estimated_duration_seconds: f64,
    pub data_transfer_gb: f64,
    pub impact_score: f64,
}

/// Migration type and strategy
#[derive(Debug, Clone)]
pub enum MigrationType {
    LiveMigration,     // Zero-downtime migration
    ColdMigration,     // Stop, transfer, start
    HotStandby,        // Deploy to new node, switch traffic
    GradualTransition, // Gradually move traffic
}

/// Migration priority levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum MigrationPriority {
    Critical, // Emergency rebalancing
    High,     // Performance optimization
    Medium,   // Load distribution
    Low,      // Maintenance
}

/// Completed migration record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletedMigration {
    pub migration_id: String,
    pub model_id: String,
    pub from_node: String,
    pub to_node: String,
    pub started_at: DateTime<Utc>,
    pub completed_at: DateTime<Utc>,
    pub success: bool,
    pub error_message: Option<String>,
    pub actual_duration_seconds: f64,
    pub data_transferred_gb: f64,
}

/// Cluster coordinator for managing node assignments
pub struct ClusterCoordinator {
    config: ClusterConfig,
    nodes: Arc<RwLock<HashMap<String, NodeInfo>>>,
    model_assignments: Arc<RwLock<HashMap<String, Vec<String>>>>, // model_id -> node_ids
    deployments: Arc<RwLock<HashMap<String, ModelDeployment>>>,
    migration_history: Arc<RwLock<Vec<CompletedMigration>>>,
    active_migrations: Arc<RwLock<HashMap<String, Migration>>>,
}

impl ClusterCoordinator {
    /// Create a new cluster coordinator
    pub async fn new(config: ClusterConfig) -> Result<Self> {
        Ok(Self {
            config,
            nodes: Arc::new(RwLock::new(HashMap::new())),
            model_assignments: Arc::new(RwLock::new(HashMap::new())),
            deployments: Arc::new(RwLock::new(HashMap::new())),
            migration_history: Arc::new(RwLock::new(Vec::new())),
            active_migrations: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Register a node with the coordinator
    pub async fn register_node(&self, mut node_info: NodeInfo) -> Result<()> {
        node_info.last_heartbeat = Utc::now();
        info!("Registering node: {} with capacity {}", node_info.id, node_info.capacity);
        self.nodes.write().await.insert(node_info.id.clone(), node_info);
        Ok(())
    }

    /// Unregister a node
    pub async fn unregister_node(&self, node_id: &str) -> Result<()> {
        info!("Unregistering node: {}", node_id);
        self.nodes.write().await.remove(node_id);

        // Remove from model assignments and trigger migrations
        let mut assignments = self.model_assignments.write().await;
        let mut models_to_migrate = Vec::new();

        for (model_id, nodes) in assignments.iter_mut() {
            if let Some(pos) = nodes.iter().position(|id| id == node_id) {
                nodes.remove(pos);
                if nodes.is_empty() {
                    models_to_migrate.push(model_id.clone());
                }
            }
        }

        drop(assignments);

        // Schedule emergency migrations for affected models
        for model_id in models_to_migrate {
            if let Err(e) = self.emergency_migrate_model(&model_id).await {
                error!(
                    "Failed to emergency migrate model {} from node {}: {}",
                    model_id, node_id, e
                );
            }
        }

        Ok(())
    }

    /// Update node heartbeat and health status
    pub async fn update_node_heartbeat(&self, node_id: &str, health: NodeHealth) -> Result<()> {
        let mut nodes = self.nodes.write().await;
        if let Some(node) = nodes.get_mut(node_id) {
            node.last_heartbeat = Utc::now();
            node.health_status = health;
            debug!("Updated heartbeat for node {}", node_id);
        }
        Ok(())
    }

    /// Find the best node for a model with intelligent affinity and
    /// compatibility checking
    pub async fn find_best_node_for_model(&self, model_id: &str) -> Result<String> {
        let nodes = self.nodes.read().await;
        let deployments = self.deployments.read().await;

        debug!("🎯 Finding best node for model {} with intelligent selection", model_id);

        // ✅ ENHANCED: Use model_id for model-specific node affinity and compatibility
        // checking
        let mut candidate_scores = Vec::new();

        for (node_id, node_info) in nodes.iter() {
            if !matches!(node_info.health_status, NodeHealth::Healthy) {
                continue;
            }

            if node_info.current_load >= node_info.capacity {
                continue;
            }

            // Calculate comprehensive node fitness score
            let fitness_score = self
                .calculate_comprehensive_node_fitness(model_id, node_info, &*deployments)
                .await?;

            candidate_scores.push((node_id.clone(), fitness_score as f64));
        }

        if candidate_scores.is_empty() {
            return Err(anyhow::anyhow!("No healthy nodes available with capacity"));
        }

        // Sort by fitness score (higher is better)
        candidate_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let (best_node_id, best_score) = &candidate_scores[0];

        info!(
            "🎯 Selected node {} for model {} (fitness score: {:.3})",
            best_node_id, model_id, best_score
        );

        Ok(best_node_id.clone())
    }

    /// Find optimal node for model with resource constraints and model-specific
    /// optimization
    pub async fn find_optimal_node_for_model(
        &self,
        model_id: &str, /* ✅ ENHANCED: Use model_id for model-specific optimization and
                         * supported_model_types filtering */
        memory_requirement_gb: f64,
        cpu_cores_requirement: usize,
        gpu_memory_requirement_gb: Option<f64>,
    ) -> Result<String> {
        let nodes = self.nodes.read();

        let nodes_guard = nodes.await;
        let best_node = nodes_guard
            .values()
            .filter(|node| {
                // Health check
                matches!(node.health_status, NodeHealth::Healthy) &&
                // Basic capacity check
                node.current_load < node.capacity &&
                // Memory requirement
                node.available_memory_gb >= memory_requirement_gb &&
                // CPU requirement
                node.cpu_cores >= cpu_cores_requirement &&
                // GPU requirement (if specified)
                gpu_memory_requirement_gb.map_or(true, |gpu_req| {
                    node.gpu_memory_gb.map_or(false, |gpu_mem| gpu_mem >= gpu_req)
                }) &&
                // ✅ ENHANCED: Model-specific supported_model_types filtering
                self.is_model_type_supported(model_id, &node.supported_model_types)
            })
            .min_by(|a, b| {
                // ✅ ENHANCED: Multi-criteria optimization with model-specific considerations
                let a_score = self.calculate_enhanced_node_fitness_score(
                    a,
                    model_id,
                    memory_requirement_gb,
                    cpu_cores_requirement,
                );
                let b_score = self.calculate_enhanced_node_fitness_score(
                    b,
                    model_id,
                    memory_requirement_gb,
                    cpu_cores_requirement,
                );
                a_score.partial_cmp(&b_score).unwrap_or(std::cmp::Ordering::Equal)
            })
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "No nodes available meeting requirements for model {}: {} GB memory, {} CPU \
                     cores, {:?} GPU memory",
                    model_id,
                    memory_requirement_gb,
                    cpu_cores_requirement,
                    gpu_memory_requirement_gb
                )
            })?;

        Ok(best_node.id.clone())
    }

    /// Calculate node fitness score for placement optimization
    fn calculate_node_fitness_score(
        &self,
        node: &NodeInfo,
        memory_requirement_gb: f64,
        cpu_cores_requirement: usize,
    ) -> f64 {
        let load_ratio = node.current_load as f64 / node.capacity as f64;
        let memory_ratio = memory_requirement_gb / node.available_memory_gb;
        let cpu_ratio = cpu_cores_requirement as f64 / node.cpu_cores as f64;

        // Lower score is better
        // Penalize high utilization and tight resource fit
        load_ratio * 0.4 + memory_ratio * 0.3 + cpu_ratio * 0.3
    }

    /// Calculate enhanced node fitness score with model-specific optimization
    fn calculate_enhanced_node_fitness_score(
        &self,
        node: &NodeInfo,
        model_id: &str,
        memory_requirement_gb: f64,
        cpu_cores_requirement: usize,
    ) -> f64 {
        // Base fitness score (lower is better)
        let mut base_score =
            self.calculate_node_fitness_score(node, memory_requirement_gb, cpu_cores_requirement);

        // Model-specific optimizations (adjustments to base score)

        // GPU optimization for GPU-dependent models
        if model_id.to_lowercase().contains("llama")
            || model_id.to_lowercase().contains("mistral")
            || model_id.to_lowercase().contains("gpt")
        {
            if node.gpu_memory_gb.is_some() {
                base_score *= 0.8; // 20% bonus for GPU availability
            } else {
                base_score *= 1.2; // 20% penalty for missing GPU
            }
        }

        // Memory optimization for large models
        if model_id.to_lowercase().contains("70b") || model_id.to_lowercase().contains("large") {
            let memory_headroom = node.available_memory_gb - memory_requirement_gb;
            if memory_headroom > 8.0 {
                base_score *= 0.9; // Bonus for large memory headroom
            }
        }

        // Network optimization for chat/instruct models
        if model_id.to_lowercase().contains("chat") || model_id.to_lowercase().contains("instruct")
        {
            if node.network_bandwidth_gbps > 5.0 {
                base_score *= 0.95; // Small bonus for high bandwidth
            }
        }

        // CPU optimization for code models
        if model_id.to_lowercase().contains("code") || model_id.to_lowercase().contains("codellama")
        {
            if node.cpu_cores > 16 {
                base_score *= 0.9; // Bonus for many CPU cores
            }
        }

        base_score
    }

    /// Update node load
    pub async fn update_node_load(&self, node_id: &str, delta: i32) -> Result<()> {
        let mut nodes = self.nodes.write().await;

        if let Some(node) = nodes.get_mut(node_id) {
            if delta > 0 {
                node.current_load = node.current_load.saturating_add(delta.abs() as usize);
            } else {
                node.current_load = node.current_load.saturating_sub(delta.abs() as usize);
            }
            debug!("Updated node {} load to {}", node_id, node.current_load);
        }

        Ok(())
    }

    /// Deploy model to specified node
    pub async fn deploy_model(&self, model_id: &str, node_id: &str, memory_gb: f64) -> Result<()> {
        info!("Deploying model {} to node {}", model_id, node_id);

        // Check node availability
        {
            let nodes = self.nodes.read().await;
            let node =
                nodes.get(node_id).ok_or_else(|| anyhow::anyhow!("Node {} not found", node_id))?;

            if !matches!(node.health_status, NodeHealth::Healthy) {
                return Err(anyhow::anyhow!("Node {} is not healthy", node_id));
            }

            if node.current_load >= node.capacity {
                return Err(anyhow::anyhow!("Node {} is at capacity", node_id));
            }
        }

        // Create deployment record
        let deployment = ModelDeployment {
            model_id: model_id.to_string(),
            node_id: node_id.to_string(),
            memory_usage_gb: memory_gb,
            cpu_usage_percent: 0.0, // Will be updated by monitoring
            requests_per_second: 0.0,
            deployed_at: Utc::now(),
            deployment_status: DeploymentStatus::Deploying,
            migration_cost: None,
        };

        self.deployments.write().await.insert(model_id.to_string(), deployment);

        // Update assignments
        self.model_assignments
            .write()
            .await
            .entry(model_id.to_string())
            .or_insert_with(Vec::new)
            .push(node_id.to_string());

        // Update node load
        self.update_node_load(node_id, 1).await?;

        Ok(())
    }

    /// Rebalance the cluster with sophisticated migration planning
    pub async fn rebalance_cluster(&self) -> Result<()> {
        info!("Starting intelligent cluster rebalance");

        let migration_plan = self.create_migration_plan().await?;

        if migration_plan.migrations.is_empty() {
            debug!("Cluster is already balanced");
            return Ok(());
        }

        info!(
            "Executing migration plan: {} migrations, estimated {} minutes, {} GB transfer",
            migration_plan.migrations.len(),
            migration_plan.estimated_duration_minutes,
            migration_plan.total_transfer_gb
        );

        self.execute_migration_plan(migration_plan).await?;

        Ok(())
    }

    /// Create comprehensive migration plan
    async fn create_migration_plan(&self) -> Result<MigrationPlan> {
        let nodes = self.nodes.read().await;
        let deployments = self.deployments.read().await;

        if nodes.is_empty() {
            return Ok(MigrationPlan {
                migrations: Vec::new(),
                estimated_duration_minutes: 0.0,
                total_transfer_gb: 0.0,
                risk_score: 0.0,
                priority: MigrationPriority::Low,
            });
        }

        // Calculate load statistics
        let total_load: usize = nodes.values().map(|n| n.current_load).sum();
        let avg_load = total_load as f64 / nodes.len() as f64;
        let load_threshold = 0.2; // 20% deviation threshold

        // Find overloaded and underloaded nodes
        let overloaded: Vec<_> = nodes
            .values()
            .filter(|n| n.current_load as f64 > avg_load * (1.0 + load_threshold))
            .collect();

        let underloaded: Vec<_> = nodes
            .values()
            .filter(|n| (n.current_load as f64) < avg_load * (1.0 - load_threshold))
            .collect();

        let mut migrations = Vec::new();
        let mut total_transfer_gb = 0.0;

        // Create migrations from overloaded to underloaded nodes
        for overloaded_node in &overloaded {
            for underloaded_node in &underloaded {
                if overloaded_node.current_load <= underloaded_node.current_load {
                    continue; // Skip if already balanced
                }

                // Find models on overloaded node that can be migrated
                let candidate_models: Vec<_> = deployments
                    .values()
                    .filter(|d| d.node_id == overloaded_node.id)
                    .filter(|d| matches!(d.deployment_status, DeploymentStatus::Running))
                    .collect();

                for deployment in candidate_models {
                    if underloaded_node.current_load >= underloaded_node.capacity {
                        break; // Target node is full
                    }

                    // Calculate migration parameters
                    let migration_type =
                        self.determine_migration_type(deployment, &underloaded_node);
                    let data_transfer = deployment.memory_usage_gb * 1.2; // Overhead factor
                    let duration = self.estimate_migration_duration(
                        data_transfer,
                        overloaded_node.network_bandwidth_gbps,
                        &migration_type,
                    );

                    migrations.push(Migration {
                        model_id: deployment.model_id.clone(),
                        from_node: overloaded_node.id.clone(),
                        to_node: underloaded_node.id.clone(),
                        migration_type,
                        estimated_duration_seconds: duration,
                        data_transfer_gb: data_transfer,
                        impact_score: self.calculate_migration_impact(deployment),
                    });

                    total_transfer_gb += data_transfer;

                    // Stop if we've balanced enough
                    if migrations.len() >= 10 {
                        // Limit migrations per plan
                        break;
                    }
                }
            }
        }

        // Sort migrations by impact (lowest impact first)
        migrations.sort_by(|a, b| {
            a.impact_score.partial_cmp(&b.impact_score).unwrap_or(std::cmp::Ordering::Equal)
        });

        let estimated_duration_minutes =
            migrations.iter().map(|m| m.estimated_duration_seconds / 60.0).fold(0.0, f64::max); // Parallel execution, so max duration

        let risk_score = self.calculate_plan_risk_score(&migrations);
        let priority = self.determine_migration_priority(&overloaded, &underloaded, risk_score);

        Ok(MigrationPlan {
            migrations,
            estimated_duration_minutes,
            total_transfer_gb,
            risk_score,
            priority,
        })
    }

    /// Determine appropriate migration type
    fn determine_migration_type(
        &self,
        deployment: &ModelDeployment,
        target_node: &NodeInfo,
    ) -> MigrationType {
        // Consider request rate, model size, and node capabilities
        if deployment.requests_per_second > 100.0 {
            MigrationType::LiveMigration // High traffic models need zero downtime
        } else if deployment.memory_usage_gb > 10.0 {
            MigrationType::HotStandby // Large models benefit from standby approach
        } else if target_node.network_bandwidth_gbps > 1.0 {
            MigrationType::GradualTransition // Good network supports gradual transition
        } else {
            MigrationType::ColdMigration // Fallback for simple cases
        }
    }

    /// Estimate migration duration based on data size and method
    fn estimate_migration_duration(
        &self,
        data_gb: f64,
        bandwidth_gbps: f64,
        migration_type: &MigrationType,
    ) -> f64 {
        let base_transfer_time = data_gb / bandwidth_gbps.max(0.1); // Seconds

        match migration_type {
            MigrationType::LiveMigration => base_transfer_time * 1.5, // Overhead for live sync
            MigrationType::ColdMigration => base_transfer_time + 30.0, // Plus startup time
            MigrationType::HotStandby => base_transfer_time + 60.0,   /* Plus deployment and
                                                                        * switch time */
            MigrationType::GradualTransition => base_transfer_time * 2.0, // Gradual process
        }
    }

    /// Calculate migration impact score (lower is better)
    fn calculate_migration_impact(&self, deployment: &ModelDeployment) -> f64 {
        let request_impact = deployment.requests_per_second / 1000.0; // Normalize
        let memory_impact = deployment.memory_usage_gb / 100.0; // Normalize
        let cpu_impact = deployment.cpu_usage_percent / 100.0;

        request_impact * 0.5 + memory_impact * 0.3 + cpu_impact * 0.2
    }

    /// Calculate overall risk score for migration plan
    fn calculate_plan_risk_score(&self, migrations: &[Migration]) -> f64 {
        if migrations.is_empty() {
            return 0.0;
        }

        let total_data = migrations.iter().map(|m| m.data_transfer_gb).sum::<f64>();
        let max_duration =
            migrations.iter().map(|m| m.estimated_duration_seconds).fold(0.0, f64::max);
        let avg_impact =
            migrations.iter().map(|m| m.impact_score).sum::<f64>() / migrations.len() as f64;

        // Risk factors: data volume, duration, impact
        (total_data / 100.0) * 0.3 + (max_duration / 3600.0) * 0.4 + avg_impact * 0.3
    }

    /// Determine migration priority based on cluster state
    fn determine_migration_priority(
        &self,
        overloaded: &[&NodeInfo],
        underloaded: &[&NodeInfo],
        risk_score: f64,
    ) -> MigrationPriority {
        let overload_severity = overloaded
            .iter()
            .map(|n| n.current_load as f64 / n.capacity as f64)
            .fold(0.0, f64::max);

        if overload_severity > 0.9 {
            MigrationPriority::Critical
        } else if overload_severity > 0.8 || risk_score < 0.3 {
            MigrationPriority::High
        } else if overload_severity > 0.7 || !underloaded.is_empty() {
            MigrationPriority::Medium
        } else {
            MigrationPriority::Low
        }
    }

    /// Execute migration plan with parallel processing and monitoring
    async fn execute_migration_plan(&self, plan: MigrationPlan) -> Result<()> {
        let migration_tasks: Vec<_> = plan
            .migrations
            .into_iter()
            .enumerate()
            .map(|(i, migration)| {
                let active_migrations = Arc::clone(&self.active_migrations);
                let migration_history = Arc::clone(&self.migration_history);

                async move {
                    let migration_id = format!("migration_{}", i);

                    // Record active migration
                    active_migrations.write().await.insert(migration_id.clone(), migration.clone());

                    let started_at = Utc::now();
                    let result = self.execute_single_migration(&migration).await;
                    let completed_at = Utc::now();

                    // Record completion
                    let duration = (completed_at - started_at).num_seconds() as f64;
                    let completed_migration = CompletedMigration {
                        migration_id: migration_id.clone(),
                        model_id: migration.model_id.clone(),
                        from_node: migration.from_node.clone(),
                        to_node: migration.to_node.clone(),
                        started_at,
                        completed_at,
                        success: result.is_ok(),
                        error_message: result.as_ref().err().map(|e| e.to_string()),
                        actual_duration_seconds: duration,
                        data_transferred_gb: migration.data_transfer_gb,
                    };

                    migration_history.write().await.push(completed_migration);
                    active_migrations.write().await.remove(&migration_id);

                    result
                }
            })
            .collect();

        // Execute migrations with limited concurrency
        let max_concurrent = self.config.max_concurrent_migrations.unwrap_or(3);
        let semaphore = Arc::new(tokio::sync::Semaphore::new(max_concurrent));

        let migration_futures = migration_tasks.into_iter().map(|task| {
            let semaphore = Arc::clone(&semaphore);
            async move {
                let _permit = semaphore.acquire().await?;
                task.await
            }
        });

        // Wait for all migrations to complete
        let results: Vec<Result<()>> = futures::future::join_all(migration_futures).await;

        // Report results
        let successful = results.iter().filter(|r| r.is_ok()).count();
        let failed = results.len() - successful;

        info!("Migration plan completed: {} successful, {} failed", successful, failed);

        if failed > 0 {
            warn!("Some migrations failed, cluster may not be optimally balanced");
        }

        Ok(())
    }

    /// Execute a single migration
    async fn execute_single_migration(&self, migration: &Migration) -> Result<()> {
        info!(
            "Executing migration: {} from {} to {}",
            migration.model_id, migration.from_node, migration.to_node
        );

        match migration.migration_type {
            MigrationType::LiveMigration => self.execute_live_migration(migration).await,
            MigrationType::ColdMigration => self.execute_cold_migration(migration).await,
            MigrationType::HotStandby => self.execute_hot_standby_migration(migration).await,
            MigrationType::GradualTransition => self.execute_gradual_migration(migration).await,
        }
    }

    /// Execute live migration (zero downtime)
    async fn execute_live_migration(&self, migration: &Migration) -> Result<()> {
        debug!("Starting live migration for model {}", migration.model_id);

        // 1. Deploy model to target node
        self.deploy_model(&migration.model_id, &migration.to_node, migration.data_transfer_gb)
            .await?;

        // 2. Wait for deployment to be ready
        sleep(Duration::from_secs(30)).await;

        // 3. Start routing traffic to new node gradually
        // (In real implementation, this would interact with load balancer)

        // 4. Wait for traffic to fully migrate
        sleep(Duration::from_secs(60)).await;

        // 5. Stop old deployment
        self.stop_model_deployment(&migration.model_id, &migration.from_node).await?;

        info!("Live migration completed for model {}", migration.model_id);
        Ok(())
    }

    /// Execute cold migration (with downtime)
    async fn execute_cold_migration(&self, migration: &Migration) -> Result<()> {
        debug!("Starting cold migration for model {}", migration.model_id);

        // 1. Stop model on source node
        self.stop_model_deployment(&migration.model_id, &migration.from_node).await?;

        // 2. Transfer model data (simulated)
        let transfer_time = migration.data_transfer_gb / 2.0; // Assume 2 GB/s
        sleep(Duration::from_secs(transfer_time as u64)).await;

        // 3. Deploy model on target node
        self.deploy_model(&migration.model_id, &migration.to_node, migration.data_transfer_gb)
            .await?;

        info!("Cold migration completed for model {}", migration.model_id);
        Ok(())
    }

    /// Execute hot standby migration
    async fn execute_hot_standby_migration(&self, migration: &Migration) -> Result<()> {
        debug!("Starting hot standby migration for model {}", migration.model_id);

        // 1. Deploy standby instance
        self.deploy_model(&migration.model_id, &migration.to_node, migration.data_transfer_gb)
            .await?;

        // 2. Sync data to standby
        sleep(Duration::from_secs(30)).await;

        // 3. Switch traffic atomically
        // (In real implementation, this would be atomic routing switch)

        // 4. Stop old instance
        self.stop_model_deployment(&migration.model_id, &migration.from_node).await?;

        info!("Hot standby migration completed for model {}", migration.model_id);
        Ok(())
    }

    /// Execute gradual migration
    async fn execute_gradual_migration(&self, migration: &Migration) -> Result<()> {
        debug!("Starting gradual migration for model {}", migration.model_id);

        // 1. Deploy to target with limited capacity
        self.deploy_model(&migration.model_id, &migration.to_node, migration.data_transfer_gb)
            .await?;

        // 2. Gradually shift traffic (25%, 50%, 75%, 100%)
        for percentage in [25, 50, 75, 100] {
            debug!("Shifting {}% traffic to new node", percentage);
            sleep(Duration::from_secs(30)).await;
        }

        // 3. Stop old deployment
        self.stop_model_deployment(&migration.model_id, &migration.from_node).await?;

        info!("Gradual migration completed for model {}", migration.model_id);
        Ok(())
    }

    /// Stop model deployment on a node
    async fn stop_model_deployment(&self, model_id: &str, node_id: &str) -> Result<()> {
        info!("Stopping model {} on node {}", model_id, node_id);

        // Update deployment status
        if let Some(deployment) = self.deployments.write().await.get_mut(model_id) {
            deployment.deployment_status = DeploymentStatus::Stopping;
        }

        // Remove from assignments
        if let Some(nodes) = self.model_assignments.write().await.get_mut(model_id) {
            nodes.retain(|id| id != node_id);
        }

        // Update node load
        self.update_node_load(node_id, -1).await?;

        // Simulate stopping time
        sleep(Duration::from_secs(5)).await;

        Ok(())
    }

    /// Emergency migrate a model to any available node
    async fn emergency_migrate_model(&self, model_id: &str) -> Result<()> {
        warn!("Emergency migration required for model {}", model_id);

        let target_node = self.find_best_node_for_model(model_id).await?;

        // Quick deployment without elaborate planning
        let migration = Migration {
            model_id: model_id.to_string(),
            from_node: "emergency".to_string(),
            to_node: target_node,
            migration_type: MigrationType::ColdMigration,
            estimated_duration_seconds: 60.0,
            data_transfer_gb: 1.0, // Estimate
            impact_score: 1.0,     // High impact for emergency
        };

        self.execute_single_migration(&migration).await?;

        info!("Emergency migration completed for model {}", model_id);
        Ok(())
    }

    /// Get cluster topology
    pub async fn get_topology(&self) -> HashMap<String, NodeInfo> {
        self.nodes.read().await.clone()
    }

    /// Calculate comprehensive node fitness score for model placement
    async fn calculate_comprehensive_node_fitness(
        &self,
        model_id: &str,
        node_info: &NodeInfo,
        deployments: &HashMap<String, ModelDeployment>,
    ) -> Result<f64> {
        let mut fitness_score = 0.0;

        // Base load balance score (0.0-0.4)
        let load_ratio = node_info.current_load as f64 / node_info.capacity as f64;
        fitness_score += (1.0 - load_ratio) * 0.4;

        // Resource availability score (0.0-0.3) - factor in existing deployments
        let memory_availability = node_info.available_memory_gb / 32.0; // Assume 32GB baseline

        // Reduce fitness if model is already deployed on this node (avoid duplication)
        let deployment_penalty = if deployments.contains_key(model_id) {
            0.05 // Small penalty for duplicate deployments
        } else {
            0.0
        };
        fitness_score -= deployment_penalty;
        fitness_score += memory_availability.min(1.0) * 0.15;

        if let Some(gpu_memory) = node_info.gpu_memory_gb {
            let gpu_availability = gpu_memory / 16.0; // Assume 16GB GPU baseline
            fitness_score += gpu_availability.min(1.0) * 0.15;
        }

        // Model-specific compatibility score (0.0-0.2)
        fitness_score += self.calculate_model_node_compatibility(model_id, node_info).await;

        // Supported model types bonus (0.0-0.1)
        if self.is_model_type_supported(model_id, &node_info.supported_model_types) {
            fitness_score += 0.1;
        }

        Ok(fitness_score.clamp(0.0, 1.0))
    }

    /// Calculate model-node compatibility score
    async fn calculate_model_node_compatibility(
        &self,
        model_id: &str,
        node_info: &NodeInfo,
    ) -> f64 {
        let mut compatibility_score = 0.0;

        // CPU cores adequacy based on model requirements
        let cpu_score = (node_info.cpu_cores as f64 / 8.0).min(1.0); // 8 cores baseline
        compatibility_score += cpu_score * 0.1;

        // Model-specific compatibility adjustments
        let model_type_bonus = if model_id.contains("large") || model_id.contains("70b") {
            // Large models need more resources
            if node_info.cpu_cores >= 16 && node_info.available_memory_gb >= 64.0 {
                0.2 // Bonus for high-capacity nodes with large models
            } else {
                -0.1 // Penalty if insufficient resources
            }
        } else if model_id.contains("small") || model_id.contains("7b") {
            // Small models are more flexible
            0.1
        } else {
            0.0
        };
        compatibility_score += model_type_bonus;

        // Network bandwidth adequacy
        let network_score = (node_info.network_bandwidth_gbps / 10.0).min(1.0); // 10 Gbps baseline
        compatibility_score += network_score * 0.1;

        compatibility_score
    }

    /// Check if model type is supported on the node
    fn is_model_type_supported(&self, model_id: &str, supported_types: &[String]) -> bool {
        // Extract model type from model_id (simplified heuristic)
        for supported_type in supported_types {
            if model_id.to_lowercase().contains(&supported_type.to_lowercase()) {
                return true;
            }
        }

        // Default supported types
        supported_types
            .iter()
            .any(|t| matches!(t.as_str(), "llama" | "mistral" | "gpt" | "gemma" | "general"))
    }

    /// Get migration statistics
    pub async fn get_migration_statistics(&self) -> (usize, usize, f64) {
        let history = self.migration_history.read().await;
        let active = self.active_migrations.read().await;

        let completed = history.len();
        let active_count = active.len();
        let success_rate = if completed > 0 {
            history.iter().filter(|m| m.success).count() as f64 / completed as f64
        } else {
            0.0
        };

        (completed, active_count, success_rate)
    }

    /// Get active nodes (stub implementation)
    pub async fn get_active_nodes(&self) -> Result<Vec<NodeInfo>> {
        let nodes = self.nodes.read().await;
        Ok(nodes.values()
            .filter(|node| matches!(node.health_status, NodeHealth::Healthy | NodeHealth::Degraded))
            .cloned()
            .collect())
    }
}
