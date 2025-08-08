//! Task Progress Aggregator
//! 
//! Aggregates progress across subtasks and updates parent task progress.
//! Handles real-time progress updates from agent streams.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use tokio::sync::{RwLock, mpsc, broadcast};
use tracing::{info, warn, debug};

use crate::tools::task_management::{TaskManager,  TaskStatus};
use crate::tui::ui::chat::agent_stream_manager::{AgentStreamManager, AgentStatus};

/// Progress aggregation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressAggregatorConfig {
    /// Update interval for progress calculations
    pub update_interval: Duration,
    
    /// Enable automatic parent task updates
    pub auto_update_parents: bool,
    
    /// Progress calculation method
    pub calculation_method: ProgressCalculationMethod,
    
    /// Minimum progress change to trigger update
    pub min_progress_delta: f32,
}

impl Default for ProgressAggregatorConfig {
    fn default() -> Self {
        Self {
            update_interval: Duration::from_secs(5),
            auto_update_parents: true,
            calculation_method: ProgressCalculationMethod::WeightedAverage,
            min_progress_delta: 0.05, // 5% change
        }
    }
}

/// Progress calculation method
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProgressCalculationMethod {
    /// Simple average of all subtask progress
    SimpleAverage,
    
    /// Weighted by estimated effort
    WeightedAverage,
    
    /// Minimum progress of all subtasks
    MinimumProgress,
    
    /// Custom calculation function
    Custom,
}

/// Task progress aggregator
pub struct TaskProgressAggregator {
    /// Configuration
    config: ProgressAggregatorConfig,
    
    /// Task manager reference
    task_manager: Arc<TaskManager>,
    
    /// Agent stream manager reference
    agent_stream_manager: Arc<AgentStreamManager>,
    
    /// Task hierarchy tracking
    task_hierarchy: Arc<RwLock<TaskHierarchy>>,
    
    /// Progress tracking
    progress_tracker: Arc<RwLock<ProgressTracker>>,
    
    /// Update channel
    update_tx: mpsc::Sender<ProgressUpdate>,
    update_rx: Arc<RwLock<mpsc::Receiver<ProgressUpdate>>>,
    
    /// Broadcast channel for UI updates
    broadcast_tx: broadcast::Sender<AggregatedProgress>,
}

/// Task hierarchy information
#[derive(Debug, Default)]
struct TaskHierarchy {
    /// Parent task to subtasks mapping
    parent_to_subtasks: HashMap<String, Vec<String>>,
    
    /// Subtask to parent mapping
    subtask_to_parent: HashMap<String, String>,
    
    /// Task to agent mapping
    task_to_agent: HashMap<String, String>,
}

/// Progress tracking information
#[derive(Debug, Default)]
struct ProgressTracker {
    /// Individual task progress
    task_progress: HashMap<String, TaskProgress>,
    
    /// Aggregated parent progress
    parent_progress: HashMap<String, AggregatedProgress>,
    
    /// Last update timestamps
    last_updates: HashMap<String, Instant>,
}

/// Individual task progress
#[derive(Debug, Clone)]
pub struct TaskProgress {
    pub task_id: String,
    pub status: TaskStatus,
    pub progress: f32, // 0.0 to 1.0
    pub agent_id: Option<String>,
    pub started_at: Option<Instant>,
    pub updated_at: Instant,
    pub estimated_remaining: Option<Duration>,
    pub message: Option<String>,
}

/// Aggregated progress for parent task
#[derive(Debug, Clone)]
pub struct AggregatedProgress {
    pub parent_task_id: String,
    pub overall_progress: f32,
    pub subtask_count: usize,
    pub completed_count: usize,
    pub in_progress_count: usize,
    pub failed_count: usize,
    pub status: TaskStatus,
    pub estimated_completion: Option<Instant>,
    pub subtask_details: Vec<SubtaskProgress>,
}

/// Subtask progress summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubtaskProgress {
    pub task_id: String,
    pub description: String,
    pub progress: f32,
    pub status: TaskStatus,
    pub agent_id: Option<String>,
}

/// Progress update message
#[derive(Debug, Clone)]
enum ProgressUpdate {
    TaskProgress {
        task_id: String,
        progress: f32,
        status: TaskStatus,
        message: Option<String>,
    },
    TaskCompleted {
        task_id: String,
        success: bool,
    },
    AgentStatusChange {
        agent_id: String,
        status: AgentStatus,
    },
}

impl TaskProgressAggregator {
    pub fn new(
        task_manager: Arc<TaskManager>,
        agent_stream_manager: Arc<AgentStreamManager>,
        config: ProgressAggregatorConfig,
    ) -> Self {
        let (update_tx, update_rx) = mpsc::channel(100);
        let (broadcast_tx, _) = broadcast::channel(50);
        
        Self {
            config,
            task_manager,
            agent_stream_manager,
            task_hierarchy: Arc::new(RwLock::new(TaskHierarchy::default())),
            progress_tracker: Arc::new(RwLock::new(ProgressTracker::default())),
            update_tx,
            update_rx: Arc::new(RwLock::new(update_rx)),
            broadcast_tx,
        }
    }
    
    /// Register a task hierarchy
    pub async fn register_task_hierarchy(
        &self,
        parent_task_id: String,
        subtask_ids: Vec<String>,
        task_to_agent_map: HashMap<String, String>,
    ) -> Result<()> {
        let mut hierarchy = self.task_hierarchy.write().await;
        
        // Store parent-subtask relationships
        hierarchy.parent_to_subtasks.insert(parent_task_id.clone(), subtask_ids.clone());
        
        // Store subtask-parent relationships
        for subtask_id in &subtask_ids {
            hierarchy.subtask_to_parent.insert(subtask_id.clone(), parent_task_id.clone());
        }
        
        // Store task-agent mappings
        for (task_id, agent_id) in task_to_agent_map {
            hierarchy.task_to_agent.insert(task_id, agent_id);
        }
        
        // Initialize progress tracking
        let mut tracker = self.progress_tracker.write().await;
        for subtask_id in &subtask_ids {
            tracker.task_progress.insert(subtask_id.clone(), TaskProgress {
                task_id: subtask_id.clone(),
                status: TaskStatus::Todo,
                progress: 0.0,
                agent_id: hierarchy.task_to_agent.get(subtask_id).cloned(),
                started_at: None,
                updated_at: Instant::now(),
                estimated_remaining: None,
                message: None,
            });
        }
        
        info!("Registered task hierarchy: {} with {} subtasks", parent_task_id, subtask_ids.len());
        Ok(())
    }
    
    /// Start monitoring progress
    pub async fn start_monitoring(&self) -> Result<()> {
        // Start update processing task
        let update_processor = self.clone();
        tokio::spawn(async move {
            update_processor.process_updates().await;
        });
        
        // Start periodic aggregation task
        let aggregator = self.clone();
        tokio::spawn(async move {
            aggregator.periodic_aggregation().await;
        });
        
        // Start agent stream monitoring
        let stream_monitor = self.clone();
        tokio::spawn(async move {
            stream_monitor.monitor_agent_streams().await;
        });
        
        Ok(())
    }
    
    /// Process progress updates
    async fn process_updates(&self) {
        let mut rx = self.update_rx.write().await;
        
        while let Some(update) = rx.recv().await {
            match update {
                ProgressUpdate::TaskProgress { task_id, progress, status, message } => {
                    if let Err(e) = self.update_task_progress(&task_id, progress, status, message).await {
                        warn!("Failed to update task progress: {}", e);
                    }
                }
                ProgressUpdate::TaskCompleted { task_id, success } => {
                    let status = if success { TaskStatus::Done } else { TaskStatus::Cancelled };
                    if let Err(e) = self.update_task_progress(&task_id, 1.0, status, None).await {
                        warn!("Failed to update completed task: {}", e);
                    }
                }
                ProgressUpdate::AgentStatusChange { agent_id, status } => {
                    if let Err(e) = self.handle_agent_status_change(&agent_id, status).await {
                        warn!("Failed to handle agent status change: {}", e);
                    }
                }
            }
        }
    }
    
    /// Update individual task progress
    async fn update_task_progress(
        &self,
        task_id: &str,
        progress: f32,
        status: TaskStatus,
        message: Option<String>,
    ) -> Result<()> {
        let mut tracker = self.progress_tracker.write().await;
        
        if let Some(task_progress) = tracker.task_progress.get_mut(task_id) {
            let old_progress = task_progress.progress;
            
            task_progress.progress = progress.clamp(0.0, 1.0);
            task_progress.status = status.clone();
            task_progress.updated_at = Instant::now();
            
            if let Some(msg) = message {
                task_progress.message = Some(msg);
            }
            
            // Set started_at if transitioning from Todo
            if old_progress == 0.0 && progress > 0.0 && task_progress.started_at.is_none() {
                task_progress.started_at = Some(Instant::now());
            }
            
            // Check if we need to update parent
            let progress_delta = (progress - old_progress).abs();
            if progress_delta >= self.config.min_progress_delta || status == TaskStatus::Done {
                drop(tracker); // Release lock before aggregating
                self.aggregate_parent_progress(task_id).await?;
            }
        }
        
        Ok(())
    }
    
    /// Handle agent status changes
    async fn handle_agent_status_change(&self, agent_id: &str, status: AgentStatus) -> Result<()> {
        let hierarchy = self.task_hierarchy.read().await;
        
        // Find tasks assigned to this agent
        let task_ids: Vec<String> = hierarchy.task_to_agent.iter()
            .filter_map(|(task_id, agent)| {
                if agent == agent_id {
                    Some(task_id.clone())
                } else {
                    None
                }
            })
            .collect();
            
        drop(hierarchy);
        
        // Update task status based on agent status
        for task_id in task_ids {
            let (new_status, progress) = match status {
                AgentStatus::Running => (TaskStatus::InProgress, None),
                AgentStatus::Completed => (TaskStatus::Done, Some(1.0)),
                AgentStatus::Failed => (TaskStatus::Cancelled, None),
                _ => continue,
            };
            
            if let Some(prog) = progress {
                self.update_task_progress(&task_id, prog, new_status, None).await?;
            } else {
                let mut tracker = self.progress_tracker.write().await;
                if let Some(task_progress) = tracker.task_progress.get_mut(&task_id) {
                    task_progress.status = new_status;
                }
            }
        }
        
        Ok(())
    }
    
    /// Aggregate progress for parent task
    async fn aggregate_parent_progress(&self, subtask_id: &str) -> Result<()> {
        let hierarchy = self.task_hierarchy.read().await;
        
        if let Some(parent_id) = hierarchy.subtask_to_parent.get(subtask_id) {
            let parent_id = parent_id.clone();
            let subtask_ids = hierarchy.parent_to_subtasks.get(&parent_id)
                .cloned()
                .unwrap_or_default();
                
            drop(hierarchy);
            
            // Calculate aggregated progress
            let aggregated = self.calculate_aggregated_progress(&parent_id, &subtask_ids).await?;
            
            // Store aggregated progress
            let mut tracker = self.progress_tracker.write().await;
            tracker.parent_progress.insert(parent_id.clone(), aggregated.clone());
            tracker.last_updates.insert(parent_id.clone(), Instant::now());
            drop(tracker);
            
            // Update parent task in task manager if configured
            if self.config.auto_update_parents {
                if let Err(e) = self.update_parent_task(&parent_id, &aggregated).await {
                    warn!("Failed to update parent task: {}", e);
                }
            }
            
            // Broadcast update
            let _ = self.broadcast_tx.send(aggregated);
        }
        
        Ok(())
    }
    
    /// Calculate aggregated progress
    async fn calculate_aggregated_progress(
        &self,
        parent_id: &str,
        subtask_ids: &[String],
    ) -> Result<AggregatedProgress> {
        let tracker = self.progress_tracker.read().await;
        
        let mut subtask_details = Vec::new();
        let mut total_progress = 0.0;
        let mut completed_count = 0;
        let mut in_progress_count = 0;
        let mut failed_count = 0;
        let mut weights = Vec::new();
        
        for subtask_id in subtask_ids {
            if let Some(progress) = tracker.task_progress.get(subtask_id) {
                subtask_details.push(SubtaskProgress {
                    task_id: subtask_id.clone(),
                    description: self.get_task_description(subtask_id).await.unwrap_or_else(|| format!("Subtask {}", subtask_id)),
                    progress: progress.progress,
                    status: progress.status.clone(),
                    agent_id: progress.agent_id.clone(),
                });
                
                match &progress.status {
                    TaskStatus::Done => completed_count += 1,
                    TaskStatus::InProgress => in_progress_count += 1,
                    TaskStatus::Cancelled => failed_count += 1,
                    _ => {}
                }
                
                // Calculate weight based on method
                let weight = match self.config.calculation_method {
                    ProgressCalculationMethod::SimpleAverage => 1.0,
                    ProgressCalculationMethod::WeightedAverage => {
                        // Get effort from task if available, otherwise use default weight
                        self.get_task_effort_weight(subtask_id).await.unwrap_or(1.0)
                    }
                    _ => 1.0,
                };
                
                weights.push(weight);
                total_progress += progress.progress * weight;
            }
        }
        
        // Calculate overall progress
        let total_weight: f32 = weights.iter().sum();
        let overall_progress = if total_weight > 0.0 {
            match self.config.calculation_method {
                ProgressCalculationMethod::MinimumProgress => {
                    subtask_details.iter()
                        .map(|s| s.progress)
                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                        .unwrap_or(0.0)
                }
                _ => total_progress / total_weight,
            }
        } else {
            0.0
        };
        
        // Determine overall status
        let status = if failed_count > 0 && failed_count == subtask_ids.len() {
            TaskStatus::Cancelled
        } else if completed_count == subtask_ids.len() {
            TaskStatus::Done
        } else if in_progress_count > 0 || completed_count > 0 {
            TaskStatus::InProgress
        } else {
            TaskStatus::Todo
        };
        
        // Estimate completion time
        let estimated_completion = self.estimate_completion_time(&tracker, subtask_ids);
        
        Ok(AggregatedProgress {
            parent_task_id: parent_id.to_string(),
            overall_progress,
            subtask_count: subtask_ids.len(),
            completed_count,
            in_progress_count,
            failed_count,
            status,
            estimated_completion,
            subtask_details,
        })
    }
    
    /// Estimate completion time based on current progress
    fn estimate_completion_time(
        &self,
        tracker: &ProgressTracker,
        subtask_ids: &[String],
    ) -> Option<Instant> {
        let mut max_remaining = Duration::from_secs(0);
        
        for subtask_id in subtask_ids {
            if let Some(progress) = tracker.task_progress.get(subtask_id) {
                if progress.progress < 1.0 && progress.started_at.is_some() {
                    let elapsed = progress.updated_at.duration_since(progress.started_at.unwrap());
                    let rate = progress.progress / elapsed.as_secs_f32();
                    
                    if rate > 0.0 {
                        let remaining = (1.0 - progress.progress) / rate;
                        let remaining_duration = Duration::from_secs_f32(remaining);
                        max_remaining = max_remaining.max(remaining_duration);
                    }
                }
            }
        }
        
        if max_remaining > Duration::from_secs(0) {
            Some(Instant::now() + max_remaining)
        } else {
            None
        }
    }
    
    /// Update parent task in task manager
    async fn update_parent_task(&self, parent_id: &str, aggregated: &AggregatedProgress) -> Result<()> {
        // Get the parent task from task manager
        let tasks = self.task_manager.get_tasks().await;
        
        if let Some(parent_task) = tasks.get(parent_id) {
            debug!(
                "Parent task {} progress updated: {:.1}% (status: {:?})",
                parent_id,
                aggregated.overall_progress * 100.0,
                aggregated.status
            );
            
            // Log aggregated progress details
            info!(
                "Task '{}' progress: {:.1}% | Subtasks: {} total, {} completed, {} in progress, {} failed",
                parent_task.title,
                aggregated.overall_progress * 100.0,
                aggregated.subtask_count,
                aggregated.completed_count,
                aggregated.in_progress_count,
                aggregated.failed_count
            );
            
            // Store progress information in memory for cognitive awareness
            if aggregated.overall_progress >= 1.0 || aggregated.status == TaskStatus::Done {
                info!("Task '{}' completed!", parent_task.title);
            }
            
            // Note: Since TaskManager doesn't expose a public update method,
            // we rely on the task manager's own synchronization mechanisms
            // and focus on broadcasting progress updates for UI consumption
        } else {
            warn!("Parent task {} not found in task manager", parent_id);
        }
        
        Ok(())
    }
    
    /// Periodic aggregation task
    async fn periodic_aggregation(&self) {
        let mut interval = tokio::time::interval(self.config.update_interval);
        
        loop {
            interval.tick().await;
            
            let hierarchy = self.task_hierarchy.read().await;
            let parent_ids: Vec<String> = hierarchy.parent_to_subtasks.keys().cloned().collect();
            drop(hierarchy);
            
            for parent_id in parent_ids {
                if let Err(e) = self.aggregate_parent_progress_by_parent(&parent_id).await {
                    warn!("Failed to aggregate progress for {}: {}", parent_id, e);
                }
            }
        }
    }
    
    /// Aggregate progress by parent ID
    async fn aggregate_parent_progress_by_parent(&self, parent_id: &str) -> Result<()> {
        let hierarchy = self.task_hierarchy.read().await;
        let subtask_ids = hierarchy.parent_to_subtasks.get(parent_id)
            .cloned()
            .unwrap_or_default();
        drop(hierarchy);
        
        if !subtask_ids.is_empty() {
            let aggregated = self.calculate_aggregated_progress(parent_id, &subtask_ids).await?;
            
            let mut tracker = self.progress_tracker.write().await;
            tracker.parent_progress.insert(parent_id.to_string(), aggregated.clone());
            drop(tracker);
            
            let _ = self.broadcast_tx.send(aggregated);
        }
        
        Ok(())
    }
    
    /// Monitor agent streams for updates
    async fn monitor_agent_streams(&self) {
        info!("Starting agent stream monitoring for task progress");
        
        // Create a channel to receive agent stream updates
        // Note: AgentStreamManager doesn't expose update_rx directly,
        // so we need to poll for status changes periodically
        let agent_stream_manager = self.agent_stream_manager.clone();
        let update_tx = self.update_tx.clone();
        let task_hierarchy = self.task_hierarchy.clone();
        let mut interval = tokio::time::interval(Duration::from_secs(2));
        
        loop {
            interval.tick().await;
            
            // Check all agent streams for status updates
            let hierarchy = task_hierarchy.read().await;
            let agent_task_pairs: Vec<(String, String)> = hierarchy.task_to_agent.iter()
                .map(|(t, a)| (t.clone(), a.clone()))
                .collect();
            drop(hierarchy);
            
            for (task_id, agent_id) in agent_task_pairs {
                // Get agent stream status
                if let Some(stream) = agent_stream_manager.get_stream(&agent_id).await {
                    // Check for status changes
                    match stream.status {
                        AgentStatus::Completed => {
                            let _ = update_tx.send(ProgressUpdate::TaskCompleted {
                                task_id: task_id.clone(),
                                success: true,
                            }).await;
                        }
                        AgentStatus::Failed => {
                            let _ = update_tx.send(ProgressUpdate::TaskCompleted {
                                task_id: task_id.clone(),
                                success: false,
                            }).await;
                        }
                        AgentStatus::Running => {
                            // Check latest message for progress hints
                            if let Some(message) = stream.messages.back() {
                                if let Some(progress) = Self::extract_progress_from_message(&message.content) {
                                    let _ = update_tx.send(ProgressUpdate::TaskProgress {
                                        task_id: task_id.clone(),
                                        progress,
                                        status: TaskStatus::InProgress,
                                        message: Some(message.content.clone()),
                                    }).await;
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }
    
    /// Get current aggregated progress for a parent task
    pub async fn get_aggregated_progress(&self, parent_task_id: &str) -> Option<AggregatedProgress> {
        let tracker = self.progress_tracker.read().await;
        tracker.parent_progress.get(parent_task_id).cloned()
    }
    
    /// Subscribe to progress updates
    pub fn subscribe(&self) -> broadcast::Receiver<AggregatedProgress> {
        self.broadcast_tx.subscribe()
    }
    
    /// Send a progress update
    pub async fn send_progress_update(
        &self,
        task_id: String,
        progress: f32,
        status: TaskStatus,
        message: Option<String>,
    ) -> Result<()> {
        self.update_tx.send(ProgressUpdate::TaskProgress {
            task_id,
            progress,
            status,
            message,
        }).await
        .map_err(|e| anyhow!("Failed to send progress update: {}", e))
    }
    
    /// Get task description from task manager
    async fn get_task_description(&self, task_id: &str) -> Option<String> {
        let tasks = self.task_manager.get_tasks().await;
        tasks.get(task_id).map(|task| task.title.clone())
    }
    
    /// Get task effort weight based on estimated effort
    async fn get_task_effort_weight(&self, task_id: &str) -> Option<f32> {
        let tasks = self.task_manager.get_tasks().await;
        tasks.get(task_id).and_then(|task| {
            task.estimate.map(|duration| {
                // Convert duration to hours as weight
                // Tasks with longer estimates get higher weight
                (duration.as_secs_f32() / 3600.0).max(0.5)
            })
        })
    }
    
    /// Extract progress percentage from agent message
    fn extract_progress_from_message(content: &str) -> Option<f32> {
        // Look for common progress patterns in messages
        // Examples: "50% complete", "progress: 0.5", "completed 3/6", etc.
        
        // Pattern 1: "X% complete" or "X% done"
        if let Some(caps) = regex::Regex::new(r"(\d+)%\s*(complete|done|finished|progress)")
            .ok()
            .and_then(|re| re.captures(content)) {
            if let Some(percent) = caps.get(1).and_then(|m| m.as_str().parse::<f32>().ok()) {
                return Some(percent / 100.0);
            }
        }
        
        // Pattern 2: "progress: X" where X is 0.0 to 1.0
        if let Some(caps) = regex::Regex::new(r"progress:\s*(\d*\.?\d+)")
            .ok()
            .and_then(|re| re.captures(content)) {
            if let Some(progress) = caps.get(1).and_then(|m| m.as_str().parse::<f32>().ok()) {
                if progress >= 0.0 && progress <= 1.0 {
                    return Some(progress);
                }
            }
        }
        
        // Pattern 3: "completed X/Y" or "X of Y done"
        if let Some(caps) = regex::Regex::new(r"(\d+)\s*(?:/|of)\s*(\d+)")
            .ok()
            .and_then(|re| re.captures(content)) {
            if let (Some(completed), Some(total)) = (
                caps.get(1).and_then(|m| m.as_str().parse::<f32>().ok()),
                caps.get(2).and_then(|m| m.as_str().parse::<f32>().ok())
            ) {
                if total > 0.0 {
                    return Some(completed / total);
                }
            }
        }
        
        None
    }
}

// Implement Clone for TaskProgressAggregator
impl Clone for TaskProgressAggregator {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            task_manager: self.task_manager.clone(),
            agent_stream_manager: self.agent_stream_manager.clone(),
            task_hierarchy: self.task_hierarchy.clone(),
            progress_tracker: self.progress_tracker.clone(),
            update_tx: self.update_tx.clone(),
            update_rx: self.update_rx.clone(),
            broadcast_tx: self.broadcast_tx.clone(),
        }
    }
}