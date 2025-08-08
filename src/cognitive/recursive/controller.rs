//! Recursion Controller for Recursive Cognitive Processing
//!
//! Manages recursive depth, resource limits, and control flow to ensure
//! safe and efficient recursive processing without infinite loops.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, RwLock};

use super::{
    RecursionDepth,
    RecursionType,
    RecursiveConfig,
    RecursiveContext,
    RecursiveThoughtId,
    ResourceUsage,
    TerminationReason,
};
use crate::memory::fractal::ScaleLevel;

/// Limits for recursive processing
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RecursionLimits {
    /// Maximum recursion depth
    pub max_depth: u32,

    /// Maximum total execution time
    pub max_total_time: Duration,

    /// Maximum time per recursion step
    pub max_step_time: Duration,

    /// Maximum memory usage (bytes)
    pub max_memory: u64,

    /// Maximum CPU cycles
    pub max_cpu_cycles: u64,

    /// Maximum number of active recursions
    pub max_concurrent_recursions: u32,

    /// Stack overflow detection threshold
    pub stack_overflow_threshold: u32,

    /// Infinite loop detection parameters
    pub loop_detection: LoopDetectionParams,

    /// Resource monitoring frequency
    pub monitoring_frequency: Duration,
}

impl Default for RecursionLimits {
    fn default() -> Self {
        Self {
            max_depth: 50,
            max_total_time: Duration::from_secs(300), // 5 minutes
            max_step_time: Duration::from_secs(30),   // 30 seconds
            max_memory: 1024 * 1024 * 1024,           // 1GB
            max_cpu_cycles: 10_000_000_000,           // 10B cycles
            max_concurrent_recursions: 10,
            stack_overflow_threshold: 1000,
            loop_detection: LoopDetectionParams::default(),
            monitoring_frequency: Duration::from_millis(100),
        }
    }
}

impl RecursionLimits {
    /// Create limits from configuration
    pub fn fromconfig(config: &RecursiveConfig) -> Self {
        Self {
            max_depth: config.max_depth,
            max_total_time: Duration::from_secs(300),
            max_step_time: Duration::from_secs(30),
            max_memory: config.resource_limits.max_memory,
            max_cpu_cycles: config.resource_limits.max_compute,
            max_concurrent_recursions: 10,
            stack_overflow_threshold: config.max_depth * 2,
            loop_detection: LoopDetectionParams::default(),
            monitoring_frequency: Duration::from_millis(100),
        }
    }

    /// Create strict limits for safety-critical contexts
    pub fn strict() -> Self {
        Self {
            max_depth: 10,
            max_total_time: Duration::from_secs(60),
            max_step_time: Duration::from_secs(5),
            max_memory: 100 * 1024 * 1024, // 100MB
            max_cpu_cycles: 1_000_000_000, // 1B cycles
            max_concurrent_recursions: 3,
            stack_overflow_threshold: 20,
            loop_detection: LoopDetectionParams::strict(),
            monitoring_frequency: Duration::from_millis(50),
        }
    }

    /// Create permissive limits for research contexts
    pub fn permissive() -> Self {
        Self {
            max_depth: 100,
            max_total_time: Duration::from_secs(1800), // 30 minutes
            max_step_time: Duration::from_secs(120),   // 2 minutes
            max_memory: 8 * 1024 * 1024 * 1024,        // 8GB
            max_cpu_cycles: 100_000_000_000,           // 100B cycles
            max_concurrent_recursions: 50,
            stack_overflow_threshold: 200,
            loop_detection: LoopDetectionParams::permissive(),
            monitoring_frequency: Duration::from_millis(200),
        }
    }
}

/// Parameters for infinite loop detection
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LoopDetectionParams {
    /// Enable loop detection
    pub enabled: bool,

    /// Minimum cycle length to detect
    pub min_cycle_length: u32,

    /// Maximum cycle length to detect
    pub max_cycle_length: u32,

    /// Similarity threshold for detecting repeated states
    pub similarity_threshold: f64,

    /// Number of repetitions before flagging as loop
    pub repetition_threshold: u32,

    /// History window size for loop detection
    pub history_window: u32,
}

impl Default for LoopDetectionParams {
    fn default() -> Self {
        Self {
            enabled: true,
            min_cycle_length: 2,
            max_cycle_length: 20,
            similarity_threshold: 0.95,
            repetition_threshold: 3,
            history_window: 100,
        }
    }
}

impl LoopDetectionParams {
    /// Strict loop detection for safety
    pub fn strict() -> Self {
        Self {
            enabled: true,
            min_cycle_length: 1,
            max_cycle_length: 10,
            similarity_threshold: 0.9,
            repetition_threshold: 2,
            history_window: 50,
        }
    }

    /// Permissive loop detection for exploration
    pub fn permissive() -> Self {
        Self {
            enabled: true,
            min_cycle_length: 3,
            max_cycle_length: 50,
            similarity_threshold: 0.98,
            repetition_threshold: 5,
            history_window: 200,
        }
    }
}

/// Current state of recursion control
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RecursionState {
    /// Current recursion depth
    pub current_depth: RecursionDepth,

    /// Number of active recursions
    pub active_recursions: u32,

    /// Total resource usage
    pub total_resource_usage: ResourceUsage,

    /// Start time of current recursion session
    pub session_start: DateTime<Utc>,

    /// Current recursion path
    pub recursion_path: Vec<RecursionPathNode>,

    /// State history for loop detection
    pub state_history: VecDeque<StateSnapshot>,

    /// Current safety level
    pub safety_level: SafetyLevel,

    /// Warnings and alerts
    pub warnings: Vec<RecursionWarning>,
}

impl Default for RecursionState {
    fn default() -> Self {
        Self {
            current_depth: RecursionDepth::zero(),
            active_recursions: 0,
            total_resource_usage: ResourceUsage::default(),
            session_start: Utc::now(),
            recursion_path: Vec::new(),
            state_history: VecDeque::new(),
            safety_level: SafetyLevel::Normal,
            warnings: Vec::new(),
        }
    }
}

/// Node in the recursion path
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RecursionPathNode {
    /// Thought ID at this level
    pub thought_id: RecursiveThoughtId,

    /// Depth level
    pub depth: RecursionDepth,

    /// Scale level
    pub scale_level: ScaleLevel,

    /// Recursion type
    pub recursion_type: RecursionType,

    /// Entry timestamp
    pub entry_time: DateTime<Utc>,

    /// Resource state at entry
    pub entry_resources: ResourceUsage,

    /// Input hash for loop detection
    pub input_hash: String,
}

/// Snapshot of state for loop detection
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StateSnapshot {
    /// Timestamp of snapshot
    pub timestamp: DateTime<Utc>,

    /// Current depth
    pub depth: RecursionDepth,

    /// State hash
    pub state_hash: String,

    /// Resource usage at this point
    pub resources: ResourceUsage,

    /// Input content hash
    pub input_hash: String,

    /// Output content hash
    pub output_hash: Option<String>,
}

/// Safety levels for recursion
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum SafetyLevel {
    /// Normal operation
    Normal,

    /// Caution advised
    Caution,

    /// Warning state
    Warning,

    /// Critical state - immediate action required
    Critical,

    /// Emergency stop required
    Emergency,
}

/// Types of recursion warnings
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RecursionWarning {
    /// Warning type
    pub warning_type: WarningType,

    /// Warning message
    pub message: String,

    /// Severity level
    pub severity: WarningSeverity,

    /// Timestamp when warning was issued
    pub timestamp: DateTime<Utc>,

    /// Associated recursion depth
    pub depth: RecursionDepth,

    /// Suggested action
    pub suggested_action: SuggestedAction,
}

/// Types of warnings
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum WarningType {
    /// Approaching depth limit
    DepthLimit,

    /// Approaching time limit
    TimeLimit,

    /// High resource usage
    ResourceUsage,

    /// Potential infinite loop detected
    PotentialLoop,

    /// Stack overflow risk
    StackOverflow,

    /// Performance degradation
    PerformanceDegradation,

    /// Quality deterioration
    QualityDegradation,
}

/// Warning severity levels
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, PartialOrd)]
pub enum WarningSeverity {
    /// Informational
    Info,

    /// Low severity
    Low,

    /// Medium severity
    Medium,

    /// High severity
    High,

    /// Critical severity
    Critical,
}

/// Suggested actions for warnings
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum SuggestedAction {
    /// Continue with monitoring
    Monitor,

    /// Reduce recursion depth
    ReduceDepth,

    /// Increase resource limits
    IncreaseResources,

    /// Optimize processing
    Optimize,

    /// Pause recursion
    Pause,

    /// Terminate recursion
    Terminate,

    /// Emergency stop
    EmergencyStop,
}

/// Statistics for recursion control
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RecursionStatistics {
    /// Total recursions executed
    pub total_recursions: u64,

    /// Successful recursions
    pub successful_recursions: u64,

    /// Failed recursions
    pub failed_recursions: u64,

    /// Recursions terminated due to limits
    pub limit_terminated: u64,

    /// Average recursion depth
    pub average_depth: f64,

    /// Maximum depth reached
    pub max_depth_reached: u32,

    /// Average execution time
    pub average_execution_time: Duration,

    /// Total resource usage
    pub total_resource_usage: ResourceUsage,

    /// Loop detections
    pub loop_detections: u64,

    /// Warning counts by type
    pub warning_counts: HashMap<WarningType, u64>,
}

impl Default for RecursionStatistics {
    fn default() -> Self {
        Self {
            total_recursions: 0,
            successful_recursions: 0,
            failed_recursions: 0,
            limit_terminated: 0,
            average_depth: 0.0,
            max_depth_reached: 0,
            average_execution_time: Duration::from_secs(0),
            total_resource_usage: ResourceUsage::default(),
            loop_detections: 0,
            warning_counts: HashMap::new(),
        }
    }
}

/// Main recursion controller
pub struct RecursionController {
    /// Recursion limits
    limits: RecursionLimits,

    /// Current state
    state: Arc<RwLock<RecursionState>>,

    /// Statistics
    statistics: Arc<RwLock<RecursionStatistics>>,

    /// Active recursion tracking
    active_recursions: Arc<RwLock<HashMap<RecursiveThoughtId, RecursionSession>>>,

    /// Loop detector
    loop_detector: Arc<Mutex<LoopDetector>>,

    /// Resource monitor
    resource_monitor: Arc<Mutex<ResourceMonitor>>,

    /// Emergency stop flag
    emergency_stop: Arc<RwLock<bool>>,
}

/// Individual recursion session tracking
#[derive(Clone, Debug)]
pub struct RecursionSession {
    /// Session ID
    pub id: RecursiveThoughtId,

    /// Start time
    pub start_time: Instant,

    /// Current depth
    pub current_depth: RecursionDepth,

    /// Resource usage for this session
    pub resource_usage: ResourceUsage,

    /// Context
    pub context: RecursiveContext,

    /// Session-specific limits
    pub session_limits: Option<RecursionLimits>,
}

/// Loop detection system
pub struct LoopDetector {
    /// Detection parameters
    params: LoopDetectionParams,

    /// State history buffer
    state_buffer: VecDeque<StateSnapshot>,

    /// Detected loops
    detected_loops: Vec<DetectedLoop>,

    /// Last detection time
    last_detection: Option<Instant>,
}

/// Detected loop information
#[derive(Clone, Debug)]
pub struct DetectedLoop {
    /// Loop start position in history
    pub start_position: usize,

    /// Loop length
    pub length: u32,

    /// Similarity score
    pub similarity: f64,

    /// Detection time
    pub detection_time: Instant,

    /// Loop pattern
    pub pattern: Vec<String>,
}

/// Resource monitoring system
pub struct ResourceMonitor {
    /// Current resource usage
    pub current_usage: ResourceUsage,

    /// Resource usage history
    pub usage_history: VecDeque<(Instant, ResourceUsage)>,

    /// Peak usage
    pub peak_usage: ResourceUsage,

    /// Last monitoring time
    pub last_monitor: Instant,
}

impl RecursionController {
    /// Create a new recursion controller
    pub async fn new(limits: RecursionLimits) -> Result<Self> {
        let loop_detector = LoopDetector {
            params: limits.loop_detection.clone(),
            state_buffer: VecDeque::with_capacity(limits.loop_detection.history_window as usize),
            detected_loops: Vec::new(),
            last_detection: None,
        };

        let resource_monitor = ResourceMonitor {
            current_usage: ResourceUsage::default(),
            usage_history: VecDeque::new(),
            peak_usage: ResourceUsage::default(),
            last_monitor: Instant::now(),
        };

        Ok(Self {
            limits,
            state: Arc::new(RwLock::new(RecursionState::default())),
            statistics: Arc::new(RwLock::new(RecursionStatistics::default())),
            active_recursions: Arc::new(RwLock::new(HashMap::new())),
            loop_detector: Arc::new(Mutex::new(loop_detector)),
            resource_monitor: Arc::new(Mutex::new(resource_monitor)),
            emergency_stop: Arc::new(RwLock::new(false)),
        })
    }

    /// Check if recursion can proceed
    pub async fn can_recurse(&self, context: &RecursiveContext) -> Result<bool> {
        // Check emergency stop
        if *self.emergency_stop.read().await {
            return Ok(false);
        }

        let state = self.state.read().await;

        // Check depth limits
        if context.depth.as_u32() >= self.limits.max_depth {
            return Ok(false);
        }

        // Check concurrent recursion limits
        if state.active_recursions >= self.limits.max_concurrent_recursions {
            return Ok(false);
        }

        // Check time limits
        let elapsed = Utc::now().signed_duration_since(state.session_start);
        if elapsed.to_std().unwrap_or(Duration::from_secs(0)) >= self.limits.max_total_time {
            return Ok(false);
        }

        // Check resource limits
        if state.total_resource_usage.memory_bytes >= self.limits.max_memory {
            return Ok(false);
        }

        if state.total_resource_usage.compute_cycles >= self.limits.max_cpu_cycles {
            return Ok(false);
        }

        // Check safety level
        if state.safety_level >= SafetyLevel::Critical {
            return Ok(false);
        }

        Ok(true)
    }

    /// Start a new recursion session
    pub async fn start_recursion(
        &self,
        thought_id: RecursiveThoughtId,
        context: RecursiveContext,
    ) -> Result<()> {
        // Check if recursion can proceed
        if !self.can_recurse(&context).await? {
            return Err(anyhow::anyhow!("Recursion not allowed due to limits"));
        }

        let session = RecursionSession {
            id: thought_id.clone(),
            start_time: Instant::now(),
            current_depth: context.depth,
            resource_usage: ResourceUsage::default(),
            context: context.clone(),
            session_limits: None,
        };

        // Add to active sessions
        {
            let mut active = self.active_recursions.write().await;
            active.insert(thought_id.clone(), session);
        }

        // Update state
        {
            let mut state = self.state.write().await;
            state.active_recursions += 1;
            state.current_depth = context.depth;

            // Add to recursion path
            state.recursion_path.push(RecursionPathNode {
                thought_id,
                depth: context.depth,
                scale_level: context.scale_level,
                recursion_type: context.recursion_type,
                entry_time: Utc::now(),
                entry_resources: context.resource_usage.clone(),
                input_hash: self.compute_hash("input"), // Simplified
            });
        }

        // Update statistics
        {
            let mut stats = self.statistics.write().await;
            stats.total_recursions += 1;
            if context.depth.as_u32() > stats.max_depth_reached {
                stats.max_depth_reached = context.depth.as_u32();
            }
        }

        Ok(())
    }

    /// End a recursion session
    pub async fn end_recursion(
        &self,
        thought_id: &RecursiveThoughtId,
        success: bool,
        termination_reason: TerminationReason,
    ) -> Result<()> {
        // Remove from active sessions
        let session = {
            let mut active = self.active_recursions.write().await;
            active.remove(thought_id)
        };

        if let Some(session) = session {
            let execution_time = session.start_time.elapsed();

            // Update state
            {
                let mut state = self.state.write().await;
                state.active_recursions = state.active_recursions.saturating_sub(1);

                // Remove from recursion path
                state.recursion_path.retain(|node| node.thought_id != *thought_id);

                // Update resource usage
                state.total_resource_usage.compute_cycles += session.resource_usage.compute_cycles;
                state.total_resource_usage.memory_bytes = state
                    .total_resource_usage
                    .memory_bytes
                    .saturating_sub(session.resource_usage.memory_bytes);
            }

            // Update statistics
            {
                let mut stats = self.statistics.write().await;
                if success {
                    stats.successful_recursions += 1;
                } else {
                    stats.failed_recursions += 1;
                }

                if matches!(
                    termination_reason,
                    TerminationReason::DepthLimit
                        | TerminationReason::TimeLimit
                        | TerminationReason::ResourceLimit
                ) {
                    stats.limit_terminated += 1;
                }

                // Update averages
                let total = stats.total_recursions as f64;
                stats.average_depth = (stats.average_depth * (total - 1.0)
                    + session.current_depth.as_u32() as f64)
                    / total;

                let old_avg_ms = stats.average_execution_time.as_millis() as f64;
                let new_time_ms = execution_time.as_millis() as f64;
                let new_avg_ms = (old_avg_ms * (total - 1.0) + new_time_ms) / total;
                stats.average_execution_time = Duration::from_millis(new_avg_ms as u64);
            }
        }

        Ok(())
    }

    /// Monitor and update recursion state
    pub async fn monitor(&self) -> Result<()> {
        // Update resource monitoring
        self.update_resource_monitoring().await?;

        // Check for loops
        self.check_for_loops().await?;

        // Update safety level
        self.update_safety_level().await?;

        // Generate warnings if necessary
        self.generate_warnings().await?;

        Ok(())
    }

    /// Update resource monitoring
    async fn update_resource_monitoring(&self) -> Result<()> {
        let mut monitor = self.resource_monitor.lock().await;
        let now = Instant::now();

        // Calculate current resource usage from active sessions
        let active = self.active_recursions.read().await;
        let mut total_usage = ResourceUsage::default();

        for session in active.values() {
            total_usage.memory_bytes += session.resource_usage.memory_bytes;
            total_usage.compute_cycles += session.resource_usage.compute_cycles;
            total_usage.network_calls += session.resource_usage.network_calls;
            total_usage.energy_cost += session.resource_usage.energy_cost;
        }

        monitor.current_usage = total_usage.clone();

        // Update peak usage
        if total_usage.memory_bytes > monitor.peak_usage.memory_bytes {
            monitor.peak_usage.memory_bytes = total_usage.memory_bytes;
        }
        if total_usage.compute_cycles > monitor.peak_usage.compute_cycles {
            monitor.peak_usage.compute_cycles = total_usage.compute_cycles;
        }

        // Add to history
        monitor.usage_history.push_back((now, total_usage));

        // Keep history bounded
        while monitor.usage_history.len() > 1000 {
            monitor.usage_history.pop_front();
        }

        monitor.last_monitor = now;

        Ok(())
    }

    /// Check for infinite loops
    async fn check_for_loops(&self) -> Result<()> {
        let mut detector = self.loop_detector.lock().await;

        if !detector.params.enabled {
            return Ok(());
        }

        // Simple loop detection based on state similarity
        if detector.state_buffer.len() >= detector.params.min_cycle_length as usize * 2 {
            for cycle_len in detector.params.min_cycle_length..=detector.params.max_cycle_length {
                if detector.state_buffer.len() < cycle_len as usize * 2 {
                    continue;
                }

                let recent_states: Vec<_> =
                    detector.state_buffer.iter().rev().take(cycle_len as usize * 2).collect();

                // Check for repeating pattern
                let mut similarity_count = 0;
                for i in 0..cycle_len as usize {
                    let state1 = &recent_states[i];
                    let state2 = &recent_states[i + cycle_len as usize];

                    let similarity = self.calculate_state_similarity(state1, state2);
                    if similarity >= detector.params.similarity_threshold {
                        similarity_count += 1;
                    }
                }

                if similarity_count >= detector.params.repetition_threshold {
                    // Loop detected
                    let detected_loop = DetectedLoop {
                        start_position: detector.state_buffer.len() - cycle_len as usize * 2,
                        length: cycle_len,
                        similarity: similarity_count as f64 / cycle_len as f64,
                        detection_time: Instant::now(),
                        pattern: recent_states.iter().map(|s| s.state_hash.clone()).collect(),
                    };

                    detector.detected_loops.push(detected_loop);
                    detector.last_detection = Some(Instant::now());

                    // Update statistics
                    {
                        let mut stats = self.statistics.write().await;
                        stats.loop_detections += 1;
                    }

                    // Generate warning
                    self.add_warning(RecursionWarning {
                        warning_type: WarningType::PotentialLoop,
                        message: format!(
                            "Potential infinite loop detected with cycle length {}",
                            cycle_len
                        ),
                        severity: WarningSeverity::High,
                        timestamp: Utc::now(),
                        depth: RecursionDepth(cycle_len),
                        suggested_action: SuggestedAction::Terminate,
                    })
                    .await?;

                    break;
                }
            }
        }

        Ok(())
    }

    /// Update safety level based on current state
    async fn update_safety_level(&self) -> Result<()> {
        let mut state = self.state.write().await;
        let monitor = self.resource_monitor.lock().await;

        let mut safety_level = SafetyLevel::Normal;

        // Check depth
        let depth_ratio = state.current_depth.as_u32() as f64 / self.limits.max_depth as f64;
        if depth_ratio > 0.9 {
            safety_level = SafetyLevel::Critical;
        } else if depth_ratio > 0.8 {
            safety_level = SafetyLevel::Warning;
        } else if depth_ratio > 0.7 {
            safety_level = SafetyLevel::Caution;
        }

        // Check resource usage
        let memory_ratio =
            monitor.current_usage.memory_bytes as f64 / self.limits.max_memory as f64;
        if memory_ratio > 0.9 {
            safety_level = safety_level.max(SafetyLevel::Critical);
        } else if memory_ratio > 0.8 {
            safety_level = safety_level.max(SafetyLevel::Warning);
        }

        // Check for recent loop detections
        let detector = self.loop_detector.lock().await;
        if let Some(last_detection) = detector.last_detection {
            if last_detection.elapsed() < Duration::from_secs(30) {
                safety_level = safety_level.max(SafetyLevel::Warning);
            }
        }

        state.safety_level = safety_level;

        Ok(())
    }

    /// Generate warnings based on current state
    async fn generate_warnings(&self) -> Result<()> {
        let state = self.state.read().await;
        let monitor = self.resource_monitor.lock().await;

        let mut new_warnings = Vec::new();

        // Depth warnings
        let depth_ratio = state.current_depth.as_u32() as f64 / self.limits.max_depth as f64;
        if depth_ratio > 0.8 {
            new_warnings.push(RecursionWarning {
                warning_type: WarningType::DepthLimit,
                message: format!(
                    "Approaching depth limit: {}/{}",
                    state.current_depth.as_u32(),
                    self.limits.max_depth
                ),
                severity: if depth_ratio > 0.9 {
                    WarningSeverity::Critical
                } else {
                    WarningSeverity::High
                },
                timestamp: Utc::now(),
                depth: state.current_depth,
                suggested_action: if depth_ratio > 0.9 {
                    SuggestedAction::Terminate
                } else {
                    SuggestedAction::ReduceDepth
                },
            });
        }

        // Memory warnings
        let memory_ratio =
            monitor.current_usage.memory_bytes as f64 / self.limits.max_memory as f64;
        if memory_ratio > 0.8 {
            new_warnings.push(RecursionWarning {
                warning_type: WarningType::ResourceUsage,
                message: format!("High memory usage: {:.1}%", memory_ratio * 100.0),
                severity: if memory_ratio > 0.9 {
                    WarningSeverity::Critical
                } else {
                    WarningSeverity::High
                },
                timestamp: Utc::now(),
                depth: state.current_depth,
                suggested_action: SuggestedAction::Optimize,
            });
        }

        // Add warnings to state
        drop(state);
        {
            let mut state = self.state.write().await;
            state.warnings.extend(new_warnings.clone());

            // Keep warnings bounded
            let warnings_len = state.warnings.len();
            if warnings_len > 100 {
                state.warnings.drain(0..warnings_len - 100);
            }
        }

        // Update warning statistics
        {
            let mut warning_stats = self.statistics.write().await;
            for warning in new_warnings {
                *warning_stats.warning_counts.entry(warning.warning_type).or_insert(0) += 1;
            }
        }

        Ok(())
    }

    /// Add a warning to the current state
    async fn add_warning(&self, warning: RecursionWarning) -> Result<()> {
        let mut state = self.state.write().await;
        state.warnings.push(warning.clone());

        // Update statistics
        {
            let mut warning_statistics = self.statistics.write().await;
            *warning_statistics.warning_counts.entry(warning.warning_type).or_insert(0) += 1;
        }

        Ok(())
    }

    /// Calculate similarity between two states
    fn calculate_state_similarity(&self, state1: &StateSnapshot, state2: &StateSnapshot) -> f64 {
        // Simple similarity based on hash comparison
        if state1.state_hash == state2.state_hash {
            1.0
        } else if state1.input_hash == state2.input_hash {
            0.8
        } else {
            0.0
        }
    }

    /// Compute hash for content (simplified)
    fn compute_hash(&self, content: &str) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        format!("{:?}", hasher.finalize())[..16].to_string()
    }

    /// Emergency stop all recursions
    pub async fn emergency_stop(&self) -> Result<()> {
        {
            let mut stop = self.emergency_stop.write().await;
            *stop = true;
        }

        // Clear all active recursions
        {
            let mut active = self.active_recursions.write().await;
            active.clear();
        }

        // Update state
        let current_depth = {
            let state = self.state.read().await;
            state.current_depth
        };

        {
            let mut state = self.state.write().await;
            state.active_recursions = 0;
            state.safety_level = SafetyLevel::Emergency;
            state.warnings.push(RecursionWarning {
                warning_type: WarningType::StackOverflow,
                message: "Emergency stop activated".to_string(),
                severity: WarningSeverity::Critical,
                timestamp: Utc::now(),
                depth: current_depth,
                suggested_action: SuggestedAction::EmergencyStop,
            });
        }

        Ok(())
    }

    /// Reset emergency stop
    pub async fn reset_emergency_stop(&self) -> Result<()> {
        let mut stop = self.emergency_stop.write().await;
        *stop = false;

        {
            let mut state = self.state.write().await;
            state.safety_level = SafetyLevel::Normal;
        }

        Ok(())
    }

    /// Get current recursion state
    pub async fn get_state(&self) -> RecursionState {
        self.state.read().await.clone()
    }

    /// Get current statistics
    pub async fn get_statistics(&self) -> RecursionStatistics {
        self.statistics.read().await.clone()
    }

    /// Update limits
    pub async fn update_limits(&mut self, new_limits: RecursionLimits) {
        self.limits = new_limits.clone();

        // Update loop detector parameters
        {
            let mut detector = self.loop_detector.lock().await;
            detector.params = new_limits.loop_detection;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_recursion_controller_creation() {
        let limits = RecursionLimits::default();
        let controller = RecursionController::new(limits).await.unwrap();

        let state = controller.get_state().await;
        assert_eq!(state.active_recursions, 0);
        assert_eq!(state.current_depth.as_u32(), 0);
    }

    #[tokio::test]
    async fn test_recursion_limits() {
        let mut context = RecursiveContext::default();
        context.depth = RecursionDepth(5);

        let limits = RecursionLimits::strict();
        let controller = RecursionController::new(limits).await.unwrap();

        assert!(controller.can_recurse(&context).await.unwrap());

        // Test depth limit
        context.depth = RecursionDepth(15);
        assert!(!controller.can_recurse(&context).await.unwrap());
    }

    #[tokio::test]
    async fn test_emergency_stop() {
        let limits = RecursionLimits::default();
        let controller = RecursionController::new(limits).await.unwrap();

        let context = RecursiveContext::default();
        assert!(controller.can_recurse(&context).await.unwrap());

        controller.emergency_stop().await.unwrap();
        assert!(!controller.can_recurse(&context).await.unwrap());

        controller.reset_emergency_stop().await.unwrap();
        assert!(controller.can_recurse(&context).await.unwrap());
    }
}
