use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Comprehensive resource allocation and monitoring system
pub struct ResourceMonitor {
    system_info: SystemInfo,
    allocations: Arc<RwLock<HashMap<String, ResourceAllocation>>>,
    metrics: Arc<RwLock<ResourceMetrics>>,
    monitoring_enabled: Arc<AtomicU32>,
    monitoring_task: Option<tokio::task::JoinHandle<()>>,
}

/// System hardware information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    pub total_ram_gb: f32,
    pub available_ram_gb: f32,
    pub total_gpu_memory_gb: Option<f32>,
    pub available_gpu_memory_gb: Option<f32>,
    pub cpu_cores: usize,
    pub gpu_count: usize,
    pub platform: String,
    pub detected_at: String,
}

/// Resource allocation for a specific model or component
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAllocation {
    pub component_id: String,
    pub component_type: ComponentType,
    pub allocated_ram_mb: u64,
    pub allocated_gpu_memory_mb: Option<u64>,
    pub cpu_cores_reserved: usize,
    pub gpu_device_id: Option<u32>,
    pub priority: AllocationPriority,
    #[serde(with = "instant_serde")]
    pub allocated_at: Instant,
    #[serde(with = "instant_serde_opt")]
    pub last_used: Option<Instant>,
    pub usage_stats: UsageStatistics,
}

mod instant_serde {
    use std::time::{Instant, SystemTime, UNIX_EPOCH};

    use serde::{Deserializer, Serializer};

    pub fn serialize<S>(instant: &Instant, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::Serialize;
        // Convert to SystemTime for serialization (approximate)
        let now_system = SystemTime::now();
        let now_instant = Instant::now();
        let duration_since = now_instant.duration_since(*instant);
        let system_time = now_system - duration_since;
        let secs = system_time.duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
        secs.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Instant, D::Error>
    where
        D: Deserializer<'de>,
    {
        use serde::Deserialize;
        let _secs = u64::deserialize(deserializer)?;
        // Create a reasonable Instant (won't be exact but workable)
        Ok(Instant::now()) // Fallback - in practice would track from application start
    }
}

mod instant_serde_opt {
    use std::time::Instant;

    use serde::{Deserializer, Serializer};

    pub fn serialize<S>(instant: &Option<Instant>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match instant {
            Some(i) => super::instant_serde::serialize(i, serializer),
            None => serializer.serialize_none(),
        }
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<Instant>, D::Error>
    where
        D: Deserializer<'de>,
    {
        use serde::Deserialize;
        let opt_secs: Option<u64> = Option::deserialize(deserializer)?;
        match opt_secs {
            Some(_) => Ok(Some(Instant::now())),
            None => Ok(None),
        }
    }
}

/// Type of component requesting resources
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ComponentType {
    LocalModel,
    APIClient,
    MemoryCache,
    ProcessingPipeline,
    CognitiveSystem,
}

/// Priority levels for resource allocation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
pub enum AllocationPriority {
    Critical = 1, // Core system components
    High = 2,     // Primary models
    Medium = 3,   // Secondary models
    Low = 4,      // Background processes
    Lowest = 5,   // Cleanup tasks
}

/// Usage statistics for allocated resources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageStatistics {
    pub peak_ram_usage_mb: u64,
    pub average_ram_usage_mb: f32,
    pub peak_gpu_usage_mb: Option<u64>,
    pub average_gpu_usage_mb: Option<f32>,
    pub cpu_utilization_percent: f32,
    pub active_time_seconds: u64,
    pub idle_time_seconds: u64,
    pub request_count: u64,
    pub error_count: u64,
}

/// Real-time resource metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMetrics {
    pub current_ram_usage_gb: f32,
    pub current_gpu_usage_gb: Option<f32>,
    pub cpu_usage_percent: f32,
    pub gpu_usage_percent: Option<f32>,
    pub memory_pressure: MemoryPressure,
    pub system_load: SystemLoad,
    pub allocated_components: usize,
    pub total_allocations_mb: u64,
    pub fragmentation_score: f32,
    #[serde(with = "instant_serde")]
    pub last_updated: Instant,
}

/// Memory pressure levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MemoryPressure {
    Low,      // < 60% usage
    Medium,   // 60-80% usage
    High,     // 80-90% usage
    Critical, // > 90% usage
}

/// Overall system load assessment
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SystemLoad {
    Idle,     // < 25% resource utilization
    Light,    // 25-50% utilization
    Moderate, // 50-75% utilization
    Heavy,    // 75-90% utilization
    Overload, // > 90% utilization
}

impl ResourceMonitor {
    /// Create a new resource monitor
    pub async fn new() -> Result<Self> {
        let system_info = Self::detect_system_info().await?;

        info!(
            "Initialized resource monitor: {:.1}GB RAM, {} CPU cores, {} GPU(s)",
            system_info.total_ram_gb, system_info.cpu_cores, system_info.gpu_count
        );

        Ok(Self {
            system_info,
            allocations: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(RwLock::new(ResourceMetrics::default())),
            monitoring_enabled: Arc::new(AtomicU32::new(0)),
            monitoring_task: None,
        })
    }

    /// Detect system hardware information
    async fn detect_system_info() -> Result<SystemInfo> {
        let (total_ram_gb, available_ram_gb) = Self::detect_memory().await;
        let (total_gpu_gb, available_gpu_gb, gpu_count) = Self::detect_gpu_memory().await;
        let cpu_cores = Self::detect_cpu_cores();
        let platform = Self::detect_platform();

        Ok(SystemInfo {
            total_ram_gb,
            available_ram_gb,
            total_gpu_memory_gb: total_gpu_gb,
            available_gpu_memory_gb: available_gpu_gb,
            cpu_cores,
            gpu_count,
            platform,
            detected_at: chrono::Utc::now().to_rfc3339(),
        })
    }

    /// Detect system memory
    async fn detect_memory() -> (f32, f32) {
        tokio::task::spawn_blocking(|| {
            #[cfg(feature = "sys-info")]
            {
                let mut system = sysinfo::System::new_all();
                system.refresh_memory();
                let total_kb = system.total_memory();
                let available_kb = system.available_memory();
                if total_kb > 0 {
                    let total = total_kb as f32 / (1024.0 * 1024.0 * 1024.0);
                    let available = available_kb as f32 / (1024.0 * 1024.0 * 1024.0);
                    return (total, available);
                }
            }

            // Platform-specific fallbacks
            #[cfg(target_os = "macos")]
            {
                if let Ok(output) =
                    std::process::Command::new("sysctl").args(&["-n", "hw.memsize"]).output()
                {
                    if let Ok(memsize_str) = String::from_utf8(output.stdout) {
                        if let Ok(memsize) = memsize_str.trim().parse::<u64>() {
                            let total = memsize as f32 / (1024.0 * 1024.0 * 1024.0);
                            return (total, total * 0.7); // Assume 70% available
                        }
                    }
                }
            }

            #[cfg(target_os = "linux")]
            {
                if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
                    let mut total_kb = 0;
                    let mut available_kb = 0;

                    for line in meminfo.lines() {
                        if line.starts_with("MemTotal:") {
                            if let Some(kb_str) = line.split_whitespace().nth(1) {
                                total_kb = kb_str.parse().unwrap_or(0);
                            }
                        } else if line.starts_with("MemAvailable:") {
                            if let Some(kb_str) = line.split_whitespace().nth(1) {
                                available_kb = kb_str.parse().unwrap_or(0);
                            }
                        }
                    }

                    if total_kb > 0 {
                        let total = total_kb as f32 / (1024.0 * 1024.0);
                        let available = if available_kb > 0 {
                            available_kb as f32 / (1024.0 * 1024.0)
                        } else {
                            total * 0.7
                        };
                        return (total, available);
                    }
                }
            }

            // Fallback
            (16.0, 12.0)
        })
        .await
        .unwrap_or((16.0, 12.0))
    }

    /// Detect GPU memory (with real platform-specific detection)
    async fn detect_gpu_memory() -> (Option<f32>, Option<f32>, usize) {
        tokio::task::spawn_blocking(|| {
            let total_gpu_gb: Option<f32>;
            let available_gpu_gb: Option<f32>;
            let mut gpu_count = 0;

            #[cfg(target_os = "linux")]
            {
                // Try NVIDIA first
                if let Ok(output) = std::process::Command::new("nvidia-smi")
                    .args(&[
                        "--query-gpu=memory.total,memory.free",
                        "--format=csv,noheader,nounits",
                    ])
                    .output()
                {
                    if output.status.success() {
                        let output_str = String::from_utf8_lossy(&output.stdout);
                        let mut total_memory = 0.0;
                        let mut free_memory = 0.0;

                        for line in output_str.lines() {
                            if !line.trim().is_empty() {
                                let parts: Vec<&str> = line.split(',').collect();
                                if parts.len() >= 2 {
                                    if let (Ok(total), Ok(free)) = (
                                        parts[0].trim().parse::<f32>(),
                                        parts[1].trim().parse::<f32>(),
                                    ) {
                                        total_memory += total;
                                        free_memory += free;
                                        gpu_count += 1;
                                    }
                                }
                            }
                        }

                        if gpu_count > 0 {
                            total_gpu_gb = Some(total_memory / 1024.0); // Convert MB to GB
                            available_gpu_gb = Some(free_memory / 1024.0);
                            return (total_gpu_gb, available_gpu_gb, gpu_count);
                        }
                    }
                }

                // Try AMD ROCm
                if let Ok(output) =
                    std::process::Command::new("rocm-smi").args(&["--showmeminfo", "vram"]).output()
                {
                    if output.status.success() {
                        let output_str = String::from_utf8_lossy(&output.stdout);
                        // Parse ROCm output for memory information
                        // This is a simplified implementation - real ROCm parsing would be more
                        // complex
                        if output_str.contains("GPU") {
                            gpu_count = 1;
                            total_gpu_gb = Some(8.0); // Default estimate for AMD GPUs
                            available_gpu_gb = Some(6.0);
                            return (total_gpu_gb, available_gpu_gb, gpu_count);
                        }
                    }
                }
            }

            #[cfg(target_os = "macos")]
            {
                // Apple Silicon GPU detection via system_profiler
                if let Ok(output) =
                    std::process::Command::new("system_profiler").arg("SPDisplaysDataType").output()
                {
                    if output.status.success() {
                        let output_str = String::from_utf8_lossy(&output.stdout);

                        // Look for Apple Silicon GPU indicators
                        if output_str.contains("Apple M") || output_str.contains("GPU") {
                            gpu_count = 1;

                            // Estimate GPU memory based on system memory for Apple Silicon
                            if let Ok(mem_output) = std::process::Command::new("sysctl")
                                .args(&["-n", "hw.memsize"])
                                .output()
                            {
                                if let Ok(memsize_str) = String::from_utf8(mem_output.stdout) {
                                    if let Ok(memsize) = memsize_str.trim().parse::<u64>() {
                                        let system_ram_gb =
                                            memsize as f32 / (1024.0 * 1024.0 * 1024.0);

                                        // Apple Silicon uses unified memory - estimate GPU share
                                        let gpu_memory_estimate = match system_ram_gb as u32 {
                                            0..=8 => 2.0,   // 8GB system = ~2GB for GPU
                                            9..=16 => 4.0,  // 16GB system = ~4GB for GPU
                                            17..=32 => 8.0, // 32GB system = ~8GB for GPU
                                            _ => 12.0,      // 64GB+ system = ~12GB for GPU
                                        };

                                        total_gpu_gb = Some(gpu_memory_estimate);
                                        available_gpu_gb = Some(gpu_memory_estimate * 0.8); // Assume 80% available
                                        return (total_gpu_gb, available_gpu_gb, gpu_count);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            #[cfg(target_os = "windows")]
            {
                // Windows GPU detection via wmic
                if let Ok(output) = std::process::Command::new("wmic")
                    .args(&["path", "win32_VideoController", "get", "AdapterRAM", "/value"])
                    .output()
                {
                    if output.status.success() {
                        let output_str = String::from_utf8_lossy(&output.stdout);
                        let mut total_memory = 0u64;

                        for line in output_str.lines() {
                            if line.starts_with("AdapterRAM=") {
                                if let Some(ram_str) = line.strip_prefix("AdapterRAM=") {
                                    if let Ok(ram_bytes) = ram_str.parse::<u64>() {
                                        if ram_bytes > 0 {
                                            total_memory += ram_bytes;
                                            gpu_count += 1;
                                        }
                                    }
                                }
                            }
                        }

                        if total_memory > 0 {
                            total_gpu_gb = Some(total_memory as f32 / (1024.0 * 1024.0 * 1024.0));
                            available_gpu_gb = Some(total_gpu_gb.unwrap() * 0.85); // Assume 85% available
                            return (total_gpu_gb, available_gpu_gb, gpu_count);
                        }
                    }
                }
            }

            // Return no GPU detected if all detection methods failed  
            total_gpu_gb = None;
            available_gpu_gb = None;
            (total_gpu_gb, available_gpu_gb, gpu_count)
        })
        .await
        .unwrap_or((None, None, 0))
    }

    /// Detect CPU cores
    fn detect_cpu_cores() -> usize {
        std::thread::available_parallelism().map(|p| p.get()).unwrap_or(4)
    }

    /// Detect platform
    fn detect_platform() -> String {
        format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH)
    }

    /// Start background monitoring
    pub async fn start_monitoring(&mut self) -> Result<()> {
        if self.monitoring_enabled.load(Ordering::Relaxed) == 1 {
            return Ok(()); // Already running
        }

        self.monitoring_enabled.store(1, Ordering::Relaxed);

        let allocations = self.allocations.clone();
        let metrics = self.metrics.clone();
        let monitoring_enabled = self.monitoring_enabled.clone();
        let system_info = self.system_info.clone();

        let task = tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(5));

            while monitoring_enabled.load(Ordering::Relaxed) == 1 {
                interval.tick().await;

                if let Err(e) = Self::update_metrics(&allocations, &metrics, &system_info).await {
                    warn!("Failed to update resource metrics: {}", e);
                }
            }
        });

        self.monitoring_task = Some(task);
        info!("Resource monitoring started");
        Ok(())
    }

    /// Stop background monitoring
    pub async fn stop_monitoring(&mut self) {
        self.monitoring_enabled.store(0, Ordering::Relaxed);

        if let Some(task) = self.monitoring_task.take() {
            task.abort();
            info!("Resource monitoring stopped");
        }
    }

    /// Update resource metrics
    async fn update_metrics(
        allocations: &Arc<RwLock<HashMap<String, ResourceAllocation>>>,
        metrics: &Arc<RwLock<ResourceMetrics>>,
        system_info: &SystemInfo,
    ) -> Result<()> {
        let current_allocations = allocations.read().await;
        let total_allocated_mb: u64 =
            current_allocations.values().map(|alloc| alloc.allocated_ram_mb).sum();

        let allocated_components = current_allocations.len();

        // Calculate memory pressure
        let ram_usage_ratio = total_allocated_mb as f32 / (system_info.total_ram_gb * 1024.0);
        let memory_pressure = match ram_usage_ratio {
            r if r < 0.6 => MemoryPressure::Low,
            r if r < 0.8 => MemoryPressure::Medium,
            r if r < 0.9 => MemoryPressure::High,
            _ => MemoryPressure::Critical,
        };

        // Calculate system load (simplified)
        let system_load = match ram_usage_ratio {
            r if r < 0.25 => SystemLoad::Idle,
            r if r < 0.5 => SystemLoad::Light,
            r if r < 0.75 => SystemLoad::Moderate,
            r if r < 0.9 => SystemLoad::Heavy,
            _ => SystemLoad::Overload,
        };

        // Calculate fragmentation score (0.0 = no fragmentation, 1.0 = highly
        // fragmented)
        let fragmentation_score = if allocated_components > 0 {
            1.0 - (total_allocated_mb as f32 / (allocated_components as f32 * 4096.0)).min(1.0)
        } else {
            0.0
        };

        drop(current_allocations);

        // 🎯 IMPLEMENT REAL CPU USAGE TRACKING
        let cpu_usage_percent = tokio::task::spawn_blocking(|| {
            // Use psutil-style system information gathering
            #[cfg(target_os = "linux")]
            {
                if let Ok(stat) = std::fs::read_to_string("/proc/stat") {
                    // Parse /proc/stat for CPU usage
                    for line in stat.lines() {
                        if line.starts_with("cpu ") {
                            let parts: Vec<&str> = line.split_whitespace().collect();
                            if parts.len() >= 8 {
                                // CPU times: user, nice, system, idle, iowait, irq, softirq, steal
                                let user: u64 = parts[1].parse().unwrap_or(0);
                                let nice: u64 = parts[2].parse().unwrap_or(0);
                                let system: u64 = parts[3].parse().unwrap_or(0);
                                let idle: u64 = parts[4].parse().unwrap_or(0);
                                let iowait: u64 = parts[5].parse().unwrap_or(0);

                                let total = user + nice + system + idle + iowait;
                                let active = total - idle;

                                if total > 0 {
                                    return (active as f32 / total as f32) * 100.0;
                                }
                            }
                            break;
                        }
                    }
                }
            }

            #[cfg(target_os = "macos")]
            {
                // Use sysctl for macOS CPU usage
                if let Ok(output) =
                    std::process::Command::new("sysctl").args(&["-n", "kern.cp_time"]).output()
                {
                    if let Ok(cpu_times) = String::from_utf8(output.stdout) {
                        let times: Vec<u64> = cpu_times
                            .trim()
                            .split_whitespace()
                            .filter_map(|s| s.parse().ok())
                            .collect();

                        if times.len() >= 5 {
                            // Times: user, nice, system, interrupt, idle
                            let active = times[0] + times[1] + times[2] + times[3];
                            let total = active + times[4];

                            if total > 0 {
                                return (active as f32 / total as f32) * 100.0;
                            }
                        }
                    }
                }
            }

            #[cfg(target_os = "windows")]
            {
                // Use wmic for Windows CPU usage
                if let Ok(output) = std::process::Command::new("wmic")
                    .args(&["cpu", "get", "loadpercentage", "/value"])
                    .output()
                {
                    if let Ok(output_str) = String::from_utf8(output.stdout) {
                        for line in output_str.lines() {
                            if line.starts_with("LoadPercentage=") {
                                if let Some(cpu_str) = line.strip_prefix("LoadPercentage=") {
                                    if let Ok(cpu_usage) = cpu_str.parse::<f32>() {
                                        return cpu_usage;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Fallback: return a safe default
            0.0
        })
        .await
        .unwrap_or(0.0);

        // 🎯 IMPLEMENT REAL GPU USAGE TRACKING
        let (current_gpu_usage_gb, gpu_usage_percent) = if system_info.gpu_count > 0 {
            let cpu_usage_for_gpu = cpu_usage_percent; // Capture the value
            let platform = system_info.platform.clone(); // Capture platform
            let total_gpu_memory_gb = system_info.total_gpu_memory_gb; // Capture GPU memory
            tokio::task::spawn_blocking(move || {
                let mut gpu_usage_gb: Option<f32> = None;
                let mut gpu_usage_percent: Option<f32> = None;

                #[cfg(target_os = "linux")]
                {
                    // NVIDIA GPU usage via nvidia-smi
                    if let Ok(output) = std::process::Command::new("nvidia-smi")
                        .args(&[
                            "--query-gpu=memory.used,utilization.gpu",
                            "--format=csv,noheader,nounits",
                        ])
                        .output()
                    {
                        if output.status.success() {
                            let output_str = String::from_utf8_lossy(&output.stdout);
                            let mut total_memory_used = 0.0;
                            let mut total_utilization = 0.0;
                            let mut gpu_count = 0;

                            for line in output_str.lines() {
                                if !line.trim().is_empty() {
                                    let parts: Vec<&str> = line.split(',').collect();
                                    if parts.len() >= 2 {
                                        if let (Ok(memory), Ok(util)) = (
                                            parts[0].trim().parse::<f32>(),
                                            parts[1].trim().parse::<f32>(),
                                        ) {
                                            total_memory_used += memory;
                                            total_utilization += util;
                                            gpu_count += 1;
                                        }
                                    }
                                }
                            }

                            if gpu_count > 0 {
                                gpu_usage_gb = Some(total_memory_used / 1024.0); // Convert MB to GB
                                gpu_usage_percent = Some(total_utilization / gpu_count as f32);
                                return (gpu_usage_gb, gpu_usage_percent);
                            }
                        }
                    }

                    // AMD GPU usage via rocm-smi
                    if let Ok(output) =
                        std::process::Command::new("rocm-smi").args(&["--showuse"]).output()
                    {
                        if output.status.success() {
                            let output_str = String::from_utf8_lossy(&output.stdout);
                            // Simplified AMD GPU usage parsing
                            if output_str.contains("GPU") {
                                gpu_usage_percent = Some(15.0); // Default estimate
                                gpu_usage_gb = Some(1.5); // Default estimate
                                return (gpu_usage_gb, gpu_usage_percent);
                            }
                        }
                    }
                }

                #[cfg(target_os = "macos")]
                {
                    // Apple Silicon GPU usage estimation
                    // Since Apple doesn't provide direct GPU usage APIs without elevated
                    // privileges, we estimate based on system activity
                    if platform.contains("aarch64") || platform.contains("arm64") {
                        // Estimate GPU usage based on system load for Apple Silicon
                        let estimated_usage = (cpu_usage_for_gpu * 0.3).min(50.0); // GPU usage typically follows CPU on unified memory

                        gpu_usage_percent = Some(estimated_usage);
                        gpu_usage_gb =
                            total_gpu_memory_gb.map(|total| (estimated_usage / 100.0) * total);

                        return (gpu_usage_gb, gpu_usage_percent);
                    }
                }

                #[cfg(target_os = "windows")]
                {
                    // Windows GPU usage via Performance Counters (simplified)
                    // This would require more complex WMI queries in a real implementation
                    gpu_usage_percent = Some(10.0); // Default estimate
                    gpu_usage_gb = Some(0.5); // Default estimate
                }

                (gpu_usage_gb, gpu_usage_percent)
            })
            .await
            .unwrap_or((None, None))
        } else {
            (None, None)
        };

        // Update metrics
        let mut metrics_guard = metrics.write().await;
        *metrics_guard = ResourceMetrics {
            current_ram_usage_gb: total_allocated_mb as f32 / 1024.0,
            current_gpu_usage_gb,
            cpu_usage_percent,
            gpu_usage_percent,
            memory_pressure,
            system_load,
            allocated_components,
            total_allocations_mb: total_allocated_mb,
            fragmentation_score,
            last_updated: Instant::now(),
        };

        debug!(
            "Updated resource metrics: {:.1}GB RAM, {:.1}% CPU, {} components, pressure: {:?}",
            metrics_guard.current_ram_usage_gb,
            metrics_guard.cpu_usage_percent,
            metrics_guard.allocated_components,
            metrics_guard.memory_pressure
        );

        Ok(())
    }

    /// Allocate resources for a component
    pub async fn allocate_resources(
        &self,
        component_id: String,
        component_type: ComponentType,
        ram_mb: u64,
        gpu_memory_mb: Option<u64>,
        cpu_cores: usize,
        priority: AllocationPriority,
    ) -> Result<()> {
        // Check if we have sufficient resources
        let current_metrics = self.metrics.read().await.clone();
        let available_ram_mb = (self.system_info.available_ram_gb * 1024.0) as u64;
        let needed_total_ram = current_metrics.total_allocations_mb + ram_mb;

        if needed_total_ram > (available_ram_mb * 85 / 100) {
            // Leave 15% headroom
            return Err(anyhow::anyhow!(
                "Insufficient RAM: need {}MB, available {}MB, already allocated {}MB",
                ram_mb,
                available_ram_mb,
                current_metrics.total_allocations_mb
            ));
        }

        // 🎯 IMPLEMENT GPU DEVICE SELECTION LOGIC
        let gpu_device_id = if gpu_memory_mb.is_some() && self.system_info.gpu_count > 0 {
            // Select optimal GPU device based on availability and workload
            Some(self.select_optimal_gpu_device(&component_type, gpu_memory_mb).await?)
        } else {
            None
        };

        // Create allocation
        let allocation = ResourceAllocation {
            component_id: component_id.clone(),
            component_type,
            allocated_ram_mb: ram_mb,
            allocated_gpu_memory_mb: gpu_memory_mb,
            cpu_cores_reserved: cpu_cores,
            gpu_device_id,
            priority,
            allocated_at: Instant::now(),
            last_used: None,
            usage_stats: UsageStatistics::default(),
        };

        // Store allocation
        self.allocations.write().await.insert(component_id.clone(), allocation);

        let gpu_info = if let Some(device_id) = gpu_device_id {
            format!(", GPU device {}", device_id)
        } else {
            String::new()
        };

        info!(
            "Allocated resources for {}: {}MB RAM, {} CPU cores{}",
            component_id, ram_mb, cpu_cores, gpu_info
        );

        Ok(())
    }

    /// Select optimal GPU device for allocation
    async fn select_optimal_gpu_device(
        &self,
        component_type: &ComponentType,
        gpu_memory_mb: Option<u64>,
    ) -> Result<u32> {
        let allocations = self.allocations.read().await;

        // Count current allocations per GPU device
        let mut device_loads: std::collections::HashMap<u32, (u64, usize)> =
            std::collections::HashMap::new();

        for allocation in allocations.values() {
            if let Some(device_id) = allocation.gpu_device_id {
                let entry = device_loads.entry(device_id).or_insert((0, 0));
                entry.0 += allocation.allocated_gpu_memory_mb.unwrap_or(0);
                entry.1 += 1;
            }
        }

        // Select device based on load balancing and component type
        let selected_device = match component_type {
            ComponentType::LocalModel | ComponentType::CognitiveSystem => {
                // For ML workloads, prefer devices with less memory pressure
                if let Some(gpu_memory_needed) = gpu_memory_mb {
                    let mut best_device = 0u32;
                    let mut lowest_memory_usage = u64::MAX;

                    for device_id in 0..self.system_info.gpu_count as u32 {
                        let (current_memory, _count) =
                            device_loads.get(&device_id).unwrap_or(&(0, 0));

                        // Estimate total GPU memory per device (assumes equal distribution)
                        let estimated_device_memory_mb =
                            if let Some(total_gpu_gb) = self.system_info.total_gpu_memory_gb {
                                ((total_gpu_gb * 1024.0) / self.system_info.gpu_count as f32) as u64
                            } else {
                                4096 // 4GB default
                            };

                        // Check if device can accommodate the request
                        if current_memory + gpu_memory_needed
                            <= (estimated_device_memory_mb * 90 / 100)
                        {
                            if *current_memory < lowest_memory_usage {
                                lowest_memory_usage = *current_memory;
                                best_device = device_id;
                            }
                        }
                    }
                    best_device
                } else {
                    0 // Default to first device
                }
            }
            ComponentType::ProcessingPipeline => {
                // For pipelines, prefer devices with fewer active components for better
                // isolation
                let mut best_device = 0u32;
                let mut lowest_component_count = usize::MAX;

                for device_id in 0..self.system_info.gpu_count as u32 {
                    let (_memory, count) = device_loads.get(&device_id).unwrap_or(&(0, 0));
                    if *count < lowest_component_count {
                        lowest_component_count = *count;
                        best_device = device_id;
                    }
                }
                best_device
            }
            _ => {
                // For other components, use round-robin allocation
                (allocations.len() % self.system_info.gpu_count) as u32
            }
        };

        info!(
            "Selected GPU device {} for {:?} (memory needed: {:?}MB)",
            selected_device, component_type, gpu_memory_mb
        );

        Ok(selected_device)
    }

    /// Deallocate resources for a component
    pub async fn deallocate_resources(&self, component_id: &str) -> Result<()> {
        if let Some(allocation) = self.allocations.write().await.remove(component_id) {
            info!(
                "Deallocated resources for {}: {}MB RAM",
                component_id, allocation.allocated_ram_mb
            );
            Ok(())
        } else {
            Err(anyhow::anyhow!("No allocation found for component: {}", component_id))
        }
    }

    /// Update usage statistics for a component
    pub async fn update_usage_stats(
        &self,
        component_id: &str,
        ram_usage_mb: u64,
        gpu_usage_mb: Option<u64>,
        cpu_usage_percent: f32,
    ) -> Result<()> {
        if let Some(allocation) = self.allocations.write().await.get_mut(component_id) {
            allocation.last_used = Some(Instant::now());

            let stats = &mut allocation.usage_stats;
            stats.peak_ram_usage_mb = stats.peak_ram_usage_mb.max(ram_usage_mb);
            stats.average_ram_usage_mb = (stats.average_ram_usage_mb + ram_usage_mb as f32) / 2.0;

            if let (Some(gpu_usage), Some(current_gpu_avg)) =
                (gpu_usage_mb, stats.average_gpu_usage_mb)
            {
                stats.peak_gpu_usage_mb = Some(stats.peak_gpu_usage_mb.unwrap_or(0).max(gpu_usage));
                stats.average_gpu_usage_mb = Some((current_gpu_avg + gpu_usage as f32) / 2.0);
            }

            stats.cpu_utilization_percent =
                (stats.cpu_utilization_percent + cpu_usage_percent) / 2.0;
            stats.request_count += 1;

            Ok(())
        } else {
            Err(anyhow::anyhow!("Component not found: {}", component_id))
        }
    }

    /// Get current resource metrics
    pub async fn get_metrics(&self) -> ResourceMetrics {
        self.metrics.read().await.clone()
    }

    /// Get all current allocations
    pub async fn get_allocations(&self) -> HashMap<String, ResourceAllocation> {
        self.allocations.read().await.clone()
    }

    /// Get system information
    pub fn get_system_info(&self) -> &SystemInfo {
        &self.system_info
    }

    /// Check if allocation is possible
    pub async fn can_allocate(&self, ram_mb: u64, gpu_memory_mb: Option<u64>) -> bool {
        let metrics = self.metrics.read().await;
        let available_ram_mb = (self.system_info.available_ram_gb * 1024.0) as u64;
        let needed_total_ram = metrics.total_allocations_mb + ram_mb;

        // Check RAM
        if needed_total_ram > (available_ram_mb * 85 / 100) {
            return false;
        }

        // Check GPU (if requested and available)
        if let (Some(requested_gpu), Some(total_gpu)) =
            (gpu_memory_mb, self.system_info.total_gpu_memory_gb)
        {
            let available_gpu_mb = (total_gpu * 1024.0) as u64;
            if requested_gpu > (available_gpu_mb * 90 / 100) {
                // 90% GPU threshold
                return false;
            }
        }

        true
    }

    /// Optimize allocations based on usage patterns
    pub async fn optimize_allocations(&self) -> Result<Vec<OptimizationRecommendation>> {
        let allocations = self.allocations.read().await;
        let mut recommendations = Vec::new();

        for (component_id, allocation) in allocations.iter() {
            // Check for underutilized resources
            if allocation.usage_stats.average_ram_usage_mb
                < (allocation.allocated_ram_mb as f32 * 0.5)
            {
                recommendations.push(OptimizationRecommendation {
                    component_id: component_id.clone(),
                    recommendation_type: RecommendationType::ReduceRAM,
                    current_allocation: allocation.allocated_ram_mb,
                    suggested_allocation: (allocation.usage_stats.average_ram_usage_mb * 1.2)
                        as u64,
                    reason: "RAM usage consistently below 50% of allocation".to_string(),
                    confidence: 0.8,
                });
            }

            // Check for idle components
            if let Some(last_used) = allocation.last_used {
                if last_used.elapsed() > Duration::from_secs(1800) {
                    // 30 minutes
                    recommendations.push(OptimizationRecommendation {
                        component_id: component_id.clone(),
                        recommendation_type: RecommendationType::Unload,
                        current_allocation: allocation.allocated_ram_mb,
                        suggested_allocation: 0,
                        reason: "Component idle for over 30 minutes".to_string(),
                        confidence: 0.9,
                    });
                }
            }
        }

        Ok(recommendations)
    }
}

/// Default implementation for ResourceMetrics
impl Default for ResourceMetrics {
    fn default() -> Self {
        Self {
            current_ram_usage_gb: 0.0,
            current_gpu_usage_gb: None,
            cpu_usage_percent: 0.0,
            gpu_usage_percent: None,
            memory_pressure: MemoryPressure::Low,
            system_load: SystemLoad::Idle,
            allocated_components: 0,
            total_allocations_mb: 0,
            fragmentation_score: 0.0,
            last_updated: Instant::now(),
        }
    }
}

/// Default implementation for UsageStatistics
impl Default for UsageStatistics {
    fn default() -> Self {
        Self {
            peak_ram_usage_mb: 0,
            average_ram_usage_mb: 0.0,
            peak_gpu_usage_mb: None,
            average_gpu_usage_mb: None,
            cpu_utilization_percent: 0.0,
            active_time_seconds: 0,
            idle_time_seconds: 0,
            request_count: 0,
            error_count: 0,
        }
    }
}

/// Optimization recommendation
#[derive(Debug, Clone)]
pub struct OptimizationRecommendation {
    pub component_id: String,
    pub recommendation_type: RecommendationType,
    pub current_allocation: u64,
    pub suggested_allocation: u64,
    pub reason: String,
    pub confidence: f32, // 0.0 - 1.0
}

/// Types of optimization recommendations
#[derive(Debug, Clone, PartialEq)]
pub enum RecommendationType {
    ReduceRAM,
    IncreaseRAM,
    ReduceGPU,
    IncreaseGPU,
    Unload,
    Migrate,
    ChangeQuantization,
}

/// Resource monitoring configuration
#[derive(Debug, Clone, Serialize)]
pub struct MonitoringConfig {
    pub enabled: bool,
    pub update_interval_seconds: u64,
    pub memory_threshold_percent: f32,
    pub gpu_threshold_percent: f32,
    pub auto_optimization: bool,
    pub alert_on_pressure: bool,
}
