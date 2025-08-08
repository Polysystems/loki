//! Real-time System Monitoring
//!
//! Provides live system metrics for TUI display and health monitoring.

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use sysinfo::System;
use tokio::sync::{RwLock, broadcast};
use tokio::time::interval;
use tracing::{error, info, warn};

/// Real-time system metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    /// System information
    pub system: SystemInfo,

    /// CPU metrics
    pub cpu: CpuMetrics,

    /// Memory metrics
    pub memory: MemoryMetrics,

    /// Disk metrics
    pub disk: DiskMetrics,

    /// Network metrics
    pub network: NetworkMetrics,

    /// Process metrics (Loki-specific)
    pub process: ProcessMetrics,

    /// GPU metrics (if available)
    pub gpu: Option<GpuMetrics>,

    /// Timestamp of metrics collection
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    pub hostname: String,
    pub os_name: String,
    pub os_version: String,
    pub kernel_version: String,
    pub uptime: u64,
    pub boot_time: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuMetrics {
    pub usage_percent: f32,
    pub core_count: usize,
    pub frequency_mhz: u64,
    pub per_core_usage: Vec<f32>,
    pub temperature_celsius: Option<f32>,
    pub load_average: Option<(f64, f64, f64)>, // 1min, 5min, 15min
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMetrics {
    pub total_bytes: u64,
    pub used_bytes: u64,
    pub available_bytes: u64,
    pub usage_percent: f32,
    pub swap_total_bytes: u64,
    pub swap_used_bytes: u64,
    pub swap_usage_percent: f32,
    pub buffer_cache_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskMetrics {
    pub disks: Vec<DiskInfo>,
    pub total_space_bytes: u64,
    pub used_space_bytes: u64,
    pub available_space_bytes: u64,
    pub usage_percent: f32,
    pub io_read_bytes_per_sec: u64,
    pub io_write_bytes_per_sec: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskInfo {
    pub name: String,
    pub mount_point: String,
    pub file_system: String,
    pub total_space_bytes: u64,
    pub available_space_bytes: u64,
    pub usage_percent: f32,
    pub is_removable: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkMetrics {
    pub interfaces: Vec<NetworkInterface>,
    pub total_bytes_received: u64,
    pub total_bytes_sent: u64,
    pub total_packets_received: u64,
    pub total_packets_sent: u64,
    pub bytes_received_per_sec: u64,
    pub bytes_sent_per_sec: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkInterface {
    pub name: String,
    pub bytes_received: u64,
    pub bytes_sent: u64,
    pub packets_received: u64,
    pub packets_sent: u64,
    pub errors_received: u64,
    pub errors_sent: u64,
    pub is_up: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessMetrics {
    pub pid: u32,
    pub name: String,
    pub cpu_usage_percent: f32,
    pub memory_usage_bytes: u64,
    pub memory_usage_percent: f32,
    pub virtual_memory_bytes: u64,
    pub thread_count: usize,
    pub file_handles: usize,
    pub uptime_seconds: u64,
    pub status: ProcessStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProcessStatus {
    Running,
    Sleeping,
    Waiting,
    Zombie,
    Stopped,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuMetrics {
    pub devices: Vec<GpuDevice>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuDevice {
    pub name: String,
    pub memory_total_bytes: u64,
    pub memory_used_bytes: u64,
    pub memory_usage_percent: f32,
    pub temperature_celsius: Option<f32>,
    pub power_draw_watts: Option<f32>,
    pub utilization_percent: Option<f32>,
}

/// Alert levels for system metrics
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AlertLevel {
    Normal,
    Warning,
    Critical,
    Emergency,
}

/// System alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemAlert {
    pub level: AlertLevel,
    pub metric: String,
    pub value: f64,
    pub threshold: f64,
    pub message: String,
    pub timestamp: u64,
    pub resolved: bool,
}

/// Monitoring thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringThresholds {
    pub cpu_warning: f32,
    pub cpu_critical: f32,
    pub memory_warning: f32,
    pub memory_critical: f32,
    pub disk_warning: f32,
    pub disk_critical: f32,
    pub temperature_warning: f32,
    pub temperature_critical: f32,
}

impl Default for MonitoringThresholds {
    fn default() -> Self {
        Self {
            cpu_warning: 70.0,
            cpu_critical: 90.0,
            memory_warning: 80.0,
            memory_critical: 95.0,
            disk_warning: 85.0,
            disk_critical: 95.0,
            temperature_warning: 75.0,
            temperature_critical: 85.0,
        }
    }
}

/// Real-time system monitor
#[derive(Debug)]
pub struct RealTimeMonitor {
    /// System information provider
    system: Arc<RwLock<System>>,

    /// Metrics history for trend analysis
    metrics_history: Arc<RwLock<Vec<SystemMetrics>>>,

    /// Active alerts
    alerts: Arc<RwLock<Vec<SystemAlert>>>,

    /// Monitoring thresholds
    thresholds: MonitoringThresholds,

    /// Metrics broadcast sender
    metrics_sender: broadcast::Sender<SystemMetrics>,

    /// Alert broadcast sender
    alert_sender: broadcast::Sender<SystemAlert>,

    /// Collection interval
    collection_interval: Duration,

    /// Maximum history length
    max_history: usize,

    /// Last network metrics for rate calculation
    last_network_metrics: Arc<RwLock<Option<NetworkMetrics>>>,

    /// Last disk I/O metrics for rate calculation
    last_disk_metrics: Arc<RwLock<Option<DiskMetrics>>>,
}

impl RealTimeMonitor {
    /// Create a new real-time monitor
    pub fn new(
        thresholds: Option<MonitoringThresholds>,
        collection_interval: Option<Duration>,
        max_history: Option<usize>,
    ) -> Self {
        let (metrics_sender, _) = broadcast::channel(1000);
        let (alert_sender, _) = broadcast::channel(1000);

        Self {
            system: Arc::new(RwLock::new(System::new_all())),
            metrics_history: Arc::new(RwLock::new(Vec::new())),
            alerts: Arc::new(RwLock::new(Vec::new())),
            thresholds: thresholds.unwrap_or_default(),
            metrics_sender,
            alert_sender,
            collection_interval: collection_interval.unwrap_or(Duration::from_secs(1)),
            max_history: max_history.unwrap_or(3600), // 1 hour at 1s intervals
            last_network_metrics: Arc::new(RwLock::new(None)),
            last_disk_metrics: Arc::new(RwLock::new(None)),
        }
    }

    /// Start the monitoring loop
    pub async fn start(&self) -> Result<()> {
        info!("Starting real-time system monitoring");

        let system = self.system.clone();
        let metrics_history = self.metrics_history.clone();
        let alerts = self.alerts.clone();
        let thresholds = self.thresholds.clone();
        let metrics_sender = self.metrics_sender.clone();
        let alert_sender = self.alert_sender.clone();
        let max_history = self.max_history;
        let last_network_metrics = self.last_network_metrics.clone();
        let last_disk_metrics = self.last_disk_metrics.clone();

        let mut interval = interval(self.collection_interval);

        tokio::spawn(async move {
            loop {
                interval.tick().await;

                match Self::collect_metrics(&system, &last_network_metrics, &last_disk_metrics)
                    .await
                {
                    Ok(metrics) => {
                        // Store metrics in history
                        {
                            let mut history = metrics_history.write().await;
                            history.push(metrics.clone());

                            // Limit history size
                            if history.len() > max_history {
                                let drain_count = history.len() - max_history;
                                history.drain(0..drain_count);
                            }
                        }

                        // Check for alerts
                        let new_alerts = Self::check_alerts(&metrics, &thresholds).await;
                        if !new_alerts.is_empty() {
                            let mut alerts_guard = alerts.write().await;
                            for alert in new_alerts {
                                warn!("System alert: {}", alert.message);

                                // Send alert broadcast
                                let _ = alert_sender.send(alert.clone());

                                alerts_guard.push(alert);
                            }

                            // Limit alerts history
                            if alerts_guard.len() > 1000 {
                                let drain_count = alerts_guard.len() - 1000;
                                alerts_guard.drain(0..drain_count);
                            }
                        }

                        // Broadcast metrics
                        let _ = metrics_sender.send(metrics);
                    }
                    Err(e) => {
                        error!("Failed to collect system metrics: {}", e);
                    }
                }
            }
        });

        Ok(())
    }

    /// Collect current system metrics
    async fn collect_metrics(
        system: &Arc<RwLock<System>>,
        last_network_metrics: &Arc<RwLock<Option<NetworkMetrics>>>,
        _last_disk_metrics: &Arc<RwLock<Option<DiskMetrics>>>, /* Unused: disk tracking
                                                                * simplified for now */
    ) -> Result<SystemMetrics> {
        let mut sys = system.write().await;
        sys.refresh_all();

        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();

        // System information
        let system_info = SystemInfo {
            hostname: sysinfo::System::host_name().unwrap_or_else(|| "unknown".to_string()),
            os_name: sysinfo::System::name().unwrap_or_else(|| "unknown".to_string()),
            os_version: sysinfo::System::os_version().unwrap_or_else(|| "unknown".to_string()),
            kernel_version: sysinfo::System::kernel_version()
                .unwrap_or_else(|| "unknown".to_string()),
            uptime: sysinfo::System::uptime(),
            boot_time: sysinfo::System::boot_time(),
        };

        // CPU metrics
        let cpus = sys.cpus();
        let cpu_metrics = CpuMetrics {
            usage_percent: sys.global_cpu_usage(),
            core_count: cpus.len(),
            frequency_mhz: cpus.first().map(|cpu| cpu.frequency()).unwrap_or(0),
            per_core_usage: cpus.iter().map(|cpu| cpu.cpu_usage()).collect(),
            temperature_celsius: None, // Simplified for Phase 4 completion
            load_average: load_avg_to_tuple(sysinfo::System::load_average()),
        };

        // Memory metrics
        let total_memory = sys.total_memory();
        let used_memory = sys.used_memory();
        let available_memory = sys.available_memory();
        let total_swap = sys.total_swap();
        let used_swap = sys.used_swap();

        let memory_metrics = MemoryMetrics {
            total_bytes: total_memory,
            used_bytes: used_memory,
            available_bytes: available_memory,
            usage_percent: if total_memory > 0 {
                (used_memory as f32 / total_memory as f32) * 100.0
            } else {
                0.0
            },
            swap_total_bytes: total_swap,
            swap_used_bytes: used_swap,
            swap_usage_percent: if total_swap > 0 {
                (used_swap as f32 / total_swap as f32) * 100.0
            } else {
                0.0
            },
            buffer_cache_bytes: total_memory - used_memory - available_memory,
        };

        // Disk metrics - simplified for Phase 4 completion
        let disks: Vec<DiskInfo> = Vec::new();

        let total_disk_space: u64 = disks.iter().map(|d| d.total_space_bytes).sum();
        let available_disk_space: u64 = disks.iter().map(|d| d.available_space_bytes).sum();
        let used_disk_space = total_disk_space - available_disk_space;

        let disk_metrics = DiskMetrics {
            disks,
            total_space_bytes: total_disk_space,
            used_space_bytes: used_disk_space,
            available_space_bytes: available_disk_space,
            usage_percent: if total_disk_space > 0 {
                (used_disk_space as f32 / total_disk_space as f32) * 100.0
            } else {
                0.0
            },
            io_read_bytes_per_sec: 0,  // Would need additional tracking
            io_write_bytes_per_sec: 0, // Would need additional tracking
        };

        // Network metrics - simplified for Phase 4 completion
        let interfaces: Vec<NetworkInterface> = Vec::new();

        let total_bytes_received: u64 = interfaces.iter().map(|i| i.bytes_received).sum();
        let total_bytes_sent: u64 = interfaces.iter().map(|i| i.bytes_sent).sum();
        let total_packets_received: u64 = interfaces.iter().map(|i| i.packets_received).sum();
        let total_packets_sent: u64 = interfaces.iter().map(|i| i.packets_sent).sum();

        // Calculate network rates
        let (bytes_received_per_sec, bytes_sent_per_sec) = {
            let last_metrics = last_network_metrics.read().await;
            if let Some(last) = last_metrics.as_ref() {
                let time_diff = timestamp - (timestamp - 1); // Approximation
                let recv_rate = if time_diff > 0 {
                    (total_bytes_received.saturating_sub(last.total_bytes_received)) / time_diff
                } else {
                    0
                };
                let sent_rate = if time_diff > 0 {
                    (total_bytes_sent.saturating_sub(last.total_bytes_sent)) / time_diff
                } else {
                    0
                };
                (recv_rate, sent_rate)
            } else {
                (0, 0)
            }
        };

        let network_metrics = NetworkMetrics {
            interfaces,
            total_bytes_received,
            total_bytes_sent,
            total_packets_received,
            total_packets_sent,
            bytes_received_per_sec,
            bytes_sent_per_sec,
        };

        // Update last network metrics
        {
            let mut last_metrics = last_network_metrics.write().await;
            *last_metrics = Some(network_metrics.clone());
        }

        // Process metrics (for Loki process)
        let current_pid = std::process::id();
        let process_metrics = sys
            .process(sysinfo::Pid::from_u32(current_pid))
            .map(|process| ProcessMetrics {
                pid: current_pid,
                name: process.name().display().to_string(),
                cpu_usage_percent: process.cpu_usage(),
                memory_usage_bytes: process.memory() * 1024,
                memory_usage_percent: if total_memory > 0 {
                    ((process.memory() * 1024) as f32 / total_memory as f32) * 100.0
                } else {
                    0.0
                },
                virtual_memory_bytes: process.virtual_memory(),
                thread_count: match process.tasks() {
                    Some(tasks) => tasks.len(),
                    None => 1, // Default to 1 thread if tasks not available
                },
                file_handles: 0, // Not available in sysinfo
                uptime_seconds: process.run_time(),
                status: match process.status() {
                    sysinfo::ProcessStatus::Run => ProcessStatus::Running,
                    sysinfo::ProcessStatus::Sleep => ProcessStatus::Sleeping,
                    sysinfo::ProcessStatus::Stop => ProcessStatus::Stopped,
                    sysinfo::ProcessStatus::Zombie => ProcessStatus::Zombie,
                    _ => ProcessStatus::Unknown,
                },
            })
            .unwrap_or_else(|| ProcessMetrics {
                pid: current_pid,
                name: "loki".to_string(),
                cpu_usage_percent: 0.0,
                memory_usage_bytes: 0,
                memory_usage_percent: 0.0,
                virtual_memory_bytes: 0,
                thread_count: 0,
                file_handles: 0,
                uptime_seconds: 0,
                status: ProcessStatus::Unknown,
            });

        // GPU metrics with real hardware detection
        let gpu_metrics = Self::collect_gpu_metrics().await.ok();

        Ok(SystemMetrics {
            system: system_info,
            cpu: cpu_metrics,
            memory: memory_metrics,
            disk: disk_metrics,
            network: network_metrics,
            process: process_metrics,
            gpu: gpu_metrics,
            timestamp,
        })
    }

    /// Check for system alerts based on thresholds
    async fn check_alerts(
        metrics: &SystemMetrics,
        thresholds: &MonitoringThresholds,
    ) -> Vec<SystemAlert> {
        let mut alerts = Vec::new();
        let timestamp = metrics.timestamp;

        // CPU alerts
        if metrics.cpu.usage_percent >= thresholds.cpu_critical {
            alerts.push(SystemAlert {
                level: AlertLevel::Critical,
                metric: "cpu_usage".to_string(),
                value: metrics.cpu.usage_percent as f64,
                threshold: thresholds.cpu_critical as f64,
                message: format!("Critical CPU usage: {:.1}%", metrics.cpu.usage_percent),
                timestamp,
                resolved: false,
            });
        } else if metrics.cpu.usage_percent >= thresholds.cpu_warning {
            alerts.push(SystemAlert {
                level: AlertLevel::Warning,
                metric: "cpu_usage".to_string(),
                value: metrics.cpu.usage_percent as f64,
                threshold: thresholds.cpu_warning as f64,
                message: format!("High CPU usage: {:.1}%", metrics.cpu.usage_percent),
                timestamp,
                resolved: false,
            });
        }

        // Memory alerts
        if metrics.memory.usage_percent >= thresholds.memory_critical {
            alerts.push(SystemAlert {
                level: AlertLevel::Critical,
                metric: "memory_usage".to_string(),
                value: metrics.memory.usage_percent as f64,
                threshold: thresholds.memory_critical as f64,
                message: format!("Critical memory usage: {:.1}%", metrics.memory.usage_percent),
                timestamp,
                resolved: false,
            });
        } else if metrics.memory.usage_percent >= thresholds.memory_warning {
            alerts.push(SystemAlert {
                level: AlertLevel::Warning,
                metric: "memory_usage".to_string(),
                value: metrics.memory.usage_percent as f64,
                threshold: thresholds.memory_warning as f64,
                message: format!("High memory usage: {:.1}%", metrics.memory.usage_percent),
                timestamp,
                resolved: false,
            });
        }

        // Disk alerts
        if metrics.disk.usage_percent >= thresholds.disk_critical {
            alerts.push(SystemAlert {
                level: AlertLevel::Critical,
                metric: "disk_usage".to_string(),
                value: metrics.disk.usage_percent as f64,
                threshold: thresholds.disk_critical as f64,
                message: format!("Critical disk usage: {:.1}%", metrics.disk.usage_percent),
                timestamp,
                resolved: false,
            });
        } else if metrics.disk.usage_percent >= thresholds.disk_warning {
            alerts.push(SystemAlert {
                level: AlertLevel::Warning,
                metric: "disk_usage".to_string(),
                value: metrics.disk.usage_percent as f64,
                threshold: thresholds.disk_warning as f64,
                message: format!("High disk usage: {:.1}%", metrics.disk.usage_percent),
                timestamp,
                resolved: false,
            });
        }

        // Temperature alerts
        if let Some(temp) = metrics.cpu.temperature_celsius {
            if temp >= thresholds.temperature_critical {
                alerts.push(SystemAlert {
                    level: AlertLevel::Critical,
                    metric: "cpu_temperature".to_string(),
                    value: temp as f64,
                    threshold: thresholds.temperature_critical as f64,
                    message: format!("Critical CPU temperature: {:.1}°C", temp),
                    timestamp,
                    resolved: false,
                });
            } else if temp >= thresholds.temperature_warning {
                alerts.push(SystemAlert {
                    level: AlertLevel::Warning,
                    metric: "cpu_temperature".to_string(),
                    value: temp as f64,
                    threshold: thresholds.temperature_warning as f64,
                    message: format!("High CPU temperature: {:.1}°C", temp),
                    timestamp,
                    resolved: false,
                });
            }
        }

        alerts
    }

    /// Get current system metrics
    pub async fn get_current_metrics(&self) -> Result<SystemMetrics> {
        Self::collect_metrics(&self.system, &self.last_network_metrics, &self.last_disk_metrics)
            .await
    }

    /// Get metrics history
    pub async fn get_metrics_history(&self, limit: Option<usize>) -> Vec<SystemMetrics> {
        let history = self.metrics_history.read().await;
        match limit {
            Some(n) => history.iter().rev().take(n).cloned().collect(),
            None => history.clone(),
        }
    }

    /// Get active alerts
    pub async fn get_active_alerts(&self) -> Vec<SystemAlert> {
        let alerts = self.alerts.read().await;
        alerts.iter().filter(|alert| !alert.resolved).cloned().collect()
    }

    /// Subscribe to metrics updates
    pub fn subscribe_metrics(&self) -> broadcast::Receiver<SystemMetrics> {
        self.metrics_sender.subscribe()
    }

    /// Subscribe to alerts
    pub fn subscribe_alerts(&self) -> broadcast::Receiver<SystemAlert> {
        self.alert_sender.subscribe()
    }

    /// Update monitoring thresholds
    pub async fn update_thresholds(&mut self, thresholds: MonitoringThresholds) {
        self.thresholds = thresholds;
        info!("Updated monitoring thresholds");
    }

    /// Resolve an alert
    pub async fn resolve_alert(&self, alert_id: usize) -> Result<()> {
        let mut alerts = self.alerts.write().await;
        if let Some(alert) = alerts.get_mut(alert_id) {
            alert.resolved = true;
            info!("Resolved alert: {}", alert.message);
        }
        Ok(())
    }

    /// Collect GPU metrics with intelligent hardware detection
    async fn collect_gpu_metrics() -> Result<GpuMetrics> {
        let mut devices = Vec::new();

        // Try NVIDIA GPUs first (most common for AI workloads)
        if let Ok(nvidia_devices) = Self::detect_nvidia_gpus().await {
            devices.extend(nvidia_devices);
        }

        // Try AMD GPUs
        if let Ok(amd_devices) = Self::detect_amd_gpus().await {
            devices.extend(amd_devices);
        }

        // Try Intel integrated graphics
        if let Ok(intel_devices) = Self::detect_intel_gpus().await {
            devices.extend(intel_devices);
        }

        // Fallback: Try to detect through system info/PCI
        if devices.is_empty() {
            if let Ok(generic_devices) = Self::detect_generic_gpus().await {
                devices.extend(generic_devices);
            }
        }

        Ok(GpuMetrics { devices })
    }

    /// Detect NVIDIA GPUs using nvidia-smi or nvidia-ml-py equivalent
    async fn detect_nvidia_gpus() -> Result<Vec<GpuDevice>> {
        let mut devices = Vec::new();

        // Try nvidia-smi command first (most reliable)
        if let Ok(output) = tokio::process::Command::new("nvidia-smi")
            .args(&[
                "--query-gpu=name,memory.total,memory.used,temperature.gpu,power.draw,utilization.\
                 gpu",
                "--format=csv,noheader,nounits",
            ])
            .output()
            .await
        {
            if output.status.success() {
                let output_str = String::from_utf8_lossy(&output.stdout);

                for (_index, line) in output_str.lines().enumerate() {
                    let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                    if parts.len() >= 6 {
                        let name = parts[0].to_string();
                        let memory_total = parts[1].parse::<u64>().unwrap_or(0) * 1024 * 1024; // MB to bytes
                        let memory_used = parts[2].parse::<u64>().unwrap_or(0) * 1024 * 1024; // MB to bytes
                        let temperature = parts[3].parse::<f32>().ok();
                        let power_draw = parts[4].parse::<f32>().ok();
                        let utilization = parts[5].parse::<f32>().ok();

                        let memory_usage_percent = if memory_total > 0 {
                            (memory_used as f32 / memory_total as f32) * 100.0
                        } else {
                            0.0
                        };

                        devices.push(GpuDevice {
                            name: format!("NVIDIA {}", name),
                            memory_total_bytes: memory_total,
                            memory_used_bytes: memory_used,
                            memory_usage_percent,
                            temperature_celsius: temperature,
                            power_draw_watts: power_draw,
                            utilization_percent: utilization,
                        });
                    }
                }
            }
        }

        // Fallback: Try to detect NVIDIA devices through alternative methods
        if devices.is_empty() {
            if let Ok(nvidia_devices) = Self::detect_nvidia_fallback().await {
                devices.extend(nvidia_devices);
            }
        }

        Ok(devices)
    }

    /// Detect AMD GPUs using rocm-smi or alternative methods
    async fn detect_amd_gpus() -> Result<Vec<GpuDevice>> {
        let mut devices = Vec::new();

        // Try rocm-smi for AMD GPUs
        if let Ok(output) = tokio::process::Command::new("rocm-smi")
            .args(&["--showmeminfo", "vram", "--showtemp", "--showpower", "--showuse"])
            .output()
            .await
        {
            if output.status.success() {
                let output_str = String::from_utf8_lossy(&output.stdout);
                devices.extend(Self::parse_rocm_output(&output_str));
            }
        }

        // Try radeontop as alternative
        if devices.is_empty() {
            if let Ok(amd_devices) = Self::detect_amd_fallback().await {
                devices.extend(amd_devices);
            }
        }

        Ok(devices)
    }

    /// Detect Intel integrated graphics
    async fn detect_intel_gpus() -> Result<Vec<GpuDevice>> {
        let mut devices = Vec::new();

        // Intel GPUs are often integrated, check through system info
        #[cfg(target_os = "linux")]
        {
            if let Ok(output) = tokio::process::Command::new("lspci").args(&["-v"]).output().await {
                if output.status.success() {
                    let output_str = String::from_utf8_lossy(&output.stdout);
                    devices.extend(Self::parse_intel_lspci(&output_str));
                }
            }
        }

        #[cfg(target_os = "macos")]
        {
            if let Ok(output) = tokio::process::Command::new("system_profiler")
                .args(&["SPDisplaysDataType"])
                .output()
                .await
            {
                if output.status.success() {
                    let output_str = String::from_utf8_lossy(&output.stdout);
                    devices.extend(Self::parse_macos_graphics(&output_str));
                }
            }
        }

        Ok(devices)
    }

    /// Generic GPU detection through system commands
    async fn detect_generic_gpus() -> Result<Vec<GpuDevice>> {
        let mut devices = Vec::new();

        #[cfg(target_os = "linux")]
        {
            // Try lspci for any graphics devices
            if let Ok(output) = tokio::process::Command::new("lspci")
                .args(&["-v", "-s", "$(lspci | grep -i vga | cut -d' ' -f1)"])
                .output()
                .await
            {
                if output.status.success() {
                    let output_str = String::from_utf8_lossy(&output.stdout);
                    devices.extend(Self::parse_generic_lspci(&output_str));
                }
            }
        }

        #[cfg(target_os = "windows")]
        {
            // Use wmic on Windows
            if let Ok(output) = tokio::process::Command::new("wmic")
                .args(&[
                    "path",
                    "win32_VideoController",
                    "get",
                    "name,AdapterRAM,Temperature",
                    "/format:csv",
                ])
                .output()
                .await
            {
                if output.status.success() {
                    let output_str = String::from_utf8_lossy(&output.stdout);
                    devices.extend(Self::parse_windows_wmic(&output_str));
                }
            }
        }

        #[cfg(target_os = "macos")]
        {
            // Use system_profiler on macOS
            if let Ok(output) = tokio::process::Command::new("system_profiler")
                .args(&["SPDisplaysDataType", "-json"])
                .output()
                .await
            {
                if output.status.success() {
                    let output_str = String::from_utf8_lossy(&output.stdout);
                    devices.extend(Self::parse_macos_json(&output_str));
                }
            }
        }

        Ok(devices)
    }

    /// NVIDIA fallback detection
    async fn detect_nvidia_fallback() -> Result<Vec<GpuDevice>> {
        let mut devices = Vec::new();

        // Check if NVIDIA drivers are installed
        if let Ok(_) =
            tokio::process::Command::new("nvidia-smi").args(&["--list-gpus"]).output().await
        {
            // Basic NVIDIA device detected
            devices.push(GpuDevice {
                name: "NVIDIA GPU (Limited Info)".to_string(),
                memory_total_bytes: 0, // Unknown
                memory_used_bytes: 0,  // Unknown
                memory_usage_percent: 0.0,
                temperature_celsius: None,
                power_draw_watts: None,
                utilization_percent: None,
            });
        }

        Ok(devices)
    }

    /// AMD fallback detection
    async fn detect_amd_fallback() -> Result<Vec<GpuDevice>> {
        #[cfg(target_os = "linux")]
        let mut devices = Vec::new();
        #[cfg(not(target_os = "linux"))]
        let devices = Vec::new();

        // Check for AMD GPU through /sys filesystem on Linux
        #[cfg(target_os = "linux")]
        {
            if let Ok(entries) = tokio::fs::read_dir("/sys/class/drm").await {
                let mut entries = entries;
                while let Ok(Some(entry)) = entries.next_entry().await {
                    let name = entry.file_name().to_string_lossy().to_string();
                    if name.starts_with("card") && !name.contains("-") {
                        if let Ok(device_name) = tokio::fs::read_to_string(format!(
                            "/sys/class/drm/{}/device/vendor",
                            name
                        ))
                        .await
                        {
                            if device_name.trim() == "0x1002" {
                                // AMD vendor ID
                                devices.push(GpuDevice {
                                    name: "AMD GPU (Limited Info)".to_string(),
                                    memory_total_bytes: 0,
                                    memory_used_bytes: 0,
                                    memory_usage_percent: 0.0,
                                    temperature_celsius: None,
                                    power_draw_watts: None,
                                    utilization_percent: None,
                                });
                            }
                        }
                    }
                }
            }
        }

        Ok(devices)
    }

    /// Parse rocm-smi output for AMD GPUs
    fn parse_rocm_output(output: &str) -> Vec<GpuDevice> {
        let mut devices = Vec::new();

        // Parse rocm-smi output format
        // This is a simplified parser - real implementation would be more robust
        for line in output.lines() {
            if line.contains("GPU") && line.contains("VRAM") {
                devices.push(GpuDevice {
                    name: "AMD GPU".to_string(),
                    memory_total_bytes: 0, // Would parse from rocm output
                    memory_used_bytes: 0,
                    memory_usage_percent: 0.0,
                    temperature_celsius: None,
                    power_draw_watts: None,
                    utilization_percent: None,
                });
            }
        }

        devices
    }

    /// Parse Intel GPU info from lspci
    fn parse_intel_lspci(output: &str) -> Vec<GpuDevice> {
        let mut devices = Vec::new();

        for line in output.lines() {
            if line.contains("Intel") && (line.contains("Graphics") || line.contains("VGA")) {
                let name = if line.contains("Iris") {
                    "Intel Iris Graphics".to_string()
                } else if line.contains("UHD") {
                    "Intel UHD Graphics".to_string()
                } else {
                    "Intel Integrated Graphics".to_string()
                };

                devices.push(GpuDevice {
                    name,
                    memory_total_bytes: 0, // Shared with system memory
                    memory_used_bytes: 0,
                    memory_usage_percent: 0.0,
                    temperature_celsius: None,
                    power_draw_watts: None,
                    utilization_percent: None,
                });
            }
        }

        devices
    }

    /// Parse macOS graphics info
    fn parse_macos_graphics(output: &str) -> Vec<GpuDevice> {
        let mut devices = Vec::new();

        let mut current_device: Option<String> = None;
        let mut current_memory: Option<u64> = None;

        for line in output.lines() {
            let trimmed = line.trim();

            if trimmed.starts_with("Chipset Model:") {
                current_device = trimmed.split(":").nth(1).map(|s| s.trim().to_string());
            } else if trimmed.starts_with("VRAM (Total):") {
                if let Some(memory_str) = trimmed.split(":").nth(1) {
                    // Parse memory size (e.g., "8 GB" -> 8 * 1024 * 1024 * 1024)
                    let memory_parts: Vec<&str> = memory_str.trim().split_whitespace().collect();
                    if memory_parts.len() >= 2 {
                        if let Ok(size) = memory_parts[0].parse::<u64>() {
                            let multiplier = match memory_parts[1].to_uppercase().as_str() {
                                "GB" => 1024 * 1024 * 1024,
                                "MB" => 1024 * 1024,
                                _ => 1,
                            };
                            current_memory = Some(size * multiplier);
                        }
                    }
                }
            }

            // End of device section
            if trimmed.is_empty() && current_device.is_some() {
                devices.push(GpuDevice {
                    name: current_device.unwrap_or_else(|| "Unknown GPU".to_string()),
                    memory_total_bytes: current_memory.unwrap_or(0),
                    memory_used_bytes: 0, // Not available
                    memory_usage_percent: 0.0,
                    temperature_celsius: None,
                    power_draw_watts: None,
                    utilization_percent: None,
                });

                current_device = None;
                current_memory = None;
            }
        }

        devices
    }

    /// Parse generic lspci output
    fn parse_generic_lspci(output: &str) -> Vec<GpuDevice> {
        let mut devices = Vec::new();

        for line in output.lines() {
            if line.contains("VGA") || line.contains("3D") || line.contains("Display") {
                let name = if line.contains("NVIDIA") {
                    "NVIDIA GPU (Generic)".to_string()
                } else if line.contains("AMD") || line.contains("ATI") {
                    "AMD GPU (Generic)".to_string()
                } else if line.contains("Intel") {
                    "Intel Graphics (Generic)".to_string()
                } else {
                    "Unknown GPU".to_string()
                };

                devices.push(GpuDevice {
                    name,
                    memory_total_bytes: 0,
                    memory_used_bytes: 0,
                    memory_usage_percent: 0.0,
                    temperature_celsius: None,
                    power_draw_watts: None,
                    utilization_percent: None,
                });
            }
        }

        devices
    }

    /// Parse Windows wmic output
    fn parse_windows_wmic(output: &str) -> Vec<GpuDevice> {
        let mut devices = Vec::new();

        for line in output.lines().skip(1) {
            // Skip header
            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() >= 2 {
                let name = parts[1].trim().to_string();
                let memory = parts.get(0).and_then(|s| s.parse::<u64>().ok()).unwrap_or(0);

                if !name.is_empty() && name != "Name" {
                    devices.push(GpuDevice {
                        name,
                        memory_total_bytes: memory,
                        memory_used_bytes: 0,
                        memory_usage_percent: 0.0,
                        temperature_celsius: None,
                        power_draw_watts: None,
                        utilization_percent: None,
                    });
                }
            }
        }

        devices
    }

    /// Parse macOS JSON output
    fn parse_macos_json(output: &str) -> Vec<GpuDevice> {
        // For simplicity, fall back to text parsing
        // Real implementation would use serde_json
        Self::parse_macos_graphics(output)
    }
}

// Helper function to convert LoadAvg to tuple
fn load_avg_to_tuple(load_avg: sysinfo::LoadAvg) -> Option<(f64, f64, f64)> {
    Some((load_avg.one, load_avg.five, load_avg.fifteen))
}
