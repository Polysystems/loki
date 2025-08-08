//! Real-time system and GPU metrics collector for TUI
//!
//! This module provides actual system metrics collection including CPU, memory, disk, network,
//! and GPU statistics for the Real-time Integration view in the TUI.

use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::Result;
use sysinfo::{ Pid,System};
use tokio::sync::RwLock;

#[cfg(feature = "gpu-nvidia")]
use nvml_wrapper::Nvml;

use crate::monitoring::real_time::{
    CpuMetrics, DiskInfo, DiskMetrics, GpuDevice, GpuMetrics, MemoryMetrics,
    NetworkInterface, NetworkMetrics, ProcessMetrics, ProcessStatus, SystemInfo, SystemMetrics,
};

/// Real-time metrics collector for system resources
pub struct RealTimeMetricsCollector {
    /// System information collector
    system: Arc<RwLock<System>>,

    /// Last network stats for calculating rates
    last_network_stats: Arc<RwLock<NetworkStats>>,

    /// Last disk IO stats for calculating rates
    last_disk_io_stats: Arc<RwLock<DiskIOStats>>,

    /// Last update time
    last_update: Arc<RwLock<Instant>>,

    /// Process ID for self-monitoring
    pid: Pid,

    /// NVIDIA Management Library handle (if available)
    #[cfg(feature = "gpu-nvidia")]
    nvml: Option<Arc<Nvml>>,
}

impl std::fmt::Debug for RealTimeMetricsCollector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RealTimeMetricsCollector")
            .field("pid", &self.pid)
            .field("has_nvml", &cfg!(feature = "gpu-nvidia"))
            .finish()
    }
}

#[derive(Debug, Clone)]
struct NetworkStats {
    total_received: u64,
    total_sent: u64,
    timestamp: Instant,
}

impl Default for NetworkStats {
    fn default() -> Self {
        Self {
            total_received: 0,
            total_sent: 0,
            timestamp: Instant::now(),
        }
    }
}

#[derive(Debug, Clone)]
struct DiskIOStats {
    read_bytes: u64,
    write_bytes: u64,
    timestamp: Instant,
}

impl Default for DiskIOStats {
    fn default() -> Self {
        Self {
            read_bytes: 0,
            write_bytes: 0,
            timestamp: Instant::now(),
        }
    }
}

impl RealTimeMetricsCollector {
    /// Create a new metrics collector
    pub fn new() -> Self {
        let mut system = System::new_all();
        system.refresh_all();

        // Initialize NVML for GPU monitoring if available
        #[cfg(feature = "gpu-nvidia")]
        let nvml = match Nvml::init() {
            Ok(nvml) => {
                println!("✅ NVIDIA GPU monitoring initialized");
                Some(Arc::new(nvml))
            }
            Err(e) => {
                eprintln!("NVIDIA GPU monitoring not available: {}", e);
                None
            }
        };
        Self {
            system: Arc::new(RwLock::new(system)),
            last_network_stats: Arc::new(RwLock::new(NetworkStats::default())),
            last_disk_io_stats: Arc::new(RwLock::new(DiskIOStats::default())),
            last_update: Arc::new(RwLock::new(Instant::now())),
            pid: Pid::from_u32(std::process::id()),
            #[cfg(feature = "gpu-nvidia")]
            nvml,
        }
    }

    /// Collect current system metrics
    pub async fn collect_metrics(&self) -> Result<SystemMetrics> {
        let mut system = self.system.write().await;

        // Refresh system data
        system.refresh_all();
        system.refresh_cpu_all();
        system.refresh_memory();
        system.refresh_all(); // Updated API combines disk and network refresh
        system.refresh_processes(sysinfo::ProcessesToUpdate::All, true);

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Collect system info
        let system_info = self.collect_system_info(&system).await;

        // Collect CPU metrics
        let cpu_metrics = self.collect_cpu_metrics(&system).await;

        // Collect memory metrics
        let memory_metrics = self.collect_memory_metrics(&system).await;

        // Collect disk metrics
        let disk_metrics = self.collect_disk_metrics(&system).await?;

        // Collect network metrics
        let network_metrics = self.collect_network_metrics(&system).await?;

        // Collect process metrics
        let process_metrics = self.collect_process_metrics(&system).await;

        // Collect GPU metrics if available
        let gpu_metrics = self.collect_gpu_metrics().await;

        // Update last update time
        *self.last_update.write().await = Instant::now();

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

    /// Collect system information
    async fn collect_system_info(&self, system: &System) -> SystemInfo {
        let hostname = System::host_name().unwrap_or_else(|| "unknown".to_string());
        let os_name = System::name().unwrap_or_else(|| "unknown".to_string());
        let os_version = System::os_version().unwrap_or_else(|| "unknown".to_string());
        let kernel_version = System::kernel_version().unwrap_or_else(|| "unknown".to_string());
        let uptime = System::uptime();
        let boot_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
            .saturating_sub(uptime);

        SystemInfo {
            hostname,
            os_name,
            os_version,
            kernel_version,
            uptime,
            boot_time,
        }
    }

    /// Collect CPU metrics
    async fn collect_cpu_metrics(&self, system: &System) -> CpuMetrics {
        let usage_percent = system.global_cpu_usage();
        let cpus = system.cpus();
        let core_count = cpus.len();
        let frequency_mhz = if !cpus.is_empty() {
            cpus[0].frequency()
        } else {
            0
        };

        let per_core_usage: Vec<f32> = cpus
            .iter()
            .map(|cpu| cpu.cpu_usage())
            .collect();

        // Temperature might not be available on all systems
        let temperature_celsius = None; // Would require platform-specific implementation

        // Load average (Unix-like systems)
        let load_avg = System::load_average();
        let load_average = Some((load_avg.one, load_avg.five, load_avg.fifteen));

        CpuMetrics {
            usage_percent,
            core_count,
            frequency_mhz,
            per_core_usage,
            temperature_celsius,
            load_average,
        }
    }

    /// Collect memory metrics
    async fn collect_memory_metrics(&self, system: &System) -> MemoryMetrics {
        let total_bytes = system.total_memory();
        let used_bytes = system.used_memory();
        let available_bytes = system.available_memory();
        let usage_percent = (used_bytes as f32 / total_bytes as f32) * 100.0;

        let swap_total_bytes = system.total_swap();
        let swap_used_bytes = system.used_swap();
        let swap_usage_percent = if swap_total_bytes > 0 {
            (swap_used_bytes as f32 / swap_total_bytes as f32) * 100.0
        } else {
            0.0
        };

        // Buffer/cache calculation (approximation)
        let buffer_cache_bytes = system.free_memory().saturating_sub(available_bytes);

        MemoryMetrics {
            total_bytes,
            used_bytes,
            available_bytes,
            usage_percent,
            swap_total_bytes,
            swap_used_bytes,
            swap_usage_percent,
            buffer_cache_bytes,
        }
    }

    /// Collect disk metrics
    async fn collect_disk_metrics(&self, system: &System) -> Result<DiskMetrics> {
        let mut total_space_bytes = 0u64;
        let mut used_space_bytes = 0u64;
        let mut available_space_bytes = 0u64;

        let disks: Vec<DiskInfo> = sysinfo::Disks::new_with_refreshed_list()
            .iter()
            .map(|disk| {
                let total = disk.total_space();
                let available = disk.available_space();
                let used = total - available;
                let usage_percent = if total > 0 {
                    (used as f32 / total as f32) * 100.0
                } else {
                    0.0
                };

                total_space_bytes += total;
                available_space_bytes += available;
                used_space_bytes += used;

                DiskInfo {
                    name: disk.name().to_string_lossy().to_string(),
                    mount_point: disk.mount_point().to_string_lossy().to_string(),
                    file_system: disk.file_system().to_string_lossy().to_string(),
                    total_space_bytes: total,
                    available_space_bytes: available,
                    usage_percent,
                    is_removable: disk.is_removable(),
                }
            })
            .collect();

        let usage_percent = if total_space_bytes > 0 {
            (used_space_bytes as f32 / total_space_bytes as f32) * 100.0
        } else {
            0.0
        };

        // Calculate IO rates (would need platform-specific implementation for accurate values)
        let now = Instant::now();
        let mut last_io = self.last_disk_io_stats.write().await;
        let elapsed = now.duration_since(last_io.timestamp).as_secs_f64();

        // These are placeholder values - real implementation would need platform-specific code
        let io_read_bytes_per_sec = if elapsed > 0.0 {
            0 // Would calculate from actual disk IO stats
        } else {
            0
        };

        let io_write_bytes_per_sec = if elapsed > 0.0 {
            0 // Would calculate from actual disk IO stats
        } else {
            0
        };

        last_io.timestamp = now;

        Ok(DiskMetrics {
            disks,
            total_space_bytes,
            used_space_bytes,
            available_space_bytes,
            usage_percent,
            io_read_bytes_per_sec,
            io_write_bytes_per_sec,
        })
    }

    /// Collect network metrics
    async fn collect_network_metrics(&self, system: &System) -> Result<NetworkMetrics> {
        let networks = sysinfo::Networks::new_with_refreshed_list();
        let mut total_bytes_received = 0u64;
        let mut total_bytes_sent = 0u64;
        let mut total_packets_received = 0u64;
        let mut total_packets_sent = 0u64;

        let interfaces: Vec<NetworkInterface> = networks
            .iter()
            .map(|(name, data)| {
                let received = data.total_received();
                let sent = data.total_transmitted();
                let packets_received = data.total_packets_received();
                let packets_sent = data.total_packets_transmitted();
                let errors_received = data.total_errors_on_received();
                let errors_sent = data.total_errors_on_transmitted();

                total_bytes_received += received;
                total_bytes_sent += sent;
                total_packets_received += packets_received;
                total_packets_sent += packets_sent;

                NetworkInterface {
                    name: name.clone(),
                    bytes_received: received,
                    bytes_sent: sent,
                    packets_received,
                    packets_sent,
                    errors_received,
                    errors_sent,
                    is_up: true, // sysinfo doesn't provide this directly
                }
            })
            .collect();

        // Calculate rates
        let now = Instant::now();
        let mut last_stats = self.last_network_stats.write().await;
        let elapsed = now.duration_since(last_stats.timestamp).as_secs_f64();

        let bytes_received_per_sec = if elapsed > 0.0 {
            ((total_bytes_received.saturating_sub(last_stats.total_received)) as f64 / elapsed) as u64
        } else {
            0
        };

        let bytes_sent_per_sec = if elapsed > 0.0 {
            ((total_bytes_sent.saturating_sub(last_stats.total_sent)) as f64 / elapsed) as u64
        } else {
            0
        };

        // Update last stats
        last_stats.total_received = total_bytes_received;
        last_stats.total_sent = total_bytes_sent;
        last_stats.timestamp = now;

        Ok(NetworkMetrics {
            interfaces,
            total_bytes_received,
            total_bytes_sent,
            total_packets_received,
            total_packets_sent,
            bytes_received_per_sec,
            bytes_sent_per_sec,
        })
    }

    /// Collect process metrics for the current Loki process
    async fn collect_process_metrics(&self, system: &System) -> ProcessMetrics {
        if let Some(process) = system.process(self.pid) {
            let cpu_usage_percent = process.cpu_usage();
            let memory_usage_bytes = process.memory();
            let virtual_memory_bytes = process.virtual_memory();
            let total_memory = system.total_memory();
            let memory_usage_percent = (memory_usage_bytes as f32 / total_memory as f32) * 100.0;

            // Get thread count - sysinfo 0.36 doesn't expose tasks directly
            let thread_count = 1; // Default to 1, actual count would need platform-specific code

            // File handles would require platform-specific implementation
            let file_handles = 0;

            let uptime_seconds = process.run_time();

            let status = match process.status() {
                sysinfo::ProcessStatus::Run => ProcessStatus::Running,
                sysinfo::ProcessStatus::Sleep => ProcessStatus::Sleeping,
                sysinfo::ProcessStatus::Stop => ProcessStatus::Stopped,
                sysinfo::ProcessStatus::Zombie => ProcessStatus::Zombie,
                _ => ProcessStatus::Unknown,
            };

            ProcessMetrics {
                pid: self.pid.as_u32(),
                name: process.name().to_string_lossy().to_string(),
                cpu_usage_percent,
                memory_usage_bytes,
                memory_usage_percent,
                virtual_memory_bytes,
                thread_count,
                file_handles,
                uptime_seconds,
                status,
            }
        } else {
            // Fallback if process not found
            ProcessMetrics {
                pid: self.pid.as_u32(),
                name: "loki".to_string(),
                cpu_usage_percent: 0.0,
                memory_usage_bytes: 0,
                memory_usage_percent: 0.0,
                virtual_memory_bytes: 0,
                thread_count: 1,
                file_handles: 0,
                uptime_seconds: 0,
                status: ProcessStatus::Unknown,
            }
        }
    }

    /// Collect GPU metrics if available
    async fn collect_gpu_metrics(&self) -> Option<GpuMetrics> {
        #[cfg(feature = "gpu-nvidia")]
        {
            if let Some(nvml) = &self.nvml {
                match self.collect_nvidia_gpu_metrics(nvml).await {
                    Ok(metrics) => return Some(metrics),
                    Err(e) => {
                        eprintln!("Failed to collect NVIDIA GPU metrics: {}", e);
                    }
                }
            }
        }

        // Check for Apple Metal GPU on macOS
        #[cfg(target_os = "macos")]
        {
            if let Some(metrics) = self.collect_apple_gpu_metrics().await {
                return Some(metrics);
            }
        }

        None
    }

    #[cfg(feature = "gpu-nvidia")]
    async fn collect_nvidia_gpu_metrics(&self, nvml: &Nvml) -> Result<GpuMetrics> {
        let device_count = nvml.device_count()?;
        let mut devices = Vec::new();

        for i in 0..device_count {
            let device = nvml.device_by_index(i)?;
            let name = device.name()?;

            let memory_info = device.memory_info()?;
            let memory_total_bytes = memory_info.total;
            let memory_used_bytes = memory_info.used;
            let memory_usage_percent = (memory_used_bytes as f32 / memory_total_bytes as f32) * 100.0;

            let temperature_celsius = device.temperature(nvml_wrapper::enum_wrappers::device::TemperatureSensor::Gpu)
                .ok()
                .map(|t| t as f32);

            let power_draw_watts = device.power_usage()
                .ok()
                .map(|p| p as f32 / 1000.0); // Convert milliwatts to watts

            let utilization = device.utilization_rates()?;
            let utilization_percent = Some(utilization.gpu as f32);

            devices.push(GpuDevice {
                name,
                memory_total_bytes,
                memory_used_bytes,
                memory_usage_percent,
                temperature_celsius,
                power_draw_watts,
                utilization_percent,
            });
        }

        Ok(GpuMetrics { devices })
    }

    #[cfg(target_os = "macos")]
    async fn collect_apple_gpu_metrics(&self) -> Option<GpuMetrics> {
        // Apple GPU metrics would require Metal Performance Shaders or IOKit
        // This is a placeholder that returns mock data for M1/M2/M3 chips

        // Check if we're on Apple Silicon
        if std::env::consts::ARCH == "aarch64" {
            Some(GpuMetrics {
                devices: vec![GpuDevice {
                    name: "Apple GPU".to_string(),
                    memory_total_bytes: 0, // Unified memory, not separately reported
                    memory_used_bytes: 0,
                    memory_usage_percent: 0.0,
                    temperature_celsius: None,
                    power_draw_watts: None,
                    utilization_percent: None,
                }],
            })
        } else {
            None
        }
    }

    /// Get a snapshot of current metrics without full refresh
    pub async fn get_current_snapshot(&self) -> Result<SystemMetrics> {
        self.collect_metrics().await
    }
}

/// Extension methods for integrating with RealTimeMetricsAggregator
impl RealTimeMetricsCollector {
    /// Convert collected metrics to the format expected by RealTimeMetricsAggregator
    pub async fn get_system_metrics(&self) -> Result<SystemMetrics> {
        self.collect_metrics().await
    }
}
