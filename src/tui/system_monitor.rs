//! System Monitoring Module
//! 
//! Provides GPU, network, and system resource monitoring capabilities.

use anyhow::{Result};
use sysinfo::{System, Networks};
use std::sync::Arc;
use tokio::sync::RwLock;
use std::time::{Duration, Instant};
use tracing::{debug, info};

#[cfg(feature = "cuda")]
use nvml_wrapper::Nvml;

/// System metrics collected by the monitor
#[derive(Debug, Clone, Default)]
pub struct SystemMetrics {
    pub gpu_usage: f32,
    pub gpu_memory_used: u64,
    pub gpu_memory_total: u64,
    pub gpu_temperature: f32,
    pub network_in_bytes: u64,
    pub network_out_bytes: u64,
    pub network_in_rate: f64, // bytes per second
    pub network_out_rate: f64, // bytes per second
}

/// System monitor for collecting hardware metrics
pub struct SystemMonitor {
    /// Sysinfo system instance
    system: Arc<RwLock<System>>,
    
    /// Network monitoring instance
    networks: Arc<RwLock<Networks>>,
    
    /// NVML handle for GPU monitoring (if available)
    #[cfg(feature = "cuda")]
    nvml: Option<Arc<Nvml>>,
    
    /// Last network measurement for rate calculation
    last_network_measurement: Arc<RwLock<NetworkMeasurement>>,
    
    /// Update interval
    update_interval: Duration,
}

#[derive(Debug, Clone)]
struct NetworkMeasurement {
    timestamp: Instant,
    bytes_in: u64,
    bytes_out: u64,
}

impl Default for NetworkMeasurement {
    fn default() -> Self {
        Self {
            timestamp: Instant::now(),
            bytes_in: 0,
            bytes_out: 0,
        }
    }
}

impl SystemMonitor {
    /// Create a new system monitor
    pub async fn new() -> Result<Self> {
        info!("Initializing system monitor");
        
        let system = Arc::new(RwLock::new(System::new_all()));
        
        #[cfg(feature = "cuda")]
        let nvml = match Nvml::init() {
            Ok(nvml) => {
                info!("NVML initialized successfully for GPU monitoring");
                Some(Arc::new(nvml))
            }
            Err(e) => {
                warn!("Failed to initialize NVML for GPU monitoring: {}", e);
                None
            }
        };
        
        let networks = Arc::new(RwLock::new(Networks::new_with_refreshed_list()));
        
        Ok(Self {
            system,
            networks,
            #[cfg(feature = "cuda")]
            nvml,
            last_network_measurement: Arc::new(RwLock::new(NetworkMeasurement::default())),
            update_interval: Duration::from_secs(1),
        })
    }
    
    /// Collect current system metrics
    pub async fn collect_metrics(&self) -> Result<SystemMetrics> {
        let mut metrics = SystemMetrics::default();
        
        // Collect GPU metrics
        #[cfg(feature = "cuda")]
        if let Some(ref nvml) = self.nvml {
            match self.collect_gpu_metrics(nvml).await {
                Ok(gpu_metrics) => {
                    metrics.gpu_usage = gpu_metrics.0;
                    metrics.gpu_memory_used = gpu_metrics.1;
                    metrics.gpu_memory_total = gpu_metrics.2;
                    metrics.gpu_temperature = gpu_metrics.3;
                }
                Err(e) => {
                    debug!("Failed to collect GPU metrics: {}", e);
                }
            }
        }
        
        // Collect network metrics
        let network_metrics = self.collect_network_metrics().await?;
        metrics.network_in_bytes = network_metrics.0;
        metrics.network_out_bytes = network_metrics.1;
        metrics.network_in_rate = network_metrics.2;
        metrics.network_out_rate = network_metrics.3;
        
        Ok(metrics)
    }
    
    /// Collect GPU metrics using NVML
    #[cfg(feature = "cuda")]
    async fn collect_gpu_metrics(&self, nvml: &Nvml) -> Result<(f32, u64, u64, f32)> {
        // Get the first GPU device
        let device = nvml.device_by_index(0)
            .context("Failed to get GPU device")?;
        
        // Get utilization
        let utilization = device.utilization_rates()
            .context("Failed to get GPU utilization")?;
        
        // Get memory info
        let memory_info = device.memory_info()
            .context("Failed to get GPU memory info")?;
        
        // Get temperature (optional, may fail on some GPUs)
        let temperature = device.temperature(nvml_wrapper::enum_wrappers::device::TemperatureSensor::Gpu)
            .unwrap_or(0);
        
        Ok((
            utilization.gpu as f32,
            memory_info.used,
            memory_info.total,
            temperature as f32,
        ))
    }
    
    /// Collect network metrics
    async fn collect_network_metrics(&self) -> Result<(u64, u64, f64, f64)> {
        let mut networks = self.networks.write().await;
        networks.refresh(true);
        
        // Calculate total bytes
        let mut total_in = 0u64;
        let mut total_out = 0u64;
        
        for (interface_name, network) in networks.iter() {
            debug!("Network interface {}: rx={}, tx={}", 
                interface_name, 
                network.received(), 
                network.transmitted()
            );
            total_in += network.received();
            total_out += network.transmitted();
        }
        
        // Calculate rates
        let mut last_measurement = self.last_network_measurement.write().await;
        let now = Instant::now();
        let elapsed = now.duration_since(last_measurement.timestamp).as_secs_f64();
        
        let in_rate = if elapsed > 0.0 && total_in >= last_measurement.bytes_in {
            ((total_in - last_measurement.bytes_in) as f64) / elapsed
        } else if elapsed > 0.0 && total_in < last_measurement.bytes_in {
            // Counter may have reset, just use the current value
            (total_in as f64) / elapsed
        } else {
            0.0
        };
        
        let out_rate = if elapsed > 0.0 && total_out >= last_measurement.bytes_out {
            ((total_out - last_measurement.bytes_out) as f64) / elapsed
        } else if elapsed > 0.0 && total_out < last_measurement.bytes_out {
            // Counter may have reset, just use the current value
            (total_out as f64) / elapsed
        } else {
            0.0
        };
        
        // Update last measurement
        *last_measurement = NetworkMeasurement {
            timestamp: now,
            bytes_in: total_in,
            bytes_out: total_out,
        };
        
        Ok((total_in, total_out, in_rate, out_rate))
    }
    
    /// Get GPU usage percentage (0.0 to 100.0)
    pub async fn get_gpu_usage(&self) -> f32 {
        #[cfg(feature = "cuda")]
        if let Some(ref nvml) = self.nvml {
            if let Ok(device) = nvml.device_by_index(0) {
                if let Ok(utilization) = device.utilization_rates() {
                    return utilization.gpu as f32;
                }
            }
        }
        0.0
    }
    
    /// Get network I/O rates in bytes per second
    pub async fn get_network_rates(&self) -> (f64, f64) {
        if let Ok(metrics) = self.collect_network_metrics().await {
            (metrics.2, metrics.3)
        } else {
            (0.0, 0.0)
        }
    }
    
    /// Start background monitoring task
    pub async fn start_monitoring(&self) -> tokio::task::JoinHandle<()> {
        let monitor = self.clone();
        let interval = self.update_interval;
        
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);
            loop {
                ticker.tick().await;
                
                if let Ok(metrics) = monitor.collect_metrics().await {
                    debug!(
                        "System metrics - GPU: {:.1}%, Network In: {:.2} KB/s, Out: {:.2} KB/s",
                        metrics.gpu_usage,
                        metrics.network_in_rate / 1024.0,
                        metrics.network_out_rate / 1024.0
                    );
                }
            }
        })
    }
}

impl Clone for SystemMonitor {
    fn clone(&self) -> Self {
        Self {
            system: self.system.clone(),
            networks: self.networks.clone(),
            #[cfg(feature = "cuda")]
            nvml: self.nvml.clone(),
            last_network_measurement: self.last_network_measurement.clone(),
            update_interval: self.update_interval,
        }
    }
}

/// Helper function to format bytes into human-readable string
pub fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_index = 0;
    
    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }
    
    if unit_index == 0 {
        format!("{} {}", size as u64, UNITS[unit_index])
    } else {
        format!("{:.2} {}", size, UNITS[unit_index])
    }
}

/// Helper function to format rate into human-readable string
pub fn format_rate(bytes_per_second: f64) -> String {
    format_bytes(bytes_per_second as u64) + "/s"
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_system_monitor_creation() {
        let monitor = SystemMonitor::new().await;
        assert!(monitor.is_ok());
    }
    
    #[tokio::test]
    async fn test_collect_metrics() {
        let monitor = SystemMonitor::new().await.unwrap();
        let metrics = monitor.collect_metrics().await;
        assert!(metrics.is_ok());
    }
    
    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1536), "1.50 KB");
        assert_eq!(format_bytes(1048576), "1.00 MB");
        assert_eq!(format_bytes(1073741824), "1.00 GB");
    }
    
    #[test]
    fn test_format_rate() {
        assert_eq!(format_rate(1024.0), "1.00 KB/s");
        assert_eq!(format_rate(1048576.0), "1.00 MB/s");
    }
}