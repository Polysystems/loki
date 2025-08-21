//! Real-time system metrics collection

use std::sync::Arc;
use tokio::sync::RwLock;
use sysinfo::{System, Disks, Networks, Components};
use anyhow::Result;
use tracing::debug;

/// System metrics collector
pub struct SystemMetrics {
    system: Arc<RwLock<System>>,
    disks: Arc<RwLock<Disks>>,
    networks: Arc<RwLock<Networks>>,
    components: Arc<RwLock<Components>>,
}

impl SystemMetrics {
    pub fn new() -> Self {
        let mut system = System::new_all();
        system.refresh_all();
        
        let disks = Disks::new_with_refreshed_list();
        let networks = Networks::new_with_refreshed_list();
        let components = Components::new_with_refreshed_list();
        
        Self {
            system: Arc::new(RwLock::new(system)),
            disks: Arc::new(RwLock::new(disks)),
            networks: Arc::new(RwLock::new(networks)),
            components: Arc::new(RwLock::new(components)),
        }
    }
    
    /// Get current CPU usage (percentage)
    pub async fn get_cpu_usage(&self) -> f32 {
        let mut system = self.system.write().await;
        system.refresh_cpu_usage();
        
        // Calculate average CPU usage across all cores
        let cpu_count = system.cpus().len() as f32;
        let total_usage: f32 = system.cpus().iter().map(|cpu| cpu.cpu_usage()).sum();
        
        if cpu_count > 0.0 {
            total_usage / cpu_count
        } else {
            0.0
        }
    }
    
    /// Get CPU usage per core
    pub async fn get_cpu_per_core(&self) -> Vec<f32> {
        let mut system = self.system.write().await;
        system.refresh_cpu_usage();
        
        system.cpus().iter().map(|cpu| cpu.cpu_usage()).collect()
    }
    
    /// Get memory usage (used, total) in bytes
    pub async fn get_memory_usage(&self) -> (u64, u64) {
        let mut system = self.system.write().await;
        system.refresh_memory();
        
        let used = system.used_memory();
        let total = system.total_memory();
        
        (used, total)
    }
    
    /// Get swap usage (used, total) in bytes
    pub async fn get_swap_usage(&self) -> (u64, u64) {
        let mut system = self.system.write().await;
        system.refresh_memory();
        
        let used = system.used_swap();
        let total = system.total_swap();
        
        (used, total)
    }
    
    /// Get disk usage for all disks
    pub async fn get_disk_usage(&self) -> Vec<DiskInfo> {
        let mut disks = self.disks.write().await;
        disks.refresh(true);
        
        disks.iter()
            .map(|disk| DiskInfo {
                name: disk.name().to_string_lossy().to_string(),
                mount_point: disk.mount_point().to_string_lossy().to_string(),
                total_space: disk.total_space(),
                available_space: disk.available_space(),
                used_space: disk.total_space() - disk.available_space(),
            })
            .collect()
    }
    
    /// Get network statistics (bytes received, bytes sent)
    pub async fn get_network_stats(&self) -> (u64, u64) {
        let mut networks = self.networks.write().await;
        networks.refresh(true);
        
        let mut total_received = 0u64;
        let mut total_transmitted = 0u64;
        
        for (_, data) in networks.iter() {
            total_received += data.total_received();
            total_transmitted += data.total_transmitted();
        }
        
        (total_received, total_transmitted)
    }
    
    /// Get process count
    pub async fn get_process_count(&self) -> usize {
        let mut system = self.system.write().await;
        system.refresh_processes(sysinfo::ProcessesToUpdate::All, true);
        
        system.processes().len()
    }
    
    /// Get top processes by CPU usage
    pub async fn get_top_processes_by_cpu(&self, limit: usize) -> Vec<ProcessInfo> {
        let mut system = self.system.write().await;
        system.refresh_processes(sysinfo::ProcessesToUpdate::All, true);
        
        let mut processes: Vec<ProcessInfo> = system.processes()
            .iter()
            .map(|(pid, process)| ProcessInfo {
                pid: pid.as_u32(),
                name: process.name().to_string_lossy().to_string(),
                cpu_usage: process.cpu_usage(),
                memory_usage: process.memory(),
                status: format!("{:?}", process.status()),
            })
            .collect();
        
        // Sort by CPU usage (descending)
        processes.sort_by(|a, b| b.cpu_usage.partial_cmp(&a.cpu_usage).unwrap());
        processes.truncate(limit);
        
        processes
    }
    
    /// Get top processes by memory usage
    pub async fn get_top_processes_by_memory(&self, limit: usize) -> Vec<ProcessInfo> {
        let mut system = self.system.write().await;
        system.refresh_processes(sysinfo::ProcessesToUpdate::All, true);
        
        let mut processes: Vec<ProcessInfo> = system.processes()
            .iter()
            .map(|(pid, process)| ProcessInfo {
                pid: pid.as_u32(),
                name: process.name().to_string_lossy().to_string(),
                cpu_usage: process.cpu_usage(),
                memory_usage: process.memory(),
                status: format!("{:?}", process.status()),
            })
            .collect();
        
        // Sort by memory usage (descending)
        processes.sort_by(|a, b| b.memory_usage.cmp(&a.memory_usage));
        processes.truncate(limit);
        
        processes
    }
    
    /// Get system uptime in seconds
    pub async fn get_uptime(&self) -> u64 {
        let system = self.system.read().await;
        sysinfo::System::uptime()
    }
    
    /// Get system load average (1, 5, 15 minutes)
    pub async fn get_load_average(&self) -> (f64, f64, f64) {
        let system = self.system.read().await;
        let load_avg = sysinfo::System::load_average();
        (load_avg.one, load_avg.five, load_avg.fifteen)
    }
    
    /// Refresh all system information
    pub async fn refresh_all(&self) {
        let mut system = self.system.write().await;
        system.refresh_all();
        
        let mut disks = self.disks.write().await;
        disks.refresh(true);
        
        let mut networks = self.networks.write().await;
        networks.refresh(true);
        
        let mut components = self.components.write().await;
        components.refresh(true);
        
        debug!("Refreshed all system metrics");
    }
}

#[derive(Debug, Clone)]
pub struct DiskInfo {
    pub name: String,
    pub mount_point: String,
    pub total_space: u64,
    pub available_space: u64,
    pub used_space: u64,
}

#[derive(Debug, Clone)]
pub struct ProcessInfo {
    pub pid: u32,
    pub name: String,
    pub cpu_usage: f32,
    pub memory_usage: u64,
    pub status: String,
}

impl Default for SystemMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Format bytes to human readable string
pub fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_index = 0;
    
    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }
    
    format!("{:.2} {}", size, UNITS[unit_index])
}

/// Format percentage
pub fn format_percentage(value: f32) -> String {
    format!("{:.1}%", value)
}

/// Format uptime to human readable string
pub fn format_uptime(seconds: u64) -> String {
    let days = seconds / 86400;
    let hours = (seconds % 86400) / 3600;
    let minutes = (seconds % 3600) / 60;
    
    if days > 0 {
        format!("{}d {}h {}m", days, hours, minutes)
    } else if hours > 0 {
        format!("{}h {}m", hours, minutes)
    } else {
        format!("{}m", minutes)
    }
}