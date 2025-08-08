use serde::{Deserialize, Serialize};

/// Device types supported by the compute backend
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceType {
    Cpu,
    Cuda,
    Metal,
    OpenCL,
}

/// Information about a compute device
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceInfo {
    pub compute_capability: String,
    pub memory_total: usize,
    pub max_threads: usize,
}

/// Represents a compute device
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Device {
    pub id: String,
    pub name: String,
    pub device_type: DeviceType,
    pub info: DeviceInfo,
}

impl Device {
    /// Check if this is a GPU device
    pub fn is_gpu(&self) -> bool {
        matches!(self.device_type, DeviceType::Cuda | DeviceType::Metal | DeviceType::OpenCL)
    }

    /// Get device memory in MB
    pub fn memory_mb(&self) -> usize {
        self.info.memory_total / 1024 / 1024
    }

    /// Get a unique key for this device
    pub fn key(&self) -> String {
        self.id.clone()
    }
}
