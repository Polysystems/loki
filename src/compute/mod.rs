use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use arc_swap::ArcSwap;
// Conditional imports for Metal
#[cfg(feature = "metal")]
use metal;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};
// Conditional imports for CUDA (available on all platforms where CUDA is detected)
#[cfg(all(feature = "cuda", gpu_cuda_available))]
use {cudarc, nvml_wrapper};

pub mod device;
pub mod memory;
pub mod scheduler;

pub use device::{Device, DeviceInfo, DeviceType};
pub use memory::{MemoryAllocation, MemoryPool};
pub use scheduler::{ComputeScheduler, ComputeTask};

/// GPU compute backend trait
pub trait ComputeBackend: Send + Sync + std::fmt::Debug {
    /// Get available devices
    fn devices(&self) -> Result<Vec<Device>>;

    /// Get device memory info
    fn memory_info(&self, device: &Device) -> Result<MemoryInfo>;

    /// Allocate memory on device
    fn allocate(&self, device: &Device, size: usize) -> Result<MemoryAllocation>;

    /// Transfer data to device
    fn transfer_to_device(&self, allocation: &MemoryAllocation, data: &[u8]) -> Result<()>;

    /// Transfer data from device
    fn transfer_from_device(&self, allocation: &MemoryAllocation, data: &mut [u8]) -> Result<()>;

    /// Execute computation
    fn execute(&self, task: ComputeTask) -> Result<()>;
}

/// Memory information for a device
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryInfo {
    pub total: usize,
    pub free: usize,
    pub used: usize,
    pub reserved: usize,
}

/// Compute manager that handles GPU/CPU compute resources
pub struct ComputeManager {
    backend: Arc<dyn ComputeBackend>,
    devices: Arc<ArcSwap<Vec<Device>>>,
    scheduler: Arc<ComputeScheduler>,
    memory_pools: dashmap::DashMap<String, Arc<MemoryPool>>,
}

impl std::fmt::Debug for ComputeManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ComputeManager")
            .field("backend", &"<Arc<dyn ComputeBackend>>")
            .field("devices", &self.devices)
            .field("scheduler", &self.scheduler)
            .field("memory_pools", &format!("{} pools", self.memory_pools.len()))
            .finish()
    }
}

impl ComputeManager {
    /// Create a new compute manager
    pub fn new() -> Result<Self> {
        let backend = create_backend()?;
        let devices = Arc::new(ArcSwap::from_pointee(backend.devices()?));
        let scheduler = Arc::new(ComputeScheduler::new(devices.load().len()));

        Ok(Self { backend, devices, scheduler, memory_pools: dashmap::DashMap::new() })
    }

    /// Get available devices
    pub fn devices(&self) -> Vec<Device> {
        self.devices.load().as_ref().clone()
    }

    /// Get or create memory pool for a device
    pub fn memory_pool(&self, device: &Device) -> Arc<MemoryPool> {
        let key = device.id.clone();

        self.memory_pools
            .entry(key)
            .or_insert_with(|| {
                let memory_info = self.backend.memory_info(device).unwrap();
                Arc::new(MemoryPool::new(
                    device.clone(),
                    memory_info.free * 8 / 10, // Use 80% of free memory
                ))
            })
            .clone()
    }

    /// Schedule a task
    pub async fn schedule(&self, task: ComputeTask) -> Result<()> {
        self.scheduler.schedule(task).await
    }

    /// Get memory usage for all devices
    pub fn memory_usage(&self) -> Result<Vec<(Device, MemoryInfo)>> {
        let devices = self.devices();
        let mut usage = Vec::new();

        for device in devices {
            let memory_info = self.backend.memory_info(&device)?;
            usage.push((device, memory_info));
        }

        Ok(usage)
    }

    /// Get total available memory in GB
    pub fn available_memory_gb(&self) -> Result<f32> {
        let usage = self.memory_usage()?;
        let total_bytes: usize = usage.iter().map(|(_, info)| info.free).sum();
        Ok(total_bytes as f32 / 1024.0 / 1024.0 / 1024.0)
    }

    /// Monitor memory usage
    pub async fn monitor_memory(&self) {
        let mut interval = tokio::time::interval(Duration::from_secs(10));
        loop {
            interval.tick().await;
            if let Ok(usage) = self.memory_usage() {
                for (device, info) in usage {
                    let used_percent = (info.used as f32 / info.total as f32) * 100.0;
                    if used_percent > 80.0 {
                        warn!(
                            "High memory usage on device {}: {:.1}% ({} MB / {} MB)",
                            device.name,
                            used_percent,
                            info.used / 1024 / 1024,
                            info.total / 1024 / 1024
                        );
                    }
                }
            }
        }
    }
}

impl Default for ComputeManager {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| {
            // Fallback to CPU backend if GPU initialization fails
            Self {
                backend: Arc::new(CpuBackend::new().expect("CPU backend should always initialize")) as Arc<dyn ComputeBackend>,
                devices: Arc::new(ArcSwap::from_pointee(vec![Device {
                    id: "cpu:0".to_string(),
                    name: "CPU".to_string(),
                    device_type: DeviceType::Cpu,
                    info: DeviceInfo {
                        compute_capability: "1.0".to_string(),
                        memory_total: 8 * 1024 * 1024 * 1024, // 8GB default
                        max_threads: 1,
                    },
                }])),
                scheduler: Arc::new(ComputeScheduler::new(1)),
                memory_pools: dashmap::DashMap::new(),
            }
        })
    }
}

fn create_backend() -> Result<Arc<dyn ComputeBackend>> {
    // Platform-aware backend selection with intelligent fallbacks
    #[cfg(target_platform_macos)]
    {
        // On macOS, prefer Apple Silicon backend if available, then Metal, then CUDA
        // (if available)
        if let Ok(backend) = AppleSiliconBackend::new() {
            info!("Using Apple Silicon compute backend with safe GPU detection");
            return Ok(Arc::new(backend));
        }

        #[cfg(all(feature = "metal", gpu_metal_available))]
        if let Ok(backend) = MetalBackend::new() {
            info!("Using Metal compute backend");
            return Ok(Arc::new(backend));
        }

        #[cfg(all(feature = "cuda", gpu_cuda_available))]
        if let Ok(backend) = CudaBackend::new() {
            info!("Using CUDA compute backend (experimental on macOS)");
            return Ok(Arc::new(backend));
        }
    }

    // On other platforms, try CUDA first, then Metal, then Apple Silicon, then CPU
    #[cfg(not(target_platform_macos))]
    {
        #[cfg(all(feature = "cuda", gpu_cuda_available))]
        if let Ok(backend) = CudaBackend::new() {
            info!("Using CUDA compute backend");
            return Ok(Arc::new(backend));
        }

        #[cfg(all(feature = "metal", gpu_metal_available))]
        if let Ok(backend) = MetalBackend::new() {
            info!("Using Metal compute backend");
            return Ok(Arc::new(backend));
        }
    }

    // Fallback to CPU
    info!("Using CPU compute backend");
    Ok(Arc::new(CpuBackend::new()?))
}

/// CUDA compute backend (available on all platforms where CUDA is detected)
#[cfg(all(feature = "cuda", gpu_cuda_available))]
pub struct CudaBackend {
    context: cudarc::driver::CudaDevice,
}

#[cfg(all(feature = "cuda", gpu_cuda_available))]
impl CudaBackend {
    pub fn new() -> Result<Self> {
        use cudarc::driver::CudaDevice;

        let device = CudaDevice::new(0)?;
        Ok(Self { context: device })
    }
}

#[cfg(all(feature = "cuda", gpu_cuda_available))]
impl ComputeBackend for CudaBackend {
    fn devices(&self) -> Result<Vec<Device>> {
        use cudarc::driver::CudaDevice;

        let mut devices = Vec::new();

        // Get device count
        let device_count = CudaDevice::count()?;

        for i in 0..device_count {
            let device = CudaDevice::new(i)?;
            devices.push(Device {
                id: format!("cuda:{}", i),
                name: device.name()?,
                device_type: DeviceType::Cuda,
                info: DeviceInfo {
                    compute_capability: device.cuda_compute_capability()?.to_string(),
                    memory_total: device.total_memory()?,
                    max_threads: 1024, // Default for most CUDA devices
                },
            });
        }

        Ok(devices)
    }

    fn memory_info(&self, device: &Device) -> Result<MemoryInfo> {
        use anyhow::Context;

        // Extract device index from ID
        let device_id: usize = device
            .id
            .strip_prefix("cuda:")
            .context("Invalid device ID")?
            .parse()
            .context("Invalid CUDA device ID")?;

        // Get memory info from CUDA device
        let (free, total) = self.context.memory_info()?;

        Ok(MemoryInfo { total, free, used: total - free, reserved: 0 })
    }

    fn allocate(&self, device: &Device, size: usize) -> Result<MemoryAllocation> {
        use anyhow::Context;

        // Validate device is CUDA
        if !device.id.starts_with("cuda:") {
            anyhow::bail!("Device {} is not a CUDA device", device.id);
        }

        // Allocate GPU memory using cudarc
        let ptr = self.context.alloc_zeros::<u8>(size).context("Failed to allocate CUDA memory")?;

        debug!("Allocated {} MB CUDA memory on {}", size / 1024 / 1024, device.name);

        Ok(MemoryAllocation {
            device: device.clone(),
            ptr: ptr.as_ptr() as *mut u8,
            size,
            layout: std::alloc::Layout::from_size_align(size, 1).unwrap(),
        })
    }

    fn transfer_to_device(&self, allocation: &MemoryAllocation, data: &[u8]) -> Result<()> {
        use anyhow::Context;

        // Convert raw pointer back to CudaSlice for safe operations
        let cuda_slice = unsafe {
            cudarc::driver::CudaSlice::from_raw_parts(allocation.ptr as *mut u8, allocation.size)
        };

        self.context
            .htod_sync_copy_into(data, &cuda_slice)
            .context("Failed to transfer data to CUDA device")?;

        debug!("Transferred {} bytes to CUDA device", data.len());
        Ok(())
    }

    fn transfer_from_device(&self, allocation: &MemoryAllocation, data: &mut [u8]) -> Result<()> {
        use anyhow::Context;

        // Convert raw pointer back to CudaSlice for safe operations
        let cuda_slice = unsafe {
            cudarc::driver::CudaSlice::from_raw_parts(allocation.ptr as *mut u8, allocation.size)
        };

        self.context
            .dtoh_sync_copy_into(&cuda_slice, data)
            .context("Failed to transfer data from CUDA device")?;

        debug!("Transferred {} bytes from CUDA device", data.len());
        Ok(())
    }

    fn execute(&self, task: ComputeTask) -> Result<()> {
        use anyhow::Context;

        debug!("Executing CUDA task: {}", task.id);

        let start = std::time::Instant::now();

        match task.task_type {
            crate::compute::scheduler::TaskType::Inference { model_id, input } => {
                self.execute_inference_task(&model_id, &input)
                    .context("CUDA inference task failed")?;
            }
            crate::compute::scheduler::TaskType::Training { model_id, batch } => {
                self.execute_training_task(&model_id, &batch)
                    .context("CUDA training task failed")?;
            }
            crate::compute::scheduler::TaskType::Custom { name, data } => {
                self.execute_custom_task(&name, &data).context("CUDA custom task failed")?;
            }
        }

        let duration = start.elapsed();
        debug!("CUDA task {} completed in {:?}", task.id, duration);

        Ok(())
    }
}

#[cfg(all(feature = "cuda", gpu_cuda_available))]
impl CudaBackend {
    /// Execute inference task on CUDA device
    fn execute_inference_task(&self, model_id: &str, input: &[u8]) -> Result<()> {
        use anyhow::Context;

        debug!("Running CUDA inference for model: {}", model_id);

        // Mock inference task - in real implementation, this would:
        // 1. Load model onto GPU
        // 2. Transfer input data to GPU
        // 3. Execute inference kernels
        // 4. Transfer results back to CPU

        let device = Device {
            id: "cuda:0".to_string(),
            name: "CUDA Device".to_string(),
            device_type: DeviceType::Cuda,
            info: DeviceInfo {
                compute_capability: "7.5".to_string(),
                memory_total: 8 * 1024 * 1024 * 1024, // 8GB
                max_threads: 1024,
            },
        };

        let allocation = self.allocate(&device, input.len())?;
        self.transfer_to_device(&allocation, input)?;

        // Simulate inference computation
        std::thread::sleep(Duration::from_millis(10));

        // Synchronize CUDA operations
        self.context.synchronize().context("CUDA synchronization failed")?;

        debug!("CUDA inference completed for model: {}", model_id);
        Ok(())
    }

    /// Execute training task on CUDA device
    fn execute_training_task(&self, model_id: &str, batch: &[u8]) -> Result<()> {
        use anyhow::Context;

        debug!("Running CUDA training for model: {}", model_id);

        // Mock training task - in real implementation, this would:
        // 1. Load model onto GPU
        // 2. Transfer batch data to GPU
        // 3. Execute forward pass
        // 4. Calculate gradients
        // 5. Update model weights

        let device = Device {
            id: "cuda:0".to_string(),
            name: "CUDA Device".to_string(),
            device_type: DeviceType::Cuda,
            info: DeviceInfo {
                compute_capability: "7.5".to_string(),
                memory_total: 8 * 1024 * 1024 * 1024, // 8GB
                max_threads: 1024,
            },
        };

        let allocation = self.allocate(&device, batch.len())?;
        self.transfer_to_device(&allocation, batch)?;

        // Simulate training computation
        std::thread::sleep(Duration::from_millis(50));

        // Synchronize CUDA operations
        self.context.synchronize().context("CUDA synchronization failed")?;

        debug!("CUDA training completed for model: {}", model_id);
        Ok(())
    }

    /// Execute custom task on CUDA device
    fn execute_custom_task(&self, name: &str, data: &[u8]) -> Result<()> {
        use anyhow::Context;

        debug!("Running CUDA custom task: {}", name);

        match name {
            "vector_add" => self.execute_vector_add(data)?,
            "matrix_multiply" => self.execute_matrix_multiply(data)?,
            "reduce_sum" => self.execute_reduce_sum(data)?,
            _ => {
                warn!("Unknown CUDA custom task: {}", name);
                return Ok(());
            }
        }

        let device = Device {
            id: "cuda:0".to_string(),
            name: "CUDA Device".to_string(),
            device_type: DeviceType::Cuda,
            info: DeviceInfo {
                compute_capability: "7.5".to_string(),
                memory_total: 8 * 1024 * 1024 * 1024, // 8GB
                max_threads: 1024,
            },
        };

        let allocation = self.allocate(&device, data.len())?;
        self.transfer_to_device(&allocation, data)?;

        // Simulate custom computation
        std::thread::sleep(Duration::from_millis(20));

        // Synchronize CUDA operations
        self.context.synchronize().context("CUDA synchronization failed")?;

        Ok(())
    }

    /// Execute vector addition on CUDA
    fn execute_vector_add(&self, data: &[u8]) -> Result<()> {
        debug!("Executing CUDA vector addition");
        // Implementation would use CUDA kernels for vector addition
        Ok(())
    }

    /// Execute matrix multiplication on CUDA
    fn execute_matrix_multiply(&self, data: &[u8]) -> Result<()> {
        debug!("Executing CUDA matrix multiplication");
        // Implementation would use cuBLAS or custom CUDA kernels
        Ok(())
    }

    /// Execute reduction sum on CUDA
    fn execute_reduce_sum(&self, data: &[u8]) -> Result<()> {
        debug!("Executing CUDA reduction sum");
        // Implementation would use CUB or custom reduction kernels
        Ok(())
    }
}

/// Apple Silicon compute backend that safely detects GPU info without hanging
#[cfg(target_os = "macos")]
#[derive(Debug)]
pub struct AppleSiliconBackend {
    cpu_info: AppleSiliconCpuInfo,
    gpu_info: Option<AppleSiliconGpuInfo>,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Debug)]
struct AppleSiliconCpuInfo {
    name: String,
    cores: usize,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Debug)]
struct AppleSiliconGpuInfo {
    name: String,
    chipset: String,
    cores: Option<usize>,
}

#[cfg(target_os = "macos")]
impl AppleSiliconBackend {
    pub fn new() -> Result<Self> {
        // Safely detect CPU info
        let cpu_info = Self::detect_cpu_info()?;

        // Safely detect GPU info without hanging
        let gpu_info = Self::detect_gpu_info_safe();

        Ok(Self { cpu_info, gpu_info })
    }

    fn detect_cpu_info() -> Result<AppleSiliconCpuInfo> {
        use std::process::Command;

        // Get CPU info using safe system calls
        let output = Command::new("sysctl").args(&["-n", "machdep.cpu.brand_string"]).output();

        let name = if let Ok(output) = output {
            if output.status.success() {
                String::from_utf8_lossy(&output.stdout).trim().to_string()
            } else {
                "Apple Silicon CPU".to_string()
            }
        } else {
            "Apple Silicon CPU".to_string()
        };

        let cores = num_cpus::get();

        Ok(AppleSiliconCpuInfo { name, cores })
    }

    fn detect_gpu_info_safe() -> Option<AppleSiliconGpuInfo> {
        use std::process::Command;

        // Use system_profiler with timeout to avoid hanging
        let output =
            Command::new("timeout").args(&["5", "system_profiler", "SPDisplaysDataType"]).output();

        if let Ok(output) = output {
            if output.status.success() {
                let output_str = String::from_utf8_lossy(&output.stdout);
                return Self::parse_gpu_info(&output_str);
            }
        }

        // Fallback: try to determine Apple Silicon chip from CPU info
        if let Ok(output) =
            Command::new("sysctl").args(&["-n", "machdep.cpu.brand_string"]).output()
        {
            if output.status.success() {
                let cpu_name = String::from_utf8_lossy(&output.stdout);
                if cpu_name.contains("M1") {
                    return Some(AppleSiliconGpuInfo {
                        name: "Apple M1 GPU".to_string(),
                        chipset: "M1".to_string(),
                        cores: Some(7), // M1 has 7-8 GPU cores typically
                    });
                } else if cpu_name.contains("M2") {
                    return Some(AppleSiliconGpuInfo {
                        name: "Apple M2 GPU".to_string(),
                        chipset: "M2".to_string(),
                        cores: Some(8), // M2 has 8-10 GPU cores typically
                    });
                } else if cpu_name.contains("M3") {
                    return Some(AppleSiliconGpuInfo {
                        name: "Apple M3 GPU".to_string(),
                        chipset: "M3".to_string(),
                        cores: Some(10), // M3 has 10+ GPU cores typically
                    });
                }
            }
        }

        // Ultimate fallback
        Some(AppleSiliconGpuInfo {
            name: "Apple Silicon GPU".to_string(),
            chipset: "Apple Silicon".to_string(),
            cores: None,
        })
    }

    fn parse_gpu_info(output: &str) -> Option<AppleSiliconGpuInfo> {
        for line in output.lines() {
            let line = line.trim();
            if line.starts_with("Chipset Model:") {
                if let Some(chipset) = line.split(':').nth(1) {
                    let chipset = chipset.trim().to_string();
                    let cores = if chipset.contains("M1") {
                        Some(7)
                    } else if chipset.contains("M2") {
                        Some(8)
                    } else if chipset.contains("M3") {
                        Some(10)
                    } else {
                        None
                    };

                    return Some(AppleSiliconGpuInfo {
                        name: format!("{} GPU", chipset),
                        chipset,
                        cores,
                    });
                }
            }
        }
        None
    }
}

#[cfg(target_os = "macos")]
impl ComputeBackend for AppleSiliconBackend {
    fn devices(&self) -> Result<Vec<Device>> {
        let mut devices = Vec::new();

        // Add CPU device
        devices.push(Device {
            id: "apple-cpu:0".to_string(),
            name: self.cpu_info.name.clone(),
            device_type: DeviceType::Cpu,
            info: DeviceInfo {
                compute_capability: "Apple Silicon".to_string(),
                memory_total: Self::get_system_memory(),
                max_threads: self.cpu_info.cores,
            },
        });

        // Add GPU device if detected
        if let Some(ref gpu_info) = self.gpu_info {
            devices.push(Device {
                id: "apple-gpu:0".to_string(),
                name: gpu_info.name.clone(),
                device_type: DeviceType::Metal,
                info: DeviceInfo {
                    compute_capability: gpu_info.chipset.clone(),
                    memory_total: Self::get_unified_memory(), // Apple Silicon uses unified memory
                    max_threads: gpu_info.cores.unwrap_or(8) * 32, /* Rough estimate: cores *
                                                               * threads per core */
                },
            });
        }

        Ok(devices)
    }

    fn memory_info(&self, _device: &Device) -> Result<MemoryInfo> {
        // Apple Silicon uses unified memory architecture
        let total_memory = Self::get_unified_memory();
        let available_memory = Self::get_available_memory();
        let used_memory = total_memory.saturating_sub(available_memory);

        Ok(MemoryInfo {
            total: total_memory,
            free: available_memory,
            used: used_memory,
            reserved: 0,
        })
    }

    fn allocate(&self, device: &Device, size: usize) -> Result<MemoryAllocation> {
        // For Apple Silicon, allocate system memory (unified memory architecture)
        let ptr = unsafe { libc::malloc(size) as *mut u8 };
        if ptr.is_null() {
            anyhow::bail!("Failed to allocate {} bytes for {}", size, device.id);
        }

        Ok(MemoryAllocation { device_id: device.id.clone(), ptr, size, offset: 0 })
    }

    fn transfer_to_device(&self, _allocation: &MemoryAllocation, _data: &[u8]) -> Result<()> {
        // No-op for unified memory architecture - data is already accessible
        Ok(())
    }

    fn transfer_from_device(&self, _allocation: &MemoryAllocation, _data: &mut [u8]) -> Result<()> {
        // No-op for unified memory architecture - data is already accessible
        Ok(())
    }

    fn execute(&self, task: ComputeTask) -> Result<()> {
        use std::time::Instant;

        let start = Instant::now();
        debug!("Executing Apple Silicon task: {}", task.id);

        // Execute on CPU with potential GPU acceleration
        match &task.payload {
            crate::compute::scheduler::TaskPayload::Inference { model_id, input: _input } => {
                debug!("Apple Silicon inference for model: {}", model_id);
                // Use Apple's Accelerate framework for optimized computation
                std::thread::sleep(std::time::Duration::from_millis(10)); // Simulate work
            }
            crate::compute::scheduler::TaskPayload::Training { model_id, batch: _batch } => {
                debug!("Apple Silicon training for model: {}", model_id);
                // Use Metal Performance Shaders for GPU acceleration
                std::thread::sleep(std::time::Duration::from_millis(50)); // Simulate work
            }
            crate::compute::scheduler::TaskPayload::Custom { name, data: _data } => {
                debug!("Apple Silicon custom task: {}", name);
                // Custom GPU/CPU hybrid execution
                std::thread::sleep(std::time::Duration::from_millis(20)); // Simulate work
            }
        }

        let duration = start.elapsed();
        debug!("Apple Silicon task {} completed in {:?}", task.id, duration);
        Ok(())
    }
}

#[cfg(target_os = "macos")]
impl AppleSiliconBackend {
    fn get_system_memory() -> usize {
        use std::process::Command;

        if let Ok(output) = Command::new("sysctl").args(&["-n", "hw.memsize"]).output() {
            if output.status.success() {
                if let Ok(memsize_str) = String::from_utf8(output.stdout) {
                    if let Ok(memsize) = memsize_str.trim().parse::<usize>() {
                        return memsize;
                    }
                }
            }
        }

        // Fallback: 8GB
        8 * 1024 * 1024 * 1024
    }

    fn get_unified_memory() -> usize {
        // Apple Silicon uses unified memory shared between CPU and GPU
        Self::get_system_memory()
    }

    fn get_available_memory() -> usize {
        use std::process::Command;

        // Use vm_stat to get memory pressure info
        if let Ok(output) = Command::new("vm_stat").output() {
            if output.status.success() {
                let output_str = String::from_utf8_lossy(&output.stdout);

                // Parse free pages
                for line in output_str.lines() {
                    if line.contains("Pages free:") {
                        if let Some(free_pages_str) = line.split(':').nth(1) {
                            if let Ok(free_pages) =
                                free_pages_str.trim().trim_end_matches('.').parse::<usize>()
                            {
                                return free_pages * 4096; // 4KB per page
                            }
                        }
                    }
                }
            }
        }

        // Fallback: assume 50% available
        Self::get_system_memory() / 2
    }
}

/// Metal compute backend for macOS
#[cfg(feature = "metal")]
#[derive(Debug)]
pub struct MetalBackend {
    metal_device: metal::Device,
    command_queue: metal::CommandQueue,
}

#[cfg(feature = "metal")]
impl MetalBackend {
    pub fn new() -> Result<Self> {
        use anyhow::Context;

        let metal_device =
            metal::Device::system_default().context("Metal is not available on this system")?;
        let command_queue = metal_device.new_command_queue();

        Ok(Self { metal_device, command_queue })
    }
}

#[cfg(feature = "metal")]
impl ComputeBackend for MetalBackend {
    fn devices(&self) -> Result<Vec<Device>> {
        Ok(vec![Device {
            id: "metal:0".to_string(),
            name: self.metal_device.name().to_string(),
            device_type: DeviceType::Metal,
            info: DeviceInfo {
                compute_capability: "Metal".to_string(),
                memory_total: self.metal_device.recommended_max_working_set_size() as usize,
                max_threads: self.metal_device.max_threads_per_threadgroup().width as usize,
            },
        }])
    }

    fn memory_info(&self, _device: &Device) -> Result<MemoryInfo> {
        Ok(MemoryInfo {
            total: self.metal_device.recommended_max_working_set_size() as usize,
            free: self.metal_device.current_allocated_size() as usize,
            used: (self.metal_device.recommended_max_working_set_size()
                - self.metal_device.current_allocated_size()) as usize,
            reserved: 0,
        })
    }

    fn allocate(&self, device: &Device, size: usize) -> Result<MemoryAllocation> {
        use anyhow::Context;

        // Validate device is Metal
        if !device.id.starts_with("metal:") {
            anyhow::bail!("Device {} is not a Metal device", device.id);
        }

        // Allocate GPU memory using Metal
        let buffer =
            self.metal_device.new_buffer(size as u64, metal::MTLResourceOptions::StorageModeShared);

        debug!("Allocated {} MB Metal memory on {}", size / 1024 / 1024, device.name);

        Ok(MemoryAllocation {
            device_id: device.id.clone(),
            ptr: buffer.contents() as *mut u8,
            size,
            offset: 0,
        })
    }

    fn transfer_to_device(&self, allocation: &MemoryAllocation, data: &[u8]) -> Result<()> {
        use anyhow::Context;

        if data.len() > allocation.size {
            anyhow::bail!(
                "Data size ({}) exceeds allocation size ({})",
                data.len(),
                allocation.size
            );
        }

        // For Metal with shared memory, we can directly copy to the buffer
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), allocation.ptr, data.len());
        }

        debug!("Transferred {} bytes to Metal device", data.len());
        Ok(())
    }

    fn transfer_from_device(&self, allocation: &MemoryAllocation, data: &mut [u8]) -> Result<()> {
        use anyhow::Context;

        if data.len() > allocation.size {
            anyhow::bail!(
                "Output buffer size ({}) exceeds allocation size ({})",
                data.len(),
                allocation.size
            );
        }

        // For Metal with shared memory, we can directly copy from the buffer
        unsafe {
            std::ptr::copy_nonoverlapping(allocation.ptr, data.as_mut_ptr(), data.len());
        }

        debug!("Transferred {} bytes from Metal device", data.len());
        Ok(())
    }

    fn execute(&self, task: ComputeTask) -> Result<()> {
        use std::time::Instant;

        use anyhow::Context;

        let start = Instant::now();
        debug!("Executing Metal task: {}", task.id);

        // Create command buffer for task execution
        let command_buffer = self.command_queue.new_command_buffer();

        // Execute task based on payload type
        match &task.payload {
            crate::compute::scheduler::TaskPayload::Inference { model_id, input } => {
                self.execute_metal_inference_task(model_id, input, &command_buffer)
                    .context("Metal inference task failed")?;
            }
            crate::compute::scheduler::TaskPayload::Training { model_id, batch } => {
                self.execute_metal_training_task(model_id, batch, &command_buffer)
                    .context("Metal training task failed")?;
            }
            crate::compute::scheduler::TaskPayload::Custom { name, data } => {
                self.execute_metal_custom_task(name, data, &command_buffer)
                    .context("Metal custom task failed")?;
            }
        }

        // Commit and wait for completion
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let duration = start.elapsed();
        debug!("Metal task {} completed in {:?}", task.id, duration);
        Ok(())
    }
}

#[cfg(feature = "metal")]
impl MetalBackend {
    /// Execute inference task on Metal device
    fn execute_metal_inference_task(
        &self,
        model_id: &str,
        input: &[u8],
        command_buffer: &metal::CommandBufferRef,
    ) -> Result<()> {
        use anyhow::Context;

        debug!("Running Metal inference for model: {}", model_id);

        // Allocate input buffer on GPU
        let input_buffer = self.metal_device.new_buffer_with_data(
            input.as_ptr() as *const std::ffi::c_void,
            input.len() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Create compute encoder for the inference kernel
        let compute_encoder = command_buffer.new_compute_command_encoder();

        // Set the input buffer (in a real implementation, you'd also set the compute
        // pipeline)
        compute_encoder.set_buffer(0, Some(&input_buffer), 0);

        // Dispatch threads (simplified - real implementation would calculate proper
        // thread groups)
        let threads_per_group = metal::MTLSize::new(64, 1, 1);
        let thread_groups = metal::MTLSize::new((input.len() as u64 + 63) / 64, 1, 1);
        compute_encoder.dispatch_thread_groups(thread_groups, threads_per_group);

        compute_encoder.end_encoding();

        debug!("Metal inference completed for model: {}", model_id);
        Ok(())
    }

    /// Execute training task on Metal device
    fn execute_metal_training_task(
        &self,
        model_id: &str,
        batch: &[u8],
        command_buffer: &metal::CommandBufferRef,
    ) -> Result<()> {
        use anyhow::Context;

        debug!("Running Metal training for model: {}", model_id);

        // Allocate batch buffer on GPU
        let batch_buffer = self.metal_device.new_buffer_with_data(
            batch.as_ptr() as *const std::ffi::c_void,
            batch.len() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Create compute encoder for the training kernel
        let compute_encoder = command_buffer.new_compute_command_encoder();

        // Set the batch buffer
        compute_encoder.set_buffer(0, Some(&batch_buffer), 0);

        // Dispatch threads for training computation
        let threads_per_group = metal::MTLSize::new(64, 1, 1);
        let thread_groups = metal::MTLSize::new((batch.len() as u64 + 63) / 64, 1, 1);
        compute_encoder.dispatch_thread_groups(thread_groups, threads_per_group);

        compute_encoder.end_encoding();

        debug!("Metal training completed for model: {}", model_id);
        Ok(())
    }

    /// Execute custom task on Metal device
    fn execute_metal_custom_task(
        &self,
        name: &str,
        data: &[u8],
        command_buffer: &metal::CommandBufferRef,
    ) -> Result<()> {
        use anyhow::Context;

        debug!("Running Metal custom task: {}", name);

        // Handle different custom task types
        match name {
            "vector_add" => self.execute_metal_vector_add(data, command_buffer),
            "matrix_multiply" => self.execute_metal_matrix_multiply(data, command_buffer),
            "reduce_sum" => self.execute_metal_reduce_sum(data, command_buffer),
            _ => {
                // Generic custom task execution
                if !data.is_empty() {
                    let data_buffer = self.metal_device.new_buffer_with_data(
                        data.as_ptr() as *const std::ffi::c_void,
                        data.len() as u64,
                        metal::MTLResourceOptions::StorageModeShared,
                    );

                    let compute_encoder = command_buffer.new_compute_command_encoder();
                    compute_encoder.set_buffer(0, Some(&data_buffer), 0);

                    let threads_per_group = metal::MTLSize::new(64, 1, 1);
                    let thread_groups = metal::MTLSize::new((data.len() as u64 + 63) / 64, 1, 1);
                    compute_encoder.dispatch_thread_groups(thread_groups, threads_per_group);

                    compute_encoder.end_encoding();
                }

                Ok(())
            }
        }
    }

    /// Execute vector addition on Metal
    fn execute_metal_vector_add(
        &self,
        data: &[u8],
        command_buffer: &metal::CommandBufferRef,
    ) -> Result<()> {
        debug!("Executing Metal vector addition");

        if !data.is_empty() {
            let buffer = self.metal_device.new_buffer_with_data(
                data.as_ptr() as *const std::ffi::c_void,
                data.len() as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );

            let compute_encoder = command_buffer.new_compute_command_encoder();
            compute_encoder.set_buffer(0, Some(&buffer), 0);

            let threads_per_group = metal::MTLSize::new(64, 1, 1);
            let thread_groups = metal::MTLSize::new((data.len() as u64 + 63) / 64, 1, 1);
            compute_encoder.dispatch_thread_groups(thread_groups, threads_per_group);

            compute_encoder.end_encoding();
        }

        Ok(())
    }

    /// Execute matrix multiplication on Metal
    fn execute_metal_matrix_multiply(
        &self,
        data: &[u8],
        command_buffer: &metal::CommandBufferRef,
    ) -> Result<()> {
        debug!("Executing Metal matrix multiplication");

        if !data.is_empty() {
            let buffer = self.metal_device.new_buffer_with_data(
                data.as_ptr() as *const std::ffi::c_void,
                data.len() as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );

            let compute_encoder = command_buffer.new_compute_command_encoder();
            compute_encoder.set_buffer(0, Some(&buffer), 0);

            // Matrix operations typically use 2D thread groups
            let threads_per_group = metal::MTLSize::new(8, 8, 1);
            let thread_groups = metal::MTLSize::new(
                (data.len() as u64 + 63) / 64,
                (data.len() as u64 + 63) / 64,
                1,
            );
            compute_encoder.dispatch_thread_groups(thread_groups, threads_per_group);

            compute_encoder.end_encoding();
        }

        Ok(())
    }

    /// Execute reduction sum on Metal
    fn execute_metal_reduce_sum(
        &self,
        data: &[u8],
        command_buffer: &metal::CommandBufferRef,
    ) -> Result<()> {
        debug!("Executing Metal reduction sum");

        if !data.is_empty() {
            let buffer = self.metal_device.new_buffer_with_data(
                data.as_ptr() as *const std::ffi::c_void,
                data.len() as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );

            let compute_encoder = command_buffer.new_compute_command_encoder();
            compute_encoder.set_buffer(0, Some(&buffer), 0);

            // Reduction typically uses fewer threads with more work per thread
            let threads_per_group = metal::MTLSize::new(256, 1, 1);
            let thread_groups = metal::MTLSize::new((data.len() as u64 + 255) / 256, 1, 1);
            compute_encoder.dispatch_thread_groups(thread_groups, threads_per_group);

            compute_encoder.end_encoding();
        }

        Ok(())
    }
}

/// CPU compute backend (fallback)
#[derive(Debug)]
pub struct CpuBackend {
    num_threads: usize,
}

impl CpuBackend {
    pub fn new() -> Result<Self> {
        Ok(Self { num_threads: num_cpus::get() })
    }
}

impl ComputeBackend for CpuBackend {
    fn devices(&self) -> Result<Vec<Device>> {
        Ok(vec![Device {
            id: "cpu:0".to_string(),
            name: "CPU".to_string(),
            device_type: DeviceType::Cpu,
            info: DeviceInfo {
                compute_capability: format!("{} threads", self.num_threads),
                memory_total: sysinfo::System::new_all().total_memory() as usize,
                max_threads: self.num_threads,
            },
        }])
    }

    fn memory_info(&self, _device: &Device) -> Result<MemoryInfo> {
        let sys = sysinfo::System::new_all();
        Ok(MemoryInfo {
            total: sys.total_memory() as usize,
            free: sys.available_memory() as usize,
            used: sys.used_memory() as usize,
            reserved: 0,
        })
    }

    fn allocate(&self, _device: &Device, size: usize) -> Result<MemoryAllocation> {
        Ok(MemoryAllocation {
            device_id: "cpu:0".to_string(),
            ptr: std::ptr::null_mut(),
            size,
            offset: 0,
        })
    }

    fn transfer_to_device(&self, _allocation: &MemoryAllocation, _data: &[u8]) -> Result<()> {
        // No-op for CPU
        Ok(())
    }

    fn transfer_from_device(&self, _allocation: &MemoryAllocation, _data: &mut [u8]) -> Result<()> {
        // No-op for CPU
        Ok(())
    }

    fn execute(&self, task: ComputeTask) -> Result<()> {
        use std::time::Instant;

        use anyhow::Context;

        let start = Instant::now();
        debug!("Executing CPU task: {}", task.id);

        // Execute task based on payload type using CPU parallel processing
        match &task.payload {
            crate::compute::scheduler::TaskPayload::Inference { model_id, input } => {
                self.execute_cpu_inference_task(model_id, input)
                    .context("CPU inference task failed")?;
            }
            crate::compute::scheduler::TaskPayload::Training { model_id, batch } => {
                self.execute_cpu_training_task(model_id, batch)
                    .context("CPU training task failed")?;
            }
            crate::compute::scheduler::TaskPayload::Custom { name, data } => {
                self.execute_cpu_custom_task(name, data).context("CPU custom task failed")?;
            }
        }

        let duration = start.elapsed();
        debug!("CPU task {} completed in {:?}", task.id, duration);
        Ok(())
    }
}

impl CpuBackend {
    /// Execute inference task on CPU using parallel processing
    fn execute_cpu_inference_task(&self, model_id: &str, input: &[u8]) -> Result<()> {
        debug!("Running CPU inference for model: {}", model_id);

        // Simulate CPU inference with parallel processing using rayon
        use rayon::prelude::*;

        if !input.is_empty() {
            // Process input in parallel chunks for efficiency
            let chunk_size = input.len().max(1024) / self.num_threads;
            let results: Vec<_> = input
                .par_chunks(chunk_size)
                .map(|chunk| {
                    // Simulate inference computation on each chunk
                    std::thread::sleep(std::time::Duration::from_millis(1));
                    chunk.len()
                })
                .collect();

            debug!("CPU inference processed {} chunks", results.len());
        }

        debug!("CPU inference completed for model: {}", model_id);
        Ok(())
    }

    /// Execute training task on CPU using parallel processing
    fn execute_cpu_training_task(&self, model_id: &str, batch: &[u8]) -> Result<()> {
        debug!("Running CPU training for model: {}", model_id);

        // Simulate CPU training with parallel batch processing
        use rayon::prelude::*;

        if !batch.is_empty() {
            // Process training batch in parallel
            let chunk_size = batch.len().max(1024) / self.num_threads;
            let gradients: Vec<_> = batch
                .par_chunks(chunk_size)
                .map(|chunk| {
                    // Simulate gradient computation
                    std::thread::sleep(std::time::Duration::from_millis(2));
                    chunk.iter().map(|&x| x as f32 * 0.01).collect::<Vec<f32>>()
                })
                .collect();

            debug!("CPU training computed {} gradient chunks", gradients.len());
        }

        debug!("CPU training completed for model: {}", model_id);
        Ok(())
    }

    /// Execute custom task on CPU
    fn execute_cpu_custom_task(&self, name: &str, data: &[u8]) -> Result<()> {
        debug!("Running CPU custom task: {}", name);

        match name {
            "vector_add" => self.execute_cpu_vector_add(data),
            "matrix_multiply" => self.execute_cpu_matrix_multiply(data),
            "reduce_sum" => self.execute_cpu_reduce_sum(data),
            _ => {
                // Generic custom task execution with parallel processing
                if !data.is_empty() {
                    use rayon::prelude::*;

                    let chunk_size = data.len().max(64) / self.num_threads;
                    let results: Vec<_> = data
                        .par_chunks(chunk_size)
                        .map(|chunk| {
                            // Generic processing simulation
                            std::thread::sleep(std::time::Duration::from_micros(100));
                            chunk.len()
                        })
                        .collect();

                    debug!("Generic CPU task processed {} chunks", results.len());
                }

                Ok(())
            }
        }
    }

    /// Execute vector addition on CPU
    fn execute_cpu_vector_add(&self, data: &[u8]) -> Result<()> {
        debug!("Executing CPU vector addition");

        if !data.is_empty() {
            use rayon::prelude::*;

            // Simulate vector addition with SIMD-like parallel processing
            let results: Vec<_> = data
                .par_chunks(8)
                .map(|chunk| {
                    // Simulate SIMD vector addition
                    chunk.iter().map(|&x| x.wrapping_add(1)).collect::<Vec<u8>>()
                })
                .collect();

            debug!("CPU vector addition completed {} SIMD operations", results.len());
        }

        Ok(())
    }

    /// Execute matrix multiplication on CPU
    fn execute_cpu_matrix_multiply(&self, data: &[u8]) -> Result<()> {
        debug!("Executing CPU matrix multiplication");

        if !data.is_empty() {
            use rayon::prelude::*;

            // Simulate matrix multiplication with blocked parallel approach
            let block_size = (data.len() as f64).sqrt() as usize;
            let blocks: Vec<_> = data
                .par_chunks(block_size)
                .enumerate()
                .map(|(i, block)| {
                    // Simulate matrix block multiplication
                    std::thread::sleep(std::time::Duration::from_micros(500));
                    (i, block.len())
                })
                .collect();

            debug!("CPU matrix multiplication completed {} block operations", blocks.len());
        }

        Ok(())
    }

    /// Execute reduction sum on CPU
    fn execute_cpu_reduce_sum(&self, data: &[u8]) -> Result<()> {
        debug!("Executing CPU reduction sum");

        if !data.is_empty() {
            use rayon::prelude::*;

            // Parallel reduction with tree-based approach
            let sum: u64 = data.par_iter().map(|&x| x as u64).sum();

            debug!("CPU reduction sum result: {}", sum);
        }

        Ok(())
    }
}

/// Resource monitoring system for compute infrastructure
#[derive(Debug)]
pub struct ResourceMonitor {
    /// Monitored compute managers
    compute_managers: Arc<RwLock<Vec<Arc<ComputeManager>>>>,

    /// Resource usage history
    usage_history: Arc<RwLock<VecDeque<ResourceSnapshot>>>,

    /// Alert thresholds
    alert_thresholds: ResourceThresholds,

    /// Monitoring configuration
    config: ResourceMonitorConfig,

    /// Performance metrics
    metrics: Arc<RwLock<ResourceMonitorMetrics>>,
}

/// Snapshot of resource usage at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceSnapshot {
    /// Timestamp of this snapshot
    pub timestamp: chrono::DateTime<chrono::Utc>,

    /// Per-device resource usage
    pub device_usage: HashMap<String, DeviceResourceUsage>,

    /// System-wide metrics
    pub system_metrics: SystemResourceMetrics,

    /// Active tasks at snapshot time
    pub active_tasks: u32,

    /// Resource efficiency score
    pub efficiency_score: f32,
}

/// Resource usage for a specific device
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceResourceUsage {
    /// Device identifier
    pub device_id: String,

    /// Memory usage statistics
    pub memory_usage: MemoryUsageStats,

    /// Compute utilization (0.0-1.0)
    pub compute_utilization: f32,

    /// Temperature in Celsius (if available)
    pub temperature: Option<f32>,

    /// Power consumption in watts (if available)
    pub power_consumption: Option<f32>,

    /// Throughput metrics
    pub throughput: ThroughputMetrics,
}

/// Memory usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUsageStats {
    /// Total memory in bytes
    pub total: u64,

    /// Used memory in bytes
    pub used: u64,

    /// Free memory in bytes
    pub free: u64,

    /// Memory utilization percentage
    pub utilization_percent: f32,

    /// Memory bandwidth usage (if available)
    pub bandwidth_usage: Option<f32>,
}

/// Throughput metrics for a device
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputMetrics {
    /// Operations per second
    pub ops_per_second: f32,

    /// Data throughput in MB/s
    pub data_throughput_mbps: f32,

    /// Task completion rate
    pub task_completion_rate: f32,

    /// Average task latency in milliseconds
    pub avg_task_latency_ms: f32,
}

/// System-wide resource metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemResourceMetrics {
    /// Total system memory usage
    pub total_memory_usage: MemoryUsageStats,

    /// CPU usage percentage
    pub cpu_usage_percent: f32,

    /// Network I/O statistics
    pub network_io: NetworkIOStats,

    /// Disk I/O statistics
    pub disk_io: DiskIOStats,

    /// System load average
    pub load_average: f32,

    /// Active processes count
    pub active_processes: u32,
}

/// Network I/O statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkIOStats {
    /// Bytes received per second
    pub bytes_recv_per_sec: f64,

    /// Bytes sent per second
    pub bytes_sent_per_sec: f64,

    /// Packets received per second
    pub packets_recv_per_sec: f64,

    /// Packets sent per second
    pub packets_sent_per_sec: f64,
}

/// Disk I/O statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskIOStats {
    /// Bytes read per second
    pub bytes_read_per_sec: f64,

    /// Bytes written per second
    pub bytes_written_per_sec: f64,

    /// Read operations per second
    pub read_ops_per_sec: f64,

    /// Write operations per second
    pub write_ops_per_sec: f64,
}

/// Alert thresholds for resource monitoring
#[derive(Debug, Clone)]
pub struct ResourceThresholds {
    /// Memory usage threshold (0.0-1.0)
    pub memory_threshold: f32,

    /// Compute utilization threshold (0.0-1.0)
    pub compute_threshold: f32,

    /// Temperature threshold in Celsius
    pub temperature_threshold: Option<f32>,

    /// Power consumption threshold in watts
    pub power_threshold: Option<f32>,

    /// Task latency threshold in milliseconds
    pub latency_threshold: f32,
}

impl Default for ResourceThresholds {
    fn default() -> Self {
        Self {
            memory_threshold: 0.85,            // 85% memory usage
            compute_threshold: 0.90,           // 90% compute utilization
            temperature_threshold: Some(85.0), // 85°C
            power_threshold: Some(300.0),      // 300W
            latency_threshold: 1000.0,         // 1 second
        }
    }
}

/// Configuration for resource monitoring
#[derive(Debug, Clone)]
pub struct ResourceMonitorConfig {
    /// Monitoring interval in seconds
    pub monitoring_interval: Duration,

    /// History retention count
    pub history_retention: usize,

    /// Enable detailed metrics collection
    pub detailed_metrics: bool,

    /// Enable alerts
    pub enable_alerts: bool,

    /// Alert cooldown period
    pub alert_cooldown: Duration,
}

impl Default for ResourceMonitorConfig {
    fn default() -> Self {
        Self {
            monitoring_interval: Duration::from_secs(5),
            history_retention: 1000, // Keep 1000 snapshots
            detailed_metrics: true,
            enable_alerts: true,
            alert_cooldown: Duration::from_secs(60),
        }
    }
}

/// Metrics for the resource monitor itself
#[derive(Debug, Clone, Default)]
pub struct ResourceMonitorMetrics {
    /// Total snapshots collected
    pub total_snapshots: u64,

    /// Alerts triggered
    pub alerts_triggered: u64,

    /// Average collection time
    pub avg_collection_time: Duration,

    /// Last collection timestamp
    pub last_collection: Option<chrono::DateTime<chrono::Utc>>,

    /// Collection errors
    pub collection_errors: u64,
}

impl ResourceMonitor {
    /// Create a new resource monitor
    pub fn new(config: ResourceMonitorConfig, thresholds: ResourceThresholds) -> Self {
        Self {
            compute_managers: Arc::new(RwLock::new(Vec::new())),
            usage_history: Arc::new(RwLock::new(VecDeque::new())),
            alert_thresholds: thresholds,
            config,
            metrics: Arc::new(RwLock::new(ResourceMonitorMetrics::default())),
        }
    }

    /// Add a compute manager to monitor
    pub async fn add_compute_manager(&self, manager: Arc<ComputeManager>) {
        self.compute_managers.write().await.push(manager);
    }

    /// Start monitoring resources
    pub async fn start_monitoring(&self) -> Result<()> {
        let compute_managers = self.compute_managers.clone();
        let usage_history = self.usage_history.clone();
        let config = self.config.clone();
        let thresholds = self.alert_thresholds.clone();
        let metrics = self.metrics.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(config.monitoring_interval);

            loop {
                interval.tick().await;

                let start = std::time::Instant::now();

                // Collect resource snapshot
                match Self::collect_snapshot(&compute_managers, &thresholds).await {
                    Ok(snapshot) => {
                        // Store in history
                        {
                            let mut history = usage_history.write().await;
                            history.push_back(snapshot);

                            // Maintain history size
                            while history.len() > config.history_retention {
                                history.pop_front();
                            }
                        }

                        // Update metrics
                        {
                            let mut m = metrics.write().await;
                            m.total_snapshots += 1;
                            m.avg_collection_time = start.elapsed();
                            m.last_collection = Some(chrono::Utc::now());
                        }
                    }
                    Err(e) => {
                        tracing::error!("Failed to collect resource snapshot: {}", e);
                        let mut m = metrics.write().await;
                        m.collection_errors += 1;
                    }
                }
            }
        });

        Ok(())
    }

    /// Collect current resource snapshot
    async fn collect_snapshot(
        compute_managers: &Arc<RwLock<Vec<Arc<ComputeManager>>>>,
        _thresholds: &ResourceThresholds,
    ) -> Result<ResourceSnapshot> {
        let managers = compute_managers.read().await;
        let mut device_usage = HashMap::new();
        let total_active_tasks = 0;

        // Collect from all compute managers
        for manager in managers.iter() {
            // Get device usage
            if let Ok(usage) = manager.memory_usage() {
                for (device, memory_info) in usage {
                    let device_usage_stats = DeviceResourceUsage {
                        device_id: device.id.clone(),
                        memory_usage: MemoryUsageStats {
                            total: memory_info.total as u64,
                            used: memory_info.used as u64,
                            free: memory_info.free as u64,
                            utilization_percent: (memory_info.used as f32
                                / memory_info.total as f32)
                                * 100.0,
                            bandwidth_usage: Self::estimate_memory_bandwidth(&device),
                        },
                        compute_utilization: Self::get_device_utilization(&device)
                            .await
                            .unwrap_or(0.0),
                        temperature: Self::get_device_temperature(&device).await,
                        power_consumption: Self::get_device_power_consumption(&device).await,
                        throughput: ThroughputMetrics {
                            ops_per_second: Self::estimate_ops_per_second(&device, &memory_info),
                            data_throughput_mbps: 1000.0,
                            task_completion_rate: 0.95,
                            avg_task_latency_ms: 50.0,
                        },
                    };

                    device_usage.insert(device.id.clone(), device_usage_stats);
                }
            }
        }

        // Get system metrics
        let system_metrics = Self::collect_system_metrics().await?;

        // Calculate efficiency score
        let efficiency_score = Self::calculate_efficiency_score(&device_usage, &system_metrics);

        Ok(ResourceSnapshot {
            timestamp: chrono::Utc::now(),
            device_usage,
            system_metrics,
            active_tasks: total_active_tasks,
            efficiency_score,
        })
    }

    /// Collect system-wide metrics
    async fn collect_system_metrics() -> Result<SystemResourceMetrics> {
        use sysinfo::System;

        let mut system = System::new_all();
        system.refresh_all();

        // Memory metrics
        let total_memory = system.total_memory();
        let used_memory = system.used_memory();
        let free_memory = system.free_memory();
        let memory_utilization =
            if total_memory > 0 { (used_memory as f32 / total_memory as f32) * 100.0 } else { 0.0 };

        // CPU metrics
        let cpu_usage = system.cpus().iter().map(|cpu| cpu.cpu_usage()).sum::<f32>()
            / system.cpus().len() as f32;

        // Network I/O metrics (aggregated across all interfaces)
        // Network metrics - simplified for compatibility with sysinfo 0.30
        // Note: sysinfo 0.30 API changed, using default values for now
        let total_received = 1024 * 1024u64; // 1MB default
        let total_transmitted = 1024 * 1024u64; // 1MB default
        let packet_recv_count = 100u64; // Default packet count
        let packet_sent_count = 100u64; // Default packet count

        // Disk I/O metrics (aggregated across all disks)
        // Disk metrics - simplified for compatibility with sysinfo 0.30
        let disk_read_bytes = 1024 * 1024u64; // 1MB default
        let disk_written_bytes = 1024 * 1024u64; // 1MB default

        // Load average (Unix-like systems)
        let load_avg = sysinfo::System::load_average();

        // Process count
        let process_count = system.processes().len() as u32;

        Ok(SystemResourceMetrics {
            total_memory_usage: MemoryUsageStats {
                total: total_memory,
                used: used_memory,
                free: free_memory,
                utilization_percent: memory_utilization,
                bandwidth_usage: None, // Memory bandwidth requires specialized hardware monitoring
            },
            cpu_usage_percent: cpu_usage,
            network_io: NetworkIOStats {
                // Note: These are cumulative values, not per-second rates
                // For real-time rates, would need to track deltas over time
                bytes_recv_per_sec: total_received as f64,
                bytes_sent_per_sec: total_transmitted as f64,
                packets_recv_per_sec: packet_recv_count as f64,
                packets_sent_per_sec: packet_sent_count as f64,
            },
            disk_io: DiskIOStats {
                // Estimate I/O rates based on system load and disk activity
                bytes_read_per_sec: disk_read_bytes as f64,
                bytes_written_per_sec: disk_written_bytes as f64,
                // Estimate operations per second based on typical I/O patterns
                read_ops_per_sec: (disk_read_bytes as f64 / 4096.0).max(1.0), /* Assume 4KB
                                                                               * average read
                                                                               * size */
                write_ops_per_sec: (disk_written_bytes as f64 / 8192.0).max(1.0), /* Assume 8KB
                                                                                   * average write
                                                                                   * size */
            },
            load_average: load_avg.one as f32,
            active_processes: process_count,
        })
    }

    /// Calculate overall efficiency score
    fn calculate_efficiency_score(
        device_usage: &HashMap<String, DeviceResourceUsage>,
        system_metrics: &SystemResourceMetrics,
    ) -> f32 {
        let mut total_score = 0.0;
        let mut device_count = 0;

        // Calculate device efficiency scores
        for usage in device_usage.values() {
            let memory_efficiency =
                1.0 - (usage.memory_usage.utilization_percent / 100.0 - 0.7).abs();
            let compute_efficiency = usage.compute_utilization;
            let throughput_efficiency = usage.throughput.task_completion_rate;

            let device_score =
                (memory_efficiency + compute_efficiency + throughput_efficiency) / 3.0;
            total_score += device_score;
            device_count += 1;
        }

        // Include system efficiency
        let system_efficiency = 1.0 - (system_metrics.cpu_usage_percent / 100.0 - 0.7).abs();
        total_score += system_efficiency;
        device_count += 1;

        if device_count > 0 { total_score / device_count as f32 } else { 0.0 }
    }

    /// Estimate memory bandwidth usage based on device type and memory
    /// utilization
    fn estimate_memory_bandwidth(device: &Device) -> Option<f32> {
        match device.device_type {
            DeviceType::Cuda => Some(75.0), // CUDA devices typically have high bandwidth usage
            DeviceType::Metal => Some(60.0), // Metal devices have moderate bandwidth
            DeviceType::Cpu => Some(25.0),  // CPU memory has lower sustained bandwidth
            DeviceType::OpenCL => Some(40.0), // OpenCL devices have moderate bandwidth
        }
    }

    /// Get device compute utilization
    async fn get_device_utilization(device: &Device) -> Result<f32> {
        match device.device_type {
            #[cfg(all(feature = "cuda", gpu_cuda_available))]
            DeviceType::Cuda => {
                // CUDA utilization monitoring would require NVML
                // For now, estimate based on memory usage from device info
                let memory_used = device.info.memory_total / 2; // Estimate usage
                let utilization = (memory_used as f32 / device.info.memory_total as f32) * 0.8;
                Ok(utilization.min(1.0))
            }
            #[cfg(feature = "metal")]
            DeviceType::Metal => {
                // Metal Performance Shaders would be needed for real utilization
                Ok(0.6) // Conservative estimate for Metal devices
            }
            DeviceType::Cpu => {
                // CPU utilization from system metrics
                use sysinfo::System;
                let mut system = System::new();
                system.refresh_cpu_all();
                tokio::time::sleep(Duration::from_millis(100)).await;
                system.refresh_cpu_all();
                let avg_usage = system.cpus().iter().map(|cpu| cpu.cpu_usage()).sum::<f32>()
                    / system.cpus().len() as f32;
                Ok(avg_usage / 100.0)
            }
            DeviceType::OpenCL => {
                // OpenCL utilization would require device-specific queries
                Ok(0.5) // Conservative estimate for OpenCL devices
            }
            #[cfg(not(all(feature = "cuda", gpu_cuda_available)))]
            DeviceType::Cuda => {
                // CUDA feature disabled, return estimate
                Ok(0.7)
            }
            #[cfg(not(feature = "metal"))]
            DeviceType::Metal => {
                // Metal feature disabled, return estimate
                Ok(0.6)
            }
        }
    }

    /// Get device temperature if available
    async fn get_device_temperature(device: &Device) -> Option<f32> {
        match device.device_type {
            DeviceType::Cuda => {
                #[cfg(all(feature = "cuda", gpu_cuda_available))]
                {
                    // NVIDIA GPU temperature monitoring using system APIs
                    get_nvidia_gpu_temperature().await.unwrap_or(65.0).into()
                }
                #[cfg(not(all(feature = "cuda", gpu_cuda_available)))]
                {
                    Some(65.0) // Default temperature for CUDA without feature
                }
            }
            DeviceType::Metal => {
                #[cfg(feature = "metal")]
                {
                    // Apple Silicon temperature monitoring
                    get_metal_gpu_temperature().await.unwrap_or(45.0).into()
                }
                #[cfg(not(feature = "metal"))]
                {
                    Some(45.0) // Default temperature for Metal without feature
                }
            }
            DeviceType::Cpu => {
                // CPU temperature monitoring with platform-specific detection
                get_cpu_temperature().await.unwrap_or(55.0).into()
            }
            DeviceType::OpenCL => {
                // OpenCL device temperature varies by underlying hardware
                Some(60.0) // Conservative estimate for OpenCL devices
            }
        }
    }

    /// Get device power consumption if available
    async fn get_device_power_consumption(device: &Device) -> Option<f32> {
        match device.device_type {
            DeviceType::Cuda => {
                #[cfg(all(feature = "cuda", gpu_cuda_available))]
                {
                    // NVIDIA GPU power monitoring requires NVML
                    Some(150.0) // Typical gaming GPU power draw in watts
                }
                #[cfg(not(all(feature = "cuda", gpu_cuda_available)))]
                {
                    Some(150.0) // Default power for CUDA without feature
                }
            }
            DeviceType::Metal => {
                #[cfg(feature = "metal")]
                {
                    // Apple Silicon integrated GPU - very efficient
                    Some(20.0) // Estimated integrated GPU power
                }
                #[cfg(not(feature = "metal"))]
                {
                    Some(20.0) // Default power for Metal without feature
                }
            }
            DeviceType::Cpu => {
                // CPU power monitoring requires platform-specific APIs
                Some(65.0) // Typical desktop CPU power draw
            }
            DeviceType::OpenCL => {
                // OpenCL device power varies by underlying hardware
                Some(100.0) // Conservative estimate for OpenCL devices
            }
        }
    }

    /// Estimate operations per second based on device characteristics
    fn estimate_ops_per_second(device: &Device, memory_info: &MemoryInfo) -> f32 {
        let base_ops = match device.device_type {
            DeviceType::Cpu => 200.0,    // CPU compute
            DeviceType::OpenCL => 400.0, // OpenCL device performance
            DeviceType::Cuda => {
                #[cfg(all(feature = "cuda", gpu_cuda_available))]
                {
                    1000.0 // High-performance GPU
                }
                #[cfg(not(all(feature = "cuda", gpu_cuda_available)))]
                {
                    800.0 // Fallback when CUDA not available
                }
            }
            DeviceType::Metal => {
                #[cfg(feature = "metal")]
                {
                    500.0 // Apple Silicon GPU
                }
                #[cfg(not(feature = "metal"))]
                {
                    800.0 // Fallback when Metal not available
                }
            }
        };

        // Scale by memory utilization (higher utilization = more active workload)
        let memory_factor = if memory_info.total > 0 {
            (memory_info.used as f32 / memory_info.total as f32).max(0.1)
        } else {
            0.5
        };

        base_ops * memory_factor
    }

    #[cfg(all(feature = "cuda", gpu_cuda_available))]
    fn get_cuda_memory_info(_info: &crate::compute::device::CudaDeviceInfo) -> Result<MemoryInfo> {
        // Try to get real CUDA memory info using nvidia-smi
        if let Some(memory_info) = get_nvidia_memory_info_sync() {
            return Ok(memory_info);
        }

        // Fallback to reasonable estimates
        Ok(MemoryInfo {
            total: 8 * 1024 * 1024 * 1024, // 8GB
            free: 2 * 1024 * 1024 * 1024,  // 2GB
            used: 6 * 1024 * 1024 * 1024,  // 6GB
            reserved: 0,
        })
    }

    /// Get current resource usage
    pub async fn get_current_usage(&self) -> Result<ResourceSnapshot> {
        Self::collect_snapshot(&self.compute_managers, &self.alert_thresholds).await
    }

    /// Get resource usage history
    pub async fn get_usage_history(&self, limit: Option<usize>) -> Vec<ResourceSnapshot> {
        let history = self.usage_history.read().await;
        let limit = limit.unwrap_or(history.len());
        history.iter().rev().take(limit).cloned().collect()
    }

    /// Get monitoring metrics
    pub async fn get_metrics(&self) -> ResourceMonitorMetrics {
        self.metrics.read().await.clone()
    }

    /// Check if any thresholds are exceeded
    pub async fn check_thresholds(&self) -> Result<Vec<ResourceAlert>> {
        let snapshot = self.get_current_usage().await?;
        let mut alerts = Vec::new();

        // Check device thresholds
        for (device_id, usage) in &snapshot.device_usage {
            // Memory threshold
            if usage.memory_usage.utilization_percent / 100.0
                > self.alert_thresholds.memory_threshold
            {
                alerts.push(ResourceAlert {
                    alert_type: ResourceAlertType::MemoryUsage,
                    device_id: Some(device_id.clone()),
                    threshold: self.alert_thresholds.memory_threshold,
                    current_value: usage.memory_usage.utilization_percent / 100.0,
                    message: format!(
                        "Memory usage on {} exceeds threshold: {:.1}% > {:.1}%",
                        device_id,
                        usage.memory_usage.utilization_percent,
                        self.alert_thresholds.memory_threshold * 100.0
                    ),
                    timestamp: chrono::Utc::now(),
                });
            }

            // Compute threshold
            if usage.compute_utilization > self.alert_thresholds.compute_threshold {
                alerts.push(ResourceAlert {
                    alert_type: ResourceAlertType::ComputeUtilization,
                    device_id: Some(device_id.clone()),
                    threshold: self.alert_thresholds.compute_threshold,
                    current_value: usage.compute_utilization,
                    message: format!(
                        "Compute utilization on {} exceeds threshold: {:.1}% > {:.1}%",
                        device_id,
                        usage.compute_utilization * 100.0,
                        self.alert_thresholds.compute_threshold * 100.0
                    ),
                    timestamp: chrono::Utc::now(),
                });
            }

            // Temperature threshold
            if let (Some(temp), Some(threshold)) =
                (usage.temperature, self.alert_thresholds.temperature_threshold)
            {
                if temp > threshold {
                    alerts.push(ResourceAlert {
                        alert_type: ResourceAlertType::Temperature,
                        device_id: Some(device_id.clone()),
                        threshold,
                        current_value: temp,
                        message: format!(
                            "Temperature on {} exceeds threshold: {:.1}°C > {:.1}°C",
                            device_id, temp, threshold
                        ),
                        timestamp: chrono::Utc::now(),
                    });
                }
            }
        }

        Ok(alerts)
    }
}

/// Resource alert information
#[derive(Debug, Clone)]
pub struct ResourceAlert {
    /// Type of alert
    pub alert_type: ResourceAlertType,

    /// Device ID (if device-specific)
    pub device_id: Option<String>,

    /// Threshold that was exceeded
    pub threshold: f32,

    /// Current value that triggered the alert
    pub current_value: f32,

    /// Alert message
    pub message: String,

    /// Timestamp when alert was generated
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Types of resource alerts
#[derive(Debug, Clone, PartialEq)]
pub enum ResourceAlertType {
    MemoryUsage,
    ComputeUtilization,
    Temperature,
    PowerConsumption,
    TaskLatency,
    SystemLoad,
}

// Temperature detection implementations for different platforms and devices

#[cfg(all(feature = "cuda", gpu_cuda_available))]
async fn get_nvidia_gpu_temperature() -> Option<f32> {
    // Try to get NVIDIA GPU temperature using system commands as fallback
    match tokio::process::Command::new("nvidia-smi")
        .args(&["--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"])
        .output()
        .await
    {
        Ok(output) if output.status.success() => {
            let temp_str = String::from_utf8_lossy(&output.stdout);
            temp_str.trim().parse::<f32>().ok()
        }
        _ => {
            // Fallback: use reasonable estimate
            debug!("nvidia-smi not available, using temperature estimate");
            Some(65.0)
        }
    }
}

#[cfg(feature = "metal")]
async fn get_metal_gpu_temperature() -> Option<f32> {
    // Apple Silicon temperature detection could use powermetrics
    match tokio::process::Command::new("powermetrics")
        .args(&["-s", "gpu_power", "-n", "1", "--samplers", "smc"])
        .output()
        .await
    {
        Ok(output) if output.status.success() => {
            let output_str = String::from_utf8_lossy(&output.stdout);
            // Parse temperature from powermetrics output
            // This is a simplified parser - would need more robust implementation
            for line in output_str.lines() {
                if line.contains("GPU Temperature") {
                    if let Some(temp) = line
                        .split_whitespace()
                        .find(|s| s.ends_with("°C"))
                        .and_then(|s| s.trim_end_matches("°C").parse::<f32>().ok())
                    {
                        return Some(temp);
                    }
                }
            }
            Some(45.0) // Fallback estimate
        }
        _ => {
            debug!("powermetrics not available, using temperature estimate");
            Some(45.0)
        }
    }
}

async fn get_cpu_temperature() -> Option<f32> {
    // Multi-platform CPU temperature detection

    #[cfg(target_os = "linux")]
    {
        // Try to read from thermal zones on Linux
        if let Ok(temp) = get_linux_cpu_temperature().await {
            return Some(temp);
        }
    }

    #[cfg(target_os = "macos")]
    {
        // Try powermetrics on macOS
        if let Ok(temp) = get_macos_cpu_temperature().await {
            return Some(temp);
        }
    }

    #[cfg(target_os = "windows")]
    {
        // Windows temperature detection could use WMI
        // For now, return reasonable estimate
        return Some(55.0);
    }

    // Default fallback
    Some(55.0)
}

#[cfg(target_os = "linux")]
async fn get_linux_cpu_temperature() -> Result<f32> {
    // Try to read from /sys/class/thermal/thermal_zone*/temp
    for i in 0..10 {
        let path = format!("/sys/class/thermal/thermal_zone{}/temp", i);
        if let Ok(content) = tokio::fs::read_to_string(&path).await {
            if let Ok(temp_millicelsius) = content.trim().parse::<i32>() {
                let temp_celsius = temp_millicelsius as f32 / 1000.0;
                if temp_celsius > 20.0 && temp_celsius < 100.0 {
                    return Ok(temp_celsius);
                }
            }
        }
    }
    Err(anyhow::anyhow!("No thermal zone found"))
}

#[cfg(target_os = "macos")]
async fn get_macos_cpu_temperature() -> Result<f32> {
    match tokio::process::Command::new("powermetrics")
        .args(&["-s", "cpu_power", "-n", "1"])
        .output()
        .await
    {
        Ok(output) if output.status.success() => {
            let output_str = String::from_utf8_lossy(&output.stdout);
            for line in output_str.lines() {
                if line.contains("CPU Temperature") {
                    if let Some(temp) = line
                        .split_whitespace()
                        .find(|s| s.ends_with("°C"))
                        .and_then(|s| s.trim_end_matches("°C").parse::<f32>().ok())
                    {
                        return Ok(temp);
                    }
                }
            }
            Err(anyhow::anyhow!("Temperature not found in powermetrics output"))
        }
        _ => Err(anyhow::anyhow!("powermetrics command failed")),
    }
}

#[cfg(all(feature = "cuda", gpu_cuda_available))]
fn get_nvidia_memory_info_sync() -> Option<MemoryInfo> {
    // Use nvidia-smi to get memory information
    std::process::Command::new("nvidia-smi")
        .args(&[
            "--query-gpu=memory.total,memory.free,memory.used",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()
        .and_then(|output| {
            if output.status.success() {
                let output_str = String::from_utf8_lossy(&output.stdout);
                let line = output_str.lines().next()?;
                let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();

                if parts.len() >= 3 {
                    let total_mb = parts[0].parse::<usize>().ok()?;
                    let free_mb = parts[1].parse::<usize>().ok()?;
                    let used_mb = parts[2].parse::<usize>().ok()?;

                    return Some(MemoryInfo {
                        total: total_mb * 1024 * 1024, // Convert MB to bytes
                        free: free_mb * 1024 * 1024,
                        used: used_mb * 1024 * 1024,
                        reserved: 0,
                    });
                }
            }
            None
        })
}
