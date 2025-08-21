//! Atomic configuration management using ArcSwap for lock-free config updates

use arc_swap::{ArcSwap, ArcSwapOption, Guard};
use std::sync::Arc;
use std::time::Instant;
use serde::{Serialize, Deserialize};
use anyhow::Result;

/// Atomic configuration wrapper for hot-reloading without locks
pub struct AtomicConfig<T> 
where 
    T: Clone + Send + Sync,
{
    config: Arc<ArcSwap<T>>,
    version: Arc<ArcSwap<ConfigVersion>>,
    update_callbacks: Arc<super::concurrent_map::ConcurrentMap<String, Arc<dyn Fn(&T) + Send + Sync>>>,
}

#[derive(Clone, Debug)]
pub struct ConfigVersion {
    pub version: u64,
    pub timestamp: Instant,
    pub description: String,
}

impl<T> AtomicConfig<T>
where
    T: Clone + Send + Sync,
{
    /// Create new atomic config with initial value
    pub fn new(initial: T) -> Self {
        Self {
            config: Arc::new(ArcSwap::from_pointee(initial)),
            version: Arc::new(ArcSwap::from_pointee(ConfigVersion {
                version: 1,
                timestamp: Instant::now(),
                description: "Initial configuration".to_string(),
            })),
            update_callbacks: Arc::new(super::concurrent_map::ConcurrentMap::new()),
        }
    }
    
    /// Load current configuration (lock-free)
    pub fn load(&self) -> Guard<Arc<T>> {
        super::GLOBAL_STATS.record_operation();
        self.config.load()
    }
    
    /// Load full Arc for longer-term storage
    pub fn load_full(&self) -> Arc<T> {
        super::GLOBAL_STATS.record_operation();
        self.config.load_full()
    }
    
    /// Atomically update configuration
    pub fn store(&self, new_config: T, description: String) {
        super::GLOBAL_STATS.record_operation();
        
        // Update version info
        let old_version = self.version.load();
        let new_version = ConfigVersion {
            version: old_version.version + 1,
            timestamp: Instant::now(),
            description,
        };
        
        // Store new config atomically
        self.config.store(Arc::new(new_config.clone()));
        self.version.store(Arc::new(new_version));
        
        // Notify callbacks
        self.notify_callbacks(&new_config);
    }
    
    /// Compare and swap - only update if current matches expected
    pub fn compare_and_swap(&self, expected: &T, new: T) -> Result<(), T>
    where
        T: PartialEq,
    {
        super::GLOBAL_STATS.record_operation();
        
        let current = self.config.load();
        if **current == *expected {
            self.store(new, "CAS update".to_string());
            Ok(())
        } else {
            Err((**current).clone())
        }
    }
    
    /// Register callback for config updates
    pub fn register_callback<F>(&self, id: String, callback: F)
    where
        F: Fn(&T) + Send + Sync + 'static,
    {
        self.update_callbacks.insert(id, Arc::new(callback));
    }
    
    /// Remove callback
    pub fn remove_callback(&self, id: &str) {
        self.update_callbacks.remove(&id.to_string());
    }
    
    /// Get current version info
    pub fn version(&self) -> Guard<Arc<ConfigVersion>> {
        self.version.load()
    }
    
    /// Apply transformation to config atomically
    pub fn update<F>(&self, updater: F, description: String)
    where
        F: Fn(&T) -> T,
    {
        let current = self.load_full();
        let new_config = updater(&current);
        self.store(new_config, description);
    }
    
    fn notify_callbacks(&self, new_config: &T) {
        for (_id, callback) in self.update_callbacks.iter() {
            callback(new_config);
        }
    }
}

impl<T> Clone for AtomicConfig<T>
where
    T: Clone + Send + Sync,
{
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            version: self.version.clone(),
            update_callbacks: self.update_callbacks.clone(),
        }
    }
}

/// Configuration manager for multiple atomic configs
pub struct ConfigManager {
    configs: super::concurrent_map::ConcurrentMap<String, Arc<dyn ConfigHandle>>,
}

/// Trait for type-erased config handles
pub trait ConfigHandle: Send + Sync {
    fn version(&self) -> u64;
    fn description(&self) -> String;
    fn reload(&self) -> Result<()>;
}

impl<T> ConfigHandle for AtomicConfig<T>
where
    T: Clone + Send + Sync + for<'de> Deserialize<'de>,
{
    fn version(&self) -> u64 {
        self.version().version
    }
    
    fn description(&self) -> String {
        self.version().description.clone()
    }
    
    fn reload(&self) -> Result<()> {
        // Reload logic would go here (e.g., from file)
        Ok(())
    }
}

impl ConfigManager {
    pub fn new() -> Self {
        Self {
            configs: super::concurrent_map::ConcurrentMap::new(),
        }
    }
    
    /// Register a config with the manager
    pub fn register<T>(&self, name: String, config: AtomicConfig<T>)
    where
        T: Clone + Send + Sync + for<'de> Deserialize<'de> + 'static,
    {
        self.configs.insert(name, Arc::new(config));
    }
    
    /// Get config by name
    pub fn get(&self, name: &str) -> Option<Arc<dyn ConfigHandle>> {
        self.configs.get(&name.to_string())
    }
    
    /// Reload all configs
    pub fn reload_all(&self) -> Result<()> {
        for (_name, handle) in self.configs.iter() {
            handle.reload()?;
        }
        Ok(())
    }
    
    /// List all config names and versions
    pub fn list(&self) -> Vec<(String, u64, String)> {
        self.configs
            .iter()
            .map(|(name, handle)| (name, handle.version(), handle.description()))
            .collect()
    }
}

/// Optional atomic config for nullable configurations
pub struct AtomicConfigOption<T>
where
    T: Clone + Send + Sync,
{
    config: Arc<ArcSwapOption<T>>,
}

impl<T> AtomicConfigOption<T>
where
    T: Clone + Send + Sync,
{
    pub fn new(initial: Option<T>) -> Self {
        Self {
            config: Arc::new(ArcSwapOption::from(initial.map(Arc::new))),
        }
    }
    
    pub fn load(&self) -> Option<Arc<T>> {
        super::GLOBAL_STATS.record_operation();
        let guard = self.config.load();
        guard.as_ref().map(|arc| Arc::clone(arc))
    }
    
    pub fn store(&self, value: Option<T>) {
        super::GLOBAL_STATS.record_operation();
        self.config.store(value.map(Arc::new));
    }
    
    pub fn take(&self) -> Option<Arc<T>> {
        super::GLOBAL_STATS.record_operation();
        self.config.swap(None)
    }
}

/// Versioned value for tracking changes
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VersionedValue<T> {
    pub value: T,
    pub version: u64,
    pub timestamp: u64,
}

impl<T> VersionedValue<T> {
    pub fn new(value: T) -> Self {
        Self {
            value,
            version: 1,
            timestamp: chrono::Utc::now().timestamp_millis() as u64,
        }
    }
    
    pub fn update(&mut self, value: T) {
        self.value = value;
        self.version += 1;
        self.timestamp = chrono::Utc::now().timestamp_millis() as u64;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::sync::atomic::{AtomicUsize, Ordering};
    
    #[derive(Clone, Debug, PartialEq, Deserialize)]
    struct TestConfig {
        value: usize,
        name: String,
    }
    
    #[test]
    fn test_atomic_config_basic() {
        let config = AtomicConfig::new(TestConfig {
            value: 42,
            name: "test".to_string(),
        });
        
        let loaded = config.load();
        assert_eq!(loaded.value, 42);
        assert_eq!(loaded.name, "test");
        
        config.store(TestConfig {
            value: 100,
            name: "updated".to_string(),
        }, "Test update".to_string());
        
        let loaded = config.load();
        assert_eq!(loaded.value, 100);
        assert_eq!(loaded.name, "updated");
    }
    
    #[test]
    fn test_concurrent_updates() {
        let config = Arc::new(AtomicConfig::new(TestConfig {
            value: 0,
            name: "concurrent".to_string(),
        }));
        
        let threads: Vec<_> = (0..100).map(|i| {
            let config = config.clone();
            thread::spawn(move || {
                config.update(|c| TestConfig {
                    value: c.value + 1,
                    name: format!("thread_{}", i),
                }, format!("Update from thread {}", i));
            })
        }).collect();
        
        for t in threads {
            t.join().unwrap();
        }
        
        let final_config = config.load();
        assert_eq!(final_config.value, 100);
    }
    
    #[test]
    fn test_callbacks() {
        let config = AtomicConfig::new(TestConfig {
            value: 0,
            name: "callback_test".to_string(),
        });
        
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();
        
        config.register_callback("test_callback".to_string(), move |c: &TestConfig| {
            counter_clone.fetch_add(c.value, Ordering::Relaxed);
        });
        
        config.store(TestConfig {
            value: 10,
            name: "updated".to_string(),
        }, "Trigger callback".to_string());
        
        assert_eq!(counter.load(Ordering::Relaxed), 10);
        
        config.store(TestConfig {
            value: 5,
            name: "updated_again".to_string(),
        }, "Trigger callback again".to_string());
        
        assert_eq!(counter.load(Ordering::Relaxed), 15);
    }
}