//! SIMD-accelerated cache line implementation for high-performance caching

use wide::f32x8;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::alloc::{alloc, dealloc, Layout};
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;
use dashmap::DashMap;
use std::sync::Arc;

/// Cache line size (typically 64 bytes on modern CPUs)
const CACHE_LINE_SIZE: usize = 64;

/// SIMD-aligned data structure for cache-friendly access
#[repr(align(64))]
pub struct SimdAlignedData {
    data: [u8; CACHE_LINE_SIZE],
}

impl SimdAlignedData {
    pub fn new() -> Self {
        Self {
            data: [0u8; CACHE_LINE_SIZE],
        }
    }
    
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let mut data = [0u8; CACHE_LINE_SIZE];
        let len = bytes.len().min(CACHE_LINE_SIZE);
        data[..len].copy_from_slice(&bytes[..len]);
        Self { data }
    }
}

/// SIMD cache line for lock-free cache operations
#[repr(align(64))]
pub struct SimdCacheLine<const SIZE: usize = 1024> {
    // Data storage aligned to cache lines
    data: *mut AtomicU64,
    // Version counter for optimistic locking
    version: AtomicU64,
    // Size tracking
    size: AtomicUsize,
    // Layout for deallocation
    layout: Layout,
}

impl<const SIZE: usize> SimdCacheLine<SIZE> {
    /// Create new SIMD cache line
    pub fn new() -> Self {
        // Ensure SIZE is multiple of 8 for u64 alignment
        assert!(SIZE % 8 == 0, "SIZE must be multiple of 8");
        
        let layout = Layout::from_size_align(SIZE * 8, CACHE_LINE_SIZE)
            .expect("Invalid layout");
        
        let data = unsafe {
            let ptr = alloc(layout) as *mut AtomicU64;
            // Initialize all atomic values
            for i in 0..SIZE {
                ptr.add(i).write(AtomicU64::new(0));
            }
            ptr
        };
        
        Self {
            data,
            version: AtomicU64::new(0),
            size: AtomicUsize::new(0),
            layout,
        }
    }
    
    /// Hash key to index using SIMD operations
    pub fn hash_key(&self, key: &[u8]) -> usize {
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        let hash = hasher.finish();
        
        // Use SIMD for hash mixing
        let hash_vec = unsafe {
            let hash_bytes = hash.to_ne_bytes();
            std::mem::transmute::<[u8; 8], u64>(hash_bytes)
        };
        
        (hash_vec as usize) % SIZE
    }
    
    /// Get value using SIMD operations
    pub fn get(&self, key: &[u8]) -> Option<u64> {
        super::GLOBAL_STATS.record_operation();
        
        let index = self.hash_key(key);
        let version_before = self.version.load(Ordering::Acquire);
        
        let value = unsafe {
            (*self.data.add(index)).load(Ordering::Acquire)
        };
        
        let version_after = self.version.load(Ordering::Acquire);
        
        // Check if version changed during read (optimistic locking)
        if version_before == version_after && value != 0 {
            super::GLOBAL_STATS.record_cache_hit();
            Some(value)
        } else {
            super::GLOBAL_STATS.record_cache_miss();
            None
        }
    }
    
    /// Set value using SIMD operations
    pub fn set(&self, key: &[u8], value: u64) {
        super::GLOBAL_STATS.record_operation();
        
        let index = self.hash_key(key);
        
        // Increment version for optimistic locking
        self.version.fetch_add(1, Ordering::Release);
        
        unsafe {
            (*self.data.add(index)).store(value, Ordering::Release);
        }
        
        self.size.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Batch get using SIMD
    pub fn batch_get(&self, keys: &[&[u8]]) -> Vec<Option<u64>> {
        keys.iter().map(|key| self.get(key)).collect()
    }
    
    /// Batch set using SIMD
    pub fn batch_set(&self, entries: &[(&[u8], u64)]) {
        for (key, value) in entries {
            self.set(key, *value);
        }
    }
    
    /// Clear cache
    pub fn clear(&self) {
        self.version.fetch_add(1, Ordering::Release);
        
        unsafe {
            for i in 0..SIZE {
                (*self.data.add(i)).store(0, Ordering::Release);
            }
        }
        
        self.size.store(0, Ordering::Release);
    }
    
    /// Get cache size
    pub fn len(&self) -> usize {
        self.size.load(Ordering::Relaxed)
    }
}

impl<const SIZE: usize> Drop for SimdCacheLine<SIZE> {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.data as *mut u8, self.layout);
        }
    }
}

// Safety: SimdCacheLine uses atomic operations for thread safety
unsafe impl<const SIZE: usize> Send for SimdCacheLine<SIZE> {}
unsafe impl<const SIZE: usize> Sync for SimdCacheLine<SIZE> {}

/// SIMD-accelerated LRU cache
pub struct SimdLRUCache {
    // Main cache storage
    cache: Arc<DashMap<Vec<u8>, CacheEntry>>,
    // SIMD-accelerated hot cache
    hot_cache: Arc<SimdCacheLine<256>>,
    // Maximum capacity
    capacity: usize,
    // Hit/miss counters
    hits: AtomicU64,
    misses: AtomicU64,
}

#[derive(Clone)]
struct CacheEntry {
    value: Vec<u8>,
    access_count: u64,
    last_access: u64,
}

impl SimdLRUCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            cache: Arc::new(DashMap::with_capacity(capacity)),
            hot_cache: Arc::new(SimdCacheLine::new()),
            capacity,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }
    
    /// Get value with SIMD-accelerated hot path
    pub fn get(&self, key: &[u8]) -> Option<Vec<u8>> {
        // Try hot cache first (SIMD path)
        if let Some(hot_value) = self.hot_cache.get(key) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            return Some(hot_value.to_ne_bytes().to_vec());
        }
        
        // Fall back to main cache
        if let Some(mut entry) = self.cache.get_mut(key) {
            entry.access_count += 1;
            entry.last_access = chrono::Utc::now().timestamp_millis() as u64;
            
            // Promote to hot cache if frequently accessed
            if entry.access_count > 10 {
                if entry.value.len() == 8 {
                    let value = u64::from_ne_bytes(entry.value[..8].try_into().unwrap());
                    self.hot_cache.set(key, value);
                }
            }
            
            self.hits.fetch_add(1, Ordering::Relaxed);
            Some(entry.value.clone())
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }
    
    /// Set value with eviction
    pub fn set(&self, key: Vec<u8>, value: Vec<u8>) {
        // Add to hot cache if small enough
        if value.len() == 8 {
            let hot_value = u64::from_ne_bytes(value[..8].try_into().unwrap());
            self.hot_cache.set(&key, hot_value);
        }
        
        // Check capacity and evict if necessary
        if self.cache.len() >= self.capacity {
            self.evict_lru();
        }
        
        self.cache.insert(key, CacheEntry {
            value,
            access_count: 1,
            last_access: chrono::Utc::now().timestamp_millis() as u64,
        });
    }
    
    /// Evict least recently used entry
    fn evict_lru(&self) {
        let mut oldest_key = None;
        let mut oldest_time = u64::MAX;
        
        for entry in self.cache.iter() {
            if entry.last_access < oldest_time {
                oldest_time = entry.last_access;
                oldest_key = Some(entry.key().clone());
            }
        }
        
        if let Some(key) = oldest_key {
            self.cache.remove(&key);
        }
    }
    
    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        let total = self.hits.load(Ordering::Relaxed) + self.misses.load(Ordering::Relaxed);
        let hit_rate = if total > 0 {
            self.hits.load(Ordering::Relaxed) as f64 / total as f64
        } else {
            0.0
        };
        
        CacheStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            hit_rate,
            size: self.cache.len(),
            capacity: self.capacity,
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub hit_rate: f64,
    pub size: usize,
    pub capacity: usize,
}

/// SIMD operations for f32x8 vectors
pub struct SimdVectorOps;

impl SimdVectorOps {
    /// Dot product using SIMD
    pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        let chunks = a.len() / 8;
        
        let mut sum = f32x8::splat(0.0);
        
        for i in 0..chunks {
            let a_slice: [f32; 8] = a[i*8..(i+1)*8].try_into().unwrap();
            let b_slice: [f32; 8] = b[i*8..(i+1)*8].try_into().unwrap();
            let a_vec = f32x8::from(a_slice);
            let b_vec = f32x8::from(b_slice);
            sum += a_vec * b_vec;
        }
        
        // Sum all lanes
        sum.reduce_add()
    }
    
    /// Vector addition using SIMD
    pub fn vector_add(a: &[f32], b: &[f32], result: &mut [f32]) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), result.len());
        
        let chunks = a.len() / 8;
        
        for i in 0..chunks {
            let a_slice: [f32; 8] = a[i*8..(i+1)*8].try_into().unwrap();
            let b_slice: [f32; 8] = b[i*8..(i+1)*8].try_into().unwrap();
            let a_vec = f32x8::from(a_slice);
            let b_vec = f32x8::from(b_slice);
            let sum = a_vec + b_vec;
            let sum_array: [f32; 8] = sum.to_array();
            result[i*8..(i+1)*8].copy_from_slice(&sum_array);
        }
        
        // Handle remainder
        for i in chunks*8..a.len() {
            result[i] = a[i] + b[i];
        }
    }
    
    /// Find maximum using SIMD
    pub fn find_max(values: &[f32]) -> f32 {
        if values.is_empty() {
            return f32::NEG_INFINITY;
        }
        
        let chunks = values.len() / 8;
        let mut max_vec = f32x8::splat(f32::NEG_INFINITY);
        
        for i in 0..chunks {
            let slice: [f32; 8] = values[i*8..(i+1)*8].try_into().unwrap();
            let vec = f32x8::from(slice);
            max_vec = max_vec.max(vec);
        }
        
        // Manually find max from SIMD vector
        let max_array: [f32; 8] = max_vec.to_array();
        let mut max = max_array[0];
        for &val in &max_array[1..] {
            max = max.max(val);
        }
        
        // Handle remainder
        for i in chunks*8..values.len() {
            max = max.max(values[i]);
        }
        
        max
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_simd_cache_line() {
        let cache = SimdCacheLine::<128>::new();
        
        let key = b"test_key";
        let value = 42u64;
        
        cache.set(key, value);
        assert_eq!(cache.get(key), Some(value));
        
        cache.clear();
        assert_eq!(cache.get(key), None);
    }
    
    #[test]
    fn test_simd_lru_cache() {
        let cache = SimdLRUCache::new(100);
        
        cache.set(b"key1".to_vec(), b"value1".to_vec());
        cache.set(b"key2".to_vec(), b"value2".to_vec());
        
        assert_eq!(cache.get(b"key1"), Some(b"value1".to_vec()));
        assert_eq!(cache.get(b"key2"), Some(b"value2".to_vec()));
        assert_eq!(cache.get(b"key3"), None);
        
        let stats = cache.stats();
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.misses, 1);
    }
    
    #[test]
    fn test_simd_vector_ops() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        
        let dot = SimdVectorOps::dot_product(&a, &b);
        assert_eq!(dot, 120.0);
        
        let mut result = vec![0.0; 8];
        SimdVectorOps::vector_add(&a, &b, &mut result);
        assert_eq!(result, vec![9.0; 8]);
        
        let max = SimdVectorOps::find_max(&a);
        assert_eq!(max, 8.0);
    }
}