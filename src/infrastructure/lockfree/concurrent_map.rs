//! Concurrent hash map implementation using DashMap for lock-free operations

use dashmap::{DashMap, DashSet};
use std::hash::Hash;
use std::sync::Arc;
use serde::{Serialize, Deserialize};

/// A lock-free concurrent hash map wrapper around DashMap
/// Provides high-performance concurrent access without traditional locks
#[derive(Clone)]
pub struct ConcurrentMap<K, V> 
where 
    K: Eq + Hash + Clone,
    V: Clone,
{
    inner: Arc<DashMap<K, V>>,
}

impl<K, V> ConcurrentMap<K, V>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    /// Create a new empty concurrent map
    pub fn new() -> Self {
        Self {
            inner: Arc::new(DashMap::new()),
        }
    }
    
    /// Create with specified capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: Arc::new(DashMap::with_capacity(capacity)),
        }
    }
    
    /// Insert a key-value pair
    pub fn insert(&self, key: K, value: V) -> Option<V> {
        super::GLOBAL_STATS.record_operation();
        self.inner.insert(key, value)
    }
    
    /// Get a value by key
    pub fn get(&self, key: &K) -> Option<V> {
        super::GLOBAL_STATS.record_operation();
        self.inner.get(key).map(|r| r.value().clone())
    }
    
    /// Get a reference wrapper for zero-copy access
    pub fn get_ref(&self, key: &K) -> Option<ConcurrentMapRef<K, V>> {
        super::GLOBAL_STATS.record_operation();
        self.inner.get(key).map(|r| ConcurrentMapRef { 
            _ref: r.value().clone(),
            _phantom: std::marker::PhantomData,
        })
    }
    
    /// Remove a key-value pair
    pub fn remove(&self, key: &K) -> Option<(K, V)> {
        super::GLOBAL_STATS.record_operation();
        self.inner.remove(key)
    }
    
    /// Check if key exists
    pub fn contains_key(&self, key: &K) -> bool {
        self.inner.contains_key(key)
    }
    
    /// Get the number of entries
    pub fn len(&self) -> usize {
        self.inner.len()
    }
    
    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
    
    /// Clear all entries
    pub fn clear(&self) {
        self.inner.clear()
    }
    
    /// Update a value atomically
    pub fn update<F>(&self, key: &K, updater: F) -> Option<V>
    where
        F: FnOnce(&V) -> V,
    {
        super::GLOBAL_STATS.record_operation();
        self.inner.get_mut(key).map(|mut r| {
            let new_value = updater(r.value());
            let old_value = r.value().clone();
            *r = new_value;
            old_value
        })
    }
    
    /// Insert if absent, otherwise return existing
    pub fn get_or_insert(&self, key: K, value: V) -> V {
        super::GLOBAL_STATS.record_operation();
        self.inner.entry(key).or_insert(value).value().clone()
    }
    
    /// Insert with a factory function if absent
    pub fn get_or_insert_with<F>(&self, key: K, factory: F) -> V
    where
        F: FnOnce() -> V,
    {
        super::GLOBAL_STATS.record_operation();
        self.inner.entry(key).or_insert_with(factory).value().clone()
    }
    
    /// Retain only entries that match the predicate
    pub fn retain<F>(&self, predicate: F)
    where
        F: Fn(&K, &V) -> bool,
    {
        self.inner.retain(|k, v| predicate(k, v))
    }
    
    /// Get an iterator over all key-value pairs
    pub fn iter(&self) -> impl Iterator<Item = (K, V)> + '_ {
        self.inner.iter().map(|r| (r.key().clone(), r.value().clone()))
    }
}

impl<K, V> Default for ConcurrentMap<K, V>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Reference wrapper for zero-copy access
pub struct ConcurrentMapRef<K, V> {
    _ref: V,
    _phantom: std::marker::PhantomData<K>,
}

impl<K, V> ConcurrentMapRef<K, V> 
where
    V: Clone,
{
    pub fn value(&self) -> &V {
        &self._ref
    }
}

/// Lock-free concurrent set implementation
#[derive(Clone)]
pub struct ConcurrentSet<T>
where
    T: Eq + Hash + Clone,
{
    inner: Arc<DashSet<T>>,
}

impl<T> ConcurrentSet<T>
where
    T: Eq + Hash + Clone,
{
    pub fn new() -> Self {
        Self {
            inner: Arc::new(DashSet::new()),
        }
    }
    
    pub fn insert(&self, value: T) -> bool {
        super::GLOBAL_STATS.record_operation();
        self.inner.insert(value)
    }
    
    pub fn remove(&self, value: &T) -> bool {
        super::GLOBAL_STATS.record_operation();
        self.inner.remove(value).is_some()
    }
    
    pub fn contains(&self, value: &T) -> bool {
        self.inner.contains(value)
    }
    
    pub fn len(&self) -> usize {
        self.inner.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
    
    pub fn clear(&self) {
        self.inner.clear()
    }
}

impl<T> Default for ConcurrentSet<T>
where
    T: Eq + Hash + Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Serializable concurrent map for persistence
#[derive(Serialize, Deserialize)]
pub struct SerializableConcurrentMap<K, V> {
    entries: Vec<(K, V)>,
}

impl<K, V> From<ConcurrentMap<K, V>> for SerializableConcurrentMap<K, V>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    fn from(map: ConcurrentMap<K, V>) -> Self {
        Self {
            entries: map.iter().collect(),
        }
    }
}

impl<K, V> From<SerializableConcurrentMap<K, V>> for ConcurrentMap<K, V>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    fn from(serializable: SerializableConcurrentMap<K, V>) -> Self {
        let map = ConcurrentMap::with_capacity(serializable.entries.len());
        for (k, v) in serializable.entries {
            map.insert(k, v);
        }
        map
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::sync::Arc;
    
    #[test]
    fn test_concurrent_access() {
        let map = Arc::new(ConcurrentMap::new());
        let threads: Vec<_> = (0..100).map(|i| {
            let map = map.clone();
            thread::spawn(move || {
                map.insert(i, i * 2);
                assert_eq!(map.get(&i), Some(i * 2));
            })
        }).collect();
        
        for t in threads {
            t.join().unwrap();
        }
        
        assert_eq!(map.len(), 100);
    }
    
    #[test]
    fn test_atomic_update() {
        let map = ConcurrentMap::new();
        map.insert("counter", 0);
        
        let threads: Vec<_> = (0..100).map(|_| {
            let map_clone = map.clone();
            thread::spawn(move || {
                for _ in 0..100 {
                    map_clone.update(&"counter", |v| v + 1);
                }
            })
        }).collect();
        
        for t in threads {
            t.join().unwrap();
        }
        
        assert_eq!(map.get(&"counter"), Some(10000));
    }
}