//! Integration tests for lock-free infrastructure

use loki::infrastructure::lockfree::*;
use std::sync::Arc;
use std::thread;
use bytes::Bytes;

#[test]
fn test_concurrent_map() {
    let map = Arc::new(ConcurrentMap::new());
    
    // Spawn multiple threads to test concurrent access
    let threads: Vec<_> = (0..10).map(|i| {
        let map = map.clone();
        thread::spawn(move || {
            for j in 0..100 {
                let key = format!("key_{}_{}", i, j);
                let value = i * 100 + j;
                map.insert(key.clone(), value);
                assert_eq!(map.get(&key), Some(value));
            }
        })
    }).collect();
    
    // Wait for all threads
    for t in threads {
        t.join().unwrap();
    }
    
    // Verify all entries
    assert_eq!(map.len(), 1000);
}

#[test]
fn test_atomic_config() {
    #[derive(Clone, Debug, PartialEq)]
    struct Config {
        value: usize,
        name: String,
    }
    
    let config = AtomicConfig::new(Config {
        value: 42,
        name: "initial".to_string(),
    });
    
    // Test concurrent updates
    let threads: Vec<_> = (0..10).map(|i| {
        let config = config.clone();
        thread::spawn(move || {
            config.update(|c| Config {
                value: c.value + 1,
                name: format!("thread_{}", i),
            }, format!("Update {}", i));
        })
    }).collect();
    
    for t in threads {
        t.join().unwrap();
    }
    
    // Value should have been incremented 10 times
    let final_config = config.load();
    assert_eq!(final_config.value, 52);
}

#[test]
fn test_zero_copy_ring_buffer() {
    let buffer = ZeroCopyRingBuffer::new(10);
    
    // Test write and read
    let data = Bytes::from("Hello, World!");
    buffer.write(data.clone()).unwrap();
    
    let read_data = buffer.read().unwrap();
    assert_eq!(read_data, data);
    
    // Test buffer full
    for i in 0..10 {
        let data = Bytes::from(format!("Data {}", i));
        let _ = buffer.write(data);
    }
    
    // Should fail - buffer full
    let result = buffer.write(Bytes::from("overflow"));
    assert!(result.is_err());
}

#[test]
fn test_event_queue_priority() {
    let queue = LockFreeEventQueue::new(100);
    
    // Push events with different priorities
    queue.push(Event::new(
        "test".to_string(),
        Bytes::from("low"),
        EventPriority::Low
    )).unwrap();
    
    queue.push(Event::new(
        "test".to_string(),
        Bytes::from("critical"),
        EventPriority::Critical
    )).unwrap();
    
    queue.push(Event::new(
        "test".to_string(),
        Bytes::from("normal"),
        EventPriority::Normal
    )).unwrap();
    
    queue.push(Event::new(
        "test".to_string(),
        Bytes::from("high"),
        EventPriority::High
    )).unwrap();
    
    // Should pop in priority order
    assert_eq!(queue.pop().unwrap().payload, Bytes::from("critical"));
    assert_eq!(queue.pop().unwrap().payload, Bytes::from("high"));
    assert_eq!(queue.pop().unwrap().payload, Bytes::from("normal"));
    assert_eq!(queue.pop().unwrap().payload, Bytes::from("low"));
}

#[test]
fn test_simd_cache() {
    let cache = SimdCacheLine::<256>::new();
    
    // Test set and get
    let key = b"test_key";
    let value = 42u64;
    
    cache.set(key, value);
    assert_eq!(cache.get(key), Some(value));
    
    // Test batch operations
    let entries = vec![
        (&b"key1"[..], 10u64),
        (&b"key2"[..], 20u64),
        (&b"key3"[..], 30u64),
    ];
    
    cache.batch_set(&entries);
    
    let keys = vec![&b"key1"[..], &b"key2"[..], &b"key3"[..]];
    let results = cache.batch_get(&keys);
    
    assert_eq!(results[0], Some(10));
    assert_eq!(results[1], Some(20));
    assert_eq!(results[2], Some(30));
}

#[test]
fn test_lock_free_performance() {
    use std::time::Instant;
    
    let map = Arc::new(ConcurrentMap::new());
    let num_threads = 10;
    let ops_per_thread = 10000;
    
    let start = Instant::now();
    
    let threads: Vec<_> = (0..num_threads).map(|i| {
        let map = map.clone();
        thread::spawn(move || {
            for j in 0..ops_per_thread {
                let key = format!("key_{}_{}", i, j);
                map.insert(key.clone(), j);
                let _ = map.get(&key);
            }
        })
    }).collect();
    
    for t in threads {
        t.join().unwrap();
    }
    
    let duration = start.elapsed();
    let total_ops = num_threads * ops_per_thread * 2; // insert + get
    let ops_per_sec = total_ops as f64 / duration.as_secs_f64();
    
    println!("Lock-free performance: {:.0} ops/sec", ops_per_sec);
    
    // Should achieve at least 100K ops/sec
    assert!(ops_per_sec > 100_000.0, "Performance too low: {:.0} ops/sec", ops_per_sec);
}