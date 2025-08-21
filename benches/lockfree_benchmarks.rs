use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use std::thread;
use std::time::Duration;
use parking_lot::RwLock;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main, Throughput};
use dashmap::DashMap;
use crossbeam_queue::ArrayQueue;
use tokio::runtime::Runtime;

use loki::infrastructure::lockfree::{
    ConcurrentMap, ZeroCopyRingBuffer, LockFreeEventQueue, Event, EventPriority, IndexedRingBuffer,
    CrossScaleIndex, CrossScaleIndexConfig, AtomicContextAnalytics,
    ContextErrorType
};
use loki::tui::event_bus::{EventBus, SystemEvent, TabId, EventHistoryEntry};
use loki::streaming::{RingBuffer, StreamBuffer, StreamChunk};
use loki::memory::cache::SimdCache;

/// Benchmark lock-free HashMap vs traditional RwLock HashMap
fn concurrent_map_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_map");
    
    // Set throughput for better measurement
    group.throughput(Throughput::Elements(1000));
    
    // Benchmark DashMap
    group.bench_function("dashmap_insert", |b| {
        let map = Arc::new(DashMap::new());
        b.iter(|| {
            for i in 0..1000 {
                map.insert(black_box(i), black_box(i * 2));
            }
            map.clear();
        });
    });
    
    // Benchmark RwLock HashMap for comparison
    group.bench_function("rwlock_hashmap_insert", |b| {
        let map = Arc::new(RwLock::new(HashMap::new()));
        b.iter(|| {
            for i in 0..1000 {
                map.write().insert(black_box(i), black_box(i * 2));
            }
            map.write().clear();
        });
    });
    
    // Benchmark concurrent reads
    group.bench_function("dashmap_read", |b| {
        let map = Arc::new(DashMap::new());
        // Pre-populate
        for i in 0..1000 {
            map.insert(i, i * 2);
        }
        
        b.iter(|| {
            for i in 0..1000 {
                let _ = map.get(&black_box(i));
            }
        });
    });
    
    group.bench_function("rwlock_hashmap_read", |b| {
        let map = Arc::new(RwLock::new(HashMap::new()));
        // Pre-populate
        {
            let mut guard = map.write();
            for i in 0..1000 {
                guard.insert(i, i * 2);
            }
        }
        
        b.iter(|| {
            for i in 0..1000 {
                let guard = map.read();
                let _ = guard.get(&black_box(i));
            }
        });
    });
    
    group.finish();
}

/// Benchmark lock-free queue vs traditional Mutex queue
fn queue_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("queue");
    group.throughput(Throughput::Elements(10000));
    
    // ArrayQueue benchmark
    group.bench_function("arrayqueue", |b| {
        let queue = Arc::new(ArrayQueue::<u64>::new(10000));
        b.iter(|| {
            // Fill queue
            for i in 0..5000 {
                let _ = queue.push(black_box(i));
            }
            // Drain queue
            for _ in 0..5000 {
                let _ = queue.pop();
            }
        });
    });
    
    // Mutex VecDeque for comparison
    group.bench_function("mutex_vecdeque", |b| {
        let queue = Arc::new(Mutex::new(std::collections::VecDeque::<u64>::new()));
        b.iter(|| {
            // Fill queue
            for i in 0..5000 {
                queue.lock().unwrap().push_back(black_box(i));
            }
            // Drain queue
            for _ in 0..5000 {
                let _ = queue.lock().unwrap().pop_front();
            }
        });
    });
    
    group.finish();
}

/// Benchmark stream processing components
fn streaming_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming");
    group.throughput(Throughput::Elements(1000));
    
    // Lock-free RingBuffer
    group.bench_function("lockfree_ringbuffer", |b| {
        let buffer = RingBuffer::new(1000);
        b.iter(|| {
            // Write phase
            for i in 0..500 {
                buffer.write(black_box(vec![i as u8; 64]));
            }
            // Read phase
            for _ in 0..500 {
                let _ = buffer.read();
            }
        });
    });
    
    // StreamBuffer with lock-free components
    group.bench_function("stream_buffer", |b| {
        let buffer = StreamBuffer::new(1000);
        b.iter(|| {
            // Write phase
            for i in 0..500 {
                let chunk = StreamChunk {
                    data: black_box(vec![i as u8; 64].into()),
                    sequence: black_box(i),
                    timestamp: std::time::Instant::now(),
                };
                let _ = buffer.write(chunk);
            }
            // Read phase
            for _ in 0..500 {
                let _ = buffer.read();
            }
        });
    });
    
    group.finish();
}

/// Benchmark event bus performance
fn event_bus_benchmarks(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("event_bus");
    group.throughput(Throughput::Elements(100));
    
    group.bench_function("event_publishing", |b| {
        let bus = Arc::new(EventBus::new(1000));
        
        b.iter(|| {
            rt.block_on(async {
                for i in 0..100 {
                    let _ = bus.publish(black_box(SystemEvent::ModelSelected {
                        model_id: format!("model_{}", i),
                        source: TabId::Chat,
                    })).await;
                }
            });
        });
    });
    
    group.finish();
}

/// Benchmark memory cache performance
fn memory_cache_benchmarks(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("memory_cache");
    group.throughput(Throughput::Elements(1000));
    
    group.bench_function("cache_operations", |b| {
        let cache = SimdCache::new(1024 * 1024); // 1MB cache
        
        b.iter(|| {
            rt.block_on(async {
                // Simulate cache operations
                for i in 0..1000 {
                    let memory_id = loki::memory::MemoryId::new(black_box(format!("item_{}", i)));
                    let memory_item = loki::memory::MemoryItem {
                        id: memory_id.clone(),
                        content: black_box(format!("Content for item {}", i)),
                        metadata: loki::memory::MemoryMetadata {
                            source: "benchmark".to_string(),
                            tags: vec!["performance".to_string()],
                            importance: 0.5,
                            associations: vec![],
                        },
                        created_at: chrono::Utc::now(),
                        updated_at: chrono::Utc::now(),
                        embedding: Some(vec![0.1; 384]),
                    };
                    
                    // Put and get operations
                    let _ = cache.put(&memory_id, &memory_item);
                    let _ = cache.get(&memory_id);
                }
            });
        });
    });
    
    group.finish();
}

/// Stress test with high concurrency
fn concurrent_stress_test(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_stress");
    group.sample_size(10); // Fewer samples for stress tests
    
    group.bench_function("high_concurrency_dashmap", |b| {
        b.iter(|| {
            let map = Arc::new(DashMap::new());
            let handles: Vec<_> = (0..100).map(|thread_id| {
                let map = map.clone();
                thread::spawn(move || {
                    for i in 0..100 {
                        let key = thread_id * 1000 + i;
                        map.insert(black_box(key), black_box(key * 2));
                        let _ = map.get(&key);
                        map.remove(&key);
                    }
                })
            }).collect();
            
            for handle in handles {
                handle.join().unwrap();
            }
        });
    });
    
    group.bench_function("high_concurrency_queue", |b| {
        b.iter(|| {
            let queue = Arc::new(ArrayQueue::<usize>::new(10000));
            let handles: Vec<_> = (0..50).map(|thread_id| {
                let queue = queue.clone();
                thread::spawn(move || {
                    // Producer
                    for i in 0..100 {
                        let item = thread_id * 1000 + i;
                        while queue.push(black_box(item)).is_err() {
                            thread::yield_now();
                        }
                    }
                })
            }).collect();
            
            // Consumer
            let consumer_handles: Vec<_> = (0..50).map(|_| {
                let queue = queue.clone();
                thread::spawn(move || {
                    for _ in 0..100 {
                        while queue.pop().is_none() {
                            thread::yield_now();
                        }
                    }
                })
            }).collect();
            
            for handle in handles {
                handle.join().unwrap();
            }
            
            for handle in consumer_handles {
                handle.join().unwrap();
            }
        });
    });
    
    group.finish();
}

/// Benchmark cross-scale indexing performance
fn cross_scale_index_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("cross_scale_index");
    group.throughput(Throughput::Elements(1000));
    
    group.bench_function("index_operations", |b| {
        let config = CrossScaleIndexConfig::default();
        let index = CrossScaleIndex::new(config);
        
        b.iter(|| {
            // Insert operations across different scales
            for i in 0..1000 {
                let scale_level = match i % 4 {
                    0 => loki::memory::fractal::ScaleLevel::Atomic,
                    1 => loki::memory::fractal::ScaleLevel::Concept,
                    2 => loki::memory::fractal::ScaleLevel::Schema,
                    _ => loki::memory::fractal::ScaleLevel::Worldview,
                };
                
                let data = format!("data_{}", i);
                let metadata = loki::infrastructure::lockfree::IndexEntryMetadata {
                    priority: 1.0,
                    confidence: 0.8,
                    quality_score: 0.9,
                    source: "benchmark".to_string(),
                    tags: vec!["performance".to_string()],
                    created_at: std::time::Instant::now(),
                };
                
                let _ = index.insert(black_box(scale_level), black_box(data), black_box(metadata));
            }
        });
    });
    
    group.finish();
}

criterion_group!(
    lockfree_benches,
    concurrent_map_benchmarks,
    queue_benchmarks,
    streaming_benchmarks,
    event_bus_benchmarks,
    memory_cache_benchmarks,
    concurrent_stress_test,
    cross_scale_index_benchmarks
);

criterion_main!(lockfree_benches);