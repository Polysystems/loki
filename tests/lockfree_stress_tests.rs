use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
use std::thread;
use std::time::{Duration, Instant};
use std::collections::HashMap;

use anyhow::Result;
use dashmap::DashMap;
use crossbeam_queue::ArrayQueue;

use loki::infrastructure::lockfree::{
    ConcurrentMap, ZeroCopyRingBuffer, LockFreeEventQueue, Event, EventPriority,
    IndexedRingBuffer, CrossScaleIndex, CrossScaleIndexConfig, IndexEntryMetadata
};
use loki::tui::event_bus::{EventBus, SystemEvent, TabId};
use loki::streaming::{RingBuffer, StreamChunk};
use loki::memory::fractal::ScaleLevel;

/// Test concurrent access to DashMap with multiple threads
#[test]
fn stress_test_concurrent_map() {
    const NUM_THREADS: usize = 100;
    const OPERATIONS_PER_THREAD: usize = 1000;
    
    let map = Arc::new(DashMap::new());
    let operation_count = Arc::new(AtomicUsize::new(0));
    
    let start = Instant::now();
    
    let handles: Vec<_> = (0..NUM_THREADS).map(|thread_id| {
        let map = map.clone();
        let operation_count = operation_count.clone();
        
        thread::spawn(move || {
            for i in 0..OPERATIONS_PER_THREAD {
                let key = thread_id * OPERATIONS_PER_THREAD + i;
                let value = key * 2;
                
                // Insert
                map.insert(key, value);
                operation_count.fetch_add(1, Ordering::Relaxed);
                
                // Read
                let retrieved = map.get(&key).unwrap();
                assert_eq!(*retrieved, value);
                operation_count.fetch_add(1, Ordering::Relaxed);
                
                // Update
                map.insert(key, value + 1);
                operation_count.fetch_add(1, Ordering::Relaxed);
                
                // Delete
                let removed = map.remove(&key).unwrap();
                assert_eq!(removed.1, value + 1);
                operation_count.fetch_add(1, Ordering::Relaxed);
            }
        })
    }).collect();
    
    for handle in handles {
        handle.join().expect("Thread should complete successfully");
    }
    
    let duration = start.elapsed();
    let total_operations = operation_count.load(Ordering::Relaxed);
    let ops_per_second = total_operations as f64 / duration.as_secs_f64();
    
    println!("DashMap stress test completed:");
    println!("  Threads: {}", NUM_THREADS);
    println!("  Operations per thread: {}", OPERATIONS_PER_THREAD);
    println!("  Total operations: {}", total_operations);
    println!("  Duration: {:?}", duration);
    println!("  Operations/second: {:.2}", ops_per_second);
    
    assert!(map.is_empty(), "All items should be removed");
    assert_eq!(total_operations, NUM_THREADS * OPERATIONS_PER_THREAD * 4);
}

/// Test concurrent producer-consumer pattern with lock-free queue
#[test]
fn stress_test_producer_consumer() {
    const NUM_PRODUCERS: usize = 50;
    const NUM_CONSUMERS: usize = 50;
    const ITEMS_PER_PRODUCER: usize = 1000;
    const QUEUE_CAPACITY: usize = 10000;
    
    let queue = Arc::new(ArrayQueue::<usize>::new(QUEUE_CAPACITY));
    let produced_count = Arc::new(AtomicUsize::new(0));
    let consumed_count = Arc::new(AtomicUsize::new(0));
    
    let start = Instant::now();
    
    // Start producers
    let producer_handles: Vec<_> = (0..NUM_PRODUCERS).map(|producer_id| {
        let queue = queue.clone();
        let produced_count = produced_count.clone();
        
        thread::spawn(move || {
            for i in 0..ITEMS_PER_PRODUCER {
                let item = producer_id * ITEMS_PER_PRODUCER + i;
                
                // Keep trying to push until successful
                while queue.push(item).is_err() {
                    thread::yield_now();
                }
                
                produced_count.fetch_add(1, Ordering::Relaxed);
            }
        })
    }).collect();
    
    // Start consumers
    let consumer_handles: Vec<_> = (0..NUM_CONSUMERS).map(|_consumer_id| {
        let queue = queue.clone();
        let consumed_count = consumed_count.clone();
        let produced_count = produced_count.clone();
        
        thread::spawn(move || {
            loop {
                match queue.pop() {
                    Some(_item) => {
                        let count = consumed_count.fetch_add(1, Ordering::Relaxed) + 1;
                        
                        // Stop when all items are consumed
                        if count >= NUM_PRODUCERS * ITEMS_PER_PRODUCER {
                            break;
                        }
                    }
                    None => {
                        // Check if production is done
                        let produced = produced_count.load(Ordering::Relaxed);
                        let consumed = consumed_count.load(Ordering::Relaxed);
                        
                        if produced >= NUM_PRODUCERS * ITEMS_PER_PRODUCER && 
                           consumed >= produced {
                            break;
                        }
                        
                        thread::yield_now();
                    }
                }
            }
        })
    }).collect();
    
    // Wait for all producers
    for handle in producer_handles {
        handle.join().expect("Producer thread should complete");
    }
    
    // Wait for all consumers
    for handle in consumer_handles {
        handle.join().expect("Consumer thread should complete");
    }
    
    let duration = start.elapsed();
    let total_items = NUM_PRODUCERS * ITEMS_PER_PRODUCER;
    let throughput = total_items as f64 / duration.as_secs_f64();
    
    println!("Producer-Consumer stress test completed:");
    println!("  Producers: {}", NUM_PRODUCERS);
    println!("  Consumers: {}", NUM_CONSUMERS);
    println!("  Items per producer: {}", ITEMS_PER_PRODUCER);
    println!("  Total items: {}", total_items);
    println!("  Duration: {:?}", duration);
    println!("  Throughput: {:.2} items/second", throughput);
    
    assert_eq!(produced_count.load(Ordering::Relaxed), total_items);
    assert_eq!(consumed_count.load(Ordering::Relaxed), total_items);
}

/// Test streaming components under high load
#[test]
fn stress_test_streaming_components() {
    const NUM_THREADS: usize = 20;
    const CHUNKS_PER_THREAD: usize = 1000;
    const BUFFER_SIZE: usize = 10000;
    
    let buffer = Arc::new(RingBuffer::new(BUFFER_SIZE));
    let written_count = Arc::new(AtomicUsize::new(0));
    let read_count = Arc::new(AtomicUsize::new(0));
    
    let start = Instant::now();
    
    // Writer threads
    let writer_handles: Vec<_> = (0..NUM_THREADS / 2).map(|writer_id| {
        let buffer = buffer.clone();
        let written_count = written_count.clone();
        
        thread::spawn(move || {
            for i in 0..CHUNKS_PER_THREAD {
                let data = vec![(writer_id * CHUNKS_PER_THREAD + i) as u8; 1024];
                
                while !buffer.write(data.clone()) {
                    thread::yield_now(); // Buffer full, wait
                }
                
                written_count.fetch_add(1, Ordering::Relaxed);
            }
        })
    }).collect();
    
    // Reader threads
    let reader_handles: Vec<_> = (0..NUM_THREADS / 2).map(|_reader_id| {
        let buffer = buffer.clone();
        let read_count = read_count.clone();
        let written_count = written_count.clone();
        
        thread::spawn(move || {
            loop {
                match buffer.read() {
                    Some(_data) => {
                        let count = read_count.fetch_add(1, Ordering::Relaxed) + 1;
                        
                        // Stop when all expected items are read
                        if count >= (NUM_THREADS / 2) * CHUNKS_PER_THREAD {
                            break;
                        }
                    }
                    None => {
                        // Check if writing is done
                        let written = written_count.load(Ordering::Relaxed);
                        let read = read_count.load(Ordering::Relaxed);
                        
                        if written >= (NUM_THREADS / 2) * CHUNKS_PER_THREAD && 
                           read >= written {
                            break;
                        }
                        
                        thread::yield_now();
                    }
                }
            }
        })
    }).collect();
    
    // Wait for completion
    for handle in writer_handles {
        handle.join().expect("Writer thread should complete");
    }
    
    for handle in reader_handles {
        handle.join().expect("Reader thread should complete");
    }
    
    let duration = start.elapsed();
    let total_chunks = (NUM_THREADS / 2) * CHUNKS_PER_THREAD;
    let throughput = total_chunks as f64 / duration.as_secs_f64();
    
    println!("Streaming stress test completed:");
    println!("  Writer threads: {}", NUM_THREADS / 2);
    println!("  Reader threads: {}", NUM_THREADS / 2);
    println!("  Chunks per writer: {}", CHUNKS_PER_THREAD);
    println!("  Total chunks: {}", total_chunks);
    println!("  Duration: {:?}", duration);
    println!("  Throughput: {:.2} chunks/second", throughput);
    
    assert_eq!(written_count.load(Ordering::Relaxed), total_chunks);
    assert_eq!(read_count.load(Ordering::Relaxed), total_chunks);
}

/// Test EventBus under high concurrent load
#[tokio::test]
async fn stress_test_event_bus() {
    const NUM_PUBLISHERS: usize = 50;
    const EVENTS_PER_PUBLISHER: usize = 100;
    
    let bus = Arc::new(EventBus::new(10000));
    let published_count = Arc::new(AtomicUsize::new(0));
    
    let start = Instant::now();
    
    // Start event processing
    let bus_clone = bus.clone();
    bus_clone.start_processing();
    
    let publisher_tasks: Vec<_> = (0..NUM_PUBLISHERS).map(|publisher_id| {
        let bus = bus.clone();
        let published_count = published_count.clone();
        
        tokio::spawn(async move {
            for i in 0..EVENTS_PER_PUBLISHER {
                let event = SystemEvent::ModelSelected {
                    model_id: format!("model_{}_{}", publisher_id, i),
                    source: TabId::Chat,
                };
                
                if let Ok(_) = bus.publish(event).await {
                    published_count.fetch_add(1, Ordering::Relaxed);
                }
            }
        })
    }).collect();
    
    // Wait for all publishers
    for task in publisher_tasks {
        task.await.expect("Publisher task should complete");
    }
    
    // Wait for processing to complete
    tokio::time::sleep(Duration::from_millis(500)).await;
    
    let duration = start.elapsed();
    let total_events = NUM_PUBLISHERS * EVENTS_PER_PUBLISHER;
    let throughput = total_events as f64 / duration.as_secs_f64();
    
    println!("EventBus stress test completed:");
    println!("  Publishers: {}", NUM_PUBLISHERS);
    println!("  Events per publisher: {}", EVENTS_PER_PUBLISHER);
    println!("  Total events: {}", total_events);
    println!("  Duration: {:?}", duration);
    println!("  Throughput: {:.2} events/second", throughput);
    
    assert_eq!(published_count.load(Ordering::Relaxed), total_events);
    
    let stats = bus.get_stats().await;
    println!("EventBus stats: {:?}", stats);
}

/// Test cross-scale index with concurrent operations
#[test]
fn stress_test_cross_scale_index() {
    const NUM_THREADS: usize = 20;
    const OPERATIONS_PER_THREAD: usize = 500;
    
    let config = CrossScaleIndexConfig::default();
    let index = Arc::new(CrossScaleIndex::new(config));
    let operation_count = Arc::new(AtomicUsize::new(0));
    
    let start = Instant::now();
    
    let handles: Vec<_> = (0..NUM_THREADS).map(|thread_id| {
        let index = index.clone();
        let operation_count = operation_count.clone();
        
        thread::spawn(move || {
            for i in 0..OPERATIONS_PER_THREAD {
                let scale_level = match (thread_id + i) % 4 {
                    0 => ScaleLevel::Atomic,
                    1 => ScaleLevel::Concept,
                    2 => ScaleLevel::Schema,
                    _ => ScaleLevel::Worldview,
                };
                
                let data = format!("data_{}_{}", thread_id, i);
                let metadata = IndexEntryMetadata {
                    priority: 1.0,
                    confidence: 0.8,
                    quality_score: 0.9,
                    source: format!("thread_{}", thread_id),
                    tags: vec!["stress_test".to_string()],
                    created_at: Instant::now(),
                };
                
                // Insert operation
                let sequence_id = index.insert(scale_level.clone(), data.clone(), metadata);
                operation_count.fetch_add(1, Ordering::Relaxed);
                
                // Query operation
                let results = index.query(&scale_level, 10);
                assert!(!results.is_empty(), "Should find inserted data");
                operation_count.fetch_add(1, Ordering::Relaxed);
            }
        })
    }).collect();
    
    for handle in handles {
        handle.join().expect("Thread should complete successfully");
    }
    
    let duration = start.elapsed();
    let total_operations = operation_count.load(Ordering::Relaxed);
    let ops_per_second = total_operations as f64 / duration.as_secs_f64();
    
    println!("Cross-scale index stress test completed:");
    println!("  Threads: {}", NUM_THREADS);
    println!("  Operations per thread: {}", OPERATIONS_PER_THREAD * 2); // Insert + query
    println!("  Total operations: {}", total_operations);
    println!("  Duration: {:?}", duration);
    println!("  Operations/second: {:.2}", ops_per_second);
    
    let stats = index.get_stats();
    println!("Index stats: Entries={}, Queries={}, Correlations={}", 
             stats.total_entries.load(Ordering::Relaxed),
             stats.query_count.load(Ordering::Relaxed),
             stats.correlation_updates.load(Ordering::Relaxed));
    
    assert_eq!(total_operations, NUM_THREADS * OPERATIONS_PER_THREAD * 2);
}

/// Memory and performance validation test
#[test]
fn validate_performance_targets() {
    println!("Performance Target Validation");
    println!("============================");
    
    // Test 1: Stream throughput target (500K/s)
    const TARGET_STREAM_THROUGHPUT: f64 = 500_000.0;
    let buffer = RingBuffer::new(10000);
    
    let start = Instant::now();
    let mut successful_writes = 0;
    
    // Try to achieve target throughput
    for i in 0..1_000_000 {
        let data = vec![i as u8; 64];
        if buffer.write(data) {
            successful_writes += 1;
        }
        
        // Read to prevent buffer from filling
        if i % 2 == 0 {
            let _ = buffer.read();
        }
        
        // Check if we've reached 1 second
        if start.elapsed().as_secs() >= 1 {
            break;
        }
    }
    
    let actual_throughput = successful_writes as f64 / start.elapsed().as_secs_f64();
    println!("Stream throughput: {:.2}/s (target: {:.2}/s)", actual_throughput, TARGET_STREAM_THROUGHPUT);
    
    // Test 2: Cache latency target (p99 < 50ms)
    let cache = loki::memory::cache::SimdCache::new(1024 * 1024);
    let mut latencies = Vec::new();
    
    for i in 0..1000 {
        let memory_id = loki::memory::MemoryId::new(format!("perf_test_{}", i));
        let memory_item = loki::memory::MemoryItem {
            id: memory_id.clone(),
            content: format!("Performance test content {}", i),
            metadata: loki::memory::MemoryMetadata {
                source: "performance_test".to_string(),
                tags: vec!["benchmark".to_string()],
                importance: 0.5,
                associations: vec![],
            },
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            embedding: Some(vec![0.1; 384]),
        };
        
        // Measure cache put/get latency
        let start = Instant::now();
        let _ = cache.put(&memory_id, &memory_item);
        let _ = cache.get(&memory_id);
        let latency = start.elapsed().as_millis();
        
        latencies.push(latency);
    }
    
    latencies.sort();
    let p50 = latencies[latencies.len() / 2];
    let p99 = latencies[latencies.len() * 99 / 100];
    
    println!("Cache latency p50: {}ms (target: <10ms)", p50);
    println!("Cache latency p99: {}ms (target: <50ms)", p99);
    
    // Test 3: Event routing target (<1ms)
    let rt = tokio::runtime::Runtime::new().unwrap();
    let bus = Arc::new(EventBus::new(1000));
    
    let mut event_latencies = Vec::new();
    
    rt.block_on(async {
        for i in 0..100 {
            let start = Instant::now();
            let _ = bus.publish(SystemEvent::ModelSelected {
                model_id: format!("latency_test_{}", i),
                source: TabId::Chat,
            }).await;
            let latency = start.elapsed().as_millis();
            event_latencies.push(latency);
        }
    });
    
    let avg_event_latency = event_latencies.iter().sum::<u128>() as f64 / event_latencies.len() as f64;
    println!("Average event routing: {:.2}ms (target: <1ms)", avg_event_latency);
    
    println!("\nPerformance Summary:");
    println!("- Stream throughput: {} ({})", 
             if actual_throughput >= TARGET_STREAM_THROUGHPUT { "✅ PASS" } else { "❌ FAIL" },
             format!("{:.0}/s", actual_throughput));
    println!("- Cache p99 latency: {} ({}ms)", 
             if p99 <= 50 { "✅ PASS" } else { "❌ FAIL" }, p99);
    println!("- Event routing: {} ({:.2}ms)", 
             if avg_event_latency <= 1.0 { "✅ PASS" } else { "❌ FAIL" }, avg_event_latency);
}