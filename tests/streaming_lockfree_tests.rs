use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use bytes::Bytes;
use loki::streaming::{RingBuffer, StreamBuffer, ZeroCopyRingBuffer};

#[test]
fn test_lock_free_ring_buffer_concurrent_access() {
    let buffer = Arc::new(RingBuffer::<u64>::new(1000));
    let total_items = 10000;
    let num_producers = 4;
    let num_consumers = 4;
    let items_per_producer = total_items / num_producers;

    // Shared counters
    let items_produced = Arc::new(AtomicUsize::new(0));
    let items_consumed = Arc::new(AtomicUsize::new(0));

    // Spawn producers
    let mut producer_handles = Vec::new();
    for producer_id in 0..num_producers {
        let buffer_clone = buffer.clone();
        let items_produced_clone = items_produced.clone();
        
        let handle = thread::spawn(move || {
            for i in 0..items_per_producer {
                let item = (producer_id as u64) * 1000000 + i as u64;
                
                // Retry until successful (non-blocking)
                while !buffer_clone.write(item) {
                    std::hint::spin_loop();
                }
                
                items_produced_clone.fetch_add(1, Ordering::Relaxed);
            }
        });
        producer_handles.push(handle);
    }

    // Spawn consumers
    let mut consumer_handles = Vec::new();
    for _consumer_id in 0..num_consumers {
        let buffer_clone = buffer.clone();
        let items_consumed_clone = items_consumed.clone();
        let items_produced_clone = items_produced.clone();
        
        let handle = thread::spawn(move || {
            loop {
                if let Some(_item) = buffer_clone.read() {
                    items_consumed_clone.fetch_add(1, Ordering::Relaxed);
                } else {
                    // Check if we're done
                    if items_produced_clone.load(Ordering::Relaxed) == total_items
                        && items_consumed_clone.load(Ordering::Relaxed) 
                           == items_produced_clone.load(Ordering::Relaxed) {
                        break;
                    }
                    std::hint::spin_loop();
                }
            }
        });
        consumer_handles.push(handle);
    }

    // Wait for producers
    for handle in producer_handles {
        handle.join().unwrap();
    }

    // Wait for consumers
    for handle in consumer_handles {
        handle.join().unwrap();
    }

    // Validate results
    assert_eq!(items_produced.load(Ordering::Relaxed), total_items);
    assert_eq!(items_consumed.load(Ordering::Relaxed), total_items);
    assert!(buffer.is_empty());
}

#[test]
fn test_zero_copy_ring_buffer_throughput() {
    let buffer = Arc::new(ZeroCopyRingBuffer::new(1000));
    let num_threads = 8;
    let bytes_per_thread = 1000;
    
    let start_time = Instant::now();
    
    let mut handles = Vec::new();
    
    // Producer threads
    for thread_id in 0..num_threads / 2 {
        let buffer_clone = buffer.clone();
        let handle = thread::spawn(move || {
            for i in 0..bytes_per_thread {
                let data = format!("thread_{}_item_{}", thread_id, i);
                let bytes = Bytes::from(data);
                
                while buffer_clone.write(bytes.clone()).is_err() {
                    std::hint::spin_loop();
                }
            }
        });
        handles.push(handle);
    }
    
    // Consumer threads
    for _thread_id in 0..num_threads / 2 {
        let buffer_clone = buffer.clone();
        let handle = thread::spawn(move || {
            let mut consumed = 0;
            let target = bytes_per_thread * (num_threads / 2);
            
            while consumed < target {
                if buffer_clone.read().is_some() {
                    consumed += 1;
                } else {
                    std::hint::spin_loop();
                }
            }
        });
        handles.push(handle);
    }
    
    // Wait for completion
    for handle in handles {
        handle.join().unwrap();
    }
    
    let elapsed = start_time.elapsed();
    let total_operations = bytes_per_thread * num_threads;
    let ops_per_sec = total_operations as f64 / elapsed.as_secs_f64();
    
    println!("Zero-copy throughput: {:.0} ops/sec", ops_per_sec);
    
    // Should achieve at least 100K ops/sec on modern hardware
    assert!(ops_per_sec > 100_000.0, "Throughput too low: {:.0} ops/sec", ops_per_sec);
}

#[test]
fn test_stream_buffer_atomic_consistency() {
    let buffer = StreamBuffer::new(100);
    let num_writers = 4;
    let num_readers = 4;
    let writes_per_thread = 250;
    
    let writes_completed = Arc::new(AtomicUsize::new(0));
    let reads_completed = Arc::new(AtomicUsize::new(0));
    
    let mut handles = Vec::new();
    
    // Writer threads
    for writer_id in 0..num_writers {
        let buffer = buffer.clone();
        let writes_completed = writes_completed.clone();
        
        let handle = thread::spawn(move || {
            for i in 0..writes_per_thread {
                let data = format!("writer_{}_msg_{}", writer_id, i).into_bytes();
                let sequence = writer_id as u64 * 1000 + i as u64;
                
                while !buffer.write(data.clone(), sequence) {
                    std::hint::spin_loop();
                }
                
                writes_completed.fetch_add(1, Ordering::Relaxed);
            }
        });
        handles.push(handle);
    }
    
    // Reader threads
    for _reader_id in 0..num_readers {
        let buffer = buffer.clone();
        let reads_completed = reads_completed.clone();
        let writes_completed = writes_completed.clone();
        let total_writes = num_writers * writes_per_thread;
        
        let handle = thread::spawn(move || {
            loop {
                if let Some((_data, _metadata)) = buffer.read() {
                    reads_completed.fetch_add(1, Ordering::Relaxed);
                } else {
                    // Check completion
                    if writes_completed.load(Ordering::Relaxed) == total_writes
                        && reads_completed.load(Ordering::Relaxed) == total_writes {
                        break;
                    }
                    std::hint::spin_loop();
                }
            }
        });
        handles.push(handle);
    }
    
    // Wait for all threads
    for handle in handles {
        handle.join().unwrap();
    }
    
    let final_writes = writes_completed.load(Ordering::Relaxed);
    let final_reads = reads_completed.load(Ordering::Relaxed);
    
    assert_eq!(final_writes, num_writers * writes_per_thread);
    assert_eq!(final_reads, num_writers * writes_per_thread);
}

#[test]
fn test_lock_free_buffer_memory_ordering() {
    // Test that our memory ordering guarantees are correct
    let buffer = Arc::new(RingBuffer::<usize>::new(10));
    let iterations = 1000;
    
    let buffer_writer = buffer.clone();
    let buffer_reader = buffer.clone();
    
    let write_handle = thread::spawn(move || {
        for i in 0..iterations {
            while !buffer_writer.write(i) {
                std::hint::spin_loop();
            }
        }
    });
    
    let read_handle = thread::spawn(move || {
        let mut last_value = None;
        let mut count = 0;
        
        while count < iterations {
            if let Some(value) = buffer_reader.read() {
                if let Some(last) = last_value {
                    // Values should be monotonically increasing
                    assert!(value > last, "Memory ordering violation: {} <= {}", value, last);
                }
                last_value = Some(value);
                count += 1;
            } else {
                std::hint::spin_loop();
            }
        }
    });
    
    write_handle.join().unwrap();
    read_handle.join().unwrap();
}

#[test]
fn test_buffer_statistics_accuracy() {
    let buffer = ZeroCopyRingBuffer::new(100);
    
    // Write some data
    for i in 0..50 {
        let data = Bytes::from(format!("test_data_{}", i));
        buffer.write(data).unwrap();
    }
    
    let stats = buffer.stats();
    assert_eq!(stats.current_size, 50);
    assert_eq!(stats.capacity, 100);
    assert!(!stats.bytes_written == 0);
    
    // Read some data
    for _ in 0..25 {
        buffer.read().unwrap();
    }
    
    let stats = buffer.stats();
    assert_eq!(stats.current_size, 25);
    assert!(stats.bytes_read > 0);
    assert!(stats.bytes_written > stats.bytes_read);
}

#[test]
fn test_buffer_timeout_operations() {
    let buffer = ZeroCopyRingBuffer::new(2);
    
    // Fill buffer
    buffer.write(Bytes::from("data1")).unwrap();
    buffer.write(Bytes::from("data2")).unwrap();
    
    // Write should timeout when full
    let start = Instant::now();
    let result = buffer.write_timeout(Bytes::from("data3"), Duration::from_millis(100));
    let elapsed = start.elapsed();
    
    assert!(result.is_err());
    assert!(elapsed >= Duration::from_millis(100));
    assert!(elapsed < Duration::from_millis(150)); // Allow some tolerance
    
    // Read should succeed with timeout
    let result = buffer.read_timeout(Duration::from_millis(100));
    assert!(result.is_ok());
    
    // Now write should succeed
    let result = buffer.write_timeout(Bytes::from("data3"), Duration::from_millis(100));
    assert!(result.is_ok());
}

#[cfg(test)]
mod performance_tests {
    use super::*;
    use std::sync::Mutex;
    
    #[test]
    fn benchmark_lock_free_vs_mutex() {
        const ITERATIONS: usize = 100_000;
        const NUM_THREADS: usize = 4;
        
        // Test lock-free version
        let lock_free_buffer = Arc::new(RingBuffer::<usize>::new(1000));
        let lock_free_start = Instant::now();
        
        let mut lock_free_handles = Vec::new();
        for thread_id in 0..NUM_THREADS {
            let buffer = lock_free_buffer.clone();
            let handle = thread::spawn(move || {
                let offset = thread_id * ITERATIONS;
                for i in 0..ITERATIONS {
                    while !buffer.write(offset + i) {
                        std::hint::spin_loop();
                    }
                }
            });
            lock_free_handles.push(handle);
        }
        
        for handle in lock_free_handles {
            handle.join().unwrap();
        }
        let lock_free_duration = lock_free_start.elapsed();
        
        // Test mutex-based version for comparison
        let mutex_buffer = Arc::new(Mutex::new(Vec::<usize>::new()));
        let mutex_start = Instant::now();
        
        let mut mutex_handles = Vec::new();
        for thread_id in 0..NUM_THREADS {
            let buffer = mutex_buffer.clone();
            let handle = thread::spawn(move || {
                let offset = thread_id * ITERATIONS;
                for i in 0..ITERATIONS {
                    let mut vec = buffer.lock().unwrap();
                    vec.push(offset + i);
                }
            });
            mutex_handles.push(handle);
        }
        
        for handle in mutex_handles {
            handle.join().unwrap();
        }
        let mutex_duration = mutex_start.elapsed();
        
        let speedup = mutex_duration.as_nanos() as f64 / lock_free_duration.as_nanos() as f64;
        println!("Lock-free vs Mutex speedup: {:.2}x", speedup);
        println!("Lock-free time: {:?}", lock_free_duration);
        println!("Mutex time: {:?}", mutex_duration);
        
        // Lock-free should be at least as fast as mutex-based
        assert!(lock_free_duration <= mutex_duration, 
                "Lock-free implementation slower than mutex: {:.2}x", 1.0 / speedup);
    }
}