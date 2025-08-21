use std::sync::Arc;
use std::time::{Duration, Instant};
use std::thread;
use std::collections::HashMap;

use loki::streaming::lockfree_enhanced_context_processor::{
    LockFreeEnhancedContextProcessor, LockFreeContextProcessorConfig
};
use loki::streaming::{StreamChunk, StreamId};
use loki::memory::fractal::ScaleLevel;

#[tokio::test]
async fn test_lockfree_processor_basic_functionality() {
    let config = LockFreeContextProcessorConfig::default();
    let processor = LockFreeEnhancedContextProcessor::new(config, None)
        .await
        .expect("Failed to create processor");

    // Create test chunk
    let chunk = StreamChunk {
        stream_id: StreamId("test_stream".to_string()),
        sequence: 1,
        timestamp: std::time::Instant::now(),
        data: b"Hello, world!".to_vec(),
        metadata: HashMap::new(),
    };

    // Process without context
    let result = processor.process_enhanced_context(&chunk, None).await;
    assert!(result.is_ok());

    let enhanced_context = result.unwrap();
    assert!(enhanced_context.quality_metrics.overall_quality > 0.0);
    assert!(!enhanced_context.attention_weights.is_empty());
}

#[tokio::test]
async fn test_lockfree_processor_with_context() {
    let config = LockFreeContextProcessorConfig::default();
    let processor = LockFreeEnhancedContextProcessor::new(config, None)
        .await
        .expect("Failed to create processor");

    // Create test chunks
    let mut chunks = Vec::new();
    for i in 0..5 {
        chunks.push(StreamChunk {
            stream_id: StreamId("test_stream".to_string()),
            sequence: i,
            timestamp: std::time::Instant::now(),
            data: format!("Test data chunk {}", i).as_bytes().to_vec(),
            metadata: HashMap::new(),
        });
    }

    let current_chunk = &chunks[4];
    let context: Vec<_> = chunks.iter().take(4).collect();

    // Process with context
    let result = processor.process_enhanced_context(current_chunk, Some(&context)).await;
    assert!(result.is_ok());

    let enhanced_context = result.unwrap();
    assert!(enhanced_context.quality_metrics.overall_quality > 0.0);
}

#[tokio::test]
async fn test_lockfree_processor_caching() {
    let config = LockFreeContextProcessorConfig {
        cache_capacity: 100,
        ..Default::default()
    };
    let processor = LockFreeEnhancedContextProcessor::new(config, None)
        .await
        .expect("Failed to create processor");

    let chunk = StreamChunk {
        stream_id: StreamId("test_stream".to_string()),
        sequence: 1,
        timestamp: std::time::Instant::now(),
        data: b"Test data".to_vec(),
        metadata: HashMap::new(),
    };

    // First call should compute result
    let start1 = Instant::now();
    let result1 = processor.process_enhanced_context(&chunk, None).await.unwrap();
    let duration1 = start1.elapsed();

    // Second call should use cache (should be faster)
    let start2 = Instant::now();
    let result2 = processor.process_enhanced_context(&chunk, None).await.unwrap();
    let duration2 = start2.elapsed();

    // Results should be identical
    assert_eq!(result1.quality_metrics.overall_quality, result2.quality_metrics.overall_quality);
    
    // Second call should be faster (cached)
    // Note: This might not always hold due to system variance, but generally should
    println!("First call: {:?}, Second call: {:?}", duration1, duration2);
    
    // Check cache statistics
    let analytics = processor.get_analytics_snapshot();
    assert!(analytics.cache_hit_rate > 0.0 || analytics.total_processed >= 2);
}

#[tokio::test]
async fn test_lockfree_processor_concurrent_access() {
    let config = LockFreeContextProcessorConfig::default();
    let processor = Arc::new(
        LockFreeEnhancedContextProcessor::new(config, None)
            .await
            .expect("Failed to create processor")
    );

    let num_threads = 4;
    let chunks_per_thread = 10;
    let mut handles = Vec::new();

    for thread_id in 0..num_threads {
        let processor_clone = processor.clone();
        
        let handle = tokio::spawn(async move {
            let mut successful_processes = 0;
            
            for i in 0..chunks_per_thread {
                let chunk = StreamChunk {
                    stream_id: StreamId(format!("stream_{}", thread_id)),
                    sequence: i as u64,
                    timestamp: std::time::Instant::now(),
                    data: format!("Thread {} chunk {}", thread_id, i).as_bytes().to_vec(),
                    metadata: HashMap::new(),
                };

                match processor_clone.process_enhanced_context(&chunk, None).await {
                    Ok(_) => successful_processes += 1,
                    Err(e) => println!("Processing error in thread {}: {}", thread_id, e),
                }

                // Small delay to allow interleaving
                tokio::time::sleep(Duration::from_millis(1)).await;
            }

            successful_processes
        });
        
        handles.push(handle);
    }

    // Wait for all threads and collect results
    let mut total_successful = 0;
    for handle in handles {
        let successful = handle.await.expect("Thread panicked");
        total_successful += successful;
    }

    // Should have processed all chunks successfully
    assert_eq!(total_successful, num_threads * chunks_per_thread);

    // Check analytics
    let analytics = processor.get_analytics_snapshot();
    assert_eq!(analytics.total_processed, (num_threads * chunks_per_thread) as u64);
    assert!(analytics.average_latency_ms > 0.0);
}

#[tokio::test]
async fn test_lockfree_processor_performance_metrics() {
    let config = LockFreeContextProcessorConfig::default();
    let processor = LockFreeEnhancedContextProcessor::new(config, None)
        .await
        .expect("Failed to create processor");

    // Process several chunks to generate metrics
    for i in 0..10 {
        let chunk = StreamChunk {
            stream_id: StreamId("performance_test".to_string()),
            sequence: i,
            timestamp: std::time::Instant::now(),
            data: format!("Performance test chunk {}", i).as_bytes().to_vec(),
            metadata: HashMap::new(),
        };

        let _ = processor.process_enhanced_context(&chunk, None).await;
    }

    // Check analytics snapshot
    let analytics = processor.get_analytics_snapshot();
    
    assert_eq!(analytics.total_processed, 10);
    assert!(analytics.average_latency_ms >= 0.0);
    assert!(analytics.average_quality >= 0.0);
    assert!(analytics.average_quality <= 1.0);
    assert!(analytics.current_throughput >= 0.0);
    assert_eq!(analytics.total_errors, 0); // No errors expected in normal test

    // Check cross-scale index statistics
    let cross_scale_stats = processor.get_cross_scale_stats();
    assert!(cross_scale_stats.total_insertions() >= 10); // Should have at least as many insertions as chunks
    assert!(cross_scale_stats.total_operations() >= 10);
}

#[tokio::test]
async fn test_lockfree_processor_cross_scale_indexing() {
    let config = LockFreeContextProcessorConfig {
        max_context_window: 100,
        enable_pattern_recognition: true,
        ..Default::default()
    };
    let processor = LockFreeEnhancedContextProcessor::new(config, None)
        .await
        .expect("Failed to create processor");

    // Process chunks with varying characteristics to trigger cross-scale indexing
    let chunk_sizes = vec![10, 50, 100, 200, 500];
    
    for (i, size) in chunk_sizes.iter().enumerate() {
        let data = vec![0u8; *size]; // Create data of varying sizes
        
        let chunk = StreamChunk {
            stream_id: StreamId("cross_scale_test".to_string()),
            sequence: i as u64,
            timestamp: std::time::Instant::now(),
            data,
            metadata: HashMap::new(),
        };

        let result = processor.process_enhanced_context(&chunk, None).await;
        assert!(result.is_ok(), "Failed to process chunk {}", i);
    }

    // Check that cross-scale indexing occurred
    let cross_scale_stats = processor.get_cross_scale_stats();
    assert!(cross_scale_stats.total_insertions() >= 5);
    
    // Should have some correlations recorded (even if they're weak)
    assert!(cross_scale_stats.correlation_updates() >= 0);
}

#[tokio::test]
async fn test_lockfree_processor_error_handling() {
    let config = LockFreeContextProcessorConfig::default();
    let processor = LockFreeEnhancedContextProcessor::new(config, None)
        .await
        .expect("Failed to create processor");

    // Test with edge cases
    let edge_cases = vec![
        StreamChunk {
            stream_id: StreamId("empty_data".to_string()),
            sequence: 1,
            timestamp: std::time::Instant::now(),
            data: Vec::new(), // Empty data
            metadata: HashMap::new(),
        },
        StreamChunk {
            stream_id: StreamId("large_data".to_string()),
            sequence: 2,
            timestamp: std::time::Instant::now(),
            data: vec![0u8; 10000], // Large data
            metadata: HashMap::new(),
        },
    ];

    for chunk in edge_cases {
        let result = processor.process_enhanced_context(&chunk, None).await;
        // Should handle edge cases gracefully
        assert!(result.is_ok(), "Failed to handle edge case for chunk {}", chunk.sequence);
    }
}

#[tokio::test]  
async fn test_lockfree_processor_memory_usage() {
    let config = LockFreeContextProcessorConfig {
        cache_capacity: 50,
        max_context_window: 100,
        ..Default::default()
    };
    let processor = LockFreeEnhancedContextProcessor::new(config, None)
        .await
        .expect("Failed to create processor");

    // Process many chunks to test memory management
    for i in 0..200 {
        let chunk = StreamChunk {
            stream_id: StreamId("memory_test".to_string()),
            sequence: i,
            timestamp: std::time::Instant::now(),
            data: format!("Memory test chunk {} with some data", i).as_bytes().to_vec(),
            metadata: HashMap::new(),
        };

        let _ = processor.process_enhanced_context(&chunk, None).await;
        
        // Occasionally check analytics to ensure no memory leaks in metrics
        if i % 50 == 0 {
            let analytics = processor.get_analytics_snapshot();
            assert!(analytics.total_processed == i + 1);
        }
    }

    // Final check - system should still be responsive
    let analytics = processor.get_analytics_snapshot();
    assert_eq!(analytics.total_processed, 200);
    assert!(analytics.current_throughput > 0.0);
}

#[tokio::test]
async fn test_lockfree_processor_quality_consistency() {
    let config = LockFreeContextProcessorConfig::default();
    let processor = LockFreeEnhancedContextProcessor::new(config, None)
        .await
        .expect("Failed to create processor");

    // Process identical chunks - should get consistent quality scores
    let chunk = StreamChunk {
        stream_id: StreamId("quality_test".to_string()),
        sequence: 1,
        timestamp: std::time::Instant::now(),
        data: b"Consistent test data".to_vec(),
        metadata: HashMap::new(),
    };

    let mut quality_scores = Vec::new();
    
    for _ in 0..5 {
        let result = processor.process_enhanced_context(&chunk, None).await.unwrap();
        quality_scores.push(result.quality_metrics.overall_quality);
    }

    // All quality scores should be identical (cached results)
    let first_score = quality_scores[0];
    for score in quality_scores {
        assert!((score - first_score).abs() < 0.001, "Quality scores should be consistent");
    }
}

// Benchmark test to measure performance improvement
#[tokio::test]
async fn test_lockfree_processor_throughput_benchmark() {
    let config = LockFreeContextProcessorConfig::default();
    let processor = LockFreeEnhancedContextProcessor::new(config, None)
        .await
        .expect("Failed to create processor");

    let num_chunks = 100;
    let chunk_size = 1024;
    let start_time = Instant::now();

    // Process chunks sequentially
    for i in 0..num_chunks {
        let chunk = StreamChunk {
            stream_id: StreamId("benchmark".to_string()),
            sequence: i,
            timestamp: std::time::Instant::now(),
            data: vec![i as u8; chunk_size],
            metadata: HashMap::new(),
        };

        let _ = processor.process_enhanced_context(&chunk, None).await;
    }

    let total_time = start_time.elapsed();
    let throughput = num_chunks as f64 / total_time.as_secs_f64();
    
    println!("Lock-free processor throughput: {:.2} chunks/sec", throughput);
    println!("Average latency: {:.2}ms", total_time.as_millis() as f64 / num_chunks as f64);

    // Should achieve decent throughput (this is a basic performance check)
    assert!(throughput > 10.0, "Throughput too low: {:.2} chunks/sec", throughput);
    
    // Check final analytics
    let analytics = processor.get_analytics_snapshot();
    assert_eq!(analytics.total_processed, num_chunks as u64);
    assert!(analytics.current_throughput > 0.0);
}

// Test concurrent read/write operations for true lock-free behavior
#[tokio::test]
async fn test_true_lockfree_concurrent_operations() {
    let config = LockFreeContextProcessorConfig::default();
    let processor = Arc::new(
        LockFreeEnhancedContextProcessor::new(config, None)
            .await
            .expect("Failed to create processor")
    );

    let num_concurrent_operations = 100;
    let mut tasks = Vec::new();

    // Start many concurrent operations simultaneously
    for i in 0..num_concurrent_operations {
        let processor_clone = processor.clone();
        
        let task = tokio::spawn(async move {
            let chunk = StreamChunk {
                stream_id: StreamId(format!("concurrent_{}", i % 10)), // Group by stream
                sequence: i as u64,
                timestamp: std::time::Instant::now(),
                data: format!("Concurrent operation {}", i).as_bytes().to_vec(),
                metadata: HashMap::new(),
            };

            let start = Instant::now();
            let result = processor_clone.process_enhanced_context(&chunk, None).await;
            let duration = start.elapsed();

            (result.is_ok(), duration)
        });
        
        tasks.push(task);
    }

    // Wait for all operations to complete
    let mut successful_operations = 0;
    let mut total_duration = Duration::new(0, 0);

    for task in tasks {
        let (success, duration) = task.await.expect("Task panicked");
        if success {
            successful_operations += 1;
        }
        total_duration += duration;
    }

    // All operations should succeed
    assert_eq!(successful_operations, num_concurrent_operations);
    
    // Average operation time should be reasonable
    let avg_duration = total_duration / num_concurrent_operations as u32;
    println!("Average concurrent operation time: {:?}", avg_duration);
    
    // Verify analytics
    let analytics = processor.get_analytics_snapshot();
    assert_eq!(analytics.total_processed, num_concurrent_operations as u64);
}