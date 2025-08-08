use std::sync::Arc;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use loki::cognitive::{CognitiveConfig, CognitiveSystem};
use loki::compute::ComputeManager;
use loki::memory::{CognitiveMemory, MemoryConfig, MemoryMetadata};
use loki::streaming::StreamManager;
use loki::tools::{TaskConfig, TaskManager, TaskPlatform, TaskPriority};
use tempfile::TempDir;
use tokio::runtime::Runtime;

/// Benchmark memory operations
fn memory_benchmarks(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let temp_dir = TempDir::new().unwrap();
    let memory_config = MemoryConfig {
        persistence_path: temp_dir.path().join("memory"),
        max_memory_mb: 512,
        context_window: 4096,
        enable_persistence: true,
        embedding_dimension: 384,
        cache_size_mb: 64,
        max_age_days: 30,
    };

    let memory =
        rt.block_on(async { Arc::new(CognitiveMemory::new(memory_config).await.unwrap()) });

    let mut group = c.benchmark_group("memory");

    // Benchmark memory storage
    group.bench_function("storage", |b| {
        let memory = memory.clone();
        let rt = Runtime::new().unwrap();
        let mut counter = 0;

        b.iter(|| {
            rt.block_on(async {
                let _ = memory
                    .store(
                        black_box(format!("Benchmark entry {}", counter)),
                        black_box(vec![format!("Test data {}", counter)]),
                        black_box(MemoryMetadata {
                            source: "benchmark".to_string(),
                            tags: vec!["performance".to_string()],
                            importance: 0.5,
                            associations: vec![],
                        }),
                    )
                    .await;
                counter += 1;
            });
        });
    });

    // Benchmark memory retrieval
    group.bench_function("retrieval", |b| {
        let memory = memory.clone();
        let rt = Runtime::new().unwrap();

        // Pre-populate with some data
        rt.block_on(async {
            for i in 0..100 {
                let _ = memory
                    .store(
                        format!("Retrieval test entry {}", i),
                        vec![format!("Test data {}", i)],
                        MemoryMetadata {
                            source: "benchmark".to_string(),
                            tags: vec!["retrieval_test".to_string()],
                            importance: 0.5,
                            associations: vec![],
                        },
                    )
                    .await;
            }
        });

        let mut counter = 0;
        b.iter(|| {
            rt.block_on(async {
                let _ = memory
                    .retrieve_similar(
                        black_box(&format!("Retrieval test entry {}", counter % 100)),
                        black_box(5),
                    )
                    .await;
                counter += 1;
            });
        });
    });

    group.finish();
}

/// Benchmark cognitive processing
fn cognitive_benchmarks(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let temp_dir = TempDir::new().unwrap();
    let cognitive_config = CognitiveConfig {
        memory_config: MemoryConfig {
            persistence_path: temp_dir.path().join("cognitive"),
            max_memory_mb: 512,
            context_window: 4096,
            enable_persistence: true,
            embedding_dimension: 384,
            cache_size_mb: 64,
            max_age_days: 30,
        },
        orchestrator_model: "llama3.2:3b".to_string(),
        context_window: 4096,
        stream_batch_size: 16,
        background_tasks_enabled: false,
        monitoring_interval: Duration::from_secs(30),
        max_agents: 2,
    };

    let cognitive_system = rt.block_on(async {
        let compute_manager = Arc::new(ComputeManager::new().unwrap());
        let stream_manager = Arc::new(StreamManager::new().unwrap());
        Arc::new(
            CognitiveSystem::new(compute_manager, stream_manager, cognitive_config).await.unwrap(),
        )
    });

    let mut group = c.benchmark_group("cognitive");

    // Benchmark query processing
    group.bench_function("query_processing", |b| {
        let cognitive_system = cognitive_system.clone();
        let rt = Runtime::new().unwrap();
        let mut counter = 0;

        b.iter(|| {
            rt.block_on(async {
                let query = format!("What is the status of benchmark test {}?", counter);
                let _ = cognitive_system.process_query(black_box(&query)).await;
                counter += 1;
            });
        });
    });

    group.finish();
}

/// Benchmark tool operations
fn tool_benchmarks(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let temp_dir = TempDir::new().unwrap();
    let cognitive_config = CognitiveConfig {
        memory_config: MemoryConfig {
            persistence_path: temp_dir.path().join("tools"),
            max_memory_mb: 256,
            context_window: 2048,
            enable_persistence: true,
            embedding_dimension: 384,
            cache_size_mb: 32,
            max_age_days: 30,
        },
        orchestrator_model: "llama3.2:3b".to_string(),
        context_window: 2048,
        stream_batch_size: 8,
        background_tasks_enabled: false,
        monitoring_interval: Duration::from_secs(60),
        max_agents: 1,
    };

    let (task_manager, cognitive_system) = rt.block_on(async {
        let memory_config = MemoryConfig {
            persistence_path: temp_dir.path().join("memory"),
            max_memory_mb: 256,
            context_window: 2048,
            enable_persistence: true,
            embedding_dimension: 384,
            cache_size_mb: 32,
            max_age_days: 30,
        };

        let memory = Arc::new(CognitiveMemory::new(memory_config).await.unwrap());
        let compute_manager = Arc::new(ComputeManager::new().unwrap());
        let stream_manager = Arc::new(StreamManager::new().unwrap());
        let cognitive_system = Arc::new(
            CognitiveSystem::new(compute_manager, stream_manager, cognitive_config).await.unwrap(),
        );

        let task_config = TaskConfig {
            auto_create_from_conversations: true,
            cognitive_awareness_level: 0.7,
            ..Default::default()
        };

        let task_manager =
            TaskManager::new(task_config, cognitive_system.clone(), memory, None).await.unwrap();

        (task_manager, cognitive_system)
    });

    let mut group = c.benchmark_group("tools");

    // Benchmark task creation
    group.bench_function("task_creation", |b| {
        let task_manager = task_manager.clone();
        let rt = Runtime::new().unwrap();
        let mut counter = 0;

        b.iter(|| {
            rt.block_on(async {
                let _ = task_manager
                    .create_task(
                        black_box(&format!("Benchmark task {}", counter)),
                        black_box(Some("Performance test task")),
                        black_box(TaskPriority::Medium),
                        black_box(None),
                        black_box(TaskPlatform::Internal),
                    )
                    .await;
                counter += 1;
            });
        });
    });

    // Benchmark task extraction from text
    group.bench_function("task_extraction", |b| {
        let task_manager = task_manager.clone();
        let rt = Runtime::new().unwrap();
        let mut counter = 0;

        b.iter(|| {
            rt.block_on(async {
                let text = format!(
                    "We need to complete task {} and also should finish the benchmark test by \
                     tomorrow. Must remember to update the documentation.",
                    counter
                );
                let _ = task_manager.extract_tasks_from_conversation(black_box(&text)).await;
                counter += 1;
            });
        });
    });

    group.finish();
}

/// Benchmark system scalability
fn scalability_benchmarks(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let temp_dir = TempDir::new().unwrap();
    let memory_config = MemoryConfig {
        persistence_path: temp_dir.path().join("scalability"),
        max_memory_mb: 1024,
        context_window: 8192,
        enable_persistence: true,
        embedding_dimension: 384,
        cache_size_mb: 128,
        max_age_days: 30,
    };

    let memory =
        rt.block_on(async { Arc::new(CognitiveMemory::new(memory_config).await.unwrap()) });

    let mut group = c.benchmark_group("scalability");

    // Benchmark concurrent memory operations
    for concurrency in [1, 5, 10, 20].iter() {
        group.bench_with_input(
            BenchmarkId::new("concurrent_memory_ops", concurrency),
            concurrency,
            |b, &concurrency| {
                let memory = memory.clone();
                let rt = Runtime::new().unwrap();

                b.iter(|| {
                    rt.block_on(async {
                        let handles: Vec<_> = (0..concurrency)
                            .map(|i| {
                                let memory = memory.clone();
                                tokio::spawn(async move {
                                    memory
                                        .store(
                                            format!("Concurrent entry {}", i),
                                            vec![format!("Data {}", i)],
                                            MemoryMetadata {
                                                source: "concurrency_test".to_string(),
                                                tags: vec!["concurrent".to_string()],
                                                importance: 0.5,
                                                associations: vec![],
                                            },
                                        )
                                        .await
                                })
                            })
                            .collect();

                        for handle in handles {
                            let _ = handle.await;
                        }
                    });
                });
            },
        );
    }

    group.finish();
}

/// Benchmark memory usage patterns
fn memory_usage_benchmarks(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("memory_usage");

    // Benchmark different memory sizes
    for memory_size in [128, 256, 512, 1024].iter() {
        group.bench_with_input(
            BenchmarkId::new("memory_initialization", memory_size),
            memory_size,
            |b, &memory_size| {
                b.iter(|| {
                    rt.block_on(async {
                        let temp_dir = TempDir::new().unwrap();
                        let memory_config = MemoryConfig {
                            persistence_path: temp_dir.path().join("memory_test"),
                            max_memory_mb: black_box(memory_size),
                            context_window: 4096,
                            enable_persistence: true,
                            embedding_dimension: 384,
                            cache_size_mb: memory_size / 4,
                            max_age_days: 30,
                        };

                        let _ = CognitiveMemory::new(memory_config).await;
                    });
                });
            },
        );
    }

    group.finish();
}

/// Benchmark different context window sizes
fn context_window_benchmarks(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("context_window");

    // Benchmark different context sizes
    for context_size in [1024, 2048, 4096, 8192].iter() {
        group.bench_with_input(
            BenchmarkId::new("cognitive_initialization", context_size),
            context_size,
            |b, &context_size| {
                b.iter(|| {
                    rt.block_on(async {
                        let temp_dir = TempDir::new().unwrap();
                        let cognitive_config = CognitiveConfig {
                            memory_config: MemoryConfig {
                                persistence_path: temp_dir.path().join("context_test"),
                                max_memory_mb: 256,
                                context_window: black_box(context_size),
                                enable_persistence: true,
                                embedding_dimension: 384,
                                cache_size_mb: 32,
                                max_age_days: 30,
                            },
                            orchestrator_model: "llama3.2:3b".to_string(),
                            context_window: context_size,
                            stream_batch_size: 8,
                            background_tasks_enabled: false,
                            monitoring_interval: Duration::from_secs(60),
                            max_agents: 1,
                        };

                        let compute_manager = Arc::new(ComputeManager::new().unwrap());
                        let stream_manager = Arc::new(StreamManager::new().unwrap());
                        let _ =
                            CognitiveSystem::new(compute_manager, stream_manager, cognitive_config)
                                .await;
                    });
                });
            },
        );
    }

    group.finish();
}

/// Benchmark end-to-end workflows
fn workflow_benchmarks(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let temp_dir = TempDir::new().unwrap();
    let (cognitive_system, memory, task_manager) = rt.block_on(async {
        let memory_config = MemoryConfig {
            persistence_path: temp_dir.path().join("workflow"),
            max_memory_mb: 512,
            context_window: 4096,
            enable_persistence: true,
            embedding_dimension: 384,
            cache_size_mb: 64,
            max_age_days: 30,
        };

        let cognitive_config = CognitiveConfig {
            memory_config: memory_config.clone(),
            orchestrator_model: "llama3.2:3b".to_string(),
            context_window: 4096,
            stream_batch_size: 16,
            background_tasks_enabled: false,
            monitoring_interval: Duration::from_secs(30),
            max_agents: 2,
        };

        let memory = Arc::new(CognitiveMemory::new(memory_config).await.unwrap());
        let compute_manager = Arc::new(ComputeManager::new().unwrap());
        let stream_manager = Arc::new(StreamManager::new().unwrap());
        let cognitive_system = Arc::new(
            CognitiveSystem::new(compute_manager, stream_manager, cognitive_config).await.unwrap(),
        );

        let task_config = TaskConfig {
            auto_create_from_conversations: true,
            cognitive_awareness_level: 0.7,
            ..Default::default()
        };

        let task_manager =
            TaskManager::new(task_config, cognitive_system.clone(), memory.clone(), None)
                .await
                .unwrap();

        (cognitive_system, memory, task_manager)
    });

    let mut group = c.benchmark_group("workflow");

    // Benchmark complete task workflow
    group.bench_function("complete_task_workflow", |b| {
        let cognitive_system = cognitive_system.clone();
        let memory = memory.clone();
        let task_manager = task_manager.clone();
        let rt = Runtime::new().unwrap();
        let mut counter = 0;

        b.iter(|| {
            rt.block_on(async {
                // 1. Store context in memory
                let _ = memory
                    .store(
                        black_box(format!("Workflow context {}", counter)),
                        black_box(vec![format!("Context data {}", counter)]),
                        black_box(MemoryMetadata {
                            source: "workflow".to_string(),
                            tags: vec!["workflow".to_string()],
                            importance: 0.7,
                            associations: vec![],
                        }),
                    )
                    .await;

                // 2. Process with cognitive system
                let query = format!("Analyze workflow step {}", counter);
                let _ = cognitive_system.process_query(black_box(&query)).await;

                // 3. Create task
                let _ = task_manager
                    .create_task(
                        black_box(&format!("Workflow task {}", counter)),
                        black_box(Some("Generated from workflow")),
                        black_box(TaskPriority::Medium),
                        black_box(None),
                        black_box(TaskPlatform::Internal),
                    )
                    .await;

                // 4. Retrieve related memories
                let _ = memory
                    .retrieve_similar(
                        black_box(&format!("Workflow context {}", counter)),
                        black_box(3),
                    )
                    .await;

                counter += 1;
            });
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    memory_benchmarks,
    cognitive_benchmarks,
    tool_benchmarks,
    scalability_benchmarks,
    memory_usage_benchmarks,
    context_window_benchmarks,
    workflow_benchmarks
);

criterion_main!(benches);
