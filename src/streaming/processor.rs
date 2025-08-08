use std::sync::Arc;

use anyhow::Result;
use tokio::sync::mpsc;
use tracing::{debug, trace};

/// Processor configuration
#[derive(Debug, Clone)]
pub struct ProcessorConfig {
    pub batch_size: usize,
    pub batch_timeout_ms: u64,
    pub num_workers: usize,
}

/// Stream processor for batch processing
pub struct StreamProcessor {
    config: ProcessorConfig,
    workers: Vec<tokio::task::JoinHandle<()>>,
}

impl StreamProcessor {
    /// Create a new stream processor
    pub fn new(config: ProcessorConfig) -> Self {
        Self { config, workers: Vec::new() }
    }

    /// Start processing
    pub fn start(&mut self) -> mpsc::Sender<ProcessorInput> {
        let (tx, rx) = mpsc::channel(1000);
        let rx = Arc::new(tokio::sync::Mutex::new(rx));

        // Start worker threads
        for worker_id in 0..self.config.num_workers {
            let rx = rx.clone();
            let config = self.config.clone();

            let handle = tokio::spawn(async move {
                worker_loop(worker_id, rx, config).await;
            });

            self.workers.push(handle);
        }

        tx
    }

    /// Stop processing
    pub async fn stop(&mut self) {
        // Workers will stop when channel is dropped
        for handle in self.workers.drain(..) {
            let _ = handle.await;
        }
    }
}

/// Input to the processor
#[derive(Debug)]
pub struct ProcessorInput {
    pub id: String,
    pub data: Vec<u8>,
    pub callback: mpsc::Sender<ProcessorOutput>,
}

/// Output from the processor
#[derive(Debug)]
pub struct ProcessorOutput {
    pub id: String,
    pub result: Result<Vec<u8>>,
}

/// Worker loop for processing
async fn worker_loop(
    worker_id: usize,
    rx: Arc<tokio::sync::Mutex<mpsc::Receiver<ProcessorInput>>>,
    config: ProcessorConfig,
) {
    debug!("Processor worker {} started", worker_id);

    let mut batch = Vec::with_capacity(config.batch_size);
    let mut last_batch_time = tokio::time::Instant::now();

    loop {
        // Try to fill batch
        let timeout = tokio::time::Duration::from_millis(config.batch_timeout_ms);
        let deadline = last_batch_time + timeout;

        loop {
            // Create the receive future outside select!
            let mut rx_guard = rx.lock().await;

            tokio::select! {
                _ = tokio::time::sleep_until(deadline) => {
                    drop(rx_guard); // Release lock before processing
                    // Timeout reached, process what we have
                    if !batch.is_empty() {
                        process_batch(&mut batch).await;
                        last_batch_time = tokio::time::Instant::now();
                    }
                    break;
                }
                result = rx_guard.recv() => {
                    drop(rx_guard); // Release lock after receiving
                    match result {
                        Some(input) => {
                            batch.push(input);
                            if batch.len() >= config.batch_size {
                                process_batch(&mut batch).await;
                                last_batch_time = tokio::time::Instant::now();
                                break;
                            }
                        }
                        None => {
                            // Channel closed
                            if !batch.is_empty() {
                                process_batch(&mut batch).await;
                            }
                            debug!("Processor worker {} stopped", worker_id);
                            return;
                        }
                    }
                }
            }
        }
    }
}

/// Process a batch of inputs with cognitive streaming
async fn process_batch(batch: &mut Vec<ProcessorInput>) {
    trace!("Processing cognitive batch of {} items", batch.len());

    // Group similar items for better processing efficiency while preserving
    // callbacks
    let mut cognitive_batches: std::collections::HashMap<String, Vec<ProcessorInput>> =
        std::collections::HashMap::new();

    // Categorize inputs by content type
    for input in batch.drain(..) {
        let category = classify_input(&input.data);
        cognitive_batches.entry(category).or_insert_with(Vec::new).push(input);
    }

    // Process each category separately for optimal cognitive patterns
    for (category, inputs) in cognitive_batches {
        trace!("Processing {} items in category: {}", inputs.len(), category);

        // Enhanced request-response mapping: maintain input-callback pairs
        let processing_tasks: Vec<_> = inputs
            .into_iter()
            .map(|input| {
                let category_clone = category.clone();
                let input_id = input.id.clone();
                let callback = input.callback.clone();

                async move {
                    let processing_result = process_single(input.data, &category_clone).await;

                    let output =
                        ProcessorOutput { id: input_id.clone(), result: processing_result };

                    // Send response back through the original callback channel
                    if let Err(e) = callback.send(output).await {
                        tracing::warn!("Failed to send response for request {}: {}", input_id, e);
                    } else {
                        debug!("Successfully sent response for request: {}", input_id);
                    }

                    input_id // Return the ID for logging
                }
            })
            .collect();

        // Execute all processing tasks concurrently while maintaining request-response
        // mapping
        let completed_ids = futures::future::join_all(processing_tasks).await;

        debug!(
            "Completed processing for {} requests in category '{}': {:?}",
            completed_ids.len(),
            category,
            completed_ids
        );
    }
}

/// Classify input data for cognitive processing
#[inline(always)] // Backend optimization: aggressive inlining for classification hot path
fn classify_input(data: &[u8]) -> String {
    // Backend optimization: use fast branch prediction for classification patterns
    crate::compiler_backend_optimization::register_optimization::low_register_pressure(|| {
        // Use bit operations for fast size comparisons
        let len = data.len();
        if len < 100 {
            "short_text".to_string()
        } else if len < 1000 {
            "medium_content".to_string()
        } else if data.starts_with(b"{") || data.starts_with(b"[") {
            "json_data".to_string()
        } else if data.iter().all(|&b| b.is_ascii()) {
            "text_content".to_string()
        } else {
            "binary_data".to_string()
        }
    })
}

/// Process a single item with cognitive enhancement
async fn process_single(data: Vec<u8>, category: &str) -> Result<Vec<u8>> {
    match category {
        "short_text" => {
            // For short text, apply basic text processing
            let text = String::from_utf8_lossy(&data);
            let processed = text
                .lines()
                .map(|line| line.trim())
                .filter(|line| !line.is_empty())
                .collect::<Vec<_>>()
                .join("\n");
            Ok(processed.into_bytes())
        }

        "json_data" => {
            // For JSON data, parse and reformat for better structure
            if let Ok(text) = std::str::from_utf8(&data) {
                if let Ok(value) = serde_json::from_str::<serde_json::Value>(text) {
                    let formatted = serde_json::to_string_pretty(&value)?;
                    return Ok(formatted.into_bytes());
                }
            }
            // Fallback to original data if parsing fails
            Ok(data)
        }

        "text_content" => {
            // For text content, apply advanced text processing
            let text = String::from_utf8_lossy(&data);
            let processed = process_text_content(&text);
            Ok(processed.into_bytes())
        }

        "binary_data" => {
            // For binary data, apply compression or encoding
            use std::io::Write;

            use flate2::Compression;
            use flate2::write::GzEncoder;

            let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
            encoder.write_all(&data)?;
            let compressed = encoder.finish()?;

            // Return compressed data if it's smaller, otherwise original
            if compressed.len() < data.len() { Ok(compressed) } else { Ok(data) }
        }

        _ => {
            // Backend optimization: SIMD-optimized byte processing
            let mut result = data;
            
            crate::compiler_backend_optimization::codegen_optimization::loop_optimization::hint_loop_bounds(result.len() / 8, |_| {
                // Backend optimization: vectorized chunk processing is handled by outer scope
            });
            
            // Apply backend optimized byte transformations
            for chunk in result.chunks_mut(8) {
                // Use fast bit operations for transformation
                for byte in chunk {
                    *byte = byte.wrapping_add(1); // Simple transformation with compiler hints
                }
            }
            
            Ok(result)
        }
    }
}

/// Process text content with cognitive patterns
fn process_text_content(text: &str) -> String {
    // Apply text processing patterns aligned with cognitive architecture
    text.lines()
        .map(|line| {
            let line = line.trim();
            if line.is_empty() {
                return String::new();
            }

            // Apply cognitive processing patterns
            if line.len() > 200 {
                // Long lines: break into cognitive chunks
                line.chars()
                    .collect::<Vec<_>>()
                    .chunks(100)
                    .map(|chunk| chunk.iter().collect::<String>())
                    .collect::<Vec<_>>()
                    .join("\n  ")
            } else {
                // Short lines: enhance for readability
                line.to_string()
            }
        })
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}
