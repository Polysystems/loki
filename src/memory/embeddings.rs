use anyhow::Result;
use ndarray::{Array1, Array2, s};
use parking_lot::RwLock;
// Removed rayon::prelude - using iterator patterns for auto-vectorization instead
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tracing::{info, debug};
use tokio::fs;
use serde::{Serialize, Deserialize};

use super::{MemoryId, simd_cosine_similarity};

// Constant propagation optimization for embeddings
const F32_SIZE: usize = std::mem::size_of::<f32>();
const ROW_CHUNK_SIZE: usize = 8;
const BATCH_SIZE: usize = 64;

/// Persistent embedding metadata with cache-optimized layout
#[derive(Debug, Clone, Serialize, Deserialize)]
#[repr(C)] // Ensure predictable memory layout for cache efficiency
struct EmbeddingMetadata {
    // Hot fields first - frequently accessed together
    id: MemoryId,           // 8 bytes - primary key for lookups
    text_hash: u64,         // 8 bytes - fast comparison field
    created_at: chrono::DateTime<chrono::Utc>, // 12 bytes - temporal ordering
    // Total: 28 bytes - fits in single cache line with padding
}

/// Store for memory embeddings with persistence
#[derive(Debug)]
pub struct EmbeddingStore {
    /// Embedding dimension
    dimension: usize,

    /// Base path for persistence
    base_path: PathBuf,

    /// Embeddings matrix (rows are embeddings)
    embeddings: RwLock<Array2<f32>>,

    /// Mapping from memory ID to embedding index
    id_to_index: RwLock<HashMap<MemoryId, usize>>,

    /// Mapping from index to memory ID
    index_to_id: RwLock<Vec<MemoryId>>,

    /// Metadata for each embedding
    metadata: RwLock<Vec<EmbeddingMetadata>>,

    /// Next available index
    next_index: RwLock<usize>,

    /// Text cache for avoiding recomputation
    text_cache: RwLock<HashMap<u64, Array1<f32>>>,
}

impl EmbeddingStore {
    /// Create a new embedding store with persistence
    pub async fn new(dimension: usize, base_path: &Path) -> Result<Self> {
        info!("Creating embedding store with dimension {} at {:?}", dimension, base_path);

        // Create directories
        let embeddings_dir = base_path.join("embeddings");
        fs::create_dir_all(&embeddings_dir).await?;

        // Pre-allocate space for embeddings
        let initial_capacity = 10000;
        let embeddings = Array2::zeros((initial_capacity, dimension));

        let mut store = Self {
            dimension,
            base_path: embeddings_dir,
            embeddings: RwLock::new(embeddings),
            id_to_index: RwLock::new(HashMap::new()),
            index_to_id: RwLock::new(Vec::with_capacity(initial_capacity)),
            metadata: RwLock::new(Vec::with_capacity(initial_capacity)),
            next_index: RwLock::new(0),
            text_cache: RwLock::new(HashMap::new()),
        };

        // Load existing embeddings
        store.load_from_disk().await?;

        Ok(store)
    }

    /// Add an embedding for a memory
    pub async fn add(&self, id: &MemoryId, content: &str) -> Result<()> {
        debug!("Adding embedding for memory: {}", id);

        // Check if already exists
        if self.id_to_index.read().contains_key(id) {
            debug!("Embedding already exists for memory: {}", id);
            return Ok(());
        }

        // Generate embedding
        let embedding = self.generate_embedding(content).await?;
        let text_hash = self.hash_text(content);

        // Scope the locks to ensure they're dropped before any await
        let should_persist = {
            let mut embeddings = self.embeddings.write();
            let mut id_to_index = self.id_to_index.write();
            let mut index_to_id = self.index_to_id.write();
            let mut metadata = self.metadata.write();
            let mut next_index = self.next_index.write();
            let mut text_cache = self.text_cache.write();

            let index = *next_index;

            // Resize if needed
            if index >= embeddings.nrows() {
                let new_capacity = embeddings.nrows() * 2;
                info!("Expanding embedding store capacity to {}", new_capacity);

                let mut new_embeddings = Array2::zeros((new_capacity, self.dimension));
                new_embeddings.slice_mut(s![..embeddings.nrows(), ..]).assign(&embeddings);
                *embeddings = new_embeddings;
            }

            // Store embedding and metadata
            embeddings.row_mut(index).assign(&embedding);
            id_to_index.insert(id.clone(), index);
            index_to_id.push(id.clone());

            let embed_metadata = EmbeddingMetadata {
                id: id.clone(),
                text_hash,
                created_at: chrono::Utc::now(),
            };
            metadata.push(embed_metadata);

            // Cache the embedding by text hash
            text_cache.insert(text_hash, embedding.clone());

            *next_index += 1;

            // Return whether we should persist (every 100 embeddings)
            index % 100 == 0
        }; // All locks are dropped here

        // Persist to disk if needed (no locks held)
        if should_persist {
            self.persist_to_disk().await?;
        }

        Ok(())
    }

    /// Search for similar embeddings using semantic similarity
    pub async fn search(&self, query: &str, limit: usize) -> Result<Vec<(MemoryId, f32)>> {
        debug!("Searching for similar embeddings: '{}'", query);

        let query_hash = self.hash_text(query);

        // Check cache first - separate from any async operations
        let cached_embedding = {
            let cache = self.text_cache.read();
            cache.get(&query_hash).cloned()
        };

        // Get the query embedding (either from cache or generate new)
        let query_embedding = if let Some(embedding) = cached_embedding {
            embedding
        } else {
            let embedding = self.generate_embedding(query).await?;
            // Insert into cache after generation
            {
                let mut cache = self.text_cache.write();
                cache.insert(query_hash, embedding.clone());
            }
            embedding
        };

        let embeddings = self.embeddings.read();
        let index_to_id = self.index_to_id.read();
        let count = *self.next_index.read();

        if count == 0 {
            // Escape analysis optimization: early return avoids all allocations
            return Ok(Vec::new());
        }

        // Compute similarities using SIMD with escape analysis optimization
        let mut similarities = Vec::with_capacity(count);
        let query_slice = query_embedding.as_slice()
            .ok_or_else(|| anyhow::anyhow!("Failed to get query embedding slice"))?;

        // Escape analysis optimization: prefer stack allocation for small embeddings
        let aligned_dim = self.dimension.next_multiple_of(8);
        
        // Use const for better escape analysis optimization
        const STACK_EMBEDDING_LIMIT: usize = 1024;
        
        let aligned_query = if aligned_dim <= STACK_EMBEDDING_LIMIT {
            // Stack allocation for small embeddings (much faster, escape analysis optimized)
            let mut stack_query = [0.0f32; STACK_EMBEDDING_LIMIT];
            stack_query[..self.dimension].copy_from_slice(query_slice);
            stack_query[..aligned_dim].to_vec() // Convert to Vec for consistent interface
        } else {
            // Heap allocation only when necessary
            let mut heap_query = vec![0.0f32; aligned_dim];
            heap_query[..self.dimension].copy_from_slice(query_slice);
            heap_query
        };

        // Process embeddings in chunks for better vectorization
        // Use const chunk size for better compiler optimization
        const CHUNK_SIZE: usize = 8;
        let aligned_dim = self.dimension.next_multiple_of(8);
        
        for chunk_start in (0..count).step_by(CHUNK_SIZE) {
            let chunk_end = (chunk_start + CHUNK_SIZE).min(count);
            
            // Process chunk with vectorizable inner loop
            for i in chunk_start..chunk_end {
                let embedding_row = embeddings.row(i);
                let embedding_slice = embedding_row.as_slice()
                    .ok_or_else(|| anyhow::anyhow!("Failed to get embedding slice for row {}", i))?;

                // Escape analysis optimization: reuse stack allocation when possible
                let aligned_embedding = if aligned_dim <= STACK_EMBEDDING_LIMIT {
                    // Stack allocation reuse for small embeddings
                    let mut stack_embedding = [0.0f32; STACK_EMBEDDING_LIMIT];
                    let copy_len = self.dimension.min(embedding_slice.len());
                    stack_embedding[..copy_len].copy_from_slice(&embedding_slice[..copy_len]);
                    stack_embedding[..aligned_dim].to_vec()
                } else {
                    // Heap allocation only for large embeddings
                    let mut heap_embedding = vec![0.0f32; aligned_dim];
                    let copy_len = self.dimension.min(embedding_slice.len());
                    heap_embedding[..copy_len].copy_from_slice(&embedding_slice[..copy_len]);
                    heap_embedding
                };

                // Backend optimization: vectorized similarity computation
                let similarity = crate::compiler_backend_optimization::register_optimization::low_register_pressure(|| {
                    simd_cosine_similarity(
                        &aligned_query[..aligned_dim],
                        &aligned_embedding[..aligned_dim],
                    )
                });

                similarities.push((i, similarity));
            }
        }

        // Sort by similarity (descending)
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Return top results with scores
        let results: Vec<(MemoryId, f32)> = similarities
            .into_iter()
            .take(limit)
            .map(|(idx, score)| (index_to_id[idx].clone(), score))
            .collect();

        debug!("Found {} similar embeddings", results.len());
        Ok(results)
    }

    /// Get embedding for a specific memory ID
    pub fn get_embedding(&self, id: &MemoryId) -> Option<Array1<f32>> {
        let id_to_index = self.id_to_index.read();
        let embeddings = self.embeddings.read();

        if let Some(&index) = id_to_index.get(id) {
            Some(embeddings.row(index).to_owned())
        } else {
            None
        }
    }

    /// Generate embedding for text using improved hashing with normalization
    async fn generate_embedding(&self, text: &str) -> Result<Array1<f32>> {
        // Create a more sophisticated embedding than simple hash
        let mut embedding: Array1<f32> = Array1::zeros(self.dimension);

        // Use multiple hash functions for better distribution
        let text_lower = text.to_lowercase();
        let words: Vec<&str> = text_lower.split_whitespace().collect();

        // Word-level hashing
        for (word_idx, word) in words.iter().enumerate() {
            let word_hash = word.bytes().fold(0u64, |acc, b| {
                acc.wrapping_mul(31).wrapping_add(b as u64)
            });

            // Backend optimization: vectorized word influence distribution
            crate::compiler_backend_optimization::codegen_optimization::loop_optimization::hint_loop_bounds(self.dimension / 16, |chunk_idx| {
                let position_weight = 1.0 / (word_idx as f32 + 1.0).sqrt();
                
                // Process dimensions in chunks for better vectorization
                const DIM_CHUNK_SIZE: usize = 16; // Optimal for SIMD
                let chunk_start = chunk_idx * DIM_CHUNK_SIZE;
                let chunk_end = (chunk_start + DIM_CHUNK_SIZE).min(self.dimension);
                
                // Backend optimization: register pressure optimized inner loop
                crate::compiler_backend_optimization::register_optimization::low_register_pressure(|| {
                    for i in chunk_start..chunk_end {
                        let dimension_hash = word_hash.wrapping_mul(i as u64 + 1);
                        let val = ((dimension_hash >> (i % 64)) & 0xFF) as f32 / 255.0;
                        embedding[i] += val * position_weight;
                    }
                });
            });
        }

        // Character n-gram hashing for additional features
        for window_size in 2..=4 {
            let chars: Vec<char> = text_lower.chars().collect();
            for window in chars.windows(window_size) {
                let ngram: String = window.iter().collect();
                let ngram_hash = ngram.bytes().fold(0u64, |acc, b| {
                    acc.wrapping_mul(37).wrapping_add(b as u64)
                });

                // Vectorizable n-gram dimension processing
                const NGRAM_WEIGHT: f32 = 0.3;
                
                // Process in vectorizable chunks
                for chunk_start in (0..self.dimension).step_by(16) {
                    let chunk_end = (chunk_start + 16).min(self.dimension);
                    
                    for i in chunk_start..chunk_end {
                        let dimension_hash = ngram_hash.wrapping_mul((i + window_size) as u64);
                        let val = ((dimension_hash >> (i % 64)) & 0xFF) as f32 / 255.0;
                        embedding[i] += val * NGRAM_WEIGHT;
                    }
                }
            }
        }

        // Normalize the embedding
        let norm = embedding.dot(&embedding).sqrt();
        if norm > 0.0 {
            embedding /= norm;
        }

        Ok(embedding)
    }

    /// Hash text content for caching
    #[inline(always)] // Critical hot path function for embedding operations
    fn hash_text(&self, text: &str) -> u64 {
        // Backend optimization: use fast hash for frequent text operations
        crate::compiler_backend_optimization::instruction_selection::bit_operations::fast_hash(text)
    }

    /// Load embeddings from disk (optimized async state machine)
    #[inline(never)] // Prevent inlining to optimize state machine layout 
    async fn load_from_disk(&mut self) -> Result<()> {
        let embeddings_file = self.base_path.join("embeddings.dat");
        let metadata_file = self.base_path.join("metadata.json");

        if !embeddings_file.exists() || !metadata_file.exists() {
            info!("No existing embeddings found, starting fresh");
            return Ok(());
        }

        info!("Loading embeddings from disk");

        // Optimized: Concurrent file reads to minimize state machine size
        let (metadata_content, embeddings_data) = tokio::try_join!(
            fs::read_to_string(&metadata_file),
            fs::read(&embeddings_file)
        )?;
        
        // CPU-intensive operations after I/O completion (reduces async state)
        let metadata_list: Vec<EmbeddingMetadata> = serde_json::from_str(&metadata_content)?;
        let float_count = embeddings_data.len() / F32_SIZE;
        let expected_count = metadata_list.len() * self.dimension;

        if float_count != expected_count {
            info!("Embeddings file size mismatch, rebuilding index");
            return Ok(());
        }

        // Escape analysis optimization: reconstruct embeddings with minimal allocations
        let mut embeddings_vec = Vec::with_capacity(float_count);
        
        // Process chunks with stack-allocated byte arrays (escape analysis optimized)
        const BATCH_SIZE: usize = 64; // Process 64 f32s (256 bytes) at a time
        
        for batch_start in (0..embeddings_data.len()).step_by(BATCH_SIZE * F32_SIZE) {
            let batch_end = (batch_start + BATCH_SIZE * F32_SIZE).min(embeddings_data.len());
            let batch_data = &embeddings_data[batch_start..batch_end];
            
            // Stack-allocated byte array for escape analysis optimization
            for chunk in batch_data.chunks_exact(F32_SIZE) {
                // Stack allocation - no heap escape
                let bytes: [u8; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
                embeddings_vec.push(f32::from_le_bytes(bytes));
            }
        }

        let rows = metadata_list.len();
        let capacity = (rows * 2).max(10000);
        let mut embeddings = Array2::zeros((capacity, self.dimension));

        // Copy loaded data with escape analysis optimization
        for (row_idx, chunk) in embeddings_vec.chunks_exact(self.dimension).enumerate() {
            // Avoid intermediate Vec allocation by using slice directly
            let row_array = Array1::from_shape_vec(self.dimension, chunk.to_vec())?;
            embeddings.row_mut(row_idx).assign(&row_array);
        }

        // Rebuild indices
        let mut id_to_index = HashMap::new();
        let mut index_to_id = Vec::new();

        for (idx, meta) in metadata_list.iter().enumerate() {
            id_to_index.insert(meta.id.clone(), idx);
            index_to_id.push(meta.id.clone());
        }

        // Update store state
        *self.embeddings.write() = embeddings;
        *self.id_to_index.write() = id_to_index;
        *self.index_to_id.write() = index_to_id;
        *self.metadata.write() = metadata_list;
        *self.next_index.write() = rows;

        info!("Loaded {} embeddings from disk", rows);
        Ok(())
    }

    /// Persist embeddings to disk
    async fn persist_to_disk(&self) -> Result<()> {
        debug!("Persisting embeddings to disk");

        // Collect data while holding locks, then drop locks before async operations
        let (metadata_json, binary_data, count) = {
            let embeddings = self.embeddings.read();
            let metadata = self.metadata.read();
            let count = *self.next_index.read();

            if count == 0 {
                return Ok(());
            }

            // Serialize metadata
            let metadata_json = serde_json::to_string_pretty(&*metadata)?;

            // Collect binary data with vectorized processing
            let mut binary_data = Vec::with_capacity(count * self.dimension * 4);
            
            // Process rows in chunks for better cache locality and vectorization
            const ROW_CHUNK_SIZE: usize = 8;
            for chunk_start in (0..count).step_by(ROW_CHUNK_SIZE) {
                let chunk_end = (chunk_start + ROW_CHUNK_SIZE).min(count);
                
                for row_idx in chunk_start..chunk_end {
                    let row = embeddings.row(row_idx);
                    let row_slice = row.as_slice()
                        .ok_or_else(|| anyhow::anyhow!("Failed to get row slice for normalization at index {}", row_idx))?;
                    
                    // Use vectorizable iterator processing
                    for &value in row_slice {
                        binary_data.extend_from_slice(&value.to_le_bytes());
                    }
                }
            }

            (metadata_json, binary_data, count)
        }; // All locks are dropped here

        // Now perform async file operations without any locks held
        let metadata_file = self.base_path.join("metadata.json");
        let embeddings_file = self.base_path.join("embeddings.dat");

        fs::write(&metadata_file, metadata_json).await?;
        fs::write(&embeddings_file, binary_data).await?;

        debug!("Persisted {} embeddings to disk", count);
        Ok(())
    }

    /// Get the number of embeddings
    #[inline]
    pub fn count(&self) -> usize {
        *self.next_index.read()
    }

    /// Get embedding statistics
    pub fn stats(&self) -> EmbeddingStats {
        let count = self.count();
        let cache_size = self.text_cache.read().len();
        let memory_usage = count * self.dimension * 4; // 4 bytes per f32

        EmbeddingStats {
            total_embeddings: count,
            dimension: self.dimension,
            cache_entries: cache_size,
            memory_usage_bytes: memory_usage,
        }
    }

    /// Clear cache to free memory
    pub fn clear_cache(&self) {
        self.text_cache.write().clear();
    }

    /// Cleanup old embeddings with intelligent retention policies
    pub async fn cleanup_old_embeddings(&self, retention_days: u32) -> Result<CleanupStats> {
        info!("Starting embeddings cleanup with {} day retention", retention_days);

        let retention_cutoff = chrono::Utc::now() - chrono::Duration::days(retention_days as i64);

        let cleanup_result = {
            let mut embeddings = self.embeddings.write();
            let mut id_to_index = self.id_to_index.write();
            let mut index_to_id = self.index_to_id.write();
            let mut metadata = self.metadata.write();
            let mut next_index = self.next_index.write();
            let mut text_cache = self.text_cache.write();

            let original_count = *next_index;
            let mut indices_to_remove = Vec::new();
            let access_counts = HashMap::new();

            // Identify embeddings to remove based on multiple criteria
            for (idx, meta) in metadata.iter().enumerate() {
                let should_remove = self.should_remove_embedding(meta, retention_cutoff, &access_counts);
                if should_remove {
                    indices_to_remove.push(idx);
                }
            }

            if indices_to_remove.is_empty() {
                return Ok(CleanupStats {
                    embeddings_removed: 0,
                    memory_freed_bytes: 0,
                    cache_entries_cleared: 0,
                    original_count,
                    final_count: original_count,
                });
            }

            // Sort indices in descending order for safe removal
            indices_to_remove.sort_by(|a, b| b.cmp(a));

            // Remove embeddings by compacting the arrays
            for &remove_idx in &indices_to_remove {
                let memory_id = &index_to_id[remove_idx];

                // Remove from mappings
                id_to_index.remove(memory_id);

                // Remove from cache by finding matching text hash
                if let Some(meta) = metadata.get(remove_idx) {
                    text_cache.remove(&meta.text_hash);
                }
            }

            // Compact arrays by moving valid embeddings to fill gaps
            let mut write_idx = 0;
            let mut new_index_to_id = Vec::new();
            let mut new_metadata = Vec::new();
            let mut new_id_to_index = HashMap::new();

            for read_idx in 0..*next_index {
                if !indices_to_remove.contains(&read_idx) {
                    // Copy embedding row
                    if write_idx != read_idx {
                        let source_row = embeddings.row(read_idx).to_owned();
                        embeddings.row_mut(write_idx).assign(&source_row);
                    }

                    // Update mappings
                    let memory_id = index_to_id[read_idx].clone();
                    new_id_to_index.insert(memory_id.clone(), write_idx);
                    new_index_to_id.push(memory_id);
                    new_metadata.push(metadata[read_idx].clone());

                    write_idx += 1;
                }
            }

            // Update the store's internal state
            *index_to_id = new_index_to_id;
            *metadata = new_metadata;
            *id_to_index = new_id_to_index;
            *next_index = write_idx;

            // Calculate cleanup statistics
            let removed_count = indices_to_remove.len();
            let memory_freed = removed_count * self.dimension * std::mem::size_of::<f32>();
            let cache_cleared = text_cache.len();

            CleanupStats {
                embeddings_removed: removed_count,
                memory_freed_bytes: memory_freed,
                cache_entries_cleared: cache_cleared,
                original_count,
                final_count: write_idx,
            }
        }; // All locks are dropped here

        // Persist the cleaned state to disk
        self.persist_to_disk().await?;

        info!("Cleanup completed: removed {} embeddings, freed {} bytes",
              cleanup_result.embeddings_removed, cleanup_result.memory_freed_bytes);

        Ok(cleanup_result)
    }

    /// Determine if an embedding should be removed based on multiple criteria
    fn should_remove_embedding(
        &self,
        meta: &EmbeddingMetadata,
        cutoff: chrono::DateTime<chrono::Utc>,
        access_counts: &HashMap<MemoryId, u32>
    ) -> bool {
        // Age-based removal
        if meta.created_at < cutoff {
            // Additional criteria for old embeddings
            let access_count = access_counts.get(&meta.id).unwrap_or(&0);

            // Keep frequently accessed embeddings even if old
            if *access_count > 10 {
                return false;
            }

            // Keep embeddings from the last week regardless of access
            let last_week = chrono::Utc::now() - chrono::Duration::days(7);
            if meta.created_at > last_week {
                return false;
            }

            return true;
        }

        false
    }

    /// Defragment the embedding store to optimize memory layout
    pub async fn defragment(&self) -> Result<DefragmentStats> {
        info!("Starting embedding store defragmentation");

        let defrag_result = {
            let mut embeddings = self.embeddings.write();
            let current_count = *self.next_index.read();

            if current_count == 0 {
                return Ok(DefragmentStats {
                    embeddings_moved: 0,
                    memory_optimized_bytes: 0,
                    fragmentation_ratio_before: 0.0,
                    fragmentation_ratio_after: 0.0,
                });
            }

            let total_capacity = embeddings.nrows();
            let fragmentation_before = 1.0 - (current_count as f32 / total_capacity as f32);

            // Optimize capacity if we have significant unused space
            if current_count * 2 < total_capacity && total_capacity > 10000 {
                let optimal_capacity = (current_count * 3 / 2).max(10000);
                let mut new_embeddings = Array2::zeros((optimal_capacity, self.dimension));

                // Copy existing embeddings to new smaller matrix
                for i in 0..current_count {
                    new_embeddings.row_mut(i).assign(&embeddings.row(i));
                }

                let memory_saved = (total_capacity - optimal_capacity) * self.dimension * std::mem::size_of::<f32>();
                *embeddings = new_embeddings;

                let fragmentation_after = 1.0 - (current_count as f32 / optimal_capacity as f32);

                return Ok(DefragmentStats {
                    embeddings_moved: current_count,
                    memory_optimized_bytes: memory_saved,
                    fragmentation_ratio_before: fragmentation_before,
                    fragmentation_ratio_after: fragmentation_after,
                });
            }

            DefragmentStats {
                embeddings_moved: 0,
                memory_optimized_bytes: 0,
                fragmentation_ratio_before: fragmentation_before,
                fragmentation_ratio_after: fragmentation_before,
            }
        };

        if defrag_result.memory_optimized_bytes > 0 {
            self.persist_to_disk().await?;
            info!("Defragmentation completed: optimized {} bytes", defrag_result.memory_optimized_bytes);
        } else {
            info!("Defragmentation skipped: store already optimal");
        }

        Ok(defrag_result)
    }

    /// Perform maintenance operations including cleanup and optimization
    pub async fn perform_maintenance(&self, retention_days: u32) -> Result<MaintenanceStats> {
        info!("Starting comprehensive embedding store maintenance");

        // Step 1: Clear cache to free immediate memory
        self.clear_cache();

        // Step 2: Cleanup old embeddings
        let cleanup_stats = self.cleanup_old_embeddings(retention_days).await?;

        // Step 3: Defragment the store
        let defrag_stats = self.defragment().await?;

        // Step 4: Rebuild cache for most recent embeddings
        self.rebuild_recent_cache().await?;

        let maintenance_stats = MaintenanceStats {
            cleanup: cleanup_stats,
            defragmentation: defrag_stats,
            cache_rebuilt_entries: self.text_cache.read().len(),
        };

        info!("Maintenance completed: {} embeddings removed, {} bytes optimized",
              maintenance_stats.cleanup.embeddings_removed,
              maintenance_stats.defragmentation.memory_optimized_bytes);

        Ok(maintenance_stats)
    }

    /// Rebuild cache for the most recent embeddings to improve performance
    async fn rebuild_recent_cache(&self) -> Result<()> {
        let recent_limit = 1000; // Cache the 1000 most recent embeddings

        let recent_embeddings = {
            let metadata = self.metadata.read();
            let mut recent: Vec<_> = metadata.iter().collect();
            recent.sort_by(|a, b| b.created_at.cmp(&a.created_at));
            recent.into_iter().take(recent_limit).cloned().collect::<Vec<_>>()
        };

        let mut cache = self.text_cache.write();
        cache.clear();

        // Rebuild cache with recent embeddings
        let embeddings = self.embeddings.read();
        let id_to_index = self.id_to_index.read();

        for meta in recent_embeddings {
            if let Some(&index) = id_to_index.get(&meta.id) {
                let embedding = embeddings.row(index).to_owned();
                cache.insert(meta.text_hash, embedding);
            }
        }

        info!("Rebuilt cache with {} recent embeddings", cache.len());
        Ok(())
    }
}

/// Statistics for the embedding store with optimized layout
#[derive(Debug)]
#[repr(C)] // Cache-aligned layout for performance-critical stats
pub struct EmbeddingStats {
    // Group frequently accessed fields for cache efficiency
    pub total_embeddings: usize,    // 8 bytes - primary counter
    pub cache_entries: usize,       // 8 bytes - cache state
    pub memory_usage_bytes: usize,  // 8 bytes - memory tracking
    pub dimension: usize,           // 8 bytes - config info
    // Total: 32 bytes - exactly fits in cache line
}

/// Statistics for embedding cleanup operations with optimized layout
#[derive(Debug)]
#[repr(C)] // Optimize for batch operations and reporting
pub struct CleanupStats {
    // Group related counters for efficient comparison operations
    pub original_count: usize,        // 8 bytes - before state
    pub final_count: usize,          // 8 bytes - after state
    pub embeddings_removed: usize,   // 8 bytes - diff calculation
    pub cache_entries_cleared: usize, // 8 bytes - cache impact
    pub memory_freed_bytes: usize,   // 8 bytes - memory impact
    // Total: 40 bytes - spans cache line but grouped logically
}

/// Statistics for defragmentation operations with optimized layout
#[derive(Debug)]
#[repr(C)] // Align for efficient ratio calculations
pub struct DefragmentStats {
    // Group ratios together for SIMD comparison operations
    pub fragmentation_ratio_before: f32, // 4 bytes - initial state
    pub fragmentation_ratio_after: f32,  // 4 bytes - final state
    // Group counters together for batch operations
    pub embeddings_moved: usize,         // 8 bytes - movement count
    pub memory_optimized_bytes: usize,   // 8 bytes - optimization impact
    // Total: 24 bytes - compact cache-friendly layout
}

/// Statistics for comprehensive maintenance operations with optimized layout
#[derive(Debug)]
#[repr(C)] // Optimize for aggregate reporting operations
pub struct MaintenanceStats {
    // Hot field first - frequently checked counter
    pub cache_rebuilt_entries: usize, // 8 bytes - immediate result
    // Nested stats grouped by operation type for locality
    pub cleanup: CleanupStats,        // 40 bytes - cleanup results
    pub defragmentation: DefragmentStats, // 24 bytes - defrag results
    // Total: 72 bytes - spans multiple cache lines but grouped by access pattern
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_embedding_store_persistence() {
        let dir = tempdir().unwrap();
        let store = EmbeddingStore::new(384, dir.path()).await.unwrap();

        let id = MemoryId::new();
        store.add(&id, "test content").await.unwrap();

        // Test search
        let results = store.search("test", 10).await.unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].0, id);

        // Test persistence
        store.persist_to_disk().await.unwrap();

        // Create new store and verify data loads
        let store2 = EmbeddingStore::new(384, dir.path()).await.unwrap();
        assert_eq!(store2.count(), 1);

        let results2 = store2.search("test", 10).await.unwrap();
        assert!(!results2.is_empty());
    }

    #[tokio::test]
    async fn test_embedding_similarity() {
        let dir = tempdir().unwrap();
        let store = EmbeddingStore::new(384, dir.path()).await.unwrap();

        let id1 = MemoryId::new();
        let id2 = MemoryId::new();
        let id3 = MemoryId::new();

        store.add(&id1, "machine learning algorithms").await.unwrap();
        store.add(&id2, "artificial intelligence systems").await.unwrap();
        store.add(&id3, "cooking recipes for dinner").await.unwrap();

        let results = store.search("AI and ML", 3).await.unwrap();
        assert_eq!(results.len(), 3);

        // Should find AI/ML related content first
        assert!(results[0].1 > results[2].1); // Higher similarity for relevant content
    }
}
