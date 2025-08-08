use anyhow::{Context, Result};
use rocksdb::{DB, Options};
use std::path::Path;
use tracing::{debug, info, warn};

use super::{MemoryId, MemoryItem};

/// Persistent storage for memories
#[derive(Debug)]
pub struct PersistentMemory {
    /// RocksDB instance
    db: DB,
}

impl PersistentMemory {
    /// Create a new persistent memory store with robust error handling
    pub async fn new(base_path: &Path) -> Result<Self> {
        info!("Opening persistent memory at {:?}", base_path);

        // Create directory if needed
        tokio::fs::create_dir_all(base_path).await?;

        let db_path = base_path.join("memories");

        // Try to open RocksDB with automatic lock file cleanup
        let db = Self::open_rocksdb_with_cleanup(&db_path).await
            .with_context(|| format!("Failed to open RocksDB at {:?}", db_path))?;

        Ok(Self { db })
    }

    /// Open RocksDB with automatic stale lock file cleanup
    async fn open_rocksdb_with_cleanup(db_path: &Path) -> Result<DB> {
        // Configure RocksDB
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_max_open_files(1000);
        opts.set_write_buffer_size(64 * 1024 * 1024); // 64MB

        // First attempt to open normally
        match DB::open(&opts, db_path) {
            Ok(db) => {
                info!("✅ RocksDB opened successfully");
                return Ok(db);
            }
            Err(e) => {
                let error_msg = e.to_string();

                // Check if it's a lock file issue
                if error_msg.contains("lock file") || error_msg.contains("LOCK") {
                    warn!("🔧 Detected stale RocksDB lock file, attempting cleanup...");

                    if let Err(cleanup_err) = Self::cleanup_stale_lock_file(db_path).await {
                        warn!("Failed to cleanup lock file: {}", cleanup_err);
                    }

                    // Try opening again after cleanup
                    match DB::open(&opts, db_path) {
                        Ok(db) => {
                            info!("✅ RocksDB opened successfully after lock cleanup");
                            return Ok(db);
                        }
                        Err(second_err) => {
                            warn!("Failed to open RocksDB even after lock cleanup: {}", second_err);
                            return Err(second_err.into());
                        }
                    }
                } else {
                    // Non-lock related error, just return it
                    return Err(e.into());
                }
            }
        }
    }

    /// Clean up stale RocksDB lock file
    async fn cleanup_stale_lock_file(db_path: &Path) -> Result<()> {
        let lock_file = db_path.join("LOCK");

        if lock_file.exists() {
            info!("🧹 Removing stale RocksDB lock file: {:?}", lock_file);

            // Check if the lock file is actually stale by looking at its size
            let metadata = tokio::fs::metadata(&lock_file).await?;

            if metadata.len() == 0 {
                // Empty lock file is definitely stale
                tokio::fs::remove_file(&lock_file).await?;
                info!("✅ Removed empty stale lock file");
            } else {
                // Non-empty lock file might indicate active use
                // Check if file is older than 5 minutes (probably stale)
                if let Ok(modified) = metadata.modified() {
                    let age = modified.elapsed().unwrap_or_default();
                    if age.as_secs() > 300 { // 5 minutes
                        tokio::fs::remove_file(&lock_file).await?;
                        info!("✅ Removed old lock file (age: {:?})", age);
                    } else {
                        return Err(anyhow::anyhow!(
                            "Lock file appears to be from active RocksDB instance (age: {:?})",
                            age
                        ));
                    }
                }
            }
        }

        Ok(())
    }

    /// Save a memory item
    pub async fn save(&self, item: &MemoryItem) -> Result<()> {
        // Get key using format! to access string value
        let key = format!("{:?}", item.id);
        let value = rmp_serde::to_vec(item)
            .context("Failed to serialize memory")?;

        self.db.put(key.as_bytes(), value)
            .context("Failed to save memory")?;

        Ok(())
    }

    /// Load a memory item
    pub async fn load(&self, id: &MemoryId) -> Result<Option<MemoryItem>> {
        let key = format!("{:?}", id);

        match self.db.get(key.as_bytes())? {
            Some(value) => {
                let item = rmp_serde::from_slice(&value)
                    .context("Failed to deserialize memory")?;
                Ok(Some(item))
            }
            None => Ok(None),
        }
    }

    /// Save all memories
    pub async fn save_all(&self, items: &[MemoryItem]) -> Result<()> {
        debug!("Saving {} memories to disk", items.len());

        for item in items {
            self.save(item).await?;
        }

        // Flush to ensure persistence
        self.db.flush().context("Failed to flush database")?;

        Ok(())
    }

    /// Load all memories (optimized async state machine with batching)
    #[inline(never)] // Large async function - prevent inlining for better code generation
    pub async fn load_all(&self) -> Result<Vec<MemoryItem>> {
        // Optimize: Process synchronously since RocksDB iterator is not async anyway
        let mut raw_items = Vec::new();
        let iter = self.db.iterator(rocksdb::IteratorMode::Start);
        
        for result in iter {
            match result {
                Ok((_key, value)) => raw_items.push(value.to_vec()),
                Err(e) => debug!("Failed to read from database: {}", e),
            }
        }

        // CPU-intensive deserialization (optimized batch processing)
        let items = Self::deserialize_batch(&raw_items);
        info!("Loaded {} memories from disk", items.len());
        Ok(items)
    }
    
    /// Helper for batch deserialization (pure function, no async state)
    #[inline(always)]
    fn deserialize_batch(raw_data: &[Vec<u8>]) -> Vec<MemoryItem> {
        raw_data
            .iter()
            .filter_map(|value| {
                match rmp_serde::from_slice(value) {
                    Ok(item) => Some(item),
                    Err(e) => {
                        debug!("Failed to deserialize memory: {}", e);
                        None
                    }
                }
            })
            .collect()
    }

    /// Delete a memory
    pub async fn delete(&self, id: &MemoryId) -> Result<()> {
        let key = format!("{:?}", id);
        self.db.delete(key.as_bytes())
            .context("Failed to delete memory")?;
        Ok(())
    }

    /// Compact the database
    pub async fn compact(&self) -> Result<()> {
        self.db.compact_range(None::<&[u8]>, None::<&[u8]>);
        Ok(())
    }
}
