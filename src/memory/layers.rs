use anyhow::Result;
use std::collections::VecDeque;
use tracing::debug;

use super::{MemoryId, MemoryItem};

/// Type of memory layer
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    /// Short-term memory (working memory)
    ShortTerm,
    /// Long-term memory layer with index
    LongTerm(usize),
}

/// A memory layer in the cognitive architecture
#[derive(Debug)]
pub struct MemoryLayer {
    /// Layer type
    layer_type: LayerType,

    /// Memory items in this layer
    items: VecDeque<MemoryItem>,

    /// Maximum capacity
    capacity: usize,
}

impl MemoryLayer {
    /// Create a new memory layer
    pub fn new(layer_type: LayerType, capacity: usize) -> Self {
        debug!("Creating {:?} memory layer with capacity {}", layer_type, capacity);

        Self {
            layer_type,
            items: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Add a memory item
    pub fn add(&mut self, item: MemoryItem) -> Result<()> {
        // If at capacity, remove oldest
        if self.items.len() >= self.capacity {
            self.items.pop_front();
        }

        self.items.push_back(item);
        Ok(())
    }

    /// Get a memory item by ID
    pub fn get(&self, id: &MemoryId) -> Option<&MemoryItem> {
        self.items.iter().find(|item| &item.id == id)
    }

    /// Get all items
    pub fn all(&self) -> Vec<MemoryItem> {
        self.items.iter().cloned().collect()
    }

    /// Get layer utilization (0.0 to 1.0)
    pub fn utilization(&self) -> f32 {
        self.items.len() as f32 / self.capacity as f32
    }

    /// Extract least relevant items
    pub fn extract_least_relevant(&mut self, count: usize) -> Result<Vec<MemoryItem>> {
        // Sort by relevance score
        let mut sorted_items: Vec<_> = self.items.drain(..).collect();
        sorted_items.sort_by(|a, b| {
            a.relevance_score
                .partial_cmp(&b.relevance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Take the least relevant
        let extracted: Vec<_> = sorted_items.drain(..count.min(sorted_items.len())).collect();

        // Put back the most relevant
        self.items = sorted_items.into();

        Ok(extracted)
    }

    /// Apply decay to all items
    pub fn apply_decay(&mut self, decay_rate: f32) {
        for item in &mut self.items {
            item.relevance_score *= decay_rate;
        }

        // Remove items with very low relevance
        self.items.retain(|item| item.relevance_score > 0.01);
    }

    /// Get the number of items
    #[inline]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Check if empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Get an iterator over the items
    pub fn iter(&self) -> impl Iterator<Item = &MemoryItem> {
        self.items.iter()
    }

    /// Get a mutable iterator over the items
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut MemoryItem> {
        self.items.iter_mut()
    }

    /// Get all items in the layer
    pub fn get_all_items(&self) -> Vec<MemoryItem> {
        self.items.iter().cloned().collect()
    }
    
    /// Get the capacity of this layer
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}
