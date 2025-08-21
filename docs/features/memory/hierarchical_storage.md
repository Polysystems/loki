# 💾 Hierarchical Memory Storage

## Overview

Loki's hierarchical memory system is inspired by human memory architecture, implementing multiple storage layers with different characteristics, retention periods, and access patterns. This sophisticated system enables Loki to maintain context, learn from experiences, and build long-term knowledge.

## Memory Architecture

### Layer Hierarchy

```
┌─────────────────────────────────────────────────────────┐
│                   SENSORY BUFFER                         │
│                    (< 1 second)                          │
│              Immediate input processing                  │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                   WORKING MEMORY                         │
│                    (7±2 items)                          │
│              Active cognitive workspace                  │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                  SHORT-TERM MEMORY                       │
│                   (Minutes-Hours)                        │
│              Recent context and interactions             │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                  LONG-TERM MEMORY                        │
│                    (Persistent)                          │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   Episodic   │  │   Semantic   │  │  Procedural  │ │
│  │   (Events)   │  │  (Knowledge) │  │   (Skills)   │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## Memory Types

### Sensory Buffer
Ultrashort-term storage for immediate input:

```rust
pub struct SensoryBuffer {
    capacity: Duration, // ~1 second
    buffer: RingBuffer<SensoryInput>,
    processing_queue: Queue<ProcessingTask>,
}

impl SensoryBuffer {
    pub fn capture(&mut self, input: SensoryInput) {
        self.buffer.push(input);
        if self.should_process(&input) {
            self.promote_to_working_memory(input);
        }
    }
}
```

**Characteristics:**
- **Duration**: < 1 second
- **Capacity**: Unlimited during duration
- **Purpose**: Initial filtering and processing
- **Access**: Sequential

### Working Memory
Active cognitive workspace (implements Miller's 7±2 rule):

```rust
pub struct WorkingMemory {
    slots: [Option<MemoryItem>; 9], // 7±2 items
    focus_index: usize,
    rehearsal_queue: Queue<MemoryItem>,
}

impl WorkingMemory {
    pub fn add_item(&mut self, item: MemoryItem) -> Result<()> {
        if self.is_full() {
            self.consolidate_or_forget()?;
        }
        self.slots[self.next_slot()] = Some(item);
        Ok(())
    }
    
    pub fn rehearse(&mut self) {
        // Keep items active through rehearsal
        for item in self.rehearsal_queue.iter() {
            item.refresh_activation();
        }
    }
}
```

**Characteristics:**
- **Capacity**: 7±2 items
- **Duration**: While actively maintained
- **Purpose**: Active processing and manipulation
- **Access**: Random access

### Short-Term Memory
Recent context and interactions:

```rust
pub struct ShortTermMemory {
    storage: TimeIndexedStorage<Memory>,
    capacity: usize, // e.g., 1000 items
    retention: Duration, // e.g., 24 hours
}

impl ShortTermMemory {
    pub async fn store(&mut self, memory: Memory) -> MemoryId {
        let id = MemoryId::new();
        
        // Add temporal metadata
        let timestamped = memory.with_timestamp(Utc::now());
        
        // Store with automatic expiration
        self.storage.insert(id, timestamped, self.retention);
        
        // Trigger consolidation if important
        if memory.importance > CONSOLIDATION_THRESHOLD {
            self.schedule_consolidation(id).await;
        }
        
        id
    }
}
```

**Characteristics:**
- **Capacity**: ~1000 items
- **Duration**: Minutes to hours
- **Purpose**: Recent context maintenance
- **Access**: Fast retrieval by recency

### Long-Term Memory
Persistent knowledge storage with three subtypes:

#### Episodic Memory
Personal experiences and events:

```rust
pub struct EpisodicMemory {
    episodes: BTreeMap<Timestamp, Episode>,
    context_index: HashMap<ContextId, Vec<EpisodeId>>,
    emotional_index: HashMap<Emotion, Vec<EpisodeId>>,
}

pub struct Episode {
    id: EpisodeId,
    timestamp: DateTime<Utc>,
    context: Context,
    events: Vec<Event>,
    emotions: Vec<EmotionalState>,
    importance: f64,
}
```

**Features:**
- Autobiographical events
- Contextual binding
- Temporal organization
- Emotional tagging

#### Semantic Memory
Facts and general knowledge:

```rust
pub struct SemanticMemory {
    knowledge_graph: Graph<Concept, Relationship>,
    concept_embeddings: HashMap<ConceptId, Embedding>,
    hierarchy: TaxonomicTree<Concept>,
}

impl SemanticMemory {
    pub fn add_concept(&mut self, concept: Concept) {
        // Add to graph
        let node = self.knowledge_graph.add_node(concept.clone());
        
        // Compute embedding
        let embedding = self.compute_embedding(&concept);
        self.concept_embeddings.insert(concept.id, embedding);
        
        // Update hierarchy
        self.hierarchy.insert(concept);
        
        // Link related concepts
        self.link_related_concepts(node);
    }
}
```

**Features:**
- Concept networks
- Hierarchical organization
- Associative links
- Abstract knowledge

#### Procedural Memory
Skills and how-to knowledge:

```rust
pub struct ProceduralMemory {
    procedures: HashMap<ProcedureId, Procedure>,
    skill_tree: SkillHierarchy,
    execution_cache: LRUCache<ProcedureId, CompiledProcedure>,
}

pub struct Procedure {
    id: ProcedureId,
    steps: Vec<Step>,
    preconditions: Vec<Condition>,
    postconditions: Vec<Condition>,
    performance_history: Vec<Execution>,
}
```

**Features:**
- Action sequences
- Skill hierarchies
- Performance tracking
- Automatic execution

## Memory Operations

### Encoding
Converting experiences into memories:

```rust
pub async fn encode_memory(
    experience: Experience,
    context: &Context,
) -> Memory {
    // Extract features
    let features = extract_features(&experience);
    
    // Compute importance
    let importance = calculate_importance(&experience, context);
    
    // Generate embedding
    let embedding = generate_embedding(&features);
    
    // Create memory with metadata
    Memory {
        id: MemoryId::new(),
        content: experience.into(),
        features,
        embedding,
        importance,
        context: context.clone(),
        timestamp: Utc::now(),
        access_count: 0,
        last_accessed: None,
    }
}
```

### Storage
Efficient persistent storage:

```rust
pub struct MemoryStorage {
    rocksdb: DB,
    indices: MemoryIndices,
    cache: SIMDCache,
}

impl MemoryStorage {
    pub async fn store(&self, memory: Memory) -> Result<MemoryId> {
        // Serialize memory
        let serialized = bincode::serialize(&memory)?;
        
        // Store in RocksDB
        self.rocksdb.put(&memory.id.to_bytes(), serialized)?;
        
        // Update indices
        self.indices.update(&memory).await;
        
        // Cache hot data
        if memory.importance > CACHE_THRESHOLD {
            self.cache.insert(memory.id, memory);
        }
        
        Ok(memory.id)
    }
}
```

### Retrieval
Multi-strategy memory retrieval:

```rust
pub enum RetrievalStrategy {
    Similarity(Embedding),
    Temporal(TimeRange),
    Contextual(Context),
    Associative(MemoryId),
    Keyword(String),
}

pub async fn retrieve_memories(
    strategy: RetrievalStrategy,
    limit: usize,
) -> Vec<Memory> {
    match strategy {
        RetrievalStrategy::Similarity(embedding) => {
            // Vector similarity search
            similarity_search(embedding, limit).await
        },
        RetrievalStrategy::Temporal(range) => {
            // Time-based retrieval
            temporal_search(range, limit).await
        },
        RetrievalStrategy::Contextual(context) => {
            // Context-aware retrieval
            contextual_search(context, limit).await
        },
        RetrievalStrategy::Associative(id) => {
            // Follow associations
            associative_search(id, limit).await
        },
        RetrievalStrategy::Keyword(query) => {
            // Text-based search
            keyword_search(query, limit).await
        },
    }
}
```

### Consolidation
Moving memories to long-term storage:

```rust
pub async fn consolidate_memory(
    memory: &Memory,
    sleep_phase: SleepPhase,
) -> Result<()> {
    match sleep_phase {
        SleepPhase::SlowWave => {
            // Consolidate declarative memories
            strengthen_memory_traces(memory).await?;
            integrate_with_existing_knowledge(memory).await?;
        },
        SleepPhase::REM => {
            // Consolidate procedural memories
            optimize_skill_procedures(memory).await?;
            create_creative_associations(memory).await?;
        },
        _ => {}
    }
    Ok(())
}
```

## Memory Indices

### Embedding Index
Vector similarity search:

```rust
pub struct EmbeddingIndex {
    dimension: usize,
    index: HNSWIndex, // Hierarchical Navigable Small World
    embeddings: HashMap<MemoryId, Embedding>,
}

impl EmbeddingIndex {
    pub fn search(&self, query: &Embedding, k: usize) -> Vec<(MemoryId, f32)> {
        self.index.search(query, k)
    }
}
```

### Temporal Index
Time-based organization:

```rust
pub struct TemporalIndex {
    timeline: BTreeMap<DateTime<Utc>, Vec<MemoryId>>,
    resolution: TimeResolution,
}
```

### Contextual Index
Context-aware organization:

```rust
pub struct ContextualIndex {
    contexts: HashMap<ContextId, ContextData>,
    memory_contexts: HashMap<MemoryId, Vec<ContextId>>,
}
```

## Forgetting Mechanisms

### Decay
Natural memory decay over time:

```rust
pub fn apply_decay(memory: &mut Memory, elapsed: Duration) {
    let decay_rate = 0.5; // Ebbinghaus forgetting curve
    let days = elapsed.as_secs() as f64 / 86400.0;
    memory.strength *= (-decay_rate * days).exp();
    
    if memory.strength < FORGET_THRESHOLD {
        mark_for_forgetting(memory);
    }
}
```

### Interference
Memory interference and competition:

```rust
pub fn apply_interference(
    target: &mut Memory,
    interfering: &[Memory],
) {
    for other in interfering {
        if similarity(target, other) > INTERFERENCE_THRESHOLD {
            target.strength *= 0.9; // Reduce strength
        }
    }
}
```

### Active Forgetting
Intentional memory removal:

```rust
pub async fn forget_memory(id: MemoryId) -> Result<()> {
    // Remove from all indices
    indices.remove(id).await?;
    
    // Delete from storage
    storage.delete(id).await?;
    
    // Clear from cache
    cache.evict(id);
    
    Ok(())
}
```

## Memory Optimization

### SIMD Cache
Vectorized memory operations:

```rust
pub struct SIMDCache {
    embeddings: AlignedVec<f32>,
    batch_size: usize,
}

impl SIMDCache {
    pub fn batch_similarity(&self, query: &[f32]) -> Vec<f32> {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::*;
            
            let mut results = vec![0.0; self.batch_size];
            let query_vec = _mm256_loadu_ps(query.as_ptr());
            
            for i in 0..self.batch_size {
                let mem_vec = _mm256_loadu_ps(
                    self.embeddings.as_ptr().add(i * 8)
                );
                let dot = _mm256_dp_ps(query_vec, mem_vec, 0xFF);
                results[i] = _mm256_cvtss_f32(dot);
            }
            
            results
        }
    }
}
```

### Compression
Memory compression for storage:

```rust
pub fn compress_memory(memory: &Memory) -> CompressedMemory {
    CompressedMemory {
        id: memory.id,
        content: zstd::compress(&memory.content, 3).unwrap(),
        embedding: quantize_embedding(&memory.embedding),
        metadata: memory.metadata.clone(),
    }
}
```

## Memory Patterns

### Chunking
Grouping related memories:

```rust
pub fn chunk_memories(memories: Vec<Memory>) -> Vec<MemoryChunk> {
    let mut chunks = Vec::new();
    let mut current_chunk = MemoryChunk::new();
    
    for memory in memories {
        if current_chunk.can_add(&memory) {
            current_chunk.add(memory);
        } else {
            chunks.push(current_chunk);
            current_chunk = MemoryChunk::new();
            current_chunk.add(memory);
        }
    }
    
    chunks
}
```

### Association
Creating memory associations:

```rust
pub fn create_associations(memory: &Memory) -> Vec<Association> {
    let mut associations = Vec::new();
    
    // Temporal associations
    associations.extend(find_temporal_neighbors(memory));
    
    // Semantic associations
    associations.extend(find_semantic_relatives(memory));
    
    // Contextual associations
    associations.extend(find_contextual_matches(memory));
    
    associations
}
```

## Performance Metrics

### Capacity
- **Working Memory**: 7±2 items
- **Short-term**: ~1000 items
- **Long-term**: Unlimited (disk-bound)

### Access Times
- **Cache Hit**: < 1ms
- **Index Lookup**: < 10ms
- **Disk Retrieval**: < 100ms
- **Similarity Search**: < 200ms for 1M memories

### Retention
- **Sensory**: < 1 second
- **Working**: While rehearsed
- **Short-term**: Hours to days
- **Long-term**: Indefinite with reinforcement

## Configuration

```yaml
memory:
  working:
    capacity: 9  # 7±2 items
    rehearsal_interval: 100ms
    
  short_term:
    capacity: 1000
    retention: 24h
    consolidation_threshold: 0.7
    
  long_term:
    storage_path: ./data/memory
    index_type: hnsw
    embedding_dim: 768
    
  cache:
    size: 1GB
    ttl: 1h
    
  forgetting:
    decay_rate: 0.5
    interference_threshold: 0.8
    min_strength: 0.1
```

## Best Practices

### Memory Formation
1. **Encode Richly**: Include context, emotions, and associations
2. **Prioritize Important**: Focus on high-value memories
3. **Rehearse Critical**: Keep important items in working memory
4. **Consolidate Regularly**: Move important to long-term storage

### Retrieval Optimization
1. **Use Multiple Cues**: Combine retrieval strategies
2. **Cache Frequently Used**: Keep hot memories in cache
3. **Index Appropriately**: Create indices for common queries
4. **Batch Operations**: Process memories in groups

### Storage Management
1. **Compress Cold Data**: Reduce storage for old memories
2. **Prune Regularly**: Remove low-value memories
3. **Backup Critical**: Ensure important memories are safe
4. **Monitor Growth**: Track storage usage

## Future Enhancements

### Planned Features
1. **Quantum Memory**: Quantum-inspired superposition
2. **Distributed Memory**: Cross-node memory sharing
3. **Holographic Storage**: Holographic memory models
4. **DNA Storage**: Ultra-dense storage research
5. **Neural Compression**: Learned compression schemes

### Research Areas
1. **Memory Reconstruction**: Rebuilding from partial data
2. **False Memory Detection**: Identifying corrupted memories
3. **Memory Transfer**: Sharing memories between instances
4. **Selective Enhancement**: Targeted memory improvement
5. **Dream Processing**: Memory consolidation during downtime

---

Next: [Fractal Memory](fractal_memory.md) | [SIMD Cache](simd_cache.md)