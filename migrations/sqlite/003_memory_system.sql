-- SQLite Memory System Tables for Loki
-- Migration: 003_memory_system.sql
-- Description: Hierarchical memory system with knowledge graphs, embeddings, and fractal storage
-- Hierarchical memory layers
CREATE TABLE IF NOT EXISTS memory_layers (
    id TEXT PRIMARY KEY DEFAULT (hex(randomblob(16))),
    layer_id TEXT UNIQUE NOT NULL,
    layer_name TEXT NOT NULL,
    layer_type TEXT NOT NULL,
    -- 'sensory', 'working', 'episodic', 'semantic', 'procedural', 'meta'
    hierarchy_level INTEGER NOT NULL DEFAULT 0,
    parent_layer_id TEXT REFERENCES memory_layers(layer_id),
    storage_capacity INTEGER DEFAULT 0,
    retention_duration_ms INTEGER DEFAULT 0,
    compression_ratio REAL DEFAULT 1.0,
    access_frequency REAL DEFAULT 0.0,
    consolidation_threshold REAL DEFAULT 0.8,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT DEFAULT '{}'
);
-- Memory entities and knowledge nodes
CREATE TABLE IF NOT EXISTS memory_entities (
    id TEXT PRIMARY KEY DEFAULT (hex(randomblob(16))),
    entity_id TEXT UNIQUE NOT NULL,
    entity_type TEXT NOT NULL,
    -- 'concept', 'event', 'pattern', 'skill', 'relation', 'emotion'
    layer_id TEXT REFERENCES memory_layers(layer_id),
    content_hash TEXT NOT NULL,
    content_text TEXT,
    content_data BLOB,
    embedding_vector TEXT,
    -- Serialized vector data
    confidence_score REAL DEFAULT 0.5,
    relevance_score REAL DEFAULT 0.5,
    access_count INTEGER DEFAULT 0,
    last_accessed DATETIME DEFAULT CURRENT_TIMESTAMP,
    importance_weight REAL DEFAULT 1.0,
    decay_rate REAL DEFAULT 0.0001,
    consolidation_level REAL DEFAULT 0.0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT DEFAULT '{}'
);
-- Knowledge graph relationships
CREATE TABLE IF NOT EXISTS memory_relations (
    id TEXT PRIMARY KEY DEFAULT (hex(randomblob(16))),
    relation_id TEXT UNIQUE NOT NULL,
    source_entity_id TEXT REFERENCES memory_entities(entity_id),
    target_entity_id TEXT REFERENCES memory_entities(entity_id),
    relation_type TEXT NOT NULL,
    -- 'causes', 'enables', 'inhibits', 'resembles', 'contains', 'follows'
    strength REAL DEFAULT 0.5,
    bidirectional INTEGER DEFAULT 0,
    -- BOOLEAN as INTEGER
    temporal_order INTEGER DEFAULT 0,
    causal_weight REAL DEFAULT 0.0,
    evidence_count INTEGER DEFAULT 1,
    confidence REAL DEFAULT 0.5,
    last_reinforced DATETIME DEFAULT CURRENT_TIMESTAMP,
    decay_rate REAL DEFAULT 0.0001,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT DEFAULT '{}'
);
-- Fractal memory patterns and self-similar structures
CREATE TABLE IF NOT EXISTS fractal_memories (
    id TEXT PRIMARY KEY DEFAULT (hex(randomblob(16))),
    fractal_id TEXT UNIQUE NOT NULL,
    entity_id TEXT REFERENCES memory_entities(entity_id),
    pattern_template TEXT NOT NULL,
    -- JSON as TEXT
    scale_factor REAL DEFAULT 1.0,
    recursion_depth INTEGER DEFAULT 1,
    self_similarity_score REAL DEFAULT 0.0,
    fractal_dimension REAL DEFAULT 1.0,
    scaling_invariance REAL DEFAULT 0.0,
    pattern_variants TEXT DEFAULT '[]',
    -- JSON array as TEXT
    generation_rules TEXT DEFAULT '{}',
    compression_efficiency REAL DEFAULT 0.0,
    reconstruction_fidelity REAL DEFAULT 1.0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT DEFAULT '{}'
);
-- Episodic memory sequences and temporal patterns
CREATE TABLE IF NOT EXISTS episodic_sequences (
    id TEXT PRIMARY KEY DEFAULT (hex(randomblob(16))),
    sequence_id TEXT UNIQUE NOT NULL,
    episode_title TEXT,
    start_time DATETIME NOT NULL,
    end_time DATETIME,
    duration_ms INTEGER DEFAULT 0,
    sequence_events TEXT NOT NULL DEFAULT '[]',
    -- JSON array as TEXT
    contextual_factors TEXT DEFAULT '{}',
    emotional_valence REAL DEFAULT 0.0,
    significance_score REAL DEFAULT 0.5,
    recall_accuracy REAL DEFAULT 1.0,
    consolidation_status TEXT DEFAULT 'fresh',
    interference_level REAL DEFAULT 0.0,
    retrieval_cues TEXT DEFAULT '[]',
    -- JSON array as TEXT
    associated_entities TEXT DEFAULT '[]',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT DEFAULT '{}'
);
-- Memory embedding clusters and semantic spaces
CREATE TABLE IF NOT EXISTS embedding_clusters (
    id TEXT PRIMARY KEY DEFAULT (hex(randomblob(16))),
    cluster_id TEXT UNIQUE NOT NULL,
    cluster_name TEXT,
    semantic_space TEXT NOT NULL,
    -- 'conceptual', 'linguistic', 'perceptual', 'procedural'
    centroid_vector TEXT,
    -- Serialized vector
    cluster_radius REAL DEFAULT 0.0,
    member_count INTEGER DEFAULT 0,
    cluster_density REAL DEFAULT 0.0,
    intra_cluster_similarity REAL DEFAULT 0.0,
    inter_cluster_distance REAL DEFAULT 0.0,
    cluster_stability REAL DEFAULT 0.0,
    formation_threshold REAL DEFAULT 0.8,
    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT DEFAULT '{}'
);
-- Memory access patterns and usage analytics
CREATE TABLE IF NOT EXISTS memory_access_patterns (
    id TEXT PRIMARY KEY DEFAULT (hex(randomblob(16))),
    pattern_id TEXT UNIQUE NOT NULL,
    entity_id TEXT REFERENCES memory_entities(entity_id),
    access_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    access_type TEXT NOT NULL,
    -- 'read', 'write', 'update', 'strengthen', 'weaken'
    access_context TEXT DEFAULT '{}',
    retrieval_latency_ms INTEGER DEFAULT 0,
    retrieval_confidence REAL DEFAULT 1.0,
    interference_detected INTEGER DEFAULT 0,
    -- BOOLEAN as INTEGER
    priming_effects TEXT DEFAULT '{}',
    spreading_activation TEXT DEFAULT '{}',
    memory_trace_strength REAL DEFAULT 1.0,
    consolidation_triggered INTEGER DEFAULT 0,
    -- BOOLEAN as INTEGER
    session_id TEXT REFERENCES cognitive_sessions(session_id),
    metadata TEXT DEFAULT '{}'
);
-- Create performance indexes for memory operations
CREATE INDEX IF NOT EXISTS idx_memory_layers_layer_type ON memory_layers(layer_type);
CREATE INDEX IF NOT EXISTS idx_memory_layers_hierarchy_level ON memory_layers(hierarchy_level);
CREATE INDEX IF NOT EXISTS idx_memory_layers_parent_layer_id ON memory_layers(parent_layer_id);
CREATE INDEX IF NOT EXISTS idx_memory_entities_entity_type ON memory_entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_memory_entities_layer_id ON memory_entities(layer_id);
CREATE INDEX IF NOT EXISTS idx_memory_entities_content_hash ON memory_entities(content_hash);
CREATE INDEX IF NOT EXISTS idx_memory_entities_confidence_score ON memory_entities(confidence_score);
CREATE INDEX IF NOT EXISTS idx_memory_entities_relevance_score ON memory_entities(relevance_score);
CREATE INDEX IF NOT EXISTS idx_memory_entities_importance_weight ON memory_entities(importance_weight);
CREATE INDEX IF NOT EXISTS idx_memory_entities_last_accessed ON memory_entities(last_accessed);
CREATE INDEX IF NOT EXISTS idx_memory_entities_consolidation_level ON memory_entities(consolidation_level);
CREATE INDEX IF NOT EXISTS idx_memory_relations_source_entity_id ON memory_relations(source_entity_id);
CREATE INDEX IF NOT EXISTS idx_memory_relations_target_entity_id ON memory_relations(target_entity_id);
CREATE INDEX IF NOT EXISTS idx_memory_relations_relation_type ON memory_relations(relation_type);
CREATE INDEX IF NOT EXISTS idx_memory_relations_strength ON memory_relations(strength);
CREATE INDEX IF NOT EXISTS idx_memory_relations_confidence ON memory_relations(confidence);
CREATE INDEX IF NOT EXISTS idx_fractal_memories_entity_id ON fractal_memories(entity_id);
CREATE INDEX IF NOT EXISTS idx_fractal_memories_scale_factor ON fractal_memories(scale_factor);
CREATE INDEX IF NOT EXISTS idx_fractal_memories_recursion_depth ON fractal_memories(recursion_depth);
CREATE INDEX IF NOT EXISTS idx_fractal_memories_self_similarity_score ON fractal_memories(self_similarity_score);
CREATE INDEX IF NOT EXISTS idx_episodic_sequences_start_time ON episodic_sequences(start_time);
CREATE INDEX IF NOT EXISTS idx_episodic_sequences_significance_score ON episodic_sequences(significance_score);
CREATE INDEX IF NOT EXISTS idx_episodic_sequences_consolidation_status ON episodic_sequences(consolidation_status);
CREATE INDEX IF NOT EXISTS idx_embedding_clusters_semantic_space ON embedding_clusters(semantic_space);
CREATE INDEX IF NOT EXISTS idx_embedding_clusters_cluster_density ON embedding_clusters(cluster_density);
CREATE INDEX IF NOT EXISTS idx_embedding_clusters_cluster_stability ON embedding_clusters(cluster_stability);
CREATE INDEX IF NOT EXISTS idx_memory_access_patterns_entity_id ON memory_access_patterns(entity_id);
CREATE INDEX IF NOT EXISTS idx_memory_access_patterns_access_type ON memory_access_patterns(access_type);
CREATE INDEX IF NOT EXISTS idx_memory_access_patterns_access_timestamp ON memory_access_patterns(access_timestamp);
CREATE INDEX IF NOT EXISTS idx_memory_access_patterns_session_id ON memory_access_patterns(session_id);
-- Add triggers for automatic updates
CREATE TRIGGER IF NOT EXISTS update_memory_entities_updated_at
AFTER
UPDATE ON memory_entities FOR EACH ROW BEGIN
UPDATE memory_entities
SET updated_at = CURRENT_TIMESTAMP
WHERE id = NEW.id;
END;
-- Useful views for memory analysis (SQLite version)
CREATE VIEW IF NOT EXISTS memory_hierarchy_summary AS
SELECT ml.layer_name,
    ml.layer_type,
    ml.hierarchy_level,
    COUNT(me.id) as entity_count,
    AVG(me.importance_weight) as avg_importance,
    AVG(me.consolidation_level) as avg_consolidation,
    SUM(me.access_count) as total_accesses
FROM memory_layers ml
    LEFT JOIN memory_entities me ON ml.layer_id = me.layer_id
GROUP BY ml.layer_id,
    ml.layer_name,
    ml.layer_type,
    ml.hierarchy_level
ORDER BY ml.hierarchy_level;
CREATE VIEW IF NOT EXISTS knowledge_graph_metrics AS
SELECT source.entity_type as source_type,
    target.entity_type as target_type,
    mr.relation_type,
    COUNT(*) as relation_count,
    AVG(mr.strength) as avg_strength,
    AVG(mr.confidence) as avg_confidence
FROM memory_relations mr
    JOIN memory_entities source ON mr.source_entity_id = source.entity_id
    JOIN memory_entities target ON mr.target_entity_id = target.entity_id
GROUP BY source.entity_type,
    target.entity_type,
    mr.relation_type
ORDER BY relation_count DESC;
