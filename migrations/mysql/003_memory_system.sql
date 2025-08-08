-- MySQL Memory System Tables for Loki
-- Migration: 003_memory_system.sql
-- Description: Hierarchical memory system with knowledge graphs, embeddings, and fractal storage
-- Hierarchical memory layers
CREATE TABLE IF NOT EXISTS memory_layers (
    id VARCHAR(36) PRIMARY KEY DEFAULT (UUID()),
    layer_id VARCHAR(255) UNIQUE NOT NULL,
    layer_name VARCHAR(255) NOT NULL,
    layer_type VARCHAR(100) NOT NULL,
    -- 'sensory', 'working', 'episodic', 'semantic', 'procedural', 'meta'
    hierarchy_level INT NOT NULL DEFAULT 0,
    parent_layer_id VARCHAR(255),
    storage_capacity BIGINT DEFAULT 0,
    retention_duration_ms BIGINT DEFAULT 0,
    compression_ratio DECIMAL(10, 6) DEFAULT 1.0,
    access_frequency DECIMAL(10, 6) DEFAULT 0.0,
    consolidation_threshold DECIMAL(5, 4) DEFAULT 0.8,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSON,
    FOREIGN KEY (parent_layer_id) REFERENCES memory_layers(layer_id)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_unicode_ci;
-- Memory entities and knowledge nodes
CREATE TABLE IF NOT EXISTS memory_entities (
    id VARCHAR(36) PRIMARY KEY DEFAULT (UUID()),
    entity_id VARCHAR(255) UNIQUE NOT NULL,
    entity_type VARCHAR(100) NOT NULL,
    -- 'concept', 'event', 'pattern', 'skill', 'relation', 'emotion'
    layer_id VARCHAR(255),
    content_hash VARCHAR(64) NOT NULL,
    content_text TEXT,
    content_data LONGBLOB,
    embedding_vector TEXT,
    -- Serialized vector data
    confidence_score DECIMAL(5, 4) DEFAULT 0.5,
    relevance_score DECIMAL(5, 4) DEFAULT 0.5,
    access_count BIGINT DEFAULT 0,
    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    importance_weight DECIMAL(10, 6) DEFAULT 1.0,
    decay_rate DECIMAL(10, 8) DEFAULT 0.0001,
    consolidation_level DECIMAL(5, 4) DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    metadata JSON,
    FOREIGN KEY (layer_id) REFERENCES memory_layers(layer_id)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_unicode_ci;
-- Knowledge graph relationships
CREATE TABLE IF NOT EXISTS memory_relations (
    id VARCHAR(36) PRIMARY KEY DEFAULT (UUID()),
    relation_id VARCHAR(255) UNIQUE NOT NULL,
    source_entity_id VARCHAR(255),
    target_entity_id VARCHAR(255),
    relation_type VARCHAR(100) NOT NULL,
    -- 'causes', 'enables', 'inhibits', 'resembles', 'contains', 'follows'
    strength DECIMAL(5, 4) DEFAULT 0.5,
    bidirectional BOOLEAN DEFAULT FALSE,
    temporal_order INT DEFAULT 0,
    causal_weight DECIMAL(10, 6) DEFAULT 0.0,
    evidence_count INT DEFAULT 1,
    confidence DECIMAL(5, 4) DEFAULT 0.5,
    last_reinforced TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    decay_rate DECIMAL(10, 8) DEFAULT 0.0001,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSON,
    FOREIGN KEY (source_entity_id) REFERENCES memory_entities(entity_id),
    FOREIGN KEY (target_entity_id) REFERENCES memory_entities(entity_id)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_unicode_ci;
-- Fractal memory patterns and self-similar structures
CREATE TABLE IF NOT EXISTS fractal_memories (
    id VARCHAR(36) PRIMARY KEY DEFAULT (UUID()),
    fractal_id VARCHAR(255) UNIQUE NOT NULL,
    entity_id VARCHAR(255),
    pattern_template JSON NOT NULL,
    scale_factor DECIMAL(10, 6) DEFAULT 1.0,
    recursion_depth INT DEFAULT 1,
    self_similarity_score DECIMAL(5, 4) DEFAULT 0.0,
    fractal_dimension DECIMAL(10, 6) DEFAULT 1.0,
    scaling_invariance DECIMAL(5, 4) DEFAULT 0.0,
    pattern_variants JSON,
    generation_rules JSON,
    compression_efficiency DECIMAL(5, 4) DEFAULT 0.0,
    reconstruction_fidelity DECIMAL(5, 4) DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSON,
    FOREIGN KEY (entity_id) REFERENCES memory_entities(entity_id)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_unicode_ci;
-- Episodic memory sequences and temporal patterns
CREATE TABLE IF NOT EXISTS episodic_sequences (
    id VARCHAR(36) PRIMARY KEY DEFAULT (UUID()),
    sequence_id VARCHAR(255) UNIQUE NOT NULL,
    episode_title VARCHAR(500),
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP NULL,
    duration_ms BIGINT DEFAULT 0,
    sequence_events JSON NOT NULL,
    contextual_factors JSON,
    emotional_valence DECIMAL(5, 4) DEFAULT 0.0,
    significance_score DECIMAL(5, 4) DEFAULT 0.5,
    recall_accuracy DECIMAL(5, 4) DEFAULT 1.0,
    consolidation_status VARCHAR(50) DEFAULT 'fresh',
    interference_level DECIMAL(5, 4) DEFAULT 0.0,
    retrieval_cues JSON,
    associated_entities JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSON
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_unicode_ci;
-- Memory access patterns and usage analytics
CREATE TABLE IF NOT EXISTS memory_access_patterns (
    id VARCHAR(36) PRIMARY KEY DEFAULT (UUID()),
    pattern_id VARCHAR(255) UNIQUE NOT NULL,
    entity_id VARCHAR(255),
    access_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    access_type VARCHAR(50) NOT NULL,
    -- 'read', 'write', 'update', 'strengthen', 'weaken'
    access_context JSON,
    retrieval_latency_ms INT DEFAULT 0,
    retrieval_confidence DECIMAL(5, 4) DEFAULT 1.0,
    interference_detected BOOLEAN DEFAULT FALSE,
    priming_effects JSON,
    spreading_activation JSON,
    memory_trace_strength DECIMAL(10, 6) DEFAULT 1.0,
    consolidation_triggered BOOLEAN DEFAULT FALSE,
    session_id VARCHAR(255),
    metadata JSON,
    FOREIGN KEY (entity_id) REFERENCES memory_entities(entity_id),
    FOREIGN KEY (session_id) REFERENCES cognitive_sessions(session_id)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_unicode_ci;
-- Create performance indexes for memory operations
CREATE INDEX idx_memory_layers_layer_type ON memory_layers(layer_type);
CREATE INDEX idx_memory_layers_hierarchy_level ON memory_layers(hierarchy_level);
CREATE INDEX idx_memory_layers_parent_layer_id ON memory_layers(parent_layer_id);
CREATE INDEX idx_memory_entities_entity_type ON memory_entities(entity_type);
CREATE INDEX idx_memory_entities_layer_id ON memory_entities(layer_id);
CREATE INDEX idx_memory_entities_content_hash ON memory_entities(content_hash);
CREATE INDEX idx_memory_entities_confidence_score ON memory_entities(confidence_score);
CREATE INDEX idx_memory_entities_relevance_score ON memory_entities(relevance_score);
CREATE INDEX idx_memory_entities_importance_weight ON memory_entities(importance_weight);
CREATE INDEX idx_memory_entities_last_accessed ON memory_entities(last_accessed);
CREATE INDEX idx_memory_entities_consolidation_level ON memory_entities(consolidation_level);
CREATE INDEX idx_memory_relations_source_entity_id ON memory_relations(source_entity_id);
CREATE INDEX idx_memory_relations_target_entity_id ON memory_relations(target_entity_id);
CREATE INDEX idx_memory_relations_relation_type ON memory_relations(relation_type);
CREATE INDEX idx_memory_relations_strength ON memory_relations(strength);
CREATE INDEX idx_memory_relations_confidence ON memory_relations(confidence);
CREATE INDEX idx_fractal_memories_entity_id ON fractal_memories(entity_id);
CREATE INDEX idx_fractal_memories_scale_factor ON fractal_memories(scale_factor);
CREATE INDEX idx_fractal_memories_recursion_depth ON fractal_memories(recursion_depth);
CREATE INDEX idx_fractal_memories_self_similarity_score ON fractal_memories(self_similarity_score);
CREATE INDEX idx_episodic_sequences_start_time ON episodic_sequences(start_time);
CREATE INDEX idx_episodic_sequences_significance_score ON episodic_sequences(significance_score);
CREATE INDEX idx_episodic_sequences_consolidation_status ON episodic_sequences(consolidation_status);
CREATE INDEX idx_memory_access_patterns_entity_id ON memory_access_patterns(entity_id);
CREATE INDEX idx_memory_access_patterns_access_type ON memory_access_patterns(access_type);
CREATE INDEX idx_memory_access_patterns_access_timestamp ON memory_access_patterns(access_timestamp);
CREATE INDEX idx_memory_access_patterns_session_id ON memory_access_patterns(session_id);
