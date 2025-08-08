-- PostgreSQL Memory System Tables for Loki
-- Migration: 003_memory_system.sql
-- Description: Hierarchical memory system with knowledge graphs, embeddings, and fractal storage
-- Hierarchical memory layers
CREATE TABLE IF NOT EXISTS memory_layers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    layer_id VARCHAR(255) UNIQUE NOT NULL,
    layer_name VARCHAR(255) NOT NULL,
    layer_type VARCHAR(100) NOT NULL,
    -- 'sensory', 'working', 'episodic', 'semantic', 'procedural', 'meta'
    hierarchy_level INTEGER NOT NULL DEFAULT 0,
    parent_layer_id VARCHAR(255) REFERENCES memory_layers(layer_id),
    storage_capacity BIGINT DEFAULT 0,
    retention_duration_ms BIGINT DEFAULT 0,
    compression_ratio DECIMAL(10, 6) DEFAULT 1.0,
    access_frequency DECIMAL(10, 6) DEFAULT 0.0,
    consolidation_threshold DECIMAL(5, 4) DEFAULT 0.8,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);
-- Memory entities and knowledge nodes
CREATE TABLE IF NOT EXISTS memory_entities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_id VARCHAR(255) UNIQUE NOT NULL,
    entity_type VARCHAR(100) NOT NULL,
    -- 'concept', 'event', 'pattern', 'skill', 'relation', 'emotion'
    layer_id VARCHAR(255) REFERENCES memory_layers(layer_id),
    content_hash VARCHAR(64) NOT NULL,
    content_text TEXT,
    content_data BYTEA,
    embedding_vector VECTOR(768),
    -- High-dimensional representation
    confidence_score DECIMAL(5, 4) DEFAULT 0.5,
    relevance_score DECIMAL(5, 4) DEFAULT 0.5,
    access_count BIGINT DEFAULT 0,
    last_accessed TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    importance_weight DECIMAL(10, 6) DEFAULT 1.0,
    decay_rate DECIMAL(10, 8) DEFAULT 0.0001,
    consolidation_level DECIMAL(5, 4) DEFAULT 0.0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);
-- Knowledge graph relationships
CREATE TABLE IF NOT EXISTS memory_relations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    relation_id VARCHAR(255) UNIQUE NOT NULL,
    source_entity_id VARCHAR(255) REFERENCES memory_entities(entity_id),
    target_entity_id VARCHAR(255) REFERENCES memory_entities(entity_id),
    relation_type VARCHAR(100) NOT NULL,
    -- 'causes', 'enables', 'inhibits', 'resembles', 'contains', 'follows'
    strength DECIMAL(5, 4) DEFAULT 0.5,
    bidirectional BOOLEAN DEFAULT false,
    temporal_order INTEGER DEFAULT 0,
    causal_weight DECIMAL(10, 6) DEFAULT 0.0,
    evidence_count INTEGER DEFAULT 1,
    confidence DECIMAL(5, 4) DEFAULT 0.5,
    last_reinforced TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    decay_rate DECIMAL(10, 8) DEFAULT 0.0001,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);
-- Fractal memory patterns and self-similar structures
CREATE TABLE IF NOT EXISTS fractal_memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    fractal_id VARCHAR(255) UNIQUE NOT NULL,
    entity_id VARCHAR(255) REFERENCES memory_entities(entity_id),
    pattern_template JSONB NOT NULL,
    scale_factor DECIMAL(10, 6) DEFAULT 1.0,
    recursion_depth INTEGER DEFAULT 1,
    self_similarity_score DECIMAL(5, 4) DEFAULT 0.0,
    fractal_dimension DECIMAL(10, 6) DEFAULT 1.0,
    scaling_invariance DECIMAL(5, 4) DEFAULT 0.0,
    pattern_variants JSONB DEFAULT '[]',
    generation_rules JSONB DEFAULT '{}',
    compression_efficiency DECIMAL(5, 4) DEFAULT 0.0,
    reconstruction_fidelity DECIMAL(5, 4) DEFAULT 1.0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);
-- Episodic memory sequences and temporal patterns
CREATE TABLE IF NOT EXISTS episodic_sequences (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sequence_id VARCHAR(255) UNIQUE NOT NULL,
    episode_title VARCHAR(500),
    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    end_time TIMESTAMP WITH TIME ZONE,
    duration_ms BIGINT DEFAULT 0,
    sequence_events JSONB NOT NULL DEFAULT '[]',
    contextual_factors JSONB DEFAULT '{}',
    emotional_valence DECIMAL(5, 4) DEFAULT 0.0,
    significance_score DECIMAL(5, 4) DEFAULT 0.5,
    recall_accuracy DECIMAL(5, 4) DEFAULT 1.0,
    consolidation_status VARCHAR(50) DEFAULT 'fresh',
    interference_level DECIMAL(5, 4) DEFAULT 0.0,
    retrieval_cues JSONB DEFAULT '[]',
    associated_entities JSONB DEFAULT '[]',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);
-- Memory embedding clusters and semantic spaces
CREATE TABLE IF NOT EXISTS embedding_clusters (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    cluster_id VARCHAR(255) UNIQUE NOT NULL,
    cluster_name VARCHAR(255),
    semantic_space VARCHAR(100) NOT NULL,
    -- 'conceptual', 'linguistic', 'perceptual', 'procedural'
    centroid_vector VECTOR(768),
    cluster_radius DECIMAL(10, 6) DEFAULT 0.0,
    member_count INTEGER DEFAULT 0,
    cluster_density DECIMAL(10, 6) DEFAULT 0.0,
    intra_cluster_similarity DECIMAL(5, 4) DEFAULT 0.0,
    inter_cluster_distance DECIMAL(10, 6) DEFAULT 0.0,
    cluster_stability DECIMAL(5, 4) DEFAULT 0.0,
    formation_threshold DECIMAL(5, 4) DEFAULT 0.8,
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);
-- Memory access patterns and usage analytics
CREATE TABLE IF NOT EXISTS memory_access_patterns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pattern_id VARCHAR(255) UNIQUE NOT NULL,
    entity_id VARCHAR(255) REFERENCES memory_entities(entity_id),
    access_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    access_type VARCHAR(50) NOT NULL,
    -- 'read', 'write', 'update', 'strengthen', 'weaken'
    access_context JSONB DEFAULT '{}',
    retrieval_latency_ms INTEGER DEFAULT 0,
    retrieval_confidence DECIMAL(5, 4) DEFAULT 1.0,
    interference_detected BOOLEAN DEFAULT false,
    priming_effects JSONB DEFAULT '{}',
    spreading_activation JSONB DEFAULT '{}',
    memory_trace_strength DECIMAL(10, 6) DEFAULT 1.0,
    consolidation_triggered BOOLEAN DEFAULT false,
    session_id VARCHAR(255) REFERENCES cognitive_sessions(session_id),
    metadata JSONB DEFAULT '{}'
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
CREATE TRIGGER update_memory_entities_updated_at BEFORE
UPDATE ON memory_entities FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
-- Memory consolidation and cleanup functions
CREATE OR REPLACE FUNCTION consolidate_memory_entity(entity_id_param VARCHAR(255)) RETURNS BOOLEAN AS $$
DECLARE current_consolidation DECIMAL(5, 4);
access_frequency DECIMAL(10, 6);
importance DECIMAL(10, 6);
BEGIN -- Calculate consolidation level based on access patterns and importance
SELECT COALESCE(consolidation_level, 0.0),
    COALESCE(access_count, 0)::DECIMAL / EXTRACT(
        EPOCH
        FROM (NOW() - created_at)
    ) * 86400,
    COALESCE(importance_weight, 1.0) INTO current_consolidation,
    access_frequency,
    importance
FROM memory_entities
WHERE entity_id = entity_id_param;
-- Update consolidation level
UPDATE memory_entities
SET consolidation_level = LEAST(
        1.0,
        current_consolidation + (access_frequency * importance * 0.01)
    ),
    updated_at = NOW()
WHERE entity_id = entity_id_param;
RETURN TRUE;
END;
$$ LANGUAGE plpgsql;
-- Memory decay function
CREATE OR REPLACE FUNCTION apply_memory_decay() RETURNS INTEGER AS $$
DECLARE affected_rows INTEGER := 0;
BEGIN -- Apply exponential decay to memory entities
UPDATE memory_entities
SET importance_weight = importance_weight * EXP(
        - decay_rate * EXTRACT(
            EPOCH
            FROM (NOW() - last_accessed)
        ) / 86400
    ),
    updated_at = NOW()
WHERE last_accessed < NOW() - INTERVAL '1 day'
    AND decay_rate > 0;
GET DIAGNOSTICS affected_rows = ROW_COUNT;
-- Apply decay to relations
UPDATE memory_relations
SET strength = strength * EXP(
        - decay_rate * EXTRACT(
            EPOCH
            FROM (NOW() - last_reinforced)
        ) / 86400
    )
WHERE last_reinforced < NOW() - INTERVAL '1 day'
    AND decay_rate > 0
    AND strength > 0.01;
-- Keep minimum strength
RETURN affected_rows;
END;
$$ LANGUAGE plpgsql;
-- Useful views for memory analysis
CREATE OR REPLACE VIEW memory_hierarchy_summary AS
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
CREATE OR REPLACE VIEW knowledge_graph_metrics AS
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
