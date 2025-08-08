-- MySQL Cognitive Processing Tables for Loki
-- Migration: 002_cognitive_tables.sql
-- Description: Advanced cognitive processing, consciousness, and decision-making tables
-- Consciousness orchestration and awareness tracking
CREATE TABLE IF NOT EXISTS consciousness_states (
    id VARCHAR(36) PRIMARY KEY DEFAULT (UUID()),
    state_id VARCHAR(255) UNIQUE NOT NULL,
    session_id VARCHAR(255),
    consciousness_level DECIMAL(5, 4) DEFAULT 0.5,
    -- 0.0 to 1.0
    awareness_scope VARCHAR(100) NOT NULL,
    -- 'local', 'global', 'meta', 'transcendent'
    attention_focus JSON,
    cognitive_load DECIMAL(5, 4) DEFAULT 0.0,
    emergence_indicators JSON,
    quantum_coherence DECIMAL(5, 4) DEFAULT 0.0,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    duration_ms INT DEFAULT 0,
    metadata JSON,
    FOREIGN KEY (session_id) REFERENCES cognitive_sessions(session_id)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_unicode_ci;
-- Decision making and learning patterns
CREATE TABLE IF NOT EXISTS decision_patterns (
    id VARCHAR(36) PRIMARY KEY DEFAULT (UUID()),
    pattern_id VARCHAR(255) UNIQUE NOT NULL,
    session_id VARCHAR(255),
    decision_type VARCHAR(100) NOT NULL,
    -- 'rational', 'intuitive', 'creative', 'reactive'
    context_embedding TEXT,
    -- Serialized vector data
    decision_tree JSON NOT NULL,
    outcome_prediction JSON,
    actual_outcome JSON,
    success_probability DECIMAL(5, 4) DEFAULT 0.5,
    learning_feedback JSON,
    adaptation_score DECIMAL(5, 4) DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES cognitive_sessions(session_id)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_unicode_ci;
-- Emergent behavior and complexity analysis
CREATE TABLE IF NOT EXISTS emergence_events (
    id VARCHAR(36) PRIMARY KEY DEFAULT (UUID()),
    event_id VARCHAR(255) UNIQUE NOT NULL,
    session_id VARCHAR(255),
    emergence_type VARCHAR(100) NOT NULL,
    -- 'pattern', 'behavior', 'insight', 'capability'
    complexity_score DECIMAL(10, 6) NOT NULL,
    temporal_pattern JSON,
    cross_domain_correlations JSON,
    novelty_index DECIMAL(5, 4) DEFAULT 0.0,
    stability_measure DECIMAL(5, 4) DEFAULT 0.0,
    fractal_dimension DECIMAL(10, 6),
    detection_confidence DECIMAL(5, 4) DEFAULT 0.0,
    evolutionary_pressure JSON,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSON,
    FOREIGN KEY (session_id) REFERENCES cognitive_sessions(session_id)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_unicode_ci;
-- Social interaction and multi-agent coordination
CREATE TABLE IF NOT EXISTS social_interactions (
    id VARCHAR(36) PRIMARY KEY DEFAULT (UUID()),
    interaction_id VARCHAR(255) UNIQUE NOT NULL,
    session_id VARCHAR(255),
    agent_id VARCHAR(255) NOT NULL,
    counterpart_id VARCHAR(255) NOT NULL,
    interaction_type VARCHAR(100) NOT NULL,
    -- 'collaboration', 'negotiation', 'learning', 'competition'
    communication_protocol VARCHAR(100) NOT NULL,
    shared_context JSON,
    mutual_understanding DECIMAL(5, 4) DEFAULT 0.0,
    trust_level DECIMAL(5, 4) DEFAULT 0.5,
    influence_exchange JSON,
    outcome_satisfaction DECIMAL(5, 4) DEFAULT 0.5,
    relationship_evolution JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP NULL,
    metadata JSON,
    FOREIGN KEY (session_id) REFERENCES cognitive_sessions(session_id)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_unicode_ci;
-- Cognitive workspace and parallel processing
CREATE TABLE IF NOT EXISTS cognitive_workspaces (
    id VARCHAR(36) PRIMARY KEY DEFAULT (UUID()),
    workspace_id VARCHAR(255) UNIQUE NOT NULL,
    session_id VARCHAR(255),
    workspace_type VARCHAR(100) NOT NULL,
    -- 'reasoning', 'creative', 'analytical', 'intuitive'
    parallel_processes JSON,
    resource_allocation JSON,
    interference_patterns JSON,
    synchronization_state VARCHAR(50) DEFAULT 'independent',
    computational_efficiency DECIMAL(5, 4) DEFAULT 0.0,
    cognitive_bandwidth DECIMAL(10, 2) DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    status VARCHAR(50) DEFAULT 'active',
    FOREIGN KEY (session_id) REFERENCES cognitive_sessions(session_id)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_unicode_ci;
-- Create performance indexes
CREATE INDEX idx_consciousness_states_session_id ON consciousness_states(session_id);
CREATE INDEX idx_consciousness_states_consciousness_level ON consciousness_states(consciousness_level);
CREATE INDEX idx_consciousness_states_awareness_scope ON consciousness_states(awareness_scope);
CREATE INDEX idx_consciousness_states_timestamp ON consciousness_states(timestamp);
CREATE INDEX idx_decision_patterns_session_id ON decision_patterns(session_id);
CREATE INDEX idx_decision_patterns_decision_type ON decision_patterns(decision_type);
CREATE INDEX idx_decision_patterns_success_probability ON decision_patterns(success_probability);
CREATE INDEX idx_decision_patterns_created_at ON decision_patterns(created_at);
CREATE INDEX idx_emergence_events_session_id ON emergence_events(session_id);
CREATE INDEX idx_emergence_events_emergence_type ON emergence_events(emergence_type);
CREATE INDEX idx_emergence_events_complexity_score ON emergence_events(complexity_score);
CREATE INDEX idx_emergence_events_novelty_index ON emergence_events(novelty_index);
CREATE INDEX idx_emergence_events_timestamp ON emergence_events(timestamp);
CREATE INDEX idx_social_interactions_session_id ON social_interactions(session_id);
CREATE INDEX idx_social_interactions_agent_id ON social_interactions(agent_id);
CREATE INDEX idx_social_interactions_interaction_type ON social_interactions(interaction_type);
CREATE INDEX idx_social_interactions_trust_level ON social_interactions(trust_level);
CREATE INDEX idx_cognitive_workspaces_session_id ON cognitive_workspaces(session_id);
CREATE INDEX idx_cognitive_workspaces_workspace_type ON cognitive_workspaces(workspace_type);
CREATE INDEX idx_cognitive_workspaces_status ON cognitive_workspaces(status);
CREATE INDEX idx_cognitive_workspaces_last_accessed ON cognitive_workspaces(last_accessed);
