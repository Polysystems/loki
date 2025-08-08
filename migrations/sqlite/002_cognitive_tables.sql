-- SQLite Cognitive Processing Tables for Loki
-- Migration: 002_cognitive_tables.sql
-- Description: Advanced cognitive processing, consciousness, and decision-making tables
-- Consciousness orchestration and awareness tracking
CREATE TABLE IF NOT EXISTS consciousness_states (
    id TEXT PRIMARY KEY DEFAULT (hex(randomblob(16))),
    state_id TEXT UNIQUE NOT NULL,
    session_id TEXT REFERENCES cognitive_sessions(session_id),
    consciousness_level REAL DEFAULT 0.5,
    -- 0.0 to 1.0
    awareness_scope TEXT NOT NULL,
    -- 'local', 'global', 'meta', 'transcendent'
    attention_focus TEXT DEFAULT '{}',
    -- JSON stored as TEXT
    cognitive_load REAL DEFAULT 0.0,
    emergence_indicators TEXT DEFAULT '{}',
    quantum_coherence REAL DEFAULT 0.0,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    duration_ms INTEGER DEFAULT 0,
    metadata TEXT DEFAULT '{}'
);
-- Decision making and learning patterns
CREATE TABLE IF NOT EXISTS decision_patterns (
    id TEXT PRIMARY KEY DEFAULT (hex(randomblob(16))),
    pattern_id TEXT UNIQUE NOT NULL,
    session_id TEXT REFERENCES cognitive_sessions(session_id),
    decision_type TEXT NOT NULL,
    -- 'rational', 'intuitive', 'creative', 'reactive'
    context_embedding TEXT,
    -- Stored as serialized vector
    decision_tree TEXT NOT NULL,
    -- JSON stored as TEXT
    outcome_prediction TEXT DEFAULT '{}',
    actual_outcome TEXT DEFAULT '{}',
    success_probability REAL DEFAULT 0.5,
    learning_feedback TEXT DEFAULT '{}',
    adaptation_score REAL DEFAULT 0.0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
-- Emergent behavior and complexity analysis
CREATE TABLE IF NOT EXISTS emergence_events (
    id TEXT PRIMARY KEY DEFAULT (hex(randomblob(16))),
    event_id TEXT UNIQUE NOT NULL,
    session_id TEXT REFERENCES cognitive_sessions(session_id),
    emergence_type TEXT NOT NULL,
    -- 'pattern', 'behavior', 'insight', 'capability'
    complexity_score REAL NOT NULL,
    temporal_pattern TEXT DEFAULT '{}',
    cross_domain_correlations TEXT DEFAULT '{}',
    novelty_index REAL DEFAULT 0.0,
    stability_measure REAL DEFAULT 0.0,
    fractal_dimension REAL,
    detection_confidence REAL DEFAULT 0.0,
    evolutionary_pressure TEXT DEFAULT '{}',
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT DEFAULT '{}'
);
-- Social interaction and multi-agent coordination
CREATE TABLE IF NOT EXISTS social_interactions (
    id TEXT PRIMARY KEY DEFAULT (hex(randomblob(16))),
    interaction_id TEXT UNIQUE NOT NULL,
    session_id TEXT REFERENCES cognitive_sessions(session_id),
    agent_id TEXT NOT NULL,
    counterpart_id TEXT NOT NULL,
    interaction_type TEXT NOT NULL,
    -- 'collaboration', 'negotiation', 'learning', 'competition'
    communication_protocol TEXT NOT NULL,
    shared_context TEXT DEFAULT '{}',
    mutual_understanding REAL DEFAULT 0.0,
    trust_level REAL DEFAULT 0.5,
    influence_exchange TEXT DEFAULT '{}',
    outcome_satisfaction REAL DEFAULT 0.5,
    relationship_evolution TEXT DEFAULT '{}',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    ended_at DATETIME,
    metadata TEXT DEFAULT '{}'
);
-- Cognitive workspace and parallel processing
CREATE TABLE IF NOT EXISTS cognitive_workspaces (
    id TEXT PRIMARY KEY DEFAULT (hex(randomblob(16))),
    workspace_id TEXT UNIQUE NOT NULL,
    session_id TEXT REFERENCES cognitive_sessions(session_id),
    workspace_type TEXT NOT NULL,
    -- 'reasoning', 'creative', 'analytical', 'intuitive'
    parallel_processes TEXT DEFAULT '[]',
    -- JSON array as TEXT
    resource_allocation TEXT DEFAULT '{}',
    interference_patterns TEXT DEFAULT '{}',
    synchronization_state TEXT DEFAULT 'independent',
    computational_efficiency REAL DEFAULT 0.0,
    cognitive_bandwidth REAL DEFAULT 0.0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_accessed DATETIME DEFAULT CURRENT_TIMESTAMP,
    status TEXT DEFAULT 'active'
);
-- Quantum cognitive processing (experimental)
CREATE TABLE IF NOT EXISTS quantum_cognitive_states (
    id TEXT PRIMARY KEY DEFAULT (hex(randomblob(16))),
    state_id TEXT UNIQUE NOT NULL,
    session_id TEXT REFERENCES cognitive_sessions(session_id),
    superposition_amplitude REAL DEFAULT 0.0,
    entanglement_degree REAL DEFAULT 0.0,
    decoherence_time_ms INTEGER DEFAULT 0,
    quantum_coherence REAL DEFAULT 0.0,
    measurement_outcome TEXT DEFAULT '{}',
    probability_distribution TEXT DEFAULT '{}',
    quantum_advantage REAL DEFAULT 0.0,
    classical_equivalent_cost REAL DEFAULT 0.0,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT DEFAULT '{}'
);
-- Create performance indexes
CREATE INDEX IF NOT EXISTS idx_consciousness_states_session_id ON consciousness_states(session_id);
CREATE INDEX IF NOT EXISTS idx_consciousness_states_consciousness_level ON consciousness_states(consciousness_level);
CREATE INDEX IF NOT EXISTS idx_consciousness_states_awareness_scope ON consciousness_states(awareness_scope);
CREATE INDEX IF NOT EXISTS idx_consciousness_states_timestamp ON consciousness_states(timestamp);
CREATE INDEX IF NOT EXISTS idx_decision_patterns_session_id ON decision_patterns(session_id);
CREATE INDEX IF NOT EXISTS idx_decision_patterns_decision_type ON decision_patterns(decision_type);
CREATE INDEX IF NOT EXISTS idx_decision_patterns_success_probability ON decision_patterns(success_probability);
CREATE INDEX IF NOT EXISTS idx_decision_patterns_created_at ON decision_patterns(created_at);
CREATE INDEX IF NOT EXISTS idx_emergence_events_session_id ON emergence_events(session_id);
CREATE INDEX IF NOT EXISTS idx_emergence_events_emergence_type ON emergence_events(emergence_type);
CREATE INDEX IF NOT EXISTS idx_emergence_events_complexity_score ON emergence_events(complexity_score);
CREATE INDEX IF NOT EXISTS idx_emergence_events_novelty_index ON emergence_events(novelty_index);
CREATE INDEX IF NOT EXISTS idx_emergence_events_timestamp ON emergence_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_social_interactions_session_id ON social_interactions(session_id);
CREATE INDEX IF NOT EXISTS idx_social_interactions_agent_id ON social_interactions(agent_id);
CREATE INDEX IF NOT EXISTS idx_social_interactions_interaction_type ON social_interactions(interaction_type);
CREATE INDEX IF NOT EXISTS idx_social_interactions_trust_level ON social_interactions(trust_level);
CREATE INDEX IF NOT EXISTS idx_cognitive_workspaces_session_id ON cognitive_workspaces(session_id);
CREATE INDEX IF NOT EXISTS idx_cognitive_workspaces_workspace_type ON cognitive_workspaces(workspace_type);
CREATE INDEX IF NOT EXISTS idx_cognitive_workspaces_status ON cognitive_workspaces(status);
CREATE INDEX IF NOT EXISTS idx_cognitive_workspaces_last_accessed ON cognitive_workspaces(last_accessed);
CREATE INDEX IF NOT EXISTS idx_quantum_cognitive_states_session_id ON quantum_cognitive_states(session_id);
CREATE INDEX IF NOT EXISTS idx_quantum_cognitive_states_quantum_coherence ON quantum_cognitive_states(quantum_coherence);
CREATE INDEX IF NOT EXISTS idx_quantum_cognitive_states_quantum_advantage ON quantum_cognitive_states(quantum_advantage);
CREATE INDEX IF NOT EXISTS idx_quantum_cognitive_states_timestamp ON quantum_cognitive_states(timestamp);
-- Add triggers for automatic timestamp updates
CREATE TRIGGER IF NOT EXISTS update_decision_patterns_updated_at
AFTER
UPDATE ON decision_patterns FOR EACH ROW BEGIN
UPDATE decision_patterns
SET updated_at = CURRENT_TIMESTAMP
WHERE id = NEW.id;
END;
CREATE TRIGGER IF NOT EXISTS update_cognitive_workspaces_last_accessed
AFTER
UPDATE ON cognitive_workspaces FOR EACH ROW BEGIN
UPDATE cognitive_workspaces
SET last_accessed = CURRENT_TIMESTAMP
WHERE id = NEW.id;
END;
-- Create useful views for cognitive analysis (SQLite version)
CREATE VIEW IF NOT EXISTS cognitive_efficiency_summary AS
SELECT session_id,
    COUNT(*) as total_decisions,
    AVG(success_probability) as avg_success_probability,
    AVG(adaptation_score) as avg_adaptation_score,
    CAST(
        COUNT(
            CASE
                WHEN success_probability > 0.7 THEN 1
            END
        ) AS REAL
    ) / COUNT(*) as high_confidence_ratio
FROM decision_patterns
GROUP BY session_id;
CREATE VIEW IF NOT EXISTS emergence_complexity_trends AS
SELECT session_id,
    emergence_type,
    COUNT(*) as event_count,
    AVG(complexity_score) as avg_complexity,
    AVG(novelty_index) as avg_novelty,
    AVG(stability_measure) as avg_stability,
    MAX(complexity_score) as peak_complexity
FROM emergence_events
GROUP BY session_id,
    emergence_type
ORDER BY avg_complexity DESC;
