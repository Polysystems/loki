-- PostgreSQL Cognitive Processing Tables for Loki
-- Migration: 002_cognitive_tables.sql
-- Description: Advanced cognitive processing, consciousness, and decision-making tables
-- Consciousness orchestration and awareness tracking
CREATE TABLE IF NOT EXISTS consciousness_states (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    state_id VARCHAR(255) UNIQUE NOT NULL,
    session_id VARCHAR(255) REFERENCES cognitive_sessions(session_id),
    consciousness_level DECIMAL(5, 4) DEFAULT 0.5,
    -- 0.0 to 1.0
    awareness_scope VARCHAR(100) NOT NULL,
    -- 'local', 'global', 'meta', 'transcendent'
    attention_focus JSONB DEFAULT '{}',
    -- Current focus areas
    cognitive_load DECIMAL(5, 4) DEFAULT 0.0,
    emergence_indicators JSONB DEFAULT '{}',
    quantum_coherence DECIMAL(5, 4) DEFAULT 0.0,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    duration_ms INTEGER DEFAULT 0,
    metadata JSONB DEFAULT '{}'
);
-- Decision making and learning patterns
CREATE TABLE IF NOT EXISTS decision_patterns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pattern_id VARCHAR(255) UNIQUE NOT NULL,
    session_id VARCHAR(255) REFERENCES cognitive_sessions(session_id),
    decision_type VARCHAR(100) NOT NULL,
    -- 'rational', 'intuitive', 'creative', 'reactive'
    context_embedding VECTOR(768),
    -- For similarity searches (requires pgvector extension)
    decision_tree JSONB NOT NULL,
    outcome_prediction JSONB DEFAULT '{}',
    actual_outcome JSONB DEFAULT '{}',
    success_probability DECIMAL(5, 4) DEFAULT 0.5,
    learning_feedback JSONB DEFAULT '{}',
    adaptation_score DECIMAL(5, 4) DEFAULT 0.0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
-- Emergent behavior and complexity analysis
CREATE TABLE IF NOT EXISTS emergence_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_id VARCHAR(255) UNIQUE NOT NULL,
    session_id VARCHAR(255) REFERENCES cognitive_sessions(session_id),
    emergence_type VARCHAR(100) NOT NULL,
    -- 'pattern', 'behavior', 'insight', 'capability'
    complexity_score DECIMAL(10, 6) NOT NULL,
    temporal_pattern JSONB DEFAULT '{}',
    cross_domain_correlations JSONB DEFAULT '{}',
    novelty_index DECIMAL(5, 4) DEFAULT 0.0,
    stability_measure DECIMAL(5, 4) DEFAULT 0.0,
    fractal_dimension DECIMAL(10, 6),
    detection_confidence DECIMAL(5, 4) DEFAULT 0.0,
    evolutionary_pressure JSONB DEFAULT '{}',
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);
-- Social interaction and multi-agent coordination
CREATE TABLE IF NOT EXISTS social_interactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    interaction_id VARCHAR(255) UNIQUE NOT NULL,
    session_id VARCHAR(255) REFERENCES cognitive_sessions(session_id),
    agent_id VARCHAR(255) NOT NULL,
    counterpart_id VARCHAR(255) NOT NULL,
    interaction_type VARCHAR(100) NOT NULL,
    -- 'collaboration', 'negotiation', 'learning', 'competition'
    communication_protocol VARCHAR(100) NOT NULL,
    shared_context JSONB DEFAULT '{}',
    mutual_understanding DECIMAL(5, 4) DEFAULT 0.0,
    trust_level DECIMAL(5, 4) DEFAULT 0.5,
    influence_exchange JSONB DEFAULT '{}',
    outcome_satisfaction DECIMAL(5, 4) DEFAULT 0.5,
    relationship_evolution JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    ended_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}'
);
-- Cognitive workspace and parallel processing
CREATE TABLE IF NOT EXISTS cognitive_workspaces (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id VARCHAR(255) UNIQUE NOT NULL,
    session_id VARCHAR(255) REFERENCES cognitive_sessions(session_id),
    workspace_type VARCHAR(100) NOT NULL,
    -- 'reasoning', 'creative', 'analytical', 'intuitive'
    parallel_processes JSONB DEFAULT '[]',
    resource_allocation JSONB DEFAULT '{}',
    interference_patterns JSONB DEFAULT '{}',
    synchronization_state VARCHAR(50) DEFAULT 'independent',
    computational_efficiency DECIMAL(5, 4) DEFAULT 0.0,
    cognitive_bandwidth DECIMAL(10, 2) DEFAULT 0.0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_accessed TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    status VARCHAR(50) DEFAULT 'active'
);
-- Quantum cognitive processing (experimental)
CREATE TABLE IF NOT EXISTS quantum_cognitive_states (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    state_id VARCHAR(255) UNIQUE NOT NULL,
    session_id VARCHAR(255) REFERENCES cognitive_sessions(session_id),
    superposition_amplitude DECIMAL(10, 8) DEFAULT 0.0,
    entanglement_degree DECIMAL(5, 4) DEFAULT 0.0,
    decoherence_time_ms INTEGER DEFAULT 0,
    quantum_coherence DECIMAL(5, 4) DEFAULT 0.0,
    measurement_outcome JSONB DEFAULT '{}',
    probability_distribution JSONB DEFAULT '{}',
    quantum_advantage DECIMAL(5, 4) DEFAULT 0.0,
    classical_equivalent_cost DECIMAL(10, 2) DEFAULT 0.0,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
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
CREATE TRIGGER update_decision_patterns_updated_at BEFORE
UPDATE ON decision_patterns FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_cognitive_workspaces_last_accessed BEFORE
UPDATE ON cognitive_workspaces FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
-- Add some useful views for cognitive analysis
CREATE OR REPLACE VIEW cognitive_efficiency_summary AS
SELECT session_id,
    COUNT(*) as total_decisions,
    AVG(success_probability) as avg_success_probability,
    AVG(adaptation_score) as avg_adaptation_score,
    COUNT(
        CASE
            WHEN success_probability > 0.7 THEN 1
        END
    )::DECIMAL / COUNT(*) as high_confidence_ratio
FROM decision_patterns
GROUP BY session_id;
CREATE OR REPLACE VIEW emergence_complexity_trends AS
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
