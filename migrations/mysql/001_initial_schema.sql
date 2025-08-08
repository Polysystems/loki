-- MySQL Initial Schema for Loki Cognitive System
-- Migration: 001_initial_schema.sql
-- Description: Basic tables for cognitive operations and system management
-- Core cognitive session tracking
CREATE TABLE IF NOT EXISTS cognitive_sessions (
    id VARCHAR(36) PRIMARY KEY DEFAULT (UUID()),
    session_id VARCHAR(255) UNIQUE NOT NULL,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP NULL,
    status VARCHAR(50) DEFAULT 'active',
    context JSON,
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_unicode_ci;
-- Thought processing and decision tracking
CREATE TABLE IF NOT EXISTS cognitive_thoughts (
    id VARCHAR(36) PRIMARY KEY DEFAULT (UUID()),
    session_id VARCHAR(255),
    thought_id VARCHAR(255) UNIQUE NOT NULL,
    thought_type VARCHAR(100) NOT NULL,
    -- 'reasoning', 'decision', 'observation', 'hypothesis'
    content TEXT NOT NULL,
    confidence_score DECIMAL(5, 4) DEFAULT 0.5,
    processing_time_ms INT DEFAULT 0,
    parent_thought_id VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSON,
    FOREIGN KEY (session_id) REFERENCES cognitive_sessions(session_id)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_unicode_ci;
-- System performance and health monitoring
CREATE TABLE IF NOT EXISTS system_metrics (
    id VARCHAR(36) PRIMARY KEY DEFAULT (UUID()),
    metric_name VARCHAR(255) NOT NULL,
    metric_value DECIMAL(15, 6) NOT NULL,
    unit VARCHAR(50),
    source_component VARCHAR(255) NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSON
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_unicode_ci;
-- External API interactions and tool usage
CREATE TABLE IF NOT EXISTS api_interactions (
    id VARCHAR(36) PRIMARY KEY DEFAULT (UUID()),
    interaction_id VARCHAR(255) UNIQUE NOT NULL,
    provider VARCHAR(255) NOT NULL,
    endpoint VARCHAR(500) NOT NULL,
    method VARCHAR(10) NOT NULL,
    request_data JSON,
    response_data JSON,
    status_code INT,
    response_time_ms INT,
    success BOOLEAN DEFAULT TRUE,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_unicode_ci;
-- Create indexes for better performance
CREATE INDEX idx_cognitive_sessions_session_id ON cognitive_sessions(session_id);
CREATE INDEX idx_cognitive_sessions_status ON cognitive_sessions(status);
CREATE INDEX idx_cognitive_sessions_created_at ON cognitive_sessions(created_at);
CREATE INDEX idx_cognitive_thoughts_session_id ON cognitive_thoughts(session_id);
CREATE INDEX idx_cognitive_thoughts_thought_type ON cognitive_thoughts(thought_type);
CREATE INDEX idx_cognitive_thoughts_parent_thought_id ON cognitive_thoughts(parent_thought_id);
CREATE INDEX idx_cognitive_thoughts_created_at ON cognitive_thoughts(created_at);
CREATE INDEX idx_system_metrics_metric_name ON system_metrics(metric_name);
CREATE INDEX idx_system_metrics_source_component ON system_metrics(source_component);
CREATE INDEX idx_system_metrics_timestamp ON system_metrics(timestamp);
CREATE INDEX idx_api_interactions_provider ON api_interactions(provider);
CREATE INDEX idx_api_interactions_success ON api_interactions(success);
CREATE INDEX idx_api_interactions_created_at ON api_interactions(created_at);
