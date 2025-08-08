-- SQLite Initial Schema for Loki Cognitive System
-- Migration: 001_initial_schema.sql
-- Description: Basic tables for cognitive operations and system management
-- Enable foreign key support
PRAGMA foreign_keys = ON;
-- Core cognitive session tracking
CREATE TABLE IF NOT EXISTS cognitive_sessions (
    id TEXT PRIMARY KEY DEFAULT (hex(randomblob(16))),
    session_id TEXT UNIQUE NOT NULL,
    started_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    ended_at DATETIME,
    status TEXT DEFAULT 'active',
    context TEXT DEFAULT '{}',
    -- JSON stored as TEXT
    metadata TEXT DEFAULT '{}',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
-- Thought processing and decision tracking
CREATE TABLE IF NOT EXISTS cognitive_thoughts (
    id TEXT PRIMARY KEY DEFAULT (hex(randomblob(16))),
    session_id TEXT REFERENCES cognitive_sessions(session_id),
    thought_id TEXT UNIQUE NOT NULL,
    thought_type TEXT NOT NULL,
    -- 'reasoning', 'decision', 'observation', 'hypothesis'
    content TEXT NOT NULL,
    confidence_score REAL DEFAULT 0.5,
    processing_time_ms INTEGER DEFAULT 0,
    parent_thought_id TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT DEFAULT '{}'
);
-- System performance and health monitoring
CREATE TABLE IF NOT EXISTS system_metrics (
    id TEXT PRIMARY KEY DEFAULT (hex(randomblob(16))),
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    unit TEXT,
    source_component TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT DEFAULT '{}'
);
-- External API interactions and tool usage
CREATE TABLE IF NOT EXISTS api_interactions (
    id TEXT PRIMARY KEY DEFAULT (hex(randomblob(16))),
    interaction_id TEXT UNIQUE NOT NULL,
    provider TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    method TEXT NOT NULL,
    request_data TEXT DEFAULT '{}',
    response_data TEXT DEFAULT '{}',
    status_code INTEGER,
    response_time_ms INTEGER,
    success INTEGER DEFAULT 1,
    -- BOOLEAN as INTEGER
    error_message TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_cognitive_sessions_session_id ON cognitive_sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_cognitive_sessions_status ON cognitive_sessions(status);
CREATE INDEX IF NOT EXISTS idx_cognitive_sessions_created_at ON cognitive_sessions(created_at);
CREATE INDEX IF NOT EXISTS idx_cognitive_thoughts_session_id ON cognitive_thoughts(session_id);
CREATE INDEX IF NOT EXISTS idx_cognitive_thoughts_thought_type ON cognitive_thoughts(thought_type);
CREATE INDEX IF NOT EXISTS idx_cognitive_thoughts_parent_thought_id ON cognitive_thoughts(parent_thought_id);
CREATE INDEX IF NOT EXISTS idx_cognitive_thoughts_created_at ON cognitive_thoughts(created_at);
CREATE INDEX IF NOT EXISTS idx_system_metrics_metric_name ON system_metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_system_metrics_source_component ON system_metrics(source_component);
CREATE INDEX IF NOT EXISTS idx_system_metrics_timestamp ON system_metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_api_interactions_provider ON api_interactions(provider);
CREATE INDEX IF NOT EXISTS idx_api_interactions_success ON api_interactions(success);
CREATE INDEX IF NOT EXISTS idx_api_interactions_created_at ON api_interactions(created_at);
-- Create trigger to automatically update the updated_at column
CREATE TRIGGER IF NOT EXISTS update_cognitive_sessions_updated_at
AFTER
UPDATE ON cognitive_sessions FOR EACH ROW BEGIN
UPDATE cognitive_sessions
SET updated_at = CURRENT_TIMESTAMP
WHERE id = NEW.id;
END;
