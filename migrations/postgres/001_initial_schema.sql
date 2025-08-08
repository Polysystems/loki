-- PostgreSQL Initial Schema for Loki Cognitive System
-- Migration: 001_initial_schema.sql
-- Description: Basic tables for cognitive operations and system management
-- Core cognitive session tracking
CREATE TABLE IF NOT EXISTS cognitive_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(255) UNIQUE NOT NULL,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    ended_at TIMESTAMP WITH TIME ZONE,
    status VARCHAR(50) DEFAULT 'active',
    context JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
-- Thought processing and decision tracking
CREATE TABLE IF NOT EXISTS cognitive_thoughts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(255) REFERENCES cognitive_sessions(session_id),
    thought_id VARCHAR(255) UNIQUE NOT NULL,
    thought_type VARCHAR(100) NOT NULL,
    -- 'reasoning', 'decision', 'observation', 'hypothesis'
    content TEXT NOT NULL,
    confidence_score DECIMAL(5, 4) DEFAULT 0.5,
    processing_time_ms INTEGER DEFAULT 0,
    parent_thought_id VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);
-- System performance and health monitoring
CREATE TABLE IF NOT EXISTS system_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    metric_name VARCHAR(255) NOT NULL,
    metric_value DECIMAL(15, 6) NOT NULL,
    unit VARCHAR(50),
    source_component VARCHAR(255) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);
-- External API interactions and tool usage
CREATE TABLE IF NOT EXISTS api_interactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    interaction_id VARCHAR(255) UNIQUE NOT NULL,
    provider VARCHAR(255) NOT NULL,
    endpoint VARCHAR(500) NOT NULL,
    method VARCHAR(10) NOT NULL,
    request_data JSONB DEFAULT '{}',
    response_data JSONB DEFAULT '{}',
    status_code INTEGER,
    response_time_ms INTEGER,
    success BOOLEAN DEFAULT true,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
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
-- Add some useful functions for cognitive operations
CREATE OR REPLACE FUNCTION update_updated_at_column() RETURNS TRIGGER AS $$ BEGIN NEW.updated_at = NOW();
RETURN NEW;
END;
$$ language 'plpgsql';
-- Trigger to automatically update the updated_at column
CREATE TRIGGER update_cognitive_sessions_updated_at BEFORE
UPDATE ON cognitive_sessions FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
