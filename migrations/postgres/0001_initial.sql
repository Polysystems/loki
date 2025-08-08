-- PostgreSQL Initial Migration for Loki Cognitive System
-- Migration: 0001_initial.sql
-- Description: Core database foundation with essential tables for cognitive processing
-- Rust 2025 Edition: Advanced PostgreSQL features, JSON optimization, and performance tuning
-- Enable essential PostgreSQL extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";
-- Set session parameters for optimal performance
SET statement_timeout = '30s';
SET lock_timeout = '10s';
SET idle_in_transaction_session_timeout = '60s';
-- Core system configuration table with JSONB optimization
CREATE TABLE IF NOT EXISTS system_config (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    config_key VARCHAR(255) UNIQUE NOT NULL,
    config_value TEXT NOT NULL,
    config_type VARCHAR(50) DEFAULT 'string' CHECK (
        config_type IN ('string', 'integer', 'boolean', 'json', 'float')
    ),
    description TEXT,
    is_secret BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
-- Essential configuration entries
INSERT INTO system_config (
        config_key,
        config_value,
        config_type,
        description
    )
VALUES (
        'loki_version',
        '0.2.0',
        'string',
        'Loki system version'
    ),
    (
        'database_schema_version',
        '0001',
        'string',
        'Current database schema version'
    ),
    (
        'cognitive_processing_enabled',
        'true',
        'boolean',
        'Enable cognitive processing features'
    ),
    (
        'memory_retention_days',
        '365',
        'integer',
        'Default memory retention period in days'
    ),
    (
        'max_concurrent_thoughts',
        '1000',
        'integer',
        'Maximum concurrent cognitive thoughts'
    ),
    (
        'postgres_optimization_enabled',
        'true',
        'boolean',
        'PostgreSQL-specific optimizations enabled'
    ) ON CONFLICT (config_key) DO NOTHING;
-- Basic cognitive identity and session management with JSONB
CREATE TABLE IF NOT EXISTS cognitive_identity (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    identity_name VARCHAR(255) UNIQUE NOT NULL,
    identity_type VARCHAR(50) DEFAULT 'autonomous_agent' CHECK (
        identity_type IN (
            'autonomous_agent',
            'human_user',
            'system_process',
            'cognitive_module'
        )
    ),
    capabilities JSONB DEFAULT '{}',
    personality_traits JSONB DEFAULT '{}',
    goals JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_active TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE
);
-- GIN index for JSONB capabilities search
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_cognitive_identity_capabilities_gin ON cognitive_identity USING GIN (capabilities);
-- Default Loki identity
INSERT INTO cognitive_identity (identity_name, identity_type, capabilities)
VALUES (
        'loki_primary',
        'autonomous_agent',
        '{"reasoning": true, "learning": true, "self_modification": true}'::jsonb
    ) ON CONFLICT (identity_name) DO NOTHING;
-- Basic cognitive session tracking with advanced indexing
CREATE TABLE IF NOT EXISTS basic_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id VARCHAR(255) UNIQUE NOT NULL,
    identity_id UUID REFERENCES cognitive_identity(id) ON DELETE
    SET NULL,
        session_type VARCHAR(50) DEFAULT 'standard' CHECK (
            session_type IN (
                'standard',
                'training',
                'debug',
                'analysis',
                'emergency'
            )
        ),
        started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        ended_at TIMESTAMP WITH TIME ZONE,
        status VARCHAR(50) DEFAULT 'active' CHECK (
            status IN (
                'active',
                'paused',
                'completed',
                'failed',
                'terminated'
            )
        ),
        context JSONB DEFAULT '{}',
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
-- Partial index for active sessions only (PostgreSQL optimization)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_basic_sessions_active ON basic_sessions (session_id, started_at)
WHERE status = 'active';
-- GIN index for context search
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_basic_sessions_context_gin ON basic_sessions USING GIN (context);
-- Basic thought tracking with full-text search capabilities
CREATE TABLE IF NOT EXISTS basic_thoughts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id VARCHAR(255) REFERENCES basic_sessions(session_id) ON DELETE CASCADE,
    thought_id VARCHAR(255) UNIQUE NOT NULL,
    thought_content TEXT NOT NULL,
    thought_type VARCHAR(50) DEFAULT 'observation' CHECK (
        thought_type IN (
            'observation',
            'hypothesis',
            'decision',
            'reasoning',
            'memory',
            'goal',
            'plan'
        )
    ),
    confidence DECIMAL(5, 4) DEFAULT 0.5000 CHECK (
        confidence >= 0.0
        AND confidence <= 1.0
    ),
    processing_time_ms INTEGER DEFAULT 0 CHECK (processing_time_ms >= 0),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    -- Full-text search vector for PostgreSQL
    search_vector tsvector GENERATED ALWAYS AS (to_tsvector('english', thought_content)) STORED
);
-- GIN index for full-text search
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_basic_thoughts_search ON basic_thoughts USING GIN (search_vector);
-- GIN index for metadata search
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_basic_thoughts_metadata_gin ON basic_thoughts USING GIN (metadata);
-- Basic system health and monitoring with time-series optimization
CREATE TABLE IF NOT EXISTS system_health (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name VARCHAR(255) NOT NULL,
    metric_value DECIMAL(15, 6) NOT NULL,
    unit VARCHAR(50),
    component VARCHAR(255) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    status VARCHAR(50) DEFAULT 'normal' CHECK (
        status IN ('normal', 'warning', 'critical', 'unknown')
    ),
    metadata JSONB DEFAULT '{}'
);
-- Composite index for time-series queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_system_health_timeseries ON system_health (component, metric_name, timestamp DESC);
-- Partial indexes for different status levels
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_system_health_warnings ON system_health (timestamp DESC)
WHERE status IN ('warning', 'critical');
-- Basic error logging with advanced PostgreSQL features
CREATE TABLE IF NOT EXISTS error_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    error_type VARCHAR(50) NOT NULL CHECK (
        error_type IN ('DEBUG', 'INFO', 'WARN', 'ERROR', 'FATAL')
    ),
    error_message TEXT NOT NULL,
    component VARCHAR(255) NOT NULL,
    severity VARCHAR(50) DEFAULT 'medium' CHECK (
        severity IN ('low', 'medium', 'high', 'critical')
    ),
    context JSONB DEFAULT '{}',
    stack_trace TEXT,
    resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMP WITH TIME ZONE,
    resolved_by VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    -- Full-text search for error messages
    search_vector tsvector GENERATED ALWAYS AS (
        to_tsvector(
            'english',
            error_message || ' ' || COALESCE(stack_trace, '')
        )
    ) STORED
);
-- Indexes for error analysis
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_error_log_search ON error_log USING GIN (search_vector);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_error_log_analysis ON error_log (component, error_type, severity, created_at DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_error_log_unresolved ON error_log (created_at DESC)
WHERE resolved = FALSE;
-- PostgreSQL-specific: Partitioned table for high-frequency performance metrics
CREATE TABLE IF NOT EXISTS performance_metrics (
    id UUID DEFAULT uuid_generate_v4(),
    metric_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    cpu_usage DECIMAL(5, 2) CHECK (
        cpu_usage >= 0.0
        AND cpu_usage <= 100.0
    ),
    memory_usage DECIMAL(5, 2) CHECK (
        memory_usage >= 0.0
        AND memory_usage <= 100.0
    ),
    disk_io_rate DECIMAL(10, 2) CHECK (disk_io_rate >= 0.0),
    network_io_rate DECIMAL(10, 2) CHECK (network_io_rate >= 0.0),
    cognitive_load DECIMAL(5, 4) CHECK (
        cognitive_load >= 0.0
        AND cognitive_load <= 1.0
    ),
    response_time_ms INTEGER CHECK (response_time_ms >= 0),
    PRIMARY KEY (id, metric_timestamp)
) PARTITION BY RANGE (metric_timestamp);
-- Create partitions for the next few years
CREATE TABLE IF NOT EXISTS performance_metrics_2024 PARTITION OF performance_metrics FOR
VALUES
FROM ('2024-01-01') TO ('2025-01-01');
CREATE TABLE IF NOT EXISTS performance_metrics_2025 PARTITION OF performance_metrics FOR
VALUES
FROM ('2025-01-01') TO ('2026-01-01');
CREATE TABLE IF NOT EXISTS performance_metrics_2026 PARTITION OF performance_metrics FOR
VALUES
FROM ('2026-01-01') TO ('2027-01-01');
-- Create useful views for common queries
CREATE OR REPLACE VIEW active_sessions_view AS
SELECT bs.session_id,
    bs.session_type,
    bs.started_at,
    ci.identity_name,
    ci.identity_type,
    COUNT(bt.id) as thought_count,
    MAX(bt.created_at) as last_thought_at,
    AVG(bt.confidence) as avg_confidence,
    EXTRACT(
        EPOCH
        FROM (NOW() - bs.started_at)
    ) as session_duration_seconds
FROM basic_sessions bs
    JOIN cognitive_identity ci ON bs.identity_id = ci.id
    LEFT JOIN basic_thoughts bt ON bs.session_id = bt.session_id
WHERE bs.status = 'active'
GROUP BY bs.session_id,
    bs.session_type,
    bs.started_at,
    ci.identity_name,
    ci.identity_type;
CREATE OR REPLACE VIEW system_status_view AS
SELECT component,
    COUNT(*) as metric_count,
    AVG(metric_value) as avg_value,
    MIN(metric_value) as min_value,
    MAX(metric_value) as max_value,
    MAX(timestamp) as last_update,
    status,
    PERCENTILE_CONT(0.95) WITHIN GROUP (
        ORDER BY metric_value
    ) as p95_value
FROM system_health
WHERE timestamp > NOW() - INTERVAL '1 hour'
GROUP BY component,
    status;
CREATE OR REPLACE VIEW error_summary_view AS
SELECT component,
    error_type,
    severity,
    COUNT(*) as error_count,
    COUNT(*) FILTER (
        WHERE resolved = TRUE
    ) as resolved_count,
    MAX(created_at) as latest_error,
    AVG(
        EXTRACT(
            EPOCH
            FROM (resolved_at - created_at)
        )
    ) FILTER (
        WHERE resolved = TRUE
    ) as avg_resolution_time_seconds
FROM error_log
WHERE created_at > NOW() - INTERVAL '24 hours'
GROUP BY component,
    error_type,
    severity;
-- Create PostgreSQL-specific stored procedures for health monitoring
CREATE OR REPLACE FUNCTION perform_health_check() RETURNS TABLE(
        component_name TEXT,
        metric_name TEXT,
        metric_value DECIMAL,
        status TEXT
    ) AS $$
DECLARE total_sessions INTEGER;
active_sessions INTEGER;
recent_errors INTEGER;
avg_response_time DECIMAL;
BEGIN -- Gather metrics
SELECT COUNT(*) INTO total_sessions
FROM basic_sessions;
SELECT COUNT(*) INTO active_sessions
FROM basic_sessions
WHERE status = 'active';
SELECT COUNT(*) INTO recent_errors
FROM error_log
WHERE created_at > NOW() - INTERVAL '1 hour'
    AND error_type IN ('ERROR', 'FATAL');
SELECT AVG(response_time_ms) INTO avg_response_time
FROM performance_metrics
WHERE metric_timestamp > NOW() - INTERVAL '1 hour';
-- Insert health metrics
INSERT INTO system_health (
        metric_name,
        metric_value,
        unit,
        component,
        status
    )
VALUES (
        'total_sessions',
        total_sessions,
        'count',
        'session_manager',
        'normal'
    ),
    (
        'active_sessions',
        active_sessions,
        'count',
        'session_manager',
        'normal'
    ),
    (
        'recent_errors',
        recent_errors,
        'count',
        'error_tracking',
        CASE
            WHEN recent_errors > 10 THEN 'warning'
            ELSE 'normal'
        END
    ),
    (
        'avg_response_time',
        COALESCE(avg_response_time, 0),
        'milliseconds',
        'performance',
        CASE
            WHEN COALESCE(avg_response_time, 0) > 1000 THEN 'warning'
            ELSE 'normal'
        END
    );
-- Return current status
RETURN QUERY
SELECT 'session_manager'::TEXT,
    'total_sessions'::TEXT,
    total_sessions::DECIMAL,
    'normal'::TEXT
UNION ALL
SELECT 'session_manager'::TEXT,
    'active_sessions'::TEXT,
    active_sessions::DECIMAL,
    'normal'::TEXT
UNION ALL
SELECT 'error_tracking'::TEXT,
    'recent_errors'::TEXT,
    recent_errors::DECIMAL,
    CASE
        WHEN recent_errors > 10 THEN 'warning'
        ELSE 'normal'
    END;
END;
$$ LANGUAGE plpgsql;
-- Create automatic update triggers
CREATE OR REPLACE FUNCTION update_updated_at_column() RETURNS TRIGGER AS $$ BEGIN NEW.updated_at = NOW();
RETURN NEW;
END;
$$ LANGUAGE plpgsql;
CREATE TRIGGER update_system_config_updated_at BEFORE
UPDATE ON system_config FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE OR REPLACE FUNCTION update_last_active_column() RETURNS TRIGGER AS $$ BEGIN NEW.last_active = NOW();
RETURN NEW;
END;
$$ LANGUAGE plpgsql;
CREATE TRIGGER update_cognitive_identity_last_active BEFORE
UPDATE ON cognitive_identity FOR EACH ROW EXECUTE FUNCTION update_last_active_column();
-- Initialize system health baseline
INSERT INTO system_health (
        metric_name,
        metric_value,
        unit,
        component,
        status
    )
VALUES (
        'initialization_complete',
        1.0,
        'boolean',
        'database',
        'normal'
    ),
    (
        'schema_version',
        0001,
        'version',
        'database',
        'normal'
    ),
    (
        'tables_created',
        6,
        'count',
        'database',
        'normal'
    ),
    (
        'partitions_created',
        4,
        'count',
        'database',
        'normal'
    ),
    (
        'postgres_extensions_enabled',
        3,
        'count',
        'database',
        'normal'
    ) ON CONFLICT DO NOTHING;
-- Log successful initialization
INSERT INTO error_log (
        error_type,
        error_message,
        component,
        severity,
        context
    )
VALUES (
        'INFO',
        'Database initialized successfully with migration 0001_initial.sql',
        'migration_system',
        'low',
        jsonb_build_object(
            'postgres_version',
            version(),
            'timestamp',
            NOW(),
            'migration',
            '0001_initial',
            'extensions',
            ARRAY ['uuid-ossp', 'pg_trgm', 'btree_gin']
        )
    );
-- Analyze tables for optimal query planning
ANALYZE system_config,
cognitive_identity,
basic_sessions,
basic_thoughts,
system_health,
error_log;
-- Create helpful comments for schema documentation
COMMENT ON TABLE system_config IS 'Core system configuration with JSONB optimization for dynamic settings';
COMMENT ON TABLE cognitive_identity IS 'Cognitive identity management with JSONB capabilities and personality traits';
COMMENT ON TABLE basic_sessions IS 'Session tracking with partial indexing for active sessions';
COMMENT ON TABLE basic_thoughts IS 'Thought tracking with full-text search and metadata indexing';
COMMENT ON TABLE system_health IS 'System health monitoring with time-series optimization';
COMMENT ON TABLE error_log IS 'Error logging with full-text search and resolution tracking';
COMMENT ON TABLE performance_metrics IS 'High-frequency performance metrics with partitioning';
-- Set up row-level security (optional, can be enabled later)
-- ALTER TABLE cognitive_identity ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE basic_sessions ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE basic_thoughts ENABLE ROW LEVEL SECURITY;
