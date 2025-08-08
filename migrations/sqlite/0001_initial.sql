-- SQLite Initial Migration for Loki Cognitive System
-- Migration: 0001_initial.sql
-- Description: Core database foundation with essential tables for cognitive processing
-- Rust 2025 Edition: ACID compliance, performance optimizations, and type safety
-- Enable essential SQLite features for cognitive processing
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA cache_size = 10000;
PRAGMA temp_store = MEMORY;
-- Core system configuration table
CREATE TABLE IF NOT EXISTS system_config (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    config_key TEXT UNIQUE NOT NULL,
    config_value TEXT NOT NULL,
    config_type TEXT DEFAULT 'string',
    description TEXT,
    is_secret INTEGER DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
-- Essential configuration entries
INSERT
    OR IGNORE INTO system_config (
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
    );
-- Basic cognitive identity and session management
CREATE TABLE IF NOT EXISTS cognitive_identity (
    id TEXT PRIMARY KEY DEFAULT (hex(randomblob(16))),
    identity_name TEXT UNIQUE NOT NULL,
    identity_type TEXT DEFAULT 'autonomous_agent',
    capabilities TEXT DEFAULT '{}',
    -- JSON
    personality_traits TEXT DEFAULT '{}',
    -- JSON
    goals TEXT DEFAULT '{}',
    -- JSON
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_active DATETIME DEFAULT CURRENT_TIMESTAMP,
    is_active INTEGER DEFAULT 1
);
-- Default Loki identity
INSERT
    OR IGNORE INTO cognitive_identity (identity_name, identity_type, capabilities)
VALUES (
        'loki_primary',
        'autonomous_agent',
        '{"reasoning": true, "learning": true, "self_modification": true}'
    );
-- Basic cognitive session tracking (foundation for more complex sessions)
CREATE TABLE IF NOT EXISTS basic_sessions (
    id TEXT PRIMARY KEY DEFAULT (hex(randomblob(16))),
    session_id TEXT UNIQUE NOT NULL,
    identity_id TEXT REFERENCES cognitive_identity(id),
    session_type TEXT DEFAULT 'standard',
    started_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    ended_at DATETIME,
    status TEXT DEFAULT 'active',
    context TEXT DEFAULT '{}',
    -- JSON
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
-- Basic thought tracking (foundation for complex cognitive thoughts)
CREATE TABLE IF NOT EXISTS basic_thoughts (
    id TEXT PRIMARY KEY DEFAULT (hex(randomblob(16))),
    session_id TEXT REFERENCES basic_sessions(session_id),
    thought_id TEXT UNIQUE NOT NULL,
    thought_content TEXT NOT NULL,
    thought_type TEXT DEFAULT 'observation',
    confidence REAL DEFAULT 0.5,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
-- Basic system health and monitoring
CREATE TABLE IF NOT EXISTS system_health (
    id TEXT PRIMARY KEY DEFAULT (hex(randomblob(16))),
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    unit TEXT,
    component TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    status TEXT DEFAULT 'normal'
);
-- Basic error logging
CREATE TABLE IF NOT EXISTS error_log (
    id TEXT PRIMARY KEY DEFAULT (hex(randomblob(16))),
    error_type TEXT NOT NULL,
    error_message TEXT NOT NULL,
    component TEXT NOT NULL,
    severity TEXT DEFAULT 'info',
    context TEXT DEFAULT '{}',
    -- JSON
    resolved INTEGER DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
-- Create essential indexes for performance
CREATE INDEX IF NOT EXISTS idx_system_config_key ON system_config(config_key);
CREATE INDEX IF NOT EXISTS idx_cognitive_identity_name ON cognitive_identity(identity_name);
CREATE INDEX IF NOT EXISTS idx_basic_sessions_session_id ON basic_sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_basic_sessions_status ON basic_sessions(status);
CREATE INDEX IF NOT EXISTS idx_basic_thoughts_session_id ON basic_thoughts(session_id);
CREATE INDEX IF NOT EXISTS idx_basic_thoughts_type ON basic_thoughts(thought_type);
CREATE INDEX IF NOT EXISTS idx_system_health_component ON system_health(component);
CREATE INDEX IF NOT EXISTS idx_system_health_timestamp ON system_health(timestamp);
CREATE INDEX IF NOT EXISTS idx_error_log_component ON error_log(component);
CREATE INDEX IF NOT EXISTS idx_error_log_severity ON error_log(severity);
CREATE INDEX IF NOT EXISTS idx_error_log_resolved ON error_log(resolved);
-- Create triggers for automatic timestamp updates
CREATE TRIGGER IF NOT EXISTS update_system_config_timestamp
AFTER
UPDATE ON system_config FOR EACH ROW BEGIN
UPDATE system_config
SET updated_at = CURRENT_TIMESTAMP
WHERE id = NEW.id;
END;
CREATE TRIGGER IF NOT EXISTS update_cognitive_identity_active
AFTER
UPDATE ON cognitive_identity FOR EACH ROW BEGIN
UPDATE cognitive_identity
SET last_active = CURRENT_TIMESTAMP
WHERE id = NEW.id;
END;
-- Create views for common queries
CREATE VIEW IF NOT EXISTS active_sessions AS
SELECT bs.*,
    ci.identity_name,
    ci.identity_type,
    (
        SELECT COUNT(*)
        FROM basic_thoughts bt
        WHERE bt.session_id = bs.session_id
    ) as thought_count
FROM basic_sessions bs
    JOIN cognitive_identity ci ON bs.identity_id = ci.id
WHERE bs.status = 'active';
CREATE VIEW IF NOT EXISTS system_status AS
SELECT component,
    COUNT(*) as metric_count,
    AVG(metric_value) as avg_value,
    MAX(timestamp) as last_update
FROM system_health
WHERE timestamp > datetime('now', '-1 hour')
GROUP BY component;
-- Initialize system health baseline
INSERT
    OR IGNORE INTO system_health (metric_name, metric_value, unit, component)
VALUES (
        'initialization_complete',
        1.0,
        'boolean',
        'database'
    ),
    ('schema_version', 0001, 'version', 'database'),
    ('tables_created', 8, 'count', 'database');
-- Log successful initialization
INSERT INTO error_log (error_type, error_message, component, severity)
VALUES (
        'INFO',
        'Database initialized successfully with migration 0001_initial.sql',
        'migration_system',
        'info'
    );
