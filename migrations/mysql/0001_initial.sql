-- MySQL Initial Migration for Loki Cognitive System
-- Migration: 0001_initial.sql
-- Description: Core database foundation with essential tables for cognitive processing
-- Rust 2025 Edition: InnoDB optimizations, UTF8MB4 support, and performance features
-- Set MySQL session variables for optimal performance
SET sql_mode = 'STRICT_TRANS_TABLES,NO_ZERO_DATE,NO_ZERO_IN_DATE,ERROR_FOR_DIVISION_BY_ZERO';
SET innodb_lock_wait_timeout = 120;
SET foreign_key_checks = 1;
-- Core system configuration table
CREATE TABLE IF NOT EXISTS system_config (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    config_key VARCHAR(255) UNIQUE NOT NULL,
    config_value TEXT NOT NULL,
    config_type ENUM('string', 'integer', 'boolean', 'json', 'float') DEFAULT 'string',
    description TEXT,
    is_secret BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_config_key (config_key),
    INDEX idx_config_type (config_type)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_unicode_ci;
-- Essential configuration entries
INSERT IGNORE INTO system_config (
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
        'mysql_optimization_enabled',
        'true',
        'boolean',
        'MySQL-specific optimizations enabled'
    );
-- Basic cognitive identity and session management
CREATE TABLE IF NOT EXISTS cognitive_identity (
    id CHAR(36) PRIMARY KEY DEFAULT (UUID()),
    identity_name VARCHAR(255) UNIQUE NOT NULL,
    identity_type ENUM(
        'autonomous_agent',
        'human_user',
        'system_process',
        'cognitive_module'
    ) DEFAULT 'autonomous_agent',
    capabilities JSON DEFAULT '{}',
    personality_traits JSON DEFAULT '{}',
    goals JSON DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    INDEX idx_identity_name (identity_name),
    INDEX idx_identity_type (identity_type),
    INDEX idx_is_active (is_active),
    INDEX idx_last_active (last_active)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_unicode_ci;
-- Default Loki identity
INSERT IGNORE INTO cognitive_identity (identity_name, identity_type, capabilities)
VALUES (
        'loki_primary',
        'autonomous_agent',
        JSON_OBJECT(
            'reasoning',
            true,
            'learning',
            true,
            'self_modification',
            true
        )
    );
-- Basic cognitive session tracking (foundation for more complex sessions)
CREATE TABLE IF NOT EXISTS basic_sessions (
    id CHAR(36) PRIMARY KEY DEFAULT (UUID()),
    session_id VARCHAR(255) UNIQUE NOT NULL,
    identity_id CHAR(36),
    session_type ENUM(
        'standard',
        'training',
        'debug',
        'analysis',
        'emergency'
    ) DEFAULT 'standard',
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP NULL,
    status ENUM(
        'active',
        'paused',
        'completed',
        'failed',
        'terminated'
    ) DEFAULT 'active',
    context JSON DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (identity_id) REFERENCES cognitive_identity(id) ON DELETE
    SET NULL,
        INDEX idx_session_id (session_id),
        INDEX idx_identity_id (identity_id),
        INDEX idx_session_type (session_type),
        INDEX idx_status (status),
        INDEX idx_started_at (started_at),
        INDEX idx_created_at (created_at)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_unicode_ci;
-- Basic thought tracking (foundation for complex cognitive thoughts)
CREATE TABLE IF NOT EXISTS basic_thoughts (
    id CHAR(36) PRIMARY KEY DEFAULT (UUID()),
    session_id VARCHAR(255),
    thought_id VARCHAR(255) UNIQUE NOT NULL,
    thought_content TEXT NOT NULL,
    thought_type ENUM(
        'observation',
        'hypothesis',
        'decision',
        'reasoning',
        'memory',
        'goal',
        'plan'
    ) DEFAULT 'observation',
    confidence DECIMAL(5, 4) DEFAULT 0.5000,
    processing_time_ms INT UNSIGNED DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES basic_sessions(session_id) ON DELETE CASCADE,
    INDEX idx_session_id (session_id),
    INDEX idx_thought_id (thought_id),
    INDEX idx_thought_type (thought_type),
    INDEX idx_confidence (confidence),
    INDEX idx_created_at (created_at),
    FULLTEXT idx_thought_content (thought_content)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_unicode_ci;
-- Basic system health and monitoring
CREATE TABLE IF NOT EXISTS system_health (
    id CHAR(36) PRIMARY KEY DEFAULT (UUID()),
    metric_name VARCHAR(255) NOT NULL,
    metric_value DECIMAL(15, 6) NOT NULL,
    unit VARCHAR(50),
    component VARCHAR(255) NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status ENUM('normal', 'warning', 'critical', 'unknown') DEFAULT 'normal',
    metadata JSON DEFAULT '{}',
    INDEX idx_metric_name (metric_name),
    INDEX idx_component (component),
    INDEX idx_timestamp (timestamp),
    INDEX idx_status (status),
    INDEX idx_composite_metric (component, metric_name, timestamp)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_unicode_ci;
-- Basic error logging with enhanced MySQL features
CREATE TABLE IF NOT EXISTS error_log (
    id CHAR(36) PRIMARY KEY DEFAULT (UUID()),
    error_type ENUM('DEBUG', 'INFO', 'WARN', 'ERROR', 'FATAL') NOT NULL,
    error_message TEXT NOT NULL,
    component VARCHAR(255) NOT NULL,
    severity ENUM('low', 'medium', 'high', 'critical') DEFAULT 'medium',
    context JSON DEFAULT '{}',
    stack_trace TEXT NULL,
    resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMP NULL,
    resolved_by VARCHAR(255) NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_error_type (error_type),
    INDEX idx_component (component),
    INDEX idx_severity (severity),
    INDEX idx_resolved (resolved),
    INDEX idx_created_at (created_at),
    INDEX idx_composite_error (component, error_type, resolved),
    FULLTEXT idx_error_message (error_message)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_unicode_ci;
-- MySQL-specific optimization: Create partitioned table for high-frequency metrics
CREATE TABLE IF NOT EXISTS performance_metrics (
    id CHAR(36) DEFAULT (UUID()),
    metric_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    cpu_usage DECIMAL(5, 2),
    memory_usage DECIMAL(5, 2),
    disk_io_rate DECIMAL(10, 2),
    network_io_rate DECIMAL(10, 2),
    cognitive_load DECIMAL(5, 4),
    response_time_ms INT UNSIGNED,
    PRIMARY KEY (id, metric_timestamp),
    INDEX idx_timestamp (metric_timestamp),
    INDEX idx_cpu_usage (cpu_usage),
    INDEX idx_memory_usage (memory_usage),
    INDEX idx_cognitive_load (cognitive_load)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_unicode_ci PARTITION BY RANGE (UNIX_TIMESTAMP(metric_timestamp)) (
    PARTITION p_current
    VALUES LESS THAN (UNIX_TIMESTAMP('2025-01-01')),
        PARTITION p_2025
    VALUES LESS THAN (UNIX_TIMESTAMP('2026-01-01')),
        PARTITION p_2026
    VALUES LESS THAN (UNIX_TIMESTAMP('2027-01-01')),
        PARTITION p_future
    VALUES LESS THAN MAXVALUE
);
-- Create useful views for common queries
CREATE OR REPLACE VIEW active_sessions_view AS
SELECT bs.session_id,
    bs.session_type,
    bs.started_at,
    ci.identity_name,
    ci.identity_type,
    COUNT(bt.id) as thought_count,
    MAX(bt.created_at) as last_thought_at,
    AVG(bt.confidence) as avg_confidence
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
    status
FROM system_health
WHERE timestamp > DATE_SUB(NOW(), INTERVAL 1 HOUR)
GROUP BY component,
    status;
CREATE OR REPLACE VIEW error_summary_view AS
SELECT component,
    error_type,
    severity,
    COUNT(*) as error_count,
    COUNT(
        CASE
            WHEN resolved = TRUE THEN 1
        END
    ) as resolved_count,
    MAX(created_at) as latest_error
FROM error_log
WHERE created_at > DATE_SUB(NOW(), INTERVAL 24 HOUR)
GROUP BY component,
    error_type,
    severity;
-- Create stored procedure for health check
DELIMITER // CREATE PROCEDURE IF NOT EXISTS PerformHealthCheck() BEGIN
DECLARE total_sessions INT DEFAULT 0;
DECLARE active_sessions INT DEFAULT 0;
DECLARE recent_errors INT DEFAULT 0;
SELECT COUNT(*) INTO total_sessions
FROM basic_sessions;
SELECT COUNT(*) INTO active_sessions
FROM basic_sessions
WHERE status = 'active';
SELECT COUNT(*) INTO recent_errors
FROM error_log
WHERE created_at > DATE_SUB(NOW(), INTERVAL 1 HOUR)
    AND error_type IN ('ERROR', 'FATAL');
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
        IF(recent_errors > 10, 'warning', 'normal')
    );
END // DELIMITER;
-- Initialize system health baseline
INSERT IGNORE INTO system_health (
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
        'mysql_engine',
        1.0,
        'boolean',
        'database',
        'normal'
    );
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
        JSON_OBJECT(
            'mysql_version',
            VERSION(),
            'timestamp',
            NOW(),
            'migration',
            '0001_initial'
        )
    );
-- Create event for automated health checks (runs every hour)
SET GLOBAL event_scheduler = ON;
CREATE EVENT IF NOT EXISTS automated_health_check ON SCHEDULE EVERY 1 HOUR STARTS CURRENT_TIMESTAMP DO CALL PerformHealthCheck();
-- Optimize table statistics
ANALYZE TABLE system_config,
cognitive_identity,
basic_sessions,
basic_thoughts,
system_health,
error_log,
performance_metrics;
