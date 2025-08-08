//! Comprehensive SQL Database Integration for Loki
//!
//! This module provides advanced database connectivity, query building, and integration
//! with Loki's cognitive architecture for intelligent data management.

use anyhow::{Context, Result};
// use async_trait::async_trait; // Unused import
use serde::{Deserialize, Serialize};
use sqlx::{Pool, Postgres, Sqlite, MySql, Column, TypeInfo, Row};
use std::collections::HashMap;
use deadpool_redis::redis;
use tracing::{info, warn};

/// Database configuration supporting multiple database types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    /// Primary database URL (PostgreSQL for production)
    pub primary_url: String,
    /// SQLite path for local development/caching
    pub sqlite_path: String,
    /// MySQL URL for additional integrations
    pub mysql_url: Option<String>,
    /// Redis URL for caching
    pub redis_url: String,
    /// MongoDB URL for document storage
    pub mongo_url: Option<String>,
    /// Connection pool settings
    pub max_connections: u32,
    /// Connection timeout in seconds
    pub connection_timeout: u64,
    /// Enable query analytics
    pub analytics_enabled: bool,
    /// Enable cognitive query optimization
    pub cognitive_optimization: bool,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            primary_url: std::env::var("DATABASE_URL")
                .unwrap_or_else(|_| "postgresql://loki:password@localhost:5432/loki".to_string()),
            sqlite_path: "./data/loki.db".to_string(),
            mysql_url: std::env::var("MYSQL_URL").ok(),
            redis_url: std::env::var("REDIS_URL")
                .unwrap_or_else(|_| "redis://localhost:6379".to_string()),
            mongo_url: std::env::var("MONGODB_URL").ok(),
            max_connections: 10,
            connection_timeout: 30,
            analytics_enabled: true,
            cognitive_optimization: true,
        }
    }
}

/// Unified database interface supporting multiple backends
pub struct DatabaseManager {
    /// Primary PostgreSQL pool
    postgres_pool: Option<Pool<Postgres>>,
    /// SQLite pool for local operations
    sqlite_pool: Option<Pool<Sqlite>>,
    /// MySQL pool for integrations
    mysql_pool: Option<Pool<MySql>>,
    /// Redis connection for caching
    redis_pool: Option<deadpool_redis::Pool>,
    /// MongoDB client
    mongo_client: Option<mongodb::Client>,
    /// Configuration
    config: DatabaseConfig,
}

impl DatabaseManager {
    /// Create a new database manager
    pub async fn new(config: DatabaseConfig) -> Result<Self> {
        info!("🗄️ Initializing database manager with multi-backend support");

        let mut manager = Self {
            postgres_pool: None,
            sqlite_pool: None,
            mysql_pool: None,
            redis_pool: None,
            mongo_client: None,
            config,
        };

        // Initialize connection pools based on configuration
        
        // Initialize primary database (PostgreSQL)
        if !manager.config.primary_url.is_empty() && manager.config.primary_url != "postgresql://localhost/loki" {
            match manager.init_postgres().await {
                Ok(_) => info!("✅ PostgreSQL initialized"),
                Err(e) => {
                    warn!("⚠️ PostgreSQL initialization failed: {}. Continuing without PostgreSQL.", e);
                }
            }
        }
        
        // Initialize SQLite (always available as fallback)
        match manager.init_sqlite().await {
            Ok(_) => info!("✅ SQLite initialized"),
            Err(e) => {
                warn!("⚠️ SQLite initialization failed: {}. This may impact local caching.", e);
            }
        }
        
        // Initialize MySQL if configured
        if let Some(mysql_url) = &manager.config.mysql_url {
            if !mysql_url.is_empty() && mysql_url != "mysql://localhost/loki" {
                match manager.init_mysql().await {
                    Ok(_) => info!("✅ MySQL initialized"),
                    Err(e) => {
                        warn!("⚠️ MySQL initialization failed: {}. Continuing without MySQL.", e);
                    }
                }
            }
        }
        
        // Initialize Redis if configured
        let redis_url = &manager.config.redis_url;
        if !redis_url.is_empty() && redis_url != "redis://localhost:6379" {
            match manager.init_redis().await {
                Ok(_) => info!("✅ Redis initialized"),
                Err(e) => {
                    warn!("⚠️ Redis initialization failed: {}. Caching will use fallback.", e);
                }
            }
        }
        
        // Initialize MongoDB if configured
        if let Some(mongo_url) = &manager.config.mongo_url {
            if !mongo_url.is_empty() && mongo_url != "mongodb://localhost:27017" {
                match manager.init_mongodb().await {
                    Ok(_) => info!("✅ MongoDB initialized"),
                    Err(e) => {
                        warn!("⚠️ MongoDB initialization failed: {}. Document storage unavailable.", e);
                    }
                }
            }
        }

        info!("✅ Database manager initialized with all configured backends");
        Ok(manager)
    }

    /// Initialize PostgreSQL connection pool
    async fn init_postgres(&mut self) -> Result<()> {
        info!("🐘 Connecting to PostgreSQL: {}", mask_credentials(&self.config.primary_url));

        let options = sqlx::postgres::PgPoolOptions::new()
            .max_connections(self.config.max_connections)
            .acquire_timeout(std::time::Duration::from_secs(self.config.connection_timeout));

        let pool = options
            .connect(&self.config.primary_url)
            .await
            .context("Failed to connect to PostgreSQL")?;

        // Test connection
        sqlx::query("SELECT 1").fetch_one(&pool).await?;

        self.postgres_pool = Some(pool);
        info!("✅ PostgreSQL connected successfully");
        Ok(())
    }

    /// Initialize SQLite connection pool
    async fn init_sqlite(&mut self) -> Result<()> {
        info!("🗃️ Connecting to SQLite: {}", self.config.sqlite_path);

        // Ensure directory exists
        if let Some(parent) = std::path::Path::new(&self.config.sqlite_path).parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        let database_url = format!("sqlite:{}", self.config.sqlite_path);
        let options = sqlx::sqlite::SqlitePoolOptions::new()
            .max_connections(self.config.max_connections)
            .acquire_timeout(std::time::Duration::from_secs(self.config.connection_timeout));

        let pool = options
            .connect(&database_url)
            .await
            .context("Failed to connect to SQLite")?;

        // Test connection
        sqlx::query("SELECT 1").fetch_one(&pool).await?;

        self.sqlite_pool = Some(pool);
        info!("✅ SQLite connected successfully");
        Ok(())
    }

    /// Initialize MySQL connection pool
    async fn init_mysql(&mut self) -> Result<()> {
        if let Some(mysql_url) = &self.config.mysql_url {
            info!("🐬 Connecting to MySQL: {}", mask_credentials(mysql_url));

            let options = sqlx::mysql::MySqlPoolOptions::new()
                .max_connections(self.config.max_connections)
                .acquire_timeout(std::time::Duration::from_secs(self.config.connection_timeout));

            let pool = options
                .connect(mysql_url)
                .await
                .context("Failed to connect to MySQL")?;

            // Test connection
            sqlx::query("SELECT 1").fetch_one(&pool).await?;

            self.mysql_pool = Some(pool);
            info!("✅ MySQL connected successfully");
        }
        Ok(())
    }

    /// Initialize Redis connection pool
    async fn init_redis(&mut self) -> Result<()> {
        info!("🔴 Connecting to Redis: {}", mask_credentials(&self.config.redis_url));

        let redisconfig = deadpool_redis::Config::from_url(&self.config.redis_url);
        let pool = redisconfig
            .create_pool(Some(deadpool_redis::Runtime::Tokio1))
            .context("Failed to create Redis pool")?;

        // Test connection
        let mut conn = pool.get().await.context("Failed to get Redis connection")?;
        let _: String = redis::cmd("PING").query_async::<String>(&mut *conn).await.context("Redis ping failed")?;

        self.redis_pool = Some(pool);
        info!("✅ Redis connected successfully");
        Ok(())
    }

    /// Initialize MongoDB client
    async fn init_mongodb(&mut self) -> Result<()> {
        if let Some(mongo_url) = &self.config.mongo_url {
            info!("🍃 Connecting to MongoDB: {}", mask_credentials(mongo_url));

            let client = mongodb::Client::with_uri_str(mongo_url)
                .await
                .context("Failed to connect to MongoDB")?;

            // Test connection
            client
                .database("admin")
                .run_command(mongodb::bson::doc! {"ping": 1})
                .await
                .context("MongoDB ping failed")?;

            self.mongo_client = Some(client);
            info!("✅ MongoDB connected successfully");
        }
        Ok(())
    }

    /// Execute a query on PostgreSQL
    pub async fn execute_postgres(&self, query: &str) -> Result<Vec<HashMap<String, serde_json::Value>>> {
        if let Some(pool) = &self.postgres_pool {
            info!("executing postgres");
            let rows = sqlx::query(query).fetch_all(pool).await?;
            Ok(rows.into_iter().map(postgres_row_to_map).collect())
        } else {
            Err(anyhow::anyhow!("PostgreSQL not available"))
        }
    }

    /// Execute a query on SQLite
    pub async fn execute_sqlite(&self, query: &str) -> Result<Vec<HashMap<String, serde_json::Value>>> {
        if let Some(pool) = &self.sqlite_pool {
            info!("executing sqlite");
            let rows = sqlx::query(query).fetch_all(pool).await?;
            Ok(rows.into_iter().map(sqlite_row_to_map).collect())
        } else {
            Err(anyhow::anyhow!("SQLite not available"))
        }
    }

    /// Execute a query on MySQL
    pub async fn execute_mysql(&self, query: &str) -> Result<Vec<HashMap<String, serde_json::Value>>> {
        if let Some(pool) = &self.mysql_pool {
            info!("executing mysql");
            let rows = sqlx::query(query).fetch_all(pool).await?;
            Ok(rows.into_iter().map(mysql_row_to_map).collect())
        } else {
            Err(anyhow::anyhow!("MySQL not available"))
        }
    }

    /// Execute a smart query that chooses the optimal backend
    pub async fn execute_smart(&self, query: &str) -> Result<QueryResult> {
        let backend = self.select_optimal_backend(query);
        let start_time = std::time::Instant::now();

        let rows = match backend {
            DatabaseBackend::Postgres => self.execute_postgres(query).await?,
            DatabaseBackend::Sqlite => self.execute_sqlite(query).await?,
            DatabaseBackend::Mysql => self.execute_mysql(query).await?,
        };

        let duration = start_time.elapsed();

        Ok(QueryResult {
            rows,
            backend,
            duration,
            query: query.to_string(),
        })
    }

    /// Health check for all connected databases
    pub async fn health_check(&self) -> HashMap<String, bool> {
        let mut status = HashMap::new();

        // Check PostgreSQL
        if let Some(pool) = &self.postgres_pool {
            status.insert("postgres".to_string(),
                sqlx::query("SELECT 1").fetch_one(pool).await.is_ok()
            );
        }

        // Check SQLite
        if let Some(pool) = &self.sqlite_pool {
            status.insert("sqlite".to_string(),
                sqlx::query("SELECT 1").fetch_one(pool).await.is_ok()
            );
        }

        // Check MySQL
        if let Some(pool) = &self.mysql_pool {
            status.insert("mysql".to_string(),
                sqlx::query("SELECT 1").fetch_one(pool).await.is_ok()
            );
        }

        // Check Redis
        if let Some(pool) = &self.redis_pool {
            let redis_ok = if let Ok(mut conn) = pool.get().await {
                redis::cmd("PING").query_async::<String>(&mut *conn).await.is_ok()
            } else {
                false
            };
            status.insert("redis".to_string(), redis_ok);
        }

        // Check MongoDB
        if let Some(client) = &self.mongo_client {
            let mongo_ok = client
                .database("admin")
                .run_command(mongodb::bson::doc! {"ping": 1})
                .await
                .is_ok();
            status.insert("mongodb".to_string(), mongo_ok);
        }

        status
    }

    /// Get available database backends
    pub fn available_backends(&self) -> Vec<DatabaseBackend> {
        let mut backends = Vec::new();

        if self.postgres_pool.is_some() {
            backends.push(DatabaseBackend::Postgres);
        }
        if self.sqlite_pool.is_some() {
            backends.push(DatabaseBackend::Sqlite);
        }
        if self.mysql_pool.is_some() {
            backends.push(DatabaseBackend::Mysql);
        }

        backends
    }

    /// Select optimal backend for a query
    fn select_optimal_backend(&self, query: &str) -> DatabaseBackend {
        let query_lower = query.to_lowercase();

        // PostgreSQL for JSON queries and complex operations
        if (query_lower.contains("json") || query.contains("->") || query.contains("->>")) && self.postgres_pool.is_some() {
            return DatabaseBackend::Postgres;
        }

        // SQLite for simple queries and local operations
        if (query_lower.contains("memory") || query_lower.contains("cache") || query_lower.contains("temp")) && self.sqlite_pool.is_some() {
            return DatabaseBackend::Sqlite;
        }

        // Default preference: PostgreSQL > SQLite > MySQL
        if self.postgres_pool.is_some() {
            DatabaseBackend::Postgres
        } else if self.sqlite_pool.is_some() {
            DatabaseBackend::Sqlite
        } else {
            DatabaseBackend::Mysql
        }
    }

    /// Get pool references for direct access
    pub fn postgres_pool(&self) -> Option<&Pool<Postgres>> {
        self.postgres_pool.as_ref()
    }

    pub fn sqlite_pool(&self) -> Option<&Pool<Sqlite>> {
        self.sqlite_pool.as_ref()
    }

    pub fn mysql_pool(&self) -> Option<&Pool<MySql>> {
        self.mysql_pool.as_ref()
    }

    pub fn redis_pool(&self) -> Option<&deadpool_redis::Pool> {
        self.redis_pool.as_ref()
    }

    pub fn mongo_client(&self) -> Option<&mongodb::Client> {
        self.mongo_client.as_ref()
    }
}

/// Query execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    pub rows: Vec<HashMap<String, serde_json::Value>>,
    pub backend: DatabaseBackend,
    pub duration: std::time::Duration,
    pub query: String,
}

/// Database backend types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DatabaseBackend {
    Postgres,
    Sqlite,
    Mysql,
}

impl std::fmt::Display for DatabaseBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Postgres => write!(f, "PostgreSQL"),
            Self::Sqlite => write!(f, "SQLite"),
            Self::Mysql => write!(f, "MySQL"),
        }
    }
}

// Helper functions for row conversion
fn postgres_row_to_map(row: sqlx::postgres::PgRow) -> HashMap<String, serde_json::Value> {
    let mut map = HashMap::new();

    for column in row.columns() {
        let name = column.name();
        let value = match column.type_info().name() {
            "TEXT" | "VARCHAR" => {
                if let Ok(val) = row.try_get::<String, _>(name) {
                    serde_json::Value::String(val)
                } else {
                    serde_json::Value::Null
                }
            }
            "INTEGER" | "INT4" => {
                if let Ok(val) = row.try_get::<i32, _>(name) {
                    serde_json::Value::Number(val.into())
                } else {
                    serde_json::Value::Null
                }
            }
            "BIGINT" | "INT8" => {
                if let Ok(val) = row.try_get::<i64, _>(name) {
                    serde_json::Value::Number(val.into())
                } else {
                    serde_json::Value::Null
                }
            }
            "BOOLEAN" => {
                if let Ok(val) = row.try_get::<bool, _>(name) {
                    serde_json::Value::Bool(val)
                } else {
                    serde_json::Value::Null
                }
            }
            "JSON" | "JSONB" => {
                if let Ok(val) = row.try_get::<serde_json::Value, _>(name) {
                    val
                } else {
                    serde_json::Value::Null
                }
            }
            _ => serde_json::Value::Null,
        };
        map.insert(name.to_string(), value);
    }

    map
}

fn sqlite_row_to_map(row: sqlx::sqlite::SqliteRow) -> HashMap<String, serde_json::Value> {
    let mut map = HashMap::new();

    for column in row.columns() {
        let name = column.name();
        let value = match column.type_info().name() {
            "TEXT" => {
                if let Ok(val) = row.try_get::<String, _>(name) {
                    serde_json::Value::String(val)
                } else {
                    serde_json::Value::Null
                }
            }
            "INTEGER" => {
                if let Ok(val) = row.try_get::<i64, _>(name) {
                    serde_json::Value::Number(val.into())
                } else {
                    serde_json::Value::Null
                }
            }
            "REAL" => {
                if let Ok(val) = row.try_get::<f64, _>(name) {
                    if let Some(num) = serde_json::Number::from_f64(val) {
                        serde_json::Value::Number(num)
                    } else {
                        serde_json::Value::Null
                    }
                } else {
                    serde_json::Value::Null
                }
            }
            "BOOLEAN" => {
                if let Ok(val) = row.try_get::<bool, _>(name) {
                    serde_json::Value::Bool(val)
                } else {
                    serde_json::Value::Null
                }
            }
            _ => serde_json::Value::Null,
        };
        map.insert(name.to_string(), value);
    }

    map
}

fn mysql_row_to_map(row: sqlx::mysql::MySqlRow) -> HashMap<String, serde_json::Value> {
    let mut map = HashMap::new();

    for column in row.columns() {
        let name = column.name();
        let value = match column.type_info().name() {
            "VARCHAR" | "TEXT" => {
                if let Ok(val) = row.try_get::<String, _>(name) {
                    serde_json::Value::String(val)
                } else {
                    serde_json::Value::Null
                }
            }
            "INT" | "INTEGER" => {
                if let Ok(val) = row.try_get::<i32, _>(name) {
                    serde_json::Value::Number(val.into())
                } else {
                    serde_json::Value::Null
                }
            }
            "BIGINT" => {
                if let Ok(val) = row.try_get::<i64, _>(name) {
                    serde_json::Value::Number(val.into())
                } else {
                    serde_json::Value::Null
                }
            }
            "BOOLEAN" | "TINYINT" => {
                if let Ok(val) = row.try_get::<bool, _>(name) {
                    serde_json::Value::Bool(val)
                } else {
                    serde_json::Value::Null
                }
            }
            "JSON" => {
                if let Ok(val) = row.try_get::<serde_json::Value, _>(name) {
                    val
                } else {
                    serde_json::Value::Null
                }
            }
            _ => serde_json::Value::Null,
        };
        map.insert(name.to_string(), value);
    }

    map
}

impl DatabaseManager {
    /// Check if a specific database backend is connected
    pub fn is_connected(&self, backend: &str) -> Result<bool> {
        match backend.to_lowercase().as_str() {
            "postgresql" | "postgres" => Ok(self.postgres_pool.is_some()),
            "sqlite" => Ok(self.sqlite_pool.is_some()),
            "mysql" => Ok(self.mysql_pool.is_some()),
            "redis" => Ok(self.redis_pool.is_some()),
            "mongodb" | "mongo" => Ok(self.mongo_client.is_some()),
            _ => Ok(false),
        }
    }
    
    /// Get the number of active connections
    pub async fn active_connections(&self) -> Result<usize> {
        let mut total = 0;
        
        if let Some(pg_pool) = &self.postgres_pool {
            total += pg_pool.size() as usize;
        }
        
        if let Some(sqlite_pool) = &self.sqlite_pool {
            total += sqlite_pool.size() as usize;
        }
        
        if let Some(mysql_pool) = &self.mysql_pool {
            total += mysql_pool.size() as usize;
        }
        
        // Redis connections are managed by deadpool differently
        if let Some(redis_pool) = &self.redis_pool {
            let status = redis_pool.status();
            total += status.size as usize;
        }
        
        Ok(total)
    }
    
    /// Get the maximum number of connections configured
    pub fn max_connections(&self) -> usize {
        self.config.max_connections as usize
    }
}

/// Mask credentials in URL for logging
fn mask_credentials(url: &str) -> String {
    if let Ok(parsed) = url::Url::parse(url) {
        let mut masked = parsed.clone();
        if !parsed.username().is_empty() {
            let _ = masked.set_username("***");
        }
        if parsed.password().is_some() {
            let _ = masked.set_password(Some("***"));
        }
        masked.to_string()
    } else {
        "***invalid_url***".to_string()
    }
}
