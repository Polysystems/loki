//! Database Cognitive Tool
//!
//! This tool exposes Loki's existing DatabaseManager as a cognitive tool,
//! enabling AI-driven database operations with safety validation and learning.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use serde_json::{Value};
use tokio::sync::RwLock;
use tracing::{ error, info, warn};

use crate::database::{DatabaseManager, QueryResult, DatabaseBackend};
use crate::memory::CognitiveMemory;
use crate::safety::ActionValidator;

/// Database cognitive tool configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseCognitiveConfig {
    /// Enable query learning and optimization
    pub enable_query_learning: bool,
    
    /// Enable automatic query validation
    pub enable_query_validation: bool,
    
    /// Maximum query execution time in seconds
    pub max_execution_time: u64,
    
    /// Enable query result caching
    pub enable_result_caching: bool,
    
    /// Cache TTL in seconds
    pub cache_ttl: u64,
    
    /// Maximum cache size (number of entries)
    pub max_cache_size: usize,
    
    /// Enable query pattern recognition
    pub enable_pattern_recognition: bool,
    
    /// Enable cognitive query suggestions
    pub enable_query_suggestions: bool,
    
    /// Enable performance monitoring
    pub enable_performance_monitoring: bool,
}

impl Default for DatabaseCognitiveConfig {
    fn default() -> Self {
        Self {
            enable_query_learning: true,
            enable_query_validation: true,
            max_execution_time: 30,
            enable_result_caching: true,
            cache_ttl: 300,
            max_cache_size: 1000,
            enable_pattern_recognition: true,
            enable_query_suggestions: true,
            enable_performance_monitoring: true,
        }
    }
}

/// Query execution context with AI enhancements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryContext {
    /// Original query
    pub query: String,
    
    /// Intended operation (SELECT, INSERT, UPDATE, DELETE, etc.)
    pub operation_type: QueryOperationType,
    
    /// Risk assessment score (0.0 - 1.0)
    pub risk_score: f32,
    
    /// Suggested optimizations
    pub optimizations: Vec<QueryOptimization>,
    
    /// Expected execution time estimate
    pub estimated_execution_time: Duration,
    
    /// Preferred database backend
    pub preferred_backend: Option<DatabaseBackend>,
}

/// Types of database operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueryOperationType {
    Select,
    Insert,
    Update,
    Delete,
    Create,
    Drop,
    Alter,
    Index,
    Unknown,
}

/// Query optimization suggestions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryOptimization {
    /// Type of optimization
    pub optimization_type: OptimizationType,
    
    /// Description of the optimization
    pub description: String,
    
    /// Optimized query (if applicable)
    pub optimized_query: Option<String>,
    
    /// Expected performance improvement
    pub performance_improvement: f32,
}

/// Types of query optimizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationType {
    IndexSuggestion,
    QueryRewrite,
    BackendSelection,
    Caching,
    Batching,
    ParameterBinding,
}

/// Cached query result
#[derive(Debug, Clone)]
struct CachedResult {
    result: QueryResult,
    cached_at: Instant,
    ttl: Duration,
    access_count: usize,
}

/// Query performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryMetrics {
    pub total_queries: usize,
    pub successful_queries: usize,
    pub failed_queries: usize,
    pub avg_execution_time: Duration,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub query_patterns: HashMap<String, usize>,
    pub backend_usage: HashMap<DatabaseBackend, usize>,
}

impl Default for QueryMetrics {
    fn default() -> Self {
        Self {
            total_queries: 0,
            successful_queries: 0,
            failed_queries: 0,
            avg_execution_time: Duration::from_millis(0),
            cache_hits: 0,
            cache_misses: 0,
            query_patterns: HashMap::new(),
            backend_usage: HashMap::new(),
        }
    }
}

/// Database Cognitive Tool - AI-enhanced database operations
pub struct DatabaseCognitiveTool {
    /// Configuration
    config: DatabaseCognitiveConfig,
    
    /// Reference to database manager
    database_manager: Arc<DatabaseManager>,
    
    /// Reference to cognitive memory
    cognitive_memory: Arc<CognitiveMemory>,
    
    /// Safety validator
    safety_validator: Arc<ActionValidator>,
    
    /// Query result cache
    query_cache: Arc<RwLock<HashMap<String, CachedResult>>>,
    
    /// Performance metrics
    metrics: Arc<RwLock<QueryMetrics>>,
    
    /// Learned query patterns
    learned_patterns: Arc<RwLock<HashMap<String, QueryPattern>>>,
}

/// Learned query pattern for optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
struct QueryPattern {
    pattern: String,
    frequency: usize,
    avg_execution_time: Duration,
    optimal_backend: DatabaseBackend,
    last_used: SystemTime,
}

impl DatabaseCognitiveTool {
    /// Create a new database cognitive tool
    pub async fn new(
        config: DatabaseCognitiveConfig,
        database_manager: Arc<DatabaseManager>,
        cognitive_memory: Arc<CognitiveMemory>,
        safety_validator: Arc<ActionValidator>,
    ) -> Result<Self> {
        info!("🧠 Initializing Database Cognitive Tool");
        
        let tool = Self {
            config,
            database_manager,
            cognitive_memory,
            safety_validator,
            query_cache: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(RwLock::new(QueryMetrics::default())),
            learned_patterns: Arc::new(RwLock::new(HashMap::new())),
        };
        
        // Validate database connections
        tool.validate_database_health().await?;
        
        info!("✅ Database Cognitive Tool initialized successfully");
        Ok(tool)
    }
    
    /// Validate database health
    async fn validate_database_health(&self) -> Result<()> {
        let health_status = self.database_manager.health_check().await;
        
        let healthy_backends: Vec<_> = health_status
            .iter()
            .filter(|(_, is_healthy)| **is_healthy)
            .map(|(backend, _)| backend)
            .collect();
        
        if healthy_backends.is_empty() {
            return Err(anyhow!("No healthy database backends available"));
        }
        
        info!("✅ Database health validated. Available backends: {:?}", healthy_backends);
        Ok(())
    }
    
    /// Execute a database query with AI enhancements
    pub async fn execute_query(
        &self,
        query: &str,
        parameters: Option<HashMap<String, Value>>,
    ) -> Result<DatabaseQueryResult> {
        let start_time = Instant::now();
        
        // Analyze and validate query
        let query_context = self.analyze_query(query).await?;
        
        // Safety validation
        if self.config.enable_query_validation {
            self.validate_query_safety(&query_context).await?;
        }
        
        // Check cache if enabled
        if self.config.enable_result_caching {
            let cache_key = self.generate_cache_key(query, &parameters);
            if let Some(cached_result) = self.get_cached_result(&cache_key).await {
                self.update_metrics(|m| m.cache_hits += 1).await;
                return Ok(DatabaseQueryResult {
                    query_result: cached_result.result,
                    query_context,
                    execution_time: cached_result.cached_at.elapsed(),
                    from_cache: true,
                    suggestions: Vec::new(),
                });
            }
            self.update_metrics(|m| m.cache_misses += 1).await;
        }
        
        // Execute query with optimal backend
        let result = if let Some(preferred_backend) = query_context.preferred_backend {
            self.execute_on_backend(query, preferred_backend).await
        } else {
            self.database_manager.execute_smart(query).await
        };
        
        let execution_time = start_time.elapsed();
        
        // Update metrics and learning
        self.update_metrics(|m| {
            m.total_queries += 1;
            if result.is_ok() {
                m.successful_queries += 1;
            } else {
                m.failed_queries += 1;
            }
            
            // Update average execution time
            let total_successful = m.successful_queries;
            if total_successful > 0 {
                m.avg_execution_time = (m.avg_execution_time * (total_successful - 1) as u32 + execution_time) / total_successful as u32;
            }
        }).await;
        
        match result {
            Ok(query_result) => {
                // Cache result if enabled
                if self.config.enable_result_caching {
                    let cache_key = self.generate_cache_key(query, &parameters);
                    self.cache_result(cache_key, query_result.clone()).await;
                }
                
                // Learn from successful query
                if self.config.enable_query_learning {
                    self.learn_from_query(query, &query_result, execution_time).await?;
                }
                
                // Generate suggestions for future optimization
                let suggestions = if self.config.enable_query_suggestions {
                    self.generate_query_suggestions(&query_context, &query_result).await
                } else {
                    Vec::new()
                };
                
                Ok(DatabaseQueryResult {
                    query_result,
                    query_context,
                    execution_time,
                    from_cache: false,
                    suggestions,
                })
            }
            Err(e) => {
                error!("Database query failed: {}", e);
                
                // Try to provide helpful error analysis
                let error_analysis = self.analyze_query_error(query, &e).await;
                
                Err(anyhow!("Query execution failed: {}. Analysis: {}", e, error_analysis))
            }
        }
    }
    
    /// Analyze query and create context
    async fn analyze_query(&self, query: &str) -> Result<QueryContext> {
        let query_lower = query.to_lowercase().trim().to_string();
        
        // Determine operation type
        let operation_type = if query_lower.starts_with("select") {
            QueryOperationType::Select
        } else if query_lower.starts_with("insert") {
            QueryOperationType::Insert
        } else if query_lower.starts_with("update") {
            QueryOperationType::Update
        } else if query_lower.starts_with("delete") {
            QueryOperationType::Delete
        } else if query_lower.starts_with("create") {
            QueryOperationType::Create
        } else if query_lower.starts_with("drop") {
            QueryOperationType::Drop
        } else if query_lower.starts_with("alter") {
            QueryOperationType::Alter
        } else {
            QueryOperationType::Unknown
        };
        
        // Calculate risk score
        let risk_score = self.calculate_risk_score(&query_lower, &operation_type);
        
        // Generate optimizations
        let optimizations = if self.config.enable_pattern_recognition {
            self.generate_optimizations(&query_lower, &operation_type).await
        } else {
            Vec::new()
        };
        
        // Estimate execution time based on learned patterns
        let estimated_execution_time = self.estimate_execution_time(&query_lower).await;
        
        // Select preferred backend
        let preferred_backend = self.select_optimal_backend(&query_lower, &operation_type).await;
        
        Ok(QueryContext {
            query: query.to_string(),
            operation_type,
            risk_score,
            optimizations,
            estimated_execution_time,
            preferred_backend,
        })
    }
    
    /// Calculate risk score for a query
    fn calculate_risk_score(&self, query: &str, operation_type: &QueryOperationType) -> f32 {
        let mut risk_score: f32 = 0.0;
        
        // Base risk by operation type
        match operation_type {
            QueryOperationType::Select => risk_score += 0.1,
            QueryOperationType::Insert => risk_score += 0.3,
            QueryOperationType::Update => risk_score += 0.5,
            QueryOperationType::Delete => risk_score += 0.7,
            QueryOperationType::Drop => risk_score += 0.9,
            QueryOperationType::Create => risk_score += 0.4,
            QueryOperationType::Alter => risk_score += 0.6,
            QueryOperationType::Index => risk_score += 0.2, // Index operations are low risk
            QueryOperationType::Unknown => risk_score += 0.8,
        }
        
        // Additional risk factors
        if query.contains("*") && matches!(operation_type, QueryOperationType::Select) {
            risk_score += 0.2; // SELECT *
        }
        
        if !query.contains("where") && matches!(operation_type, QueryOperationType::Update | QueryOperationType::Delete) {
            risk_score += 0.4; // UPDATE/DELETE without WHERE
        }
        
        if query.contains("drop table") || query.contains("truncate") {
            risk_score += 0.5; // Destructive operations
        }
        
        if query.len() > 1000 {
            risk_score += 0.2; // Complex queries
        }
        
        risk_score.min(1.0)
    }
    
    /// Generate query optimizations
    async fn generate_optimizations(&self, query: &str, operation_type: &QueryOperationType) -> Vec<QueryOptimization> {
        let mut optimizations = Vec::new();
        
        // Index suggestions for SELECT queries
        if matches!(operation_type, QueryOperationType::Select) && query.contains("where") {
            optimizations.push(QueryOptimization {
                optimization_type: OptimizationType::IndexSuggestion,
                description: "Consider adding indexes on WHERE clause columns".to_string(),
                optimized_query: None,
                performance_improvement: 0.3,
            });
        }
        
        // Backend selection optimization
        if query.contains("json") || query.contains("->") {
            optimizations.push(QueryOptimization {
                optimization_type: OptimizationType::BackendSelection,
                description: "PostgreSQL recommended for JSON operations".to_string(),
                optimized_query: None,
                performance_improvement: 0.2,
            });
        }
        
        // Caching suggestion for repeated SELECT queries
        if matches!(operation_type, QueryOperationType::Select) {
            optimizations.push(QueryOptimization {
                optimization_type: OptimizationType::Caching,
                description: "Result caching recommended for this SELECT query".to_string(),
                optimized_query: None,
                performance_improvement: 0.5,
            });
        }
        
        optimizations
    }
    
    /// Estimate query execution time based on learned patterns
    async fn estimate_execution_time(&self, query: &str) -> Duration {
        let patterns = self.learned_patterns.read().await;
        
        // Look for similar patterns
        for (pattern, pattern_data) in patterns.iter() {
            if query.contains(pattern) {
                return pattern_data.avg_execution_time;
            }
        }
        
        // Default estimation based on query complexity
        let base_time = if query.len() < 100 {
            Duration::from_millis(10)
        } else if query.len() < 500 {
            Duration::from_millis(50)
        } else {
            Duration::from_millis(200)
        };
        
        base_time
    }
    
    /// Select optimal backend for query
    async fn select_optimal_backend(&self, query: &str, operation_type: &QueryOperationType) -> Option<DatabaseBackend> {
        // JSON operations prefer PostgreSQL
        if query.contains("json") || query.contains("->") || query.contains("->>") {
            return Some(DatabaseBackend::Postgres);
        }
        
        // Simple operations can use SQLite
        if matches!(operation_type, QueryOperationType::Select) && query.len() < 200 {
            return Some(DatabaseBackend::Sqlite);
        }
        
        // Complex operations prefer PostgreSQL
        if matches!(operation_type, QueryOperationType::Create | QueryOperationType::Alter) {
            return Some(DatabaseBackend::Postgres);
        }
        
        None // Let DatabaseManager decide
    }
    
    /// Validate query safety
    async fn validate_query_safety(&self, context: &QueryContext) -> Result<()> {
        // High-risk queries require additional validation
        if context.risk_score > 0.7 {
            warn!("High-risk query detected: {} (risk: {:.2})", context.query, context.risk_score);
            
            // Use safety validator for destructive operations
            // This would integrate with the existing ActionValidator
            if matches!(context.operation_type, QueryOperationType::Drop | QueryOperationType::Delete) {
                return Err(anyhow!("Destructive operation blocked by safety validator"));
            }
        }
        
        Ok(())
    }
    
    /// Execute query on specific backend
    async fn execute_on_backend(&self, query: &str, backend: DatabaseBackend) -> Result<QueryResult> {
        match backend {
            DatabaseBackend::Postgres => self.database_manager.execute_postgres(query).await
                .map(|rows| QueryResult { rows, backend, duration: Duration::from_millis(0), query: query.to_string() }),
            DatabaseBackend::Sqlite => self.database_manager.execute_sqlite(query).await
                .map(|rows| QueryResult { rows, backend, duration: Duration::from_millis(0), query: query.to_string() }),
            DatabaseBackend::Mysql => self.database_manager.execute_mysql(query).await
                .map(|rows| QueryResult { rows, backend, duration: Duration::from_millis(0), query: query.to_string() }),
        }
    }
    
    /// Generate cache key for query and parameters
    fn generate_cache_key(&self, query: &str, parameters: &Option<HashMap<String, Value>>) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        query.hash(&mut hasher);
        if let Some(params) = parameters {
            for (k, v) in params {
                k.hash(&mut hasher);
                v.to_string().hash(&mut hasher);
            }
        }
        format!("query_{:x}", hasher.finish())
    }
    
    /// Get cached result if valid
    async fn get_cached_result(&self, cache_key: &str) -> Option<CachedResult> {
        let cache = self.query_cache.read().await;
        
        if let Some(cached) = cache.get(cache_key) {
            if cached.cached_at.elapsed() < cached.ttl {
                return Some(cached.clone());
            }
        }
        
        None
    }
    
    /// Cache query result
    async fn cache_result(&self, cache_key: String, result: QueryResult) {
        let mut cache = self.query_cache.write().await;
        
        // Clean up expired entries if cache is full
        if cache.len() >= self.config.max_cache_size {
            cache.retain(|_, cached| cached.cached_at.elapsed() < cached.ttl);
        }
        
        // Add new result to cache
        cache.insert(cache_key, CachedResult {
            result,
            cached_at: Instant::now(),
            ttl: Duration::from_secs(self.config.cache_ttl),
            access_count: 1,
        });
    }
    
    /// Learn from successful query execution
    async fn learn_from_query(&self, query: &str, result: &QueryResult, execution_time: Duration) -> Result<()> {
        if !self.config.enable_query_learning {
            return Ok(());
        }
        
        // Extract pattern from query
        let pattern = self.extract_query_pattern(query);
        
        let mut patterns = self.learned_patterns.write().await;
        
        if let Some(existing_pattern) = patterns.get_mut(&pattern) {
            // Update existing pattern
            existing_pattern.frequency += 1;
            existing_pattern.avg_execution_time = (existing_pattern.avg_execution_time * (existing_pattern.frequency - 1) as u32 + execution_time) / existing_pattern.frequency as u32;
            existing_pattern.optimal_backend = result.backend;
            existing_pattern.last_used = SystemTime::now();
        } else {
            // Create new pattern
            patterns.insert(pattern.clone(), QueryPattern {
                pattern,
                frequency: 1,
                avg_execution_time: execution_time,
                optimal_backend: result.backend,
                last_used: SystemTime::now(),
            });
        }
        
        Ok(())
    }
    
    /// Extract pattern from query for learning
    fn extract_query_pattern(&self, query: &str) -> String {
        let query_lower = query.to_lowercase();
        
        // Simple pattern extraction - replace literals with placeholders
        let pattern = query_lower
            .split_whitespace()
            .map(|word| {
                if word.parse::<i64>().is_ok() {
                    "NUMBER".to_string()
                } else if word.starts_with('\'') && word.ends_with('\'') {
                    "STRING".to_string()
                } else {
                    word.to_string()
                }
            })
            .collect::<Vec<_>>()
            .join(" ");
        
        // Limit pattern length
        if pattern.len() > 100 {
            pattern[..100].to_string()
        } else {
            pattern
        }
    }
    
    /// Generate suggestions for query optimization
    async fn generate_query_suggestions(&self, context: &QueryContext, _result: &QueryResult) -> Vec<String> {
        let mut suggestions = Vec::new();
        
        for optimization in &context.optimizations {
            suggestions.push(format!(
                "{}: {} (Est. improvement: {:.1}%)",
                optimization.optimization_type.as_str(),
                optimization.description,
                optimization.performance_improvement * 100.0
            ));
        }
        
        suggestions
    }
    
    /// Analyze query execution error
    async fn analyze_query_error(&self, query: &str, error: &anyhow::Error) -> String {
        let error_msg = error.to_string().to_lowercase();
        
        if error_msg.contains("syntax error") {
            "SQL syntax error detected. Please check query structure.".to_string()
        } else if error_msg.contains("table") && error_msg.contains("does not exist") {
            "Referenced table does not exist. Please verify table names.".to_string()
        } else if error_msg.contains("column") && error_msg.contains("does not exist") {
            "Referenced column does not exist. Please verify column names.".to_string()
        } else if error_msg.contains("timeout") {
            format!("Query execution timeout. Consider optimization or increasing timeout limit (current: {}s).", self.config.max_execution_time)
        } else if error_msg.contains("permission") || error_msg.contains("denied") {
            "Permission denied. Check database access privileges.".to_string()
        } else {
            format!("Database error: {}. Query: {}", error_msg, query.chars().take(100).collect::<String>())
        }
    }
    
    /// Update performance metrics
    async fn update_metrics<F>(&self, update_fn: F)
    where
        F: FnOnce(&mut QueryMetrics),
    {
        let mut metrics = self.metrics.write().await;
        update_fn(&mut *metrics);
    }
    
    /// Get current performance metrics
    pub async fn get_metrics(&self) -> QueryMetrics {
        self.metrics.read().await.clone()
    }
    
    /// Get database health status
    pub async fn get_database_health(&self) -> HashMap<String, bool> {
        self.database_manager.health_check().await
    }
    
    /// Clear query cache
    pub async fn clear_cache(&self) {
        self.query_cache.write().await.clear();
        info!("🧹 Database query cache cleared");
    }
    
    /// Get available database backends
    pub async fn get_available_backends(&self) -> Vec<DatabaseBackend> {
        self.database_manager.available_backends()
    }
}

/// Enhanced database query result with AI insights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseQueryResult {
    /// The actual query result
    pub query_result: QueryResult,
    
    /// Query analysis context
    pub query_context: QueryContext,
    
    /// Execution time
    pub execution_time: Duration,
    
    /// Whether result came from cache
    pub from_cache: bool,
    
    /// AI-generated suggestions for optimization
    pub suggestions: Vec<String>,
}

impl OptimizationType {
    fn as_str(&self) -> &'static str {
        match self {
            OptimizationType::IndexSuggestion => "Index Suggestion",
            OptimizationType::QueryRewrite => "Query Rewrite",
            OptimizationType::BackendSelection => "Backend Selection",
            OptimizationType::Caching => "Caching",
            OptimizationType::Batching => "Batching",
            OptimizationType::ParameterBinding => "Parameter Binding",
        }
    }
}