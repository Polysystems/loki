//! Universal API Interface Tool
//!
//! This tool provides user-configurable connections to any REST/GraphQL APIs
//! with dynamic schema discovery, authentication management, and cognitive learning.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use anyhow::{Context, Result, anyhow};
use reqwest::{Client, Method, header::{HeaderMap, HeaderName, HeaderValue}};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::RwLock;
use tracing::{info, warn};
use url::Url;

use crate::memory::CognitiveMemory;
use crate::safety::ActionValidator;

/// API connector configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiConnectorConfig {
    /// Default request timeout in seconds
    pub default_timeout: u64,
    
    /// Enable automatic schema discovery
    pub enable_schema_discovery: bool,
    
    /// Enable request/response caching
    pub enable_caching: bool,
    
    /// Cache TTL in seconds
    pub cache_ttl: u64,
    
    /// Maximum cache size (number of entries)
    pub max_cache_size: usize,
    
    /// Enable rate limiting
    pub enable_rate_limiting: bool,
    
    /// Default rate limit (requests per minute)
    pub default_rate_limit: u32,
    
    /// Enable retry logic
    pub enable_retries: bool,
    
    /// Maximum retry attempts
    pub max_retries: u32,
    
    /// Enable cognitive learning from API interactions
    pub enable_api_learning: bool,
    
    /// Enable performance monitoring
    pub enable_performance_monitoring: bool,
}

impl Default for ApiConnectorConfig {
    fn default() -> Self {
        Self {
            default_timeout: 30,
            enable_schema_discovery: true,
            enable_caching: true,
            cache_ttl: 300,
            max_cache_size: 1000,
            enable_rate_limiting: true,
            default_rate_limit: 60,
            enable_retries: true,
            max_retries: 3,
            enable_api_learning: true,
            enable_performance_monitoring: true,
        }
    }
}

/// API endpoint configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiEndpointConfig {
    /// Unique identifier for this API
    pub api_id: String,
    
    /// Human-readable name
    pub name: String,
    
    /// Base URL of the API
    pub base_url: String,
    
    /// API type (REST, GraphQL, SOAP, etc.)
    pub api_type: ApiType,
    
    /// Authentication configuration
    pub auth: Option<AuthConfig>,
    
    /// Default headers
    pub default_headers: HashMap<String, String>,
    
    /// API schema (if known)
    pub schema: Option<ApiSchema>,
    
    /// Rate limiting configuration
    pub rate_limit: Option<RateLimitConfig>,
    
    /// Custom timeout for this API
    pub timeout: Option<u64>,
    
    /// API-specific configuration
    pub custom_config: HashMap<String, Value>,
}

/// Types of APIs supported
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ApiType {
    Rest,
    GraphQL,
    Soap,
    JsonRpc,
    Custom(String),
}

/// Authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthConfig {
    /// Authentication type
    pub auth_type: AuthType,
    
    /// Authentication credentials
    pub credentials: AuthCredentials,
    
    /// Token refresh settings (for OAuth)
    pub refresh_config: Option<RefreshConfig>,
}

/// Authentication types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuthType {
    None,
    ApiKey,
    BearerToken,
    BasicAuth,
    OAuth2,
    Custom(String),
}

/// Authentication credentials
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthCredentials {
    /// API key or token
    pub token: Option<String>,
    
    /// Username (for basic auth)
    pub username: Option<String>,
    
    /// Password (for basic auth)
    pub password: Option<String>,
    
    /// Client ID (for OAuth)
    pub client_id: Option<String>,
    
    /// Client secret (for OAuth)
    pub client_secret: Option<String>,
    
    /// OAuth scopes
    pub scopes: Option<Vec<String>>,
    
    /// Custom authentication parameters
    pub custom_params: HashMap<String, String>,
}

/// Token refresh configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefreshConfig {
    /// Refresh token
    pub refresh_token: String,
    
    /// Token endpoint URL
    pub token_endpoint: String,
    
    /// Auto-refresh before expiry
    pub auto_refresh: bool,
    
    /// Refresh threshold (seconds before expiry)
    pub refresh_threshold: u64,
}

/// API schema information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiSchema {
    /// Schema version
    pub version: String,
    
    /// Available endpoints
    pub endpoints: HashMap<String, EndpointSchema>,
    
    /// Common data models
    pub models: HashMap<String, ModelSchema>,
}

/// Endpoint schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndpointSchema {
    /// HTTP method
    pub method: String,
    
    /// Endpoint path
    pub path: String,
    
    /// Description
    pub description: Option<String>,
    
    /// Parameters
    pub parameters: Vec<ParameterSchema>,
    
    /// Request body schema
    pub request_body: Option<ModelSchema>,
    
    /// Response schema
    pub response: Option<ModelSchema>,
}

/// Parameter schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterSchema {
    /// Parameter name
    pub name: String,
    
    /// Parameter type
    pub param_type: String,
    
    /// Parameter location (query, path, header, body)
    pub location: ParameterLocation,
    
    /// Whether parameter is required
    pub required: bool,
    
    /// Description
    pub description: Option<String>,
    
    /// Default value
    pub default: Option<Value>,
}

/// Parameter locations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterLocation {
    Query,
    Path,
    Header,
    Body,
}

/// Data model schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSchema {
    /// Model name
    pub name: String,
    
    /// Model fields
    pub fields: HashMap<String, FieldSchema>,
    
    /// Model description
    pub description: Option<String>,
}

/// Field schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldSchema {
    /// Field type
    pub field_type: String,
    
    /// Whether field is required
    pub required: bool,
    
    /// Field description
    pub description: Option<String>,
    
    /// Field format/constraints
    pub format: Option<String>,
}

/// Rate limiting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    /// Requests per minute
    pub requests_per_minute: u32,
    
    /// Burst limit
    pub burst_limit: u32,
    
    /// Rate limit reset interval
    pub reset_interval: Duration,
}

/// API request context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiRequestContext {
    /// Request ID
    pub request_id: String,
    
    /// API endpoint ID
    pub api_id: String,
    
    /// Endpoint path
    pub endpoint: String,
    
    /// HTTP method
    pub method: HttpMethod,
    
    /// Query parameters
    pub query_params: HashMap<String, String>,
    
    /// Path parameters
    pub path_params: HashMap<String, String>,
    
    /// Request headers
    pub headers: HashMap<String, String>,
    
    /// Request body
    pub body: Option<Value>,
    
    /// Expected response format
    pub expected_response: Option<String>,
}

/// HTTP methods
#[derive(Debug, Clone, Serialize, Deserialize, Hash)]
pub enum HttpMethod {
    Get,
    Post,
    Put,
    Patch,
    Delete,
    Head,
    Options,
}

impl From<HttpMethod> for Method {
    fn from(method: HttpMethod) -> Self {
        match method {
            HttpMethod::Get => Method::GET,
            HttpMethod::Post => Method::POST,
            HttpMethod::Put => Method::PUT,
            HttpMethod::Patch => Method::PATCH,
            HttpMethod::Delete => Method::DELETE,
            HttpMethod::Head => Method::HEAD,
            HttpMethod::Options => Method::OPTIONS,
        }
    }
}

/// API response result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiResponse {
    /// Request ID
    pub request_id: String,
    
    /// HTTP status code
    pub status_code: u16,
    
    /// Response headers
    pub headers: HashMap<String, String>,
    
    /// Response body
    pub body: Value,
    
    /// Response time
    pub response_time: Duration,
    
    /// Whether response came from cache
    pub from_cache: bool,
    
    /// Parsed response (if schema available)
    pub parsed_response: Option<Value>,
    
    /// Success status
    pub success: bool,
    
    /// Error message (if any)
    pub error: Option<String>,
}

/// Rate limiting state
#[derive(Debug, Clone)]
struct RateLimitState {
    requests_made: u32,
    window_start: Instant,
    last_request: Instant,
}

/// Cached API response
#[derive(Debug, Clone)]
struct CachedResponse {
    response: ApiResponse,
    cached_at: Instant,
    ttl: Duration,
    access_count: usize,
}

/// API usage metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiMetrics {
    pub total_requests: usize,
    pub successful_requests: usize,
    pub failed_requests: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub avg_response_time: Duration,
    pub rate_limit_hits: usize,
    pub api_usage: HashMap<String, usize>,
    pub endpoint_usage: HashMap<String, usize>,
}

impl Default for ApiMetrics {
    fn default() -> Self {
        Self {
            total_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            cache_hits: 0,
            cache_misses: 0,
            avg_response_time: Duration::from_millis(0),
            rate_limit_hits: 0,
            api_usage: HashMap::new(),
            endpoint_usage: HashMap::new(),
        }
    }
}

/// Universal API Interface Tool
pub struct ApiConnectorTool {
    /// Configuration
    config: ApiConnectorConfig,
    
    /// HTTP client
    client: Client,
    
    /// Reference to cognitive memory
    cognitive_memory: Arc<CognitiveMemory>,
    
    /// Safety validator
    safety_validator: Arc<ActionValidator>,
    
    /// Configured API endpoints
    api_endpoints: Arc<RwLock<HashMap<String, ApiEndpointConfig>>>,
    
    /// Response cache
    response_cache: Arc<RwLock<HashMap<String, CachedResponse>>>,
    
    /// Rate limiting state
    rate_limits: Arc<RwLock<HashMap<String, RateLimitState>>>,
    
    /// Performance metrics
    metrics: Arc<RwLock<ApiMetrics>>,
    
    /// Learned API patterns
    learned_patterns: Arc<RwLock<HashMap<String, ApiPattern>>>,
}

/// Learned API usage pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ApiPattern {
    api_id: String,
    endpoint: String,
    frequency: usize,
    avg_response_time: Duration,
    success_rate: f32,
    common_parameters: HashMap<String, String>,
    last_used: SystemTime,
}

impl ApiConnectorTool {
    /// Create a new API connector tool
    pub async fn new(
        config: ApiConnectorConfig,
        cognitive_memory: Arc<CognitiveMemory>,
        safety_validator: Arc<ActionValidator>,
    ) -> Result<Self> {
        info!("🌐 Initializing Universal API Interface Tool");
        
        let client = Client::builder()
            .timeout(Duration::from_secs(config.default_timeout))
            .build()
            .context("Failed to create HTTP client")?;
        
        let tool = Self {
            config,
            client,
            cognitive_memory,
            safety_validator,
            api_endpoints: Arc::new(RwLock::new(HashMap::new())),
            response_cache: Arc::new(RwLock::new(HashMap::new())),
            rate_limits: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(RwLock::new(ApiMetrics::default())),
            learned_patterns: Arc::new(RwLock::new(HashMap::new())),
        };
        
        info!("✅ Universal API Interface Tool initialized successfully");
        Ok(tool)
    }
    
    /// Register a new API endpoint configuration
    pub async fn register_api(&self, api_config: ApiEndpointConfig) -> Result<()> {
        info!("📝 Registering API: {} ({})", api_config.name, api_config.api_id);
        
        // Validate API configuration
        self.validate_api_config(&api_config).await?;
        
        // Attempt schema discovery if enabled
        let mut config = api_config;
        if self.config.enable_schema_discovery && config.schema.is_none() {
            match self.discover_api_schema(&config).await {
                Ok(schema) => {
                    config.schema = Some(schema);
                    info!("🔍 API schema discovered for: {}", config.api_id);
                }
                Err(e) => {
                    warn!("⚠️  Failed to discover schema for {}: {}", config.api_id, e);
                }
            }
        }
        
        // Store API configuration
        self.api_endpoints.write().await.insert(config.api_id.clone(), config);
        
        info!("✅ API registered successfully");
        Ok(())
    }
    
    /// Execute API request with full context and error handling
    pub async fn execute_api_request(&self, context: ApiRequestContext) -> Result<ApiResponse> {
        let start_time = Instant::now();
        
        // Validate request
        self.validate_api_request(&context).await?;
        
        // Check rate limiting
        if self.config.enable_rate_limiting {
            self.check_rate_limit(&context.api_id).await?;
        }
        
        // Check cache if enabled
        if self.config.enable_caching {
            let cache_key = self.generate_cache_key(&context);
            if let Some(cached_response) = self.get_cached_response(&cache_key).await {
                self.update_metrics(|m| m.cache_hits += 1).await;
                return Ok(cached_response.response);
            }
            self.update_metrics(|m| m.cache_misses += 1).await;
        }
        
        // Execute request with retries
        let response = if self.config.enable_retries {
            self.execute_with_retries(&context).await?
        } else {
            self.execute_single_request(&context).await?
        };
        
        // Update rate limiting
        if self.config.enable_rate_limiting {
            self.update_rate_limit(&context.api_id).await;
        }
        
        // Cache response if enabled and successful
        if self.config.enable_caching && response.success {
            let cache_key = self.generate_cache_key(&context);
            self.cache_response(cache_key, response.clone()).await;
        }
        
        // Update metrics and learning
        let response_time = start_time.elapsed();
        self.update_metrics(|m| {
            m.total_requests += 1;
            if response.success {
                m.successful_requests += 1;
            } else {
                m.failed_requests += 1;
            }
            
            // Update average response time
            let total = m.total_requests;
            if total > 0 {
                m.avg_response_time = (m.avg_response_time * (total - 1) as u32 + response_time) / total as u32;
            }
            
            // Update API and endpoint usage
            *m.api_usage.entry(context.api_id.clone()).or_insert(0) += 1;
            *m.endpoint_usage.entry(format!("{}:{}", context.api_id, context.endpoint)).or_insert(0) += 1;
        }).await;
        
        // Learn from successful requests
        if self.config.enable_api_learning && response.success {
            self.learn_from_request(&context, &response).await?;
        }
        
        Ok(response)
    }
    
    /// Validate API configuration
    async fn validate_api_config(&self, config: &ApiEndpointConfig) -> Result<()> {
        // Validate base URL
        Url::parse(&config.base_url)
            .context("Invalid base URL")?;
        
        // Validate authentication if present
        if let Some(auth) = &config.auth {
            match &auth.auth_type {
                AuthType::ApiKey => {
                    if auth.credentials.token.is_none() {
                        return Err(anyhow!("API key authentication requires a token"));
                    }
                }
                AuthType::BearerToken => {
                    if auth.credentials.token.is_none() {
                        return Err(anyhow!("Bearer token authentication requires a token"));
                    }
                }
                AuthType::BasicAuth => {
                    if auth.credentials.username.is_none() || auth.credentials.password.is_none() {
                        return Err(anyhow!("Basic authentication requires username and password"));
                    }
                }
                AuthType::OAuth2 => {
                    if auth.credentials.client_id.is_none() || auth.credentials.client_secret.is_none() {
                        return Err(anyhow!("OAuth2 requires client ID and secret"));
                    }
                }
                _ => {} // Other auth types
            }
        }
        
        Ok(())
    }
    
    /// Discover API schema through various methods
    async fn discover_api_schema(&self, config: &ApiEndpointConfig) -> Result<ApiSchema> {
        match config.api_type {
            ApiType::Rest => self.discover_openapi_schema(config).await,
            ApiType::GraphQL => self.discover_graphql_schema(config).await,
            _ => {
                // Return basic schema for other types
                Ok(ApiSchema {
                    version: "1.0".to_string(),
                    endpoints: HashMap::new(),
                    models: HashMap::new(),
                })
            }
        }
    }
    
    /// Discover OpenAPI/Swagger schema
    async fn discover_openapi_schema(&self, config: &ApiEndpointConfig) -> Result<ApiSchema> {
        // Try common OpenAPI documentation endpoints
        let openapi_urls = vec![
            format!("{}/openapi.json", config.base_url),
            format!("{}/swagger.json", config.base_url),
            format!("{}/v1/openapi.json", config.base_url),
            format!("{}/api-docs", config.base_url),
        ];
        
        for url in openapi_urls {
            if let Ok(response) = self.client.get(&url).send().await {
                if response.status().is_success() {
                    if let Ok(schema_json) = response.json::<Value>().await {
                        return self.parse_openapi_schema(schema_json);
                    }
                }
            }
        }
        
        // Return empty schema if discovery fails
        Ok(ApiSchema {
            version: "unknown".to_string(),
            endpoints: HashMap::new(),
            models: HashMap::new(),
        })
    }
    
    /// Discover GraphQL schema through introspection
    async fn discover_graphql_schema(&self, config: &ApiEndpointConfig) -> Result<ApiSchema> {
        let introspection_query = json!({
            "query": "query IntrospectionQuery { __schema { queryType { name } mutationType { name } subscriptionType { name } types { name kind } } }"
        });
        
        let response = self.client
            .post(&format!("{}/graphql", config.base_url))
            .json(&introspection_query)
            .send()
            .await?;
        
        if response.status().is_success() {
            let schema_json = response.json::<Value>().await?;
            self.parse_graphql_schema(schema_json)
        } else {
            Ok(ApiSchema {
                version: "unknown".to_string(),
                endpoints: HashMap::new(),
                models: HashMap::new(),
            })
        }
    }
    
    /// Parse OpenAPI schema JSON
    fn parse_openapi_schema(&self, schema_json: Value) -> Result<ApiSchema> {
        let mut endpoints = HashMap::new();
        let models = HashMap::new();
        
        let version = schema_json.get("openapi")
            .or_else(|| schema_json.get("swagger"))
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();
        
        // Parse paths
        if let Some(paths) = schema_json.get("paths") {
            if let Some(paths_obj) = paths.as_object() {
                for (path, path_info) in paths_obj {
                    if let Some(path_obj) = path_info.as_object() {
                        for (method, method_info) in path_obj {
                            if method != "parameters" { // Skip parameters at path level
                                let endpoint_id = format!("{}_{}", method.to_uppercase(), path);
                                
                                let description = method_info.get("summary")
                                    .or_else(|| method_info.get("description"))
                                    .and_then(|v| v.as_str())
                                    .map(|s| s.to_string());
                                
                                endpoints.insert(endpoint_id, EndpointSchema {
                                    method: method.to_uppercase(),
                                    path: path.clone(),
                                    description,
                                    parameters: Vec::new(), // Would parse parameters from schema
                                    request_body: None,     // Would parse request body schema
                                    response: None,         // Would parse response schema
                                });
                            }
                        }
                    }
                }
            }
        }
        
        Ok(ApiSchema {
            version,
            endpoints,
            models,
        })
    }
    
    /// Parse GraphQL schema JSON
    fn parse_graphql_schema(&self, schema_json: Value) -> Result<ApiSchema> {
        let mut endpoints = HashMap::new();
        let models = HashMap::new();
        
        // Extract GraphQL types and create basic endpoint mappings
        if let Some(data) = schema_json.get("data") {
            if let Some(schema) = data.get("__schema") {
                if let Some(query_type) = schema.get("queryType") {
                    if let Some(name) = query_type.get("name").and_then(|v| v.as_str()) {
                        endpoints.insert("query".to_string(), EndpointSchema {
                            method: "POST".to_string(),
                            path: "/graphql".to_string(),
                            description: Some(format!("GraphQL Query Type: {}", name)),
                            parameters: Vec::new(),
                            request_body: None,
                            response: None,
                        });
                    }
                }
                
                if let Some(mutation_type) = schema.get("mutationType") {
                    if let Some(name) = mutation_type.get("name").and_then(|v| v.as_str()) {
                        endpoints.insert("mutation".to_string(), EndpointSchema {
                            method: "POST".to_string(),
                            path: "/graphql".to_string(),
                            description: Some(format!("GraphQL Mutation Type: {}", name)),
                            parameters: Vec::new(),
                            request_body: None,
                            response: None,
                        });
                    }
                }
            }
        }
        
        Ok(ApiSchema {
            version: "GraphQL".to_string(),
            endpoints,
            models,
        })
    }
    
    /// Validate API request context
    async fn validate_api_request(&self, context: &ApiRequestContext) -> Result<()> {
        // Check if API is registered
        let apis = self.api_endpoints.read().await;
        if !apis.contains_key(&context.api_id) {
            return Err(anyhow!("Unknown API: {}", context.api_id));
        }
        
        // Additional validation would go here
        // - Schema validation
        // - Parameter validation
        // - Security checks
        
        Ok(())
    }
    
    /// Check rate limiting for API
    async fn check_rate_limit(&self, api_id: &str) -> Result<()> {
        let mut rate_limits = self.rate_limits.write().await;
        let now = Instant::now();
        
        // Get API config for rate limit settings
        let apis = self.api_endpoints.read().await;
        let api_config = apis.get(api_id).ok_or_else(|| anyhow!("Unknown API: {}", api_id))?;
        
        let limit = api_config.rate_limit.as_ref()
            .map(|rl| rl.requests_per_minute)
            .unwrap_or(self.config.default_rate_limit);
        
        let state = rate_limits.entry(api_id.to_string()).or_insert_with(|| RateLimitState {
            requests_made: 0,
            window_start: now,
            last_request: now,
        });
        
        // Check if we need to reset the window
        if now.duration_since(state.window_start) >= Duration::from_secs(60) {
            state.requests_made = 0;
            state.window_start = now;
        }
        
        // Check rate limit
        if state.requests_made >= limit {
            self.update_metrics(|m| m.rate_limit_hits += 1).await;
            return Err(anyhow!("Rate limit exceeded for API: {}", api_id));
        }
        
        Ok(())
    }
    
    /// Update rate limiting state
    async fn update_rate_limit(&self, api_id: &str) {
        let mut rate_limits = self.rate_limits.write().await;
        let now = Instant::now();
        
        if let Some(state) = rate_limits.get_mut(api_id) {
            state.requests_made += 1;
            state.last_request = now;
        }
    }
    
    /// Execute request with retry logic
    async fn execute_with_retries(&self, context: &ApiRequestContext) -> Result<ApiResponse> {
        let mut last_error = None;
        
        for attempt in 1..=self.config.max_retries {
            match self.execute_single_request(context).await {
                Ok(response) => {
                    if response.success || attempt == self.config.max_retries {
                        return Ok(response);
                    }
                    // Retry on failure unless it's the last attempt
                    last_error = response.error.clone();
                }
                Err(e) => {
                    last_error = Some(e.to_string());
                    if attempt == self.config.max_retries {
                        return Err(e);
                    }
                }
            }
            
            // Exponential backoff
            let delay = Duration::from_millis(100 * (2_u64.pow(attempt - 1)));
            tokio::time::sleep(delay).await;
        }
        
        Err(anyhow!("Request failed after {} retries. Last error: {:?}", 
            self.config.max_retries, last_error))
    }
    
    /// Execute single API request
    async fn execute_single_request(&self, context: &ApiRequestContext) -> Result<ApiResponse> {
        let start_time = Instant::now();
        
        // Get API configuration
        let apis = self.api_endpoints.read().await;
        let api_config = apis.get(&context.api_id)
            .ok_or_else(|| anyhow!("Unknown API: {}", context.api_id))?
            .clone();
        drop(apis);
        
        // Build request URL
        let mut url = format!("{}{}", api_config.base_url.trim_end_matches('/'), context.endpoint);
        
        // Replace path parameters
        for (key, value) in &context.path_params {
            url = url.replace(&format!("{{{}}}", key), value);
        }
        
        // Build request
        let method: Method = context.method.clone().into();
        let mut request = self.client.request(method, &url);
        
        // Add query parameters
        if !context.query_params.is_empty() {
            request = request.query(&context.query_params);
        }
        
        // Add headers
        let mut headers = HeaderMap::new();
        
        // Add default headers from API config
        for (key, value) in &api_config.default_headers {
            if let (Ok(name), Ok(val)) = (HeaderName::from_bytes(key.as_bytes()), HeaderValue::from_str(value)) {
                headers.insert(name, val);
            }
        }
        
        // Add request-specific headers
        for (key, value) in &context.headers {
            if let (Ok(name), Ok(val)) = (HeaderName::from_bytes(key.as_bytes()), HeaderValue::from_str(value)) {
                headers.insert(name, val);
            }
        }
        
        // Add authentication headers
        if let Some(auth) = &api_config.auth {
            self.add_auth_headers(&mut headers, auth)?;
        }
        
        request = request.headers(headers);
        
        // Add request body if present
        if let Some(body) = &context.body {
            request = request.json(body);
        }
        
        // Set custom timeout if specified
        if let Some(timeout) = api_config.timeout {
            request = request.timeout(Duration::from_secs(timeout));
        }
        
        // Execute request
        let response = request.send().await
            .context("Failed to execute HTTP request")?;
        
        let status_code = response.status().as_u16();
        let success = response.status().is_success();
        
        // Extract response headers
        let response_headers: HashMap<String, String> = response.headers()
            .iter()
            .map(|(name, value)| {
                (name.to_string(), value.to_str().unwrap_or("").to_string())
            })
            .collect();
        
        // Parse response body
        let body_text = response.text().await
            .context("Failed to read response body")?;
        
        let body: Value = if body_text.is_empty() {
            Value::Null
        } else {
            serde_json::from_str(&body_text).unwrap_or_else(|_| Value::String(body_text))
        };
        
        let response_time = start_time.elapsed();
        
        let api_response = ApiResponse {
            request_id: context.request_id.clone(),
            status_code,
            headers: response_headers,
            body,
            response_time,
            from_cache: false,
            parsed_response: None, // Would implement based on schema
            success,
            error: if success { None } else { Some(format!("HTTP {}", status_code)) },
        };
        
        Ok(api_response)
    }
    
    /// Add authentication headers
    fn add_auth_headers(&self, headers: &mut HeaderMap, auth: &AuthConfig) -> Result<()> {
        match &auth.auth_type {
            AuthType::None => {}
            AuthType::ApiKey => {
                if let Some(token) = &auth.credentials.token {
                    headers.insert("X-API-Key", HeaderValue::from_str(token)?);
                }
            }
            AuthType::BearerToken => {
                if let Some(token) = &auth.credentials.token {
                    let bearer = format!("Bearer {}", token);
                    headers.insert("Authorization", HeaderValue::from_str(&bearer)?);
                }
            }
            AuthType::BasicAuth => {
                if let (Some(username), Some(password)) = (&auth.credentials.username, &auth.credentials.password) {
                    let credentials = encode_base64(&format!("{}:{}", username, password));
                    let basic = format!("Basic {}", credentials);
                    headers.insert("Authorization", HeaderValue::from_str(&basic)?);
                }
            }
            AuthType::OAuth2 => {
                if let Some(token) = &auth.credentials.token {
                    let bearer = format!("Bearer {}", token);
                    headers.insert("Authorization", HeaderValue::from_str(&bearer)?);
                }
            }
            AuthType::Custom(header_name) => {
                if let Some(token) = &auth.credentials.token {
                    let header_name = HeaderName::from_bytes(header_name.as_bytes())?;
                    headers.insert(header_name, HeaderValue::from_str(token)?);
                }
            }
        }
        
        Ok(())
    }
    
    /// Generate cache key for request
    fn generate_cache_key(&self, context: &ApiRequestContext) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        context.api_id.hash(&mut hasher);
        context.endpoint.hash(&mut hasher);
        context.method.hash(&mut hasher);
        
        // Hash query parameters by sorting keys and hashing key-value pairs
        let mut query_pairs: Vec<_> = context.query_params.iter().collect();
        query_pairs.sort_by_key(|&(k, _)| k);
        for (key, value) in query_pairs {
            key.hash(&mut hasher);
            value.hash(&mut hasher);
        }
        
        format!("api_{:x}", hasher.finish())
    }
    
    /// Get cached response if valid
    async fn get_cached_response(&self, cache_key: &str) -> Option<CachedResponse> {
        let cache = self.response_cache.read().await;
        
        if let Some(cached) = cache.get(cache_key) {
            if cached.cached_at.elapsed() < cached.ttl {
                return Some(cached.clone());
            }
        }
        
        None
    }
    
    /// Cache API response
    async fn cache_response(&self, cache_key: String, response: ApiResponse) {
        let mut cache = self.response_cache.write().await;
        
        // Clean up expired entries if cache is getting large
        if cache.len() >= self.config.max_cache_size {
            cache.retain(|_, cached| cached.cached_at.elapsed() < cached.ttl);
        }
        
        cache.insert(cache_key, CachedResponse {
            response,
            cached_at: Instant::now(),
            ttl: Duration::from_secs(self.config.cache_ttl),
            access_count: 1,
        });
    }
    
    /// Learn from successful API request
    async fn learn_from_request(&self, context: &ApiRequestContext, response: &ApiResponse) -> Result<()> {
        if !self.config.enable_api_learning {
            return Ok(());
        }
        
        let pattern_key = format!("{}:{}", context.api_id, context.endpoint);
        let mut patterns = self.learned_patterns.write().await;
        
        if let Some(existing_pattern) = patterns.get_mut(&pattern_key) {
            // Update existing pattern
            existing_pattern.frequency += 1;
            existing_pattern.avg_response_time = (existing_pattern.avg_response_time * (existing_pattern.frequency - 1) as u32 + response.response_time) / existing_pattern.frequency as u32;
            existing_pattern.success_rate = (existing_pattern.success_rate * (existing_pattern.frequency - 1) as f32 + if response.success { 1.0 } else { 0.0 }) / existing_pattern.frequency as f32;
            existing_pattern.last_used = SystemTime::now();
            
            // Update common parameters
            for (key, value) in &context.query_params {
                existing_pattern.common_parameters.insert(key.clone(), value.clone());
            }
        } else {
            // Create new pattern
            patterns.insert(pattern_key.clone(), ApiPattern {
                api_id: context.api_id.clone(),
                endpoint: context.endpoint.clone(),
                frequency: 1,
                avg_response_time: response.response_time,
                success_rate: if response.success { 1.0 } else { 0.0 },
                common_parameters: context.query_params.clone(),
                last_used: SystemTime::now(),
            });
        }
        
        Ok(())
    }
    
    /// Update performance metrics
    async fn update_metrics<F>(&self, update_fn: F)
    where
        F: FnOnce(&mut ApiMetrics),
    {
        let mut metrics = self.metrics.write().await;
        update_fn(&mut *metrics);
    }
    
    /// Get current performance metrics
    pub async fn get_metrics(&self) -> ApiMetrics {
        self.metrics.read().await.clone()
    }
    
    /// Get registered APIs
    pub async fn get_registered_apis(&self) -> HashMap<String, ApiEndpointConfig> {
        self.api_endpoints.read().await.clone()
    }
    
    /// Get learned API patterns
    pub async fn get_learned_patterns(&self) -> HashMap<String, ApiPattern> {
        self.learned_patterns.read().await.clone()
    }
    
    /// Clear response cache
    pub async fn clear_cache(&self) {
        self.response_cache.write().await.clear();
        info!("🧹 API response cache cleared");
    }
    
    /// Remove API configuration
    pub async fn unregister_api(&self, api_id: &str) -> Result<()> {
        self.api_endpoints.write().await.remove(api_id);
        self.rate_limits.write().await.remove(api_id);
        info!("🗑️  API unregistered: {}", api_id);
        Ok(())
    }
}

// Simple base64 encoding for basic auth
fn encode_base64(input: &str) -> String {
    let chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let char_map: Vec<char> = chars.chars().collect();
    
    let bytes = input.as_bytes();
    let mut result = String::new();
    
    for chunk in bytes.chunks(3) {
        let mut buf = [0u8; 3];
        for (i, &byte) in chunk.iter().enumerate() {
            buf[i] = byte;
        }
        
        let b = ((buf[0] as u32) << 16) | ((buf[1] as u32) << 8) | (buf[2] as u32);
        
        result.push(char_map[((b >> 18) & 0x3F) as usize]);
        result.push(char_map[((b >> 12) & 0x3F) as usize]);
        
        if chunk.len() > 1 {
            result.push(char_map[((b >> 6) & 0x3F) as usize]);
        } else {
            result.push('=');
        }
        
        if chunk.len() > 2 {
            result.push(char_map[(b & 0x3F) as usize]);
        } else {
            result.push('=');
        }
    }
    
    result
}