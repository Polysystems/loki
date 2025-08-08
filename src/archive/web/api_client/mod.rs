mod duration_serde;
mod status_code_serde;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context as AnyhowContext, Result};
use parking_lot::RwLock;
use rand::prelude::*;
use rand::distributions::Alphanumeric;
use reqwest::{Client, Method, StatusCode};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::{debug, info, warn};

use crate::memory::{CognitiveMemory, MemoryMetadata};

/// API endpoint configuration
#[derive(Debug, Clone)]
pub struct ApiEndpoint {
    pub name: String,
    pub base_url: String,
    pub auth: AuthMethod,
    pub headers: HashMap<String, String>,
    pub rate_limit: Option<RateLimit>,
    pub timeout: Duration,
}

#[derive(Debug, Clone)]
pub enum AuthMethod {
    None,
    ApiKey {
        header: String,
        key: String,
    },
    Bearer {
        token: String,
    },
    Basic {
        username: String,
        password: String,
    },
    OAuth2 {
        client_id: String,
        client_secret: String,
        token_url: String,
        access_token: Option<String>,
        refresh_token: Option<String>,
        expires_at: Option<chrono::DateTime<chrono::Utc>>,
        scopes: Vec<String>,
    },
}

#[derive(Debug, Clone)]
pub struct RateLimit {
    pub requests_per_second: f32,
    pub burst_size: usize,
}

/// OAuth2 token response structure (RFC 6749)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OAuth2TokenResponse {
    pub access_token: String,
    pub token_type: String,
    pub expires_in: Option<u64>,
    pub refresh_token: Option<String>,
    pub scope: Option<String>,
}

/// OAuth2 error response structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OAuth2ErrorResponse {
    pub error: String,
    pub error_description: Option<String>,
    pub error_uri: Option<String>,
}

/// API response wrapper
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiResponse {
    #[serde(with = "status_code_serde")]
    pub status: StatusCode,
    pub headers: HashMap<String, String>,
    pub body: Value,
    #[serde(with = "duration_serde")]
    pub request_duration: Duration,
}

/// Generic API client with memory integration
pub struct ApiClient {
    /// HTTP client
    client: Client,

    /// Configured endpoints
    endpoints: Arc<RwLock<HashMap<String, ApiEndpoint>>>,

    /// Memory system
    memory: Arc<CognitiveMemory>,

    /// Request history
    request_history: Arc<RwLock<Vec<RequestRecord>>>,

    /// Rate limiters per endpoint
    rate_limiters: Arc<RwLock<HashMap<String, TokenBucket>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RequestRecord {
    endpoint: String,
    method: String,
    path: String,
    timestamp: chrono::DateTime<chrono::Utc>,
    duration: Duration,
    status: u16,
    cached: bool,
}

impl ApiClient {
    /// Create a new API client
    pub async fn new(memory: Arc<CognitiveMemory>) -> Result<Self> {
        info!("Initializing API client");

        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .user_agent("Loki/1.0 (The Shapeshifter)")
            .build()?;

        Ok(Self {
            client,
            endpoints: Arc::new(RwLock::new(HashMap::new())),
            memory,
            request_history: Arc::new(RwLock::new(Vec::with_capacity(1000))),
            rate_limiters: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Register a new API endpoint
    pub fn register_endpoint(&self, endpoint: ApiEndpoint) {
        let name = endpoint.name.clone();
        let endpoint_name = endpoint.name.clone();

        // Create rate limiter if needed
        if let Some(rate_limit) = &endpoint.rate_limit {
            let bucket = TokenBucket::new(rate_limit.burst_size, rate_limit.requests_per_second);
            self.rate_limiters.write().insert(name.clone(), bucket);
        }

        self.endpoints.write().insert(name, endpoint);
        info!("Registered API endpoint: {}", endpoint_name);
    }

    /// Make a GET request
    pub async fn get(&self, endpoint_name: &str, path: &str) -> Result<ApiResponse> {
        self.request(endpoint_name, Method::GET, path, None).await
    }

    /// Make a POST request
    pub async fn post(
        &self,
        endpoint_name: &str,
        path: &str,
        body: Option<Value>,
    ) -> Result<ApiResponse> {
        self.request(endpoint_name, Method::POST, path, body).await
    }

    /// Make a generic request
    pub async fn request(
        &self,
        endpoint_name: &str,
        method: Method,
        path: &str,
        body: Option<Value>,
    ) -> Result<ApiResponse> {
        // Get endpoint configuration
        let endpoint = self
            .endpoints
            .read()
            .get(endpoint_name)
            .cloned()
            .context(format!("Unknown endpoint: {}", endpoint_name))?;

        // Check cache first
        let cache_key = format!("api:{}:{}:{}", endpoint_name, method, path);
        if method == Method::GET {
            if let Ok(Some(cached)) = self.memory.retrieve_by_key(&cache_key).await {
                if let Ok(response) = serde_json::from_str::<ApiResponse>(&cached.content) {
                    debug!("Using cached API response");
                    self.record_request(
                        endpoint_name,
                        &method,
                        path,
                        Duration::ZERO,
                        response.status.as_u16(),
                        true,
                    );
                    return Ok(response);
                }
            }
        }

        // Check rate limit
        if endpoint.rate_limit.is_some() {
            self.wait_for_rate_limit(endpoint_name).await?;
        }

        // Build request
        let url = format!("{}{}", endpoint.base_url, path);
        let mut request = self.client.request(method.clone(), &url);

        // Add authentication
        request = match &endpoint.auth {
            AuthMethod::None => request,
            AuthMethod::ApiKey { header, key } => request.header(header, key),
            AuthMethod::Bearer { token } => request.bearer_auth(token),
            AuthMethod::Basic { username, password } => {
                request.basic_auth(username, Some(password))
            }
            AuthMethod::OAuth2 {
                client_id,
                client_secret,
                token_url,
                access_token,
                refresh_token,
                expires_at,
                scopes,
            } => {
                // Comprehensive OAuth2 implementation with automatic token refresh
                match self
                    .handle_oauth2_auth(
                        endpoint_name,
                        client_id,
                        client_secret,
                        token_url,
                        access_token,
                        refresh_token,
                        expires_at,
                        scopes,
                    )
                    .await
                {
                    Ok(token) => request.bearer_auth(token),
                    Err(e) => {
                        warn!("OAuth2 authentication failed: {}", e);
                        return Err(e);
                    }
                }
            }
        };

        // Add custom headers
        for (key, value) in &endpoint.headers {
            request = request.header(key, value);
        }

        // Add body if provided
        if let Some(body) = body {
            request = request.json(&body);
        }

        // Set timeout
        request = request.timeout(endpoint.timeout);

        info!("Making {} request to {}{}", method, endpoint.base_url, path);
        let start = std::time::Instant::now();

        // Execute request
        let response =
            request.send().await.context(format!("Failed to send request to {}", endpoint_name))?;

        let duration = start.elapsed();
        let status = response.status();

        // Extract headers
        let headers: HashMap<String, String> = response
            .headers()
            .iter()
            .filter_map(|(k, v)| v.to_str().ok().map(|v| (k.to_string(), v.to_string())))
            .collect();

        // Parse body
        let body = if response.status().is_success() {
            response.json::<Value>().await.unwrap_or_else(|_| Value::Null)
        } else {
            let text = response.text().await.unwrap_or_else(|_| "".to_string());
            Value::String(text)
        };

        let api_response = ApiResponse { status, headers, body, request_duration: duration };

        // Cache successful GET requests
        if method == Method::GET && status.is_success() {
            let cache_content = serde_json::to_string(&api_response)?;
            self.memory
                .store(
                    cache_content,
                    vec![cache_key],
                    MemoryMetadata {
                        source: "api_cache".to_string(),
                        tags: vec![endpoint_name.to_string()],
                        importance: 0.3,
                        associations: vec![],

                        context: Some("Generated from automated fix".to_string()),
                        created_at: chrono::Utc::now(),
                        accessed_count: 0,
                        last_accessed: None,
                        version: 1,
                    category: "general".to_string(),
                        timestamp: chrono::Utc::now(),
                        expiration: None,
                    },
                )
                .await?;
        }

        // Record request
        self.record_request(endpoint_name, &method, path, duration, status.as_u16(), false);

        // Store in memory if important
        if status.is_server_error() || status.is_client_error() {
            self.memory
                .store(
                    format!("API error: {} {} - Status: {}", method, url, status),
                    vec![format!("{:?}", api_response.body)],
                    MemoryMetadata {
                        source: "api_client".to_string(),
                        tags: vec!["error".to_string(), endpoint_name.to_string()],
                        importance: 0.7,
                        associations: vec![],

                        context: Some("Generated from automated fix".to_string()),
                        created_at: chrono::Utc::now(),
                        accessed_count: 0,
                        last_accessed: None,
                        version: 1,
                    category: "general".to_string(),
                        timestamp: chrono::Utc::now(),
                        expiration: None,
                    },
                )
                .await?;
        }

        Ok(api_response)
    }

    /// Wait for rate limit
    async fn wait_for_rate_limit(&self, endpoint_name: &str) -> Result<()> {
        if let Some(bucket) = self.rate_limiters.write().get_mut(endpoint_name) {
            while !bucket.try_consume(1) {
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        }
        Ok(())
    }

    /// Record a request
    fn record_request(
        &self,
        endpoint: &str,
        method: &Method,
        path: &str,
        duration: Duration,
        status: u16,
        cached: bool,
    ) {
        let mut history = self.request_history.write();
        history.push(RequestRecord {
            endpoint: endpoint.to_string(),
            method: method.to_string(),
            path: path.to_string(),
            timestamp: chrono::Utc::now(),
            duration,
            status,
            cached,
        });

        // Limit history size
        while history.len() > 1000 {
            history.remove(0);
        }
    }

    /// Get request statistics
    pub fn get_stats(&self, endpoint_name: &str) -> ApiStats {
        let history = self.request_history.read();
        let endpoint_requests: Vec<_> =
            history.iter().filter(|r| r.endpoint == endpoint_name).collect();

        let total_requests = endpoint_requests.len();
        let cached_requests = endpoint_requests.iter().filter(|r| r.cached).count();
        let error_requests = endpoint_requests.iter().filter(|r| r.status >= 400).count();

        let avg_duration = if !endpoint_requests.is_empty() {
            let total: Duration =
                endpoint_requests.iter().filter(|r| !r.cached).map(|r| r.duration).sum();
            total / endpoint_requests.len() as u32
        } else {
            Duration::ZERO
        };

        ApiStats {
            total_requests,
            cached_requests,
            error_requests,
            average_duration: avg_duration,
            cache_hit_rate: if total_requests > 0 {
                cached_requests as f32 / total_requests as f32
            } else {
                0.0
            },
        }
    }

    /// List all registered endpoints
    pub fn list_endpoints(&self) -> Vec<String> {
        self.endpoints.read().keys().cloned().collect()
    }
}

/// API statistics
#[derive(Debug, Clone)]
pub struct ApiStats {
    pub total_requests: usize,
    pub cached_requests: usize,
    pub error_requests: usize,
    pub average_duration: Duration,
    pub cache_hit_rate: f32,
}

/// Token bucket rate limiter
struct TokenBucket {
    capacity: usize,
    tokens: f32,
    refill_rate: f32,
    last_refill: std::time::Instant,
}

impl TokenBucket {
    fn new(capacity: usize, refill_rate: f32) -> Self {
        Self {
            capacity,
            tokens: capacity as f32,
            refill_rate,
            last_refill: std::time::Instant::now(),
        }
    }

    fn try_consume(&mut self, tokens: usize) -> bool {
        self.refill();

        if self.tokens >= tokens as f32 {
            self.tokens -= tokens as f32;
            true
        } else {
            false
        }
    }

    fn refill(&mut self) {
        let now = std::time::Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f32();

        self.tokens = (self.tokens + elapsed * self.refill_rate).min(self.capacity as f32);
        self.last_refill = now;
    }
}

/// Helper to configure common API endpoints
impl ApiClient {
    /// Configure GitHub API
    pub fn configure_github(&self, token: Option<String>) {
        let mut endpoint = ApiEndpoint {
            name: "github".to_string(),
            base_url: "https://api.github.com".to_string(),
            auth: AuthMethod::None,
            headers: HashMap::new(),
            rate_limit: Some(RateLimit { requests_per_second: 1.0, burst_size: 5 }),
            timeout: Duration::from_secs(30),
        };

        endpoint.headers.insert("Accept".to_string(), "application/vnd.github.v3+json".to_string());

        if let Some(token) = token {
            endpoint.auth = AuthMethod::Bearer { token };
        }

        self.register_endpoint(endpoint);
    }

    /// Configure OpenAI API
    pub fn configure_openai(&self, api_key: String) {
        let endpoint = ApiEndpoint {
            name: "openai".to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
            auth: AuthMethod::Bearer { token: api_key },
            headers: HashMap::from([("Content-Type".to_string(), "application/json".to_string())]),
            rate_limit: Some(RateLimit { requests_per_second: 1.0, burst_size: 3 }),
            timeout: Duration::from_secs(60),
        };

        self.register_endpoint(endpoint);
    }

    /// Configure Anthropic API
    pub fn configure_anthropic(&self, api_key: String) {
        let endpoint = ApiEndpoint {
            name: "anthropic".to_string(),
            base_url: "https://api.anthropic.com/v1".to_string(),
            auth: AuthMethod::ApiKey { header: "x-api-key".to_string(), key: api_key },
            headers: HashMap::from([
                ("Content-Type".to_string(), "application/json".to_string()),
                ("anthropic-version".to_string(), "2023-06-01".to_string()),
            ]),
            rate_limit: Some(RateLimit { requests_per_second: 0.5, burst_size: 2 }),
            timeout: Duration::from_secs(120),
        };

        self.register_endpoint(endpoint);
    }

    /// Configure all API providers from ApiKeysConfig
    pub fn configure_from_api_keys(&self, apiconfig: &crate::config::ApiKeysConfig) {
        // GitHub
        if let Some(github) = &apiconfig.github {
            self.configure_github(Some(github.token.clone()));
        }

        // OpenAI
        if let Some(key) = &apiconfig.ai_models.openai {
            self.configure_openai(key.clone());
        }

        // Anthropic
        if let Some(key) = &apiconfig.ai_models.anthropic {
            self.configure_anthropic(key.clone());
        }

        // Add more providers as needed
    }

    /// Comprehensive OAuth2 authentication handler with PKCE and token refresh
    async fn handle_oauth2_auth(
        &self,
        endpoint_name: &str,
        client_id: &str,
        client_secret: &str,
        token_url: &str,
        access_token: &Option<String>,
        refresh_token: &Option<String>,
        expires_at: &Option<chrono::DateTime<chrono::Utc>>,
        scopes: &[String],
    ) -> Result<String> {
        // Check if current token is valid and not expired
        if let (Some(token), Some(expiry)) = (access_token, expires_at) {
            let now = chrono::Utc::now();
            // Refresh 5 minutes before expiry to account for clock skew
            let refresh_threshold = *expiry - chrono::Duration::minutes(5);

            if now < refresh_threshold {
                debug!("Using existing OAuth2 token for endpoint: {}", endpoint_name);
                return Ok(token.clone());
            }
        }

        // Try to refresh token if we have a refresh token
        if let Some(refresh_tok) = refresh_token {
            match self
                .refresh_oauth2_token(
                    endpoint_name,
                    client_id,
                    client_secret,
                    token_url,
                    refresh_tok,
                )
                .await
            {
                Ok(new_token) => return Ok(new_token),
                Err(e) => {
                    warn!("Token refresh failed for {}: {}", endpoint_name, e);
                    // Fall through to full OAuth2 flow
                }
            }
        }

        // Perform full OAuth2 authorization code flow with PKCE
        self.perform_oauth2_flow(endpoint_name, client_id, client_secret, token_url, scopes).await
    }

    /// Refresh OAuth2 token using refresh token
    async fn refresh_oauth2_token(
        &self,
        endpoint_name: &str,
        client_id: &str,
        client_secret: &str,
        token_url: &str,
        refresh_token: &str,
    ) -> Result<String> {
        info!("Refreshing OAuth2 token for endpoint: {}", endpoint_name);

        let params = [
            ("grant_type", "refresh_token"),
            ("refresh_token", refresh_token),
            ("client_id", client_id),
            ("client_secret", client_secret),
        ];

        let response = self
            .client
            .post(token_url)
            .header("Content-Type", "application/x-www-form-urlencoded")
            .header("Accept", "application/json")
            .form(&params)
            .send()
            .await
            .context("Failed to send token refresh request")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!(
                "Token refresh failed with status {}: {}",
                status,
                error_text
            ));
        }

        let token_response: OAuth2TokenResponse =
            response.json().await.context("Failed to parse token refresh response")?;

        // Update the endpoint configuration with new tokens
        self.update_oauth2_endpoint(endpoint_name, &token_response).await?;

        // Store refresh event in memory for monitoring
        self.memory
            .store(
                format!("OAuth2 token refreshed for endpoint: {}", endpoint_name),
                vec![format!("expires_in: {}", token_response.expires_in.unwrap_or(3600))],
                MemoryMetadata {
                    source: "oauth2_client".to_string(),
                    tags: vec![
                        "oauth2".to_string(),
                        "token_refresh".to_string(),
                        endpoint_name.to_string(),
                    ],
                    importance: 0.6,
                    associations: vec![],

                    context: Some("Generated from automated fix".to_string()),
                    created_at: chrono::Utc::now(),
                    accessed_count: 0,
                    last_accessed: None,
                    version: 1,
                    category: "general".to_string(),
                    timestamp: chrono::Utc::now(),
                    expiration: None,
                },
            )
            .await?;

        Ok(token_response.access_token)
    }

    /// Perform full OAuth2 authorization code flow with PKCE
    async fn perform_oauth2_flow(
        &self,
        endpoint_name: &str,
        client_id: &str,
        client_secret: &str,
        token_url: &str,
        scopes: &[String],
    ) -> Result<String> {
        info!("Starting OAuth2 authorization flow for endpoint: {}", endpoint_name);

        // Generate PKCE parameters for security
        let pkce_verifier = self.generate_pkce_verifier();
        let pkce_challenge = self.generate_pkce_challenge(&pkce_verifier)?;

        // Step 1: Build authorization URL with PKCE challenge
        let auth_endpoint = match endpoint_name {
            "github" => "https://github.com/login/oauth/authorize",
            "google" => "https://accounts.google.com/o/oauth2/v2/auth",
            "microsoft" => "https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
            _ => return Err(anyhow::anyhow!("Unknown OAuth2 provider: {}", endpoint_name)),
        };

        let redirect_uri = "http://localhost:8080/oauth/callback";
        let state = self.generate_oauth_state();

        let auth_url = format!(
            "{}?client_id={}&redirect_uri={}&response_type=code&scope={}&state={}&code_challenge={}&code_challenge_method=S256",
            auth_endpoint,
            urlencoding::encode(client_id),
            urlencoding::encode(redirect_uri),
            urlencoding::encode(&scopes.join(" ")),
            urlencoding::encode(&state),
            urlencoding::encode(&pkce_challenge)
        );

        // Step 2: Check for automated flow (CI/CD) or interactive flow
        let auth_code = if let Ok(code) = std::env::var("LOKI_OAUTH2_AUTH_CODE") {
            info!("Using OAuth2 authorization code from environment");
            code
        } else {
            // Step 3: Start local HTTP server to receive callback
            let (tx, rx) = tokio::sync::oneshot::channel();
            let server_handle = self.start_oauth_callback_server(tx, state.clone()).await?;

            // Step 4: Open browser for user authorization
            info!("Opening browser for OAuth2 authorization: {}", auth_url);
            // Note: Browser opening functionality requires the 'open' crate
            // For now, just print the URL for manual opening
            println!("Please visit this URL in your browser: {}", auth_url);

            // Step 5: Wait for callback with timeout
            let auth_code = tokio::time::timeout(
                tokio::time::Duration::from_secs(300), // 5 minute timeout
                rx
            ).await
                .map_err(|_| anyhow::anyhow!("OAuth2 callback timeout"))?
                .map_err(|_| anyhow::anyhow!("OAuth2 callback server error"))?;

            // Shutdown callback server
            server_handle.abort();

            auth_code
        };

        // Step 6: Exchange authorization code for access token
        let token_response = self.exchange_oauth_code_for_token(
            token_url,
            client_id,
            client_secret,
            &auth_code,
            redirect_uri,
            &pkce_verifier,
        ).await?;

        // Step 7: Extract and store tokens
        let access_token = token_response
            .get("access_token")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("No access token in response"))?
            .to_string();

        // Store refresh token if available
        if let Some(refresh_token) = token_response.get("refresh_token").and_then(|v| v.as_str()) {
            let mut endpoints = self.endpoints.write();
            if let Some(endpoint) = endpoints.get_mut(endpoint_name) {
                endpoint.headers.insert(
                    "X-Refresh-Token".to_string(),
                    refresh_token.to_string(),
                );
            }
        }

        info!("OAuth2 flow completed successfully for {}", endpoint_name);
        Ok(access_token)
    }

    /// Generate PKCE code verifier (RFC 7636)
    fn generate_pkce_verifier(&self) -> String {
        // Generate random 128-character string
        let verifier: String = rand::thread_rng()
            .sample_iter(Alphanumeric)
            .take(128)
            .map(char::from)
            .collect();

        verifier
    }

    /// Generate PKCE code challenge from verifier
    fn generate_pkce_challenge(&self, verifier: &str) -> Result<String> {
        use base64::Engine as _;
        use base64::engine::general_purpose::URL_SAFE_NO_PAD;
        use sha2::{Digest, Sha256};

        // SHA256 hash of the verifier
        let mut hasher = Sha256::new();
        hasher.update(verifier.as_bytes());
        let result = hasher.finalize();

        // Base64url encode without padding
        let challenge = URL_SAFE_NO_PAD.encode(&result);

        Ok(challenge)
    }

    /// Generate secure random state parameter for OAuth2
    fn generate_oauth_state(&self) -> String {
        rand::thread_rng()
            .sample_iter(Alphanumeric)
            .take(32)
            .map(char::from)
            .collect()
    }

    /// Start local HTTP server to receive OAuth2 callback
    async fn start_oauth_callback_server(
        &self,
        tx: tokio::sync::oneshot::Sender<String>,
        expected_state: String,
    ) -> Result<tokio::task::JoinHandle<()>> {
        use warp::Filter;

        // Wrap the sender in an Arc<Mutex<Option<T>>> to allow sharing without Clone
        let tx = std::sync::Arc::new(std::sync::Mutex::new(Some(tx)));
        let expected_state_clone = expected_state.clone();

        let routes = warp::path!("oauth" / "callback")
            .and(warp::query::<std::collections::HashMap<String, String>>())
            .map(move |params: std::collections::HashMap<String, String>| {
                // Verify state parameter
                if params.get("state") != Some(&expected_state_clone) {
                    return warp::reply::html("OAuth2 Error: Invalid state parameter");
                }

                // Extract authorization code
                if let Some(code) = params.get("code") {
                    if let Ok(mut sender) = tx.lock() {
                        if let Some(tx) = sender.take() {
                            let _ = tx.send(code.clone());
                        }
                    }
                    warp::reply::html("Authorization successful! You can close this window.")
                } else if let Some(error) = params.get("error") {
                    let error_desc = params.get("error_description").map(|s| s.as_str()).unwrap_or("");
                    match (error.as_str(), error_desc) {
                        ("access_denied", _) => warp::reply::html("OAuth2 Error: Access denied by user"),
                        ("invalid_request", _) => warp::reply::html("OAuth2 Error: Invalid request"),
                        ("unauthorized_client", _) => warp::reply::html("OAuth2 Error: Unauthorized client"),
                        ("unsupported_response_type", _) => warp::reply::html("OAuth2 Error: Unsupported response type"),
                        ("invalid_scope", _) => warp::reply::html("OAuth2 Error: Invalid scope"),
                        ("server_error", _) => warp::reply::html("OAuth2 Error: Server error"),
                        ("temporarily_unavailable", _) => warp::reply::html("OAuth2 Error: Temporarily unavailable"),
                        _ => warp::reply::html("OAuth2 Error: Unknown error occurred"),
                    }
                } else {
                    warp::reply::html("OAuth2 Error: No authorization code received")
                }
            });

        let server = warp::serve(routes).bind(([127, 0, 0, 1], 8080));

        Ok(tokio::spawn(async move {
            server.await;
        }))
    }

    /// Exchange OAuth2 authorization code for access token
    async fn exchange_oauth_code_for_token(
        &self,
        token_url: &str,
        client_id: &str,
        client_secret: &str,
        code: &str,
        redirect_uri: &str,
        pkce_verifier: &str,
    ) -> Result<serde_json::Value> {
        let params = [
            ("grant_type", "authorization_code"),
            ("code", code),
            ("client_id", client_id),
            ("client_secret", client_secret),
            ("redirect_uri", redirect_uri),
            ("code_verifier", pkce_verifier),
        ];

        let response = self.client
            .post(token_url)
            .form(&params)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!(
                "Token exchange failed with status {}: {}",
                status,
                error_text
            ));
        }

        let token_data: serde_json::Value = response.json().await?;
        Ok(token_data)
    }

    /// Update endpoint configuration with new OAuth2 tokens
    async fn update_oauth2_endpoint(
        &self,
        endpoint_name: &str,
        token_response: &OAuth2TokenResponse,
    ) -> Result<()> {
        let mut endpoints = self.endpoints.write();

        if let Some(endpoint) = endpoints.get_mut(endpoint_name) {
            if let AuthMethod::OAuth2 { access_token, refresh_token, expires_at, .. } =
                &mut endpoint.auth
            {
                // Update tokens
                *access_token = Some(token_response.access_token.clone());

                if let Some(new_refresh_token) = &token_response.refresh_token {
                    *refresh_token = Some(new_refresh_token.clone());
                }

                // Calculate expiry time
                if let Some(expires_in) = token_response.expires_in {
                    *expires_at =
                        Some(chrono::Utc::now() + chrono::Duration::seconds(expires_in as i64));
                }

                debug!("Updated OAuth2 tokens for endpoint: {}", endpoint_name);
            }
        }

        Ok(())
    }

    /// Configure OAuth2 endpoint with comprehensive parameters
    pub fn configure_oauth2_endpoint(
        &self,
        name: &str,
        base_url: &str,
        client_id: &str,
        client_secret: &str,
        token_url: &str,
        scopes: Vec<String>,
        initial_access_token: Option<String>,
        initial_refresh_token: Option<String>,
    ) {
        let endpoint = ApiEndpoint {
            name: name.to_string(),
            base_url: base_url.to_string(),
            auth: AuthMethod::OAuth2 {
                client_id: client_id.to_string(),
                client_secret: client_secret.to_string(),
                token_url: token_url.to_string(),
                access_token: initial_access_token,
                refresh_token: initial_refresh_token,
                expires_at: None,
                scopes,
            },
            headers: HashMap::from([
                ("Content-Type".to_string(), "application/json".to_string()),
                ("Accept".to_string(), "application/json".to_string()),
            ]),
            rate_limit: Some(RateLimit { requests_per_second: 1.0, burst_size: 5 }),
            timeout: Duration::from_secs(30),
        };

        self.register_endpoint(endpoint);
        info!("Configured OAuth2 endpoint: {}", name);
    }

    /// Generate OAuth2 authorization URL with PKCE
    pub fn generate_authorization_url(
        &self,
        authorization_url: &str,
        client_id: &str,
        redirect_uri: &str,
        scopes: &[String],
        state: &str,
    ) -> Result<(String, String)> {
        let pkce_verifier = self.generate_pkce_verifier();
        let pkce_challenge = self.generate_pkce_challenge(&pkce_verifier)?;

        let mut url =
            reqwest::Url::parse(authorization_url).context("Invalid authorization URL")?;

        url.query_pairs_mut()
            .append_pair("response_type", "code")
            .append_pair("client_id", client_id)
            .append_pair("redirect_uri", redirect_uri)
            .append_pair("scope", &scopes.join(" "))
            .append_pair("state", state)
            .append_pair("code_challenge", &pkce_challenge)
            .append_pair("code_challenge_method", "S256");

        Ok((url.to_string(), pkce_verifier))
    }

    /// Exchange authorization code for tokens
    pub async fn exchange_code_for_tokens(
        &self,
        endpoint_name: &str,
        token_url: &str,
        client_id: &str,
        client_secret: &str,
        authorization_code: &str,
        redirect_uri: &str,
        pkce_verifier: &str,
    ) -> Result<OAuth2TokenResponse> {
        let params = [
            ("grant_type", "authorization_code"),
            ("code", authorization_code),
            ("redirect_uri", redirect_uri),
            ("client_id", client_id),
            ("client_secret", client_secret),
            ("code_verifier", pkce_verifier),
        ];

        let response = self
            .client
            .post(token_url)
            .header("Content-Type", "application/x-www-form-urlencoded")
            .header("Accept", "application/json")
            .form(&params)
            .send()
            .await
            .context("Failed to exchange authorization code")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!(
                "Token exchange failed with status {}: {}",
                status,
                error_text
            ));
        }

        let token_response: OAuth2TokenResponse =
            response.json().await.context("Failed to parse token exchange response")?;

        // Update endpoint configuration
        self.update_oauth2_endpoint(endpoint_name, &token_response).await?;

        // Store successful authentication in memory
        self.memory
            .store(
                format!("OAuth2 authorization completed for endpoint: {}", endpoint_name),
                vec![format!("access_token_length: {}", token_response.access_token.len())],
                MemoryMetadata {
                    source: "oauth2_client".to_string(),
                    tags: vec![
                        "oauth2".to_string(),
                        "authorization".to_string(),
                        endpoint_name.to_string(),
                    ],
                    importance: 0.8,
                    associations: vec![],

                    context: Some("Generated from automated fix".to_string()),
                    created_at: chrono::Utc::now(),
                    accessed_count: 0,
                    last_accessed: None,
                    version: 1,
                    category: "general".to_string(),
                    timestamp: chrono::Utc::now(),
                    expiration: None,
                },
            )
            .await?;

        Ok(token_response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_bucket() {
        let mut bucket = TokenBucket::new(10, 1.0);

        // Should be able to consume initial capacity
        assert!(bucket.try_consume(5));
        assert!(bucket.try_consume(5));
        assert!(!bucket.try_consume(1));

        // Wait for refill
        std::thread::sleep(Duration::from_secs(2));
        assert!(bucket.try_consume(2));
    }
}
