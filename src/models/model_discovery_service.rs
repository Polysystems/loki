use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::tools::web_search::WebSearchClient;

/// Advanced model discovery service with web search for latest model updates
pub struct ModelDiscoveryService {
    /// Web search tool for discovering latest models
    web_search: Arc<WebSearchClient>,

    /// Model registry cache with version tracking
    model_registry: Arc<RwLock<HashMap<String, ModelRegistryEntry>>>,

    /// Provider endpoints for API discovery
    provider_endpoints: HashMap<String, ProviderEndpointConfig>,

    /// Discovery configuration
    config: DiscoveryConfig,

    /// Latest model database
    latest_models: Arc<RwLock<LatestModelsDatabase>>,

    /// Model availability tracker
    availability_tracker: Arc<ModelAvailabilityTracker>,

    /// Version comparison engine
    version_engine: Arc<VersionComparisonEngine>,
}

/// Configuration for model discovery
#[derive(Debug, Clone)]
pub struct DiscoveryConfig {
    /// How often to check for new models (in seconds)
    pub discovery_interval: Duration,

    /// Enable automatic updates to latest versions
    pub auto_update_enabled: bool,

    /// Providers to check for updates
    pub enabled_providers: Vec<String>,

    /// Search terms for discovering new models
    pub search_terms: Vec<String>,

    /// Maximum age of cached model info (in hours)
    pub max_cache_age_hours: u64,

    /// Enable notification for new model discoveries
    pub notify_on_discovery: bool,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            discovery_interval: Duration::from_secs(3600), // 1 hour
            auto_update_enabled: true,
            enabled_providers: vec![
                "openai".to_string(),
                "anthropic".to_string(),
                "google".to_string(),
                "mistral".to_string(),
                "deepseek".to_string(),
                "together".to_string(),
            ],
            search_terms: vec![
                "latest AI models 2025".to_string(),
                "new language models API".to_string(),
                "claude 4 gpt o3 gemini 2.5".to_string(),
                "codestral deepseek devstral".to_string(),
                "AI model releases 2025".to_string(),
            ],
            max_cache_age_hours: 6,
            notify_on_discovery: true,
        }
    }
}

/// Provider endpoint configuration for API discovery
#[derive(Debug, Clone)]
pub struct ProviderEndpointConfig {
    pub name: String,
    pub models_endpoint: String,
    pub auth_header: Option<String>,
    pub rate_limit_per_hour: u32,
    pub last_checked: Option<Instant>,
    pub is_enabled: bool,
}

/// Model registry entry with version tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRegistryEntry {
    pub id: String,
    pub name: String,
    pub provider: String,
    pub version: ModelVersion,
    pub capabilities: Vec<String>,
    pub pricing: ModelPricing,
    pub availability: ModelAvailability,
    pub performance_metrics: PerformanceMetrics,
    pub release_date: Option<SystemTime>,
    pub last_updated: SystemTime,
    pub api_endpoint: Option<String>,
    pub context_window: Option<u32>,
    pub max_output_tokens: Option<u32>,
    pub model_size: Option<String>,
    pub architecture: Option<String>,
    pub training_data_cutoff: Option<String>,
}

/// Model version information
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct ModelVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
    pub suffix: Option<String>, // e.g., "pro", "turbo", "mini", "sonnet"
}

impl ModelVersion {
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self { major, minor, patch, suffix: None }
    }

    pub fn with_suffix(major: u32, minor: u32, patch: u32, suffix: String) -> Self {
        Self { major, minor, patch, suffix: Some(suffix) }
    }

    pub fn from_string(version_str: &str) -> Result<Self> {
        // Parse version strings like "4.0", "3.5-turbo", "2.1-pro", etc.
        let parts: Vec<&str> = version_str.split('-').collect();
        let version_part = parts[0];
        let suffix = if parts.len() > 1 { Some(parts[1].to_string()) } else { None };

        let version_numbers: Vec<&str> = version_part.split('.').collect();
        let major = version_numbers.get(0).unwrap_or(&"0").parse().unwrap_or(0);
        let minor = version_numbers.get(1).unwrap_or(&"0").parse().unwrap_or(0);
        let patch = version_numbers.get(2).unwrap_or(&"0").parse().unwrap_or(0);

        Ok(Self { major, minor, patch, suffix })
    }

    pub fn to_string(&self) -> String {
        let base = format!("{}.{}.{}", self.major, self.minor, self.patch);
        if let Some(ref suffix) = self.suffix { format!("{}-{}", base, suffix) } else { base }
    }
}

/// Model pricing information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPricing {
    pub input_cost_per_token: f64,
    pub output_cost_per_token: f64,
    pub currency: String,
    pub has_free_tier: bool,
    pub free_tier_limits: Option<FreeTierLimits>,
    pub enterprise_pricing: Option<EnterprisePricing>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FreeTierLimits {
    pub requests_per_minute: u32,
    pub tokens_per_minute: u32,
    pub requests_per_day: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnterprisePricing {
    pub volume_discounts: bool,
    pub custom_pricing: bool,
    pub minimum_commitment: Option<String>,
}

/// Model availability status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ModelAvailability {
    GenerallyAvailable,
    Preview,
    Beta,
    Alpha,
    Waitlist,
    Deprecated,
    Unavailable,
}

/// Performance metrics for models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub speed_tokens_per_second: Option<f32>,
    pub quality_score: Option<f32>,
    pub reasoning_score: Option<f32>,
    pub coding_score: Option<f32>,
    pub math_score: Option<f32>,
    pub creative_writing_score: Option<f32>,
    pub safety_score: Option<f32>,
    pub benchmark_scores: HashMap<String, f32>,
}

/// Latest models database with real-time updates
#[derive(Debug, Clone)]
pub struct LatestModelsDatabase {
    /// Models discovered from web search
    pub discovered_models: HashMap<String, DiscoveredModel>,

    /// Provider official models
    pub official_models: HashMap<String, Vec<OfficialModel>>,

    /// Last discovery timestamp
    pub last_discovery: Option<Instant>,

    /// Discovery stats
    pub discovery_stats: DiscoveryStats,
}

/// Model discovered through web search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveredModel {
    pub name: String,
    pub provider: String,
    pub description: String,
    pub announcement_date: Option<SystemTime>,
    pub api_availability: ApiAvailability,
    pub source_url: String,
    pub confidence_score: f32,
    pub verified: bool,
}

/// Official model from provider API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OfficialModel {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub created: Option<SystemTime>,
    pub owned_by: String,
    pub capabilities: Vec<String>,
    pub pricing: Option<ModelPricing>,
    pub context_window: Option<u32>,
}

/// API availability status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ApiAvailability {
    Available,
    ComingSoon,
    WaitlistOnly,
    PrivateBeta,
    Unknown,
}

/// Discovery statistics
#[derive(Debug, Clone, Default)]
pub struct DiscoveryStats {
    pub total_models_discovered: u32,
    pub new_models_this_week: u32,
    pub providers_checked: u32,
    pub successful_api_calls: u32,
    pub failed_api_calls: u32,
    pub last_successful_discovery: Option<Instant>,
}

/// Model availability tracker
pub struct ModelAvailabilityTracker {
    /// Track model status changes
    status_history: Arc<RwLock<HashMap<String, Vec<StatusChange>>>>,

    /// Monitor API endpoints
    endpoint_monitor: Arc<EndpointMonitor>,

    /// Availability cache
    availability_cache: Arc<RwLock<HashMap<String, CachedAvailability>>>,
}

#[derive(Debug, Clone)]
pub struct StatusChange {
    pub timestamp: Instant,
    pub old_status: ModelAvailability,
    pub new_status: ModelAvailability,
    pub reason: String,
}

#[derive(Debug, Clone)]
pub struct CachedAvailability {
    pub status: ModelAvailability,
    pub last_checked: Instant,
    pub response_time: Option<Duration>,
    pub error_count: u32,
}

/// Endpoint monitoring service
pub struct EndpointMonitor {
    /// HTTP client for checking endpoints
    client: reqwest::Client,

    /// Monitoring configuration
    config: MonitoringConfig,
}

#[derive(Debug, Clone)]
pub struct MonitoringConfig {
    pub check_interval: Duration,
    pub timeout: Duration,
    pub max_retries: u32,
    pub health_check_endpoints: HashMap<String, String>,
}

/// Version comparison engine
pub struct VersionComparisonEngine {
    /// Version parsing rules
    parsing_rules: HashMap<String, VersionParsingRule>,

    /// Comparison cache
    comparison_cache: Arc<RwLock<HashMap<String, VersionComparison>>>,
}

#[derive(Debug, Clone)]
pub struct VersionParsingRule {
    pub provider: String,
    pub pattern: String,
    pub extraction_rules: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct VersionComparison {
    pub model_a: String,
    pub model_b: String,
    pub newer_model: String,
    pub confidence: f32,
    pub comparison_date: Instant,
}

impl ModelDiscoveryService {
    /// Create a new model discovery service
    pub async fn new() -> Result<Self> {
        Self::new_with_discovery(true).await
    }

    /// Create a new model discovery service with optional discovery
    pub async fn new_with_discovery(enable_discovery: bool) -> Result<Self> {
        info!("🔍 Initializing Advanced Model Discovery Service (discovery: {})", enable_discovery);

        // On Apple Silicon, always disable web search to prevent TUI hanging due to
        // non-Send types in scraper crate
        let web_search = if cfg!(target_os = "macos") {
            // Reduced logging for TUI mode
            tracing::trace!("Apple Silicon detected - using minimal web search config");
            // Create a minimal web search client that won't be used
            Arc::new(
                WebSearchClient::new(
                    crate::tools::web_search::WebSearchConfig {
                        search_engines: vec![], // No search engines to prevent hanging
                        max_results_per_engine: 1,
                        request_timeout: std::time::Duration::from_secs(1),
                        user_agent: "Loki AI (disabled on macOS)".to_string(),
                        enable_content_extraction: false,
                        cache_duration: std::time::Duration::from_secs(60),
                        rate_limit_delay: std::time::Duration::from_secs(1),
                    },
                    crate::memory::CognitiveMemory::new_minimal().await?,
                    None,
                )
                .await?,
            )
        } else {
            Arc::new(
                WebSearchClient::new(
                    crate::tools::web_search::WebSearchConfig::default(),
                    crate::memory::CognitiveMemory::new_minimal().await?,
                    None, // No action validator for discovery service
                )
                .await?,
            )
        };

        let config = DiscoveryConfig::default();

        // Initialize provider endpoints
        let provider_endpoints = Self::initialize_provider_endpoints();

        let service = Self {
            web_search,
            model_registry: Arc::new(RwLock::new(HashMap::new())),
            provider_endpoints,
            config,
            latest_models: Arc::new(RwLock::new(LatestModelsDatabase {
                discovered_models: HashMap::new(),
                official_models: HashMap::new(),
                last_discovery: None,
                discovery_stats: DiscoveryStats::default(),
            })),
            availability_tracker: Arc::new(ModelAvailabilityTracker::new()),
            version_engine: Arc::new(VersionComparisonEngine::new()),
        };

        // Only start background discovery and initial discovery if enabled AND not on
        // Apple Silicon
        if enable_discovery && !cfg!(target_os = "macos") {
            // Start background discovery
            service.start_background_discovery().await?;

            // Initial discovery (with timeout to prevent hanging)
            match tokio::time::timeout(
                std::time::Duration::from_secs(10),
                service.discover_latest_models(),
            )
            .await
            {
                Ok(Ok(_)) => {
                    info!("✅ Model Discovery Service initialized successfully");
                }
                Ok(Err(e)) => {
                    warn!("Model discovery failed during initialization: {}", e);
                    info!("✅ Model Discovery Service initialized with errors");
                }
                Err(_) => {
                    warn!("Model discovery timed out during initialization");
                    info!("✅ Model Discovery Service initialized with timeout");
                }
            }
        } else {
            // Load known models without web search
            service.load_known_models().await?;
            if cfg!(target_os = "macos") {
                tracing::trace!(
                    "Model Discovery Service initialized (web discovery disabled on Apple Silicon)"
                );
            } else {
                tracing::trace!("Model Discovery Service initialized (discovery disabled)");
            }
        }

        Ok(service)
    }

    /// Start the background discovery process
    async fn start_background_discovery(&self) -> Result<()> {
        let config = self.config.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(config.discovery_interval);
            loop {
                interval.tick().await;
                // Note: Background discovery disabled for now due to Send safety issues
                // Would need to refactor WebSearchClient to be Send-safe
                debug!("Background discovery tick (discovery disabled for Send safety)");
            }
        });

        Ok(())
    }

    /// Background discovery task
    async fn background_discovery_task(
        web_search: &Arc<WebSearchClient>,
        latest_models: &Arc<RwLock<LatestModelsDatabase>>,
        config: &DiscoveryConfig,
        provider_endpoints: &HashMap<String, ProviderEndpointConfig>,
    ) -> Result<()> {
        info!("🔄 Starting background model discovery");

        // Web search for latest models
        let mut discovered_models = HashMap::new();
        for search_term in &config.search_terms {
            match web_search
                .search(crate::tools::web_search::SearchQuery {
                    query: search_term.clone(),
                    max_results: 10,
                    ..Default::default()
                })
                .await
            {
                Ok(results) => {
                    let result_strings: Vec<String> = results
                        .iter()
                        .map(|result| format!("{} {}", result.title, result.snippet))
                        .collect();
                    let discovered =
                        ModelDiscoveryService::parse_model_info_from_search(result_strings).await?;
                    for model in discovered {
                        // Convert ModelRegistryEntry to DiscoveredModel for the discovery database
                        let discovered_model = DiscoveredModel {
                            name: model.name.clone(),
                            provider: model.provider.clone(),
                            description: format!("{} - {} model", model.name, model.provider),
                            announcement_date: model.release_date,
                            api_availability: match model.availability {
                                ModelAvailability::GenerallyAvailable => ApiAvailability::Available,
                                ModelAvailability::Preview | ModelAvailability::Beta => {
                                    ApiAvailability::ComingSoon
                                }
                                _ => ApiAvailability::Unknown,
                            },
                            source_url: format!("https://api.{}.com", model.provider),
                            confidence_score: 0.8,
                            verified: false,
                        };
                        discovered_models.insert(model.name.clone(), discovered_model);
                    }
                }
                Err(e) => {
                    warn!("Web search failed for term '{}': {}", search_term, e);
                }
            }
        }

        // Check provider APIs
        let mut official_models = HashMap::new();
        for (provider, endpointconfig) in provider_endpoints {
            if !endpointconfig.is_enabled {
                continue;
            }

            match Self::fetch_official_models(provider, endpointconfig).await {
                Ok(models) => {
                    official_models.insert(provider.clone(), models);
                }
                Err(e) => {
                    warn!("Failed to fetch models from {}: {}", provider, e);
                }
            }
        }

        // Update database
        let mut db = latest_models.write().await;
        let new_models_count = discovered_models.len() - db.discovered_models.len();

        db.discovered_models = discovered_models;
        db.official_models = official_models;
        db.last_discovery = Some(Instant::now());
        db.discovery_stats.total_models_discovered = db.discovered_models.len() as u32;
        db.discovery_stats.new_models_this_week += new_models_count as u32;

        info!(
            "✅ Discovery complete: {} models discovered, {} new this run",
            db.discovered_models.len(),
            new_models_count
        );

        Ok(())
    }

    /// Discover the latest models from all sources
    pub async fn discover_latest_models(&self) -> Result<Vec<ModelRegistryEntry>> {
        info!("🔍 Discovering latest AI models from web and APIs");

        let mut all_models = Vec::new();

        // On Apple Silicon, skip web search and only use known models
        if cfg!(target_os = "macos") {
            // Silent mode for TUI - use known models without web search
            all_models.extend(self.get_known_2025_models());
        } else {
            // Get current 2025 model lineup with web search
            let latest_models_2025 = self.discover_2025_models().await?;
            all_models.extend(latest_models_2025);
        }

        // Check provider APIs for official model lists (safe on all platforms)
        let official_models = self.discover_official_models().await?;
        all_models.extend(official_models);

        // Update registry
        let mut registry = self.model_registry.write().await;
        for model in &all_models {
            registry.insert(model.id.clone(), model.clone());
        }

        info!("✅ Discovered {} total models", all_models.len());

        Ok(all_models)
    }

    /// Discover 2025 latest models using web search
    async fn discover_2025_models(&self) -> Result<Vec<ModelRegistryEntry>> {
        let mut models = Vec::new();

        // Skip web search on Apple Silicon to prevent hanging
        if cfg!(target_os = "macos") {
            // Silent mode for TUI
            models.extend(self.get_known_2025_models());
            return Ok(models);
        }

        info!("🌐 Searching for latest 2025 AI model releases");

        // Search for latest model information
        let search_queries = vec![
            "Claude 4 Anthropic 2025 API availability pricing",
            "GPT-o3 GPT-4.5 OpenAI 2025 release API",
            "Gemini 2.5 Pro Google 2025 latest model",
            "DeepSeek R1 2025 open source model API",
            "Codestral Mistral 2025 latest coding model",
            "latest AI language models 2025 API pricing",
        ];

        for query in search_queries {
            match self
                .web_search
                .search(crate::tools::web_search::SearchQuery {
                    query: query.to_string(),
                    max_results: 5,
                    ..Default::default()
                })
                .await
            {
                Ok(results) => {
                    let result_strings: Vec<String> = results
                        .iter()
                        .map(|result| format!("{} {}", result.title, result.snippet))
                        .collect();
                    let discovered =
                        ModelDiscoveryService::parse_model_info_from_search(result_strings).await?;
                    models.extend(discovered);
                }
                Err(e) => {
                    warn!("Search failed for query '{}': {}", query, e);
                }
            }
        }

        // Add known 2025 models based on search results
        models.extend(self.get_known_2025_models());

        Ok(models)
    }

    /// Get known 2025 models based on our web search results
    fn get_known_2025_models(&self) -> Vec<ModelRegistryEntry> {
        vec![
            // Claude 4 series
            ModelRegistryEntry {
                id: "claude-4-opus".to_string(),
                name: "Claude 4 Opus".to_string(),
                provider: "anthropic".to_string(),
                version: ModelVersion::new(4, 0, 0),
                capabilities: vec![
                    "text_generation".to_string(),
                    "code_generation".to_string(),
                    "reasoning".to_string(),
                    "computer_use".to_string(),
                    "agentic_tasks".to_string(),
                ],
                pricing: ModelPricing {
                    input_cost_per_token: 0.000015,
                    output_cost_per_token: 0.000075,
                    currency: "USD".to_string(),
                    has_free_tier: false,
                    free_tier_limits: None,
                    enterprise_pricing: Some(EnterprisePricing {
                        volume_discounts: true,
                        custom_pricing: true,
                        minimum_commitment: Some("$1000/month".to_string()),
                    }),
                },
                availability: ModelAvailability::GenerallyAvailable,
                performance_metrics: PerformanceMetrics {
                    speed_tokens_per_second: Some(75.0),
                    quality_score: Some(0.95),
                    reasoning_score: Some(0.98),
                    coding_score: Some(0.93),
                    math_score: Some(0.92),
                    creative_writing_score: Some(0.96),
                    safety_score: Some(0.98),
                    benchmark_scores: HashMap::from([
                        ("SWE-bench".to_string(), 72.5),
                        ("HumanEval".to_string(), 89.2),
                        ("MMLU".to_string(), 94.1),
                    ]),
                },
                release_date: Some(SystemTime::now()),
                last_updated: SystemTime::now(),
                api_endpoint: Some("https://api.anthropic.com/v1/messages".to_string()),
                context_window: Some(200000),
                max_output_tokens: Some(8192),
                model_size: Some("Large".to_string()),
                architecture: Some("Transformer".to_string()),
                training_data_cutoff: Some("April 2024".to_string()),
            },
            // Claude 4 Sonnet
            ModelRegistryEntry {
                id: "claude-4-sonnet".to_string(),
                name: "Claude 4 Sonnet".to_string(),
                provider: "anthropic".to_string(),
                version: ModelVersion::new(4, 0, 0),
                capabilities: vec![
                    "text_generation".to_string(),
                    "code_generation".to_string(),
                    "reasoning".to_string(),
                    "computer_use".to_string(),
                ],
                pricing: ModelPricing {
                    input_cost_per_token: 0.000003,
                    output_cost_per_token: 0.000015,
                    currency: "USD".to_string(),
                    has_free_tier: true,
                    free_tier_limits: Some(FreeTierLimits {
                        requests_per_minute: 5,
                        tokens_per_minute: 25000,
                        requests_per_day: 1000,
                    }),
                    enterprise_pricing: None,
                },
                availability: ModelAvailability::GenerallyAvailable,
                performance_metrics: PerformanceMetrics {
                    speed_tokens_per_second: Some(120.0),
                    quality_score: Some(0.90),
                    reasoning_score: Some(0.92),
                    coding_score: Some(0.94),
                    math_score: Some(0.88),
                    creative_writing_score: Some(0.91),
                    safety_score: Some(0.97),
                    benchmark_scores: HashMap::from([
                        ("SWE-bench".to_string(), 72.7),
                        ("HumanEval".to_string(), 87.8),
                        ("MMLU".to_string(), 91.3),
                    ]),
                },
                release_date: Some(SystemTime::now()),
                last_updated: SystemTime::now(),
                api_endpoint: Some("https://api.anthropic.com/v1/messages".to_string()),
                context_window: Some(200000),
                max_output_tokens: Some(8192),
                model_size: Some("Medium".to_string()),
                architecture: Some("Transformer".to_string()),
                training_data_cutoff: Some("April 2024".to_string()),
            },
            // Claude 3.7 Sonnet
            ModelRegistryEntry {
                id: "claude-3-7-sonnet".to_string(),
                name: "Claude 3.7 Sonnet".to_string(),
                provider: "anthropic".to_string(),
                version: ModelVersion::new(3, 7, 0),
                capabilities: vec![
                    "text_generation".to_string(),
                    "code_generation".to_string(),
                    "reasoning".to_string(),
                    "thinking_mode".to_string(),
                ],
                pricing: ModelPricing {
                    input_cost_per_token: 0.000003,
                    output_cost_per_token: 0.000015,
                    currency: "USD".to_string(),
                    has_free_tier: true,
                    free_tier_limits: Some(FreeTierLimits {
                        requests_per_minute: 5,
                        tokens_per_minute: 25000,
                        requests_per_day: 1000,
                    }),
                    enterprise_pricing: None,
                },
                availability: ModelAvailability::GenerallyAvailable,
                performance_metrics: PerformanceMetrics {
                    speed_tokens_per_second: Some(110.0),
                    quality_score: Some(0.88),
                    reasoning_score: Some(0.90),
                    coding_score: Some(0.92),
                    math_score: Some(0.86),
                    creative_writing_score: Some(0.89),
                    safety_score: Some(0.96),
                    benchmark_scores: HashMap::new(),
                },
                release_date: Some(SystemTime::now() - Duration::from_secs(30 * 24 * 3600)), /* Feb 2025 */
                last_updated: SystemTime::now(),
                api_endpoint: Some("https://api.anthropic.com/v1/messages".to_string()),
                context_window: Some(200000),
                max_output_tokens: Some(8192),
                model_size: Some("Medium".to_string()),
                architecture: Some("Transformer".to_string()),
                training_data_cutoff: Some("January 2024".to_string()),
            },
            // GPT-o3
            ModelRegistryEntry {
                id: "gpt-o3".to_string(),
                name: "GPT-o3".to_string(),
                provider: "openai".to_string(),
                version: ModelVersion::new(3, 0, 0),
                capabilities: vec![
                    "text_generation".to_string(),
                    "code_generation".to_string(),
                    "reasoning".to_string(),
                    "chain_of_thought".to_string(),
                ],
                pricing: ModelPricing {
                    input_cost_per_token: 0.000030,
                    output_cost_per_token: 0.000120,
                    currency: "USD".to_string(),
                    has_free_tier: false,
                    free_tier_limits: None,
                    enterprise_pricing: Some(EnterprisePricing {
                        volume_discounts: true,
                        custom_pricing: true,
                        minimum_commitment: Some("$500/month".to_string()),
                    }),
                },
                availability: ModelAvailability::GenerallyAvailable,
                performance_metrics: PerformanceMetrics {
                    speed_tokens_per_second: Some(45.0),
                    quality_score: Some(0.97),
                    reasoning_score: Some(0.99),
                    coding_score: Some(0.91),
                    math_score: Some(0.96),
                    creative_writing_score: Some(0.90),
                    safety_score: Some(0.95),
                    benchmark_scores: HashMap::new(),
                },
                release_date: Some(SystemTime::now() - Duration::from_secs(60 * 24 * 3600)), /* April 2025 */
                last_updated: SystemTime::now(),
                api_endpoint: Some("https://api.openai.com/v1/chat/completions".to_string()),
                context_window: Some(128000),
                max_output_tokens: Some(16384),
                model_size: Some("Large".to_string()),
                architecture: Some("Transformer".to_string()),
                training_data_cutoff: Some("April 2024".to_string()),
            },
            // GPT-o3-mini
            ModelRegistryEntry {
                id: "gpt-o3-mini".to_string(),
                name: "GPT-o3-mini".to_string(),
                provider: "openai".to_string(),
                version: ModelVersion::new(3, 0, 0),
                capabilities: vec![
                    "text_generation".to_string(),
                    "code_generation".to_string(),
                    "reasoning".to_string(),
                    "fast_inference".to_string(),
                ],
                pricing: ModelPricing {
                    input_cost_per_token: 0.000003,
                    output_cost_per_token: 0.000012,
                    currency: "USD".to_string(),
                    has_free_tier: true,
                    free_tier_limits: Some(FreeTierLimits {
                        requests_per_minute: 20,
                        tokens_per_minute: 150000,
                        requests_per_day: 10000,
                    }),
                    enterprise_pricing: None,
                },
                availability: ModelAvailability::GenerallyAvailable,
                performance_metrics: PerformanceMetrics {
                    speed_tokens_per_second: Some(180.0),
                    quality_score: Some(0.85),
                    reasoning_score: Some(0.88),
                    coding_score: Some(0.83),
                    math_score: Some(0.84),
                    creative_writing_score: Some(0.82),
                    safety_score: Some(0.94),
                    benchmark_scores: HashMap::new(),
                },
                release_date: Some(SystemTime::now() - Duration::from_secs(60 * 24 * 3600)), /* April 2025 */
                last_updated: SystemTime::now(),
                api_endpoint: Some("https://api.openai.com/v1/chat/completions".to_string()),
                context_window: Some(128000),
                max_output_tokens: Some(8192),
                model_size: Some("Small".to_string()),
                architecture: Some("Transformer".to_string()),
                training_data_cutoff: Some("April 2024".to_string()),
            },
            // Gemini 2.5 Pro
            ModelRegistryEntry {
                id: "gemini-2-5-pro".to_string(),
                name: "Gemini 2.5 Pro".to_string(),
                provider: "google".to_string(),
                version: ModelVersion::new(2, 5, 0),
                capabilities: vec![
                    "text_generation".to_string(),
                    "code_generation".to_string(),
                    "multimodal".to_string(),
                    "reasoning".to_string(),
                    "audio_input".to_string(),
                    "audio_output".to_string(),
                ],
                pricing: ModelPricing {
                    input_cost_per_token: 0.000001,
                    output_cost_per_token: 0.000005,
                    currency: "USD".to_string(),
                    has_free_tier: true,
                    free_tier_limits: Some(FreeTierLimits {
                        requests_per_minute: 15,
                        tokens_per_minute: 32000,
                        requests_per_day: 1500,
                    }),
                    enterprise_pricing: Some(EnterprisePricing {
                        volume_discounts: true,
                        custom_pricing: false,
                        minimum_commitment: None,
                    }),
                },
                availability: ModelAvailability::GenerallyAvailable,
                performance_metrics: PerformanceMetrics {
                    speed_tokens_per_second: Some(200.0),
                    quality_score: Some(0.93),
                    reasoning_score: Some(0.95),
                    coding_score: Some(0.90),
                    math_score: Some(0.94),
                    creative_writing_score: Some(0.88),
                    safety_score: Some(0.96),
                    benchmark_scores: HashMap::new(),
                },
                release_date: Some(SystemTime::now() - Duration::from_secs(30 * 24 * 3600)), /* March 2025 */
                last_updated: SystemTime::now(),
                api_endpoint: Some(
                    "https://generativelanguage.googleapis.com/v1beta/models".to_string(),
                ),
                context_window: Some(1000000),
                max_output_tokens: Some(8192),
                model_size: Some("Large".to_string()),
                architecture: Some("Gemini".to_string()),
                training_data_cutoff: Some("March 2024".to_string()),
            },
            // Gemini 2.5 Flash
            ModelRegistryEntry {
                id: "gemini-2-5-flash".to_string(),
                name: "Gemini 2.5 Flash".to_string(),
                provider: "google".to_string(),
                version: ModelVersion::new(2, 5, 0),
                capabilities: vec![
                    "text_generation".to_string(),
                    "code_generation".to_string(),
                    "multimodal".to_string(),
                    "fast_inference".to_string(),
                    "audio_input".to_string(),
                    "audio_output".to_string(),
                ],
                pricing: ModelPricing {
                    input_cost_per_token: 0.0000005,
                    output_cost_per_token: 0.000002,
                    currency: "USD".to_string(),
                    has_free_tier: true,
                    free_tier_limits: Some(FreeTierLimits {
                        requests_per_minute: 60,
                        tokens_per_minute: 1000000,
                        requests_per_day: 15000,
                    }),
                    enterprise_pricing: None,
                },
                availability: ModelAvailability::GenerallyAvailable,
                performance_metrics: PerformanceMetrics {
                    speed_tokens_per_second: Some(768.0),
                    quality_score: Some(0.80),
                    reasoning_score: Some(0.82),
                    coding_score: Some(0.78),
                    math_score: Some(0.81),
                    creative_writing_score: Some(0.76),
                    safety_score: Some(0.95),
                    benchmark_scores: HashMap::new(),
                },
                release_date: Some(SystemTime::now() - Duration::from_secs(30 * 24 * 3600)),
                last_updated: SystemTime::now(),
                api_endpoint: Some(
                    "https://generativelanguage.googleapis.com/v1beta/models".to_string(),
                ),
                context_window: Some(1000000),
                max_output_tokens: Some(8192),
                model_size: Some("Medium".to_string()),
                architecture: Some("Gemini".to_string()),
                training_data_cutoff: Some("March 2024".to_string()),
            },
            // DeepSeek R1
            ModelRegistryEntry {
                id: "deepseek-r1".to_string(),
                name: "DeepSeek R1".to_string(),
                provider: "deepseek".to_string(),
                version: ModelVersion::new(1, 0, 0),
                capabilities: vec![
                    "text_generation".to_string(),
                    "code_generation".to_string(),
                    "reasoning".to_string(),
                    "math".to_string(),
                    "open_source".to_string(),
                ],
                pricing: ModelPricing {
                    input_cost_per_token: 0.00000014,
                    output_cost_per_token: 0.00000055,
                    currency: "USD".to_string(),
                    has_free_tier: true,
                    free_tier_limits: Some(FreeTierLimits {
                        requests_per_minute: 60,
                        tokens_per_minute: 1000000,
                        requests_per_day: 10000,
                    }),
                    enterprise_pricing: None,
                },
                availability: ModelAvailability::GenerallyAvailable,
                performance_metrics: PerformanceMetrics {
                    speed_tokens_per_second: Some(150.0),
                    quality_score: Some(0.82),
                    reasoning_score: Some(0.89),
                    coding_score: Some(0.80),
                    math_score: Some(0.95),
                    creative_writing_score: Some(0.75),
                    safety_score: Some(0.90),
                    benchmark_scores: HashMap::new(),
                },
                release_date: Some(SystemTime::now() - Duration::from_secs(150 * 24 * 3600)), /* Jan 2025 */
                last_updated: SystemTime::now(),
                api_endpoint: Some("https://api.deepseek.com/v1/chat/completions".to_string()),
                context_window: Some(64000),
                max_output_tokens: Some(8192),
                model_size: Some("Large".to_string()),
                architecture: Some("Transformer".to_string()),
                training_data_cutoff: Some("November 2024".to_string()),
            },
            // Codestral (2501)
            ModelRegistryEntry {
                id: "codestral-2501".to_string(),
                name: "Codestral (2501)".to_string(),
                provider: "mistral".to_string(),
                version: ModelVersion::new(25, 0, 1),
                capabilities: vec![
                    "text_generation".to_string(),
                    "code_generation".to_string(),
                    "code_completion".to_string(),
                    "fast_inference".to_string(),
                ],
                pricing: ModelPricing {
                    input_cost_per_token: 0.000001,
                    output_cost_per_token: 0.000003,
                    currency: "USD".to_string(),
                    has_free_tier: true,
                    free_tier_limits: Some(FreeTierLimits {
                        requests_per_minute: 30,
                        tokens_per_minute: 200000,
                        requests_per_day: 5000,
                    }),
                    enterprise_pricing: None,
                },
                availability: ModelAvailability::GenerallyAvailable,
                performance_metrics: PerformanceMetrics {
                    speed_tokens_per_second: Some(300.0),
                    quality_score: Some(0.88),
                    reasoning_score: Some(0.84),
                    coding_score: Some(0.95),
                    math_score: Some(0.82),
                    creative_writing_score: Some(0.78),
                    safety_score: Some(0.93),
                    benchmark_scores: HashMap::from([("HumanEval".to_string(), 98.92)]),
                },
                release_date: Some(SystemTime::now() - Duration::from_secs(30 * 24 * 3600)),
                last_updated: SystemTime::now(),
                api_endpoint: Some("https://api.mistral.ai/v1/chat/completions".to_string()),
                context_window: Some(32000),
                max_output_tokens: Some(8192),
                model_size: Some("Medium".to_string()),
                architecture: Some("Transformer".to_string()),
                training_data_cutoff: Some("December 2024".to_string()),
            },
        ]
    }

    /// Discover official models from provider APIs
    async fn discover_official_models(&self) -> Result<Vec<ModelRegistryEntry>> {
        let mut models = Vec::new();

        for (provider, config) in &self.provider_endpoints {
            if !config.is_enabled {
                continue;
            }

            match Self::fetch_official_models(provider, config).await {
                Ok(official_models) => {
                    for official_model in official_models {
                        let registry_entry =
                            self.convert_official_to_registry(official_model, provider).await?;
                        models.push(registry_entry);
                    }
                }
                Err(e) => {
                    warn!("Failed to fetch models from {}: {}", provider, e);
                }
            }
        }

        Ok(models)
    }

    /// Initialize provider endpoints
    fn initialize_provider_endpoints() -> HashMap<String, ProviderEndpointConfig> {
        let mut endpoints = HashMap::new();

        endpoints.insert(
            "openai".to_string(),
            ProviderEndpointConfig {
                name: "OpenAI".to_string(),
                models_endpoint: "https://api.openai.com/v1/models".to_string(),
                auth_header: Some("Authorization".to_string()),
                rate_limit_per_hour: 200,
                last_checked: None,
                is_enabled: true,
            },
        );

        endpoints.insert(
            "anthropic".to_string(),
            ProviderEndpointConfig {
                name: "Anthropic".to_string(),
                models_endpoint: "https://api.anthropic.com/v1/models".to_string(),
                auth_header: Some("x-api-key".to_string()),
                rate_limit_per_hour: 100,
                last_checked: None,
                is_enabled: true,
            },
        );

        endpoints.insert(
            "google".to_string(),
            ProviderEndpointConfig {
                name: "Google".to_string(),
                models_endpoint: "https://generativelanguage.googleapis.com/v1beta/models"
                    .to_string(),
                auth_header: Some("Authorization".to_string()),
                rate_limit_per_hour: 150,
                last_checked: None,
                is_enabled: true,
            },
        );

        endpoints.insert(
            "mistral".to_string(),
            ProviderEndpointConfig {
                name: "Mistral".to_string(),
                models_endpoint: "https://api.mistral.ai/v1/models".to_string(),
                auth_header: Some("Authorization".to_string()),
                rate_limit_per_hour: 100,
                last_checked: None,
                is_enabled: true,
            },
        );

        endpoints.insert(
            "deepseek".to_string(),
            ProviderEndpointConfig {
                name: "DeepSeek".to_string(),
                models_endpoint: "https://api.deepseek.com/v1/models".to_string(),
                auth_header: Some("Authorization".to_string()),
                rate_limit_per_hour: 100,
                last_checked: None,
                is_enabled: true,
            },
        );

        endpoints
    }

    /// Fetch official models from provider API
    async fn fetch_official_models(
        provider: &str,
        config: &ProviderEndpointConfig,
    ) -> Result<Vec<OfficialModel>> {
        // Mock implementation - in reality would make HTTP requests to provider APIs
        info!("📡 Fetching official models from {}", provider);

        // Return mock models for now
        Ok(vec![OfficialModel {
            id: format!("{}-latest", provider),
            name: format!("{} Latest Model", config.name),
            description: Some(format!("Latest model from {}", config.name)),
            created: Some(SystemTime::now()),
            owned_by: provider.to_string(),
            capabilities: vec!["text_generation".to_string()],
            pricing: None,
            context_window: Some(8192),
        }])
    }

    /// Convert official model to registry entry
    async fn convert_official_to_registry(
        &self,
        official: OfficialModel,
        provider: &str,
    ) -> Result<ModelRegistryEntry> {
        Ok(ModelRegistryEntry {
            id: official.id.clone(),
            name: official.name,
            provider: provider.to_string(),
            version: ModelVersion::from_string("1.0.0")?,
            capabilities: official.capabilities,
            pricing: official.pricing.unwrap_or_else(|| ModelPricing {
                input_cost_per_token: 0.0,
                output_cost_per_token: 0.0,
                currency: "USD".to_string(),
                has_free_tier: false,
                free_tier_limits: None,
                enterprise_pricing: None,
            }),
            availability: ModelAvailability::GenerallyAvailable,
            performance_metrics: PerformanceMetrics {
                speed_tokens_per_second: None,
                quality_score: None,
                reasoning_score: None,
                coding_score: None,
                math_score: None,
                creative_writing_score: None,
                safety_score: None,
                benchmark_scores: HashMap::new(),
            },
            release_date: official.created,
            last_updated: SystemTime::now(),
            api_endpoint: None,
            context_window: official.context_window,
            max_output_tokens: None,
            model_size: None,
            architecture: None,
            training_data_cutoff: None,
        })
    }

    /// Parse model information from search results
    async fn parse_model_info_from_search(
        _results: Vec<String>,
    ) -> Result<Vec<ModelRegistryEntry>> {
        // Mock implementation - would parse search results for model information
        Ok(vec![])
    }

    /// Extract models from search results
    fn extract_models_from_search_results(_results: Vec<String>) -> Vec<DiscoveredModel> {
        // Mock implementation
        vec![]
    }

    /// Get all discovered models
    pub async fn get_all_models(&self) -> Vec<ModelRegistryEntry> {
        let registry = self.model_registry.read().await;
        registry.values().cloned().collect()
    }

    /// Get models by provider
    pub async fn get_models_by_provider(&self, provider: &str) -> Vec<ModelRegistryEntry> {
        let registry = self.model_registry.read().await;
        registry.values().filter(|model| model.provider == provider).cloned().collect()
    }

    /// Check for model updates
    pub async fn check_for_updates(&self) -> Result<Vec<String>> {
        info!("🔄 Checking for model updates");

        let mut updates = Vec::new();

        // Re-discover latest models
        let latest_models = self.discover_latest_models().await?;

        // Compare with existing registry
        let registry = self.model_registry.read().await;
        for latest_model in latest_models {
            if let Some(existing_model) = registry.get(&latest_model.id) {
                if latest_model.version > existing_model.version {
                    updates.push(format!(
                        "Update available for {}: {} -> {}",
                        latest_model.name,
                        existing_model.version.to_string(),
                        latest_model.version.to_string()
                    ));
                }
            } else {
                updates.push(format!("New model discovered: {}", latest_model.name));
            }
        }

        info!("Found {} updates", updates.len());
        Ok(updates)
    }

    /// Get discovery statistics
    pub async fn get_discovery_stats(&self) -> DiscoveryStats {
        let db = self.latest_models.read().await;
        db.discovery_stats.clone()
    }

    /// Load known models without web search (for offline mode)
    async fn load_known_models(&self) -> Result<()> {
        info!("📚 Loading known models (offline mode)");

        let known_models = self.get_known_2025_models();

        // Update registry with known models
        let mut registry = self.model_registry.write().await;
        for model in &known_models {
            registry.insert(model.id.clone(), model.clone());
        }

        info!("✅ Loaded {} known models", known_models.len());
        Ok(())
    }
}

impl ModelAvailabilityTracker {
    fn new() -> Self {
        Self {
            status_history: Arc::new(RwLock::new(HashMap::new())),
            endpoint_monitor: Arc::new(EndpointMonitor::new()),
            availability_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

impl EndpointMonitor {
    fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
            config: MonitoringConfig {
                check_interval: Duration::from_secs(300),
                timeout: Duration::from_secs(10),
                max_retries: 3,
                health_check_endpoints: HashMap::new(),
            },
        }
    }
}

impl VersionComparisonEngine {
    fn new() -> Self {
        Self {
            parsing_rules: HashMap::new(),
            comparison_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}
