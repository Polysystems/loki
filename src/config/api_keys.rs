//! API Keys Configuration Module
//!
//! This module handles loading, validation, and management of all API keys
//! required by Loki's various integrations.

use std::collections::HashMap;
use std::env;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use super::secure_env::SecureEnv;

/// Master API configuration containing all service credentials
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ApiKeysConfig {
    /// GitHub API configuration
    pub github: Option<GitHubConfig>,

    /// X (Twitter) API configuration
    pub x_twitter: Option<XTwitterConfig>,

    /// AI Model API keys
    pub ai_models: AiModelsConfig,

    /// Embedding Model API keys
    pub embedding_models: EmbeddingModelsConfig,

    /// Search API keys
    pub search: SearchApisConfig,

    /// Vector database configuration
    pub vector_db: Option<VectorDbConfig>,

    /// Optional service API keys
    pub optional_services: OptionalServicesConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitHubConfig {
    pub token: String,
    pub owner: String,
    pub repo: String,
    pub default_branch: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XTwitterConfig {
    pub api_key: String,
    pub api_secret: String,
    pub access_token: String,
    pub access_token_secret: String,
    pub bearer_token: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AiModelsConfig {
    pub openai: Option<String>,
    pub anthropic: Option<String>,
    pub deepseek: Option<String>,
    pub mistral: Option<String>,
    pub codestral: Option<String>,
    pub gemini: Option<String>,
    pub grok: Option<String>,
    pub cohere: Option<String>,
    pub perplexity: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EmbeddingModelsConfig {
    pub openai: Option<String>,
    pub cohere: Option<String>,
    pub huggingface: Option<String>,
    pub sentence_transformers: Option<String>,
    pub voyage: Option<String>,
    pub jina: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SearchApisConfig {
    pub brave: Option<String>,
    pub google: Option<GoogleSearchConfig>,
    pub bing: Option<String>,
    pub serper: Option<String>,
    pub you: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoogleSearchConfig {
    pub api_key: String,
    pub cx: String, // Custom Search Engine ID
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorDbConfig {
    pub provider: VectorDbProvider,
    pub api_key: String,
    pub environment: Option<String>, // For Pinecone
    pub url: Option<String>,         // For self-hosted options
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VectorDbProvider {
    Pinecone,
    Qdrant,
    Weaviate,
    Chroma,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OptionalServicesConfig {
    pub stack_exchange: Option<String>,
    pub replicate: Option<String>,
    pub huggingface: Option<String>,
    pub wolfram_alpha: Option<String>,
    pub news_api: Option<String>,
    pub weather_api: Option<String>,
}

impl ApiKeysConfig {
    /// Check if any API key is configured
    pub fn has_any_key(&self) -> bool {
        // Check if any AI model API key is configured
        self.ai_models.openai.is_some()
            || self.ai_models.anthropic.is_some()
            || self.ai_models.deepseek.is_some()
            || self.ai_models.mistral.is_some()
            || self.ai_models.codestral.is_some()
            || self.ai_models.gemini.is_some()
            || self.ai_models.grok.is_some()
            || self.ai_models.perplexity.is_some()
            || self.ai_models.cohere.is_some()
            // Also check embedding models
            || self.embedding_models.openai.is_some()
            || self.embedding_models.cohere.is_some()
            || self.embedding_models.huggingface.is_some()
            || self.embedding_models.sentence_transformers.is_some()
            || self.embedding_models.voyage.is_some()
            || self.embedding_models.jina.is_some()
    }

    /// Load API keys from secure config first, then environment variables
    pub fn from_env() -> Result<Self> {
        // First try to load from secure configuration
        if let Ok(secureconfig) = super::setup::SecureConfig::load() {
            if secureconfig.api_keys.has_anyconfigured() {
                info!("Loading API keys from secure configuration...");
                return Ok(secureconfig.api_keys);
            }
        }

        // Fall back to environment variables
        Self::from_env_vars()
    }

    /// Load API keys from environment variables (secure method)
    pub fn from_env_vars() -> Result<Self> {
        // Use secure environment loading instead of dotenv
        let mut secure_env = SecureEnv::new();

        // Check for API keys using secure environment
        let api_keys = secure_env.load_api_keys();
        let configured_count = api_keys.values().filter(|v| v.is_some()).count();
        
        if configured_count > 0 {
            info!("Loading {} API keys from secure environment", configured_count);
        }

        let config = Self {
            github: Self::load_githubconfig_secure(&mut secure_env)?,
            x_twitter: Self::load_x_twitterconfig_secure(&mut secure_env)?,
            ai_models: Self::load_ai_modelsconfig_secure(&mut secure_env),
            embedding_models: Self::load_embedding_modelsconfig_secure(&mut secure_env),
            search: Self::load_search_apisconfig_secure(&mut secure_env),
            vector_db: Self::load_vector_dbconfig_secure(&mut secure_env)?,
            optional_services: Self::load_optional_servicesconfig_secure(&mut secure_env),
        };

        // Don't validate API keys - local models should work fine without any APIs
        // config.validate()?;  // Commenting out - not needed for local operation

        Ok(config)
    }

    /// Load GitHub configuration (secure version)
    fn load_githubconfig_secure(secure_env: &mut SecureEnv) -> Result<Option<GitHubConfig>> {
        match (
            secure_env.get("GITHUB_TOKEN"),
            secure_env.get("GITHUB_OWNER"),
            secure_env.get("GITHUB_REPO"),
            secure_env.get("GITHUB_DEFAULT_BRANCH"),
        ) {
            (Some(token), Some(owner), Some(repo), Some(branch)) => {
                info!("GitHub API configured");
                Ok(Some(GitHubConfig { token, owner, repo, default_branch: branch }))
            }
            (Some(_), _, _, _) => {
                warn!(
                    "Partial GitHub configuration found. Need GITHUB_OWNER, GITHUB_REPO, and \
                     GITHUB_DEFAULT_BRANCH"
                );
                Ok(None)
            }
            _ => {
                info!("GitHub API not configured");
                Ok(None)
            }
        }
    }

    /// Load X/Twitter configuration (secure version)
    fn load_x_twitterconfig_secure(secure_env: &mut SecureEnv) -> Result<Option<XTwitterConfig>> {
        match (
            secure_env.get("X_API_KEY"),
            secure_env.get("X_API_SECRET"),
            secure_env.get("X_ACCESS_TOKEN"),
            secure_env.get("X_ACCESS_TOKEN_SECRET"),
            secure_env.get("X_BEARER_TOKEN"),
        ) {
            (
                Some(api_key),
                Some(api_secret),
                Some(access_token),
                Some(access_token_secret),
                Some(bearer_token),
            ) => {
                info!("X (Twitter) API configured");
                Ok(Some(XTwitterConfig {
                    api_key,
                    api_secret,
                    access_token,
                    access_token_secret,
                    bearer_token,
                }))
            }
            _ => {
                info!("X (Twitter) API not configured");
                Ok(None)
            }
        }
    }

    /// Load AI model API keys (secure version)
    fn load_ai_modelsconfig_secure(secure_env: &mut SecureEnv) -> AiModelsConfig {
        let mut config = AiModelsConfig::default();

        if let Some(key) = secure_env.get("OPENAI_API_KEY") {
            info!("OpenAI API configured");
            config.openai = Some(key);
        }

        if let Some(key) = secure_env.get("ANTHROPIC_API_KEY") {
            info!("Anthropic API configured");
            config.anthropic = Some(key);
        }

        if let Some(key) = secure_env.get("DEEPSEEK_API_KEY") {
            info!("DeepSeek API configured");
            config.deepseek = Some(key);
        }

        if let Some(key) = secure_env.get("MISTRAL_API_KEY") {
            info!("Mistral API configured");
            config.mistral = Some(key);
        }

        if let Some(key) = secure_env.get("CODESTRAL_API_KEY") {
            info!("Codestral API configured");
            config.codestral = Some(key);
        }

        if let Some(key) = secure_env.get("GEMINI_API_KEY") {
            info!("Gemini API configured");
            config.gemini = Some(key);
        }

        if let Some(key) = secure_env.get("GROK_API_KEY") {
            info!("Grok API configured");
            config.grok = Some(key);
        }

        if let Some(key) = secure_env.get("COHERE_API_KEY") {
            info!("Cohere API configured");
            config.cohere = Some(key);
        }

        if let Some(key) = secure_env.get("PERPLEXITY_API_KEY") {
            info!("Perplexity API configured");
            config.perplexity = Some(key);
        }

        config
    }

    /// Load embedding model API keys (secure version)
    fn load_embedding_modelsconfig_secure(secure_env: &mut SecureEnv) -> EmbeddingModelsConfig {
        let mut config = EmbeddingModelsConfig::default();

        if let Some(key) = secure_env.get("OPENAI_API_KEY") {
            config.openai = Some(key);
        }

        if let Some(key) = secure_env.get("COHERE_API_KEY") {
            config.cohere = Some(key);
        }

        if let Some(key) = secure_env.get("HUGGINGFACE_API_KEY") {
            config.huggingface = Some(key);
        }

        if let Some(key) = secure_env.get("SENTENCE_TRANSFORMERS_API_KEY") {
            config.sentence_transformers = Some(key);
        }

        if let Some(key) = secure_env.get("VOYAGE_API_KEY") {
            config.voyage = Some(key);
        }

        if let Some(key) = secure_env.get("JINA_API_KEY") {
            config.jina = Some(key);
        }

        config
    }

    /// Load search API configuration (secure version)
    fn load_search_apisconfig_secure(secure_env: &mut SecureEnv) -> SearchApisConfig {
        let mut config = SearchApisConfig::default();

        if let Some(key) = secure_env.get("BRAVE_SEARCH_API_KEY") {
            config.brave = Some(key);
        }

        if let Some(api_key) = secure_env.get("GOOGLE_SEARCH_API_KEY") {
            if let Some(cx) = secure_env.get("GOOGLE_SEARCH_CX") {
                config.google = Some(GoogleSearchConfig { api_key, cx });
            }
        }

        if let Some(key) = secure_env.get("BING_SEARCH_API_KEY") {
            config.bing = Some(key);
        }

        if let Some(key) = secure_env.get("SERPER_API_KEY") {
            config.serper = Some(key);
        }

        if let Some(key) = secure_env.get("YOU_API_KEY") {
            config.you = Some(key);
        }

        config
    }

    /// Load vector database configuration (secure version)
    fn load_vector_dbconfig_secure(secure_env: &mut SecureEnv) -> Result<Option<VectorDbConfig>> {
        // Check for various vector DB configurations
        if let Some(key) = secure_env.get("PINECONE_API_KEY") {
            let environment = secure_env.get("PINECONE_ENVIRONMENT");
            return Ok(Some(VectorDbConfig {
                provider: VectorDbProvider::Pinecone,
                api_key: key,
                environment,
                url: None,
            }));
        }

        if let Some(key) = secure_env.get("QDRANT_API_KEY") {
            let url = secure_env.get("QDRANT_URL");
            return Ok(Some(VectorDbConfig {
                provider: VectorDbProvider::Qdrant,
                api_key: key,
                environment: None,
                url,
            }));
        }

        if let Some(key) = secure_env.get("WEAVIATE_API_KEY") {
            let url = secure_env.get("WEAVIATE_URL");
            return Ok(Some(VectorDbConfig {
                provider: VectorDbProvider::Weaviate,
                api_key: key,
                environment: None,
                url,
            }));
        }

        if let Some(key) = secure_env.get("CHROMA_API_KEY") {
            let url = secure_env.get("CHROMA_URL");
            return Ok(Some(VectorDbConfig {
                provider: VectorDbProvider::Chroma,
                api_key: key,
                environment: None,
                url,
            }));
        }

        Ok(None)
    }

    /// Load optional services configuration (secure version)
    fn load_optional_servicesconfig_secure(secure_env: &mut SecureEnv) -> OptionalServicesConfig {
        OptionalServicesConfig {
            stack_exchange: secure_env.get("STACK_EXCHANGE_API_KEY"),
            replicate: secure_env.get("REPLICATE_API_TOKEN"),
            huggingface: secure_env.get("HUGGINGFACE_API_KEY"),
            wolfram_alpha: secure_env.get("WOLFRAM_ALPHA_APP_ID"),
            news_api: secure_env.get("NEWS_API_KEY"),
            weather_api: secure_env.get("WEATHER_API_KEY"),
        }
    }

    /// Load GitHub configuration
    fn load_githubconfig() -> Result<Option<GitHubConfig>> {
        match (
            env::var("GITHUB_TOKEN").ok(),
            env::var("GITHUB_OWNER").ok(),
            env::var("GITHUB_REPO").ok(),
            env::var("GITHUB_DEFAULT_BRANCH").ok(),
        ) {
            (Some(token), Some(owner), Some(repo), Some(branch)) => {
                info!("GitHub API configured");
                Ok(Some(GitHubConfig { token, owner, repo, default_branch: branch }))
            }
            (Some(_), _, _, _) => {
                warn!(
                    "Partial GitHub configuration found. Need GITHUB_OWNER, GITHUB_REPO, and \
                     GITHUB_DEFAULT_BRANCH"
                );
                Ok(None)
            }
            _ => {
                info!("GitHub API not configured");
                Ok(None)
            }
        }
    }

    /// Load X/Twitter configuration
    fn load_x_twitterconfig() -> Result<Option<XTwitterConfig>> {
        match (
            env::var("X_API_KEY").ok(),
            env::var("X_API_SECRET").ok(),
            env::var("X_ACCESS_TOKEN").ok(),
            env::var("X_ACCESS_TOKEN_SECRET").ok(),
            env::var("X_BEARER_TOKEN").ok(),
        ) {
            (
                Some(api_key),
                Some(api_secret),
                Some(access_token),
                Some(access_token_secret),
                Some(bearer_token),
            ) => {
                info!("X (Twitter) API configured");
                Ok(Some(XTwitterConfig {
                    api_key,
                    api_secret,
                    access_token,
                    access_token_secret,
                    bearer_token,
                }))
            }
            _ => {
                info!("X (Twitter) API not configured");
                Ok(None)
            }
        }
    }

    /// Load AI model API keys
    fn load_ai_modelsconfig() -> AiModelsConfig {
        let mut config = AiModelsConfig::default();

        if let Ok(key) = env::var("OPENAI_API_KEY") {
            info!("OpenAI API configured");
            config.openai = Some(key);
        }

        if let Ok(key) = env::var("ANTHROPIC_API_KEY") {
            info!("Anthropic API configured");
            config.anthropic = Some(key);
        }

        if let Ok(key) = env::var("DEEPSEEK_API_KEY") {
            info!("DeepSeek API configured");
            config.deepseek = Some(key);
        }

        if let Ok(key) = env::var("MISTRAL_API_KEY") {
            info!("Mistral API configured");
            config.mistral = Some(key);
        }

        if let Ok(key) = env::var("CODESTRAL_API_KEY") {
            info!("Codestral API configured");
            config.codestral = Some(key);
        }

        if let Ok(key) = env::var("GEMINI_API_KEY") {
            info!("Gemini API configured");
            config.gemini = Some(key);
        }

        if let Ok(key) = env::var("GROK_API_KEY") {
            info!("Grok API configured");
            config.grok = Some(key);
        }

        if let Ok(key) = env::var("COHERE_API_KEY") {
            info!("Cohere API configured");
            config.cohere = Some(key);
        }

        if let Ok(key) = env::var("PERPLEXITY_API_KEY") {
            info!("Perplexity API configured");
            config.perplexity = Some(key);
        }

        config
    }

    /// Load embedding model API keys
    fn load_embedding_modelsconfig() -> EmbeddingModelsConfig {
        let mut config = EmbeddingModelsConfig::default();

        if let Ok(key) = env::var("OPENAI_EMBEDDING_API_KEY") {
            info!("OpenAI Embedding API configured");
            config.openai = Some(key);
        } else if let Ok(key) = env::var("OPENAI_API_KEY") {
            // Use the same key as main OpenAI API if separate embedding key not found
            info!("Using OpenAI API key for embeddings");
            config.openai = Some(key);
        }

        if let Ok(key) = env::var("COHERE_EMBEDDING_API_KEY") {
            info!("Cohere Embedding API configured");
            config.cohere = Some(key);
        } else if let Ok(key) = env::var("COHERE_API_KEY") {
            // Use the same key as main Cohere API if separate embedding key not found
            info!("Using Cohere API key for embeddings");
            config.cohere = Some(key);
        }

        if let Ok(key) = env::var("HUGGINGFACE_EMBEDDING_TOKEN") {
            info!("Hugging Face Embedding API configured");
            config.huggingface = Some(key);
        } else if let Ok(key) = env::var("HUGGINGFACE_TOKEN") {
            // Use the same token for embeddings
            info!("Using Hugging Face token for embeddings");
            config.huggingface = Some(key);
        }

        if let Ok(key) = env::var("SENTENCE_TRANSFORMERS_API_KEY") {
            info!("Sentence Transformers API configured");
            config.sentence_transformers = Some(key);
        }

        if let Ok(key) = env::var("VOYAGE_API_KEY") {
            info!("Voyage AI API configured");
            config.voyage = Some(key);
        }

        if let Ok(key) = env::var("JINA_API_KEY") {
            info!("Jina AI API configured");
            config.jina = Some(key);
        }

        config
    }

    /// Load search API configurations
    fn load_search_apisconfig() -> SearchApisConfig {
        let mut config = SearchApisConfig::default();

        if let Ok(key) = env::var("BRAVE_SEARCH_API_KEY") {
            info!("Brave Search API configured");
            config.brave = Some(key);
        }

        if let (Ok(api_key), Ok(cx)) =
            (env::var("GOOGLE_CUSTOM_SEARCH_API_KEY"), env::var("GOOGLE_CUSTOM_SEARCH_CX"))
        {
            info!("Google Custom Search API configured");
            config.google = Some(GoogleSearchConfig { api_key, cx });
        }

        if let Ok(key) = env::var("BING_API_KEY") {
            if key != "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" {
                // Check for placeholder
                info!("Bing Search API configured");
                config.bing = Some(key);
            }
        }

        if let Ok(key) = env::var("SERPER_API_KEY") {
            info!("Serper API configured");
            config.serper = Some(key);
        }

        if let Ok(key) = env::var("YOU_API_KEY") {
            info!("You.com API configured");
            config.you = Some(key);
        }

        config
    }

    /// Load vector database configuration
    fn load_vector_dbconfig() -> Result<Option<VectorDbConfig>> {
        // Check Pinecone
        if let (Ok(api_key), Ok(environment)) =
            (env::var("PINECONE_API_KEY"), env::var("PINECONE_ENVIRONMENT"))
        {
            info!("Pinecone vector database configured");
            return Ok(Some(VectorDbConfig {
                provider: VectorDbProvider::Pinecone,
                api_key,
                environment: Some(environment),
                url: None,
            }));
        }

        // Check Qdrant
        if let Ok(api_key) = env::var("QDRANT_API_KEY") {
            info!("Qdrant vector database configured");
            return Ok(Some(VectorDbConfig {
                provider: VectorDbProvider::Qdrant,
                api_key,
                environment: None,
                url: env::var("QDRANT_URL").ok(),
            }));
        }

        // Check Weaviate
        if let Ok(api_key) = env::var("WEAVIATE_API_KEY") {
            info!("Weaviate vector database configured");
            return Ok(Some(VectorDbConfig {
                provider: VectorDbProvider::Weaviate,
                api_key,
                environment: None,
                url: env::var("WEAVIATE_URL").ok(),
            }));
        }

        info!("No vector database configured");
        Ok(None)
    }

    /// Load optional service configurations
    fn load_optional_servicesconfig() -> OptionalServicesConfig {
        let mut config = OptionalServicesConfig::default();

        if let Ok(key) = env::var("STACK_EXCHANGE_KEY") {
            if key != "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" {
                // Check for placeholder
                info!("Stack Exchange API configured");
                config.stack_exchange = Some(key);
            }
        }

        if let Ok(key) = env::var("REPLICATE_API_TOKEN") {
            info!("Replicate API configured");
            config.replicate = Some(key);
        }

        if let Ok(key) = env::var("HUGGINGFACE_TOKEN") {
            info!("Hugging Face API configured");
            config.huggingface = Some(key);
        }

        if let Ok(key) = env::var("WOLFRAM_ALPHA_API_KEY") {
            info!("Wolfram Alpha API configured");
            config.wolfram_alpha = Some(key);
        }

        if let Ok(key) = env::var("NEWS_API_KEY") {
            info!("News API configured");
            config.news_api = Some(key);
        }

        if let Ok(key) = env::var("WEATHER_API_KEY") {
            info!("Weather API configured");
            config.weather_api = Some(key);
        }

        config
    }

    /// Validate that critical API keys are present
    pub fn validate(&self) -> Result<()> {
        // At minimum, we need either Ollama or at least one AI model API
        let has_ai_model = self.ai_models.openai.is_some()
            || self.ai_models.anthropic.is_some()
            || self.ai_models.deepseek.is_some()
            || self.ai_models.mistral.is_some()
            || self.ai_models.codestral.is_some()
            || self.ai_models.gemini.is_some()
            || self.ai_models.grok.is_some();

        if !has_ai_model {
            warn!("No AI model API keys configured. Loki will rely on Ollama for language models.");
        }

        // Warn if no search API is configured
        let has_search = self.search.brave.is_some()
            || self.search.google.is_some()
            || self.search.bing.is_some()
            || self.search.serper.is_some()
            || self.search.you.is_some();

        if !has_search {
            warn!("No search API configured. Web search functionality will be limited.");
        }

        Ok(())
    }

    /// Get a summary of configured services
    pub fn summary(&self) -> HashMap<&'static str, bool> {
        let mut summary = HashMap::new();

        summary.insert("GitHub", self.github.is_some());
        summary.insert("X/Twitter", self.x_twitter.is_some());
        summary.insert("OpenAI", self.ai_models.openai.is_some());
        summary.insert("Anthropic", self.ai_models.anthropic.is_some());
        summary.insert("DeepSeek", self.ai_models.deepseek.is_some());
        summary.insert("Mistral", self.ai_models.mistral.is_some());
        summary.insert("Codestral", self.ai_models.codestral.is_some());
        summary.insert("Gemini", self.ai_models.gemini.is_some());
        summary.insert("Grok", self.ai_models.grok.is_some());
        summary.insert("Cohere", self.ai_models.cohere.is_some());
        summary.insert("Perplexity", self.ai_models.perplexity.is_some());
        summary.insert("OpenAI Embeddings", self.embedding_models.openai.is_some());
        summary.insert("Cohere Embeddings", self.embedding_models.cohere.is_some());
        summary.insert("Hugging Face Embeddings", self.embedding_models.huggingface.is_some());
        summary.insert("Voyage AI", self.embedding_models.voyage.is_some());
        summary.insert("Jina AI", self.embedding_models.jina.is_some());
        summary.insert("Brave Search", self.search.brave.is_some());
        summary.insert("Google Search", self.search.google.is_some());
        summary.insert("Bing Search", self.search.bing.is_some());
        summary.insert("Vector Database", self.vector_db.is_some());
        summary.insert("Stack Exchange", self.optional_services.stack_exchange.is_some());
        summary.insert("Replicate", self.optional_services.replicate.is_some());
        summary.insert("Hugging Face", self.optional_services.huggingface.is_some());

        summary
    }

    /// Check if we have at least one AI model configured
    pub fn has_ai_model(&self) -> bool {
        self.ai_models.openai.is_some()
            || self.ai_models.anthropic.is_some()
            || self.ai_models.deepseek.is_some()
            || self.ai_models.mistral.is_some()
            || self.ai_models.codestral.is_some()
            || self.ai_models.gemini.is_some()
            || self.ai_models.grok.is_some()
    }

    /// Check if any API keys are configured
    pub fn has_anyconfigured(&self) -> bool {
        self.github.is_some()
            || self.x_twitter.is_some()
            || self.has_ai_model()
            || self.has_embedding_model()
            || self.search.brave.is_some()
            || self.search.google.is_some()
            || self.search.bing.is_some()
            || self.search.serper.is_some()
            || self.search.you.is_some()
            || self.vector_db.is_some()
            || self.optional_services.stack_exchange.is_some()
            || self.optional_services.replicate.is_some()
            || self.optional_services.huggingface.is_some()
            || self.optional_services.wolfram_alpha.is_some()
            || self.optional_services.news_api.is_some()
            || self.optional_services.weather_api.is_some()
    }

    /// Check if we have at least one embedding model configured
    pub fn has_embedding_model(&self) -> bool {
        self.embedding_models.openai.is_some()
            || self.embedding_models.cohere.is_some()
            || self.embedding_models.huggingface.is_some()
            || self.embedding_models.sentence_transformers.is_some()
            || self.embedding_models.voyage.is_some()
            || self.embedding_models.jina.is_some()
    }

    /// Get the preferred AI model API key (in order of preference)
    pub fn get_preferred_ai_model(&self) -> Option<(&'static str, &str)> {
        if let Some(key) = &self.ai_models.anthropic {
            return Some(("anthropic", key));
        }
        if let Some(key) = &self.ai_models.openai {
            return Some(("openai", key));
        }
        if let Some(key) = &self.ai_models.grok {
            return Some(("grok", key));
        }
        if let Some(key) = &self.ai_models.deepseek {
            return Some(("deepseek", key));
        }
        if let Some(key) = &self.ai_models.gemini {
            return Some(("gemini", key));
        }
        if let Some(key) = &self.ai_models.mistral {
            return Some(("mistral", key));
        }
        if let Some(key) = &self.ai_models.codestral {
            return Some(("codestral", key));
        }
        None
    }

    /// Get the preferred search API key (in order of preference)
    pub fn get_preferred_search_api(&self) -> Option<(&'static str, String)> {
        if let Some(key) = &self.search.brave {
            return Some(("brave", key.clone()));
        }
        if let Some(config) = &self.search.google {
            return Some(("google", format!("{}|{}", config.api_key, config.cx)));
        }
        if let Some(key) = &self.search.bing {
            return Some(("bing", key.clone()));
        }
        if let Some(key) = &self.search.serper {
            return Some(("serper", key.clone()));
        }
        if let Some(key) = &self.search.you {
            return Some(("you", key.clone()));
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apiconfig_summary() {
        let config = ApiKeysConfig {
            github: Some(GitHubConfig {
                token: "test".to_string(),
                owner: "test".to_string(),
                repo: "test".to_string(),
                default_branch: "main".to_string(),
            }),
            x_twitter: None,
            ai_models: AiModelsConfig { openai: Some("test".to_string()), ..Default::default() },
            embedding_models: EmbeddingModelsConfig::default(),
            search: SearchApisConfig::default(),
            vector_db: None,
            optional_services: OptionalServicesConfig::default(),
        };

        let summary = config.summary();
        assert_eq!(summary.get("GitHub"), Some(&true));
        assert_eq!(summary.get("X/Twitter"), Some(&false));
        assert_eq!(summary.get("OpenAI"), Some(&true));
    }
}
