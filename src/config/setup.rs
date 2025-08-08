//! Interactive Setup Module
//!
//! Provides a user-friendly CLI interface for configuring API keys and secrets.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};
use dialoguer::theme::ColorfulTheme;
use dialoguer::{Confirm, Input, MultiSelect, Password, Select};
use directories::ProjectDirs;
use serde::{Deserialize, Serialize};
use tokio::process::Command as TokioCommand;

use super::api_keys::{
    ApiKeysConfig,
    GitHubConfig,
    GoogleSearchConfig,
    VectorDbConfig,
    VectorDbProvider,
    XTwitterConfig,
};

/// Secure configuration storage for API keys and model preferences
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SecureConfig {
    /// API keys configuration
    pub api_keys: ApiKeysConfig,

    /// Model management configuration
    pub modelconfig: ModelManagementConfig,

    /// Configuration metadata
    pub metadata: ConfigMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigMetadata {
    /// Configuration version
    pub version: String,

    /// Last updated timestamp
    pub last_updated: chrono::DateTime<chrono::Utc>,

    /// Configured services count
    pub configured_services: usize,
}

/// Model management configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelManagementConfig {
    /// Task-specific model assignments
    pub task_assignments: HashMap<TaskType, ModelAssignment>,

    /// Available local models
    pub local_models: Vec<String>,

    /// Model preferences
    pub preferences: ModelPreferences,
}

/// Different task types that can be assigned to specific models
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum TaskType {
    /// General reasoning and analysis
    Reasoning,
    /// Code generation and programming
    Coding,
    /// Social media posting and engagement
    SocialPosting,
    /// Creative content generation
    Creative,
    /// Web search and summarization
    Search,
    /// Data analysis and processing
    DataAnalysis,
    /// Image understanding and description
    Vision,
    /// Memory and knowledge management
    Memory,
    /// Tool usage and function calling
    ToolUsage,
    /// Default fallback for unspecified tasks
    Default,
}

/// Model assignment for a specific task
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelAssignment {
    /// Primary model to use (API or local)
    pub primary: ModelReference,

    /// Fallback model if primary fails
    pub fallback: Option<ModelReference>,

    /// Whether to use local models when available
    pub prefer_local: bool,
}

/// Reference to a specific model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelReference {
    /// API-based model (provider:model_name)
    Api { provider: String, model: String },

    /// Local Ollama model
    Local { model: String },

    /// Auto-select best available
    Auto,
}

/// User preferences for model selection
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelPreferences {
    /// Prefer local models over API when available
    pub prefer_local: bool,

    /// Maximum cost per API call (in cents)
    pub max_cost_per_call: Option<f64>,

    /// Preferred model size (small, medium, large)
    pub preferred_size: ModelSize,

    /// Whether to automatically download missing local models
    pub auto_download: bool,
}

/// Model size preference
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub enum ModelSize {
    Small, // <7B parameters
    #[default]
    Medium, // 7B-20B parameters
    Large, // >20B parameters
    Any,   // No preference
}

impl Default for ConfigMetadata {
    fn default() -> Self {
        Self {
            version: "1.0.0".to_string(),
            last_updated: chrono::Utc::now(),
            configured_services: 0,
        }
    }
}

impl SecureConfig {
    /// Load secure configuration from disk
    pub fn load() -> Result<Self> {
        let config_path = Self::config_file_path()?;

        if config_path.exists() {
            let contents = fs::read_to_string(&config_path)
                .context("Failed to read secure configuration file")?;
            let config: Self =
                toml::from_str(&contents).context("Failed to parse secure configuration file")?;
            Ok(config)
        } else {
            Ok(Self::default())
        }
    }

    /// Save secure configuration to disk
    pub fn save(&mut self) -> Result<()> {
        let config_path = Self::config_file_path()?;

        // Update metadata
        self.metadata.last_updated = chrono::Utc::now();
        self.metadata.configured_services = self.countconfigured_services();

        let contents =
            toml::to_string_pretty(self).context("Failed to serialize secure configuration")?;

        if let Some(parent) = config_path.parent() {
            fs::create_dir_all(parent).context("Failed to create configuration directory")?;
        }

        fs::write(&config_path, contents).context("Failed to write secure configuration file")?;

        // Set restrictive permissions (Unix only)
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&config_path)?.permissions();
            perms.set_mode(0o600); // Only owner can read/write
            fs::set_permissions(&config_path, perms)?;
        }

        Ok(())
    }

    /// Get the secure configuration file path
    fn config_file_path() -> Result<PathBuf> {
        let dirs = ProjectDirs::from("dev", "loki", "loki")
            .context("Failed to determine configuration directory")?;
        Ok(dirs.config_dir().join("api_keys.toml"))
    }

    /// Count configured services
    fn countconfigured_services(&self) -> usize {
        let summary = self.api_keys.summary();
        summary.values().filter(|&&configured| configured).count()
    }

    /// Export configuration to environment variables format
    pub fn export_to_env(&self) -> Result<String> {
        let mut env_vars = Vec::new();

        // GitHub
        if let Some(ref github) = self.api_keys.github {
            env_vars.push(format!("export GITHUB_TOKEN=\"{}\"", github.token));
            env_vars.push(format!("export GITHUB_OWNER=\"{}\"", github.owner));
            env_vars.push(format!("export GITHUB_REPO=\"{}\"", github.repo));
            env_vars.push(format!("export GITHUB_DEFAULT_BRANCH=\"{}\"", github.default_branch));
        }

        // X/Twitter
        if let Some(ref x) = self.api_keys.x_twitter {
            env_vars.push(format!("export X_API_KEY=\"{}\"", x.api_key));
            env_vars.push(format!("export X_API_SECRET=\"{}\"", x.api_secret));
            env_vars.push(format!("export X_ACCESS_TOKEN=\"{}\"", x.access_token));
            env_vars.push(format!("export X_ACCESS_TOKEN_SECRET=\"{}\"", x.access_token_secret));
        }

        // AI Models
        if let Some(ref key) = self.api_keys.ai_models.openai {
            env_vars.push(format!("export OPENAI_API_KEY=\"{}\"", key));
        }
        if let Some(ref key) = self.api_keys.ai_models.anthropic {
            env_vars.push(format!("export ANTHROPIC_API_KEY=\"{}\"", key));
        }
        if let Some(ref key) = self.api_keys.ai_models.deepseek {
            env_vars.push(format!("export DEEPSEEK_API_KEY=\"{}\"", key));
        }
        if let Some(ref key) = self.api_keys.ai_models.mistral {
            env_vars.push(format!("export MISTRAL_API_KEY=\"{}\"", key));
        }
        if let Some(ref key) = self.api_keys.ai_models.gemini {
            env_vars.push(format!("export GEMINI_API_KEY=\"{}\"", key));
        }

        // Search APIs
        if let Some(ref key) = self.api_keys.search.brave {
            env_vars.push(format!("export BRAVE_SEARCH_API_KEY=\"{}\"", key));
        }
        if let Some(ref google) = self.api_keys.search.google {
            env_vars.push(format!("export GOOGLE_CUSTOM_SEARCH_API_KEY=\"{}\"", google.api_key));
            env_vars.push(format!("export GOOGLE_CUSTOM_SEARCH_CX=\"{}\"", google.cx));
        }

        Ok(env_vars.join("\n"))
    }
}

/// Interactive setup wizard
pub struct SetupWizard {
    pub config: SecureConfig,
    theme: ColorfulTheme,
}

impl SetupWizard {
    /// Create a new setup wizard
    pub fn new() -> Result<Self> {
        let config = SecureConfig::load().unwrap_or_default();
        Ok(Self { config, theme: ColorfulTheme::default() })
    }

    /// Run the interactive setup wizard
    pub async fn run(&mut self) -> Result<()> {
        self.show_welcome();

        loop {
            self.show_status();
            let choice = self.show_main_menu()?;

            match choice {
                0 => break, // Exit
                1 => self.setup_github()?,
                2 => self.setup_x_twitter()?,
                3 => self.setup_ai_models()?,
                4 => self.setup_embedding_models()?,
                5 => self.setup_model_management().await?,
                6 => self.setup_search_apis()?,
                7 => self.setup_vector_database()?,
                8 => self.setup_optional_services()?,
                9 => self.showconfiguration()?,
                10 => self.exportconfiguration()?,
                11 => self.resetconfiguration()?,
                _ => continue,
            }
        }

        self.config.save()?;
        self.show_completion();
        Ok(())
    }

    fn show_welcome(&self) {
        println!("\n🚀 Welcome to Loki API Setup Wizard!");
        println!("=====================================");
        println!("This wizard will help you configure API keys for all Loki services.");
        println!("Your credentials will be stored securely in your local configuration.\n");
    }

    fn show_status(&self) {
        let summary = self.config.api_keys.summary();
        let configured_count = summary.values().filter(|&&configured| configured).count();
        let total_services = summary.len();

        println!(
            "📊 Configuration Status: {}/{} services configured",
            configured_count, total_services
        );
        println!("{}", "─".repeat(50));

        for (service, configured) in summary.iter() {
            let status = if *configured { "✅" } else { "❌" };
            println!("{} {}", status, service);
        }
        println!();
    }

    fn show_main_menu(&self) -> Result<usize> {
        let options = vec![
            "Exit setup",
            "🐙 Configure GitHub Integration",
            "🐦 Configure X/Twitter Integration",
            "🤖 Configure AI Model Providers",
            "🔤 Configure Embedding Models",
            "🧠 Model Management & Assignment",
            "🔍 Configure Search APIs",
            "🗄️  Configure Vector Database",
            "⚙️  Configure Optional Services",
            "📋 Show Current Configuration",
            "📤 Export Configuration (.env)",
            "🗑️  Reset All Configuration",
        ];

        Select::with_theme(&self.theme)
            .with_prompt("What would you like to configure?")
            .items(&options)
            .default(0)
            .interact()
            .context("Failed to show main menu")
    }

    fn setup_github(&mut self) -> Result<()> {
        println!("\n🐙 GitHub Integration Setup");
        println!("============================");
        println!("To use GitHub integration, you need a Personal Access Token.");
        println!("Create one at: https://github.com/settings/tokens\n");

        if self.config.api_keys.github.is_some() {
            let reconfigure = Confirm::with_theme(&self.theme)
                .with_prompt("GitHub is already configured. Reconfigure?")
                .default(false)
                .interact()?;

            if !reconfigure {
                return Ok(());
            }
        }

        let token: String = Password::with_theme(&self.theme)
            .with_prompt("GitHub Personal Access Token")
            .interact()?;

        let owner: String = Input::with_theme(&self.theme)
            .with_prompt("GitHub Username/Organization")
            .interact()?;

        let repo: String = Input::with_theme(&self.theme)
            .with_prompt("Default Repository Name")
            .default("loki".to_string())
            .interact()?;

        let default_branch: String = Input::with_theme(&self.theme)
            .with_prompt("Default Branch")
            .default("main".to_string())
            .interact()?;

        self.config.api_keys.github = Some(GitHubConfig { token, owner, repo, default_branch });

        println!("✅ GitHub configuration saved!");
        Ok(())
    }

    fn setup_x_twitter(&mut self) -> Result<()> {
        println!("\n🐦 X/Twitter Integration Setup");
        println!("===============================");
        println!("To use X/Twitter integration, you need API credentials from:");
        println!("https://developer.twitter.com/en/portal/dashboard\n");

        if self.config.api_keys.x_twitter.is_some() {
            let reconfigure = Confirm::with_theme(&self.theme)
                .with_prompt("X/Twitter is already configured. Reconfigure?")
                .default(false)
                .interact()?;

            if !reconfigure {
                return Ok(());
            }
        }

        let api_key: String =
            Password::with_theme(&self.theme).with_prompt("API Key").interact()?;

        let api_secret: String =
            Password::with_theme(&self.theme).with_prompt("API Secret").interact()?;

        let access_token: String =
            Password::with_theme(&self.theme).with_prompt("Access Token").interact()?;

        let access_token_secret: String =
            Password::with_theme(&self.theme).with_prompt("Access Token Secret").interact()?;

        let bearer_token: String =
            Password::with_theme(&self.theme).with_prompt("Bearer Token").interact()?;
        
        
        self.config.api_keys.x_twitter =
            Some(XTwitterConfig { api_key, api_secret, access_token, access_token_secret, bearer_token });

        println!("✅ X/Twitter configuration saved!");
        Ok(())
    }

    fn setup_ai_models(&mut self) -> Result<()> {
        println!("\n🤖 AI Model Providers Setup");
        println!("============================");

        let providers = vec![
            ("OpenAI", "openai"),
            ("Anthropic (Claude)", "anthropic"),
            ("DeepSeek", "deepseek"),
            ("Mistral", "mistral"),
            ("Codestral", "codestral"),
            ("Google Gemini", "gemini"),
            ("xAI Grok", "grok"),
            ("Cohere", "cohere"),
            ("Perplexity", "perplexity"),
            ("Back to main menu", "back"),
        ];

        loop {
            let choice = Select::with_theme(&self.theme)
                .with_prompt("Which AI provider would you like to configure?")
                .items(&providers.iter().map(|(name, _)| name).collect::<Vec<_>>())
                .default(0)
                .interact()?;

            if providers[choice].1 == "back" {
                break;
            }

            let (provider_name, provider_key) = providers[choice];

            // Check if already configured
            let current_key = match provider_key {
                "openai" => &self.config.api_keys.ai_models.openai,
                "anthropic" => &self.config.api_keys.ai_models.anthropic,
                "deepseek" => &self.config.api_keys.ai_models.deepseek,
                "mistral" => &self.config.api_keys.ai_models.mistral,
                "codestral" => &self.config.api_keys.ai_models.codestral,
                "gemini" => &self.config.api_keys.ai_models.gemini,
                "grok" => &self.config.api_keys.ai_models.grok,
                "cohere" => &self.config.api_keys.ai_models.cohere,
                "perplexity" => &self.config.api_keys.ai_models.perplexity,
                _ => &None,
            };

            if current_key.is_some() {
                let reconfigure = Confirm::with_theme(&self.theme)
                    .with_prompt(&format!("{} is already configured. Reconfigure?", provider_name))
                    .default(false)
                    .interact()?;

                if !reconfigure {
                    continue;
                }
            }

            println!("\n{} API Key Setup", provider_name);
            let api_key: String = Password::with_theme(&self.theme)
                .with_prompt(&format!("{} API Key", provider_name))
                .interact()?;

            // Save the API key
            match provider_key {
                "openai" => self.config.api_keys.ai_models.openai = Some(api_key),
                "anthropic" => self.config.api_keys.ai_models.anthropic = Some(api_key),
                "deepseek" => self.config.api_keys.ai_models.deepseek = Some(api_key),
                "mistral" => self.config.api_keys.ai_models.mistral = Some(api_key),
                "codestral" => self.config.api_keys.ai_models.codestral = Some(api_key),
                "gemini" => self.config.api_keys.ai_models.gemini = Some(api_key),
                "grok" => self.config.api_keys.ai_models.grok = Some(api_key),
                "cohere" => self.config.api_keys.ai_models.cohere = Some(api_key),
                "perplexity" => self.config.api_keys.ai_models.perplexity = Some(api_key),
                _ => {}
            }

            println!("✅ {} configuration saved!", provider_name);
        }

        Ok(())
    }

    fn setup_embedding_models(&mut self) -> Result<()> {
        println!("\n🔤 Embedding Models Setup");
        println!("=========================");
        println!("Embedding models are used for vector search and semantic similarity.");
        println!("Configure providers for text embeddings used in vector databases.\n");

        let providers = vec![
            ("OpenAI Embeddings", "openai"),
            ("Cohere Embeddings", "cohere"),
            ("Hugging Face Embeddings", "huggingface"),
            ("Sentence Transformers", "sentence_transformers"),
            ("Voyage AI", "voyage"),
            ("Jina AI", "jina"),
            ("Back to main menu", "back"),
        ];

        loop {
            let choice = Select::with_theme(&self.theme)
                .with_prompt("Which embedding provider would you like to configure?")
                .items(&providers.iter().map(|(name, _)| name).collect::<Vec<_>>())
                .default(0)
                .interact()?;

            if providers[choice].1 == "back" {
                break;
            }

            let (provider_name, provider_key) = providers[choice];

            // Check if already configured
            let current_key = match provider_key {
                "openai" => &self.config.api_keys.embedding_models.openai,
                "cohere" => &self.config.api_keys.embedding_models.cohere,
                "huggingface" => &self.config.api_keys.embedding_models.huggingface,
                "sentence_transformers" => {
                    &self.config.api_keys.embedding_models.sentence_transformers
                }
                "voyage" => &self.config.api_keys.embedding_models.voyage,
                "jina" => &self.config.api_keys.embedding_models.jina,
                _ => &None,
            };

            if current_key.is_some() {
                let reconfigure = Confirm::with_theme(&self.theme)
                    .with_prompt(&format!("{} is already configured. Reconfigure?", provider_name))
                    .default(false)
                    .interact()?;

                if !reconfigure {
                    continue;
                }
            }

            println!("\n{} Setup", provider_name);

            // Show provider-specific instructions
            match provider_key {
                "openai" => {
                    println!("Get your API key from: https://platform.openai.com/api-keys");
                    println!(
                        "Supports models: text-embedding-3-small, text-embedding-3-large, \
                         text-embedding-ada-002"
                    );
                }
                "cohere" => {
                    println!("Get your API key from: https://dashboard.cohere.ai/api-keys");
                    println!("Supports models: embed-english-v3.0, embed-multilingual-v3.0");
                }
                "huggingface" => {
                    println!("Get your token from: https://huggingface.co/settings/tokens");
                    println!("Access to thousands of embedding models via Inference API");
                }
                "sentence_transformers" => {
                    println!("API key for SentenceTransformers cloud service (if using hosted)");
                    println!("Many models available for local deployment as well");
                }
                "voyage" => {
                    println!("Get your API key from: https://www.voyageai.com/");
                    println!("High-quality embedding models optimized for retrieval");
                }
                "jina" => {
                    println!("Get your API key from: https://jina.ai/");
                    println!("Specialized in multimodal embeddings and neural search");
                }
                _ => {}
            }

            let api_key: String = Password::with_theme(&self.theme)
                .with_prompt(&format!("{} API Key", provider_name))
                .interact()?;

            // Save the API key
            match provider_key {
                "openai" => self.config.api_keys.embedding_models.openai = Some(api_key),
                "cohere" => self.config.api_keys.embedding_models.cohere = Some(api_key),
                "huggingface" => self.config.api_keys.embedding_models.huggingface = Some(api_key),
                "sentence_transformers" => {
                    self.config.api_keys.embedding_models.sentence_transformers = Some(api_key)
                }
                "voyage" => self.config.api_keys.embedding_models.voyage = Some(api_key),
                "jina" => self.config.api_keys.embedding_models.jina = Some(api_key),
                _ => {}
            }

            println!("✅ {} configuration saved!", provider_name);
        }

        Ok(())
    }

    async fn setup_model_management(&mut self) -> Result<()> {
        println!("\n🧠 Model Management & Task Assignment");
        println!("=====================================");
        println!("Configure which models to use for different tasks.");
        println!("You can use API models, local Ollama models, or a hybrid approach.\n");

        let options = vec![
            "📊 Show Current Model Status",
            "🔍 Scan for Local Models",
            "⬇️  Download Ollama Models",
            "🎯 Configure Task Assignments",
            "⚙️  Set Model Preferences",
            "🔄 Auto-Configure (Recommended)",
            "Back to main menu",
        ];

        loop {
            let choice = Select::with_theme(&self.theme)
                .with_prompt("What would you like to do?")
                .items(&options)
                .default(0)
                .interact()?;

            match choice {
                0 => self.show_model_status().await?,
                1 => self.scan_local_models(true).await?,
                2 => self.download_models().await?,
                3 => self.configure_task_assignments().await?,
                4 => self.set_model_preferences()?,
                5 => self.autoconfigure_models().await?,
                6 => break,
                _ => continue,
            }
        }

        Ok(())
    }

    pub async fn show_model_status(&self) -> Result<()> {
        println!("\n📊 Current Model Status");
        println!("=======================");

        // Show available API providers
        println!("\n🌐 API Models Available:");
        let api_keys = &self.config.api_keys;

        if let Some(_) = &api_keys.ai_models.openai {
            println!("  ✅ OpenAI: gpt-4, gpt-3.5-turbo, gpt-4-turbo");
        } else {
            println!("  ❌ OpenAI: Not configured");
        }

        if let Some(_) = &api_keys.ai_models.anthropic {
            println!("  ✅ Anthropic: claude-3-opus, claude-3-sonnet, claude-3-haiku");
        } else {
            println!("  ❌ Anthropic: Not configured");
        }

        if let Some(_) = &api_keys.ai_models.deepseek {
            println!("  ✅ DeepSeek: deepseek-chat, deepseek-coder");
        } else {
            println!("  ❌ DeepSeek: Not configured");
        }

        if let Some(_) = &api_keys.ai_models.mistral {
            println!("  ✅ Mistral: mistral-large, mistral-medium");
        } else {
            println!("  ❌ Mistral: Not configured");
        }

        if let Some(_) = &api_keys.ai_models.codestral {
            println!("  ✅ Codestral: codestral-latest");
        } else {
            println!("  ❌ Codestral: Not configured");
        }

        // Show local models
        println!("\n🏠 Local Models (Ollama):");
        if self.config.modelconfig.local_models.is_empty() {
            println!("  ❌ No local models found. Run 'Scan for Local Models' first.");
        } else {
            for model in &self.config.modelconfig.local_models {
                println!("  ✅ {}", model);
            }
        }

        // Show current task assignments
        println!("\n🎯 Current Task Assignments:");
        if self.config.modelconfig.task_assignments.is_empty() {
            println!("  ❌ No task assignments configured. Using defaults.");
        } else {
            for (task, assignment) in &self.config.modelconfig.task_assignments {
                println!("  {:?}: {:?}", task, assignment.primary);
            }
        }

        println!("\nPress Enter to continue...");
        Input::<String>::with_theme(&self.theme).with_prompt("").allow_empty(true).interact()?;

        Ok(())
    }

    pub async fn scan_local_models(&mut self, do_print: bool) -> Result<()> {
        if do_print {
            println!("\n🔍 Scanning for Local Ollama Models...");
        }
        // Check if Ollama is available
        let output = TokioCommand::new("ollama").arg("list").output().await;

        match output {
            Ok(output) => {
                if output.status.success() {
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    let mut models = Vec::new();

                    // Parse Ollama list output
                    for line in stdout.lines().skip(1) {
                        // Skip header
                        if let Some(model_name) = line.split_whitespace().next() {
                            if !model_name.is_empty() && model_name != "NAME" {
                                models.push(model_name.to_string());
                            }
                        }
                    }

                    if models.is_empty() && do_print {
                        println!(
                            "❌ No local models found. You may need to download some models first."
                        );
                    } else if do_print {
                        println!("✅ Found {} local models:", models.len());
                        for model in &models {
                            println!("  • {}", model);
                        }
                        self.config.modelconfig.local_models = models;
                        println!("\n✅ Local model list updated!");
                    } else {
                        self.config.modelconfig.local_models = models;
                    }
                } else if do_print {
                    println!("❌ Failed to get model list from Ollama");
                }
            }
            Err(_) => {
                println!("❌ Ollama not found. Please install Ollama first:");
                println!("   Visit: https://ollama.ai/");
                println!("   Or run: curl -fsSL https://ollama.ai/install.sh | sh");
            }
        }

        if do_print {
            println!("\nPress Enter to continue...");
            Input::<String>::with_theme(&self.theme).with_prompt("").allow_empty(true).interact()?;
        }
        Ok(())
    }

    async fn download_models(&mut self) -> Result<()> {
        println!("\n⬇️  Download Ollama Models");
        println!("==========================");

        let recommended_models = vec![
            ("deepseek-coder:6.7b", "🔥 Excellent for coding and reasoning (6.7B)"),
            ("llama3.2:3b", "⚡ Fast general-purpose model (3B)"),
            ("gemma2:9b", "🎯 Google's efficient model (9B)"),
            ("qwen2.5:7b", "🧠 Strong reasoning capabilities (7B)"),
            ("mistral:7b", "🚀 Fast and capable (7B)"),
            ("codellama:7b", "💻 Meta's code specialist (7B)"),
            ("llama3.2:1b", "⚡ Ultra-fast tiny model (1B)"),
            ("nomic-embed-text", "📊 Text embeddings for vector search"),
        ];

        println!("Recommended models for different tasks:\n");
        for (i, (model, description)) in recommended_models.iter().enumerate() {
            println!("  {}. {} - {}", i + 1, model, description);
        }

        println!("\nSelect models to download (use Space to select, Enter to confirm):");

        let selections = MultiSelect::with_theme(&self.theme)
            .with_prompt("Which models would you like to download?")
            .items(
                &recommended_models
                    .iter()
                    .map(|(model, desc)| format!("{} - {}", model, desc))
                    .collect::<Vec<_>>(),
            )
            .interact()?;

        if selections.is_empty() {
            println!("No models selected.");
            return Ok(());
        }

        for selection in selections {
            let (model, _) = recommended_models[selection];
            println!("\n📥 Downloading {}...", model);

            let output = TokioCommand::new("ollama").arg("pull").arg(model).output().await;

            match output {
                Ok(output) => {
                    if output.status.success() {
                        println!("✅ Successfully downloaded {}", model);
                    } else {
                        println!("❌ Failed to download {}", model);
                        let stderr = String::from_utf8_lossy(&output.stderr);
                        if !stderr.is_empty() {
                            println!("   Error: {}", stderr);
                        }
                    }
                }
                Err(e) => {
                    println!("❌ Error downloading {}: {}", model, e);
                }
            }
        }

        // Refresh local models list
        self.scan_local_models(false).await?;

        Ok(())
    }

    async fn configure_task_assignments(&mut self) -> Result<()> {
        println!("\n🎯 Configure Task Assignments");
        println!("=============================");
        println!("Assign specific models to different types of tasks.\n");

        let tasks = vec![
            (TaskType::Reasoning, "🧠 General reasoning and analysis"),
            (TaskType::Coding, "💻 Code generation and programming"),
            (TaskType::SocialPosting, "📱 Social media posting"),
            (TaskType::Creative, "🎨 Creative content generation"),
            (TaskType::Search, "🔍 Web search and summarization"),
            (TaskType::DataAnalysis, "📊 Data analysis and processing"),
            (TaskType::Memory, "🧩 Memory and knowledge management"),
            (TaskType::ToolUsage, "🔧 Tool usage and function calling"),
            (TaskType::Default, "⚙️  Default/fallback for other tasks"),
        ];

        for (task_type, description) in tasks {
            println!("\n{}", description);

            let current = self.config.modelconfig.task_assignments.get(&task_type);
            if let Some(assignment) = current {
                println!("Current: {:?}", assignment.primary);
            } else {
                println!("Current: Not configured");
            }

            let assign = Confirm::with_theme(&self.theme)
                .with_prompt("Configure this task?")
                .default(false)
                .interact()?;

            if assign {
                let assignment = self.select_model_for_task(&task_type).await?;
                self.config.modelconfig.task_assignments.insert(task_type, assignment);
            }
        }

        println!("\n✅ Task assignments updated!");
        Ok(())
    }

    async fn select_model_for_task(&self, task_type: &TaskType) -> Result<ModelAssignment> {
        let mut options: Vec<String> = Vec::new();

        // Add API models
        if let Some(_) = &self.config.api_keys.ai_models.openai {
            options.push("🌐 OpenAI GPT-4".to_string());
            options.push("🌐 OpenAI GPT-3.5-Turbo".to_string());
        }

        if let Some(_) = &self.config.api_keys.ai_models.anthropic {
            options.push("🌐 Anthropic Claude-3-Opus".to_string());
            options.push("🌐 Anthropic Claude-3-Sonnet".to_string());
        }

        if let Some(_) = &self.config.api_keys.ai_models.deepseek {
            options.push("🌐 DeepSeek Chat".to_string());
            options.push("🌐 DeepSeek Coder".to_string());
        }

        if let Some(_) = &self.config.api_keys.ai_models.codestral {
            options.push("🌐 Codestral Latest".to_string());
        }

        // Add local models
        for model in &self.config.modelconfig.local_models {
            options.push(format!("🏠 Local: {}", model));
        }

        // Add auto option
        options.push("🤖 Auto-select best available".to_string());

        if options.is_empty() {
            println!("❌ No models available. Configure API keys or download local models first.");
            return Ok(ModelAssignment {
                primary: ModelReference::Auto,
                fallback: None,
                prefer_local: false,
            });
        }

        let choice = Select::with_theme(&self.theme)
            .with_prompt(&format!("Select model for {:?}", task_type))
            .items(&options.iter().map(|s| s.as_str()).collect::<Vec<_>>())
            .default(0)
            .interact()?;

        let primary = if choice == options.len() - 1 {
            ModelReference::Auto
        } else {
            let selected = &options[choice];
            if selected.starts_with("🌐") {
                // Parse API model
                if selected.contains("OpenAI GPT-4") {
                    ModelReference::Api {
                        provider: "openai".to_string(),
                        model: "gpt-4".to_string(),
                    }
                } else if selected.contains("OpenAI GPT-3.5") {
                    ModelReference::Api {
                        provider: "openai".to_string(),
                        model: "gpt-3.5-turbo".to_string(),
                    }
                } else if selected.contains("Claude-3-Opus") {
                    ModelReference::Api {
                        provider: "anthropic".to_string(),
                        model: "claude-3-opus-20240229".to_string(),
                    }
                } else if selected.contains("Claude-3-Sonnet") {
                    ModelReference::Api {
                        provider: "anthropic".to_string(),
                        model: "claude-3-sonnet-20240229".to_string(),
                    }
                } else if selected.contains("DeepSeek Chat") {
                    ModelReference::Api {
                        provider: "deepseek".to_string(),
                        model: "deepseek-chat".to_string(),
                    }
                } else if selected.contains("DeepSeek Coder") {
                    ModelReference::Api {
                        provider: "deepseek".to_string(),
                        model: "deepseek-coder".to_string(),
                    }
                } else if selected.contains("Codestral") {
                    ModelReference::Api {
                        provider: "codestral".to_string(),
                        model: "codestral-latest".to_string(),
                    }
                } else {
                    ModelReference::Auto
                }
            } else if selected.starts_with("🏠") {
                // Parse local model
                let model_name = selected.replace("🏠 Local: ", "");
                ModelReference::Local { model: model_name }
            } else {
                ModelReference::Auto
            }
        };

        // Ask about fallback
        let use_fallback = Confirm::with_theme(&self.theme)
            .with_prompt("Configure a fallback model?")
            .default(false)
            .interact()?;

        let fallback = if use_fallback {
            // Show different options for fallback
            let mut fallback_options = options.clone();
            fallback_options.retain(|x| x != &options[choice]); // Remove the primary choice

            if !fallback_options.is_empty() {
                let _fallback_choice = Select::with_theme(&self.theme)
                    .with_prompt("Select fallback model")
                    .items(&fallback_options.iter().map(|s| s.as_str()).collect::<Vec<_>>())
                    .default(0)
                    .interact()?;

                // Similar parsing logic as above
                Some(ModelReference::Auto) // Simplified for now
            } else {
                None
            }
        } else {
            None
        };

        let prefer_local = Confirm::with_theme(&self.theme)
            .with_prompt("Prefer local models when available?")
            .default(true)
            .interact()?;

        Ok(ModelAssignment { primary, fallback, prefer_local })
    }

    fn set_model_preferences(&mut self) -> Result<()> {
        println!("\n⚙️  Model Preferences");
        println!("====================");

        let prefer_local = Confirm::with_theme(&self.theme)
            .with_prompt("Prefer local models over API models when available?")
            .default(self.config.modelconfig.preferences.prefer_local)
            .interact()?;

        let auto_download = Confirm::with_theme(&self.theme)
            .with_prompt("Automatically download missing local models?")
            .default(self.config.modelconfig.preferences.auto_download)
            .interact()?;

        let size_options = vec![
            "Small (fast, <7B)",
            "Medium (balanced, 7B-20B)",
            "Large (powerful, >20B)",
            "Any (no preference)",
        ];
        let size_choice = Select::with_theme(&self.theme)
            .with_prompt("Preferred model size")
            .items(&size_options)
            .default(1) // Medium
            .interact()?;

        let preferred_size = match size_choice {
            0 => ModelSize::Small,
            1 => ModelSize::Medium,
            2 => ModelSize::Large,
            3 => ModelSize::Any,
            _ => ModelSize::Medium,
        };

        self.config.modelconfig.preferences = ModelPreferences {
            prefer_local,
            max_cost_per_call: None, // Could add this later
            preferred_size,
            auto_download,
        };

        println!("✅ Model preferences updated!");
        Ok(())
    }

    async fn autoconfigure_models(&mut self) -> Result<()> {
        println!("\n🔄 Auto-Configuring Models...");
        println!("===============================");

        // First scan for local models
        self.scan_local_models(true).await?;

        // Set up smart defaults based on available models and APIs
        println!("🤖 Setting up intelligent defaults...");

        let mut assignments = HashMap::new();

        // Configure based on what's available
        if let Some(_) = &self.config.api_keys.ai_models.deepseek {
            // DeepSeek is great for reasoning and coding
            assignments.insert(
                TaskType::Reasoning,
                ModelAssignment {
                    primary: ModelReference::Api {
                        provider: "deepseek".to_string(),
                        model: "deepseek-chat".to_string(),
                    },
                    fallback: None,
                    prefer_local: false,
                },
            );
            assignments.insert(
                TaskType::Coding,
                ModelAssignment {
                    primary: ModelReference::Api {
                        provider: "deepseek".to_string(),
                        model: "deepseek-coder".to_string(),
                    },
                    fallback: None,
                    prefer_local: false,
                },
            );
        }

        if let Some(_) = &self.config.api_keys.ai_models.anthropic {
            // Claude is excellent for creative tasks
            assignments.insert(
                TaskType::Creative,
                ModelAssignment {
                    primary: ModelReference::Api {
                        provider: "anthropic".to_string(),
                        model: "claude-3-sonnet-20240229".to_string(),
                    },
                    fallback: None,
                    prefer_local: false,
                },
            );
        }

        if let Some(_) = &self.config.api_keys.ai_models.codestral {
            // Codestral for coding if DeepSeek not available
            if !assignments.contains_key(&TaskType::Coding) {
                assignments.insert(
                    TaskType::Coding,
                    ModelAssignment {
                        primary: ModelReference::Api {
                            provider: "codestral".to_string(),
                            model: "codestral-latest".to_string(),
                        },
                        fallback: None,
                        prefer_local: false,
                    },
                );
            }
        }

        // Use local models for social posting if available
        if !self.config.modelconfig.local_models.is_empty() {
            let local_model = self.config.modelconfig.local_models[0].clone();
            assignments.insert(
                TaskType::SocialPosting,
                ModelAssignment {
                    primary: ModelReference::Local { model: local_model },
                    fallback: None,
                    prefer_local: true,
                },
            );
        }

        // Set auto for remaining tasks
        for task_type in [
            TaskType::Search,
            TaskType::DataAnalysis,
            TaskType::Memory,
            TaskType::ToolUsage,
            TaskType::Default,
        ] {
            if !assignments.contains_key(&task_type) {
                assignments.insert(
                    task_type,
                    ModelAssignment {
                        primary: ModelReference::Auto,
                        fallback: None,
                        prefer_local: true,
                    },
                );
            }
        }

        self.config.modelconfig.task_assignments = assignments;

        // Set smart preferences
        self.config.modelconfig.preferences = ModelPreferences {
            prefer_local: true,
            max_cost_per_call: Some(10.0), // 10 cents max per call
            preferred_size: ModelSize::Medium,
            auto_download: false, // Let user decide
        };

        println!("✅ Auto-configuration complete!");
        println!("📋 Summary:");
        for (task, assignment) in &self.config.modelconfig.task_assignments {
            println!("  {:?}: {:?}", task, assignment.primary);
        }

        Ok(())
    }

    fn setup_search_apis(&mut self) -> Result<()> {
        println!("\n🔍 Search APIs Setup");
        println!("====================");

        let providers = vec![
            ("Brave Search", "brave"),
            ("Google Custom Search", "google"),
            ("Bing Search", "bing"),
            ("Serper API", "serper"),
            ("You.com Search", "you"),
            ("Back to main menu", "back"),
        ];

        loop {
            let choice = Select::with_theme(&self.theme)
                .with_prompt("Which search provider would you like to configure?")
                .items(&providers.iter().map(|(name, _)| name).collect::<Vec<_>>())
                .default(0)
                .interact()?;

            if providers[choice].1 == "back" {
                break;
            }

            let (provider_name, provider_key) = providers[choice];

            match provider_key {
                "brave" => {
                    if self.config.api_keys.search.brave.is_some() {
                        let reconfigure = Confirm::with_theme(&self.theme)
                            .with_prompt("Brave Search is already configured. Reconfigure?")
                            .default(false)
                            .interact()?;
                        if !reconfigure {
                            continue;
                        }
                    }

                    println!("Get your API key from: https://api.search.brave.com/");
                    let api_key: String = Password::with_theme(&self.theme)
                        .with_prompt("Brave Search API Key")
                        .interact()?;

                    self.config.api_keys.search.brave = Some(api_key);
                }
                "google" => {
                    if self.config.api_keys.search.google.is_some() {
                        let reconfigure = Confirm::with_theme(&self.theme)
                            .with_prompt("Google Search is already configured. Reconfigure?")
                            .default(false)
                            .interact()?;
                        if !reconfigure {
                            continue;
                        }
                    }

                    println!("Get your API key from: https://console.developers.google.com/");
                    println!("Create a Custom Search Engine at: https://cse.google.com/");

                    let api_key: String = Password::with_theme(&self.theme)
                        .with_prompt("Google Custom Search API Key")
                        .interact()?;

                    let cx: String = Input::with_theme(&self.theme)
                        .with_prompt("Custom Search Engine ID (CX)")
                        .interact()?;

                    self.config.api_keys.search.google = Some(GoogleSearchConfig { api_key, cx });
                }
                "bing" => {
                    if self.config.api_keys.search.bing.is_some() {
                        let reconfigure = Confirm::with_theme(&self.theme)
                            .with_prompt("Bing Search is already configured. Reconfigure?")
                            .default(false)
                            .interact()?;
                        if !reconfigure {
                            continue;
                        }
                    }

                    println!("Get your API key from: https://www.microsoft.com/en-us/bing/apis/bing-web-search-api");
                    let api_key: String = Password::with_theme(&self.theme)
                        .with_prompt("Bing Search API Key")
                        .interact()?;

                    self.config.api_keys.search.bing = Some(api_key);
                }
                "serper" => {
                    if self.config.api_keys.search.serper.is_some() {
                        let reconfigure = Confirm::with_theme(&self.theme)
                            .with_prompt("Serper is already configured. Reconfigure?")
                            .default(false)
                            .interact()?;
                        if !reconfigure {
                            continue;
                        }
                    }

                    println!("Get your API key from: https://serper.dev/");
                    let api_key: String = Password::with_theme(&self.theme)
                        .with_prompt("Serper API Key")
                        .interact()?;

                    self.config.api_keys.search.serper = Some(api_key);
                }
                "you" => {
                    if self.config.api_keys.search.you.is_some() {
                        let reconfigure = Confirm::with_theme(&self.theme)
                            .with_prompt("You.com Search is already configured. Reconfigure?")
                            .default(false)
                            .interact()?;
                        if !reconfigure {
                            continue;
                        }
                    }

                    println!("Get your API key from: https://api.you.com/");
                    let api_key: String = Password::with_theme(&self.theme)
                        .with_prompt("You.com API Key")
                        .interact()?;

                    self.config.api_keys.search.you = Some(api_key);
                }
                _ => continue,
            }

            println!("✅ {} configuration saved!", provider_name);
        }

        Ok(())
    }

    fn setup_vector_database(&mut self) -> Result<()> {
        println!("\n🗄️ Vector Database Setup");
        println!("=========================");

        if self.config.api_keys.vector_db.is_some() {
            let reconfigure = Confirm::with_theme(&self.theme)
                .with_prompt("Vector database is already configured. Reconfigure?")
                .default(false)
                .interact()?;

            if !reconfigure {
                return Ok(());
            }
        }

        let providers = vec!["Pinecone", "Qdrant", "Weaviate", "Skip"];
        let choice = Select::with_theme(&self.theme)
            .with_prompt("Which vector database would you like to use?")
            .items(&providers)
            .default(0)
            .interact()?;

        match choice {
            0 => {
                // Pinecone
                println!("Get your API key from: https://www.pinecone.io/");
                let api_key: String =
                    Password::with_theme(&self.theme).with_prompt("Pinecone API Key").interact()?;

                let environment: String = Input::with_theme(&self.theme)
                    .with_prompt("Pinecone Environment")
                    .default("us-east-1-aws".to_string())
                    .interact()?;

                self.config.api_keys.vector_db = Some(VectorDbConfig {
                    provider: VectorDbProvider::Pinecone,
                    api_key,
                    environment: Some(environment),
                    url: None,
                });
            }
            1 => {
                // Qdrant
                let api_key: String =
                    Password::with_theme(&self.theme).with_prompt("Qdrant API Key").interact()?;

                let url: String = Input::with_theme(&self.theme)
                    .with_prompt("Qdrant URL")
                    .default("https://your-cluster.qdrant.io".to_string())
                    .interact()?;

                self.config.api_keys.vector_db = Some(VectorDbConfig {
                    provider: VectorDbProvider::Qdrant,
                    api_key,
                    environment: None,
                    url: Some(url),
                });
            }
            2 => {
                // Weaviate
                let api_key: String =
                    Password::with_theme(&self.theme).with_prompt("Weaviate API Key").interact()?;

                let url: String = Input::with_theme(&self.theme)
                    .with_prompt("Weaviate URL")
                    .default("https://your-cluster.weaviate.network".to_string())
                    .interact()?;

                self.config.api_keys.vector_db = Some(VectorDbConfig {
                    provider: VectorDbProvider::Weaviate,
                    api_key,
                    environment: None,
                    url: Some(url),
                });
            }
            _ => return Ok(()),
        }

        println!("✅ Vector database configuration saved!");
        Ok(())
    }

    fn setup_optional_services(&mut self) -> Result<()> {
        println!("\n⚙️ Optional Services Setup");
        println!("===========================");

        let services = vec![
            ("Stack Exchange API", "stack_exchange", "https://api.stackexchange.com/"),
            ("Replicate API", "replicate", "https://replicate.com/account/api-tokens"),
            ("Hugging Face", "huggingface", "https://huggingface.co/settings/tokens"),
            ("Wolfram Alpha", "wolfram", "https://developer.wolframalpha.com/"),
            ("News API", "news", "https://newsapi.org/"),
            ("Weather API", "weather", "https://openweathermap.org/api"),
            ("Back to main menu", "back", ""),
        ];

        loop {
            let choice = Select::with_theme(&self.theme)
                .with_prompt("Which optional service would you like to configure?")
                .items(&services.iter().map(|(name, _, _)| name).collect::<Vec<_>>())
                .default(0)
                .interact()?;

            if services[choice].1 == "back" {
                break;
            }

            let (service_name, service_key, url) = services[choice];

            // Check if already configured
            let current_key = match service_key {
                "stack_exchange" => &self.config.api_keys.optional_services.stack_exchange,
                "replicate" => &self.config.api_keys.optional_services.replicate,
                "huggingface" => &self.config.api_keys.optional_services.huggingface,
                "wolfram" => &self.config.api_keys.optional_services.wolfram_alpha,
                "news" => &self.config.api_keys.optional_services.news_api,
                "weather" => &self.config.api_keys.optional_services.weather_api,
                _ => &None,
            };

            if current_key.is_some() {
                let reconfigure = Confirm::with_theme(&self.theme)
                    .with_prompt(&format!("{} is already configured. Reconfigure?", service_name))
                    .default(false)
                    .interact()?;

                if !reconfigure {
                    continue;
                }
            }

            println!("\n{} Setup", service_name);
            if !url.is_empty() {
                println!("Get your API key from: {}", url);
            }

            let api_key: String = Password::with_theme(&self.theme)
                .with_prompt(&format!("{} API Key", service_name))
                .interact()?;

            // Save the API key
            match service_key {
                "stack_exchange" => {
                    self.config.api_keys.optional_services.stack_exchange = Some(api_key)
                }
                "replicate" => self.config.api_keys.optional_services.replicate = Some(api_key),
                "huggingface" => self.config.api_keys.optional_services.huggingface = Some(api_key),
                "wolfram" => self.config.api_keys.optional_services.wolfram_alpha = Some(api_key),
                "news" => self.config.api_keys.optional_services.news_api = Some(api_key),
                "weather" => self.config.api_keys.optional_services.weather_api = Some(api_key),
                _ => {}
            }

            println!("✅ {} configuration saved!", service_name);
        }

        Ok(())
    }

    fn showconfiguration(&self) -> Result<()> {
        println!("\n📋 Current Configuration");
        println!("=========================");

        let summary = self.config.api_keys.summary();

        println!("Configured Services:");
        for (service, configured) in summary.iter() {
            let status = if *configured { "✅ Configured" } else { "❌ Not configured" };
            println!("  {} - {}", service, status);
        }

        println!(
            "\nLast updated: {}",
            self.config.metadata.last_updated.format("%Y-%m-%d %H:%M:%S UTC")
        );
        println!("Configuration version: {}", self.config.metadata.version);

        let _continue_prompt = Confirm::with_theme(&self.theme)
            .with_prompt("Press Enter to continue")
            .default(true)
            .interact()?;

        Ok(())
    }

    fn exportconfiguration(&self) -> Result<()> {
        println!("\n📤 Export Configuration");
        println!("=======================");

        let env_content = self.config.export_to_env()?;

        if env_content.is_empty() {
            println!("No configuration to export. Please configure some services first.");
            return Ok(());
        }

        let export_to_file = Confirm::with_theme(&self.theme)
            .with_prompt("Export to .env file?")
            .default(true)
            .interact()?;

        if export_to_file {
            let env_path = PathBuf::from(".env");
            fs::write(&env_path, env_content)?;
            println!("✅ Configuration exported to .env file");
        } else {
            println!("\nEnvironment variables:");
            println!("{}", env_content);
        }

        Ok(())
    }

    fn resetconfiguration(&mut self) -> Result<()> {
        println!("\n🗑️ Reset Configuration");
        println!("=======================");

        let confirm = Confirm::with_theme(&self.theme)
            .with_prompt("Are you sure you want to reset ALL configuration? This cannot be undone.")
            .default(false)
            .interact()?;

        if confirm {
            self.config = SecureConfig::default();
            println!("✅ All configuration has been reset.");
        }

        Ok(())
    }

    fn show_completion(&self) {
        println!("\n🎉 Setup Complete!");
        println!("==================");

        let summary = self.config.api_keys.summary();
        let configured_count = summary.values().filter(|&&configured| configured).count();

        println!("You have configured {} services.", configured_count);
        println!("Your API keys are securely stored and ready to use with Loki!");
        println!("\nNext steps:");
        println!("  1. Run 'loki check-apis' to verify your configuration");
        println!("  2. Try 'loki test-provider <provider>' to test a specific provider");
        println!("  3. Start using Loki with 'loki run' or 'loki tui'");
        println!();
    }
}
