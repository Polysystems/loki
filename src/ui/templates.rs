// Setup Templates - Pre-configured model combinations
//
// This provides curated setup templates that users can select
// without needing to understand the underlying complexity.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::info;

use super::{
    CostClass,
    CostEstimate,
    PerformanceProfile,
    QualityLevel,
    ResourceUsageClass,
    ResponseTimeClass,
    SetupId,
};
use crate::models::orchestrator::RoutingStrategy;

/// Setup template for easy user selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SetupTemplate {
    pub id: SetupId,
    pub name: String,
    pub description: String,
    pub category: SetupCategory,
    pub models: Vec<TemplateModel>,
    pub routing_strategy: RoutingStrategy,
    pub cost_estimate: CostEstimate,
    pub performance_profile: PerformanceProfile,
    pub tags: Vec<String>,
    pub creator: Option<String>,
    pub rating: f32,
    pub usage_count: u64,
    pub is_featured: bool,
    pub best_for: Vec<String>,
    pub setup_time_seconds: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateModel {
    pub model_id: String,
    pub provider_type: ModelProviderType,
    pub role: ModelRole,
    pub priority: u8,
    pub auto_load: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelProviderType {
    Local(String), // Ollama model name
    API(String),   // Provider name
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelRole {
    Primary,      // Main model for most tasks
    Secondary,    // Backup/alternative model
    Specialist,   // For specific task types
    Fallback,     // Emergency fallback
    Orchestrator, // Coordinates other models
}

#[derive(Debug, Clone, Serialize, Deserialize, Eq, Hash, PartialEq)]
pub enum SetupCategory {
    Development,
    Writing,
    Research,
    Creative,
    Analysis,
    Education,
    Enterprise,
    Productivity,
    Innovation,
}

impl std::fmt::Display for SetupCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SetupCategory::Development => write!(f, "development"),
            SetupCategory::Writing => write!(f, "writing"),
            SetupCategory::Research => write!(f, "research"),
            SetupCategory::Creative => write!(f, "creative"),
            SetupCategory::Analysis => write!(f, "analysis"),
            SetupCategory::Education => write!(f, "education"),
            SetupCategory::Enterprise => write!(f, "enterprise"),
            SetupCategory::Productivity => write!(f, "productivity"),
            SetupCategory::Innovation => write!(f, "innovation"),
        }
    }
}

/// Manages setup templates
pub struct SetupTemplateManager {
    templates: Arc<RwLock<HashMap<SetupId, SetupTemplate>>>,
    categories: Arc<RwLock<HashMap<SetupCategory, Vec<SetupId>>>>,
}

impl SetupTemplateManager {
    pub async fn new() -> Result<Self> {
        Ok(Self {
            templates: Arc::new(RwLock::new(HashMap::new())),
            categories: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Load default setup templates
    pub async fn load_default_templates(&self) -> Result<()> {
        
        let templates = vec![
            self.create_lightning_fast_template(),
            self.create_balanced_pro_template(),
            self.create_premium_quality_template(),
            self.create_research_beast_template(),
            self.create_code_completion_template(),
            self.create_writing_assistant_template(),
        ];

        let mut template_map = self.templates.write().await;
        let mut category_map = self.categories.write().await;

        for template in templates {
            let category = template.category.clone();
            let id = template.id.clone();

            template_map.insert(id.clone(), template);
            category_map.entry(category).or_insert_with(Vec::new).push(id);
        }

        info!("Loaded {} default templates", template_map.len());
        Ok(())
    }

    /// Get template by ID
    pub async fn get_template(&self, id: &SetupId) -> Result<Option<SetupTemplate>> {
        Ok(self.templates.read().await.get(id).cloned())
    }

    /// List all templates
    pub async fn list_templates(&self) -> Result<Vec<SetupTemplate>> {
        Ok(self.templates.read().await.values().cloned().collect())
    }

    /// Get templates by category
    pub async fn get_by_category(&self, category: &str) -> Result<Vec<SetupTemplate>> {
        let category_enum = match category.to_lowercase().as_str() {
            "development" => SetupCategory::Development,
            "writing" => SetupCategory::Writing,
            "research" => SetupCategory::Research,
            "creative" => SetupCategory::Creative,
            "analysis" => SetupCategory::Analysis,
            "education" => SetupCategory::Education,
            "enterprise" => SetupCategory::Enterprise,
            "productivity" => SetupCategory::Productivity,
            "innovation" => SetupCategory::Innovation,
            _ => return Ok(Vec::new()),
        };

        let categories = self.categories.read().await;
        let templates = self.templates.read().await;

        if let Some(template_ids) = categories.get(&category_enum) {
            let mut result = Vec::new();
            for id in template_ids {
                if let Some(template) = templates.get(id) {
                    result.push(template.clone());
                }
            }
            Ok(result)
        } else {
            Ok(Vec::new())
        }
    }

    /// Get featured templates
    pub async fn get_featured_templates(&self) -> Result<Vec<SetupTemplate>> {
        let templates = self.templates.read().await;
        Ok(templates.values().filter(|t| t.is_featured).cloned().collect())
    }

    // Template creation methods

    fn create_lightning_fast_template(&self) -> SetupTemplate {
        SetupTemplate {
            id: SetupId::from_string("lightning-fast".to_string()),
            name: "Lightning Fast".to_string(),
            description: "Single local model for instant responses".to_string(),
            category: SetupCategory::Productivity,
            models: vec![TemplateModel {
                model_id: "magicoder-7b".to_string(),
                provider_type: ModelProviderType::Local("magicoder:7b-s-cl-q5_K_M".to_string()),
                role: ModelRole::Primary,
                priority: 1,
                auto_load: true,
            }],
            routing_strategy: RoutingStrategy::CapabilityBased,
            cost_estimate: CostEstimate {
                hourly_cost_cents: 0.0,
                per_request_cost_cents: 0.0,
                cost_class: CostClass::Free,
                explanation: "Local model only - no API costs".to_string(),
            },
            performance_profile: PerformanceProfile {
                response_time: ResponseTimeClass::Lightning,
                quality_level: QualityLevel::Good,
                cost_class: CostClass::Free,
                resource_usage: ResourceUsageClass::Light,
            },
            tags: vec!["fast".to_string(), "free".to_string(), "local".to_string()],
            creator: Some("Loki Team".to_string()),
            rating: 4.5,
            usage_count: 1250,
            is_featured: true,
            best_for: vec![
                "Quick coding questions".to_string(),
                "Simple explanations".to_string(),
                "Fast prototyping".to_string(),
            ],
            setup_time_seconds: 30,
        }
    }

    fn create_balanced_pro_template(&self) -> SetupTemplate {
        SetupTemplate {
            id: SetupId::from_string("balanced-pro".to_string()),
            name: "Balanced Pro".to_string(),
            description: "2 local models + API fallback for optimal quality/cost".to_string(),
            category: SetupCategory::Development,
            models: vec![
                TemplateModel {
                    model_id: "deepseek-coder-v2".to_string(),
                    provider_type: ModelProviderType::Local("deepseek-coder-v2:latest".to_string()),
                    role: ModelRole::Primary,
                    priority: 1,
                    auto_load: true,
                },
                TemplateModel {
                    model_id: "magicoder-7b".to_string(),
                    provider_type: ModelProviderType::Local("magicoder:7b-s-cl-q5_K_M".to_string()),
                    role: ModelRole::Secondary,
                    priority: 2,
                    auto_load: true,
                },
                TemplateModel {
                    model_id: "claude-3.5-haiku".to_string(),
                    provider_type: ModelProviderType::API("anthropic".to_string()),
                    role: ModelRole::Fallback,
                    priority: 3,
                    auto_load: false,
                },
            ],
            routing_strategy: RoutingStrategy::CapabilityBased,
            cost_estimate: CostEstimate {
                hourly_cost_cents: 10.0,
                per_request_cost_cents: 0.5,
                cost_class: CostClass::Low,
                explanation: "Mostly free local models with occasional API usage".to_string(),
            },
            performance_profile: PerformanceProfile {
                response_time: ResponseTimeClass::Fast,
                quality_level: QualityLevel::High,
                cost_class: CostClass::Low,
                resource_usage: ResourceUsageClass::Moderate,
            },
            tags: vec!["balanced".to_string(), "professional".to_string(), "hybrid".to_string()],
            creator: Some("Loki Team".to_string()),
            rating: 4.8,
            usage_count: 2150,
            is_featured: true,
            best_for: vec![
                "Professional development".to_string(),
                "Code review".to_string(),
                "Technical writing".to_string(),
            ],
            setup_time_seconds: 120,
        }
    }

    fn create_premium_quality_template(&self) -> SetupTemplate {
        SetupTemplate {
            id: SetupId::from_string("premium-quality".to_string()),
            name: "Premium Quality".to_string(),
            description: "Best models working together for highest quality".to_string(),
            category: SetupCategory::Enterprise,
            models: vec![
                TemplateModel {
                    model_id: "claude-3.5-sonnet".to_string(),
                    provider_type: ModelProviderType::API("anthropic".to_string()),
                    role: ModelRole::Orchestrator,
                    priority: 1,
                    auto_load: true,
                },
                TemplateModel {
                    model_id: "gpt-4o".to_string(),
                    provider_type: ModelProviderType::API("openai".to_string()),
                    role: ModelRole::Specialist,
                    priority: 2,
                    auto_load: false,
                },
                TemplateModel {
                    model_id: "deepseek-coder-v2".to_string(),
                    provider_type: ModelProviderType::Local("deepseek-coder-v2:latest".to_string()),
                    role: ModelRole::Specialist,
                    priority: 3,
                    auto_load: true,
                },
            ],
            routing_strategy: RoutingStrategy::CapabilityBased,
            cost_estimate: CostEstimate {
                hourly_cost_cents: 50.0,
                per_request_cost_cents: 2.5,
                cost_class: CostClass::Medium,
                explanation: "Premium API models for highest quality results".to_string(),
            },
            performance_profile: PerformanceProfile {
                response_time: ResponseTimeClass::Moderate,
                quality_level: QualityLevel::Excellence,
                cost_class: CostClass::Medium,
                resource_usage: ResourceUsageClass::Moderate,
            },
            tags: vec!["premium".to_string(), "quality".to_string(), "enterprise".to_string()],
            creator: Some("Loki Team".to_string()),
            rating: 4.9,
            usage_count: 850,
            is_featured: true,
            best_for: vec![
                "Complex projects".to_string(),
                "Research".to_string(),
                "Critical decisions".to_string(),
            ],
            setup_time_seconds: 60,
        }
    }

    fn create_research_beast_template(&self) -> SetupTemplate {
        SetupTemplate {
            id: SetupId::from_string("research-beast".to_string()),
            name: "Research Beast".to_string(),
            description: "5-model ensemble for comprehensive analysis".to_string(),
            category: SetupCategory::Research,
            models: vec![
                TemplateModel {
                    model_id: "claude-3.5-sonnet".to_string(),
                    provider_type: ModelProviderType::API("anthropic".to_string()),
                    role: ModelRole::Orchestrator,
                    priority: 1,
                    auto_load: true,
                },
                TemplateModel {
                    model_id: "gpt-4o".to_string(),
                    provider_type: ModelProviderType::API("openai".to_string()),
                    role: ModelRole::Specialist,
                    priority: 2,
                    auto_load: true,
                },
                TemplateModel {
                    model_id: "perplexity-pro".to_string(),
                    provider_type: ModelProviderType::API("perplexity".to_string()),
                    role: ModelRole::Specialist,
                    priority: 3,
                    auto_load: false,
                },
                TemplateModel {
                    model_id: "deepseek-coder-v2".to_string(),
                    provider_type: ModelProviderType::Local("deepseek-coder-v2:latest".to_string()),
                    role: ModelRole::Specialist,
                    priority: 4,
                    auto_load: true,
                },
                TemplateModel {
                    model_id: "wizardcoder-34b".to_string(),
                    provider_type: ModelProviderType::Local(
                        "wizardcoder:34b-python-q4_K_M".to_string(),
                    ),
                    role: ModelRole::Secondary,
                    priority: 5,
                    auto_load: false,
                },
            ],
            routing_strategy: RoutingStrategy::LoadBased,
            cost_estimate: CostEstimate {
                hourly_cost_cents: 100.0,
                per_request_cost_cents: 5.0,
                cost_class: CostClass::High,
                explanation: "Multiple premium models for comprehensive analysis".to_string(),
            },
            performance_profile: PerformanceProfile {
                response_time: ResponseTimeClass::Thorough,
                quality_level: QualityLevel::Excellence,
                cost_class: CostClass::High,
                resource_usage: ResourceUsageClass::Intensive,
            },
            tags: vec!["research".to_string(), "ensemble".to_string(), "comprehensive".to_string()],
            creator: Some("Loki Team".to_string()),
            rating: 4.7,
            usage_count: 420,
            is_featured: true,
            best_for: vec![
                "Academic research".to_string(),
                "Complex problem solving".to_string(),
                "Multi-perspective analysis".to_string(),
            ],
            setup_time_seconds: 180,
        }
    }

    fn create_code_completion_template(&self) -> SetupTemplate {
        SetupTemplate {
            id: SetupId::from_string("code-completion".to_string()),
            name: "Code Completion Master".to_string(),
            description: "Optimized for fast, accurate code completion".to_string(),
            category: SetupCategory::Development,
            models: vec![
                TemplateModel {
                    model_id: "deepseek-coder-v2".to_string(),
                    provider_type: ModelProviderType::Local("deepseek-coder-v2:latest".to_string()),
                    role: ModelRole::Primary,
                    priority: 1,
                    auto_load: true,
                },
                TemplateModel {
                    model_id: "magicoder-7b".to_string(),
                    provider_type: ModelProviderType::Local("magicoder:7b-s-cl-q5_K_M".to_string()),
                    role: ModelRole::Secondary,
                    priority: 2,
                    auto_load: true,
                },
            ],
            routing_strategy: RoutingStrategy::CapabilityBased,
            cost_estimate: CostEstimate {
                hourly_cost_cents: 0.0,
                per_request_cost_cents: 0.0,
                cost_class: CostClass::Free,
                explanation: "Local models only - perfect for continuous coding".to_string(),
            },
            performance_profile: PerformanceProfile {
                response_time: ResponseTimeClass::Lightning,
                quality_level: QualityLevel::High,
                cost_class: CostClass::Free,
                resource_usage: ResourceUsageClass::Moderate,
            },
            tags: vec!["coding".to_string(), "completion".to_string(), "fast".to_string()],
            creator: Some("Loki Team".to_string()),
            rating: 4.6,
            usage_count: 1850,
            is_featured: false,
            best_for: vec![
                "Code completion".to_string(),
                "Quick fixes".to_string(),
                "Continuous development".to_string(),
            ],
            setup_time_seconds: 45,
        }
    }

    fn create_writing_assistant_template(&self) -> SetupTemplate {
        SetupTemplate {
            id: SetupId::from_string("writing-assistant".to_string()),
            name: "Writing Assistant Pro".to_string(),
            description: "Perfect for all writing tasks with style and clarity".to_string(),
            category: SetupCategory::Writing,
            models: vec![
                TemplateModel {
                    model_id: "claude-3.5-sonnet".to_string(),
                    provider_type: ModelProviderType::API("anthropic".to_string()),
                    role: ModelRole::Primary,
                    priority: 1,
                    auto_load: true,
                },
                TemplateModel {
                    model_id: "gpt-4o".to_string(),
                    provider_type: ModelProviderType::API("openai".to_string()),
                    role: ModelRole::Secondary,
                    priority: 2,
                    auto_load: false,
                },
            ],
            routing_strategy: RoutingStrategy::CapabilityBased,
            cost_estimate: CostEstimate {
                hourly_cost_cents: 30.0,
                per_request_cost_cents: 1.5,
                cost_class: CostClass::Medium,
                explanation: "High-quality API models for professional writing".to_string(),
            },
            performance_profile: PerformanceProfile {
                response_time: ResponseTimeClass::Fast,
                quality_level: QualityLevel::Premium,
                cost_class: CostClass::Medium,
                resource_usage: ResourceUsageClass::Light,
            },
            tags: vec!["writing".to_string(), "creative".to_string(), "professional".to_string()],
            creator: Some("Loki Team".to_string()),
            rating: 4.8,
            usage_count: 950,
            is_featured: false,
            best_for: vec![
                "Creative writing".to_string(),
                "Technical documentation".to_string(),
                "Content creation".to_string(),
            ],
            setup_time_seconds: 30,
        }
    }
}
