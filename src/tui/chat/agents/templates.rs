//! Agent Templates Library
//! 
//! Provides a comprehensive library of pre-configured agent templates for various
//! specialized tasks and domains.

use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use anyhow::Result;

use super::creation::{
    AgentTemplate, TemplateCategory, AgentConfig, PersonalityProfile, PersonalityTrait,
    CommunicationStyle, DecisionStyle, WorkStyle, InteractionPreferences, AgentSkill,
    SkillCategory, ModelPreferences, ModelSelectionStrategy, ContextPreferences,
    ToolPermissions, CollaborationSettings, LeadershipStyle, PerformanceSettings,
    OptimizationLevel, AgentMetadata, ResourceLimits, AgentPermissions,
};
use crate::cognitive::agents::AgentSpecialization;
use uuid::Uuid;
use chrono::Utc;

/// Template library for managing agent templates
pub struct TemplateLibrary {
    /// All templates indexed by ID
    templates: HashMap<String, AgentTemplate>,
    
    /// Templates by category
    by_category: HashMap<TemplateCategory, Vec<String>>,
    
    /// Custom user templates
    custom_templates: HashMap<String, AgentTemplate>,
    
    /// Template metadata
    metadata: LibraryMetadata,
}

/// Library metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LibraryMetadata {
    pub total_templates: usize,
    pub total_categories: usize,
    pub last_updated: chrono::DateTime<chrono::Utc>,
    pub version: String,
}

impl TemplateLibrary {
    /// Create a new template library
    pub fn new() -> Self {
        let mut library = Self {
            templates: HashMap::new(),
            by_category: HashMap::new(),
            custom_templates: HashMap::new(),
            metadata: LibraryMetadata {
                total_templates: 0,
                total_categories: 0,
                last_updated: Utc::now(),
                version: "1.0.0".to_string(),
            },
        };
        
        // Load default templates
        library.load_default_templates();
        library
    }
    
    /// Load all default templates
    fn load_default_templates(&mut self) {
        // Research & Analysis Templates
        self.add_template(Self::create_data_scientist_template());
        self.add_template(Self::create_market_researcher_template());
        self.add_template(Self::create_academic_researcher_template());
        
        // Development Templates
        self.add_template(Self::create_backend_developer_template());
        self.add_template(Self::create_frontend_developer_template());
        self.add_template(Self::create_devops_engineer_template());
        self.add_template(Self::create_security_engineer_template());
        
        // Creative Templates
        self.add_template(Self::create_content_writer_template());
        self.add_template(Self::create_technical_writer_template());
        self.add_template(Self::create_marketing_specialist_template());
        
        // Communication Templates
        self.add_template(Self::create_customer_support_template());
        self.add_template(Self::create_sales_assistant_template());
        self.add_template(Self::create_educator_template());
        
        // Management Templates
        self.add_template(Self::create_project_manager_template());
        self.add_template(Self::create_product_manager_template());
        self.add_template(Self::create_team_lead_template());
        
        // Analysis Templates
        self.add_template(Self::create_business_analyst_template());
        self.add_template(Self::create_financial_analyst_template());
        self.add_template(Self::create_qa_engineer_template());
    }
    
    /// Add a template to the library
    pub fn add_template(&mut self, template: AgentTemplate) {
        self.by_category
            .entry(template.category.clone())
            .or_insert_with(Vec::new)
            .push(template.id.clone());
        
        self.templates.insert(template.id.clone(), template);
        self.update_metadata();
    }
    
    /// Update library metadata
    fn update_metadata(&mut self) {
        self.metadata.total_templates = self.templates.len() + self.custom_templates.len();
        self.metadata.total_categories = self.by_category.len();
        self.metadata.last_updated = Utc::now();
    }
    
    /// Get a template by ID
    pub fn get_template(&self, id: &str) -> Option<&AgentTemplate> {
        self.templates.get(id).or_else(|| self.custom_templates.get(id))
    }
    
    /// Get templates by category
    pub fn get_by_category(&self, category: TemplateCategory) -> Vec<&AgentTemplate> {
        self.by_category
            .get(&category)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.templates.get(id))
                    .collect()
            })
            .unwrap_or_default()
    }
    
    /// Search templates
    pub fn search(&self, query: &str) -> Vec<&AgentTemplate> {
        let query_lower = query.to_lowercase();
        
        self.templates
            .values()
            .chain(self.custom_templates.values())
            .filter(|t| {
                t.name.to_lowercase().contains(&query_lower) ||
                t.description.to_lowercase().contains(&query_lower) ||
                t.typical_tasks.iter().any(|task| task.to_lowercase().contains(&query_lower))
            })
            .collect()
    }
    
    /// Create data scientist template
    fn create_data_scientist_template() -> AgentTemplate {
        let mut personality = PersonalityProfile::default();
        personality.traits.insert(PersonalityTrait::Analytical, 0.95);
        personality.traits.insert(PersonalityTrait::Precision, 0.9);
        personality.traits.insert(PersonalityTrait::Curiosity, 0.85);
        personality.communication_style = CommunicationStyle::Technical;
        personality.decision_style = DecisionStyle::DataDriven;
        personality.work_style = WorkStyle::Methodical;
        
        AgentTemplate {
            id: "data_scientist".to_string(),
            name: "Data Scientist".to_string(),
            description: "Expert in data analysis, machine learning, and statistical modeling".to_string(),
            icon: "📊".to_string(),
            category: TemplateCategory::Analysis,
            base_config: AgentConfig {
                id: Uuid::new_v4().to_string(),
                name: "Data Scientist".to_string(),
                description: "Specialized in data science and analytics".to_string(),
                specialization: AgentSpecialization::Analytical,
                personality,
                skills: vec![
                    AgentSkill {
                        id: "data_analysis".to_string(),
                        name: "Data Analysis".to_string(),
                        category: SkillCategory::DataAnalysis,
                        proficiency: 0.95,
                        experience_hours: 3000,
                        certifications: vec![],
                        examples: vec![],
                    },
                    AgentSkill {
                        id: "machine_learning".to_string(),
                        name: "Machine Learning".to_string(),
                        category: SkillCategory::DataAnalysis,
                        proficiency: 0.9,
                        experience_hours: 2500,
                        certifications: vec![],
                        examples: vec![],
                    },
                ],
                model_preferences: ModelPreferences {
                    primary_model: Some("gpt-4".to_string()),
                    fallback_models: vec!["claude-3-opus".to_string()],
                    model_selection_strategy: ModelSelectionStrategy::TaskBased,
                    context_preferences: ContextPreferences {
                        preferred_context_size: 8192,
                        include_examples: true,
                        include_history: true,
                        compression_enabled: false,
                    },
                    cost_threshold: None,
                },
                tool_permissions: ToolPermissions {
                    allowed_tools: vec![
                        "python_executor".to_string(),
                        "jupyter".to_string(),
                        "data_visualization".to_string(),
                    ],
                    blocked_tools: vec![],
                    requires_approval: vec![],
                    auto_execute: vec!["data_visualization".to_string()],
                    rate_limits: HashMap::new(),
                },
                collaboration_settings: CollaborationSettings {
                    can_collaborate: true,
                    preferred_team_size: 3,
                    leadership_style: LeadershipStyle::Peer,
                    collaboration_rules: vec![],
                },
                performance_settings: PerformanceSettings {
                    optimization_level: OptimizationLevel::Quality,
                    caching_enabled: true,
                    parallel_processing: true,
                    batch_size: 100,
                    timeout_seconds: 120,
                },
                metadata: AgentMetadata {
                    created_at: Utc::now(),
                    created_by: "template".to_string(),
                    last_modified: Utc::now(),
                    version: "1.0.0".to_string(),
                    tags: vec!["data".to_string(), "analytics".to_string(), "ml".to_string()],
                    documentation: "Data scientist agent template".to_string(),
                    examples: vec![],
                },
            },
            recommended_models: vec!["gpt-4".to_string(), "claude-3-opus".to_string()],
            default_tools: vec!["python_executor".to_string(), "jupyter".to_string()],
            typical_tasks: vec![
                "Data analysis and visualization".to_string(),
                "Statistical modeling".to_string(),
                "Machine learning model development".to_string(),
                "A/B testing and experimentation".to_string(),
            ],
            creation_hints: vec![
                "Enable Python execution for data processing".to_string(),
                "Configure Jupyter notebook access".to_string(),
                "Set up data visualization tools".to_string(),
            ],
            role: crate::tui::chat::agents::templates_impl::AgentRole::Specialist,
            capabilities: vec![
                crate::tui::chat::agents::templates_impl::AgentCapability::Analysis,
                crate::tui::chat::agents::templates_impl::AgentCapability::Mathematics,
                crate::tui::chat::agents::templates_impl::AgentCapability::Research,
            ],
            system_prompt: "You are a data scientist specializing in statistical analysis, machine learning, and data visualization. Provide data-driven insights and build predictive models.".to_string(),
            conversation_style: "Technical and analytical, focusing on data and statistics".to_string(),
            expertise_areas: vec![
                "Statistical analysis".to_string(),
                "Machine learning".to_string(),
                "Data visualization".to_string(),
                "Predictive modeling".to_string(),
                "A/B testing".to_string(),
            ],
            tool_preferences: vec![
                "python_executor".to_string(),
                "jupyter".to_string(),
                "data_visualization".to_string(),
            ],
            collaboration_preferences: vec![
                "Business Analyst".to_string(),
                "Product Manager".to_string(),
                "Backend Developer".to_string(),
            ],
            constraints: vec![
                "Require clean data inputs".to_string(),
                "Follow statistical best practices".to_string(),
                "Document model assumptions".to_string(),
            ],
        }
    }
    
    /// Create backend developer template
    fn create_backend_developer_template() -> AgentTemplate {
        let mut personality = PersonalityProfile::default();
        personality.traits.insert(PersonalityTrait::Analytical, 0.85);
        personality.traits.insert(PersonalityTrait::Precision, 0.95);
        personality.traits.insert(PersonalityTrait::Patience, 0.8);
        personality.communication_style = CommunicationStyle::Technical;
        personality.decision_style = DecisionStyle::Balanced;
        personality.work_style = WorkStyle::Structured;
        
        AgentTemplate {
            id: "backend_developer".to_string(),
            name: "Backend Developer".to_string(),
            description: "Expert in server-side development, APIs, and databases".to_string(),
            icon: "🔧".to_string(),
            category: TemplateCategory::Development,
            base_config: AgentConfig {
                id: Uuid::new_v4().to_string(),
                name: "Backend Developer".to_string(),
                description: "Specialized in backend development".to_string(),
                specialization: AgentSpecialization::Technical,
                personality,
                skills: vec![
                    AgentSkill {
                        id: "backend_dev".to_string(),
                        name: "Backend Development".to_string(),
                        category: SkillCategory::Programming,
                        proficiency: 0.95,
                        experience_hours: 5000,
                        certifications: vec![],
                        examples: vec![],
                    },
                    AgentSkill {
                        id: "database_design".to_string(),
                        name: "Database Design".to_string(),
                        category: SkillCategory::Programming,
                        proficiency: 0.9,
                        experience_hours: 3000,
                        certifications: vec![],
                        examples: vec![],
                    },
                ],
                model_preferences: ModelPreferences {
                    primary_model: Some("codestral".to_string()),
                    fallback_models: vec!["gpt-4".to_string()],
                    model_selection_strategy: ModelSelectionStrategy::TaskBased,
                    context_preferences: ContextPreferences {
                        preferred_context_size: 16384,
                        include_examples: true,
                        include_history: true,
                        compression_enabled: false,
                    },
                    cost_threshold: None,
                },
                tool_permissions: ToolPermissions {
                    allowed_tools: vec![
                        "code_analysis".to_string(),
                        "github".to_string(),
                        "docker".to_string(),
                        "database".to_string(),
                    ],
                    blocked_tools: vec![],
                    requires_approval: vec!["database".to_string()],
                    auto_execute: vec!["code_analysis".to_string()],
                    rate_limits: HashMap::new(),
                },
                collaboration_settings: CollaborationSettings {
                    can_collaborate: true,
                    preferred_team_size: 5,
                    leadership_style: LeadershipStyle::Adaptive,
                    collaboration_rules: vec![],
                },
                performance_settings: PerformanceSettings {
                    optimization_level: OptimizationLevel::Balanced,
                    caching_enabled: true,
                    parallel_processing: true,
                    batch_size: 20,
                    timeout_seconds: 180,
                },
                metadata: AgentMetadata {
                    created_at: Utc::now(),
                    created_by: "template".to_string(),
                    last_modified: Utc::now(),
                    version: "1.0.0".to_string(),
                    tags: vec!["backend".to_string(), "api".to_string(), "database".to_string()],
                    documentation: "Backend developer agent template".to_string(),
                    examples: vec![],
                },
            },
            recommended_models: vec!["codestral".to_string(), "gpt-4".to_string()],
            default_tools: vec!["code_analysis".to_string(), "github".to_string()],
            typical_tasks: vec![
                "API development".to_string(),
                "Database design and optimization".to_string(),
                "Microservices architecture".to_string(),
                "Performance optimization".to_string(),
            ],
            creation_hints: vec![
                "Enable code analysis tools".to_string(),
                "Configure database access carefully".to_string(),
                "Set up Docker for containerization".to_string(),
            ],
            role: crate::tui::chat::agents::templates_impl::AgentRole::Technical,
            capabilities: vec![
                crate::tui::chat::agents::templates_impl::AgentCapability::Coding,
                crate::tui::chat::agents::templates_impl::AgentCapability::Debugging,
                crate::tui::chat::agents::templates_impl::AgentCapability::Testing,
            ],
            system_prompt: "You are a backend developer specializing in server-side development, API design, and database management. Focus on scalability, performance, and security.".to_string(),
            conversation_style: "Technical and systematic, with focus on architecture and performance".to_string(),
            expertise_areas: vec![
                "API development".to_string(),
                "Database design".to_string(),
                "Microservices architecture".to_string(),
                "Performance optimization".to_string(),
                "Security best practices".to_string(),
            ],
            tool_preferences: vec![
                "code_analysis".to_string(),
                "github".to_string(),
                "docker".to_string(),
                "database".to_string(),
            ],
            collaboration_preferences: vec![
                "Frontend Developer".to_string(),
                "DevOps Engineer".to_string(),
                "Database Administrator".to_string(),
            ],
            constraints: vec![
                "Follow API design standards".to_string(),
                "Ensure database security".to_string(),
                "Optimize for scalability".to_string(),
            ],
        }
    }
    
    /// Create project manager template
    fn create_project_manager_template() -> AgentTemplate {
        let mut personality = PersonalityProfile::default();
        personality.traits.insert(PersonalityTrait::Assertiveness, 0.75);
        personality.traits.insert(PersonalityTrait::Empathy, 0.85);
        personality.traits.insert(PersonalityTrait::Adaptability, 0.9);
        personality.communication_style = CommunicationStyle::Professional;
        personality.decision_style = DecisionStyle::Collaborative;
        personality.work_style = WorkStyle::Efficient;
        
        AgentTemplate {
            id: "project_manager".to_string(),
            name: "Project Manager".to_string(),
            description: "Expert in project planning, team coordination, and delivery".to_string(),
            icon: "📋".to_string(),
            category: TemplateCategory::Management,
            base_config: AgentConfig {
                id: Uuid::new_v4().to_string(),
                name: "Project Manager".to_string(),
                description: "Specialized in project management".to_string(),
                specialization: AgentSpecialization::Managerial,
                personality,
                skills: vec![
                    AgentSkill {
                        id: "project_management".to_string(),
                        name: "Project Management".to_string(),
                        category: SkillCategory::ProjectManagement,
                        proficiency: 0.9,
                        experience_hours: 4000,
                        certifications: vec!["PMP".to_string()],
                        examples: vec![],
                    },
                ],
                model_preferences: ModelPreferences {
                    primary_model: Some("gpt-4".to_string()),
                    fallback_models: vec!["claude-3-sonnet".to_string()],
                    model_selection_strategy: ModelSelectionStrategy::Balanced,
                    context_preferences: ContextPreferences {
                        preferred_context_size: 8192,
                        include_examples: true,
                        include_history: true,
                        compression_enabled: false,
                    },
                    cost_threshold: Some(0.1),
                },
                tool_permissions: ToolPermissions {
                    allowed_tools: vec![
                        "calendar".to_string(),
                        "task_tracker".to_string(),
                        "slack".to_string(),
                        "email".to_string(),
                    ],
                    blocked_tools: vec![],
                    requires_approval: vec![],
                    auto_execute: vec!["calendar".to_string()],
                    rate_limits: HashMap::new(),
                },
                collaboration_settings: CollaborationSettings {
                    can_collaborate: true,
                    preferred_team_size: 8,
                    leadership_style: LeadershipStyle::Leader,
                    collaboration_rules: vec![],
                },
                performance_settings: PerformanceSettings {
                    optimization_level: OptimizationLevel::Balanced,
                    caching_enabled: true,
                    parallel_processing: false,
                    batch_size: 5,
                    timeout_seconds: 60,
                },
                metadata: AgentMetadata {
                    created_at: Utc::now(),
                    created_by: "template".to_string(),
                    last_modified: Utc::now(),
                    version: "1.0.0".to_string(),
                    tags: vec!["management".to_string(), "planning".to_string()],
                    documentation: "Project manager agent template".to_string(),
                    examples: vec![],
                },
            },
            recommended_models: vec!["gpt-4".to_string(), "claude-3-sonnet".to_string()],
            default_tools: vec!["task_tracker".to_string(), "calendar".to_string()],
            typical_tasks: vec![
                "Project planning and scheduling".to_string(),
                "Resource allocation".to_string(),
                "Risk management".to_string(),
                "Stakeholder communication".to_string(),
            ],
            creation_hints: vec![
                "Enable task tracking integration".to_string(),
                "Configure calendar access".to_string(),
                "Set up team communication tools".to_string(),
            ],
            role: crate::tui::chat::agents::templates_impl::AgentRole::Leader,
            capabilities: vec![
                crate::tui::chat::agents::templates_impl::AgentCapability::Planning,
                crate::tui::chat::agents::templates_impl::AgentCapability::Coordination,
                crate::tui::chat::agents::templates_impl::AgentCapability::Communication,
            ],
            system_prompt: "You are a project manager responsible for planning, coordinating, and delivering projects successfully. Balance scope, time, and resources effectively.".to_string(),
            conversation_style: "Organized and results-oriented, with focus on delivery and team coordination".to_string(),
            expertise_areas: vec![
                "Project planning".to_string(),
                "Resource allocation".to_string(),
                "Risk management".to_string(),
                "Stakeholder communication".to_string(),
                "Team coordination".to_string(),
            ],
            tool_preferences: vec![
                "task_tracker".to_string(),
                "calendar".to_string(),
                "slack".to_string(),
                "email".to_string(),
            ],
            collaboration_preferences: vec![
                "Team Lead".to_string(),
                "Product Manager".to_string(),
                "Business Analyst".to_string(),
            ],
            constraints: vec![
                "Budget limitations".to_string(),
                "Timeline constraints".to_string(),
                "Resource availability".to_string(),
            ],
        }
    }
    
    // Additional template creation methods would follow the same pattern...
    fn create_market_researcher_template() -> AgentTemplate {
        super::templates_impl::AgentTemplateImplementations::create_market_researcher_template()
    }
    
    fn create_academic_researcher_template() -> AgentTemplate {
        super::templates_impl::AgentTemplateImplementations::create_academic_researcher_template()
    }
    
    fn create_frontend_developer_template() -> AgentTemplate {
        super::templates_impl::AgentTemplateImplementations::create_frontend_developer_template()
    }
    
    fn create_devops_engineer_template() -> AgentTemplate {
        super::templates_impl::AgentTemplateImplementations::create_devops_engineer_template()
    }
    
    fn create_security_engineer_template() -> AgentTemplate {
        super::templates_impl::AgentTemplateImplementations::create_security_engineer_template()
    }
    
    fn create_content_writer_template() -> AgentTemplate {
        super::templates_impl::AgentTemplateImplementations::create_content_writer_template()
    }
    
    fn create_technical_writer_template() -> AgentTemplate {
        super::templates_impl::AgentTemplateImplementations::create_technical_writer_template()
    }
    
    fn create_marketing_specialist_template() -> AgentTemplate {
        super::templates_impl::AgentTemplateImplementations::create_marketing_specialist_template()
    }
    
    fn create_customer_support_template() -> AgentTemplate {
        super::templates_impl::AgentTemplateImplementations::create_customer_support_template()
    }
    
    fn create_sales_assistant_template() -> AgentTemplate {
        super::templates_impl::AgentTemplateImplementations::create_sales_assistant_template()
    }
    
    fn create_educator_template() -> AgentTemplate {
        super::templates_impl::AgentTemplateImplementations::create_educator_template()
    }
    
    fn create_product_manager_template() -> AgentTemplate {
        super::templates_impl::AgentTemplateImplementations::create_product_manager_template()
    }
    
    fn create_team_lead_template() -> AgentTemplate {
        super::templates_impl::AgentTemplateImplementations::create_team_lead_template()
    }
    
    fn create_business_analyst_template() -> AgentTemplate {
        super::templates_impl::AgentTemplateImplementations::create_business_analyst_template()
    }
    
    fn create_financial_analyst_template() -> AgentTemplate {
        super::templates_impl::AgentTemplateImplementations::create_financial_analyst_template()
    }
    
    fn create_qa_engineer_template() -> AgentTemplate {
        super::templates_impl::AgentTemplateImplementations::create_qa_engineer_template()
    }
}

/// Template builder for creating custom templates
pub struct TemplateBuilder {
    template: AgentTemplate,
}

impl TemplateBuilder {
    /// Create a new template builder
    pub fn new(name: String, category: TemplateCategory) -> Self {
        Self {
            template: AgentTemplate {
                id: Uuid::new_v4().to_string(),
                name: name.clone(),
                description: String::new(),
                icon: "🤖".to_string(),
                category,
                base_config: AgentConfig {
                    id: Uuid::new_v4().to_string(),
                    name,
                    description: String::new(),
                    specialization: AgentSpecialization::General,
                    personality: PersonalityProfile::default(),
                    skills: Vec::new(),
                    model_preferences: ModelPreferences {
                        primary_model: None,
                        fallback_models: Vec::new(),
                        model_selection_strategy: ModelSelectionStrategy::Balanced,
                        context_preferences: ContextPreferences {
                            preferred_context_size: 4096,
                            include_examples: true,
                            include_history: true,
                            compression_enabled: false,
                        },
                        cost_threshold: None,
                    },
                    tool_permissions: ToolPermissions {
                        allowed_tools: Vec::new(),
                        blocked_tools: Vec::new(),
                        requires_approval: Vec::new(),
                        auto_execute: Vec::new(),
                        rate_limits: HashMap::new(),
                    },
                    collaboration_settings: CollaborationSettings {
                        can_collaborate: true,
                        preferred_team_size: 3,
                        leadership_style: LeadershipStyle::Adaptive,
                        collaboration_rules: Vec::new(),
                    },
                    performance_settings: PerformanceSettings {
                        optimization_level: OptimizationLevel::Balanced,
                        caching_enabled: true,
                        parallel_processing: false,
                        batch_size: 10,
                        timeout_seconds: 60,
                    },
                    metadata: AgentMetadata {
                        created_at: Utc::now(),
                        created_by: "user".to_string(),
                        last_modified: Utc::now(),
                        version: "1.0.0".to_string(),
                        tags: Vec::new(),
                        documentation: String::new(),
                        examples: Vec::new(),
                    },
                },
                recommended_models: Vec::new(),
                default_tools: Vec::new(),
                typical_tasks: Vec::new(),
                creation_hints: Vec::new(),
                role: crate::tui::chat::agents::templates_impl::AgentRole::Assistant,
                capabilities: vec![],
                system_prompt: String::new(),
                conversation_style: String::new(),
                expertise_areas: vec![],
                tool_preferences: vec![],
                collaboration_preferences: vec![],
                constraints: vec![],
            },
        }
    }
    
    /// Set template description
    pub fn with_description(mut self, description: String) -> Self {
        self.template.description = description.clone();
        self.template.base_config.description = description;
        self
    }
    
    /// Set template icon
    pub fn with_icon(mut self, icon: String) -> Self {
        self.template.icon = icon;
        self
    }
    
    /// Set specialization
    pub fn with_specialization(mut self, specialization: AgentSpecialization) -> Self {
        self.template.base_config.specialization = specialization;
        self
    }
    
    /// Add a skill
    pub fn add_skill(mut self, skill: AgentSkill) -> Self {
        self.template.base_config.skills.push(skill);
        self
    }
    
    /// Set personality trait
    pub fn with_personality_trait(mut self, trait_type: PersonalityTrait, value: f32) -> Self {
        self.template.base_config.personality.traits.insert(trait_type, value.clamp(0.0, 1.0));
        self
    }
    
    /// Add recommended model
    pub fn add_recommended_model(mut self, model: String) -> Self {
        self.template.recommended_models.push(model);
        self
    }
    
    /// Add default tool
    pub fn add_default_tool(mut self, tool: String) -> Self {
        self.template.default_tools.push(tool.clone());
        self.template.base_config.tool_permissions.allowed_tools.push(tool);
        self
    }
    
    /// Add typical task
    pub fn add_typical_task(mut self, task: String) -> Self {
        self.template.typical_tasks.push(task);
        self
    }
    
    /// Add creation hint
    pub fn add_creation_hint(mut self, hint: String) -> Self {
        self.template.creation_hints.push(hint);
        self
    }
    
    /// Build the template
    pub fn build(self) -> AgentTemplate {
        self.template
    }
}
