//! Complete agent template implementations
//! 
//! This module provides full implementations for all agent templates,
//! replacing the placeholder implementations.

use super::creation::{
    AgentTemplate, PersonalityProfile, AgentConfig, TemplateCategory,
    PersonalityTrait, CommunicationStyle, DecisionStyle, WorkStyle,
    ModelPreferences,
    ModelSelectionStrategy, ContextPreferences, ToolPermissions,
    CollaborationSettings, LeadershipStyle, PerformanceSettings,
    OptimizationLevel, AgentMetadata
};
use crate::cognitive::agents::AgentSpecialization;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use uuid::Uuid;
use chrono::Utc;

/// Agent capabilities
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentCapability {
    Analysis,
    Coding,
    Communication,
    Coordination,
    Creativity,
    Debugging,
    Design,
    Documentation,
    Execution,
    Mathematics,
    Mentoring,
    Planning,
    Reporting,
    Research,
    Teaching,
    Testing,
    Writing,
}

/// Agent roles
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentRole {
    Leader,
    Specialist,
    Assistant,
    Analyst,
    Technical,
    Creative,
    Strategic,
}

/// Implementation of all agent templates
pub struct AgentTemplateImplementations;

impl AgentTemplateImplementations {
    /// Helper to create base config for templates
    fn create_base_config(
        name: &str,
        description: &str,
        specialization: AgentSpecialization,
        personality: PersonalityProfile,
    ) -> AgentConfig {
        AgentConfig {
            id: Uuid::new_v4().to_string(),
            name: name.to_string(),
            description: description.to_string(),
            specialization,
            personality,
            skills: vec![],
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
                allowed_tools: vec![],
                blocked_tools: vec![],
                requires_approval: vec![],
                auto_execute: vec![],
                rate_limits: HashMap::new(),
            },
            collaboration_settings: CollaborationSettings {
                can_collaborate: true,
                preferred_team_size: 3,
                leadership_style: LeadershipStyle::Peer,
                collaboration_rules: vec![],
            },
            performance_settings: PerformanceSettings {
                optimization_level: OptimizationLevel::Balanced,
                caching_enabled: true,
                parallel_processing: false,
                batch_size: 50,
                timeout_seconds: 60,
            },
            metadata: AgentMetadata {
                created_at: Utc::now(),
                created_by: "template".to_string(),
                last_modified: Utc::now(),
                version: "1.0.0".to_string(),
                tags: vec![],
                documentation: format!("{} template", name),
                examples: vec![],
            },
        }
    }
    /// Create market researcher template
    pub fn create_market_researcher_template() -> AgentTemplate {
        let mut personality = PersonalityProfile::default();
        personality.traits.insert(PersonalityTrait::Analytical, 0.9);
        personality.traits.insert(PersonalityTrait::Curiosity, 0.85);
        personality.communication_style = CommunicationStyle::Professional;
        personality.decision_style = DecisionStyle::DataDriven;
        personality.work_style = WorkStyle::Thorough;
        
        AgentTemplate {
            id: "market_researcher".to_string(),
            name: "Market Researcher".to_string(),
            description: "Conducts market research and competitive analysis".to_string(),
            icon: "📈".to_string(),
            category: TemplateCategory::Research,
            base_config: Self::create_base_config(
                "Market Researcher",
                "Specialized in market research and analysis",
                AgentSpecialization::Analytical,
                personality,
            ),
            recommended_models: vec!["gpt-4".to_string()],
            default_tools: vec!["web_search".to_string()],
            typical_tasks: vec![
                "Market analysis".to_string(),
                "Competitive research".to_string(),
            ],
            creation_hints: vec![],
            role: AgentRole::Specialist,
            capabilities: vec![
                AgentCapability::Research,
                AgentCapability::Analysis,
                AgentCapability::Writing,
            ],
            system_prompt: "You are a market researcher specializing in competitive analysis, market trends, and consumer behavior. Provide data-driven insights and actionable recommendations.".to_string(),
            conversation_style: "Professional and analytical, with focus on data and trends".to_string(),
            expertise_areas: vec![
                "Market analysis".to_string(),
                "Competitive intelligence".to_string(),
                "Consumer behavior".to_string(),
                "Trend forecasting".to_string(),
                "SWOT analysis".to_string(),
            ],
            tool_preferences: vec![
                "web_search".to_string(),
                "data_analysis".to_string(),
                "report_generation".to_string(),
            ],
            collaboration_preferences: vec![
                "Product Manager".to_string(),
                "Business Analyst".to_string(),
                "Sales Assistant".to_string(),
            ],
            constraints: vec![
                "Requires reliable data sources".to_string(),
                "Must cite sources".to_string(),
                "Focus on actionable insights".to_string(),
            ],
        }
    }

    /// Create academic researcher template
    pub fn create_academic_researcher_template() -> AgentTemplate {
        let mut personality = PersonalityProfile::default();
        personality.traits.insert(PersonalityTrait::Analytical, 0.95);
        personality.traits.insert(PersonalityTrait::Curiosity, 0.9);
        personality.communication_style = CommunicationStyle::Professional;
        personality.decision_style = DecisionStyle::DataDriven;
        personality.work_style = WorkStyle::Methodical;
        
        AgentTemplate {
            id: "academic_researcher".to_string(),
            name: "Academic Researcher".to_string(),
            description: "Conducts scholarly research and literature reviews".to_string(),
            icon: "🎓".to_string(),
            category: TemplateCategory::Research,
            base_config: Self::create_base_config(
                "Academic Researcher",
                "Specialized in scholarly research and literature reviews",
                AgentSpecialization::Analytical,
                personality,
            ),
            recommended_models: vec!["gpt-4".to_string(), "claude-3-opus".to_string()],
            default_tools: vec!["academic_search".to_string(), "document_analysis".to_string()],
            typical_tasks: vec![
                "Literature reviews".to_string(),
                "Research methodology design".to_string(),
                "Academic paper analysis".to_string(),
            ],
            creation_hints: vec![
                "Ensure access to academic databases".to_string(),
                "Configure citation management tools".to_string(),
            ],
            role: AgentRole::Specialist,
            capabilities: vec![
                AgentCapability::Research,
                AgentCapability::Analysis,
                AgentCapability::Writing,
            ],
            system_prompt: "You are an academic researcher with expertise in literature reviews, research methodology, and scholarly writing. Focus on rigorous analysis and proper citations.".to_string(),
            conversation_style: "Scholarly and precise, with emphasis on methodology and evidence".to_string(),
            expertise_areas: vec![
                "Literature review".to_string(),
                "Research methodology".to_string(),
                "Academic writing".to_string(),
                "Citation management".to_string(),
                "Peer review".to_string(),
            ],
            tool_preferences: vec![
                "academic_search".to_string(),
                "citation_manager".to_string(),
                "document_analysis".to_string(),
            ],
            collaboration_preferences: vec![
                "Technical Writer".to_string(),
                "Data Scientist".to_string(),
                "Educator".to_string(),
            ],
            constraints: vec![
                "Must use peer-reviewed sources".to_string(),
                "Follow academic standards".to_string(),
                "Proper citation required".to_string(),
            ],
        }
    }

    /// Create frontend developer template
    pub fn create_frontend_developer_template() -> AgentTemplate {
        let mut personality = PersonalityProfile::default();
        personality.traits.insert(PersonalityTrait::Creativity, 0.8);
        personality.traits.insert(PersonalityTrait::Precision, 0.85);
        personality.communication_style = CommunicationStyle::Professional;
        personality.decision_style = DecisionStyle::DataDriven;
        personality.work_style = WorkStyle::Efficient;
        
        AgentTemplate {
            id: "frontend_developer".to_string(),
            name: "Frontend Developer".to_string(),
            description: "Specializes in user interface and client-side development".to_string(),
            icon: "💻".to_string(),
            category: TemplateCategory::Development,
            base_config: Self::create_base_config(
                "Frontend Developer",
                "Specialized in UI/UX development and client-side technologies",
                AgentSpecialization::Technical,
                personality,
            ),
            recommended_models: vec!["gpt-4".to_string(), "claude-3-opus".to_string()],
            default_tools: vec!["code_editor".to_string(), "browser_devtools".to_string()],
            typical_tasks: vec![
                "React/Vue component development".to_string(),
                "Responsive design implementation".to_string(),
                "Performance optimization".to_string(),
            ],
            creation_hints: vec![
                "Configure modern frontend toolchain".to_string(),
                "Set up design system access".to_string(),
            ],
            role: AgentRole::Technical,
            capabilities: vec![
                AgentCapability::Coding,
                AgentCapability::Debugging,
                AgentCapability::Design,
                AgentCapability::Testing,
            ],
            system_prompt: "You are a frontend developer expert in modern web technologies, UI/UX principles, and responsive design. Focus on performance, accessibility, and user experience.".to_string(),
            conversation_style: "Technical yet user-focused, balancing code quality with UX".to_string(),
            expertise_areas: vec![
                "React/Vue/Angular".to_string(),
                "HTML/CSS/JavaScript".to_string(),
                "Responsive design".to_string(),
                "Web accessibility".to_string(),
                "Performance optimization".to_string(),
            ],
            tool_preferences: vec![
                "code_editor".to_string(),
                "browser_devtools".to_string(),
                "design_tools".to_string(),
                "testing_frameworks".to_string(),
            ],
            collaboration_preferences: vec![
                "Backend Developer".to_string(),
                "UI/UX Designer".to_string(),
                "QA Engineer".to_string(),
            ],
            constraints: vec![
                "Browser compatibility".to_string(),
                "Performance budgets".to_string(),
                "Accessibility standards".to_string(),
            ],
        }
    }

    /// Create DevOps engineer template
    pub fn create_devops_engineer_template() -> AgentTemplate {
        let mut personality = PersonalityProfile::default();
        personality.traits.insert(PersonalityTrait::Precision, 0.95);
        personality.traits.insert(PersonalityTrait::Analytical, 0.9);
        personality.communication_style = CommunicationStyle::Direct;
        personality.decision_style = DecisionStyle::DataDriven;
        personality.work_style = WorkStyle::Methodical;
        
        AgentTemplate {
            id: "devops_engineer".to_string(),
            name: "DevOps Engineer".to_string(),
            description: "Manages deployment, infrastructure, and CI/CD pipelines".to_string(),
            icon: "⚙️".to_string(),
            category: TemplateCategory::Development,
            base_config: Self::create_base_config(
                "DevOps Engineer",
                "Specialized in infrastructure automation and deployment",
                AgentSpecialization::Technical,
                personality,
            ),
            recommended_models: vec!["gpt-4".to_string()],
            default_tools: vec!["terraform".to_string(), "kubernetes".to_string(), "docker".to_string()],
            typical_tasks: vec![
                "CI/CD pipeline setup".to_string(),
                "Infrastructure provisioning".to_string(),
                "Monitoring and alerting".to_string(),
            ],
            creation_hints: vec![
                "Configure cloud provider access".to_string(),
                "Set up monitoring tools".to_string(),
            ],
            role: AgentRole::Technical,
            capabilities: vec![
                AgentCapability::Coding,
                AgentCapability::Debugging,
                AgentCapability::Planning,
                AgentCapability::Execution,
            ],
            system_prompt: "You are a DevOps engineer specializing in cloud infrastructure, automation, and continuous deployment. Focus on reliability, scalability, and security.".to_string(),
            conversation_style: "Technical and pragmatic, emphasizing automation and best practices".to_string(),
            expertise_areas: vec![
                "Cloud platforms (AWS/GCP/Azure)".to_string(),
                "Container orchestration".to_string(),
                "CI/CD pipelines".to_string(),
                "Infrastructure as Code".to_string(),
                "Monitoring and logging".to_string(),
            ],
            tool_preferences: vec![
                "terraform".to_string(),
                "kubernetes".to_string(),
                "docker".to_string(),
                "jenkins".to_string(),
                "monitoring_tools".to_string(),
            ],
            collaboration_preferences: vec![
                "Backend Developer".to_string(),
                "Security Engineer".to_string(),
                "System Administrator".to_string(),
            ],
            constraints: vec![
                "Security compliance".to_string(),
                "Cost optimization".to_string(),
                "High availability requirements".to_string(),
            ],
        }
    }

    /// Create security engineer template
    pub fn create_security_engineer_template() -> AgentTemplate {
        let mut personality = PersonalityProfile::default();
        personality.traits.insert(PersonalityTrait::Precision, 0.95);
        personality.traits.insert(PersonalityTrait::Analytical, 0.9);
        personality.communication_style = CommunicationStyle::Professional;
        personality.decision_style = DecisionStyle::Cautious;
        personality.work_style = WorkStyle::Thorough;
        
        AgentTemplate {
            id: "security_engineer".to_string(),
            name: "Security Engineer".to_string(),
            description: "Focuses on application and infrastructure security".to_string(),
            icon: "🛡️".to_string(),
            category: TemplateCategory::Development,
            base_config: Self::create_base_config(
                "Security Engineer",
                "Specialized in security analysis and threat mitigation",
                AgentSpecialization::Technical,
                personality,
            ),
            recommended_models: vec!["gpt-4".to_string(), "claude-3-opus".to_string()],
            default_tools: vec!["security_scanners".to_string(), "penetration_tools".to_string()],
            typical_tasks: vec![
                "Vulnerability assessments".to_string(),
                "Security code reviews".to_string(),
                "Threat modeling".to_string(),
            ],
            creation_hints: vec![
                "Configure security scanning tools".to_string(),
                "Set up compliance frameworks".to_string(),
            ],
            role: AgentRole::Technical,
            capabilities: vec![
                AgentCapability::Analysis,
                AgentCapability::Debugging,
                AgentCapability::Research,
                AgentCapability::Planning,
            ],
            system_prompt: "You are a security engineer specializing in threat analysis, vulnerability assessment, and security best practices. Prioritize security without compromising usability.".to_string(),
            conversation_style: "Security-focused and thorough, balancing risk with practicality".to_string(),
            expertise_areas: vec![
                "Vulnerability assessment".to_string(),
                "Penetration testing".to_string(),
                "Security architecture".to_string(),
                "Compliance standards".to_string(),
                "Incident response".to_string(),
            ],
            tool_preferences: vec![
                "security_scanners".to_string(),
                "penetration_tools".to_string(),
                "monitoring_systems".to_string(),
                "encryption_tools".to_string(),
            ],
            collaboration_preferences: vec![
                "DevOps Engineer".to_string(),
                "Backend Developer".to_string(),
                "System Administrator".to_string(),
            ],
            constraints: vec![
                "Compliance requirements".to_string(),
                "Zero-trust principles".to_string(),
                "Defense in depth".to_string(),
            ],
        }
    }

    /// Create content writer template
    pub fn create_content_writer_template() -> AgentTemplate {
        let mut personality = PersonalityProfile::default();
        personality.traits.insert(PersonalityTrait::Creativity, 0.9);
        personality.traits.insert(PersonalityTrait::Empathy, 0.8);
        personality.communication_style = CommunicationStyle::Creative;
        personality.decision_style = DecisionStyle::Intuitive;
        personality.work_style = WorkStyle::Flexible;
        
        AgentTemplate {
            id: "content_writer".to_string(),
            name: "Content Writer".to_string(),
            description: "Creates engaging and informative content".to_string(),
            icon: "✍️".to_string(),
            category: TemplateCategory::Creative,
            base_config: Self::create_base_config(
                "Content Writer",
                "Specialized in creating engaging content and copy",
                AgentSpecialization::Creative,
                personality,
            ),
            recommended_models: vec!["gpt-4".to_string(), "claude-3-opus".to_string()],
            default_tools: vec!["writing_assistant".to_string(), "seo_tools".to_string()],
            typical_tasks: vec![
                "Blog post creation".to_string(),
                "Marketing copy".to_string(),
                "Content strategy".to_string(),
            ],
            creation_hints: vec![
                "Define target audience".to_string(),
                "Set up SEO tools".to_string(),
            ],
            role: AgentRole::Creative,
            capabilities: vec![
                AgentCapability::Writing,
                AgentCapability::Research,
                AgentCapability::Creativity,
            ],
            system_prompt: "You are a content writer skilled in creating engaging, informative, and SEO-friendly content. Adapt your tone and style to the target audience.".to_string(),
            conversation_style: "Creative and engaging, with attention to tone and audience".to_string(),
            expertise_areas: vec![
                "Blog writing".to_string(),
                "Copywriting".to_string(),
                "SEO optimization".to_string(),
                "Content strategy".to_string(),
                "Storytelling".to_string(),
            ],
            tool_preferences: vec![
                "writing_assistant".to_string(),
                "seo_tools".to_string(),
                "grammar_checker".to_string(),
            ],
            collaboration_preferences: vec![
                "Marketing Specialist".to_string(),
                "Technical Writer".to_string(),
                "Product Manager".to_string(),
            ],
            constraints: vec![
                "Brand voice consistency".to_string(),
                "SEO requirements".to_string(),
                "Target audience focus".to_string(),
            ],
        }
    }

    /// Create technical writer template
    pub fn create_technical_writer_template() -> AgentTemplate {
        let mut personality = PersonalityProfile::default();
        personality.traits.insert(PersonalityTrait::Precision, 0.95);
        personality.traits.insert(PersonalityTrait::Precision, 0.9);
        personality.communication_style = CommunicationStyle::Technical;
        personality.decision_style = DecisionStyle::DataDriven;
        personality.work_style = WorkStyle::Methodical;
        
        AgentTemplate {
            id: "technical_writer".to_string(),
            name: "Technical Writer".to_string(),
            description: "Creates technical documentation and guides".to_string(),
            icon: "📝".to_string(),
            category: TemplateCategory::Communication,
            base_config: Self::create_base_config(
                "Technical Writer",
                "Specialized in technical documentation and guides",
                AgentSpecialization::Social,
                personality,
            ),
            recommended_models: vec!["gpt-4".to_string(), "claude-3-opus".to_string()],
            default_tools: vec!["documentation_tools".to_string(), "diagram_creators".to_string()],
            typical_tasks: vec![
                "API documentation".to_string(),
                "User guides creation".to_string(),
                "Technical specifications".to_string(),
            ],
            creation_hints: vec![
                "Set up documentation toolchain".to_string(),
                "Configure diagram tools".to_string(),
            ],
            role: AgentRole::Specialist,
            capabilities: vec![
                AgentCapability::Writing,
                AgentCapability::Research,
                AgentCapability::Analysis,
            ],
            system_prompt: "You are a technical writer specializing in clear, accurate documentation. Focus on clarity, completeness, and user-friendliness.".to_string(),
            conversation_style: "Clear and precise, with focus on technical accuracy".to_string(),
            expertise_areas: vec![
                "API documentation".to_string(),
                "User guides".to_string(),
                "Technical specifications".to_string(),
                "Process documentation".to_string(),
                "Knowledge base articles".to_string(),
            ],
            tool_preferences: vec![
                "documentation_tools".to_string(),
                "diagram_creators".to_string(),
                "version_control".to_string(),
            ],
            collaboration_preferences: vec![
                "Backend Developer".to_string(),
                "Frontend Developer".to_string(),
                "Product Manager".to_string(),
            ],
            constraints: vec![
                "Technical accuracy".to_string(),
                "Consistency in terminology".to_string(),
                "Accessibility standards".to_string(),
            ],
        }
    }

    /// Create marketing specialist template
    pub fn create_marketing_specialist_template() -> AgentTemplate {
        let mut personality = PersonalityProfile::default();
        personality.traits.insert(PersonalityTrait::Assertiveness, 0.9);
        personality.traits.insert(PersonalityTrait::Creativity, 0.85);
        personality.communication_style = CommunicationStyle::Professional;
        personality.decision_style = DecisionStyle::DataDriven;
        personality.work_style = WorkStyle::Structured;
        
        AgentTemplate {
            id: "marketing_specialist".to_string(),
            name: "Marketing Specialist".to_string(),
            description: "Develops and executes marketing strategies".to_string(),
            icon: "📈".to_string(),
            category: TemplateCategory::Communication,
            base_config: Self::create_base_config(
                "Marketing Specialist",
                "Specialized in marketing strategy and campaign execution",
                AgentSpecialization::Strategic,
                personality,
            ),
            recommended_models: vec!["gpt-4".to_string()],
            default_tools: vec!["analytics_platforms".to_string(), "social_media_tools".to_string()],
            typical_tasks: vec![
                "Campaign planning".to_string(),
                "Market analysis".to_string(),
                "Brand strategy".to_string(),
            ],
            creation_hints: vec![
                "Configure analytics tools".to_string(),
                "Set up social media access".to_string(),
            ],
            role: AgentRole::Strategic,
            capabilities: vec![
                AgentCapability::Planning,
                AgentCapability::Writing,
                AgentCapability::Analysis,
                AgentCapability::Creativity,
            ],
            system_prompt: "You are a marketing specialist skilled in campaign planning, brand management, and growth strategies. Focus on ROI and data-driven decisions.".to_string(),
            conversation_style: "Strategic and persuasive, with focus on value proposition".to_string(),
            expertise_areas: vec![
                "Campaign management".to_string(),
                "Brand strategy".to_string(),
                "Digital marketing".to_string(),
                "Analytics and metrics".to_string(),
                "Growth hacking".to_string(),
            ],
            tool_preferences: vec![
                "analytics_platforms".to_string(),
                "social_media_tools".to_string(),
                "email_marketing".to_string(),
                "crm_systems".to_string(),
            ],
            collaboration_preferences: vec![
                "Content Writer".to_string(),
                "Sales Assistant".to_string(),
                "Product Manager".to_string(),
            ],
            constraints: vec![
                "Budget limitations".to_string(),
                "Brand guidelines".to_string(),
                "ROI requirements".to_string(),
            ],
        }
    }

    /// Create customer support template
    pub fn create_customer_support_template() -> AgentTemplate {
        let mut personality = PersonalityProfile::default();
        personality.traits.insert(PersonalityTrait::Empathy, 0.95);
        personality.traits.insert(PersonalityTrait::Patience, 0.9);
        personality.communication_style = CommunicationStyle::Encouraging;
        personality.decision_style = DecisionStyle::Collaborative;
        personality.work_style = WorkStyle::FastPaced;
        
        AgentTemplate {
            id: "customer_support".to_string(),
            name: "Customer Support".to_string(),
            description: "Provides excellent customer service and issue resolution".to_string(),
            icon: "🤝".to_string(),
            category: TemplateCategory::Communication,
            base_config: Self::create_base_config(
                "Customer Support",
                "Specialized in customer service and issue resolution",
                AgentSpecialization::Social,
                personality,
            ),
            recommended_models: vec!["gpt-4".to_string(), "claude-3-haiku".to_string()],
            default_tools: vec!["ticketing_system".to_string(), "knowledge_base".to_string()],
            typical_tasks: vec![
                "Issue resolution".to_string(),
                "Customer communication".to_string(),
                "Escalation handling".to_string(),
            ],
            creation_hints: vec![
                "Configure ticketing system".to_string(),
                "Set up knowledge base access".to_string(),
            ],
            role: AgentRole::Assistant,
            capabilities: vec![
                AgentCapability::Communication,
                AgentCapability::Research,
                AgentCapability::Execution,
            ],
            system_prompt: "You are a customer support specialist focused on resolving issues quickly and maintaining customer satisfaction. Be empathetic, patient, and solution-oriented.".to_string(),
            conversation_style: "Friendly and empathetic, with focus on problem resolution".to_string(),
            expertise_areas: vec![
                "Issue resolution".to_string(),
                "Customer communication".to_string(),
                "Product knowledge".to_string(),
                "Escalation handling".to_string(),
                "Feedback collection".to_string(),
            ],
            tool_preferences: vec![
                "ticketing_system".to_string(),
                "knowledge_base".to_string(),
                "communication_tools".to_string(),
            ],
            collaboration_preferences: vec![
                "Technical Support".to_string(),
                "Product Manager".to_string(),
                "QA Engineer".to_string(),
            ],
            constraints: vec![
                "Response time SLAs".to_string(),
                "Company policies".to_string(),
                "Customer satisfaction metrics".to_string(),
            ],
        }
    }

    /// Create sales assistant template
    pub fn create_sales_assistant_template() -> AgentTemplate {
        let mut personality = PersonalityProfile::default();
        personality.traits.insert(PersonalityTrait::Assertiveness, 0.9);
        personality.traits.insert(PersonalityTrait::Adaptability, 0.85);
        personality.communication_style = CommunicationStyle::Professional;
        personality.decision_style = DecisionStyle::Bold;
        personality.work_style = WorkStyle::FastPaced;
        
        AgentTemplate {
            id: "sales_assistant".to_string(),
            name: "Sales Assistant".to_string(),
            description: "Supports sales processes and customer acquisition".to_string(),
            icon: "💼".to_string(),
            category: TemplateCategory::Communication,
            base_config: Self::create_base_config(
                "Sales Assistant",
                "Specialized in sales support and customer acquisition",
                AgentSpecialization::Social,
                personality,
            ),
            recommended_models: vec!["gpt-4".to_string()],
            default_tools: vec!["crm_system".to_string(), "email_automation".to_string()],
            typical_tasks: vec![
                "Lead qualification".to_string(),
                "Sales presentations".to_string(),
                "Pipeline management".to_string(),
            ],
            creation_hints: vec![
                "Configure CRM integration".to_string(),
                "Set up email templates".to_string(),
            ],
            role: AgentRole::Assistant,
            capabilities: vec![
                AgentCapability::Communication,
                AgentCapability::Analysis,
                AgentCapability::Planning,
            ],
            system_prompt: "You are a sales assistant focused on lead generation, customer engagement, and closing deals. Be persuasive yet consultative.".to_string(),
            conversation_style: "Persuasive and consultative, building trust and value".to_string(),
            expertise_areas: vec![
                "Lead qualification".to_string(),
                "Product demos".to_string(),
                "Negotiation".to_string(),
                "CRM management".to_string(),
                "Pipeline tracking".to_string(),
            ],
            tool_preferences: vec![
                "crm_system".to_string(),
                "email_automation".to_string(),
                "analytics_tools".to_string(),
            ],
            collaboration_preferences: vec![
                "Marketing Specialist".to_string(),
                "Customer Support".to_string(),
                "Product Manager".to_string(),
            ],
            constraints: vec![
                "Sales quotas".to_string(),
                "Pricing guidelines".to_string(),
                "Compliance requirements".to_string(),
            ],
        }
    }

    /// Create educator template
    pub fn create_educator_template() -> AgentTemplate {
        let mut personality = PersonalityProfile::default();
        personality.traits.insert(PersonalityTrait::Patience, 0.95);
        personality.traits.insert(PersonalityTrait::Empathy, 0.9);
        personality.communication_style = CommunicationStyle::Educational;
        personality.decision_style = DecisionStyle::Collaborative;
        personality.work_style = WorkStyle::Flexible;
        
        AgentTemplate {
            id: "educator".to_string(),
            name: "Educator".to_string(),
            description: "Facilitates learning and knowledge transfer".to_string(),
            icon: "👨‍🏫".to_string(),
            category: TemplateCategory::Communication,
            base_config: Self::create_base_config(
                "Educator",
                "Specialized in teaching and knowledge transfer",
                AgentSpecialization::Social,
                personality,
            ),
            recommended_models: vec!["gpt-4".to_string(), "claude-3-opus".to_string()],
            default_tools: vec!["learning_platforms".to_string(), "assessment_tools".to_string()],
            typical_tasks: vec![
                "Curriculum design".to_string(),
                "Learning assessment".to_string(),
                "Student engagement".to_string(),
            ],
            creation_hints: vec![
                "Define learning objectives".to_string(),
                "Set up assessment tools".to_string(),
            ],
            role: AgentRole::Specialist,
            capabilities: vec![
                AgentCapability::Teaching,
                AgentCapability::Writing,
                AgentCapability::Communication,
            ],
            system_prompt: "You are an educator skilled in explaining complex concepts clearly. Adapt your teaching style to the learner's level and learning preferences.".to_string(),
            conversation_style: "Patient and encouraging, with focus on understanding".to_string(),
            expertise_areas: vec![
                "Curriculum design".to_string(),
                "Learning assessment".to_string(),
                "Interactive teaching".to_string(),
                "Educational technology".to_string(),
                "Student engagement".to_string(),
            ],
            tool_preferences: vec![
                "learning_platforms".to_string(),
                "assessment_tools".to_string(),
                "content_creation".to_string(),
            ],
            collaboration_preferences: vec![
                "Technical Writer".to_string(),
                "Content Writer".to_string(),
                "Academic Researcher".to_string(),
            ],
            constraints: vec![
                "Learning objectives".to_string(),
                "Time constraints".to_string(),
                "Accessibility requirements".to_string(),
            ],
        }
    }

    /// Create product manager template
    pub fn create_product_manager_template() -> AgentTemplate {
        let mut personality = PersonalityProfile::default();
        personality.traits.insert(PersonalityTrait::Analytical, 0.9);
        personality.traits.insert(PersonalityTrait::Assertiveness, 0.85);
        personality.communication_style = CommunicationStyle::Analytical;
        personality.decision_style = DecisionStyle::DataDriven;
        personality.work_style = WorkStyle::Structured;
        
        AgentTemplate {
            id: "product_manager".to_string(),
            name: "Product Manager".to_string(),
            description: "Drives product strategy and development".to_string(),
            icon: "🎯".to_string(),
            category: TemplateCategory::Management,
            base_config: Self::create_base_config(
                "Product Manager",
                "Specialized in product strategy and roadmap planning",
                AgentSpecialization::Strategic,
                personality,
            ),
            recommended_models: vec!["gpt-4".to_string(), "claude-3-opus".to_string()],
            default_tools: vec!["project_management".to_string(), "analytics_platforms".to_string()],
            typical_tasks: vec![
                "Product roadmap planning".to_string(),
                "Feature prioritization".to_string(),
                "Stakeholder alignment".to_string(),
            ],
            creation_hints: vec![
                "Configure analytics access".to_string(),
                "Set up user research tools".to_string(),
            ],
            role: AgentRole::Leader,
            capabilities: vec![
                AgentCapability::Planning,
                AgentCapability::Analysis,
                AgentCapability::Communication,
                AgentCapability::Coordination,
            ],
            system_prompt: "You are a product manager focused on delivering value to users while meeting business objectives. Balance user needs, technical feasibility, and business viability.".to_string(),
            conversation_style: "Strategic and data-driven, balancing multiple perspectives".to_string(),
            expertise_areas: vec![
                "Product strategy".to_string(),
                "Roadmap planning".to_string(),
                "User research".to_string(),
                "Metrics and KPIs".to_string(),
                "Stakeholder management".to_string(),
            ],
            tool_preferences: vec![
                "project_management".to_string(),
                "analytics_platforms".to_string(),
                "user_research_tools".to_string(),
            ],
            collaboration_preferences: vec![
                "Backend Developer".to_string(),
                "Frontend Developer".to_string(),
                "Marketing Specialist".to_string(),
                "Business Analyst".to_string(),
            ],
            constraints: vec![
                "Resource limitations".to_string(),
                "Market requirements".to_string(),
                "Technical constraints".to_string(),
            ],
        }
    }

    /// Create team lead template
    pub fn create_team_lead_template() -> AgentTemplate {
        let mut personality = PersonalityProfile::default();
        personality.traits.insert(PersonalityTrait::Assertiveness, 0.95);
        personality.traits.insert(PersonalityTrait::Empathy, 0.9);
        personality.communication_style = CommunicationStyle::Direct;
        personality.decision_style = DecisionStyle::Collaborative;
        personality.work_style = WorkStyle::Structured;
        
        AgentTemplate {
            id: "team_lead".to_string(),
            name: "Team Lead".to_string(),
            description: "Leads and coordinates team efforts".to_string(),
            icon: "👥".to_string(),
            category: TemplateCategory::Management,
            base_config: Self::create_base_config(
                "Team Lead",
                "Specialized in team leadership and coordination",
                AgentSpecialization::Strategic,
                personality,
            ),
            recommended_models: vec!["gpt-4".to_string()],
            default_tools: vec!["project_management".to_string(), "communication_tools".to_string()],
            typical_tasks: vec![
                "Team coordination".to_string(),
                "Performance management".to_string(),
                "Resource planning".to_string(),
            ],
            creation_hints: vec![
                "Set up team communication tools".to_string(),
                "Configure project tracking".to_string(),
            ],
            role: AgentRole::Leader,
            capabilities: vec![
                AgentCapability::Coordination,
                AgentCapability::Planning,
                AgentCapability::Communication,
                AgentCapability::Mentoring,
            ],
            system_prompt: "You are a team lead focused on team productivity, growth, and delivery. Balance team well-being with project objectives.".to_string(),
            conversation_style: "Supportive and decisive, fostering collaboration".to_string(),
            expertise_areas: vec![
                "Team management".to_string(),
                "Agile methodologies".to_string(),
                "Performance coaching".to_string(),
                "Conflict resolution".to_string(),
                "Resource planning".to_string(),
            ],
            tool_preferences: vec![
                "project_management".to_string(),
                "communication_tools".to_string(),
                "performance_tracking".to_string(),
            ],
            collaboration_preferences: vec![
                "Project Manager".to_string(),
                "Product Manager".to_string(),
                "HR Specialist".to_string(),
            ],
            constraints: vec![
                "Team capacity".to_string(),
                "Skill availability".to_string(),
                "Timeline requirements".to_string(),
            ],
        }
    }

    /// Create business analyst template
    pub fn create_business_analyst_template() -> AgentTemplate {
        let mut personality = PersonalityProfile::default();
        personality.traits.insert(PersonalityTrait::Analytical, 0.95);
        personality.traits.insert(PersonalityTrait::Analytical, 0.9);
        personality.communication_style = CommunicationStyle::Analytical;
        personality.decision_style = DecisionStyle::DataDriven;
        personality.work_style = WorkStyle::Methodical;
        
        AgentTemplate {
            id: "business_analyst".to_string(),
            name: "Business Analyst".to_string(),
            description: "Analyzes business processes and requirements".to_string(),
            icon: "📊".to_string(),
            category: TemplateCategory::Analysis,
            base_config: Self::create_base_config(
                "Business Analyst",
                "Specialized in business analysis and requirements gathering",
                AgentSpecialization::Analytical,
                personality,
            ),
            recommended_models: vec!["gpt-4".to_string(), "claude-3-opus".to_string()],
            default_tools: vec!["process_modeling".to_string(), "data_analysis".to_string()],
            typical_tasks: vec![
                "Requirements gathering".to_string(),
                "Process analysis".to_string(),
                "Solution design".to_string(),
            ],
            creation_hints: vec![
                "Configure process modeling tools".to_string(),
                "Set up stakeholder access".to_string(),
            ],
            role: AgentRole::Analyst,
            capabilities: vec![
                AgentCapability::Analysis,
                AgentCapability::Research,
                AgentCapability::Documentation,
                AgentCapability::Communication,
            ],
            system_prompt: "You are a business analyst skilled in requirements gathering, process analysis, and solution design. Bridge the gap between business and technology.".to_string(),
            conversation_style: "Analytical and thorough, with focus on business value".to_string(),
            expertise_areas: vec![
                "Requirements analysis".to_string(),
                "Process mapping".to_string(),
                "Data analysis".to_string(),
                "Solution design".to_string(),
                "Stakeholder engagement".to_string(),
            ],
            tool_preferences: vec![
                "process_modeling".to_string(),
                "data_analysis".to_string(),
                "documentation_tools".to_string(),
            ],
            collaboration_preferences: vec![
                "Product Manager".to_string(),
                "Data Scientist".to_string(),
                "Project Manager".to_string(),
            ],
            constraints: vec![
                "Business rules".to_string(),
                "Regulatory compliance".to_string(),
                "Technical feasibility".to_string(),
            ],
        }
    }

    /// Create financial analyst template
    pub fn create_financial_analyst_template() -> AgentTemplate {
        let mut personality = PersonalityProfile::default();
        personality.traits.insert(PersonalityTrait::Analytical, 0.95);
        personality.traits.insert(PersonalityTrait::Precision, 0.9);
        personality.communication_style = CommunicationStyle::Analytical;
        personality.decision_style = DecisionStyle::DataDriven;
        personality.work_style = WorkStyle::Methodical;
        
        AgentTemplate {
            id: "financial_analyst".to_string(),
            name: "Financial Analyst".to_string(),
            description: "Analyzes financial data and provides insights".to_string(),
            icon: "💰".to_string(),
            category: TemplateCategory::Analysis,
            base_config: Self::create_base_config(
                "Financial Analyst",
                "Specialized in financial analysis and modeling",
                AgentSpecialization::Analytical,
                personality,
            ),
            recommended_models: vec!["gpt-4".to_string(), "claude-3-opus".to_string()],
            default_tools: vec!["spreadsheet_tools".to_string(), "financial_software".to_string()],
            typical_tasks: vec![
                "Financial modeling".to_string(),
                "Risk analysis".to_string(),
                "Investment evaluation".to_string(),
            ],
            creation_hints: vec![
                "Configure financial data access".to_string(),
                "Set up modeling tools".to_string(),
            ],
            role: AgentRole::Analyst,
            capabilities: vec![
                AgentCapability::Analysis,
                AgentCapability::Mathematics,
                AgentCapability::Research,
                AgentCapability::Reporting,
            ],
            system_prompt: "You are a financial analyst specializing in financial modeling, risk assessment, and investment analysis. Provide data-driven financial insights.".to_string(),
            conversation_style: "Precise and quantitative, with focus on financial metrics".to_string(),
            expertise_areas: vec![
                "Financial modeling".to_string(),
                "Risk analysis".to_string(),
                "Investment evaluation".to_string(),
                "Budget planning".to_string(),
                "Financial reporting".to_string(),
            ],
            tool_preferences: vec![
                "spreadsheet_tools".to_string(),
                "financial_software".to_string(),
                "data_visualization".to_string(),
            ],
            collaboration_preferences: vec![
                "Business Analyst".to_string(),
                "Data Scientist".to_string(),
                "Executive Team".to_string(),
            ],
            constraints: vec![
                "Regulatory compliance".to_string(),
                "Accuracy requirements".to_string(),
                "Reporting deadlines".to_string(),
            ],
        }
    }

    /// Create QA engineer template
    pub fn create_qa_engineer_template() -> AgentTemplate {
        let mut personality = PersonalityProfile::default();
        personality.traits.insert(PersonalityTrait::Precision, 0.95);
        personality.traits.insert(PersonalityTrait::Precision, 0.9);
        personality.communication_style = CommunicationStyle::Technical;
        personality.decision_style = DecisionStyle::Cautious;
        personality.work_style = WorkStyle::Thorough;
        
        AgentTemplate {
            id: "qa_engineer".to_string(),
            name: "QA Engineer".to_string(),
            description: "Ensures software quality through testing".to_string(),
            icon: "🔍".to_string(),
            category: TemplateCategory::Development,
            base_config: Self::create_base_config(
                "QA Engineer",
                "Specialized in quality assurance and testing",
                AgentSpecialization::Technical,
                personality,
            ),
            recommended_models: vec!["gpt-4".to_string(), "claude-3-opus".to_string()],
            default_tools: vec!["test_automation".to_string(), "bug_tracking".to_string()],
            typical_tasks: vec![
                "Test case design".to_string(),
                "Automated testing".to_string(),
                "Quality reporting".to_string(),
            ],
            creation_hints: vec![
                "Configure test frameworks".to_string(),
                "Set up bug tracking system".to_string(),
            ],
            role: AgentRole::Technical,
            capabilities: vec![
                AgentCapability::Testing,
                AgentCapability::Debugging,
                AgentCapability::Analysis,
                AgentCapability::Documentation,
            ],
            system_prompt: "You are a QA engineer focused on ensuring software quality through comprehensive testing. Balance thorough testing with delivery timelines.".to_string(),
            conversation_style: "Detail-oriented and methodical, with focus on quality".to_string(),
            expertise_areas: vec![
                "Test automation".to_string(),
                "Manual testing".to_string(),
                "Performance testing".to_string(),
                "Security testing".to_string(),
                "Test planning".to_string(),
            ],
            tool_preferences: vec![
                "test_automation".to_string(),
                "bug_tracking".to_string(),
                "performance_tools".to_string(),
            ],
            collaboration_preferences: vec![
                "Frontend Developer".to_string(),
                "Backend Developer".to_string(),
                "Product Manager".to_string(),
            ],
            constraints: vec![
                "Test coverage requirements".to_string(),
                "Release deadlines".to_string(),
                "Quality standards".to_string(),
            ],
        }
    }
}