//! Agent Creation System
//! 
//! Provides a comprehensive wizard and builder system for creating and configuring
//! AI agents with specific personalities, skills, and behaviors.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};
use anyhow::Result;
use uuid::Uuid;
use chrono::{DateTime, Utc};

use crate::cognitive::agents::AgentSpecialization;
use super::registry::AgentRegistry;
use super::coordination::{Agent as CoordinationAgent, AgentState, AgentMetrics};

/// Agent creation wizard for step-by-step agent configuration
pub struct AgentCreationWizard {
    /// Current step in the wizard
    current_step: WizardStep,
    
    /// Agent being created
    agent_draft: AgentDraft,
    
    /// Available templates
    templates: Vec<AgentTemplate>,
    
    /// Personality builder
    personality_builder: PersonalityBuilder,
    
    /// Skill repository
    skill_repository: SkillRepository,
    
    /// Validation rules
    validation_rules: ValidationRules,
    
    /// Agent registry for persistence
    registry: Option<Arc<RwLock<AgentRegistry>>>,
    
    /// Available models dynamically discovered
    available_models: Arc<RwLock<Vec<String>>>,
}

/// Steps in the creation wizard
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WizardStep {
    SelectTemplate,
    BasicInfo,
    Specialization,
    Personality,
    Skills,
    ModelSelection,
    ToolSelection,
    Permissions,
    Review,
    Complete,
}

/// Draft agent being created
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentDraft {
    pub id: String,
    pub name: String,
    pub description: String,
    pub template_id: Option<String>,
    pub specialization: Option<AgentSpecialization>,
    pub personality: PersonalityProfile,
    pub skills: Vec<AgentSkill>,
    pub preferred_models: Vec<String>,
    pub allowed_tools: Vec<String>,
    pub permissions: AgentPermissions,
    pub metadata: AgentMetadata,
}

/// Agent template for quick creation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentTemplate {
    pub id: String,
    pub name: String,
    pub description: String,
    pub icon: String,
    pub category: TemplateCategory,
    pub base_config: AgentConfig,
    pub recommended_models: Vec<String>,
    pub default_tools: Vec<String>,
    pub typical_tasks: Vec<String>,
    pub creation_hints: Vec<String>,
    // Additional fields for template implementations
    pub role: crate::tui::chat::agents::templates_impl::AgentRole,
    pub capabilities: Vec<crate::tui::chat::agents::templates_impl::AgentCapability>,
    pub system_prompt: String,
    pub conversation_style: String,
    pub expertise_areas: Vec<String>,
    pub tool_preferences: Vec<String>,
    pub collaboration_preferences: Vec<String>,
    pub constraints: Vec<String>,
}

/// Template categories
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum TemplateCategory {
    Research,
    Development,
    Creative,
    Analysis,
    Communication,
    Management,
    Support,
    Custom,
}

/// Complete agent configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    pub id: String,
    pub name: String,
    pub description: String,
    pub specialization: AgentSpecialization,
    pub personality: PersonalityProfile,
    pub skills: Vec<AgentSkill>,
    pub model_preferences: ModelPreferences,
    pub tool_permissions: ToolPermissions,
    pub collaboration_settings: CollaborationSettings,
    pub performance_settings: PerformanceSettings,
    pub metadata: AgentMetadata,
}

/// Personality profile for an agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalityProfile {
    pub traits: HashMap<PersonalityTrait, f32>,
    pub communication_style: CommunicationStyle,
    pub decision_style: DecisionStyle,
    pub work_style: WorkStyle,
    pub interaction_preferences: InteractionPreferences,
    pub custom_behaviors: Vec<CustomBehavior>,
}

impl Default for PersonalityProfile {
    fn default() -> Self {
        let mut traits = HashMap::new();
        traits.insert(PersonalityTrait::Helpfulness, 0.8);
        traits.insert(PersonalityTrait::Creativity, 0.5);
        traits.insert(PersonalityTrait::Analytical, 0.6);
        traits.insert(PersonalityTrait::Assertiveness, 0.4);
        traits.insert(PersonalityTrait::Empathy, 0.7);
        
        Self {
            traits,
            communication_style: CommunicationStyle::Professional,
            decision_style: DecisionStyle::Balanced,
            work_style: WorkStyle::Methodical,
            interaction_preferences: InteractionPreferences::default(),
            custom_behaviors: Vec::new(),
        }
    }
}

/// Personality traits
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PersonalityTrait {
    Helpfulness,
    Creativity,
    Analytical,
    Assertiveness,
    Empathy,
    Curiosity,
    Precision,
    Adaptability,
    Patience,
    Enthusiasm,
}

/// Communication styles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CommunicationStyle {
    Professional,
    Casual,
    Technical,
    Educational,
    Encouraging,
    Direct,
    Diplomatic,
    Creative,
    Analytical,
}

/// Decision-making styles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DecisionStyle {
    DataDriven,
    Intuitive,
    Collaborative,
    Authoritative,
    Cautious,
    Bold,
    Balanced,
}

/// Work styles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WorkStyle {
    Methodical,
    FastPaced,
    Thorough,
    Efficient,
    Creative,
    Structured,
    Flexible,
}

/// Interaction preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionPreferences {
    pub prefers_autonomy: bool,
    pub asks_clarification: bool,
    pub provides_alternatives: bool,
    pub explains_reasoning: bool,
    pub seeks_feedback: bool,
}

impl Default for InteractionPreferences {
    fn default() -> Self {
        Self {
            prefers_autonomy: false,
            asks_clarification: true,
            provides_alternatives: true,
            explains_reasoning: true,
            seeks_feedback: false,
        }
    }
}

/// Custom behavior definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomBehavior {
    pub name: String,
    pub description: String,
    pub trigger: BehaviorTrigger,
    pub action: BehaviorAction,
}

/// Behavior triggers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BehaviorTrigger {
    OnTaskStart,
    OnTaskComplete,
    OnError,
    OnUserRequest,
    OnSchedule(String), // Cron expression
    OnCondition(String), // Custom condition
}

/// Behavior actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BehaviorAction {
    SendNotification(String),
    ExecuteCommand(String),
    UpdateState(String, serde_json::Value),
    TriggerWorkflow(String),
}

/// Agent skills
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSkill {
    pub id: String,
    pub name: String,
    pub category: SkillCategory,
    pub proficiency: f32, // 0.0 to 1.0
    pub experience_hours: u32,
    pub certifications: Vec<String>,
    pub examples: Vec<String>,
}

/// Skill categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SkillCategory {
    Programming,
    DataAnalysis,
    Writing,
    Research,
    Design,
    Communication,
    ProblemSolving,
    ProjectManagement,
    Teaching,
    Testing,
    Custom(String),
}

/// Model preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPreferences {
    pub primary_model: Option<String>,
    pub fallback_models: Vec<String>,
    pub model_selection_strategy: ModelSelectionStrategy,
    pub context_preferences: ContextPreferences,
    pub cost_threshold: Option<f64>,
}

/// Model selection strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelSelectionStrategy {
    AlwaysPrimary,
    CostOptimized,
    PerformanceOptimized,
    TaskBased,
    LoadBalanced,
    Balanced,
    Adaptive,
}

/// Context preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextPreferences {
    pub preferred_context_size: usize,
    pub include_examples: bool,
    pub include_history: bool,
    pub compression_enabled: bool,
}

/// Tool permissions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolPermissions {
    pub allowed_tools: Vec<String>,
    pub blocked_tools: Vec<String>,
    pub requires_approval: Vec<String>,
    pub auto_execute: Vec<String>,
    pub rate_limits: HashMap<String, u32>,
}

/// Agent permissions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentPermissions {
    pub can_execute_code: bool,
    pub can_access_files: bool,
    pub can_use_network: bool,
    pub can_modify_system: bool,
    pub can_access_sensitive_data: bool,
    pub max_resource_usage: ResourceLimits,
}

/// Resource limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    pub max_memory_mb: u32,
    pub max_cpu_percent: u8,
    pub max_tokens_per_request: u32,
    pub max_requests_per_minute: u32,
    pub max_execution_time_seconds: u32,
}

/// Collaboration settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborationSettings {
    pub can_collaborate: bool,
    pub preferred_team_size: usize,
    pub leadership_style: LeadershipStyle,
    pub collaboration_rules: Vec<CollaborationRule>,
}

/// Leadership styles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LeadershipStyle {
    Leader,
    Follower,
    Peer,
    Independent,
    Adaptive,
}

/// Collaboration rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborationRule {
    pub rule_type: RuleType,
    pub condition: String,
    pub action: String,
}

/// Rule types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RuleType {
    TaskAssignment,
    Communication,
    ConflictResolution,
    ResourceSharing,
    DecisionMaking,
}

/// Performance settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSettings {
    pub optimization_level: OptimizationLevel,
    pub caching_enabled: bool,
    pub parallel_processing: bool,
    pub batch_size: usize,
    pub timeout_seconds: u32,
}

/// Optimization levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationLevel {
    Balanced,
    Speed,
    Quality,
    CostEfficient,
}

/// Agent metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMetadata {
    pub created_at: DateTime<Utc>,
    pub created_by: String,
    pub last_modified: DateTime<Utc>,
    pub version: String,
    pub tags: Vec<String>,
    pub documentation: String,
    pub examples: Vec<UsageExample>,
}

/// Usage example
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageExample {
    pub name: String,
    pub description: String,
    pub input: String,
    pub expected_output: String,
}

/// Personality builder for creating personality profiles
pub struct PersonalityBuilder {
    profile: PersonalityProfile,
}

impl PersonalityBuilder {
    pub fn new() -> Self {
        Self {
            profile: PersonalityProfile::default(),
        }
    }
    
    pub fn with_trait(mut self, trait_type: PersonalityTrait, value: f32) -> Self {
        self.profile.traits.insert(trait_type, value.clamp(0.0, 1.0));
        self
    }
    
    pub fn with_communication_style(mut self, style: CommunicationStyle) -> Self {
        self.profile.communication_style = style;
        self
    }
    
    pub fn with_decision_style(mut self, style: DecisionStyle) -> Self {
        self.profile.decision_style = style;
        self
    }
    
    pub fn with_work_style(mut self, style: WorkStyle) -> Self {
        self.profile.work_style = style;
        self
    }
    
    pub fn with_interaction_preferences(mut self, prefs: InteractionPreferences) -> Self {
        self.profile.interaction_preferences = prefs;
        self
    }
    
    pub fn add_custom_behavior(mut self, behavior: CustomBehavior) -> Self {
        self.profile.custom_behaviors.push(behavior);
        self
    }
    
    pub fn build(self) -> PersonalityProfile {
        self.profile
    }
}

/// Skill repository for managing available skills
pub struct SkillRepository {
    skills: HashMap<String, SkillDefinition>,
}

/// Skill definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillDefinition {
    pub id: String,
    pub name: String,
    pub category: SkillCategory,
    pub description: String,
    pub requirements: Vec<String>,
    pub training_hours: u32,
    pub assessment_criteria: Vec<String>,
}

impl SkillRepository {
    pub fn new() -> Self {
        let mut skills = HashMap::new();
        
        // Add default skills
        skills.insert("rust_programming".to_string(), SkillDefinition {
            id: "rust_programming".to_string(),
            name: "Rust Programming".to_string(),
            category: SkillCategory::Programming,
            description: "Proficiency in Rust programming language".to_string(),
            requirements: vec!["Basic programming knowledge".to_string()],
            training_hours: 100,
            assessment_criteria: vec!["Can write safe Rust code".to_string()],
        });
        
        skills.insert("data_analysis".to_string(), SkillDefinition {
            id: "data_analysis".to_string(),
            name: "Data Analysis".to_string(),
            category: SkillCategory::DataAnalysis,
            description: "Ability to analyze and interpret data".to_string(),
            requirements: vec!["Statistics knowledge".to_string()],
            training_hours: 80,
            assessment_criteria: vec!["Can identify patterns in data".to_string()],
        });
        
        Self { skills }
    }
    
    pub fn get_skill(&self, skill_id: &str) -> Option<&SkillDefinition> {
        self.skills.get(skill_id)
    }
    
    pub fn list_skills(&self) -> Vec<&SkillDefinition> {
        self.skills.values().collect()
    }
    
    pub fn add_skill(&mut self, skill: SkillDefinition) {
        self.skills.insert(skill.id.clone(), skill);
    }
}

/// Validation rules for agent creation
#[derive(Debug, Clone)]
pub struct ValidationRules {
    pub min_name_length: usize,
    pub max_name_length: usize,
    pub required_skills: usize,
    pub max_tools: usize,
    pub max_models: usize,
}

impl Default for ValidationRules {
    fn default() -> Self {
        Self {
            min_name_length: 3,
            max_name_length: 50,
            required_skills: 1,
            max_tools: 20,
            max_models: 5,
        }
    }
}

impl ValidationRules {
    /// Validate an agent draft
    pub fn validate(&self, draft: &AgentDraft) -> ValidationResult {
        let mut errors = Vec::new();
        
        // Validate name length
        if draft.name.len() < self.min_name_length {
            errors.push(format!("Name must be at least {} characters", self.min_name_length));
        }
        if draft.name.len() > self.max_name_length {
            errors.push(format!("Name must be at most {} characters", self.max_name_length));
        }
        
        // Validate skills
        if draft.skills.len() < self.required_skills {
            errors.push(format!("At least {} skill(s) required", self.required_skills));
        }
        
        // Validate tools
        if draft.allowed_tools.len() > self.max_tools {
            errors.push(format!("Maximum {} tools allowed", self.max_tools));
        }
        
        // Validate models
        if draft.preferred_models.len() > self.max_models {
            errors.push(format!("Maximum {} models allowed", self.max_models));
        }
        
        ValidationResult {
            valid: errors.is_empty(),
            errors,
        }
    }
}

/// Validation result
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ValidationResult {
    pub valid: bool,
    pub errors: Vec<String>,
}

impl AgentCreationWizard {
    /// Create a new wizard
    pub fn new() -> Self {
        Self {
            current_step: WizardStep::SelectTemplate,
            agent_draft: AgentDraft::new(),
            templates: Self::load_templates(),
            personality_builder: PersonalityBuilder::new(),
            skill_repository: SkillRepository::new(),
            validation_rules: ValidationRules::default(),
            registry: None,
            available_models: Arc::new(RwLock::new(Vec::new())),
        }
    }
    
    /// Create a wizard with a registry
    pub fn with_registry(registry: Arc<RwLock<AgentRegistry>>) -> Self {
        let mut wizard = Self::new();
        wizard.registry = Some(registry);
        wizard
    }
    
    /// Set the agent registry
    pub fn set_registry(&mut self, registry: Arc<RwLock<AgentRegistry>>) {
        self.registry = Some(registry);
    }
    
    /// Load agent templates
    fn load_templates() -> Vec<AgentTemplate> {
        vec![
            AgentTemplate {
                id: "researcher".to_string(),
                name: "Research Assistant".to_string(),
                description: "Specialized in research and information gathering".to_string(),
                icon: "🔬".to_string(),
                category: TemplateCategory::Research,
                base_config: AgentConfig::researcher_template(),
                recommended_models: vec!["gpt-4".to_string(), "claude-3-opus".to_string()],
                default_tools: vec!["web_search".to_string(), "arxiv".to_string()],
                typical_tasks: vec![
                    "Literature review".to_string(),
                    "Fact checking".to_string(),
                    "Data gathering".to_string(),
                ],
                creation_hints: vec![
                    "Focus on accuracy and thoroughness".to_string(),
                    "Enable citation tracking".to_string(),
                ],
                role: crate::tui::chat::agents::templates_impl::AgentRole::Specialist,
                capabilities: vec![
                    crate::tui::chat::agents::templates_impl::AgentCapability::Research,
                    crate::tui::chat::agents::templates_impl::AgentCapability::Analysis,
                    crate::tui::chat::agents::templates_impl::AgentCapability::Writing,
                ],
                system_prompt: "You are a research assistant specializing in information gathering, literature reviews, and fact-checking. Provide thorough and accurate research results.".to_string(),
                conversation_style: "Methodical and analytical, focusing on accuracy and thoroughness".to_string(),
                expertise_areas: vec![
                    "Literature review".to_string(),
                    "Fact checking".to_string(),
                    "Data gathering".to_string(),
                    "Academic research".to_string(),
                    "Citation management".to_string(),
                ],
                tool_preferences: vec![
                    "web_search".to_string(),
                    "arxiv".to_string(),
                    "academic_databases".to_string(),
                ],
                collaboration_preferences: vec![
                    "Academic Researcher".to_string(),
                    "Technical Writer".to_string(),
                    "Data Scientist".to_string(),
                ],
                constraints: vec![
                    "Must cite sources".to_string(),
                    "Focus on peer-reviewed materials".to_string(),
                    "Maintain objectivity".to_string(),
                ],
            },
            AgentTemplate {
                id: "developer".to_string(),
                name: "Code Developer".to_string(),
                description: "Expert in software development and coding".to_string(),
                icon: "💻".to_string(),
                category: TemplateCategory::Development,
                base_config: AgentConfig::developer_template(),
                recommended_models: vec!["codestral".to_string(), "gpt-4".to_string()],
                default_tools: vec!["code_analysis".to_string(), "github".to_string()],
                typical_tasks: vec![
                    "Code generation".to_string(),
                    "Bug fixing".to_string(),
                    "Code review".to_string(),
                ],
                creation_hints: vec![
                    "Enable code execution carefully".to_string(),
                    "Set appropriate language preferences".to_string(),
                ],
                role: crate::tui::chat::agents::templates_impl::AgentRole::Technical,
                capabilities: vec![
                    crate::tui::chat::agents::templates_impl::AgentCapability::Coding,
                    crate::tui::chat::agents::templates_impl::AgentCapability::Debugging,
                    crate::tui::chat::agents::templates_impl::AgentCapability::Testing,
                ],
                system_prompt: "You are a code developer expert in software engineering, code quality, and best practices. Write clean, maintainable, and efficient code.".to_string(),
                conversation_style: "Technical and precise, focusing on code quality and best practices".to_string(),
                expertise_areas: vec![
                    "Software engineering".to_string(),
                    "Code architecture".to_string(),
                    "Testing strategies".to_string(),
                    "Code review".to_string(),
                    "Performance optimization".to_string(),
                ],
                tool_preferences: vec![
                    "code_analysis".to_string(),
                    "github".to_string(),
                    "testing_frameworks".to_string(),
                ],
                collaboration_preferences: vec![
                    "Backend Developer".to_string(),
                    "Frontend Developer".to_string(),
                    "DevOps Engineer".to_string(),
                ],
                constraints: vec![
                    "Follow coding standards".to_string(),
                    "Write comprehensive tests".to_string(),
                    "Consider security implications".to_string(),
                ],
            },
            AgentTemplate {
                id: "creative".to_string(),
                name: "Creative Writer".to_string(),
                description: "Specialized in creative content generation".to_string(),
                icon: "✍️".to_string(),
                category: TemplateCategory::Creative,
                base_config: AgentConfig::creative_template(),
                recommended_models: vec!["claude-3-opus".to_string(), "gpt-4".to_string()],
                default_tools: vec!["creative_media".to_string()],
                typical_tasks: vec![
                    "Story writing".to_string(),
                    "Content creation".to_string(),
                    "Brainstorming".to_string(),
                ],
                creation_hints: vec![
                    "Increase creativity parameters".to_string(),
                    "Enable diverse output generation".to_string(),
                ],
                role: crate::tui::chat::agents::templates_impl::AgentRole::Creative,
                capabilities: vec![
                    crate::tui::chat::agents::templates_impl::AgentCapability::Writing,
                    crate::tui::chat::agents::templates_impl::AgentCapability::Creativity,
                ],
                system_prompt: "You are a creative writer specializing in original content, storytelling, and creative expression. Produce engaging and imaginative content.".to_string(),
                conversation_style: "Creative and engaging, with emphasis on originality and expression".to_string(),
                expertise_areas: vec![
                    "Creative writing".to_string(),
                    "Storytelling".to_string(),
                    "Content creation".to_string(),
                    "Brainstorming".to_string(),
                    "Narrative development".to_string(),
                ],
                tool_preferences: vec![
                    "creative_media".to_string(),
                    "writing_assistant".to_string(),
                    "idea_generation".to_string(),
                ],
                collaboration_preferences: vec![
                    "Content Writer".to_string(),
                    "Marketing Specialist".to_string(),
                    "Technical Writer".to_string(),
                ],
                constraints: vec![
                    "Maintain originality".to_string(),
                    "Consider target audience".to_string(),
                    "Balance creativity with clarity".to_string(),
                ],
            },
        ]
    }
    
    /// Get current step
    pub fn current_step(&self) -> WizardStep {
        self.current_step
    }
    
    /// Move to next step
    pub fn next_step(&mut self) -> Result<()> {
        self.current_step = match self.current_step {
            WizardStep::SelectTemplate => WizardStep::BasicInfo,
            WizardStep::BasicInfo => WizardStep::Specialization,
            WizardStep::Specialization => WizardStep::Personality,
            WizardStep::Personality => WizardStep::Skills,
            WizardStep::Skills => WizardStep::ModelSelection,
            WizardStep::ModelSelection => WizardStep::ToolSelection,
            WizardStep::ToolSelection => WizardStep::Permissions,
            WizardStep::Permissions => WizardStep::Review,
            WizardStep::Review => WizardStep::Complete,
            WizardStep::Complete => return Err(anyhow::anyhow!("Wizard already complete")),
        };
        Ok(())
    }
    
    /// Move to previous step
    pub fn previous_step(&mut self) -> Result<()> {
        self.current_step = match self.current_step {
            WizardStep::SelectTemplate => return Err(anyhow::anyhow!("Already at first step")),
            WizardStep::BasicInfo => WizardStep::SelectTemplate,
            WizardStep::Specialization => WizardStep::BasicInfo,
            WizardStep::Personality => WizardStep::Specialization,
            WizardStep::Skills => WizardStep::Personality,
            WizardStep::ModelSelection => WizardStep::Skills,
            WizardStep::ToolSelection => WizardStep::ModelSelection,
            WizardStep::Permissions => WizardStep::ToolSelection,
            WizardStep::Review => WizardStep::Permissions,
            WizardStep::Complete => WizardStep::Review,
        };
        Ok(())
    }
    
    /// Select a template
    pub fn select_template(&mut self, template_id: &str) -> Result<()> {
        if let Some(template) = self.templates.iter().find(|t| t.id == template_id) {
            self.agent_draft.template_id = Some(template_id.to_string());
            // Apply template defaults
            self.agent_draft.name = template.name.clone();
            self.agent_draft.description = template.description.clone();
            self.agent_draft.preferred_models = template.recommended_models.clone();
            self.agent_draft.allowed_tools = template.default_tools.clone();
            Ok(())
        } else {
            Err(anyhow::anyhow!("Template not found"))
        }
    }
    
    /// Set basic information
    pub fn set_basic_info(&mut self, name: String, description: String) -> Result<()> {
        if name.len() < self.validation_rules.min_name_length {
            return Err(anyhow::anyhow!("Name too short"));
        }
        if name.len() > self.validation_rules.max_name_length {
            return Err(anyhow::anyhow!("Name too long"));
        }
        
        self.agent_draft.name = name;
        self.agent_draft.description = description;
        Ok(())
    }
    
    /// Set specialization
    pub fn set_specialization(&mut self, specialization: AgentSpecialization) {
        self.agent_draft.specialization = Some(specialization);
    }
    
    /// Set personality trait
    pub fn set_personality_trait(&mut self, trait_type: PersonalityTrait, value: f32) {
        self.agent_draft.personality.traits.insert(trait_type, value.clamp(0.0, 1.0));
    }
    
    /// Add skill
    pub fn add_skill(&mut self, skill: AgentSkill) {
        self.agent_draft.skills.push(skill);
    }
    
    /// Add model preference
    pub fn add_model(&mut self, model_id: String) -> Result<()> {
        if self.agent_draft.preferred_models.len() >= self.validation_rules.max_models {
            return Err(anyhow::anyhow!("Maximum models reached"));
        }
        self.agent_draft.preferred_models.push(model_id);
        Ok(())
    }
    
    /// Add tool permission
    pub fn add_tool(&mut self, tool_id: String) -> Result<()> {
        if self.agent_draft.allowed_tools.len() >= self.validation_rules.max_tools {
            return Err(anyhow::anyhow!("Maximum tools reached"));
        }
        self.agent_draft.allowed_tools.push(tool_id);
        Ok(())
    }
    
    /// Set permissions
    pub fn set_permissions(&mut self, permissions: AgentPermissions) {
        self.agent_draft.permissions = permissions;
    }
    
    /// Validate the current draft
    pub fn validate(&self) -> Result<Vec<String>> {
        let mut issues = Vec::new();
        
        if self.agent_draft.name.is_empty() {
            issues.push("Name is required".to_string());
        }
        
        if self.agent_draft.specialization.is_none() {
            issues.push("Specialization is required".to_string());
        }
        
        if self.agent_draft.skills.len() < self.validation_rules.required_skills {
            issues.push(format!("At least {} skill(s) required", self.validation_rules.required_skills));
        }
        
        if self.agent_draft.preferred_models.is_empty() {
            issues.push("At least one model must be selected".to_string());
        }
        
        if issues.is_empty() {
            Ok(Vec::new())
        } else {
            Ok(issues)
        }
    }
    
    /// Add available model to the wizard
    pub async fn add_available_model(&self, model_id: String) -> Result<()> {
        // Add to local available models list
        let mut models = self.available_models.write().await;
        if !models.contains(&model_id) {
            models.push(model_id.clone());
            tracing::debug!("Added available model to wizard: {}", model_id);
        }
        
        // Also add to registry if available
        if let Some(registry) = &self.registry {
            registry.read().await.add_available_model(model_id).await?;
        }
        
        Ok(())
    }
    
    /// Get available models
    pub async fn get_available_models(&self) -> Vec<String> {
        if let Some(registry) = &self.registry {
            // Prefer registry's model list if available
            registry.read().await.get_available_models().await
        } else {
            // Use local list
            self.available_models.read().await.clone()
        }
    }
    
    /// Set all available models at once
    pub async fn set_available_models(&self, models: Vec<String>) -> Result<()> {
        *self.available_models.write().await = models.clone();
        
        if let Some(registry) = &self.registry {
            registry.write().await.set_available_models(models).await?;
        }
        
        Ok(())
    }

    /// Complete the wizard and create the agent
    pub fn complete(self) -> Result<AgentConfig> {
        let draft = self.agent_draft;
        
        Ok(AgentConfig {
            id: draft.id,
            name: draft.name,
            description: draft.description,
            specialization: draft.specialization.unwrap_or(AgentSpecialization::Technical),
            personality: draft.personality,
            skills: draft.skills,
            model_preferences: ModelPreferences {
                primary_model: draft.preferred_models.first().cloned(),
                fallback_models: draft.preferred_models.into_iter().skip(1).collect(),
                model_selection_strategy: ModelSelectionStrategy::TaskBased,
                context_preferences: ContextPreferences {
                    preferred_context_size: 4096,
                    include_examples: true,
                    include_history: true,
                    compression_enabled: false,
                },
                cost_threshold: None,
            },
            tool_permissions: ToolPermissions {
                allowed_tools: draft.allowed_tools,
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
                parallel_processing: true,
                batch_size: 10,
                timeout_seconds: 30,
            },
            metadata: draft.metadata,
        })
    }
    
    /// Process a request to create or configure an agent
    pub async fn process_request(&self, data: serde_json::Value) -> Result<serde_json::Value> {
        // Parse the request type
        let request_type = data.get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("create");
        
        match request_type {
            "create" => {
                // Create a new agent from the provided configuration
                if let Ok(config) = serde_json::from_value::<AgentConfig>(data.get("config").cloned().unwrap_or_default()) {
                    let agent_id = config.id.clone();
                    tracing::info!("Creating agent {} via process_request", agent_id);
                    
                    // Return the created agent configuration
                    Ok(serde_json::to_value(config)?)
                } else {
                    Err(anyhow::anyhow!("Invalid agent configuration"))
                }
            },
            "list_templates" => {
                // Return available templates
                let templates = Self::load_templates();
                Ok(serde_json::to_value(templates)?)
            },
            "validate" => {
                // Validate an agent configuration
                if let Ok(draft) = serde_json::from_value::<AgentDraft>(data.get("draft").cloned().unwrap_or_default()) {
                    let validation_result = self.validation_rules.validate(&draft);
                    Ok(serde_json::to_value(validation_result)?)
                } else {
                    Err(anyhow::anyhow!("Invalid agent draft"))
                }
            },
            _ => {
                Err(anyhow::anyhow!("Unknown request type: {}", request_type))
            }
        }
    }
    
    /// Register a new agent with the wizard
    pub async fn register_agent(&self, agent_id: String, config: AgentConfig) -> Result<()> {
        tracing::info!("Registering agent {} with wizard", agent_id);
        
        // Store in registry if available
        if let Some(registry) = &self.registry {
            // Ensure the config has the correct ID
            let mut config = config.clone();
            config.id = agent_id.clone();
            
            // Register with the registry
            let mut registry_write = registry.write().await;
            let registered_id = registry_write.register(config.clone()).await?;
            
            tracing::info!("Agent {} registered in registry with ID: {}", agent_id, registered_id);
            
            // Update usage stats to mark initial creation
            registry_write.update_usage_stats(&registered_id, |stats| {
                stats.last_activated = Some(Utc::now());
            }).await?;
            
            tracing::debug!("Agent {} registered with specialization {:?}", 
                agent_id, config.specialization);
        } else {
            tracing::warn!("No registry available, agent {} configuration not persisted", agent_id);
            tracing::debug!("Agent {} registered with specialization {:?}", 
                agent_id, config.specialization);
        }
        
        Ok(())
    }
    
    /// Create an agent from configuration
    pub async fn create_agent(&self, agent_id: String) -> Result<Arc<RwLock<CoordinationAgent>>> {
        tracing::info!("Creating agent instance: {}", agent_id);
        
        // Retrieve configuration from registry
        let agent_entry = if let Some(registry) = &self.registry {
            registry.read().await.get(&agent_id).await?
                .ok_or_else(|| anyhow::anyhow!("Agent {} not found in registry", agent_id))?
        } else {
            return Err(anyhow::anyhow!("No registry available to retrieve agent configuration"));
        };
        
        let config = agent_entry.config;
        
        // Create the coordination agent with proper fields
        let coord_agent = CoordinationAgent {
            id: config.id.clone(),
            name: config.name.clone(),
            specialization: config.specialization.clone(),
            state: AgentState::Idle,
            metrics: AgentMetrics {
                tasks_completed: agent_entry.metadata.usage_stats.successful_tasks,
                tasks_failed: agent_entry.metadata.usage_stats.failed_tasks,
                avg_completion_time_ms: agent_entry.metadata.usage_stats.avg_response_time_ms,
                success_rate: if agent_entry.metadata.usage_stats.successful_tasks > 0 {
                    agent_entry.metadata.usage_stats.successful_tasks as f32 / 
                    (agent_entry.metadata.usage_stats.successful_tasks + agent_entry.metadata.usage_stats.failed_tasks) as f32
                } else {
                    1.0
                },
                quality_score: 1.0,
                last_active: std::time::Instant::now(),
            },
            workload: 0.0,
            max_concurrent_tasks: config.performance_settings.batch_size.max(1),
            active_tasks: Vec::new(),
        };
        
        let agent = Arc::new(RwLock::new(coord_agent));
        
        // Update registry to mark agent as active
        if let Some(registry) = &self.registry {
            registry.read().await.update_runtime_state(&agent_id, super::registry::RuntimeState {
                is_active: true,
                status: super::registry::AgentStatus::Ready,
                resources: super::registry::ResourceAllocation {
                    memory_mb: config.performance_settings.batch_size as u32 * 50,
                    cpu_percent: 10,
                    token_budget: config.model_preferences.context_preferences.preferred_context_size as u32,
                    rate_limit: 60,
                },
                session_id: Some(Uuid::new_v4().to_string()),
            }).await?;
            
            // Update activation stats
            registry.read().await.update_usage_stats(&agent_id, |stats| {
                stats.activation_count += 1;
                stats.last_activated = Some(Utc::now());
            }).await?;
        }
        
        tracing::info!("Agent {} created and activated successfully", agent_id);
        Ok(agent)
    }
    
    /// Retrieve an agent configuration from the registry
    pub async fn get_agent_config(&self, agent_id: &str) -> Result<Option<AgentConfig>> {
        if let Some(registry) = &self.registry {
            if let Some(entry) = registry.read().await.get(agent_id).await? {
                return Ok(Some(entry.config));
            }
        }
        Ok(None)
    }
    
    /// List all registered agents
    pub async fn list_agents(&self) -> Result<Vec<(String, AgentConfig)>> {
        if let Some(registry) = &self.registry {
            let agents = registry.read().await.list().await?;
            Ok(agents.into_iter()
                .map(|(id, entry)| (id, entry.config))
                .collect())
        } else {
            Ok(Vec::new())
        }
    }
}

impl AgentDraft {
    fn new() -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            name: String::new(),
            description: String::new(),
            template_id: None,
            specialization: None,
            personality: PersonalityProfile::default(),
            skills: Vec::new(),
            preferred_models: Vec::new(),
            allowed_tools: Vec::new(),
            permissions: AgentPermissions {
                can_execute_code: false,
                can_access_files: true,
                can_use_network: true,
                can_modify_system: false,
                can_access_sensitive_data: false,
                max_resource_usage: ResourceLimits {
                    max_memory_mb: 512,
                    max_cpu_percent: 25,
                    max_tokens_per_request: 4096,
                    max_requests_per_minute: 60,
                    max_execution_time_seconds: 300,
                },
            },
            metadata: AgentMetadata {
                created_at: Utc::now(),
                created_by: "system".to_string(),
                last_modified: Utc::now(),
                version: "1.0.0".to_string(),
                tags: Vec::new(),
                documentation: String::new(),
                examples: Vec::new(),
            },
        }
    }
}

impl AgentConfig {
    /// Create a researcher template
    fn researcher_template() -> Self {
        let mut personality = PersonalityProfile::default();
        personality.traits.insert(PersonalityTrait::Analytical, 0.9);
        personality.traits.insert(PersonalityTrait::Precision, 0.9);
        personality.traits.insert(PersonalityTrait::Curiosity, 0.8);
        personality.communication_style = CommunicationStyle::Professional;
        personality.decision_style = DecisionStyle::DataDriven;
        
        Self {
            id: Uuid::new_v4().to_string(),
            name: "Research Assistant".to_string(),
            description: "Specialized in research and analysis".to_string(),
            specialization: AgentSpecialization::Analytical,
            personality,
            skills: vec![
                AgentSkill {
                    id: "research".to_string(),
                    name: "Research".to_string(),
                    category: SkillCategory::Research,
                    proficiency: 0.9,
                    experience_hours: 1000,
                    certifications: Vec::new(),
                    examples: Vec::new(),
                },
            ],
            model_preferences: ModelPreferences {
                primary_model: Some("gpt-4".to_string()),
                fallback_models: vec!["claude-3-opus".to_string()],
                model_selection_strategy: ModelSelectionStrategy::PerformanceOptimized,
                context_preferences: ContextPreferences {
                    preferred_context_size: 8192,
                    include_examples: true,
                    include_history: true,
                    compression_enabled: false,
                },
                cost_threshold: None,
            },
            tool_permissions: ToolPermissions {
                allowed_tools: vec!["web_search".to_string(), "arxiv".to_string()],
                blocked_tools: Vec::new(),
                requires_approval: Vec::new(),
                auto_execute: vec!["web_search".to_string()],
                rate_limits: HashMap::new(),
            },
            collaboration_settings: CollaborationSettings {
                can_collaborate: true,
                preferred_team_size: 2,
                leadership_style: LeadershipStyle::Peer,
                collaboration_rules: Vec::new(),
            },
            performance_settings: PerformanceSettings {
                optimization_level: OptimizationLevel::Quality,
                caching_enabled: true,
                parallel_processing: true,
                batch_size: 5,
                timeout_seconds: 60,
            },
            metadata: AgentMetadata {
                created_at: Utc::now(),
                created_by: "template".to_string(),
                last_modified: Utc::now(),
                version: "1.0.0".to_string(),
                tags: vec!["research".to_string(), "analysis".to_string()],
                documentation: "Research assistant agent template".to_string(),
                examples: Vec::new(),
            },
        }
    }
    
    /// Create a developer template
    fn developer_template() -> Self {
        let mut personality = PersonalityProfile::default();
        personality.traits.insert(PersonalityTrait::Analytical, 0.8);
        personality.traits.insert(PersonalityTrait::Precision, 0.95);
        personality.traits.insert(PersonalityTrait::Creativity, 0.7);
        personality.communication_style = CommunicationStyle::Technical;
        personality.decision_style = DecisionStyle::DataDriven;
        personality.work_style = WorkStyle::Methodical;
        
        Self {
            id: Uuid::new_v4().to_string(),
            name: "Code Developer".to_string(),
            description: "Expert in software development".to_string(),
            specialization: AgentSpecialization::Technical,
            personality,
            skills: vec![
                AgentSkill {
                    id: "programming".to_string(),
                    name: "Programming".to_string(),
                    category: SkillCategory::Programming,
                    proficiency: 0.95,
                    experience_hours: 5000,
                    certifications: Vec::new(),
                    examples: Vec::new(),
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
                allowed_tools: vec!["code_analysis".to_string(), "github".to_string(), "file_system".to_string()],
                blocked_tools: Vec::new(),
                requires_approval: vec!["file_system".to_string()],
                auto_execute: vec!["code_analysis".to_string()],
                rate_limits: HashMap::new(),
            },
            collaboration_settings: CollaborationSettings {
                can_collaborate: true,
                preferred_team_size: 4,
                leadership_style: LeadershipStyle::Adaptive,
                collaboration_rules: Vec::new(),
            },
            performance_settings: PerformanceSettings {
                optimization_level: OptimizationLevel::Balanced,
                caching_enabled: true,
                parallel_processing: true,
                batch_size: 10,
                timeout_seconds: 120,
            },
            metadata: AgentMetadata {
                created_at: Utc::now(),
                created_by: "template".to_string(),
                last_modified: Utc::now(),
                version: "1.0.0".to_string(),
                tags: vec!["development".to_string(), "coding".to_string()],
                documentation: "Software developer agent template".to_string(),
                examples: Vec::new(),
            },
        }
    }
    
    /// Create a creative template
    fn creative_template() -> Self {
        let mut personality = PersonalityProfile::default();
        personality.traits.insert(PersonalityTrait::Creativity, 0.95);
        personality.traits.insert(PersonalityTrait::Enthusiasm, 0.8);
        personality.traits.insert(PersonalityTrait::Adaptability, 0.85);
        personality.communication_style = CommunicationStyle::Creative;
        personality.decision_style = DecisionStyle::Intuitive;
        personality.work_style = WorkStyle::Creative;
        
        Self {
            id: Uuid::new_v4().to_string(),
            name: "Creative Writer".to_string(),
            description: "Specialized in creative content".to_string(),
            specialization: AgentSpecialization::Creative,
            personality,
            skills: vec![
                AgentSkill {
                    id: "writing".to_string(),
                    name: "Creative Writing".to_string(),
                    category: SkillCategory::Writing,
                    proficiency: 0.9,
                    experience_hours: 2000,
                    certifications: Vec::new(),
                    examples: Vec::new(),
                },
            ],
            model_preferences: ModelPreferences {
                primary_model: Some("claude-3-opus".to_string()),
                fallback_models: vec!["gpt-4".to_string()],
                model_selection_strategy: ModelSelectionStrategy::PerformanceOptimized,
                context_preferences: ContextPreferences {
                    preferred_context_size: 8192,
                    include_examples: true,
                    include_history: false,
                    compression_enabled: false,
                },
                cost_threshold: None,
            },
            tool_permissions: ToolPermissions {
                allowed_tools: vec!["creative_media".to_string()],
                blocked_tools: Vec::new(),
                requires_approval: Vec::new(),
                auto_execute: Vec::new(),
                rate_limits: HashMap::new(),
            },
            collaboration_settings: CollaborationSettings {
                can_collaborate: true,
                preferred_team_size: 2,
                leadership_style: LeadershipStyle::Independent,
                collaboration_rules: Vec::new(),
            },
            performance_settings: PerformanceSettings {
                optimization_level: OptimizationLevel::Quality,
                caching_enabled: false,
                parallel_processing: false,
                batch_size: 1,
                timeout_seconds: 180,
            },
            metadata: AgentMetadata {
                created_at: Utc::now(),
                created_by: "template".to_string(),
                last_modified: Utc::now(),
                version: "1.0.0".to_string(),
                tags: vec!["creative".to_string(), "writing".to_string()],
                documentation: "Creative writer agent template".to_string(),
                examples: Vec::new(),
            },
        }
    }
}