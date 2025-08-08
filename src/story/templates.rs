//! Story templates for common workflows

use super::types::*;
use super::engine::StoryEngine;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use tracing::{debug, info};
use uuid::Uuid;

/// Story template for common development workflows
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoryTemplate {
    pub id: TemplateId,
    pub name: String,
    pub description: String,
    pub category: TemplateCategory,
    pub initial_context: HashMap<String, String>,
    pub plot_structure: Vec<PlotTemplate>,
    pub required_capabilities: Vec<String>,
    pub metadata: TemplateMetadata,
}

/// Template identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TemplateId(pub Uuid);

impl fmt::Display for TemplateId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Template category
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TemplateCategory {
    Development,
    Debugging,
    Refactoring,
    Documentation,
    Testing,
    Deployment,
    Maintenance,
    Research,
    Custom(String),
}

/// Plot point template
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlotTemplate {
    pub plot_type: PlotType,
    pub description_template: String,
    pub dependencies: Vec<usize>, // Indices of dependent plot points
    pub estimated_duration: Option<chrono::Duration>,
    pub auto_complete_conditions: Vec<CompletionCondition>,
}

/// Completion condition for automatic plot progression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompletionCondition {
    FileExists(String),
    FileContains { path: String, pattern: String },
    TestsPassing,
    BuildSuccessful,
    ContextContains(String),
    TimeElapsed(chrono::Duration),
    ManualConfirmation,
}

/// Template metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateMetadata {
    pub author: String,
    pub version: String,
    pub tags: Vec<String>,
    pub usage_count: usize,
    pub success_rate: f32,
    pub average_duration: Option<chrono::Duration>,
}

/// Story template manager
#[derive(Debug)]
pub struct StoryTemplateManager {
    templates: HashMap<TemplateId, StoryTemplate>,
    story_engine: Option<Arc<StoryEngine>>,
}

impl StoryTemplateManager {
    /// Create a new template manager
    pub fn new(story_engine: Arc<StoryEngine>) -> Self {
        let mut manager = Self {
            templates: HashMap::new(),
            story_engine: Some(story_engine),
        };
        
        // Initialize built-in templates
        manager.init_builtin_templates();
        
        manager
    }
    
    /// Create an empty template manager (for initialization)
    pub fn new_empty() -> Self {
        let mut manager = Self {
            templates: HashMap::new(),
            story_engine: None,
        };
        
        // Initialize built-in templates
        manager.init_builtin_templates();
        
        manager
    }
    
    /// Set the story engine reference
    pub fn set_story_engine(&mut self, story_engine: Arc<StoryEngine>) {
        self.story_engine = Some(story_engine);
    }
    
    /// Initialize built-in templates
    fn init_builtin_templates(&mut self) {
        // REST API Development Template
        self.register_template(StoryTemplate {
            id: TemplateId(Uuid::new_v4()),
            name: "REST API Development".to_string(),
            description: "Complete workflow for developing a REST API with CRUD operations".to_string(),
            category: TemplateCategory::Development,
            initial_context: HashMap::from([
                ("api_type".to_string(), "RESTful".to_string()),
                ("operations".to_string(), "CRUD".to_string()),
            ]),
            plot_structure: vec![
                PlotTemplate {
                    plot_type: PlotType::Goal { objective: "Design REST API specification".to_string() },
                    description_template: "Design {api_name} API with {operations} operations".to_string(),
                    dependencies: vec![],
                    estimated_duration: Some(chrono::Duration::hours(1)),
                    auto_complete_conditions: vec![
                        CompletionCondition::FileExists("api/openapi.yaml".to_string()),
                    ],
                },
                PlotTemplate {
                    plot_type: PlotType::Task { 
                        description: "Create database models".to_string(),
                        completed: false,
                    },
                    description_template: "Create database models for {entity_name}".to_string(),
                    dependencies: vec![0],
                    estimated_duration: Some(chrono::Duration::hours(2)),
                    auto_complete_conditions: vec![
                        CompletionCondition::FileExists("src/models/mod.rs".to_string()),
                    ],
                },
                PlotTemplate {
                    plot_type: PlotType::Task {
                        description: "Implement API endpoints".to_string(),
                        completed: false,
                    },
                    description_template: "Implement {operations} endpoints for {api_name}".to_string(),
                    dependencies: vec![1],
                    estimated_duration: Some(chrono::Duration::hours(4)),
                    auto_complete_conditions: vec![
                        CompletionCondition::FileExists("src/api/routes.rs".to_string()),
                        CompletionCondition::BuildSuccessful,
                    ],
                },
                PlotTemplate {
                    plot_type: PlotType::Task {
                        description: "Write API tests".to_string(),
                        completed: false,
                    },
                    description_template: "Write integration tests for {api_name} endpoints".to_string(),
                    dependencies: vec![2],
                    estimated_duration: Some(chrono::Duration::hours(2)),
                    auto_complete_conditions: vec![
                        CompletionCondition::TestsPassing,
                    ],
                },
                PlotTemplate {
                    plot_type: PlotType::Task {
                        description: "Document API".to_string(),
                        completed: false,
                    },
                    description_template: "Create API documentation for {api_name}".to_string(),
                    dependencies: vec![3],
                    estimated_duration: Some(chrono::Duration::hours(1)),
                    auto_complete_conditions: vec![
                        CompletionCondition::FileExists("docs/api.md".to_string()),
                    ],
                },
            ],
            required_capabilities: vec![
                "api_design".to_string(),
                "database_modeling".to_string(),
                "testing".to_string(),
            ],
            metadata: TemplateMetadata {
                author: "Loki".to_string(),
                version: "1.0.0".to_string(),
                tags: vec!["api".to_string(), "rest".to_string(), "backend".to_string()],
                usage_count: 0,
                success_rate: 0.0,
                average_duration: None,
            },
        });
        
        // Bug Investigation Template
        self.register_template(StoryTemplate {
            id: TemplateId(Uuid::new_v4()),
            name: "Bug Investigation".to_string(),
            description: "Systematic approach to investigating and fixing bugs".to_string(),
            category: TemplateCategory::Debugging,
            initial_context: HashMap::from([
                ("investigation_type".to_string(), "bug".to_string()),
            ]),
            plot_structure: vec![
                PlotTemplate {
                    plot_type: PlotType::Issue {
                        error: "Bug reported".to_string(),
                        resolved: false,
                    },
                    description_template: "Investigate {bug_description}".to_string(),
                    dependencies: vec![],
                    estimated_duration: Some(chrono::Duration::minutes(30)),
                    auto_complete_conditions: vec![
                        CompletionCondition::ManualConfirmation,
                    ],
                },
                PlotTemplate {
                    plot_type: PlotType::Discovery {
                        insight: "Root cause identified".to_string(),
                    },
                    description_template: "Identify root cause of {bug_description}".to_string(),
                    dependencies: vec![0],
                    estimated_duration: Some(chrono::Duration::hours(1)),
                    auto_complete_conditions: vec![
                        CompletionCondition::ContextContains("root_cause".to_string()),
                    ],
                },
                PlotTemplate {
                    plot_type: PlotType::Task {
                        description: "Implement fix".to_string(),
                        completed: false,
                    },
                    description_template: "Fix {bug_description} in {affected_component}".to_string(),
                    dependencies: vec![1],
                    estimated_duration: Some(chrono::Duration::hours(2)),
                    auto_complete_conditions: vec![
                        CompletionCondition::BuildSuccessful,
                    ],
                },
                PlotTemplate {
                    plot_type: PlotType::Task {
                        description: "Write regression test".to_string(),
                        completed: false,
                    },
                    description_template: "Add regression test for {bug_description}".to_string(),
                    dependencies: vec![2],
                    estimated_duration: Some(chrono::Duration::hours(1)),
                    auto_complete_conditions: vec![
                        CompletionCondition::TestsPassing,
                    ],
                },
                PlotTemplate {
                    plot_type: PlotType::Issue {
                        error: "Bug verified fixed".to_string(),
                        resolved: true,
                    },
                    description_template: "Verify {bug_description} is resolved".to_string(),
                    dependencies: vec![3],
                    estimated_duration: Some(chrono::Duration::minutes(30)),
                    auto_complete_conditions: vec![
                        CompletionCondition::ManualConfirmation,
                    ],
                },
            ],
            required_capabilities: vec![
                "debugging".to_string(),
                "testing".to_string(),
                "problem_solving".to_string(),
            ],
            metadata: TemplateMetadata {
                author: "Loki".to_string(),
                version: "1.0.0".to_string(),
                tags: vec!["bug".to_string(), "debug".to_string(), "fix".to_string()],
                usage_count: 0,
                success_rate: 0.0,
                average_duration: None,
            },
        });
        
        // Refactoring Template
        self.register_template(StoryTemplate {
            id: TemplateId(Uuid::new_v4()),
            name: "Code Refactoring".to_string(),
            description: "Structured approach to refactoring code for better maintainability".to_string(),
            category: TemplateCategory::Refactoring,
            initial_context: HashMap::from([
                ("refactor_type".to_string(), "improvement".to_string()),
            ]),
            plot_structure: vec![
                PlotTemplate {
                    plot_type: PlotType::Goal {
                        objective: "Analyze code for refactoring opportunities".to_string(),
                    },
                    description_template: "Analyze {target_module} for refactoring".to_string(),
                    dependencies: vec![],
                    estimated_duration: Some(chrono::Duration::hours(1)),
                    auto_complete_conditions: vec![
                        CompletionCondition::ContextContains("refactoring_plan".to_string()),
                    ],
                },
                PlotTemplate {
                    plot_type: PlotType::Task {
                        description: "Create refactoring plan".to_string(),
                        completed: false,
                    },
                    description_template: "Plan refactoring for {target_module}".to_string(),
                    dependencies: vec![0],
                    estimated_duration: Some(chrono::Duration::minutes(30)),
                    auto_complete_conditions: vec![
                        CompletionCondition::ManualConfirmation,
                    ],
                },
                PlotTemplate {
                    plot_type: PlotType::Transformation {
                        before: "Original code structure".to_string(),
                        after: "Refactored code structure".to_string(),
                    },
                    description_template: "Refactor {target_module} following {pattern}".to_string(),
                    dependencies: vec![1],
                    estimated_duration: Some(chrono::Duration::hours(3)),
                    auto_complete_conditions: vec![
                        CompletionCondition::BuildSuccessful,
                    ],
                },
                PlotTemplate {
                    plot_type: PlotType::Task {
                        description: "Update tests".to_string(),
                        completed: false,
                    },
                    description_template: "Update tests for refactored {target_module}".to_string(),
                    dependencies: vec![2],
                    estimated_duration: Some(chrono::Duration::hours(1)),
                    auto_complete_conditions: vec![
                        CompletionCondition::TestsPassing,
                    ],
                },
                PlotTemplate {
                    plot_type: PlotType::Discovery {
                        insight: "Performance metrics".to_string(),
                    },
                    description_template: "Measure performance impact of refactoring".to_string(),
                    dependencies: vec![3],
                    estimated_duration: Some(chrono::Duration::minutes(30)),
                    auto_complete_conditions: vec![
                        CompletionCondition::ContextContains("performance_metrics".to_string()),
                    ],
                },
            ],
            required_capabilities: vec![
                "code_analysis".to_string(),
                "refactoring".to_string(),
                "testing".to_string(),
            ],
            metadata: TemplateMetadata {
                author: "Loki".to_string(),
                version: "1.0.0".to_string(),
                tags: vec!["refactor".to_string(), "clean_code".to_string(), "maintenance".to_string()],
                usage_count: 0,
                success_rate: 0.0,
                average_duration: None,
            },
        });
        
        // Feature Development Template
        self.register_template(StoryTemplate {
            id: TemplateId(Uuid::new_v4()),
            name: "Feature Development".to_string(),
            description: "End-to-end feature development workflow".to_string(),
            category: TemplateCategory::Development,
            initial_context: HashMap::from([
                ("development_type".to_string(), "feature".to_string()),
            ]),
            plot_structure: vec![
                PlotTemplate {
                    plot_type: PlotType::Goal {
                        objective: "Design feature specification".to_string(),
                    },
                    description_template: "Design {feature_name} feature".to_string(),
                    dependencies: vec![],
                    estimated_duration: Some(chrono::Duration::hours(2)),
                    auto_complete_conditions: vec![
                        CompletionCondition::FileExists("docs/features/{feature_name}.md".to_string()),
                    ],
                },
                PlotTemplate {
                    plot_type: PlotType::Decision {
                        question: "Technical approach".to_string(),
                        choice: "".to_string(),
                    },
                    description_template: "Choose implementation approach for {feature_name}".to_string(),
                    dependencies: vec![0],
                    estimated_duration: Some(chrono::Duration::minutes(30)),
                    auto_complete_conditions: vec![
                        CompletionCondition::ContextContains("implementation_approach".to_string()),
                    ],
                },
                PlotTemplate {
                    plot_type: PlotType::Task {
                        description: "Implement core functionality".to_string(),
                        completed: false,
                    },
                    description_template: "Implement {feature_name} core logic".to_string(),
                    dependencies: vec![1],
                    estimated_duration: Some(chrono::Duration::hours(4)),
                    auto_complete_conditions: vec![
                        CompletionCondition::BuildSuccessful,
                    ],
                },
                PlotTemplate {
                    plot_type: PlotType::Task {
                        description: "Add UI components".to_string(),
                        completed: false,
                    },
                    description_template: "Create UI for {feature_name}".to_string(),
                    dependencies: vec![2],
                    estimated_duration: Some(chrono::Duration::hours(2)),
                    auto_complete_conditions: vec![
                        CompletionCondition::FileExists("src/ui/{feature_name}.rs".to_string()),
                    ],
                },
                PlotTemplate {
                    plot_type: PlotType::Task {
                        description: "Write feature tests".to_string(),
                        completed: false,
                    },
                    description_template: "Write comprehensive tests for {feature_name}".to_string(),
                    dependencies: vec![3],
                    estimated_duration: Some(chrono::Duration::hours(2)),
                    auto_complete_conditions: vec![
                        CompletionCondition::TestsPassing,
                    ],
                },
                PlotTemplate {
                    plot_type: PlotType::Interaction {
                        with: "User".to_string(),
                        action: "Feature review".to_string(),
                    },
                    description_template: "Review {feature_name} with stakeholders".to_string(),
                    dependencies: vec![4],
                    estimated_duration: Some(chrono::Duration::hours(1)),
                    auto_complete_conditions: vec![
                        CompletionCondition::ManualConfirmation,
                    ],
                },
            ],
            required_capabilities: vec![
                "feature_design".to_string(),
                "implementation".to_string(),
                "ui_development".to_string(),
                "testing".to_string(),
            ],
            metadata: TemplateMetadata {
                author: "Loki".to_string(),
                version: "1.0.0".to_string(),
                tags: vec!["feature".to_string(), "development".to_string(), "full_stack".to_string()],
                usage_count: 0,
                success_rate: 0.0,
                average_duration: None,
            },
        });
        
        // Performance Optimization Template
        self.register_template(StoryTemplate {
            id: TemplateId(Uuid::new_v4()),
            name: "Performance Optimization".to_string(),
            description: "Systematic performance analysis and optimization".to_string(),
            category: TemplateCategory::Maintenance,
            initial_context: HashMap::from([
                ("optimization_type".to_string(), "performance".to_string()),
            ]),
            plot_structure: vec![
                PlotTemplate {
                    plot_type: PlotType::Discovery {
                        insight: "Performance baseline".to_string(),
                    },
                    description_template: "Profile {target_component} performance".to_string(),
                    dependencies: vec![],
                    estimated_duration: Some(chrono::Duration::hours(1)),
                    auto_complete_conditions: vec![
                        CompletionCondition::ContextContains("baseline_metrics".to_string()),
                    ],
                },
                PlotTemplate {
                    plot_type: PlotType::Task {
                        description: "Identify bottlenecks".to_string(),
                        completed: false,
                    },
                    description_template: "Analyze performance bottlenecks in {target_component}".to_string(),
                    dependencies: vec![0],
                    estimated_duration: Some(chrono::Duration::hours(2)),
                    auto_complete_conditions: vec![
                        CompletionCondition::ContextContains("bottlenecks_identified".to_string()),
                    ],
                },
                PlotTemplate {
                    plot_type: PlotType::Decision {
                        question: "Optimization strategy".to_string(),
                        choice: "".to_string(),
                    },
                    description_template: "Choose optimization approach for {bottleneck}".to_string(),
                    dependencies: vec![1],
                    estimated_duration: Some(chrono::Duration::minutes(30)),
                    auto_complete_conditions: vec![
                        CompletionCondition::ManualConfirmation,
                    ],
                },
                PlotTemplate {
                    plot_type: PlotType::Transformation {
                        before: "Unoptimized code".to_string(),
                        after: "Optimized code".to_string(),
                    },
                    description_template: "Optimize {target_component} using {strategy}".to_string(),
                    dependencies: vec![2],
                    estimated_duration: Some(chrono::Duration::hours(3)),
                    auto_complete_conditions: vec![
                        CompletionCondition::BuildSuccessful,
                        CompletionCondition::TestsPassing,
                    ],
                },
                PlotTemplate {
                    plot_type: PlotType::Discovery {
                        insight: "Performance improvement".to_string(),
                    },
                    description_template: "Measure performance improvement".to_string(),
                    dependencies: vec![3],
                    estimated_duration: Some(chrono::Duration::hours(1)),
                    auto_complete_conditions: vec![
                        CompletionCondition::ContextContains("improvement_metrics".to_string()),
                    ],
                },
            ],
            required_capabilities: vec![
                "performance_analysis".to_string(),
                "optimization".to_string(),
                "profiling".to_string(),
            ],
            metadata: TemplateMetadata {
                author: "Loki".to_string(),
                version: "1.0.0".to_string(),
                tags: vec!["performance".to_string(), "optimization".to_string(), "profiling".to_string()],
                usage_count: 0,
                success_rate: 0.0,
                average_duration: None,
            },
        });
    }
    
    /// Register a template
    pub fn register_template(&mut self, template: StoryTemplate) {
        info!("Registering template: {} ({})", template.name, template.id.0);
        self.templates.insert(template.id, template);
    }
    
    /// Get available templates
    pub fn get_templates(&self) -> Vec<&StoryTemplate> {
        self.templates.values().collect()
    }
    
    /// Get templates by category
    pub fn get_templates_by_category(&self, category: &TemplateCategory) -> Vec<&StoryTemplate> {
        self.templates
            .values()
            .filter(|t| &t.category == category)
            .collect()
    }
    
    /// Find templates by tags
    pub fn find_templates_by_tags(&self, tags: &[String]) -> Vec<&StoryTemplate> {
        self.templates
            .values()
            .filter(|t| tags.iter().any(|tag| t.metadata.tags.contains(tag)))
            .collect()
    }
    
    /// Instantiate a template into a story
    pub async fn instantiate_template(
        &mut self,
        template_id: TemplateId,
        story_type: StoryType,
        context: HashMap<String, String>,
    ) -> Result<StoryId> {
        let template = self.templates
            .get(&template_id)
            .ok_or_else(|| anyhow::anyhow!("Template not found"))?
            .clone();
        
        // Create story with template context
        let mut story_context = template.initial_context.clone();
        story_context.extend(context);
        
        let story_engine = self.story_engine
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Story engine not initialized"))?;
            
        // Create story based on type
        let story_id = match &story_type {
            StoryType::Agent { agent_id, .. } => {
                story_engine.create_agent_story(
                    agent_id.clone(),
                    format!("{} - {}", template.name, Uuid::new_v4())
                ).await?
            }
            StoryType::Codebase { root_path, .. } => {
                story_engine.create_codebase_story(
                    root_path.clone(),
                    "rust".to_string() // Default to Rust, could be improved to detect language
                ).await?
            }
            _ => {
                // For other types, create as agent story with generated ID
                story_engine.create_agent_story(
                    Uuid::new_v4().to_string(),
                    format!("{} - {}", template.name, Uuid::new_v4())
                ).await?
            }
        };
        
        // Add template plot points
        for (i, plot_template) in template.plot_structure.iter().enumerate() {
            // Resolve dependencies
            let dependencies = plot_template.dependencies
                .iter()
                .map(|&dep_idx| {
                    if dep_idx < i {
                        format!("plot_{}", dep_idx)
                    } else {
                        String::new()
                    }
                })
                .filter(|s| !s.is_empty())
                .collect();
            
            // Interpolate description
            let description = self.interpolate_template(
                &plot_template.description_template,
                &story_context,
            );
            
            // Create plot type with interpolated values
            let plot_type = match &plot_template.plot_type {
                PlotType::Goal { .. } => PlotType::Goal { objective: description },
                PlotType::Task { completed, .. } => PlotType::Task { 
                    description, 
                    completed: *completed 
                },
                PlotType::Decision { question, .. } => PlotType::Decision {
                    question: question.clone(),
                    choice: String::new(),
                },
                PlotType::Discovery { .. } => PlotType::Discovery { insight: description },
                PlotType::Issue { resolved, .. } => PlotType::Issue {
                    error: description,
                    resolved: *resolved,
                },
                PlotType::Transformation { before, after } => PlotType::Transformation {
                    before: before.clone(),
                    after: after.clone(),
                },
                PlotType::Interaction { with, .. } => PlotType::Interaction {
                    with: with.clone(),
                    action: description,
                },
                PlotType::Analysis { findings, .. } => PlotType::Analysis {
                    subject: description,
                    findings: findings.clone(),
                },
                PlotType::Progress { percentage, .. } => PlotType::Progress {
                    milestone: description,
                    percentage: *percentage,
                },
                PlotType::Action { action_type, parameters, .. } => PlotType::Action {
                    action_type: action_type.clone(),
                    parameters: parameters.clone(),
                    outcome: String::new(),
                },
            };
            
            story_engine.add_plot_point(
                story_id,
                plot_type,
                dependencies,
            ).await?;
        }
        
        // Update template usage
        if let Some(template) = self.templates.get_mut(&template_id) {
            template.metadata.usage_count += 1;
        }
        
        info!("Instantiated template {} as story {}", template.name, story_id.0);
        
        Ok(story_id)
    }
    
    /// Get template recommendations based on context
    pub fn recommend_templates(&self, context: &str) -> Vec<(&StoryTemplate, f32)> {
        let mut recommendations = Vec::new();
        let context_lower = context.to_lowercase();
        
        for template in self.templates.values() {
            let mut score = 0.0;
            
            // Check name match
            if context_lower.contains(&template.name.to_lowercase()) {
                score += 0.5;
            }
            
            // Check description match
            let desc_words: Vec<&str> = template.description.split_whitespace().collect();
            for word in desc_words {
                if context_lower.contains(&word.to_lowercase()) {
                    score += 0.1;
                }
            }
            
            // Check tag match
            for tag in &template.metadata.tags {
                if context_lower.contains(&tag.to_lowercase()) {
                    score += 0.2;
                }
            }
            
            // Boost by success rate
            score += template.metadata.success_rate * 0.2;
            
            if score > 0.0 {
                recommendations.push((template, score));
            }
        }
        
        // Sort by score
        recommendations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        recommendations
    }
    
    /// Update template metrics after completion
    pub async fn update_template_metrics(
        &mut self,
        template_id: TemplateId,
        success: bool,
        duration: chrono::Duration,
    ) -> Result<()> {
        if let Some(template) = self.templates.get_mut(&template_id) {
            let usage_count = template.metadata.usage_count as f32;
            
            // Update success rate
            let current_successes = template.metadata.success_rate * usage_count;
            let new_success_rate = if success {
                (current_successes + 1.0) / (usage_count + 1.0)
            } else {
                current_successes / (usage_count + 1.0)
            };
            template.metadata.success_rate = new_success_rate;
            
            // Update average duration
            if let Some(avg_duration) = template.metadata.average_duration {
                let total_duration = avg_duration * usage_count as i32 + duration;
                template.metadata.average_duration = Some(total_duration / (usage_count as i32 + 1));
            } else {
                template.metadata.average_duration = Some(duration);
            }
            
            debug!("Updated metrics for template {}: success_rate={:.2}, avg_duration={:?}",
                template.name, new_success_rate, template.metadata.average_duration);
        }
        
        Ok(())
    }
    
    // Helper methods
    
    fn interpolate_template(&self, template: &str, context: &HashMap<String, String>) -> String {
        let mut result = template.to_string();
        
        for (key, value) in context {
            result = result.replace(&format!("{{{}}}", key), value);
        }
        
        result
    }
}

/// Template instance tracker
#[derive(Debug)]
pub struct TemplateInstanceTracker {
    instances: HashMap<StoryId, TemplateInstance>,
}

/// Template instance
#[derive(Debug, Clone)]
pub struct TemplateInstance {
    pub story_id: StoryId,
    pub template_id: TemplateId,
    pub started_at: chrono::DateTime<chrono::Utc>,
    pub completed_at: Option<chrono::DateTime<chrono::Utc>>,
    pub current_plot_index: usize,
    pub context: HashMap<String, String>,
}

impl TemplateInstanceTracker {
    /// Create a new instance tracker
    pub fn new() -> Self {
        Self {
            instances: HashMap::new(),
        }
    }
    
    /// Track a new instance
    pub fn track_instance(&mut self, instance: TemplateInstance) {
        self.instances.insert(instance.story_id, instance);
    }
    
    /// Update instance progress
    pub fn update_progress(&mut self, story_id: StoryId, plot_index: usize) {
        if let Some(instance) = self.instances.get_mut(&story_id) {
            instance.current_plot_index = plot_index;
        }
    }
    
    /// Complete an instance
    pub fn complete_instance(&mut self, story_id: StoryId) -> Option<chrono::Duration> {
        if let Some(instance) = self.instances.get_mut(&story_id) {
            instance.completed_at = Some(chrono::Utc::now());
            instance.completed_at.map(|end| end - instance.started_at)
        } else {
            None
        }
    }
    
    /// Get active instances
    pub fn get_active_instances(&self) -> Vec<&TemplateInstance> {
        self.instances
            .values()
            .filter(|i| i.completed_at.is_none())
            .collect()
    }
}