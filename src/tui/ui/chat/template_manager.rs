//! Chat template and snippet management
//! 
//! Provides support for creating, managing, and using chat templates
//! and code snippets for quick insertion and reuse.

use std::collections::HashMap;
use std::path::{ PathBuf};
use anyhow::{Result, anyhow};
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};

/// Template category
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TemplateCategory {
    Greeting,
    Question,
    CodeRequest,
    BugReport,
    FeatureRequest,
    Documentation,
    Analysis,
    Workflow,
    Custom(String),
}

/// Template visibility
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TemplateVisibility {
    Private,
    Shared,
    System,
}

/// Chat template
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatTemplate {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub category: TemplateCategory,
    pub content: String,
    pub variables: Vec<TemplateVariable>,
    pub tags: Vec<String>,
    pub visibility: TemplateVisibility,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub usage_count: usize,
    pub shortcuts: Vec<String>,
}

/// Template variable definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateVariable {
    pub name: String,
    pub description: Option<String>,
    pub default_value: Option<String>,
    pub required: bool,
    pub var_type: VariableType,
    pub options: Option<Vec<String>>,
}

/// Variable types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum VariableType {
    Text,
    Number,
    Boolean,
    Date,
    Time,
    Choice,
    FilePath,
    Code,
}

/// Code snippet
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeSnippet {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub language: String,
    pub code: String,
    pub variables: Vec<TemplateVariable>,
    pub tags: Vec<String>,
    pub created_at: DateTime<Utc>,
    pub usage_count: usize,
    pub shortcuts: Vec<String>,
}

/// Template manager
#[derive(Clone)]
pub struct TemplateManager {
    /// Chat templates
    templates: HashMap<String, ChatTemplate>,
    
    /// Code snippets
    snippets: HashMap<String, CodeSnippet>,
    
    /// Template shortcuts (shortcut -> template_id)
    shortcuts: HashMap<String, String>,
    
    /// Storage path
    storage_path: Option<PathBuf>,
    
    /// Built-in templates
    builtin_templates: HashMap<String, ChatTemplate>,
}

impl TemplateManager {
    pub fn new() -> Self {
        let mut manager = Self {
            templates: HashMap::new(),
            snippets: HashMap::new(),
            shortcuts: HashMap::new(),
            storage_path: None,
            builtin_templates: HashMap::new(),
        };
        
        manager.load_builtin_templates();
        manager
    }
    
    /// Set storage path for persistent storage
    pub fn set_storage_path(&mut self, path: PathBuf) {
        self.storage_path = Some(path);
    }
    
    /// Load default templates (public method for initialization)
    pub fn load_default_templates(&mut self) {
        self.load_builtin_templates();
    }
    
    /// Load built-in templates
    fn load_builtin_templates(&mut self) {
        // Greeting templates
        self.add_builtin_template(ChatTemplate {
            id: "greeting_general".to_string(),
            name: "General Greeting".to_string(),
            description: Some("A friendly greeting to start a conversation".to_string()),
            category: TemplateCategory::Greeting,
            content: "Hello! I'm working on {{project_name}} and I need help with {{topic}}. Could you assist me?".to_string(),
            variables: vec![
                TemplateVariable {
                    name: "project_name".to_string(),
                    description: Some("Name of your project".to_string()),
                    default_value: Some("my project".to_string()),
                    required: false,
                    var_type: VariableType::Text,
                    options: None,
                },
                TemplateVariable {
                    name: "topic".to_string(),
                    description: Some("What you need help with".to_string()),
                    default_value: None,
                    required: true,
                    var_type: VariableType::Text,
                    options: None,
                },
            ],
            tags: vec!["greeting".to_string(), "start".to_string()],
            visibility: TemplateVisibility::System,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            usage_count: 0,
            shortcuts: vec!["hi".to_string(), "hello".to_string()],
        });
        
        // Code request templates
        self.add_builtin_template(ChatTemplate {
            id: "code_function".to_string(),
            name: "Function Implementation".to_string(),
            description: Some("Request implementation of a specific function".to_string()),
            category: TemplateCategory::CodeRequest,
            content: "Please implement a {{language}} function that {{description}}.\n\nFunction signature:\n```{{language}}\n{{signature}}\n```\n\nRequirements:\n{{requirements}}".to_string(),
            variables: vec![
                TemplateVariable {
                    name: "language".to_string(),
                    description: Some("Programming language".to_string()),
                    default_value: Some("rust".to_string()),
                    required: true,
                    var_type: VariableType::Choice,
                    options: Some(vec![
                        "rust".to_string(),
                        "python".to_string(),
                        "javascript".to_string(),
                        "typescript".to_string(),
                        "go".to_string(),
                        "java".to_string(),
                        "cpp".to_string(),
                    ]),
                },
                TemplateVariable {
                    name: "description".to_string(),
                    description: Some("What the function should do".to_string()),
                    default_value: None,
                    required: true,
                    var_type: VariableType::Text,
                    options: None,
                },
                TemplateVariable {
                    name: "signature".to_string(),
                    description: Some("Function signature".to_string()),
                    default_value: None,
                    required: true,
                    var_type: VariableType::Code,
                    options: None,
                },
                TemplateVariable {
                    name: "requirements".to_string(),
                    description: Some("Additional requirements".to_string()),
                    default_value: Some("- Efficient implementation\n- Proper error handling\n- Documentation".to_string()),
                    required: false,
                    var_type: VariableType::Text,
                    options: None,
                },
            ],
            tags: vec!["code".to_string(), "function".to_string(), "implementation".to_string()],
            visibility: TemplateVisibility::System,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            usage_count: 0,
            shortcuts: vec!["func".to_string(), "implement".to_string()],
        });
        
        // Bug report template
        self.add_builtin_template(ChatTemplate {
            id: "bug_report".to_string(),
            name: "Bug Report".to_string(),
            description: Some("Report a bug with detailed information".to_string()),
            category: TemplateCategory::BugReport,
            content: "I've encountered a bug:\n\n**Description**: {{description}}\n\n**Steps to reproduce**:\n{{steps}}\n\n**Expected behavior**: {{expected}}\n\n**Actual behavior**: {{actual}}\n\n**Environment**:\n- OS: {{os}}\n- Version: {{version}}\n\n**Error message** (if any):\n```\n{{error}}\n```".to_string(),
            variables: vec![
                TemplateVariable {
                    name: "description".to_string(),
                    description: Some("Brief description of the bug".to_string()),
                    default_value: None,
                    required: true,
                    var_type: VariableType::Text,
                    options: None,
                },
                TemplateVariable {
                    name: "steps".to_string(),
                    description: Some("Steps to reproduce".to_string()),
                    default_value: Some("1. \n2. \n3. ".to_string()),
                    required: true,
                    var_type: VariableType::Text,
                    options: None,
                },
                TemplateVariable {
                    name: "expected".to_string(),
                    description: Some("What should happen".to_string()),
                    default_value: None,
                    required: true,
                    var_type: VariableType::Text,
                    options: None,
                },
                TemplateVariable {
                    name: "actual".to_string(),
                    description: Some("What actually happens".to_string()),
                    default_value: None,
                    required: true,
                    var_type: VariableType::Text,
                    options: None,
                },
                TemplateVariable {
                    name: "os".to_string(),
                    description: Some("Operating system".to_string()),
                    default_value: None,
                    required: false,
                    var_type: VariableType::Text,
                    options: None,
                },
                TemplateVariable {
                    name: "version".to_string(),
                    description: Some("Software version".to_string()),
                    default_value: None,
                    required: false,
                    var_type: VariableType::Text,
                    options: None,
                },
                TemplateVariable {
                    name: "error".to_string(),
                    description: Some("Error message or stack trace".to_string()),
                    default_value: None,
                    required: false,
                    var_type: VariableType::Code,
                    options: None,
                },
            ],
            tags: vec!["bug".to_string(), "issue".to_string(), "problem".to_string()],
            visibility: TemplateVisibility::System,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            usage_count: 0,
            shortcuts: vec!["bug".to_string()],
        });
        
        // Analysis template
        self.add_builtin_template(ChatTemplate {
            id: "code_analysis".to_string(),
            name: "Code Analysis Request".to_string(),
            description: Some("Request analysis of code".to_string()),
            category: TemplateCategory::Analysis,
            content: "Please analyze the following {{language}} code:\n\n```{{language}}\n{{code}}\n```\n\nI'm particularly interested in:\n{{focus_areas}}\n\nPlease provide suggestions for improvement.".to_string(),
            variables: vec![
                TemplateVariable {
                    name: "language".to_string(),
                    description: Some("Programming language".to_string()),
                    default_value: Some("rust".to_string()),
                    required: true,
                    var_type: VariableType::Choice,
                    options: Some(vec![
                        "rust".to_string(),
                        "python".to_string(),
                        "javascript".to_string(),
                        "typescript".to_string(),
                        "go".to_string(),
                    ]),
                },
                TemplateVariable {
                    name: "code".to_string(),
                    description: Some("Code to analyze".to_string()),
                    default_value: None,
                    required: true,
                    var_type: VariableType::Code,
                    options: None,
                },
                TemplateVariable {
                    name: "focus_areas".to_string(),
                    description: Some("Areas to focus on".to_string()),
                    default_value: Some("- Performance\n- Error handling\n- Code style\n- Best practices".to_string()),
                    required: false,
                    var_type: VariableType::Text,
                    options: None,
                },
            ],
            tags: vec!["analysis".to_string(), "review".to_string(), "code".to_string()],
            visibility: TemplateVisibility::System,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            usage_count: 0,
            shortcuts: vec!["analyze".to_string(), "review".to_string()],
        });
    }
    
    /// Add a built-in template
    fn add_builtin_template(&mut self, template: ChatTemplate) {
        // Register shortcuts
        for shortcut in &template.shortcuts {
            self.shortcuts.insert(shortcut.clone(), template.id.clone());
        }
        
        self.builtin_templates.insert(template.id.clone(), template);
    }
    
    /// Create a new template
    pub fn create_template(&mut self, template: ChatTemplate) -> Result<()> {
        if self.templates.contains_key(&template.id) || self.builtin_templates.contains_key(&template.id) {
            return Err(anyhow!("Template with ID {} already exists", template.id));
        }
        
        // Register shortcuts
        for shortcut in &template.shortcuts {
            if self.shortcuts.contains_key(shortcut) {
                return Err(anyhow!("Shortcut '{}' is already in use", shortcut));
            }
        }
        
        for shortcut in &template.shortcuts {
            self.shortcuts.insert(shortcut.clone(), template.id.clone());
        }
        
        self.templates.insert(template.id.clone(), template);
        self.save_templates()?;
        
        Ok(())
    }
    
    /// Get a template by ID
    pub fn get_template(&self, id: &str) -> Option<&ChatTemplate> {
        self.templates.get(id).or_else(|| self.builtin_templates.get(id))
    }
    
    /// Get a template by shortcut
    pub fn get_template_by_shortcut(&self, shortcut: &str) -> Option<&ChatTemplate> {
        self.shortcuts.get(shortcut)
            .and_then(|id| self.get_template(id))
    }
    
    /// List all templates
    pub fn list_templates(&self) -> Vec<&ChatTemplate> {
        let mut templates: Vec<_> = self.templates.values()
            .chain(self.builtin_templates.values())
            .collect();
        
        templates.sort_by(|a, b| a.name.cmp(&b.name));
        templates
    }
    
    /// List templates by category
    pub fn list_templates_by_category(&self, category: &TemplateCategory) -> Vec<&ChatTemplate> {
        self.list_templates()
            .into_iter()
            .filter(|t| &t.category == category)
            .collect()
    }
    
    /// Search templates
    pub fn search_templates(&self, query: &str) -> Vec<&ChatTemplate> {
        let query_lower = query.to_lowercase();
        
        self.list_templates()
            .into_iter()
            .filter(|t| {
                t.name.to_lowercase().contains(&query_lower) ||
                t.description.as_ref().map(|d| d.to_lowercase().contains(&query_lower)).unwrap_or(false) ||
                t.tags.iter().any(|tag| tag.to_lowercase().contains(&query_lower))
            })
            .collect()
    }
    
    /// Apply a template with variables
    pub fn apply_template(&mut self, template_id: &str, variables: HashMap<String, String>) -> Result<String> {
        let template = self.get_template(template_id)
            .ok_or_else(|| anyhow!("Template not found: {}", template_id))?
            .clone();
        
        let mut content = template.content.clone();
        
        // Validate required variables
        for var in &template.variables {
            if var.required && !variables.contains_key(&var.name) {
                if let Some(default) = &var.default_value {
                    content = content.replace(&format!("{{{{{}}}}}", var.name), default);
                } else {
                    return Err(anyhow!("Required variable '{}' not provided", var.name));
                }
            }
        }
        
        // Apply variables
        for (name, value) in &variables {
            content = content.replace(&format!("{{{{{}}}}}", name), value);
        }
        
        // Apply defaults for missing optional variables
        for var in &template.variables {
            if !var.required && !variables.contains_key(&var.name) {
                if let Some(default) = &var.default_value {
                    content = content.replace(&format!("{{{{{}}}}}", var.name), default);
                } else {
                    content = content.replace(&format!("{{{{{}}}}}", var.name), "");
                }
            }
        }
        
        // Update usage count
        if let Some(template) = self.templates.get_mut(template_id) {
            template.usage_count += 1;
            template.updated_at = Utc::now();
            let _ = self.save_templates();
        }
        
        Ok(content)
    }
    
    /// Create a code snippet
    pub fn create_snippet(&mut self, snippet: CodeSnippet) -> Result<()> {
        if self.snippets.contains_key(&snippet.id) {
            return Err(anyhow!("Snippet with ID {} already exists", snippet.id));
        }
        
        // Register shortcuts
        for shortcut in &snippet.shortcuts {
            if self.shortcuts.contains_key(shortcut) {
                return Err(anyhow!("Shortcut '{}' is already in use", shortcut));
            }
        }
        
        for shortcut in &snippet.shortcuts {
            self.shortcuts.insert(shortcut.clone(), snippet.id.clone());
        }
        
        self.snippets.insert(snippet.id.clone(), snippet);
        self.save_snippets()?;
        
        Ok(())
    }
    
    /// Get a snippet by ID
    pub fn get_snippet(&self, id: &str) -> Option<&CodeSnippet> {
        self.snippets.get(id)
    }
    
    /// List all snippets
    pub fn list_snippets(&self) -> Vec<&CodeSnippet> {
        let mut snippets: Vec<_> = self.snippets.values().collect();
        snippets.sort_by(|a, b| a.name.cmp(&b.name));
        snippets
    }
    
    /// List snippets by language
    pub fn list_snippets_by_language(&self, language: &str) -> Vec<&CodeSnippet> {
        self.snippets.values()
            .filter(|s| s.language.eq_ignore_ascii_case(language))
            .collect()
    }
    
    /// Apply a snippet with variables
    pub fn apply_snippet(&mut self, snippet_id: &str, variables: HashMap<String, String>) -> Result<String> {
        let snippet = self.get_snippet(snippet_id)
            .ok_or_else(|| anyhow!("Snippet not found: {}", snippet_id))?
            .clone();
        
        let mut code = snippet.code.clone();
        
        // Apply variables (same logic as templates)
        for var in &snippet.variables {
            if let Some(value) = variables.get(&var.name) {
                code = code.replace(&format!("{{{{{}}}}}", var.name), value);
            } else if let Some(default) = &var.default_value {
                code = code.replace(&format!("{{{{{}}}}}", var.name), default);
            } else if var.required {
                return Err(anyhow!("Required variable '{}' not provided", var.name));
            } else {
                code = code.replace(&format!("{{{{{}}}}}", var.name), "");
            }
        }
        
        // Update usage count
        if let Some(snippet) = self.snippets.get_mut(snippet_id) {
            snippet.usage_count += 1;
            let _ = self.save_snippets();
        }
        
        Ok(code)
    }
    
    /// Delete a template
    pub fn delete_template(&mut self, template_id: &str) -> Result<()> {
        if self.builtin_templates.contains_key(template_id) {
            return Err(anyhow!("Cannot delete built-in template"));
        }
        
        if let Some(template) = self.templates.remove(template_id) {
            // Remove shortcuts
            for shortcut in &template.shortcuts {
                self.shortcuts.remove(shortcut);
            }
            
            self.save_templates()?;
            Ok(())
        } else {
            Err(anyhow!("Template not found: {}", template_id))
        }
    }
    
    /// Delete a snippet
    pub fn delete_snippet(&mut self, snippet_id: &str) -> Result<()> {
        if let Some(snippet) = self.snippets.remove(snippet_id) {
            // Remove shortcuts
            for shortcut in &snippet.shortcuts {
                self.shortcuts.remove(shortcut);
            }
            
            self.save_snippets()?;
            Ok(())
        } else {
            Err(anyhow!("Snippet not found: {}", snippet_id))
        }
    }
    
    /// Save templates to disk
    fn save_templates(&self) -> Result<()> {
        if let Some(storage_path) = &self.storage_path {
            let templates_path = storage_path.join("templates.json");
            let json = serde_json::to_string_pretty(&self.templates)?;
            std::fs::write(templates_path, json)?;
        }
        Ok(())
    }
    
    /// Save snippets to disk
    fn save_snippets(&self) -> Result<()> {
        if let Some(storage_path) = &self.storage_path {
            let snippets_path = storage_path.join("snippets.json");
            let json = serde_json::to_string_pretty(&self.snippets)?;
            std::fs::write(snippets_path, json)?;
        }
        Ok(())
    }
    
    /// Load templates from disk
    pub fn load_templates(&mut self) -> Result<()> {
        if let Some(storage_path) = &self.storage_path {
            let templates_path = storage_path.join("templates.json");
            if templates_path.exists() {
                let json = std::fs::read_to_string(templates_path)?;
                self.templates = serde_json::from_str(&json)?;
                
                // Rebuild shortcuts
                for template in self.templates.values() {
                    for shortcut in &template.shortcuts {
                        self.shortcuts.insert(shortcut.clone(), template.id.clone());
                    }
                }
            }
        }
        Ok(())
    }
    
    /// Load snippets from disk
    pub fn load_snippets(&mut self) -> Result<()> {
        if let Some(storage_path) = &self.storage_path {
            let snippets_path = storage_path.join("snippets.json");
            if snippets_path.exists() {
                let json = std::fs::read_to_string(snippets_path)?;
                self.snippets = serde_json::from_str(&json)?;
                
                // Rebuild shortcuts
                for snippet in self.snippets.values() {
                    for shortcut in &snippet.shortcuts {
                        self.shortcuts.insert(shortcut.clone(), snippet.id.clone());
                    }
                }
            }
        }
        Ok(())
    }
}

impl Default for TemplateManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Template variable input widget state
#[derive(Debug, Clone)]
pub struct TemplateInputState {
    pub template_id: String,
    pub variable_values: HashMap<String, String>,
    pub current_variable_index: usize,
    pub error_message: Option<String>,
}

impl TemplateInputState {
    pub fn new(template_id: String, template: &ChatTemplate) -> Self {
        let mut variable_values = HashMap::new();
        
        // Pre-fill with defaults
        for var in &template.variables {
            if let Some(default) = &var.default_value {
                variable_values.insert(var.name.clone(), default.clone());
            }
        }
        
        Self {
            template_id,
            variable_values,
            current_variable_index: 0,
            error_message: None,
        }
    }
    
    /// Get the current variable being edited
    pub fn current_variable<'a>(&self, template: &'a ChatTemplate) -> Option<&'a TemplateVariable> {
        template.variables.get(self.current_variable_index)
    }
    
    /// Move to next variable
    pub fn next_variable(&mut self, template: &ChatTemplate) {
        if self.current_variable_index < template.variables.len().saturating_sub(1) {
            self.current_variable_index += 1;
        }
    }
    
    /// Move to previous variable
    pub fn previous_variable(&mut self) {
        if self.current_variable_index > 0 {
            self.current_variable_index -= 1;
        }
    }
    
    /// Validate all required variables are filled
    pub fn validate(&self, template: &ChatTemplate) -> Result<()> {
        for var in &template.variables {
            if var.required && !self.variable_values.contains_key(&var.name) {
                return Err(anyhow!("Required variable '{}' is not filled", var.name));
            }
        }
        Ok(())
    }
}