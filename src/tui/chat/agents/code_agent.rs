//! Specialized Code Implementation Agent
//! 
//! An intelligent agent specifically designed for code generation, implementation,
//! and collaborative coding tasks with direct editor integration.

use std::sync::Arc;
use std::collections::HashMap;
use tokio::sync::{RwLock, mpsc};
use anyhow::{Result, Context};
use serde::{Serialize, Deserialize};
use uuid::Uuid;

use crate::tui::bridges::EditorBridge;
use crate::tui::chat::editor::{EditAction, CursorPosition, SelectionRange};
use crate::cognitive::agents::AgentSpecialization;
use crate::models::orchestrator::{TaskRequest, TaskType, TaskConstraints};

/// Specialized agent for code implementation
pub struct CodeAgent {
    /// Agent ID
    id: String,
    
    /// Agent name
    name: String,
    
    /// Specialization details
    specialization: CodeSpecialization,
    
    /// Editor bridge for direct code manipulation
    editor_bridge: Option<Arc<crate::tui::bridges::EditorBridge>>,
    
    /// Current task
    current_task: Arc<RwLock<Option<CodeTask>>>,
    
    /// Implementation state
    state: Arc<RwLock<CodeAgentState>>,
    
    /// Progress tracker
    progress: Arc<RwLock<ImplementationProgress>>,
    
    /// Message channel for updates
    update_tx: mpsc::Sender<AgentUpdate>,
    
    /// Model orchestrator for code generation
    model_orchestrator: Option<Arc<crate::models::ModelOrchestrator>>,
    
    /// Filesystem operations flag (we'll use direct tokio::fs for now)
    enable_filesystem: bool,
}

/// Code specialization types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CodeSpecialization {
    /// General code implementation
    General {
        languages: Vec<String>,
        frameworks: Vec<String>,
    },
    
    /// Backend development
    Backend {
        languages: Vec<String>,
        databases: Vec<String>,
        apis: Vec<String>,
    },
    
    /// Frontend development
    Frontend {
        frameworks: Vec<String>,
        ui_libraries: Vec<String>,
        build_tools: Vec<String>,
    },
    
    /// Algorithm implementation
    Algorithm {
        complexity_level: String,
        domains: Vec<String>,
    },
    
    /// Testing and QA
    Testing {
        test_frameworks: Vec<String>,
        coverage_tools: Vec<String>,
    },
    
    /// DevOps and infrastructure
    DevOps {
        ci_cd_tools: Vec<String>,
        cloud_platforms: Vec<String>,
        iac_tools: Vec<String>,
    },
    
    /// Code review and refactoring
    Refactoring {
        analysis_tools: Vec<String>,
        patterns: Vec<String>,
    },
}

/// Code implementation task
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeTask {
    pub id: String,
    pub description: String,
    pub requirements: Vec<String>,
    pub language: Option<String>,
    pub framework: Option<String>,
    pub files: Vec<String>,
    pub context: HashMap<String, String>,
    pub priority: TaskPriority,
    pub estimated_complexity: ComplexityLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskPriority {
    Low,
    Normal,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplexityLevel {
    Trivial,
    Simple,
    Moderate,
    Complex,
    VeryComplex,
}

/// Agent state
#[derive(Debug, Clone)]
pub struct CodeAgentState {
    pub status: AgentStatus,
    pub current_file: Option<String>,
    pub current_line: Option<usize>,
    pub generated_code: Vec<GeneratedCode>,
    pub errors: Vec<String>,
    pub suggestions: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AgentStatus {
    Idle,
    Analyzing,
    Planning,
    Implementing,
    Testing,
    Reviewing,
    Completed,
    Failed,
}

#[derive(Debug, Clone)]
pub struct GeneratedCode {
    pub file_path: String,
    pub content: String,
    pub language: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Implementation progress
#[derive(Debug, Clone)]
pub struct ImplementationProgress {
    pub total_steps: usize,
    pub completed_steps: usize,
    pub current_step: String,
    pub percentage: f32,
    pub estimated_time_remaining: Option<std::time::Duration>,
}

/// Agent update message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentUpdate {
    pub agent_id: String,
    pub update_type: UpdateType,
    pub message: String,
    pub data: Option<serde_json::Value>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UpdateType {
    StatusChange,
    Progress,
    CodeGenerated,
    Error,
    Suggestion,
    Completed,
}

impl CodeAgent {
    /// Get agent ID
    pub fn id(&self) -> &str {
        &self.id
    }
    
    /// Get agent name
    pub fn name(&self) -> &str {
        &self.name
    }
    
    /// Create a new code agent
    pub fn new(
        name: String,
        specialization: CodeSpecialization,
        update_tx: mpsc::Sender<AgentUpdate>,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            name,
            specialization,
            editor_bridge: None,
            current_task: Arc::new(RwLock::new(None)),
            state: Arc::new(RwLock::new(CodeAgentState {
                status: AgentStatus::Idle,
                current_file: None,
                current_line: None,
                generated_code: Vec::new(),
                errors: Vec::new(),
                suggestions: Vec::new(),
            })),
            progress: Arc::new(RwLock::new(ImplementationProgress {
                total_steps: 0,
                completed_steps: 0,
                current_step: String::new(),
                percentage: 0.0,
                estimated_time_remaining: None,
            })),
            update_tx,
            model_orchestrator: None,
            enable_filesystem: true,
        }
    }
    
    /// Set model orchestrator for code generation
    pub fn set_model_orchestrator(&mut self, orchestrator: Arc<crate::models::ModelOrchestrator>) {
        self.model_orchestrator = Some(orchestrator);
    }
    
    /// Set editor bridge for direct editor integration
    pub fn set_editor_bridge(&mut self, bridge: Arc<crate::tui::bridges::EditorBridge>) {
        self.editor_bridge = Some(bridge);
    }
    
    /// Execute a task (wrapper for Arc<Self>)
    pub async fn execute_task(self: Arc<Self>, task: CodeTask) -> Result<()> {
        self.accept_task(task).await
    }
    
    /// Accept a code implementation task
    pub async fn accept_task(&self, task: CodeTask) -> Result<()> {
        // Update state
        {
            let mut state = self.state.write().await;
            state.status = AgentStatus::Analyzing;
            state.errors.clear();
            state.suggestions.clear();
        }
        
        // Store task
        *self.current_task.write().await = Some(task.clone());
        
        // Send update
        self.send_update(
            UpdateType::StatusChange,
            format!("Accepted task: {}", task.description),
            None,
        ).await?;
        
        // Start implementation
        self.implement_task(task).await?;
        
        Ok(())
    }
    
    /// Implement the code task
    async fn implement_task(&self, task: CodeTask) -> Result<()> {
        // Phase 1: Analyze requirements
        self.update_status(AgentStatus::Analyzing).await?;
        let analysis = self.analyze_requirements(&task).await?;
        
        // Phase 2: Create implementation plan
        self.update_status(AgentStatus::Planning).await?;
        let plan = self.create_implementation_plan(&task, &analysis).await?;
        
        // Phase 3: Generate code
        self.update_status(AgentStatus::Implementing).await?;
        let code = self.generate_code(&task, &plan).await?;
        
        // Phase 4: Write to editor AND filesystem
        for generated in &code {
            // Write to actual filesystem if enabled
            if self.enable_filesystem && !generated.file_path.is_empty() {
                self.write_to_filesystem(&generated.file_path, &generated.content).await?;
            }
            
            // Also transfer to editor if bridge is available
            if let Some(ref bridge) = self.editor_bridge {
                bridge.transfer_code_to_editor(
                    self.id.clone(),
                    generated.content.clone(),
                    Some(generated.language.clone()),
                    Some(generated.file_path.clone()),
                ).await?;
            }
        }
        
        // Phase 5: Test implementation
        self.update_status(AgentStatus::Testing).await?;
        self.test_implementation(&code).await?;
        
        // Phase 6: Review and refine
        self.update_status(AgentStatus::Reviewing).await?;
        self.review_and_refine(&code).await?;
        
        // Complete
        self.update_status(AgentStatus::Completed).await?;
        
        Ok(())
    }
    
    /// Analyze task requirements
    async fn analyze_requirements(&self, task: &CodeTask) -> Result<RequirementsAnalysis> {
        self.update_progress(
            "Analyzing requirements",
            0.1,
        ).await?;
        
        let analysis = RequirementsAnalysis {
            main_components: self.identify_components(&task.description),
            dependencies: self.identify_dependencies(&task.requirements),
            patterns: self.identify_patterns(&task.description),
            estimated_loc: self.estimate_lines_of_code(&task.estimated_complexity),
        };
        
        self.send_update(
            UpdateType::Progress,
            format!("Identified {} components, {} dependencies", 
                analysis.main_components.len(),
                analysis.dependencies.len()),
            Some(serde_json::to_value(&analysis)?),
        ).await?;
        
        Ok(analysis)
    }
    
    /// Create implementation plan
    async fn create_implementation_plan(
        &self,
        task: &CodeTask,
        analysis: &RequirementsAnalysis,
    ) -> Result<ImplementationPlan> {
        self.update_progress(
            "Creating implementation plan",
            0.2,
        ).await?;
        
        let mut steps = Vec::new();
        
        // Add setup steps
        steps.push(ImplementationStep {
            name: "Setup project structure".to_string(),
            description: "Create necessary files and directories".to_string(),
            estimated_time: std::time::Duration::from_secs(60),
            dependencies: Vec::new(),
        });
        
        // Add component implementation steps
        for component in &analysis.main_components {
            steps.push(ImplementationStep {
                name: format!("Implement {}", component),
                description: format!("Create {} with required functionality", component),
                estimated_time: std::time::Duration::from_secs(300),
                dependencies: Vec::new(),
            });
        }
        
        // Add testing step
        steps.push(ImplementationStep {
            name: "Add tests".to_string(),
            description: "Create unit and integration tests".to_string(),
            estimated_time: std::time::Duration::from_secs(180),
            dependencies: vec!["implementation".to_string()],
        });
        
        let plan = ImplementationPlan {
            steps,
            total_estimated_time: std::time::Duration::from_secs(1000),
            parallelizable: false,
        };
        
        // Update progress with total steps
        {
            let mut progress = self.progress.write().await;
            progress.total_steps = plan.steps.len();
            progress.estimated_time_remaining = Some(plan.total_estimated_time);
        }
        
        Ok(plan)
    }
    
    /// Generate code implementation
    async fn generate_code(
        &self,
        task: &CodeTask,
        plan: &ImplementationPlan,
    ) -> Result<Vec<GeneratedCode>> {
        let mut generated = Vec::new();
        
        for (i, step) in plan.steps.iter().enumerate() {
            self.update_progress(
                &step.name,
                0.3 + (0.4 * i as f32 / plan.steps.len() as f32),
            ).await?;
            
            // Generate code for this step
            let code = self.generate_step_code(task, step).await?;
            
            if let Some(code) = code {
                generated.push(code.clone());
                
                // Store in state
                self.state.write().await.generated_code.push(code);
                
                // Send update
                self.send_update(
                    UpdateType::CodeGenerated,
                    format!("Generated code for: {}", step.name),
                    None,
                ).await?;
            }
            
            // Update completed steps
            {
                let mut progress = self.progress.write().await;
                progress.completed_steps = i + 1;
            }
        }
        
        Ok(generated)
    }
    
    /// Generate code for a specific step
    async fn generate_step_code(
        &self,
        task: &CodeTask,
        step: &ImplementationStep,
    ) -> Result<Option<GeneratedCode>> {
        let language = task.language.as_deref().unwrap_or("rust");
        let file_name = format!("{}.{}", 
            step.name.to_lowercase().replace(" ", "_"),
            self.get_file_extension(language));
        
        // Use model orchestrator if available
        let code = if let Some(ref orchestrator) = self.model_orchestrator {
            // Create a detailed prompt for code generation
            let prompt = self.create_code_generation_prompt(task, step, language);
            
            // Create task request for model
            let task_request = crate::models::orchestrator::TaskRequest {
                task_type: crate::models::orchestrator::TaskType::CodeGeneration { 
                    language: language.to_string() 
                },
                content: prompt.clone(),
                constraints: crate::models::orchestrator::TaskConstraints {
                    max_tokens: Some(2000),
                    context_size: Some(4096),
                    max_time: Some(std::time::Duration::from_secs(30)),
                    max_latency_ms: Some(30000),
                    max_cost_cents: None,
                    quality_threshold: Some(0.8),
                    priority: "normal".to_string(),
                    prefer_local: false,
                    require_streaming: false,
                    required_capabilities: vec!["code_generation".to_string()],
                    creativity_level: Some(0.7),
                    formality_level: Some(0.8),
                    target_audience: Some("developers".to_string()),
                },
                context_integration: true,
                memory_integration: false,
                cognitive_enhancement: false,
            };
            
            // Execute with fallback
            match orchestrator.execute_with_fallback(task_request).await {
                Ok(response) => {
                    // Extract code from response
                    self.extract_code_from_response(&response.content, language)
                }
                Err(e) => {
                    tracing::warn!("Model orchestrator failed, using template: {}", e);
                    self.generate_template_code(language, &step.name, &step.description)
                }
            }
        } else {
            // Fall back to template if no orchestrator
            self.generate_template_code(language, &step.name, &step.description)
        };
        
        Ok(Some(GeneratedCode {
            file_path: file_name,
            content: code,
            language: language.to_string(),
            timestamp: chrono::Utc::now(),
        }))
    }
    
    /// Create a detailed prompt for code generation
    fn create_code_generation_prompt(&self, task: &CodeTask, step: &ImplementationStep, language: &str) -> String {
        format!(
            "Generate {} code for the following task:\n\n\
            Task: {}\n\
            Step: {}\n\
            Description: {}\n\n\
            Requirements:\n\
            - Write clean, idiomatic {} code\n\
            - Include appropriate error handling\n\
            - Add helpful comments\n\
            - Follow best practices\n\
            - Make the code production-ready\n\n\
            Please provide only the code without markdown code blocks.",
            language,
            task.description,
            step.name,
            step.description,
            language
        )
    }
    
    /// Extract code from model response
    fn extract_code_from_response(&self, response: &str, language: &str) -> String {
        // If response contains code blocks, extract them
        if response.contains("```") {
            let parts: Vec<&str> = response.split("```").collect();
            if parts.len() >= 3 {
                // Get the code block (usually the second element)
                let code_block = parts[1];
                // Remove language identifier if present
                if code_block.starts_with(language) {
                    return code_block[language.len()..].trim().to_string();
                } else if let Some(pos) = code_block.find('\n') {
                    return code_block[pos + 1..].to_string();
                }
                return code_block.to_string();
            }
        }
        // Return as-is if no code blocks found
        response.to_string()
    }
    
    /// Generate template code as fallback
    fn generate_template_code(&self, language: &str, name: &str, description: &str) -> String {
        match language {
            "rust" => self.generate_rust_template(name, description),
            "python" => self.generate_python_template(name, description),
            "javascript" => self.generate_js_template(name, description),
            _ => format!("// Implementation for: {}\n// {}\n\n// Add your code here\n", name, description),
        }
    }
    
    /// Generate Rust template
    fn generate_rust_template(&self, name: &str, description: &str) -> String {
        let struct_name = name.replace(" ", "").replace("-", "_");
        format!(r#"//! {}
//! 
//! {}

use anyhow::Result;
use tracing::{{info, debug}};

/// Main implementation for {}
#[derive(Debug, Clone)]
pub struct {} {{
    /// Configuration or state fields
    config: Config,
}}

/// Configuration for {}
#[derive(Debug, Clone, Default)]
pub struct Config {{
    // Add configuration fields as needed
}}

impl {} {{
    /// Create a new instance with default configuration
    pub fn new() -> Self {{
        Self {{
            config: Config::default(),
        }}
    }}
    
    /// Create with custom configuration
    pub fn with_config(config: Config) -> Self {{
        Self {{ config }}
    }}
    
    /// Main functionality implementation
    pub async fn execute(&self) -> Result<()> {{
        info!("Executing {}");
        
        // Step 1: Initialization
        self.initialize().await?;
        
        // Step 2: Main processing
        self.process().await?;
        
        // Step 3: Cleanup
        self.cleanup().await?;
        
        info!("Successfully completed {}");
        Ok(())
    }}
    
    /// Initialize resources
    async fn initialize(&self) -> Result<()> {{
        debug!("Initializing...");
        // Add initialization logic here
        Ok(())
    }}
    
    /// Main processing logic
    async fn process(&self) -> Result<()> {{
        debug!("Processing...");
        // Add main logic here
        Ok(())
    }}
    
    /// Cleanup resources
    async fn cleanup(&self) -> Result<()> {{
        debug!("Cleaning up...");
        // Add cleanup logic here
        Ok(())
    }}
}}

#[cfg(test)]
mod tests {{
    use super::*;
    
    #[tokio::test]
    async fn test_execute() {{
        let instance = {}::new();
        assert!(instance.execute().await.is_ok());
    }}
}}
"#, name, description, name, struct_name, struct_name, struct_name, name, name, struct_name)
    }
    
    /// Generate Python template
    fn generate_python_template(&self, name: &str, description: &str) -> String {
        format!(r#"""{}

{}
"""

class {}:
    """Main implementation class."""
    
    def __init__(self):
        """Initialize the {}."""
        # Add initialization
        pass
    
    def execute(self):
        """Main functionality."""
        # Implementation goes here
        raise NotImplementedError("Implement {}")

if __name__ == "__main__":
    instance = {}()
    instance.execute()
"#, name, description,
    name.replace(" ", ""),
    name.replace(" ", ""),
    name,
    name.replace(" ", ""))
    }
    
    /// Generate JavaScript template
    fn generate_js_template(&self, name: &str, description: &str) -> String {
        format!(r#"/**
 * {}
 * 
 * {}
 */

class {} {{
    constructor() {{
        // Initialize
    }}
    
    /**
     * Main functionality
     */
    async execute() {{
        // Implementation goes here
        throw new Error("Implement: {}");
    }}
}}

module.exports = {};
"#, name, description,
    name.replace(" ", ""),
    name,
    name.replace(" ", ""))
    }
    
    /// Get file extension for language
    fn get_file_extension(&self, language: &str) -> &str {
        match language {
            "rust" => "rs",
            "python" => "py",
            "javascript" => "js",
            "typescript" => "ts",
            "go" => "go",
            "java" => "java",
            "cpp" => "cpp",
            "c" => "c",
            _ => "txt",
        }
    }
    
    /// Test the implementation
    async fn test_implementation(&self, code: &[GeneratedCode]) -> Result<()> {
        self.update_progress(
            "Testing implementation",
            0.8,
        ).await?;
        
        // Run tests if available
        // This would integrate with the test runner
        
        self.send_update(
            UpdateType::Progress,
            "Tests completed successfully".to_string(),
            None,
        ).await?;
        
        Ok(())
    }
    
    /// Review and refine the code
    async fn review_and_refine(&self, code: &[GeneratedCode]) -> Result<()> {
        self.update_progress(
            "Reviewing and refining",
            0.9,
        ).await?;
        
        // Add suggestions
        let mut suggestions = Vec::new();
        suggestions.push("Consider adding error handling".to_string());
        suggestions.push("Add documentation comments".to_string());
        suggestions.push("Consider performance optimizations".to_string());
        
        self.state.write().await.suggestions = suggestions.clone();
        
        for suggestion in suggestions {
            self.send_update(
                UpdateType::Suggestion,
                suggestion,
                None,
            ).await?;
        }
        
        Ok(())
    }
    
    /// Update agent status
    async fn update_status(&self, status: AgentStatus) -> Result<()> {
        self.state.write().await.status = status.clone();
        self.send_update(
            UpdateType::StatusChange,
            format!("Status: {:?}", status),
            None,
        ).await
    }
    
    /// Update progress
    async fn update_progress(&self, step: &str, percentage: f32) -> Result<()> {
        {
            let mut progress = self.progress.write().await;
            progress.current_step = step.to_string();
            progress.percentage = percentage;
        }
        
        self.send_update(
            UpdateType::Progress,
            format!("Progress: {} ({}%)", step, (percentage * 100.0) as u32),
            None,
        ).await
    }
    
    /// Send update message
    async fn send_update(
        &self,
        update_type: UpdateType,
        message: String,
        data: Option<serde_json::Value>,
    ) -> Result<()> {
        let update = AgentUpdate {
            agent_id: self.id.clone(),
            update_type,
            message,
            data,
            timestamp: chrono::Utc::now(),
        };
        
        self.update_tx.send(update).await
            .context("Failed to send agent update")?;
        
        Ok(())
    }
    
    /// Helper functions for analysis
    fn identify_components(&self, description: &str) -> Vec<String> {
        // Simple component identification
        let mut components = Vec::new();
        
        if description.contains("api") || description.contains("API") {
            components.push("API Handler".to_string());
        }
        if description.contains("database") || description.contains("storage") {
            components.push("Database Layer".to_string());
        }
        if description.contains("ui") || description.contains("interface") {
            components.push("User Interface".to_string());
        }
        if description.contains("auth") || description.contains("login") {
            components.push("Authentication".to_string());
        }
        
        if components.is_empty() {
            components.push("Main Component".to_string());
        }
        
        components
    }
    
    fn identify_dependencies(&self, requirements: &[String]) -> Vec<String> {
        // Identify potential dependencies from requirements
        let mut deps = Vec::new();
        
        for req in requirements {
            let lower = req.to_lowercase();
            if lower.contains("http") {
                deps.push("HTTP client/server".to_string());
            }
            if lower.contains("json") {
                deps.push("JSON parsing".to_string());
            }
            if lower.contains("async") {
                deps.push("Async runtime".to_string());
            }
        }
        
        deps
    }
    
    fn identify_patterns(&self, description: &str) -> Vec<String> {
        // Identify design patterns
        let mut patterns = Vec::new();
        
        if description.contains("factory") {
            patterns.push("Factory Pattern".to_string());
        }
        if description.contains("singleton") {
            patterns.push("Singleton Pattern".to_string());
        }
        if description.contains("observer") || description.contains("event") {
            patterns.push("Observer Pattern".to_string());
        }
        
        patterns
    }
    
    fn estimate_lines_of_code(&self, complexity: &ComplexityLevel) -> usize {
        match complexity {
            ComplexityLevel::Trivial => 50,
            ComplexityLevel::Simple => 150,
            ComplexityLevel::Moderate => 500,
            ComplexityLevel::Complex => 1500,
            ComplexityLevel::VeryComplex => 5000,
        }
    }
}

/// Requirements analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
struct RequirementsAnalysis {
    main_components: Vec<String>,
    dependencies: Vec<String>,
    patterns: Vec<String>,
    estimated_loc: usize,
}

/// Implementation plan
#[derive(Debug, Clone)]
struct ImplementationPlan {
    steps: Vec<ImplementationStep>,
    total_estimated_time: std::time::Duration,
    parallelizable: bool,
}

#[derive(Debug, Clone)]
struct ImplementationStep {
    name: String,
    description: String,
    estimated_time: std::time::Duration,
    dependencies: Vec<String>,
}

impl CodeAgent {
    /// Write generated code to filesystem
    async fn write_to_filesystem(&self, path: &str, content: &str) -> Result<()> {
        // Ensure parent directory exists
        if let Some(parent) = std::path::Path::new(path).parent() {
            if !parent.as_os_str().is_empty() {
                tokio::fs::create_dir_all(parent).await?;
            }
        }
        
        // Write the file
        tokio::fs::write(path, content).await?;
        tracing::info!("CodeAgent wrote file: {} ({} bytes)", path, content.len());
        
        // Send update
        self.send_update(
            UpdateType::CodeGenerated,
            format!("Created file: {}", path),
            Some(serde_json::json!({
                "file_path": path,
                "size": content.len(),
            })),
        ).await?;
        
        Ok(())
    }
    
    /// Create directory structure for project
    pub async fn create_project_structure(&self, base_path: &str, structure: Vec<String>) -> Result<()> {
        for path in structure {
            let full_path = format!("{}/{}", base_path, path);
            
            if path.ends_with('/') {
                // It's a directory
                tokio::fs::create_dir_all(&full_path).await?;
                tracing::info!("Created directory: {}", full_path);
            } else {
                // It's a file - ensure parent dir exists first
                if let Some(parent) = std::path::Path::new(&full_path).parent() {
                    tokio::fs::create_dir_all(parent).await?;
                }
                // Create empty file
                tokio::fs::write(&full_path, "").await?;
                tracing::info!("Created file: {}", full_path);
            }
        }
        
        Ok(())
    }
}

/// Factory for creating specialized code agents
pub struct CodeAgentFactory;

impl CodeAgentFactory {
    /// Create a code agent based on requirements
    pub fn create_agent(
        requirements: &str,
        update_tx: mpsc::Sender<AgentUpdate>,
    ) -> CodeAgent {
        let specialization = Self::determine_specialization(requirements);
        let name = Self::generate_agent_name(&specialization);
        
        let mut agent = CodeAgent::new(name, specialization, update_tx);
        // Model orchestrator should be set after creation if available
        agent
    }
    
    fn determine_specialization(requirements: &str) -> CodeSpecialization {
        let lower = requirements.to_lowercase();
        
        if lower.contains("frontend") || lower.contains("ui") || lower.contains("react") {
            CodeSpecialization::Frontend {
                frameworks: vec!["React".to_string(), "Vue".to_string()],
                ui_libraries: vec!["Material-UI".to_string()],
                build_tools: vec!["Webpack".to_string()],
            }
        } else if lower.contains("backend") || lower.contains("api") || lower.contains("server") {
            CodeSpecialization::Backend {
                languages: vec!["Rust".to_string(), "Python".to_string()],
                databases: vec!["PostgreSQL".to_string()],
                apis: vec!["REST".to_string(), "GraphQL".to_string()],
            }
        } else if lower.contains("test") || lower.contains("qa") {
            CodeSpecialization::Testing {
                test_frameworks: vec!["Jest".to_string(), "Pytest".to_string()],
                coverage_tools: vec!["Coverage.py".to_string()],
            }
        } else {
            CodeSpecialization::General {
                languages: vec!["Rust".to_string(), "Python".to_string(), "JavaScript".to_string()],
                frameworks: vec![],
            }
        }
    }
    
    fn generate_agent_name(specialization: &CodeSpecialization) -> String {
        match specialization {
            CodeSpecialization::Frontend { .. } => "Frontend Developer Agent".to_string(),
            CodeSpecialization::Backend { .. } => "Backend Developer Agent".to_string(),
            CodeSpecialization::Testing { .. } => "QA Engineer Agent".to_string(),
            CodeSpecialization::Algorithm { .. } => "Algorithm Specialist Agent".to_string(),
            CodeSpecialization::DevOps { .. } => "DevOps Engineer Agent".to_string(),
            CodeSpecialization::Refactoring { .. } => "Code Refactoring Agent".to_string(),
            CodeSpecialization::General { .. } => "General Code Agent".to_string(),
        }
    }
}