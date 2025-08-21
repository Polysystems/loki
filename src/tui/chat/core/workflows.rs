//! Extended Interactive Workflow System for Chat
//! 
//! This module extends the basic interactive workflow system to support
//! tool execution, multi-step tasks, and context-aware interactions.

use std::collections::HashMap;
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

/// Extended interactive workflow states
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ExtendedWorkflowState {
    /// No active workflow
    None,
    
    /// API setup workflow (existing)
    ApiSetup {
        current_provider: Option<String>,
        step: ApiSetupStep,
        collected_keys: HashMap<String, String>,
    },
    
    /// Tool execution workflow
    ToolExecution {
        tool_name: String,
        step: ToolExecutionStep,
        collected_params: HashMap<String, Value>,
        required_params: Vec<ParameterRequirement>,
    },
    
    /// Multi-step task workflow
    MultiStepTask {
        task_id: String,
        task_description: String,
        current_step: usize,
        total_steps: usize,
        steps: Vec<TaskStep>,
        accumulated_results: HashMap<String, Value>,
    },
    
    /// Confirmation dialog
    ConfirmationDialog {
        action: String,
        message: String,
        details: Option<Value>,
        on_confirm: Box<WorkflowContinuation>,
        on_cancel: Box<WorkflowContinuation>,
    },
    
    /// Context collection workflow
    ContextCollection {
        purpose: String,
        required_context: Vec<ContextRequirement>,
        collected_context: HashMap<String, Value>,
        current_index: usize,
    },
}

/// API setup steps (from original)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ApiSetupStep {
    SelectProvider,
    EnterKey { provider: String },
    Confirm { provider: String, key: String },
    Complete,
}

/// Tool execution steps
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ToolExecutionStep {
    /// Collecting parameters
    CollectingParameters { current_param: String },
    
    /// Confirming execution
    ConfirmExecution,
    
    /// Executing tool
    Executing,
    
    /// Showing results
    ShowingResults { result: Value },
}

/// Task step in multi-step workflow
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskStep {
    pub name: String,
    pub description: String,
    pub step_type: StepType,
    pub required_input: Vec<String>,
    pub produces_output: Vec<String>,
}

/// Types of steps in a task
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum StepType {
    UserInput { prompt: String },
    ToolExecution { tool: String, params: Value },
    Analysis { target: String },
    Decision { options: Vec<String> },
    Review { content: String },
}

/// Continuation after a workflow action
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkflowContinuation {
    pub next_state: ExtendedWorkflowState,
    pub message: Option<String>,
}

/// Parameter requirement for tool execution
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParameterRequirement {
    pub name: String,
    pub param_type: String,
    pub description: String,
    pub required: bool,
    pub default: Option<Value>,
    pub validation: Option<ParameterValidation>,
}

/// Parameter validation rules
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParameterValidation {
    pub min_length: Option<usize>,
    pub max_length: Option<usize>,
    pub pattern: Option<String>,
    pub valid_values: Option<Vec<String>>,
}

/// Context requirement
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextRequirement {
    pub name: String,
    pub description: String,
    pub context_type: ContextType,
    pub required: bool,
}

/// Types of context that can be collected
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ContextType {
    File,
    Directory,
    CodeSnippet,
    Description,
    Choice { options: Vec<String> },
    Number { min: Option<f64>, max: Option<f64> },
}

/// Workflow manager for handling interactive workflows
pub struct WorkflowManager {
    /// Current workflow state
    current_state: ExtendedWorkflowState,
    
    /// Workflow history
    history: Vec<WorkflowHistoryEntry>,
    
    /// Available workflow templates
    templates: HashMap<String, WorkflowTemplate>,
}

/// Entry in workflow history
#[derive(Debug, Clone)]
struct WorkflowHistoryEntry {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub state: ExtendedWorkflowState,
    pub user_input: String,
    pub result: WorkflowResult,
}

/// Result of a workflow action
#[derive(Debug, Clone)]
enum WorkflowResult {
    Success { message: String },
    Error { message: String },
    Cancelled,
}

/// Workflow template
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowTemplate {
    pub name: String,
    pub description: String,
    pub initial_state: ExtendedWorkflowState,
    pub tags: Vec<String>,
}

impl WorkflowManager {
    /// Create a new workflow manager
    pub fn new() -> Self {
        let mut manager = Self {
            current_state: ExtendedWorkflowState::None,
            history: Vec::new(),
            templates: HashMap::new(),
        };
        
        // Initialize templates
        manager.init_templates();
        
        manager
    }
    
    /// Initialize workflow templates
    fn init_templates(&mut self) {
        // Code analysis workflow
        self.templates.insert(
            "code_analysis".to_string(),
            WorkflowTemplate {
                name: "Code Analysis".to_string(),
                description: "Analyze code with multiple tools".to_string(),
                initial_state: ExtendedWorkflowState::ContextCollection {
                    purpose: "code_analysis".to_string(),
                    required_context: vec![
                        ContextRequirement {
                            name: "target".to_string(),
                            description: "File or directory to analyze".to_string(),
                            context_type: ContextType::File,
                            required: true,
                        },
                        ContextRequirement {
                            name: "analysis_type".to_string(),
                            description: "Type of analysis to perform".to_string(),
                            context_type: ContextType::Choice {
                                options: vec![
                                    "complexity".to_string(),
                                    "security".to_string(),
                                    "performance".to_string(),
                                    "all".to_string(),
                                ],
                            },
                            required: true,
                        },
                    ],
                    collected_context: HashMap::new(),
                    current_index: 0,
                },
                tags: vec!["analysis".to_string(), "code".to_string()],
            },
        );
        
        // Debugging workflow
        self.templates.insert(
            "debug_assistant".to_string(),
            WorkflowTemplate {
                name: "Debug Assistant".to_string(),
                description: "Step-by-step debugging assistance".to_string(),
                initial_state: ExtendedWorkflowState::MultiStepTask {
                    task_id: uuid::Uuid::new_v4().to_string(),
                    task_description: "Debug code issue".to_string(),
                    current_step: 0,
                    total_steps: 4,
                    steps: vec![
                        TaskStep {
                            name: "describe_issue".to_string(),
                            description: "Describe the issue".to_string(),
                            step_type: StepType::UserInput {
                                prompt: "What issue are you experiencing?".to_string(),
                            },
                            required_input: vec![],
                            produces_output: vec!["issue_description".to_string()],
                        },
                        TaskStep {
                            name: "identify_file".to_string(),
                            description: "Identify relevant files".to_string(),
                            step_type: StepType::UserInput {
                                prompt: "Which file(s) are involved?".to_string(),
                            },
                            required_input: vec!["issue_description".to_string()],
                            produces_output: vec!["target_files".to_string()],
                        },
                        TaskStep {
                            name: "analyze_code".to_string(),
                            description: "Analyze the code".to_string(),
                            step_type: StepType::ToolExecution {
                                tool: "code_analyzer".to_string(),
                                params: json!({ "action": "debug_analysis" }),
                            },
                            required_input: vec!["target_files".to_string()],
                            produces_output: vec!["analysis_results".to_string()],
                        },
                        TaskStep {
                            name: "suggest_fixes".to_string(),
                            description: "Suggest fixes".to_string(),
                            step_type: StepType::Analysis {
                                target: "analysis_results".to_string(),
                            },
                            required_input: vec!["analysis_results".to_string()],
                            produces_output: vec!["suggested_fixes".to_string()],
                        },
                    ],
                    accumulated_results: HashMap::new(),
                },
                tags: vec!["debug".to_string(), "assistant".to_string()],
            },
        );
    }
    
    /// Get current workflow state
    pub fn current_state(&self) -> &ExtendedWorkflowState {
        &self.current_state
    }
    
    /// Start a new workflow
    pub fn start_workflow(&mut self, template_name: &str) -> Result<String> {
        if let Some(template) = self.templates.get(template_name) {
            self.current_state = template.initial_state.clone();
            Ok(self.format_current_prompt())
        } else {
            Err(anyhow!("Unknown workflow template '{}'. Available templates: {:?}", template_name, self.templates.keys().collect::<Vec<_>>()))
        }
    }
    
    /// Process user input for current workflow
    pub fn process_input(&mut self, input: &str) -> Result<WorkflowResponse> {
        let input = input.trim();
        
        // Check for cancel
        if input.eq_ignore_ascii_case("cancel") || input.eq_ignore_ascii_case("exit") {
            self.cancel_workflow();
            return Ok(WorkflowResponse {
                message: "Workflow cancelled.".to_string(),
                next_prompt: None,
                completed: true,
                result: None,
            });
        }
        
        // Process based on current state - clone to avoid borrow issues
        match self.current_state.clone() {
            ExtendedWorkflowState::None => {
                Err(anyhow!("No active workflow. Please start a workflow first using 'start_workflow()'"))
            }
            ExtendedWorkflowState::ToolExecution { tool_name, mut step, mut collected_params, required_params } => {
                self.process_tool_execution(input, &tool_name, &mut step, &mut collected_params, &required_params)
            }
            ExtendedWorkflowState::MultiStepTask { 
                task_id, 
                mut current_step, 
                total_steps, 
                steps, 
                mut accumulated_results,
                .. 
            } => {
                self.process_multi_step_task(input, &task_id, &mut current_step, &total_steps, &steps, &mut accumulated_results)
            }
            ExtendedWorkflowState::ConfirmationDialog { 
                action, 
                on_confirm, 
                on_cancel,
                .. 
            } => {
                self.process_confirmation(input, &action, &on_confirm, &on_cancel)
            }
            ExtendedWorkflowState::ContextCollection {
                purpose,
                required_context,
                mut collected_context,
                mut current_index,
            } => {
                self.process_context_collection(input, &purpose, &required_context, &mut collected_context, &mut current_index)
            }
            _ => {
                // Handle other states
                Ok(WorkflowResponse {
                    message: "Processing...".to_string(),
                    next_prompt: None,
                    completed: false,
                    result: None,
                })
            }
        }
    }
    
    /// Process tool execution workflow
    fn process_tool_execution(
        &mut self,
        input: &str,
        tool_name: &str,
        step: &mut ToolExecutionStep,
        collected_params: &mut HashMap<String, Value>,
        required_params: &[ParameterRequirement],
    ) -> Result<WorkflowResponse> {
        match step {
            ToolExecutionStep::CollectingParameters { current_param } => {
                // Find current parameter requirement
                let param_req = required_params.iter()
                    .find(|p| p.name == *current_param)
                    .ok_or_else(|| anyhow!("Parameter requirement not found for param '{}'", current_param))?;
                
                // Validate and store input
                let value = self.parse_parameter_value(input, &param_req.param_type)?;
                if let Some(validation) = &param_req.validation {
                    self.validate_parameter(&value, validation)?;
                }
                collected_params.insert(current_param.clone(), value);
                
                // Find next required parameter
                if let Some(next_param) = required_params.iter()
                    .find(|p| p.required && !collected_params.contains_key(&p.name))
                {
                    *current_param = next_param.name.clone();
                    Ok(WorkflowResponse {
                        message: format!("Parameter '{}' recorded.", param_req.name),
                        next_prompt: Some(format!(
                            "Please provide {} ({}): {}",
                            next_param.name,
                            next_param.param_type,
                            next_param.description
                        )),
                        completed: false,
                        result: None,
                    })
                } else {
                    // All parameters collected, move to confirmation
                    *step = ToolExecutionStep::ConfirmExecution;
                    Ok(WorkflowResponse {
                        message: format!("All parameters collected for {}.", tool_name),
                        next_prompt: Some(format!(
                            "Ready to execute {} with parameters:\n{}\n\nConfirm? (yes/no)",
                            tool_name,
                            serde_json::to_string_pretty(collected_params)?
                        )),
                        completed: false,
                        result: None,
                    })
                }
            }
            ToolExecutionStep::ConfirmExecution => {
                if input.eq_ignore_ascii_case("yes") || input.eq_ignore_ascii_case("y") {
                    *step = ToolExecutionStep::Executing;
                    Ok(WorkflowResponse {
                        message: format!("Executing {}...", tool_name),
                        next_prompt: None,
                        completed: false,
                        result: Some(json!({
                            "action": "execute_tool",
                            "tool": tool_name,
                            "params": collected_params
                        })),
                    })
                } else {
                    self.cancel_workflow();
                    Ok(WorkflowResponse {
                        message: "Tool execution cancelled.".to_string(),
                        next_prompt: None,
                        completed: true,
                        result: None,
                    })
                }
            }
            _ => {
                Ok(WorkflowResponse {
                    message: "Processing...".to_string(),
                    next_prompt: None,
                    completed: false,
                    result: None,
                })
            }
        }
    }
    
    /// Process multi-step task workflow
    fn process_multi_step_task(
        &mut self,
        input: &str,
        task_id: &str,
        current_step: &mut usize,
        total_steps: &usize,
        steps: &[TaskStep],
        accumulated_results: &mut HashMap<String, Value>,
    ) -> Result<WorkflowResponse> {
        if *current_step >= steps.len() {
            // Task completed
            self.current_state = ExtendedWorkflowState::None;
            return Ok(WorkflowResponse {
                message: "Task completed successfully!".to_string(),
                next_prompt: None,
                completed: true,
                result: Some(json!({
                    "task_id": task_id,
                    "results": accumulated_results
                })),
            });
        }
        
        let step = &steps[*current_step];
        
        match &step.step_type {
            StepType::UserInput { prompt } => {
                // Store user input
                for output in &step.produces_output {
                    accumulated_results.insert(output.clone(), json!(input));
                }
                
                *current_step += 1;
                
                // Get next prompt
                let next_prompt = if *current_step < steps.len() {
                    Some(self.format_step_prompt(&steps[*current_step], accumulated_results))
                } else {
                    None
                };
                
                Ok(WorkflowResponse {
                    message: format!("Step {} of {} completed.", current_step, total_steps),
                    next_prompt,
                    completed: false,
                    result: None,
                })
            }
            StepType::Decision { options } => {
                // Validate decision
                if !options.iter().any(|o| o.eq_ignore_ascii_case(input)) {
                    return Ok(WorkflowResponse {
                        message: format!("Invalid option. Please choose one of: {}", options.join(", ")),
                        next_prompt: Some(step.description.clone()),
                        completed: false,
                        result: None,
                    });
                }
                
                // Store decision
                for output in &step.produces_output {
                    accumulated_results.insert(output.clone(), json!(input));
                }
                
                *current_step += 1;
                
                Ok(WorkflowResponse {
                    message: format!("Selected: {}", input),
                    next_prompt: if *current_step < steps.len() {
                        Some(self.format_step_prompt(&steps[*current_step], accumulated_results))
                    } else {
                        None
                    },
                    completed: false,
                    result: None,
                })
            }
            _ => {
                // Automatic steps
                *current_step += 1;
                
                Ok(WorkflowResponse {
                    message: format!("Executing: {}", step.description),
                    next_prompt: None,
                    completed: false,
                    result: Some(json!({
                        "action": "execute_step",
                        "step": step
                    })),
                })
            }
        }
    }
    
    /// Process confirmation dialog
    fn process_confirmation(
        &mut self,
        input: &str,
        action: &str,
        on_confirm: &WorkflowContinuation,
        on_cancel: &WorkflowContinuation,
    ) -> Result<WorkflowResponse> {
        if input.eq_ignore_ascii_case("yes") || input.eq_ignore_ascii_case("y") {
            self.current_state = on_confirm.next_state.clone();
            Ok(WorkflowResponse {
                message: on_confirm.message.clone().unwrap_or_else(|| format!("Confirmed: {}", action)),
                next_prompt: Some(self.format_current_prompt()),
                completed: false,
                result: None,
            })
        } else {
            self.current_state = on_cancel.next_state.clone();
            Ok(WorkflowResponse {
                message: on_cancel.message.clone().unwrap_or_else(|| "Cancelled.".to_string()),
                next_prompt: if self.current_state == ExtendedWorkflowState::None {
                    None
                } else {
                    Some(self.format_current_prompt())
                },
                completed: self.current_state == ExtendedWorkflowState::None,
                result: None,
            })
        }
    }
    
    /// Process context collection
    fn process_context_collection(
        &mut self,
        input: &str,
        purpose: &str,
        required_context: &[ContextRequirement],
        collected_context: &mut HashMap<String, Value>,
        current_index: &mut usize,
    ) -> Result<WorkflowResponse> {
        if *current_index >= required_context.len() {
            // All context collected
            self.current_state = ExtendedWorkflowState::None;
            return Ok(WorkflowResponse {
                message: "Context collection complete.".to_string(),
                next_prompt: None,
                completed: true,
                result: Some(json!({
                    "purpose": purpose,
                    "context": collected_context
                })),
            });
        }
        
        let requirement = &required_context[*current_index];
        
        // Validate and store input based on context type
        let value = match &requirement.context_type {
            ContextType::File => {
                // Validate file exists (in real implementation)
                json!(input)
            }
            ContextType::Directory => {
                json!(input)
            }
            ContextType::Choice { options } => {
                if !options.iter().any(|o| o.eq_ignore_ascii_case(input)) {
                    return Ok(WorkflowResponse {
                        message: format!("Invalid choice. Options: {}", options.join(", ")),
                        next_prompt: Some(requirement.description.clone()),
                        completed: false,
                        result: None,
                    });
                }
                json!(input)
            }
            ContextType::Number { min, max } => {
                let num: f64 = input.parse()
                    .map_err(|e| anyhow!("Invalid number: {}", e))?;
                
                if let Some(min_val) = min {
                    if num < *min_val {
                        return Err(anyhow!("Number {} must be at least {} (minimum value constraint)", num, min_val));
                    }
                }
                if let Some(max_val) = max {
                    if num > *max_val {
                        return Err(anyhow!("Number {} must be at most {} (maximum value constraint)", num, max_val));
                    }
                }
                
                json!(num)
            }
            _ => json!(input),
        };
        
        collected_context.insert(requirement.name.clone(), value);
        *current_index += 1;
        
        // Get next prompt
        let next_prompt = if *current_index < required_context.len() {
            Some(format!(
                "{}\n({})",
                required_context[*current_index].description,
                self.format_context_type(&required_context[*current_index].context_type)
            ))
        } else {
            None
        };
        
        Ok(WorkflowResponse {
            message: format!("'{}' recorded.", requirement.name),
            next_prompt,
            completed: false,
            result: None,
        })
    }
    
    /// Cancel current workflow
    fn cancel_workflow(&mut self) {
        self.current_state = ExtendedWorkflowState::None;
    }
    
    /// Format current prompt based on state
    fn format_current_prompt(&self) -> String {
        match &self.current_state {
            ExtendedWorkflowState::None => "No active workflow.".to_string(),
            ExtendedWorkflowState::ToolExecution { required_params, step, .. } => {
                match step {
                    ToolExecutionStep::CollectingParameters { current_param } => {
                        if let Some(param) = required_params.iter().find(|p| p.name == *current_param) {
                            format!("{} ({}): {}", param.name, param.param_type, param.description)
                        } else {
                            "Collecting parameters...".to_string()
                        }
                    }
                    _ => "Processing tool execution...".to_string(),
                }
            }
            ExtendedWorkflowState::MultiStepTask { current_step, steps, accumulated_results, .. } => {
                if *current_step < steps.len() {
                    self.format_step_prompt(&steps[*current_step], accumulated_results)
                } else {
                    "Task completing...".to_string()
                }
            }
            _ => "Processing...".to_string(),
        }
    }
    
    /// Format prompt for a task step
    fn format_step_prompt(&self, step: &TaskStep, _context: &HashMap<String, Value>) -> String {
        match &step.step_type {
            StepType::UserInput { prompt } => prompt.clone(),
            StepType::Decision { options } => {
                format!("{}\nOptions: {}", step.description, options.join(", "))
            }
            _ => step.description.clone(),
        }
    }
    
    /// Format context type for display
    fn format_context_type(&self, context_type: &ContextType) -> String {
        match context_type {
            ContextType::File => "file path".to_string(),
            ContextType::Directory => "directory path".to_string(),
            ContextType::CodeSnippet => "code snippet".to_string(),
            ContextType::Description => "text description".to_string(),
            ContextType::Choice { options } => format!("choose from: {}", options.join(", ")),
            ContextType::Number { min, max } => {
                match (min, max) {
                    (Some(min), Some(max)) => format!("number between {} and {}", min, max),
                    (Some(min), None) => format!("number >= {}", min),
                    (None, Some(max)) => format!("number <= {}", max),
                    (None, None) => "number".to_string(),
                }
            }
        }
    }
    
    /// Parse parameter value based on type
    fn parse_parameter_value(&self, input: &str, param_type: &str) -> Result<Value> {
        match param_type {
            "string" => Ok(json!(input)),
            "number" | "integer" => {
                let num: f64 = input.parse()
                    .map_err(|e| anyhow!("Invalid number: {}", e))?;
                Ok(json!(num))
            }
            "boolean" => {
                let val = input.eq_ignore_ascii_case("true") || 
                         input.eq_ignore_ascii_case("yes") ||
                         input.eq_ignore_ascii_case("1");
                Ok(json!(val))
            }
            "json" => {
                serde_json::from_str(input)
                    .map_err(|e| anyhow!("Invalid JSON: {}", e))
            }
            _ => Ok(json!(input)),
        }
    }
    
    /// Validate parameter value
    fn validate_parameter(&self, value: &Value, validation: &ParameterValidation) -> Result<()> {
        if let Some(pattern) = &validation.pattern {
            if let Some(str_val) = value.as_str() {
                let re = regex::Regex::new(pattern)?;
                if !re.is_match(str_val) {
                    return Err(anyhow!("Value '{}' does not match required pattern '{}'", value, pattern.as_str()));
                }
            }
        }
        
        if let Some(valid_values) = &validation.valid_values {
            if let Some(str_val) = value.as_str() {
                if !valid_values.contains(&str_val.to_string()) {
                    return Err(anyhow!("Invalid value. Must be one of: {:?}", valid_values));
                }
            }
        }
        
        Ok(())
    }
    
    /// List available workflow templates
    pub fn list_templates(&self) -> Vec<&WorkflowTemplate> {
        self.templates.values().collect()
    }
}

/// Response from workflow processing
#[derive(Debug, Clone)]
pub struct WorkflowResponse {
    /// Message to display to user
    pub message: String,
    
    /// Next prompt to show (if any)
    pub next_prompt: Option<String>,
    
    /// Whether the workflow is completed
    pub completed: bool,
    
    /// Result data (for execution)
    pub result: Option<Value>,
}

impl Default for WorkflowManager {
    fn default() -> Self {
        Self::new()
    }
}