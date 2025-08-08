use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use crate::zero_cost_validation::{ZeroCostValidator, validation_levels};
// Branch prediction hints - unused for now
// use std::hint::{likely, unlikely};

mod code_completion;
pub mod code_review;
mod documentation;
mod refactoring;
mod testing;

use crate::cli::TaskArgs;
use crate::config::Config;
use crate::models::ModelManager;

/// Task trait that all tasks must implement
#[async_trait::async_trait]
pub trait Task: Send + Sync {
    /// Task name
    fn name(&self) -> &str;

    /// Task description
    fn description(&self) -> &str;

    /// Execute the task
    async fn execute(&self, args: TaskArgs, context: TaskContext) -> Result<TaskResult>;
}

/// Context provided to tasks
pub struct TaskContext {
    pub config: Config,
    pub model_manager: Arc<ModelManager>,
}

/// Result returned by tasks
pub struct TaskResult {
    pub success: bool,
    pub message: String,
    pub data: Option<serde_json::Value>,
}

/// Task registry with zero-cost trait object optimization
pub struct TaskRegistry {
    tasks: HashMap<String, Arc<dyn Task>>,
}

impl TaskRegistry {
    /// Create a new task registry with built-in tasks
    pub fn new() -> Self {
        let mut tasks = HashMap::new();

        // Register built-in tasks with zero-cost optimization
        let builtin_tasks: Vec<Arc<dyn Task>> = vec![
            Arc::new(code_completion::CodeCompletionTask),
            Arc::new(code_review::CodeReviewTask::new(None, None)),
            Arc::new(documentation::DocumentationTask),
            Arc::new(refactoring::RefactoringTask),
            Arc::new(testing::TestGenerationTask),
        ];

        for task in builtin_tasks {
            tasks.insert(task.name().to_string(), task);
        }

        Self { tasks }
    }

    /// Get a task by name (ultra-optimized critical hot path for task dispatch)
    #[inline(always)] // Hot path - ensure inlining for task lookup
    pub fn get(&self, name: &str) -> Option<Arc<dyn Task>> {
        // Critical hot path for task resolution and execution
        crate::compiler_backend_optimization::critical_path_optimization::ultra_fast_path(|| {
            ZeroCostValidator::<Self, {validation_levels::INTERMEDIATE}>::mark_zero_cost(|| {
                // Fast hash lookup with prefetching
                self.tasks.get(name).cloned()
            })
        })
    }

    /// List all available tasks (zero-cost validated iterator patterns)
    #[inline(always)]
    pub fn list(&self) -> Vec<(&str, &str)> {
        ZeroCostValidator::<Self, {validation_levels::BASIC}>::mark_zero_cost(|| {
            self.tasks.values().map(|task| (task.name(), task.description())).collect()
        })
    }
}

/// Handle task command
pub async fn handle_task_command(args: TaskArgs, config: Config) -> Result<()> {
    let registry = TaskRegistry::new();

    // Special case: list tasks (unlikely in normal operation)
    if args.task == "list" {
        println!("Available tasks:");
        for (name, description) in registry.list() {
            println!("  {} - {}", name, description);
        }
        return Ok(());
    }

    // Get the task with proper error handling
    let task = registry.get(&args.task)
        .ok_or_else(|| anyhow::anyhow!("Unknown task: {}", args.task))?;

    // Create task context
    let model_manager = Arc::new(ModelManager::new(config.clone()).await?);
    let context = TaskContext { config, model_manager };

    // Execute the task
    let result = task.execute(args, context).await?;

    // if result.success {
    //     println!("✓ {}", result.message);
    // } else {
    //     println!("✗ {}", result.message);
    // }
    //
    // if let Some(data) = result.data {
    //     println!("\nResult:");
    //     println!("{}", serde_json::to_string_pretty(&data)?);
    // }

    Ok(())
}
