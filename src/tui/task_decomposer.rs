//! Task Decomposer
//! 
//! This module provides intelligent task decomposition capabilities,
//! breaking complex tasks into manageable subtasks that can be delegated
//! to different agents.

use std::collections::HashMap;
use std::time::Duration;

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::tools::task_management::TaskPriority;
use super::nlp::core::orchestrator::{ExtractedTask, ExtractedTaskType};

/// Task decomposition result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecomposedTask {
    /// The original task
    pub parent_task: ExtractedTask,
    
    /// Decomposed subtasks
    pub subtasks: Vec<Subtask>,
    
    /// Execution strategy
    pub execution_strategy: ExecutionStrategy,
    
    /// Task dependencies graph
    pub dependencies: TaskDependencyGraph,
    
    /// Estimated total effort
    pub total_effort: Duration,
    
    /// Complexity score (0.0 to 1.0)
    pub complexity_score: f64,
}

/// Individual subtask
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Subtask {
    pub id: String,
    pub description: String,
    pub task_type: ExtractedTaskType,
    pub priority: TaskPriority,
    pub estimated_effort: Duration,
    pub required_capabilities: Vec<String>,
    pub preferred_agent_type: AgentType,
    pub dependencies: Vec<String>, // IDs of other subtasks
    pub parallel_group: Option<usize>, // Tasks in same group can run in parallel
}

/// Agent types for task delegation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AgentType {
    CodeGeneration,
    CodeAnalysis,
    Documentation,
    Testing,
    Research,
    Design,
    DataProcessing,
    GeneralPurpose,
}

/// Execution strategy for the decomposed task
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionStrategy {
    /// Execute all subtasks sequentially
    Sequential,
    
    /// Execute subtasks in parallel where possible
    Parallel { max_concurrent: usize },
    
    /// Execute in phases (groups of parallel tasks)
    Phased { phases: Vec<ExecutionPhase> },
    
    /// Pipeline execution (output of one feeds into next)
    Pipeline,
}

/// Execution phase for phased strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionPhase {
    pub phase_number: usize,
    pub subtask_ids: Vec<String>,
    pub description: String,
}

/// Task dependency graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskDependencyGraph {
    /// Adjacency list representation
    pub edges: HashMap<String, Vec<String>>,
    
    /// Topological ordering of tasks
    pub topological_order: Vec<String>,
}

/// Task decomposer
pub struct TaskDecomposer {
    /// Decomposition strategies
    strategies: Vec<Box<dyn DecompositionStrategy>>,
}

/// Trait for decomposition strategies
trait DecompositionStrategy: Send + Sync {
    /// Check if this strategy applies to the task
    fn applies_to(&self, task: &ExtractedTask) -> bool;
    
    /// Decompose the task
    fn decompose(&self, task: &ExtractedTask) -> Result<Vec<Subtask>>;
    
    /// Get strategy name
    fn name(&self) -> &str;
}

impl TaskDecomposer {
    pub fn new() -> Self {
        Self {
            strategies: vec![
                Box::new(CodeImplementationStrategy),
                Box::new(ResearchStrategy),
                Box::new(DocumentationStrategy),
                Box::new(TestingStrategy),
                Box::new(DesignStrategy),
                Box::new(GeneralStrategy),
            ],
        }
    }
    
    /// Decompose a task into subtasks
    pub fn decompose_task(&self, task: &ExtractedTask) -> Result<DecomposedTask> {
        info!("Decomposing task: {}", task.description);
        
        // Find applicable strategy
        let strategy = self.strategies
            .iter()
            .find(|s| s.applies_to(task))
            .ok_or_else(|| anyhow!("No decomposition strategy found for task"))?;
            
        info!("Using decomposition strategy: {}", strategy.name());
        
        // Decompose into subtasks
        let mut subtasks = strategy.decompose(task)?;
        
        // Assign IDs to subtasks
        for (i, subtask) in subtasks.iter_mut().enumerate() {
            subtask.id = format!("subtask_{}", i + 1);
        }
        
        // Build dependency graph
        let dependencies = self.build_dependency_graph(&subtasks)?;
        
        // Determine execution strategy
        let execution_strategy = self.determine_execution_strategy(&subtasks, &dependencies)?;
        
        // Calculate total effort
        let total_effort = subtasks
            .iter()
            .map(|s| s.estimated_effort)
            .fold(Duration::from_secs(0), |acc, d| acc + d);
            
        // Calculate complexity score
        let complexity_score = self.calculate_complexity_score(&subtasks, &dependencies);
        
        Ok(DecomposedTask {
            parent_task: task.clone(),
            subtasks,
            execution_strategy,
            dependencies,
            total_effort,
            complexity_score,
        })
    }
    
    /// Build dependency graph from subtasks
    fn build_dependency_graph(&self, subtasks: &[Subtask]) -> Result<TaskDependencyGraph> {
        let mut edges = HashMap::new();
        
        // Build adjacency list
        for subtask in subtasks {
            edges.insert(subtask.id.clone(), subtask.dependencies.clone());
        }
        
        // Perform topological sort
        let topological_order = self.topological_sort(&edges)?;
        
        Ok(TaskDependencyGraph {
            edges,
            topological_order,
        })
    }
    
    /// Topological sort using Kahn's algorithm
    fn topological_sort(&self, edges: &HashMap<String, Vec<String>>) -> Result<Vec<String>> {
        let mut in_degree: HashMap<String, usize> = HashMap::new();
        let mut queue = Vec::new();
        let mut result = Vec::new();
        
        // Calculate in-degrees
        for (node, _) in edges {
            in_degree.entry(node.clone()).or_insert(0);
        }
        
        for (_, deps) in edges {
            for dep in deps {
                *in_degree.entry(dep.clone()).or_insert(0) += 1;
            }
        }
        
        // Find nodes with no dependencies
        for (node, &degree) in &in_degree {
            if degree == 0 {
                queue.push(node.clone());
            }
        }
        
        // Process queue
        while let Some(node) = queue.pop() {
            result.push(node.clone());
            
            if let Some(deps) = edges.get(&node) {
                for dep in deps {
                    if let Some(degree) = in_degree.get_mut(dep) {
                        *degree -= 1;
                        if *degree == 0 {
                            queue.push(dep.clone());
                        }
                    }
                }
            }
        }
        
        // Check for cycles
        if result.len() != edges.len() {
            return Err(anyhow!("Circular dependency detected in task graph"));
        }
        
        Ok(result)
    }
    
    /// Determine execution strategy based on dependencies
    fn determine_execution_strategy(
        &self,
        subtasks: &[Subtask],
        dependencies: &TaskDependencyGraph,
    ) -> Result<ExecutionStrategy> {
        // Group tasks by parallel execution capability
        let mut phases = Vec::new();
        let mut processed = std::collections::HashSet::new();
        
        for task_id in &dependencies.topological_order {
            if processed.contains(task_id) {
                continue;
            }
            
            // Find all tasks that can run in parallel with this one
            let mut phase_tasks = vec![task_id.clone()];
            processed.insert(task_id.clone());
            
            for other_id in &dependencies.topological_order {
                if processed.contains(other_id) {
                    continue;
                }
                
                // Check if they have mutual dependencies
                let has_dependency = dependencies.edges.get(task_id)
                    .map(|deps| deps.contains(other_id))
                    .unwrap_or(false)
                    || dependencies.edges.get(other_id)
                    .map(|deps| deps.contains(task_id))
                    .unwrap_or(false);
                    
                if !has_dependency {
                    phase_tasks.push(other_id.clone());
                    processed.insert(other_id.clone());
                }
            }
            
            phases.push(ExecutionPhase {
                phase_number: phases.len() + 1,
                subtask_ids: phase_tasks,
                description: format!("Phase {}", phases.len() + 1),
            });
        }
        
        // Choose strategy based on phases
        if phases.len() == 1 && phases[0].subtask_ids.len() > 1 {
            Ok(ExecutionStrategy::Parallel { 
                max_concurrent: phases[0].subtask_ids.len().min(4) 
            })
        } else if phases.len() > 1 {
            Ok(ExecutionStrategy::Phased { phases })
        } else {
            Ok(ExecutionStrategy::Sequential)
        }
    }
    
    /// Calculate task complexity score
    fn calculate_complexity_score(
        &self,
        subtasks: &[Subtask],
        dependencies: &TaskDependencyGraph,
    ) -> f64 {
        let num_subtasks = subtasks.len() as f64;
        let num_dependencies = dependencies.edges.values()
            .map(|deps| deps.len())
            .sum::<usize>() as f64;
        
        let avg_effort = subtasks.iter()
            .map(|s| s.estimated_effort.as_secs() as f64)
            .sum::<f64>() / num_subtasks;
            
        // Normalize to 0-1 range
        let complexity = (num_subtasks / 10.0).min(0.5) +
                        (num_dependencies / 20.0).min(0.3) +
                        (avg_effort / 7200.0).min(0.2); // 2 hours max
                        
        complexity.min(1.0)
    }
}

/// Code implementation decomposition strategy
struct CodeImplementationStrategy;

impl DecompositionStrategy for CodeImplementationStrategy {
    fn applies_to(&self, task: &ExtractedTask) -> bool {
        matches!(task.task_type, ExtractedTaskType::CodingTask)
    }
    
    fn decompose(&self, task: &ExtractedTask) -> Result<Vec<Subtask>> {
        let mut subtasks = Vec::new();
        
        // 1. Design/Architecture
        subtasks.push(Subtask {
            id: String::new(),
            description: format!("Design architecture for: {}", task.description),
            task_type: ExtractedTaskType::DesignTask,
            priority: task.priority.clone(),
            estimated_effort: Duration::from_secs(900), // 15 mins
            required_capabilities: vec!["design".to_string(), "architecture".to_string()],
            preferred_agent_type: AgentType::Design,
            dependencies: vec![],
            parallel_group: Some(0),
        });
        
        // 2. Implementation
        subtasks.push(Subtask {
            id: String::new(),
            description: format!("Implement: {}", task.description),
            task_type: ExtractedTaskType::CodingTask,
            priority: task.priority.clone(),
            estimated_effort: task.estimated_effort.unwrap_or(Duration::from_secs(1800)),
            required_capabilities: vec!["coding".to_string(), "implementation".to_string()],
            preferred_agent_type: AgentType::CodeGeneration,
            dependencies: vec!["subtask_1".to_string()],
            parallel_group: Some(1),
        });
        
        // 3. Testing
        subtasks.push(Subtask {
            id: String::new(),
            description: format!("Write tests for: {}", task.description),
            task_type: ExtractedTaskType::TestingTask,
            priority: task.priority.clone(),
            estimated_effort: Duration::from_secs(1200), // 20 mins
            required_capabilities: vec!["testing".to_string()],
            preferred_agent_type: AgentType::Testing,
            dependencies: vec!["subtask_2".to_string()],
            parallel_group: Some(2),
        });
        
        // 4. Documentation
        subtasks.push(Subtask {
            id: String::new(),
            description: format!("Document: {}", task.description),
            task_type: ExtractedTaskType::DocumentationTask,
            priority: TaskPriority::Medium,
            estimated_effort: Duration::from_secs(600), // 10 mins
            required_capabilities: vec!["documentation".to_string()],
            preferred_agent_type: AgentType::Documentation,
            dependencies: vec!["subtask_2".to_string()],
            parallel_group: Some(2),
        });
        
        Ok(subtasks)
    }
    
    fn name(&self) -> &str {
        "CodeImplementationStrategy"
    }
}

/// Research task decomposition strategy
struct ResearchStrategy;

impl DecompositionStrategy for ResearchStrategy {
    fn applies_to(&self, task: &ExtractedTask) -> bool {
        matches!(task.task_type, ExtractedTaskType::ResearchTask)
    }
    
    fn decompose(&self, task: &ExtractedTask) -> Result<Vec<Subtask>> {
        let mut subtasks = Vec::new();
        
        // 1. Information gathering
        subtasks.push(Subtask {
            id: String::new(),
            description: format!("Gather information about: {}", task.description),
            task_type: ExtractedTaskType::ResearchTask,
            priority: task.priority.clone(),
            estimated_effort: Duration::from_secs(900),
            required_capabilities: vec!["research".to_string(), "web_search".to_string()],
            preferred_agent_type: AgentType::Research,
            dependencies: vec![],
            parallel_group: Some(0),
        });
        
        // 2. Analysis
        subtasks.push(Subtask {
            id: String::new(),
            description: format!("Analyze findings for: {}", task.description),
            task_type: ExtractedTaskType::GeneralTask,
            priority: task.priority.clone(),
            estimated_effort: Duration::from_secs(600),
            required_capabilities: vec!["analysis".to_string()],
            preferred_agent_type: AgentType::DataProcessing,
            dependencies: vec!["subtask_1".to_string()],
            parallel_group: Some(1),
        });
        
        // 3. Summary/Report
        subtasks.push(Subtask {
            id: String::new(),
            description: format!("Create summary report for: {}", task.description),
            task_type: ExtractedTaskType::DocumentationTask,
            priority: task.priority.clone(),
            estimated_effort: Duration::from_secs(600),
            required_capabilities: vec!["writing".to_string()],
            preferred_agent_type: AgentType::Documentation,
            dependencies: vec!["subtask_2".to_string()],
            parallel_group: Some(2),
        });
        
        Ok(subtasks)
    }
    
    fn name(&self) -> &str {
        "ResearchStrategy"
    }
}

/// Documentation task decomposition strategy
struct DocumentationStrategy;

impl DecompositionStrategy for DocumentationStrategy {
    fn applies_to(&self, task: &ExtractedTask) -> bool {
        matches!(task.task_type, ExtractedTaskType::DocumentationTask)
    }
    
    fn decompose(&self, task: &ExtractedTask) -> Result<Vec<Subtask>> {
        // Simple documentation tasks don't need decomposition
        if task.estimated_effort.unwrap_or(Duration::from_secs(0)).as_secs() < 1800 {
            return Ok(vec![]);
        }
        
        let mut subtasks = Vec::new();
        
        // 1. Outline creation
        subtasks.push(Subtask {
            id: String::new(),
            description: format!("Create outline for: {}", task.description),
            task_type: ExtractedTaskType::DesignTask,
            priority: task.priority.clone(),
            estimated_effort: Duration::from_secs(600),
            required_capabilities: vec!["planning".to_string()],
            preferred_agent_type: AgentType::Design,
            dependencies: vec![],
            parallel_group: Some(0),
        });
        
        // 2. Content writing
        subtasks.push(Subtask {
            id: String::new(),
            description: format!("Write content for: {}", task.description),
            task_type: ExtractedTaskType::DocumentationTask,
            priority: task.priority.clone(),
            estimated_effort: Duration::from_secs(1200),
            required_capabilities: vec!["writing".to_string()],
            preferred_agent_type: AgentType::Documentation,
            dependencies: vec!["subtask_1".to_string()],
            parallel_group: Some(1),
        });
        
        // 3. Review and polish
        subtasks.push(Subtask {
            id: String::new(),
            description: format!("Review and polish: {}", task.description),
            task_type: ExtractedTaskType::DocumentationTask,
            priority: task.priority.clone(),
            estimated_effort: Duration::from_secs(600),
            required_capabilities: vec!["editing".to_string()],
            preferred_agent_type: AgentType::Documentation,
            dependencies: vec!["subtask_2".to_string()],
            parallel_group: Some(2),
        });
        
        Ok(subtasks)
    }
    
    fn name(&self) -> &str {
        "DocumentationStrategy"
    }
}

/// Testing task decomposition strategy
struct TestingStrategy;

impl DecompositionStrategy for TestingStrategy {
    fn applies_to(&self, task: &ExtractedTask) -> bool {
        matches!(task.task_type, ExtractedTaskType::TestingTask)
    }
    
    fn decompose(&self, task: &ExtractedTask) -> Result<Vec<Subtask>> {
        let mut subtasks = Vec::new();
        
        // 1. Test plan
        subtasks.push(Subtask {
            id: String::new(),
            description: format!("Create test plan for: {}", task.description),
            task_type: ExtractedTaskType::DesignTask,
            priority: task.priority.clone(),
            estimated_effort: Duration::from_secs(600),
            required_capabilities: vec!["test_planning".to_string()],
            preferred_agent_type: AgentType::Testing,
            dependencies: vec![],
            parallel_group: Some(0),
        });
        
        // 2. Test implementation
        subtasks.push(Subtask {
            id: String::new(),
            description: format!("Implement tests for: {}", task.description),
            task_type: ExtractedTaskType::TestingTask,
            priority: task.priority.clone(),
            estimated_effort: Duration::from_secs(1200),
            required_capabilities: vec!["test_implementation".to_string()],
            preferred_agent_type: AgentType::Testing,
            dependencies: vec!["subtask_1".to_string()],
            parallel_group: Some(1),
        });
        
        // 3. Test execution and reporting
        subtasks.push(Subtask {
            id: String::new(),
            description: format!("Execute tests and report results for: {}", task.description),
            task_type: ExtractedTaskType::TestingTask,
            priority: task.priority.clone(),
            estimated_effort: Duration::from_secs(600),
            required_capabilities: vec!["test_execution".to_string()],
            preferred_agent_type: AgentType::Testing,
            dependencies: vec!["subtask_2".to_string()],
            parallel_group: Some(2),
        });
        
        Ok(subtasks)
    }
    
    fn name(&self) -> &str {
        "TestingStrategy"
    }
}

/// Design task decomposition strategy
struct DesignStrategy;

impl DecompositionStrategy for DesignStrategy {
    fn applies_to(&self, task: &ExtractedTask) -> bool {
        matches!(task.task_type, ExtractedTaskType::DesignTask)
    }
    
    fn decompose(&self, task: &ExtractedTask) -> Result<Vec<Subtask>> {
        // Simple design tasks don't need decomposition
        if task.estimated_effort.unwrap_or(Duration::from_secs(0)).as_secs() < 2400 {
            return Ok(vec![]);
        }
        
        let mut subtasks = Vec::new();
        
        // 1. Requirements analysis
        subtasks.push(Subtask {
            id: String::new(),
            description: format!("Analyze requirements for: {}", task.description),
            task_type: ExtractedTaskType::ResearchTask,
            priority: task.priority.clone(),
            estimated_effort: Duration::from_secs(900),
            required_capabilities: vec!["analysis".to_string()],
            preferred_agent_type: AgentType::Research,
            dependencies: vec![],
            parallel_group: Some(0),
        });
        
        // 2. Design creation
        subtasks.push(Subtask {
            id: String::new(),
            description: format!("Create design for: {}", task.description),
            task_type: ExtractedTaskType::DesignTask,
            priority: task.priority.clone(),
            estimated_effort: Duration::from_secs(1800),
            required_capabilities: vec!["design".to_string()],
            preferred_agent_type: AgentType::Design,
            dependencies: vec!["subtask_1".to_string()],
            parallel_group: Some(1),
        });
        
        // 3. Design documentation
        subtasks.push(Subtask {
            id: String::new(),
            description: format!("Document design decisions for: {}", task.description),
            task_type: ExtractedTaskType::DocumentationTask,
            priority: task.priority.clone(),
            estimated_effort: Duration::from_secs(600),
            required_capabilities: vec!["documentation".to_string()],
            preferred_agent_type: AgentType::Documentation,
            dependencies: vec!["subtask_2".to_string()],
            parallel_group: Some(2),
        });
        
        Ok(subtasks)
    }
    
    fn name(&self) -> &str {
        "DesignStrategy"
    }
}

/// General task decomposition strategy (fallback)
struct GeneralStrategy;

impl DecompositionStrategy for GeneralStrategy {
    fn applies_to(&self, task: &ExtractedTask) -> bool {
        true // Always applies as fallback
    }
    
    fn decompose(&self, task: &ExtractedTask) -> Result<Vec<Subtask>> {
        // Only decompose if task is complex enough
        if task.estimated_effort.unwrap_or(Duration::from_secs(0)).as_secs() < 2400 {
            return Ok(vec![]);
        }
        
        let mut subtasks = Vec::new();
        
        // Generic decomposition into 3 phases
        subtasks.push(Subtask {
            id: String::new(),
            description: format!("Analyze and plan: {}", task.description),
            task_type: ExtractedTaskType::GeneralTask,
            priority: task.priority.clone(),
            estimated_effort: Duration::from_secs(900),
            required_capabilities: vec!["analysis".to_string()],
            preferred_agent_type: AgentType::GeneralPurpose,
            dependencies: vec![],
            parallel_group: Some(0),
        });
        
        subtasks.push(Subtask {
            id: String::new(),
            description: format!("Execute main work for: {}", task.description),
            task_type: task.task_type.clone(),
            priority: task.priority.clone(),
            estimated_effort: Duration::from_secs(1800),
            required_capabilities: vec!["execution".to_string()],
            preferred_agent_type: AgentType::GeneralPurpose,
            dependencies: vec!["subtask_1".to_string()],
            parallel_group: Some(1),
        });
        
        subtasks.push(Subtask {
            id: String::new(),
            description: format!("Review and finalize: {}", task.description),
            task_type: ExtractedTaskType::GeneralTask,
            priority: task.priority.clone(),
            estimated_effort: Duration::from_secs(600),
            required_capabilities: vec!["review".to_string()],
            preferred_agent_type: AgentType::GeneralPurpose,
            dependencies: vec!["subtask_2".to_string()],
            parallel_group: Some(2),
        });
        
        Ok(subtasks)
    }
    
    fn name(&self) -> &str {
        "GeneralStrategy"
    }
}