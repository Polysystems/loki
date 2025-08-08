//! Intelligent task mapping from stories

use super::types::*;
use anyhow::Result;
use std::collections::HashMap;
use tracing::debug;

/// Task mapper for converting between story contexts and actionable tasks
#[derive(Debug, Clone)]
pub struct TaskMapper {
    /// Task extraction patterns
    patterns: Vec<TaskPattern>,
}

/// Pattern for extracting tasks from plot points
#[derive(Debug, Clone)]
struct TaskPattern {
    plot_type_filter: fn(&PlotType) -> bool,
    extractor: fn(&PlotPoint) -> Option<MappedTask>,
}

impl TaskMapper {
    /// Create a new task mapper
    pub fn new() -> Self {
        let patterns = vec![
            // Extract tasks from Task plot points
            TaskPattern {
                plot_type_filter: |pt| matches!(pt, PlotType::Task { .. }),
                extractor: |pp| {
                    if let PlotType::Task { description, completed } = &pp.plot_type {
                        Some(MappedTask {
                            id: format!("task-{}", pp.id.0),
                            description: description.clone(),
                            story_context: pp.description.clone(),
                            status: if *completed {
                                TaskStatus::Completed
                            } else {
                                TaskStatus::Pending
                            },
                            assigned_to: None,
                            created_at: pp.timestamp,
                            updated_at: pp.timestamp,
                            plot_point: Some(pp.id),
                        })
                    } else {
                        None
                    }
                },
            },
            // Extract tasks from unresolved issues
            TaskPattern {
                plot_type_filter: |pt| matches!(pt, PlotType::Issue { resolved: false, .. }),
                extractor: |pp| {
                    if let PlotType::Issue { error, resolved: false } = &pp.plot_type {
                        Some(MappedTask {
                            id: format!("issue-{}", pp.id.0),
                            description: format!("Fix: {}", error),
                            story_context: pp.description.clone(),
                            status: TaskStatus::Pending,
                            assigned_to: None,
                            created_at: pp.timestamp,
                            updated_at: pp.timestamp,
                            plot_point: Some(pp.id),
                        })
                    } else {
                        None
                    }
                },
            },
            // Extract tasks from goals
            TaskPattern {
                plot_type_filter: |pt| matches!(pt, PlotType::Goal { .. }),
                extractor: |pp| {
                    if let PlotType::Goal { objective } = &pp.plot_type {
                        Some(MappedTask {
                            id: format!("goal-{}", pp.id.0),
                            description: format!("Achieve: {}", objective),
                            story_context: pp.description.clone(),
                            status: TaskStatus::Pending,
                            assigned_to: None,
                            created_at: pp.timestamp,
                            updated_at: pp.timestamp,
                            plot_point: Some(pp.id),
                        })
                    } else {
                        None
                    }
                },
            },
        ];

        Self { patterns }
    }

    /// Map a story to tasks
    pub async fn map_story_to_tasks(&self, story: &Story) -> Result<TaskMap> {
        let mut tasks = Vec::new();
        let mut dependencies = HashMap::new();

        // Extract all plot points
        let mut all_plot_points = Vec::new();
        for arc in &story.arcs {
            all_plot_points.extend(&arc.plot_points);
        }

        // Apply patterns to extract tasks
        for plot_point in &all_plot_points {
            for pattern in &self.patterns {
                if (pattern.plot_type_filter)(&plot_point.plot_type) {
                    if let Some(task) = (pattern.extractor)(plot_point) {
                        tasks.push(task);
                    }
                }
            }
        }

        // Infer dependencies based on temporal ordering
        self.infer_dependencies(&mut tasks, &mut dependencies);

        // Calculate completion percentage
        let completed = tasks.iter().filter(|t| t.status == TaskStatus::Completed).count();
        let total = tasks.len();
        let completion_percentage = if total > 0 {
            (completed as f32 / total as f32) * 100.0
        } else {
            0.0
        };

        debug!(
            "Mapped {} tasks from story {} ({:.1}% complete)",
            tasks.len(),
            story.id.0,
            completion_percentage
        );

        Ok(TaskMap {
            story_id: story.id,
            tasks,
            dependencies,
            completion_percentage,
        })
    }

    /// Extract tasks from a specific arc
    pub async fn map_arc_to_tasks(&self, arc: &StoryArc) -> Result<Vec<MappedTask>> {
        let mut tasks = Vec::new();

        for plot_point in &arc.plot_points {
            for pattern in &self.patterns {
                if (pattern.plot_type_filter)(&plot_point.plot_type) {
                    if let Some(task) = (pattern.extractor)(plot_point) {
                        tasks.push(task);
                    }
                }
            }
        }

        Ok(tasks)
    }

    /// Update task status based on plot points
    pub async fn update_task_from_plot_point(
        &self,
        task_map: &mut TaskMap,
        plot_point: &PlotPoint,
    ) -> Result<()> {
        // Check if this plot point resolves any tasks
        match &plot_point.plot_type {
            PlotType::Task { completed: true, .. } => {
                // Mark corresponding task as completed
                if let Some(task) = task_map.tasks.iter_mut()
                    .find(|t| t.plot_point == Some(plot_point.id))
                {
                    task.status = TaskStatus::Completed;
                    task.updated_at = chrono::Utc::now();
                }
            }
            PlotType::Issue { resolved: true, .. } => {
                // Mark corresponding issue task as completed
                if let Some(task) = task_map.tasks.iter_mut()
                    .find(|t| t.plot_point == Some(plot_point.id))
                {
                    task.status = TaskStatus::Completed;
                    task.updated_at = chrono::Utc::now();
                }
            }
            _ => {}
        }

        // Recalculate completion percentage
        let completed = task_map.tasks.iter()
            .filter(|t| t.status == TaskStatus::Completed)
            .count();
        let total = task_map.tasks.len();
        task_map.completion_percentage = if total > 0 {
            (completed as f32 / total as f32) * 100.0
        } else {
            0.0
        };

        Ok(())
    }

    /// Infer task dependencies
    fn infer_dependencies(
        &self,
        tasks: &mut [MappedTask],
        dependencies: &mut HashMap<String, Vec<String>>,
    ) {
        // Sort tasks by creation time
        tasks.sort_by_key(|t| t.created_at);

        // Simple heuristic: tasks created later might depend on earlier unfinished tasks
        for i in 0..tasks.len() {
            if tasks[i].status != TaskStatus::Completed {
                let mut task_deps = Vec::new();

                // Look for earlier uncompleted tasks
                for j in 0..i {
                    if tasks[j].status != TaskStatus::Completed {
                        // Check if descriptions suggest dependency
                        if self.suggests_dependency(&tasks[j].description, &tasks[i].description) {
                            task_deps.push(tasks[j].id.clone());
                        }
                    }
                }

                if !task_deps.is_empty() {
                    dependencies.insert(tasks[i].id.clone(), task_deps);
                }
            }
        }
    }

    /// Check if task descriptions suggest dependency
    fn suggests_dependency(&self, earlier: &str, later: &str) -> bool {
        let earlier_lower = earlier.to_lowercase();
        let later_lower = later.to_lowercase();

        // Simple keyword matching
        let dependency_keywords = [
            ("implement", "test"),
            ("create", "use"),
            ("define", "implement"),
            ("design", "build"),
            ("setup", "configure"),
            ("install", "use"),
        ];

        for (first, second) in dependency_keywords {
            if earlier_lower.contains(first) && later_lower.contains(second) {
                return true;
            }
        }

        false
    }
}

/// Task grouping utilities
impl TaskMapper {
    /// Group tasks by status
    pub fn group_by_status(task_map: &TaskMap) -> HashMap<TaskStatus, Vec<&MappedTask>> {
        let mut groups: HashMap<TaskStatus, Vec<&MappedTask>> = HashMap::new();

        for task in &task_map.tasks {
            groups.entry(task.status.clone())
                .or_insert_with(Vec::new)
                .push(task);
        }

        groups
    }

    /// Find blocked tasks
    pub fn find_blocked_tasks(task_map: &TaskMap) -> Vec<&MappedTask> {
        task_map.tasks.iter()
            .filter(|t| t.status == TaskStatus::Blocked)
            .collect()
    }

    /// Get task timeline
    pub fn get_task_timeline(task_map: &TaskMap) -> Vec<(chrono::DateTime<chrono::Utc>, &MappedTask)> {
        let mut timeline: Vec<_> = task_map.tasks.iter()
            .map(|t| (t.created_at, t))
            .collect();

        timeline.sort_by_key(|(time, _)| *time);
        timeline
    }
}
