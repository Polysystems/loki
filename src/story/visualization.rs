//! Story visualization and analytics

use super::types::*;
use super::engine::StoryEngine;
use anyhow::Result;
use chrono::{DateTime, Duration, Utc, Timelike};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Story analytics and visualization data
#[derive(Debug, Clone)]
pub struct StoryAnalytics {
    story_engine: Arc<StoryEngine>,
}

/// Story statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoryStatistics {
    pub total_stories: usize,
    pub stories_by_type: HashMap<String, usize>,
    pub total_plot_points: usize,
    pub plot_points_by_type: HashMap<String, usize>,
    pub active_arcs: usize,
    pub completed_arcs: usize,
    pub average_arc_duration: Option<Duration>,
    pub task_completion_rate: f32,
    pub most_active_stories: Vec<(StoryId, String, usize)>,
    pub recent_activity: Vec<ActivityEvent>,
}

/// Activity event for timeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivityEvent {
    pub timestamp: DateTime<Utc>,
    pub story_id: StoryId,
    pub story_title: String,
    pub event_type: String,
    pub description: String,
}

/// Story timeline visualization
#[derive(Debug, Clone, Serialize)]
pub struct StoryTimeline {
    pub story_id: StoryId,
    pub title: String,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub arcs: Vec<ArcTimeline>,
    pub milestones: Vec<Milestone>,
}

/// Arc timeline
#[derive(Debug, Clone, Serialize)]
pub struct ArcTimeline {
    pub arc_id: StoryArcId,
    pub title: String,
    pub start_time: DateTime<Utc>,
    pub end_time: Option<DateTime<Utc>>,
    pub plot_points: Vec<PlotPointVisualization>,
}

/// Plot point for visualization
#[derive(Debug, Clone, Serialize)]
pub struct PlotPointVisualization {
    pub timestamp: DateTime<Utc>,
    pub plot_type: String,
    pub description: String,
    pub importance: f32,
}

/// Milestone in story
#[derive(Debug, Clone, Serialize)]
pub struct Milestone {
    pub timestamp: DateTime<Utc>,
    pub title: String,
    pub milestone_type: MilestoneType,
}

/// Types of milestones
#[derive(Debug, Clone, Serialize)]
pub enum MilestoneType {
    GoalAchieved,
    MajorDecision,
    IssueResolved,
    ArcCompleted,
}

/// Story graph for relationship visualization
#[derive(Debug, Clone, Serialize)]
pub struct StoryGraph {
    pub nodes: Vec<StoryNode>,
    pub edges: Vec<StoryEdge>,
}

/// Node in story graph
#[derive(Debug, Clone, Serialize)]
pub struct StoryNode {
    pub id: String,
    pub story_id: StoryId,
    pub title: String,
    pub node_type: String,
    pub size: f32, // Based on activity
    pub color: String, // Based on type
}

/// Edge in story graph
#[derive(Debug, Clone, Serialize)]
pub struct StoryEdge {
    pub source: String,
    pub target: String,
    pub edge_type: EdgeType,
    pub weight: f32,
}

/// Types of edges
#[derive(Debug, Clone, Serialize)]
pub enum EdgeType {
    Dependency,
    Reference,
    Interaction,
    Synchronization,
}

/// Heatmap data for activity visualization
#[derive(Debug, Clone, Serialize)]
pub struct ActivityHeatmap {
    pub days: Vec<String>, // Date strings
    pub hours: Vec<usize>, // 0-23
    pub data: Vec<Vec<usize>>, // Activity counts
}

impl StoryAnalytics {
    /// Create new analytics instance
    pub fn new(story_engine: Arc<StoryEngine>) -> Self {
        Self { story_engine }
    }
    
    /// Get overall story statistics
    pub async fn get_statistics(&self) -> Result<StoryStatistics> {
        let all_stories = self.story_engine.get_stories_by_type(|_| true);
        
        let mut stories_by_type: HashMap<String, usize> = HashMap::new();
        let mut plot_points_by_type: HashMap<String, usize> = HashMap::new();
        let mut total_plot_points = 0;
        let mut active_arcs = 0;
        let mut completed_arcs = 0;
        let mut arc_durations = Vec::new();
        let mut story_activity: Vec<(StoryId, String, usize)> = Vec::new();
        let mut recent_events = Vec::new();
        
        // Analyze each story
        for story in &all_stories {
            // Count by type
            let type_name = match &story.story_type {
                StoryType::Codebase { .. } => "Codebase",
                StoryType::Directory { .. } => "Directory",
                StoryType::File { .. } => "File",
                StoryType::Agent { .. } => "Agent",
                StoryType::Task { .. } => "Task",
                StoryType::System { .. } => "System",
                StoryType::Bug { .. } => "Bug",
                StoryType::Feature { .. } => "Feature",
                StoryType::Epic { .. } => "Epic",
                StoryType::Learning { .. } => "Learning",
                StoryType::Performance { .. } => "Performance",
                StoryType::Security { .. } => "Security",
                StoryType::Documentation { .. } => "Documentation",
                StoryType::Testing { .. } => "Testing",
                StoryType::Refactoring { .. } => "Refactoring",
                StoryType::Dependencies { .. } => "Dependencies",
                StoryType::Deployment { .. } => "Deployment",
                StoryType::Research { .. } => "Research",
            };
            *stories_by_type.entry(type_name.to_string()).or_insert(0) += 1;
            
            // Count plot points and arcs
            let mut story_plot_count = 0;
            for arc in &story.arcs {
                if arc.status == ArcStatus::Completed {
                    completed_arcs += 1;
                    if let Some(end) = arc.completed_at {
                        arc_durations.push(end - arc.started_at);
                    }
                } else if arc.status == ArcStatus::Active {
                    active_arcs += 1;
                }
                
                for plot_point in &arc.plot_points {
                    total_plot_points += 1;
                    story_plot_count += 1;
                    
                    let plot_type_name = self.plot_type_name(&plot_point.plot_type);
                    *plot_points_by_type.entry(plot_type_name.clone()).or_insert(0) += 1;
                    
                    // Add to recent events
                    if recent_events.len() < 20 {
                        recent_events.push(ActivityEvent {
                            timestamp: plot_point.timestamp,
                            story_id: story.id,
                            story_title: story.title.clone(),
                            event_type: plot_type_name,
                            description: plot_point.description.clone(),
                        });
                    }
                }
            }
            
            story_activity.push((story.id, story.title.clone(), story_plot_count));
        }
        
        // Sort by activity
        story_activity.sort_by(|a, b| b.2.cmp(&a.2));
        story_activity.truncate(10);
        
        // Sort recent events by time
        recent_events.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        
        // Calculate task completion rate
        let task_completion_rate = self.calculate_task_completion_rate(&all_stories).await?;
        
        // Calculate average arc duration
        let average_arc_duration = if arc_durations.is_empty() {
            None
        } else {
            let total_seconds: i64 = arc_durations.iter().map(|d| d.num_seconds()).sum();
            Some(Duration::seconds(total_seconds / arc_durations.len() as i64))
        };
        
        Ok(StoryStatistics {
            total_stories: all_stories.len(),
            stories_by_type,
            total_plot_points,
            plot_points_by_type,
            active_arcs,
            completed_arcs,
            average_arc_duration,
            task_completion_rate,
            most_active_stories: story_activity,
            recent_activity: recent_events,
        })
    }
    
    /// Generate timeline for a story
    pub fn generate_timeline(&self, story_id: StoryId) -> Result<StoryTimeline> {
        let story = self.story_engine
            .get_story(&story_id)
            .ok_or_else(|| anyhow::anyhow!("Story not found"))?;
        
        let mut arcs_timeline = Vec::new();
        let mut milestones = Vec::new();
        
        for arc in &story.arcs {
            let plot_points: Vec<_> = arc.plot_points
                .iter()
                .map(|pp| PlotPointVisualization {
                    timestamp: pp.timestamp,
                    plot_type: self.plot_type_name(&pp.plot_type),
                    description: pp.description.clone(),
                    importance: pp.importance,
                })
                .collect();
            
            // Extract milestones
            for pp in &arc.plot_points {
                if let Some(milestone) = self.extract_milestone(pp) {
                    milestones.push(milestone);
                }
            }
            
            arcs_timeline.push(ArcTimeline {
                arc_id: arc.id,
                title: arc.title.clone(),
                start_time: arc.started_at,
                end_time: arc.completed_at,
                plot_points,
            });
        }
        
        let end_time = story.arcs
            .iter()
            .filter_map(|a| a.completed_at)
            .max()
            .unwrap_or(story.updated_at);
        
        Ok(StoryTimeline {
            story_id,
            title: story.title,
            start_time: story.created_at,
            end_time,
            arcs: arcs_timeline,
            milestones,
        })
    }
    
    /// Generate story relationship graph
    pub fn generate_story_graph(&self) -> Result<StoryGraph> {
        let all_stories = self.story_engine.get_stories_by_type(|_| true);
        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        
        // Create nodes
        for story in &all_stories {
            let node_type = match &story.story_type {
                StoryType::Codebase { .. } => "codebase",
                StoryType::Directory { .. } => "directory",
                StoryType::File { .. } => "file",
                StoryType::Agent { .. } => "agent",
                StoryType::Task { .. } => "task",
                StoryType::System { .. } => "system",
                StoryType::Bug { .. } => "bug",
                StoryType::Feature { .. } => "feature",
                StoryType::Epic { .. } => "epic",
                _ => "other",
            };
            
            let activity = story.arcs.iter()
                .map(|a| a.plot_points.len())
                .sum::<usize>() as f32;
            
            nodes.push(StoryNode {
                id: story.id.0.to_string(),
                story_id: story.id,
                title: story.title.clone(),
                node_type: node_type.to_string(),
                size: (activity.log2() + 1.0) * 10.0, // Log scale for size
                color: self.type_to_color(node_type),
            });
        }
        
        // Create edges based on relationships
        for story in &all_stories {
            // Dependencies
            for dep_id in &story.metadata.dependencies {
                edges.push(StoryEdge {
                    source: story.id.0.to_string(),
                    target: dep_id.0.to_string(),
                    edge_type: EdgeType::Dependency,
                    weight: 1.0,
                });
            }
            
            // Related stories
            for related_id in &story.metadata.related_stories {
                edges.push(StoryEdge {
                    source: story.id.0.to_string(),
                    target: related_id.0.to_string(),
                    edge_type: EdgeType::Reference,
                    weight: 0.5,
                });
            }
            
            // Parent-child relationships
            match &story.story_type {
                StoryType::Directory { parent_story: Some(parent), .. } |
                StoryType::File { parent_story: parent, .. } => {
                    edges.push(StoryEdge {
                        source: parent.0.to_string(),
                        target: story.id.0.to_string(),
                        edge_type: EdgeType::Dependency,
                        weight: 1.5,
                    });
                }
                _ => {}
            }
        }
        
        Ok(StoryGraph { nodes, edges })
    }
    
    /// Generate activity heatmap
    pub fn generate_activity_heatmap(&self, days: usize) -> Result<ActivityHeatmap> {
        let all_stories = self.story_engine.get_stories_by_type(|_| true);
        let now = Utc::now();
        let start = now - Duration::days(days as i64);
        
        // Initialize data structure
        let mut data = vec![vec![0usize; 24]; days];
        let mut day_labels = Vec::new();
        
        // Generate day labels
        for i in 0..days {
            let date = start + Duration::days(i as i64);
            day_labels.push(date.format("%Y-%m-%d").to_string());
        }
        
        // Count activities
        for story in &all_stories {
            for arc in &story.arcs {
                for plot_point in &arc.plot_points {
                    if plot_point.timestamp >= start && plot_point.timestamp <= now {
                        let day_offset = (plot_point.timestamp - start).num_days() as usize;
                        let hour = plot_point.timestamp.time().hour() as usize;
                        
                        if day_offset < days {
                            data[day_offset][hour] += 1;
                        }
                    }
                }
            }
        }
        
        Ok(ActivityHeatmap {
            days: day_labels,
            hours: (0..24).collect(),
            data,
        })
    }
    
    /// Get story insights
    pub fn get_story_insights(&self, story_id: StoryId) -> Result<StoryInsights> {
        let story = self.story_engine
            .get_story(&story_id)
            .ok_or_else(|| anyhow::anyhow!("Story not found"))?;
        
        let mut key_themes = HashMap::new();
        let mut common_issues = Vec::new();
        let mut decision_patterns = HashMap::new();
        let mut collaboration_frequency = HashMap::new();
        
        // Analyze plot points
        for arc in &story.arcs {
            for plot_point in &arc.plot_points {
                // Extract themes from context tokens
                for token in &plot_point.context_tokens {
                    *key_themes.entry(token.clone()).or_insert(0) += 1;
                }
                
                // Collect issues
                if let PlotType::Issue { error, resolved } = &plot_point.plot_type {
                    common_issues.push((error.clone(), *resolved));
                }
                
                // Analyze decisions
                if let PlotType::Decision { question, choice } = &plot_point.plot_type {
                    let category = self.categorize_decision(question);
                    *decision_patterns.entry(category).or_insert(0) += 1;
                }
                
                // Track interactions
                if let PlotType::Interaction { with, .. } = &plot_point.plot_type {
                    *collaboration_frequency.entry(with.clone()).or_insert(0) += 1;
                }
            }
        }
        
        // Sort and limit results
        let mut themes: Vec<_> = key_themes.into_iter().collect();
        themes.sort_by(|a, b| b.1.cmp(&a.1));
        themes.truncate(10);
        
        let mut collaborators: Vec<_> = collaboration_frequency.into_iter().collect();
        collaborators.sort_by(|a, b| b.1.cmp(&a.1));
        
        Ok(StoryInsights {
            story_id,
            key_themes: themes.into_iter().map(|(k, v)| (k, v)).collect(),
            common_issues,
            decision_patterns,
            top_collaborators: collaborators,
            complexity_score: self.calculate_complexity(&story),
            momentum_score: self.calculate_momentum(&story),
        })
    }
    
    // Helper methods
    
    fn plot_type_name(&self, plot_type: &PlotType) -> String {
        match plot_type {
            PlotType::Goal { .. } => "Goal",
            PlotType::Task { .. } => "Task",
            PlotType::Decision { .. } => "Decision",
            PlotType::Discovery { .. } => "Discovery",
            PlotType::Issue { .. } => "Issue",
            PlotType::Transformation { .. } => "Transformation",
            PlotType::Interaction { .. } => "Interaction",
            PlotType::Progress { .. } => "Progress",
            PlotType::Analysis { .. } => "Analysis",
            PlotType::Action { .. } => "Action",
            PlotType::Reasoning { .. } => "Reasoning",
            PlotType::Event { .. } => "Event",
            PlotType::Context { .. } => "Context",
        }.to_string()
    }
    
    fn extract_milestone(&self, plot_point: &PlotPoint) -> Option<Milestone> {
        match &plot_point.plot_type {
            PlotType::Goal { objective } if plot_point.importance > 0.8 => {
                Some(Milestone {
                    timestamp: plot_point.timestamp,
                    title: objective.clone(),
                    milestone_type: MilestoneType::GoalAchieved,
                })
            }
            PlotType::Decision { question, .. } if plot_point.importance > 0.8 => {
                Some(Milestone {
                    timestamp: plot_point.timestamp,
                    title: question.clone(),
                    milestone_type: MilestoneType::MajorDecision,
                })
            }
            PlotType::Issue { error, resolved: true } => {
                Some(Milestone {
                    timestamp: plot_point.timestamp,
                    title: format!("Resolved: {}", error),
                    milestone_type: MilestoneType::IssueResolved,
                })
            }
            _ => None,
        }
    }
    
    fn type_to_color(&self, node_type: &str) -> String {
        match node_type {
            "codebase" => "#4CAF50",
            "directory" => "#2196F3",
            "file" => "#FF9800",
            "agent" => "#9C27B0",
            "task" => "#F44336",
            "system" => "#00BCD4",
            _ => "#757575",
        }.to_string()
    }
    
    fn categorize_decision(&self, question: &str) -> String {
        let question_lower = question.to_lowercase();
        
        if question_lower.contains("framework") || question_lower.contains("library") {
            "Technology Choice".to_string()
        } else if question_lower.contains("implement") || question_lower.contains("approach") {
            "Implementation Strategy".to_string()
        } else if question_lower.contains("optimize") || question_lower.contains("performance") {
            "Optimization".to_string()
        } else {
            "General".to_string()
        }
    }
    
    async fn calculate_task_completion_rate(&self, stories: &[Story]) -> Result<f32> {
        let mut total_tasks = 0;
        let mut completed_tasks = 0;
        
        for story in stories {
            let task_map = self.story_engine.create_task_map(story.id).await?;
            total_tasks += task_map.tasks.len();
            completed_tasks += task_map.tasks
                .iter()
                .filter(|t| t.status == TaskStatus::Completed)
                .count();
        }
        
        Ok(if total_tasks > 0 {
            completed_tasks as f32 / total_tasks as f32
        } else {
            0.0
        })
    }
    
    fn calculate_complexity(&self, story: &Story) -> f32 {
        let plot_count = story.arcs.iter()
            .map(|a| a.plot_points.len())
            .sum::<usize>() as f32;
        
        let arc_count = story.arcs.len() as f32;
        let dependency_count = story.metadata.dependencies.len() as f32;
        
        // Weighted complexity score
        (plot_count * 0.3 + arc_count * 5.0 + dependency_count * 2.0) / 10.0
    }
    
    fn calculate_momentum(&self, story: &Story) -> f32 {
        let now = Utc::now();
        let recent_activity = story.arcs.iter()
            .flat_map(|a| &a.plot_points)
            .filter(|pp| now - pp.timestamp < Duration::days(7))
            .count() as f32;
        
        let total_activity = story.arcs.iter()
            .map(|a| a.plot_points.len())
            .sum::<usize>() as f32;
        
        if total_activity > 0.0 {
            recent_activity / total_activity
        } else {
            0.0
        }
    }
}

/// Story insights
#[derive(Debug, Clone, Serialize)]
pub struct StoryInsights {
    pub story_id: StoryId,
    pub key_themes: Vec<(String, usize)>,
    pub common_issues: Vec<(String, bool)>,
    pub decision_patterns: HashMap<String, usize>,
    pub top_collaborators: Vec<(String, usize)>,
    pub complexity_score: f32,
    pub momentum_score: f32,
}