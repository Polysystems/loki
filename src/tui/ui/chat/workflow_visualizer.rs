//! Workflow visualization for the chat interface
//! 
//! Provides visual representations of multi-step workflows, tool chains,
//! and process flows in the terminal.

use ratatui::{
    layout::{Alignment, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
    Frame,
};
use std::collections::HashMap;
use chrono::{DateTime, Utc, Duration};

/// Workflow visualizer
#[derive(Clone)]
pub struct WorkflowVisualizer {
    /// Active workflows
    workflows: HashMap<String, Workflow>,
    
    /// Visualization settings
    settings: VisualizationSettings,
    
    /// Animation states
    animations: HashMap<String, AnimationState>,
}

/// Workflow definition
#[derive(Debug, Clone)]
pub struct Workflow {
    pub id: String,
    pub name: String,
    pub description: String,
    pub steps: Vec<WorkflowStep>,
    pub connections: Vec<Connection>,
    pub state: WorkflowState,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub metadata: WorkflowMetadata,
}

/// Workflow step
#[derive(Debug, Clone)]
pub struct WorkflowStep {
    pub id: String,
    pub name: String,
    pub step_type: StepType,
    pub status: StepStatus,
    pub progress: Option<f32>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub error: Option<String>,
    pub outputs: Vec<StepOutput>,
}

/// Step types
#[derive(Debug, Clone, PartialEq)]
pub enum StepType {
    Tool { name: String },
    Decision { condition: String },
    Parallel { branches: Vec<String> },
    Loop { iterations: Option<usize> },
    Transform { operation: String },
    Wait { duration: Duration },
}

/// Step status
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StepStatus {
    Pending,
    Running,
    Success,
    Failed,
    Skipped,
    Cancelled,
}

/// Step output
#[derive(Debug, Clone)]
pub struct StepOutput {
    pub name: String,
    pub value: String,
    pub output_type: OutputType,
}

#[derive(Debug, Clone)]
pub enum OutputType {
    Text,
    Number,
    Boolean,
    Json,
    Binary,
}

/// Connection between steps
#[derive(Debug, Clone)]
pub struct Connection {
    pub from: String,
    pub to: String,
    pub connection_type: ConnectionType,
    pub label: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConnectionType {
    Sequential,
    Conditional { condition: String },
    Parallel,
    Error,
}

/// Workflow state
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WorkflowState {
    Draft,
    Running,
    Paused,
    Completed,
    Failed,
    Cancelled,
}

/// Workflow metadata
#[derive(Debug, Clone)]
pub struct WorkflowMetadata {
    pub total_steps: usize,
    pub completed_steps: usize,
    pub failed_steps: usize,
    pub skipped_steps: usize,
    pub estimated_duration: Option<Duration>,
    pub actual_duration: Option<Duration>,
}

/// Visualization settings
#[derive(Debug, Clone)]
pub struct VisualizationSettings {
    pub layout: LayoutStyle,
    pub show_timing: bool,
    pub show_connections: bool,
    pub show_outputs: bool,
    pub compact_mode: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum LayoutStyle {
    Vertical,
    Horizontal,
    Tree,
    Graph,
}

/// Animation state for workflows
#[derive(Debug, Clone)]
struct AnimationState {
    pulse_phase: f32,
    flow_offset: f32,
}

impl WorkflowVisualizer {
    pub fn new() -> Self {
        Self {
            workflows: HashMap::new(),
            settings: VisualizationSettings {
                layout: LayoutStyle::Vertical,
                show_timing: true,
                show_connections: true,
                show_outputs: false,
                compact_mode: false,
            },
            animations: HashMap::new(),
        }
    }
    
    /// Add or update a workflow
    pub fn update_workflow(&mut self, workflow: Workflow) {
        let id = workflow.id.clone();
        self.workflows.insert(id.clone(), workflow);
        
        if !self.animations.contains_key(&id) {
            self.animations.insert(id, AnimationState {
                pulse_phase: 0.0,
                flow_offset: 0.0,
            });
        }
    }
    
    /// Update or add a step to an existing workflow
    pub fn update_workflow_step(&mut self, workflow_id: &str, step: WorkflowStep) {
        if let Some(workflow) = self.workflows.get_mut(workflow_id) {
            // Check if step already exists
            let step_exists = workflow.steps.iter().any(|s| s.id == step.id);
            if !step_exists {
                workflow.steps.push(step);
            } else {
                // Update existing step
                if let Some(existing_step) = workflow.steps.iter_mut().find(|s| s.id == step.id) {
                    *existing_step = step;
                }
            }
            
            // Update metadata
            workflow.metadata.total_steps = workflow.steps.len();
            workflow.metadata.completed_steps = workflow.steps.iter()
                .filter(|s| s.status == StepStatus::Success)
                .count();
            workflow.metadata.failed_steps = workflow.steps.iter()
                .filter(|s| s.status == StepStatus::Failed)
                .count();
            workflow.metadata.skipped_steps = workflow.steps.iter()
                .filter(|s| s.status == StepStatus::Skipped)
                .count();
        }
    }
    
    /// Update workflow state
    pub fn update_workflow_state(&mut self, workflow_id: &str, state: WorkflowState) {
        if let Some(workflow) = self.workflows.get_mut(workflow_id) {
            workflow.state = state;
            if state == WorkflowState::Completed || state == WorkflowState::Failed {
                workflow.completed_at = Some(Utc::now());
                if let Some(started) = workflow.started_at {
                    workflow.metadata.actual_duration = Some(Utc::now().signed_duration_since(started));
                }
            }
        }
    }
    
    /// Update animation states
    pub fn update_animations(&mut self) {
        for (_, anim) in self.animations.iter_mut() {
            anim.pulse_phase = (anim.pulse_phase + 0.1) % (2.0 * std::f32::consts::PI);
            anim.flow_offset = (anim.flow_offset + 0.05) % 1.0;
        }
    }
    
    /// Render a workflow
    pub fn render_workflow(
        &self,
        f: &mut Frame,
        area: Rect,
        workflow_id: &str,
        theme: &super::theme_engine::ChatTheme,
    ) {
        if let Some(workflow) = self.workflows.get(workflow_id) {
            match self.settings.layout {
                LayoutStyle::Vertical => self.render_vertical_layout(f, area, workflow, theme),
                LayoutStyle::Horizontal => self.render_horizontal_layout(f, area, workflow, theme),
                LayoutStyle::Tree => self.render_tree_layout(f, area, workflow, theme),
                LayoutStyle::Graph => self.render_graph_layout(f, area, workflow, theme),
            }
        }
    }
    
    /// Render vertical layout
    fn render_vertical_layout(
        &self,
        f: &mut Frame,
        area: Rect,
        workflow: &Workflow,
        theme: &super::theme_engine::ChatTheme,
    ) {
        // Header
        let header = self.create_workflow_header(workflow, theme);
        
        // Calculate step heights
        let step_count = workflow.steps.len();
        let available_height = area.height.saturating_sub(4); // Header + borders
        let step_height = if self.settings.compact_mode { 3 } else { 5 };
        
        // Create scrollable area if needed
        let content_height = (step_count as u16) * step_height;
        let needs_scroll = content_height > available_height;
        
        // Render header
        let header_area = Rect {
            x: area.x,
            y: area.y,
            width: area.width,
            height: 3,
        };
        f.render_widget(header, header_area);
        
        // Render steps
        let steps_area = Rect {
            x: area.x,
            y: area.y + 3,
            width: area.width,
            height: area.height.saturating_sub(3),
        };
        
        let mut y_offset = steps_area.y;
        for (i, step) in workflow.steps.iter().enumerate() {
            if y_offset + step_height > steps_area.y + steps_area.height {
                break; // Stop if we run out of space
            }
            
            let step_area = Rect {
                x: steps_area.x,
                y: y_offset,
                width: steps_area.width,
                height: step_height,
            };
            
            self.render_step(f, step_area, step, i, workflow, theme);
            
            // Draw connection to next step
            if i < workflow.steps.len() - 1 && self.settings.show_connections {
                self.render_vertical_connection(f, step_area, &workflow.connections, step.id.as_str(), theme);
            }
            
            y_offset += step_height;
        }
        
        // Render scroll indicator if needed
        if needs_scroll {
            self.render_scroll_indicator(f, steps_area, theme);
        }
    }
    
    /// Render horizontal layout
    fn render_horizontal_layout(
        &self,
        f: &mut Frame,
        area: Rect,
        workflow: &Workflow,
        theme: &super::theme_engine::ChatTheme,
    ) {
        let header = self.create_workflow_header(workflow, theme);
        
        // Render header
        let header_area = Rect {
            x: area.x,
            y: area.y,
            width: area.width,
            height: 3,
        };
        f.render_widget(header, header_area);
        
        // Calculate step widths
        let step_count = workflow.steps.len();
        let step_width = (area.width / step_count as u16).max(20);
        
        // Render steps horizontally
        let steps_area = Rect {
            x: area.x,
            y: area.y + 3,
            width: area.width,
            height: area.height.saturating_sub(3),
        };
        
        let mut x_offset = steps_area.x;
        for (i, step) in workflow.steps.iter().enumerate() {
            if x_offset + step_width > steps_area.x + steps_area.width {
                break;
            }
            
            let step_area = Rect {
                x: x_offset,
                y: steps_area.y,
                width: step_width.saturating_sub(1), // Leave space for connections
                height: steps_area.height,
            };
            
            self.render_step_horizontal(f, step_area, step, i, workflow, theme);
            
            // Draw connection to next step
            if i < workflow.steps.len() - 1 && self.settings.show_connections {
                self.render_horizontal_connection(f, step_area, theme);
            }
            
            x_offset += step_width;
        }
    }
    
    /// Render tree layout
    fn render_tree_layout(
        &self,
        f: &mut Frame,
        area: Rect,
        workflow: &Workflow,
        theme: &super::theme_engine::ChatTheme,
    ) {
        // Simplified tree layout - would be more complex in real implementation
        self.render_vertical_layout(f, area, workflow, theme);
    }
    
    /// Render graph layout
    fn render_graph_layout(
        &self,
        f: &mut Frame,
        area: Rect,
        workflow: &Workflow,
        theme: &super::theme_engine::ChatTheme,
    ) {
        // Simplified graph layout - would use actual graph layout algorithms
        self.render_vertical_layout(f, area, workflow, theme);
    }
    
    /// Create workflow header
    fn create_workflow_header<'a>(&self, workflow: &Workflow, theme: &'a super::theme_engine::ChatTheme) -> Paragraph<'a> {
        let state_icon = match workflow.state {
            WorkflowState::Draft => "📝",
            WorkflowState::Running => "▶️",
            WorkflowState::Paused => "⏸️",
            WorkflowState::Completed => "✅",
            WorkflowState::Failed => "❌",
            WorkflowState::Cancelled => "🚫",
        };
        
        let state_color = match workflow.state {
            WorkflowState::Running => theme.colors.info,
            WorkflowState::Completed => theme.colors.success,
            WorkflowState::Failed => theme.colors.error,
            WorkflowState::Paused => theme.colors.warning,
            _ => theme.colors.foreground,
        };
        
        let progress = workflow.metadata.completed_steps as f32 / workflow.metadata.total_steps as f32;
        let progress_bar = self.create_progress_bar(progress, 20);
        
        let header_text = vec![
            Line::from(vec![
                Span::raw(state_icon),
                Span::raw(" "),
                Span::styled(workflow.name.clone(), Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(" - "),
                Span::styled(workflow.description.clone(), Style::default().fg(theme.colors.foreground_dim)),
            ]),
            Line::from(vec![
                Span::raw("Progress: "),
                Span::raw(progress_bar),
                Span::raw(format!(" {}/{}", workflow.metadata.completed_steps, workflow.metadata.total_steps)),
            ]),
        ];
        
        Paragraph::new(header_text)
            .block(Block::default()
                .borders(Borders::BOTTOM)
                .border_style(Style::default().fg(state_color)))
    }
    
    /// Render individual step
    fn render_step(
        &self,
        f: &mut Frame,
        area: Rect,
        step: &WorkflowStep,
        index: usize,
        workflow: &Workflow,
        theme: &super::theme_engine::ChatTheme,
    ) {
        let status_icon = self.get_status_icon(step.status);
        let status_color = self.get_status_color(step.status, theme);
        
        let type_icon = match &step.step_type {
            StepType::Tool { .. } => "🔧",
            StepType::Decision { .. } => "🔀",
            StepType::Parallel { .. } => "⚡",
            StepType::Loop { .. } => "🔄",
            StepType::Transform { .. } => "🔄",
            StepType::Wait { .. } => "⏳",
        };
        
        let mut content = vec![
            Line::from(vec![
                Span::raw(format!("{:02}. ", index + 1)),
                Span::raw(status_icon),
                Span::raw(" "),
                Span::raw(type_icon),
                Span::raw(" "),
                Span::styled(&step.name, Style::default().fg(status_color)),
            ]),
        ];
        
        // Add progress bar for running steps
        if step.status == StepStatus::Running {
            if let Some(progress) = step.progress {
                let progress_bar = self.create_mini_progress_bar(progress, 10);
                content.push(Line::from(vec![
                    Span::raw("    "),
                    Span::raw(progress_bar),
                ]));
            }
        }
        
        // Add timing information
        if self.settings.show_timing {
            if let Some(duration) = self.calculate_step_duration(step) {
                content.push(Line::from(vec![
                    Span::raw("    "),
                    Span::styled(
                        format!("⏱️  {}", self.format_duration(duration)),
                        Style::default().fg(theme.colors.foreground_dim),
                    ),
                ]));
            }
        }
        
        // Add error message if failed
        if let Some(error) = &step.error {
            content.push(Line::from(vec![
                Span::raw("    "),
                Span::styled(
                    format!("Error: {}", error),
                    Style::default().fg(theme.colors.error),
                ),
            ]));
        }
        
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(status_color));
        
        let paragraph = Paragraph::new(content)
            .block(block)
            .wrap(Wrap { trim: false });
        
        f.render_widget(paragraph, area);
    }
    
    /// Render step in horizontal layout
    fn render_step_horizontal(
        &self,
        f: &mut Frame,
        area: Rect,
        step: &WorkflowStep,
        _index: usize,
        _workflow: &Workflow,
        theme: &super::theme_engine::ChatTheme,
    ) {
        let status_icon = self.get_status_icon(step.status);
        let status_color = self.get_status_color(step.status, theme);
        
        let content = vec![
            Line::from(vec![Span::raw(status_icon)]),
            Line::from(vec![Span::styled(&step.name, Style::default().fg(status_color))]),
        ];
        
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(status_color));
        
        let paragraph = Paragraph::new(content)
            .block(block)
            .alignment(Alignment::Center);
        
        f.render_widget(paragraph, area);
    }
    
    /// Render vertical connection between steps
    fn render_vertical_connection(
        &self,
        f: &mut Frame,
        area: Rect,
        _connections: &[Connection],
        from_id: &str,
        theme: &super::theme_engine::ChatTheme,
    ) {
        let connection_area = Rect {
            x: area.x + area.width / 2,
            y: area.y + area.height - 1,
            width: 1,
            height: 2,
        };
        
        let connection_char = if let Some(anim) = self.animations.get(from_id) {
            if anim.flow_offset > 0.5 { "┃" } else { "│" }
        } else {
            "│"
        };
        
        let connection_widget = Paragraph::new(connection_char)
            .style(Style::default().fg(theme.colors.border));
        
        f.render_widget(connection_widget, connection_area);
    }
    
    /// Render horizontal connection between steps
    fn render_horizontal_connection(
        &self,
        f: &mut Frame,
        area: Rect,
        theme: &super::theme_engine::ChatTheme,
    ) {
        let connection_area = Rect {
            x: area.x + area.width,
            y: area.y + area.height / 2,
            width: 1,
            height: 1,
        };
        
        let connection_widget = Paragraph::new("→")
            .style(Style::default().fg(theme.colors.border));
        
        f.render_widget(connection_widget, connection_area);
    }
    
    /// Create progress bar
    fn create_progress_bar(&self, progress: f32, width: usize) -> String {
        let filled = (progress * width as f32) as usize;
        let empty = width.saturating_sub(filled);
        format!("[{}{}]", "█".repeat(filled), "░".repeat(empty))
    }
    
    /// Create mini progress bar
    fn create_mini_progress_bar(&self, progress: f32, width: usize) -> String {
        let filled = (progress * width as f32) as usize;
        let empty = width.saturating_sub(filled);
        format!("{}{}", "▰".repeat(filled), "▱".repeat(empty))
    }
    
    /// Get status icon
    fn get_status_icon(&self, status: StepStatus) -> &'static str {
        match status {
            StepStatus::Pending => "⏳",
            StepStatus::Running => "🔄",
            StepStatus::Success => "✅",
            StepStatus::Failed => "❌",
            StepStatus::Skipped => "⏭️",
            StepStatus::Cancelled => "🚫",
        }
    }
    
    /// Get status color
    fn get_status_color(&self, status: StepStatus, theme: &super::theme_engine::ChatTheme) -> Color {
        match status {
            StepStatus::Pending => theme.colors.foreground_dim,
            StepStatus::Running => theme.colors.info,
            StepStatus::Success => theme.colors.success,
            StepStatus::Failed => theme.colors.error,
            StepStatus::Skipped => theme.colors.warning,
            StepStatus::Cancelled => theme.colors.foreground_dim,
        }
    }
    
    /// Calculate step duration
    fn calculate_step_duration(&self, step: &WorkflowStep) -> Option<Duration> {
        match (step.started_at, step.completed_at) {
            (Some(start), Some(end)) => Some(end.signed_duration_since(start)),
            (Some(start), None) if step.status == StepStatus::Running => {
                Some(Utc::now().signed_duration_since(start))
            }
            _ => None,
        }
    }
    
    /// Format duration
    fn format_duration(&self, duration: Duration) -> String {
        let total_seconds = duration.num_seconds();
        if total_seconds < 60 {
            format!("{}s", total_seconds)
        } else if total_seconds < 3600 {
            format!("{}m {}s", total_seconds / 60, total_seconds % 60)
        } else {
            format!("{}h {}m", total_seconds / 3600, (total_seconds % 3600) / 60)
        }
    }
    
    /// Render scroll indicator
    fn render_scroll_indicator(&self, f: &mut Frame, area: Rect, theme: &super::theme_engine::ChatTheme) {
        let indicator_area = Rect {
            x: area.x + area.width - 1,
            y: area.y,
            width: 1,
            height: area.height,
        };
        
        let indicator = Paragraph::new("▼")
            .style(Style::default().fg(theme.colors.foreground_dim));
        
        f.render_widget(indicator, indicator_area);
    }
}

/// Create a sample workflow for testing
pub fn create_sample_workflow() -> Workflow {
    let steps = vec![
        WorkflowStep {
            id: "step1".to_string(),
            name: "Initialize".to_string(),
            step_type: StepType::Tool { name: "setup".to_string() },
            status: StepStatus::Success,
            progress: None,
            started_at: Some(Utc::now() - Duration::minutes(5)),
            completed_at: Some(Utc::now() - Duration::minutes(4)),
            error: None,
            outputs: vec![],
        },
        WorkflowStep {
            id: "step2".to_string(),
            name: "Process Data".to_string(),
            step_type: StepType::Tool { name: "processor".to_string() },
            status: StepStatus::Running,
            progress: Some(0.65),
            started_at: Some(Utc::now() - Duration::minutes(4)),
            completed_at: None,
            error: None,
            outputs: vec![],
        },
        WorkflowStep {
            id: "step3".to_string(),
            name: "Validate Results".to_string(),
            step_type: StepType::Decision { condition: "result.valid".to_string() },
            status: StepStatus::Pending,
            progress: None,
            started_at: None,
            completed_at: None,
            error: None,
            outputs: vec![],
        },
    ];
    
    Workflow {
        id: "sample".to_string(),
        name: "Data Processing Pipeline".to_string(),
        description: "Processes and validates incoming data".to_string(),
        steps,
        connections: vec![
            Connection {
                from: "step1".to_string(),
                to: "step2".to_string(),
                connection_type: ConnectionType::Sequential,
                label: None,
            },
            Connection {
                from: "step2".to_string(),
                to: "step3".to_string(),
                connection_type: ConnectionType::Sequential,
                label: None,
            },
        ],
        state: WorkflowState::Running,
        started_at: Some(Utc::now() - Duration::minutes(5)),
        completed_at: None,
        metadata: WorkflowMetadata {
            total_steps: 3,
            completed_steps: 1,
            failed_steps: 0,
            skipped_steps: 0,
            estimated_duration: Some(Duration::minutes(10)),
            actual_duration: None,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_workflow_creation() {
        let workflow = create_sample_workflow();
        assert_eq!(workflow.steps.len(), 3);
        assert_eq!(workflow.state, WorkflowState::Running);
        assert_eq!(workflow.metadata.completed_steps, 1);
    }
    
    #[test]
    fn test_workflow_visualizer() {
        let mut visualizer = WorkflowVisualizer::new();
        let workflow = create_sample_workflow();
        
        visualizer.update_workflow(workflow.clone());
        assert!(visualizer.workflows.contains_key(&workflow.id));
    }
}