//! Agents management subtab

use std::sync::Arc;
use std::collections::HashMap;
use tokio::sync::{RwLock, mpsc};
use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Gauge, List, ListItem, Paragraph, Row, Table, Sparkline},
    Frame,
};
use crossterm::event::{KeyCode, KeyEvent};
use anyhow::Result;
use chrono::{DateTime, Local};

use super::SubtabController;
use crate::tui::chat::orchestration::OrchestrationManager;
use crate::tui::chat::agents::manager::{AgentManager, CollaborationMode};
use crate::cognitive::agents::{
    AgentSpecialization, AgentCapability, AgentStatus as CognitiveAgentStatus,
    LoadBalancingStrategy, AgentEntry,
};
use crate::tui::chat::ui_enhancements::{StatusIndicator, Status, AnimationState, MetricsSparkline};

/// Extension methods for AgentSpecialization
trait AgentSpecializationExt {
    fn display_name(&self) -> &str;
    fn description(&self) -> &str;
    fn icon(&self) -> &str;
    fn capabilities(&self) -> Vec<&str>;
}

impl AgentSpecializationExt for AgentSpecialization {
    fn display_name(&self) -> &str {
        match self {
            Self::Analytical => "Analytical",
            Self::Creative => "Creative",
            Self::Strategic => "Strategic",
            Self::Social => "Social",
            Self::Guardian => "Guardian",
            Self::Learning => "Learning",
            Self::Coordinator => "Coordinator",
            Self::Technical => "Technical",
            Self::Managerial => "Managerial",
            Self::General => "General",
            Self::Empathetic => "Empathetic",
        }
    }
    
    fn description(&self) -> &str {
        match self {
            Self::Analytical => "Deep analysis, pattern recognition, and logical reasoning",
            Self::Creative => "Innovation, out-of-the-box thinking, and novel solutions",
            Self::Strategic => "Long-term planning, system design, and high-level decisions",
            Self::Social => "Communication, collaboration, and understanding human context",
            Self::Guardian => "Safety validation, ethical assessment, and responsible decision-making",
            Self::Learning => "Continuous improvement, knowledge acquisition, and adaptation",
            Self::Coordinator => "Meta-level orchestration, agent coordination, and system optimization",
            Self::Technical => "Code implementation, system integration, and technical problem-solving",
            Self::Managerial => "Project management, resource coordination, and team leadership",
            Self::General => "Versatile problem-solving across multiple domains and contexts",
            Self::Empathetic => "Emotional understanding, support, and interpersonal connections",
        }
    }
    
    fn icon(&self) -> &str {
        match self {
            Self::Analytical => "🔬",
            Self::Creative => "🎨",
            Self::Strategic => "♟️",
            Self::Social => "👥",
            Self::Guardian => "🛡️",
            Self::Learning => "📚",
            Self::Coordinator => "🎯",
            Self::Technical => "💻",
            Self::Managerial => "👔",
            Self::General => "🔄",
            Self::Empathetic => "💖",
        }
    }
    
    fn capabilities(&self) -> Vec<&str> {
        match self {
            Self::Analytical => vec!["Data Analysis", "Pattern Matching", "Statistical Reasoning", "Root Cause Analysis"],
            Self::Creative => vec!["Brainstorming", "Alternative Solutions", "Design Thinking", "Innovation"],
            Self::Strategic => vec!["Planning", "Architecture Design", "Risk Assessment", "Goal Setting"],
            Self::Social => vec!["Communication", "Team Coordination", "Context Understanding", "Empathy"],
            Self::Guardian => vec!["Safety Validation", "Bias Detection", "Impact Assessment", "Compliance"],
            Self::Learning => vec!["Knowledge Synthesis", "Pattern Learning", "Skill Acquisition", "Adaptation"],
            Self::Coordinator => vec!["Task Distribution", "Agent Orchestration", "Workflow Management", "System Integration"],
            Self::Technical => vec!["Coding", "Debugging", "System Design", "Tool Integration"],
            Self::Managerial => vec!["Project Management", "Resource Planning", "Team Leadership", "Process Optimization"],
            Self::General => vec!["Multi-Domain Knowledge", "Flexible Problem Solving", "Context Switching", "Generalist Approach"],
            Self::Empathetic => vec!["Emotional Intelligence", "Active Listening", "Conflict Resolution", "Supportive Communication"],
        }
    }
}

/// Extension methods for CognitiveAgentStatus
trait AgentStatusExt {
    fn display_color(&self) -> Color;
    fn icon(&self) -> &str;
    fn display_name(&self) -> &str;
}

impl AgentStatusExt for CognitiveAgentStatus {
    fn display_color(&self) -> Color {
        match self {
            Self::Active => Color::Green,
            Self::Busy => Color::Yellow,
            Self::Idle => Color::Blue,
            Self::Offline => Color::DarkGray,
        }
    }
    
    fn icon(&self) -> &str {
        match self {
            Self::Active => "●",
            Self::Busy => "◐",
            Self::Idle => "○",
            Self::Offline => "✗",
        }
    }
    
    fn display_name(&self) -> &str {
        match self {
            Self::Active => "Active",
            Self::Busy => "Busy",
            Self::Idle => "Idle",
            Self::Offline => "Offline",
        }
    }
}

/// Extended agent information for UI display
#[derive(Debug, Clone)]
struct AgentInfo {
    /// Core agent entry from cognitive system
    entry: AgentEntry,
    /// Agent specialization
    specialization: AgentSpecialization,
    /// Current task description
    current_task: Option<String>,
    /// Last activity timestamp
    last_active: DateTime<Local>,
    /// Success rate (0.0 - 1.0)
    success_rate: f32,
    /// Average response time in seconds
    average_response_time: f32,
    /// Total tasks completed
    tasks_completed: u32,
    /// Performance history (last 20 values)
    performance_history: Vec<f32>,
}

/// View mode for agents tab
#[derive(Debug, Clone, Copy, PartialEq)]
enum ViewMode {
    List,
    Details,
    Capabilities,
    Performance,
}

/// Agents management tab
pub struct AgentsTab {
    /// Reference to orchestration manager
    orchestration: Arc<RwLock<OrchestrationManager>>,
    
    /// Reference to agent manager
    agent_manager: Arc<RwLock<AgentManager>>,
    
    /// Available agents
    agents: Vec<AgentInfo>,
    
    /// Selected agent index
    selected_index: usize,
    
    /// Current view mode
    view_mode: ViewMode,
    
    /// Filter by specialization
    specialization_filter: Option<AgentSpecialization>,
    
    /// Filter by status
    status_filter: Option<CognitiveAgentStatus>,
    
    /// Show only active agents
    show_active_only: bool,
    
    /// Command sender for agent operations
    command_tx: Option<mpsc::Sender<AgentCommand>>,
    
    /// Status indicator for agent operations
    status_indicator: StatusIndicator,
    
    /// Animation for active agents
    agent_animations: HashMap<String, AnimationState>,
    
    /// Performance metrics sparkline
    performance_sparkline: MetricsSparkline,
    
    /// Task completion sparkline
    task_sparkline: MetricsSparkline,
}

/// Commands for agent operations
#[derive(Debug, Clone)]
enum AgentCommand {
    SpawnAgent(AgentSpecialization),
    TerminateAgent(String),
    PauseAgent(String),
    ResumeAgent(String),
    UpdateLoadBalancing(LoadBalancingStrategy),
    UpdateCollaborationMode(CollaborationMode),
    EnableSystem,
    DisableSystem,
}

impl AgentsTab {
    pub fn new() -> Self {
        // Create dummy references for now
        let dummy_orchestration = Arc::new(RwLock::new(OrchestrationManager::default()));
        let dummy_agent_manager = Arc::new(RwLock::new(AgentManager::default()));
        
        Self {
            orchestration: dummy_orchestration,
            agent_manager: dummy_agent_manager,
            agents: Vec::new(),
            selected_index: 0,
            view_mode: ViewMode::List,
            specialization_filter: None,
            status_filter: None,
            show_active_only: false,
            command_tx: None,
            status_indicator: StatusIndicator::new(),
            agent_animations: HashMap::new(),
            performance_sparkline: MetricsSparkline::new("Agent Performance".to_string(), 50),
            task_sparkline: MetricsSparkline::new("Task Completion Rate".to_string(), 50),
        }
    }
    
    /// Set the references
    pub fn set_references(
        &mut self,
        orchestration: Arc<RwLock<OrchestrationManager>>,
        agent_manager: Arc<RwLock<AgentManager>>,
    ) {
        self.orchestration = orchestration;
        self.agent_manager = agent_manager.clone();
        self.refresh_agents();
    }
    
    /// Set the command channel
    pub fn set_command_channel(&mut self, tx: mpsc::Sender<AgentCommand>) {
        self.command_tx = Some(tx);
    }
    
    /// Refresh agent list from the agent manager
    fn refresh_agents(&mut self) {
        self.agents.clear();
        
        // Update status
        self.status_indicator.set_status(Status::Processing, "Loading agents...".to_string());
        
        // Get agent configuration from manager
        let (enabled, specializations, collaboration_mode) = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                let manager = self.agent_manager.read().await;
                (
                    manager.agent_system_enabled,
                    manager.active_specializations.clone(),
                    manager.collaboration_mode,
                )
            })
        });
        
        if !enabled {
            self.status_indicator.set_status(Status::Warning, "Agent system disabled".to_string());
            return;
        }
        
        // Create agent entries for each active specialization
        for (idx, specialization) in specializations.iter().enumerate() {
            let agent_id = format!("{}-{:03}", specialization.display_name().to_lowercase(), idx + 1);
            
            // Determine capabilities based on specialization
            let capabilities = match specialization {
                AgentSpecialization::Analytical => vec![
                    AgentCapability::DataAnalysis,
                    AgentCapability::PatternRecognition,
                    AgentCapability::StatisticalProcessing,
                ],
                AgentSpecialization::Creative => vec![
                    AgentCapability::ContentGeneration,
                    AgentCapability::IdeaSynthesis,
                    AgentCapability::ConceptualBlending,
                ],
                AgentSpecialization::Strategic => vec![
                    AgentCapability::LongTermPlanning,
                    AgentCapability::RiskAssessment,
                    AgentCapability::DecisionAnalysis,
                ],
                AgentSpecialization::Guardian => vec![
                    AgentCapability::SafetyValidation,
                    AgentCapability::EthicalAssessment,
                    AgentCapability::SecurityMonitoring,
                ],
                AgentSpecialization::Technical => vec![
                    AgentCapability::ContentGeneration,
                    AgentCapability::DataAnalysis,
                    AgentCapability::PatternRecognition,
                ],
                _ => vec![AgentCapability::DataAnalysis],
            };
            
            // Simulate varying loads and statuses
            let (load, status, current_task) = match idx % 4 {
                0 => (0.2, CognitiveAgentStatus::Active, None),
                1 => (0.7, CognitiveAgentStatus::Busy, Some("Processing cognitive reasoning task".to_string())),
                2 => (0.4, CognitiveAgentStatus::Active, Some("Analyzing patterns in data".to_string())),
                _ => (0.1, CognitiveAgentStatus::Idle, None),
            };
            
            // Clone status for animation check
            let status_for_animation = status.clone();
            
            // Create agent entry
            let entry = AgentEntry {
                agent_id: agent_id.clone(),
                capabilities,
                current_load: load,
                status,
                task_count: (idx * 15 + 10) as usize,
                performance_score: 0.85 + (idx as f32 * 0.05).min(0.1),
            };
            
            // Generate performance history
            let mut performance_history = Vec::new();
            for i in 0..20 {
                performance_history.push(
                    0.7 + (i as f32 * 0.01) + ((idx as f32 * 0.1).sin() * 0.15)
                );
            }
            
            // Create animation for active agents (check before moving entry)
            if status_for_animation == CognitiveAgentStatus::Active {
                self.agent_animations.insert(
                    agent_id.clone(),
                    AnimationState::new(0.0, 1.0, std::time::Duration::from_secs(2))
                );
            }
            
            // Create extended info
            let info = AgentInfo {
                entry,
                specialization: specialization.clone(),
                current_task,
                last_active: Local::now() - chrono::Duration::minutes((idx * 2) as i64),
                success_rate: 0.88 + (idx as f32 * 0.03).min(0.1),
                average_response_time: 1.5 + (idx as f32 * 0.8).min(3.0),
                tasks_completed: (idx * 15 + 10) as u32,
                performance_history,
            };
            
            self.agents.push(info);
        }
        
        // Update metrics
        let total_agents = self.agents.len();
        let active_agents = self.agents.iter().filter(|a| a.entry.status == CognitiveAgentStatus::Active).count();
        let avg_performance = self.agents.iter().map(|a| a.entry.performance_score).sum::<f32>() / total_agents.max(1) as f32;
        let total_tasks = self.agents.iter().map(|a| a.tasks_completed).sum::<u32>();
        
        self.performance_sparkline.add_point((avg_performance * 100.0) as u64);
        self.task_sparkline.add_point(total_tasks as u64);
        
        // Update status
        self.status_indicator.set_status(
            Status::Success,
            format!("{} agents ({} active)", total_agents, active_agents)
        );
    }
    
    /// Get filtered agents
    fn get_filtered_agents(&self) -> Vec<&AgentInfo> {
        self.agents
            .iter()
            .filter(|agent| {
                // Specialization filter
                if let Some(filter) = &self.specialization_filter {
                    if agent.specialization != *filter {
                        return false;
                    }
                }
                
                // Status filter
                if let Some(filter) = &self.status_filter {
                    if agent.entry.status != *filter {
                        return false;
                    }
                }
                
                // Active filter
                if self.show_active_only && agent.entry.status == CognitiveAgentStatus::Offline {
                    return false;
                }
                
                true
            })
            .collect()
    }
    
    /// Send command if channel is available
    async fn send_command(&self, cmd: AgentCommand) -> Result<()> {
        if let Some(tx) = &self.command_tx {
            tx.send(cmd).await?;
        }
        Ok(())
    }
    
    /// Render agent list view
    fn render_list_view(&mut self, f: &mut Frame, area: Rect) {
        let filtered_agents = self.get_filtered_agents();
        
        // Create table rows with animated icons
        let rows: Vec<Row> = filtered_agents
            .iter()
            .enumerate()
            .map(|(i, agent)| {
                let selected = i == self.selected_index;
                let style = if selected {
                    Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
                } else {
                    Style::default()
                };
                
                // Get animated icon for active agents
                let status_icon = if agent.entry.status == CognitiveAgentStatus::Active {
                    if let Some(anim) = self.agent_animations.get(&agent.entry.agent_id) {
                        let frames = vec!["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
                        let index = ((anim.current_value() * frames.len() as f32) as usize) % frames.len();
                        frames[index].to_string()
                    } else {
                        agent.entry.status.icon().to_string()
                    }
                } else {
                    agent.entry.status.icon().to_string()
                };
                
                Row::new(vec![
                    status_icon,
                    format!("{} {}", agent.specialization.icon(), agent.entry.agent_id),
                    agent.specialization.display_name().to_string(),
                    agent.current_task.as_deref().unwrap_or("-").to_string(),
                    format!("{}", agent.tasks_completed),
                    format!("{:.1}%", agent.success_rate * 100.0),
                    format!("{:.2}", agent.entry.current_load),
                ])
                .style(style)
                .height(1)
            })
            .collect();
        
        let table = Table::new(
            rows,
            [
                Constraint::Length(3),   // Status icon
                Constraint::Length(20),  // Agent ID
                Constraint::Length(15),  // Specialization
                Constraint::Min(20),     // Current task
                Constraint::Length(8),   // Tasks
                Constraint::Length(8),   // Success
                Constraint::Length(8),   // Load
            ],
        )
        .header(
            Row::new(vec!["", "Agent ID", "Specialization", "Current Task", "Tasks", "Success", "Load"])
                .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
                .bottom_margin(1)
        )
        .block(Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .title(" Agents "));
        
        f.render_widget(table, area);
    }
    
    /// Render agent details view
    fn render_details_view(&self, f: &mut Frame, area: Rect) {
        let filtered_agents = self.get_filtered_agents();
        
        if let Some(agent) = filtered_agents.get(self.selected_index) {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(8),   // Basic info
                    Constraint::Length(6),   // Status and metrics
                    Constraint::Min(5),      // Current task details
                ])
                .split(area);
            
            // Basic info
            let info_lines = vec![
                Line::from(vec![
                    Span::styled("Agent: ", Style::default().fg(Color::DarkGray)),
                    Span::styled(
                        format!("{} {}", agent.specialization.icon(), agent.entry.agent_id),
                        Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
                    ),
                ]),
                Line::from(vec![
                    Span::styled("Specialization: ", Style::default().fg(Color::DarkGray)),
                    Span::styled(agent.specialization.display_name(), Style::default().fg(Color::Yellow)),
                ]),
                Line::from(vec![
                    Span::styled("Status: ", Style::default().fg(Color::DarkGray)),
                    Span::styled(
                        format!("{} {}", agent.entry.status.icon(), agent.entry.status.display_name()),
                        Style::default().fg(agent.entry.status.display_color())
                    ),
                ]),
                Line::from(vec![
                    Span::styled("Current Load: ", Style::default().fg(Color::DarkGray)),
                    Span::styled(
                        format!("{:.0}%", agent.entry.current_load * 100.0),
                        Style::default().fg(if agent.entry.current_load > 0.8 {
                            Color::Red
                        } else if agent.entry.current_load > 0.5 {
                            Color::Yellow
                        } else {
                            Color::Green
                        })
                    ),
                ]),
                Line::from(vec![
                    Span::styled("Description: ", Style::default().fg(Color::DarkGray)),
                    Span::raw(agent.specialization.description()),
                ]),
            ];
            
            let info = Paragraph::new(info_lines)
                .block(Block::default()
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded)
                    .title(" Agent Details "));
            f.render_widget(info, chunks[0]);
            
            // Metrics
            let metrics_chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([
                    Constraint::Percentage(33),
                    Constraint::Percentage(33),
                    Constraint::Percentage(34),
                ])
                .split(chunks[1]);
            
            // Success rate gauge
            let success_gauge = Gauge::default()
                .block(Block::default()
                    .borders(Borders::ALL)
                    .title(" Performance Score "))
                .gauge_style(Style::default().fg(Color::Green))
                .ratio(agent.entry.performance_score as f64);
            f.render_widget(success_gauge, metrics_chunks[0]);
            
            // Tasks completed
            let tasks_text = Paragraph::new(format!("{}", agent.tasks_completed))
                .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
                .alignment(Alignment::Center)
                .block(Block::default()
                    .borders(Borders::ALL)
                    .title(" Tasks Completed "));
            f.render_widget(tasks_text, metrics_chunks[1]);
            
            // Current load gauge
            let load_gauge = Gauge::default()
                .block(Block::default()
                    .borders(Borders::ALL)
                    .title(" Current Load "))
                .gauge_style(Style::default().fg(if agent.entry.current_load > 0.8 {
                    Color::Red
                } else if agent.entry.current_load > 0.5 {
                    Color::Yellow
                } else {
                    Color::Green
                }))
                .ratio(agent.entry.current_load as f64);
            f.render_widget(load_gauge, metrics_chunks[2]);
            
            // Current task
            let task_lines = if let Some(task) = &agent.current_task {
                vec![
                    Line::from(vec![
                        Span::styled("Task: ", Style::default().fg(Color::DarkGray)),
                        Span::raw(task),
                    ]),
                    Line::from(vec![
                        Span::styled("Started: ", Style::default().fg(Color::DarkGray)),
                        Span::raw(agent.last_active.format("%H:%M:%S").to_string()),
                    ]),
                    Line::from(vec![
                        Span::styled("Response Time: ", Style::default().fg(Color::DarkGray)),
                        Span::raw(format!("{:.1}s average", agent.average_response_time)),
                    ]),
                ]
            } else {
                vec![Line::from("No active task")]
            };
            
            let task_info = Paragraph::new(task_lines)
                .block(Block::default()
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded)
                    .title(" Current Task "));
            f.render_widget(task_info, chunks[2]);
        }
    }
    
    /// Render capabilities view
    fn render_capabilities_view(&self, f: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Length(25),  // Specializations list
                Constraint::Min(40),     // Capabilities details
            ])
            .split(area);
        
        // Get all possible specializations
        let specializations = vec![
            AgentSpecialization::Analytical,
            AgentSpecialization::Creative,
            AgentSpecialization::Strategic,
            AgentSpecialization::Social,
            AgentSpecialization::Guardian,
            AgentSpecialization::Learning,
            AgentSpecialization::Coordinator,
            AgentSpecialization::Technical,
        ];
        
        // Specializations list
        let items: Vec<ListItem> = specializations
            .iter()
            .map(|spec| {
                let selected = self.specialization_filter.as_ref() == Some(spec);
                let style = if selected {
                    Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
                } else {
                    Style::default()
                };
                ListItem::new(format!("{} {}", spec.icon(), spec.display_name())).style(style)
            })
            .collect();
        
        let spec_list = List::new(items)
            .block(Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .title(" Specializations "));
        f.render_widget(spec_list, chunks[0]);
        
        // Capabilities details
        if let Some(selected_spec) = &self.specialization_filter {
            let mut lines = vec![
                Line::from(vec![
                    Span::raw(selected_spec.icon()),
                    Span::raw(" "),
                    Span::styled(selected_spec.display_name(), Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
                ]),
                Line::from(""),
                Line::from(vec![
                    Span::styled("Description:", Style::default().fg(Color::Yellow)),
                ]),
                Line::from(selected_spec.description()),
                Line::from(""),
                Line::from(vec![
                    Span::styled("Core Capabilities:", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
                ]),
            ];
            
            for capability in selected_spec.capabilities() {
                lines.push(Line::from(vec![
                    Span::raw("  • "),
                    Span::raw(capability),
                ]));
            }
            
            // Add active agent count for this specialization
            let active_count = self.agents.iter()
                .filter(|a| a.specialization == *selected_spec)
                .count();
            
            lines.push(Line::from(""));
            lines.push(Line::from(vec![
                Span::styled("Active Agents: ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    active_count.to_string(),
                    Style::default().fg(if active_count > 0 { Color::Green } else { Color::DarkGray })
                ),
            ]));
            
            let capabilities = Paragraph::new(lines)
                .block(Block::default()
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded)
                    .title(" Specialization Details "));
            f.render_widget(capabilities, chunks[1]);
        } else {
            let help = Paragraph::new("Select a specialization to view its capabilities")
                .style(Style::default().fg(Color::DarkGray))
                .alignment(Alignment::Center)
                .block(Block::default()
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded)
                    .title(" Capabilities "));
            f.render_widget(help, chunks[1]);
        }
    }
    
    /// Render performance view
    fn render_performance_view(&self, f: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(10),  // Overview metrics
                Constraint::Min(10),     // Performance graphs
            ])
            .split(area);
        
        // Calculate aggregate metrics
        let total_agents = self.agents.len();
        let active_agents = self.agents.iter().filter(|a| a.entry.status == CognitiveAgentStatus::Active || a.entry.status == CognitiveAgentStatus::Busy).count();
        let avg_load = if total_agents > 0 {
            self.agents.iter().map(|a| a.entry.current_load).sum::<f32>() / total_agents as f32
        } else {
            0.0
        };
        let avg_performance = if total_agents > 0 {
            self.agents.iter().map(|a| a.entry.performance_score).sum::<f32>() / total_agents as f32
        } else {
            0.0
        };
        
        // Overview metrics
        let metrics_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(25),
                Constraint::Percentage(25),
                Constraint::Percentage(25),
                Constraint::Percentage(25),
            ])
            .split(chunks[0]);
        
        // Total agents
        let total_block = Paragraph::new(vec![
            Line::from("Total Agents"),
            Line::from(""),
            Line::from(Span::styled(
                total_agents.to_string(),
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
            )),
        ])
        .alignment(Alignment::Center)
        .block(Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded));
        f.render_widget(total_block, metrics_chunks[0]);
        
        // Active agents
        let active_block = Paragraph::new(vec![
            Line::from("Active"),
            Line::from(""),
            Line::from(Span::styled(
                format!("{}/{}", active_agents, total_agents),
                Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)
            )),
        ])
        .alignment(Alignment::Center)
        .block(Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded));
        f.render_widget(active_block, metrics_chunks[1]);
        
        // Average load
        let load_block = Paragraph::new(vec![
            Line::from("Avg Load"),
            Line::from(""),
            Line::from(Span::styled(
                format!("{:.1}%", avg_load * 100.0),
                Style::default().fg(if avg_load > 0.8 {
                    Color::Red
                } else if avg_load > 0.5 {
                    Color::Yellow
                } else {
                    Color::Green
                }).add_modifier(Modifier::BOLD)
            )),
        ])
        .alignment(Alignment::Center)
        .block(Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded));
        f.render_widget(load_block, metrics_chunks[2]);
        
        // Average performance
        let perf_block = Paragraph::new(vec![
            Line::from("Avg Performance"),
            Line::from(""),
            Line::from(Span::styled(
                format!("{:.1}%", avg_performance * 100.0),
                Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)
            )),
        ])
        .alignment(Alignment::Center)
        .block(Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded));
        f.render_widget(perf_block, metrics_chunks[3]);
        
        // Performance graphs
        if let Some(selected_agent) = self.get_filtered_agents().get(self.selected_index) {
            let graph_chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([
                    Constraint::Percentage(50),
                    Constraint::Percentage(50),
                ])
                .split(chunks[1]);
            
            // Performance history sparkline
            let perf_data: Vec<u64> = selected_agent.performance_history
                .iter()
                .map(|v| (*v * 100.0) as u64)
                .collect();
            
            let perf_sparkline = Sparkline::default()
                .block(Block::default()
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded)
                    .title(format!(" {} Performance History ", selected_agent.entry.agent_id)))
                .data(&perf_data)
                .style(Style::default().fg(Color::Cyan));
            f.render_widget(perf_sparkline, graph_chunks[0]);
            
            // Agent statistics
            let stats_lines = vec![
                Line::from(vec![
                    Span::styled("Agent: ", Style::default().fg(Color::DarkGray)),
                    Span::styled(&selected_agent.entry.agent_id, Style::default().fg(Color::Cyan)),
                ]),
                Line::from(""),
                Line::from(vec![
                    Span::styled("Tasks Completed: ", Style::default().fg(Color::DarkGray)),
                    Span::raw(selected_agent.tasks_completed.to_string()),
                ]),
                Line::from(vec![
                    Span::styled("Success Rate: ", Style::default().fg(Color::DarkGray)),
                    Span::raw(format!("{:.1}%", selected_agent.success_rate * 100.0)),
                ]),
                Line::from(vec![
                    Span::styled("Avg Response: ", Style::default().fg(Color::DarkGray)),
                    Span::raw(format!("{:.1}s", selected_agent.average_response_time)),
                ]),
                Line::from(vec![
                    Span::styled("Current Load: ", Style::default().fg(Color::DarkGray)),
                    Span::raw(format!("{:.1}%", selected_agent.entry.current_load * 100.0)),
                ]),
                Line::from(vec![
                    Span::styled("Performance: ", Style::default().fg(Color::DarkGray)),
                    Span::raw(format!("{:.1}%", selected_agent.entry.performance_score * 100.0)),
                ]),
            ];
            
            let stats = Paragraph::new(stats_lines)
                .block(Block::default()
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded)
                    .title(" Agent Statistics "));
            f.render_widget(stats, graph_chunks[1]);
        } else {
            let no_data = Paragraph::new("No agent selected")
                .style(Style::default().fg(Color::DarkGray))
                .alignment(Alignment::Center)
                .block(Block::default()
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded)
                    .title(" Performance Data "));
            f.render_widget(no_data, chunks[1]);
        }
    }
}

impl SubtabController for AgentsTab {
    fn render(&mut self, f: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),   // Title and filters
                Constraint::Min(10),     // Main content
                Constraint::Length(3),   // Help bar
                Constraint::Length(4),   // Metrics sparklines
            ])
            .split(area);
        
        // Get agent manager state
        let (agent_enabled, collaboration_mode, load_balancing) = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                let manager = self.agent_manager.read().await;
                (
                    manager.agent_system_enabled,
                    manager.collaboration_mode.clone(),
                    manager.load_balancing_strategy.clone(),
                )
            })
        });
        
        // Title with filters and status
        let title_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Min(20),
                Constraint::Length(40),
            ])
            .split(chunks[0]);
            
        let filter_text = format!(
            "🤖 Agents {} - {} agents{}{} | Mode: {:?}",
            if agent_enabled { "[ENABLED]" } else { "[DISABLED]" },
            self.get_filtered_agents().len(),
            if let Some(spec_filter) = &self.specialization_filter {
                format!(" | {}: {}", spec_filter.icon(), spec_filter.display_name())
            } else {
                String::new()
            },
            if self.show_active_only { " | Active Only" } else { "" },
            self.view_mode
        );
        let title = Paragraph::new(filter_text)
            .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
            .alignment(Alignment::Center);
        f.render_widget(title, title_chunks[0]);
        
        // Status indicator
        let status_line = self.status_indicator.render();
        let status = Paragraph::new(status_line)
            .alignment(Alignment::Right);
        f.render_widget(status, title_chunks[1]);
        
        // Main content based on agent system state
        if !agent_enabled {
            // Show disabled state
            let disabled_msg = Paragraph::new(vec![
                Line::from(""),
                Line::from("Agent System is currently DISABLED"),
                Line::from(""),
                Line::from("Press 'e' to enable the agent system"),
                Line::from(""),
                Line::from(format!("Current settings:")),
                Line::from(format!("  Collaboration Mode: {:?}", collaboration_mode)),
                Line::from(format!("  Load Balancing: {:?}", load_balancing)),
            ])
            .style(Style::default().fg(Color::DarkGray))
            .alignment(Alignment::Center)
            .block(Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .title(" Agent System Disabled "));
            f.render_widget(disabled_msg, chunks[1]);
        } else {
            match self.view_mode {
                ViewMode::List => self.render_list_view(f, chunks[1]),
                ViewMode::Details => self.render_details_view(f, chunks[1]),
                ViewMode::Capabilities => self.render_capabilities_view(f, chunks[1]),
                ViewMode::Performance => self.render_performance_view(f, chunks[1]),
            }
        }
        
        // Help bar
        let help_text = match self.view_mode {
            ViewMode::List => "↑/↓ Navigate | Enter Details | v View Mode | s Specialization Filter | a Active Only | e Enable/Disable",
            ViewMode::Details => "↑/↓ Navigate | Esc Back | v View Mode | r Refresh",
            ViewMode::Capabilities => "↑/↓ Navigate Types | v View Mode | Enter Select | Esc Clear",
            ViewMode::Performance => "↑/↓ Select Agent | v View Mode | r Refresh | Esc Back",
        };
        
        let help = Paragraph::new(help_text)
            .style(Style::default().fg(Color::DarkGray))
            .alignment(Alignment::Center)
            .block(Block::default()
                .borders(Borders::TOP)
                .border_style(Style::default().fg(Color::DarkGray)));
        f.render_widget(help, chunks[2]);
        
        // Metrics sparklines
        let sparkline_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(50),
                Constraint::Percentage(50),
            ])
            .split(chunks[3]);
            
        self.performance_sparkline.render(f, sparkline_chunks[0]);
        self.task_sparkline.render(f, sparkline_chunks[1]);
    }
    
    fn handle_input(&mut self, key: KeyEvent) -> Result<()> {
        match key.code {
            // Navigation
            KeyCode::Up => {
                if self.selected_index > 0 {
                    self.selected_index -= 1;
                }
            }
            KeyCode::Down => {
                let max_index = self.get_filtered_agents().len().saturating_sub(1);
                if self.selected_index < max_index {
                    self.selected_index += 1;
                }
            }
            
            // View mode switching
            KeyCode::Char('v') => {
                self.view_mode = match self.view_mode {
                    ViewMode::List => ViewMode::Details,
                    ViewMode::Details => ViewMode::Capabilities,
                    ViewMode::Capabilities => ViewMode::Performance,
                    ViewMode::Performance => ViewMode::List,
                };
            }
            
            // Filters
            KeyCode::Char('s') => {
                if self.view_mode == ViewMode::List {
                    // Cycle through specializations
                    let specs = vec![
                        None,
                        Some(AgentSpecialization::Analytical),
                        Some(AgentSpecialization::Creative),
                        Some(AgentSpecialization::Strategic),
                        Some(AgentSpecialization::Social),
                        Some(AgentSpecialization::Guardian),
                        Some(AgentSpecialization::Learning),
                        Some(AgentSpecialization::Coordinator),
                        Some(AgentSpecialization::Technical),
                    ];
                    
                    let current_idx = specs.iter().position(|s| s == &self.specialization_filter).unwrap_or(0);
                    let next_idx = (current_idx + 1) % specs.len();
                    self.specialization_filter = specs[next_idx].clone();
                    self.selected_index = 0;
                }
            }
            
            KeyCode::Char('a') => {
                self.show_active_only = !self.show_active_only;
                self.selected_index = 0;
            }
            
            // Enable/disable system
            KeyCode::Char('e') => {
                tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current().block_on(async {
                        let mut manager = self.agent_manager.write().await;
                        manager.agent_system_enabled = !manager.agent_system_enabled;
                        
                        if manager.agent_system_enabled {
                            let _ = self.send_command(AgentCommand::EnableSystem).await;
                        } else {
                            let _ = self.send_command(AgentCommand::DisableSystem).await;
                        }
                    })
                });
                self.refresh_agents();
            }
            
            // Refresh
            KeyCode::Char('r') => {
                self.refresh_agents();
            }
            
            // Actions based on view mode
            KeyCode::Enter => {
                match self.view_mode {
                    ViewMode::List => self.view_mode = ViewMode::Details,
                    ViewMode::Capabilities => {
                        // In capabilities view, Enter selects the specialization for filtering
                        let specs = vec![
                            AgentSpecialization::Analytical,
                            AgentSpecialization::Creative,
                            AgentSpecialization::Strategic,
                            AgentSpecialization::Social,
                            AgentSpecialization::Guardian,
                            AgentSpecialization::Learning,
                            AgentSpecialization::Coordinator,
                            AgentSpecialization::Technical,
                        ];
                        
                        // Find which specialization is selected based on navigation
                        if let Some(spec) = specs.get(self.selected_index) {
                            self.specialization_filter = Some(spec.clone());
                        }
                    }
                    _ => {}
                }
            }
            
            KeyCode::Esc => {
                match self.view_mode {
                    ViewMode::Details | ViewMode::Performance => self.view_mode = ViewMode::List,
                    ViewMode::Capabilities => self.specialization_filter = None,
                    _ => {}
                }
            }
            
            _ => {}
        }
        
        Ok(())
    }
    
    fn update(&mut self) -> Result<()> {
        // Periodically refresh agent states
        // In a real implementation, this would be triggered by actual state changes
        Ok(())
    }
    
    fn name(&self) -> &str {
        "Agents"
    }
}