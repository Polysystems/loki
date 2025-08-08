//! Interactive Control System for Autonomous Intelligence
//!
//! This module provides interactive controls for managing and influencing
//! Loki's autonomous intelligence systems through the TUI.

use anyhow::Result;
use crossterm::event::{KeyCode, KeyModifiers};
use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph},
    Frame,
};

use crate::tui::autonomous_data_types::*;

/// Available control actions for autonomous systems
#[derive(Debug, Clone, PartialEq)]
pub enum AutonomousControlAction {
    // Goal Management
    CreateGoal { name: String, goal_type: GoalType, priority: Priority },
    ModifyGoalPriority { goal_id: String, new_priority: Priority },
    SuspendGoal { goal_id: String },
    ResumeGoal { goal_id: String },
    ApproveStrategicPlan { plan_id: String },
    ModifyPlanMilestone { plan_id: String, milestone_id: String, new_date: String },
    IdentifySynergyOpportunities,
    
    // Agent Coordination
    AssignAgentToGoal { agent_id: String, goal_id: String },
    ModifyAgentRole { agent_id: String, new_role: String },
    InitiateCoordinationProtocol { protocol_type: ProtocolType, participant_ids: Vec<String> },
    ConfigureConsensusThreshold { threshold: f32 },
    EnableEmergentBehaviorDetection { enabled: bool },
    
    // Thermodynamic Control
    AdjustEntropyTarget { new_target: f32 },
    BalanceGradients { value_weight: f32, harmony_weight: f32, intuition_weight: f32 },
    InitiateEntropyReduction,
    SetEntropyThreshold { warning: f32, critical: f32 },
    OptimizeFreeEnergy,
    ConfigureThermodynamicSafetyBounds { min_stability: f32, max_entropy: f32 },
    
    // Learning Control
    SetLearningRate { new_rate: f32 },
    EnableMetaLearning { enabled: bool },
    FocusLearningObjective { objective_id: String },
    ConfigureAdaptationSpeed { speed: f32 },
    InitiateKnowledgeIntegration,
    SetLearningPhaseTransition { phase: String },
    
    // Recursive Reasoning
    SetMaxRecursionDepth { depth: u32 },
    EnablePatternReplication { enabled: bool },
    OptimizeReasoningTemplates,
    ConfigureScaleCoordination { scales: Vec<String> },
    SetConvergenceThreshold { threshold: f32 },
    EnableCrossScaleCommunication { enabled: bool },
    
    // Safety Control
    ConfigureSafetyValidation { strictness: f32 },
    UpdateFilteringRules { rule_type: String, parameters: Vec<String> },
    SetExternalRequestLimit { limit_per_hour: u32 },
    EnableAnomalyDetection { sensitivity: f32 },
    ConfigureActionValidation { pre_conditions: bool, post_conditions: bool },
    
    // System Control
    EmergencyStop,
    RestartAutonomousLoop,
    SaveSystemSnapshot,
    LoadSystemSnapshot { snapshot_id: String },
    ExportSystemMetrics { format: String },
    ResetToBaseline { preserve_learning: bool },
}

/// Control mode state
#[derive(Debug, Clone, PartialEq)]
pub enum ControlMode {
    Normal,
    GoalCreation,
    GradientAdjustment,
    AgentAssignment,
    ThermodynamicTuning,
    SafetyConfiguration,
    RecursiveControl,
    LearningConfiguration,
    SystemManagement,
}

/// Interactive control state for cognitive systems
pub struct CognitiveControlState {
    pub control_mode: ControlMode,
    pub selected_action: usize,
    pub available_actions: Vec<AutonomousControlAction>,
    pub input_buffer: String,
    pub confirmation_pending: Option<AutonomousControlAction>,
    pub control_history: Vec<(chrono::DateTime<chrono::Utc>, AutonomousControlAction, bool)>,
}

impl Default for CognitiveControlState {
    fn default() -> Self {
        Self {
            control_mode: ControlMode::Normal,
            selected_action: 0,
            available_actions: vec![],
            input_buffer: String::new(),
            confirmation_pending: None,
            control_history: vec![],
        }
    }
}

impl CognitiveControlState {
    /// Handle key input for cognitive controls
    pub fn handle_key(&mut self, key: KeyCode, modifiers: KeyModifiers) -> Result<Option<AutonomousControlAction>> {
        match self.control_mode {
            ControlMode::Normal => self.handle_normal_mode(key, modifiers),
            ControlMode::GoalCreation => self.handle_goal_creation_mode(key, modifiers),
            ControlMode::GradientAdjustment => self.handle_gradient_adjustment_mode(key, modifiers),
            ControlMode::AgentAssignment => self.handle_agent_assignment_mode(key, modifiers),
            ControlMode::ThermodynamicTuning => self.handle_thermodynamic_tuning_mode(key, modifiers),
            ControlMode::SafetyConfiguration => self.handle_safety_configuration_mode(key, modifiers),
            ControlMode::RecursiveControl => self.handle_recursive_control_mode(key, modifiers),
            ControlMode::LearningConfiguration => self.handle_learning_configuration_mode(key, modifiers),
            ControlMode::SystemManagement => self.handle_system_management_mode(key, modifiers),
        }
    }
    
    fn handle_normal_mode(&mut self, key: KeyCode, _modifiers: KeyModifiers) -> Result<Option<AutonomousControlAction>> {
        match key {
            // Navigation
            KeyCode::Up => {
                if self.selected_action > 0 {
                    self.selected_action -= 1;
                }
                Ok(None)
            }
            KeyCode::Down => {
                if self.selected_action < self.available_actions.len().saturating_sub(1) {
                    self.selected_action += 1;
                }
                Ok(None)
            }
            
            // Action shortcuts
            KeyCode::Char('g') => {
                self.control_mode = ControlMode::GoalCreation;
                self.input_buffer.clear();
                Ok(None)
            }
            KeyCode::Char('t') => {
                self.control_mode = ControlMode::ThermodynamicTuning;
                Ok(None)
            }
            KeyCode::Char('a') => {
                self.control_mode = ControlMode::AgentAssignment;
                Ok(None)
            }
            KeyCode::Char('r') => {
                self.control_mode = ControlMode::GradientAdjustment;
                Ok(None)
            }
            KeyCode::Char('s') => {
                self.control_mode = ControlMode::SafetyConfiguration;
                Ok(None)
            }
            KeyCode::Char('d') => {
                self.control_mode = ControlMode::RecursiveControl;
                Ok(None)
            }
            KeyCode::Char('l') => {
                self.control_mode = ControlMode::LearningConfiguration;
                Ok(None)
            }
            KeyCode::Char('m') => {
                self.control_mode = ControlMode::SystemManagement;
                Ok(None)
            }
            
            // Emergency controls
            KeyCode::Char('!') => {
                self.confirmation_pending = Some(AutonomousControlAction::EmergencyStop);
                Ok(None)
            }
            
            // Execute selected action
            KeyCode::Enter => {
                if let Some(action) = self.confirmation_pending.take() {
                    self.control_history.push((chrono::Utc::now(), action.clone(), true));
                    Ok(Some(action))
                } else if self.selected_action < self.available_actions.len() {
                    let action = self.available_actions[self.selected_action].clone();
                    self.confirmation_pending = Some(action);
                    Ok(None)
                } else {
                    Ok(None)
                }
            }
            
            // Cancel
            KeyCode::Esc => {
                self.confirmation_pending = None;
                Ok(None)
            }
            
            _ => Ok(None)
        }
    }
    
    fn handle_goal_creation_mode(&mut self, key: KeyCode, _modifiers: KeyModifiers) -> Result<Option<AutonomousControlAction>> {
        match key {
            KeyCode::Char(c) => {
                self.input_buffer.push(c);
                Ok(None)
            }
            KeyCode::Backspace => {
                self.input_buffer.pop();
                Ok(None)
            }
            KeyCode::Enter => {
                if !self.input_buffer.is_empty() {
                    let action = AutonomousControlAction::CreateGoal {
                        name: self.input_buffer.clone(),
                        goal_type: GoalType::Strategic,
                        priority: Priority::Medium,
                    };
                    self.input_buffer.clear();
                    self.control_mode = ControlMode::Normal;
                    self.control_history.push((chrono::Utc::now(), action.clone(), true));
                    Ok(Some(action))
                } else {
                    Ok(None)
                }
            }
            KeyCode::Esc => {
                self.control_mode = ControlMode::Normal;
                self.input_buffer.clear();
                Ok(None)
            }
            _ => Ok(None)
        }
    }
    
    fn handle_gradient_adjustment_mode(&mut self, key: KeyCode, _modifiers: KeyModifiers) -> Result<Option<AutonomousControlAction>> {
        // Simplified for now - in a real implementation, this would have sliders
        match key {
            KeyCode::Char('1') => {
                let action = AutonomousControlAction::BalanceGradients {
                    value_weight: 0.4,
                    harmony_weight: 0.3,
                    intuition_weight: 0.3,
                };
                self.control_mode = ControlMode::Normal;
                Ok(Some(action))
            }
            KeyCode::Char('2') => {
                let action = AutonomousControlAction::BalanceGradients {
                    value_weight: 0.3,
                    harmony_weight: 0.4,
                    intuition_weight: 0.3,
                };
                self.control_mode = ControlMode::Normal;
                Ok(Some(action))
            }
            KeyCode::Char('3') => {
                let action = AutonomousControlAction::BalanceGradients {
                    value_weight: 0.3,
                    harmony_weight: 0.3,
                    intuition_weight: 0.4,
                };
                self.control_mode = ControlMode::Normal;
                Ok(Some(action))
            }
            KeyCode::Esc => {
                self.control_mode = ControlMode::Normal;
                Ok(None)
            }
            _ => Ok(None)
        }
    }
    
    fn handle_agent_assignment_mode(&mut self, key: KeyCode, _modifiers: KeyModifiers) -> Result<Option<AutonomousControlAction>> {
        // Placeholder implementation
        match key {
            KeyCode::Esc => {
                self.control_mode = ControlMode::Normal;
                Ok(None)
            }
            _ => Ok(None)
        }
    }
    
    fn handle_thermodynamic_tuning_mode(&mut self, key: KeyCode, _modifiers: KeyModifiers) -> Result<Option<AutonomousControlAction>> {
        match key {
            KeyCode::Char('+') | KeyCode::Char('=') => {
                let action = AutonomousControlAction::AdjustEntropyTarget { new_target: 0.5 };
                Ok(Some(action))
            }
            KeyCode::Char('-') => {
                let action = AutonomousControlAction::AdjustEntropyTarget { new_target: 0.3 };
                Ok(Some(action))
            }
            KeyCode::Char('r') => {
                let action = AutonomousControlAction::InitiateEntropyReduction;
                self.control_mode = ControlMode::Normal;
                Ok(Some(action))
            }
            KeyCode::Esc => {
                self.control_mode = ControlMode::Normal;
                Ok(None)
            }
            _ => Ok(None)
        }
    }
    
    fn handle_safety_configuration_mode(&mut self, key: KeyCode, _modifiers: KeyModifiers) -> Result<Option<AutonomousControlAction>> {
        match key {
            KeyCode::Char('1') => {
                let action = AutonomousControlAction::ConfigureSafetyValidation { strictness: 0.8 };
                self.control_mode = ControlMode::Normal;
                Ok(Some(action))
            }
            KeyCode::Char('2') => {
                let action = AutonomousControlAction::EnableAnomalyDetection { sensitivity: 0.7 };
                self.control_mode = ControlMode::Normal;
                Ok(Some(action))
            }
            KeyCode::Char('3') => {
                let action = AutonomousControlAction::SetExternalRequestLimit { limit_per_hour: 100 };
                self.control_mode = ControlMode::Normal;
                Ok(Some(action))
            }
            KeyCode::Char('v') => {
                let action = AutonomousControlAction::ConfigureActionValidation { 
                    pre_conditions: true, 
                    post_conditions: true 
                };
                self.control_mode = ControlMode::Normal;
                Ok(Some(action))
            }
            KeyCode::Esc => {
                self.control_mode = ControlMode::Normal;
                Ok(None)
            }
            _ => Ok(None)
        }
    }
    
    fn handle_recursive_control_mode(&mut self, key: KeyCode, _modifiers: KeyModifiers) -> Result<Option<AutonomousControlAction>> {
        match key {
            KeyCode::Char('1') => {
                let action = AutonomousControlAction::SetMaxRecursionDepth { depth: 5 };
                self.control_mode = ControlMode::Normal;
                Ok(Some(action))
            }
            KeyCode::Char('2') => {
                let action = AutonomousControlAction::SetMaxRecursionDepth { depth: 10 };
                self.control_mode = ControlMode::Normal;
                Ok(Some(action))
            }
            KeyCode::Char('p') => {
                let action = AutonomousControlAction::EnablePatternReplication { enabled: true };
                self.control_mode = ControlMode::Normal;
                Ok(Some(action))
            }
            KeyCode::Char('c') => {
                let action = AutonomousControlAction::SetConvergenceThreshold { threshold: 0.85 };
                self.control_mode = ControlMode::Normal;
                Ok(Some(action))
            }
            KeyCode::Char('o') => {
                let action = AutonomousControlAction::OptimizeReasoningTemplates;
                self.control_mode = ControlMode::Normal;
                Ok(Some(action))
            }
            KeyCode::Esc => {
                self.control_mode = ControlMode::Normal;
                Ok(None)
            }
            _ => Ok(None)
        }
    }
    
    fn handle_learning_configuration_mode(&mut self, key: KeyCode, _modifiers: KeyModifiers) -> Result<Option<AutonomousControlAction>> {
        match key {
            KeyCode::Char('+') | KeyCode::Char('=') => {
                let action = AutonomousControlAction::SetLearningRate { new_rate: 0.1 };
                Ok(Some(action))
            }
            KeyCode::Char('-') => {
                let action = AutonomousControlAction::SetLearningRate { new_rate: 0.01 };
                Ok(Some(action))
            }
            KeyCode::Char('m') => {
                let action = AutonomousControlAction::EnableMetaLearning { enabled: true };
                self.control_mode = ControlMode::Normal;
                Ok(Some(action))
            }
            KeyCode::Char('i') => {
                let action = AutonomousControlAction::InitiateKnowledgeIntegration;
                self.control_mode = ControlMode::Normal;
                Ok(Some(action))
            }
            KeyCode::Char('a') => {
                let action = AutonomousControlAction::ConfigureAdaptationSpeed { speed: 0.5 };
                self.control_mode = ControlMode::Normal;
                Ok(Some(action))
            }
            KeyCode::Esc => {
                self.control_mode = ControlMode::Normal;
                Ok(None)
            }
            _ => Ok(None)
        }
    }
    
    fn handle_system_management_mode(&mut self, key: KeyCode, _modifiers: KeyModifiers) -> Result<Option<AutonomousControlAction>> {
        match key {
            KeyCode::Char('s') => {
                let action = AutonomousControlAction::SaveSystemSnapshot;
                self.control_mode = ControlMode::Normal;
                Ok(Some(action))
            }
            KeyCode::Char('e') => {
                let action = AutonomousControlAction::ExportSystemMetrics { format: "json".to_string() };
                self.control_mode = ControlMode::Normal;
                Ok(Some(action))
            }
            KeyCode::Char('r') => {
                let action = AutonomousControlAction::RestartAutonomousLoop;
                self.confirmation_pending = Some(action);
                self.control_mode = ControlMode::Normal;
                Ok(None)
            }
            KeyCode::Char('b') => {
                let action = AutonomousControlAction::ResetToBaseline { preserve_learning: true };
                self.confirmation_pending = Some(action);
                self.control_mode = ControlMode::Normal;
                Ok(None)
            }
            KeyCode::Esc => {
                self.control_mode = ControlMode::Normal;
                Ok(None)
            }
            _ => Ok(None)
        }
    }
    
    /// Update available actions based on current cognitive state
    pub fn update_available_actions(&mut self, cognitive_data: &CognitiveData) {
        self.available_actions.clear();
        
        // Goal management actions
        if cognitive_data.active_goals.len() < 10 {
            self.available_actions.push(AutonomousControlAction::CreateGoal {
                name: "New Strategic Goal".to_string(),
                goal_type: GoalType::Strategic,
                priority: Priority::Medium,
            });
        }
        
        // Check for synergy opportunities
        if cognitive_data.synergy_opportunities.is_empty() && cognitive_data.active_goals.len() > 2 {
            self.available_actions.push(AutonomousControlAction::IdentifySynergyOpportunities);
        }
        
        // Strategic plan actions
        for plan in &cognitive_data.strategic_plans {
            if plan.milestones.iter().any(|m| matches!(m.status, crate::tui::autonomous_data_types::MilestoneStatus::Pending)) {
                self.available_actions.push(AutonomousControlAction::ApproveStrategicPlan {
                    plan_id: plan.id.clone(),
                });
                break;
            }
        }
        
        // Thermodynamic actions
        if cognitive_data.thermodynamic_state.entropy_production_rate > 0.05 {
            self.available_actions.push(AutonomousControlAction::InitiateEntropyReduction);
        }
        
        if cognitive_data.thermodynamic_state.free_energy > 0.7 {
            self.available_actions.push(AutonomousControlAction::OptimizeFreeEnergy);
        }
        
        // Check gradient alignment
        let gradients = &cognitive_data.three_gradient_state;
        if gradients.overall_coherence < 0.7 {
            self.available_actions.push(AutonomousControlAction::BalanceGradients {
                value_weight: 0.33,
                harmony_weight: 0.33,
                intuition_weight: 0.34,
            });
        }
        
        // Safety actions
        if cognitive_data.safety_validation.validation_success_rate < 0.95 {
            self.available_actions.push(AutonomousControlAction::ConfigureSafetyValidation {
                strictness: 0.9,
            });
        }
        
        if cognitive_data.safety_validation.external_requests_filtered > 100 {
            self.available_actions.push(AutonomousControlAction::UpdateFilteringRules {
                rule_type: "adaptive".to_string(),
                parameters: vec!["threshold:0.8".to_string()],
            });
        }
        
        // Learning actions
        if !cognitive_data.learning_architecture.meta_learning_active {
            self.available_actions.push(AutonomousControlAction::EnableMetaLearning { enabled: true });
        }
        
        if cognitive_data.learning_architecture.learning_rate < 0.01 {
            self.available_actions.push(AutonomousControlAction::SetLearningRate { new_rate: 0.05 });
        }
        
        if cognitive_data.meta_learning_insights.len() > 5 {
            self.available_actions.push(AutonomousControlAction::InitiateKnowledgeIntegration);
        }
        
        // Recursive reasoning actions
        if cognitive_data.recursive_processor_status.pattern_discovery_rate > 0.8 {
            self.available_actions.push(AutonomousControlAction::EnablePatternReplication { enabled: true });
        }
        
        if cognitive_data.recursive_processor_status.convergence_success_rate < 0.6 {
            self.available_actions.push(AutonomousControlAction::OptimizeReasoningTemplates);
        }
        
        // Agent coordination actions
        if cognitive_data.agent_coordination.coordination_efficiency < 0.7 {
            self.available_actions.push(AutonomousControlAction::EnableEmergentBehaviorDetection { enabled: true });
        }
        
        // System actions
        self.available_actions.push(AutonomousControlAction::SaveSystemSnapshot);
        
        if cognitive_data.autonomous_loop_status.success_rate < 0.9 {
            self.available_actions.push(AutonomousControlAction::RestartAutonomousLoop);
        }
    }
}

/// Draw the control panel for autonomous systems
pub fn draw_cognitive_control_panel(f: &mut Frame, area: Rect, state: &CognitiveControlState) {
    
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // Mode indicator
            Constraint::Min(5),     // Available actions
            Constraint::Length(3),  // Input area
            Constraint::Length(5),  // Control history
        ])
        .split(area);
    
    // Mode indicator
    let mode_text = match state.control_mode {
        ControlMode::Normal => "Normal - [G]oal [T]hermo [A]gent [R]gradient [S]afety [D]epth [L]earn [M]anage",
        ControlMode::GoalCreation => "Goal Creation Mode - Enter goal name",
        ControlMode::GradientAdjustment => "Gradient Adjustment - Press 1/2/3 for V/H/I balance presets",
        ControlMode::AgentAssignment => "Agent Assignment Mode - Select agent and goal",
        ControlMode::ThermodynamicTuning => "Thermodynamic - [+/-] entropy [R]educe [O]ptimize",
        ControlMode::SafetyConfiguration => "Safety Config - [1]Strictness [2]Anomaly [3]Limits [V]alidation",
        ControlMode::RecursiveControl => "Recursive - [1/2]Depth [P]attern [C]onverge [O]ptimize",
        ControlMode::LearningConfiguration => "Learning - [+/-]Rate [M]eta [I]ntegrate [A]dapt",
        ControlMode::SystemManagement => "System - [S]ave [E]xport [R]estart [B]aseline",
    };
    
    let mode_widget = Paragraph::new(mode_text)
        .block(Block::default()
            .borders(Borders::ALL)
            .title("Control Mode")
            .border_style(Style::default().fg(Color::Cyan)))
        .style(Style::default().fg(Color::Cyan))
        .alignment(Alignment::Center);
    f.render_widget(mode_widget, chunks[0]);
    
    // Available actions
    let action_items: Vec<ListItem> = state.available_actions.iter().enumerate().map(|(i, action)| {
        let selected = i == state.selected_action;
        let style = if selected {
            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::White)
        };
        
        let text = match action {
            // Goal Management
            AutonomousControlAction::CreateGoal { name, goal_type, priority } => 
                format!("📝 Create {} Goal: {} [{}]", 
                    match goal_type {
                        GoalType::Strategic => "Strategic",
                        GoalType::Tactical => "Tactical",
                        GoalType::Operational => "Operational",
                        GoalType::Learning => "Learning",
                        GoalType::Maintenance => "Maintenance",
                        GoalType::Safety => "Safety",
                    },
                    name,
                    match priority {
                        Priority::Critical => "CRITICAL",
                        Priority::High => "HIGH",
                        Priority::Medium => "MED",
                        Priority::Low => "LOW",
                    }
                ),
            AutonomousControlAction::ModifyGoalPriority { goal_id, new_priority } => 
                format!("🎯 Change Goal {} Priority to {:?}", goal_id, new_priority),
            AutonomousControlAction::ApproveStrategicPlan { plan_id } => 
                format!("✅ Approve Strategic Plan: {}", plan_id),
            AutonomousControlAction::IdentifySynergyOpportunities => 
                "🔍 Identify Goal Synergy Opportunities".to_string(),
                
            // Thermodynamic Control
            AutonomousControlAction::InitiateEntropyReduction => 
                "🔥 Initiate Entropy Reduction Process".to_string(),
            AutonomousControlAction::OptimizeFreeEnergy => 
                "⚡ Optimize Free Energy Minimization".to_string(),
            AutonomousControlAction::BalanceGradients { value_weight, harmony_weight, intuition_weight } => 
                format!("⚖️ Balance Gradients: V:{:.0}% H:{:.0}% I:{:.0}%", 
                    value_weight * 100.0, harmony_weight * 100.0, intuition_weight * 100.0),
            AutonomousControlAction::SetEntropyThreshold { warning, critical } => 
                format!("🌡️ Set Entropy Thresholds: Warn:{:.2} Crit:{:.2}", warning, critical),
                
            // Learning Control
            AutonomousControlAction::EnableMetaLearning { enabled } => 
                format!("🧠 {} Meta-Learning", if *enabled { "Enable" } else { "Disable" }),
            AutonomousControlAction::SetLearningRate { new_rate } => 
                format!("📈 Set Learning Rate to {:.3}", new_rate),
            AutonomousControlAction::InitiateKnowledgeIntegration => 
                "🔗 Initiate Knowledge Integration Process".to_string(),
                
            // Safety Control
            AutonomousControlAction::ConfigureSafetyValidation { strictness } => 
                format!("🛡️ Set Safety Validation Strictness: {:.0}%", strictness * 100.0),
            AutonomousControlAction::EnableAnomalyDetection { sensitivity } => 
                format!("🚨 Enable Anomaly Detection (Sensitivity: {:.0}%)", sensitivity * 100.0),
            AutonomousControlAction::UpdateFilteringRules { rule_type, .. } => 
                format!("🔐 Update {} Filtering Rules", rule_type),
                
            // Recursive Reasoning
            AutonomousControlAction::EnablePatternReplication { enabled } => 
                format!("🔄 {} Pattern Replication", if *enabled { "Enable" } else { "Disable" }),
            AutonomousControlAction::OptimizeReasoningTemplates => 
                "🎯 Optimize Reasoning Templates".to_string(),
            AutonomousControlAction::SetMaxRecursionDepth { depth } => 
                format!("📊 Set Max Recursion Depth to {}", depth),
                
            // Agent Coordination
            AutonomousControlAction::EnableEmergentBehaviorDetection { enabled } => 
                format!("✨ {} Emergent Behavior Detection", if *enabled { "Enable" } else { "Disable" }),
            AutonomousControlAction::ConfigureConsensusThreshold { threshold } => 
                format!("🤝 Set Consensus Threshold: {:.0}%", threshold * 100.0),
                
            // System Control
            AutonomousControlAction::SaveSystemSnapshot => 
                "💾 Save System Snapshot".to_string(),
            AutonomousControlAction::RestartAutonomousLoop => 
                "🔄 Restart Autonomous Loop".to_string(),
            AutonomousControlAction::ExportSystemMetrics { format } => 
                format!("📊 Export System Metrics ({})", format),
            AutonomousControlAction::ResetToBaseline { preserve_learning } => 
                format!("🔧 Reset to Baseline ({})", 
                    if *preserve_learning { "Preserve Learning" } else { "Full Reset" }),
                    
            _ => format!("{:?}", action),
        };
        
        ListItem::new(text).style(style)
    }).collect();
    
    let actions_list = List::new(action_items)
        .block(Block::default()
            .borders(Borders::ALL)
            .title("Available Actions")
            .border_style(Style::default().fg(Color::Green)));
    f.render_widget(actions_list, chunks[1]);
    
    // Input area / Confirmation
    if let Some(pending) = &state.confirmation_pending {
        let confirm_text = vec![
            Line::from(vec![
                Span::styled("Confirm action: ", Style::default().fg(Color::Yellow)),
                Span::raw(format!("{:?}", pending)),
            ]),
            Line::from(""),
            Line::from("Press ENTER to confirm, ESC to cancel"),
        ];
        
        let confirm_widget = Paragraph::new(confirm_text)
            .block(Block::default()
                .borders(Borders::ALL)
                .title("Confirmation Required")
                .border_style(Style::default().fg(Color::Yellow)))
            .alignment(Alignment::Center);
        f.render_widget(confirm_widget, chunks[2]);
    } else if state.control_mode == ControlMode::GoalCreation {
        let input_widget = Paragraph::new(state.input_buffer.as_str())
            .block(Block::default()
                .borders(Borders::ALL)
                .title("Goal Name")
                .border_style(Style::default().fg(Color::Cyan)))
            .style(Style::default().fg(Color::White));
        f.render_widget(input_widget, chunks[2]);
    } else {
        let help_widget = Paragraph::new("↑/↓ Navigate | ENTER Select | ESC Cancel")
            .block(Block::default()
                .borders(Borders::ALL)
                .title("Controls")
                .border_style(Style::default().fg(Color::Gray)))
            .alignment(Alignment::Center);
        f.render_widget(help_widget, chunks[2]);
    }
    
    // Control history
    let history_items: Vec<ListItem> = state.control_history.iter().rev().take(3).map(|(timestamp, action, success)| {
        let time_str = timestamp.format("%H:%M:%S").to_string();
        let status = if *success { "✓" } else { "✗" };
        let color = if *success { Color::Green } else { Color::Red };
        
        ListItem::new(Line::from(vec![
            Span::styled(time_str, Style::default().fg(Color::Gray)),
            Span::raw(" "),
            Span::styled(status, Style::default().fg(color)),
            Span::raw(" "),
            Span::raw(format!("{:?}", action)),
        ]))
    }).collect();
    
    let history_list = List::new(history_items)
        .block(Block::default()
            .borders(Borders::ALL)
            .title("Recent Actions")
            .border_style(Style::default().fg(Color::Blue)));
    f.render_widget(history_list, chunks[3]);
}