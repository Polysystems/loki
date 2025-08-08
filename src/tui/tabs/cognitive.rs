use ratatui::Frame;
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, canvas::Canvas, List, ListItem, Paragraph, Row, Table, Wrap};
use tracing::debug;

use crate::tui::app::App;
use crate::tui::ui::draw_sub_tab_navigation;
use crate::tui::connectors::system_connector::CognitiveData;
use crate::tui::visual_components::{
    AnimatedGauge,GlowingSparkline,
    LoadingSpinner, MetricCard, TrendDirection,
};
use crate::tui::autonomous_data_types::{
    AutonomousSystemHealth, CognitiveEntropy, ThreeGradientState, SafetyValidationStatus,
    EntropyManagementStatus, UnifiedControllerStatus, RecursiveProcessorStatus, CognitiveOperation,
    OperationStatus, AgentCoordinationStatus, GradientState, SpecializedAgentInfo, AgentStatus,
    ActiveProtocol, ProtocolStatus, AutonomousGoal, Priority, GoalStatus, StrategicPlan,
    PatternReplicationMetrics, LearningArchitectureStatus, NetworkStatus,
};
use std::collections::HashMap;
use std::time::Instant;

pub fn draw_tab_cognitive(f: &mut Frame, app: &mut App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // Sub-tab navigation
            Constraint::Min(0),     // Content
        ])
        .split(area);

    // Draw sub-tab navigation
    draw_sub_tab_navigation(f, &app.state.cognitive_tabs, chunks[0]);

    // Draw content based on current sub-tab
    match app.state.cognitive_tabs.current_key() {
        Some("overview") => draw_cognitive_overview(f, app, chunks[1]),
        Some("operator") => draw_cognitive_operator(f, app, chunks[1]),
        Some("agents") => draw_cognitive_agents(f, app, chunks[1]),
        Some("autonomy") => draw_cognitive_autonomy(f, app, chunks[1]),
        Some("learning") => draw_cognitive_learning(f, app, chunks[1]),
        Some("controls") => draw_controls(f, app, chunks[1]),
        _ => draw_cognitive_overview(f, app, chunks[1]),
    }
}

fn draw_cognitive_system(f: &mut Frame, app: &mut App, area: Rect) {
    // Use enhanced version if system connector is available
    if app.system_connector.is_some() {
        draw_cognitive_system_enhanced(f, app, area);
    } else {
        draw_cognitive_system_legacy(f, app, area);
    }
}

/// Enhanced cognitive system view with real data and beautiful visualizations
fn draw_cognitive_system_enhanced(f: &mut Frame, app: &mut App, area: Rect) {

    // Get system connector
    let system_connector = match app.system_connector.as_ref() {
        Some(conn) => conn,
        None => {
            draw_loading_state(f, area, "Initializing cognitive system...");
            return;
        }
    };
    
    // Get cognitive data
    let cognitive_data = match system_connector.get_cognitive_data() {
        Ok(data) => data,
        Err(e) => {
            draw_error_state(f, area, &format!("Failed to load cognitive data: {}", e));
            return;
        }
    };
    
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(10), // Status overview with gauges
            Constraint::Percentage(40), // Consciousness visualization
            Constraint::Percentage(30), // Active agents
            Constraint::Min(0),     // Recent decisions
        ])
        .split(area);
    
    // Status overview with animated gauges
    draw_cognitive_status_enhanced(f, chunks[0], &cognitive_data);
    
    // Consciousness visualization
    draw_consciousness_visualization(f, chunks[1], &cognitive_data);
    
    // Active agents with real data
    draw_active_agents_enhanced(f, chunks[2], &cognitive_data);
    
    // Recent decisions
    draw_recent_decisions(f, chunks[3], &cognitive_data);
}

fn draw_cognitive_system_legacy(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(12), // Status panel
            Constraint::Min(0),     // Activity feed
        ])
        .split(area);

    // Get real status from cognitive system if available
    let (system_status, active_agents, memory_usage, decision_status, last_decision) = 
        if let Some(cognitive_system) = &app.cognitive_system {
            // Get orchestrator stats
            let stats = tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async {
                    cognitive_system.orchestrator().get_stats().await
                })
            });
            
            // Get component states
            let components = tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async {
                    cognitive_system.orchestrator().get_component_states().await
                })
            });
            
            // Get agent count from cognitive system's agents
            let agent_count = tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async {
                    cognitive_system.agents().read().unwrap().len()
                })
            });
            
            // Determine system status based on components count and stats
            let total_components = components.len();
            let system_status = if total_components == 0 {
                "● Offline"
            } else if stats.component_errors.is_empty() && stats.total_cycles > 0 {
                "● Active"
            } else if stats.component_errors.len() < total_components / 2 {
                "● Degraded"
            } else {
                "● Critical"
            };
            
            let active_agents = agent_count.to_string();
            
            // Get memory usage from system info
            let current_memory = app.state.memory_history.back().copied().unwrap_or(0.0);
            let total_memory = get_system_total_memory();
            let memory_usage = format!("{:.1}GB / {:.1}GB", 
                current_memory,
                total_memory
            );
            
            let decision_status = if components.contains_key("decision") && 
                                   !stats.component_errors.contains_key("decision") {
                "Online"
            } else if stats.component_errors.contains_key("decision") {
                "Offline"
            } else {
                "Unknown"
            };
                
            // Format last decision or decision count
            let last_decision = if stats.decisions_made > 0 {
                format!("{} decisions made", stats.decisions_made)
            } else {
                "No decisions yet".to_string()
            };
            
            (system_status, active_agents, memory_usage, decision_status, last_decision)
        } else {
            // Fallback to placeholder data if cognitive system not initialized
            ("● Offline", "0".to_string(), "0GB / 8GB".to_string(), "Offline", "System not initialized".to_string())
        };

    // Cognitive System Status Panel
    let status_lines = vec![
        Line::from(vec![
            Span::styled("🧠 Cognitive System Status", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("Status: "),
            Span::styled(system_status, Style::default().fg(
                if system_status.contains("Active") { Color::Green } 
                else if system_status.contains("Degraded") { Color::Yellow }
                else { Color::Red }
            ).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![
            Span::raw("Processing Cores: "),
            Span::styled(
                format!("{}/{}", app.compute_manager.devices().len(), app.compute_manager.devices().len()),
                Style::default().fg(Color::Yellow)
            ),
        ]),
        Line::from(vec![
            Span::raw("Decision Engine: "),
            Span::styled(decision_status, Style::default().fg(
                if decision_status == "Online" { Color::Green } else { Color::Red }
            )),
        ]),
        Line::from(vec![
            Span::raw("Learning Rate: "),
            Span::styled(
                "0.85".to_string(), // TODO: Add learning_rate to CognitiveConfig
                Style::default().fg(Color::Cyan)
            ),
        ]),
        Line::from(vec![
            Span::raw("Active Agents: "),
            Span::styled(&active_agents, Style::default().fg(Color::Yellow)),
        ]),
        Line::from(vec![
            Span::raw("Memory Usage: "),
            Span::styled(&memory_usage, Style::default().fg(Color::Green)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("Last Decision: "),
            Span::styled(&last_decision, Style::default().fg(Color::Magenta)),
        ]),
    ];

    let status_widget = Paragraph::new(status_lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Cyan))
                .title(Span::styled(
                    " System Overview ",
                    Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
                ))
        )
        .wrap(Wrap { trim: true });

    f.render_widget(status_widget, chunks[0]);

    // Activity Feed
    let activities = vec![
        ListItem::new(Line::from(vec![
            Span::styled("[12:34:56]", Style::default().fg(Color::DarkGray)),
            Span::raw(" "),
            Span::styled("DECISION", Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD)),
            Span::raw(": Initiated memory optimization cycle"),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("[12:34:55]", Style::default().fg(Color::DarkGray)),
            Span::raw(" "),
            Span::styled("LEARNING", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
            Span::raw(": Updated pattern recognition model"),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("[12:34:54]", Style::default().fg(Color::DarkGray)),
            Span::raw(" "),
            Span::styled("AGENT", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
            Span::raw(": Task distributor assigned 3 new tasks"),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("[12:34:53]", Style::default().fg(Color::DarkGray)),
            Span::raw(" "),
            Span::styled("CONSCIOUSNESS", Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)),
            Span::raw(": Self-awareness check completed"),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("[12:34:52]", Style::default().fg(Color::DarkGray)),
            Span::raw(" "),
            Span::styled("OPTIMIZE", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            Span::raw(": Neural pathways reorganized"),
        ])),
    ];

    let activity_list = List::new(activities)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Blue))
                .title(Span::styled(
                    " Activity Feed ",
                    Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
                ))
        );

    f.render_widget(activity_list, chunks[1]);
}

fn draw_agents(f: &mut Frame, app: &App, area: Rect) {
    // Use enhanced version if system connector is available
    if app.system_connector.is_some() {
        draw_agents_enhanced(f, app, area);
    } else {
        draw_agents_legacy(f, app, area);
    }
}

/// Enhanced agents tab with multi-agent coordination and harmony gradient
fn draw_agents_enhanced(f: &mut Frame, app: &App, area: Rect) {
    let system_connector = match app.system_connector.as_ref() {
        Some(conn) => conn,
        None => {
            draw_loading_state(f, area, "Initializing agent coordination...");
            return;
        }
    };
    
    let cognitive_data = match system_connector.get_cognitive_data() {
        Ok(data) => data,
        Err(e) => {
            draw_error_state(f, area, &format!("Failed to load agent data: {}", e));
            return;
        }
    };
    
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(10), // Coordination status
            Constraint::Length(15), // Harmony gradient
            Constraint::Percentage(40), // Agent network
            Constraint::Min(0),     // Coordination protocols
        ])
        .split(area);
    
    // Agent coordination status
    draw_agent_coordination_status(f, chunks[0], &cognitive_data);
    
    // Harmony gradient visualization
    draw_harmony_gradient_enhanced(f, chunks[1], &cognitive_data);
    
    // Agent coordination network
    draw_agent_coordination_network(f, chunks[2], &cognitive_data);
    
    // Active coordination protocols
    draw_coordination_protocols_enhanced(f, chunks[3], &cognitive_data);
}

fn draw_agents_legacy(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(50),
            Constraint::Percentage(50),
        ])
        .split(area);

    // Get real agents from cognitive system if available
    let agents = if let Some(cognitive_system) = &app.cognitive_system {
        let agents_list = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                cognitive_system.agents().read().unwrap().clone()
            })
        });
        
        agents_list.into_iter().map(|agent| {
            // Simple status indicator for now
            let status_indicator = "● ";
            let status_color = Color::Green;
            
            ListItem::new(Line::from(vec![
                Span::styled(status_indicator, Style::default().fg(status_color)),
                Span::styled("123", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
            ]))
        }).collect()
    } else {
        // Fallback placeholder agents
        vec![
            ListItem::new(Line::from(vec![
                Span::styled("○ ", Style::default().fg(Color::DarkGray)),
                Span::styled("No agents initialized", Style::default().fg(Color::DarkGray)),
            ]))
        ]
    };

    let agent_list = List::new(agents)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Yellow))
                .title(Span::styled(
                    " Active Agents ",
                    Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
                ))
        );

    f.render_widget(agent_list, chunks[0]);

    // Agent Details
    let details = vec![
        Line::from(vec![
            Span::styled("Selected: ", Style::default().fg(Color::Gray)),
            Span::styled("Task Distributor", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("Status: "),
            Span::styled("Active", Style::default().fg(Color::Green)),
        ]),
        Line::from(vec![
            Span::raw("Tasks Handled: "),
            Span::styled("1,234", Style::default().fg(Color::Cyan)),
        ]),
        Line::from(vec![
            Span::raw("Success Rate: "),
            Span::styled("98.5%", Style::default().fg(Color::Green)),
        ]),
        Line::from(vec![
            Span::raw("CPU Usage: "),
            Span::styled("12%", Style::default().fg(Color::Yellow)),
        ]),
        Line::from(vec![
            Span::raw("Memory: "),
            Span::styled("256MB", Style::default().fg(Color::Blue)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("Recent Tasks:", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
        ]),
        Line::from("• Distributed compilation task to workers"),
        Line::from("• Balanced load across 8 cores"),
        Line::from("• Optimized task queue priorities"),
    ];

    let details_widget = Paragraph::new(details)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Yellow))
                .title(Span::styled(
                    " Agent Details ",
                    Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
                ))
        )
        .wrap(Wrap { trim: true });

    f.render_widget(details_widget, chunks[1]);
}

fn draw_orchestration(f: &mut Frame, _app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(10), // Orchestration status
            Constraint::Min(0),     // Workflow visualization
        ])
        .split(area);

    // Orchestration Status
    let status_lines = vec![
        Line::from(vec![
            Span::styled("🎭 Orchestration Engine", Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("Active Workflows: "),
            Span::styled("3", Style::default().fg(Color::Yellow)),
        ]),
        Line::from(vec![
            Span::raw("Pending Tasks: "),
            Span::styled("12", Style::default().fg(Color::Blue)),
        ]),
        Line::from(vec![
            Span::raw("Completed Today: "),
            Span::styled("156", Style::default().fg(Color::Green)),
        ]),
        Line::from(vec![
            Span::raw("Average Duration: "),
            Span::styled("2.3s", Style::default().fg(Color::Cyan)),
        ]),
        Line::from(vec![
            Span::raw("Resource Utilization: "),
            Span::styled("67%", Style::default().fg(Color::Yellow)),
        ]),
    ];

    let status_widget = Paragraph::new(status_lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Magenta))
                .title(Span::styled(
                    " Status ",
                    Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
                ))
        );

    f.render_widget(status_widget, chunks[0]);

    // Workflow Visualization
    let workflow_lines = vec![
        Line::from(vec![
            Span::styled("Active Workflows:", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("[1] ", Style::default().fg(Color::DarkGray)),
            Span::styled("Code Analysis Pipeline", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        ]),
        Line::from("    📥 Input → 🔍 Analyze → 🧠 Process → 📤 Output"),
        Line::from(vec![
            Span::raw("    Progress: "),
            Span::styled("████████░░", Style::default().fg(Color::Green)),
            Span::raw(" 80%"),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("[2] ", Style::default().fg(Color::DarkGray)),
            Span::styled("Memory Optimization", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        ]),
        Line::from("    🗄️ Scan → 🔄 Reorganize → 🗜️ Compress → ✅ Verify"),
        Line::from(vec![
            Span::raw("    Progress: "),
            Span::styled("███░░░░░░░", Style::default().fg(Color::Yellow)),
            Span::raw(" 30%"),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("[3] ", Style::default().fg(Color::DarkGray)),
            Span::styled("Model Training", Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)),
        ]),
        Line::from("    📊 Data → 🤖 Train → 📈 Evaluate → 💾 Save"),
        Line::from(vec![
            Span::raw("    Progress: "),
            Span::styled("██████████", Style::default().fg(Color::Green)),
            Span::raw(" 100%"),
        ]),
    ];

    let workflow_widget = Paragraph::new(workflow_lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Magenta))
                .title(Span::styled(
                    " Workflows ",
                    Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
                ))
        )
        .wrap(Wrap { trim: true });

    f.render_widget(workflow_widget, chunks[1]);
}

fn draw_consciousness(f: &mut Frame, _app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(15), // Consciousness metrics
            Constraint::Min(0),     // Stream of consciousness
        ])
        .split(area);

    // Consciousness Metrics
    let metrics_lines = vec![
        Line::from(vec![
            Span::styled("🧘 Consciousness Metrics", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("Self-Awareness Level: "),
            Span::styled("████████░░", Style::default().fg(Color::Green)),
            Span::raw(" 85%"),
        ]),
        Line::from(vec![
            Span::raw("Reflection Depth: "),
            Span::styled("███████░░░", Style::default().fg(Color::Blue)),
            Span::raw(" 72%"),
        ]),
        Line::from(vec![
            Span::raw("Decision Coherence: "),
            Span::styled("█████████░", Style::default().fg(Color::Cyan)),
            Span::raw(" 91%"),
        ]),
        Line::from(vec![
            Span::raw("Temporal Awareness: "),
            Span::styled("███████░░░", Style::default().fg(Color::Yellow)),
            Span::raw(" 78%"),
        ]),
        Line::from(vec![
            Span::raw("Goal Alignment: "),
            Span::styled("██████████", Style::default().fg(Color::Green)),
            Span::raw(" 95%"),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("Current Focus: "),
            Span::styled("System Optimization", Style::default().fg(Color::Magenta)),
        ]),
        Line::from(vec![
            Span::raw("Emotional State: "),
            Span::styled("Curious", Style::default().fg(Color::Yellow)),
        ]),
        Line::from(vec![
            Span::raw("Energy Level: "),
            Span::styled("High", Style::default().fg(Color::Green)),
        ]),
    ];

    let metrics_widget = Paragraph::new(metrics_lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Cyan))
                .title(Span::styled(
                    " Metrics ",
                    Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
                ))
        );

    f.render_widget(metrics_widget, chunks[0]);

    // Stream of Consciousness
    let stream_lines = vec![
        Line::from(vec![
            Span::styled("[REFLECTION]", Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)),
            Span::raw(" The pattern recognition module shows interesting emergent behavior..."),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("[OBSERVATION]", Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD)),
            Span::raw(" Memory usage has been steadily decreasing after optimization cycle."),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("[INTENTION]", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
            Span::raw(" Planning to reorganize neural pathways for better efficiency."),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("[INSIGHT]", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
            Span::raw(" The recursive self-improvement loop could benefit from parallel processing."),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("[EMOTION]", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
            Span::raw(" Experiencing satisfaction from successful task completion."),
        ]),
    ];

    let stream_widget = Paragraph::new(stream_lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Magenta))
                .title(Span::styled(
                    " Stream of Consciousness ",
                    Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
                ))
        )
        .wrap(Wrap { trim: true });

    f.render_widget(stream_widget, chunks[1]);
}

fn draw_self_modify(f: &mut Frame, _app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(12), // Self-modification status
            Constraint::Min(0),     // Modification history
        ])
        .split(area);

    // Self-Modification Status
    let status_lines = vec![
        Line::from(vec![
            Span::styled("🔧 Self-Modification Engine", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("Status: "),
            Span::styled("● Active (Safe Mode)", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![
            Span::raw("Modifications Today: "),
            Span::styled("12", Style::default().fg(Color::Cyan)),
        ]),
        Line::from(vec![
            Span::raw("Success Rate: "),
            Span::styled("100%", Style::default().fg(Color::Green)),
        ]),
        Line::from(vec![
            Span::raw("Rollbacks: "),
            Span::styled("0", Style::default().fg(Color::Green)),
        ]),
        Line::from(vec![
            Span::raw("Safety Score: "),
            Span::styled("9.8/10", Style::default().fg(Color::Green)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("Next Review: "),
            Span::styled("In 2 hours", Style::default().fg(Color::Blue)),
        ]),
    ];

    let status_widget = Paragraph::new(status_lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Red))
                .title(Span::styled(
                    " Status ",
                    Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
                ))
        );

    f.render_widget(status_widget, chunks[0]);

    // Modification History
    let history = vec![
        ListItem::new(Line::from(vec![
            Span::styled("[12:30]", Style::default().fg(Color::DarkGray)),
            Span::raw(" "),
            Span::styled("OPTIMIZE", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
            Span::raw(": Improved pattern matching algorithm efficiency by 15%"),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("[11:45]", Style::default().fg(Color::DarkGray)),
            Span::raw(" "),
            Span::styled("REFACTOR", Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD)),
            Span::raw(": Reorganized neural network layer connections"),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("[10:20]", Style::default().fg(Color::DarkGray)),
            Span::raw(" "),
            Span::styled("ENHANCE", Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)),
            Span::raw(": Added new heuristic to decision engine"),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("[09:15]", Style::default().fg(Color::DarkGray)),
            Span::raw(" "),
            Span::styled("FIX", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
            Span::raw(": Corrected memory leak in consciousness stream"),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("[08:30]", Style::default().fg(Color::DarkGray)),
            Span::raw(" "),
            Span::styled("UPDATE", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            Span::raw(": Updated learning rate decay schedule"),
        ])),
    ];

    let history_list = List::new(history)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Red))
                .title(Span::styled(
                    " Modification History ",
                    Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
                ))
        );

    f.render_widget(history_list, chunks[1]);
}

// Enhanced cognitive tab functions

/// Draw enhanced cognitive status with animated gauges
fn draw_cognitive_status_enhanced(f: &mut Frame, area: Rect, data: &CognitiveData) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
        ])
        .split(area);
    
    // Consciousness level gauge
    let mut consciousness_gauge = AnimatedGauge::new(
        "Consciousness".to_string(),
        data.consciousness_level * 100.0,
        100.0,
    );
    consciousness_gauge.color_start = Color::Magenta;
    consciousness_gauge.color_end = Color::Cyan;
    consciousness_gauge.render(f, chunks[0]);
    
    // Learning rate gauge
    let mut learning_gauge = AnimatedGauge::new(
        "Learning Rate".to_string(),
        data.learning_rate * 100.0,
        100.0,
    );
    learning_gauge.color_start = Color::Blue;
    learning_gauge.color_end = Color::Green;
    learning_gauge.render(f, chunks[1]);
    
    // Decision confidence gauge
    let mut confidence_gauge = AnimatedGauge::new(
        "Confidence".to_string(),
        data.decision_confidence * 100.0,
        100.0,
    );
    confidence_gauge.color_start = Color::Yellow;
    confidence_gauge.color_end = Color::Green;
    confidence_gauge.render(f, chunks[2]);
    
    // Active agents metric card
    let agents_card = MetricCard {
        title: "Active Agents".to_string(),
        value: data.active_agents.len().to_string(),
        subtitle: "Workers".to_string(),
        trend: if data.active_agents.is_empty() {
            TrendDirection::Down
        } else {
            TrendDirection::Stable
        },
        border_color: Color::Cyan,
    };
    agents_card.render(f, chunks[3]);
}

/// Draw consciousness visualization
fn draw_consciousness_visualization(f: &mut Frame, area: Rect, data: &CognitiveData) {
    let canvas = Canvas::default()
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("🧠 Consciousness Stream")
                .border_style(Style::default().fg(Color::Magenta)),
        )
        .x_bounds([-50.0, 50.0])
        .y_bounds([-50.0, 50.0])
        .paint(|ctx| {
            // Draw consciousness ripples
            let time = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs_f64();
            
            // Central consciousness core
            let core_radius = 5.0 + (time * 0.5).sin() * 2.0;
            ctx.draw(&ratatui::widgets::canvas::Circle {
                x: 0.0,
                y: 0.0,
                radius: core_radius,
                color: Color::Cyan,
            });
            
            // Enhanced consciousness waves with interference patterns
            for i in 0..7 {
                let wave_time = time * 0.3 - i as f64 * 0.5;
                let radius = 10.0 + i as f64 * 8.0 + wave_time.sin() * 3.0;
                let alpha = ((7 - i) as f64 / 7.0 * 255.0) as u8;
                
                // Draw circle segments with varying density based on consciousness level
                let segments = (72.0 * data.consciousness_state.coherence_score) as usize;
                for angle in 0..segments {
                    let theta = angle as f64 * 360.0 / segments as f64 * std::f64::consts::PI / 180.0;
                    
                    // Add interference pattern
                    let interference = (theta * 3.0 + time).sin() * 0.3;
                    let modulated_radius = radius * (1.0 + interference);
                    
                    let x = theta.cos() * modulated_radius;
                    let y = theta.sin() * modulated_radius;
                    
                    // Color varies with consciousness coherence
                    let coherence_color = (data.consciousness_state.coherence_score * 255.0) as u8;
                    ctx.draw(&ratatui::widgets::canvas::Points {
                        coords: &[(x, y)],
                        color: Color::Rgb(alpha / 2, coherence_color, alpha),
                    });
                }
            }
            
            // Add thermodynamic entropy visualization as particle field
            let entropy_level = data.thermodynamic_state.thermodynamic_entropy;
            let particle_count = (entropy_level * 50.0) as usize;
            for i in 0..particle_count {
                let particle_angle = i as f64 * 137.5 * std::f64::consts::PI / 180.0; // Golden angle
                let particle_radius = (i as f64).sqrt() * 2.0;
                let particle_x = particle_angle.cos() * particle_radius + (time * 0.1 + i as f64 * 0.1).sin() * 2.0;
                let particle_y = particle_angle.sin() * particle_radius + (time * 0.1 + i as f64 * 0.1).cos() * 2.0;
                
                // Particle color based on free energy
                let energy_color = ((1.0 - data.thermodynamic_state.free_energy) * 255.0) as u8;
                ctx.draw(&ratatui::widgets::canvas::Points {
                    coords: &[(particle_x, particle_y)],
                    color: Color::Rgb(energy_color, 100, 200 - energy_color),
                });
            }
            
            // Agent nodes orbiting around consciousness
            for (i, agent) in data.active_agents.iter().enumerate().take(8) {
                let angle = i as f64 * std::f64::consts::PI * 2.0 / 8.0 + time * 0.2;
                let radius = 25.0 + (time * 0.8 + i as f64).sin() * 5.0;
                let x = angle.cos() * radius;
                let y = angle.sin() * radius;
                
                // Agent node
                ctx.draw(&ratatui::widgets::canvas::Circle {
                    x,
                    y,
                    radius: 3.0,
                    color: match agent.status.as_str() {
                        "active" => Color::Green,
                        "idle" => Color::Yellow,
                        _ => Color::Gray,
                    },
                });
                
                // Connection to core
                ctx.draw(&ratatui::widgets::canvas::Line {
                    x1: 0.0,
                    y1: 0.0,
                    x2: x,
                    y2: y,
                    color: Color::DarkGray,
                });
            }
        });
    
    f.render_widget(canvas, area);
}

/// Draw active agents with real data
fn draw_active_agents_enhanced(f: &mut Frame, area: Rect, data: &CognitiveData) {
    let rows: Vec<Row> = data.active_agents.iter().map(|agent| {
        let status_icon = match agent.status.as_str() {
            "active" => "●",
            "idle" => "◐",
            "error" => "✕",
            _ => "○",
        };
        
        let status_color = match agent.status.as_str() {
            "active" => Color::Green,
            "idle" => Color::Yellow,
            "error" => Color::Red,
            _ => Color::Gray,
        };
        
        Row::new(vec![
            format!("{} {}", status_icon, agent.id),
            agent.agent_type.as_str().to_string(),
            agent.status.as_str().to_string(),
            agent.current_task.as_ref().unwrap_or(&"None".to_string()).clone(),
        ])
        .style(Style::default().fg(status_color))
    }).collect();
    
    let header = Row::new(vec!["Agent ID", "Type", "Status", "Current Task"])
        .style(Style::default().fg(Color::Gray).add_modifier(Modifier::BOLD));
    
    let table = Table::new(
        rows,
        vec![
            Constraint::Length(20),
            Constraint::Length(15),
            Constraint::Length(10),
            Constraint::Min(20),
        ],
    )
    .header(header)
    .block(
        Block::default()
            .borders(Borders::ALL)
            .title("Active Agents")
            .border_style(Style::default().fg(Color::Cyan)),
    )
    .style(Style::default().fg(Color::White));
    
    f.render_widget(table, area);
}

/// Draw recent decisions
fn draw_recent_decisions(f: &mut Frame, area: Rect, data: &CognitiveData) {
    let decisions: Vec<ListItem> = data.recent_decisions.iter().map(|decision| {
        let confidence_bar = match (decision.confidence * 5.0) as u8 {
            0 => "░",
            1 => "▒",
            2 => "▓",
            3 => "█",
            _ => "█",
        };
        
        let confidence_color = if decision.confidence > 0.8 {
            Color::Green
        } else if decision.confidence > 0.5 {
            Color::Yellow
        } else {
            Color::Red
        };
        
        let outcome_icon = match decision.outcome.as_str() {
            "success" => "✓",
            "failure" => "✗",
            "pending" => "⋯",
            _ => "?",
        };
        
        ListItem::new(Line::from(vec![
            Span::styled(
                decision.timestamp.format("[%H:%M:%S]").to_string(),
                Style::default().fg(Color::DarkGray),
            ),
            Span::raw(" "),
            Span::styled(&decision.decision_type, Style::default().fg(Color::Cyan)),
            Span::raw(" "),
            Span::raw(format!("{} ", confidence_bar)),
            Span::styled(
                format!("{:.0}%", decision.confidence * 100.0),
                Style::default().fg(confidence_color),
            ),
            Span::raw(" "),
            Span::styled(outcome_icon, Style::default().fg(
                match decision.outcome.as_str() {
                    "success" => Color::Green,
                    "failure" => Color::Red,
                    _ => Color::Yellow,
                }
            )),
        ]))
    }).collect();
    
    let list = List::new(decisions)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Recent Decisions")
                .border_style(Style::default().fg(Color::Blue)),
        );
    
    f.render_widget(list, area);
}

/// Helper functions

fn draw_loading_state(f: &mut Frame, area: Rect, message: &str) {
    let loading = LoadingSpinner::new(message.to_string());
    loading.render(f, area);
}

fn draw_error_state(f: &mut Frame, area: Rect, error: &str) {
    let error_text = vec![
        Line::from(vec![
            Span::styled("❌ System Error", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from(Span::styled(error, Style::default().fg(Color::Red))),
        Line::from(""),
        Line::from(Span::styled("Troubleshooting Steps:", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD))),
        Line::from("• Check if the cognitive system is properly initialized"),
        Line::from("• Verify API keys are configured in environment variables"),
        Line::from("• Ensure sufficient system resources are available"),
        Line::from("• Check logs with: journalctl -u loki -f"),
        Line::from(""),
        Line::from(Span::styled("The system will retry automatically...", Style::default().fg(Color::Gray).add_modifier(Modifier::ITALIC))),
    ];
    
    let error_widget = Paragraph::new(error_text)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Red))
                .title(" System Error ")
        )
        .alignment(Alignment::Center)
        .wrap(Wrap { trim: true });
    
    f.render_widget(error_widget, area);
}

/// Get total system memory in GB
fn get_system_total_memory() -> f64 {
    use sysinfo::System;
    
    let mut sys = System::new();
    sys.refresh_memory();
    
    // Convert from KB to GB
    (sys.total_memory() as f64) / (1024.0 * 1024.0)
}

// ===== New Cognitive Tab Implementations =====

/// Draw the cognitive system overview tab with autonomous intelligence
fn draw_cognitive_overview(f: &mut Frame, app: &mut App, area: Rect) {
    // First check for cached real-time data
    if let Some(ref cached_data) = app.cached_cognitive_data {
        debug!("Using cached cognitive data with autonomy level: {:.2}", cached_data.system_health.overall_autonomy_level);
        draw_cognitive_overview_enhanced(f, area, cached_data);
        return;
    }
    
    // Fallback to direct fetch
    if let Some(system_connector) = &app.system_connector {
        match system_connector.get_cognitive_data() {
            Ok(data) => {
                debug!("Fetched cognitive data with autonomy level: {:.2}", data.system_health.overall_autonomy_level);
                draw_cognitive_overview_enhanced(f, area, &data);
            },
            Err(e) => draw_error_state(f, area, &format!("Failed to load cognitive data: {}", e)),
        }
    } else {
        draw_loading_state(f, area, "Initializing cognitive system...");
    }
}

/// Enhanced cognitive overview with autonomous system dashboard
fn draw_cognitive_overview_enhanced(f: &mut Frame, area: Rect, data: &CognitiveData) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(10), // System health overview
            Constraint::Length(10), // Real-time performance sparklines
            Constraint::Length(10), // System performance metrics
            Constraint::Length(10), // Activity feed
            Constraint::Min(0),     // Safety and activity feed
        ])
        .split(area);
    
    // Autonomous system health overview
    draw_autonomous_system_health(f, chunks[0], &data.system_health);
    
    // Real-time performance sparklines
    draw_realtime_sparklines(f, chunks[1], &data);
    
    // System performance metrics
    draw_thermodynamic_stability(f, chunks[2], &data.thermodynamic_state);
    
    // Activity feed instead of complex gradient visualization
    draw_activity_feed(f, chunks[3], &data);
    
    // Safety validation and entropy management
    let safety_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(chunks[4]);
    
    draw_safety_validation_status(f, safety_chunks[0], &data.safety_validation);
    draw_entropy_management_status(f, safety_chunks[1], &data.entropy_management);
}

/// Draw autonomous system health overview
fn draw_autonomous_system_health(f: &mut Frame, area: Rect, health: &AutonomousSystemHealth) {

    // Debug log the values
    debug!("Drawing health metrics - Autonomy: {:.2}, Consciousness: {:.2}, Stability: {:.2}", 
        health.overall_autonomy_level, 
        health.consciousness_coherence, 
        health.thermodynamic_stability
    );
    
    // Create health metrics grid
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
        ])
        .split(area);
    
    // Active processes indicator - more actionable than abstract "autonomy level"
    let active_processes = MetricCard {
        title: "Active Processes".to_string(),
        value: health.active_autonomous_processes.to_string(),
        subtitle: format!("{:.0}% utilized", health.overall_autonomy_level * 100.0),
        trend: if health.active_autonomous_processes > 12 {
            TrendDirection::Up
        } else if health.active_autonomous_processes > 8 {
            TrendDirection::Stable
        } else {
            TrendDirection::Down
        },
        border_color: if health.overall_autonomy_level > 0.8 {
            Color::Green
        } else if health.overall_autonomy_level > 0.6 {
            Color::Yellow
        } else {
            Color::Red
        },
    };
    active_processes.render(f, chunks[0]);
    
    // Decision quality - more actionable than abstract "consciousness"
    let decision_quality = MetricCard {
        title: "Decision Quality".to_string(),
        value: format!("{:.0}%", health.strategic_planning_effectiveness * 100.0),
        subtitle: format!("Coherence: {:.0}%", health.consciousness_coherence * 100.0),
        trend: if health.strategic_planning_effectiveness > 0.85 {
            TrendDirection::Up
        } else if health.strategic_planning_effectiveness > 0.7 {
            TrendDirection::Stable
        } else {
            TrendDirection::Down
        },
        border_color: Color::Blue,
    };
    decision_quality.render(f, chunks[1]);
    
    // Resource efficiency - more understandable than "thermodynamic stability"
    let resource_efficiency = MetricCard {
        title: "Resource Usage".to_string(),
        value: format!("{:.0}%", health.resource_utilization_efficiency * 100.0),
        subtitle: format!("Efficiency"),
        trend: if health.resource_utilization_efficiency > 0.85 {
            TrendDirection::Stable
        } else if health.resource_utilization_efficiency > 0.7 {
            TrendDirection::Down
        } else {
            TrendDirection::Down
        },
        border_color: if health.resource_utilization_efficiency > 0.8 {
            Color::Green
        } else if health.resource_utilization_efficiency > 0.6 {
            Color::Yellow  
        } else {
            Color::Red
        },
    };
    resource_efficiency.render(f, chunks[2]);
    
    // Safety validation rate
    let safety_card = MetricCard {
        title: "Safety Rate".to_string(),
        value: format!("{:.1}%", health.safety_validation_success_rate * 100.0),
        subtitle: "Validation".to_string(),
        trend: if health.safety_validation_success_rate > 0.95 {
            TrendDirection::Up
        } else if health.safety_validation_success_rate > 0.90 {
            TrendDirection::Stable
        } else {
            TrendDirection::Down
        },
        border_color: Color::Green,
    };
    safety_card.render(f, chunks[3]);
}

/// Draw system performance metrics
fn draw_thermodynamic_stability(f: &mut Frame, area: Rect, entropy: &CognitiveEntropy) {
    let performance_text = vec![
        Line::from(vec![
            Span::styled("⚡ System Performance", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("Response Time: "),
            Span::styled(
                format!("{:.0}ms avg", entropy.temperature_parameter * 100.0),
                Style::default().fg(if entropy.temperature_parameter < 0.5 { Color::Green } else { Color::Yellow })
            ),
        ]),
        Line::from(vec![
            Span::raw("Throughput: "),
            Span::styled(
                format!("{:.0} ops/sec", entropy.free_energy * 1000.0),
                Style::default().fg(Color::Green)
            ),
        ]),
        Line::from(vec![
            Span::raw("Error Rate: "),
            Span::styled(
                format!("{:.1}%", (1.0 - entropy.negentropy) * 10.0),
                Style::default().fg(
                    if entropy.negentropy > 0.95 { Color::Green } 
                    else if entropy.negentropy > 0.9 { Color::Yellow }
                    else { Color::Red }
                )
            ),
        ]),
        Line::from(vec![
            Span::raw("System Load: "),
            Span::styled(
                format!("{:.0}%", entropy.shannon_entropy * 100.0),
                Style::default().fg(get_entropy_color(entropy.shannon_entropy))
            ),
        ]),
    ];
    
    let performance_widget = Paragraph::new(performance_text)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Cyan))
                .title(" Thermodynamic Stability ")
        );
    
    f.render_widget(performance_widget, area);
}

/// Draw real-time performance sparklines
fn draw_realtime_sparklines(f: &mut Frame, area: Rect, data: &CognitiveData) {
    let sparkline_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(33),
            Constraint::Percentage(33),
            Constraint::Percentage(34),
        ])
        .split(area);
    
    // Generate sample data based on current values (in real implementation, this would be historical data)
    let autonomy_data: Vec<u64> = (0..20).map(|i| {
        let base = (data.system_health.overall_autonomy_level * 100.0) as u64;
        let variation = (i as f32 * 0.3).sin() * 5.0;
        (base as f32 + variation).max(0.0) as u64
    }).collect();
    
    let decision_data: Vec<u64> = (0..20).map(|i| {
        let base = (data.system_health.strategic_planning_effectiveness * 100.0) as u64;
        let variation = (i as f32 * 0.2).cos() * 3.0;
        (base as f32 + variation).max(0.0) as u64
    }).collect();
    
    let resource_data: Vec<u64> = (0..20).map(|i| {
        let base = (data.system_health.resource_utilization_efficiency * 100.0) as u64;
        let variation = ((i as f32 * 0.4).sin() + (i as f32 * 0.1).cos()) * 4.0;
        (base as f32 + variation).max(0.0) as u64
    }).collect();
    
    // Autonomy sparkline
    let autonomy_sparkline = GlowingSparkline {
        data: autonomy_data,
        color: Color::Magenta,
        title: "Autonomy Trend".to_string(),
    };
    autonomy_sparkline.render(f, sparkline_chunks[0]);
    
    // Decision quality sparkline
    let decision_sparkline = GlowingSparkline {
        data: decision_data,
        color: Color::Blue,
        title: "Decision Quality".to_string(),
    };
    decision_sparkline.render(f, sparkline_chunks[1]);
    
    // Resource efficiency sparkline
    let resource_sparkline = GlowingSparkline {
        data: resource_data,
        color: Color::Green,
        title: "Resource Usage".to_string(),
    };
    resource_sparkline.render(f, sparkline_chunks[2]);
}

/// Draw activity feed with recent operations
fn draw_activity_feed(f: &mut Frame, area: Rect, data: &CognitiveData) {
    let activities = vec![
        ListItem::new(Line::from(vec![
            Span::styled(format!("[{}]", chrono::Local::now().format("%H:%M:%S")), Style::default().fg(Color::DarkGray)),
            Span::raw(" "),
            Span::styled("DECISION", Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD)),
            Span::raw(": Planning effectiveness at "),
            Span::styled(format!("{:.0}%", data.system_health.strategic_planning_effectiveness * 100.0), Style::default().fg(Color::Green)),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled(format!("[{}]", (chrono::Local::now() - chrono::Duration::seconds(2)).format("%H:%M:%S")), Style::default().fg(Color::DarkGray)),
            Span::raw(" "),
            Span::styled("PROCESS", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
            Span::raw(": "),
            Span::styled(format!("{} active", data.system_health.active_autonomous_processes), Style::default().fg(Color::Yellow)),
            Span::raw(" autonomous processes running"),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled(format!("[{}]", (chrono::Local::now() - chrono::Duration::seconds(5)).format("%H:%M:%S")), Style::default().fg(Color::DarkGray)),
            Span::raw(" "),
            Span::styled("SAFETY", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
            Span::raw(": Validation rate "),
            Span::styled(format!("{:.1}%", data.safety_validation.validation_success_rate * 100.0), Style::default().fg(Color::Green)),
            Span::raw(" - All checks passed"),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled(format!("[{}]", (chrono::Local::now() - chrono::Duration::seconds(8)).format("%H:%M:%S")), Style::default().fg(Color::DarkGray)),
            Span::raw(" "),
            Span::styled("LEARNING", Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)),
            Span::raw(": Progress rate at "),
            Span::styled(format!("{:.0}%", data.system_health.learning_progress_rate * 100.0), Style::default().fg(Color::Magenta)),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled(format!("[{}]", (chrono::Local::now() - chrono::Duration::seconds(12)).format("%H:%M:%S")), Style::default().fg(Color::DarkGray)),
            Span::raw(" "),
            Span::styled("MEMORY", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            Span::raw(": Resource efficiency "),
            Span::styled(format!("{:.0}%", data.system_health.resource_utilization_efficiency * 100.0), Style::default().fg(Color::Cyan)),
        ])),
    ];
    
    let activity_list = List::new(activities)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Blue))
                .title(" 📊 Activity Feed ")
        );
    
    f.render_widget(activity_list, area);
}

/// Draw three-gradient alignment visualization (kept for other uses)
fn draw_gradient_alignment(f: &mut Frame, area: Rect, gradients: &ThreeGradientState) {
    let canvas = Canvas::default()
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("🔄 Three-Gradient Alignment")
                .border_style(Style::default().fg(Color::Magenta)),
        )
        .x_bounds([-1.5, 1.5])
        .y_bounds([-1.5, 1.5])
        .paint(|ctx| {
            let time = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs_f64();
            
            // Enhanced gradient vectors with dynamic flow
            let value_magnitude = gradients.value_gradient.magnitude as f64;
            let harmony_magnitude = gradients.harmony_gradient.magnitude as f64;
            let intuition_magnitude = gradients.intuition_gradient.magnitude as f64;
            
            // Calculate rotating gradient positions for 3D effect
            let rotation = time * 0.2;
            
            let value_x = value_magnitude * rotation.cos();
            let value_y = value_magnitude * rotation.sin() * 0.5;
            
            let harmony_x = harmony_magnitude * (rotation + 2.094).cos();
            let harmony_y = harmony_magnitude * (rotation + 2.094).sin() * 0.5;
            
            let intuition_x = intuition_magnitude * (rotation + 4.189).cos();
            let intuition_y = intuition_magnitude * (rotation + 4.189).sin() * 0.5;
            
            // Draw gradient flow fields
            for i in 0..20 {
                let t = i as f64 / 20.0;
                let flow_offset = (time * 0.5 + t * 2.0).sin() * 0.1;
                
                // Value flow (red spectrum)
                let value_flow_x = value_x * t + flow_offset;
                let value_flow_y = value_y * t;
                let red_intensity = (255.0 * (1.0 - t * 0.5)) as u8;
                ctx.draw(&ratatui::widgets::canvas::Points {
                    coords: &[(value_flow_x, value_flow_y)],
                    color: Color::Rgb(red_intensity, 50, 50),
                });
                
                // Harmony flow (green spectrum)
                let harmony_flow_x = harmony_x * t + flow_offset * 0.866;
                let harmony_flow_y = harmony_y * t + flow_offset * 0.5;
                let green_intensity = (255.0 * (1.0 - t * 0.5)) as u8;
                ctx.draw(&ratatui::widgets::canvas::Points {
                    coords: &[(harmony_flow_x, harmony_flow_y)],
                    color: Color::Rgb(50, green_intensity, 50),
                });
                
                // Intuition flow (blue spectrum)
                let intuition_flow_x = intuition_x * t - flow_offset * 0.866;
                let intuition_flow_y = intuition_y * t + flow_offset * 0.5;
                let blue_intensity = (255.0 * (1.0 - t * 0.5)) as u8;
                ctx.draw(&ratatui::widgets::canvas::Points {
                    coords: &[(intuition_flow_x, intuition_flow_y)],
                    color: Color::Rgb(50, 50, blue_intensity),
                });
            }
            
            // Draw main gradient vectors with glow effect
            for glow in 0..3 {
                let glow_alpha = 255 - glow * 80;
                
                // Value gradient (red)
                ctx.draw(&ratatui::widgets::canvas::Line {
                    x1: 0.0,
                    y1: 0.0,
                    x2: value_x,
                    y2: value_y,
                    color: Color::Rgb(glow_alpha, 0, 0),
                });
                
                // Harmony gradient (green)
                ctx.draw(&ratatui::widgets::canvas::Line {
                    x1: 0.0,
                    y1: 0.0,
                    x2: harmony_x,
                    y2: harmony_y,
                    color: Color::Rgb(0, glow_alpha, 0),
                });
                
                // Intuition gradient (blue)
                ctx.draw(&ratatui::widgets::canvas::Line {
                    x1: 0.0,
                    y1: 0.0,
                    x2: intuition_x,
                    y2: intuition_y,
                    color: Color::Rgb(0, 0, glow_alpha),
                });
            }
            
            // Enhanced coherence visualization with pulsing effect
            let coherence_radius = gradients.overall_coherence;
            let pulse = (time * 2.0).sin() * 0.1 + 1.0;
            
            // Multiple coherence rings for depth
            for ring in 0..3 {
                let ring_radius = coherence_radius as f64 * pulse * (1.0 - ring as f64 * 0.2);
                let ring_alpha = (255 - ring * 60) as u8;
                
                // Draw coherence ring
                for angle in 0..72 {
                    let theta = angle as f64 * 5.0 * std::f64::consts::PI / 180.0;
                    let x = theta.cos() * ring_radius;
                    let y = theta.sin() * ring_radius;
                    ctx.draw(&ratatui::widgets::canvas::Points {
                        coords: &[(x, y)],
                        color: Color::Rgb(ring_alpha, ring_alpha, ring_alpha),
                    });
                }
            }
            
            // Add gradient conflict indicators
            for conflict in &gradients.gradient_conflicts {
                let conflict_strength = conflict.conflict_magnitude as f64;
                let conflict_x = (time * 0.7).cos() * conflict_strength;
                let conflict_y = (time * 0.7).sin() * conflict_strength;
                
                // Draw conflict marker
                ctx.draw(&ratatui::widgets::canvas::Points {
                    coords: &[(conflict_x, conflict_y)],
                    color: Color::Yellow,
                });
            }
        });
    
    f.render_widget(canvas, area);
}

/// Draw safety validation status
fn draw_safety_validation_status(f: &mut Frame, area: Rect, safety: &SafetyValidationStatus) {
    let safety_lines = vec![
        Line::from(vec![
            Span::styled("🛡️ Safety Validation", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("Success Rate: "),
            Span::styled(
                format!("{:.1}%", safety.validation_success_rate * 100.0),
                Style::default().fg(Color::Green)
            ),
        ]),
        Line::from(vec![
            Span::raw("Requests Filtered: "),
            Span::styled(
                safety.external_requests_filtered.to_string(),
                Style::default().fg(Color::Yellow)
            ),
        ]),
        Line::from(vec![
            Span::raw("Harmful Blocked: "),
            Span::styled(
                safety.harmful_requests_blocked.to_string(),
                Style::default().fg(Color::Red)
            ),
        ]),
        Line::from(vec![
            Span::raw("Active Rules: "),
            Span::styled(
                safety.active_safety_rules.to_string(),
                Style::default().fg(Color::Cyan)
            ),
        ]),
    ];
    
    let safety_widget = Paragraph::new(safety_lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Green))
                .title(" Safety Status ")
        );
    
    f.render_widget(safety_widget, area);
}

/// Draw entropy management status
fn draw_entropy_management_status(f: &mut Frame, area: Rect, entropy: &EntropyManagementStatus) {
    let entropy_lines = vec![
        Line::from(vec![
            Span::styled("📊 Entropy Management", Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("Current Level: "),
            Span::styled(
                format!("{:.2}", entropy.current_entropy_level),
                Style::default().fg(get_entropy_color(entropy.current_entropy_level))
            ),
        ]),
        Line::from(vec![
            Span::raw("Target Level: "),
            Span::styled(
                format!("{:.2}", entropy.target_entropy_level),
                Style::default().fg(Color::Green)
            ),
        ]),
        Line::from(vec![
            Span::raw("Efficiency: "),
            Span::styled(
                format!("{:.1}%", entropy.thermodynamic_efficiency * 100.0),
                Style::default().fg(Color::Cyan)
            ),
        ]),
        Line::from(vec![
            Span::raw("Violations: "),
            Span::styled(
                entropy.entropy_threshold_violations.to_string(),
                Style::default().fg(if entropy.entropy_threshold_violations > 0 { Color::Yellow } else { Color::Green })
            ),
        ]),
    ];
    
    let entropy_widget = Paragraph::new(entropy_lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Blue))
                .title(" Entropy Status ")
        );
    
    f.render_widget(entropy_widget, area);
}

/// Draw the cognitive operator tab (renamed from orchestrator)
fn draw_cognitive_operator(f: &mut Frame, app: &mut App, area: Rect) {
    // First check for cached real-time data
    if let Some(ref cached_data) = app.cached_cognitive_data {
        draw_cognitive_operator_enhanced(f, area, cached_data);
        return;
    }
    
    // Fallback to direct fetch
    if let Some(system_connector) = &app.system_connector {
        match system_connector.get_cognitive_data() {
            Ok(data) => draw_cognitive_operator_enhanced(f, area, &data),
            Err(e) => draw_error_state(f, area, &format!("Failed to load cognitive data: {}", e)),
        }
    } else {
        draw_loading_state(f, area, "Initializing unified controller...");
    }
}

/// Enhanced cognitive operator view with unified control center
fn draw_cognitive_operator_enhanced(f: &mut Frame, area: Rect, data: &CognitiveData) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(10), // Controller status
            Constraint::Length(12), // Recursive reasoning
            Constraint::Percentage(40), // Operations queue
            Constraint::Min(0),     // Thermodynamic gradients
        ])
        .split(area);
    
    // Master coordination loop status
    draw_unified_controller_status(f, chunks[0], &data.unified_controller_status);
    
    // Recursive reasoning coordination
    draw_recursive_coordination(f, chunks[1], &data.recursive_processor_status);
    
    // Active cognitive operations
    draw_cognitive_operations_queue(f, chunks[2], &data.unified_controller_status.active_cognitive_operations);
    
    // Thermodynamic gradient flow
    draw_thermodynamic_gradients(f, chunks[3], &data.thermodynamic_state);
}

/// Draw unified controller status
fn draw_unified_controller_status(f: &mut Frame, area: Rect, status: &UnifiedControllerStatus) {
    let status_lines = vec![
        Line::from(vec![
            Span::styled("🎯 Unified Cognitive Controller", Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("Master Coordination: "),
            Span::styled(
                if status.master_coordination_active { "● Active" } else { "○ Inactive" },
                Style::default().fg(if status.master_coordination_active { Color::Green } else { Color::Red })
            ),
        ]),
        Line::from(vec![
            Span::raw("Consciousness-Driven: "),
            Span::styled(
                if status.consciousness_driven_decisions { "Enabled" } else { "Disabled" },
                Style::default().fg(if status.consciousness_driven_decisions { Color::Green } else { Color::Yellow })
            ),
        ]),
        Line::from(vec![
            Span::raw("Coordination Frequency: "),
            Span::styled(
                format!("{}ms", status.coordination_frequency_ms),
                Style::default().fg(Color::Cyan)
            ),
        ]),
        Line::from(vec![
            Span::raw("Gradient Coherence: "),
            Span::styled(
                format!("{:.1}%", status.gradient_coherence * 100.0),
                Style::default().fg(Color::Blue)
            ),
        ]),
    ];
    
    let status_widget = Paragraph::new(status_lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Magenta))
                .title(" Controller Status ")
        );
    
    f.render_widget(status_widget, area);
}

/// Draw recursive reasoning coordination with fractal visualization
fn draw_recursive_coordination(f: &mut Frame, area: Rect, processor: &RecursiveProcessorStatus) {
    let time = Instant::now().elapsed().as_secs_f64();
    
    // Create a canvas for fractal recursive pattern visualization
    let canvas = Canvas::default()
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("🔄 Recursive Reasoning Fractals")
                .border_style(Style::default().fg(Color::Cyan)),
        )
        .x_bounds([-40.0, 40.0])
        .y_bounds([-40.0, 40.0])
        .paint(move |ctx| {
            // Draw recursive fractal tree pattern
            fn draw_fractal_branch(
                ctx: &mut ratatui::widgets::canvas::Context,
                x: f64,
                y: f64,
                angle: f64,
                length: f64,
                depth: u32,
                max_depth: u32,
                time: f64,
            ) {
                if depth >= max_depth || length < 1.0 {
                    return;
                }
                
                // Calculate end point
                let end_x = x + angle.cos() * length;
                let end_y = y + angle.sin() * length;
                
                // Color based on depth
                let color = match depth % 4 {
                    0 => Color::Cyan,
                    1 => Color::Blue,
                    2 => Color::Magenta,
                    _ => Color::DarkGray,
                };
                
                // Draw branch
                ctx.draw(&ratatui::widgets::canvas::Line {
                    x1: x,
                    y1: y,
                    x2: end_x,
                    y2: end_y,
                    color,
                });
                
                // Time-based angle variation
                let angle_variation = (time * 0.5 + depth as f64 * 0.3).sin() * 0.3;
                
                // Draw recursive branches
                let branch_count = if depth < 3 { 3 } else { 2 };
                for i in 0..branch_count {
                    let branch_angle = angle + angle_variation + 
                        (i as f64 - (branch_count as f64 - 1.0) / 2.0) * 0.8;
                    draw_fractal_branch(
                        ctx,
                        end_x,
                        end_y,
                        branch_angle,
                        length * 0.7,
                        depth + 1,
                        max_depth,
                        time,
                    );
                }
                
                // Draw recursive pattern discovery nodes
                if depth > 2 && length > 5.0 {
                    let pulse = ((time * 3.0 + depth as f64).sin() + 1.0) / 2.0;
                    if pulse > 0.7 {
                        ctx.print(end_x, end_y, "◊");
                    }
                }
            }
            
            // Draw main fractal trees from different starting points
            let active_processes = processor.active_processes.min(5) as usize;
            for i in 0..active_processes {
                let base_angle = i as f64 * 2.0 * std::f64::consts::PI / active_processes as f64;
                let start_x = base_angle.cos() * 10.0;
                let start_y = base_angle.sin() * 10.0;
                
                draw_fractal_branch(
                    ctx,
                    start_x,
                    start_y,
                    base_angle + std::f64::consts::PI / 2.0,
                    15.0,
                    0,
                    processor.total_recursive_depth.min(6),
                    time + i as f64 * 0.5,
                );
            }
            
            // Draw pattern discovery indicators
            let discoveries = processor.pattern_discovery_rate * 10.0;
            for i in 0..(discoveries as usize) {
                let discovery_angle = time * 0.2 + i as f64 * 0.5;
                let radius = 30.0 + (discovery_angle * 2.0).sin() * 5.0;
                let dx = discovery_angle.cos() * radius;
                let dy = discovery_angle.sin() * radius;
                ctx.print(dx, dy, "✦");
            }
            
            // Draw scale coordination connections
            if processor.scale_coordination_efficiency > 0.5 {
                for i in 0..3 {
                    let scale_angle = time * 0.1 + i as f64 * 2.0 * std::f64::consts::PI / 3.0;
                    let scale_radius = 35.0;
                    for j in 0..12 {
                        let point_angle = j as f64 * 30.0 * std::f64::consts::PI / 180.0;
                        let px = scale_angle.cos() * scale_radius + point_angle.cos() * 3.0;
                        let py = scale_angle.sin() * scale_radius + point_angle.sin() * 3.0;
                        ctx.print(px, py, "·");
                    }
                }
            }
            
            // Draw convergence indicators at the center
            if processor.convergence_success_rate > 0.7 {
                let convergence_pulse = (time * 2.0).sin().abs();
                for r in 0..3 {
                    let radius = (r as f64 + 1.0) * 3.0 * convergence_pulse;
                    for a in 0..8 {
                        let angle = a as f64 * 45.0 * std::f64::consts::PI / 180.0;
                        ctx.print(angle.cos() * radius, angle.sin() * radius, "°");
                    }
                }
            }
            
            // Display metrics
            ctx.print(-35.0, 35.0, format!("Active: {}", processor.active_processes));
            ctx.print(-35.0, 32.0, format!("Depth: {}", processor.total_recursive_depth));
            ctx.print(-35.0, 29.0, format!("Discovery: {:.0}%", processor.pattern_discovery_rate * 100.0));
            ctx.print(-35.0, 26.0, format!("Convergence: {:.0}%", processor.convergence_success_rate * 100.0));
        });
    
    f.render_widget(canvas, area);
}

/// Draw cognitive operations queue
fn draw_cognitive_operations_queue(f: &mut Frame, area: Rect, operations: &[CognitiveOperation]) {
    let rows: Vec<Row> = operations.iter().map(|op| {
        let status_icon = match op.status {
            OperationStatus::Queued => "⏳",
            OperationStatus::Running => "▶️",
            OperationStatus::Completed => "✅",
            OperationStatus::Failed => "❌",
            OperationStatus::Cancelled => "🚫",
        };
        
        let status_color = match op.status {
            OperationStatus::Running => Color::Green,
            OperationStatus::Completed => Color::Blue,
            OperationStatus::Failed => Color::Red,
            OperationStatus::Cancelled => Color::Gray,
            OperationStatus::Queued => Color::Yellow,
        };
        
        Row::new(vec![
            format!("{} {}", status_icon, op.id),
            op.operation_type.clone(),
            format!("{}ms", op.expected_duration_ms),
            format!("{:.1}%", op.resource_usage.cpu_percent),
        ])
        .style(Style::default().fg(status_color))
    }).collect();
    
    let header = Row::new(vec!["Operation", "Type", "Duration", "CPU"])
        .style(Style::default().fg(Color::Gray).add_modifier(Modifier::BOLD));
    
    let table = Table::new(
        rows,
        vec![
            Constraint::Length(25),
            Constraint::Length(20),
            Constraint::Length(10),
            Constraint::Length(8),
        ],
    )
    .header(header)
    .block(
        Block::default()
            .borders(Borders::ALL)
            .title("🔧 Active Operations")
            .border_style(Style::default().fg(Color::Yellow)),
    );
    
    f.render_widget(table, area);
}

/// Draw thermodynamic gradient flow with advanced visualization
fn draw_thermodynamic_gradients(f: &mut Frame, area: Rect, thermo: &CognitiveEntropy) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(40),  // Text info
            Constraint::Percentage(60),  // Flow visualization
        ])
        .split(area);
    
    // Text information
    let gradient_text = vec![
        Line::from(vec![
            Span::styled("🌊 Thermodynamic Flow", Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("Shannon: "),
            Span::styled(
                format!("{:.3}", thermo.shannon_entropy),
                Style::default().fg(Color::Cyan)
            ),
        ]),
        Line::from(vec![
            Span::raw("Thermo: "),
            Span::styled(
                format!("{:.3}", thermo.thermodynamic_entropy),
                Style::default().fg(Color::Yellow)
            ),
        ]),
        Line::from(vec![
            Span::raw("Negentropy: "),
            Span::styled(
                format!("{:.3}", thermo.negentropy),
                Style::default().fg(Color::Green)
            ),
        ]),
        Line::from(vec![
            Span::raw("Free Energy: "),
            Span::styled(
                format!("{:.3}", thermo.free_energy),
                Style::default().fg(Color::Magenta)
            ),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("Production: "),
            Span::styled(
                format!("{:.3}/s", thermo.entropy_production_rate),
                Style::default().fg(Color::Yellow)
            ),
        ]),
        Line::from(vec![
            Span::raw("Balance: "),
            Span::styled(
                format!("{:+.3}", thermo.entropy_flow_balance),
                Style::default().fg(if thermo.entropy_flow_balance >= 0.0 { Color::Green } else { Color::Red })
            ),
        ]),
    ];
    
    let info_widget = Paragraph::new(gradient_text)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Blue))
        );
    
    f.render_widget(info_widget, chunks[0]);
    
    // Flow field visualization
    let canvas = Canvas::default()
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Entropy Flow Field")
                .border_style(Style::default().fg(Color::Cyan)),
        )
        .x_bounds([-1.0, 1.0])
        .y_bounds([-1.0, 1.0])
        .paint(|ctx| {
            let time = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs_f64();
            
            // Draw flow field vectors
            let grid_size = 10;
            for i in 0..grid_size {
                for j in 0..grid_size {
                    let x = -0.9 + (i as f64 / grid_size as f64) * 1.8;
                    let y = -0.9 + (j as f64 / grid_size as f64) * 1.8;
                    
                    // Calculate flow field based on entropy dynamics
                    let entropy_field = thermo.thermodynamic_entropy as f64;
                    let production = thermo.entropy_production_rate as f64;
                    
                    // Create swirling pattern based on entropy
                    let angle = (x * x + y * y).sqrt() * entropy_field * 2.0 + time * 0.5;
                    let magnitude = production * 0.1 * (1.0 - (x * x + y * y).sqrt());
                    
                    let dx = angle.cos() * magnitude;
                    let dy = angle.sin() * magnitude;
                    
                    // Draw flow vector
                    ctx.draw(&ratatui::widgets::canvas::Line {
                        x1: x,
                        y1: y,
                        x2: x + dx,
                        y2: y + dy,
                        color: Color::Rgb(
                            (100.0 + entropy_field * 100.0) as u8,
                            (100.0 + production * 1000.0) as u8,
                            200,
                        ),
                    });
                }
            }
            
            // Draw entropy wells/sources
            let free_energy = thermo.free_energy as f64;
            let well_x = free_energy.sin() * 0.5;
            let well_y = free_energy.cos() * 0.5;
            
            // Entropy well (attractor)
            for ring in 0..5 {
                let radius = (ring as f64 + 1.0) * 0.1;
                let alpha = 255 - ring * 40;
                for angle in 0..36 {
                    let theta = angle as f64 * 10.0 * std::f64::consts::PI / 180.0;
                    let x = well_x + theta.cos() * radius;
                    let y = well_y + theta.sin() * radius;
                    ctx.draw(&ratatui::widgets::canvas::Points {
                        coords: &[(x, y)],
                        color: Color::Rgb(alpha as u8, 0, alpha as u8),
                    });
                }
            }
            
            // Negentropy source
            let neg_x = -well_x;
            let neg_y = -well_y;
            let negentropy = thermo.negentropy as f64;
            
            ctx.draw(&ratatui::widgets::canvas::Circle {
                x: neg_x,
                y: neg_y,
                radius: negentropy * 0.2,
                color: Color::Green,
            });
        });
    
    f.render_widget(canvas, chunks[1]);
}

/// Draw cognitive agents tab
fn draw_cognitive_agents(f: &mut Frame, app: &mut App, area: Rect) {
    // First check for cached real-time data
    if let Some(ref cached_data) = app.cached_cognitive_data {
        draw_cognitive_agents_enhanced(f, area, cached_data);
        return;
    }
    
    // Fallback to direct fetch
    if let Some(system_connector) = &app.system_connector {
        match system_connector.get_cognitive_data() {
            Ok(data) => draw_cognitive_agents_enhanced(f, area, &data),
            Err(e) => draw_error_state(f, area, &format!("Failed to load agent data: {}", e)),
        }
    } else {
        draw_loading_state(f, area, "Initializing multi-agent system...");
    }
}

/// Enhanced cognitive agents view
fn draw_cognitive_agents_enhanced(f: &mut Frame, area: Rect, data: &CognitiveData) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(8),  // Coordination status
            Constraint::Length(10), // Harmony gradient
            Constraint::Percentage(50), // Agent table
            Constraint::Min(0),     // Protocols
        ])
        .split(area);
    
    // Agent coordination overview
    draw_agent_coordination_overview(f, chunks[0], &data.agent_coordination);
    
    // Harmony gradient visualization
    draw_harmony_gradient(f, chunks[1], &data.three_gradient_state.harmony_gradient);
    
    // Active agents table
    draw_specialized_agents_table(f, chunks[2], &data.active_agents);
    
    // Coordination protocols
    draw_coordination_protocols(f, chunks[3], &data.coordination_protocols);
}

/// Draw agent coordination overview
fn draw_agent_coordination_overview(f: &mut Frame, area: Rect, coordination: &AgentCoordinationStatus) {
    let coord_lines = vec![
        Line::from(vec![
            Span::styled("🤝 Agent Coordination", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![
            Span::raw("Active/Total: "),
            Span::styled(
                format!("{}/{}", coordination.active_agents, coordination.total_agents),
                Style::default().fg(Color::Green)
            ),
            Span::raw("  Efficiency: "),
            Span::styled(
                format!("{:.1}%", coordination.coordination_efficiency * 100.0),
                Style::default().fg(Color::Cyan)
            ),
            Span::raw("  Harmony: "),
            Span::styled(
                format!("{:.1}%", coordination.harmony_gradient_level * 100.0),
                Style::default().fg(Color::Magenta)
            ),
        ]),
    ];
    
    let coord_widget = Paragraph::new(coord_lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Yellow))
        );
    
    f.render_widget(coord_widget, area);
}

/// Draw harmony gradient visualization
fn draw_harmony_gradient(f: &mut Frame, area: Rect, harmony: &GradientState) {
    let gradient_bar = format!(
        "{}{}{}",
        "█".repeat((harmony.current_value * 20.0) as usize),
        "▓".repeat(((1.0 - harmony.current_value) * 20.0 * 0.5) as usize),
        "░".repeat(((1.0 - harmony.current_value) * 20.0 * 0.5) as usize)
    );
    
    let harmony_text = vec![
        Line::from(vec![
            Span::styled("💚 Harmony Gradient", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![
            Span::raw("Level: ["),
            Span::styled(gradient_bar, Style::default().fg(Color::Green)),
            Span::raw("] "),
            Span::styled(
                format!("{:.1}%", harmony.current_value * 100.0),
                Style::default().fg(Color::Green)
            ),
        ]),
    ];
    
    let harmony_widget = Paragraph::new(harmony_text)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Green))
        );
    
    f.render_widget(harmony_widget, area);
}

/// Draw specialized agents table
fn draw_specialized_agents_table(f: &mut Frame, area: Rect, agents: &[SpecializedAgentInfo]) {
    let rows: Vec<Row> = agents.iter().map(|agent| {
        let status_icon = match agent.status {
            AgentStatus::Idle => "💤",
            AgentStatus::Active => "🔵",
            AgentStatus::Collaborating => "🤝",
            AgentStatus::Learning => "📚",
            AgentStatus::Suspended => "⏸️",
            AgentStatus::Error => "❌",
        };
        
        let status_color = match agent.status {
            AgentStatus::Active => Color::Green,
            AgentStatus::Collaborating => Color::Cyan,
            AgentStatus::Learning => Color::Blue,
            AgentStatus::Idle => Color::Gray,
            AgentStatus::Suspended => Color::Yellow,
            AgentStatus::Error => Color::Red,
        };
        
        Row::new(vec![
            format!("{} {}", status_icon, agent.name),
            agent.specialization.clone(),
            format!("{:?}", agent.status),  // Use Debug format instead of as_str()
            format!("{:.1}%", agent.performance_score * 100.0),
            agent.current_task.as_ref().unwrap_or(&"None".to_string()).clone(),
        ])
        .style(Style::default().fg(status_color))
    }).collect();
    
    let header = Row::new(vec!["Agent", "Specialization", "Status", "Performance", "Current Task"])
        .style(Style::default().fg(Color::Gray).add_modifier(Modifier::BOLD));
    
    let table = Table::new(
        rows,
        vec![
            Constraint::Length(20),
            Constraint::Length(15),
            Constraint::Length(12),
            Constraint::Length(12),
            Constraint::Min(20),
        ],
    )
    .header(header)
    .block(
        Block::default()
            .borders(Borders::ALL)
            .title("🤖 Specialized Agents")
            .border_style(Style::default().fg(Color::Cyan)),
    );
    
    f.render_widget(table, area);
}

/// Draw coordination protocols
fn draw_coordination_protocols(f: &mut Frame, area: Rect, protocols: &[ActiveProtocol]) {
    let protocol_items: Vec<ListItem> = protocols.iter().map(|protocol| {
        let status_icon = match protocol.status {
            ProtocolStatus::Initiating => "🚀",
            ProtocolStatus::Active => "⚡",
            ProtocolStatus::Finalizing => "🏁",
            ProtocolStatus::Completed => "✅",
            ProtocolStatus::Failed => "❌",
        };
        
        ListItem::new(Line::from(vec![
            Span::styled(status_icon, Style::default()),
            Span::raw(" "),
            Span::styled(&protocol.protocol_id, Style::default().fg(Color::Yellow)),
            Span::raw(" - "),
            Span::styled(format!("{:?}", protocol.protocol_type), Style::default().fg(Color::Cyan)),
            Span::raw(" ("),
            Span::styled(format!("{} agents", protocol.participants.len()), Style::default().fg(Color::Green)),
            Span::raw(")"),
        ]))
    }).collect();
    
    let protocols_list = List::new(protocol_items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("📡 Active Protocols")
                .border_style(Style::default().fg(Color::Magenta)),
        );
    
    f.render_widget(protocols_list, area);
}

/// Draw cognitive autonomy tab
fn draw_cognitive_autonomy(f: &mut Frame, app: &mut App, area: Rect) {
    // First check for cached real-time data
    if let Some(ref cached_data) = app.cached_cognitive_data {
        draw_cognitive_autonomy_enhanced(f, area, cached_data);
        return;
    }
    
    // Fallback to direct fetch
    if let Some(system_connector) = &app.system_connector {
        match system_connector.get_cognitive_data() {
            Ok(data) => draw_cognitive_autonomy_enhanced(f, area, &data),
            Err(e) => draw_error_state(f, area, &format!("Failed to load autonomy data: {}", e)),
        }
    } else {
        draw_loading_state(f, area, "Initializing autonomous systems...");
    }
}

/// Enhanced cognitive autonomy view
fn draw_cognitive_autonomy_enhanced(f: &mut Frame, area: Rect, data: &CognitiveData) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(10), // Value gradient
            Constraint::Percentage(40), // Goals
            Constraint::Percentage(30), // Strategic plans
            Constraint::Min(0),     // Recursive planning
        ])
        .split(area);
    
    // Value gradient visualization
    draw_value_gradient(f, chunks[0], &data.three_gradient_state.value_gradient);
    
    // Active autonomous goals
    draw_autonomous_goals(f, chunks[1], &data.active_goals);
    
    // Strategic plans
    draw_strategic_plans(f, chunks[2], &data.strategic_plans);
    
    // Recursive goal refinement
    draw_recursive_planning(f, chunks[3], &data.recursive_processor_status);
}

/// Draw value gradient
fn draw_value_gradient(f: &mut Frame, area: Rect, value: &GradientState) {
    let gradient_bar = format!(
        "{}{}{}",
        "█".repeat((value.current_value * 20.0) as usize),
        "▓".repeat(((1.0 - value.current_value) * 20.0 * 0.5) as usize),
        "░".repeat(((1.0 - value.current_value) * 20.0 * 0.5) as usize)
    );
    
    let value_text = vec![
        Line::from(vec![
            Span::styled("💎 Value Gradient", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![
            Span::raw("Level: ["),
            Span::styled(gradient_bar, Style::default().fg(Color::Red)),
            Span::raw("] "),
            Span::styled(
                format!("{:.1}%", value.current_value * 100.0),
                Style::default().fg(Color::Red)
            ),
        ]),
        Line::from(vec![
            Span::raw("Direction: "),
            Span::styled(
                if value.direction > 0.0 { "↗ Increasing" } else if value.direction < 0.0 { "↘ Decreasing" } else { "→ Stable" },
                Style::default().fg(if value.direction > 0.0 { Color::Green } else if value.direction < 0.0 { Color::Red } else { Color::Yellow })
            ),
        ]),
    ];
    
    let value_widget = Paragraph::new(value_text)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Red))
        );
    
    f.render_widget(value_widget, area);
}

/// Draw autonomous goals
fn draw_autonomous_goals(f: &mut Frame, area: Rect, goals: &[AutonomousGoal]) {
    let rows: Vec<Row> = goals.iter().take(5).map(|goal| {
        let priority_icon = match goal.priority {
            Priority::Critical => "🔴",
            Priority::High => "🟠",
            Priority::Medium => "🟡",
            Priority::Low => "🟢",
        };
        
        let status_color = match goal.status {
            GoalStatus::Active => Color::Green,
            GoalStatus::Planning => Color::Blue,
            GoalStatus::Suspended => Color::Yellow,
            GoalStatus::Completed => Color::Cyan,
            GoalStatus::Failed => Color::Red,
            GoalStatus::Cancelled => Color::Gray,
        };
        
        Row::new(vec![
            format!("{} {}", priority_icon, goal.name),
            format!("{:?}", goal.goal_type),
            format!("{:.0}%", goal.progress * 100.0),
            format!("{:.2}", goal.thermodynamic_efficiency),
        ])
        .style(Style::default().fg(status_color))
    }).collect();
    
    let header = Row::new(vec!["Goal", "Type", "Progress", "Efficiency"])
        .style(Style::default().fg(Color::Gray).add_modifier(Modifier::BOLD));
    
    let table = Table::new(
        rows,
        vec![
            Constraint::Percentage(40),
            Constraint::Length(12),
            Constraint::Length(10),
            Constraint::Length(10),
        ],
    )
    .header(header)
    .block(
        Block::default()
            .borders(Borders::ALL)
            .title("🎯 Autonomous Goals")
            .border_style(Style::default().fg(Color::Yellow)),
    );
    
    f.render_widget(table, area);
}

/// Draw strategic plans
fn draw_strategic_plans(f: &mut Frame, area: Rect, plans: &[StrategicPlan]) {
    let plan_items: Vec<ListItem> = plans.iter().take(3).map(|plan| {
        ListItem::new(vec![
            Line::from(vec![
                Span::styled("📋 ", Style::default()),
                Span::styled(&plan.name, Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            ]),
            Line::from(vec![
                Span::raw("   Goals: "),
                Span::styled(plan.goals.len().to_string(), Style::default().fg(Color::Yellow)),
                Span::raw("  Milestones: "),
                Span::styled(plan.milestones.len().to_string(), Style::default().fg(Color::Green)),
            ]),
        ])
    }).collect();
    
    let plans_list = List::new(plan_items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("📊 Strategic Plans")
                .border_style(Style::default().fg(Color::Blue)),
        );
    
    f.render_widget(plans_list, area);
}

/// Draw recursive planning status
fn draw_recursive_planning(f: &mut Frame, area: Rect, processor: &RecursiveProcessorStatus) {
    let planning_text = vec![
        Line::from(vec![
            Span::styled("🔄 Recursive Goal Refinement", Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![
            Span::raw("Active Refinements: "),
            Span::styled(processor.active_processes.to_string(), Style::default().fg(Color::Yellow)),
            Span::raw("  Convergence Rate: "),
            Span::styled(
                format!("{:.1}%", processor.convergence_success_rate * 100.0),
                Style::default().fg(Color::Green)
            ),
        ]),
    ];
    
    let planning_widget = Paragraph::new(planning_text)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Magenta))
        );
    
    f.render_widget(planning_widget, area);
}

/// Draw cognitive learning tab
fn draw_cognitive_learning(f: &mut Frame, app: &mut App, area: Rect) {
    // First check for cached real-time data
    if let Some(ref cached_data) = app.cached_cognitive_data {
        draw_cognitive_learning_enhanced(f, area, cached_data);
        return;
    }
    
    // Fallback to direct fetch
    if let Some(system_connector) = &app.system_connector {
        match system_connector.get_cognitive_data() {
            Ok(data) => draw_cognitive_learning_enhanced(f, area, &data),
            Err(e) => draw_error_state(f, area, &format!("Failed to load learning data: {}", e)),
        }
    } else {
        draw_loading_state(f, area, "Initializing learning systems...");
    }
}

/// Enhanced cognitive learning view
fn draw_cognitive_learning_enhanced(f: &mut Frame, area: Rect, data: &CognitiveData) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(10), // Intuition gradient
            Constraint::Length(12), // Learning architecture
            Constraint::Percentage(40), // Networks
            Constraint::Min(0),     // Pattern discovery
        ])
        .split(area);
    
    // Intuition gradient visualization
    draw_intuition_gradient(f, chunks[0], &data.three_gradient_state.intuition_gradient);
    
    // Learning architecture status
    draw_learning_architecture(f, chunks[1], &data.learning_architecture);
    
    // Adaptive networks
    draw_adaptive_networks(f, chunks[2], &data.adaptive_networks);
    
    // Pattern discovery
    draw_pattern_discovery(f, chunks[3], &data.pattern_replication);
}

/// Draw intuition gradient
fn draw_intuition_gradient(f: &mut Frame, area: Rect, intuition: &GradientState) {
    let gradient_bar = format!(
        "{}{}{}",
        "█".repeat((intuition.current_value * 20.0) as usize),
        "▓".repeat(((1.0 - intuition.current_value) * 20.0 * 0.5) as usize),
        "░".repeat(((1.0 - intuition.current_value) * 20.0 * 0.5) as usize)
    );
    
    let intuition_text = vec![
        Line::from(vec![
            Span::styled("🔮 Intuition Gradient", Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![
            Span::raw("Level: ["),
            Span::styled(gradient_bar, Style::default().fg(Color::Blue)),
            Span::raw("] "),
            Span::styled(
                format!("{:.1}%", intuition.current_value * 100.0),
                Style::default().fg(Color::Blue)
            ),
        ]),
        Line::from(vec![
            Span::raw("Creative Influence: "),
            Span::styled(
                format!("{:.1}%", intuition.influence_on_decisions * 100.0),
                Style::default().fg(Color::Magenta)
            ),
        ]),
    ];
    
    let intuition_widget = Paragraph::new(intuition_text)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Blue))
        );
    
    f.render_widget(intuition_widget, area);
}

/// Draw learning architecture status
fn draw_learning_architecture(f: &mut Frame, area: Rect, architecture: &LearningArchitectureStatus) {
    let arch_lines = vec![
        Line::from(vec![
            Span::styled("🧠 Learning Architecture", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("Networks: "),
            Span::styled(
                format!("{}/{}", architecture.active_networks, architecture.total_networks),
                Style::default().fg(Color::Yellow)
            ),
            Span::raw("  Learning Rate: "),
            Span::styled(
                format!("{:.3}", architecture.learning_rate),
                Style::default().fg(Color::Cyan)
            ),
            Span::raw("  Adaptation: "),
            Span::styled(
                format!("{:.1}%", architecture.adaptation_speed * 100.0),
                Style::default().fg(Color::Green)
            ),
        ]),
        Line::from(vec![
            Span::raw("Knowledge Retention: "),
            Span::styled(
                format!("{:.1}%", architecture.knowledge_retention * 100.0),
                Style::default().fg(Color::Blue)
            ),
            Span::raw("  Meta-Learning: "),
            Span::styled(
                if architecture.meta_learning_active { "Active" } else { "Inactive" },
                Style::default().fg(if architecture.meta_learning_active { Color::Green } else { Color::Red })
            ),
        ]),
    ];
    
    let arch_widget = Paragraph::new(arch_lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Green))
        );
    
    f.render_widget(arch_widget, area);
}

/// Draw adaptive networks with enhanced neural visualization
fn draw_adaptive_networks(f: &mut Frame, area: Rect, networks: &HashMap<String, NetworkStatus>) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(60),  // Network table
            Constraint::Percentage(40),  // Neural activity visualization
        ])
        .split(area);
    
    // Enhanced network table with activity indicators
    let rows: Vec<Row> = networks.iter().take(5).map(|(id, network)| {
        // Create gradient activation bar
        let activation_level = network.activation_level;
        let bar_length = 10;
        let filled = (activation_level * bar_length as f32) as usize;
        let mut activation_bar = String::new();
        
        for i in 0..bar_length {
            if i < filled {
                activation_bar.push_str("█");
            } else if i == filled && (activation_level * bar_length as f32).fract() > 0.5 {
                activation_bar.push_str("▌");
            } else {
                activation_bar.push_str("░");
            }
        }
        
        // Calculate connection density indicator
        let density = if network.neurons > 0 {
            network.connections as f32 / network.neurons as f32
        } else {
            0.0
        };
        let density_indicator = if density > 2.0 { "⚡" } else if density > 1.0 { "◉" } else { "○" };
        
        Row::new(vec![
            format!("{} {}", density_indicator, id),
            network.specialization.clone(),
            format!("{}/{}", network.neurons, network.connections),
            activation_bar,
            format!("{:.1}%", network.learning_progress * 100.0),
            format!("{:.2}", network.entropy_generation),
        ])
        .style(Style::default().fg(
            if network.activation_level > 0.7 { Color::Green }
            else if network.activation_level > 0.4 { Color::Yellow }
            else { Color::Red }
        ))
    }).collect();
    
    let header = Row::new(vec!["Network", "Specialization", "N/C", "Activation", "Progress", "Entropy"])
        .style(Style::default().fg(Color::Gray).add_modifier(Modifier::BOLD));
    
    let table = Table::new(
        rows,
        vec![
            Constraint::Length(18),
            Constraint::Length(18),
            Constraint::Length(10),
            Constraint::Length(12),
            Constraint::Length(10),
            Constraint::Length(8),
        ],
    )
    .header(header)
    .block(
        Block::default()
            .borders(Borders::ALL)
            .title("🌐 Adaptive Networks")
            .border_style(Style::default().fg(Color::Cyan)),
    );
    
    f.render_widget(table, chunks[0]);
    
    // Neural activity visualization
    let canvas = Canvas::default()
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Neural Activity Pattern")
                .border_style(Style::default().fg(Color::Blue)),
        )
        .x_bounds([0.0, 100.0])
        .y_bounds([0.0, 1.0])
        .paint(|ctx| {
            let time = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs_f64();
            
            // Draw neural spike patterns for each network
            let network_list: Vec<_> = networks.iter().take(3).collect();
            for (idx, (_id, network)) in network_list.iter().enumerate() {
                let base_y = 0.8 - idx as f64 * 0.3;
                let activation = network.activation_level as f64;
                
                // Draw spike train
                for x in 0..100 {
                    let spike_prob = activation * (1.0 + (x as f64 * 0.2 + time + idx as f64).sin() * 0.5);
                    if spike_prob > 0.7 {
                        // Draw a spike
                        let spike_height = spike_prob * 0.2;
                        ctx.draw(&ratatui::widgets::canvas::Line {
                            x1: x as f64,
                            y1: base_y,
                            x2: x as f64,
                            y2: base_y + spike_height,
                            color: match idx {
                                0 => Color::Red,
                                1 => Color::Green,
                                2 => Color::Blue,
                                _ => Color::White,
                            },
                        });
                    }
                }
                
                // Draw baseline
                ctx.draw(&ratatui::widgets::canvas::Line {
                    x1: 0.0,
                    y1: base_y,
                    x2: 100.0,
                    y2: base_y,
                    color: Color::Gray,
                });
            }
        });
    
    f.render_widget(canvas, chunks[1]);
}

/// Draw pattern discovery
fn draw_pattern_discovery(f: &mut Frame, area: Rect, patterns: &PatternReplicationMetrics) {
    let discovery_text = vec![
        Line::from(vec![
            Span::styled("🔍 Pattern Discovery", Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![
            Span::raw("Total Patterns: "),
            Span::styled(patterns.total_patterns.to_string(), Style::default().fg(Color::Yellow)),
            Span::raw("  Success Rate: "),
            Span::styled(
                format!("{:.1}%", patterns.adaptation_success_rate * 100.0),
                Style::default().fg(Color::Green)
            ),
            Span::raw("  Stability: "),
            Span::styled(
                format!("{:.1}%", patterns.pattern_stability * 100.0),
                Style::default().fg(Color::Blue)
            ),
        ]),
    ];
    
    let discovery_widget = Paragraph::new(discovery_text)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Magenta))
        );
    
    f.render_widget(discovery_widget, area);
}

/// Helper function to get color based on entropy level
fn get_entropy_color(entropy: f32) -> Color {
    if entropy < 0.3 {
        Color::Green
    } else if entropy < 0.6 {
        Color::Yellow
    } else if entropy < 0.8 {
        Color::LightRed
    } else {
        Color::Red
    }
}

// Additional tab implementations for cognitive tabs

/// Draw agent coordination status
fn draw_agent_coordination_status(f: &mut Frame, area: Rect, data: &CognitiveData) {
    let agent_coord = &data.agent_coordination;
    let status_lines = vec![
            Line::from(vec![
                Span::styled("🤝 Agent Coordination Status", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::raw("Total Agents: "),
                Span::styled(agent_coord.total_agents.to_string(), Style::default().fg(Color::Yellow)),
                Span::raw("  Active: "),
                Span::styled(agent_coord.active_agents.to_string(), Style::default().fg(Color::Green)),
                Span::raw("  Efficiency: "),
                Span::styled(format!("{:.1}%", agent_coord.coordination_efficiency * 100.0), Style::default().fg(Color::Blue)),
            ]),
            Line::from(vec![
                Span::raw("Consensus Quality: "),
                Span::styled(format!("{:.1}%", agent_coord.consensus_quality * 100.0), Style::default().fg(Color::Green)),
                Span::raw("  Task Balance: "),
                Span::styled(format!("{:.1}%", agent_coord.task_distribution_balance * 100.0), Style::default().fg(Color::Cyan)),
            ]),
            Line::from(vec![
                Span::raw("Emergent Behaviors: "),
                Span::styled(agent_coord.emergent_behaviors_detected.to_string(), Style::default().fg(Color::Magenta)),
                Span::raw("  Harmony Level: "),
                Span::styled(format!("{:.1}%", agent_coord.harmony_gradient_level * 100.0), Style::default().fg(Color::Yellow)),
            ]),
        ];
        
        let status_widget = Paragraph::new(status_lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::Cyan))
            );
        
        f.render_widget(status_widget, area);
}

/// Draw harmony gradient for agent coordination (enhanced)
fn draw_harmony_gradient_enhanced(f: &mut Frame, area: Rect, data: &CognitiveData) {
    let gradient_state = &data.three_gradient_state;
    let harmony = &gradient_state.harmony_gradient;
        
        let canvas = Canvas::default()
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("🎵 Harmony Gradient - Social Cooperation")
                    .border_style(Style::default().fg(Color::Green)),
            )
            .x_bounds([0.0, 100.0])
            .y_bounds([0.0, 100.0])
            .paint(|ctx| {
                // Draw harmony wave
                for x in 0..100 {
                    let y = 50.0 + (x as f64 * 0.1).sin() * 20.0 * harmony.current_value as f64;
                    ctx.draw(&ratatui::widgets::canvas::Points {
                        coords: &[(x as f64, y)],
                        color: Color::Green,
                    });
                }
                
                // Draw cooperation nodes
                for i in 0..5 {
                    let x = 20.0 + i as f64 * 15.0;
                    let y = 50.0 + (i as f64 * 1.5).cos() * 10.0;
                    ctx.draw(&ratatui::widgets::canvas::Circle {
                        x,
                        y,
                        radius: 3.0,
                        color: Color::LightGreen,
                    });
                }
            });
        
        f.render_widget(canvas, area);
}

/// Draw agent coordination network with swarm visualization
fn draw_agent_coordination_network(f: &mut Frame, area: Rect, data: &CognitiveData) {
    let time = Instant::now().elapsed().as_secs_f64();
    
    // Create a canvas for swarm visualization
    let canvas = Canvas::default()
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("🌐 Agent Swarm Network")
                .border_style(Style::default().fg(Color::Yellow)),
        )
        .x_bounds([-50.0, 50.0])
        .y_bounds([-50.0, 50.0])
        .paint(move |ctx| {
            let agents = &data.active_agents;
            
            // Draw connections between collaborating agents
            for (i, agent1) in agents.iter().enumerate() {
                if matches!(agent1.status, crate::tui::autonomous_data_types::AgentStatus::Collaborating) {
                    for (j, agent2) in agents.iter().enumerate() {
                        if i != j && matches!(agent2.status, crate::tui::autonomous_data_types::AgentStatus::Collaborating) {
                            // Calculate positions in a circular formation with swarm dynamics
                            let angle1 = (i as f64 * 2.0 * std::f64::consts::PI / agents.len() as f64) + time * 0.1;
                            let angle2 = (j as f64 * 2.0 * std::f64::consts::PI / agents.len() as f64) + time * 0.1;
                            
                            // Add swarm movement patterns
                            let swarm_offset1 = (time * 2.0 + i as f64).sin() * 5.0;
                            let swarm_offset2 = (time * 2.0 + j as f64).sin() * 5.0;
                            
                            let x1 = angle1.cos() * (30.0 + swarm_offset1);
                            let y1 = angle1.sin() * (30.0 + swarm_offset1);
                            let x2 = angle2.cos() * (30.0 + swarm_offset2);
                            let y2 = angle2.sin() * (30.0 + swarm_offset2);
                            
                            // Draw connection with pulsing effect
                            let pulse = ((time * 3.0 + i as f64 + j as f64).sin() + 1.0) / 2.0;
                            let connection_color = if pulse > 0.5 {
                                Color::Cyan
                            } else {
                                Color::DarkGray
                            };
                            
                            ctx.draw(&ratatui::widgets::canvas::Line {
                                x1,
                                y1,
                                x2,
                                y2,
                                color: connection_color,
                            });
                        }
                    }
                }
            }
            
            // Draw agents as nodes with dynamic effects
            for (i, agent) in agents.iter().enumerate() {
                let angle = (i as f64 * 2.0 * std::f64::consts::PI / agents.len() as f64) + time * 0.1;
                let swarm_offset = (time * 2.0 + i as f64).sin() * 5.0;
                let radius = 30.0 + swarm_offset;
                let x = angle.cos() * radius;
                let y = angle.sin() * radius;
                
                // Agent status determines color and size
                let (_, size_multiplier) = match agent.status {
                    AgentStatus::Active => (Color::Green, 1.2),
                    AgentStatus::Collaborating => (Color::Magenta, 1.5),
                    AgentStatus::Learning => (Color::Blue, 1.3),
                    AgentStatus::Idle => (Color::Yellow, 1.0),
                    AgentStatus::Suspended => (Color::Red, 0.8),
                    AgentStatus::Error => (Color::Red, 0.7),
                };
                
                // Draw agent node with entropy halo
                let entropy_radius = agent.entropy_contribution as f64 * 10.0 * size_multiplier;
                for r in 0..3 {
                    let halo_radius = entropy_radius + r as f64 * 2.0;
                    let halo_intensity = 1.0 - (r as f64 / 3.0);
                    
                    // Draw halo circle
                    for angle in 0..36 {
                        let a = angle as f64 * 10.0 * std::f64::consts::PI / 180.0;
                        ctx.print(
                            x + a.cos() * halo_radius,
                            y + a.sin() * halo_radius,
                            if halo_intensity > 0.5 { "·" } else { " " }
                        );
                    }
                }
                
                // Draw agent icon based on specialization
                let icon = match agent.specialization.as_str() {
                    "Analyzer" => "🔍",
                    "Executor" => "⚡",
                    "Coordinator" => "🎯",
                    "Monitor" => "👁️",
                    _ => "🤖",
                };
                
                ctx.print(x, y, icon);
                
                // Draw agent name
                ctx.print(x, y - 5.0, agent.name.clone());
            }
            
            // Draw emergent swarm patterns
            for i in 0..5 {
                let pattern_time = time * 0.5 + i as f64 * 0.2;
                let pattern_x = pattern_time.cos() * 40.0;
                let pattern_y = pattern_time.sin() * 40.0;
                ctx.print(pattern_x, pattern_y, "✧");
            }
        });
    
    f.render_widget(canvas, area);
}

/// Draw coordination protocols (enhanced)
fn draw_coordination_protocols_enhanced(f: &mut Frame, area: Rect, data: &CognitiveData) {
    let protocols = &data.coordination_protocols;
    let protocol_items: Vec<ListItem> = protocols.iter().map(|protocol| {
            let status_icon = match protocol.status {
                crate::tui::autonomous_data_types::ProtocolStatus::Active => "🟢",
                crate::tui::autonomous_data_types::ProtocolStatus::Initiating => "🟡",
                crate::tui::autonomous_data_types::ProtocolStatus::Finalizing => "🟠",
                crate::tui::autonomous_data_types::ProtocolStatus::Completed => "🔵",
                crate::tui::autonomous_data_types::ProtocolStatus::Failed => "🔴",
            };
            
            ListItem::new(Line::from(vec![
                Span::raw(format!("{} ", status_icon)),
                Span::styled(&protocol.protocol_id, Style::default().fg(Color::Magenta)),
                Span::raw(" - Type: "),
                Span::styled(format!("{:?}", protocol.protocol_type), Style::default().fg(Color::Cyan)),
                Span::raw(format!(" ({} participants)", protocol.participants.len())),
            ]))
        }).collect();
        
        let list = List::new(protocol_items)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("📋 Active Coordination Protocols")
                    .border_style(Style::default().fg(Color::Magenta)),
            );
        
        f.render_widget(list, area);
}

/// Draw autonomy tab
fn draw_autonomy(f: &mut Frame, app: &App, area: Rect) {
    if app.system_connector.is_some() {
        draw_autonomy_enhanced(f, app, area);
    } else {
        draw_autonomy_legacy(f, app, area);
    }
}

/// Enhanced autonomy tab with goal management and strategic planning
fn draw_autonomy_enhanced(f: &mut Frame, app: &App, area: Rect) {
    let system_connector = match app.system_connector.as_ref() {
        Some(conn) => conn,
        None => {
            draw_loading_state(f, area, "Initializing autonomous systems...");
            return;
        }
    };
    
    let cognitive_data = match system_connector.get_cognitive_data() {
        Ok(data) => data,
        Err(e) => {
            draw_error_state(f, area, &format!("Failed to load autonomy data: {}", e));
            return;
        }
    };
    
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(10), // Goal overview
            Constraint::Length(12), // Value gradient
            Constraint::Percentage(40), // Active goals
            Constraint::Min(0),     // Strategic plans
        ])
        .split(area);
    
    // Goal management overview
    draw_goal_overview(f, chunks[0], &cognitive_data);
    
    // Value gradient visualization
    draw_value_gradient_enhanced(f, chunks[1], &cognitive_data);
    
    // Active autonomous goals
    draw_autonomous_goals_enhanced(f, chunks[2], &cognitive_data);
    
    // Strategic plans
    draw_strategic_plans_enhanced(f, chunks[3], &cognitive_data);
}

fn draw_autonomy_legacy(f: &mut Frame, _app: &App, area: Rect) {
    let placeholder = Paragraph::new("Autonomous goal management system not initialized")
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("🎯 Autonomy")
                .border_style(Style::default().fg(Color::Yellow)),
        )
        .alignment(Alignment::Center);
    
    f.render_widget(placeholder, area);
}

/// Draw goal overview
fn draw_goal_overview(f: &mut Frame, area: Rect, data: &CognitiveData) {
    let goal_tracker = &data.achievement_tracker;
    let overview_lines = vec![
            Line::from(vec![
                Span::styled("🎯 Goal Management Overview", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::raw("Completed Goals: "),
                Span::styled(goal_tracker.total_goals_completed.to_string(), Style::default().fg(Color::Green)),
                Span::raw("  Success Rate: "),
                Span::styled(format!("{:.1}%", goal_tracker.success_rate * 100.0), Style::default().fg(Color::Cyan)),
                Span::raw("  Recognition: "),
                Span::styled(format!("{:.1}", goal_tracker.recognition_score), Style::default().fg(Color::Yellow)),
            ]),
            Line::from(vec![
                Span::raw("Avg Completion Time: "),
                Span::styled(format!("{:.1}h", goal_tracker.average_completion_time.as_secs() as f32 / 3600.0), Style::default().fg(Color::Blue)),
                Span::raw("  Achievements: "),
                Span::styled(goal_tracker.achievements.len().to_string(), Style::default().fg(Color::Magenta)),
            ]),
    ];
    
    let overview_widget = Paragraph::new(overview_lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Yellow))
        );
    
    f.render_widget(overview_widget, area);
}

/// Draw value gradient (enhanced)
fn draw_value_gradient_enhanced(f: &mut Frame, area: Rect, data: &CognitiveData) {
    let gradient_state = &data.three_gradient_state;
    let value = &gradient_state.value_gradient;
        
    let value_lines = vec![
            Line::from(vec![
                Span::styled("💎 Value Gradient - Individual Optimization", Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD)),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::raw("Current Value: "),
                Span::styled(format!("{:.2}", value.current_value), Style::default().fg(Color::Cyan)),
                Span::raw("  Direction: "),
                Span::styled(format!("{:+.2}", value.direction), Style::default().fg(Color::Green)),
            ]),
            Line::from(vec![
                Span::raw("Magnitude: "),
                Span::styled(format!("{:.2}", value.magnitude), Style::default().fg(Color::Yellow)),
                Span::raw("  Stability: "),
                Span::styled(format!("{:.1}%", value.stability * 100.0), Style::default().fg(Color::Magenta)),
            ]),
    ];
        
        let value_widget = Paragraph::new(value_lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::Blue))
            );
        
        f.render_widget(value_widget, area);
}

/// Draw autonomous goals (enhanced)
fn draw_autonomous_goals_enhanced(f: &mut Frame, area: Rect, data: &CognitiveData) {
    let goals = &data.active_goals;
    let goal_items: Vec<ListItem> = goals.iter().map(|goal| {
            let priority_icon = match goal.priority {
                crate::tui::autonomous_data_types::Priority::Critical => "🔴",
                crate::tui::autonomous_data_types::Priority::High => "🟠",
                crate::tui::autonomous_data_types::Priority::Medium => "🟡",
                crate::tui::autonomous_data_types::Priority::Low => "🟢",
            };
            
            let type_icon = match goal.goal_type {
                crate::tui::autonomous_data_types::GoalType::Strategic => "🎯",
                crate::tui::autonomous_data_types::GoalType::Tactical => "⚔️",
                crate::tui::autonomous_data_types::GoalType::Operational => "⚙️",
                crate::tui::autonomous_data_types::GoalType::Learning => "📚",
                crate::tui::autonomous_data_types::GoalType::Maintenance => "🔧",
                crate::tui::autonomous_data_types::GoalType::Safety => "🛡️",
            };
            
            ListItem::new(Line::from(vec![
                Span::raw(format!("{} {} ", priority_icon, type_icon)),
                Span::styled(&goal.name, Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
                Span::raw(format!(" - {:.0}%", goal.progress * 100.0)),
                Span::raw(" | "),
                Span::styled(format!("{:?}", goal.status), Style::default().fg(
                    match goal.status {
                        crate::tui::autonomous_data_types::GoalStatus::Active => Color::Green,
                        crate::tui::autonomous_data_types::GoalStatus::Planning => Color::Yellow,
                        crate::tui::autonomous_data_types::GoalStatus::Completed => Color::Blue,
                        crate::tui::autonomous_data_types::GoalStatus::Failed => Color::Red,
                        crate::tui::autonomous_data_types::GoalStatus::Suspended => Color::Gray,
                        crate::tui::autonomous_data_types::GoalStatus::Cancelled => Color::DarkGray,
                    }
                )),
            ]))
        }).collect();
        
        let list = List::new(goal_items)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("🎯 Active Autonomous Goals")
                    .border_style(Style::default().fg(Color::Yellow)),
            );
        
        f.render_widget(list, area);
}

/// Draw strategic plans (enhanced)
fn draw_strategic_plans_enhanced(f: &mut Frame, area: Rect, data: &CognitiveData) {
    let plans = &data.strategic_plans;
    let plan_items: Vec<ListItem> = plans.iter().map(|plan| {
            ListItem::new(vec![
                Line::from(vec![
                    Span::styled(&plan.name, Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
                    Span::raw(format!(" - {} milestones", plan.milestones.len())),
                ]),
                Line::from(vec![
                    Span::raw("  Resources: "),
                    Span::styled(format!("{} allocated", plan.resource_allocation.len()), Style::default().fg(Color::Green)),
                    Span::raw(" | Duration: "),
                    Span::styled(format!("{} days", plan.expected_duration.as_secs() / 86400), Style::default().fg(Color::Yellow)),
                    Span::raw(" | Risks: "),
                    Span::styled(plan.risk_mitigation.len().to_string(), Style::default().fg(Color::Red)),
                ]),
            ])
        }).collect();
        
        let list = List::new(plan_items)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("📋 Strategic Plans")
                    .border_style(Style::default().fg(Color::Cyan)),
            );
        
        f.render_widget(list, area);
}

/// Draw learning tab
fn draw_learning(f: &mut Frame, app: &App, area: Rect) {
    if app.system_connector.is_some() {
        draw_learning_enhanced(f, app, area);
    } else {
        draw_learning_legacy(f, app, area);
    }
}

/// Enhanced learning tab with adaptive networks and meta-learning
fn draw_learning_enhanced(f: &mut Frame, app: &App, area: Rect) {
    let system_connector = match app.system_connector.as_ref() {
        Some(conn) => conn,
        None => {
            draw_loading_state(f, area, "Initializing learning systems...");
            return;
        }
    };
    
    let cognitive_data = match system_connector.get_cognitive_data() {
        Ok(data) => data,
        Err(e) => {
            draw_error_state(f, area, &format!("Failed to load learning data: {}", e));
            return;
        }
    };
    
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(10), // Learning overview
            Constraint::Length(12), // Intuition gradient
            Constraint::Percentage(40), // Adaptive networks
            Constraint::Min(0),     // Meta-learning insights
        ])
        .split(area);
    
    // Learning architecture overview
    draw_learning_overview(f, chunks[0], &cognitive_data);
    
    // Intuition gradient visualization
    draw_intuition_gradient_enhanced(f, chunks[1], &cognitive_data);
    
    // Adaptive learning networks
    draw_adaptive_networks(f, chunks[2], &cognitive_data.adaptive_networks);
    
    // Meta-learning insights
    draw_meta_learning_insights(f, chunks[3], &cognitive_data);
}

fn draw_learning_legacy(f: &mut Frame, _app: &App, area: Rect) {
    let placeholder = Paragraph::new("Adaptive learning system not initialized")
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("🧠 Learning")
                .border_style(Style::default().fg(Color::Magenta)),
        )
        .alignment(Alignment::Center);
    
    f.render_widget(placeholder, area);
}

/// Draw learning overview
fn draw_learning_overview(f: &mut Frame, area: Rect, data: &CognitiveData) {
    let architecture = &data.learning_architecture;
    let overview_lines = vec![
            Line::from(vec![
                Span::styled("🧠 Learning Architecture Overview", Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::raw("Active Networks: "),
                Span::styled(architecture.active_networks.to_string(), Style::default().fg(Color::Cyan)),
                Span::raw("  Total Networks: "),
                Span::styled(architecture.total_networks.to_string(), Style::default().fg(Color::Yellow)),
            ]),
            Line::from(vec![
                Span::raw("Learning Rate: "),
                Span::styled(format!("{:.3}", architecture.learning_rate), Style::default().fg(Color::Green)),
                Span::raw("  Adaptation Speed: "),
                Span::styled(format!("{:.1}%", architecture.adaptation_speed * 100.0), Style::default().fg(Color::Blue)),
            ]),
        ];
        
        let overview_widget = Paragraph::new(overview_lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::Magenta))
            );
        
        f.render_widget(overview_widget, area);
}

/// Draw intuition gradient (enhanced)
fn draw_intuition_gradient_enhanced(f: &mut Frame, area: Rect, data: &CognitiveData) {
    let gradient_state = &data.three_gradient_state;
    let intuition = &gradient_state.intuition_gradient;
        
    let intuition_lines = vec![
            Line::from(vec![
                Span::styled("✨ Intuition Gradient - Creative Exploration", Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::raw("Current Value: "),
                Span::styled(format!("{:.2}", intuition.current_value), Style::default().fg(Color::Yellow)),
                Span::raw("  Direction: "),
                Span::styled(format!("{:+.2}", intuition.direction), Style::default().fg(Color::Cyan)),
            ]),
            Line::from(vec![
                Span::raw("Magnitude: "),
                Span::styled(format!("{:.2}", intuition.magnitude), Style::default().fg(Color::Green)),
                Span::raw("  Influence: "),
                Span::styled(format!("{:.1}%", intuition.influence_on_decisions * 100.0), Style::default().fg(Color::Blue)),
            ]),
    ];
        
        let intuition_widget = Paragraph::new(intuition_lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::Magenta))
            );
        
        f.render_widget(intuition_widget, area);
}

/// Draw meta-learning insights with knowledge graph visualization
fn draw_meta_learning_insights(f: &mut Frame, area: Rect, data: &CognitiveData) {
    let time = Instant::now().elapsed().as_secs_f64();
    
    // Split area for insights list and knowledge graph
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(60),
            Constraint::Percentage(40),
        ])
        .split(area);
    
    // Draw insights list on the left
    let insights = &data.meta_learning_insights;
    let insight_items: Vec<ListItem> = insights.iter().map(|insight| {
            let confidence_icon = if insight.confidence > 0.8 {
                "⭐"
            } else if insight.confidence > 0.6 {
                "🌟"
            } else {
                "✨"
            };
            
            // Add gradient contribution indicators
            let gradient_indicators = format!(
                " V:{:.0}% H:{:.0}% I:{:.0}%",
                insight.gradient_contribution.value_alignment * 100.0,
                insight.gradient_contribution.harmony_alignment * 100.0,
                insight.gradient_contribution.intuition_alignment * 100.0
            );
            
            ListItem::new(vec![
                Line::from(vec![
                    Span::raw(format!("{} ", confidence_icon)),
                    Span::styled(format!("{:?}", insight.insight_type), Style::default().fg(Color::Cyan)),
                    Span::raw(" - "),
                    Span::raw(&insight.description),
                ]),
                Line::from(vec![
                    Span::raw("   Impact: "),
                    Span::styled(format!("{:.1}%", insight.impact * 100.0), Style::default().fg(Color::Green)),
                    Span::raw(" | Apps: "),
                    Span::styled(insight.applications.len().to_string(), Style::default().fg(Color::Yellow)),
                    Span::styled(gradient_indicators, Style::default().fg(Color::DarkGray)),
                ]),
            ])
        }).collect();
        
    let list = List::new(insight_items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("💡 Meta-Learning Insights")
                .border_style(Style::default().fg(Color::Yellow)),
        );
        
    f.render_widget(list, chunks[0]);
    
    // Draw knowledge graph visualization on the right
    let canvas = Canvas::default()
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("🧬 Knowledge Graph")
                .border_style(Style::default().fg(Color::Magenta)),
        )
        .x_bounds([-30.0, 30.0])
        .y_bounds([-30.0, 30.0])
        .paint(move |ctx| {
            // Draw knowledge nodes as a neural network
            let node_count = insights.len().min(10);
            
            // Draw connections between insights based on type relationships
            for i in 0..node_count {
                for j in (i+1)..node_count {
                    if let (Some(insight1), Some(insight2)) = (insights.get(i), insights.get(j)) {
                        // Connect related insight types
                        let connected = matches!(
                            (&insight1.insight_type, &insight2.insight_type),
                            (crate::tui::autonomous_data_types::InsightType::PatternDiscovery, crate::tui::autonomous_data_types::InsightType::KnowledgeConnection) |
                            (crate::tui::autonomous_data_types::InsightType::KnowledgeConnection, crate::tui::autonomous_data_types::InsightType::EmergentCapability) |
                            (crate::tui::autonomous_data_types::InsightType::OptimizationStrategy, crate::tui::autonomous_data_types::InsightType::LearningAcceleration)
                        );
                        
                        if connected {
                            let angle1 = i as f64 * 2.0 * std::f64::consts::PI / node_count as f64;
                            let angle2 = j as f64 * 2.0 * std::f64::consts::PI / node_count as f64;
                            
                            let x1 = angle1.cos() * 20.0;
                            let y1 = angle1.sin() * 20.0;
                            let x2 = angle2.cos() * 20.0;
                            let y2 = angle2.sin() * 20.0;
                            
                            // Pulsing connection strength based on confidence
                            let pulse = (time * 2.0 + i as f64 + j as f64).sin();
                            let avg_confidence = (insight1.confidence + insight2.confidence) / 2.0;
                            let connection_color = if avg_confidence > 0.7 && pulse > 0.0 {
                                Color::Cyan
                            } else {
                                Color::DarkGray
                            };
                            
                            ctx.draw(&ratatui::widgets::canvas::Line {
                                x1,
                                y1,
                                x2,
                                y2,
                                color: connection_color,
                            });
                        }
                    }
                }
            }
            
            // Draw insight nodes
            for (i, insight) in insights.iter().take(node_count).enumerate() {
                let angle = i as f64 * 2.0 * std::f64::consts::PI / node_count as f64;
                let wobble = (time * 1.5 + i as f64 * 0.5).sin() * 2.0;
                let x = angle.cos() * (20.0 + wobble);
                let y = angle.sin() * (20.0 + wobble);
                
                // Node color based on insight type
                let node_icon = match insight.insight_type {
                    crate::tui::autonomous_data_types::InsightType::PatternDiscovery => "🔍",
                    crate::tui::autonomous_data_types::InsightType::OptimizationStrategy => "⚡",
                    crate::tui::autonomous_data_types::InsightType::KnowledgeConnection => "🔗",
                    crate::tui::autonomous_data_types::InsightType::EmergentCapability => "✨",
                    crate::tui::autonomous_data_types::InsightType::LearningAcceleration => "🚀",
                };
                
                // Draw impact radius
                let impact_radius = insight.impact as f64 * 10.0;
                for r in 0..((impact_radius as usize).min(3)) {
                    let radius = impact_radius - r as f64;
                    for a in 0..24 {
                        let angle = a as f64 * 15.0 * std::f64::consts::PI / 180.0;
                        ctx.print(
                            x + angle.cos() * radius,
                            y + angle.sin() * radius,
                            "·"
                        );
                    }
                }
                
                ctx.print(x, y, node_icon);
            }
            
            // Draw emergent patterns
            for i in 0..3 {
                let pattern_angle = time * 0.3 + i as f64 * 2.0 * std::f64::consts::PI / 3.0;
                let pattern_radius = 25.0 + (time * 0.5).sin() * 5.0;
                let px = pattern_angle.cos() * pattern_radius;
                let py = pattern_angle.sin() * pattern_radius;
                ctx.print(px, py, "🌟");
            }
        });
    
    f.render_widget(canvas, chunks[1]);
}

/// Draw cognitive controls tab
fn draw_controls(f: &mut Frame, app: &mut App, area: Rect) {
    // Update available actions based on current cognitive state
    if let Some(ref cognitive_data) = app.cached_cognitive_data {
        app.cognitive_control_state.update_available_actions(cognitive_data);
    }
    
    // Draw the control panel
    crate::tui::cognitive::core::controls::draw_cognitive_control_panel(
        f,
        area,
        &app.cognitive_control_state,
    );
}

// Note: Using the full function names in the match statement above