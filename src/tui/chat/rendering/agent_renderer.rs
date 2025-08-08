//! Agent management UI renderer

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::widgets::{Block, Borders, List, ListItem, Paragraph};

use crate::tui::chat::agents::manager::AgentManager;
use super::ChatRenderer;

/// Renders agent management UI
pub struct AgentRenderer {
    pub show_agent_details: bool,
    pub highlight_active_agents: bool,
}

impl AgentRenderer {
    pub fn new() -> Self {
        Self {
            show_agent_details: true,
            highlight_active_agents: true,
        }
    }
    
    /// Render the agent management view
    pub fn render_agent_management(&self, f: &mut Frame, area: Rect, agent_manager: &AgentManager) {
        // Split into sections
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),   // Header
                Constraint::Length(5),   // Active agents
                Constraint::Min(0),      // Agent details
            ])
            .split(area);
            
        // Render header
        self.render_header(f, chunks[0], agent_manager);
        
        // Render active agents
        self.render_active_agents(f, chunks[1], agent_manager);
        
        // Render selected agent details
        if self.show_agent_details {
            self.render_agent_details(f, chunks[2], agent_manager);
        }
    }
    
    fn render_header(&self, f: &mut Frame, area: Rect, agent_manager: &AgentManager) {
        let header_text = format!(
            "🤖 Agents | Mode: {:?} | Specializations: {}",
            agent_manager.collaboration_mode,
            agent_manager.active_specializations.len()
        );
        
        let header = Paragraph::new(header_text)
            .style(Style::default().fg(Color::Cyan))
            .block(Block::default().borders(Borders::ALL));
            
        f.render_widget(header, area);
    }
    
    fn render_active_agents(&self, f: &mut Frame, area: Rect, agent_manager: &AgentManager) {
        use ratatui::text::{Line, Span};
        
        let agent_items: Vec<ListItem> = if agent_manager.active_specializations.is_empty() {
            vec![ListItem::new(Line::from(vec![
                Span::styled("No active agents", Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC))
            ]))]
        } else {
            agent_manager.active_specializations
                .iter()
                .enumerate()
                .map(|(i, spec)| {
                    let is_selected = i == agent_manager.selected_index;
                    let (icon, color, description) = match spec {
                        crate::tui::chat::agents::AgentSpecialization::Analytical => 
                            ("🔍", Color::Blue, "Data analysis and insights"),
                        crate::tui::chat::agents::AgentSpecialization::Creative => 
                            ("🎨", Color::Magenta, "Creative solutions and ideas"),
                        crate::tui::chat::agents::AgentSpecialization::Strategic => 
                            ("♟️", Color::Green, "Strategic planning and decisions"),
                        crate::tui::chat::agents::AgentSpecialization::Technical => 
                            ("⚙️", Color::Yellow, "Technical implementation"),
                        crate::tui::chat::agents::AgentSpecialization::Social =>
                            ("👥", Color::Cyan, "Social interaction and communication"),
                        crate::tui::chat::agents::AgentSpecialization::Guardian =>
                            ("🛡️", Color::Red, "Safety and protection"),
                        crate::tui::chat::agents::AgentSpecialization::Learning =>
                            ("📚", Color::LightBlue, "Learning and adaptation"),
                        _ => ("❓", Color::Gray, "Unknown specialization"),
                    };
                    
                    let style = if is_selected {
                        Style::default().bg(Color::DarkGray).fg(color).add_modifier(Modifier::BOLD)
                    } else {
                        Style::default().fg(color)
                    };
                    
                    ListItem::new(Line::from(vec![
                        Span::raw(format!("{} ", icon)),
                        Span::styled(format!("{:?}", spec), style),
                        Span::raw(" - "),
                        Span::styled(description, Style::default().fg(Color::Gray)),
                    ]))
                })
                .collect()
        };
            
        let agents_list = List::new(agent_items)
            .block(Block::default().borders(Borders::ALL).title(" Active Specializations "))
            .highlight_style(Style::default().bg(Color::DarkGray));
            
        f.render_widget(agents_list, area);
    }
    
    fn render_agent_details(&self, f: &mut Frame, area: Rect, agent_manager: &AgentManager) {
        use ratatui::text::{Line, Span};
        use ratatui::widgets::Gauge;
        
        // Split area for details and gauge
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(5),      // Details
                Constraint::Length(3),   // Consensus gauge
            ])
            .split(area);
        
        // Get selected specialization details
        let details_content = if agent_manager.selected_index < agent_manager.active_specializations.len() {
            let spec = &agent_manager.active_specializations[agent_manager.selected_index];
            
            let capabilities = match spec {
                crate::tui::chat::agents::AgentSpecialization::Analytical => 
                    vec!["Pattern recognition", "Statistical analysis", "Trend identification"],
                crate::tui::chat::agents::AgentSpecialization::Creative => 
                    vec!["Brainstorming", "Novel solutions", "Artistic expression"],
                crate::tui::chat::agents::AgentSpecialization::Strategic => 
                    vec!["Long-term planning", "Risk assessment", "Decision trees"],
                crate::tui::chat::agents::AgentSpecialization::Technical => 
                    vec!["Code generation", "System design", "Debugging"],
                crate::tui::chat::agents::AgentSpecialization::Social =>
                    vec!["Communication", "Collaboration", "Empathy"],
                crate::tui::chat::agents::AgentSpecialization::Guardian =>
                    vec!["Safety checks", "Risk mitigation", "Compliance"],
                crate::tui::chat::agents::AgentSpecialization::Learning =>
                    vec!["Knowledge acquisition", "Adaptation", "Pattern learning"],
                _ => vec!["Unknown capabilities"],
            };
            
            let mut lines = vec![
                Line::from(vec![
                    Span::styled("Selected: ", Style::default().fg(Color::Gray)),
                    Span::styled(format!("{:?}", spec), Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
                ]),
                Line::from(""),
                Line::from(vec![
                    Span::styled("Capabilities:", Style::default().fg(Color::Yellow)),
                ]),
            ];
            
            for cap in capabilities {
                lines.push(Line::from(vec![
                    Span::raw("  • "),
                    Span::raw(cap),
                ]));
            }
            
            lines.push(Line::from(""));
            lines.push(Line::from(vec![
                Span::raw("Min Agents Required: "),
                Span::styled(
                    agent_manager.min_agents_for_consensus.to_string(), 
                    Style::default().fg(Color::Green)
                ),
            ]));
            
            lines
        } else {
            vec![Line::from(vec![
                Span::styled("No specialization selected", Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC))
            ])]
        };
        
        let details = Paragraph::new(details_content)
            .block(Block::default().borders(Borders::ALL).title(" Agent Configuration "));
            
        f.render_widget(details, chunks[0]);
        
        // Render consensus threshold gauge
        let consensus_gauge = Gauge::default()
            .block(Block::default().borders(Borders::ALL).title(" Consensus Threshold "))
            .gauge_style(Style::default().fg(Color::Green))
            .percent((agent_manager.consensus_threshold * 100.0) as u16)
            .label(format!("{}%", (agent_manager.consensus_threshold * 100.0) as u8));
            
        f.render_widget(consensus_gauge, chunks[1]);
    }
}

impl ChatRenderer for AgentRenderer {
    fn render(&self, f: &mut Frame, area: Rect) {
        // Create a default agent manager for rendering
        let agent_manager = AgentManager::placeholder();
        
        // Use the agent management renderer
        self.render_agent_management(f, area, &agent_manager);
    }
}