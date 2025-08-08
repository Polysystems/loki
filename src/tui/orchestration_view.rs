//! Orchestration View Module
//!
//! This module provides UI components for visualizing and managing orchestration
//! of cognitive agents and distributed decision-making processes.

use anyhow::Result;
use ratatui::{prelude::*, widgets::*};
use crate::tui::state::AppState;

/// Orchestration view component for the TUI
#[derive(Debug, Clone)]
pub struct OrchestrationView {
    /// Current view state
    pub selected_tab: usize,

    /// Agent orchestration status
    pub agent_status: Vec<AgentStatus>,

    /// Decision process status
    pub decision_processes: Vec<DecisionProcess>,
}

/// Agent status information
#[derive(Debug, Clone)]
pub struct AgentStatus {
    pub id: String,
    pub specialization: String,
    pub status: String,
    pub current_tasks: usize,
    pub performance_score: f32,
}

/// Decision process information
#[derive(Debug, Clone)]
pub struct DecisionProcess {
    pub id: String,
    pub description: String,
    pub progress: f32,
    pub participants: usize,
    pub deadline: Option<String>,
}

impl Default for OrchestrationView {
    fn default() -> Self {
        Self {
            selected_tab: 0,
            agent_status: Vec::new(),
            decision_processes: Vec::new(),
        }
    }
}

impl OrchestrationView {
    /// Create a new orchestration view
    pub fn new() -> Self {
        Self::default()
    }

    /// Update the view with current app state
    pub fn update(&mut self, app_state: &AppState) -> Result<()> {
        // Update agent status from app state
        self.agent_status = app_state
            .available_agents
            .iter()
            .enumerate()
            .map(|(i, agent)| AgentStatus {
                id: format!("agent_{}", i),
                specialization: agent.name.clone(),
                status: "Active".to_string(),
                current_tasks: 0,
                performance_score: 1.0,
            })
            .collect();

        Ok(())
    }

    /// Render the orchestration view
    pub fn render(&self, f: &mut Frame, area: Rect) -> Result<()> {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),  // Header
                Constraint::Min(10),    // Content
            ])
            .split(area);

        // Render header
        let header = Paragraph::new("🎭 Orchestration Dashboard")
            .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
            .alignment(Alignment::Center)
            .block(Block::default().borders(Borders::ALL));
        f.render_widget(header, chunks[0]);

        // Render content tabs
        let content_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(50),  // Agent status
                Constraint::Percentage(50),  // Decision processes
            ])
            .split(chunks[1]);

        self.render_agent_status(f, content_chunks[0])?;
        self.render_decision_processes(f, content_chunks[1])?;

        Ok(())
    }

    /// Render agent status panel
    fn render_agent_status(&self, f: &mut Frame, area: Rect) -> Result<()> {
        let items: Vec<ListItem> = self
            .agent_status
            .iter()
            .map(|agent| {
                ListItem::new(format!(
                    "{}: {} ({}) - Tasks: {}, Score: {:.2}",
                    agent.id,
                    agent.specialization,
                    agent.status,
                    agent.current_tasks,
                    agent.performance_score
                ))
            })
            .collect();

        let list = List::new(items)
            .block(Block::default()
                .title("Agent Status")
                .borders(Borders::ALL))
            .style(Style::default().fg(Color::White))
            .highlight_style(Style::default().add_modifier(Modifier::BOLD))
            .highlight_symbol(">> ");

        f.render_widget(list, area);
        Ok(())
    }

    /// Render decision processes panel
    fn render_decision_processes(&self, f: &mut Frame, area: Rect) -> Result<()> {
        let items: Vec<ListItem> = self
            .decision_processes
            .iter()
            .map(|process| {
                ListItem::new(format!(
                    "{}: {} - Progress: {:.1}%, Participants: {}",
                    process.id,
                    process.description,
                    process.progress * 100.0,
                    process.participants
                ))
            })
            .collect();

        let list = List::new(items)
            .block(Block::default()
                .title("Decision Processes")
                .borders(Borders::ALL))
            .style(Style::default().fg(Color::White))
            .highlight_style(Style::default().add_modifier(Modifier::BOLD))
            .highlight_symbol(">> ");

        f.render_widget(list, area);
        Ok(())
    }

    /// Handle navigation input
    pub fn handle_input(&mut self, key: ratatui::crossterm::event::KeyCode) -> Result<()> {
        match key {
            ratatui::crossterm::event::KeyCode::Tab => {
                self.selected_tab = (self.selected_tab + 1) % 2;
            }
            _ => {}
        }
        Ok(())
    }
}
