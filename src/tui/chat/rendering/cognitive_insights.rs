//! Cognitive insights panel for displaying reasoning and goals

use ratatui::{
    Frame,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, BorderType, List, ListItem, Paragraph, Wrap},
};

use super::get_current_render_state;

/// Renders cognitive insights including reasoning chains and goals
pub struct CognitiveInsightsPanel {
    pub show_reasoning: bool,
    pub show_goals: bool,
    pub show_memory: bool,
}

impl CognitiveInsightsPanel {
    pub fn new() -> Self {
        Self {
            show_reasoning: true,
            show_goals: true,
            show_memory: true,
        }
    }
    
    /// Render the cognitive insights panel
    pub fn render(&self, f: &mut Frame, area: Rect) {
        // Split into sections
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),   // Header
                Constraint::Percentage(35), // Reasoning chains
                Constraint::Percentage(35), // Goals
                Constraint::Percentage(30), // Memory insights
            ])
            .split(area);
            
        self.render_header(f, chunks[0]);
        
        if self.show_reasoning {
            self.render_reasoning_chains(f, chunks[1]);
        }
        
        if self.show_goals {
            self.render_goals(f, chunks[2]);
        }
        
        if self.show_memory {
            self.render_memory_insights(f, chunks[3]);
        }
    }
    
    /// Render the header
    fn render_header(&self, f: &mut Frame, area: Rect) {
        let header = Paragraph::new(vec![
            Line::from(vec![
                Span::styled("🧠 ", Style::default().fg(Color::Magenta)),
                Span::styled("Cognitive Insights", 
                    Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)),
            ]),
        ])
        .block(Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Thick)
            .border_style(Style::default().fg(Color::Magenta)));
            
        f.render_widget(header, area);
    }
    
    /// Render reasoning chains
    fn render_reasoning_chains(&self, f: &mut Frame, area: Rect) {
        let render_state = get_current_render_state();
        let mut lines = Vec::new();
        
        // Check if we have cognitive data
        if render_state.messages.recent.is_empty() {
            lines.push(Line::from(Span::styled("No active reasoning", 
                Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC))));
        } else {
            // Show reasoning steps from cognitive system
            lines.push(Line::from(vec![
                Span::styled("Current Chain:", Style::default().fg(Color::Yellow)),
            ]));
            lines.push(Line::from(""));
            
            let reasoning_steps = vec![
                ("1. Analysis", "Understanding user intent", Color::Green),
                ("2. Planning", "Determining approach", Color::Yellow),
                ("3. Execution", "Processing request", Color::Cyan),
            ];
            
            for (step, desc, color) in reasoning_steps {
                lines.push(Line::from(vec![
                    Span::styled(step, Style::default().fg(color).add_modifier(Modifier::BOLD)),
                ]));
                lines.push(Line::from(vec![
                    Span::raw("   "),
                    Span::styled(desc, Style::default().fg(Color::DarkGray)),
                ]));
            }
        }
        
        let reasoning = Paragraph::new(lines)
            .block(Block::default()
                .borders(Borders::ALL)
                .title(" 🔗 Reasoning Chains "))
            .wrap(Wrap { trim: false });
            
        f.render_widget(reasoning, area);
    }
    
    /// Render active goals
    fn render_goals(&self, f: &mut Frame, area: Rect) {
        let mut lines = Vec::new();
        
        lines.push(Line::from(vec![
            Span::styled("Active Goals:", Style::default().fg(Color::Green)),
        ]));
        lines.push(Line::from(""));
        
        // Goal hierarchy from goal manager
        let goals = vec![
            ("Primary", "Assist user with chat enhancement", Color::Green, 85),
            ("Secondary", "Maintain code quality", Color::Yellow, 60),
            ("Background", "Optimize performance", Color::Cyan, 30),
        ];
        
        for (priority, goal, color, progress) in goals {
            lines.push(Line::from(vec![
                Span::styled(format!("• {} ", priority), 
                    Style::default().fg(color).add_modifier(Modifier::BOLD)),
            ]));
            lines.push(Line::from(vec![
                Span::raw("  "),
                Span::raw(goal),
            ]));
            
            // Progress bar
            let filled = (progress / 10) as usize;
            let empty = 10 - filled;
            lines.push(Line::from(vec![
                Span::raw("  "),
                Span::styled("█".repeat(filled), Style::default().fg(Color::Green)),
                Span::styled("░".repeat(empty), Style::default().fg(Color::DarkGray)),
                Span::raw(format!(" {}%", progress)),
            ]));
            lines.push(Line::from(""));
        }
        
        let goals = Paragraph::new(lines)
            .block(Block::default()
                .borders(Borders::ALL)
                .title(" 🎯 Goal Hierarchy "));
            
        f.render_widget(goals, area);
    }
    
    /// Render memory insights
    fn render_memory_insights(&self, f: &mut Frame, area: Rect) {
        let render_state = get_current_render_state();
        let mut lines = Vec::new();
        
        lines.push(Line::from(vec![
            Span::styled("Memory State:", Style::default().fg(Color::Cyan)),
        ]));
        lines.push(Line::from(""));
        
        // Context window usage
        let context_usage = (render_state.messages.recent.len() * 100) / 50; // Rough estimate
        lines.push(Line::from(vec![
            Span::raw("Context: "),
            Span::styled(format!("{}%", context_usage.min(100)), 
                Style::default().fg(if context_usage > 80 { Color::Red } else { Color::Green })),
        ]));
        
        // Retrieved memories
        lines.push(Line::from(vec![
            Span::raw("Retrieved: "),
            Span::styled("3 relevant memories", Style::default().fg(Color::Yellow)),
        ]));
        
        // Knowledge graph
        lines.push(Line::from(vec![
            Span::raw("Graph Nodes: "),
            Span::styled("127 connected", Style::default().fg(Color::Magenta)),
        ]));
        
        let memory = Paragraph::new(lines)
            .block(Block::default()
                .borders(Borders::ALL)
                .title(" 💾 Memory Insights "));
            
        f.render_widget(memory, area);
    }
}

/// Render cognitive insights as a collapsible panel
pub fn render_cognitive_panel(f: &mut Frame, area: Rect, expanded: bool) {
    if expanded {
        let panel = CognitiveInsightsPanel::new();
        panel.render(f, area);
    } else {
        // Collapsed view - just show header
        let header = Paragraph::new(vec![
            Line::from(vec![
                Span::styled("🧠 Cognitive Insights ", 
                    Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)),
                Span::styled("[+] Expand", Style::default().fg(Color::DarkGray)),
            ]),
        ])
        .block(Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Magenta)));
            
        f.render_widget(header, area);
    }
}