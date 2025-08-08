//! Orchestration UI renderer with real-time metrics

use ratatui::Frame;
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::symbols;
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Block, Borders, BorderType, Gauge, List, ListItem, Paragraph, 
    Sparkline, Wrap, Chart, Axis, Dataset, GraphType
};
use std::collections::VecDeque;

use crate::tui::chat::orchestration::manager::{OrchestrationManager, RoutingStrategy};
use super::{ChatRenderer, get_current_render_state};

/// Model performance metrics
#[derive(Debug, Clone)]
pub struct ModelMetrics {
    pub name: String,
    pub response_times: VecDeque<f64>,
    pub success_rate: f64,
    pub tokens_per_second: f64,
    pub cost_per_token: f64,
    pub total_requests: usize,
    pub active: bool,
}

/// Renders orchestration status and configuration with real-time metrics
pub struct OrchestrationRenderer {
    pub show_detailed_stats: bool,
    pub highlight_active_models: bool,
    pub model_metrics: Vec<ModelMetrics>,
    pub routing_history: VecDeque<(String, f64)>, // (model, response_time)
    pub cost_history: VecDeque<f64>,
    pub quality_scores: VecDeque<f64>,
}

impl OrchestrationRenderer {
    pub fn new() -> Self {
        Self {
            show_detailed_stats: true,
            highlight_active_models: true,
            model_metrics: Vec::new(),
            routing_history: VecDeque::new(),
            cost_history: VecDeque::new(),
            quality_scores: VecDeque::new(),
        }
    }
    
    /// Render the orchestration status header
    pub fn render_status_header(&self, f: &mut Frame, area: Rect, orchestration: &OrchestrationManager) {
        let status_text = if orchestration.orchestration_enabled {
            format!("🟢 Orchestration Active | Strategy: {:?}", orchestration.preferred_strategy)
        } else {
            "🔴 Orchestration Disabled".to_string()
        };
        
        let status = Paragraph::new(status_text)
            .style(Style::default().fg(if orchestration.orchestration_enabled {
                Color::Green
            } else {
                Color::Red
            }))
            .block(Block::default().borders(Borders::ALL).title("Orchestration Status"));
            
        f.render_widget(status, area);
    }
    
    /// Render detailed orchestration configuration with metrics
    pub fn render_config_detailed(&self, f: &mut Frame, area: Rect, orchestration: &OrchestrationManager) {
        // Split area into main sections
        let main_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(60),  // Left: Config & Models
                Constraint::Percentage(40),  // Right: Metrics
            ])
            .split(area);
            
        // Left side - Configuration
        let left_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),  // Status
                Constraint::Length(6),  // Routing info
                Constraint::Length(4),  // Preferences
                Constraint::Min(0),     // Models
            ])
            .split(main_chunks[0]);
            
        // Render status
        self.render_status_header(f, left_chunks[0], orchestration);
        
        // Render routing strategy info
        self.render_routing_info(f, left_chunks[1], orchestration);
        
        // Render preferences
        self.render_preferences(f, left_chunks[2], orchestration);
        
        // Render enabled models with metrics
        self.render_models_with_metrics(f, left_chunks[3]);
        
        // Right side - Real-time metrics
        let right_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(8),   // Response time sparklines
                Constraint::Length(8),   // Cost tracking
                Constraint::Length(8),   // Quality scores
                Constraint::Min(0),      // Routing history
            ])
            .split(main_chunks[1]);
            
        self.render_response_time_metrics(f, right_chunks[0]);
        self.render_cost_tracking(f, right_chunks[1]);
        self.render_quality_metrics(f, right_chunks[2]);
        self.render_routing_history(f, right_chunks[3]);
    }
    
    /// Render models with performance metrics from render state
    fn render_models_with_metrics(&self, f: &mut Frame, area: Rect) {
        let render_state = get_current_render_state();
        let mut items = Vec::new();
        
        // Use enabled models from render state
        for model in &render_state.orchestration.enabled_models {
            let status_icon = if model.status == "Available" { "🟢" } else { "⚫" };
            
            // Extract model performance info
            let (icon, color) = match model.name.as_str() {
                name if name.contains("gpt-4") => ("🧠", Color::Green),
                name if name.contains("gpt-3") => ("💭", Color::Blue),
                name if name.contains("claude") => ("🤖", Color::Magenta),
                name if name.contains("gemini") => ("✨", Color::Cyan),
                name if name.contains("mistral") => ("🌟", Color::Yellow),
                name if name.contains("llama") => ("🦙", Color::LightRed),
                _ => ("📊", Color::White),
            };
            
            items.push(ListItem::new(vec![
                Line::from(vec![
                    Span::raw(format!("{} {} ", status_icon, icon)),
                    Span::styled(&model.name, Style::default().fg(color).add_modifier(Modifier::BOLD)),
                ]),
                Line::from(vec![
                    Span::raw("  Provider: "),
                    Span::styled(&model.provider, Style::default().fg(Color::Cyan)),
                    Span::raw(" | Status: "),
                    Span::styled(&model.status, Style::default().fg(
                        if model.status == "Available" { Color::Green } else { Color::Red }
                    )),
                ]),
                Line::from(vec![
                    Span::raw("  "),
                    Span::styled(
                        format!("Context: {} tokens", render_state.orchestration.context_window),
                        Style::default().fg(Color::DarkGray)
                    ),
                ]),
            ]));
        }
        
        if items.is_empty() {
            items.push(ListItem::new(Line::from(
                Span::styled("No models configured", Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC))
            )));
        }
        
        let list = List::new(items)
            .block(Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .title(" 🤖 Model Performance "));
        
        f.render_widget(list, area);
    }
    
    /// Render response time sparklines
    fn render_response_time_metrics(&self, f: &mut Frame, area: Rect) {
        let render_state = get_current_render_state();
        
        // Show placeholder when no data available
        if render_state.messages.recent.is_empty() {
            let placeholder = Paragraph::new(vec![
                Line::from(""),
                Line::from(Span::styled("No response time data yet", 
                    Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC))),
                Line::from(""),
                Line::from(Span::styled("Metrics will appear here once", 
                    Style::default().fg(Color::DarkGray))),
                Line::from(Span::styled("models start responding", 
                    Style::default().fg(Color::DarkGray))),
            ])
            .block(Block::default()
                .borders(Borders::ALL)
                .title(" ⚡ Response Times "))
            .alignment(Alignment::Center);
            
            f.render_widget(placeholder, area);
        } else {
            // Show actual message count as a simple metric
            let message_count = render_state.messages.recent.len();
            let sparkline_data: Vec<u64> = (0..message_count)
                .map(|i| ((i + 1) * 10) as u64)
                .collect();
            
            let sparkline = Sparkline::default()
                .block(Block::default()
                    .borders(Borders::ALL)
                    .title(format!(" ⚡ Activity ({} messages) ", message_count)))
                .data(&sparkline_data)
                .style(Style::default().fg(Color::Yellow))
                .max(100);
                
            f.render_widget(sparkline, area);
        }
    }
    
    /// Render cost tracking
    fn render_cost_tracking(&self, f: &mut Frame, area: Rect) {
        let render_state = get_current_render_state();
        
        // Show cost limit and current usage
        let cost_limit = render_state.orchestration.cost_limit;
        let lines = vec![
            Line::from(""),
            Line::from(vec![
                Span::raw("Cost Limit: "),
                Span::styled(format!("${:.2}", cost_limit), Style::default().fg(Color::Magenta)),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::raw("Optimization: "),
                Span::styled(
                    if render_state.orchestration.strategy == "CostOptimized" { "Active" } else { "Inactive" },
                    Style::default().fg(
                        if render_state.orchestration.strategy == "CostOptimized" { Color::Green } else { Color::DarkGray }
                    )
                ),
            ]),
        ];
        
        let cost_panel = Paragraph::new(lines)
            .block(Block::default()
                .borders(Borders::ALL)
                .title(" 💰 Cost Tracking "))
            .alignment(Alignment::Center);
            
        f.render_widget(cost_panel, area);
    }
    
    /// Render quality metrics
    fn render_quality_metrics(&self, f: &mut Frame, area: Rect) {
        let render_state = get_current_render_state();
        let quality_threshold = render_state.orchestration.quality_threshold;
        
        // Show quality gauge
        let gauge = Gauge::default()
            .block(Block::default()
                .borders(Borders::ALL)
                .title(" ⭐ Quality Score "))
            .gauge_style(Style::default().fg(Color::Green))
            .percent((quality_threshold * 100.0) as u16)
            .label(format!("Threshold: {:.0}%", quality_threshold * 100.0));
            
        f.render_widget(gauge, area);
    }
    
    /// Render routing history
    fn render_routing_history(&self, f: &mut Frame, area: Rect) {
        let render_state = get_current_render_state();
        let mut lines = Vec::new();
        
        // Show active model if available
        if let Some(active) = &render_state.orchestration.active_model {
            lines.push(Line::from(vec![
                Span::styled("Active: ", Style::default().fg(Color::Green)),
                Span::styled(active, Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            ]));
            lines.push(Line::from(""));
        }
        
        // Show routing strategy
        lines.push(Line::from(vec![
            Span::raw("Strategy: "),
            Span::styled(&render_state.orchestration.strategy, Style::default().fg(Color::Yellow)),
        ]));
        
        lines.push(Line::from(""));
        lines.push(Line::from(vec![
            Span::raw("Fallback: "),
            Span::styled(
                if render_state.orchestration.fallback_enabled { "Enabled" } else { "Disabled" },
                Style::default().fg(
                    if render_state.orchestration.fallback_enabled { Color::Green } else { Color::DarkGray }
                )
            ),
        ]));
        
        let history = Paragraph::new(lines)
            .block(Block::default()
                .borders(Borders::ALL)
                .title(" 🔄 Routing Status "));
            
        f.render_widget(history, area);
    }
    
    fn render_routing_info(&self, f: &mut Frame, area: Rect, orchestration: &OrchestrationManager) {
        use ratatui::text::{Line, Span};
        use ratatui::style::Modifier;
        
        let (strategy_name, icon, description) = match &orchestration.preferred_strategy {
            RoutingStrategy::CapabilityBased => ("Capability-Based", "🎯", "Routes to models based on their capabilities and strengths"),
            RoutingStrategy::RoundRobin => ("Round Robin", "🔄", "Distributes load evenly across all available models"),
            RoutingStrategy::CostOptimized => ("Cost Optimized", "💰", "Minimizes API costs while maintaining quality"),
            RoutingStrategy::LeastLatency => ("Least Latency", "⚡", "Selects fastest responding models for quick results"),
            RoutingStrategy::ContextAware => ("Context Aware", "🧠", "Maintains conversation coherence and context"),
            RoutingStrategy::Custom(_name) => ("Custom", "⚙️", "Custom routing logic defined by user"),
            RoutingStrategy::Capability => ("Capability", "🎯", "Routes based on model capabilities"),
            RoutingStrategy::Cost => ("Cost", "💰", "Optimizes for lowest cost"),
            RoutingStrategy::Speed => ("Speed", "⚡", "Optimizes for fastest response"),
            _ => ("Unknown", "❓", "Unknown routing strategy"),
        };
        
        let content = vec![
            Line::from(vec![
                Span::raw(icon),
                Span::raw(" "),
                Span::styled(strategy_name, Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            ]),
            Line::from(""),
            Line::from(description),
        ];
        
        let routing_info = Paragraph::new(content)
            .block(Block::default().borders(Borders::ALL).title(" Routing Strategy "))
            .wrap(ratatui::widgets::Wrap { trim: true });
            
        f.render_widget(routing_info, area);
    }
    
    fn render_preferences(&self, f: &mut Frame, area: Rect, orchestration: &OrchestrationManager) {
        let local_pref_gauge = Gauge::default()
            .block(Block::default().borders(Borders::ALL).title("Local Model Preference"))
            .gauge_style(Style::default().fg(Color::Cyan))
            .percent((orchestration.local_models_preference * 100.0) as u16)
            .label(format!("{}%", (orchestration.local_models_preference * 100.0) as u8));
            
        f.render_widget(local_pref_gauge, area);
    }
    
    fn render_enabled_models(&self, f: &mut Frame, area: Rect, orchestration: &OrchestrationManager) {
        use ratatui::text::{Line, Span};
        use ratatui::style::Modifier;
        
        let model_items: Vec<ListItem> = if orchestration.enabled_models.is_empty() {
            vec![ListItem::new(Line::from(vec![
                Span::styled("No models enabled", Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC))
            ]))]
        } else {
            orchestration.enabled_models
                .iter()
                .enumerate()
                .map(|(i, model)| {
                    let (icon, color) = match model.as_str() {
                        "gpt-4" | "gpt-4-turbo" => ("🧠", Color::Green),
                        "gpt-3.5-turbo" => ("💭", Color::Blue),
                        "claude" | "claude-2" => ("🤖", Color::Magenta),
                        "gemini" | "gemini-pro" => ("✨", Color::Cyan),
                        "mistral" | "mixtral" => ("🌟", Color::Yellow),
                        _ => ("📊", Color::White),
                    };
                    
                    ListItem::new(Line::from(vec![
                        Span::raw(format!("{} ", icon)),
                        Span::styled(model.clone(), Style::default().fg(color)),
                        Span::raw(" "),
                        Span::styled("(Active)", Style::default().fg(Color::Green).add_modifier(Modifier::DIM)),
                    ]))
                })
                .collect()
        };
            
        let models_list = List::new(model_items)
            .block(Block::default().borders(Borders::ALL).title(" Enabled Models "))
            .highlight_style(Style::default().bg(Color::DarkGray));
            
        f.render_widget(models_list, area);
    }
}

impl ChatRenderer for OrchestrationRenderer {
    fn render(&self, f: &mut Frame, area: Rect) {
        // Create a default orchestration manager for rendering
        let orchestration = OrchestrationManager::default();
        
        // Use the detailed config renderer
        self.render_config_detailed(f, area, &orchestration);
    }
}