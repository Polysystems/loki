//! Orchestration subtab implementation

use std::sync::Arc;
use tokio::sync::RwLock;
use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, List, ListItem, Paragraph, Wrap},
    Frame,
};
use crossterm::event::{KeyCode, KeyEvent};
use anyhow::Result;

use super::SubtabController;
use crate::tui::chat::orchestration::OrchestrationManager;
use crate::tui::chat::orchestration::RoutingStrategy;

/// Orchestration configuration subtab
pub struct OrchestrationTab {
    /// Orchestration manager
    manager: Arc<RwLock<OrchestrationManager>>,
    
    /// Current selection index
    selected_index: usize,
    
    /// Edit mode
    edit_mode: bool,
    
    /// Input buffer for editing
    input_buffer: String,
    
    /// Available configuration options
    config_options: Vec<ConfigOption>,
}

#[derive(Debug, Clone)]
enum ConfigOption {
    ToggleOrchestration,
    SelectStrategy,
    SetParallelAgents,
    ToggleFallback,
    SetContextWindow,
    SetTemperature,
    ToggleStreaming,
    ConfigureRetries,
    ModelSelection,
}

impl OrchestrationTab {
    /// Create a new orchestration tab
    pub fn new(manager: Arc<RwLock<OrchestrationManager>>) -> Self {
        let config_options = vec![
            ConfigOption::ToggleOrchestration,
            ConfigOption::SelectStrategy,
            ConfigOption::SetParallelAgents,
            ConfigOption::ToggleFallback,
            ConfigOption::SetContextWindow,
            ConfigOption::SetTemperature,
            ConfigOption::ToggleStreaming,
            ConfigOption::ConfigureRetries,
            ConfigOption::ModelSelection,
        ];
        
        Self {
            manager,
            selected_index: 0,
            edit_mode: false,
            input_buffer: String::new(),
            config_options,
        }
    }
    
    /// Render the configuration panel
    fn render_config_panel(&self, f: &mut Frame, area: Rect, config: &OrchestrationConfig) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),  // Title
                Constraint::Min(10),    // Options list
                Constraint::Length(4),  // Help text
            ])
            .split(area);
        
        // Title
        let title = Paragraph::new("🎭 Orchestration Configuration")
            .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
            .alignment(Alignment::Center)
            .block(Block::default().borders(Borders::BOTTOM));
        f.render_widget(title, chunks[0]);
        
        // Configuration options
        let items: Vec<ListItem> = self.config_options
            .iter()
            .enumerate()
            .map(|(i, option)| {
                let selected = i == self.selected_index;
                let (icon, label, value) = self.format_option(option, config);
                
                let style = if selected {
                    Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
                } else {
                    Style::default()
                };
                
                let content = if self.edit_mode && selected {
                    format!("{} {} > {}", icon, label, self.input_buffer)
                } else {
                    format!("{} {}: {}", icon, label, value)
                };
                
                ListItem::new(content).style(style)
            })
            .collect();
        
        let list = List::new(items)
            .block(Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .title(" Options "));
        f.render_widget(list, chunks[1]);
        
        // Help text
        let help_text = if self.edit_mode {
            "Press Enter to save, Esc to cancel"
        } else {
            "↑/↓: Navigate | Enter: Edit | Space: Toggle | q: Back"
        };
        
        let help = Paragraph::new(help_text)
            .style(Style::default().fg(Color::DarkGray))
            .alignment(Alignment::Center)
            .block(Block::default().borders(Borders::TOP));
        f.render_widget(help, chunks[2]);
    }
    
    /// Render the status panel
    fn render_status_panel(&self, f: &mut Frame, area: Rect, config: &OrchestrationConfig) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),  // Title
                Constraint::Min(5),     // Status info
            ])
            .split(area);
        
        // Title
        let title = Paragraph::new("📊 Current Status")
            .style(Style::default().fg(Color::Green).add_modifier(Modifier::BOLD))
            .alignment(Alignment::Center)
            .block(Block::default().borders(Borders::BOTTOM));
        f.render_widget(title, chunks[0]);
        
        // Status information
        let mut status_lines = vec![
            Line::from(vec![
                Span::raw("Status: "),
                Span::styled(
                    if config.enabled { "Active" } else { "Inactive" },
                    Style::default().fg(if config.enabled { Color::Green } else { Color::Red })
                ),
            ]),
            Line::from(vec![
                Span::raw("Strategy: "),
                Span::styled(
                    format!("{:?}", config.strategy),
                    Style::default().fg(Color::Cyan)
                ),
            ]),
            Line::from(vec![
                Span::raw("Models: "),
                Span::styled(
                    format!("{} active", config.models.len()),
                    Style::default().fg(Color::Blue)
                ),
            ]),
            Line::from(vec![
                Span::raw("Parallel Agents: "),
                Span::styled(
                    config.parallel_agents.to_string(),
                    Style::default().fg(Color::Magenta)
                ),
            ]),
        ];
        
        if !config.models.is_empty() {
            status_lines.push(Line::from(""));
            status_lines.push(Line::from(Span::styled("Active Models:", Style::default().add_modifier(Modifier::UNDERLINED))));
            for model in &config.models {
                status_lines.push(Line::from(format!("  • {}", model)));
            }
        }
        
        let status = Paragraph::new(status_lines)
            .block(Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .title(" Status "))
            .wrap(Wrap { trim: false });
        f.render_widget(status, chunks[1]);
    }
    
    /// Format configuration option for display
    fn format_option(&self, option: &ConfigOption, config: &OrchestrationConfig) -> (&str, &str, String) {
        match option {
            ConfigOption::ToggleOrchestration => (
                if config.enabled { "✅" } else { "❌" },
                "Orchestration",
                if config.enabled { "Enabled" } else { "Disabled" }.to_string()
            ),
            ConfigOption::SelectStrategy => (
                "🎯",
                "Routing Strategy",
                format!("{:?}", config.strategy)
            ),
            ConfigOption::SetParallelAgents => (
                "👥",
                "Parallel Agents",
                config.parallel_agents.to_string()
            ),
            ConfigOption::ToggleFallback => (
                if config.fallback { "✅" } else { "❌" },
                "Fallback",
                if config.fallback { "Enabled" } else { "Disabled" }.to_string()
            ),
            ConfigOption::SetContextWindow => (
                "📏",
                "Context Window",
                format!("{} tokens", config.context_window)
            ),
            ConfigOption::SetTemperature => (
                "🌡️",
                "Temperature",
                format!("{:.2}", config.temperature)
            ),
            ConfigOption::ToggleStreaming => (
                if config.streaming { "✅" } else { "❌" },
                "Streaming",
                if config.streaming { "Enabled" } else { "Disabled" }.to_string()
            ),
            ConfigOption::ConfigureRetries => (
                "🔄",
                "Max Retries",
                config.max_retries.to_string()
            ),
            ConfigOption::ModelSelection => (
                "🤖",
                "Model Selection",
                format!("{} models", config.models.len())
            ),
        }
    }
    
    /// Handle option selection
    async fn handle_option_select(&mut self) -> Result<()> {
        let option = &self.config_options[self.selected_index];
        
        match option {
            ConfigOption::ToggleOrchestration => {
                let mut manager = self.manager.write().await;
                manager.orchestration_enabled = !manager.orchestration_enabled;
            }
            ConfigOption::ToggleFallback => {
                let mut manager = self.manager.write().await;
                manager.allow_fallback = !manager.allow_fallback;
            }
            ConfigOption::ToggleStreaming => {
                let mut manager = self.manager.write().await;
                manager.stream_responses = !manager.stream_responses;
            }
            _ => {
                // Enter edit mode for other options
                self.edit_mode = true;
                let manager = self.manager.read().await;
                
                // Pre-fill input buffer with current value
                self.input_buffer = match option {
                    ConfigOption::SetParallelAgents => manager.parallel_models.to_string(),
                    ConfigOption::SetContextWindow => manager.context_window.to_string(),
                    ConfigOption::SetTemperature => format!("{:.2}", manager.temperature),
                    ConfigOption::ConfigureRetries => manager.max_retries.to_string(),
                    _ => String::new(),
                };
            }
        }
        
        Ok(())
    }
    
    /// Save edited value
    async fn save_edit(&mut self) -> Result<()> {
        let option = &self.config_options[self.selected_index];
        let mut manager = self.manager.write().await;
        
        match option {
            ConfigOption::SelectStrategy => {
                // Parse strategy from input
                match self.input_buffer.to_lowercase().as_str() {
                    "roundrobin" | "round" => manager.preferred_strategy = RoutingStrategy::RoundRobin,
                    "leastlatency" | "latency" => manager.preferred_strategy = RoutingStrategy::LeastLatency,
                    "contextaware" | "context" => manager.preferred_strategy = RoutingStrategy::ContextAware,
                    _ => {}
                }
            }
            ConfigOption::SetParallelAgents => {
                if let Ok(value) = self.input_buffer.parse::<usize>() {
                    if value > 0 && value <= 10 {
                        manager.parallel_models = value;
                    }
                }
            }
            ConfigOption::SetContextWindow => {
                if let Ok(value) = self.input_buffer.parse::<usize>() {
                    if value >= 100 && value <= 128000 {
                        manager.context_window = value;
                    }
                }
            }
            ConfigOption::SetTemperature => {
                if let Ok(value) = self.input_buffer.parse::<f32>() {
                    if value >= 0.0 && value <= 2.0 {
                        manager.temperature = value;
                    }
                }
            }
            ConfigOption::ConfigureRetries => {
                if let Ok(value) = self.input_buffer.parse::<u32>() {
                    if value <= 10 {
                        manager.max_retries = value;
                    }
                }
            }
            _ => {}
        }
        
        self.edit_mode = false;
        self.input_buffer.clear();
        Ok(())
    }
}

impl SubtabController for OrchestrationTab {
    fn render(&mut self, f: &mut Frame, area: Rect) {
        // Use tokio's block_on to get the config synchronously for rendering
        let config = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                let manager = self.manager.read().await;
                OrchestrationConfig {
                    enabled: manager.orchestration_enabled,
                    strategy: manager.preferred_strategy.clone(),
                    models: manager.enabled_models.clone(),
                    parallel_agents: manager.parallel_models,
                    fallback: manager.allow_fallback,
                    context_window: manager.context_window,
                    temperature: manager.temperature,
                    streaming: manager.stream_responses,
                    max_retries: manager.max_retries,
                }
            })
        });
        
        // Split area into two panels
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(60),  // Configuration panel
                Constraint::Percentage(40),  // Status panel
            ])
            .split(area);
        
        self.render_config_panel(f, chunks[0], &config);
        self.render_status_panel(f, chunks[1], &config);
    }
    
    fn handle_input(&mut self, key: KeyEvent) -> Result<()> {
        if self.edit_mode {
            match key.code {
                KeyCode::Char(c) => {
                    self.input_buffer.push(c);
                }
                KeyCode::Backspace => {
                    self.input_buffer.pop();
                }
                KeyCode::Enter => {
                    tokio::task::block_in_place(|| {
                        tokio::runtime::Handle::current().block_on(self.save_edit())
                    })?;
                }
                KeyCode::Esc => {
                    self.edit_mode = false;
                    self.input_buffer.clear();
                }
                _ => {}
            }
        } else {
            match key.code {
                KeyCode::Up => {
                    if self.selected_index > 0 {
                        self.selected_index -= 1;
                    }
                }
                KeyCode::Down => {
                    if self.selected_index < self.config_options.len() - 1 {
                        self.selected_index += 1;
                    }
                }
                KeyCode::Enter | KeyCode::Char(' ') => {
                    tokio::task::block_in_place(|| {
                        tokio::runtime::Handle::current().block_on(self.handle_option_select())
                    })?;
                }
                _ => {}
            }
        }
        
        Ok(())
    }
    
    fn update(&mut self) -> Result<()> {
        // No continuous updates needed
        Ok(())
    }
    
    fn name(&self) -> &str {
        "Orchestration"
    }
}

/// Configuration snapshot for rendering
struct OrchestrationConfig {
    enabled: bool,
    strategy: RoutingStrategy,
    models: Vec<String>,
    parallel_agents: usize,
    fallback: bool,
    context_window: usize,
    temperature: f32,
    streaming: bool,
    max_retries: u32,
}