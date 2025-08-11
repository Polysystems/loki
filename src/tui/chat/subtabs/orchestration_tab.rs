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
    
    /// Model selection mode
    model_selection_mode: bool,
    
    /// Selected model index (for model selection)
    model_selection_index: usize,
    
    /// Available models
    available_models: Vec<ModelInfo>,
    
    /// Selected models
    selected_models: Vec<String>,
    
    /// Input buffer for editing
    input_buffer: String,
    
    /// Available configuration options
    config_options: Vec<ConfigOption>,
}

#[derive(Debug, Clone)]
struct ModelInfo {
    name: String,
    provider: String,
    is_local: bool,
    has_gpu: bool,
    capabilities: Vec<String>,
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
    OrchestrationMode,
    SelectAllLocal,
    SelectAllAPI,
    ClearSelection,
}

impl OrchestrationTab {
    /// Create a new orchestration tab
    pub fn new(manager: Arc<RwLock<OrchestrationManager>>) -> Self {
        let config_options = vec![
            ConfigOption::ToggleOrchestration,
            ConfigOption::OrchestrationMode,
            ConfigOption::ModelSelection,
            ConfigOption::SelectAllLocal,
            ConfigOption::SelectAllAPI,
            ConfigOption::ClearSelection,
            ConfigOption::SelectStrategy,
            ConfigOption::SetParallelAgents,
            ConfigOption::ToggleFallback,
            ConfigOption::SetContextWindow,
            ConfigOption::SetTemperature,
            ConfigOption::ToggleStreaming,
            ConfigOption::ConfigureRetries,
        ];
        
        // Initialize with some example models
        let available_models = Self::discover_available_models();
        
        // Initialize selected models from the manager
        let selected_models = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                let mgr = manager.read().await;
                mgr.enabled_models.clone()
            })
        });
        
        Self {
            manager,
            selected_index: 0,
            edit_mode: false,
            model_selection_mode: false,
            model_selection_index: 0,
            available_models,
            selected_models,
            input_buffer: String::new(),
            config_options,
        }
    }
    
    /// Discover available models from various sources
    fn discover_available_models() -> Vec<ModelInfo> {
        let mut models = Vec::new();
        
        // Check for Ollama models
        if let Ok(output) = std::process::Command::new("ollama")
            .arg("list")
            .output()
        {
            if output.status.success() {
                let output_str = String::from_utf8_lossy(&output.stdout);
                for line in output_str.lines().skip(1) {
                    if let Some(model_name) = line.split_whitespace().next() {
                        if !model_name.is_empty() {
                            models.push(ModelInfo {
                                name: model_name.to_string(),
                                provider: "ollama".to_string(),
                                is_local: true,
                                has_gpu: cfg!(any(feature = "cuda", feature = "metal")),
                                capabilities: vec!["chat".to_string(), "completion".to_string()],
                            });
                        }
                    }
                }
            }
        }
        
        // Add API models if keys are configured
        if std::env::var("OPENAI_API_KEY").is_ok() {
            models.extend(vec![
                ModelInfo {
                    name: "gpt-4".to_string(),
                    provider: "openai".to_string(),
                    is_local: false,
                    has_gpu: false,
                    capabilities: vec!["chat".to_string(), "code".to_string(), "reasoning".to_string()],
                },
                ModelInfo {
                    name: "gpt-4-turbo".to_string(),
                    provider: "openai".to_string(),
                    is_local: false,
                    has_gpu: false,
                    capabilities: vec!["chat".to_string(), "code".to_string(), "fast".to_string()],
                },
            ]);
        }
        
        if std::env::var("ANTHROPIC_API_KEY").is_ok() {
            models.extend(vec![
                ModelInfo {
                    name: "claude-3-opus".to_string(),
                    provider: "anthropic".to_string(),
                    is_local: false,
                    has_gpu: false,
                    capabilities: vec!["chat".to_string(), "code".to_string(), "analysis".to_string()],
                },
                ModelInfo {
                    name: "claude-3-sonnet".to_string(),
                    provider: "anthropic".to_string(),
                    is_local: false,
                    has_gpu: false,
                    capabilities: vec!["chat".to_string(), "balanced".to_string()],
                },
            ]);
        }
        
        models
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
                if self.selected_models.is_empty() {
                    "Press Enter to select".to_string()
                } else {
                    format!("{} selected", self.selected_models.len())
                }
            ),
            ConfigOption::OrchestrationMode => (
                "🎭",
                "Orchestration Mode",
                match config.parallel_agents {
                    1 => "Single Model".to_string(),
                    2..=3 => "Ensemble".to_string(),
                    4..=5 => "Voting".to_string(),
                    _ => "Adaptive".to_string(),
                }
            ),
            ConfigOption::SelectAllLocal => (
                "📍",
                "Select All Local",
                "Quick action".to_string()
            ),
            ConfigOption::SelectAllAPI => (
                "☁️",
                "Select All API",
                "Quick action".to_string()
            ),
            ConfigOption::ClearSelection => (
                "🗑️",
                "Clear Selection",
                "Quick action".to_string()
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
            ConfigOption::ModelSelection => {
                // Enter model selection mode
                self.model_selection_mode = true;
                self.model_selection_index = 0;
            }
            ConfigOption::OrchestrationMode => {
                // Cycle through orchestration modes
                let mut manager = self.manager.write().await;
                manager.parallel_models = match manager.parallel_models {
                    1 => 3,  // Single -> Ensemble (3 models)
                    3 => 5,  // Ensemble -> Voting (5 models)
                    5 => 2,  // Voting -> Sequential (2 models)
                    _ => 1,  // Back to Single
                };
                // Update ensemble configuration based on mode
                manager.ensemble_enabled = manager.parallel_models > 1;
            }
            ConfigOption::SelectAllLocal => {
                // Select all local models
                self.selected_models.clear();
                for model in &self.available_models {
                    if model.is_local {
                        self.selected_models.push(model.name.clone());
                    }
                }
                self.update_manager_models().await;
            }
            ConfigOption::SelectAllAPI => {
                // Select all API models
                self.selected_models.clear();
                for model in &self.available_models {
                    if !model.is_local {
                        self.selected_models.push(model.name.clone());
                    }
                }
                self.update_manager_models().await;
            }
            ConfigOption::ClearSelection => {
                // Clear all selected models
                self.selected_models.clear();
                self.update_manager_models().await;
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
    
    /// Update the manager with selected models
    async fn update_manager_models(&self) {
        let mut manager = self.manager.write().await;
        manager.enabled_models = self.selected_models.clone();
    }
    
    /// Render model selection UI
    fn render_model_selection(&mut self, f: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),  // Title
                Constraint::Min(10),    // Models list
                Constraint::Length(3),  // Help
            ])
            .split(area);
        
        // Title
        let title = Paragraph::new("🤖 Select Models for Orchestration")
            .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
            .alignment(Alignment::Center)
            .block(Block::default().borders(Borders::BOTTOM));
        f.render_widget(title, chunks[0]);
        
        // Models list with checkboxes
        let items: Vec<ListItem> = self.available_models
            .iter()
            .enumerate()
            .map(|(i, model)| {
                let selected = i == self.model_selection_index;
                let checked = self.selected_models.contains(&model.name);
                
                let checkbox = if checked { "✅" } else { "☐" };
                let gpu_icon = if model.has_gpu { "🚀" } else { "" };
                let location = if model.is_local { "📍" } else { "☁️" };
                
                let line = format!(
                    "{} {} {} {} ({}) {} - {}",
                    if selected { "▶" } else { " " },
                    checkbox,
                    location,
                    model.name,
                    model.provider,
                    gpu_icon,
                    model.capabilities.join(", ")
                );
                
                let style = if selected {
                    Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
                } else if checked {
                    Style::default().fg(Color::Green)
                } else {
                    Style::default()
                };
                
                ListItem::new(line).style(style)
            })
            .collect();
        
        let list = List::new(items)
            .block(Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .title(format!(" {} models available, {} selected ", 
                    self.available_models.len(),
                    self.selected_models.len()
                )));
        f.render_widget(list, chunks[1]);
        
        // Help text
        let help = vec![
            Line::from(vec![
                Span::styled("[Space]", Style::default().fg(Color::Yellow)),
                Span::raw(" Toggle  "),
                Span::styled("[a]", Style::default().fg(Color::Yellow)),
                Span::raw(" Select All  "),
                Span::styled("[l]", Style::default().fg(Color::Yellow)),
                Span::raw(" Local Only  "),
                Span::styled("[c]", Style::default().fg(Color::Yellow)),
                Span::raw(" Clear  "),
                Span::styled("[Enter]", Style::default().fg(Color::Yellow)),
                Span::raw(" Confirm  "),
                Span::styled("[Esc]", Style::default().fg(Color::Yellow)),
                Span::raw(" Cancel"),
            ]),
        ];
        
        let help_widget = Paragraph::new(help)
            .style(Style::default().fg(Color::DarkGray))
            .alignment(Alignment::Center)
            .block(Block::default().borders(Borders::TOP));
        f.render_widget(help_widget, chunks[2]);
    }
}

impl SubtabController for OrchestrationTab {
    fn render(&mut self, f: &mut Frame, area: Rect) {
        // If in model selection mode, render that instead
        if self.model_selection_mode {
            self.render_model_selection(f, area);
            return;
        }
        
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
        // Handle model selection mode
        if self.model_selection_mode {
            match key.code {
                KeyCode::Up => {
                    if self.model_selection_index > 0 {
                        self.model_selection_index -= 1;
                    }
                }
                KeyCode::Down => {
                    if self.model_selection_index < self.available_models.len() - 1 {
                        self.model_selection_index += 1;
                    }
                }
                KeyCode::Char(' ') => {
                    // Toggle model selection
                    if let Some(model) = self.available_models.get(self.model_selection_index) {
                        if let Some(pos) = self.selected_models.iter().position(|m| m == &model.name) {
                            self.selected_models.remove(pos);
                        } else {
                            self.selected_models.push(model.name.clone());
                        }
                    }
                }
                KeyCode::Char('a') => {
                    // Select all models
                    self.selected_models.clear();
                    for model in &self.available_models {
                        self.selected_models.push(model.name.clone());
                    }
                }
                KeyCode::Char('l') => {
                    // Select all local models
                    self.selected_models.clear();
                    for model in &self.available_models {
                        if model.is_local {
                            self.selected_models.push(model.name.clone());
                        }
                    }
                }
                KeyCode::Char('c') => {
                    // Clear selection
                    self.selected_models.clear();
                }
                KeyCode::Enter => {
                    // Confirm selection and update manager
                    tokio::task::block_in_place(|| {
                        tokio::runtime::Handle::current().block_on(self.update_manager_models())
                    });
                    self.model_selection_mode = false;
                }
                KeyCode::Esc => {
                    // Cancel without saving
                    self.model_selection_mode = false;
                }
                _ => {}
            }
        } else if self.edit_mode {
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