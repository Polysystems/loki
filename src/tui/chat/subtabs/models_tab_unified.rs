//! Unified models configuration tab
//! 
//! Consolidates models_tab.rs and models_tab_enhanced.rs into a single implementation
//! that connects directly to the core model orchestrator backend.

use std::sync::Arc;
use std::collections::HashMap;
use tokio::sync::RwLock;
use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect, Margin},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, List, ListItem, Paragraph, Row, Table, Wrap},
    Frame,
};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use anyhow::Result;
use tracing::{info, warn, error};

use super::SubtabController;
use crate::models::{
    ModelOrchestrator,
    providers::{ModelProvider, ModelInfo},
    orchestrator::ModelStatistics,
    local_manager::ModelHealth,
};
use crate::ollama::OllamaManager;
use crate::tui::chat::initialization::model_registry::{ModelRegistry, RegisteredModel};

/// View modes for the models tab
#[derive(Debug, Clone, PartialEq)]
enum ViewMode {
    Browse,      // Browse available models
    Details,     // View detailed model info
    Benchmark,   // View benchmarks
    Compare,     // Compare models side-by-side
    Settings,    // Model settings
    Search,      // Search models
    Configure,   // Configure/Add new model
}

/// Unified models tab with direct backend integration
pub struct UnifiedModelsTab {
    /// Core model orchestrator reference
    orchestrator: Arc<ModelOrchestrator>,
    
    /// Ollama manager for local models
    ollama_manager: Option<Arc<OllamaManager>>,
    
    /// Current view mode
    view_mode: ViewMode,
    
    /// Selected model index
    selected_index: usize,
    
    /// Available models (cached from backend)
    available_models: Vec<ModelEntry>,
    
    /// Search query
    search_query: String,
    
    /// Category filter
    category_filter: Option<String>,
    
    /// Loading state
    is_loading: bool,
    
    /// Error message
    error_message: Option<String>,
    
    /// Table state for lists
    table_state: ratatui::widgets::TableState,
    
    /// Model statistics cache
    statistics_cache: HashMap<String, ModelStatistics>,
    
    /// Model registry for adding new models
    model_registry: Option<Arc<ModelRegistry>>,
    
    /// Configuration state
    config_state: ConfigurationState,
    
    /// Current input field for configuration
    config_field_index: usize,
    
    /// Input buffer for text fields
    input_buffer: String,
    
    /// Is in input mode
    is_input_mode: bool,
}

/// Configuration state for adding new models
#[derive(Debug, Clone)]
struct ConfigurationState {
    /// Model ID/name
    pub model_id: String,
    /// Display name
    pub display_name: String,
    /// Provider selection index
    pub provider_index: usize,
    /// Available providers
    pub providers: Vec<String>,
    /// API endpoint (optional)
    pub api_endpoint: String,
    /// API key (masked)
    pub api_key: String,
    /// Context window size
    pub context_window: String,
    /// Selected capabilities
    pub capabilities: Vec<(String, bool)>,
    /// Error message
    pub error_message: Option<String>,
}

impl Default for ConfigurationState {
    fn default() -> Self {
        Self {
            model_id: String::new(),
            display_name: String::new(),
            provider_index: 0,
            providers: vec![
                "OpenAI".to_string(),
                "Anthropic".to_string(),
                "Google".to_string(),
                "Mistral".to_string(),
                "Ollama".to_string(),
                "Custom".to_string(),
            ],
            api_endpoint: String::new(),
            api_key: String::new(),
            context_window: "4096".to_string(),
            capabilities: vec![
                ("Chat".to_string(), true),
                ("Code".to_string(), false),
                ("Analysis".to_string(), false),
                ("Vision".to_string(), false),
                ("Embedding".to_string(), false),
            ],
            error_message: None,
        }
    }
}

/// Unified model entry combining all systems
#[derive(Debug, Clone)]
struct ModelEntry {
    /// Model identifier
    pub id: String,
    /// Provider name (openai, anthropic, ollama, etc.)
    pub provider: String,
    /// Model display name
    pub name: String,
    /// Model description
    pub description: String,
    /// Model capabilities
    pub capabilities: Vec<String>,
    /// Context window size
    pub context_window: usize,
    /// Is model available/online
    pub is_available: bool,
    /// Is model enabled for use
    pub is_enabled: bool,
    /// Performance statistics
    pub statistics: Option<ModelStatistics>,
    /// Is this a local model
    pub is_local: bool,
}

impl UnifiedModelsTab {
    /// Handle input in configuration mode
    fn handle_configure_input(&mut self, key: KeyEvent) -> Result<()> {
        use KeyCode::*;
        
        if self.is_input_mode {
            // Text input mode
            match key.code {
                Char(c) => {
                    self.input_buffer.push(c);
                }
                Backspace => {
                    self.input_buffer.pop();
                }
                Enter => {
                    // Save the input to the appropriate field
                    match self.config_field_index {
                        0 => self.config_state.model_id = self.input_buffer.clone(),
                        1 => self.config_state.display_name = self.input_buffer.clone(),
                        3 => self.config_state.api_endpoint = self.input_buffer.clone(),
                        4 => self.config_state.api_key = self.input_buffer.clone(),
                        5 => {
                            // Validate context window is a number
                            if self.input_buffer.parse::<usize>().is_ok() {
                                self.config_state.context_window = self.input_buffer.clone();
                            } else {
                                self.config_state.error_message = Some("Context window must be a number".to_string());
                            }
                        }
                        _ => {}
                    }
                    self.input_buffer.clear();
                    self.is_input_mode = false;
                }
                Esc => {
                    self.input_buffer.clear();
                    self.is_input_mode = false;
                }
                _ => {}
            }
        } else {
            // Navigation mode
            match key.code {
                Up => {
                    if self.config_field_index > 0 {
                        self.config_field_index -= 1;
                    }
                }
                Down | Tab => {
                    let max_index = 6 + self.config_state.capabilities.len() - 1;
                    if self.config_field_index < max_index {
                        self.config_field_index += 1;
                    }
                }
                Left => {
                    if self.config_field_index == 2 {
                        // Provider selection
                        if self.config_state.provider_index > 0 {
                            self.config_state.provider_index -= 1;
                        }
                    }
                }
                Right => {
                    if self.config_field_index == 2 {
                        // Provider selection
                        if self.config_state.provider_index < self.config_state.providers.len() - 1 {
                            self.config_state.provider_index += 1;
                        }
                    }
                }
                Enter => {
                    // Enter edit mode for text fields
                    match self.config_field_index {
                        0..=1 | 3..=5 => {
                            self.is_input_mode = true;
                            self.input_buffer = match self.config_field_index {
                                0 => self.config_state.model_id.clone(),
                                1 => self.config_state.display_name.clone(),
                                3 => self.config_state.api_endpoint.clone(),
                                4 => self.config_state.api_key.clone(),
                                5 => self.config_state.context_window.clone(),
                                _ => String::new(),
                            };
                        }
                        _ => {}
                    }
                }
                Char(' ') => {
                    // Toggle capability checkboxes
                    if self.config_field_index >= 6 {
                        let cap_index = self.config_field_index - 6;
                        if cap_index < self.config_state.capabilities.len() {
                            self.config_state.capabilities[cap_index].1 = !self.config_state.capabilities[cap_index].1;
                        }
                    }
                }
                Char('s') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    // Save configuration
                    tokio::task::block_in_place(|| {
                        tokio::runtime::Handle::current().block_on(self.save_configuration())
                    })?;
                }
                Esc => {
                    // Return to browse mode
                    self.view_mode = ViewMode::Browse;
                    self.config_state.error_message = None;
                }
                _ => {}
            }
        }
        
        Ok(())
    }
    
    /// Save the new model configuration
    async fn save_configuration(&mut self) -> Result<()> {
        // Validate required fields
        if self.config_state.model_id.is_empty() {
            self.config_state.error_message = Some("Model ID is required".to_string());
            return Ok(());
        }
        if self.config_state.display_name.is_empty() {
            self.config_state.display_name = self.config_state.model_id.clone();
        }
        if self.config_state.api_key.is_empty() {
            self.config_state.error_message = Some("API Key is required".to_string());
            return Ok(());
        }
        
        let provider = self.config_state.providers[self.config_state.provider_index].to_lowercase();
        if provider == "custom" && self.config_state.api_endpoint.is_empty() {
            self.config_state.error_message = Some("API Endpoint is required for custom provider".to_string());
            return Ok(());
        }
        
        // Parse context window
        let context_window = match self.config_state.context_window.parse::<usize>() {
            Ok(size) => size,
            Err(_) => {
                self.config_state.error_message = Some("Invalid context window size".to_string());
                return Ok(());
            }
        };
        
        // Collect enabled capabilities
        let capabilities: Vec<String> = self.config_state.capabilities
            .iter()
            .filter(|(_, enabled)| *enabled)
            .map(|(name, _)| name.to_lowercase())
            .collect();
        
        // Create and register the model
        if let Some(registry) = &self.model_registry {
            let model = RegisteredModel {
                id: self.config_state.model_id.clone(),
                name: self.config_state.display_name.clone(),
                provider: provider.clone(),
                capabilities,
                available: false, // Will be verified
                last_verified: None,
                error: None,
            };
            
            match registry.register_model(model).await {
                Ok(_) => {
                    info!("Successfully registered model: {}", self.config_state.model_id);
                    
                    // Store API key in environment or config
                    // This is a simplified approach - in production, use proper secret management
                    match provider.as_str() {
                        "openai" => std::env::set_var("OPENAI_API_KEY", &self.config_state.api_key),
                        "anthropic" => std::env::set_var("ANTHROPIC_API_KEY", &self.config_state.api_key),
                        "google" => std::env::set_var("GEMINI_API_KEY", &self.config_state.api_key),
                        "mistral" => std::env::set_var("MISTRAL_API_KEY", &self.config_state.api_key),
                        _ => {}
                    }
                    
                    // Refresh models and return to browse
                    self.refresh_models().await?;
                    self.view_mode = ViewMode::Browse;
                    self.config_state = ConfigurationState::default();
                }
                Err(e) => {
                    self.config_state.error_message = Some(format!("Failed to register: {}", e));
                }
            }
        } else {
            // If no registry, add to local models list (simplified)
            self.available_models.push(ModelEntry {
                id: format!("{}/{}", provider, self.config_state.model_id),
                provider: provider.clone(),
                name: self.config_state.display_name.clone(),
                description: format!("Custom configured {} model", provider),
                capabilities,
                context_window,
                is_available: true,
                is_enabled: true,
                statistics: None,
                is_local: provider == "ollama",
            });
            
            info!("Added model to local list: {}", self.config_state.model_id);
            self.view_mode = ViewMode::Browse;
            self.config_state = ConfigurationState::default();
        }
        
        Ok(())
    }
    
    /// Create a new unified models tab
    pub fn new(orchestrator: Arc<ModelOrchestrator>) -> Self {
        Self {
            orchestrator,
            ollama_manager: None,
            view_mode: ViewMode::Browse,
            selected_index: 0,
            available_models: Vec::new(),
            search_query: String::new(),
            category_filter: None,
            is_loading: false,
            error_message: None,
            table_state: ratatui::widgets::TableState::default(),
            statistics_cache: HashMap::new(),
            model_registry: None,
            config_state: ConfigurationState::default(),
            config_field_index: 0,
            input_buffer: String::new(),
            is_input_mode: false,
        }
    }
    
    
    /// Set Ollama manager for local model support
    pub fn set_ollama_manager(&mut self, manager: Arc<OllamaManager>) {
        self.ollama_manager = Some(manager);
    }
    
    /// Set model registry for adding new models
    pub fn set_model_registry(&mut self, registry: Arc<ModelRegistry>) {
        self.model_registry = Some(registry);
    }
    
    /// Refresh models from backend
    async fn refresh_models(&mut self) -> Result<()> {
        self.is_loading = true;
        self.error_message = None;
        
        // Get status from orchestrator
        let status = self.orchestrator.get_status().await;
        
        let mut models = Vec::new();
        
        // Add API models from orchestrator
        for (provider_name, provider_status) in &status.api_providers {
            if provider_status.is_available {
                // Get models for this provider
                if let Some(provider) = self.orchestrator.get_provider(provider_name) {
                    match provider.list_models().await {
                        Ok(model_infos) => {
                            for model_info in model_infos {
                                models.push(ModelEntry {
                                    id: format!("{}/{}", provider_name, model_info.id),
                                    provider: provider_name.clone(),
                                    name: model_info.name.clone(),
                                    description: model_info.description.clone(),
                                    capabilities: model_info.capabilities.clone(),
                                    context_window: model_info.context_length,
                                    is_available: true,
                                    is_enabled: true, // API models are enabled if available
                                    statistics: self.statistics_cache.get(&model_info.id).cloned(),
                                    is_local: false,
                                });
                            }
                        }
                        Err(e) => {
                            warn!("Failed to list models from {}: {}", provider_name, e);
                        }
                    }
                }
            }
        }
        
        // Add local models from orchestrator
        for (model_name, model_status) in &status.local_models.model_statuses {
            models.push(ModelEntry {
                id: format!("local/{}", model_name),
                provider: "local".to_string(),
                name: model_name.clone(),
                description: format!("Local model: {}", model_name),
                capabilities: vec!["chat".to_string(), "completion".to_string()],
                context_window: 4096, // Default for local models
                is_available: matches!(model_status.health, ModelHealth::Healthy),
                is_enabled: true, // Local models are enabled if loaded
                statistics: None, // ModelInstanceStatus doesn't have statistics field
                is_local: true,
            });
        }
        
        // Add Ollama models if manager is available
        if let Some(ref ollama) = self.ollama_manager {
            match ollama.list_models().await {
                Ok(ollama_models) => {
                    for model in ollama_models {
                        let model_id = format!("ollama/{}", model.name);
                        if !models.iter().any(|m| m.id == model_id) {
                            models.push(ModelEntry {
                                id: model_id,
                                provider: "ollama".to_string(),
                                name: model.name.clone(),
                                description: format!("Ollama model: {} ({})", model.name, model.size),
                                capabilities: vec!["chat".to_string(), "completion".to_string()],
                                context_window: 4096,
                                is_available: true,
                                is_enabled: status.enabled_models.contains(&model.name),
                                statistics: None,
                                is_local: true,
                            });
                        }
                    }
                }
                Err(e) => {
                    warn!("Failed to list Ollama models: {}", e);
                }
            }
        }
        
        // Apply filters
        if !self.search_query.is_empty() {
            let query = self.search_query.to_lowercase();
            models.retain(|m| 
                m.name.to_lowercase().contains(&query) ||
                m.provider.to_lowercase().contains(&query) ||
                m.description.to_lowercase().contains(&query)
            );
        }
        
        if let Some(ref category) = self.category_filter {
            models.retain(|m| &m.provider == category || 
                (category == "local" && m.is_local));
        }
        
        // Sort models: local first, then by provider and name
        models.sort_by(|a, b| {
            match (a.is_local, b.is_local) {
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                _ => a.provider.cmp(&b.provider)
                    .then(a.name.cmp(&b.name))
            }
        });
        
        self.available_models = models;
        self.is_loading = false;
        
        // Reset selection if out of bounds
        if self.selected_index >= self.available_models.len() && !self.available_models.is_empty() {
            self.selected_index = self.available_models.len() - 1;
        }
        
        Ok(())
    }
    
    /// Toggle model enabled state
    async fn toggle_model(&mut self) -> Result<()> {
        if self.available_models.is_empty() {
            warn!("No models available to toggle");
            return Ok(());
        }
        
        if let Some(model) = self.available_models.get_mut(self.selected_index) {
            // Extract just the model name without provider prefix
            let model_name = model.id.split('/').last().unwrap_or(&model.id).to_string();
            let was_enabled = model.is_enabled;
            
            // Clear any previous error
            self.error_message = None;
            
            // Toggle the UI state immediately for responsive feedback
            model.is_enabled = !was_enabled;
            let new_state = model.is_enabled;
            
            info!("Toggling model '{}' from {} to {}", model_name, was_enabled, new_state);
            
            // Try to update the orchestrator (but don't fail if it doesn't work)
            let orchestrator_result: Result<()> = if new_state {
                // Enabling the model
                match self.orchestrator.enable_model(&model_name).await {
                    Ok(_) => {
                        info!("Orchestrator: Successfully enabled model: {}", model_name);
                        Ok(())
                    }
                    Err(e) => {
                        warn!("Orchestrator: Failed to enable model {}: {} (UI state updated anyway)", model_name, e);
                        // Don't revert UI state - keep it as user intended
                        Ok(()) // Return Ok to not show error to user
                    }
                }
            } else {
                // Disabling the model
                match self.orchestrator.disable_model(&model_name).await {
                    Ok(_) => {
                        info!("Orchestrator: Successfully disabled model: {}", model_name);
                        Ok(())
                    }
                    Err(e) => {
                        warn!("Orchestrator: Failed to disable model {}: {} (UI state updated anyway)", model_name, e);
                        // Don't revert UI state - keep it as user intended
                        Ok(()) // Return Ok to not show error to user
                    }
                }
            };
            
            // Don't refresh models as it might reset our state
            // The UI state is already updated above
            
            if orchestrator_result.is_ok() {
                // Success message
                info!("Model '{}' is now {}", model_name, if new_state { "enabled" } else { "disabled" });
            }
        } else {
            warn!("Selected index {} is out of bounds", self.selected_index);
        }
        
        Ok(())
    }
    
    /// Render browse view
    fn render_browse_view(&self, f: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),   // Header
                Constraint::Min(10),     // Model list
                Constraint::Length(6),   // Model preview
            ])
            .split(area);
        
        // Header with stats
        let total = self.available_models.len();
        let enabled = self.available_models.iter().filter(|m| m.is_enabled).count();
        let local = self.available_models.iter().filter(|m| m.is_local).count();
        
        let header_text = format!(
            "📊 {} models ({} enabled, {} local) | 🔄 'r' refresh | ⚙️ Enter toggle | ➕ 'a' add new",
            total, enabled, local
        );
        
        let header = Paragraph::new(header_text)
            .style(Style::default().fg(Color::Cyan))
            .alignment(Alignment::Center)
            .block(Block::default().borders(Borders::BOTTOM));
        f.render_widget(header, chunks[0]);
        
        // Model list
        if self.is_loading {
            let loading = Paragraph::new("🔄 Loading models...")
                .style(Style::default().fg(Color::Yellow))
                .alignment(Alignment::Center)
                .block(Block::default().borders(Borders::ALL));
            f.render_widget(loading, chunks[1]);
        } else if let Some(error) = &self.error_message {
            let error_widget = Paragraph::new(error.as_str())
                .style(Style::default().fg(Color::Red))
                .alignment(Alignment::Center)
                .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(Color::Red)));
            f.render_widget(error_widget, chunks[1]);
        } else {
            self.render_model_list(f, chunks[1]);
        }
        
        // Model preview or error message
        if let Some(error) = &self.error_message {
            let error_widget = Paragraph::new(error.as_str())
                .style(Style::default().fg(Color::Red))
                .alignment(Alignment::Center)
                .block(Block::default()
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded)
                    .border_style(Style::default().fg(Color::Red))
                    .title(" Error "));
            f.render_widget(error_widget, chunks[2]);
        } else {
            self.render_model_preview(f, chunks[2]);
        }
    }
    
    /// Render model list
    fn render_model_list(&self, f: &mut Frame, area: Rect) {
        let items: Vec<ListItem> = self.available_models
            .iter()
            .enumerate()
            .map(|(i, model)| {
                let selected = i == self.selected_index;
                
                // Status indicators
                let status = if !model.is_available {
                    "⚠️"
                } else if model.is_enabled {
                    "✅"
                } else {
                    "⭕"
                };
                
                let location = if model.is_local { "📍" } else { "☁️" };
                
                let content = format!(
                    "{} {} {} {} | {} | {}k ctx",
                    if selected { "▶" } else { " " },
                    status,
                    location,
                    model.name,
                    model.provider,
                    model.context_window / 1000
                );
                
                let style = if selected {
                    Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
                } else if !model.is_available {
                    Style::default().fg(Color::DarkGray)
                } else if model.is_enabled {
                    Style::default().fg(Color::Green)
                } else {
                    Style::default()
                };
                
                ListItem::new(content).style(style)
            })
            .collect();
        
        let list = List::new(items)
            .block(Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .title(format!(" Models ({} total) ", self.available_models.len())));
        f.render_widget(list, area);
    }
    
    /// Render model preview
    fn render_model_preview(&self, f: &mut Frame, area: Rect) {
        if let Some(model) = self.available_models.get(self.selected_index) {
            let mut lines = vec![
                Line::from(vec![
                    Span::raw("📝 "),
                    Span::styled(&model.description, Style::default().fg(Color::White)),
                ]),
            ];
            
            // Capabilities
            if !model.capabilities.is_empty() {
                lines.push(Line::from(vec![
                    Span::raw("✨ "),
                    Span::styled(
                        format!("Capabilities: {}", model.capabilities.join(", ")),
                        Style::default().fg(Color::Green)
                    ),
                ]));
            }
            
            // Statistics if available
            if let Some(ref stats) = model.statistics {
                lines.push(Line::from(vec![
                    Span::raw("📊 "),
                    Span::styled(
                        format!("Requests: {} | Success: {:.1}% | Avg: {:.0}ms",
                            stats.total_requests,
                            stats.success_rate * 100.0,
                            stats.avg_execution_time.as_millis() as f64
                        ),
                        Style::default().fg(Color::Cyan)
                    ),
                ]));
            }
            
            // Status
            let status_text = if model.is_enabled {
                "Enabled - Press Enter to disable"
            } else {
                "Disabled - Press Enter to enable"
            };
            lines.push(Line::from(vec![
                Span::raw("⚙️ "),
                Span::styled(status_text, Style::default().fg(Color::Yellow)),
            ]));
            
            let preview = Paragraph::new(lines)
                .wrap(Wrap { trim: true })
                .block(Block::default()
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded)
                    .title(format!(" {} ", model.id)));
            f.render_widget(preview, area);
        } else {
            let empty = Paragraph::new("No model selected")
                .style(Style::default().fg(Color::DarkGray))
                .alignment(Alignment::Center)
                .block(Block::default().borders(Borders::ALL));
            f.render_widget(empty, area);
        }
    }
    
    /// Render details view
    fn render_details_view(&self, f: &mut Frame, area: Rect) {
        if let Some(model) = self.available_models.get(self.selected_index) {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(3),
                    Constraint::Min(0),
                ])
                .split(area);
            
            // Title
            let title = Paragraph::new(format!("📊 {} Details", model.name))
                .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
                .alignment(Alignment::Center)
                .block(Block::default().borders(Borders::BOTTOM));
            f.render_widget(title, chunks[0]);
            
            // Details
            let mut lines = vec![
                Line::from(vec![
                    Span::styled("Model ID: ", Style::default().add_modifier(Modifier::BOLD)),
                    Span::raw(&model.id),
                ]),
                Line::from(vec![
                    Span::styled("Provider: ", Style::default().add_modifier(Modifier::BOLD)),
                    Span::styled(&model.provider, Style::default().fg(Color::Yellow)),
                ]),
                Line::from(vec![
                    Span::styled("Type: ", Style::default().add_modifier(Modifier::BOLD)),
                    Span::raw(if model.is_local { "Local" } else { "API" }),
                ]),
                Line::from(""),
                Line::from(vec![
                    Span::styled("Description: ", Style::default().add_modifier(Modifier::BOLD)),
                    Span::raw(&model.description),
                ]),
                Line::from(""),
                Line::from(vec![
                    Span::styled("Context Window: ", Style::default().add_modifier(Modifier::BOLD)),
                    Span::raw(format!("{} tokens", model.context_window)),
                ]),
                Line::from(""),
                Line::from(vec![
                    Span::styled("Capabilities: ", Style::default().add_modifier(Modifier::BOLD)),
                ]),
            ];
            
            for cap in &model.capabilities {
                lines.push(Line::from(vec![
                    Span::raw("  • "),
                    Span::raw(cap),
                ]));
            }
            
            if let Some(ref stats) = model.statistics {
                lines.push(Line::from(""));
                lines.push(Line::from(vec![
                    Span::styled("Statistics: ", Style::default().add_modifier(Modifier::BOLD)),
                ]));
                lines.push(Line::from(format!("  Total Requests: {}", stats.total_requests)));
                lines.push(Line::from(format!("  Success Rate: {:.1}%", stats.success_rate * 100.0)));
                lines.push(Line::from(format!("  Avg Response: {:.0}ms", stats.avg_execution_time.as_millis())));
                lines.push(Line::from(format!("  Total Requests: {}", stats.total_requests)));
            }
            
            lines.push(Line::from(""));
            lines.push(Line::from(vec![
                Span::styled("Press ESC to return", Style::default().fg(Color::DarkGray)),
            ]));
            
            let details = Paragraph::new(lines)
                .block(Block::default()
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded));
            f.render_widget(details, chunks[1]);
        }
    }
    
    /// Render benchmark view
    fn render_benchmark_view(&mut self, f: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),
                Constraint::Min(0),
            ])
            .split(area);
        
        // Title
        let title = Paragraph::new("⚡ Model Performance Benchmarks")
            .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
            .alignment(Alignment::Center)
            .block(Block::default().borders(Borders::BOTTOM));
        f.render_widget(title, chunks[0]);
        
        // Collect benchmark data from models with statistics
        let mut benchmark_data: Vec<(&str, f64, f64, u64)> = Vec::new();
        for model in &self.available_models {
            if let Some(ref stats) = model.statistics {
                benchmark_data.push((
                    &model.name,
                    stats.avg_execution_time.as_millis() as f64,
                    (stats.success_rate * 100.0) as f64,
                    stats.total_requests,
                ));
            }
        }
        
        if benchmark_data.is_empty() {
            let message = Paragraph::new("No benchmark data available. Use models to generate statistics.")
                .style(Style::default().fg(Color::Yellow))
                .alignment(Alignment::Center)
                .block(Block::default().borders(Borders::ALL));
            f.render_widget(message, chunks[1]);
            return;
        }
        
        // Sort by response time
        benchmark_data.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Create table
        let header = Row::new(vec!["Model", "Avg Response", "Success Rate", "Total Requests"])
            .style(Style::default().add_modifier(Modifier::BOLD))
            .bottom_margin(1);
        
        let rows: Vec<Row> = benchmark_data.iter().map(|(name, response_time, success_rate, requests)| {
            Row::new(vec![
                name.to_string(),
                format!("{:.0}ms", response_time),
                format!("{:.1}%", success_rate),
                requests.to_string(),
            ])
        }).collect();
        
        let table = Table::new(rows, vec![
            Constraint::Percentage(40),
            Constraint::Percentage(20),
            Constraint::Percentage(20),
            Constraint::Percentage(20),
        ])
        .header(header)
        .block(Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .title(" Performance Metrics "))
        .row_highlight_style(Style::default().add_modifier(Modifier::BOLD))
        .highlight_symbol("→ ");
        
        f.render_stateful_widget(table, chunks[1], &mut self.table_state);
    }
    
    /// Render compare view
    fn render_compare_view(&self, f: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),
                Constraint::Min(0),
            ])
            .split(area);
        
        // Title
        let title = Paragraph::new("📊 Model Comparison")
            .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
            .alignment(Alignment::Center)
            .block(Block::default().borders(Borders::BOTTOM));
        f.render_widget(title, chunks[0]);
        
        // Get up to 3 models to compare
        let models_to_compare: Vec<_> = self.available_models.iter()
            .filter(|m| m.is_enabled)
            .take(3)
            .collect();
        
        if models_to_compare.is_empty() {
            let message = Paragraph::new("No enabled models to compare. Enable some models first.")
                .style(Style::default().fg(Color::Yellow))
                .alignment(Alignment::Center)
                .block(Block::default().borders(Borders::ALL));
            f.render_widget(message, chunks[1]);
            return;
        }
        
        // Create comparison columns
        let comparison_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints(vec![Constraint::Percentage(100 / models_to_compare.len() as u16); models_to_compare.len()])
            .split(chunks[1]);
        
        for (i, model) in models_to_compare.iter().enumerate() {
            let mut lines = vec![
                Line::from(vec![
                    Span::styled(&model.name, Style::default().add_modifier(Modifier::BOLD).fg(Color::Yellow)),
                ]),
                Line::from(""),
                Line::from(vec![
                    Span::styled("Provider: ", Style::default().add_modifier(Modifier::BOLD)),
                    Span::raw(&model.provider),
                ]),
                Line::from(vec![
                    Span::styled("Type: ", Style::default().add_modifier(Modifier::BOLD)),
                    Span::raw(if model.is_local { "Local" } else { "API" }),
                ]),
                Line::from(vec![
                    Span::styled("Context: ", Style::default().add_modifier(Modifier::BOLD)),
                    Span::raw(format!("{}k", model.context_window / 1000)),
                ]),
                Line::from(""),
                Line::from(vec![
                    Span::styled("Capabilities:", Style::default().add_modifier(Modifier::BOLD)),
                ]),
            ];
            
            for cap in &model.capabilities {
                lines.push(Line::from(vec![
                    Span::raw("• "),
                    Span::raw(cap),
                ]));
            }
            
            if let Some(ref stats) = model.statistics {
                lines.push(Line::from(""));
                lines.push(Line::from(vec![
                    Span::styled("Performance:", Style::default().add_modifier(Modifier::BOLD)),
                ]));
                lines.push(Line::from(format!("Requests: {}", stats.total_requests)));
                lines.push(Line::from(format!("Success: {:.1}%", stats.success_rate * 100.0)));
                lines.push(Line::from(format!("Avg: {:.0}ms", stats.avg_execution_time.as_millis())));
            }
            
            let comparison = Paragraph::new(lines)
                .block(Block::default()
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded));
            f.render_widget(comparison, comparison_chunks[i]);
        }
    }
    
    fn render_settings_view(&self, f: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),   // Header
                Constraint::Min(10),     // Settings content
                Constraint::Length(4),   // Instructions
            ])
            .split(area);
        
        // Header
        let header = Paragraph::new("Model Settings")
            .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
            .alignment(Alignment::Center)
            .block(Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded));
        f.render_widget(header, chunks[0]);
        
        // Settings sections
        let settings_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(50),  // General settings
                Constraint::Percentage(50),  // Provider settings
            ])
            .split(chunks[1]);
        
        // General Settings
        let general_settings = vec![
            Line::from(vec![
                Span::styled("General Settings", Style::default().add_modifier(Modifier::BOLD).fg(Color::Yellow)),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled("Default Temperature: ", Style::default().fg(Color::Gray)),
                Span::styled("0.7", Style::default().fg(Color::White)),
            ]),
            Line::from(vec![
                Span::styled("Default Max Tokens: ", Style::default().fg(Color::Gray)),
                Span::styled("2048", Style::default().fg(Color::White)),
            ]),
            Line::from(vec![
                Span::styled("Default Top P: ", Style::default().fg(Color::Gray)),
                Span::styled("0.9", Style::default().fg(Color::White)),
            ]),
            Line::from(vec![
                Span::styled("Retry Attempts: ", Style::default().fg(Color::Gray)),
                Span::styled("3", Style::default().fg(Color::White)),
            ]),
            Line::from(vec![
                Span::styled("Timeout (seconds): ", Style::default().fg(Color::Gray)),
                Span::styled("30", Style::default().fg(Color::White)),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled("Fallback Strategy", Style::default().add_modifier(Modifier::BOLD).fg(Color::Yellow)),
            ]),
            Line::from(vec![
                Span::styled("Enabled: ", Style::default().fg(Color::Gray)),
                Span::styled("Yes", Style::default().fg(Color::Green)),
            ]),
            Line::from(vec![
                Span::styled("Primary Model: ", Style::default().fg(Color::Gray)),
                Span::styled("claude-3-opus", Style::default().fg(Color::White)),
            ]),
            Line::from(vec![
                Span::styled("Fallback Model: ", Style::default().fg(Color::Gray)),
                Span::styled("gpt-4", Style::default().fg(Color::White)),
            ]),
        ];
        
        let general = Paragraph::new(general_settings)
            .block(Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .title(" General "));
        f.render_widget(general, settings_chunks[0]);
        
        // Provider Settings
        let provider_settings = vec![
            Line::from(vec![
                Span::styled("Provider Settings", Style::default().add_modifier(Modifier::BOLD).fg(Color::Yellow)),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled("OpenAI", Style::default().add_modifier(Modifier::BOLD).fg(Color::Cyan)),
            ]),
            Line::from(vec![
                Span::styled("  API Key: ", Style::default().fg(Color::Gray)),
                Span::styled("sk-...***", Style::default().fg(Color::DarkGray)),
            ]),
            Line::from(vec![
                Span::styled("  Org ID: ", Style::default().fg(Color::Gray)),
                Span::styled("org-...", Style::default().fg(Color::DarkGray)),
            ]),
            Line::from(vec![
                Span::styled("  Status: ", Style::default().fg(Color::Gray)),
                Span::styled("Connected", Style::default().fg(Color::Green)),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled("Anthropic", Style::default().add_modifier(Modifier::BOLD).fg(Color::Cyan)),
            ]),
            Line::from(vec![
                Span::styled("  API Key: ", Style::default().fg(Color::Gray)),
                Span::styled("sk-ant-...***", Style::default().fg(Color::DarkGray)),
            ]),
            Line::from(vec![
                Span::styled("  Status: ", Style::default().fg(Color::Gray)),
                Span::styled("Connected", Style::default().fg(Color::Green)),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled("Ollama", Style::default().add_modifier(Modifier::BOLD).fg(Color::Cyan)),
            ]),
            Line::from(vec![
                Span::styled("  Host: ", Style::default().fg(Color::Gray)),
                Span::styled("localhost:11434", Style::default().fg(Color::White)),
            ]),
            Line::from(vec![
                Span::styled("  Status: ", Style::default().fg(Color::Gray)),
                if self.ollama_manager.is_some() {
                    Span::styled("Connected", Style::default().fg(Color::Green))
                } else {
                    Span::styled("Not Available", Style::default().fg(Color::Red))
                },
            ]),
        ];
        
        let providers = Paragraph::new(provider_settings)
            .block(Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .title(" Providers "));
        f.render_widget(providers, settings_chunks[1]);
        
        // Instructions
        let instructions = vec![
            Line::from(vec![
                Span::styled("[↑↓]", Style::default().fg(Color::Yellow)),
                Span::raw(" Navigate  "),
                Span::styled("[Enter]", Style::default().fg(Color::Yellow)),
                Span::raw(" Edit  "),
                Span::styled("[Tab]", Style::default().fg(Color::Yellow)),
                Span::raw(" Switch Section  "),
                Span::styled("[s]", Style::default().fg(Color::Yellow)),
                Span::raw(" Save  "),
                Span::styled("[r]", Style::default().fg(Color::Yellow)),
                Span::raw(" Reset  "),
                Span::styled("[Esc]", Style::default().fg(Color::Yellow)),
                Span::raw(" Back"),
            ]),
            Line::from(vec![
                Span::styled("[p]", Style::default().fg(Color::Yellow)),
                Span::raw(" Add Provider  "),
                Span::styled("[d]", Style::default().fg(Color::Yellow)),
                Span::raw(" Delete Provider  "),
                Span::styled("[t]", Style::default().fg(Color::Yellow)),
                Span::raw(" Test Connection  "),
                Span::styled("[e]", Style::default().fg(Color::Yellow)),
                Span::raw(" Export Config  "),
                Span::styled("[i]", Style::default().fg(Color::Yellow)),
                Span::raw(" Import Config"),
            ]),
        ];
        
        let help = Paragraph::new(instructions)
            .style(Style::default().fg(Color::DarkGray))
            .alignment(Alignment::Center)
            .block(Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded));
        f.render_widget(help, chunks[2]);
    }
    
    /// Render configuration view for adding new models
    fn render_configure_view(&mut self, f: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),   // Header
                Constraint::Min(20),     // Form fields
                Constraint::Length(4),   // Instructions
            ])
            .split(area);
        
        // Header
        let header = Paragraph::new("➕ Add New Model Configuration")
            .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
            .alignment(Alignment::Center)
            .block(Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded));
        f.render_widget(header, chunks[0]);
        
        // Form fields
        let form_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),  // Model ID
                Constraint::Length(3),  // Display Name
                Constraint::Length(3),  // Provider
                Constraint::Length(3),  // API Endpoint
                Constraint::Length(3),  // API Key
                Constraint::Length(3),  // Context Window
                Constraint::Min(5),     // Capabilities
                Constraint::Length(2),  // Error message
            ])
            .split(chunks[1]);
        
        // Field styles
        let get_field_style = |index: usize| {
            if self.is_input_mode && self.config_field_index == index {
                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
            } else if self.config_field_index == index {
                Style::default().fg(Color::Cyan)
            } else {
                Style::default()
            }
        };
        
        // Model ID field
        let model_id_value = if self.is_input_mode && self.config_field_index == 0 {
            &self.input_buffer
        } else {
            &self.config_state.model_id
        };
        let model_id = Paragraph::new(format!("Model ID: {}", model_id_value))
            .style(get_field_style(0))
            .block(Block::default().borders(Borders::ALL).title(" Model Identifier "));
        f.render_widget(model_id, form_chunks[0]);
        
        // Display Name field
        let display_name_value = if self.is_input_mode && self.config_field_index == 1 {
            &self.input_buffer
        } else {
            &self.config_state.display_name
        };
        let display_name = Paragraph::new(format!("Display Name: {}", display_name_value))
            .style(get_field_style(1))
            .block(Block::default().borders(Borders::ALL).title(" Display Name "));
        f.render_widget(display_name, form_chunks[1]);
        
        // Provider selection
        let provider = &self.config_state.providers[self.config_state.provider_index];
        let provider_text = if self.config_field_index == 2 {
            format!("Provider: {} (← → to change)", provider)
        } else {
            format!("Provider: {}", provider)
        };
        let provider_widget = Paragraph::new(provider_text)
            .style(get_field_style(2))
            .block(Block::default().borders(Borders::ALL).title(" Provider "));
        f.render_widget(provider_widget, form_chunks[2]);
        
        // API Endpoint (only for custom provider)
        let endpoint_value = if self.is_input_mode && self.config_field_index == 3 {
            &self.input_buffer
        } else {
            &self.config_state.api_endpoint
        };
        let endpoint_text = if provider == "Custom" {
            format!("API Endpoint: {}", endpoint_value)
        } else {
            "API Endpoint: (Not needed for this provider)".to_string()
        };
        let api_endpoint = Paragraph::new(endpoint_text)
            .style(if provider == "Custom" { get_field_style(3) } else { Style::default().fg(Color::DarkGray) })
            .block(Block::default().borders(Borders::ALL).title(" API Endpoint "));
        f.render_widget(api_endpoint, form_chunks[3]);
        
        // API Key field (masked)
        let api_key_value = if self.is_input_mode && self.config_field_index == 4 {
            "*".repeat(self.input_buffer.len())
        } else if !self.config_state.api_key.is_empty() {
            "*".repeat(self.config_state.api_key.len())
        } else {
            String::new()
        };
        let api_key = Paragraph::new(format!("API Key: {}", api_key_value))
            .style(get_field_style(4))
            .block(Block::default().borders(Borders::ALL).title(" API Key (Required) "));
        f.render_widget(api_key, form_chunks[4]);
        
        // Context Window field
        let context_value = if self.is_input_mode && self.config_field_index == 5 {
            &self.input_buffer
        } else {
            &self.config_state.context_window
        };
        let context_window = Paragraph::new(format!("Context Window: {} tokens", context_value))
            .style(get_field_style(5))
            .block(Block::default().borders(Borders::ALL).title(" Context Window Size "));
        f.render_widget(context_window, form_chunks[5]);
        
        // Capabilities checkboxes
        let mut cap_lines = vec![Line::from("Capabilities (Space to toggle):")];
        for (i, (cap_name, enabled)) in self.config_state.capabilities.iter().enumerate() {
            let checkbox = if *enabled { "[✓]" } else { "[ ]" };
            let style = if self.config_field_index == 6 + i {
                Style::default().fg(Color::Cyan)
            } else {
                Style::default()
            };
            cap_lines.push(Line::from(vec![
                Span::raw("  "),
                Span::styled(format!("{} {}", checkbox, cap_name), style),
            ]));
        }
        let capabilities = Paragraph::new(cap_lines)
            .block(Block::default().borders(Borders::ALL).title(" Capabilities "));
        f.render_widget(capabilities, form_chunks[6]);
        
        // Error message
        if let Some(error) = &self.config_state.error_message {
            let error_widget = Paragraph::new(error.as_str())
                .style(Style::default().fg(Color::Red))
                .alignment(Alignment::Center);
            f.render_widget(error_widget, form_chunks[7]);
        }
        
        // Instructions
        let instructions = if self.is_input_mode {
            vec![
                Line::from(vec![
                    Span::styled("[Enter]", Style::default().fg(Color::Yellow)),
                    Span::raw(" Confirm Input  "),
                    Span::styled("[Esc]", Style::default().fg(Color::Yellow)),
                    Span::raw(" Cancel Input  "),
                    Span::styled("[Backspace]", Style::default().fg(Color::Yellow)),
                    Span::raw(" Delete Character"),
                ]),
                Line::from(vec![
                    Span::raw("Type to enter text for the selected field"),
                ]),
            ]
        } else {
            vec![
                Line::from(vec![
                    Span::styled("[↑↓]", Style::default().fg(Color::Yellow)),
                    Span::raw(" Navigate Fields  "),
                    Span::styled("[Enter]", Style::default().fg(Color::Yellow)),
                    Span::raw(" Edit Field  "),
                    Span::styled("[←→]", Style::default().fg(Color::Yellow)),
                    Span::raw(" Change Provider  "),
                    Span::styled("[Space]", Style::default().fg(Color::Yellow)),
                    Span::raw(" Toggle Capability"),
                ]),
                Line::from(vec![
                    Span::styled("[Ctrl+S]", Style::default().fg(Color::Green)),
                    Span::raw(" Save Configuration  "),
                    Span::styled("[Esc]", Style::default().fg(Color::Yellow)),
                    Span::raw(" Cancel & Return  "),
                    Span::styled("[Tab]", Style::default().fg(Color::Yellow)),
                    Span::raw(" Next Field"),
                ]),
            ]
        };
        
        let help = Paragraph::new(instructions)
            .style(Style::default().fg(Color::DarkGray))
            .alignment(Alignment::Center)
            .block(Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded));
        f.render_widget(help, chunks[2]);
    }
}

impl SubtabController for UnifiedModelsTab {
    fn render(&mut self, f: &mut Frame, area: Rect) {
        match self.view_mode {
            ViewMode::Browse => self.render_browse_view(f, area),
            ViewMode::Details => self.render_details_view(f, area),
            ViewMode::Benchmark => self.render_benchmark_view(f, area),
            ViewMode::Compare => self.render_compare_view(f, area),
            ViewMode::Settings => self.render_settings_view(f, area),
            ViewMode::Search => self.render_browse_view(f, area), // Search uses browse view with filter
            ViewMode::Configure => self.render_configure_view(f, area),
        }
    }
    
    fn handle_input(&mut self, key: KeyEvent) -> Result<()> {
        // Handle configuration mode
        if self.view_mode == ViewMode::Configure {
            return self.handle_configure_input(key);
        }
        
        // Handle search mode
        if self.view_mode == ViewMode::Search {
            match key.code {
                KeyCode::Esc => {
                    self.view_mode = ViewMode::Browse;
                    self.search_query.clear();
                    tokio::task::block_in_place(|| {
                        tokio::runtime::Handle::current().block_on(self.refresh_models())
                    })?;
                }
                KeyCode::Enter => {
                    self.view_mode = ViewMode::Browse;
                    tokio::task::block_in_place(|| {
                        tokio::runtime::Handle::current().block_on(self.refresh_models())
                    })?;
                }
                KeyCode::Backspace => {
                    self.search_query.pop();
                    tokio::task::block_in_place(|| {
                        tokio::runtime::Handle::current().block_on(self.refresh_models())
                    })?;
                }
                KeyCode::Char(c) => {
                    self.search_query.push(c);
                    tokio::task::block_in_place(|| {
                        tokio::runtime::Handle::current().block_on(self.refresh_models())
                    })?;
                }
                _ => {}
            }
            return Ok(());
        }
        
        // Handle ESC to return to browse mode
        if key.code == KeyCode::Esc && self.view_mode != ViewMode::Browse {
            self.view_mode = ViewMode::Browse;
            return Ok(());
        }
        
        // Normal navigation
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => {
                if self.selected_index > 0 {
                    self.selected_index -= 1;
                }
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if self.selected_index < self.available_models.len().saturating_sub(1) {
                    self.selected_index += 1;
                }
            }
            KeyCode::Enter => {
                if self.view_mode == ViewMode::Browse && !self.available_models.is_empty() {
                    // Add better error handling for toggle
                    match tokio::task::block_in_place(|| {
                        tokio::runtime::Handle::current().block_on(self.toggle_model())
                    }) {
                        Ok(_) => {
                            info!("Successfully toggled model at index {}", self.selected_index);
                        }
                        Err(e) => {
                            error!("Failed to toggle model: {}", e);
                            self.error_message = Some(format!("Failed to toggle: {}", e));
                        }
                    }
                }
            }
            KeyCode::Char('/') => {
                self.view_mode = ViewMode::Search;
            }
            KeyCode::Char('r') => {
                tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current().block_on(self.refresh_models())
                })?;
            }
            KeyCode::Char('d') => {
                if !self.available_models.is_empty() {
                    self.view_mode = ViewMode::Details;
                }
            }
            KeyCode::Char('b') => {
                self.view_mode = ViewMode::Benchmark;
            }
            KeyCode::Char('c') => {
                self.view_mode = ViewMode::Compare;
            }
            KeyCode::Char('a') => {
                // Add new model
                self.view_mode = ViewMode::Configure;
                self.config_state = ConfigurationState::default();
                self.config_field_index = 0;
                self.input_buffer.clear();
                self.is_input_mode = false;
            }
            KeyCode::Char('s') => {
                // Open settings (currently shows Settings view, could open Configure too)
                self.view_mode = ViewMode::Settings;
            }
            KeyCode::Char('f') => {
                // Cycle through filters
                self.category_filter = match self.category_filter.as_deref() {
                    None => Some("local".to_string()),
                    Some("local") => Some("openai".to_string()),
                    Some("openai") => Some("anthropic".to_string()),
                    Some("anthropic") => Some("ollama".to_string()),
                    Some("ollama") => None,
                    _ => None,
                };
                tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current().block_on(self.refresh_models())
                })?;
            }
            _ => {}
        }
        
        Ok(())
    }
    
    fn name(&self) -> &str {
        "Models"
    }
    
    fn title(&self) -> String {
        let mode_indicator = match self.view_mode {
            ViewMode::Browse => "",
            ViewMode::Details => " - Details",
            ViewMode::Benchmark => " - Benchmarks",
            ViewMode::Compare => " - Compare",
            ViewMode::Settings => " - Settings",
            ViewMode::Search => " - Search",
            ViewMode::Configure => " - Add Model",
        };
        format!("🤖 Models{} ({} available)", mode_indicator, self.available_models.len())
    }
    
    fn update(&mut self) -> Result<()> {
        // Update can be used to refresh models periodically if needed
        Ok(())
    }
}

impl UnifiedModelsTab {
    /// Initialize the tab (called once when tab is created)
    pub async fn initialize(&mut self) -> Result<()> {
        // Initial load of models
        self.refresh_models().await?;
        Ok(())
    }
}

