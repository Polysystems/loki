//! Models configuration subtab

use std::sync::Arc;
use tokio::sync::RwLock;
use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, List, ListItem, Paragraph},
    Frame,
};
use crossterm::event::{KeyCode, KeyEvent};
use anyhow::Result;

use super::SubtabController;
use crate::tui::chat::orchestration::{OrchestrationManager, model_persistence::{ModelPersistence, ModelSettings}};
use crate::tui::chat::models::{discovery::ModelDiscoveryEngine, catalog::ModelCatalog};

/// Models configuration tab
pub struct ModelsTab {
    /// Orchestration manager
    manager: Arc<RwLock<OrchestrationManager>>,
    
    /// Model discovery engine (optional)
    discovery_engine: Option<Arc<ModelDiscoveryEngine>>,
    
    /// Cached model catalog
    catalog: Option<ModelCatalog>,
    
    /// Selected model index
    selected_index: usize,
    
    /// Available models
    available_models: Vec<ModelInfo>,
    
    /// Model persistence
    persistence: ModelPersistence,
    
    /// Edit mode for adding new model
    add_model_mode: bool,
    
    /// New model input buffer
    new_model_input: String,
    
    /// New model provider
    new_model_provider: String,
    
    /// Loading state
    is_loading: bool,
    
    /// Search query
    search_query: String,
}

#[derive(Debug, Clone)]
struct ModelInfo {
    name: String,
    provider: String,
    enabled: bool,
    capabilities: Vec<String>,
}

impl ModelsTab {
    /// Render add model dialog
    fn render_add_model(&self, f: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .margin(2)
            .constraints([
                Constraint::Length(3),   // Title
                Constraint::Length(3),   // Model name input
                Constraint::Length(3),   // Provider selection
                Constraint::Length(3),   // Instructions
                Constraint::Min(1),      // Spacer
            ])
            .split(area);
        
        // Title
        let title = Paragraph::new("➕ Add New Model")
            .style(Style::default().fg(Color::Green).add_modifier(Modifier::BOLD))
            .alignment(Alignment::Center)
            .block(Block::default().borders(Borders::ALL).border_type(BorderType::Double));
        f.render_widget(title, chunks[0]);
        
        // Model name input
        let input = Paragraph::new(Line::from(vec![
            Span::raw("Model Name: "),
            Span::styled(&self.new_model_input, Style::default().fg(Color::Yellow)),
            Span::styled("_", Style::default().fg(Color::Gray).add_modifier(Modifier::SLOW_BLINK)),
        ]))
        .block(Block::default().borders(Borders::ALL).title(" Enter Model ID "));
        f.render_widget(input, chunks[1]);
        
        // Provider selection
        let provider = Paragraph::new(Line::from(vec![
            Span::raw("Provider: "),
            Span::styled(&self.new_model_provider, Style::default().fg(Color::Cyan)),
            Span::raw(" (Tab to change)"),
        ]))
        .block(Block::default().borders(Borders::ALL).title(" Select Provider "));
        f.render_widget(provider, chunks[2]);
        
        // Instructions
        let instructions = Paragraph::new(vec![
            Line::from("Enter: Save | Esc: Cancel | Tab: Change Provider"),
        ])
        .style(Style::default().fg(Color::DarkGray))
        .alignment(Alignment::Center);
        f.render_widget(instructions, chunks[3]);
    }
    /// Create a new models tab with optional discovery engine
    pub fn new_with_discovery(
        manager: Arc<RwLock<OrchestrationManager>>,
        discovery_engine: Option<Arc<ModelDiscoveryEngine>>,
        fallback_models: Vec<crate::tui::chat::ActiveModel>,
    ) -> Self {
        let mut tab = Self::new(manager.clone(), fallback_models);
        tab.discovery_engine = discovery_engine;
        
        // Start discovery in background if engine available
        if let Some(ref engine) = tab.discovery_engine {
            let engine_clone = engine.clone();
            tokio::spawn(async move {
                let catalog = engine_clone.get_catalog().await;
                tracing::info!("Model catalog ready with {} models", catalog.get_all_models().len());
            });
        }
        
        tab
    }
    
    /// Create a new models tab
    pub fn new(
        manager: Arc<RwLock<OrchestrationManager>>, 
        available_models: Vec<crate::tui::chat::ActiveModel>
    ) -> Self {
        // Convert ActiveModel to ModelInfo
        let mut model_infos = Vec::new();
        
        // Get currently enabled models from orchestration manager
        let enabled_models = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                manager.read().await.enabled_models.clone()
            })
        });
        
        for model in available_models {
            let capabilities = match model.provider.as_str() {
                "openai" => vec!["chat".to_string(), "code".to_string()],
                "anthropic" => vec!["chat".to_string(), "analysis".to_string()],
                "google" => vec!["chat".to_string(), "multimodal".to_string()],
                "ollama" => vec!["chat".to_string(), "local".to_string()],
                "deepseek" => vec!["chat".to_string(), "code".to_string()],
                "grok" => vec!["chat".to_string(), "analysis".to_string()],
                _ => vec!["chat".to_string()],
            };
            
            model_infos.push(ModelInfo {
                name: model.name.clone(),
                provider: model.provider,
                enabled: enabled_models.contains(&model.name),
                capabilities,
            });
        }
        
        // If no models provided, use defaults
        if model_infos.is_empty() {
            model_infos = vec![
                ModelInfo {
                    name: "No models available".to_string(),
                    provider: "Please configure API keys".to_string(),
                    enabled: false,
                    capabilities: vec![],
                },
            ];
        }
        
        let mut tab = Self {
            manager,
            discovery_engine: None,
            catalog: None,
            selected_index: 0,
            available_models: model_infos,
            persistence: ModelPersistence::new(),
            add_model_mode: false,
            new_model_input: String::new(),
            new_model_provider: "openai".to_string(),
            is_loading: false,
            search_query: String::new(),
        };
        
        // Load persisted model states
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                if let Ok(Some(config)) = tab.persistence.load().await {
                    // Apply persisted enabled states
                    for model in &mut tab.available_models {
                        model.enabled = config.enabled_models.contains(&model.name);
                    }
                }
            })
        });
        
        tab
    }
    
    /// Toggle model enabled state
    async fn toggle_model(&mut self) -> Result<()> {
        if self.selected_index < self.available_models.len() {
            let model = &mut self.available_models[self.selected_index];
            model.enabled = !model.enabled;
            
            // Update orchestration manager
            let enabled_models: Vec<String> = self.available_models
                .iter()
                .filter(|m| m.enabled)
                .map(|m| m.name.clone())
                .collect();
            
            let mut manager = self.manager.write().await;
            manager.enabled_models = enabled_models.clone();
            
            // Persist the change
            self.persistence.update_enabled_models(enabled_models).await?;
        }
        
        Ok(())
    }
    
    /// Refresh models from discovery engine
    async fn refresh_from_discovery(&mut self) -> Result<()> {
        if let Some(ref engine) = self.discovery_engine {
            self.is_loading = true;
            
            // Get fresh catalog
            self.catalog = Some(engine.get_catalog().await);
            
            if let Some(ref catalog) = self.catalog {
                // Convert catalog entries to ModelInfo
                let mut new_models = Vec::new();
                
                for entry in catalog.get_all_models() {
                    new_models.push(ModelInfo {
                        name: entry.id.clone(),
                        provider: entry.provider.clone(),
                        enabled: self.manager.read().await.enabled_models.contains(&entry.id),
                        capabilities: entry.capabilities.iter().map(|c| format!("{:?}", c)).collect(),
                    });
                }
                
                if !new_models.is_empty() {
                    self.available_models = new_models;
                    tracing::info!("Refreshed {} models from discovery engine", self.available_models.len());
                }
            }
            
            self.is_loading = false;
        }
        Ok(())
    }
    
    /// Add a new model
    async fn add_model(&mut self) -> Result<()> {
        if self.new_model_input.is_empty() {
            return Ok(());
        }
        
        let new_model = ModelInfo {
            name: self.new_model_input.clone(),
            provider: self.new_model_provider.clone(),
            enabled: true,
            capabilities: match self.new_model_provider.as_str() {
                "openai" => vec!["chat".to_string(), "code".to_string()],
                "anthropic" => vec!["chat".to_string(), "analysis".to_string()],
                "google" => vec!["chat".to_string(), "multimodal".to_string()],
                "ollama" => vec!["chat".to_string(), "local".to_string()],
                _ => vec!["chat".to_string()],
            },
        };
        
        // Add to list
        self.available_models.push(new_model);
        
        // Update orchestration manager
        let enabled_models: Vec<String> = self.available_models
            .iter()
            .filter(|m| m.enabled)
            .map(|m| m.name.clone())
            .collect();
        
        let mut manager = self.manager.write().await;
        manager.enabled_models = enabled_models.clone();
        
        // Persist the change
        self.persistence.update_enabled_models(enabled_models).await?;
        
        // Clear input
        self.new_model_input.clear();
        self.add_model_mode = false;
        
        Ok(())
    }
}

impl SubtabController for ModelsTab {
    fn render(&mut self, f: &mut Frame, area: Rect) {
        // Different layout for add model mode
        if self.add_model_mode {
            self.render_add_model(f, area);
            return;
        }
        
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),   // Title
                Constraint::Min(10),     // Model list
                Constraint::Length(5),   // Model details
            ])
            .split(area);
        
        // Title
        let title = Paragraph::new("🤖 Model Configuration")
            .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
            .alignment(Alignment::Center)
            .block(Block::default().borders(Borders::BOTTOM));
        f.render_widget(title, chunks[0]);
        
        // Model list
        let items: Vec<ListItem> = self.available_models
            .iter()
            .enumerate()
            .map(|(i, model)| {
                let selected = i == self.selected_index;
                let style = if selected {
                    Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
                } else {
                    Style::default()
                };
                
                let status = if model.enabled { "✅" } else { "❌" };
                let content = format!("{} {} ({})", status, model.name, model.provider);
                
                ListItem::new(content).style(style)
            })
            .collect();
        
        let list = List::new(items)
            .block(Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .title(" Available Models "));
        f.render_widget(list, chunks[1]);
        
        // Model details
        if self.selected_index < self.available_models.len() {
            let model = &self.available_models[self.selected_index];
            let details = vec![
                Line::from(vec![
                    Span::raw("Capabilities: "),
                    Span::styled(
                        model.capabilities.join(", "),
                        Style::default().fg(Color::Green),
                    ),
                ]),
                Line::from(""),
                Line::from(Span::styled(
                    "Space: toggle | ↑/↓: navigate | A: add model | D: delete",
                    Style::default().fg(Color::DarkGray),
                )),
            ];
            
            let details_widget = Paragraph::new(details)
                .block(Block::default()
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded)
                    .title(" Details "));
            f.render_widget(details_widget, chunks[2]);
        }
    }
    
    fn handle_input(&mut self, key: KeyEvent) -> Result<()> {
        // Handle add model mode input
        if self.add_model_mode {
            match key.code {
                KeyCode::Esc => {
                    self.add_model_mode = false;
                    self.new_model_input.clear();
                }
                KeyCode::Enter => {
                    tokio::task::block_in_place(|| {
                        tokio::runtime::Handle::current().block_on(self.add_model())
                    })?;
                }
                KeyCode::Backspace => {
                    self.new_model_input.pop();
                }
                KeyCode::Tab => {
                    // Cycle through providers
                    self.new_model_provider = match self.new_model_provider.as_str() {
                        "openai" => "anthropic".to_string(),
                        "anthropic" => "google".to_string(),
                        "google" => "ollama".to_string(),
                        "ollama" => "deepseek".to_string(),
                        "deepseek" => "grok".to_string(),
                        _ => "openai".to_string(),
                    };
                }
                KeyCode::Char(c) => {
                    self.new_model_input.push(c);
                }
                _ => {}
            }
            return Ok(());
        }
        
        // Normal mode input handling
        match key.code {
            KeyCode::Up => {
                if self.selected_index > 0 {
                    self.selected_index -= 1;
                }
            }
            KeyCode::Down => {
                if self.selected_index < self.available_models.len() - 1 {
                    self.selected_index += 1;
                }
            }
            KeyCode::Char(' ') | KeyCode::Enter => {
                tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current().block_on(self.toggle_model())
                })?;
            }
            KeyCode::Char('a') | KeyCode::Char('A') => {
                self.add_model_mode = true;
            }
            KeyCode::Char('r') | KeyCode::Char('R') => {
                // Refresh from discovery
                if self.discovery_engine.is_some() {
                    tokio::task::block_in_place(|| {
                        tokio::runtime::Handle::current().block_on(self.refresh_from_discovery())
                    })?;
                }
            }
            KeyCode::Char('/') => {
                // TODO: Implement search mode
                tracing::info!("Search mode not yet implemented");
            }
            KeyCode::Char('d') | KeyCode::Char('D') => {
                // Delete selected model (if not the last one)
                if self.available_models.len() > 1 && self.selected_index < self.available_models.len() {
                    self.available_models.remove(self.selected_index);
                    if self.selected_index >= self.available_models.len() {
                        self.selected_index = self.available_models.len() - 1;
                    }
                    
                    // Update persistence
                    let enabled_models: Vec<String> = self.available_models
                        .iter()
                        .filter(|m| m.enabled)
                        .map(|m| m.name.clone())
                        .collect();
                    
                    tokio::task::block_in_place(|| {
                        tokio::runtime::Handle::current().block_on(async {
                            let mut manager = self.manager.write().await;
                            manager.enabled_models = enabled_models.clone();
                            let _ = self.persistence.update_enabled_models(enabled_models).await;
                        })
                    });
                }
            }
            _ => {}
        }
        
        Ok(())
    }
    
    fn update(&mut self) -> Result<()> {
        // Sync with orchestration manager
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                let manager = self.manager.read().await;
                for model in &mut self.available_models {
                    model.enabled = manager.enabled_models.contains(&model.name);
                }
            })
        });
        
        Ok(())
    }
    
    fn name(&self) -> &str {
        "Models"
    }
}