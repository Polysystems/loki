//! Enhanced models configuration subtab that uses ModelDiscoveryEngine
//!
//! This replaces the basic models_tab with real model discovery and catalog browsing

use std::sync::Arc;
use tokio::sync::RwLock;
use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect, Margin},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, List, ListItem, Paragraph, Gauge, Sparkline, Row, Table, Wrap},
    Frame,
};
use crossterm::event::{KeyCode, KeyEvent};
use anyhow::Result;

use super::SubtabController;
use crate::tui::chat::{
    models::{discovery::ModelDiscoveryEngine, catalog::{ModelCatalog, ModelEntry, ModelCategory}},
    orchestration::OrchestrationManager,
};

/// View modes for the models tab
#[derive(Debug, Clone, PartialEq)]
enum ViewMode {
    Browse,      // Browse discovered models
    Details,     // View detailed model info
    Benchmark,   // Run benchmarks
    Compare,     // Compare models side-by-side
    Search,      // Search models
}

/// Enhanced models tab with real model discovery
pub struct EnhancedModelsTab {
    /// Model discovery engine
    discovery_engine: Arc<ModelDiscoveryEngine>,
    
    /// Cached model catalog
    catalog: ModelCatalog,
    
    /// Orchestration manager reference
    orchestration: Arc<RwLock<OrchestrationManager>>,
    
    /// Current view mode
    view_mode: ViewMode,
    
    /// Selected model index in current view
    selected_index: usize,
    
    /// Currently displayed models (filtered/sorted)
    displayed_models: Vec<ModelEntry>,
    
    /// Search query
    search_query: String,
    
    /// Selected category filter
    category_filter: Option<ModelCategory>,
    
    /// Comparison selections
    compare_models: Vec<String>,
    
    /// Loading state
    is_loading: bool,
    
    /// Error message
    error_message: Option<String>,
}

impl EnhancedModelsTab {
    /// Create a new enhanced models tab
    pub fn new(
        discovery_engine: Arc<ModelDiscoveryEngine>,
        orchestration: Arc<RwLock<OrchestrationManager>>,
    ) -> Self {
        let catalog = ModelCatalog::new();
        let displayed_models = Vec::new();
        
        Self {
            discovery_engine,
            catalog,
            orchestration,
            view_mode: ViewMode::Browse,
            selected_index: 0,
            displayed_models,
            search_query: String::new(),
            category_filter: None,
            compare_models: Vec::new(),
            is_loading: false,
            error_message: None,
        }
    }
    
    /// Refresh model catalog from discovery engine
    async fn refresh_catalog(&mut self) -> Result<()> {
        self.is_loading = true;
        self.error_message = None;
        
        match self.discovery_engine.discover_all().await {
            Ok(count) => {
                self.catalog = self.discovery_engine.get_catalog().await;
                self.update_displayed_models();
                tracing::info!("Discovered {} models", count);
            }
            Err(e) => {
                self.error_message = Some(format!("Failed to discover models: {}", e));
                tracing::error!("Model discovery failed: {}", e);
            }
        }
        
        self.is_loading = false;
        Ok(())
    }
    
    /// Update displayed models based on filters
    fn update_displayed_models(&mut self) {
        let mut models = self.catalog.get_all_models();
        
        // Apply category filter
        if let Some(category) = &self.category_filter {
            models = self.catalog.get_by_category(category.clone());
        }
        
        // Apply search filter
        if !self.search_query.is_empty() {
            let query = self.search_query.to_lowercase();
            models.retain(|m| 
                m.name.to_lowercase().contains(&query) ||
                m.provider.to_lowercase().contains(&query) ||
                m.description.to_lowercase().contains(&query)
            );
        }
        
        // Sort by provider and name
        models.sort_by(|a, b| {
            a.provider.cmp(&b.provider)
                .then(a.name.cmp(&b.name))
        });
        
        self.displayed_models = models;
        
        // Reset selection if out of bounds
        if self.selected_index >= self.displayed_models.len() && !self.displayed_models.is_empty() {
            self.selected_index = self.displayed_models.len() - 1;
        }
    }
    
    /// Render the browse view
    fn render_browse_view(&self, f: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),   // Header with stats
                Constraint::Length(3),   // Search/filter bar
                Constraint::Min(10),     // Model list
                Constraint::Length(6),   // Selected model preview
            ])
            .split(area);
        
        // Header with discovery stats
        let total = self.catalog.get_all_models().len();
        let providers = self.discovery_engine.get_provider_count();
        let header_text = format!(
            "📊 {} models from {} providers | 🔄 Press 'r' to refresh",
            total, providers
        );
        
        let header = Paragraph::new(header_text)
            .style(Style::default().fg(Color::Cyan))
            .alignment(Alignment::Center)
            .block(Block::default()
                .borders(Borders::BOTTOM)
                .border_style(Style::default().fg(Color::DarkGray)));
        f.render_widget(header, chunks[0]);
        
        // Search and filter bar
        self.render_search_bar(f, chunks[1]);
        
        // Model list
        if self.is_loading {
            let loading = Paragraph::new("🔄 Discovering models...")
                .style(Style::default().fg(Color::Yellow))
                .alignment(Alignment::Center)
                .block(Block::default().borders(Borders::ALL));
            f.render_widget(loading, chunks[2]);
        } else if let Some(error) = &self.error_message {
            let error_widget = Paragraph::new(error.as_str())
                .style(Style::default().fg(Color::Red))
                .alignment(Alignment::Center)
                .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(Color::Red)));
            f.render_widget(error_widget, chunks[2]);
        } else {
            self.render_model_list(f, chunks[2]);
        }
        
        // Selected model preview
        self.render_model_preview(f, chunks[3]);
    }
    
    /// Render search/filter bar
    fn render_search_bar(&self, f: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(60),  // Search
                Constraint::Percentage(40),  // Category filter
            ])
            .split(area);
        
        // Search input
        let search_text = if self.search_query.is_empty() {
            "🔍 Type to search models...".to_string()
        } else {
            format!("🔍 {}", self.search_query)
        };
        
        let search = Paragraph::new(search_text)
            .style(if self.view_mode == ViewMode::Search {
                Style::default().fg(Color::Yellow)
            } else {
                Style::default().fg(Color::DarkGray)
            })
            .block(Block::default().borders(Borders::ALL).title(" Search "));
        f.render_widget(search, chunks[0]);
        
        // Category filter
        let category_text = if let Some(cat) = &self.category_filter {
            format!("📁 {:?}", cat)
        } else {
            "📁 All Categories".to_string()
        };
        
        let filter = Paragraph::new(category_text)
            .style(Style::default().fg(Color::Green))
            .block(Block::default().borders(Borders::ALL).title(" Filter "));
        f.render_widget(filter, chunks[1]);
    }
    
    /// Render the model list
    fn render_model_list(&self, f: &mut Frame, area: Rect) {
        let items: Vec<ListItem> = self.displayed_models
            .iter()
            .enumerate()
            .map(|(i, model)| {
                let selected = i == self.selected_index;
                
                // Model icon based on category
                let icon = match model.category {
                    ModelCategory::Chat => "💬",
                    ModelCategory::Code => "💻",
                    ModelCategory::Vision => "👁️",
                    ModelCategory::Audio => "🎵",
                    ModelCategory::Embedding => "🔤",
                    ModelCategory::Specialized => "⚡",
                };
                
                // Availability indicator
                let status = if model.availability.is_available {
                    "✅"
                } else {
                    "⚠️"
                };
                
                // Price indicator
                let price_indicator = if let Some(price) = model.pricing.per_1k_tokens {
                    if price < 0.001 {
                        "💚" // Very cheap
                    } else if price < 0.01 {
                        "💛" // Moderate
                    } else {
                        "💰" // Expensive
                    }
                } else {
                    "❓"
                };
                
                let content = format!(
                    "{} {} {} {} | {} | {}k ctx",
                    if selected { "▶" } else { " " },
                    status,
                    icon,
                    model.name,
                    model.provider,
                    model.context_window / 1000
                );
                
                let style = if selected {
                    Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
                } else if !model.availability.is_available {
                    Style::default().fg(Color::DarkGray)
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
                .title(format!(" {} Models ", self.displayed_models.len())));
        f.render_widget(list, area);
    }
    
    /// Render model preview
    fn render_model_preview(&self, f: &mut Frame, area: Rect) {
        if let Some(model) = self.displayed_models.get(self.selected_index) {
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
            
            // Performance metrics
            if let Some(latency) = model.performance.avg_latency_ms {
                lines.push(Line::from(vec![
                    Span::raw("⚡ "),
                    Span::styled(
                        format!("Latency: {}ms", latency),
                        Style::default().fg(Color::Cyan)
                    ),
                ]));
            }
            
            // Pricing
            if let Some(price) = model.pricing.per_1k_tokens {
                lines.push(Line::from(vec![
                    Span::raw("💵 "),
                    Span::styled(
                        format!("${:.4}/1k tokens", price),
                        Style::default().fg(Color::Yellow)
                    ),
                ]));
            }
            
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
}

impl SubtabController for EnhancedModelsTab {
    fn render(&mut self, f: &mut Frame, area: Rect) {
        match self.view_mode {
            ViewMode::Browse => self.render_browse_view(f, area),
            ViewMode::Details => {
                // TODO: Implement detailed view
                self.render_browse_view(f, area)
            }
            ViewMode::Benchmark => {
                // TODO: Implement benchmark view
                self.render_browse_view(f, area)
            }
            ViewMode::Compare => {
                // TODO: Implement comparison view
                self.render_browse_view(f, area)
            }
            ViewMode::Search => self.render_browse_view(f, area),
        }
    }
    
    fn handle_input(&mut self, key: KeyEvent) -> Result<()> {
        // Handle search mode
        if self.view_mode == ViewMode::Search {
            match key.code {
                KeyCode::Esc => {
                    self.view_mode = ViewMode::Browse;
                    self.search_query.clear();
                    self.update_displayed_models();
                }
                KeyCode::Enter => {
                    self.view_mode = ViewMode::Browse;
                    self.update_displayed_models();
                }
                KeyCode::Backspace => {
                    self.search_query.pop();
                    self.update_displayed_models();
                }
                KeyCode::Char(c) => {
                    self.search_query.push(c);
                    self.update_displayed_models();
                }
                _ => {}
            }
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
                if self.selected_index < self.displayed_models.len().saturating_sub(1) {
                    self.selected_index += 1;
                }
            }
            KeyCode::Char('/') => {
                self.view_mode = ViewMode::Search;
            }
            KeyCode::Char('r') => {
                // Refresh catalog
                tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current().block_on(self.refresh_catalog())
                })?;
            }
            KeyCode::Char('f') => {
                // Cycle through category filters
                self.category_filter = match self.category_filter {
                    None => Some(ModelCategory::Chat),
                    Some(ModelCategory::Chat) => Some(ModelCategory::Code),
                    Some(ModelCategory::Code) => Some(ModelCategory::Vision),
                    Some(ModelCategory::Vision) => Some(ModelCategory::Audio),
                    Some(ModelCategory::Audio) => Some(ModelCategory::Embedding),
                    Some(ModelCategory::Embedding) => Some(ModelCategory::Specialized),
                    Some(ModelCategory::Specialized) => None,
                };
                self.update_displayed_models();
            }
            KeyCode::Enter => {
                // Enable/disable model in orchestration
                if let Some(model) = self.displayed_models.get(self.selected_index) {
                    tokio::task::block_in_place(|| {
                        tokio::runtime::Handle::current().block_on(async {
                            let mut orch = self.orchestration.write().await;
                            if orch.enabled_models.contains(&model.id) {
                                orch.enabled_models.retain(|m| m != &model.id);
                            } else {
                                orch.enabled_models.push(model.id.clone());
                            }
                        })
                    });
                }
            }
            KeyCode::Char('c') => {
                // Add to comparison
                if let Some(model) = self.displayed_models.get(self.selected_index) {
                    if !self.compare_models.contains(&model.id) {
                        self.compare_models.push(model.id.clone());
                        if self.compare_models.len() >= 2 {
                            self.view_mode = ViewMode::Compare;
                        }
                    }
                }
            }
            KeyCode::Char('d') => {
                // View details
                if self.selected_index < self.displayed_models.len() {
                    self.view_mode = ViewMode::Details;
                }
            }
            KeyCode::Char('b') => {
                // Run benchmark
                if self.selected_index < self.displayed_models.len() {
                    self.view_mode = ViewMode::Benchmark;
                }
            }
            _ => {}
        }
        
        Ok(())
    }
    
    fn name(&self) -> &str {
        "Models"
    }
    
    fn title(&self) -> String {
        format!("🤖 Model Catalog ({} models)", self.displayed_models.len())
    }
}