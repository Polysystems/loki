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
    models::{discovery::{ModelDiscoveryEngine, AvailabilityStatus}, catalog::{ModelCatalog, ModelEntry, ModelCategory}},
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
    
    /// Table state for benchmark view
    table_state: ratatui::widgets::TableState,
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
            table_state: ratatui::widgets::TableState::default(),
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
                    ModelCategory::Instruct => "📝",
                    ModelCategory::Video => "🎬",
                    ModelCategory::General => "⚡",
                };
                
                // Availability indicator
                let status = match model.availability {
                    AvailabilityStatus::Available => "✅",
                    AvailabilityStatus::Limited => "🟡",
                    AvailabilityStatus::Unavailable => "⚠️",
                    AvailabilityStatus::Deprecated => "🔴",
                    AvailabilityStatus::Unknown => "❓",
                };
                
                // Price indicator
                // Use average of input/output pricing for indicator
                let avg_price = (model.pricing.input_per_1k_tokens + model.pricing.output_per_1k_tokens) / 2.0;
                let price_indicator = if avg_price < 0.001 {
                    "💚" // Very cheap
                } else if avg_price < 0.01 {
                    "💛" // Moderate
                } else {
                    "💰" // Expensive
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
                } else if model.availability != AvailabilityStatus::Available {
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
            if let Some(latency) = model.performance.latency_ms {
                lines.push(Line::from(vec![
                    Span::raw("⚡ "),
                    Span::styled(
                        format!("Latency: {}ms", latency),
                        Style::default().fg(Color::Cyan)
                    ),
                ]));
            }
            
            // Pricing
            let avg_price = (model.pricing.input_per_1k_tokens + model.pricing.output_per_1k_tokens) / 2.0;
            if avg_price > 0.0 {
                lines.push(Line::from(vec![
                    Span::raw("💵 "),
                    Span::styled(
                        format!("${:.4}/1k tokens (avg)", avg_price),
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

impl EnhancedModelsTab {
    /// Render benchmark view showing performance metrics
    fn render_benchmark_view(&mut self, f: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),   // Title
                Constraint::Min(0),      // Benchmark results
            ])
            .split(area);
        
        // Title
        let title = Paragraph::new("⚡ Model Benchmarks")
            .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
            .alignment(Alignment::Center)
            .block(Block::default().borders(Borders::BOTTOM));
        f.render_widget(title, chunks[0]);
        
        // Collect benchmark data
        let mut benchmark_data: Vec<(&str, f64, f64, u64)> = Vec::new();
        for model in &self.displayed_models {
            // Use performance metrics directly from model
            if model.performance.latency_ms.is_some() || model.performance.tokens_per_second.is_some() {
                // Extract available performance data
                let latency = model.performance.latency_ms.unwrap_or(0.0);
                let tokens_per_sec = model.performance.tokens_per_second.unwrap_or(0.0);
                let accuracy = model.performance.accuracy_score.unwrap_or(1.0) * 100.0;
                benchmark_data.push((
                    &model.id,
                    latency,
                    accuracy,
                    tokens_per_sec as u64,
                ));
            }
        }
        
        // Sort by response time
        benchmark_data.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Create benchmark table
        let header = Row::new(vec!["Model", "Latency (ms)", "Accuracy", "Tokens/sec"])
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
        .highlight_style(Style::default().add_modifier(Modifier::BOLD))
        .highlight_symbol("→ ");
        
        f.render_stateful_widget(table, chunks[1], &mut self.table_state);
        
        // Instructions
        let instructions = Paragraph::new("Use ↑/↓ to navigate | Press B to run benchmark | ESC to return")
            .style(Style::default().fg(Color::DarkGray))
            .alignment(Alignment::Center);
        f.render_widget(instructions, Rect {
            x: chunks[1].x,
            y: chunks[1].y + chunks[1].height - 1,
            width: chunks[1].width,
            height: 1,
        });
    }
    
    /// Render comparison view for multiple models
    fn render_compare_view(&mut self, f: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),   // Title
                Constraint::Min(0),      // Comparison
            ])
            .split(area);
        
        // Title
        let title = Paragraph::new("📊 Model Comparison")
            .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
            .alignment(Alignment::Center)
            .block(Block::default().borders(Borders::BOTTOM));
        f.render_widget(title, chunks[0]);
        
        // Get up to 3 models to compare
        let models_to_compare: Vec<_> = self.displayed_models.iter()
            .take(3)
            .collect();
        
        if models_to_compare.is_empty() {
            let message = Paragraph::new("No models available for comparison")
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
                    Span::styled(&model.id, Style::default().add_modifier(Modifier::BOLD).fg(Color::Yellow)),
                ]),
                Line::from(""),
                Line::from(vec![
                    Span::styled("Provider: ", Style::default().add_modifier(Modifier::BOLD)),
                    Span::raw(&model.provider),
                ]),
                Line::from(""),
                Line::from(vec![
                    Span::styled("Capabilities:", Style::default().add_modifier(Modifier::BOLD)),
                ]),
            ];
            
            for cap in &model.capabilities {
                lines.push(Line::from(vec![
                    Span::raw("• "),
                    Span::raw(format!("{:?}", cap)),
                ]));
            }
            
            lines.push(Line::from(""));
            
            // Use performance metrics directly from model
            if model.performance.latency_ms.is_some() || model.performance.tokens_per_second.is_some() {
                lines.push(Line::from(vec![
                    Span::styled("Performance:", Style::default().add_modifier(Modifier::BOLD)),
                ]));
                if let Some(latency) = model.performance.latency_ms {
                    lines.push(Line::from(format!("Latency: {:.0}ms", latency)));
                }
                if let Some(tps) = model.performance.tokens_per_second {
                    lines.push(Line::from(format!("Speed: {:.0} tok/s", tps)));
                }
                if let Some(accuracy) = model.performance.accuracy_score {
                    lines.push(Line::from(format!("Accuracy: {:.1}%", accuracy * 100.0)));
                }
            }
            
            let comparison = Paragraph::new(lines)
                .block(Block::default()
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded));
            f.render_widget(comparison, comparison_chunks[i]);
        }
        
        // Instructions
        let instructions = Paragraph::new("Press SPACE to select models for comparison | ESC to return")
            .style(Style::default().fg(Color::DarkGray))
            .alignment(Alignment::Center);
        f.render_widget(instructions, Rect {
            x: chunks[1].x,
            y: chunks[1].y + chunks[1].height - 1,
            width: chunks[1].width,
            height: 1,
        });
    }
    
    /// Render detailed view of selected model
    fn render_details_view(&mut self, f: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),   // Title
                Constraint::Min(0),      // Details
            ])
            .split(area);
        
        // Title
        let title = Paragraph::new("📊 Model Details")
            .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
            .alignment(Alignment::Center)
            .block(Block::default().borders(Borders::BOTTOM));
        f.render_widget(title, chunks[0]);
        
        // Get selected model details
        if let Some(model) = self.displayed_models.get(self.selected_index) {
            let mut lines = vec![
                Line::from(vec![
                    Span::styled("Model: ", Style::default().add_modifier(Modifier::BOLD)),
                    Span::raw(&model.id),
                ]),
                Line::from(vec![
                    Span::styled("Provider: ", Style::default().add_modifier(Modifier::BOLD)),
                    Span::styled(&model.provider, Style::default().fg(Color::Yellow)),
                ]),
                Line::from(""),
                Line::from(vec![
                    Span::styled("Capabilities: ", Style::default().add_modifier(Modifier::BOLD)),
                ]),
            ];
            
            for cap in &model.capabilities {
                lines.push(Line::from(vec![
                    Span::raw("  • "),
                    Span::styled(format!("{:?}", cap), Style::default().fg(Color::Green)),
                ]));
            }
            
            lines.push(Line::from(""));
            lines.push(Line::from(vec![
                Span::styled("Statistics: ", Style::default().add_modifier(Modifier::BOLD)),
            ]));
            
            // Use performance metrics directly from model
            if model.performance.latency_ms.is_some() || model.performance.tokens_per_second.is_some() {
                if let Some(latency) = model.performance.latency_ms {
                    lines.push(Line::from(vec![
                        Span::raw("  Latency: "),
                        Span::styled(format!("{:.0}ms", latency), Style::default().fg(Color::Yellow)),
                    ]));
                }
                if let Some(tps) = model.performance.tokens_per_second {
                    lines.push(Line::from(vec![
                        Span::raw("  Throughput: "),
                        Span::styled(format!("{:.0} tokens/sec", tps), Style::default().fg(Color::Cyan)),
                    ]));
                }
                if let Some(accuracy) = model.performance.accuracy_score {
                    lines.push(Line::from(vec![
                        Span::raw("  Accuracy Score: "),
                        Span::styled(format!("{:.1}%", accuracy * 100.0), Style::default().fg(Color::Green)),
                    ]));
                }
            } else {
                lines.push(Line::from("  No performance data available"));
            }
            
            lines.push(Line::from(""));
            lines.push(Line::from(vec![
                Span::styled("Press ESC to return to browse mode", Style::default().fg(Color::DarkGray)),
            ]));
            
            let details = Paragraph::new(lines)
                .block(Block::default()
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded)
                    .title(" Model Information "));
            f.render_widget(details, chunks[1]);
        }
    }
}

impl SubtabController for EnhancedModelsTab {
    fn render(&mut self, f: &mut Frame, area: Rect) {
        match self.view_mode {
            ViewMode::Browse => self.render_browse_view(f, area),
            ViewMode::Details => {
                self.render_details_view(f, area)
            }
            ViewMode::Benchmark => {
                self.render_benchmark_view(f, area)
            }
            ViewMode::Compare => {
                self.render_compare_view(f, area)
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
                    Some(ModelCategory::Embedding) => Some(ModelCategory::Instruct),
                    Some(ModelCategory::Instruct) => Some(ModelCategory::Video),
                    Some(ModelCategory::Video) => Some(ModelCategory::General),
                    Some(ModelCategory::General) => None,
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