//! Chat settings subtab

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
use anyhow::{Result, Context};

use super::SubtabController;
use crate::tui::chat::ChatSettings;
use crate::tui::chat::orchestration::{OrchestrationManager, RoutingStrategy, OrchestrationSetup};
use crate::tui::chat::ui_enhancements::{Theme, SmoothProgress, ToastManager, ToastType};

/// Setting categories
#[derive(Debug, Clone, Copy, PartialEq)]
enum SettingCategory {
    General,
    ModelConfig,
    Orchestration,
    Interface,
    Performance,
}

impl SettingCategory {
    fn all() -> Vec<Self> {
        vec![
            Self::General,
            Self::ModelConfig,
            Self::Orchestration,
            Self::Interface,
            Self::Performance,
        ]
    }
    
    fn name(&self) -> &str {
        match self {
            Self::General => "General",
            Self::ModelConfig => "Model Configuration",
            Self::Orchestration => "Orchestration",
            Self::Interface => "Interface",
            Self::Performance => "Performance",
        }
    }
}

/// Individual setting item
#[derive(Debug, Clone)]
struct SettingItem {
    name: String,
    description: String,
    value: SettingValue,
    category: SettingCategory,
}

/// Setting value types
#[derive(Debug, Clone)]
enum SettingValue {
    Bool(bool),
    Float(f32),
    Integer(u32),
    String(String),
    Selection { current: usize, options: Vec<String> },
}

impl SettingValue {
    fn to_string(&self) -> String {
        match self {
            Self::Bool(b) => if *b { "✓ Enabled" } else { "✗ Disabled" }.to_string(),
            Self::Float(f) => format!("{:.2}", f),
            Self::Integer(i) => i.to_string(),
            Self::String(s) => s.clone(),
            Self::Selection { current, options } => {
                options.get(*current).cloned().unwrap_or_else(|| "None".to_string())
            }
        }
    }
    
    fn toggle(&mut self) {
        match self {
            Self::Bool(b) => *b = !*b,
            Self::Selection { current, options } => {
                *current = (*current + 1) % options.len();
            }
            _ => {}
        }
    }
    
    fn set_string(&mut self, value: String) -> Result<()> {
        match self {
            Self::String(s) => {
                *s = value;
                Ok(())
            }
            Self::Float(f) => {
                // Try to parse as float
                match value.parse::<f32>() {
                    Ok(parsed) if parsed >= 0.0 && parsed <= 1.0 => {
                        *f = parsed;
                        Ok(())
                    }
                    _ => Err(anyhow::anyhow!("Invalid float value. Must be between 0.0 and 1.0"))
                }
            }
            Self::Integer(i) => {
                // Try to parse as integer
                match value.parse::<u32>() {
                    Ok(parsed) => {
                        *i = parsed;
                        Ok(())
                    }
                    _ => Err(anyhow::anyhow!("Invalid integer value"))
                }
            }
            _ => Err(anyhow::anyhow!("This value type doesn't support string editing"))
        }
    }
    
    fn can_increment(&self) -> bool {
        matches!(self, Self::Float(_) | Self::Integer(_))
    }
    
    fn can_decrement(&self) -> bool {
        matches!(self, Self::Float(_) | Self::Integer(_))
    }
    
    fn increment(&mut self) {
        match self {
            Self::Float(f) => *f = (*f + 0.1).min(1.0),
            Self::Integer(i) => *i = i.saturating_add(1),
            _ => {}
        }
    }
    
    fn decrement(&mut self) {
        match self {
            Self::Float(f) => *f = (*f - 0.1).max(0.0),
            Self::Integer(i) => *i = i.saturating_sub(1),
            _ => {}
        }
    }
}

/// Chat settings tab
pub struct SettingsTab {
    /// Reference to orchestration manager
    orchestration: Arc<RwLock<OrchestrationManager>>,
    
    /// Current settings (working copy)
    settings: ChatSettings,
    
    /// Orchestration settings (working copy)
    orchestration_settings: OrchestrationManager,
    
    /// All setting items
    items: Vec<SettingItem>,
    
    /// Current category
    current_category: SettingCategory,
    
    /// Selected item index within category
    selected_index: usize,
    
    /// Edit mode for string values
    edit_mode: bool,
    
    /// Temporary string buffer for editing
    edit_buffer: String,
    
    /// Changes made flag
    has_changes: bool,
    
    /// Current theme
    theme: Theme,
    
    /// Save progress indicator
    save_progress: SmoothProgress,
    
    /// Toast notifications
    toast_manager: ToastManager,
}

impl SettingsTab {
    pub fn new() -> Self {
        // Create dummy orchestration for now
        let dummy_orchestration = Arc::new(RwLock::new(OrchestrationManager::default()));
        
        Self {
            orchestration: dummy_orchestration,
            settings: ChatSettings::default(),
            orchestration_settings: OrchestrationManager::default(),
            items: Vec::new(),
            current_category: SettingCategory::General,
            selected_index: 0,
            edit_mode: false,
            edit_buffer: String::new(),
            has_changes: false,
            theme: Theme::default(),
            save_progress: SmoothProgress::new(),
            toast_manager: ToastManager::new(),
        }
    }
    
    /// Set the references
    pub fn set_state(&mut self, orchestration: Arc<RwLock<OrchestrationManager>>) {
        self.orchestration = orchestration;
        self.load_settings();
    }
    
    /// Set the current settings
    pub fn set_settings(&mut self, settings: ChatSettings) {
        self.settings = settings;
        self.build_items();
    }
    
    /// Load settings from state and disk
    fn load_settings(&mut self) {
        // Load orchestration settings
        self.orchestration_settings = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                let orch = self.orchestration.read().await;
                orch.clone()
            })
        });
        
        // Try to load chat settings from disk
        if let Ok(loaded_settings) = self.load_settings_from_disk() {
            self.settings = loaded_settings;
            self.toast_manager.add_toast(
                "Settings loaded from disk".to_string(),
                ToastType::Info
            );
        } else {
            tracing::info!("Using default settings");
        }
        
        // Build items list
        self.build_items();
    }
    
    /// Check if a setting value can be edited as string
    fn can_edit_as_string(value: &SettingValue) -> bool {
        matches!(value, SettingValue::String(_) | SettingValue::Float(_) | SettingValue::Integer(_))
    }
    
    /// Load settings from disk using StatePersistence
    fn load_settings_from_disk(&self) -> Result<ChatSettings> {
        // Get the data directory
        let data_dir = std::env::var("LOKI_DATA_DIR")
            .unwrap_or_else(|_| "./data".to_string());
        let storage_dir = std::path::PathBuf::from(data_dir);
        
        // Create persistence handler
        let persistence = crate::tui::chat::state::persistence::StatePersistence::new(storage_dir)?;
        
        // Load settings
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                persistence.load_settings().await
            })
        })
    }
    
    /// Save settings back to state and persist to disk
    fn save_settings(&mut self) {
        if !self.has_changes {
            return;
        }
        
        // Start save progress animation
        self.save_progress.set_progress(0.0);
        
        // Save orchestration settings
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                let mut orch = self.orchestration.write().await;
                *orch = self.orchestration_settings.clone();
            })
        });
        
        self.save_progress.set_progress(0.5);
        
        // Persist chat settings to disk
        if let Err(e) = self.persist_settings_to_disk() {
            tracing::error!("Failed to persist settings: {}", e);
            self.toast_manager.add_toast(
                format!("Failed to save settings: {}", e),
                ToastType::Error
            );
        } else {
            self.toast_manager.add_toast(
                "Settings saved successfully!".to_string(),
                ToastType::Success
            );
        }
        
        self.save_progress.set_progress(1.0);
        self.has_changes = false;
    }
    
    /// Persist settings to disk using StatePersistence
    fn persist_settings_to_disk(&self) -> Result<()> {
        // Get the data directory
        let data_dir = std::env::var("LOKI_DATA_DIR")
            .unwrap_or_else(|_| "./data".to_string());
        let storage_dir = std::path::PathBuf::from(data_dir);
        
        // Create persistence handler
        let persistence = crate::tui::chat::state::persistence::StatePersistence::new(storage_dir)?;
        
        // Save settings
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                persistence.save_settings(&self.settings).await
            })
        })?;
        
        Ok(())
    }
    
    /// Get the current settings
    pub fn get_settings(&self) -> ChatSettings {
        self.settings.clone()
    }
    
    /// Check if settings have changed
    pub fn has_unsaved_changes(&self) -> bool {
        self.has_changes
    }
    
    /// Get the changed settings if any
    pub fn get_changed_settings(&self) -> Option<ChatSettings> {
        if self.has_changes {
            Some(self.settings.clone())
        } else {
            None
        }
    }
    
    /// Build the items list
    fn build_items(&mut self) {
        self.items.clear();
        
        // General settings
        self.items.push(SettingItem {
            name: "Store History".to_string(),
            description: "Save chat history to disk".to_string(),
            value: SettingValue::Bool(self.settings.store_history),
            category: SettingCategory::General,
        });
        
        self.items.push(SettingItem {
            name: "Auto Save".to_string(),
            description: "Automatically save conversations".to_string(),
            value: SettingValue::Bool(self.settings.auto_save),
            category: SettingCategory::General,
        });
        
        
        // Model Configuration
        self.items.push(SettingItem {
            name: "Temperature".to_string(),
            description: "Model randomness (0.0 - 1.0)".to_string(),
            value: SettingValue::Float(self.settings.temperature),
            category: SettingCategory::ModelConfig,
        });
        
        self.items.push(SettingItem {
            name: "Max Tokens".to_string(),
            description: "Maximum response length".to_string(),
            value: SettingValue::Integer(self.settings.max_tokens),
            category: SettingCategory::ModelConfig,
        });
        
        
        // Orchestration settings
        self.items.push(SettingItem {
            name: "Orchestration Enabled".to_string(),
            description: "Enable multi-model orchestration".to_string(),
            value: SettingValue::Bool(self.orchestration_settings.orchestration_enabled),
            category: SettingCategory::Orchestration,
        });
        
        self.items.push(SettingItem {
            name: "Orchestration Mode".to_string(),
            description: "How to coordinate multiple models".to_string(),
            value: SettingValue::Selection {
                current: match self.orchestration_settings.current_setup {
                    OrchestrationSetup::SingleModel => 0,
                    OrchestrationSetup::MultiModelRouting => 1,
                    OrchestrationSetup::EnsembleVoting => 2,
                    OrchestrationSetup::SpecializedAgents => 3,
                },
                options: vec![
                    "Single Model".to_string(),
                    "Multi-Model Routing".to_string(),
                    "Ensemble Voting".to_string(),
                    "Specialized Agents".to_string(),
                ],
            },
            category: SettingCategory::Orchestration,
        });
        
        self.items.push(SettingItem {
            name: "Routing Strategy".to_string(),
            description: "How to route queries to models".to_string(),
            value: SettingValue::Selection {
                current: match &self.orchestration_settings.preferred_strategy {
                    RoutingStrategy::RoundRobin => 0,
                    RoutingStrategy::LeastLatency => 1,
                    RoutingStrategy::ContextAware => 2,
                    RoutingStrategy::CapabilityBased => 3,
                    RoutingStrategy::CostOptimized => 4,
                    RoutingStrategy::Custom(_) => 5,
                    RoutingStrategy::Capability => 3,
                    RoutingStrategy::Cost => 4,
                    RoutingStrategy::Speed => 1,
                    RoutingStrategy::Quality => 3,
                    RoutingStrategy::QualityFirst => 3,
                    RoutingStrategy::Availability => 0,
                    RoutingStrategy::Hybrid => 2,
                    RoutingStrategy::Adaptive => 2,
                },
                options: vec![
                    "Round Robin".to_string(),
                    "Least Latency".to_string(),
                    "Context Aware".to_string(),
                    "Capability Based".to_string(),
                    "Cost Optimized".to_string(),
                    "Custom".to_string(),
                ],
            },
            category: SettingCategory::Orchestration,
        });
        
        self.items.push(SettingItem {
            name: "Local Model Preference".to_string(),
            description: "Prefer local over API models (0.0 - 1.0)".to_string(),
            value: SettingValue::Float(self.orchestration_settings.local_models_preference),
            category: SettingCategory::Orchestration,
        });
        
        // Interface settings
        self.items.push(SettingItem {
            name: "Dark Theme".to_string(),
            description: "Use dark theme for the interface".to_string(),
            value: SettingValue::Bool(self.settings.dark_theme),
            category: SettingCategory::Interface,
        });
        
        self.items.push(SettingItem {
            name: "Word Wrap".to_string(),
            description: "Wrap long lines in chat".to_string(),
            value: SettingValue::Bool(self.settings.word_wrap),
            category: SettingCategory::Interface,
        });
        
        
        // Performance settings
        self.items.push(SettingItem {
            name: "Thread Count".to_string(),
            description: "Number of worker threads".to_string(),
            value: SettingValue::Integer(self.settings.threads as u32),
            category: SettingCategory::Performance,
        });
        
        // Add some string-based settings for demonstration
        self.items.push(SettingItem {
            name: "API Endpoint".to_string(),
            description: "Custom API endpoint URL".to_string(),
            value: SettingValue::String(self.settings.api_endpoint.clone().unwrap_or_else(|| "default".to_string())),
            category: SettingCategory::General,
        });
        
        self.items.push(SettingItem {
            name: "Model Name".to_string(),
            description: "Default model name override".to_string(),
            value: SettingValue::String(self.settings.default_model.clone().unwrap_or_else(|| "auto".to_string())),
            category: SettingCategory::ModelConfig,
        });
    }
    
    /// Get items for current category
    fn get_category_items(&self) -> Vec<&SettingItem> {
        self.items
            .iter()
            .filter(|item| item.category == self.current_category)
            .collect()
    }
    
    /// Update setting value based on item
    fn update_setting_value(&mut self, item: &SettingItem) {
        match (item.name.as_str(), &item.value) {
            // General settings
            ("Store History", SettingValue::Bool(v)) => self.settings.store_history = *v,
            ("Auto Save", SettingValue::Bool(v)) => self.settings.auto_save = *v,
            
            // Model Configuration
            ("Temperature", SettingValue::Float(v)) => self.settings.temperature = *v,
            ("Max Tokens", SettingValue::Integer(v)) => self.settings.max_tokens = *v,
            
            // Orchestration
            ("Orchestration Enabled", SettingValue::Bool(v)) => self.orchestration_settings.orchestration_enabled = *v,
            ("Orchestration Mode", SettingValue::Selection { current, .. }) => {
                self.orchestration_settings.current_setup = match current {
                    0 => OrchestrationSetup::SingleModel,
                    1 => OrchestrationSetup::MultiModelRouting,
                    2 => OrchestrationSetup::EnsembleVoting,
                    3 => OrchestrationSetup::SpecializedAgents,
                    _ => OrchestrationSetup::SingleModel,
                };
            }
            ("Routing Strategy", SettingValue::Selection { current, .. }) => {
                self.orchestration_settings.preferred_strategy = match current {
                    0 => RoutingStrategy::RoundRobin,
                    1 => RoutingStrategy::LeastLatency,
                    2 => RoutingStrategy::ContextAware,
                    3 => RoutingStrategy::CapabilityBased,
                    4 => RoutingStrategy::CostOptimized,
                    5 => RoutingStrategy::Custom("Custom".to_string()),
                    _ => RoutingStrategy::CapabilityBased,
                };
            }
            ("Local Model Preference", SettingValue::Float(v)) => self.orchestration_settings.local_models_preference = *v,
            
            // Interface
            ("Dark Theme", SettingValue::Bool(v)) => self.settings.dark_theme = *v,
            ("Word Wrap", SettingValue::Bool(v)) => self.settings.word_wrap = *v,
            
            // Performance
            ("Thread Count", SettingValue::Integer(v)) => self.settings.threads = *v as usize,
            
            // String settings
            ("API Endpoint", SettingValue::String(v)) => {
                self.settings.api_endpoint = if v == "default" { None } else { Some(v.clone()) };
            }
            ("Model Name", SettingValue::String(v)) => {
                self.settings.default_model = if v == "auto" { None } else { Some(v.clone()) };
            }
            
            _ => {}
        }
    }
}

impl SubtabController for SettingsTab {
    fn render(&mut self, f: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),   // Title
                Constraint::Min(10),     // Main content
                Constraint::Length(4),   // Help/status
            ])
            .split(area);
        
        // Apply theme
        let bg_style = Style::default().bg(self.theme.background);
        
        // Title with save progress
        let title_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Min(20),
                Constraint::Length(30),
            ])
            .split(chunks[0]);
            
        let title_text = format!(
            "⚙️  Settings - {} {}",
            self.current_category.name(),
            if self.has_changes { "(Modified)" } else { "" }
        );
        let title = Paragraph::new(title_text)
            .style(Style::default().fg(self.theme.primary).add_modifier(Modifier::BOLD))
            .alignment(Alignment::Center);
        f.render_widget(title, title_chunks[0]);
        
        // Save progress indicator
        if self.has_changes {
            self.save_progress.render(f, title_chunks[1], "Save Progress");
        }
        
        // Main content area - split into categories and settings
        let content_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Length(25),  // Category list
                Constraint::Min(50),     // Settings list
            ])
            .split(chunks[1]);
        
        // Category list
        let categories: Vec<ListItem> = SettingCategory::all()
            .into_iter()
            .map(|cat| {
                let style = if cat == self.current_category {
                    Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
                } else {
                    Style::default()
                };
                ListItem::new(cat.name().to_string()).style(style)
            })
            .collect();
        
        let category_list = List::new(categories)
            .block(Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(self.theme.border))
                .title(" Categories (PgUp/PgDn to switch) "))
            .style(bg_style);
        f.render_widget(category_list, content_chunks[0]);
        
        // Settings list
        let category_items = self.get_category_items();
        let items: Vec<ListItem> = category_items
            .iter()
            .enumerate()
            .map(|(i, item)| {
                let selected = i == self.selected_index;
                let style = if selected {
                    Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
                } else {
                    Style::default()
                };
                
                let value_style = if selected && self.edit_mode {
                    Style::default().fg(Color::Green)
                } else {
                    Style::default().fg(Color::Cyan)
                };
                
                let content = vec![
                    Line::from(vec![
                        Span::styled(item.name.clone(), style),
                        Span::raw(": "),
                        Span::styled(
                            if selected && self.edit_mode {
                                self.edit_buffer.clone()
                            } else {
                                item.value.to_string()
                            },
                            value_style
                        ),
                    ]),
                    Line::from(vec![
                        Span::raw("  "),
                        Span::styled(item.description.clone(), Style::default().fg(Color::DarkGray)),
                    ]),
                ];
                
                ListItem::new(content)
            })
            .collect();
        
        let settings_list = List::new(items)
            .block(Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .title(format!(" {} Settings ", self.current_category.name())));
        f.render_widget(settings_list, content_chunks[1]);
        
        // Help/status bar
        let help_text = if self.edit_mode {
            vec![
                Line::from("Editing mode - Enter to confirm, Esc to cancel"),
            ]
        } else {
            vec![
                Line::from("↑/↓ Navigate Items | PgUp/PgDn Switch Category | Space/Enter Toggle/Edit | ←/→ Adjust Values | s Save | r Reset"),
                Line::from("For bool: Space to toggle | For numbers: ←/→ to adjust or Enter to edit | For strings: Enter to edit"),
            ]
        };
        
        let help_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Min(20),
                Constraint::Length(40),
            ])
            .split(chunks[2]);
            
        let help = Paragraph::new(help_text)
            .block(Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(self.theme.border))
                .title(" Help "))
            .style(bg_style);
        f.render_widget(help, help_chunks[0]);
        
        // Toast notifications
        self.toast_manager.update();
        self.toast_manager.render(f, help_chunks[1]);
    }
    
    fn handle_input(&mut self, key: KeyEvent) -> Result<()> {
        if self.edit_mode {
            // Handle string editing
            match key.code {
                KeyCode::Enter => {
                    // Apply the edited value
                    let category_items = self.get_category_items();
                    if let Some(item) = category_items.get(self.selected_index) {
                        let item_name = item.name.clone();
                        if let Some(mut_item) = self.items.iter_mut().find(|i| i.name == item_name) {
                            match mut_item.value.set_string(self.edit_buffer.clone()) {
                                Ok(()) => {
                                    let item_clone = mut_item.clone();
                                    self.update_setting_value(&item_clone);
                                    self.has_changes = true;
                                    self.toast_manager.add_toast(
                                        format!("Updated {}", item_name),
                                        ToastType::Success
                                    );
                                }
                                Err(e) => {
                                    self.toast_manager.add_toast(
                                        format!("Error: {}", e),
                                        ToastType::Error
                                    );
                                }
                            }
                        }
                    }
                    self.edit_mode = false;
                    self.edit_buffer.clear();
                }
                KeyCode::Esc => {
                    self.edit_mode = false;
                    self.edit_buffer.clear();
                    self.toast_manager.add_toast(
                        "Edit cancelled".to_string(),
                        ToastType::Info
                    );
                }
                KeyCode::Char(c) => {
                    // Allow certain characters based on value type
                    let category_items = self.get_category_items();
                    if let Some(item) = category_items.get(self.selected_index) {
                        match &item.value {
                            SettingValue::Float(_) => {
                                // Only allow digits and decimal point
                                if c.is_numeric() || (c == '.' && !self.edit_buffer.contains('.')) {
                                    self.edit_buffer.push(c);
                                }
                            }
                            SettingValue::Integer(_) => {
                                // Only allow digits
                                if c.is_numeric() {
                                    self.edit_buffer.push(c);
                                }
                            }
                            SettingValue::String(_) => {
                                // Allow most characters for strings
                                self.edit_buffer.push(c);
                            }
                            _ => {}
                        }
                    }
                }
                KeyCode::Backspace => {
                    self.edit_buffer.pop();
                }
                _ => {}
            }
        } else {
            match key.code {
                // Navigation
                KeyCode::Up => {
                    if self.selected_index > 0 {
                        self.selected_index -= 1;
                    }
                }
                KeyCode::Down => {
                    let max_index = self.get_category_items().len().saturating_sub(1);
                    if self.selected_index < max_index {
                        self.selected_index += 1;
                    }
                }
                
                // Left arrow: decrement numeric values
                KeyCode::Left => {
                    let category_items = self.get_category_items();
                    if let Some(item) = category_items.get(self.selected_index) {
                        let item_name = item.name.clone();
                        if let Some(mut_item) = self.items.iter_mut().find(|i| i.name == item_name) {
                            if mut_item.value.can_decrement() {
                                mut_item.value.decrement();
                                let item_clone = mut_item.clone();
                                self.update_setting_value(&item_clone);
                                self.has_changes = true;
                                self.toast_manager.add_toast(
                                    format!("Decreased {}", item_name),
                                    ToastType::Info
                                );
                            }
                        }
                    }
                }
                
                // Right arrow: increment numeric values
                KeyCode::Right => {
                    let category_items = self.get_category_items();
                    if let Some(item) = category_items.get(self.selected_index) {
                        let item_name = item.name.clone();
                        if let Some(mut_item) = self.items.iter_mut().find(|i| i.name == item_name) {
                            if mut_item.value.can_increment() {
                                mut_item.value.increment();
                                let item_clone = mut_item.clone();
                                self.update_setting_value(&item_clone);
                                self.has_changes = true;
                                self.toast_manager.add_toast(
                                    format!("Increased {}", item_name),
                                    ToastType::Info
                                );
                            }
                        }
                    }
                }
                
                // Category navigation with PageUp/PageDown
                KeyCode::PageUp => {
                    // Switch to previous category
                    let categories = SettingCategory::all();
                    let current_idx = categories.iter().position(|c| *c == self.current_category).unwrap_or(0);
                    let prev_idx = if current_idx > 0 { current_idx - 1 } else { categories.len() - 1 };
                    self.current_category = categories[prev_idx];
                    self.selected_index = 0;
                    self.toast_manager.add_toast(
                        format!("Category: {}", self.current_category.name()),
                        ToastType::Info
                    );
                }
                
                KeyCode::PageDown => {
                    // Switch to next category
                    let categories = SettingCategory::all();
                    let current_idx = categories.iter().position(|c| *c == self.current_category).unwrap_or(0);
                    let next_idx = (current_idx + 1) % categories.len();
                    self.current_category = categories[next_idx];
                    self.selected_index = 0;
                    self.toast_manager.add_toast(
                        format!("Category: {}", self.current_category.name()),
                        ToastType::Info
                    );
                }
                
                // Category navigation with brackets (alternative)
                KeyCode::Char('[') => {
                    // Switch to previous category
                    let categories = SettingCategory::all();
                    let current_idx = categories.iter().position(|c| *c == self.current_category).unwrap_or(0);
                    let prev_idx = if current_idx > 0 { current_idx - 1 } else { categories.len() - 1 };
                    self.current_category = categories[prev_idx];
                    self.selected_index = 0;
                    self.toast_manager.add_toast(
                        format!("Category: {}", self.current_category.name()),
                        ToastType::Info
                    );
                }
                
                KeyCode::Char(']') => {
                    // Switch to next category
                    let categories = SettingCategory::all();
                    let current_idx = categories.iter().position(|c| *c == self.current_category).unwrap_or(0);
                    let next_idx = (current_idx + 1) % categories.len();
                    self.current_category = categories[next_idx];
                    self.selected_index = 0;
                    self.toast_manager.add_toast(
                        format!("Category: {}", self.current_category.name()),
                        ToastType::Info
                    );
                }
                
                // Value modification with Space/Enter
                KeyCode::Char(' ') | KeyCode::Enter => {
                    let category_items = self.get_category_items();
                    if let Some(item) = category_items.get(self.selected_index) {
                        let item_name = item.name.clone();
                        if let Some(mut_item) = self.items.iter_mut().find(|i| i.name == item_name) {
                            // Check if this value can be edited as string
                            if Self::can_edit_as_string(&mut_item.value) {
                                // Enter edit mode for string/number values
                                self.edit_mode = true;
                                self.edit_buffer = mut_item.value.to_string();
                            } else {
                                // Toggle for boolean and selection values
                                mut_item.value.toggle();
                                let item_clone = mut_item.clone();
                                self.update_setting_value(&item_clone);
                                self.has_changes = true;
                            }
                        }
                    }
                }
                
                // Actions
                KeyCode::Char('s') => {
                    self.save_progress.set_progress(0.0);
                    self.toast_manager.add_toast("Saving settings...".to_string(), ToastType::Info);
                    self.save_settings();
                    self.save_progress.set_progress(1.0);
                    self.toast_manager.add_toast("Settings saved successfully!".to_string(), ToastType::Success);
                    self.has_changes = false;
                }
                
                KeyCode::Char('r') => {
                    self.toast_manager.add_toast("Resetting to saved settings...".to_string(), ToastType::Warning);
                    self.load_settings();
                    self.has_changes = false;
                    self.toast_manager.add_toast("Settings reset".to_string(), ToastType::Info);
                }
                
                KeyCode::Char('t') => {
                    // Theme toggle
                    self.theme = match self.settings.dark_theme {
                        true => Theme::light(),
                        false => Theme::dark(),
                    };
                    self.settings.dark_theme = !self.settings.dark_theme;
                    self.has_changes = true;
                    self.toast_manager.add_toast(
                        format!("Theme changed to {}", if self.settings.dark_theme { "Dark" } else { "Light" }),
                        ToastType::Info
                    );
                }
                
                _ => {}
            }
        }
        
        Ok(())
    }
    
    fn update(&mut self) -> Result<()> {
        // Auto-save if configured
        if self.has_changes && self.settings.auto_save {
            self.save_settings();
        }
        Ok(())
    }
    
    fn name(&self) -> &str {
        "Settings"
    }
}