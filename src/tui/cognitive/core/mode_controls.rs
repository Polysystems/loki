//! Cognitive Mode Controls for Chat UI
//!
//! This module provides UI controls for toggling and configuring different
//! cognitive processing modes in the chat interface.

use std::sync::Arc;
use anyhow::Result;
use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph, Gauge},
    Frame,
};
use tokio::sync::RwLock;
use tracing::{info};
use serde::{Serialize, Deserialize};

use crate::tui::{
    chat::integrations::cognitive::CognitiveChatEnhancement,
    cognitive_stream_integration::CognitiveMode,
};

/// Cognitive processing modes
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum CognitiveProcessingMode {
    /// Minimal processing - fast responses
    Minimal,
    /// Standard processing - balanced
    Standard,
    /// Deep processing - thorough analysis
    Deep,
    /// Creative mode - emphasis on novelty
    Creative,
    /// Analytical mode - emphasis on logic
    Analytical,
    /// Empathetic mode - emphasis on emotional understanding
    Empathetic,
    /// Research mode - autonomous information gathering
    Research,
    /// Dream mode - free association
    Dream,
}

impl CognitiveProcessingMode {
    /// Get display name
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Minimal => "Minimal",
            Self::Standard => "Standard",
            Self::Deep => "Deep",
            Self::Creative => "Creative",
            Self::Analytical => "Analytical",
            Self::Empathetic => "Empathetic",
            Self::Research => "Research",
            Self::Dream => "Dream",
        }
    }
    
    /// Get description
    pub fn description(&self) -> &'static str {
        match self {
            Self::Minimal => "Fast responses with basic processing",
            Self::Standard => "Balanced speed and depth",
            Self::Deep => "Thorough analysis and reasoning",
            Self::Creative => "Emphasis on novel ideas and connections",
            Self::Analytical => "Logical analysis and structured thinking",
            Self::Empathetic => "Focus on emotional understanding",
            Self::Research => "Autonomous information gathering",
            Self::Dream => "Free association and exploration",
        }
    }
    
    /// Get icon
    pub fn icon(&self) -> &'static str {
        match self {
            Self::Minimal => "⚡",
            Self::Standard => "🧠",
            Self::Deep => "🌊",
            Self::Creative => "🎨",
            Self::Analytical => "📊",
            Self::Empathetic => "💝",
            Self::Research => "🔬",
            Self::Dream => "🌙",
        }
    }
    
    /// Get color
    pub fn color(&self) -> Color {
        match self {
            Self::Minimal => Color::Green,
            Self::Standard => Color::Cyan,
            Self::Deep => Color::Blue,
            Self::Creative => Color::Magenta,
            Self::Analytical => Color::Yellow,
            Self::Empathetic => Color::LightRed,
            Self::Research => Color::LightBlue,
            Self::Dream => Color::LightMagenta,
        }
    }
    
    /// Convert to consciousness mode
    pub fn to_consciousness_mode(&self) -> CognitiveMode {
        match self {
            Self::Minimal => CognitiveMode::Minimal,
            Self::Deep => CognitiveMode::Deep,
            Self::Dream => CognitiveMode::Continuous,
            _ => CognitiveMode::Standard,
        }
    }
    
    /// Get cognitive depth
    pub fn cognitive_depth(&self) -> f64 {
        match self {
            Self::Minimal => 0.2,
            Self::Standard => 0.5,
            Self::Deep => 0.8,
            Self::Creative => 0.7,
            Self::Analytical => 0.7,
            Self::Empathetic => 0.6,
            Self::Research => 0.9,
            Self::Dream => 1.0,
        }
    }
}

/// Cognitive mode settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveModeSettings {
    /// Current mode
    pub mode: CognitiveProcessingMode,
    
    /// Enable background processing
    pub background_enabled: bool,
    
    /// Enable consciousness stream
    pub consciousness_enabled: bool,
    
    /// Enable memory learning
    pub learning_enabled: bool,
    
    /// Enable session persistence
    pub persistence_enabled: bool,
    
    /// Custom depth override (0.0-1.0)
    pub custom_depth: Option<f64>,
    
    /// Response time preference (ms)
    pub target_response_time: u64,
}

impl Default for CognitiveModeSettings {
    fn default() -> Self {
        Self {
            mode: CognitiveProcessingMode::Standard,
            background_enabled: true,
            consciousness_enabled: true,
            learning_enabled: true,
            persistence_enabled: true,
            custom_depth: None,
            target_response_time: 2000, // 2 seconds
        }
    }
}

/// Cognitive mode controller
pub struct CognitiveModeController {
    /// Current settings
    settings: Arc<RwLock<CognitiveModeSettings>>,
    
    /// Enhancement reference
    enhancement: Option<Arc<CognitiveChatEnhancement>>,
    
    /// Mode presets
    presets: Vec<ModePreset>,
}

/// Mode preset configuration
#[derive(Debug, Clone)]
pub struct ModePreset {
    pub name: String,
    pub mode: CognitiveProcessingMode,
    pub settings: CognitiveModeSettings,
}

impl CognitiveModeController {
    /// Create new controller
    pub fn new() -> Self {
        let presets = vec![
            ModePreset {
                name: "Quick Chat".to_string(),
                mode: CognitiveProcessingMode::Minimal,
                settings: CognitiveModeSettings {
                    mode: CognitiveProcessingMode::Minimal,
                    background_enabled: false,
                    consciousness_enabled: false,
                    learning_enabled: false,
                    persistence_enabled: false,
                    custom_depth: Some(0.1),
                    target_response_time: 500,
                },
            },
            ModePreset {
                name: "Balanced Assistant".to_string(),
                mode: CognitiveProcessingMode::Standard,
                settings: CognitiveModeSettings::default(),
            },
            ModePreset {
                name: "Deep Thinker".to_string(),
                mode: CognitiveProcessingMode::Deep,
                settings: CognitiveModeSettings {
                    mode: CognitiveProcessingMode::Deep,
                    background_enabled: true,
                    consciousness_enabled: true,
                    learning_enabled: true,
                    persistence_enabled: true,
                    custom_depth: Some(0.9),
                    target_response_time: 5000,
                },
            },
            ModePreset {
                name: "Creative Partner".to_string(),
                mode: CognitiveProcessingMode::Creative,
                settings: CognitiveModeSettings {
                    mode: CognitiveProcessingMode::Creative,
                    background_enabled: true,
                    consciousness_enabled: true,
                    learning_enabled: true,
                    persistence_enabled: true,
                    custom_depth: Some(0.8),
                    target_response_time: 3000,
                },
            },
        ];
        
        Self {
            settings: Arc::new(RwLock::new(CognitiveModeSettings::default())),
            enhancement: None,
            presets,
        }
    }
    
    /// Set enhancement reference
    pub fn set_enhancement(&mut self, enhancement: Arc<CognitiveChatEnhancement>) {
        self.enhancement = Some(enhancement);
    }
    
    /// Get current settings
    pub async fn get_settings(&self) -> CognitiveModeSettings {
        self.settings.read().await.clone()
    }
    
    /// Set mode
    pub async fn set_mode(&self, mode: CognitiveProcessingMode) -> Result<()> {
        info!("Setting cognitive mode to: {:?}", mode);
        
        let mut settings = self.settings.write().await;
        settings.mode = mode;
        
        // Apply to enhancement
        if let Some(enhancement) = &self.enhancement {
            let mode_str = format!("{}", mode.to_consciousness_mode());
            enhancement.set_cognitive_mode(&mode_str).await;
        }
        
        Ok(())
    }
    
    /// Apply preset
    pub async fn apply_preset(&self, preset_name: &str) -> Result<()> {
        if let Some(preset) = self.presets.iter().find(|p| p.name == preset_name) {
            *self.settings.write().await = preset.settings.clone();
            
            // Apply to enhancement
            if let Some(enhancement) = &self.enhancement {
                let mode_str = format!("{}", preset.mode.to_consciousness_mode());
                enhancement.set_cognitive_mode(&mode_str).await;
                
                // Toggle features
                if preset.settings.background_enabled != enhancement.deep_processing_enabled {
                    enhancement.toggle_deep_processing().await;
                }
            }
            
            info!("Applied preset: {}", preset_name);
            Ok(())
        } else {
            Err(anyhow::anyhow!("Preset not found: {}", preset_name))
        }
    }
    
    /// Toggle feature
    pub async fn toggle_feature(&self, feature: CognitiveFeature) -> Result<bool> {
        let mut settings = self.settings.write().await;
        
        let new_state = match feature {
            CognitiveFeature::Background => {
                settings.background_enabled = !settings.background_enabled;
                settings.background_enabled
            }
            CognitiveFeature::Consciousness => {
                settings.consciousness_enabled = !settings.consciousness_enabled;
                settings.consciousness_enabled
            }
            CognitiveFeature::Learning => {
                settings.learning_enabled = !settings.learning_enabled;
                settings.learning_enabled
            }
            CognitiveFeature::Persistence => {
                settings.persistence_enabled = !settings.persistence_enabled;
                settings.persistence_enabled
            }
        };
        
        info!("Toggled {:?} to: {}", feature, new_state);
        Ok(new_state)
    }
    
    /// Set custom depth
    pub async fn set_custom_depth(&self, depth: f64) -> Result<()> {
        let depth = depth.clamp(0.0, 1.0);
        self.settings.write().await.custom_depth = Some(depth);
        
        info!("Set custom cognitive depth: {:.2}", depth);
        Ok(())
    }
    
    /// Get effective depth
    pub async fn get_effective_depth(&self) -> f64 {
        let settings = self.settings.read().await;
        settings.custom_depth.unwrap_or_else(|| settings.mode.cognitive_depth())
    }
}

/// Cognitive features that can be toggled
#[derive(Debug, Clone, Copy)]
pub enum CognitiveFeature {
    Background,
    Consciousness,
    Learning,
    Persistence,
}

/// UI component for mode selection
pub struct CognitiveModeSelector {
    controller: Arc<CognitiveModeController>,
    selected_index: usize,
    show_details: bool,
}

impl CognitiveModeSelector {
    pub fn new(controller: Arc<CognitiveModeController>) -> Self {
        Self {
            controller,
            selected_index: 0,
            show_details: false,
        }
    }
    
    /// Handle key input
    pub async fn handle_key(&mut self, key: char) -> Result<bool> {
        match key {
            '\t' => {
                self.next_mode();
                Ok(true)
            }
            ' ' => {
                self.apply_selected().await?;
                Ok(true)
            }
            'd' => {
                self.show_details = !self.show_details;
                Ok(true)
            }
            _ => Ok(false),
        }
    }
    
    /// Next mode
    fn next_mode(&mut self) {
        self.selected_index = (self.selected_index + 1) % 8;
    }
    
    /// Apply selected mode
    async fn apply_selected(&self) -> Result<()> {
        let mode = match self.selected_index {
            0 => CognitiveProcessingMode::Minimal,
            1 => CognitiveProcessingMode::Standard,
            2 => CognitiveProcessingMode::Deep,
            3 => CognitiveProcessingMode::Creative,
            4 => CognitiveProcessingMode::Analytical,
            5 => CognitiveProcessingMode::Empathetic,
            6 => CognitiveProcessingMode::Research,
            _ => CognitiveProcessingMode::Dream,
        };
        
        self.controller.set_mode(mode).await
    }
    
    /// Render selector
    pub async fn render(&self, f: &mut Frame<'_>, area: Rect) {
        let settings = self.controller.get_settings().await;
        
        if self.show_details {
            self.render_detailed(f, area, &settings).await;
        } else {
            self.render_compact(f, area, &settings);
        }
    }
    
    /// Render compact view
    fn render_compact(&self, f: &mut Frame, area: Rect, settings: &CognitiveModeSettings) {
        let mode = settings.mode;
        let depth = settings.custom_depth.unwrap_or_else(|| mode.cognitive_depth());
        
        let text = vec![
            Line::from(vec![
                Span::styled(
                    format!("{} ", mode.icon()),
                    Style::default().fg(mode.color())
                ),
                Span::styled(
                    "Cognitive Mode: ",
                    Style::default().fg(Color::Gray)
                ),
                Span::styled(
                    mode.display_name(),
                    Style::default()
                        .fg(mode.color())
                        .add_modifier(Modifier::BOLD)
                ),
                Span::raw(" | "),
                Span::styled(
                    format!("Depth: {:.0}%", depth * 100.0),
                    Style::default().fg(Color::Cyan)
                ),
            ]),
        ];
        
        let paragraph = Paragraph::new(text)
            .alignment(Alignment::Center);
        
        f.render_widget(paragraph, area);
    }
    
    /// Render detailed view
    async fn render_detailed(&self, f: &mut Frame<'_>, area: Rect, settings: &CognitiveModeSettings) {
        let block = Block::default()
            .borders(Borders::ALL)
            .title(" 🧠 Cognitive Mode Settings ")
            .style(Style::default().fg(Color::Cyan));
        
        let inner = block.inner(area);
        f.render_widget(block, area);
        
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),  // Mode selector
                Constraint::Length(3),  // Depth gauge
                Constraint::Length(6),  // Features
                Constraint::Min(3),     // Description
            ])
            .split(inner);
        
        // Mode selector
        self.render_mode_list(f, chunks[0], settings);
        
        // Depth gauge
        self.render_depth_gauge(f, chunks[1], settings).await;
        
        // Feature toggles
        self.render_features(f, chunks[2], settings);
        
        // Description
        self.render_description(f, chunks[3], settings);
    }
    
    fn render_mode_list(&self, f: &mut Frame, area: Rect, settings: &CognitiveModeSettings) {
        let modes = vec![
            CognitiveProcessingMode::Minimal,
            CognitiveProcessingMode::Standard,
            CognitiveProcessingMode::Deep,
            CognitiveProcessingMode::Creative,
            CognitiveProcessingMode::Analytical,
            CognitiveProcessingMode::Empathetic,
            CognitiveProcessingMode::Research,
            CognitiveProcessingMode::Dream,
        ];
        
        let items: Vec<ListItem> = modes.iter().enumerate()
            .map(|(i, mode)| {
                let selected = i == self.selected_index;
                let active = *mode == settings.mode;
                
                let style = if selected && active {
                    Style::default().fg(mode.color()).add_modifier(Modifier::BOLD | Modifier::UNDERLINED)
                } else if selected {
                    Style::default().fg(mode.color()).add_modifier(Modifier::UNDERLINED)
                } else if active {
                    Style::default().fg(mode.color()).add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(Color::Gray)
                };
                
                ListItem::new(format!("{} {}", mode.icon(), mode.display_name()))
                    .style(style)
            })
            .collect();
        
        let list = List::new(items)
            .direction(ratatui::widgets::ListDirection::TopToBottom);
        
        f.render_widget(list, area);
    }
    
    async fn render_depth_gauge(&self, f: &mut Frame<'_>, area: Rect, settings: &CognitiveModeSettings) {
        let depth = self.controller.get_effective_depth().await;
        
        let gauge = Gauge::default()
            .block(Block::default().title("Cognitive Depth"))
            .gauge_style(Style::default().fg(settings.mode.color()))
            .percent((depth * 100.0) as u16)
            .label(format!("{:.0}%", depth * 100.0));
        
        f.render_widget(gauge, area);
    }
    
    fn render_features(&self, f: &mut Frame, area: Rect, settings: &CognitiveModeSettings) {
        let features = vec![
            ("Background Processing", settings.background_enabled, "🌙"),
            ("Consciousness Stream", settings.consciousness_enabled, "🌊"),
            ("Memory Learning", settings.learning_enabled, "🧠"),
            ("Session Persistence", settings.persistence_enabled, "💾"),
        ];
        
        let items: Vec<ListItem> = features.iter()
            .map(|(name, enabled, icon)| {
                let checkbox = if *enabled { "☑" } else { "☐" };
                let color = if *enabled { Color::Green } else { Color::Gray };
                
                ListItem::new(format!("{} {} {}", checkbox, icon, name))
                    .style(Style::default().fg(color))
            })
            .collect();
        
        let list = List::new(items)
            .block(Block::default().title("Features"));
        
        f.render_widget(list, area);
    }
    
    fn render_description(&self, f: &mut Frame, area: Rect, settings: &CognitiveModeSettings) {
        let description = Paragraph::new(settings.mode.description())
            .block(Block::default().title("Description"))
            .style(Style::default().fg(Color::White))
            .wrap(ratatui::widgets::Wrap { trim: true });
        
        f.render_widget(description, area);
    }
}

/// Create mode selector line for status bar
pub fn create_mode_status_line(settings: &CognitiveModeSettings) -> Line<'static> {
    let mode = settings.mode;
    let depth = settings.custom_depth.unwrap_or_else(|| mode.cognitive_depth());
    
    Line::from(vec![
        Span::styled(
            format!("{} ", mode.icon()),
            Style::default().fg(mode.color())
        ),
        Span::raw("Mode: "),
        Span::styled(
            mode.display_name(),
            Style::default().fg(mode.color()).add_modifier(Modifier::BOLD)
        ),
        Span::raw(" ["),
        Span::styled(
            format!("{:.0}%", depth * 100.0),
            Style::default().fg(Color::Cyan)
        ),
        Span::raw("]")
    ])
}