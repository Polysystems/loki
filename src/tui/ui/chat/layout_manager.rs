//! Dynamic layout management for multi-panel chat interface
//! 
//! Provides flexible panel arrangements, resizing, and responsive layouts
//! for maximizing terminal real estate.

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    symbols,
    widgets::{Block, Borders, BorderType},
    Frame,
};
use std::collections::HashMap;

/// Layout configuration for chat panels
#[derive(Debug, Clone)]
pub struct LayoutConfig {
    /// Main layout direction
    pub direction: LayoutDirection,
    
    /// Panel size constraints
    pub panel_constraints: HashMap<PanelType, PanelConstraint>,
    
    /// Minimum panel sizes
    pub min_sizes: HashMap<PanelType, (u16, u16)>, // (width, height)
    
    /// Panel borders configuration
    pub borders: BorderConfig,
    
    /// Responsive breakpoints
    pub breakpoints: Vec<ResponsiveBreakpoint>,
}

/// Layout direction preference
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LayoutDirection {
    Horizontal,
    Vertical,
    Grid,
    Floating,
}

/// Panel size constraint
#[derive(Debug, Clone)]
pub enum PanelConstraint {
    Fixed(u16),
    Percentage(u16),
    Min(u16),
    Max(u16),
    Ratio(u16, u16), // numerator, denominator
}

/// Border configuration
#[derive(Debug, Clone)]
pub struct BorderConfig {
    pub style: BorderStyle,
    pub color: Color,
    pub highlight_color: Color,
    pub rounded: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum BorderStyle {
    Single,
    Double,
    Thick,
    None,
}

/// Responsive breakpoint
#[derive(Debug, Clone)]
pub struct ResponsiveBreakpoint {
    pub width: u16,
    pub height: u16,
    pub layout: LayoutPreset,
}

/// Predefined layout presets
#[derive(Debug, Clone, Copy)]
pub enum LayoutPreset {
    SinglePanel,
    SideBySide,
    ThreeColumn,
    MainWithSidebar,
    QuadGrid,
    FullscreenWithOverlay,
}

/// Chat layout result
#[derive(Debug, Clone)]
pub struct ChatLayout {
    /// Panel areas
    pub panels: HashMap<PanelType, Rect>,
    
    /// Active layout preset
    pub preset: LayoutPreset,
    
    /// Available space
    pub total_area: Rect,
}

/// Panel types
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PanelType {
    Chat,
    ToolOutput,
    Context,
    Workflow,
    Preview,
    StatusBar,
    InputArea,
    /// Dynamic agent stream panel
    AgentStream { agent_id: String },
    /// Overview of all active agents
    AgentOverview,
}

impl Default for LayoutConfig {
    fn default() -> Self {
        let mut panel_constraints = HashMap::new();
        panel_constraints.insert(PanelType::Chat, PanelConstraint::Percentage(60));
        panel_constraints.insert(PanelType::ToolOutput, PanelConstraint::Percentage(40));
        panel_constraints.insert(PanelType::Context, PanelConstraint::Fixed(30));
        panel_constraints.insert(PanelType::Workflow, PanelConstraint::Min(20));
        panel_constraints.insert(PanelType::Preview, PanelConstraint::Percentage(50));
        panel_constraints.insert(PanelType::StatusBar, PanelConstraint::Fixed(3));
        panel_constraints.insert(PanelType::InputArea, PanelConstraint::Fixed(3));
        panel_constraints.insert(PanelType::AgentOverview, PanelConstraint::Percentage(30));
        
        let mut min_sizes = HashMap::new();
        min_sizes.insert(PanelType::Chat, (40, 10));
        min_sizes.insert(PanelType::ToolOutput, (30, 8));
        min_sizes.insert(PanelType::Context, (25, 5));
        min_sizes.insert(PanelType::Workflow, (30, 10));
        min_sizes.insert(PanelType::Preview, (40, 10));
        min_sizes.insert(PanelType::StatusBar, (0, 1));
        min_sizes.insert(PanelType::InputArea, (0, 3));
        min_sizes.insert(PanelType::AgentOverview, (35, 8));
        
        Self {
            direction: LayoutDirection::Horizontal,
            panel_constraints,
            min_sizes,
            borders: BorderConfig {
                style: BorderStyle::Single,
                color: Color::Rgb(64, 64, 64),
                highlight_color: Color::Cyan,
                rounded: true,
            },
            breakpoints: vec![
                ResponsiveBreakpoint {
                    width: 180,
                    height: 50,
                    layout: LayoutPreset::ThreeColumn,
                },
                ResponsiveBreakpoint {
                    width: 120,
                    height: 40,
                    layout: LayoutPreset::SideBySide,
                },
                ResponsiveBreakpoint {
                    width: 80,
                    height: 30,
                    layout: LayoutPreset::MainWithSidebar,
                },
                ResponsiveBreakpoint {
                    width: 0,
                    height: 0,
                    layout: LayoutPreset::SinglePanel,
                },
            ],
        }
    }
}

/// Advanced layout manager
#[derive(Clone)]
pub struct LayoutManager {
    /// Current configuration
    config: LayoutConfig,
    
    /// Current preset
    pub preset: LayoutPreset,
    
    /// Panel visibility
    visibility: HashMap<PanelType, bool>,
    
    /// Panel focus state
    focused_panel: Option<PanelType>,
    
    /// Layout cache
    pub cache: Option<ChatLayout>,
    
    /// Resize handles
    resize_handles: Vec<ResizeHandle>,
}

/// Resize handle for panel boundaries
#[derive(Debug, Clone)]
struct ResizeHandle {
    pub position: (u16, u16),
    pub direction: Direction,
    pub panels: (PanelType, PanelType),
}

impl LayoutManager {
    pub fn new() -> Self {
        let mut visibility = HashMap::new();
        visibility.insert(PanelType::Chat, true);
        visibility.insert(PanelType::ToolOutput, true);  // Enable tool output panel
        visibility.insert(PanelType::Context, true); // Start with Context panel visible
        visibility.insert(PanelType::Workflow, false);
        visibility.insert(PanelType::Preview, false);
        visibility.insert(PanelType::StatusBar, true);
        visibility.insert(PanelType::InputArea, true);
        visibility.insert(PanelType::AgentOverview, false);
        
        Self {
            config: LayoutConfig::default(),
            preset: LayoutPreset::MainWithSidebar,  // Start with multi-panel layout
            visibility,
            focused_panel: Some(PanelType::Chat),
            cache: None,
            resize_handles: Vec::new(),
        }
    }
    
    /// Calculate layout for given area
    pub fn calculate_layout(&mut self, area: Rect) -> ChatLayout {
        // Check cache
        if let Some(cached) = &self.cache {
            if cached.total_area == area {
                return cached.clone();
            }
        }
        
        // Determine layout preset based on breakpoints
        let preset = self.determine_preset(area);
        
        // Calculate panel areas based on preset
        let panels = match preset {
            LayoutPreset::SinglePanel => self.layout_single_panel(area),
            LayoutPreset::SideBySide => self.layout_side_by_side(area),
            LayoutPreset::ThreeColumn => self.layout_three_column(area),
            LayoutPreset::MainWithSidebar => self.layout_main_with_sidebar(area),
            LayoutPreset::QuadGrid => self.layout_quad_grid(area),
            LayoutPreset::FullscreenWithOverlay => self.layout_fullscreen_overlay(area),
        };
        
        let layout = ChatLayout {
            panels,
            preset,
            total_area: area,
        };
        
        // Update cache
        self.cache = Some(layout.clone());
        
        layout
    }
    
    /// Determine layout preset based on terminal size
    fn determine_preset(&self, area: Rect) -> LayoutPreset {
        for breakpoint in &self.config.breakpoints {
            if area.width >= breakpoint.width && area.height >= breakpoint.height {
                return breakpoint.layout;
            }
        }
        LayoutPreset::SinglePanel
    }
    
    /// Single panel layout
    fn layout_single_panel(&self, area: Rect) -> HashMap<PanelType, Rect> {
        let mut panels = HashMap::new();
        
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1), // Status bar
                Constraint::Min(0),    // Main content
                Constraint::Length(3), // Input area
            ])
            .split(area);
        
        if self.is_visible(PanelType::StatusBar) {
            panels.insert(PanelType::StatusBar, chunks[0]);
        }
        
        panels.insert(PanelType::Chat, chunks[1]);
        
        if self.is_visible(PanelType::InputArea) {
            panels.insert(PanelType::InputArea, chunks[2]);
        }
        
        panels
    }
    
    /// Side-by-side layout
    fn layout_side_by_side(&self, area: Rect) -> HashMap<PanelType, Rect> {
        let mut panels = HashMap::new();
        
        // Vertical split for status and content
        let v_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1), // Status bar
                Constraint::Min(0),    // Main content
                Constraint::Length(3), // Input area
            ])
            .split(area);
        
        if self.is_visible(PanelType::StatusBar) {
            panels.insert(PanelType::StatusBar, v_chunks[0]);
        }
        
        // Horizontal split for main content
        let h_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(60), // Chat
                Constraint::Percentage(40), // Tool output
            ])
            .split(v_chunks[1]);
        
        panels.insert(PanelType::Chat, h_chunks[0]);
        
        if self.is_visible(PanelType::ToolOutput) {
            panels.insert(PanelType::ToolOutput, h_chunks[1]);
        }
        
        if self.is_visible(PanelType::InputArea) {
            panels.insert(PanelType::InputArea, v_chunks[2]);
        }
        
        panels
    }
    
    /// Three column layout
    fn layout_three_column(&self, area: Rect) -> HashMap<PanelType, Rect> {
        let mut panels = HashMap::new();
        
        let v_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1),
                Constraint::Min(0),
                Constraint::Length(3),
            ])
            .split(area);
        
        if self.is_visible(PanelType::StatusBar) {
            panels.insert(PanelType::StatusBar, v_chunks[0]);
        }
        
        // Three column split
        let h_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(25), // Context
                Constraint::Percentage(50), // Chat
                Constraint::Percentage(25), // Tool output
            ])
            .split(v_chunks[1]);
        
        if self.is_visible(PanelType::Context) {
            panels.insert(PanelType::Context, h_chunks[0]);
        }
        
        panels.insert(PanelType::Chat, h_chunks[1]);
        
        if self.is_visible(PanelType::ToolOutput) {
            panels.insert(PanelType::ToolOutput, h_chunks[2]);
        }
        
        if self.is_visible(PanelType::InputArea) {
            panels.insert(PanelType::InputArea, v_chunks[2]);
        }
        
        panels
    }
    
    /// Main with sidebar layout
    fn layout_main_with_sidebar(&self, area: Rect) -> HashMap<PanelType, Rect> {
        let mut panels = HashMap::new();
        
        let v_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1),
                Constraint::Min(0),
                Constraint::Length(3),
            ])
            .split(area);
        
        if self.is_visible(PanelType::StatusBar) {
            panels.insert(PanelType::StatusBar, v_chunks[0]);
        }
        
        let h_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Min(0),    // Main
                Constraint::Length(35), // Sidebar
            ])
            .split(v_chunks[1]);
        
        panels.insert(PanelType::Chat, h_chunks[0]);
        
        // Sidebar can show context or workflow
        if self.is_visible(PanelType::Context) {
            panels.insert(PanelType::Context, h_chunks[1]);
        } else if self.is_visible(PanelType::Workflow) {
            panels.insert(PanelType::Workflow, h_chunks[1]);
        }
        
        if self.is_visible(PanelType::InputArea) {
            panels.insert(PanelType::InputArea, v_chunks[2]);
        }
        
        panels
    }
    
    /// Quad grid layout
    fn layout_quad_grid(&self, area: Rect) -> HashMap<PanelType, Rect> {
        let mut panels = HashMap::new();
        
        let v_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1),
                Constraint::Percentage(50),
                Constraint::Percentage(50),
                Constraint::Length(3),
            ])
            .split(area);
        
        if self.is_visible(PanelType::StatusBar) {
            panels.insert(PanelType::StatusBar, v_chunks[0]);
        }
        
        // Top row
        let top_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(v_chunks[1]);
        
        panels.insert(PanelType::Chat, top_chunks[0]);
        
        if self.is_visible(PanelType::ToolOutput) {
            panels.insert(PanelType::ToolOutput, top_chunks[1]);
        }
        
        // Bottom row
        let bottom_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(v_chunks[2]);
        
        if self.is_visible(PanelType::Context) {
            panels.insert(PanelType::Context, bottom_chunks[0]);
        }
        
        if self.is_visible(PanelType::Workflow) {
            panels.insert(PanelType::Workflow, bottom_chunks[1]);
        }
        
        if self.is_visible(PanelType::InputArea) {
            panels.insert(PanelType::InputArea, v_chunks[3]);
        }
        
        panels
    }
    
    /// Fullscreen with overlay layout
    fn layout_fullscreen_overlay(&self, area: Rect) -> HashMap<PanelType, Rect> {
        let mut panels = HashMap::new();
        
        // Main fullscreen panel
        panels.insert(PanelType::Chat, area);
        
        // Overlay panels (floating)
        if self.is_visible(PanelType::Preview) {
            let overlay = centered_rect(80, 80, area);
            panels.insert(PanelType::Preview, overlay);
        }
        
        // Floating status bar at top
        if self.is_visible(PanelType::StatusBar) {
            let status_area = Rect {
                x: area.x,
                y: area.y,
                width: area.width,
                height: 1,
            };
            panels.insert(PanelType::StatusBar, status_area);
        }
        
        // Floating input at bottom
        if self.is_visible(PanelType::InputArea) {
            let input_area = Rect {
                x: area.x + 2,
                y: area.y + area.height - 4,
                width: area.width - 4,
                height: 3,
            };
            panels.insert(PanelType::InputArea, input_area);
        }
        
        panels
    }
    
    /// Check if panel is visible
    pub fn is_visible(&self, panel: PanelType) -> bool {
        *self.visibility.get(&panel).unwrap_or(&false)
    }
    
    /// Toggle panel visibility
    pub fn toggle_panel(&mut self, panel: PanelType) {
        let visible = self.visibility.entry(panel).or_insert(false);
        *visible = !*visible;
        self.cache = None; // Invalidate cache
    }
    
    /// Set focused panel
    pub fn set_focus(&mut self, panel: PanelType) {
        self.focused_panel = Some(panel);
    }
    
    /// Cycle focus through visible panels
    pub fn cycle_focus(&mut self) {
        let focusable_panels = vec![
            PanelType::Chat,
            PanelType::Context,
            PanelType::ToolOutput,
            PanelType::Workflow,
            PanelType::Preview,
        ];
        
        // Get list of visible panels
        let visible_panels: Vec<PanelType> = focusable_panels
            .into_iter()
            .filter(|p| self.is_visible(p.clone()))
            .collect();
        
        if visible_panels.is_empty() {
            return;
        }
        
        // Find current focus index
        let current_idx = self.focused_panel.clone()
            .and_then(|p| visible_panels.iter().position(|vp| *vp == p))
            .unwrap_or(0);
        
        // Move to next panel
        let next_idx = (current_idx + 1) % visible_panels.len();
        self.focused_panel = Some(visible_panels[next_idx].clone());
    }
    
    /// Get focused panel
    pub fn get_focus(&self) -> Option<PanelType> {
        self.focused_panel.clone()
    }
    
    /// Create a styled block for a panel
    pub fn create_panel_block(&self, panel: PanelType) -> Block<'static> {
        let is_focused = self.focused_panel == Some(panel.clone());
        
        let title = match &panel {
            PanelType::Chat => " 💬 Chat ".to_string(),
            PanelType::Context => " 🧠 Context & Reasoning ".to_string(),
            PanelType::ToolOutput => " 🔧 Tool Output ".to_string(),
            PanelType::Workflow => " 🔄 Workflow ".to_string(),
            PanelType::Preview => " 👁️ Preview ".to_string(),
            PanelType::StatusBar => " 📊 Status ".to_string(),
            PanelType::InputArea => " ⌨️ Input ".to_string(),
            PanelType::AgentStream { agent_id } => format!(" 🤖 Agent: {} ", agent_id),
            PanelType::AgentOverview => " 🤖 Agent Overview ".to_string(),
        };
        
        let border_style = if is_focused {
            Style::default().fg(self.config.borders.highlight_color).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(self.config.borders.color)
        };
        
        let mut block = Block::default()
            .title(title)
            .borders(Borders::ALL)
            .border_style(border_style);
            
        if self.config.borders.rounded {
            block = block.border_type(BorderType::Rounded);
        }
        
        block
    }
    
    /// Render panel borders with focus indication
    pub fn render_borders(&self, f: &mut Frame, layout: &ChatLayout) {
        for (panel_type, area) in &layout.panels {
            let is_focused = self.focused_panel == Some(panel_type.clone());
            let border_color = if is_focused {
                self.config.borders.highlight_color
            } else {
                self.config.borders.color
            };
            
            let border_type = match self.config.borders.style {
                BorderStyle::Single => symbols::border::PLAIN,
                BorderStyle::Double => symbols::border::DOUBLE,
                BorderStyle::Thick => symbols::border::THICK,
                BorderStyle::None => continue,
            };
            
            let mut block = Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(border_color));
            
            if self.config.borders.rounded {
                block = block.border_set(symbols::border::ROUNDED);
            } else {
                block = block.border_set(border_type);
            }
            
            // Add panel title
            let title = format!(" {} ", panel_type_name(panel_type.clone()));
            block = block.title(title);
            
            // Don't clear the area - just render the border
            // This was causing panels to clear each other's content
            f.render_widget(block, *area);
        }
    }
    
    /// Cycle through layout presets
    pub fn cycle_preset(&mut self) {
        self.preset = match self.preset {
            LayoutPreset::SinglePanel => LayoutPreset::SideBySide,
            LayoutPreset::SideBySide => LayoutPreset::ThreeColumn,
            LayoutPreset::ThreeColumn => LayoutPreset::MainWithSidebar,
            LayoutPreset::MainWithSidebar => LayoutPreset::QuadGrid,
            LayoutPreset::QuadGrid => LayoutPreset::FullscreenWithOverlay,
            LayoutPreset::FullscreenWithOverlay => LayoutPreset::SinglePanel,
        };
        self.cache = None;
    }
    
    /// Set a specific preset
    pub fn set_preset(&mut self, preset: LayoutPreset) {
        self.preset = preset;
        self.cache = None;
    }
    
    /// Save current layout configuration
    pub fn save_config(&self) -> LayoutConfig {
        self.config.clone()
    }
    
    /// Load layout configuration
    pub fn load_config(&mut self, config: LayoutConfig) {
        self.config = config;
        self.cache = None;
    }
    
    /// Allocate a new agent panel
    pub fn allocate_agent_panel(&mut self, agent_id: String) -> Option<PanelType> {
        let panel = PanelType::AgentStream { agent_id };
        
        // Add constraints for the new panel
        self.config.panel_constraints.insert(
            panel.clone(),
            PanelConstraint::Percentage(25),
        );
        
        // Add minimum size
        self.config.min_sizes.insert(
            panel.clone(),
            (30, 8),
        );
        
        // Make it visible
        self.visibility.insert(panel.clone(), true);
        
        // Invalidate cache to force recalculation
        self.cache = None;
        
        // Optimize layout for multiple panels
        self.optimize_for_agents(self.count_agent_panels());
        
        Some(panel)
    }
    
    /// Remove an agent panel
    pub fn remove_agent_panel(&mut self, agent_id: &str) {
        let panel = PanelType::AgentStream { agent_id: agent_id.to_string() };
        
        self.config.panel_constraints.remove(&panel);
        self.config.min_sizes.remove(&panel);
        self.visibility.remove(&panel);
        
        // If this was the focused panel, clear focus
        if self.focused_panel == Some(panel) {
            self.focused_panel = None;
        }
        
        self.cache = None;
        
        // Re-optimize layout
        self.optimize_for_agents(self.count_agent_panels());
    }
    
    /// Count active agent panels
    fn count_agent_panels(&self) -> usize {
        self.visibility.iter()
            .filter(|(panel, visible)| {
                **visible && matches!(panel, PanelType::AgentStream { .. })
            })
            .count()
    }
    
    /// Optimize layout based on number of active agents
    pub fn optimize_for_agents(&mut self, agent_count: usize) {
        let preset = match agent_count {
            0 => {
                // No agents - use default layout
                if self.is_visible(PanelType::ToolOutput) || self.is_visible(PanelType::Context) {
                    LayoutPreset::SideBySide
                } else {
                    LayoutPreset::SinglePanel
                }
            }
            1..=2 => LayoutPreset::ThreeColumn,
            3..=4 => LayoutPreset::QuadGrid,
            _ => {
                // Many agents - use a tabbed or scrollable layout
                // For now, use quad grid and limit visible agents
                LayoutPreset::QuadGrid
            }
        };
        
        self.set_preset(preset);
    }
    
    /// Get all active agent panels
    pub fn get_agent_panels(&self) -> Vec<(String, bool)> {
        self.visibility.iter()
            .filter_map(|(panel, visible)| {
                if let PanelType::AgentStream { agent_id } = panel {
                    Some((agent_id.clone(), *visible))
                } else {
                    None
                }
            })
            .collect()
    }
    
    /// Focus on a specific agent panel
    pub fn focus_agent_panel(&mut self, agent_id: &str) {
        let panel = PanelType::AgentStream { agent_id: agent_id.to_string() };
        if self.visibility.get(&panel).copied().unwrap_or(false) {
            self.set_focus(panel);
        }
    }
}

/// Get panel type display name
fn panel_type_name(panel: PanelType) -> String {
    match panel {
        PanelType::Chat => "Chat".to_string(),
        PanelType::ToolOutput => "Tool Output".to_string(),
        PanelType::Context => "Context".to_string(),
        PanelType::Workflow => "Workflow".to_string(),
        PanelType::Preview => "Preview".to_string(),
        PanelType::StatusBar => "Status".to_string(),
        PanelType::InputArea => "Input".to_string(),
        PanelType::AgentStream { agent_id } => format!("Agent: {}", agent_id),
        PanelType::AgentOverview => "Agent Overview".to_string(),
    }
}

/// Create centered rectangle
fn centered_rect(percent_x: u16, percent_y: u16, area: Rect) -> Rect {
    let vertical = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(area);
    
    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(vertical[1])[1]
}