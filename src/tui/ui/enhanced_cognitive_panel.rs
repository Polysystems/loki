//! Enhanced Cognitive Panel with Real-Time Updates
//! 
//! Provides rich visualization of cognitive processing with real-time data updates

use std::sync::Arc;
use std::time::Instant;
use anyhow::Result;
use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    symbols,
    text::{Line, Span},
    widgets::{
        Block, Borders, BorderType, Gauge, List, ListItem, Paragraph, 
       Sparkline, Tabs, Chart, Dataset, Axis, GraphType
    },
    Frame,
};
use tokio::sync::{broadcast};

use crate::tui::{
    cognitive::core::data_stream::{CognitiveDataStream, CognitiveDisplayState},
    cognitive::integration::main::CognitiveModality,
};

/// Enhanced cognitive panel tabs
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CognitivePanelTab {
    Overview,      // Main metrics and status
    Reasoning,     // Active reasoning chain
    ProcessingSteps, // Detailed cognitive processing steps
    Insights,      // Generated insights
    Modalities,    // Active cognitive modalities
    Learning,      // Learning progress
    Background,    // Background thoughts
    Performance,   // Performance metrics
}

/// Enhanced cognitive panel
pub struct EnhancedCognitivePanel {
    /// Data stream
    data_stream: Arc<CognitiveDataStream>,
    
    /// Current display state
    current_state: CognitiveDisplayState,
    
    /// State update receiver
    state_update_rx: broadcast::Receiver<()>,
    
    /// Current tab
    current_tab: CognitivePanelTab,
    
    /// Scroll states
    reasoning_scroll: usize,
    insights_scroll: usize,
    background_scroll: usize,
    
    /// Animation state
    animation_frame: usize,
    last_update: Instant,
    
    /// Panel visibility
    is_visible: bool,
}

impl EnhancedCognitivePanel {
    /// Create new enhanced cognitive panel
    pub fn new(data_stream: Arc<CognitiveDataStream>) -> Self {
        let state_update_rx = data_stream.subscribe_state_changes();
        
        Self {
            data_stream,
            current_state: CognitiveDisplayState::default(),
            state_update_rx,
            current_tab: CognitivePanelTab::Overview,
            reasoning_scroll: 0,
            insights_scroll: 0,
            background_scroll: 0,
            animation_frame: 0,
            last_update: Instant::now(),
            is_visible: true,
        }
    }
    
    /// Update panel state
    pub async fn update(&mut self) -> Result<()> {
        // Check for state updates
        while let Ok(()) = self.state_update_rx.try_recv() {
            self.current_state = self.data_stream.get_display_state().await;
        }
        
        // Update animation
        self.update_animations();
        
        Ok(())
    }
    
    /// Update animations (synchronous version for UI rendering)
    pub fn update_animations(&mut self) {
        if self.last_update.elapsed().as_millis() > 100 {
            self.animation_frame = (self.animation_frame + 1) % 60;
            self.last_update = Instant::now();
        }
    }
    
    /// Toggle visibility
    pub fn toggle_visibility(&mut self) {
        self.is_visible = !self.is_visible;
    }
    
    /// Set visibility
    pub fn set_visible(&mut self, visible: bool) {
        self.is_visible = visible;
    }
    
    /// Next tab
    pub fn next_tab(&mut self) {
        use CognitivePanelTab::*;
        self.current_tab = match self.current_tab {
            Overview => Reasoning,
            Reasoning => ProcessingSteps,
            ProcessingSteps => Insights,
            Insights => Modalities,
            Modalities => Learning,
            Learning => Background,
            Background => Performance,
            Performance => Overview,
        };
    }
    
    /// Previous tab
    pub fn prev_tab(&mut self) {
        use CognitivePanelTab::*;
        self.current_tab = match self.current_tab {
            Overview => Performance,
            Reasoning => Overview,
            ProcessingSteps => Reasoning,
            Insights => ProcessingSteps,
            Modalities => Insights,
            Learning => Modalities,
            Background => Learning,
            Performance => Background,
        };
    }
    
    /// Render the panel
    pub fn render(&self, f: &mut Frame, area: Rect) {
        if !self.is_visible {
            return;
        }
        
        // Main panel block
        let block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .title(self.get_title())
            .title_style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
            .border_style(Style::default().fg(self.get_border_color()));
        
        let inner = block.inner(area);
        f.render_widget(block, area);
        
        // Tab layout
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),  // Tab bar
                Constraint::Min(1),     // Content
            ])
            .split(inner);
        
        self.render_tabs(f, chunks[0]);
        
        // Render content based on current tab
        match self.current_tab {
            CognitivePanelTab::Overview => self.render_overview(f, chunks[1]),
            CognitivePanelTab::Reasoning => self.render_reasoning(f, chunks[1]),
            CognitivePanelTab::ProcessingSteps => self.render_processing_steps(f, chunks[1]),
            CognitivePanelTab::Insights => self.render_insights(f, chunks[1]),
            CognitivePanelTab::Modalities => self.render_modalities(f, chunks[1]),
            CognitivePanelTab::Learning => self.render_learning(f, chunks[1]),
            CognitivePanelTab::Background => self.render_background(f, chunks[1]),
            CognitivePanelTab::Performance => self.render_performance(f, chunks[1]),
        }
    }
    
    /// Get panel title with status
    fn get_title(&self) -> String {
        let status_icon = if self.current_state.processing_status.is_processing {
            self.get_animated_processing_icon()
        } else {
            "💤"
        };
        
        format!(" {} Cognitive Processing ", status_icon)
    }
    
    /// Get animated processing icon
    fn get_animated_processing_icon(&self) -> &'static str {
        match self.animation_frame % 8 {
            0 => "⠋",
            1 => "⠙",
            2 => "⠹",
            3 => "⠸",
            4 => "⠼",
            5 => "⠴",
            6 => "⠦",
            _ => "⠧",
        }
    }
    
    /// Get border color based on state
    fn get_border_color(&self) -> Color {
        if self.current_state.processing_status.is_processing {
            match (self.animation_frame / 10) % 3 {
                0 => Color::Cyan,
                1 => Color::Blue,
                _ => Color::Magenta,
            }
        } else {
            Color::DarkGray
        }
    }
    
    /// Render tab bar
    fn render_tabs(&self, f: &mut Frame, area: Rect) {
        let tab_titles = vec![
            "Overview", "Reasoning", "Processing", "Insights", 
            "Modalities", "Learning", "Background", "Performance"
        ];
        
        let tabs = Tabs::new(tab_titles)
            .block(Block::default().borders(Borders::BOTTOM))
            .select(self.current_tab as usize)
            .style(Style::default().fg(Color::White))
            .highlight_style(
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD | Modifier::UNDERLINED)
            );
        
        f.render_widget(tabs, area);
    }
    
    /// Render overview tab
    fn render_overview(&self, f: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(6),   // Awareness & coherence gauges
                Constraint::Length(4),   // Current focus
                Constraint::Length(6),   // Modality bars
                Constraint::Min(1),      // Active tasks
            ])
            .split(area);
        
        // Awareness and coherence gauges
        self.render_consciousness_gauges(f, chunks[0]);
        
        // Current focus
        self.render_current_focus(f, chunks[1]);
        
        // Active modalities
        self.render_modality_overview(f, chunks[2]);
        
        // Active tasks
        self.render_active_tasks(f, chunks[3]);
    }
    
    /// Render consciousness gauges
    fn render_consciousness_gauges(&self, f: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Percentage(50),
                Constraint::Percentage(50),
            ])
            .split(area);
        
        let metrics = &self.current_state.consciousness_metrics;
        
        // Awareness gauge
        let awareness_gauge = Gauge::default()
            .block(Block::default().title("🧠 Awareness Level"))
            .gauge_style(Style::default().fg(self.get_awareness_color(metrics.awareness_level)))
            .percent((metrics.awareness_level * 100.0) as u16)
            .label(format!("{:.1}%", metrics.awareness_level * 100.0));
        
        f.render_widget(awareness_gauge, chunks[0]);
        
        // Coherence gauge
        let coherence_gauge = Gauge::default()
            .block(Block::default().title("🔮 Gradient Coherence"))
            .gauge_style(Style::default().fg(self.get_coherence_color(metrics.coherence)))
            .percent((metrics.coherence * 100.0) as u16)
            .label(format!("{:.1}%", metrics.coherence * 100.0));
        
        f.render_widget(coherence_gauge, chunks[1]);
    }
    
    /// Render current focus
    fn render_current_focus(&self, f: &mut Frame, area: Rect) {
        let focus_text = if self.current_state.consciousness_metrics.current_focus.is_empty() {
            "No specific focus".to_string()
        } else {
            self.current_state.consciousness_metrics.current_focus.clone()
        };
        
        let paragraph = Paragraph::new(vec![
            Line::from(vec![
                Span::styled("Current Focus: ", Style::default().fg(Color::DarkGray)),
                Span::styled(focus_text, Style::default().fg(Color::Yellow).add_modifier(Modifier::ITALIC)),
            ]),
            Line::from(vec![
                Span::styled("Free Energy: ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    format!("{:.2}", self.current_state.consciousness_metrics.free_energy),
                    Style::default().fg(Color::Green)
                ),
            ]),
        ])
        .block(Block::default().borders(Borders::NONE));
        
        f.render_widget(paragraph, area);
    }
    
    /// Render modality overview
    fn render_modality_overview(&self, f: &mut Frame, area: Rect) {
        if self.current_state.modality_activations.is_empty() {
            let paragraph = Paragraph::new("No active modalities")
                .style(Style::default().fg(Color::DarkGray))
                .alignment(Alignment::Center);
            f.render_widget(paragraph, area);
            return;
        }
        
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints(
                self.current_state.modality_activations
                    .iter()
                    .map(|_| Constraint::Ratio(1, self.current_state.modality_activations.len() as u32))
                    .collect::<Vec<_>>()
            )
            .split(area);
        
        for (i, (modality, level)) in self.current_state.modality_activations.iter().enumerate() {
            if i < chunks.len() {
                self.render_modality_bar(f, chunks[i], modality, *level);
            }
        }
    }
    
    /// Render single modality bar
    fn render_modality_bar(&self, f: &mut Frame, area: Rect, modality: &CognitiveModality, level: f64) {
        let color = self.get_modality_color(modality);
        let icon = self.get_modality_icon(modality);
        
        let bar_height = ((area.height as f64) * level) as u16;
        let bar_area = Rect {
            x: area.x,
            y: area.y + area.height - bar_height,
            width: area.width,
            height: bar_height,
        };
        
        // Draw background
        let bg = Block::default()
            .style(Style::default().bg(Color::Black));
        f.render_widget(bg, area);
        
        // Draw activation bar
        let bar = Block::default()
            .style(Style::default().bg(color));
        f.render_widget(bar, bar_area);
        
        // Draw label
        let label = Paragraph::new(vec![
            Line::from(icon),
            Line::from(format!("{:.0}%", level * 100.0)),
        ])
        .style(Style::default().fg(Color::White))
        .alignment(Alignment::Center);
        
        f.render_widget(label, area);
    }
    
    /// Render active tasks
    fn render_active_tasks(&self, f: &mut Frame, area: Rect) {
        let block = Block::default()
            .borders(Borders::ALL)
            .title("Active Processing Tasks");
        
        let inner = block.inner(area);
        f.render_widget(block, area);
        
        if self.current_state.processing_status.active_tasks.is_empty() {
            let paragraph = Paragraph::new("No active tasks")
                .style(Style::default().fg(Color::DarkGray))
                .alignment(Alignment::Center);
            f.render_widget(paragraph, inner);
        } else {
            let items: Vec<ListItem> = self.current_state.processing_status.active_tasks
                .iter()
                .map(|task| {
                    ListItem::new(Line::from(vec![
                        Span::raw("• "),
                        Span::styled(task, Style::default().fg(Color::Cyan)),
                    ]))
                })
                .collect();
            
            let list = List::new(items);
            f.render_widget(list, inner);
        }
    }
    
    /// Render processing steps tab
    fn render_processing_steps(&self, f: &mut Frame, area: Rect) {
        if self.current_state.processing_steps.is_empty() {
            let paragraph = Paragraph::new("No processing steps recorded yet")
                .style(Style::default().fg(Color::DarkGray))
                .alignment(Alignment::Center);
            f.render_widget(paragraph, area);
        } else {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(3),  // Header
                    Constraint::Min(1),     // Steps list
                ])
                .split(area);
            
            // Header
            let header = Paragraph::new(vec![
                Line::from(vec![
                    Span::styled("Cognitive Processing Steps", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
                ]),
                Line::from(vec![
                    Span::styled(
                        format!("Total: {} steps | Showing most recent", self.current_state.processing_steps.len()),
                        Style::default().fg(Color::DarkGray)
                    ),
                ]),
            ])
            .block(Block::default().borders(Borders::BOTTOM));
            f.render_widget(header, chunks[0]);
            
            // Processing steps
            let items: Vec<ListItem> = self.current_state.processing_steps
                .iter()
                .rev()
                .map(|step| {
                    let step_icon = match step.step_type {
                        crate::tui::cognitive::core::data_stream::ProcessingStepType::MemoryRetrieval => "🔍",
                        crate::tui::cognitive::core::data_stream::ProcessingStepType::PatternRecognition => "🧩",
                        crate::tui::cognitive::core::data_stream::ProcessingStepType::ConceptFormation => "💡",
                        crate::tui::cognitive::core::data_stream::ProcessingStepType::ReasoningConstruction => "🔗",
                        crate::tui::cognitive::core::data_stream::ProcessingStepType::ToolSelection => "🔧",
                        crate::tui::cognitive::core::data_stream::ProcessingStepType::ResponseGeneration => "💬",
                        crate::tui::cognitive::core::data_stream::ProcessingStepType::SelfReflection => "🪞",
                        crate::tui::cognitive::core::data_stream::ProcessingStepType::ContextAnalysis => "📊",
                        crate::tui::cognitive::core::data_stream::ProcessingStepType::GoalEvaluation => "🎯",
                        crate::tui::cognitive::core::data_stream::ProcessingStepType::CreativeSynthesis => "🎨",
                    };
                    
                    let elapsed = step.timestamp.elapsed();
                    ListItem::new(vec![
                        Line::from(vec![
                            Span::styled(
                                format!("[{}] ", Self::format_elapsed(elapsed)),
                                Style::default().fg(Color::DarkGray)
                            ),
                            Span::styled(
                                format!("{} ", step_icon),
                                Style::default()
                            ),
                            Span::styled(
                                format!("{:?}", step.step_type),
                                Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)
                            ),
                            Span::styled(
                                format!(" ({}ms)", step.duration_ms),
                                Style::default().fg(Color::Green)
                            ),
                        ]),
                        Line::from(vec![
                            Span::raw("  "),
                            Span::styled(&step.description, Style::default().fg(Color::White)),
                        ]),
                        Line::from(vec![
                            Span::raw("  "),
                            Span::styled("In: ", Style::default().fg(Color::DarkGray)),
                            Span::styled(&step.input_summary, Style::default().fg(Color::Blue)),
                        ]),
                        Line::from(vec![
                            Span::raw("  "),
                            Span::styled("Out: ", Style::default().fg(Color::DarkGray)),
                            Span::styled(&step.output_summary, Style::default().fg(Color::Yellow)),
                        ]),
                        Line::from(""), // Spacing
                    ])
                })
                .collect();
            
            let list = List::new(items)
                .block(Block::default().borders(Borders::NONE));
            
            f.render_widget(list, chunks[1]);
        }
    }
    
    /// Render reasoning tab
    fn render_reasoning(&self, f: &mut Frame, area: Rect) {
        if let Some(reasoning) = &self.current_state.active_reasoning {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(3),  // Header
                    Constraint::Min(1),     // Steps
                ])
                .split(area);
            
            // Header with current step
            let header = Paragraph::new(vec![
                Line::from(vec![
                    Span::styled("Chain: ", Style::default().fg(Color::DarkGray)),
                    Span::styled(&reasoning.chain_id[..8], Style::default().fg(Color::Cyan)),
                    Span::raw(" | "),
                    Span::styled("Step: ", Style::default().fg(Color::DarkGray)),
                    Span::styled(
                        format!("{}/{}", reasoning.step_number, reasoning.total_steps),
                        Style::default().fg(Color::Yellow)
                    ),
                    Span::raw(" | "),
                    Span::styled("Confidence: ", Style::default().fg(Color::DarkGray)),
                    Span::styled(
                        format!("{:.0}%", reasoning.confidence * 100.0),
                        Style::default().fg(self.get_confidence_color(reasoning.confidence))
                    ),
                ]),
                Line::from(vec![
                    Span::styled("Current: ", Style::default().fg(Color::DarkGray)),
                    Span::styled(&reasoning.current_step, Style::default().fg(Color::White)),
                ]),
            ])
            .block(Block::default().borders(Borders::BOTTOM));
            
            f.render_widget(header, chunks[0]);
            
            // Reasoning steps
            self.render_reasoning_steps(f, chunks[1], reasoning);
        } else {
            let paragraph = Paragraph::new("No active reasoning chain")
                .style(Style::default().fg(Color::DarkGray))
                .alignment(Alignment::Center);
            f.render_widget(paragraph, area);
        }
    }
    
    /// Render reasoning steps
    fn render_reasoning_steps(&self, f: &mut Frame, area: Rect, reasoning: &crate::tui::cognitive::core::data_stream::ActiveReasoningDisplay) {
        let items: Vec<ListItem> = reasoning.recent_steps
            .iter()
            .rev()
            .map(|step| {
                let elapsed = step.timestamp.elapsed();
                ListItem::new(vec![
                    Line::from(vec![
                        Span::styled(
                            format!("[{}] ", Self::format_elapsed(elapsed)),
                            Style::default().fg(Color::DarkGray)
                        ),
                        Span::styled(
                            &step.step_type,
                            Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)
                        ),
                        Span::styled(
                            format!(" ({:.0}%)", step.confidence * 100.0),
                            Style::default().fg(self.get_confidence_color(step.confidence))
                        ),
                    ]),
                    Line::from(vec![
                        Span::raw("  "),
                        Span::styled(&step.content, Style::default().fg(Color::White)),
                    ]),
                    Line::from(""), // Spacing
                ])
            })
            .collect();
        
        let list = List::new(items)
            .block(Block::default().borders(Borders::NONE));
        
        f.render_stateful_widget(
            list,
            area,
            &mut ratatui::widgets::ListState::default().with_selected(Some(self.reasoning_scroll))
        );
    }
    
    /// Render insights tab
    fn render_insights(&self, f: &mut Frame, area: Rect) {
        if self.current_state.recent_insights.is_empty() {
            let paragraph = Paragraph::new("No insights generated yet")
                .style(Style::default().fg(Color::DarkGray))
                .alignment(Alignment::Center);
            f.render_widget(paragraph, area);
        } else {
            let items: Vec<ListItem> = self.current_state.recent_insights
                .iter()
                .rev()
                .map(|insight| {
                    ListItem::new(vec![
                        Line::from(vec![
                            Span::styled(
                                format!("[{}] ", insight.category),
                                Style::default().fg(self.get_insight_category_color(&insight.category))
                                    .add_modifier(Modifier::BOLD)
                            ),
                            Span::styled(
                                format!("R:{:.0}% N:{:.0}%", insight.relevance * 100.0, insight.novelty * 100.0),
                                Style::default().fg(Color::DarkGray)
                            ),
                        ]),
                        Line::from(vec![
                            Span::raw("💡 "),
                            Span::styled(&insight.content, Style::default().fg(Color::Yellow)),
                        ]),
                        Line::from(""), // Spacing
                    ])
                })
                .collect();
            
            let list = List::new(items)
                .block(Block::default().title("Recent Insights").borders(Borders::ALL));
            
            f.render_widget(list, area);
        }
    }
    
    /// Render modalities tab
    fn render_modalities(&self, f: &mut Frame, area: Rect) {
        let modality_info = vec![
            (CognitiveModality::Logical, "🔍", "Logical", "Systematic reasoning and deduction"),
            (CognitiveModality::Creative, "🎨", "Creative", "Novel connections and synthesis"),
            (CognitiveModality::Emotional, "💝", "Emotional", "Empathy and feeling analysis"),
            (CognitiveModality::Social, "👥", "Social", "Theory of mind and social dynamics"),
            (CognitiveModality::Abstract, "🌌", "Abstract", "High-level conceptual thinking"),
            (CognitiveModality::Analytical, "📊", "Analytical", "Data analysis and patterns"),
            (CognitiveModality::Narrative, "📖", "Narrative", "Story understanding and creation"),
            (CognitiveModality::Intuitive, "✨", "Intuitive", "Holistic and emergent insights"),
        ];
        
        let items: Vec<ListItem> = modality_info.iter()
            .map(|(modality, icon, name, desc)| {
                let activation = self.current_state.modality_activations
                    .iter()
                    .find(|(m, _)| m == modality)
                    .map(|(_, level)| *level)
                    .unwrap_or(0.0);
                
                let color = if activation > 0.0 {
                    self.get_modality_color(modality)
                } else {
                    Color::DarkGray
                };
                
                ListItem::new(vec![
                    Line::from(vec![
                        Span::styled(format!("{} ", icon), Style::default().fg(color)),
                        Span::styled(*name, Style::default().fg(color).add_modifier(Modifier::BOLD)),
                        Span::raw(" "),
                        Span::styled(
                            if activation > 0.0 {
                                format!("[{:.0}%]", activation * 100.0)
                            } else {
                                "[Inactive]".to_string()
                            },
                            Style::default().fg(color)
                        ),
                    ]),
                    Line::from(vec![
                        Span::raw("  "),
                        Span::styled(*desc, Style::default().fg(Color::Gray)),
                    ]),
                    Line::from(""), // Spacing
                ])
            })
            .collect();
        
        let list = List::new(items)
            .block(Block::default().title("Cognitive Modalities").borders(Borders::ALL));
        
        f.render_widget(list, area);
    }
    
    /// Render learning tab
    fn render_learning(&self, f: &mut Frame, area: Rect) {
        if self.current_state.learning_events.is_empty() {
            let paragraph = Paragraph::new("No learning events recorded")
                .style(Style::default().fg(Color::DarkGray))
                .alignment(Alignment::Center);
            f.render_widget(paragraph, area);
        } else {
            let items: Vec<ListItem> = self.current_state.learning_events
                .iter()
                .rev()
                .map(|event| {
                    let delta_color = if event.understanding_delta > 0.0 {
                        Color::Green
                    } else {
                        Color::Red
                    };
                    
                    ListItem::new(vec![
                        Line::from(vec![
                            Span::styled("📚 ", Style::default().fg(Color::Blue)),
                            Span::styled(&event.topic, Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
                            Span::raw(" "),
                            Span::styled(
                                format!("({:+.1}%)", event.understanding_delta * 100.0),
                                Style::default().fg(delta_color)
                            ),
                        ]),
                        Line::from(vec![
                            Span::raw("  💡 "),
                            Span::styled(&event.key_realization, Style::default().fg(Color::Yellow)),
                        ]),
                        Line::from(""), // Spacing
                    ])
                })
                .collect();
            
            let list = List::new(items)
                .block(Block::default().title("Learning Progress").borders(Borders::ALL));
            
            f.render_widget(list, area);
        }
    }
    
    /// Render background thoughts tab
    fn render_background(&self, f: &mut Frame, area: Rect) {
        if self.current_state.background_thoughts.is_empty() {
            let paragraph = Paragraph::new("No background thoughts")
                .style(Style::default().fg(Color::DarkGray))
                .alignment(Alignment::Center);
            f.render_widget(paragraph, area);
        } else {
            let items: Vec<ListItem> = self.current_state.background_thoughts
                .iter()
                .rev()
                .map(|thought| {
                    let importance_color = match thought.importance {
                        i if i > 0.8 => Color::Red,
                        i if i > 0.6 => Color::Yellow,
                        i if i > 0.4 => Color::Blue,
                        _ => Color::Gray,
                    };
                    
                    ListItem::new(vec![
                        Line::from(vec![
                            Span::styled(
                                format!("[{}] ", thought.category),
                                Style::default().fg(Color::Magenta)
                            ),
                            Span::styled(
                                format!("⚡{:.0}", thought.importance * 10.0),
                                Style::default().fg(importance_color)
                            ),
                        ]),
                        Line::from(vec![
                            Span::raw("💭 "),
                            Span::styled(&thought.content, Style::default().fg(Color::White)),
                        ]),
                        Line::from(""), // Spacing
                    ])
                })
                .collect();
            
            let list = List::new(items)
                .block(Block::default().title("Background Processing").borders(Borders::ALL));
            
            f.render_widget(list, area);
        }
    }
    
    /// Render performance tab
    fn render_performance(&self, f: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(8),   // Resource usage chart
                Constraint::Length(6),   // Processing metrics
                Constraint::Min(1),      // History sparklines
            ])
            .split(area);
        
        // Resource usage
        self.render_resource_chart(f, chunks[0]);
        
        // Processing metrics
        self.render_processing_metrics(f, chunks[1]);
        
        // History sparklines
        self.render_history_sparklines(f, chunks[2]);
    }
    
    /// Render resource usage chart
    fn render_resource_chart(&self, f: &mut Frame, area: Rect) {
        let history = &self.current_state.processing_status.resource_history;
        if history.is_empty() {
            return;
        }
        
        // Convert to chart data
        let data: Vec<(f64, f64)> = history.iter()
            .enumerate()
            .map(|(i, (_, usage))| (i as f64, *usage * 100.0))
            .collect();
        
        let datasets = vec![
            Dataset::default()
                .name("CPU")
                .marker(symbols::Marker::Dot)
                .style(Style::default().fg(Color::Cyan))
                .graph_type(GraphType::Line)
                .data(&data),
        ];
        
        let chart = Chart::new(datasets)
            .block(Block::default().title("Resource Usage").borders(Borders::ALL))
            .x_axis(
                Axis::default()
                    .title("Time")
                    .style(Style::default().fg(Color::Gray))
                    .bounds([0.0, history.len() as f64])
            )
            .y_axis(
                Axis::default()
                    .title("Usage %")
                    .style(Style::default().fg(Color::Gray))
                    .bounds([0.0, 100.0])
                    .labels(vec!["0", "50", "100"])
            );
        
        f.render_widget(chart, area);
    }
    
    /// Render processing metrics
    fn render_processing_metrics(&self, f: &mut Frame, area: Rect) {
        let status = &self.current_state.processing_status;
        
        let lines = vec![
            Line::from(vec![
                Span::styled("Status: ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    if status.is_processing { "Active" } else { "Idle" },
                    Style::default().fg(if status.is_processing { Color::Green } else { Color::Gray })
                ),
            ]),
            Line::from(vec![
                Span::styled("Depth: ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    format!("{:.1}%", status.processing_depth * 100.0),
                    Style::default().fg(Color::Cyan)
                ),
            ]),
            Line::from(vec![
                Span::styled("Active Tasks: ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    status.active_tasks.len().to_string(),
                    Style::default().fg(Color::Yellow)
                ),
            ]),
            Line::from(vec![
                Span::styled("Resource Usage: ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    format!("{:.1}%", status.resource_usage * 100.0),
                    Style::default().fg(self.get_resource_color(status.resource_usage))
                ),
            ]),
        ];
        
        let paragraph = Paragraph::new(lines)
            .block(Block::default().borders(Borders::ALL).title("Metrics"));
        
        f.render_widget(paragraph, area);
    }
    
    /// Render history sparklines
    fn render_history_sparklines(&self, f: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(area);
        
        // Awareness history
        let awareness_data: Vec<u64> = self.current_state.consciousness_metrics.awareness_history
            .iter()
            .map(|(_, level)| (*level * 100.0) as u64)
            .collect();
        
        if !awareness_data.is_empty() {
            let awareness_sparkline = Sparkline::default()
                .block(Block::default().title("Awareness Trend").borders(Borders::ALL))
                .data(&awareness_data)
                .style(Style::default().fg(Color::Cyan));
            f.render_widget(awareness_sparkline, chunks[0]);
        }
        
        // Coherence history
        let coherence_data: Vec<u64> = self.current_state.consciousness_metrics.coherence_history
            .iter()
            .map(|(_, level)| (*level * 100.0) as u64)
            .collect();
        
        if !coherence_data.is_empty() {
            let coherence_sparkline = Sparkline::default()
                .block(Block::default().title("Coherence Trend").borders(Borders::ALL))
                .data(&coherence_data)
                .style(Style::default().fg(Color::Magenta));
            f.render_widget(coherence_sparkline, chunks[1]);
        }
    }
    
    // Helper methods
    
    fn get_awareness_color(&self, level: f64) -> Color {
        match level {
            l if l > 0.8 => Color::Green,
            l if l > 0.6 => Color::Yellow,
            l if l > 0.4 => Color::Blue,
            _ => Color::Red,
        }
    }
    
    fn get_coherence_color(&self, level: f64) -> Color {
        match level {
            l if l > 0.8 => Color::Magenta,
            l if l > 0.6 => Color::Blue,
            l if l > 0.4 => Color::Cyan,
            _ => Color::Gray,
        }
    }
    
    fn get_confidence_color(&self, confidence: f64) -> Color {
        match confidence {
            c if c > 0.9 => Color::Green,
            c if c > 0.7 => Color::Yellow,
            c if c > 0.5 => Color::Blue,
            _ => Color::Red,
        }
    }
    
    fn get_resource_color(&self, usage: f64) -> Color {
        match usage {
            u if u > 0.9 => Color::Red,
            u if u > 0.7 => Color::Yellow,
            u if u > 0.5 => Color::Blue,
            _ => Color::Green,
        }
    }
    
    fn get_modality_color(&self, modality: &CognitiveModality) -> Color {
        match modality {
            CognitiveModality::Logical => Color::Blue,
            CognitiveModality::Creative => Color::Magenta,
            CognitiveModality::Emotional => Color::Red,
            CognitiveModality::Social => Color::Yellow,
            CognitiveModality::Abstract => Color::Cyan,
            CognitiveModality::Analytical => Color::Green,
            CognitiveModality::Narrative => Color::LightMagenta,
            CognitiveModality::Intuitive => Color::LightCyan,
        }
    }
    
    fn get_modality_icon(&self, modality: &CognitiveModality) -> &'static str {
        match modality {
            CognitiveModality::Logical => "🔍",
            CognitiveModality::Creative => "🎨",
            CognitiveModality::Emotional => "💝",
            CognitiveModality::Social => "👥",
            CognitiveModality::Abstract => "🌌",
            CognitiveModality::Analytical => "📊",
            CognitiveModality::Narrative => "📖",
            CognitiveModality::Intuitive => "✨",
        }
    }
    
    fn get_insight_category_color(&self, category: &str) -> Color {
        match category {
            "Goals" => Color::Green,
            "Learning" => Color::Blue,
            "Social" => Color::Yellow,
            "Creative" => Color::Magenta,
            "Self" => Color::Cyan,
            "Thermodynamic" => Color::Red,
            "Temporal" => Color::LightBlue,
            _ => Color::Gray,
        }
    }
    
    fn format_elapsed(elapsed: std::time::Duration) -> String {
        if elapsed.as_secs() < 60 {
            format!("{}s", elapsed.as_secs())
        } else if elapsed.as_secs() < 3600 {
            format!("{}m", elapsed.as_secs() / 60)
        } else {
            format!("{}h", elapsed.as_secs() / 3600)
        }
    }
}