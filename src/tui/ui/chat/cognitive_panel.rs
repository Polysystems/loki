//! Cognitive Panel Integration for Chat Tab
//!
//! This module integrates the cognitive activity indicators into the chat tab,
//! providing a dedicated panel for real-time cognitive monitoring.

use std::sync::Arc;
use anyhow::Result;
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph, Tabs},
    Frame,
};

use crate::tui::{
    ui::cognitive_indicators::{CognitiveActivityIndicator, CognitiveMiniIndicator},
    chat::integrations::cognitive::CognitiveChatEnhancement,
};

/// Cognitive panel tabs
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CognitivePanelTab {
    Activity,
    Insights,
    Modalities,
    Stream,
}

/// Cognitive panel for displaying real-time cognitive activity
pub struct CognitivePanel {
    /// Current tab
    current_tab: CognitivePanelTab,
    
    /// Activity indicator component
    activity_indicator: CognitiveActivityIndicator,
    
    /// Mini indicator for compact display
    mini_indicator: CognitiveMiniIndicator,
    
    /// Recent cognitive insights
    recent_insights: Vec<String>,
    
    /// Active modalities
    active_modalities: Vec<String>,
    
    /// Cognitive narrative
    consciousness_narrative: String,
    
    /// Panel visibility
    is_visible: bool,
    
    /// Expanded view
    is_expanded: bool,
}

impl CognitivePanel {
    /// Create new cognitive panel
    pub fn new() -> Self {
        Self {
            current_tab: CognitivePanelTab::Activity,
            activity_indicator: CognitiveActivityIndicator::new(),
            mini_indicator: CognitiveMiniIndicator::new(),
            recent_insights: Vec::new(),
            active_modalities: Vec::new(),
            consciousness_narrative: String::new(),
            is_visible: true,
            is_expanded: false,
        }
    }
    
    /// Set cognitive enhancement reference
    pub fn set_enhancement(&mut self, enhancement: Arc<CognitiveChatEnhancement>) {
        self.activity_indicator.set_enhancement(enhancement);
    }
    
    /// Update panel from cognitive state
    pub async fn update(&mut self, enhancement: &CognitiveChatEnhancement) -> Result<()> {
        // Update activity indicator
        self.activity_indicator.update().await;
        
        // Update mini indicator
        let activity = enhancement.get_cognitive_activity();
        self.mini_indicator.update(&activity);
        
        // Update insights if visible
        if self.is_visible && self.current_tab == CognitivePanelTab::Insights {
            self.update_insights(enhancement).await?;
        }
        
        // Update narrative if on stream tab
        if self.is_visible && self.current_tab == CognitivePanelTab::Stream {
            self.update_narrative(enhancement).await?;
        }
        
        Ok(())
    }
    
    /// Update recent insights
    async fn update_insights(&mut self, enhancement: &CognitiveChatEnhancement) -> Result<()> {
        // For now, using placeholder data since cognitive_stream is not a field in CognitiveChatEnhancement
        self.recent_insights = vec![
            "Goals - Active: Processing interactions".to_string(),
            "Learning - Adapting to patterns".to_string(),
            "Thermodynamic - Entropy: 30%, Efficiency: 85%".to_string(),
        ];
        Ok(())
    }
    
    /// Update cognitive narrative
    async fn update_narrative(&mut self, enhancement: &CognitiveChatEnhancement) -> Result<()> {
        // For now, using placeholder narrative since cognitive_stream is not a field in CognitiveChatEnhancement
        self.consciousness_narrative = "Cognitive system actively processing and learning from interactions.".to_string();
        Ok(())
    }
    
    /// Toggle panel visibility
    pub fn toggle_visibility(&mut self) {
        self.is_visible = !self.is_visible;
    }
    
    /// Toggle expanded view
    pub fn toggle_expanded(&mut self) {
        self.is_expanded = !self.is_expanded;
    }
    
    /// Switch to next tab
    pub fn next_tab(&mut self) {
        self.current_tab = match self.current_tab {
            CognitivePanelTab::Activity => CognitivePanelTab::Insights,
            CognitivePanelTab::Insights => CognitivePanelTab::Modalities,
            CognitivePanelTab::Modalities => CognitivePanelTab::Stream,
            CognitivePanelTab::Stream => CognitivePanelTab::Activity,
        };
    }
    
    /// Switch to previous tab
    pub fn prev_tab(&mut self) {
        self.current_tab = match self.current_tab {
            CognitivePanelTab::Activity => CognitivePanelTab::Stream,
            CognitivePanelTab::Insights => CognitivePanelTab::Activity,
            CognitivePanelTab::Modalities => CognitivePanelTab::Insights,
            CognitivePanelTab::Stream => CognitivePanelTab::Modalities,
        };
    }
    
    /// Render the cognitive panel
    pub fn render(&mut self, f: &mut Frame, area: Rect) {
        if !self.is_visible {
            return;
        }
        
        // Create layout
        let chunks = if self.is_expanded {
            Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(3),  // Tabs
                    Constraint::Min(10),    // Content
                ])
                .split(area)
        } else {
            Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(3),  // Mini display
                ])
                .split(area)
        };
        
        if self.is_expanded {
            // Render tabs
            let tab_titles = vec!["Activity", "Insights", "Modalities", "Stream"];
            let tabs = Tabs::new(tab_titles)
                .block(Block::default().borders(Borders::ALL).title(" 🧠 Cognitive Panel "))
                .select(self.current_tab as usize)
                .style(Style::default())
                .highlight_style(Style::default().add_modifier(Modifier::BOLD).fg(Color::Yellow));
            f.render_widget(tabs, chunks[0]);
            
            // Render content based on current tab
            match self.current_tab {
                CognitivePanelTab::Activity => {
                    self.activity_indicator.render_panel(f, chunks[1]);
                }
                CognitivePanelTab::Insights => {
                    self.render_insights(f, chunks[1]);
                }
                CognitivePanelTab::Modalities => {
                    self.render_modalities(f, chunks[1]);
                }
                CognitivePanelTab::Stream => {
                    self.render_stream(f, chunks[1]);
                }
            }
        } else {
            // Render mini indicator
            let mini_block = Block::default()
                .borders(Borders::ALL)
                .title(" 🧠 Cognitive ");
            let mini_paragraph = Paragraph::new(self.mini_indicator.render_line())
                .block(mini_block);
            f.render_widget(mini_paragraph, chunks[0]);
        }
    }
    
    /// Render insights tab
    fn render_insights(&self, f: &mut Frame, area: Rect) {
        let items: Vec<ListItem> = self.recent_insights
            .iter()
            .map(|insight| {
                ListItem::new(vec![
                    Line::from(vec![
                        Span::styled("• ", Style::default().fg(Color::Yellow)),
                        Span::raw(insight),
                    ])
                ])
            })
            .collect();
        
        let list = List::new(items)
            .block(Block::default()
                .borders(Borders::ALL)
                .title(" Recent Insights "));
        
        f.render_widget(list, area);
    }
    
    /// Render modalities tab
    fn render_modalities(&self, f: &mut Frame, area: Rect) {
        let text = if self.active_modalities.is_empty() {
            vec![Line::from("No active modalities")]
        } else {
            self.active_modalities
                .iter()
                .map(|m| Line::from(format!("• {}", m)))
                .collect()
        };
        
        let paragraph = Paragraph::new(text)
            .block(Block::default()
                .borders(Borders::ALL)
                .title(" Active Modalities "));
        
        f.render_widget(paragraph, area);
    }
    
    /// Render stream tab
    fn render_stream(&self, f: &mut Frame, area: Rect) {
        let paragraph = Paragraph::new(self.consciousness_narrative.clone())
            .block(Block::default()
                .borders(Borders::ALL)
                .title(" Cognitive Stream "))
            .wrap(ratatui::widgets::Wrap { trim: true });
        
        f.render_widget(paragraph, area);
    }
}