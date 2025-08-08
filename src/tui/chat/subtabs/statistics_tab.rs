//! Statistics subtab for chat interface

use std::sync::Arc;
use tokio::sync::RwLock;
use ratatui::{Frame, layout::Rect};
use crossterm::event::KeyEvent;
use anyhow::Result;

use super::SubtabController;
use crate::tui::chat::ChatState;
use crate::tui::chat::statistics::{StatisticsDashboard, DashboardConfig};

/// Statistics tab for viewing chat analytics
pub struct StatisticsTab {
    /// Statistics dashboard
    dashboard: StatisticsDashboard,
    
    /// Chat state reference
    chat_state: Arc<RwLock<ChatState>>,
}

impl StatisticsTab {
    /// Create a new statistics tab
    pub fn new(chat_state: Arc<RwLock<ChatState>>) -> Self {
        let config = DashboardConfig::default();
        let dashboard = StatisticsDashboard::new(chat_state.clone(), config);
        
        Self {
            dashboard,
            chat_state,
        }
    }
}

impl SubtabController for StatisticsTab {
    fn render(&mut self, f: &mut Frame, area: Rect) {
        self.dashboard.render(f, area);
    }
    
    fn handle_input(&mut self, key: KeyEvent) -> Result<()> {
        self.dashboard.handle_input(key)
    }
    
    fn update(&mut self) -> Result<()> {
        // Update dashboard metrics
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                self.dashboard.update().await
            })
        })
    }
    
    fn name(&self) -> &str {
        "Statistics"
    }
    
    fn title(&self) -> String {
        "📊 Statistics".to_string()
    }
}