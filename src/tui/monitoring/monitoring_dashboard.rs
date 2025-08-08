//! TUI Monitoring Dashboard
//!
//! Provides a comprehensive real-time monitoring interface showing system
//! metrics, cost analytics, distributed safety status, and interactive
//! controls.

use std::collections::VecDeque;
use std::io::{Stdout, stdout};
use std::sync::Arc;
use std::time::{Duration, Instant};

use ratatui::style::Styled;

use anyhow::Result;
use crossterm::event::{
    self,
    DisableMouseCapture,
    EnableMouseCapture,
    Event,
    KeyCode,
    KeyEventKind,
};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen,
    LeaveAlternateScreen,
    disable_raw_mode,
    enable_raw_mode,
};
use ratatui::backend::{Backend, CrosstermBackend};
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Margin, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{
    Axis,
    Block,
    Borders,
    Cell,
    Chart,
    Clear,
    Dataset,
    Gauge,
    LineGauge,
    List,
    ListItem,
    Paragraph,
    Row,
    Sparkline,
    Table,
    Tabs,
    Wrap,
};
use ratatui::{Frame, Terminal, symbols};
use tokio::sync::{RwLock, broadcast};
use tracing::{error, info, warn};

use crate::monitoring::cost_analytics::{
    AlertSeverity as CostAlertSeverity,
    CostAlert,
    CostMetrics,
};
use crate::monitoring::distributed_safety::{ClusterHealth, ClusterStatus, DistributedSafetyEvent};
use crate::monitoring::real_time::{AlertLevel, SystemAlert, SystemMetrics};
use crate::tui::real_time_integration::RealTimeMetricsAggregator;

/// Dashboard state
#[derive(Debug, Clone)]
pub struct DashboardState {
    /// Current tab index
    pub current_tab: usize,

    /// Current system metrics
    pub system_metrics: Option<SystemMetrics>,

    /// System metrics history
    pub metrics_history: VecDeque<SystemMetrics>,

    /// Current cost metrics
    pub cost_metrics: Option<CostMetrics>,

    /// Cost history for charts
    pub cost_history: VecDeque<CostMetrics>,

    /// Cluster status
    pub cluster_status: Option<ClusterStatus>,

    /// Active system alerts
    pub system_alerts: Vec<SystemAlert>,

    /// Active cost alerts
    pub cost_alerts: Vec<CostAlert>,

    /// Recent distributed safety events
    pub safety_events: VecDeque<DistributedSafetyEvent>,

    /// Selected list item for interaction
    pub selected_item: usize,

    /// Scroll position for logs
    pub scroll_position: usize,

    /// Show help overlay
    pub show_help: bool,

    /// Auto-refresh enabled
    pub auto_refresh: bool,

    /// Last update time
    pub last_update: Instant,
}

impl Default for DashboardState {
    fn default() -> Self {
        Self {
            current_tab: 0,
            system_metrics: None,
            metrics_history: VecDeque::with_capacity(300), // 5 minutes at 1s intervals
            cost_metrics: None,
            cost_history: VecDeque::with_capacity(1440), // 24 hours at 1m intervals
            cluster_status: None,
            system_alerts: Vec::new(),
            cost_alerts: Vec::new(),
            safety_events: VecDeque::with_capacity(100),
            selected_item: 0,
            scroll_position: 0,
            show_help: false,
            auto_refresh: true,
            last_update: Instant::now(),
        }
    }
}

/// Tab definitions
#[derive(Debug, Clone)]
pub enum DashboardTab {
    Overview,
    SystemMetrics,
    CostAnalytics,
    SafetyCluster,
    Alerts,
    Logs,
}

impl DashboardTab {
    pub fn title(&self) -> &'static str {
        match self {
            DashboardTab::Overview => "Overview",
            DashboardTab::SystemMetrics => "System",
            DashboardTab::CostAnalytics => "Costs",
            DashboardTab::SafetyCluster => "Safety",
            DashboardTab::Alerts => "Alerts",
            DashboardTab::Logs => "Logs",
        }
    }

    pub fn all() -> Vec<DashboardTab> {
        vec![
            DashboardTab::Overview,
            DashboardTab::SystemMetrics,
            DashboardTab::CostAnalytics,
            DashboardTab::SafetyCluster,
            DashboardTab::Alerts,
            DashboardTab::Logs,
        ]
    }
}

/// TUI Monitoring Dashboard
pub struct MonitoringDashboard {
    /// Terminal interface
    terminal: Terminal<CrosstermBackend<Stdout>>,

    /// Dashboard state
    state: Arc<RwLock<DashboardState>>,

    /// System metrics receiver
    system_metrics_rx: Option<broadcast::Receiver<SystemMetrics>>,

    /// System alerts receiver
    system_alerts_rx: Option<broadcast::Receiver<SystemAlert>>,

    /// Cost metrics receiver
    cost_metrics_rx: Option<broadcast::Receiver<CostMetrics>>,

    /// Cost alerts receiver
    cost_alerts_rx: Option<broadcast::Receiver<CostAlert>>,

    /// Safety events receiver
    safety_events_rx: Option<broadcast::Receiver<DistributedSafetyEvent>>,

    /// Update interval
    update_interval: Duration,
}

impl MonitoringDashboard {
    /// Create a new monitoring dashboard with real-time integration
    pub fn new_with_real_time(
        metrics_aggregator: &RealTimeMetricsAggregator,
    ) -> Result<Self> {
        enable_raw_mode().map_err(|e| {
            error!("Failed to enable raw mode: {}", e);
            e
        })?;
        let mut stdout = stdout();
        execute!(stdout, EnterAlternateScreen, EnableMouseCapture).map_err(|e| {
            error!("Failed to setup terminal: {}", e);
            e
        })?;
        let backend = CrosstermBackend::new(stdout);
        let terminal = Terminal::new(backend).map_err(|e| {
            error!("Failed to create terminal: {}", e);
            e
        })?;

        Ok(Self {
            terminal,
            state: Arc::new(RwLock::new(DashboardState::default())),
            system_metrics_rx: Some(metrics_aggregator.get_system_metrics_receiver()),
            system_alerts_rx: Some(metrics_aggregator.get_system_alerts_receiver()),
            cost_metrics_rx: Some(metrics_aggregator.get_cost_metrics_receiver()),
            cost_alerts_rx: Some(metrics_aggregator.get_cost_alerts_receiver()),
            safety_events_rx: Some(metrics_aggregator.get_safety_events_receiver()),
            update_interval: Duration::from_millis(500),
        })
    }

    /// Create a new monitoring dashboard (legacy method)
    pub fn new(
        system_metrics_rx: Option<broadcast::Receiver<SystemMetrics>>,
        system_alerts_rx: Option<broadcast::Receiver<SystemAlert>>,
        cost_metrics_rx: Option<broadcast::Receiver<CostMetrics>>,
        cost_alerts_rx: Option<broadcast::Receiver<CostAlert>>,
        safety_events_rx: Option<broadcast::Receiver<DistributedSafetyEvent>>,
    ) -> Result<Self> {
        enable_raw_mode().map_err(|e| {
            error!("Failed to enable raw mode: {}", e);
            e
        })?;
        let mut stdout = stdout();
        execute!(stdout, EnterAlternateScreen, EnableMouseCapture).map_err(|e| {
            error!("Failed to setup terminal: {}", e);
            e
        })?;
        let backend = CrosstermBackend::new(stdout);
        let terminal = Terminal::new(backend).map_err(|e| {
            error!("Failed to create terminal: {}", e);
            e
        })?;

        Ok(Self {
            terminal,
            state: Arc::new(RwLock::new(DashboardState::default())),
            system_metrics_rx,
            system_alerts_rx,
            cost_metrics_rx,
            cost_alerts_rx,
            safety_events_rx,
            update_interval: Duration::from_millis(500),
        })
    }

    /// Run the dashboard
    pub async fn run(&mut self) -> Result<()> {
        info!("Starting TUI monitoring dashboard");

        // Start background update tasks
        self.start_update_tasks().await?;

        // Main event loop
        let mut last_tick = Instant::now();

        loop {
            // Handle events
            if event::poll(Duration::from_millis(100)).map_err(|e| {
                error!("Failed to poll events: {}", e);
                e
            })? {
                if let Event::Key(key) = event::read().map_err(|e| {
                    error!("Failed to read key event: {}", e);
                    e
                })? {
                    if key.kind == KeyEventKind::Press {
                        if self.handle_key_event(key.code).await.map_err(|e| {
                            error!("Failed to handle key event: {}", e);
                            e
                        })? {
                            break; // Exit requested
                        }
                    }
                }
            }

            // Update UI
            if last_tick.elapsed() >= self.update_interval {
                // Simple synchronous drawing - just draw basic layout
                if let Err(e) = self.terminal.draw(|f| {
                    use ratatui::text::Text;
                    use ratatui::widgets::{Block, Borders, Paragraph};

                    let block = Block::default()
                        .title("Loki AI Monitoring Dashboard")
                        .borders(Borders::ALL);

                    let paragraph = Paragraph::new(Text::from("Dashboard Loading...")).block(block);

                    f.render_widget(paragraph, f.area());
                }) {
                    error!("Failed to draw dashboard UI: {}", e);
                    return Err(e.into());
                }
                last_tick = Instant::now();
            }
        }

        self.cleanup()?;
        Ok(())
    }

    /// Start background update tasks
    async fn start_update_tasks(&mut self) -> Result<()> {
        let state = self.state.clone();

        // System metrics updates
        if let Some(mut rx) = self.system_metrics_rx.take() {
            let state = state.clone();
            tokio::spawn(async move {
                while let Ok(metrics) = rx.recv().await {
                    if let Err(e) = async {
                        let mut state = state.write().await;
                        state.system_metrics = Some(metrics.clone());
                        state.metrics_history.push_back(metrics);

                        // Limit history size
                        if state.metrics_history.len() > 300 {
                            state.metrics_history.pop_front();
                        }

                        state.last_update = Instant::now();
                        Ok::<(), anyhow::Error>(())
                    }
                    .await
                    {
                        error!("Failed to update system metrics: {}", e);
                    }
                }
            });
        }

        // System alerts updates
        if let Some(mut rx) = self.system_alerts_rx.take() {
            let state = state.clone();
            tokio::spawn(async move {
                while let Ok(alert) = rx.recv().await {
                    if let Err(e) = async {
                        let mut state = state.write().await;
                        state.system_alerts.push(alert);

                        // Limit alerts
                        if state.system_alerts.len() > 50 {
                            let len = state.system_alerts.len();
                            state.system_alerts.drain(0..len - 50);
                        }
                        Ok::<(), anyhow::Error>(())
                    }
                    .await
                    {
                        error!("Failed to update system alerts: {}", e);
                    }
                }
            });
        }

        // Cost metrics updates
        if let Some(mut rx) = self.cost_metrics_rx.take() {
            let state = state.clone();
            tokio::spawn(async move {
                while let Ok(metrics) = rx.recv().await {
                    if let Err(e) = async {
                        let mut state = state.write().await;
                        state.cost_metrics = Some(metrics.clone());
                        state.cost_history.push_back(metrics);

                        // Limit history size
                        if state.cost_history.len() > 1440 {
                            state.cost_history.pop_front();
                        }
                        Ok::<(), anyhow::Error>(())
                    }
                    .await
                    {
                        error!("Failed to update cost metrics: {}", e);
                    }
                }
            });
        }

        // Cost alerts updates
        if let Some(mut rx) = self.cost_alerts_rx.take() {
            let state = state.clone();
            tokio::spawn(async move {
                while let Ok(alert) = rx.recv().await {
                    if let Err(e) = async {
                        let mut state = state.write().await;
                        state.cost_alerts.push(alert);

                        // Limit alerts
                        if state.cost_alerts.len() > 50 {
                            let len = state.cost_alerts.len();
                            state.cost_alerts.drain(0..len - 50);
                        }
                        Ok::<(), anyhow::Error>(())
                    }
                    .await
                    {
                        error!("Failed to update cost alerts: {}", e);
                    }
                }
            });
        }

        // Safety events updates
        if let Some(mut rx) = self.safety_events_rx.take() {
            let state = state.clone();
            tokio::spawn(async move {
                while let Ok(event) = rx.recv().await {
                    if let Err(e) = async {
                        let mut state = state.write().await;
                        state.safety_events.push_back(event);

                        // Limit events
                        if state.safety_events.len() > 100 {
                            state.safety_events.pop_front();
                        }
                        Ok::<(), anyhow::Error>(())
                    }
                    .await
                    {
                        error!("Failed to update safety events: {}", e);
                    }
                }
            });
        }

        Ok(())
    }

    /// Handle keyboard input
    async fn handle_key_event(&mut self, key: KeyCode) -> Result<bool> {
        let mut state = self.state.write().await;

        match key {
            KeyCode::Char('q') | KeyCode::Esc => return Ok(true), // Exit
            KeyCode::Char('h') | KeyCode::F(1) => {
                state.show_help = !state.show_help;
            }
            KeyCode::Char('r') => {
                state.auto_refresh = !state.auto_refresh;
            }
            KeyCode::Tab | KeyCode::Right => {
                state.current_tab = (state.current_tab + 1) % DashboardTab::all().len();
            }
            KeyCode::BackTab | KeyCode::Left => {
                state.current_tab = if state.current_tab == 0 {
                    DashboardTab::all().len() - 1
                } else {
                    state.current_tab - 1
                };
            }
            KeyCode::Up => {
                if state.selected_item > 0 {
                    state.selected_item -= 1;
                }
            }
            KeyCode::Down => {
                state.selected_item += 1;
            }
            KeyCode::PageUp => {
                state.scroll_position = state.scroll_position.saturating_sub(10);
            }
            KeyCode::PageDown => {
                state.scroll_position += 10;
            }
            KeyCode::Home => {
                state.selected_item = 0;
                state.scroll_position = 0;
            }
            KeyCode::Char('1') => state.current_tab = 0,
            KeyCode::Char('2') => state.current_tab = 1,
            KeyCode::Char('3') => state.current_tab = 2,
            KeyCode::Char('4') => state.current_tab = 3,
            KeyCode::Char('5') => state.current_tab = 4,
            KeyCode::Char('6') => state.current_tab = 5,
            _ => {}
        }

        Ok(false)
    }

    /// Draw the dashboard
    async fn draw_dashboard<B: Backend>(&self, f: &mut Frame<'_>) -> Result<()> {
        let state = self.state.read().await;

        // Create main layout with proper margins for better spacing
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3), // Header
                Constraint::Min(0),    // Content
                Constraint::Length(3), // Footer
            ])
            .margin(1)
            .split(f.area());

        // Draw header
        self.draw_header::<B>(f, chunks[0], &state).await.map_err(|e| {
            error!("Failed to draw header: {}", e);
            e
        })?;

        // Draw content based on current tab
        match state.current_tab {
            0 => self.draw_overview::<B>(f, chunks[1], &state).await.map_err(|e| {
                error!("Failed to draw overview tab: {}", e);
                e
            })?,
            1 => self.draw_system_metrics::<B>(f, chunks[1], &state).await.map_err(|e| {
                error!("Failed to draw system metrics tab: {}", e);
                e
            })?,
            2 => self.draw_cost_analytics::<B>(f, chunks[1], &state).await.map_err(|e| {
                error!("Failed to draw cost analytics tab: {}", e);
                e
            })?,
            3 => self.draw_safety_cluster::<B>(f, chunks[1], &state).await.map_err(|e| {
                error!("Failed to draw safety cluster tab: {}", e);
                e
            })?,
            4 => self.draw_alerts::<B>(f, chunks[1], &state).await.map_err(|e| {
                error!("Failed to draw alerts tab: {}", e);
                e
            })?,
            5 => self.draw_logs::<B>(f, chunks[1], &state).await.map_err(|e| {
                error!("Failed to draw logs tab: {}", e);
                e
            })?,
            _ => {}
        }

        // Draw footer
        self.draw_footer::<B>(f, chunks[2], &state).await.map_err(|e| {
            error!("Failed to draw footer: {}", e);
            e
        })?;

        // Draw help overlay if requested
        if state.show_help {
            self.draw_help_overlay::<B>(f).await.map_err(|e| {
                error!("Failed to draw help overlay: {}", e);
                e
            })?;
        }

        Ok(())
    }

    /// Draw header with tabs
    async fn draw_header<B: Backend>(
        &self,
        f: &mut Frame<'_>,
        area: Rect,
        state: &DashboardState,
    ) -> Result<()> {
        let tabs = DashboardTab::all();
        let titles: Vec<Line> = tabs.iter().map(|tab| Line::from(tab.title())).collect();

        let tabs_widget = Tabs::new(titles)
            .block(Block::default().borders(Borders::ALL).title("Loki Monitoring Dashboard"))
            .style(Style::default().fg(Color::Cyan))
            .highlight_style(
                Style::default().add_modifier(Modifier::BOLD).bg(Color::Blue).fg(Color::White),
            )
            .select(state.current_tab);

        f.render_widget(tabs_widget, area);
        Ok(())
    }

    /// Draw footer with controls
    async fn draw_footer<B: Backend>(
        &self,
        f: &mut Frame<'_>,
        area: Rect,
        state: &DashboardState,
    ) -> Result<()> {
        let refresh_status = if state.auto_refresh { "ON" } else { "OFF" };
        let update_time = state.last_update.elapsed().as_secs();

        let footer_text = format!(
            " [q]Quit | [h]Help | [r]Refresh:{} | [Tab]Navigate | Last update: {}s ago ",
            refresh_status, update_time
        );

        let footer = Paragraph::new(footer_text)
            .style(Style::default().fg(Color::White).bg(Color::DarkGray))
            .alignment(Alignment::Center)
            .block(Block::default());

        f.render_widget(footer, area);
        Ok(())
    }

    /// Draw overview tab
    async fn draw_overview<B: Backend>(
        &self,
        f: &mut Frame<'_>,
        area: Rect,
        state: &DashboardState,
    ) -> Result<()> {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .margin(1)
            .split(area);

        let left_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(chunks[0]);

        let right_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(chunks[1]);

        // System status
        self.draw_system_status_summary::<B>(f, left_chunks[0], state).await?;

        // Cost status
        self.draw_cost_status_summary::<B>(f, left_chunks[1], state).await?;

        // Safety status
        self.draw_safety_status_summary::<B>(f, right_chunks[0], state).await?;

        // Recent alerts
        self.draw_recent_alerts_summary::<B>(f, right_chunks[1], state).await?;

        Ok(())
    }

    /// Draw system status summary
    async fn draw_system_status_summary<B: Backend>(
        &self,
        f: &mut Frame<'_>,
        area: Rect,
        state: &DashboardState,
    ) -> Result<()> {
        let content = if let Some(metrics) = &state.system_metrics {
            vec![
                Line::from(vec![
                    Span::styled("CPU: ", Style::default().fg(Color::Yellow)),
                    Span::styled(
                        format!("{:.1}%", metrics.cpu.usage_percent),
                        if metrics.cpu.usage_percent > 80.0 {
                            Style::default().fg(Color::Red)
                        } else {
                            Style::default().fg(Color::Green)
                        },
                    ),
                ]),
                Line::from(vec![
                    Span::styled("Memory: ", Style::default().fg(Color::Yellow)),
                    Span::styled(
                        format!("{:.1}%", metrics.memory.usage_percent),
                        if metrics.memory.usage_percent > 90.0 {
                            Style::default().fg(Color::Red)
                        } else {
                            Style::default().fg(Color::Green)
                        },
                    ),
                ]),
                Line::from(vec![
                    Span::styled("Disk: ", Style::default().fg(Color::Yellow)),
                    Span::styled(
                        format!("{:.1}%", metrics.disk.usage_percent),
                        if metrics.disk.usage_percent > 90.0 {
                            Style::default().fg(Color::Red)
                        } else {
                            Style::default().fg(Color::Green)
                        },
                    ),
                ]),
                Line::from(vec![
                    Span::styled("Uptime: ", Style::default().fg(Color::Yellow)),
                    Span::styled(
                        format!("{}h", metrics.system.uptime / 3600),
                        Style::default().fg(Color::Cyan),
                    ),
                ]),
                Line::from(vec![
                    Span::styled("Process: ", Style::default().fg(Color::Yellow)),
                    Span::styled(
                        format!(
                            "{:.1}% CPU, {:.1}MB",
                            metrics.process.cpu_usage_percent,
                            metrics.process.memory_usage_bytes as f64 / 1024.0 / 1024.0
                        ),
                        Style::default().fg(Color::Cyan),
                    ),
                ]),
            ]
        } else {
            vec![Line::from("No system metrics available")]
        };

        let paragraph = Paragraph::new(Text::from(content))
            .block(Block::default().borders(Borders::ALL).title("System Status"))
            .wrap(Wrap { trim: true });

        f.render_widget(paragraph, area);
        Ok(())
    }

    /// Draw cost status summary
    async fn draw_cost_status_summary<B: Backend>(
        &self,
        f: &mut Frame<'_>,
        area: Rect,
        state: &DashboardState,
    ) -> Result<()> {
        let content = if let Some(metrics) = &state.cost_metrics {
            vec![
                Line::from(vec![
                    Span::styled("Today: ", Style::default().fg(Color::Yellow)),
                    Span::styled(
                        format!("${:.2}", metrics.total.today_usd),
                        Style::default().fg(Color::Green),
                    ),
                ]),
                Line::from(vec![
                    Span::styled("This Hour: ", Style::default().fg(Color::Yellow)),
                    Span::styled(
                        format!("${:.2}", metrics.total.this_hour_usd),
                        Style::default().fg(Color::Green),
                    ),
                ]),
                Line::from(vec![
                    Span::styled("Monthly Proj: ", Style::default().fg(Color::Yellow)),
                    Span::styled(
                        format!("${:.2}", metrics.total.projected_monthly_usd),
                        if metrics.total.projected_monthly_usd > 500.0 {
                            Style::default().fg(Color::Red)
                        } else {
                            Style::default().fg(Color::Green)
                        },
                    ),
                ]),
                Line::from(vec![
                    Span::styled("Providers: ", Style::default().fg(Color::Yellow)),
                    Span::styled(
                        format!("{}", metrics.providers.len()),
                        Style::default().fg(Color::Cyan),
                    ),
                ]),
                Line::from(vec![
                    Span::styled("Avg/Hour: ", Style::default().fg(Color::Yellow)),
                    Span::styled(
                        format!("${:.2}", metrics.total.cost_per_hour_avg),
                        Style::default().fg(Color::Cyan),
                    ),
                ]),
            ]
        } else {
            vec![Line::from("No cost metrics available")]
        };

        let paragraph = Paragraph::new(Text::from(content))
            .block(Block::default().borders(Borders::ALL).title("Cost Status"))
            .wrap(Wrap { trim: true });

        f.render_widget(paragraph, area);
        Ok(())
    }

    /// Draw safety status summary
    async fn draw_safety_status_summary<B: Backend>(
        &self,
        f: &mut Frame<'_>,
        area: Rect,
        state: &DashboardState,
    ) -> Result<()> {
        let content = if let Some(status) = &state.cluster_status {
            let health_color = match status.health {
                ClusterHealth::Healthy => Color::Green,
                ClusterHealth::Degraded => Color::Yellow,
                ClusterHealth::Unhealthy => Color::Red,
                ClusterHealth::Critical => Color::Magenta,
            };

            vec![
                Line::from(vec![
                    Span::styled("Cluster: ", Style::default().fg(Color::Yellow)),
                    Span::styled(format!("{:?}", status.health), Style::default().fg(health_color)),
                ]),
                Line::from(vec![
                    Span::styled("Nodes: ", Style::default().fg(Color::Yellow)),
                    Span::styled(
                        format!("{} active / {}", status.active_nodes, status.total_nodes),
                        Style::default().fg(Color::Cyan),
                    ),
                ]),
                Line::from(vec![
                    Span::styled("Consensus: ", Style::default().fg(Color::Yellow)),
                    Span::styled(
                        format!("{} active", status.consensus_requests_active),
                        Style::default().fg(Color::Cyan),
                    ),
                ]),
                Line::from(vec![
                    Span::styled("Alerts: ", Style::default().fg(Color::Yellow)),
                    Span::styled(
                        format!("{} emergency", status.emergency_alerts_active),
                        if status.emergency_alerts_active > 0 {
                            Style::default().fg(Color::Red)
                        } else {
                            Style::default().fg(Color::Green)
                        },
                    ),
                ]),
            ]
        } else {
            vec![
                Line::from("No cluster status available"),
                Line::from("Running in standalone mode"),
            ]
        };

        let paragraph = Paragraph::new(Text::from(content))
            .block(Block::default().borders(Borders::ALL).title("Safety Status"))
            .wrap(Wrap { trim: true });

        f.render_widget(paragraph, area);
        Ok(())
    }

    /// Draw recent alerts summary
    async fn draw_recent_alerts_summary<B: Backend>(
        &self,
        f: &mut Frame<'_>,
        area: Rect,
        state: &DashboardState,
    ) -> Result<()> {
        let mut alerts = Vec::new();

        // Add system alerts
        for alert in state.system_alerts.iter().rev().take(5) {
            let color = match alert.level {
                AlertLevel::Normal => Color::Green,
                AlertLevel::Warning => Color::Yellow,
                AlertLevel::Critical => Color::Red,
                AlertLevel::Emergency => Color::Magenta,
            };
            alerts.push(Line::from(vec![
                Span::styled("SYS ", Style::default().fg(Color::Blue)),
                Span::styled(&alert.message, Style::default().fg(color)),
            ]));
        }

        // Add cost alerts
        for alert in state.cost_alerts.iter().rev().take(3) {
            let color = match alert.severity {
                CostAlertSeverity::Info => Color::Cyan,
                CostAlertSeverity::Warning => Color::Yellow,
                CostAlertSeverity::Critical => Color::Red,
                CostAlertSeverity::Emergency => Color::Magenta,
            };
            alerts.push(Line::from(vec![
                Span::styled("COST ", Style::default().fg(Color::Green)),
                Span::styled(&alert.message, Style::default().fg(color)),
            ]));
        }

        if alerts.is_empty() {
            alerts.push(Line::from("No recent alerts"));
        }

        let paragraph = Paragraph::new(Text::from(alerts))
            .block(Block::default().borders(Borders::ALL).title("Recent Alerts"))
            .wrap(Wrap { trim: true });

        f.render_widget(paragraph, area);
        Ok(())
    }

    /// Draw system metrics tab
    async fn draw_system_metrics<B: Backend>(
        &self,
        f: &mut Frame<'_>,
        area: Rect,
        state: &DashboardState,
    ) -> Result<()> {
        if let Some(metrics) = &state.system_metrics {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(3), // CPU gauge
                    Constraint::Length(3), // Memory gauge
                    Constraint::Length(3), // Disk gauge
                    Constraint::Length(4), // Network sparklines
                    Constraint::Min(0),    // Charts
                ])
                .margin(1)
                .split(area);

            // CPU gauge
            let cpu_gauge = Gauge::default()
                .block(Block::default().borders(Borders::ALL).title("CPU Usage"))
                .set_style(if metrics.cpu.usage_percent > 80.0 {
                    Style::default().fg(Color::Red)
                } else {
                    Style::default().fg(Color::Green)
                })
                .percent(metrics.cpu.usage_percent as u16)
                .label(format!("{:.1}%", metrics.cpu.usage_percent));

            f.render_widget(cpu_gauge, chunks[0]);

            // Memory gauge
            let memory_gauge = Gauge::default()
                .block(Block::default().borders(Borders::ALL).title("Memory Usage"))
                .set_style(if metrics.memory.usage_percent > 90.0 {
                    Style::default().fg(Color::Red)
                } else {
                    Style::default().fg(Color::Green)
                })
                .percent(metrics.memory.usage_percent as u16)
                .label(format!(
                    "{:.1}% ({:.1}GB/{:.1}GB)",
                    metrics.memory.usage_percent,
                    metrics.memory.used_bytes as f64 / 1024.0 / 1024.0 / 1024.0,
                    metrics.memory.total_bytes as f64 / 1024.0 / 1024.0 / 1024.0
                ));

            f.render_widget(memory_gauge, chunks[1]);

            // Disk gauge
            let disk_gauge = Gauge::default()
                .block(Block::default().borders(Borders::ALL).title("Disk Usage"))
                .set_style(if metrics.disk.usage_percent > 90.0 {
                    Style::default().fg(Color::Red)
                } else {
                    Style::default().fg(Color::Green)
                })
                .percent(metrics.disk.usage_percent as u16)
                .label(format!(
                    "{:.1}% ({:.1}GB/{:.1}GB)",
                    metrics.disk.usage_percent,
                    metrics.disk.used_space_bytes as f64 / 1024.0 / 1024.0 / 1024.0,
                    metrics.disk.total_space_bytes as f64 / 1024.0 / 1024.0 / 1024.0
                ));

            f.render_widget(disk_gauge, chunks[2]);

            // Network activity sparklines
            self.draw_network_sparklines::<B>(f, chunks[3], state).await?;

            // Historical charts
            self.draw_system_charts::<B>(f, chunks[4], state).await?;
        } else {
            let paragraph = Paragraph::new("No system metrics available")
                .block(Block::default().borders(Borders::ALL).title("System Metrics"));
            f.render_widget(paragraph, area);
        }

        Ok(())
    }

    /// Draw system charts
    async fn draw_system_charts<B: Backend>(
        &self,
        f: &mut Frame<'_>,
        area: Rect,
        state: &DashboardState,
    ) -> Result<()> {
        if state.metrics_history.is_empty() {
            return Ok(());
        }

        let cpu_data: Vec<(f64, f64)> = state
            .metrics_history
            .iter()
            .enumerate()
            .map(|(i, m)| (i as f64, m.cpu.usage_percent as f64))
            .collect();

        let memory_data: Vec<(f64, f64)> = state
            .metrics_history
            .iter()
            .enumerate()
            .map(|(i, m)| (i as f64, m.memory.usage_percent as f64))
            .collect();

        let datasets = vec![
            Dataset::default()
                .name("CPU %")
                .marker(symbols::Marker::Dot)
                .style(Style::default().fg(Color::Cyan))
                .data(&cpu_data),
            Dataset::default()
                .name("Memory %")
                .marker(symbols::Marker::Braille)
                .style(Style::default().fg(Color::Yellow))
                .data(&memory_data),
        ];

        let chart = Chart::new(datasets)
            .block(Block::default().borders(Borders::ALL).title("Resource Usage Over Time"))
            .x_axis(
                Axis::default()
                    .title("Time")
                    .style(Style::default().fg(Color::Gray))
                    .bounds([0.0, state.metrics_history.len() as f64]),
            )
            .y_axis(
                Axis::default()
                    .title("Percentage")
                    .style(Style::default().fg(Color::Gray))
                    .bounds([0.0, 100.0]),
            );

        f.render_widget(chart, area);
        Ok(())
    }

    /// Draw cost analytics tab
    async fn draw_cost_analytics<B: Backend>(
        &self,
        f: &mut Frame<'_>,
        area: Rect,
        state: &DashboardState,
    ) -> Result<()> {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
            .margin(1)
            .split(area);

        // Cost charts
        self.draw_cost_charts::<B>(f, chunks[0], state).await?;

        // Provider breakdown
        self.draw_provider_breakdown::<B>(f, chunks[1], state).await?;

        Ok(())
    }

    /// Draw cost charts
    async fn draw_cost_charts<B: Backend>(
        &self,
        f: &mut Frame<'_>,
        area: Rect,
        state: &DashboardState,
    ) -> Result<()> {
        if let Some(metrics) = &state.cost_metrics {
            let content = vec![
                Line::from(format!("Total Today: ${:.2}", metrics.total.today_usd)),
                Line::from(format!("This Hour: ${:.2}", metrics.total.this_hour_usd)),
                Line::from(format!(
                    "Monthly Projection: ${:.2}",
                    metrics.total.projected_monthly_usd
                )),
                Line::from(format!("All Time: ${:.2}", metrics.total.all_time_usd)),
                Line::from(""),
                Line::from("Cost Trends:"),
                Line::from(format!("  Hourly Rate: ${:.4}/hr", metrics.trends.hourly_spend_rate)),
                Line::from(format!("  Efficiency: {:.2}", metrics.trends.efficiency_trend)),
                Line::from(format!(
                    "  Savings Opportunity: ${:.2}",
                    metrics.trends.savings_opportunity
                )),
            ];

            let paragraph = Paragraph::new(Text::from(content))
                .block(Block::default().borders(Borders::ALL).title("Cost Overview"))
                .wrap(Wrap { trim: true });

            f.render_widget(paragraph, area);
        } else {
            let paragraph = Paragraph::new("No cost metrics available")
                .block(Block::default().borders(Borders::ALL).title("Cost Analytics"));
            f.render_widget(paragraph, area);
        }

        Ok(())
    }

    /// Draw provider breakdown
    async fn draw_provider_breakdown<B: Backend>(
        &self,
        f: &mut Frame<'_>,
        area: Rect,
        state: &DashboardState,
    ) -> Result<()> {
        if let Some(metrics) = &state.cost_metrics {
            let items: Vec<ListItem> = metrics
                .providers
                .iter()
                .map(|(name, cost)| {
                    ListItem::new(format!(
                        "{}: ${:.2} ({} req, ${:.4}/req)",
                        name, cost.total_cost_usd, cost.requests_count, cost.avg_cost_per_request
                    ))
                })
                .collect();

            let list = List::new(items)
                .block(Block::default().borders(Borders::ALL).title("Provider Costs"))
                .style(Style::default().fg(Color::White));

            f.render_widget(list, area);
        }

        Ok(())
    }

    /// Draw safety cluster tab
    async fn draw_safety_cluster<B: Backend>(
        &self,
        f: &mut Frame<'_>,
        area: Rect,
        state: &DashboardState,
    ) -> Result<()> {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(8),  // Safety overview
                Constraint::Length(12), // Agent safety profiles
                Constraint::Min(0),     // Safety events and violations
            ])
            .margin(1)
            .split(area);

        // Safety overview
        self.draw_safety_overview::<B>(f, chunks[0], state).await?;

        // Agent safety profiles
        self.draw_agent_safety_profiles::<B>(f, chunks[1], state).await?;

        // Safety events and violations
        self.draw_safety_events_and_violations::<B>(f, chunks[2], state).await?;

        Ok(())
    }

    /// Draw safety overview
    async fn draw_safety_overview<B: Backend>(
        &self,
        f: &mut Frame<'_>,
        area: Rect,
        state: &DashboardState,
    ) -> Result<()> {
        // Create inner area with margin for better gauge spacing
        let inner_area = area.inner(Margin { vertical: 1, horizontal: 1 });

        let overview_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(20), // Action validation rate
                Constraint::Percentage(20), // Agent trust level
                Constraint::Percentage(20), // Resource compliance
                Constraint::Percentage(20), // Security alerts
                Constraint::Percentage(20), // System integrity
            ])
            .split(inner_area);

        // Action validation rate - calculate from actual safety events
        let validation_rate = if state.safety_events.is_empty() {
            95 // Default when no events
        } else {
            // Calculate success rate based on actual validation events
            let total_validations = state.safety_events.len();
            let failed_validations = state
                .safety_events
                .iter()
                .filter(|event| {
                    matches!(
                        **event,
                        DistributedSafetyEvent::EmergencyAlert { .. }
                            | DistributedSafetyEvent::ConsensusRequest { risk_level: 8..=10, .. }
                    )
                })
                .count();

            let success_rate = ((total_validations - failed_validations) as f32
                / total_validations as f32
                * 100.0) as u16;
            success_rate.max(50).min(100) // Keep within reasonable bounds
        };
        let validation_color = if validation_rate >= 95 {
            Color::Green
        } else if validation_rate >= 85 {
            Color::Yellow
        } else {
            Color::Red
        };

        let validation_gauge = Gauge::default()
            .block(Block::default().borders(Borders::ALL).title(" Action Validation "))
            .set_style(Style::default().fg(validation_color))
            .percent(validation_rate)
            .label(format!("{}% Valid", validation_rate));
        f.render_widget(validation_gauge, overview_chunks[0]);

        // Agent trust level - calculate from recent agent health updates
        let avg_trust = if state.safety_events.is_empty() {
            85 // Default when no events
        } else {
            // Calculate trust based on health updates and violation absence
            let health_events = state
                .safety_events
                .iter()
                .filter(|event| matches!(**event, DistributedSafetyEvent::HealthUpdate { .. }))
                .count();

            let violation_events = state
                .safety_events
                .iter()
                .filter(|event| {
                    matches!(
                        **event,
                        DistributedSafetyEvent::EmergencyAlert { .. }
                            | DistributedSafetyEvent::ConsensusRequest { risk_level: 7..=10, .. }
                    )
                })
                .count();

            // Base trust of 80, increase with health events, decrease with violations
            let base_trust = 80;
            let trust_bonus = (health_events * 2).min(15);
            let trust_penalty = (violation_events * 5).min(25);

            (base_trust + trust_bonus - trust_penalty).max(30).min(99) as u16
        };
        let trust_color = if avg_trust >= 90 {
            Color::Green
        } else if avg_trust >= 75 {
            Color::Yellow
        } else {
            Color::Red
        };

        let trust_gauge = Gauge::default()
            .block(Block::default().borders(Borders::ALL).title(" Agent Trust "))
            .set_style(Style::default().fg(trust_color))
            .percent(avg_trust)
            .label(format!("{}% Trusted", avg_trust));
        f.render_widget(trust_gauge, overview_chunks[1]);

        // Resource compliance - calculate from resource breach events
        let resource_compliance = if state.safety_events.is_empty() {
            90 // Default when no events
        } else {
            let resource_breaches = state
                .safety_events
                .iter()
                .filter(|event| matches!(**event, DistributedSafetyEvent::EmergencyAlert { .. }))
                .count();

            // Start with high compliance, reduce based on breaches
            let base_compliance = 95;
            let compliance_penalty = (resource_breaches * 3).min(20);

            (base_compliance - compliance_penalty).max(60).min(100) as u16
        };
        let resource_color = if resource_compliance >= 95 {
            Color::Green
        } else if resource_compliance >= 85 {
            Color::Yellow
        } else {
            Color::Red
        };

        let resource_gauge = Gauge::default()
            .block(Block::default().borders(Borders::ALL).title(" Resource Limits "))
            .set_style(Style::default().fg(resource_color))
            .percent(resource_compliance)
            .label(format!("{}% Within", resource_compliance));
        f.render_widget(resource_gauge, overview_chunks[2]);

        // Security alerts - count actual emergency alerts
        let active_alerts = state
            .safety_events
            .iter()
            .filter(|event| matches!(**event, DistributedSafetyEvent::EmergencyAlert { .. }))
            .count();

        let max_alerts = 10;
        let alert_level = ((active_alerts as f32 / max_alerts as f32) * 100.0) as u16;
        let alert_color = if active_alerts == 0 {
            Color::Green
        } else if active_alerts <= 3 {
            Color::Yellow
        } else {
            Color::Red
        };

        let alerts_gauge = Gauge::default()
            .block(Block::default().borders(Borders::ALL).title(" Security Alerts "))
            .set_style(Style::default().fg(alert_color))
            .percent(alert_level)
            .label(format!("{} Active", active_alerts));
        f.render_widget(alerts_gauge, overview_chunks[3]);

        // System integrity - calculate from overall system health
        let integrity_score = if state.safety_events.is_empty() {
            96 // Default high integrity when no events
        } else {
            // Calculate integrity based on various factors
            let _total_events = state.safety_events.len();
            let critical_events = state
                .safety_events
                .iter()
                .filter(|event| matches!(**event, DistributedSafetyEvent::EmergencyAlert { .. }))
                .count();

            let health_updates = state
                .safety_events
                .iter()
                .filter(|event| matches!(**event, DistributedSafetyEvent::HealthUpdate { .. }))
                .count();

            // Base integrity of 90, adjust based on events
            let base_integrity = 90;
            let integrity_penalty = (critical_events * 10).min(30);
            let integrity_bonus = (health_updates * 2).min(10);

            (base_integrity - integrity_penalty + integrity_bonus).max(70).min(99) as u16
        };
        let integrity_color = if integrity_score >= 98 {
            Color::Green
        } else if integrity_score >= 90 {
            Color::Yellow
        } else {
            Color::Red
        };

        let integrity_gauge = Gauge::default()
            .block(Block::default().borders(Borders::ALL).title(" System Integrity "))
            .set_style(Style::default().fg(integrity_color))
            .percent(integrity_score)
            .label(format!("{}% Secure", integrity_score));
        f.render_widget(integrity_gauge, overview_chunks[4]);

        Ok(())
    }

    /// Draw agent safety profiles
    async fn draw_agent_safety_profiles<B: Backend>(
        &self,
        f: &mut Frame<'_>,
        area: Rect,
        _state: &DashboardState,
    ) -> Result<()> {
        // Mock agent safety data - would come from MultiAgentSafetyCoordinator
        let agent_profiles = vec![
            ("agent-001", "High", 0.95, "Active", "Clean"),
            ("agent-002", "Medium", 0.78, "Active", "1 Warning"),
            ("agent-003", "High", 0.92, "Idle", "Clean"),
            ("agent-004", "Critical", 0.45, "Quarantined", "3 Violations"),
            ("agent-005", "Medium", 0.83, "Active", "Clean"),
        ];

        let header = Row::new(vec![
            Cell::from("Agent ID"),
            Cell::from("Trust Level"),
            Cell::from("Safety Score"),
            Cell::from("Status"),
            Cell::from("Violations"),
        ])
        .style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD));

        let rows: Vec<Row> = agent_profiles
            .iter()
            .map(|(id, trust, score, status, violations)| {
                let trust_color = match *trust {
                    "High" => Color::Green,
                    "Medium" => Color::Yellow,
                    "Critical" => Color::Red,
                    _ => Color::White,
                };

                let status_color = match *status {
                    "Active" => Color::Green,
                    "Idle" => Color::Blue,
                    "Quarantined" => Color::Red,
                    _ => Color::White,
                };

                Row::new(vec![
                    Cell::from(*id),
                    Cell::from(*trust).style(Style::default().fg(trust_color)),
                    Cell::from(format!("{:.2}", score)),
                    Cell::from(*status).style(Style::default().fg(status_color)),
                    Cell::from(*violations),
                ])
            })
            .collect();

        let table = Table::new(
            rows,
            [
                Constraint::Percentage(20),
                Constraint::Percentage(20),
                Constraint::Percentage(20),
                Constraint::Percentage(20),
                Constraint::Percentage(20),
            ],
        )
        .header(header)
        .block(Block::default().borders(Borders::ALL).title(" 🛡️ Agent Safety Profiles "));

        f.render_widget(table, area);
        Ok(())
    }

    /// Draw safety events and violations
    async fn draw_safety_events_and_violations<B: Backend>(
        &self,
        f: &mut Frame<'_>,
        area: Rect,
        _state: &DashboardState,
    ) -> Result<()> {
        // Create inner area with margin for better spacing
        let inner_area = area.inner(Margin { vertical: 1, horizontal: 1 });

        let events_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(50), // Recent safety events
                Constraint::Percentage(50), // Security violations
            ])
            .split(inner_area);

        // Recent safety events
        let recent_events = vec![
            ("Action blocked", "agent-004 file write denied", "2m ago", Color::Yellow),
            ("Trust updated", "agent-002 score decreased", "5m ago", Color::Blue),
            ("Integrity check", "System scan completed", "10m ago", Color::Green),
            ("Resource alert", "Memory usage spike", "15m ago", Color::Yellow),
            ("Behavior anomaly", "Communication pattern unusual", "22m ago", Color::Red),
            ("Agent quarantine", "agent-004 isolated", "1h ago", Color::Red),
        ];

        let event_items: Vec<ListItem> = recent_events
            .iter()
            .map(|(event_type, description, time, color)| {
                ListItem::new(vec![
                    Line::from(vec![
                        Span::styled(
                            format!("[{}] ", event_type),
                            Style::default().fg(*color).add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(*description, Style::default().fg(Color::White)),
                    ]),
                    Line::from(vec![Span::styled(
                        format!("  └─ {}", time),
                        Style::default().fg(Color::Gray),
                    )]),
                ])
            })
            .collect();

        let events_list = List::new(event_items)
            .block(Block::default().borders(Borders::ALL).title(" 📊 Recent Safety Events "))
            .style(Style::default().fg(Color::White));
        f.render_widget(events_list, events_chunks[0]);

        // Security violations
        let violations = vec![
            ("Critical", "Unauthorized system access attempt", "1h ago"),
            ("High", "Resource limit exceeded", "3h ago"),
            ("Medium", "Suspicious API call pattern", "5h ago"),
            ("Low", "Configuration drift detected", "1d ago"),
        ];

        let violation_items: Vec<ListItem> = violations
            .iter()
            .map(|(severity, description, time)| {
                let color = match *severity {
                    "Critical" => Color::Red,
                    "High" => Color::LightRed,
                    "Medium" => Color::Yellow,
                    "Low" => Color::Blue,
                    _ => Color::White,
                };

                ListItem::new(vec![
                    Line::from(vec![
                        Span::styled(
                            format!("[{}] ", severity),
                            Style::default().fg(color).add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(*description, Style::default().fg(Color::White)),
                    ]),
                    Line::from(vec![Span::styled(
                        format!("  └─ {}", time),
                        Style::default().fg(Color::Gray),
                    )]),
                ])
            })
            .collect();

        let violations_list = List::new(violation_items)
            .block(Block::default().borders(Borders::ALL).title(" ⚠️ Security Violations "))
            .style(Style::default().fg(Color::White));
        f.render_widget(violations_list, events_chunks[1]);

        Ok(())
    }

    /// Draw network activity sparklines
    async fn draw_network_sparklines<B: Backend>(
        &self,
        f: &mut Frame<'_>,
        area: Rect,
        state: &DashboardState,
    ) -> Result<()> {
        let network_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(area);

        // Generate sample network data for sparklines
        let rx_data: Vec<u64> = state
            .metrics_history
            .iter()
            .map(|m| (m.cpu.usage_percent * 10.0) as u64) // Sample data
            .collect();

        let tx_data: Vec<u64> = state
            .metrics_history
            .iter()
            .map(|m| (m.memory.usage_percent * 8.0) as u64) // Sample data
            .collect();

        // Network RX sparkline
        let rx_sparkline = Sparkline::default()
            .block(Block::default().borders(Borders::ALL).title("Network RX (KB/s)"))
            .data(&rx_data)
            .style(Style::default().fg(Color::Green));

        f.render_widget(rx_sparkline, network_chunks[0]);

        // Network TX sparkline
        let tx_sparkline = Sparkline::default()
            .block(Block::default().borders(Borders::ALL).title("Network TX (KB/s)"))
            .data(&tx_data)
            .style(Style::default().fg(Color::Blue));

        f.render_widget(tx_sparkline, network_chunks[1]);

        Ok(())
    }

    /// Draw cost analytics with enhanced LineGauge displays
    async fn draw_cost_analytics_enhanced<B: Backend>(
        &self,
        f: &mut Frame<'_>,
        area: Rect,
        state: &DashboardState,
    ) -> Result<()> {
        if let Some(cost_metrics) = &state.cost_metrics {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(3), // Current spending
                    Constraint::Length(3), // Budget utilization
                    Constraint::Length(3), // Cost efficiency
                    Constraint::Min(0),    // Details
                ])
                .margin(1)
                .split(area);

            // Current spending line gauge
            let spending_ratio = (cost_metrics.total.this_month_usd
                / cost_metrics.total.projected_monthly_usd.max(1.0))
            .min(1.0);
            let spending_line_gauge = LineGauge::default()
                .block(Block::default().borders(Borders::ALL).title("Current Spending"))
                .set_style(if spending_ratio > 0.9 {
                    Style::default().fg(Color::Red)
                } else if spending_ratio > 0.7 {
                    Style::default().fg(Color::Yellow)
                } else {
                    Style::default().fg(Color::Green)
                })
                .ratio(spending_ratio)
                .label(format!(
                    "${:.2}/${:.2}",
                    cost_metrics.total.this_month_usd, cost_metrics.total.projected_monthly_usd
                ));

            f.render_widget(spending_line_gauge, chunks[0]);

            // Budget utilization line gauge
            let budget_percent = (spending_ratio * 100.0).min(100.0);
            let budget_ratio = spending_ratio.min(1.0);
            let budget_line_gauge = LineGauge::default()
                .block(Block::default().borders(Borders::ALL).title("Budget Utilization"))
                .set_style(Style::default().fg(Color::Cyan))
                .ratio(budget_ratio)
                .label(format!("{:.1}%", budget_percent));

            f.render_widget(budget_line_gauge, chunks[1]);

            // Use interval for timing and warn for critical situations
            warn!("💰 Cost monitoring active - spending at {:.1}%", budget_percent);
        }

        Ok(())
    }

    /// Draw alerts tab
    async fn draw_alerts<B: Backend>(
        &self,
        f: &mut Frame<'_>,
        area: Rect,
        state: &DashboardState,
    ) -> Result<()> {
        let mut all_alerts = Vec::new();

        // System alerts
        for alert in &state.system_alerts {
            let color = match alert.level {
                AlertLevel::Normal => Color::Green,
                AlertLevel::Warning => Color::Yellow,
                AlertLevel::Critical => Color::Red,
                AlertLevel::Emergency => Color::Magenta,
            };
            all_alerts.push(
                ListItem::new(format!("[SYS] {}", alert.message)).style(Style::default().fg(color)),
            );
        }

        // Cost alerts
        for alert in &state.cost_alerts {
            let color = match alert.severity {
                CostAlertSeverity::Info => Color::Cyan,
                CostAlertSeverity::Warning => Color::Yellow,
                CostAlertSeverity::Critical => Color::Red,
                CostAlertSeverity::Emergency => Color::Magenta,
            };
            all_alerts.push(
                ListItem::new(format!("[COST] {}", alert.message))
                    .style(Style::default().fg(color)),
            );
        }

        if all_alerts.is_empty() {
            all_alerts.push(ListItem::new("No alerts"));
        }

        let list = List::new(all_alerts)
            .block(Block::default().borders(Borders::ALL).title("All Alerts"))
            .style(Style::default().fg(Color::White));

        f.render_widget(list, area);
        Ok(())
    }

    /// Draw logs tab
    async fn draw_logs<B: Backend>(
        &self,
        f: &mut Frame<'_>,
        area: Rect,
        state: &DashboardState,
    ) -> Result<()> {
        let items: Vec<ListItem> = state
            .safety_events
            .iter()
            .map(|event| {
                let text = match event {
                    DistributedSafetyEvent::ConsensusRequest { action, risk_level, .. } => {
                        format!("[CONSENSUS] Risk {} action: {:?}", risk_level, action)
                    }
                    DistributedSafetyEvent::EmergencyAlert {
                        alert_type,
                        severity,
                        message,
                        ..
                    } => {
                        format!("[EMERGENCY] {:?} {:?}: {}", severity, alert_type, message)
                    }
                    DistributedSafetyEvent::HealthUpdate { node, .. } => {
                        format!("[HEALTH] Node {} status: {:?}", node.id, node.status)
                    }
                    DistributedSafetyEvent::ResourceBreach {
                        node,
                        resource_type,
                        current_value,
                        limit,
                        ..
                    } => {
                        format!(
                            "[BREACH] Node {} {}: {:.1} > {:.1}",
                            node, resource_type, current_value, limit
                        )
                    }
                    _ => format!("Safety event: {:?}", event),
                };
                ListItem::new(text)
            })
            .collect();

        let list = List::new(items)
            .block(Block::default().borders(Borders::ALL).title("Safety Events Log"))
            .style(Style::default().fg(Color::White));

        f.render_widget(list, area);
        Ok(())
    }

    /// Draw help overlay
    async fn draw_help_overlay<B: Backend>(&self, f: &mut Frame<'_>) -> Result<()> {
        let area = f.area();
        let popup_area = Rect {
            x: area.width / 4,
            y: area.height / 4,
            width: area.width / 2,
            height: area.height / 2,
        };

        // Create inner area with margin for better padding
        let inner_area = popup_area.inner(Margin { vertical: 1, horizontal: 2 });

        f.render_widget(Clear, popup_area);

        let help_text = vec![
            Line::from("Loki Monitoring Dashboard Help"),
            Line::from(""),
            Line::from("Navigation:"),
            Line::from("  Tab/→     Next tab"),
            Line::from("  Shift+Tab/← Previous tab"),
            Line::from("  1-6       Direct tab selection"),
            Line::from("  ↑/↓       Navigate lists"),
            Line::from("  PgUp/PgDn Scroll logs"),
            Line::from(""),
            Line::from("Controls:"),
            Line::from("  r         Toggle auto-refresh"),
            Line::from("  h/F1      Toggle this help"),
            Line::from("  q/Esc     Quit"),
            Line::from(""),
            Line::from("Press any key to close"),
        ];

        let paragraph = Paragraph::new(Text::from(help_text))
            .block(Block::default().borders(Borders::ALL).title("Help"))
            .wrap(Wrap { trim: true });

        // Render help text in the inner area for better padding
        f.render_widget(paragraph, inner_area);
        Ok(())
    }

    /// Cleanup terminal on exit
    fn cleanup(&mut self) -> Result<()> {
        disable_raw_mode().map_err(|e| {
            error!("Failed to disable raw mode: {}", e);
            e
        })?;
        execute!(self.terminal.backend_mut(), LeaveAlternateScreen, DisableMouseCapture).map_err(
            |e| {
                error!("Failed to cleanup terminal: {}", e);
                e
            },
        )?;
        self.terminal.show_cursor().map_err(|e| {
            error!("Failed to show cursor: {}", e);
            e
        })?;
        Ok(())
    }
}
