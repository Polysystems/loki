//! Story visualization tab for TUI

use crate::story::{
    StoryEngine, StoryAnalytics, StoryStatistics, Story, StoryStatus, PlotType,
    StoryArc, StoryArcId, PlotPoint, PlotPointId, StoryType, engine::StoryEvent,
};
use crossterm::event::{KeyCode, KeyModifiers};
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{
        BarChart, Block, Borders, List, ListItem, ListState,
        Paragraph, Row, Table, Tabs, Wrap,
    },
    Frame,
};
use std::sync::Arc;

/// Story visualization tab
#[derive(Debug)]
pub struct StoriesTab {
    /// Story engine reference
    story_engine: Option<Arc<StoryEngine>>,
    
    /// Story analytics
    analytics: Option<StoryAnalytics>,
    
    /// Cached statistics
    cached_stats: Option<StoryStatistics>,
    
    /// All stories from the engine
    stories: Vec<Story>,
    
    /// Selected story index
    selected_story: usize,
    
    /// List state for story selection
    list_state: ListState,
    
    /// Currently selected story details
    selected_story_details: Option<Story>,
    
    /// Current view mode
    view_mode: StoryViewMode,
    
    /// Sub-tabs for different views
    sub_tabs: Vec<&'static str>,
    current_sub_tab: usize,
    
    /// Story creation form state
    creating_story: bool,
    story_form: StoryCreationForm,
    
    /// Flag to indicate refresh needed
    needs_refresh: bool,
    
    /// Arc creation form state
    creating_arc: bool,
    arc_form: ArcCreationForm,
    
    /// Selected arc for task operations
    selected_arc_index: usize,
    
    /// Task list state for the tasks view
    task_list_state: ListState,
    
    /// Event receiver for story updates
    event_rx: Option<tokio::sync::broadcast::Receiver<StoryEvent>>,
    
    /// Last update time for rate limiting
    last_update: std::time::Instant,
    
    /// Story autonomy configuration
    story_autonomy_enabled: bool,
    autonomy_config: StoryAutonomyConfig,
    config_selected_field: ConfigField,
}

/// Story autonomy configuration
#[derive(Debug, Clone)]
pub struct StoryAutonomyConfig {
    pub auto_maintenance: bool,
    pub pr_review_enabled: bool,
    pub bug_detection_enabled: bool,
    pub quality_monitoring: bool,
    pub performance_optimization: bool,
    pub security_scanning: bool,
    pub test_generation: bool,
    pub refactoring_enabled: bool,
    pub dependency_updates: bool,
    
    pub maintenance_interval_hours: u32,
    pub pr_review_threshold: f32,
    pub quality_threshold: f32,
}

impl Default for StoryAutonomyConfig {
    fn default() -> Self {
        Self {
            auto_maintenance: false,
            pr_review_enabled: true,
            bug_detection_enabled: true,
            quality_monitoring: true,
            performance_optimization: false,
            security_scanning: true,
            test_generation: false,
            refactoring_enabled: false,
            dependency_updates: false,
            
            maintenance_interval_hours: 24,
            pr_review_threshold: 0.7,
            quality_threshold: 0.8,
        }
    }
}

/// Configuration field selection
#[derive(Debug, Clone, Copy, PartialEq)]
enum ConfigField {
    AutoMaintenance,
    PrReview,
    BugDetection,
    QualityMonitoring,
    PerformanceOptimization,
    SecurityScanning,
    TestGeneration,
    Refactoring,
    DependencyUpdates,
    MaintenanceInterval,
    PrReviewThreshold,
    QualityThreshold,
}

/// Different view modes for stories
#[derive(Debug, Clone, Copy, PartialEq)]
enum StoryViewMode {
    Overview,
    Timeline,
    Analytics,
    Tasks,
    Relationships,
    Configuration,
}

impl Default for StoriesTab {
    fn default() -> Self {
        let mut list_state = ListState::default();
        list_state.select(Some(0));
        let mut task_list_state = ListState::default();
        task_list_state.select(Some(0));
        
        Self {
            story_engine: None,
            analytics: None,
            cached_stats: None,
            stories: Vec::new(),
            selected_story: 0,
            list_state,
            selected_story_details: None,
            view_mode: StoryViewMode::Overview,
            sub_tabs: vec!["Overview", "Timeline", "Analytics", "Tasks", "Relationships", "Configuration"],
            current_sub_tab: 0,
            creating_story: false,
            story_form: StoryCreationForm::default(),
            needs_refresh: false,
            creating_arc: false,
            arc_form: ArcCreationForm::default(),
            selected_arc_index: 0,
            task_list_state,
            event_rx: None,
            last_update: std::time::Instant::now(),
            story_autonomy_enabled: false,
            autonomy_config: StoryAutonomyConfig::default(),
            config_selected_field: ConfigField::AutoMaintenance,
        }
    }
}

/// Story creation form
#[derive(Debug, Clone, Default)]
struct StoryCreationForm {
    title: String,
    story_type: StoryTypeSelection,
    description: String,
    current_field: StoryFormField,
}

/// Arc creation form
#[derive(Debug, Clone, Default)]
struct ArcCreationForm {
    title: String,
    description: String,
    current_field: ArcFormField,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum ArcFormField {
    Title,
    Description,
}

impl Default for ArcFormField {
    fn default() -> Self {
        ArcFormField::Title
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum StoryFormField {
    Title,
    Type,
    Description,
}

impl Default for StoryFormField {
    fn default() -> Self {
        StoryFormField::Title
    }
}

#[derive(Debug, Clone, PartialEq)]
enum StoryTypeSelection {
    Feature,
    Bug,
    Task,
    Epic,
    Performance,
    Documentation,
    Testing,
    Research,
}

impl Default for StoryTypeSelection {
    fn default() -> Self {
        StoryTypeSelection::Feature
    }
}

impl Clone for StoriesTab {
    fn clone(&self) -> Self {
        Self {
            story_engine: self.story_engine.clone(),
            analytics: self.analytics.clone(),
            cached_stats: self.cached_stats.clone(),
            stories: self.stories.clone(),
            selected_story: self.selected_story,
            list_state: self.list_state.clone(),
            selected_story_details: self.selected_story_details.clone(),
            view_mode: self.view_mode,
            sub_tabs: self.sub_tabs.clone(),
            current_sub_tab: self.current_sub_tab,
            creating_story: self.creating_story,
            story_form: self.story_form.clone(),
            needs_refresh: self.needs_refresh,
            creating_arc: self.creating_arc,
            arc_form: self.arc_form.clone(),
            selected_arc_index: self.selected_arc_index,
            task_list_state: self.task_list_state.clone(),
            event_rx: None, // Can't clone receiver, will need to re-subscribe
            last_update: self.last_update.clone(),
            story_autonomy_enabled: self.story_autonomy_enabled,
            autonomy_config: self.autonomy_config.clone(),
            config_selected_field: self.config_selected_field,
        }
    }
}

impl StoriesTab {
    /// Set the story engine
    pub fn set_story_engine(&mut self, engine: Arc<StoryEngine>) {
        self.analytics = Some(StoryAnalytics::new(engine.clone()));
        
        // Subscribe to story events
        self.event_rx = Some(engine.subscribe());
        
        self.story_engine = Some(engine.clone());
        self.refresh_stories();
    }
    
    /// Refresh the stories list from the engine
    pub fn refresh_stories(&mut self) {
        if let Some(engine) = &self.story_engine {
            self.stories = engine.get_stories_by_type(|_| true);
            
            // Update selected story details if we have a selection
            if let Some(selected) = self.list_state.selected() {
                if selected < self.stories.len() {
                    self.selected_story_details = Some(self.stories[selected].clone());
                }
            }
        }
    }
    
    /// Get the count of stories
    pub fn story_count(&self) -> usize {
        self.stories.len()
    }
    
    /// Get the story autonomy configuration
    pub fn autonomy_config(&self) -> &StoryAutonomyConfig {
        &self.autonomy_config
    }
    
    /// Set story autonomy auto-maintenance status
    pub fn set_autonomy_auto_maintenance(&mut self, enabled: bool) {
        self.autonomy_config.auto_maintenance = enabled;
    }
    
    /// Update cached statistics
    pub async fn update_stats(&mut self) {
        if let Some(analytics) = &self.analytics {
            if let Ok(stats) = analytics.get_statistics().await {
                self.cached_stats = Some(stats);
            }
        }
    }
    
    /// Process story events from the event stream
    fn process_events(&mut self) {
        let mut needs_update = false;
        
        if let Some(rx) = &mut self.event_rx {
            // Try to receive events without blocking
            while let Ok(event) = rx.try_recv() {
                match event {
                    StoryEvent::StoryCreated(_) |
                    StoryEvent::StoryUpdated(_) => {
                        // Schedule a refresh on next render
                        self.needs_refresh = true;
                    }
                    StoryEvent::ArcCompleted(story_id, _) |
                    StoryEvent::PlotPointAdded(story_id, _) => {
                        // Update specific story if it's selected
                        if let Some(selected) = &self.selected_story_details {
                            if selected.id == story_id {
                                needs_update = true;
                            }
                        }
                        self.needs_refresh = true;
                    }
                    _ => {}
                }
            }
        }
        
        // Update selected story details if needed (after releasing the borrow)
        if needs_update {
            self.update_selected_story_details();
        }
    }
    
    /// Render the stories tab
    pub fn render(&mut self, f: &mut Frame, area: Rect) {
        // Process any pending events (rate limited to once per 100ms)
        if self.last_update.elapsed() > std::time::Duration::from_millis(100) {
            self.process_events();
            self.last_update = std::time::Instant::now();
        }
        
        // Check if refresh is needed
        if self.needs_refresh {
            self.refresh_stories();
            self.needs_refresh = false;
        }
        
        // Show creation form if active
        if self.creating_story {
            self.render_story_creation_form(f, area);
            return;
        }
        
        // Show arc creation form if active
        if self.creating_arc {
            self.render_arc_creation_form(f, area);
            return;
        }
        
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3), // Tab bar
                Constraint::Min(0),    // Content
            ])
            .split(area);
        
        // Render sub-tabs
        let tabs = Tabs::new(self.sub_tabs.clone())
            .block(Block::default().borders(Borders::ALL).title("Story Views"))
            .select(self.current_sub_tab)
            .style(Style::default().fg(Color::Gray))
            .highlight_style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD));
        
        f.render_widget(tabs, chunks[0]);
        
        // Render content based on current view
        match self.view_mode {
            StoryViewMode::Overview => self.render_overview(f, chunks[1]),
            StoryViewMode::Timeline => self.render_timeline(f, chunks[1]),
            StoryViewMode::Analytics => self.render_analytics(f, chunks[1]),
            StoryViewMode::Tasks => {
                // Split tasks view into list and visualization
                let task_chunks = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
                    .split(chunks[1]);
                
                self.render_tasks(f, task_chunks[0]);
                self.render_task_mapping(f, task_chunks[1]);
            },
            StoryViewMode::Relationships => self.render_relationships(f, chunks[1]),
            StoryViewMode::Configuration => self.render_configuration(f, chunks[1]),
        }
    }
    
    /// Render story creation form
    fn render_story_creation_form(&self, f: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .margin(2)
            .constraints([
                Constraint::Length(3),  // Title
                Constraint::Length(6),  // Title input
                Constraint::Length(3),  // Type/Template
                Constraint::Length(5),  // Type/Template selection
                Constraint::Length(3),  // Description
                Constraint::Min(5),     // Description input
                Constraint::Length(3),  // Instructions
            ])
            .split(area);
        
        let block = Block::default()
            .borders(Borders::ALL)
            .title("Create New Story")
            .border_style(Style::default().fg(Color::Green));
        
        f.render_widget(block, area);
        
        // Title
        let title_label = Paragraph::new("Story Title:")
            .style(Style::default().fg(Color::Yellow));
        f.render_widget(title_label, chunks[0]);
        
        let title_input = Paragraph::new(self.story_form.title.as_str())
            .block(Block::default()
                .borders(Borders::ALL)
                .border_style(if self.story_form.current_field == StoryFormField::Title {
                    Style::default().fg(Color::Yellow)
                } else {
                    Style::default().fg(Color::DarkGray)
                }))
            .style(Style::default().fg(Color::White));
        f.render_widget(title_input, chunks[1]);
        
        // Type/Template
        let type_label = Paragraph::new("Type/Template:")
            .style(Style::default().fg(Color::Yellow));
        f.render_widget(type_label, chunks[2]);
        
        let type_text = match self.story_form.story_type {
            StoryTypeSelection::Feature => "✨ Feature Development - End-to-end feature workflow",
            StoryTypeSelection::Bug => "🐛 Bug Investigation - Systematic debugging approach",
            StoryTypeSelection::Task => "📋 Task - Simple work item or todo",
            StoryTypeSelection::Epic => "🏔️ Epic - Large initiative with multiple objectives",
            StoryTypeSelection::Performance => "⚡ Performance Optimization - Analysis and optimization",
            StoryTypeSelection::Documentation => "📚 Documentation - Writing or updating docs",
            StoryTypeSelection::Testing => "🧪 Testing - Test creation or improvement",
            StoryTypeSelection::Research => "🔬 Research - Investigation or exploration",
        };
        
        let type_selection = Paragraph::new(type_text)
            .block(Block::default()
                .borders(Borders::ALL)
                .border_style(if self.story_form.current_field == StoryFormField::Type {
                    Style::default().fg(Color::Yellow)
                } else {
                    Style::default().fg(Color::DarkGray)
                }))
            .style(Style::default().fg(Color::Cyan));
        f.render_widget(type_selection, chunks[3]);
        
        // Description
        let desc_label = Paragraph::new("Description:")
            .style(Style::default().fg(Color::Yellow));
        f.render_widget(desc_label, chunks[4]);
        
        let desc_input = Paragraph::new(self.story_form.description.as_str())
            .block(Block::default()
                .borders(Borders::ALL)
                .border_style(if self.story_form.current_field == StoryFormField::Description {
                    Style::default().fg(Color::Yellow)
                } else {
                    Style::default().fg(Color::DarkGray)
                }))
            .style(Style::default().fg(Color::White))
            .wrap(Wrap { trim: false });
        f.render_widget(desc_input, chunks[5]);
        
        // Instructions
        let instructions = vec![
            Line::from(vec![
                Span::styled("[Tab]", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
                Span::raw(" Next field | "),
                Span::styled("[←→]", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
                Span::raw(" Change type | "),
                Span::styled("[Enter]", Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD)),
                Span::raw(" Create | "),
                Span::styled("[Esc]", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
                Span::raw(" Cancel"),
            ]),
        ];
        
        let instructions_widget = Paragraph::new(instructions)
            .style(Style::default().fg(Color::DarkGray));
        f.render_widget(instructions_widget, chunks[6]);
    }
    
    /// Render arc creation form
    fn render_arc_creation_form(&self, f: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .margin(2)
            .constraints([
                Constraint::Length(3),  // Title
                Constraint::Length(3),  // Title input
                Constraint::Length(3),  // Description
                Constraint::Min(5),     // Description input
                Constraint::Length(3),  // Instructions
            ])
            .split(area);
        
        let block = Block::default()
            .borders(Borders::ALL)
            .title("Create New Story Arc")
            .border_style(Style::default().fg(Color::Magenta));
        
        f.render_widget(block, area);
        
        // Title
        let title_label = Paragraph::new("Arc Title:")
            .style(Style::default().fg(Color::Yellow));
        f.render_widget(title_label, chunks[0]);
        
        let title_input = Paragraph::new(self.arc_form.title.as_str())
            .block(Block::default()
                .borders(Borders::ALL)
                .border_style(if self.arc_form.current_field == ArcFormField::Title {
                    Style::default().fg(Color::Yellow)
                } else {
                    Style::default().fg(Color::DarkGray)
                }))
            .style(Style::default().fg(Color::White));
        f.render_widget(title_input, chunks[1]);
        
        // Description
        let desc_label = Paragraph::new("Arc Description:")
            .style(Style::default().fg(Color::Yellow));
        f.render_widget(desc_label, chunks[2]);
        
        let desc_input = Paragraph::new(self.arc_form.description.as_str())
            .block(Block::default()
                .borders(Borders::ALL)
                .border_style(if self.arc_form.current_field == ArcFormField::Description {
                    Style::default().fg(Color::Yellow)
                } else {
                    Style::default().fg(Color::DarkGray)
                }))
            .style(Style::default().fg(Color::White))
            .wrap(Wrap { trim: false });
        f.render_widget(desc_input, chunks[3]);
        
        // Instructions
        let instructions = vec![
            Line::from(vec![
                Span::styled("[Tab]", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
                Span::raw(" Next field | "),
                Span::styled("[Enter]", Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD)),
                Span::raw(" Create arc | "),
                Span::styled("[Esc]", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
                Span::raw(" Cancel"),
            ]),
        ];
        
        let instructions_widget = Paragraph::new(instructions)
            .style(Style::default().fg(Color::DarkGray));
        f.render_widget(instructions_widget, chunks[4]);
    }
    
    /// Render overview
    fn render_overview(&self, f: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(30),
                Constraint::Percentage(40),
                Constraint::Percentage(30),
            ])
            .split(area);
        
        // Left: Story statistics
        self.render_statistics(f, chunks[0]);
        
        // Middle: Actual stories list
        self.render_stories_list(f, chunks[1]);
        
        // Right: Selected story details
        self.render_story_details(f, chunks[2]);
    }
    
    /// Render story statistics
    fn render_statistics(&self, f: &mut Frame, area: Rect) {
        let stats_block = Block::default()
            .borders(Borders::ALL)
            .title("Story Statistics")
            .border_style(Style::default().fg(Color::Cyan));
        
        let total_stories = self.stories.len();
        let active_stories = self.stories.iter().filter(|s| s.status == StoryStatus::Active).count();
        let completed_stories = self.stories.iter().filter(|s| s.status == StoryStatus::Completed).count();
        let total_plot_points: usize = self.stories.iter()
            .flat_map(|s| &s.arcs)
            .map(|a| a.plot_points.len())
            .sum();
        
        let stats_text = vec![
            Line::from(vec![
                Span::raw("Total Stories: "),
                Span::styled(
                    total_stories.to_string(),
                    Style::default().fg(Color::Cyan),
                ),
            ]),
            Line::from(vec![
                Span::raw("Active: "),
                Span::styled(
                    active_stories.to_string(),
                    Style::default().fg(Color::Green),
                ),
            ]),
            Line::from(vec![
                Span::raw("Completed: "),
                Span::styled(
                    completed_stories.to_string(),
                    Style::default().fg(Color::Blue),
                ),
            ]),
            Line::from(vec![
                Span::raw("Plot Points: "),
                Span::styled(
                    total_plot_points.to_string(),
                    Style::default().fg(Color::Yellow),
                ),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled("[N]", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
                Span::raw(" New Story"),
            ]),
            Line::from(vec![
                Span::styled("[E]", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
                Span::raw(" Edit Story"),
            ]),
            Line::from(vec![
                Span::styled("[D]", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
                Span::raw(" Delete Story"),
            ]),
        ];
        
        let paragraph = Paragraph::new(stats_text)
            .block(stats_block)
            .wrap(Wrap { trim: true });
        
        f.render_widget(paragraph, area);
    }
    
    /// Render stories list
    fn render_stories_list(&self, f: &mut Frame, area: Rect) {
        let stories_block = Block::default()
            .borders(Borders::ALL)
            .title("Stories")
            .border_style(Style::default().fg(Color::Magenta));
        
        if self.stories.is_empty() {
            let empty_msg = Paragraph::new("No stories created yet.\n\nPress [N] to create a new story.")
                .block(stories_block)
                .style(Style::default().fg(Color::DarkGray))
                .alignment(ratatui::layout::Alignment::Center);
            f.render_widget(empty_msg, area);
            return;
        }
        
        let items: Vec<ListItem> = self.stories
            .iter()
            .map(|story| {
                let status_icon = match story.status {
                    StoryStatus::NotStarted => "⏸️",
                    StoryStatus::Draft => "📝",
                    StoryStatus::Active => "🟢",
                    StoryStatus::Completed => "✅",
                    StoryStatus::Archived => "📦",
                };
                
                let type_icon = self.get_story_type_icon(&story.story_type);
                let summary_preview = story.summary.chars().take(50).collect::<String>();
                
                ListItem::new(vec![
                    Line::from(vec![
                        Span::raw(format!("{} {} ", status_icon, type_icon)),
                        Span::styled(&story.title, Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
                    ]),
                    Line::from(vec![
                        Span::raw("  "),
                        Span::styled(
                            summary_preview,
                            Style::default().fg(Color::Gray),
                        ),
                        if story.summary.len() > 50 { Span::raw("...") } else { Span::raw("") },
                    ]),
                ])
            })
            .collect();
        
        let list = List::new(items)
            .block(stories_block)
            .highlight_style(
                Style::default()
                    .bg(Color::DarkGray)
                    .add_modifier(Modifier::BOLD),
            )
            .highlight_symbol("► ");
        
        f.render_stateful_widget(list, area, &mut self.list_state.clone());
    }
    
    /// Render story details
    fn render_story_details(&self, f: &mut Frame, area: Rect) {
        let details_block = Block::default()
            .borders(Borders::ALL)
            .title("Story Details")
            .border_style(Style::default().fg(Color::Yellow));
        
        if let Some(story) = &self.selected_story_details {
            let mut lines = vec![
                Line::from(vec![
                    Span::styled("Title: ", Style::default().fg(Color::Gray)),
                    Span::styled(&story.title, Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
                ]),
                Line::from(vec![
                    Span::styled("Status: ", Style::default().fg(Color::Gray)),
                    Span::styled(story.status.to_string(), self.get_status_color(story.status.clone())),
                ]),
                Line::from(vec![
                    Span::styled("Created: ", Style::default().fg(Color::Gray)),
                    Span::raw(story.created_at.format("%Y-%m-%d %H:%M").to_string()),
                ]),
                Line::from(""),
                Line::from(vec![
                    Span::styled("Summary:", Style::default().fg(Color::Gray)),
                ]),
            ];
            
            // Add summary lines
            for line in story.summary.lines() {
                lines.push(Line::from(vec![
                    Span::raw("  "),
                    Span::raw(line),
                ]));
            }
            
            lines.push(Line::from(""));
            lines.push(Line::from(vec![
                Span::styled(format!("Arcs: {}", story.arcs.len()), Style::default().fg(Color::Cyan)),
            ]));
            
            // Show current arc if any
            if let Some(current_arc_id) = &story.current_arc {
                if let Some(arc) = story.arcs.iter().find(|a| a.id == *current_arc_id) {
                    lines.push(Line::from(vec![
                        Span::styled("Current Arc: ", Style::default().fg(Color::Gray)),
                        Span::styled(&arc.title, Style::default().fg(Color::Green)),
                    ]));
                }
            }
            
            let paragraph = Paragraph::new(lines)
                .block(details_block)
                .wrap(Wrap { trim: true });
            
            f.render_widget(paragraph, area);
        } else {
            let empty_msg = Paragraph::new("Select a story to view details")
                .block(details_block)
                .style(Style::default().fg(Color::DarkGray))
                .alignment(ratatui::layout::Alignment::Center);
            f.render_widget(empty_msg, area);
        }
    }
    
    /// Render timeline view
    fn render_timeline(&self, f: &mut Frame, area: Rect) {
        if let Some(story) = &self.selected_story_details {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(3),  // Story title
                    Constraint::Min(0),     // Timeline
                ])
                .split(area);
            
            // Story title
            let title_block = Block::default()
                .borders(Borders::ALL)
                .title("Story Timeline")
                .border_style(Style::default().fg(Color::Cyan));
            
            let title_text = vec![
                Line::from(vec![
                    Span::styled(&story.title, Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
                    Span::raw(" - "),
                    Span::styled(story.status.to_string(), self.get_status_color(story.status.clone())),
                ]),
            ];
            
            let title_widget = Paragraph::new(title_text)
                .block(title_block);
            
            f.render_widget(title_widget, chunks[0]);
            
            // Timeline with arcs
            if story.arcs.is_empty() {
                let empty_msg = Paragraph::new("No arcs created yet.\n\nPress [A] to add a new arc.")
                    .block(Block::default().borders(Borders::ALL))
                    .style(Style::default().fg(Color::DarkGray))
                    .alignment(ratatui::layout::Alignment::Center);
                f.render_widget(empty_msg, chunks[1]);
            } else {
                self.render_arcs_timeline(f, chunks[1], story);
            }
        } else {
            let block = Block::default()
                .borders(Borders::ALL)
                .title("Story Timeline");
            
            let paragraph = Paragraph::new("Select a story to view its timeline")
                .block(block)
                .style(Style::default().fg(Color::DarkGray))
                .alignment(ratatui::layout::Alignment::Center);
            
            f.render_widget(paragraph, area);
        }
    }
    
    /// Render story arcs as timeline
    fn render_arcs_timeline(&self, f: &mut Frame, area: Rect, story: &Story) {
        let block = Block::default()
            .borders(Borders::ALL)
            .title("Story Arcs")
            .border_style(Style::default().fg(Color::Magenta));
        
        let mut lines = Vec::new();
        
        for (idx, arc) in story.arcs.iter().enumerate() {
            let is_current = story.current_arc.as_ref().map_or(false, |id| *id == arc.id);
            
            // Arc header
            let arc_status_icon = match arc.status {
                crate::story::types::ArcStatus::Planning => "📋",
                crate::story::types::ArcStatus::Active => "🟢",
                crate::story::types::ArcStatus::Paused => "⏸️",
                crate::story::types::ArcStatus::Completed => "✅",
                crate::story::types::ArcStatus::Abandoned => "❌",
            };
            
            let arc_style = if is_current {
                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::White)
            };
            
            lines.push(Line::from(vec![
                Span::raw(format!("{}. ", idx + 1)),
                Span::raw(arc_status_icon),
                Span::raw(" "),
                Span::styled(&arc.title, arc_style),
                if is_current { Span::styled(" (Current)", Style::default().fg(Color::Yellow)) } else { Span::raw("") },
            ]));
            
            lines.push(Line::from(vec![
                Span::raw("   "),
                Span::styled(&arc.description, Style::default().fg(Color::Gray)),
            ]));
            
            // Show plot points count
            lines.push(Line::from(vec![
                Span::raw("   "),
                Span::styled(
                    format!("Plot Points: {}", arc.plot_points.len()),
                    Style::default().fg(Color::Cyan),
                ),
                Span::raw(" | "),
                Span::styled(
                    format!("Started: {}", arc.started_at.format("%Y-%m-%d")),
                    Style::default().fg(Color::Blue),
                ),
                if let Some(completed) = arc.completed_at {
                    Span::styled(
                        format!(" | Completed: {}", completed.format("%Y-%m-%d")),
                        Style::default().fg(Color::Green),
                    )
                } else {
                    Span::raw("")
                },
            ]));
            
            // Show recent plot points
            let recent_points: Vec<_> = arc.plot_points.iter().rev().take(3).collect();
            for point in recent_points {
                let point_icon = match &point.plot_type {
                    PlotType::Goal { .. } => "🎯",
                    PlotType::Task { .. } => "📋",
                    PlotType::Decision { .. } => "🤔",
                    PlotType::Discovery { .. } => "💡",
                    PlotType::Issue { .. } => "⚠️",
                    PlotType::Transformation { .. } => "🔄",
                    PlotType::Interaction { .. } => "🤝",
                    PlotType::Progress { .. } => "📊",
                    PlotType::Analysis { .. } => "🔍",
                    PlotType::Action { .. } => "⚡",
                };
                
                let point_desc_preview = point.description.chars().take(60).collect::<String>();
                lines.push(Line::from(vec![
                    Span::raw("     "),
                    Span::raw(point_icon),
                    Span::raw(" "),
                    Span::styled(
                        point_desc_preview,
                        Style::default().fg(Color::DarkGray),
                    ),
                ]));
            }
            
            lines.push(Line::from(""));
        }
        
        let paragraph = Paragraph::new(lines)
            .block(block)
            .wrap(Wrap { trim: true })
            .scroll((0, 0));
        
        f.render_widget(paragraph, area);
    }
    
    /// Render analytics view
    fn render_analytics(&self, f: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Percentage(50),
                Constraint::Percentage(50),
            ])
            .split(area);
        
        // Top: Story type distribution
        self.render_story_type_distribution(f, chunks[0]);
        
        // Bottom: Plot point analysis
        self.render_plot_point_analysis(f, chunks[1]);
    }
    
    /// Render story type distribution chart
    fn render_story_type_distribution(&self, f: &mut Frame, area: Rect) {
        let mut type_counts = std::collections::HashMap::new();
        
        // Count stories by type
        for story in &self.stories {
            let type_name = match &story.story_type {
                crate::story::types::StoryType::Feature { .. } => "Feature",
                crate::story::types::StoryType::Bug { .. } => "Bug",
                crate::story::types::StoryType::Task { .. } => "Task",
                crate::story::types::StoryType::Epic { .. } => "Epic",
                crate::story::types::StoryType::Performance { .. } => "Performance",
                crate::story::types::StoryType::Documentation { .. } => "Docs",
                crate::story::types::StoryType::Testing { .. } => "Testing",
                crate::story::types::StoryType::Research { .. } => "Research",
                crate::story::types::StoryType::Security { .. } => "Security",
                crate::story::types::StoryType::Refactoring { .. } => "Refactor",
                crate::story::types::StoryType::Agent { .. } => "Agent",
                crate::story::types::StoryType::System { .. } => "System",
                _ => "Other",
            };
            *type_counts.entry(type_name).or_insert(0) += 1;
        }
        
        if type_counts.is_empty() {
            let empty_msg = Paragraph::new("No stories to analyze.\n\nCreate stories to see analytics.")
                .block(Block::default().borders(Borders::ALL).title("Stories by Type"))
                .style(Style::default().fg(Color::DarkGray))
                .alignment(ratatui::layout::Alignment::Center);
            f.render_widget(empty_msg, area);
            return;
        }
        
        let data: Vec<(&str, u64)> = type_counts
            .iter()
            .map(|(k, v)| (*k, *v as u64))
            .collect();
        
        let bar_chart = BarChart::default()
            .block(Block::default().borders(Borders::ALL).title("Stories by Type").border_style(Style::default().fg(Color::Cyan)))
            .data(&data)
            .bar_width(9)
            .bar_gap(2)
            .value_style(Style::default().fg(Color::Cyan));
        
        f.render_widget(bar_chart, area);
    }
    
    /// Render plot point analysis table
    fn render_plot_point_analysis(&self, f: &mut Frame, area: Rect) {
        let mut plot_type_counts = std::collections::HashMap::new();
        let mut total_plot_points = 0;
        
        // Count plot points by type
        for story in &self.stories {
            for arc in &story.arcs {
                for plot_point in &arc.plot_points {
                    total_plot_points += 1;
                    let plot_type_name = match &plot_point.plot_type {
                        PlotType::Goal { .. } => "Goal",
                        PlotType::Task { .. } => "Task",
                        PlotType::Decision { .. } => "Decision",
                        PlotType::Discovery { .. } => "Discovery",
                        PlotType::Issue { .. } => "Issue",
                        PlotType::Transformation { .. } => "Transform",
                        PlotType::Interaction { .. } => "Interaction",
                        PlotType::Progress { .. } => "Progress",
                        PlotType::Analysis { .. } => "Analysis",
                        PlotType::Action { .. } => "Action",
                    };
                    *plot_type_counts.entry(plot_type_name).or_insert(0) += 1;
                }
            }
        }
        
        if plot_type_counts.is_empty() {
            let empty_msg = Paragraph::new("No plot points to analyze.\n\nAdd arcs and plot points to stories.")
                .block(Block::default().borders(Borders::ALL).title("Plot Points"))
                .style(Style::default().fg(Color::DarkGray))
                .alignment(ratatui::layout::Alignment::Center);
            f.render_widget(empty_msg, area);
            return;
        }
        
        let rows: Vec<Row> = plot_type_counts
            .iter()
            .map(|(plot_type, count)| {
                Row::new(vec![
                    plot_type.to_string(),
                    count.to_string(),
                    self.create_mini_bar(*count, total_plot_points),
                ])
            })
            .collect();
        
        let widths = [
            Constraint::Length(15),
            Constraint::Length(10),
            Constraint::Min(20),
        ];
        
        let table = Table::new(rows, widths)
            .header(Row::new(vec!["Type", "Count", "Distribution"]).style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)))
            .block(Block::default().borders(Borders::ALL).title("Plot Points").border_style(Style::default().fg(Color::Magenta)));
        
        f.render_widget(table, area);
    }
    
    /// Render tasks view
    fn render_tasks(&self, f: &mut Frame, area: Rect) {
        if let Some(story) = &self.selected_story_details {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(3),  // Header
                    Constraint::Min(0),     // Tasks list
                ])
                .split(area);
            
            // Header
            let header_block = Block::default()
                .borders(Borders::ALL)
                .title("Story Tasks")
                .border_style(Style::default().fg(Color::Yellow));
            
            let task_count = story.arcs.iter()
                .flat_map(|arc| &arc.plot_points)
                .filter(|pp| matches!(pp.plot_type, PlotType::Task { .. }))
                .count();
            
            let header_text = vec![
                Line::from(vec![
                    Span::styled("Total Tasks: ", Style::default().fg(Color::Gray)),
                    Span::styled(task_count.to_string(), Style::default().fg(Color::Cyan)),
                    Span::raw(" | "),
                    Span::styled("[T]", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
                    Span::raw(" New Task | "),
                    Span::styled("[Space]", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
                    Span::raw(" Toggle Complete"),
                ]),
            ];
            
            let header_widget = Paragraph::new(header_text)
                .block(header_block);
            
            f.render_widget(header_widget, chunks[0]);
            
            // Tasks list
            self.render_tasks_list(f, chunks[1], story);
        } else {
            let block = Block::default()
                .borders(Borders::ALL)
                .title("Story Tasks");
            
            let paragraph = Paragraph::new("Select a story to view its tasks")
                .block(block)
                .style(Style::default().fg(Color::DarkGray))
                .alignment(ratatui::layout::Alignment::Center);
            
            f.render_widget(paragraph, area);
        }
    }
    
    /// Render tasks list from plot points
    fn render_tasks_list(&self, f: &mut Frame, area: Rect, story: &Story) {
        let mut tasks = Vec::new();
        
        // Extract all task plot points with indices
        for (arc_idx, arc) in story.arcs.iter().enumerate() {
            for (plot_idx, plot_point) in arc.plot_points.iter().enumerate() {
                if let PlotType::Task { description, completed } = &plot_point.plot_type {
                    tasks.push((arc_idx, plot_idx, arc, plot_point, description, *completed));
                }
            }
        }
        
        if tasks.is_empty() {
            let empty_msg = Paragraph::new("No tasks in this story.\n\nTasks are created from Task plot points in story arcs.\n\nPress [P] in Timeline view to add plot points.")
                .block(Block::default().borders(Borders::ALL))
                .style(Style::default().fg(Color::DarkGray))
                .alignment(ratatui::layout::Alignment::Center);
            f.render_widget(empty_msg, area);
            return;
        }
        
        // Create list items
        let items: Vec<ListItem> = tasks
            .iter()
            .map(|(_, _, arc, plot_point, description, completed)| {
                let status_icon = if *completed { "✅" } else { "⬜" };
                let status_style = if *completed {
                    Style::default().fg(Color::Green).add_modifier(Modifier::CROSSED_OUT)
                } else {
                    Style::default().fg(Color::White)
                };
                
                ListItem::new(vec![
                    Line::from(vec![
                        Span::raw(status_icon),
                        Span::raw(" "),
                        Span::styled(description.as_str(), status_style),
                    ]),
                    Line::from(vec![
                        Span::raw("   Arc: "),
                        Span::styled(&arc.title, Style::default().fg(Color::Cyan)),
                        Span::raw(" | "),
                        Span::styled(
                            plot_point.timestamp.format("%Y-%m-%d %H:%M").to_string(),
                            Style::default().fg(Color::DarkGray),
                        ),
                    ]),
                    if !plot_point.tags.is_empty() {
                        Line::from(vec![
                            Span::raw("   Tags: "),
                            Span::styled(
                                plot_point.tags.join(", "),
                                Style::default().fg(Color::Blue),
                            ),
                        ])
                    } else {
                        Line::from("")
                    },
                ])
            })
            .collect();
        
        let tasks_list = List::new(items)
            .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(Color::Yellow)))
            .highlight_style(
                Style::default()
                    .bg(Color::DarkGray)
                    .add_modifier(Modifier::BOLD),
            )
            .highlight_symbol("► ");
        
        f.render_stateful_widget(tasks_list, area, &mut self.task_list_state.clone());
    }
    
    /// Render relationships view
    fn render_relationships(&self, f: &mut Frame, area: Rect) {
        let block = Block::default()
            .borders(Borders::ALL)
            .title("Story Relationships")
            .border_style(Style::default().fg(Color::Magenta));
        
        if let Some(story) = &self.selected_story_details {
            let mut lines = vec![
                Line::from(vec![
                    Span::styled("Current Story: ", Style::default().fg(Color::Gray)),
                    Span::styled(&story.title, Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
                ]),
                Line::from(""),
            ];
            
            // Show dependencies
            if !story.metadata.dependencies.is_empty() {
                lines.push(Line::from(vec![
                    Span::styled("Dependencies:", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
                ]));
                for dep_id in &story.metadata.dependencies {
                    if let Some(dep_story) = self.stories.iter().find(|s| s.id == *dep_id) {
                        lines.push(Line::from(vec![
                            Span::raw("  → "),
                            Span::styled(&dep_story.title, Style::default().fg(Color::Cyan)),
                        ]));
                    }
                }
                lines.push(Line::from(""));
            }
            
            // Show related stories
            if !story.metadata.related_stories.is_empty() {
                lines.push(Line::from(vec![
                    Span::styled("Related Stories:", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
                ]));
                for rel_id in &story.metadata.related_stories {
                    if let Some(rel_story) = self.stories.iter().find(|s| s.id == *rel_id) {
                        lines.push(Line::from(vec![
                            Span::raw("  ↔ "),
                            Span::styled(&rel_story.title, Style::default().fg(Color::Blue)),
                        ]));
                    }
                }
            }
            
            let paragraph = Paragraph::new(lines)
                .block(block)
                .wrap(Wrap { trim: true });
            
            f.render_widget(paragraph, area);
        } else {
            let paragraph = Paragraph::new("Select a story to view its relationships")
                .block(block)
                .style(Style::default().fg(Color::DarkGray))
                .alignment(ratatui::layout::Alignment::Center);
            
            f.render_widget(paragraph, area);
        }
    }
    
    /// Render configuration view
    fn render_configuration(&self, f: &mut Frame, area: Rect) {
        let block = Block::default()
            .title("Story Autonomy Configuration")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Cyan));
        
        let inner_area = block.inner(area);
        f.render_widget(block, area);
        
        // Split into two columns
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
            .split(inner_area);
        
        // Left column - toggle options
        self.render_config_toggles(f, chunks[0]);
        
        // Right column - numeric settings and status
        self.render_config_settings(f, chunks[1]);
    }
    
    /// Render configuration toggle options
    fn render_config_toggles(&self, f: &mut Frame, area: Rect) {
        let toggles = vec![
            ("Auto Maintenance", self.autonomy_config.auto_maintenance, ConfigField::AutoMaintenance),
            ("PR Review", self.autonomy_config.pr_review_enabled, ConfigField::PrReview),
            ("Bug Detection", self.autonomy_config.bug_detection_enabled, ConfigField::BugDetection),
            ("Quality Monitoring", self.autonomy_config.quality_monitoring, ConfigField::QualityMonitoring),
            ("Performance Optimization", self.autonomy_config.performance_optimization, ConfigField::PerformanceOptimization),
            ("Security Scanning", self.autonomy_config.security_scanning, ConfigField::SecurityScanning),
            ("Test Generation", self.autonomy_config.test_generation, ConfigField::TestGeneration),
            ("Refactoring", self.autonomy_config.refactoring_enabled, ConfigField::Refactoring),
            ("Dependency Updates", self.autonomy_config.dependency_updates, ConfigField::DependencyUpdates),
        ];
        
        let block = Block::default()
            .title("Features")
            .borders(Borders::ALL);
        
        let list_items: Vec<ListItem> = toggles
            .iter()
            .map(|(name, enabled, field)| {
                let checkbox = if *enabled { "[✓]" } else { "[ ]" };
                let style = if self.config_selected_field == *field {
                    Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
                } else if *enabled {
                    Style::default().fg(Color::Green)
                } else {
                    Style::default().fg(Color::DarkGray)
                };
                
                ListItem::new(format!("{} {}", checkbox, name)).style(style)
            })
            .collect();
        
        let list = List::new(list_items)
            .block(block)
            .highlight_style(Style::default().bg(Color::DarkGray));
        
        f.render_widget(list, area);
    }
    
    /// Render configuration settings and status
    fn render_config_settings(&self, f: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(10), // Settings
                Constraint::Length(8),  // Status
                Constraint::Min(0),     // Instructions
            ])
            .split(area);
        
        // Settings block
        let settings_block = Block::default()
            .title("Settings")
            .borders(Borders::ALL);
        
        let settings_text = vec![
            Line::from(format!("Maintenance Interval: {} hours", self.autonomy_config.maintenance_interval_hours)),
            Line::from(format!("PR Review Threshold: {:.0}%", self.autonomy_config.pr_review_threshold * 100.0)),
            Line::from(format!("Quality Threshold: {:.0}%", self.autonomy_config.quality_threshold * 100.0)),
        ];
        
        let settings = Paragraph::new(settings_text)
            .block(settings_block)
            .style(Style::default().fg(Color::White));
        
        f.render_widget(settings, chunks[0]);
        
        // Status block
        let status_block = Block::default()
            .title("Status")
            .borders(Borders::ALL);
        
        let status_text = vec![
            Line::from(vec![
                Span::raw("Story Autonomy: "),
                if self.story_autonomy_enabled {
                    Span::styled("ENABLED", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD))
                } else {
                    Span::styled("DISABLED", Style::default().fg(Color::Red))
                },
            ]),
            Line::from(""),
            Line::from(format!("Active Features: {}", 
                [
                    self.autonomy_config.pr_review_enabled,
                    self.autonomy_config.bug_detection_enabled,
                    self.autonomy_config.quality_monitoring,
                    self.autonomy_config.performance_optimization,
                    self.autonomy_config.security_scanning,
                    self.autonomy_config.test_generation,
                    self.autonomy_config.refactoring_enabled,
                    self.autonomy_config.dependency_updates,
                ].iter().filter(|&&x| x).count()
            )),
        ];
        
        let status = Paragraph::new(status_text)
            .block(status_block)
            .style(Style::default().fg(Color::White));
        
        f.render_widget(status, chunks[1]);
        
        // Instructions
        let instructions_block = Block::default()
            .title("Controls")
            .borders(Borders::ALL);
        
        let instructions = vec![
            Line::from("Space: Toggle feature"),
            Line::from("Enter: Enable/Disable autonomy"),
            Line::from("↑/↓: Navigate"),
            Line::from("+/-: Adjust values"),
            Line::from("Tab: Switch tabs"),
        ];
        
        let help = Paragraph::new(instructions)
            .block(instructions_block)
            .style(Style::default().fg(Color::DarkGray));
        
        f.render_widget(help, chunks[2]);
    }
    
    /// Handle key events
    pub fn handle_key(&mut self, key: KeyCode, _modifiers: KeyModifiers) -> bool {
        if self.creating_story {
            return self.handle_story_creation_key(key);
        }
        
        if self.creating_arc {
            return self.handle_arc_creation_key(key);
        }
        
        match key {
            KeyCode::Tab => {
                self.current_sub_tab = (self.current_sub_tab + 1) % self.sub_tabs.len();
                self.view_mode = match self.current_sub_tab {
                    0 => StoryViewMode::Overview,
                    1 => StoryViewMode::Timeline,
                    2 => StoryViewMode::Analytics,
                    3 => StoryViewMode::Tasks,
                    4 => StoryViewMode::Relationships,
                    5 => StoryViewMode::Configuration,
                    _ => StoryViewMode::Overview,
                };
                true
            }
            KeyCode::Up => {
                match self.view_mode {
                    StoryViewMode::Overview => {
                        if !self.stories.is_empty() {
                            let current = self.list_state.selected().unwrap_or(0);
                            if current > 0 {
                                self.list_state.select(Some(current - 1));
                                self.selected_story = current - 1;
                                self.update_selected_story_details();
                            }
                        }
                    }
                    StoryViewMode::Tasks => {
                        if let Some(story) = &self.selected_story_details {
                            let task_count = story.arcs.iter()
                                .flat_map(|arc| &arc.plot_points)
                                .filter(|pp| matches!(pp.plot_type, PlotType::Task { .. }))
                                .count();
                            
                            if task_count > 0 {
                                let current = self.task_list_state.selected().unwrap_or(0);
                                if current > 0 {
                                    self.task_list_state.select(Some(current - 1));
                                }
                            }
                        }
                    }
                    StoryViewMode::Configuration => {
                        // Navigate configuration fields
                        match self.config_selected_field {
                            ConfigField::AutoMaintenance => self.config_selected_field = ConfigField::DependencyUpdates,
                            ConfigField::PrReview => self.config_selected_field = ConfigField::AutoMaintenance,
                            ConfigField::BugDetection => self.config_selected_field = ConfigField::PrReview,
                            ConfigField::QualityMonitoring => self.config_selected_field = ConfigField::BugDetection,
                            ConfigField::PerformanceOptimization => self.config_selected_field = ConfigField::QualityMonitoring,
                            ConfigField::SecurityScanning => self.config_selected_field = ConfigField::PerformanceOptimization,
                            ConfigField::TestGeneration => self.config_selected_field = ConfigField::SecurityScanning,
                            ConfigField::Refactoring => self.config_selected_field = ConfigField::TestGeneration,
                            ConfigField::DependencyUpdates => self.config_selected_field = ConfigField::Refactoring,
                            ConfigField::MaintenanceInterval => self.config_selected_field = ConfigField::DependencyUpdates,
                            ConfigField::PrReviewThreshold => self.config_selected_field = ConfigField::MaintenanceInterval,
                            ConfigField::QualityThreshold => self.config_selected_field = ConfigField::PrReviewThreshold,
                        }
                    }
                    _ => {}
                }
                true
            }
            KeyCode::Down => {
                match self.view_mode {
                    StoryViewMode::Overview => {
                        if !self.stories.is_empty() {
                            let current = self.list_state.selected().unwrap_or(0);
                            if current < self.stories.len() - 1 {
                                self.list_state.select(Some(current + 1));
                                self.selected_story = current + 1;
                                self.update_selected_story_details();
                            }
                        }
                    }
                    StoryViewMode::Tasks => {
                        if let Some(story) = &self.selected_story_details {
                            let task_count = story.arcs.iter()
                                .flat_map(|arc| &arc.plot_points)
                                .filter(|pp| matches!(pp.plot_type, PlotType::Task { .. }))
                                .count();
                            
                            if task_count > 0 {
                                let current = self.task_list_state.selected().unwrap_or(0);
                                if current < task_count - 1 {
                                    self.task_list_state.select(Some(current + 1));
                                }
                            }
                        }
                    }
                    StoryViewMode::Configuration => {
                        // Navigate configuration fields downward
                        match self.config_selected_field {
                            ConfigField::AutoMaintenance => self.config_selected_field = ConfigField::PrReview,
                            ConfigField::PrReview => self.config_selected_field = ConfigField::BugDetection,
                            ConfigField::BugDetection => self.config_selected_field = ConfigField::QualityMonitoring,
                            ConfigField::QualityMonitoring => self.config_selected_field = ConfigField::PerformanceOptimization,
                            ConfigField::PerformanceOptimization => self.config_selected_field = ConfigField::SecurityScanning,
                            ConfigField::SecurityScanning => self.config_selected_field = ConfigField::TestGeneration,
                            ConfigField::TestGeneration => self.config_selected_field = ConfigField::Refactoring,
                            ConfigField::Refactoring => self.config_selected_field = ConfigField::DependencyUpdates,
                            ConfigField::DependencyUpdates => self.config_selected_field = ConfigField::AutoMaintenance,
                            ConfigField::MaintenanceInterval => self.config_selected_field = ConfigField::PrReviewThreshold,
                            ConfigField::PrReviewThreshold => self.config_selected_field = ConfigField::QualityThreshold,
                            ConfigField::QualityThreshold => self.config_selected_field = ConfigField::MaintenanceInterval,
                        }
                    }
                    _ => {}
                }
                true
            }
            KeyCode::Char('n') | KeyCode::Char('N') => {
                self.creating_story = true;
                self.story_form = StoryCreationForm::default();
                true
            }
            KeyCode::Char('a') | KeyCode::Char('A') if self.view_mode == StoryViewMode::Timeline => {
                if self.selected_story_details.is_some() {
                    self.creating_arc = true;
                    self.arc_form = ArcCreationForm::default();
                }
                true
            }
            KeyCode::Char('t') | KeyCode::Char('T') if self.view_mode == StoryViewMode::Tasks => {
                // Add new task plot point
                if let Some(story) = &mut self.selected_story_details {
                    if let Some(current_arc_id) = &story.current_arc {
                        if let Some(arc) = story.arcs.iter_mut().find(|a| a.id == *current_arc_id) {
                            // Create a new task plot point
                            let new_task = PlotPoint {
                                id: PlotPointId::new(),
                                title: String::from("New Task"),
                                description: "New task".to_string(),
                                sequence_number: 0,
                                timestamp: chrono::Utc::now(),
                                plot_type: PlotType::Task {
                                    description: "New task".to_string(),
                                    completed: false,
                                },
                                status: crate::story::types::PlotPointStatus::Pending,
                                estimated_duration: None,
                                actual_duration: None,
                                context_tokens: vec![],
                                importance: 0.5,
                                metadata: crate::story::types::PlotMetadata::default(),
                                tags: vec![],
                                consequences: vec![],
                            };
                            arc.plot_points.push(new_task);
                            self.needs_refresh = true;
                        }
                    }
                }
                true
            }
            KeyCode::Char(' ') if self.view_mode == StoryViewMode::Configuration => {
                // Toggle configuration options
                match self.config_selected_field {
                    ConfigField::AutoMaintenance => self.autonomy_config.auto_maintenance = !self.autonomy_config.auto_maintenance,
                    ConfigField::PrReview => self.autonomy_config.pr_review_enabled = !self.autonomy_config.pr_review_enabled,
                    ConfigField::BugDetection => self.autonomy_config.bug_detection_enabled = !self.autonomy_config.bug_detection_enabled,
                    ConfigField::QualityMonitoring => self.autonomy_config.quality_monitoring = !self.autonomy_config.quality_monitoring,
                    ConfigField::PerformanceOptimization => self.autonomy_config.performance_optimization = !self.autonomy_config.performance_optimization,
                    ConfigField::SecurityScanning => self.autonomy_config.security_scanning = !self.autonomy_config.security_scanning,
                    ConfigField::TestGeneration => self.autonomy_config.test_generation = !self.autonomy_config.test_generation,
                    ConfigField::Refactoring => self.autonomy_config.refactoring_enabled = !self.autonomy_config.refactoring_enabled,
                    ConfigField::DependencyUpdates => self.autonomy_config.dependency_updates = !self.autonomy_config.dependency_updates,
                    _ => {} // Numeric fields handled differently
                }
                true
            }
            KeyCode::Enter if self.view_mode == StoryViewMode::Configuration => {
                // Toggle story autonomy enabled/disabled
                self.story_autonomy_enabled = !self.story_autonomy_enabled;
                // Apply configuration if enabled
                if self.story_autonomy_enabled {
                    // TODO: Apply configuration to story autonomy system
                }
                true
            }
            KeyCode::Char(' ') if self.view_mode == StoryViewMode::Tasks => {
                // Toggle task completion
                if let Some(story) = &mut self.selected_story_details {
                    if let Some(selected_idx) = self.task_list_state.selected() {
                        let mut task_idx = 0;
                        'outer: for arc in &mut story.arcs {
                            for plot_point in &mut arc.plot_points {
                                if let PlotType::Task { description: _, completed } = &mut plot_point.plot_type {
                                    if task_idx == selected_idx {
                                        *completed = !*completed;
                                        self.needs_refresh = true;
                                        break 'outer;
                                    }
                                    task_idx += 1;
                                }
                            }
                        }
                    }
                }
                true
            }
            KeyCode::Char('r') | KeyCode::Char('R') => {
                self.refresh_stories();
                true
            }
            KeyCode::Char('e') | KeyCode::Char('E') => {
                // Edit selected story
                if self.selected_story_details.is_some() {
                    // Would open story editor
                    tracing::info!("Edit story requested");
                }
                true
            }
            KeyCode::Char('d') | KeyCode::Char('D') => {
                // Delete selected story
                if let Some(story) = &self.selected_story_details {
                    if let Some(engine) = &self.story_engine {
                        engine.stories.remove(&story.id);
                        self.refresh_stories();
                    }
                }
                true
            }
            _ => false,
        }
    }
    
    /// Handle key events during story creation
    fn handle_story_creation_key(&mut self, key: KeyCode) -> bool {
        match key {
            KeyCode::Esc => {
                self.creating_story = false;
                true
            }
            KeyCode::Tab => {
                self.story_form.current_field = match self.story_form.current_field {
                    StoryFormField::Title => StoryFormField::Type,
                    StoryFormField::Type => StoryFormField::Description,
                    StoryFormField::Description => StoryFormField::Title,
                };
                true
            }
            KeyCode::Left | KeyCode::Right if self.story_form.current_field == StoryFormField::Type => {
                // Cycle through story types
                self.story_form.story_type = match (&self.story_form.story_type, key) {
                    (StoryTypeSelection::Feature, KeyCode::Right) => StoryTypeSelection::Bug,
                    (StoryTypeSelection::Bug, KeyCode::Right) => StoryTypeSelection::Task,
                    (StoryTypeSelection::Task, KeyCode::Right) => StoryTypeSelection::Epic,
                    (StoryTypeSelection::Epic, KeyCode::Right) => StoryTypeSelection::Performance,
                    (StoryTypeSelection::Performance, KeyCode::Right) => StoryTypeSelection::Documentation,
                    (StoryTypeSelection::Documentation, KeyCode::Right) => StoryTypeSelection::Testing,
                    (StoryTypeSelection::Testing, KeyCode::Right) => StoryTypeSelection::Research,
                    (StoryTypeSelection::Research, KeyCode::Right) => StoryTypeSelection::Feature,
                    
                    (StoryTypeSelection::Feature, KeyCode::Left) => StoryTypeSelection::Research,
                    (StoryTypeSelection::Bug, KeyCode::Left) => StoryTypeSelection::Feature,
                    (StoryTypeSelection::Task, KeyCode::Left) => StoryTypeSelection::Bug,
                    (StoryTypeSelection::Epic, KeyCode::Left) => StoryTypeSelection::Task,
                    (StoryTypeSelection::Performance, KeyCode::Left) => StoryTypeSelection::Epic,
                    (StoryTypeSelection::Documentation, KeyCode::Left) => StoryTypeSelection::Performance,
                    (StoryTypeSelection::Testing, KeyCode::Left) => StoryTypeSelection::Documentation,
                    (StoryTypeSelection::Research, KeyCode::Left) => StoryTypeSelection::Testing,
                    _ => self.story_form.story_type.clone(),
                };
                true
            }
            KeyCode::Enter => {
                if !self.story_form.title.is_empty() {
                    self.submit_story_creation();
                    self.creating_story = false;
                }
                true
            }
            KeyCode::Char(c) => {
                match self.story_form.current_field {
                    StoryFormField::Title => {
                        self.story_form.title.push(c);
                    }
                    StoryFormField::Description => {
                        self.story_form.description.push(c);
                    }
                    _ => {}
                }
                true
            }
            KeyCode::Backspace => {
                match self.story_form.current_field {
                    StoryFormField::Title => {
                        self.story_form.title.pop();
                    }
                    StoryFormField::Description => {
                        self.story_form.description.pop();
                    }
                    _ => {}
                }
                true
            }
            _ => false,
        }
    }
    
    /// Handle key events during arc creation
    fn handle_arc_creation_key(&mut self, key: KeyCode) -> bool {
        match key {
            KeyCode::Esc => {
                self.creating_arc = false;
                true
            }
            KeyCode::Tab => {
                self.arc_form.current_field = match self.arc_form.current_field {
                    ArcFormField::Title => ArcFormField::Description,
                    ArcFormField::Description => ArcFormField::Title,
                };
                true
            }
            KeyCode::Enter => {
                if !self.arc_form.title.is_empty() {
                    self.submit_arc_creation();
                    self.creating_arc = false;
                }
                true
            }
            KeyCode::Char(c) => {
                match self.arc_form.current_field {
                    ArcFormField::Title => {
                        self.arc_form.title.push(c);
                    }
                    ArcFormField::Description => {
                        self.arc_form.description.push(c);
                    }
                }
                true
            }
            KeyCode::Backspace => {
                match self.arc_form.current_field {
                    ArcFormField::Title => {
                        self.arc_form.title.pop();
                    }
                    ArcFormField::Description => {
                        self.arc_form.description.pop();
                    }
                }
                true
            }
            _ => false,
        }
    }
    
    /// Submit arc creation
    fn submit_arc_creation(&mut self) {
        if let Some(story) = &mut self.selected_story_details {
            if let Some(engine) = &self.story_engine {
                let new_arc = StoryArc {
                    id: StoryArcId(uuid::Uuid::new_v4()),
                    title: self.arc_form.title.clone(),
                    description: self.arc_form.description.clone(),
                    sequence_number: story.arcs.len() as i32,
                    plot_points: Vec::new(),
                    started_at: chrono::Utc::now(),
                    completed_at: None,
                    status: crate::story::types::ArcStatus::Active,
                };
                
                let arc_id = new_arc.id;
                story.arcs.push(new_arc);
                story.current_arc = Some(arc_id);
                story.updated_at = chrono::Utc::now();
                
                // Update in engine
                if let Some(mut story_ref) = engine.stories.get_mut(&story.id) {
                    *story_ref = story.clone();
                }
                
                // Clear the form
                self.arc_form = ArcCreationForm::default();
                self.needs_refresh = true;
                
                tracing::info!("Arc created successfully: {}", self.arc_form.title);
            }
        }
    }
    
    /// Submit story creation
    fn submit_story_creation(&mut self) {
        if let Some(engine) = &self.story_engine {
            // Map the form selection to actual StoryType
            let story_type = match self.story_form.story_type {
                StoryTypeSelection::Feature => StoryType::Feature {
                    feature_name: self.story_form.title.clone(),
                    description: self.story_form.description.clone(),
                },
                StoryTypeSelection::Bug => StoryType::Bug {
                    issue_id: format!("BUG-{}", chrono::Utc::now().timestamp()),
                    severity: "medium".to_string(),
                },
                StoryTypeSelection::Task => StoryType::Task {
                    task_id: format!("TASK-{}", chrono::Utc::now().timestamp()),
                    parent_story: None,
                },
                StoryTypeSelection::Epic => StoryType::Epic {
                    epic_name: self.story_form.title.clone(),
                    objectives: self.story_form.description
                        .lines()
                        .filter(|l| !l.trim().is_empty())
                        .map(|l| l.to_string())
                        .collect(),
                },
                StoryTypeSelection::Performance => StoryType::Performance {
                    component: self.story_form.title.clone(),
                    metrics: vec!["latency".to_string(), "throughput".to_string()],
                },
                StoryTypeSelection::Documentation => StoryType::Documentation {
                    doc_type: "technical".to_string(),
                    target_audience: "developers".to_string(),
                },
                StoryTypeSelection::Testing => StoryType::Testing {
                    test_type: "integration".to_string(),
                    coverage_areas: vec![self.story_form.title.clone()],
                },
                StoryTypeSelection::Research => StoryType::Research {
                    research_topic: self.story_form.title.clone(),
                    hypotheses: self.story_form.description
                        .lines()
                        .filter(|l| !l.trim().is_empty())
                        .map(|l| l.to_string())
                        .collect(),
                },
            };
            
            // Clone needed values before moving into async block
            let title = self.story_form.title.clone();
            let summary = self.story_form.description.clone();
            let engine_clone = engine.clone();
            
            // Spawn async task to create story
            tokio::spawn(async move {
                match engine_clone.create_story(
                    story_type,
                    title.clone(),
                    summary,
                    vec![], // Empty tags for now
                    crate::story::types::Priority::Medium, // Default priority
                ).await {
                    Ok(story_id) => {
                        tracing::info!("Story created successfully: {} ({})", title, story_id);
                    }
                    Err(e) => {
                        tracing::error!("Failed to create story: {}", e);
                    }
                }
            });
            
            // Clear the form
            self.story_form = StoryCreationForm::default();
            
            // Schedule a refresh after a short delay to show the new story
            self.needs_refresh = true;
        }
    }
    
    /// Update selected story details
    fn update_selected_story_details(&mut self) {
        if let Some(selected) = self.list_state.selected() {
            if selected < self.stories.len() {
                self.selected_story_details = Some(self.stories[selected].clone());
            }
        }
    }
    
    // Helper methods
    
    fn event_type_color(&self, event_type: &str) -> Color {
        match event_type {
            "Goal" => Color::Green,
            "Task" => Color::Yellow,
            "Decision" => Color::Cyan,
            "Discovery" => Color::Magenta,
            "Issue" => Color::Red,
            "Transformation" => Color::Blue,
            "Interaction" => Color::LightBlue,
            _ => Color::Gray,
        }
    }
    
    fn create_mini_bar(&self, value: usize, total: usize) -> String {
        let percentage = if total > 0 {
            (value as f32 / total as f32 * 20.0) as usize
        } else {
            0
        };
        
        "█".repeat(percentage) + &"░".repeat(20 - percentage)
    }
    
    /// Render task mapping visualization
    fn render_task_mapping(&self, f: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),  // Header
                Constraint::Min(0),     // Content
            ])
            .split(area);
        
        // Header
        let header_block = Block::default()
            .borders(Borders::ALL)
            .title("Task Mapping Visualization")
            .border_style(Style::default().fg(Color::Cyan));
        
        let header_text = vec![
            Line::from(vec![
                Span::styled("Visual representation of story tasks and their relationships", Style::default().fg(Color::Gray)),
            ]),
        ];
        
        let header_widget = Paragraph::new(header_text)
            .block(header_block);
        
        f.render_widget(header_widget, chunks[0]);
        
        // Content - visualize task dependencies and flow
        if let Some(story) = &self.selected_story_details {
            self.render_task_flow_chart(f, chunks[1], story);
        } else {
            let empty_msg = Paragraph::new("Select a story to view task mapping")
                .block(Block::default().borders(Borders::ALL))
                .style(Style::default().fg(Color::DarkGray))
                .alignment(ratatui::layout::Alignment::Center);
            f.render_widget(empty_msg, chunks[1]);
        }
    }
    
    /// Render task flow chart
    fn render_task_flow_chart(&self, f: &mut Frame, area: Rect, story: &Story) {
        use ratatui::widgets::canvas::{Canvas, Context as CanvasContext, Line as CanvasLine, Points};
        
        let canvas = Canvas::default()
            .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(Color::DarkGray)))
            .marker(ratatui::symbols::Marker::Dot)
            .paint(|ctx: &mut CanvasContext| {
                // Calculate positions for arcs and tasks
                let arc_count = story.arcs.len();
                if arc_count == 0 {
                    return;
                }
                
                let width = 100.0;
                let height = 50.0;
                let arc_spacing = width / (arc_count as f64 + 1.0);
                
                // Draw arcs as vertical columns
                for (arc_idx, arc) in story.arcs.iter().enumerate() {
                    let x = arc_spacing * (arc_idx as f64 + 1.0);
                    
                    // Draw arc title
                    // Note: Canvas print requires owned strings due to lifetime constraints
                    ctx.print(x, height - 2.0, arc.title.clone());
                    
                    // Draw tasks within arc
                    let task_count = arc.plot_points.iter()
                        .filter(|pp| matches!(pp.plot_type, PlotType::Task { .. }))
                        .count();
                    
                    if task_count > 0 {
                        let task_spacing = (height - 10.0) / (task_count as f64 + 1.0);
                        let mut task_idx = 0;
                        
                        for plot_point in &arc.plot_points {
                            if let PlotType::Task { description, completed } = &plot_point.plot_type {
                                let y = height - 10.0 - (task_spacing * (task_idx as f64 + 1.0));
                                
                                // Draw task point
                                let color = if *completed { Color::Green } else { Color::Yellow };
                                ctx.draw(&Points {
                                    coords: &[(x, y)],
                                    color,
                                });
                                
                                // Draw task description (truncated)
                                let truncated = if description.len() > 15 {
                                    format!("{}...", &description[..12])
                                } else {
                                    description.clone()
                                };
                                ctx.print(x + 2.0, y, truncated);
                                
                                // Draw connections to next arc if exists
                                if arc_idx < arc_count - 1 {
                                    let next_x = arc_spacing * (arc_idx as f64 + 2.0);
                                    ctx.draw(&CanvasLine {
                                        x1: x,
                                        y1: y,
                                        x2: next_x,
                                        y2: y,
                                        color: Color::DarkGray,
                                    });
                                }
                                
                                task_idx += 1;
                            }
                        }
                    }
                }
            })
            .x_bounds([0.0, 100.0])
            .y_bounds([0.0, 50.0]);
        
        f.render_widget(canvas, area);
    }
    
    /// Get icon for story type
    fn get_story_type_icon(&self, story_type: &crate::story::types::StoryType) -> &'static str {
        use crate::story::types::StoryType;
        match story_type {
            StoryType::Feature { .. } => "✨",
            StoryType::Bug { .. } => "🐛",
            StoryType::Task { .. } => "📋",
            StoryType::Epic { .. } => "🏔️",
            StoryType::Performance { .. } => "⚡",
            StoryType::Documentation { .. } => "📚",
            StoryType::Testing { .. } => "🧪",
            StoryType::Research { .. } => "🔬",
            StoryType::Security { .. } => "🔐",
            StoryType::Refactoring { .. } => "🔧",
            StoryType::Learning { .. } => "🎓",
            StoryType::Deployment { .. } => "🚀",
            StoryType::Dependencies { .. } => "📦",
            StoryType::System { .. } => "⚙️",
            StoryType::Agent { .. } => "🤖",
            StoryType::Codebase { .. } => "💻",
            StoryType::Directory { .. } => "📁",
            StoryType::File { .. } => "📄",
        }
    }
    
    /// Get color for story status
    fn get_status_color(&self, status: StoryStatus) -> Style {
        match status {
            StoryStatus::NotStarted => Style::default().fg(Color::Yellow),
            StoryStatus::Draft => Style::default().fg(Color::Gray),
            StoryStatus::Active => Style::default().fg(Color::Green),
            StoryStatus::Completed => Style::default().fg(Color::Blue),
            StoryStatus::Archived => Style::default().fg(Color::DarkGray),
        }
    }
}