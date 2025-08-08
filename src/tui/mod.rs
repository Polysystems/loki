//! Terminal User Interface module

// Core modules
pub mod app;
pub mod run;
pub mod state;

// Event-driven architecture
pub mod event_bus;
pub mod shared_state;
pub mod tab_registry;
pub mod integration_hub;
pub mod state_sync;
pub mod ui_bridge;
pub mod error_handling;

// Bridge infrastructure for cross-tab communication
pub mod bridges;

// Connector modules
pub mod autonomous_intelligence_connector;
pub mod system_connector;
pub mod tool_connector;
pub mod autonomous_data_types;
pub mod system_monitor;

// Session and state management
pub mod session_manager;
pub mod cluster_state;
pub mod settings;

// Story and analysis modules
pub mod story_driven_code_analysis;
pub mod story_memory_integration;

// Processing and integration modules
pub mod cognitive_stream_integration;
pub mod autonomous_realtime_updater;
pub mod real_time_integration;

// Natural language processing is in nlp/ subdirectory

// Chat system modules are in chat/ subdirectory

// View modules
pub mod agent_specialization_view;
pub mod agent_task_mapper;
pub mod collaborative_view;
pub mod cost_optimization_view;
pub mod orchestration_view;
pub mod plugin_view;

// Task and monitoring
pub mod task_decomposer;
pub mod task_progress_aggregator;
pub mod multiplexer;

// UI components
pub mod components;
pub mod visual_components;
pub mod widgets;
pub mod x;

// Modular subdirectories
pub mod chat;
pub mod cognitive;
pub mod nlp;
pub mod monitoring;
pub mod ui;  // Will be removed after migration
pub mod tabs;  // Moved up from ui/tabs/

// Backwards compatibility alias
pub mod connectors {
    pub use super::system_connector;
    pub use super::tool_connector;
}

// Re-export main types
pub use app::App;
pub use system_connector::SystemConnector;
pub use tool_connector::ToolSystemConnector;

// TUI entry point function
pub async fn run_tui(
    compute_manager: std::sync::Arc<crate::compute::ComputeManager>,
    stream_manager: std::sync::Arc<crate::streaming::StreamManager>,
    cluster_manager: Option<std::sync::Arc<crate::cluster::ClusterManager>>,
) -> anyhow::Result<()> {
    use crossterm::{
        execute,
        terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    };
    use ratatui::{
        backend::CrosstermBackend,
        Terminal,
    };
    use std::io;
    
    // Create the TUI app with required cluster manager
    let cluster_manager = match cluster_manager {
        Some(cm) => cm,
        None => {
            std::sync::Arc::new(crate::cluster::ClusterManager::new(
                crate::cluster::ClusterConfig::default()
            ).await?)
        }
    };
    
    let mut app = App::new(compute_manager, stream_manager, cluster_manager).await?;
    
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    
    // Run the main loop
    let res = run_app(&mut terminal, &mut app).await;
    
    // Restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
    )?;
    terminal.show_cursor()?;
    
    if let Err(err) = res {
        println!("Error: {:?}", err);
        return Err(err);
    }
    
    Ok(())
}

// Main application loop
async fn run_app<B: ratatui::backend::Backend>(
    terminal: &mut ratatui::Terminal<B>,
    app: &mut App,
) -> anyhow::Result<()> {
    use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
    use std::time::Duration;
    use crate::tui::state::ViewState;
    
    loop {
        // Draw the UI using the full UI system
        terminal.draw(|f| {
            crate::tui::ui::draw(f, app);
        })?;
        
        // Handle input
        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') if key.modifiers.contains(KeyModifiers::CONTROL) => return Ok(()),
                        // Tab switching with number keys
                        KeyCode::Char('1') => app.state.current_view = ViewState::Dashboard,
                        KeyCode::Char('2') => app.state.current_view = ViewState::Chat,
                        KeyCode::Char('3') => app.state.current_view = ViewState::Utilities,
                        KeyCode::Char('4') => app.state.current_view = ViewState::Memory,
                        KeyCode::Char('5') => app.state.current_view = ViewState::Cognitive,
                        KeyCode::Char('6') => app.state.current_view = ViewState::Streams,
                        KeyCode::Char('7') => app.state.current_view = ViewState::Models,
                        // Tab key navigation
                        KeyCode::Tab if !key.modifiers.contains(KeyModifiers::SHIFT) => {
                            // Cycle through tabs forward
                            let next_view = match app.state.current_view {
                                ViewState::Dashboard => ViewState::Chat,
                                ViewState::Chat => ViewState::Utilities,
                                ViewState::Utilities => ViewState::Memory,
                                ViewState::Memory => ViewState::Cognitive,
                                ViewState::Cognitive => ViewState::Streams,
                                ViewState::Streams => ViewState::Models,
                                ViewState::Models => ViewState::Dashboard,
                                _ => ViewState::Dashboard,
                            };
                            app.state.current_view = next_view;
                        }
                        KeyCode::BackTab | KeyCode::Tab if key.modifiers.contains(KeyModifiers::SHIFT) => {
                            // Cycle through tabs backward
                            let prev_view = match app.state.current_view {
                                ViewState::Dashboard => ViewState::Models,
                                ViewState::Chat => ViewState::Dashboard,
                                ViewState::Utilities => ViewState::Chat,
                                ViewState::Memory => ViewState::Utilities,
                                ViewState::Cognitive => ViewState::Memory,
                                ViewState::Streams => ViewState::Cognitive,
                                ViewState::Models => ViewState::Streams,
                                _ => ViewState::Dashboard,
                            };
                            app.state.current_view = prev_view;
                        }
                        _ => {
                            // Handle other keys with the app - use handle_key which has full input handling
                            app.handle_key(key).await?;
                        }
                    }
                }
            }
        }
        
        // Update app state
        app.update().await?;
    }
}