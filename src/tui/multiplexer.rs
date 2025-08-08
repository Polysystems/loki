//! Terminal Multiplexer for Loki
//!
//! Provides tmux/screen-like functionality integrated with Loki's cognitive and
//! monitoring systems. Features include session management, pane splitting,
//! AI-assisted commands, and persistence.

use std::collections::HashMap;
use std::io::{Stdout, stdout};
use std::process::Stdio;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen,
    LeaveAlternateScreen,
    disable_raw_mode,
    enable_raw_mode,
};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Style};
use ratatui::widgets::{Block, Borders, List, ListItem, Paragraph, Wrap};
use ratatui::{Frame, Terminal};
use serde::{Deserialize, Serialize};
use tokio::io::AsyncWriteExt;
use tokio::process::{Child as TokioChild, Command as TokioCommand};
use tokio::sync::{RwLock, broadcast, mpsc};
use tokio::time::interval;
use tracing::{debug, error, info};
use uuid::Uuid;

use crate::cognitive::CognitiveSystem;
use crate::monitoring::AdvancedMonitoring;

/// Unique identifier for sessions
pub type SessionId = Uuid;

/// Unique identifier for panes
pub type PaneId = Uuid;

/// Terminal multiplexer session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiplexerSession {
    /// Unique session identifier
    pub id: SessionId,

    /// Session name
    pub name: String,

    /// Session creation time
    pub created_at: u64,

    /// Last activity time
    pub last_activity: u64,

    /// Session status
    pub status: SessionStatus,

    /// Layout configuration
    pub layout: SessionLayout,

    /// Active panes in this session
    pub panes: HashMap<PaneId, MultiplexerPane>,

    /// Currently active pane
    pub active_pane: Option<PaneId>,

    /// Session-specific environment variables
    pub environment: HashMap<String, String>,

    /// Working directory
    pub working_directory: String,

    /// Session metadata
    pub metadata: SessionMetadata,
}

/// Session status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SessionStatus {
    Active,
    Detached,
    Suspended,
    Terminated,
}

/// Session layout configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionLayout {
    /// Layout type
    pub layout_type: LayoutType,

    /// Split configuration
    pub splits: Vec<LayoutSplit>,

    /// Pane constraints
    pub constraints: Vec<LayoutConstraint>,
}

/// Layout types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LayoutType {
    Single,
    Horizontal,
    Vertical,
    Grid { rows: usize, cols: usize },
    Custom,
}

/// Layout split configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutSplit {
    /// Split direction
    pub direction: SplitDirection,

    /// Split ratio (0.0 to 1.0)
    pub ratio: f32,

    /// Child panes
    pub panes: Vec<PaneId>,
}

/// Split direction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SplitDirection {
    Horizontal,
    Vertical,
}

/// Layout constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LayoutConstraint {
    Percentage(u16),
    Length(u16),
    Min(u16),
    Max(u16),
    Ratio(u32, u32),
}

/// Session metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMetadata {
    /// Session tags
    pub tags: Vec<String>,

    /// Session description
    pub description: String,

    /// Auto-attach on startup
    pub auto_attach: bool,

    /// Persist across restarts
    pub persistent: bool,

    /// Associated project
    pub project: Option<String>,
}

/// Multiplexer pane
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiplexerPane {
    /// Unique pane identifier
    pub id: PaneId,

    /// Pane name
    pub name: String,

    /// Pane type
    pub pane_type: PaneType,

    /// Pane status
    pub status: PaneStatus,

    /// Pane size and position
    pub geometry: PaneGeometry,

    /// Pane content buffer
    pub buffer: PaneBuffer,

    /// Pane-specific configuration
    pub config: PaneConfig,

    /// Creation time
    pub created_at: u64,

    /// Last activity
    pub last_activity: u64,
}

/// Pane types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PaneType {
    /// Standard shell terminal
    Terminal { shell: String, command: Option<String> },

    /// Loki monitoring dashboard
    MonitoringDashboard,

    /// Loki cognitive interface
    CognitiveInterface,

    /// Log viewer
    LogViewer { file_path: String, follow: bool },

    /// File editor
    Editor { file_path: String, mode: EditorMode },

    /// Custom application
    Application { command: String, args: Vec<String> },

    /// AI-assisted terminal
    AiTerminal { model: String, context: String },
}

/// Editor modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EditorMode {
    View,
    Edit,
    Diff,
}

/// Pane status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PaneStatus {
    Active,
    Inactive,
    Running,
    Suspended,
    Completed,
    Error,
}

/// Pane geometry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaneGeometry {
    pub x: u16,
    pub y: u16,
    pub width: u16,
    pub height: u16,
}

/// Pane content buffer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaneBuffer {
    /// Text lines
    pub lines: Vec<String>,

    /// Maximum buffer size
    pub max_lines: usize,

    /// Current scroll position
    pub scroll_position: usize,

    /// Cursor position
    pub cursor_position: (u16, u16),

    /// Buffer metadata
    pub metadata: BufferMetadata,
}

/// Buffer metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BufferMetadata {
    /// Total bytes
    pub total_bytes: usize,

    /// Line count
    pub line_count: usize,

    /// Last modified
    pub last_modified: u64,

    /// Encoding
    pub encoding: String,
}

/// Pane configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaneConfig {
    /// Auto-scroll
    pub auto_scroll: bool,

    /// Word wrap
    pub word_wrap: bool,

    /// Show line numbers
    pub show_line_numbers: bool,

    /// Color scheme
    pub color_scheme: String,

    /// Font configuration
    pub font: FontConfig,

    /// Key bindings
    pub key_bindings: HashMap<String, String>,
}

/// Font configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FontConfig {
    pub family: String,
    pub size: u16,
    pub bold: bool,
    pub italic: bool,
}

/// Multiplexer command types
#[derive(Debug, Clone)]
pub enum MultiplexerCommand {
    /// Create new session
    CreateSession { name: String, layout: Option<LayoutType>, working_dir: Option<String> },

    /// Attach to session
    AttachSession { session_id: SessionId },

    /// Detach from session
    DetachSession,

    /// List sessions
    ListSessions,

    /// Create new pane
    CreatePane {
        session_id: SessionId,
        pane_type: PaneType,
        split_direction: Option<SplitDirection>,
    },

    /// Split pane
    SplitPane { session_id: SessionId, pane_id: PaneId, direction: SplitDirection, ratio: f32 },

    /// Switch to pane
    SwitchPane { session_id: SessionId, pane_id: PaneId },

    /// Close pane
    ClosePane { session_id: SessionId, pane_id: PaneId },

    /// Resize pane
    ResizePane { session_id: SessionId, pane_id: PaneId, width: u16, height: u16 },

    /// Send input to pane
    SendInput { session_id: SessionId, pane_id: PaneId, input: String },

    /// Copy mode
    CopyMode { session_id: SessionId, pane_id: PaneId },

    /// Paste content
    Paste { session_id: SessionId, pane_id: PaneId, content: String },

    /// AI assistance
    AiAssist { session_id: SessionId, pane_id: PaneId, query: String },
}

/// Multiplexer events
#[derive(Debug, Clone)]
pub enum MultiplexerEvent {
    /// Session created
    SessionCreated { session: MultiplexerSession },

    /// Session attached
    SessionAttached { session_id: SessionId },

    /// Session detached
    SessionDetached { session_id: SessionId },

    /// Pane created
    PaneCreated { session_id: SessionId, pane: MultiplexerPane },

    /// Pane output
    PaneOutput { session_id: SessionId, pane_id: PaneId, output: String },

    /// Pane closed
    PaneClosed { session_id: SessionId, pane_id: PaneId },

    /// Layout changed
    LayoutChanged { session_id: SessionId, layout: SessionLayout },

    /// AI assistance
    AiResponse { session_id: SessionId, pane_id: PaneId, response: String },
}

/// Terminal multiplexer
pub struct TerminalMultiplexer {
    /// All sessions
    sessions: Arc<RwLock<HashMap<SessionId, MultiplexerSession>>>,

    /// Currently attached session
    current_session: Arc<RwLock<Option<SessionId>>>,

    /// Terminal interface
    terminal: Option<Terminal<CrosstermBackend<Stdout>>>,

    /// Command sender
    command_tx: mpsc::Sender<MultiplexerCommand>,

    /// Command receiver
    command_rx: Option<mpsc::Receiver<MultiplexerCommand>>,

    /// Event broadcaster
    event_tx: broadcast::Sender<MultiplexerEvent>,

    /// Running processes
    processes: Arc<RwLock<HashMap<PaneId, TokioChild>>>,

    /// Cognitive system integration
    cognitive_system: Option<Arc<CognitiveSystem>>,

    /// Monitoring system integration
    monitoring: Option<Arc<AdvancedMonitoring>>,

    /// Configuration
    config: MultiplexerConfig,

    /// Clipboard
    clipboard: Arc<RwLock<String>>,

    /// Session persistence
    persistence_enabled: bool,
}

/// Multiplexer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiplexerConfig {
    /// Default shell
    pub default_shell: String,

    /// Maximum sessions
    pub max_sessions: usize,

    /// Maximum panes per session
    pub max_panes_per_session: usize,

    /// Buffer size per pane
    pub buffer_size: usize,

    /// Auto-save interval
    pub auto_save_interval: Duration,

    /// Session timeout
    pub session_timeout: Duration,

    /// Key bindings
    pub key_bindings: KeyBindings,

    /// UI settings
    pub ui: UiConfig,
}

/// Key bindings configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyBindings {
    /// Prefix key
    pub prefix: String,

    /// Session commands
    pub sessions: HashMap<String, String>,

    /// Pane commands
    pub panes: HashMap<String, String>,

    /// Window commands
    pub windows: HashMap<String, String>,

    /// Copy mode
    pub copy_mode: HashMap<String, String>,
}

/// UI configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UiConfig {
    /// Color scheme
    pub color_scheme: String,

    /// Border style
    pub border_style: String,

    /// Status bar
    pub status_bar: StatusBarConfig,

    /// Pane titles
    pub pane_titles: bool,

    /// Activity indicators
    pub activity_indicators: bool,
}

/// Status bar configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusBarConfig {
    /// Show status bar
    pub enabled: bool,

    /// Position
    pub position: StatusBarPosition,

    /// Format string
    pub format: String,

    /// Update interval
    pub update_interval: Duration,
}

/// Status bar position
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StatusBarPosition {
    Top,
    Bottom,
}

impl Default for MultiplexerConfig {
    fn default() -> Self {
        let mut session_bindings = HashMap::new();
        session_bindings.insert("c".to_string(), "new-session".to_string());
        session_bindings.insert("d".to_string(), "detach".to_string());
        session_bindings.insert("s".to_string(), "list-sessions".to_string());
        session_bindings.insert("$".to_string(), "rename-session".to_string());

        let mut pane_bindings = HashMap::new();
        pane_bindings.insert("\"".to_string(), "split-horizontal".to_string());
        pane_bindings.insert("%".to_string(), "split-vertical".to_string());
        pane_bindings.insert("x".to_string(), "kill-pane".to_string());
        pane_bindings.insert("o".to_string(), "next-pane".to_string());
        pane_bindings.insert("{".to_string(), "swap-pane-up".to_string());
        pane_bindings.insert("}".to_string(), "swap-pane-down".to_string());

        let mut copy_bindings = HashMap::new();
        copy_bindings.insert("[".to_string(), "copy-mode".to_string());
        copy_bindings.insert("]".to_string(), "paste".to_string());
        copy_bindings.insert("y".to_string(), "copy-selection".to_string());

        Self {
            default_shell: std::env::var("SHELL").unwrap_or_else(|_| "/bin/bash".to_string()),
            max_sessions: 100,
            max_panes_per_session: 50,
            buffer_size: 10000,
            auto_save_interval: Duration::from_secs(60),
            session_timeout: Duration::from_secs(86400), // 24 hours
            key_bindings: KeyBindings {
                prefix: "C-b".to_string(),
                sessions: session_bindings,
                panes: pane_bindings,
                windows: HashMap::new(),
                copy_mode: copy_bindings,
            },
            ui: UiConfig {
                color_scheme: "default".to_string(),
                border_style: "rounded".to_string(),
                status_bar: StatusBarConfig {
                    enabled: true,
                    position: StatusBarPosition::Bottom,
                    format: "[#{session_name}] #{pane_index}: #{pane_title} | #{cpu}% CPU | \
                             ${cost:.2}"
                        .to_string(),
                    update_interval: Duration::from_secs(1),
                },
                pane_titles: true,
                activity_indicators: true,
            },
        }
    }
}

impl TerminalMultiplexer {
    /// Create a new terminal multiplexer
    pub fn new(
        config: Option<MultiplexerConfig>,
        cognitive_system: Option<Arc<CognitiveSystem>>,
        monitoring: Option<Arc<AdvancedMonitoring>>,
    ) -> Result<Self> {
        let config = config.unwrap_or_default();
        let (command_tx, command_rx) = mpsc::channel(1000);
        let (event_tx, _) = broadcast::channel(1000);

        // Initialize terminal
        enable_raw_mode()?;
        let mut stdout = stdout();
        execute!(stdout, EnterAlternateScreen)?;
        let backend = CrosstermBackend::new(stdout);
        let terminal = Terminal::new(backend)?;

        Ok(Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
            current_session: Arc::new(RwLock::new(None)),
            terminal: Some(terminal),
            command_tx,
            command_rx: Some(command_rx),
            event_tx,
            processes: Arc::new(RwLock::new(HashMap::new())),
            cognitive_system,
            monitoring,
            config,
            clipboard: Arc::new(RwLock::new(String::new())),
            persistence_enabled: true,
        })
    }

    /// Start the multiplexer
    pub async fn start(&mut self) -> Result<()> {
        info!("Starting terminal multiplexer");

        // Start command processor
        self.start_command_processor().await?;

        // Start session manager
        self.start_session_manager().await?;

        // Load persisted sessions
        if self.persistence_enabled {
            self.load_sessions().await?;
        }

        // Create default session if none exist
        let sessions = self.sessions.read().await;
        if sessions.is_empty() {
            drop(sessions);
            self.create_default_session().await?;
        }

        info!("Terminal multiplexer started");
        Ok(())
    }

    /// Run the multiplexer (blocking)
    pub async fn run(&mut self) -> Result<()> {
        info!("Running terminal multiplexer interface");

        let mut last_tick = Instant::now();
        let tick_rate = Duration::from_millis(50);

        loop {
            // Handle events
            if event::poll(Duration::from_millis(100))? {
                if let Event::Key(key) = event::read()? {
                    if key.kind == KeyEventKind::Press {
                        if self.handle_key_event(key.code, key.modifiers).await? {
                            break; // Exit requested
                        }
                    }
                }
            }

            // Update UI
            if last_tick.elapsed() >= tick_rate {
                if let Some(ref mut terminal) = self.terminal {
                    // Pre-fetch data outside of the draw closure
                    let sessions_snapshot = {
                        let sessions = self.sessions.read().await;
                        sessions.clone()
                    };
                    let current_session_id = {
                        let current = self.current_session.read().await;
                        *current
                    };
                    let monitoring = self.monitoring.clone();

                    terminal.draw(|f| {
                        // Use synchronous drawing with pre-fetched data
                        if let Some(session_id) = current_session_id {
                            if let Some(session) = sessions_snapshot.get(&session_id) {
                                draw_session_non_async(f, session, &monitoring);
                            } else {
                                draw_no_session_non_async(f);
                            }
                        } else {
                            draw_no_session_non_async(f);
                        }
                    })?;
                }
                last_tick = Instant::now();
            }
        }

        self.cleanup()?;
        Ok(())
    }

    /// Start command processor
    async fn start_command_processor(&mut self) -> Result<()> {
        let mut command_rx = self.command_rx.take().unwrap();
        let sessions = self.sessions.clone();
        let current_session = self.current_session.clone();
        let event_tx = self.event_tx.clone();
        let processes = self.processes.clone();
        let cognitive_system = self.cognitive_system.clone();
        let monitoring = self.monitoring.clone();
        let config = self.config.clone();
        let clipboard = self.clipboard.clone();

        tokio::spawn(async move {
            while let Some(command) = command_rx.recv().await {
                if let Err(e) = Self::process_command(
                    command,
                    &sessions,
                    &current_session,
                    &event_tx,
                    &processes,
                    &cognitive_system,
                    &monitoring,
                    &config,
                    &clipboard,
                )
                .await
                {
                    error!("Failed to process command: {}", e);
                }
            }
        });

        Ok(())
    }

    /// Process a multiplexer command
    async fn process_command(
        command: MultiplexerCommand,
        sessions: &Arc<RwLock<HashMap<SessionId, MultiplexerSession>>>,
        current_session: &Arc<RwLock<Option<SessionId>>>,
        event_tx: &broadcast::Sender<MultiplexerEvent>,
        processes: &Arc<RwLock<HashMap<PaneId, TokioChild>>>,
        cognitive_system: &Option<Arc<CognitiveSystem>>,
        _monitoring: &Option<Arc<AdvancedMonitoring>>,
        config: &MultiplexerConfig,
        _clipboard: &Arc<RwLock<String>>,
    ) -> Result<()> {
        match command {
            MultiplexerCommand::CreateSession { name, layout, working_dir } => {
                let session_id = Uuid::new_v4();
                let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();

                let session = MultiplexerSession {
                    id: session_id,
                    name: name.clone(),
                    created_at: timestamp,
                    last_activity: timestamp,
                    status: SessionStatus::Active,
                    layout: SessionLayout {
                        layout_type: layout.unwrap_or(LayoutType::Single),
                        splits: Vec::new(),
                        constraints: vec![LayoutConstraint::Percentage(100)],
                    },
                    panes: HashMap::new(),
                    active_pane: None,
                    environment: std::env::vars().collect(),
                    working_directory: working_dir.unwrap_or_else(|| {
                        std::env::current_dir().unwrap_or_default().to_string_lossy().to_string()
                    }),
                    metadata: SessionMetadata {
                        tags: Vec::new(),
                        description: format!("Session {}", name),
                        auto_attach: false,
                        persistent: true,
                        project: None,
                    },
                };

                // Add session
                {
                    let mut sessions_guard = sessions.write().await;
                    sessions_guard.insert(session_id, session.clone());
                }

                // Set as current session
                {
                    let mut current = current_session.write().await;
                    *current = Some(session_id);
                }

                // Broadcast event
                let _ = event_tx.send(MultiplexerEvent::SessionCreated { session });

                info!("Created session: {} ({})", name, session_id);
            }

            MultiplexerCommand::CreatePane { session_id, pane_type, split_direction: _split_direction } => {
                let pane_id = Uuid::new_v4();
                let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();

                let pane = MultiplexerPane {
                    id: pane_id,
                    name: format!("pane-{}", pane_id.to_string()[..8].to_string()),
                    pane_type: pane_type.clone(),
                    status: PaneStatus::Active,
                    geometry: PaneGeometry { x: 0, y: 0, width: 80, height: 24 },
                    buffer: PaneBuffer {
                        lines: Vec::new(),
                        max_lines: config.buffer_size,
                        scroll_position: 0,
                        cursor_position: (0, 0),
                        metadata: BufferMetadata {
                            total_bytes: 0,
                            line_count: 0,
                            last_modified: timestamp,
                            encoding: "UTF-8".to_string(),
                        },
                    },
                    config: PaneConfig {
                        auto_scroll: true,
                        word_wrap: true,
                        show_line_numbers: false,
                        color_scheme: "default".to_string(),
                        font: FontConfig {
                            family: "monospace".to_string(),
                            size: 12,
                            bold: false,
                            italic: false,
                        },
                        key_bindings: HashMap::new(),
                    },
                    created_at: timestamp,
                    last_activity: timestamp,
                };

                // Add pane to session
                {
                    let mut sessions_guard = sessions.write().await;
                    if let Some(session) = sessions_guard.get_mut(&session_id) {
                        session.panes.insert(pane_id, pane.clone());
                        if session.active_pane.is_none() {
                            session.active_pane = Some(pane_id);
                        }
                        session.last_activity = timestamp;
                    }
                }

                // Start process for terminal panes
                if let PaneType::Terminal { shell, command } = &pane_type {
                    Self::start_terminal_process(pane_id, shell, command.as_deref(), processes)
                        .await?;
                }

                // Broadcast event
                let _ = event_tx.send(MultiplexerEvent::PaneCreated { session_id, pane });

                info!("Created pane: {} in session {}", pane_id, session_id);
            }

            MultiplexerCommand::SendInput { session_id: _session_id, pane_id, input } => {
                // Send input to pane process
                {
                    let mut processes_guard = processes.write().await;
                    if let Some(process) = processes_guard.get_mut(&pane_id) {
                        if let Some(stdin) = process.stdin.as_mut() {
                            stdin.write_all(input.as_bytes()).await?;
                            stdin.write_all(b"\n").await?;
                            stdin.flush().await?;
                        }
                    }
                }

                debug!("Sent input to pane {}: {}", pane_id, input);
            }

            MultiplexerCommand::AiAssist { session_id, pane_id, query } => {
                if let Some(_cognitive) = cognitive_system {
                    // Get AI assistance
                    // This would integrate with the cognitive system
                    let response = format!("AI assistance for: {}", query);

                    let _ = event_tx.send(MultiplexerEvent::AiResponse {
                        session_id,
                        pane_id,
                        response,
                    });
                }
            }

            // Handle other commands...
            _ => {
                debug!("Command not yet implemented: {:?}", command);
            }
        }

        Ok(())
    }

    /// Start a terminal process for a pane
    async fn start_terminal_process(
        pane_id: PaneId,
        shell: &str,
        command: Option<&str>,
        processes: &Arc<RwLock<HashMap<PaneId, TokioChild>>>,
    ) -> Result<()> {
        let mut cmd = TokioCommand::new(shell);

        if let Some(command) = command {
            cmd.arg("-c").arg(command);
        }

        cmd.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());

        let child = cmd.spawn()?;

        // Store the process
        {
            let mut processes_guard = processes.write().await;
            processes_guard.insert(pane_id, child);
        }

        info!("Started terminal process for pane {}", pane_id);
        Ok(())
    }

    /// Start session manager
    async fn start_session_manager(&self) -> Result<()> {
        let sessions = self.sessions.clone();
        let config = self.config.clone();

        tokio::spawn(async move {
            let mut interval = interval(config.auto_save_interval);

            loop {
                interval.tick().await;

                // Auto-save sessions
                if let Err(e) = Self::save_sessions(&sessions).await {
                    error!("Failed to auto-save sessions: {}", e);
                }

                // Check for timeout sessions
                let current_time =
                    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();

                let mut sessions_guard = sessions.write().await;
                let mut to_remove = Vec::new();

                for (session_id, session) in sessions_guard.iter_mut() {
                    if current_time - session.last_activity > config.session_timeout.as_secs() {
                        session.status = SessionStatus::Terminated;
                        to_remove.push(*session_id);
                    }
                }

                for session_id in to_remove {
                    sessions_guard.remove(&session_id);
                    info!("Removed timeout session: {}", session_id);
                }
            }
        });

        Ok(())
    }

    /// Load persisted sessions
    async fn load_sessions(&self) -> Result<()> {
        // Implementation would load from disk
        info!("Loading persisted sessions");
        Ok(())
    }

    /// Save sessions to persistence
    async fn save_sessions(
        _sessions: &Arc<RwLock<HashMap<SessionId, MultiplexerSession>>>,
    ) -> Result<()> {
        // Implementation would save to disk
        debug!("Auto-saving sessions");
        Ok(())
    }

    /// Create a default session
    async fn create_default_session(&self) -> Result<()> {
        let command = MultiplexerCommand::CreateSession {
            name: "main".to_string(),
            layout: Some(LayoutType::Single),
            working_dir: None,
        };

        self.command_tx.send(command).await?;
        Ok(())
    }

    /// Handle keyboard input
    async fn handle_key_event(&mut self, key: KeyCode, modifiers: KeyModifiers) -> Result<bool> {
        match (key, modifiers) {
            (KeyCode::Char('q'), KeyModifiers::CONTROL) => return Ok(true), // Exit
            (KeyCode::Char('c'), KeyModifiers::CONTROL) => {
                // Create new session
                let command = MultiplexerCommand::CreateSession {
                    name: format!("session-{}", Uuid::new_v4().to_string()[..8].to_string()),
                    layout: Some(LayoutType::Single),
                    working_dir: None,
                };
                self.command_tx.send(command).await?;
            }
            (KeyCode::Char('d'), KeyModifiers::CONTROL) => {
                // Detach session
                let command = MultiplexerCommand::DetachSession;
                self.command_tx.send(command).await?;
            }
            (KeyCode::Char('"'), KeyModifiers::CONTROL) => {
                // Split horizontal
                if let Some(session_id) = *self.current_session.read().await {
                    let command = MultiplexerCommand::CreatePane {
                        session_id,
                        pane_type: PaneType::Terminal {
                            shell: self.config.default_shell.clone(),
                            command: None,
                        },
                        split_direction: Some(SplitDirection::Horizontal),
                    };
                    self.command_tx.send(command).await?;
                }
            }
            (KeyCode::Char('%'), KeyModifiers::CONTROL) => {
                // Split vertical
                if let Some(session_id) = *self.current_session.read().await {
                    let command = MultiplexerCommand::CreatePane {
                        session_id,
                        pane_type: PaneType::Terminal {
                            shell: self.config.default_shell.clone(),
                            command: None,
                        },
                        split_direction: Some(SplitDirection::Vertical),
                    };
                    self.command_tx.send(command).await?;
                }
            }
            _ => {}
        }

        Ok(false)
    }

    /// Draw the multiplexer interface
    async fn draw_multiplexer(&self, f: &mut Frame<'_>) -> Result<()> {
        let sessions = self.sessions.read().await;
        let current_session_id = *self.current_session.read().await;

        if let Some(session_id) = current_session_id {
            if let Some(session) = sessions.get(&session_id) {
                self.draw_session(f, session).await?;
            } else {
                self.draw_no_session(f).await?;
            }
        } else {
            self.draw_session_list(f, &sessions).await?;
        }

        Ok(())
    }

    /// Draw a session
    async fn draw_session(&self, f: &mut Frame<'_>, session: &MultiplexerSession) -> Result<()> {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(0),    // Main content
                Constraint::Length(1), // Status bar
            ])
            .split(f.area());

        // Draw panes
        self.draw_panes(f, chunks[0], session).await?;

        // Draw status bar
        self.draw_status_bar(f, chunks[1], session).await?;

        Ok(())
    }

    /// Draw panes in session
    async fn draw_panes(
        &self,
        f: &mut Frame<'_>,
        area: Rect,
        session: &MultiplexerSession,
    ) -> Result<()> {
        if session.panes.is_empty() {
            let paragraph = Paragraph::new(
                "No panes in session. Press Ctrl+\" to split horizontally or Ctrl+% to split \
                 vertically.",
            )
            .block(Block::default().borders(Borders::ALL).title("Empty Session"));
            f.render_widget(paragraph, area);
            return Ok(());
        }

        // Simple layout for now - could be made more sophisticated
        match session.layout.layout_type {
            LayoutType::Single => {
                if let Some(pane) = session.panes.values().next() {
                    self.draw_pane(f, area, pane, true).await?;
                }
            }
            LayoutType::Horizontal => {
                let constraints: Vec<Constraint> = session
                    .panes
                    .iter()
                    .map(|_| Constraint::Percentage(100 / session.panes.len() as u16))
                    .collect();

                let chunks = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints(constraints)
                    .split(area);

                for (i, pane) in session.panes.values().enumerate() {
                    if i < chunks.len() {
                        let is_active = session.active_pane == Some(pane.id);
                        self.draw_pane(f, chunks[i], pane, is_active).await?;
                    }
                }
            }
            LayoutType::Vertical => {
                let constraints: Vec<Constraint> = session
                    .panes
                    .iter()
                    .map(|_| Constraint::Percentage(100 / session.panes.len() as u16))
                    .collect();

                let chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints(constraints)
                    .split(area);

                for (i, pane) in session.panes.values().enumerate() {
                    if i < chunks.len() {
                        let is_active = session.active_pane == Some(pane.id);
                        self.draw_pane(f, chunks[i], pane, is_active).await?;
                    }
                }
            }
            _ => {
                // Default to single pane
                if let Some(pane) = session.panes.values().next() {
                    self.draw_pane(f, area, pane, true).await?;
                }
            }
        }

        Ok(())
    }

    /// Draw a single pane
    async fn draw_pane(
        &self,
        f: &mut Frame<'_>,
        area: Rect,
        pane: &MultiplexerPane,
        is_active: bool,
    ) -> Result<()> {
        let border_style = if is_active {
            Style::default().fg(Color::Cyan)
        } else {
            Style::default().fg(Color::Gray)
        };

        let title = match &pane.pane_type {
            PaneType::Terminal { .. } => format!("Terminal: {}", pane.name),
            PaneType::MonitoringDashboard => "Monitoring Dashboard".to_string(),
            PaneType::CognitiveInterface => "Cognitive Interface".to_string(),
            PaneType::LogViewer { file_path, .. } => format!("Log: {}", file_path),
            PaneType::Editor { file_path, .. } => format!("Editor: {}", file_path),
            PaneType::Application { command, .. } => format!("App: {}", command),
            PaneType::AiTerminal { model, .. } => format!("AI Terminal: {}", model),
        };

        let block = Block::default().borders(Borders::ALL).title(title).border_style(border_style);

        // Display pane content
        let content = if pane.buffer.lines.is_empty() {
            match &pane.pane_type {
                PaneType::Terminal { .. } => "Terminal ready. Type commands here.".to_string(),
                PaneType::MonitoringDashboard => {
                    "Monitoring dashboard will appear here.".to_string()
                }
                PaneType::CognitiveInterface => "Cognitive interface ready.".to_string(),
                _ => "Pane content will appear here.".to_string(),
            }
        } else {
            pane.buffer.lines.join("\n")
        };

        let paragraph = Paragraph::new(content).block(block).wrap(Wrap { trim: true });

        f.render_widget(paragraph, area);
        Ok(())
    }

    /// Draw status bar
    async fn draw_status_bar(
        &self,
        f: &mut Frame<'_>,
        area: Rect,
        session: &MultiplexerSession,
    ) -> Result<()> {
        let mut status_text = format!("[{}] {} panes", session.name, session.panes.len());

        // Add monitoring info if available
        if let Some(monitoring) = &self.monitoring {
            if let Ok(metrics) = monitoring.collect_metrics().await {
                status_text.push_str(&format!(" | CPU: {:.1}%", metrics.cpu_usage));
                status_text.push_str(&format!(
                    " | Mem: {:.1}GB",
                    metrics.memory_used as f64 / 1024.0 / 1024.0 / 1024.0
                ));
            }

            // Cost metrics are not available in the monitoring system yet
            // Would need to integrate with cost analytics module
        }

        let paragraph = Paragraph::new(status_text)
            .style(Style::default().fg(Color::White).bg(Color::Blue))
            .alignment(Alignment::Left);

        f.render_widget(paragraph, area);
        Ok(())
    }

    /// Draw session list
    async fn draw_session_list(
        &self,
        f: &mut Frame<'_>,
        sessions: &HashMap<SessionId, MultiplexerSession>,
    ) -> Result<()> {
        let items: Vec<ListItem> = sessions
            .values()
            .map(|session| {
                let status_icon = match session.status {
                    SessionStatus::Active => "●",
                    SessionStatus::Detached => "○",
                    SessionStatus::Suspended => "⏸",
                    SessionStatus::Terminated => "✗",
                };

                ListItem::new(format!(
                    "{} {} ({} panes)",
                    status_icon,
                    session.name,
                    session.panes.len()
                ))
            })
            .collect();

        let list = List::new(items)
            .block(Block::default().borders(Borders::ALL).title("Sessions"))
            .style(Style::default().fg(Color::White));

        f.render_widget(list, f.area());
        Ok(())
    }

    /// Draw no session screen
    async fn draw_no_session(&self, f: &mut Frame<'_>) -> Result<()> {
        let paragraph = Paragraph::new("No active session. Press Ctrl+C to create a new session.")
            .block(Block::default().borders(Borders::ALL).title("Loki Terminal Multiplexer"))
            .alignment(Alignment::Center);

        f.render_widget(paragraph, f.area());
        Ok(())
    }

    /// Cleanup terminal on exit
    fn cleanup(&mut self) -> Result<()> {
        disable_raw_mode()?;
        if let Some(ref mut terminal) = self.terminal {
            execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
            terminal.show_cursor()?;
        }
        Ok(())
    }

    /// Get command sender for external use
    pub fn command_sender(&self) -> mpsc::Sender<MultiplexerCommand> {
        self.command_tx.clone()
    }

    /// Subscribe to multiplexer events
    pub fn subscribe_events(&self) -> broadcast::Receiver<MultiplexerEvent> {
        self.event_tx.subscribe()
    }
}

impl Default for PaneBuffer {
    fn default() -> Self {
        Self {
            lines: Vec::new(),
            max_lines: 1000,
            scroll_position: 0,
            cursor_position: (0, 0),
            metadata: BufferMetadata {
                total_bytes: 0,
                line_count: 0,
                last_modified: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
                encoding: "UTF-8".to_string(),
            },
        }
    }
}

impl Default for PaneConfig {
    fn default() -> Self {
        Self {
            auto_scroll: true,
            word_wrap: true,
            show_line_numbers: false,
            color_scheme: "default".to_string(),
            font: FontConfig {
                family: "monospace".to_string(),
                size: 12,
                bold: false,
                italic: false,
            },
            key_bindings: HashMap::new(),
        }
    }
}

/// Synchronous multiplexer draw function to avoid borrow checker issues
async fn draw_multiplexer_sync(
    f: &mut Frame<'_>,
    sessions: &Arc<RwLock<HashMap<SessionId, MultiplexerSession>>>,
    current_session: &Arc<RwLock<Option<SessionId>>>,
    monitoring: &Option<Arc<AdvancedMonitoring>>,
) -> Result<()> {
    let sessions_guard = sessions.read().await;
    let current_session_id = *current_session.read().await;

    if let Some(session_id) = current_session_id {
        if let Some(session) = sessions_guard.get(&session_id) {
            draw_session_sync(f, session, monitoring).await?;
        } else {
            draw_no_session_sync(f).await?;
        }
    } else {
        draw_session_list_sync(f, &sessions_guard).await?;
    }

    Ok(())
}

/// Draw a session synchronously
async fn draw_session_sync(
    f: &mut Frame<'_>,
    session: &MultiplexerSession,
    monitoring: &Option<Arc<AdvancedMonitoring>>,
) -> Result<()> {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(0),    // Main content
            Constraint::Length(1), // Status bar
        ])
        .split(f.area());

    // Draw panes
    draw_panes_sync(f, chunks[0], session).await?;

    // Draw status bar
    draw_status_bar_sync(f, chunks[1], session, monitoring).await?;

    Ok(())
}

/// Draw panes synchronously
async fn draw_panes_sync(
    f: &mut Frame<'_>,
    area: Rect,
    session: &MultiplexerSession,
) -> Result<()> {
    if session.panes.is_empty() {
        let paragraph = Paragraph::new(
            "No panes in session. Press Ctrl+\" to split horizontally or Ctrl+% to split \
             vertically.",
        )
        .block(Block::default().borders(Borders::ALL).title("Empty Session"));
        f.render_widget(paragraph, area);
        return Ok(());
    }

    // Simple layout for now
    match session.layout.layout_type {
        LayoutType::Single => {
            if let Some(pane) = session.panes.values().next() {
                draw_pane_sync(f, area, pane, true).await?;
            }
        }
        LayoutType::Horizontal => {
            let constraints: Vec<Constraint> = session
                .panes
                .iter()
                .map(|_| Constraint::Percentage(100 / session.panes.len() as u16))
                .collect();

            let chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints(constraints)
                .split(area);

            for (i, pane) in session.panes.values().enumerate() {
                if i < chunks.len() {
                    let is_active = session.active_pane == Some(pane.id);
                    draw_pane_sync(f, chunks[i], pane, is_active).await?;
                }
            }
        }
        LayoutType::Vertical => {
            let constraints: Vec<Constraint> = session
                .panes
                .iter()
                .map(|_| Constraint::Percentage(100 / session.panes.len() as u16))
                .collect();

            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints(constraints)
                .split(area);

            for (i, pane) in session.panes.values().enumerate() {
                if i < chunks.len() {
                    let is_active = session.active_pane == Some(pane.id);
                    draw_pane_sync(f, chunks[i], pane, is_active).await?;
                }
            }
        }
        _ => {
            if let Some(pane) = session.panes.values().next() {
                draw_pane_sync(f, area, pane, true).await?;
            }
        }
    }

    Ok(())
}

/// Draw a single pane synchronously
async fn draw_pane_sync(
    f: &mut Frame<'_>,
    area: Rect,
    pane: &MultiplexerPane,
    is_active: bool,
) -> Result<()> {
    let border_style =
        if is_active { Style::default().fg(Color::Cyan) } else { Style::default().fg(Color::Gray) };

    let title = match &pane.pane_type {
        PaneType::Terminal { .. } => format!("Terminal: {}", pane.name),
        PaneType::MonitoringDashboard => "Monitoring Dashboard".to_string(),
        PaneType::CognitiveInterface => "Cognitive Interface".to_string(),
        PaneType::LogViewer { file_path, .. } => format!("Log: {}", file_path),
        PaneType::Editor { file_path, .. } => format!("Editor: {}", file_path),
        PaneType::Application { command, .. } => format!("App: {}", command),
        PaneType::AiTerminal { model, .. } => format!("AI Terminal: {}", model),
    };

    let block = Block::default().borders(Borders::ALL).title(title).border_style(border_style);

    let content = if pane.buffer.lines.is_empty() {
        match &pane.pane_type {
            PaneType::Terminal { .. } => "Terminal ready. Type commands here.".to_string(),
            PaneType::MonitoringDashboard => "Monitoring dashboard will appear here.".to_string(),
            PaneType::CognitiveInterface => "Cognitive interface ready.".to_string(),
            _ => "Pane content will appear here.".to_string(),
        }
    } else {
        pane.buffer.lines.join("\n")
    };

    let paragraph = Paragraph::new(content).block(block).wrap(Wrap { trim: true });

    f.render_widget(paragraph, area);
    Ok(())
}

/// Draw status bar synchronously
async fn draw_status_bar_sync(
    f: &mut Frame<'_>,
    area: Rect,
    session: &MultiplexerSession,
    monitoring: &Option<Arc<AdvancedMonitoring>>,
) -> Result<()> {
    let mut status_text = format!("[{}] {} panes", session.name, session.panes.len());

    if let Some(monitoring) = monitoring {
        if let Ok(metrics) = monitoring.collect_metrics().await {
            status_text.push_str(&format!(" | CPU: {:.1}%", metrics.cpu_usage));
            status_text.push_str(&format!(
                " | Mem: {:.1}GB",
                metrics.memory_used as f64 / 1024.0 / 1024.0 / 1024.0
            ));
        }

        // Cost metrics integration would be added here when available
    }

    let paragraph = Paragraph::new(status_text)
        .style(Style::default().fg(Color::White).bg(Color::Blue))
        .alignment(Alignment::Left);

    f.render_widget(paragraph, area);
    Ok(())
}

/// Draw session list synchronously
async fn draw_session_list_sync(
    f: &mut Frame<'_>,
    sessions: &HashMap<SessionId, MultiplexerSession>,
) -> Result<()> {
    let items: Vec<ListItem> = sessions
        .values()
        .map(|session| {
            let status_icon = match session.status {
                SessionStatus::Active => "●",
                SessionStatus::Detached => "○",
                SessionStatus::Suspended => "⏸",
                SessionStatus::Terminated => "✗",
            };

            ListItem::new(format!(
                "{} {} ({} panes)",
                status_icon,
                session.name,
                session.panes.len()
            ))
        })
        .collect();

    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title("Sessions"))
        .style(Style::default().fg(Color::White));

    f.render_widget(list, f.area());
    Ok(())
}

/// Draw no session screen synchronously
async fn draw_no_session_sync(f: &mut Frame<'_>) -> Result<()> {
    let paragraph = Paragraph::new("No active session. Press Ctrl+C to create a new session.")
        .block(Block::default().borders(Borders::ALL).title("Loki Terminal Multiplexer"))
        .alignment(Alignment::Center);

    f.render_widget(paragraph, f.area());
    Ok(())
}

// Non-async drawing functions for use within terminal.draw()

/// Draw session non-async
fn draw_session_non_async(
    f: &mut Frame<'_>,
    session: &MultiplexerSession,
    monitoring: &Option<Arc<AdvancedMonitoring>>,
) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(0),    // Main content
            Constraint::Length(1), // Status bar
        ])
        .split(f.area());

    // Draw panes
    draw_panes_non_async(f, chunks[0], session);

    // Draw status bar
    draw_status_bar_non_async(f, chunks[1], session, monitoring);
}

/// Draw panes non-async
fn draw_panes_non_async(
    f: &mut Frame<'_>,
    area: Rect,
    session: &MultiplexerSession,
) {
    if session.panes.is_empty() {
        let paragraph = Paragraph::new("No panes in this session")
            .block(Block::default().borders(Borders::ALL).title("Empty Session"))
            .alignment(Alignment::Center);
        f.render_widget(paragraph, area);
        return;
    }

    // For simplicity, just draw the first pane in the area
    // In a real implementation, you'd handle layouts properly
    if let Some((_pane_id, pane)) = session.panes.iter().next() {
        let title = match &pane.pane_type {
            PaneType::Terminal { shell, command } => {
                if let Some(cmd) = command {
                    format!("Pane {} - {} ({})", pane.id, cmd, shell)
                } else {
                    format!("Pane {} - {}", pane.id, shell)
                }
            }
            PaneType::MonitoringDashboard => format!("Pane {} - Monitoring", pane.id),
            PaneType::CognitiveInterface => format!("Pane {} - Cognitive", pane.id),
            PaneType::LogViewer { file_path, .. } => format!("Pane {} - Logs: {}", pane.id, file_path),
            PaneType::Editor { file_path, .. } => format!("Pane {} - Editor: {}", pane.id, file_path),
            PaneType::Application { command, .. } => format!("Pane {} - App: {}", pane.id, command),
            PaneType::AiTerminal { model, .. } => format!("Pane {} - AI: {}", pane.id, model),
        };

        let pane_block = Block::default()
            .borders(Borders::ALL)
            .title(title);

        let inner_area = pane_block.inner(area);
        f.render_widget(pane_block, area);

        // Draw pane content based on type
        let content_text = match &pane.pane_type {
            PaneType::Terminal { shell, command } => {
                format!("Shell: {}\nCommand: {}\nStatus: {:?}", 
                    shell, 
                    command.as_deref().unwrap_or("none"), 
                    pane.status)
            }
            PaneType::LogViewer { file_path, follow } => {
                format!("Viewing: {}\nFollow: {}\nStatus: {:?}", file_path, follow, pane.status)
            }
            PaneType::Editor { file_path, mode } => {
                format!("Editing: {}\nMode: {:?}\nStatus: {:?}", file_path, mode, pane.status)
            }
            PaneType::Application { command, args } => {
                format!("Running: {}\nArgs: {:?}\nStatus: {:?}", command, args, pane.status)
            }
            PaneType::AiTerminal { model, context } => {
                format!("Model: {}\nContext: {}\nStatus: {:?}", model, context, pane.status)
            }
            _ => format!("Status: {:?}", pane.status),
        };

        let content = Paragraph::new(content_text)
            .alignment(Alignment::Left);
        f.render_widget(content, inner_area);
    }
}

/// Draw status bar non-async
fn draw_status_bar_non_async(
    f: &mut Frame<'_>,
    area: Rect,
    session: &MultiplexerSession,
    _monitoring: &Option<Arc<AdvancedMonitoring>>,
) {
    let status_text = format!(
        " Session: {} | Status: {:?} | Panes: {} | Press ? for help ",
        session.name,
        session.status,
        session.panes.len()
    );

    let paragraph = Paragraph::new(status_text)
        .style(Style::default().fg(Color::White).bg(Color::Blue))
        .alignment(Alignment::Left);

    f.render_widget(paragraph, area);
}

/// Draw no session screen non-async
fn draw_no_session_non_async(f: &mut Frame<'_>) {
    let paragraph = Paragraph::new("No active session. Press Ctrl+C to create a new session.")
        .block(Block::default().borders(Borders::ALL).title("Loki Terminal Multiplexer"))
        .alignment(Alignment::Center);

    f.render_widget(paragraph, f.area());
}
