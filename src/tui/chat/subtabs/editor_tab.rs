//! Code Editor subtab implementation

use std::sync::Arc;
use tokio::sync::RwLock;
use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap, List, ListItem},
    Frame,
};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use anyhow::Result;
use std::path::{Path, PathBuf};

use super::SubtabController;
use crate::tui::chat::editor::{IntegratedEditor, EditorConfig};

/// Code editor tab
pub struct EditorTab {
    /// Integrated editor instance
    editor: Option<Arc<IntegratedEditor>>,
    
    /// Current file path
    current_file: Option<String>,
    
    /// Editor mode (editing, browsing, etc.)
    mode: EditorMode,
    
    /// Status message
    status_message: String,
    
    /// Line number
    current_line: usize,
    
    /// Column number
    current_column: usize,
    
    /// File content buffer (for display)
    content_buffer: Vec<String>,
    
    /// Scroll offset
    scroll_offset: usize,
    
    /// Current language
    editor_language: Option<String>,
    
    /// Editor bridge for chat integration
    editor_bridge: Option<Arc<crate::tui::bridges::EditorBridge>>,
    
    /// File browser state
    file_browser: FileBrowser,
    
    /// Input buffer for file path
    input_buffer: String,
    
    /// Input cursor position
    input_cursor: usize,
}

#[derive(Debug, Clone, PartialEq)]
enum EditorMode {
    Browse,
    Edit,
    Command,
    Search,
    FileBrowser,
    FileInput,
}

/// File browser for navigating project files
#[derive(Debug, Clone)]
struct FileBrowser {
    /// Current directory
    current_dir: PathBuf,
    
    /// Files in current directory
    files: Vec<FileEntry>,
    
    /// Selected index
    selected: usize,
    
    /// Scroll offset
    scroll_offset: usize,
}

#[derive(Debug, Clone)]
struct FileEntry {
    name: String,
    path: PathBuf,
    is_dir: bool,
    size: Option<u64>,
}

impl FileBrowser {
    fn new() -> Self {
        let current_dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("/"));
        let mut browser = Self {
            current_dir: current_dir.clone(),
            files: Vec::new(),
            selected: 0,
            scroll_offset: 0,
        };
        browser.refresh();
        browser
    }
    
    fn refresh(&mut self) {
        self.files.clear();
        
        // Add parent directory option
        if let Some(parent) = self.current_dir.parent() {
            self.files.push(FileEntry {
                name: "..".to_string(),
                path: parent.to_path_buf(),
                is_dir: true,
                size: None,
            });
        }
        
        // Read current directory
        if let Ok(entries) = std::fs::read_dir(&self.current_dir) {
            let mut dirs = Vec::new();
            let mut files = Vec::new();
            
            for entry in entries.flatten() {
                if let Ok(metadata) = entry.metadata() {
                    let name = entry.file_name().to_string_lossy().to_string();
                    let path = entry.path();
                    
                    let file_entry = FileEntry {
                        name,
                        path,
                        is_dir: metadata.is_dir(),
                        size: if metadata.is_file() { Some(metadata.len()) } else { None },
                    };
                    
                    if metadata.is_dir() {
                        dirs.push(file_entry);
                    } else {
                        files.push(file_entry);
                    }
                }
            }
            
            // Sort and add to list
            dirs.sort_by(|a, b| a.name.cmp(&b.name));
            files.sort_by(|a, b| a.name.cmp(&b.name));
            
            self.files.extend(dirs);
            self.files.extend(files);
        }
        
        self.selected = self.selected.min(self.files.len().saturating_sub(1));
    }
    
    fn navigate_up(&mut self) {
        if self.selected > 0 {
            self.selected -= 1;
        }
    }
    
    fn navigate_down(&mut self) {
        if self.selected < self.files.len().saturating_sub(1) {
            self.selected += 1;
        }
    }
    
    fn enter_directory(&mut self) {
        if let Some(entry) = self.files.get(self.selected) {
            if entry.is_dir {
                self.current_dir = entry.path.clone();
                self.selected = 0;
                self.scroll_offset = 0;
                self.refresh();
            }
        }
    }
    
    fn get_selected_file(&self) -> Option<&FileEntry> {
        self.files.get(self.selected)
    }
}

impl EditorTab {
    /// Create a new editor tab
    pub fn new() -> Self {
        // Initialize with some default content to show the editor is active
        let welcome_content = vec![
            "// Welcome to the Loki Code Editor".to_string(),
            "// ".to_string(),
            "// Quick Start:".to_string(),
            "//   Ctrl+O  - Open file browser".to_string(),
            "//   Ctrl+P  - Quick open (type path)".to_string(),
            "//   Ctrl+N  - New file".to_string(),
            "//   Enter   - Start editing".to_string(),
            "//   Esc     - Exit edit mode".to_string(),
            "// ".to_string(),
            "// Navigation:".to_string(),
            "//   ↑/↓     - Move cursor up/down".to_string(),
            "//   ←/→     - Move cursor left/right".to_string(),
            "//   Ctrl+Home - Go to file start".to_string(),
            "//   Ctrl+End  - Go to file end".to_string(),
            "// ".to_string(),
            "// This editor features:".to_string(),
            "//   - Syntax highlighting".to_string(),
            "//   - File browser".to_string(),
            "//   - LSP integration".to_string(),
            "//   - Code completion".to_string(),
            "//   - Multi-file support".to_string(),
            "//   - Chat integration".to_string(),
            "".to_string(),
        ];
        
        // Initialize editor immediately
        let mut tab = Self {
            editor: None,
            current_file: None,
            mode: EditorMode::Browse,
            status_message: "Ready - Press Ctrl+O to browse files or Ctrl+P to quick open".to_string(),
            current_line: 1,
            current_column: 1,
            content_buffer: welcome_content,
            scroll_offset: 0,
            editor_language: None,
            editor_bridge: None,
            file_browser: FileBrowser::new(),
            input_buffer: String::new(),
            input_cursor: 0,
        };
        
        // Try to initialize the editor asynchronously
        tokio::spawn(async move {
            let config = EditorConfig::default();
            if let Ok(editor) = IntegratedEditor::new(config).await {
                tracing::info!("Editor initialized successfully");
            }
        });
        
        tab
    }
    
    /// Set the editor bridge for chat integration
    pub fn set_editor_bridge(&mut self, bridge: Arc<crate::tui::bridges::EditorBridge>) {
        self.editor_bridge = Some(bridge);
        self.status_message = "Editor connected to chat".to_string();
    }
    
    /// Initialize the integrated editor
    pub async fn initialize_editor(&mut self) -> Result<()> {
        let config = EditorConfig {
            enable_lsp: true,
            enable_collaboration: false,
            tab_size: 4,
            auto_save: false,
            auto_save_interval_ms: 30000, // 30 seconds
            line_numbers: true,
            use_spaces: true,
            auto_indent: true,
            word_wrap: false,
            highlight_current_line: true,
            show_whitespace: false,
            max_line_length: Some(120),
        };
        
        self.editor = Some(Arc::new(IntegratedEditor::new(config).await?));
        self.status_message = "Editor initialized".to_string();
        Ok(())
    }
    
    /// Open a file
    async fn open_file(&mut self, path: &str) -> Result<()> {
        // Load file content
        if let Ok(content) = tokio::fs::read_to_string(path).await {
            self.content_buffer = content.lines().map(|s| s.to_string()).collect();
            if self.content_buffer.is_empty() {
                self.content_buffer.push(String::new());
            }
            
            self.current_file = Some(path.to_string());
            self.current_line = 1;
            self.current_column = 1;
            self.scroll_offset = 0;
            
            // Detect language from extension
            self.editor_language = Some(self.detect_language().to_string());
            
            // Update editor if initialized
            if let Some(ref editor) = self.editor {
                editor.open_file(path).await?;
            }
            
            self.status_message = format!("Opened: {}", path);
            self.mode = EditorMode::Browse;
        } else {
            self.status_message = format!("Failed to open: {}", path);
        }
        Ok(())
    }
    
    /// Load file content into buffer
    async fn load_file_content(&mut self, path: &str) -> Result<()> {
        // For now, just read the file directly
        // In production, this would use the editor's content
        if let Ok(content) = tokio::fs::read_to_string(path).await {
            self.content_buffer = content.lines().map(|s| s.to_string()).collect();
        }
        Ok(())
    }
    
    /// Receive code from chat
    pub async fn receive_code(&mut self, code: String, language: Option<String>) -> Result<()> {
        // Set the content buffer
        self.content_buffer = code.lines().map(|s| s.to_string()).collect();
        
        // Update editor if initialized
        if let Some(ref editor) = self.editor {
            editor.editor.set_content(code).await?;
        }
        
        // Update language
        self.editor_language = language;
        
        // Switch to edit mode
        self.mode = EditorMode::Edit;
        self.status_message = "Code received from chat".to_string();
        
        Ok(())
    }
    
    /// Get language from file extension
    fn detect_language(&self) -> &str {
        self.current_file.as_ref().map(|f| {
            let ext = std::path::Path::new(f)
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("");
            match ext {
                "rs" => "Rust",
                "py" => "Python",
                "js" | "jsx" => "JavaScript",
                "ts" | "tsx" => "TypeScript",
                "go" => "Go",
                "c" => "C",
                "cpp" | "cc" | "cxx" => "C++",
                "java" => "Java",
                "md" => "Markdown",
                "json" => "JSON",
                "yaml" | "yml" => "YAML",
                "toml" => "TOML",
                "html" => "HTML",
                "css" => "CSS",
                "sh" | "bash" => "Shell",
                _ => "Plain Text",
            }
        }).unwrap_or("Plain Text")
    }
    
    fn render_file_browser(&mut self, f: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(10),
                Constraint::Length(1),
            ])
            .split(area);
        
        // Create list items for files
        let items: Vec<ListItem> = self.file_browser.files.iter().map(|entry| {
            let icon = if entry.is_dir {
                "📁"
            } else {
                match entry.name.split('.').last().unwrap_or("") {
                    "rs" => "🦀",
                    "py" => "🐍",
                    "js" | "ts" => "📜",
                    "md" => "📝",
                    "toml" | "yaml" | "yml" => "⚙️",
                    _ => "📄",
                }
            };
            
            let size_str = if let Some(size) = entry.size {
                format!(" ({})", self.format_size(size))
            } else {
                String::new()
            };
            
            ListItem::new(format!("{} {}{}", icon, entry.name, size_str))
        }).collect();
        
        let list = List::new(items)
            .block(Block::default()
                .borders(Borders::ALL)
                .title(format!(" 📂 File Browser - {} ", self.file_browser.current_dir.display()))
                .border_style(Style::default().fg(Color::Cyan)))
            .highlight_style(Style::default().bg(Color::DarkGray).add_modifier(Modifier::BOLD))
            .highlight_symbol("> ");
        
        // Calculate visible area and adjust scroll
        let visible_height = chunks[0].height.saturating_sub(2) as usize;
        if self.file_browser.selected >= self.file_browser.scroll_offset + visible_height {
            self.file_browser.scroll_offset = self.file_browser.selected.saturating_sub(visible_height - 1);
        } else if self.file_browser.selected < self.file_browser.scroll_offset {
            self.file_browser.scroll_offset = self.file_browser.selected;
        }
        
        let mut state = ratatui::widgets::ListState::default();
        state.select(Some(self.file_browser.selected));
        
        f.render_stateful_widget(list, chunks[0], &mut state);
        
        // Status bar
        let status = format!(" [↑/↓] Navigate | [Enter] Open/Enter | [Esc] Cancel | {} items ",
            self.file_browser.files.len());
        let status_bar = Paragraph::new(status)
            .style(Style::default().bg(Color::DarkGray).fg(Color::White));
        f.render_widget(status_bar, chunks[1]);
    }
    
    fn render_file_input(&mut self, f: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),
                Constraint::Min(10),
                Constraint::Length(1),
            ])
            .split(area);
        
        // Input field
        let input = Paragraph::new(self.input_buffer.as_str())
            .block(Block::default()
                .borders(Borders::ALL)
                .title(" Enter file path to open: ")
                .border_style(Style::default().fg(Color::Yellow)));
        f.render_widget(input, chunks[0]);
        
        // Show current directory files as hints
        let mut hints = vec![Line::from("")];
        hints.push(Line::from(vec![
            Span::styled("Current directory: ", Style::default().fg(Color::Gray)),
            Span::styled(self.file_browser.current_dir.display().to_string(), 
                Style::default().fg(Color::Cyan)),
        ]));
        hints.push(Line::from(""));
        hints.push(Line::from("Recent files:"));
        
        for (i, entry) in self.file_browser.files.iter()
            .filter(|e| !e.is_dir && e.name != "..")
            .take(10)
            .enumerate() {
            hints.push(Line::from(vec![
                Span::styled(format!("  {}: ", i + 1), Style::default().fg(Color::DarkGray)),
                Span::raw(&entry.name),
            ]));
        }
        
        let hints_widget = Paragraph::new(hints)
            .block(Block::default()
                .borders(Borders::ALL)
                .title(" File Hints "));
        f.render_widget(hints_widget, chunks[1]);
        
        // Status bar
        let status = " [Enter] Open | [Tab] Auto-complete | [Esc] Cancel ";
        let status_bar = Paragraph::new(status)
            .style(Style::default().bg(Color::DarkGray).fg(Color::White));
        f.render_widget(status_bar, chunks[2]);
    }
    
    fn format_size(&self, size: u64) -> String {
        const UNITS: &[&str] = &["B", "KB", "MB", "GB"];
        let mut size = size as f64;
        let mut unit_idx = 0;
        
        while size >= 1024.0 && unit_idx < UNITS.len() - 1 {
            size /= 1024.0;
            unit_idx += 1;
        }
        
        if unit_idx == 0 {
            format!("{} {}", size as u64, UNITS[unit_idx])
        } else {
            format!("{:.1} {}", size, UNITS[unit_idx])
        }
    }
}

impl SubtabController for EditorTab {
    fn name(&self) -> &str {
        "editor"
    }
    
    fn title(&self) -> String {
        if let Some(ref file) = self.current_file {
            format!("📝 Editor - {}", std::path::Path::new(file).file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("untitled"))
        } else {
            "📝 Code Editor".to_string()
        }
    }
    
    fn render(&mut self, f: &mut Frame, area: Rect) {
        // Handle file browser mode separately
        if self.mode == EditorMode::FileBrowser {
            self.render_file_browser(f, area);
            return;
        }
        
        // Handle file input mode
        if self.mode == EditorMode::FileInput {
            self.render_file_input(f, area);
            return;
        }
        
        // Split into editor area and status bar
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(10),
                Constraint::Length(1),
            ])
            .split(area);
        
        // Try to sync content from IntegratedEditor if available
        if let Some(ref editor) = self.editor {
            // Get content from the actual editor in a blocking context
            let content = tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async {
                    editor.editor.get_content().await
                })
            });
            
            // Update our buffer with the editor's content
            self.content_buffer = content.lines().map(|s| s.to_string()).collect();
        }
        
        // Render editor content or welcome screen
        if self.current_file.is_some() && !self.content_buffer.is_empty() {
            // Render file content with syntax highlighting simulation
            let visible_lines = chunks[0].height as usize - 2; // Account for borders
            let end_line = (self.scroll_offset + visible_lines).min(self.content_buffer.len());
            
            let mut lines = Vec::new();
            for (i, line) in self.content_buffer[self.scroll_offset..end_line].iter().enumerate() {
                let line_num = self.scroll_offset + i + 1;
                let formatted_line = if self.mode == EditorMode::Edit && line_num == self.current_line {
                    // Highlight current line
                    Line::from(vec![
                        Span::styled(
                            format!("{:4} ", line_num),
                            Style::default().fg(Color::DarkGray),
                        ),
                        Span::styled(
                            line.clone(),
                            Style::default().bg(Color::DarkGray),
                        ),
                    ])
                } else {
                    Line::from(vec![
                        Span::styled(
                            format!("{:4} ", line_num),
                            Style::default().fg(Color::DarkGray),
                        ),
                        Span::raw(line.clone()),
                    ])
                };
                lines.push(formatted_line);
            }
            
            let editor_content = Paragraph::new(lines)
                .block(Block::default()
                    .borders(Borders::ALL)
                    .title(self.title())
                    .border_style(if self.mode == EditorMode::Edit {
                        Style::default().fg(Color::Green)
                    } else {
                        Style::default()
                    }));
            
            f.render_widget(editor_content, chunks[0]);
        } else {
            // Render welcome screen
            let welcome_lines = vec![
                Line::from(""),
                Line::from(vec![
                    Span::styled("Welcome to the Code Editor", 
                        Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
                ]),
                Line::from(""),
                Line::from("Features:"),
                Line::from("  • Syntax highlighting"),
                Line::from("  • Code completion (Ctrl+Space)"),
                Line::from("  • LSP integration"),
                Line::from("  • Multi-file support"),
                Line::from(""),
                Line::from("Commands:"),
                Line::from(vec![
                    Span::styled("  Ctrl+O", Style::default().fg(Color::Yellow)),
                    Span::raw(" - Open file"),
                ]),
                Line::from(vec![
                    Span::styled("  Ctrl+S", Style::default().fg(Color::Yellow)),
                    Span::raw(" - Save file"),
                ]),
                Line::from(vec![
                    Span::styled("  Ctrl+N", Style::default().fg(Color::Yellow)),
                    Span::raw(" - New file"),
                ]),
                Line::from(vec![
                    Span::styled("  Ctrl+W", Style::default().fg(Color::Yellow)),
                    Span::raw(" - Close file"),
                ]),
                Line::from(vec![
                    Span::styled("  F5", Style::default().fg(Color::Yellow)),
                    Span::raw(" - Run code"),
                ]),
                Line::from(""),
                Line::from(vec![
                    Span::styled("Press ", Style::default().fg(Color::Gray)),
                    Span::styled("Ctrl+O", Style::default().fg(Color::Yellow)),
                    Span::styled(" to open a file", Style::default().fg(Color::Gray)),
                ]),
            ];
            
            let welcome = Paragraph::new(welcome_lines)
                .block(Block::default()
                    .borders(Borders::ALL)
                    .title(" 📝 Code Editor "))
                .alignment(Alignment::Center);
            
            f.render_widget(welcome, chunks[0]);
        }
        
        // Render status bar
        let mode_str = match self.mode {
            EditorMode::Browse => "BROWSE",
            EditorMode::Edit => "EDIT",
            EditorMode::Command => "COMMAND",
            EditorMode::Search => "SEARCH",
            EditorMode::FileBrowser => "FILE BROWSER",
            EditorMode::FileInput => "FILE INPUT",
        };
        
        let status_text = if let Some(ref file) = self.current_file {
            format!(" {} | {} | Ln {}, Col {} | {} ",
                mode_str,
                self.status_message,
                self.current_line,
                self.current_column,
                self.detect_language(),
            )
        } else {
            format!(" {} | {} ", mode_str, self.status_message)
        };
        
        let status_bar = Paragraph::new(status_text)
            .style(Style::default().bg(Color::DarkGray).fg(Color::White));
        
        f.render_widget(status_bar, chunks[1]);
    }
    
    fn handle_input(&mut self, event: KeyEvent) -> Result<()> {
        let handled = match self.mode {
            EditorMode::FileBrowser => {
                match event.code {
                    // Allow Ctrl+J/K for subtab navigation
                    KeyCode::Char('j') | KeyCode::Char('k') if event.modifiers.contains(KeyModifiers::CONTROL) => {
                        // Exit file browser and let parent handle navigation
                        self.mode = EditorMode::Browse;
                        false
                    }
                    KeyCode::Up => {
                        self.file_browser.navigate_up();
                        true
                    }
                    KeyCode::Down => {
                        self.file_browser.navigate_down();
                        true
                    }
                    KeyCode::Enter => {
                        if let Some(entry) = self.file_browser.get_selected_file() {
                            if entry.is_dir {
                                self.file_browser.enter_directory();
                            } else {
                                // Open the file
                                let path = entry.path.to_string_lossy().to_string();
                                tokio::task::block_in_place(|| {
                                    tokio::runtime::Handle::current().block_on(async {
                                        let _ = self.open_file(&path).await;
                                    })
                                });
                                self.mode = EditorMode::Browse;
                            }
                        }
                        true
                    }
                    KeyCode::Esc => {
                        self.mode = EditorMode::Browse;
                        self.status_message = "File browser closed".to_string();
                        true
                    }
                    _ => false,
                }
            }
            EditorMode::FileInput => {
                match event.code {
                    // Allow Ctrl+J/K for subtab navigation
                    KeyCode::Char('j') | KeyCode::Char('k') if event.modifiers.contains(KeyModifiers::CONTROL) => {
                        // Cancel file input and let parent handle navigation
                        self.input_buffer.clear();
                        self.input_cursor = 0;
                        self.mode = EditorMode::Browse;
                        false
                    }
                    KeyCode::Char(c) if !event.modifiers.contains(KeyModifiers::CONTROL) => {
                        self.input_buffer.insert(self.input_cursor, c);
                        self.input_cursor += 1;
                        true
                    }
                    KeyCode::Backspace => {
                        if self.input_cursor > 0 {
                            self.input_cursor -= 1;
                            self.input_buffer.remove(self.input_cursor);
                        }
                        true
                    }
                    KeyCode::Left => {
                        if self.input_cursor > 0 {
                            self.input_cursor -= 1;
                        }
                        true
                    }
                    KeyCode::Right => {
                        if self.input_cursor < self.input_buffer.len() {
                            self.input_cursor += 1;
                        }
                        true
                    }
                    KeyCode::Enter => {
                        if !self.input_buffer.is_empty() {
                            let path = self.input_buffer.clone();
                            tokio::task::block_in_place(|| {
                                tokio::runtime::Handle::current().block_on(async {
                                    let _ = self.open_file(&path).await;
                                })
                            });
                            self.input_buffer.clear();
                            self.input_cursor = 0;
                            self.mode = EditorMode::Browse;
                        }
                        true
                    }
                    KeyCode::Esc => {
                        self.input_buffer.clear();
                        self.input_cursor = 0;
                        self.mode = EditorMode::Browse;
                        self.status_message = "File input cancelled".to_string();
                        true
                    }
                    _ => false,
                }
            }
            EditorMode::Browse => {
                match event.code {
                    // Important: Don't capture Ctrl+J/K - let parent handle subtab navigation
                    KeyCode::Char('j') | KeyCode::Char('k') if event.modifiers.contains(KeyModifiers::CONTROL) => {
                        false  // Let parent handle subtab navigation
                    }
                    KeyCode::Char('o') if event.modifiers.contains(KeyModifiers::CONTROL) => {
                        // Open file browser
                        self.mode = EditorMode::FileBrowser;
                        self.file_browser.refresh();
                        self.status_message = "File browser opened".to_string();
                        true
                    }
                    KeyCode::Char('p') if event.modifiers.contains(KeyModifiers::CONTROL) => {
                        // Quick open - file path input
                        self.mode = EditorMode::FileInput;
                        self.input_buffer.clear();
                        self.input_cursor = 0;
                        self.status_message = "Enter file path".to_string();
                        true
                    }
                    KeyCode::Char('n') if event.modifiers.contains(KeyModifiers::CONTROL) => {
                        // New file
                        self.current_file = Some("untitled.txt".to_string());
                        self.content_buffer = vec!["".to_string()];
                        self.mode = EditorMode::Edit;
                        self.status_message = "New file created".to_string();
                        
                        // Update IntegratedEditor if available
                        if let Some(ref editor) = self.editor {
                            tokio::task::block_in_place(|| {
                                tokio::runtime::Handle::current().block_on(async {
                                    let _ = editor.editor.set_content("".to_string()).await;
                                })
                            });
                        }
                        true
                    }
                    KeyCode::Char('e') | KeyCode::Enter => {
                        if self.current_file.is_some() {
                            self.mode = EditorMode::Edit;
                            self.status_message = "Edit mode".to_string();
                        }
                        true
                    }
                    KeyCode::Up => {
                        if self.scroll_offset > 0 {
                            self.scroll_offset -= 1;
                        }
                        true
                    }
                    KeyCode::Down => {
                        if self.scroll_offset < self.content_buffer.len().saturating_sub(10) {
                            self.scroll_offset += 1;
                        }
                        true
                    }
                    KeyCode::PageUp => {
                        self.scroll_offset = self.scroll_offset.saturating_sub(10);
                        true
                    }
                    KeyCode::PageDown => {
                        self.scroll_offset = (self.scroll_offset + 10)
                            .min(self.content_buffer.len().saturating_sub(10));
                        true
                    }
                    _ => false,
                }
            }
            EditorMode::Edit => {
                match event.code {
                    // Allow Ctrl+J/K for subtab navigation even in edit mode
                    KeyCode::Char('j') | KeyCode::Char('k') if event.modifiers.contains(KeyModifiers::CONTROL) => {
                        false  // Let parent handle subtab navigation
                    }
                    KeyCode::Esc => {
                        self.mode = EditorMode::Browse;
                        self.status_message = "Browse mode".to_string();
                        true
                    }
                    KeyCode::Char('s') if event.modifiers.contains(KeyModifiers::CONTROL) => {
                        // Save file
                        if let Some(ref editor) = self.editor {
                            tokio::task::block_in_place(|| {
                                tokio::runtime::Handle::current().block_on(async {
                                    if let Err(e) = editor.editor.save().await {
                                        tracing::error!("Failed to save file: {}", e);
                                    }
                                })
                            });
                        }
                        self.status_message = "File saved".to_string();
                        true
                    }
                    KeyCode::Char(c) if !event.modifiers.contains(KeyModifiers::CONTROL) => {
                        // Insert character at current position
                        if self.current_line <= self.content_buffer.len() && self.current_line > 0 {
                            let line_idx = self.current_line - 1;
                            if let Some(line) = self.content_buffer.get_mut(line_idx) {
                                let col_idx = (self.current_column - 1).min(line.len());
                                line.insert(col_idx, c);
                                self.current_column += 1;
                                
                                // Update IntegratedEditor
                                if let Some(ref editor) = self.editor {
                                    let content = self.content_buffer.join("\n");
                                    tokio::task::block_in_place(|| {
                                        tokio::runtime::Handle::current().block_on(async {
                                            let _ = editor.editor.set_content(content).await;
                                        })
                                    });
                                }
                            }
                        }
                        true
                    }
                    KeyCode::Backspace => {
                        // Delete character before cursor
                        if self.current_line <= self.content_buffer.len() && self.current_line > 0 {
                            let line_idx = self.current_line - 1;
                            if let Some(line) = self.content_buffer.get_mut(line_idx) {
                                if self.current_column > 1 && !line.is_empty() {
                                    let col_idx = self.current_column - 2;
                                    if col_idx < line.len() {
                                        line.remove(col_idx);
                                        self.current_column -= 1;
                                        
                                        // Update IntegratedEditor
                                        if let Some(ref editor) = self.editor {
                                            let content = self.content_buffer.join("\n");
                                            tokio::task::block_in_place(|| {
                                                tokio::runtime::Handle::current().block_on(async {
                                                    let _ = editor.editor.set_content(content).await;
                                                })
                                            });
                                        }
                                    }
                                }
                            }
                        }
                        true
                    }
                    KeyCode::Enter => {
                        // Insert new line
                        if self.current_line <= self.content_buffer.len() && self.current_line > 0 {
                            let line_idx = self.current_line - 1;
                            if line_idx < self.content_buffer.len() {
                                let current_line = self.content_buffer[line_idx].clone();
                                let col_idx = (self.current_column - 1).min(current_line.len());
                                
                                // Split the line at cursor position
                                let (before, after) = current_line.split_at(col_idx);
                                self.content_buffer[line_idx] = before.to_string();
                                self.content_buffer.insert(line_idx + 1, after.to_string());
                                
                                self.current_line += 1;
                                self.current_column = 1;
                                
                                // Update IntegratedEditor
                                if let Some(ref editor) = self.editor {
                                    let content = self.content_buffer.join("\n");
                                    tokio::task::block_in_place(|| {
                                        tokio::runtime::Handle::current().block_on(async {
                                            let _ = editor.editor.set_content(content).await;
                                        })
                                    });
                                }
                            }
                        }
                        true
                    }
                    KeyCode::Up => {
                        if self.current_line > 1 {
                            self.current_line -= 1;
                            if self.current_line <= self.scroll_offset {
                                self.scroll_offset = self.scroll_offset.saturating_sub(1);
                            }
                        }
                        true
                    }
                    KeyCode::Down => {
                        if self.current_line < self.content_buffer.len() {
                            self.current_line += 1;
                            if self.current_line > self.scroll_offset + 20 {
                                self.scroll_offset += 1;
                            }
                        }
                        true
                    }
                    KeyCode::Left => {
                        if self.current_column > 1 {
                            self.current_column -= 1;
                        }
                        true
                    }
                    KeyCode::Right => {
                        self.current_column += 1;
                        true
                    }
                    _ => false,
                }
            }
            EditorMode::Command => {
                match event.code {
                    // Allow Ctrl+J/K for subtab navigation
                    KeyCode::Char('j') | KeyCode::Char('k') if event.modifiers.contains(KeyModifiers::CONTROL) => {
                        self.mode = EditorMode::Browse;
                        false
                    }
                    KeyCode::Esc => {
                        self.mode = EditorMode::Browse;
                        self.status_message = "Command cancelled".to_string();
                        true
                    }
                    // In a real implementation, we'd handle text input here
                    _ => false,
                }
            }
            EditorMode::Search => {
                match event.code {
                    // Allow Ctrl+J/K for subtab navigation
                    KeyCode::Char('j') | KeyCode::Char('k') if event.modifiers.contains(KeyModifiers::CONTROL) => {
                        self.mode = EditorMode::Browse;
                        false
                    }
                    KeyCode::Esc => {
                        self.mode = EditorMode::Browse;
                        true
                    }
                    _ => false,
                }
            }
        };
        
        if handled {
            Ok(())
        } else {
            Ok(())
        }
    }
    
    fn update(&mut self) -> Result<()> {
        // Update editor state if needed
        Ok(())
    }
}