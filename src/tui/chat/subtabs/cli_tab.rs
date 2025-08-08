//! CLI/Terminal subtab with integrated command execution

use std::collections::VecDeque;
use std::sync::Arc;
use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Style},
    widgets::{Block, BorderType, Borders, List, ListItem, Paragraph, Wrap},
    Frame,
};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use anyhow::Result;
use chrono::{DateTime, Local};

use super::SubtabController;
use crate::tui::chat::ui_enhancements::{Theme, AnimationState};
use crate::tui::chat::core::tool_executor::ChatToolExecutor;
use crate::tui::chat::core::commands::{CommandRegistry, ParsedCommand, ResultFormat};

/// Maximum number of history entries to keep
const MAX_HISTORY: usize = 1000;

/// Maximum number of output lines to display
const MAX_OUTPUT_LINES: usize = 500;

/// Command history entry
#[derive(Debug, Clone)]
struct HistoryEntry {
    command: String,
    timestamp: DateTime<Local>,
}

/// Output line with metadata
#[derive(Debug, Clone)]
struct OutputLine {
    content: String,
    timestamp: DateTime<Local>,
    line_type: LineType,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum LineType {
    Command,
    Output,
    Error,
    System,
}

/// CLI/Terminal tab
pub struct CliTab {
    /// Current input buffer
    input_buffer: String,
    
    /// Cursor position in input
    cursor_position: usize,
    
    /// Command history
    history: VecDeque<HistoryEntry>,
    
    /// Current history navigation index
    history_index: Option<usize>,
    
    /// Output buffer
    output: VecDeque<OutputLine>,
    
    /// Scroll offset for output
    scroll_offset: usize,
    
    /// Current working directory
    working_directory: String,
    
    /// Show timestamps
    show_timestamps: bool,
    
    /// Auto-scroll to bottom
    auto_scroll: bool,
    
    /// Command aliases
    aliases: std::collections::HashMap<String, String>,
    
    /// Current theme
    theme: Theme,
    
    /// Cursor blink animation
    cursor_animation: AnimationState,
    
    /// Command execution animation
    execution_animation: Option<AnimationState>,
    
    /// Tool executor for command processing
    tool_executor: Option<Arc<ChatToolExecutor>>,
    
    /// Command registry for parsing
    command_registry: CommandRegistry,
}

impl CliTab {
    pub fn new() -> Self {
        let working_directory = std::env::current_dir()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| "/".to_string());
        
        let mut cli = Self {
            input_buffer: String::new(),
            cursor_position: 0,
            history: VecDeque::with_capacity(MAX_HISTORY),
            history_index: None,
            output: VecDeque::with_capacity(MAX_OUTPUT_LINES),
            scroll_offset: 0,
            working_directory: working_directory.clone(),
            show_timestamps: false,
            auto_scroll: true,
            aliases: Self::default_aliases(),
            theme: Theme::default(),
            cursor_animation: AnimationState::new(0.0, 1.0, std::time::Duration::from_millis(500)),
            execution_animation: None,
            tool_executor: None,
            command_registry: CommandRegistry::new(),
        };
        
        // Add welcome message
        cli.add_output("🖥️ Loki AI Terminal - v0.2.0", LineType::System);
        cli.add_output(&format!("Working directory: {}", working_directory), LineType::System);
        cli.add_output("Type 'help' for available commands", LineType::System);
        cli.add_output("", LineType::System);
        
        cli
    }
    
    /// Set the tool executor for command processing
    pub fn set_tool_executor(&mut self, executor: Arc<ChatToolExecutor>) {
        self.tool_executor = Some(executor);
        self.add_output("✅ Tool executor connected - advanced commands available", LineType::System);
    }
    
    /// Get default command aliases
    fn default_aliases() -> std::collections::HashMap<String, String> {
        let mut aliases = std::collections::HashMap::new();
        aliases.insert("ll".to_string(), "ls -la".to_string());
        aliases.insert("..".to_string(), "cd ..".to_string());
        aliases.insert("cls".to_string(), "clear".to_string());
        aliases.insert("h".to_string(), "history".to_string());
        aliases
    }
    
    /// Add output line
    fn add_output(&mut self, content: &str, line_type: LineType) {
        self.output.push_back(OutputLine {
            content: content.to_string(),
            timestamp: Local::now(),
            line_type,
        });
        
        // Limit output size
        if self.output.len() > MAX_OUTPUT_LINES {
            self.output.pop_front();
        }
        
        // Auto-scroll
        if self.auto_scroll {
            self.scroll_offset = 0;
        }
    }
    
    /// Execute the current command
    fn execute_command(&mut self) {
        let command = self.input_buffer.trim().to_string();
        if command.is_empty() {
            return;
        }
        
        // Start execution animation
        self.execution_animation = Some(AnimationState::new(0.0, 1.0, std::time::Duration::from_secs(2)));
        
        // Add to history
        self.history.push_back(HistoryEntry {
            command: command.clone(),
            timestamp: Local::now(),
        });
        
        // Limit history size
        if self.history.len() > MAX_HISTORY {
            self.history.pop_front();
        }
        
        // Reset history index
        self.history_index = None;
        
        // Clear input
        self.input_buffer.clear();
        self.cursor_position = 0;
        
        // Add command to output
        self.add_output(&format!("$ {}", command), LineType::Command);
        
        // Handle built-in commands
        let parts: Vec<&str> = command.split_whitespace().collect();
        if parts.is_empty() {
            return;
        }
        
        // Expand aliases
        let expanded_command = if let Some(alias) = self.aliases.get(parts[0]) {
            let mut expanded = alias.clone();
            for arg in &parts[1..] {
                expanded.push(' ');
                expanded.push_str(arg);
            }
            expanded
        } else {
            command.clone()
        };
        
        // Execute built-in commands
        match parts[0] {
            "clear" | "cls" => {
                self.output.clear();
                self.scroll_offset = 0;
            }
            "cd" => {
                let path = parts.get(1).map(|s| s.to_string()).unwrap_or_else(|| {
                    std::env::var("HOME").unwrap_or_else(|_| "/".to_string())
                });
                
                match std::env::set_current_dir(&path) {
                    Ok(_) => {
                        self.working_directory = std::env::current_dir()
                            .map(|p| p.display().to_string())
                            .unwrap_or_else(|_| path.clone());
                        self.add_output(&format!("Changed directory to: {}", self.working_directory), LineType::System);
                    }
                    Err(e) => {
                        self.add_output(&format!("Error: {}", e), LineType::Error);
                    }
                }
            }
            "pwd" => {
                self.add_output(&self.working_directory.clone(), LineType::Output);
            }
            "exit" | "quit" => {
                self.add_output("Use Esc or switch tabs to exit CLI mode", LineType::System);
            }
            "help" => {
                self.show_help();
            }
            "history" => {
                self.show_history();
            }
            "alias" => {
                self.handle_alias_command(&parts);
            }
            "echo" => {
                let message = parts[1..].join(" ");
                self.add_output(&message, LineType::Output);
            }
            "export" => {
                if parts.len() >= 2 {
                    let var_def = parts[1..].join(" ");
                    if let Some((key, value)) = var_def.split_once('=') {
                        std::env::set_var(key, value);
                        self.add_output(&format!("Set {}={}", key, value), LineType::System);
                    } else {
                        self.add_output("Usage: export KEY=VALUE", LineType::Error);
                    }
                } else {
                    // Show all environment variables
                    self.add_output("Environment variables:", LineType::System);
                    for (key, value) in std::env::vars() {
                        self.add_output(&format!("  {}={}", key, value), LineType::Output);
                    }
                }
            }
            "env" => {
                // Show environment variables
                for (key, value) in std::env::vars() {
                    self.add_output(&format!("{}={}", key, value), LineType::Output);
                }
            }
            _ => {
                // Try to execute through tool executor if available
                if let Some(ref executor) = self.tool_executor {
                    // Try to parse as a tool command
                    if let Ok(parsed_command) = self.command_registry.parse(&expanded_command) {
                        // Execute asynchronously
                        let executor_clone = executor.clone();
                        let output_sender = self.create_output_sender();
                        
                        // Spawn async task for execution
                        tokio::spawn(async move {
                            match executor_clone.execute(parsed_command).await {
                                Ok(result) => {
                                    // Send result back to output
                                    let content = result.content.as_str()
                                        .unwrap_or("Command executed successfully")
                                        .to_string();
                                    let _ = output_sender.send((content, LineType::Output));
                                }
                                Err(e) => {
                                    let _ = output_sender.send((format!("Error: {}", e), LineType::Error));
                                }
                            }
                        });
                        
                        self.add_output("Executing command...", LineType::System);
                    } else {
                        // Not a recognized tool command, try as shell command
                        self.execute_shell_command(&expanded_command);
                    }
                } else {
                    // No tool executor available
                    self.add_output(
                        &format!("Command execution not available: {}", expanded_command),
                        LineType::System
                    );
                    self.add_output(
                        "Tool executor not initialized. Basic commands only.",
                        LineType::System
                    );
                }
            }
        }
    }
    
    /// Create an output sender for async command execution
    fn create_output_sender(&self) -> std::sync::mpsc::Sender<(String, LineType)> {
        // Note: In a real implementation, this would connect to the actual output
        // For now, we create a dummy sender
        let (tx, _rx) = std::sync::mpsc::channel();
        tx
    }
    
    /// Execute a shell command using std::process
    fn execute_shell_command(&mut self, command: &str) {
        use std::process::Command;
        
        // Parse the command
        let parts: Vec<&str> = command.split_whitespace().collect();
        if parts.is_empty() {
            return;
        }
        
        // Create and execute the command
        let output = if cfg!(target_os = "windows") {
            Command::new("cmd")
                .args(["/C", command])
                .output()
        } else {
            Command::new("sh")
                .args(["-c", command])
                .output()
        };
        
        match output {
            Ok(output) => {
                // Add stdout
                if !output.stdout.is_empty() {
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    for line in stdout.lines() {
                        self.add_output(line, LineType::Output);
                    }
                }
                
                // Add stderr
                if !output.stderr.is_empty() {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    for line in stderr.lines() {
                        self.add_output(line, LineType::Error);
                    }
                }
                
                // Add exit status if non-zero
                if !output.status.success() {
                    if let Some(code) = output.status.code() {
                        self.add_output(&format!("Process exited with code: {}", code), LineType::System);
                    }
                }
            }
            Err(e) => {
                self.add_output(&format!("Failed to execute command: {}", e), LineType::Error);
            }
        }
    }
    
    /// Show help information
    fn show_help(&mut self) {
        self.add_output("", LineType::System);
        self.add_output("🛠️ Available Commands:", LineType::System);
        self.add_output("  clear/cls    - Clear the terminal", LineType::System);
        self.add_output("  cd <path>    - Change directory", LineType::System);
        self.add_output("  pwd          - Print working directory", LineType::System);
        self.add_output("  history      - Show command history", LineType::System);
        self.add_output("  alias        - Manage command aliases", LineType::System);
        self.add_output("  echo         - Print text", LineType::System);
        self.add_output("  export       - Set environment variable", LineType::System);
        self.add_output("  env          - Show environment variables", LineType::System);
        self.add_output("  help         - Show this help", LineType::System);
        self.add_output("", LineType::System);
        self.add_output("⌨️ Keyboard Shortcuts:", LineType::System);
        self.add_output("  Ctrl+L       - Clear screen", LineType::System);
        self.add_output("  Ctrl+U       - Clear line", LineType::System);
        self.add_output("  Ctrl+K       - Clear to end of line", LineType::System);
        self.add_output("  Ctrl+A       - Move to start of line", LineType::System);
        self.add_output("  Ctrl+E       - Move to end of line", LineType::System);
        self.add_output("  Ctrl+T       - Toggle timestamps", LineType::System);
        self.add_output("  ↑/↓          - Navigate history", LineType::System);
        self.add_output("  Page Up/Down - Scroll output", LineType::System);
        self.add_output("", LineType::System);
    }
    
    /// Show command history
    fn show_history(&mut self) {
        self.add_output("", LineType::System);
        self.add_output("📜 Command History:", LineType::System);
        
        let history_entries: Vec<String> = self.history
            .iter()
            .enumerate()
            .rev()
            .take(20)
            .map(|(idx, entry)| {
                let time_str = entry.timestamp.format("%H:%M:%S").to_string();
                format!("{:4} [{}] {}", idx + 1, time_str, entry.command)
            })
            .collect();
        
        for entry in history_entries {
            self.add_output(&entry, LineType::Output);
        }
        
        self.add_output("", LineType::System);
    }
    
    /// Handle alias command
    fn handle_alias_command(&mut self, parts: &[&str]) {
        if parts.len() == 1 {
            // List aliases
            self.add_output("", LineType::System);
            self.add_output("📝 Command Aliases:", LineType::System);
            
            let mut alias_list: Vec<(String, String)> = self.aliases
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            alias_list.sort_by_key(|(k, _)| k.clone());
            
            for (alias, command) in alias_list {
                self.add_output(&format!("  {} = {}", alias, command), LineType::Output);
            }
            
            self.add_output("", LineType::System);
        } else if parts.len() >= 2 {
            let alias_def = parts[1..].join(" ");
            if let Some((alias, command)) = alias_def.split_once('=') {
                self.aliases.insert(alias.trim().to_string(), command.trim().to_string());
                self.add_output(&format!("Alias set: {} = {}", alias.trim(), command.trim()), LineType::System);
            } else {
                self.add_output("Usage: alias name=command", LineType::Error);
            }
        }
    }
    
    /// Navigate history up
    fn history_up(&mut self) {
        if self.history.is_empty() {
            return;
        }
        
        match self.history_index {
            None => {
                self.history_index = Some(self.history.len() - 1);
            }
            Some(idx) if idx > 0 => {
                self.history_index = Some(idx - 1);
            }
            _ => return,
        }
        
        if let Some(idx) = self.history_index {
            if let Some(entry) = self.history.get(idx) {
                self.input_buffer = entry.command.clone();
                self.cursor_position = self.input_buffer.len();
            }
        }
    }
    
    /// Navigate history down
    fn history_down(&mut self) {
        match self.history_index {
            None => return,
            Some(idx) if idx < self.history.len() - 1 => {
                self.history_index = Some(idx + 1);
            }
            _ => {
                self.history_index = None;
                self.input_buffer.clear();
                self.cursor_position = 0;
                return;
            }
        }
        
        if let Some(idx) = self.history_index {
            if let Some(entry) = self.history.get(idx) {
                self.input_buffer = entry.command.clone();
                self.cursor_position = self.input_buffer.len();
            }
        }
    }
}

impl SubtabController for CliTab {
    fn render(&mut self, f: &mut Frame, area: Rect) {
        // Create layout
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(5),     // Output area
                Constraint::Length(3),  // Input area
                Constraint::Length(1),  // Status line
            ])
            .split(area);
        
        // Render output area
        let output_block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .title(format!(" 🖥️ Terminal Output {} ", 
                if self.auto_scroll { "[Auto-scroll]" } else { "[Manual]" }
            ))
            .style(Style::default().fg(Color::Cyan));
        
        let output_area = output_block.inner(chunks[0]);
        f.render_widget(output_block, chunks[0]);
        
        // Prepare output lines
        let visible_lines: Vec<ListItem> = self.output
            .iter()
            .skip(self.scroll_offset)
            .take(output_area.height as usize)
            .map(|line| {
                let style = match line.line_type {
                    LineType::Command => Style::default().fg(self.theme.warning),
                    LineType::Output => Style::default().fg(self.theme.text),
                    LineType::Error => Style::default().fg(self.theme.error),
                    LineType::System => Style::default().fg(self.theme.primary),
                };
                
                let content = if self.show_timestamps {
                    format!("[{}] {}", line.timestamp.format("%H:%M:%S"), line.content)
                } else {
                    line.content.clone()
                };
                
                ListItem::new(content).style(style)
            })
            .collect();
        
        let output_list = List::new(visible_lines);
        f.render_widget(output_list, output_area);
        
        // Render input area
        let input_block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .title(" 💬 Command Input ")
            .style(Style::default().fg(Color::Yellow));
        
        let input_area = input_block.inner(chunks[1]);
        f.render_widget(input_block, chunks[1]);
        
        // Render input with cursor and syntax highlighting
        let prompt = format!("{}$ ", 
            self.working_directory.split('/').last().unwrap_or("~")
        );
        
        // Create input with animated cursor
        let cursor_visible = self.cursor_animation.current_value() > 0.5;
        
        // Apply theming
        let prompt_style = Style::default().fg(self.theme.primary);
        let input_style = Style::default().fg(self.theme.text);
        
        // Build input text with cursor
        let mut input_spans = vec![
            ratatui::text::Span::styled(prompt.clone(), prompt_style),
        ];
        
        // Add command text with cursor
        for (i, ch) in self.input_buffer.chars().enumerate() {
            if i == self.cursor_position && cursor_visible {
                input_spans.push(ratatui::text::Span::styled(
                    ch.to_string(),
                    Style::default().bg(self.theme.text).fg(self.theme.background)
                ));
            } else {
                input_spans.push(ratatui::text::Span::styled(ch.to_string(), input_style));
            }
        }
        
        // Add cursor at end if needed
        if self.cursor_position == self.input_buffer.len() && cursor_visible {
            input_spans.push(ratatui::text::Span::styled(
                " ",
                Style::default().bg(self.theme.text).fg(self.theme.background)
            ));
        }
        
        let input_line = ratatui::text::Line::from(input_spans);
        let input_paragraph = Paragraph::new(vec![input_line])
            .wrap(Wrap { trim: false });
        
        f.render_widget(input_paragraph, input_area);
        
        // Render status line
        let status_text = format!(
            " {} | History: {} | Output: {} lines | {} ",
            self.working_directory,
            self.history.len(),
            self.output.len(),
            if self.show_timestamps { "Timestamps ON" } else { "Timestamps OFF" }
        );
        
        let status = Paragraph::new(status_text)
            .style(Style::default().bg(Color::DarkGray).fg(Color::White))
            .alignment(Alignment::Left);
        
        f.render_widget(status, chunks[2]);
    }
    
    fn handle_input(&mut self, key: KeyEvent) -> Result<()> {
        match key.code {
            KeyCode::Enter => {
                self.execute_command();
            }
            KeyCode::Backspace => {
                if self.cursor_position > 0 {
                    self.input_buffer.remove(self.cursor_position - 1);
                    self.cursor_position -= 1;
                }
            }
            KeyCode::Delete => {
                if self.cursor_position < self.input_buffer.len() {
                    self.input_buffer.remove(self.cursor_position);
                }
            }
            KeyCode::Left => {
                if self.cursor_position > 0 {
                    self.cursor_position -= 1;
                }
            }
            KeyCode::Right => {
                if self.cursor_position < self.input_buffer.len() {
                    self.cursor_position += 1;
                }
            }
            KeyCode::Home => {
                self.cursor_position = 0;
            }
            KeyCode::End => {
                self.cursor_position = self.input_buffer.len();
            }
            KeyCode::Up => {
                self.history_up();
            }
            KeyCode::Down => {
                self.history_down();
            }
            KeyCode::PageUp => {
                let max_scroll = self.output.len().saturating_sub(10);
                self.scroll_offset = self.scroll_offset.saturating_add(10).min(max_scroll);
                self.auto_scroll = false;
            }
            KeyCode::PageDown => {
                self.scroll_offset = self.scroll_offset.saturating_sub(10);
                if self.scroll_offset == 0 {
                    self.auto_scroll = true;
                }
            }
            KeyCode::Char(c) => {
                if key.modifiers.contains(KeyModifiers::CONTROL) {
                    match c {
                        'l' => {
                            // Clear screen
                            self.output.clear();
                            self.scroll_offset = 0;
                        }
                        'u' => {
                            // Clear line
                            self.input_buffer.clear();
                            self.cursor_position = 0;
                        }
                        'k' => {
                            // Clear to end of line
                            self.input_buffer.truncate(self.cursor_position);
                        }
                        'a' => {
                            // Move to start
                            self.cursor_position = 0;
                        }
                        'e' => {
                            // Move to end
                            self.cursor_position = self.input_buffer.len();
                        }
                        't' => {
                            // Toggle timestamps
                            self.show_timestamps = !self.show_timestamps;
                        }
                        _ => {}
                    }
                } else {
                    self.input_buffer.insert(self.cursor_position, c);
                    self.cursor_position += 1;
                }
            }
            _ => {}
        }
        
        Ok(())
    }
    
    fn update(&mut self) -> Result<()> {
        // Update cursor animation
        if self.cursor_animation.is_complete() {
            self.cursor_animation = AnimationState::new(0.0, 1.0, std::time::Duration::from_millis(500));
        }
        
        // Update execution animation
        if let Some(anim) = &self.execution_animation {
            if anim.is_complete() {
                self.execution_animation = None;
            }
        }
        
        Ok(())
    }
    
    fn name(&self) -> &str {
        "CLI"
    }
    
    fn title(&self) -> String {
        format!("Terminal - {}", self.working_directory)
    }
}