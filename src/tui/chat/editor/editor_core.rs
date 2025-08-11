//! Core Editor Implementation
//! 
//! Provides the fundamental text editing capabilities including buffer management,
//! cursor control, and edit operations.

use std::collections::VecDeque;
use std::sync::Arc;
use std::process::Command;
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};
use anyhow::Result;
use tracing::{info, debug};

/// Code editor core
pub struct CodeEditor {
    /// Current editor state
    state: Arc<RwLock<EditorState>>,
    
    /// Edit history for undo/redo
    history: Arc<RwLock<EditHistory>>,
    
    /// Editor configuration
    config: EditorConfig,
    
    /// File metadata
    file_info: Arc<RwLock<Option<FileInfo>>>,
}

/// Editor state
#[derive(Debug, Clone)]
pub struct EditorState {
    /// Text buffer (lines of text)
    pub buffer: Vec<String>,
    
    /// Cursor position
    pub cursor: CursorPosition,
    
    /// Selection range
    pub selection: Option<SelectionRange>,
    
    /// Viewport (visible area)
    pub viewport: Viewport,
    
    /// Modified flag
    pub modified: bool,
    
    /// Read-only mode
    pub readonly: bool,
}

/// Editor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EditorConfig {
    pub tab_size: usize,
    pub use_spaces: bool,
    pub auto_indent: bool,
    pub line_numbers: bool,
    pub word_wrap: bool,
    pub highlight_current_line: bool,
    pub show_whitespace: bool,
    pub max_line_length: Option<usize>,
    pub enable_lsp: bool,
    pub enable_collaboration: bool,
    pub auto_save: bool,
    pub auto_save_interval_ms: u64,
}

impl Default for EditorConfig {
    fn default() -> Self {
        Self {
            tab_size: 4,
            use_spaces: true,
            auto_indent: true,
            line_numbers: true,
            word_wrap: false,
            highlight_current_line: true,
            show_whitespace: false,
            max_line_length: Some(120),
            enable_lsp: true,
            enable_collaboration: false,
            auto_save: true,
            auto_save_interval_ms: 30000,
        }
    }
}

/// Cursor position
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct CursorPosition {
    pub line: usize,
    pub column: usize,
}

/// Selection range
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SelectionRange {
    pub start: CursorPosition,
    pub end: CursorPosition,
}

/// Viewport
#[derive(Debug, Clone, Copy)]
pub struct Viewport {
    pub top_line: usize,
    pub visible_lines: usize,
    pub left_column: usize,
    pub visible_columns: usize,
}

/// Result of code execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    /// Whether execution was successful
    pub success: bool,
    /// Standard output
    pub output: String,
    /// Standard error (if any)
    pub error: Option<String>,
    /// Execution time in milliseconds
    pub execution_time: u64,
}

/// Edit action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EditAction {
    Insert { position: CursorPosition, text: String },
    Delete { range: SelectionRange },
    Replace { range: SelectionRange, text: String },
    MoveCursor { position: CursorPosition },
    Select { range: SelectionRange },
    Indent { lines: Vec<usize> },
    Unindent { lines: Vec<usize> },
    Comment { lines: Vec<usize> },
    Uncomment { lines: Vec<usize> },
}

/// Edit history for undo/redo
struct EditHistory {
    undo_stack: VecDeque<EditAction>,
    redo_stack: VecDeque<EditAction>,
    max_history: usize,
}

/// File information
#[derive(Debug, Clone)]
struct FileInfo {
    path: String,
    language: Option<String>,
    encoding: String,
    line_ending: LineEnding,
    last_saved: Option<chrono::DateTime<chrono::Utc>>,
}

/// Line ending types
#[derive(Debug, Clone, Copy)]
enum LineEnding {
    LF,
    CRLF,
    CR,
}

impl CodeEditor {
    /// Get the current file path
    pub async fn get_file_path(&self) -> Option<String> {
        let file_info = self.file_info.read().await;
        file_info.as_ref().map(|info| info.path.clone())
    }
    
    /// Get the current language
    pub async fn get_language(&self) -> Option<String> {
        let file_info = self.file_info.read().await;
        file_info.as_ref().and_then(|info| info.language.clone())
    }
    
    /// Detect language from file extension
    fn detect_language(path: &str) -> Option<String> {
        let extension = std::path::Path::new(path)
            .extension()
            .and_then(|ext| ext.to_str())?;
        
        let language = match extension {
            "rs" => "rust",
            "py" => "python",
            "js" | "mjs" => "javascript",
            "ts" => "typescript",
            "jsx" => "javascriptreact",
            "tsx" => "typescriptreact",
            "go" => "go",
            "c" => "c",
            "cpp" | "cc" | "cxx" => "cpp",
            "h" | "hpp" => "cpp",
            "java" => "java",
            "cs" => "csharp",
            "rb" => "ruby",
            "php" => "php",
            "swift" => "swift",
            "kt" | "kts" => "kotlin",
            "scala" => "scala",
            "sh" | "bash" => "bash",
            "zsh" => "zsh",
            "fish" => "fish",
            "ps1" => "powershell",
            "yml" | "yaml" => "yaml",
            "json" => "json",
            "xml" => "xml",
            "html" | "htm" => "html",
            "css" => "css",
            "scss" | "sass" => "scss",
            "less" => "less",
            "sql" => "sql",
            "md" | "markdown" => "markdown",
            "tex" => "latex",
            "r" => "r",
            "lua" => "lua",
            "vim" => "vim",
            "toml" => "toml",
            "ini" | "cfg" => "ini",
            "dockerfile" => "dockerfile",
            "makefile" | "mk" => "makefile",
            _ => return None,
        };
        
        Some(language.to_string())
    }
    
    /// Create a new code editor
    pub fn new(config: EditorConfig) -> Self {
        Self {
            state: Arc::new(RwLock::new(EditorState {
                buffer: vec![String::new()],
                cursor: CursorPosition::default(),
                selection: None,
                viewport: Viewport {
                    top_line: 0,
                    visible_lines: 50,
                    left_column: 0,
                    visible_columns: 120,
                },
                modified: false,
                readonly: false,
            })),
            history: Arc::new(RwLock::new(EditHistory {
                undo_stack: VecDeque::new(),
                redo_stack: VecDeque::new(),
                max_history: 1000,
            })),
            config,
            file_info: Arc::new(RwLock::new(None)),
        }
    }
    
    /// Open a file
    pub async fn open_file(&self, path: &str) -> Result<()> {
        // Check if file exists, if not create it
        let content = match tokio::fs::read_to_string(path).await {
            Ok(content) => content,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                // File doesn't exist, create it
                info!("File {} doesn't exist, creating it", path);
                
                // Ensure parent directory exists
                if let Some(parent) = std::path::Path::new(path).parent() {
                    if !parent.as_os_str().is_empty() {
                        tokio::fs::create_dir_all(parent).await?;
                    }
                }
                
                // Create empty file
                tokio::fs::write(path, "").await?;
                String::new()
            }
            Err(e) => return Err(e.into()),
        };
        
        let lines: Vec<String> = content.lines().map(|s| s.to_string()).collect();
        
        let mut state = self.state.write().await;
        state.buffer = if lines.is_empty() {
            vec![String::new()]
        } else {
            lines
        };
        state.cursor = CursorPosition::default();
        state.selection = None;
        state.modified = false;
        
        // Detect language from extension
        let language = Self::detect_language(path);
        
        *self.file_info.write().await = Some(FileInfo {
            path: path.to_string(),
            language,
            encoding: "UTF-8".to_string(),
            line_ending: LineEnding::LF,
            last_saved: Some(chrono::Utc::now()),
        });
        
        info!("Opened file: {}", path);
        Ok(())
    }
    
    /// Save the current file
    pub async fn save(&self) -> Result<()> {
        let file_info = self.file_info.read().await;
        if let Some(info) = file_info.as_ref() {
            let path = info.path.clone(); // Clone the path before dropping file_info
            let content = self.get_content().await;
            
            // Ensure parent directory exists
            if let Some(parent) = std::path::Path::new(&path).parent() {
                if !parent.as_os_str().is_empty() {
                    tokio::fs::create_dir_all(parent).await?;
                }
            }
            
            tokio::fs::write(&path, content).await?;
            
            let mut state = self.state.write().await;
            state.modified = false;
            
            drop(file_info); // Now we can safely drop file_info
            if let Some(info) = self.file_info.write().await.as_mut() {
                info.last_saved = Some(chrono::Utc::now());
            }
            
            info!("Saved file: {}", path); // Use the cloned path
        } else {
            return Err(anyhow::anyhow!("Cannot save: No file path has been set for the editor"));
        }
        Ok(())
    }
    
    /// Save to a specific path
    pub async fn save_as(&self, path: &str) -> Result<()> {
        let content = self.get_content().await;
        
        // Ensure parent directory exists
        if let Some(parent) = std::path::Path::new(path).parent() {
            if !parent.as_os_str().is_empty() {
                tokio::fs::create_dir_all(parent).await?;
            }
        }
        
        // Write the file
        tokio::fs::write(path, &content).await?;
        info!("Saved file as: {}", path);
        
        // Update file info
        *self.file_info.write().await = Some(FileInfo {
            path: path.to_string(),
            language: Self::detect_language(path),
            encoding: "UTF-8".to_string(),
            line_ending: LineEnding::LF,
            last_saved: Some(chrono::Utc::now()),
        });
        
        let mut state = self.state.write().await;
        state.modified = false;
        
        Ok(())
    }
    
    /// Get current content as string
    pub async fn get_content(&self) -> String {
        let state = self.state.read().await;
        state.buffer.join("\n")
    }
    
    /// Set content
    pub async fn set_content(&self, content: String) -> Result<()> {
        let lines: Vec<String> = content.lines().map(|s| s.to_string()).collect();
        
        let mut state = self.state.write().await;
        state.buffer = if lines.is_empty() {
            vec![String::new()]
        } else {
            lines
        };
        state.modified = true;
        
        // Auto-save if we have a file path and auto-save is enabled
        let should_save = self.config.auto_save && {
            let file_info = self.file_info.read().await;
            file_info.is_some()
        };
        
        if should_save {
            drop(state); // Release the lock before saving
            self.save().await?;
        }
        
        Ok(())
    }
    
    /// Create and write to a new file
    pub async fn create_file(&self, path: &str, content: &str) -> Result<()> {
        // Ensure parent directory exists
        if let Some(parent) = std::path::Path::new(path).parent() {
            if !parent.as_os_str().is_empty() {
                tokio::fs::create_dir_all(parent).await?;
            }
        }
        
        // Write the file
        tokio::fs::write(path, content).await?;
        info!("Created file: {}", path);
        
        // Load it into the editor
        self.open_file(path).await?;
        
        Ok(())
    }
    
    /// Apply an edit action
    pub async fn apply_edit(&self, action: EditAction) -> Result<()> {
        // Record in history
        self.record_action(action.clone()).await;
        
        let mut state = self.state.write().await;
        
        match action {
            EditAction::Insert { position, text } => {
                self.insert_text(&mut state, position, &text);
            }
            EditAction::Delete { range } => {
                self.delete_range(&mut state, range);
            }
            EditAction::Replace { range, text } => {
                self.delete_range(&mut state, range);
                self.insert_text(&mut state, range.start, &text);
            }
            EditAction::MoveCursor { position } => {
                state.cursor = position;
                state.selection = None;
            }
            EditAction::Select { range } => {
                state.selection = Some(range);
                state.cursor = range.end;
            }
            EditAction::Indent { ref lines } => {
                for &line_idx in lines {
                    if line_idx < state.buffer.len() {
                        let indent = if self.config.use_spaces {
                            " ".repeat(self.config.tab_size)
                        } else {
                            "\t".to_string()
                        };
                        state.buffer[line_idx].insert_str(0, &indent);
                    }
                }
            }
            EditAction::Unindent { ref lines } => {
                for &line_idx in lines {
                    if line_idx < state.buffer.len() {
                        let line = &mut state.buffer[line_idx];
                        if line.starts_with('\t') {
                            line.remove(0);
                        } else {
                            let spaces_to_remove = line.chars()
                                .take(self.config.tab_size)
                                .take_while(|c| *c == ' ')
                                .count();
                            for _ in 0..spaces_to_remove {
                                line.remove(0);
                            }
                        }
                    }
                }
            }
            EditAction::Comment { ref lines } => {
                let comment_str = self.get_comment_string().await;
                for &line_idx in lines {
                    if line_idx < state.buffer.len() {
                        state.buffer[line_idx].insert_str(0, &format!("{} ", comment_str));
                    }
                }
            }
            EditAction::Uncomment { ref lines } => {
                let comment_str = self.get_comment_string().await;
                for &line_idx in lines {
                    if line_idx < state.buffer.len() {
                        let line = &mut state.buffer[line_idx];
                        if line.trim_start().starts_with(&comment_str) {
                            if let Some(pos) = line.find(&comment_str) {
                                line.drain(pos..pos + comment_str.len());
                                if line.chars().nth(0) == Some(' ') {
                                    line.remove(0);
                                }
                            }
                        }
                    }
                }
            }
        }
        
        state.modified = true;
        Ok(())
    }
    
    /// Insert text at position
    fn insert_text(&self, state: &mut EditorState, position: CursorPosition, text: &str) {
        if position.line >= state.buffer.len() {
            // Extend buffer if needed
            while state.buffer.len() <= position.line {
                state.buffer.push(String::new());
            }
        }
        
        let line = &mut state.buffer[position.line];
        let col = position.column.min(line.len());
        
        if text.contains('\n') {
            // Multi-line insert
            let lines: Vec<&str> = text.split('\n').collect();
            let rest_of_line = line.split_off(col);
            line.push_str(lines[0]);
            
            for i in 1..lines.len() - 1 {
                state.buffer.insert(position.line + i, lines[i].to_string());
            }
            
            if lines.len() > 1 {
                if let Some(last) = lines.last() {
                    let last_line = format!("{}{}", last, rest_of_line);
                    state.buffer.insert(position.line + lines.len() - 1, last_line);
                }
            }
            
            // Update cursor
            state.cursor = CursorPosition {
                line: position.line + lines.len() - 1,
                column: lines.last().map(|l| l.len()).unwrap_or(0),
            };
        } else {
            // Single line insert
            line.insert_str(col, text);
            state.cursor = CursorPosition {
                line: position.line,
                column: col + text.len(),
            };
        }
    }
    
    /// Delete a range of text
    fn delete_range(&self, state: &mut EditorState, range: SelectionRange) {
        let start = range.start;
        let end = range.end;
        
        if start.line == end.line {
            // Single line deletion
            let line = &mut state.buffer[start.line];
            let start_col = start.column.min(line.len());
            let end_col = end.column.min(line.len());
            line.drain(start_col..end_col);
        } else {
            // Multi-line deletion
            let start_line = &mut state.buffer[start.line];
            let start_col = start.column.min(start_line.len());
            start_line.truncate(start_col);
            
            if end.line < state.buffer.len() {
                let end_line = state.buffer[end.line].clone();
                let end_col = end.column.min(end_line.len());
                state.buffer[start.line].push_str(&end_line[end_col..]);
                
                // Remove intermediate lines
                for _ in 0..(end.line - start.line) {
                    if start.line + 1 < state.buffer.len() {
                        state.buffer.remove(start.line + 1);
                    }
                }
            }
        }
        
        state.cursor = start;
    }
    
    /// Record action in history
    async fn record_action(&self, action: EditAction) {
        let mut history = self.history.write().await;
        history.undo_stack.push_back(action);
        if history.undo_stack.len() > history.max_history {
            history.undo_stack.pop_front();
        }
        history.redo_stack.clear();
    }
    
    /// Undo last action
    pub async fn undo(&self) -> Result<()> {
        let mut history = self.history.write().await;
        if let Some(action) = history.undo_stack.pop_back() {
            // Create reverse action
            let reverse_action = self.create_reverse_action(&action).await;
            
            // Apply the reverse action
            drop(history);
            if let Some(reverse) = reverse_action {
                self.apply_edit_without_history(reverse).await?;
            }
            
            // Add to redo stack
            let mut history = self.history.write().await;
            history.redo_stack.push_back(action);
        }
        Ok(())
    }
    
    /// Create a reverse action for undo
    async fn create_reverse_action(&self, action: &EditAction) -> Option<EditAction> {
        let state = self.state.read().await;
        
        match action {
            EditAction::Insert { position, text } => {
                // Reverse of insert is delete
                let end_position = self.calculate_end_position(*position, text, &state);
                Some(EditAction::Delete {
                    range: SelectionRange {
                        start: *position,
                        end: end_position,
                    }
                })
            }
            EditAction::Delete { range } => {
                // Reverse of delete requires the deleted text
                // For now, we can't perfectly restore deleted text without storing it
                // This would require enhancing the EditAction to store deleted content
                None
            }
            EditAction::Replace { range, text: _ } => {
                // Similar issue - we need the original text to reverse a replace
                None
            }
            EditAction::MoveCursor { position: _ } => {
                // Cursor movement doesn't need reversal in undo
                None
            }
            EditAction::Select { range: _ } => {
                // Selection doesn't need reversal in undo
                None
            }
            EditAction::Indent { lines } => {
                // Reverse of indent is unindent
                Some(EditAction::Unindent { lines: lines.clone() })
            }
            EditAction::Unindent { lines } => {
                // Reverse of unindent is indent
                Some(EditAction::Indent { lines: lines.clone() })
            }
            EditAction::Comment { lines } => {
                // Reverse of comment is uncomment
                Some(EditAction::Uncomment { lines: lines.clone() })
            }
            EditAction::Uncomment { lines } => {
                // Reverse of uncomment is comment
                Some(EditAction::Comment { lines: lines.clone() })
            }
        }
    }
    
    /// Calculate end position after inserting text
    fn calculate_end_position(&self, start: CursorPosition, text: &str, _state: &EditorState) -> CursorPosition {
        let lines: Vec<&str> = text.split('\n').collect();
        if lines.len() == 1 {
            CursorPosition {
                line: start.line,
                column: start.column + lines[0].len(),
            }
        } else {
            CursorPosition {
                line: start.line + lines.len() - 1,
                column: lines.last().map(|l| l.len()).unwrap_or(0),
            }
        }
    }
    
    /// Insert text at position (async wrapper)
    async fn insert_at_position(&self, position: CursorPosition, text: &str) {
        let mut state = self.state.write().await;
        self.insert_text(&mut state, position, text);
        state.modified = true;
    }
    
    /// Delete range async wrapper (different name to avoid conflict)
    async fn delete_range_async(&self, range: SelectionRange) {
        let mut state = self.state.write().await;
        self.delete_range(&mut state, range);
        state.modified = true;
    }
    
    /// Apply edit without recording to history (for undo/redo)
    async fn apply_edit_without_history(&self, action: EditAction) -> Result<()> {
        match action {
            EditAction::Insert { position, text } => {
                self.insert_at_position(position, &text).await;
            }
            EditAction::Delete { range } => {
                self.delete_range_async(range).await;
            }
            EditAction::Replace { range, text } => {
                self.delete_range_async(range).await;
                self.insert_at_position(range.start, &text).await;
            }
            EditAction::MoveCursor { position } => {
                let mut state = self.state.write().await;
                state.cursor = position;
            }
            EditAction::Select { range } => {
                let mut state = self.state.write().await;
                state.selection = Some(range);
            }
            EditAction::Indent { ref lines } => {
                let mut state = self.state.write().await;
                for &line_idx in lines {
                    if line_idx < state.buffer.len() {
                        state.buffer[line_idx].insert_str(0, "    ");
                    }
                }
                state.modified = true;
            }
            EditAction::Unindent { ref lines } => {
                let mut state = self.state.write().await;
                for &line_idx in lines {
                    if line_idx < state.buffer.len() {
                        let line = &mut state.buffer[line_idx];
                        if line.starts_with("    ") {
                            line.drain(0..4);
                        } else if line.starts_with('\t') {
                            line.drain(0..1);
                        }
                    }
                }
                state.modified = true;
            }
            EditAction::Comment { ref lines } => {
                let comment_str = self.get_comment_string().await;
                let mut state = self.state.write().await;
                for &line_idx in lines {
                    if line_idx < state.buffer.len() {
                        state.buffer[line_idx].insert_str(0, &format!("{} ", comment_str));
                    }
                }
                state.modified = true;
            }
            EditAction::Uncomment { ref lines } => {
                let comment_str = self.get_comment_string().await;
                let mut state = self.state.write().await;
                for &line_idx in lines {
                    if line_idx < state.buffer.len() {
                        let line = &mut state.buffer[line_idx];
                        if line.starts_with(&format!("{} ", comment_str)) {
                            line.drain(0..comment_str.len() + 1);
                        } else if line.starts_with(&comment_str) {
                            line.drain(0..comment_str.len());
                        }
                    }
                }
                state.modified = true;
            }
        }
        
        Ok(())
    }
    
    /// Redo last undone action
    pub async fn redo(&self) -> Result<()> {
        let mut history = self.history.write().await;
        if let Some(action) = history.redo_stack.pop_back() {
            drop(history);
            self.apply_edit(action).await?;
        }
        Ok(())
    }
    
    /// Format the document
    pub async fn format(&self) -> Result<()> {
        // Basic formatting - can be extended
        let mut state = self.state.write().await;
        
        // Trim trailing whitespace
        for line in &mut state.buffer {
            *line = line.trim_end().to_string();
        }
        
        // Ensure final newline
        if state.buffer.last().map(|l| !l.is_empty()).unwrap_or(false) {
            state.buffer.push(String::new());
        }
        
        state.modified = true;
        Ok(())
    }
    
    /// Get comment string for current language
    async fn get_comment_string(&self) -> String {
        let file_info = self.file_info.read().await;
        if let Some(info) = file_info.as_ref() {
            if let Some(ref lang) = info.language {
                return match lang.as_str() {
                    "rust" => "//".to_string(),
                    "python" => "#".to_string(),
                    "javascript" | "typescript" => "//".to_string(),
                    "html" | "xml" => "<!--".to_string(),
                    "css" => "/*".to_string(),
                    _ => "//".to_string(),
                };
            }
        }
        "//".to_string()
    }
    
    /// Execute the current buffer content
    pub async fn execute(&self) -> Result<ExecutionResult> {
        let content = self.get_content().await;
        let file_info = self.file_info.read().await;
        
        let language = if let Some(info) = file_info.as_ref() {
            info.language.clone().unwrap_or_else(|| "text".to_string())
        } else {
            // Try to detect from content
            Self::detect_language_from_content(&content)
        };
        
        // Execute based on language
        match language.as_str() {
            "python" => self.execute_python(&content).await,
            "javascript" | "js" => self.execute_javascript(&content).await,
            "rust" => self.execute_rust(&content).await,
            "shell" | "bash" | "sh" => self.execute_shell(&content).await,
            _ => Ok(ExecutionResult {
                success: false,
                output: format!("Language '{}' is not supported for execution", language),
                error: None,
                execution_time: 0,
            }),
        }
    }
    
    /// Execute Python code
    async fn execute_python(&self, code: &str) -> Result<ExecutionResult> {
        let start = std::time::Instant::now();
        let output = Command::new("python3")
            .arg("-c")
            .arg(code)
            .output()?;
        
        Ok(ExecutionResult {
            success: output.status.success(),
            output: String::from_utf8_lossy(&output.stdout).to_string(),
            error: if output.stderr.is_empty() {
                None
            } else {
                Some(String::from_utf8_lossy(&output.stderr).to_string())
            },
            execution_time: start.elapsed().as_millis() as u64,
        })
    }
    
    /// Execute JavaScript code
    async fn execute_javascript(&self, code: &str) -> Result<ExecutionResult> {
        let start = std::time::Instant::now();
        let output = Command::new("node")
            .arg("-e")
            .arg(code)
            .output()?;
        
        Ok(ExecutionResult {
            success: output.status.success(),
            output: String::from_utf8_lossy(&output.stdout).to_string(),
            error: if output.stderr.is_empty() {
                None
            } else {
                Some(String::from_utf8_lossy(&output.stderr).to_string())
            },
            execution_time: start.elapsed().as_millis() as u64,
        })
    }
    
    /// Execute Rust code (requires compilation)
    async fn execute_rust(&self, code: &str) -> Result<ExecutionResult> {
        use uuid::Uuid;
        
        let temp_file = format!("/tmp/rust_exec_{}.rs", Uuid::new_v4());
        tokio::fs::write(&temp_file, code).await?;
        
        let start = std::time::Instant::now();
        
        // Compile
        let compile_output = Command::new("rustc")
            .arg(&temp_file)
            .arg("-o")
            .arg("/tmp/rust_exec")
            .output()?;
        
        if !compile_output.status.success() {
            let _ = tokio::fs::remove_file(&temp_file).await;
            return Ok(ExecutionResult {
                success: false,
                output: String::new(),
                error: Some(String::from_utf8_lossy(&compile_output.stderr).to_string()),
                execution_time: start.elapsed().as_millis() as u64,
            });
        }
        
        // Run
        let run_output = Command::new("/tmp/rust_exec").output()?;
        
        // Clean up
        let _ = tokio::fs::remove_file(&temp_file).await;
        let _ = tokio::fs::remove_file("/tmp/rust_exec").await;
        
        Ok(ExecutionResult {
            success: run_output.status.success(),
            output: String::from_utf8_lossy(&run_output.stdout).to_string(),
            error: if run_output.stderr.is_empty() {
                None
            } else {
                Some(String::from_utf8_lossy(&run_output.stderr).to_string())
            },
            execution_time: start.elapsed().as_millis() as u64,
        })
    }
    
    /// Execute shell script
    async fn execute_shell(&self, code: &str) -> Result<ExecutionResult> {
        let start = std::time::Instant::now();
        let output = Command::new("sh")
            .arg("-c")
            .arg(code)
            .output()?;
        
        Ok(ExecutionResult {
            success: output.status.success(),
            output: String::from_utf8_lossy(&output.stdout).to_string(),
            error: if output.stderr.is_empty() {
                None
            } else {
                Some(String::from_utf8_lossy(&output.stderr).to_string())
            },
            execution_time: start.elapsed().as_millis() as u64,
        })
    }
    
    /// Detect language from content
    fn detect_language_from_content(content: &str) -> String {
        // Check for shebang
        if let Some(first_line) = content.lines().next() {
            if first_line.starts_with("#!") {
                if first_line.contains("python") {
                    return "python".to_string();
                } else if first_line.contains("node") {
                    return "javascript".to_string();
                } else if first_line.contains("bash") || first_line.contains("sh") {
                    return "shell".to_string();
                }
            }
        }
        
        // Check for language-specific patterns
        if content.contains("def ") && content.contains("import ") {
            "python".to_string()
        } else if content.contains("fn main()") || content.contains("use std::") {
            "rust".to_string()
        } else if content.contains("function") || content.contains("const ") {
            "javascript".to_string()
        } else {
            "text".to_string()
        }
    }
    
    
    /// Get current cursor position
    pub async fn get_cursor(&self) -> CursorPosition {
        self.state.read().await.cursor
    }
    
    /// Get current selection
    pub async fn get_selection(&self) -> Option<SelectionRange> {
        self.state.read().await.selection
    }
    
    /// Check if modified
    pub async fn is_modified(&self) -> bool {
        self.state.read().await.modified
    }
}
