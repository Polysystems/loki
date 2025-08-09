//! Main chat conversation subtab

use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, BorderType, Borders, List, ListItem, Paragraph, Wrap},
    Frame,
};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use anyhow::Result;

use super::SubtabController;
use crate::tui::chat::state::ChatState;
use crate::tui::chat::handlers::InputProcessor;
use crate::tui::chat::orchestration::{OrchestrationManager, ModelCapability};
use crate::tui::chat::integrations::ToolIntegration;
use crate::tui::run::AssistantResponseType;
use crate::tui::chat::ui_enhancements::{TypingIndicator, SmoothScroll, ToastManager, ToastType, StreamingIndicator};

/// Main chat conversation tab
pub struct ChatTab {
    /// Chat state
    state: Arc<RwLock<ChatState>>,
    
    /// Input processor
    input_processor: InputProcessor,
    
    /// Orchestration manager
    orchestration: Option<Arc<RwLock<OrchestrationManager>>>,
    
    /// Tool integration
    tools: Option<Arc<ToolIntegration>>,
    
    /// Response receiver (optional - will be set after initialization)
    response_rx: Option<mpsc::Receiver<AssistantResponseType>>,
    
    /// Input mode
    input_mode: bool,
    
    /// Scroll offset for chat history
    scroll_offset: usize,
    
    /// Scroll offset for input area (when text wraps to multiple lines)
    input_scroll_offset: usize,
    
    /// Typing indicator for AI responses
    typing_indicator: TypingIndicator,
    
    /// Smooth scrolling
    smooth_scroll: SmoothScroll,
    
    /// Toast notifications
    toast_manager: ToastManager,
    
    /// Streaming indicator for real-time updates
    streaming_indicator: StreamingIndicator,
    
    /// Current selected model (from orchestration)
    selected_model: Option<String>,
    
    /// Active tool executions
    active_tools: Vec<String>,
    
    /// Cognitive insights buffer
    cognitive_insights: Vec<(String, String, chrono::DateTime<chrono::Utc>)>,
    
    /// Show insights panel
    show_insights_panel: bool,
    
    /// Whether the chat area is currently focused (vs input area)
    chat_focused: bool,
}

impl ChatTab {
    /// Update orchestration state
    fn update_orchestration_state(&mut self) {
        // This is called during render to ensure state is current
        // Empty for now but can be expanded
    }
    
    /// Check if the chat tab is in input mode
    pub fn is_input_mode(&self) -> bool {
        self.input_mode
    }
    
    /// Wrap text to fit within the given width
    fn wrap_text(text: &str, width: usize) -> Vec<String> {
        let mut lines = Vec::new();
        
        for line in text.lines() {
            if line.is_empty() {
                lines.push(String::new());
            } else if line.len() <= width {
                lines.push(line.to_string());
            } else {
                // Wrap long lines
                let mut current_line = String::new();
                let mut current_len = 0;
                
                for ch in line.chars() {
                    if current_len >= width {
                        lines.push(current_line.clone());
                        current_line.clear();
                        current_len = 0;
                    }
                    current_line.push(ch);
                    current_len += 1;
                }
                
                if !current_line.is_empty() {
                    lines.push(current_line);
                }
            }
        }
        
        if lines.is_empty() {
            lines.push(String::new());
        }
        
        lines
    }
    
    /// Create a new chat tab
    pub fn new(
        state: Arc<RwLock<ChatState>>,
        message_tx: mpsc::Sender<(String, usize)>,
    ) -> Self {
        Self {
            state,
            input_processor: InputProcessor::new(message_tx),
            orchestration: None,
            tools: None,
            response_rx: None,
            input_mode: true,
            scroll_offset: 0,
            input_scroll_offset: 0,
            typing_indicator: TypingIndicator::new(),
            smooth_scroll: SmoothScroll::new(),
            toast_manager: ToastManager::new(),
            streaming_indicator: StreamingIndicator::new(),
            selected_model: None,
            active_tools: Vec::new(),
            cognitive_insights: Vec::new(),
            show_insights_panel: false,
            chat_focused: false,
        }
    }
    
    /// Load a message into the input buffer
    pub fn load_message(&mut self, message: String) {
        self.input_processor.set_buffer(message);
        self.toast_manager.add_toast(
            "Message loaded from history".to_string(),
            ToastType::Info
        );
    }
    
    /// Set the response receiver
    pub fn set_response_receiver(&mut self, rx: mpsc::Receiver<AssistantResponseType>) {
        self.response_rx = Some(rx);
    }
    
    /// Set the tool integration
    pub fn set_tools(&mut self, tools: Arc<ToolIntegration>) {
        self.tools = Some(tools);
    }
    
    /// Set the orchestration manager
    pub fn set_orchestration(&mut self, orchestration: Arc<RwLock<OrchestrationManager>>) {
        self.orchestration = Some(orchestration.clone());
        
        // Get first enabled model or default model from persistence
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                let manager = orchestration.read().await;
                
                // Try to get default model first
                let persistence = crate::tui::chat::orchestration::model_persistence::ModelPersistence::new();
                if let Ok(Some(config)) = persistence.load().await {
                    if let Some(default_model) = config.default_model {
                        if manager.enabled_models.contains(&default_model) {
                            self.selected_model = Some(default_model);
                            return;
                        }
                    }
                }
                
                // Otherwise use first enabled model
                if !manager.enabled_models.is_empty() {
                    self.selected_model = Some(manager.enabled_models[0].clone());
                }
            })
        });
    }
    
    /// Render the messages area
    fn render_messages(&mut self, f: &mut Frame, area: Rect, messages: &[AssistantResponseType]) {
        // Collect all message lines as ListItems
        let mut list_items: Vec<ListItem> = Vec::new();
        
        for msg in messages {
            match msg {
                AssistantResponseType::Message { author, message, timestamp, .. } => {
                    // Create a multi-line ListItem for this message
                    let mut lines = Vec::new();
                    
                    // Add header line
                    lines.push(Line::from(vec![
                        Span::styled(
                            format!("[{}] ", timestamp),
                            Style::default().fg(Color::DarkGray),
                        ),
                        Span::styled(
                            format!("{}: ", author),
                            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
                        ),
                    ]));
                    
                    // Add message content with manual wrapping for long lines
                    let max_width = area.width.saturating_sub(4) as usize; // Account for borders and padding
                    for line in message.lines() {
                        if line.trim().is_empty() {
                            lines.push(Line::from(""));
                        } else if line.len() <= max_width {
                            // Line fits, add as-is
                            lines.push(Line::from(Span::styled(
                                line.to_string(),
                                Style::default().fg(Color::White),
                            )));
                        } else {
                            // Line is too long, wrap it manually
                            let wrapped = Self::wrap_text(line, max_width);
                            for wrapped_line in wrapped {
                                lines.push(Line::from(Span::styled(
                                    wrapped_line,
                                    Style::default().fg(Color::White),
                                )));
                            }
                        }
                    }
                    
                    // Add the complete message as a single ListItem
                    list_items.push(ListItem::new(lines));
                }
                AssistantResponseType::Stream { author, partial_content, stream_state, timestamp, .. } => {
                    let mut lines = Vec::new();
                    
                    // Determine streaming status and progress
                    let (status_indicator, color) = match stream_state {
                        crate::tui::run::StreamingState::Streaming { progress, .. } => {
                            // Animated streaming indicator
                            // Simple animation using timestamp
                            let animation_frame = (chrono::Utc::now().timestamp() % 4) as usize;
                            let dots = ".".repeat(animation_frame);
                            (format!(" 📡{:<3}", dots), Color::Cyan)
                        }
                        crate::tui::run::StreamingState::Complete => ("✓".to_string(), Color::Green),
                        crate::tui::run::StreamingState::Failed { .. } => ("❌".to_string(), Color::Red),
                        crate::tui::run::StreamingState::Queued => ("⏳".to_string(), Color::Yellow),
                        crate::tui::run::StreamingState::Processing { .. } => ("⚙️".to_string(), Color::Blue),
                    };
                    
                    // Add header with streaming status
                    lines.push(Line::from(vec![
                        Span::styled(
                            format!("[{}] ", timestamp),
                            Style::default().fg(Color::DarkGray),
                        ),
                        Span::styled(
                            format!("{}: ", author),
                            Style::default().fg(color).add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(
                            status_indicator,
                            Style::default().fg(color).add_modifier(Modifier::DIM),
                        ),
                    ]));
                    
                    // Add partial content with manual wrapping for long lines
                    let max_width = area.width.saturating_sub(4) as usize;
                    for line in partial_content.lines() {
                        if line.trim().is_empty() {
                            lines.push(Line::from(""));
                        } else if line.len() <= max_width {
                            lines.push(Line::from(Span::styled(
                                line.to_string(),
                                Style::default().fg(Color::White),
                            )));
                        } else {
                            // Wrap long lines manually
                            let wrapped = Self::wrap_text(line, max_width);
                            for wrapped_line in wrapped {
                                lines.push(Line::from(Span::styled(
                                    wrapped_line,
                                    Style::default().fg(Color::White),
                                )));
                            }
                        }
                    }
                    
                    // Add cursor/typing indicator if still streaming
                    if matches!(stream_state, crate::tui::run::StreamingState::Streaming { .. }) {
                        lines.push(Line::from(Span::styled(
                            "█", // Blinking cursor
                            Style::default().fg(color).add_modifier(Modifier::RAPID_BLINK),
                        )));
                    }
                    
                    list_items.push(ListItem::new(lines));
                }
                AssistantResponseType::Error { error_type, message, .. } => {
                    let mut lines = Vec::new();
                    lines.push(Line::from(Span::styled(
                        format!("Error [{}]", error_type),
                        Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
                    )));
                    lines.push(Line::from(Span::styled(
                        message.clone(),
                        Style::default().fg(Color::Red),
                    )));
                    list_items.push(ListItem::new(lines));
                }
                _ => {}
            }
        }
        
        // Add typing indicator if AI is processing
        let typing_text = self.typing_indicator.render();
        if !typing_text.is_empty() {
            list_items.push(ListItem::new(Line::from(Span::styled(
                typing_text,
                Style::default().fg(Color::Cyan).add_modifier(Modifier::DIM),
            ))));
        }
        
        // Calculate scrolling for the list
        let total_items = list_items.len();
        let visible_height = area.height.saturating_sub(2) as usize;
        
        // For now, let's just show ALL messages and let the List widget handle overflow
        // This is simpler and should work better than trying to manually paginate
        let visible_items = list_items;
        
        // We'll improve scrolling later if needed, but for now let's get messages displaying
        
        // Add scroll indicators to title
        let title = {
            let focus_indicator = if self.chat_focused { " [FOCUSED]" } else { "" };
            format!(" Chat History{} ({} messages) [Tab: toggle focus] ", focus_indicator, total_items)
        };
        
        // Create the List widget with only the visible items
        let messages_list = List::new(visible_items)
            .block(Block::default()
                .borders(Borders::ALL)
                .border_type(if self.chat_focused { BorderType::Thick } else { BorderType::Rounded })
                .border_style(if self.chat_focused {
                    Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(Color::DarkGray)
                })
                .title(title)
                .title_alignment(Alignment::Center));
        
        // Render the list without state
        f.render_widget(messages_list, area);
        
        // Add scroll bar indicator on the right
        if total_items > visible_height && area.width > 3 {
            let scrollbar_x = area.x + area.width - 1;
            let scrollbar_height = area.height.saturating_sub(2);
            let scrollbar_y = area.y + 1;
            
            // Calculate scrollbar position
            // When scroll_offset = 0, we're at bottom (scrollbar at bottom)
            // When scroll_offset = max, we're at top (scrollbar at top)
            let max_scroll = total_items.saturating_sub(visible_height);
            let scroll_ratio = if max_scroll > 0 {
                1.0 - (self.scroll_offset.min(max_scroll) as f32 / max_scroll as f32)
            } else {
                1.0
            };
            let scrollbar_pos = ((scrollbar_height.saturating_sub(1)) as f32 * scroll_ratio) as u16;
            
            // Draw scrollbar track
            for y in 0..scrollbar_height {
                let symbol = if y == scrollbar_pos {
                    "█" // Scrollbar handle
                } else {
                    "│" // Scrollbar track
                };
                let style = if y == scrollbar_pos {
                    Style::default().fg(Color::Cyan)
                } else {
                    Style::default().fg(Color::DarkGray)
                };
                
                f.render_widget(
                    Paragraph::new(symbol).style(style),
                    Rect {
                        x: scrollbar_x,
                        y: scrollbar_y + y,
                        width: 1,
                        height: 1,
                    }
                );
            }
        }
        
        // Render toast notifications in top-right corner
        self.toast_manager.update();
        let toast_area = Rect {
            x: area.x + area.width.saturating_sub(40),
            y: area.y + 1,
            width: 38.min(area.width.saturating_sub(4)),
            height: 5.min(area.height.saturating_sub(2)),
        };
        self.toast_manager.render(f, toast_area);
    }
    
    /// Render the status bar
    fn render_status_bar(&mut self, f: &mut Frame, area: Rect) {
        let orchestration_status = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                if let Some(orch) = &self.orchestration {
                    let manager = orch.read().await;
                    
                    // Always sync selected model with orchestration manager - prefer local models
                    if !manager.enabled_models.is_empty() {
                        // If we don't have a selected model, or it's not in the enabled list, pick the first local one
                        if self.selected_model.is_none() || 
                           (self.selected_model.as_ref().map_or(true, |s| !manager.enabled_models.contains(s))) {
                            // Prefer local/Ollama models over API models
                            let selected = manager.enabled_models.iter()
                                .find(|m| !m.contains("gpt") && !m.contains("claude") && !m.contains("gemini"))
                                .or_else(|| manager.enabled_models.first())
                                .unwrap();
                            self.selected_model = Some(selected.clone());
                            tracing::debug!("Updated selected model to: {}", selected);
                        }
                    }
                    
                    (
                        manager.orchestration_enabled,
                        manager.enabled_models.len(),
                        format!("{:?}", manager.preferred_strategy),
                    )
                } else {
                    (false, 0, "None".to_string())
                }
            })
        });
        
        let model_text = self.selected_model
            .as_ref()
            .filter(|m| !m.is_empty()) // Filter out empty strings
            .map(|m| {
                // Shorten long model names for display
                let display_name = if m.len() > 20 {
                    format!("🤖 {}...", &m[..17])
                } else {
                    format!("🤖 {}", m)
                };
                display_name
            })
            .unwrap_or_else(|| "⚠️ No Model Selected".to_string());
        
        let orchestration_text = if orchestration_status.0 {
            format!("🔄 Orchestration ON ({} models)", orchestration_status.1)
        } else {
            "🔴 Orchestration OFF".to_string()
        };
        
        let tools_text = if !self.active_tools.is_empty() {
            format!("🔧 Tools: {}", self.active_tools.join(", "))
        } else {
            "🔧 No active tools".to_string()
        };
        
        let status_line = Line::from(vec![
            Span::raw(" "),
            Span::styled(model_text, Style::default().fg(Color::Cyan)),
            Span::raw(" | "),
            Span::styled(orchestration_text, Style::default().fg(Color::Green)),
            Span::raw(" | "),
            Span::styled(tools_text, Style::default().fg(Color::Yellow)),
            Span::raw(" | "),
            Span::styled("T: tools", Style::default().fg(Color::DarkGray)),
        ]);
        
        let status_bar = Paragraph::new(status_line)
            .block(Block::default()
                .borders(Borders::BOTTOM)
                .border_style(Style::default().fg(Color::DarkGray)));
        
        f.render_widget(status_bar, area);
    }
    
    /// Render the input area
    fn render_input(&self, f: &mut Frame, area: Rect) {
        let input_style = if self.input_mode && !self.chat_focused {
            Style::default().fg(Color::Yellow)
        } else {
            Style::default().fg(Color::DarkGray)
        };
        
        // Get the input text and wrap it
        let text = self.input_processor.get_buffer();
        let text_width = area.width.saturating_sub(2) as usize; // Account for borders
        let wrapped_lines = Self::wrap_text(text, text_width);
        let total_lines = wrapped_lines.len();
        let visible_lines = area.height.saturating_sub(2) as usize; // Account for borders
        
        // Determine which lines to show based on scroll offset
        let display_text = if total_lines > visible_lines {
            // Text exceeds visible area, apply scroll offset
            let start_line = self.input_scroll_offset.min(total_lines.saturating_sub(visible_lines));
            let end_line = (start_line + visible_lines).min(total_lines);
            wrapped_lines[start_line..end_line].join("\n")
        } else {
            // All text fits, show everything
            wrapped_lines.join("\n")
        };
        
        // Add scroll indicator to title when needed
        let title = if self.input_mode && !self.chat_focused {
            if total_lines > visible_lines {
                let scroll_info = if self.input_scroll_offset == 0 {
                    "(top)"
                } else if self.input_scroll_offset >= total_lines.saturating_sub(visible_lines) {
                    "(bottom)"
                } else {
                    "(scrolling)"
                };
                format!(" Message {} [FOCUSED] {} ", scroll_info, "▌")
            } else {
                format!(" Message [FOCUSED] (Enter to send) {} ", "▌")
            }
        } else if self.chat_focused {
            " Message (Tab to focus input) ".to_string()
        } else {
            " Message (Tab to focus) ".to_string()
        };
        
        // Create the input widget with the visible text only
        let input = Paragraph::new(display_text)
            .style(input_style)
            .block(Block::default()
                .borders(Borders::ALL)
                .border_type(if self.input_mode && !self.chat_focused { BorderType::Thick } else { BorderType::Rounded })
                .title(title)
                .border_style(if self.input_mode && !self.chat_focused {
                    Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(Color::DarkGray)
                }));
        
        f.render_widget(input, area);
        
        // Calculate cursor position for wrapped text
        if self.input_mode && total_lines > 0 {
            // Calculate cursor position in the wrapped text
            let text_len = text.len();
            let mut cursor_line = 0;
            let mut cursor_col = 0;
            let mut char_count = 0;
            
            for (line_idx, line) in wrapped_lines.iter().enumerate() {
                if char_count + line.len() >= text_len {
                    cursor_line = line_idx;
                    cursor_col = text_len - char_count;
                    break;
                }
                char_count += line.len();
            }
            
            // Adjust for scroll offset
            if cursor_line >= self.input_scroll_offset && 
               cursor_line < self.input_scroll_offset + visible_lines {
                let visible_cursor_line = cursor_line - self.input_scroll_offset;
                let cursor_y = area.y + 1 + visible_cursor_line as u16;
                
                #[allow(deprecated)]
                f.set_cursor(
                    area.x + 1 + cursor_col as u16,
                    cursor_y,
                );
            }
        }
    }
    
    /// Render cognitive insights panel
    fn render_insights_panel(&self, f: &mut Frame, area: Rect) {
        // Create insights list
        let mut lines: Vec<Line> = vec![
            Line::from(vec![
                Span::styled("💭 Cognitive Insights", 
                    Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            ]),
            Line::from(""),
        ];
        
        if self.cognitive_insights.is_empty() {
            lines.push(Line::from(Span::styled(
                "No insights yet...",
                Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC),
            )));
        } else {
            // Display insights in reverse order (newest first)
            for (insight_type, content, timestamp) in self.cognitive_insights.iter().rev() {
                // Add insight type with icon
                let (icon, color) = match insight_type.as_str() {
                    "Pattern" => ("🔍", Color::Blue),
                    "Anomaly" => ("⚠️", Color::Yellow),
                    "Suggestion" => ("💡", Color::Green),
                    "Observation" => ("👁", Color::Cyan),
                    "Reflection" => ("🤔", Color::Magenta),
                    _ => ("💭", Color::White),
                };
                
                lines.push(Line::from(vec![
                    Span::raw(format!("{} ", icon)),
                    Span::styled(
                        insight_type.clone(),
                        Style::default().fg(color).add_modifier(Modifier::BOLD),
                    ),
                ]));
                
                // Add timestamp
                lines.push(Line::from(Span::styled(
                    format!("  {}", timestamp.format("%H:%M:%S")),
                    Style::default().fg(Color::DarkGray),
                )));
                
                // Add content (wrapped manually)
                let max_width = area.width.saturating_sub(4) as usize;
                let mut current_line = String::new();
                let mut current_len = 0;
                
                for word in content.split_whitespace() {
                    let word_len = word.len();
                    if current_len > 0 && current_len + word_len + 1 > max_width {
                        // Line would be too long, push current and start new
                        lines.push(Line::from(Span::styled(
                            format!("  {}", current_line),
                            Style::default().fg(Color::White),
                        )));
                        current_line = word.to_string();
                        current_len = word_len;
                    } else {
                        if current_len > 0 {
                            current_line.push(' ');
                            current_len += 1;
                        }
                        current_line.push_str(word);
                        current_len += word_len;
                    }
                }
                
                // Push remaining line
                if !current_line.is_empty() {
                    lines.push(Line::from(Span::styled(
                        format!("  {}", current_line),
                        Style::default().fg(Color::White),
                    )));
                }
                
                lines.push(Line::from(""));
            }
        }
        
        // Add toggle hint
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "Press I to toggle insights panel",
            Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC),
        )));
        
        let insights_widget = Paragraph::new(lines)
            .block(Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .title(" 🧠 Cognitive Insights ")
                .border_style(Style::default().fg(Color::Cyan)))
            .wrap(Wrap { trim: false });
        
        f.render_widget(insights_widget, area);
    }
    
    /// Format a message for display
    fn format_message(&self, msg: &AssistantResponseType) -> Vec<ListItem<'static>> {
        match msg {
            AssistantResponseType::Message { id: _, author, message, timestamp, is_editing: _, edit_history: _, streaming_state: _, metadata: _ } => {
                // Get the terminal width for proper wrapping
                let max_width = 100; // Conservative width to ensure messages fit
                
                // Create header with timestamp and author
                let mut items = vec![
                    ListItem::new(Line::from(vec![
                        Span::styled(
                            format!("[{}] ", timestamp),
                            Style::default().fg(Color::DarkGray),
                        ),
                        Span::styled(
                            format!("{}: ", author),
                            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
                        ),
                    ])),
                ];
                
                // Wrap long lines to fit within terminal width
                for line in message.lines() {
                    if line.trim().is_empty() {
                        items.push(ListItem::new(""));
                    } else {
                        // Wrap long lines
                        let mut current_line = String::new();
                        let mut current_len = 0;
                        
                        for word in line.split_whitespace() {
                            let word_len = word.len();
                            if current_len > 0 && current_len + word_len + 1 > max_width {
                                // Line would be too long, push current and start new
                                items.push(ListItem::new(current_line.clone())
                                    .style(Style::default().fg(Color::White)));
                                current_line = word.to_string();
                                current_len = word_len;
                            } else {
                                // Add word to current line
                                if !current_line.is_empty() {
                                    current_line.push(' ');
                                    current_len += 1;
                                }
                                current_line.push_str(word);
                                current_len += word_len;
                            }
                        }
                        
                        // Push remaining line
                        if !current_line.is_empty() {
                            items.push(ListItem::new(current_line)
                                .style(Style::default().fg(Color::White)));
                        }
                    }
                }
                
                // Add empty line for spacing
                items.push(ListItem::new(""));
                
                items
            }
            AssistantResponseType::Stream { id: _, author, partial_content, timestamp: _, stream_state, metadata: _ } => {
                let is_complete = matches!(stream_state, crate::tui::run::StreamingState::Complete);
                vec![
                    ListItem::new(Line::from(vec![
                        Span::styled(
                            format!("{}: ", author),
                            Style::default().fg(Color::Green),
                        ),
                        Span::raw(if is_complete { "" } else { "..." }),
                    ])),
                    ListItem::new(partial_content.clone())
                        .style(Style::default().fg(Color::White)),
                ]
            }
            AssistantResponseType::Error { id: _, error_type, message, timestamp: _, metadata: _ } => {
                vec![
                    ListItem::new(Line::from(vec![
                        Span::styled(
                            format!("Error [{}]", error_type),
                            Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
                        ),
                    ])),
                    ListItem::new(message.clone())
                        .style(Style::default().fg(Color::Red)),
                    ListItem::new(""),
                ]
            }
            AssistantResponseType::ToolExecution { tool_name, input, output, .. } => {
                vec![
                    ListItem::new(Line::from(vec![
                        Span::styled("🛠️ Tool: ", Style::default().fg(Color::Magenta)),
                        Span::styled(tool_name.clone(), Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
                    ])),
                    ListItem::new(format!("   Input: {}", input))
                        .style(Style::default().fg(Color::DarkGray)),
                    ListItem::new(format!("   Output: {}", output))
                        .style(Style::default().fg(Color::White)),
                    ListItem::new(""),
                ]
            }
            AssistantResponseType::Action { command, .. } => {
                vec![
                    ListItem::new(Line::from(vec![
                        Span::styled("⚡ Action: ", Style::default().fg(Color::Blue)),
                        Span::styled(command.clone(), Style::default().fg(Color::Cyan)),
                    ])),
                    ListItem::new(""),
                ]
            }
            _ => vec![],
        }
    }
}

impl SubtabController for ChatTab {
    fn render(&mut self, f: &mut Frame, area: Rect) {
        // Ensure we have current orchestration state
        self.update_orchestration_state();
        
        // Get current state
        let state = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                self.state.read().await.clone()
            })
        });
        
        // Check if we need a streaming indicator area
        let has_streaming = matches!(
            self.streaming_indicator.state, 
            crate::tui::chat::ui_enhancements::StreamingState::Streaming { .. } |
            crate::tui::chat::ui_enhancements::StreamingState::Connecting |
            crate::tui::chat::ui_enhancements::StreamingState::Completing
        );
        
        // Calculate input area height based on text content
        let text = self.input_processor.get_buffer();
        let text_width = area.width.saturating_sub(2) as usize;
        let lines_needed = if text.is_empty() {
            1
        } else {
            let mut line_count = 1;
            let mut current_line_len = 0;
            for ch in text.chars() {
                if ch == '\n' {
                    line_count += 1;
                    current_line_len = 0;
                } else {
                    current_line_len += 1;
                    if current_line_len >= text_width.max(1) {
                        line_count += 1;
                        current_line_len = 0;
                    }
                }
            }
            line_count
        };
        
        // Dynamic input height: min 3, max 10 lines
        let input_height = (lines_needed + 2).min(10).max(3) as u16;
        
        // Split area horizontally if insights panel is shown
        let (main_area, insights_area) = if self.show_insights_panel {
            let horizontal_chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([
                    Constraint::Percentage(70),  // Main chat area
                    Constraint::Percentage(30),  // Insights panel
                ])
                .split(area);
            (horizontal_chunks[0], Some(horizontal_chunks[1]))
        } else {
            (area, None)
        };
        
        // Split area based on whether streaming is active
        let chunks = if has_streaming {
            Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(2),           // Status bar
                    Constraint::Length(3),           // Streaming indicator
                    Constraint::Min(5),              // Messages area
                    Constraint::Length(input_height), // Dynamic input area
                ])
                .split(main_area)
        } else {
            Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(2),           // Status bar
                    Constraint::Min(5),              // Messages area
                    Constraint::Length(input_height), // Dynamic input area
                ])
                .split(main_area)
        };
        
        // Render cognitive insights panel if visible
        if let Some(insights_area) = insights_area {
            self.render_insights_panel(f, insights_area);
        }
        
        // Render status bar
        self.render_status_bar(f, chunks[0]);
        
        // Render streaming indicator if active
        if has_streaming {
            self.streaming_indicator.render(f, chunks[1]);
            self.render_messages(f, chunks[2], &state.messages);
            self.render_input(f, chunks[3]);
        } else {
            self.render_messages(f, chunks[1], &state.messages);
            self.render_input(f, chunks[2]);
        }
    }
    
    fn handle_input(&mut self, key: KeyEvent) -> Result<()> {
        // Handle scrolling keys first - these work regardless of input mode
        // PageUp/PageDown always scroll the chat history
        match key.code {
            KeyCode::PageUp => {
                // Page up to see older messages (scroll up in history)
                // Scroll by a reasonable number of messages, not lines
                self.scroll_offset += 3;  // Scroll 3 messages at a time
                self.smooth_scroll.scroll_to(self.scroll_offset);
                self.toast_manager.add_toast(
                    "Scrolling up in history".to_string(),
                    ToastType::Info
                );
                return Ok(());
            }
            KeyCode::PageDown => {
                // Page down to see newer messages (scroll down towards recent)
                self.scroll_offset = self.scroll_offset.saturating_sub(3);  // Scroll 3 messages at a time
                if self.scroll_offset == 0 {
                    self.toast_manager.add_toast(
                        "At latest messages".to_string(),
                        ToastType::Info
                    );
                }
                self.smooth_scroll.scroll_to(self.scroll_offset);
                return Ok(());
            }
            // Arrow keys for scrolling when chat is focused
            KeyCode::Up if self.chat_focused && !key.modifiers.contains(KeyModifiers::CONTROL) => {
                // Scroll up one line when chat is focused
                self.scroll_offset += 1;
                self.smooth_scroll.scroll_to(self.scroll_offset);
                return Ok(());
            }
            KeyCode::Down if self.chat_focused && !key.modifiers.contains(KeyModifiers::CONTROL) => {
                // Scroll down one line when chat is focused
                if self.scroll_offset > 0 {
                    self.scroll_offset -= 1;
                    self.smooth_scroll.scroll_to(self.scroll_offset);
                }
                return Ok(());
            }
            _ => {}
        }
        
        // Handle Ctrl+Shift+arrow keys for input area scrolling (only when in input mode)
        if self.input_mode && key.modifiers.contains(KeyModifiers::CONTROL | KeyModifiers::SHIFT) {
            match key.code {
                KeyCode::Up => {
                    // Scroll input area up
                    if self.input_scroll_offset > 0 {
                        self.input_scroll_offset -= 1;
                    }
                    return Ok(());
                }
                KeyCode::Down => {
                    // Scroll input area down
                    // Calculate max scroll based on wrapped lines
                    let text = self.input_processor.get_buffer();
                    let text_width = 80; // Approximate, will be recalculated in render
                    let wrapped_lines = Self::wrap_text(text, text_width);
                    let max_scroll = wrapped_lines.len().saturating_sub(3); // Assuming ~3 visible lines
                    
                    if self.input_scroll_offset < max_scroll {
                        self.input_scroll_offset += 1;
                    }
                    return Ok(());
                }
                _ => {}
            }
        }
        
        // Handle Ctrl+arrow keys for chat history scrolling (works in both modes)
        if key.modifiers.contains(KeyModifiers::CONTROL) && !key.modifiers.contains(KeyModifiers::SHIFT) {
            match key.code {
                KeyCode::Up => {
                    // Ctrl+Up scrolls chat history up one line (older messages)
                    self.scroll_offset += 1;
                    self.smooth_scroll.scroll_to(self.scroll_offset);
                    return Ok(());
                }
                KeyCode::Down => {
                    // Ctrl+Down scrolls chat history down one line (newer messages)
                    if self.scroll_offset > 0 {
                        self.scroll_offset -= 1;
                        self.smooth_scroll.scroll_to(self.scroll_offset);
                    }
                    return Ok(());
                }
                KeyCode::Home => {
                    // Ctrl+Home jumps to oldest messages (top of history)
                    let state = tokio::task::block_in_place(|| {
                        tokio::runtime::Handle::current().block_on(async {
                            self.state.read().await.messages.len()
                        })
                    });
                    // Set a large offset to show the beginning
                    self.scroll_offset = state * 2; // Large enough to reach the top
                    self.smooth_scroll.scroll_to(self.scroll_offset);
                    self.toast_manager.add_toast(
                        "Jumped to oldest messages".to_string(),
                        ToastType::Info
                    );
                    return Ok(());
                }
                KeyCode::End => {
                    // Ctrl+End jumps to newest messages (bottom)
                    self.scroll_offset = 0;
                    self.smooth_scroll.scroll_to(0);
                    self.toast_manager.add_toast(
                        "Jumped to newest messages".to_string(),
                        ToastType::Info
                    );
                    return Ok(());
                }
                _ => {}
            }
        }
        
        // Handle mode switching and other keys
        match key.code {
            KeyCode::Tab => {
                // Toggle between chat focus and input focus
                if self.input_mode {
                    // Already in input mode, toggle focus between chat and input
                    self.chat_focused = !self.chat_focused;
                    if self.chat_focused {
                        self.toast_manager.add_toast(
                            "Focused on chat history (use arrows to scroll)".to_string(),
                            ToastType::Info
                        );
                    } else {
                        self.toast_manager.add_toast(
                            "Focused on input area".to_string(),
                            ToastType::Info
                        );
                        // Reset input scroll when focusing on input
                        self.input_scroll_offset = 0;
                    }
                } else {
                    // Not in input mode, enter it
                    self.input_mode = true;
                    self.chat_focused = false;
                    // Don't auto-scroll - preserve user's current scroll position
                    // Only reset input scroll
                    self.input_scroll_offset = 0;
                }
            }
            KeyCode::Esc => {
                // Exit input mode with Escape key or switch focus to chat
                if self.input_mode && !self.chat_focused {
                    // If input is focused, switch to chat focus
                    self.chat_focused = true;
                    self.toast_manager.add_toast(
                        "Switched to chat history".to_string(),
                        ToastType::Info
                    );
                } else if self.chat_focused {
                    // If chat is focused, exit input mode entirely
                    self.input_mode = false;
                    self.toast_manager.add_toast(
                        "Exited input mode".to_string(),
                        ToastType::Info
                    );
                }
                // Don't reset scroll - user might want to stay at current position
            }
            KeyCode::Char('i') | KeyCode::Char('I') if !self.input_mode => {
                // Toggle insights panel with 'i' key when not in input mode
                self.show_insights_panel = !self.show_insights_panel;
                if self.show_insights_panel {
                    self.toast_manager.add_toast(
                        "Cognitive insights panel opened".to_string(),
                        ToastType::Info
                    );
                } else {
                    self.toast_manager.add_toast(
                        "Cognitive insights panel closed".to_string(),
                        ToastType::Info
                    );
                }
            }
            KeyCode::Char('m') if !self.input_mode => {
                // Quick model switching with 'm' key when not in input mode
                if let Some(orch) = &self.orchestration {
                    tokio::task::block_in_place(|| {
                        tokio::runtime::Handle::current().block_on(async {
                            let manager = orch.read().await;
                            if !manager.enabled_models.is_empty() {
                                let current_idx = self.selected_model.as_ref()
                                    .and_then(|m| manager.enabled_models.iter().position(|em| em == m))
                                    .unwrap_or(0);
                                let next_idx = (current_idx + 1) % manager.enabled_models.len();
                                self.selected_model = Some(manager.enabled_models[next_idx].clone());
                                
                                // Persist the selection
                                let persistence = crate::tui::chat::orchestration::model_persistence::ModelPersistence::new();
                                let _ = persistence.set_default_model(self.selected_model.clone()).await;
                            }
                        })
                    });
                }
            }
            // Handle arrow keys for scrolling when chat is focused
            KeyCode::Up if self.chat_focused => {
                // Scroll chat history up one line (older messages)
                self.scroll_offset += 1;
                self.smooth_scroll.scroll_to(self.scroll_offset);
                return Ok(());
            }
            KeyCode::Down if self.chat_focused => {
                // Scroll chat history down one line (newer messages)
                if self.scroll_offset > 0 {
                    self.scroll_offset -= 1;
                    self.smooth_scroll.scroll_to(self.scroll_offset);
                }
                return Ok(());
            }
            _ => {
                if self.input_mode && !self.chat_focused {
                    // When in input mode AND input is focused, handle all input through the processor
                    // Arrow keys in input mode navigate history, not scroll
                    tokio::task::block_in_place(|| {
                        tokio::runtime::Handle::current().block_on(
                            self.input_processor.handle_key(key.code)
                        )
                    })?;
                    
                    // Auto-scroll input area to keep cursor visible
                    // This happens after typing new characters
                    let text = self.input_processor.get_buffer();
                    let text_width = 80; // Approximate width
                    let wrapped_lines = Self::wrap_text(text, text_width);
                    let total_lines = wrapped_lines.len();
                    
                    // Calculate cursor line
                    let mut cursor_line = 0;
                    let mut char_count = 0;
                    for (line_idx, line) in wrapped_lines.iter().enumerate() {
                        if char_count + line.len() >= text.len() {
                            cursor_line = line_idx;
                            break;
                        }
                        char_count += line.len();
                    }
                    
                    // Auto-scroll to keep cursor visible (assuming ~3 visible lines)
                    let visible_lines = 3;
                    if cursor_line >= self.input_scroll_offset + visible_lines {
                        self.input_scroll_offset = cursor_line.saturating_sub(visible_lines - 1);
                    } else if cursor_line < self.input_scroll_offset {
                        self.input_scroll_offset = cursor_line;
                    }
                } else {
                    // Navigation mode - handle scrolling
                    match key.code {
                        KeyCode::Up => {
                            // Scroll up to see older messages (increase offset)
                            self.scroll_offset += 1;
                            self.smooth_scroll.scroll_to(self.scroll_offset);
                        }
                        KeyCode::Down => {
                            // Scroll down to see newer messages (decrease offset)
                            if self.scroll_offset > 0 {
                                self.scroll_offset -= 1;
                                self.smooth_scroll.scroll_to(self.scroll_offset);
                            }
                        }
                        // PageUp/PageDown are handled above for both modes
                        KeyCode::Home => {
                            // Jump to oldest messages
                            let state = tokio::task::block_in_place(|| {
                                tokio::runtime::Handle::current().block_on(async {
                                    self.state.read().await.messages.len()
                                })
                            });
                            self.scroll_offset = state * 2; // Large value to reach the top
                            self.smooth_scroll.scroll_to(self.scroll_offset);
                        }
                        KeyCode::End => {
                            // Jump to newest messages
                            self.scroll_offset = 0;
                            self.smooth_scroll.scroll_to(0);
                        }
                        KeyCode::Char('i') | KeyCode::Char('a') => {
                            // Enter input mode with 'i' or 'a' (vim-like)
                            self.input_mode = true;
                        }
                        _ => {}
                    }
                }
            }
        }
        
        Ok(())
    }
    
    fn update(&mut self) -> Result<()> {
        // Update streaming indicator animation
        self.streaming_indicator.update();
        
        // Check for streaming responses
        let (is_streaming, streaming_info) = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                let state = self.state.read().await;
                let streaming = state.messages.iter().any(|msg| {
                    matches!(msg, AssistantResponseType::Stream { stream_state, .. } 
                        if !matches!(stream_state, crate::tui::run::StreamingState::Complete))
                });
                
                // Get current streaming chunk if available
                let chunk = state.messages.iter().rev().find_map(|msg| {
                    if let AssistantResponseType::Stream { partial_content, .. } = msg {
                        Some(partial_content.clone())
                    } else {
                        None
                    }
                });
                
                (streaming, chunk)
            })
        });
        
        // Update indicators based on streaming state
        if is_streaming {
            self.typing_indicator.show();
            
            // Update streaming indicator
            if !matches!(self.streaming_indicator.state, crate::tui::chat::ui_enhancements::StreamingState::Streaming { .. }) {
                self.streaming_indicator.start_streaming();
            }
            
            // Add chunk if available
            if let Some(chunk) = streaming_info {
                self.streaming_indicator.add_chunk(&chunk);
            }
        } else {
            self.typing_indicator.hide();
            
            // Complete streaming if it was active
            if matches!(self.streaming_indicator.state, crate::tui::chat::ui_enhancements::StreamingState::Streaming { .. }) {
                self.streaming_indicator.complete();
            }
        }
        
        // Check for incoming messages
        if let Some(rx) = &mut self.response_rx {
            // Try to receive messages without blocking
            let mut received_count = 0;
            const MAX_MESSAGES_PER_UPDATE: usize = 5; // Prevent flooding
            
            loop {
                if received_count >= MAX_MESSAGES_PER_UPDATE {
                    break; // Process at most 5 messages per update to prevent UI freeze
                }
                
                match rx.try_recv() {
                    Ok(message) => {
                        received_count += 1;
                        
                        // Check for duplicate messages before adding
                        let is_duplicate = tokio::task::block_in_place(|| {
                            tokio::runtime::Handle::current().block_on(async {
                                let state = self.state.read().await;
                                // Check if this exact message was just added
                                if let Some(last_msg) = state.messages.last() {
                                    matches!((last_msg, &message), 
                                        (AssistantResponseType::Message { message: m1, .. }, 
                                         AssistantResponseType::Message { message: m2, .. }) if m1 == m2)
                                } else {
                                    false
                                }
                            })
                        });
                        
                        if is_duplicate {
                            tracing::debug!("Skipping duplicate message");
                            continue;
                        }
                        
                        // Add toast notification based on message type
                        match &message {
                            AssistantResponseType::Message { author, .. } => {
                                self.toast_manager.add_toast(
                                    format!("New message from {}", author),
                                    ToastType::Info
                                );
                            }
                            AssistantResponseType::Error { error_type, .. } => {
                                self.toast_manager.add_toast(
                                    format!("Error: {}", error_type),
                                    ToastType::Error
                                );
                            }
                            _ => {}
                        }
                        
                        // Handle streaming messages differently
                        match &message {
                            AssistantResponseType::Stream { .. } => {
                                // Update streaming indicator
                                self.streaming_indicator.add_chunk("");
                                
                                // For streaming messages, update or append
                                tokio::task::block_in_place(|| {
                                    tokio::runtime::Handle::current().block_on(async {
                                        let mut state = self.state.write().await;
                                        
                                        // Check if we should update the last message or add new
                                        if let Some(last) = state.messages.last_mut() {
                                            if let AssistantResponseType::Stream { id: last_id, .. } = last {
                                                if let AssistantResponseType::Stream { id: new_id, .. } = &message {
                                                    if last_id == new_id {
                                                        // Update existing stream message
                                                        *last = message;
                                                        return;
                                                    }
                                                }
                                            }
                                        }
                                        
                                        // Add as new message
                                        state.add_message_to_chat(message, 0);
                                    })
                                });
                            }
                            _ => {
                                // Regular message handling
                                tokio::task::block_in_place(|| {
                                    tokio::runtime::Handle::current().block_on(async {
                                        let mut state = self.state.write().await;
                                        state.add_message_to_chat(message, 0);
                                        tracing::info!("💬 Added AI response to chat state");
                                    })
                                });
                                
                                // Stop streaming indicator when complete message arrives
                                self.streaming_indicator.complete();
                            }
                        }
                        // Auto-scroll to bottom to show new message
                        // Note: scroll_offset of 0 means we're at the bottom (most recent messages)
                        self.scroll_offset = 0;
                        self.smooth_scroll.scroll_to(0);
                    }
                    Err(mpsc::error::TryRecvError::Empty) => break,
                    Err(mpsc::error::TryRecvError::Disconnected) => {
                        tracing::warn!("Response receiver disconnected");
                        self.response_rx = None;
                        break;
                    }
                }
            }
        }
        
        // Auto-scroll to bottom when new messages arrive
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                let state = self.state.read().await;
                if !state.messages.is_empty() && self.scroll_offset == 0 {
                    self.smooth_scroll.scroll_to(0); // Ensure smooth animation
                }
            })
        });
        
        Ok(())
    }
    
    fn name(&self) -> &str {
        "Chat"
    }
}