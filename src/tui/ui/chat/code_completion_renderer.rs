//! Code completion UI renderer for the chat interface
//! 
//! Renders code completion suggestions in a popup overlay

use ratatui::{
    layout::{Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Padding, Paragraph},
    Frame,
};

use crate::tui::chat::{CodeCompletionSuggestion, CompletionKind};

/// Code completion renderer
#[derive(Clone)]
pub struct CodeCompletionRenderer {
    /// Maximum width for the completion popup
    max_width: u16,
    
    /// Maximum height for the completion popup
    max_height: u16,
    
    /// Style configuration
    styles: CompletionStyles,
}

/// Styles for code completion UI
#[derive(Clone)]
pub struct CompletionStyles {
    pub popup_style: Style,
    pub selected_style: Style,
    pub header_style: Style,
    pub kind_style: Style,
    pub detail_style: Style,
}

impl Default for CompletionStyles {
    fn default() -> Self {
        Self {
            popup_style: Style::default()
                .bg(Color::Rgb(40, 44, 52))
                .fg(Color::Rgb(200, 200, 200)),
            selected_style: Style::default()
                .bg(Color::Rgb(61, 89, 161))
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
            header_style: Style::default()
                .fg(Color::Rgb(150, 150, 150))
                .add_modifier(Modifier::ITALIC),
            kind_style: Style::default()
                .fg(Color::Rgb(120, 150, 200)),
            detail_style: Style::default()
                .fg(Color::Rgb(130, 130, 130)),
        }
    }
}

impl CodeCompletionRenderer {
    pub fn new() -> Self {
        Self {
            max_width: 60,
            max_height: 15,
            styles: CompletionStyles::default(),
        }
    }
    
    /// Render code completion popup
    pub fn render(
        &self,
        frame: &mut Frame,
        area: Rect,
        suggestions: &[CodeCompletionSuggestion],
        selected_index: Option<usize>,
        cursor_position: (u16, u16),
    ) {
        if suggestions.is_empty() {
            return;
        }
        
        // Calculate popup dimensions
        let popup_width = self.calculate_width(suggestions).min(self.max_width);
        let popup_height = (suggestions.len() as u16 + 2).min(self.max_height); // +2 for borders
        
        // Calculate popup position (below cursor if possible, above if not enough space)
        let popup_area = self.calculate_popup_area(area, cursor_position, popup_width, popup_height);
        
        // Create the completion list items
        let items: Vec<ListItem> = suggestions
            .iter()
            .enumerate()
            .map(|(i, suggestion)| {
                let is_selected = selected_index == Some(i);
                self.create_completion_item(suggestion, is_selected)
            })
            .collect();
        
        // Create the list widget
        let list = List::new(items)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(self.styles.popup_style)
                    .style(self.styles.popup_style)
                    .padding(Padding::horizontal(1)),
            )
            .highlight_style(self.styles.selected_style);
        
        // Render the list
        frame.render_widget(list, popup_area);
        
        // If there's a selected item with documentation, render it
        if let Some(index) = selected_index {
            if let Some(suggestion) = suggestions.get(index) {
                if let Some(doc) = &suggestion.detail {
                    self.render_documentation(frame, popup_area, doc);
                }
            }
        }
    }
    
    /// Calculate the width needed for the popup
    fn calculate_width(&self, suggestions: &[CodeCompletionSuggestion]) -> u16 {
        suggestions
            .iter()
            .map(|s| {
                let label_len = s.text.len() as u16;
                let kind_len = self.format_kind(&s.kind).len() as u16;
                let detail_len = s.detail.as_ref().map(|d| d.len() as u16).unwrap_or(0);
                
                // Label + kind + detail + spacing
                label_len + kind_len + detail_len + 6
            })
            .max()
            .unwrap_or(20)
            .max(20) // Minimum width
    }
    
    /// Calculate popup area based on cursor position and available space
    fn calculate_popup_area(
        &self,
        area: Rect,
        cursor_position: (u16, u16),
        width: u16,
        height: u16,
    ) -> Rect {
        let (cursor_x, cursor_y) = cursor_position;
        
        // Try to position below cursor
        let below_space = area.bottom().saturating_sub(cursor_y);
        let above_space = cursor_y.saturating_sub(area.top());
        
        let y = if below_space >= height {
            // Enough space below
            cursor_y + 1
        } else if above_space >= height {
            // Not enough space below, but enough above
            cursor_y.saturating_sub(height)
        } else {
            // Use whatever space is available
            if below_space > above_space {
                cursor_y + 1
            } else {
                area.top()
            }
        };
        
        // Adjust x position to keep popup within bounds
        let x = if cursor_x + width > area.right() {
            area.right().saturating_sub(width)
        } else {
            cursor_x
        };
        
        Rect {
            x,
            y,
            width: width.min(area.width),
            height: height.min(area.height),
        }
    }
    
    /// Create a list item for a completion suggestion
    fn create_completion_item(&self, suggestion: &CodeCompletionSuggestion, is_selected: bool) -> ListItem {
        let mut spans = vec![
            // Label
            Span::styled(
                suggestion.text.clone(),
                if is_selected {
                    self.styles.selected_style
                } else {
                    Style::default().fg(Color::White)
                },
            ),
            Span::raw(" "),
        ];
        
        // Kind
        spans.push(Span::styled(
            format!("[{}]", self.format_kind(&suggestion.kind)),
            self.styles.kind_style,
        ));
        
        // Detail (if available)
        if let Some(detail) = &suggestion.detail {
            spans.push(Span::raw(" "));
            spans.push(Span::styled(
                format!("- {}", detail),
                self.styles.detail_style,
            ));
        }
        
        // Score indicator removed - field not available
        
        ListItem::new(Line::from(spans))
    }
    
    /// Format completion kind for display
    fn format_kind(&self, kind: &CompletionKind) -> &'static str {
        match kind {
            CompletionKind::Function => "fn",
            CompletionKind::Variable => "var",
            CompletionKind::Class => "class",
            CompletionKind::Module => "mod",
            CompletionKind::Keyword => "kw",
            CompletionKind::Snippet => "snip",
            CompletionKind::Type => "type",
            CompletionKind::Property => "prop",
            CompletionKind::Method => "method",
            CompletionKind::Constant => "const",
        }
    }
    
    /// Render documentation popup
    fn render_documentation(
        &self,
        frame: &mut Frame,
        completion_area: Rect,
        documentation: &str,
    ) {
        // Position documentation to the right of completions if space allows
        let doc_width = 40.min(frame.area().width.saturating_sub(completion_area.right() + 2));
        if doc_width < 20 {
            return; // Not enough space
        }
        
        let doc_height = 10.min(completion_area.height);
        let doc_area = Rect {
            x: completion_area.right() + 1,
            y: completion_area.y,
            width: doc_width,
            height: doc_height,
        };
        
        let doc_widget = Paragraph::new(documentation)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Documentation")
                    .title_style(self.styles.header_style)
                    .border_style(self.styles.popup_style)
                    .style(self.styles.popup_style),
            )
            .wrap(ratatui::widgets::Wrap { trim: true })
            .style(self.styles.detail_style);
        
        frame.render_widget(doc_widget, doc_area);
    }
}

impl Default for CodeCompletionRenderer {
    fn default() -> Self {
        Self::new()
    }
}