//! Template UI components for the chat interface
//! 
//! Provides visual components for template selection, preview, and variable input.

use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, List, ListItem, Paragraph, Wrap},
    Frame,
};

use super::template_manager::{
    ChatTemplate, CodeSnippet, TemplateCategory, TemplateInputState, 
    TemplateVariable
};

/// Template picker UI
#[derive(Clone)]
pub struct TemplatePicker {
    /// Currently selected template index
    selected_index: usize,
    
    /// Filter by category
    filter_category: Option<TemplateCategory>,
    
    /// Search query
    search_query: String,
    
    /// Show preview
    show_preview: bool,
    
    /// Style configuration
    styles: TemplateStyles,
}

/// Template input dialog
#[derive(Clone)]
pub struct TemplateInputDialog {
    /// Input state
    pub state: TemplateInputState,
    
    /// Style configuration
    styles: TemplateStyles,
}

/// Styles for template UI
#[derive(Clone)]
pub struct TemplateStyles {
    pub border_style: Style,
    pub title_style: Style,
    pub selected_style: Style,
    pub category_style: Style,
    pub variable_style: Style,
    pub required_style: Style,
    pub preview_style: Style,
    pub error_style: Style,
}

impl Default for TemplateStyles {
    fn default() -> Self {
        Self {
            border_style: Style::default().fg(Color::Rgb(100, 100, 100)),
            title_style: Style::default()
                .fg(Color::Rgb(200, 200, 200))
                .add_modifier(Modifier::BOLD),
            selected_style: Style::default()
                .bg(Color::Rgb(50, 50, 60))
                .add_modifier(Modifier::BOLD),
            category_style: Style::default()
                .fg(Color::Rgb(150, 150, 200))
                .add_modifier(Modifier::ITALIC),
            variable_style: Style::default().fg(Color::Rgb(100, 200, 100)),
            required_style: Style::default()
                .fg(Color::Rgb(255, 150, 150))
                .add_modifier(Modifier::BOLD),
            preview_style: Style::default().fg(Color::Rgb(180, 180, 180)),
            error_style: Style::default()
                .fg(Color::Red)
                .add_modifier(Modifier::BOLD),
        }
    }
}

impl TemplatePicker {
    pub fn new() -> Self {
        Self {
            selected_index: 0,
            filter_category: None,
            search_query: String::new(),
            show_preview: true,
            styles: TemplateStyles::default(),
        }
    }
    
    /// Render the template picker
    pub fn render(
        &self,
        frame: &mut Frame,
        area: Rect,
        templates: &[&ChatTemplate],
    ) {
        let layout: Vec<Rect> = if self.show_preview {
            Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                .split(area)
                .to_vec()
        } else {
            vec![area, Rect::default()]
        };
        
        // Render template list
        self.render_template_list(frame, layout[0], templates);
        
        // Render preview if enabled
        if self.show_preview && !templates.is_empty() {
            if let Some(template) = templates.get(self.selected_index) {
                self.render_template_preview(frame, layout[1], template);
            }
        }
    }
    
    /// Render template list
    fn render_template_list(
        &self,
        frame: &mut Frame,
        area: Rect,
        templates: &[&ChatTemplate],
    ) {
        let items: Vec<ListItem> = templates
            .iter()
            .enumerate()
            .map(|(idx, template)| {
                let is_selected = idx == self.selected_index;
                
                let mut spans = vec![
                    Span::styled(
                        format!("{:<20}", template.name),
                        if is_selected {
                            self.styles.selected_style
                        } else {
                            Style::default()
                        },
                    ),
                ];
                
                // Add category
                spans.push(Span::raw(" "));
                spans.push(Span::styled(
                    format!("[{:?}]", template.category),
                    self.styles.category_style,
                ));
                
                // Add shortcuts if any
                if !template.shortcuts.is_empty() {
                    spans.push(Span::raw(" "));
                    spans.push(Span::styled(
                        format!("({})", template.shortcuts.join(", ")),
                        Style::default().fg(Color::DarkGray),
                    ));
                }
                
                // Add usage count
                if template.usage_count > 0 {
                    spans.push(Span::raw(" "));
                    spans.push(Span::styled(
                        format!("★{}", template.usage_count),
                        Style::default().fg(Color::Yellow),
                    ));
                }
                
                ListItem::new(Line::from(spans))
            })
            .collect();
        
        let list = List::new(items)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Templates")
                    .title_style(self.styles.title_style)
                    .border_style(self.styles.border_style),
            )
            .highlight_style(self.styles.selected_style);
        
        frame.render_widget(list, area);
    }
    
    /// Render template preview
    fn render_template_preview(
        &self,
        frame: &mut Frame,
        area: Rect,
        template: &ChatTemplate,
    ) {
        let layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(6), // Info
                Constraint::Min(5),    // Content preview
                Constraint::Length(8), // Variables
            ])
            .split(area);
        
        // Template info
        let info_text = vec![
            Line::from(vec![
                Span::raw("Name: "),
                Span::styled(&template.name, self.styles.title_style),
            ]),
            Line::from(vec![
                Span::raw("Description: "),
                Span::raw(template.description.as_deref().unwrap_or("No description")),
            ]),
            Line::from(vec![
                Span::raw("Tags: "),
                Span::styled(template.tags.join(", "), Style::default().fg(Color::Cyan)),
            ]),
        ];
        
        let info = Paragraph::new(info_text)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Template Info")
                    .title_style(self.styles.title_style)
                    .border_style(self.styles.border_style),
            );
        
        frame.render_widget(info, layout[0]);
        
        // Content preview
        let preview = Paragraph::new(template.content.clone())
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Preview")
                    .title_style(self.styles.title_style)
                    .border_style(self.styles.border_style),
            )
            .style(self.styles.preview_style)
            .wrap(Wrap { trim: true });
        
        frame.render_widget(preview, layout[1]);
        
        // Variables
        let var_items: Vec<ListItem> = template.variables
            .iter()
            .map(|var| {
                let mut spans = vec![
                    Span::styled(
                        format!("{{{{{}}}}}", var.name),
                        self.styles.variable_style,
                    ),
                ];
                
                if var.required {
                    spans.push(Span::styled(" *", self.styles.required_style));
                }
                
                spans.push(Span::raw(format!(" - {}", 
                    var.description.as_deref().unwrap_or("No description")
                )));
                
                if let Some(default) = &var.default_value {
                    spans.push(Span::styled(
                        format!(" (default: {})", default),
                        Style::default().fg(Color::DarkGray),
                    ));
                }
                
                ListItem::new(Line::from(spans))
            })
            .collect();
        
        let variables = List::new(var_items)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Variables")
                    .title_style(self.styles.title_style)
                    .border_style(self.styles.border_style),
            );
        
        frame.render_widget(variables, layout[2]);
    }
    
    /// Move selection up
    pub fn select_previous(&mut self, _count: usize) {
        if self.selected_index > 0 {
            self.selected_index = self.selected_index.saturating_sub(1);
        }
    }
    
    /// Move selection down
    pub fn select_next(&mut self, _count: usize) {
        self.selected_index += 1;
    }
    
    /// Get selected template index
    pub fn selected_index(&self) -> usize {
        self.selected_index
    }
    
    /// Toggle preview
    pub fn toggle_preview(&mut self) {
        self.show_preview = !self.show_preview;
    }
    
    /// Set category filter
    pub fn set_filter(&mut self, category: Option<TemplateCategory>) {
        self.filter_category = category;
        self.selected_index = 0;
    }
}

impl TemplateInputDialog {
    pub fn new(state: TemplateInputState) -> Self {
        Self {
            state,
            styles: TemplateStyles::default(),
        }
    }
    
    /// Render the input dialog
    pub fn render(
        &self,
        frame: &mut Frame,
        area: Rect,
        template: &ChatTemplate,
    ) {
        // Clear the area
        frame.render_widget(Clear, area);
        
        // Calculate dialog size (centered)
        let dialog_width = 60.min(area.width - 4);
        let dialog_height = (template.variables.len() as u16 * 4 + 8).min(area.height - 4);
        
        let dialog_area = Rect {
            x: area.x + (area.width - dialog_width) / 2,
            y: area.y + (area.height - dialog_height) / 2,
            width: dialog_width,
            height: dialog_height,
        };
        
        let block = Block::default()
            .borders(Borders::ALL)
            .title(format!("Fill Template: {}", template.name))
            .title_style(self.styles.title_style)
            .border_style(self.styles.border_style);
        
        let inner = block.inner(dialog_area);
        frame.render_widget(block, dialog_area);
        
        // Layout for variables
        let var_height = 3;
        let constraints: Vec<Constraint> = template.variables
            .iter()
            .map(|_| Constraint::Length(var_height))
            .chain(std::iter::once(Constraint::Min(1))) // Space for error/help
            .collect();
        
        let layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints(constraints)
            .split(inner);
        
        // Render each variable input
        for (idx, var) in template.variables.iter().enumerate() {
            if idx < layout.len() - 1 {
                self.render_variable_input(frame, layout[idx], var, idx == self.state.current_variable_index);
            }
        }
        
        // Render error message or help text
        if let Some(error) = &self.state.error_message {
            let error_text = Paragraph::new(error.as_str())
                .style(self.styles.error_style)
                .alignment(Alignment::Center);
            frame.render_widget(error_text, layout[layout.len() - 1]);
        } else {
            let help_text = Paragraph::new("Tab: Next field | Shift+Tab: Previous | Enter: Apply | Esc: Cancel")
                .style(Style::default().fg(Color::DarkGray))
                .alignment(Alignment::Center);
            frame.render_widget(help_text, layout[layout.len() - 1]);
        }
    }
    
    /// Render a single variable input
    fn render_variable_input(
        &self,
        frame: &mut Frame,
        area: Rect,
        variable: &TemplateVariable,
        is_focused: bool,
    ) {
        let layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(1), Constraint::Length(1)])
            .split(area);
        
        // Variable label
        let mut label_spans = vec![
            Span::styled(&variable.name, self.styles.variable_style),
        ];
        
        if variable.required {
            label_spans.push(Span::styled(" *", self.styles.required_style));
        }
        
        if let Some(desc) = &variable.description {
            label_spans.push(Span::raw(format!(" - {}", desc)));
        }
        
        let label = Paragraph::new(Line::from(label_spans));
        frame.render_widget(label, layout[0]);
        
        // Input field
        let value = self.state.variable_values
            .get(&variable.name)
            .map(|s| s.as_str())
            .unwrap_or("");
        
        let input_style = if is_focused {
            Style::default().bg(Color::Rgb(40, 40, 50))
        } else {
            Style::default()
        };
        
        let input = Paragraph::new(value)
            .style(input_style)
            .block(
                Block::default()
                    .borders(Borders::BOTTOM)
                    .border_style(if is_focused {
                        Style::default().fg(Color::Cyan)
                    } else {
                        self.styles.border_style
                    }),
            );
        
        frame.render_widget(input, layout[1]);
    }
}

impl Default for TemplatePicker {
    fn default() -> Self {
        Self::new()
    }
}

/// Snippet picker UI (similar to template picker but for code snippets)
#[derive(Clone)]
pub struct SnippetPicker {
    selected_index: usize,
    filter_language: Option<String>,
    styles: TemplateStyles,
}

impl SnippetPicker {
    pub fn new() -> Self {
        Self {
            selected_index: 0,
            filter_language: None,
            styles: TemplateStyles::default(),
        }
    }
    
    /// Render snippet picker
    pub fn render(
        &self,
        frame: &mut Frame,
        area: Rect,
        snippets: &[&CodeSnippet],
    ) {
        let items: Vec<ListItem> = snippets
            .iter()
            .enumerate()
            .map(|(idx, snippet)| {
                let is_selected = idx == self.selected_index;
                
                let mut spans = vec![
                    Span::styled(
                        format!("{:<20}", snippet.name),
                        if is_selected {
                            self.styles.selected_style
                        } else {
                            Style::default()
                        },
                    ),
                ];
                
                // Add language
                spans.push(Span::raw(" "));
                spans.push(Span::styled(
                    format!("[{}]", snippet.language),
                    Style::default().fg(Color::Green),
                ));
                
                // Add tags
                if !snippet.tags.is_empty() {
                    spans.push(Span::raw(" "));
                    spans.push(Span::styled(
                        snippet.tags.join(", "),
                        Style::default().fg(Color::Cyan),
                    ));
                }
                
                ListItem::new(Line::from(spans))
            })
            .collect();
        
        let list = List::new(items)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Code Snippets")
                    .title_style(self.styles.title_style)
                    .border_style(self.styles.border_style),
            )
            .highlight_style(self.styles.selected_style);
        
        frame.render_widget(list, area);
    }
}