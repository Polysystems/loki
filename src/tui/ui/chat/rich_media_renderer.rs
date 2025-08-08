//! Rich media rendering for the chat interface
//! 
//! Provides support for rendering images, charts, diagrams, and other rich media
//! in the terminal using various techniques like ASCII art, Unicode blocks, and more.

use ratatui::{
    layout::{Alignment, Constraint,  Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Frame,
};
use std::collections::HashMap;
use image::{DynamicImage, GenericImageView};

/// Rich media types supported
#[derive(Debug, Clone, PartialEq)]
pub enum MediaType {
    Image { format: ImageFormat },
    Chart { chart_type: ChartType },
    Diagram { diagram_type: DiagramType },
    Table { rows: usize, cols: usize },
    Code { language: String, highlighted: bool },
    Math { format: MathFormat },
}

#[derive(Debug, Clone, PartialEq)]
pub enum ImageFormat {
    Png,
    Jpeg,
    Gif,
    Svg,
    WebP,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ChartType {
    Line,
    Bar,
    Pie,
    Scatter,
    Area,
    Heatmap,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DiagramType {
    Flowchart,
    Sequence,
    Class,
    State,
    Gantt,
    Network,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MathFormat {
    LaTeX,
    AsciiMath,
    MathML,
}

/// Rich media content
#[derive(Debug, Clone)]
pub struct RichMediaContent {
    pub id: String,
    pub media_type: MediaType,
    pub title: Option<String>,
    pub data: MediaData,
    pub metadata: HashMap<String, String>,
}

/// Media data storage
#[derive(Debug, Clone)]
pub enum MediaData {
    Image(Vec<u8>),
    Chart(ChartData),
    Diagram(String), // Mermaid or PlantUML syntax
    Table(TableData),
    Math(String),
}

#[derive(Debug, Clone)]
pub struct ChartData {
    pub labels: Vec<String>,
    pub datasets: Vec<Dataset>,
    pub options: ChartOptions,
}

#[derive(Debug, Clone)]
pub struct Dataset {
    pub label: String,
    pub data: Vec<f64>,
    pub color: Color,
}

#[derive(Debug, Clone)]
pub struct ChartOptions {
    pub show_legend: bool,
    pub show_grid: bool,
    pub show_axes: bool,
}

#[derive(Debug, Clone)]
pub struct TableData {
    pub headers: Vec<String>,
    pub rows: Vec<Vec<String>>,
    pub alignments: Vec<Alignment>,
}

/// Rich media renderer
#[derive(Clone)]
pub struct RichMediaRenderer {
    /// Image rendering settings
    image_settings: ImageRenderSettings,
    
    /// Chart rendering settings
    chart_settings: ChartRenderSettings,
    
    /// Style configuration
    styles: MediaStyles,
}

/// Image rendering settings
#[derive(Clone)]
pub struct ImageRenderSettings {
    /// Use Unicode blocks for better quality
    pub use_unicode_blocks: bool,
    
    /// Use color (if terminal supports it)
    pub use_color: bool,
    
    /// Maximum width in characters
    pub max_width: u16,
    
    /// Maximum height in characters
    pub max_height: u16,
    
    /// Aspect ratio preservation
    pub preserve_aspect_ratio: bool,
}

/// Chart rendering settings
#[derive(Clone)]
pub struct ChartRenderSettings {
    /// Height of bar charts
    pub bar_height: u16,
    
    /// Width of chart area
    pub chart_width: u16,
    
    /// Show values on charts
    pub show_values: bool,
    
    /// Precision for decimal values
    pub value_precision: usize,
}

/// Styles for media rendering
#[derive(Clone)]
pub struct MediaStyles {
    pub border_style: Style,
    pub title_style: Style,
    pub caption_style: Style,
    pub error_style: Style,
    pub axis_style: Style,
    pub grid_style: Style,
}

impl Default for ImageRenderSettings {
    fn default() -> Self {
        Self {
            use_unicode_blocks: true,
            use_color: true,
            max_width: 80,
            max_height: 40,
            preserve_aspect_ratio: true,
        }
    }
}

impl Default for ChartRenderSettings {
    fn default() -> Self {
        Self {
            bar_height: 20,
            chart_width: 60,
            show_values: true,
            value_precision: 1,
        }
    }
}

impl Default for MediaStyles {
    fn default() -> Self {
        Self {
            border_style: Style::default().fg(Color::Rgb(100, 100, 100)),
            title_style: Style::default()
                .fg(Color::Rgb(200, 200, 200))
                .add_modifier(Modifier::BOLD),
            caption_style: Style::default()
                .fg(Color::Rgb(150, 150, 150))
                .add_modifier(Modifier::ITALIC),
            error_style: Style::default()
                .fg(Color::Red)
                .add_modifier(Modifier::BOLD),
            axis_style: Style::default().fg(Color::Rgb(120, 120, 120)),
            grid_style: Style::default().fg(Color::Rgb(60, 60, 60)),
        }
    }
}

impl RichMediaRenderer {
    pub fn new() -> Self {
        Self {
            image_settings: ImageRenderSettings::default(),
            chart_settings: ChartRenderSettings::default(),
            styles: MediaStyles::default(),
        }
    }
    
    /// Render rich media content
    pub fn render(
        &self,
        frame: &mut Frame,
        area: Rect,
        content: &RichMediaContent,
    ) {
        match &content.media_type {
            MediaType::Image { .. } => {
                if let MediaData::Image(data) = &content.data {
                    self.render_image(frame, area, data, content.title.as_deref());
                }
            }
            MediaType::Chart { chart_type } => {
                if let MediaData::Chart(data) = &content.data {
                    self.render_chart(frame, area, chart_type, data, content.title.as_deref());
                }
            }
            MediaType::Diagram { diagram_type } => {
                if let MediaData::Diagram(syntax) = &content.data {
                    self.render_diagram(frame, area, diagram_type, syntax, content.title.as_deref());
                }
            }
            MediaType::Table { .. } => {
                if let MediaData::Table(data) = &content.data {
                    self.render_table(frame, area, data, content.title.as_deref());
                }
            }
            MediaType::Math { format } => {
                if let MediaData::Math(expr) = &content.data {
                    self.render_math(frame, area, format, expr, content.title.as_deref());
                }
            }
            _ => {}
        }
    }
    
    /// Render an image using ASCII/Unicode art
    fn render_image(
        &self,
        frame: &mut Frame,
        area: Rect,
        image_data: &[u8],
        title: Option<&str>,
    ) {
        // Create block with title
        let block = Block::default()
            .borders(Borders::ALL)
            .title(title.unwrap_or("Image"))
            .title_style(self.styles.title_style)
            .border_style(self.styles.border_style);
        
        let inner_area = block.inner(area);
        frame.render_widget(block, area);
        
        // Try to load and render the image
        match image::load_from_memory(image_data) {
            Ok(img) => {
                let ascii_art = self.image_to_ascii(&img, inner_area.width, inner_area.height);
                let paragraph = Paragraph::new(ascii_art)
                    .alignment(Alignment::Center);
                frame.render_widget(paragraph, inner_area);
            }
            Err(_) => {
                let error_msg = Paragraph::new("Failed to load image")
                    .style(self.styles.error_style)
                    .alignment(Alignment::Center);
                frame.render_widget(error_msg, inner_area);
            }
        }
    }
    
    /// Convert image to ASCII art
    fn image_to_ascii(&self, img: &DynamicImage, max_width: u16, max_height: u16) -> Vec<Line> {
        let (width, height) = img.dimensions();
        
        // Calculate scaling to fit within bounds
        let scale_x = width as f32 / max_width as f32;
        let scale_y = height as f32 / (max_height * 2) as f32; // Characters are ~2x taller than wide
        let scale = scale_x.max(scale_y).max(1.0);
        
        let new_width = (width as f32 / scale) as u32;
        let new_height = (height as f32 / scale) as u32;
        
        // Resize image
        let resized = img.resize(new_width, new_height, image::imageops::FilterType::Lanczos3);
        
        let mut lines = Vec::new();
        
        if self.image_settings.use_unicode_blocks {
            // Use Unicode block characters for better quality
            let blocks = [' ', '░', '▒', '▓', '█'];
            
            for y in 0..resized.height() {
                let mut spans = Vec::new();
                for x in 0..resized.width() {
                    let pixel = resized.get_pixel(x, y);
                    let brightness = (pixel[0] as u32 + pixel[1] as u32 + pixel[2] as u32) / 3;
                    let block_idx = (brightness * (blocks.len() - 1) as u32 / 255) as usize;
                    
                    if self.image_settings.use_color {
                        spans.push(Span::styled(
                            blocks[block_idx].to_string(),
                            Style::default().fg(Color::Rgb(pixel[0], pixel[1], pixel[2])),
                        ));
                    } else {
                        spans.push(Span::raw(blocks[block_idx].to_string()));
                    }
                }
                lines.push(Line::from(spans));
            }
        } else {
            // Use simple ASCII characters
            let ascii_chars = " .:-=+*#%@";
            
            for y in 0..resized.height() {
                let mut line = String::new();
                for x in 0..resized.width() {
                    let pixel = resized.get_pixel(x, y);
                    let brightness = (pixel[0] as u32 + pixel[1] as u32 + pixel[2] as u32) / 3;
                    let char_idx = (brightness * (ascii_chars.len() - 1) as u32 / 255) as usize;
                    line.push(ascii_chars.chars().nth(char_idx).unwrap_or(' '));
                }
                lines.push(Line::from(line));
            }
        }
        
        lines
    }
    
    /// Render a chart
    fn render_chart(
        &self,
        frame: &mut Frame,
        area: Rect,
        chart_type: &ChartType,
        data: &ChartData,
        title: Option<&str>,
    ) {
        let block = Block::default()
            .borders(Borders::ALL)
            .title(title.unwrap_or("Chart"))
            .title_style(self.styles.title_style)
            .border_style(self.styles.border_style);
        
        let inner_area = block.inner(area);
        frame.render_widget(block, area);
        
        match chart_type {
            ChartType::Bar => self.render_bar_chart(frame, inner_area, data),
            ChartType::Line => self.render_line_chart(frame, inner_area, data),
            _ => {
                let msg = Paragraph::new(format!("{:?} chart rendering not yet implemented", chart_type))
                    .style(self.styles.caption_style)
                    .alignment(Alignment::Center);
                frame.render_widget(msg, inner_area);
            }
        }
    }
    
    /// Render a simple bar chart
    fn render_bar_chart(
        &self,
        frame: &mut Frame,
        area: Rect,
        data: &ChartData,
    ) {
        if data.datasets.is_empty() || data.labels.is_empty() {
            return;
        }
        
        let dataset = &data.datasets[0]; // Use first dataset for now
        let max_value = dataset.data.iter().cloned().fold(0.0, f64::max);
        
        if max_value == 0.0 {
            return;
        }
        
        let bar_width = area.width / data.labels.len() as u16;
        let mut x = area.x;
        
        for (_i, (_label, &value)) in data.labels.iter().zip(&dataset.data).enumerate() {
            let bar_height = ((value / max_value) * area.height as f64) as u16;
            let bar_area = Rect {
                x,
                y: area.y + area.height - bar_height,
                width: bar_width.saturating_sub(1), // Gap between bars
                height: bar_height,
            };
            
            // Draw bar
            let bar = Block::default()
                .style(Style::default().bg(dataset.color));
            frame.render_widget(bar, bar_area);
            
            // Draw value on top
            if self.chart_settings.show_values && bar_height > 0 {
                let value_text = format!("{:.1}", value);
                let value_widget = Paragraph::new(value_text)
                    .style(Style::default().fg(Color::White))
                    .alignment(Alignment::Center);
                
                let value_area = Rect {
                    x: bar_area.x,
                    y: bar_area.y.saturating_sub(1),
                    width: bar_area.width,
                    height: 1,
                };
                frame.render_widget(value_widget, value_area);
            }
            
            x += bar_width;
        }
    }
    
    /// Render a simple line chart
    fn render_line_chart(
        &self,
        frame: &mut Frame,
        area: Rect,
        _data: &ChartData,
    ) {
        // Simple ASCII line chart implementation
        let msg = Paragraph::new("Line chart visualization")
            .style(self.styles.caption_style)
            .alignment(Alignment::Center);
        frame.render_widget(msg, area);
    }
    
    /// Render a diagram
    fn render_diagram(
        &self,
        frame: &mut Frame,
        area: Rect,
        diagram_type: &DiagramType,
        syntax: &str,
        title: Option<&str>,
    ) {
        let block = Block::default()
            .borders(Borders::ALL)
            .title(title.unwrap_or("Diagram"))
            .title_style(self.styles.title_style)
            .border_style(self.styles.border_style);
        
        let inner_area = block.inner(area);
        frame.render_widget(block, area);
        
        // For now, just display the diagram syntax
        let content = vec![
            Line::from(format!("{:?} Diagram", diagram_type)),
            Line::from(""),
            Line::from("Diagram rendering in terminal:"),
            Line::from(""),
        ];
        
        let syntax_lines: Vec<Line> = syntax
            .lines()
            .map(|line| Line::from(line))
            .collect();
        
        let mut all_lines = content;
        all_lines.extend(syntax_lines);
        
        let paragraph = Paragraph::new(all_lines)
            .style(self.styles.caption_style)
            .wrap(ratatui::widgets::Wrap { trim: true });
        
        frame.render_widget(paragraph, inner_area);
    }
    
    /// Render a table
    fn render_table(
        &self,
        frame: &mut Frame,
        area: Rect,
        data: &TableData,
        title: Option<&str>,
    ) {
        use ratatui::widgets::{Table, Row, Cell};
        
        // Create header row
        let header_cells: Vec<Cell> = data.headers
            .iter()
            .map(|h| Cell::from(h.as_str()).style(self.styles.title_style))
            .collect();
        let header = Row::new(header_cells).height(1);
        
        // Create data rows
        let rows: Vec<Row> = data.rows
            .iter()
            .map(|row| {
                let cells: Vec<Cell> = row
                    .iter()
                    .map(|cell| Cell::from(cell.as_str()))
                    .collect();
                Row::new(cells).height(1)
            })
            .collect();
        
        // Calculate column widths
        let col_count = data.headers.len().max(1);
        let col_width = area.width / col_count as u16;
        let widths: Vec<Constraint> = (0..col_count)
            .map(|_| Constraint::Length(col_width))
            .collect();
        
        // Create table
        let table = Table::new(rows, widths)
            .header(header)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(title.unwrap_or("Table"))
                    .title_style(self.styles.title_style)
                    .border_style(self.styles.border_style)
            )
            .column_spacing(1);
        
        frame.render_widget(table, area);
    }
    
    /// Render mathematical expressions
    fn render_math(
        &self,
        frame: &mut Frame,
        area: Rect,
        format: &MathFormat,
        expression: &str,
        title: Option<&str>,
    ) {
        let block = Block::default()
            .borders(Borders::ALL)
            .title(title.unwrap_or("Math"))
            .title_style(self.styles.title_style)
            .border_style(self.styles.border_style);
        
        let inner_area = block.inner(area);
        frame.render_widget(block, area);
        
        // For now, display the raw expression
        let content = vec![
            Line::from(format!("{:?} Expression:", format)),
            Line::from(""),
            Line::from(expression),
        ];
        
        let paragraph = Paragraph::new(content)
            .style(self.styles.caption_style)
            .alignment(Alignment::Center)
            .wrap(ratatui::widgets::Wrap { trim: true });
        
        frame.render_widget(paragraph, inner_area);
    }
}

impl Default for RichMediaRenderer {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper function to detect media type from content
pub fn detect_media_type(content: &[u8], filename: Option<&str>) -> Option<MediaType> {
    // Check by file extension first
    if let Some(name) = filename {
        let ext = name.split('.').last()?.to_lowercase();
        match ext.as_str() {
            "png" => return Some(MediaType::Image { format: ImageFormat::Png }),
            "jpg" | "jpeg" => return Some(MediaType::Image { format: ImageFormat::Jpeg }),
            "gif" => return Some(MediaType::Image { format: ImageFormat::Gif }),
            "svg" => return Some(MediaType::Image { format: ImageFormat::Svg }),
            "webp" => return Some(MediaType::Image { format: ImageFormat::WebP }),
            _ => {}
        }
    }
    
    // Check by magic bytes
    if content.len() >= 8 {
        if content.starts_with(b"\x89PNG\r\n\x1a\n") {
            return Some(MediaType::Image { format: ImageFormat::Png });
        }
        if content.starts_with(b"\xFF\xD8\xFF") {
            return Some(MediaType::Image { format: ImageFormat::Jpeg });
        }
        if content.starts_with(b"GIF87a") || content.starts_with(b"GIF89a") {
            return Some(MediaType::Image { format: ImageFormat::Gif });
        }
    }
    
    None
}