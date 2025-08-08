//! Markdown-like text formatting parser for the chat interface
//! 
//! Supports basic markdown syntax including bold, italic, code,
//! headers, lists, and links.

use ratatui::style::{Color, Modifier, Style};
use regex::Regex;

/// Formatted text result from markdown parsing
#[derive(Debug, Clone)]
pub struct FormattedText {
    pub lines: Vec<FormattedLine>,
}

/// A formatted line containing styled spans
#[derive(Debug, Clone)]
pub struct FormattedLine {
    pub spans: Vec<FormattedSpan>,
    pub line_type: LineType,
}

/// Individual formatted span with style
#[derive(Debug, Clone)]
pub struct FormattedSpan {
    pub content: String,
    pub style: Style,
}

/// Line types for special formatting
#[derive(Debug, Clone, PartialEq)]
pub enum LineType {
    Normal,
    Header { level: u8 },
    ListItem { ordered: bool, level: u8 },
    CodeBlock { language: Option<String> },
    BlockQuote,
    HorizontalRule,
}

/// Markdown parser
#[derive(Clone)]
pub struct MarkdownParser {
    /// Compiled regex patterns
    patterns: MarkdownPatterns,
}

/// Regex patterns for markdown syntax
#[derive(Clone)]
struct MarkdownPatterns {
    bold: Regex,
    italic: Regex,
    code: Regex,
    link: Regex,
    header: Regex,
    list_item: Regex,
    code_block: Regex,
    block_quote: Regex,
    horizontal_rule: Regex,
}

impl MarkdownParser {
    pub fn new() -> Self {
        Self {
            patterns: MarkdownPatterns {
                bold: Regex::new(r"\*\*([^*]+)\*\*").unwrap(),
                italic: Regex::new(r"\*([^*]+)\*|_([^_]+)_").unwrap(),
                code: Regex::new(r"`([^`]+)`").unwrap(),
                link: Regex::new(r"\[([^\]]+)\]\(([^)]+)\)").unwrap(),
                header: Regex::new(r"^(#{1,6})\s+(.+)$").unwrap(),
                list_item: Regex::new(r"^(\s*)([-*+]|\d+\.)\s+(.+)$").unwrap(),
                code_block: Regex::new(r"^```(\w+)?$").unwrap(),
                block_quote: Regex::new(r"^>\s+(.+)$").unwrap(),
                horizontal_rule: Regex::new(r"^---+$|^\*\*\*+$|^___+$").unwrap(),
            },
        }
    }
    
    /// Parse markdown text into formatted text
    pub fn parse(&self, text: &str) -> FormattedText {
        let mut lines = Vec::new();
        let mut in_code_block = false;
        let mut code_language = None;
        
        for line in text.lines() {
            // Check for code block boundaries
            if let Some(captures) = self.patterns.code_block.captures(line) {
                in_code_block = !in_code_block;
                if in_code_block {
                    code_language = captures.get(1).map(|m| m.as_str().to_string());
                } else {
                    code_language = None;
                }
                continue;
            }
            
            // Handle code block content
            if in_code_block {
                lines.push(FormattedLine {
                    spans: vec![FormattedSpan {
                        content: line.to_string(),
                        style: Style::default().fg(Color::Green),
                    }],
                    line_type: LineType::CodeBlock { language: code_language.clone() },
                });
                continue;
            }
            
            // Parse line type
            let (line_type, content) = self.parse_line_type(line);
            
            // Parse inline formatting
            let spans = self.parse_inline_formatting(content);
            
            lines.push(FormattedLine { spans, line_type });
        }
        
        FormattedText { lines }
    }
    
    /// Parse line type and extract content
    fn parse_line_type<'a>(&self, line: &'a str) -> (LineType, &'a str) {
        // Check for horizontal rule
        if self.patterns.horizontal_rule.is_match(line) {
            return (LineType::HorizontalRule, "");
        }
        
        // Check for header
        if let Some(captures) = self.patterns.header.captures(line) {
            let level = captures.get(1).unwrap().as_str().len() as u8;
            let content = captures.get(2).unwrap().as_str();
            return (LineType::Header { level }, content);
        }
        
        // Check for list item
        if let Some(captures) = self.patterns.list_item.captures(line) {
            let indent = captures.get(1).unwrap().as_str().len();
            let marker = captures.get(2).unwrap().as_str();
            let content = captures.get(3).unwrap().as_str();
            let ordered = marker.chars().next().unwrap().is_numeric();
            let level = (indent / 2) as u8;
            return (LineType::ListItem { ordered, level }, content);
        }
        
        // Check for block quote
        if let Some(captures) = self.patterns.block_quote.captures(line) {
            let content = captures.get(1).unwrap().as_str();
            return (LineType::BlockQuote, content);
        }
        
        (LineType::Normal, line)
    }
    
    /// Parse inline formatting within a line
    fn parse_inline_formatting(&self, text: &str) -> Vec<FormattedSpan> {
        let mut spans = Vec::new();
        let mut segments: Vec<(usize, usize, Style)> = Vec::new();
        
        // Find all bold segments
        for mat in self.patterns.bold.find_iter(text) {
            segments.push((
                mat.start(),
                mat.end(),
                Style::default().add_modifier(Modifier::BOLD),
            ));
        }
        
        // Find all italic segments
        for mat in self.patterns.italic.find_iter(text) {
            segments.push((
                mat.start(),
                mat.end(),
                Style::default().add_modifier(Modifier::ITALIC),
            ));
        }
        
        // Find all code segments
        for mat in self.patterns.code.find_iter(text) {
            segments.push((
                mat.start(),
                mat.end(),
                Style::default().fg(Color::Cyan).bg(Color::Rgb(40, 40, 40)),
            ));
        }
        
        // Find all links
        for mat in self.patterns.link.find_iter(text) {
            segments.push((
                mat.start(),
                mat.end(),
                Style::default().fg(Color::Blue).add_modifier(Modifier::UNDERLINED),
            ));
        }
        
        // Sort segments by start position
        segments.sort_by_key(|s| s.0);
        
        // Build spans
        let mut last_end = 0;
        for (start, end, style) in segments {
            // Add unstyled text before this segment
            if start > last_end {
                spans.push(FormattedSpan {
                    content: text[last_end..start].to_string(),
                    style: Style::default(),
                });
            }
            
            // Add styled segment
            let content = self.extract_formatted_content(&text[start..end]);
            spans.push(FormattedSpan { content, style });
            
            last_end = end;
        }
        
        // Add remaining unstyled text
        if last_end < text.len() {
            spans.push(FormattedSpan {
                content: text[last_end..].to_string(),
                style: Style::default(),
            });
        }
        
        // If no formatting was found, return the whole text as a single span
        if spans.is_empty() {
            spans.push(FormattedSpan {
                content: text.to_string(),
                style: Style::default(),
            });
        }
        
        spans
    }
    
    /// Extract content from formatted text (remove markdown syntax)
    fn extract_formatted_content(&self, text: &str) -> String {
        // Remove bold markers
        if let Some(captures) = self.patterns.bold.captures(text) {
            return captures.get(1).unwrap().as_str().to_string();
        }
        
        // Remove italic markers
        if let Some(captures) = self.patterns.italic.captures(text) {
            return captures.get(1)
                .or(captures.get(2))
                .unwrap()
                .as_str()
                .to_string();
        }
        
        // Remove code markers
        if let Some(captures) = self.patterns.code.captures(text) {
            return captures.get(1).unwrap().as_str().to_string();
        }
        
        // Format links
        if let Some(captures) = self.patterns.link.captures(text) {
            let text = captures.get(1).unwrap().as_str();
            let url = captures.get(2).unwrap().as_str();
            return format!("{} ({})", text, url);
        }
        
        text.to_string()
    }
    
    /// Apply line type styling to spans
    pub fn apply_line_type_style(line_type: &LineType, spans: &mut [FormattedSpan]) {
        match line_type {
            LineType::Header { level } => {
                let color = match level {
                    1 => Color::Cyan,
                    2 => Color::Blue,
                    3 => Color::Magenta,
                    _ => Color::White,
                };
                for span in spans {
                    span.style = span.style.fg(color).add_modifier(Modifier::BOLD);
                }
            }
            LineType::BlockQuote => {
                for span in spans {
                    span.style = span.style.fg(Color::Gray).add_modifier(Modifier::ITALIC);
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_header_parsing() {
        let parser = MarkdownParser::new();
        let result = parser.parse("# Header 1\n## Header 2");
        assert_eq!(result.lines.len(), 2);
        assert_eq!(result.lines[0].line_type, LineType::Header { level: 1 });
        assert_eq!(result.lines[1].line_type, LineType::Header { level: 2 });
    }
    
    #[test]
    fn test_list_parsing() {
        let parser = MarkdownParser::new();
        let result = parser.parse("- Item 1\n  - Nested item\n1. Ordered item");
        assert_eq!(result.lines.len(), 3);
        assert_eq!(result.lines[0].line_type, LineType::ListItem { ordered: false, level: 0 });
        assert_eq!(result.lines[1].line_type, LineType::ListItem { ordered: false, level: 1 });
        assert_eq!(result.lines[2].line_type, LineType::ListItem { ordered: true, level: 0 });
    }
}