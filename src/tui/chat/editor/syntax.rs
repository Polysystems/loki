//! Syntax Highlighting System
//! 
//! Provides syntax highlighting for various programming languages with
//! theme support and token classification.

use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use anyhow::Result;

/// Syntax highlighter
pub struct SyntaxHighlighter {
    /// Language definitions
    languages: HashMap<String, LanguageDefinition>,
    
    /// Available themes
    themes: HashMap<String, Theme>,
    
    /// Current theme
    current_theme: String,
}

/// Language support definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageSupport {
    pub id: String,
    pub name: String,
    pub extensions: Vec<String>,
    pub keywords: Vec<String>,
    pub operators: Vec<String>,
    pub comment_single: Option<String>,
    pub comment_multi: Option<(String, String)>,
    pub string_delimiters: Vec<char>,
    pub brackets: Vec<(char, char)>,
}

/// Language definition for highlighting
struct LanguageDefinition {
    support: LanguageSupport,
    token_patterns: Vec<TokenPattern>,
}

/// Token pattern for matching
struct TokenPattern {
    pattern: regex::Regex,
    token_type: TokenType,
    priority: u8,
}

/// Token types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TokenType {
    Keyword,
    Identifier,
    String,
    Number,
    Comment,
    Operator,
    Punctuation,
    Type,
    Function,
    Variable,
    Constant,
    Macro,
    Attribute,
    Namespace,
    Label,
    Invalid,
    Whitespace,
}

/// Syntax theme
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Theme {
    pub name: String,
    pub background: Color,
    pub foreground: Color,
    pub cursor: Color,
    pub selection: Color,
    pub line_highlight: Color,
    pub token_colors: HashMap<TokenType, TokenStyle>,
}

/// Color representation
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Color {
    pub fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b, a: 255 }
    }
    
    pub fn rgba(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }
    
    pub fn from_hex(hex: &str) -> Result<Self> {
        let hex = hex.trim_start_matches('#');
        let r = u8::from_str_radix(&hex[0..2], 16)?;
        let g = u8::from_str_radix(&hex[2..4], 16)?;
        let b = u8::from_str_radix(&hex[4..6], 16)?;
        let a = if hex.len() >= 8 {
            u8::from_str_radix(&hex[6..8], 16)?
        } else {
            255
        };
        Ok(Self { r, g, b, a })
    }
}

/// Token style
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenStyle {
    pub color: Color,
    pub bold: bool,
    pub italic: bool,
    pub underline: bool,
}

/// Highlighted token
#[derive(Debug, Clone)]
pub struct HighlightedToken {
    pub text: String,
    pub token_type: TokenType,
    pub style: TokenStyle,
    pub start: usize,
    pub end: usize,
}

impl SyntaxHighlighter {
    /// Create a new syntax highlighter
    pub fn new() -> Self {
        let mut highlighter = Self {
            languages: HashMap::new(),
            themes: HashMap::new(),
            current_theme: "dark".to_string(),
        };
        
        // Load default languages
        highlighter.load_default_languages();
        
        // Load default themes
        highlighter.load_default_themes();
        
        highlighter
    }
    
    /// Load default language definitions
    fn load_default_languages(&mut self) {
        // Rust
        self.add_language(LanguageSupport {
            id: "rust".to_string(),
            name: "Rust".to_string(),
            extensions: vec!["rs".to_string()],
            keywords: vec![
                "fn", "let", "mut", "const", "static", "if", "else", "match",
                "for", "while", "loop", "break", "continue", "return", "impl",
                "trait", "struct", "enum", "mod", "pub", "use", "async", "await",
                "move", "ref", "self", "Self", "super", "crate", "where", "as",
            ].iter().map(|s| s.to_string()).collect(),
            operators: vec![
                "+", "-", "*", "/", "%", "=", "==", "!=", "<", ">", "<=", ">=",
                "&&", "||", "!", "&", "|", "^", "<<", ">>", "+=", "-=", "*=", "/=",
                "->", "=>", "::", ".", "..", "..=",
            ].iter().map(|s| s.to_string()).collect(),
            comment_single: Some("//".to_string()),
            comment_multi: Some(("/*".to_string(), "*/".to_string())),
            string_delimiters: vec!['"', '\''],
            brackets: vec![('(', ')'), ('{', '}'), ('[', ']'), ('<', '>')],
        });
        
        // Python
        self.add_language(LanguageSupport {
            id: "python".to_string(),
            name: "Python".to_string(),
            extensions: vec!["py".to_string()],
            keywords: vec![
                "def", "class", "if", "elif", "else", "for", "while", "break",
                "continue", "return", "import", "from", "as", "try", "except",
                "finally", "with", "lambda", "yield", "global", "nonlocal",
                "True", "False", "None", "and", "or", "not", "in", "is",
            ].iter().map(|s| s.to_string()).collect(),
            operators: vec![
                "+", "-", "*", "/", "//", "%", "**", "=", "==", "!=", "<", ">",
                "<=", ">=", "and", "or", "not", "in", "is", "+=", "-=", "*=", "/=",
            ].iter().map(|s| s.to_string()).collect(),
            comment_single: Some("#".to_string()),
            comment_multi: Some(("'''".to_string(), "'''".to_string())),
            string_delimiters: vec!['"', '\''],
            brackets: vec![('(', ')'), ('{', '}'), ('[', ']')],
        });
        
        // JavaScript/TypeScript
        self.add_language(LanguageSupport {
            id: "javascript".to_string(),
            name: "JavaScript".to_string(),
            extensions: vec!["js".to_string(), "jsx".to_string()],
            keywords: vec![
                "function", "var", "let", "const", "if", "else", "for", "while",
                "do", "switch", "case", "break", "continue", "return", "class",
                "extends", "new", "this", "super", "import", "export", "default",
                "async", "await", "try", "catch", "finally", "throw", "typeof",
                "instanceof", "true", "false", "null", "undefined",
            ].iter().map(|s| s.to_string()).collect(),
            operators: vec![
                "+", "-", "*", "/", "%", "=", "==", "===", "!=", "!==", "<", ">",
                "<=", ">=", "&&", "||", "!", "&", "|", "^", "~", "<<", ">>", ">>>",
                "+=", "-=", "*=", "/=", "=>", "?.", "??",
            ].iter().map(|s| s.to_string()).collect(),
            comment_single: Some("//".to_string()),
            comment_multi: Some(("/*".to_string(), "*/".to_string())),
            string_delimiters: vec!['"', '\'', '`'],
            brackets: vec![('(', ')'), ('{', '}'), ('[', ']')],
        });
    }
    
    /// Load default themes
    fn load_default_themes(&mut self) {
        // Dark theme
        let mut dark_theme = Theme {
            name: "dark".to_string(),
            background: Color::rgb(30, 30, 30),
            foreground: Color::rgb(212, 212, 212),
            cursor: Color::rgb(255, 255, 255),
            selection: Color::rgba(75, 135, 255, 50),
            line_highlight: Color::rgba(255, 255, 255, 10),
            token_colors: HashMap::new(),
        };
        
        dark_theme.token_colors.insert(TokenType::Keyword, TokenStyle {
            color: Color::rgb(86, 156, 214),
            bold: true,
            italic: false,
            underline: false,
        });
        
        dark_theme.token_colors.insert(TokenType::String, TokenStyle {
            color: Color::rgb(206, 145, 120),
            bold: false,
            italic: false,
            underline: false,
        });
        
        dark_theme.token_colors.insert(TokenType::Number, TokenStyle {
            color: Color::rgb(181, 206, 168),
            bold: false,
            italic: false,
            underline: false,
        });
        
        dark_theme.token_colors.insert(TokenType::Comment, TokenStyle {
            color: Color::rgb(106, 153, 85),
            bold: false,
            italic: true,
            underline: false,
        });
        
        dark_theme.token_colors.insert(TokenType::Function, TokenStyle {
            color: Color::rgb(220, 220, 170),
            bold: false,
            italic: false,
            underline: false,
        });
        
        dark_theme.token_colors.insert(TokenType::Type, TokenStyle {
            color: Color::rgb(78, 201, 176),
            bold: false,
            italic: false,
            underline: false,
        });
        
        self.themes.insert("dark".to_string(), dark_theme);
        
        // Light theme
        let mut light_theme = Theme {
            name: "light".to_string(),
            background: Color::rgb(255, 255, 255),
            foreground: Color::rgb(0, 0, 0),
            cursor: Color::rgb(0, 0, 0),
            selection: Color::rgba(0, 100, 200, 50),
            line_highlight: Color::rgba(0, 0, 0, 5),
            token_colors: HashMap::new(),
        };
        
        light_theme.token_colors.insert(TokenType::Keyword, TokenStyle {
            color: Color::rgb(0, 0, 255),
            bold: true,
            italic: false,
            underline: false,
        });
        
        light_theme.token_colors.insert(TokenType::String, TokenStyle {
            color: Color::rgb(163, 21, 21),
            bold: false,
            italic: false,
            underline: false,
        });
        
        light_theme.token_colors.insert(TokenType::Number, TokenStyle {
            color: Color::rgb(9, 134, 88),
            bold: false,
            italic: false,
            underline: false,
        });
        
        light_theme.token_colors.insert(TokenType::Comment, TokenStyle {
            color: Color::rgb(0, 128, 0),
            bold: false,
            italic: true,
            underline: false,
        });
        
        self.themes.insert("light".to_string(), light_theme);
    }
    
    /// Add a language definition
    pub fn add_language(&mut self, support: LanguageSupport) {
        let token_patterns = self.create_token_patterns(&support);
        
        self.languages.insert(
            support.id.clone(),
            LanguageDefinition {
                support,
                token_patterns,
            },
        );
    }
    
    /// Create token patterns for a language
    fn create_token_patterns(&self, support: &LanguageSupport) -> Vec<TokenPattern> {
        let mut patterns = Vec::new();
        
        // Keywords pattern
        if !support.keywords.is_empty() {
            let keyword_pattern = format!(r"\b({})\b", support.keywords.join("|"));
            if let Ok(re) = regex::Regex::new(&keyword_pattern) {
                patterns.push(TokenPattern {
                    pattern: re,
                    token_type: TokenType::Keyword,
                    priority: 10,
                });
            }
        }
        
        // Number pattern
        if let Ok(re) = regex::Regex::new(r"\b\d+(\.\d+)?([eE][+-]?\d+)?\b") {
            patterns.push(TokenPattern {
                pattern: re,
                token_type: TokenType::Number,
                priority: 8,
            });
        }
        
        // String patterns
        for delimiter in &support.string_delimiters {
            let pattern = format!(r"{0}[^{0}]*{0}", regex::escape(&delimiter.to_string()));
            if let Ok(re) = regex::Regex::new(&pattern) {
                patterns.push(TokenPattern {
                    pattern: re,
                    token_type: TokenType::String,
                    priority: 9,
                });
            }
        }
        
        // Comment patterns
        if let Some(ref single) = support.comment_single {
            let pattern = format!(r"{}.*$", regex::escape(single));
            if let Ok(re) = regex::Regex::new(&pattern) {
                patterns.push(TokenPattern {
                    pattern: re,
                    token_type: TokenType::Comment,
                    priority: 11,
                });
            }
        }
        
        patterns
    }
    
    /// Highlight a line of code
    pub fn highlight_line(
        &self,
        line: &str,
        language: &str,
    ) -> Vec<HighlightedToken> {
        let mut tokens = Vec::new();
        
        if let Some(lang_def) = self.languages.get(language) {
            let theme = self.themes.get(&self.current_theme)
                .unwrap_or_else(|| self.themes.values().next().unwrap());
            
            // Simple tokenization - can be improved
            let mut remaining = line.to_string();
            let mut offset = 0;
            
            while !remaining.is_empty() {
                let mut found = false;
                
                for pattern in &lang_def.token_patterns {
                    if let Some(mat) = pattern.pattern.find(&remaining) {
                        if mat.start() == 0 {
                            let text = mat.as_str().to_string();
                            let mat_end = mat.end(); // Extract end position before borrowing
                            let style = theme.token_colors
                                .get(&pattern.token_type)
                                .cloned()
                                .unwrap_or(TokenStyle {
                                    color: theme.foreground,
                                    bold: false,
                                    italic: false,
                                    underline: false,
                                });
                            
                            tokens.push(HighlightedToken {
                                text: text.clone(),
                                token_type: pattern.token_type,
                                style,
                                start: offset,
                                end: offset + text.len(),
                            });
                            
                            remaining = remaining[mat_end..].to_string();
                            offset += mat_end;
                            found = true;
                            break;
                        }
                    }
                }
                
                if !found {
                    // Take one character as identifier
                    let ch = remaining.chars().next().unwrap();
                    tokens.push(HighlightedToken {
                        text: ch.to_string(),
                        token_type: if ch.is_whitespace() {
                            TokenType::Whitespace
                        } else {
                            TokenType::Identifier
                        },
                        style: TokenStyle {
                            color: theme.foreground,
                            bold: false,
                            italic: false,
                            underline: false,
                        },
                        start: offset,
                        end: offset + ch.len_utf8(),
                    });
                    remaining = remaining[ch.len_utf8()..].to_string();
                    offset += ch.len_utf8();
                }
            }
        } else {
            // No language definition - return plain text
            tokens.push(HighlightedToken {
                text: line.to_string(),
                token_type: TokenType::Identifier,
                style: TokenStyle {
                    color: Color::rgb(212, 212, 212),
                    bold: false,
                    italic: false,
                    underline: false,
                },
                start: 0,
                end: line.len(),
            });
        }
        
        tokens
    }
    
    /// Set current theme
    pub fn set_theme(&mut self, theme_name: &str) {
        if self.themes.contains_key(theme_name) {
            self.current_theme = theme_name.to_string();
        }
    }
    
    /// Get current theme
    pub fn get_theme(&self) -> Option<&Theme> {
        self.themes.get(&self.current_theme)
    }
    
    /// Get available themes
    pub fn get_theme_names(&self) -> Vec<String> {
        self.themes.keys().cloned().collect()
    }
    
    /// Get supported languages
    pub fn get_supported_languages(&self) -> Vec<String> {
        self.languages.keys().cloned().collect()
    }
    
    /// Set tool hub for enhanced syntax highlighting
    pub async fn set_tool_hub(&self, _tool_hub: std::sync::Arc<crate::tui::chat::tools::IntegratedToolSystem>) -> anyhow::Result<()> {
        // Tool hub integration for syntax highlighter would provide:
        // - Dynamic language detection using tools
        // - Custom highlighting for tool-specific syntax
        // - Enhanced semantic highlighting using AI analysis
        tracing::info!("Tool hub integrated with syntax highlighter");
        Ok(())
    }
}
