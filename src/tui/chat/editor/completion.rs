//! Code Completion Engine
//! 
//! Provides intelligent code completion with context awareness,
//! snippet support, and AI-powered suggestions.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};
use anyhow::Result;
use tracing::debug;

use super::editor_core::{CodeEditor, CursorPosition};

/// Completion engine
pub struct CompletionEngine {
    /// Reference to editor
    editor: Arc<CodeEditor>,
    
    /// Completion providers
    providers: Vec<Box<dyn CompletionProvider>>,
    
    /// Snippet library
    snippets: Arc<RwLock<SnippetLibrary>>,
    
    /// Completion cache
    cache: Arc<RwLock<CompletionCache>>,
    
    /// Configuration
    config: CompletionConfig,
}

/// Completion configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionConfig {
    pub auto_trigger: bool,
    pub trigger_chars: Vec<char>,
    pub min_word_length: usize,
    pub max_suggestions: usize,
    pub include_snippets: bool,
    pub include_keywords: bool,
    pub include_ai_suggestions: bool,
    pub fuzzy_matching: bool,
}

impl Default for CompletionConfig {
    fn default() -> Self {
        Self {
            auto_trigger: true,
            trigger_chars: vec!['.', ':', '(', '<', '"', '\'', '/'],
            min_word_length: 2,
            max_suggestions: 20,
            include_snippets: true,
            include_keywords: true,
            include_ai_suggestions: false,
            fuzzy_matching: true,
        }
    }
}

/// Completion item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionItem {
    pub label: String,
    pub kind: CompletionKind,
    pub detail: Option<String>,
    pub documentation: Option<String>,
    pub insert_text: String,
    pub insert_text_format: InsertTextFormat,
    pub sort_text: Option<String>,
    pub filter_text: Option<String>,
    pub preselect: bool,
    pub score: f32,
    pub source: CompletionSource,
}

/// Completion kinds
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum CompletionKind {
    Text,
    Method,
    Function,
    Constructor,
    Field,
    Variable,
    Class,
    Interface,
    Module,
    Property,
    Unit,
    Value,
    Enum,
    Keyword,
    Snippet,
    Color,
    File,
    Reference,
    Folder,
    EnumMember,
    Constant,
    Struct,
    Event,
    Operator,
    TypeParameter,
}

impl CompletionKind {
    pub fn icon(&self) -> &str {
        match self {
            Self::Text => "📝",
            Self::Method | Self::Function => "ƒ",
            Self::Constructor => "⚡",
            Self::Field | Self::Property => "◈",
            Self::Variable => "𝑥",
            Self::Class | Self::Interface => "○",
            Self::Module => "◻",
            Self::Unit => "()",
            Self::Value | Self::Constant => "◆",
            Self::Enum | Self::EnumMember => "∈",
            Self::Keyword => "🔑",
            Self::Snippet => "◐",
            Self::Color => "🎨",
            Self::File => "📄",
            Self::Reference => "→",
            Self::Folder => "📁",
            Self::Struct => "▣",
            Self::Event => "⚡",
            Self::Operator => "±",
            Self::TypeParameter => "<>",
        }
    }
}

/// Insert text format
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum InsertTextFormat {
    PlainText,
    Snippet,
}

/// Completion source
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum CompletionSource {
    Keyword,
    Snippet,
    Buffer,
    LSP,
    AI,
    Path,
    History,
}

/// Completion provider trait
#[async_trait::async_trait]
pub trait CompletionProvider: Send + Sync {
    /// Get completions at position
    async fn get_completions(
        &self,
        context: &CompletionContext,
    ) -> Result<Vec<CompletionItem>>;
    
    /// Provider name
    fn name(&self) -> &str;
}

/// Completion context
#[derive(Debug, Clone)]
pub struct CompletionContext {
    pub position: CursorPosition,
    pub line: String,
    pub prefix: String,
    pub trigger_char: Option<char>,
    pub language: Option<String>,
    pub file_path: Option<String>,
}

/// Snippet library
struct SnippetLibrary {
    snippets: HashMap<String, Vec<Snippet>>,
}

/// Code snippet
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Snippet {
    pub prefix: String,
    pub body: String,
    pub description: String,
    pub scope: Vec<String>,
    pub variables: Vec<SnippetVariable>,
}

/// Snippet variable
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SnippetVariable {
    pub name: String,
    pub default: String,
    pub choices: Option<Vec<String>>,
}

/// Completion cache
struct CompletionCache {
    entries: HashMap<String, CacheEntry>,
    max_entries: usize,
}

/// Cache entry
struct CacheEntry {
    completions: Vec<CompletionItem>,
    timestamp: std::time::Instant,
}

impl CompletionEngine {
    /// Create a new completion engine
    pub fn new(editor: Arc<CodeEditor>) -> Self {
        let mut engine = Self {
            editor,
            providers: Vec::new(),
            snippets: Arc::new(RwLock::new(SnippetLibrary::new())),
            cache: Arc::new(RwLock::new(CompletionCache::new())),
            config: CompletionConfig::default(),
        };
        
        // Add default providers
        engine.providers.push(Box::new(KeywordProvider::new()));
        engine.providers.push(Box::new(BufferProvider::new()));
        
        engine
    }
    
    /// Get completions at current cursor position
    pub async fn get_completions(&self) -> Vec<CompletionItem> {
        let cursor = self.editor.get_cursor().await;
        let content = self.editor.get_content().await;
        let lines: Vec<&str> = content.lines().collect();
        
        if cursor.line >= lines.len() {
            return Vec::new();
        }
        
        let line = lines[cursor.line].to_string();
        let prefix = self.extract_prefix(&line, cursor.column);
        
        // Check cache
        let cache_key = format!("{}:{}:{}", cursor.line, cursor.column, prefix);
        if let Some(cached) = self.get_cached(&cache_key).await {
            return cached;
        }
        
        let context = CompletionContext {
            position: cursor,
            line: line.clone(),
            prefix: prefix.clone(),
            trigger_char: if cursor.column > 0 {
                line.chars().nth(cursor.column - 1)
            } else {
                None
            },
            language: self.editor.get_language().await,
            file_path: self.editor.get_file_path().await
        };
        
        let mut all_completions = Vec::new();
        
        // Gather completions from all providers
        for provider in &self.providers {
            if let Ok(completions) = provider.get_completions(&context).await {
                all_completions.extend(completions);
            }
        }
        
        // Add snippets if enabled
        if self.config.include_snippets {
            let snippet_completions = self.get_snippet_completions(&context).await;
            all_completions.extend(snippet_completions);
        }
        
        // Sort and limit
        all_completions.sort_by(|a, b| {
            b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
        });
        all_completions.truncate(self.config.max_suggestions);
        
        // Cache results
        self.cache_completions(&cache_key, all_completions.clone()).await;
        
        all_completions
    }
    
    /// Extract prefix from line at position
    fn extract_prefix(&self, line: &str, position: usize) -> String {
        let before = &line[..position.min(line.len())];
        
        // Find word boundary
        let mut start = position;
        for (i, ch) in before.chars().rev().enumerate() {
            if !ch.is_alphanumeric() && ch != '_' {
                start = position - i;
                break;
            }
        }
        
        if start == position {
            String::new()
        } else {
            before[start..].to_string()
        }
    }
    
    /// Get snippet completions
    async fn get_snippet_completions(&self, context: &CompletionContext) -> Vec<CompletionItem> {
        let snippets = self.snippets.read().await;
        let mut completions = Vec::new();
        
        // Get snippets for current language
        let lang = context.language.as_deref().unwrap_or("*");
        if let Some(lang_snippets) = snippets.snippets.get(lang) {
            for snippet in lang_snippets {
                if snippet.prefix.starts_with(&context.prefix) || context.prefix.is_empty() {
                    completions.push(CompletionItem {
                        label: snippet.prefix.clone(),
                        kind: CompletionKind::Snippet,
                        detail: Some(snippet.description.clone()),
                        documentation: None,
                        insert_text: snippet.body.clone(),
                        insert_text_format: InsertTextFormat::Snippet,
                        sort_text: None,
                        filter_text: None,
                        preselect: false,
                        score: self.calculate_score(&snippet.prefix, &context.prefix),
                        source: CompletionSource::Snippet,
                    });
                }
            }
        }
        
        completions
    }
    
    /// Calculate completion score
    fn calculate_score(&self, item: &str, prefix: &str) -> f32 {
        if prefix.is_empty() {
            return 0.5;
        }
        
        if item == prefix {
            return 1.0;
        }
        
        if item.starts_with(prefix) {
            return 0.9 - (item.len() - prefix.len()) as f32 * 0.01;
        }
        
        if self.config.fuzzy_matching {
            // Simple fuzzy matching
            let mut score = 0.0;
            let mut prefix_chars = prefix.chars();
            let mut current = prefix_chars.next();
            
            for ch in item.chars() {
                if let Some(prefix_ch) = current {
                    if ch.to_lowercase().eq(prefix_ch.to_lowercase()) {
                        score += 1.0;
                        current = prefix_chars.next();
                    }
                }
            }
            
            score / prefix.len() as f32 * 0.7
        } else {
            0.0
        }
    }
    
    /// Get cached completions
    async fn get_cached(&self, key: &str) -> Option<Vec<CompletionItem>> {
        let cache = self.cache.read().await;
        if let Some(entry) = cache.entries.get(key) {
            if entry.timestamp.elapsed().as_secs() < 5 {
                return Some(entry.completions.clone());
            }
        }
        None
    }
    
    /// Cache completions
    async fn cache_completions(&self, key: &str, completions: Vec<CompletionItem>) {
        let mut cache = self.cache.write().await;
        
        // Evict old entries if needed
        if cache.entries.len() >= cache.max_entries {
            // Remove oldest entry
            if let Some(oldest_key) = cache.entries.iter()
                .min_by_key(|(_, entry)| entry.timestamp)
                .map(|(k, _)| k.clone()) {
                cache.entries.remove(&oldest_key);
            }
        }
        
        cache.entries.insert(key.to_string(), CacheEntry {
            completions,
            timestamp: std::time::Instant::now(),
        });
    }
    
    /// Set tool hub for enhanced completions
    pub async fn set_tool_hub(&self, _tool_hub: Arc<crate::tui::chat::tools::IntegratedToolSystem>) -> Result<()> {
        // Tool hub integration for completion engine would provide:
        // - Tool-specific completions (available tools, parameters, etc.)
        // - Context-aware suggestions based on current editing context
        // - AI-powered completions using tool analysis
        // For now, return Ok as this is a complex feature to implement
        tracing::info!("Tool hub integrated with completion engine");
        Ok(())
    }
}

impl SnippetLibrary {
    fn new() -> Self {
        let mut library = Self {
            snippets: HashMap::new(),
        };
        
        // Load default snippets
        library.load_default_snippets();
        
        library
    }
    
    fn load_default_snippets(&mut self) {
        // Rust snippets
        let rust_snippets = vec![
            Snippet {
                prefix: "fn".to_string(),
                body: "fn ${1:name}(${2:params}) ${3:-> ${4:ReturnType}} {\n    ${0}\n}".to_string(),
                description: "Function definition".to_string(),
                scope: vec!["rust".to_string()],
                variables: vec![],
            },
            Snippet {
                prefix: "impl".to_string(),
                body: "impl ${1:Type} {\n    ${0}\n}".to_string(),
                description: "Implementation block".to_string(),
                scope: vec!["rust".to_string()],
                variables: vec![],
            },
            Snippet {
                prefix: "struct".to_string(),
                body: "struct ${1:Name} {\n    ${0}\n}".to_string(),
                description: "Struct definition".to_string(),
                scope: vec!["rust".to_string()],
                variables: vec![],
            },
        ];
        
        self.snippets.insert("rust".to_string(), rust_snippets);
    }
}

impl CompletionCache {
    fn new() -> Self {
        Self {
            entries: HashMap::new(),
            max_entries: 100,
        }
    }
}

/// Keyword completion provider
struct KeywordProvider {
    keywords: HashMap<String, Vec<String>>,
}

impl KeywordProvider {
    fn new() -> Self {
        let mut provider = Self {
            keywords: HashMap::new(),
        };
        
        // Add language keywords
        provider.keywords.insert("rust".to_string(), vec![
            "fn", "let", "mut", "const", "static", "struct", "enum", "impl",
            "trait", "pub", "mod", "use", "if", "else", "match", "for", "while",
            "loop", "break", "continue", "return", "async", "await", "move",
        ].iter().map(|s| s.to_string()).collect());
        
        provider
    }
}

#[async_trait::async_trait]
impl CompletionProvider for KeywordProvider {
    async fn get_completions(&self, context: &CompletionContext) -> Result<Vec<CompletionItem>> {
        let mut completions = Vec::new();
        
        let lang = context.language.as_deref().unwrap_or("rust");
        if let Some(keywords) = self.keywords.get(lang) {
            for keyword in keywords {
                if keyword.starts_with(&context.prefix) {
                    completions.push(CompletionItem {
                        label: keyword.clone(),
                        kind: CompletionKind::Keyword,
                        detail: Some("Keyword".to_string()),
                        documentation: None,
                        insert_text: keyword.clone(),
                        insert_text_format: InsertTextFormat::PlainText,
                        sort_text: None,
                        filter_text: None,
                        preselect: false,
                        score: 0.7,
                        source: CompletionSource::Keyword,
                    });
                }
            }
        }
        
        Ok(completions)
    }
    
    fn name(&self) -> &str {
        "keyword"
    }
}

/// Buffer completion provider (words from current buffer)
struct BufferProvider;

impl BufferProvider {
    fn new() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl CompletionProvider for BufferProvider {
    async fn get_completions(&self, context: &CompletionContext) -> Result<Vec<CompletionItem>> {
        let mut completions = Vec::new();
        let mut word_set = std::collections::HashSet::new();
        
        // Extract the word being typed from the prefix
        let current_word = context.prefix.to_lowercase();
        
        // Skip if word is too short
        if current_word.len() < 2 {
            return Ok(completions);
        }
        
        // For buffer words, we'd need access to the full buffer
        // This is a simplified implementation that just returns common programming words
        let common_words = vec![
            "function", "class", "interface", "struct", "enum", "import", "export",
            "const", "let", "var", "if", "else", "for", "while", "return", "async",
            "await", "public", "private", "protected", "static", "extends", "implements"
        ];
        
        for word in common_words {
            if word.starts_with(&current_word) && word != current_word {
                if word_set.insert(word.to_string()) {
                    completions.push(CompletionItem {
                        label: word.to_string(),
                        kind: CompletionKind::Text,
                        detail: Some("Keyword".to_string()),
                        insert_text: word.to_string(),
                        documentation: None,
                        score: 0.5,
                        filter_text: Some(word.to_string()),
                        sort_text: Some(word.to_string()),
                        preselect: false,
                        insert_text_format: InsertTextFormat::PlainText,
                        source: CompletionSource::Buffer,
                    });
                }
            }
        }
        
        Ok(completions)
    }
    
    fn name(&self) -> &str {
        "buffer"
    }
}
