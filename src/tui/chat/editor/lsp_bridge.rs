//! LSP Bridge Module
//! 
//! Provides Language Server Protocol integration for the code editor.

use std::sync::Arc;
use tokio::sync::RwLock;
use anyhow::Result;
use serde::{Serialize, Deserialize};

/// LSP Bridge for language server integration
pub struct LSPBridge {
    capabilities: LSPCapabilities,
    diagnostics: Arc<RwLock<Vec<Diagnostic>>>,
    completions_cache: Arc<RwLock<Vec<CompletionItem>>>,
    current_language: Arc<RwLock<Option<String>>>,
    tool_hub: Arc<RwLock<Option<Arc<crate::tui::chat::tools::IntegratedToolSystem>>>>,
}

/// LSP capabilities
#[derive(Debug, Clone)]
pub struct LSPCapabilities {
    pub completion: bool,
    pub hover: bool,
    pub signature_help: bool,
    pub goto_definition: bool,
    pub formatting: bool,
    pub diagnostics: bool,
}

impl Default for LSPCapabilities {
    fn default() -> Self {
        Self {
            completion: true,
            hover: true,
            signature_help: true,
            goto_definition: true,
            formatting: true,
            diagnostics: true,
        }
    }
}

/// Diagnostic severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagnosticSeverity {
    Error,
    Warning,
    Information,
    Hint,
}

/// Diagnostic information
#[derive(Debug, Clone)]
pub struct Diagnostic {
    pub severity: DiagnosticSeverity,
    pub message: String,
    pub line: usize,
    pub column: usize,
    pub source: String,
}

/// Completion item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionItem {
    pub label: String,
    pub kind: CompletionKind,
    pub detail: Option<String>,
    pub documentation: Option<String>,
    pub insert_text: String,
}

/// Completion kind
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
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

impl LSPBridge {
    /// Create a new LSP bridge
    pub async fn new() -> Result<Self> {
        Ok(Self {
            capabilities: LSPCapabilities::default(),
            diagnostics: Arc::new(RwLock::new(Vec::new())),
            completions_cache: Arc::new(RwLock::new(Vec::new())),
            current_language: Arc::new(RwLock::new(None)),
            tool_hub: Arc::new(RwLock::new(None)),
        })
    }
    
    /// Get completions at current position
    pub async fn get_completions(&self) -> Result<Vec<CompletionItem>> {
        // Check if we have cached completions
        let cached = self.completions_cache.read().await;
        if !cached.is_empty() {
            return Ok(cached.clone());
        }
        drop(cached);
        
        // Generate completions based on current language
        let language = self.current_language.read().await.clone();
        let completions = match language.as_deref() {
            Some("rust") => self.get_rust_completions().await,
            Some("python") => self.get_python_completions().await,
            Some("javascript") | Some("typescript") => self.get_js_completions().await,
            _ => self.get_generic_completions().await,
        };
        
        // Cache the completions
        *self.completions_cache.write().await = completions.clone();
        Ok(completions)
    }
    
    async fn get_rust_completions(&self) -> Vec<CompletionItem> {
        vec![
            CompletionItem {
                label: "println!".to_string(),
                kind: CompletionKind::Snippet,
                detail: Some("Print line macro".to_string()),
                documentation: Some("Prints to stdout with newline".to_string()),
                insert_text: "println!(\"{}\")".to_string(),
            },
            CompletionItem {
                label: "vec!".to_string(),
                kind: CompletionKind::Snippet,
                detail: Some("Vector macro".to_string()),
                documentation: Some("Creates a Vec<T> containing the arguments".to_string()),
                insert_text: "vec![$0]".to_string(),
            },
            CompletionItem {
                label: "async".to_string(),
                kind: CompletionKind::Keyword,
                detail: Some("Async function".to_string()),
                documentation: Some("Defines an asynchronous function".to_string()),
                insert_text: "async ".to_string(),
            },
            CompletionItem {
                label: "impl".to_string(),
                kind: CompletionKind::Keyword,
                detail: Some("Implementation block".to_string()),
                documentation: Some("Implements methods for a type".to_string()),
                insert_text: "impl ".to_string(),
            },
        ]
    }
    
    async fn get_python_completions(&self) -> Vec<CompletionItem> {
        vec![
            CompletionItem {
                label: "print".to_string(),
                kind: CompletionKind::Function,
                detail: Some("Print function".to_string()),
                documentation: Some("Prints to stdout".to_string()),
                insert_text: "print($0)".to_string(),
            },
            CompletionItem {
                label: "def".to_string(),
                kind: CompletionKind::Keyword,
                detail: Some("Function definition".to_string()),
                documentation: Some("Defines a function".to_string()),
                insert_text: "def $0():".to_string(),
            },
            CompletionItem {
                label: "class".to_string(),
                kind: CompletionKind::Keyword,
                detail: Some("Class definition".to_string()),
                documentation: Some("Defines a class".to_string()),
                insert_text: "class $0:".to_string(),
            },
        ]
    }
    
    async fn get_js_completions(&self) -> Vec<CompletionItem> {
        vec![
            CompletionItem {
                label: "console.log".to_string(),
                kind: CompletionKind::Method,
                detail: Some("Console logging".to_string()),
                documentation: Some("Logs to console".to_string()),
                insert_text: "console.log($0)".to_string(),
            },
            CompletionItem {
                label: "function".to_string(),
                kind: CompletionKind::Keyword,
                detail: Some("Function declaration".to_string()),
                documentation: Some("Declares a function".to_string()),
                insert_text: "function $0() {".to_string(),
            },
            CompletionItem {
                label: "const".to_string(),
                kind: CompletionKind::Keyword,
                detail: Some("Constant declaration".to_string()),
                documentation: Some("Declares a constant".to_string()),
                insert_text: "const $0 = ".to_string(),
            },
        ]
    }
    
    async fn get_generic_completions(&self) -> Vec<CompletionItem> {
        vec![
            CompletionItem {
                label: "TODO".to_string(),
                kind: CompletionKind::Snippet,
                detail: Some("TODO comment".to_string()),
                documentation: Some("Adds a TODO comment".to_string()),
                insert_text: "// TODO: $0".to_string(),
            },
        ]
    }
    
    /// Get diagnostics
    pub async fn get_diagnostics(&self) -> Result<Vec<Diagnostic>> {
        Ok(self.diagnostics.read().await.clone())
    }
    
    /// Format document
    pub async fn format_document(&self, content: &str) -> Result<String> {
        let language = self.current_language.read().await.clone();
        
        // Basic formatting based on language
        match language.as_deref() {
            Some("rust") => {
                // Simple Rust formatting
                let formatted = content
                    .lines()
                    .map(|line| {
                        // Basic indentation for blocks
                        if line.trim().ends_with('{') {
                            format!("{}\n", line)
                        } else if line.trim().starts_with('}') {
                            line.to_string()
                        } else {
                            line.to_string()
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                Ok(formatted)
            }
            Some("python") => {
                // Simple Python formatting
                let formatted = content
                    .lines()
                    .map(|line| {
                        // Ensure consistent indentation
                        if line.trim().ends_with(':') {
                            format!("{}\n", line)
                        } else {
                            line.to_string()
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                Ok(formatted)
            }
            _ => Ok(content.to_string()),
        }
    }
    
    /// Notify of a change
    pub async fn notify_change(&self, action: &super::EditAction) -> Result<()> {
        use super::EditAction;
        
        // Clear completions cache on change
        self.completions_cache.write().await.clear();
        
        // Update diagnostics based on action
        match action {
            EditAction::Insert { text, .. } => {
                // Check for common issues
                if text.contains("unsafe") {
                    self.add_diagnostic(
                        DiagnosticSeverity::Warning,
                        "Use of unsafe code detected".to_string(),
                        0, 0, // Would need position tracking
                    ).await;
                }
            }
            EditAction::Delete { .. } => {
                // Clear diagnostics on delete
                self.diagnostics.write().await.clear();
            }
            _ => {}
        }
        
        Ok(())
    }
    
    async fn add_diagnostic(&self, severity: DiagnosticSeverity, message: String, line: usize, column: usize) {
        self.diagnostics.write().await.push(Diagnostic {
            severity,
            message,
            line,
            column,
            source: "lsp-bridge".to_string(),
        });
    }
    
    /// Set tool hub for enhanced LSP features
    pub async fn set_tool_hub(&self, tool_hub: Arc<crate::tui::chat::tools::IntegratedToolSystem>) -> Result<()> {
        *self.tool_hub.write().await = Some(tool_hub.clone());
        
        // With tool hub integration, we can provide:
        // - Tool-aware completions (e.g., available tool commands)
        // - Enhanced diagnostics using tool analysis
        // - Code actions powered by available tools
        
        // Add tool-specific completions
        let mut completions = self.completions_cache.write().await;
        completions.push(CompletionItem {
            label: "@tool".to_string(),
            kind: CompletionKind::Snippet,
            detail: Some("Execute tool command".to_string()),
            documentation: Some("Executes a tool from the integrated tool system".to_string()),
            insert_text: "@tool $0".to_string(),
        });
        
        tracing::info!("Tool hub integrated with LSP bridge - enhanced features enabled");
        Ok(())
    }
    
    /// Set current language for context-aware features
    pub async fn set_language(&self, language: String) -> Result<()> {
        *self.current_language.write().await = Some(language);
        // Clear completions to regenerate for new language
        self.completions_cache.write().await.clear();
        Ok(())
    }
}