//! Code Editor Integration Module
//! 
//! Provides comprehensive code editing capabilities integrated with the chat interface,
//! including syntax highlighting, code completion, and collaborative editing.

pub mod editor_core;
pub mod syntax;
pub mod completion;
pub mod lsp_bridge;
pub mod collaboration;
pub mod monaco_features;

// Re-export commonly used types
pub use editor_core::{CodeEditor, EditorState, EditorConfig, EditAction, ExecutionResult, SelectionRange};
pub use syntax::{SyntaxHighlighter, LanguageSupport, Theme};
pub use completion::{CompletionEngine, CompletionItem, CompletionKind, InsertTextFormat, CompletionSource};
pub use lsp_bridge::{LSPBridge, LSPCapabilities, DiagnosticSeverity};
pub use collaboration::{CollaborativeSession, EditOperation, CursorPosition};
pub use monaco_features::{MonacoFeatures, IntelliSenseEngine, SnippetManager, MultiCursorManager};

use std::sync::Arc;
use anyhow::Result;
use tracing::info;

/// Integrated code editor system
pub struct IntegratedEditor {
    pub editor: Arc<CodeEditor>,
    pub syntax: Arc<SyntaxHighlighter>,
    pub completion: Arc<CompletionEngine>,
    pub lsp: Option<Arc<LSPBridge>>,
    pub collab: Option<Arc<CollaborativeSession>>,
    pub monaco: Arc<MonacoFeatures>,
}

impl IntegratedEditor {
    /// Create a new integrated editor
    pub async fn new(config: EditorConfig) -> Result<Self> {
        info!("Initializing integrated code editor");
        
        // Create core editor
        let editor = Arc::new(CodeEditor::new(config.clone()));
        
        // Create syntax highlighter
        let syntax = Arc::new(SyntaxHighlighter::new());
        
        // Create completion engine
        let completion = Arc::new(CompletionEngine::new(editor.clone()));
        
        // Initialize LSP if configured
        let lsp = if config.enable_lsp {
            Some(Arc::new(LSPBridge::new().await?))
        } else {
            None
        };
        
        // Initialize collaboration if configured
        let collab = if config.enable_collaboration {
            Some(Arc::new(CollaborativeSession::new(editor.clone()).await?))
        } else {
            None
        };
        
        // Initialize Monaco features
        let monaco = Arc::new(MonacoFeatures::new());
        monaco.initialize(editor.clone()).await?;
        
        Ok(Self {
            editor,
            syntax,
            completion,
            lsp,
            collab,
            monaco,
        })
    }
    
    /// Open a file in the editor
    pub async fn open_file(&self, path: &str) -> Result<()> {
        self.editor.open_file(path).await
    }
    
    /// Get current file content
    pub async fn get_content(&self) -> String {
        self.editor.get_content().await
    }
    
    /// Apply an edit
    pub async fn apply_edit(&self, action: EditAction) -> Result<()> {
        // Apply to editor
        self.editor.apply_edit(action.clone()).await?;
        
        // Notify LSP if connected
        if let Some(lsp) = &self.lsp {
            lsp.notify_change(&action).await?;
        }
        
        // Broadcast to collaboration session if active
        if let Some(collab) = &self.collab {
            collab.broadcast_edit(&action).await?;
        }
        
        Ok(())
    }
    
    /// Get completions at cursor
    pub async fn get_completions(&self) -> Vec<CompletionItem> {
        let mut completions = self.completion.get_completions().await;
        
        // Add LSP completions if available
        if let Some(lsp) = &self.lsp {
            if let Ok(lsp_completions) = lsp.get_completions().await {
                // Convert LSP completions to editor completions
                let converted_completions = lsp_completions.into_iter().map(|lsp_item| CompletionItem {
                    label: lsp_item.label,
                    kind: Self::convert_completion_kind(lsp_item.kind),
                    detail: lsp_item.detail,
                    documentation: lsp_item.documentation,
                    insert_text: lsp_item.insert_text,
                    insert_text_format: InsertTextFormat::PlainText, // Default format
                    sort_text: None,
                    filter_text: None,
                    preselect: false,
                    score: 0.8, // Default score for LSP completions
                    source: CompletionSource::LSP,
                }).collect::<Vec<_>>();
                completions.extend(converted_completions);
            }
        }
        
        completions
    }
    
    /// Get diagnostics
    pub async fn get_diagnostics(&self) -> Vec<Diagnostic> {
        if let Some(lsp) = &self.lsp {
            let lsp_diagnostics = lsp.get_diagnostics().await.unwrap_or_default();
            // Convert LSP diagnostics to editor diagnostics
            lsp_diagnostics.into_iter().map(|d| Diagnostic {
                severity: d.severity,
                message: d.message,
                line: d.line,
                column: d.column,
                source: d.source,
            }).collect()
        } else {
            Vec::new()
        }
    }
    
    /// Format document
    pub async fn format(&self) -> Result<()> {
        if let Some(lsp) = &self.lsp {
            let content = self.editor.get_content().await;
            let formatted = lsp.format_document(&content).await?;
            self.editor.set_content(formatted).await?;
        } else {
            // Use built-in formatter
            self.editor.format().await?;
        }
        Ok(())
    }

    /// Set tool hub for integrated features
    pub async fn set_tool_hub(&self, tool_hub: Arc<crate::tui::chat::tools::IntegratedToolSystem>) -> Result<()> {
        // Integrate tool hub with completion engine for tool-aware completions
        if let Err(e) = self.completion.set_tool_hub(tool_hub.clone()).await {
            tracing::warn!("Failed to set tool hub for completion engine: {}", e);
        }
        
        // Integrate with syntax highlighter for tool-specific syntax
        if let Err(e) = self.syntax.set_tool_hub(tool_hub.clone()).await {
            tracing::warn!("Failed to set tool hub for syntax highlighter: {}", e);
        }
        
        // If LSP bridge is available, integrate tools for enhanced language support
        if let Some(lsp) = &self.lsp {
            if let Err(e) = lsp.set_tool_hub(tool_hub.clone()).await {
                tracing::warn!("Failed to set tool hub for LSP bridge: {}", e);
            }
        }
        
        // If collaborative session is active, integrate tools for real-time assistance
        if let Some(collab) = &self.collab {
            if let Err(e) = collab.set_tool_hub(tool_hub).await {
                tracing::warn!("Failed to set tool hub for collaborative session: {}", e);
            }
        }
        
        tracing::info!("Tool hub successfully integrated with editor components");
        Ok(())
    }
    
    /// Convert LSP CompletionKind to editor CompletionKind
    fn convert_completion_kind(lsp_kind: lsp_bridge::CompletionKind) -> CompletionKind {
        use lsp_bridge::CompletionKind as LspKind;
        
        match lsp_kind {
            LspKind::Text => CompletionKind::Text,
            LspKind::Method => CompletionKind::Method,
            LspKind::Function => CompletionKind::Function,
            LspKind::Constructor => CompletionKind::Constructor,
            LspKind::Field => CompletionKind::Field,
            LspKind::Variable => CompletionKind::Variable,
            LspKind::Class => CompletionKind::Class,
            LspKind::Interface => CompletionKind::Interface,
            LspKind::Module => CompletionKind::Module,
            LspKind::Property => CompletionKind::Property,
            LspKind::Unit => CompletionKind::Unit,
            LspKind::Value => CompletionKind::Value,
            LspKind::Enum => CompletionKind::Enum,
            LspKind::Keyword => CompletionKind::Keyword,
            LspKind::Snippet => CompletionKind::Snippet,
            LspKind::Color => CompletionKind::Color,
            LspKind::File => CompletionKind::File,
            LspKind::Reference => CompletionKind::Reference,
            LspKind::Folder => CompletionKind::Folder,
            LspKind::EnumMember => CompletionKind::EnumMember,
            LspKind::Constant => CompletionKind::Constant,
            LspKind::Struct => CompletionKind::Struct,
            LspKind::Event => CompletionKind::Event,
            LspKind::Operator => CompletionKind::Operator,
            LspKind::TypeParameter => CompletionKind::TypeParameter,
        }
    }
    
    /// Process a request for editor operations
    pub async fn process_request(&self, data: serde_json::Value) -> Result<serde_json::Value> {
        let request_type = data.get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("edit");
        
        match request_type {
            "edit" => {
                // Apply an edit to the current file
                if let Ok(action) = serde_json::from_value::<EditAction>(data.get("action").cloned().unwrap_or_default()) {
                    self.apply_edit(action).await?;
                    Ok(serde_json::json!({
                        "status": "applied",
                        "content": self.get_content().await
                    }))
                } else {
                    Err(anyhow::anyhow!("Invalid edit action"))
                }
            },
            "open" => {
                // Open a file
                let path = data.get("path")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("Missing file path"))?;
                
                self.open_file(path).await?;
                Ok(serde_json::json!({
                    "status": "opened",
                    "path": path,
                    "content": self.get_content().await
                }))
            },
            "complete" => {
                // Get completions at current position
                let completions = self.get_completions().await;
                Ok(serde_json::to_value(completions)?)
            },
            "highlight" => {
                // Get syntax highlighting
                let content = self.get_content().await;
                let language = data.get("language")
                    .and_then(|v| v.as_str())
                    .unwrap_or("javascript");
                
                // Apply syntax highlighting
                let mut highlighted_lines = Vec::new();
                for line in content.lines() {
                    let tokens = self.syntax.highlight_line(line, language);
                    highlighted_lines.push(serde_json::json!({
                        "text": line,
                        "tokens": tokens.iter().map(|t| serde_json::json!({
                            "text": t.text,
                            "type": format!("{:?}", t.token_type),
                            "style": {
                                "color": format!("#{:02x}{:02x}{:02x}", t.style.color.r, t.style.color.g, t.style.color.b),
                                "bold": t.style.bold,
                                "italic": t.style.italic,
                                "underline": t.style.underline,
                            }
                        })).collect::<Vec<_>>()
                    }));
                }
                
                Ok(serde_json::json!({
                    "content": content,
                    "language": language,
                    "highlighted": true,
                    "lines": highlighted_lines,
                    "theme": self.syntax.get_theme().map(|t| t.name.clone()).unwrap_or_else(|| "default".to_string())
                }))
            },
            _ => {
                Err(anyhow::anyhow!("Unknown request type: {}", request_type))
            }
        }
    }
    
    /// Apply changes to a file
    pub async fn apply_changes(&self, file: String, changes: Vec<String>) -> Result<()> {
        tracing::info!("Applying {} changes to file {}", changes.len(), file);
        
        // Open the file first
        self.open_file(&file).await?;
        
        // Apply each change as an edit action
        for (index, change) in changes.iter().enumerate() {
            let position = crate::tui::chat::editor::editor_core::CursorPosition {
                line: index,
                column: 0,
            };
            
            let action = EditAction::Insert {
                position,
                text: change.clone(),
            };
            
            self.apply_edit(action).await?;
        }
        
        Ok(())
    }
    
    /// Create a new file with content
    pub async fn create_file(&self, path: String, content: String) -> Result<()> {
        tracing::info!("Creating file {} with {} bytes of content", path, content.len());
        
        // Create and open the new file
        self.editor.create_file(&path, &content).await?;
        
        Ok(())
    }
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
