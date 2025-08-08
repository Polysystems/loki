//! Editor Bridge for seamless chat-to-editor integration
//! 
//! Provides bidirectional communication between the chat interface and code editor,
//! enabling automatic code flow, agent coordination, and real-time synchronization.

use std::sync::Arc;
use std::collections::HashMap;
use tokio::sync::{RwLock, mpsc};
use anyhow::{Result, Context};
use serde::{Serialize, Deserialize};
use serde_json::Value as JsonValue;

use super::event_bridge::EventBridge;
use crate::tui::event_bus::{SystemEvent, TabId};
use crate::tui::chat::editor::{IntegratedEditor, EditorConfig, EditAction, CursorPosition, SelectionRange};

/// Editor bridge for cross-tab communication
pub struct EditorBridge {
    /// Event bridge for communication
    event_bridge: Arc<EventBridge>,
    
    /// Active editor instances by tab ID
    editors: Arc<RwLock<HashMap<String, Arc<IntegratedEditor>>>>,
    
    /// Code generation requests queue
    code_requests: Arc<RwLock<Vec<CodeGenerationRequest>>>,
    
    /// Active code sessions (chat ID -> editor session)
    active_sessions: Arc<RwLock<HashMap<String, EditorSession>>>,
    
    /// Editor state cache
    state_cache: Arc<RwLock<HashMap<String, EditorStateSnapshot>>>,
}

/// Code generation request from chat
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeGenerationRequest {
    pub id: String,
    pub chat_id: String,
    pub description: String,
    pub language: Option<String>,
    pub context: Option<String>,
    pub requirements: Vec<String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Active editor session linked to chat
#[derive(Debug, Clone)]
pub struct EditorSession {
    pub session_id: String,
    pub chat_id: String,
    pub editor_tab_id: String,
    pub current_file: Option<String>,
    pub agent_ids: Vec<String>,
    pub is_collaborative: bool,
    pub started_at: chrono::DateTime<chrono::Utc>,
}

/// Snapshot of editor state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EditorStateSnapshot {
    pub content: String,
    pub cursor_position: (usize, usize),
    pub selection: Option<(usize, usize, usize, usize)>,
    pub language: String,
    pub file_path: Option<String>,
    pub modified: bool,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Code execution request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeExecutionRequest {
    pub code: String,
    pub language: String,
    pub session_id: Option<String>,
}

/// Code execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeExecutionResult {
    pub success: bool,
    pub output: String,
    pub error: Option<String>,
    pub execution_time_ms: u64,
}

impl EditorBridge {
    /// Create a new editor bridge
    pub fn new(event_bridge: Arc<EventBridge>) -> Self {
        Self {
            event_bridge,
            editors: Arc::new(RwLock::new(HashMap::new())),
            code_requests: Arc::new(RwLock::new(Vec::new())),
            active_sessions: Arc::new(RwLock::new(HashMap::new())),
            state_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Initialize the bridge
    pub async fn initialize(&self) -> Result<()> {
        // Set up event listeners
        self.setup_event_listeners().await?;
        
        tracing::info!("Editor bridge initialized");
        Ok(())
    }
    
    /// Set up event listeners for chat-editor communication
    async fn setup_event_listeners(&self) -> Result<()> {
        let bridge = Arc::new(self.clone());
        
        // Listen for code generation requests from chat
        let bridge_clone = bridge.clone();
        self.event_bridge.on_event("code_generation_request", move |event| {
            let bridge = bridge_clone.clone();
            Box::pin(async move {
                if let SystemEvent::CustomEvent { name, data, .. } = event {
                    if name == "code_generation_request" {
                        if let Ok(request) = serde_json::from_value::<CodeGenerationRequest>(data) {
                            let _ = bridge.handle_code_generation_request(request).await;
                        }
                    }
                }
                Ok(())
            })
        }).await?;
        
        // Listen for editor state changes
        let bridge_clone = bridge.clone();
        self.event_bridge.on_event("editor_state_change", move |event| {
            let bridge = bridge_clone.clone();
            Box::pin(async move {
                if let SystemEvent::CustomEvent { name, data, .. } = event {
                    if name == "editor_state_change" {
                        if let Ok(snapshot) = serde_json::from_value::<EditorStateSnapshot>(data) {
                            let _ = bridge.handle_editor_state_change(snapshot).await;
                        }
                    }
                }
                Ok(())
            })
        }).await?;
        
        Ok(())
    }
    
    /// Request code generation and open in editor
    pub async fn request_code_generation(
        &self,
        chat_id: String,
        description: String,
        language: Option<String>,
        context: Option<String>,
    ) -> Result<String> {
        let request = CodeGenerationRequest {
            id: uuid::Uuid::new_v4().to_string(),
            chat_id: chat_id.clone(),
            description,
            language,
            context,
            requirements: Vec::new(),
            timestamp: chrono::Utc::now(),
        };
        
        // Store request
        self.code_requests.write().await.push(request.clone());
        
        // Create editor session
        let session = EditorSession {
            session_id: request.id.clone(),
            chat_id,
            editor_tab_id: format!("editor_{}", request.id),
            current_file: None,
            agent_ids: Vec::new(),
            is_collaborative: true,
            started_at: chrono::Utc::now(),
        };
        
        self.active_sessions.write().await.insert(session.session_id.clone(), session.clone());
        
        // Emit event to open editor
        self.event_bridge.emit(SystemEvent::CustomEvent {
            name: "open_editor_for_code".to_string(),
            data: serde_json::to_value(&request)?,
            source: TabId::Chat,
            target: Some(TabId::Custom("editor".to_string())),
        }).await?;
        
        Ok(session.session_id)
    }
    
    /// Handle code generation request
    async fn handle_code_generation_request(&self, request: CodeGenerationRequest) -> Result<()> {
        tracing::info!("Handling code generation request: {}", request.description);
        
        // Open editor with generated code placeholder
        let initial_content = format!(
            "// Code generation requested: {}\n// Language: {}\n// Generating...\n",
            request.description,
            request.language.as_deref().unwrap_or("auto-detect")
        );
        
        // Create or get editor instance
        let editor = self.get_or_create_editor(&request.chat_id).await?;
        
        // Set initial content
        editor.editor.set_content(initial_content).await?;
        
        // Spawn agents for code generation
        self.spawn_code_agents(&request).await?;
        
        Ok(())
    }
    
    /// Handle editor state changes
    async fn handle_editor_state_change(&self, snapshot: EditorStateSnapshot) -> Result<()> {
        // Cache the state
        if let Some(file_path) = &snapshot.file_path {
            self.state_cache.write().await.insert(file_path.clone(), snapshot.clone());
        }
        
        // Notify chat about changes if there's an active session
        for (_, session) in self.active_sessions.read().await.iter() {
            if session.current_file.as_deref() == snapshot.file_path.as_deref() {
                self.event_bridge.emit(SystemEvent::CustomEvent {
                    name: "editor_content_updated".to_string(),
                    data: serde_json::to_value(&snapshot)?,
                    source: TabId::Custom("editor".to_string()),
                    target: Some(TabId::Chat),
                }).await?;
            }
        }
        
        Ok(())
    }
    
    /// Get or create an editor instance
    async fn get_or_create_editor(&self, tab_id: &str) -> Result<Arc<IntegratedEditor>> {
        let mut editors = self.editors.write().await;
        
        if let Some(editor) = editors.get(tab_id) {
            Ok(editor.clone())
        } else {
            let config = EditorConfig::default();
            let editor = Arc::new(IntegratedEditor::new(config).await?);
            editors.insert(tab_id.to_string(), editor.clone());
            Ok(editor)
        }
    }
    
    /// Spawn agents for code generation
    async fn spawn_code_agents(&self, request: &CodeGenerationRequest) -> Result<()> {
        use crate::tui::chat::agents::code_agent::{CodeAgentFactory, CodeAgent, AgentUpdate, CodeTask, TaskPriority, ComplexityLevel};
        
        // Create update channel for agent
        let (update_tx, mut update_rx) = tokio::sync::mpsc::channel(100);
        
        // Create agent through factory
        let mut agent = CodeAgentFactory::create_agent(&request.description, update_tx);
        
        // Set editor bridge for the agent
        agent.set_editor_bridge(Arc::new(self.clone()));
        
        // Create a code task for the agent
        let task = CodeTask {
            id: request.id.clone(),
            description: request.description.clone(),
            requirements: request.requirements.clone(),
            language: request.language.clone(),
            framework: None,
            files: vec![],
            context: std::collections::HashMap::new(),
            priority: TaskPriority::Normal,
            estimated_complexity: ComplexityLevel::Moderate,
        };
        
        // Spawn agent task execution
        let agent = Arc::new(agent);
        let agent_clone = agent.clone();
        let agent_id = agent.id().to_string();
        let session_id = request.id.clone();
        
        // Add agent to session
        if let Some(session) = self.active_sessions.write().await.get_mut(&session_id) {
            session.agent_ids.push(agent_id.clone());
        }
        
        tokio::spawn(async move {
            tracing::info!("Agent {} starting task execution", agent_id);
            if let Err(e) = agent_clone.execute_task(task).await {
                tracing::error!("Agent {} task execution failed: {}", agent_id, e);
            } else {
                tracing::info!("Agent {} completed task execution", agent_id);
            }
        });
        
        // Spawn update handler to forward agent updates
        let event_bridge = self.event_bridge.clone();
        let agent_name = agent.name().to_string();
        tokio::spawn(async move {
            while let Some(update) = update_rx.recv().await {
                // Forward agent updates as events
                event_bridge.emit(SystemEvent::CustomEvent {
                    name: "agent_update".to_string(),
                    data: serde_json::to_value(&update).unwrap_or_default(),
                    source: TabId::Custom("editor".to_string()),
                    target: Some(TabId::Chat),
                }).await?;
            }
            Ok::<(), anyhow::Error>(())
        });
        
        // Emit event about agent spawn
        self.event_bridge.emit(SystemEvent::CustomEvent {
            name: "spawn_code_agents".to_string(),
            data: serde_json::json!({
                "request_id": request.id,
                "description": request.description,
                "language": request.language,
                "chat_id": request.chat_id,
                "agent_id": agent.id(),
                "agent_name": agent_name,
            }),
            source: TabId::Custom("editor".to_string()),
            target: Some(TabId::Custom("agents".to_string())),
        }).await?;
        
        tracing::info!("Spawned code agent {} for request {}", agent_name, request.id);
        Ok(())
    }
    
    /// Transfer code from chat to editor
    pub async fn transfer_code_to_editor(
        &self,
        chat_id: String,
        code: String,
        language: Option<String>,
        file_name: Option<String>,
    ) -> Result<()> {
        let editor = self.get_or_create_editor(&chat_id).await?;
        
        // Set the code content
        editor.editor.set_content(code.clone()).await?;
        
        // If file name is provided, open/create the file
        if let Some(ref name) = file_name {
            editor.open_file(name).await?;
        }
        
        // Create a snapshot
        let snapshot = EditorStateSnapshot {
            content: code,
            cursor_position: (0, 0),
            selection: None,
            language: language.unwrap_or_else(|| "plaintext".to_string()),
            file_path: file_name,
            modified: true,
            timestamp: chrono::Utc::now(),
        };
        
        // Emit event for UI update
        self.event_bridge.emit(SystemEvent::CustomEvent {
            name: "code_transferred_to_editor".to_string(),
            data: serde_json::to_value(&snapshot)?,
            source: TabId::Chat,
            target: Some(TabId::Custom("editor".to_string())),
        }).await?;
        
        Ok(())
    }
    
    /// Execute code from editor
    pub async fn execute_code(
        &self,
        session_id: Option<String>,
        code: Option<String>,
    ) -> Result<CodeExecutionResult> {
        // Get code from editor if not provided
        let execution_code = if let Some(code) = code {
            code
        } else if let Some(session_id) = session_id {
            if let Some(session) = self.active_sessions.read().await.get(&session_id) {
                let editor = self.get_or_create_editor(&session.chat_id).await?;
                editor.get_content().await
            } else {
                return Err(anyhow::anyhow!("Session not found"));
            }
        } else {
            return Err(anyhow::anyhow!("No code or session provided"));
        };
        
        // Execute through the editor's execution system
        let editor = self.get_or_create_editor("default").await?;
        editor.editor.set_content(execution_code).await?;
        
        let result = editor.editor.execute().await?;
        
        Ok(CodeExecutionResult {
            success: result.success,
            output: result.output,
            error: result.error,
            execution_time_ms: result.execution_time,
        })
    }
    
    /// Get active editor sessions
    pub async fn get_active_sessions(&self) -> Vec<EditorSession> {
        self.active_sessions.read().await.values().cloned().collect()
    }
    
    /// Close an editor session
    pub async fn close_session(&self, session_id: &str) -> Result<()> {
        self.active_sessions.write().await.remove(session_id);
        
        // Emit close event
        self.event_bridge.emit(SystemEvent::CustomEvent {
            name: "editor_session_closed".to_string(),
            data: serde_json::json!({ "session_id": session_id }),
            source: TabId::Custom("editor".to_string()),
            target: None,
        }).await?;
        
        Ok(())
    }
    
    /// Sync editor content with chat context
    pub async fn sync_with_chat(&self, chat_id: &str) -> Result<()> {
        if let Some(editor) = self.editors.read().await.get(chat_id) {
            let content = editor.get_content().await;
            
            // Send content to chat as context
            self.event_bridge.emit(SystemEvent::CustomEvent {
                name: "editor_context_sync".to_string(),
                data: serde_json::json!({
                    "chat_id": chat_id,
                    "content": content,
                    "timestamp": chrono::Utc::now(),
                }),
                source: TabId::Custom("editor".to_string()),
                target: Some(TabId::Chat),
            }).await?;
        }
        
        Ok(())
    }
    
    /// Apply agent-suggested edit
    pub async fn apply_agent_edit(
        &self,
        session_id: &str,
        agent_id: &str,
        edit: EditAction,
    ) -> Result<()> {
        if let Some(session) = self.active_sessions.read().await.get(session_id) {
            let editor = self.get_or_create_editor(&session.chat_id).await?;
            
            // Apply the edit
            editor.apply_edit(edit.clone()).await?;
            
            // Log agent action
            tracing::info!("Agent {} applied edit to session {}", agent_id, session_id);
            
            // Notify chat about agent edit
            self.event_bridge.emit(SystemEvent::CustomEvent {
                name: "agent_edit_applied".to_string(),
                data: serde_json::json!({
                    "session_id": session_id,
                    "agent_id": agent_id,
                    "edit": edit,
                }),
                source: TabId::Custom("editor".to_string()),
                target: Some(TabId::Chat),
            }).await?;
        }
        
        Ok(())
    }
}

impl Clone for EditorBridge {
    fn clone(&self) -> Self {
        Self {
            event_bridge: self.event_bridge.clone(),
            editors: self.editors.clone(),
            code_requests: self.code_requests.clone(),
            active_sessions: self.active_sessions.clone(),
            state_cache: self.state_cache.clone(),
        }
    }
}