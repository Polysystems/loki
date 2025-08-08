//! Real-time collaborative editing for chat interface
//! 
//! Provides WebSocket-based real-time collaboration features including
//! shared cursor positions, live typing indicators, and conflict resolution.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};
use uuid::Uuid;
use anyhow::{Result, anyhow};

/// Collaboration manager handles all real-time collaboration features
pub struct CollaborationManager {
    /// Active collaboration sessions
    sessions: Arc<RwLock<HashMap<String, CollaborationSession>>>,
    
    /// Current user ID
    user_id: String,
    
    /// WebSocket connection handler
    ws_handler: Option<WebSocketHandler>,
    
    /// Event channel sender
    event_sender: mpsc::UnboundedSender<CollaborationEvent>,
    
    /// Event channel receiver
    event_receiver: Option<mpsc::UnboundedReceiver<CollaborationEvent>>,
    
    /// Operational transform engine
    ot_engine: OperationalTransform,
    
    /// Conflict resolution strategy
    conflict_resolver: ConflictResolver,
}

/// A collaboration session
#[derive(Debug, Clone)]
pub struct CollaborationSession {
    pub id: String,
    pub name: String,
    pub created_at: DateTime<Utc>,
    pub participants: HashMap<String, UserPresence>,
    pub is_host: bool,
    pub permission_level: PermissionLevel,
    pub state: SessionState,
}

/// User presence information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserPresence {
    pub user_id: String,
    pub username: String,
    pub color: String,
    pub cursor_position: Option<CursorPosition>,
    pub selection: Option<TextSelection>,
    pub is_typing: bool,
    pub last_activity: DateTime<Utc>,
    pub status: UserStatus,
}

/// Cursor position in chat
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CursorPosition {
    pub message_id: Option<String>,
    pub line: usize,
    pub column: usize,
}

/// Text selection range
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextSelection {
    pub start: CursorPosition,
    pub end: CursorPosition,
}

/// User status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum UserStatus {
    Active,
    Idle,
    Away,
    Offline,
}

/// Permission levels for collaboration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PermissionLevel {
    ReadOnly,
    Commenter,
    Editor,
    Admin,
}

/// Session state
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SessionState {
    Connecting,
    Connected,
    Syncing,
    Active,
    Disconnected,
    Error(String),
}

/// Collaboration events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CollaborationEvent {
    /// User joined the session
    UserJoined {
        user: UserPresence,
        timestamp: DateTime<Utc>,
    },
    
    /// User left the session
    UserLeft {
        user_id: String,
        timestamp: DateTime<Utc>,
    },
    
    /// Cursor position update
    CursorUpdate {
        user_id: String,
        position: CursorPosition,
    },
    
    /// Selection update
    SelectionUpdate {
        user_id: String,
        selection: Option<TextSelection>,
    },
    
    /// Text edit operation
    EditOperation {
        operation: EditOperation,
        user_id: String,
        timestamp: DateTime<Utc>,
    },
    
    /// Message sent
    MessageSent {
        message_id: String,
        user_id: String,
        content: String,
        timestamp: DateTime<Utc>,
    },
    
    /// Typing indicator
    TypingIndicator {
        user_id: String,
        is_typing: bool,
    },
    
    /// Session state change
    SessionStateChange {
        old_state: SessionState,
        new_state: SessionState,
    },
    
    /// Sync request
    SyncRequest {
        from_version: u64,
        to_version: u64,
    },
    
    /// Sync response
    SyncResponse {
        operations: Vec<EditOperation>,
        current_version: u64,
    },
}

/// Edit operation for operational transformation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EditOperation {
    pub id: String,
    pub operation_type: OperationType,
    pub position: usize,
    pub content: String,
    pub length: usize,
    pub version: u64,
}

/// Operation types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OperationType {
    Insert,
    Delete,
    Replace,
}

/// WebSocket handler for real-time communication
struct WebSocketHandler {
    url: String,
    connection: Option<tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>>,
    sender: mpsc::UnboundedSender<CollaborationEvent>,
}

/// Operational transform engine for conflict-free editing
#[derive(Clone)]
struct OperationalTransform {
    /// Operation history
    history: VecDeque<EditOperation>,
    
    /// Current document version
    version: u64,
    
    /// Maximum history size
    max_history_size: usize,
}

/// Conflict resolver for handling edit conflicts
#[derive(Clone)]
struct ConflictResolver {
    /// Resolution strategy
    strategy: ConflictResolutionStrategy,
}

/// Conflict resolution strategies
#[derive(Debug, Clone)]
enum ConflictResolutionStrategy {
    /// Last write wins
    LastWriteWins,
    
    /// Merge changes intelligently
    SmartMerge,
    
    /// Manual resolution required
    Manual,
}

impl CollaborationManager {
    pub fn new(user_id: String) -> Self {
        let (event_sender, event_receiver) = mpsc::unbounded_channel();
        
        Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
            user_id,
            ws_handler: None,
            event_sender,
            event_receiver: Some(event_receiver),
            ot_engine: OperationalTransform::new(),
            conflict_resolver: ConflictResolver::new(ConflictResolutionStrategy::SmartMerge),
        }
    }
    
    /// Create a new collaboration session
    pub async fn create_session(&mut self, name: String) -> Result<String> {
        let session_id = Uuid::new_v4().to_string();
        
        let session = CollaborationSession {
            id: session_id.clone(),
            name,
            created_at: Utc::now(),
            participants: HashMap::new(),
            is_host: true,
            permission_level: PermissionLevel::Admin,
            state: SessionState::Connecting,
        };
        
        self.sessions.write().await.insert(session_id.clone(), session);
        
        // Initialize WebSocket connection
        self.connect_websocket(&session_id).await?;
        
        Ok(session_id)
    }
    
    /// Join an existing session
    pub async fn join_session(&mut self, session_id: String, username: String) -> Result<()> {
        // Connect to WebSocket
        self.connect_websocket(&session_id).await?;
        
        // Create user presence
        let user_presence = UserPresence {
            user_id: self.user_id.clone(),
            username,
            color: self.generate_user_color(),
            cursor_position: None,
            selection: None,
            is_typing: false,
            last_activity: Utc::now(),
            status: UserStatus::Active,
        };
        
        // Send join event
        self.send_event(CollaborationEvent::UserJoined {
            user: user_presence,
            timestamp: Utc::now(),
        }).await?;
        
        Ok(())
    }
    
    /// Leave current session
    pub async fn leave_session(&mut self, session_id: &str) -> Result<()> {
        // Send leave event
        self.send_event(CollaborationEvent::UserLeft {
            user_id: self.user_id.clone(),
            timestamp: Utc::now(),
        }).await?;
        
        // Remove session
        self.sessions.write().await.remove(session_id);
        
        // Close WebSocket connection
        if let Some(handler) = &mut self.ws_handler {
            handler.close().await?;
        }
        
        Ok(())
    }
    
    /// Send an edit operation
    pub async fn send_edit(&mut self, operation: EditOperation) -> Result<()> {
        // Apply operational transformation
        let transformed_op = self.ot_engine.transform(operation.clone())?;
        
        // Send event
        self.send_event(CollaborationEvent::EditOperation {
            operation: transformed_op,
            user_id: self.user_id.clone(),
            timestamp: Utc::now(),
        }).await?;
        
        Ok(())
    }
    
    /// Update cursor position
    pub async fn update_cursor(&mut self, position: CursorPosition) -> Result<()> {
        self.send_event(CollaborationEvent::CursorUpdate {
            user_id: self.user_id.clone(),
            position,
        }).await
    }
    
    /// Update selection
    pub async fn update_selection(&mut self, selection: Option<TextSelection>) -> Result<()> {
        self.send_event(CollaborationEvent::SelectionUpdate {
            user_id: self.user_id.clone(),
            selection,
        }).await
    }
    
    /// Update typing indicator
    pub async fn update_typing(&mut self, is_typing: bool) -> Result<()> {
        self.send_event(CollaborationEvent::TypingIndicator {
            user_id: self.user_id.clone(),
            is_typing,
        }).await
    }
    
    /// Send a message to the collaboration session
    pub async fn send_message(&mut self, message_id: String, content: String) -> Result<()> {
        self.send_event(CollaborationEvent::MessageSent {
            message_id,
            user_id: self.user_id.clone(),
            content,
            timestamp: Utc::now(),
        }).await
    }
    
    /// Process incoming events
    pub async fn process_events(&mut self) -> Result<Vec<CollaborationEvent>> {
        let mut events = Vec::new();
        let mut pending_edit_operations = Vec::new();
        let mut pending_user_joins = Vec::new();
        let mut pending_user_leaves = Vec::new();
        
        // Collect events first to avoid borrowing conflicts
        if let Some(receiver) = &mut self.event_receiver {
            while let Ok(event) = receiver.try_recv() {
                // Store updates to apply later
                match &event {
                    CollaborationEvent::EditOperation { operation, .. } => {
                        pending_edit_operations.push(operation.clone());
                    }
                    CollaborationEvent::UserJoined { user, .. } => {
                        pending_user_joins.push(user.clone());
                    }
                    CollaborationEvent::UserLeft { user_id, .. } => {
                        pending_user_leaves.push(user_id.clone());
                    }
                    _ => {}
                }
                
                events.push(event);
            }
        }
        
        // Apply edit operations
        for operation in pending_edit_operations {
            self.ot_engine.apply_remote(operation)?;
        }
        
        // Apply user join/leave updates
        if !pending_user_joins.is_empty() || !pending_user_leaves.is_empty() {
            let mut sessions = self.sessions.write().await;
            if let Some(session) = sessions.values_mut().next() {
                for user in pending_user_joins {
                    session.participants.insert(user.user_id.clone(), user);
                }
                for user_id in pending_user_leaves {
                    session.participants.remove(&user_id);
                }
            }
        }
        
        Ok(events)
    }
    
    /// Get active participants in current session
    pub async fn get_participants(&self) -> Vec<UserPresence> {
        if let Some(session) = self.get_current_session().await {
            session.participants.values().cloned().collect()
        } else {
            Vec::new()
        }
    }
    
    /// Connect to WebSocket server
    async fn connect_websocket(&mut self, session_id: &str) -> Result<()> {
        let url = format!("ws://localhost:8080/collab/{}", session_id);
        
        let mut handler = WebSocketHandler::new(url, self.event_sender.clone());
        handler.connect().await?;
        
        self.ws_handler = Some(handler);
        
        Ok(())
    }
    
    /// Send event through WebSocket
    async fn send_event(&mut self, event: CollaborationEvent) -> Result<()> {
        if let Some(handler) = &mut self.ws_handler {
            handler.send(event).await
        } else {
            Err(anyhow!("No active WebSocket connection"))
        }
    }
    
    /// Get current session
    async fn get_current_session(&self) -> Option<CollaborationSession> {
        let sessions = self.sessions.read().await;
        sessions.values().next().cloned()
    }
    
    /// Get mutable reference to current session
    async fn get_current_session_mut(&mut self) -> Option<CollaborationSession> {
        let mut sessions = self.sessions.write().await;
        sessions.values_mut().next().map(|s| s.clone())
    }
    
    /// Generate a unique color for user
    fn generate_user_color(&self) -> String {
        let colors = vec![
            "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FECA57",
            "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E2",
        ];
        
        let index = self.user_id.chars().map(|c| c as usize).sum::<usize>() % colors.len();
        colors[index].to_string()
    }
}

impl Clone for CollaborationManager {
    fn clone(&self) -> Self {
        // Create new channels for the clone
        let (event_sender, event_receiver) = mpsc::unbounded_channel();
        
        Self {
            sessions: self.sessions.clone(),
            user_id: self.user_id.clone(),
            ws_handler: None, // WebSocket handler can't be cloned, will need to reconnect
            event_sender,
            event_receiver: Some(event_receiver),
            ot_engine: self.ot_engine.clone(),
            conflict_resolver: self.conflict_resolver.clone(),
        }
    }
}

impl OperationalTransform {
    fn new() -> Self {
        Self {
            history: VecDeque::with_capacity(1000),
            version: 0,
            max_history_size: 1000,
        }
    }
    
    /// Transform an operation against concurrent operations
    fn transform(&mut self, operation: EditOperation) -> Result<EditOperation> {
        // Simplified transformation logic
        // In a real implementation, this would handle complex transformations
        let mut transformed = operation.clone();
        transformed.version = self.version + 1;
        
        Ok(transformed)
    }
    
    /// Apply a remote operation
    fn apply_remote(&mut self, operation: EditOperation) -> Result<()> {
        // Add to history
        self.history.push_back(operation);
        
        // Trim history if needed
        while self.history.len() > self.max_history_size {
            self.history.pop_front();
        }
        
        self.version += 1;
        
        Ok(())
    }
}

impl ConflictResolver {
    fn new(strategy: ConflictResolutionStrategy) -> Self {
        Self { strategy }
    }
    
    /// Resolve conflicts between operations
    fn resolve(&self, op1: &EditOperation, op2: &EditOperation) -> Result<Vec<EditOperation>> {
        match self.strategy {
            ConflictResolutionStrategy::LastWriteWins => {
                // Return the operation with later timestamp
                Ok(vec![op2.clone()])
            }
            ConflictResolutionStrategy::SmartMerge => {
                // Implement smart merging logic
                self.smart_merge(op1, op2)
            }
            ConflictResolutionStrategy::Manual => {
                Err(anyhow!("Manual conflict resolution required"))
            }
        }
    }
    
    /// Smart merge implementation
    fn smart_merge(&self, op1: &EditOperation, op2: &EditOperation) -> Result<Vec<EditOperation>> {
        // Simplified smart merge
        // In a real implementation, this would analyze the operations and merge intelligently
        if op1.position + op1.length < op2.position {
            // Non-overlapping operations
            Ok(vec![op1.clone(), op2.clone()])
        } else {
            // Overlapping - needs more complex resolution
            Ok(vec![op2.clone()])
        }
    }
}

impl WebSocketHandler {
    fn new(url: String, sender: mpsc::UnboundedSender<CollaborationEvent>) -> Self {
        Self {
            url,
            connection: None,
            sender,
        }
    }
    
    /// Connect to WebSocket server
    async fn connect(&mut self) -> Result<()> {
        // In a real implementation, this would establish a WebSocket connection
        // For now, we'll use a placeholder
        Ok(())
    }
    
    /// Send event through WebSocket
    async fn send(&mut self, event: CollaborationEvent) -> Result<()> {
        // In a real implementation, this would serialize and send the event
        // For now, we'll just echo it back through the channel
        self.sender.send(event)?;
        Ok(())
    }
    
    /// Close WebSocket connection
    async fn close(&mut self) -> Result<()> {
        self.connection = None;
        Ok(())
    }
}

/// Collaboration UI state
#[derive(Debug, Clone)]
pub struct CollaborationUI {
    /// Show participant list
    pub show_participants: bool,
    
    /// Show typing indicators
    pub show_typing_indicators: bool,
    
    /// Show cursors
    pub show_cursors: bool,
    
    /// Highlight active edits
    pub highlight_active_edits: bool,
    
    /// Participant list position
    pub participants_position: ParticipantListPosition,
}

/// Participant list position
#[derive(Debug, Clone, PartialEq)]
pub enum ParticipantListPosition {
    Top,
    Right,
    Bottom,
    Floating,
}

impl Default for CollaborationUI {
    fn default() -> Self {
        Self {
            show_participants: true,
            show_typing_indicators: true,
            show_cursors: true,
            highlight_active_edits: true,
            participants_position: ParticipantListPosition::Right,
        }
    }
}