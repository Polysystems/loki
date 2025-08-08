use std::collections::HashMap;
use std::process::Command;
use std::time::Instant;
use regex::Regex;
use serde::{Deserialize, Serialize};
use {reqwest, serde_json};

use crate::config::ApiKeysConfig;

/// Enhanced message structure with editing capability and streaming support
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AssistantResponseType {
    Message {
        id: String,
        author: String,
        message: String,
        timestamp: String,
        is_editing: bool,
        edit_history: Vec<String>,
        streaming_state: StreamingState,
        metadata: MessageMetadata,
    },
    Action {
        id: String,
        author: String,
        command: String,
        timestamp: String,
        is_editing: bool,
        edit_history: Vec<String>,
        execution_state: ExecutionState,
        metadata: MessageMetadata,
    },
    Code {
        id: String,
        author: String,
        language: String,
        code: String,
        timestamp: String,
        is_editing: bool,
        edit_history: Vec<String>,
        metadata: MessageMetadata,
    },
    Error {
        id: String,
        error_type: String,
        message: String,
        timestamp: String,
        metadata: MessageMetadata,
    },
    ToolUse {
        id: String,
        tool_name: String,
        parameters: serde_json::Value,
        result: Option<serde_json::Value>,
        timestamp: String,
        status: ToolUseStatus,
        metadata: MessageMetadata,
    },
    Stream {
        id: String,
        author: String,
        partial_content: String,
        timestamp: String,
        stream_state: StreamingState,
        metadata: MessageMetadata,
    },
    ChatMessage {
        id: String,
        role: String,
        content: String,
        timestamp: String,
        metadata: MessageMetadata,
    },
    UserMessage {
        id: String,
        content: String,
        timestamp: String,
        metadata: MessageMetadata,
    },
    SystemMessage {
        id: String,
        content: String,
        timestamp: String,
        metadata: MessageMetadata,
    },
    ToolExecution {
        id: String,
        tool_name: String,
        input: serde_json::Value,
        output: serde_json::Value,
        timestamp: String,
        metadata: MessageMetadata,
    },
    ThinkingMessage {
        id: String,
        content: String,
        timestamp: String,
        metadata: MessageMetadata,
    },
}

/// Tool use status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ToolUseStatus {
    Pending,
    Executing,
    Completed,
    Failed { error: String },
}

/// Streaming state for messages being generated
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StreamingState {
    Complete,
    Streaming { progress: f32, estimated_total_tokens: Option<u32> },
    Queued,
    Processing { stage: ProcessingStage },
    Failed { error: String },
}

/// Processing stages for AI responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProcessingStage {
    ModelSelection,
    ContextPreperation,
    Generation,
    Validation,
    PostProcessing,
}

/// Execution state for action messages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionState {
    NotExecuted,
    Executing { progress: f32 },
    Completed { result: String },
    Failed { error: String },
}

/// Additional metadata for messages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageMetadata {
    pub model_used: Option<String>,
    pub tokens_used: Option<u32>,
    pub generation_time_ms: Option<u64>,
    pub confidence_score: Option<f32>,
    pub temperature: Option<f32>,
    pub is_favorited: bool,
    pub tags: Vec<String>,
    pub user_rating: Option<u8>, // 1-5 rating
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub last_edited: Option<chrono::DateTime<chrono::Utc>>,
    pub edit_count: u32,
}

impl Default for MessageMetadata {
    fn default() -> Self {
        Self {
            model_used: None,
            tokens_used: None,
            generation_time_ms: None,
            confidence_score: None,
            temperature: None,
            is_favorited: false,
            tags: Vec::new(),
            user_rating: None,
            created_at: chrono::Utc::now(),
            last_edited: None,
            edit_count: 0,
        }
    }
}

impl AssistantResponseType {
    /// Create a new user message
    pub fn new_user_message(content: String) -> Self {
        Self::Message {
            id: uuid::Uuid::new_v4().to_string(),
            author: "You".to_string(),
            message: content,
            timestamp: chrono::Utc::now().format("%H:%M:%S").to_string(),
            is_editing: false,
            edit_history: Vec::new(),
            streaming_state: StreamingState::Complete,
            metadata: MessageMetadata::default(),
        }
    }

    /// Create a new AI message with streaming support
    pub fn new_ai_message(content: String, model: Option<String>) -> Self {
        let mut metadata = MessageMetadata::default();
        metadata.model_used = model;

        Self::Message {
            id: uuid::Uuid::new_v4().to_string(),
            author: "Loki".to_string(),
            message: content,
            timestamp: chrono::Utc::now().format("%H:%M:%S").to_string(),
            is_editing: false,
            edit_history: Vec::new(),
            streaming_state: StreamingState::Complete,
            metadata,
        }
    }

    /// Create a streaming message placeholder
    pub fn new_streaming_placeholder(model: Option<String>) -> Self {
        let mut metadata = MessageMetadata::default();
        metadata.model_used = model;

        Self::Message {
            id: uuid::Uuid::new_v4().to_string(),
            author: "Loki".to_string(),
            message: String::new(),
            timestamp: chrono::Utc::now().format("%H:%M:%S").to_string(),
            is_editing: false,
            edit_history: Vec::new(),
            streaming_state: StreamingState::Processing { stage: ProcessingStage::ModelSelection },
            metadata,
        }
    }

    /// Create a new action with executing state
    pub fn new_action_executing(command: String, progress: f32) -> Self {
        Self::Action {
            id: uuid::Uuid::new_v4().to_string(),
            author: "Loki".to_string(),
            command,
            timestamp: chrono::Utc::now().format("%H:%M:%S").to_string(),
            is_editing: false,
            edit_history: Vec::new(),
            execution_state: ExecutionState::Executing { progress },
            metadata: MessageMetadata::default(),
        }
    }

    /// Create a new completed action
    pub fn new_action_completed(command: String, result: String) -> Self {
        Self::Action {
            id: uuid::Uuid::new_v4().to_string(),
            author: "Loki".to_string(),
            command,
            timestamp: chrono::Utc::now().format("%H:%M:%S").to_string(),
            is_editing: false,
            edit_history: Vec::new(),
            execution_state: ExecutionState::Completed { result },
            metadata: MessageMetadata::default(),
        }
    }

    /// Create a new failed action
    pub fn new_action_failed(command: String, error: String) -> Self {
        Self::Action {
            id: uuid::Uuid::new_v4().to_string(),
            author: "Loki".to_string(),
            command,
            timestamp: chrono::Utc::now().format("%H:%M:%S").to_string(),
            is_editing: false,
            edit_history: Vec::new(),
            execution_state: ExecutionState::Failed { error },
            metadata: MessageMetadata::default(),
        }
    }

    /// Create a new code block message
    pub fn new_code(language: String, code: String, author: String) -> Self {
        Self::Code {
            id: uuid::Uuid::new_v4().to_string(),
            author,
            language,
            code,
            timestamp: chrono::Utc::now().format("%H:%M:%S").to_string(),
            is_editing: false,
            edit_history: Vec::new(),
            metadata: MessageMetadata::default(),
        }
    }

    /// Create a new error message
    pub fn new_error(error_type: String, message: String) -> Self {
        Self::Error {
            id: uuid::Uuid::new_v4().to_string(),
            error_type,
            message,
            timestamp: chrono::Utc::now().format("%H:%M:%S").to_string(),
            metadata: MessageMetadata::default(),
        }
    }

    /// Create a new tool use message
    pub fn new_tool_use(tool_name: String, parameters: serde_json::Value) -> Self {
        Self::ToolUse {
            id: uuid::Uuid::new_v4().to_string(),
            tool_name,
            parameters,
            result: None,
            timestamp: chrono::Utc::now().format("%H:%M:%S").to_string(),
            status: ToolUseStatus::Pending,
            metadata: MessageMetadata::default(),
        }
    }

    /// Create a new stream message
    pub fn new_stream(author: String) -> Self {
        Self::Stream {
            id: uuid::Uuid::new_v4().to_string(),
            author,
            partial_content: String::new(),
            timestamp: chrono::Utc::now().format("%H:%M:%S").to_string(),
            stream_state: StreamingState::Queued,
            metadata: MessageMetadata::default(),
        }
    }

    /// Create a new chat message
    pub fn new_chat_message(role: String, content: String) -> Self {
        Self::ChatMessage {
            id: uuid::Uuid::new_v4().to_string(),
            role,
            content,
            timestamp: chrono::Utc::now().format("%H:%M:%S").to_string(),
            metadata: MessageMetadata::default(),
        }
    }

    /// Update execution state for actions
    pub fn update_execution_state(&mut self, new_state: ExecutionState) {
        if let AssistantResponseType::Action { execution_state, .. } = self {
            *execution_state = new_state;
        }
    }

    /// Check if action is currently executing
    pub fn is_executing(&self) -> bool {
        match self {
            AssistantResponseType::Action { execution_state, .. } => {
                matches!(execution_state, ExecutionState::Executing { .. })
            }
            _ => false,
        }
    }

    /// Get unique ID of the message
    pub fn get_id(&self) -> &str {
        match self {
            AssistantResponseType::Message { id, .. } => id,
            AssistantResponseType::Action { id, .. } => id,
            AssistantResponseType::Code { id, .. } => id,
            AssistantResponseType::Error { id, .. } => id,
            AssistantResponseType::ToolUse { id, .. } => id,
            AssistantResponseType::Stream { id, .. } => id,
            AssistantResponseType::ChatMessage { id, .. } => id,
            AssistantResponseType::UserMessage { id, .. } => id,
            AssistantResponseType::SystemMessage { id, .. } => id,
            AssistantResponseType::ToolExecution { id, .. } => id,
            AssistantResponseType::ThinkingMessage { id, .. } => id,
        }
    }

    pub fn get_author(&self) -> &str {
        match self {
            AssistantResponseType::Message { author, .. } => author,
            AssistantResponseType::Action { author, .. } => author,
            AssistantResponseType::Code { author, .. } => author,
            AssistantResponseType::Error { .. } => "System",
            AssistantResponseType::ToolUse { .. } => "Tool",
            AssistantResponseType::Stream { author, .. } => author,
            AssistantResponseType::ChatMessage { role, .. } => role,
            AssistantResponseType::UserMessage { .. } => "User",
            AssistantResponseType::SystemMessage { .. } => "System",
            AssistantResponseType::ToolExecution { .. } => "Tool",
            AssistantResponseType::ThinkingMessage { .. } => "Assistant",
        }
    }

    pub fn get_content(&self) -> &str {
        match self {
            AssistantResponseType::Message { message, .. } => message,
            AssistantResponseType::Action { command, .. } => command,
            AssistantResponseType::Code { code, .. } => code,
            AssistantResponseType::Error { message, .. } => message,
            AssistantResponseType::ToolUse { tool_name, .. } => tool_name,
            AssistantResponseType::Stream { partial_content, .. } => partial_content,
            AssistantResponseType::ChatMessage { content, .. } => content,
            AssistantResponseType::UserMessage { content, .. } => content,
            AssistantResponseType::SystemMessage { content, .. } => content,
            AssistantResponseType::ToolExecution { tool_name, .. } => tool_name,
            AssistantResponseType::ThinkingMessage { content, .. } => content,
        }
    }

    pub fn get_timestamp(&self) -> &str {
        match self {
            AssistantResponseType::Message { timestamp, .. } => timestamp,
            AssistantResponseType::Action { timestamp, .. } => timestamp,
            AssistantResponseType::Code { timestamp, .. } => timestamp,
            AssistantResponseType::Error { timestamp, .. } => timestamp,
            AssistantResponseType::ToolUse { timestamp, .. } => timestamp,
            AssistantResponseType::Stream { timestamp, .. } => timestamp,
            AssistantResponseType::ChatMessage { timestamp, .. } => timestamp,
            AssistantResponseType::UserMessage { timestamp, .. } => timestamp,
            AssistantResponseType::SystemMessage { timestamp, .. } => timestamp,
            AssistantResponseType::ToolExecution { timestamp, .. } => timestamp,
            AssistantResponseType::ThinkingMessage { timestamp, .. } => timestamp,
        }
    }

    pub fn is_action(&self) -> bool {
        matches!(self, AssistantResponseType::Action { .. })
    }

    pub fn is_editing(&self) -> bool {
        match self {
            AssistantResponseType::Message { is_editing, .. } => *is_editing,
            AssistantResponseType::Action { is_editing, .. } => *is_editing,
            AssistantResponseType::Code { is_editing, .. } => *is_editing,
            _ => false,
        }
    }

    pub fn is_streaming(&self) -> bool {
        match self {
            AssistantResponseType::Message { streaming_state, .. } => {
                !matches!(streaming_state, StreamingState::Complete | StreamingState::Failed { .. })
            }
            AssistantResponseType::Stream { stream_state, .. } => {
                !matches!(stream_state, StreamingState::Complete | StreamingState::Failed { .. })
            }
            _ => false,
        }
    }

    /// Start editing this message
    pub fn start_edit(&mut self) {
        match self {
            AssistantResponseType::Message { is_editing, .. } => *is_editing = true,
            AssistantResponseType::Action { is_editing, .. } => *is_editing = true,
            AssistantResponseType::Code { is_editing, .. } => *is_editing = true,
            _ => {}
        }
    }

    /// Finish editing and save to history
    pub fn finish_edit(&mut self, new_content: String) {
        match self {
            AssistantResponseType::Message {
                message, is_editing, edit_history, metadata, ..
            } => {
                edit_history.push(message.clone());
                *message = new_content;
                *is_editing = false;
                metadata.edit_count += 1;
                metadata.last_edited = Some(chrono::Utc::now());
            }
            AssistantResponseType::Action {
                command, is_editing, edit_history, metadata, ..
            } => {
                edit_history.push(command.clone());
                *command = new_content;
                *is_editing = false;
                metadata.edit_count += 1;
                metadata.last_edited = Some(chrono::Utc::now());
            }
            AssistantResponseType::Code {
                code, is_editing, edit_history, metadata, ..
            } => {
                edit_history.push(code.clone());
                *code = new_content;
                *is_editing = false;
                metadata.edit_count += 1;
                metadata.last_edited = Some(chrono::Utc::now());
            }
            _ => {}
        }
    }

    /// Update streaming content
    pub fn update_streaming_content(&mut self, new_content: String, progress: f32) {
        if let AssistantResponseType::Message { message, streaming_state, .. } = self {
            *message = new_content;
            *streaming_state = StreamingState::Streaming { progress, estimated_total_tokens: None };
        }
    }

    /// Complete streaming
    pub fn complete_streaming(
        &mut self,
        final_content: String,
        tokens_used: Option<u32>,
        generation_time: Option<u64>,
    ) {
        match self {
            AssistantResponseType::Message { message, streaming_state, metadata, .. } => {
                *message = final_content;
                *streaming_state = StreamingState::Complete;
                metadata.tokens_used = tokens_used;
                metadata.generation_time_ms = generation_time;
            }
            _ => {}
        }
    }

    /// Get streaming progress (0.0 to 1.0)
    pub fn get_streaming_progress(&self) -> f32 {
        match self {
            AssistantResponseType::Message { streaming_state, .. } => match streaming_state {
                StreamingState::Streaming { progress, .. } => *progress,
                StreamingState::Complete => 1.0,
                StreamingState::Failed { .. } => 0.0,
                _ => 0.0,
            },
            _ => 1.0,
        }
    }

    /// Get display content with streaming effects
    pub fn get_display_content(&self, current_time: Instant) -> String {
        let base_content = self.get_content();

        if self.is_streaming() {
            // More dynamic loading animations
            let elapsed = current_time.elapsed().as_secs_f32();
            let spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
            let spinner_idx = (elapsed * 4.0) as usize % spinner_chars.len();
            let spinner = spinner_chars[spinner_idx];
            
            // Pulsing dots animation
            let dots_cycle = (elapsed * 2.0) as usize % 8;
            let dots = match dots_cycle {
                0 => "   ",
                1 => "•  ",
                2 => "•• ",
                3 => "•••",
                4 => "•••",
                5 => "•• ",
                6 => "•  ",
                7 => "   ",
                _ => "   ",
            };

            if base_content.is_empty() {
                match self {
                    AssistantResponseType::Message { streaming_state, .. } => match streaming_state
                    {
                        StreamingState::Processing { stage } => {
                            format!("{} {} {}", spinner, Self::get_processing_stage_text(stage), dots)
                        }
                        StreamingState::Queued => format!("{} Queued for processing {}", spinner, dots),
                        StreamingState::Streaming { progress, .. } => {
                            let progress_bar = Self::get_progress_bar(*progress);
                            format!("{} Generating response {} {}", spinner, progress_bar, dots)
                        },
                        _ => format!("{} Generating response {}", spinner, dots),
                    },
                    AssistantResponseType::Action { execution_state, .. } => match execution_state {
                        ExecutionState::Executing { progress } => {
                            let progress_bar = Self::get_progress_bar(*progress);
                            format!("{} Executing command {} {}", spinner, progress_bar, dots)
                        },
                        _ => format!("{} Processing action {}", spinner, dots),
                    },
                    AssistantResponseType::Stream { stream_state, .. } => match stream_state {
                        StreamingState::Processing { stage } => {
                            format!("{} {} {}", spinner, Self::get_processing_stage_text(stage), dots)
                        }
                        StreamingState::Queued => format!("{} Queued for streaming {}", spinner, dots),
                        StreamingState::Streaming { progress, .. } => {
                            let progress_bar = Self::get_progress_bar(*progress);
                            format!("{} Streaming {} {}", spinner, progress_bar, dots)
                        },
                        _ => format!("{} Streaming {}", spinner, dots),
                    },
                    _ => format!("{} Processing {}", spinner, dots),
                }
            } else {
                format!("{} ▋", base_content) // Cursor indicator for streaming
            }
        } else {
            base_content.to_string()
        }
    }

    fn get_processing_stage_text(stage: &ProcessingStage) -> &'static str {
        match stage {
            ProcessingStage::ModelSelection => "🎯 Selecting optimal model",
            ProcessingStage::ContextPreperation => "📚 Preparing context",
            ProcessingStage::Generation => "🧠 Generating response",
            ProcessingStage::Validation => "✅ Validating output",
            ProcessingStage::PostProcessing => "🎨 Finalizing response",
        }
    }
    
    /// Generate a visual progress bar
    fn get_progress_bar(progress: f32) -> String {
        let width = 20;
        let filled = (progress * width as f32) as usize;
        let empty = width - filled;
        format!("[{}{}] {:.0}%", 
            "█".repeat(filled),
            "░".repeat(empty),
            progress * 100.0
        )
    }
}

#[derive(Debug, Clone)]
pub struct RunModelOptions {
    pub system_prompt: String,
    pub parse_actions: bool,
}

// Struct to hold the response and its state
#[derive(Debug, Clone)]
pub struct ModelResult {
    pub response: Option<String>,
    pub is_complete: bool,
    pub error: Option<String>,
    pub token_count: Option<usize>,
    pub generation_time_ms: Option<u64>,
}

impl ModelResult {
    /// Create an error result
    fn error(msg: impl Into<String>) -> Self {
        Self {
            response: None,
            is_complete: true,
            error: Some(msg.into()),
            token_count: None,
            generation_time_ms: None,
        }
    }
    
    /// Create a success result
    fn success(content: impl Into<String>) -> Self {
        Self {
            response: Some(content.into()),
            is_complete: true,
            error: None,
            token_count: None,
            generation_time_ms: None,
        }
    }
    
    /// Add timing information
    fn with_timing(mut self, start_time: Instant) -> Self {
        self.generation_time_ms = Some(start_time.elapsed().as_millis() as u64);
        self
    }
    
    /// Add token count
    fn with_tokens(mut self, count: usize) -> Self {
        self.token_count = Some(count);
        self
    }
}

pub async fn run_model(
    model: String,
    provider: String,
    prompt: String,
    options: Option<RunModelOptions>,
) -> ModelResult {
    let start_time = Instant::now();
    let apiconfig = ApiKeysConfig::from_env().expect("failed to load api keys").ai_models;
    let system_prompt = options.as_ref().map_or("", |o| &o.system_prompt);

    let client = reqwest::Client::new();

    match provider.to_lowercase().as_str() {
        "openai" => {
            let api_key = match apiconfig.openai.as_ref() {
                Some(key) => key,
                None => {
                    return ModelResult::error("OpenAI API key not set").with_timing(start_time);
                }
            };
            let response = client
                .post("https://api.openai.com/v1/chat/completions")
                .header("Authorization", format!("Bearer {}", api_key))
                .header("Content-Type", "application/json")
                .json(&serde_json::json!({
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    "stream": false
                }))
                .send()
                .await;

            match response {
                Ok(resp) => {
                    let json = resp.json::<serde_json::Value>().await;
                    match json {
                        Ok(data) => {
                            if let Some(choices) = data.get("choices").and_then(|c| c.as_array()) {
                                if let Some(choice) = choices.get(0) {
                                    if let Some(message) = choice.get("message") {
                                        if let Some(content) =
                                            message.get("content").and_then(|c| c.as_str())
                                        {
                                            let mut result = ModelResult::success(content);
                                            
                                            if let Some(usage) = data.get("usage") {
                                                if let Some(total) = usage.get("total_tokens").and_then(|t| t.as_u64()) {
                                                    result = result.with_tokens(total as usize);
                                                }
                                            }
                                            
                                            result.with_timing(start_time)
                                        } else {
                                            ModelResult::error("Missing 'content' in OpenAI response")
                                                .with_timing(start_time)
                                        }
                                    } else {
                                        ModelResult::error("Missing 'message' in OpenAI choice")
                                            .with_timing(start_time)
                                    }
                                } else {
                                    ModelResult::error("No choices in OpenAI response")
                                        .with_timing(start_time)
                                }
                            } else {
                                ModelResult::error("Invalid format in OpenAI response")
                                    .with_timing(start_time)
                            }
                        }
                        Err(e) => ModelResult::error(format!("Failed to parse OpenAI response: {}", e))
                            .with_timing(start_time),
                    }
                }
                Err(e) => ModelResult::error(format!("OpenAI request failed: {}", e))
                    .with_timing(start_time),
            }
        }
        "anthropic" => {
            let api_key = match apiconfig.anthropic.as_ref() {
                Some(key) => key,
                None => {
                    return ModelResult {
                        response: None,
                        is_complete: true,
                        error: Some("Anthropic API key not set".to_string()),
                        token_count: None,
                        generation_time_ms: None,
                    };
                }
            };
            let response = client
                .post("https://api.anthropic.com/v1/messages")
                .header("x-api-key", api_key)
                .header("anthropic-version", "2023-06-01")
                .header("Content-Type", "application/json")
                .json(&serde_json::json!({
                    "model": model,
                    "max_tokens": 1024,
                    "messages": [
                        {"role": "user", "content": system_prompt}
                    ],
                    "system": system_prompt,
                    "stream": false
                }))
                .send()
                .await;

            match response {
                Ok(resp) => {
                    let json = resp.json::<serde_json::Value>().await;
                    match json {
                        Ok(data) => {
                            let content = data["content"][0]["text"].as_str();
                            match content {
                                Some(text) => {
                                    let mut result = ModelResult::success(text);
                                    
                                    // Anthropic returns usage in a different format
                                    if let Some(usage) = data.get("usage") {
                                        if let Some(input) = usage.get("input_tokens").and_then(|t| t.as_u64()) {
                                            if let Some(output) = usage.get("output_tokens").and_then(|t| t.as_u64()) {
                                                result = result.with_tokens((input + output) as usize);
                                            }
                                        }
                                    }
                                    
                                    result.with_timing(start_time)
                                },
                                None => {
                    return ModelResult {
                                        response: None,
                                        is_complete: true,
                                        error: Some("No content in Anthropic response".to_string()),
                                        token_count: None,
                                        generation_time_ms: None,
                                    };
                                }
                            }
                        }
                        Err(e) => {
                            return ModelResult {
                                response: None,
                                is_complete: true,
                                error: Some(format!("Failed to parse Anthropic response: {}", e)),
                                token_count: None,
                                generation_time_ms: None,
                            };
                        }
                    }
                }
                Err(e) => {
                    return ModelResult {
                        response: None,
                        is_complete: true,
                        error: Some(format!("Anthropic request failed: {}", e)),
                        token_count: None,
                        generation_time_ms: None,
                    };
                }
            }
        }
        "deepseek" => {
            let api_key = match apiconfig.deepseek.as_ref() {
                Some(key) => key,
                None => {
                    return ModelResult {
                        response: None,
                        is_complete: true,
                        error: Some("DeepSeek API key not set".to_string()),
                        token_count: None,
                        generation_time_ms: None,
                    };
                }
            };
            let response = client
                .post("https://api.deepseek.com/v1/chat/completions")
                .header("Authorization", format!("Bearer {}", api_key))
                .header("Content-Type", "application/json")
                .json(&serde_json::json!({
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": system_prompt}
                    ],
                    "stream": false
                }))
                .send()
                .await;

            match response {
                Ok(resp) => {
                    let json = resp.json::<serde_json::Value>().await;
                    match json {
                        Ok(data) => {
                            let content = data["choices"][0]["message"]["content"].as_str();
                            match content {
                                Some(text) => ModelResult {
                                    response: Some(text.to_string()),
                                    is_complete: true,
                                    error: None,
                                    token_count: None,
                                    generation_time_ms: None,
                                },
                                None => ModelResult {
                                    response: None,
                                    is_complete: true,
                                    error: Some("No content in DeepSeek response".to_string()),
                                    token_count: None,
                                    generation_time_ms: None,
                                },
                            }
                        }
                        Err(e) => ModelResult {
                            response: None,
                            is_complete: true,
                            error: Some(format!("Failed to parse DeepSeek response: {}", e)),
                            token_count: None,
                            generation_time_ms: None,
                        },
                    }
                }
                Err(e) => ModelResult {
                    response: None,
                    is_complete: true,
                    error: Some(format!("DeepSeek request failed: {}", e)),
                    token_count: None,
                    generation_time_ms: None,
                },
            }
        }
        "grok" => {
            let api_key = match apiconfig.grok.as_ref() {
                Some(key) => key,
                None => return ModelResult { response: None, is_complete: true, error: None, token_count: None, generation_time_ms: None },
            };
            let response = client
                .post("https://api.x.ai/v1/chat/completions")
                .header("Authorization", format!("Bearer {}", api_key))
                .header("Content-Type", "application/json")
                .json(&serde_json::json!({
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": system_prompt}
                    ],
                    "stream": false
                }))
                .send()
                .await;

            match response {
                Ok(resp) => {
                    let json = resp.json::<serde_json::Value>().await;
                    match json {
                        Ok(data) => {
                            let content = data["choices"][0]["message"]["content"].as_str();
                            match content {
                                Some(text) => ModelResult {
                                    response: Some(text.to_string()),
                                    is_complete: true,
                                    error: None,
                                    token_count: None,
                                    generation_time_ms: None,
                                },
                                None => {
                                    return ModelResult {
                                        response: None,
                                        is_complete: true,
                                        error: Some("No content in Grok response".to_string()),
                                        token_count: None,
                                        generation_time_ms: None,
                                    };
                                }
                            }
                        }
                        Err(e) => {
                            return ModelResult {
                                response: None,
                                is_complete: true,
                                error: Some(format!("Failed to parse Grok response: {}", e)),
                                token_count: None,
                                generation_time_ms: None,
                            };
                        }
                    }
                }
                Err(e) => {
                    return ModelResult {
                        response: None,
                        is_complete: true,
                        error: Some(format!("Grok request failed: {}", e)),
                        token_count: None,
                        generation_time_ms: None,
                    };
                }
            }
        }
        "ollama" => {
            let output = Command::new("ollama")
                .arg("run")
                .arg(&model)
                .arg(format!("{{system_prompt:\"{}\", user: \"{}\"", system_prompt, prompt))
                .output();

            match output {
                Ok(output) => {
                    if output.status.success() {
                        ModelResult {
                            response: Some(String::from_utf8_lossy(&output.stdout).to_string()),
                            is_complete: true,
                            error: None,
                            token_count: None,
                            generation_time_ms: None,
                        }
                    } else {
                        ModelResult {
                            response: None,
                            is_complete: true,
                            error: Some(format!(
                                "Ollama command failed: {}",
                                String::from_utf8_lossy(&output.stderr)
                            )),
                            token_count: None,
                            generation_time_ms: None,
                        }
                    }
                }
                Err(e) => {
                    return ModelResult {
                        response: None,
                        is_complete: true,
                        error: Some(format!("Failed to run ollama: {}", e)),
                        token_count: None,
                        generation_time_ms: None,
                    };
                }
            }
        }
        _ => ModelResult {
            response: None,
            is_complete: true,
            error: Some(format!("Unsupported provider: {}", provider)),
            token_count: None,
            generation_time_ms: None,
        },
    }
}


#[derive(Debug)]
pub struct LokiArtifact {
    fields: HashMap<String, String>,
    content: Option<String>,
}

pub fn parse_loki_artifacts(input: &str) -> (String, Vec<LokiArtifact>) {
    let re_full = Regex::new(r#"<LokiArtifact(?P<attrs>[^>]*)>(?P<content>.*?)</LokiArtifact>"#).unwrap();
    let re_self_closing = Regex::new(r#"<LokiArtifact(?P<attrs>[^>]*)\s*/>"#).unwrap();

    let mut artifacts = Vec::new();
    let mut output = input.to_string();

    for caps in re_full.captures_iter(input) {
        let attrs = parse_attrs(caps.name("attrs").unwrap().as_str());
        let content = Some(caps.name("content").unwrap().as_str().to_string());
        artifacts.push(LokiArtifact { fields: attrs.clone(), content: content.clone() });

        let replacement = format!("\nartifact\n{}", content.unwrap_or_default());
        output = output.replace(caps.get(0).unwrap().as_str(), &replacement);
    }

    for caps in re_self_closing.captures_iter(input) {
        let attrs = parse_attrs(caps.name("attrs").unwrap().as_str());
        artifacts.push(LokiArtifact { fields: attrs.clone(), content: None });

        output = output.replace(caps.get(0).unwrap().as_str(), "\nartifact");
    }

    (output, artifacts)
}

pub fn parse_attrs(attr_str: &str) -> HashMap<String, String> {
    let attr_re = Regex::new(r#"\s*(\w+)\s*=\s*"([^"]*)""#).unwrap();
    attr_re
        .captures_iter(attr_str)
        .map(|cap| (cap[1].to_string(), cap[2].to_string()))
        .collect()
}