//! Cognitive Command Router
//! 
//! Routes cognitive commands to the appropriate processing system

use anyhow::Result;
use crate::tui::chat::core::commands::{ParsedCommand, CommandResult, ResultFormat};
use crate::tui::chat::integrations::cognitive::CognitiveChatEnhancement;
use std::sync::Arc;

/// Check if a command is a cognitive command
pub fn is_cognitive_command(command: &str) -> bool {
    matches!(
        command,
        "think" | "create" | "empathize" | "evolve" | "story" |
        "reflect" | "ponder" | "imagine" | "innovate" | 
        "understand" | "feel" | "adapt" | "improve" | "narrative" |
        "analyze" | "dream" | "meditate" | "contemplate"
    )
}

/// Route cognitive command to appropriate handler
pub async fn route_cognitive_command(
    command: ParsedCommand,
    cognitive_enhancement: &Arc<CognitiveChatEnhancement>,
    session_id: &str,
) -> Result<CommandResult> {
    // Build the command string for processing
    let mut command_str = format!("/{}", command.command);
    
    // Add main argument if present
    if let Some(topic) = command.args.get("topic").and_then(|v| v.as_str()) {
        command_str.push_str(&format!(" {}", topic));
    } else if let Some(prompt) = command.args.get("prompt").and_then(|v| v.as_str()) {
        command_str.push_str(&format!(" {}", prompt));
    } else if let Some(context) = command.args.get("context").and_then(|v| v.as_str()) {
        command_str.push_str(&format!(" {}", context));
    } else if let Some(target) = command.args.get("target").and_then(|v| v.as_str()) {
        command_str.push_str(&format!(" {}", target));
    } else if let Some(focus) = command.args.get("focus").and_then(|v| v.as_str()) {
        command_str.push_str(&format!(" {}", focus));
    }
    
    // Add additional parameters
    for (key, value) in &command.args {
        if !matches!(key.as_str(), "topic" | "prompt" | "context" | "target" | "focus") {
            if let Some(str_val) = value.as_str() {
                command_str.push_str(&format!(" {}:{}", key, str_val));
            }
        }
    }
    
    // Build context
    let context = serde_json::json!({
        "command": command.command,
        "args": command.args,
        "session_id": session_id,
    });
    
    // Process through cognitive enhancement
    let response = cognitive_enhancement
        .process_message(&command_str)
        .await;
    
    // Convert to CommandResult
    Ok(CommandResult {
        success: true,
        content: serde_json::json!({
            "response": response.content,
            "reasoning": response.reasoning,
            "confidence": response.confidence,
            "insights": response.cognitive_insights,
        }),
        format: ResultFormat::Text,
        warnings: vec![],
        suggestions: vec![],  // No suggestions field in CognitiveResponse
        metadata: [
            ("command".to_string(), serde_json::json!(command.command)),
            ("confidence".to_string(), serde_json::json!(response.confidence)),
        ].into(),
    })
}