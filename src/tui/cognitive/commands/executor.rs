//! Cognitive Command Executor for Chat Interface
//!
//! Handles execution of cognitive commands like /think, /create, /empathize

use std::sync::Arc;
use anyhow::{Result, Context};
use serde_json::json;
use tracing::info;

use crate::tui::chat::core::commands::{ParsedCommand, CommandResult, ResultFormat};
use crate::tui::nlp::core::orchestrator::NaturalLanguageOrchestrator;

/// Executor for cognitive commands
pub struct CognitiveCommandExecutor {
    orchestrator: Arc<NaturalLanguageOrchestrator>,
}

impl CognitiveCommandExecutor {
    /// Create new cognitive command executor
    pub fn new(orchestrator: Arc<NaturalLanguageOrchestrator>) -> Self {
        Self { orchestrator }
    }
    
    /// Execute cognitive command
    pub async fn execute(&self, command: ParsedCommand) -> Result<CommandResult> {
        match command.command.as_str() {
            "think" => self.execute_think(command).await,
            "create" => self.execute_create(command).await,
            "empathize" => self.execute_empathize(command).await,
            "evolve" => self.execute_evolve(command).await,
            "story" => self.execute_story(command).await,
            _ => Err(anyhow::anyhow!("Unknown cognitive command: {}", command.command)),
        }
    }
    
    /// Execute /think command
    async fn execute_think(&self, command: ParsedCommand) -> Result<CommandResult> {
        let topic = command.args.get("topic")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Topic required for thinking"))?;
        
        let depth = command.args.get("depth")
            .and_then(|v| v.as_str())
            .unwrap_or("standard");
        
        info!("🤔 Deep thinking about: {} (depth: {})", topic, depth);
        
        // Process through orchestrator
        let response = self.orchestrator
            .process_input("think_session", &format!("/think {}", topic))
            .await
            .context("Failed to process think command")?;
        
        Ok(CommandResult {
            success: true,
            content: json!({
                "thought": response.primary_response,
                "reasoning": String::new(),
                "confidence": response.confidence,
                "suggestions": response.suggestions,
            }),
            format: ResultFormat::Text,
            warnings: vec![],
            suggestions: response.suggestions,
            metadata: [
                ("command".to_string(), json!("think")),
                ("depth".to_string(), json!(depth)),
                ("topic".to_string(), json!(topic)),
            ].into(),
        })
    }
    
    /// Execute /create command
    async fn execute_create(&self, command: ParsedCommand) -> Result<CommandResult> {
        let prompt = command.args.get("prompt")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Creative prompt required"))?;
        
        let style = command.args.get("style")
            .and_then(|v| v.as_str())
            .unwrap_or("innovative");
        
        info!("🎨 Creative generation for: {} (style: {})", prompt, style);
        
        let response = self.orchestrator
            .process_input("create_session", &format!("/create {}", prompt))
            .await
            .context("Failed to process create command")?;
        
        Ok(CommandResult {
            success: true,
            content: json!({
                "creation": response.primary_response,
                "insights": response.suggestions,
                "confidence": response.confidence,
            }),
            format: ResultFormat::Text,
            warnings: vec![],
            suggestions: vec![
                "Try different creative styles: artistic, practical, abstract".to_string(),
                "Combine ideas with /create \"<idea1> + <idea2>\"".to_string(),
            ],
            metadata: [
                ("command".to_string(), json!("create")),
                ("style".to_string(), json!(style)),
            ].into(),
        })
    }
    
    /// Execute /empathize command
    async fn execute_empathize(&self, command: ParsedCommand) -> Result<CommandResult> {
        let context = command.args.get("context")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        
        info!("💝 Activating empathy mode");
        
        let input = if context.is_empty() {
            "/empathize".to_string()
        } else {
            format!("/empathize {}", context)
        };
        
        let response = self.orchestrator
            .process_input("empathy_session", &input)
            .await
            .context("Failed to process empathize command")?;
        
        Ok(CommandResult {
            success: true,
            content: json!({
                "response": response.primary_response,
                "emotional_understanding": "Active",
                "suggestions": response.suggestions,
            }),
            format: ResultFormat::Text,
            warnings: vec![],
            suggestions: response.suggestions,
            metadata: [
                ("command".to_string(), json!("empathize")),
                ("mode".to_string(), json!("emotional_intelligence")),
            ].into(),
        })
    }
    
    /// Execute /evolve command
    async fn execute_evolve(&self, command: ParsedCommand) -> Result<CommandResult> {
        let focus = command.args.get("focus")
            .and_then(|v| v.as_str())
            .unwrap_or("general");
        
        let confirm = command.args.get("confirm")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        
        info!("🧬 Evolution request - focus: {}, confirmed: {}", focus, confirm);
        
        if !confirm {
            return Ok(CommandResult {
                success: true,
                content: json!({
                    "status": "preview",
                    "message": format!(
                        "Evolution Preview - Focus: {}\n\n\
                        This would trigger autonomous self-improvement in the {} domain.\n\
                        To confirm, use: /evolve {} true",
                        focus, focus, focus
                    ),
                    "safety": "Evolution requires explicit confirmation for safety",
                }),
                format: ResultFormat::Json,
                warnings: vec!["Evolution is a powerful feature - use with caution".to_string()],
                suggestions: vec![
                    format!("Review current {} capabilities first", focus),
                    "Check evolution logs with /logs evolution".to_string(),
                ],
                metadata: [
                    ("command".to_string(), json!("evolve")),
                    ("focus".to_string(), json!(focus)),
                    ("confirmed".to_string(), json!(false)),
                ].into(),
            });
        }
        
        let response = self.orchestrator
            .process_input("evolve_session", &format!("/evolve {}", focus))
            .await
            .context("Failed to process evolve command")?;
        
        Ok(CommandResult {
            success: true,
            content: json!({
                "evolution_triggered": true,
                "focus": focus,
                "response": response.primary_response,
            }),
            format: ResultFormat::Json,
            warnings: vec![],
            suggestions: response.suggestions,
            metadata: [
                ("command".to_string(), json!("evolve")),
                ("focus".to_string(), json!(focus)),
                ("confirmed".to_string(), json!(true)),
            ].into(),
        })
    }
    
    /// Execute /story command
    async fn execute_story(&self, command: ParsedCommand) -> Result<CommandResult> {
        let mode = command.args.get("mode")
            .and_then(|v| v.as_str())
            .unwrap_or("analyze");
        
        let target = command.args.get("target")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        
        info!("📖 Story-driven {} mode for: {}", mode, target);
        
        let input = format!("/story {} {}", mode, target);
        
        let response = self.orchestrator
            .process_input("story_session", &input)
            .await
            .context("Failed to process story command")?;
        
        Ok(CommandResult {
            success: true,
            content: json!({
                "mode": mode,
                "analysis": response.primary_response,
                "narrative_insights": response.suggestions,
            }),
            format: ResultFormat::Text,
            warnings: vec![],
            suggestions: response.suggestions,
            metadata: [
                ("command".to_string(), json!("story")),
                ("mode".to_string(), json!(mode)),
                ("target".to_string(), json!(target)),
            ].into(),
        })
    }
}

/// Check if a command is a cognitive command
pub fn is_cognitive_command(command: &str) -> bool {
    matches!(
        command,
        "think" | "create" | "empathize" | "evolve" | "story" |
        "reflect" | "ponder" | "imagine" | "innovate" | 
        "understand" | "feel" | "adapt" | "improve" | "narrative"
    )
}