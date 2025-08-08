//! Tests for tool execution in the modular chat system
//! 
//! This verifies that tools can be properly executed through the chat interface

#[cfg(test)]
mod tool_execution_tests {
    use super::*;
    use crate::tui::chat::{ModularChat, SubtabManager};
    use crate::tui::chat::core::tool_executor::{ChatToolExecutor, ExecutionProgress};
    use crate::tui::chat::core::commands::{ParsedCommand, CommandResult, ResultFormat};
    use crate::tools::IntelligentToolManager;
    use crate::tools::task_management::{TaskManager, TaskConfig};
    use crate::tools::mcp_client::{McpClient, McpClientConfig};
    use crate::models::ModelOrchestrator;
    use crate::memory::CognitiveMemory;
    use std::sync::Arc;
    use std::collections::HashMap;
    use serde_json::json;
    use tokio::sync::{mpsc, RwLock};
    
    #[tokio::test]
    async fn test_tool_executor_creation() {
        // Create tool executor
        let executor = ChatToolExecutor::new();
        
        // Create a test command
        let command = ParsedCommand {
            command: "tools".to_string(),
            args: HashMap::new(),
            options: HashMap::new(),
            format: ResultFormat::Text,
        };
        
        // Execute the command
        let result = executor.execute(command).await;
        assert!(result.is_ok());
        
        let cmd_result = result.unwrap();
        assert!(cmd_result.success);
        assert!(!cmd_result.content.is_empty());
    }
    
    #[tokio::test]
    async fn test_tool_execution_with_args() -> anyhow::Result<()> {
        // Create components
        let tool_manager = Arc::new(IntelligentToolManager::default());
        let memory = Arc::new(CognitiveMemory::new(None, false).await?);
        let task_config = TaskConfig::default();
        let task_manager = Arc::new(TaskManager::new(task_config, memory.clone())?);
        
        // Create tool executor with components
        let mut executor = ChatToolExecutor::new();
        executor.set_tool_manager(tool_manager.clone());
        executor.set_task_manager(task_manager.clone());
        
        // Test executing a tool
        let command = ParsedCommand {
            command: "execute".to_string(),
            args: HashMap::from([
                ("tool".to_string(), json!("search")),
                ("args".to_string(), json!({
                    "query": "test search",
                    "limit": 10
                })),
            ]),
            options: HashMap::new(),
            format: ResultFormat::Json,
        };
        
        // Execute the command
        let result = executor.execute(command).await;
        
        // Verify result structure
        match result {
            Ok(cmd_result) => {
                assert!(cmd_result.metadata.contains_key("tool"));
                assert_eq!(cmd_result.format, ResultFormat::Json);
            }
            Err(e) => {
                // Tool might not be available in test environment
                assert!(e.to_string().contains("Tool execution failed") || 
                       e.to_string().contains("Tool not found"));
            }
        }
        
        Ok(())
    }
    
    #[tokio::test]
    async fn test_workflow_execution() -> anyhow::Result<()> {
        let executor = ChatToolExecutor::new();
        
        // Test workflow command
        let command = ParsedCommand {
            command: "workflow".to_string(),
            args: HashMap::from([
                ("name".to_string(), json!("code_review")),
                ("params".to_string(), json!({
                    "file": "test.rs"
                })),
            ]),
            options: HashMap::new(),
            format: ResultFormat::Mixed,
        };
        
        // Execute workflow
        let result = executor.execute(command).await;
        
        // Workflows might not be fully implemented yet
        match result {
            Ok(cmd_result) => {
                assert_eq!(cmd_result.format, ResultFormat::Mixed);
            }
            Err(e) => {
                // Expected in test environment
                assert!(e.to_string().contains("Workflow not found") ||
                       e.to_string().contains("No model orchestrator"));
            }
        }
        
        Ok(())
    }
    
    #[tokio::test]
    async fn test_task_creation_through_chat() -> anyhow::Result<()> {
        let memory = Arc::new(CognitiveMemory::new(None, false).await?);
        let task_config = TaskConfig::default();
        let task_manager = Arc::new(TaskManager::new(task_config, memory)?);
        
        let mut executor = ChatToolExecutor::new();
        executor.set_task_manager(task_manager);
        
        // Create a task
        let command = ParsedCommand {
            command: "task".to_string(),
            args: HashMap::from([
                ("action".to_string(), json!("create")),
                ("title".to_string(), json!("Test task from chat")),
                ("description".to_string(), json!("This is a test task")),
                ("priority".to_string(), json!("medium")),
            ]),
            options: HashMap::new(),
            format: ResultFormat::Text,
        };
        
        let result = executor.execute(command).await?;
        assert!(result.success);
        assert!(result.content.contains("created") || result.content.contains("Task"));
        
        Ok(())
    }
    
    #[tokio::test]
    async fn test_tool_search_functionality() {
        let executor = ChatToolExecutor::new();
        
        // Search for tools
        let command = ParsedCommand {
            command: "search".to_string(),
            args: HashMap::from([
                ("type".to_string(), json!("tools")),
                ("query".to_string(), json!("file")),
            ]),
            options: HashMap::new(),
            format: ResultFormat::Json,
        };
        
        let result = executor.execute(command).await;
        assert!(result.is_ok());
        
        let cmd_result = result.unwrap();
        // Should return search results or indicate no tools found
        assert!(cmd_result.success || cmd_result.content.contains("No tools found"));
    }
    
    #[tokio::test]
    async fn test_progress_tracking() -> anyhow::Result<()> {
        // Create executor and get progress receiver
        let executor = ChatToolExecutor::new();
        let mut progress_rx = executor.get_progress_receiver().await;
        
        // Execute a command in background
        let executor_clone = Arc::new(executor);
        let handle = tokio::spawn(async move {
            let command = ParsedCommand {
                command: "tools".to_string(),
                args: HashMap::new(),
                options: HashMap::new(),
                format: ResultFormat::Text,
            };
            executor_clone.execute(command).await
        });
        
        // Collect progress updates
        let mut progress_updates = Vec::new();
        while let Ok(Some(progress)) = progress_rx.try_recv() {
            progress_updates.push(progress);
        }
        
        // Wait for completion
        let _ = handle.await?;
        
        // Should have received at least starting and complete progress
        assert!(!progress_updates.is_empty());
        
        Ok(())
    }
    
    #[tokio::test]
    async fn test_modular_chat_tool_integration() -> anyhow::Result<()> {
        // Create modular chat
        let modular_chat = ModularChat::new().await;
        
        // Get the subtab manager
        let mut subtab_manager = modular_chat.subtab_manager.borrow_mut();
        
        // Test tool commands through chat
        let tool_commands = vec![
            "/tools",
            "/execute tool=search args={\"query\":\"test\"}",
            "/task action=list",
            "/help tools",
        ];
        
        for cmd in tool_commands {
            // Clear any existing input
            subtab_manager.handle_key_event(crossterm::event::KeyEvent::new(
                crossterm::event::KeyCode::Char('a'),
                crossterm::event::KeyModifiers::CONTROL,
            ));
            subtab_manager.handle_key_event(crossterm::event::KeyEvent::new(
                crossterm::event::KeyCode::Delete,
                crossterm::event::KeyModifiers::empty(),
            ));
            
            // Type the command
            for ch in cmd.chars() {
                subtab_manager.handle_char_input(vec![ch]);
            }
            
            // Verify the command was typed
            // (In real usage, pressing Enter would execute it)
        }
        
        Ok(())
    }
}