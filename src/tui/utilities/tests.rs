//! Tests for the utilities module

#[cfg(test)]
mod tests {
    use super::super::*;
    use std::sync::Arc;
    use tokio::sync::RwLock;

    #[tokio::test]
    async fn test_utilities_state_creation() {
        let state = UtilitiesState::new();
        assert_eq!(state.cache.tools.len(), 0);
        assert_eq!(state.cache.plugins.len(), 0);
        assert_eq!(state.cache.daemons.len(), 0);
        assert_eq!(state.cache.mcp_servers.len(), 0);
    }

    #[tokio::test]
    async fn test_config_manager_creation() {
        let config_manager = config::ConfigManager::new();
        assert!(config_manager.is_ok());
        
        if let Ok(manager) = config_manager {
            let all_config = manager.get_all();
            assert_eq!(all_config.tools.len(), 0);
            assert_eq!(all_config.plugins.len(), 0);
            assert_eq!(all_config.mcp_servers.len(), 0);
            assert_eq!(all_config.daemons.len(), 0);
        }
    }

    #[tokio::test]
    async fn test_search_overlay() {
        let mut overlay = components::SearchOverlay::new();
        assert!(!overlay.is_active());
        
        overlay.activate();
        assert!(overlay.is_active());
        
        overlay.input_char('t');
        overlay.input_char('e');
        overlay.input_char('s');
        overlay.input_char('t');
        assert_eq!(overlay.get_query(), "test");
        
        overlay.deactivate();
        assert!(!overlay.is_active());
    }

    #[tokio::test]
    async fn test_system_metrics() {
        let metrics = metrics::SystemMetrics::new();
        
        // Test CPU usage
        let cpu_usage = metrics.get_cpu_usage().await;
        assert!(cpu_usage >= 0.0 && cpu_usage <= 100.0);
        
        // Test memory usage
        let (used, total) = metrics.get_memory_usage().await;
        assert!(used <= total);
        assert!(total > 0);
        
        // Test process count
        let process_count = metrics.get_process_count().await;
        assert!(process_count > 0);
        
        // Test uptime
        let uptime = metrics.get_uptime().await;
        assert!(uptime > 0);
    }

    #[tokio::test]
    async fn test_utilities_subtab_manager() {
        let state = Arc::new(RwLock::new(UtilitiesState::new()));
        let manager = integration::UtilitiesSubtabManager::new(
            state.clone(),
            None,
            None,
            None,
            None,
        );
        
        assert!(manager.is_ok());
        
        if let Ok(mut mgr) = manager {
            // Test tab count
            assert_eq!(mgr.tab_count(), 5); // Tools, MCP, Plugins, Daemon, Monitoring
            
            // Test tab names
            assert_eq!(mgr.current_tab_name(), "Tools");
            
            mgr.next_tab();
            assert_eq!(mgr.current_tab_name(), "MCP");
            
            mgr.next_tab();
            assert_eq!(mgr.current_tab_name(), "Plugins");
            
            mgr.previous_tab();
            assert_eq!(mgr.current_tab_name(), "MCP");
        }
    }

    #[tokio::test]
    async fn test_tool_bridge() {
        use crate::tools::IntelligentToolManager;
        
        // Create a tool manager (would need mock in real tests)
        let tool_manager = Arc::new(IntelligentToolManager::new());
        let bridge = bridges::ToolBridge::new(tool_manager);
        
        // Test tool config retrieval - should return error for non-existent tool
        let config = bridge.get_tool_config("test-tool").await;
        assert!(config.is_err()); // Non-existent tool should error
        
        // Test setting tool enabled state - should handle gracefully
        let result = bridge.set_tool_enabled("test-tool", true).await;
        // Should error for non-existent tool but not panic
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_format_functions() {
        // Test format_bytes
        assert_eq!(metrics::format_bytes(0), "0.00 B");
        assert_eq!(metrics::format_bytes(1024), "1.00 KB");
        assert_eq!(metrics::format_bytes(1024 * 1024), "1.00 MB");
        assert_eq!(metrics::format_bytes(1024 * 1024 * 1024), "1.00 GB");
        
        // Test format_percentage
        assert_eq!(metrics::format_percentage(0.0), "0.0%");
        assert_eq!(metrics::format_percentage(50.5), "50.5%");
        assert_eq!(metrics::format_percentage(100.0), "100.0%");
        
        // Test format_uptime
        assert_eq!(metrics::format_uptime(0), "0m");
        assert_eq!(metrics::format_uptime(60), "1m");
        assert_eq!(metrics::format_uptime(3600), "1h 0m");
        assert_eq!(metrics::format_uptime(86400), "1d 0h 0m");
    }

    #[tokio::test]
    async fn test_command_palette() {
        let mut palette = components::CommandPalette::new();
        assert!(!palette.is_open());
        
        palette.open();
        assert!(palette.is_open());
        
        // Test adding command
        palette.input('t');
        palette.input('e');
        palette.input('s');
        palette.input('t');
        assert_eq!(palette.current_input(), "test");
        
        palette.close();
        assert!(!palette.is_open());
    }

    #[tokio::test]
    async fn test_cached_metrics() {
        let metrics = CachedMetrics::default();
        assert_eq!(metrics.tools_active, 0);
        assert_eq!(metrics.mcp_servers_connected, 0);
        assert_eq!(metrics.plugins_loaded, 0);
        assert_eq!(metrics.daemons_running, 0);
        assert_eq!(metrics.cpu_usage, 0.0);
        assert_eq!(metrics.memory_usage, 0.0);
    }

    #[tokio::test]
    async fn test_plugin_config_serialization() {
        let config = config::PluginConfig {
            id: "test-plugin".to_string(),
            enabled: true,
            auto_update: false,
            settings: serde_json::json!({
                "key": "value",
                "number": 42
            }),
        };
        
        // Test serialization
        let json = serde_json::to_string(&config);
        assert!(json.is_ok());
        
        // Test deserialization
        if let Ok(json_str) = json {
            let parsed: Result<config::PluginConfig, _> = serde_json::from_str(&json_str);
            assert!(parsed.is_ok());
            
            if let Ok(parsed_config) = parsed {
                assert_eq!(parsed_config.id, config.id);
                assert_eq!(parsed_config.enabled, config.enabled);
                assert_eq!(parsed_config.auto_update, config.auto_update);
            }
        }
    }

    #[tokio::test]
    async fn test_mcp_server_config_serialization() {
        use std::collections::HashMap;
        
        let mut env_vars = HashMap::new();
        env_vars.insert("TEST_VAR".to_string(), "test_value".to_string());
        
        let config = config::McpServerConfig {
            name: "test-server".to_string(),
            command: "mcp-server".to_string(),
            args: vec!["--port".to_string(), "8080".to_string()],
            auto_connect: true,
            env_vars,
        };
        
        // Test serialization
        let json = serde_json::to_string(&config);
        assert!(json.is_ok());
        
        // Test deserialization
        if let Ok(json_str) = json {
            let parsed: Result<config::McpServerConfig, _> = serde_json::from_str(&json_str);
            assert!(parsed.is_ok());
            
            if let Ok(parsed_config) = parsed {
                assert_eq!(parsed_config.name, config.name);
                assert_eq!(parsed_config.command, config.command);
                assert_eq!(parsed_config.args, config.args);
                assert_eq!(parsed_config.auto_connect, config.auto_connect);
            }
        }
    }
}