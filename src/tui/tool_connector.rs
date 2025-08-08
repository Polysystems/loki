use std::sync::Arc;
use anyhow::Result;
use tracing::{debug, info};

use crate::tools::{
    IntelligentToolManager,
    metrics_collector::{ToolMetricsCollector, ToolMetrics, ActiveToolSession, SystemMetrics, ToolStatus},
};

/// Connector between the tool system and TUI
pub struct ToolSystemConnector {
    /// Intelligent tool manager instance
    tool_manager: Arc<IntelligentToolManager>,
    /// Metrics collector
    metrics_collector: Arc<ToolMetricsCollector>,
}

impl ToolSystemConnector {
    /// Create a new tool system connector
    pub async fn new(tool_manager: Arc<IntelligentToolManager>) -> Result<Self> {
        let metrics_collector = Arc::new(ToolMetricsCollector::new());

        // Initialize metrics for all available tools
        let connector = Self {
            tool_manager,
            metrics_collector: metrics_collector.clone(),
        };

        connector.initialize_tool_metrics().await?;

        Ok(connector)
    }

    /// Initialize metrics for all available tools
    async fn initialize_tool_metrics(&self) -> Result<()> {
        info!("Initializing tool metrics");

        // Get available tools from the manager
        let available_tools = self.tool_manager.get_available_tools().await?;

        // Map of tool IDs to display names
        let tool_display_names = vec![
            ("web_search", "Web Search"),
            ("file_system", "File System"),
            ("github", "GitHub API"),
            ("discord", "Discord Bot"),
            ("email", "Email Processor"),
            ("blender", "Blender Integration"),
            ("calendar", "Calendar Manager"),
            ("code_analysis", "Code Analysis"),
            ("slack", "Slack Integration"),
            ("websocket", "WebSocket Client"),
            ("arxiv", "ArXiv Search"),
            ("doc_crawler", "Doc Crawler"),
            ("vision", "Vision System"),
            ("computer_use", "Computer Use"),
            ("task_management", "Task Management"),
            ("creative_media", "Creative Media"),
        ];

        // Initialize metrics for each tool
        for (tool_id, display_name) in tool_display_names {
            self.metrics_collector.initialize_tool(
                tool_id.to_string(),
                display_name.to_string()
            ).await;
        }

        // Also initialize any tools found in the manager that aren't in our list
        for tool_id in available_tools {
            // Check if we already initialized this tool
            let metrics = self.metrics_collector.get_all_tool_metrics().await;
            if !metrics.iter().any(|m| m.tool_id == tool_id) {
                // Use the tool ID as display name if not in our mapping
                self.metrics_collector.initialize_tool(
                    tool_id.clone(),
                    tool_id.clone()
                ).await;
            }
        }

        Ok(())
    }

    /// Get tool metrics formatted for TUI display
    pub async fn get_tool_metrics_for_tui(&self) -> Result<TuiToolData> {
        let metrics_data = self.metrics_collector.get_tui_display_data().await;

        // Get tool health from the manager
        let tool_health = self.tool_manager.check_tool_health().await?;

        // Get active sessions from the manager
        let manager_sessions = self.tool_manager.get_active_sessions().await?;

        // Merge data from both sources
        let mut enhanced_metrics = metrics_data.tool_metrics;

        // Update metrics with health status from manager
        for metric in &mut enhanced_metrics {
            if let Some(health) = tool_health.get(&metric.tool_id) {
                // Update status based on health enum variant
                metric.status = match health {
                    crate::tools::intelligent_manager::ToolHealthStatus::Healthy => ToolStatus::Active,
                    crate::tools::intelligent_manager::ToolHealthStatus::Degraded { .. } => ToolStatus::Processing,
                    crate::tools::intelligent_manager::ToolHealthStatus::Warning { .. } => ToolStatus::Processing,
                    crate::tools::intelligent_manager::ToolHealthStatus::Critical { .. } => ToolStatus::Error,
                    crate::tools::intelligent_manager::ToolHealthStatus::Unknown { .. } => ToolStatus::Idle,
                };

                // ToolHealthStatus doesn't have success_rate field, provide default
                // Success rate would need to be calculated from tool statistics
                let success_rate = 0.8; // Default success rate
                metric.success_rate = success_rate * 100.0;
            }
        }

        // Convert manager sessions to our format
        let mut active_sessions = metrics_data.active_sessions;
        for manager_session in manager_sessions {
            // Check if we already have this session
            if !active_sessions.iter().any(|s| s.session_id == manager_session.session_id) {
                let duration = std::time::Instant::now().duration_since(manager_session.start_time);
                active_sessions.push(ActiveToolSession {
                    session_id: manager_session.session_id.clone(),
                    tool_id: manager_session.tool_id.clone(),
                    tool_name: self.get_tool_display_name(&manager_session.tool_id).await,
                    status: crate::tools::metrics_collector::SessionStatus::Active,
                    start_time: chrono::Utc::now() - chrono::Duration::from_std(duration).unwrap_or_default(),
                    duration: chrono::Duration::from_std(duration).unwrap_or_default(),
                    request_type: "General".to_string(), // TODO: Add request tracking to ToolSessionDetails
                });
            }
        }

        Ok(TuiToolData {
            tool_metrics: enhanced_metrics,
            active_sessions,
            system_metrics: metrics_data.system_metrics,
        })
    }

    /// Start tracking a tool invocation
    pub async fn track_tool_start(
        &self,
        tool_id: &str,
        session_id: String,
        request_type: String,
    ) -> Result<()> {
        self.metrics_collector.record_invocation_start(
            tool_id,
            session_id,
            request_type
        ).await
    }

    /// Complete tracking a tool invocation
    pub async fn track_tool_complete(
        &self,
        tool_id: &str,
        session_id: &str,
        success: bool,
        response_time_ms: u64,
    ) -> Result<()> {
        self.metrics_collector.record_invocation_complete(
            tool_id,
            session_id,
            success,
            response_time_ms
        ).await
    }

    /// Get tool display name
    async fn get_tool_display_name(&self, tool_id: &str) -> String {
        match tool_id {
            "web_search" => "Web Search",
            "file_system" => "File System",
            "github" => "GitHub API",
            "discord" => "Discord Bot",
            "email" => "Email Processor",
            "blender" => "Blender Integration",
            "calendar" => "Calendar Manager",
            "code_analysis" => "Code Analysis",
            "slack" => "Slack Integration",
            "websocket" => "WebSocket Client",
            "arxiv" => "ArXiv Search",
            "doc_crawler" => "Doc Crawler",
            "vision" => "Vision System",
            "computer_use" => "Computer Use",
            "task_management" => "Task Management",
            "creative_media" => "Creative Media",
            _ => tool_id,
        }.to_string()
    }

    /// Refresh tool data
    pub async fn refresh_tools(&self) -> Result<()> {
        debug!("Refreshing tool data");
        // Re-initialize metrics for any new tools
        self.initialize_tool_metrics().await
    }
}

/// Data structure for TUI display
#[derive(Debug, Clone)]
pub struct TuiToolData {
    pub tool_metrics: Vec<ToolMetrics>,
    pub active_sessions: Vec<ActiveToolSession>,
    pub system_metrics: SystemMetrics,
}
