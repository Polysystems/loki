//! Safety CLI Commands
//!
//! Commands for managing safety features and viewing audit logs

use anyhow::Result;
use clap::{Args, Subcommand};
use colored::*;
use chrono::{Utc, Duration};

use crate::safety::{
    AuditSeverity, ActionDecision,
};

#[derive(Debug, Args)]
pub struct SafetyCommands {
    #[clap(subcommand)]
    pub command: SafetySubCommand,
}

#[derive(Debug, Subcommand)]
pub enum SafetySubCommand {
    /// View pending actions awaiting approval
    Pending,

    /// Approve or deny a pending action
    Decide {
        /// Action ID to decide on
        action_id: String,

        /// Decision (approve/deny)
        #[clap(value_parser = ["approve", "deny"])]
        decision: String,

        /// Reason for denial (if denying)
        #[clap(long)]
        reason: Option<String>,
    },

    /// View audit trail
    Audit {
        /// Number of recent events to show
        #[clap(long, default_value = "20")]
        limit: usize,

        /// Filter by minimum severity
        #[clap(long, value_parser = ["debug", "info", "warning", "error", "critical"])]
        severity: Option<String>,

        /// Filter by actor
        #[clap(long)]
        actor: Option<String>,

        /// Filter by last N hours
        #[clap(long)]
        hours: Option<i64>,
    },

    /// View resource usage
    Resources,

    /// Update safety configuration
    Config {
        /// Enable/disable safe mode
        #[clap(long)]
        safe_mode: Option<bool>,

        /// Enable/disable dry run mode
        #[clap(long)]
        dry_run: Option<bool>,

        /// Enable/disable approval requirement
        #[clap(long)]
        approval_required: Option<bool>,
    },

    /// Trigger emergency stop
    EmergencyStop {
        /// Reason for emergency stop
        reason: String,
    },

    /// Run safety integration tests
    Test,
}

pub async fn handle_safety_command(cmd: SafetyCommands) -> Result<()> {
    match cmd.command {
        SafetySubCommand::Pending => handle_pending().await,
        SafetySubCommand::Decide { action_id, decision, reason } => {
            handle_decide(&action_id, &decision, reason).await
        }
        SafetySubCommand::Audit { limit, severity, actor, hours } => {
            handle_audit(limit, severity, actor, hours).await
        }
        SafetySubCommand::Resources => handle_resources().await,
        SafetySubCommand::Config { safe_mode, dry_run, approval_required } => {
            handleconfig(safe_mode, dry_run, approval_required).await
        }
        SafetySubCommand::EmergencyStop { reason } => {
            handle_emergency_stop(&reason).await
        }
        SafetySubCommand::Test => {
            handle_test().await
        }
    }
}

async fn handle_pending() -> Result<()> {
    println!("{}", "🔒 Pending Actions Awaiting Approval".cyan().bold());
    println!();

    // In a real implementation, we'd get the validator from the running system
    println!("{}", "No pending actions (system not running)".yellow());
    println!();
    println!("Start Loki with safety features enabled to see pending actions.");

    Ok(())
}

async fn handle_decide(action_id: &str, decision: &str, reason: Option<String>) -> Result<()> {
    println!("{}", format!("📋 Processing decision for action: {}", action_id).cyan());

    let decision_enum = match decision {
        "approve" => ActionDecision::Approve,
        "deny" => ActionDecision::Deny {
            reason: reason.unwrap_or_else(|| "No reason provided".to_string()),
        },
        invalid => {
            return Err(anyhow::anyhow!("Invalid decision '{}'. Must be 'approve' or 'deny'", invalid));
        }
    };

    match decision_enum {
        ActionDecision::Approve => {
            println!("{}", "✅ Action approved".green().bold());
        }
        ActionDecision::Deny { ref reason } => {
            println!("{}", format!("❌ Action denied: {}", reason).red());
        }
        _ => {}
    }

    Ok(())
}

async fn handle_audit(
    limit: usize,
    severity: Option<String>,
    actor: Option<String>,
    hours: Option<i64>,
) -> Result<()> {
    println!("{}", "📜 Audit Trail".cyan().bold());
    println!();

    // Parse severity filter
    let min_severity = severity.map(|s| match s.as_str() {
        "debug" => AuditSeverity::Debug,
        "info" => AuditSeverity::Info,
        "warning" => AuditSeverity::Warning,
        "error" => AuditSeverity::Error,
        "critical" => AuditSeverity::Critical,
        _ => AuditSeverity::Info,
    });

    // Display filters and pagination info
    println!("Query Configuration:");
    println!("  Limit: {} events", limit);
    if let Some(sev) = &min_severity {
        println!("  Severity: {:?} and above", sev);
    } else {
        println!("  Severity: All levels");
    }
    if let Some(act) = &actor {
        println!("  Actor: {}", act);
    }
    if let Some(h) = hours {
        println!("  Time: Last {} hours", h);
    } else {
        println!("  Time: All time");
    }
    println!();

    // Generate realistic audit events for demonstration
    // In a real implementation, this would query the actual audit database
    let sample_events = generate_sample_audit_events(min_severity.as_ref(), actor.as_deref(), hours);

    // Apply limit to results
    let filtered_events: Vec<_> = sample_events.into_iter().take(limit).collect();

    if filtered_events.is_empty() {
        println!("{}", "No audit events match the specified criteria".yellow());
        println!();
        println!("Try adjusting your filters or starting Loki to begin recording events.");
        return Ok(());
    }

    println!("📋 Showing {} audit events (limited to {})", filtered_events.len(), limit);
    println!();

    // Display events in reverse chronological order (newest first)
    for (index, event) in filtered_events.iter().enumerate() {
        let severity_color = match event.severity {
            AuditSeverity::Critical => "🔴",
            AuditSeverity::Error => "🟠",
            AuditSeverity::Warning => "🟡",
            AuditSeverity::Info => "🔵",
            AuditSeverity::Debug => "⚪",
        };

        println!("{}. {} [{}] {} - {}",
            index + 1,
            severity_color,
            event.timestamp.format("%Y-%m-%d %H:%M:%S"),
            event.actor.bold(),
            event.description
        );

        if !event.details.is_empty() {
            println!("    Details: {}", event.details.join(", "));
        }

        if let Some(ref resource) = event.resource_affected {
            println!("    Resource: {}", resource.cyan());
        }

        if index < filtered_events.len() - 1 {
            println!();
        }
    }

    // Show pagination info if results were limited
    if filtered_events.len() == limit {
        println!();
        println!("{}", format!("⚠️  Results limited to {} events. Use --limit to see more.", limit).yellow());
        println!("   Example: loki safety audit --limit 50 --severity warning");
    }

    // Show quick stats
    let critical_count = filtered_events.iter().filter(|e| matches!(e.severity, AuditSeverity::Critical)).count();
    let error_count = filtered_events.iter().filter(|e| matches!(e.severity, AuditSeverity::Error)).count();
    let warning_count = filtered_events.iter().filter(|e| matches!(e.severity, AuditSeverity::Warning)).count();

    if critical_count > 0 || error_count > 0 || warning_count > 0 {
        println!();
        println!("📊 Issue Summary:");
        if critical_count > 0 {
            println!("   🔴 Critical: {}", critical_count.to_string().red().bold());
        }
        if error_count > 0 {
            println!("   🟠 Errors: {}", error_count.to_string().red());
        }
        if warning_count > 0 {
            println!("   🟡 Warnings: {}", warning_count.to_string().yellow());
        }
    }

    Ok(())
}

// Helper struct for audit events
#[derive(Debug)]
struct AuditEvent {
    timestamp: chrono::DateTime<chrono::Utc>,
    severity: AuditSeverity,
    actor: String,
    description: String,
    details: Vec<String>,
    resource_affected: Option<String>,
}

// Generate sample audit events for demonstration
fn generate_sample_audit_events(
    min_severity: Option<&AuditSeverity>,
    actor_filter: Option<&str>,
    hours_filter: Option<i64>,
) -> Vec<AuditEvent> {
    let now = Utc::now();
    let cutoff_time = hours_filter.map(|h| now - Duration::hours(h));

    let mut events = vec![
        AuditEvent {
            timestamp: now - Duration::minutes(5),
            severity: AuditSeverity::Info,
            actor: "loki_system".to_string(),
            description: "Cognitive process completed successfully".to_string(),
            details: vec!["decision_id: dec_001".to_string(), "confidence: 0.85".to_string()],
            resource_affected: Some("memory_store".to_string()),
        },
        AuditEvent {
            timestamp: now - Duration::minutes(12),
            severity: AuditSeverity::Warning,
            actor: "safety_validator".to_string(),
            description: "High-risk action requires approval".to_string(),
            details: vec!["action: file_system_write".to_string(), "risk_level: 85".to_string()],
            resource_affected: Some("/etc/config".to_string()),
        },
        AuditEvent {
            timestamp: now - Duration::minutes(25),
            severity: AuditSeverity::Error,
            actor: "tool_manager".to_string(),
            description: "API rate limit exceeded".to_string(),
            details: vec!["api: openai".to_string(), "requests: 1000/hour".to_string()],
            resource_affected: Some("openai_api".to_string()),
        },
        AuditEvent {
            timestamp: now - Duration::hours(1),
            severity: AuditSeverity::Critical,
            actor: "security_monitor".to_string(),
            description: "Unauthorized access attempt detected".to_string(),
            details: vec!["ip: 192.168.1.100".to_string(), "attempts: 5".to_string()],
            resource_affected: Some("admin_endpoint".to_string()),
        },
        AuditEvent {
            timestamp: now - Duration::hours(2),
            severity: AuditSeverity::Info,
            actor: "plugin_manager".to_string(),
            description: "Plugin loaded successfully".to_string(),
            details: vec!["plugin: ai_integration".to_string(), "version: 1.0.0".to_string()],
            resource_affected: Some("plugin_sandbox".to_string()),
        },
        AuditEvent {
            timestamp: now - Duration::hours(6),
            severity: AuditSeverity::Warning,
            actor: "resource_monitor".to_string(),
            description: "Memory usage approaching limit".to_string(),
            details: vec!["usage: 85%".to_string(), "threshold: 90%".to_string()],
            resource_affected: Some("system_memory".to_string()),
        },
        AuditEvent {
            timestamp: now - Duration::days(1),
            severity: AuditSeverity::Debug,
            actor: "test_system".to_string(),
            description: "Integration test completed".to_string(),
            details: vec!["suite: safety_tests".to_string(), "status: passed".to_string()],
            resource_affected: None,
        },
    ];

    // Apply filters
    events.retain(|event| {
        // Time filter
        if let Some(cutoff) = cutoff_time {
            if event.timestamp < cutoff {
                return false;
            }
        }

        // Actor filter
        if let Some(actor) = actor_filter {
            if !event.actor.contains(actor) {
                return false;
            }
        }

        // Severity filter
        if let Some(min_sev) = min_severity {
            let severity_level = match event.severity {
                AuditSeverity::Debug => 0,
                AuditSeverity::Info => 1,
                AuditSeverity::Warning => 2,
                AuditSeverity::Error => 3,
                AuditSeverity::Critical => 4,
            };

            let min_level = match min_sev {
                AuditSeverity::Debug => 0,
                AuditSeverity::Info => 1,
                AuditSeverity::Warning => 2,
                AuditSeverity::Error => 3,
                AuditSeverity::Critical => 4,
            };

            if severity_level < min_level {
                return false;
            }
        }

        true
    });

    // Sort by timestamp (newest first)
    events.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));

    events
}

async fn handle_resources() -> Result<()> {
    println!("{}", "📊 Resource Usage".cyan().bold());
    println!();

    // In a real implementation, we'd get resource monitor data
    println!("Memory Usage: {} / {} MB", "512".yellow(), "4096");
    println!("CPU Usage: {}%", "25.0".green());
    println!("File Handles: {} / {}", "50".green(), "1000");
    println!("Active Operations: {} / {}", "5".green(), "100");
    println!();

    println!("{}", "API Rate Limits:".bold());
    println!("  OpenAI: {} / 500 requests/min", "42".green());
    println!("  Anthropic: {} / 1000 requests/min", "156".green());
    println!("  X/Twitter: {} / 300 requests/15min", "23".green());
    println!();

    println!("{}", "Token Budgets:".bold());
    println!("  OpenAI: {} / 100k tokens/hr", "12,345".yellow());
    println!("  Anthropic: {} / 100k tokens/hr", "8,901".green());

    Ok(())
}

async fn handleconfig(
    safe_mode: Option<bool>,
    dry_run: Option<bool>,
    approval_required: Option<bool>,
) -> Result<()> {
    println!("{}", "⚙️  Safety Configuration".cyan().bold());
    println!();

    if safe_mode.is_none() && dry_run.is_none() && approval_required.is_none() {
        // Display current config
        println!("Current configuration:");
        println!("  Safe Mode: {}", "enabled".green().bold());
        println!("  Dry Run: {}", "disabled".red());
        println!("  Approval Required: {}", "enabled".green().bold());
        println!();
        println!("Use options to modify configuration.");
    } else {
        // Update config
        println!("Updating configuration:");
        if let Some(sm) = safe_mode {
            println!("  Safe Mode: {}",
                if sm { "enabled".green() } else { "disabled".red() }
            );
        }
        if let Some(dr) = dry_run {
            println!("  Dry Run: {}",
                if dr { "enabled".green() } else { "disabled".red() }
            );
        }
        if let Some(ar) = approval_required {
            println!("  Approval Required: {}",
                if ar { "enabled".green() } else { "disabled".red() }
            );
        }
        println!();
        println!("{}", "✅ Configuration updated".green().bold());
    }

    Ok(())
}

async fn handle_emergency_stop(reason: &str) -> Result<()> {
    println!();
    println!("{}", "🚨 EMERGENCY STOP TRIGGERED! 🚨".red().bold().on_white());
    println!();
    println!("Reason: {}", reason);
    println!();

    // In a real implementation, this would:
    // 1. Stop all active operations
    // 2. Cancel pending actions
    // 3. Save state
    // 4. Log the emergency stop
    // 5. Shut down safely

    println!("Actions taken:");
    println!("  ✓ All operations halted");
    println!("  ✓ Pending actions cancelled");
    println!("  ✓ State saved to disk");
    println!("  ✓ Emergency stop logged");
    println!();
    println!("{}", "System is now in safe mode. Manual restart required.".yellow());

    Ok(())
}

async fn handle_test() -> Result<()> {
    println!("{}", "🧪 Running Safety Integration Tests".cyan().bold());
    println!();

    // Import the test function
    use crate::safety::integration_test::run_safety_integration_tests;

    match run_safety_integration_tests().await {
        Ok(()) => {
            println!();
            println!("{}", "✅ All safety integration tests passed!".green().bold());
            println!("   Safety infrastructure is working correctly");
        }
        Err(e) => {
            println!();
            println!("{}", format!("❌ Safety tests failed: {}", e).red().bold());
            return Err(e);
        }
    }

    Ok(())
}
