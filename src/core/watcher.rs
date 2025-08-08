use std::path::Path;

use anyhow::Result;
use notify::{Config, Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use tokio::sync::mpsc;
use tracing::info;

use super::assistant::DefaultAssistant;

/// Watch a path for changes and trigger assistant actions
pub async fn watch_path(path: &Path, assistant: DefaultAssistant, _interactive: bool) -> Result<()> {

    let (tx, mut rx) = mpsc::channel(100);

    // Create watcher with async channel
    let mut watcher = RecommendedWatcher::new(
        move |res: Result<Event, notify::Error>| {
            if let Ok(event) = res {
                let _ = tx.blocking_send(event);
            }
        },
        Config::default(),
    )?;

    // Start watching
    watcher.watch(path, RecursiveMode::Recursive)?;

    println!("👀 Watching for changes... (Press Ctrl+C to stop)");

    // Handle events
    while let Some(event) = rx.recv().await {
        match event.kind {
            EventKind::Create(_) | EventKind::Modify(_) => {
                for path in event.paths {
                    if should_process_file(&path) {
                        info!("File changed: {:?}", path);
                        handle_file_change(&path, &assistant).await?;
                    }
                }
            }
            _ => {} // Ignore other events
        }
    }

    Ok(())
}

/// Check if a file should be processed
fn should_process_file(path: &Path) -> bool {
    // Skip hidden files and directories
    if let Some(name) = path.file_name() {
        if name.to_string_lossy().starts_with('.') {
            return false;
        }
    }

    // Skip common ignore patterns
    let path_str = path.to_string_lossy();
    if path_str.contains("/target/")
        || path_str.contains("/node_modules/")
        || path_str.contains("/.git/")
        || path_str.contains("/dist/")
        || path_str.contains("/build/")
    {
        return false;
    }

    // Only process text files
    matches!(
        path.extension().and_then(|e| e.to_str()),
        Some("rs" | "py" | "js" | "ts" | "java" | "go" | "cpp" | "c" | "h" | "toml" | "json")
    )
}

/// Handle a file change event
async fn handle_file_change(path: &Path, assistant: &DefaultAssistant) -> Result<()> {
    println!("\n📝 File changed: {}", path.display());

    // Analyze the changed file
    let analysis = assistant.analyze_path(path).await?;

    // Show quick summary
    if !analysis.issues.is_empty() {
        println!("⚠️  Found {} issue(s):", analysis.issues.len());
        for issue in analysis.issues.iter().take(3) {
            println!(
                "   - {}: {}",
                match issue.severity {
                    super::analyzer::Severity::Error => "ERROR",
                    super::analyzer::Severity::Warning => "WARN",
                    super::analyzer::Severity::Info => "INFO",
                },
                issue.message
            );
        }
        if analysis.issues.len() > 3 {
            println!("   ... and {} more", analysis.issues.len() - 3);
        }
    } else {
        println!("✅ No issues found");
    }

    println!(); // Empty line for readability
    Ok(())
}
