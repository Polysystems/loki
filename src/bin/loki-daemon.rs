use std::env;

use anyhow::Result;
// Import from the loki library
use loki::daemon::{DaemonConfig, DaemonProcess};
use tracing::Level;
use tracing_subscriber::prelude::*;
use tracing_subscriber::{EnvFilter, fmt};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    let env_filter =
        EnvFilter::builder().with_default_directive(Level::INFO.into()).from_env_lossy();

    tracing_subscriber::registry()
        .with(fmt::layer().with_target(true).with_thread_ids(true))
        .with(env_filter)
        .init();

    // Parse command line arguments
    let args: Vec<String> = env::args().collect();

    // Create daemon configuration
    let config = DaemonConfig::default();

    // Create and start daemon process
    let mut daemon_process = DaemonProcess::new(config);

    match args.get(1).map(|s| s.as_str()) {
        Some("stop") => {
            daemon_process.stop().await?;
        }
        Some("status") => {
            if daemon_process.is_running().await {
                println!("Daemon is running");
            } else {
                println!("Daemon is not running");
            }
        }
        _ => {
            daemon_process.start(None).await?;
        }
    }

    Ok(())
}
