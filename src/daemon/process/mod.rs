pub mod utils;

use std::path::PathBuf;
use std::process;

use anyhow::{Context, Result};
#[cfg(unix)]
use nix::unistd::{fork, setsid, ForkResult};
use serde::{Deserialize, Serialize};
use tokio::signal;
use tracing::{error, info, warn};

use super::{DaemonConfig, DaemonServer};
// Import from the internal cognitive module
use crate::cognitive::CognitiveSystem;

/// Process status for the daemon
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ProcessStatus {
    /// Process is starting up
    Starting,
    /// Process is running normally
    Running,
    /// Process is shutting down
    Stopping,
    /// Process has stopped
    Stopped,
    /// Process encountered an error
    Error { message: String },
}

/// Daemon process manager
pub struct DaemonProcess {
    config: DaemonConfig,
    server: Option<DaemonServer>,
}

impl DaemonProcess {
    /// Create a new daemon process
    pub fn new(config: DaemonConfig) -> Self {
        Self { config, server: None }
    }

    /// Start the daemon process
    pub async fn start(
        &mut self,
        cognitive_system: Option<std::sync::Arc<CognitiveSystem>>,
    ) -> Result<()> {
        info!("Starting Loki daemon process");

        // Check if daemon is already running
        if super::utils::is_daemon_running(&self.config).await {
            anyhow::bail!("Daemon is already running");
        }

        // Create PID file
        super::utils::create_pid_file(&self.config).await.context("Failed to create PID file")?;

        // Create and configure daemon server
        let mut server = DaemonServer::new(self.config.clone())?;

        if let Some(cognitive_system) = cognitive_system {
            server = server.with_cognitive_system(cognitive_system);
        }

        // Setup signal handlers
        self.setup_signal_handlers(&server).await?;

        // Start the server
        server.start().await?;

        // Clean up on exit
        self.cleanup().await?;

        Ok(())
    }

    /// Detach from the terminal (daemonize)
    async fn detach(&self) -> Result<()> {
        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;

            // First fork
            match unsafe { fork() }.context("First fork failed")? {
                ForkResult::Parent { .. } => {
                    // Parent process exits
                    std::process::exit(0);
                }
                ForkResult::Child => {
                    // Child continues
                }
            }

            // Create new session
            setsid().context("Failed to create new session")?;

            // Second fork to ensure we can't acquire a controlling terminal
            match unsafe { fork() }.context("Second fork failed")? {
                ForkResult::Parent { .. } => {
                    // First child exits
                    std::process::exit(0);
                }
                ForkResult::Child => {
                    // Second child continues as daemon
                }
            }

            // Change to root directory to avoid locking filesystems
            std::env::set_current_dir("/").context("Failed to change to root directory")?;

            // Redirect stdin, stdout, stderr to /dev/null
            let dev_null = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open("/dev/null")
                .context("Failed to open /dev/null")?;

            let fd = dev_null.as_raw_fd();

            unsafe {
                libc::dup2(fd, libc::STDIN_FILENO);
                libc::dup2(fd, libc::STDOUT_FILENO);
                libc::dup2(fd, libc::STDERR_FILENO);
            }

            info!("Successfully detached from terminal");
        }

        #[cfg(not(unix))]
        {
            warn!("Process detachment not supported on this platform");
        }

        Ok(())
    }

    /// Setup signal handlers for graceful shutdown
    async fn setup_signal_handlers(&self, server: &DaemonServer) -> Result<()> {
        let shutdown_tx = server.shutdown_sender();

        tokio::spawn(async move {
            #[cfg(unix)]
            {
                use signal::unix::{SignalKind, signal};

                let mut sigterm =
                    signal(SignalKind::terminate()).expect("Failed to register SIGTERM handler");
                let mut sigint =
                    signal(SignalKind::interrupt()).expect("Failed to register SIGINT handler");
                let mut sighup =
                    signal(SignalKind::hangup()).expect("Failed to register SIGHUP handler");

                tokio::select! {
                    _ = sigterm.recv() => {
                        info!("Received SIGTERM, shutting down gracefully");
                    }
                    _ = sigint.recv() => {
                        info!("Received SIGINT, shutting down gracefully");
                    }
                    _ = sighup.recv() => {
                        info!("Received SIGHUP, shutting down gracefully");
                    }
                }
            }

            #[cfg(not(unix))]
            {
                // Windows Ctrl+C handling
                match signal::ctrl_c().await {
                    Ok(()) => {
                        info!("Received Ctrl+C, shutting down gracefully");
                    }
                    Err(err) => {
                        error!("Failed to listen for Ctrl+C: {}", err);
                    }
                }
            }

            // Send shutdown signal
            if let Err(e) = shutdown_tx.send(()) {
                error!("Error sending shutdown signal: {}", e);
            }
        });

        Ok(())
    }

    /// Clean up resources on exit
    async fn cleanup(&self) -> Result<()> {
        info!("Cleaning up daemon resources");

        // Remove PID file
        super::utils::remove_pid_file(&self.config).await.context("Failed to remove PID file")?;

        // Remove socket file if it exists
        if self.config.socket_path.exists() {
            tokio::fs::remove_file(&self.config.socket_path)
                .await
                .context("Failed to remove socket file")?;
        }

        info!("Daemon cleanup complete");
        Ok(())
    }

    /// Stop the daemon process
    pub async fn stop(&mut self) -> Result<()> {
        if let Some(server) = &self.server {
            server.stop().await?;
        }
        self.cleanup().await?;
        Ok(())
    }

    /// Get the current process ID
    pub fn pid(&self) -> u32 {
        process::id()
    }

    /// Check if the process is running
    pub async fn is_running(&self) -> bool {
        super::utils::is_daemon_running(&self.config).await
    }
}
