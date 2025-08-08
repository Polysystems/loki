use nix::sys::signal::{kill, Signal};
use nix::unistd::Pid;
use super::*;

/// Kill a process by PID
pub async fn kill_process(pid: u32) -> Result<()> {
    #[cfg(unix)]
    {

        let pid = Pid::from_raw(pid as i32);

        // Try TERM first, then KILL
        match kill(pid, Signal::SIGTERM) {
            Ok(()) => {
                info!("Sent SIGTERM to process {}", pid);

                // Wait a bit for graceful shutdown
                tokio::time::sleep(std::time::Duration::from_secs(5)).await;

                // Check if process is still running
                if kill(pid, None).is_ok() {
                    // Still running, force kill
                    kill(pid, Signal::SIGKILL).context("Failed to force kill process")?;
                    info!("Force killed process {}", pid);
                }
            }
            Err(e) => {
                warn!("Failed to send SIGTERM to process {}: {}", pid, e);
                // Try force kill directly
                kill(pid, Signal::SIGKILL).context("Failed to force kill process")?;
            }
        }
    }

    #[cfg(not(unix))]
    {
        // Windows process termination
        warn!("Process termination not implemented for this platform");
    }

    Ok(())
}

/// Read PID from PID file
pub async fn read_pid_file(pid_file: &PathBuf) -> Result<Option<u32>> {
    if !pid_file.exists() {
        return Ok(None);
    }

    let content = tokio::fs::read_to_string(pid_file).await.context("Failed to read PID file")?;

    let pid: u32 = content.trim().parse().context("Invalid PID in file")?;

    Ok(Some(pid))
}

/// Check if a process is running by PID
pub fn is_process_running(pid: u32) -> bool {
    #[cfg(unix)]
    {

        let pid = Pid::from_raw(pid as i32);
        kill(pid, None).is_ok()
    }

    #[cfg(not(unix))]
    {
        // Windows process check would go here
        false
    }
}
