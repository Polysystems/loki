pub mod async_optimization;
pub mod fs;
pub mod progress;
pub mod syntax;

#[cfg(feature = "stub_tracking")]
pub mod stub_tracking;

use std::time::{Duration, Instant};

/// Format a duration in human-readable format
pub fn format_duration(duration: Duration) -> String {
    let secs = duration.as_secs();
    let millis = duration.subsec_millis();

    if secs == 0 {
        format!("{}ms", millis)
    } else if secs < 60 {
        format!("{}.{:03}s", secs, millis)
    } else if secs < 3600 {
        let mins = secs / 60;
        let secs = secs % 60;
        format!("{}m {}s", mins, secs)
    } else {
        let hours = secs / 3600;
        let mins = (secs % 3600) / 60;
        format!("{}h {}m", hours, mins)
    }
}

/// Simple timer for measuring elapsed time
pub struct Timer {
    start: Instant,
}

impl Timer {
    pub fn new() -> Self {
        Self { start: Instant::now() }
    }

    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    pub fn elapsed_formatted(&self) -> String {
        format_duration(self.elapsed())
    }
}

impl Default for Timer {
    fn default() -> Self {
        Self::new()
    }
}
