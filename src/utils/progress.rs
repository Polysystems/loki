use std::io::{self, Write};
use std::sync::Arc;

use tokio::sync::Mutex;

/// Simple progress indicator
pub struct Progress {
    message: String,
    current: Arc<Mutex<usize>>,
    total: usize,
}

impl Progress {
    /// Create a new progress indicator
    pub fn new(message: String, total: usize) -> Self {
        Self { message, current: Arc::new(Mutex::new(0)), total }
    }

    /// Update progress
    pub async fn update(&self, amount: usize) {
        let mut current = self.current.lock().await;
        *current = (*current + amount).min(self.total);
        self.display(*current);
    }

    /// Set progress to a specific value
    pub async fn set(&self, value: usize) {
        let mut current = self.current.lock().await;
        *current = value.min(self.total);
        self.display(*current);
    }

    /// Display progress
    fn display(&self, current: usize) {
        let percentage =
            if self.total > 0 { (current as f64 / self.total as f64 * 100.0) as u8 } else { 0 };

        print!("\r{}: [", self.message);

        let bar_width = 30;
        let filled = (bar_width * current / self.total.max(1)).min(bar_width);

        for i in 0..bar_width {
            if i < filled {
                print!("█");
            } else {
                print!("░");
            }
        }

        print!("] {}%", percentage);
        io::stdout().flush().unwrap();

        if current >= self.total {
            println!();
        }
    }

    /// Finish the progress
    pub async fn finish(&self) {
        self.set(self.total).await;
    }
}

/// Spinner for indeterminate progress
pub struct Spinner {
    message: String,
    frames: Vec<&'static str>,
    current: usize,
}

impl Spinner {
    /// Create a new spinner
    pub fn new(message: String) -> Self {
        Self {
            message, frames: vec!["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"], current: 0
        }
    }

    /// Update the spinner
    pub fn tick(&mut self) {
        print!("\r{} {} ", self.frames[self.current], self.message);
        io::stdout().flush().unwrap();
        self.current = (self.current + 1) % self.frames.len();
    }

    /// Stop the spinner with a message
    pub fn stop_with_message(&self, message: &str) {
        print!("\r{}\n", message);
        io::stdout().flush().unwrap();
    }
}
