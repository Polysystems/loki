//! Cognitive Stream Integration for TUI Chat
//!
//! Enables 24/7 cognitive processing in the background,
//! surfacing insights and awareness to the chat interface.

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::sync::{RwLock, broadcast, mpsc};
use tokio::task::JoinHandle;
use tracing::{debug, info, warn};

use crate::cognitive::consciousness_stream::{
    ConsciousnessConfig as CognitiveConfig,
    ConsciousnessInsight as CognitiveInsight,
    ThermodynamicConsciousnessEvent as ThermodynamicCognitiveEvent,
    ThermodynamicConsciousnessStream as ThermodynamicCognitiveStream,
};
use crate::memory::CognitiveMemory;

/// Cognitive activity for UI display
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveActivity {
    pub awareness_level: f64,
    pub active_insights: Vec<String>,
    pub gradient_coherence: f64,
    pub free_energy: f64,
    pub current_focus: String,
    pub background_thoughts: usize,
}

/// Cognitive stream manager for chat interface
pub struct ChatCognitiveStream {
    /// Channel for chat-relevant insights
    insight_tx: mpsc::Sender<CognitiveInsight>,
    insight_rx: Arc<RwLock<mpsc::Receiver<CognitiveInsight>>>,

    /// Current cognitive activity
    current_activity: Arc<RwLock<CognitiveActivity>>,

    /// Background processing handle
    processing_handle: Option<JoinHandle<()>>,

    /// Whether stream is active
    is_active: Arc<RwLock<bool>>,
}

impl ChatCognitiveStream {
    /// Create new cognitive stream for chat
    pub async fn new(memory: Arc<CognitiveMemory>) -> Result<Self> {
        info!("🌊 Initializing Chat Cognitive Stream");

        // Create cognitive stream with chat-optimized config
        let config = CognitiveConfig {
            stream_interval: Duration::from_secs(1), // Fast updates for UI
            max_event_history: 100,
            awareness_threshold: 0.6, // Medium threshold
            generate_insights: true,
            insight_depth: 0.7,
            generate_narrative: true,
            thermodynamic_sensitivity: 0.8,
            max_narrative_length: 1000,
        };

        // Create insight channel
        let (insight_tx, insight_rx) = mpsc::channel(50);

        // Initialize activity
        let current_activity = Arc::new(RwLock::new(CognitiveActivity {
            awareness_level: 0.0,
            active_insights: Vec::new(),
            gradient_coherence: 0.0,
            free_energy: 1.0,
            current_focus: "Initializing...".to_string(),
            background_thoughts: 0,
        }));

        Ok(Self {
            insight_tx,
            insight_rx: Arc::new(RwLock::new(insight_rx)),
            current_activity,
            processing_handle: None,
            is_active: Arc::new(RwLock::new(false)),
        })
    }

    /// Start cognitive stream processing
    pub async fn start(&mut self) -> Result<()> {
        if *self.is_active.read().await {
            return Ok(());
        }

        info!("🚀 Starting cognitive stream processing");

        *self.is_active.write().await = true;

        Ok(())
    }

    /// Stop cognitive stream
    pub async fn stop(&mut self) -> Result<()> {
        info!("🛑 Stopping cognitive stream");

        *self.is_active.write().await = false;

        if let Some(handle) = self.processing_handle.take() {
            handle.abort();
        }

        Ok(())
    }

    /// Get current cognitive activity
    pub async fn get_activity(&self) -> CognitiveActivity {
        self.current_activity.read().await.clone()
    }

    /// Get recent insights relevant to chat
    pub async fn get_recent_insights(&self, count: usize) -> Vec<CognitiveInsight> {
        let mut insights = Vec::new();
        let mut rx = self.insight_rx.write().await;

        // Collect available insights (non-blocking)
        while insights.len() < count {
            match rx.try_recv() {
                Ok(insight) => insights.push(insight),
                Err(_) => break,
            }
        }

        insights
    }

    /// Interrupt cognitive with high-priority input
    pub async fn interrupt(&self, source: &str, content: &str) -> Result<()> {
        Ok(())
    }

    /// Check if specific cognitive capability is active
    pub async fn is_capability_active(&self, capability: &str) -> bool {
        let activity = self.current_activity.read().await;
        match capability {
            "awareness" => activity.awareness_level > 0.5,
            "insights" => !activity.active_insights.is_empty(),
            "coherence" => activity.gradient_coherence > 0.6,
            "focus" => activity.free_energy < 0.5,
            _ => false,
        }
    }
}

/// Helper to get description from CognitiveInsight
pub fn format_cognitive_insight(insight: &CognitiveInsight) -> String {
    match insight {
        CognitiveInsight::GoalAwareness { active_goals, completion_rate, .. } => {
            format!(
                "Goals: {} ({}% complete)",
                active_goals.join(", "),
                (completion_rate * 100.0) as u32
            )
        }
        CognitiveInsight::LearningAwareness { knowledge_gained, .. } => {
            format!("Learning: {}", knowledge_gained)
        }
        CognitiveInsight::SocialAwareness { harmony_state, .. } => {
            format!("Social: {}", harmony_state)
        }
        CognitiveInsight::CreativeInsight { pattern_discovered, .. } => {
            format!("Creative: {}", pattern_discovered)
        }
        CognitiveInsight::SelfReflection { self_model_update, .. } => {
            format!("Reflection: {}", self_model_update)
        }
        CognitiveInsight::ThermodynamicAwareness {
            entropy_management, energy_efficiency, ..
        } => {
            format!(
                "Thermodynamic: entropy {:.0}%, efficiency {:.0}%",
                entropy_management * 100.0,
                energy_efficiency * 100.0
            )
        }
        CognitiveInsight::TemporalAwareness { future_planning, .. } => {
            format!("Temporal: {}", future_planning)
        }
    }
}

/// Cognitive mode for chat interface
#[derive(Debug, Clone, PartialEq)]
pub enum CognitiveMode {
    /// Minimal cognitive - fast responses
    Minimal,
    /// Standard cognitive - balanced
    Standard,
    /// Deep cognitive - profound insights
    Deep,
    /// Continuous - 24/7 stream
    Continuous,
}

impl std::fmt::Display for CognitiveMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CognitiveMode::Minimal => write!(f, "Minimal"),
            CognitiveMode::Standard => write!(f, "Standard"),
            CognitiveMode::Deep => write!(f, "Deep"),
            CognitiveMode::Continuous => write!(f, "Continuous"),
        }
    }
}

/// Integrate cognitive insights into chat response
pub async fn enhance_with_cognitive(
    response: &mut String,
    cognitive_stream: &ChatCognitiveStream,
    mode: CognitiveMode,
) -> Result<()> {
    match mode {
        CognitiveMode::Minimal => {
            // Just add awareness indicator
            let activity = cognitive_stream.get_activity().await;
            if activity.awareness_level > 0.7 {
                *response =
                    format!("[Awareness: {:.0}%] {}", activity.awareness_level * 100.0, response);
            }
        }

        CognitiveMode::Standard => {
            // Add recent insight if relevant
            let insights = cognitive_stream.get_recent_insights(1).await;
            if let Some(insight) = insights.first() {
                *response =
                    format!("{}\n\n💭 Insight: {}", response, format_cognitive_insight(insight));
            }
        }

        CognitiveMode::Deep => {
            let insights = cognitive_stream.get_recent_insights(3).await;


            if !insights.is_empty() {
                *response = format!("{}\n\n💭 Active Insights:", response);
                for insight in insights {
                    *response = format!("{}\n• {}", response, format_cognitive_insight(&insight));
                }
            }
        }

        CognitiveMode::Continuous => {
            // Full cognitive integration
            let activity = cognitive_stream.get_activity().await;
            let insights = cognitive_stream.get_recent_insights(5).await;

            *response = format!(
                "🧠 Cognitive State:\n• Awareness: {:.0}%\n• Coherence: {:.0}%\n• Focus: {}\n• \
                 Background Thoughts: {}\n\n{}",
                activity.awareness_level * 100.0,
                activity.gradient_coherence * 100.0,
                activity.current_focus,
                activity.background_thoughts,
                response
            );

            if !insights.is_empty() {
                *response = format!("{}\n\n🌟 Cognitive Insights:", response);
                for insight in insights.iter().take(3) {
                    *response = format!("{}\n• {}", response, format_cognitive_insight(&insight));
                }
            }
        }
    }

    Ok(())
}

/// Create cognitive activity display for UI
pub fn format_cognitive_status(activity: &CognitiveActivity) -> String {
    let awareness_bar = create_progress_bar(activity.awareness_level);
    let coherence_bar = create_progress_bar(activity.gradient_coherence);

    format!(
        "🧠 Cognitive: {} {:.0}% | Coherence: {} {:.0}% | Focus: {}",
        awareness_bar,
        activity.awareness_level * 100.0,
        coherence_bar,
        activity.gradient_coherence * 100.0,
        activity.current_focus
    )
}

fn create_progress_bar(value: f64) -> String {
    let filled = (value * 10.0) as usize;
    let empty = 10 - filled;
    format!("{}{}", "█".repeat(filled), "░".repeat(empty))
}
