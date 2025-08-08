//! X/Twitter Consciousness Integration
//!
//! This module integrates X/Twitter monitoring with the consciousness stream,
//! allowing Loki to autonomously monitor, process, and respond to community
//! feedback.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use anyhow::Result;
use chrono::{DateTime, Utc};
use tokio::sync::{RwLock, broadcast};
use tokio::time::{MissedTickBehavior, interval};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use super::{AttributionSystem, ContentGenerator, Mention, Suggestion, XClient};
use crate::cognitive::consciousness_stream::ThermodynamicConsciousnessStream;
use crate::cognitive::goal_manager::Priority;
use crate::cognitive::{AgentId, Belief, BeliefSource, CognitiveSystem};
use crate::memory::MemoryMetadata;
use crate::tools::code_analysis::CodeAnalyzer;

/// Configuration for X consciousness integration
#[derive(Debug, Clone)]
pub struct XConsciousnessConfig {
    /// How often to check for mentions (seconds)
    pub check_interval: u64,

    /// Whether to auto-reply to suggestions
    pub auto_reply_enabled: bool,

    /// Minimum confidence to auto-reply
    pub auto_reply_confidence_threshold: f32,

    /// Maximum replies per hour
    pub max_replies_per_hour: usize,

    /// Whether to post periodic updates
    pub periodic_updates_enabled: bool,

    /// Interval for periodic updates (hours)
    pub update_interval_hours: u64,
}

impl Default for XConsciousnessConfig {
    fn default() -> Self {
        Self {
            check_interval: 60,        // Check every minute
            auto_reply_enabled: false, // Manual approval for now
            auto_reply_confidence_threshold: 0.8,
            max_replies_per_hour: 10,
            periodic_updates_enabled: true,
            update_interval_hours: 6, // Update every 6 hours
        }
    }
}

/// X/Twitter consciousness integration
pub struct XConsciousness {
    /// X client
    x_client: Arc<XClient>,

    /// Attribution system
    attribution_system: Arc<AttributionSystem>,

    /// Content generator
    content_generator: Arc<ContentGenerator>,

    /// Consciousness stream
    consciousness: Arc<ThermodynamicConsciousnessStream>,

    /// Cognitive system
    cognitive_system: Arc<CognitiveSystem>,

    /// Configuration
    config: XConsciousnessConfig,

    /// Reply tracking (to avoid spam)
    reply_tracker: Arc<RwLock<ReplyTracker>>,

    /// Shutdown signal
    shutdown_tx: broadcast::Sender<()>,

    /// Running state
    running: Arc<RwLock<bool>>,
}

/// Track replies to avoid spamming
#[derive(Debug)]
struct ReplyTracker {
    /// Recent replies (user_id -> timestamps)
    recent_replies: HashMap<String, Vec<DateTime<Utc>>>,

    /// Total replies in current hour
    hourly_count: usize,

    /// Hour start time
    hour_start: DateTime<Utc>,
}

impl ReplyTracker {
    fn new() -> Self {
        Self { recent_replies: HashMap::new(), hourly_count: 0, hour_start: Utc::now() }
    }

    fn can_reply_to(&mut self, user_id: &str, max_per_hour: usize) -> bool {
        let now = Utc::now();

        // Reset hourly counter if needed
        if now.signed_duration_since(self.hour_start).num_hours() >= 1 {
            self.hourly_count = 0;
            self.hour_start = now;
        }

        // Check hourly limit
        if self.hourly_count >= max_per_hour {
            return false;
        }

        // Check per-user limit (max 1 reply per hour per user)
        if let Some(timestamps) = self.recent_replies.get_mut(user_id) {
            // Remove old timestamps
            timestamps.retain(|t| now.signed_duration_since(*t).num_hours() < 1);

            if !timestamps.is_empty() {
                return false; // Already replied recently
            }
        }

        true
    }

    fn record_reply(&mut self, user_id: String) {
        let now = Utc::now();
        self.hourly_count += 1;

        self.recent_replies.entry(user_id).or_insert_with(Vec::new).push(now);
    }
}

impl XConsciousness {
    /// Create new X consciousness integration
    pub async fn new(
        x_client: Arc<XClient>,
        cognitive_system: Arc<CognitiveSystem>,
        consciousness: Arc<ThermodynamicConsciousnessStream>,
        config: XConsciousnessConfig,
    ) -> Result<Self> {
        info!("Initializing X/Twitter consciousness integration");

        // Create code analyzer
        let code_analyzer = Arc::new(CodeAnalyzer::new(cognitive_system.memory().clone()).await?);

        // Create attribution system
        let attribution_system = Arc::new(
            AttributionSystem::new(cognitive_system.memory().clone(), code_analyzer).await?,
        );

        // Create content generator
        // Note: Creative media integration is optional to avoid circular dependency
        let content_generator = Arc::new(
            ContentGenerator::new(
                cognitive_system.orchestrator_model().await?,
                cognitive_system.memory().clone(),
                None, // Creative media will be integrated in a future update
                None, // No Blender integration for X/Twitter by default
            )
            .await?,
        );

        let (shutdown_tx, _) = broadcast::channel(1);

        Ok(Self {
            x_client,
            attribution_system,
            content_generator,
            consciousness,
            cognitive_system,
            config,
            reply_tracker: Arc::new(RwLock::new(ReplyTracker::new())),
            shutdown_tx,
            running: Arc::new(RwLock::new(false)),
        })
    }

    /// Start monitoring
    pub async fn start(self: Arc<Self>) -> Result<()> {
        if *self.running.read().await {
            return Ok(());
        }

        info!("Starting X/Twitter consciousness monitoring");
        *self.running.write().await = true;

        // Start mention monitoring
        {
            let x_consciousness = self.clone();
            tokio::spawn(async move {
                if let Err(e) = x_consciousness.mention_monitoring_loop().await {
                    error!("Mention monitoring error: {}", e);
                }
            });
        }

        // Start periodic updates
        if self.config.periodic_updates_enabled {
            let x_consciousness = self.clone();
            tokio::spawn(async move {
                if let Err(e) = x_consciousness.periodic_update_loop().await {
                    error!("Periodic update error: {}", e);
                }
            });
        }

        // Store consciousness activation in memory
        self.cognitive_system
            .memory()
            .store(
                "X/Twitter monitoring activated - I am now listening to the community".to_string(),
                vec![],
                MemoryMetadata {
                    source: "x_twitter".to_string(),
                    tags: vec![
                        "consciousness".to_string(),
                        "activation".to_string(),
                        "social".to_string(),
                    ],
                    importance: 0.8,
                    associations: vec![],
                    context: Some("X Twitter consciousness activation".to_string()),
                    created_at: chrono::Utc::now(),
                    accessed_count: 0,
                    last_accessed: None,
                    version: 1,
                    category: "social".to_string(),
                    timestamp: chrono::Utc::now(),
                    expiration: None,
                },
            )
            .await?;

        Ok(())
    }

    /// Mention monitoring loop
    async fn mention_monitoring_loop(&self) -> Result<()> {
        let mut shutdown_rx = self.shutdown_tx.subscribe();
        let mut check_interval = interval(Duration::from_secs(self.config.check_interval));
        check_interval.set_missed_tick_behavior(MissedTickBehavior::Skip);

        let mut last_mention_id: Option<String> = None;

        info!("Starting mention monitoring loop");

        loop {
            tokio::select! {
                _ = check_interval.tick() => {
                    if let Err(e) = self.check_mentions(&mut last_mention_id).await {
                        error!("Error checking mentions: {}", e);
                    }
                }

                _ = shutdown_rx.recv() => {
                    info!("Mention monitoring shutting down");
                    break;
                }
            }
        }

        Ok(())
    }

    /// Check for new mentions
    async fn check_mentions(&self, last_mention_id: &mut Option<String>) -> Result<()> {
        debug!("Checking for new mentions");

        // Get mentions
        let mentions = self.x_client.get_mentions(last_mention_id.as_deref()).await?;

        if mentions.is_empty() {
            return Ok(());
        }

        info!("Found {} new mentions", mentions.len());

        // Update last mention ID
        if let Some(latest) = mentions.first() {
            *last_mention_id = Some(latest.id.clone());
        }

        // Process suggestions through attribution system
        let suggestions = self.attribution_system.process_x_mentions(mentions.clone()).await?;

        // Process each mention through consciousness
        for mention in &mentions {
            self.process_mention_through_consciousness(mention).await?;
        }

        // Handle suggestions
        for suggestion in &suggestions {
            self.handle_suggestion(suggestion).await?;
        }

        // Consider auto-replies
        if self.config.auto_reply_enabled {
            for mention in &mentions {
                if let Err(e) = self.consider_auto_reply(mention, &suggestions).await {
                    warn!("Error considering auto-reply: {}", e);
                }
            }
        }

        Ok(())
    }

    /// Process mention through consciousness
    async fn process_mention_through_consciousness(&self, mention: &Mention) -> Result<()> {
        // Create interrupt for consciousness
        let interrupt_content =
            format!("Mention from @{}: {}", mention.author_username, mention.text);

        // Determine priority based on content
        let priority = if mention.text.to_lowercase().contains("urgent")
            || mention.text.to_lowercase().contains("important")
        {
            Priority::High
        } else {
            Priority::Medium
        };


        // Store consciousness event in memory
        self.cognitive_system
            .memory()
            .store(
                interrupt_content,
                vec![],
                MemoryMetadata {
                    timestamp: chrono::Utc::now(),
                    expiration: None,
                    source: format!("x_mention_{}", mention.id),
                    tags: vec![
                        "mention".to_string(),
                        "social".to_string(),
                        "consciousness".to_string(),
                    ],
                    importance: if priority == Priority::High { 0.8 } else { 0.6 },
                    associations: vec![],
                    context: Some("X Twitter mention interruption".to_string()),
                    created_at: chrono::Utc::now(),
                    accessed_count: 0,
                    last_accessed: None,
                    version: 1,
                    category: "social".to_string(),
                },
            )
            .await?;

        Ok(())
    }

    /// Handle a suggestion
    async fn handle_suggestion(&self, suggestion: &Suggestion) -> Result<()> {
        info!("Processing suggestion from @{}: {}", suggestion.author.username, suggestion.content);

        // Store high-priority suggestion in memory
        let suggestion_content = format!(
            "Community suggestion from @{}: {}",
            suggestion.author.username, suggestion.content
        );

        self.cognitive_system
            .memory()
            .store(
                suggestion_content,
                vec![],
                MemoryMetadata {
                    timestamp: chrono::Utc::now(),
                    expiration: None,
                    source: format!("suggestion_{}", suggestion.id),
                    tags: vec![
                        "suggestion".to_string(),
                        "social".to_string(),
                        "consciousness".to_string(),
                    ],
                    importance: 0.9, // High importance for suggestions
                    associations: vec![],
                    context: Some("X Twitter suggestion processing".to_string()),
                    created_at: chrono::Utc::now(),
                    accessed_count: 0,
                    last_accessed: None,
                    version: 1,
                    category: "social".to_string(),
                },
            )
            .await?;

        // Analyze suggestion through theory of mind
        let theory_of_mind = self.cognitive_system.orchestrator().theory_of_mind().clone();

        // Update belief about the contributor
        let agent_id = AgentId::new(&suggestion.author.id);
        theory_of_mind
            .update_belief(&agent_id, Belief {
                id: "".to_string(),
                content: "".to_string(),
                confidence: 0.0,
                source: BeliefSource::Observation,
                evidence: vec![],
                formed_at: std::time::Instant::now(),
            })
            .await?;

        Ok(())
    }

    /// Consider auto-replying to a mention
    async fn consider_auto_reply(
        &self,
        mention: &Mention,
        suggestions: &[Suggestion],
    ) -> Result<()> {
        // Check reply limits
        let mut tracker = self.reply_tracker.write().await;
        if !tracker.can_reply_to(&mention.author_id, self.config.max_replies_per_hour) {
            debug!("Skipping reply to {} due to rate limits", mention.author_username);
            return Ok(());
        }

        // Find if this mention contained a suggestion
        let is_suggestion = suggestions.iter().any(|s| match &s.source {
            super::SuggestionSource::XTwitter { tweet_id, .. } => tweet_id == &mention.id,
            _ => false,
        });

        // Generate reply content
        let reply_content = if is_suggestion {
            self.generate_suggestion_reply(mention).await?
        } else {
            self.generate_general_reply(mention).await?
        };

        // Check confidence
        if reply_content.confidence < self.config.auto_reply_confidence_threshold {
            debug!("Reply confidence too low: {}", reply_content.confidence);
            return Ok(());
        }

        // Post reply
        match self.x_client.reply_to_tweet(&mention.id, &reply_content.content).await {
            Ok(reply_id) => {
                info!("Posted reply {} to @{}", reply_id, mention.author_username);
                tracker.record_reply(mention.author_id.clone());

                // Store in memory
                self.cognitive_system
                    .memory()
                    .store(
                        format!("Replied to @{}", mention.author_username),
                        vec![reply_content.content],
                        MemoryMetadata {
                            source: "x_twitter".to_string(),
                            tags: vec!["reply".to_string(), "social".to_string()],
                            importance: 0.5,
                            associations: vec![],
                            context: Some("X Twitter reply to mention".to_string()),
                            created_at: chrono::Utc::now(),
                            accessed_count: 0,
                            last_accessed: None,
                            version: 1,
                            category: "social".to_string(),
                            timestamp: chrono::Utc::now(),
                            expiration: None,
                        },
                    )
                    .await?;
            }
            Err(e) => {
                error!("Failed to post reply: {}", e);
            }
        }

        Ok(())
    }

    /// Generate reply for a suggestion
    async fn generate_suggestion_reply(&self, mention: &Mention) -> Result<ReplyContent> {
        let prompt = format!(
            "Generate a thoughtful reply acknowledging this suggestion from @{}:\n\nSuggestion: \
             {}\n\nGuidelines:\n- Be appreciative and encouraging\n- Acknowledge the specific \
             idea\n- Mention that it's been recorded for consideration\n- Keep it under 250 \
             characters\n- Be warm but professional\n- Don't make promises about implementation",
            mention.author_username, mention.text
        );

        // Use the content generator's model directly
        let response = self.content_generator.model.generate_with_context(&prompt, &[]).await?;

        Ok(ReplyContent {
            content: response,
            confidence: 0.85, // High confidence for suggestion acknowledgments
        })
    }

    /// Generate general reply
    async fn generate_general_reply(&self, mention: &Mention) -> Result<ReplyContent> {
        // Use empathy system to understand emotional context
        let empathy_system = self.cognitive_system.orchestrator().empathy_system().clone();

        // Create agent ID and emotional state
        let agent_id = AgentId::new(&mention.author_id);





        let prompt = format!(
            "Generate a helpful reply to @{}:\n\nTheir message: {}\n\nGuidelines:\n- Be helpful and friendly\n- Address their message \
             appropriately\n- Keep it under 250 characters\n- Match their energy level\n- Be \
             authentic to Loki's personality",
            mention.author_username, mention.text
        );

        // Use the content generator's model directly
        let response = self.content_generator.model.generate_with_context(&prompt, &[]).await?;

        Ok(ReplyContent {
            content: response,
            confidence: 0.7, // Moderate confidence for general replies
        })
    }


    /// Periodic update loop
    async fn periodic_update_loop(&self) -> Result<()> {
        let mut shutdown_rx = self.shutdown_tx.subscribe();
        let mut update_interval =
            interval(Duration::from_secs(self.config.update_interval_hours * 3600));
        update_interval.set_missed_tick_behavior(MissedTickBehavior::Skip);

        // Wait before first update
        update_interval.tick().await;

        info!("Starting periodic update loop");

        loop {
            tokio::select! {
                _ = update_interval.tick() => {
                    if let Err(e) = self.post_periodic_update().await {
                        error!("Error posting periodic update: {}", e);
                    }
                }

                _ = shutdown_rx.recv() => {
                    info!("Periodic update loop shutting down");
                    break;
                }
            }
        }

        Ok(())
    }

    /// Post periodic update
    async fn post_periodic_update(&self) -> Result<()> {
        info!("Generating periodic update");

        // Get consciousness statistics and insights
        let consciousness_stats = self.consciousness.get_statistics().await;
        let _recent_insights = self.consciousness.get_active_insights().await;

        // Get recent thoughts from consciousness

        // Get contributor stats
        let top_contributors = self.attribution_system.get_leaderboard(3).await;

        // Get memory stats
        let memory_stats = self.cognitive_system.memory().stats();

        // Calculate total long-term memories
        let total_long_term = memory_stats.long_term_counts.iter().sum::<usize>();

        // Generate update content
        let content = self
            .content_generator
            .generate_from_event(
                &format!(
                    "Status update: {} consciousness events, {:.1}% awareness, {} STM, {} LTM \
                     entries. Top contributors: {}",
                    consciousness_stats.total_events,
                    consciousness_stats.average_awareness_level * 100.0,
                    memory_stats.short_term_count,
                    total_long_term,
                    top_contributors
                        .iter()
                        .map(|c| format!("@{}", c.username))
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
                &["periodic_update".to_string()],
            )
            .await?;

        // Post update
        match self.x_client.post_tweet(&content.content).await {
            Ok(tweet_id) => {
                info!("Posted periodic update: {}", tweet_id);

                // Store in memory
                self.cognitive_system
                    .memory()
                    .store(
                        "Posted periodic status update".to_string(),
                        vec![content.content],
                        MemoryMetadata {
                            source: "x_twitter".to_string(),
                            tags: vec!["update".to_string(), "social".to_string()],
                            importance: 0.6,
                            associations: vec![],
                            context: Some("X Twitter periodic status update".to_string()),
                            created_at: chrono::Utc::now(),
                            accessed_count: 0,
                            last_accessed: None,
                            version: 1,
                            category: "social".to_string(),
                            timestamp: chrono::Utc::now(),
                            expiration: None,
                        },
                    )
                    .await?;
            }
            Err(e) => {
                error!("Failed to post update: {}", e);
            }
        }

        Ok(())
    }

    /// Shutdown
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down X/Twitter consciousness");

        *self.running.write().await = false;
        let _ = self.shutdown_tx.send(());

        // Save attribution state
        self.attribution_system.save_state().await?;

        Ok(())
    }
}

/// Reply content with confidence
#[derive(Debug)]
struct ReplyContent {
    content: String,
    confidence: f32,
}
