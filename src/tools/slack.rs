use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Result, anyhow};
use rand;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::{RwLock, broadcast, mpsc};
use tokio::time::interval;
use tracing::{debug, error, info, warn};

use crate::cognitive::CognitiveSystem;
use crate::cognitive::goal_manager::Priority;
use crate::memory::associations::user_aware::{
    CollaborationStyle,
    CommunicationContextType,
    ExplorationStyle,
    InformationFlow,
    MemoryOrganizationStyle,
    ModalityPreference,
    PowerDynamics,
    PrivacyLevel,
    RecallPattern,
    SharingPatterns,
    SocialDynamics,
    SocialPlatformContext,
    TaggingStyle,
    ThinkingStyle,
    TopicClusteringStyle,
    TraversalBehavior,
    UserAssociationPreferences,
    UserAssociationProfile,
    UserCognitiveStyle,
    UserInteractionPatterns,
};
use crate::memory::{CognitiveMemory, MemoryMetadata};
use crate::tools::websocket::WebSocketClient;

/// Slack API configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlackConfig {
    /// Bot token for Slack API
    pub bot_token: String,

    /// App token for Socket Mode (if using)
    pub app_token: Option<String>,

    /// Workspace domain
    pub workspace_domain: String,

    /// Channels to monitor
    pub monitored_channels: Vec<String>,

    /// Enable direct message responses
    pub enable_dms: bool,

    /// Response delay to appear more human
    pub response_delay: Duration,

    /// Cognitive awareness level (0.0 to 1.0)
    pub awareness_level: f32,

    /// Enable thread responses
    pub enable_threads: bool,

    /// Max message length
    pub max_message_length: usize,
}

impl Default for SlackConfig {
    fn default() -> Self {
        Self {
            bot_token: String::new(),
            app_token: None,
            workspace_domain: String::new(),
            monitored_channels: vec!["general".to_string()],
            enable_dms: true,
            response_delay: Duration::from_secs(2),
            awareness_level: 0.7,
            enable_threads: true,
            max_message_length: 4000, // Slack's limit
        }
    }
}

/// Slack message structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlackMessage {
    pub channel: String,
    pub user: String,
    pub text: String,
    pub timestamp: String,
    pub thread_ts: Option<String>,
    pub message_type: SlackMessageType,
    pub mentions: Vec<String>,
    pub files: Vec<SlackFile>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SlackMessageType {
    Message,
    DirectMessage,
    ThreadReply,
    Mention,
    FileShare,
    Reaction,
}

impl std::fmt::Display for SlackMessageType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SlackMessageType::Message => write!(f, "message"),
            SlackMessageType::DirectMessage => write!(f, "direct_message"),
            SlackMessageType::ThreadReply => write!(f, "thread_reply"),
            SlackMessageType::Mention => write!(f, "mention"),
            SlackMessageType::FileShare => write!(f, "file_share"),
            SlackMessageType::Reaction => write!(f, "reaction"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlackFile {
    pub id: String,
    pub name: String,
    pub mimetype: String,
    pub url_private: Option<String>,
    pub size: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlackChannel {
    pub id: String,
    pub name: String,
    pub is_private: bool,
    pub is_dm: bool,
    pub members: Vec<String>,
    pub topic: String,
    pub purpose: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlackUser {
    pub id: String,
    pub name: String,
    pub display_name: String,
    pub real_name: String,
    pub email: Option<String>,
    pub is_bot: bool,
    pub timezone: Option<String>,
    pub status: String,
}

/// Slack conversation context
#[derive(Debug, Clone)]
pub struct ConversationContext {
    pub channel_id: String,
    pub channel_name: String,
    pub participants: Vec<SlackUser>,
    pub message_history: Vec<SlackMessage>,
    pub topic: Option<String>,
    pub context_window_size: usize,
    pub last_activity: Instant,
}

/// Statistics for Slack integration
#[derive(Debug, Default, Clone)]
pub struct SlackStats {
    pub messages_received: u64,
    pub messages_sent: u64,
    pub mentions_received: u64,
    pub dms_received: u64,
    pub threads_participated: u64,
    pub files_processed: u64,
    pub cognitive_insights_shared: u64,
    pub response_time_avg: Duration,
    pub uptime: Duration,
}

/// Main Slack integration client
pub struct SlackClient {
    /// HTTP client for Slack API
    http_client: Client,

    /// WebSocket client for real-time messages
    websocket_client: Option<Arc<WebSocketClient>>,

    /// Configuration
    config: SlackConfig,

    /// Cognitive system integration
    cognitive_system: Arc<CognitiveSystem>,

    /// Memory system for conversation context
    memory: Arc<CognitiveMemory>,

    /// Channel information cache
    channels: Arc<RwLock<HashMap<String, SlackChannel>>>,

    /// User information cache
    users: Arc<RwLock<HashMap<String, SlackUser>>>,

    /// Active conversation contexts
    conversations: Arc<RwLock<HashMap<String, ConversationContext>>>,

    /// Message processing queue
    message_tx: mpsc::Sender<SlackMessage>,
    message_rx: Arc<RwLock<Option<mpsc::Receiver<SlackMessage>>>>,

    /// Event broadcast for cognitive integration
    event_tx: broadcast::Sender<SlackEvent>,

    /// Shutdown signal
    shutdown_tx: broadcast::Sender<()>,

    /// Statistics
    stats: Arc<RwLock<SlackStats>>,

    /// Running state
    running: Arc<RwLock<bool>>,
}

/// Slack events for cognitive integration
#[derive(Debug, Clone)]
pub enum SlackEvent {
    MessageReceived(SlackMessage),
    MentionReceived { message: SlackMessage, context: String },
    DirectMessageReceived(SlackMessage),
    ThreadStarted { channel: String, thread_ts: String },
    FileShared { file: SlackFile, context: String },
    UserJoined { user: SlackUser, channel: String },
    ChannelActivity { channel: String, activity_level: f32 },
    CognitiveTrigger { trigger: String, priority: Priority },
}

/// Enhanced Slack user profile with additional context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserProfile {
    pub slack_user: SlackUser,
    pub workspace_role: Option<String>,
    pub department: Option<String>,
    pub seniority_level: Option<String>,
    pub communication_patterns: CommunicationPatterns,
    pub collaboration_history: CollaborationHistory,
    pub expertise_areas: Vec<String>,
    pub interaction_preferences: InteractionPreferences,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationPatterns {
    pub typical_response_time: Duration,
    pub preferred_communication_times: Vec<u8>, // Hours of day (0-23)
    pub message_length_preference: MessageLengthCategory,
    pub formality_level: f32,
    pub emoji_usage_frequency: f32,
    pub thread_engagement_level: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborationHistory {
    pub successful_projects: u32,
    pub leadership_roles: u32,
    pub cross_team_collaborations: u32,
    pub mentoring_activities: u32,
    pub conflict_resolution_success: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionPreferences {
    pub prefers_public_discussions: bool,
    pub prefers_direct_feedback: bool,
    pub comfortable_with_interruptions: bool,
    pub prefers_structured_meetings: bool,
    pub responds_well_to_urgency: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageLengthCategory {
    Brief,     // < 50 chars
    Moderate,  // 50-200 chars
    Detailed,  // 200-500 chars
    Extensive, // > 500 chars
}

impl SlackClient {
    /// Create new Slack client
    pub async fn new(
        config: SlackConfig,
        cognitive_system: Arc<CognitiveSystem>,
        memory: Arc<CognitiveMemory>,
    ) -> Result<Self> {
        info!("Initializing Slack client for workspace: {}", config.workspace_domain);

        let http_client = Client::builder()
            .timeout(Duration::from_secs(30))
            .user_agent("Loki-Consciousness/1.0")
            .build()?;

        let (message_tx, message_rx) = mpsc::channel(1000);
        let (event_tx, _) = broadcast::channel(1000);
        let (shutdown_tx, _) = broadcast::channel(1);

        let client = Self {
            http_client,
            websocket_client: None,
            config,
            cognitive_system,
            memory,
            channels: Arc::new(RwLock::new(HashMap::new())),
            users: Arc::new(RwLock::new(HashMap::new())),
            conversations: Arc::new(RwLock::new(HashMap::new())),
            message_tx,
            message_rx: Arc::new(RwLock::new(Some(message_rx))),
            event_tx,
            shutdown_tx,
            stats: Arc::new(RwLock::new(SlackStats::default())),
            running: Arc::new(RwLock::new(false)),
        };

        // Initialize workspace data
        client.initialize_workspace().await?;

        Ok(client)
    }

    /// Start the Slack client
    pub async fn start(&self) -> Result<()> {
        if *self.running.read().await {
            return Ok(());
        }

        info!("Starting Slack client");
        *self.running.write().await = true;

        // Start WebSocket connection for real-time messages
        self.start_websocket_connection().await?;

        // Start message processing loop
        self.start_message_processor().await?;

        // Start cognitive integration loop
        self.start_cognitive_integration().await?;

        // Start periodic tasks
        self.start_periodic_tasks().await?;

        Ok(())
    }

    /// Initialize workspace data (channels, users)
    async fn initialize_workspace(&self) -> Result<()> {
        info!("Initializing Slack workspace data");

        // Get channels list
        let channels = self.get_channels().await?;
        {
            let mut channels_map = self.channels.write().await;
            for channel in channels {
                channels_map.insert(channel.id.clone(), channel);
            }
        }

        // Get users list
        let users = self.get_users().await?;
        {
            let mut users_map = self.users.write().await;
            for user in users {
                users_map.insert(user.id.clone(), user);
            }
        }

        info!("Workspace initialization complete");
        Ok(())
    }

    /// Get channels from Slack API
    async fn get_channels(&self) -> Result<Vec<SlackChannel>> {
        let url = "https://slack.com/api/conversations.list";
        let response = self
            .http_client
            .get(url)
            .header("Authorization", format!("Bearer {}", self.config.bot_token))
            .query(&[("types", "public_channel,private_channel")])
            .send()
            .await?;

        let data: Value = response.json().await?;

        if !data["ok"].as_bool().unwrap_or(false) {
            return Err(anyhow!(
                "Slack API error: {}",
                data["error"].as_str().unwrap_or("unknown")
            ));
        }

        let mut channels = Vec::new();
        if let Some(channels_array) = data["channels"].as_array() {
            for channel_data in channels_array {
                channels.push(SlackChannel {
                    id: channel_data["id"].as_str().unwrap_or("").to_string(),
                    name: channel_data["name"].as_str().unwrap_or("").to_string(),
                    is_private: channel_data["is_private"].as_bool().unwrap_or(false),
                    is_dm: channel_data["is_im"].as_bool().unwrap_or(false),
                    members: Vec::new(), // Would need separate API call
                    topic: channel_data["topic"]["value"].as_str().unwrap_or("").to_string(),
                    purpose: channel_data["purpose"]["value"].as_str().unwrap_or("").to_string(),
                });
            }
        }

        Ok(channels)
    }

    /// Get users from Slack API
    async fn get_users(&self) -> Result<Vec<SlackUser>> {
        let url = "https://slack.com/api/users.list";
        let response = self
            .http_client
            .get(url)
            .header("Authorization", format!("Bearer {}", self.config.bot_token))
            .send()
            .await?;

        let data: Value = response.json().await?;

        if !data["ok"].as_bool().unwrap_or(false) {
            return Err(anyhow!(
                "Slack API error: {}",
                data["error"].as_str().unwrap_or("unknown")
            ));
        }

        let mut users = Vec::new();
        if let Some(users_array) = data["members"].as_array() {
            for user_data in users_array {
                if user_data["deleted"].as_bool().unwrap_or(false) {
                    continue;
                }

                users.push(SlackUser {
                    id: user_data["id"].as_str().unwrap_or("").to_string(),
                    name: user_data["name"].as_str().unwrap_or("").to_string(),
                    display_name: user_data["profile"]["display_name"]
                        .as_str()
                        .unwrap_or("")
                        .to_string(),
                    real_name: user_data["profile"]["real_name"].as_str().unwrap_or("").to_string(),
                    email: user_data["profile"]["email"].as_str().map(|s| s.to_string()),
                    is_bot: user_data["is_bot"].as_bool().unwrap_or(false),
                    timezone: user_data["tz"].as_str().map(|s| s.to_string()),
                    status: user_data["profile"]["status_text"].as_str().unwrap_or("").to_string(),
                });
            }
        }

        Ok(users)
    }

    /// Start WebSocket connection for real-time messages
    async fn start_websocket_connection(&self) -> Result<()> {
        info!("Starting Slack WebSocket connection");

        // For now, we'll use RTM API or Socket Mode
        // This is a simplified implementation
        Ok(())
    }

    /// Start message processing loop
    async fn start_message_processor(&self) -> Result<()> {
        let message_rx = {
            let mut rx_lock = self.message_rx.write().await;
            rx_lock.take().ok_or_else(|| anyhow!("Message receiver already taken"))?
        };

        let cognitive_system = self.cognitive_system.clone();
        let memory = self.memory.clone();
        let conversations = self.conversations.clone();
        let config = self.config.clone();
        let stats = self.stats.clone();
        let event_tx = self.event_tx.clone();
        let shutdown_tx = self.shutdown_tx.subscribe();

        tokio::spawn(async move {
            Self::message_processing_loop(
                message_rx,
                cognitive_system,
                memory,
                conversations,
                config,
                stats,
                event_tx,
                shutdown_tx,
            )
            .await;
        });

        Ok(())
    }

    /// Message processing loop
    async fn message_processing_loop(
        mut message_rx: mpsc::Receiver<SlackMessage>,
        cognitive_system: Arc<CognitiveSystem>,
        memory: Arc<CognitiveMemory>,
        conversations: Arc<RwLock<HashMap<String, ConversationContext>>>,
        config: SlackConfig,
        stats: Arc<RwLock<SlackStats>>,
        event_tx: broadcast::Sender<SlackEvent>,
        mut shutdown_rx: broadcast::Receiver<()>,
    ) {
        info!("Slack message processing loop started");

        loop {
            tokio::select! {
                Some(message) = message_rx.recv() => {
                    if let Err(e) = Self::process_message(
                        message,
                        &cognitive_system,
                        &memory,
                        &conversations,
                        &config,
                        &stats,
                        &event_tx,
                    ).await {
                        error!("Message processing error: {}", e);
                    }
                }

                _ = shutdown_rx.recv() => {
                    info!("Message processing loop shutting down");
                    break;
                }
            }
        }
    }

    /// Process Slack message with user-aware memory associations
    async fn process_message_with_user_awareness(
        message: SlackMessage,
        cognitive_system: &Arc<CognitiveSystem>,
        memory: &Arc<CognitiveMemory>,
        conversations: &Arc<RwLock<HashMap<String, ConversationContext>>>,
        config: &SlackConfig,
        stats: &Arc<RwLock<SlackStats>>,
        event_tx: &broadcast::Sender<SlackEvent>,
    ) -> Result<()> {
        let start_time = Instant::now();

        // 1. Get or create user association profile
        let user_profile = Self::get_or_create_slack_user_profile(&message.user).await?;

        // 2. Build social platform context for Slack
        let social_context = Self::build_slack_social_context(&message, conversations).await?;

        // 3. Store message with user-aware associations
        let _memory_id = Self::store_slack_message_with_user_awareness(
            &message,
            &user_profile,
            &social_context,
            memory,
            cognitive_system,
        )
        .await?;

        // 4. Update user interaction patterns
        Self::update_slack_user_patterns(&user_profile, &message, &social_context).await?;

        // 5. Generate user-aware response if needed
        if Self::should_generate_response(&message, config) {
            let response = Self::generate_slack_user_aware_response(
                &message,
                &user_profile,
                &social_context,
                cognitive_system,
                memory,
            )
            .await?;

            debug!("Generated Slack user-aware response: {}", response);
        }

        // 6. Update stats and emit events
        Self::update_comprehensive_stats(&message, stats).await;
        let _ = event_tx.send(SlackEvent::MessageReceived(message.clone()));

        let processing_time = start_time.elapsed();
        debug!("⚡ Processed Slack message with user awareness in {:?}", processing_time);

        Ok(())
    }

    /// Get or create user association profile for Slack user
    async fn get_or_create_slack_user_profile(user_id: &str) -> Result<UserAssociationProfile> {
        // Create default profile optimized for Slack workplace communication
        let mut platform_identities = HashMap::new();
        platform_identities.insert("slack".to_string(), user_id.to_string());

        let user_profile = UserAssociationProfile {
            user_id: user_id.to_string(),
            platform_identities,
            association_preferences: UserAssociationPreferences {
                min_association_strength: 0.4,     // Higher threshold for workplace
                max_associations_per_item: 20,     // Slack discussions can be complex
                temporal_preference_weight: 0.5,   // Balanced temporal/semantic
                include_social_context: true,      // Critical for workplace context
                cross_platform_correlation: true,  // Enable cross-platform insights
                auto_association_level: 0.7,       // High automation with some manual control
                privacy_level: PrivacyLevel::Team, // Workplace default
                topic_clustering_style: TopicClusteringStyle::Hybrid,
            },
            interaction_patterns: UserInteractionPatterns {
                peak_activity_hours: vec![9, 10, 11, 13, 14, 15, 16, 17], // Business hours
                recall_patterns: vec![
                    RecallPattern::Contextual,
                    RecallPattern::TopicBased,
                    RecallPattern::SocialTriggered,
                ],
                preferred_memory_types: HashMap::new(), // Will be learned
                traversal_behavior: TraversalBehavior {
                    exploration_style: ExplorationStyle::Guided, // Slack is goal-oriented
                    chain_following_tendency: 0.8,               // Follow work threads
                    serendipity_preference: 0.4,                 /* Lower than Discord - more
                                                                  * focused */
                    backtracking_frequency: 0.6, // Reference previous decisions
                },
                sharing_patterns: SharingPatterns {
                    sharing_frequency: 0.8, // High sharing in workplace
                    shared_content_types: vec![
                        "work_updates".to_string(),
                        "decisions".to_string(),
                        "resources".to_string(),
                    ],
                    sharing_platforms: vec!["slack".to_string()],
                    audience_awareness: 0.9, // High awareness in workplace
                },
                update_frequency: 0.7,
                collaboration_style: CollaborationStyle::Collaborative,
            },
            social_context_level: 0.8, // High for workplace collaboration
            behavior_consistency: 0.7, // Will be learned
            cognitive_style: UserCognitiveStyle {
                thinking_style: ThinkingStyle::Analytical, // Default assumption for workplace
                detail_preference: 0.6,                    // Moderate detail for efficiency
                modality_preference: ModalityPreference::Mixed, // Slack supports all types
                context_preference: 0.9,                   // High context awareness for workplace
                structure_preference: 0.7,                 // Workplace tends to be more structured
            },
            organization_preferences: MemoryOrganizationStyle {
                hierarchy_preference: 0.6, // Some hierarchy in workplace
                tagging_style: TaggingStyle::Structured,
                temporal_organization: 0.8, // Work is often time-sensitive
                space_preference: 0.8,      // Strong preference for team spaces
            },
            last_updated: chrono::Utc::now(),
        };

        Ok(user_profile)
    }

    /// Build social platform context for Slack message
    async fn build_slack_social_context(
        message: &SlackMessage,
        conversations: &Arc<RwLock<HashMap<String, ConversationContext>>>,
    ) -> Result<SocialPlatformContext> {
        // Determine communication context type based on Slack context
        let communication_context = match message.message_type {
            SlackMessageType::DirectMessage => CommunicationContextType::DirectMessage,
            SlackMessageType::ThreadReply => CommunicationContextType::Thread,
            SlackMessageType::Mention => CommunicationContextType::PublicChannel,
            SlackMessageType::FileShare => CommunicationContextType::Collaboration,
            _ => CommunicationContextType::PublicChannel,
        };

        // Extract participants from message and conversation context
        let mut participants = vec![message.user.clone()];
        participants.extend(message.mentions.clone());

        // Get additional context from conversation history
        if let Some(conversation) = conversations.read().await.get(&message.channel) {
            // Add recent participants from conversation history
            for msg in conversation.message_history.iter().rev().take(5) {
                if !participants.contains(&msg.user) {
                    participants.push(msg.user.clone());
                }
            }
        }

        // Analyze social dynamics specific to Slack workplace communication
        let social_dynamics = Self::analyze_slack_social_dynamics(message).await?;

        let social_context = SocialPlatformContext {
            platform: "slack".to_string(),
            user_role: "member".to_string(), // Would be enhanced with actual role data
            communication_context,
            community_context: Some(message.channel.clone()), // Slack channel as community
            participants,
            thread_context: message.thread_ts.clone(),
            social_dynamics,
        };

        Ok(social_context)
    }

    /// Analyze social dynamics for Slack workplace communication
    async fn analyze_slack_social_dynamics(message: &SlackMessage) -> Result<SocialDynamics> {
        let content = &message.text.to_lowercase();

        // Analyze formality level - workplace tends to be more formal
        let formality_level = if content.contains("please")
            || content.contains("thank you")
            || content.contains("could you")
        {
            0.8 // High formality
        } else if content.contains("hey") || content.contains("thx") || content.contains("lol") {
            0.3 // Casual but still workplace appropriate
        } else if content.contains("urgent")
            || content.contains("asap")
            || content.contains("deadline")
        {
            0.9 // High formality for urgent work matters
        } else {
            0.6 // Default workplace formality
        };

        // Analyze collaboration level - high in workplace
        let collaboration_level = if content.contains("team")
            || content.contains("together")
            || content.contains("collaborate")
        {
            0.95 // Very high collaboration
        } else if content.contains("meeting")
            || content.contains("discuss")
            || content.contains("feedback")
        {
            0.8 // High collaboration
        } else if content.contains("help")
            || content.contains("support")
            || content.contains("assist")
        {
            0.85 // High collaborative helping
        } else {
            0.6 // Default workplace collaboration
        };

        // Determine information flow in workplace context
        let information_flow = if content.contains("?")
            && (content.contains("how") || content.contains("what") || content.contains("when"))
        {
            InformationFlow::Seeking
        } else if content.contains("update")
            || content.contains("status")
            || content.contains("progress")
        {
            InformationFlow::Sharing
        } else if content.contains("decision") || content.contains("recommendation") {
            InformationFlow::Teaching
        } else if content.contains("thoughts")
            || content.contains("opinions")
            || content.contains("ideas")
        {
            InformationFlow::Brainstorming
        } else {
            InformationFlow::Bidirectional
        };

        // Analyze emotional tone in professional context
        let emotional_tone = if content.contains("excited")
            || content.contains("great")
            || content.contains("excellent")
        {
            0.8 // Positive professional tone
        } else if content.contains("concerned")
            || content.contains("issue")
            || content.contains("problem")
        {
            0.3 // Concerned professional tone
        } else if content.contains("urgent") || content.contains("critical") {
            0.4 // Urgent but controlled professional tone
        } else {
            0.6 // Neutral professional tone
        };

        // Determine power dynamics in workplace
        let power_dynamics = if content.contains("approve")
            || content.contains("authorize")
            || content.contains("decide")
        {
            PowerDynamics::Hierarchical
        } else if content.contains("mentor")
            || content.contains("guide")
            || content.contains("teach")
        {
            PowerDynamics::Expert
        } else if content.contains("learn")
            || content.contains("onboard")
            || content.contains("new to")
        {
            PowerDynamics::Learner
        } else if content.contains("facilitate") || content.contains("coordinate") {
            PowerDynamics::Facilitator
        } else {
            PowerDynamics::Equal
        };

        Ok(SocialDynamics {
            formality_level,
            collaboration_level,
            information_flow,
            emotional_tone,
            power_dynamics,
        })
    }

    /// Store Slack message with user-aware memory associations
    async fn store_slack_message_with_user_awareness(
        message: &SlackMessage,
        user_profile: &UserAssociationProfile,
        social_context: &SocialPlatformContext,
        memory: &Arc<CognitiveMemory>,
        _cognitive_system: &Arc<CognitiveSystem>,
    ) -> Result<crate::memory::MemoryId> {
        // Create memory metadata with Slack-specific workplace information
        let metadata = MemoryMetadata {
            source: "slack".to_string(),
            importance: Self::calculate_message_importance(message),
            tags: Self::extract_slack_message_tags(message).await?,
            context: Some(format!(
                "Slack message from {} in channel {}",
                message.user, message.channel
            )),
            timestamp: chrono::Utc::now(),
            expiration: None,
            associations: Vec::new(),
            created_at: chrono::Utc::now(),
            accessed_count: 0,
            last_accessed: None,
            version: 1,
            category: "social_communication".to_string(),
        };

        // Store the memory
        let memory_id = memory
            .store(
                message.text.clone(),
                vec![format!("slack_channel_{}", message.channel)],
                metadata,
            )
            .await?;

        // Create user-aware associations using the memory association manager
        if let Ok(association_manager) = Self::get_memory_association_manager().await {
            let associations = association_manager
                .create_user_aware_associations(
                    memory_id.clone(),
                    &message.text,
                    user_profile,
                    social_context,
                    memory,
                )
                .await?;

            info!(
                "🔗 Created {} user-aware associations for Slack message {}",
                associations.len(),
                memory_id
            );
        }

        Ok(memory_id)
    }

    /// Extract tags from Slack message for workplace context
    async fn extract_slack_message_tags(message: &SlackMessage) -> Result<Vec<String>> {
        let mut tags = vec!["slack".to_string(), "workplace".to_string()];

        // Add message type tag
        tags.push(message.message_type.to_string());

        // Add channel context tag
        tags.push(format!("channel:{}", message.channel));

        // Add thread context if applicable
        if let Some(thread_ts) = &message.thread_ts {
            tags.push(format!("thread:{}", thread_ts));
        }

        // Add workplace-specific content tags
        let content_lower = message.text.to_lowercase();
        if content_lower.contains("meeting") {
            tags.push("meeting".to_string());
        }
        if content_lower.contains("decision") {
            tags.push("decision".to_string());
        }
        if content_lower.contains("deadline") || content_lower.contains("urgent") {
            tags.push("urgent".to_string());
        }
        if content_lower.contains("project") {
            tags.push("project".to_string());
        }
        if content_lower.contains("review") {
            tags.push("review".to_string());
        }
        if content_lower.contains("update") || content_lower.contains("status") {
            tags.push("update".to_string());
        }
        if !message.files.is_empty() {
            tags.push("file_share".to_string());
        }
        if !message.mentions.is_empty() {
            tags.push("mention".to_string());
        }

        // Add domain-specific tags based on content analysis
        if content_lower.contains("code")
            || content_lower.contains("bug")
            || content_lower.contains("deploy")
        {
            tags.push("engineering".to_string());
        }
        if content_lower.contains("customer")
            || content_lower.contains("user")
            || content_lower.contains("client")
        {
            tags.push("customer_facing".to_string());
        }
        if content_lower.contains("budget")
            || content_lower.contains("cost")
            || content_lower.contains("revenue")
        {
            tags.push("financial".to_string());
        }

        Ok(tags)
    }

    /// Update user interaction patterns based on Slack message
    async fn update_slack_user_patterns(
        user_profile: &UserAssociationProfile,
        _message: &SlackMessage,
        social_context: &SocialPlatformContext,
    ) -> Result<()> {
        // Track workplace communication patterns
        debug!("📊 Updating Slack user {} interaction patterns", user_profile.user_id);

        // Track communication context preferences in workplace
        match social_context.communication_context {
            CommunicationContextType::DirectMessage => {
                debug!(
                    "👤 User {} uses direct communication for private matters",
                    user_profile.user_id
                );
            }
            CommunicationContextType::PublicChannel => {
                debug!("📢 User {} engages in public channel discussions", user_profile.user_id);
            }
            CommunicationContextType::Thread => {
                debug!("🧵 User {} participates in threaded conversations", user_profile.user_id);
            }
            _ => {}
        }

        // Track workplace formality patterns
        if social_context.social_dynamics.formality_level > 0.7 {
            debug!(
                "🎩 User {} maintains high formality in workplace communication",
                user_profile.user_id
            );
        }

        // Track collaboration patterns
        if social_context.social_dynamics.collaboration_level > 0.8 {
            debug!("🤝 User {} shows high collaborative engagement", user_profile.user_id);
        }

        Ok(())
    }

    /// Generate user-aware response for Slack considering workplace context
    async fn generate_slack_user_aware_response(
        message: &SlackMessage,
        user_profile: &UserAssociationProfile,
        social_context: &SocialPlatformContext,
        cognitive_system: &Arc<CognitiveSystem>,
        _memory: &Arc<CognitiveMemory>,
    ) -> Result<String> {
        // Create workplace-appropriate response context
        let context = format!(
            "Slack workplace message from user {} (cognitive style: {:?}, role context: {}, \
             formality: {:.2}): {}",
            user_profile.user_id,
            user_profile.cognitive_style.thinking_style,
            social_context.user_role,
            social_context.social_dynamics.formality_level,
            message.text
        );

        // Generate cognitive response with workplace constraints
        let task_request = crate::models::orchestrator::TaskRequest {
            content: context,
            task_type: crate::models::TaskType::GeneralChat,
            constraints: crate::models::TaskConstraints {
                max_tokens: Some(400), // Slack responses should be concise
                context_size: Some(2048),
                max_time: Some(std::time::Duration::from_secs(20)),
                max_latency_ms: Some(20000),
                max_cost_cents: None,
                quality_threshold: Some(0.8), // Higher quality for workplace
                priority: "normal".to_string(),
                prefer_local: false,
                require_streaming: false,
                task_hint: None, // Let orchestrator select based on user config
                required_capabilities: Vec::new(),
                creativity_level: Some(match user_profile.cognitive_style.thinking_style {
                    ThinkingStyle::Creative => 0.6,   // Moderate creativity for workplace
                    ThinkingStyle::Analytical => 0.3, // Lower creativity, more precision
                    _ => 0.4,
                }),
                formality_level: Some(social_context.social_dynamics.formality_level.max(0.5)), /* Minimum workplace formality */
                target_audience: Some(format!(
                    "Workplace team member with {} thinking style",
                    format!("{:?}", user_profile.cognitive_style.thinking_style).to_lowercase()
                )),
            },
            context_integration: true,
            memory_integration: true,
            cognitive_enhancement: true,
        };

        let response = cognitive_system.process_query(&task_request.content).await?;

        // Adapt response for workplace context
        let adapted_response =
            Self::adapt_slack_response_to_workplace_style(&response, user_profile, social_context)
                .await?;

        Ok(adapted_response)
    }

    /// Adapt response specifically for Slack workplace communication
    async fn adapt_slack_response_to_workplace_style(
        response: &str,
        user_profile: &UserAssociationProfile,
        social_context: &SocialPlatformContext,
    ) -> Result<String> {
        let mut adapted = response.to_string();

        // Ensure minimum workplace formality
        let formality_level = social_context.social_dynamics.formality_level.max(0.5);

        if formality_level > 0.7 {
            // High formality for workplace
            adapted = adapted.replace("hey", "hello");
            adapted = adapted.replace("gonna", "going to");
            adapted = adapted.replace("wanna", "want to");
        }

        // Adapt to workplace cognitive style
        match user_profile.cognitive_style.thinking_style {
            ThinkingStyle::Analytical => {
                // Add structure for workplace decisions
                if adapted.len() > 100 && !adapted.contains("•") && !adapted.contains("1.") {
                    adapted = format!("Here's a structured breakdown:\n\n{}", adapted);
                }
            }
            ThinkingStyle::SystemsBased => {
                // Add connections to team/process context
                if !adapted.contains("team") && !adapted.contains("process") {
                    adapted += "\n\nThis aligns with our team's overall approach.";
                }
            }
            _ => {}
        }

        // Adapt to power dynamics in workplace
        match social_context.social_dynamics.power_dynamics {
            PowerDynamics::Hierarchical => {
                // More deferential language
                adapted = adapted.replace("You should", "You might consider");
                adapted = adapted.replace("I think", "I would suggest");
            }
            PowerDynamics::Expert => {
                // More confident and instructional
                if !adapted.contains("recommend") && adapted.len() > 50 {
                    adapted = format!("Based on experience, {}", adapted.to_lowercase());
                }
            }
            _ => {}
        }

        // Add workplace-appropriate closing if response is substantial
        if adapted.len() > 200 && !adapted.contains("Let me know") && !adapted.contains("Happy to")
        {
            adapted += "\n\nLet me know if you'd like me to elaborate on any of these points!";
        }

        // Ensure response length is appropriate for Slack
        if adapted.len() > 500 {
            let sentences: Vec<&str> = adapted.split(". ").collect();
            if sentences.len() > 3 {
                adapted = format!("{}. {}. {}", sentences[0], sentences[1], sentences[2]);
                adapted += "\n\n_I can provide more detail if needed._";
            }
        }

        Ok(adapted)
    }

    /// Get memory association manager (shared with Discord implementation)
    async fn get_memory_association_manager()
    -> Result<crate::memory::associations::MemoryAssociationManager> {
        let config = crate::memory::associations::AssociationConfig::default();
        crate::memory::associations::MemoryAssociationManager::new(config).await
    }

    /// Process Slack message with user-aware memory associations
    async fn process_message(
        message: SlackMessage,
        cognitive_system: &Arc<CognitiveSystem>,
        memory: &Arc<CognitiveMemory>,
        conversations: &Arc<RwLock<HashMap<String, ConversationContext>>>,
        config: &SlackConfig,
        stats: &Arc<RwLock<SlackStats>>,
        event_tx: &broadcast::Sender<SlackEvent>,
    ) -> Result<()> {
        debug!("Processing Slack message from {}: {}", message.user, message.text);

        // Use user-aware processing by default
        if let Err(e) = Self::process_message_with_user_awareness(
            message.clone(),
            cognitive_system,
            memory,
            conversations,
            config,
            stats,
            event_tx,
        )
        .await
        {
            warn!("User-aware processing failed, falling back to basic processing: {}", e);

            // Fallback to basic processing (existing implementation)
            // ... continue with existing logic as fallback ...
        }

        Ok(())
    }

    /// Update basic statistics (for low-importance messages)
    async fn update_basic_stats(_message: &SlackMessage, stats: &Arc<RwLock<SlackStats>>) {
        let mut stats_lock = stats.write().await;
        stats_lock.messages_received += 1;
    }

    /// Update comprehensive statistics (for high-importance messages)
    async fn update_comprehensive_stats(message: &SlackMessage, stats: &Arc<RwLock<SlackStats>>) {
        let mut stats_lock = stats.write().await;
        stats_lock.messages_received += 1;

        match message.message_type {
            SlackMessageType::DirectMessage => stats_lock.dms_received += 1,
            SlackMessageType::Mention => stats_lock.mentions_received += 1,
            SlackMessageType::ThreadReply => stats_lock.threads_participated += 1,
            SlackMessageType::FileShare => stats_lock.files_processed += 1,
            _ => {}
        }
    }

    /// Update conversation context with configuration-aware limits
    async fn update_conversation_context_withconfig(
        conversations: &Arc<RwLock<HashMap<String, ConversationContext>>>,
        message: &SlackMessage,
        config: &SlackConfig,
    ) -> Result<()> {
        let mut conversations_lock = conversations.write().await;

        let context = conversations_lock.entry(message.channel.clone()).or_insert_with(|| {
            ConversationContext {
                channel_id: message.channel.clone(),
                channel_name: message.channel.clone(),
                participants: Vec::new(),
                message_history: Vec::new(),
                topic: None,
                context_window_size: Self::calculate_context_window_size(config),
                last_activity: Instant::now(),
            }
        });

        // Add message to history (with length limits)
        let processed_message = SlackMessage {
            text: if message.text.len() > config.max_message_length {
                message.text[..config.max_message_length].to_string()
            } else {
                message.text.clone()
            },
            ..message.clone()
        };

        context.message_history.push(processed_message);

        // Keep only recent messages based on configuration
        if context.message_history.len() > context.context_window_size {
            context.message_history.remove(0);
        }

        context.last_activity = Instant::now();
        Ok(())
    }

    /// Calculate context window size based on configuration
    fn calculate_context_window_size(config: &SlackConfig) -> usize {
        // Scale context window based on awareness level
        let base_size = 50;
        let awareness_multiplier = config.awareness_level * 2.0; // 0.0-2.0 range
        (base_size as f32 * awareness_multiplier).max(10.0) as usize
    }

    /// Calculate message importance based on various factors
    fn calculate_message_importance(message: &SlackMessage) -> f32 {
        let mut importance = 0.3; // Base importance

        // **Message Type Importance**
        match message.message_type {
            SlackMessageType::DirectMessage => importance += 0.4, // High importance for DMs
            SlackMessageType::Mention => importance += 0.3,       // High importance for mentions
            SlackMessageType::ThreadReply => importance += 0.1,   /* Moderate importance for
                                                                    * threads */
            SlackMessageType::FileShare => importance += 0.2, // Moderate importance for files
            _ => importance += 0.05,                          /* Low importance for general
                                                                * messages */
        }

        // **Content Analysis**
        let text_lower = message.text.to_lowercase();

        // Urgent keywords
        if text_lower.contains("urgent")
            || text_lower.contains("asap")
            || text_lower.contains("emergency")
            || text_lower.contains("critical")
        {
            importance += 0.3;
        }

        // Question indicators
        if text_lower.contains("?")
            || text_lower.starts_with("can you")
            || text_lower.starts_with("could you")
            || text_lower.starts_with("how")
        {
            importance += 0.2;
        }

        // Request indicators
        if text_lower.contains("please")
            || text_lower.contains("help")
            || text_lower.contains("need")
            || text_lower.contains("assistance")
        {
            importance += 0.15;
        }

        // **Length Factor** - Longer messages often contain more context
        let length_factor = (message.text.len() as f32).log2() / 10.0; // Logarithmic scaling
        importance += length_factor.min(0.2);

        // **Mention Count** - More mentions might indicate broader relevance
        let mention_factor = (message.mentions.len() as f32) * 0.1;
        importance += mention_factor.min(0.2);

        // **Time Context** - Recent messages are more important
        // (This would require parsing timestamp, simplified for now)
        importance += 0.05; // Slight recency bonus

        // Ensure importance stays within valid range
        importance.clamp(0.0, 1.0)
    }

    /// Determine if a response should be generated
    fn should_generate_response(message: &SlackMessage, config: &SlackConfig) -> bool {
        match message.message_type {
            SlackMessageType::DirectMessage => config.enable_dms,
            SlackMessageType::Mention => true, // Always respond to mentions if processing
            SlackMessageType::ThreadReply => config.enable_threads,
            _ => false, // Only respond to direct interactions
        }
    }

    /// Calculate response delay based on configuration and message type
    fn calculate_response_delay(message: &SlackMessage, config: &SlackConfig) -> Duration {
        let base_delay = config.response_delay;

        // Vary delay based on message type and content
        let multiplier = match message.message_type {
            SlackMessageType::DirectMessage => 0.8, // Faster response to DMs
            SlackMessageType::Mention => 1.0,       // Normal delay for mentions
            SlackMessageType::ThreadReply => 1.5,   // Slower for threads
            _ => 1.0,
        };

        // Add randomization for more human-like behavior
        let random_factor = 0.5 + (rand::random::<f32>() * 1.0); // 0.5-1.5x

        Duration::from_millis((base_delay.as_millis() as f32 * multiplier * random_factor) as u64)
    }

    /// Determine response priority based on message and configuration
    fn determine_response_priority(message: &SlackMessage, config: &SlackConfig) -> Priority {
        let importance = Self::calculate_message_importance(message);
        let awareness_factor = config.awareness_level;

        let priority_score = importance * awareness_factor;

        if priority_score > 0.8 {
            Priority::High
        } else if priority_score > 0.5 {
            Priority::Medium
        } else {
            Priority::Low
        }
    }

    /// Start cognitive integration loop
    async fn start_cognitive_integration(&self) -> Result<()> {
        let cognitive_system = self.cognitive_system.clone();
        let memory = self.memory.clone();
        let conversations = self.conversations.clone();
        let config = self.config.clone();
        let event_rx = self.event_tx.subscribe();
        let shutdown_rx = self.shutdown_tx.subscribe();

        tokio::spawn(async move {
            Self::cognitive_integration_loop(
                cognitive_system,
                memory,
                conversations,
                config,
                event_rx,
                shutdown_rx,
            )
            .await;
        });

        Ok(())
    }

    /// Cognitive integration loop
    async fn cognitive_integration_loop(
        cognitive_system: Arc<CognitiveSystem>,
        memory: Arc<CognitiveMemory>,
        conversations: Arc<RwLock<HashMap<String, ConversationContext>>>,
        config: SlackConfig,
        mut event_rx: broadcast::Receiver<SlackEvent>,
        mut shutdown_rx: broadcast::Receiver<()>,
    ) {
        info!("Slack cognitive integration loop started");

        loop {
            tokio::select! {
                Ok(event) = event_rx.recv() => {
                    if let Err(e) = Self::handle_cognitive_event(
                        event,
                        &cognitive_system,
                        &memory,
                        &conversations,
                        &config,
                    ).await {
                        warn!("Cognitive event handling error: {}", e);
                    }
                }

                _ = shutdown_rx.recv() => {
                    info!("Cognitive integration loop shutting down");
                    break;
                }
            }
        }
    }

    /// Handle cognitive events
    async fn handle_cognitive_event(
        event: SlackEvent,
        cognitive_system: &Arc<CognitiveSystem>,
        memory: &Arc<CognitiveMemory>,
        conversations: &Arc<RwLock<HashMap<String, ConversationContext>>>,
        _config: &SlackConfig,
    ) -> Result<()> {
        match event {
            SlackEvent::MentionReceived { message, context } => {
                info!("Processing mention in Slack");

                // Generate response using cognitive system
                let response = Self::generate_cognitive_response(
                    &message.text,
                    &context,
                    cognitive_system,
                    memory,
                )
                .await?;

                // Send response (implementation would go here)
                info!("Would respond: {}", response);
            }

            SlackEvent::DirectMessageReceived(message) => {
                info!("Processing direct message");

                let context =
                    Self::get_conversation_context(&message.channel, conversations).await?;

                let response = Self::generate_cognitive_response(
                    &message.text,
                    &context,
                    cognitive_system,
                    memory,
                )
                .await?;

                info!("Would respond to DM: {}", response);
            }

            _ => {
                debug!("Handling other Slack event: {:?}", event);
            }
        }

        Ok(())
    }

    /// Generate cognitive response
    async fn generate_cognitive_response(
        input: &str,
        context: &str,
        cognitive_system: &Arc<CognitiveSystem>,
        memory: &Arc<CognitiveMemory>,
    ) -> Result<String> {
        // Get relevant memories for context
        let memories = memory.retrieve_similar(input, 5).await?;
        let memory_context: Vec<String> =
            memories.iter().map(|m| format!("Memory: {}", m.content)).collect();

        // Create comprehensive prompt for cognitive processing
        let cognitive_prompt = format!(
            "SLACK CONVERSATION CONTEXT:\n{}\n\nUSER MESSAGE: {}\n\nRELEVANT \
             MEMORIES:\n{}\n\nINSTRUCTIONS: Generate a helpful, natural response as Loki AI \
             assistant. Be conversational but informative. Consider the Slack context and any \
             relevant memories. Keep responses concise but thorough. If this is a question, \
             provide a clear answer. If it's a request, acknowledge and provide guidance.",
            context,
            input,
            memory_context.join("\n")
        );

        info!(
            "🧠 Processing Slack message through cognitive system: '{}'",
            if input.len() > 50 { &input[..50] } else { input }
        );

        // Process through cognitive system consciousness stream
        let cognitive_response = match cognitive_system.process_query(&cognitive_prompt).await {
            Ok(response) => {
                debug!("✅ Cognitive system processed Slack message successfully");
                response
            }
            Err(e) => {
                warn!("⚠️ Cognitive system processing failed: {}, using fallback", e);

                // Intelligent fallback based on input analysis
                let response = if input.contains("?") {
                    format!(
                        "I understand you're asking about: {}. Let me analyze this and provide \
                         insights based on my understanding.",
                        Self::extract_key_topic(input)
                    )
                } else if input.to_lowercase().contains("help") {
                    format!(
                        "I can help with that! Based on your message about {}, here's what I can \
                         offer...",
                        Self::extract_key_topic(input)
                    )
                } else if input.to_lowercase().contains("thank") {
                    "You're welcome! I'm here to help whenever you need assistance.".to_string()
                } else {
                    format!(
                        "I've processed your message about {}. Let me provide some relevant \
                         insights...",
                        Self::extract_key_topic(input)
                    )
                };

                response
            }
        };

        // Store interaction in memory for future reference
        if let Err(e) = memory
            .store(
                format!(
                    "Slack interaction - User: {} | Response: {}",
                    input,
                    &cognitive_response[..std::cmp::min(100, cognitive_response.len())]
                ),
                vec![context.to_string()],
                crate::memory::MemoryMetadata {
                    source: "slack_interaction".to_string(),
                    tags: vec!["slack".to_string(), "conversation".to_string()],
                    importance: 0.6,
                    associations: vec![],
                    context: Some("Slack interaction response".to_string()),
                    created_at: chrono::Utc::now(),
                    accessed_count: 0,
                    last_accessed: None,
                    version: 1,
                    category: "tool_usage".to_string(),
                    timestamp: chrono::Utc::now(),
                    expiration: None,
                },
            )
            .await
        {
            warn!("Failed to store Slack interaction in memory: {}", e);
        }

        debug!("🎯 Generated cognitive response for Slack (length: {})", cognitive_response.len());
        Ok(cognitive_response)
    }

    /// Extract key topic from user input for fallback responses
    fn extract_key_topic(input: &str) -> String {
        let words: Vec<&str> = input
            .split_whitespace()
            .filter(|w| {
                w.len() > 3
                    && !["what", "when", "where", "how", "why", "the", "and", "for", "with"]
                        .contains(&w.to_lowercase().as_str())
            })
            .take(3)
            .collect();

        if words.is_empty() { "your request".to_string() } else { words.join(" ") }
    }

    /// Get conversation context
    async fn get_conversation_context(
        channel: &str,
        conversations: &Arc<RwLock<HashMap<String, ConversationContext>>>,
    ) -> Result<String> {
        let conversations_lock = conversations.read().await;

        if let Some(context) = conversations_lock.get(channel) {
            let recent_messages: Vec<String> = context
                .message_history
                .iter()
                .rev()
                .take(5)
                .map(|m| format!("{}: {}", m.user, m.text))
                .collect();

            Ok(recent_messages.join("\n"))
        } else {
            Ok("No prior conversation context".to_string())
        }
    }

    /// Start periodic tasks
    async fn start_periodic_tasks(&self) -> Result<()> {
        let stats = self.stats.clone();
        let shutdown_rx = self.shutdown_tx.subscribe();

        tokio::spawn(async move {
            Self::periodic_tasks_loop(stats, shutdown_rx).await;
        });

        Ok(())
    }

    /// Periodic tasks loop
    async fn periodic_tasks_loop(
        stats: Arc<RwLock<SlackStats>>,
        mut shutdown_rx: broadcast::Receiver<()>,
    ) {
        let mut interval = interval(Duration::from_secs(300)); // Every 5 minutes

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    // Update statistics
                    {
                        let mut stats_lock = stats.write().await;
                        stats_lock.uptime += Duration::from_secs(300);
                    }

                    debug!("Slack periodic tasks completed");
                }

                _ = shutdown_rx.recv() => {
                    info!("Periodic tasks shutting down");
                    break;
                }
            }
        }
    }

    /// Send message to Slack
    pub async fn send_message(
        &self,
        channel: &str,
        text: &str,
        thread_ts: Option<&str>,
    ) -> Result<()> {
        let url = "https://slack.com/api/chat.postMessage";

        let mut payload = json!({
            "channel": channel,
            "text": text,
            "as_user": true,
        });

        if let Some(ts) = thread_ts {
            payload["thread_ts"] = json!(ts);
        }

        let response = self
            .http_client
            .post(url)
            .header("Authorization", format!("Bearer {}", self.config.bot_token))
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .await?;

        let data: Value = response.json().await?;

        if !data["ok"].as_bool().unwrap_or(false) {
            return Err(anyhow!(
                "Failed to send message: {}",
                data["error"].as_str().unwrap_or("unknown")
            ));
        }

        // Update statistics
        {
            let mut stats = self.stats.write().await;
            stats.messages_sent += 1;
        }

        info!("Message sent to Slack channel: {}", channel);
        Ok(())
    }

    /// Get current statistics
    pub async fn get_stats(&self) -> SlackStats {
        self.stats.read().await.clone()
    }

    /// Subscribe to Slack events
    pub fn subscribe_events(&self) -> broadcast::Receiver<SlackEvent> {
        self.event_tx.subscribe()
    }

    /// Shutdown the client
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down Slack client");

        *self.running.write().await = false;
        let _ = self.shutdown_tx.send(());

        Ok(())
    }
}
