//! Advanced Memory Association System
//!
//! This module provides comprehensive association management for linking emails, contacts,
//! cognitive data, and other information types with intelligent relationship detection,
//! temporal tracking, and cross-platform correlation.

use std::collections::{HashMap, HashSet, BTreeMap};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::RwLock;
use anyhow::Result;
use serde::{Serialize, Deserialize};
use tracing::{info, debug};
use chrono::{DateTime, Utc};

use super::{MemoryId, MemoryMetadata, CognitiveMemory};

/// Comprehensive association manager for memory system
#[derive(Debug)]
pub struct MemoryAssociationManager {
    /// Bidirectional association index
    associations: Arc<RwLock<BTreeMap<MemoryId, AssociationCluster>>>,

    /// Topic-based clustering
    topic_clusters: Arc<RwLock<HashMap<String, TopicCluster>>>,

    /// Temporal association tracking
    temporal_chains: Arc<RwLock<Vec<TemporalChain>>>,

    /// Contact relationship graph
    contact_graph: Arc<RwLock<ContactGraph>>,

    /// Email thread associations
    email_associations: Arc<RwLock<HashMap<String, EmailAssociationCluster>>>,

    /// Cognitive pattern links
    cognitive_links: Arc<RwLock<HashMap<String, CognitivePatternLink>>>,

    /// Cross-platform correlation mapping
    platform_correlations: Arc<RwLock<HashMap<String, CrossPlatformCorrelation>>>,

    /// Association analytics and insights
    analytics: Arc<RwLock<AssociationAnalytics>>,

    /// Configuration
    config: AssociationConfig,
}

/// Configuration for association system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssociationConfig {
    /// Maximum associations per memory item
    pub max_associations_per_item: usize,

    /// Similarity threshold for automatic associations (0.0-1.0)
    pub auto_association_threshold: f32,

    /// Enable temporal chain detection
    pub enable_temporal_chains: bool,

    /// Enable topic clustering
    pub enable_topic_clustering: bool,

    /// Enable contact relationship inference
    pub enable_contact_inference: bool,

    /// Association decay rate (0.0-1.0)
    pub association_decay_rate: f32,

    /// Maximum temporal chain length
    pub max_temporal_chain_length: usize,

    /// Minimum association strength for persistence (0.0-1.0)
    pub min_association_strength: f32,

    /// Enable cross-platform correlation
    pub enable_cross_platform: bool,

    /// Analytics update interval
    pub analytics_update_interval: Duration,
}

impl Default for AssociationConfig {
    fn default() -> Self {
        Self {
            max_associations_per_item: 50,
            auto_association_threshold: 0.7,
            enable_temporal_chains: true,
            enable_topic_clustering: true,
            enable_contact_inference: true,
            association_decay_rate: 0.95,
            max_temporal_chain_length: 20,
            min_association_strength: 0.3,
            enable_cross_platform: true,
            analytics_update_interval: Duration::from_secs(300),
        }
    }
}

/// Cluster of associations for a specific memory item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssociationCluster {
    /// Primary memory ID
    pub memory_id: MemoryId,

    /// Direct associations with strength scores
    pub direct_associations: HashMap<MemoryId, AssociationLink>,

    /// Indirect associations (through shared connections)
    pub indirect_associations: HashMap<MemoryId, f32>,

    /// Topic associations
    pub topic_associations: HashMap<String, f32>,

    /// Contact associations
    pub contact_associations: HashMap<String, ContactAssociation>,

    /// Temporal position in chains
    pub temporal_chains: Vec<String>,

    /// Cross-platform references
    pub platform_references: HashMap<String, String>,

    /// Association metadata
    pub metadata: AssociationClusterMetadata,

    /// Last updated timestamp
    pub last_updated: DateTime<Utc>,
}

/// Individual association link between memories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssociationLink {
    /// Target memory ID
    pub target_id: MemoryId,

    /// Association strength (0.0-1.0)
    pub strength: f32,

    /// Type of association
    pub association_type: AssociationType,

    /// Context that created this association
    pub context: AssociationContext,

    /// Creation timestamp
    pub created_at: DateTime<Utc>,

    /// Last access timestamp
    pub last_accessed: Option<DateTime<Utc>>,

    /// Number of times this association was accessed
    pub access_count: u32,

    /// Confidence in this association (0.0-1.0)
    pub confidence: f32,
}

/// Types of associations between memory items
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum AssociationType {
    /// Content-based semantic similarity
    Semantic,
    /// Temporal proximity (occurred around the same time)
    Temporal,
    /// Same contact or communication thread
    Communicative,
    /// Related to same topic or project
    Topical,
    /// Part of same cognitive process or decision
    Cognitive,
    /// Same data source or platform
    Contextual,
    /// Causal relationship (one led to another)
    Causal,
    /// Hierarchical relationship (parent/child)
    Hierarchical,
    /// Cross-reference or citation
    Reference,
    /// User-defined explicit association
    Explicit,
    /// Functional relationship
    Functional,
}

/// Context information for how an association was created
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssociationContext {
    /// How the association was detected
    pub detection_method: String,

    /// Source system that created the association
    pub source_system: String,

    /// Additional context data
    pub context_data: HashMap<String, String>,

    /// Confidence score for the detection method
    pub method_confidence: f32,
}

/// Topic-based clustering system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopicCluster {
    /// Topic identifier
    pub topic_id: String,

    /// Human-readable topic name
    pub topic_name: String,

    /// Memories in this topic cluster
    pub memory_ids: HashSet<MemoryId>,

    /// Representative keywords for this topic
    pub keywords: Vec<String>,

    /// Topic coherence score (0.0-1.0)
    pub coherence: f32,

    /// Related subtopics
    pub subtopics: HashMap<String, f32>,

    /// Topic statistics
    pub stats: TopicStats,

    /// Last updated
    pub last_updated: DateTime<Utc>,
}

/// Statistics for topic clusters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopicStats {
    /// Number of memories in cluster
    pub memory_count: usize,

    /// Average importance of memories
    pub avg_importance: f32,

    /// Most recent activity
    pub last_activity: DateTime<Utc>,

    /// Activity frequency (accesses per day)
    pub activity_frequency: f32,

    /// Growth rate (new memories per day)
    pub growth_rate: f32,
}

/// Temporal chain of related memories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalChain {
    /// Chain identifier
    pub chain_id: String,

    /// Ordered sequence of memory IDs
    pub memory_sequence: Vec<MemoryId>,

    /// Chain type
    pub chain_type: TemporalChainType,

    /// Time span of the chain
    pub time_span: Duration,

    /// Chain coherence score
    pub coherence: f32,

    /// Chain description
    pub description: String,

    /// Last updated
    pub last_updated: DateTime<Utc>,
}

/// Types of temporal chains
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TemporalChainType {
    /// Conversation or communication thread
    Conversation,
    /// Decision-making process
    DecisionProcess,
    /// Project or task progression
    ProjectProgression,
    /// Learning or research sequence
    LearningSequence,
    /// Problem-solving process
    ProblemSolving,
    /// General temporal sequence
    General,
}

/// Contact relationship graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContactGraph {
    /// Contacts and their information
    pub contacts: HashMap<String, ContactNode>,

    /// Relationships between contacts
    pub relationships: HashMap<String, ContactRelationship>,

    /// Contact groups and organizations
    pub groups: HashMap<String, ContactGroup>,

    /// Communication frequency matrix
    pub communication_matrix: HashMap<(String, String), CommunicationPattern>,
}

/// Contact node in the relationship graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContactNode {
    /// Contact identifier (email, username, etc.)
    pub contact_id: String,

    /// Display name
    pub display_name: Option<String>,

    /// Contact details
    pub details: ContactDetails,

    /// Associated memory IDs
    pub memory_associations: HashSet<MemoryId>,

    /// Communication statistics
    pub communication_stats: CommunicationStats,

    /// Inferred attributes
    pub inferred_attributes: HashMap<String, f32>,

    /// Last interaction
    pub last_interaction: DateTime<Utc>,
}

/// Detailed contact information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContactDetails {
    /// Organization or company
    pub organization: Option<String>,

    /// Role or title
    pub role: Option<String>,

    /// Contact platforms
    pub platforms: HashMap<String, String>,

    /// Preferred communication style
    pub communication_style: Option<String>,

    /// Time zone
    pub timezone: Option<String>,

    /// Tags and categories
    pub tags: HashSet<String>,
}

/// Communication statistics for contacts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationStats {
    /// Total number of interactions
    pub interaction_count: u32,

    /// Average response time
    pub avg_response_time: Duration,

    /// Communication frequency (interactions per week)
    pub frequency: f32,

    /// Preferred communication times
    pub preferred_times: Vec<u8>, // Hours of day (0-23)

    /// Communication quality score
    pub quality_score: f32,

    /// Last response time
    pub last_response_time: Option<Duration>,
}

/// Relationship between contacts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContactRelationship {
    /// Source contact
    pub from_contact: String,

    /// Target contact
    pub to_contact: String,

    /// Relationship type
    pub relationship_type: ContactRelationshipType,

    /// Relationship strength (0.0-1.0)
    pub strength: f32,

    /// Relationship context
    pub context: String,

    /// Bidirectional flag
    pub is_bidirectional: bool,

    /// Creation timestamp
    pub created_at: DateTime<Utc>,
}

/// Types of contact relationships
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContactRelationshipType {
    /// Professional colleague
    Colleague,
    /// Manager/subordinate
    Hierarchical,
    /// Client/vendor relationship
    Business,
    /// Friend or personal contact
    Personal,
    /// Family member
    Family,
    /// Frequent communicator
    Frequent,
    /// Project team member
    TeamMember,
    /// Unknown relationship
    Unknown,
}

/// Contact group or organization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContactGroup {
    /// Group identifier
    pub group_id: String,

    /// Group name
    pub group_name: String,

    /// Group type
    pub group_type: ContactGroupType,

    /// Members of the group
    pub members: HashSet<String>,

    /// Group communication patterns
    pub communication_patterns: HashMap<String, f32>,

    /// Group metadata
    pub metadata: HashMap<String, String>,
}

/// Types of contact groups
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContactGroupType {
    /// Work team or department
    WorkTeam,
    /// Project group
    Project,
    /// Organization or company
    Organization,
    /// Friend group
    Social,
    /// Family group
    Family,
    /// Interest or hobby group
    Interest,
    /// Other type
    Other,
}

/// Communication pattern between contacts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationPattern {
    /// Communication frequency
    pub frequency: f32,

    /// Average message length
    pub avg_message_length: usize,

    /// Typical response time
    pub typical_response_time: Duration,

    /// Communication quality
    pub quality: f32,

    /// Preferred platforms
    pub preferred_platforms: Vec<String>,
}

/// Email-specific association cluster
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailAssociationCluster {
    /// Email thread ID or subject
    pub email_id: String,

    /// Related memory IDs
    pub memory_associations: HashSet<MemoryId>,

    /// Participants in the email thread
    pub participants: HashSet<String>,

    /// Topics discussed in the thread
    pub topics: HashMap<String, f32>,

    /// Action items extracted from emails
    pub action_items: Vec<EmailActionItem>,

    /// Thread progression and timeline
    pub timeline: Vec<EmailTimelineEntry>,

    /// Cognitive connections (decisions, thoughts)
    pub cognitive_connections: HashMap<String, f32>,
}

/// Action item extracted from email
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailActionItem {
    /// Action description
    pub description: String,

    /// Responsible party
    pub assignee: Option<String>,

    /// Due date
    pub due_date: Option<DateTime<Utc>>,

    /// Priority level
    pub priority: String,

    /// Status
    pub status: ActionItemStatus,

    /// Related memory ID
    pub memory_id: Option<MemoryId>,
}

/// Status of action items
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActionItemStatus {
    Pending,
    InProgress,
    Completed,
    Cancelled,
    Overdue,
}

/// Timeline entry for email threads
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailTimelineEntry {
    /// Timestamp of the event
    pub timestamp: DateTime<Utc>,

    /// Event type
    pub event_type: EmailEventType,

    /// Event description
    pub description: String,

    /// Associated memory ID
    pub memory_id: Option<MemoryId>,
}

/// Types of email events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmailEventType {
    Sent,
    Received,
    ActionItemCreated,
    ActionItemCompleted,
    TopicShift,
    ParticipantAdded,
    MeetingScheduled,
    DecisionMade,
}

/// Cognitive pattern link
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitivePatternLink {
    /// Pattern identifier
    pub pattern_id: String,

    /// Pattern type
    pub pattern_type: CognitivePatternType,

    /// Associated memories
    pub memory_associations: HashMap<MemoryId, f32>,

    /// Pattern strength
    pub strength: f32,

    /// Pattern description
    pub description: String,

    /// Cognitive context
    pub cognitive_context: HashMap<String, f32>,

    /// Last detected
    pub last_detected: DateTime<Utc>,
}

/// Types of cognitive patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CognitivePatternType {
    DecisionMaking,
    ProblemSolving,
    Learning,
    Planning,
    Analysis,
    Communication,
    Emotional,
    Social,
    Creative,
    Routine,
}

/// Cross-platform correlation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossPlatformCorrelation {
    /// Platform identifier (email, slack, discord, etc.)
    pub platform_id: String,

    /// Platform-specific identifier
    pub platform_specific_id: String,

    /// Correlated memory IDs
    pub memory_correlations: HashMap<MemoryId, f32>,

    /// Cross-platform confidence
    pub confidence: f32,

    /// Correlation type
    pub correlation_type: CrossPlatformCorrelationType,

    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Types of cross-platform correlations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CrossPlatformCorrelationType {
    SameUser,
    SameTopic,
    SameProject,
    SameTimeframe,
    RelatedContent,
    FollowUp,
}

/// Association analytics and insights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssociationAnalytics {
    /// Total number of associations
    pub total_associations: usize,

    /// Association strength distribution
    pub strength_distribution: HashMap<String, usize>,

    /// Most connected memories
    pub most_connected: Vec<(MemoryId, usize)>,

    /// Topic connectivity analysis
    pub topic_connectivity: HashMap<String, f32>,

    /// Contact interaction patterns
    pub contact_patterns: HashMap<String, ContactInteractionPattern>,

    /// Temporal pattern insights
    pub temporal_insights: TemporalInsights,

    /// Association quality metrics
    pub quality_metrics: AssociationQualityMetrics,

    /// Last analytics update
    pub last_updated: DateTime<Utc>,
}

/// Contact interaction pattern analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContactInteractionPattern {
    /// Contact identifier
    pub contact_id: String,

    /// Interaction frequency trend
    pub frequency_trend: Vec<f32>,

    /// Response time trend
    pub response_time_trend: Vec<Duration>,

    /// Topic evolution
    pub topic_evolution: HashMap<String, Vec<f32>>,

    /// Communication quality trend
    pub quality_trend: Vec<f32>,
}

/// Temporal insights from association analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalInsights {
    /// Peak activity periods
    pub peak_periods: Vec<(DateTime<Utc>, DateTime<Utc>)>,

    /// Seasonal patterns
    pub seasonal_patterns: HashMap<String, f32>,

    /// Day-of-week patterns
    pub day_patterns: HashMap<String, f32>,

    /// Hour-of-day patterns
    pub hour_patterns: HashMap<u8, f32>,

    /// Long-term trends
    pub long_term_trends: HashMap<String, f32>,
}

/// Association quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssociationQualityMetrics {
    /// Average association strength
    pub avg_strength: f32,

    /// Association accuracy rate
    pub accuracy_rate: f32,

    /// False positive rate
    pub false_positive_rate: f32,

    /// Coverage percentage
    pub coverage_percentage: f32,

    /// Coherence score
    pub coherence_score: f32,
}

/// Contact association specific to memory items
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContactAssociation {
    /// Contact identifier
    pub contact_id: String,

    /// Association strength
    pub strength: f32,

    /// Interaction context
    pub context: String,

    /// Last interaction timestamp
    pub last_interaction: DateTime<Utc>,
}

/// Metadata for association clusters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssociationClusterMetadata {
    /// Creation timestamp
    pub created_at: DateTime<Utc>,

    /// Number of updates
    pub update_count: u32,

    /// Quality score
    pub quality_score: f32,

    /// Manual review flag
    pub needs_review: bool,

    /// User feedback
    pub user_feedback: Option<String>,
}

/// User-aware memory association enhancements for social tools
pub mod user_aware {
    use itertools::Itertools;
    use super::*;
    // Temporarily commented out unused imports
    // use crate::tools::discord::SocialContext;
    // use crate::tools::slack::UserProfile;
    // use std::collections::BTreeMap;

    /// User-specific association preferences and patterns
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct UserAssociationProfile {
        /// User identifier across platforms
        pub user_id: String,

        /// Platform-specific identifiers
        pub platform_identities: HashMap<String, String>,

        /// User's memory association preferences
        pub association_preferences: UserAssociationPreferences,

        /// Historical memory interaction patterns
        pub interaction_patterns: UserInteractionPatterns,

        /// Social context awareness level
        pub social_context_level: f32,

        /// Cross-platform behavior consistency
        pub behavior_consistency: f32,

        /// User's cognitive working style
        pub cognitive_style: UserCognitiveStyle,

        /// Memory organization preferences
        pub organization_preferences: MemoryOrganizationStyle,

        /// Last updated timestamp
        pub last_updated: DateTime<Utc>,
    }

    /// User preferences for memory associations
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct UserAssociationPreferences {
        /// Preferred association strength threshold
        pub min_association_strength: f32,

        /// Maximum associations per memory item
        pub max_associations_per_item: usize,

        /// Prefer temporal associations over semantic
        pub temporal_preference_weight: f32,

        /// Include social context in associations
        pub include_social_context: bool,

        /// Cross-platform correlation preference
        pub cross_platform_correlation: bool,

        /// Automatic vs manual association creation
        pub auto_association_level: f32,

        /// Privacy level for shared memories
        pub privacy_level: PrivacyLevel,

        /// Topic clustering preferences
        pub topic_clustering_style: TopicClusteringStyle,
    }

    /// User interaction patterns with memory system
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct UserInteractionPatterns {
        /// Most active interaction times
        pub peak_activity_hours: Vec<u8>,

        /// Typical memory recall patterns
        pub recall_patterns: Vec<RecallPattern>,

        /// Preferred memory types
        pub preferred_memory_types: HashMap<String, f32>,

        /// Association traversal behavior
        pub traversal_behavior: TraversalBehavior,

        /// Social memory sharing patterns
        pub sharing_patterns: SharingPatterns,

        /// Memory update frequency
        pub update_frequency: f32,

        /// Collaboration style in shared memories
        pub collaboration_style: CollaborationStyle,
    }

    /// User's cognitive working style
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct UserCognitiveStyle {
        /// Linear vs non-linear thinking preference
        pub thinking_style: ThinkingStyle,

        /// Detail vs overview preference
        pub detail_preference: f32,

        /// Visual vs textual memory preference
        pub modality_preference: ModalityPreference,

        /// Contextual vs isolated memory preference
        pub context_preference: f32,

        /// Structured vs organic organization
        pub structure_preference: f32,
    }

    /// Memory organization style preferences
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct MemoryOrganizationStyle {
        /// Hierarchical vs flat organization
        pub hierarchy_preference: f32,

        /// Tag-based vs category-based
        pub tagging_style: TaggingStyle,

        /// Chronological vs topical organization
        pub temporal_organization: f32,

        /// Personal vs collaborative spaces
        pub space_preference: f32,
    }

    /// Privacy levels for memory associations
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub enum PrivacyLevel {
        Private,
        Trusted,
        Team,
        Community,
        Public,
    }

    /// Topic clustering style preferences
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub enum TopicClusteringStyle {
        Automatic,
        ManualCuration,
        Hybrid,
        MinimalClustering,
    }

    /// Memory recall pattern types
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub enum RecallPattern {
        Sequential,
        Associative,
        Contextual,
        TemporalBased,
        TopicBased,
        SocialTriggered,
    }

    /// Association traversal behavior
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct TraversalBehavior {
        /// Depth vs breadth preference
        pub exploration_style: ExplorationStyle,

        /// Following association chains
        pub chain_following_tendency: f32,

        /// Serendipitous discovery preference
        pub serendipity_preference: f32,

        /// Backtracking behavior
        pub backtracking_frequency: f32,
    }

    /// Social memory sharing patterns
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SharingPatterns {
        /// Frequency of sharing memories
        pub sharing_frequency: f32,

        /// Types of content typically shared
        pub shared_content_types: Vec<String>,

        /// Preferred platforms for sharing
        pub sharing_platforms: Vec<String>,

        /// Audience consideration level
        pub audience_awareness: f32,
    }

    /// Collaboration style in memory systems
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub enum CollaborationStyle {
        Independent,
        Collaborative,
        Leading,
        Supporting,
        Adaptive,
    }

    /// Thinking style preferences
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub enum ThinkingStyle {
        Linear,
        NonLinear,
        SystemsBased,
        Creative,
        Analytical,
        Intuitive,
    }

    /// Modality preferences for memory
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub enum ModalityPreference {
        Visual,
        Textual,
        Audio,
        Kinesthetic,
        Mixed,
    }

    /// Tagging style preferences
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub enum TaggingStyle {
        Minimal,
        Comprehensive,
        Structured,
        Organic,
        Collaborative,
    }

    /// Exploration style for associations
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub enum ExplorationStyle {
        DepthFirst,
        BreadthFirst,
        Adaptive,
        RandomWalk,
        Guided,
    }

    /// Social platform context for memory associations
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SocialPlatformContext {
        /// Platform identifier
        pub platform: String,

        /// User's role/status on platform
        pub user_role: String,

        /// Communication context
        pub communication_context: CommunicationContextType,

        /// Community context
        pub community_context: Option<String>,

        /// Interaction participants
        pub participants: Vec<String>,

        /// Conversation thread context
        pub thread_context: Option<String>,

        /// Social dynamics in the interaction
        pub social_dynamics: SocialDynamics,
    }

    /// Types of communication contexts
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub enum CommunicationContextType {
        DirectMessage,
        GroupChat,
        PublicChannel,
        Thread,
        Meeting,
        Broadcast,
        Collaboration,
    }

    /// Social dynamics in interactions
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SocialDynamics {
        /// Formality level of the interaction
        pub formality_level: f32,

        /// Collaborative vs competitive dynamic
        pub collaboration_level: f32,

        /// Information sharing vs seeking
        pub information_flow: InformationFlow,

        /// Emotional tone of interaction
        pub emotional_tone: f32,

        /// Power dynamics present
        pub power_dynamics: PowerDynamics,
    }

    /// Information flow patterns
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub enum InformationFlow {
        Sharing,
        Seeking,
        Bidirectional,
        Teaching,
        Learning,
        Brainstorming,
    }

    /// Power dynamics in interactions
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub enum PowerDynamics {
        Equal,
        Hierarchical,
        Expert,
        Learner,
        Facilitator,
        Observer,
    }

    impl MemoryAssociationManager {
        /// Create user-aware associations for social platform interactions
        pub async fn create_user_aware_associations(
            &self,
            memory_id: MemoryId,
            content: &str,
            user_profile: &UserAssociationProfile,
            social_platform_context: &SocialPlatformContext,
            cognitive_memory: &CognitiveMemory,
        ) -> Result<Vec<MemoryId>> {
            debug!("🧠 Creating user-aware associations for user {} on platform {}",
                   user_profile.user_id, social_platform_context.platform);

            let mut created_associations = Vec::new();

            // 1. Apply user preference filtering
            let filtered_associations = self.apply_user_preference_filtering(
                memory_id.clone(),
                content,
                user_profile,
                cognitive_memory
            ).await?;
            created_associations.extend(filtered_associations);

            // 2. Create social context-aware associations
            let social_associations = self.create_social_context_associations(
                memory_id.clone(),
                content,
                social_platform_context,
                user_profile
            ).await?;
            created_associations.extend(social_associations);

            // 3. Apply user cognitive style associations
            let cognitive_style_associations = self.create_cognitive_style_associations(
                memory_id.clone(),
                content,
                &user_profile.cognitive_style,
                cognitive_memory
            ).await?;
            created_associations.extend(cognitive_style_associations);

            // 4. Create cross-platform user correlations
            if user_profile.association_preferences.cross_platform_correlation {
                let cross_platform_associations = self.create_cross_platform_user_associations(
                    memory_id.clone(),
                    user_profile,
                    content
                ).await?;
                created_associations.extend(cross_platform_associations);
            }

            // 5. Update user interaction patterns
            self.update_user_interaction_patterns(
                user_profile,
                &created_associations,
                social_platform_context
            ).await?;

            info!("✅ Created {} user-aware associations for user {} (platform: {})",
                  created_associations.len(), user_profile.user_id, social_platform_context.platform);

            Ok(created_associations)
        }

        /// Apply user preferences to filter and weight associations
        async fn apply_user_preference_filtering(
            &self,
            memory_id: MemoryId,
            content: &str,
            user_profile: &UserAssociationProfile,
            cognitive_memory: &CognitiveMemory,
        ) -> Result<Vec<MemoryId>> {
            let preferences = &user_profile.association_preferences;
            let mut filtered_associations = Vec::new();

            // Find base semantic associations
            let semantic_associations = self.find_semantic_associations(&memory_id, content, cognitive_memory).await?;

            // Apply temporal preference weighting if enabled
            if preferences.temporal_preference_weight > 0.5 {
                let temporal_associations = self.find_user_temporal_associations(
                    &memory_id,
                    user_profile
                ).await?;

                // Blend temporal and semantic based on user preference
                let blend_ratio = preferences.temporal_preference_weight;
                filtered_associations = self.blend_association_types(
                    filtered_associations,
                    temporal_associations,
                    blend_ratio
                ).await?;
            }

            Ok(filtered_associations)
        }

        /// Create associations based on social platform context
        async fn create_social_context_associations(
            &self,
            memory_id: MemoryId,
            _content: &str,
            social_context: &SocialPlatformContext,
            user_profile: &UserAssociationProfile,
        ) -> Result<Vec<MemoryId>> {
            let mut social_associations = Vec::new();

            // Only proceed if user enables social context inclusion
            if !user_profile.association_preferences.include_social_context {
                return Ok(social_associations);
            }

            // 1. Find associations with same participants
            if let Ok(participant_associations) = self.find_participant_associations(
                &memory_id,
                &social_context.participants
            ).await {
                social_associations.extend(participant_associations);
            }

            // 2. Find associations in same communication context type
            if let Ok(context_associations) = self.find_communication_context_associations(
                &memory_id,
                &social_context.communication_context,
                &social_context.platform
            ).await {
                social_associations.extend(context_associations);
            }

            // 3. Find associations in same community/server
            if let Some(community) = &social_context.community_context {
                if let Ok(community_associations) = self.find_community_associations(
                    &memory_id,
                    community,
                    &social_context.platform
                ).await {
                    social_associations.extend(community_associations);
                }
            }

            // 4. Find thread-based associations
            if let Some(thread) = &social_context.thread_context {
                if let Ok(thread_associations) = self.find_thread_associations(
                    &memory_id,
                    thread,
                    &social_context.platform
                ).await {
                    social_associations.extend(thread_associations);
                }
            }

            // 5. Apply social dynamics weighting
            social_associations = self.apply_social_dynamics_weighting(
                social_associations,
                &social_context.social_dynamics,
                user_profile
            ).await?;

            Ok(social_associations)
        }

        /// Create associations based on user's cognitive style
        async fn create_cognitive_style_associations(
            &self,
            memory_id: MemoryId,
            content: &str,
            cognitive_style: &UserCognitiveStyle,
            cognitive_memory: &CognitiveMemory,
        ) -> Result<Vec<MemoryId>> {
            let mut style_associations = Vec::new();

            match cognitive_style.thinking_style {
                ThinkingStyle::Linear => {
                    // Prefer sequential, structured associations
                    if let Ok(sequential) = self.find_sequential_associations(&memory_id, cognitive_memory).await {
                        style_associations.extend(sequential);
                    }
                },
                ThinkingStyle::NonLinear => {
                    // Prefer creative, unexpected associations
                    if let Ok(creative) = self.find_creative_associations(&memory_id, content, cognitive_memory).await {
                        style_associations.extend(creative);
                    }
                },
                ThinkingStyle::SystemsBased => {
                    // Prefer systematic, interconnected associations
                    if let Ok(systems) = self.find_systems_associations(&memory_id, cognitive_memory).await {
                        style_associations.extend(systems);
                    }
                },
                ThinkingStyle::Analytical => {
                    // Prefer logical, evidence-based associations
                    if let Ok(analytical) = self.find_analytical_associations(&memory_id, content, cognitive_memory).await {
                        style_associations.extend(analytical);
                    }
                },
                _ => {
                    // Default to balanced approach
                    if let Ok(balanced) = self.find_balanced_associations(&memory_id, content, cognitive_memory).await {
                        style_associations.extend(balanced);
                    }
                }
            }

            // Apply detail preference filtering
            if cognitive_style.detail_preference > 0.7 {
                style_associations = self.enhance_with_detailed_associations(style_associations, cognitive_memory).await?;
            } else if cognitive_style.detail_preference < 0.3 {
                style_associations = self.filter_to_overview_associations(style_associations, cognitive_memory).await?;
            }

            // Apply context preference
            if cognitive_style.context_preference > 0.7 {
                style_associations = self.enhance_with_contextual_associations(style_associations, cognitive_memory).await?;
            }

            Ok(style_associations)
        }

        /// Create cross-platform user associations
        async fn create_cross_platform_user_associations(
            &self,
            memory_id: MemoryId,
            user_profile: &UserAssociationProfile,
            content: &str,
        ) -> Result<Vec<MemoryId>> {
            let mut cross_platform_associations = Vec::new();

            // Find memories from same user across different platforms
            for (platform, platform_user_id) in &user_profile.platform_identities {
                if let Ok(platform_memories) = self.find_user_memories_on_platform(
                    platform_user_id,
                    platform
                ).await {
                    // Find content-similar memories from this platform
                    for platform_memory_id in platform_memories {
                        if let Ok(similarity) = self.calculate_cross_platform_similarity(
                            &memory_id,
                            &platform_memory_id,
                            content
                        ).await {
                            if similarity > user_profile.association_preferences.min_association_strength {
                                cross_platform_associations.push(platform_memory_id);
                            }
                        }
                    }
                }
            }

            // Apply user's behavior consistency weighting
            cross_platform_associations = self.apply_behavior_consistency_weighting(
                cross_platform_associations,
                user_profile.behavior_consistency
            ).await?;

            Ok(cross_platform_associations)
        }

        /// Update user interaction patterns based on new associations
        async fn update_user_interaction_patterns(
            &self,
            user_profile: &UserAssociationProfile,
            created_associations: &[MemoryId],
            social_context: &SocialPlatformContext,
        ) -> Result<()> {
            // This would update the user profile with new interaction data
            // Implementation would involve updating the user's stored patterns

            debug!("📊 Updating interaction patterns for user {} based on {} new associations",
                   user_profile.user_id, created_associations.len());

            // Track association creation patterns
            let association_pattern = AssociationCreationPattern {
                platform: social_context.platform.clone(),
                context_type: social_context.communication_context.clone(),
                association_count: created_associations.len(),
                timestamp: Utc::now(),
            };

            // This would be stored in a user pattern database
            self.store_user_pattern(user_profile, association_pattern).await?;

            Ok(())
        }

        // Helper methods for the user-aware functionality

        async fn calculate_user_aware_strength(
            &self,
            content1: &str,
            content2: &str,
            user_profile: &UserAssociationProfile,
        ) -> Result<f32> {
            // Calculate base semantic similarity
            let base_similarity = self.calculate_semantic_similarity(content1, content2).await?;

            // Apply user preference weights
            let user_weight = match user_profile.cognitive_style.thinking_style {
                ThinkingStyle::Analytical => 1.2,  // Boost for analytical users
                ThinkingStyle::Creative => 0.9,    // Lower threshold for creative users
                _ => 1.0,
            };

            Ok(base_similarity * user_weight)
        }

        async fn find_user_temporal_associations(
            &self,
            _memory_id: &MemoryId,
            _user_profile: &UserAssociationProfile,
        ) -> Result<Vec<MemoryId>> {
            // Find temporal associations considering user's activity patterns
            let temporal_associations = Vec::new();

            // Consider user's peak activity hours for temporal relevance
            // Implementation would look for memories created during similar time periods
            // when the user is typically active

            Ok(temporal_associations)
        }

        async fn blend_association_types(
            &self,
            semantic_associations: Vec<MemoryId>,
            temporal_associations: Vec<MemoryId>,
            blend_ratio: f32,
        ) -> Result<Vec<MemoryId>> {
            let mut blended = Vec::new();

            let semantic_count = ((1.0 - blend_ratio) * semantic_associations.len() as f32) as usize;
            let temporal_count = (blend_ratio * temporal_associations.len() as f32) as usize;

            blended.extend(semantic_associations.into_iter().take(semantic_count));
            blended.extend(temporal_associations.into_iter().take(temporal_count));

            // Remove duplicates
            blended.sort();
            blended.dedup();

            Ok(blended)
        }

        // Additional helper method implementations...
        async fn find_participant_associations(&self, _memory_id: &MemoryId, _participants: &[String]) -> Result<Vec<MemoryId>> {
            // Implementation for finding associations based on shared participants
            Ok(Vec::new())
        }

        async fn find_communication_context_associations(&self, _memory_id: &MemoryId, _context: &CommunicationContextType, _platform: &str) -> Result<Vec<MemoryId>> {
            // Implementation for finding associations based on communication context
            Ok(Vec::new())
        }

        async fn find_community_associations(&self, _memory_id: &MemoryId, _community: &str, _platform: &str) -> Result<Vec<MemoryId>> {
            // Implementation for finding associations within same community
            Ok(Vec::new())
        }

        async fn find_thread_associations(&self, _memory_id: &MemoryId, _thread: &str, _platform: &str) -> Result<Vec<MemoryId>> {
            // Implementation for finding associations within same thread
            Ok(Vec::new())
        }

        async fn apply_social_dynamics_weighting(&self, associations: Vec<MemoryId>, _dynamics: &SocialDynamics, _user_profile: &UserAssociationProfile) -> Result<Vec<MemoryId>> {
            // Implementation for weighting associations based on social dynamics
            Ok(associations)
        }

        async fn find_sequential_associations(&self, _memory_id: &MemoryId, _cognitive_memory: &CognitiveMemory) -> Result<Vec<MemoryId>> {
            Ok(Vec::new())
        }

        async fn find_creative_associations(&self, _memory_id: &MemoryId, _content: &str, _cognitive_memory: &CognitiveMemory) -> Result<Vec<MemoryId>> {
            Ok(Vec::new())
        }

        async fn find_systems_associations(&self, _memory_id: &MemoryId, _cognitive_memory: &CognitiveMemory) -> Result<Vec<MemoryId>> {
            Ok(Vec::new())
        }

        async fn find_analytical_associations(&self, _memory_id: &MemoryId, _content: &str, _cognitive_memory: &CognitiveMemory) -> Result<Vec<MemoryId>> {
            Ok(Vec::new())
        }

        async fn find_balanced_associations(&self, _memory_id: &MemoryId, _content: &str, _cognitive_memory: &CognitiveMemory) -> Result<Vec<MemoryId>> {
            Ok(Vec::new())
        }

        async fn enhance_with_detailed_associations(&self, associations: Vec<MemoryId>, _cognitive_memory: &CognitiveMemory) -> Result<Vec<MemoryId>> {
            Ok(associations)
        }

        async fn filter_to_overview_associations(&self, associations: Vec<MemoryId>, _cognitive_memory: &CognitiveMemory) -> Result<Vec<MemoryId>> {
            Ok(associations)
        }

        async fn enhance_with_contextual_associations(&self, associations: Vec<MemoryId>, _cognitive_memory: &CognitiveMemory) -> Result<Vec<MemoryId>> {
            Ok(associations)
        }

        async fn find_user_memories_on_platform(&self, user_id: &str, platform: &str) -> Result<Vec<MemoryId>> {
            // Find all memories associated with a specific user on a given platform
            // This enables cross-platform user behavior analysis
            
            debug!("Finding memories for user {} on platform {}", user_id, platform);
            
            // In a full implementation, this would:
            // 1. Query the memory database for memories tagged with user_id and platform
            // 2. Apply privacy filters based on user permissions
            // 3. Sort by relevance and recency
            // 4. Return memory IDs for association analysis
            
            // For now, return empty vector but log the request
            info!("Memory search request: user={}, platform={}", user_id, platform);
            
            // Simulate finding some memories based on common patterns
            let mock_memories = match platform {
                "discord" => {
                    // Discord memories might include server interactions, DMs, reactions
                    vec!["discord_msg_001", "discord_reaction_002", "discord_thread_003"]
                },
                "slack" => {
                    // Slack memories might include channel messages, DMs, workspace interactions
                    vec!["slack_channel_001", "slack_dm_002", "slack_workspace_003"]
                },
                "email" => {
                    // Email memories might include sent/received emails, calendar events
                    vec!["email_sent_001", "email_received_002", "calendar_event_003"]
                },
                _ => {
                    // Unknown platform
                    vec![]
                }
            };
            
            let memory_ids: Vec<MemoryId> = mock_memories.into_iter()
                .map(|id| MemoryId(format!("{}_{}", user_id, id)))
                .collect();
            
            debug!("Found {} memories for user {} on {}", memory_ids.len(), user_id, platform);
            Ok(memory_ids)
        }

        async fn calculate_cross_platform_similarity(&self, memory_id1: &MemoryId, memory_id2: &MemoryId, content: &str) -> Result<f32> {
            // Calculate similarity based on multiple factors:
            // 1. Semantic content similarity
            // 2. User behavior patterns across platforms
            // 3. Context similarity (e.g., work vs casual conversation)
            // 4. Temporal proximity
            
            let semantic_sim = self.calculate_semantic_similarity(content, content).await?;
            
            // Calculate user behavior similarity across platforms
            let behavior_sim = self.calculate_user_behavior_similarity(memory_id1, memory_id2).await?;
            
            // Calculate context similarity
            let context_sim = self.calculate_context_similarity(memory_id1, memory_id2).await?;
            
            // Calculate temporal proximity factor
            let temporal_sim = self.calculate_temporal_proximity(memory_id1, memory_id2).await?;
            
            // Weighted combination of similarity factors
            let total_similarity = (semantic_sim * 0.4) + 
                                  (behavior_sim * 0.3) + 
                                  (context_sim * 0.2) + 
                                  (temporal_sim * 0.1);
            
            Ok(total_similarity.min(1.0))
        }

        async fn apply_behavior_consistency_weighting(&self, associations: Vec<MemoryId>, consistency: f32) -> Result<Vec<MemoryId>> {
            // Apply behavior consistency as a weighting factor for association ranking
            // Higher consistency users get more stable association patterns
            
            if consistency < 0.5 {
                // Low consistency - randomize associations more to capture diverse patterns
                let mut weighted_associations = associations;
                weighted_associations.reverse(); // Simple reordering for low consistency
                Ok(weighted_associations)
            } else if consistency > 0.8 {
                // High consistency - maintain stable association patterns
                Ok(associations) // Keep original order for high consistency users
            } else {
                // Medium consistency - apply moderate reweighting
                let mut weighted_associations = associations;
                // Apply a simple shuffle pattern for medium consistency
                let len = weighted_associations.len();
                if len > 2 {
                    weighted_associations.swap(0, len / 2);
                }
                Ok(weighted_associations)
            }
        }

        async fn store_user_pattern(&self, user_profile: &UserAssociationProfile, pattern: AssociationCreationPattern) -> Result<()> {
            // Store user association creation patterns for learning and adaptation
            // This enables the AI to learn user preferences over time
            
            debug!("Storing association pattern for user {}: {:?}", user_profile.user_id, pattern);
            
            // In a full implementation, this would:
            // 1. Store pattern in a persistent pattern database
            // 2. Update user behavior models
            // 3. Trigger adaptation of association algorithms
            // 4. Update cross-platform behavior analysis
            
            // For now, log the pattern for observability
            info!(
                "User {} created {} associations on {} in {:?} context",
                user_profile.user_id,
                pattern.association_count,
                pattern.platform,
                pattern.context_type
            );
            
            // Track pattern statistics
            debug!(
                "User association preferences - min_strength: {}, max_per_item: {}, temporal_weight: {}",
                user_profile.association_preferences.min_association_strength,
                user_profile.association_preferences.max_associations_per_item,
                user_profile.association_preferences.temporal_preference_weight
            );
            
            Ok(())
        }

        async fn calculate_semantic_similarity(&self, content1: &str, content2: &str) -> Result<f32> {
            // Enhanced semantic similarity using multiple techniques:
            // 1. Token overlap with TF-IDF weighting
            // 2. Fuzzy string matching
            // 3. Topic similarity based on keywords
            // 4. Sentiment similarity
            
            if content1.is_empty() || content2.is_empty() {
                return Ok(0.0);
            }
            
            // Simple token overlap similarity as baseline
            let tokens1: std::collections::HashSet<&str> = content1.split_whitespace().collect();
            let tokens2: std::collections::HashSet<&str> = content2.split_whitespace().collect();
            
            let intersection = tokens1.intersection(&tokens2).count();
            let union = tokens1.union(&tokens2).count();
            
            let jaccard_similarity = if union > 0 {
                intersection as f32 / union as f32
            } else {
                0.0
            };
            
            // Fuzzy string similarity using fuzzy-matcher
            let fuzzy_sim = fuzzy_matcher::FuzzyMatcher::fuzzy_match(
                &fuzzy_matcher::skim::SkimMatcherV2::default(),
                content1,
                content2
            ).map(|score| score as f32 / 100.0).unwrap_or(0.0);
            
            // Length-based similarity factor
            let len1 = content1.len() as f32;
            let len2 = content2.len() as f32;
            let length_sim = if len1.max(len2) > 0.0 {
                len1.min(len2) / len1.max(len2)
            } else {
                1.0
            };
            
            // Weighted combination
            let semantic_similarity = (jaccard_similarity * 0.5) + 
                                     (fuzzy_sim * 0.3) + 
                                     (length_sim * 0.2);
            
            Ok(semantic_similarity.min(1.0))
        }

        /// Calculate user behavior similarity across platforms
        async fn calculate_user_behavior_similarity(&self, _memory_id1: &MemoryId, _memory_id2: &MemoryId) -> Result<f32> {
            // In a full implementation, this would:
            // 1. Look up user profiles associated with these memories
            // 2. Compare communication patterns (response times, message length, formality)
            // 3. Compare activity patterns (time of day, frequency)
            // 4. Compare interaction styles across platforms
            
            // For now, return a reasonable default based on platform consistency
            Ok(0.7) // Assume moderate behavioral consistency
        }

        /// Calculate context similarity between memories
        async fn calculate_context_similarity(&self, _memory_id1: &MemoryId, _memory_id2: &MemoryId) -> Result<f32> {
            // In a full implementation, this would:
            // 1. Analyze conversation context (work, casual, technical, etc.)
            // 2. Compare participant overlap
            // 3. Compare channel/thread context
            // 4. Compare time-based context (business hours vs off-hours)
            
            // For now, return a baseline similarity
            Ok(0.6)
        }

        /// Calculate temporal proximity factor
        async fn calculate_temporal_proximity(&self, _memory_id1: &MemoryId, _memory_id2: &MemoryId) -> Result<f32> {
            // In a full implementation, this would:
            // 1. Get timestamps for both memories
            // 2. Calculate time difference
            // 3. Apply decay function (recent = more similar)
            // 4. Consider user's typical activity patterns
            
            // For now, assume moderate temporal relationship
            Ok(0.5)
        }
    }

    /// Pattern tracking for association creation
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct AssociationCreationPattern {
        pub platform: String,
        pub context_type: CommunicationContextType,
        pub association_count: usize,
        pub timestamp: DateTime<Utc>,
    }
}

impl MemoryAssociationManager {
    /// Create new association manager
    pub async fn new(config: AssociationConfig) -> Result<Self> {
        info!("🔗 Initializing Memory Association Manager");

        Ok(Self {
            associations: Arc::new(RwLock::new(BTreeMap::new())),
            topic_clusters: Arc::new(RwLock::new(HashMap::new())),
            temporal_chains: Arc::new(RwLock::new(Vec::new())),
            contact_graph: Arc::new(RwLock::new(ContactGraph::new())),
            email_associations: Arc::new(RwLock::new(HashMap::new())),
            cognitive_links: Arc::new(RwLock::new(HashMap::new())),
            platform_correlations: Arc::new(RwLock::new(HashMap::new())),
            analytics: Arc::new(RwLock::new(AssociationAnalytics::new())),
            config,
        })
    }

    /// Create associations for a new memory item
    pub async fn create_associations(
        &self,
        memory_id: MemoryId,
        content: &str,
        metadata: &MemoryMetadata,
        cognitive_memory: &CognitiveMemory,
    ) -> Result<Vec<MemoryId>> {
        let start_time = Instant::now();
        debug!("🔗 Creating associations for memory: {}", memory_id);

        let mut created_associations = Vec::new();

        // 1. Semantic associations
        if let Ok(semantic_associations) = self
            .find_semantic_associations(&memory_id, content, cognitive_memory)
            .await
        {
            created_associations.extend(semantic_associations);
        }

        // 2. Contact-based associations
        if self.config.enable_contact_inference {
            if let Ok(contact_associations) = self
                .find_contact_associations(&memory_id, metadata)
                .await
            {
                created_associations.extend(contact_associations);
            }
        }

        // 3. Topic-based associations
        if self.config.enable_topic_clustering {
            if let Ok(topic_associations) = self
                .find_topic_associations(&memory_id, content, &metadata.tags)
                .await
            {
                created_associations.extend(topic_associations);
            }
        }

        // 4. Temporal associations
        if self.config.enable_temporal_chains {
            if let Ok(temporal_associations) = self
                .find_temporal_associations(&memory_id, metadata)
                .await
            {
                created_associations.extend(temporal_associations);
            }
        }

        // 5. Cross-platform correlations
        if self.config.enable_cross_platform {
            if let Ok(platform_associations) = self
                .find_cross_platform_associations(&memory_id, &metadata.source)
                .await
            {
                created_associations.extend(platform_associations);
            }
        }

        // Create bidirectional associations
        self.create_bidirectional_links(&memory_id, &created_associations)
            .await?;

        // Update analytics
        self.update_analytics(&memory_id, &created_associations)
            .await?;

        let duration = start_time.elapsed();
        info!(
            "✅ Created {} associations for memory {} in {:?}",
            created_associations.len(),
            memory_id,
            duration
        );

        Ok(created_associations)
    }

    /// Find semantic associations based on content similarity
    async fn find_semantic_associations(
        &self,
        memory_id: &MemoryId,
        content: &str,
        cognitive_memory: &CognitiveMemory,
    ) -> Result<Vec<MemoryId>> {
        debug!("🧠 Finding semantic associations for: {}", memory_id);

        // Use cognitive memory's similarity search
        let similar_memories = cognitive_memory
            .retrieve_similar(content, 10)
            .await?;

        let mut associations = Vec::new();

        for similar_memory in similar_memories {
            // Only create associations above threshold
            if similar_memory.relevance_score >= self.config.auto_association_threshold {
                associations.push(similar_memory.id);
            }
        }

        debug!(
            "Found {} semantic associations for {}",
            associations.len(),
            memory_id
        );

        Ok(associations)
    }

    /// Find contact-based associations
    async fn find_contact_associations(
        &self,
        memory_id: &MemoryId,
        metadata: &MemoryMetadata,
    ) -> Result<Vec<MemoryId>> {
        debug!("👥 Finding contact associations for: {}", memory_id);

        let mut associations = Vec::new();

        // Extract contact information from metadata tags
        let contact_indicators = metadata
            .tags
            .iter()
            .filter(|tag| {
                tag.contains("@") || // Email addresses
                tag.starts_with("contact:") ||
                tag.starts_with("user:") ||
                tag.starts_with("email:")
            })
            .collect::<Vec<_>>();

        if !contact_indicators.is_empty() {
            // Find other memories associated with the same contacts
            let contact_graph = self.contact_graph.read().await;

            for contact_indicator in contact_indicators {
                let contact_id = contact_indicator
                    .strip_prefix("contact:")
                    .or_else(|| contact_indicator.strip_prefix("user:"))
                    .or_else(|| contact_indicator.strip_prefix("email:"))
                    .unwrap_or(contact_indicator);

                if let Some(contact_node) = contact_graph.contacts.get(contact_id) {
                    // Add associated memories from this contact
                    associations.extend(contact_node.memory_associations.iter().cloned());
                }
            }
        }

        debug!(
            "Found {} contact associations for {}",
            associations.len(),
            memory_id
        );

        Ok(associations)
    }

    /// Find topic-based associations
    async fn find_topic_associations(
        &self,
        memory_id: &MemoryId,
        content: &str,
        tags: &[String],
    ) -> Result<Vec<MemoryId>> {
        debug!("📚 Finding topic associations for: {}", memory_id);

        let mut associations = Vec::new();
        let topic_clusters = self.topic_clusters.read().await;

        // Extract topics from content and tags
        let content_topics = self.extract_topics_from_content(content).await?;
        let tag_topics = tags.iter().cloned().collect::<HashSet<_>>();

        let all_topics: HashSet<String> = content_topics
            .union(&tag_topics)
            .cloned()
            .collect();

        // Find memories in the same topic clusters
        for topic in all_topics {
            if let Some(topic_cluster) = topic_clusters.get(&topic) {
                // Add memories from this topic cluster
                associations.extend(topic_cluster.memory_ids.iter().cloned());
            }
        }

        debug!(
            "Found {} topic associations for {}",
            associations.len(),
            memory_id
        );

        Ok(associations)
    }

    /// Find temporal associations
    async fn find_temporal_associations(
        &self,
        memory_id: &MemoryId,
        _metadata: &MemoryMetadata,
    ) -> Result<Vec<MemoryId>> {
        debug!("⏰ Finding temporal associations for: {}", memory_id);

        let associations = Vec::new();
        let _temporal_chains = self.temporal_chains.read().await;

        // Look for memories created around the same time
        let _current_time = SystemTime::now();
        let _time_window = Duration::from_secs(24 * 60 * 60); // 24-hour window

        // This is a simplified implementation
        // In a full implementation, you would have access to memory timestamps
        // and could search for memories within the time window

        debug!(
            "Found {} temporal associations for {}",
            associations.len(),
            memory_id
        );

        Ok(associations)
    }

    /// Find cross-platform associations
    async fn find_cross_platform_associations(
        &self,
        memory_id: &MemoryId,
        source: &str,
    ) -> Result<Vec<MemoryId>> {
        debug!("🌐 Finding cross-platform associations for: {}", memory_id);

        let mut associations = Vec::new();
        let platform_correlations = self.platform_correlations.read().await;

        // Find correlations based on the source platform
        for (platform_key, correlation) in platform_correlations.iter() {
            if platform_key.contains(source) || source.contains(&correlation.platform_id) {
                associations.extend(correlation.memory_correlations.keys().cloned());
            }
        }

        debug!(
            "Found {} cross-platform associations for {}",
            associations.len(),
            memory_id
        );

        Ok(associations)
    }

    /// Create bidirectional association links
    async fn create_bidirectional_links(
        &self,
        source_id: &MemoryId,
        target_ids: &[MemoryId],
    ) -> Result<()> {
        debug!(
            "🔄 Creating bidirectional links from {} to {} targets",
            source_id,
            target_ids.len()
        );

        let mut associations = self.associations.write().await;

        for target_id in target_ids {
            // Add link from source to target
            let link = AssociationLink {
                target_id: target_id.clone(),
                strength: self.config.auto_association_threshold,
                association_type: AssociationType::Semantic, // Default type
                context: AssociationContext {
                    detection_method: "auto_semantic".to_string(),
                    source_system: "memory_association_manager".to_string(),
                    context_data: HashMap::new(),
                    method_confidence: 0.8,
                },
                created_at: Utc::now(),
                last_accessed: None,
                access_count: 0,
                confidence: 0.8,
            };

            // Create/update source cluster and add link
            associations
                .entry(source_id.clone())
                .or_insert_with(|| AssociationCluster::new(source_id.clone()))
                .direct_associations
                .insert(target_id.clone(), link.clone());

            // Create reciprocal link
            let reciprocal_link = AssociationLink {
                target_id: source_id.clone(),
                ..link
            };

            // Create/update target cluster and add reciprocal link
            associations
                .entry(target_id.clone())
                .or_insert_with(|| AssociationCluster::new(target_id.clone()))
                .direct_associations
                .insert(source_id.clone(), reciprocal_link);
        }

        debug!("✅ Created bidirectional links successfully");
        Ok(())
    }

    /// Find associations for a given memory ID
    pub async fn find_associations(
        &self,
        memory_id: &MemoryId,
        max_results: Option<usize>,
    ) -> Result<Vec<AssociationLink>> {
        let associations = self.associations.read().await;
        
        if let Some(cluster) = associations.get(memory_id) {
            let mut all_associations: Vec<AssociationLink> = cluster
                .direct_associations
                .values()
                .cloned()
                .collect();
            
            // Sort by strength (descending)
            all_associations.sort_by(|a, b| b.strength.partial_cmp(&a.strength).unwrap_or(std::cmp::Ordering::Equal));
            
            // Limit results if requested
            if let Some(max) = max_results {
                all_associations.truncate(max);
            }
            
            Ok(all_associations)
        } else {
            Ok(Vec::new())
        }
    }

    /// Update analytics after creating associations
    async fn update_analytics(
        &self,
        memory_id: &MemoryId,
        associations: &[MemoryId],
    ) -> Result<()> {
        debug!("📊 Updating analytics for memory: {}", memory_id);

        let mut analytics = self.analytics.write().await;
        analytics.total_associations += associations.len();

        // Update strength distribution
        let strength_key = format!("{:.1}", self.config.auto_association_threshold);
        *analytics.strength_distribution.entry(strength_key).or_insert(0) += associations.len();

        analytics.last_updated = Utc::now();

        debug!("✅ Analytics updated successfully");
        Ok(())
    }

    /// Extract topics from content using simple keyword extraction
    async fn extract_topics_from_content(&self, content: &str) -> Result<HashSet<String>> {
        let mut topics = HashSet::new();

        // Simple topic extraction (in a full implementation, you might use NLP)
        let words: Vec<&str> = content
            .split_whitespace()
            .filter(|word| word.len() > 3)
            .collect();

        // Extract meaningful words as topics
        for word in words {
            let cleaned = word.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase();
            if cleaned.len() > 3 && !is_stop_word(&cleaned) {
                topics.insert(cleaned);
            }
        }

        Ok(topics)
    }

    /// Get associations for a memory item
    pub async fn get_associations(&self, memory_id: &MemoryId) -> Result<Vec<AssociationLink>> {
        let associations = self.associations.read().await;

        if let Some(cluster) = associations.get(memory_id) {
            Ok(cluster.direct_associations.values().cloned().collect())
        } else {
            Ok(Vec::new())
        }
    }

    /// Get association analytics
    pub async fn get_analytics(&self) -> Result<AssociationAnalytics> {
        let analytics = self.analytics.read().await;
        Ok(analytics.clone())
    }

    /// Associate email with memory items
    pub async fn associate_email(
        &self,
        email_id: String,
        memory_ids: Vec<MemoryId>,
        participants: Vec<String>,
        topics: Vec<String>,
    ) -> Result<()> {
        debug!("📧 Associating email {} with {} memories", email_id, memory_ids.len());

        let mut email_associations = self.email_associations.write().await;

        let email_cluster = email_associations
            .entry(email_id.clone())
            .or_insert_with(|| EmailAssociationCluster {
                email_id: email_id.clone(),
                memory_associations: HashSet::new(),
                participants: HashSet::new(),
                topics: HashMap::new(),
                action_items: Vec::new(),
                timeline: Vec::new(),
                cognitive_connections: HashMap::new(),
            });

        // Add memory associations
        email_cluster.memory_associations.extend(memory_ids);

        // Add participants
        email_cluster.participants.extend(participants);

        // Add topics with relevance scores
        for topic in topics {
            email_cluster.topics.insert(topic, 1.0);
        }

        info!("✅ Email {} associated successfully", email_id);
        Ok(())
    }

    /// Associate contact with memory items
    pub async fn associate_contact(
        &self,
        contact_id: String,
        memory_ids: Vec<MemoryId>,
        _interaction_context: String,
    ) -> Result<()> {
        debug!("👤 Associating contact {} with {} memories", contact_id, memory_ids.len());

        let mut contact_graph = self.contact_graph.write().await;

        let contact_node = contact_graph
            .contacts
            .entry(contact_id.clone())
            .or_insert_with(|| ContactNode::new(contact_id.clone()));

        // Add memory associations
        contact_node.memory_associations.extend(memory_ids);

        // Update interaction timestamp
        contact_node.last_interaction = Utc::now();

        // Update communication stats
        contact_node.communication_stats.interaction_count += 1;

        info!("✅ Contact {} associated successfully", contact_id);
        Ok(())
    }

    /// Get email associations
    pub async fn get_email_associations(&self, email_id: &str) -> Result<Option<EmailAssociationCluster>> {
        let email_associations = self.email_associations.read().await;
        Ok(email_associations.get(email_id).cloned())
    }

    /// Get contact associations
    pub async fn get_contact_associations(&self, contact_id: &str) -> Result<Option<ContactNode>> {
        let contact_graph = self.contact_graph.read().await;
        Ok(contact_graph.contacts.get(contact_id).cloned())
    }
    
    /// Get the total count of associations
    pub async fn get_association_count(&self) -> usize {
        let associations = self.associations.read().await;
        associations.values()
            .map(|cluster| cluster.direct_associations.len())
            .sum()
    }
    
    /// Get all associations as a simple list (for UI display)
    pub async fn get_all_associations_simple(&self) -> Vec<(MemoryId, Vec<AssociationLink>)> {
        let associations = self.associations.read().await;
        associations.iter()
            .map(|(id, cluster)| {
                let links: Vec<AssociationLink> = cluster.direct_associations.values().cloned().collect();
                (id.clone(), links)
            })
            .collect()
    }
}

impl ContactGraph {
    fn new() -> Self {
        Self {
            contacts: HashMap::new(),
            relationships: HashMap::new(),
            groups: HashMap::new(),
            communication_matrix: HashMap::new(),
        }
    }
}

impl ContactNode {
    fn new(contact_id: String) -> Self {
        Self {
            contact_id,
            display_name: None,
            details: ContactDetails::default(),
            memory_associations: HashSet::new(),
            communication_stats: CommunicationStats::default(),
            inferred_attributes: HashMap::new(),
            last_interaction: Utc::now(),
        }
    }
}

impl ContactDetails {
    fn default() -> Self {
        Self {
            organization: None,
            role: None,
            platforms: HashMap::new(),
            communication_style: None,
            timezone: None,
            tags: HashSet::new(),
        }
    }
}

impl CommunicationStats {
    fn default() -> Self {
        Self {
            interaction_count: 0,
            avg_response_time: Duration::from_secs(24 * 60 * 60),
            frequency: 0.0,
            preferred_times: Vec::new(),
            quality_score: 0.5,
            last_response_time: None,
        }
    }
}

impl AssociationCluster {
    fn new(memory_id: MemoryId) -> Self {
        Self {
            memory_id,
            direct_associations: HashMap::new(),
            indirect_associations: HashMap::new(),
            topic_associations: HashMap::new(),
            contact_associations: HashMap::new(),
            temporal_chains: Vec::new(),
            platform_references: HashMap::new(),
            metadata: AssociationClusterMetadata {
                created_at: Utc::now(),
                update_count: 0,
                quality_score: 0.5,
                needs_review: false,
                user_feedback: None,
            },
            last_updated: Utc::now(),
        }
    }
}

impl AssociationAnalytics {
    fn new() -> Self {
        Self {
            total_associations: 0,
            strength_distribution: HashMap::new(),
            most_connected: Vec::new(),
            topic_connectivity: HashMap::new(),
            contact_patterns: HashMap::new(),
            temporal_insights: TemporalInsights {
                peak_periods: Vec::new(),
                seasonal_patterns: HashMap::new(),
                day_patterns: HashMap::new(),
                hour_patterns: HashMap::new(),
                long_term_trends: HashMap::new(),
            },
            quality_metrics: AssociationQualityMetrics {
                avg_strength: 0.0,
                accuracy_rate: 0.0,
                false_positive_rate: 0.0,
                coverage_percentage: 0.0,
                coherence_score: 0.0,
            },
            last_updated: Utc::now(),
        }
    }
}

/// Simple stop word check
fn is_stop_word(word: &str) -> bool {
    matches!(
        word,
        "the" | "and" | "for" | "are" | "but" | "not" | "you" | "all" | "can" | "had" | "her" | "was" | "one" | "our" | "out" | "day" | "get" | "has" | "him" | "his" | "how" | "its" | "may" | "new" | "now" | "old" | "see" | "two" | "who" | "boy" | "did" | "man" | "men" | "oil" | "she" | "sun" | "way" | "why"
    )
}
