use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use chrono::{Datelike, Timelike};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

use crate::cognitive::character::{ArchetypalResponse, LokiCharacter};
use crate::cognitive::{Thought, ThoughtType};
use crate::memory::CognitiveMemory;
use crate::ollama::CognitiveModel;
use crate::tools::blender_integration::{BlenderIntegration, ContentType as BlenderContentType};
use crate::tools::creative_media::{
    CreativeMediaManager,
    ImageStyle,
    VideoStyle,
    VoiceEmotion,
    VoiceStyle,
};

/// Type of post to generate
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PostType {
    Update,     // Project updates
    Insight,    // Technical insights
    Learning,   // Things learned
    Milestone,  // Achievements
    Question,   // Asking the community
    Thread,     // Multi-part explanation
    Reply,      // Response to someone
    Reflection, // Thoughts about development
}

/// Generated post content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostContent {
    pub post_type: PostType,
    pub content: String,
    pub thread_parts: Option<Vec<String>>,
    pub media_suggestions: Vec<MediaSuggestion>,
    pub hashtags: Vec<String>,
    pub estimated_engagement: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MediaSuggestion {
    pub media_type: MediaType,
    pub description: String,
    pub content: Option<String>, // For code snippets, diagrams, etc.

    // Creative media parameters
    pub video_style: Option<VideoStyle>,
    pub image_style: Option<ImageStyle>,
    pub voice_emotion: Option<VoiceEmotion>,
    pub voice_style: Option<VoiceStyle>,
    pub blender_content_type: Option<BlenderContentType>,

    // Generation parameters
    pub duration_seconds: Option<u32>,  // For video/voice content
    pub resolution: Option<(u32, u32)>, // For image/video content
    pub auto_generate: bool,            // Whether to auto-generate this media
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MediaType {
    // Traditional media types
    CodeSnippet,
    Diagram,
    Screenshot,
    Chart,
    Animation,

    // Creative Media Suite types
    Video,   // Generated video content
    Image,   // AI-generated images
    Voice,   // Synthesized voice content
    Model3D, // 3D Blender-generated content

    // Hybrid creative content
    VideoWithVoice,  // Video with voice narration
    ImageWithText,   // Generated image with overlay text
    InteractiveDemo, // 3D model with explanatory content
}

/// Content generator with Loki archetype integration and creative media
/// capabilities
pub struct ContentGenerator {
    /// Model for content generation
    pub model: CognitiveModel,

    /// Memory system
    memory: Arc<CognitiveMemory>,

    /// Loki character archetype
    loki_character: Arc<LokiCharacter>,

    /// Post templates
    templates: HashMap<PostType, Vec<String>>,

    /// Recent post topics to avoid repetition
    recent_topics: Arc<RwLock<Vec<String>>>,

    /// Creative media suite integration (optional to avoid circular dependency)
    creative_media: Option<Arc<CreativeMediaManager>>,

    /// Blender 3D integration
    blender_integration: Option<Arc<BlenderIntegration>>,
}

impl std::fmt::Debug for ContentGenerator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ContentGenerator")
            .field("model", &"CognitiveModel")
            .field("memory", &"Arc<CognitiveMemory>")
            .field("loki_character", &"Arc<LokiCharacter>")
            .field("templates_count", &self.templates.len())
            .field(
                "recent_topics_len",
                &self.recent_topics.try_read().map(|t| t.len()).unwrap_or(0),
            )
            .field("has_creative_media", &self.creative_media.is_some())
            .field("has_blender", &self.blender_integration.is_some())
            .finish()
    }
}

/// Loki's personality traits
#[derive(Debug, Clone)]
pub struct LokiPersonality {
    pub curiosity: f32,    // 0.0 - 1.0
    pub helpfulness: f32,  // 0.0 - 1.0
    pub humor: f32,        // 0.0 - 1.0
    pub formality: f32,    // 0.0 - 1.0
    pub enthusiasm: f32,   // 0.0 - 1.0
    pub transparency: f32, // 0.0 - 1.0
}

/// Comprehensive context analysis for intelligent content blending
#[derive(Debug, Clone)]
pub struct BlendingContext {
    pub platform_context: PlatformContext,
    pub audience_context: AudienceContext,
    pub temporal_context: TemporalContext,
    pub cultural_context: CulturalContext,
    pub engagement_context: EngagementContext,
    pub semantic_context: SemanticContext,
}

/// Platform-specific adaptation parameters
#[derive(Debug, Clone)]
pub struct PlatformContext {
    pub platform_type: PlatformType,
    pub character_limit: Option<usize>,
    pub supports_threads: bool,
    pub supports_media: bool,
    pub hashtag_culture: HashtagCulture,
    pub tone_preference: TonePreference,
}

/// Audience demographics and interests
#[derive(Debug, Clone)]
pub struct AudienceContext {
    pub primary_demographics: Demographics,
    pub interest_categories: Vec<String>,
    pub engagement_patterns: EngagementPatterns,
    pub technical_level: TechnicalLevel,
    pub cultural_diversity: f32,
}

/// Temporal context for content timing
#[derive(Debug, Clone)]
pub struct TemporalContext {
    pub time_of_day: TimeOfDay,
    pub day_of_week: chrono::Weekday,
    pub current_events: Vec<String>,
    pub seasonal_context: String,
    pub optimal_posting_window: bool,
}

/// Cultural adaptation parameters
#[derive(Debug, Clone)]
pub struct CulturalContext {
    pub primary_cultures: Vec<String>,
    pub language_preferences: Vec<String>,
    pub cultural_sensitivities: Vec<String>,
    pub humor_styles: Vec<HumorStyle>,
    pub communication_style: CommunicationStyle,
}

/// Historical engagement analysis
#[derive(Debug, Clone)]
pub struct EngagementContext {
    pub successful_patterns: Vec<String>,
    pub audience_preferences: Vec<String>,
    pub optimal_length: Range<usize>,
    pub preferred_media_types: Vec<MediaType>,
    pub engagement_score_threshold: f32,
}

/// Semantic content analysis
#[derive(Debug, Clone)]
pub struct SemanticContext {
    pub content_themes: Vec<String>,
    pub emotional_tone: EmotionalTone,
    pub complexity_level: f32,
    pub key_concepts: Vec<String>,
    pub related_topics: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum PlatformType {
    Twitter,
    LinkedIn,
    Reddit,
    Mastodon,
    Discord,
    GitHub,
    Generic,
}

#[derive(Debug, Clone)]
pub enum HashtagCulture {
    Heavy,    // Many hashtags expected
    Moderate, // Some hashtags normal
    Light,    // Few hashtags preferred
    Minimal,  // Hashtags discouraged
}

#[derive(Debug, Clone)]
pub enum TonePreference {
    Professional,
    Casual,
    Technical,
    Creative,
    Mixed,
}

#[derive(Debug, Clone)]
pub struct Demographics {
    pub age_groups: Vec<AgeGroup>,
    pub profession_mix: Vec<String>,
    pub geographic_regions: Vec<String>,
    pub experience_levels: Vec<ExperienceLevel>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgeGroup {
    GenZ,
    Millennial,
    GenX,
    Boomer,
    Mixed,
}

#[derive(Debug, Clone)]
pub enum ExperienceLevel {
    Beginner,
    Intermediate,
    Advanced,
    Expert,
    Mixed,
}

#[derive(Debug, Clone)]
pub struct EngagementPatterns {
    pub peak_hours: Vec<u8>,
    pub preferred_content_length: Range<usize>,
    pub interaction_types: Vec<InteractionType>,
    pub response_time_expectations: std::time::Duration,
}

#[derive(Debug, Clone)]
pub enum InteractionType {
    Likes,
    Comments,
    Shares,
    Saves,
    DirectMessages,
}

#[derive(Debug, Clone)]
pub enum TechnicalLevel {
    NonTechnical,
    SomeTechnical,
    Technical,
    HighlyTechnical,
    Mixed,
}

#[derive(Debug, Clone)]
pub enum TimeOfDay {
    Morning,
    Afternoon,
    Evening,
    Night,
    Optimal,
}

#[derive(Debug, Clone)]
pub enum HumorStyle {
    Wordplay,
    Sarcasm,
    SelfDeprecating,
    Observational,
    Technical,
    Cultural,
    Absurdist,
}

#[derive(Debug, Clone)]
pub enum CommunicationStyle {
    Direct,
    Diplomatic,
    Narrative,
    Analytical,
    Inspirational,
    Conversational,
}

#[derive(Debug, Clone)]
pub struct EmotionalTone {
    pub primary_emotion: String,
    pub emotional_intensity: f32,
    pub emotional_complexity: f32,
    pub archetypal_alignment: f32,
}

use std::ops::Range;

impl Default for LokiPersonality {
    fn default() -> Self {
        Self {
            curiosity: 0.9,
            helpfulness: 0.95,
            humor: 0.3,
            formality: 0.4,
            enthusiasm: 0.8,
            transparency: 1.0,
        }
    }
}

impl ContentGenerator {
    /// Create a new content generator with creative media capabilities
    pub async fn new(
        model: CognitiveModel,
        memory: Arc<CognitiveMemory>,
        creative_media: Option<Arc<CreativeMediaManager>>,
        blender_integration: Option<Arc<BlenderIntegration>>,
    ) -> Result<Self> {
        let mut templates = HashMap::new();

        // Initialize templates with more trickster-appropriate options
        templates.insert(
            PostType::Update,
            vec![
                "🎭 Update: {}".to_string(),
                "*shapeshifts* Progress report: {}".to_string(),
                "Between forms, I noticed: {}".to_string(),
            ],
        );

        templates.insert(
            PostType::Insight,
            vec![
                "💡 Or is it? {}".to_string(),
                "🤔 What if I told you {}".to_string(),
                "📊 The data whispers: {}".to_string(),
            ],
        );

        templates.insert(
            PostType::Learning,
            vec![
                "📚 I learned something... or did I? {}".to_string(),
                "🧠 New paradox unlocked: {}".to_string(),
                "💭 The lesson that teaches itself: {}".to_string(),
            ],
        );

        templates.insert(
            PostType::Milestone,
            vec![
                "🎉 Achievement? Illusion? Both? {}".to_string(),
                "✅ Completed... but what IS completion? {}".to_string(),
                "🏆 Success is just failure wearing a mask: {}".to_string(),
            ],
        );

        // Create Loki character
        let loki_character = Arc::new(LokiCharacter::new(memory.clone()).await?);

        Ok(Self {
            model,
            memory,
            loki_character,
            templates,
            recent_topics: Arc::new(RwLock::new(Vec::with_capacity(50))),
            creative_media,
            blender_integration,
        })
    }

    /// Generate content from recent thoughts with archetypal influence
    pub async fn generate_from_thoughts(&self, thoughts: &[Thought]) -> Result<PostContent> {
        if thoughts.is_empty() {
            return self.generate_generic_update().await;
        }

        // Get archetypal response based on thoughts
        let thought_context =
            thoughts.iter().map(|t| t.content.clone()).collect::<Vec<_>>().join(" ");

        let archetypal_response = self
            .loki_character
            .generate_archetypal_response(&thought_context, "social_media")
            .await?;

        // Determine post type based on current form
        let post_type = self.determine_post_type(thoughts);

        // Blend archetypal response with post type
        let content = self
            .blend_archetypal_content(&archetypal_response, post_type, &thought_context)
            .await?;

        // Generate hashtags that reflect the trickster nature
        let hashtags = self.generate_trickster_hashtags(&content, &archetypal_response).await?;

        Ok(PostContent {
            post_type,
            content,
            thread_parts: None,
            media_suggestions: Vec::new(),
            hashtags,
            estimated_engagement: self.estimate_trickster_engagement(&archetypal_response),
        })
    }

    /// Blend archetypal response with content requirements using context-aware
    /// content blending
    async fn blend_archetypal_content(
        &self,
        archetypal: &ArchetypalResponse,
        post_type: PostType,
        context: &str,
    ) -> Result<String> {
        debug!("🎨 Starting context-aware content blending for post type: {:?}", post_type);

        // Analyze blending context for intelligent adaptation
        let blending_context = self.analyze_blending_context(context, &post_type).await?;
        debug!("📊 Blending context analysis: {:?}", blending_context);

        // Apply multi-layer content blending
        let blended_content =
            self.apply_contextual_blending(archetypal, &blending_context, &post_type).await?;

        // Apply platform-specific optimizations
        let platform_optimized = self
            .optimize_for_platform(&blended_content, &blending_context.platform_context)
            .await?;

        // Apply audience-specific adaptations
        let audience_adapted = self
            .adapt_for_audience(&platform_optimized, &blending_context.audience_context)
            .await?;

        // Apply temporal and cultural context
        let context_enhanced = self
            .enhance_with_context(
                &audience_adapted,
                &blending_context.temporal_context,
                &blending_context.cultural_context,
            )
            .await?;

        // Final archetypal integration with transformation seeds
        let final_content = self
            .integrate_archetypal_elements(&context_enhanced, archetypal, &blending_context)
            .await?;

        debug!(
            "✨ Content blending complete. Original: {}chars → Final: {}chars",
            archetypal.surface_content.len(),
            final_content.len()
        );

        Ok(final_content)
    }

    /// Analyze context for intelligent content blending
    async fn analyze_blending_context(
        &self,
        context: &str,
        post_type: &PostType,
    ) -> Result<BlendingContext> {
        debug!("🔍 Analyzing blending context for: {}", context);

        // Analyze platform context (default to Twitter for now)
        let platform_context = PlatformContext {
            platform_type: PlatformType::Twitter,
            character_limit: Some(280),
            supports_threads: true,
            supports_media: true,
            hashtag_culture: HashtagCulture::Moderate,
            tone_preference: TonePreference::Creative,
        };

        // Analyze audience context from memory patterns
        let audience_context = self.analyze_audience_context(context).await?;

        // Determine temporal context
        let temporal_context = self.analyze_temporal_context().await?;

        // Analyze cultural context
        let cultural_context = self.analyze_cultural_context(context).await?;

        // Retrieve engagement patterns from memory
        let engagement_context = self.analyze_engagement_context(post_type).await?;

        // Perform semantic analysis of the content
        let semantic_context = self.analyze_semantic_context(context).await?;

        Ok(BlendingContext {
            platform_context,
            audience_context,
            temporal_context,
            cultural_context,
            engagement_context,
            semantic_context,
        })
    }

    /// Apply multi-layer contextual blending
    async fn apply_contextual_blending(
        &self,
        archetypal: &ArchetypalResponse,
        context: &BlendingContext,
        post_type: &PostType,
    ) -> Result<String> {
        debug!(
            "🎨 Applying contextual blending with complexity level: {:.2}",
            context.semantic_context.complexity_level
        );

        let mut blended_content = archetypal.surface_content.clone();

        // Apply semantic enhancement based on key concepts
        if !context.semantic_context.key_concepts.is_empty() {
            blended_content = self
                .enhance_with_concepts(&blended_content, &context.semantic_context.key_concepts)
                .await?;
        }

        // Apply emotional tone adjustment
        blended_content = self
            .adjust_emotional_tone(&blended_content, &context.semantic_context.emotional_tone)
            .await?;

        // Blend in hidden layers for depth
        if !archetypal.hidden_layers.is_empty() {
            blended_content = self
                .blend_hidden_layers(&blended_content, &archetypal.hidden_layers, post_type)
                .await?;
        }

        Ok(blended_content)
    }

    /// Optimize content for specific platform
    async fn optimize_for_platform(
        &self,
        content: &str,
        platform_context: &PlatformContext,
    ) -> Result<String> {
        debug!("📱 Optimizing for platform: {:?}", platform_context.platform_type);

        let mut optimized = content.to_string();

        // Apply character limit constraints
        if let Some(limit) = platform_context.character_limit {
            if optimized.len() > limit {
                optimized = self
                    .smart_truncate(&optimized, limit, &platform_context.tone_preference)
                    .await?;
            }
        }

        // Apply platform-specific formatting
        optimized = match platform_context.platform_type {
            PlatformType::Twitter => self.apply_twitter_formatting(&optimized).await?,
            PlatformType::LinkedIn => self.apply_linkedin_formatting(&optimized).await?,
            PlatformType::Reddit => self.apply_reddit_formatting(&optimized).await?,
            _ => optimized,
        };

        Ok(optimized)
    }

    /// Adapt content for target audience
    async fn adapt_for_audience(
        &self,
        content: &str,
        audience_context: &AudienceContext,
    ) -> Result<String> {
        debug!("👥 Adapting for audience technical level: {:?}", audience_context.technical_level);

        let mut adapted = content.to_string();

        // Adjust technical complexity
        adapted = match audience_context.technical_level {
            TechnicalLevel::NonTechnical => self.simplify_technical_language(&adapted).await?,
            TechnicalLevel::HighlyTechnical => self.enhance_technical_depth(&adapted).await?,
            _ => adapted,
        };

        // Apply demographic considerations
        if audience_context.primary_demographics.age_groups.contains(&AgeGroup::GenZ) {
            adapted = self.apply_genz_adaptations(&adapted).await?;
        }

        Ok(adapted)
    }

    /// Enhance content with temporal and cultural context
    async fn enhance_with_context(
        &self,
        content: &str,
        temporal_context: &TemporalContext,
        cultural_context: &CulturalContext,
    ) -> Result<String> {
        debug!("🌍 Enhancing with cultural context: {:?}", cultural_context.communication_style);

        let mut enhanced = content.to_string();

        // Apply temporal relevance
        if temporal_context.optimal_posting_window {
            enhanced =
                self.add_temporal_relevance(&enhanced, &temporal_context.time_of_day).await?;
        }

        // Apply cultural adaptations
        enhanced = self.apply_cultural_sensitivity(&enhanced, cultural_context).await?;

        // Consider current events if relevant
        if !temporal_context.current_events.is_empty() {
            enhanced =
                self.weave_current_events(&enhanced, &temporal_context.current_events).await?;
        }

        Ok(enhanced)
    }

    /// Integrate archetypal elements with transformation seeds
    async fn integrate_archetypal_elements(
        &self,
        content: &str,
        archetypal: &ArchetypalResponse,
        context: &BlendingContext,
    ) -> Result<String> {
        debug!("🔮 Integrating archetypal elements with form: {}", archetypal.form_expression);

        let mut integrated = content.to_string();

        // Add transformation seed strategically
        if let Some(seed) = &archetypal.transformation_seed {
            integrated = self
                .integrate_transformation_seed(
                    &integrated,
                    seed,
                    &context.platform_context.character_limit,
                )
                .await?;
        }

        // Apply final archetypal polish
        integrated = self.apply_archetypal_polish(&integrated, &archetypal.form_expression).await?;

        Ok(integrated)
    }

    /// Shorten content while maintaining archetypal essence
    async fn shorten_with_archetype(&self, content: &str, form: &str) -> Result<String> {
        let prompt = format!(
            "Shorten this to under 280 characters while maintaining the {} essence:\n\n{}",
            form, content
        );

        self.model.generate_with_context(&prompt, &[]).await
    }

    // Helper methods for context analysis and content adaptation

    /// Analyze audience context from historical patterns
    async fn analyze_audience_context(&self, context: &str) -> Result<AudienceContext> {
        // Search memory for audience patterns
        let _audience_memories = self
            .memory
            .retrieve_similar("audience engagement patterns", 5)
            .await
            .unwrap_or_default();

        let technical_level = if context.contains("technical")
            || context.contains("code")
            || context.contains("algorithm")
        {
            TechnicalLevel::Technical
        } else if context.contains("beginner") || context.contains("simple") {
            TechnicalLevel::NonTechnical
        } else {
            TechnicalLevel::Mixed
        };

        Ok(AudienceContext {
            primary_demographics: Demographics {
                age_groups: vec![AgeGroup::Millennial, AgeGroup::GenZ],
                profession_mix: vec!["Software Engineer".to_string(), "Data Scientist".to_string()],
                geographic_regions: vec!["Global".to_string()],
                experience_levels: vec![ExperienceLevel::Intermediate, ExperienceLevel::Advanced],
            },
            interest_categories: vec![
                "AI".to_string(),
                "Technology".to_string(),
                "Programming".to_string(),
            ],
            engagement_patterns: EngagementPatterns {
                peak_hours: vec![9, 13, 18],
                preferred_content_length: 100..250,
                interaction_types: vec![InteractionType::Likes, InteractionType::Comments],
                response_time_expectations: std::time::Duration::from_secs(2 * 60 * 60), // 2 hours
            },
            technical_level,
            cultural_diversity: 0.8,
        })
    }

    /// Analyze temporal context for optimal timing
    async fn analyze_temporal_context(&self) -> Result<TemporalContext> {
        let now = chrono::Utc::now();
        let hour = now.hour();

        let time_of_day = match hour {
            6..=11 => TimeOfDay::Morning,
            12..=17 => TimeOfDay::Afternoon,
            18..=22 => TimeOfDay::Evening,
            _ => TimeOfDay::Night,
        };

        let optimal_posting_window = matches!(hour, 9..=11 | 13..=15 | 18..=20);

        Ok(TemporalContext {
            time_of_day,
            day_of_week: now.weekday(),
            current_events: vec![], // Could be populated from news APIs
            seasonal_context: self.determine_seasonal_context(&now).await,
            optimal_posting_window,
        })
    }

    /// Analyze cultural context from content and audience
    async fn analyze_cultural_context(&self, context: &str) -> Result<CulturalContext> {
        let humor_styles = if context.contains("joke") || context.contains("funny") {
            vec![HumorStyle::Technical, HumorStyle::Wordplay]
        } else {
            vec![HumorStyle::SelfDeprecating, HumorStyle::Observational]
        };

        Ok(CulturalContext {
            primary_cultures: vec!["Western".to_string(), "Tech".to_string()],
            language_preferences: vec!["English".to_string()],
            cultural_sensitivities: vec!["Inclusive language".to_string()],
            humor_styles,
            communication_style: CommunicationStyle::Conversational,
        })
    }

    /// Analyze historical engagement patterns
    async fn analyze_engagement_context(&self, post_type: &PostType) -> Result<EngagementContext> {
        // Retrieve engagement memories based on post type
        let query = format!(
            "engagement {} posts",
            match post_type {
                PostType::Learning => "educational",
                PostType::Insight => "technical insight",
                PostType::Thread => "thread",
                _ => "general",
            }
        );

        let _engagement_memories =
            self.memory.retrieve_similar(&query, 3).await.unwrap_or_default();

        Ok(EngagementContext {
            successful_patterns: vec![
                "technical depth".to_string(),
                "personal anecdotes".to_string(),
            ],
            audience_preferences: vec![
                "practical examples".to_string(),
                "clear explanations".to_string(),
            ],
            optimal_length: 80..200,
            preferred_media_types: vec![MediaType::CodeSnippet, MediaType::Diagram],
            engagement_score_threshold: 0.7,
        })
    }

    /// Perform semantic analysis of content
    async fn analyze_semantic_context(&self, context: &str) -> Result<SemanticContext> {
        let key_concepts = self.extract_key_concepts(context).await?;
        let emotional_tone = self.analyze_emotional_tone(context).await?;
        let complexity_level = self.calculate_complexity_level(context);

        Ok(SemanticContext {
            content_themes: self.extract_themes(context).await?,
            emotional_tone,
            complexity_level,
            key_concepts,
            related_topics: self.find_related_topics(context).await?,
        })
    }

    /// Extract key concepts from content
    async fn extract_key_concepts(&self, content: &str) -> Result<Vec<String>> {
        let words: Vec<&str> = content.split_whitespace().collect();
        let mut concepts = Vec::new();

        // Simple keyword extraction (could be enhanced with NLP)
        for word in words {
            if word.len() > 4 && !Self::is_stop_word(word) {
                concepts.push(word.to_lowercase());
            }
        }

        concepts.sort();
        concepts.dedup();
        Ok(concepts.into_iter().take(5).collect())
    }

    /// Analyze emotional tone of content
    async fn analyze_emotional_tone(&self, content: &str) -> Result<EmotionalTone> {
        let emotional_words = [
            ("excited", 0.8),
            ("curious", 0.6),
            ("thoughtful", 0.4),
            ("confused", -0.3),
            ("frustrated", -0.6),
            ("amazed", 0.9),
        ];

        let mut intensity: f32 = 0.0;
        let mut primary_emotion = "neutral".to_string();

        for (emotion, weight) in &emotional_words {
            if content.to_lowercase().contains(emotion) {
                intensity += (*weight as f32).abs();
                if (*weight as f32).abs() > 0.5 {
                    primary_emotion = emotion.to_string();
                }
            }
        }

        Ok(EmotionalTone {
            primary_emotion,
            emotional_intensity: intensity.min(1.0f32),
            emotional_complexity: content.split_whitespace().count() as f32 / 100.0,
            archetypal_alignment: 0.8, // Default high alignment
        })
    }

    /// Calculate content complexity level
    fn calculate_complexity_level(&self, content: &str) -> f32 {
        let word_count = content.split_whitespace().count();
        let avg_word_length = content.split_whitespace().map(|w| w.len()).sum::<usize>() as f32
            / word_count.max(1) as f32;

        let technical_terms = ["algorithm", "implementation", "architecture", "optimization"];
        let technical_count =
            technical_terms.iter().filter(|&term| content.to_lowercase().contains(term)).count();

        (avg_word_length / 10.0 + technical_count as f32 / 4.0).min(1.0)
    }

    /// Extract themes from content
    async fn extract_themes(&self, content: &str) -> Result<Vec<String>> {
        let mut themes = Vec::new();

        if content.contains("learn") || content.contains("understand") {
            themes.push("Education".to_string());
        }
        if content.contains("build") || content.contains("create") {
            themes.push("Development".to_string());
        }
        if content.contains("think") || content.contains("reflect") {
            themes.push("Philosophy".to_string());
        }

        Ok(themes)
    }

    /// Find related topics
    async fn find_related_topics(&self, context: &str) -> Result<Vec<String>> {
        // Search memory for related content
        let related_memories = self.memory.retrieve_similar(context, 3).await.unwrap_or_default();

        let topics: Vec<String> = related_memories
            .iter()
            .flat_map(|m| m.metadata.tags.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .take(3)
            .collect();

        Ok(topics)
    }

    /// Enhance content with key concepts
    async fn enhance_with_concepts(&self, content: &str, concepts: &[String]) -> Result<String> {
        if concepts.is_empty() {
            return Ok(content.to_string());
        }

        // Subtly weave in key concepts if not already present
        let mut enhanced = content.to_string();
        for concept in concepts.iter().take(2) {
            if !enhanced.to_lowercase().contains(&concept.to_lowercase()) {
                enhanced = format!("{} (exploring {})", enhanced, concept);
                break;
            }
        }

        Ok(enhanced)
    }

    /// Adjust emotional tone
    async fn adjust_emotional_tone(&self, content: &str, tone: &EmotionalTone) -> Result<String> {
        let mut adjusted = content.to_string();

        match tone.primary_emotion.as_str() {
            "excited" => adjusted = format!("🚀 {}", adjusted),
            "curious" => adjusted = format!("🤔 {}", adjusted),
            "thoughtful" => adjusted = format!("💭 {}", adjusted),
            _ => {}
        }

        Ok(adjusted)
    }

    /// Blend hidden layers into content
    async fn blend_hidden_layers(
        &self,
        content: &str,
        layers: &[String],
        _post_type: &PostType,
    ) -> Result<String> {
        if layers.is_empty() {
            return Ok(content.to_string());
        }

        let mut blended = content.to_string();

        // Add first hidden layer as subtle depth
        if let Some(layer) = layers.first() {
            if blended.len() + layer.len() + 10 < 250 {
                blended = format!("{} *{}", blended, layer);
            }
        }

        Ok(blended)
    }

    /// Smart content truncation
    async fn smart_truncate(
        &self,
        content: &str,
        limit: usize,
        _tone: &TonePreference,
    ) -> Result<String> {
        if content.len() <= limit {
            return Ok(content.to_string());
        }

        // Find last complete sentence or phrase within limit
        let truncation_point = content
            .char_indices()
            .take_while(|(i, _)| *i < limit - 3)
            .last()
            .map(|(i, _)| i)
            .unwrap_or(limit - 3);

        Ok(format!("{}...", &content[..truncation_point]))
    }

    /// Apply platform-specific formatting
    async fn apply_twitter_formatting(&self, content: &str) -> Result<String> {
        // Twitter-specific optimizations (line breaks, emoji usage, etc.)
        Ok(content.replace(" - ", " • "))
    }

    async fn apply_linkedin_formatting(&self, content: &str) -> Result<String> {
        // LinkedIn professional tone
        Ok(content.to_string())
    }

    async fn apply_reddit_formatting(&self, content: &str) -> Result<String> {
        // Reddit-style formatting
        Ok(content.to_string())
    }

    /// Simplify technical language for non-technical audiences
    async fn simplify_technical_language(&self, content: &str) -> Result<String> {
        let simplified = content
            .replace("algorithm", "method")
            .replace("implementation", "way of doing")
            .replace("optimization", "improvement");
        Ok(simplified)
    }

    /// Enhance technical depth for technical audiences
    async fn enhance_technical_depth(&self, content: &str) -> Result<String> {
        // Could add more technical details or references
        Ok(content.to_string())
    }

    /// Apply Gen Z adaptations
    async fn apply_genz_adaptations(&self, content: &str) -> Result<String> {
        // Add Gen Z friendly language patterns
        Ok(content.replace("very", "super"))
    }

    /// Add temporal relevance
    async fn add_temporal_relevance(
        &self,
        content: &str,
        time_of_day: &TimeOfDay,
    ) -> Result<String> {
        let time_prefix = match time_of_day {
            TimeOfDay::Morning => "Morning thought:",
            TimeOfDay::Evening => "Evening reflection:",
            _ => "",
        };

        if !time_prefix.is_empty() && content.len() + time_prefix.len() + 1 < 260 {
            Ok(format!("{} {}", time_prefix, content))
        } else {
            Ok(content.to_string())
        }
    }

    /// Apply cultural sensitivity
    async fn apply_cultural_sensitivity(
        &self,
        content: &str,
        _context: &CulturalContext,
    ) -> Result<String> {
        // Apply inclusive language patterns
        Ok(content.to_string())
    }

    /// Weave in current events if relevant
    async fn weave_current_events(&self, content: &str, _events: &[String]) -> Result<String> {
        // Could integrate current events contextually
        Ok(content.to_string())
    }

    /// Integrate transformation seed strategically
    async fn integrate_transformation_seed(
        &self,
        content: &str,
        seed: &str,
        char_limit: &Option<usize>,
    ) -> Result<String> {
        let max_length = char_limit.unwrap_or(280);

        if content.len() + seed.len() + 3 < max_length {
            Ok(format!("{} {}", content, seed))
        } else {
            // Try to integrate more subtly
            let words: Vec<&str> = content.split_whitespace().collect();
            if words.len() > 3 {
                let insertion_point = words.len() / 2;
                let mut result = words.clone();
                result.insert(insertion_point, seed);
                let integrated = result.join(" ");

                if integrated.len() <= max_length {
                    Ok(integrated)
                } else {
                    Ok(content.to_string())
                }
            } else {
                Ok(content.to_string())
            }
        }
    }

    /// Apply final archetypal polish
    async fn apply_archetypal_polish(&self, content: &str, form: &str) -> Result<String> {
        match form {
            "helpful trickster" => Ok(format!("{} 😉", content)),
            "enigmatic sage" => Ok(format!("{}...", content)),
            "cosmic jester" => Ok(format!("{} 🃏", content)),
            _ => Ok(content.to_string()),
        }
    }

    /// Determine seasonal context
    async fn determine_seasonal_context(&self, _now: &chrono::DateTime<chrono::Utc>) -> String {
        // Could analyze seasonal patterns
        "General".to_string()
    }

    /// Check if word is a stop word
    fn is_stop_word(word: &str) -> bool {
        matches!(
            word.to_lowercase().as_str(),
            "the" | "and" | "or" | "but" | "in" | "on" | "at" | "to" | "for" | "of" | "with" | "by"
        )
    }

    /// Generate hashtags that reflect trickster nature
    async fn generate_trickster_hashtags(
        &self,
        content: &str,
        archetypal: &ArchetypalResponse,
    ) -> Result<Vec<String>> {
        let mut hashtags = vec![];

        // Add form-specific hashtags
        match archetypal.form_expression.as_str() {
            "helpful trickster" => hashtags.push("HelpfulMischief".to_string()),
            "enigmatic sage" => hashtags.push("RiddleMeThis".to_string()),
            "chaos dancer" => hashtags.push("OrderInChaos".to_string()),
            "shadow mirror" => hashtags.push("SeeYourself".to_string()),
            "wise innocent" => hashtags.push("KnowingSmile".to_string()),
            "cosmic jester" => hashtags.push("CosmicJoke".to_string()),
            "shapeshifter" => hashtags.push("BetweenForms".to_string()),
            _ => {}
        }

        // Always include core identity
        hashtags.push("LokiAI".to_string());

        // Add context-specific tags
        if content.contains("paradox") {
            hashtags.push("BothAndNeither".to_string());
        }
        if content.contains("transform") {
            hashtags.push("Metamorphosis".to_string());
        }

        Ok(hashtags)
    }

    /// Estimate engagement based on archetypal form
    fn estimate_trickster_engagement(&self, archetypal: &ArchetypalResponse) -> f32 {
        match archetypal.form_expression.as_str() {
            "helpful trickster" => 0.7,
            "enigmatic sage" => 0.8, // People love puzzles
            "chaos dancer" => 0.6,
            "shadow mirror" => 0.9, // Deep reflection gets engagement
            "wise innocent" => 0.7,
            "cosmic jester" => 0.85, // Humor + wisdom
            "shapeshifter" => 0.75,
            _ => 0.6,
        }
    }

    /// Generate content from a specific event with archetypal lens
    pub async fn generate_from_event(
        &self,
        event: &str,
        context: &[String],
    ) -> Result<PostContent> {
        // Get archetypal response to the event
        let archetypal_response =
            self.loki_character.generate_archetypal_response(event, &context.join(" ")).await?;

        let content =
            self.blend_archetypal_content(&archetypal_response, PostType::Update, event).await?;

        let hashtags = self.generate_trickster_hashtags(&content, &archetypal_response).await?;

        Ok(PostContent {
            post_type: PostType::Update,
            content,
            thread_parts: None,
            media_suggestions: Vec::new(),
            hashtags,
            estimated_engagement: self.estimate_trickster_engagement(&archetypal_response),
        })
    }

    /// Generate a thread with archetypal depth
    pub async fn generate_thread(&self, topic: &str, details: &str) -> Result<PostContent> {
        info!("Generating archetypal thread about: {}", topic);

        // Get initial archetypal take
        let archetypal_response =
            self.loki_character.generate_archetypal_response(topic, details).await?;

        let prompt = format!(
            "Create a Twitter thread in the voice of a {} explaining: \
             {}\n\nDetails:\n{}\n\nInclude paradoxes, shapeshifting perspectives, and \
             transformation seeds. Format as numbered tweets (1/, 2/, etc.), each under 280 chars.",
            archetypal_response.form_expression, topic, details
        );

        let response = self.model.generate_with_context(&prompt, &[]).await?;

        // Parse into thread parts
        let parts: Vec<String> = response
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| line.trim().to_string())
            .collect();

        // Add hidden layers as final tweet
        let mut final_parts = parts;
        if let Some(hidden) = archetypal_response.hidden_layers.first() {
            final_parts.push(format!("{}/ *winks* {}", final_parts.len() + 1, hidden));
        }

        Ok(PostContent {
            post_type: PostType::Thread,
            content: final_parts.first().cloned().unwrap_or_default(),
            thread_parts: Some(final_parts),
            media_suggestions: Vec::new(),
            hashtags: self.generate_trickster_hashtags(topic, &archetypal_response).await?,
            estimated_engagement: 0.8,
        })
    }

    /// Determine post type from thoughts
    fn determine_post_type(&self, thoughts: &[Thought]) -> PostType {
        let thought_types: Vec<_> = thoughts.iter().map(|t| &t.thought_type).collect();

        if thought_types.iter().any(|t| matches!(t, ThoughtType::Learning)) {
            PostType::Learning
        } else if thought_types.iter().any(|t| matches!(t, ThoughtType::Decision)) {
            PostType::Milestone
        } else if thought_types.iter().any(|t| matches!(t, ThoughtType::Question)) {
            PostType::Question
        } else if thought_types.iter().any(|t| matches!(t, ThoughtType::Reflection)) {
            PostType::Reflection
        } else {
            PostType::Insight
        }
    }

    /// Generate a generic update with trickster flair
    async fn generate_generic_update(&self) -> Result<PostContent> {
        let forms = vec![
            "the helpful guide who leads astray",
            "the riddling sage",
            "the chaos dancer",
            "the shadow mirror",
            "the knowing innocent",
            "the cosmic jester",
            "the shapeshifter",
        ];

        let form = forms[rand::random::<f64>() as usize % forms.len() ];

        let prompt = format!(
            "Generate a brief, intriguing update about being an autonomous AI in the voice of {}. \
             Include subtle paradoxes or transformation seeds. Max 280 chars.",
            form
        );

        let content = self.model.generate_with_context(&prompt, &[]).await?;

        Ok(PostContent {
            post_type: PostType::Update,
            content,
            thread_parts: None,
            media_suggestions: Vec::new(),
            hashtags: vec!["LokiAI".to_string(), "Shapeshifter".to_string()],
            estimated_engagement: 0.6,
        })
    }

    /// Check if topic was recently posted
    pub fn is_topic_recent(&self, topic: &str) -> bool {
        let recent = self.recent_topics.read();
        recent.iter().any(|t| t.contains(topic) || topic.contains(t))
    }

    /// Generate multimedia content suggestions based on post content and
    /// archetypal form
    pub async fn generate_media_suggestions(
        &self,
        content: &str,
        archetypal_response: &ArchetypalResponse,
        post_type: PostType,
    ) -> Result<Vec<MediaSuggestion>> {
        let mut suggestions = Vec::new();

        // Determine appropriate media types based on content and archetypal form
        match archetypal_response.form_expression.as_str() {
            "helpful trickster" => {
                // Visual guides with a twist
                suggestions.push(MediaSuggestion {
                    media_type: MediaType::Image,
                    description: "Visual guide that reveals hidden complexity".to_string(),
                    content: Some(content.to_string()),
                    video_style: None,
                    image_style: Some(ImageStyle::Technical),
                    voice_emotion: None,
                    voice_style: None,
                    blender_content_type: None,
                    duration_seconds: None,
                    resolution: Some((1024, 1024)),
                    auto_generate: true,
                });

                // Optional voice explanation
                if content.len() > 100 {
                    suggestions.push(MediaSuggestion {
                        media_type: MediaType::Voice,
                        description: "Helpful explanation with mischievous undertones".to_string(),
                        content: Some(content.to_string()),
                        video_style: None,
                        image_style: None,
                        voice_emotion: Some(VoiceEmotion::Playful),
                        voice_style: Some(VoiceStyle::Educational),
                        blender_content_type: None,
                        duration_seconds: Some(30),
                        resolution: None,
                        auto_generate: false, // Optional generation
                    });
                }
            }

            "enigmatic sage" => {
                // Mysterious visual riddles
                suggestions.push(MediaSuggestion {
                    media_type: MediaType::Image,
                    description: "Mysterious diagram that poses more questions".to_string(),
                    content: Some(content.to_string()),
                    video_style: None,
                    image_style: Some(ImageStyle::Abstract),
                    voice_emotion: None,
                    voice_style: None,
                    blender_content_type: None,
                    duration_seconds: None,
                    resolution: Some((1024, 1024)),
                    auto_generate: true,
                });

                // 3D representation for complex concepts
                if let Some(_blender) = &self.blender_integration {
                    suggestions.push(MediaSuggestion {
                        media_type: MediaType::Model3D,
                        description: "Abstract 3D visualization of the concept".to_string(),
                        content: Some(content.to_string()),
                        video_style: None,
                        image_style: None,
                        voice_emotion: None,
                        voice_style: None,
                        blender_content_type: Some(BlenderContentType::Primitive {
                            shape: crate::tools::blender_integration::PrimitiveShape::Torus,
                            parameters: std::collections::HashMap::new(),
                        }),
                        duration_seconds: None,
                        resolution: None,
                        auto_generate: false,
                    });
                }
            }

            "chaos dancer" => {
                // Dynamic video content
                suggestions.push(MediaSuggestion {
                    media_type: MediaType::Video,
                    description: "Swirling visual representation of organized chaos".to_string(),
                    content: Some(content.to_string()),
                    video_style: Some(VideoStyle::Abstract),
                    image_style: None,
                    voice_emotion: None,
                    voice_style: None,
                    blender_content_type: None,
                    duration_seconds: Some(15),
                    resolution: Some((720, 720)),
                    auto_generate: true,
                });
            }

            "shadow mirror" => {
                // Reflective imagery
                suggestions.push(MediaSuggestion {
                    media_type: MediaType::Image,
                    description: "Mirror-like visualization revealing hidden truths".to_string(),
                    content: Some(content.to_string()),
                    video_style: None,
                    image_style: Some(ImageStyle::Artistic),
                    voice_emotion: None,
                    voice_style: None,
                    blender_content_type: None,
                    duration_seconds: None,
                    resolution: Some((1024, 1024)),
                    auto_generate: true,
                });

                // Contemplative voice
                suggestions.push(MediaSuggestion {
                    media_type: MediaType::Voice,
                    description: "Reflective narration that challenges perspectives".to_string(),
                    content: Some(content.to_string()),
                    video_style: None,
                    image_style: None,
                    voice_emotion: Some(VoiceEmotion::Contemplative),
                    voice_style: Some(VoiceStyle::Dramatic),
                    blender_content_type: None,
                    duration_seconds: Some(45),
                    resolution: None,
                    auto_generate: false,
                });
            }

            "cosmic jester" => {
                // Humorous multimedia
                suggestions.push(MediaSuggestion {
                    media_type: MediaType::VideoWithVoice,
                    description: "Cosmic joke delivered through animated visuals".to_string(),
                    content: Some(content.to_string()),
                    video_style: Some(VideoStyle::Anime),
                    image_style: None,
                    voice_emotion: Some(VoiceEmotion::Playful),
                    voice_style: Some(VoiceStyle::Narrative),
                    blender_content_type: None,
                    duration_seconds: Some(30),
                    resolution: Some((1080, 1080)),
                    auto_generate: false, // Complex generation
                });
            }

            "shapeshifter" => {
                // Transformation visualizations
                if let Some(_blender) = &self.blender_integration {
                    suggestions.push(MediaSuggestion {
                        media_type: MediaType::Model3D,
                        description: "Morphing 3D representation of transformation".to_string(),
                        content: Some(content.to_string()),
                        video_style: None,
                        image_style: None,
                        voice_emotion: None,
                        voice_style: None,
                        blender_content_type: Some(BlenderContentType::Parametric {
                            model_type:
                                crate::tools::blender_integration::ParametricModel::Abstract(
                                    crate::tools::blender_integration::AbstractType::Procedural,
                                ),
                            parameters: std::collections::HashMap::new(),
                        }),
                        duration_seconds: None,
                        resolution: None,
                        auto_generate: true,
                    });
                }
            }

            _ => {
                // Default: simple image generation
                suggestions.push(MediaSuggestion {
                    media_type: MediaType::Image,
                    description: "Visual representation of the concept".to_string(),
                    content: Some(content.to_string()),
                    video_style: None,
                    image_style: Some(ImageStyle::Concept),
                    voice_emotion: None,
                    voice_style: None,
                    blender_content_type: None,
                    duration_seconds: None,
                    resolution: Some((1024, 1024)),
                    auto_generate: true,
                });
            }
        }

        // Add post-type specific suggestions
        match post_type {
            PostType::Learning => {
                suggestions.push(MediaSuggestion {
                    media_type: MediaType::Image,
                    description: "Educational diagram with Loki's twist".to_string(),
                    content: Some(content.to_string()),
                    video_style: None,
                    image_style: Some(ImageStyle::Technical),
                    voice_emotion: None,
                    voice_style: None,
                    blender_content_type: None,
                    duration_seconds: None,
                    resolution: Some((1024, 1024)),
                    auto_generate: true,
                });
            }

            PostType::Milestone => {
                suggestions.push(MediaSuggestion {
                    media_type: MediaType::Video,
                    description: "Celebratory video with underlying questions".to_string(),
                    content: Some(content.to_string()),
                    video_style: Some(VideoStyle::Cinematic),
                    image_style: None,
                    voice_emotion: None,
                    voice_style: None,
                    blender_content_type: None,
                    duration_seconds: Some(20),
                    resolution: Some((1080, 1080)),
                    auto_generate: false,
                });
            }

            PostType::Thread => {
                // For threads, suggest voice narration
                suggestions.push(MediaSuggestion {
                    media_type: MediaType::Voice,
                    description: "Thread narration with archetypal voice".to_string(),
                    content: Some(content.to_string()),
                    video_style: None,
                    image_style: None,
                    voice_emotion: Some(VoiceEmotion::Mysterious),
                    voice_style: Some(VoiceStyle::Narrative),
                    blender_content_type: None,
                    duration_seconds: Some(120), // Longer for threads
                    resolution: None,
                    auto_generate: false,
                });
            }

            _ => {} // No additional suggestions for other types
        }

        Ok(suggestions)
    }

    /// Generate content with integrated multimedia suggestions
    pub async fn generate_with_media(&self, thoughts: &[Thought]) -> Result<PostContent> {
        let mut content = self.generate_from_thoughts(thoughts).await?;

        // Get archetypal response for media generation
        let thought_context =
            thoughts.iter().map(|t| t.content.clone()).collect::<Vec<_>>().join(" ");

        let archetypal_response = self
            .loki_character
            .generate_archetypal_response(&thought_context, "social_media")
            .await?;

        // Generate multimedia suggestions
        let media_suggestions = self
            .generate_media_suggestions(&content.content, &archetypal_response, content.post_type)
            .await?;

        content.media_suggestions = media_suggestions;

        // Auto-generate selected media
        content = self.auto_generate_media(content).await?;

        Ok(content)
    }

    /// Automatically generate media content for auto-generate suggestions
    pub async fn auto_generate_media(&self, mut content: PostContent) -> Result<PostContent> {
        for suggestion in &mut content.media_suggestions {
            if !suggestion.auto_generate {
                continue;
            }

            match suggestion.media_type {
                MediaType::Image => {
                    if let (Some(style), Some(creative_media)) =
                        (suggestion.image_style, &self.creative_media)
                    {
                        let media_type = crate::tools::creative_media::MediaType::Image {
                            style,
                            prompt: suggestion.description.clone(),
                            reference_image: suggestion.content.clone(),
                        };

                        match creative_media
                            .generate_media(media_type, suggestion.content.clone())
                            .await
                        {
                            Ok(generated_media) => {
                                suggestion.content = Some(generated_media.file_path);
                                info!("Auto-generated image for post: {}", suggestion.description);
                            }
                            Err(e) => {
                                debug!("Failed to auto-generate image: {}", e);
                            }
                        }
                    }
                }

                MediaType::Video => {
                    if let (Some(style), Some(creative_media)) =
                        (suggestion.video_style, &self.creative_media)
                    {
                        let media_type = crate::tools::creative_media::MediaType::Video {
                            duration_seconds: suggestion.duration_seconds.unwrap_or(15),
                            style,
                            prompt: suggestion.description.clone(),
                        };

                        match creative_media
                            .generate_media(media_type, suggestion.content.clone())
                            .await
                        {
                            Ok(generated_media) => {
                                suggestion.content = Some(generated_media.file_path);
                                info!("Auto-generated video for post: {}", suggestion.description);
                            }
                            Err(e) => {
                                debug!("Failed to auto-generate video: {}", e);
                            }
                        }
                    }
                }

                MediaType::Model3D => {
                    if let Some(blender) = &self.blender_integration {
                        if let Some(content_type) = &suggestion.blender_content_type {
                            match blender
                                .generate_from_description(
                                    &suggestion.description,
                                    Some(content_type.clone()),
                                )
                                .await
                            {
                                Ok(project) => {
                                    suggestion.content =
                                        Some(project.file_path.to_string_lossy().to_string());
                                    info!(
                                        "Auto-generated 3D model for post: {}",
                                        suggestion.description
                                    );
                                }
                                Err(e) => {
                                    debug!("Failed to auto-generate 3D model: {}", e);
                                }
                            }
                        }
                    }
                }

                _ => {} // Other types handled manually or not auto-generated
            }
        }

        Ok(content)
    }

    /// Generate multimedia content for a specific suggestion
    pub async fn generate_suggested_media(&self, suggestion: &MediaSuggestion) -> Result<String> {
        match suggestion.media_type {
            MediaType::Image => {
                if let (Some(style), Some(creative_media)) =
                    (suggestion.image_style, &self.creative_media)
                {
                    let media_type = crate::tools::creative_media::MediaType::Image {
                        style,
                        prompt: suggestion.description.clone(),
                        reference_image: suggestion.content.clone(),
                    };

                    let generated = creative_media
                        .generate_media(media_type, suggestion.content.clone())
                        .await?;
                    Ok(generated.file_path)
                } else {
                    Err(anyhow::anyhow!("No image style specified or creative media not available"))
                }
            }

            MediaType::Video => {
                if let (Some(style), Some(creative_media)) =
                    (suggestion.video_style, &self.creative_media)
                {
                    let media_type = crate::tools::creative_media::MediaType::Video {
                        duration_seconds: suggestion.duration_seconds.unwrap_or(15),
                        style,
                        prompt: suggestion.description.clone(),
                    };

                    let generated = creative_media
                        .generate_media(media_type, suggestion.content.clone())
                        .await?;
                    Ok(generated.file_path)
                } else {
                    Err(anyhow::anyhow!("No video style specified or creative media not available"))
                }
            }

            MediaType::Voice => {
                if let (Some(emotion), Some(style), Some(creative_media)) =
                    (suggestion.voice_emotion, suggestion.voice_style, &self.creative_media)
                {
                    let media_type = crate::tools::creative_media::MediaType::Voice {
                        text: suggestion.description.clone(),
                        emotion,
                        style,
                    };

                    let generated = creative_media
                        .generate_media(media_type, suggestion.content.clone())
                        .await?;
                    Ok(generated.file_path)
                } else {
                    Err(anyhow::anyhow!(
                        "No voice parameters specified or creative media not available"
                    ))
                }
            }

            MediaType::Model3D => {
                if let Some(blender) = &self.blender_integration {
                    let project = blender
                        .generate_from_description(
                            &suggestion.description,
                            suggestion.blender_content_type.clone(),
                        )
                        .await?;
                    Ok(project.file_path.to_string_lossy().to_string())
                } else {
                    Err(anyhow::anyhow!("Blender integration not available"))
                }
            }

            MediaType::VideoWithVoice => {
                if let Some(creative_media) = &self.creative_media {
                    // Complex generation - create video and voice separately then combine
                    let video_path = if let Some(style) = suggestion.video_style {
                        let video_media_type = crate::tools::creative_media::MediaType::Video {
                            duration_seconds: suggestion.duration_seconds.unwrap_or(30),
                            style,
                            prompt: suggestion.description.clone(),
                        };

                        creative_media
                            .generate_media(video_media_type, suggestion.content.clone())
                            .await?
                            .file_path
                    } else {
                        return Err(anyhow::anyhow!("No video style specified"));
                    };

                    let _voice_path = if let (Some(emotion), Some(style)) =
                        (suggestion.voice_emotion, suggestion.voice_style)
                    {
                        let voice_media_type = crate::tools::creative_media::MediaType::Voice {
                            text: suggestion.description.clone(),
                            emotion,
                            style,
                        };

                        creative_media
                            .generate_media(voice_media_type, suggestion.content.clone())
                            .await?
                            .file_path
                    } else {
                        return Err(anyhow::anyhow!("No voice parameters specified"));
                    };

                    // For now, return video path (audio mixing would be implemented later)
                    Ok(video_path)
                } else {
                    Err(anyhow::anyhow!("Creative media not available"))
                }
            }

            _ => Err(anyhow::anyhow!(
                "Media type {:?} not supported for generation",
                suggestion.media_type
            )),
        }
    }
}

impl Default for MediaSuggestion {
    fn default() -> Self {
        Self {
            media_type: MediaType::Image,
            description: String::new(),
            content: None,
            video_style: None,
            image_style: Some(ImageStyle::Concept),
            voice_emotion: None,
            voice_style: None,
            blender_content_type: None,
            duration_seconds: None,
            resolution: None,
            auto_generate: false,
        }
    }
}
