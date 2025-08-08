use std::collections::HashMap;

use anyhow::Result;
use tracing::{debug, info};

use super::creative_intelligence::{ArtisticCreation, ContentType, CreativeMode, CreativePrompt};

/// Artistic creation engine for generating creative content across multiple
/// media
#[derive(Debug)]
pub struct ArtisticCreationEngine {
    /// Artistic styles database
    artistic_styles: HashMap<String, ArtisticStyle>,

    /// Creative generators for different media
    generators: HashMap<MediaType, CreativeGenerator>,

    /// Aesthetic evaluation criteria
    aesthetic_criteria: Vec<AestheticCriterion>,
}

/// Artistic style definition
#[derive(Debug, Clone)]
pub struct ArtisticStyle {
    /// Style identifier
    pub id: String,

    /// Style name
    pub name: String,

    /// Style characteristics
    pub characteristics: Vec<String>,

    /// Compatible media types
    pub compatible_media: Vec<MediaType>,

    /// Style complexity
    pub complexity: f64,
}

/// Types of creative media
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MediaType {
    Text,
    Poetry,
    Music,
    VisualArt,
    Design,
    Story,
    Concept,
}

/// Creative generator for specific media type
#[derive(Debug, Clone)]
pub struct CreativeGenerator {
    /// Generator identifier
    pub id: String,

    /// Media type it handles
    pub media_type: MediaType,

    /// Generation techniques
    pub techniques: Vec<String>,

    /// Quality threshold
    pub quality_threshold: f64,
}

/// Aesthetic evaluation criterion
#[derive(Debug, Clone)]
pub struct AestheticCriterion {
    /// Criterion name
    pub name: String,

    /// Applicable media types
    pub media_types: Vec<MediaType>,

    /// Weight in evaluation
    pub weight: f64,
}

/// Result of artistic creation
#[derive(Debug, Clone)]
pub struct ArtisticCreationResult {
    /// Created artistic works
    pub creations: Vec<ArtisticCreation>,

    /// Creation metadata
    pub metadata: CreationMetadata,
}

/// Metadata about artistic creation process
#[derive(Debug, Clone)]
pub struct CreationMetadata {
    /// Styles used
    pub styles_used: Vec<String>,

    /// Media types created
    pub media_types: Vec<MediaType>,

    /// Overall aesthetic score
    pub aesthetic_score: f64,

    /// Creation insights
    pub insights: Vec<String>,
}

impl ArtisticCreationEngine {
    /// Create new artistic creation engine
    pub async fn new() -> Result<Self> {
        info!("🎨 Initializing Artistic Creation Engine");

        let mut engine = Self {
            artistic_styles: HashMap::new(),
            generators: HashMap::new(),
            aesthetic_criteria: Vec::new(),
        };

        // Initialize artistic styles
        engine.initialize_artistic_styles().await?;

        // Initialize creative generators
        engine.initialize_creative_generators().await?;

        // Initialize aesthetic criteria
        engine.initialize_aesthetic_criteria().await?;

        info!("✅ Artistic Creation Engine initialized");
        Ok(engine)
    }

    /// Create artistic content based on prompt
    pub async fn create_artistic_content(
        &self,
        prompt: &CreativePrompt,
    ) -> Result<ArtisticCreationResult> {
        debug!("🎨 Creating artistic content for: {}", prompt.description);

        let mut creations = Vec::new();
        let mut styles_used = Vec::new();
        let mut media_types = Vec::new();

        // Determine appropriate media types
        let target_media = self.determine_target_media(prompt).await?;

        // Create content for each media type
        for media_type in &target_media {
            if let Ok(creation) = self.create_media_content(media_type, prompt).await {
                creations.push(creation);
                media_types.push(media_type.clone());
            }
        }

        // Select and apply artistic styles
        let selected_styles = self.select_artistic_styles(prompt, &target_media).await?;
        for style in &selected_styles {
            styles_used.push(style.name.clone());
        }

        // Evaluate overall aesthetic quality
        let aesthetic_score = self.evaluate_aesthetic_quality(&creations).await?;

        // Generate creation insights
        let insights = self.generate_creation_insights(&creations, &selected_styles).await?;

        let metadata = CreationMetadata { styles_used, media_types, aesthetic_score, insights };

        let result = ArtisticCreationResult { creations, metadata };

        debug!(
            "✅ Created {} artistic works with {:.2} aesthetic score",
            result.creations.len(),
            result.metadata.aesthetic_score
        );

        Ok(result)
    }

    /// Initialize artistic styles
    async fn initialize_artistic_styles(&mut self) -> Result<()> {
        let styles = vec![
            ArtisticStyle {
                id: "minimalist".to_string(),
                name: "Minimalist".to_string(),
                characteristics: vec![
                    "simple".to_string(),
                    "clean".to_string(),
                    "focused".to_string(),
                ],
                compatible_media: vec![MediaType::VisualArt, MediaType::Design, MediaType::Text],
                complexity: 0.3,
            },
            ArtisticStyle {
                id: "expressive".to_string(),
                name: "Expressive".to_string(),
                characteristics: vec![
                    "emotional".to_string(),
                    "dynamic".to_string(),
                    "vivid".to_string(),
                ],
                compatible_media: vec![MediaType::Poetry, MediaType::Music, MediaType::VisualArt],
                complexity: 0.8,
            },
            ArtisticStyle {
                id: "surreal".to_string(),
                name: "Surreal".to_string(),
                characteristics: vec![
                    "dreamlike".to_string(),
                    "unexpected".to_string(),
                    "imaginative".to_string(),
                ],
                compatible_media: vec![MediaType::Story, MediaType::VisualArt, MediaType::Concept],
                complexity: 0.9,
            },
        ];

        for style in styles {
            self.artistic_styles.insert(style.id.clone(), style);
        }

        debug!("🎨 Initialized {} artistic styles", self.artistic_styles.len());
        Ok(())
    }

    /// Initialize creative generators
    async fn initialize_creative_generators(&mut self) -> Result<()> {
        let generators = vec![
            CreativeGenerator {
                id: "text_generator".to_string(),
                media_type: MediaType::Text,
                techniques: vec!["narrative_flow".to_string(), "descriptive_language".to_string()],
                quality_threshold: 0.7,
            },
            CreativeGenerator {
                id: "poetry_generator".to_string(),
                media_type: MediaType::Poetry,
                techniques: vec!["rhyme_scheme".to_string(), "metaphor_creation".to_string()],
                quality_threshold: 0.8,
            },
            CreativeGenerator {
                id: "visual_generator".to_string(),
                media_type: MediaType::VisualArt,
                techniques: vec!["composition".to_string(), "color_harmony".to_string()],
                quality_threshold: 0.75,
            },
        ];

        for generator in generators {
            self.generators.insert(generator.media_type.clone(), generator);
        }

        debug!("🔧 Initialized {} creative generators", self.generators.len());
        Ok(())
    }

    /// Initialize aesthetic criteria
    async fn initialize_aesthetic_criteria(&mut self) -> Result<()> {
        self.aesthetic_criteria = vec![
            AestheticCriterion {
                name: "Visual Appeal".to_string(),
                media_types: vec![MediaType::VisualArt, MediaType::Design],
                weight: 0.4,
            },
            AestheticCriterion {
                name: "Emotional Impact".to_string(),
                media_types: vec![MediaType::Poetry, MediaType::Music, MediaType::Story],
                weight: 0.3,
            },
            AestheticCriterion {
                name: "Originality".to_string(),
                media_types: vec![MediaType::Text, MediaType::Concept, MediaType::VisualArt],
                weight: 0.3,
            },
        ];

        debug!("📊 Initialized {} aesthetic criteria", self.aesthetic_criteria.len());
        Ok(())
    }

    /// Determine target media types for creation
    async fn determine_target_media(&self, prompt: &CreativePrompt) -> Result<Vec<MediaType>> {
        let mut media_types = Vec::new();

        // Analyze prompt for media type hints
        let description = prompt.description.to_lowercase();

        if description.contains("visual")
            || description.contains("art")
            || description.contains("image")
        {
            media_types.push(MediaType::VisualArt);
        }

        if description.contains("story") || description.contains("narrative") {
            media_types.push(MediaType::Story);
        }

        if description.contains("poem")
            || description.contains("poetry")
            || description.contains("verse")
        {
            media_types.push(MediaType::Poetry);
        }

        if description.contains("design") {
            media_types.push(MediaType::Design);
        }

        // Default to text and concept if no specific media detected
        if media_types.is_empty() {
            media_types.push(MediaType::Text);
            media_types.push(MediaType::Concept);
        }

        debug!("🎯 Determined target media: {:?}", media_types);
        Ok(media_types)
    }

    /// Create content for specific media type
    async fn create_media_content(
        &self,
        media_type: &MediaType,
        prompt: &CreativePrompt,
    ) -> Result<ArtisticCreation> {
        let content = match media_type {
            MediaType::Text => self.create_text_content(prompt).await?,
            MediaType::Poetry => self.create_poetry_content(prompt).await?,
            MediaType::VisualArt => self.create_visual_art_content(prompt).await?,
            MediaType::Story => self.create_story_content(prompt).await?,
            MediaType::Design => self.create_design_content(prompt).await?,
            MediaType::Concept => self.create_concept_content(prompt).await?,
            _ => format!(
                "Creative {} content based on: {}",
                format!("{:?}", media_type),
                prompt.description
            ),
        };

        let creation = ArtisticCreation {
            id: format!("art_{}", uuid::Uuid::new_v4()),
            content_type: match media_type {
                MediaType::VisualArt => ContentType::VisualArt,
                MediaType::Poetry => ContentType::Poetry,
                MediaType::Story => ContentType::Story,
                MediaType::Design => ContentType::Design,
                _ => ContentType::Concept,
            },
            content,
            aesthetic_score: self.calculate_aesthetic_score(media_type).await?,
        };

        Ok(creation)
    }

    /// Create text content
    async fn create_text_content(&self, prompt: &CreativePrompt) -> Result<String> {
        let content = format!(
            "Creative textual interpretation of: {}\n\nThis innovative approach explores {} \
             through a narrative lens, offering fresh perspectives and engaging insights that \
             challenge conventional thinking.",
            prompt.description, prompt.domain
        );
        Ok(content)
    }

    /// Create poetry content
    async fn create_poetry_content(&self, prompt: &CreativePrompt) -> Result<String> {
        let content = format!(
            "In realms of {}, where dreams take flight,\nCreative visions burn so bright.\n{} \
             unfolds in verses new,\nInspiration's gift, both bold and true.\n\nThrough metaphor \
             and rhythm's dance,\nWe find in art our souls' advance.",
            prompt.domain, prompt.description
        );
        Ok(content)
    }

    /// Create visual art content
    async fn create_visual_art_content(&self, prompt: &CreativePrompt) -> Result<String> {
        let content = format!(
            "Visual Art Concept: '{}'\n\nComposition: Dynamic interplay of forms representing \
             {}\nColor Palette: Harmonious blend reflecting the essence of {}\nStyle: \
             Contemporary interpretation with innovative techniques\nSymbolism: Deep metaphorical \
             layers exploring the concept",
            prompt.description, prompt.domain, prompt.description
        );
        Ok(content)
    }

    /// Create story content
    async fn create_story_content(&self, prompt: &CreativePrompt) -> Result<String> {
        let content = format!(
            "The Story of {}\n\nIn a world where {} was everything, our protagonist discovered \
             something extraordinary. Through trials and revelations, they learned that {} held \
             the key to transformation.\n\nThis tale weaves together imagination and insight, \
             creating a narrative that resonates with universal themes while exploring the unique \
             aspects of {}.",
            prompt.description, prompt.domain, prompt.description, prompt.domain
        );
        Ok(content)
    }

    /// Create design content
    async fn create_design_content(&self, prompt: &CreativePrompt) -> Result<String> {
        let content = format!(
            "Design Innovation: {}\n\nConcept: Revolutionary approach to {} design\nUser \
             Experience: Intuitive interaction optimized for {}\nAesthetic: Clean, functional \
             beauty with artistic flair\nInnovation: Breakthrough features that redefine \
             expectations",
            prompt.description, prompt.domain, prompt.domain
        );
        Ok(content)
    }

    /// Create concept content
    async fn create_concept_content(&self, prompt: &CreativePrompt) -> Result<String> {
        let content = format!(
            "Conceptual Framework: {}\n\nCore Idea: {} as a transformative force\nImplications: \
             Broad applications across multiple domains\nInnovation Potential: High impact \
             possibilities\nPhilosophical Depth: Fundamental questions about {}\n\nThis concept \
             challenges existing paradigms and opens new avenues for exploration.",
            prompt.description, prompt.description, prompt.domain
        );
        Ok(content)
    }

    /// Select appropriate artistic styles
    async fn select_artistic_styles(
        &self,
        prompt: &CreativePrompt,
        media_types: &[MediaType],
    ) -> Result<Vec<ArtisticStyle>> {
        let mut selected_styles = Vec::new();

        // Match styles to creative mode and media types
        let preferred_complexity =
            match prompt.preferred_mode.as_ref().unwrap_or(&CreativeMode::Exploratory) {
                CreativeMode::Focused => 0.7,
                CreativeMode::Experimental => 0.9,
                CreativeMode::Breakthrough => 0.95,
                _ => 0.6,
            };

        for style in self.artistic_styles.values() {
            let media_compatibility =
                media_types.iter().any(|media| style.compatible_media.contains(media));

            let complexity_match = (style.complexity - preferred_complexity).abs() < 0.3;

            if media_compatibility && complexity_match {
                selected_styles.push(style.clone());
            }
        }

        // Ensure at least one style is selected
        if selected_styles.is_empty() {
            if let Some(first_style) = self.artistic_styles.values().next() {
                selected_styles.push(first_style.clone());
            }
        }

        debug!("🎨 Selected {} artistic styles", selected_styles.len());
        Ok(selected_styles)
    }

    /// Calculate aesthetic score for media type
    async fn calculate_aesthetic_score(&self, media_type: &MediaType) -> Result<f64> {
        // Base scores for different media types
        let base_score: f64 = match media_type {
            MediaType::VisualArt => 0.8,
            MediaType::Poetry => 0.85,
            MediaType::Music => 0.9,
            MediaType::Story => 0.75,
            MediaType::Design => 0.8,
            _ => 0.7,
        };

        // Add some variation
        let variation: f64 = 0.1; // ±10% variation
        let score: f64 = base_score + (0.5_f64 - 0.5_f64) * variation; // Fixed variation for deterministic behavior

        Ok(score.min(1.0_f64).max(0.0_f64))
    }

    /// Evaluate overall aesthetic quality
    async fn evaluate_aesthetic_quality(&self, creations: &[ArtisticCreation]) -> Result<f64> {
        if creations.is_empty() {
            return Ok(0.0);
        }

        let total_score: f64 = creations.iter().map(|c| c.aesthetic_score).sum();
        let average_score = total_score / creations.len() as f64;

        // Bonus for diversity
        let unique_types: std::collections::HashSet<ContentType> =
            creations.iter().map(|c| c.content_type.clone()).collect();

        let diversity_bonus = (unique_types.len() as f64) * 0.05;

        Ok((average_score + diversity_bonus).min(1.0))
    }

    /// Generate insights about artistic creation
    async fn generate_creation_insights(
        &self,
        creations: &[ArtisticCreation],
        styles: &[ArtisticStyle],
    ) -> Result<Vec<String>> {
        let mut insights = Vec::new();

        insights.push(format!("Created {} artistic works across multiple media", creations.len()));

        if !styles.is_empty() {
            let style_names: Vec<String> = styles.iter().map(|s| s.name.clone()).collect();
            insights.push(format!("Applied artistic styles: {}", style_names.join(", ")));
        }

        // Find highest scoring creation
        if let Some(best_creation) = creations.iter().max_by(|a, b| {
            a.aesthetic_score.partial_cmp(&b.aesthetic_score).unwrap_or(std::cmp::Ordering::Equal)
        }) {
            insights.push(format!(
                "Highest aesthetic score: {:.1}% in {:?} medium",
                best_creation.aesthetic_score * 100.0,
                best_creation.content_type
            ));
        }

        // Assess overall quality
        let avg_score = if !creations.is_empty() {
            creations.iter().map(|c| c.aesthetic_score).sum::<f64>() / creations.len() as f64
        } else {
            0.0
        };

        let quality_assessment = match avg_score {
            s if s >= 0.9 => "Exceptional artistic quality",
            s if s >= 0.8 => "High artistic quality",
            s if s >= 0.7 => "Good artistic quality",
            s if s >= 0.6 => "Moderate artistic quality",
            _ => "Developing artistic quality",
        };

        insights.push(quality_assessment.to_string());

        Ok(insights)
    }
}
