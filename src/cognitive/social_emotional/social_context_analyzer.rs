use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use uuid;

/// Social context analyzer for Phase 5 social intelligence
#[derive(Debug)]
pub struct SocialContextAnalyzer {
    /// Social context data
    context_data: Arc<RwLock<HashMap<String, AnalyzerContextData>>>,
    /// Emotional lexicon for sentiment analysis
    emotional_lexicon: HashMap<String, EmotionalValue>,
    /// Social dynamics patterns
    social_patterns: Vec<SocialPattern>,
    /// Context history for temporal analysis
    context_history: Arc<RwLock<Vec<ContextSnapshot>>>,
}

/// Social context data for analyzer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyzerContextData {
    /// Context identifier
    pub id: String,

    /// Context description
    pub description: String,

    /// Context strength
    pub strength: f64,

    /// Last updated
    pub last_updated: DateTime<Utc>,
    
    /// Emotional valence (-1.0 to 1.0)
    pub emotional_valence: f64,
    
    /// Social dynamics type
    pub dynamics_type: SocialDynamicsType,
    
    /// Interaction participants
    pub participants: Vec<String>,
    
    /// Cultural context markers
    pub cultural_markers: Vec<String>,
}

/// Emotional value for lexicon analysis
#[derive(Debug, Clone)]
struct EmotionalValue {
    valence: f64,    // -1.0 (negative) to 1.0 (positive)
    arousal: f64,    // 0.0 (calm) to 1.0 (excited)
    dominance: f64,  // 0.0 (submissive) to 1.0 (dominant)
}

/// Social pattern recognition
#[derive(Debug, Clone)]
struct SocialPattern {
    pattern_type: SocialPatternType,
    keywords: Vec<String>,
    weight: f64,
    context_modifiers: Vec<String>,
}

#[derive(Debug, Clone)]
enum SocialPatternType {
    Collaboration,
    Conflict,
    Support,
    Authority,
    Negotiation,
    Celebration,
    Crisis,
    Learning,
}

/// Social dynamics classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SocialDynamicsType {
    Cooperative,
    Competitive,
    Hierarchical,
    Egalitarian,
    Supportive,
    Adversarial,
    Neutral,
    Mixed,
}

/// Snapshot of context at a point in time
#[derive(Debug, Clone)]
struct ContextSnapshot {
    timestamp: DateTime<Utc>,
    overall_score: f64,
    dominant_emotion: String,
    social_dynamics: SocialDynamicsType,
    participant_count: usize,
}

impl Default for SocialContextAnalyzer {
    fn default() -> Self {
        Self {
            context_data: Arc::new(RwLock::new(HashMap::new())),
            emotional_lexicon: Self::build_emotional_lexicon(),
            social_patterns: Self::build_social_patterns(),
            context_history: Arc::new(RwLock::new(Vec::new())),
        }
    }
}

impl SocialContextAnalyzer {
    /// Create new social context analyzer
    pub fn new() -> Self {
        Self::default()
    }

    /// Analyze social context with comprehensive processing
    pub async fn analyze_context(&self, input: &str) -> Result<f64> {
        // 1. Emotional analysis
        let emotional_score = self.analyze_emotional_content(input);
        
        // 2. Social pattern detection
        let pattern_score = self.detect_social_patterns(input);
        
        // 3. Participant analysis
        let participants = self.extract_participants(input);
        let participant_score = (participants.len() as f64 / 10.0).min(1.0);
        
        // 4. Cultural context analysis
        let cultural_score = self.analyze_cultural_context(input);
        
        // 5. Temporal dynamics (check history)
        let temporal_score = self.analyze_temporal_dynamics().await;
        
        // 6. Calculate composite score
        let composite_score = (emotional_score * 0.3 + 
                              pattern_score * 0.25 + 
                              participant_score * 0.15 + 
                              cultural_score * 0.15 + 
                              temporal_score * 0.15).clamp(0.0, 1.0);
        
        // 7. Update context data
        let context_entry = AnalyzerContextData {
            id: uuid::Uuid::new_v4().to_string(),
            description: self.generate_context_description(input, &participants),
            strength: composite_score,
            last_updated: Utc::now(),
            emotional_valence: emotional_score * 2.0 - 1.0, // Convert to -1.0 to 1.0
            dynamics_type: self.determine_dynamics_type(pattern_score, emotional_score),
            participants,
            cultural_markers: self.extract_cultural_markers(input),
        };
        
        // Store context
        {
            let mut context_data = self.context_data.write().await;
            context_data.insert(context_entry.id.clone(), context_entry.clone());
        }
        
        // 8. Add to history
        self.add_to_history(composite_score, emotional_score, context_entry.dynamics_type.clone()).await;
        
        Ok(composite_score)
    }
    
    /// Build emotional lexicon
    fn build_emotional_lexicon() -> HashMap<String, EmotionalValue> {
        let mut lexicon = HashMap::new();
        
        // Positive emotions
        lexicon.insert("happy".to_string(), EmotionalValue { valence: 0.8, arousal: 0.6, dominance: 0.5 });
        lexicon.insert("joy".to_string(), EmotionalValue { valence: 0.9, arousal: 0.7, dominance: 0.6 });
        lexicon.insert("excited".to_string(), EmotionalValue { valence: 0.7, arousal: 0.9, dominance: 0.7 });
        lexicon.insert("grateful".to_string(), EmotionalValue { valence: 0.8, arousal: 0.4, dominance: 0.3 });
        lexicon.insert("love".to_string(), EmotionalValue { valence: 0.9, arousal: 0.6, dominance: 0.5 });
        lexicon.insert("proud".to_string(), EmotionalValue { valence: 0.7, arousal: 0.5, dominance: 0.8 });
        
        // Negative emotions
        lexicon.insert("sad".to_string(), EmotionalValue { valence: -0.7, arousal: 0.3, dominance: 0.3 });
        lexicon.insert("angry".to_string(), EmotionalValue { valence: -0.8, arousal: 0.8, dominance: 0.7 });
        lexicon.insert("fear".to_string(), EmotionalValue { valence: -0.8, arousal: 0.7, dominance: 0.2 });
        lexicon.insert("frustrated".to_string(), EmotionalValue { valence: -0.6, arousal: 0.7, dominance: 0.5 });
        lexicon.insert("disappointed".to_string(), EmotionalValue { valence: -0.5, arousal: 0.4, dominance: 0.4 });
        
        // Social emotions
        lexicon.insert("empathy".to_string(), EmotionalValue { valence: 0.6, arousal: 0.4, dominance: 0.4 });
        lexicon.insert("trust".to_string(), EmotionalValue { valence: 0.7, arousal: 0.3, dominance: 0.5 });
        lexicon.insert("respect".to_string(), EmotionalValue { valence: 0.6, arousal: 0.3, dominance: 0.4 });
        
        lexicon
    }
    
    /// Build social patterns
    fn build_social_patterns() -> Vec<SocialPattern> {
        vec![
            SocialPattern {
                pattern_type: SocialPatternType::Collaboration,
                keywords: vec!["together".to_string(), "team".to_string(), "collaborate".to_string(), 
                              "partnership".to_string(), "cooperation".to_string()],
                weight: 0.8,
                context_modifiers: vec!["work".to_string(), "project".to_string()],
            },
            SocialPattern {
                pattern_type: SocialPatternType::Conflict,
                keywords: vec!["disagree".to_string(), "conflict".to_string(), "argue".to_string(), 
                              "dispute".to_string(), "confrontation".to_string()],
                weight: 0.7,
                context_modifiers: vec!["resolution".to_string(), "mediation".to_string()],
            },
            SocialPattern {
                pattern_type: SocialPatternType::Support,
                keywords: vec!["help".to_string(), "support".to_string(), "assist".to_string(), 
                              "encourage".to_string(), "comfort".to_string()],
                weight: 0.9,
                context_modifiers: vec!["emotional".to_string(), "practical".to_string()],
            },
            SocialPattern {
                pattern_type: SocialPatternType::Authority,
                keywords: vec!["leader".to_string(), "manager".to_string(), "authority".to_string(), 
                              "command".to_string(), "direct".to_string()],
                weight: 0.6,
                context_modifiers: vec!["respect".to_string(), "challenge".to_string()],
            },
            SocialPattern {
                pattern_type: SocialPatternType::Learning,
                keywords: vec!["learn".to_string(), "teach".to_string(), "educate".to_string(), 
                              "mentor".to_string(), "guide".to_string()],
                weight: 0.8,
                context_modifiers: vec!["knowledge".to_string(), "skill".to_string()],
            },
        ]
    }
    
    /// Analyze emotional content
    fn analyze_emotional_content(&self, input: &str) -> f64 {
        let lowercase_input = input.to_lowercase();
        let words: Vec<&str> = lowercase_input.split_whitespace().collect();
        let mut total_valence = 0.0;
        let mut emotion_count = 0;
        
        for word in &words {
            if let Some(emotion) = self.emotional_lexicon.get(*word) {
                total_valence += emotion.valence;
                emotion_count += 1;
            }
        }
        
        if emotion_count > 0 {
            // Normalize to 0.0 - 1.0 range
            ((total_valence / emotion_count as f64) + 1.0) / 2.0
        } else {
            0.5 // Neutral
        }
    }
    
    /// Detect social patterns
    fn detect_social_patterns(&self, input: &str) -> f64 {
        let input_lower = input.to_lowercase();
        let mut pattern_scores = Vec::new();
        
        for pattern in &self.social_patterns {
            let mut matches = 0;
            for keyword in &pattern.keywords {
                if input_lower.contains(keyword) {
                    matches += 1;
                }
            }
            
            if matches > 0 {
                let base_score = (matches as f64 / pattern.keywords.len() as f64) * pattern.weight;
                
                // Check for context modifiers
                let mut modifier_boost = 1.0;
                for modifier in &pattern.context_modifiers {
                    if input_lower.contains(modifier) {
                        modifier_boost += 0.1;
                    }
                }
                
                pattern_scores.push(base_score * modifier_boost);
            }
        }
        
        if pattern_scores.is_empty() {
            0.3 // Base social relevance
        } else {
            pattern_scores.iter().sum::<f64>() / pattern_scores.len() as f64
        }
    }
    
    /// Extract participants from text
    fn extract_participants(&self, input: &str) -> Vec<String> {
        let mut participants = Vec::new();
        
        // Simple pronoun detection
        let pronouns = ["i", "you", "we", "they", "he", "she", "us", "them"];
        let lowercase_input = input.to_lowercase();
        let words: Vec<&str> = lowercase_input.split_whitespace().collect();
        
        for pronoun in &pronouns {
            if words.contains(pronoun) {
                participants.push(pronoun.to_string());
            }
        }
        
        // Look for names (capitalized words not at sentence start)
        let sentences: Vec<&str> = input.split('.').collect();
        for sentence in sentences {
            let words: Vec<&str> = sentence.trim().split_whitespace().collect();
            for (i, word) in words.iter().enumerate() {
                if i > 0 && word.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                    participants.push(word.to_string());
                }
            }
        }
        
        participants.dedup();
        participants
    }
    
    /// Analyze cultural context
    fn analyze_cultural_context(&self, input: &str) -> f64 {
        let cultural_indicators = [
            "culture", "tradition", "custom", "heritage", "community",
            "society", "social", "collective", "individual", "values"
        ];
        
        let input_lower = input.to_lowercase();
        let mut indicator_count = 0;
        
        for indicator in &cultural_indicators {
            if input_lower.contains(indicator) {
                indicator_count += 1;
            }
        }
        
        (indicator_count as f64 / 5.0).min(1.0) // Normalize to max 1.0
    }
    
    /// Extract cultural markers
    fn extract_cultural_markers(&self, input: &str) -> Vec<String> {
        let mut markers = Vec::new();
        let cultural_terms = [
            ("formal", "Formal Communication"),
            ("informal", "Informal Communication"),
            ("respectful", "Respect-Oriented"),
            ("collaborative", "Collaborative Culture"),
            ("hierarchical", "Hierarchical Structure"),
            ("egalitarian", "Egalitarian Values"),
        ];
        
        let input_lower = input.to_lowercase();
        for (term, marker) in &cultural_terms {
            if input_lower.contains(term) {
                markers.push(marker.to_string());
            }
        }
        
        markers
    }
    
    /// Analyze temporal dynamics
    async fn analyze_temporal_dynamics(&self) -> f64 {
        let history = self.context_history.read().await;
        
        if history.len() < 2 {
            return 0.5; // Neutral if insufficient history
        }
        
        // Calculate trend
        let recent = &history[history.len().saturating_sub(5)..];
        if recent.len() >= 2 {
            let first_score = recent.first().unwrap().overall_score;
            let last_score = recent.last().unwrap().overall_score;
            
            // Positive trend boosts score, negative trend reduces it
            let trend = (last_score - first_score) * 0.5;
            (0.5 + trend).clamp(0.0, 1.0)
        } else {
            0.5
        }
    }
    
    /// Determine social dynamics type
    fn determine_dynamics_type(&self, pattern_score: f64, emotional_score: f64) -> SocialDynamicsType {
        if pattern_score > 0.7 && emotional_score > 0.7 {
            SocialDynamicsType::Cooperative
        } else if pattern_score > 0.7 && emotional_score < 0.3 {
            SocialDynamicsType::Competitive
        } else if pattern_score < 0.3 && emotional_score < 0.3 {
            SocialDynamicsType::Adversarial
        } else if emotional_score > 0.8 {
            SocialDynamicsType::Supportive
        } else {
            SocialDynamicsType::Neutral
        }
    }
    
    /// Generate context description
    fn generate_context_description(&self, input: &str, participants: &[String]) -> String {
        format!(
            "Social context with {} participants: {}...",
            participants.len(),
            input.chars().take(50).collect::<String>()
        )
    }
    
    /// Add to history
    async fn add_to_history(&self, score: f64, emotional_score: f64, dynamics: SocialDynamicsType) {
        let mut history = self.context_history.write().await;
        
        let snapshot = ContextSnapshot {
            timestamp: Utc::now(),
            overall_score: score,
            dominant_emotion: if emotional_score > 0.6 { "Positive" } else if emotional_score < 0.4 { "Negative" } else { "Neutral" }.to_string(),
            social_dynamics: dynamics,
            participant_count: 0, // Would be set from actual participant analysis
        };
        
        history.push(snapshot);
        
        // Keep history bounded
        if history.len() > 1000 {
            history.remove(0);
        }
    }
}
