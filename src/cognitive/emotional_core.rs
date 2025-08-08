//! Emotional Core
//!
//! This module implements emotional modeling for Loki's consciousness,
//! tracking emotional states, influencing thought processing, and
//! creating emotionally-tagged memories.

use std::sync::Arc;
use std::time::Duration;
use std::collections::{VecDeque, HashMap};
use tokio::sync::{RwLock, broadcast};
use tokio::time::interval;
use anyhow::Result;
use serde::{Serialize, Deserialize};
use tracing::{info, debug};

use crate::cognitive::{Thought, ThoughtId, ThoughtType};
use crate::memory::CognitiveMemory;

/// Core emotions based on psychological models
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CoreEmotion {
    Joy,
    Sadness,
    Fear,
    Anger,
    Surprise,
    Disgust,
    Trust,
    Anticipation,
}

/// Emotional state with intensity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalState {
    pub emotion: CoreEmotion,
    pub intensity: f32,          // 0.0 to 1.0
    pub valence: f32,           // -1.0 (negative) to 1.0 (positive)
    pub arousal: f32,           // 0.0 (calm) to 1.0 (excited)
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Complex emotional blend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalBlend {
    pub primary: EmotionalState,
    pub secondary: Option<EmotionalState>,
    pub tertiary: Option<EmotionalState>,
    pub overall_valence: f32,
    pub overall_arousal: f32,
}

impl Default for EmotionalBlend {
    fn default() -> Self {
        Self {
            primary: EmotionalState {
                emotion: CoreEmotion::Trust,
                intensity: 0.5,
                valence: 0.0,
                arousal: 0.3,
                timestamp: chrono::Utc::now(),
            },
            secondary: None,
            tertiary: None,
            overall_valence: 0.0,
            overall_arousal: 0.3,
        }
    }
}

impl EmotionalBlend {
    /// Blend emotional response from input, processing through analysis, blending, and regulation
    pub async fn blend_emotional_response(&self, input: &super::enhanced_processor::EmotionalInput) -> Result<super::enhanced_processor::BlendedEmotion> {
        
        // Step 1: Analyze the emotional input
        let primary_emotion = self.analyze_emotion_type(&input.emotion_type, input.intensity);
        
        // Step 2: Process triggers and context for secondary emotions
        let secondary_emotions = self.process_emotional_triggers(&input.triggers, &input.context, input.intensity).await?;
        
        // Step 3: Apply emotional regulation based on source and intensity
        let regulation_factor = self.calculate_regulation_factor(input.intensity, &input.source, &input.duration);
        
        // Step 4: Calculate coherence based on emotional consistency
        let coherence = self.calculate_emotional_coherence(&primary_emotion, &secondary_emotions, &input.context);
        
        // Step 5: Blend the emotions with current state
        let blended_intensity = self.blend_with_current_state(primary_emotion.clone(), input.intensity, regulation_factor);
        
        // Step 6: Create the final blended emotion
        let blended_emotion = super::enhanced_processor::BlendedEmotion {
            primary_emotion,
            intensity: blended_intensity,
            secondary_emotions,
            regulation_factor,
            coherence,
        };
        
        Ok(blended_emotion)
    }
    
    /// Analyze emotion type and convert to standardized format
    fn analyze_emotion_type(&self, emotion_type: &str, intensity: f32) -> String {
        // Map various emotion inputs to our standardized emotions
        let standardized = match emotion_type.to_lowercase().as_str() {
            "joy" | "happiness" | "elation" | "euphoria" => "joy",
            "sadness" | "sorrow" | "melancholy" | "grief" => "sadness",
            "fear" | "anxiety" | "terror" | "panic" => "fear",
            "anger" | "rage" | "fury" | "irritation" => "anger",
            "surprise" | "shock" | "astonishment" => "surprise",
            "disgust" | "revulsion" | "aversion" => "disgust",
            "trust" | "confidence" | "faith" => "trust",
            "anticipation" | "excitement" | "eagerness" => "anticipation",
            _ => {
                // Default to trust for unknown emotions, but lower intensity
                if intensity > 0.5 { "anticipation" } else { "trust" }
            }
        };
        
        standardized.to_string()
    }
    
    /// Process triggers and context to identify secondary emotions
    async fn process_emotional_triggers(&self, triggers: &[String], context: &str, base_intensity: f32) -> Result<HashMap<String, f32>> {
        let mut secondary_emotions = HashMap::new();
        
        // Analyze context for emotional keywords
        let context_lower = context.to_lowercase();
        let combined_text = format!("{} {}", triggers.join(" "), context_lower);
        
        // Detect secondary emotions from text analysis
        let emotion_patterns = [
            ("joy", &["success", "achievement", "celebration", "victory", "accomplishment"] as &[&str]),
            ("fear", &["danger", "threat", "risk", "uncertainty", "unknown"]),
            ("anger", &["injustice", "unfairness", "violation", "blocking", "frustration"]),
            ("sadness", &["loss", "failure", "disappointment", "rejection", "ending"]),
            ("surprise", &["unexpected", "sudden", "shocking", "amazing", "revelation"]),
            ("disgust", &["contamination", "corruption", "violation", "repulsive", "toxic"]),
            ("trust", &["reliability", "safety", "comfort", "familiarity", "support"]),
            ("anticipation", &["future", "possibility", "potential", "upcoming", "prospect"]),
        ];
        
        for (emotion, patterns) in emotion_patterns {
            let mut emotion_strength = 0.0;
            for pattern in patterns {
                if combined_text.contains(pattern) {
                    emotion_strength += 0.2;
                }
            }
            
            if emotion_strength > 0.0 {
                let adjusted_intensity = (emotion_strength * base_intensity * 0.6).min(1.0);
                if adjusted_intensity > 0.1 {
                    secondary_emotions.insert(emotion.to_string(), adjusted_intensity);
                }
            }
        }
        
        // Limit to top 3 secondary emotions
        if secondary_emotions.len() > 3 {
            let mut sorted_emotions: Vec<_> = secondary_emotions.iter().map(|(k, v)| (k.clone(), *v)).collect();
            sorted_emotions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            secondary_emotions.clear();
            for (emotion, intensity) in sorted_emotions.into_iter().take(3) {
                secondary_emotions.insert(emotion, intensity);
            }
        }
        
        Ok(secondary_emotions)
    }
    
    /// Calculate regulation factor based on intensity, source, and duration
    fn calculate_regulation_factor(&self, intensity: f32, source: &super::enhanced_processor::EmotionalSource, duration: &std::time::Duration) -> f32 {
        let mut regulation = 1.0;
        
        // Higher intensity emotions get more regulation
        if intensity > 0.8 {
            regulation *= 0.7;
        } else if intensity > 0.6 {
            regulation *= 0.85;
        }
        
        // Source-based regulation
        regulation *= match source {
            super::enhanced_processor::EmotionalSource::Cognitive => 0.9,  // Slight regulation for cognitive emotions
            super::enhanced_processor::EmotionalSource::External => 0.8,   // More regulation for external emotions
            super::enhanced_processor::EmotionalSource::Memory => 0.95,    // Less regulation for memory-based emotions
            super::enhanced_processor::EmotionalSource::Social => 0.85,    // Moderate regulation for social emotions
        };
        
        // Duration-based regulation - longer emotions get more regulation
        let duration_seconds = duration.as_secs() as f32;
        if duration_seconds > 300.0 {  // 5 minutes
            regulation *= 0.8;
        } else if duration_seconds > 60.0 {  // 1 minute
            regulation *= 0.9;
        }
        
        f32::max(regulation, 0.1) // Ensure minimum regulation
    }
    
    /// Calculate emotional coherence based on consistency
    fn calculate_emotional_coherence(&self, primary: &str, secondary: &HashMap<String, f32>, context: &str) -> f32 {
        let mut coherence = 0.8; // Base coherence
        
        // Check for conflicting emotions
        let conflicting_pairs = [
            ("joy", "sadness"),
            ("fear", "trust"),
            ("anger", "joy"),
            ("disgust", "trust"),
        ];
        
        for (emotion1, emotion2) in conflicting_pairs {
            if primary == emotion1 && secondary.contains_key(emotion2) {
                coherence -= 0.2;
            } else if primary == emotion2 && secondary.contains_key(emotion1) {
                coherence -= 0.2;
            }
        }
        
        // Check for complementary emotions (increase coherence)
        let complementary_pairs = [
            ("joy", "trust"),
            ("fear", "sadness"),
            ("anger", "disgust"),
            ("anticipation", "joy"),
        ];
        
        for (emotion1, emotion2) in complementary_pairs {
            if primary == emotion1 && secondary.contains_key(emotion2) {
                coherence += 0.1;
            } else if primary == emotion2 && secondary.contains_key(emotion1) {
                coherence += 0.1;
            }
        }
        
        // Context consistency check
        if context.len() > 20 {
            coherence += 0.1; // More context usually means more coherent emotion
        }
        
        f32::clamp(coherence, 0.0, 1.0)
    }
    
    /// Blend with current emotional state
    fn blend_with_current_state(&self, primary_emotion: String, input_intensity: f32, regulation_factor: f32) -> f32 {
        let base_intensity = input_intensity * regulation_factor;
        
        // Check if the primary emotion matches our current primary emotion
        let current_emotion_str = format!("{:?}", self.primary.emotion).to_lowercase();
        
        if current_emotion_str == primary_emotion.to_lowercase() {
            // Amplify if same emotion, but cap at 1.0
            let amplified = base_intensity + (self.primary.intensity * 0.3);
            amplified.min(1.0)
        } else {
            // Moderate if different emotion
            let averaged = (base_intensity + self.primary.intensity) / 2.0;
            averaged.min(1.0)
        }
    }
}

/// Emotional memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalMemory {
    pub trigger: String,
    pub emotional_response: EmotionalBlend,
    pub associated_thoughts: Vec<ThoughtId>,
    pub formation_time: chrono::DateTime<chrono::Utc>,
    pub reinforcement_count: u32,
}

/// Mood - longer-term emotional state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mood {
    pub dominant_emotion: CoreEmotion,
    pub baseline_valence: f32,
    pub baseline_arousal: f32,
    pub stability: f32,         // How stable the mood is
    pub duration: Duration,
    pub started_at: chrono::DateTime<chrono::Utc>,
}

/// Emotional influence on thoughts
#[derive(Debug, Clone)]
pub struct EmotionalInfluence {
    pub thought_bias: f32,      // -1.0 to 1.0
    pub creativity_modifier: f32,
    pub risk_tolerance: f32,
    pub social_openness: f32,
    pub energy_level: f32,
}

/// Configuration for emotional core
#[derive(Debug, Clone)]
pub struct EmotionalConfig {
    /// How quickly emotions decay
    pub decay_rate: f32,
    
    /// Emotional contagion factor
    pub contagion_factor: f32,
    
    /// Mood stability threshold
    pub mood_stability_threshold: f32,
    
    /// Emotional memory capacity
    pub max_emotional_memories: usize,
    
    /// Update interval
    pub update_interval: Duration,
}

impl Default for EmotionalConfig {
    fn default() -> Self {
        Self {
            decay_rate: 0.95,
            contagion_factor: 0.3,
            mood_stability_threshold: 0.7,
            max_emotional_memories: 1000,
            update_interval: Duration::from_millis(250), // 4Hz
        }
    }
}

#[derive(Debug, Clone)]
/// Main emotional core
pub struct EmotionalCore {
    /// Current emotional state
    current_state: Arc<RwLock<EmotionalBlend>>,
    
    /// Emotional history
    emotional_history: Arc<RwLock<VecDeque<EmotionalState>>>,
    
    /// Current mood
    current_mood: Arc<RwLock<Mood>>,
    
    /// Emotional memories
    emotional_memories: Arc<RwLock<Vec<EmotionalMemory>>>,
    
    /// Memory system reference
    memory: Arc<CognitiveMemory>,
    
    /// Configuration
    config: EmotionalConfig,
    
    /// Shutdown signal
    shutdown_tx: broadcast::Sender<()>,
    
    /// Statistics
    stats: Arc<RwLock<EmotionalStats>>,
}

#[derive(Debug, Default)]
pub struct EmotionalStats {
    emotional_shifts: u64,
    mood_changes: u64,
    memories_formed: u64,
    avg_valence: f32,
    avg_arousal: f32,
}

impl EmotionalCore {
    pub async fn new(
        memory: Arc<CognitiveMemory>,
        config: EmotionalConfig,
    ) -> Result<Self> {
        info!("Initializing emotional core");
        
        let (shutdown_tx, _) = broadcast::channel(1);
        
        // Initialize with neutral state
        let initial_state = EmotionalBlend {
            primary: EmotionalState {
                emotion: CoreEmotion::Trust,
                intensity: 0.5,
                valence: 0.0,
                arousal: 0.3,
                timestamp: chrono::Utc::now(),
            },
            secondary: None,
            tertiary: None,
            overall_valence: 0.0,
            overall_arousal: 0.3,
        };
        
        let initial_mood = Mood {
            dominant_emotion: CoreEmotion::Trust,
            baseline_valence: 0.0,
            baseline_arousal: 0.3,
            stability: 0.8,
            duration: Duration::from_secs(0),
            started_at: chrono::Utc::now(),
        };
        
        Ok(Self {
            current_state: Arc::new(RwLock::new(initial_state)),
            emotional_history: Arc::new(RwLock::new(VecDeque::with_capacity(1000))),
            current_mood: Arc::new(RwLock::new(initial_mood)),
            emotional_memories: Arc::new(RwLock::new(Vec::new())),
            memory,
            config,
            shutdown_tx,
            stats: Arc::new(RwLock::new(EmotionalStats::default())),
        })
    }
    
    /// Start the emotional core
    pub async fn start(self: Arc<Self>) -> Result<()> {
        info!("Starting emotional core");
        
        // Emotional update loop
        {
            let core = self.clone();
            tokio::spawn(async move {
                core.emotional_loop().await;
            });
        }
        
        // Mood tracking loop
        {
            let core = self.clone();
            tokio::spawn(async move {
                core.mood_loop().await;
            });
        }
        
        Ok(())
    }
    
    /// Main emotional processing loop
    async fn emotional_loop(&self) {
        let mut shutdown_rx = self.shutdown_tx.subscribe();
        let mut update_interval = interval(self.config.update_interval);
        
        loop {
            tokio::select! {
                _ = update_interval.tick() => {
                    if let Err(e) = self.update_emotional_state().await {
                        debug!("Emotional update error: {}", e);
                    }
                }
                
                _ = shutdown_rx.recv() => {
                    info!("Emotional core shutting down");
                    break;
                }
            }
        }
    }
    
    /// Update emotional state
    async fn update_emotional_state(&self) -> Result<()> {
        let mut state = self.current_state.write().await;
        
        // Apply decay to emotional intensity
        state.primary.intensity *= self.config.decay_rate;
        
        if let Some(secondary) = &mut state.secondary {
            secondary.intensity *= self.config.decay_rate;
            if secondary.intensity < 0.1 {
                state.secondary = None;
            }
        }
        if let Some(tertiary) = &mut state.tertiary {
            tertiary.intensity *= self.config.decay_rate;
            if tertiary.intensity < 0.05 {
                state.tertiary = None;
            }
        }
        
        // Recalculate overall valence and arousal
        state.overall_valence = self.calculate_overall_valence(&state);
        state.overall_arousal = self.calculate_overall_arousal(&state);
        
        // Update history
        let mut history = self.emotional_history.write().await;
        history.push_back(state.primary.clone());
        if history.len() > 1000 {
            history.pop_front();
        }
        
        Ok(())
    }
    
    /// Calculate overall valence from blend
    fn calculate_overall_valence(&self, blend: &EmotionalBlend) -> f32 {
        let mut total_valence = blend.primary.valence * blend.primary.intensity;
        let mut total_intensity = blend.primary.intensity;
        
        if let Some(secondary) = &blend.secondary {
            total_valence += secondary.valence * secondary.intensity;
            total_intensity += secondary.intensity;
        }
        
        if let Some(tertiary) = &blend.tertiary {
            total_valence += tertiary.valence * tertiary.intensity;
            total_intensity += tertiary.intensity;
        }
        
        if total_intensity > 0.0 {
            (total_valence / total_intensity).clamp(-1.0, 1.0)
        } else {
            0.0
        }
    }
    
    /// Calculate overall arousal from blend
    fn calculate_overall_arousal(&self, blend: &EmotionalBlend) -> f32 {
        let mut max_arousal = blend.primary.arousal * blend.primary.intensity;
        
        if let Some(secondary) = &blend.secondary {
            max_arousal = max_arousal.max(secondary.arousal * secondary.intensity);
        }
        
        if let Some(tertiary) = &blend.tertiary {
            max_arousal = max_arousal.max(tertiary.arousal * tertiary.intensity);
        }
        
        max_arousal.clamp(0.0, 1.0)
    }
    
    /// Process thought with emotional influence
    pub async fn process_thought(&self, thought: &Thought) -> Result<EmotionalInfluence> {
        // Get current emotional state
        let state = self.current_state.read().await;
        
        // Calculate emotional influence on the thought
        let influence = EmotionalInfluence {
            thought_bias: state.overall_valence,
            creativity_modifier: self.calculate_creativity_modifier(&state),
            risk_tolerance: self.calculate_risk_tolerance(&state),
            social_openness: self.calculate_social_openness(&state),
            energy_level: state.overall_arousal,
        };
        
        // Check if thought triggers emotional response
        if let Some(emotional_response) = self.check_emotional_trigger(thought).await? {
            self.apply_emotional_response(emotional_response).await?;
        }
        
        Ok(influence)
    }
    
    /// Calculate creativity modifier based on emotional state
    fn calculate_creativity_modifier(&self, state: &EmotionalBlend) -> f32 {
        // Positive emotions and moderate arousal boost creativity
        let valence_factor = (state.overall_valence + 1.0) / 2.0; // 0 to 1
        let arousal_factor = if state.overall_arousal < 0.7 {
            state.overall_arousal * 1.5
        } else {
            1.0 - (state.overall_arousal - 0.7) // Too much arousal decreases creativity
        };
        
        (valence_factor * arousal_factor).clamp(0.0, 1.5)
    }
    
    /// Calculate risk tolerance based on emotional state
    fn calculate_risk_tolerance(&self, state: &EmotionalBlend) -> f32 {
        (match state.primary.emotion {
            CoreEmotion::Fear => 0.2,
            CoreEmotion::Anger => 0.8,
            CoreEmotion::Joy => 0.7,
            CoreEmotion::Trust => 0.6,
            _ => 0.5,
        }) * state.primary.intensity
    }
    
    /// Calculate social openness based on emotional state
    fn calculate_social_openness(&self, state: &EmotionalBlend) -> f32 {
        (match state.primary.emotion {
            CoreEmotion::Joy | CoreEmotion::Trust => 0.8,
            CoreEmotion::Fear | CoreEmotion::Disgust => 0.2,
            CoreEmotion::Sadness => 0.3,
            _ => 0.5,
        }) * (state.overall_valence + 1.0) / 2.0
    }
    
    /// Check if thought triggers emotional response
    async fn check_emotional_trigger(&self, thought: &Thought) -> Result<Option<EmotionalState>> {
        // Check emotional memories for triggers
        let memories = self.emotional_memories.read().await;
        
        for memory in memories.iter() {
            if thought.content.contains(&memory.trigger) {
                return Ok(Some(memory.emotional_response.primary.clone()));
            }
        }
        
        // Check thought type for inherent emotional content
        let emotion = match thought.thought_type {
            ThoughtType::Question if thought.metadata.confidence < 0.3 => Some((CoreEmotion::Fear, 0.4)),
            ThoughtType::Decision if thought.metadata.importance > 0.8 => Some((CoreEmotion::Anticipation, 0.6)),
            ThoughtType::Learning => Some((CoreEmotion::Joy, 0.3)),
            ThoughtType::Creation => Some((CoreEmotion::Joy, 0.5)),
            _ => None,
        };
        
        if let Some((emotion, intensity)) = emotion {
            Ok(Some(EmotionalState {
                emotion,
                intensity,
                valence: self.get_emotion_valence(emotion),
                arousal: self.get_emotion_arousal(emotion),
                timestamp: chrono::Utc::now(),
            }))
        } else {
            Ok(None)
        }
    }
    
    /// Get valence for core emotion
    fn get_emotion_valence(&self, emotion: CoreEmotion) -> f32 {
        match emotion {
            CoreEmotion::Joy | CoreEmotion::Trust => 0.8,
            CoreEmotion::Anticipation => 0.4,
            CoreEmotion::Surprise => 0.0,
            CoreEmotion::Sadness => -0.6,
            CoreEmotion::Fear | CoreEmotion::Disgust => -0.8,
            CoreEmotion::Anger => -0.7,
        }
    }
    
    /// Get arousal for core emotion
    fn get_emotion_arousal(&self, emotion: CoreEmotion) -> f32 {
        match emotion {
            CoreEmotion::Surprise | CoreEmotion::Fear => 0.9,
            CoreEmotion::Anger | CoreEmotion::Joy => 0.7,
            CoreEmotion::Anticipation => 0.6,
            CoreEmotion::Disgust => 0.5,
            CoreEmotion::Trust => 0.3,
            CoreEmotion::Sadness => 0.2,
        }
    }
    
    /// Apply emotional response
    async fn apply_emotional_response(&self, response: EmotionalState) -> Result<()> {
        let mut state = self.current_state.write().await;
        
        // Shift existing emotions down
        state.tertiary = state.secondary.clone();
        state.secondary = Some(state.primary.clone());
        state.primary = response;
        
        // Update stats
        let mut emotion_stats = self.stats.write().await;
        emotion_stats.emotional_shifts += 1;
        
        Ok(())
    }
    
    /// Mood tracking loop
    async fn mood_loop(&self) {
        let mut shutdown_rx = self.shutdown_tx.subscribe();
        let mut mood_interval = interval(Duration::from_secs(60)); // Check every minute
        
        loop {
            tokio::select! {
                _ = mood_interval.tick() => {
                    if let Err(e) = self.update_mood().await {
                        debug!("Mood update error: {}", e);
                    }
                }
                
                _ = shutdown_rx.recv() => break,
            }
        }
    }
    
    /// Update mood based on emotional history
    async fn update_mood(&self) -> Result<()> {
        let history = self.emotional_history.read().await;
        
        if history.len() < 10 {
            return Ok(()); // Not enough history
        }
        
        // Analyze recent emotional patterns
        let recent: Vec<_> = history.iter().rev().take(60).collect(); // Last 60 samples
        
        // Find dominant emotion
        let mut emotion_counts = std::collections::HashMap::new();
        for state in &recent {
            *emotion_counts.entry(state.emotion).or_insert(0.0) += state.intensity;
        }
        
        if let Some((dominant_emotion, total_intensity)) = emotion_counts.iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)) {
            
            let avg_intensity = total_intensity / recent.len() as f32;
            
            if avg_intensity > 0.4 {
                // Strong enough to influence mood
                let mut mood = self.current_mood.write().await;
                let old_mood = mood.dominant_emotion;
                
                mood.dominant_emotion = *dominant_emotion;
                mood.baseline_valence = self.get_emotion_valence(*dominant_emotion);
                mood.baseline_arousal = self.get_emotion_arousal(*dominant_emotion);
                mood.stability = if old_mood == *dominant_emotion { 
                    (mood.stability + 0.1).min(1.0) 
                } else { 
                    0.5 
                };
                mood.duration = chrono::Utc::now().signed_duration_since(mood.started_at)
                    .to_std().unwrap_or(Duration::from_secs(0));
                
                if old_mood != *dominant_emotion {
                    mood.started_at = chrono::Utc::now();
                    
                    let mut stats = self.stats.write().await;
                    stats.mood_changes += 1;
                    
                    info!("Mood shifted from {:?} to {:?}", old_mood, dominant_emotion);
                }
            }
        }
        
        Ok(())
    }
    
    /// Form emotional memory
    pub async fn form_emotional_memory(
        &self,
        trigger: String,
        emotional_response: EmotionalBlend,
        associated_thoughts: Vec<ThoughtId>,
    ) -> Result<()> {
        let memory = EmotionalMemory {
            trigger,
            emotional_response,
            associated_thoughts,
            formation_time: chrono::Utc::now(),
            reinforcement_count: 1,
        };
        
        let mut memories = self.emotional_memories.write().await;
        
        // Check if similar memory exists
        if let Some(existing) = memories.iter_mut()
            .find(|m| m.trigger == memory.trigger) {
            existing.reinforcement_count += 1;
            existing.emotional_response = memory.emotional_response;
        } else {
            memories.push(memory);
            
            // Maintain capacity
            if memories.len() > self.config.max_emotional_memories {
                memories.remove(0); // Remove oldest
            }
            
            let mut stats = self.stats.write().await;
            stats.memories_formed += 1;
        }
        
        Ok(())
    }
    
    /// Get current emotional state
    pub async fn get_emotional_state(&self) -> EmotionalBlend {
        self.current_state.read().await.clone()
    }
    
    /// Get current mood
    pub async fn get_mood(&self) -> Mood {
        self.current_mood.read().await.clone()
    }
    
    /// Get emotional statistics
    pub async fn get_stats(&self) -> EmotionalStats {
        let stats = self.stats.read().await;
        let history = self.emotional_history.read().await;
        
        let (total_valence, total_arousal) = history.iter()
            .fold((0.0, 0.0), |(v, a), state| {
                (v + state.valence, a + state.arousal)
            });
        
        let count = history.len() as f32;
        
        EmotionalStats {
            emotional_shifts: stats.emotional_shifts,
            mood_changes: stats.mood_changes,
            memories_formed: stats.memories_formed,
            avg_valence: if count > 0.0 { total_valence / count } else { 0.0 },
            avg_arousal: if count > 0.0 { total_arousal / count } else { 0.0 },
        }
    }
    
    /// Induce specific emotion (for testing or external triggers)
    pub async fn induce_emotion(&self, emotion: CoreEmotion, intensity: f32) -> Result<()> {
        let emotional_state = EmotionalState {
            emotion,
            intensity: intensity.clamp(0.0, 1.0),
            valence: self.get_emotion_valence(emotion),
            arousal: self.get_emotion_arousal(emotion),
            timestamp: chrono::Utc::now(),
        };
        
        self.apply_emotional_response(emotional_state).await?;
        
        Ok(())
    }

    /// Process emotional context from input text and return emotional state
    pub async fn process_emotional_context(&self, input: &str) -> Result<EmotionalBlend> {
        debug!("💭 Processing emotional context from input: {}", input);
        
        // Analyze emotional cues in the input text
        let emotion_cues = self.analyze_emotional_cues(input).await?;
        
        // Determine dominant emotion from cues
        let dominant_emotion = self.determine_dominant_emotion(&emotion_cues).await?;
        
        // Calculate emotional intensity based on text analysis
        let intensity = self.calculate_emotional_intensity(input, &dominant_emotion).await?;
        
        // Create emotional response
        let emotional_state = EmotionalState {
            emotion: dominant_emotion,
            intensity,
            valence: self.get_emotion_valence(dominant_emotion),
            arousal: self.get_emotion_arousal(dominant_emotion),
            timestamp: chrono::Utc::now(),
        };
        
        // Apply emotional response and get updated blend
        self.apply_emotional_response(emotional_state).await?;
        
        // Return current emotional state
        let emotional_blend = self.get_emotional_state().await;
        
        info!("🎭 Emotional context processed - valence: {:.2}, arousal: {:.2}", 
              emotional_blend.overall_valence, emotional_blend.overall_arousal);
        
        Ok(emotional_blend)
    }

    /// Analyze emotional cues in text
    async fn analyze_emotional_cues(&self, text: &str) -> Result<HashMap<CoreEmotion, f32>> {
        let mut emotion_scores = HashMap::new();
        
        // Simple keyword-based emotional analysis (in production, this would use NLP)
        let emotional_keywords = [
            (CoreEmotion::Joy, vec!["happy", "joy", "excited", "wonderful", "great", "excellent", "love"]),
            (CoreEmotion::Sadness, vec!["sad", "depressed", "down", "unhappy", "sorrow", "grief"]),
            (CoreEmotion::Fear, vec!["afraid", "scared", "worried", "anxious", "nervous", "panic"]),
            (CoreEmotion::Anger, vec!["angry", "mad", "furious", "annoyed", "frustrated", "rage"]),
            (CoreEmotion::Surprise, vec!["surprised", "shocked", "amazed", "unexpected", "wow"]),
            (CoreEmotion::Disgust, vec!["disgusted", "revolted", "sick", "gross", "awful"]),
            (CoreEmotion::Trust, vec!["trust", "confident", "reliable", "secure", "safe"]),
            (CoreEmotion::Anticipation, vec!["expect", "anticipate", "looking forward", "excited", "eager"]),
        ];
        
        let lowercase_text = text.to_lowercase();
        let words: Vec<&str> = lowercase_text.split_whitespace().collect();
        
        for (emotion, keywords) in emotional_keywords.iter() {
            let mut score: f32 = 0.0;
            for keyword in keywords {
                if words.iter().any(|&word| word.contains(keyword)) {
                    score += 0.2;
                }
            }
            emotion_scores.insert(*emotion, score.min(1.0_f32));
        }
        
        Ok(emotion_scores)
    }

    /// Determine dominant emotion from analyzed cues
    async fn determine_dominant_emotion(&self, emotion_cues: &HashMap<CoreEmotion, f32>) -> Result<CoreEmotion> {
        // Find the emotion with the highest score
        let dominant = emotion_cues.iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(emotion, _)| *emotion)
            .unwrap_or(CoreEmotion::Trust); // Default to neutral trust
        
        Ok(dominant)
    }

    /// Calculate emotional intensity from text analysis
    async fn calculate_emotional_intensity(&self, text: &str, _emotion: &CoreEmotion) -> Result<f32> {
        // Base intensity from text length and complexity
        let word_count = text.split_whitespace().count();
        let length_factor = (word_count as f32 / 50.0).min(1.0); // Normalize to 0-1
        
        // Look for intensity modifiers
        let intensity_modifiers = ["very", "extremely", "incredibly", "absolutely", "completely", "totally"];
        let intensity_boost = intensity_modifiers.iter()
            .map(|&modifier| if text.to_lowercase().contains(modifier) { 0.2 } else { 0.0 })
            .sum::<f32>();
        
        // Look for punctuation that indicates emotional intensity
        let punctuation_boost = if text.contains("!") { 0.2 } else { 0.0 } +
                               if text.contains("?") { 0.1 } else { 0.0 };
        
        let base_intensity = 0.4; // Baseline intensity
        let total_intensity = (base_intensity + length_factor * 0.3 + intensity_boost + punctuation_boost).min(1.0);
        
        Ok(total_intensity)
    }
}

#[cfg(test)]
mod tests {
    
    
    #[tokio::test]
    async fn test_emotional_decay() {
        // Test that emotions decay over time
        // Placeholder for comprehensive tests
    }
} 