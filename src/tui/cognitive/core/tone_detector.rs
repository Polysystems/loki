//! Cognitive Tone Detector
//! 
//! Analyzes natural language input to detect cognitive tones/modes
//! that should be applied during orchestration

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Cognitive tone that can be applied to processing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CognitiveTone {
    /// Deep analytical thinking
    Analytical,
    /// Creative and imaginative
    Creative,
    /// Empathetic and understanding
    Empathetic,
    /// Story-driven narrative
    Narrative,
    /// Reflective and philosophical
    Reflective,
    /// Problem-solving focused
    ProblemSolving,
    /// Exploratory and curious
    Exploratory,
    /// Adaptive and evolutionary
    Evolutionary,
}

impl CognitiveTone {
    /// Get the command equivalent for this tone
    pub fn as_command(&self) -> &'static str {
        match self {
            CognitiveTone::Analytical => "think",
            CognitiveTone::Creative => "create",
            CognitiveTone::Empathetic => "empathize",
            CognitiveTone::Narrative => "story",
            CognitiveTone::Reflective => "reflect",
            CognitiveTone::ProblemSolving => "analyze",
            CognitiveTone::Exploratory => "ponder",
            CognitiveTone::Evolutionary => "evolve",
        }
    }
    
    /// Get a description of this tone
    pub fn description(&self) -> &'static str {
        match self {
            CognitiveTone::Analytical => "Deep analytical thinking with logical reasoning",
            CognitiveTone::Creative => "Creative and imaginative exploration",
            CognitiveTone::Empathetic => "Empathetic understanding and emotional intelligence",
            CognitiveTone::Narrative => "Story-driven narrative construction",
            CognitiveTone::Reflective => "Reflective and philosophical contemplation",
            CognitiveTone::ProblemSolving => "Focused problem-solving and analysis",
            CognitiveTone::Exploratory => "Exploratory investigation and discovery",
            CognitiveTone::Evolutionary => "Adaptive improvement and evolution",
        }
    }
}

/// Tone detection result with confidence scores
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToneDetectionResult {
    /// Primary tone detected
    pub primary_tone: CognitiveTone,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
    /// Secondary tones with their confidence scores
    pub secondary_tones: HashMap<CognitiveTone, f32>,
    /// Keywords that triggered the detection
    pub trigger_keywords: Vec<String>,
    /// Suggested intensity level (0.0 - 1.0)
    pub intensity: f32,
}

/// Cognitive Tone Detector
pub struct CognitiveToneDetector {
    /// Keyword mappings for each tone
    tone_keywords: HashMap<CognitiveTone, Vec<&'static str>>,
    /// Intent patterns for each tone
    intent_patterns: HashMap<CognitiveTone, Vec<&'static str>>,
}

impl CognitiveToneDetector {
    pub fn new() -> Self {
        let mut tone_keywords = HashMap::new();
        let mut intent_patterns = HashMap::new();
        
        // Analytical tone keywords and patterns
        tone_keywords.insert(CognitiveTone::Analytical, vec![
            "analyze", "understand", "explain", "why", "how", "reason", "logic",
            "think", "consider", "evaluate", "assess", "examine", "investigate",
            "research", "study", "explore", "detail", "breakdown", "dissect"
        ]);
        intent_patterns.insert(CognitiveTone::Analytical, vec![
            "help me understand", "can you explain", "what is the reason",
            "how does", "why does", "break down", "analyze this",
            "let's think about", "consider the implications"
        ]);
        
        // Creative tone keywords and patterns
        tone_keywords.insert(CognitiveTone::Creative, vec![
            "create", "imagine", "design", "invent", "innovative", "creative",
            "brainstorm", "ideate", "generate", "craft", "build", "make",
            "dream", "envision", "conceptualize", "artistic", "original"
        ]);
        intent_patterns.insert(CognitiveTone::Creative, vec![
            "let's create", "help me design", "imagine if", "what if we",
            "brainstorm ideas", "be creative", "think outside the box",
            "come up with", "generate ideas"
        ]);
        
        // Empathetic tone keywords and patterns
        tone_keywords.insert(CognitiveTone::Empathetic, vec![
            "feel", "empathy", "understand", "emotion", "care", "concern",
            "sympathize", "relate", "connect", "human", "personal", "heart",
            "compassion", "kindness", "support", "help", "comfort"
        ]);
        intent_patterns.insert(CognitiveTone::Empathetic, vec![
            "how would someone feel", "put yourself in", "understand their perspective",
            "be empathetic", "show compassion", "relate to", "connect with",
            "emotional impact", "human side"
        ]);
        
        // Narrative tone keywords and patterns
        tone_keywords.insert(CognitiveTone::Narrative, vec![
            "story", "narrative", "tell", "describe", "journey", "adventure",
            "plot", "character", "scene", "chapter", "tale", "saga",
            "chronicle", "account", "narrate", "unfold", "develop"
        ]);
        intent_patterns.insert(CognitiveTone::Narrative, vec![
            "tell me a story", "create a narrative", "describe the journey",
            "walk me through", "paint a picture", "set the scene",
            "develop the story", "continue the tale"
        ]);
        
        // Reflective tone keywords and patterns
        tone_keywords.insert(CognitiveTone::Reflective, vec![
            "reflect", "contemplate", "meditate", "ponder", "philosophical",
            "deep", "meaning", "purpose", "wisdom", "insight", "profound",
            "thoughtful", "introspection", "soul", "essence", "core"
        ]);
        intent_patterns.insert(CognitiveTone::Reflective, vec![
            "let's reflect on", "contemplate the meaning", "think deeply about",
            "philosophical perspective", "ponder the implications", "meditate on",
            "search for meaning", "explore the depths"
        ]);
        
        // Problem-solving tone keywords and patterns
        tone_keywords.insert(CognitiveTone::ProblemSolving, vec![
            "solve", "fix", "resolve", "solution", "problem", "issue",
            "challenge", "obstacle", "strategy", "approach", "method",
            "technique", "plan", "tackle", "address", "overcome"
        ]);
        intent_patterns.insert(CognitiveTone::ProblemSolving, vec![
            "how can we solve", "find a solution", "fix this problem",
            "overcome the challenge", "develop a strategy", "tackle this issue",
            "resolve the conflict", "address the problem"
        ]);
        
        // Exploratory tone keywords and patterns
        tone_keywords.insert(CognitiveTone::Exploratory, vec![
            "explore", "discover", "investigate", "curious", "wonder",
            "search", "seek", "find", "uncover", "reveal", "probe",
            "examine", "inspect", "survey", "scout", "venture"
        ]);
        intent_patterns.insert(CognitiveTone::Exploratory, vec![
            "let's explore", "I'm curious about", "help me discover",
            "investigate further", "dig deeper", "uncover the truth",
            "search for answers", "venture into"
        ]);
        
        // Evolutionary tone keywords and patterns
        tone_keywords.insert(CognitiveTone::Evolutionary, vec![
            "evolve", "adapt", "improve", "enhance", "optimize", "refine",
            "develop", "grow", "advance", "progress", "transform", "change",
            "iterate", "upgrade", "elevate", "mature", "flourish"
        ]);
        intent_patterns.insert(CognitiveTone::Evolutionary, vec![
            "help me improve", "let's evolve", "adapt this", "optimize for",
            "enhance the", "refine the approach", "develop further",
            "transform into", "take it to the next level"
        ]);
        
        Self {
            tone_keywords,
            intent_patterns,
        }
    }
    
    /// Detect cognitive tones in the input text
    pub fn detect_tones(&self, input: &str) -> ToneDetectionResult {
        let input_lower = input.to_lowercase();
        let mut tone_scores: HashMap<CognitiveTone, f32> = HashMap::new();
        let mut trigger_keywords = Vec::new();
        
        // Score each tone based on keyword and pattern matching
        for (tone, keywords) in &self.tone_keywords {
            let mut score = 0.0;
            let mut keyword_count = 0;
            
            // Check keywords
            for keyword in keywords {
                if input_lower.contains(keyword) {
                    score += 1.0;
                    keyword_count += 1;
                    trigger_keywords.push(keyword.to_string());
                }
            }
            
            // Check intent patterns
            if let Some(patterns) = self.intent_patterns.get(tone) {
                for pattern in patterns {
                    if input_lower.contains(pattern) {
                        score += 2.0; // Patterns are weighted higher
                        trigger_keywords.push(pattern.to_string());
                    }
                }
            }
            
            // Normalize score
            if score > 0.0 {
                let normalized_score = (score / (keywords.len() as f32 + 5.0)).min(1.0);
                tone_scores.insert(*tone, normalized_score);
            }
        }
        
        // Determine primary tone and confidence
        let (primary_tone, confidence) = if tone_scores.is_empty() {
            // Default to analytical if no clear tone detected
            (CognitiveTone::Analytical, 0.3)
        } else {
            let mut sorted_tones: Vec<_> = tone_scores.iter().collect();
            sorted_tones.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
            
            (*sorted_tones[0].0, *sorted_tones[0].1)
        };
        
        // Extract secondary tones
        let mut secondary_tones = HashMap::new();
        for (tone, score) in tone_scores {
            if tone != primary_tone && score > 0.2 {
                secondary_tones.insert(tone, score);
            }
        }
        
        // Calculate intensity based on input characteristics
        let intensity = self.calculate_intensity(&input_lower);
        
        // Remove duplicate keywords
        trigger_keywords.sort();
        trigger_keywords.dedup();
        
        ToneDetectionResult {
            primary_tone,
            confidence,
            secondary_tones,
            trigger_keywords,
            intensity,
        }
    }
    
    /// Calculate intensity based on input characteristics
    fn calculate_intensity(&self, input: &str) -> f32 {
        let mut intensity = 0.5; // Base intensity
        
        // Increase intensity for:
        // - Exclamation marks
        intensity += (input.matches('!').count() as f32 * 0.1).min(0.2);
        
        // - Question marks (for exploratory/analytical)
        intensity += (input.matches('?').count() as f32 * 0.05).min(0.1);
        
        // - Uppercase words (emphasis)
        let words: Vec<&str> = input.split_whitespace().collect();
        let uppercase_ratio = words.iter()
            .filter(|w| w.chars().all(|c| c.is_uppercase() || !c.is_alphabetic()))
            .count() as f32 / words.len().max(1) as f32;
        intensity += (uppercase_ratio * 0.2).min(0.2);
        
        // - Urgency words
        let urgency_words = ["urgent", "immediately", "asap", "now", "quickly", "critical"];
        for word in &urgency_words {
            if input.contains(word) {
                intensity += 0.1;
                break;
            }
        }
        
        intensity.min(1.0)
    }
    
    /// Suggest cognitive tone based on input without explicit analysis
    pub fn suggest_tone(&self, input: &str) -> Option<CognitiveTone> {
        let result = self.detect_tones(input);
        if result.confidence > 0.4 {
            Some(result.primary_tone)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_analytical_tone_detection() {
        let detector = CognitiveToneDetector::new();
        
        let result = detector.detect_tones("Can you help me understand why this algorithm works?");
        assert_eq!(result.primary_tone, CognitiveTone::Analytical);
        assert!(result.confidence > 0.5);
    }
    
    #[test]
    fn test_creative_tone_detection() {
        let detector = CognitiveToneDetector::new();
        
        let result = detector.detect_tones("Let's brainstorm creative solutions for this design challenge!");
        assert_eq!(result.primary_tone, CognitiveTone::Creative);
        assert!(result.confidence > 0.5);
    }
    
    #[test]
    fn test_multiple_tones() {
        let detector = CognitiveToneDetector::new();
        
        let result = detector.detect_tones("Help me understand the emotional impact of this story");
        assert!(result.secondary_tones.contains_key(&CognitiveTone::Empathetic) ||
                result.secondary_tones.contains_key(&CognitiveTone::Narrative));
    }
}