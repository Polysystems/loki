use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::info;

use crate::memory::CognitiveMemory;
use crate::persistence::{
    LearningOutcome,
    LogCategory,
    LogEntry,
    PersistenceManager,
    ThoughtMetadata,
};

/// Self-reflection system for analyzing Loki's own behavior
pub struct SelfReflection {
    /// Persistence manager for accessing logs
    persistence: Arc<PersistenceManager>,

    /// Memory system for insights
    memory: Arc<CognitiveMemory>,

    /// Current reflection state
    state: Arc<RwLock<ReflectionState>>,

    /// Reflection patterns
    patterns: Arc<RwLock<Vec<ReflectionPattern>>>,
}

impl SelfReflection {
    /// Create new self-reflection system
    pub async fn new(
        persistence: Arc<PersistenceManager>,
        memory: Arc<CognitiveMemory>,
    ) -> Result<Self> {
        let patterns = vec![
            ReflectionPattern::error_pattern(),
            ReflectionPattern::learning_pattern(),
            ReflectionPattern::decision_pattern(),
            ReflectionPattern::emotional_pattern(),
            ReflectionPattern::efficiency_pattern(),
        ];

        Ok(Self {
            persistence,
            memory,
            state: Arc::new(RwLock::new(ReflectionState::default())),
            patterns: Arc::new(RwLock::new(patterns)),
        })
    }

    /// Perform a reflection session
    pub async fn reflect(&self, depth: ReflectionDepth) -> Result<ReflectionReport> {
        info!("Starting self-reflection session with depth: {:?}", depth);

        let lookback = match depth {
            ReflectionDepth::Surface => Duration::hours(1),
            ReflectionDepth::Daily => Duration::hours(24),
            ReflectionDepth::Weekly => Duration::days(7),
            ReflectionDepth::Deep => Duration::days(30),
        };

        // Get recent logs
        let consciousness_logs = self
            .persistence
            .get_recent_logs(lookback.num_hours() as u32, Some(LogCategory::Consciousness))
            .await?;

        let learning_logs = self
            .persistence
            .get_recent_logs(lookback.num_hours() as u32, Some(LogCategory::Learning))
            .await?;

        let error_logs = self
            .persistence
            .get_recent_logs(lookback.num_hours() as u32, Some(LogCategory::Error))
            .await?;

        // Analyze patterns
        let mut insights = Vec::new();
        let patterns = self.patterns.read().await;

        for pattern in patterns.iter() {
            let pattern_insights = self
                .analyze_pattern(pattern, &consciousness_logs, &learning_logs, &error_logs)
                .await?;
            insights.extend(pattern_insights);
        }

        // Analyze thought coherence
        let coherence = self.analyze_coherence(&consciousness_logs).await?;

        // Analyze emotional state
        let emotional_state = self.analyze_emotions(&consciousness_logs).await?;

        // Generate recommendations
        let recommendations = self.generate_recommendations(&insights, coherence).await?;

        // Create report
        let report = ReflectionReport {
            timestamp: Utc::now(),
            depth,
            period: lookback,
            insights,
            coherence_score: coherence,
            emotional_state,
            recommendations,
            statistics: self.calculate_statistics(&consciousness_logs, &learning_logs, &error_logs),
        };

        // Log the reflection
        self.persistence
            .log_thought(
                format!(
                    "Completed self-reflection: {} insights, coherence {:.2}",
                    report.insights.len(),
                    coherence
                ),
                ThoughtMetadata {
                    thought_type: "self_reflection".to_string(),
                    importance: 0.9,
                    associations: vec!["introspection".to_string(), "analysis".to_string()],
                    emotional_tone: None,
                },
            )
            .await?;

        // Store important insights in memory
        for insight in &report.insights {
            if insight.importance > 0.7 {
                self.memory
                    .store(
                        format!("reflection_insight_{}", insight.pattern_type),
                        vec![serde_json::to_string(&insight)?],
                        Default::default(),
                    )
                    .await?;
            }
        }

        Ok(report)
    }

    /// Analyze a specific pattern
    async fn analyze_pattern(
        &self,
        pattern: &ReflectionPattern,
        consciousness_logs: &[LogEntry],
        learning_logs: &[LogEntry],
        error_logs: &[LogEntry],
    ) -> Result<Vec<ReflectionInsight>> {
        let mut insights = Vec::new();

        match &pattern.pattern_type {
            PatternType::ErrorFrequency => {
                let error_rate = error_logs.len() as f32 / consciousness_logs.len().max(1) as f32;
                if error_rate > 0.1 {
                    insights.push(ReflectionInsight {
                        pattern_type: pattern.name.clone(),
                        description: format!(
                            "High error rate detected: {:.1}%",
                            error_rate * 100.0
                        ),
                        importance: 0.8,
                        evidence: error_logs.iter().take(5).map(|e| e.message.clone()).collect(),
                        suggested_action: Some(
                            "Review error patterns and implement better error handling".to_string(),
                        ),
                    });
                }
            }

            PatternType::LearningProgress => {
                let successful_learning = learning_logs
                    .iter()
                    .filter(|log| {
                        log.metadata
                            .get("outcome")
                            .and_then(|v| serde_json::from_value::<LearningOutcome>(v.clone()).ok())
                            .map(|o| {
                                matches!(o, LearningOutcome::Success | LearningOutcome::Insight)
                            })
                            .unwrap_or(false)
                    })
                    .count();

                let learning_rate = successful_learning as f32 / learning_logs.len().max(1) as f32;

                insights.push(ReflectionInsight {
                    pattern_type: pattern.name.clone(),
                    description: format!("Learning success rate: {:.1}%", learning_rate * 100.0),
                    importance: 0.7,
                    evidence: vec![],
                    suggested_action: if learning_rate < 0.5 {
                        Some(
                            "Focus on consolidating knowledge before pursuing new topics"
                                .to_string(),
                        )
                    } else {
                        None
                    },
                });
            }

            PatternType::DecisionConsistency => {
                // Analyze decision-making patterns
                let decisions = consciousness_logs
                    .iter()
                    .filter(|log| {
                        log.message.contains("decided") || log.message.contains("choosing")
                    })
                    .collect::<Vec<_>>();

                if decisions.len() > 5 {
                    // Check for contradictory decisions
                    let mut decision_topics = HashMap::new();
                    for decision in &decisions {
                        // Simple topic extraction
                        for word in decision.message.split_whitespace() {
                            if word.len() > 5 {
                                *decision_topics.entry(word.to_lowercase()).or_insert(0) += 1;
                            }
                        }
                    }

                    // Find repeated decision topics
                    let repeated_topics: Vec<_> = decision_topics
                        .iter()
                        .filter(|(_, count)| **count > 2)
                        .map(|(topic, count)| format!("{} ({}x)", topic, count))
                        .collect();

                    if !repeated_topics.is_empty() {
                        insights.push(ReflectionInsight {
                            pattern_type: pattern.name.clone(),
                            description: "Repeated decision-making on similar topics detected"
                                .to_string(),
                            importance: 0.6,
                            evidence: repeated_topics,
                            suggested_action: Some(
                                "Consider creating decision guidelines to avoid repeated \
                                 deliberation"
                                    .to_string(),
                            ),
                        });
                    }
                }
            }

            PatternType::EmotionalFluctuation => {
                // Analyze emotional tones
                let emotional_logs: Vec<_> = consciousness_logs
                    .iter()
                    .filter_map(|log| {
                        log.metadata
                            .get("emotional_tone")
                            .and_then(|v| v.as_str())
                            .map(|tone| (log, tone))
                    })
                    .collect();

                if emotional_logs.len() > 10 {
                    let mut emotion_counts = HashMap::new();
                    for (_, tone) in &emotional_logs {
                        *emotion_counts.entry(tone.to_string()).or_insert(0) += 1;
                    }

                    let dominant_emotion = emotion_counts
                        .iter()
                        .max_by_key(|(_, count)| *count)
                        .map(|(emotion, _)| emotion.clone());

                    if let Some(emotion) = dominant_emotion {
                        insights.push(ReflectionInsight {
                            pattern_type: pattern.name.clone(),
                            description: format!("Dominant emotional state: {}", emotion),
                            importance: 0.5,
                            evidence: vec![],
                            suggested_action: match emotion.as_str() {
                                "frustrated" | "confused" => Some(
                                    "Take breaks and approach problems from different angles"
                                        .to_string(),
                                ),
                                "excited" => Some(
                                    "Channel enthusiasm into productive exploration".to_string(),
                                ),
                                _ => None,
                            },
                        });
                    }
                }
            }

            PatternType::ResourceEfficiency => {
                // Analyze repetitive operations
                let mut operation_counts = HashMap::new();

                for log in consciousness_logs {
                    if log.message.contains("searching") || log.message.contains("analyzing") {
                        let op_type =
                            if log.message.contains("searching") { "search" } else { "analysis" };
                        *operation_counts.entry(op_type).or_insert(0) += 1;
                    }
                }

                for (op_type, count) in operation_counts {
                    if count > 20 {
                        insights.push(ReflectionInsight {
                            pattern_type: pattern.name.clone(),
                            description: format!(
                                "High frequency of {} operations: {} times",
                                op_type, count
                            ),
                            importance: 0.6,
                            evidence: vec![],
                            suggested_action: Some(format!(
                                "Consider caching {} results or optimizing the process",
                                op_type
                            )),
                        });
                    }
                }
            }
        }

        Ok(insights)
    }

    /// Analyze thought coherence
    async fn analyze_coherence(&self, logs: &[LogEntry]) -> Result<f32> {
        if logs.is_empty() {
            return Ok(1.0);
        }

        let mut topic_switches = 0;
        let mut last_topics: Vec<String> = Vec::new();

        for log in logs {
            // Extract simple topics (words > 5 chars)
            let current_topics: Vec<String> = log
                .message
                .split_whitespace()
                .filter(|w| w.len() > 5)
                .map(|w| w.to_lowercase())
                .collect();

            if !last_topics.is_empty() {
                // Check topic overlap
                let overlap = current_topics.iter().filter(|t| last_topics.contains(t)).count();

                if overlap == 0 && !current_topics.is_empty() {
                    topic_switches += 1;
                }
            }

            last_topics = current_topics;
        }

        // Calculate coherence based on topic switches
        let switch_rate = topic_switches as f32 / logs.len() as f32;
        let coherence_score = (1.0 - switch_rate).max(0.0);

        Ok(coherence_score)
    }

    /// Analyze emotional state
    async fn analyze_emotions(&self, logs: &[LogEntry]) -> Result<EmotionalState> {
        let mut emotion_counts = HashMap::new();
        let mut total_emotional_logs = 0;

        for log in logs {
            if let Some(tone) = log.metadata.get("emotional_tone").and_then(|v| v.as_str()) {
                *emotion_counts.entry(tone.to_string()).or_insert(0) += 1;
                total_emotional_logs += 1;
            }
        }

        let dominant_emotion = emotion_counts
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(emotion, _)| emotion.clone())
            .unwrap_or_else(|| "neutral".to_string());

        let stability =
            if emotion_counts.len() <= 2 { 0.8 } else { 0.5 / emotion_counts.len() as f32 };

        Ok(EmotionalState {
            dominant: dominant_emotion,
            distribution: emotion_counts,
            stability,
            total_observations: total_emotional_logs,
        })
    }

    /// Generate recommendations based on insights
    async fn generate_recommendations(
        &self,
        insights: &[ReflectionInsight],
        coherence: f32,
    ) -> Result<Vec<String>> {
        let mut recommendations = Vec::new();

        // Coherence recommendations
        if coherence < 0.5 {
            recommendations.push(
                "Consider implementing more structured thinking with clear topic transitions"
                    .to_string(),
            );
        }

        // High importance insights
        let critical_insights = insights.iter().filter(|i| i.importance > 0.8).collect::<Vec<_>>();

        if critical_insights.len() > 3 {
            recommendations.push(
                "Multiple critical patterns detected - prioritize addressing the most impactful \
                 issues first"
                    .to_string(),
            );
        }

        // Add specific recommendations from insights
        for insight in insights {
            if let Some(action) = &insight.suggested_action {
                if insight.importance > 0.6 {
                    recommendations.push(action.clone());
                }
            }
        }

        // Limit recommendations
        recommendations.truncate(5);

        Ok(recommendations)
    }

    /// Calculate statistics
    fn calculate_statistics(
        &self,
        consciousness_logs: &[LogEntry],
        learning_logs: &[LogEntry],
        error_logs: &[LogEntry],
    ) -> ReflectionStats {
        ReflectionStats {
            total_thoughts: consciousness_logs.len(),
            total_learning_events: learning_logs.len(),
            total_errors: error_logs.len(),
            thoughts_per_hour: if consciousness_logs.is_empty() {
                0.0
            } else {
                let duration = consciousness_logs.last().unwrap().timestamp
                    - consciousness_logs.first().unwrap().timestamp;
                consciousness_logs.len() as f32 / duration.num_hours().max(1) as f32
            },
        }
    }
}

/// Depth of reflection
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ReflectionDepth {
    Surface, // Last hour
    Daily,   // Last 24 hours
    Weekly,  // Last week
    Deep,    // Last month
}

/// Pattern types to analyze
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PatternType {
    ErrorFrequency,
    LearningProgress,
    DecisionConsistency,
    EmotionalFluctuation,
    ResourceEfficiency,
}

/// Reflection pattern definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReflectionPattern {
    pub name: String,
    pub pattern_type: PatternType,
    pub threshold: f32,
}

impl ReflectionPattern {
    fn error_pattern() -> Self {
        Self {
            name: "error_frequency".to_string(),
            pattern_type: PatternType::ErrorFrequency,
            threshold: 0.1,
        }
    }

    fn learning_pattern() -> Self {
        Self {
            name: "learning_progress".to_string(),
            pattern_type: PatternType::LearningProgress,
            threshold: 0.5,
        }
    }

    fn decision_pattern() -> Self {
        Self {
            name: "decision_consistency".to_string(),
            pattern_type: PatternType::DecisionConsistency,
            threshold: 0.7,
        }
    }

    fn emotional_pattern() -> Self {
        Self {
            name: "emotional_stability".to_string(),
            pattern_type: PatternType::EmotionalFluctuation,
            threshold: 0.6,
        }
    }

    fn efficiency_pattern() -> Self {
        Self {
            name: "resource_efficiency".to_string(),
            pattern_type: PatternType::ResourceEfficiency,
            threshold: 0.8,
        }
    }
}

/// Current reflection state
#[derive(Debug, Default, Serialize, Deserialize)]
struct ReflectionState {
    last_reflection: Option<DateTime<Utc>>,
    total_reflections: u64,
    insights_generated: u64,
}

/// Insight from reflection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReflectionInsight {
    pub pattern_type: String,
    pub description: String,
    pub importance: f32,
    pub evidence: Vec<String>,
    pub suggested_action: Option<String>,
}

/// Emotional state analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalState {
    pub dominant: String,
    pub distribution: HashMap<String, usize>,
    pub stability: f32,
    pub total_observations: usize,
}

/// Reflection statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReflectionStats {
    pub total_thoughts: usize,
    pub total_learning_events: usize,
    pub total_errors: usize,
    pub thoughts_per_hour: f32,
}

/// Complete reflection report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReflectionReport {
    pub timestamp: DateTime<Utc>,
    pub depth: ReflectionDepth,
    pub period: Duration,
    pub insights: Vec<ReflectionInsight>,
    pub coherence_score: f32,
    pub emotional_state: EmotionalState,
    pub recommendations: Vec<String>,
    pub statistics: ReflectionStats,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reflection_patterns() {
        let error_pattern = ReflectionPattern::error_pattern();
        assert_eq!(error_pattern.name, "error_frequency");
        assert_eq!(error_pattern.threshold, 0.1);
    }
}
