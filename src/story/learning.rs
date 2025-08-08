//! Story-based learning and adaptation system

use super::types::*;
use super::engine::StoryEngine;
use super::templates::TemplateId;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, Mutex};
use tracing::{debug, info};

/// Story learning system that adapts based on patterns
pub struct StoryLearningSystem {
    /// Reference to story engine
    story_engine: Arc<StoryEngine>,
    
    /// Learned patterns
    patterns: Arc<RwLock<Vec<LearnedPattern>>>,
    
    /// Adaptation strategies
    strategies: Arc<RwLock<Vec<AdaptationStrategy>>>,
    
    /// Learning metrics
    metrics: Arc<Mutex<LearningMetrics>>,
    
    /// Configuration
    config: LearningConfig,
}

/// Learned pattern from stories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedPattern {
    pub id: uuid::Uuid,
    pub name: String,
    pub pattern_type: PatternType,
    pub occurrences: Vec<PatternOccurrence>,
    pub confidence: f32,
    pub success_rate: f32,
    pub learned_from: Vec<StoryId>,
    pub metadata: PatternMetadata,
}

/// Pattern type
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PatternType {
    /// Sequence of plot points that frequently occur together
    PlotSequence(Vec<PlotType>),
    
    /// Common resolution strategies for issues
    ResolutionStrategy {
        issue_type: String,
        resolution_steps: Vec<String>,
    },
    
    /// Task completion patterns
    TaskCompletion {
        task_type: String,
        typical_duration: chrono::Duration,
        common_blockers: Vec<String>,
    },
    
    /// Context patterns that lead to success
    ContextualSuccess {
        context_keys: Vec<String>,
        success_indicators: Vec<String>,
    },
    
    /// Collaboration patterns between agents
    CollaborationPattern {
        agent_types: Vec<String>,
        interaction_sequence: Vec<String>,
    },
    
    /// Custom pattern
    Custom(HashMap<String, serde_json::Value>),
}

/// Pattern occurrence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternOccurrence {
    pub story_id: StoryId,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub context: HashMap<String, String>,
    pub outcome: PatternOutcome,
}

/// Pattern outcome
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PatternOutcome {
    Success { metrics: HashMap<String, f32> },
    Failure { reason: String },
    Partial { completion_rate: f32 },
}

/// Pattern metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternMetadata {
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub last_updated: chrono::DateTime<chrono::Utc>,
    pub stability_score: f32,
    pub tags: Vec<String>,
}

/// Adaptation strategy based on learned patterns
#[derive(Debug, Clone)]
pub struct AdaptationStrategy {
    pub id: uuid::Uuid,
    pub name: String,
    pub trigger_pattern: PatternType,
    pub adaptations: Vec<Adaptation>,
    pub effectiveness: f32,
}

/// Specific adaptation
#[derive(Debug, Clone)]
pub enum Adaptation {
    /// Suggest alternative plot sequence
    SuggestAlternativePlot(Vec<PlotType>),
    
    /// Recommend template based on context
    RecommendTemplate(TemplateId),
    
    /// Adjust task priorities
    AdjustPriorities(HashMap<String, f32>),
    
    /// Modify context based on success patterns
    EnrichContext(HashMap<String, String>),
    
    /// Suggest agent for task
    SuggestAgent(String),
    
    /// Custom adaptation
    Custom(String, serde_json::Value),
}

/// Learning metrics
#[derive(Debug, Clone, Default)]
pub struct LearningMetrics {
    pub total_patterns_learned: usize,
    pub successful_adaptations: usize,
    pub failed_adaptations: usize,
    pub average_confidence: f32,
    pub learning_rate: f32,
    pub adaptation_effectiveness: f32,
}

/// Learning configuration
#[derive(Debug, Clone)]
pub struct LearningConfig {
    /// Minimum occurrences to consider a pattern
    pub min_pattern_occurrences: usize,
    
    /// Confidence threshold for pattern recognition
    pub confidence_threshold: f32,
    
    /// Learning rate for adaptation
    pub learning_rate: f32,
    
    /// Maximum patterns to maintain
    pub max_patterns: usize,
    
    /// Enable automatic adaptation
    pub auto_adapt: bool,
    
    /// Pattern decay rate (for forgetting old patterns)
    pub decay_rate: f32,
}

impl Default for LearningConfig {
    fn default() -> Self {
        Self {
            min_pattern_occurrences: 3,
            confidence_threshold: 0.7,
            learning_rate: 0.1,
            max_patterns: 1000,
            auto_adapt: true,
            decay_rate: 0.95,
        }
    }
}

impl StoryLearningSystem {
    /// Create new learning system
    pub fn new(story_engine: Arc<StoryEngine>, config: LearningConfig) -> Self {
        Self {
            story_engine,
            patterns: Arc::new(RwLock::new(Vec::new())),
            strategies: Arc::new(RwLock::new(Vec::new())),
            metrics: Arc::new(Mutex::new(LearningMetrics::default())),
            config,
        }
    }
    
    /// Analyze stories and learn patterns
    pub async fn analyze_and_learn(&self) -> Result<Vec<LearnedPattern>> {
        let mut new_patterns = Vec::new();
        
        // Analyze plot sequences
        let sequence_patterns = self.analyze_plot_sequences().await?;
        new_patterns.extend(sequence_patterns);
        
        // Analyze resolution strategies
        let resolution_patterns = self.analyze_resolution_strategies().await?;
        new_patterns.extend(resolution_patterns);
        
        // Analyze task completion patterns
        let task_patterns = self.analyze_task_patterns().await?;
        new_patterns.extend(task_patterns);
        
        // Analyze contextual success patterns
        let context_patterns = self.analyze_context_patterns().await?;
        new_patterns.extend(context_patterns);
        
        // Update patterns
        let mut patterns = self.patterns.write().await;
        for pattern in new_patterns.iter() {
            // Check if pattern already exists
            if let Some(existing) = patterns.iter_mut().find(|p| {
                matches!(&p.pattern_type, pt if pt == &pattern.pattern_type)
            }) {
                // Update existing pattern
                existing.occurrences.extend(pattern.occurrences.clone());
                existing.confidence = self.calculate_confidence(&existing.occurrences);
                existing.success_rate = self.calculate_success_rate(&existing.occurrences);
                existing.metadata.last_updated = chrono::Utc::now();
            } else {
                // Add new pattern
                patterns.push(pattern.clone());
            }
        }
        
        // Apply decay to old patterns
        self.apply_pattern_decay(&mut patterns).await;
        
        // Limit pattern count
        if patterns.len() > self.config.max_patterns {
            patterns.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
            patterns.truncate(self.config.max_patterns);
        }
        
        // Update metrics
        let mut metrics = self.metrics.lock().await;
        metrics.total_patterns_learned = patterns.len();
        metrics.average_confidence = patterns.iter()
            .map(|p| p.confidence)
            .sum::<f32>() / patterns.len() as f32;
        
        info!("Learned {} patterns from stories", new_patterns.len());
        
        Ok(new_patterns)
    }
    
    /// Generate adaptations based on current context
    pub async fn generate_adaptations(
        &self,
        context: &HashMap<String, String>,
    ) -> Result<Vec<AdaptationSuggestion>> {
        let patterns = self.patterns.read().await;
        let mut suggestions = Vec::new();
        
        // Find matching patterns
        for pattern in patterns.iter() {
            if pattern.confidence >= self.config.confidence_threshold {
                if let Some(adaptation) = self.create_adaptation_for_pattern(pattern, context).await? {
                    suggestions.push(adaptation);
                }
            }
        }
        
        // Sort by relevance
        suggestions.sort_by(|a, b| b.relevance.partial_cmp(&a.relevance).unwrap());
        
        Ok(suggestions)
    }
    
    /// Apply adaptation to a story
    pub async fn apply_adaptation(
        &self,
        story_id: StoryId,
        adaptation: &AdaptationSuggestion,
    ) -> Result<()> {
        match &adaptation.adaptation {
            Adaptation::SuggestAlternativePlot(plot_sequence) => {
                // Add suggested plot points
                for plot_type in plot_sequence {
                    self.story_engine.add_plot_point(
                        story_id,
                        plot_type.clone(),
                        vec![format!("learned:{}", adaptation.pattern_id)],
                    ).await?;
                }
            }
            
            Adaptation::RecommendTemplate(template_id) => {
                // Log template recommendation
                info!("Recommended template {} for story {}", template_id.0, story_id.0);
            }
            
            Adaptation::AdjustPriorities(priorities) => {
                // Update story metadata with priorities
                if let Some(mut story) = self.story_engine.stories.get_mut(&story_id) {
                    for (key, priority) in priorities {
                        story.metadata.custom_data.insert(
                            format!("priority_{}", key),
                            serde_json::json!(priority),
                        );
                    }
                }
            }
            
            Adaptation::EnrichContext(context_additions) => {
                // Add to story context
                // Note: Story doesn't have context field, would need to use metadata
                if let Some(mut story) = self.story_engine.stories.get_mut(&story_id) {
                    for (key, value) in context_additions {
                        story.metadata.custom_data.insert(
                            key.clone(),
                            serde_json::json!(value),
                        );
                    }
                }
            }
            
            Adaptation::SuggestAgent(agent_type) => {
                // Add agent suggestion to story
                self.story_engine.add_plot_point(
                    story_id,
                    PlotType::Discovery {
                        insight: format!("Suggested agent type: {}", agent_type),
                    },
                    vec!["agent_suggestion".to_string()],
                ).await?;
            }
            
            Adaptation::Custom(name, value) => {
                debug!("Applied custom adaptation: {} = {:?}", name, value);
            }
        }
        
        // Track adaptation
        let mut metrics = self.metrics.lock().await;
        metrics.successful_adaptations += 1;
        
        Ok(())
    }
    
    /// Get pattern statistics
    pub async fn get_pattern_statistics(&self) -> PatternStatistics {
        let patterns = self.patterns.read().await;
        let metrics = self.metrics.lock().await;
        
        let pattern_distribution = patterns.iter()
            .fold(HashMap::new(), |mut acc, p| {
                let type_name = match &p.pattern_type {
                    PatternType::PlotSequence(_) => "PlotSequence",
                    PatternType::ResolutionStrategy { .. } => "ResolutionStrategy",
                    PatternType::TaskCompletion { .. } => "TaskCompletion",
                    PatternType::ContextualSuccess { .. } => "ContextualSuccess",
                    PatternType::CollaborationPattern { .. } => "CollaborationPattern",
                    PatternType::Custom(_) => "Custom",
                };
                *acc.entry(type_name.to_string()).or_insert(0) += 1;
                acc
            });
        
        PatternStatistics {
            total_patterns: patterns.len(),
            average_confidence: metrics.average_confidence,
            pattern_distribution,
            most_successful_patterns: patterns.iter()
                .filter(|p| p.success_rate > 0.8)
                .take(5)
                .map(|p| (p.name.clone(), p.success_rate))
                .collect(),
            learning_effectiveness: metrics.adaptation_effectiveness,
        }
    }
    
    /// Train on historical data
    pub async fn train_on_history(&self) -> Result<()> {
        info!("Training on historical story data...");
        
        // Analyze all stories
        let learned_patterns = self.analyze_and_learn().await?;
        
        // Generate adaptation strategies
        let strategies = self.generate_strategies_from_patterns(&learned_patterns).await?;
        
        // Update strategies
        let mut strategy_store = self.strategies.write().await;
        strategy_store.extend(strategies);
        
        // Calculate effectiveness
        let effectiveness = self.calculate_overall_effectiveness().await?;
        
        let mut metrics = self.metrics.lock().await;
        metrics.adaptation_effectiveness = effectiveness;
        
        info!("Training complete. Effectiveness: {:.2}%", effectiveness * 100.0);
        
        Ok(())
    }
    
    // Helper methods
    
    async fn analyze_plot_sequences(&self) -> Result<Vec<LearnedPattern>> {
        let mut patterns = Vec::new();
        let mut sequence_map: HashMap<Vec<String>, Vec<PatternOccurrence>> = HashMap::new();
        
        // Analyze sequences in all stories
        for story_ref in self.story_engine.stories.iter() {
            let story = story_ref.value();
            
            // Extract plot sequences (sliding window of 3-5 plot points)
            for arc in &story.arcs {
                for window_size in 3..=5 {
                    for window in arc.plot_points.windows(window_size) {
                        let sequence_key: Vec<String> = window.iter()
                            .map(|p| format!("{:?}", p.plot_type))
                            .collect();
                    
                    let occurrence = PatternOccurrence {
                        story_id: story.id,
                        timestamp: chrono::Utc::now(),
                        context: HashMap::new(), // Story doesn't have context field
                        outcome: self.determine_outcome(story),
                    };
                    
                        sequence_map.entry(sequence_key)
                            .or_insert_with(Vec::new)
                            .push(occurrence);
                    }
                }
            }
        }
        
        // Create patterns from sequences
        for (sequence, occurrences) in sequence_map {
            if occurrences.len() >= self.config.min_pattern_occurrences {
                let pattern = LearnedPattern {
                    id: uuid::Uuid::new_v4(),
                    name: format!("Plot Sequence #{}", patterns.len() + 1),
                    pattern_type: PatternType::PlotSequence(vec![]), // Simplified
                    occurrences: occurrences.clone(),
                    confidence: self.calculate_confidence(&occurrences),
                    success_rate: self.calculate_success_rate(&occurrences),
                    learned_from: occurrences.iter().map(|o| o.story_id).collect(),
                    metadata: PatternMetadata {
                        created_at: chrono::Utc::now(),
                        last_updated: chrono::Utc::now(),
                        stability_score: 0.5,
                        tags: vec!["sequence".to_string()],
                    },
                };
                
                patterns.push(pattern);
            }
        }
        
        Ok(patterns)
    }
    
    async fn analyze_resolution_strategies(&self) -> Result<Vec<LearnedPattern>> {
        let mut patterns = Vec::new();
        let mut resolution_map: HashMap<String, Vec<(StoryId, Vec<String>)>> = HashMap::new();
        
        for story_ref in self.story_engine.stories.iter() {
            let story = story_ref.value();
            
            // Find issues and their resolutions
            let mut current_issue = None;
            let mut resolution_steps = Vec::new();
            
            for arc in &story.arcs {
                for plot_point in &arc.plot_points {
                    match &plot_point.plot_type {
                        PlotType::Issue { error, resolved } => {
                        if *resolved && current_issue.is_some() {
                            // Found resolution
                            resolution_map.entry(current_issue.unwrap())
                                .or_insert_with(Vec::new)
                                .push((story.id, resolution_steps.clone()));
                        }
                        current_issue = Some(error.clone());
                        resolution_steps.clear();
                    }
                    _ => {
                        if current_issue.is_some() {
                            resolution_steps.push(format!("{:?}", plot_point.plot_type));
                            }
                        }
                    }
                }
            }
        }
        
        // Create patterns from resolutions
        for (issue_type, resolutions) in resolution_map {
            if resolutions.len() >= self.config.min_pattern_occurrences {
                let occurrences: Vec<PatternOccurrence> = resolutions.iter()
                    .map(|(story_id, _)| PatternOccurrence {
                        story_id: *story_id,
                        timestamp: chrono::Utc::now(),
                        context: HashMap::new(),
                        outcome: PatternOutcome::Success {
                            metrics: HashMap::from([("resolution_time".to_string(), 1.0)]),
                        },
                    })
                    .collect();
                
                let pattern = LearnedPattern {
                    id: uuid::Uuid::new_v4(),
                    name: format!("Resolution for: {}", issue_type),
                    pattern_type: PatternType::ResolutionStrategy {
                        issue_type,
                        resolution_steps: resolutions[0].1.clone(),
                    },
                    occurrences,
                    confidence: 0.8,
                    success_rate: 0.9,
                    learned_from: resolutions.iter().map(|(id, _)| *id).collect(),
                    metadata: PatternMetadata {
                        created_at: chrono::Utc::now(),
                        last_updated: chrono::Utc::now(),
                        stability_score: 0.7,
                        tags: vec!["resolution".to_string()],
                    },
                };
                
                patterns.push(pattern);
            }
        }
        
        Ok(patterns)
    }
    
    async fn analyze_task_patterns(&self) -> Result<Vec<LearnedPattern>> {
        let mut patterns = Vec::new();
        let mut task_map: HashMap<String, Vec<(StoryId, chrono::Duration)>> = HashMap::new();
        
        for story_ref in self.story_engine.stories.iter() {
            let story = story_ref.value();
            
            // Analyze task completion times
            for arc in &story.arcs {
                for plot_point in &arc.plot_points {
                    if let PlotType::Task { description, completed } = &plot_point.plot_type {
                        if *completed {
                        let duration = chrono::Utc::now() - plot_point.timestamp;
                            task_map.entry(description.clone())
                                .or_insert_with(Vec::new)
                                .push((story.id, duration));
                        }
                    }
                }
            }
        }
        
        // Create patterns from task data
        for (task_type, completions) in task_map {
            if completions.len() >= self.config.min_pattern_occurrences {
                let avg_duration = completions.iter()
                    .map(|(_, d)| d.num_seconds())
                    .sum::<i64>() / completions.len() as i64;
                
                let pattern = LearnedPattern {
                    id: uuid::Uuid::new_v4(),
                    name: format!("Task Pattern: {}", task_type),
                    pattern_type: PatternType::TaskCompletion {
                        task_type: task_type.clone(),
                        typical_duration: chrono::Duration::seconds(avg_duration),
                        common_blockers: vec![],
                    },
                    occurrences: completions.iter()
                        .map(|(story_id, _)| PatternOccurrence {
                            story_id: *story_id,
                            timestamp: chrono::Utc::now(),
                            context: HashMap::new(),
                            outcome: PatternOutcome::Success {
                                metrics: HashMap::from([("duration".to_string(), avg_duration as f32)]),
                            },
                        })
                        .collect(),
                    confidence: 0.75,
                    success_rate: 1.0,
                    learned_from: completions.iter().map(|(id, _)| *id).collect(),
                    metadata: PatternMetadata {
                        created_at: chrono::Utc::now(),
                        last_updated: chrono::Utc::now(),
                        stability_score: 0.6,
                        tags: vec!["task".to_string()],
                    },
                };
                
                patterns.push(pattern);
            }
        }
        
        Ok(patterns)
    }
    
    async fn analyze_context_patterns(&self) -> Result<Vec<LearnedPattern>> {
        let mut patterns = Vec::new();
        let mut context_success_map: HashMap<Vec<String>, Vec<(StoryId, bool)>> = HashMap::new();
        
        for story_ref in self.story_engine.stories.iter() {
            let story = story_ref.value();
            
            // Get context keys from metadata custom_data instead
            let mut context_keys: Vec<String> = story.metadata.custom_data.keys().cloned().collect();
            context_keys.sort();
            
            // Determine if story was successful
            let is_successful = matches!(self.determine_outcome(story), PatternOutcome::Success { .. });
            
            context_success_map.entry(context_keys)
                .or_insert_with(Vec::new)
                .push((story.id, is_successful));
        }
        
        // Create patterns from successful contexts
        for (context_keys, outcomes) in context_success_map {
            let success_count = outcomes.iter().filter(|(_, success)| *success).count();
            let success_rate = success_count as f32 / outcomes.len() as f32;
            
            if outcomes.len() >= self.config.min_pattern_occurrences && success_rate > 0.7 {
                let pattern = LearnedPattern {
                    id: uuid::Uuid::new_v4(),
                    name: format!("Successful Context Pattern #{}", patterns.len() + 1),
                    pattern_type: PatternType::ContextualSuccess {
                        context_keys: context_keys.clone(),
                        success_indicators: vec![],
                    },
                    occurrences: outcomes.iter()
                        .map(|(story_id, success)| PatternOccurrence {
                            story_id: *story_id,
                            timestamp: chrono::Utc::now(),
                            context: HashMap::new(),
                            outcome: if *success {
                                PatternOutcome::Success {
                                    metrics: HashMap::from([("success".to_string(), 1.0)]),
                                }
                            } else {
                                PatternOutcome::Failure {
                                    reason: "Unknown".to_string(),
                                }
                            },
                        })
                        .collect(),
                    confidence: success_rate,
                    success_rate,
                    learned_from: outcomes.iter().map(|(id, _)| *id).collect(),
                    metadata: PatternMetadata {
                        created_at: chrono::Utc::now(),
                        last_updated: chrono::Utc::now(),
                        stability_score: 0.8,
                        tags: vec!["context".to_string()],
                    },
                };
                
                patterns.push(pattern);
            }
        }
        
        Ok(patterns)
    }
    
    fn determine_outcome(&self, story: &Story) -> PatternOutcome {
        let total_tasks = story.arcs.iter()
            .flat_map(|arc| &arc.plot_points)
            .filter(|p| matches!(&p.plot_type, PlotType::Task { .. }))
            .count();
        
        let completed_tasks = story.arcs.iter()
            .flat_map(|arc| &arc.plot_points)
            .filter(|p| matches!(&p.plot_type, PlotType::Task { completed: true, .. }))
            .count();
        
        let unresolved_issues = story.arcs.iter()
            .flat_map(|arc| &arc.plot_points)
            .filter(|p| matches!(&p.plot_type, PlotType::Issue { resolved: false, .. }))
            .count();
        
        if unresolved_issues > 0 {
            PatternOutcome::Failure {
                reason: format!("{} unresolved issues", unresolved_issues),
            }
        } else if total_tasks > 0 {
            let completion_rate = completed_tasks as f32 / total_tasks as f32;
            if completion_rate >= 1.0 {
                PatternOutcome::Success {
                    metrics: HashMap::from([
                        ("completion_rate".to_string(), 1.0),
                        ("tasks_completed".to_string(), completed_tasks as f32),
                    ]),
                }
            } else {
                PatternOutcome::Partial { completion_rate }
            }
        } else {
            PatternOutcome::Success {
                metrics: HashMap::from([("no_tasks".to_string(), 1.0)]),
            }
        }
    }
    
    fn calculate_confidence(&self, occurrences: &[PatternOccurrence]) -> f32 {
        let success_count = occurrences.iter()
            .filter(|o| matches!(&o.outcome, PatternOutcome::Success { .. }))
            .count();
        
        let base_confidence = success_count as f32 / occurrences.len() as f32;
        
        // Adjust for occurrence count
        let occurrence_factor = (occurrences.len() as f32 / 10.0).min(1.0);
        
        base_confidence * occurrence_factor
    }
    
    fn calculate_success_rate(&self, occurrences: &[PatternOccurrence]) -> f32 {
        let success_count = occurrences.iter()
            .filter(|o| matches!(&o.outcome, PatternOutcome::Success { .. }))
            .count();
        
        success_count as f32 / occurrences.len() as f32
    }
    
    async fn apply_pattern_decay(&self, patterns: &mut Vec<LearnedPattern>) {
        for pattern in patterns.iter_mut() {
            // Apply decay based on age
            let age = chrono::Utc::now() - pattern.metadata.last_updated;
            let decay_factor = self.config.decay_rate.powf(age.num_days() as f32 / 30.0);
            
            pattern.confidence *= decay_factor;
            pattern.metadata.stability_score *= decay_factor;
        }
        
        // Remove patterns below threshold
        patterns.retain(|p| p.confidence >= 0.1);
    }
    
    async fn create_adaptation_for_pattern(
        &self,
        pattern: &LearnedPattern,
        context: &HashMap<String, String>,
    ) -> Result<Option<AdaptationSuggestion>> {
        match &pattern.pattern_type {
            PatternType::PlotSequence(sequence) => {
                Ok(Some(AdaptationSuggestion {
                    pattern_id: pattern.id,
                    pattern_name: pattern.name.clone(),
                    adaptation: Adaptation::SuggestAlternativePlot(sequence.clone()),
                    relevance: pattern.confidence,
                    explanation: format!("Based on successful pattern with {:.0}% confidence", 
                        pattern.confidence * 100.0),
                }))
            }
            
            PatternType::ResolutionStrategy { issue_type, resolution_steps } => {
                if context.values().any(|v| v.contains(issue_type)) {
                    Ok(Some(AdaptationSuggestion {
                        pattern_id: pattern.id,
                        pattern_name: pattern.name.clone(),
                        adaptation: Adaptation::Custom(
                            "resolution_steps".to_string(),
                            serde_json::json!(resolution_steps),
                        ),
                        relevance: pattern.confidence * 1.5, // Boost relevance for matching issue
                        explanation: format!("Proven resolution strategy for {}", issue_type),
                    }))
                } else {
                    Ok(None)
                }
            }
            
            PatternType::ContextualSuccess { context_keys, .. } => {
                let missing_keys: Vec<_> = context_keys.iter()
                    .filter(|k| !context.contains_key(*k))
                    .cloned()
                    .collect();
                
                if !missing_keys.is_empty() {
                    let suggestions: HashMap<String, String> = missing_keys.iter()
                        .map(|k| (k.clone(), format!("suggested_value_for_{}", k)))
                        .collect();
                    
                    Ok(Some(AdaptationSuggestion {
                        pattern_id: pattern.id,
                        pattern_name: pattern.name.clone(),
                        adaptation: Adaptation::EnrichContext(suggestions),
                        relevance: pattern.success_rate,
                        explanation: format!("Adding context keys that lead to {:.0}% success rate", 
                            pattern.success_rate * 100.0),
                    }))
                } else {
                    Ok(None)
                }
            }
            
            _ => Ok(None),
        }
    }
    
    async fn generate_strategies_from_patterns(
        &self,
        patterns: &[LearnedPattern],
    ) -> Result<Vec<AdaptationStrategy>> {
        let mut strategies = Vec::new();
        
        for pattern in patterns {
            if pattern.confidence >= self.config.confidence_threshold {
                let strategy = AdaptationStrategy {
                    id: uuid::Uuid::new_v4(),
                    name: format!("Strategy for {}", pattern.name),
                    trigger_pattern: pattern.pattern_type.clone(),
                    adaptations: vec![], // Would be filled based on pattern type
                    effectiveness: pattern.success_rate,
                };
                
                strategies.push(strategy);
            }
        }
        
        Ok(strategies)
    }
    
    async fn calculate_overall_effectiveness(&self) -> Result<f32> {
        let metrics = self.metrics.lock().await;
        
        if metrics.successful_adaptations + metrics.failed_adaptations == 0 {
            return Ok(0.0);
        }
        
        let success_rate = metrics.successful_adaptations as f32 / 
            (metrics.successful_adaptations + metrics.failed_adaptations) as f32;
        
        Ok(success_rate)
    }
}

/// Adaptation suggestion
#[derive(Debug, Clone)]
pub struct AdaptationSuggestion {
    pub pattern_id: uuid::Uuid,
    pub pattern_name: String,
    pub adaptation: Adaptation,
    pub relevance: f32,
    pub explanation: String,
}

/// Pattern statistics
#[derive(Debug, Clone)]
pub struct PatternStatistics {
    pub total_patterns: usize,
    pub average_confidence: f32,
    pub pattern_distribution: HashMap<String, usize>,
    pub most_successful_patterns: Vec<(String, f32)>,
    pub learning_effectiveness: f32,
}