//! Story Elements to Memory Learning Integration
//!
//! This module connects story-driven insights to the memory learning system,
//! allowing Loki to learn from narrative patterns and story understanding.

use std::sync::Arc;
use anyhow::{Result};
use tokio::sync::RwLock;
use tracing::{info, debug, warn};
use serde::{Serialize, Deserialize};
use chrono::Utc;

use crate::tui::{
    cognitive::integration::memory_learning::{CognitiveMemoryLearning, LearningResult},
    story_driven_code_analysis::{CodeNarrative, CodeCharacter, CodeTheme},
    chat::integrations::story::StoryCommandResult,
    cognitive::persistence::state::{PersistedInsight, LearningOutcome, LearningType},
};

/// Story-based learning insights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoryLearningInsight {
    pub insight_type: StoryInsightType,
    pub content: String,
    pub narrative_context: String,
    pub relevance: f64,
    pub patterns_identified: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StoryInsightType {
    CharacterEvolution,
    PlotPattern,
    ThemeRecurrence,
    ConflictResolution,
    NarrativeStructure,
}

/// Story memory learning integration
pub struct StoryMemoryIntegration {
    /// Memory learning system
    memory_learning: Arc<CognitiveMemoryLearning>,
    
    /// Story insights cache
    insights_cache: Arc<RwLock<Vec<StoryLearningInsight>>>,
}

impl StoryMemoryIntegration {
    /// Create new story memory integration
    pub fn new(memory_learning: Arc<CognitiveMemoryLearning>) -> Self {
        Self {
            memory_learning,
            insights_cache: Arc::new(RwLock::new(Vec::new())),
        }
    }
    
    /// Learn from code narrative
    pub async fn learn_from_narrative(
        &self,
        narrative: &CodeNarrative,
        session_id: &str,
    ) -> Result<LearningResult> {
        info!("📖 Learning from code narrative");
        
        let mut insights = Vec::new();
        let mut learnings = Vec::new();
        
        // Learn from character patterns
        for character in &narrative.characters {
            let character_insights = self.extract_character_insights(character).await?;
            insights.extend(character_insights);
        }
        
        // Learn from themes
        for theme in &narrative.themes {
            let theme_learning = self.extract_theme_patterns(theme).await?;
            learnings.push(theme_learning);
        }
        
        // Learn from story arc
        let arc_insights = self.extract_arc_patterns(&narrative.story_arc).await?;
        insights.extend(arc_insights);
        
        // Learn from relationships
        let relationship_patterns = self.extract_relationship_patterns(&narrative.relationships).await?;
        learnings.extend(relationship_patterns);
        
        // Store insights in memory
        let mut result = LearningResult::default();
        
        for insight in &insights {
            if insight.relevance > 0.7 {
                let persisted = PersistedInsight {
                    timestamp: Utc::now(),
                    content: insight.content.clone(),
                    category: format!("story_{:?}", insight.insight_type),
                    relevance: insight.relevance,
                    source_modalities: vec!["narrative".to_string()],
                };
                
                self.memory_learning.learn_from_persisted(
                    vec![persisted],
                    vec![],
                    session_id,
                ).await?;
                
                result.insights_stored += 1;
            }
        }
        
        // Store learnings
        for learning in &learnings {
            self.memory_learning.learn_from_persisted(
                vec![],
                vec![learning.clone()],
                session_id,
            ).await?;
            
            result.concepts_learned += 1;
        }
        
        // Cache insights
        self.insights_cache.write().await.extend(insights);
        
        Ok(result)
    }
    
    /// Learn from story command results
    pub async fn learn_from_story_command(
        &self,
        result: &StoryCommandResult,
        session_id: &str,
    ) -> Result<()> {
        info!("📚 Learning from story command execution");
        
        // Learn from narrative if present
        if let Some(narrative) = &result.narrative {
            self.learn_from_narrative(narrative, session_id).await?;
        }
        
        // Extract insights from content
        let content_insights = self.extract_content_insights(&result.content).await?;
        
        for insight in content_insights {
            if insight.relevance > 0.6 {
                let memory_data = serde_json::json!({
                    "type": "story_command_insight",
                    "content": insight.content,
                    "narrative_context": insight.narrative_context,
                    "patterns": insight.patterns_identified,
                    "timestamp": Utc::now(),
                });
                
                // Store the memory data through the memory learning system
                if let Err(e) = self.memory_learning.store_insight(memory_data.clone()).await {
                    warn!("Failed to store story insight: {}", e);
                } else {
                    debug!("Story insight recorded successfully: {:?}", memory_data);
                }
            }
        }
        
        Ok(())
    }
    
    /// Extract character insights
    async fn extract_character_insights(
        &self,
        character: &CodeCharacter,
    ) -> Result<Vec<StoryLearningInsight>> {
        let mut insights = Vec::new();
        
        // Learn from character role patterns
        insights.push(StoryLearningInsight {
            insight_type: StoryInsightType::CharacterEvolution,
            content: format!(
                "Component '{}' plays {} role with traits: {}",
                character.name,
                format!("{:?}", character.role),
                character.traits.join(", ")
            ),
            narrative_context: character.arc.current_state.clone(),
            relevance: 0.8,
            patterns_identified: character.traits.clone(),
        });
        
        // Learn from conflicts
        if !character.conflicts.is_empty() {
            insights.push(StoryLearningInsight {
                insight_type: StoryInsightType::ConflictResolution,
                content: format!(
                    "Component '{}' has conflicts: {}",
                    character.name,
                    character.conflicts.join(", ")
                ),
                narrative_context: "Identifying technical debt and issues".to_string(),
                relevance: 0.9,
                patterns_identified: character.conflicts.clone(),
            });
        }
        
        Ok(insights)
    }
    
    /// Extract theme patterns
    async fn extract_theme_patterns(&self, theme: &CodeTheme) -> Result<LearningOutcome> {
        Ok(LearningOutcome {
            timestamp: Utc::now(),
            learning_type: LearningType::PatternRecognition,
            content: format!(
                "Theme '{}' ({:?}) appears with {:.0}% prevalence",
                theme.name,
                theme.theme_type,
                theme.prevalence * 100.0
            ),
            confidence: theme.prevalence,
            applied: false,
        })
    }
    
    /// Extract story arc patterns
    async fn extract_arc_patterns(
        &self,
        arc: &crate::tui::story_driven_code_analysis::StoryArc,
    ) -> Result<Vec<StoryLearningInsight>> {
        let mut insights = Vec::new();
        
        insights.push(StoryLearningInsight {
            insight_type: StoryInsightType::NarrativeStructure,
            content: format!(
                "Codebase follows '{}' genre with climax: {}",
                arc.genre,
                arc.climax
            ),
            narrative_context: arc.resolution.clone(),
            relevance: 0.85,
            patterns_identified: vec![arc.genre.clone()],
        });
        
        Ok(insights)
    }
    
    /// Extract relationship patterns
    async fn extract_relationship_patterns(
        &self,
        relationships: &[crate::tui::story_driven_code_analysis::CharacterRelationship],
    ) -> Result<Vec<LearningOutcome>> {
        let mut learnings = Vec::new();
        
        // Group relationships by type
        let mut relationship_counts = std::collections::HashMap::new();
        for rel in relationships {
            *relationship_counts.entry(format!("{:?}", rel.relationship_type))
                .or_insert(0) += 1;
        }
        
        for (rel_type, count) in relationship_counts {
            learnings.push(LearningOutcome {
                timestamp: Utc::now(),
                learning_type: LearningType::PatternRecognition,
                content: format!(
                    "Found {} relationships of type {}",
                    count,
                    rel_type
                ),
                confidence: 0.9,
                applied: false,
            });
        }
        
        Ok(learnings)
    }
    
    /// Extract insights from content
    async fn extract_content_insights(&self, content: &str) -> Result<Vec<StoryLearningInsight>> {
        let mut insights = Vec::new();
        
        // Simple pattern extraction (would be more sophisticated)
        if content.contains("Story Arc") || content.contains("narrative") {
            insights.push(StoryLearningInsight {
                insight_type: StoryInsightType::NarrativeStructure,
                content: "Narrative structure identified in analysis".to_string(),
                narrative_context: content[..content.len().min(200)].to_string(),
                relevance: 0.7,
                patterns_identified: vec!["narrative".to_string()],
            });
        }
        
        Ok(insights)
    }
    
    /// Get learned story patterns
    pub async fn get_learned_patterns(&self) -> Vec<StoryLearningInsight> {
        self.insights_cache.read().await.clone()
    }
    
    /// Apply story learnings to improve analysis
    pub async fn apply_learnings_to_analysis(
        &self,
        target: &str,
    ) -> Result<Vec<String>> {
        let insights = self.insights_cache.read().await;
        let mut suggestions = Vec::new();
        
        for insight in insights.iter() {
            match insight.insight_type {
                StoryInsightType::CharacterEvolution => {
                    suggestions.push(format!(
                        "Consider character evolution pattern: {}",
                        insight.content
                    ));
                }
                StoryInsightType::ConflictResolution => {
                    suggestions.push(format!(
                        "Apply conflict resolution pattern: {}",
                        insight.content
                    ));
                }
                _ => {}
            }
        }
        
        Ok(suggestions)
    }
}

/// Integration helper for cognitive chat
pub struct StoryMemoryChatIntegration;

impl StoryMemoryChatIntegration {
    /// Create integration between story and memory systems
    pub async fn integrate_systems(
        memory_learning: Arc<CognitiveMemoryLearning>,
        story_enhancement: &mut crate::tui::chat::integrations::story::StoryChatEnhancement,
    ) -> Result<Arc<StoryMemoryIntegration>> {
        let integration = Arc::new(StoryMemoryIntegration::new(memory_learning));
        
        // Store reference in story enhancement for automatic learning
        // This would require adding a field to StoryChatEnhancement
        
        info!("🔗 Story-memory integration established");
        Ok(integration)
    }
    
    /// Format learned patterns for chat display
    pub fn format_learned_patterns(patterns: &[StoryLearningInsight]) -> String {
        let mut output = "🧠 Learned Story Patterns:\n\n".to_string();
        
        for (i, pattern) in patterns.iter().enumerate().take(5) {
            output.push_str(&format!(
                "{}. {} (Relevance: {:.0}%)\n   {}
\n",
                i + 1,
                pattern.content,
                pattern.relevance * 100.0,
                pattern.patterns_identified.join(", ")
            ));
        }
        
        output
    }
}