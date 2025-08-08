//! Story-based context retrieval for agents

use super::types::*;
use super::engine::StoryEngine;
use super::context_chain::ContextSegment;
use anyhow::Result;
use std::sync::Arc;
use tracing::debug;

/// Context retrieval system for agents
pub struct StoryContextRetriever {
    /// Reference to story engine
    story_engine: Arc<StoryEngine>,
    
    /// Configuration
    config: ContextRetrievalConfig,
}

/// Configuration for context retrieval
#[derive(Debug, Clone)]
pub struct ContextRetrievalConfig {
    /// Maximum context tokens to retrieve
    pub max_tokens: usize,
    
    /// Maximum search depth for related segments
    pub max_search_depth: usize,
    
    /// Minimum relevance score for inclusion
    pub min_relevance_score: f32,
    
    /// Enable cross-story context retrieval
    pub enable_cross_story: bool,
    
    /// Prioritize recent context
    pub recency_weight: f32,
    
    /// Prioritize important context
    pub importance_weight: f32,
}

impl Default for ContextRetrievalConfig {
    fn default() -> Self {
        Self {
            max_tokens: 4096,
            max_search_depth: 5,
            min_relevance_score: 0.3,
            enable_cross_story: true,
            recency_weight: 0.3,
            importance_weight: 0.7,
        }
    }
}

/// Retrieved context package for agents
#[derive(Debug, Clone)]
pub struct RetrievedContext {
    /// Primary context segments
    pub primary_segments: Vec<ContextSegment>,
    
    /// Related segments from other stories
    pub related_segments: Vec<(StoryId, ContextSegment)>,
    
    /// Extracted tasks relevant to current context
    pub relevant_tasks: Vec<MappedTask>,
    
    /// Key insights and discoveries
    pub insights: Vec<String>,
    
    /// Active goals from stories
    pub active_goals: Vec<String>,
    
    /// Recent decisions for context
    pub recent_decisions: Vec<(String, String)>, // (question, choice)
    
    /// Total token count
    pub total_tokens: usize,
    
    /// Context quality score
    pub quality_score: f32,
}

impl StoryContextRetriever {
    /// Create a new context retriever
    pub fn new(
        story_engine: Arc<StoryEngine>,
        config: ContextRetrievalConfig,
    ) -> Self {
        Self {
            story_engine,
            config,
        }
    }
    
    /// Retrieve context for an agent
    pub async fn retrieve_context_for_agent(
        &self,
        agent_id: &str,
        query: Option<&str>,
    ) -> Result<RetrievedContext> {
        // Find agent's story
        let agent_stories = self.story_engine.get_stories_by_type(|st| {
            matches!(st, StoryType::Agent { agent_id: aid, .. } if aid == agent_id)
        });
        
        if agent_stories.is_empty() {
            // Create new agent story if none exists
            let story_id = self.story_engine.create_agent_story(
                agent_id.to_string(),
                "Context Retrieval Agent".to_string(),
            ).await?;
            
            return self.retrieve_context_for_story(story_id, query).await;
        }
        
        // Use the most recent agent story
        let agent_story = agent_stories
            .into_iter()
            .max_by_key(|s| s.updated_at)
            .unwrap();
        
        self.retrieve_context_for_story(agent_story.id, query).await
    }
    
    /// Retrieve context for a specific story
    pub async fn retrieve_context_for_story(
        &self,
        story_id: StoryId,
        query: Option<&str>,
    ) -> Result<RetrievedContext> {
        let story = self.story_engine
            .get_story(&story_id)
            .ok_or_else(|| anyhow::anyhow!("Story not found"))?;
        
        // Get primary context from story's chain
        let primary_segments = self.get_primary_segments(&story).await?;
        
        // Get related segments if cross-story is enabled
        let related_segments = if self.config.enable_cross_story {
            self.get_related_segments(&story, query).await?
        } else {
            vec![]
        };
        
        // Extract insights and goals
        let (insights, active_goals, recent_decisions) = 
            self.extract_key_information(&story).await?;
        
        // Get relevant tasks
        let task_map = self.story_engine.create_task_map(story_id).await?;
        let relevant_tasks = self.filter_relevant_tasks(task_map.tasks, query);
        
        // Calculate total tokens and quality
        let total_tokens = self.calculate_token_count(&primary_segments, &related_segments);
        let quality_score = self.calculate_quality_score(
            &primary_segments,
            &related_segments,
            &relevant_tasks,
        );
        
        debug!(
            "Retrieved context for story {}: {} primary segments, {} related, {} tasks",
            story_id.0,
            primary_segments.len(),
            related_segments.len(),
            relevant_tasks.len()
        );
        
        Ok(RetrievedContext {
            primary_segments,
            related_segments,
            relevant_tasks,
            insights,
            active_goals,
            recent_decisions,
            total_tokens,
            quality_score,
        })
    }
    
    /// Retrieve context for a task
    pub async fn retrieve_context_for_task(
        &self,
        task_description: &str,
    ) -> Result<RetrievedContext> {
        // Find stories with similar tasks
        let all_stories = self.story_engine.get_stories_by_type(|_| true);
        let mut relevant_contexts = Vec::new();
        
        for story in all_stories {
            let task_map = self.story_engine.create_task_map(story.id).await?;
            
            // Find similar tasks
            let similar_tasks: Vec<_> = task_map.tasks
                .into_iter()
                .filter(|t| self.is_task_similar(&t.description, task_description))
                .collect();
            
            if !similar_tasks.is_empty() {
                let context = self.retrieve_context_for_story(story.id, Some(task_description)).await?;
                relevant_contexts.push((story.id, context, similar_tasks));
            }
        }
        
        // Merge contexts
        self.merge_contexts(relevant_contexts)
    }
    
    /// Get primary segments from a story's context chain
    async fn get_primary_segments(&self, story: &Story) -> Result<Vec<ContextSegment>> {
        if let Some(chain) = self.story_engine.context_chains.get(&story.context_chain) {
            let segments = chain.segments.read().await;
            
            // Get recent segments up to token limit
            let mut selected_segments = Vec::new();
            let mut token_count = 0;
            
            for segment in segments.iter().rev() {
                if token_count + segment.tokens.len() > self.config.max_tokens {
                    break;
                }
                
                // Apply importance threshold
                if segment.importance >= self.config.min_relevance_score {
                    selected_segments.push(segment.clone());
                    token_count += segment.tokens.len();
                }
            }
            
            selected_segments.reverse();
            Ok(selected_segments)
        } else {
            Ok(vec![])
        }
    }
    
    /// Get related segments from other stories
    async fn get_related_segments(
        &self,
        story: &Story,
        query: Option<&str>,
    ) -> Result<Vec<(StoryId, ContextSegment)>> {
        let mut related = Vec::new();
        
        // Get related stories
        for related_story_id in &story.metadata.related_stories {
            if let Some(related_story) = self.story_engine.get_story(related_story_id) {
                if let Some(chain) = self.story_engine.context_chains.get(&related_story.context_chain) {
                    // Find relevant segments
                    let segments = chain.segments.read().await;
                    
                    for segment in segments.iter() {
                        if self.is_segment_relevant(segment, query) {
                            related.push((*related_story_id, segment.clone()));
                            
                            // Limit related segments
                            if related.len() >= 10 {
                                return Ok(related);
                            }
                        }
                    }
                }
            }
        }
        
        Ok(related)
    }
    
    /// Extract key information from story
    async fn extract_key_information(
        &self,
        story: &Story,
    ) -> Result<(Vec<String>, Vec<String>, Vec<(String, String)>)> {
        let mut insights = Vec::new();
        let mut goals = Vec::new();
        let mut decisions = Vec::new();
        
        for arc in &story.arcs {
            for plot_point in &arc.plot_points {
                match &plot_point.plot_type {
                    PlotType::Discovery { insight } => {
                        insights.push(insight.clone());
                    }
                    PlotType::Goal { objective } => {
                        goals.push(objective.clone());
                    }
                    PlotType::Decision { question, choice } => {
                        decisions.push((question.clone(), choice.clone()));
                    }
                    _ => {}
                }
            }
        }
        
        // Keep only recent items
        insights.truncate(5);
        goals.truncate(3);
        decisions.truncate(3);
        
        Ok((insights, goals, decisions))
    }
    
    /// Filter tasks relevant to query
    fn filter_relevant_tasks(
        &self,
        tasks: Vec<MappedTask>,
        query: Option<&str>,
    ) -> Vec<MappedTask> {
        if let Some(query) = query {
            let query_lower = query.to_lowercase();
            tasks
                .into_iter()
                .filter(|task| {
                    task.description.to_lowercase().contains(&query_lower) ||
                    task.story_context.to_lowercase().contains(&query_lower)
                })
                .take(10)
                .collect()
        } else {
            // Return incomplete tasks
            tasks
                .into_iter()
                .filter(|t| t.status != TaskStatus::Completed)
                .take(10)
                .collect()
        }
    }
    
    /// Check if a segment is relevant to query
    fn is_segment_relevant(&self, segment: &ContextSegment, query: Option<&str>) -> bool {
        if segment.importance < self.config.min_relevance_score {
            return false;
        }
        
        if let Some(query) = query {
            let query_lower = query.to_lowercase();
            segment.content.to_lowercase().contains(&query_lower) ||
            segment.tokens.iter().any(|t| t.to_lowercase().contains(&query_lower))
        } else {
            true
        }
    }
    
    /// Check if two task descriptions are similar
    fn is_task_similar(&self, task1: &str, task2: &str) -> bool {
        // Simple similarity check - in production use embeddings
        let task1_lower = task1.to_lowercase();
        let task2_lower = task2.to_lowercase();
        
        // Check for common keywords
        let keywords1: Vec<&str> = task1_lower.split_whitespace().collect();
        let keywords2: Vec<&str> = task2_lower.split_whitespace().collect();
        
        let common_keywords = keywords1
            .iter()
            .filter(|k| keywords2.contains(k) && k.len() > 3)
            .count();
        
        common_keywords >= 2
    }
    
    /// Merge multiple contexts
    fn merge_contexts(
        &self,
        contexts: Vec<(StoryId, RetrievedContext, Vec<MappedTask>)>,
    ) -> Result<RetrievedContext> {
        let mut merged = RetrievedContext {
            primary_segments: Vec::new(),
            related_segments: Vec::new(),
            relevant_tasks: Vec::new(),
            insights: Vec::new(),
            active_goals: Vec::new(),
            recent_decisions: Vec::new(),
            total_tokens: 0,
            quality_score: 0.0,
        };
        
        let mut token_count = 0;
        
        for (story_id, context, tasks) in contexts {
            // Add segments up to token limit
            for segment in context.primary_segments {
                if token_count + segment.tokens.len() <= self.config.max_tokens {
                    merged.related_segments.push((story_id, segment.clone()));
                    token_count += segment.tokens.len();
                }
            }
            
            // Merge other data
            merged.relevant_tasks.extend(tasks);
            merged.insights.extend(context.insights);
            merged.active_goals.extend(context.active_goals);
            merged.recent_decisions.extend(context.recent_decisions);
        }
        
        // Deduplicate
        merged.insights.sort();
        merged.insights.dedup();
        merged.active_goals.sort();
        merged.active_goals.dedup();
        
        merged.total_tokens = token_count;
        merged.quality_score = self.calculate_quality_score(
            &merged.primary_segments,
            &merged.related_segments,
            &merged.relevant_tasks,
        );
        
        Ok(merged)
    }
    
    /// Calculate token count
    fn calculate_token_count(
        &self,
        primary: &[ContextSegment],
        related: &[(StoryId, ContextSegment)],
    ) -> usize {
        let primary_tokens: usize = primary.iter().map(|s| s.tokens.len()).sum();
        let related_tokens: usize = related.iter().map(|(_, s)| s.tokens.len()).sum();
        primary_tokens + related_tokens
    }
    
    /// Calculate quality score
    fn calculate_quality_score(
        &self,
        primary: &[ContextSegment],
        related: &[(StoryId, ContextSegment)],
        tasks: &[MappedTask],
    ) -> f32 {
        let avg_importance = if primary.is_empty() {
            0.0
        } else {
            primary.iter().map(|s| s.importance).sum::<f32>() / primary.len() as f32
        };
        
        let coverage = (primary.len() + related.len()) as f32 / 20.0; // Normalize to 0-1
        let task_relevance = (tasks.len() as f32 / 10.0).min(1.0);
        
        // Weighted average
        avg_importance * self.config.importance_weight +
        coverage * (1.0 - self.config.importance_weight) * 0.5 +
        task_relevance * 0.2
    }
}