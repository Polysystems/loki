//! Integration tests for story-driven PR review

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use loki::cognitive::{
        StoryDrivenPrReview, StoryDrivenPrReviewConfig, 
        ReviewPattern, ReviewPatternType,
    };
    use loki::story::{StoryEngine, PlotPoint, PlotType};
    use loki::memory::{CognitiveMemory, MemoryConfig};
    use std::sync::Arc;
    use std::path::PathBuf;
    
    #[tokio::test]
    async fn test_pr_review_initialization() -> Result<()> {
        // Initialize memory
        let memory = Arc::new(CognitiveMemory::new(MemoryConfig::default()).await?);
        
        // Initialize story engine
        let story_engine = Arc::new(StoryEngine::new(memory.clone()).await?);
        
        // Create PR review config
        let config = StoryDrivenPrReviewConfig::default();
        
        // Initialize PR reviewer
        let pr_reviewer = StoryDrivenPrReview::new(
            config,
            story_engine,
            None,
            memory,
            None,
        ).await?;
        
        // If we get here without error, initialization succeeded
        assert!(true);
        
        Ok(())
    }
    
    #[tokio::test]
    async fn test_narrative_context() -> Result<()> {
        // Initialize components
        let memory = Arc::new(CognitiveMemory::new(MemoryConfig::default()).await?);
        let story_engine = Arc::new(StoryEngine::new(memory.clone()).await?);
        
        // Create a codebase story
        let story_id = story_engine
            .create_codebase_story(
                PathBuf::from("."),
                "rust".to_string()
            )
            .await?;
        
        // Add plot points
        story_engine
            .add_plot_point(
                story_id.clone(),
                PlotType::Goal {
                    objective: "Improve error handling".to_string(),
                },
                vec!["error-handling".to_string()]
            )
            .await?;
        
        // Verify story was created
        let story = story_engine.get_story(&story_id).await?;
        assert!(!story.segments.is_empty());
        
        Ok(())
    }
    
    #[tokio::test]
    async fn test_review_patterns() -> Result<()> {
        // Test that review patterns can be created and used
        let pattern = ReviewPattern {
            pattern_id: "test_pattern".to_string(),
            pattern_type: ReviewPatternType::CodeSmell,
            description: "Test pattern".to_string(),
            detection_rules: vec!["unwrap()".to_string()],
            severity: loki::tools::code_analysis::IssueSeverity::Warning,
            suggested_fix: Some("Use ? operator".to_string()),
            occurrences: 0,
            false_positive_rate: 0.1,
        };
        
        assert_eq!(pattern.pattern_id, "test_pattern");
        assert!(matches!(pattern.pattern_type, ReviewPatternType::CodeSmell));
        
        Ok(())
    }
}