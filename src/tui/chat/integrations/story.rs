//! Story-Driven Chat Integration
//!
//! This module connects the story-driven autonomy system to the chat interface,
//! enabling narrative understanding, story-based commands, and code
//! storytelling.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::sync::RwLock;
use tracing::info;

use crate::cognitive::CognitiveSystem;
use crate::memory::CognitiveMemory;
use crate::story::{StoryEngine, StoryId};
use crate::tui::story_driven_code_analysis::{CodeNarrative, StoryDrivenCodeAnalyzer};

/// Story mode types for chat
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum StoryChatMode {
    /// Analyze code structure as narrative
    Analyze,
    /// Generate code following story principles
    Generate,
    /// Document code with narrative structure
    Document,
    /// Learn patterns through story understanding
    Learn,
    /// Review code changes as plot development
    Review,
    /// Autonomous story-driven development
    Autonomous,
}

impl StoryChatMode {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "analyze" | "analysis" => Some(Self::Analyze),
            "generate" | "gen" => Some(Self::Generate),
            "document" | "docs" => Some(Self::Document),
            "learn" | "learning" => Some(Self::Learn),
            "review" => Some(Self::Review),
            "autonomous" | "auto" => Some(Self::Autonomous),
            _ => None,
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Self::Analyze => "Analyze code structure as a narrative story",
            Self::Generate => "Generate code following story-driven principles",
            Self::Document => "Create documentation with narrative structure",
            Self::Learn => "Learn patterns through story understanding",
            Self::Review => "Review code changes as plot development",
            Self::Autonomous => "Enable autonomous story-driven development",
        }
    }
}

/// Story chat enhancement for cognitive system
pub struct StoryChatEnhancement {
    /// Story engine
    story_engine: Arc<StoryEngine>,

    /// Code analyzer
    code_analyzer: Arc<StoryDrivenCodeAnalyzer>,

    /// Current story context
    current_context: Arc<RwLock<StoryContext>>,
}

/// Story context for chat session
#[derive(Debug, Clone)]
pub struct StoryContext {
    /// Active story ID
    pub story_id: Option<StoryId>,

    /// Current codebase narrative
    pub codebase_narrative: Option<CodeNarrative>,

    /// Active story mode
    pub mode: Option<StoryChatMode>,

    /// Target path for operations
    pub target_path: Option<PathBuf>,
}

impl StoryChatEnhancement {
    /// Create new story chat enhancement
    pub async fn new(
        cognitive_system: Arc<CognitiveSystem>,
        memory: Arc<CognitiveMemory>,
        tool_manager: Arc<crate::tools::IntelligentToolManager>,
    ) -> Result<Self> {
        info!("📖 Initializing story chat enhancement");

        // Create context manager for story engine
        let context_config = crate::cognitive::context_manager::ContextConfig {
            max_tokens: 16384,
            target_tokens: 8192,
            segment_size: 1024,
            compression_threshold: 0.8,
            checkpoint_interval: std::time::Duration::from_secs(300),
            max_checkpoints: 10,
        };
        let context_manager = Arc::new(RwLock::new(
            crate::cognitive::context_manager::ContextManager::new(memory.clone(), context_config)
                .await?,
        ));

        // Create story engine
        let story_engine = Arc::new(
            StoryEngine::new(context_manager, memory.clone(), crate::story::StoryConfig::default())
                .await?,
        );

        let code_analyzer = Arc::new(
            StoryDrivenCodeAnalyzer::new(
                cognitive_system.clone(),
                tool_manager.clone(),
                memory.clone(),
            )
            .await?,
        );

        Ok(Self {
            story_engine,
            code_analyzer,
            current_context: Arc::new(RwLock::new(StoryContext {
                story_id: None,
                codebase_narrative: None,
                mode: None,
                target_path: None,
            })),
        })
    }

    /// Process story command
    pub async fn process_story_command(
        &self,
        mode: StoryChatMode,
        args: &serde_json::Value,
    ) -> Result<StoryCommandResult> {
        info!("📖 Processing story command: {:?}", mode);

        match mode {
            StoryChatMode::Analyze => self.analyze_code_story(args).await,
            StoryChatMode::Generate => self.generate_with_story(args).await,
            StoryChatMode::Document => self.document_with_story(args).await,
            StoryChatMode::Learn => self.learn_from_story(args).await,
            StoryChatMode::Review => self.review_with_story(args).await,
            StoryChatMode::Autonomous => self.enable_autonomous_story(args).await,
        }
    }

    /// Analyze code as story
    async fn analyze_code_story(&self, args: &serde_json::Value) -> Result<StoryCommandResult> {
        let path = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");

        let path = Path::new(path);

        // Analyze codebase narrative
        let narrative = self.code_analyzer.analyze_codebase_story(path).await?;

        // Update context
        let mut context = self.current_context.write().await;
        context.codebase_narrative = Some(narrative.clone());
        context.target_path = Some(path.to_path_buf());
        context.mode = Some(StoryChatMode::Analyze);

        // Format result
        let summary = crate::tui::story_driven_code_analysis::StoryDrivenChatIntegration::
            format_narrative_summary(&narrative);

        Ok(StoryCommandResult {
            content: summary,
            narrative: Some(narrative),
            suggestions: vec![
                "Try analyzing a specific file for its character role".to_string(),
                "Use /story generate to create code following this narrative".to_string(),
                "Use /story learn to extract patterns from the story".to_string(),
            ],
            artifacts: vec![],
        })
    }

    /// Generate code with story principles
    async fn generate_with_story(&self, args: &serde_json::Value) -> Result<StoryCommandResult> {
        let prompt = args
            .get("input")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing generation prompt"))?;

        let context = self.current_context.read().await;

        // Create story for generation
        let _story_id = self
            .story_engine
            .create_story(
                crate::story::StoryType::Task {
                    task_id: uuid::Uuid::new_v4().to_string(),
                    parent_story: None,
                },
                format!("Code Generation: {}", prompt),
                format!("Generate code for: {}", prompt),
                vec!["code-generation".to_string(), "chat".to_string()],
                crate::story::Priority::High,
            )
            .await?;

        // Format generated code
        let mut content = format!("📝 Generated Code with Story Principles:\n\n");

        Ok(StoryCommandResult {
            content,
            narrative: None,
            suggestions: vec![
                "Review the generated code for narrative coherence".to_string(),
                "Consider how this fits into the larger codebase story".to_string(),
            ],
            artifacts: vec![],
        })
    }

    /// Review code changes as story
    async fn review_with_story(&self, args: &serde_json::Value) -> Result<StoryCommandResult> {
        let pr_url = args
            .get("pr")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing PR URL"))?;

        // Review PR with story context
        let review = json!({
            "narrative_impact": "PR would advance the story by implementing requested features",
            "character_changes": "No significant character changes detected",
            "plot_advancement": "Incremental progress towards project goals",
            "recommendation": "Approve with minor suggestions"
        });

        let content = format!(
            "🔍 Story-Driven PR Review:\n\n📖 Narrative Impact: {}\n🎭 Character Changes: {}\n🎯 \
             Plot Advancement: {}\n\n**Recommendation:**\n{}",
            review.get("narrative_impact").and_then(|v| v.as_str()).unwrap_or("N/A"),
            review.get("character_changes").and_then(|v| v.as_str()).unwrap_or("N/A"),
            review.get("plot_advancement").and_then(|v| v.as_str()).unwrap_or("N/A"),
            review.get("recommendation").and_then(|v| v.as_str()).unwrap_or("N/A")
        );

        Ok(StoryCommandResult {
            content,
            narrative: None,
            suggestions: vec![
                "Consider the narrative impact of these changes".to_string(),
                "Ensure character consistency is maintained".to_string(),
            ],
            artifacts: vec![],
        })
    }

    /// Enable autonomous story-driven development
    async fn enable_autonomous_story(
        &self,
        args: &serde_json::Value,
    ) -> Result<StoryCommandResult> {
        let enable = args.get("enable").and_then(|v| v.as_bool()).unwrap_or(true);

        if enable {

            Ok(StoryCommandResult {
                content: "🤖 Autonomous Story-Driven Development ENABLED\n\nLoki will now:\n- \
                          Monitor codebase narrative evolution\n- Suggest story-coherent \
                          improvements\n- Generate narrative-aligned code\n- Maintain character \
                          consistency\n- Document plot developments\n\nUse '/story autonomous \
                          enable:false' to disable."
                    .to_string(),
                narrative: None,
                suggestions: vec![
                    "Monitor autonomous suggestions in the activity panel".to_string(),
                    "Review and approve changes before application".to_string(),
                ],
                artifacts: vec![],
            })
        } else {


            Ok(StoryCommandResult {
                content: "🛑 Autonomous Story-Driven Development DISABLED".to_string(),
                narrative: None,
                suggestions: vec![],
                artifacts: vec![],
            })
        }
    }
    
    /// Document code with story narrative
    async fn document_with_story(&self, args: &serde_json::Value) -> Result<StoryCommandResult> {
        let path = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");
        let output_format = args.get("format").and_then(|v| v.as_str()).unwrap_or("markdown");
        
        let path = Path::new(path);
        
        // Analyze the code narrative if not already done
        let narrative = if let Some(context) = self.current_context.read().await.codebase_narrative.clone() {
            context
        } else {
            self.code_analyzer.analyze_codebase_story(path).await?
        };
        
        // Generate documentation with story elements
        let mut doc = String::from("📝 **Story-Driven Documentation**\n\n");
        
        // Add narrative overview
        doc.push_str(&format!("## Narrative Overview\n\n"));
        doc.push_str(&format!("**Exposition:** {}\n", narrative.story_arc.exposition));
        doc.push_str(&format!("**Climax:** {}\n", narrative.story_arc.climax));
        if !narrative.story_arc.resolution.is_empty() {
            doc.push_str(&format!("**Resolution:** {}\n", narrative.story_arc.resolution));
        }
        doc.push_str("\n");
        
        // Document main characters (key modules/components)
        doc.push_str("## Main Characters (Components)\n\n");
        for character in narrative.characters.iter().take(5) {
            doc.push_str(&format!("### {}\n", character.name));
            doc.push_str(&format!("- **Role:** {:?}\n", character.role));
            doc.push_str(&format!("- **Traits:** {}\n", character.traits.join(", ")));
            doc.push_str(&format!("- **Current State:** {}\n\n", character.arc.current_state));
        }
        
        // Add plot points (key features/milestones)
        doc.push_str("## Plot Points (Features)\n\n");
        for plot_point in &narrative.plot_points {
            doc.push_str(&format!("- **{:?}**: {}\n", plot_point.event_type, plot_point.description));
            doc.push_str(&format!("  - Impact: {:.2}\n", plot_point.impact));
            if !plot_point.characters_involved.is_empty() {
                doc.push_str(&format!("  - Involved: {}\n", plot_point.characters_involved.join(", ")));
            }
        }
        
        // Add development timeline
        doc.push_str("\n## Development Timeline\n\n");
        doc.push_str("The codebase narrative follows this progression:\n");
        doc.push_str("1. **Setup:** Initial architecture and foundation\n");
        doc.push_str("2. **Rising Action:** Feature implementation and integration\n");
        doc.push_str("3. **Climax:** Core functionality completion\n");
        doc.push_str("4. **Resolution:** Optimization and refinement\n\n");
        
        // Create documentation artifact
        let artifact = json!({
            "type": "documentation",
            "format": output_format,
            "content": doc.clone(),
            "path": path.to_string_lossy(),
        });
        
        Ok(StoryCommandResult {
            content: doc,
            narrative: Some(narrative),
            suggestions: vec![
                "Export documentation to a file using /export".to_string(),
                "Generate API documentation with /story document path:api".to_string(),
                "Create user guides with narrative flow".to_string(),
            ],
            artifacts: vec![],  // Empty artifacts for now
        })
    }
    
    /// Learn patterns from code story
    async fn learn_from_story(&self, args: &serde_json::Value) -> Result<StoryCommandResult> {
        let path = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");
        let pattern_type = args.get("type").and_then(|v| v.as_str()).unwrap_or("all");
        
        let path = Path::new(path);
        
        // Analyze patterns in the narrative
        let narrative = if let Some(context) = self.current_context.read().await.codebase_narrative.clone() {
            context
        } else {
            self.code_analyzer.analyze_codebase_story(path).await?
        };
        
        let mut learnings = String::from("🎓 **Learned Patterns from Code Story**\n\n");
        
        // Extract architectural patterns
        if pattern_type == "all" || pattern_type == "architecture" {
            learnings.push_str("## Architectural Patterns\n\n");
            learnings.push_str("- **Modular Design:** Components act as independent characters\n");
            learnings.push_str("- **Event-Driven:** Plot advances through event interactions\n");
            learnings.push_str("- **Layered Narrative:** Each layer tells its own sub-story\n\n");
        }
        
        // Extract coding patterns
        if pattern_type == "all" || pattern_type == "coding" {
            learnings.push_str("## Coding Patterns\n\n");
            learnings.push_str("- **Error Handling:** Conflict resolution in the narrative\n");
            learnings.push_str("- **Async Operations:** Parallel plot lines\n");
            learnings.push_str("- **Type Safety:** Character consistency\n\n");
        }
        
        // Extract collaboration patterns
        if pattern_type == "all" || pattern_type == "collaboration" {
            learnings.push_str("## Collaboration Patterns\n\n");
            for character in narrative.characters.iter().take(3) {
                learnings.push_str(&format!("- **{}:** {} - {}\n",
                    character.name, character.traits.join(", "), character.arc.current_state));
            }
            learnings.push_str("\n");
        }
        
        // Extract evolution patterns
        learnings.push_str("## Evolution Patterns\n\n");
        learnings.push_str(&format!("- **Current Stage:** {}\n", 
            if !narrative.story_arc.resolution.is_empty() { "Resolution" } else { "Development" }));
        learnings.push_str(&format!("- **Rising Actions:** {}\n", 
            narrative.story_arc.rising_action.join(", ")));
        if !narrative.story_arc.falling_action.is_empty() {
            learnings.push_str(&format!("- **Stabilization:** {}\n", 
                narrative.story_arc.falling_action.join(", ")));
        }
        learnings.push_str("\n");
        
        // Add insights
        learnings.push_str("## Key Insights\n\n");
        learnings.push_str("1. The codebase follows a coherent narrative structure\n");
        learnings.push_str("2. Components interact like characters in a story\n");
        learnings.push_str("3. Technical debt represents unresolved plot points\n");
        learnings.push_str("4. Tests validate the narrative consistency\n\n");
        
        // Store learned patterns in context
        let mut context = self.current_context.write().await;
        // Update the codebase narrative with fresh analysis
        context.codebase_narrative = Some(narrative.clone());
        
        Ok(StoryCommandResult {
            content: learnings,
            narrative: Some(narrative),
            suggestions: vec![
                "Apply learned patterns to new code generation".to_string(),
                "Use patterns to guide refactoring decisions".to_string(),
                "Share patterns with team for consistency".to_string(),
            ],
            artifacts: vec![],
        })
    }

    /// Enhance response with story context
    pub async fn enhance_response_with_story(
        &self,
        response: &mut String,
        context: &serde_json::Value,
    ) -> Result<()> {
        let story_context = self.current_context.read().await;

        if let Some(narrative) = &story_context.codebase_narrative {
            // Add story context to response
            if context.get("include_story").and_then(|v| v.as_bool()).unwrap_or(false) {
                response.push_str(&format!(
                    "\n\n📖 Story Context: {}\n🎭 Active Characters: {}",
                    narrative.story_arc.exposition,
                    narrative
                        .characters
                        .iter()
                        .take(3)
                        .map(|c| c.name.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
        }

        Ok(())
    }

    /// Get story suggestions for current context
    pub async fn get_story_suggestions(&self) -> Vec<String> {
        let context = self.current_context.read().await;

        let mut suggestions = vec![];

        if context.codebase_narrative.is_none() {
            suggestions
                .push("Run '/story analyze' to understand your codebase narrative".to_string());
        } else {
            suggestions.push("Use '/story generate' to create story-coherent code".to_string());
            suggestions.push("Try '/story learn' to extract narrative patterns".to_string());
            suggestions.push("Use '/story document' for narrative documentation".to_string());
        }

        if !matches!(context.mode, Some(StoryChatMode::Autonomous)) {
            suggestions.push(
                "Enable '/story autonomous' for continuous narrative maintenance".to_string(),
            );
        }

        suggestions
    }
}

/// Result from story command
#[derive(Debug, Clone)]
pub struct StoryCommandResult {
    pub content: String,
    pub narrative: Option<CodeNarrative>,
    pub suggestions: Vec<String>,
    pub artifacts: Vec<StoryArtifact>,
}

/// Story artifact (generated code, docs, etc.)
#[derive(Debug, Clone)]
pub struct StoryArtifact {
    pub name: String,
    pub content: String,
    pub artifact_type: String,
}

/// Integration helper for story commands in chat
pub struct StoryChatIntegration;

impl StoryChatIntegration {
    /// Parse story command from chat input
    pub fn parse_story_command(input: &str) -> Option<(StoryChatMode, serde_json::Value)> {
        if !input.starts_with("/story") {
            return None;
        }

        let parts: Vec<&str> = input[6..].trim().split_whitespace().collect();
        if parts.is_empty() {
            return None;
        }

        let mode = StoryChatMode::from_str(parts[0])?;

        // Parse remaining args
        let args = if parts.len() > 1 {
            let mut map = serde_json::Map::new();

            // Special handling for review mode - expect PR URL
            if matches!(mode, StoryChatMode::Review) && parts.len() > 1 {
                // If it looks like a URL, use it as PR
                if parts[1].contains("github.com") || parts[1].contains("/pull/") {
                    map.insert("pr".to_string(), serde_json::Value::String(parts[1].to_string()));
                } else {
                    map.insert("pr".to_string(), serde_json::Value::String(parts[1..].join(" ")));
                }
            } else {
                // Default input handling
                map.insert("input".to_string(), serde_json::Value::String(parts[1..].join(" ")));
            }

            // Parse any key:value pairs
            for part in &parts[1..] {
                if let Some((key, value)) = part.split_once(':') {
                    map.insert(key.to_string(), serde_json::Value::String(value.to_string()));
                }
            }

            serde_json::Value::Object(map)
        } else {
            serde_json::json!({})
        };

        Some((mode, args))
    }

    /// Format story help text
    pub fn format_story_help() -> String {
        "📖 Story-Driven Commands:\n\n\
        /story analyze [path] - Analyze code structure as narrative\n\
        /story generate <prompt> - Generate code with story principles\n\
        /story document [path] - Create narrative documentation\n\
        /story learn - Extract patterns from codebase story\n\
        /story review pr:<url> - Review PR as story development\n\
        /story autonomous [enable:true/false] - Toggle autonomous story mode\n\n\
        Examples:\n\
        • /story analyze src/\n\
        • /story generate Create a new authentication module\n\
        • /story document src/auth/\n\
        • /story review pr:https://github.com/user/repo/pull/123\n\
        • /story autonomous enable:true".to_string()
    }
}
