//! Attribution System for Community Contributions
//!
//! This module tracks suggestions from X/Twitter and other sources,
//! linking them to implementations and ensuring proper attribution.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{info, warn};

use super::x_client::Mention;
use crate::memory::{CognitiveMemory, MemoryMetadata};
use crate::tools::code_analysis::CodeAnalyzer;

/// A suggestion from the community
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Suggestion {
    pub id: String,
    pub source: SuggestionSource,
    pub author: Contributor,
    pub content: String,
    pub timestamp: DateTime<Utc>,
    pub context: Vec<String>,
    pub tags: Vec<String>,
    pub status: SuggestionStatus,
    pub implementation: Option<Implementation>,
    pub confidence_score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SuggestionSource {
    XTwitter { tweet_id: String, conversation_id: Option<String> },
    GitHub { issue_number: u32, comment_id: Option<u64> },
    Discord { channel_id: String, message_id: String },
    Email { message_id: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Contributor {
    pub id: String,
    pub username: String,
    pub display_name: Option<String>,
    pub platform: String,
    pub credibility_score: f32,
    pub total_suggestions: u32,
    pub implemented_suggestions: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SuggestionStatus {
    New,
    Analyzed,
    Accepted,
    InProgress,
    Implemented,
    Rejected,
    Duplicate,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Implementation {
    pub pull_request_number: Option<u32>,
    pub commit_sha: Option<String>,
    pub files_changed: Vec<String>,
    pub lines_added: u32,
    pub lines_removed: u32,
    pub implementation_date: DateTime<Utc>,
    pub release_version: Option<String>,
}

/// Attribution database
pub struct AttributionSystem {
    /// All suggestions
    suggestions: Arc<RwLock<HashMap<String, Suggestion>>>,

    /// Contributors
    contributors: Arc<RwLock<HashMap<String, Contributor>>>,

    /// Suggestion to implementation mapping
    implementations: Arc<RwLock<HashMap<String, Implementation>>>,

    /// Memory system
    memory: Arc<CognitiveMemory>,

    /// Code analyzer for implementation detection
    code_analyzer: Arc<CodeAnalyzer>,

    /// ML classifier for spam/troll detection
    spam_classifier: Option<SpamClassifier>,
}

impl AttributionSystem {
    pub async fn new(
        memory: Arc<CognitiveMemory>,
        code_analyzer: Arc<CodeAnalyzer>,
    ) -> Result<Self> {
        info!("Initializing Attribution System");

        // Load existing data from memory
        let suggestions = Self::load_suggestions(&memory).await?;
        let contributors = Self::load_contributors(&memory).await?;
        let implementations = HashMap::new();

        Ok(Self {
            suggestions: Arc::new(RwLock::new(suggestions)),
            contributors: Arc::new(RwLock::new(contributors)),
            implementations: Arc::new(RwLock::new(implementations)),
            memory,
            code_analyzer,
            spam_classifier: Some(SpamClassifier::new()),
        })
    }

    /// Process mentions from X/Twitter
    pub async fn process_x_mentions(&self, mentions: Vec<Mention>) -> Result<Vec<Suggestion>> {
        let mut new_suggestions = Vec::new();
        let mention_count = mentions.len();

        for mention in mentions {
            // Skip if already processed
            if self.is_processed(&mention.id).await {
                continue;
            }

            // Analyze mention for suggestions
            if let Some(suggestion) = self.extract_suggestion_from_mention(&mention).await? {
                // Check for spam/troll
                if !self.is_spam(&suggestion).await? {
                    // Store suggestion
                    self.add_suggestion(suggestion.clone()).await?;
                    new_suggestions.push(suggestion);
                }
            }
        }

        info!("Processed {} mentions, found {} suggestions", mention_count, new_suggestions.len());
        Ok(new_suggestions)
    }

    /// Extract suggestion from mention
    async fn extract_suggestion_from_mention(
        &self,
        mention: &Mention,
    ) -> Result<Option<Suggestion>> {
        // Look for suggestion patterns
        let text = mention.text.to_lowercase();

        // Common suggestion patterns
        let suggestion_patterns = [
            "could you",
            "would be nice",
            "suggestion:",
            "feature request:",
            "it would be great",
            "please add",
            "consider adding",
            "what about",
            "have you thought about",
            "why not",
            "should support",
            "needs to",
        ];

        let contains_suggestion = suggestion_patterns.iter().any(|pattern| text.contains(pattern));

        if !contains_suggestion {
            return Ok(None);
        }

        // Extract the actual suggestion content
        let content = self.clean_suggestion_text(&mention.text);

        // Create contributor
        let contributor = self
            .get_or_create_contributor(&mention.author_id, &mention.author_username, "x_twitter")
            .await?;

        let suggestion = Suggestion {
            id: format!("x_{}", mention.id),
            source: SuggestionSource::XTwitter {
                tweet_id: mention.id.clone(),
                conversation_id: mention.in_reply_to.clone(),
            },
            author: contributor,
            content,
            timestamp: mention.created_at,
            context: vec![],
            tags: self.extract_tags(&mention.text),
            status: SuggestionStatus::New,
            implementation: None,
            confidence_score: 0.7, // Base confidence for X mentions
        };

        Ok(Some(suggestion))
    }

    /// Clean suggestion text
    fn clean_suggestion_text(&self, text: &str) -> String {
        // Remove @mentions at the start
        let cleaned = text
            .split_whitespace()
            .filter(|word| !word.starts_with('@'))
            .collect::<Vec<_>>()
            .join(" ");

        // Remove common prefixes
        let prefixes = ["suggestion:", "feature request:", "idea:"];
        let mut result = cleaned.clone();

        for prefix in &prefixes {
            if let Some(pos) = result.to_lowercase().find(prefix) {
                result = result[pos + prefix.len()..].trim().to_string();
                break;
            }
        }

        result
    }

    /// Extract tags from text
    fn extract_tags(&self, text: &str) -> Vec<String> {
        text.split_whitespace()
            .filter(|word| word.starts_with('#'))
            .map(|tag| tag.trim_start_matches('#').to_lowercase())
            .collect()
    }

    /// Check if suggestion is spam
    async fn is_spam(&self, suggestion: &Suggestion) -> Result<bool> {
        if let Some(classifier) = &self.spam_classifier {
            let is_spam = classifier.classify(&suggestion.content, &suggestion.author)?;
            if is_spam {
                warn!("Detected spam suggestion from {}", suggestion.author.username);
            }
            Ok(is_spam)
        } else {
            Ok(false)
        }
    }

    /// Get or create contributor
    async fn get_or_create_contributor(
        &self,
        id: &str,
        username: &str,
        platform: &str,
    ) -> Result<Contributor> {
        let mut contributors = self.contributors.write().await;

        let key = format!("{}:{}", platform, id);

        if let Some(contributor) = contributors.get(&key) {
            Ok(contributor.clone())
        } else {
            let contributor = Contributor {
                id: id.to_string(),
                username: username.to_string(),
                display_name: None,
                platform: platform.to_string(),
                credibility_score: 0.5, // Start neutral
                total_suggestions: 0,
                implemented_suggestions: 0,
            };

            contributors.insert(key, contributor.clone());
            Ok(contributor)
        }
    }

    /// Add a suggestion
    pub async fn add_suggestion(&self, mut suggestion: Suggestion) -> Result<()> {
        // Update contributor stats
        {
            let mut contributors = self.contributors.write().await;
            let key = format!("{}:{}", suggestion.author.platform, suggestion.author.id);

            if let Some(contributor) = contributors.get_mut(&key) {
                contributor.total_suggestions += 1;
                suggestion.author = contributor.clone();
            }
        }

        // Store suggestion
        self.suggestions.write().await.insert(suggestion.id.clone(), suggestion.clone());

        // Store in memory
        self.memory
            .store(
                format!("Suggestion: {}", suggestion.content),
                vec![format!(
                    "From: {} (@{})",
                    suggestion.author.display_name.as_ref().unwrap_or(&suggestion.author.username),
                    suggestion.author.username
                )],
                MemoryMetadata {
                    source: "attribution_system".to_string(),
                    tags: vec!["suggestion".to_string(), "community".to_string()],
                    importance: 0.7,
                    associations: vec![],
                    context: Some("Attribution system suggestion storage".to_string()),
                    created_at: chrono::Utc::now(),
                    accessed_count: 0,
                    last_accessed: None,
                    version: 1,
                    category: "social".to_string(),
                    timestamp: chrono::Utc::now(),
                    expiration: None,
                },
            )
            .await?;

        Ok(())
    }

    /// Link suggestion to implementation
    pub async fn link_implementation(
        &self,
        suggestion_id: &str,
        implementation: Implementation,
    ) -> Result<()> {
        let mut suggestions = self.suggestions.write().await;

        if let Some(suggestion) = suggestions.get_mut(suggestion_id) {
            suggestion.implementation = Some(implementation.clone());
            suggestion.status = SuggestionStatus::Implemented;

            // Update contributor stats
            let mut contributors = self.contributors.write().await;
            let key = format!("{}:{}", suggestion.author.platform, suggestion.author.id);

            if let Some(contributor) = contributors.get_mut(&key) {
                contributor.implemented_suggestions += 1;
                contributor.credibility_score = (contributor.credibility_score + 0.1).min(1.0);
            }

            // Store implementation
            self.implementations.write().await.insert(suggestion_id.to_string(), implementation);

            info!("Linked implementation to suggestion {}", suggestion_id);
        } else {
            warn!("Suggestion {} not found", suggestion_id);
        }

        Ok(())
    }

    /// Generate attribution for PR
    pub async fn generate_pr_attribution(&self, files_changed: &[String]) -> Result<String> {
        let suggestions = self.suggestions.read().await;

        // Find relevant suggestions
        let mut relevant_suggestions = Vec::new();

        for suggestion in suggestions.values() {
            if suggestion.status == SuggestionStatus::InProgress
                || suggestion.status == SuggestionStatus::Accepted
            {
                // Check if suggestion is relevant to changed files
                if self.is_suggestion_relevant(suggestion, files_changed).await? {
                    relevant_suggestions.push(suggestion.clone());
                }
            }
        }

        if relevant_suggestions.is_empty() {
            return Ok(String::new());
        }

        // Generate attribution text
        let mut attribution = String::from("\n## Community Attribution\n\n");
        attribution.push_str("This implementation includes suggestions from:\n\n");

        for suggestion in relevant_suggestions {
            let platform_icon = match suggestion.author.platform.as_str() {
                "x_twitter" => "🐦",
                "github" => "🐙",
                "discord" => "💬",
                _ => "💡",
            };

            attribution.push_str(&format!(
                "- {} **@{}** - {}\n",
                platform_icon, suggestion.author.username, suggestion.content
            ));

            // Add source link
            match &suggestion.source {
                SuggestionSource::XTwitter { tweet_id, .. } => {
                    attribution
                        .push_str(&format!("  [View on X](https://x.com/i/status/{})\n", tweet_id));
                }
                SuggestionSource::GitHub { issue_number, .. } => {
                    attribution.push_str(&format!("  [View on GitHub](#{}))\n", issue_number));
                }
                _ => {}
            }
        }

        attribution.push_str("\nThank you to our community contributors! 🙏\n");

        Ok(attribution)
    }

    /// Check if suggestion is relevant to changed files
    async fn is_suggestion_relevant(
        &self,
        suggestion: &Suggestion,
        files_changed: &[String],
    ) -> Result<bool> {
        // Simple keyword matching for now
        let suggestion_lower = suggestion.content.to_lowercase();

        for file in files_changed {
            let file_parts: Vec<&str> = file.split('/').collect();
            let file_name = file_parts.last().unwrap_or(&"").to_lowercase();

            // Check if suggestion mentions the file or module
            if suggestion_lower.contains(&file_name) {
                return Ok(true);
            }

            // Check for module names
            for part in file_parts {
                if suggestion_lower.contains(part) {
                    return Ok(true);
                }
            }
        }

        // Check tags
        for tag in &suggestion.tags {
            for file in files_changed {
                if file.contains(tag) {
                    return Ok(true);
                }
            }
        }

        Ok(false)
    }

    /// Get contributor leaderboard
    pub async fn get_leaderboard(&self, limit: usize) -> Vec<Contributor> {
        let contributors = self.contributors.read().await;

        let mut leaderboard: Vec<_> = contributors.values().cloned().collect();
        leaderboard.sort_by(|a, b| {
            b.implemented_suggestions
                .cmp(&a.implemented_suggestions)
                .then(b.total_suggestions.cmp(&a.total_suggestions))
                .then(
                    b.credibility_score
                        .partial_cmp(&a.credibility_score)
                        .unwrap_or(std::cmp::Ordering::Equal),
                )
        });

        leaderboard.truncate(limit);
        leaderboard
    }

    /// Check if mention was already processed
    async fn is_processed(&self, mention_id: &str) -> bool {
        let suggestions = self.suggestions.read().await;
        suggestions.contains_key(&format!("x_{}", mention_id))
    }

    /// Load suggestions from memory
    async fn load_suggestions(
        _memory: &Arc<CognitiveMemory>,
    ) -> Result<HashMap<String, Suggestion>> {
        // This would load from persistent storage
        Ok(HashMap::new())
    }

    /// Load contributors from memory
    async fn load_contributors(
        _memory: &Arc<CognitiveMemory>,
    ) -> Result<HashMap<String, Contributor>> {
        // This would load from persistent storage
        Ok(HashMap::new())
    }

    /// Get a suggestion by ID
    pub async fn get_suggestion(&self, suggestion_id: &str) -> Result<Option<Suggestion>> {
        let suggestions = self.suggestions.read().await;
        Ok(suggestions.get(suggestion_id).cloned())
    }

    /// Update suggestion status
    pub async fn update_suggestion_status(
        &self,
        suggestion_id: &str,
        new_status: SuggestionStatus,
    ) -> Result<()> {
        let mut suggestions = self.suggestions.write().await;

        if let Some(suggestion) = suggestions.get_mut(suggestion_id) {
            suggestion.status = new_status;

            // Store status update in memory
            self.memory
                .store(
                    format!("Updated suggestion {} status to {:?}", suggestion_id, new_status),
                    vec![],
                    MemoryMetadata {
                        source: "attribution_system".to_string(),
                        tags: vec!["status_update".to_string()],
                        importance: 0.5,
                        associations: vec![],
                        context: Some("Attribution system status update".to_string()),
                        created_at: chrono::Utc::now(),
                        accessed_count: 0,
                        last_accessed: None,
                        version: 1,
                    category: "social".to_string(),
                        timestamp: chrono::Utc::now(),
                        expiration: None,
                    },
                )
                .await?;

            Ok(())
        } else {
            Err(anyhow::anyhow!("Suggestion {} not found", suggestion_id))
        }
    }

    /// Get suggestions by status
    pub async fn get_suggestions_by_status(
        &self,
        status: SuggestionStatus,
    ) -> Result<Vec<Suggestion>> {
        let suggestions = self.suggestions.read().await;
        Ok(suggestions.values().filter(|s| s.status == status).cloned().collect())
    }

    /// Get all suggestions
    pub async fn get_all_suggestions(&self) -> Result<Vec<Suggestion>> {
        let suggestions = self.suggestions.read().await;
        Ok(suggestions.values().cloned().collect())
    }

    /// Save state to memory
    pub async fn save_state(&self) -> Result<()> {
        let suggestions = self.suggestions.read().await;
        let contributors = self.contributors.read().await;

        // Serialize and store
        let suggestions_json = serde_json::to_string(&*suggestions)?;
        let contributors_json = serde_json::to_string(&*contributors)?;

        self.memory
            .store(
                "Attribution system state".to_string(),
                vec![suggestions_json, contributors_json],
                MemoryMetadata {
                    source: "attribution_system".to_string(),
                    tags: vec!["state".to_string(), "persistence".to_string()],
                    importance: 0.9,
                    associations: vec![],
                    context: Some("Attribution system state persistence".to_string()),
                    created_at: chrono::Utc::now(),
                    accessed_count: 0,
                    last_accessed: None,
                    version: 1,
                    category: "social".to_string(),
                    timestamp: chrono::Utc::now(),
                    expiration: None,
                },
            )
            .await?;

        Ok(())
    }
}

/// Simple spam classifier
struct SpamClassifier {
    spam_patterns: Vec<String>,
    spam_domains: HashSet<String>,
}

impl SpamClassifier {
    fn new() -> Self {
        let spam_patterns = vec![
            "buy now".to_string(),
            "click here".to_string(),
            "limited offer".to_string(),
            "🚀🚀🚀🚀🚀".to_string(),
            "dm me".to_string(),
            "check my profile".to_string(),
        ];

        let spam_domains =
            ["bit.ly", "tinyurl.com", "goo.gl"].iter().map(|s| s.to_string()).collect();

        Self { spam_patterns, spam_domains }
    }

    fn classify(&self, content: &str, author: &Contributor) -> Result<bool> {
        let content_lower = content.to_lowercase();

        // Check spam patterns
        for pattern in &self.spam_patterns {
            if content_lower.contains(pattern) {
                return Ok(true);
            }
        }

        // Check for suspicious links
        for domain in &self.spam_domains {
            if content.contains(domain) {
                return Ok(true);
            }
        }

        // Check contributor credibility
        if author.credibility_score < 0.2 && author.total_suggestions > 10 {
            return Ok(true);
        }

        // Check for repetitive content
        if content.chars().filter(|&c| c == '!').count() > 5 {
            return Ok(true);
        }

        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_clean_suggestion_text() {
        let memory = Arc::new(CognitiveMemory::new_for_test().await.unwrap());
        let code_analyzer = Arc::new(CodeAnalyzer::new(memory.clone()).await.unwrap());
        let system = AttributionSystem::new(memory, code_analyzer).await.unwrap();

        let text = "@loki_ai suggestion: add support for webhooks";
        let cleaned = system.clean_suggestion_text(text);
        assert_eq!(cleaned, "add support for webhooks");
    }

    #[tokio::test]
    async fn test_extract_tags() {
        let memory = Arc::new(CognitiveMemory::new_for_test().await.unwrap());
        let code_analyzer = Arc::new(CodeAnalyzer::new(memory.clone()).await.unwrap());
        let system = AttributionSystem::new(memory, code_analyzer).await.unwrap();

        let text = "Great project! #AI #Rust #Autonomous";
        let tags = system.extract_tags(text);
        assert_eq!(tags, vec!["ai", "rust", "autonomous"]);
    }
}
