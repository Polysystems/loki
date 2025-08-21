//! Conversation indexing for efficient retrieval

use std::collections::{HashMap, HashSet};
use regex::Regex;
use anyhow::{Result, Context};
use tracing::warn;

/// Enhanced conversation context indexer
#[derive(Debug, Default)]
pub struct ConversationIndexer {
    /// Index of messages by topic
    topic_index: HashMap<String, Vec<usize>>,
    
    /// Index of messages by entity
    entity_index: HashMap<String, Vec<usize>>,
    
    /// Index of messages by timestamp
    time_index: Vec<(chrono::DateTime<chrono::Utc>, usize)>,
    
    /// Common topics and their keywords
    topic_patterns: HashMap<String, Vec<String>>,
    
    /// Entity extraction patterns
    entity_patterns: Vec<(Regex, String)>,
}

impl ConversationIndexer {
    /// Create a new indexer
    pub fn new() -> Self {
        let mut indexer = Self::default();
        if let Err(e) = indexer.initialize_patterns() {
            warn!("Failed to initialize some indexer patterns: {}", e);
        }
        indexer
    }
    
    /// Initialize topic and entity patterns
    fn initialize_patterns(&mut self) -> Result<()> {
        // Initialize topic patterns
        self.topic_patterns.insert("coding".to_string(), vec![
            "code".to_string(), "function".to_string(), "class".to_string(), 
            "implementation".to_string(), "algorithm".to_string(), "debug".to_string(),
            "error".to_string(), "bug".to_string(), "refactor".to_string()
        ]);
        
        self.topic_patterns.insert("ai".to_string(), vec![
            "model".to_string(), "neural".to_string(), "machine learning".to_string(),
            "gpt".to_string(), "claude".to_string(), "llm".to_string(), "agent".to_string(),
            "cognitive".to_string(), "consciousness".to_string()
        ]);
        
        self.topic_patterns.insert("database".to_string(), vec![
            "database".to_string(), "sql".to_string(), "query".to_string(),
            "table".to_string(), "postgres".to_string(), "mongodb".to_string(),
            "redis".to_string(), "cache".to_string()
        ]);
        
        self.topic_patterns.insert("api".to_string(), vec![
            "api".to_string(), "endpoint".to_string(), "rest".to_string(),
            "graphql".to_string(), "request".to_string(), "response".to_string(),
            "http".to_string(), "webhook".to_string()
        ]);
        
        self.topic_patterns.insert("devops".to_string(), vec![
            "deploy".to_string(), "docker".to_string(), "kubernetes".to_string(),
            "ci/cd".to_string(), "pipeline".to_string(), "container".to_string(),
            "aws".to_string(), "cloud".to_string()
        ]);
        
        // Initialize entity patterns
        self.entity_patterns = vec![
            // URLs
            (Regex::new(r"https?://[^\s]+")
                .context("Failed to compile URL regex")?, "url".to_string()),
            // File paths
            (Regex::new(r"(?:/[^/\s]+)+\.[a-zA-Z]+")
                .context("Failed to compile file path regex")?, "file".to_string()),
            // Email addresses
            (Regex::new(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
                .context("Failed to compile email regex")?, "email".to_string()),
            // IP addresses
            (Regex::new(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
                .context("Failed to compile IP address regex")?, "ip".to_string()),
            // GitHub repos
            (Regex::new(r"(?:github\.com/)?([a-zA-Z0-9-]+/[a-zA-Z0-9-_.]+)")
                .context("Failed to compile GitHub repo regex")?, "github".to_string()),
            // Package names (npm, cargo, etc)
            (Regex::new(r"(?:npm|cargo|pip|gem) (?:install|add) ([a-zA-Z0-9-_]+)")
                .context("Failed to compile package name regex")?, "package".to_string()),
            // Function/method names
            (Regex::new(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(")
                .context("Failed to compile function name regex")?, "function".to_string()),
            // Class names (PascalCase)
            (Regex::new(r"\b([A-Z][a-zA-Z0-9]+)\b")
                .context("Failed to compile class name regex")?, "class".to_string()),
            // Quoted strings
            (Regex::new(r#""([^"]+)"|'([^']+)'"#)
                .context("Failed to compile quoted string regex")?, "quoted".to_string()),
            // Version numbers
            (Regex::new(r"\bv?(\d+\.\d+(?:\.\d+)?)\b")
                .context("Failed to compile version number regex")?, "version".to_string()),
        ];
        Ok(())
    }
    
    /// Index a message
    pub fn index_message(&mut self, message_id: usize, content: &str, timestamp: chrono::DateTime<chrono::Utc>) {
        // Add to time index
        self.time_index.push((timestamp, message_id));
        
        // Extract and index topics
        let topics = self.extract_topics(content);
        for topic in topics {
            self.topic_index
                .entry(topic)
                .or_insert_with(Vec::new)
                .push(message_id);
        }
        
        // Extract and index entities
        let entities = self.extract_entities(content);
        for entity in entities {
            self.entity_index
                .entry(entity)
                .or_insert_with(Vec::new)
                .push(message_id);
        }
    }
    
    /// Extract topics from content
    fn extract_topics(&self, content: &str) -> HashSet<String> {
        let mut topics = HashSet::new();
        let content_lower = content.to_lowercase();
        
        for (topic, keywords) in &self.topic_patterns {
            for keyword in keywords {
                if content_lower.contains(keyword) {
                    topics.insert(topic.clone());
                    break; // One match is enough for this topic
                }
            }
        }
        
        // Also extract hashtag-style topics
        if let Ok(hashtag_re) = Regex::new(r"#([a-zA-Z0-9_]+)") {
            for cap in hashtag_re.captures_iter(content) {
                if let Some(tag) = cap.get(1) {
                    topics.insert(tag.as_str().to_lowercase());
                }
            }
        }
        
        topics
    }
    
    /// Extract entities from content
    fn extract_entities(&self, content: &str) -> HashSet<String> {
        let mut entities = HashSet::new();
        
        for (pattern, entity_type) in &self.entity_patterns {
            for cap in pattern.captures_iter(content) {
                if let Some(entity) = cap.get(1).or(cap.get(0)) {
                    let entity_value = entity.as_str();
                    // Store as "type:value" for better categorization
                    entities.insert(format!("{}:{}", entity_type, entity_value));
                    // Also store just the value for general search
                    entities.insert(entity_value.to_string());
                }
            }
        }
        
        entities
    }
    
    /// Search by topic
    pub fn search_by_topic(&self, topic: &str) -> Vec<usize> {
        self.topic_index.get(topic).cloned().unwrap_or_default()
    }
    
    /// Search by entity
    pub fn search_by_entity(&self, entity: &str) -> Vec<usize> {
        self.entity_index.get(entity).cloned().unwrap_or_default()
    }
    
    /// Search by time range
    pub fn search_by_time_range(
        &self, 
        start: chrono::DateTime<chrono::Utc>, 
        end: chrono::DateTime<chrono::Utc>
    ) -> Vec<usize> {
        self.time_index
            .iter()
            .filter(|(timestamp, _)| *timestamp >= start && *timestamp <= end)
            .map(|(_, id)| *id)
            .collect()
    }
    
    /// Get all topics
    pub fn get_all_topics(&self) -> Vec<(String, usize)> {
        let mut topics: Vec<_> = self.topic_index
            .iter()
            .map(|(topic, ids)| (topic.clone(), ids.len()))
            .collect();
        topics.sort_by(|a, b| b.1.cmp(&a.1)); // Sort by frequency
        topics
    }
    
    /// Get all entities of a specific type
    pub fn get_entities_by_type(&self, entity_type: &str) -> Vec<String> {
        let prefix = format!("{}:", entity_type);
        self.entity_index
            .keys()
            .filter(|k| k.starts_with(&prefix))
            .map(|k| k[prefix.len()..].to_string())
            .collect()
    }
    
    /// Get message statistics
    pub fn get_statistics(&self) -> IndexStatistics {
        IndexStatistics {
            total_messages: self.time_index.len(),
            total_topics: self.topic_index.len(),
            total_entities: self.entity_index.len(),
            most_common_topic: self.get_all_topics().first().map(|(t, _)| t.clone()),
            entity_types: self.get_entity_type_counts(),
        }
    }
    
    /// Get entity type counts
    fn get_entity_type_counts(&self) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for key in self.entity_index.keys() {
            if let Some(colon_pos) = key.find(':') {
                let entity_type = &key[..colon_pos];
                *counts.entry(entity_type.to_string()).or_insert(0) += 1;
            }
        }
        counts
    }
    
    /// Clear all indices
    pub fn clear(&mut self) {
        self.topic_index.clear();
        self.entity_index.clear();
        self.time_index.clear();
    }
}

/// Statistics about the indexed content
#[derive(Debug, Clone)]
pub struct IndexStatistics {
    pub total_messages: usize,
    pub total_topics: usize,
    pub total_entities: usize,
    pub most_common_topic: Option<String>,
    pub entity_types: HashMap<String, usize>,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_topic_extraction() {
        let indexer = ConversationIndexer::new();
        let content = "I'm working on a machine learning model using Python and TensorFlow";
        let topics = indexer.extract_topics(content);
        
        assert!(topics.contains("ai"));
        assert!(topics.contains("coding"));
    }
    
    #[test]
    fn test_entity_extraction() {
        let indexer = ConversationIndexer::new();
        let content = "Check out https://github.com/example/repo and email me at test@example.com";
        let entities = indexer.extract_entities(content);
        
        assert!(entities.iter().any(|e| e.contains("url:https://github.com/example/repo")));
        assert!(entities.iter().any(|e| e.contains("email:test@example.com")));
    }
}