//! Search engine implementation

use std::sync::Arc;
use anyhow::Result;
use chrono::Utc;
use regex::Regex;
use tokio::sync::RwLock;

use super::{ChatSearchFilters, ChatSearchResult};
use super::filters::MessageTypeFilter;
use crate::tui::chat::state::ChatState;
use crate::tui::run::AssistantResponseType;

/// Chat search engine
pub struct SearchEngine {
    /// Maximum results to return
    max_results: usize,
    
    /// Reference to chat state
    chat_state: Option<Arc<RwLock<ChatState>>>,
    
    /// Compiled regex cache
    regex_cache: std::collections::HashMap<String, Regex>,
}

impl Default for SearchEngine {
    fn default() -> Self {
        Self {
            max_results: 100,
            chat_state: None,
            regex_cache: std::collections::HashMap::new(),
        }
    }
}

impl SearchEngine {
    /// Create a new search engine
    pub fn new(max_results: usize) -> Self {
        Self {
            max_results,
            chat_state: None,
            regex_cache: std::collections::HashMap::new(),
        }
    }
    
    /// Create with chat state reference
    pub fn with_chat_state(max_results: usize, chat_state: Arc<RwLock<ChatState>>) -> Self {
        Self {
            max_results,
            chat_state: Some(chat_state),
            regex_cache: std::collections::HashMap::new(),
        }
    }
    
    /// Search messages with filters
    pub async fn search(&mut self, filters: &ChatSearchFilters) -> Result<Vec<ChatSearchResult>> {
        let mut results = Vec::new();
        
        // If no chat state, return empty results
        let chat_state = match &self.chat_state {
            Some(state) => state,
            None => return Ok(results),
        };
        
        let state = chat_state.read().await;
        
        // Get search pattern
        let pattern = if let Some(query) = &filters.query {
            if filters.use_regex {
                // Compile regex if not cached
                if !self.regex_cache.contains_key(query) {
                    let regex = Regex::new(query)?;
                    self.regex_cache.insert(query.clone(), regex);
                }
                Some(self.regex_cache.get(query).unwrap())
            } else {
                None
            }
        } else {
            None
        };
        
        // Search through all messages
        for (msg_idx, message) in state.messages.iter().enumerate() {
            if !self.message_matches_filters(message, filters, pattern) {
                continue;
            }
            
            // Extract message content
            let (content, author) = match message {
                AssistantResponseType::Message { message, .. } => {
                    (message.clone(), "Assistant".to_string())
                }
                AssistantResponseType::UserMessage { content, .. } => {
                    (content.clone(), "User".to_string())
                }
                AssistantResponseType::SystemMessage { content, .. } => {
                    if !filters.include_system {
                        continue;
                    }
                    (content.clone(), "System".to_string())
                }
                AssistantResponseType::ToolExecution { tool_name, output, .. } => {
                    (format!("Tool: {}\nOutput: {}", tool_name, output), "Tool".to_string())
                }
                AssistantResponseType::ThinkingMessage { .. } => {
                    continue; // Skip thinking messages
                }
                _ => continue,
            };
            
            // Calculate relevance score
            let score = self.calculate_relevance_score(&content, filters);
            
            // Get context
            let context_before = self.get_context_before(&state.messages, msg_idx, 2);
            let context_after = self.get_context_after(&state.messages, msg_idx, 2);
            
            // Create snippet
            let snippet = self.create_snippet(&content, filters.query.as_deref());
            
            results.push(ChatSearchResult {
                chat_id: state.id.parse::<usize>().unwrap_or(0),
                message_index: msg_idx,
                snippet,
                full_content: content.to_string(),
                author,
                timestamp: Utc::now(), // In real implementation, messages would have timestamps
                score,
                context_before,
                context_after,
            });
        }
        
        // Sort by relevance score
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        
        // Limit results
        results.truncate(self.max_results);
        
        Ok(results)
    }
    
    /// Check if message matches filters
    fn message_matches_filters(
        &self,
        message: &AssistantResponseType,
        filters: &ChatSearchFilters,
        pattern: Option<&Regex>,
    ) -> bool {
        // Check message type filter
        if let Some(type_filter) = &filters.message_type {
            match (type_filter, message) {
                (MessageTypeFilter::UserMessages, AssistantResponseType::UserMessage { .. }) => {},
                (MessageTypeFilter::AssistantMessages, AssistantResponseType::Message { .. }) => {},
                (MessageTypeFilter::SystemMessages, AssistantResponseType::SystemMessage { .. }) => {},
                (MessageTypeFilter::ToolExecutions, AssistantResponseType::ToolExecution { .. }) => {},
                (MessageTypeFilter::All, _) => {},
                _ => return false,
            }
        }
        
        // Check text query
        if let Some(query) = &filters.query {
            let content = self.extract_content(message);
            
            if let Some(regex) = pattern {
                if !regex.is_match(&content) {
                    return false;
                }
            } else {
                let content_to_search = if filters.case_sensitive {
                    content
                } else {
                    content.to_lowercase()
                };
                
                let query_to_search = if filters.case_sensitive {
                    query.clone()
                } else {
                    query.to_lowercase()
                };
                
                if !content_to_search.contains(&query_to_search) {
                    return false;
                }
            }
        }
        
        true
    }
    
    /// Extract content from message
    fn extract_content(&self, message: &AssistantResponseType) -> String {
        match message {
            AssistantResponseType::Message { message, .. } => message.clone(),
            AssistantResponseType::UserMessage { content, .. } => content.clone(),
            AssistantResponseType::SystemMessage { content, .. } => content.clone(),
            AssistantResponseType::ToolExecution { tool_name, output, .. } => {
                format!("Tool: {}\nOutput: {}", tool_name, output)
            }
            _ => String::new(),
        }
    }
    
    /// Calculate relevance score
    fn calculate_relevance_score(&self, content: &str, filters: &ChatSearchFilters) -> f32 {
        let mut score = 1.0;
        
        if let Some(query) = &filters.query {
            let content_lower = content.to_lowercase();
            let query_lower = query.to_lowercase();
            
            // Count occurrences
            let count = content_lower.matches(&query_lower).count();
            score += count as f32 * 0.1;
            
            // Boost for exact match
            if content_lower == query_lower {
                score += 2.0;
            }
            
            // Boost for match at beginning
            if content_lower.starts_with(&query_lower) {
                score += 0.5;
            }
            
            // Boost for word boundaries
            let words: Vec<&str> = content_lower.split_whitespace().collect();
            if words.iter().any(|&word| word == query_lower) {
                score += 0.5;
            }
        }
        
        score
    }
    
    /// Create snippet with highlighted query
    fn create_snippet(&self, content: &str, query: Option<&str>) -> String {
        const MAX_SNIPPET_LEN: usize = 150;
        
        if let Some(q) = query {
            let lower_content = content.to_lowercase();
            let lower_query = q.to_lowercase();
            
            if let Some(pos) = lower_content.find(&lower_query) {
                // Find word boundaries
                let start = content[..pos]
                    .rfind(char::is_whitespace)
                    .map(|p| p + 1)
                    .unwrap_or(0)
                    .max(pos.saturating_sub(50));
                
                let end = content[pos + q.len()..]
                    .find(char::is_whitespace)
                    .map(|p| pos + q.len() + p)
                    .unwrap_or(content.len())
                    .min(pos + q.len() + 50);
                
                let mut snippet = String::new();
                if start > 0 {
                    snippet.push_str("...");
                }
                snippet.push_str(&content[start..end]);
                if end < content.len() {
                    snippet.push_str("...");
                }
                
                return snippet;
            }
        }
        
        // No query or not found, return beginning of content
        if content.len() <= MAX_SNIPPET_LEN {
            content.to_string()
        } else {
            format!("{}...", &content[..MAX_SNIPPET_LEN])
        }
    }
    
    /// Get context lines before
    fn get_context_before(
        &self,
        messages: &[AssistantResponseType],
        index: usize,
        count: usize,
    ) -> Vec<String> {
        let start = index.saturating_sub(count);
        messages[start..index]
            .iter()
            .map(|msg| self.extract_content(msg))
            .collect()
    }
    
    /// Get context lines after
    fn get_context_after(
        &self,
        messages: &[AssistantResponseType],
        index: usize,
        count: usize,
    ) -> Vec<String> {
        let end = (index + 1 + count).min(messages.len());
        messages[index + 1..end]
            .iter()
            .map(|msg| self.extract_content(msg))
            .collect()
    }
}