//! Search engine implementation with fuzzy search capabilities

use std::sync::Arc;
use std::cmp::min;
use anyhow::Result;
use chrono::Utc;
use regex::Regex;
use tokio::sync::RwLock;

use super::{ChatSearchFilters, ChatSearchResult};
use super::filters::MessageTypeFilter;
use crate::tui::chat::state::ChatState;
use crate::tui::run::AssistantResponseType;

/// Chat search engine with semantic and fuzzy search capabilities
pub struct SearchEngine {
    /// Maximum results to return
    max_results: usize,
    
    /// Reference to chat state
    chat_state: Option<Arc<RwLock<ChatState>>>,
    
    /// Multiple chat states for cross-chat search
    chat_states: std::collections::HashMap<usize, Arc<RwLock<ChatState>>>,
    
    /// Compiled regex cache
    regex_cache: std::collections::HashMap<String, Regex>,
    
    /// Semantic search index (word -> message indices)
    semantic_index: std::collections::HashMap<String, Vec<usize>>,
    
    /// Cross-chat semantic index (chat_id -> word -> message indices)
    cross_chat_index: std::collections::HashMap<usize, std::collections::HashMap<String, Vec<usize>>>,
    
    /// TF-IDF scores for relevance ranking
    tfidf_cache: std::collections::HashMap<String, f32>,
    
    /// Search result cache (query -> results)
    result_cache: std::collections::HashMap<String, Vec<ChatSearchResult>>,
    
    /// Maximum cache size
    max_cache_size: usize,
}

impl Default for SearchEngine {
    fn default() -> Self {
        Self {
            max_results: 100,
            chat_state: None,
            chat_states: std::collections::HashMap::new(),
            regex_cache: std::collections::HashMap::new(),
            semantic_index: std::collections::HashMap::new(),
            cross_chat_index: std::collections::HashMap::new(),
            tfidf_cache: std::collections::HashMap::new(),
            result_cache: std::collections::HashMap::new(),
            max_cache_size: 50,
        }
    }
}

impl SearchEngine {
    /// Create a new search engine
    pub fn new(max_results: usize) -> Self {
        Self {
            max_results,
            chat_state: None,
            chat_states: std::collections::HashMap::new(),
            regex_cache: std::collections::HashMap::new(),
            semantic_index: std::collections::HashMap::new(),
            cross_chat_index: std::collections::HashMap::new(),
            tfidf_cache: std::collections::HashMap::new(),
            result_cache: std::collections::HashMap::new(),
            max_cache_size: 50,
        }
    }
    
    /// Create with chat state reference
    pub fn with_chat_state(max_results: usize, chat_state: Arc<RwLock<ChatState>>) -> Self {
        let mut engine = Self {
            max_results,
            chat_state: Some(chat_state.clone()),
            chat_states: std::collections::HashMap::new(),
            regex_cache: std::collections::HashMap::new(),
            semantic_index: std::collections::HashMap::new(),
            cross_chat_index: std::collections::HashMap::new(),
            tfidf_cache: std::collections::HashMap::new(),
            result_cache: std::collections::HashMap::new(),
            max_cache_size: 50,
        };
        
        // Build initial index
        let _ = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(engine.build_index())
        });
        
        engine
    }
    
    /// Build semantic search index
    pub async fn build_index(&mut self) -> Result<()> {
        self.semantic_index.clear();
        self.tfidf_cache.clear();
        
        if let Some(chat_state) = &self.chat_state {
            let state = chat_state.read().await;
            let mut document_frequency: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
            let total_docs = state.messages.len();
            
            // First pass: build index and document frequency
            for (idx, message) in state.messages.iter().enumerate() {
                let content = self.extract_content(message);
                let words = self.tokenize(&content);
                let mut word_set = std::collections::HashSet::new();
                
                for word in words {
                    // Add to semantic index
                    self.semantic_index
                        .entry(word.clone())
                        .or_insert_with(Vec::new)
                        .push(idx);
                    
                    // Track unique words per document for DF
                    word_set.insert(word);
                }
                
                // Update document frequency
                for word in word_set {
                    *document_frequency.entry(word).or_insert(0) += 1;
                }
            }
            
            // Second pass: calculate TF-IDF scores
            for (word, doc_freq) in document_frequency {
                let idf = ((total_docs as f32) / (doc_freq as f32)).ln();
                self.tfidf_cache.insert(word, idf);
            }
        }
        
        Ok(())
    }
    
    /// Tokenize content into words
    fn tokenize(&self, content: &str) -> Vec<String> {
        content
            .to_lowercase()
            .split_whitespace()
            .filter_map(|word| {
                // Remove punctuation and filter short words
                let cleaned: String = word.chars()
                    .filter(|c| c.is_alphanumeric())
                    .collect();
                if cleaned.len() > 2 {
                    Some(cleaned)
                } else {
                    None
                }
            })
            .collect()
    }
    
    /// Perform semantic search
    pub async fn semantic_search(&mut self, query: &str, max_results: usize) -> Result<Vec<ChatSearchResult>> {
        let mut results = Vec::new();
        
        if let Some(chat_state) = &self.chat_state {
            let state = chat_state.read().await;
            let query_words = self.tokenize(query);
            let mut message_scores: std::collections::HashMap<usize, f32> = std::collections::HashMap::new();
            
            // Calculate relevance scores using TF-IDF
            for word in &query_words {
                if let Some(indices) = self.semantic_index.get(word) {
                    let idf = self.tfidf_cache.get(word).unwrap_or(&1.0);
                    
                    for &idx in indices {
                        let tf = 1.0; // Simplified TF (could be improved)
                        *message_scores.entry(idx).or_insert(0.0) += tf * idf;
                    }
                }
            }
            
            // Sort by score and create results
            let mut scored_messages: Vec<(usize, f32)> = message_scores.into_iter().collect();
            scored_messages.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            
            for (idx, score) in scored_messages.iter().take(max_results) {
                if let Some(message) = state.messages.get(*idx) {
                    let content = self.extract_content(message);
                    let author = match message {
                        AssistantResponseType::UserMessage { .. } => "User".to_string(),
                        AssistantResponseType::Message { .. } => "Assistant".to_string(),
                        _ => "System".to_string(),
                    };
                    
                    results.push(ChatSearchResult {
                        chat_id: state.id.parse::<usize>().unwrap_or(0),
                        message_index: *idx,
                        snippet: self.create_snippet(&content, Some(query)),
                        full_content: content,
                        author,
                        timestamp: Utc::now(),
                        score: *score,
                        context_before: self.get_context_before(&state.messages, *idx, 2),
                        context_after: self.get_context_after(&state.messages, *idx, 2),
                    });
                }
            }
        }
        
        Ok(results)
    }
    
    /// Search messages with filters
    pub async fn search(&mut self, filters: &ChatSearchFilters) -> Result<Vec<ChatSearchResult>> {
        // Check cache first
        if let Some(query) = &filters.query {
            let cache_key = format!("{:?}", filters);
            if let Some(cached) = self.result_cache.get(&cache_key) {
                return Ok(cached.clone());
            }
        }
        
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
                self.regex_cache.get(query)
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
        
        // Sort by relevance score (handle NaN gracefully)
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        
        // Limit results
        results.truncate(self.max_results);
        
        // Cache results if there's a query
        if let Some(query) = &filters.query {
            let cache_key = format!("{:?}", filters);
            
            // Implement LRU cache eviction
            if self.result_cache.len() >= self.max_cache_size {
                // Remove oldest entry (simplified - could use proper LRU)
                if let Some(first_key) = self.result_cache.keys().next().cloned() {
                    self.result_cache.remove(&first_key);
                }
            }
            
            self.result_cache.insert(cache_key, results.clone());
        }
        
        Ok(results)
    }
    
    /// Perform fuzzy search with Levenshtein distance
    pub async fn fuzzy_search(&mut self, query: &str, max_distance: usize) -> Result<Vec<ChatSearchResult>> {
        let mut results = Vec::new();
        
        // If no chat state, return empty results
        let chat_state = match &self.chat_state {
            Some(state) => state,
            None => return Ok(results),
        };
        
        let state = chat_state.read().await;
        
        // Search through all messages
        for (msg_idx, message) in state.messages.iter().enumerate() {
            let content = self.extract_content(message);
            
            // Calculate fuzzy match score
            let score = self.fuzzy_match_score(&content, query, max_distance);
            
            if score > 0.0 {
                let (_, author) = match message {
                    AssistantResponseType::Message { .. } => {
                        (content.clone(), "Assistant".to_string())
                    }
                    AssistantResponseType::UserMessage { .. } => {
                        (content.clone(), "User".to_string())
                    }
                    _ => continue,
                };
                
                results.push(ChatSearchResult {
                    chat_id: state.id.parse::<usize>().unwrap_or(0),
                    message_index: msg_idx,
                    snippet: self.create_snippet(&content, Some(query)),
                    full_content: content.clone(),
                    author,
                    timestamp: Utc::now(),
                    score,
                    context_before: self.get_context_before(&state.messages, msg_idx, 2),
                    context_after: self.get_context_after(&state.messages, msg_idx, 2),
                });
            }
        }
        
        // Sort by score
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(self.max_results);
        
        Ok(results)
    }
    
    /// Calculate fuzzy match score for a text against a query
    fn fuzzy_match_score(&self, text: &str, query: &str, max_distance: usize) -> f32 {
        let text_lower = text.to_lowercase();
        let query_lower = query.to_lowercase();
        let words: Vec<&str> = text_lower.split_whitespace().collect();
        let mut best_score = 0.0f32;
        
        // Check each word in the text
        for word in words {
            let distance = self.levenshtein_distance(word, &query_lower);
            if distance <= max_distance {
                // Convert distance to score (0 distance = 1.0 score)
                let score = 1.0 - (distance as f32 / query_lower.len().max(1) as f32);
                best_score = best_score.max(score);
            }
        }
        
        // Also check for substring fuzzy matches
        if text_lower.len() >= query_lower.len() {
            for i in 0..=(text_lower.len() - query_lower.len()) {
                let substring = &text_lower[i..i + query_lower.len()];
                let distance = self.levenshtein_distance(substring, &query_lower);
                if distance <= max_distance {
                    let score = 1.0 - (distance as f32 / query_lower.len().max(1) as f32);
                    best_score = best_score.max(score * 0.8); // Slightly lower weight for substring matches
                }
            }
        }
        
        best_score
    }
    
    /// Calculate Levenshtein distance between two strings
    fn levenshtein_distance(&self, s1: &str, s2: &str) -> usize {
        let len1 = s1.chars().count();
        let len2 = s2.chars().count();
        
        if len1 == 0 { return len2; }
        if len2 == 0 { return len1; }
        
        let s1_chars: Vec<char> = s1.chars().collect();
        let s2_chars: Vec<char> = s2.chars().collect();
        
        let mut prev_row: Vec<usize> = (0..=len2).collect();
        let mut curr_row = vec![0; len2 + 1];
        
        for i in 1..=len1 {
            curr_row[0] = i;
            for j in 1..=len2 {
                let cost = if s1_chars[i - 1] == s2_chars[j - 1] { 0 } else { 1 };
                curr_row[j] = min(
                    min(prev_row[j] + 1, curr_row[j - 1] + 1),
                    prev_row[j - 1] + cost
                );
            }
            std::mem::swap(&mut prev_row, &mut curr_row);
        }
        
        prev_row[len2]
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
                    content.clone()
                } else {
                    content.to_lowercase()
                };
                
                let query_to_search = if filters.case_sensitive {
                    query.clone()
                } else {
                    query.to_lowercase()
                };
                
                // Support advanced query operators
                if query.contains(" AND ") {
                    // All terms must be present
                    let terms: Vec<&str> = query_to_search.split(" AND ").collect();
                    if !terms.iter().all(|term| content_to_search.contains(term.trim())) {
                        return false;
                    }
                } else if query.contains(" OR ") {
                    // At least one term must be present
                    let terms: Vec<&str> = query_to_search.split(" OR ").collect();
                    if !terms.iter().any(|term| content_to_search.contains(term.trim())) {
                        return false;
                    }
                } else if query.starts_with("NOT ") {
                    // Exclude messages containing the term
                    let term = &query_to_search[4..];
                    if content_to_search.contains(term) {
                        return false;
                    }
                } else if query.starts_with('"') && query.ends_with('"') {
                    // Exact phrase search
                    let phrase = &query_to_search[1..query_to_search.len()-1];
                    if !content_to_search.contains(phrase) {
                        return false;
                    }
                } else {
                    // Simple contains check
                    if !content_to_search.contains(&query_to_search) {
                        return false;
                    }
                }
            }
        }
        
        // Check for additional advanced filters
        // Filter by content length
        if filters.min_length.is_some() || filters.max_length.is_some() {
            let content = self.extract_content(message);
            let length = content.len();
            
            if let Some(min) = filters.min_length {
                if length < min {
                    return false;
                }
            }
            
            if let Some(max) = filters.max_length {
                if length > max {
                    return false;
                }
            }
        }
        
        // Filter by whether message has code blocks
        if filters.has_code_blocks.unwrap_or(false) {
            let content = self.extract_content(message);
            if !content.contains("```") {
                return false;
            }
        }
        
        // Filter by whether message has links
        if filters.has_links.unwrap_or(false) {
            let content = self.extract_content(message);
            if !content.contains("http://") && !content.contains("https://") {
                return false;
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
    
    /// Add a chat state for cross-chat search
    pub fn add_chat_state(&mut self, chat_id: usize, chat_state: Arc<RwLock<ChatState>>) {
        self.chat_states.insert(chat_id, chat_state.clone());
        
        // Build index for this chat
        let _ = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(self.build_chat_index(chat_id))
        });
    }
    
    /// Build index for a specific chat
    async fn build_chat_index(&mut self, chat_id: usize) -> Result<()> {
        if let Some(chat_state) = self.chat_states.get(&chat_id) {
            let state = chat_state.read().await;
            let mut chat_index = std::collections::HashMap::new();
            
            for (idx, message) in state.messages.iter().enumerate() {
                let content = self.extract_content(message);
                let words = self.tokenize(&content);
                
                for word in words {
                    chat_index
                        .entry(word)
                        .or_insert_with(Vec::new)
                        .push(idx);
                }
            }
            
            self.cross_chat_index.insert(chat_id, chat_index);
        }
        
        Ok(())
    }
    
    /// Search across multiple chats
    pub async fn cross_chat_search(
        &mut self,
        query: &str,
        chat_ids: Option<Vec<usize>>,
    ) -> Result<Vec<ChatSearchResult>> {
        let mut all_results = Vec::new();
        
        // Determine which chats to search
        let chats_to_search: Vec<usize> = if let Some(ids) = chat_ids {
            ids
        } else {
            self.chat_states.keys().cloned().collect()
        };
        
        // Search each chat
        for chat_id in chats_to_search {
            if let Some(chat_state) = self.chat_states.get(&chat_id) {
                let state = chat_state.read().await;
                
                // Search using the cross-chat index
                if let Some(chat_index) = self.cross_chat_index.get(&chat_id) {
                    let query_words = self.tokenize(query);
                    let mut message_scores: std::collections::HashMap<usize, f32> = std::collections::HashMap::new();
                    
                    // Calculate relevance scores
                    for word in &query_words {
                        if let Some(indices) = chat_index.get(word) {
                            for &idx in indices {
                                *message_scores.entry(idx).or_insert(0.0) += 1.0;
                            }
                        }
                    }
                    
                    // Create results for matching messages
                    for (msg_idx, score) in message_scores {
                        if let Some(message) = state.messages.get(msg_idx) {
                            let content = self.extract_content(message);
                            let author = match message {
                                AssistantResponseType::UserMessage { .. } => "User".to_string(),
                                AssistantResponseType::Message { .. } => "Assistant".to_string(),
                                _ => "System".to_string(),
                            };
                            
                            all_results.push(ChatSearchResult {
                                chat_id,
                                message_index: msg_idx,
                                snippet: self.create_snippet(&content, Some(query)),
                                full_content: content,
                                author,
                                timestamp: Utc::now(),
                                score,
                                context_before: self.get_context_before(&state.messages, msg_idx, 1),
                                context_after: self.get_context_after(&state.messages, msg_idx, 1),
                            });
                        }
                    }
                }
            }
        }
        
        // Sort by relevance across all chats
        all_results.sort_by(|a, b| {
            b.score.partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        
        // Limit results
        all_results.truncate(self.max_results);
        
        Ok(all_results)
    }
    
    /// Get statistics about indexed chats
    pub fn get_index_stats(&self) -> std::collections::HashMap<usize, usize> {
        let mut stats = std::collections::HashMap::new();
        
        for (chat_id, index) in &self.cross_chat_index {
            let total_words: usize = index.values().map(|v| v.len()).sum();
            stats.insert(*chat_id, total_words);
        }
        
        stats
    }
}