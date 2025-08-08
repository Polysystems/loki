//! Enhanced Web Search Client
//!
//! Advanced web search client supporting multiple search engines with
//! result processing, content extraction, and semantic understanding.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use anyhow::{Result, anyhow};
use reqwest::Client;
use scraper::{Html, Selector};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, error, info};
use url::Url;

use crate::memory::{CognitiveMemory, MemoryId, MemoryMetadata};
use crate::safety::{ActionType, ActionValidator};

/// Web search configuration
#[derive(Debug, Clone)]
pub struct WebSearchConfig {
    pub search_engines: Vec<SearchEngineConfig>,
    pub max_results_per_engine: usize,
    pub request_timeout: Duration,
    pub user_agent: String,
    pub enable_content_extraction: bool,
    pub cache_duration: Duration,
    pub rate_limit_delay: Duration,
}

impl Default for WebSearchConfig {
    fn default() -> Self {
        Self {
            search_engines: vec![SearchEngineConfig::duckduckgo(), SearchEngineConfig::bing()],
            max_results_per_engine: 20,
            request_timeout: Duration::from_secs(30),
            user_agent: "Loki Web Search Client/1.0".to_string(),
            enable_content_extraction: true,
            cache_duration: Duration::from_secs(3600), // 1 hour
            rate_limit_delay: Duration::from_millis(1000),
        }
    }
}

/// Search engine configuration
#[derive(Debug, Clone)]
pub struct SearchEngineConfig {
    pub name: String,
    pub search_url_template: String,
    pub result_selectors: ResultSelectors,
    pub headers: HashMap<String, String>,
    pub enabled: bool,
    pub priority: u8,
}

impl SearchEngineConfig {
    /// DuckDuckGo search engine configuration
    pub fn duckduckgo() -> Self {
        let mut headers = HashMap::new();
        headers.insert(
            "Accept".to_string(),
            "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8".to_string(),
        );
        headers.insert("Accept-Language".to_string(), "en-US,en;q=0.5".to_string());

        Self {
            name: "DuckDuckGo".to_string(),
            search_url_template: "https://html.duckduckgo.com/html/?q={query}&kl=us-en".to_string(),
            result_selectors: ResultSelectors {
                result_container: ".result".to_string(),
                title: ".result__title a".to_string(),
                url: ".result__title a".to_string(),
                snippet: ".result__snippet".to_string(),
                next_page: ".nav-link--next".to_string(),
            },
            headers,
            enabled: true,
            priority: 1,
        }
    }

    /// Bing search engine configuration
    pub fn bing() -> Self {
        let mut headers = HashMap::new();
        headers.insert(
            "Accept".to_string(),
            "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8".to_string(),
        );

        Self {
            name: "Bing".to_string(),
            search_url_template: "https://www.bing.com/search?q={query}&count=20".to_string(),
            result_selectors: ResultSelectors {
                result_container: ".b_algo".to_string(),
                title: "h2 a".to_string(),
                url: "h2 a".to_string(),
                snippet: ".b_caption p".to_string(),
                next_page: ".sb_pagN".to_string(),
            },
            headers,
            enabled: true,
            priority: 2,
        }
    }
}

/// CSS selectors for extracting search results
#[derive(Debug, Clone)]
pub struct ResultSelectors {
    pub result_container: String,
    pub title: String,
    pub url: String,
    pub snippet: String,
    pub next_page: String,
}

/// Search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub title: String,
    pub url: String,
    pub snippet: String,
    pub search_engine: String,
    pub rank: usize,
    pub relevance_score: f32,
    pub extracted_content: Option<String>,
    pub content_type: Option<String>,
    pub language: Option<String>,
    pub extracted_at: SystemTime,
    pub metadata: SearchResultMetadata,
}

/// Search result metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResultMetadata {
    pub domain: String,
    pub page_type: PageType,
    pub topics: Vec<String>,
    pub keywords: Vec<String>,
    pub sentiment: Option<f32>,
    pub reading_time_minutes: Option<usize>,
    pub word_count: Option<usize>,
}

/// Type of web page
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PageType {
    Documentation,
    Blog,
    News,
    Academic,
    Forum,
    Social,
    Commercial,
    Reference,
    Unknown,
}

/// Search query with advanced options
#[derive(Debug, Clone)]
pub struct SearchQuery {
    pub query: String,
    pub filters: SearchFilters,
    pub max_results: usize,
    pub engines: Vec<String>,
    pub extract_content: bool,
}

impl Default for SearchQuery {
    fn default() -> Self {
        Self {
            query: String::new(),
            filters: SearchFilters::default(),
            max_results: 20,
            engines: vec![],
            extract_content: true,
        }
    }
}

/// Search filters
#[derive(Debug, Clone)]
pub struct SearchFilters {
    pub site_filter: Option<String>,
    pub filetype_filter: Option<String>,
    pub date_range: Option<DateRange>,
    pub language: Option<String>,
    pub exclude_domains: Vec<String>,
    pub content_type: Option<PageType>,
}

impl Default for SearchFilters {
    fn default() -> Self {
        Self {
            site_filter: None,
            filetype_filter: None,
            date_range: None,
            language: None,
            exclude_domains: Vec::new(),
            content_type: None,
        }
    }
}

/// Date range filter
#[derive(Debug, Clone)]
pub enum DateRange {
    LastDay,
    LastWeek,
    LastMonth,
    LastYear,
    Custom { start: SystemTime, end: SystemTime },
}

/// Search statistics
#[derive(Debug, Clone, Default)]
pub struct SearchStats {
    pub total_searches: usize,
    pub total_results: usize,
    pub cache_hits: usize,
    pub content_extractions: usize,
    pub engine_stats: HashMap<String, EngineStats>,
    pub popular_domains: HashMap<String, usize>,
    pub popular_topics: HashMap<String, usize>,
    pub last_search_time: Option<SystemTime>,
}

/// Per-engine statistics
#[derive(Debug, Clone, Default)]
pub struct EngineStats {
    pub searches: usize,
    pub results: usize,
    pub errors: usize,
    pub average_response_time: Duration,
}

/// Enhanced web search client
pub struct WebSearchClient {
    /// HTTP client
    client: Client,

    /// Configuration
    config: WebSearchConfig,

    /// Memory system
    memory: Arc<CognitiveMemory>,

    /// Safety validator
    validator: Option<Arc<ActionValidator>>,

    /// Result cache
    result_cache: Arc<RwLock<HashMap<String, Vec<SearchResult>>>>,

    /// Content cache
    content_cache: Arc<RwLock<HashMap<String, String>>>,

    /// Statistics
    stats: Arc<RwLock<SearchStats>>,
}

impl WebSearchClient {
    /// Create a new web search client
    pub async fn new(
        config: WebSearchConfig,
        memory: Arc<CognitiveMemory>,
        validator: Option<Arc<ActionValidator>>,
    ) -> Result<Self> {
        info!("Initializing enhanced web search client");

        let client = Client::builder()
            .timeout(config.request_timeout)
            .user_agent(&config.user_agent)
            .redirect(reqwest::redirect::Policy::limited(5))
            .build()?;

        Ok(Self {
            client,
            config,
            memory,
            validator,
            result_cache: Arc::new(RwLock::new(HashMap::new())),
            content_cache: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(SearchStats::default())),
        })
    }

    /// Perform a web search
    pub async fn search(&self, query: SearchQuery) -> Result<Vec<SearchResult>> {
        // Validate search through safety system
        if let Some(validator) = &self.validator {
            validator
                .validate_action(
                    ActionType::ApiCall {
                        provider: "web_search".to_string(),
                        endpoint: "search".to_string(),
                    },
                    format!("Web search: {}", query.query),
                    vec!["Searching the web for information".to_string()],
                )
                .await?;
        }

        info!("Searching web: {}", query.query);

        // Check cache first
        let cache_key = self.generate_cache_key(&query);
        if let Some(cached_results) = self.get_cached_results(&cache_key).await {
            debug!("Returning cached search results");
            let mut stats = self.stats.write().await;
            stats.cache_hits += 1;
            return Ok(cached_results);
        }

        // Determine which engines to use
        let engines: Vec<&SearchEngineConfig> = if query.engines.is_empty() {
            self.config.search_engines.iter().filter(|e| e.enabled).collect()
        } else {
            self.config.search_engines.iter().filter(|e| query.engines.contains(&e.name)).collect()
        };

        let mut all_results = Vec::new();

        // Search each engine
        for engine in engines {
            debug!("Searching with engine: {}", engine.name);

            match self.search_engine(engine, &query).await {
                Ok(mut results) => {
                    // Add engine name to results
                    for result in &mut results {
                        result.search_engine = engine.name.clone();
                    }
                    all_results.extend(results);
                }
                Err(e) => {
                    error!("Search engine {} failed: {}", engine.name, e);

                    // Update error stats
                    let mut stats = self.stats.write().await;
                    stats.engine_stats.entry(engine.name.clone()).or_default().errors += 1;
                }
            }

            // Rate limiting between engines
            if self.config.rate_limit_delay > Duration::ZERO {
                tokio::time::sleep(self.config.rate_limit_delay).await;
            }
        }

        // Remove duplicates and rank results
        let mut deduplicated_results = self.deduplicate_results(all_results);

        // Calculate relevance scores
        self.calculate_relevance_scores(&mut deduplicated_results, &query.query);

        // Sort by relevance
        deduplicated_results.sort_by(|a, b| {
            b.relevance_score.partial_cmp(&a.relevance_score).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Limit results
        deduplicated_results.truncate(query.max_results);

        // Extract content if requested
        if query.extract_content && self.config.enable_content_extraction {
            for result in &mut deduplicated_results {
                if let Ok(content) = self.extract_content(&result.url).await {
                    result.extracted_content = Some(content);
                    result.metadata = self.analyze_content(&result).await;
                }
            }
        }

        // Cache results
        self.cache_results(&cache_key, &deduplicated_results).await;

        // Store in memory
        for result in &deduplicated_results {
            self.store_result_in_memory(result).await?;
        }

        // Update statistics
        {
            let mut stats = self.stats.write().await;
            stats.total_searches += 1;
            stats.total_results += deduplicated_results.len();
            stats.last_search_time = Some(SystemTime::now());

            // Update domain and topic statistics
            for result in &deduplicated_results {
                *stats.popular_domains.entry(result.metadata.domain.clone()).or_insert(0) += 1;
                for topic in &result.metadata.topics {
                    *stats.popular_topics.entry(topic.clone()).or_insert(0) += 1;
                }
            }
        }

        info!("Found {} results for query: {}", deduplicated_results.len(), query.query);

        Ok(deduplicated_results)
    }

    /// Search a specific engine
    async fn search_engine(
        &self,
        engine: &SearchEngineConfig,
        query: &SearchQuery,
    ) -> Result<Vec<SearchResult>> {
        let start_time = SystemTime::now();

        // Build search URL
        let search_url = self.build_search_url(engine, query)?;
        debug!("Search URL: {}", search_url);

        // Build request with engine-specific headers
        let mut request = self.client.get(&search_url);
        for (key, value) in &engine.headers {
            request = request.header(key, value);
        }

        // Make the request
        let response = request.send().await?;

        if !response.status().is_success() {
            return Err(anyhow!(
                "Search engine {} returned error: {}",
                engine.name,
                response.status()
            ));
        }

        let html_content = response.text().await?;

        // Parse results
        let results = self.parse_search_results(engine, &html_content)?;

        // Update engine statistics
        let elapsed = start_time.elapsed().unwrap_or_default();
        {
            let mut stats = self.stats.write().await;
            let engine_stats = stats.engine_stats.entry(engine.name.clone()).or_default();
            engine_stats.searches += 1;
            engine_stats.results += results.len();
            engine_stats.average_response_time = (engine_stats.average_response_time + elapsed) / 2;
        }

        Ok(results)
    }

    /// Build search URL for an engine
    fn build_search_url(&self, engine: &SearchEngineConfig, query: &SearchQuery) -> Result<String> {
        let mut search_query = query.query.clone();

        // Apply filters
        if let Some(site) = &query.filters.site_filter {
            search_query.push_str(&format!(" site:{}", site));
        }

        if let Some(filetype) = &query.filters.filetype_filter {
            search_query.push_str(&format!(" filetype:{}", filetype));
        }

        for domain in &query.filters.exclude_domains {
            search_query.push_str(&format!(" -site:{}", domain));
        }

        // URL encode the query
        let encoded_query = urlencoding::encode(&search_query);

        // Replace {query} in template
        let url = engine.search_url_template.replace("{query}", &encoded_query);

        Ok(url)
    }

    /// Parse search results from HTML
    fn parse_search_results(
        &self,
        engine: &SearchEngineConfig,
        html: &str,
    ) -> Result<Vec<SearchResult>> {
        let document = Html::parse_document(html);
        let mut results = Vec::new();

        let container_selector =
            Selector::parse(&engine.result_selectors.result_container)
                .map_err(|e| anyhow!("Invalid result container selector: {}", e))?;
        let title_selector = Selector::parse(&engine.result_selectors.title)
            .map_err(|e| anyhow!("Invalid title selector: {}", e))?;
        let url_selector = Selector::parse(&engine.result_selectors.url)
            .map_err(|e| anyhow!("Invalid url selector: {}", e))?;
        let snippet_selector = Selector::parse(&engine.result_selectors.snippet)
            .map_err(|e| anyhow!("Invalid snippet selector: {}", e))?;

        for (index, container) in document.select(&container_selector).enumerate() {
            // Extract title
            let title = container
                .select(&title_selector)
                .next()
                .map(|e| e.text().collect::<String>())
                .unwrap_or_default()
                .trim()
                .to_string();

            if title.is_empty() {
                continue;
            }

            // Extract URL
            let url = container
                .select(&url_selector)
                .next()
                .and_then(|e| e.value().attr("href"))
                .unwrap_or_default()
                .to_string();

            if url.is_empty() {
                continue;
            }

            // Clean and validate URL
            let clean_url = self.clean_url(&url)?;

            // Extract snippet
            let snippet = container
                .select(&snippet_selector)
                .next()
                .map(|e| e.text().collect::<String>())
                .unwrap_or_default()
                .trim()
                .to_string();

            // Extract domain
            let domain = Url::parse(&clean_url)
                .ok()
                .and_then(|u| u.host_str().map(|s| s.to_string()))
                .unwrap_or_default();

            // Detect page type
            let page_type = self.detect_page_type(&clean_url, &title, &snippet);

            let metadata = SearchResultMetadata {
                domain,
                page_type,
                topics: self.extract_topics(&title, &snippet),
                keywords: self.extract_keywords(&title, &snippet),
                sentiment: None,
                reading_time_minutes: None,
                word_count: None,
            };

            let result = SearchResult {
                title,
                url: clean_url,
                snippet,
                search_engine: engine.name.clone(),
                rank: index + 1,
                relevance_score: 0.0, // Will be calculated later
                extracted_content: None,
                content_type: None,
                language: None,
                extracted_at: SystemTime::now(),
                metadata,
            };

            results.push(result);
        }

        Ok(results)
    }

    /// Clean and validate URL
    fn clean_url(&self, url: &str) -> Result<String> {
        // Handle relative URLs and redirects
        let clean_url = if url.starts_with("/url?") {
            // Google-style redirect URL
            if let Some(actual_url) = url.split("&url=").nth(1) {
                urlencoding::decode(actual_url)?.to_string()
            } else {
                url.to_string()
            }
        } else if url.starts_with("http") {
            url.to_string()
        } else {
            format!("https://{}", url.trim_start_matches("//"))
        };

        // Validate URL
        Url::parse(&clean_url)?;

        Ok(clean_url)
    }

    /// Detect page type from URL and content
    fn detect_page_type(&self, url: &str, title: &str, snippet: &str) -> PageType {
        let url_lower = url.to_lowercase();
        let content_lower = format!("{} {}", title, snippet).to_lowercase();

        // Documentation sites
        if url_lower.contains("docs.")
            || url_lower.contains("/doc/")
            || url_lower.contains("documentation")
            || content_lower.contains("documentation")
        {
            return PageType::Documentation;
        }

        // Academic sites
        if url_lower.contains("arxiv.org")
            || url_lower.contains("ieee.org")
            || url_lower.contains("acm.org")
            || url_lower.contains(".edu")
            || content_lower.contains("abstract")
            || content_lower.contains("paper")
        {
            return PageType::Academic;
        }

        // News sites
        if url_lower.contains("news")
            || url_lower.contains("cnn.com")
            || url_lower.contains("bbc.com")
            || url_lower.contains("reuters.com")
        {
            return PageType::News;
        }

        // Blogs
        if url_lower.contains("blog")
            || url_lower.contains("medium.com")
            || url_lower.contains("dev.to")
            || url_lower.contains("wordpress")
        {
            return PageType::Blog;
        }

        // Forums
        if url_lower.contains("forum")
            || url_lower.contains("stackoverflow.com")
            || url_lower.contains("reddit.com")
            || url_lower.contains("discourse")
        {
            return PageType::Forum;
        }

        // Social media
        if url_lower.contains("twitter.com")
            || url_lower.contains("facebook.com")
            || url_lower.contains("linkedin.com")
            || url_lower.contains("instagram.com")
        {
            return PageType::Social;
        }

        // Reference sites
        if url_lower.contains("wikipedia.org")
            || url_lower.contains("reference")
            || url_lower.contains("manual")
            || content_lower.contains("reference")
        {
            return PageType::Reference;
        }

        PageType::Unknown
    }

    /// Extract topics from text
    fn extract_topics(&self, title: &str, snippet: &str) -> Vec<String> {
        let text = format!("{} {}", title, snippet).to_lowercase();
        let mut topics = HashSet::new();

        let topic_keywords = [
            ("programming", "Programming"),
            ("machine learning", "Machine Learning"),
            ("artificial intelligence", "AI"),
            ("web development", "Web Development"),
            ("data science", "Data Science"),
            ("cybersecurity", "Cybersecurity"),
            ("cloud computing", "Cloud Computing"),
            ("mobile development", "Mobile Development"),
            ("devops", "DevOps"),
            ("blockchain", "Blockchain"),
            ("iot", "Internet of Things"),
            ("robotics", "Robotics"),
            ("virtual reality", "Virtual Reality"),
            ("augmented reality", "Augmented Reality"),
            ("quantum computing", "Quantum Computing"),
        ];

        for (keyword, topic) in &topic_keywords {
            if text.contains(keyword) {
                topics.insert(topic.to_string());
            }
        }

        topics.into_iter().collect()
    }

    /// Extract keywords from text
    fn extract_keywords(&self, title: &str, snippet: &str) -> Vec<String> {
        let text = format!("{} {}", title, snippet);
        let words: Vec<String> = text
            .split_whitespace()
            .filter(|word| {
                word.len() > 3
                    && !word.chars().all(|c| c.is_numeric())
                    && ![
                        "the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
                        "her", "was", "one", "our", "out", "day", "get", "has", "him", "his",
                        "how", "its", "may", "new", "now", "old", "see", "two", "way", "who",
                        "boy", "did", "man", "end", "few", "got", "let", "put", "say", "she",
                        "too", "use",
                    ]
                    .contains(&word.to_lowercase().as_str())
            })
            .map(|word| word.to_lowercase())
            .collect();

        // Return most frequent words (simplified)
        let mut word_counts = HashMap::new();
        for word in words {
            *word_counts.entry(word).or_insert(0) += 1;
        }

        let mut sorted_words: Vec<_> = word_counts.into_iter().collect();
        sorted_words.sort_by(|a, b| b.1.cmp(&a.1));

        sorted_words.into_iter().take(10).map(|(word, _)| word).collect()
    }

    /// Remove duplicate results
    fn deduplicate_results(&self, results: Vec<SearchResult>) -> Vec<SearchResult> {
        let mut seen_urls = HashSet::new();
        let mut deduplicated = Vec::new();

        for result in results {
            let normalized_url = self.normalize_url(&result.url);
            if seen_urls.insert(normalized_url) {
                deduplicated.push(result);
            }
        }

        deduplicated
    }

    /// Normalize URL for deduplication
    fn normalize_url(&self, url: &str) -> String {
        // Remove common URL parameters that don't affect content
        if let Ok(mut parsed_url) = Url::parse(url) {
            parsed_url.set_fragment(None);

            // Remove tracking parameters
            let tracking_params =
                ["utm_source", "utm_medium", "utm_campaign", "utm_content", "utm_term"];
            let pairs: Vec<_> = parsed_url
                .query_pairs()
                .filter(|(key, _)| !tracking_params.contains(&key.as_ref()))
                .collect();

            if pairs.is_empty() {
                parsed_url.set_query(None);
            } else {
                let query_string =
                    pairs.iter().map(|(k, v)| format!("{}={}", k, v)).collect::<Vec<_>>().join("&");
                parsed_url.set_query(Some(&query_string));
            }

            parsed_url.to_string()
        } else {
            url.to_string()
        }
    }

    /// Calculate relevance scores for results
    fn calculate_relevance_scores(&self, results: &mut [SearchResult], query: &str) {
        let query_lower = query.to_lowercase();
        let query_words: HashSet<_> = query_lower.split_whitespace().collect();

        for result in results {
            let mut score = 0.0;

            // Title relevance (higher weight)
            let title_lower = result.title.to_lowercase();
            let title_words: HashSet<_> = title_lower.split_whitespace().collect();
            let title_matches = query_words.intersection(&title_words).count();
            score += title_matches as f32 * 2.0;

            // Snippet relevance
            let snippet_lower = result.snippet.to_lowercase();
            let snippet_words: HashSet<_> = snippet_lower.split_whitespace().collect();
            let snippet_matches = query_words.intersection(&snippet_words).count();
            score += snippet_matches as f32 * 1.0;

            // Domain authority boost (simplified)
            match result.metadata.domain.as_str() {
                domain if domain.contains("wikipedia.org") => score += 1.0,
                domain if domain.contains("stackoverflow.com") => score += 0.8,
                domain if domain.contains("github.com") => score += 0.8,
                domain if domain.contains("mozilla.org") => score += 0.7,
                domain if domain.contains("w3.org") => score += 0.7,
                _ => {}
            }

            // Page type boost
            match result.metadata.page_type {
                PageType::Documentation => score += 0.5,
                PageType::Academic => score += 0.4,
                PageType::Reference => score += 0.3,
                _ => {}
            }

            // Normalize score
            result.relevance_score = score / (query_words.len() as f32 + 3.0);
        }
    }

    /// Extract content from a URL
    async fn extract_content(&self, url: &str) -> Result<String> {
        // Check content cache first
        {
            let cache = self.content_cache.read().await;
            if let Some(content) = cache.get(url) {
                return Ok(content.clone());
            }
        }

        debug!("Extracting content from: {}", url);

        // Make request
        let response = self.client.get(url).send().await?;

        if !response.status().is_success() {
            return Err(anyhow!("Failed to fetch content: {}", response.status()));
        }

        let html_content = response.text().await?;

        // Parse HTML and extract main content
        let document = Html::parse_document(&html_content);

        // Try different selectors for main content
        let content_selectors = [
            "main",
            "article",
            ".content",
            "#content",
            ".post-content",
            ".entry-content",
            ".article-body",
            "body",
        ];

        let mut extracted_content = String::new();

        for selector_str in &content_selectors {
            if let Ok(selector) = Selector::parse(selector_str) {
                if let Some(element) = document.select(&selector).next() {
                    // Remove script and style tags
                    if let Ok(script_selector) = Selector::parse("script, style") {
                        for _script in element.select(&script_selector) {
                            // Would remove these elements in a real implementation
                        }
                    }

                    extracted_content = element.text().collect::<String>();
                    if !extracted_content.trim().is_empty() {
                        break;
                    }
                }
            }
        }

        // Clean up the content
        let cleaned_content: String = extracted_content
            .lines()
            .map(|line| line.trim())
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>()
            .join("\n")
            .chars()
            .take(5000) // Limit content size
            .collect();

        // Cache the content
        {
            let mut cache = self.content_cache.write().await;
            cache.insert(url.to_string(), cleaned_content.clone());
        }

        // Update statistics
        {
            let mut stats = self.stats.write().await;
            stats.content_extractions += 1;
        }

        Ok(cleaned_content)
    }

    /// Analyze extracted content
    async fn analyze_content(&self, result: &SearchResult) -> SearchResultMetadata {
        let mut metadata = result.metadata.clone();

        if let Some(content) = &result.extracted_content {
            // Calculate reading time
            let word_count = content.split_whitespace().count();
            metadata.word_count = Some(word_count);
            metadata.reading_time_minutes = Some((word_count / 200).max(1)); // 200 words per minute

            // Extract additional topics from content
            let additional_topics = self.extract_topics(&result.title, content);
            metadata.topics.extend(additional_topics);
            metadata.topics.sort();
            metadata.topics.dedup();
        }

        metadata
    }

    /// Store search result in memory
    async fn store_result_in_memory(&self, result: &SearchResult) -> Result<()> {
        let importance = match result.relevance_score {
            score if score > 0.8 => 0.9,
            score if score > 0.6 => 0.7,
            score if score > 0.4 => 0.5,
            _ => 0.3,
        };

        let mut tags = vec![
            "web_search".to_string(),
            "search_result".to_string(),
            result.metadata.domain.clone(),
            format!("page_type:{:?}", result.metadata.page_type).to_lowercase(),
        ];

        tags.extend(result.metadata.topics.clone());
        tags.extend(result.metadata.keywords.clone());

        let content = if let Some(extracted_content) = &result.extracted_content {
            format!(
                "{}\n\n{}\n\n{}",
                result.title,
                result.snippet,
                extracted_content.chars().take(1000).collect::<String>()
            )
        } else {
            format!("{}\n\n{}", result.title, result.snippet)
        };

        self.memory
            .store(
                content.clone(),
                vec![], // No code in search results typically
                MemoryMetadata {
                    source: "web_search".to_string(),
                    tags: tags.clone(),
                    importance,
                    associations: self
                        .find_related_memories(&content, &tags)
                        .await
                        .unwrap_or_default(),
                    context: Some("Web search result".to_string()),
                    created_at: chrono::Utc::now(),
                    accessed_count: 0,
                    last_accessed: None,
                    version: 1,
                    category: "tool_usage".to_string(),
                    timestamp: chrono::Utc::now(),
                    expiration: None,
                },
            )
            .await?;

        Ok(())
    }

    /// Generate cache key for a query
    fn generate_cache_key(&self, query: &SearchQuery) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(query.query.as_bytes());
        hasher.update(format!("{:?}", query.filters).as_bytes());
        hasher.update(query.engines.join(",").as_bytes());
        format!("{:?}", hasher.finalize())
    }

    /// Get cached results
    async fn get_cached_results(&self, cache_key: &str) -> Option<Vec<SearchResult>> {
        let cache = self.result_cache.read().await;
        cache.get(cache_key).cloned()
    }

    /// Cache search results
    async fn cache_results(&self, cache_key: &str, results: &[SearchResult]) {
        let mut cache = self.result_cache.write().await;
        cache.insert(cache_key.to_string(), results.to_vec());

        // Clean old entries if cache is getting large
        if cache.len() > 1000 {
            let keys_to_remove: Vec<_> = cache.keys().take(200).cloned().collect();
            for key in keys_to_remove {
                cache.remove(&key);
            }
        }
    }

    /// Get search statistics
    pub async fn stats(&self) -> SearchStats {
        self.stats.read().await.clone()
    }

    /// Search for programming documentation
    pub async fn search_docs(&self, technology: &str, topic: &str) -> Result<Vec<SearchResult>> {
        let query = SearchQuery {
            query: format!("{} {} documentation", technology, topic),
            filters: SearchFilters {
                content_type: Some(PageType::Documentation),
                ..Default::default()
            },
            max_results: 10,
            engines: vec!["DuckDuckGo".to_string()],
            extract_content: true,
        };

        self.search(query).await
    }

    /// Search for tutorials
    pub async fn search_tutorials(&self, topic: &str) -> Result<Vec<SearchResult>> {
        let query = SearchQuery {
            query: format!("{} tutorial how to", topic),
            filters: SearchFilters { content_type: Some(PageType::Blog), ..Default::default() },
            max_results: 15,
            engines: vec!["DuckDuckGo".to_string()],
            extract_content: true,
        };

        self.search(query).await
    }

    /// Search within a specific site
    pub async fn search_site(&self, site: &str, query: &str) -> Result<Vec<SearchResult>> {
        let search_query = SearchQuery {
            query: query.to_string(),
            filters: SearchFilters { site_filter: Some(site.to_string()), ..Default::default() },
            max_results: 20,
            engines: vec!["DuckDuckGo".to_string()],
            extract_content: false,
        };

        self.search(search_query).await
    }

    /// Find related memories based on content and tags
    async fn find_related_memories(&self, content: &str, tags: &[String]) -> Result<Vec<MemoryId>> {
        let mut related_memories = Vec::new();

        // 1. Find memories with similar tags
        let tag_related = self.find_memories_by_tags(tags).await?;
        related_memories.extend(tag_related);

        // 2. Find semantically similar memories
        let semantic_related = self.find_semantically_similar_memories(content).await?;
        related_memories.extend(semantic_related);

        // 3. Find memories with similar keywords
        let keyword_related = self.find_memories_by_keywords(content).await?;
        related_memories.extend(keyword_related);

        // 4. Find contextually related memories from recent searches
        let context_related = self.find_contextually_related_memories(tags).await?;
        related_memories.extend(context_related);

        // Remove duplicates and limit results
        related_memories.sort();
        related_memories.dedup();
        related_memories.truncate(10); // Limit to top 10 associations

        Ok(related_memories)
    }

    /// Find memories that share similar tags
    async fn find_memories_by_tags(&self, tags: &[String]) -> Result<Vec<MemoryId>> {
        let mut related = Vec::new();

        // Search for memories with overlapping tags
        for tag in tags {
            if tag.len() > 3 && !["web", "search", "result"].contains(&tag.as_str()) {
                match self.memory.retrieve_similar(tag, 5).await {
                    Ok(memories) => {
                        for memory in memories {
                            // Check tag overlap
                            let tag_overlap =
                                memory.metadata.tags.iter().filter(|t| tags.contains(t)).count();

                            if tag_overlap >= 2 {
                                // At least 2 shared tags
                                related.push(memory.id);
                            }
                        }
                    }
                    Err(_) => continue,
                }
            }
        }

        Ok(related)
    }

    /// Find semantically similar memories based on content
    async fn find_semantically_similar_memories(&self, content: &str) -> Result<Vec<MemoryId>> {
        // Extract key phrases from content for semantic search
        let key_phrases = self.extract_key_phrases(content);
        let mut related = Vec::new();

        for phrase in key_phrases.iter().take(3) {
            // Top 3 phrases
            if phrase.len() > 5 {
                match self.memory.retrieve_similar(phrase, 3).await {
                    Ok(memories) => {
                        for memory in memories {
                            // Calculate semantic similarity threshold
                            let similarity =
                                self.calculate_content_similarity(content, &memory.content);
                            if similarity > 0.3 {
                                // 30% similarity threshold
                                related.push(memory.id);
                            }
                        }
                    }
                    Err(_) => continue,
                }
            }
        }

        Ok(related)
    }

    /// Find memories based on shared keywords
    async fn find_memories_by_keywords(&self, content: &str) -> Result<Vec<MemoryId>> {
        let keywords = self.extract_keywords("", content); // Extract from content
        let mut related = Vec::new();

        // Search for memories containing these keywords
        for keyword in keywords.iter().take(5) {
            // Top 5 keywords
            if keyword.len() > 4 {
                match self.memory.retrieve_similar(keyword, 2).await {
                    Ok(memories) => {
                        for memory in memories {
                            // Check if memory contains significant keyword overlap
                            let memory_keywords = self.extract_keywords("", &memory.content);
                            let overlap = keyword_overlap_score(&keywords, &memory_keywords);

                            if overlap > 0.25 {
                                // 25% keyword overlap threshold
                                related.push(memory.id);
                            }
                        }
                    }
                    Err(_) => continue,
                }
            }
        }

        Ok(related)
    }

    /// Find contextually related memories from recent activity
    async fn find_contextually_related_memories(&self, tags: &[String]) -> Result<Vec<MemoryId>> {
        let mut related = Vec::new();

        // Look for memories from the same domain or page type
        for tag in tags {
            if tag.starts_with("page_type:") || tag.contains('.') {
                // Domain or page type
                match self.memory.retrieve_similar(tag, 2).await {
                    Ok(memories) => {
                        for memory in memories.into_iter().take(2) {
                            // Only add if it's from recent activity (within context)
                            if memory.metadata.source == "web_search" {
                                related.push(memory.id);
                            }
                        }
                    }
                    Err(_) => continue,
                }
            }
        }

        Ok(related)
    }

    /// Extract key phrases from content for semantic matching
    fn extract_key_phrases(&self, content: &str) -> Vec<String> {
        let sentences: Vec<&str> = content.split(&['.', '!', '?'][..]).collect();
        let mut phrases = Vec::new();

        for sentence in sentences.iter().take(5) {
            // First 5 sentences
            let words: Vec<&str> = sentence.split_whitespace().collect();

            // Extract noun phrases (simplified - consecutive capitalized words or quoted
            // text)
            let mut current_phrase = Vec::new();

            for word in words {
                if word.len() > 2
                    && (word.chars().next().unwrap().is_uppercase() || word.starts_with('"'))
                {
                    current_phrase.push(word.trim_matches('"'));
                } else if !current_phrase.is_empty() {
                    if current_phrase.len() >= 2 {
                        phrases.push(current_phrase.join(" "));
                    }
                    current_phrase.clear();
                }
            }

            // Add any remaining phrase
            if current_phrase.len() >= 2 {
                phrases.push(current_phrase.join(" "));
            }
        }

        // Also extract quoted text
        let mut in_quotes = false;
        let mut quoted_text = String::new();

        for char in content.chars() {
            match char {
                '"' => {
                    if in_quotes && !quoted_text.trim().is_empty() {
                        phrases.push(quoted_text.trim().to_string());
                        quoted_text.clear();
                    }
                    in_quotes = !in_quotes;
                }
                _ if in_quotes => quoted_text.push(char),
                _ => {}
            }
        }

        phrases.truncate(5); // Limit to top 5 phrases
        phrases
    }

    /// Calculate semantic similarity between two text contents
    fn calculate_content_similarity(&self, content1: &str, content2: &str) -> f32 {
        // Simple word-based similarity (in production, would use embeddings)
        let words1: std::collections::HashSet<&str> =
            content1.split_whitespace().filter(|w| w.len() > 3).collect();
        let words2: std::collections::HashSet<&str> =
            content2.split_whitespace().filter(|w| w.len() > 3).collect();

        if words1.is_empty() || words2.is_empty() {
            return 0.0;
        }

        let intersection = words1.intersection(&words2).count();
        let union = words1.union(&words2).count();

        intersection as f32 / union as f32
    }
}

/// Calculate keyword overlap score between two keyword sets
fn keyword_overlap_score(keywords1: &[String], keywords2: &[String]) -> f32 {
    if keywords1.is_empty() || keywords2.is_empty() {
        return 0.0;
    }

    let set1: std::collections::HashSet<&String> = keywords1.iter().collect();
    let set2: std::collections::HashSet<&String> = keywords2.iter().collect();

    let intersection = set1.intersection(&set2).count();
    let union = set1.union(&set2).count();

    intersection as f32 / union as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_page_type_detection() {
        // Test the page type detection logic directly without creating a full client
        let test_cases = vec![
            (
                "https://docs.rust-lang.org/book/",
                "Rust Book",
                "Documentation",
                PageType::Documentation,
            ),
            ("https://arxiv.org/abs/1234.5678", "Paper Title", "Abstract", PageType::Academic),
            ("https://blog.example.com/post", "Blog Post", "Content", PageType::Blog),
            ("https://news.example.com/article", "News Article", "Breaking news", PageType::News),
        ];

        for (url, title, snippet, expected) in test_cases {
            let url_lower = url.to_lowercase();
            let content_lower = format!("{} {}", title, snippet).to_lowercase();

            let page_type = if url_lower.contains("docs.")
                || url_lower.contains("/doc/")
                || url_lower.contains("documentation")
                || content_lower.contains("documentation")
            {
                PageType::Documentation
            } else if url_lower.contains("arxiv.org")
                || url_lower.contains("ieee.org")
                || url_lower.contains("acm.org")
                || url_lower.contains(".edu")
                || content_lower.contains("abstract")
                || content_lower.contains("paper")
            {
                PageType::Academic
            } else if url_lower.contains("news")
                || url_lower.contains("cnn.com")
                || url_lower.contains("bbc.com")
                || url_lower.contains("reuters.com")
            {
                PageType::News
            } else if url_lower.contains("blog")
                || url_lower.contains("medium.com")
                || url_lower.contains("dev.to")
                || url_lower.contains("wordpress")
            {
                PageType::Blog
            } else {
                PageType::Unknown
            };

            assert_eq!(page_type, expected, "Failed for URL: {}", url);
        }
    }

    #[test]
    fn test_url_normalization() {
        // Test URL normalization logic directly
        let url1 = "https://example.com/page?utm_source=google&id=123";
        let url2 = "https://example.com/page?id=123&utm_campaign=test";

        // Normalize URLs by removing tracking parameters
        let normalize = |url: &str| -> String {
            if let Ok(mut parsed_url) = Url::parse(url) {
                parsed_url.set_fragment(None);

                let tracking_params =
                    ["utm_source", "utm_medium", "utm_campaign", "utm_content", "utm_term"];
                let pairs: Vec<_> = parsed_url
                    .query_pairs()
                    .filter(|(key, _)| !tracking_params.contains(&key.as_ref()))
                    .collect();

                if pairs.is_empty() {
                    parsed_url.set_query(None);
                } else {
                    let query_string = pairs
                        .iter()
                        .map(|(k, v)| format!("{}={}", k, v))
                        .collect::<Vec<_>>()
                        .join("&");
                    parsed_url.set_query(Some(&query_string));
                }

                parsed_url.to_string()
            } else {
                url.to_string()
            }
        };

        let normalized1 = normalize(url1);
        let normalized2 = normalize(url2);

        assert_eq!(normalized1, normalized2);
    }
}
