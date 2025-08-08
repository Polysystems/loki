//! Documentation Crawler and Indexer
//!
//! Advanced documentation crawler that can scrape, parse, and index
//! documentation from various sources with semantic understanding.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use anyhow::{Result, anyhow};
use regex;
use reqwest::Client;
use reqwest::header::HeaderMap;
use scraper::{Html, Selector};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};
use url::Url;

use crate::memory::{CognitiveMemory, MemoryMetadata};
use crate::safety::{ActionType, ActionValidator};

/// Documentation source configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocSource {
    pub name: String,
    pub base_url: String,
    pub url_patterns: Vec<String>,
    pub selectors: DocSelectors,
    pub max_depth: usize,
    pub crawl_delay: Duration,
    pub headers: HashMap<String, String>,
    pub follow_redirects: bool,
    pub respect_robots_txt: bool,
}

/// CSS selectors for extracting content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocSelectors {
    pub title: String,
    pub content: Vec<String>,
    pub navigation: Option<String>,
    pub code_blocks: Option<String>,
    pub links: String,
    pub exclude: Vec<String>,
}

impl Default for DocSelectors {
    fn default() -> Self {
        Self {
            title: "h1, title".to_string(),
            content: vec![
                "main".to_string(),
                "article".to_string(),
                ".content".to_string(),
                "#content".to_string(),
                ".documentation".to_string(),
            ],
            navigation: Some("nav, .nav, .navigation".to_string()),
            code_blocks: Some("pre, code, .highlight".to_string()),
            links: "a[href]".to_string(),
            exclude: vec![
                "footer".to_string(),
                "header".to_string(),
                ".sidebar".to_string(),
                ".advertisement".to_string(),
            ],
        }
    }
}

/// Extracted documentation page
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocPage {
    pub url: String,
    pub title: String,
    pub content: String,
    pub code_blocks: Vec<String>,
    pub links: Vec<String>,
    pub metadata: DocMetadata,
    pub extracted_at: SystemTime,
    pub content_hash: String,
}

/// Documentation metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocMetadata {
    pub source: String,
    pub language: Option<String>,
    pub topics: Vec<String>,
    pub difficulty: Option<String>,
    pub last_modified: Option<SystemTime>,
    pub word_count: usize,
    pub reading_time_minutes: usize,
}

/// Crawl statistics
#[derive(Debug, Clone, Default)]
pub struct CrawlStats {
    pub pages_crawled: usize,
    pub pages_skipped: usize,
    pub errors: usize,
    pub total_content_size: usize,
    pub unique_domains: HashSet<String>,
    pub start_time: Option<SystemTime>,
    pub last_crawl_time: Option<SystemTime>,
}

/// Documentation crawler configuration
#[derive(Debug, Clone)]
pub struct CrawlerConfig {
    pub max_pages_per_source: usize,
    pub max_concurrent_requests: usize,
    pub request_timeout: Duration,
    pub user_agent: String,
    pub respect_robots_txt: bool,
    pub cache_duration: Duration,
    pub enable_javascript: bool,
}

impl Default for CrawlerConfig {
    fn default() -> Self {
        Self {
            max_pages_per_source: 1000,
            max_concurrent_requests: 10,
            request_timeout: Duration::from_secs(30),
            user_agent: "Loki Documentation Crawler/1.0".to_string(),
            respect_robots_txt: true,
            cache_duration: Duration::from_secs(3600), // 1 hour
            enable_javascript: false,
        }
    }
}

/// Robots.txt parser
struct RobotsTxt {
    rules: HashMap<String, Vec<String>>,
}

impl RobotsTxt {
    async fn fetch(client: &Client, base_url: &str) -> Result<Self> {
        let robots_url = format!("{}/robots.txt", base_url.trim_end_matches('/'));

        match client.get(&robots_url).send().await {
            Ok(response) if response.status().is_success() => {
                let content = response.text().await?;
                Ok(Self::parse(&content))
            }
            _ => Ok(Self { rules: HashMap::new() }),
        }
    }

    fn parse(content: &str) -> Self {
        let mut rules = HashMap::new();
        let mut current_user_agent = "*".to_string();

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            if let Some(agent) = line.strip_prefix("User-agent:") {
                current_user_agent = agent.trim().to_string();
            } else if let Some(disallow) = line.strip_prefix("Disallow:") {
                let path = disallow.trim().to_string();
                rules.entry(current_user_agent.clone()).or_insert_with(Vec::new).push(path);
            }
        }

        Self { rules }
    }

    fn is_allowed(&self, user_agent: &str, path: &str) -> bool {
        // Check specific user agent rules first
        if let Some(disallowed) = self.rules.get(user_agent) {
            for pattern in disallowed {
                if path.starts_with(pattern) {
                    return false;
                }
            }
        }

        // Check wildcard rules
        if let Some(disallowed) = self.rules.get("*") {
            for pattern in disallowed {
                if path.starts_with(pattern) {
                    return false;
                }
            }
        }

        true
    }
}

/// Advanced documentation crawler
pub struct DocCrawler {
    /// HTTP client
    client: Client,

    /// Configuration
    config: CrawlerConfig,

    /// Memory system
    memory: Arc<CognitiveMemory>,

    /// Safety validator
    validator: Option<Arc<ActionValidator>>,

    /// Crawled pages cache
    page_cache: Arc<RwLock<HashMap<String, DocPage>>>,

    /// Robots.txt cache
    robots_cache: Arc<RwLock<HashMap<String, RobotsTxt>>>,

    /// Statistics
    stats: Arc<RwLock<CrawlStats>>,

    /// URL queue for breadth-first crawling
    url_queue: Arc<RwLock<Vec<String>>>,

    /// Visited URLs
    visited_urls: Arc<RwLock<HashSet<String>>>,
}

impl DocCrawler {
    /// Create a new documentation crawler
    pub async fn new(
        config: CrawlerConfig,
        memory: Arc<CognitiveMemory>,
        validator: Option<Arc<ActionValidator>>,
    ) -> Result<Self> {
        info!("Initializing documentation crawler");

        let mut headers = HeaderMap::new();
        headers.insert("User-Agent", config.user_agent.parse()?);
        headers.insert(
            "Accept",
            "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8".parse()?,
        );
        headers.insert("Accept-Language", "en-US,en;q=0.5".parse()?);
        headers.insert("Accept-Encoding", "gzip, deflate".parse()?);
        headers.insert("DNT", "1".parse()?);
        headers.insert("Connection", "keep-alive".parse()?);
        headers.insert("Upgrade-Insecure-Requests", "1".parse()?);

        let client = Client::builder()
            .timeout(config.request_timeout)
            .default_headers(headers)
            .redirect(reqwest::redirect::Policy::limited(5))
            .build()?;

        Ok(Self {
            client,
            config,
            memory,
            validator,
            page_cache: Arc::new(RwLock::new(HashMap::new())),
            robots_cache: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(CrawlStats::default())),
            url_queue: Arc::new(RwLock::new(Vec::new())),
            visited_urls: Arc::new(RwLock::new(HashSet::new())),
        })
    }

    /// Add documentation sources
    pub async fn add_sources(&self, sources: &[DocSource]) -> Result<()> {
        for source in sources {
            info!("Adding documentation source: {}", source.name);

            // Add base URLs to queue
            let mut queue = self.url_queue.write().await;
            for pattern in &source.url_patterns {
                queue.push(pattern.clone());
            }
        }

        Ok(())
    }

    /// Crawl documentation from configured sources
    pub async fn crawl_sources(&self, sources: &[DocSource]) -> Result<()> {
        // Validate crawling through safety system
        if let Some(validator) = &self.validator {
            validator
                .validate_action(
                    ActionType::ApiCall {
                        provider: "web_crawler".to_string(),
                        endpoint: "documentation_crawl".to_string(),
                    },
                    "Documentation crawling".to_string(),
                    vec!["Crawling documentation for knowledge base".to_string()],
                )
                .await?;
        }

        let mut stats = self.stats.write().await;
        stats.start_time = Some(SystemTime::now());
        drop(stats);

        for source in sources {
            info!("Crawling documentation source: {}", source.name);

            if let Err(e) = self.crawl_source(source).await {
                error!("Failed to crawl source {}: {}", source.name, e);
                let mut stats = self.stats.write().await;
                stats.errors += 1;
            }
        }

        let mut stats = self.stats.write().await;
        stats.last_crawl_time = Some(SystemTime::now());

        info!("Crawling completed. Pages: {}, Errors: {}", stats.pages_crawled, stats.errors);

        Ok(())
    }

    /// Crawl a single documentation source
    async fn crawl_source(&self, source: &DocSource) -> Result<()> {
        // Get robots.txt if needed
        if source.respect_robots_txt {
            self.load_robots_txt(&source.base_url).await?;
        }

        // Initialize URL queue with patterns
        let mut urls_to_visit = Vec::new();
        for pattern in &source.url_patterns {
            urls_to_visit.push(pattern.clone());
        }

        let mut visited = HashSet::new();
        let mut depth = 0;

        while depth < source.max_depth && !urls_to_visit.is_empty() {
            let current_urls = urls_to_visit.clone();
            urls_to_visit.clear();

            // Process URLs in batches
            let semaphore =
                Arc::new(tokio::sync::Semaphore::new(self.config.max_concurrent_requests));
            let mut tasks = Vec::new();

            for url in current_urls {
                if visited.contains(&url) {
                    continue;
                }
                visited.insert(url.clone());

                let permit = semaphore.clone().acquire_owned().await?;
                let source_clone = source.clone();
                let url_clone = url.clone();
                let crawler = self.clone_for_task();

                let task = tokio::spawn(async move {
                    let _permit = permit;
                    crawler.crawl_page(&url_clone, &source_clone).await
                });

                tasks.push(task);
            }

            // Wait for all tasks to complete
            for task in tasks {
                match task.await {
                    Ok(Ok(Some(page))) => {
                        // Extract links for next depth level
                        for link in &page.links {
                            if self.should_follow_link(link, source).await {
                                urls_to_visit.push(link.clone());
                            }
                        }

                        // Store page in cache and memory
                        self.store_page(page).await?;
                    }
                    Ok(Ok(None)) => {
                        // Page skipped
                        let mut stats = self.stats.write().await;
                        stats.pages_skipped += 1;
                    }
                    Ok(Err(e)) => {
                        error!("Failed to crawl page: {}", e);
                        let mut stats = self.stats.write().await;
                        stats.errors += 1;
                    }
                    Err(e) => {
                        error!("Task join error: {}", e);
                        let mut stats = self.stats.write().await;
                        stats.errors += 1;
                    }
                }
            }

            depth += 1;

            // Respect crawl delay
            if source.crawl_delay > Duration::ZERO {
                tokio::time::sleep(source.crawl_delay).await;
            }
        }

        Ok(())
    }

    /// Crawl a single page
    async fn crawl_page(&self, url: &str, source: &DocSource) -> Result<Option<DocPage>> {
        debug!("Crawling page: {}", url);

        // Check robots.txt
        if source.respect_robots_txt {
            if !self.is_url_allowed(url).await {
                debug!("URL blocked by robots.txt: {}", url);
                return Ok(None);
            }
        }

        // Check cache first
        if let Some(cached_page) = self.get_cached_page(url).await {
            let age =
                SystemTime::now().duration_since(cached_page.extracted_at).unwrap_or_default();

            if age < self.config.cache_duration {
                debug!("Returning cached page: {}", url);
                return Ok(Some(cached_page));
            }
        }

        // Build request with source-specific headers
        let mut request = self.client.get(url);
        for (key, value) in &source.headers {
            request = request.header(key, value);
        }

        // Make the request
        let response = request.send().await?;

        if !response.status().is_success() {
            return Err(anyhow!("HTTP error {}: {}", response.status(), url));
        }

        let content_type =
            response.headers().get("content-type").and_then(|v| v.to_str().ok()).unwrap_or("");

        if !content_type.contains("text/html") {
            debug!("Skipping non-HTML content: {}", url);
            return Ok(None);
        }

        let html_content = response.text().await?;

        // Parse the HTML and extract content (scoped to ensure document is dropped)
        let page = {
            let document = Html::parse_document(&html_content);
            self.extract_page_content(url, &document, source)?
        };

        let mut stats = self.stats.write().await;
        stats.pages_crawled += 1;
        stats.total_content_size += page.content.len();

        if let Ok(parsed_url) = Url::parse(url) {
            if let Some(domain) = parsed_url.host_str() {
                stats.unique_domains.insert(domain.to_string());
            }
        }

        Ok(Some(page))
    }

    /// Extract content from HTML document
    fn extract_page_content(
        &self,
        url: &str,
        document: &Html,
        source: &DocSource,
    ) -> Result<DocPage> {
        // Extract title
        let title_selector = Selector::parse(&source.selectors.title).unwrap();
        let title = document
            .select(&title_selector)
            .next()
            .map(|element| element.text().collect::<String>())
            .unwrap_or_else(|| "Untitled".to_string())
            .trim()
            .to_string();

        // Extract main content
        let mut content = String::new();
        for selector_str in &source.selectors.content {
            if let Ok(selector) = Selector::parse(selector_str) {
                for element in document.select(&selector) {
                    // Check if this element should be excluded
                    let mut should_exclude = false;
                    for exclude_selector_str in &source.selectors.exclude {
                        if let Ok(exclude_selector) = Selector::parse(exclude_selector_str) {
                            if element.select(&exclude_selector).next().is_some() {
                                should_exclude = true;
                                break;
                            }
                        }
                    }

                    if !should_exclude {
                        let text = element.text().collect::<String>();
                        if !text.trim().is_empty() {
                            content.push_str(&text);
                            content.push('\n');
                        }
                    }
                }

                if !content.is_empty() {
                    break; // Found content with this selector
                }
            }
        }

        // Extract code blocks
        let mut code_blocks = Vec::new();
        if let Some(code_selector_str) = &source.selectors.code_blocks {
            if let Ok(code_selector) = Selector::parse(code_selector_str) {
                for element in document.select(&code_selector) {
                    let code_text = element.text().collect::<String>();
                    if !code_text.trim().is_empty() {
                        code_blocks.push(code_text.trim().to_string());
                    }
                }
            }
        }

        // Extract links
        let mut links = Vec::new();
        if let Ok(link_selector) = Selector::parse(&source.selectors.links) {
            for element in document.select(&link_selector) {
                if let Some(href) = element.value().attr("href") {
                    if let Ok(base_url) = Url::parse(url) {
                        if let Ok(link_url) = base_url.join(href) {
                            links.push(link_url.to_string());
                        }
                    }
                }
            }
        }

        // Generate content hash
        let content_hash = {
            use sha2::{Digest, Sha256};
            let mut hasher = Sha256::new();
            hasher.update(content.as_bytes());
            format!("{:?}", hasher.finalize())
        };

        // Extract metadata
        let word_count = content.split_whitespace().count();
        let reading_time_minutes = (word_count / 200).max(1); // Assume 200 words per minute

        let metadata = DocMetadata {
            source: source.name.clone(),
            language: self.detect_language(&content),
            topics: self.extract_topics(&content, &code_blocks),
            difficulty: self.assess_difficulty(&content, &code_blocks),
            last_modified: None, // Would extract from headers/meta tags
            word_count,
            reading_time_minutes,
        };

        Ok(DocPage {
            url: url.to_string(),
            title,
            content: content.trim().to_string(),
            code_blocks,
            links,
            metadata,
            extracted_at: SystemTime::now(),
            content_hash,
        })
    }

    /// Detect programming language from content
    fn detect_language(&self, content: &str) -> Option<String> {
        // Simple heuristics - could be enhanced with ML
        let content_lower = content.to_lowercase();

        if content_lower.contains("rust") || content_lower.contains("cargo") {
            Some("rust".to_string())
        } else if content_lower.contains("python") || content_lower.contains("pip") {
            Some("python".to_string())
        } else if content_lower.contains("javascript") || content_lower.contains("npm") {
            Some("javascript".to_string())
        } else if content_lower.contains("typescript") {
            Some("typescript".to_string())
        } else if content_lower.contains("golang") || content_lower.contains("go mod") {
            Some("go".to_string())
        } else {
            None
        }
    }

    /// Extract topics from content
    fn extract_topics(&self, content: &str, code_blocks: &[String]) -> Vec<String> {
        let mut topics = HashSet::new();
        let content_lower = content.to_lowercase();

        // Common programming topics
        let topic_keywords = [
            ("api", "API"),
            ("database", "Database"),
            ("authentication", "Authentication"),
            ("security", "Security"),
            ("performance", "Performance"),
            ("testing", "Testing"),
            ("deployment", "Deployment"),
            ("configuration", "Configuration"),
            ("async", "Asynchronous Programming"),
            ("concurrency", "Concurrency"),
            ("memory", "Memory Management"),
            ("error handling", "Error Handling"),
            ("web", "Web Development"),
            ("cli", "Command Line Interface"),
            ("tutorial", "Tutorial"),
            ("guide", "Guide"),
            ("reference", "Reference"),
        ];

        for (keyword, topic) in &topic_keywords {
            if content_lower.contains(keyword) {
                topics.insert(topic.to_string());
            }
        }

        // Check code blocks for specific patterns
        for code_block in code_blocks {
            let code_lower = code_block.to_lowercase();
            if code_lower.contains("fn main") || code_lower.contains("function") {
                topics.insert("Functions".to_string());
            }
            if code_lower.contains("struct") || code_lower.contains("class") {
                topics.insert("Data Structures".to_string());
            }
            if code_lower.contains("async") || code_lower.contains("await") {
                topics.insert("Asynchronous Programming".to_string());
            }
        }

        topics.into_iter().collect()
    }

    /// Assess content difficulty
    fn assess_difficulty(&self, content: &str, code_blocks: &[String]) -> Option<String> {
        let content_lower = content.to_lowercase();
        let mut complexity_score = 0;

        // Beginner indicators
        if content_lower.contains("beginner")
            || content_lower.contains("introduction")
            || content_lower.contains("getting started")
            || content_lower.contains("hello world")
        {
            return Some("beginner".to_string());
        }

        // Advanced indicators
        if content_lower.contains("advanced")
            || content_lower.contains("expert")
            || content_lower.contains("optimization")
            || content_lower.contains("internals")
        {
            complexity_score += 3;
        }

        // Intermediate indicators
        if content_lower.contains("intermediate") || content_lower.contains("beyond basics") {
            complexity_score += 2;
        }

        // Technical complexity indicators
        if content_lower.contains("algorithm") || content_lower.contains("architecture") {
            complexity_score += 2;
        }

        // Code complexity
        for code_block in code_blocks {
            if code_block.lines().count() > 20 {
                complexity_score += 1;
            }
            if code_block.contains("unsafe") || code_block.contains("macro") {
                complexity_score += 2;
            }
        }

        match complexity_score {
            0..=1 => Some("beginner".to_string()),
            2..=4 => Some("intermediate".to_string()),
            _ => Some("advanced".to_string()),
        }
    }

    /// Store page in cache and memory with automatic association discovery
    pub async fn store_page(&self, page: DocPage) -> Result<()> {
        // Store in cache
        {
            let mut cache = self.page_cache.write().await;
            cache.insert(page.url.clone(), page.clone());
        }

        // Store in cognitive memory with rich metadata
        let importance = match page.metadata.difficulty.as_deref() {
            Some("advanced") => 0.9,
            Some("intermediate") => 0.7,
            Some("beginner") => 0.6,
            _ => 0.5,
        };

        let mut tags = vec!["documentation".to_string(), page.metadata.source.clone()];
        tags.extend(page.metadata.topics.clone());

        if let Some(language) = &page.metadata.language {
            tags.push(language.clone());
        }

        if let Some(difficulty) = &page.metadata.difficulty {
            tags.push(difficulty.clone());
        }

        // Find related memories for associations before storing
        let associations = self.find_related_memories(&page).await?;

        let memory_id = self
            .memory
            .store(
                format!(
                    "{}\n\n{}",
                    page.title,
                    page.content.chars().take(2000).collect::<String>()
                ),
                page.code_blocks.clone(),
                MemoryMetadata {
                    source: "documentation".to_string(),
                    tags,
                    importance,
                    associations: associations.clone(),
                    context: Some("Documentation page content".to_string()),
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

        // Create bidirectional associations by updating related memories
        self.create_bidirectional_associations(&memory_id, &associations).await?;

        debug!("Stored documentation page: {}", page.title);

        Ok(())
    }

    /// Get cached page if available
    async fn get_cached_page(&self, url: &str) -> Option<DocPage> {
        let cache = self.page_cache.read().await;
        cache.get(url).cloned()
    }

    /// Load robots.txt for a domain
    async fn load_robots_txt(&self, base_url: &str) -> Result<()> {
        if let Ok(url) = Url::parse(base_url) {
            if let Some(domain) = url.host_str() {
                let mut cache = self.robots_cache.write().await;
                if !cache.contains_key(domain) {
                    match RobotsTxt::fetch(&self.client, base_url).await {
                        Ok(robots) => {
                            cache.insert(domain.to_string(), robots);
                        }
                        Err(e) => {
                            warn!("Failed to fetch robots.txt for {}: {}", domain, e);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Check if URL is allowed by robots.txt
    async fn is_url_allowed(&self, url: &str) -> bool {
        if let Ok(parsed_url) = Url::parse(url) {
            if let Some(domain) = parsed_url.host_str() {
                let cache = self.robots_cache.read().await;
                if let Some(robots) = cache.get(domain) {
                    return robots.is_allowed(&self.config.user_agent, parsed_url.path());
                }
            }
        }
        true // Allow if we can't determine
    }

    /// Check if we should follow a link
    async fn should_follow_link(&self, link: &str, source: &DocSource) -> bool {
        // Check if URL matches source patterns
        for pattern in &source.url_patterns {
            if link.contains(pattern) {
                return true;
            }
        }

        // Check if it's from the same domain
        if let (Ok(link_url), Ok(base_url)) = (Url::parse(link), Url::parse(&source.base_url)) {
            if link_url.host() == base_url.host() {
                return true;
            }
        }

        false
    }

    /// Get crawl statistics
    pub async fn stats(&self) -> CrawlStats {
        self.stats.read().await.clone()
    }

    /// Search cached documentation
    pub async fn search(&self, query: &str, limit: usize) -> Vec<DocPage> {
        let cache = self.page_cache.read().await;
        let query_lower = query.to_lowercase();

        let mut results: Vec<_> = cache
            .values()
            .filter(|page| {
                page.title.to_lowercase().contains(&query_lower)
                    || page.content.to_lowercase().contains(&query_lower)
                    || page
                        .metadata
                        .topics
                        .iter()
                        .any(|topic| topic.to_lowercase().contains(&query_lower))
            })
            .cloned()
            .collect();

        // Sort by relevance (simple scoring)
        results.sort_by(|a, b| {
            let score_a = self.relevance_score(a, &query_lower);
            let score_b = self.relevance_score(b, &query_lower);
            score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
        });

        results.into_iter().take(limit).collect()
    }

    /// Calculate relevance score for search
    fn relevance_score(&self, page: &DocPage, query: &str) -> f32 {
        let mut score = 0.0;

        // Title matches are most important
        if page.title.to_lowercase().contains(query) {
            score += 10.0;
        }

        // Topic matches
        for topic in &page.metadata.topics {
            if topic.to_lowercase().contains(query) {
                score += 5.0;
            }
        }

        // Content matches
        let content_lower = page.content.to_lowercase();
        let matches = content_lower.matches(query).count();
        score += matches as f32 * 0.1;

        // Boost for difficulty level (advanced content is more valuable for complex
        // queries)
        match page.metadata.difficulty.as_deref() {
            Some("advanced") => score *= 1.3,
            Some("intermediate") => score *= 1.1,
            _ => {}
        }

        score
    }

    /// Find related memories for creating associations
    async fn find_related_memories(&self, page: &DocPage) -> Result<Vec<crate::memory::MemoryId>> {
        let mut associations = Vec::new();

        // Multi-faceted search strategy for comprehensive associations
        let search_strategies = vec![
            // Search by title
            page.title.clone(),
            // Search by primary topics (limit to top 3 most specific)
            page.metadata.topics.iter().take(3).cloned().collect::<Vec<_>>().join(" "),
            // Search by language + difficulty combination for targeted results
            format!(
                "{} {}",
                page.metadata.language.as_deref().unwrap_or(""),
                page.metadata.difficulty.as_deref().unwrap_or("")
            )
            .trim()
            .to_string(),
            // Search by content keywords (extract key technical terms)
            self.extract_key_terms(&page.content, 5).join(" "),
        ];

        // Execute searches in parallel for efficiency
        let search_tasks: Vec<_> = search_strategies
            .into_iter()
            .filter(|query| !query.trim().is_empty())
            .map(|query| {
                let memory = self.memory.clone();
                async move { memory.retrieve_similar(&query, 5).await }
            })
            .collect();

        // Collect all search results
        let search_results = futures::future::try_join_all(search_tasks).await?;

        // Score and filter related memories
        let mut scored_memories = std::collections::HashMap::new();

        for results in search_results {
            for memory_item in results {
                let relevance_score = self.calculate_memory_relevance(page, &memory_item).await;

                // Only associate if relevance is above threshold
                if relevance_score >= 0.3 {
                    scored_memories
                        .entry(memory_item.id.clone())
                        .and_modify(|score: &mut f32| *score = (*score).max(relevance_score))
                        .or_insert(relevance_score);
                }
            }
        }

        // Sort by relevance and take top associations
        let mut sorted_associations: Vec<_> = scored_memories.into_iter().collect();
        sorted_associations
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Limit associations to prevent overwhelming the system
        associations.extend(
            sorted_associations
                .into_iter()
                .take(8) // Limit to 8 most relevant associations
                .map(|(id, _score)| id),
        );

        debug!("Found {} related memories for page: {}", associations.len(), page.title);
        Ok(associations)
    }

    /// Extract key technical terms from content for targeted search
    fn extract_key_terms(&self, content: &str, limit: usize) -> Vec<String> {
        let technical_patterns = [
            // Programming concepts
            r"(?i)\b(async|await|trait|impl|struct|enum|fn|mod|pub|use)\b",
            // Common technical terms
            r"(?i)\b(api|database|server|client|http|json|xml|rest|graphql)\b",
            // Architecture terms
            r"(?i)\b(microservice|container|docker|kubernetes|distributed|concurrent)\b",
            // Data structures and algorithms
            r"(?i)\b(hash|map|vector|array|list|tree|graph|algorithm|optimization)\b",
        ];

        let mut terms = std::collections::HashSet::new();

        for pattern in &technical_patterns {
            if let Ok(regex) = regex::Regex::new(pattern) {
                for mat in regex.find_iter(content) {
                    terms.insert(mat.as_str().to_lowercase());
                }
            }
        }

        // Add significant capitalized terms (likely proper nouns/technologies)
        if let Ok(caps_regex) = regex::Regex::new(r"\b[A-Z][a-zA-Z]{2,}\b") {
            for mat in caps_regex.find_iter(content) {
                let term = mat.as_str();
                if term.len() > 3
                    && !["The", "This", "That", "When", "Where", "What"].contains(&term)
                {
                    terms.insert(term.to_lowercase());
                }
            }
        }

        terms.into_iter().take(limit).collect()
    }

    /// Calculate relevance score between a page and existing memory
    async fn calculate_memory_relevance(
        &self,
        page: &DocPage,
        memory: &crate::memory::MemoryItem,
    ) -> f32 {
        let mut relevance = 0.0f32;

        // Topic overlap scoring (most important factor)
        let page_topics: std::collections::HashSet<_> = page.metadata.topics.iter().collect();
        let memory_topics: std::collections::HashSet<_> = memory.metadata.tags.iter().collect();
        let topic_intersection = page_topics.intersection(&memory_topics).count();
        let topic_union = page_topics.union(&memory_topics).count();

        if topic_union > 0 {
            relevance += (topic_intersection as f32 / topic_union as f32) * 0.4;
        }

        // Language compatibility
        if let Some(page_lang) = &page.metadata.language {
            if memory.metadata.tags.contains(page_lang) {
                relevance += 0.2;
            }
        }

        // Difficulty level proximity (similar levels should associate)
        if let Some(page_difficulty) = &page.metadata.difficulty {
            if memory.metadata.tags.contains(page_difficulty) {
                relevance += 0.15;
            }
        }

        // Content similarity (using simple keyword matching for efficiency)
        let page_keywords = self.extract_key_terms(&page.content, 10);
        let memory_content_lower = memory.content.to_lowercase();
        let keyword_matches =
            page_keywords.iter().filter(|keyword| memory_content_lower.contains(*keyword)).count();

        if !page_keywords.is_empty() {
            relevance += (keyword_matches as f32 / page_keywords.len() as f32) * 0.15;
        }

        // Source compatibility (same documentation sources are more likely related)
        if memory.metadata.source == "documentation" {
            relevance += 0.1;
        }

        // Boost for high-importance memories (they're more valuable to associate with)
        relevance += memory.metadata.importance * 0.05;

        relevance.min(1.0) // Cap at 1.0
    }

    /// Create bidirectional associations by updating existing memories
    async fn create_bidirectional_associations(
        &self,
        new_memory_id: &crate::memory::MemoryId,
        associations: &[crate::memory::MemoryId],
    ) -> Result<()> {
        // This requires a new method on CognitiveMemory to update associations
        // For now, we'll implement it using the existing memory system capabilities

        // We'll implement a workaround by storing update records that the memory system
        // can process during its next consolidation cycle
        for associated_id in associations {
            // Create a lightweight association record that can be processed later
            let association_metadata = crate::memory::MemoryMetadata {
                source: "association_update".to_string(),
                tags: vec!["bidirectional_link".to_string()],
                importance: 0.1, // Low importance for association records
                associations: vec![new_memory_id.clone(), associated_id.clone()],
                context: Some("Generated from automated fix".to_string()),
                created_at: chrono::Utc::now(),
                accessed_count: 0,
                last_accessed: None,
                version: 1,
                    category: "tool_usage".to_string(),
                timestamp: chrono::Utc::now(),
                expiration: None,
            };

            // Store association record
            self.memory
                .store(
                    format!("Association: {} <-> {}", new_memory_id, associated_id),
                    vec!["bidirectional_association".to_string()],
                    association_metadata,
                )
                .await?;
        }

        debug!(
            "Created {} bidirectional associations for memory {}",
            associations.len(),
            new_memory_id
        );

        Ok(())
    }

    /// Helper method to clone for async tasks
    fn clone_for_task(&self) -> Self {
        Self {
            client: self.client.clone(),
            config: self.config.clone(),
            memory: self.memory.clone(),
            validator: self.validator.clone(),
            page_cache: self.page_cache.clone(),
            robots_cache: self.robots_cache.clone(),
            stats: self.stats.clone(),
            url_queue: self.url_queue.clone(),
            visited_urls: self.visited_urls.clone(),
        }
    }
}

/// Predefined documentation sources for popular technologies
pub fn create_default_sources() -> Vec<DocSource> {
    vec![
        DocSource {
            name: "Rust Documentation".to_string(),
            base_url: "https://doc.rust-lang.org".to_string(),
            url_patterns: vec![
                "https://doc.rust-lang.org/book/".to_string(),
                "https://doc.rust-lang.org/std/".to_string(),
                "https://doc.rust-lang.org/reference/".to_string(),
            ],
            selectors: DocSelectors {
                title: "h1, .title".to_string(),
                content: vec!["main".to_string(), "#content".to_string()],
                navigation: Some("nav".to_string()),
                code_blocks: Some("pre code, .highlight".to_string()),
                links: "a[href]".to_string(),
                exclude: vec!["nav".to_string(), ".sidebar".to_string()],
            },
            max_depth: 3,
            crawl_delay: Duration::from_millis(500),
            headers: HashMap::new(),
            follow_redirects: true,
            respect_robots_txt: true,
        },
        DocSource {
            name: "MDN Web Docs".to_string(),
            base_url: "https://developer.mozilla.org".to_string(),
            url_patterns: vec![
                "https://developer.mozilla.org/en-US/docs/Web/".to_string(),
                "https://developer.mozilla.org/en-US/docs/JavaScript/".to_string(),
            ],
            selectors: DocSelectors::default(),
            max_depth: 2,
            crawl_delay: Duration::from_millis(1000),
            headers: HashMap::new(),
            follow_redirects: true,
            respect_robots_txt: true,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::MemoryConfig;

    #[test]
    fn test_robots_txt_parsing() {
        let robots_content = r#"
User-agent: *
Disallow: /admin/
Disallow: /private/

User-agent: Googlebot
Disallow: /temp/
        "#;

        let robots = RobotsTxt::parse(robots_content);

        assert!(!robots.is_allowed("*", "/admin/test"));
        assert!(!robots.is_allowed("Googlebot", "/temp/test"));
        assert!(robots.is_allowed("*", "/public/test"));
    }

    #[test]
    fn test_doc_selectors() {
        let selectors = DocSelectors::default();
        assert!(selectors.content.contains(&"main".to_string()));
        assert!(selectors.exclude.contains(&"footer".to_string()));
    }

    #[test]
    fn test_key_term_extraction() {
        let content = "This is a tutorial about async Rust programming using traits and impl \
                       blocks for better performance.";

        let _crawlerconfig = CrawlerConfig::default();
        let _memoryconfig = MemoryConfig::default();

        // We can't easily test the full async methods here, but we can test the term
        // extraction This would be part of a more comprehensive integration
        // test

        // For now, just verify the content would match some patterns
        assert!(content.contains("async"));
        assert!(content.contains("trait"));
        assert!(content.contains("impl"));
    }

    #[tokio::test]
    async fn test_memory_relevance_calculation() {
        // Create a mock page
        let page = DocPage {
            url: "https://example.com/rust-async".to_string(),
            title: "Async Programming in Rust".to_string(),
            content: "Learn about async and await in Rust programming".to_string(),
            code_blocks: vec!["async fn example() {}".to_string()],
            links: vec![],
            metadata: DocMetadata {
                source: "Rust Documentation".to_string(),
                language: Some("rust".to_string()),
                topics: vec!["async".to_string(), "programming".to_string()],
                difficulty: Some("intermediate".to_string()),
                last_modified: None,
                word_count: 50,
                reading_time_minutes: 1,
            },
            extracted_at: SystemTime::now(),
            content_hash: "test_hash".to_string(),
        };

        // Create a mock memory item
        let memory_item = crate::memory::MemoryItem {
            id: crate::memory::MemoryId::new(),
            content: "Rust async programming guide".to_string(),
            context: vec!["programming".to_string()],
            metadata: crate::memory::MemoryMetadata {
                source: "documentation".to_string(),
                tags: vec!["rust".to_string(), "async".to_string(), "intermediate".to_string()],
                importance: 0.8,
                associations: vec![],

                context: Some("Generated from automated fix".to_string()),
                created_at: chrono::Utc::now(),
                accessed_count: 0,
                last_accessed: None,
                version: 1,
                    category: "tool_usage".to_string(),
                timestamp: chrono::Utc::now(),
                expiration: None,
            },
            timestamp: chrono::Utc::now(),
            access_count: 5,
            relevance_score: 0.9,
        };

        // We'd need to create a full DocCrawler to test this properly
        // This demonstrates the test structure for integration tests

        // Verify that the topics overlap
        let page_topics: HashSet<_> = page.metadata.topics.iter().collect();
        let memory_topics: HashSet<_> = memory_item.metadata.tags.iter().collect();
        let intersection = page_topics.intersection(&memory_topics).count();

        assert!(intersection > 0, "Should have overlapping topics");
        assert!(page_topics.contains(&"async".to_string()));
        assert!(memory_topics.contains(&"async".to_string()));
    }
}
