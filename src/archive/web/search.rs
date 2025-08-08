use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context as AnyhowContext, Result};
use parking_lot::RwLock;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use crate::memory::{CognitiveMemory, MemoryMetadata};

/// Search result from web
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub title: String,
    pub url: String,
    pub snippet: String,
    pub source: SearchEngine,
    pub relevance_score: f32,
    pub published_date: Option<chrono::DateTime<chrono::Utc>>,
    pub domain: String,
}

/// Search engine source
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SearchEngine {
    DuckDuckGo,
    Brave,
    SearXNG,
    GoogleCustom,
    Bing,
    ArXiv,
    StackOverflow,
    GitHubCode,
}

/// Search options
#[derive(Debug, Clone)]
pub struct SearchOptions {
    pub query: String,
    pub max_results: usize,
    pub time_range: Option<TimeRange>,
    pub language: Option<String>,
    pub safe_search: bool,
    pub engines: Vec<SearchEngine>,
}

#[derive(Debug, Clone, Copy)]
pub enum TimeRange {
    Day,
    Week,
    Month,
    Year,
    AllTime,
}

/// Web search client with multiple engine support
pub struct WebSearchClient {
    /// HTTP client
    client: Client,

    /// Available search engines
    engines: HashMap<SearchEngine, Box<dyn SearchProvider>>,

    /// Memory system for caching results
    memory: Arc<CognitiveMemory>,

    /// Rate limiter
    rate_limiter: Arc<RwLock<HashMap<SearchEngine, EngineRateLimiter>>>,

    /// API keys
    api_keys: HashMap<SearchEngine, String>,
}

impl WebSearchClient {
    /// Create a new web search client
    pub async fn new(memory: Arc<CognitiveMemory>) -> Result<Self> {
        info!("Initializing web search client");

        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .user_agent("Loki/1.0 (Shapeshifting Autonomous AI; +https://github.com/loki)")
            .build()?;

        let mut engines: HashMap<SearchEngine, Box<dyn SearchProvider>> = HashMap::new();
        engines.insert(SearchEngine::DuckDuckGo, Box::new(DuckDuckGoProvider));
        engines.insert(SearchEngine::SearXNG, Box::new(SearXNGProvider::new("https://searx.be")));

        // Load API keys from environment
        let mut api_keys = HashMap::new();
        if let Ok(brave_key) = std::env::var("BRAVE_SEARCH_API_KEY") {
            api_keys.insert(SearchEngine::Brave, brave_key);
            engines.insert(SearchEngine::Brave, Box::new(BraveSearchProvider));
        }
        if let Ok(google_key) = std::env::var("GOOGLE_CUSTOM_SEARCH_API_KEY") {
            api_keys.insert(SearchEngine::GoogleCustom, google_key);
            engines.insert(SearchEngine::GoogleCustom, Box::new(GoogleCustomSearchProvider));
        }
        if let Ok(bing_key) = std::env::var("BING_API_KEY") {
            api_keys.insert(SearchEngine::Bing, bing_key);
            engines.insert(SearchEngine::Bing, Box::new(BingSearchProvider));
        }
        if let Ok(stack_key) = std::env::var("STACK_EXCHANGE_KEY") {
            api_keys.insert(SearchEngine::StackOverflow, stack_key);
            engines.insert(SearchEngine::StackOverflow, Box::new(StackOverflowProvider));
        }

        // Engines that don't require API keys
        engines.insert(SearchEngine::ArXiv, Box::new(ArXivProvider));
        engines.insert(SearchEngine::GitHubCode, Box::new(GitHubCodeProvider));

        // Initialize rate limiters
        let mut rate_limiter = HashMap::new();
        for engine in engines.keys() {
            rate_limiter.insert(*engine, EngineRateLimiter::new(*engine));
        }

        Ok(Self {
            client,
            engines,
            memory,
            rate_limiter: Arc::new(RwLock::new(rate_limiter)),
            api_keys,
        })
    }

    /// Search the web with given options
    pub async fn search(&self, options: SearchOptions) -> Result<Vec<SearchResult>> {
        info!("Searching web for: {}", options.query);

        // Check cache first
        let cache_key = format!("web_search:{}", options.query);
        let cached = self.memory.retrieve_by_key(&cache_key).await?;

        if let Some(cached_result) = cached {
            if let Ok(results) = serde_json::from_str::<Vec<SearchResult>>(&cached_result.content) {
                debug!("Using cached search results");
                return Ok(results);
            }
        }

        // Determine which engines to use
        let engines_to_use = if options.engines.is_empty() {
            self.engines.keys().copied().collect()
        } else {
            options.engines.clone()
        };

        // Search across multiple engines concurrently
        let mut all_results = Vec::new();
        let mut search_futures = Vec::new();

        for engine in engines_to_use {
            if let Some(provider) = self.engines.get(&engine) {
                // Check rate limit
                let can_search = {
                    let limiters = self.rate_limiter.read();
                    limiters.get(&engine).map(|l| l.can_request()).unwrap_or(true)
                };

                if can_search {
                    let client = self.client.clone();
                    let api_key = self.api_keys.get(&engine).cloned();
                    let provider = provider.as_ref();
                    let opts = options.clone();

                    search_futures.push(async move {
                        provider.search(&client, &opts, api_key.as_deref()).await
                    });

                    // Record request
                    {
                        let mut limiters = self.rate_limiter.write();
                        if let Some(limiter) = limiters.get_mut(&engine) {
                            limiter.record_request();
                        }
                    }
                }
            }
        }

        // Wait for all searches to complete
        let results = futures::future::join_all(search_futures).await;

        for result in results {
            match result {
                Ok(mut engine_results) => {
                    all_results.append(&mut engine_results);
                }
                Err(e) => {
                    warn!("Search engine error: {}", e);
                }
            }
        }

        // Deduplicate and rank results
        let ranked_results = self.rank_and_deduplicate(all_results);

        // Limit to requested number
        let final_results: Vec<_> = ranked_results.into_iter().take(options.max_results).collect();

        // Cache results
        if !final_results.is_empty() {
            let cache_content = serde_json::to_string(&final_results)?;
            self.memory
                .store(
                    cache_content,
                    vec![options.query.clone()],
                    MemoryMetadata {
                        source: "web_search".to_string(),
                        tags: vec!["search_cache".to_string()],
                        importance: 0.3,
                        associations: vec![],

                        context: Some("Generated from automated fix".to_string()),
                        created_at: chrono::Utc::now(),
                        accessed_count: 0,
                        last_accessed: None,
                        version: 1,
                    category: "general".to_string(),
                        timestamp: chrono::Utc::now(),
                        expiration: None,
                    },
                )
                .await?;
        }

        // Store search event in memory
        self.memory
            .store(
                format!("Web search: {} - Found {} results", options.query, final_results.len()),
                vec![],
                MemoryMetadata {
                    source: "web_search".to_string(),
                    tags: vec!["search_history".to_string()],
                    importance: 0.5,
                    associations: vec![],

                    context: Some("Generated from automated fix".to_string()),
                    created_at: chrono::Utc::now(),
                    accessed_count: 0,
                    last_accessed: None,
                    version: 1,
                    category: "general".to_string(),
                    timestamp: chrono::Utc::now(),
                    expiration: None,
                },
            )
            .await?;

        Ok(final_results)
    }

    /// Rank and deduplicate search results
    fn rank_and_deduplicate(&self, results: Vec<SearchResult>) -> Vec<SearchResult> {
        use std::collections::HashSet;

        let mut seen_urls = HashSet::new();
        let mut unique_results: Vec<SearchResult> = Vec::new();

        // First pass: collect unique URLs
        for result in results {
            if seen_urls.insert(result.url.clone()) {
                unique_results.push(result);
            }
        }

        // Sort by relevance score
        unique_results.sort_by(|a, b| {
            b.relevance_score.partial_cmp(&a.relevance_score).unwrap_or(std::cmp::Ordering::Equal)
        });

        unique_results
    }

    /// Search for recent news/updates
    pub async fn search_news(&self, query: &str, max_results: usize) -> Result<Vec<SearchResult>> {
        let options = SearchOptions {
            query: query.to_string(),
            max_results,
            time_range: Some(TimeRange::Week),
            language: Some("en".to_string()),
            safe_search: true,
            engines: vec![],
        };

        self.search(options).await
    }

    /// Search for technical documentation
    pub async fn search_docs(&self, query: &str) -> Result<Vec<SearchResult>> {
        let options = SearchOptions {
            query: format!("{} documentation reference guide", query),
            max_results: 10,
            time_range: None,
            language: Some("en".to_string()),
            safe_search: true,
            engines: vec![],
        };

        self.search(options).await
    }
}

/// Trait for search providers
#[async_trait::async_trait]
trait SearchProvider: Send + Sync {
    async fn search(
        &self,
        client: &Client,
        options: &SearchOptions,
        api_key: Option<&str>,
    ) -> Result<Vec<SearchResult>>;
}

/// DuckDuckGo search provider
struct DuckDuckGoProvider;

#[async_trait::async_trait]
impl SearchProvider for DuckDuckGoProvider {
    async fn search(
        &self,
        client: &Client,
        options: &SearchOptions,
        _api_key: Option<&str>,
    ) -> Result<Vec<SearchResult>> {
        // DuckDuckGo instant answer API
        let url = format!(
            "https://api.duckduckgo.com/?q={}&format=json&no_html=1",
            urlencoding::encode(&options.query)
        );

        let response = client.get(&url).send().await?;
        let data: serde_json::Value = response.json().await?;

        let mut results = Vec::new();

        // Parse related topics
        if let Some(topics) = data["RelatedTopics"].as_array() {
            for topic in topics.iter().take(options.max_results) {
                if let Some(text) = topic["Text"].as_str() {
                    if let Some(url) = topic["FirstURL"].as_str() {
                        results.push(SearchResult {
                            title: text.chars().take(60).collect::<String>(),
                            url: url.to_string(),
                            snippet: text.to_string(),
                            source: SearchEngine::DuckDuckGo,
                            relevance_score: 0.7,
                            published_date: None,
                            domain: extract_domain(url),
                        });
                    }
                }
            }
        }

        Ok(results)
    }
}

/// SearXNG search provider
struct SearXNGProvider {
    instance_url: String,
}

impl SearXNGProvider {
    fn new(instance_url: &str) -> Self {
        Self { instance_url: instance_url.to_string() }
    }
}

#[async_trait::async_trait]
impl SearchProvider for SearXNGProvider {
    async fn search(
        &self,
        client: &Client,
        options: &SearchOptions,
        _api_key: Option<&str>,
    ) -> Result<Vec<SearchResult>> {
        let url = format!(
            "{}/search?q={}&format=json&safesearch={}",
            self.instance_url,
            urlencoding::encode(&options.query),
            if options.safe_search { "1" } else { "0" }
        );

        let response = client.get(&url).send().await?;
        let data: serde_json::Value = response.json().await?;

        let mut results = Vec::new();

        if let Some(search_results) = data["results"].as_array() {
            for result in search_results.iter().take(options.max_results) {
                results.push(SearchResult {
                    title: result["title"].as_str().unwrap_or("").to_string(),
                    url: result["url"].as_str().unwrap_or("").to_string(),
                    snippet: result["content"].as_str().unwrap_or("").to_string(),
                    source: SearchEngine::SearXNG,
                    relevance_score: result["score"].as_f64().unwrap_or(0.5) as f32,
                    published_date: None,
                    domain: extract_domain(result["url"].as_str().unwrap_or("")),
                });
            }
        }

        Ok(results)
    }
}

/// Brave Search provider
struct BraveSearchProvider;

#[async_trait::async_trait]
impl SearchProvider for BraveSearchProvider {
    async fn search(
        &self,
        client: &Client,
        options: &SearchOptions,
        api_key: Option<&str>,
    ) -> Result<Vec<SearchResult>> {
        let api_key = api_key.context("Brave Search API key required")?;

        let url = "https://api.search.brave.com/res/v1/web/search";
        let response = client
            .get(url)
            .header("X-Subscription-Token", api_key)
            .query(&[("q", options.query.as_str()), ("count", &options.max_results.to_string())])
            .send()
            .await?;

        let data: serde_json::Value = response.json().await?;

        let mut results = Vec::new();

        if let Some(web_results) = data["web"]["results"].as_array() {
            for result in web_results {
                results.push(SearchResult {
                    title: result["title"].as_str().unwrap_or("").to_string(),
                    url: result["url"].as_str().unwrap_or("").to_string(),
                    snippet: result["description"].as_str().unwrap_or("").to_string(),
                    source: SearchEngine::Brave,
                    relevance_score: 0.8,
                    published_date: None,
                    domain: extract_domain(result["url"].as_str().unwrap_or("")),
                });
            }
        }

        Ok(results)
    }
}

/// Google Custom Search provider
struct GoogleCustomSearchProvider;

#[async_trait::async_trait]
impl SearchProvider for GoogleCustomSearchProvider {
    async fn search(
        &self,
        client: &Client,
        options: &SearchOptions,
        api_key: Option<&str>,
    ) -> Result<Vec<SearchResult>> {
        let api_key = api_key.context("Google Custom Search API key required")?;
        let cx = std::env::var("GOOGLE_CUSTOM_SEARCH_CX")
            .context("GOOGLE_CUSTOM_SEARCH_CX environment variable required")?;

        let url = "https://www.googleapis.com/customsearch/v1";
        let response = client
            .get(url)
            .query(&[
                ("key", api_key),
                ("cx", &cx),
                ("q", &options.query),
                ("num", &options.max_results.to_string()),
            ])
            .send()
            .await?;

        let data: serde_json::Value = response.json().await?;

        let mut results = Vec::new();

        if let Some(items) = data["items"].as_array() {
            for item in items {
                results.push(SearchResult {
                    title: item["title"].as_str().unwrap_or("").to_string(),
                    url: item["link"].as_str().unwrap_or("").to_string(),
                    snippet: item["snippet"].as_str().unwrap_or("").to_string(),
                    source: SearchEngine::GoogleCustom,
                    relevance_score: 0.9,
                    published_date: None,
                    domain: extract_domain(item["link"].as_str().unwrap_or("")),
                });
            }
        }

        Ok(results)
    }
}

/// Bing search provider
struct BingSearchProvider;

#[async_trait::async_trait]
impl SearchProvider for BingSearchProvider {
    async fn search(
        &self,
        client: &Client,
        options: &SearchOptions,
        api_key: Option<&str>,
    ) -> Result<Vec<SearchResult>> {
        let api_key = api_key.context("Bing API key required")?;

        let url = "https://api.bing.microsoft.com/v7.0/search";
        let response = client
            .get(url)
            .header("Ocp-Apim-Subscription-Key", api_key)
            .query(&[("q", &options.query)])
            .send()
            .await?;

        let data: serde_json::Value = response.json().await?;

        let mut results = Vec::new();
        if let Some(web_pages) = data["webPages"]["value"].as_array() {
            for page in web_pages.iter().take(options.max_results) {
                results.push(SearchResult {
                    title: page["name"].as_str().unwrap_or("").to_string(),
                    url: page["url"].as_str().unwrap_or("").to_string(),
                    snippet: page["snippet"].as_str().unwrap_or("").to_string(),
                    source: SearchEngine::Bing,
                    relevance_score: 0.8,
                    published_date: None,
                    domain: extract_domain(page["url"].as_str().unwrap_or("")),
                });
            }
        }

        Ok(results)
    }
}

/// ArXiv search provider
struct ArXivProvider;

#[async_trait::async_trait]
impl SearchProvider for ArXivProvider {
    async fn search(
        &self,
        client: &Client,
        options: &SearchOptions,
        _api_key: Option<&str>,
    ) -> Result<Vec<SearchResult>> {
        let url = format!(
            "http://export.arxiv.org/api/query?search_query=all:{}&max_results={}",
            urlencoding::encode(&options.query),
            options.max_results
        );

        let response = client.get(&url).send().await?;
        let xml = response.text().await?;

        let mut results = Vec::new();

        // Simple XML parsing for arXiv
        for entry in xml.split("<entry>").skip(1) {
            if let (Some(title), Some(link), Some(summary)) = (
                extract_xml_value(entry, "title"),
                extract_xml_value(entry, "id"),
                extract_xml_value(entry, "summary"),
            ) {
                results.push(SearchResult {
                    title: title.trim().to_string(),
                    url: link.trim().to_string(),
                    snippet: summary.trim().chars().take(200).collect(),
                    source: SearchEngine::ArXiv,
                    relevance_score: 0.9,
                    published_date: None,
                    domain: "arxiv.org".to_string(),
                });
            }
        }

        Ok(results)
    }
}

/// Stack Overflow search provider
struct StackOverflowProvider;

#[async_trait::async_trait]
impl SearchProvider for StackOverflowProvider {
    async fn search(
        &self,
        client: &Client,
        options: &SearchOptions,
        api_key: Option<&str>,
    ) -> Result<Vec<SearchResult>> {
        let base_url = if let Some(key) = api_key {
            format!("https://api.stackexchange.com/2.3/search?order=desc&sort=relevance&intitle={}&site=stackoverflow&key={}", 
                urlencoding::encode(&options.query), key)
        } else {
            format!("https://api.stackexchange.com/2.3/search?order=desc&sort=relevance&intitle={}&site=stackoverflow", 
                urlencoding::encode(&options.query))
        };

        let response = client.get(&base_url).send().await?;
        let data: serde_json::Value = response.json().await?;

        let mut results = Vec::new();
        if let Some(items) = data["items"].as_array() {
            for item in items.iter().take(options.max_results) {
                results.push(SearchResult {
                    title: item["title"].as_str().unwrap_or("").to_string(),
                    url: item["link"].as_str().unwrap_or("").to_string(),
                    snippet: format!(
                        "Score: {}, Answers: {}",
                        item["score"].as_i64().unwrap_or(0),
                        item["answer_count"].as_i64().unwrap_or(0)
                    ),
                    source: SearchEngine::StackOverflow,
                    relevance_score: 0.85,
                    published_date: None,
                    domain: "stackoverflow.com".to_string(),
                });
            }
        }

        Ok(results)
    }
}

/// GitHub Code search provider
struct GitHubCodeProvider;

#[async_trait::async_trait]
impl SearchProvider for GitHubCodeProvider {
    async fn search(
        &self,
        client: &Client,
        options: &SearchOptions,
        _api_key: Option<&str>,
    ) -> Result<Vec<SearchResult>> {
        let url =
            format!("https://api.github.com/search/code?q={}", urlencoding::encode(&options.query));

        let response = client.get(&url).header("User-Agent", "Loki-AI").send().await?;

        let data: serde_json::Value = response.json().await?;

        let mut results = Vec::new();
        if let Some(items) = data["items"].as_array() {
            for item in items.iter().take(options.max_results) {
                let repo_name = item["repository"]["full_name"].as_str().unwrap_or("");
                results.push(SearchResult {
                    title: item["name"].as_str().unwrap_or("").to_string(),
                    url: item["html_url"].as_str().unwrap_or("").to_string(),
                    snippet: format!("Repository: {}", repo_name),
                    source: SearchEngine::GitHubCode,
                    relevance_score: 0.75,
                    published_date: None,
                    domain: "github.com".to_string(),
                });
            }
        }

        Ok(results)
    }
}

/// Helper function to extract XML values
fn extract_xml_value(xml: &str, tag: &str) -> Option<String> {
    let start_tag = format!("<{}>", tag);
    let end_tag = format!("</{}>", tag);

    if let Some(start) = xml.find(&start_tag) {
        let content_start = start + start_tag.len();
        if let Some(end) = xml[content_start..].find(&end_tag) {
            return Some(xml[content_start..content_start + end].to_string());
        }
    }
    None
}

/// Simple rate limiter for each search engine
struct EngineRateLimiter {
    engine: SearchEngine,
    requests: Vec<std::time::Instant>,
    max_requests_per_minute: usize,
}

impl EngineRateLimiter {
    fn new(engine: SearchEngine) -> Self {
        let max_requests_per_minute = match engine {
            SearchEngine::DuckDuckGo => 60,
            SearchEngine::Brave => 100,
            SearchEngine::SearXNG => 30,
            SearchEngine::GoogleCustom => 100,
            SearchEngine::Bing => 100,
            SearchEngine::ArXiv => 60,
            SearchEngine::StackOverflow => 300, // Stack Exchange has generous limits
            SearchEngine::GitHubCode => 30,     // GitHub has stricter limits for unauthenticated
        };

        Self { engine, requests: Vec::new(), max_requests_per_minute }
    }

    fn can_request(&self) -> bool {
        let now = std::time::Instant::now();
        let one_minute_ago = now - Duration::from_secs(60);

        let recent_requests = self.requests.iter().filter(|&&t| t > one_minute_ago).count();

        recent_requests < self.max_requests_per_minute
    }

    fn record_request(&mut self) {
        let now = std::time::Instant::now();
        self.requests.push(now);

        // Clean old requests
        let one_minute_ago = now - Duration::from_secs(60);
        self.requests.retain(|&t| t > one_minute_ago);
    }
}

/// Extract domain from URL
fn extract_domain(url: &str) -> String {
    url.split('/').nth(2).unwrap_or("").split(':').next().unwrap_or("").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_domain() {
        assert_eq!(extract_domain("https://example.com/path"), "example.com");
        assert_eq!(extract_domain("http://sub.example.com:8080/path"), "sub.example.com");
        assert_eq!(extract_domain("invalid-url"), "");
    }
}
