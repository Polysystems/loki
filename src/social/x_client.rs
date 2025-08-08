use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context as AnyhowContext, Result};
use base64::Engine as _;
use base64::engine::general_purpose;
use parking_lot::RwLock;
use reqwest::{Client, header};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::sync::Semaphore;
use tracing::{debug, error, info, warn};

use super::oauth2::{OAuth2Client, OAuth2Config};
use crate::memory::{CognitiveMemory, MemoryMetadata};

/// X (Twitter) API configuration
#[derive(Debug, Clone)]
pub struct XConfig {
    /// Legacy API v1.1 credentials
    pub api_key: Option<String>,
    pub api_secret: Option<String>,
    pub access_token: Option<String>,
    pub access_token_secret: Option<String>,

    /// OAuth2 configuration
    pub oauth2config: Option<OAuth2Config>,

    /// Rate limiting
    pub rate_limit_window: Duration,
    pub max_requests_per_window: usize,
}

impl XConfig {
    /// Load from environment variables
    pub fn from_env() -> Result<Self> {
        // Try OAuth2 first
        let oauth2config = OAuth2Config::from_env().ok();

        // Try legacy API keys
        let api_key = std::env::var("X_API_KEY").ok();
        let api_secret = std::env::var("X_API_SECRET").ok();
        let access_token = std::env::var("X_ACCESS_TOKEN").ok();
        let access_token_secret = std::env::var("X_ACCESS_TOKEN_SECRET").ok();

        if oauth2config.is_none() && api_key.is_none() {
            anyhow::bail!(
                "No X/Twitter authentication configured. Set either OAuth2 or API v1.1 \
                 credentials."
            );
        }

        Ok(Self {
            api_key,
            api_secret,
            access_token,
            access_token_secret,
            oauth2config,
            rate_limit_window: Duration::from_secs(900), // 15 minutes
            max_requests_per_window: 300,                // Twitter's default
        })
    }
}

/// X (Twitter) client with OAuth2 and API v2 support
pub struct XClient {
    /// HTTP client
    client: Client,

    /// Configuration
    config: XConfig,

    /// OAuth2 client (if using OAuth2)
    oauth2_client: Option<OAuth2Client>,

    /// Rate limiter
    rate_limiter: Arc<AdaptiveRateLimiter>,

    /// Memory system for tracking posts
    memory: Arc<CognitiveMemory>,

    /// Post history
    post_history: Arc<RwLock<VecDeque<PostRecord>>>,

    /// Last mention ID for polling
    last_mention_id: Arc<RwLock<Option<String>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PostRecord {
    id: String,
    content: String,
    timestamp: chrono::DateTime<chrono::Utc>,
    engagement: EngagementMetrics,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EngagementMetrics {
    pub likes: u32,
    pub retweets: u32,
    pub replies: u32,
    pub impressions: u32,
}

/// Tweet object from API v2
#[derive(Debug, Deserialize)]
struct Tweet {
    id: String,
    text: String,
    created_at: Option<String>,
    author_id: Option<String>,
    public_metrics: Option<PublicMetrics>,
    referenced_tweets: Option<Vec<ReferencedTweet>>,
}

#[derive(Debug, Deserialize)]
struct PublicMetrics {
    retweet_count: u32,
    reply_count: u32,
    like_count: u32,
    _quote_count: u32,
    impression_count: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct ReferencedTweet {
    #[serde(rename = "type")]
    ref_type: String,
    id: String,
}

#[derive(Debug, Deserialize)]
struct User {
    id: String,
    username: String,
    name: String,
}

#[derive(Debug, Deserialize)]
struct TwitterApiResponse<T> {
    data: Option<T>,
    includes: Option<Includes>,
    errors: Option<Vec<TwitterError>>,
    meta: Option<Meta>,
}

#[derive(Debug, Deserialize)]
struct Includes {
    users: Option<Vec<User>>,
    tweets: Option<Vec<Tweet>>,
}

#[derive(Debug, Deserialize)]
struct Meta {
    result_count: Option<u32>,
    next_token: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TwitterError {
    detail: String,
    title: String,
    #[serde(rename = "type")]
    error_type: String,
}

impl XClient {
    /// Create a new X client
    pub async fn new(config: XConfig, memory: Arc<CognitiveMemory>) -> Result<Self> {
        info!("Initializing X (Twitter) client");

        let client = Client::builder().timeout(Duration::from_secs(30)).build()?;

        let rate_limiter = Arc::new(AdaptiveRateLimiter::new(
            config.max_requests_per_window,
            config.rate_limit_window,
        ));

        // Create OAuth2 client if config is available
        let oauth2_client = config.oauth2config.as_ref().map(|cfg| OAuth2Client::new(cfg.clone()));

        // Try to load saved OAuth2 token if available
        if let Some(oauth2) = &oauth2_client {
            let token_path = std::path::Path::new(".x_oauth_token.json");
            if token_path.exists() {
                if let Err(e) = oauth2.load_token(token_path).await {
                    warn!("Failed to load saved OAuth2 token: {}", e);
                }
            }
        }

        Ok(Self {
            client,
            config,
            oauth2_client,
            rate_limiter,
            memory,
            post_history: Arc::new(RwLock::new(VecDeque::with_capacity(1000))),
            last_mention_id: Arc::new(RwLock::new(None)),
        })
    }

    /// Get authorization header
    async fn get_auth_header(&self) -> Result<(header::HeaderName, header::HeaderValue)> {
        if let Some(oauth2) = &self.oauth2_client {
            // Use OAuth2
            let token = oauth2.get_valid_token().await?;
            Ok((
                header::AUTHORIZATION,
                header::HeaderValue::from_str(&format!("Bearer {}", token.access_token))?,
            ))
        } else if let (
            Some(api_key),
            Some(api_secret),
            Some(access_token),
            Some(access_token_secret),
        ) = (
            &self.config.api_key,
            &self.config.api_secret,
            &self.config.access_token,
            &self.config.access_token_secret,
        ) {
            // Implement proper OAuth 1.0a signing
            let timestamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs()
                .to_string();

            let nonce = self.generate_nonce();

            // Create OAuth parameters
            let mut oauth_params = std::collections::BTreeMap::new();
            oauth_params.insert("oauth_consumer_key", api_key.as_str());
            oauth_params.insert("oauth_nonce", &nonce);
            oauth_params.insert("oauth_signature_method", "HMAC-SHA1");
            oauth_params.insert("oauth_timestamp", &timestamp);
            oauth_params.insert("oauth_token", access_token.as_str());
            oauth_params.insert("oauth_version", "1.0");

            // Generate signature (method and URL will be provided by caller)
            let signature = self.generate_oauth_signature(
                "GET",                     // Default method, will be overridden as needed
                "https://api.twitter.com", // Base URL
                &oauth_params,
                api_secret,
                access_token_secret,
            )?;

            oauth_params.insert("oauth_signature", &signature);

            // Build authorization header
            let auth_header = oauth_params
                .iter()
                .map(|(k, v)| format!("{}=\"{}\"", k, self.percent_encode(v)))
                .collect::<Vec<_>>()
                .join(", ");

            Ok((
                header::AUTHORIZATION,
                header::HeaderValue::from_str(&format!("OAuth {}", auth_header))?,
            ))
        } else {
            anyhow::bail!("No authentication method available")
        }
    }

    /// Post a tweet using API v2
    pub async fn post_tweet(&self, content: &str) -> Result<String> {
        // Check rate limit
        self.rate_limiter.acquire().await?;

        info!("Posting tweet: {}", content);

        let auth = self.get_auth_header().await?;

        let payload = json!({
            "text": content
        });

        let response = self
            .client
            .post("https://api.twitter.com/2/tweets")
            .header(auth.0, auth.1)
            .header(header::CONTENT_TYPE, "application/json")
            .json(&payload)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            anyhow::bail!("Failed to post tweet: {}", error_text);
        }

        let api_response: TwitterApiResponse<Tweet> = response.json().await?;

        if let Some(errors) = api_response.errors {
            anyhow::bail!("Twitter API errors: {:?}", errors);
        }

        let tweet =
            api_response.data.ok_or_else(|| anyhow::anyhow!("No tweet data in response"))?;

        let tweet_id = tweet.id.clone();

        // Record in post history
        {
            let mut history = self.post_history.write();
            history.push_back(PostRecord {
                id: tweet_id.clone(),
                content: content.to_string(),
                timestamp: chrono::Utc::now(),
                engagement: EngagementMetrics::default(),
            });

            // Limit history size
            while history.len() > 1000 {
                history.pop_front();
            }
        }

        // Store in memory
        self.memory
            .store(
                format!("Posted to X: {}", content),
                vec![tweet_id.clone()],
                MemoryMetadata {
                    source: "x_twitter".to_string(),
                    tags: vec!["social_media".to_string(), "post".to_string()],
                    importance: 0.7,
                    associations: vec![],
                    context: Some("X Twitter post".to_string()),
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

        Ok(tweet_id)
    }

    /// Post a thread
    pub async fn post_thread(&self, tweets: Vec<String>) -> Result<Vec<String>> {
        if tweets.is_empty() {
            return Ok(Vec::new());
        }

        info!("Posting thread with {} tweets", tweets.len());

        let mut tweet_ids = Vec::new();
        let mut reply_to_id: Option<String> = None;

        for (i, content) in tweets.iter().enumerate() {
            // Add thread numbering
            let full_content = if tweets.len() > 1 {
                format!("{}/{}: {}", i + 1, tweets.len(), content)
            } else {
                content.clone()
            };

            let tweet_id = if let Some(reply_id) = &reply_to_id {
                self.reply_to_tweet(reply_id, &full_content).await?
            } else {
                self.post_tweet(&full_content).await?
            };

            tweet_ids.push(tweet_id.clone());
            reply_to_id = Some(tweet_id);

            // Small delay between thread posts
            if i < tweets.len() - 1 {
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
        }

        Ok(tweet_ids)
    }

    /// Reply to a tweet
    pub async fn reply_to_tweet(&self, tweet_id: &str, content: &str) -> Result<String> {
        self.rate_limiter.acquire().await?;

        info!("Replying to tweet {} with: {}", tweet_id, content);

        let auth = self.get_auth_header().await?;

        let payload = json!({
            "text": content,
            "reply": {
                "in_reply_to_tweet_id": tweet_id
            }
        });

        let response = self
            .client
            .post("https://api.twitter.com/2/tweets")
            .header(auth.0, auth.1)
            .header(header::CONTENT_TYPE, "application/json")
            .json(&payload)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            anyhow::bail!("Failed to reply to tweet: {}", error_text);
        }

        let api_response: TwitterApiResponse<Tweet> = response.json().await?;
        let tweet =
            api_response.data.ok_or_else(|| anyhow::anyhow!("No tweet data in response"))?;

        let reply_id = tweet.id;

        // Store in memory
        self.memory
            .store(
                format!("Replied to {}: {}", tweet_id, content),
                vec![tweet_id.to_string(), reply_id.clone()],
                MemoryMetadata {
                    source: "x_twitter".to_string(),
                    tags: vec!["social_media".to_string(), "reply".to_string()],
                    importance: 0.6,
                    associations: vec![],
                    context: Some("X Twitter reply".to_string()),
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

        Ok(reply_id)
    }

    /// Get mentions using API v2
    pub async fn get_mentions(&self, since_id: Option<&str>) -> Result<Vec<Mention>> {
        self.rate_limiter.acquire().await?;

        debug!("Fetching mentions since: {:?}", since_id);

        let auth = self.get_auth_header().await?;

        // Get authenticated user ID first
        let user_id = self.get_authenticated_user_id().await?;

        let url = format!("https://api.twitter.com/2/users/{}/mentions", user_id);
        let mut params = vec![
            ("tweet.fields", "created_at,author_id,referenced_tweets"),
            ("expansions", "author_id"),
            ("max_results", "100"),
        ];

        if let Some(since) = since_id {
            params.push(("since_id", since));
        }

        let response = self.client.get(&url).header(auth.0, auth.1).query(&params).send().await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            anyhow::bail!("Failed to get mentions: {}", error_text);
        }

        let api_response: TwitterApiResponse<Vec<Tweet>> = response.json().await?;

        let tweets = api_response.data.unwrap_or_default();
        let users = api_response.includes.and_then(|i| i.users).unwrap_or_default();

        // Create user lookup
        let user_map: HashMap<String, User> =
            users.into_iter().map(|u| (u.id.clone(), u)).collect();

        let mentions: Vec<Mention> = tweets
            .into_iter()
            .map(|tweet| {
                let author = tweet.author_id.as_ref().and_then(|id| user_map.get(id));

                Mention {
                    id: tweet.id,
                    author_id: tweet.author_id.unwrap_or_default(),
                    author_username: author.map(|u| u.username.clone()).unwrap_or_default(),
                    text: tweet.text,
                    created_at: tweet
                        .created_at
                        .and_then(|s| chrono::DateTime::parse_from_rfc3339(&s).ok())
                        .map(|dt| dt.with_timezone(&chrono::Utc))
                        .unwrap_or_else(chrono::Utc::now),
                    in_reply_to: tweet.referenced_tweets.and_then(|refs| {
                        refs.into_iter().find(|r| r.ref_type == "replied_to").map(|r| r.id)
                    }),
                }
            })
            .collect();

        // Update last mention ID
        if let Some(latest) = mentions.first() {
            *self.last_mention_id.write() = Some(latest.id.clone());
        }

        Ok(mentions)
    }

    /// Get authenticated user ID
    pub async fn get_authenticated_user_id(&self) -> Result<String> {
        let auth = self.get_auth_header().await?;

        let response = self
            .client
            .get("https://api.twitter.com/2/users/me")
            .header(auth.0, auth.1)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            anyhow::bail!("Failed to get user info: {}", error_text);
        }

        let api_response: TwitterApiResponse<User> = response.json().await?;
        let user = api_response.data.ok_or_else(|| anyhow::anyhow!("No user data in response"))?;

        Ok(user.id)
    }

    /// Get tweet metrics
    pub async fn get_tweet_metrics(&self, tweet_id: &str) -> Result<EngagementMetrics> {
        self.rate_limiter.acquire().await?;

        let auth = self.get_auth_header().await?;

        let response = self
            .client
            .get(&format!("https://api.twitter.com/2/tweets/{}", tweet_id))
            .header(auth.0, auth.1)
            .query(&[("tweet.fields", "public_metrics")])
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            anyhow::bail!("Failed to get tweet metrics: {}", error_text);
        }

        let api_response: TwitterApiResponse<Tweet> = response.json().await?;
        let tweet =
            api_response.data.ok_or_else(|| anyhow::anyhow!("No tweet data in response"))?;

        if let Some(metrics) = tweet.public_metrics {
            Ok(EngagementMetrics {
                likes: metrics.like_count,
                retweets: metrics.retweet_count,
                replies: metrics.reply_count,
                impressions: metrics.impression_count.unwrap_or(0),
            })
        } else {
            Ok(EngagementMetrics::default())
        }
    }

    /// Upload media using Twitter v1.1 API with comprehensive chunked upload
    /// support Implements sophisticated media processing following
    /// cognitive enhancement principles
    pub async fn upload_media(&self, media_data: &[u8], media_type: &str) -> Result<String> {
        self.rate_limiter.acquire().await?;

        info!("Uploading media of type: {} ({} bytes)", media_type, media_data.len());

        // Validate media type and determine upload strategy
        let upload_strategy = self.determine_upload_strategy(media_data, media_type)?;

        match upload_strategy {
            MediaUploadStrategy::Simple => self.upload_media_simple(media_data, media_type).await,
            MediaUploadStrategy::Chunked => self.upload_media_chunked(media_data, media_type).await,
        }
    }

    /// Determine the optimal upload strategy based on media size and type
    fn determine_upload_strategy(
        &self,
        media_data: &[u8],
        media_type: &str,
    ) -> Result<MediaUploadStrategy> {
        const CHUNKED_THRESHOLD: usize = 5 * 1024 * 1024; // 5MB

        // Always use chunked upload for large files or video
        if media_data.len() > CHUNKED_THRESHOLD || media_type.starts_with("video/") {
            return Ok(MediaUploadStrategy::Chunked);
        }

        // Validate media type for simple upload
        let allowed_simple_types =
            ["image/jpeg", "image/jpg", "image/png", "image/gif", "image/webp"];

        if allowed_simple_types.iter().any(|&t| t == media_type) {
            Ok(MediaUploadStrategy::Simple)
        } else {
            // Use chunked for unsupported simple upload types
            Ok(MediaUploadStrategy::Chunked)
        }
    }

    /// Simple media upload for small images
    async fn upload_media_simple(&self, media_data: &[u8], media_type: &str) -> Result<String> {
        debug!("Using simple upload strategy");

        let auth = self.get_auth_header().await?;

        // Create multipart form
        let form = reqwest::multipart::Form::new().part(
            "media",
            reqwest::multipart::Part::bytes(media_data.to_vec()).mime_str(media_type)?,
        );

        let response = self
            .client
            .post("https://upload.twitter.com/1.1/media/upload.json")
            .header(auth.0, auth.1)
            .multipart(form)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            anyhow::bail!("Failed to upload media: {}", error_text);
        }

        let upload_response: MediaUploadResponse = response.json().await?;

        info!("Media uploaded successfully: {}", upload_response.media_id_string);
        Ok(upload_response.media_id_string)
    }

    /// Chunked media upload for large files
    async fn upload_media_chunked(&self, media_data: &[u8], media_type: &str) -> Result<String> {
        debug!("Using chunked upload strategy");

        // Step 1: Initialize upload
        let media_id = self.initialize_chunked_upload(media_data.len(), media_type).await?;

        // Step 2: Upload chunks
        self.upload_chunks(&media_id, media_data).await?;

        // Step 3: Finalize upload
        self.finalize_chunked_upload(&media_id).await?;

        // Step 4: Check status for async processing (videos)
        if media_type.starts_with("video/") {
            self.wait_for_processing(&media_id).await?;
        }

        info!("Chunked media upload completed: {}", media_id);
        Ok(media_id)
    }

    /// Initialize chunked upload
    async fn initialize_chunked_upload(
        &self,
        total_bytes: usize,
        media_type: &str,
    ) -> Result<String> {
        debug!("Initializing chunked upload: {} bytes", total_bytes);

        let auth = self.get_auth_header().await?;

        let params = [
            ("command", "INIT"),
            ("total_bytes", &total_bytes.to_string()),
            ("media_type", media_type),
        ];

        let response = self
            .client
            .post("https://upload.twitter.com/1.1/media/upload.json")
            .header(auth.0, auth.1)
            .form(&params)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            anyhow::bail!("Failed to initialize chunked upload: {}", error_text);
        }

        let init_response: ChunkedUploadInitResponse = response.json().await?;
        Ok(init_response.media_id_string)
    }

    /// Upload data in chunks
    async fn upload_chunks(&self, media_id: &str, media_data: &[u8]) -> Result<()> {
        const CHUNK_SIZE: usize = 1024 * 1024; // 1MB chunks

        let total_chunks = (media_data.len() + CHUNK_SIZE - 1) / CHUNK_SIZE;
        info!("Uploading {} chunks", total_chunks);

        for (segment_index, chunk) in media_data.chunks(CHUNK_SIZE).enumerate() {
            debug!("Uploading chunk {} of {}", segment_index + 1, total_chunks);

            let auth = self.get_auth_header().await?;

            let form = reqwest::multipart::Form::new()
                .text("command", "APPEND")
                .text("media_id", media_id.to_string())
                .text("segment_index", segment_index.to_string())
                .part("media", reqwest::multipart::Part::bytes(chunk.to_vec()));

            let response = self
                .client
                .post("https://upload.twitter.com/1.1/media/upload.json")
                .header(auth.0, auth.1)
                .multipart(form)
                .send()
                .await?;

            if !response.status().is_success() {
                let error_text = response.text().await?;
                anyhow::bail!("Failed to upload chunk {}: {}", segment_index, error_text);
            }

            // Small delay between chunks to avoid rate limiting
            if segment_index < total_chunks - 1 {
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        }

        debug!("All chunks uploaded successfully");
        Ok(())
    }

    /// Finalize chunked upload
    async fn finalize_chunked_upload(&self, media_id: &str) -> Result<()> {
        debug!("Finalizing chunked upload: {}", media_id);

        let auth = self.get_auth_header().await?;

        let params = [("command", "FINALIZE"), ("media_id", media_id)];

        let response = self
            .client
            .post("https://upload.twitter.com/1.1/media/upload.json")
            .header(auth.0, auth.1)
            .form(&params)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            anyhow::bail!("Failed to finalize upload: {}", error_text);
        }

        let finalize_response: ChunkedUploadFinalizeResponse = response.json().await?;

        if let Some(error) = finalize_response.error {
            anyhow::bail!("Media processing error: {}", error.message);
        }

        debug!("Upload finalized successfully");
        Ok(())
    }

    /// Wait for asynchronous media processing (for videos)
    async fn wait_for_processing(&self, media_id: &str) -> Result<()> {
        info!("Waiting for media processing: {}", media_id);

        let mut attempts = 0;
        const MAX_ATTEMPTS: u32 = 60; // 5 minutes max wait

        loop {
            let auth = self.get_auth_header().await?;

            let params = [("command", "STATUS"), ("media_id", media_id)];

            let response = self
                .client
                .get("https://upload.twitter.com/1.1/media/upload.json")
                .header(auth.0, auth.1)
                .query(&params)
                .send()
                .await?;

            if !response.status().is_success() {
                let error_text = response.text().await?;
                anyhow::bail!("Failed to check processing status: {}", error_text);
            }

            let status_response: MediaProcessingStatusResponse = response.json().await?;

            match status_response.processing_info.as_ref().map(|p| p.state.as_str()) {
                Some("succeeded") => {
                    info!("Media processing completed successfully");
                    return Ok(());
                }
                Some("failed") => {
                    let error_msg = status_response
                        .processing_info
                        .and_then(|p| p.error)
                        .map(|e| e.message)
                        .unwrap_or_else(|| "Unknown processing error".to_string());
                    anyhow::bail!("Media processing failed: {}", error_msg);
                }
                Some("in_progress") => {
                    debug!("Media still processing...");

                    attempts += 1;
                    if attempts >= MAX_ATTEMPTS {
                        anyhow::bail!("Media processing timeout after {} attempts", MAX_ATTEMPTS);
                    }

                    // Use adaptive wait time based on progress
                    let wait_time = status_response
                        .processing_info
                        .and_then(|p| p.check_after_secs)
                        .unwrap_or(5);

                    tokio::time::sleep(Duration::from_secs(wait_time as u64)).await;
                }
                Some(state) => {
                    warn!("Unknown processing state: {}", state);
                    tokio::time::sleep(Duration::from_secs(5)).await;
                }
                None => {
                    // No processing info means it's ready
                    return Ok(());
                }
            }
        }
    }

    /// Post tweet with media attachment
    pub async fn post_tweet_with_media(
        &self,
        content: &str,
        media_data: &[u8],
        media_type: &str,
    ) -> Result<String> {
        info!("Posting tweet with media: {}", content);

        // Upload media first
        let media_id = self.upload_media(media_data, media_type).await?;

        // Post tweet with media ID
        let auth = self.get_auth_header().await?;

        let payload = json!({
            "text": content,
            "media": {
                "media_ids": [media_id]
            }
        });

        let response = self
            .client
            .post("https://api.twitter.com/2/tweets")
            .header(auth.0, auth.1)
            .header(header::CONTENT_TYPE, "application/json")
            .json(&payload)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            anyhow::bail!("Failed to post tweet with media: {}", error_text);
        }

        let api_response: TwitterApiResponse<Tweet> = response.json().await?;
        let tweet =
            api_response.data.ok_or_else(|| anyhow::anyhow!("No tweet data in response"))?;

        let tweet_id = tweet.id.clone();

        // Store in memory with media reference
        self.memory
            .store(
                format!("Posted to X with media: {}", content),
                vec![tweet_id.clone(), media_id],
                MemoryMetadata {
                    source: "x_twitter".to_string(),
                    tags: vec!["social_media".to_string(), "post".to_string(), "media".to_string()],
                    importance: 0.8, // Higher importance for media posts
                    associations: vec![],
                    context: Some("X Twitter media post".to_string()),
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

        Ok(tweet_id)
    }

    /// Validate media before upload
    pub fn validate_media(
        &self,
        media_data: &[u8],
        media_type: &str,
    ) -> Result<MediaValidationResult> {
        let mut result = MediaValidationResult {
            is_valid: true,
            issues: Vec::new(),
            estimated_processing_time: Duration::from_secs(0),
            recommended_compression: None,
        };

        // Check file size limits
        const MAX_IMAGE_SIZE: usize = 5 * 1024 * 1024; // 5MB
        const MAX_VIDEO_SIZE: usize = 512 * 1024 * 1024; // 512MB

        let max_size =
            if media_type.starts_with("video/") { MAX_VIDEO_SIZE } else { MAX_IMAGE_SIZE };

        if media_data.len() > max_size {
            result.is_valid = false;
            result.issues.push(format!(
                "File size {} exceeds maximum {} for {}",
                humanize_bytes(media_data.len()),
                humanize_bytes(max_size),
                media_type
            ));
        }

        // Check media type
        let allowed_types = [
            "image/jpeg",
            "image/jpg",
            "image/png",
            "image/gif",
            "image/webp",
            "video/mp4",
            "video/mov",
            "video/avi",
            "video/quicktime",
        ];

        if !allowed_types.iter().any(|&t| t == media_type) {
            result.is_valid = false;
            result.issues.push(format!("Unsupported media type: {}", media_type));
        }

        // Estimate processing time for videos
        if media_type.starts_with("video/") {
            let size_mb = media_data.len() as f64 / (1024.0 * 1024.0);
            result.estimated_processing_time = Duration::from_secs((size_mb * 2.0) as u64); // ~2 seconds per MB
        }

        // Suggest compression if needed
        if media_data.len() > 1024 * 1024 {
            // > 1MB
            result.recommended_compression = Some(CompressionRecommendation {
                target_size: max_size / 2, // Target 50% of max size
                quality_reduction: 0.85,   // 85% quality
                reason: "Reduce upload time and improve reliability".to_string(),
            });
        }

        Ok(result)
    }

    /// Monitor mentions continuously
    pub async fn monitor_mentions<F>(&self, mut callback: F) -> Result<()>
    where
        F: FnMut(Vec<Mention>) + Send + 'static,
    {
        info!("Starting mention monitoring");

        loop {
            let since_id = self.last_mention_id.read().clone();

            match self.get_mentions(since_id.as_deref()).await {
                Ok(mentions) => {
                    if !mentions.is_empty() {
                        info!("Found {} new mentions", mentions.len());
                        callback(mentions);
                    }
                }
                Err(e) => {
                    error!("Error fetching mentions: {}", e);
                }
            }

            // Wait before next poll
            tokio::time::sleep(Duration::from_secs(60)).await;
        }
    }

    /// Get OAuth2 authorization URL (if using OAuth2)
    pub async fn get_oauth_url(&self, state: &str) -> Result<String> {
        let oauth2 = self.oauth2_client.as_ref().context("OAuth2 not configured")?;

        oauth2.get_authorization_url(state).await
    }

    /// Complete OAuth2 flow with authorization code
    pub async fn complete_oauth(&self, code: &str) -> Result<()> {
        let oauth2 = self.oauth2_client.as_ref().context("OAuth2 not configured")?;

        oauth2.exchange_code(code).await?;

        // Save token for future use
        let token_path = std::path::Path::new(".x_oauth_token.json");
        oauth2.save_token(token_path).await?;

        Ok(())
    }

    /// Generate OAuth 1.0a nonce
    fn generate_nonce(&self) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        std::time::SystemTime::now().hash(&mut hasher);
        rand::random::<u64>().hash(&mut hasher);

        format!("{:x}", hasher.finish())
    }

    /// Generate OAuth 1.0a signature
    fn generate_oauth_signature(
        &self,
        method: &str,
        url: &str,
        oauth_params: &std::collections::BTreeMap<&str, &str>,
        consumer_secret: &str,
        token_secret: &str,
    ) -> Result<String> {
        // Create parameter string
        let param_string = oauth_params
            .iter()
            .map(|(k, v)| format!("{}={}", self.percent_encode(k), self.percent_encode(v)))
            .collect::<Vec<_>>()
            .join("&");

        // Create signature base string
        let signature_base = format!(
            "{}&{}&{}",
            method.to_uppercase(),
            self.percent_encode(url),
            self.percent_encode(&param_string)
        );

        // Create signing key
        let signing_key = format!(
            "{}&{}",
            self.percent_encode(consumer_secret),
            self.percent_encode(token_secret)
        );

        // Generate HMAC-SHA1 signature
        use hmac::{Hmac, Mac};
        use sha1::Sha1;

        type HmacSha1 = Hmac<Sha1>;

        let mut mac = HmacSha1::new_from_slice(signing_key.as_bytes())
            .map_err(|e| anyhow::anyhow!("Failed to create HMAC: {}", e))?;

        mac.update(signature_base.as_bytes());
        let result = mac.finalize();

        // Base64 encode the result
        Ok(general_purpose::STANDARD.encode(result.into_bytes()))
    }

    /// Percent encode for OAuth 1.0a
    fn percent_encode(&self, input: &str) -> String {
        input
            .chars()
            .map(|c| match c {
                'A'..='Z' | 'a'..='z' | '0'..='9' | '-' | '.' | '_' | '~' => c.to_string(),
                _ => format!("%{:02X}", c as u8),
            })
            .collect()
    }

    /// Check if the client is connected (stub implementation)
    pub async fn is_connected(&self) -> Result<bool> {
        // Check if we have valid credentials configured
        Ok(self.config.api_key.is_some() && self.config.api_secret.is_some())
    }
}

impl std::fmt::Debug for XClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("XClient")
            .field("config", &self.config)
            .field("oauth2_client", &"<OAuth2Client>")
            .field("rate_limiter", &"<AdaptiveRateLimiter>")
            .field("memory", &"<CognitiveMemory>")
            .finish()
    }
}

use std::collections::HashMap;
use hmac::KeyInit;

/// Adaptive rate limiter
pub struct AdaptiveRateLimiter {
    /// Maximum requests per window
    max_requests: usize,

    /// Time window
    window: Duration,

    /// Request timestamps
    requests: Arc<RwLock<VecDeque<Instant>>>,

    /// Semaphore for concurrent access
    semaphore: Arc<Semaphore>,
}

impl AdaptiveRateLimiter {
    pub fn new(max_requests: usize, window: Duration) -> Self {
        Self {
            max_requests,
            window,
            requests: Arc::new(RwLock::new(VecDeque::new())),
            semaphore: Arc::new(Semaphore::new(1)),
        }
    }

    pub async fn acquire(&self) -> Result<()> {
        let _permit = self.semaphore.acquire().await?;

        let now = Instant::now();
        let window_start = now - self.window;

        // Clean old requests and determine if we need to wait
        let wait_time = {
            let mut requests = self.requests.write();
            while let Some(&front) = requests.front() {
                if front < window_start {
                    requests.pop_front();
                } else {
                    break;
                }
            }

            // Check if we can make a request
            if requests.len() >= self.max_requests {
                let oldest = requests.front().copied().unwrap_or(now);
                let wait = (oldest + self.window).saturating_duration_since(now);

                if wait > Duration::ZERO {
                    Some(wait)
                } else {
                    requests.push_back(now);
                    None
                }
            } else {
                requests.push_back(now);
                None
            }
        }; // Lock is dropped here

        // Wait if necessary (outside of lock scope)
        if let Some(wait_time) = wait_time {
            info!("Rate limit reached, waiting {:?}", wait_time);
            tokio::time::sleep(wait_time).await;

            // Add request after waiting
            let mut requests = self.requests.write();
            requests.push_back(now + wait_time);
        }

        Ok(())
    }
}

impl std::fmt::Debug for AdaptiveRateLimiter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdaptiveRateLimiter")
            .field("max_requests", &self.max_requests)
            .field("window", &self.window)
            .field("current_requests", &self.requests.try_read().map(|r| r.len()).unwrap_or(0))
            .finish()
    }
}

/// Mention from another user
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mention {
    pub id: String,
    pub author_id: String,
    pub author_username: String,
    pub text: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub in_reply_to: Option<String>,
}

/// Media upload strategy
#[derive(Debug, Clone)]
enum MediaUploadStrategy {
    Simple,  // For small images
    Chunked, // For large files and videos
}

/// Media upload response for simple uploads
#[derive(Debug, Deserialize)]
struct MediaUploadResponse {
    media_id: u64,
    media_id_string: String,
    size: Option<u64>,
    expires_after_secs: Option<u64>,
}

/// Chunked upload initialization response
#[derive(Debug, Deserialize)]
struct ChunkedUploadInitResponse {
    media_id: u64,
    media_id_string: String,
    expires_after_secs: Option<u64>,
}

/// Chunked upload finalization response
#[derive(Debug, Deserialize)]
struct ChunkedUploadFinalizeResponse {
    media_id: u64,
    media_id_string: String,
    size: Option<u64>,
    expires_after_secs: Option<u64>,
    processing_info: Option<ProcessingInfo>,
    error: Option<MediaUploadError>,
}

/// Media processing status response
#[derive(Debug, Deserialize)]
struct MediaProcessingStatusResponse {
    media_id: u64,
    media_id_string: String,
    processing_info: Option<ProcessingInfo>,
    error: Option<MediaUploadError>,
}

/// Processing information for async media processing
#[derive(Debug, Deserialize)]
struct ProcessingInfo {
    state: String, // "pending", "in_progress", "failed", "succeeded"
    check_after_secs: Option<u32>,
    progress_percent: Option<u32>,
    error: Option<MediaUploadError>,
}

/// Media upload error
#[derive(Debug, Deserialize)]
struct MediaUploadError {
    code: u32,
    name: String,
    message: String,
}

/// Media validation result
#[derive(Debug)]
pub struct MediaValidationResult {
    pub is_valid: bool,
    pub issues: Vec<String>,
    pub estimated_processing_time: Duration,
    pub recommended_compression: Option<CompressionRecommendation>,
}

/// Compression recommendation for large media files
#[derive(Debug)]
pub struct CompressionRecommendation {
    pub target_size: usize,
    pub quality_reduction: f64, // 0.0 to 1.0
    pub reason: String,
}

/// Utility function to humanize byte sizes
fn humanize_bytes(bytes: usize) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];

    if bytes == 0 {
        return "0 B".to_string();
    }

    let mut size = bytes as f64;
    let mut unit_index = 0;

    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }

    if unit_index == 0 {
        format!("{} {}", bytes, UNITS[unit_index])
    } else {
        format!("{:.1} {}", size, UNITS[unit_index])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_rate_limiter() {
        let limiter = AdaptiveRateLimiter::new(2, Duration::from_millis(100));

        // First two should succeed immediately
        limiter.acquire().await.unwrap();
        limiter.acquire().await.unwrap();

        // Third should wait
        let start = Instant::now();
        limiter.acquire().await.unwrap();
        let elapsed = start.elapsed();

        assert!(elapsed >= Duration::from_millis(90)); // Allow some margin
    }
}
