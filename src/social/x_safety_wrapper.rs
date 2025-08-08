//! Safety-Aware X/Twitter Operations
//!
//! This module provides safety validation for all X/Twitter operations,
//! ensuring that social media actions comply with safety policies.

use std::sync::Arc;

use anyhow::Result;
use tracing::{error, info, warn};

use super::{Mention, XClient, XConsciousness, XConsciousnessConfig};
use crate::safety::{ActionType, ActionValidator, AuditLogger, ResourceMonitor};

/// Safety-aware wrapper for X/Twitter client
pub struct SafeXClient {
    /// The underlying X client
    inner: Arc<XClient>,

    /// Action validator
    validator: Arc<ActionValidator>,

    /// Audit logger
    audit_logger: Arc<AuditLogger>,

    /// Resource monitor
    resource_monitor: Arc<ResourceMonitor>,
}

impl SafeXClient {
    /// Create a new safety-aware X client
    pub fn new(
        x_client: Arc<XClient>,
        validator: Arc<ActionValidator>,
        audit_logger: Arc<AuditLogger>,
        resource_monitor: Arc<ResourceMonitor>,
    ) -> Self {
        Self { inner: x_client, validator, audit_logger, resource_monitor }
    }

    /// Safely post a tweet with validation
    pub async fn safe_post_tweet(&self, content: &str) -> Result<String> {
        // Check rate limits first
        if let Err(limit_exceeded) = self.resource_monitor.check_api_limit("x_twitter").await {
            warn!("X/Twitter rate limit exceeded: {}", limit_exceeded);
            return Err(anyhow::anyhow!("Rate limit exceeded: {}", limit_exceeded));
        }

        // Create action for validation
        let action = ActionType::SocialPost {
            platform: "x_twitter".to_string(),
            content: content.to_string(),
        };

        // Log action request
        self.audit_logger
            .log_action_request("x_twitter", &action, "Posting tweet to X/Twitter")
            .await?;

        // Validate the action
        self.validator
            .validate_action(
                action,
                "X/Twitter post".to_string(),
                vec![
                    "User requested social media post".to_string(),
                    format!("Content length: {} characters", content.len()),
                ],
            )
            .await?;

        // If validation passes, post the tweet
        info!("Posting validated tweet: {}", content.chars().take(50).collect::<String>());

        match self.inner.post_tweet(content).await {
            Ok(tweet_id) => {
                // Log successful action
                self.audit_logger
                    .log_action_request(
                        "x_twitter",
                        &ActionType::SocialPost {
                            platform: "x_twitter".to_string(),
                            content: content.to_string(),
                        },
                        &format!("Successfully posted tweet: {}", tweet_id),
                    )
                    .await?;

                Ok(tweet_id)
            }
            Err(e) => {
                error!("Failed to post tweet: {}", e);
                Err(e)
            }
        }
    }

    /// Safely reply to a tweet with validation
    pub async fn safe_reply_to_tweet(&self, tweet_id: &str, content: &str) -> Result<String> {
        // Check rate limits first
        if let Err(limit_exceeded) = self.resource_monitor.check_api_limit("x_twitter").await {
            warn!("X/Twitter rate limit exceeded: {}", limit_exceeded);
            return Err(anyhow::anyhow!("Rate limit exceeded: {}", limit_exceeded));
        }

        // Create action for validation
        let action = ActionType::SocialReply {
            platform: "x_twitter".to_string(),
            to: tweet_id.to_string(),
            content: content.to_string(),
        };

        // Log action request
        self.audit_logger
            .log_action_request("x_twitter", &action, "Replying to tweet on X/Twitter")
            .await?;

        // Validate the action
        self.validator
            .validate_action(
                action,
                format!("X/Twitter reply to {}", tweet_id),
                vec![
                    "Replying to user mention or interaction".to_string(),
                    format!("Reply content length: {} characters", content.len()),
                ],
            )
            .await?;

        // If validation passes, post the reply
        info!(
            "Posting validated reply to {}: {}",
            tweet_id,
            content.chars().take(50).collect::<String>()
        );

        match self.inner.reply_to_tweet(tweet_id, content).await {
            Ok(reply_id) => {
                // Log successful action
                self.audit_logger
                    .log_action_request(
                        "x_twitter",
                        &ActionType::SocialReply {
                            platform: "x_twitter".to_string(),
                            to: tweet_id.to_string(),
                            content: content.to_string(),
                        },
                        &format!("Successfully posted reply: {}", reply_id),
                    )
                    .await?;

                Ok(reply_id)
            }
            Err(e) => {
                error!("Failed to post reply: {}", e);
                Err(e)
            }
        }
    }

    /// Safely get mentions (read-only, always allowed)
    pub async fn safe_get_mentions(&self) -> Result<Vec<Mention>> {
        // Log the request
        self.audit_logger
            .log_action_request(
                "x_twitter",
                &ActionType::ApiCall {
                    provider: "x_twitter".to_string(),
                    endpoint: "mentions".to_string(),
                },
                "Fetching mentions from X/Twitter",
            )
            .await?;

        self.inner.get_mentions(None).await
    }

    /// Get the underlying X client (for read-only operations)
    pub fn inner(&self) -> &Arc<XClient> {
        &self.inner
    }
}

/// Safety-aware wrapper for X/Twitter consciousness
pub struct SafeXConsciousness {
    /// The underlying X consciousness
    inner: Arc<XConsciousness>,

    /// Action validator
    validator: Arc<ActionValidator>,

    /// Audit logger
    audit_logger: Arc<AuditLogger>,

    /// Resource monitor
    resource_monitor: Arc<ResourceMonitor>,
}

impl SafeXConsciousness {
    /// Create a new safety-aware X consciousness
    pub fn new(
        x_consciousness: Arc<XConsciousness>,
        validator: Arc<ActionValidator>,
        audit_logger: Arc<AuditLogger>,
        resource_monitor: Arc<ResourceMonitor>,
    ) -> Self {
        Self { inner: x_consciousness, validator, audit_logger, resource_monitor }
    }

    /// Start monitoring with safety checks
    pub async fn safe_start(self: Arc<Self>) -> Result<()> {
        // Log system startup
        self.audit_logger
            .log_action_request(
                "x_consciousness",
                &ActionType::SocialPost {
                    platform: "x_twitter".to_string(),
                    content: "Starting X/Twitter consciousness monitoring".to_string(),
                },
                "Initiating social media monitoring",
            )
            .await?;

        // Validate the monitoring startup
        self.validator
            .validate_action(
                ActionType::SocialPost {
                    platform: "x_twitter".to_string(),
                    content: "system_startup".to_string(),
                },
                "X/Twitter consciousness startup".to_string(),
                vec!["Starting autonomous social media monitoring".to_string()],
            )
            .await?;

        // If validation passes, start monitoring
        info!("Starting safety-aware X/Twitter consciousness monitoring");
        self.inner.clone().start().await
    }

    /// Safely shutdown X consciousness
    pub async fn safe_shutdown(&self) -> Result<()> {
        // Log shutdown
        self.audit_logger
            .log_action_request(
                "x_consciousness",
                &ActionType::SocialPost {
                    platform: "x_twitter".to_string(),
                    content: "Shutting down X/Twitter consciousness monitoring".to_string(),
                },
                "Stopping social media monitoring",
            )
            .await?;

        info!("Shutting down safety-aware X/Twitter consciousness");
        self.inner.shutdown().await
    }

    /// Get the underlying X consciousness (for read-only operations)
    pub fn inner(&self) -> &Arc<XConsciousness> {
        &self.inner
    }
}

/// Helper function to create a complete safety-aware X/Twitter system
pub async fn create_safe_x_system(
    x_client: Arc<XClient>,
    cognitive_system: Arc<crate::cognitive::CognitiveSystem>,
    consciousness: Arc<crate::cognitive::consciousness_stream::ThermodynamicConsciousnessStream>,
    validator: Arc<ActionValidator>,
    audit_logger: Arc<AuditLogger>,
    resource_monitor: Arc<ResourceMonitor>,
    config: XConsciousnessConfig,
) -> Result<(Arc<SafeXClient>, Arc<SafeXConsciousness>)> {
    info!("Creating safety-aware X/Twitter system");

    // Create safe X client
    let safe_x_client = Arc::new(SafeXClient::new(
        x_client.clone(),
        validator.clone(),
        audit_logger.clone(),
        resource_monitor.clone(),
    ));

    // Create X consciousness with the original client
    let x_consciousness =
        Arc::new(XConsciousness::new(x_client, cognitive_system, consciousness, config).await?);

    // Wrap X consciousness with safety
    let safe_x_consciousness = Arc::new(SafeXConsciousness::new(
        x_consciousness,
        validator,
        audit_logger,
        resource_monitor,
    ));

    Ok((safe_x_client, safe_x_consciousness))
}

/// Safety configuration for X/Twitter operations
#[derive(Debug, Clone)]
pub struct XSafetyConfig {
    /// Maximum posts per hour
    pub max_posts_per_hour: usize,

    /// Maximum replies per hour
    pub max_replies_per_hour: usize,

    /// Require approval for all posts
    pub require_approval_for_posts: bool,

    /// Require approval for replies
    pub require_approval_for_replies: bool,

    /// Content safety checks enabled
    pub content_safety_enabled: bool,

    /// Maximum content length
    pub max_content_length: usize,
}

impl Default for XSafetyConfig {
    fn default() -> Self {
        Self {
            max_posts_per_hour: 10,
            max_replies_per_hour: 20,
            require_approval_for_posts: true,
            require_approval_for_replies: false, // Auto-reply to mentions with high confidence
            content_safety_enabled: true,
            max_content_length: 280, // X/Twitter character limit
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_safe_x_operations() {
        // This would test the safety wrapper
        // For now, just check that the types compile
        let config = XSafetyConfig::default();
        assert!(config.content_safety_enabled);
        assert!(config.require_approval_for_posts);
    }
}
