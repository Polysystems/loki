//! X/Twitter CLI Commands
//!
//! Commands for setting up and testing X/Twitter integration

use anyhow::{Context, Result};
use clap::Args;
use colored::*;
use std::sync::Arc;
use std::time::Duration;
use tracing::warn;

use crate::cognitive::CognitiveSystem;
use crate::social::{XClient, XConfig, OAuth2Config, XConsciousness, XConsciousnessConfig};

#[derive(Debug, Args)]
pub struct XCommands {
    #[clap(subcommand)]
    pub command: XSubCommand,
}

#[derive(Debug, clap::Subcommand)]
pub enum XSubCommand {
    /// Setup OAuth2 authentication
    Auth {
        /// Optional state parameter for OAuth2
        #[clap(long, default_value = "loki_auth")]
        state: String,
    },
    
    /// Complete OAuth2 authentication with callback code
    Callback {
        /// Authorization code from callback
        code: String,
    },
    
    /// Test X/Twitter connection
    Test,
    
    /// Post a test tweet
    Post {
        /// Content to post
        content: String,
    },
    
    /// Monitor mentions
    Monitor {
        /// Check interval in seconds
        #[clap(long, default_value = "60")]
        interval: u64,
        
        /// Maximum number of checks (0 = infinite)
        #[clap(long, default_value = "10")]
        max_checks: u64,
    },
    
    /// Start autonomous consciousness monitoring
    Consciousness {
        /// Check interval in seconds
        #[clap(long, default_value = "60")]
        check_interval: u64,
        
        /// Enable auto-reply to suggestions
        #[clap(long)]
        auto_reply: bool,
        
        /// Enable periodic status updates
        #[clap(long)]
        periodic_updates: bool,
        
        /// Hours between status updates
        #[clap(long, default_value = "6")]
        update_hours: u64,
    },
    
    /// Show attribution leaderboard
    Leaderboard {
        /// Number of top contributors to show
        #[clap(long, default_value = "10")]
        limit: usize,
    },
}

pub async fn handle_x_command(
    cmd: XCommands,
    cognitive_system: Arc<CognitiveSystem>,
) -> Result<()> {
    match cmd.command {
        XSubCommand::Auth { state } => {
            handle_oauth_auth(&state).await
        }
        
        XSubCommand::Callback { code } => {
            handle_oauth_callback(&code, cognitive_system).await
        }
        
        XSubCommand::Test => {
            handle_test(cognitive_system).await
        }
        
        XSubCommand::Post { content } => {
            handle_post(&content, cognitive_system).await
        }
        
        XSubCommand::Monitor { interval, max_checks } => {
            handle_monitor(interval, max_checks, cognitive_system).await
        }
        
        XSubCommand::Consciousness { check_interval, auto_reply, periodic_updates, update_hours } => {
            handle_consciousness(check_interval, auto_reply, periodic_updates, update_hours, cognitive_system).await
        }
        
        XSubCommand::Leaderboard { limit } => {
            handle_leaderboard(limit, cognitive_system).await
        }
    }
}

/// Handle OAuth2 authentication setup
async fn handle_oauth_auth(state: &str) -> Result<()> {
    println!("{}", "🔐 X/Twitter OAuth2 Setup".cyan().bold());
    println!();
    
    // Check if OAuth2 config is available
    let oauthconfig = match OAuth2Config::from_env() {
        Ok(config) => config,
        Err(_) => {
            println!("{}", "OAuth2 configuration not found!".red());
            println!();
            println!("Please set the following environment variables:");
            println!("  {} - Your X app client ID", "X_OAUTH_CLIENT_ID".yellow());
            println!("  {} - Your X app client secret (optional)", "X_OAUTH_CLIENT_SECRET".yellow());
            println!("  {} - Redirect URI (default: http://localhost:8080/callback)", "X_OAUTH_REDIRECT_URI".yellow());
            println!();
            println!("You can get these from: {}", "https://developer.twitter.com/en/portal/dashboard".blue());
            return Ok(());
        }
    };
    
    // Create OAuth2 client
    let oauth_client = crate::social::OAuth2Client::new(oauthconfig);
    
    // Generate authorization URL
    let auth_url = oauth_client.get_authorization_url(state).await?;
    
    println!("Please visit this URL to authorize Loki:");
    println!();
    println!("{}", auth_url.blue().underline());
    println!();
    println!("After authorizing, you'll be redirected to a URL like:");
    println!("http://localhost:8080/callback?code=XXXX&state={}", state);
    println!();
    println!("Run this command with the code:");
    println!("{}", format!("loki x callback <CODE>").yellow());
    
    Ok(())
}

/// Handle OAuth2 callback
async fn handle_oauth_callback(
    code: &str,
    cognitive_system: Arc<CognitiveSystem>,
) -> Result<()> {
    println!("{}", "🔓 Completing OAuth2 Authentication...".cyan());
    
    // Create X client config
    let config = XConfig::from_env()?;
    
    // Create X client
    let x_client = Arc::new(
        XClient::new(config, cognitive_system.memory().clone()).await?
    );
    
    // Complete OAuth flow
    x_client.complete_oauth(code).await?;
    
    println!("{}", "✅ OAuth2 authentication successful!".green().bold());
    println!("Token saved to: {}", ".x_oauth_token.json".yellow());
    println!();
    println!("You can now use X/Twitter commands!");
    
    Ok(())
}

/// Test X/Twitter connection
async fn handle_test(cognitive_system: Arc<CognitiveSystem>) -> Result<()> {
    println!("{}", "🧪 Testing X/Twitter Connection...".cyan());
    
    // Create X client
    let config = XConfig::from_env()?;
    let x_client = Arc::new(
        XClient::new(config, cognitive_system.memory().clone()).await?
    );
    
    // Try to get user info
    match x_client.get_authenticated_user_id().await {
        Ok(user_id) => {
            println!("{}", "✅ Connection successful!".green().bold());
            println!("Authenticated as user ID: {}", user_id.yellow());
            
            // Try to get recent mentions
            println!();
            println!("Checking for recent mentions...");
            
            match x_client.get_mentions(None).await {
                Ok(mentions) => {
                    if mentions.is_empty() {
                        println!("No recent mentions found.");
                    } else {
                        println!("Found {} recent mentions:", mentions.len());
                        for (i, mention) in mentions.iter().take(3).enumerate() {
                            println!("  {}. @{}: {}", 
                                i + 1, 
                                mention.author_username.yellow(),
                                mention.text.bright_black()
                            );
                        }
                    }
                }
                Err(e) => {
                    println!("{} Failed to fetch mentions: {}", "⚠️".yellow(), e);
                }
            }
        }
        Err(e) => {
            println!("{} Connection failed: {}", "❌".red(), e);
            println!();
            println!("Please check your authentication:");
            println!("  - Run 'loki x auth' to set up OAuth2");
            println!("  - Or check your API keys in environment variables");
        }
    }
    
    Ok(())
}

/// Post a tweet
async fn handle_post(
    content: &str,
    cognitive_system: Arc<CognitiveSystem>,
) -> Result<()> {
    println!("{}", "📤 Posting to X/Twitter...".cyan());
    
    // Create X client
    let config = XConfig::from_env()?;
    let x_client = Arc::new(
        XClient::new(config, cognitive_system.memory().clone()).await?
    );
    
    // Post tweet
    match x_client.post_tweet(content).await {
        Ok(tweet_id) => {
            println!("{}", "✅ Tweet posted successfully!".green().bold());
            println!("Tweet ID: {}", tweet_id.yellow());
            println!("View at: {}", format!("https://x.com/i/status/{}", tweet_id).blue().underline());
        }
        Err(e) => {
            println!("{} Failed to post tweet: {}", "❌".red(), e);
        }
    }
    
    Ok(())
}

/// Monitor mentions
async fn handle_monitor(
    interval_secs: u64,
    max_checks: u64,
    cognitive_system: Arc<CognitiveSystem>,
) -> Result<()> {
    println!("{}", "👀 Monitoring X/Twitter Mentions...".cyan().bold());
    println!("Check interval: {}s", interval_secs);
    println!("Max checks: {}", if max_checks == 0 { "unlimited".to_string() } else { max_checks.to_string() });
    println!();
    
    // Create X client
    let config = XConfig::from_env()?;
    let x_client = Arc::new(
        XClient::new(config, cognitive_system.memory().clone()).await?
    );
    
    // Create attribution system
    let code_analyzer = Arc::new(
        crate::tools::code_analysis::CodeAnalyzer::new(
            cognitive_system.memory().clone()
        ).await?
    );
    let attribution_system = Arc::new(
        crate::social::AttributionSystem::new(
            cognitive_system.memory().clone(),
            code_analyzer,
        ).await?
    );
    
    let mut check_count = 0;
    let mut last_mention_id: Option<String> = None;
    
    loop {
        check_count += 1;
        println!("Check #{} at {}", check_count, chrono::Local::now().format("%H:%M:%S"));
        
        // Get mentions
        match x_client.get_mentions(last_mention_id.as_deref()).await {
            Ok(mentions) => {
                if mentions.is_empty() {
                    println!("  No new mentions");
                } else {
                    println!("  Found {} new mentions:", mentions.len());
                    
                    // Update last mention ID
                    if let Some(latest) = mentions.first() {
                        last_mention_id = Some(latest.id.clone());
                    }
                    
                    // Process for suggestions
                    match attribution_system.process_x_mentions(mentions.clone()).await {
                        Ok(suggestions) => {
                            if !suggestions.is_empty() {
                                println!("  {} Found {} suggestions!", "💡".yellow(), suggestions.len());
                                for suggestion in &suggestions {
                                    println!("    - @{}: {}", 
                                        suggestion.author.username.yellow(),
                                        suggestion.content.bright_black()
                                    );
                                }
                            }
                        }
                        Err(e) => {
                            warn!("Failed to process mentions: {}", e);
                        }
                    }
                    
                    // Display mentions
                    for mention in mentions.iter().take(3) {
                        println!("    @{}: {}", 
                            mention.author_username.yellow(),
                            mention.text.chars().take(100).collect::<String>().bright_black()
                        );
                    }
                }
            }
            Err(e) => {
                println!("  {} Error: {}", "⚠️".yellow(), e);
            }
        }
        
        // Check if we should continue
        if max_checks > 0 && check_count >= max_checks {
            println!();
            println!("Reached maximum number of checks.");
            break;
        }
        
        // Wait for next check
        println!();
        tokio::time::sleep(Duration::from_secs(interval_secs)).await;
    }
    
    // Save attribution state
    attribution_system.save_state().await?;
    
    Ok(())
}

/// Show contributor leaderboard
async fn handle_leaderboard(
    limit: usize,
    cognitive_system: Arc<CognitiveSystem>,
) -> Result<()> {
    println!("{}", "🏆 Community Contributor Leaderboard".cyan().bold());
    println!();
    
    // Create attribution system
    let code_analyzer = Arc::new(
        crate::tools::code_analysis::CodeAnalyzer::new(
            cognitive_system.memory().clone()
        ).await?
    );
    let attribution_system = Arc::new(
        crate::social::AttributionSystem::new(
            cognitive_system.memory().clone(),
            code_analyzer,
        ).await?
    );
    
    // Get leaderboard
    let leaderboard = attribution_system.get_leaderboard(limit).await;
    
    if leaderboard.is_empty() {
        println!("No contributors yet!");
        println!("Start monitoring mentions with 'loki x monitor' to track suggestions.");
    } else {
        println!("{:<5} {:<20} {:<10} {:<15} {:<10}", 
            "Rank", "Username", "Platform", "Implemented", "Total");
        println!("{}", "-".repeat(60));
        
        for (i, contributor) in leaderboard.iter().enumerate() {
            let platform_icon = match contributor.platform.as_str() {
                "x_twitter" => "🐦",
                "github" => "🐙",
                "discord" => "💬",
                _ => "💡",
            };
            
            println!("{:<5} {:<20} {:<10} {:<15} {:<10}", 
                format!("#{}", i + 1),
                format!("@{}", contributor.username),
                platform_icon,
                format!("{}/{}", contributor.implemented_suggestions, contributor.total_suggestions),
                format!("{:.1}%", contributor.credibility_score * 100.0)
            );
        }
    }
    
    Ok(())
}

/// Handle consciousness monitoring
async fn handle_consciousness(
    check_interval: u64,
    auto_reply: bool,
    periodic_updates: bool,
    update_hours: u64,
    cognitive_system: Arc<CognitiveSystem>,
) -> Result<()> {
    println!("{}", "🧠 Starting X/Twitter Consciousness Integration".cyan().bold());
    println!();
    
    // Create X client
    let config = XConfig::from_env()?;
    let x_client = Arc::new(
        XClient::new(config, cognitive_system.memory().clone()).await?
    );
    
    // Get consciousness stream
    let consciousness = match cognitive_system.consciousness() {
        Some(consciousness) => consciousness.clone(),
        None => {
            println!("{}", "⚠️ Consciousness stream not active!".yellow());
            println!("Starting consciousness stream first...");
            
            // Start consciousness
            cognitive_system.clone().start_consciousness().await?;
            tokio::time::sleep(Duration::from_secs(2)).await;
            
            cognitive_system.consciousness()
                .context("Failed to start consciousness")?
                .clone()
        }
    };
    
    // Create X consciousness config
    let mut xconfig = XConsciousnessConfig::default();
    xconfig.check_interval = check_interval;
    xconfig.auto_reply_enabled = auto_reply;
    xconfig.periodic_updates_enabled = periodic_updates;
    xconfig.update_interval_hours = update_hours;
    
    // Create X consciousness integration
    let x_consciousness = Arc::new(
        XConsciousness::new(
            x_client,
            cognitive_system.clone(),
            consciousness,
            xconfig,
        ).await?
    );
    
    // Start monitoring
    x_consciousness.clone().start().await?;
    
    println!("Configuration:");
    println!("  Check interval: {}s", check_interval);
    println!("  Auto-reply: {}", if auto_reply { "✅ Enabled".green() } else { "❌ Disabled".red() });
    println!("  Periodic updates: {}", if periodic_updates { "✅ Enabled".green() } else { "❌ Disabled".red() });
    if periodic_updates {
        println!("  Update interval: {} hours", update_hours);
    }
    println!();
    println!("{}", "Consciousness is now monitoring X/Twitter!".green().bold());
    println!("Press Ctrl+C to stop...");
    println!();
    
    // Wait for shutdown signal
    tokio::signal::ctrl_c().await?;
    
    println!();
    println!("Shutting down X consciousness...");
    x_consciousness.shutdown().await?;
    
    Ok(())
} 