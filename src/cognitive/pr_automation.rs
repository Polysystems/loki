use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use tokio::sync::{RwLock, mpsc};
use tokio::time::interval;
use tracing::{debug, error, info, warn};

use crate::cognitive::{
    Attribution,
    AttributionBridge,
    ChangeType,
    CodeChange,
    RiskLevel,
    SelfModificationPipeline,
    TestGenerator,
    TestGeneratorConfig,
};
use crate::memory::{CognitiveMemory, MemoryMetadata};
use crate::social::{AttributionSystem, Suggestion, SuggestionStatus};
use crate::tools::github::{GitHubClient, GitHubConfig};

/// Configuration for PR automation
#[derive(Debug, Clone)]
pub struct PrAutomationConfig {
    /// How often to check for new suggestions
    pub check_interval: Duration,

    /// Maximum suggestions to process per cycle
    pub max_suggestions_per_cycle: usize,

    /// Minimum confidence score to create PR
    pub min_confidence_threshold: f32,

    /// Enable automatic PR creation
    pub auto_create_prs: bool,

    /// Enable test generation for PRs
    pub generate_tests: bool,

    /// Repository path
    pub repo_path: PathBuf,

    /// Branch prefix for automated PRs
    pub branch_prefix: String,
}

impl Default for PrAutomationConfig {
    fn default() -> Self {
        Self {
            check_interval: Duration::from_secs(300), // 5 minutes
            max_suggestions_per_cycle: 5,
            min_confidence_threshold: 0.7,
            auto_create_prs: true,
            generate_tests: true,
            repo_path: PathBuf::from("."),
            branch_prefix: "auto/x-suggestion".to_string(),
        }
    }
}

/// PR automation statistics
#[derive(Debug, Clone, Default)]
pub struct PrAutomationStats {
    pub suggestions_processed: u64,
    pub prs_created: u64,
    pub prs_failed: u64,
    pub tests_generated: u64,
    pub last_run: Option<Instant>,
}

/// Automated PR creation from X suggestions
pub struct PrAutomationSystem {
    /// Configuration
    config: PrAutomationConfig,

    /// Attribution system for suggestions
    attribution_system: Arc<AttributionSystem>,

    /// Attribution bridge for processing
    attribution_bridge: Arc<AttributionBridge>,

    /// Self-modification pipeline
    self_modify: Arc<SelfModificationPipeline>,

    /// Test generator
    test_generator: Arc<TestGenerator>,

    /// GitHub client
    github_client: Option<Arc<GitHubClient>>,

    /// Memory system
    memory: Arc<CognitiveMemory>,

    /// Statistics
    stats: Arc<RwLock<PrAutomationStats>>,

    /// Active PRs being tracked
    active_prs: Arc<RwLock<HashMap<String, PrTracker>>>,

    /// Shutdown channel
    shutdown_rx: mpsc::Receiver<()>,
    shutdown_tx: mpsc::Sender<()>,
}

/// Tracks active PRs
#[derive(Debug, Clone)]
struct PrTracker {
    pub pr_number: u32,
    pub branch_name: String,
    pub suggestion_id: String,
    pub created_at: Instant,
    pub status: PrStatus,
}

#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
enum PrStatus {
    Creating,
    Open,
    Merged,
    Closed,
    Failed,
}

impl PrAutomationSystem {
    pub async fn new(
        config: PrAutomationConfig,
        attribution_system: Arc<AttributionSystem>,
        memory: Arc<CognitiveMemory>,
    ) -> Result<Self> {
        info!("Initializing PR automation system");

        // Create self-modification pipeline first
        let self_modify = Arc::new(
            SelfModificationPipeline::new(config.repo_path.clone(), memory.clone()).await?,
        );

        // Create components
        let attribution_bridge = Arc::new(
            AttributionBridge::new(attribution_system.clone(), self_modify.clone(), memory.clone())
                .await?,
        );

        let test_generator =
            Arc::new(TestGenerator::new(TestGeneratorConfig::default(), memory.clone()).await?);

        // Try to create GitHub client
        let github_client = match GitHubConfig::from_env() {
            Ok(ghconfig) => match GitHubClient::new(ghconfig, memory.clone()).await {
                Ok(client) => {
                    info!("GitHub client initialized for PR automation");
                    Some(Arc::new(client))
                }
                Err(e) => {
                    warn!("Failed to initialize GitHub client: {}", e);
                    None
                }
            },
            Err(e) => {
                warn!("GitHub config not available: {}", e);
                None
            }
        };

        let (shutdown_tx, shutdown_rx) = mpsc::channel(1);

        Ok(Self {
            config,
            attribution_system,
            attribution_bridge,
            self_modify,
            test_generator,
            github_client,
            memory,
            stats: Arc::new(RwLock::new(PrAutomationStats::default())),
            active_prs: Arc::new(RwLock::new(HashMap::new())),
            shutdown_rx,
            shutdown_tx,
        })
    }

    /// Start the automation system
    pub async fn start(mut self) -> Result<()> {
        info!("Starting PR automation system");

        // Store startup in memory
        self.memory
            .store(
                "PR automation system started - monitoring X suggestions for code improvements"
                    .to_string(),
                vec![],
                MemoryMetadata {
                    source: "pr_automation".to_string(),
                    tags: vec!["automation".to_string(), "startup".to_string()],
                    importance: 0.8,
                    associations: vec![],
                    context: Some("PR automation startup".to_string()),
                    created_at: chrono::Utc::now(),
                    accessed_count: 0,
                    last_accessed: None,
                    version: 1,
                    category: "cognitive".to_string(),
                    timestamp: chrono::Utc::now(),
                    expiration: None,
                },
            )
            .await?;

        let mut check_interval = interval(self.config.check_interval);

        loop {
            tokio::select! {
                _ = check_interval.tick() => {
                    if let Err(e) = self.process_suggestions().await {
                        error!("Error processing suggestions: {}", e);
                    }
                }
                _ = self.shutdown_rx.recv() => {
                    info!("PR automation system shutting down");
                    break;
                }
            }
        }

        Ok(())
    }

    /// Process pending suggestions
    async fn process_suggestions(&mut self) -> Result<()> {
        debug!("Checking for new suggestions to process");

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.last_run = Some(Instant::now());
        }

        // Get pending suggestions
        let suggestions =
            self.attribution_system.get_suggestions_by_status(SuggestionStatus::New).await?;

        if suggestions.is_empty() {
            debug!("No new suggestions to process");
            return Ok(());
        }

        info!("Found {} new suggestions to process", suggestions.len());

        // Process up to max_suggestions_per_cycle
        let to_process =
            suggestions.into_iter().take(self.config.max_suggestions_per_cycle).collect::<Vec<_>>();

        for suggestion in to_process {
            if let Err(e) = self.process_single_suggestion(suggestion).await {
                error!("Failed to process suggestion: {}", e);

                // Update stats
                let mut stats = self.stats.write().await;
                stats.prs_failed += 1;
            }
        }

        Ok(())
    }

    /// Process a single suggestion
    async fn process_single_suggestion(&self, suggestion: Suggestion) -> Result<()> {
        info!("Processing suggestion: {} from @{}", suggestion.id, suggestion.author.username);

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.suggestions_processed += 1;
        }

        // Convert to code change
        let code_change =
            match self.attribution_bridge.suggestion_to_code_change(&suggestion).await? {
                Some(change) => change,
                None => {
                    info!("Suggestion {} is not actionable or already implemented", suggestion.id);

                    // Update suggestion status if not already implemented
                    if !matches!(suggestion.status, SuggestionStatus::Implemented) {
                        self.attribution_system
                            .update_suggestion_status(&suggestion.id, SuggestionStatus::Analyzed)
                            .await?;
                    }

                    return Ok(());
                }
            };

        // Check confidence threshold
        if suggestion.confidence_score < self.config.min_confidence_threshold {
            info!(
                "Suggestion {} confidence {} below threshold {}",
                suggestion.id, suggestion.confidence_score, self.config.min_confidence_threshold
            );
            return Ok(());
        }

        // Generate tests if enabled
        let test_code = if self.config.generate_tests {
            match self.generate_tests_for_change(&code_change).await {
                Ok(tests) => {
                    info!("Generated tests for suggestion {}", suggestion.id);

                    // Update stats
                    let mut stats = self.stats.write().await;
                    stats.tests_generated += 1;

                    Some(tests)
                }
                Err(e) => {
                    warn!("Failed to generate tests: {}", e);
                    None
                }
            }
        } else {
            None
        };

        // Create PR if auto-create is enabled
        if self.config.auto_create_prs {
            match self.create_pr_for_change(code_change, test_code).await {
                Ok(pr_number) => {
                    info!("Created PR #{} for suggestion {}", pr_number, suggestion.id);

                    // Update stats
                    let mut stats = self.stats.write().await;
                    stats.prs_created += 1;

                    // Track the PR
                    let tracker = PrTracker {
                        pr_number,
                        branch_name: format!("{}-{}", self.config.branch_prefix, suggestion.id),
                        suggestion_id: suggestion.id.clone(),
                        created_at: Instant::now(),
                        status: PrStatus::Open,
                    };

                    self.active_prs.write().await.insert(suggestion.id.clone(), tracker);

                    // Update suggestion status
                    self.attribution_system
                        .update_suggestion_status(&suggestion.id, SuggestionStatus::Implemented)
                        .await?;

                    // Store in memory
                    self.memory
                        .store(
                            format!(
                                "Created PR #{} from X suggestion by @{}: {}",
                                pr_number, suggestion.author.username, suggestion.content
                            ),
                            vec![format!("Confidence: {}", suggestion.confidence_score)],
                            MemoryMetadata {
                                source: "pr_automation".to_string(),
                                tags: vec!["pr".to_string(), "automated".to_string()],
                                importance: 0.9,
                                associations: vec![],
                                context: Some("automated PR creation".to_string()),
                                created_at: chrono::Utc::now(),
                                accessed_count: 0,
                                last_accessed: None,
                                version: 1,
                    category: "cognitive".to_string(),
                                timestamp: chrono::Utc::now(),
                                expiration: None,
                            },
                        )
                        .await?;
                }
                Err(e) => {
                    error!("Failed to create PR for suggestion {}: {}", suggestion.id, e);

                    // Update stats
                    let mut stats = self.stats.write().await;
                    stats.prs_failed += 1;
                }
            }
        }

        Ok(())
    }

    /// Generate tests for a code change
    async fn generate_tests_for_change(&self, change: &CodeChange) -> Result<String> {
        info!("Generating tests for code change: {}", change.description);

        // Use the test generator
        let tests = self.self_modify.generate_tests(change).await?;

        Ok(tests)
    }

    /// Create a PR for a code change
    async fn create_pr_for_change(
        &self,
        change: CodeChange,
        test_code: Option<String>,
    ) -> Result<u32> {
        // If we have test code, create a modified change that includes tests
        if let Some(tests) = test_code {
            // Create test file path
            let test_file_path = if change.file_path.starts_with("src/") {
                // Convert src/module/file.rs to src/module/file_test.rs
                let mut test_path = change.file_path.clone();
                let file_stem = test_path
                    .file_stem()
                    .ok_or_else(|| anyhow::anyhow!("Invalid file path"))?
                    .to_string_lossy();
                test_path.set_file_name(format!("{}_test.rs", file_stem));
                test_path
            } else {
                // Default to tests/ directory
                PathBuf::from("tests").join(
                    change
                        .file_path
                        .file_name()
                        .ok_or_else(|| anyhow::anyhow!("Invalid file path"))?,
                )
            };

            // First apply the main change
            let pr = self.self_modify.propose_change(change).await?;

            // Then add tests as a separate change
            let test_change = CodeChange {
                file_path: test_file_path,
                change_type: ChangeType::Test,
                description: format!("Add tests for: {}", pr.title),
                reasoning: "Automated test generation to ensure code quality".to_string(),
                old_content: None,
                new_content: tests,
                line_range: None,
                risk_level: RiskLevel::Low,
                attribution: pr.description.contains("Attribution").then(|| {
                    // Extract attribution from PR description
                    Attribution {
                        contributor: "AI Test Generator".to_string(),
                        platform: "loki".to_string(),
                        suggestion_id: "generated".to_string(),
                        suggestion_text: "Automated test generation".to_string(),
                        timestamp: chrono::Utc::now(),
                    }
                }),
            };

            // Apply test change to the same branch
            // Note: This is simplified - in reality we'd need to handle branch management
            let _ = self.self_modify.propose_change(test_change).await?;

            Ok(pr.number)
        } else {
            // Just create PR for the main change
            let pr = self.self_modify.propose_change(change).await?;
            Ok(pr.number)
        }
    }

    /// Check status of active PRs using GitHub API
    pub async fn check_pr_status(&self) -> Result<()> {
        let github_client = match &self.github_client {
            Some(client) => client.clone(),
            None => {
                debug!("No GitHub client available for PR status checking");
                return Ok(());
            }
        };

        let mut active_prs = self.active_prs.write().await;
        let mut updated_prs = Vec::new();

        for (suggestion_id, tracker) in active_prs.iter() {
            debug!("Checking status of PR #{} for suggestion {}", tracker.pr_number, suggestion_id);

            // Use GitHub API to get actual PR status
            match self.fetch_pr_status(&github_client, tracker.pr_number).await {
                Ok(new_status) => {
                    if new_status != tracker.status {
                        info!(
                            "PR #{} status changed from {:?} to {:?}",
                            tracker.pr_number, tracker.status, new_status
                        );

                        // Store status change in memory
                        let _ = self
                            .memory
                            .store(
                                format!(
                                    "PR #{} status update: {:?} -> {:?}",
                                    tracker.pr_number, tracker.status, new_status
                                ),
                                vec![format!("Suggestion ID: {}", suggestion_id)],
                                MemoryMetadata {
                                    source: "pr_automation".to_string(),
                                    tags: vec!["pr".to_string(), "status".to_string()],
                                    importance: match new_status {
                                        PrStatus::Merged => 1.0,
                                        PrStatus::Closed => 0.8,
                                        _ => 0.5,
                                    },
                                    associations: vec![],
                                    context: Some("PR status tracking".to_string()),
                                    created_at: chrono::Utc::now(),
                                    accessed_count: 0,
                                    last_accessed: None,
                                    version: 1,
                                    category: "automation".to_string(),
                                    timestamp: chrono::Utc::now(),
                                    expiration: None,
                                },
                            )
                            .await;

                        updated_prs.push((suggestion_id.clone(), new_status.clone()));

                        // Handle merged PRs - update attribution
                        if new_status == PrStatus::Merged {
                            if let Err(e) = self
                                .attribution_system
                                .update_suggestion_status(
                                    suggestion_id,
                                    SuggestionStatus::Implemented,
                                )
                                .await
                            {
                                warn!("Failed to update suggestion status to merged: {}", e);
                            }
                        }
                    }
                }
                Err(e) => {
                    warn!("Failed to check PR #{} status: {}", tracker.pr_number, e);
                }
            }
        }

        // Apply status updates
        for (suggestion_id, new_status) in updated_prs {
            if let Some(tracker) = active_prs.get_mut(&suggestion_id) {
                tracker.status = new_status;
            }
        }

        // Clean up closed/merged PRs that are older than 24 hours
        let cutoff_time = Instant::now() - Duration::from_secs(24 * 60 * 60);
        active_prs.retain(|_, tracker| {
            !(matches!(tracker.status, PrStatus::Merged | PrStatus::Closed)
                && tracker.created_at < cutoff_time)
        });

        Ok(())
    }

    /// Fetch PR status from GitHub API
    async fn fetch_pr_status(
        &self,
        github_client: &GitHubClient,
        pr_number: u32,
    ) -> Result<PrStatus> {
        match github_client.get_pull_request(pr_number).await {
            Ok(pr_details) => {
                let status = match pr_details.state {
                    crate::tools::github::PRState::Open => PrStatus::Open,
                    crate::tools::github::PRState::Merged => PrStatus::Merged,
                    crate::tools::github::PRState::Closed => PrStatus::Closed,
                };

                debug!("Fetched PR #{} status: {:?}", pr_number, status);
                Ok(status)
            }
            Err(e) => {
                warn!("Failed to fetch PR #{} status: {}", pr_number, e);
                // Return failed status if we can't fetch it
                Ok(PrStatus::Failed)
            }
        }
    }

    /// Get automation statistics
    pub async fn get_stats(&self) -> PrAutomationStats {
        self.stats.read().await.clone()
    }

    /// Shutdown the automation system
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down PR automation system");
        let _ = self.shutdown_tx.send(()).await;
        Ok(())
    }
}

/// Monitor for PR automation health
pub struct PrAutomationMonitor {
    automation: Arc<PrAutomationSystem>,
}

impl PrAutomationMonitor {
    pub fn new(automation: Arc<PrAutomationSystem>) -> Self {
        Self { automation }
    }

    /// Run health checks
    pub async fn health_check(&self) -> Result<HealthStatus> {
        let stats = self.automation.get_stats().await;

        let health = if let Some(last_run) = stats.last_run {
            let time_since_last = Instant::now().duration_since(last_run);

            if time_since_last > Duration::from_secs(600) {
                HealthStatus::Warning("No runs in last 10 minutes".to_string())
            } else if stats.prs_failed > stats.prs_created {
                HealthStatus::Degraded("High failure rate".to_string())
            } else {
                HealthStatus::Healthy
            }
        } else {
            HealthStatus::Unknown
        };

        Ok(health)
    }
}

#[derive(Debug, Clone)]
pub enum HealthStatus {
    Healthy,
    Warning(String),
    Degraded(String),
    Unknown,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pr_automationconfig() {
        let config = PrAutomationConfig::default();
        assert_eq!(config.max_suggestions_per_cycle, 5);
        assert_eq!(config.min_confidence_threshold, 0.7);
        assert!(config.auto_create_prs);
        assert!(config.generate_tests);
    }
}
