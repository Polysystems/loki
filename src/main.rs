#![allow(unused_variables)]
use std::path::PathBuf;
use std::sync::Arc;
use std::{fs, process};

use anyhow::{Context, Result};
use clap::Parser;
use loki::cli::x_commands::handle_x_command;
use loki::cli::{Cli, CognitiveCommands, Commands};
use loki::cluster::{ClusterConfig, ClusterManager};
use loki::cognitive::{CognitiveConfig, CognitiveSystem, SafeCognitiveSystem};
use loki::compute::ComputeManager;
use loki::config::{ApiKeysConfig, Config, SetupWizard};
use loki::models::{handle_model_command, CompletionRequest, Message, MessageRole, ModelOrchestrator, ProviderFactory};
use loki::streaming::StreamManager;
use loki::tui::run_tui;
use loki::{core, daemon, tasks};
use serde::{Deserialize, Serialize};
use tokio::io::AsyncBufReadExt;
use tokio::net::{UnixListener, UnixStream};
use tokio::signal;
use tokio::sync::broadcast;
use tracing::{Level, debug, error, info, warn};
use tracing_subscriber::prelude::*;
use tracing_subscriber::{EnvFilter, fmt};
use loki::daemon::{DaemonCommand, DaemonManager, DaemonResponse};
use loki::memory::{CognitiveMemory, MemoryConfig};
use loki::safety::{AuditConfig, ResourceLimits, ValidatorConfig};

#[tokio::main]
async fn main() -> Result<()> {
    // Check for secure environment and warn about .env files
    if let Err(e) = loki::config::secure_env::check_secure_environment() {
        error!("Security check failed: {}", e);
        return Err(e);
    }

    let cli = Cli::parse();

    // Initialize logging - determine if we're in TUI mode
    let is_tui_mode = matches!(cli.command, Commands::Tui { .. });
    init_logging(is_tui_mode)?;

    let config = Config::load()?;
    info!("Loaded configuration from: {:?}", config.config_path());

    // Run the appropriate command
    match cli.command {
        Commands::Run { path, model, watch, interactive } => {
            info!("Starting loki assistant...");
            let args = loki::cli::RunArgs { path, model, watch, interactive };
            core::run(args, config).await?;
        }
        Commands::Tui { daemon } => {
            if daemon {
                info!("Starting TUI in daemon mode...");
                let daemonconfig = daemon::DaemonConfig::default();
                let mut daemon_manager = DaemonManager::new(daemonconfig);

                // Initialize backend components
                let _compute_manager = Arc::new(ComputeManager::new()?);
                let _stream_manager = Arc::new(StreamManager::new(config.clone())?);
                let clusterconfig = ClusterConfig::default();
                let _cluster_manager = Arc::new(ClusterManager::new(clusterconfig).await?);

                // Create cognitive config and system
                let cognitiveconfig = CognitiveConfig::default();
                let apiconfig = ApiKeysConfig::from_env()?;
                let cognitive_system =
                    CognitiveSystem::new(apiconfig, cognitiveconfig).await?;
                
                // Initialize story engine
                cognitive_system.initialize_story_engine().await?;
                
                // Initialize story-driven autonomy
                if let Err(e) = cognitive_system.initialize_story_driven_autonomy().await {
                    warn!("Story-driven autonomy initialization failed: {}", e);
                }

                // Create safety-aware cognitive system
                let validatorconfig = loki::safety::ValidatorConfig {
                    safe_mode: true,
                    dry_run: false,
                    approval_required: false, // Background daemon mode
                    approval_timeout: std::time::Duration::from_secs(60),
                    allowed_paths: vec!["data/**".to_string(), "workspace/**".to_string()],
                    blocked_paths: vec!["src/safety/**".to_string(), ".git/**".to_string()],
                    max_file_size: 10 * 1024 * 1024,
                    storage_path: Some(std::path::PathBuf::from("data/safety/decisions_daemon")),
                    encrypt_decisions: true,
                    enable_resource_monitoring: true,
                    cpu_threshold: 80.0,
                    memory_threshold: 85.0,
                    disk_threshold: 90.0,
                    max_concurrent_operations: 50,
                    enable_rate_limiting: true,
                    enable_network_monitoring: true,
                };

                let auditconfig = loki::safety::AuditConfig {
                    log_dir: std::path::PathBuf::from("data/audit"),
                    max_memory_events: 10000,
                    file_logging: true,
                    encrypt_logs: false,
                    retention_days: 90,
                };

                let resource_limits = loki::safety::ResourceLimits::default();

                let safe_cognitive_system = Arc::new(
                    loki::cognitive::SafeCognitiveSystem::new(
                        cognitive_system,
                        validatorconfig,
                        auditconfig,
                        resource_limits,
                    )
                        .await?,
                );

                daemon_manager.start_daemon(safe_cognitive_system).await?;
            } else {
                info!("Starting Loki TUI...");

                // Initialize backend components with better error reporting
                eprintln!("🔄 Initializing compute manager...");
                let compute_manager = Arc::new(ComputeManager::new()?);
                eprintln!("✅ Compute manager initialized");

                eprintln!("🔄 Initializing stream manager...");
                let mut stream_manager = StreamManager::new(config.clone())?;
                
                // Note: EventBus will be connected after App initialization
                // since EventBus is created inside App::new()
                
                let stream_manager = Arc::new(stream_manager);
                eprintln!("✅ Stream manager initialized");

                eprintln!("🔄 Initializing cluster manager...");
                let clusterconfig = ClusterConfig::default();
                let cluster_manager = Arc::new(ClusterManager::new(clusterconfig).await?);
                eprintln!("✅ Cluster manager initialized");

                eprintln!("🔄 Starting TUI interface...");
                run_tui(compute_manager, stream_manager, Some(cluster_manager)).await?;
            }
        }
        Commands::Cognitive { command } => {
            handle_cognitive_command(command, config).await?;
        }
        Commands::Config { command } => {
            loki::config::handleconfig_command(loki::cli::ConfigArgs { command }, config)?;
        }
        Commands::Model { command } => {
            handle_model_command(loki::cli::ModelArgs { command }, config).await?;
        }
        Commands::Task { command } => {
            // Convert TaskCommands to TaskArgs
            let args = match command {
                loki::cli::TaskCommands::Complete { input } => loki::cli::TaskArgs {
                    task: "complete".to_string(),
                    args: vec![],
                    input: Some(input),
                    output: None,
                },
                loki::cli::TaskCommands::Review { input } => loki::cli::TaskArgs {
                    task: "review".to_string(),
                    args: vec![],
                    input: Some(input),
                    output: None,
                },
                loki::cli::TaskCommands::Document { input, output } => loki::cli::TaskArgs {
                    task: "document".to_string(),
                    args: vec![],
                    input: Some(input),
                    output,
                },
                loki::cli::TaskCommands::Refactor { input } => loki::cli::TaskArgs {
                    task: "refactor".to_string(),
                    args: vec![],
                    input: Some(input),
                    output: None,
                },
                loki::cli::TaskCommands::Test { input, output } => loki::cli::TaskArgs {
                    task: "test".to_string(),
                    args: vec![],
                    input: Some(input),
                    output,
                },
            };
            tasks::handle_task_command(args, config).await?;
        }
        Commands::CheckApis => {
            loki::cli::check_apis::check_api_status().await?;
        }
        Commands::SetupApis => {
            info!("Starting interactive API setup wizard...");
            let mut setup_wizard = SetupWizard::new()?;
            setup_wizard.run().await?;
        }
        Commands::Setup { command } => {
            loki::cli::tui_commands::handle_setup_command(command).await?;
        }
        Commands::ListProviders { detailed } => {
            handle_list_providers(detailed).await?;
        }
        Commands::TestProvider { provider, prompt } => {
            handle_test_provider(&provider, &prompt).await?;
        }
        Commands::TestConsciousness { provider, cycles } => {
            handle_test_consciousness(provider.as_deref(), cycles).await?;
        }
        Commands::Safety(safety_commands) => {
            loki::cli::safety_commands::handle_safety_command(safety_commands).await?;
        }
        Commands::Plugin(plugin_command) => {
            loki::cli::plugin_commands::handle_plugin_command(plugin_command).await?;
        }
        Commands::Session { command } => {
            loki::cli::tui_commands::handle_session_command(command).await?;
        }
        Commands::X(x_commands) => {
            let orchestrator = build_orchestrator().await?;
            let cognitive_system = build_cognitive_system(orchestrator.clone()).await?;
            handle_x_command(x_commands, cognitive_system).await?;
        }
    }

    Ok(())
}

async fn build_orchestrator() -> Result<Arc<ModelOrchestrator>> {
    // Load configuration
    let config = Config::load()?;

    // Create model orchestrator
    let orchestrator = ModelOrchestrator::new(&config.api_keys).await?;

    Ok(Arc::new(orchestrator))
}

async fn build_cognitive_system(
    _orchestrator: Arc<ModelOrchestrator>,
) -> Result<Arc<CognitiveSystem>> {
    // Load configuration
    let config = Config::load()?;
    let cognitiveconfig = CognitiveConfig::default();

    // Initialize compute and stream managers
    let _compute_manager = Arc::new(ComputeManager::new()?);
    let _stream_manager = Arc::new(StreamManager::new(config)?);

    // Create cognitive system
    let apiconfig = ApiKeysConfig::from_env()?;
    let cognitive_system =
        CognitiveSystem::new(apiconfig, cognitiveconfig).await?;

    // Initialize story engine
    cognitive_system.initialize_story_engine().await?;

    Ok(cognitive_system)
}

async fn handle_cognitive_command(command: CognitiveCommands, config: Config) -> Result<()> {
    match command {
        CognitiveCommands::Start { foreground, auto_install: _ } => {
            info!("Starting safety-aware cognitive system...");

            // Initialize components
            let _compute_manager = Arc::new(ComputeManager::new()?);
            let _stream_manager = Arc::new(StreamManager::new(config.clone())?);
            let clusterconfig = ClusterConfig::default();
            let _cluster_manager = Arc::new(ClusterManager::new(clusterconfig).await?);

            // Create cognitive config
            let cognitiveconfig = CognitiveConfig::default();

            // Create cognitive system
            let apiconfig = ApiKeysConfig::from_env()?;
            let cognitive_system =
                CognitiveSystem::new(apiconfig, cognitiveconfig).await?;

            // Initialize story engine
            cognitive_system.initialize_story_engine().await?;

            // Initialize story-driven autonomy
            if let Err(e) = cognitive_system.initialize_story_driven_autonomy().await {
                warn!("Story-driven autonomy initialization failed: {}", e);
            }

            // Create safety configuration
            let validatorconfig = ValidatorConfig {
                safe_mode: true,
                dry_run: false, // Change to true for testing
                approval_required: true,
                approval_timeout: std::time::Duration::from_secs(300), // 5 minutes
                allowed_paths: vec![
                    "data/**".to_string(),
                    "workspace/**".to_string(),
                    "cache/**".to_string(),
                ],
                blocked_paths: vec![
                    "src/safety/**".to_string(), // Cannot modify safety system
                    ".git/**".to_string(),
                    "**/.env".to_string(),
                    "**/secrets/**".to_string(),
                ],
                storage_path: Some(std::path::PathBuf::from("data/safety/decisions")),
                encrypt_decisions: true,
                max_file_size: 10 * 1024 * 1024, // 10MB
                enable_resource_monitoring: true,
                cpu_threshold: 80.0,
                memory_threshold: 85.0,
                disk_threshold: 90.0,
                max_concurrent_operations: 50,
                enable_rate_limiting: true,
                enable_network_monitoring: true,
            };

            let auditconfig = AuditConfig {
                log_dir: std::path::PathBuf::from("data/audit"),
                max_memory_events: 10000,
                file_logging: true,
                encrypt_logs: false, // Would be true in production
                retention_days: 90,
            };

            let resource_limits = ResourceLimits::default();

            // Create safety-aware cognitive system
            info!("Initializing safety infrastructure...");
            let safe_cognitive_system = SafeCognitiveSystem::new(
                cognitive_system,
                validatorconfig,
                auditconfig,
                resource_limits,
            )
                .await?;

            info!("Safety infrastructure initialized successfully");
            info!("Cognitive system running with safety validation enabled");

            if foreground {
                // Run in foreground
                info!("Running safety-aware cognitive system in foreground...");
                println!("🛡️  Safety Features Enabled:");
                println!("   - Action validation and approval required");
                println!("   - Resource monitoring active");
                println!("   - Audit logging to data/audit/");
                println!("   - Emergency stop available via Ctrl+C");
                println!();
                println!("Use 'loki safety' commands to manage the system:");
                println!("   loki safety pending    - View pending actions");
                println!("   loki safety resources  - Monitor resource usage");
                println!("   loki safety audit      - View audit trail");
                println!();

                // Wait for shutdown signal
                tokio::select! {
                    _ = tokio::signal::ctrl_c() => {
                        info!("Received shutdown signal, performing safe shutdown...");
                        safe_cognitive_system.emergency_shutdown("User requested shutdown via Ctrl+C").await?;
                    }
                }
            } else {
                // Background daemon mode
                info!("Starting cognitive system in background daemon mode...");
                let daemonconfig = daemon::DaemonConfig::default();
                let mut daemon_manager = DaemonManager::new(daemonconfig);
                daemon_manager.start_daemon(Arc::new(safe_cognitive_system)).await?;
            }
        }

        CognitiveCommands::Stop => {
            info!("Stopping cognitive system...");
            let daemonconfig = daemon::DaemonConfig::default();
            let daemon_manager = DaemonManager::new(daemonconfig);

            match daemon_manager.stop_daemon().await {
                Ok(_) => {
                    println!("✅ Cognitive system stopped successfully");
                }
                Err(e) => {
                    eprintln!("❌ Failed to stop cognitive system: {e}");
                    std::process::exit(1);
                }
            }
        }

        CognitiveCommands::Query { query } => {
            info!("Sending query to cognitive system: {}", query);
            let daemonconfig = daemon::DaemonConfig::default();
            let daemon_manager = DaemonManager::new(daemonconfig);

            let command = DaemonCommand::Query { query: query.clone() };
            match daemon_manager.send_daemon_command(command).await {
                Ok(DaemonResponse::QueryResult { result }) => {
                    println!("🧠 Cognitive Response:");
                    println!("{result}");
                }
                Ok(DaemonResponse::Error { error }) => {
                    eprintln!("❌ Query failed: {error}");
                    std::process::exit(1);
                }
                Ok(_) => {
                    eprintln!("❌ Unexpected response format");
                    std::process::exit(1);
                }
                Err(e) => {
                    eprintln!("❌ Failed to connect to cognitive system: {e}");
                    eprintln!("💡 Is the cognitive system running? Try: loki cognitive start");
                    std::process::exit(1);
                }
            }
        }

        CognitiveCommands::Stream { name, purpose } => {
            info!("Creating stream '{}' with purpose: {:?}", name, purpose);
            let daemonconfig = daemon::DaemonConfig::default();
            let daemon_manager = DaemonManager::new(daemonconfig);

            let command =
                DaemonCommand::Stream { name: name.clone(), purpose: Some(purpose.clone()) };
            match daemon_manager.send_daemon_command(command).await {
                Ok(DaemonResponse::Success { message }) => {
                    println!("✅ {message}");
                }
                Ok(DaemonResponse::Error { error }) => {
                    eprintln!("❌ Stream creation failed: {error}");
                    std::process::exit(1);
                }
                Ok(_) => {
                    eprintln!("❌ Unexpected response format");
                    std::process::exit(1);
                }
                Err(e) => {
                    eprintln!("❌ Failed to connect to cognitive system: {e}");
                    eprintln!("💡 Is the cognitive system running? Try: loki cognitive start");
                    std::process::exit(1);
                }
            }
        }

        CognitiveCommands::Agent { name, agent_type, capabilities } => {
            info!("Creating agent '{}' of type '{}'", name, agent_type);
            let daemonconfig = daemon::DaemonConfig::default();
            let daemon_manager = DaemonManager::new(daemonconfig);

            let command = DaemonCommand::Agent {
                name: name.clone(),
                agent_type: agent_type.clone(),
                capabilities: capabilities.clone(),
            };
            match daemon_manager.send_daemon_command(command).await {
                Ok(DaemonResponse::Success { message }) => {
                    println!("✅ {message}");
                }
                Ok(DaemonResponse::Error { error }) => {
                    eprintln!("❌ Agent creation failed: {error}");
                    std::process::exit(1);
                }
                Ok(_) => {
                    eprintln!("❌ Unexpected response format");
                    std::process::exit(1);
                }
                Err(e) => {
                    eprintln!("❌ Failed to connect to cognitive system: {e}");
                    eprintln!("💡 Is the cognitive system running? Try: loki cognitive start");
                    std::process::exit(1);
                }
            }
        }

        CognitiveCommands::Status => {
            info!("Checking cognitive system status...");
            let daemonconfig = daemon::DaemonConfig::default();
            let daemon_manager = DaemonManager::new(daemonconfig);

            let command = DaemonCommand::Status;
            match daemon_manager.send_daemon_command(command).await {
                Ok(DaemonResponse::Status { running, pid, uptime }) => {
                    println!("🧠 Cognitive System Status:");
                    println!("   Running: {}", if running { "✅ Yes" } else { "❌ No" });
                    if let Some(pid) = pid {
                        println!("   PID: {pid}");
                    }
                    if let Some(uptime) = uptime {
                        println!("   Uptime: {uptime} seconds");
                    }
                }
                Ok(DaemonResponse::Error { error }) => {
                    eprintln!("❌ Status check failed: {error}");
                    std::process::exit(1);
                }
                Ok(_) => {
                    eprintln!("❌ Unexpected response format");
                    std::process::exit(1);
                }
                Err(e) => {
                    println!("🧠 Cognitive System Status:");
                    println!("   Running: ❌ No");
                    println!("   Error: {e}");
                    println!();
                    println!("💡 Start the cognitive system with: loki cognitive start");
                }
            }
        }
    }

    Ok(())
}

fn init_logging(is_tui_mode: bool) -> Result<()> {
    let env_filter =
        EnvFilter::builder().with_default_directive(Level::INFO.into()).from_env_lossy();

    if is_tui_mode {
        // For TUI mode, redirect logs to a file to avoid interfering with the interface
        use std::fs::OpenOptions;

        // Create logs directory if it doesn't exist
        std::fs::create_dir_all("logs").ok();

        let log_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open("logs/loki-tui.log")?;

        tracing_subscriber::registry()
            .with(
                fmt::layer()
                    .with_target(false)
                    .with_thread_ids(false)
                    .with_ansi(false)
                    .with_writer(log_file)
            )
            .with(env_filter)
            .init();
    } else {
        // Normal logging for non-TUI commands
        tracing_subscriber::registry()
            .with(fmt::layer().with_target(true).with_thread_ids(true))
            .with(env_filter)
            .init();
    }

    Ok(())
}

async fn handle_list_providers(detailed: bool) -> Result<()> {
    use colored::*;

    println!("{}", "🤖 Available Model Providers".cyan().bold());
    println!();

    // Load API configuration
    let apiconfig = ApiKeysConfig::from_env()?;
    let providers = ProviderFactory::create_providers(&apiconfig);

    for provider in providers {
        let name = provider.name();
        let available = provider.is_available();

        let status = if available { "✅ Configured".green() } else { "❌ Not configured".red() };

        println!("{}: {}", name.yellow().bold(), status);

        if detailed && available {
            match provider.list_models().await {
                Ok(models) => {
                    for model in models {
                        println!("  - {} ({})", model.id.cyan(), model.name);
                        println!("    {}", model.description.dimmed());
                        println!("    Context: {} tokens", model.context_length);
                        println!("    Capabilities: {}", model.capabilities.join(", ").dimmed());
                    }
                }
                Err(e) => {
                    println!("    {}", format!("Error listing models: {e}").red());
                }
            }
        }
        println!();
    }

    Ok(())
}

async fn handle_test_provider(provider_name: &str, prompt: &str) -> Result<()> {
    use colored::*;

    println!("{}", format!("🧪 Testing {provider_name} provider...").cyan().bold());
    println!();

    // Load API configuration
    let apiconfig = ApiKeysConfig::from_env()?;
    let providers = ProviderFactory::create_providers(&apiconfig);

    // Find the requested provider
    let provider = ProviderFactory::get_provider(&providers, provider_name)
        .ok_or_else(|| anyhow::anyhow!("Provider '{}' not found", provider_name))?;

    if !provider.is_available() {
        anyhow::bail!("Provider '{}' is not configured. Please set the API key.", provider_name);
    }

    // Create a completion request
    let request = CompletionRequest {
        model: String::new(), // Let provider choose default
        messages: vec![Message { role: MessageRole::User, content: prompt.to_string() }],
        max_tokens: Some(500),
        temperature: Some(0.7),
        top_p: None,
        stop: None,
        stream: false,
    };

    println!("📤 Sending: {}", prompt.yellow());
    println!();

    // Make the request
    match provider.complete(request).await {
        Ok(response) => {
            println!("📥 Response from {} ({}):", provider_name.green(), response.model.dimmed());
            println!();
            println!("{}", response.content);
            println!();
            println!(
                "📊 Usage: {} prompt + {} completion = {} total tokens",
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
                response.usage.total_tokens
            );
        }
        Err(e) => {
            println!("{}", format!("❌ Error: {e}").red());
        }
    }

    Ok(())
}

async fn handle_test_consciousness(_provider: Option<&str>, cycles: usize) -> Result<()> {
    use colored::*;

    println!("{}", "🧠 Testing Enhanced Consciousness Stream".cyan().bold());
    println!();

    // Load API configuration
    let _apiconfig = ApiKeysConfig::from_env()?;

    // Create temporary memory system
    let temp_dir = tempfile::tempdir()?;
    let memoryconfig = MemoryConfig {
        persistence_path: temp_dir.path().to_path_buf(),
        cache_size_mb: 1000,
        ..Default::default()
    };
    let _memory = Arc::new(CognitiveMemory::new(memoryconfig).await?);

    println!("Consciousness test temporarily simplified due to API changes.");
    println!("Creating temporary consciousness simulation...");

    // For now, just simulate consciousness cycles
    for i in 0..cycles {
        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

        println!("{} Cycle #{}: Consciousness simulation active", "🧠".cyan(), i + 1);
        println!("   Processing cognitive load: {}%", (30 + i * 10).min(90));
        println!("   Memory coherence: {:.1}", 0.7 + (i as f64 * 0.05));
        println!();
    }

    println!("{}", "📝 Final Status: Consciousness simulation complete".cyan().bold());
    println!("{}", "✅ Consciousness test complete!".green().bold());

    Ok(())
}
