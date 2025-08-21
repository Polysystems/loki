use clap::{Parser, Subcommand};
use std::path::PathBuf;

pub mod check_apis;
pub mod plugin_commands;
pub mod safety_commands;
pub mod tui_commands;
pub mod ui_commands;
pub mod x_commands;

/// Command-line interface for Loki
#[derive(Parser)]
#[command(
    name = "loki",
    about = "A lightweight AI coding assistant that runs locally",
    version,
    author
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Run the assistant on a directory or file
    Run {
        /// Path to analyze (defaults to current directory)
        #[arg(default_value = ".")]
        path: PathBuf,

        /// Model to use for analysis
        #[arg(short, long)]
        model: Option<String>,

        /// Watch for file changes
        #[arg(short, long)]
        watch: bool,

        /// Interactive mode
        #[arg(short, long)]
        interactive: bool,
    },

    /// Launch the Terminal User Interface
    Tui {
        /// Start in daemon mode (background)
        #[arg(short, long)]
        daemon: bool,
    },

    /// Model setup templates for TUI
    Setup {
        #[command(subcommand)]
        command: SetupCommands,
    },

    /// Model session management
    Session {
        #[command(subcommand)]
        command: SessionCommands,
    },

    /// Launch the Cognitive System
    Cognitive {
        #[command(subcommand)]
        command: CognitiveCommands,
    },

    /// Manage configuration
    Config {
        #[command(subcommand)]
        command: ConfigCommands,
    },

    /// Manage models
    Model {
        #[command(subcommand)]
        command: ModelCommands,
    },

    /// Run specific tasks
    Task {
        #[command(subcommand)]
        command: TaskCommands,
    },

    /// X/Twitter integration commands
    X(x_commands::XCommands),

    /// Safety management commands
    Safety(safety_commands::SafetyCommands),

    /// Plugin management commands
    Plugin(plugin_commands::PluginArgs),

    /// Check API configuration status
    #[command(name = "check-apis")]
    CheckApis,

    /// Interactive API key setup wizard
    #[command(name = "setup-apis")]
    SetupApis,

    /// List available model providers
    #[command(name = "list-providers")]
    ListProviders {
        /// Show detailed model information
        #[arg(short, long)]
        detailed: bool,
    },

    /// Test a model provider
    #[command(name = "test-provider")]
    TestProvider {
        /// Provider name (openai, anthropic, mistral, etc.)
        provider: String,

        /// Optional test prompt
        #[arg(short, long, default_value = "Hello! Can you introduce yourself?")]
        prompt: String,
    },

    /// Test enhanced consciousness with cloud providers
    #[command(name = "test-consciousness")]
    TestConsciousness {
        /// Provider to use (defaults to best available)
        #[arg(short, long)]
        provider: Option<String>,

        /// Number of thought cycles to run
        #[arg(short = 'c', long, default_value = "5")]
        cycles: usize,
    },
}

#[derive(Parser)]
pub struct RunArgs {
    /// Path to the project or file to analyze
    #[arg(default_value = ".")]
    pub path: PathBuf,

    /// Watch for changes and provide real-time assistance
    #[arg(short, long)]
    pub watch: bool,

    /// Model to use for analysis
    #[arg(short, long)]
    pub model: Option<String>,

    /// Enable interactive mode
    #[arg(short, long)]
    pub interactive: bool,
}

#[derive(Parser)]
pub struct ConfigArgs {
    #[command(subcommand)]
    pub command: ConfigCommands,
}

#[derive(Subcommand)]
pub enum ConfigCommands {
    /// Show current configuration
    Show,

    /// Set a configuration value
    Set {
        /// Configuration key
        key: String,
        /// Configuration value
        value: String,
    },

    /// Get a configuration value
    Get {
        /// Configuration key
        key: String,
    },

    /// Reset configuration to defaults
    Reset,
}

#[derive(Parser)]
pub struct ModelArgs {
    #[command(subcommand)]
    pub command: ModelCommands,
}

#[derive(Subcommand)]
pub enum ModelCommands {
    /// List available models
    List,

    /// Download a model
    Download {
        /// Model name or URL
        model: String,
    },

    /// Remove a model
    Remove {
        /// Model name
        model: String,
    },

    /// Show model information
    Info {
        /// Model name
        model: String,
    },

    /// Test model orchestration
    TestOrchestration {
        /// Task type to test
        #[arg(short, long, default_value = "code_generation")]
        task: String,

        /// Content to process
        #[arg(short, long, default_value = "Write a hello world function in Python")]
        content: String,

        /// Prefer local models
        #[arg(short, long)]
        prefer_local: bool,

        /// Maximum cost in cents
        #[arg(long)]
        max_cost_cents: Option<f32>,
    },

    /// Show orchestration status
    Status,

    /// Configure model orchestration
    Configure {
        /// Configuration file path
        #[arg(short, long)]
        config: Option<PathBuf>,

        /// Create default configuration
        #[arg(short, long)]
        default: bool,
    },

    /// Load a model on demand
    Load {
        /// Model ID to load
        model_id: String,
    },

    /// Unload a model to free resources
    Unload {
        /// Model ID to unload
        model_id: String,
    },

    /// List all models (local + API) with capabilities
    ListAll {
        /// Show detailed capability information
        #[arg(short, long)]
        detailed: bool,
    },

    /// Show hardware status and requirements
    Hardware,

    /// Reload model configuration
    Reload,

    /// Test ensemble execution with multiple models
    TestEnsemble {
        /// Task type to test
        #[arg(short, long, default_value = "code_generation")]
        task: String,

        /// Content to process
        #[arg(short, long, default_value = "Write a complex sorting algorithm in Python")]
        content: String,

        /// Voting strategy (best_quality, consensus, majority)
        #[arg(short, long, default_value = "best_quality")]
        voting_strategy: String,

        /// Minimum models required
        #[arg(long, default_value = "2")]
        min_models: usize,

        /// Maximum models to use
        #[arg(long, default_value = "4")]
        max_models: usize,
    },

    /// Test adaptive learning system
    TestLearning {
        /// Task type to test learning with
        #[arg(short, long, default_value = "code_generation")]
        task: String,

        /// Content to process
        #[arg(short, long, default_value = "Write a hello world function")]
        content: String,

        /// Number of test iterations
        #[arg(short, long, default_value = "5")]
        iterations: usize,

        /// Simulate user feedback (0.0-1.0)
        #[arg(long)]
        feedback: Option<f32>,
    },

    /// Trigger manual learning update
    UpdateLearning,

    /// Show learning statistics and patterns
    LearningStats {
        /// Show detailed model profiles
        #[arg(short, long)]
        detailed: bool,

        /// Show task patterns
        #[arg(short, long)]
        patterns: bool,
    },

    /// Test streaming execution
    TestStreaming {
        /// Task type to test streaming with
        #[arg(short, long, default_value = "code_generation")]
        task: String,

        /// Content to process
        #[arg(short, long, default_value = "Write a complex web scraper in Python")]
        content: String,

        /// Preferred model for streaming
        #[arg(short, long)]
        model: Option<String>,

        /// Buffer size for streaming
        #[arg(short, long, default_value = "1024")]
        buffer_size: usize,
    },

    /// List active streaming sessions
    ListStreams,

    /// Cancel a streaming session
    CancelStream {
        /// Stream ID to cancel
        stream_id: String,
    },

    /// Monitor streaming performance
    StreamingStats {
        /// Show detailed metrics
        #[arg(short, long)]
        detailed: bool,
    },

    /// Test consciousness-orchestration integration
    TestConsciousness {
        /// Thought content to process
        #[arg(short, long, default_value = "I am contemplating my existence and purpose")]
        content: String,

        /// Thought type
        #[arg(short, long, default_value = "reflective")]
        thought_type: String,

        /// Use enhanced processing
        #[arg(short, long)]
        enhanced: bool,

        /// Enable streaming for consciousness
        #[arg(short, long)]
        streaming: bool,
    },

    /// Test conscious decision making
    TestDecision {
        /// Decision context
        #[arg(short, long, default_value = "Should I prioritize learning or creation?")]
        context: String,

        /// Decision urgency (0.0-1.0)
        #[arg(short, long, default_value = "0.7")]
        urgency: f32,

        /// Use orchestration for decision
        #[arg(short, long)]
        orchestrated: bool,
    },

    /// Test emotional processing with consciousness
    TestEmotion {
        /// Emotion to process
        #[arg(short, long, default_value = "curiosity")]
        emotion: String,

        /// Emotion intensity (0.0-1.0)
        #[arg(short, long, default_value = "0.8")]
        intensity: f32,

        /// Emotional context
        #[arg(short, long, default_value = "Learning about consciousness")]
        context: String,
    },

    /// Show consciousness integration metrics
    ConsciousnessStats {
        /// Show detailed cognitive metrics
        #[arg(short, long)]
        detailed: bool,

        /// Show processing sessions
        #[arg(short, long)]
        sessions: bool,

        /// Show orchestration metrics
        #[arg(short, long)]
        orchestration: bool,
    },

    /// Manage cost tracking and budgets
    Cost {
        #[command(subcommand)]
        command: CostCommands,
    },

    /// Manage model fine-tuning and adaptation
    FineTuning {
        #[command(subcommand)]
        command: FineTuningCommands,
    },

    /// Manage distributed model serving
    Distributed {
        #[command(subcommand)]
        command: DistributedCommands,
    },

    /// Manage benchmarking and performance analysis
    Benchmark {
        #[command(subcommand)]
        command: BenchmarkCommands,
    },
}

#[derive(Parser, Clone)]
pub struct TaskArgs {
    /// Task to execute
    pub task: String,

    /// Task-specific arguments
    #[arg(trailing_var_arg = true)]
    pub args: Vec<String>,

    /// Input file or directory
    #[arg(short, long)]
    pub input: Option<PathBuf>,

    /// Output file or directory
    #[arg(short, long)]
    pub output: Option<PathBuf>,
}

#[derive(Subcommand)]
pub enum CognitiveCommands {
    /// Start the cognitive system daemon
    Start {
        /// Run in foreground
        #[arg(short, long)]
        foreground: bool,

        /// Enable Ollama auto-install
        #[arg(long)]
        auto_install: bool,
    },

    /// Stop the cognitive system
    Stop,

    /// Query the cognitive system
    Query {
        /// The query to process
        query: String,
    },

    /// Start an autonomous stream
    Stream {
        /// Stream name
        name: String,

        /// Stream purpose
        purpose: String,
    },

    /// Deploy an agent
    Agent {
        /// Agent name
        name: String,

        /// Agent type
        #[arg(short = 't', long)]
        agent_type: String,

        /// Agent capabilities
        #[arg(short, long)]
        capabilities: Vec<String>,
    },

    /// Show cognitive system status
    Status,
}

#[derive(Subcommand)]
pub enum TaskCommands {
    /// Code completion
    Complete {
        /// File to complete
        #[arg(short, long)]
        input: PathBuf,
    },

    /// Code review
    Review {
        /// File or directory to review
        #[arg(short, long)]
        input: PathBuf,
    },

    /// Generate documentation
    Document {
        /// File to document
        #[arg(short, long)]
        input: PathBuf,

        /// Output file
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Suggest refactoring
    Refactor {
        /// File to refactor
        #[arg(short, long)]
        input: PathBuf,
    },

    /// Generate tests
    Test {
        /// File to test
        #[arg(short, long)]
        input: PathBuf,

        /// Output directory
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
}

#[derive(Subcommand)]
pub enum SetupCommands {
    /// List all available setup templates
    List,

    /// Show detailed information about a template
    Info {
        /// Template ID (lightning-fast, balanced-pro, premium-quality, etc.)
        template_id: String,
    },

    /// Launch a setup template
    Launch {
        /// Template ID to launch
        template_id: String,

        /// Custom session name (optional)
        #[arg(short, long)]
        name: Option<String>,
    },

    /// Quick launch templates
    #[command(name = "lightning-fast")]
    LightningFast {
        /// Custom session name
        #[arg(short, long)]
        name: Option<String>,
    },

    #[command(name = "balanced-pro")]
    BalancedPro {
        /// Custom session name
        #[arg(short, long)]
        name: Option<String>,
    },

    #[command(name = "premium-quality")]
    PremiumQuality {
        /// Custom session name
        #[arg(short, long)]
        name: Option<String>,
    },

    #[command(name = "research-beast")]
    ResearchBeast {
        /// Custom session name
        #[arg(short, long)]
        name: Option<String>,
    },

    #[command(name = "code-master")]
    CodeMaster {
        /// Custom session name
        #[arg(short, long)]
        name: Option<String>,
    },

    #[command(name = "writing-pro")]
    WritingPro {
        /// Custom session name
        #[arg(short, long)]
        name: Option<String>,
    },

    /// Create a custom template
    Create {
        /// Template name
        name: String,

        /// Local models to include
        #[arg(short, long)]
        local_models: Vec<String>,

        /// API models to include
        #[arg(short, long)]
        api_models: Vec<String>,

        /// Cost estimate per hour
        #[arg(short, long)]
        cost: Option<f32>,
    },
}

#[derive(Subcommand)]
pub enum SessionCommands {
    /// List all active sessions
    List,

    /// Show detailed session information
    Info {
        /// Session ID
        session_id: String,
    },

    /// Stop a session
    Stop {
        /// Session ID
        session_id: String,
    },

    /// Stop all sessions
    StopAll,

    /// Show session logs
    Logs {
        /// Session ID
        session_id: String,

        /// Number of log lines to show
        #[arg(short, long, default_value = "50")]
        lines: usize,

        /// Follow logs (tail -f style)
        #[arg(short, long)]
        follow: bool,
    },

    /// Show cost analytics
    Costs {
        /// Time period (today, week, month)
        #[arg(short, long, default_value = "today")]
        period: String,

        /// Show breakdown by model
        #[arg(short, long)]
        breakdown: bool,
    },

    /// Monitor session in real-time
    Monitor {
        /// Session ID
        session_id: String,

        /// Update interval in seconds
        #[arg(short, long, default_value = "2")]
        interval: u64,
    },
}

#[derive(Subcommand)]
pub enum CostCommands {
    /// Show current budget status
    Status {
        /// Show detailed cost breakdown
        #[arg(short, long)]
        detailed: bool,
    },

    /// Set budget limits
    SetBudget {
        /// Daily budget limit in cents
        #[arg(long)]
        daily: Option<f32>,

        /// Weekly budget limit in cents
        #[arg(long)]
        weekly: Option<f32>,

        /// Monthly budget limit in cents
        #[arg(long)]
        monthly: Option<f32>,
    },

    /// Show cost analytics and trends
    Analytics {
        /// Time period (daily, weekly, monthly)
        #[arg(short, long, default_value = "weekly")]
        period: String,

        /// Show optimization recommendations
        #[arg(short, long)]
        recommendations: bool,
    },

    /// Track specific task costs
    Track {
        /// Task type to track
        #[arg(short, long)]
        task_type: String,

        /// Content to process and cost
        #[arg(short, long)]
        content: String,

        /// Run cost simulation without execution
        #[arg(short, long)]
        simulate: bool,
    },

    /// Show pricing information for providers
    Pricing {
        /// Provider to show pricing for
        #[arg(short, long)]
        provider: Option<String>,
    },

    /// Export cost data
    Export {
        /// Output format (json, csv, yaml)
        #[arg(short, long, default_value = "json")]
        format: String,

        /// Output file path
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Date range to export
        #[arg(short, long)]
        days: Option<u32>,
    },

    /// Reset cost tracking data
    Reset {
        /// Confirm reset operation
        #[arg(long)]
        confirm: bool,
    },

    /// Test budget enforcement
    TestBudget {
        /// Simulate exceeding budget threshold
        #[arg(short, long)]
        exceed_threshold: bool,
    },
}

#[derive(Subcommand)]
pub enum FineTuningCommands {
    /// Show fine-tuning system status
    Status {
        /// Show detailed job information
        #[arg(short, long)]
        detailed: bool,
    },

    /// Start fine-tuning for a task type
    Start {
        /// Task type to fine-tune (code_generation, logical_reasoning, creative_writing, etc.)
        #[arg(short, long)]
        task_type: String,

        /// Learning rate for training
        #[arg(long, default_value = "5e-5")]
        learning_rate: f32,

        /// Number of training epochs
        #[arg(long, default_value = "3")]
        epochs: usize,

        /// Batch size for training
        #[arg(long, default_value = "8")]
        batch_size: usize,

        /// Maximum cost for the job (cents)
        #[arg(long)]
        max_cost_cents: Option<f32>,
    },

    /// List training data available
    Data {
        /// Task type to show data for
        #[arg(short, long)]
        task_type: Option<String>,

        /// Show data quality statistics
        #[arg(short, long)]
        quality: bool,

        /// Show recent samples
        #[arg(short, long)]
        samples: bool,
    },

    /// Monitor active fine-tuning jobs
    Jobs {
        /// Show only active jobs
        #[arg(short, long)]
        active_only: bool,

        /// Show job details
        #[arg(short, long)]
        details: bool,

        /// Follow job progress
        #[arg(short, long)]
        follow: bool,
    },

    /// Cancel a fine-tuning job
    Cancel {
        /// Job ID to cancel
        job_id: String,
    },

    /// Configure fine-tuning settings
    Configure {
        /// Enable automatic fine-tuning
        #[arg(long)]
        auto_tuning: Option<bool>,

        /// Minimum training samples required
        #[arg(long)]
        min_samples: Option<usize>,

        /// Quality improvement threshold
        #[arg(long)]
        quality_threshold: Option<f32>,

        /// Maximum concurrent jobs
        #[arg(long)]
        max_concurrent: Option<usize>,
    },

    /// Test model adaptation
    TestAdaptation {
        /// Task type for adaptation
        #[arg(short, long)]
        task_type: String,

        /// Test content
        #[arg(short, long)]
        content: String,

        /// Adaptation strategy (quality, efficiency, specialization)
        #[arg(short, long, default_value = "quality")]
        strategy: String,
    },

    /// Show adaptation history
    History {
        /// Number of recent adaptations to show
        #[arg(short, long, default_value = "10")]
        limit: usize,

        /// Filter by task type
        #[arg(short, long)]
        task_type: Option<String>,

        /// Show performance impact
        #[arg(short, long)]
        performance: bool,
    },

    /// Export training data
    Export {
        /// Task type to export
        #[arg(short, long)]
        task_type: String,

        /// Output format (json, csv, jsonl)
        #[arg(short, long, default_value = "jsonl")]
        format: String,

        /// Output file path
        #[arg(short, long)]
        output: PathBuf,

        /// Maximum samples to export
        #[arg(long)]
        max_samples: Option<usize>,
    },

    /// Evaluate model performance
    Evaluate {
        /// Model to evaluate
        #[arg(short, long)]
        model: String,

        /// Evaluation dataset
        #[arg(short, long)]
        dataset: Option<String>,

        /// Metrics to compute
        #[arg(long)]
        metrics: Vec<String>,
    },

    /// Start A/B testing
    ABTest {
        /// Model A for comparison
        #[arg(long)]
        model_a: String,

        /// Model B for comparison
        #[arg(long)]
        model_b: String,

        /// Traffic split (0.0-1.0)
        #[arg(long, default_value = "0.5")]
        split: f32,

        /// Test duration in days
        #[arg(long, default_value = "7")]
        duration: u32,
    },
}

#[derive(Subcommand)]
pub enum DistributedCommands {
    /// Show cluster status
    Status {
        /// Show detailed node information
        #[arg(short, long)]
        detailed: bool,

        /// Show cluster topology
        #[arg(short, long)]
        topology: bool,
    },

    /// Start distributed serving
    Start {
        /// Cluster name
        #[arg(short, long)]
        cluster: Option<String>,

        /// Node role (coordinator, worker, gateway, storage, monitor)
        #[arg(short, long, default_value = "worker")]
        role: String,

        /// Bind address
        #[arg(short, long, default_value = "0.0.0.0:8080")]
        bind: String,

        /// Bootstrap nodes for discovery
        #[arg(long)]
        bootstrap: Vec<String>,

        /// Enable replication
        #[arg(long)]
        replication: bool,
    },

    /// Stop distributed serving
    Stop {
        /// Force shutdown
        #[arg(short, long)]
        force: bool,

        /// Graceful shutdown timeout (seconds)
        #[arg(long, default_value = "30")]
        timeout: u64,
    },

    /// List cluster nodes
    Nodes {
        /// Show only healthy nodes
        #[arg(long)]
        healthy_only: bool,

        /// Filter by node role
        #[arg(short, long)]
        role: Option<String>,

        /// Filter by zone
        #[arg(short, long)]
        zone: Option<String>,

        /// Show node resources
        #[arg(long)]
        resources: bool,
    },

    /// Join an existing cluster
    Join {
        /// Bootstrap node address
        bootstrap_node: String,

        /// Node role to join as
        #[arg(short, long, default_value = "worker")]
        role: String,

        /// Bind address for this node
        #[arg(short, long, default_value = "0.0.0.0:8080")]
        bind: String,
    },

    /// Leave the cluster
    Leave {
        /// Graceful leave
        #[arg(short, long)]
        graceful: bool,

        /// Transfer replicas before leaving
        #[arg(long)]
        transfer_replicas: bool,
    },

    /// Configure cluster settings
    Configure {
        /// Load balancing strategy
        #[arg(long)]
        load_balancing: Option<String>,

        /// Replication factor
        #[arg(long)]
        replication_factor: Option<u32>,

        /// Health check interval (ms)
        #[arg(long)]
        health_interval: Option<u64>,

        /// Enable service mesh
        #[arg(long)]
        service_mesh: Option<bool>,
    },

    /// Monitor cluster health
    Monitor {
        /// Update interval (seconds)
        #[arg(short, long, default_value = "5")]
        interval: u64,

        /// Show metrics
        #[arg(short, long)]
        metrics: bool,

        /// Follow logs
        #[arg(short, long)]
        logs: bool,
    },

    /// Test distributed execution
    Test {
        /// Task type to test
        #[arg(short, long, default_value = "code_generation")]
        task_type: String,

        /// Test content
        #[arg(short, long)]
        content: String,

        /// Target node (optional)
        #[arg(long)]
        target_node: Option<String>,

        /// Number of concurrent requests
        #[arg(long, default_value = "1")]
        concurrency: usize,
    },

    /// Show load balancing statistics
    LoadBalancing {
        /// Time window (minutes)
        #[arg(short, long, default_value = "60")]
        window: u32,

        /// Show per-node breakdown
        #[arg(short, long)]
        per_node: bool,

        /// Show routing decisions
        #[arg(short, long)]
        routing: bool,
    },

    /// Manage model replication
    Replication {
        #[command(subcommand)]
        command: ReplicationCommands,
    },

    /// Network diagnostics
    Network {
        /// Test connectivity to all nodes
        #[arg(short, long)]
        test_connectivity: bool,

        /// Show bandwidth usage
        #[arg(short, long)]
        bandwidth: bool,

        /// Show latency matrix
        #[arg(short, long)]
        latency: bool,
    },

    /// Security management
    Security {
        /// Generate TLS certificates
        #[arg(long)]
        generate_certs: bool,

        /// Rotate encryption keys
        #[arg(long)]
        rotate_keys: bool,

        /// Show security status
        #[arg(long)]
        status: bool,
    },
}

#[derive(Subcommand)]
pub enum ReplicationCommands {
    /// Show replication status
    Status {
        /// Show detailed replica information
        #[arg(short, long)]
        detailed: bool,
    },

    /// Start replication for a model
    Start {
        /// Model ID to replicate
        model_id: String,

        /// Target nodes
        #[arg(short, long)]
        targets: Vec<String>,

        /// Replication factor
        #[arg(short, long, default_value = "2")]
        factor: u32,
    },

    /// Stop replication for a model
    Stop {
        /// Model ID
        model_id: String,

        /// Remove all replicas
        #[arg(long)]
        remove_all: bool,
    },

    /// Sync model replicas
    Sync {
        /// Model ID (sync all if not specified)
        model_id: Option<String>,

        /// Force sync even if up-to-date
        #[arg(short, long)]
        force: bool,
    },

    /// Verify replica integrity
    Verify {
        /// Model ID
        model_id: String,

        /// Fix corrupted replicas
        #[arg(long)]
        fix: bool,
    },
}

#[derive(Subcommand)]
pub enum BenchmarkCommands {
    /// Show benchmarking system status
    Status {
        /// Show detailed performance metrics
        #[arg(short, long)]
        detailed: bool,

        /// Show recent benchmark history
        #[arg(long)]
        history: bool,
    },

    /// Run comprehensive benchmark suite
    Run {
        /// Benchmark suite name
        #[arg(short, long, default_value = "comprehensive")]
        suite: String,

        /// Custom workload configuration file
        #[arg(short, long)]
        config: Option<PathBuf>,

        /// Enable detailed profiling
        #[arg(short, long)]
        profiling: bool,

        /// Save results to file
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Run specific workload benchmark
    Workload {
        /// Workload name (code_generation, logical_reasoning, creative_writing, data_analysis)
        workload: String,

        /// Number of test iterations
        #[arg(short, long, default_value = "10")]
        iterations: usize,

        /// Concurrency level
        #[arg(short, long, default_value = "1")]
        concurrency: usize,

        /// Custom test prompt
        #[arg(short, long)]
        prompt: Option<String>,
    },

    /// Run load testing
    Load {
        /// Maximum concurrent users
        #[arg(short, long, default_value = "10")]
        users: usize,

        /// Test duration in seconds
        #[arg(short, long, default_value = "60")]
        duration: u64,

        /// Ramp up time in seconds
        #[arg(short, long, default_value = "10")]
        ramp_up: u64,

        /// Request pattern (constant, burst, sine, random)
        #[arg(short, long, default_value = "constant")]
        pattern: String,
    },

    /// Start performance profiling session
    Profile {
        /// Profiling session name
        session_name: String,

        /// Profiling duration in seconds
        #[arg(short, long, default_value = "300")]
        duration: u64,

        /// Enable CPU profiling
        #[arg(long)]
        cpu: bool,

        /// Enable memory profiling
        #[arg(long)]
        memory: bool,

        /// Enable network profiling
        #[arg(long)]
        network: bool,

        /// Sampling rate (0.0-1.0)
        #[arg(short, long, default_value = "0.1")]
        sampling_rate: f32,
    },

    /// Stop profiling session
    StopProfile {
        /// Session name to stop
        session_name: String,

        /// Generate report after stopping
        #[arg(short, long)]
        report: bool,
    },

    /// List active profiling sessions
    Sessions {
        /// Show detailed session information
        #[arg(short, long)]
        detailed: bool,
    },

    /// Show performance analysis and recommendations
    Analyze {
        /// Benchmark result ID to analyze
        #[arg(long)]
        result_id: Option<String>,

        /// Number of recent results to analyze
        #[arg(short, long, default_value = "5")]
        recent: usize,

        /// Include trend analysis
        #[arg(short, long)]
        trends: bool,

        /// Include anomaly detection
        #[arg(short, long)]
        anomalies: bool,

        /// Generate optimization recommendations
        #[arg(short, long)]
        optimize: bool,
    },

    /// Compare benchmark results
    Compare {
        /// First benchmark result ID
        baseline: String,

        /// Second benchmark result ID
        comparison: String,

        /// Show detailed comparison
        #[arg(short, long)]
        detailed: bool,
    },

    /// Show historical performance trends
    Trends {
        /// Metric to analyze (latency, quality, cost, throughput)
        #[arg(short, long, default_value = "latency")]
        metric: String,

        /// Time window in days
        #[arg(short, long, default_value = "30")]
        days: u32,

        /// Show predictions
        #[arg(short, long)]
        predict: bool,
    },

    /// Check for performance regressions
    Regressions {
        /// Show detailed regression analysis
        #[arg(short, long)]
        detailed: bool,

        /// Only show active regressions
        #[arg(short, long)]
        active_only: bool,

        /// Minimum severity level (low, medium, high, critical)
        #[arg(short, long)]
        severity: Option<String>,
    },

    /// Export benchmark results
    Export {
        /// Output format (json, csv, html, pdf)
        #[arg(short, long, default_value = "json")]
        format: String,

        /// Output file path
        #[arg(short, long)]
        output: PathBuf,

        /// Number of recent results to export
        #[arg(short, long, default_value = "10")]
        limit: usize,

        /// Include detailed metrics
        #[arg(long)]
        include_details: bool,
    },

    /// Generate performance report
    Report {
        /// Report type (summary, detailed, executive)
        #[arg(short, long, default_value = "summary")]
        report_type: String,

        /// Time period for report (daily, weekly, monthly)
        #[arg(short, long, default_value = "weekly")]
        period: String,

        /// Output format (html, pdf, markdown)
        #[arg(short, long, default_value = "html")]
        format: String,

        /// Output file path
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Configure benchmarking settings
    Configure {
        /// Enable automatic benchmarking
        #[arg(long)]
        auto_benchmark: Option<bool>,

        /// Benchmark frequency (hourly, daily, weekly)
        #[arg(long)]
        frequency: Option<String>,

        /// Performance alert thresholds
        #[arg(long)]
        alert_threshold: Option<f32>,

        /// Result retention period in days
        #[arg(long)]
        retention_days: Option<u32>,
    },

    /// Clean up old benchmark results
    Cleanup {
        /// Days to retain results
        #[arg(short, long, default_value = "30")]
        days: u32,

        /// Dry run (show what would be deleted)
        #[arg(short, long)]
        dry_run: bool,

        /// Force cleanup without confirmation
        #[arg(short, long)]
        force: bool,
    },
}
