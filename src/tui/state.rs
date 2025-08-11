// use chrono::{DateTime, Utc}; // TODO: Add when needed
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, SystemTime};
use tokio::sync::mpsc;
use std::sync::{Arc, Mutex};

use crate::cluster::ClusterStats;
use crate::tui::cluster_state::ClusterState;
use crate::compute::{Device, MemoryInfo};
use crate::models::{AgentInstance, ModelRegistryEntry, MultiAgentSystemStatus};
use crate::streaming::StreamId;
use crate::tui::ui::{
    AccountManager,
    SettingItem,
    SettingValue,
    SettingsUI,
    SocialSettings,
    SubTab,
    SubTabManager,
    Tweet,
};
use crate::tui::chat::SettingsManager;
use crate::tui::connectors::system_connector::SystemHealth;
use crate::tui::chat::ModularChat;


/// Maximum number of log entries to keep
const MAX_LOG_ENTRIES: usize = 1000;
const MAX_HISTORY_ENTRIES: usize = 100;

/// Usage statistics tracking
#[derive(Debug, Clone)]
pub struct UsageStats {
    pub total_prompts: usize,
    pub total_tokens: usize,
    pub estimated_cost: f32,
    pub avg_tokens_per_prompt: f32,
    pub peak_usage_date: String,
    pub most_used_model: String,
    pub cost_trend: String,
    pub efficiency_score: f32,
    pub optimization_tip: String,
}

impl Default for UsageStats {
    fn default() -> Self {
        Self {
            total_prompts: 0,
            total_tokens: 0,
            estimated_cost: 0.0,
            avg_tokens_per_prompt: 0.0,
            peak_usage_date: String::new(),
            most_used_model: String::new(),
            cost_trend: String::new(),
            efficiency_score: 0.0,
            optimization_tip: String::new(),
        }
    }
}

/// System information
#[derive(Debug, Clone)]
pub struct SystemInfo {
    pub network_down: f64,
    pub network_up: f64,
    pub storage_used: u64,
    pub storage_total: u64,
    pub uptime: u64,
    pub temperature: Option<f32>,
}

impl Default for SystemInfo {
    fn default() -> Self {
        Self {
            network_down: 0.0,
            network_up: 0.0,
            storage_used: 0,
            storage_total: 0,
            uptime: 0,
            temperature: None,
        }
    }
}

/// Voice interaction data
#[derive(Debug, Clone)]
pub struct VoiceInteraction {
    pub recognized_text: String,
    pub confidence: f64,
    pub wake_word_triggered: bool,
    pub timestamp: SystemTime,
    pub response_generated: bool,
    pub audio_duration: Duration,
}

/// Command history entry with execution details
#[derive(Debug, Clone)]
pub struct CommandHistoryEntry {
    pub command: String,
    pub timestamp: SystemTime,
    pub success: bool,
    pub execution_time_ms: u64,
    pub result_summary: String,
    pub command_type: String,
}

/// Command execution result tracking
#[derive(Debug, Clone)]
pub enum CommandExecutionResult {
    Success { message: String },
    Error { error: String },
    Partial { message: String, issues: Vec<String> },
}

/// Operation result for async operations
#[derive(Debug, Clone)]
pub enum OperationResult {
    DatabaseConnected { backend: String, success: bool, message: String },
    DatabaseTestResult { backend: String, success: bool, message: String },
    DatabaseConfigSaved { backend: String, success: bool, message: String },
    DatabaseMigrationComplete { backend: String, success: bool, message: String },
    DatabaseBackupComplete { backend: String, path: Option<String>, message: String },
    StoryCreated { id: String, title: String, message: String },
    StoryDeleted { id: String, message: String },
    StoryUpdated { id: String, message: String },
    StorageUnlocked,
    Error { operation: String, error: String },
}

/// Application state
pub struct AppState {
    // Current view
    pub current_view: ViewState,
    pub chat_tabs: SubTabManager,
    pub chat: ModularChat,
    pub social_tabs: SubTabManager,
    pub tweet_input: String,
    pub tweet_status: Option<String>,
    pub account_manager: AccountManager,
    pub settings_manager: SettingsManager,
    pub social_settings: SocialSettings,
    pub recent_tweets: Vec<Tweet>,
    pub recent_tweets_scroll_index: usize,
    pub usage_stats: UsageStats,
    pub system_info: SystemInfo,
    pub system_health: SystemHealth,
    pub cost_history: VecDeque<f32>,
    pub prompts_history: VecDeque<f32>,
    pub tokens_history: VecDeque<f32>,

    // Command input
    pub command_input: String,
    pub cursor_position: usize,
    
    // Chat cursor navigation
    pub chat_cursor_mode: bool,
    pub chat_cursor_row: usize,
    pub chat_cursor_col: usize,
    
    pub command_history: VecDeque<String>,
    pub detailed_command_history: VecDeque<CommandHistoryEntry>,
    pub history_index: Option<usize>,
    pub suggestions: Vec<String>,
    
    // Dynamic command suggestions
    pub command_suggestions: Vec<String>,
    pub selected_suggestion: Option<usize>,
    pub show_command_suggestions: bool,

    // System state
    pub devices: Vec<Device>,
    pub memory_info: Vec<(String, MemoryInfo)>,
    pub cluster_stats: ClusterStats,
    pub cluster_state: ClusterState,
    pub active_streams: Vec<(StreamId, String)>, // (id, status)
    pub available_models: Vec<String>,

    // Real-time activity tracking
    pub recent_activities: VecDeque<ActivityEntry>,
    pub active_model_sessions: Vec<ModelActivitySession>,
    pub ongoing_requests: Vec<ModelRequest>,
    pub current_nl_session: Option<NaturalLanguageSession>,

    // Logs
    pub log_entries: VecDeque<LogEntry>,

    // UI state
    pub selected_device: Option<usize>,
    pub selected_stream: Option<usize>,
    pub selected_model: Option<usize>,
    pub selected_log: Option<usize>,
    pub scroll_position: usize,
    pub show_help: bool,

    // Metrics history for charts
    pub cpu_history: VecDeque<f32>,
    pub gpu_history: VecDeque<f32>,
    pub memory_history: VecDeque<f32>,
    pub request_history: VecDeque<u64>,

    // Model orchestration state
    pub setup_templates: Vec<SetupTemplate>,
    pub active_sessions: Vec<ModelSession>,
    pub model_registry: Vec<ModelInfo>,
    pub apiconfigurations: HashMap<String, String>,
    pub user_favorites: Vec<String>,
    pub cost_analytics: CostAnalytics,
    pub selected_template: Option<usize>,
    pub selected_session: Option<usize>,

    pub model_view: ModelViewState,

    // Multi-agent orchestration state
    pub multi_agent_status: Option<MultiAgentSystemStatus>,
    pub available_agents: Vec<AgentInstance>,
    pub latest_models: Vec<ModelRegistryEntry>,
    pub selected_agent: Option<usize>,
    pub agent_view: AgentViewState,
    pub model_updates_available: Vec<String>,
    pub auto_update_enabled: bool,

    // Home dashboard state (consolidates analytics)
    pub home_dashboard_tabs: SubTabManager,
    pub home_dashboard_view: HomeDashboardViewState,


    // Collaborative sessions state
    pub collaborative_view: CollaborativeViewState,
    pub active_session_id: Option<String>,

    // Cost optimization state
    pub cost_optimization_view: CostOptimizationViewState,

    // Agent specialization state
    pub agent_specialization_view: AgentSpecializationViewState,

    // Plugin ecosystem state
    pub plugin_view: PluginViewState,
    pub installed_plugins: Vec<PluginInfo>,

    pub quantum_num_qubits: usize,
    pub quantum_circuit_depth: usize,
    pub quantum_success_rate: f64,
    pub quantum_advantage_history: Vec<f64>,
    pub quantum_algorithm_performance: HashMap<String, f64>,
    pub quantum_problem_counts: HashMap<String, i32>,
    pub quantum_problem_instances: Vec<QuantumProblemInstance>,
    pub quantum_cpu_utilization: f64,
    pub quantum_memory_utilization: f64,
    pub quantum_gpu_utilization: f64,
    pub quantum_network_utilization: f64,
    pub quantum_avg_circuit_depth: f64,
    pub quantum_avg_fidelity: f64,
    pub quantum_gate_count: u64,
    pub quantum_error_rate: f64,
    pub quantum_entanglement_measure: f64,
    pub quantum_coherence_time: f64,
    pub quantum_decoherence_rate: f64,
    pub quantum_bell_fidelity: f64,
    pub quantum_total_optimizations: u64,
    pub quantum_average_runtime: f64,
    pub quantum_average_advantage: f64,
    pub quantum_best_objective: f64,

    // Cognitive system state
    pub cognitive_tabs: SubTabManager,
    pub cognitive_data: crate::tui::connectors::system_connector::CognitiveData,

    // Memory system state
    pub memory_tabs: SubTabManager,
    pub selected_memory_subsystem: Option<usize>,
    pub memory_data: crate::tui::connectors::system_connector::MemoryData,
    
    // Emergent intelligence state
    
    // Database management state
    pub selected_database_backend: Option<String>,
    pub database_config_mode: bool,
    pub database_form_fields: HashMap<String, String>,
    pub database_form_active_field: Option<String>,
    pub database_connection_status: HashMap<String, bool>,
    pub database_operation_message: Option<String>,
    
    // Story management state
    pub selected_story: Option<usize>,
    pub story_creation_mode: bool,
    pub story_creation_step: StoryCreationStep,
    pub story_form_input: String,
    pub story_form_field: StoryFormField,
    pub selected_story_template: Option<String>,
    pub story_template_index: usize,
    pub story_type_index: usize,
    pub story_configuration: StoryConfiguration,
    pub story_operation_message: Option<String>,
    pub story_message_timeout: Option<std::time::Instant>,
    
    // Async operation result channel
    pub operation_result_receiver: Arc<Mutex<Option<mpsc::UnboundedReceiver<OperationResult>>>>,
    pub operation_result_sender: mpsc::UnboundedSender<OperationResult>,

    // Utilities and system management state
    pub utilities_view: UtilitiesViewState,
    pub utilities_tabs: SubTabManager,
    pub utilities_manager: crate::tui::utilities::ModularUtilities,
    pub selected_tool_tab: Option<usize>,

    // Story visualization
    pub stories_tab: crate::tui::tabs::stories::StoriesTab,

    // Natural language interface state
    pub natural_language_view: NaturalLanguageViewState,
    pub nl_interface_enabled: bool,
    pub voice_recognition_enabled: bool,
    pub speech_synthesis_enabled: bool,
    pub active_conversations: usize,
    pub avg_response_time: f64,
    pub supported_languages: Vec<String>,
    pub wake_words: Vec<String>,
    pub voice_model: String,
    pub voice_latency: f64,
    pub synthesis_voice: String,
    pub emotional_synthesis: bool,
    pub dynamic_model_switching: bool,
    pub avg_sentiment: f64,
    pub voice_accuracy: f64,
    pub user_satisfaction: f64,
    pub user_retention_rate: f64,
    pub synthesis_speed: f64,
    pub synthesis_quality: f64,
    pub synthesis_latency: f64,
    pub sentiment_analysis_enabled: bool,
    pub primary_language_model: String,
    pub noise_cancellation: bool,
    pub multi_turn_support: bool,
    pub model_quality_threshold: f64,
    pub model_load_balancing_enabled: bool,
    pub intent_categories: Vec<String>,
    pub entity_extraction_enabled: bool,
    pub echo_cancellation: bool,
    pub context_aware_synthesis: bool,
    pub audio_sample_rate: u32,
    pub audio_input_quality: f64,
    pub audio_channels: u16,
    pub wake_word_triggers: Vec<String>,
    pub wake_word_false_positives: u64,
    pub wake_word_accuracy: f64,
    pub voice_interaction_history: Vec<VoiceInteraction>,
    pub voice_confidence_threshold: f64,
    pub total_voice_interactions: u64,
    pub total_unique_users: u64,
    pub total_nl_interactions: u64,
    pub total_model_requests_today: u64,
    pub total_conversations: u64,
    pub top_language: String,
    pub top_language_percentage: f64,
    pub synthesis_volume: f64,
    pub synthesis_pitch: f64,
    pub synthesis_cache_size: u64,
    pub successful_recognitions: u64,
    pub sentiment_distribution: HashMap<String, f64>,
    pub sentiment_accuracy: f64,
    pub semantic_similarity_threshold: f64,
    pub retention_7_day: f64,
    pub retention_30_day: f64,
    pub retention_1_day: f64,
    pub response_generation_strategy: String,
    pub recognition_cache_size: u64,

    // Additional natural language interface fields
    pub intent_accuracy: f64,
    pub recent_interactions: Vec<String>,
    pub avg_conversation_length: f64,
    pub conversation_success_rate: f64,
    pub active_conversation_details: Vec<String>,
    pub conversation_history: Vec<String>,
    pub noise_level: f64,
    pub audio_buffer_size: u64,
    pub failed_recognitions: u64,
    pub active_language_models: Vec<String>,
    pub model_load_balancing_efficiency: f64,
    pub language_model_details: Vec<String>,
    pub model_avg_response_time: f64,
    pub model_throughput: f64,
    pub model_success_rate: f64,
    pub best_performing_model: String,
    pub fastest_model: String,
    pub most_reliable_model: String,
    pub peak_requests_per_hour: u64,
    pub model_switches_today: u64,
    pub model_cpu_usage: f64,
    pub model_memory_usage: f64,
    pub model_gpu_usage: f64,
    pub most_common_intent: String,
    pub most_common_intent_percentage: f64,
    pub intent_distribution: HashMap<String, f64>,
    pub language_usage_stats: Vec<(String, u64)>,
    pub active_users_today: u64,
    pub new_users_today: u64,
    pub avg_session_length: f64,
    pub avg_interactions_per_user: f64,
    pub audio_format: String,
    pub intent_confidence_threshold: f64,
    pub context_window_size: u64,
    pub context_retention_minutes: u64,
    pub max_concurrent_conversations: u64,
    pub fallback_models: Vec<String>,
    pub max_response_time: f64,
    pub min_recognition_accuracy: f64,
    pub max_memory_usage: f64,
    pub max_cpu_usage: f64,
    pub analytics_tabs: SubTabManager,
    pub settings_ui: SettingsUI,
    pub settings_tabs: SubTabManager,

    // Tool management state
    pub tool_management_mode: bool,
    pub available_tools: Vec<ToolInfo>,
    pub active_tool_sessions: Vec<ToolSessionInfo>,
    
    // Panel visibility states
    pub show_tools_panel: Option<bool>,
    pub show_cognitive_panel: Option<bool>,
    
    // Storage operation states
    pub storage_password_mode: bool,
    pub storage_password_input: String,
    pub storage_operation_message: Option<String>,
    pub storage_message_timeout: Option<std::time::Instant>,
    
    // Memory operation states
    pub memory_operation_message: Option<String>,
    pub memory_search_mode: bool,
    pub memory_search_query: String,
    pub memory_layer_move_mode: bool,
    pub memory_selected_item: Option<String>,
    pub memory_target_layer: Option<String>,
    
    // API key input states
    pub api_key_input_mode: bool,
    pub api_key_provider_input: String,
    pub api_key_value_input: String,
    
    // Chat search states
    pub chat_search_mode: bool,
    pub chat_search_query: String,
    pub chat_search_results: Vec<String>,
}

/// Tool information for UI display
#[derive(Debug, Clone)]
pub struct ToolInfo {
    pub tool_id: String,
    pub display_name: String,
    pub status: String,
    pub capabilities: Vec<String>,
}

/// Active tool session information
#[derive(Debug, Clone)]
pub struct ToolSessionInfo {
    pub session_id: String,
    pub tool_id: String,
    pub status: String,
    pub started_at: SystemTime,
    pub duration: Duration,
}

#[derive(Debug, Clone, PartialEq)]
pub enum StoryCreationStep {
    SelectType,
    SelectTemplate,
    ConfigureBasics,
    SetupPlotPoints,
    AssignCharacters,
    Review,
}

#[derive(Debug, Clone, PartialEq)]
pub enum StoryFormField {
    Title,
    Description,
    Objectives,
    Metrics,
}

#[derive(Debug, Clone)]
pub struct StoryConfiguration {
    pub story_type: Option<String>,
    pub template: Option<String>,
    pub title: String,
    pub description: String,
    pub objectives: Vec<String>,
    pub metrics: Vec<String>,
    pub characters: Vec<String>,
    pub plot_points: Vec<PlotPointConfig>,
    pub estimated_duration: Option<chrono::Duration>,
    pub context: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct PlotPointConfig {
    pub plot_type: String,
    pub description: String,
    pub estimated_duration: Option<chrono::Duration>,
}

impl Default for StoryConfiguration {
    fn default() -> Self {
        Self {
            story_type: None,
            template: None,
            title: String::new(),
            description: String::new(),
            objectives: Vec::new(),
            metrics: Vec::new(),
            characters: Vec::new(),
            plot_points: Vec::new(),
            estimated_duration: None,
            context: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ViewState {
    /// Home
    Dashboard,
    /// Chat
    Chat,
    /// Utilities 
    Utilities,
    /// Memory
    Memory,
    /// Cognitive
    Cognitive,
    /// Social
    Streams,
    /// Settings
    Models,
    /// Collaborative features (future)
    Collaborative,
    /// Plugin ecosystem (future)
    PluginEcosystem,
}

#[derive(Debug, Clone, PartialEq)]
pub enum NaturalLanguageViewState {
    Overview,
    Conversations,
    Voice,
    Models,
    Analytics,
    Configuration,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ModelViewState {
    Templates,
    Sessions,
    Registry,
    Analytics,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AgentViewState {
    Overview,
    ActiveAgents,
    LatestModels,
    ModelUpdates,
    Performance,
    ApiKeys,
}


#[derive(Debug, Clone, PartialEq)]
pub enum HomeDashboardViewState {
    Overview,      // System overview with key metrics
    Analytics,     // Consolidated analytics from removed Analytics tab  
    Performance,   // Real-time performance metrics
    Resources,     // Resource utilization and health
    Activity,      // Recent activity and logs
}

#[derive(Debug, Clone, PartialEq)]
pub enum CollaborativeViewState {
    Sessions,
    ActiveCollaboration,
    Participants,
    SharedAgents,
    Chat,
    Decisions,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CostOptimizationViewState {
    Dashboard,
    BudgetManagement,
    Forecasting,
    Optimization,
    Alerts,
    Analytics,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AgentSpecializationViewState {
    Overview,
    Specializations,
    Routing,
    Performance,
    LoadBalancing,
    Opportunities,
}

/// Model setup template
#[derive(Debug, Clone)]
pub struct SetupTemplate {
    pub id: String,
    pub name: String,
    pub description: String,
    pub cost_estimate: f32, // USD per hour
    pub is_free: bool,
    pub local_models: Vec<String>,
    pub api_models: Vec<String>,
    pub gpu_memory_required: u64, // MB
    pub setup_time_estimate: u32, // seconds
    pub complexity_level: ComplexityLevel,
    pub cost_per_hour: f32,       // For CLI compatibility
    pub models: Vec<String>,      // Combined models for CLI compatibility
    pub performance_tier: String, // For CLI compatibility
    pub use_cases: Vec<String>,   // For CLI compatibility
    pub is_local_only: bool,      // Whether template uses only local models
    pub require_streaming: bool,  // Whether template requires streaming capabilities
}

#[derive(Debug, Clone, PartialEq)]
pub enum ComplexityLevel {
    Simple,       // One-click setup
    Beginner,     // Guided setup for beginners
    Medium,       // Some configuration
    Intermediate, // Intermediate user setup
    Advanced,     // Full customization
}

/// Active model session
#[derive(Debug, Clone)]
pub struct ModelSession {
    pub id: String,
    pub name: String,
    pub template_id: String,
    pub active_models: Vec<String>,
    pub status: SessionStatus,
    pub cost_per_hour: f32,
    pub gpu_usage: f32,
    pub memory_usage: f32,
    pub request_count: u64,
    pub error_count: u64,
    pub start_time: chrono::DateTime<chrono::Local>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SessionStatus {
    Starting,
    Running,
    Active,
    Paused,
    Stopping,
    Error(String),
}

/// Model information in registry
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub model_type: ModelType,
    pub size_gb: f32,
    pub status: ModelStatus,
    pub download_progress: f32, // 0.0 to 1.0
    pub capabilities: Vec<String>,
    pub context_length: u32,
    pub cost_per_token: Option<f32>, // For API models
}

#[derive(Debug, Clone, PartialEq)]
pub enum ModelType {
    Local,
    Api,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ModelStatus {
    Available,
    Downloading,
    Downloaded,
    Installing,
    Installed,
    Error(String),
}

/// Cost analytics data
#[derive(Debug, Clone, Default)]
pub struct CostAnalytics {
    pub total_cost_today: f32,
    pub total_cost_month: f32,
    pub cost_by_model: HashMap<String, f32>,
    pub requests_today: u64,
    pub requests_month: u64,
    pub avg_cost_per_request: f32,
}

#[derive(Debug, Clone)]
pub struct LogEntry {
    pub timestamp: String,
    pub level: String,
    pub message: String,
    pub target: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LogLevel {
    Info,
    Warning,
    Error,
    Debug,
}

/// Real-time activity tracking types
#[derive(Debug, Clone)]
pub struct ActivityEntry {
    pub timestamp: String,
    pub activity_type: ActivityType,
    pub description: String,
    pub model_id: Option<String>,
    pub status: ActivityStatus,
    pub duration_ms: Option<u64>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ActivityType {
    NaturalLanguageRequest,
    ModelOrchestration,
    TaskExecution,
    ModelSwitch,
    StreamCreation,
    ClusterRebalance,
    FileSystemOperation,
    SystemMonitoring,
    ModelDiscovery,
    Session,
    Plugin,
    HealthCheck,
    ClusterStatus,
    MemoryOptimization,
    SecurityScan,
    ClusterInitialization,
    DirectoryNavigation,
    DirectoryListing,
    FileDeletion,
    DirectoryQuery,
    // Adding missing variants referenced in the codebase
    AgentCoordination,
    Task,
    Error,
    Performance,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ActivityStatus {
    Started,
    InProgress,
    Completed,
    Failed(String),
}

#[derive(Debug, Clone)]
pub struct ModelActivitySession {
    pub session_id: String,
    pub model_id: String,
    pub started_at: String,
    pub last_activity: String,
    pub request_count: u32,
    pub status: ModelSessionStatus,
    pub task_type: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ModelSessionStatus {
    Active,
    Processing,
    Idle,
    Error(String),
}

#[derive(Debug, Clone)]
pub struct ModelRequest {
    pub request_id: String,
    pub model_id: String,
    pub task: String,
    pub started_at: String,
    pub status: RequestStatus,
    pub progress: f32, // 0.0 to 1.0
}

#[derive(Debug, Clone, PartialEq)]
pub enum RequestStatus {
    Queued,
    Processing,
    Streaming,
    Completed,
    Failed(String),
}

#[derive(Debug, Clone)]
pub struct NaturalLanguageSession {
    pub session_id: String,
    pub user_input: String,
    pub parsed_intent: String,
    pub model_selected: Option<String>,
    pub started_at: String,
    pub status: NLSessionStatus,
    pub response_chunks: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum NLSessionStatus {
    Parsing,
    ModelSelection,
    Processing,
    Streaming,
    Completed,
    Failed(String),
}

impl AppState {
    pub async fn new() -> Self {
        Self::new_with_components(None, None, None, None).await
    }
    
    pub async fn new_with_components(
        model_orchestrator: Option<std::sync::Arc<crate::models::ModelOrchestrator>>,
        cognitive_system: Option<std::sync::Arc<crate::cognitive::CognitiveSystem>>,
        tool_manager: Option<std::sync::Arc<crate::tools::IntelligentToolManager>>,
        task_manager: Option<std::sync::Arc<crate::tools::task_management::TaskManager>>,
    ) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        
        Self {
            current_view: ViewState::Dashboard,
            chat: ModularChat::new_with_orchestrator(model_orchestrator, cognitive_system, tool_manager, task_manager).await,
            tweet_input: "".to_string(),
            tweet_status: None,
            account_manager: AccountManager::default(),
            settings_manager: SettingsManager::default(),
            social_settings: Default::default(),
            recent_tweets: vec![],
            recent_tweets_scroll_index: 0,
            usage_stats: Default::default(),
            system_info: Default::default(),
            system_health: Default::default(),
            cost_history: Default::default(),
            prompts_history: Default::default(),
            tokens_history: Default::default(),
            command_input: String::new(),
            cursor_position: 0,
            
            // Initialize chat cursor navigation
            chat_cursor_mode: false,
            chat_cursor_row: 0,
            chat_cursor_col: 0,
            
            command_history: VecDeque::with_capacity(MAX_HISTORY_ENTRIES),
            detailed_command_history: VecDeque::with_capacity(MAX_HISTORY_ENTRIES),
            history_index: None,
            suggestions: Vec::new(),
            
            // Initialize command suggestions
            command_suggestions: Vec::new(),
            selected_suggestion: None,
            show_command_suggestions: false,
            devices: Vec::new(),
            memory_info: Vec::new(),
            cluster_stats: ClusterStats::default(),
            cluster_state: ClusterState::default(),
            active_streams: Vec::new(),
            available_models: Vec::new(),
            recent_activities: VecDeque::new(),
            active_model_sessions: Vec::new(),
            ongoing_requests: Vec::new(),
            current_nl_session: None,
            log_entries: VecDeque::with_capacity(MAX_LOG_ENTRIES),
            selected_device: None,
            selected_stream: None,
            selected_model: None,
            selected_log: None,
            scroll_position: 0,
            show_help: false,
            cpu_history: VecDeque::with_capacity(60),
            gpu_history: VecDeque::with_capacity(60),
            memory_history: VecDeque::with_capacity(60),
            request_history: VecDeque::with_capacity(60),

            // Initialize model orchestration state
            setup_templates: Self::init_default_templates(),
            active_sessions: Vec::new(),
            model_registry: Self::init_default_models(),
            apiconfigurations: HashMap::new(),
            user_favorites: Vec::new(),
            cost_analytics: CostAnalytics::default(),
            selected_template: None,
            selected_session: None,

            model_view: ModelViewState::Templates,

            // Initialize multi-agent state
            multi_agent_status: None,
            available_agents: Vec::new(),
            latest_models: Vec::new(),
            selected_agent: None,
            agent_view: AgentViewState::Overview,
            model_updates_available: Vec::new(),
            auto_update_enabled: true,

            // Initialize home dashboard state
            home_dashboard_tabs: SubTabManager::new(vec![
                SubTab { name: "Yo".to_string(), key: "yo".to_string() },
                SubTab { name: "Monitoring".to_string(), key: "monitoring".to_string() },
                SubTab { name: "Keybindings".to_string(), key: "keybindings".to_string() },
            ]),
            home_dashboard_view: HomeDashboardViewState::Overview,


            // Initialize collaborative sessions state
            collaborative_view: CollaborativeViewState::Sessions,
            active_session_id: None,

            // Initialize cost optimization state
            cost_optimization_view: CostOptimizationViewState::Dashboard,

            // Initialize agent specialization state
            agent_specialization_view: AgentSpecializationViewState::Overview,

            // Initialize plugin ecosystem state
            plugin_view: PluginViewState::new(),
            installed_plugins: Vec::new(),

            quantum_num_qubits: 16,
            quantum_circuit_depth: 8,
            quantum_success_rate: 0.89,
            quantum_advantage_history: vec![1.2, 1.5, 1.8, 2.1, 1.9, 2.3, 2.0],
            quantum_algorithm_performance: [
                ("annealing".to_string(), 0.75),
                ("vqe".to_string(), 0.82),
                ("qaoa".to_string(), 0.78),
                ("qinn".to_string(), 0.85),
            ]
            .into_iter()
            .collect(),
            quantum_problem_counts: [
                ("QUBO".to_string(), 20),
                ("TSP".to_string(), 15),
                ("MaxCut".to_string(), 18),
                ("ML Hyper".to_string(), 12),
                ("NAS".to_string(), 8),
            ]
            .into_iter()
            .collect(),
            quantum_problem_instances: vec![
                QuantumProblemInstance {
                    id: "portfolio_optimization".to_string(),
                    problem_type: "QUBO".to_string(),
                    dimension: 4,
                    difficulty: "Medium".to_string(),
                    maximize: true,
                },
                QuantumProblemInstance {
                    id: "city_routing".to_string(),
                    problem_type: "TSP".to_string(),
                    dimension: 8,
                    difficulty: "Hard".to_string(),
                    maximize: false,
                },
                QuantumProblemInstance {
                    id: "network_partitioning".to_string(),
                    problem_type: "MaxCut".to_string(),
                    dimension: 6,
                    difficulty: "Medium".to_string(),
                    maximize: true,
                },
            ],
            quantum_cpu_utilization: 0.23,
            quantum_memory_utilization: 0.45,
            quantum_gpu_utilization: 0.18,
            quantum_network_utilization: 0.12,
            quantum_avg_circuit_depth: 8.3,
            quantum_avg_fidelity: 0.943,
            quantum_gate_count: 24765,
            quantum_error_rate: 0.0034,
            quantum_entanglement_measure: 0.847,
            quantum_coherence_time: 127.5,
            quantum_decoherence_rate: 0.78,
            quantum_bell_fidelity: 0.952,
            quantum_total_optimizations: 73,
            quantum_average_runtime: 3.2,
            quantum_average_advantage: 2.1,
            quantum_best_objective: 0.956,

            // Initialize cognitive system tabs
            cognitive_tabs: SubTabManager::new(vec![
                SubTab { name: "Overview".to_string(), key: "overview".to_string() },
                SubTab { name: "Operator".to_string(), key: "operator".to_string() },
                SubTab { name: "Agents".to_string(), key: "agents".to_string() },
                SubTab { name: "Autonomy".to_string(), key: "autonomy".to_string() },
                SubTab { name: "Learning".to_string(), key: "learning".to_string() },
                SubTab { name: "Controls".to_string(), key: "controls".to_string() },
            ]),

            // Initialize memory system tabs
            memory_tabs: SubTabManager::new(vec![
                SubTab { name: "Overview".to_string(), key: "overview".to_string() },
                SubTab { name: "Memory".to_string(), key: "memory".to_string() },
                SubTab { name: "Database".to_string(), key: "database".to_string() },
                SubTab { name: "Stories".to_string(), key: "stories".to_string() },
                SubTab { name: "Storage".to_string(), key: "storage".to_string() },
            ]),
            selected_memory_subsystem: Some(0), // Default to Fractal view
            
            // Initialize required data structures
            cognitive_data: crate::tui::connectors::system_connector::CognitiveData::default(),
            memory_data: crate::tui::connectors::system_connector::MemoryData::default(),
            
            // Initialize database management state
            selected_database_backend: Some("postgresql".to_string()),
            database_config_mode: false,
            database_form_fields: HashMap::new(),
            database_form_active_field: None,
            database_connection_status: HashMap::new(),
            database_operation_message: None,
            
            // Initialize story management state
            selected_story: None,
            story_creation_mode: false,
            story_creation_step: StoryCreationStep::SelectType,
            story_form_input: String::new(),
            story_form_field: StoryFormField::Title,
            selected_story_template: None,
            story_template_index: 0,
            story_type_index: 0,
            story_configuration: StoryConfiguration::default(),
            story_operation_message: None,
            story_message_timeout: None,
            
            // Initialize async operation channel
            operation_result_receiver: Arc::new(Mutex::new(Some(rx))),
            operation_result_sender: tx,

            // Initialize utilities and system management state
            utilities_view: UtilitiesViewState::Overview,
            utilities_tabs: SubTabManager::new(vec![
                SubTab { name: "Tools".to_string(), key: "tools".to_string() },
                SubTab { name: "MCP".to_string(), key: "mcp".to_string() },
                SubTab { name: "Plugins".to_string(), key: "plugins".to_string() },
                SubTab { name: "Daemon".to_string(), key: "daemon".to_string() },
            ]),
            utilities_manager: crate::tui::utilities::ModularUtilities::new(),
            selected_tool_tab: Some(0),

            // Initialize stories tab
            stories_tab: crate::tui::tabs::stories::StoriesTab::default(),

            // Initialize natural language interface state
            natural_language_view: NaturalLanguageViewState::Overview,
            nl_interface_enabled: false,
            voice_recognition_enabled: false,
            speech_synthesis_enabled: false,
            active_conversations: 0,
            avg_response_time: 450.0,
            supported_languages: vec![
                "English".to_string(),
                "Spanish".to_string(),
                "French".to_string(),
                "German".to_string(),
                "Chinese".to_string(),
                "Japanese".to_string(),
            ],
            wake_words: vec!["hey loki".to_string(), "loki".to_string()],
            voice_model: "Whisper".to_string(),
            voice_latency: 180.0,
            synthesis_voice: "Neural-A".to_string(),
            emotional_synthesis: true,
            dynamic_model_switching: true,
            avg_sentiment: 0.72,
            voice_accuracy: 0.94,
            user_satisfaction: 0.87,
            user_retention_rate: 0.91,
            synthesis_speed: 1.2,
            synthesis_quality: 0.89,
            synthesis_latency: 245.0,
            sentiment_analysis_enabled: true,
            primary_language_model: "gpt-4-turbo".to_string(),
            noise_cancellation: true,
            multi_turn_support: true,
            model_quality_threshold: 0.85,
            model_load_balancing_enabled: true,
            intent_categories: vec![
                "Question".to_string(),
                "Command".to_string(),
                "Request".to_string(),
                "Greeting".to_string(),
                "Information".to_string(),
            ],
            entity_extraction_enabled: true,
            echo_cancellation: true,
            context_aware_synthesis: true,
            audio_sample_rate: 44100,
            audio_input_quality: 0.92,
            audio_channels: 1,
            wake_word_triggers: vec!["hey loki".to_string(), "assistant".to_string()],
            wake_word_false_positives: 2,
            wake_word_accuracy: 0.94,
            voice_interaction_history: vec![
                VoiceInteraction {
                    recognized_text: "Hey Loki, what's the weather?".to_string(),
                    confidence: 0.94,
                    wake_word_triggered: true,
                    timestamp: SystemTime::now(),
                    response_generated: true,
                    audio_duration: Duration::from_secs(3),
                },
                VoiceInteraction {
                    recognized_text: "Set a reminder for 3 PM".to_string(),
                    confidence: 0.87,
                    wake_word_triggered: false,
                    timestamp: SystemTime::now(),
                    response_generated: true,
                    audio_duration: Duration::from_secs(2),
                },
                VoiceInteraction {
                    recognized_text: "Show me my calendar".to_string(),
                    confidence: 0.91,
                    wake_word_triggered: true,
                    timestamp: SystemTime::now(),
                    response_generated: true,
                    audio_duration: Duration::from_secs(2),
                },
            ],
            voice_confidence_threshold: 0.85,
            total_voice_interactions: 156,
            total_unique_users: 23,
            total_nl_interactions: 1247,
            total_model_requests_today: 892,
            total_conversations: 234,
            top_language: "English".to_string(),
            top_language_percentage: 0.78,
            synthesis_volume: 0.8,
            synthesis_pitch: 0.0,
            synthesis_cache_size: 1024,
            successful_recognitions: 142,
            sentiment_distribution: [
                ("positive".to_string(), 0.62),
                ("neutral".to_string(), 0.28),
                ("negative".to_string(), 0.10),
            ]
            .into_iter()
            .collect(),
            sentiment_accuracy: 0.91,
            semantic_similarity_threshold: 0.75,
            retention_7_day: 0.89,
            retention_30_day: 0.76,
            retention_1_day: 0.94,
            response_generation_strategy: "Hybrid".to_string(),
            recognition_cache_size: 512,

            // Additional natural language interface fields
            intent_accuracy: 0.89,
            recent_interactions: vec![
                "How can I help you today?".to_string(),
                "Setting up your environment...".to_string(),
                "Processing your request...".to_string(),
            ],
            avg_conversation_length: 5.2,
            conversation_success_rate: 0.92,
            active_conversation_details: vec![
                "User asking about weather".to_string(),
                "Technical support inquiry".to_string(),
                "General assistance request".to_string(),
            ],
            conversation_history: vec![
                "Previous conversation about setup".to_string(),
                "Help with configuration".to_string(),
                "Status inquiry".to_string(),
            ],
            noise_level: 0.15,
            audio_buffer_size: 4096,
            failed_recognitions: 14,
            active_language_models: vec![
                "gpt-4-turbo".to_string(),
                "claude-3-opus".to_string(),
                "gemini-pro".to_string(),
            ],
            model_load_balancing_efficiency: 0.94,
            language_model_details: vec![
                "GPT-4 Turbo: 128k context, $0.01/1k tokens".to_string(),
                "Claude 3 Opus: 200k context, $0.015/1k tokens".to_string(),
                "Gemini Pro: 1M context, $0.0005/1k tokens".to_string(),
            ],
            model_avg_response_time: 520.0,
            model_throughput: 850.0,
            model_success_rate: 0.96,
            best_performing_model: "claude-3-opus".to_string(),
            fastest_model: "gemini-pro".to_string(),
            most_reliable_model: "gpt-4-turbo".to_string(),
            peak_requests_per_hour: 1250,
            model_switches_today: 45,
            model_cpu_usage: 0.32,
            model_memory_usage: 0.58,
            model_gpu_usage: 0.24,
            most_common_intent: "Question".to_string(),
            most_common_intent_percentage: 0.42,
            intent_distribution: [
                ("Question".to_string(), 0.42),
                ("Command".to_string(), 0.28),
                ("Request".to_string(), 0.18),
                ("Greeting".to_string(), 0.08),
                ("Information".to_string(), 0.04),
            ]
            .into_iter()
            .collect(),
            language_usage_stats: vec![
                ("English".to_string(), 1847),
                ("Spanish".to_string(), 234),
                ("French".to_string(), 156),
                ("German".to_string(), 89),
                ("Chinese".to_string(), 67),
            ],
            active_users_today: 156,
            new_users_today: 23,
            avg_session_length: 12.5,
            avg_interactions_per_user: 7.8,
            audio_format: "WAV".to_string(),
            intent_confidence_threshold: 0.80,
            context_window_size: 8192,
            context_retention_minutes: 30,
            max_concurrent_conversations: 50,
            fallback_models: vec!["gpt-3.5-turbo".to_string(), "claude-3-haiku".to_string()],
            max_response_time: 5000.0,
            min_recognition_accuracy: 0.85,
            max_memory_usage: 0.75,
            max_cpu_usage: 0.80,
            analytics_tabs: SubTabManager::new(vec![
                SubTab { name: "Usage".to_string(), key: "usage".to_string() },
                SubTab { name: "System".to_string(), key: "system".to_string() },
                SubTab { name: "Stories".to_string(), key: "stories".to_string() },
                SubTab { name: "Settings".to_string(), key: "settings".to_string() },
            ]),
            settings_ui: SettingsUI::new(vec![
                SettingItem {
                    name: "Story Autonomy".to_string(),
                    value: SettingValue::Bool(false),
                    choices: None,
                },
                SettingItem {
                    name: "Cognitive Mode".to_string(),
                    value: SettingValue::Choice("Standard".to_string()),
                    choices: Some(vec!["Standard".to_string(), "Deep".to_string(), "Creative".to_string()]),
                },
                SettingItem {
                    name: "Consciousness Stream".to_string(),
                    value: SettingValue::Bool(false),
                    choices: None,
                },
                SettingItem {
                    name: "Background Processing".to_string(),
                    value: SettingValue::Bool(true),
                    choices: None,
                },
            ]),
            settings_tabs: SubTabManager::new(vec![
                SubTab { name: "General".to_string(), key: "general".to_string() },
                SubTab { name: "Configuration".to_string(), key: "configuration".to_string() },
                SubTab { name: "Safety".to_string(), key: "safety".to_string() },
            ]),
            tool_management_mode: false,
            available_tools: Vec::new(),
            active_tool_sessions: Vec::new(),
            show_tools_panel: Some(false),
            show_cognitive_panel: Some(false),
            
            // Storage operation states
            storage_password_mode: false,
            storage_password_input: String::new(),
            storage_operation_message: None,
            storage_message_timeout: None,
            
            // Memory operation states
            memory_operation_message: None,
            memory_search_mode: false,
            memory_search_query: String::new(),
            memory_layer_move_mode: false,
            memory_selected_item: None,
            memory_target_layer: None,
            
            // API key input states
            api_key_input_mode: false,
            api_key_provider_input: String::new(),
            api_key_value_input: String::new(),
            
            // Chat search states
            chat_search_mode: false,
            chat_search_query: String::new(),
            chat_search_results: Vec::new(),
            chat_tabs: SubTabManager::new(vec![
                SubTab { name: "Chat".to_string(), key: "chat".to_string() },
                SubTab { name: "Editor".to_string(), key: "editor".to_string() },
                SubTab { name: "Models".to_string(), key: "models".to_string() },
                SubTab { name: "History".to_string(), key: "history".to_string() },
                SubTab { name: "Settings".to_string(), key: "settings".to_string() },
                SubTab { name: "Orchestration".to_string(), key: "orchestration".to_string() },
                SubTab { name: "Agents".to_string(), key: "agents".to_string() },
                SubTab { name: "CLI".to_string(), key: "cli".to_string() },
                SubTab { name: "Statistics".to_string(), key: "statistics".to_string() },
            ]),
            social_tabs: SubTabManager::new(vec![
                SubTab { name: "Tweet".to_string(), key: "tweet".to_string() },
                SubTab { name: "Accounts".to_string(), key: "accounts".to_string() },
                SubTab { name: "Recent".to_string(), key: "recent".to_string() },
                SubTab { name: "Settings".to_string(), key: "settings".to_string() },
            ]),
        }
    }

    /// Add a command to history
    pub fn add_to_history(&mut self, command: String) {
        if !command.trim().is_empty() {
            self.command_history.push_front(command);
            if self.command_history.len() > MAX_HISTORY_ENTRIES {
                self.command_history.pop_back();
            }
        }
        self.history_index = None;
    }

    /// Add detailed command execution to history
    pub fn add_command_execution(
        &mut self,
        command: String,
        result: CommandExecutionResult,
        execution_time_ms: u64,
        command_type: String,
    ) {
        if !command.trim().is_empty() {
            let (success, result_summary) = match result {
                CommandExecutionResult::Success { message } => (true, message),
                CommandExecutionResult::Error { error } => (false, format!("Error: {}", error)),
                CommandExecutionResult::Partial { message, issues } => {
                    (false, format!("{} (Issues: {})", message, issues.join(", ")))
                }
            };

            let entry = CommandHistoryEntry {
                command: command.clone(),
                timestamp: SystemTime::now(),
                success,
                execution_time_ms,
                result_summary,
                command_type,
            };

            self.detailed_command_history.push_front(entry);
            if self.detailed_command_history.len() > MAX_HISTORY_ENTRIES {
                self.detailed_command_history.pop_back();
            }

            // Also add to simple command history for backwards compatibility
            self.add_to_history(command);
        }
    }

    /// Get recent command history for display
    pub fn get_recent_command_history(&self, limit: usize) -> Vec<&CommandHistoryEntry> {
        self.detailed_command_history.iter().take(limit).collect()
    }

    /// Navigate command history up
    pub fn history_up(&mut self) {
        if self.command_history.is_empty() {
            return;
        }

        match self.history_index {
            None => {
                self.history_index = Some(0);
                self.command_input = self.command_history[0].clone();
                self.cursor_position = self.command_input.len();
            }
            Some(index) if index + 1 < self.command_history.len() => {
                self.history_index = Some(index + 1);
                self.command_input = self.command_history[index + 1].clone();
                self.cursor_position = self.command_input.len();
            }
            _ => {}
        }
    }

    /// Navigate command history down
    pub fn history_down(&mut self) {
        match self.history_index {
            Some(0) => {
                self.history_index = None;
                self.command_input.clear();
                self.cursor_position = 0;
            }
            Some(index) => {
                self.history_index = Some(index - 1);
                self.command_input = self.command_history[index - 1].clone();
                self.cursor_position = self.command_input.len();
            }
            None => {}
        }
    }

    /// Add a log entry
    pub fn add_log(&mut self, level: String, message: String, target: String) {
        let log_entry = LogEntry {
            timestamp: chrono::Local::now().format("%H:%M:%S%.3f").to_string(),
            level,
            message,
            target,
        };

        self.log_entries.push_back(log_entry);

        if self.log_entries.len() > MAX_LOG_ENTRIES {
            self.log_entries.pop_front();
        }
    }

    /// Add a message as an info log entry (convenience method)
    pub fn add_message(&mut self, message: String) {
        self.add_log("INFO".to_string(), message, "system".to_string());
    }

    /// Update metrics history
    pub fn update_metrics(&mut self, cpu: f32, gpu: f32, memory: f32, requests: u64) {
        // Add new values
        self.cpu_history.push_back(cpu);
        self.gpu_history.push_back(gpu);
        self.memory_history.push_back(memory);
        self.request_history.push_back(requests);

        // Keep only last 60 values (1 minute of data at 1Hz update rate)
        if self.cpu_history.len() > 60 {
            self.cpu_history.pop_front();
        }
        if self.gpu_history.len() > 60 {
            self.gpu_history.pop_front();
        }
        if self.memory_history.len() > 60 {
            self.memory_history.pop_front();
        }
        if self.request_history.len() > 60 {
            self.request_history.pop_front();
        }
    }

    /// Reset scroll position when changing views
    pub fn reset_scroll(&mut self) {
        self.scroll_position = 0;
    }

    /// Scroll up
    pub fn scroll_up(&mut self, amount: usize) {
        self.scroll_position = self.scroll_position.saturating_sub(amount);
    }

    /// Scroll down
    pub fn scroll_down(&mut self, amount: usize, max_items: usize) {
        if max_items > 0 {
            self.scroll_position = (self.scroll_position + amount).min(max_items.saturating_sub(1));
        }
    }

    /// Select next item in current view
    pub fn select_next(&mut self) {
        match self.current_view {
            ViewState::Chat => {
                if let Some(index) = self.selected_device {
                    if index + 1 < self.devices.len() {
                        self.selected_device = Some(index + 1);
                    }
                } else if !self.devices.is_empty() {
                    self.selected_device = Some(0);
                }
            }
            ViewState::Streams => {
                if let Some(index) = self.selected_stream {
                    if index + 1 < self.active_streams.len() {
                        self.selected_stream = Some(index + 1);
                    }
                } else if !self.active_streams.is_empty() {
                    self.selected_stream = Some(0);
                }
            }
            ViewState::Models => match self.model_view {
                ModelViewState::Templates => {
                    if let Some(index) = self.selected_template {
                        if index + 1 < self.setup_templates.len() {
                            self.selected_template = Some(index + 1);
                        }
                    } else if !self.setup_templates.is_empty() {
                        self.selected_template = Some(0);
                    }
                }
                ModelViewState::Sessions => {
                    if let Some(index) = self.selected_session {
                        if index + 1 < self.active_sessions.len() {
                            self.selected_session = Some(index + 1);
                        }
                    } else if !self.active_sessions.is_empty() {
                        self.selected_session = Some(0);
                    }
                }
                ModelViewState::Registry => {
                    if let Some(index) = self.selected_model {
                        if index + 1 < self.model_registry.len() {
                            self.selected_model = Some(index + 1);
                        }
                    } else if !self.model_registry.is_empty() {
                        self.selected_model = Some(0);
                    }
                }
                _ => {}
            },
            // ViewState::Agents => match self.agent_view {
            //     AgentViewState::ActiveAgents => {
            //         if let Some(index) = self.selected_agent {
            //             if index + 1 < self.available_agents.len() {
            //                 self.selected_agent = Some(index + 1);
            //             }
            //         } else if !self.available_agents.is_empty() {
            //             self.selected_agent = Some(0);
            //         }
            //     }
            //     AgentViewState::LatestModels => {
            //         if let Some(index) = self.selected_model {
            //             if index + 1 < self.latest_models.len() {
            //                 self.selected_model = Some(index + 1);
            //             }
            //         } else if !self.latest_models.is_empty() {
            //             self.selected_model = Some(0);
            //         }
            //     }
            //     _ => {}
            // },
            // ViewState::Logs => {
            //     if let Some(index) = self.selected_log {
            //         if index + 1 < self.log_entries.len() {
            //             self.selected_log = Some(index + 1);
            //         }
            //     } else if !self.log_entries.is_empty() {
            //         self.selected_log = Some(0);
            //     }
            // }
            _ => {}
        }
    }

    /// Select previous item in current view
    pub fn select_previous(&mut self) {
        match self.current_view {
            ViewState::Chat => {
                if let Some(index) = self.selected_device {
                    if index > 0 {
                        self.selected_device = Some(index - 1);
                    }
                }
            }
            ViewState::Streams => {
                if let Some(index) = self.selected_stream {
                    if index > 0 {
                        self.selected_stream = Some(index - 1);
                    }
                }
            }
            ViewState::Models => match self.model_view {
                ModelViewState::Templates => {
                    if let Some(index) = self.selected_template {
                        if index > 0 {
                            self.selected_template = Some(index - 1);
                        }
                    }
                }
                ModelViewState::Sessions => {
                    if let Some(index) = self.selected_session {
                        if index > 0 {
                            self.selected_session = Some(index - 1);
                        }
                    }
                }
                ModelViewState::Registry => {
                    if let Some(index) = self.selected_model {
                        if index > 0 {
                            self.selected_model = Some(index - 1);
                        }
                    }
                }
                _ => {}
            },
            // ViewState::Agents => match self.agent_view {
            //     AgentViewState::ActiveAgents => {
            //         if let Some(index) = self.selected_agent {
            //             if index > 0 {
            //                 self.selected_agent = Some(index - 1);
            //             }
            //         }
            //     }
            //     AgentViewState::LatestModels => {
            //         if let Some(index) = self.selected_model {
            //             if index > 0 {
            //                 self.selected_model = Some(index - 1);
            //             }
            //         }
            //     }
            //     _ => {}
            // },
            // ViewState::Logs => {
            //     if let Some(index) = self.selected_log {
            //         if index > 0 {
            //             self.selected_log = Some(index - 1);
            //         }
            //     }
            // }
            _ => {}
        }
    }

    /// Initialize default setup templates
    fn init_default_templates() -> Vec<SetupTemplate> {
        vec![
            SetupTemplate {
                id: "lightning_fast".to_string(),
                name: "Lightning Fast".to_string(),
                description: "Single local model for instant responses".to_string(),
                cost_estimate: 0.0,
                is_free: true,
                local_models: vec!["deepseek-coder-1.3b".to_string()],
                api_models: vec![],
                gpu_memory_required: 2048,
                setup_time_estimate: 30,
                complexity_level: ComplexityLevel::Simple,
                cost_per_hour: 0.0,
                models: vec!["deepseek-coder-1.3b".to_string()],
                performance_tier: "Fast".to_string(),
                use_cases: vec!["Code completion".to_string(), "Quick queries".to_string()],
                is_local_only: false,
                require_streaming: false,
            },
            SetupTemplate {
                id: "balanced_pro".to_string(),
                name: "Balanced Pro".to_string(),
                description: "2 local models + API fallback for balanced performance".to_string(),
                cost_estimate: 0.10,
                is_free: false,
                local_models: vec!["deepseek-coder-7b".to_string(), "phi-3.5-mini".to_string()],
                api_models: vec!["claude-3-haiku".to_string()],
                gpu_memory_required: 8192,
                setup_time_estimate: 120,
                complexity_level: ComplexityLevel::Medium,
                cost_per_hour: 0.10,
                models: vec![
                    "deepseek-coder-7b".to_string(),
                    "phi-3.5-mini".to_string(),
                    "claude-3-haiku".to_string(),
                ],
                performance_tier: "Balanced".to_string(),
                use_cases: vec!["Development".to_string(), "Code review".to_string()],
                is_local_only: false,
                require_streaming: false,
            },
            SetupTemplate {
                id: "premium_quality".to_string(),
                name: "Premium Quality".to_string(),
                description: "Best available models for highest quality output".to_string(),
                cost_estimate: 0.50,
                is_free: false,
                local_models: vec!["wizardcoder-34b".to_string()],
                api_models: vec!["claude-4-sonnet".to_string(), "gpt-4-turbo".to_string()],
                gpu_memory_required: 24576,
                setup_time_estimate: 300,
                complexity_level: ComplexityLevel::Advanced,
                cost_per_hour: 0.50,
                models: vec![
                    "wizardcoder-34b".to_string(),
                    "claude-4-sonnet".to_string(),
                    "gpt-4-turbo".to_string(),
                ],
                performance_tier: "Premium".to_string(),
                use_cases: vec!["Research".to_string(), "Complex reasoning".to_string()],
                is_local_only: false,
                require_streaming: false,
            },
            SetupTemplate {
                id: "research_beast".to_string(),
                name: "Research Beast".to_string(),
                description: "5-model ensemble for complex analysis and research".to_string(),
                cost_estimate: 1.00,
                is_free: false,
                local_models: vec![
                    "deepseek-coder-33b".to_string(),
                    "wizardcoder-34b".to_string(),
                    "magicoder-7b".to_string(),
                ],
                api_models: vec!["claude-4-opus".to_string(), "gpt-4-turbo".to_string()],
                gpu_memory_required: 40960,
                setup_time_estimate: 600,
                complexity_level: ComplexityLevel::Advanced,
                cost_per_hour: 1.00,
                models: vec![
                    "deepseek-coder-33b".to_string(),
                    "wizardcoder-34b".to_string(),
                    "magicoder-7b".to_string(),
                    "claude-4-opus".to_string(),
                    "gpt-4-turbo".to_string(),
                ],
                performance_tier: "Beast".to_string(),
                use_cases: vec![
                    "Research".to_string(),
                    "Analysis".to_string(),
                    "Comparison".to_string(),
                ],
                is_local_only: false,
                require_streaming: false,
            },
            SetupTemplate {
                id: "code_master".to_string(),
                name: "Code Completion Master".to_string(),
                description: "Optimized for lightning-fast code completion".to_string(),
                cost_estimate: 0.0,
                is_free: true,
                local_models: vec!["deepseek-coder-1.3b".to_string(), "phi-3.5-mini".to_string()],
                api_models: vec![],
                gpu_memory_required: 4096,
                setup_time_estimate: 90,
                complexity_level: ComplexityLevel::Simple,
                cost_per_hour: 0.0,
                models: vec!["deepseek-coder-1.3b".to_string(), "phi-3.5-mini".to_string()],
                performance_tier: "Specialized".to_string(),
                use_cases: vec!["Code completion".to_string(), "Refactoring".to_string()],
                is_local_only: false,
                require_streaming: false,
            },
            SetupTemplate {
                id: "writing_pro".to_string(),
                name: "Writing Assistant Pro".to_string(),
                description: "Professional writing and content creation".to_string(),
                cost_estimate: 0.30,
                is_free: false,
                local_models: vec!["mistral-7b-instruct".to_string()],
                api_models: vec!["claude-3-sonnet".to_string()],
                gpu_memory_required: 8192,
                setup_time_estimate: 180,
                complexity_level: ComplexityLevel::Medium,
                cost_per_hour: 0.30,
                models: vec!["mistral-7b-instruct".to_string(), "claude-3-sonnet".to_string()],
                performance_tier: "Professional".to_string(),
                use_cases: vec!["Documentation".to_string(), "Technical writing".to_string()],
                is_local_only: false,
                require_streaming: false,
            },
        ]
    }

    /// Initialize default model registry
    fn init_default_models() -> Vec<ModelInfo> {
        vec![
            ModelInfo {
                name: "deepseek-coder-1.3b".to_string(),
                model_type: ModelType::Local,
                size_gb: 2.6,
                status: ModelStatus::Available,
                download_progress: 0.0,
                capabilities: vec!["code-completion".to_string(), "code-generation".to_string()],
                context_length: 16384,
                cost_per_token: None,
            },
            ModelInfo {
                name: "deepseek-coder-7b".to_string(),
                model_type: ModelType::Local,
                size_gb: 13.8,
                status: ModelStatus::Available,
                download_progress: 0.0,
                capabilities: vec![
                    "code-completion".to_string(),
                    "code-generation".to_string(),
                    "reasoning".to_string(),
                ],
                context_length: 32768,
                cost_per_token: None,
            },
            ModelInfo {
                name: "deepseek-coder-33b".to_string(),
                model_type: ModelType::Local,
                size_gb: 66.0,
                status: ModelStatus::Available,
                download_progress: 0.0,
                capabilities: vec![
                    "code-completion".to_string(),
                    "code-generation".to_string(),
                    "reasoning".to_string(),
                    "complex-analysis".to_string(),
                ],
                context_length: 65536,
                cost_per_token: None,
            },
            ModelInfo {
                name: "wizardcoder-34b".to_string(),
                model_type: ModelType::Local,
                size_gb: 68.0,
                status: ModelStatus::Available,
                download_progress: 0.0,
                capabilities: vec![
                    "code-generation".to_string(),
                    "problem-solving".to_string(),
                    "reasoning".to_string(),
                ],
                context_length: 32768,
                cost_per_token: None,
            },
            ModelInfo {
                name: "magicoder-7b".to_string(),
                model_type: ModelType::Local,
                size_gb: 14.0,
                status: ModelStatus::Available,
                download_progress: 0.0,
                capabilities: vec![
                    "code-generation".to_string(),
                    "instruction-following".to_string(),
                ],
                context_length: 16384,
                cost_per_token: None,
            },
            ModelInfo {
                name: "phi-3.5-mini".to_string(),
                model_type: ModelType::Local,
                size_gb: 7.6,
                status: ModelStatus::Downloaded,
                download_progress: 1.0,
                capabilities: vec!["reasoning".to_string(), "general-purpose".to_string()],
                context_length: 128000,
                cost_per_token: None,
            },
            ModelInfo {
                name: "mistral-7b-instruct".to_string(),
                model_type: ModelType::Local,
                size_gb: 14.5,
                status: ModelStatus::Available,
                download_progress: 0.0,
                capabilities: vec![
                    "instruction-following".to_string(),
                    "reasoning".to_string(),
                    "general-purpose".to_string(),
                ],
                context_length: 32768,
                cost_per_token: None,
            },
            ModelInfo {
                name: "claude-3-haiku".to_string(),
                model_type: ModelType::Api,
                size_gb: 0.0,
                status: ModelStatus::Available,
                download_progress: 1.0,
                capabilities: vec![
                    "reasoning".to_string(),
                    "general-purpose".to_string(),
                    "fast-response".to_string(),
                ],
                context_length: 200000,
                cost_per_token: Some(0.00025),
            },
            ModelInfo {
                name: "claude-3-sonnet".to_string(),
                model_type: ModelType::Api,
                size_gb: 0.0,
                status: ModelStatus::Available,
                download_progress: 1.0,
                capabilities: vec![
                    "reasoning".to_string(),
                    "general-purpose".to_string(),
                    "writing".to_string(),
                ],
                context_length: 200000,
                cost_per_token: Some(0.003),
            },
            ModelInfo {
                name: "claude-4-sonnet".to_string(),
                model_type: ModelType::Api,
                size_gb: 0.0,
                status: ModelStatus::Available,
                download_progress: 1.0,
                capabilities: vec![
                    "advanced-reasoning".to_string(),
                    "complex-analysis".to_string(),
                    "creativity".to_string(),
                ],
                context_length: 200000,
                cost_per_token: Some(0.015),
            },
            ModelInfo {
                name: "claude-4-opus".to_string(),
                model_type: ModelType::Api,
                size_gb: 0.0,
                status: ModelStatus::Available,
                download_progress: 1.0,
                capabilities: vec![
                    "advanced-reasoning".to_string(),
                    "complex-analysis".to_string(),
                    "research".to_string(),
                    "creativity".to_string(),
                ],
                context_length: 200000,
                cost_per_token: Some(0.075),
            },
            ModelInfo {
                name: "gpt-4-turbo".to_string(),
                model_type: ModelType::Api,
                size_gb: 0.0,
                status: ModelStatus::Available,
                download_progress: 1.0,
                capabilities: vec![
                    "reasoning".to_string(),
                    "code-generation".to_string(),
                    "general-purpose".to_string(),
                ],
                context_length: 128000,
                cost_per_token: Some(0.01),
            },
        ]
    }

    /// Add a new model session
    pub fn add_session(&mut self, session: ModelSession) {
        self.active_sessions.push(session);
        if self.selected_session.is_none() {
            self.selected_session = Some(0);
        }
    }

    /// Remove a model session
    pub fn remove_session(&mut self, session_id: &str) -> bool {
        if let Some(pos) = self.active_sessions.iter().position(|s| s.id == session_id) {
            self.active_sessions.remove(pos);

            // Adjust selection if needed
            if let Some(selected) = self.selected_session {
                if selected >= self.active_sessions.len() && !self.active_sessions.is_empty() {
                    self.selected_session = Some(self.active_sessions.len() - 1);
                } else if self.active_sessions.is_empty() {
                    self.selected_session = None;
                }
            }
            true
        } else {
            false
        }
    }

    /// Update model session metrics
    pub fn update_session_metrics(
        &mut self,
        session_id: &str,
        gpu_usage: f32,
        memory_usage: f32,
        request_count: u64,
        error_count: u64,
    ) {
        if let Some(session) = self.active_sessions.iter_mut().find(|s| s.id == session_id) {
            session.gpu_usage = gpu_usage;
            session.memory_usage = memory_usage;
            session.request_count = request_count;
            session.error_count = error_count;
        }
    }

    /// Update model download progress
    pub fn update_model_progress(&mut self, model_name: &str, progress: f32, status: ModelStatus) {
        if let Some(model) = self.model_registry.iter_mut().find(|m| m.name == model_name) {
            model.download_progress = progress;
            model.status = status;
        }
    }

    /// Switch model view
    pub fn set_model_view(&mut self, view: ModelViewState) {
        self.model_view = view;
        self.reset_scroll();
    }

    /// Get currently selected template
    pub fn get_selected_template(&self) -> Option<&SetupTemplate> {
        self.selected_template.and_then(|i| self.setup_templates.get(i))
    }

    /// Get currently selected session
    pub fn get_selected_session(&self) -> Option<&ModelSession> {
        self.selected_session.and_then(|i| self.active_sessions.get(i))
    }

    /// Update cost analytics
    pub fn update_cost_analytics(&mut self, model_name: &str, cost: f32, requests: u64) {
        self.cost_analytics.total_cost_today += cost;
        self.cost_analytics.requests_today += requests;

        *self.cost_analytics.cost_by_model.entry(model_name.to_string()).or_insert(0.0) += cost;

        if self.cost_analytics.requests_today > 0 {
            self.cost_analytics.avg_cost_per_request =
                self.cost_analytics.total_cost_today / self.cost_analytics.requests_today as f32;
        }
    }

    // Multi-agent orchestration methods

    /// Update multi-agent system status
    pub fn update_multi_agent_status(&mut self, status: MultiAgentSystemStatus) {
        self.multi_agent_status = Some(status);
    }

    /// Update available agents
    pub fn update_available_agents(&mut self, agents: Vec<AgentInstance>) {
        self.available_agents = agents;
        // Reset selection if out of bounds
        if let Some(selected) = self.selected_agent {
            if selected >= self.available_agents.len() {
                self.selected_agent = None;
            }
        }
    }

    /// Update latest models
    pub fn update_latest_models(&mut self, models: Vec<ModelRegistryEntry>) {
        self.latest_models = models;
        // Reset selection if out of bounds
        if let Some(selected) = self.selected_model {
            if selected >= self.latest_models.len() {
                self.selected_model = None;
            }
        }
    }

    /// Update model updates available
    pub fn update_model_updates(&mut self, updates: Vec<String>) {
        self.model_updates_available = updates;
    }

    /// Switch agent view
    pub fn set_agent_view(&mut self, view: AgentViewState) {
        self.agent_view = view;
        self.reset_scroll();
    }

    /// Get currently selected agent
    pub fn get_selected_agent(&self) -> Option<&AgentInstance> {
        self.selected_agent.and_then(|i| self.available_agents.get(i))
    }

    /// Get currently selected model
    pub fn get_selected_latest_model(&self) -> Option<&ModelRegistryEntry> {
        self.selected_model.and_then(|i| self.latest_models.get(i))
    }

    /// Select next agent
    pub fn select_next_agent(&mut self) {
        match self.agent_view {
            AgentViewState::ActiveAgents => {
                if let Some(index) = self.selected_agent {
                    if index + 1 < self.available_agents.len() {
                        self.selected_agent = Some(index + 1);
                    }
                } else if !self.available_agents.is_empty() {
                    self.selected_agent = Some(0);
                }
            }
            AgentViewState::LatestModels => {
                if let Some(index) = self.selected_model {
                    if index + 1 < self.latest_models.len() {
                        self.selected_model = Some(index + 1);
                    }
                } else if !self.latest_models.is_empty() {
                    self.selected_model = Some(0);
                }
            }
            _ => {}
        }
    }

    /// Select previous agent
    pub fn select_previous_agent(&mut self) {
        match self.agent_view {
            AgentViewState::ActiveAgents => {
                if let Some(index) = self.selected_agent {
                    if index > 0 {
                        self.selected_agent = Some(index - 1);
                    }
                }
            }
            AgentViewState::LatestModels => {
                if let Some(index) = self.selected_model {
                    if index > 0 {
                        self.selected_model = Some(index - 1);
                    }
                }
            }
            _ => {}
        }
    }

    /// Toggle auto-update for models
    pub fn toggle_auto_update(&mut self) {
        self.auto_update_enabled = !self.auto_update_enabled;
    }

    /// Check if setup is needed (no API keys configured)
    pub fn needs_setup(&self) -> bool {
        // Check if API keys are configured by looking for common API key environment
        // variables
        std::env::var("ANTHROPIC_API_KEY").is_err()
            && std::env::var("OPENAI_API_KEY").is_err()
            && std::env::var("DEEPSEEK_API_KEY").is_err()
            && std::env::var("MISTRAL_API_KEY").is_err()
            && std::env::var("GEMINI_API_KEY").is_err()
    }

    /// Get setup message for users who need to configure API keys
    pub fn get_setup_message(&self) -> String {
        if self.needs_setup() {
            "🔧 Setup needed! Run 'loki setup-apis' to configure your API keys, then restart the \
             TUI."
                .to_string()
        } else {
            "✅ Setup complete! APIs are configured.".to_string()
        }
    }

    /// Get agent status summary
    pub fn get_agent_status_summary(&self) -> String {
        if self.available_agents.is_empty() {
            "No agents available".to_string()
        } else {
            let active_count = self
                .available_agents
                .iter()
                .filter(|a| matches!(a.status, crate::models::AgentStatus::Active))
                .count();
            format!("{} active, {} total", active_count, self.available_agents.len())
        }
    }

    /// Get model updates summary
    pub fn get_model_updates_summary(&self) -> String {
        if self.model_updates_available.is_empty() {
            "All models up to date".to_string()
        } else {
            format!("{} updates available", self.model_updates_available.len())
        }
    }

    /// Activity tracking methods
    pub fn add_activity(
        &mut self,
        activity_type: ActivityType,
        description: String,
        model_id: Option<String>,
    ) {
        let activity = ActivityEntry {
            timestamp: chrono::Local::now().format("%H:%M:%S%.3f").to_string(),
            activity_type,
            description,
            model_id,
            status: ActivityStatus::Started,
            duration_ms: None,
        };

        self.recent_activities.push_back(activity);

        // Keep only recent activities (last 50)
        if self.recent_activities.len() > 50 {
            self.recent_activities.pop_front();
        }
    }

    pub fn update_activity_status(
        &mut self,
        description: &str,
        status: ActivityStatus,
        duration_ms: Option<u64>,
    ) {
        if let Some(activity) =
            self.recent_activities.iter_mut().rev().find(|a| a.description == description)
        {
            activity.status = status;
            activity.duration_ms = duration_ms;
        }
    }

    pub fn start_nl_session(&mut self, user_input: String, parsed_intent: String) -> String {
        let session_id = format!("nl_{}", chrono::Local::now().timestamp_millis());

        let session = NaturalLanguageSession {
            session_id: session_id.clone(),
            user_input,
            parsed_intent,
            model_selected: None,
            started_at: chrono::Local::now().format("%H:%M:%S%.3f").to_string(),
            status: NLSessionStatus::Parsing,
            response_chunks: Vec::new(),
        };

        self.current_nl_session = Some(session);
        session_id
    }

    pub fn update_nl_session_status(&mut self, status: NLSessionStatus) {
        if let Some(session) = &mut self.current_nl_session {
            session.status = status;
        }
    }

    pub fn set_nl_session_model(&mut self, model_id: String) {
        if let Some(session) = &mut self.current_nl_session {
            session.model_selected = Some(model_id);
            session.status = NLSessionStatus::Processing;
        }
    }

    pub fn add_nl_response_chunk(&mut self, chunk: String) {
        if let Some(session) = &mut self.current_nl_session {
            session.response_chunks.push(chunk);
        }
    }

    pub fn complete_nl_session(&mut self) {
        if let Some(mut session) = self.current_nl_session.take() {
            session.status = NLSessionStatus::Completed;

            // Add to activity log
            self.add_activity(
                ActivityType::NaturalLanguageRequest,
                format!("Completed: {}", session.user_input),
                session.model_selected.clone(),
            );
        }
    }

    pub fn start_model_request(&mut self, model_id: String, task: String) -> String {
        let request_id = format!("req_{}", chrono::Local::now().timestamp_millis());

        let request = ModelRequest {
            request_id: request_id.clone(),
            model_id: model_id.clone(),
            task: task.clone(),
            started_at: chrono::Local::now().format("%H:%M:%S%.3f").to_string(),
            status: RequestStatus::Queued,
            progress: 0.0,
        };

        self.ongoing_requests.push(request);

        // Add activity entry
        self.add_activity(
            ActivityType::TaskExecution,
            format!("Processing: {}", task),
            Some(model_id.clone()),
        );

        // Create or update model session
        self.ensure_model_session(model_id, task);

        request_id
    }

    pub fn update_request_status(
        &mut self,
        request_id: &str,
        status: RequestStatus,
        progress: f32,
    ) {
        if let Some(request) = self.ongoing_requests.iter_mut().find(|r| r.request_id == request_id)
        {
            request.status = status;
            request.progress = progress;
        }
    }

    pub fn complete_request(&mut self, request_id: &str) {
        if let Some(pos) = self.ongoing_requests.iter().position(|r| r.request_id == request_id) {
            let request = self.ongoing_requests.remove(pos);

            // Update activity
            self.update_activity_status(
                &format!("Processing: {}", request.task),
                ActivityStatus::Completed,
                Some(chrono::Local::now().timestamp_millis() as u64),
            );

            // Update model session
            if let Some(session) =
                self.active_model_sessions.iter_mut().find(|s| s.model_id == request.model_id)
            {
                session.request_count += 1;
                session.last_activity = chrono::Local::now().format("%H:%M:%S%.3f").to_string();
                session.status = ModelSessionStatus::Active;
            }
        }
    }

    fn ensure_model_session(&mut self, model_id: String, task_type: String) {
        if !self.active_model_sessions.iter().any(|s| s.model_id == model_id) {
            let session = ModelActivitySession {
                session_id: format!("ms_{}", chrono::Local::now().timestamp_millis()),
                model_id,
                started_at: chrono::Local::now().format("%H:%M:%S%.3f").to_string(),
                last_activity: chrono::Local::now().format("%H:%M:%S%.3f").to_string(),
                request_count: 0,
                status: ModelSessionStatus::Processing,
                task_type,
            };

            self.active_model_sessions.push(session);
        } else if let Some(session) =
            self.active_model_sessions.iter_mut().find(|s| s.model_id == model_id)
        {
            session.status = ModelSessionStatus::Processing;
            session.last_activity = chrono::Local::now().format("%H:%M:%S%.3f").to_string();
        }
    }
}

/// Plugin ecosystem view state
#[derive(Debug, Clone, Default)]
pub struct PluginViewState {
    pub selected_tab: usize,

    // Installed plugins
    pub installed_plugins: Vec<PluginInfo>,
    pub selected_plugin: Option<usize>,

    // Available plugins (discovered but not installed)
    pub available_plugins: Vec<PluginInfo>,
    pub selected_available: Option<usize>,
    pub scanned_directories: usize,
    pub last_scan_time: String,

    // Marketplace
    pub marketplace_plugins: Vec<MarketplacePluginInfo>,
    pub selected_marketplace: Option<usize>,
    pub marketplace_search: String,
    pub marketplace_category: String,

    // WASM Engine
    pub wasm_loaded_modules: usize,
    pub wasm_active_instances: usize,
    pub wasm_function_calls: u64,
    pub wasm_cache_hit_rate: f64,
    pub wasm_memory_usage_mb: f64,
    pub wasm_memory_limit_mb: f64,
    pub wasm_cpu_usage_percent: f64,
    pub wasm_instances: Vec<WasmInstanceInfo>,

    // Security
    pub security_violations: u64,
    pub blocked_actions: u64,
    pub quarantined_plugins: u64,
    pub security_events: Vec<SecurityEvent>,

    // Performance
    pub total_function_calls: u64,
    pub avg_execution_time_ms: f64,
    pub peak_memory_mb: f64,
    pub error_rate_percent: f64,
    pub plugins_under_load: usize,
    pub performance_data: Vec<PluginPerformanceData>,
    // Additional fields for plugin ecosystem management
    pub total_plugin_calls: u64,
    pub total_memory_usage: f64,
    pub available_updates: Vec<String>,
}

/// Plugin information for TUI display
#[derive(Debug, Clone)]
pub struct PluginInfo {
    pub id: String,
    pub name: String,
    pub version: String,
    pub author: String,
    pub description: String,
    pub plugin_type: String,
    pub state: String,
    pub capabilities: Vec<String>,
    pub load_time_ms: u64,
    pub function_calls: u64,
    pub memory_usage_mb: f64,
    pub cpu_time_ms: f64,
    pub cpu_usage: f64,
    pub error_count: u64,
    pub last_error: Option<String>,
    pub is_builtin: bool,
    pub config_schema: Option<String>,
    pub dependencies: Vec<String>,
    pub required_permissions: Vec<String>,
}

impl Default for PluginInfo {
    fn default() -> Self {
        Self {
            id: "unknown-plugin".to_string(),
            name: "Unknown Plugin".to_string(),
            version: "0.1.0".to_string(),
            author: "Unknown".to_string(),
            description: "No description available".to_string(),
            plugin_type: "Unknown".to_string(),
            state: "Stopped".to_string(),
            capabilities: vec![],
            load_time_ms: 0,
            function_calls: 0,
            memory_usage_mb: 0.0,
            cpu_time_ms: 0.0,
            cpu_usage: 0.0,
            error_count: 0,
            last_error: None,
            is_builtin: false,
            config_schema: None,
            dependencies: vec![],
            required_permissions: vec![],
        }
    }
}

/// Marketplace plugin information
#[derive(Debug, Clone)]
pub struct MarketplacePluginInfo {
    pub id: String,
    pub name: String,
    pub version: String,
    pub description: String,
    pub author: String,
    pub rating: u8,
    pub downloads: u64,
    pub category: String,
    pub size_mb: f64,
    pub is_installed: bool,
    pub verified: bool,
}

impl Default for MarketplacePluginInfo {
    fn default() -> Self {
        Self {
            id: "sample-plugin".to_string(),
            name: "Sample Plugin".to_string(),
            version: "1.0.0".to_string(),
            description: "A sample marketplace plugin".to_string(),
            author: "Developer".to_string(),
            rating: 4,
            downloads: 1000,
            category: "General".to_string(),
            size_mb: 2.5,
            is_installed: false,
            verified: true,
        }
    }
}

/// WASM instance information
#[derive(Debug, Clone)]
pub struct WasmInstanceInfo {
    pub plugin_name: String,
    pub state: String,
    pub memory_mb: f64,
    pub fuel_usage_percent: f64,
    pub function_calls: u64,
    pub error_count: u64,
}

impl Default for WasmInstanceInfo {
    fn default() -> Self {
        Self {
            plugin_name: "wasm-plugin".to_string(),
            state: "Active".to_string(),
            memory_mb: 4.2,
            fuel_usage_percent: 12.5,
            function_calls: 47,
            error_count: 0,
        }
    }
}

/// Security event information
#[derive(Debug, Clone)]
pub struct SecurityEvent {
    pub timestamp: String,
    pub plugin_name: String,
    pub event_type: String,
    pub severity: String,
    pub description: String,
}

impl Default for SecurityEvent {
    fn default() -> Self {
        Self {
            timestamp: "12:34:56".to_string(),
            plugin_name: "sample-plugin".to_string(),
            event_type: "Permission Check".to_string(),
            severity: "Low".to_string(),
            description: "Network access requested and granted".to_string(),
        }
    }
}

/// Plugin performance data
#[derive(Debug, Clone)]
pub struct PluginPerformanceData {
    pub plugin_name: String,
    pub function_calls: u64,
    pub avg_time_ms: f64,
    pub memory_mb: f64,
    pub cpu_percent: f64,
    pub error_count: u64,
    pub status: String,
}

impl Default for PluginPerformanceData {
    fn default() -> Self {
        Self {
            plugin_name: "sample-plugin".to_string(),
            function_calls: 156,
            avg_time_ms: 2.4,
            memory_mb: 8.1,
            cpu_percent: 12.3,
            error_count: 0,
            status: "Optimal".to_string(),
        }
    }
}

impl PluginViewState {
    /// Initialize with sample data for demonstration
    pub fn new() -> Self {
        Self {
            selected_tab: 0,
            installed_plugins: vec![
                PluginInfo {
                    id: "ai-assistant-plugin".to_string(),
                    name: "AI Assistant Plugin".to_string(),
                    version: "2.1.0".to_string(),
                    author: "Loki Team".to_string(),
                    description: "Advanced AI assistant with consciousness integration".to_string(),
                    plugin_type: "WASM".to_string(),
                    state: "Active".to_string(),
                    capabilities: vec!["ConsciousnessAccess".to_string(), "MemoryRead".to_string()],
                    load_time_ms: 120,
                    function_calls: 1847,
                    memory_usage_mb: 12.3,
                    cpu_time_ms: 456.7,
                    cpu_usage: 15.8,
                    error_count: 0,
                    last_error: None,
                    is_builtin: true,
                    config_schema: Some("{\"api_key\": \"string\", \"max_tokens\": \"number\"}".to_string()),
                    dependencies: vec![],
                    required_permissions: vec!["consciousness:read".to_string(), "memory:read".to_string()],
                },
                PluginInfo {
                    id: "code-analyzer".to_string(),
                    name: "Code Analyzer".to_string(),
                    version: "1.5.3".to_string(),
                    author: "DevTools Inc".to_string(),
                    description: "Real-time code analysis and suggestions".to_string(),
                    plugin_type: "Native".to_string(),
                    state: "Active".to_string(),
                    capabilities: vec![
                        "FileSystemRead".to_string(),
                        "CodeModification".to_string(),
                    ],
                    load_time_ms: 89,
                    function_calls: 924,
                    memory_usage_mb: 8.7,
                    cpu_time_ms: 234.1,
                    cpu_usage: 7.3,
                    error_count: 0,
                    last_error: None,
                    is_builtin: false,
                    config_schema: None,
                    dependencies: vec!["tree-sitter".to_string()],
                    required_permissions: vec!["file:read".to_string(), "file:write".to_string()],
                },
                PluginInfo {
                    id: "social-media-bot".to_string(),
                    name: "Social Media Bot".to_string(),
                    version: "0.9.2".to_string(),
                    author: "Social Corp".to_string(),
                    description: "Automated social media interaction and content generation"
                        .to_string(),
                    plugin_type: "Python".to_string(),
                    state: "Stopped".to_string(),
                    capabilities: vec!["SocialMedia".to_string(), "NetworkAccess".to_string()],
                    load_time_ms: 245,
                    function_calls: 67,
                    memory_usage_mb: 15.2,
                    cpu_time_ms: 89.3,
                    cpu_usage: 12.1,
                    error_count: 3,
                    last_error: Some("Rate limit exceeded".to_string()),
                    is_builtin: false,
                    config_schema: Some("{\"twitter_api_key\": \"string\"}".to_string()),
                    dependencies: vec!["twitter-api".to_string()],
                    required_permissions: vec!["network:write".to_string()],
                },
            ],
            selected_plugin: Some(0),
            available_plugins: vec![
                PluginInfo {
                    id: "performance-monitor".to_string(),
                    name: "Performance Monitor".to_string(),
                    version: "1.0.0".to_string(),
                    author: "Monitoring Solutions".to_string(),
                    description: "Advanced system performance monitoring and alerting".to_string(),
                    plugin_type: "JavaScript".to_string(),
                    state: "Available".to_string(),
                    capabilities: vec!["MemoryRead".to_string()],
                    load_time_ms: 0,
                    function_calls: 0,
                    memory_usage_mb: 0.0,
                    cpu_time_ms: 0.0,
                    cpu_usage: 0.0,
                    error_count: 0,
                    last_error: None,
                    is_builtin: false,
                    config_schema: None,
                    dependencies: vec![],
                    required_permissions: vec!["memory:read".to_string()],
                },
                PluginInfo {
                    id: "database-sync".to_string(),
                    name: "Database Sync".to_string(),
                    version: "2.3.1".to_string(),
                    author: "Data Systems".to_string(),
                    description: "Bidirectional database synchronization tool".to_string(),
                    plugin_type: "WASM".to_string(),
                    state: "Available".to_string(),
                    capabilities: vec!["NetworkAccess".to_string(), "FileSystemWrite".to_string()],
                    load_time_ms: 0,
                    function_calls: 0,
                    memory_usage_mb: 0.0,
                    cpu_time_ms: 0.0,
                    cpu_usage: 0.0,
                    error_count: 0,
                    last_error: None,
                    is_builtin: false,
                    config_schema: None,
                    dependencies: vec!["postgres".to_string(), "mysql".to_string()],
                    required_permissions: vec!["network:read".to_string(), "network:write".to_string(), "file:write".to_string()],
                },
            ],
            selected_available: None,
            scanned_directories: 8,
            last_scan_time: "2024-06-27 14:32:15".to_string(),
            marketplace_plugins: vec![
                MarketplacePluginInfo {
                    id: "enhanced-ai-reasoning".to_string(),
                    name: "Enhanced AI Reasoning".to_string(),
                    version: "3.0.0".to_string(),
                    description: "Advanced reasoning capabilities with multi-modal support"
                        .to_string(),
                    author: "AI Research Lab".to_string(),
                    rating: 5,
                    downloads: 15000,
                    category: "AI/ML".to_string(),
                    size_mb: 45.2,
                    is_installed: false,
                    verified: true,
                },
                MarketplacePluginInfo {
                    id: "cloud-storage-sync".to_string(),
                    name: "Cloud Storage Sync".to_string(),
                    version: "1.7.2".to_string(),
                    description: "Seamless cloud storage integration and synchronization"
                        .to_string(),
                    author: "Cloud Solutions".to_string(),
                    rating: 4,
                    downloads: 8500,
                    category: "Storage".to_string(),
                    size_mb: 12.8,
                    is_installed: true,
                    verified: true,
                },
                MarketplacePluginInfo {
                    id: "advanced-security-scanner".to_string(),
                    name: "Advanced Security Scanner".to_string(),
                    version: "2.1.0".to_string(),
                    description: "Comprehensive security scanning and vulnerability detection"
                        .to_string(),
                    author: "Security Corp".to_string(),
                    rating: 5,
                    downloads: 12000,
                    category: "Security".to_string(),
                    size_mb: 28.5,
                    is_installed: false,
                    verified: true,
                },
            ],
            selected_marketplace: None,
            marketplace_search: String::new(),
            marketplace_category: "All".to_string(),
            wasm_loaded_modules: 7,
            wasm_active_instances: 5,
            wasm_function_calls: 25847,
            wasm_cache_hit_rate: 94.2,
            wasm_memory_usage_mb: 45.7,
            wasm_memory_limit_mb: 512.0,
            wasm_cpu_usage_percent: 23.5,
            wasm_instances: vec![
                WasmInstanceInfo {
                    plugin_name: "AI Assistant".to_string(),
                    state: "Active".to_string(),
                    memory_mb: 12.3,
                    fuel_usage_percent: 18.4,
                    function_calls: 1847,
                    error_count: 0,
                },
                WasmInstanceInfo {
                    plugin_name: "Database Sync".to_string(),
                    state: "Active".to_string(),
                    memory_mb: 8.9,
                    fuel_usage_percent: 5.2,
                    function_calls: 432,
                    error_count: 0,
                },
                WasmInstanceInfo {
                    plugin_name: "Web Crawler".to_string(),
                    state: "Loading".to_string(),
                    memory_mb: 4.1,
                    fuel_usage_percent: 0.0,
                    function_calls: 0,
                    error_count: 0,
                },
            ],
            security_violations: 0,
            blocked_actions: 3,
            quarantined_plugins: 0,
            security_events: vec![
                SecurityEvent {
                    timestamp: "14:28:43".to_string(),
                    plugin_name: "Social Media Bot".to_string(),
                    event_type: "Network Access".to_string(),
                    severity: "Medium".to_string(),
                    description: "HTTP request to twitter.com blocked (rate limited)".to_string(),
                },
                SecurityEvent {
                    timestamp: "14:25:17".to_string(),
                    plugin_name: "Code Analyzer".to_string(),
                    event_type: "File Access".to_string(),
                    severity: "Low".to_string(),
                    description: "Read access to source files granted".to_string(),
                },
                SecurityEvent {
                    timestamp: "14:22:05".to_string(),
                    plugin_name: "AI Assistant".to_string(),
                    event_type: "Memory Access".to_string(),
                    severity: "Low".to_string(),
                    description: "Consciousness stream access granted".to_string(),
                },
            ],
            total_function_calls: 25847,
            avg_execution_time_ms: 3.2,
            peak_memory_mb: 67.4,
            error_rate_percent: 0.12,
            plugins_under_load: 2,
            performance_data: vec![
                PluginPerformanceData {
                    plugin_name: "AI Assistant".to_string(),
                    function_calls: 1847,
                    avg_time_ms: 4.2,
                    memory_mb: 12.3,
                    cpu_percent: 15.8,
                    error_count: 0,
                    status: "Optimal".to_string(),
                },
                PluginPerformanceData {
                    plugin_name: "Code Analyzer".to_string(),
                    function_calls: 924,
                    avg_time_ms: 2.1,
                    memory_mb: 8.7,
                    cpu_percent: 7.3,
                    error_count: 0,
                    status: "Optimal".to_string(),
                },
                PluginPerformanceData {
                    plugin_name: "Social Media Bot".to_string(),
                    function_calls: 67,
                    avg_time_ms: 8.9,
                    memory_mb: 15.2,
                    cpu_percent: 12.1,
                    error_count: 3,
                    status: "Warning".to_string(),
                },
            ],
            total_plugin_calls: 2838,
            total_memory_usage: 36.2,
            available_updates: vec![],
        }
    }
}


/// Quantum optimization view state

/// Utilities view state for comprehensive Loki system management
#[derive(Debug, Clone, PartialEq)]
pub enum UtilitiesViewState {
    Overview,
    Tools,
    MCP,
    Cognitive,
    Memory,
    Plugins,
    Configuration,
    CLI,
    Safety,
    Monitoring,
    Sessions,
    Database,
    Daemon,
}

/// Quantum problem instance for TUI display
#[derive(Debug, Clone)]
pub struct QuantumProblemInstance {
    pub id: String,
    pub problem_type: String,
    pub dimension: usize,
    pub difficulty: String,
    pub maximize: bool,
}

impl Default for QuantumProblemInstance {
    fn default() -> Self {
        Self {
            id: "sample_problem".to_string(),
            problem_type: "QUBO".to_string(),
            dimension: 4,
            difficulty: "Medium".to_string(),
            maximize: true,
        }
    }
}
