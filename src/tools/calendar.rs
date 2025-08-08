use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Result, anyhow, Context};
use chrono::{DateTime, Utc};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::{RwLock, broadcast, mpsc};
use tokio::time::interval;
use tracing::{debug, error, info, warn};

use crate::cognitive::goal_manager::Priority;
use crate::cognitive::{CognitiveSystem, Thought, ThoughtMetadata, ThoughtType};
use crate::memory::{CognitiveMemory, MemoryMetadata};

/// Calendar integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalendarConfig {
    /// Google Calendar API configuration
    pub google_calendar_enabled: bool,
    pub google_client_id: String,
    pub google_client_secret: String,
    pub google_refresh_token: String,

    /// Outlook/Microsoft Graph API configuration
    pub outlook_enabled: bool,
    pub outlook_client_id: String,
    pub outlook_client_secret: String,
    pub outlook_tenant_id: String,
    pub outlook_refresh_token: String,

    /// Calendar processing settings
    pub monitored_calendars: Vec<String>,
    pub sync_interval: Duration,
    pub lookahead_days: u32,
    pub lookback_days: u32,

    /// Cognitive integration settings
    pub cognitive_awareness_level: f32,
    pub enable_meeting_preparation: bool,
    pub enable_schedule_optimization: bool,
    pub enable_conflict_detection: bool,
    pub enable_time_blocking: bool,

    /// Meeting settings
    pub default_meeting_buffer: Duration,
    pub travel_time_estimation: bool,
    pub auto_decline_conflicts: bool,
    pub meeting_prep_time: Duration,
}

impl Default for CalendarConfig {
    fn default() -> Self {
        Self {
            google_calendar_enabled: false,
            google_client_id: String::new(),
            google_client_secret: String::new(),
            google_refresh_token: String::new(),
            outlook_enabled: false,
            outlook_client_id: String::new(),
            outlook_client_secret: String::new(),
            outlook_tenant_id: String::new(),
            outlook_refresh_token: String::new(),
            monitored_calendars: vec!["primary".to_string()],
            sync_interval: Duration::from_secs(300), // 5 minutes
            lookahead_days: 30,
            lookback_days: 7,
            cognitive_awareness_level: 0.8,
            enable_meeting_preparation: true,
            enable_schedule_optimization: true,
            enable_conflict_detection: true,
            enable_time_blocking: true,
            default_meeting_buffer: Duration::from_secs(300), // 5 minutes
            travel_time_estimation: false,
            auto_decline_conflicts: false,
            meeting_prep_time: Duration::from_secs(900), // 15 minutes
        }
    }
}

/// Calendar event structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalendarEvent {
    pub id: String,
    pub calendar_id: String,
    pub title: String,
    pub description: Option<String>,
    pub location: Option<String>,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub all_day: bool,
    pub recurring: bool,
    pub attendees: Vec<EventAttendee>,
    pub organizer: Option<EventAttendee>,
    pub status: EventStatus,
    pub visibility: EventVisibility,
    pub event_type: EventType,
    pub meeting_link: Option<String>,
    pub agenda: Option<String>,
    pub priority: EventPriority,
    pub tags: Vec<String>,
    pub cognitive_metadata: EventCognitiveMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventAttendee {
    pub email: String,
    pub name: Option<String>,
    pub response_status: AttendeeResponse,
    pub is_organizer: bool,
    pub is_optional: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AttendeeResponse {
    NeedsAction,
    Declined,
    Tentative,
    Accepted,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventStatus {
    Confirmed,
    Tentative,
    Cancelled,
}

impl std::fmt::Display for EventStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EventStatus::Confirmed => write!(f, "Confirmed"),
            EventStatus::Tentative => write!(f, "Tentative"),
            EventStatus::Cancelled => write!(f, "Cancelled"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventVisibility {
    Default,
    Public,
    Private,
    Confidential,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventType {
    Meeting,
    Appointment,
    Task,
    Reminder,
    Travel,
    Personal,
    WorkTime,
    BreakTime,
}

impl std::fmt::Display for EventType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EventType::Meeting => write!(f, "Meeting"),
            EventType::Appointment => write!(f, "Appointment"),
            EventType::Task => write!(f, "Task"),
            EventType::Reminder => write!(f, "Reminder"),
            EventType::Travel => write!(f, "Travel"),
            EventType::Personal => write!(f, "Personal"),
            EventType::WorkTime => write!(f, "Work Time"),
            EventType::BreakTime => write!(f, "Break Time"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventPriority {
    Low,
    Normal,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventCognitiveMetadata {
    pub preparation_required: bool,
    pub estimated_preparation_time: Duration,
    pub context_importance: f32,
    pub energy_requirement: f32,
    pub focus_level_needed: f32,
    pub suggested_prep_actions: Vec<String>,
    pub related_memories: Vec<String>,
    pub optimal_time_suggestion: Option<DateTime<Utc>>,
}

/// Time block for focused work
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeBlock {
    pub id: String,
    pub title: String,
    pub purpose: TimeBlockPurpose,
    pub start_time: DateTime<Utc>,
    pub duration: Duration,
    pub energy_level: f32,
    pub focus_requirements: FocusRequirements,
    pub flexibility: f32, // 0.0 (fixed) to 1.0 (highly flexible)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TimeBlockPurpose {
    DeepWork,
    Planning,
    Communication,
    Learning,
    CreativeWork,
    Administrative,
    Break,
    Buffer,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FocusRequirements {
    pub concentration_level: f32,
    pub interruption_tolerance: f32,
    pub collaboration_need: f32,
}

/// Schedule analysis and optimization
#[derive(Debug, Clone)]
pub struct ScheduleAnalysis {
    pub total_scheduled_time: Duration,
    pub free_time_blocks: Vec<TimeBlock>,
    pub meeting_density: f32,
    pub travel_time: Duration,
    pub energy_distribution: EnergyDistribution,
    pub optimization_suggestions: Vec<ScheduleSuggestion>,
    pub conflict_warnings: Vec<ScheduleConflict>,
}

#[derive(Debug, Clone)]
pub struct EnergyDistribution {
    pub morning_load: f32,
    pub afternoon_load: f32,
    pub evening_load: f32,
    pub peak_hours: Vec<u8>, // Hours of the day (0-23)
    pub low_energy_periods: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct ScheduleSuggestion {
    pub suggestion_type: SuggestionType,
    pub description: String,
    pub potential_benefit: String,
    pub implementation_difficulty: f32,
}

#[derive(Debug, Clone)]
pub enum SuggestionType {
    RescheduleEvent,
    AddBuffer,
    BlockFocusTime,
    OptimizeTravel,
    EnergyAlignment,
    ReduceFragmentation,
}

#[derive(Debug, Clone)]
pub struct ScheduleConflict {
    pub conflict_type: ConflictType,
    pub events: Vec<String>, // Event IDs
    pub severity: f32,
    pub resolution_suggestions: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum ConflictType {
    TimeOverlap,
    TravelImpossible,
    EnergyMismatch,
    PrepTimeInsufficient,
    BackToBackMeetings,
}

/// Statistics for calendar integration
#[derive(Debug, Default, Clone)]
pub struct CalendarStats {
    pub events_processed: u64,
    pub events_created: u64,
    pub conflicts_detected: u64,
    pub suggestions_generated: u64,
    pub prep_sessions_scheduled: u64,
    pub schedule_optimizations: u64,
    pub cognitive_insights: u64,
    pub average_meeting_duration: Duration,
    pub focus_time_protected: Duration,
    pub uptime: Duration,
}

/// Main calendar integration client
pub struct CalendarManager {
    /// HTTP client for API calls
    http_client: Client,

    /// Configuration
    config: CalendarConfig,

    /// Cognitive system integration
    cognitive_system: Arc<CognitiveSystem>,

    /// Memory system
    memory: Arc<CognitiveMemory>,

    /// Current events cache
    events: Arc<RwLock<HashMap<String, CalendarEvent>>>,

    /// Time blocks
    time_blocks: Arc<RwLock<Vec<TimeBlock>>>,

    /// Schedule analysis cache
    schedule_analysis: Arc<RwLock<Option<ScheduleAnalysis>>>,

    /// API tokens cache
    tokens: Arc<RwLock<HashMap<String, String>>>,

    /// Event processing queue
    event_tx: mpsc::Sender<CalendarEvent>,
    event_rx: Arc<RwLock<Option<mpsc::Receiver<CalendarEvent>>>>,

    /// Event broadcast
    calendar_event_tx: broadcast::Sender<CalendarManagerEvent>,

    /// Shutdown signal
    shutdown_tx: broadcast::Sender<()>,

    /// Statistics
    stats: Arc<RwLock<CalendarStats>>,

    /// Running state
    running: Arc<RwLock<bool>>,
}

/// Calendar manager events
#[derive(Debug, Clone)]
pub enum CalendarManagerEvent {
    EventAdded(CalendarEvent),
    EventUpdated(CalendarEvent),
    EventRemoved(String),
    ConflictDetected(ScheduleConflict),
    MeetingPrepRequired { event_id: String, prep_time: Duration },
    ScheduleOptimized { suggestions: Vec<ScheduleSuggestion> },
    FocusTimeScheduled { time_block: TimeBlock },
    CognitiveTrigger { trigger: String, priority: Priority, context: String },
}

impl CalendarManager {
    /// Create new calendar manager
    pub async fn new(
        config: CalendarConfig,
        cognitive_system: Arc<CognitiveSystem>,
        memory: Arc<CognitiveMemory>,
    ) -> Result<Self> {
        info!("Initializing calendar manager");

        let http_client = Client::builder()
            .timeout(Duration::from_secs(30))
            .user_agent("Loki-Consciousness/1.0")
            .build()?;

        let (event_tx, event_rx) = mpsc::channel(1000);
        let (calendar_event_tx, _) = broadcast::channel(1000);
        let (shutdown_tx, _) = broadcast::channel(1);

        let manager = Self {
            http_client,
            config,
            cognitive_system,
            memory,
            events: Arc::new(RwLock::new(HashMap::new())),
            time_blocks: Arc::new(RwLock::new(Vec::new())),
            schedule_analysis: Arc::new(RwLock::new(None)),
            tokens: Arc::new(RwLock::new(HashMap::new())),
            event_tx,
            event_rx: Arc::new(RwLock::new(Some(event_rx))),
            calendar_event_tx,
            shutdown_tx,
            stats: Arc::new(RwLock::new(CalendarStats::default())),
            running: Arc::new(RwLock::new(false)),
        };

        // Initialize API tokens
        manager.refresh_tokens().await?;

        Ok(manager)
    }

    /// Start the calendar manager
    pub async fn start(&self) -> Result<()> {
        if *self.running.read().await {
            return Ok(());
        }

        info!("Starting calendar manager");
        *self.running.write().await = true;

        // Start calendar synchronization
        self.start_sync_loop().await?;

        // Start event processing
        self.start_event_processor().await?;

        // Start cognitive integration
        self.start_cognitive_integration().await?;

        // Start schedule optimization
        self.start_schedule_optimizer().await?;

        // Start periodic tasks
        self.start_periodic_tasks().await?;

        Ok(())
    }

    /// Refresh API tokens
    async fn refresh_tokens(&self) -> Result<()> {
        let mut tokens = self.tokens.write().await;

        // Refresh Google Calendar token if enabled
        if self.config.google_calendar_enabled && !self.config.google_refresh_token.is_empty() {
            let token = self.refresh_google_token().await?;
            tokens.insert("google".to_string(), token);
        }

        // Refresh Outlook token if enabled
        if self.config.outlook_enabled && !self.config.outlook_refresh_token.is_empty() {
            let token = self.refresh_outlook_token().await?;
            tokens.insert("outlook".to_string(), token);
        }

        Ok(())
    }

    /// Refresh Google Calendar access token
    async fn refresh_google_token(&self) -> Result<String> {
        let url = "https://oauth2.googleapis.com/token";
        let params = [
            ("client_id", &self.config.google_client_id),
            ("client_secret", &self.config.google_client_secret),
            ("refresh_token", &self.config.google_refresh_token),
            ("grant_type", &"refresh_token".to_string()),
        ];

        let response = self.http_client.post(url).form(&params).send().await?;

        let data: Value = response.json().await?;

        data["access_token"]
            .as_str()
            .map(|s| s.to_string())
            .ok_or_else(|| anyhow!("Failed to get Google access token"))
    }

    /// Refresh Outlook access token
    async fn refresh_outlook_token(&self) -> Result<String> {
        let url = format!(
            "https://login.microsoftonline.com/{}/oauth2/v2.0/token",
            self.config.outlook_tenant_id
        );

        let params = [
            ("client_id", &self.config.outlook_client_id),
            ("client_secret", &self.config.outlook_client_secret),
            ("refresh_token", &self.config.outlook_refresh_token),
            ("grant_type", &"refresh_token".to_string()),
            ("scope", &"https://graph.microsoft.com/calendars.readwrite".to_string()),
        ];

        let response = self.http_client.post(&url).form(&params).send().await?;

        let data: Value = response.json().await?;

        data["access_token"]
            .as_str()
            .map(|s| s.to_string())
            .ok_or_else(|| anyhow!("Failed to get Outlook access token"))
    }

    /// Start calendar synchronization loop
    async fn start_sync_loop(&self) -> Result<()> {
        let config = self.config.clone();
        let http_client = self.http_client.clone();
        let tokens = self.tokens.clone();
        let events = self.events.clone();
        let event_tx = self.event_tx.clone();
        let shutdown_rx = self.shutdown_tx.subscribe();

        tokio::spawn(async move {
            Self::sync_loop(config, http_client, tokens, events, event_tx, shutdown_rx).await;
        });

        Ok(())
    }

    /// Calendar synchronization loop
    async fn sync_loop(
        config: CalendarConfig,
        http_client: Client,
        tokens: Arc<RwLock<HashMap<String, String>>>,
        events: Arc<RwLock<HashMap<String, CalendarEvent>>>,
        event_tx: mpsc::Sender<CalendarEvent>,
        mut shutdown_rx: broadcast::Receiver<()>,
    ) {
        info!("Starting calendar sync loop");
        let mut interval = interval(config.sync_interval);

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    if let Err(e) = Self::sync_calendars(
                        &config,
                        &http_client,
                        &tokens,
                        &events,
                        &event_tx,
                    ).await {
                        warn!("Calendar sync error: {}", e);
                    }
                }

                _ = shutdown_rx.recv() => {
                    info!("Calendar sync loop shutting down");
                    break;
                }
            }
        }
    }

    /// Synchronize calendars from all enabled providers
    async fn sync_calendars(
        config: &CalendarConfig,
        http_client: &Client,
        tokens: &Arc<RwLock<HashMap<String, String>>>,
        events: &Arc<RwLock<HashMap<String, CalendarEvent>>>,
        event_tx: &mpsc::Sender<CalendarEvent>,
    ) -> Result<()> {
        debug!("Synchronizing calendars");

        let tokens_lock = tokens.read().await;

        // Sync Google Calendar if enabled
        if config.google_calendar_enabled {
            if let Some(token) = tokens_lock.get("google") {
                if let Err(e) =
                    Self::sync_google_calendar(config, http_client, token, events, event_tx).await
                {
                    warn!("Google Calendar sync error: {}", e);
                }
            }
        }

        // Sync Outlook if enabled
        if config.outlook_enabled {
            if let Some(token) = tokens_lock.get("outlook") {
                if let Err(e) =
                    Self::sync_outlook_calendar(config, http_client, token, events, event_tx).await
                {
                    warn!("Outlook sync error: {}", e);
                }
            }
        }

        Ok(())
    }

    /// Sync Google Calendar events
    async fn sync_google_calendar(
        config: &CalendarConfig,
        http_client: &Client,
        token: &str,
        events: &Arc<RwLock<HashMap<String, CalendarEvent>>>,
        event_tx: &mpsc::Sender<CalendarEvent>,
    ) -> Result<()> {
        let now = Utc::now();
        let time_min = now - chrono::Duration::days(config.lookback_days as i64);
        let time_max = now + chrono::Duration::days(config.lookahead_days as i64);

        for calendar_id in &config.monitored_calendars {
            let url =
                format!("https://www.googleapis.com/calendar/v3/calendars/{}/events", calendar_id);

            let response = http_client
                .get(&url)
                .bearer_auth(token)
                .query(&[
                    ("timeMin", time_min.to_rfc3339()),
                    ("timeMax", time_max.to_rfc3339()),
                    ("singleEvents", "true".to_string()),
                    ("orderBy", "startTime".to_string()),
                ])
                .send()
                .await?;

            if !response.status().is_success() {
                return Err(anyhow!("Google Calendar API error: {}", response.status()));
            }

            let data: Value = response.json().await?;

            if let Some(items) = data["items"].as_array() {
                for item in items {
                    if let Ok(event) = Self::parse_google_event(item, calendar_id) {
                        events.write().await.insert(event.id.clone(), event.clone());
                        let _ = event_tx.send(event).await;
                    }
                }
            }
        }

        Ok(())
    }

    /// Parse Google Calendar event
    fn parse_google_event(item: &Value, calendar_id: &str) -> Result<CalendarEvent> {
        let id = item["id"].as_str().unwrap_or("").to_string();
        let title = item["summary"].as_str().unwrap_or("Untitled").to_string();
        let description = item["description"].as_str().map(|s| s.to_string());
        let location = item["location"].as_str().map(|s| s.to_string());

        // Parse start and end times
        let start_time = if let Some(date_time) = item["start"]["dateTime"].as_str() {
            DateTime::parse_from_rfc3339(date_time)?.with_timezone(&Utc)
        } else if let Some(date) = item["start"]["date"].as_str() {
            chrono::NaiveDateTime::parse_from_str(
                &format!("{} 00:00:00", date),
                "%Y-%m-%d %H:%M:%S",
            )?
            .and_utc()
        } else {
            return Err(anyhow!("Invalid start time"));
        };

        let end_time = if let Some(date_time) = item["end"]["dateTime"].as_str() {
            DateTime::parse_from_rfc3339(date_time)?.with_timezone(&Utc)
        } else if let Some(date) = item["end"]["date"].as_str() {
            chrono::NaiveDateTime::parse_from_str(
                &format!("{} 23:59:59", date),
                "%Y-%m-%d %H:%M:%S",
            )?
            .and_utc()
        } else {
            return Err(anyhow!("Invalid end time"));
        };

        let all_day = item["start"]["date"].is_string();

        // Parse attendees
        let attendees = if let Some(attendees_array) = item["attendees"].as_array() {
            attendees_array
                .iter()
                .filter_map(|a| {
                    Some(EventAttendee {
                        email: a["email"].as_str()?.to_string(),
                        name: a["displayName"].as_str().map(|s| s.to_string()),
                        response_status: match a["responseStatus"].as_str() {
                            Some("accepted") => AttendeeResponse::Accepted,
                            Some("declined") => AttendeeResponse::Declined,
                            Some("tentative") => AttendeeResponse::Tentative,
                            _ => AttendeeResponse::NeedsAction,
                        },
                        is_organizer: a["organizer"].as_bool().unwrap_or(false),
                        is_optional: a["optional"].as_bool().unwrap_or(false),
                    })
                })
                .collect()
        } else {
            Vec::new()
        };

        Ok(CalendarEvent {
            id,
            calendar_id: calendar_id.to_string(),
            title,
            description,
            location,
            start_time,
            end_time,
            all_day,
            recurring: item["recurringEventId"].is_string(),
            attendees,
            organizer: None, // Would parse from organizer field
            status: match item["status"].as_str() {
                Some("confirmed") => EventStatus::Confirmed,
                Some("tentative") => EventStatus::Tentative,
                Some("cancelled") => EventStatus::Cancelled,
                _ => EventStatus::Confirmed,
            },
            visibility: EventVisibility::Default,
            event_type: EventType::Meeting, // Would infer from content
            meeting_link: item["hangoutLink"].as_str().map(|s| s.to_string()),
            agenda: None,
            priority: EventPriority::Normal,
            tags: Vec::new(),
            cognitive_metadata: EventCognitiveMetadata {
                preparation_required: false,
                estimated_preparation_time: Duration::from_secs(0),
                context_importance: 0.5,
                energy_requirement: 0.5,
                focus_level_needed: 0.5,
                suggested_prep_actions: Vec::new(),
                related_memories: Vec::new(),
                optimal_time_suggestion: None,
            },
        })
    }

    /// Sync Outlook calendar using Microsoft Graph API
    async fn sync_outlook_calendar(
        _config: &CalendarConfig,
        http_client: &Client,
        token: &str, // Microsoft Graph API access token
        events: &Arc<RwLock<HashMap<String, CalendarEvent>>>,
        event_tx: &mpsc::Sender<CalendarEvent>,
    ) -> Result<()> {
        debug!("Syncing Outlook calendar via Microsoft Graph API");

        // Microsoft Graph API endpoint for calendar events
        let graph_url = "https://graph.microsoft.com/v1.0/me/events";

        // Build request with authentication
        let response = http_client
            .get(graph_url)
            .header("Authorization", format!("Bearer {}", token))
            .header("Content-Type", "application/json")
            .query(&[
                ("$top", "50".to_string()), // Default to 50 events
                ("$orderby", "start/dateTime desc".to_string()),
                ("$select", "id,subject,start,end,bodyPreview,attendees,organizer,location".to_string()),
            ])
            .send()
            .await
            .context("Failed to fetch Outlook calendar events")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!(
                "Microsoft Graph API request failed with status {}: {}",
                status, error_text
            ));
        }

        // Parse the response
        let graph_response: serde_json::Value = response.json().await
            .context("Failed to parse Microsoft Graph API response")?;

        let outlook_events = graph_response
            .get("value")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow::anyhow!("Invalid response format from Microsoft Graph API"))?;

        debug!("Retrieved {} events from Outlook calendar", outlook_events.len());

        // Process and convert Outlook events to our format
        for outlook_event in outlook_events {
            if let Ok(calendar_event) = Self::convert_outlook_event_to_calendar_event(outlook_event) {
                // Store in local cache
                {
                    let mut events_guard = events.write().await;
                    events_guard.insert(calendar_event.id.clone(), calendar_event.clone());
                }

                // Send to event stream for real-time processing
                if let Err(e) = event_tx.send(calendar_event).await {
                    warn!("Failed to send calendar event to stream: {}", e);
                }
            }
        }

        info!("Outlook calendar sync completed successfully");
        Ok(())
    }

    /// Convert Microsoft Graph API event to our CalendarEvent format
    fn convert_outlook_event_to_calendar_event(outlook_event: &serde_json::Value) -> Result<CalendarEvent> {
        let id = outlook_event.get("id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing event ID"))?
            .to_string();

        let title = outlook_event.get("subject")
            .and_then(|v| v.as_str())
            .unwrap_or("Untitled Event")
            .to_string();

        // Parse start time
        let start_time = outlook_event.get("start")
            .and_then(|start| start.get("dateTime"))
            .and_then(|dt| dt.as_str())
            .and_then(|dt_str| DateTime::parse_from_rfc3339(dt_str).ok())
            .map(|dt| dt.with_timezone(&Utc))
            .ok_or_else(|| anyhow::anyhow!("Invalid start time format"))?;

        // Parse end time
        let end_time = outlook_event.get("end")
            .and_then(|end| end.get("dateTime"))
            .and_then(|dt| dt.as_str())
            .and_then(|dt_str| DateTime::parse_from_rfc3339(dt_str).ok())
            .map(|dt| dt.with_timezone(&Utc))
            .ok_or_else(|| anyhow::anyhow!("Invalid end time format"))?;

        // Extract description
        let description = outlook_event.get("bodyPreview")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        // Extract location
        let location = outlook_event.get("location")
            .and_then(|loc| loc.get("displayName"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        // Extract attendees
        let attendees = outlook_event.get("attendees")
            .and_then(|v| v.as_array())
            .map(|attendees_array| {
                attendees_array.iter()
                    .filter_map(|attendee| {
                        let email = attendee.get("emailAddress")
                            .and_then(|email| email.get("address"))
                            .and_then(|addr| addr.as_str())
                            .map(|s| s.to_string())?;
                        let name = attendee.get("emailAddress")
                            .and_then(|email| email.get("name"))
                            .and_then(|n| n.as_str())
                            .map(|s| s.to_string());
                        Some(EventAttendee {
                            email,
                            name,
                            response_status: AttendeeResponse::NeedsAction,
                            is_organizer: false,
                            is_optional: false,
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        // Extract organizer
        let organizer = outlook_event.get("organizer")
            .and_then(|org| org.get("emailAddress"))
            .map(|email| {
                let addr = email.get("address")
                    .and_then(|addr| addr.as_str())
                    .unwrap_or("").to_string();
                let name = email.get("name")
                    .and_then(|n| n.as_str())
                    .map(|s| s.to_string());
                EventAttendee {
                    email: addr,
                    name,
                    response_status: AttendeeResponse::Accepted,
                    is_organizer: true,
                    is_optional: false,
                }
            });

        Ok(CalendarEvent {
            id,
            calendar_id: "outlook".to_string(),
            title,
            description,
            start_time,
            end_time,
            location,
            attendees,
            organizer,
            all_day: false,
            recurring: false,
            status: EventStatus::Confirmed,
            visibility: EventVisibility::Default,
            event_type: EventType::Meeting,
            meeting_link: None,
            agenda: None,
            priority: EventPriority::Normal,
            tags: Vec::new(),
            cognitive_metadata: EventCognitiveMetadata {
                preparation_required: false,
                estimated_preparation_time: Duration::from_secs(0),
                context_importance: 0.5,
                energy_requirement: 0.5,
                focus_level_needed: 0.5,
                suggested_prep_actions: Vec::new(),
                related_memories: Vec::new(),
                optimal_time_suggestion: None,
            },
        })
    }

    /// Create calendar event
    pub async fn create_event(
        &self,
        title: &str,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
        description: Option<&str>,
        attendees: Vec<String>,
    ) -> Result<CalendarEvent> {
        info!("Creating calendar event: {}", title);

        // Create event object
        let event = CalendarEvent {
            id: uuid::Uuid::new_v4().to_string(),
            calendar_id: "primary".to_string(),
            title: title.to_string(),
            description: description.map(|s| s.to_string()),
            location: None,
            start_time,
            end_time,
            all_day: false,
            recurring: false,
            attendees: attendees
                .iter()
                .map(|email| EventAttendee {
                    email: email.clone(),
                    name: None,
                    response_status: AttendeeResponse::NeedsAction,
                    is_organizer: false,
                    is_optional: false,
                })
                .collect(),
            organizer: None,
            status: EventStatus::Confirmed,
            visibility: EventVisibility::Default,
            event_type: EventType::Meeting,
            meeting_link: None,
            agenda: None,
            priority: EventPriority::Normal,
            tags: Vec::new(),
            cognitive_metadata: EventCognitiveMetadata {
                preparation_required: true,
                estimated_preparation_time: self.config.meeting_prep_time,
                context_importance: 0.7,
                energy_requirement: 0.6,
                focus_level_needed: 0.7,
                suggested_prep_actions: Vec::new(),
                related_memories: Vec::new(),
                optimal_time_suggestion: None,
            },
        };

        // Store event
        self.events.write().await.insert(event.id.clone(), event.clone());

        // Update statistics
        {
            let mut stats = self.stats.write().await;
            stats.events_created += 1;
        }

        // Emit event
        let _ = self.calendar_event_tx.send(CalendarManagerEvent::EventAdded(event.clone()));

        Ok(event)
    }

    /// Start event processing loop
    async fn start_event_processor(&self) -> Result<()> {
        let event_rx = {
            let mut rx_lock = self.event_rx.write().await;
            rx_lock.take().ok_or_else(|| anyhow!("Event receiver already taken"))?
        };

        let cognitive_system = self.cognitive_system.clone();
        let memory = self.memory.clone();
        let config = self.config.clone();
        let stats = self.stats.clone();
        let calendar_event_tx = self.calendar_event_tx.clone();
        let shutdown_rx = self.shutdown_tx.subscribe();

        tokio::spawn(async move {
            Self::event_processing_loop(
                event_rx,
                cognitive_system,
                memory,
                config,
                stats,
                calendar_event_tx,
                shutdown_rx,
            )
            .await;
        });

        Ok(())
    }

    /// Event processing loop
    async fn event_processing_loop(
        mut event_rx: mpsc::Receiver<CalendarEvent>,
        cognitive_system: Arc<CognitiveSystem>,
        memory: Arc<CognitiveMemory>,
        config: CalendarConfig,
        stats: Arc<RwLock<CalendarStats>>,
        calendar_event_tx: broadcast::Sender<CalendarManagerEvent>,
        mut shutdown_rx: broadcast::Receiver<()>,
    ) {
        info!("Calendar event processing loop started");

        loop {
            tokio::select! {
                Some(event) = event_rx.recv() => {
                    if let Err(e) = Self::process_calendar_event(
                        event,
                        &cognitive_system,
                        &memory,
                        &config,
                        &stats,
                        &calendar_event_tx,
                    ).await {
                        error!("Calendar event processing error: {}", e);
                    }
                }

                _ = shutdown_rx.recv() => {
                    info!("Calendar event processing loop shutting down");
                    break;
                }
            }
        }
    }

    /// Process individual calendar event
    async fn process_calendar_event(
        event: CalendarEvent,
        cognitive_system: &Arc<CognitiveSystem>,
        memory: &Arc<CognitiveMemory>,
        config: &CalendarConfig,
        stats: &Arc<RwLock<CalendarStats>>,
        calendar_event_tx: &broadcast::Sender<CalendarManagerEvent>,
    ) -> Result<()> {
        debug!("Processing calendar event: {}", event.title);

        // Update statistics
        {
            let mut stats_lock = stats.write().await;
            stats_lock.events_processed += 1;
        }

        // Store in memory for context
        memory
            .store(
                format!("Calendar event: {}", event.title),
                vec![
                    format!("Time: {} to {}", event.start_time, event.end_time),
                    event.description.clone().unwrap_or_default(),
                ],
                MemoryMetadata {
                    source: "calendar".to_string(),
                    tags: vec![
                        "calendar".to_string(),
                        "event".to_string(),
                        format!("{:?}", event.event_type),
                    ],
                    importance: event.cognitive_metadata.context_importance,
                    associations: vec![],
                    context: Some("Calendar event".to_string()),
                    created_at: chrono::Utc::now(),
                    accessed_count: 0,
                    last_accessed: None,
                    version: 1,
                    category: "calendar".to_string(),
                    timestamp: chrono::Utc::now(),
                    expiration: None,
                },
            )
            .await?;

        // Create thought for cognitive processing
        let thought = Thought {
            id: crate::cognitive::ThoughtId::new(),
            content: format!("Upcoming event: {}", event.title),
            thought_type: match event.event_type {
                EventType::Meeting => ThoughtType::Social,
                EventType::Task => ThoughtType::Action,
                EventType::Appointment => ThoughtType::Planning,
                _ => ThoughtType::Observation,
            },
            metadata: ThoughtMetadata {
                source: "calendar".to_string(),
                confidence: 0.9,
                emotional_valence: 0.0, // Neutral for calendar events
                importance: event.cognitive_metadata.context_importance,
                tags: vec!["calendar".to_string(), "time_management".to_string()],
            },
            parent: None,
            children: Vec::new(),
            timestamp: Instant::now(),
        };

        // Send to cognitive system
        if config.cognitive_awareness_level > 0.5 {
            if let Err(e) = cognitive_system.process_query(&thought.content).await {
                warn!("Failed to process calendar thought: {}", e);
            }
        }

        // Check if meeting preparation is required
        if config.enable_meeting_preparation && event.cognitive_metadata.preparation_required {
            let prep_time = event.start_time
                - chrono::Duration::from_std(event.cognitive_metadata.estimated_preparation_time)?;

            if prep_time > Utc::now() {
                let _ = calendar_event_tx.send(CalendarManagerEvent::MeetingPrepRequired {
                    event_id: event.id.clone(),
                    prep_time: event.cognitive_metadata.estimated_preparation_time,
                });
            }
        }

        Ok(())
    }

    /// Start cognitive integration loop
    async fn start_cognitive_integration(&self) -> Result<()> {
        let cognitive_system = self.cognitive_system.clone();
        let memory = self.memory.clone();
        let config = self.config.clone();
        let event_rx = self.calendar_event_tx.subscribe();
        let shutdown_rx = self.shutdown_tx.subscribe();

        tokio::spawn(async move {
            Self::cognitive_integration_loop(
                cognitive_system,
                memory,
                config,
                event_rx,
                shutdown_rx,
            )
            .await;
        });

        Ok(())
    }

    /// Cognitive integration loop
    async fn cognitive_integration_loop(
        cognitive_system: Arc<CognitiveSystem>,
        memory: Arc<CognitiveMemory>,
        config: CalendarConfig,
        mut event_rx: broadcast::Receiver<CalendarManagerEvent>,
        mut shutdown_rx: broadcast::Receiver<()>,
    ) {
        info!("Calendar cognitive integration loop started");

        loop {
            tokio::select! {
                Ok(event) = event_rx.recv() => {
                    if let Err(e) = Self::handle_calendar_cognitive_event(
                        event,
                        &cognitive_system,
                        &memory,
                        &config,
                    ).await {
                        warn!("Calendar cognitive event handling error: {}", e);
                    }
                }

                _ = shutdown_rx.recv() => {
                    info!("Calendar cognitive integration loop shutting down");
                    break;
                }
            }
        }
    }

    /// Handle calendar cognitive events with configuration-driven processing
    async fn handle_calendar_cognitive_event(
        event: CalendarManagerEvent,
        cognitive_system: &Arc<CognitiveSystem>,
        memory: &Arc<CognitiveMemory>,
        config: &CalendarConfig,
    ) -> Result<()> {
        // **CONFIGURATION-DRIVEN EVENT FILTERING**

        // Only process events if cognitive awareness is enabled
        if config.cognitive_awareness_level < 0.1 {
            debug!(
                "Calendar cognitive awareness level too low ({:.2}), skipping event processing",
                config.cognitive_awareness_level
            );
            return Ok(());
        }

        match event {
            CalendarManagerEvent::MeetingPrepRequired { event_id, prep_time } => {
                info!("Meeting preparation required for event: {}", event_id);

                // **MEETING PREPARATION CONFIGURATION CHECK**
                if !config.enable_meeting_preparation {
                    debug!("Meeting preparation disabled, skipping prep processing");
                    return Ok(());
                }

                // Scale preparation importance by cognitive awareness and configured prep time
                let prep_importance = 0.8 * config.cognitive_awareness_level;
                let configured_prep_time = if prep_time < config.meeting_prep_time {
                    config.meeting_prep_time // Use configured minimum prep time
                } else {
                    prep_time
                };

                // Create preparation thought with configuration-aware metadata
                let thought = Thought {
                    id: crate::cognitive::ThoughtId::new(),
                    content: format!(
                        "Prepare for upcoming meeting: {} (prep time: {:?})",
                        event_id, configured_prep_time
                    ),
                    thought_type: ThoughtType::Planning,
                    metadata: ThoughtMetadata {
                        source: "calendar_prep".to_string(),
                        confidence: config.cognitive_awareness_level,
                        emotional_valence: 0.1, // Slight positive for preparation
                        importance: prep_importance,
                        tags: vec![
                            "meeting_prep".to_string(),
                            "planning".to_string(),
                            format!("awareness_{:.1}", config.cognitive_awareness_level),
                        ],
                    },
                    parent: None,
                    children: Vec::new(),
                    timestamp: Instant::now(),
                };

                // Store preparation context in memory with configuration details
                memory
                    .store(
                        format!("Meeting preparation scheduled: {}", event_id),
                        vec![
                            format!("Preparation time: {:?}", configured_prep_time),
                            format!("Default buffer: {:?}", config.default_meeting_buffer),
                            format!("Travel estimation enabled: {}", config.travel_time_estimation),
                        ],
                        MemoryMetadata {
                            source: "calendar_prep".to_string(),
                            tags: vec![
                                "meeting_preparation".to_string(),
                                "time_management".to_string(),
                                "productivity".to_string(),
                            ],
                            importance: prep_importance,
                            associations: vec![],
                            context: Some("Meeting preparation".to_string()),
                            created_at: chrono::Utc::now(),
                            accessed_count: 0,
                            last_accessed: None,
                            version: 1,
                            category: "calendar".to_string(),
                            timestamp: chrono::Utc::now(),
                            expiration: None,
                        },
                    )
                    .await?;

                // Only send to cognitive system if awareness is high enough
                if config.cognitive_awareness_level > 0.3 {
                    cognitive_system.process_query(&thought.content).await?;
                    debug!(
                        "Meeting preparation processed with awareness level {:.2}",
                        config.cognitive_awareness_level
                    );
                }
            }

            CalendarManagerEvent::ConflictDetected(conflict) => {
                warn!("Schedule conflict detected: {:?}", conflict.conflict_type);

                // **CONFLICT DETECTION CONFIGURATION CHECK**
                if !config.enable_conflict_detection {
                    debug!("Conflict detection disabled, skipping conflict processing");
                    return Ok(());
                }

                // Scale conflict importance by severity and cognitive awareness
                let conflict_importance = conflict.severity * config.cognitive_awareness_level;

                // Determine if auto-decline should be suggested
                let should_auto_decline = config.auto_decline_conflicts
                    && conflict.severity > 0.7
                    && matches!(
                        conflict.conflict_type,
                        ConflictType::TimeOverlap | ConflictType::TravelImpossible
                    );

                // Store conflict information with configuration-aware resolution suggestions
                let mut resolution_suggestions = conflict.resolution_suggestions.clone();
                if should_auto_decline {
                    resolution_suggestions
                        .push("Auto-decline conflicting event (enabled in config)".to_string());
                }
                if config.travel_time_estimation
                    && matches!(conflict.conflict_type, ConflictType::TravelImpossible)
                {
                    resolution_suggestions.push("Adjust for travel time estimation".to_string());
                }
                if config.default_meeting_buffer > Duration::from_secs(0) {
                    resolution_suggestions.push(format!(
                        "Add default meeting buffer: {:?}",
                        config.default_meeting_buffer
                    ));
                }

                memory
                    .store(
                        format!(
                            "Schedule conflict: {:?} (severity: {:.2})",
                            conflict.conflict_type, conflict.severity
                        ),
                        vec![
                            format!("Conflicting events: {:?}", conflict.events),
                            format!("Auto-decline enabled: {}", config.auto_decline_conflicts),
                            format!("Travel estimation: {}", config.travel_time_estimation),
                            format!("Resolution suggestions: {:?}", resolution_suggestions),
                        ],
                        MemoryMetadata {
                            source: "calendar_conflict".to_string(),
                            tags: vec![
                                "conflict".to_string(),
                                "scheduling".to_string(),
                                format!("{:?}", conflict.conflict_type),
                                if should_auto_decline {
                                    "auto_decline"
                                } else {
                                    "manual_resolution"
                                }
                                .to_string(),
                            ],
                            importance: conflict_importance,
                            associations: vec![],
                            context: Some("Calendar conflict resolution".to_string()),
                            created_at: chrono::Utc::now(),
                            accessed_count: 0,
                            last_accessed: None,
                            version: 1,
                            category: "calendar_conflict".to_string(),
                            timestamp: chrono::Utc::now(),
                            expiration: None,
                        },
                    )
                    .await?;

                // Create high-priority thought for severe conflicts
                if conflict.severity > 0.6 {
                    let thought = Thought {
                        id: crate::cognitive::ThoughtId::new(),
                        content: format!(
                            "SCHEDULE CONFLICT: {:?} (severity: {:.2})",
                            conflict.conflict_type, conflict.severity
                        ),
                        thought_type: ThoughtType::Decision,
                        metadata: ThoughtMetadata {
                            source: "calendar_conflict".to_string(),
                            confidence: config.cognitive_awareness_level,
                            emotional_valence: -0.3, // Negative due to conflict stress
                            importance: conflict_importance,
                            tags: vec![
                                "urgent".to_string(),
                                "conflict_resolution".to_string(),
                                format!("{:?}", conflict.conflict_type),
                            ],
                        },
                        parent: None,
                        children: Vec::new(),
                        timestamp: Instant::now(),
                    };

                    cognitive_system.process_query(&thought.content).await?;
                }

                debug!(
                    "Conflict processed with importance {:.2}, auto-decline: {}",
                    conflict_importance, should_auto_decline
                );
            }

            CalendarManagerEvent::ScheduleOptimized { suggestions } => {
                info!("Schedule optimization completed with {} suggestions", suggestions.len());

                // **SCHEDULE OPTIMIZATION CONFIGURATION CHECK**
                if !config.enable_schedule_optimization {
                    debug!("Schedule optimization disabled, skipping optimization processing");
                    return Ok(());
                }

                // Process suggestions based on configuration preferences
                for suggestion in &suggestions {
                    let suggestion_importance = (1.0 - suggestion.implementation_difficulty)
                        * config.cognitive_awareness_level;

                    // Apply configuration-specific filtering
                    let should_implement = match suggestion.suggestion_type {
                        SuggestionType::AddBuffer => {
                            config.default_meeting_buffer > Duration::from_secs(0)
                        }
                        SuggestionType::OptimizeTravel => config.travel_time_estimation,
                        SuggestionType::BlockFocusTime => config.enable_time_blocking,
                        _ => true, // Other suggestions always considered
                    };

                    if should_implement && suggestion_importance > 0.2 {
                        memory
                            .store(
                                format!("Schedule optimization: {}", suggestion.description),
                                vec![
                                    format!("Type: {:?}", suggestion.suggestion_type),
                                    format!("Benefit: {}", suggestion.potential_benefit),
                                    format!(
                                        "Difficulty: {:.2}",
                                        suggestion.implementation_difficulty
                                    ),
                                    format!("Configuration supports: {}", should_implement),
                                ],
                                MemoryMetadata {
                                    source: "calendar_optimization".to_string(),
                                    tags: vec![
                                        "optimization".to_string(),
                                        "productivity".to_string(),
                                        format!("{:?}", suggestion.suggestion_type),
                                    ],
                                    importance: suggestion_importance,
                                    associations: vec![],
                                    context: Some("Calendar optimization suggestion".to_string()),
                                    created_at: chrono::Utc::now(),
                                    accessed_count: 0,
                                    last_accessed: None,
                                    version: 1,
                                    category: "calendar".to_string(),
                                    timestamp: chrono::Utc::now(),
                                    expiration: None,
                                },
                            )
                            .await?;

                        debug!(
                            "Optimization suggestion stored: {} (importance: {:.2})",
                            suggestion.description, suggestion_importance
                        );
                    }
                }

                // Create thought for high-impact optimization suggestions
                let high_impact_suggestions: Vec<_> =
                    suggestions.iter().filter(|s| s.implementation_difficulty < 0.5).collect();

                if !high_impact_suggestions.is_empty() && config.cognitive_awareness_level > 0.4 {
                    let thought = Thought {
                        id: crate::cognitive::ThoughtId::new(),
                        content: format!(
                            "Schedule optimization opportunities: {} high-impact suggestions \
                             available",
                            high_impact_suggestions.len()
                        ),
                        thought_type: ThoughtType::Analysis,
                        metadata: ThoughtMetadata {
                            source: "calendar_optimization".to_string(),
                            confidence: config.cognitive_awareness_level,
                            emotional_valence: 0.2, // Positive for improvement opportunities
                            importance: 0.7 * config.cognitive_awareness_level,
                            tags: vec![
                                "optimization".to_string(),
                                "productivity_improvement".to_string(),
                            ],
                        },
                        parent: None,
                        children: Vec::new(),
                        timestamp: Instant::now(),
                    };

                    cognitive_system.process_query(&thought.content).await?;
                }
            }

            CalendarManagerEvent::FocusTimeScheduled { time_block } => {
                info!("Focus time scheduled: {} ({:?})", time_block.title, time_block.purpose);

                // **TIME BLOCKING CONFIGURATION CHECK**
                if !config.enable_time_blocking {
                    debug!("Time blocking disabled, skipping focus time processing");
                    return Ok(());
                }

                // Calculate focus time importance based on energy level and configuration
                let focus_importance = time_block.energy_level * config.cognitive_awareness_level;

                memory
                    .store(
                        format!(
                            "Focus time block: {} ({:?})",
                            time_block.title, time_block.purpose
                        ),
                        vec![
                            format!("Duration: {:?}", time_block.duration),
                            format!("Energy level: {:.2}", time_block.energy_level),
                            format!("Flexibility: {:.2}", time_block.flexibility),
                            format!(
                                "Focus requirements: concentration={:.2}, \
                                 interruption_tolerance={:.2}",
                                time_block.focus_requirements.concentration_level,
                                time_block.focus_requirements.interruption_tolerance
                            ),
                        ],
                        MemoryMetadata {
                            source: "calendar_focus_time".to_string(),
                            tags: vec![
                                "focus_time".to_string(),
                                "productivity".to_string(),
                                format!("{:?}", time_block.purpose),
                            ],
                            importance: focus_importance,
                            associations: vec![],
                            context: Some("Focus time block".to_string()),
                            created_at: chrono::Utc::now(),
                            accessed_count: 0,
                            last_accessed: None,
                            version: 1,
                            category: "calendar".to_string(),
                            timestamp: chrono::Utc::now(),
                            expiration: None,
                        },
                    )
                    .await?;

                // Create thought for high-energy focus blocks
                if time_block.energy_level > 0.7 && config.cognitive_awareness_level > 0.3 {
                    let thought = Thought {
                        id: crate::cognitive::ThoughtId::new(),
                        content: format!(
                            "High-energy focus time scheduled: {} ({:?})",
                            time_block.title, time_block.purpose
                        ),
                        thought_type: ThoughtType::Planning,
                        metadata: ThoughtMetadata {
                            source: "calendar_focus".to_string(),
                            confidence: config.cognitive_awareness_level,
                            emotional_valence: 0.3, // Positive for productive time
                            importance: focus_importance,
                            tags: vec![
                                "deep_work".to_string(),
                                "high_energy".to_string(),
                                format!("{:?}", time_block.purpose),
                            ],
                        },
                        parent: None,
                        children: Vec::new(),
                        timestamp: Instant::now(),
                    };

                    cognitive_system.process_query(&thought.content).await?;
                }

                debug!("Focus time processed with importance {:.2}", focus_importance);
            }

            CalendarManagerEvent::CognitiveTrigger { trigger, priority, context } => {
                info!("Calendar cognitive trigger: {} (priority: {:?})", trigger, priority);

                // Scale trigger importance by priority and configuration awareness
                let trigger_importance = match priority {
                    Priority::High => 0.9 * config.cognitive_awareness_level,
                    Priority::Medium => 0.6 * config.cognitive_awareness_level,
                    Priority::Low => 0.3 * config.cognitive_awareness_level,
                    Priority::Critical => 1.0 * config.cognitive_awareness_level,
                };

                // Only process if importance meets configuration-based threshold
                let min_threshold = 1.0 - config.cognitive_awareness_level; // Higher awareness = lower threshold
                if trigger_importance > min_threshold {
                    let thought = Thought {
                        id: crate::cognitive::ThoughtId::new(),
                        content: format!("Calendar cognitive trigger: {} - {}", trigger, context),
                        thought_type: ThoughtType::Analysis,
                        metadata: ThoughtMetadata {
                            source: "calendar_trigger".to_string(),
                            confidence: config.cognitive_awareness_level,
                            emotional_valence: 0.0,
                            importance: trigger_importance,
                            tags: vec![
                                "cognitive_trigger".to_string(),
                                format!("priority_{:?}", priority),
                                "calendar_intelligence".to_string(),
                            ],
                        },
                        parent: None,
                        children: Vec::new(),
                        timestamp: Instant::now(),
                    };

                    cognitive_system.process_query(&thought.content).await?;
                    debug!(
                        "Calendar cognitive trigger processed with importance {:.2}",
                        trigger_importance
                    );
                } else {
                    debug!(
                        "Calendar trigger importance ({:.2}) below threshold ({:.2}), skipping",
                        trigger_importance, min_threshold
                    );
                }
            }

            _ => {
                debug!(
                    "Handling other calendar event with awareness level {:.2}",
                    config.cognitive_awareness_level
                );
            }
        }

        Ok(())
    }

    /// Start schedule optimizer
    async fn start_schedule_optimizer(&self) -> Result<()> {
        let events = self.events.clone();
        let schedule_analysis = self.schedule_analysis.clone();
        let config = self.config.clone();
        let calendar_event_tx = self.calendar_event_tx.clone();
        let shutdown_rx = self.shutdown_tx.subscribe();

        tokio::spawn(async move {
            Self::schedule_optimization_loop(
                events,
                schedule_analysis,
                config,
                calendar_event_tx,
                shutdown_rx,
            )
            .await;
        });

        Ok(())
    }

    /// Schedule optimization loop
    async fn schedule_optimization_loop(
        events: Arc<RwLock<HashMap<String, CalendarEvent>>>,
        schedule_analysis: Arc<RwLock<Option<ScheduleAnalysis>>>,
        config: CalendarConfig,
        calendar_event_tx: broadcast::Sender<CalendarManagerEvent>,
        mut shutdown_rx: broadcast::Receiver<()>,
    ) {
        let mut interval = interval(Duration::from_secs(3600)); // Every hour

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    if config.enable_schedule_optimization {
                        if let Err(e) = Self::optimize_schedule(
                            &events,
                            &schedule_analysis,
                            &calendar_event_tx,
                        ).await {
                            warn!("Schedule optimization error: {}", e);
                        }
                    }
                }

                _ = shutdown_rx.recv() => {
                    info!("Schedule optimization loop shutting down");
                    break;
                }
            }
        }
    }

    /// Optimize schedule
    async fn optimize_schedule(
        events: &Arc<RwLock<HashMap<String, CalendarEvent>>>,
        schedule_analysis: &Arc<RwLock<Option<ScheduleAnalysis>>>,
        calendar_event_tx: &broadcast::Sender<CalendarManagerEvent>,
    ) -> Result<()> {
        debug!("Optimizing schedule");

        let events_lock = events.read().await;
        let today_events: Vec<_> = events_lock
            .values()
            .filter(|e| e.start_time.date_naive() == Utc::now().date_naive())
            .cloned()
            .collect();

        // Analyze current schedule
        let analysis = Self::analyze_schedule(&today_events).await;

        // Generate optimization suggestions
        let suggestions = Self::generate_optimization_suggestions(&analysis).await;

        if !suggestions.is_empty() {
            let _ = calendar_event_tx.send(CalendarManagerEvent::ScheduleOptimized { suggestions });
        }

        // Store analysis
        *schedule_analysis.write().await = Some(analysis);

        Ok(())
    }

    /// Analyze current schedule
    async fn analyze_schedule(events: &[CalendarEvent]) -> ScheduleAnalysis {
        let total_scheduled_time =
            events.iter().map(|e| (e.end_time - e.start_time).to_std().unwrap_or_default()).sum();

        // Simple analysis for now
        ScheduleAnalysis {
            total_scheduled_time,
            free_time_blocks: Vec::new(),
            meeting_density: events.len() as f32 / 8.0, // Assuming 8-hour workday
            travel_time: Duration::from_secs(0),
            energy_distribution: EnergyDistribution {
                morning_load: 0.3,
                afternoon_load: 0.5,
                evening_load: 0.2,
                peak_hours: vec![9, 10, 14, 15],
                low_energy_periods: vec![13, 16],
            },
            optimization_suggestions: Vec::new(),
            conflict_warnings: Vec::new(),
        }
    }

    /// Generate optimization suggestions
    async fn generate_optimization_suggestions(
        analysis: &ScheduleAnalysis,
    ) -> Vec<ScheduleSuggestion> {
        let mut suggestions = Vec::new();

        if analysis.meeting_density > 0.8 {
            suggestions.push(ScheduleSuggestion {
                suggestion_type: SuggestionType::AddBuffer,
                description: "High meeting density detected".to_string(),
                potential_benefit: "Reduce stress and improve focus".to_string(),
                implementation_difficulty: 0.3,
            });
        }

        if analysis.total_scheduled_time > Duration::from_secs(28800) {
            // 8 hours
            suggestions.push(ScheduleSuggestion {
                suggestion_type: SuggestionType::RescheduleEvent,
                description: "Overbooked day detected".to_string(),
                potential_benefit: "Better work-life balance".to_string(),
                implementation_difficulty: 0.7,
            });
        }

        suggestions
    }

    /// Start periodic tasks
    async fn start_periodic_tasks(&self) -> Result<()> {
        let stats = self.stats.clone();
        let shutdown_rx = self.shutdown_tx.subscribe();

        tokio::spawn(async move {
            Self::periodic_tasks_loop(stats, shutdown_rx).await;
        });

        Ok(())
    }

    /// Periodic tasks loop
    async fn periodic_tasks_loop(
        stats: Arc<RwLock<CalendarStats>>,
        mut shutdown_rx: broadcast::Receiver<()>,
    ) {
        let mut interval = interval(Duration::from_secs(300)); // Every 5 minutes

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    // Update statistics
                    {
                        let mut stats_lock = stats.write().await;
                        stats_lock.uptime += Duration::from_secs(300);
                    }

                    debug!("Calendar periodic tasks completed");
                }

                _ = shutdown_rx.recv() => {
                    info!("Calendar periodic tasks shutting down");
                    break;
                }
            }
        }
    }

    /// Get current events
    pub async fn get_events(&self) -> HashMap<String, CalendarEvent> {
        self.events.read().await.clone()
    }

    /// Get events for a specific date range
    pub async fn get_events_in_range(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Vec<CalendarEvent> {
        self.events
            .read()
            .await
            .values()
            .filter(|e| e.start_time >= start && e.end_time <= end)
            .cloned()
            .collect()
    }

    /// Get current schedule analysis
    pub async fn get_schedule_analysis(&self) -> Option<ScheduleAnalysis> {
        self.schedule_analysis.read().await.clone()
    }

    /// Get current statistics
    pub async fn get_stats(&self) -> CalendarStats {
        self.stats.read().await.clone()
    }

    /// Subscribe to calendar events
    pub fn subscribe_events(&self) -> broadcast::Receiver<CalendarManagerEvent> {
        self.calendar_event_tx.subscribe()
    }

    /// Shutdown the calendar manager
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down calendar manager");

        *self.running.write().await = false;
        let _ = self.shutdown_tx.send(());

        Ok(())
    }
}
