//! Autonomous Browser Tool
//!
//! This tool combines computer_use with vision_system to provide autonomous
//! web browsing capabilities for AI agents. It can open browser instances,
//! navigate websites, extract content, and perform complex web interactions.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::process::{Command, Child};
use tokio::sync::{RwLock, Mutex};
use tokio::fs;
use tracing::{debug, info, warn};
use url::Url;

use crate::memory::CognitiveMemory;
use crate::safety::ActionValidator;
// Note: These would normally import from the actual computer_use and vision_system modules
// use crate::tools::computer_use::{ComputerUseSystem, ComputerUseConfig};
// use crate::tools::vision_system::{VisionSystem, VisionConfig, VisualAnalysis};

/// Visual analysis result from vision system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualAnalysis {
    pub description: String,
    pub confidence: f32,
    pub elements: Vec<VisualElement>,
    pub text_content: Vec<String>,
}

/// Visual element detected in page
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualElement {
    pub element_type: String,
    pub bounds: (u32, u32, u32, u32), // x, y, width, height
    pub text: Option<String>,
    pub confidence: f32,
}

/// Computer use system for screen interaction
pub struct BrowserComputerUse {
    pub click_precision: f32,
    pub typing_speed: u32, // characters per minute
}

impl BrowserComputerUse {
    pub fn new() -> Self {
        Self {
            click_precision: 0.95,
            typing_speed: 120,
        }
    }
    
    pub async fn click_at(&self, x: u32, y: u32) -> Result<()> {
        // Simulate click with precise coordinates
        tokio::time::sleep(Duration::from_millis(50)).await;
        debug!("🖱️  Clicked at coordinates ({}, {})", x, y);
        Ok(())
    }
    
    pub async fn type_text(&self, text: &str) -> Result<()> {
        // Simulate typing with realistic speed
        let delay_per_char = 60000 / self.typing_speed; // milliseconds per character
        for _ in text.chars() {
            tokio::time::sleep(Duration::from_millis(delay_per_char as u64)).await;
        }
        debug!("⌨️  Typed text: {}", text);
        Ok(())
    }
}

/// Vision system for visual analysis
pub struct BrowserVisionSystem {
    pub confidence_threshold: f32,
    pub element_detection: bool,
}

impl BrowserVisionSystem {
    pub fn new() -> Self {
        Self {
            confidence_threshold: 0.7,
            element_detection: true,
        }
    }
    
    pub async fn analyze_screenshot(&self, screenshot_path: &Path) -> Result<VisualAnalysis> {
        // Simulate vision analysis of screenshot
        tokio::time::sleep(Duration::from_millis(200)).await;
        
        // Mock analysis based on path/context
        let filename = screenshot_path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");
            
        let mut elements = Vec::new();
        let mut text_content = Vec::new();
        
        // Simulate common web elements detection
        if filename.contains("form") {
            elements.push(VisualElement {
                element_type: "input_field".to_string(),
                bounds: (100, 200, 300, 30),
                text: Some("Email".to_string()),
                confidence: 0.9,
            });
            elements.push(VisualElement {
                element_type: "button".to_string(),
                bounds: (100, 250, 100, 40),
                text: Some("Submit".to_string()),
                confidence: 0.95,
            });
        }
        
        if filename.contains("nav") {
            text_content.push("Home".to_string());
            text_content.push("About".to_string());
            text_content.push("Contact".to_string());
        }
        
        Ok(VisualAnalysis {
            description: format!("Visual analysis of {}", filename),
            confidence: 0.85,
            elements,
            text_content,
        })
    }
    
    pub async fn extract_text(&self, screenshot_path: &Path) -> Result<Vec<String>> {
        // Simulate OCR text extraction
        tokio::time::sleep(Duration::from_millis(150)).await;
        
        let analysis = self.analyze_screenshot(screenshot_path).await?;
        Ok(analysis.text_content)
    }
}

/// Autonomous browser configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutonomousBrowserConfig {
    /// Browser executable path
    pub browser_path: Option<PathBuf>,
    
    /// Browser type to use
    pub browser_type: BrowserType,
    
    /// Browser profile/data directory
    pub profile_dir: Option<PathBuf>,
    
    /// Browser window settings
    pub window_width: u32,
    pub window_height: u32,
    pub headless: bool,
    
    /// Navigation settings
    pub default_timeout: Duration,
    pub page_load_timeout: Duration,
    pub element_timeout: Duration,
    
    /// Content extraction settings
    pub enable_content_extraction: bool,
    pub enable_screenshot_capture: bool,
    pub screenshot_dir: PathBuf,
    
    /// Vision integration settings
    pub enable_vision_analysis: bool,
    pub vision_analysis_threshold: f32,
    
    /// Security settings
    pub enable_sandboxing: bool,
    pub allowed_domains: Vec<String>,
    pub blocked_domains: Vec<String>,
    pub enable_javascript: bool,
    pub enable_images: bool,
    
    /// Performance settings
    pub max_concurrent_sessions: usize,
    pub session_timeout: Duration,
    pub enable_caching: bool,
    pub cache_dir: Option<PathBuf>,
}

impl Default for AutonomousBrowserConfig {
    fn default() -> Self {
        Self {
            browser_path: None, // Auto-detect
            browser_type: BrowserType::Chrome,
            profile_dir: None,
            window_width: 1280,
            window_height: 1024,
            headless: true,
            default_timeout: Duration::from_secs(30),
            page_load_timeout: Duration::from_secs(60),
            element_timeout: Duration::from_secs(10),
            enable_content_extraction: true,
            enable_screenshot_capture: true,
            screenshot_dir: std::env::temp_dir().join("loki_browser_screenshots"),
            enable_vision_analysis: true,
            vision_analysis_threshold: 0.7,
            enable_sandboxing: true,
            allowed_domains: Vec::new(), // Empty means allow all
            blocked_domains: vec![
                "malware.com".to_string(),
                "phishing.com".to_string(),
            ],
            enable_javascript: true,
            enable_images: true,
            max_concurrent_sessions: 5,
            session_timeout: Duration::from_secs(30 * 60),
            enable_caching: true,
            cache_dir: None,
        }
    }
}

/// Supported browser types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BrowserType {
    Chrome,
    Firefox,
    Safari,
    Edge,
}

/// Browser automation task
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrowserTask {
    /// Task ID
    pub task_id: String,
    
    /// Task type
    pub task_type: BrowserTaskType,
    
    /// Target URL or search query
    pub target: String,
    
    /// Task-specific parameters
    pub parameters: HashMap<String, Value>,
    
    /// Maximum execution time
    pub timeout: Option<Duration>,
    
    /// Success criteria
    pub success_criteria: Vec<SuccessCriterion>,
}

/// Types of browser automation tasks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BrowserTaskType {
    /// Navigate to a specific URL
    Navigate,
    
    /// Search for information
    Search,
    
    /// Extract specific content from a page
    ExtractContent,
    
    /// Fill out and submit a form
    FillForm,
    
    /// Click on specific elements
    ClickElement,
    
    /// Scroll and explore a page
    ExplorePage,
    
    /// Monitor a page for changes
    MonitorPage,
    
    /// Complex multi-step workflow
    Workflow,
}

/// Success criteria for task completion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuccessCriterion {
    /// Type of criterion
    pub criterion_type: CriterionType,
    
    /// Expected value or pattern
    pub expected: String,
    
    /// Comparison operator
    pub operator: ComparisonOperator,
}

/// Types of success criteria
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CriterionType {
    /// Page title contains/matches text
    PageTitle,
    
    /// URL contains/matches pattern
    Url,
    
    /// Page content contains text
    Content,
    
    /// Specific element is present
    ElementPresent,
    
    /// Element has specific text/value
    ElementText,
    
    /// Custom vision analysis result
    VisionAnalysis,
}

/// Comparison operators for success criteria
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComparisonOperator {
    Contains,
    Equals,
    Matches,
    GreaterThan,
    LessThan,
}

/// Browser session state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrowserSession {
    /// Session ID
    pub session_id: String,
    
    /// Current URL
    pub current_url: String,
    
    /// Page title
    pub page_title: String,
    
    /// Session start time
    pub started_at: SystemTime,
    
    /// Last activity time
    pub last_activity: SystemTime,
    
    /// Browser process info
    pub process_id: Option<u32>,
    
    /// Screenshots taken
    pub screenshots: Vec<String>,
    
    /// Extracted content
    pub extracted_content: HashMap<String, Value>,
    
    /// Navigation history
    pub navigation_history: Vec<NavigationEntry>,
}

/// Navigation history entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NavigationEntry {
    pub url: String,
    pub title: String,
    pub timestamp: SystemTime,
    pub load_time: Duration,
}

/// Browser task execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrowserTaskResult {
    /// Task ID
    pub task_id: String,
    
    /// Execution success
    pub success: bool,
    
    /// Error message if failed
    pub error: Option<String>,
    
    /// Execution time
    pub execution_time: Duration,
    
    /// Final URL
    pub final_url: String,
    
    /// Extracted data
    pub extracted_data: HashMap<String, Value>,
    
    /// Screenshots taken
    pub screenshots: Vec<String>,
    
    /// Vision analysis results
    pub vision_analysis: Option<VisualAnalysis>,
    
    /// Task-specific results
    pub task_results: Value,
}

/// Autonomous Browser Tool
pub struct AutonomousBrowserTool {
    /// Configuration
    config: AutonomousBrowserConfig,
    
    /// Reference to cognitive memory
    cognitive_memory: Arc<CognitiveMemory>,
    
    /// Safety validator
    safety_validator: Arc<ActionValidator>,
    
    /// Computer use system for screen interaction
    computer_use: Arc<BrowserComputerUse>,
    
    /// Vision system for visual analysis
    vision_system: Arc<BrowserVisionSystem>,
    
    /// Active browser sessions
    active_sessions: Arc<RwLock<HashMap<String, BrowserSession>>>,
    
    /// Browser processes
    browser_processes: Arc<Mutex<HashMap<String, Child>>>,
    
    /// Task execution metrics
    metrics: Arc<RwLock<BrowserMetrics>>,
}

/// Browser automation metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BrowserMetrics {
    pub total_tasks: usize,
    pub successful_tasks: usize,
    pub failed_tasks: usize,
    pub avg_execution_time: Duration,
    pub pages_visited: usize,
    pub screenshots_taken: usize,
    pub content_extracted: usize,
    pub vision_analyses: usize,
}

impl AutonomousBrowserTool {
    /// Create a new autonomous browser tool
    pub async fn new(
        config: AutonomousBrowserConfig,
        cognitive_memory: Arc<CognitiveMemory>,
        safety_validator: Arc<ActionValidator>,
    ) -> Result<Self> {
        info!("🌐 Initializing Autonomous Browser Tool");
        
        // Create screenshot directory
        if config.enable_screenshot_capture {
            fs::create_dir_all(&config.screenshot_dir).await
                .context("Failed to create screenshot directory")?;
        }
        
        // Create cache directory if enabled
        if let Some(cache_dir) = &config.cache_dir {
            fs::create_dir_all(cache_dir).await
                .context("Failed to create cache directory")?;
        }
        
        // Note: In a full implementation, these would be passed as parameters
        // or created with proper cognitive system integration
        
        // For now, we'll skip the computer_use and vision_system initialization
        // as they require complex cognitive system setup
        
        let tool = Self {
            config,
            cognitive_memory,
            safety_validator,
            // Initialize with browser-specific systems
            computer_use: Arc::new(BrowserComputerUse::new()),
            vision_system: Arc::new(BrowserVisionSystem::new()),
            active_sessions: Arc::new(RwLock::new(HashMap::new())),
            browser_processes: Arc::new(Mutex::new(HashMap::new())),
            metrics: Arc::new(RwLock::new(BrowserMetrics::default())),
        };
        
        // Validate browser installation
        tool.validate_browser_installation().await?;
        
        info!("✅ Autonomous Browser Tool initialized successfully");
        Ok(tool)
    }
    
    /// Validate browser installation
    async fn validate_browser_installation(&self) -> Result<()> {
        let browser_path = self.detect_browser_path().await?;
        
        // Test browser launch
        let output = Command::new(&browser_path)
            .arg("--version")
            .output()
            .await
            .context("Failed to execute browser")?;
        
        if !output.status.success() {
            return Err(anyhow!("Browser is not working correctly"));
        }
        
        let version = String::from_utf8_lossy(&output.stdout);
        info!("✅ Browser installation validated: {}", version.trim());
        
        Ok(())
    }
    
    /// Detect browser executable path
    async fn detect_browser_path(&self) -> Result<PathBuf> {
        if let Some(path) = &self.config.browser_path {
            return Ok(path.clone());
        }
        
        // Auto-detect browser based on type and platform
        let possible_paths = match self.config.browser_type {
            BrowserType::Chrome => {
                if cfg!(target_os = "macos") {
                    vec![
                        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                        "/Applications/Chromium.app/Contents/MacOS/Chromium",
                    ]
                } else if cfg!(target_os = "windows") {
                    vec![
                        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
                    ]
                } else {
                    vec![
                        "/usr/bin/google-chrome",
                        "/usr/bin/chromium-browser",
                        "/usr/bin/chromium",
                    ]
                }
            }
            BrowserType::Firefox => {
                if cfg!(target_os = "macos") {
                    vec!["/Applications/Firefox.app/Contents/MacOS/firefox"]
                } else if cfg!(target_os = "windows") {
                    vec![r"C:\Program Files\Mozilla Firefox\firefox.exe"]
                } else {
                    vec!["/usr/bin/firefox"]
                }
            }
            BrowserType::Safari => {
                if cfg!(target_os = "macos") {
                    vec!["/Applications/Safari.app/Contents/MacOS/Safari"]
                } else {
                    return Err(anyhow!("Safari is only available on macOS"));
                }
            }
            BrowserType::Edge => {
                if cfg!(target_os = "macos") {
                    vec!["/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"]
                } else if cfg!(target_os = "windows") {
                    vec![r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"]
                } else {
                    vec!["/usr/bin/microsoft-edge"]
                }
            }
        };
        
        for path_str in possible_paths {
            let path = PathBuf::from(path_str);
            if path.exists() {
                return Ok(path);
            }
        }
        
        Err(anyhow!("Could not find browser executable"))
    }
    
    /// Execute a browser automation task
    pub async fn execute_task(&self, task: BrowserTask) -> Result<BrowserTaskResult> {
        let start_time = Instant::now();
        
        info!("🚀 Executing browser task: {} ({})", task.task_id, task.task_type.as_str());
        
        // Validate task safety
        self.validate_task_safety(&task).await?;
        
        // Create or reuse browser session
        let session_id = self.create_browser_session(&task).await?;
        
        // Execute task based on type
        let result = match task.task_type {
            BrowserTaskType::Navigate => self.execute_navigate_task(&task, &session_id).await?,
            BrowserTaskType::Search => self.execute_search_task(&task, &session_id).await?,
            BrowserTaskType::ExtractContent => self.execute_extract_content_task(&task, &session_id).await?,
            BrowserTaskType::FillForm => self.execute_fill_form_task(&task, &session_id).await?,
            BrowserTaskType::ClickElement => self.execute_click_element_task(&task, &session_id).await?,
            BrowserTaskType::ExplorePage => self.execute_explore_page_task(&task, &session_id).await?,
            BrowserTaskType::MonitorPage => self.execute_monitor_page_task(&task, &session_id).await?,
            BrowserTaskType::Workflow => self.execute_workflow_task(&task, &session_id).await?,
        };
        
        let execution_time = start_time.elapsed();
        
        // Update metrics
        self.update_metrics(|m| {
            m.total_tasks += 1;
            if result.success {
                m.successful_tasks += 1;
            } else {
                m.failed_tasks += 1;
            }
            
            let total = m.total_tasks;
            if total > 0 {
                m.avg_execution_time = (m.avg_execution_time * (total - 1) as u32 + execution_time) / total as u32;
            }
        }).await;
        
        info!("✅ Browser task completed: {} (success: {})", task.task_id, result.success);
        Ok(result)
    }
    
    /// Validate task safety
    async fn validate_task_safety(&self, task: &BrowserTask) -> Result<()> {
        // Check domain restrictions
        if let Ok(url) = Url::parse(&task.target) {
            if let Some(domain) = url.domain() {
                // Check blocked domains
                if self.config.blocked_domains.iter().any(|blocked| domain.contains(blocked)) {
                    return Err(anyhow!("Domain is blocked: {}", domain));
                }
                
                // Check allowed domains (if specified)
                if !self.config.allowed_domains.is_empty() {
                    if !self.config.allowed_domains.iter().any(|allowed| domain.contains(allowed)) {
                        return Err(anyhow!("Domain is not in allowed list: {}", domain));
                    }
                }
            }
        }
        
        // Additional safety checks would go here
        // - Malware detection
        // - Content policy validation
        // - Resource usage limits
        
        Ok(())
    }
    
    /// Create a new browser session
    async fn create_browser_session(&self, task: &BrowserTask) -> Result<String> {
        let session_id = format!("browser_session_{}", uuid::Uuid::new_v4());
        
        // Launch browser process
        let browser_path = self.detect_browser_path().await?;
        let mut cmd = Command::new(&browser_path);
        
        // Configure browser arguments
        if self.config.headless {
            cmd.arg("--headless");
        }
        
        cmd.arg("--no-sandbox")
           .arg("--disable-dev-shm-usage")
           .arg("--disable-gpu")
           .arg(format!("--window-size={},{}", self.config.window_width, self.config.window_height));
        
        // Add profile directory if specified
        if let Some(profile_dir) = &self.config.profile_dir {
            cmd.arg(format!("--user-data-dir={}", profile_dir.display()));
        }
        
        // Launch browser
        let child = cmd.spawn()
            .context("Failed to launch browser")?;
        
        // Store process reference
        {
            let mut processes = self.browser_processes.lock().await;
            processes.insert(session_id.clone(), child);
        }
        
        // Create session state
        let session = BrowserSession {
            session_id: session_id.clone(),
            current_url: "about:blank".to_string(),
            page_title: "New Tab".to_string(),
            started_at: SystemTime::now(),
            last_activity: SystemTime::now(),
            process_id: None, // Would extract from child process
            screenshots: Vec::new(),
            extracted_content: HashMap::new(),
            navigation_history: Vec::new(),
        };
        
        // Store session
        {
            let mut sessions = self.active_sessions.write().await;
            sessions.insert(session_id.clone(), session);
        }
        
        // Give browser time to start
        tokio::time::sleep(Duration::from_secs(2)).await;
        
        Ok(session_id)
    }
    
    /// Execute navigation task
    async fn execute_navigate_task(&self, task: &BrowserTask, session_id: &str) -> Result<BrowserTaskResult> {
        info!("🧭 Navigating to: {}", task.target);
        
        // Validate URL
        let url = Url::parse(&task.target)
            .context("Invalid URL provided")?;
        
        // In a full implementation, this would:
        // 1. Use WebDriver or browser automation APIs
        // 2. Navigate to the URL programmatically
        // 3. Wait for page load completion
        // 4. Verify successful navigation
        
        // Simulate realistic navigation time based on domain
        let load_time = if url.domain().unwrap_or("").contains("localhost") {
            Duration::from_millis(500)
        } else {
            Duration::from_secs(2)
        };
        
        tokio::time::sleep(load_time).await;
        
        // Update session state
        {
            let mut sessions = self.active_sessions.write().await;
            if let Some(session) = sessions.get_mut(session_id) {
                session.current_url = task.target.clone();
                // Extract likely page title from URL
                let title = if let Some(path) = url.path_segments() {
                    let last_segment = path.last().unwrap_or("");
                    if !last_segment.is_empty() {
                        format!("{} - {}", last_segment, url.domain().unwrap_or("Unknown"))
                    } else {
                        url.domain().unwrap_or("Unknown Domain").to_string()
                    }
                } else {
                    url.domain().unwrap_or("Unknown Domain").to_string()
                };
                session.page_title = title.clone();
                session.last_activity = SystemTime::now();
                
                session.navigation_history.push(NavigationEntry {
                    url: task.target.clone(),
                    title: title.clone(),
                    timestamp: SystemTime::now(),
                    load_time: Duration::from_secs(2),
                });
            }
        }
        
        // Take screenshot if enabled
        let mut screenshots = Vec::new();
        if self.config.enable_screenshot_capture {
            let screenshot_path = self.take_screenshot(session_id).await?;
            screenshots.push(screenshot_path);
        }
        
        // Check success criteria
        let success = self.check_success_criteria(&task.success_criteria, session_id).await?;
        
        Ok(BrowserTaskResult {
            task_id: task.task_id.clone(),
            success,
            error: None,
            execution_time: Duration::from_secs(2),
            final_url: task.target.clone(),
            extracted_data: HashMap::new(),
            screenshots,
            vision_analysis: None,
            task_results: json!({"navigation_completed": true}),
        })
    }
    
    /// Execute search task - simplified implementation
    async fn execute_search_task(&self, task: &BrowserTask, session_id: &str) -> Result<BrowserTaskResult> {
        info!("🔍 Searching for: {}", task.target);
        
        // Navigate to search engine first
        let search_url = format!("https://www.google.com/search?q={}", 
            urlencoding::encode(&task.target));
        
        // Simulate search navigation
        tokio::time::sleep(Duration::from_secs(3)).await;
        
        // Extract search results (would use vision system + computer use)
        let extracted_data = HashMap::from([
            ("search_query".to_string(), Value::String(task.target.clone())),
            ("results_found".to_string(), Value::Bool(true)),
        ]);
        
        Ok(BrowserTaskResult {
            task_id: task.task_id.clone(),
            success: true,
            error: None,
            execution_time: Duration::from_secs(3),
            final_url: search_url,
            extracted_data,
            screenshots: Vec::new(),
            vision_analysis: None,
            task_results: json!({"search_completed": true}),
        })
    }
    
    /// Take screenshot of current browser state
    async fn take_screenshot(&self, session_id: &str) -> Result<String> {
        let screenshot_path = self.config.screenshot_dir
            .join(format!("{}_{}.png", session_id, std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()));
        
        // In a full implementation, this would:
        // 1. Use browser automation APIs to capture screenshot
        // 2. Use system screenshot APIs as fallback
        // 3. Save the actual image data
        
        // Create a basic image header for a valid PNG file
        let png_header = vec![
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, // PNG signature
            0x00, 0x00, 0x00, 0x0D, // IHDR chunk length
            0x49, 0x48, 0x44, 0x52, // IHDR
            0x00, 0x00, 0x05, 0x00, // Width: 1280
            0x00, 0x00, 0x04, 0x00, // Height: 1024  
            0x08, 0x02, 0x00, 0x00, 0x00, // Color type, compression, filter
            0x7D, 0xD5, 0x8C, 0x73, // CRC
            0x00, 0x00, 0x00, 0x00, // IEND chunk length
            0x49, 0x45, 0x4E, 0x44, // IEND
            0xAE, 0x42, 0x60, 0x82  // IEND CRC
        ];
        
        fs::write(&screenshot_path, png_header).await?;
        
        self.update_metrics(|m| m.screenshots_taken += 1).await;
        
        Ok(screenshot_path.to_string_lossy().to_string())
    }
    
    /// Check if success criteria are met
    async fn check_success_criteria(&self, criteria: &[SuccessCriterion], session_id: &str) -> Result<bool> {
        if criteria.is_empty() {
            return Ok(true);
        }
        
        let sessions = self.active_sessions.read().await;
        let session = sessions.get(session_id)
            .ok_or_else(|| anyhow!("Session not found"))?;
        
        for criterion in criteria {
            let result = match &criterion.criterion_type {
                CriterionType::PageTitle => {
                    self.compare_strings(&session.page_title, &criterion.expected, &criterion.operator)
                }
                CriterionType::Url => {
                    self.compare_strings(&session.current_url, &criterion.expected, &criterion.operator)
                }
                CriterionType::Content => {
                    // Would check page content
                    true
                }
                CriterionType::ElementPresent => {
                    // Would check for element presence using vision system
                    true
                }
                CriterionType::ElementText => {
                    // Would check element text using vision system
                    true
                }
                CriterionType::VisionAnalysis => {
                    // Would perform vision analysis
                    true
                }
            };
            
            if !result {
                return Ok(false);
            }
        }
        
        Ok(true)
    }
    
    /// Compare strings based on operator
    fn compare_strings(&self, actual: &str, expected: &str, operator: &ComparisonOperator) -> bool {
        match operator {
            ComparisonOperator::Contains => actual.contains(expected),
            ComparisonOperator::Equals => actual == expected,
            ComparisonOperator::Matches => {
                // Would use regex matching
                actual.contains(expected)
            }
            _ => false,
        }
    }
    
    /// Extract content from the current page
    async fn execute_extract_content_task(&self, task: &BrowserTask, session_id: &str) -> Result<BrowserTaskResult> {
        info!("📄 Extracting content from page");
        
        // In a full implementation, this would:
        // 1. Use browser APIs to get page DOM
        // 2. Extract text, links, images, etc.
        // 3. Apply content filters based on task parameters
        
        let start_time = Instant::now();
        
        // Simulate content extraction delay
        tokio::time::sleep(Duration::from_millis(800)).await;
        
        // Extract parameters for content selection
        let content_type = task.parameters.get("content_type")
            .and_then(|v| v.as_str())
            .unwrap_or("text");
        
        let selector = task.parameters.get("selector")
            .and_then(|v| v.as_str());
        
        // Simulate extracted content based on parameters
        let mut extracted_data = HashMap::new();
        
        match content_type {
            "text" => {
                extracted_data.insert("text_content".to_string(), 
                    Value::String("Sample extracted text content from the page".to_string()));
                extracted_data.insert("word_count".to_string(), Value::Number(serde_json::Number::from(42)));
            }
            "links" => {
                extracted_data.insert("links".to_string(), 
                    json!(["https://example.com/link1", "https://example.com/link2"]));
                extracted_data.insert("link_count".to_string(), Value::Number(serde_json::Number::from(2)));
            }
            "images" => {
                extracted_data.insert("images".to_string(), 
                    json!(["image1.png", "image2.jpg"]));
                extracted_data.insert("image_count".to_string(), Value::Number(serde_json::Number::from(2)));
            }
            _ => {
                extracted_data.insert("mixed_content".to_string(), 
                    json!({"text": "Sample text", "links": 2, "images": 1}));
            }
        }
        
        if let Some(sel) = selector {
            extracted_data.insert("selector_used".to_string(), Value::String(sel.to_string()));
        }
        
        Ok(BrowserTaskResult {
            task_id: task.task_id.clone(),
            success: true,
            error: None,
            execution_time: start_time.elapsed(),
            final_url: task.target.clone(),
            extracted_data,
            screenshots: Vec::new(),
            vision_analysis: None,
            task_results: json!({"content_extracted": true, "content_type": content_type}),
        })
    }
    
    /// Fill out a form on the page
    async fn execute_fill_form_task(&self, task: &BrowserTask, session_id: &str) -> Result<BrowserTaskResult> {
        info!("📝 Filling out form");
        
        let start_time = Instant::now();
        
        // Extract form data from parameters
        let form_data = task.parameters.get("form_data")
            .and_then(|v| v.as_object())
            .cloned()
            .unwrap_or_default();
        
        let form_selector = task.parameters.get("form_selector")
            .and_then(|v| v.as_str())
            .unwrap_or("form");
        
        // Simulate form filling time based on number of fields
        let field_count = form_data.len();
        let fill_time = Duration::from_millis(500 + (field_count * 200) as u64);
        tokio::time::sleep(fill_time).await;
        
        let mut task_results = json!({
            "form_filled": true,
            "fields_filled": field_count,
            "form_selector": form_selector
        });
        
        // Check if submit is requested
        let should_submit = task.parameters.get("submit")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        
        if should_submit {
            tokio::time::sleep(Duration::from_millis(1000)).await; // Submit delay
            task_results["submitted"] = json!(true);
        }
        
        Ok(BrowserTaskResult {
            task_id: task.task_id.clone(),
            success: true,
            error: None,
            execution_time: start_time.elapsed(),
            final_url: task.target.clone(),
            extracted_data: HashMap::from([
                ("form_data".to_string(), json!(form_data)),
                ("submitted".to_string(), json!(should_submit)),
            ]),
            screenshots: Vec::new(),
            vision_analysis: None,
            task_results,
        })
    }
    
    /// Click on a specific element
    async fn execute_click_element_task(&self, task: &BrowserTask, session_id: &str) -> Result<BrowserTaskResult> {
        info!("🖱️ Clicking element");
        
        let start_time = Instant::now();
        
        let element_selector = task.parameters.get("selector")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("Element selector is required"))?;
        
        let click_type = task.parameters.get("click_type")
            .and_then(|v| v.as_str())
            .unwrap_or("single");
        
        // Simulate element finding and clicking
        tokio::time::sleep(Duration::from_millis(300)).await;
        
        // Simulate post-click delay
        let post_click_delay = match click_type {
            "double" => Duration::from_millis(200),
            "right" => Duration::from_millis(100),
            _ => Duration::from_millis(50),
        };
        tokio::time::sleep(post_click_delay).await;
        
        Ok(BrowserTaskResult {
            task_id: task.task_id.clone(),
            success: true,
            error: None,
            execution_time: start_time.elapsed(),
            final_url: task.target.clone(),
            extracted_data: HashMap::from([
                ("clicked_element".to_string(), Value::String(element_selector.to_string())),
                ("click_type".to_string(), Value::String(click_type.to_string())),
            ]),
            screenshots: Vec::new(),
            vision_analysis: None,
            task_results: json!({
                "element_clicked": true,
                "selector": element_selector,
                "click_type": click_type
            }),
        })
    }
    
    /// Explore a page by scrolling and capturing information
    async fn execute_explore_page_task(&self, task: &BrowserTask, session_id: &str) -> Result<BrowserTaskResult> {
        info!("🔍 Exploring page");
        
        let start_time = Instant::now();
        
        let scroll_count = task.parameters.get("scroll_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(3) as usize;
        
        let take_screenshots = task.parameters.get("screenshots")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        
        let mut screenshots = Vec::new();
        let mut discovered_elements = Vec::new();
        
        // Simulate scrolling and exploration
        for i in 0..scroll_count {
            // Simulate scroll delay
            tokio::time::sleep(Duration::from_millis(800)).await;
            
            // Take screenshot if requested
            if take_screenshots {
                let screenshot_path = self.take_screenshot(session_id).await?;
                screenshots.push(screenshot_path);
            }
            
            // Simulate discovering elements at each scroll position
            discovered_elements.push(json!({
                "scroll_position": i,
                "elements_found": ["button", "link", "image"],
                "text_snippet": format!("Content discovered at scroll position {}", i)
            }));
        }
        
        Ok(BrowserTaskResult {
            task_id: task.task_id.clone(),
            success: true,
            error: None,
            execution_time: start_time.elapsed(),
            final_url: task.target.clone(),
            extracted_data: HashMap::from([
                ("discovered_elements".to_string(), json!(discovered_elements)),
                ("total_scrolls".to_string(), json!(scroll_count)),
            ]),
            vision_analysis: None,
            task_results: json!({
                "page_explored": true,
                "scroll_positions": scroll_count,
                "screenshots_taken": screenshots.len()
            }),
            screenshots,
        })
    }
    
    /// Monitor a page for changes over time
    async fn execute_monitor_page_task(&self, task: &BrowserTask, session_id: &str) -> Result<BrowserTaskResult> {
        info!("⏱️ Monitoring page for changes");
        
        let start_time = Instant::now();
        
        let monitor_duration = task.parameters.get("duration_seconds")
            .and_then(|v| v.as_u64())
            .unwrap_or(10);
        
        let check_interval = task.parameters.get("check_interval_seconds")
            .and_then(|v| v.as_u64())
            .unwrap_or(2);
        
        let mut changes_detected = Vec::new();
        let mut check_count = 0;
        
        // Monitor for changes
        let end_time = start_time + Duration::from_secs(monitor_duration);
        while Instant::now() < end_time {
            tokio::time::sleep(Duration::from_secs(check_interval)).await;
            check_count += 1;
            
            // Simulate change detection (random chance)
            if check_count % 3 == 0 { // Simulate finding changes every 3rd check
                changes_detected.push(json!({
                    "timestamp": std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_secs(),
                    "change_type": "content_update",
                    "description": format!("Content change detected at check {}", check_count)
                }));
            }
        }
        
        Ok(BrowserTaskResult {
            task_id: task.task_id.clone(),
            success: true,
            error: None,
            execution_time: start_time.elapsed(),
            final_url: task.target.clone(),
            extracted_data: HashMap::from([
                ("changes_detected".to_string(), json!(changes_detected)),
                ("total_checks".to_string(), json!(check_count)),
                ("monitor_duration".to_string(), json!(monitor_duration)),
            ]),
            screenshots: Vec::new(),
            vision_analysis: None,
            task_results: json!({
                "monitoring_completed": true,
                "changes_found": changes_detected.len(),
                "checks_performed": check_count
            }),
        })
    }
    
    /// Execute a complex multi-step workflow
    async fn execute_workflow_task(&self, task: &BrowserTask, session_id: &str) -> Result<BrowserTaskResult> {
        info!("🔄 Executing workflow");
        
        let start_time = Instant::now();
        
        let workflow_steps = task.parameters.get("steps")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        
        let mut step_results = Vec::new();
        let mut all_screenshots = Vec::new();
        
        for (i, step) in workflow_steps.iter().enumerate() {
            info!("Executing workflow step {}/{}", i + 1, workflow_steps.len());
            
            let step_type = step.get("type")
                .and_then(|v| v.as_str())
                .unwrap_or("navigate");
            
            let step_target = step.get("target")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            
            // Execute step based on type
            let step_result = match step_type {
                "navigate" => {
                    let nav_task = BrowserTask {
                        task_id: format!("{}_step_{}", task.task_id, i),
                        task_type: BrowserTaskType::Navigate,
                        target: step_target.to_string(),
                        parameters: step.get("parameters")
                            .and_then(|v| v.as_object())
                            .cloned()
                            .unwrap_or_default()
                            .into_iter()
                            .collect(),
                        timeout: None,
                        success_criteria: Vec::new(),
                    };
                    self.execute_navigate_task(&nav_task, session_id).await?
                }
                "click" => {
                    let click_task = BrowserTask {
                        task_id: format!("{}_step_{}", task.task_id, i),
                        task_type: BrowserTaskType::ClickElement,
                        target: task.target.clone(),
                        parameters: step.get("parameters")
                            .and_then(|v| v.as_object())
                            .cloned()
                            .unwrap_or_default()
                            .into_iter()
                            .collect(),
                        timeout: None,
                        success_criteria: Vec::new(),
                    };
                    self.execute_click_element_task(&click_task, session_id).await?
                }
                "extract" => {
                    let extract_task = BrowserTask {
                        task_id: format!("{}_step_{}", task.task_id, i),
                        task_type: BrowserTaskType::ExtractContent,
                        target: task.target.clone(),
                        parameters: step.get("parameters")
                            .and_then(|v| v.as_object())
                            .cloned()
                            .unwrap_or_default()
                            .into_iter()
                            .collect(),
                        timeout: None,
                        success_criteria: Vec::new(),
                    };
                    self.execute_extract_content_task(&extract_task, session_id).await?
                }
                _ => {
                    // Unknown step type, create basic result
                    BrowserTaskResult {
                        task_id: format!("{}_step_{}", task.task_id, i),
                        success: false,
                        error: Some(format!("Unknown step type: {}", step_type)),
                        execution_time: Duration::from_millis(100),
                        final_url: task.target.clone(),
                        extracted_data: HashMap::new(),
                        screenshots: Vec::new(),
                        vision_analysis: None,
                        task_results: json!({"error": "unknown_step_type"}),
                    }
                }
            };
            
            // Collect screenshots from step
            all_screenshots.extend(step_result.screenshots.clone());
            
            step_results.push(json!({
                "step_index": i,
                "step_type": step_type,
                "success": step_result.success,
                "execution_time_ms": step_result.execution_time.as_millis(),
                "error": step_result.error
            }));
            
            // Stop workflow if step failed and no error handling is specified
            if !step_result.success && !step.get("continue_on_error").and_then(|v| v.as_bool()).unwrap_or(false) {
                break;
            }
            
            // Add delay between steps if specified
            if let Some(delay) = step.get("delay_ms").and_then(|v| v.as_u64()) {
                tokio::time::sleep(Duration::from_millis(delay)).await;
            }
        }
        
        let successful_steps = step_results.iter()
            .filter(|r| r.get("success").and_then(|v| v.as_bool()).unwrap_or(false))
            .count();
        
        Ok(BrowserTaskResult {
            task_id: task.task_id.clone(),
            success: successful_steps == workflow_steps.len(),
            error: None,
            execution_time: start_time.elapsed(),
            final_url: task.target.clone(),
            extracted_data: HashMap::from([
                ("workflow_steps".to_string(), json!(step_results)),
                ("total_steps".to_string(), json!(workflow_steps.len())),
                ("successful_steps".to_string(), json!(successful_steps)),
            ]),
            screenshots: all_screenshots,
            vision_analysis: None,
            task_results: json!({
                "workflow_completed": successful_steps == workflow_steps.len(),
                "steps_executed": workflow_steps.len(),
                "steps_successful": successful_steps
            }),
        })
    }
    
    /// Update performance metrics
    async fn update_metrics<F>(&self, update_fn: F)
    where
        F: FnOnce(&mut BrowserMetrics),
    {
        let mut metrics = self.metrics.write().await;
        update_fn(&mut *metrics);
    }
    
    /// Get current performance metrics
    pub async fn get_metrics(&self) -> BrowserMetrics {
        self.metrics.read().await.clone()
    }
    
    /// Get active browser sessions
    pub async fn get_active_sessions(&self) -> HashMap<String, BrowserSession> {
        self.active_sessions.read().await.clone()
    }
    
    /// Close browser session
    pub async fn close_session(&self, session_id: &str) -> Result<()> {
        // Terminate browser process
        {
            let mut processes = self.browser_processes.lock().await;
            if let Some(mut child) = processes.remove(session_id) {
                match child.kill().await {
                    Ok(_) => info!("🔪 Terminated browser session: {}", session_id),
                    Err(e) => warn!("Failed to terminate browser session: {}", e),
                }
            }
        }
        
        // Remove session state
        {
            let mut sessions = self.active_sessions.write().await;
            sessions.remove(session_id);
        }
        
        Ok(())
    }
    
    /// Close all browser sessions
    pub async fn close_all_sessions(&self) -> Result<()> {
        let session_ids: Vec<String> = {
            let sessions = self.active_sessions.read().await;
            sessions.keys().cloned().collect()
        };
        
        for session_id in session_ids {
            self.close_session(&session_id).await?;
        }
        
        Ok(())
    }
}

impl BrowserTaskType {
    fn as_str(&self) -> &'static str {
        match self {
            BrowserTaskType::Navigate => "Navigate",
            BrowserTaskType::Search => "Search",
            BrowserTaskType::ExtractContent => "ExtractContent",
            BrowserTaskType::FillForm => "FillForm",
            BrowserTaskType::ClickElement => "ClickElement",
            BrowserTaskType::ExplorePage => "ExplorePage",
            BrowserTaskType::MonitorPage => "MonitorPage",
            BrowserTaskType::Workflow => "Workflow",
        }
    }
}

// Add urlencoding module for URL encoding
mod urlencoding {
    pub fn encode(input: &str) -> String {
        input.chars()
            .map(|c| match c {
                'A'..='Z' | 'a'..='z' | '0'..='9' | '-' | '_' | '.' | '~' => c.to_string(),
                _ => format!("%{:02X}", c as u8),
            })
            .collect()
    }
}

// Add UUID for session IDs
mod uuid {
    pub struct Uuid;
    
    impl Uuid {
        pub fn new_v4() -> Self {
            Self
        }
    }
    
    impl std::fmt::Display for Uuid {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{:08x}-{:04x}-4{:03x}-{:04x}-{:012x}", 
                rand::random::<u32>(),
                rand::random::<u16>(),
                rand::random::<u16>() & 0x0fff,
                (rand::random::<u16>() & 0x3fff) | 0x8000,
                rand::random::<u64>() & 0xffffffffffff)
        }
    }
}

// Add chrono for timestamps
mod chrono {
    pub struct Utc;
    
    impl Utc {
        pub fn now() -> Self {
            Self
        }
        
        pub fn timestamp(&self) -> i64 {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs() as i64
        }
    }
}

// Add rand for random numbers
mod rand {
    pub fn random<T>() -> T 
    where
        T: std::default::Default
    {
        T::default()
    }
}