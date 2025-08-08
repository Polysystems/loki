use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use anyhow::{Result, anyhow};
use serde::{Serialize, Deserialize};
use serde_json::Value;
use tracing::{info, warn};
use tokio::process::Command;

use crate::cognitive::CognitiveSystem;
use crate::memory::{CognitiveMemory, MemoryMetadata};
use super::vision_system::{VisionSystem, VisualAnalysis};

/// Computer use system for screen interaction and automation
pub struct ComputerUseSystem {
    config: ComputerUseConfig,
    cognitive_system: Arc<CognitiveSystem>,
    memory: Arc<CognitiveMemory>,
    vision_system: Arc<VisionSystem>,

    /// Creative media manager for content generation
    creative_media_manager: Option<Arc<super::creative_media::CreativeMediaManager>>,

    /// Blender integration for 3D modeling
    blender_integration: Option<Arc<super::blender_integration::BlenderIntegration>>,

    /// Current screen state
    current_screen: Arc<tokio::sync::RwLock<Option<ScreenState>>>,

    /// Interaction history
    interaction_history: Arc<tokio::sync::RwLock<Vec<Interaction>>>,
}

/// Computer use configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputerUseConfig {
    /// Screenshot settings
    pub screenshot_interval_seconds: u64,
    pub screenshot_directory: PathBuf,
    pub max_screenshots: usize,

    /// Interaction settings
    pub enable_mouse_control: bool,
    pub enable_keyboard_control: bool,
    pub safety_mode: bool,
    pub max_interactions_per_minute: u32,

    /// Application automation
    pub enable_blender_automation: bool,
    pub enable_browser_automation: bool,
    pub enable_terminal_automation: bool,

    /// Cognitive integration
    pub cognitive_awareness_level: f32,
    pub store_interactions_in_memory: bool,
    pub analyze_screenshots: bool,
}

/// Screen state information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScreenState {
    pub screenshot_path: String,
    pub analysis: Option<VisualAnalysis>,
    pub ui_elements: Vec<UIElement>,
    pub active_applications: Vec<String>,
    pub screen_resolution: (u32, u32),
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// UI element detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UIElement {
    pub element_type: UIElementType,
    pub bounds: BoundingRect,
    pub text: Option<String>,
    pub clickable: bool,
    pub confidence: f32,
}

/// UI element types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UIElementType {
    Button,
    TextField,
    Label,
    Menu,
    Window,
    Tab,
    Icon,
    Checkbox,
    Slider,
    ScrollBar,
    Other(String),
}

/// Bounding rectangle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundingRect {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
}

/// Interaction record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Interaction {
    pub id: String,
    pub interaction_type: InteractionType,
    pub target: Option<String>,
    pub coordinates: Option<(i32, i32)>,
    pub success: bool,
    pub duration_ms: u64,
    pub screenshot_before: Option<String>,
    pub screenshot_after: Option<String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Interaction types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InteractionType {
    Click { button: MouseButton },
    DoubleClick { button: MouseButton },
    RightClick,
    Drag { from: (i32, i32), to: (i32, i32) },
    Type { text: String },
    KeyPress { key: String },
    KeyCombo { keys: Vec<String> },
    Scroll { direction: ScrollDirection, amount: i32 },
    Screenshot,
    ApplicationLaunch { app_name: String },
    WindowManagement { action: WindowAction },
}

/// Mouse buttons
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MouseButton {
    Left,
    Right,
    Middle,
}

/// Scroll direction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScrollDirection {
    Up,
    Down,
    Left,
    Right,
}

/// Window management actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WindowAction {
    Minimize,
    Maximize,
    Close,
    Restore,
    Move { x: i32, y: i32 },
    Resize { width: i32, height: i32 },
}

/// Creative workflow integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreativeWorkflow {
    pub id: String,
    pub name: String,
    pub steps: Vec<WorkflowStep>,
    pub current_step: usize,
    pub target_application: String,
}

/// Workflow step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowStep {
    pub action: String,
    pub target: Option<String>,
    pub parameters: HashMap<String, Value>,
    pub validation: Option<String>,
}

impl Default for ComputerUseConfig {
    fn default() -> Self {
        Self {
            screenshot_interval_seconds: 5,
            screenshot_directory: PathBuf::from("./screenshots"),
            max_screenshots: 1000,
            enable_mouse_control: true,
            enable_keyboard_control: true,
            safety_mode: true,
            max_interactions_per_minute: 60,
            enable_blender_automation: true,
            enable_browser_automation: true,
            enable_terminal_automation: false, // Disabled by default for safety
            cognitive_awareness_level: 0.8,
            store_interactions_in_memory: true,
            analyze_screenshots: true,
        }
    }
}

impl ComputerUseSystem {
    /// Create new computer use system
    pub async fn new(
        config: ComputerUseConfig,
        cognitive_system: Arc<CognitiveSystem>,
        memory: Arc<CognitiveMemory>,
        vision_system: Arc<VisionSystem>,
    ) -> Result<Self> {
        // Create screenshot directory
        tokio::fs::create_dir_all(&config.screenshot_directory).await?;

        Ok(Self {
            config,
            cognitive_system,
            memory,
            vision_system,
            creative_media_manager: None, // Will be initialized when needed
            blender_integration: None,    // Will be initialized when needed
            current_screen: Arc::new(tokio::sync::RwLock::new(None)),
            interaction_history: Arc::new(tokio::sync::RwLock::new(Vec::new())),
        })
    }

    /// Take a screenshot and analyze it
    pub async fn capture_and_analyze_screen(&self) -> Result<ScreenState> {
        let start_time = Instant::now();
        info!("📸 Capturing and analyzing screen");

        // Take screenshot
        let screenshot_path = self.take_screenshot().await?;

        // Get screen resolution
        let resolution = self.get_screen_resolution().await?;

        // Analyze screenshot with vision system if enabled
        let analysis = if self.config.analyze_screenshots {
            match self.vision_system.analyze_image(&screenshot_path).await {
                Ok(analysis) => Some(analysis),
                Err(e) => {
                    warn!("Screenshot analysis failed: {}", e);
                    None
                }
            }
        } else {
            None
        };

        // Detect UI elements
        let ui_elements = self.detect_ui_elements(&screenshot_path, &analysis).await?;

        // Get active applications
        let active_apps = self.get_active_applications().await?;

        let screen_state = ScreenState {
            screenshot_path,
            analysis,
            ui_elements,
            active_applications: active_apps,
            screen_resolution: resolution,
            timestamp: chrono::Utc::now(),
        };

        // Update current screen state
        *self.current_screen.write().await = Some(screen_state.clone());

        // Store in memory if enabled
        if self.config.store_interactions_in_memory && self.config.cognitive_awareness_level > 0.5 {
            self.store_screen_state_in_memory(&screen_state).await?;
        }

        info!("✅ Screen analysis complete in {:?}: {} UI elements detected",
              start_time.elapsed(), screen_state.ui_elements.len());

        Ok(screen_state)
    }

    /// Take a screenshot using system commands
    async fn take_screenshot(&self) -> Result<String> {
        let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S%.3f");
        let filename = format!("screenshot_{}.png", timestamp);
        let filepath = self.config.screenshot_directory.join(&filename);

        #[cfg(target_os = "macos")]
        {
            let output = Command::new("screencapture")
                .arg("-x") // No sound
                .arg(&filepath)
                .output()
                .await?;

            if !output.status.success() {
                return Err(anyhow!("Screenshot failed: {}", String::from_utf8_lossy(&output.stderr)));
            }
        }

        #[cfg(target_os = "linux")]
        {
            let output = Command::new("gnome-screenshot")
                .arg("-f")
                .arg(&filepath)
                .output()
                .await?;

            if !output.status.success() {
                return Err(anyhow!("Screenshot failed: {}", String::from_utf8_lossy(&output.stderr)));
            }
        }

        #[cfg(target_os = "windows")]
        {
            // Use PowerShell for Windows screenshots
            let ps_command = format!(
                "Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.Screen]::PrimaryScreen.Bounds | Set-Variable bounds; $bitmap = New-Object System.Drawing.Bitmap $bounds.Width,$bounds.Height; $graphics = [System.Drawing.Graphics]::FromImage($bitmap); $graphics.CopyFromScreen($bounds.Location, [System.Drawing.Point]::Empty, $bounds.Size); $bitmap.Save('{}'); $graphics.Dispose(); $bitmap.Dispose()",
                filepath.to_string_lossy()
            );

            let output = Command::new("powershell")
                .arg("-Command")
                .arg(&ps_command)
                .output()
                .await?;

            if !output.status.success() {
                return Err(anyhow!("Screenshot failed: {}", String::from_utf8_lossy(&output.stderr)));
            }
        }

        // Clean up old screenshots if we exceed the limit
        self.cleanup_old_screenshots().await?;

        Ok(filepath.to_string_lossy().to_string())
    }

    /// Get screen resolution
    async fn get_screen_resolution(&self) -> Result<(u32, u32)> {
        #[cfg(target_os = "macos")]
        {
            let output = Command::new("system_profiler")
                .arg("SPDisplaysDataType")
                .output()
                .await?;

            // Parse resolution from output (simplified)
            // In practice, you'd parse the actual output
            Ok((1920, 1080)) // Default fallback
        }

        #[cfg(not(target_os = "macos"))]
        {
            // Default resolution for other platforms
            Ok((1920, 1080))
        }
    }

    /// Detect UI elements in screenshot
    async fn detect_ui_elements(&self, screenshot_path: &str, analysis: &Option<VisualAnalysis>) -> Result<Vec<UIElement>> {
        let mut ui_elements = Vec::new();

        // If we have vision analysis, extract UI elements from it
        if let Some(analysis) = analysis {
            for object in &analysis.objects {
                // Convert detected objects to UI elements
                let element_type = match object.name.as_str() {
                    "button" => UIElementType::Button,
                    "text" | "label" => UIElementType::Label,
                    "input" | "field" => UIElementType::TextField,
                    "menu" => UIElementType::Menu,
                    "window" => UIElementType::Window,
                    _ => UIElementType::Other(object.name.clone()),
                };

                ui_elements.push(UIElement {
                    element_type: element_type.clone(),
                    bounds: BoundingRect {
                        x: (object.bounding_box.x * 1920.0) as i32, // Scale to screen resolution
                        y: (object.bounding_box.y * 1080.0) as i32,
                        width: (object.bounding_box.width * 1920.0) as i32,
                        height: (object.bounding_box.height * 1080.0) as i32,
                    },
                    text: None,
                    clickable: matches!(element_type, UIElementType::Button | UIElementType::Menu),
                    confidence: object.confidence,
                });
            }
        }

        Ok(ui_elements)
    }

    /// Get list of active applications
    async fn get_active_applications(&self) -> Result<Vec<String>> {
        let mut apps = Vec::new();

        #[cfg(target_os = "macos")]
        {
            let output = Command::new("osascript")
                .arg("-e")
                .arg("tell application \"System Events\" to get name of every process whose background only is false")
                .output()
                .await?;

            if output.status.success() {
                let apps_str = String::from_utf8_lossy(&output.stdout);
                apps = apps_str.split(", ").map(|s| s.trim().to_string()).collect();
            }
        }

        #[cfg(target_os = "linux")]
        {
            let output = Command::new("ps")
                .arg("-eo")
                .arg("comm")
                .output()
                .await?;

            if output.status.success() {
                let processes = String::from_utf8_lossy(&output.stdout);
                apps = processes.lines().skip(1).map(|s| s.trim().to_string()).collect();
            }
        }

        Ok(apps)
    }

    /// Initialize creative media manager if not already initialized
    async fn ensure_creative_media_manager(&self) -> Result<Arc<super::creative_media::CreativeMediaManager>> {
        info!("🎨 Creating Creative Media Manager instance");

        let config = super::creative_media::CreativeMediaConfig::default();

        // Create content generator
        let model = self.cognitive_system.orchestrator_model().await?;
        let content_generator = Arc::new(crate::social::ContentGenerator::new(
            model,
            self.memory.clone(),
            None, // creative_media - will be set later
            None, // blender_integration - will be set later
        ).await?);

        let creative_media_manager = Arc::new(super::creative_media::CreativeMediaManager::new(
            config,
            self.cognitive_system.clone(),
            self.memory.clone(),
            content_generator,
        ).await?);

        Ok(creative_media_manager)
    }

    /// Initialize Blender integration if not already initialized
    async fn ensure_blender_integration(&self) -> Result<Arc<super::blender_integration::BlenderIntegration>> {
        info!("🔧 Creating Blender Integration instance");

        let config = super::blender_integration::BlenderConfig::default();

        let blender_integration = Arc::new(super::blender_integration::BlenderIntegration::new(
            config,
            self.cognitive_system.clone(),
            self.memory.clone(),
        ).await?);

        Ok(blender_integration)
    }

    /// Execute an image generation → vision analysis → 3D modeling workflow
    pub async fn execute_creative_workflow(&self, image_prompt: &str) -> Result<CreativeWorkflowResult> {
        info!("🎨 Starting creative workflow: {}", image_prompt);
        let start_time = Instant::now();

        // Step 1: Generate image using creative media system
        info!("🎬 Step 1: Generating image");
        let creative_media = self.ensure_creative_media_manager().await?;

        let image_media = creative_media.generate_media(
            super::creative_media::MediaType::Image {
                style: super::creative_media::ImageStyle::Concept,
                prompt: image_prompt.to_string(),
                reference_image: None,
            },
            Some("Creative workflow generation".to_string()),
        ).await?;

        info!("✅ Generated image: {}", image_media.file_path);

        // Step 2: Analyze image with vision system
        info!("👁️ Step 2: Analyzing image with vision");
        let vision_analysis = self.vision_system.analyze_image(&image_media.file_path).await?;

        info!("✅ Vision analysis complete: {} objects, {} materials",
              vision_analysis.objects.len(), vision_analysis.materials.len());

        // Step 3: Create 3D model in Blender based on analysis
        info!("🏗️ Step 3: Creating 3D model in Blender");
        let blender_result = self.create_blender_model_from_analysis(&vision_analysis).await?;

        let workflow_result = CreativeWorkflowResult {
            id: uuid::Uuid::new_v4().to_string(),
            original_prompt: image_prompt.to_string(),
            generated_image: image_media,
            vision_analysis,
            blender_model_path: blender_result.model_path,
            blender_script: blender_result.script_content,
            total_time_ms: start_time.elapsed().as_millis() as u64,
            success: true,
            created_at: chrono::Utc::now(),
        };

        // Store workflow result in memory
        if self.config.store_interactions_in_memory {
            self.store_workflow_result_in_memory(&workflow_result).await?;
        }

        info!("🎉 Creative workflow complete in {:?}!", start_time.elapsed());

        Ok(workflow_result)
    }

    /// Create a 3D model in Blender based on vision analysis
    async fn create_blender_model_from_analysis(&self, analysis: &VisualAnalysis) -> Result<BlenderCreationResult> {
        let blender_integration = self.ensure_blender_integration().await?;

                let execution_start = std::time::Instant::now();

        // Generate Blender script based on vision analysis
        let script_generation_start = std::time::Instant::now();
        let script_content = self.generate_blender_script_from_analysis(analysis).await?;
        let script_generation_time = script_generation_start.elapsed();

        // Use generate_from_description to create the 3D model
        let model_creation_start = std::time::Instant::now();
        let project = blender_integration.generate_from_description(
            &format!("3D model based on vision analysis: {}", analysis.description),
            None // Let it infer the content type
        ).await?;

        let model_creation_time = model_creation_start.elapsed();

        let total_execution_time = execution_start.elapsed();

        tracing::info!(
            "Blender model creation completed: script generation {}ms, model creation {}ms, total {}ms",
            script_generation_time.as_millis(),
            model_creation_time.as_millis(),
            total_execution_time.as_millis()
        );

        Ok(BlenderCreationResult {
            model_path: project.file_path.to_string_lossy().to_string(),
            script_content,
            execution_time_ms: total_execution_time.as_millis() as u64,
        })
    }

    /// Generate Blender Python script from vision analysis
    async fn generate_blender_script_from_analysis(&self, analysis: &VisualAnalysis) -> Result<String> {
        let mut script = String::from("import bpy\nimport bmesh\nfrom mathutils import Vector\n\n");
        script.push_str("# Clear existing mesh objects\n");
        script.push_str("bpy.ops.object.select_all(action='SELECT')\n");
        script.push_str("bpy.ops.object.delete(use_global=False)\n\n");

        // Generate objects based on vision analysis
        for (i, object) in analysis.objects.iter().enumerate() {
            script.push_str(&format!("# Creating object: {}\n", object.name));

            match object.name.as_str() {
                "chair" => {
                    script.push_str(&self.generate_chair_script(i, object)?);
                }
                "table" => {
                    script.push_str(&self.generate_table_script(i, object)?);
                }
                "cup" => {
                    script.push_str(&self.generate_cup_script(i, object)?);
                }
                _ => {
                    script.push_str(&self.generate_generic_object_script(i, object)?);
                }
            }

            script.push_str("\n");
        }

        // Add materials based on analysis
        for material in &analysis.materials {
            script.push_str(&self.generate_material_script(material)?);
        }

        // Add lighting based on spatial information
        if analysis.spatial_info.lighting.shadows {
            script.push_str("# Add lighting\n");
            script.push_str("bpy.ops.object.light_add(type='SUN', location=(2, 2, 5))\n");
            script.push_str("sun = bpy.context.object\n");
            script.push_str("sun.data.energy = 3\n\n");
        }

        script.push_str("# Save the file\n");
        script.push_str("bpy.ops.wm.save_as_mainfile(filepath='vision_analysis_model.blend')\n");

        Ok(script)
    }

    /// Generate chair modeling script
    fn generate_chair_script(&self, index: usize, object: &super::vision_system::DetectedObject) -> Result<String> {
        Ok(format!(r#"
# Chair {}
bpy.ops.mesh.primitive_cube_add(location=(0, {}, 0))
chair = bpy.context.object
chair.name = "Chair_{}"
chair.scale = (0.6, 0.6, 0.8)
"#, index, index * 2, index))
    }

    /// Generate table modeling script
    fn generate_table_script(&self, index: usize, object: &super::vision_system::DetectedObject) -> Result<String> {
        Ok(format!(r#"
# Table {}
bpy.ops.mesh.primitive_cube_add(location=(3, {}, 0))
table = bpy.context.object
table.name = "Table_{}"
table.scale = (1.2, 0.8, 0.1)
table.location.z = 0.75
"#, index, index * 2, index))
    }

    /// Generate cup modeling script
    fn generate_cup_script(&self, index: usize, object: &super::vision_system::DetectedObject) -> Result<String> {
        Ok(format!(r#"
# Cup {}
bpy.ops.mesh.primitive_cylinder_add(location=(6, {}, 0))
cup = bpy.context.object
cup.name = "Cup_{}"
cup.scale = (0.04, 0.04, 0.05)
"#, index, index * 2, index))
    }

    /// Generate generic object script
    fn generate_generic_object_script(&self, index: usize, object: &super::vision_system::DetectedObject) -> Result<String> {
        Ok(format!(r#"
# Generic object: {}
bpy.ops.mesh.primitive_cube_add(location=(9, {}, 0))
obj = bpy.context.object
obj.name = "Object_{}"
"#, object.name, index * 2, index))
    }

    /// Generate material script
    fn generate_material_script(&self, material: &super::vision_system::Material) -> Result<String> {
        let material_name = &material.name;
        let roughness = material.properties.roughness.unwrap_or(0.5);
        let metallic = material.properties.metallic.unwrap_or(0.0);

        Ok(format!(r#"
# Material: {}
mat = bpy.data.materials.new(name="{}")
mat.use_nodes = True
bsdf = mat.node_tree.nodes["Principled BSDF"]
bsdf.inputs[9].default_value = {}  # Roughness
bsdf.inputs[6].default_value = {}  # Metallic
"#, material_name, material_name, roughness, metallic))
    }

    /// Store screen state in cognitive memory
    async fn store_screen_state_in_memory(&self, screen_state: &ScreenState) -> Result<()> {
        let memory_content = format!(
            "Screen State: {} UI elements detected, {} active applications | Resolution: {}x{}",
            screen_state.ui_elements.len(),
            screen_state.active_applications.len(),
            screen_state.screen_resolution.0,
            screen_state.screen_resolution.1
        );

        self.memory.store(
            memory_content,
            vec![screen_state.screenshot_path.clone()],
            MemoryMetadata {
                source: "computer_use".to_string(),
                tags: vec![
                    "screen_state".to_string(),
                    "ui_elements".to_string(),
                    "automation".to_string(),
                ],
                importance: 0.6,
                associations: screen_state.active_applications.iter()
                    .map(|_| crate::memory::MemoryId::new())
                    .collect(),
                context: Some("Screen state capture and UI element analysis".to_string()),
                created_at: chrono::Utc::now(),
                accessed_count: 0,
                last_accessed: None,
                version: 1,
                    category: "tool_usage".to_string(),
                timestamp: chrono::Utc::now(),
                expiration: None,
            },
        ).await?;

        Ok(())
    }

    /// Store workflow result in cognitive memory
    async fn store_workflow_result_in_memory(&self, result: &CreativeWorkflowResult) -> Result<()> {
        let memory_content = format!(
            "Creative Workflow: {} → Image → Vision Analysis → 3D Model | Objects: {}, Time: {}ms",
            result.original_prompt,
            result.vision_analysis.objects.len(),
            result.total_time_ms
        );

        self.memory.store(
            memory_content,
            vec![result.original_prompt.clone()],
            MemoryMetadata {
                source: "creative_workflow".to_string(),
                tags: vec![
                    "creative".to_string(),
                    "workflow".to_string(),
                    "vision".to_string(),
                    "3d_modeling".to_string(),
                ],
                importance: 0.9,
                associations: result.vision_analysis.objects.iter()
                    .map(|_| crate::memory::MemoryId::new())
                    .collect(),
                context: Some("Creative workflow result from image generation to 3D model".to_string()),
                created_at: chrono::Utc::now(),
                accessed_count: 0,
                last_accessed: None,
                version: 1,
                    category: "tool_usage".to_string(),
                timestamp: chrono::Utc::now(),
                expiration: None,
            },
        ).await?;

        Ok(())
    }

    /// Clean up old screenshots
    async fn cleanup_old_screenshots(&self) -> Result<()> {
        let mut entries = tokio::fs::read_dir(&self.config.screenshot_directory).await?;
        let mut files = Vec::new();

        while let Some(entry) = entries.next_entry().await? {
            if let Some(filename) = entry.file_name().to_str() {
                if filename.starts_with("screenshot_") && filename.ends_with(".png") {
                    files.push(entry.path());
                }
            }
        }

        if files.len() > self.config.max_screenshots {
            files.sort();
            let to_remove = files.len() - self.config.max_screenshots;

            for file in files.iter().take(to_remove) {
                if let Err(e) = tokio::fs::remove_file(file).await {
                    warn!("Failed to remove old screenshot {:?}: {}", file, e);
                }
            }
        }

        Ok(())
    }

    /// Execute a computer action (click, type, scroll, etc.)
    pub async fn execute_action(&self, action: InteractionType) -> Result<Interaction> {
        let start_time = Instant::now();
        
        // Check safety mode and rate limiting
        if self.config.safety_mode {
            self.check_action_safety(&action).await?;
        }
        
        // Take screenshot before action
        let screenshot_before = if self.config.analyze_screenshots {
            Some(self.take_screenshot().await?)
        } else {
            None
        };
        
        // Execute the action based on type
        let (success, target, coordinates) = match &action {
            InteractionType::Click { button } => {
                self.execute_click(button).await?
            }
            InteractionType::DoubleClick { button } => {
                self.execute_double_click(button).await?
            }
            InteractionType::RightClick => {
                self.execute_right_click().await?
            }
            InteractionType::Drag { from, to } => {
                self.execute_drag(*from, *to).await?
            }
            InteractionType::Type { text } => {
                self.execute_type(text).await?
            }
            InteractionType::KeyPress { key } => {
                self.execute_key_press(key).await?
            }
            InteractionType::KeyCombo { keys } => {
                self.execute_key_combo(keys).await?
            }
            InteractionType::Scroll { direction, amount } => {
                self.execute_scroll(direction, *amount).await?
            }
            InteractionType::Screenshot => {
                (true, Some("screenshot".to_string()), None)
            }
            InteractionType::ApplicationLaunch { app_name } => {
                self.execute_app_launch(app_name).await?
            }
            InteractionType::WindowManagement { action } => {
                self.execute_window_action(action).await?
            }
        };
        
        // Take screenshot after action
        let screenshot_after = if self.config.analyze_screenshots {
            Some(self.take_screenshot().await?)
        } else {
            None
        };
        
        let interaction = Interaction {
            id: uuid::Uuid::new_v4().to_string(),
            interaction_type: action,
            target,
            coordinates,
            success,
            duration_ms: start_time.elapsed().as_millis() as u64,
            screenshot_before,
            screenshot_after,
            timestamp: chrono::Utc::now(),
        };
        
        // Store interaction in history
        self.interaction_history.write().await.push(interaction.clone());
        
        // Store in memory if enabled
        if self.config.store_interactions_in_memory {
            self.store_interaction_in_memory(&interaction).await?;
        }
        
        Ok(interaction)
    }
    
    /// Check if an action is safe to execute
    async fn check_action_safety(&self, action: &InteractionType) -> Result<()> {
        // Rate limiting check
        let interactions = self.interaction_history.read().await;
        let recent_count = interactions.iter()
            .filter(|i| i.timestamp > chrono::Utc::now() - chrono::Duration::seconds(60))
            .count();
            
        if recent_count >= self.config.max_interactions_per_minute as usize {
            return Err(anyhow!("Rate limit exceeded: {} interactions in the last minute", recent_count));
        }
        
        // Check specific action safety
        match action {
            InteractionType::Type { text } => {
                // Check for sensitive patterns
                let sensitive_patterns = vec!["password", "token", "secret", "key"];
                for pattern in sensitive_patterns {
                    if text.to_lowercase().contains(pattern) {
                        warn!("Potentially sensitive text detected in type action");
                    }
                }
            }
            InteractionType::ApplicationLaunch { app_name } => {
                // Check if terminal automation is allowed
                if app_name.to_lowercase().contains("terminal") && !self.config.enable_terminal_automation {
                    return Err(anyhow!("Terminal automation is disabled"));
                }
            }
            _ => {}
        }
        
        Ok(())
    }
    
    /// Execute a mouse click
    async fn execute_click(&self, button: &MouseButton) -> Result<(bool, Option<String>, Option<(i32, i32)>)> {
        if !self.config.enable_mouse_control {
            return Err(anyhow!("Mouse control is disabled"));
        }
        
        info!("🖱️ Executing {:?} click", button);
        
        #[cfg(target_os = "macos")]
        {
            let output = Command::new("osascript")
                .arg("-e")
                .arg("tell application \"System Events\" to click at {100, 100}")
                .output()
                .await?;
                
            let success = output.status.success();
            Ok((success, Some("click".to_string()), Some((100, 100))))
        }
        
        #[cfg(not(target_os = "macos"))]
        {
            // Platform-specific implementation needed
            Ok((true, Some("click".to_string()), Some((100, 100))))
        }
    }
    
    /// Execute a double click
    async fn execute_double_click(&self, button: &MouseButton) -> Result<(bool, Option<String>, Option<(i32, i32)>)> {
        if !self.config.enable_mouse_control {
            return Err(anyhow!("Mouse control is disabled"));
        }
        
        info!("🖱️ Executing double {:?} click", button);
        
        #[cfg(target_os = "macos")]
        {
            let output = Command::new("osascript")
                .arg("-e")
                .arg("tell application \"System Events\" to double click at {100, 100}")
                .output()
                .await?;
                
            let success = output.status.success();
            Ok((success, Some("double_click".to_string()), Some((100, 100))))
        }
        
        #[cfg(not(target_os = "macos"))]
        {
            Ok((true, Some("double_click".to_string()), Some((100, 100))))
        }
    }
    
    /// Execute a right click
    async fn execute_right_click(&self) -> Result<(bool, Option<String>, Option<(i32, i32)>)> {
        if !self.config.enable_mouse_control {
            return Err(anyhow!("Mouse control is disabled"));
        }
        
        info!("🖱️ Executing right click");
        
        #[cfg(target_os = "macos")]
        {
            let output = Command::new("osascript")
                .arg("-e")
                .arg("tell application \"System Events\" to right click at {100, 100}")
                .output()
                .await?;
                
            let success = output.status.success();
            Ok((success, Some("right_click".to_string()), Some((100, 100))))
        }
        
        #[cfg(not(target_os = "macos"))]
        {
            Ok((true, Some("right_click".to_string()), Some((100, 100))))
        }
    }
    
    /// Execute a drag operation
    async fn execute_drag(&self, from: (i32, i32), to: (i32, i32)) -> Result<(bool, Option<String>, Option<(i32, i32)>)> {
        if !self.config.enable_mouse_control {
            return Err(anyhow!("Mouse control is disabled"));
        }
        
        info!("🖱️ Executing drag from {:?} to {:?}", from, to);
        
        #[cfg(target_os = "macos")]
        {
            let script = format!(
                "tell application \"System Events\" to click at {{{}, {}}} dragging to {{{}, {}}}",
                from.0, from.1, to.0, to.1
            );
            
            let output = Command::new("osascript")
                .arg("-e")
                .arg(&script)
                .output()
                .await?;
                
            let success = output.status.success();
            Ok((success, Some("drag".to_string()), Some(to)))
        }
        
        #[cfg(not(target_os = "macos"))]
        {
            Ok((true, Some("drag".to_string()), Some(to)))
        }
    }
    
    /// Execute text typing
    async fn execute_type(&self, text: &str) -> Result<(bool, Option<String>, Option<(i32, i32)>)> {
        if !self.config.enable_keyboard_control {
            return Err(anyhow!("Keyboard control is disabled"));
        }
        
        info!("⌨️ Typing text: {} characters", text.len());
        
        #[cfg(target_os = "macos")]
        {
            let escaped_text = text.replace("\"", "\\\"");
            let script = format!("tell application \"System Events\" to keystroke \"{}\"", escaped_text);
            
            let output = Command::new("osascript")
                .arg("-e")
                .arg(&script)
                .output()
                .await?;
                
            let success = output.status.success();
            Ok((success, Some(format!("typed {} chars", text.len())), None))
        }
        
        #[cfg(not(target_os = "macos"))]
        {
            Ok((true, Some(format!("typed {} chars", text.len())), None))
        }
    }
    
    /// Execute a key press
    async fn execute_key_press(&self, key: &str) -> Result<(bool, Option<String>, Option<(i32, i32)>)> {
        if !self.config.enable_keyboard_control {
            return Err(anyhow!("Keyboard control is disabled"));
        }
        
        info!("⌨️ Pressing key: {}", key);
        
        #[cfg(target_os = "macos")]
        {
            let key_code = match key.to_lowercase().as_str() {
                "enter" | "return" => "36",
                "tab" => "48",
                "space" => "49",
                "delete" | "backspace" => "51",
                "escape" | "esc" => "53",
                "up" => "126",
                "down" => "125",
                "left" => "123",
                "right" => "124",
                _ => return Err(anyhow!("Unknown key: {}", key)),
            };
            
            let script = format!("tell application \"System Events\" to key code {}", key_code);
            
            let output = Command::new("osascript")
                .arg("-e")
                .arg(&script)
                .output()
                .await?;
                
            let success = output.status.success();
            Ok((success, Some(format!("key_press: {}", key)), None))
        }
        
        #[cfg(not(target_os = "macos"))]
        {
            Ok((true, Some(format!("key_press: {}", key)), None))
        }
    }
    
    /// Execute a key combination
    async fn execute_key_combo(&self, keys: &[String]) -> Result<(bool, Option<String>, Option<(i32, i32)>)> {
        if !self.config.enable_keyboard_control {
            return Err(anyhow!("Keyboard control is disabled"));
        }
        
        info!("⌨️ Pressing key combo: {:?}", keys);
        
        #[cfg(target_os = "macos")]
        {
            let mut modifiers = Vec::new();
            let mut key = String::new();
            
            for k in keys {
                match k.to_lowercase().as_str() {
                    "cmd" | "command" => modifiers.push("command down"),
                    "ctrl" | "control" => modifiers.push("control down"),
                    "alt" | "option" => modifiers.push("option down"),
                    "shift" => modifiers.push("shift down"),
                    _ => key = k.clone(),
                }
            }
            
            let modifier_str = if modifiers.is_empty() {
                String::new()
            } else {
                format!(" using {{{}}}", modifiers.join(", "))
            };
            
            let script = format!(
                "tell application \"System Events\" to keystroke \"{}\"{}",
                key, modifier_str
            );
            
            let output = Command::new("osascript")
                .arg("-e")
                .arg(&script)
                .output()
                .await?;
                
            let success = output.status.success();
            Ok((success, Some(format!("key_combo: {:?}", keys)), None))
        }
        
        #[cfg(not(target_os = "macos"))]
        {
            Ok((true, Some(format!("key_combo: {:?}", keys)), None))
        }
    }
    
    /// Execute scroll action
    async fn execute_scroll(&self, direction: &ScrollDirection, amount: i32) -> Result<(bool, Option<String>, Option<(i32, i32)>)> {
        if !self.config.enable_mouse_control {
            return Err(anyhow!("Mouse control is disabled"));
        }
        
        info!("📜 Scrolling {:?} by {} units", direction, amount);
        
        #[cfg(target_os = "macos")]
        {
            let (x, y) = match direction {
                ScrollDirection::Up => (0, amount),
                ScrollDirection::Down => (0, -amount),
                ScrollDirection::Left => (-amount, 0),
                ScrollDirection::Right => (amount, 0),
            };
            
            let script = format!(
                "tell application \"System Events\" to scroll ({}, {})",
                x, y
            );
            
            let output = Command::new("osascript")
                .arg("-e")
                .arg(&script)
                .output()
                .await?;
                
            let success = output.status.success();
            Ok((success, Some(format!("scroll {:?} {}", direction, amount)), None))
        }
        
        #[cfg(not(target_os = "macos"))]
        {
            Ok((true, Some(format!("scroll {:?} {}", direction, amount)), None))
        }
    }
    
    /// Launch an application
    async fn execute_app_launch(&self, app_name: &str) -> Result<(bool, Option<String>, Option<(i32, i32)>)> {
        info!("🚀 Launching application: {}", app_name);
        
        #[cfg(target_os = "macos")]
        {
            let output = Command::new("open")
                .arg("-a")
                .arg(app_name)
                .output()
                .await?;
                
            let success = output.status.success();
            Ok((success, Some(format!("launch: {}", app_name)), None))
        }
        
        #[cfg(target_os = "linux")]
        {
            let output = Command::new(app_name)
                .spawn();
                
            let success = output.is_ok();
            Ok((success, Some(format!("launch: {}", app_name)), None))
        }
        
        #[cfg(target_os = "windows")]
        {
            let output = Command::new("cmd")
                .arg("/C")
                .arg("start")
                .arg(app_name)
                .output()
                .await?;
                
            let success = output.status.success();
            Ok((success, Some(format!("launch: {}", app_name)), None))
        }
    }
    
    /// Execute window management action
    async fn execute_window_action(&self, action: &WindowAction) -> Result<(bool, Option<String>, Option<(i32, i32)>)> {
        info!("🪟 Executing window action: {:?}", action);
        
        #[cfg(target_os = "macos")]
        {
            let script = match action {
                WindowAction::Minimize => {
                    "tell application \"System Events\" to set minimized of window 1 of (first process whose frontmost is true) to true"
                }
                WindowAction::Maximize => {
                    "tell application \"System Events\" to click button 2 of window 1 of (first process whose frontmost is true)"
                }
                WindowAction::Close => {
                    "tell application \"System Events\" to click button 1 of window 1 of (first process whose frontmost is true)"
                }
                WindowAction::Restore => {
                    "tell application \"System Events\" to set minimized of window 1 of (first process whose frontmost is true) to false"
                }
                WindowAction::Move { x, y } => {
                    return Ok((true, Some(format!("window_move: {},{}", x, y)), Some((*x, *y))));
                }
                WindowAction::Resize { width, height } => {
                    return Ok((true, Some(format!("window_resize: {}x{}", width, height)), None));
                }
            };
            
            let output = Command::new("osascript")
                .arg("-e")
                .arg(script)
                .output()
                .await?;
                
            let success = output.status.success();
            Ok((success, Some(format!("window_action: {:?}", action)), None))
        }
        
        #[cfg(not(target_os = "macos"))]
        {
            Ok((true, Some(format!("window_action: {:?}", action)), None))
        }
    }
    
    /// Store interaction in cognitive memory
    async fn store_interaction_in_memory(&self, interaction: &Interaction) -> Result<()> {
        let memory_content = format!(
            "Computer Interaction: {:?} | Success: {} | Duration: {}ms",
            interaction.interaction_type,
            interaction.success,
            interaction.duration_ms
        );
        
        self.memory.store(
            memory_content,
            vec![interaction.id.clone()],
            MemoryMetadata {
                source: "computer_use".to_string(),
                tags: vec![
                    "interaction".to_string(),
                    "automation".to_string(),
                    format!("{:?}", interaction.interaction_type),
                ],
                importance: if interaction.success { 0.7 } else { 0.9 },
                associations: vec![],
                context: Some("Computer automation interaction".to_string()),
                created_at: chrono::Utc::now(),
                accessed_count: 0,
                last_accessed: None,
                version: 1,
                category: "tools".to_string(),
                timestamp: chrono::Utc::now(),
                expiration: None,
            },
        ).await?;
        
        Ok(())
    }

    /// Get computer use statistics
    pub async fn get_stats(&self) -> ComputerUseStats {
        let interactions = self.interaction_history.read().await;

        ComputerUseStats {
            total_interactions: interactions.len(),
            successful_interactions: interactions.iter().filter(|i| i.success).count(),
            total_screenshots: 0, // Could count files in directory
            active_workflows: 0,  // Could track active workflows
        }
    }
}

/// Blender creation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlenderCreationResult {
    pub model_path: String,
    pub script_content: String,
    pub execution_time_ms: u64,
}

/// Creative workflow result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreativeWorkflowResult {
    pub id: String,
    pub original_prompt: String,
    pub generated_image: super::creative_media::GeneratedMedia,
    pub vision_analysis: VisualAnalysis,
    pub blender_model_path: String,
    pub blender_script: String,
    pub total_time_ms: u64,
    pub success: bool,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Computer use statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputerUseStats {
    pub total_interactions: usize,
    pub successful_interactions: usize,
    pub total_screenshots: usize,
    pub active_workflows: usize,
}
