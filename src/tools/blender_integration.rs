use std::collections::HashMap;
use std::io::Write;
use std::process::Command;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Result, anyhow};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tempfile::NamedTempFile;
use tokio::sync::{RwLock, broadcast};
use tokio::time::interval;
use tracing::{debug, error, info, warn};

use crate::cognitive::{CognitiveSystem, Thought, ThoughtMetadata, ThoughtType};
use crate::memory::{CognitiveMemory, MemoryMetadata};

/// Blender integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlenderConfig {
    /// Path to Blender executable
    pub blender_path: std::path::PathBuf,

    /// Enable headless mode (no GUI)
    pub headless_mode: bool,

    /// Default render engine
    pub default_render_engine: RenderEngine,

    /// Output quality settings
    pub default_render_quality: RenderQuality,

    /// Project storage path
    pub projects_path: std::path::PathBuf,

    /// Asset library path
    pub assets_path: std::path::PathBuf,

    /// Maximum script execution time
    pub max_script_timeout_seconds: u64,

    /// Enable cognitive integration
    pub cognitive_integration: bool,

    /// Auto-generate from descriptions
    pub auto_generation_enabled: bool,

    /// Export formats to support
    pub supported_formats: Vec<ExportFormat>,
}

impl Default for BlenderConfig {
    fn default() -> Self {
        Self {
            blender_path: std::path::PathBuf::from("blender"),
            headless_mode: true,
            default_render_engine: RenderEngine::Cycles,
            default_render_quality: RenderQuality::Medium,
            projects_path: std::path::PathBuf::from("./blender_projects"),
            assets_path: std::path::PathBuf::from("./blender_assets"),
            max_script_timeout_seconds: 300, // 5 minutes
            cognitive_integration: true,
            auto_generation_enabled: true,
            supported_formats: vec![
                ExportFormat::OBJ,
                ExportFormat::FBX,
                ExportFormat::GLTF,
                ExportFormat::STL,
                ExportFormat::PLY,
            ],
        }
    }
}

/// Render engines supported by Blender
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum RenderEngine {
    Cycles,    // Ray tracing engine
    Eevee,     // Real-time engine
    Workbench, // Simple viewport engine
}

/// Render quality presets
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum RenderQuality {
    Preview,    // Fast, low quality
    Medium,     // Balanced
    High,       // Slow, high quality
    Production, // Maximum quality
}

/// Export formats for 3D assets
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ExportFormat {
    OBJ,     // Wavefront OBJ
    FBX,     // Autodesk FBX
    GLTF,    // glTF 2.0
    STL,     // Stereolithography
    PLY,     // Polygon File Format
    COLLADA, // COLLADA DAE
    X3D,     // X3D format
}

/// Types of 3D content Loki can create
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContentType {
    /// Basic geometric primitives
    Primitive { shape: PrimitiveShape, parameters: HashMap<String, f32> },

    /// Parametric models
    Parametric { model_type: ParametricModel, parameters: HashMap<String, Value> },

    /// Scene compositions
    Scene { objects: Vec<SceneObject>, lighting: LightingSetup, camera: CameraSetup },

    /// Animated sequences
    Animation { objects: Vec<AnimatedObject>, duration_frames: u32, frame_rate: f32 },

    /// Material and shader networks
    Material { material_type: MaterialType, properties: HashMap<String, Value> },
}

/// Basic primitive shapes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PrimitiveShape {
    Cube,
    Sphere,
    Cylinder,
    Cone,
    Plane,
    Torus,
    Monkey, // Suzanne monkey head
}

/// Parametric model types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParametricModel {
    Architecture(ArchitectureType),
    Organic(OrganicType),
    Mechanical(MechanicalType),
    Abstract(AbstractType),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArchitectureType {
    Building,
    Room,
    Furniture,
    Landscape,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OrganicType {
    Plant,
    Terrain,
    Character,
    Animal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MechanicalType {
    Vehicle,
    Robot,
    Tool,
    Machine,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AbstractType {
    Fractal,
    Procedural,
    Artistic,
    Mathematical,
}

/// Scene object definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneObject {
    pub name: String,
    pub object_type: ContentType,
    pub transform: Transform,
    pub material: Option<String>,
}

/// 3D transformation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transform {
    pub location: [f32; 3],
    pub rotation: [f32; 3],
    pub scale: [f32; 3],
}

impl Default for Transform {
    fn default() -> Self {
        Self { location: [0.0, 0.0, 0.0], rotation: [0.0, 0.0, 0.0], scale: [1.0, 1.0, 1.0] }
    }
}

/// Lighting setup
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LightingSetup {
    pub lights: Vec<Light>,
    pub environment: Option<String>,
    pub global_illumination: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Light {
    pub light_type: LightType,
    pub location: [f32; 3],
    pub energy: f32,
    pub color: [f32; 3],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LightType {
    Sun,
    Point,
    Spot,
    Area,
}

/// Camera setup
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraSetup {
    pub location: [f32; 3],
    pub rotation: [f32; 3],
    pub focal_length: f32,
    pub clip_start: f32,
    pub clip_end: f32,
}

impl Default for CameraSetup {
    fn default() -> Self {
        Self {
            location: [7.4, -6.5, 4.4],
            rotation: [63.4, 0.0, 46.7],
            focal_length: 35.0,
            clip_start: 0.1,
            clip_end: 1000.0,
        }
    }
}

/// Animated object
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnimatedObject {
    pub object: SceneObject,
    pub keyframes: Vec<Keyframe>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Keyframe {
    pub frame: u32,
    pub transform: Transform,
    pub properties: HashMap<String, Value>,
}

/// Material types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MaterialType {
    Principled, // Physically based
    Emission,   // Emissive material
    Glass,      // Transparent glass
    Metal,      // Metallic material
    Subsurface, // Subsurface scattering
    Toon,       // Cartoon-style
}

/// Blender project
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlenderProject {
    pub id: String,
    pub name: String,
    pub description: String,
    pub content_type: ContentType,
    pub file_path: std::path::PathBuf,
    pub render_settings: RenderSettings,
    pub created_at: DateTime<Utc>,
    pub cognitive_context: Option<String>,
    pub generated_assets: Vec<String>,
}

/// Render settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenderSettings {
    pub engine: RenderEngine,
    pub resolution: (u32, u32),
    pub samples: u32,
    pub frame_range: (u32, u32),
    pub output_format: String,
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            engine: RenderEngine::Cycles,
            resolution: (1920, 1080),
            samples: 128,
            frame_range: (1, 250),
            output_format: "PNG".to_string(),
        }
    }
}

/// Blender integration events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BlenderEvent {
    ProjectCreated { project: BlenderProject },
    AssetGenerated { asset_path: String, asset_type: String, project_id: String },
    RenderCompleted { project_id: String, output_path: String, render_time_ms: u64 },
    ScriptExecuted { script_name: String, execution_time_ms: u64, success: bool },
    CognitiveIntegration { thought: String, generated_content: String },
}

/// Main Blender integration manager
pub struct BlenderIntegration {
    config: BlenderConfig,
    cognitive_system: Arc<CognitiveSystem>,
    memory: Arc<CognitiveMemory>,

    /// Active projects
    projects: Arc<RwLock<HashMap<String, BlenderProject>>>,

    /// Event broadcasting
    event_tx: broadcast::Sender<BlenderEvent>,

    /// Asset library
    assets: Arc<RwLock<HashMap<String, AssetInfo>>>,

    /// Script templates
    script_templates: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetInfo {
    pub id: String,
    pub name: String,
    pub asset_type: String,
    pub file_path: std::path::PathBuf,
    pub description: String,
    pub tags: Vec<String>,
    pub created_at: DateTime<Utc>,
}

impl BlenderIntegration {
    /// Create new Blender integration
    pub async fn new(
        config: BlenderConfig,
        cognitive_system: Arc<CognitiveSystem>,
        memory: Arc<CognitiveMemory>,
    ) -> Result<Self> {
        // Verify Blender installation
        Self::verify_blender_installation(&config.blender_path).await?;

        // Create directories
        tokio::fs::create_dir_all(&config.projects_path).await?;
        tokio::fs::create_dir_all(&config.assets_path).await?;

        let (event_tx, _) = broadcast::channel(1000);

        let mut script_templates = HashMap::new();
        script_templates.insert("basic_scene".to_string(), Self::create_basic_scene_script());
        script_templates
            .insert("parametric_model".to_string(), Self::create_parametric_model_script());
        script_templates.insert("material_setup".to_string(), Self::create_material_setup_script());
        script_templates.insert("animation_basic".to_string(), Self::create_animation_script());

        let integration = Self {
            config,
            cognitive_system,
            memory,
            projects: Arc::new(RwLock::new(HashMap::new())),
            event_tx,
            assets: Arc::new(RwLock::new(HashMap::new())),
            script_templates,
        };

        // Start background tasks
        integration.start_background_tasks().await?;

        info!("Blender integration initialized successfully");
        Ok(integration)
    }

    /// Verify Blender installation
    async fn verify_blender_installation(blender_path: &std::path::Path) -> Result<()> {
        let output = Command::new(blender_path).args(&["--version"]).output().map_err(|e| {
            anyhow!("Failed to execute Blender: {}. Make sure Blender is installed and in PATH", e)
        })?;

        if !output.status.success() {
            return Err(anyhow!("Blender version check failed"));
        }

        let version_info = String::from_utf8_lossy(&output.stdout);
        info!(
            "Blender installation verified: {}",
            version_info.lines().next().unwrap_or("Unknown version")
        );

        Ok(())
    }

    /// Generate 3D content from description
    pub async fn generate_from_description(
        &self,
        description: &str,
        content_type: Option<ContentType>,
    ) -> Result<BlenderProject> {
        info!("Generating 3D content from description: {}", description);

        // Process through cognitive system if enabled
        if self.config.cognitive_integration {
            self.process_creative_thought(description).await?;
        }

        // Determine content type if not provided
        let content_type = match content_type {
            Some(ct) => ct,
            None => self.infer_content_type_from_description(description).await?,
        };

        // Generate project
        let project = self.create_project(description, content_type).await?;

        // Generate the actual content
        self.execute_generation_script(&project).await?;

        // Store project
        {
            let mut projects = self.projects.write().await;
            projects.insert(project.id.clone(), project.clone());
        }

        // Emit event
        let _ = self.event_tx.send(BlenderEvent::ProjectCreated { project: project.clone() });

        // Store in cognitive memory
        if self.config.cognitive_integration {
            self.store_generation_memory(&project).await?;
        }

        Ok(project)
    }

    /// Create a new Blender project
    async fn create_project(
        &self,
        description: &str,
        content_type: ContentType,
    ) -> Result<BlenderProject> {
        let id = uuid::Uuid::new_v4().to_string();
        let name = format!("loki_generation_{}", &id[..8]);
        let file_path = self.config.projects_path.join(format!("{}.blend", name));

        let project = BlenderProject {
            id,
            name,
            description: description.to_string(),
            content_type,
            file_path,
            render_settings: RenderSettings::default(),
            created_at: Utc::now(),
            cognitive_context: Some("Generated from natural language description".to_string()),
            generated_assets: Vec::new(),
        };

        Ok(project)
    }

    /// Execute Blender script for content generation
    async fn execute_generation_script(&self, project: &BlenderProject) -> Result<()> {
        let script = self.generate_blender_script(project)?;
        self.execute_blender_script(&script, Some(&project.file_path)).await
    }

    /// Generate appropriate Blender Python script
    fn generate_blender_script(&self, project: &BlenderProject) -> Result<String> {
        let mut script = String::new();

        // Add imports
        script.push_str("import bpy\n");
        script.push_str("import bmesh\n");
        script.push_str("import mathutils\n");
        script.push_str("from mathutils import Vector, Euler\n");
        script.push_str("import math\n\n");

        // Clear default scene
        script.push_str("# Clear default scene\n");
        script.push_str("bpy.ops.object.select_all(action='SELECT')\n");
        script.push_str("bpy.ops.object.delete(use_global=False)\n\n");

        // Generate content based on type
        match &project.content_type {
            ContentType::Primitive { shape, parameters } => {
                script.push_str(&self.generate_primitive_script(shape, parameters));
            }
            ContentType::Parametric { model_type, parameters } => {
                script.push_str(&self.generate_parametric_script(model_type, parameters));
            }
            ContentType::Scene { objects, lighting, camera } => {
                script.push_str(&self.generate_scene_script(objects, lighting, camera));
            }
            ContentType::Animation { objects, duration_frames, frame_rate } => {
                script.push_str(&self.generate_animation_script(
                    objects,
                    *duration_frames,
                    *frame_rate,
                ));
            }
            ContentType::Material { material_type, properties } => {
                script.push_str(&self.generate_material_script(material_type, properties));
            }
        }

        // Add basic lighting if not specified
        if !matches!(project.content_type, ContentType::Scene { .. }) {
            script.push_str(&self.generate_basic_lighting());
        }

        // Add camera if not specified
        if !matches!(project.content_type, ContentType::Scene { .. }) {
            script.push_str(&self.generate_basic_camera());
        }

        // Save the file
        script.push_str(&format!(
            "\n# Save the project\nbpy.ops.wm.save_as_mainfile(filepath='{}')\n",
            project.file_path.to_string_lossy()
        ));

        Ok(script)
    }

    /// Generate script for primitive shapes
    fn generate_primitive_script(
        &self,
        shape: &PrimitiveShape,
        parameters: &HashMap<String, f32>,
    ) -> String {
        let mut script = String::new();

        match shape {
            PrimitiveShape::Cube => {
                let size = parameters.get("size").unwrap_or(&2.0);
                script.push_str(&format!("bpy.ops.mesh.primitive_cube_add(size={})\n", size));
            }
            PrimitiveShape::Sphere => {
                let radius = parameters.get("radius").unwrap_or(&1.0);
                let subdivisions = *parameters.get("subdivisions").unwrap_or(&2.0) as u32;
                script.push_str(&format!(
                    "bpy.ops.mesh.primitive_uv_sphere_add(radius={}, subdivisions={})\n",
                    radius, subdivisions
                ));
            }
            PrimitiveShape::Cylinder => {
                let radius = parameters.get("radius").unwrap_or(&1.0);
                let depth = parameters.get("depth").unwrap_or(&2.0);
                script.push_str(&format!(
                    "bpy.ops.mesh.primitive_cylinder_add(radius={}, depth={})\n",
                    radius, depth
                ));
            }
            PrimitiveShape::Cone => {
                let radius1 = parameters.get("radius1").unwrap_or(&1.0);
                let depth = parameters.get("depth").unwrap_or(&2.0);
                script.push_str(&format!(
                    "bpy.ops.mesh.primitive_cone_add(radius1={}, depth={})\n",
                    radius1, depth
                ));
            }
            PrimitiveShape::Plane => {
                let size = parameters.get("size").unwrap_or(&2.0);
                script.push_str(&format!("bpy.ops.mesh.primitive_plane_add(size={})\n", size));
            }
            PrimitiveShape::Torus => {
                let major_radius = parameters.get("major_radius").unwrap_or(&1.0);
                let minor_radius = parameters.get("minor_radius").unwrap_or(&0.25);
                script.push_str(&format!(
                    "bpy.ops.mesh.primitive_torus_add(major_radius={}, minor_radius={})\n",
                    major_radius, minor_radius
                ));
            }
            PrimitiveShape::Monkey => {
                script.push_str("bpy.ops.mesh.primitive_monkey_add()\n");
            }
        }

        script
    }

    /// Generate basic lighting setup
    fn generate_basic_lighting(&self) -> String {
        format!(
            "# Add basic lighting\nbpy.ops.object.light_add(type='SUN', location=(5, 5, 10))\nsun \
             = bpy.context.object\nsun.data.energy = 3.0\nbpy.ops.object.light_add(type='AREA', \
             location=(-5, -5, 8))\narea = bpy.context.object\narea.data.energy = 2.0\n\n"
        )
    }

    /// Generate basic camera setup
    fn generate_basic_camera(&self) -> String {
        format!(
            "# Add camera\nbpy.ops.object.camera_add(location=(7.4, -6.5, 4.4))\ncamera = \
             bpy.context.object\ncamera.rotation_euler = (1.109, 0, 0.815)\n\n"
        )
    }

    /// Execute Blender script
    async fn execute_blender_script(
        &self,
        script: &str,
        _output_file: Option<&std::path::Path>, /* TODO: Use output_file for script output
                                                 * redirection */
    ) -> Result<()> {
        let start_time = Instant::now();

        // Create temporary script file
        let mut script_file = NamedTempFile::new()?;
        script_file.write_all(script.as_bytes())?;

        // Build Blender command
        let mut cmd = Command::new(&self.config.blender_path);

        if self.config.headless_mode {
            cmd.arg("--background");
        }

        cmd.arg("--python").arg(script_file.path());

        // Execute with timeout and process output
        let output = tokio::time::timeout(
            Duration::from_secs(self.config.max_script_timeout_seconds),
            tokio::task::spawn_blocking(move || cmd.output()),
        )
        .await???;

        let execution_time = start_time.elapsed();

        // Process script output for results and error handling
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        
        if !output.status.success() {
            error!("Blender script execution failed: {}", stderr);
            return Err(anyhow!("Blender script execution failed: {}", stderr));
        }

        // Extract execution results from stdout
        let _execution_result = self.parse_blender_output(&stdout, &stderr)?;
        
        info!("Blender script executed successfully in {:?}", execution_time);
        debug!("Script output: {}", stdout.trim());
        
        if !stderr.is_empty() {
            debug!("Script warnings: {}", stderr.trim());
        }

        // Emit event
        let _ = self.event_tx.send(BlenderEvent::ScriptExecuted {
            script_name: "generated_script".to_string(),
            execution_time_ms: execution_time.as_millis() as u64,
            success: true,
        });

        Ok(())
    }

    /// Infer content type from natural language description
    async fn infer_content_type_from_description(&self, description: &str) -> Result<ContentType> {
        let desc_lower = description.to_lowercase();

        // Simple keyword-based inference (could be enhanced with ML)
        if desc_lower.contains("cube") || desc_lower.contains("box") {
            Ok(ContentType::Primitive { shape: PrimitiveShape::Cube, parameters: HashMap::new() })
        } else if desc_lower.contains("sphere") || desc_lower.contains("ball") {
            Ok(ContentType::Primitive { shape: PrimitiveShape::Sphere, parameters: HashMap::new() })
        } else if desc_lower.contains("building") || desc_lower.contains("house") {
            Ok(ContentType::Parametric {
                model_type: ParametricModel::Architecture(ArchitectureType::Building),
                parameters: HashMap::new(),
            })
        } else if desc_lower.contains("scene") || desc_lower.contains("environment") {
            Ok(ContentType::Scene {
                objects: vec![],
                lighting: LightingSetup {
                    lights: vec![],
                    environment: None,
                    global_illumination: true,
                },
                camera: CameraSetup::default(),
            })
        } else {
            // Default to a simple cube
            Ok(ContentType::Primitive { shape: PrimitiveShape::Cube, parameters: HashMap::new() })
        }
    }

    /// Render project to image/animation
    pub async fn render_project(&self, project_id: &str) -> Result<String> {
        let project = {
            let projects = self.projects.read().await;
            projects
                .get(project_id)
                .ok_or_else(|| anyhow!("Project not found: {}", project_id))?
                .clone()
        };

        let output_path = self.config.projects_path.join(format!("{}_render.png", project.name));

        let script = format!(
            "import bpy\nbpy.ops.wm.open_mainfile(filepath='{}')\nbpy.context.scene.render.\
             filepath = '{}'\nbpy.context.scene.render.engine = \
             '{}'\nbpy.context.scene.render.resolution_x = \
             {}\nbpy.context.scene.render.resolution_y = \
             {}\nbpy.ops.render.render(write_still=True)\n",
            project.file_path.to_string_lossy(),
            output_path.to_string_lossy(),
            format!("{:?}", project.render_settings.engine).to_uppercase(),
            project.render_settings.resolution.0,
            project.render_settings.resolution.1,
        );

        let start_time = Instant::now();
        self.execute_blender_script(&script, None).await?;
        let render_time = start_time.elapsed();

        // Emit event
        let _ = self.event_tx.send(BlenderEvent::RenderCompleted {
            project_id: project_id.to_string(),
            output_path: output_path.to_string_lossy().to_string(),
            render_time_ms: render_time.as_millis() as u64,
        });

        Ok(output_path.to_string_lossy().to_string())
    }

    /// Export project in various formats
    pub async fn export_project(&self, project_id: &str, format: ExportFormat) -> Result<String> {
        let project = {
            let projects = self.projects.read().await;
            projects
                .get(project_id)
                .ok_or_else(|| anyhow!("Project not found: {}", project_id))?
                .clone()
        };

        let extension = match format {
            ExportFormat::OBJ => "obj",
            ExportFormat::FBX => "fbx",
            ExportFormat::GLTF => "gltf",
            ExportFormat::STL => "stl",
            ExportFormat::PLY => "ply",
            ExportFormat::COLLADA => "dae",
            ExportFormat::X3D => "x3d",
        };

        let output_path =
            self.config.projects_path.join(format!("{}_export.{}", project.name, extension));

        let export_script = match format {
            ExportFormat::OBJ => {
                format!("bpy.ops.wm.obj_export(filepath='{}')", output_path.to_string_lossy())
            }
            ExportFormat::FBX => {
                format!("bpy.ops.export_scene.fbx(filepath='{}')", output_path.to_string_lossy())
            }
            ExportFormat::GLTF => {
                format!("bpy.ops.export_scene.gltf(filepath='{}')", output_path.to_string_lossy())
            }
            ExportFormat::STL => {
                format!("bpy.ops.export_mesh.stl(filepath='{}')", output_path.to_string_lossy())
            }
            ExportFormat::PLY => {
                format!("bpy.ops.wm.ply_export(filepath='{}')", output_path.to_string_lossy())
            }
            ExportFormat::COLLADA => format!(
                "bpy.ops.wm.collada_export(filepath='{}', apply_modifiers=True, selected=False)",
                output_path.to_string_lossy()
            ),
            ExportFormat::X3D => format!(
                "bpy.ops.export_scene.x3d(filepath='{}', use_selection=False, \
                 use_mesh_modifiers=True)",
                output_path.to_string_lossy()
            ),
        };

        let script = format!(
            "import bpy\nbpy.ops.wm.open_mainfile(filepath='{}')\nbpy.ops.object.\
             select_all(action='SELECT')\n{}\n",
            project.file_path.to_string_lossy(),
            export_script
        );

        self.execute_blender_script(&script, None).await?;

        Ok(output_path.to_string_lossy().to_string())
    }

    /// Process creative thought through cognitive system
    async fn process_creative_thought(&self, description: &str) -> Result<()> {
        let thought = Thought {
            id: crate::cognitive::ThoughtId::new(),
            content: format!("Creating 3D content: {}", description),
            thought_type: ThoughtType::Creation,
            metadata: ThoughtMetadata {
                source: "blender_integration".to_string(),
                confidence: 0.8,
                emotional_valence: 0.4, // Creative excitement
                importance: 0.8,
                tags: vec!["3d".to_string(), "blender".to_string(), "creation".to_string()],
            },
            parent: None,
            children: Vec::new(),
            timestamp: Instant::now(),
        };

        // Send to cognitive system
        self.cognitive_system.process_query(&thought.content).await?;

        Ok(())
    }

    /// Store generation in cognitive memory
    async fn store_generation_memory(&self, project: &BlenderProject) -> Result<()> {
        let memory_content = format!(
            "Generated 3D project '{}': {} | Type: {:?} | File: {}",
            project.name,
            project.description,
            project.content_type,
            project.file_path.to_string_lossy()
        );

        self.memory
            .store(
                memory_content,
                vec![project.description.clone()],
                MemoryMetadata {
                    source: "blender_integration".to_string(),
                    tags: vec!["3d".to_string(), "blender".to_string(), "generated".to_string()],
                    importance: 0.8,
                    associations: vec![],

                    context: Some("Generated from automated fix".to_string()),
                    created_at: chrono::Utc::now(),
                    accessed_count: 0,
                    last_accessed: None,
                    version: 1,
                    category: "tool_usage".to_string(),
                    timestamp: chrono::Utc::now(),
                    expiration: None,
                },
            )
            .await?;

        Ok(())
    }

    /// Start background tasks
    async fn start_background_tasks(&self) -> Result<()> {
        // Asset library scanning
        self.start_asset_scanning().await?;

        // Cognitive integration loop
        if self.config.cognitive_integration && self.config.auto_generation_enabled {
            self.start_cognitive_integration_loop().await?;
        }

        Ok(())
    }

    /// Start asset library scanning
    async fn start_asset_scanning(&self) -> Result<()> {
        let integration = self.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(3600)); // 1 hour

            loop {
                interval.tick().await;

                if let Err(e) = integration.scan_asset_library().await {
                    warn!("Asset library scan failed: {}", e);
                }
            }
        });

        Ok(())
    }

    /// Scan asset library for existing assets
    async fn scan_asset_library(&self) -> Result<()> {
        // Implementation would scan for .blend files and extract metadata
        // For now, just log that scanning occurred
        debug!("Scanning asset library for updates");
        Ok(())
    }

    /// Start cognitive integration loop
    async fn start_cognitive_integration_loop(&self) -> Result<()> {
        let integration = self.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(4 * 3600)); // Every 4 hours

            loop {
                interval.tick().await;

                if let Err(e) = integration.process_cognitive_inspirations().await {
                    warn!("Cognitive integration processing failed: {}", e);
                }
            }
        });

        Ok(())
    }

    /// Process cognitive inspirations into 3D content
    async fn process_cognitive_inspirations(&self) -> Result<()> {
        // Get recent creative thoughts from memory
        let recent_memories = self.memory.retrieve_similar("creative 3d", 5).await?;

        for memory in recent_memories {
            if memory.metadata.tags.contains(&"creative".to_string())
                || memory.metadata.tags.contains(&"visual".to_string())
            {
                // Generate 3D representation of the creative thought
                if let Ok(project) = self.generate_from_description(&memory.content, None).await {
                    let _ = self.event_tx.send(BlenderEvent::CognitiveIntegration {
                        thought: memory.content.clone(),
                        generated_content: project.id,
                    });

                    info!("Generated 3D content from cognitive inspiration: {}", project.name);
                }
            }
        }

        Ok(())
    }

    /// Get list of available projects
    pub async fn list_projects(&self) -> Vec<BlenderProject> {
        let projects = self.projects.read().await;
        projects.values().cloned().collect()
    }

    /// Get project by ID
    pub async fn get_project(&self, project_id: &str) -> Option<BlenderProject> {
        let projects = self.projects.read().await;
        projects.get(project_id).cloned()
    }

    /// Create basic scene script template
    fn create_basic_scene_script() -> String {
        "# Basic scene setup\nimport bpy\nbpy.ops.mesh.primitive_cube_add()\n".to_string()
    }

    /// Create parametric model script template
    fn create_parametric_model_script() -> String {
        "# Parametric model generation\nimport bpy\nimport bmesh\n".to_string()
    }

    /// Create material setup script template
    fn create_material_setup_script() -> String {
        "# Material setup\nimport bpy\nmat = bpy.data.materials.new('Material')\n".to_string()
    }

    /// Create animation script template
    fn create_animation_script() -> String {
        "# Animation setup\nimport bpy\nbpy.context.scene.frame_start = 1\n".to_string()
    }

    /// Generate parametric script with real procedural modeling
    fn generate_parametric_script(
        &self,
        model_type: &ParametricModel,
        parameters: &HashMap<String, Value>,
    ) -> String {
        let mut script = String::new();
        script.push_str(
            "import bpy\nimport bmesh\nimport mathutils\nfrom mathutils import Vector\nimport \
             math\n\n",
        );

        match model_type {
            ParametricModel::Architecture(arch_type) => {
                script.push_str(&self.generate_architecture_script(arch_type, parameters));
            }
            ParametricModel::Organic(organic_type) => {
                script.push_str(&self.generate_organic_script(organic_type, parameters));
            }
            ParametricModel::Mechanical(mech_type) => {
                script.push_str(&self.generate_mechanical_script(mech_type, parameters));
            }
            ParametricModel::Abstract(abstract_type) => {
                script.push_str(&self.generate_abstract_script(abstract_type, parameters));
            }
        }

        script
    }

    /// Generate architecture-specific parametric scripts
    fn generate_architecture_script(
        &self,
        arch_type: &ArchitectureType,
        parameters: &HashMap<String, Value>,
    ) -> String {
        match arch_type {
            ArchitectureType::Building => {
                let height = Self::get_float_param(parameters, "height", 10.0);
                let width = Self::get_float_param(parameters, "width", 8.0);
                let depth = Self::get_float_param(parameters, "depth", 6.0);
                let floors = Self::get_int_param(parameters, "floors", 3);

                format!(
                    r#"# Generate procedural building
def generate_building(height={}, width={}, depth={}, floors={}):
    # Create base mesh
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, height/2))
    building = bpy.context.object
    building.scale = (width, depth, height)

    # Add floors
    floor_height = height / floors
    for i in range(1, floors):
        bpy.ops.mesh.primitive_plane_add(size=width*1.1, location=(0, 0, i * floor_height))
        floor = bpy.context.object
        floor.scale = (1, depth/width, 0.1)

    # Add windows
    for floor in range(floors):
        for side in [-1, 1]:
            for window in range(int(width//2)):
                bpy.ops.mesh.primitive_cube_add(
                    size=0.8,
                    location=(side * width/2.1, window * 1.5 - depth/4, floor * floor_height + 1)
                )
                window_obj = bpy.context.object
                window_obj.scale = (0.1, 0.3, 0.6)

generate_building()
"#,
                    height, width, depth, floors
                )
            }
            ArchitectureType::Room => {
                let room_size = Self::get_float_param(parameters, "size", 5.0);
                let ceiling_height = Self::get_float_param(parameters, "ceiling_height", 3.0);

                format!(
                    r#"# Generate room interior
def generate_room(size={}, height={}):
    # Floor
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0, 0, 0))
    floor = bpy.context.object

    # Walls
    for i, location in enumerate([(size/2, 0, height/2), (-size/2, 0, height/2), (0, size/2, height/2), (0, -size/2, height/2)]):
        bpy.ops.mesh.primitive_cube_add(size=1, location=location)
        wall = bpy.context.object
        if i < 2:  # Front/back walls
            wall.scale = (0.1, size, height)
        else:  # Side walls
            wall.scale = (size, 0.1, height)

    # Ceiling
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0, 0, height))
    ceiling = bpy.context.object

generate_room()
"#,
                    room_size, ceiling_height
                )
            }
            ArchitectureType::Furniture => {
                let furniture_type = Self::get_string_param(parameters, "type", "chair");
                let size = Self::get_float_param(parameters, "size", 1.0);

                match furniture_type.as_str() {
                    "chair" => format!(
                        r#"# Generate procedural chair
def generate_chair(size={}):
    # Seat
    bpy.ops.mesh.primitive_cube_add(size=size, location=(0, 0, size*0.5))
    seat = bpy.context.object
    seat.scale = (1, 1, 0.1)

    # Backrest
    bpy.ops.mesh.primitive_cube_add(size=size, location=(0, -size*0.45, size*1.2))
    backrest = bpy.context.object
    backrest.scale = (1, 0.1, 1.4)

    # Legs
    for x in [-1, 1]:
        for y in [-1, 1]:
            bpy.ops.mesh.primitive_cylinder_add(
                radius=size*0.05,
                depth=size,
                location=(x*size*0.4, y*size*0.4, 0)
            )

generate_chair()
"#,
                        size
                    ),
                    "table" => format!(
                        r#"# Generate procedural table
def generate_table(size={}):
    # Table top
    bpy.ops.mesh.primitive_cube_add(size=size, location=(0, 0, size*0.8))
    top = bpy.context.object
    top.scale = (2, 1, 0.1)

    # Legs
    for x in [-1, 1]:
        for y in [-1, 1]:
            bpy.ops.mesh.primitive_cylinder_add(
                radius=size*0.05,
                depth=size*0.8,
                location=(x*size*0.9, y*size*0.4, size*0.4)
            )

generate_table()
"#,
                        size
                    ),
                    _ => format!(
                        r#"# Generate generic furniture
def generate_furniture():
    bpy.ops.mesh.primitive_cube_add(size={})

generate_furniture()
"#,
                        size
                    ),
                }
            }
            ArchitectureType::Landscape => {
                let terrain_size = Self::get_float_param(parameters, "terrain_size", 20.0);
                let hills = Self::get_int_param(parameters, "hills", 3);
                let trees = Self::get_int_param(parameters, "trees", 10);

                format!(
                    r#"# Generate landscape
def generate_landscape(size={}, hills={}, trees={}):
    import random
    random.seed(42)

    # Base terrain
    bpy.ops.mesh.primitive_plane_add(size=size)
    terrain = bpy.context.object

    # Subdivide for detail
    bpy.context.view_layer.objects.active = terrain
    bpy.ops.object.mode_set(mode='EDIT')
    for _ in range(3):
        bpy.ops.mesh.subdivide()
    bpy.ops.object.mode_set(mode='OBJECT')

    # Add hills
    for i in range(hills):
        hill_x = random.uniform(-size/3, size/3)
        hill_y = random.uniform(-size/3, size/3)
        hill_height = random.uniform(1, 3)

        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=random.uniform(2, 4),
            location=(hill_x, hill_y, hill_height/2)
        )
        hill = bpy.context.object
        hill.scale = (1, 1, 0.3)

    # Add trees
    for i in range(trees):
        tree_x = random.uniform(-size/2, size/2)
        tree_y = random.uniform(-size/2, size/2)

        # Trunk
        bpy.ops.mesh.primitive_cylinder_add(
            radius=0.1,
            depth=2,
            location=(tree_x, tree_y, 1)
        )

        # Foliage
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=1,
            location=(tree_x, tree_y, 2.5)
        )
        foliage = bpy.context.object
        foliage.scale = (1, 1, 0.7)

generate_landscape()
"#,
                    terrain_size, hills, trees
                )
            }
        }
    }

    /// Generate organic-type parametric scripts
    fn generate_organic_script(
        &self,
        organic_type: &OrganicType,
        parameters: &HashMap<String, Value>,
    ) -> String {
        match organic_type {
            OrganicType::Plant => {
                let height = Self::get_float_param(parameters, "height", 2.0);
                let branches = Self::get_int_param(parameters, "branches", 5);
                let leaves = Self::get_int_param(parameters, "leaves", 20);

                format!(
                    r#"# Generate procedural plant
def generate_plant(height={}, branches={}, leaves={}):
    import random
    random.seed(42)

    # Trunk
    bpy.ops.mesh.primitive_cylinder_add(radius=0.1, depth=height, location=(0, 0, height/2))
    trunk = bpy.context.object

    # Branches
    for i in range(branches):
        angle = (i / branches) * 2 * math.pi
        branch_height = height * 0.7 + random.uniform(-0.2, 0.2)
        x = math.cos(angle) * 0.5
        y = math.sin(angle) * 0.5

        bpy.ops.mesh.primitive_cylinder_add(
            radius=0.05,
            depth=0.8,
            location=(x, y, branch_height),
            rotation=(random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), angle)
        )

        # Leaves on this branch
        for leaf in range(leaves // branches):
            leaf_offset = random.uniform(0.2, 0.8)
            leaf_x = x + math.cos(angle + leaf) * 0.3
            leaf_y = y + math.sin(angle + leaf) * 0.3
            leaf_z = branch_height + leaf_offset * 0.4

            bpy.ops.mesh.primitive_plane_add(
                size=0.2,
                location=(leaf_x, leaf_y, leaf_z),
                rotation=(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0, 2*math.pi))
            )

generate_plant()
"#,
                    height, branches, leaves
                )
            }
            OrganicType::Terrain => {
                let size = Self::get_float_param(parameters, "size", 10.0);
                let subdivisions = Self::get_int_param(parameters, "subdivisions", 8);
                let height_variation = Self::get_float_param(parameters, "height_variation", 2.0);

                format!(
                    r#"# Generate procedural terrain
def generate_terrain(size={}, subdivisions={}, height_variation={}):
    # Create base plane
    bpy.ops.mesh.primitive_plane_add(size=size)
    terrain = bpy.context.object

    # Enter edit mode and subdivide
    bpy.context.view_layer.objects.active = terrain
    bpy.ops.object.mode_set(mode='EDIT')

    for _ in range(subdivisions):
        bpy.ops.mesh.subdivide()

    # Add displacement modifier for height variation
    bpy.ops.object.mode_set(mode='OBJECT')

    # Create noise texture
    terrain.modifiers.new('Displace', 'DISPLACE')
    displace = terrain.modifiers['Displace']
    displace.strength = height_variation

    # Apply some randomization to vertices
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.transform.vertex_random(offset=height_variation/4)
    bpy.ops.object.mode_set(mode='OBJECT')

generate_terrain()
"#,
                    size, subdivisions, height_variation
                )
            }
            OrganicType::Character => {
                let scale = Self::get_float_param(parameters, "scale", 1.8);
                let style = Self::get_string_param(parameters, "style", "humanoid");

                format!(
                    r#"# Generate basic character model
def generate_character(scale={}, style='{}'):
    # Head
    bpy.ops.mesh.primitive_uv_sphere_add(radius=scale*0.12, location=(0, 0, scale*0.85))
    head = bpy.context.object

    # Torso
    bpy.ops.mesh.primitive_cube_add(size=scale*0.4, location=(0, 0, scale*0.55))
    torso = bpy.context.object
    torso.scale = (1, 0.6, 1.2)

    # Arms
    for side in [-1, 1]:
        # Upper arm
        bpy.ops.mesh.primitive_cylinder_add(
            radius=scale*0.06,
            depth=scale*0.25,
            location=(side * scale*0.25, 0, scale*0.6),
            rotation=(0, 0, side * math.pi/6)
        )

        # Lower arm
        bpy.ops.mesh.primitive_cylinder_add(
            radius=scale*0.05,
            depth=scale*0.22,
            location=(side * scale*0.38, 0, scale*0.35),
            rotation=(0, 0, side * math.pi/4)
        )

        # Hand
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=scale*0.04,
            location=(side * scale*0.48, 0, scale*0.22)
        )

    # Legs
    for side in [-1, 1]:
        # Upper leg
        bpy.ops.mesh.primitive_cylinder_add(
            radius=scale*0.08,
            depth=scale*0.35,
            location=(side * scale*0.08, 0, scale*0.18)
        )

        # Lower leg
        bpy.ops.mesh.primitive_cylinder_add(
            radius=scale*0.06,
            depth=scale*0.3,
            location=(side * scale*0.08, 0, scale*-0.15)
        )

        # Foot
        bpy.ops.mesh.primitive_cube_add(
            size=scale*0.15,
            location=(side * scale*0.08, scale*0.08, scale*-0.32)
        )
        foot = bpy.context.object
        foot.scale = (0.6, 1.5, 0.4)

generate_character()
"#,
                    scale, style
                )
            }
            OrganicType::Animal => {
                let animal_type = Self::get_string_param(parameters, "type", "dog");
                let scale = Self::get_float_param(parameters, "scale", 1.0);

                match animal_type.as_str() {
                    "dog" => format!(
                        r#"# Generate dog model
def generate_dog(scale={}):
    # Body
    bpy.ops.mesh.primitive_cube_add(size=scale, location=(0, 0, scale*0.4))
    body = bpy.context.object
    body.scale = (1.5, 0.8, 0.8)

    # Head
    bpy.ops.mesh.primitive_cube_add(size=scale*0.6, location=(scale*0.9, 0, scale*0.5))
    head = bpy.context.object
    head.scale = (0.8, 0.7, 0.8)

    # Snout
    bpy.ops.mesh.primitive_cylinder_add(
        radius=scale*0.15,
        depth=scale*0.3,
        location=(scale*1.35, 0, scale*0.45),
        rotation=(0, math.pi/2, 0)
    )

    # Legs
    for x in [-1, 1]:
        for y in [-1, 1]:
            bpy.ops.mesh.primitive_cylinder_add(
                radius=scale*0.08,
                depth=scale*0.6,
                location=(x*scale*0.6, y*scale*0.3, scale*0.1)
            )

    # Tail
    bpy.ops.mesh.primitive_cylinder_add(
        radius=scale*0.05,
        depth=scale*0.4,
        location=(-scale*0.9, 0, scale*0.6),
        rotation=(math.pi/4, 0, 0)
    )

    # Ears
    for side in [-1, 1]:
        bpy.ops.mesh.primitive_plane_add(
            size=scale*0.2,
            location=(scale*0.85, side*scale*0.25, scale*0.75),
            rotation=(0, 0, side*math.pi/6)
        )

generate_dog()
"#,
                        scale
                    ),
                    _ => format!(
                        r#"# Generate generic animal
def generate_animal():
    bpy.ops.mesh.primitive_cube_add(size={})

generate_animal()
"#,
                        scale
                    ),
                }
            }
        }
    }

    /// Generate mechanical-type parametric scripts
    fn generate_mechanical_script(
        &self,
        mech_type: &MechanicalType,
        parameters: &HashMap<String, Value>,
    ) -> String {
        match mech_type {
            MechanicalType::Robot => {
                let scale = Self::get_float_param(parameters, "scale", 1.0);

                format!(
                    r#"# Generate robot model
def generate_robot(scale={}):
    # Body
    bpy.ops.mesh.primitive_cube_add(size=scale, location=(0, 0, scale))
    body = bpy.context.object
    body.scale = (1, 0.8, 1.5)

    # Head
    bpy.ops.mesh.primitive_uv_sphere_add(radius=scale*0.4, location=(0, 0, scale*2))
    head = bpy.context.object

    # Arms
    for side in [-1, 1]:
        # Upper arm
        bpy.ops.mesh.primitive_cylinder_add(
            radius=scale*0.1,
            depth=scale*0.8,
            location=(side * scale*0.7, 0, scale*1.5),
            rotation=(0, math.pi/2, 0)
        )

        # Lower arm
        bpy.ops.mesh.primitive_cylinder_add(
            radius=scale*0.08,
            depth=scale*0.7,
            location=(side * scale*1.2, 0, scale*1),
            rotation=(0, math.pi/2, 0)
        )

        # Hand
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=scale*0.12,
            location=(side * scale*1.6, 0, scale*1)
        )

    # Legs
    for side in [-1, 1]:
        # Upper leg
        bpy.ops.mesh.primitive_cylinder_add(
            radius=scale*0.12,
            depth=scale*0.9,
            location=(side * scale*0.3, 0, scale*0.1)
        )

        # Lower leg
        bpy.ops.mesh.primitive_cylinder_add(
            radius=scale*0.1,
            depth=scale*0.8,
            location=(side * scale*0.3, 0, scale*-0.5)
        )

        # Foot
        bpy.ops.mesh.primitive_cube_add(
            size=scale*0.3,
            location=(side * scale*0.3, scale*0.2, scale*-0.9)
        )
        foot = bpy.context.object
        foot.scale = (1, 2, 0.5)

generate_robot()
"#,
                    scale
                )
            }
            MechanicalType::Vehicle => {
                let vehicle_type = Self::get_string_param(parameters, "type", "car");
                let scale = Self::get_float_param(parameters, "scale", 1.0);

                match vehicle_type.as_str() {
                    "car" => format!(
                        r#"# Generate car model
def generate_car(scale={}):
    # Body
    bpy.ops.mesh.primitive_cube_add(size=scale, location=(0, 0, scale*0.3))
    body = bpy.context.object
    body.scale = (2, 1, 0.6)

    # Cabin
    bpy.ops.mesh.primitive_cube_add(size=scale*0.8, location=(scale*0.2, 0, scale*0.7))
    cabin = bpy.context.object
    cabin.scale = (1, 0.9, 0.8)

    # Hood
    bpy.ops.mesh.primitive_cube_add(size=scale*0.6, location=(-scale*0.7, 0, scale*0.35))
    hood = bpy.context.object
    hood.scale = (1, 0.9, 0.4)

    # Wheels
    for x in [-1, 1]:
        for y in [-1, 1]:
            bpy.ops.mesh.primitive_cylinder_add(
                radius=scale*0.25,
                depth=scale*0.15,
                location=(x*scale*0.8, y*scale*0.55, scale*0.25),
                rotation=(math.pi/2, 0, 0)
            )

    # Windshield
    bpy.ops.mesh.primitive_plane_add(
        size=scale*0.7,
        location=(scale*0.5, 0, scale*0.8),
        rotation=(math.pi/6, 0, 0)
    )

generate_car()
"#,
                        scale
                    ),
                    "airplane" => format!(
                        r#"# Generate airplane model
def generate_airplane(scale={}):
    # Fuselage
    bpy.ops.mesh.primitive_cylinder_add(
        radius=scale*0.15,
        depth=scale*3,
        location=(0, 0, scale*0.5),
        rotation=(0, math.pi/2, 0)
    )

    # Wings
    bpy.ops.mesh.primitive_cube_add(size=scale, location=(0, 0, scale*0.4))
    wings = bpy.context.object
    wings.scale = (3, 0.8, 0.1)

    # Tail
    bpy.ops.mesh.primitive_cube_add(size=scale*0.5, location=(-scale*1.3, 0, scale*0.8))
    tail = bpy.context.object
    tail.scale = (0.3, 1, 1.2)

    # Propeller
    bpy.ops.mesh.primitive_cylinder_add(
        radius=scale*0.4,
        depth=scale*0.05,
        location=(scale*1.5, 0, scale*0.5),
        rotation=(0, math.pi/2, 0)
    )

generate_airplane()
"#,
                        scale
                    ),
                    _ => format!(
                        r#"# Generate generic vehicle
def generate_vehicle():
    bpy.ops.mesh.primitive_cube_add(size={})

generate_vehicle()
"#,
                        scale
                    ),
                }
            }
            MechanicalType::Tool => {
                let tool_type = Self::get_string_param(parameters, "type", "hammer");
                let scale = Self::get_float_param(parameters, "scale", 1.0);

                match tool_type.as_str() {
                    "hammer" => format!(
                        r#"# Generate hammer model
def generate_hammer(scale={}):
    # Handle
    bpy.ops.mesh.primitive_cylinder_add(
        radius=scale*0.03,
        depth=scale*0.8,
        location=(0, 0, scale*0.4)
    )

    # Head
    bpy.ops.mesh.primitive_cube_add(size=scale*0.3, location=(0, 0, scale*0.85))
    head = bpy.context.object
    head.scale = (2, 0.6, 0.8)

    # Claw (back of hammer)
    bpy.ops.mesh.primitive_cube_add(size=scale*0.2, location=(-scale*0.35, 0, scale*0.85))
    claw = bpy.context.object
    claw.scale = (0.5, 0.3, 1.2)

generate_hammer()
"#,
                        scale
                    ),
                    "screwdriver" => format!(
                        r#"# Generate screwdriver model
def generate_screwdriver(scale={}):
    # Handle
    bpy.ops.mesh.primitive_cylinder_add(
        radius=scale*0.05,
        depth=scale*0.6,
        location=(0, 0, scale*0.3)
    )

    # Shaft
    bpy.ops.mesh.primitive_cylinder_add(
        radius=scale*0.02,
        depth=scale*0.8,
        location=(0, 0, scale*0.7)
    )

    # Tip
    bpy.ops.mesh.primitive_cube_add(size=scale*0.05, location=(0, 0, scale*1.1))
    tip = bpy.context.object
    tip.scale = (0.1, 1, 0.2)

generate_screwdriver()
"#,
                        scale
                    ),
                    _ => format!(
                        r#"# Generate generic tool
def generate_tool():
    bpy.ops.mesh.primitive_cube_add(size={})

generate_tool()
"#,
                        scale
                    ),
                }
            }
            MechanicalType::Machine => {
                let machine_type = Self::get_string_param(parameters, "type", "gear");
                let scale = Self::get_float_param(parameters, "scale", 1.0);

                match machine_type.as_str() {
                    "gear" => format!(
                        r#"# Generate gear model
def generate_gear(scale={}):
    import bmesh

    # Create basic gear shape
    bpy.ops.mesh.primitive_cylinder_add(radius=scale, depth=scale*0.2)
    gear = bpy.context.object

    # Enter edit mode to add gear teeth
    bpy.context.view_layer.objects.active = gear
    bpy.ops.object.mode_set(mode='EDIT')

    # Get bmesh representation
    bm = bmesh.from_mesh(gear.data)

    # Add inset for gear teeth (simplified)
    for face in bm.faces[:]:
        if abs(face.normal.z) < 0.1:  # Side faces
            bmesh.ops.inset_faces(bm, faces=[face], thickness=0.1, depth=0.05)

    # Update mesh
    bmesh.update_edit_mesh(gear.data)
    bpy.ops.object.mode_set(mode='OBJECT')

    # Add center hole
    bpy.ops.mesh.primitive_cylinder_add(
        radius=scale*0.2,
        depth=scale*0.3,
        location=(0, 0, 0)
    )
    hole = bpy.context.object

    # Boolean difference to create hole
    bool_modifier = gear.modifiers.new('Boolean', 'BOOLEAN')
    bool_modifier.operation = 'DIFFERENCE'
    bool_modifier.object = hole

generate_gear()
"#,
                        scale
                    ),
                    _ => format!(
                        r#"# Generate generic machine
def generate_machine():
    bpy.ops.mesh.primitive_cube_add(size={})

generate_machine()
"#,
                        scale
                    ),
                }
            }
        }
    }

    /// Generate abstract-type parametric scripts
    fn generate_abstract_script(
        &self,
        abstract_type: &AbstractType,
        parameters: &HashMap<String, Value>,
    ) -> String {
        match abstract_type {
            AbstractType::Fractal => {
                let iterations = Self::get_int_param(parameters, "iterations", 4);
                let scale_factor = Self::get_float_param(parameters, "scale_factor", 0.7);

                format!(
                    r#"# Generate fractal structure
def generate_fractal(iterations={}, scale_factor={}):
    import random

    def create_fractal_branch(location, scale, depth):
        if depth <= 0:
            return

        # Create current element
        bpy.ops.mesh.primitive_cube_add(size=scale, location=location)

        # Create branches
        for i in range(3):
            angle = (i * 2 * math.pi / 3) + random.uniform(-0.3, 0.3)
            new_scale = scale * scale_factor
            offset = scale * 1.5

            new_location = (
                location[0] + math.cos(angle) * offset,
                location[1] + math.sin(angle) * offset,
                location[2] + new_scale
            )

            create_fractal_branch(new_location, new_scale, depth - 1)

    create_fractal_branch((0, 0, 0), 1.0, iterations)

generate_fractal()
"#,
                    iterations, scale_factor
                )
            }
            AbstractType::Mathematical => {
                let formula = Self::get_string_param(parameters, "formula", "sin(x)*cos(y)");
                let resolution = Self::get_int_param(parameters, "resolution", 50);

                format!(
                    r#"# Generate mathematical surface
def generate_math_surface(formula='{}', resolution={}):
    # Create vertices for mathematical function
    vertices = []
    faces = []

    for i in range(resolution):
        for j in range(resolution):
            x = (i - resolution/2) / 10.0
            y = (j - resolution/2) / 10.0

            # Evaluate mathematical function
            try:
                z = eval(formula.replace('x', str(x)).replace('y', str(y)))
            except:
                z = 0

            vertices.append((x, y, z))

    # Create faces
    for i in range(resolution-1):
        for j in range(resolution-1):
            # Create quad face
            v1 = i * resolution + j
            v2 = i * resolution + j + 1
            v3 = (i + 1) * resolution + j + 1
            v4 = (i + 1) * resolution + j
            faces.append([v1, v2, v3, v4])

    # Create mesh
    mesh = bpy.data.meshes.new('MathSurface')
    mesh.from_pydata(vertices, [], faces)
    mesh.update()

    # Create object
    obj = bpy.data.objects.new('MathSurface', mesh)
    bpy.context.collection.objects.link(obj)

generate_math_surface()
"#,
                    formula, resolution
                )
            }
            AbstractType::Procedural => {
                let complexity = Self::get_int_param(parameters, "complexity", 5);
                let seed = Self::get_int_param(parameters, "seed", 42);

                format!(
                    r#"# Generate procedural structure
def generate_procedural(complexity={}, seed={}):
    import random
    random.seed(seed)

    def recursive_structure(location, scale, depth):
        if depth <= 0 or scale < 0.1:
            return

        # Create main element
        shape_type = random.choice(['cube', 'sphere', 'cylinder'])

        if shape_type == 'cube':
            bpy.ops.mesh.primitive_cube_add(size=scale, location=location)
        elif shape_type == 'sphere':
            bpy.ops.mesh.primitive_uv_sphere_add(radius=scale/2, location=location)
        else:  # cylinder
            bpy.ops.mesh.primitive_cylinder_add(radius=scale/3, depth=scale, location=location)

        # Add child elements
        num_children = random.randint(1, 4)
        for i in range(num_children):
            angle = (i * 2 * math.pi / num_children) + random.uniform(-0.5, 0.5)
            distance = scale * random.uniform(1.2, 2.0)
            new_scale = scale * random.uniform(0.4, 0.8)

            child_location = (
                location[0] + math.cos(angle) * distance,
                location[1] + math.sin(angle) * distance,
                location[2] + random.uniform(-scale, scale)
            )

            recursive_structure(child_location, new_scale, depth - 1)

    recursive_structure((0, 0, 0), 2.0, complexity)

generate_procedural()
"#,
                    complexity, seed
                )
            }
            AbstractType::Artistic => {
                let style = Self::get_string_param(parameters, "style", "modern");
                let elements = Self::get_int_param(parameters, "elements", 8);

                format!(
                    r#"# Generate artistic structure
def generate_artistic(style='{}', elements={}):
    import random
    random.seed(123)

    if style == 'modern':
        # Modern abstract art with geometric shapes
        for i in range(elements):
            shape_choice = random.choice(['cube', 'sphere', 'torus'])
            location = (
                random.uniform(-5, 5),
                random.uniform(-5, 5),
                random.uniform(-2, 8)
            )
            rotation = (
                random.uniform(0, math.pi),
                random.uniform(0, math.pi),
                random.uniform(0, math.pi)
            )
            scale = random.uniform(0.5, 2.0)

            if shape_choice == 'cube':
                bpy.ops.mesh.primitive_cube_add(size=scale, location=location, rotation=rotation)
            elif shape_choice == 'sphere':
                bpy.ops.mesh.primitive_uv_sphere_add(radius=scale/2, location=location)
            else:  # torus
                bpy.ops.mesh.primitive_torus_add(
                    major_radius=scale*0.8,
                    minor_radius=scale*0.3,
                    location=location,
                    rotation=rotation
                )

    elif style == 'organic':
        # Organic flowing forms
        for i in range(elements):
            location = (
                random.uniform(-3, 3),
                random.uniform(-3, 3),
                random.uniform(0, 5)
            )

            bpy.ops.mesh.primitive_uv_sphere_add(radius=random.uniform(0.3, 1.2), location=location)
            obj = bpy.context.object

            # Add some organic deformation
            obj.scale = (
                random.uniform(0.5, 2.0),
                random.uniform(0.5, 2.0),
                random.uniform(0.8, 1.5)
            )

generate_artistic()
"#,
                    style, elements
                )
            }
        }
    }

    /// Helper function to get float parameter with default
    fn get_float_param(params: &HashMap<String, Value>, key: &str, default: f32) -> f32 {
        params.get(key).and_then(|v| v.as_f64()).map(|f| f as f32).unwrap_or(default)
    }

    /// Helper function to get integer parameter with default
    fn get_int_param(params: &HashMap<String, Value>, key: &str, default: i32) -> i32 {
        params.get(key).and_then(|v| v.as_i64()).map(|i| i as i32).unwrap_or(default)
    }

    /// Helper function to get string parameter with default
    fn get_string_param(params: &HashMap<String, Value>, key: &str, default: &str) -> String {
        params
            .get(key)
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| default.to_string())
    }

    /// Generate scene script with full environmental setup
    fn generate_scene_script(
        &self,
        objects: &[SceneObject],
        lighting: &LightingSetup,
        camera: &CameraSetup,
    ) -> String {
        let mut script = String::new();
        script.push_str("import bpy\nimport mathutils\nfrom mathutils import Vector\n\n");

        // Clear existing scene
        script.push_str("# Clear existing mesh objects\n");
        script.push_str("bpy.ops.object.select_all(action='SELECT')\n");
        script.push_str("bpy.ops.object.delete(use_global=False, confirm=False)\n\n");

        // Create scene objects
        script.push_str("# Create scene objects\n");
        for (idx, obj) in objects.iter().enumerate() {
            script.push_str(&format!("# Object {}: {}\n", idx, obj.name));

            // Generate object based on its type
            match &obj.object_type {
                ContentType::Primitive { shape, parameters } => {
                    script.push_str(&self.generate_primitive_script(shape, parameters));
                }
                ContentType::Parametric { model_type, parameters } => {
                    script.push_str(&self.generate_parametric_script(model_type, parameters));
                }
                _ => {
                    script.push_str("bpy.ops.mesh.primitive_cube_add()\n");
                }
            }

            // Apply transform
            script.push_str(&format!(
                "obj = bpy.context.object\nobj.name = '{}'\nobj.location = ({}, {}, \
                 {})\nobj.rotation_euler = ({}, {}, {})\nobj.scale = ({}, {}, {})\n\n",
                obj.name,
                obj.transform.location[0],
                obj.transform.location[1],
                obj.transform.location[2],
                obj.transform.rotation[0],
                obj.transform.rotation[1],
                obj.transform.rotation[2],
                obj.transform.scale[0],
                obj.transform.scale[1],
                obj.transform.scale[2]
            ));

            // Apply material if specified
            if let Some(material_name) = &obj.material {
                script.push_str(&format!(
                    r#"# Apply material: {}
if '{}' in bpy.data.materials:
    obj.data.materials.append(bpy.data.materials['{}'])

"#,
                    material_name, material_name, material_name
                ));
            }
        }

        // Setup lighting
        script.push_str("# Setup lighting\n");
        for (idx, light) in lighting.lights.iter().enumerate() {
            let light_type_str = match light.light_type {
                LightType::Sun => "SUN",
                LightType::Point => "POINT",
                LightType::Spot => "SPOT",
                LightType::Area => "AREA",
            };

            script.push_str(&format!(
                "bpy.ops.object.light_add(type='{}', location=({}, {}, {}))\nlight_{} = \
                 bpy.context.object\nlight_{}.data.energy = {}\nlight_{}.data.color = ({}, {}, \
                 {})\n\n",
                light_type_str,
                light.location[0],
                light.location[1],
                light.location[2],
                idx,
                idx,
                light.energy,
                idx,
                light.color[0],
                light.color[1],
                light.color[2]
            ));
        }

        // Environment setup
        if let Some(env) = &lighting.environment {
            script.push_str(&format!(
                "# Environment setup\nworld = bpy.context.scene.world\nworld.use_nodes = \
                 True\nenv_node = world.node_tree.nodes.new(type='ShaderNodeTexEnvironment')\n# \
                 Load environment texture: {}\n\n",
                env
            ));
        }

        // Global illumination
        if lighting.global_illumination {
            script.push_str(
                "# Enable global illumination\nbpy.context.scene.render.engine = \
                 'CYCLES'\nbpy.context.scene.cycles.samples = 128\n\n",
            );
        }

        // Setup camera
        script.push_str(&format!(
            "# Setup camera\nbpy.ops.object.camera_add(location=({}, {}, {}))\ncamera = \
             bpy.context.object\ncamera.rotation_euler = ({}, {}, {})\ncamera.data.lens = \
             {}\ncamera.data.clip_start = {}\ncamera.data.clip_end = {}\nbpy.context.scene.camera \
             = camera\n\n",
            camera.location[0],
            camera.location[1],
            camera.location[2],
            camera.rotation[0],
            camera.rotation[1],
            camera.rotation[2],
            camera.focal_length,
            camera.clip_start,
            camera.clip_end
        ));

        script
    }

    /// Generate animation script with keyframe interpolation
    fn generate_animation_script(
        &self,
        objects: &[AnimatedObject],
        duration_frames: u32,
        frame_rate: f32,
    ) -> String {
        let mut script = String::new();
        script.push_str("import bpy\nimport mathutils\nfrom mathutils import Vector\n\n");

        // Set animation frame range
        script.push_str(&format!(
            "# Set animation settings\nbpy.context.scene.frame_start = \
             1\nbpy.context.scene.frame_end = {}\nbpy.context.scene.render.fps = {}\n\n",
            duration_frames, frame_rate as i32
        ));

        // Create animated objects
        for (obj_idx, animated_obj) in objects.iter().enumerate() {
            script.push_str(&format!(
                "# Animated Object {}: {}\n",
                obj_idx, animated_obj.object.name
            ));

            // Create the base object
            match &animated_obj.object.object_type {
                ContentType::Primitive { shape, parameters } => {
                    script.push_str(&self.generate_primitive_script(shape, parameters));
                }
                ContentType::Parametric { model_type, parameters } => {
                    script.push_str(&self.generate_parametric_script(model_type, parameters));
                }
                _ => {
                    script.push_str("bpy.ops.mesh.primitive_cube_add()\n");
                }
            }

            // Set initial transform
            script.push_str(&format!(
                "obj_{} = bpy.context.object\nobj_{}.name = '{}'\nobj_{}.location = ({}, {}, \
                 {})\nobj_{}.rotation_euler = ({}, {}, {})\nobj_{}.scale = ({}, {}, {})\n\n",
                obj_idx,
                obj_idx,
                animated_obj.object.name,
                obj_idx,
                animated_obj.object.transform.location[0],
                animated_obj.object.transform.location[1],
                animated_obj.object.transform.location[2],
                obj_idx,
                animated_obj.object.transform.rotation[0],
                animated_obj.object.transform.rotation[1],
                animated_obj.object.transform.rotation[2],
                obj_idx,
                animated_obj.object.transform.scale[0],
                animated_obj.object.transform.scale[1],
                animated_obj.object.transform.scale[2]
            ));

            // Set keyframes
            script.push_str(&format!("# Keyframes for object {}\n", obj_idx));
            for keyframe in &animated_obj.keyframes {
                script.push_str(&format!(
                    "# Frame {}\nbpy.context.scene.frame_set({})\nobj_{}.location = ({}, {}, \
                     {})\nobj_{}.rotation_euler = ({}, {}, {})\nobj_{}.scale = ({}, {}, \
                     {})\nobj_{}.keyframe_insert(data_path='location', \
                     index=-1)\nobj_{}.keyframe_insert(data_path='rotation_euler', \
                     index=-1)\nobj_{}.keyframe_insert(data_path='scale', index=-1)\n",
                    keyframe.frame,
                    keyframe.frame,
                    obj_idx,
                    keyframe.transform.location[0],
                    keyframe.transform.location[1],
                    keyframe.transform.location[2],
                    obj_idx,
                    keyframe.transform.rotation[0],
                    keyframe.transform.rotation[1],
                    keyframe.transform.rotation[2],
                    obj_idx,
                    keyframe.transform.scale[0],
                    keyframe.transform.scale[1],
                    keyframe.transform.scale[2],
                    obj_idx,
                    obj_idx,
                    obj_idx
                ));

                // Set custom properties from keyframe
                for (prop_name, prop_value) in &keyframe.properties {
                    if let Some(float_val) = prop_value.as_f64() {
                        script.push_str(&format!(
                            "obj_{}['{}'] = {}\nobj_{}.keyframe_insert(data_path='[\"{}\"]')\n",
                            obj_idx, prop_name, float_val, obj_idx, prop_name
                        ));
                    }
                }

                script.push_str("\n");
            }

            // Set interpolation mode to smooth
            script.push_str(&format!(
                r#"# Set smooth interpolation for object {}
if obj_{}.animation_data:
    for fcurve in obj_{}.animation_data.action.fcurves:
        for keyframe in fcurve.keyframe_points:
            keyframe.interpolation = 'BEZIER'
            keyframe.handle_left_type = 'AUTO'
            keyframe.handle_right_type = 'AUTO'

"#,
                obj_idx, obj_idx, obj_idx
            ));
        }

        // Add camera animation for cinematic effect
        script.push_str(
            r#"# Add camera movement for cinematic effect
if bpy.context.scene.camera:
    camera = bpy.context.scene.camera
    # Camera orbit animation
    import math
    for frame in range(1, int(bpy.context.scene.frame_end) + 1):
        bpy.context.scene.frame_set(frame)
        angle = (frame / bpy.context.scene.frame_end) * 2 * math.pi
        radius = 10.0
        camera.location.x = radius * math.cos(angle)
        camera.location.y = radius * math.sin(angle)
        camera.location.z = 5.0 + 2.0 * math.sin(angle * 2)

        # Point camera at origin
        direction = mathutils.Vector((0, 0, 0)) - camera.location
        camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

        camera.keyframe_insert(data_path='location', index=-1)
        camera.keyframe_insert(data_path='rotation_euler', index=-1)

"#,
        );

        script.push_str("# Set frame back to 1\nbpy.context.scene.frame_set(1)\n");
        script
    }

    /// Generate material script with realistic shader networks
    fn generate_material_script(
        &self,
        material_type: &MaterialType,
        properties: &HashMap<String, Value>,
    ) -> String {
        let mut script = String::new();
        script.push_str("import bpy\n\n");

        let material_name = Self::get_string_param(properties, "name", "Generated_Material");

        script.push_str(&format!(
            "# Create material: {}\nmat = bpy.data.materials.new(name='{}')\nmat.use_nodes = \
             True\nnodes = mat.node_tree.nodes\nlinks = mat.node_tree.links\n\n# Clear existing \
             nodes\nnodes.clear()\n\n",
            material_name, material_name
        ));

        match material_type {
            MaterialType::Principled => {
                let base_color = Self::get_color_param(properties, "base_color", [0.8, 0.8, 0.8]);
                let metallic = Self::get_float_param(properties, "metallic", 0.0);
                let roughness = Self::get_float_param(properties, "roughness", 0.5);
                let ior = Self::get_float_param(properties, "ior", 1.45);

                script.push_str(&format!(
                    "# Create Principled BSDF material\noutput = \
                     nodes.new(type='ShaderNodeOutputMaterial')\nprincipled = \
                     nodes.new(type='ShaderNodeBsdfPrincipled')\n\n# Set material \
                     properties\nprincipled.inputs['Base Color'].default_value = ({}, {}, {}, \
                     1.0)\nprincipled.inputs['Metallic'].default_value = \
                     {}\nprincipled.inputs['Roughness'].default_value = \
                     {}\nprincipled.inputs['IOR'].default_value = {}\n\n# Connect \
                     nodes\nlinks.new(principled.outputs['BSDF'], output.inputs['Surface'])\n\n# \
                     Position nodes\noutput.location = (400, 0)\nprincipled.location = (0, 0)\n\n",
                    base_color[0], base_color[1], base_color[2], metallic, roughness, ior
                ));
            }
            MaterialType::Emission => {
                let emission_color =
                    Self::get_color_param(properties, "emission_color", [1.0, 1.0, 1.0]);
                let strength = Self::get_float_param(properties, "strength", 1.0);

                script.push_str(&format!(
                    "# Create Emission material\noutput = \
                     nodes.new(type='ShaderNodeOutputMaterial')\nemission = \
                     nodes.new(type='ShaderNodeEmission')\n\n# Set emission \
                     properties\nemission.inputs['Color'].default_value = ({}, {}, {}, \
                     1.0)\nemission.inputs['Strength'].default_value = {}\n\n# Connect \
                     nodes\nlinks.new(emission.outputs['Emission'], \
                     output.inputs['Surface'])\n\n# Position nodes\noutput.location = (400, \
                     0)\nemission.location = (0, 0)\n\n",
                    emission_color[0], emission_color[1], emission_color[2], strength
                ));
            }
            MaterialType::Glass => {
                let glass_color = Self::get_color_param(properties, "color", [0.9, 0.9, 1.0]);
                let roughness = Self::get_float_param(properties, "roughness", 0.0);
                let ior = Self::get_float_param(properties, "ior", 1.52);

                script.push_str(&format!(
                    "# Create Glass material\noutput = \
                     nodes.new(type='ShaderNodeOutputMaterial')\nglass = \
                     nodes.new(type='ShaderNodeBsdfGlass')\n\n# Set glass \
                     properties\nglass.inputs['Color'].default_value = ({}, {}, {}, \
                     1.0)\nglass.inputs['Roughness'].default_value = \
                     {}\nglass.inputs['IOR'].default_value = {}\n\n# Connect \
                     nodes\nlinks.new(glass.outputs['BSDF'], output.inputs['Surface'])\n\n# \
                     Position nodes\noutput.location = (400, 0)\nglass.location = (0, 0)\n\n",
                    glass_color[0], glass_color[1], glass_color[2], roughness, ior
                ));
            }
            MaterialType::Metal => {
                let base_color = Self::get_color_param(properties, "base_color", [0.7, 0.7, 0.8]);
                let roughness = Self::get_float_param(properties, "roughness", 0.1);

                script.push_str(&format!(
                    "# Create Metallic material\noutput = \
                     nodes.new(type='ShaderNodeOutputMaterial')\nprincipled = \
                     nodes.new(type='ShaderNodeBsdfPrincipled')\n\n# Set metallic \
                     properties\nprincipled.inputs['Base Color'].default_value = ({}, {}, {}, \
                     1.0)\nprincipled.inputs['Metallic'].default_value = \
                     1.0\nprincipled.inputs['Roughness'].default_value = {}\n\n# Connect \
                     nodes\nlinks.new(principled.outputs['BSDF'], output.inputs['Surface'])\n\n# \
                     Position nodes\noutput.location = (400, 0)\nprincipled.location = (0, 0)\n\n",
                    base_color[0], base_color[1], base_color[2], roughness
                ));
            }
            MaterialType::Subsurface => {
                let base_color = Self::get_color_param(properties, "base_color", [0.8, 0.6, 0.5]);
                let subsurface_color =
                    Self::get_color_param(properties, "subsurface_color", [0.9, 0.7, 0.6]);
                let subsurface = Self::get_float_param(properties, "subsurface", 0.3);

                script.push_str(&format!(
                    "# Create Subsurface Scattering material\noutput = \
                     nodes.new(type='ShaderNodeOutputMaterial')\nprincipled = \
                     nodes.new(type='ShaderNodeBsdfPrincipled')\n\n# Set subsurface \
                     properties\nprincipled.inputs['Base Color'].default_value = ({}, {}, {}, \
                     1.0)\nprincipled.inputs['Subsurface'].default_value = \
                     {}\nprincipled.inputs['Subsurface Color'].default_value = ({}, {}, {}, \
                     1.0)\nprincipled.inputs['Subsurface Radius'].default_value = (1.0, 0.2, \
                     0.1)\n\n# Connect nodes\nlinks.new(principled.outputs['BSDF'], \
                     output.inputs['Surface'])\n\n# Position nodes\noutput.location = (400, \
                     0)\nprincipled.location = (0, 0)\n\n",
                    base_color[0],
                    base_color[1],
                    base_color[2],
                    subsurface,
                    subsurface_color[0],
                    subsurface_color[1],
                    subsurface_color[2]
                ));
            }
            MaterialType::Toon => {
                let base_color = Self::get_color_param(properties, "base_color", [0.8, 0.4, 0.2]);

                script.push_str(&format!(
                    "# Create Toon material\noutput = \
                     nodes.new(type='ShaderNodeOutputMaterial')\ntoon = \
                     nodes.new(type='ShaderNodeBsdfToon')\n\n# Set toon \
                     properties\ntoon.inputs['Color'].default_value = ({}, {}, {}, \
                     1.0)\ntoon.inputs['Size'].default_value = \
                     0.9\ntoon.inputs['Smooth'].default_value = 0.0\n\n# Connect \
                     nodes\nlinks.new(toon.outputs['BSDF'], output.inputs['Surface'])\n\n# \
                     Position nodes\noutput.location = (400, 0)\ntoon.location = (0, 0)\n\n",
                    base_color[0], base_color[1], base_color[2]
                ));
            }
        }

        // Add texture if specified
        if properties.contains_key("texture_path") {
            let texture_path = Self::get_string_param(properties, "texture_path", "");
            script.push_str(&format!(
                r#"# Add texture
if '{}' != '':
    texture = nodes.new(type='ShaderNodeTexImage')
    texture.location = (-400, 0)

    try:
        texture.image = bpy.data.images.load('{}')
        links.new(texture.outputs['Color'], principled.inputs['Base Color'])
    except:
        print('Could not load texture: {}')

"#,
                texture_path, texture_path, texture_path
            ));
        }

        script.push_str("# Assign material to active object\n");
        script.push_str("if bpy.context.active_object:\n");
        script.push_str("    bpy.context.active_object.data.materials.append(mat)\n\n");

        script
    }

    /// Helper function to get color parameter with default
    fn get_color_param(params: &HashMap<String, Value>, key: &str, default: [f32; 3]) -> [f32; 3] {
        params
            .get(key)
            .and_then(|v| v.as_array())
            .and_then(|arr| {
                if arr.len() >= 3 {
                    Some([
                        arr[0].as_f64().unwrap_or(default[0] as f64) as f32,
                        arr[1].as_f64().unwrap_or(default[1] as f64) as f32,
                        arr[2].as_f64().unwrap_or(default[2] as f64) as f32,
                    ])
                } else {
                    None
                }
            })
            .unwrap_or(default)
    }
}

impl Clone for BlenderIntegration {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            cognitive_system: self.cognitive_system.clone(),
            memory: self.memory.clone(),
            projects: self.projects.clone(),
            event_tx: self.event_tx.clone(),
            assets: self.assets.clone(),
            script_templates: self.script_templates.clone(),
        }
    }
}

impl BlenderIntegration {
    /// Parse Blender script output for execution results and error analysis
    fn parse_blender_output(&self, stdout: &str, stderr: &str) -> Result<BlenderExecutionResult> {
        let mut execution_result = BlenderExecutionResult {
            success: true,
            output_data: stdout.to_string(),
            warnings: Vec::new(),
            errors: Vec::new(),
            performance_metrics: None,
            generated_assets: Vec::new(),
        };
        
        // Parse stdout for Blender-specific output patterns
        for line in stdout.lines() {
            // Look for render completion messages
            if line.contains("Saved:") && (line.contains(".png") || line.contains(".jpg") || line.contains(".blend")) {
                if let Some(asset_path) = self.extract_asset_path(line) {
                    execution_result.generated_assets.push(asset_path);
                }
            }
            
            // Look for performance metrics
            if line.contains("Time:") && line.contains("(Saving") {
                if let Some(render_time) = self.extract_render_time(line) {
                    execution_result.performance_metrics = Some(BlenderPerformanceMetrics {
                        render_time_ms: render_time,
                        memory_usage_mb: 0, // Would need additional parsing
                        samples_completed: 0, // Would need additional parsing
                    });
                }
            }
            
            // Look for script completion messages
            if line.contains("Blender quit") {
                debug!("Blender script completed successfully");
            }
        }
        
        // Parse stderr for warnings and errors
        for line in stderr.lines() {
            if line.contains("Warning") || line.contains("WARN") {
                execution_result.warnings.push(line.to_string());
            } else if line.contains("Error") || line.contains("ERROR") || line.contains("Traceback") {
                execution_result.errors.push(line.to_string());
                execution_result.success = false;
            }
        }
        
        // Log summary
        debug!(
            "Blender execution result: success={}, warnings={}, errors={}, assets={}",
            execution_result.success,
            execution_result.warnings.len(),
            execution_result.errors.len(),
            execution_result.generated_assets.len()
        );
        
        Ok(execution_result)
    }
    
    /// Extract asset path from Blender output line
    fn extract_asset_path(&self, line: &str) -> Option<String> {
        // Example: "Saved: '/path/to/output.png'"
        if let Some(start) = line.find("'") {
            if let Some(end) = line.rfind("'") {
                if end > start {
                    return Some(line[start + 1..end].to_string());
                }
            }
        }
        None
    }
    
    /// Extract render time from Blender output line
    fn extract_render_time(&self, line: &str) -> Option<u64> {
        // Example: "Time: 00:02.34 (Saving: 00:00.05)"
        if let Some(time_start) = line.find("Time: ") {
            let time_str = &line[time_start + 6..];
            if let Some(space_pos) = time_str.find(' ') {
                let time_part = &time_str[..space_pos];
                // Parse MM:SS.ms format to milliseconds
                if let Some(colon_pos) = time_part.find(':') {
                    let minutes: u64 = time_part[..colon_pos].parse().unwrap_or(0);
                    let seconds_part = &time_part[colon_pos + 1..];
                    let seconds: f64 = seconds_part.parse().unwrap_or(0.0);
                    return Some((minutes * 60 * 1000) + (seconds * 1000.0) as u64);
                }
            }
        }
        None
    }
}

/// Integration statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlenderStats {
    pub total_projects: usize,
    pub total_assets: usize,
    pub render_time_total_ms: u64,
    pub successful_generations: u32,
    pub failed_generations: u32,
    pub cognitive_inspirations: u32,
}

impl BlenderIntegration {
    /// Get integration statistics
    pub async fn get_stats(&self) -> BlenderStats {
        let projects = self.projects.read().await;
        let assets = self.assets.read().await;

        BlenderStats {
            total_projects: projects.len(),
            total_assets: assets.len(),
            render_time_total_ms: 0, // Would track this in real implementation
            successful_generations: 0, // Would track this
            failed_generations: 0,
            cognitive_inspirations: 0,
        }
    }
}

/// Result of Blender script execution
#[derive(Debug, Clone)]
pub struct BlenderExecutionResult {
    pub success: bool,
    pub output_data: String,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
    pub performance_metrics: Option<BlenderPerformanceMetrics>,
    pub generated_assets: Vec<String>,
}

/// Performance metrics from Blender execution
#[derive(Debug, Clone)]
pub struct BlenderPerformanceMetrics {
    pub render_time_ms: u64,
    pub memory_usage_mb: u64,
    pub samples_completed: u32,
}
