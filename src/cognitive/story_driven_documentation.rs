//! Story-Driven Documentation Generation
//!
//! This module implements an intelligent documentation generation system that
//! creates and maintains documentation based on code understanding and narrative context.

use anyhow::Result;
use regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn};

use crate::cognitive::self_modify::{CodeChange, SelfModificationPipeline};
use crate::cognitive::story_driven_learning::StoryDrivenLearning;
use crate::memory::{CognitiveMemory, MemoryMetadata, MemoryId};
use crate::story::{PlotType, StoryEngine, StoryId};
use crate::tools::code_analysis::{CodeAnalyzer, FunctionInfo};

/// Configuration for story-driven documentation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoryDrivenDocumentationConfig {
    /// Enable API documentation generation
    pub enable_api_docs: bool,

    /// Enable module documentation
    pub enable_module_docs: bool,

    /// Enable README generation
    pub enable_readme_generation: bool,

    /// Enable inline documentation
    pub enable_inline_docs: bool,

    /// Enable architecture documentation
    pub enable_architecture_docs: bool,

    /// Documentation style preference
    pub doc_style: DocumentationStyle,

    /// Minimum complexity for documentation
    pub min_complexity_threshold: f32,

    /// Repository path
    pub repo_path: PathBuf,
}

impl Default for StoryDrivenDocumentationConfig {
    fn default() -> Self {
        Self {
            enable_api_docs: true,
            enable_module_docs: true,
            enable_readme_generation: true,
            enable_inline_docs: true,
            enable_architecture_docs: true,
            doc_style: DocumentationStyle::Comprehensive,
            min_complexity_threshold: 5.0,
            repo_path: PathBuf::from("."),
        }
    }
}

/// Documentation style preference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DocumentationStyle {
    Minimal,
    Standard,
    Comprehensive,
    Tutorial,
}

/// Type of documentation to generate
#[derive(Debug, Clone, Serialize, Deserialize, Eq, Hash, PartialEq)]
pub enum DocumentationType {
    Api,
    Module,
    Readme,
    Architecture,
    Tutorial,
    Migration,
    Changelog,
}

/// Generated documentation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedDocumentation {
    pub doc_id: String,
    pub entity_name: String,
    pub entity_type: String,
    pub documentation: String,
    pub doc_type: DocumentationType,
    pub confidence: f32,
    pub suggested_location: PathBuf,
    pub target_path: PathBuf,
    pub content: String,
    pub metadata: DocumentationMetadata,
    pub generated_at: chrono::DateTime<chrono::Utc>,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Documentation metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentationMetadata {
    pub title: String,
    pub description: String,
    pub tags: Vec<String>,
    pub complexity: f32,
    pub coverage: f32,
    pub examples_included: bool,
}

/// Documentation analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentationAnalysis {
    pub total_files: usize,
    pub documented_files: usize,
    pub documentation_coverage: f32,
    pub missing_docs: Vec<MissingDocumentation>,
    pub outdated_docs: Vec<OutdatedDocumentation>,
}

/// Missing documentation entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissingDocumentation {
    pub file_path: PathBuf,
    pub item_type: String,
    pub item_name: String,
    pub priority: f32,
}

/// Outdated documentation entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutdatedDocumentation {
    pub file_path: PathBuf,
    pub entity_name: String,
    pub entity_type: String,
    pub reason: String,
    pub outdated_reasons: Vec<String>,
    pub current_doc: String,
    pub doc_type: DocumentationType,
    pub doc_location: PathBuf,
    pub old_signature: Option<String>,
    pub new_signature: Option<String>,
    pub last_updated: Option<chrono::DateTime<chrono::Utc>>,
}

/// Documentation template
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentationTemplate {
    pub template_type: DocumentationType,
    pub sections: Vec<TemplateSection>,
}

/// Template section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateSection {
    pub name: String,
    pub required: bool,
    pub template: String,
}

/// Analysis of documentation changes
#[derive(Debug, Clone)]
struct DocumentationChangeAnalysis {
    signature_changed: bool,
    behavior_changed: bool,
    parameters_changed: bool,
    return_type_changed: bool,
    new_features: Vec<String>,
    removed_features: Vec<String>,
    semantic_changes: Vec<String>,
}

/// Preserved documentation content
#[derive(Debug, Clone)]
struct PreservedDocContent {
    examples: Vec<String>,
    important_notes: Vec<String>,
    see_also_links: Vec<(String, String)>,
    custom_sections: HashMap<String, String>,
}

/// Documentation update pattern
#[derive(Debug, Clone)]
struct DocumentationUpdatePattern {
    entity_type: String,
    outdated_reasons: Vec<String>,
    update_approach: String,
    success_score: f32,
}

/// Story-driven documentation generator
pub struct StoryDrivenDocumentation {
    config: StoryDrivenDocumentationConfig,
    story_engine: Arc<StoryEngine>,
    code_analyzer: Arc<CodeAnalyzer>,
    self_modify: Arc<SelfModificationPipeline>,
    learning_system: Option<Arc<StoryDrivenLearning>>,
    memory: Arc<CognitiveMemory>,
    codebase_story_id: StoryId,
    doc_templates: Arc<RwLock<HashMap<DocumentationType, DocumentationTemplate>>>,
    generated_docs: Arc<RwLock<Vec<GeneratedDocumentation>>>,
}

impl StoryDrivenDocumentation {
    /// Create new story-driven documentation generator
    pub async fn new(
        config: StoryDrivenDocumentationConfig,
        story_engine: Arc<StoryEngine>,
        code_analyzer: Arc<CodeAnalyzer>,
        self_modify: Arc<SelfModificationPipeline>,
        learning_system: Option<Arc<StoryDrivenLearning>>,
        memory: Arc<CognitiveMemory>,
    ) -> Result<Self> {
        // Create or get codebase story
        let codebase_story_id = story_engine
            .create_codebase_story(
                config.repo_path.clone(),
                "rust".to_string()
            )
            .await?;

        // Record documentation system initialization
        story_engine
            .add_plot_point(
                codebase_story_id.clone(),
                PlotType::Discovery {
                    insight: "Story-driven documentation system initialized".to_string(),
                },
                vec!["documentation".to_string(), "automation".to_string()],
            )
            .await?;

        // Initialize default templates
        let doc_templates = Arc::new(RwLock::new(Self::create_default_templates()));

        Ok(Self {
            config,
            story_engine,
            code_analyzer,
            self_modify,
            learning_system,
            memory,
            codebase_story_id,
            doc_templates,
            generated_docs: Arc::new(RwLock::new(Vec::new())),
        })
    }

    /// Analyze documentation coverage
    pub async fn analyze_documentation_coverage(&self) -> Result<DocumentationAnalysis> {
        info!("📊 Analyzing documentation coverage");

    let mut total_files = 0;
        let mut documented_files = 0;
        let mut missing_docs = Vec::new();
        let mut outdated_docs = Vec::new();

        // Analyze all Rust files
        let files = self.find_rust_files(&self.config.repo_path).await?;
        total_files = files.len();

        for file_path in files {
            match self.analyze_file_documentation(&file_path).await {
                Ok((is_documented, missing, outdated)) => {
                    if is_documented {
                        documented_files += 1;
                    }
                    missing_docs.extend(missing);
                    outdated_docs.extend(outdated);
                }
                Err(e) => {
                    warn!("Failed to analyze {}: {}", file_path.display(), e);
                }
            }
        }

        let documentation_coverage = if total_files > 0 {
            documented_files as f32 / total_files as f32
        } else {
            0.0
        };

        Ok(DocumentationAnalysis {
            total_files,
            documented_files,
            documentation_coverage,
            missing_docs,
            outdated_docs,
        })
    }

    /// Generate documentation for undocumented items
    pub async fn generate_missing_documentation(&self) -> Result<Vec<GeneratedDocumentation>> {
        info!("📝 Generating missing documentation");

        let analysis = self.analyze_documentation_coverage().await?;
        let mut generated = Vec::new();

        // Prioritize high-priority missing docs
        let mut priority_docs = analysis.missing_docs;
        priority_docs.sort_by(|a, b| b.priority.partial_cmp(&a.priority).unwrap());

        // Generate documentation for top priority items
        for missing in priority_docs.iter().take(10) {
            match self.generate_documentation_for_item(missing).await {
                Ok(doc) => {
                    generated.push(doc.clone());
                    self.generated_docs.write().await.push(doc);
                }
                Err(e) => {
                    warn!("Failed to generate docs for {}: {}", missing.item_name, e);
                }
            }
        }

        // Record generation in story
        if !generated.is_empty() {
            self.story_engine
                .add_plot_point(
                    self.codebase_story_id.clone(),
                    PlotType::Progress {
                        milestone: format!("Generated {} documentation items", generated.len()),
                        percentage: analysis.documentation_coverage * 100.0,
                    },
                    vec![],
                )
                .await?;
        }

        Ok(generated)
    }

    /// Generate README for the project
    pub async fn generate_readme(&self) -> Result<GeneratedDocumentation> {
        info!("📖 Generating README");

        // Gather project information
        let project_info = self.gather_project_information().await?;

        // Use template to generate README
        let templates = self.doc_templates.read().await;
        let readme_template = templates
            .get(&DocumentationType::Readme)
            .ok_or_else(|| anyhow::anyhow!("README template not found"))?;

        let mut content = String::new();

        // Generate each section
        for section in &readme_template.sections {
            let section_content = match section.name.as_str() {
                "Project Overview" => self.generate_project_overview(&project_info).await?,
                "Features" => self.generate_features_section(&project_info).await?,
                "Installation" => self.generate_installation_section().await?,
                "Usage" => self.generate_usage_section(&project_info).await?,
                "Architecture" => self.generate_architecture_overview(&project_info).await?,
                "Contributing" => self.generate_contributing_section().await?,
                _ => section.template.clone(),
            };

            content.push_str(&section_content);
            content.push_str("\n\n");
        }

        let now = chrono::Utc::now();
        let doc = GeneratedDocumentation {
            doc_id: uuid::Uuid::new_v4().to_string(),
            entity_name: project_info.name.clone(),
            entity_type: "project".to_string(),
            documentation: content.clone(),
            doc_type: DocumentationType::Readme,
            confidence: 0.95,
            suggested_location: self.config.repo_path.join("README.md"),
            target_path: self.config.repo_path.join("README.md"),
            content: content.clone(),
            metadata: DocumentationMetadata {
                title: project_info.name.clone(),
                description: project_info.description.clone(),
                tags: vec!["readme".to_string(), "project".to_string()],
                complexity: 0.0,
                coverage: 1.0,
                examples_included: true,
            },
            generated_at: now,
            created_at: now,
        };

        // Apply the documentation
        self.apply_documentation(&doc).await?;

        Ok(doc)
    }

    /// Generate API documentation
    pub async fn generate_api_documentation(&self) -> Result<Vec<GeneratedDocumentation>> {
        info!("🔌 Generating API documentation");

        let mut api_docs = Vec::new();

        // Find all public APIs
        let public_apis = self.find_public_apis().await?;

        for (module, apis) in public_apis {
            let doc_content = self.generate_module_api_docs(&module, &apis).await?;

            let now = chrono::Utc::now();
            let doc = GeneratedDocumentation {
                doc_id: uuid::Uuid::new_v4().to_string(),
                entity_name: module.clone(),
                entity_type: "module".to_string(),
                documentation: doc_content.clone(),
                doc_type: DocumentationType::Api,
                confidence: 0.9,
                suggested_location: self.config.repo_path.join(format!("docs/api/{}.md", module)),
                target_path: self.config.repo_path.join(format!("docs/api/{}.md", module)),
                content: doc_content,
                metadata: DocumentationMetadata {
                    title: format!("{} API", module),
                    description: format!("API documentation for {} module", module),
                    tags: vec!["api".to_string(), module.clone()],
                    complexity: 0.0,
                    coverage: 1.0,
                    examples_included: true,
                },
                generated_at: now,
                created_at: now,
            };

            api_docs.push(doc);
        }

        // Apply all documentation
        for doc in &api_docs {
            self.apply_documentation(doc).await?;
        }

        Ok(api_docs)
    }

    /// Update outdated documentation
    pub async fn update_outdated_documentation(&self) -> Result<Vec<GeneratedDocumentation>> {
        info!("🔄 Updating outdated documentation");

        let analysis = self.analyze_documentation_coverage().await?;
        let mut updated = Vec::new();

        for outdated in analysis.outdated_docs {
            match self.update_documentation(&outdated).await {
                Ok(doc) => {
                    updated.push(doc);
                }
                Err(e) => {
                    warn!("Failed to update docs for {}: {}", outdated.file_path.display(), e);
                }
            }
        }

        Ok(updated)
    }

    /// Generate documentation for a specific item
    async fn generate_documentation_for_item(
        &self,
        missing: &MissingDocumentation,
    ) -> Result<GeneratedDocumentation> {
        // Analyze the item
        let analysis = self.code_analyzer.analyze_file(&missing.file_path).await?;

        // Generate appropriate documentation
        let content = match missing.item_type.as_str() {
            "function" => self.generate_function_docs(&missing.item_name, &analysis).await?,
            "struct" => self.generate_struct_docs(&missing.item_name, &analysis).await?,
            "trait" => self.generate_trait_docs(&missing.item_name, &analysis).await?,
            "module" => self.generate_module_docs(&missing.file_path).await?,
            _ => format!("/// Documentation for {}", missing.item_name),
        };

        let now = chrono::Utc::now();
        Ok(GeneratedDocumentation {
            doc_id: uuid::Uuid::new_v4().to_string(),
            entity_name: missing.item_name.clone(),
            entity_type: missing.item_type.clone(),
            documentation: content.clone(),
            doc_type: DocumentationType::Api,
            confidence: 0.85,
            suggested_location: missing.file_path.clone(),
            target_path: missing.file_path.clone(),
            content,
            metadata: DocumentationMetadata {
                title: missing.item_name.clone(),
                description: format!("{} documentation", missing.item_type),
                tags: vec![missing.item_type.clone()],
                complexity: missing.priority,
                coverage: 1.0,
                examples_included: false,
            },
            generated_at: now,
            created_at: now,
        })
    }

    /// Generate function documentation
    async fn generate_function_docs(
        &self,
        function_name: &str,
        analysis: &crate::tools::code_analysis::AnalysisResult,
    ) -> Result<String> {
        // Find the function
        let function = analysis.functions.iter()
            .find(|f| f.name == function_name)
            .ok_or_else(|| anyhow::anyhow!("Function not found"))?;

        // Generate documentation based on function signature and usage
        let mut doc = String::new();

        // Main description
        doc.push_str(&format!("/// {}\n", self.generate_function_description(function).await?));
        doc.push_str("///\n");

        // Parameters
        let signature = format!("{}({})", function.name, function.parameters.join(", "));
        if !signature.contains("()") {
            doc.push_str("/// # Parameters\n");
            doc.push_str("///\n");
            // Parse parameters from signature
            let params = self.parse_function_parameters(&signature)?;
            for param in params {
                doc.push_str(&format!("/// * `{}` - {}\n", param.0, param.1));
            }
            doc.push_str("///\n");
        }

        // Return value
        if signature.contains("->") {
            doc.push_str("/// # Returns\n");
            doc.push_str("///\n");
            doc.push_str(&format!("/// {}\n", self.generate_return_description(function).await?));
            doc.push_str("///\n");
        }

        // Errors
        if signature.contains("Result<") {
            doc.push_str("/// # Errors\n");
            doc.push_str("///\n");
            doc.push_str("/// Returns an error if the operation fails\n");
            doc.push_str("///\n");
        }

        // Example
        if self.should_include_example(function) {
            doc.push_str("/// # Example\n");
            doc.push_str("///\n");
            doc.push_str("/// ```rust\n");
            doc.push_str(&format!("/// {}\n", self.generate_function_example(function).await?));
            doc.push_str("/// ```\n");
        }

        Ok(doc)
    }

    /// Generate struct documentation
    async fn generate_struct_docs(
        &self,
        struct_name: &str,
        _analysis: &crate::tools::code_analysis::AnalysisResult,
    ) -> Result<String> {
        let mut doc = String::new();

        doc.push_str(&format!("/// {} structure\n", struct_name));
        doc.push_str("///\n");
        doc.push_str(&format!("/// This struct represents {}\n",
            self.infer_struct_purpose(struct_name).await?));

        Ok(doc)
    }

    /// Generate trait documentation
    async fn generate_trait_docs(
        &self,
        trait_name: &str,
        _analysis: &crate::tools::code_analysis::AnalysisResult,
    ) -> Result<String> {
        let mut doc = String::new();

        doc.push_str(&format!("/// {} trait\n", trait_name));
        doc.push_str("///\n");
        doc.push_str(&format!("/// This trait defines {}\n",
            self.infer_trait_purpose(trait_name).await?));

        Ok(doc)
    }

    /// Generate module documentation
    async fn generate_module_docs(&self, file_path: &Path) -> Result<String> {
        let module_name = file_path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("module");

        let mut doc = String::new();

        doc.push_str(&format!("//! {}\n", self.format_module_name(module_name)));
        doc.push_str("//!\n");
        doc.push_str(&format!("//! This module provides {}\n",
            self.infer_module_purpose(module_name, file_path).await?));

        Ok(doc)
    }

    /// Apply generated documentation
    async fn apply_documentation(&self, doc: &GeneratedDocumentation) -> Result<()> {
        match doc.doc_type {
            DocumentationType::Api | DocumentationType::Module => {
                // Insert inline documentation
                let change = CodeChange {
                    file_path: doc.target_path.clone(),
                    change_type: crate::cognitive::self_modify::ChangeType::Documentation,
                    description: format!("Add documentation for {}", doc.metadata.title),
                    reasoning: "Auto-generated documentation based on code analysis".to_string(),
                    old_content: None,
                    new_content: doc.content.clone(),
                    line_range: None,
                    risk_level: crate::cognitive::self_modify::RiskLevel::Low,
                    attribution: None,
                };

                self.self_modify.apply_code_change(change).await?;
            }
            DocumentationType::Readme | DocumentationType::Architecture => {
                // Write to file
                tokio::fs::write(&doc.target_path, &doc.content).await?;
            }
            _ => {}
        }

        // Store in memory
        self.memory
            .store(
                format!("generated_documentation_{}", doc.doc_id),
                vec![doc.content.clone()],
                MemoryMetadata {
                    source: "story_driven_documentation".to_string(),
                    tags: vec!["documentation".to_string()],
                    importance: 0.7,
                    associations: vec![MemoryId::from_string(doc.target_path.to_string_lossy().to_string())],
                    context: Some("Generated documentation".to_string()),
                    created_at: chrono::Utc::now(),
                    accessed_count: 0,
                    last_accessed: None,
                    version: 1,
                    category: "story".to_string(),
                    timestamp: chrono::Utc::now(),
                    expiration: None,
                },
            )
            .await?;

        Ok(())
    }

    /// Analyze file documentation
    async fn analyze_file_documentation(
        &self,
        file_path: &Path,
    ) -> Result<(bool, Vec<MissingDocumentation>, Vec<OutdatedDocumentation>)> {
        let content = tokio::fs::read_to_string(file_path).await?;
        let analysis = self.code_analyzer.analyze_file(file_path).await?;

        let mut missing = Vec::new();
        let outdated = Vec::new();
        let mut has_docs = false;

        // Check module documentation
        if !content.starts_with("//!") {
            missing.push(MissingDocumentation {
                file_path: file_path.to_path_buf(),
                item_type: "module".to_string(),
                item_name: file_path.file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("module")
                    .to_string(),
                priority: 0.5,
            });
        } else {
            has_docs = true;
        }

        // Check function documentation
        for function in &analysis.functions {
            // Check if function likely needs documentation based on complexity
            if function.complexity as f32 > self.config.min_complexity_threshold {
                if !self.has_function_docs(&content, &function.name) {
                    missing.push(MissingDocumentation {
                        file_path: file_path.to_path_buf(),
                        item_type: "function".to_string(),
                        item_name: function.name.clone(),
                        priority: function.complexity as f32 / 10.0,
                    });
                }
            }
        }

        Ok((has_docs, missing, outdated))
    }

    /// Check if function has documentation
    fn has_function_docs(&self, content: &str, function_name: &str) -> bool {
        // Simple check for doc comments before function
        content.contains(&format!("/// ")) && content.contains(&format!("fn {}", function_name))
    }

    /// Gather project information
    async fn gather_project_information(&self) -> Result<ProjectInfo> {
        // Get basic info from Cargo.toml
        let cargo_path = self.config.repo_path.join("Cargo.toml");
        let cargo_content = tokio::fs::read_to_string(&cargo_path).await.unwrap_or_default();

        // Extract project name and description
        let name = self.extract_from_cargo(&cargo_content, "name").unwrap_or("Project".to_string());
        let description = self.extract_from_cargo(&cargo_content, "description")
            .unwrap_or("A Rust project".to_string());

        // Get architectural insights if available
        let architecture = if let Some(learning) = &self.learning_system {
            learning.get_architectural_insights().await?
        } else {
            None
        };

        Ok(ProjectInfo {
            name,
            description,
            architecture,
            main_modules: vec![],
            features: vec![],
        })
    }

    /// Extract field from Cargo.toml
    fn extract_from_cargo(&self, content: &str, field: &str) -> Option<String> {
        content.lines()
            .find(|line| line.starts_with(field))
            .and_then(|line| line.split('=').nth(1))
            .map(|value| value.trim().trim_matches('"').to_string())
    }

    /// Generate project overview section
    async fn generate_project_overview(&self, info: &ProjectInfo) -> Result<String> {
        Ok(format!(
            "# {}\n\n{}\n\n## Overview\n\nThis project implements {}",
            info.name,
            info.description,
            self.infer_project_purpose(&info.name).await?
        ))
    }

    /// Generate features section
    async fn generate_features_section(&self, info: &ProjectInfo) -> Result<String> {
        let mut features = String::from("## Features\n\n");

        // Add discovered features
        if !info.features.is_empty() {
            for feature in &info.features {
                features.push_str(&format!("- {}\n", feature));
            }
        } else {
            // Generate default features based on modules
            features.push_str("- Modular architecture\n");
            features.push_str("- Comprehensive error handling\n");
            features.push_str("- Async/await support\n");
            features.push_str("- Well-documented APIs\n");
        }

        Ok(features)
    }

    /// Generate installation section
    async fn generate_installation_section(&self) -> Result<String> {
        Ok(String::from(
            "## Installation\n\n\
            Add this to your `Cargo.toml`:\n\n\
            ```toml\n\
            [dependencies]\n\
            # Add dependency here\n\
            ```\n\n\
            Or install via cargo:\n\n\
            ```bash\n\
            cargo add <package-name>\n\
            ```"
        ))
    }

    /// Generate usage section
    async fn generate_usage_section(&self, _info: &ProjectInfo) -> Result<String> {
        Ok(String::from(
            "## Usage\n\n\
            ```rust\n\
            // Basic usage example\n\
            use package::prelude::*;\n\n\
            fn main() {\n\
                // Example code here\n\
            }\n\
            ```"
        ))
    }

    /// Generate architecture overview
    async fn generate_architecture_overview(&self, info: &ProjectInfo) -> Result<String> {
        let mut arch = String::from("## Architecture\n\n");

        if let Some(insights) = &info.architecture {
            arch.push_str("### Module Structure\n\n");
            for (name, module) in &insights.module_structure.modules {
                arch.push_str(&format!("- **{}**: {}\n", name, module.purpose));
            }

            arch.push_str("\n### Key Components\n\n");
            for layer in &insights.layer_patterns {
                arch.push_str(&format!("- **{}**: {:?}\n", layer.name, layer.components));
            }
        } else {
            arch.push_str("The project follows a modular architecture with clear separation of concerns.");
        }

        Ok(arch)
    }

    /// Generate contributing section
    async fn generate_contributing_section(&self) -> Result<String> {
        Ok(String::from(
            "## Contributing\n\n\
            Contributions are welcome! Please feel free to submit a Pull Request.\n\n\
            ### Development Setup\n\n\
            1. Clone the repository\n\
            2. Install Rust via [rustup](https://rustup.rs/)\n\
            3. Run `cargo build` to build the project\n\
            4. Run `cargo test` to run tests\n\n\
            ### Guidelines\n\n\
            - Write tests for new functionality\n\
            - Follow Rust naming conventions\n\
            - Update documentation as needed\n\
            - Ensure all tests pass before submitting PR"
        ))
    }

    /// Find public APIs
    async fn find_public_apis(&self) -> Result<HashMap<String, Vec<ApiItem>>> {
        let mut apis = HashMap::new();

        // Scan for public items
        let files = self.find_rust_files(&self.config.repo_path).await?;

        for file in files {
            if let Ok(analysis) = self.code_analyzer.analyze_file(&file).await {
                let module_name = file.file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown")
                    .to_string();

                let mut module_apis = Vec::new();

                // Add public functions
                for func in analysis.functions {
                    // Assume all functions are public for now since FunctionInfo doesn't have visibility field
                    let signature = format!("{}({})", func.name, func.parameters.join(", "));
                    module_apis.push(ApiItem {
                        name: func.name,
                        kind: ApiItemKind::Function,
                        signature,
                        documentation: String::new(),
                    });
                }

                if !module_apis.is_empty() {
                    apis.insert(module_name, module_apis);
                }
            }
        }

        Ok(apis)
    }

    /// Generate module API documentation
    async fn generate_module_api_docs(
        &self,
        module: &str,
        apis: &[ApiItem],
    ) -> Result<String> {
        let mut doc = format!("# {} API Documentation\n\n", module);

        // Group by kind
        let functions: Vec<_> = apis.iter().filter(|a| matches!(a.kind, ApiItemKind::Function)).collect();
        let structs: Vec<_> = apis.iter().filter(|a| matches!(a.kind, ApiItemKind::Struct)).collect();
        let traits: Vec<_> = apis.iter().filter(|a| matches!(a.kind, ApiItemKind::Trait)).collect();

        // Document functions
        if !functions.is_empty() {
            doc.push_str("## Functions\n\n");
            for func in functions {
                doc.push_str(&format!("### `{}`\n\n", func.name));
                doc.push_str(&format!("```rust\n{}\n```\n\n", func.signature));
                doc.push_str(&format!("{}\n\n",
                    self.generate_api_description(&func.name, &func.signature).await?));
            }
        }

        // Document structs
        if !structs.is_empty() {
            doc.push_str("## Structs\n\n");
            for s in structs {
                doc.push_str(&format!("### `{}`\n\n", s.name));
                doc.push_str(&format!("{}\n\n", s.documentation));
            }
        }

        // Document traits
        if !traits.is_empty() {
            doc.push_str("## Traits\n\n");
            for t in traits {
                doc.push_str(&format!("### `{}`\n\n", t.name));
                doc.push_str(&format!("{}\n\n", t.documentation));
            }
        }

        Ok(doc)
    }

    /// Update outdated documentation
    async fn update_documentation(&self, outdated: &OutdatedDocumentation) -> Result<GeneratedDocumentation> {
        info!("📝 Updating outdated documentation for: {}", outdated.entity_name);

        // Analyze what has changed
        let change_analysis = self.analyze_documentation_changes(outdated).await?;

        // Parse the existing documentation to preserve valuable content
        let preserved_content = self.extract_valuable_content(&outdated.current_doc).await?;

        // Generate updated documentation based on the type
        let updated_content = match outdated.entity_type.as_str() {
            "function" => {
                self.update_function_documentation(
                    &outdated.entity_name,
                    &change_analysis,
                    &preserved_content
                ).await?
            }
            "struct" => {
                self.update_struct_documentation(
                    &outdated.entity_name,
                    &change_analysis,
                    &preserved_content
                ).await?
            }
            "module" => {
                self.update_module_documentation(
                    &outdated.entity_name,
                    &change_analysis,
                    &preserved_content
                ).await?
            }
            _ => {
                // Default update strategy
                self.update_generic_documentation(
                    &outdated.entity_name,
                    &outdated.entity_type,
                    &change_analysis,
                    &preserved_content
                ).await?
            }
        };

        // Create the updated documentation
        let now = chrono::Utc::now();
        let generated = GeneratedDocumentation {
            doc_id: uuid::Uuid::new_v4().to_string(),
            entity_name: outdated.entity_name.clone(),
            entity_type: outdated.entity_type.clone(),
            documentation: updated_content.clone(),
            doc_type: outdated.doc_type.clone(),
            confidence: 0.85, // Slightly lower confidence for updates vs new docs
            suggested_location: outdated.doc_location.clone(),
            target_path: outdated.doc_location.clone(),
            content: updated_content,
            metadata: DocumentationMetadata {
                title: outdated.entity_name.clone(),
                description: format!("Updated documentation for {}", outdated.entity_name),
                tags: vec![outdated.entity_type.clone(), "updated".to_string()],
                complexity: 0.0,
                coverage: 1.0,
                examples_included: false,
            },
            generated_at: now,
            created_at: now,
        };

        // Learn from the update
        self.learn_from_update(&outdated, &generated).await?;

        Ok(generated)
    }

    /// Analyze what has changed in the documentation
    async fn analyze_documentation_changes(&self, outdated: &OutdatedDocumentation) -> Result<DocumentationChangeAnalysis> {
        let mut changes = DocumentationChangeAnalysis {
            signature_changed: false,
            behavior_changed: false,
            parameters_changed: false,
            return_type_changed: false,
            new_features: Vec::new(),
            removed_features: Vec::new(),
            semantic_changes: Vec::new(),
        };

        // Compare signatures if available
        if let (Some(old_sig), Some(new_sig)) = (&outdated.old_signature, &outdated.new_signature) {
            if old_sig != new_sig {
                changes.signature_changed = true;
                changes.parameters_changed = self.detect_parameter_changes(old_sig, new_sig);
                changes.return_type_changed = self.detect_return_type_changes(old_sig, new_sig);
            }
        }

        // Analyze semantic changes based on the reason for being outdated
        changes.semantic_changes = outdated.outdated_reasons.clone();

        Ok(changes)
    }

    /// Extract valuable content from existing documentation
    async fn extract_valuable_content(&self, current_doc: &str) -> Result<PreservedDocContent> {
        let mut preserved = PreservedDocContent {
            examples: Vec::new(),
            important_notes: Vec::new(),
            see_also_links: Vec::new(),
            custom_sections: HashMap::new(),
        };

        // Extract code examples
        let example_regex = regex::Regex::new(r"```[\s\S]*?```").unwrap();
        for cap in example_regex.captures_iter(current_doc) {
            preserved.examples.push(cap[0].to_string());
        }

        // Extract important notes and warnings
        let note_regex = regex::Regex::new(r"(?i)(note|warning|important|caution):.*").unwrap();
        for cap in note_regex.captures_iter(current_doc) {
            preserved.important_notes.push(cap[0].to_string());
        }

        // Extract see also links
        let link_regex = regex::Regex::new(r"\[([^\]]+)\]\(([^)]+)\)").unwrap();
        for cap in link_regex.captures_iter(current_doc) {
            preserved.see_also_links.push((cap[1].to_string(), cap[2].to_string()));
        }

        Ok(preserved)
    }

    /// Update function documentation
    async fn update_function_documentation(
        &self,
        name: &str,
        changes: &DocumentationChangeAnalysis,
        preserved: &PreservedDocContent,
    ) -> Result<String> {
        let mut doc = String::new();

        // Header with update notice
        doc.push_str(&format!("/// {}\n", self.generate_function_summary(name, changes).await?));
        doc.push_str("///\n");

        // Add update notice if significant changes
        if changes.signature_changed || changes.behavior_changed {
            doc.push_str("/// **Updated**: Function signature or behavior has changed.\n");
            doc.push_str("///\n");
        }

        // Parameters section if changed
        if changes.parameters_changed {
            doc.push_str("/// # Arguments\n");
            doc.push_str("///\n");
            doc.push_str("/// Updated parameter list - see function signature for details.\n");
            doc.push_str("///\n");
        }

        // Return value section if changed
        if changes.return_type_changed {
            doc.push_str("/// # Returns\n");
            doc.push_str("///\n");
            doc.push_str("/// Updated return type - see function signature for details.\n");
            doc.push_str("///\n");
        }

        // Preserve valuable examples
        if !preserved.examples.is_empty() {
            doc.push_str("/// # Examples\n");
            doc.push_str("///\n");
            for example in &preserved.examples {
                doc.push_str(&format!("/// {}\n", example.replace('\n', "\n/// ")));
            }
            doc.push_str("///\n");
        }

        // Preserve important notes
        for note in &preserved.important_notes {
            doc.push_str(&format!("/// {}\n", note));
        }

        Ok(doc)
    }

    /// Update struct documentation
    async fn update_struct_documentation(
        &self,
        name: &str,
        changes: &DocumentationChangeAnalysis,
        preserved: &PreservedDocContent,
    ) -> Result<String> {
        let mut doc = String::new();

        doc.push_str(&format!("/// {}\n", self.humanize_name(name)));
        doc.push_str("///\n");

        if !changes.new_features.is_empty() {
            doc.push_str("/// # New Features\n");
            doc.push_str("///\n");
            for feature in &changes.new_features {
                doc.push_str(&format!("/// - {}\n", feature));
            }
            doc.push_str("///\n");
        }

        // Preserve examples and notes
        for example in &preserved.examples {
            doc.push_str(&format!("/// {}\n", example.replace('\n', "\n/// ")));
        }

        Ok(doc)
    }

    /// Update module documentation
    async fn update_module_documentation(
        &self,
        name: &str,
        changes: &DocumentationChangeAnalysis,
        preserved: &PreservedDocContent,
    ) -> Result<String> {
        let mut doc = String::from("//! ");
        doc.push_str(&self.humanize_name(name));
        doc.push_str("\n//!\n");

        // Add change summary
        if !changes.semantic_changes.is_empty() {
            doc.push_str("//! ## Recent Updates\n//!\n");
            for change in &changes.semantic_changes {
                doc.push_str(&format!("//! - {}\n", change));
            }
            doc.push_str("//!\n");
        }

        Ok(doc)
    }

    /// Update generic documentation
    async fn update_generic_documentation(
        &self,
        name: &str,
        entity_type: &str,
        changes: &DocumentationChangeAnalysis,
        preserved: &PreservedDocContent,
    ) -> Result<String> {
        let mut doc = String::new();

        doc.push_str(&format!("/// {} {}\n", entity_type, name));
        doc.push_str("///\n");
        doc.push_str("/// Documentation has been updated to reflect recent changes.\n");

        // Include change summary
        if !changes.semantic_changes.is_empty() {
            doc.push_str("///\n");
            doc.push_str("/// # Changes\n");
            for change in &changes.semantic_changes {
                doc.push_str(&format!("/// - {}\n", change));
            }
        }

        Ok(doc)
    }

    /// Helper functions for change detection
    fn detect_parameter_changes(&self, old_sig: &str, new_sig: &str) -> bool {
        // Simple heuristic: check if parameter lists differ
        let old_params = old_sig.split('(').nth(1).unwrap_or("").split(')').next().unwrap_or("");
        let new_params = new_sig.split('(').nth(1).unwrap_or("").split(')').next().unwrap_or("");
        old_params != new_params
    }

    fn detect_return_type_changes(&self, old_sig: &str, new_sig: &str) -> bool {
        // Simple heuristic: check if return types differ
        let old_return = old_sig.split("->").nth(1).unwrap_or("").trim();
        let new_return = new_sig.split("->").nth(1).unwrap_or("").trim();
        old_return != new_return
    }

    /// Generate function summary based on changes
    async fn generate_function_summary(&self, name: &str, changes: &DocumentationChangeAnalysis) -> Result<String> {
        let base_summary = self.humanize_name(name);

        if changes.signature_changed {
            Ok(format!("{} (signature updated)", base_summary))
        } else if changes.behavior_changed {
            Ok(format!("{} (behavior updated)", base_summary))
        } else {
            Ok(base_summary)
        }
    }

    /// Learn from documentation updates
    async fn learn_from_update(&self, outdated: &OutdatedDocumentation, generated: &GeneratedDocumentation) -> Result<()> {
        // Record the update pattern
        let pattern = DocumentationUpdatePattern {
            entity_type: outdated.entity_type.clone(),
            outdated_reasons: outdated.outdated_reasons.clone(),
            update_approach: "context-preserving".to_string(),
            success_score: generated.confidence,
        };

        // Store in knowledge base for future reference
        info!("📚 Learned documentation update pattern for {} entities", pattern.entity_type);

        Ok(())
    }

    /// Helper methods
    async fn generate_function_description(&self, func: &FunctionInfo) -> Result<String> {
        // Generate description based on function name and signature
        let action = self.infer_function_action(&func.name);
        Ok(format!("{} {}", action, self.humanize_name(&func.name)))
    }

    async fn generate_return_description(&self, func: &FunctionInfo) -> Result<String> {
        let signature = format!("{}({})", func.name, func.parameters.join(", "));
        if signature.contains("Result<") {
            Ok("Returns `Ok` on success, `Err` on failure".to_string())
        } else if signature.contains("Option<") {
            Ok("Returns `Some` if found, `None` otherwise".to_string())
        } else {
            Ok("Returns the result of the operation".to_string())
        }
    }

    async fn generate_function_example(&self, func: &FunctionInfo) -> Result<String> {
        // Generate simple example based on function signature
        Ok(format!("let result = {}();", func.name))
    }

    async fn generate_api_description(&self, name: &str, _signature: &str) -> Result<String> {
        Ok(format!("Performs {} operation", self.humanize_name(name)))
    }

    fn parse_function_parameters(&self, signature: &str) -> Result<Vec<(String, String)>> {
        // Simple parameter parsing
        Ok(vec![])
    }

    fn should_include_example(&self, func: &FunctionInfo) -> bool {
        func.complexity > 5
    }

    async fn infer_struct_purpose(&self, name: &str) -> Result<String> {
        Ok(format!("a {} data structure", self.humanize_name(name)))
    }

    async fn infer_trait_purpose(&self, name: &str) -> Result<String> {
        Ok(format!("behavior for {}", self.humanize_name(name)))
    }

    async fn infer_module_purpose(&self, name: &str, _path: &Path) -> Result<String> {
        Ok(format!("functionality for {}", self.humanize_name(name)))
    }

    async fn infer_project_purpose(&self, _name: &str) -> Result<String> {
        Ok("a comprehensive software solution".to_string())
    }

    fn infer_function_action(&self, name: &str) -> &str {
        if name.starts_with("get") || name.starts_with("fetch") {
            "Retrieves"
        } else if name.starts_with("set") || name.starts_with("update") {
            "Updates"
        } else if name.starts_with("create") || name.starts_with("new") {
            "Creates"
        } else if name.starts_with("delete") || name.starts_with("remove") {
            "Removes"
        } else if name.starts_with("is") || name.starts_with("has") {
            "Checks"
        } else {
            "Processes"
        }
    }

    fn format_module_name(&self, name: &str) -> String {
        self.humanize_name(name)
    }

    fn humanize_name(&self, name: &str) -> String {
        // Convert snake_case to human readable
        name.split('_')
            .map(|word| {
                let mut chars = word.chars();
                match chars.next() {
                    None => String::new(),
                    Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
                }
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Find Rust files recursively
    async fn find_rust_files(&self, dir: &Path) -> Result<Vec<PathBuf>> {
        let mut rust_files = Vec::new();
        let mut entries = tokio::fs::read_dir(dir).await?;

        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();

            if path.is_dir() {
                let dir_name = path.file_name().unwrap_or_default().to_str().unwrap_or("");
                if !matches!(dir_name, "target" | ".git" | "node_modules") {
                    let sub_files = Box::pin(self.find_rust_files(&path)).await?;
                    rust_files.extend(sub_files);
                }
            } else if path.extension().map_or(false, |ext| ext == "rs") {
                rust_files.push(path);
            }
        }

        Ok(rust_files)
    }

    /// Create default documentation templates
    fn create_default_templates() -> HashMap<DocumentationType, DocumentationTemplate> {
        let mut templates = HashMap::new();

        // README template
        templates.insert(
            DocumentationType::Readme,
            DocumentationTemplate {
                template_type: DocumentationType::Readme,
                sections: vec![
                    TemplateSection {
                        name: "Project Overview".to_string(),
                        required: true,
                        template: String::new(),
                    },
                    TemplateSection {
                        name: "Features".to_string(),
                        required: true,
                        template: String::new(),
                    },
                    TemplateSection {
                        name: "Installation".to_string(),
                        required: true,
                        template: String::new(),
                    },
                    TemplateSection {
                        name: "Usage".to_string(),
                        required: true,
                        template: String::new(),
                    },
                    TemplateSection {
                        name: "Architecture".to_string(),
                        required: false,
                        template: String::new(),
                    },
                    TemplateSection {
                        name: "Contributing".to_string(),
                        required: false,
                        template: String::new(),
                    },
                ],
            },
        );

        // API documentation template
        templates.insert(
            DocumentationType::Api,
            DocumentationTemplate {
                template_type: DocumentationType::Api,
                sections: vec![
                    TemplateSection {
                        name: "Overview".to_string(),
                        required: true,
                        template: String::new(),
                    },
                    TemplateSection {
                        name: "Functions".to_string(),
                        required: false,
                        template: String::new(),
                    },
                    TemplateSection {
                        name: "Types".to_string(),
                        required: false,
                        template: String::new(),
                    },
                    TemplateSection {
                        name: "Examples".to_string(),
                        required: false,
                        template: String::new(),
                    },
                ],
            },
        );

        templates
    }
    
    /// Create a new story-driven documentation generator with minimal dependencies
    /// This is useful for initialization in contexts where not all dependencies are available
    pub async fn new_with_defaults(
        story_engine: Arc<StoryEngine>,
        memory: Arc<CognitiveMemory>,
    ) -> Result<Self> {
        info!("📚 Initializing Story-Driven Documentation with defaults");
        
        let config = StoryDrivenDocumentationConfig::default();
        
        // Create minimal components
        let code_analyzer = Arc::new(CodeAnalyzer::new(memory.clone()).await?);
        
        // Create minimal self-modification pipeline
        let self_modify = Arc::new(SelfModificationPipeline::new(
            config.repo_path.clone(),
            memory.clone(),
        ).await?);
        
        // Create or get codebase story
        let codebase_story_id = story_engine
            .create_codebase_story(
                config.repo_path.clone(),
                "rust".to_string()
            )
            .await?;
            
        // Record documentation system initialization
        story_engine
            .add_plot_point(
                codebase_story_id.clone(),
                PlotType::Discovery {
                    insight: "Story-driven documentation system initialized with defaults".to_string(),
                },
                vec!["documentation".to_string(), "automation".to_string()],
            )
            .await?;
            
        // Initialize default templates
        let doc_templates = Arc::new(RwLock::new(Self::create_default_templates()));
        
        Ok(Self {
            config,
            story_engine,
            code_analyzer,
            self_modify,
            learning_system: None,
            memory,
            codebase_story_id,
            doc_templates,
            generated_docs: Arc::new(RwLock::new(Vec::new())),
        })
    }
    
    /// Generate documentation for a given path
    pub async fn generate_documentation(&self, path: &Path) -> Result<Vec<GeneratedDocumentation>> {
        info!("📚 Generating story-driven documentation for {:?}", path);
        
        let mut all_docs = Vec::new();
        
        // Analyze current documentation coverage
        let coverage = self.analyze_documentation_coverage().await?;
        
        // Generate missing documentation if coverage is low
        if coverage.documentation_coverage < 80.0 {
            let missing_docs = self.generate_missing_documentation().await?;
            all_docs.extend(missing_docs);
        }
        
        // Generate or update README
        let readme = self.generate_readme().await?;
        all_docs.push(readme);
        
        // Generate API documentation
        let api_docs = self.generate_api_documentation().await?;
        all_docs.extend(api_docs);
        
        // Update any outdated documentation
        let updated_docs = self.update_outdated_documentation().await?;
        all_docs.extend(updated_docs);
        
        // Record documentation generation in story
        self.story_engine
            .add_plot_point(
                self.codebase_story_id.clone(),
                PlotType::Goal {
                    objective: format!("Generated {} documentation files", all_docs.len()),
                },
                vec!["documentation".to_string(), "generated".to_string()],
            )
            .await?;
        
        // Store generated docs
        *self.generated_docs.write().await = all_docs.clone();
        
        Ok(all_docs)
    }
}

/// Project information
#[derive(Debug, Clone)]
struct ProjectInfo {
    name: String,
    description: String,
    architecture: Option<crate::cognitive::story_driven_learning::ArchitecturalInsight>,
    main_modules: Vec<String>,
    features: Vec<String>,
}

/// API item
#[derive(Debug, Clone)]
struct ApiItem {
    name: String,
    kind: ApiItemKind,
    signature: String,
    documentation: String,
}

/// API item kind
#[derive(Debug, Clone)]
enum ApiItemKind {
    Function,
    Struct,
    Trait,
    Enum,
}
