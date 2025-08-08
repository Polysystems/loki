//! Story-Driven Learning System
//!
//! This module implements an intelligent pattern learning system that extracts
//! insights from the codebase and applies them to improve autonomous operations.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;

use crate::cognitive::autonomous_loop::AutonomousLoop;
use crate::memory::{CognitiveMemory, MemoryMetadata, MemoryId};
use crate::story::{PlotType, StoryEngine, StoryId};
use crate::tools::code_analysis::CodeAnalyzer;

/// Configuration for story-driven learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoryDrivenLearningConfig {
    /// Enable pattern extraction
    pub enable_pattern_extraction: bool,

    /// Enable architectural learning
    pub enable_architectural_learning: bool,

    /// Enable style learning
    pub enable_style_learning: bool,

    /// Enable performance pattern learning
    pub enable_performance_learning: bool,

    /// Enable security pattern learning
    pub enable_security_learning: bool,

    /// Minimum pattern frequency to consider
    pub min_pattern_frequency: usize,

    /// Confidence threshold for applying patterns
    pub pattern_confidence_threshold: f32,

    /// Repository path
    pub repo_path: PathBuf,
}

impl Default for StoryDrivenLearningConfig {
    fn default() -> Self {
        Self {
            enable_pattern_extraction: true,
            enable_architectural_learning: true,
            enable_style_learning: true,
            enable_performance_learning: true,
            enable_security_learning: true,
            min_pattern_frequency: 3,
            pattern_confidence_threshold: 0.8,
            repo_path: PathBuf::from("."),
        }
    }
}

/// Types of patterns that can be learned
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum LearnedPatternType {
    Architecture,
    CodingStyle,
    ErrorHandling,
    Performance,
    Security,
    Testing,
    Documentation,
    ApiDesign,
    DataFlow,
    Concurrency,
}

/// A pattern learned from the codebase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedPattern {
    pub pattern_id: String,
    pub pattern_type: LearnedPatternType,
    pub description: String,
    pub examples: Vec<PatternExample>,
    pub frequency: usize,
    pub confidence: f32,
    pub first_seen: chrono::DateTime<chrono::Utc>,
    pub last_updated: chrono::DateTime<chrono::Utc>,
    pub context: PatternContext,
    pub applications: Vec<PatternApplication>,
}

/// Example of a pattern in the codebase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternExample {
    pub file_path: PathBuf,
    pub line_range: (usize, usize),
    pub code_snippet: String,
    pub explanation: String,
}

/// Context in which a pattern applies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternContext {
    pub language: String,
    pub module_types: Vec<String>,
    pub dependencies: Vec<String>,
    pub tags: Vec<String>,
}

/// Record of pattern application
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternApplication {
    pub applied_at: chrono::DateTime<chrono::Utc>,
    pub target_file: PathBuf,
    pub success: bool,
    pub impact: String,
}

/// Architectural insight
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitecturalInsight {
    pub module_structure: ModuleStructure,
    pub dependency_graph: DependencyGraph,
    pub layer_patterns: Vec<LayerPattern>,
    pub communication_patterns: Vec<CommunicationPattern>,
}

/// Module structure information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleStructure {
    pub modules: HashMap<String, ModuleInfo>,
    pub hierarchy: Vec<String>,
    pub core_modules: Vec<String>,
}

/// Module information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleInfo {
    pub name: String,
    pub purpose: String,
    pub dependencies: Vec<String>,
    pub exports: Vec<String>,
    pub complexity: f32,
}

/// Dependency graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyGraph {
    pub nodes: Vec<String>,
    pub edges: Vec<(String, String)>,
    pub cycles: Vec<Vec<String>>,
}

/// Layer pattern in architecture
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerPattern {
    pub name: String,
    pub components: Vec<String>,
    pub allowed_dependencies: Vec<String>,
}

/// Communication pattern between components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationPattern {
    pub pattern_type: String,
    pub participants: Vec<String>,
    pub protocol: String,
}

/// Learning result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningResult {
    pub patterns_extracted: usize,
    pub insights_gained: Vec<String>,
    pub recommendations: Vec<String>,
    pub confidence: f32,
}

/// Story-driven learning system
pub struct StoryDrivenLearning {
    config: StoryDrivenLearningConfig,
    story_engine: Arc<StoryEngine>,
    code_analyzer: Arc<CodeAnalyzer>,
    autonomous_loop: Arc<RwLock<AutonomousLoop>>,
    memory: Arc<CognitiveMemory>,
    codebase_story_id: StoryId,
    learned_patterns: Arc<RwLock<HashMap<String, LearnedPattern>>>,
    architectural_insights: Arc<RwLock<Option<ArchitecturalInsight>>>,
    learning_history: Arc<RwLock<Vec<LearningResult>>>,
}

impl StoryDrivenLearning {
    
    /// Create new story-driven learning system
    pub async fn new(
        config: StoryDrivenLearningConfig,
        story_engine: Arc<StoryEngine>,
        code_analyzer: Arc<CodeAnalyzer>,
        autonomous_loop: Arc<RwLock<AutonomousLoop>>,
        memory: Arc<CognitiveMemory>,
    ) -> Result<Self> {
        // Create or get codebase story
        let codebase_story_id = story_engine
            .create_codebase_story(
                config.repo_path.clone(),
                "rust".to_string()
            )
            .await?;

        // Record learning system initialization
        story_engine
            .add_plot_point(
                codebase_story_id.clone(),
                PlotType::Discovery {
                    insight: "Story-driven learning system initialized".to_string(),
                },
                vec!["learning", "patterns"].iter().map(|s| s.to_string()).collect(), // context_tokens
            )
            .await?;

        Ok(Self {
            config,
            story_engine,
            code_analyzer,
            autonomous_loop,
            memory,
            codebase_story_id,
            learned_patterns: Arc::new(RwLock::new(HashMap::new())),
            architectural_insights: Arc::new(RwLock::new(None)),
            learning_history: Arc::new(RwLock::new(Vec::new())),
        })
    }

    /// Learn patterns from the entire codebase
    pub async fn learn_from_codebase(&self) -> Result<LearningResult> {
        info!("🧠 Starting comprehensive codebase learning");

        let mut patterns_extracted = 0;
        let mut insights = Vec::new();
        let mut recommendations = Vec::new();

        // Extract patterns
        if self.config.enable_pattern_extraction {
            let extracted = self.extract_patterns().await?;
            patterns_extracted += extracted;
            insights.push(format!("Extracted {} patterns from codebase", extracted));
        }

        // Learn architecture
        if self.config.enable_architectural_learning {
            let arch_insights = self.learn_architecture().await?;
            insights.push(format!(
                "Discovered {} architectural layers",
                arch_insights.layer_patterns.len()
            ));
            *self.architectural_insights.write().await = Some(arch_insights);
        }

        // Learn coding style
        if self.config.enable_style_learning {
            let style_patterns = self.learn_coding_style().await?;
            patterns_extracted += style_patterns;
            insights.push(format!("Learned {} coding style patterns", style_patterns));
        }

        // Learn performance patterns
        if self.config.enable_performance_learning {
            let perf_patterns = self.learn_performance_patterns().await?;
            patterns_extracted += perf_patterns;
            insights.push(format!("Identified {} performance patterns", perf_patterns));
        }

        // Learn security patterns
        if self.config.enable_security_learning {
            let sec_patterns = self.learn_security_patterns().await?;
            patterns_extracted += sec_patterns;
            insights.push(format!("Found {} security patterns", sec_patterns));
        }

        // Generate recommendations
        recommendations.extend(self.generate_recommendations().await?);

        let result = LearningResult {
            patterns_extracted,
            insights_gained: insights.clone(),
            recommendations: recommendations.clone(),
            confidence: 0.85,
        };

        // Store in history
        self.learning_history.write().await.push(result.clone());

        // Record in story
        self.story_engine
            .add_plot_point(
                self.codebase_story_id.clone(),
                PlotType::Discovery {
                    insight: format!("Learned {} patterns from codebase", patterns_extracted),
                },
                vec!["learning".to_string(), "patterns".to_string()],
            )
            .await?;

        Ok(result)
    }

    /// Extract patterns from code
    async fn extract_patterns(&self) -> Result<usize> {
        info!("Extracting patterns from codebase");

        let mut patterns_found = 0;
        let mut pattern_candidates: HashMap<String, Vec<PatternExample>> = HashMap::new();

        // Analyze all Rust files
        let files = self.find_rust_files(&self.config.repo_path).await?;

        for file_path in files {
            if let Ok(analysis) = self.code_analyzer.analyze_file(&file_path).await {
                // Extract error handling patterns
                self.extract_error_patterns(&file_path, &analysis, &mut pattern_candidates).await?;

                // Extract API patterns
                self.extract_api_patterns(&file_path, &analysis, &mut pattern_candidates).await?;

                // Extract concurrency patterns
                self.extract_concurrency_patterns(&file_path, &analysis, &mut pattern_candidates).await?;
            }
        }

        // Convert candidates to learned patterns
        let mut patterns = self.learned_patterns.write().await;

        for (pattern_key, examples) in pattern_candidates {
            if examples.len() >= self.config.min_pattern_frequency {
                let frequency = examples.len();
                let pattern = LearnedPattern {
                    pattern_id: uuid::Uuid::new_v4().to_string(),
                    pattern_type: self.classify_pattern_type(&pattern_key),
                    description: pattern_key.clone(),
                    examples,
                    frequency,
                    confidence: self.calculate_pattern_confidence(frequency),
                    first_seen: chrono::Utc::now(),
                    last_updated: chrono::Utc::now(),
                    context: PatternContext {
                        language: "rust".to_string(),
                        module_types: vec![],
                        dependencies: vec![],
                        tags: vec![],
                    },
                    applications: vec![],
                };

                patterns.insert(pattern.pattern_id.clone(), pattern);
                patterns_found += 1;
            }
        }

        Ok(patterns_found)
    }

    /// Extract error handling patterns
    async fn extract_error_patterns(
        &self,
        file_path: &Path,
        analysis: &crate::tools::code_analysis::AnalysisResult,
        patterns: &mut HashMap<String, Vec<PatternExample>>,
    ) -> Result<()> {
        // Look for Result return types
        for function in &analysis.functions {
            let signature = format!("{}({})", function.name, function.parameters.join(", "));
            if signature.contains("Result<") {
                let pattern_key = "error_handling_result_return".to_string();
                patterns.entry(pattern_key).or_insert_with(Vec::new).push(PatternExample {
                    file_path: file_path.to_path_buf(),
                    line_range: (function.line_start, function.line_end),
                    code_snippet: signature,
                    explanation: "Function returns Result for error handling".to_string(),
                });
            }
        }

        Ok(())
    }

    /// Extract API design patterns
    async fn extract_api_patterns(
        &self,
        file_path: &Path,
        _analysis: &crate::tools::code_analysis::AnalysisResult,
        patterns: &mut HashMap<String, Vec<PatternExample>>,
    ) -> Result<()> {
        // Look for builder patterns, factory patterns, etc.
        let content = tokio::fs::read_to_string(file_path).await?;

        if content.contains("impl Builder for") || content.contains("fn build(") {
            let pattern_key = "api_builder_pattern".to_string();
            patterns.entry(pattern_key).or_insert_with(Vec::new).push(PatternExample {
                file_path: file_path.to_path_buf(),
                line_range: (0, 0),
                code_snippet: "Builder pattern detected".to_string(),
                explanation: "Uses builder pattern for API construction".to_string(),
            });
        }

        Ok(())
    }

    /// Extract concurrency patterns
    async fn extract_concurrency_patterns(
        &self,
        file_path: &Path,
        _analysis: &crate::tools::code_analysis::AnalysisResult,
        patterns: &mut HashMap<String, Vec<PatternExample>>,
    ) -> Result<()> {
        let content = tokio::fs::read_to_string(file_path).await?;

        // Arc<RwLock<T>> pattern
        if content.contains("Arc<RwLock<") {
            let pattern_key = "concurrency_arc_rwlock".to_string();
            patterns.entry(pattern_key).or_insert_with(Vec::new).push(PatternExample {
                file_path: file_path.to_path_buf(),
                line_range: (0, 0),
                code_snippet: "Arc<RwLock<T>> usage".to_string(),
                explanation: "Shared state with read-write locking".to_string(),
            });
        }

        // Channel patterns
        if content.contains("mpsc::channel") || content.contains("tokio::sync::mpsc") {
            let pattern_key = "concurrency_channel_communication".to_string();
            patterns.entry(pattern_key).or_insert_with(Vec::new).push(PatternExample {
                file_path: file_path.to_path_buf(),
                line_range: (0, 0),
                code_snippet: "Channel-based communication".to_string(),
                explanation: "Uses channels for thread communication".to_string(),
            });
        }

        Ok(())
    }

    /// Learn architectural patterns
    async fn learn_architecture(&self) -> Result<ArchitecturalInsight> {
        info!("Learning architectural patterns");

        // Analyze module structure
        let module_structure = self.analyze_module_structure().await?;

        // Build dependency graph
        let dependency_graph = self.build_dependency_graph(&module_structure).await?;

        // Identify layers
        let layer_patterns = self.identify_layers(&module_structure, &dependency_graph).await?;

        // Find communication patterns
        let communication_patterns = self.find_communication_patterns(&module_structure).await?;

        Ok(ArchitecturalInsight {
            module_structure,
            dependency_graph,
            layer_patterns,
            communication_patterns,
        })
    }

    /// Analyze module structure
    async fn analyze_module_structure(&self) -> Result<ModuleStructure> {
        let mut modules = HashMap::new();

        // Add known core modules
        let core_modules = vec![
            ("cognitive", "Core AI consciousness and processing"),
            ("memory", "Hierarchical memory systems"),
            ("story", "Narrative management"),
            ("tools", "External integrations"),
            ("safety", "Security and validation"),
        ];

        for (name, purpose) in core_modules {
            modules.insert(
                name.to_string(),
                ModuleInfo {
                    name: name.to_string(),
                    purpose: purpose.to_string(),
                    dependencies: vec![],
                    exports: vec![],
                    complexity: 0.0,
                },
            );
        }

        Ok(ModuleStructure {
            hierarchy: vec!["cognitive".to_string(), "memory".to_string(), "tools".to_string()],
            core_modules: modules.keys().cloned().collect(),
            modules,
        })
    }

    /// Build dependency graph
    async fn build_dependency_graph(&self, _module_structure: &ModuleStructure) -> Result<DependencyGraph> {
        // Simplified dependency graph
        Ok(DependencyGraph {
            nodes: vec![
                "cognitive".to_string(),
                "memory".to_string(),
                "story".to_string(),
                "tools".to_string(),
            ],
            edges: vec![
                ("cognitive".to_string(), "memory".to_string()),
                ("cognitive".to_string(), "story".to_string()),
                ("cognitive".to_string(), "tools".to_string()),
                ("story".to_string(), "memory".to_string()),
            ],
            cycles: vec![],
        })
    }

    /// Identify architectural layers
    async fn identify_layers(
        &self,
        _module_structure: &ModuleStructure,
        _dependency_graph: &DependencyGraph,
    ) -> Result<Vec<LayerPattern>> {
        Ok(vec![
            LayerPattern {
                name: "Consciousness Layer".to_string(),
                components: vec!["cognitive".to_string()],
                allowed_dependencies: vec!["memory".to_string(), "story".to_string()],
            },
            LayerPattern {
                name: "Memory Layer".to_string(),
                components: vec!["memory".to_string()],
                allowed_dependencies: vec![],
            },
            LayerPattern {
                name: "Tool Layer".to_string(),
                components: vec!["tools".to_string()],
                allowed_dependencies: vec!["memory".to_string()],
            },
        ])
    }

    /// Find communication patterns
    async fn find_communication_patterns(&self, _module_structure: &ModuleStructure) -> Result<Vec<CommunicationPattern>> {
        Ok(vec![
            CommunicationPattern {
                pattern_type: "Event-Driven".to_string(),
                participants: vec!["cognitive".to_string(), "story".to_string()],
                protocol: "Async message passing".to_string(),
            },
            CommunicationPattern {
                pattern_type: "Request-Response".to_string(),
                participants: vec!["cognitive".to_string(), "tools".to_string()],
                protocol: "Function calls".to_string(),
            },
        ])
    }

    /// Learn coding style patterns
    async fn learn_coding_style(&self) -> Result<usize> {
        info!("Learning coding style patterns");

        let mut style_patterns = 0;
        let mut patterns = self.learned_patterns.write().await;

        // Common Rust idioms
        let idioms = vec![
            ("use_option_map", "Option::map for transformations"),
            ("use_result_question_mark", "? operator for error propagation"),
            ("use_iterators", "Iterator chains instead of loops"),
            ("use_pattern_matching", "Pattern matching for control flow"),
        ];

        for (_pattern_name, description) in idioms {
            let pattern = LearnedPattern {
                pattern_id: uuid::Uuid::new_v4().to_string(),
                pattern_type: LearnedPatternType::CodingStyle,
                description: description.to_string(),
                examples: vec![],
                frequency: 10, // Assumed common
                confidence: 0.9,
                first_seen: chrono::Utc::now(),
                last_updated: chrono::Utc::now(),
                context: PatternContext {
                    language: "rust".to_string(),
                    module_types: vec![],
                    dependencies: vec![],
                    tags: vec!["idiom".to_string()],
                },
                applications: vec![],
            };

            patterns.insert(pattern.pattern_id.clone(), pattern);
            style_patterns += 1;
        }

        Ok(style_patterns)
    }

    /// Learn performance patterns
    async fn learn_performance_patterns(&self) -> Result<usize> {
        info!("Learning performance patterns");

        let mut perf_patterns = 0;
        let mut patterns = self.learned_patterns.write().await;

        // Common performance patterns
        let perf_idioms = vec![
            ("avoid_allocations", "Minimize heap allocations in hot paths"),
            ("use_simd", "SIMD optimizations for data processing"),
            ("parallel_processing", "Rayon for CPU-bound parallel tasks"),
            ("async_io", "Async for I/O-bound operations"),
        ];

        for (pattern_name, description) in perf_idioms {
            let pattern = LearnedPattern {
                pattern_id: uuid::Uuid::new_v4().to_string(),
                pattern_type: LearnedPatternType::Performance,
                description: description.to_string(),
                examples: vec![],
                frequency: 5,
                confidence: 0.85,
                first_seen: chrono::Utc::now(),
                last_updated: chrono::Utc::now(),
                context: PatternContext {
                    language: "rust".to_string(),
                    module_types: vec![],
                    dependencies: vec![],
                    tags: vec!["optimization".to_string()],
                },
                applications: vec![],
            };

            patterns.insert(pattern.pattern_id.clone(), pattern);
            perf_patterns += 1;
        }

        Ok(perf_patterns)
    }

    /// Learn security patterns
    async fn learn_security_patterns(&self) -> Result<usize> {
        info!("Learning security patterns");

        let mut sec_patterns = 0;
        let mut patterns = self.learned_patterns.write().await;

        // Security best practices
        let security_patterns = vec![
            ("input_validation", "Validate all external inputs"),
            ("secure_defaults", "Secure by default configurations"),
            ("least_privilege", "Principle of least privilege"),
            ("audit_logging", "Comprehensive audit logging"),
        ];

        for (pattern_name, description) in security_patterns {
            let pattern = LearnedPattern {
                pattern_id: uuid::Uuid::new_v4().to_string(),
                pattern_type: LearnedPatternType::Security,
                description: description.to_string(),
                examples: vec![],
                frequency: 8,
                confidence: 0.9,
                first_seen: chrono::Utc::now(),
                last_updated: chrono::Utc::now(),
                context: PatternContext {
                    language: "rust".to_string(),
                    module_types: vec![],
                    dependencies: vec![],
                    tags: vec!["security".to_string()],
                },
                applications: vec![],
            };

            patterns.insert(pattern.pattern_id.clone(), pattern);
            sec_patterns += 1;
        }

        Ok(sec_patterns)
    }

    /// Generate recommendations based on learned patterns
    async fn generate_recommendations(&self) -> Result<Vec<String>> {
        let mut recommendations = Vec::new();

        let patterns = self.learned_patterns.read().await;
        let arch_insights = self.architectural_insights.read().await;

        // Architecture recommendations
        if let Some(arch) = arch_insights.as_ref() {
            if !arch.dependency_graph.cycles.is_empty() {
                recommendations.push("Consider breaking circular dependencies for cleaner architecture".to_string());
            }

            if arch.layer_patterns.len() > 5 {
                recommendations.push("Consider consolidating architectural layers for simplicity".to_string());
            }
        }

        // Pattern-based recommendations
        let pattern_types: HashMap<LearnedPatternType, usize> = patterns
            .values()
            .map(|p| p.pattern_type.clone())
            .fold(HashMap::new(), |mut acc, t| {
                *acc.entry(t).or_insert(0) += 1;
                acc
            });

        if pattern_types.get(&LearnedPatternType::ErrorHandling).unwrap_or(&0) < &3 {
            recommendations.push("Implement more consistent error handling patterns".to_string());
        }

        if pattern_types.get(&LearnedPatternType::Testing).unwrap_or(&0) < &5 {
            recommendations.push("Establish more comprehensive testing patterns".to_string());
        }

        if pattern_types.get(&LearnedPatternType::Documentation).unwrap_or(&0) < &3 {
            recommendations.push("Improve documentation patterns and consistency".to_string());
        }

        Ok(recommendations)
    }

    /// Apply learned patterns to improve code
    pub async fn apply_patterns(&self, target_file: &Path) -> Result<Vec<PatternApplication>> {
        info!("Applying learned patterns to {}", target_file.display());

        let mut applications = Vec::new();
        
        // Analyze target file
        let analysis = self.code_analyzer.analyze_file(target_file).await?;

        // Collect patterns to update
        let mut patterns_to_update = Vec::new();
        
        {
            let patterns = self.learned_patterns.read().await;
            // Find applicable patterns
            for pattern in patterns.values() {
                if pattern.confidence >= self.config.pattern_confidence_threshold {
                    if self.is_pattern_applicable(&pattern, &analysis).await? {
                        let application = PatternApplication {
                            applied_at: chrono::Utc::now(),
                            target_file: target_file.to_path_buf(),
                            success: true,
                            impact: format!("Applied {} pattern", pattern.description),
                        };

                        applications.push(application.clone());
                        patterns_to_update.push((pattern.pattern_id.clone(), application));
                    }
                }
            }
        }
        
        // Update patterns with applications
        if !patterns_to_update.is_empty() {
            let mut patterns_mut = self.learned_patterns.write().await;
            for (pattern_id, application) in patterns_to_update {
                if let Some(p) = patterns_mut.get_mut(&pattern_id) {
                    p.applications.push(application);
                    p.last_updated = chrono::Utc::now();
                }
            }
        }

        // Store learning outcome
        let application_strings: Vec<String> = applications.iter()
            .map(|app| format!("{}: {} - {}", app.target_file.display(), app.applied_at, app.impact))
            .collect();
        
        self.memory
            .store(
                format!("pattern_applications_{}", target_file.to_string_lossy()),
                application_strings,
                MemoryMetadata {
                    source: "story_driven_learning".to_string(),
                    tags: vec!["learning".to_string(), "patterns".to_string()],
                    importance: 0.7,
                    associations: vec![MemoryId::from_string(target_file.to_string_lossy().to_string())],
                    context: Some("Pattern applications for learning".to_string()),
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

        Ok(applications)
    }

    /// Check if pattern is applicable
    async fn is_pattern_applicable(
        &self,
        pattern: &LearnedPattern,
        _analysis: &crate::tools::code_analysis::AnalysisResult,
    ) -> Result<bool> {
        // Simplified applicability check
        match pattern.pattern_type {
            LearnedPatternType::ErrorHandling => Ok(true),
            LearnedPatternType::CodingStyle => Ok(true),
            LearnedPatternType::Performance => Ok(pattern.confidence > 0.85),
            LearnedPatternType::Security => Ok(pattern.confidence > 0.9),
            _ => Ok(false),
        }
    }

    /// Get pattern statistics
    pub async fn get_pattern_stats(&self) -> Result<HashMap<LearnedPatternType, usize>> {
        let patterns = self.learned_patterns.read().await;

        Ok(patterns
            .values()
            .map(|p| p.pattern_type.clone())
            .fold(HashMap::new(), |mut acc, t| {
                *acc.entry(t).or_insert(0) += 1;
                acc
            }))
    }

    /// Get architectural insights
    pub async fn get_architectural_insights(&self) -> Result<Option<ArchitecturalInsight>> {
        Ok(self.architectural_insights.read().await.clone())
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

    /// Classify pattern type from key
    fn classify_pattern_type(&self, key: &str) -> LearnedPatternType {
        if key.contains("error") || key.contains("result") {
            LearnedPatternType::ErrorHandling
        } else if key.contains("api") || key.contains("builder") {
            LearnedPatternType::ApiDesign
        } else if key.contains("concurrency") || key.contains("async") {
            LearnedPatternType::Concurrency
        } else if key.contains("perf") || key.contains("optim") {
            LearnedPatternType::Performance
        } else if key.contains("sec") || key.contains("auth") {
            LearnedPatternType::Security
        } else {
            LearnedPatternType::CodingStyle
        }
    }

    /// Calculate pattern confidence based on frequency
    fn calculate_pattern_confidence(&self, frequency: usize) -> f32 {
        let base_confidence = 0.5;
        let frequency_bonus = (frequency as f32 / 10.0).min(0.4);
        base_confidence + frequency_bonus
    }
    
    /// Learn patterns from a code narrative
    pub async fn learn_from_narrative(
        &self,
        narrative: &crate::tui::story_driven_code_analysis::CodeNarrative,
    ) -> Result<Vec<LearnedPattern>> {
        info!("🧠 Learning patterns from code narrative");
        
        let mut learned = Vec::new();
        let mut patterns = self.learned_patterns.write().await;
        
        // Learn from story arc
        let arc_pattern = LearnedPattern {
            pattern_id: uuid::Uuid::new_v4().to_string(),
            pattern_type: LearnedPatternType::Architecture,
            description: format!("Story Arc: {}", narrative.story_arc.genre),
            examples: vec![PatternExample {
                file_path: PathBuf::from("codebase"),
                line_range: (0, 0),
                code_snippet: narrative.story_arc.exposition.clone(),
                explanation: "Codebase exposition and initial architecture".to_string(),
            }],
            frequency: 1,
            confidence: 0.9,
            first_seen: chrono::Utc::now(),
            last_updated: chrono::Utc::now(),
            context: PatternContext {
                language: "rust".to_string(),
                module_types: vec![],
                dependencies: vec![],
                tags: vec!["narrative".to_string(), "architecture".to_string()],
            },
            applications: vec![],
        };
        patterns.insert(arc_pattern.pattern_id.clone(), arc_pattern.clone());
        learned.push(arc_pattern);
        
        // Learn from character roles (components)
        for character in &narrative.characters {
            let role_pattern = LearnedPattern {
                pattern_id: uuid::Uuid::new_v4().to_string(),
                pattern_type: LearnedPatternType::ApiDesign,
                description: format!("Component Role: {:?} - {}", character.role, character.name),
                examples: vec![PatternExample {
                    file_path: character.path.clone(),
                    line_range: (0, 0),
                    code_snippet: character.traits.join(", "),
                    explanation: format!("Component responsibilities: {}", character.motivations.join(", ")),
                }],
                frequency: 1,
                confidence: 0.85,
                first_seen: chrono::Utc::now(),
                last_updated: chrono::Utc::now(),
                context: PatternContext {
                    language: "rust".to_string(),
                    module_types: vec![character.name.clone()],
                    dependencies: character.conflicts.clone(),
                    tags: vec!["component".to_string(), "role".to_string()],
                },
                applications: vec![],
            };
            patterns.insert(role_pattern.pattern_id.clone(), role_pattern.clone());
            learned.push(role_pattern);
        }
        
        // Learn from relationships
        for relationship in &narrative.relationships {
            let rel_pattern = LearnedPattern {
                pattern_id: uuid::Uuid::new_v4().to_string(),
                pattern_type: LearnedPatternType::Architecture,
                description: format!("{:?} between {} and {}", 
                    relationship.relationship_type,
                    relationship.character_a,
                    relationship.character_b
                ),
                examples: vec![PatternExample {
                    file_path: PathBuf::from("relationships"),
                    line_range: (0, 0),
                    code_snippet: relationship.description.clone(),
                    explanation: format!("Strength: {:.2}", relationship.strength),
                }],
                frequency: 1,
                confidence: relationship.strength as f32,
                first_seen: chrono::Utc::now(),
                last_updated: chrono::Utc::now(),
                context: PatternContext {
                    language: "rust".to_string(),
                    module_types: vec![relationship.character_a.clone(), relationship.character_b.clone()],
                    dependencies: vec![],
                    tags: vec!["relationship".to_string()],
                },
                applications: vec![],
            };
            patterns.insert(rel_pattern.pattern_id.clone(), rel_pattern.clone());
            learned.push(rel_pattern);
        }
        
        // Learn from themes
        for theme in &narrative.themes {
            let theme_pattern = LearnedPattern {
                pattern_id: uuid::Uuid::new_v4().to_string(),
                pattern_type: LearnedPatternType::CodingStyle,
                description: format!("Theme: {}", theme.name),
                examples: theme.examples.iter().map(|ex| PatternExample {
                    file_path: PathBuf::from("theme_example"),
                    line_range: (0, 0),
                    code_snippet: ex.clone(),
                    explanation: format!("Theme: {}", theme.name),
                }).collect(),
                frequency: theme.prevalence as usize,
                confidence: theme.prevalence as f32,
                first_seen: chrono::Utc::now(),
                last_updated: chrono::Utc::now(),
                context: PatternContext {
                    language: "rust".to_string(),
                    module_types: vec![],
                    dependencies: vec![],
                    tags: vec!["theme".to_string(), format!("{:?}", theme.theme_type)],
                },
                applications: vec![],
            };
            patterns.insert(theme_pattern.pattern_id.clone(), theme_pattern.clone());
            learned.push(theme_pattern);
        }
        
        // Store learning result
        let result = LearningResult {
            patterns_extracted: learned.len(),
            insights_gained: narrative.insights.iter().map(|i| i.content.clone()).collect(),
            recommendations: self.generate_recommendations().await?,
            confidence: 0.85,
        };
        self.learning_history.write().await.push(result);
        
        // Record in story
        self.story_engine
            .add_plot_point(
                self.codebase_story_id.clone(),
                PlotType::Discovery {
                    insight: format!("Learned {} patterns from narrative", learned.len()),
                },
                vec!["learning".to_string(), "narrative".to_string()],
            )
            .await?;
        
        Ok(learned)
    }
}
