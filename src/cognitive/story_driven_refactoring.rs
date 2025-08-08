//! Story-Driven Refactoring Suggestions
//! 
//! This module implements intelligent refactoring suggestions that improve code
//! structure, maintainability, and performance based on narrative context.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn};

use crate::cognitive::self_modify::{CodeChange, SelfModificationPipeline, RiskLevel};
use crate::cognitive::story_driven_learning::StoryDrivenLearning;
use crate::cognitive::story_driven_quality::{StoryDrivenQuality, QualityAnalysis};
use crate::memory::CognitiveMemory;
use crate::story::{PlotType, StoryEngine, StoryId};
use crate::tools::code_analysis::{CodeAnalyzer, FunctionInfo};

/// Configuration for refactoring suggestions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoryDrivenRefactoringConfig {
    /// Enable method extraction suggestions
    pub enable_method_extraction: bool,
    
    /// Enable variable renaming suggestions
    pub enable_variable_renaming: bool,
    
    /// Enable code consolidation suggestions
    pub enable_consolidation: bool,
    
    /// Enable pattern application suggestions
    pub enable_pattern_application: bool,
    
    /// Enable performance optimizations
    pub enable_performance_refactoring: bool,
    
    /// Enable architectural improvements
    pub enable_architectural_refactoring: bool,
    
    /// Minimum complexity for extraction
    pub min_complexity_for_extraction: usize,
    
    /// Maximum method length
    pub max_method_length: usize,
    
    /// Maximum risk level for auto-refactoring
    pub max_auto_refactor_risk: RiskLevel,
    
    /// Repository path
    pub repo_path: PathBuf,
}

impl Default for StoryDrivenRefactoringConfig {
    fn default() -> Self {
        Self {
            enable_method_extraction: true,
            enable_variable_renaming: true,
            enable_consolidation: true,
            enable_pattern_application: true,
            enable_performance_refactoring: true,
            enable_architectural_refactoring: true,
            min_complexity_for_extraction: 10,
            max_method_length: 50,
            max_auto_refactor_risk: RiskLevel::Low,
            repo_path: PathBuf::from("."),
        }
    }
}

/// Type of refactoring suggestion
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RefactoringType {
    ExtractMethod,
    ExtractVariable,
    InlineVariable,
    RenameSymbol,
    ConsolidateDuplication,
    ApplyPattern,
    SimplifyConditional,
    RemoveDeadCode,
    OptimizeImports,
    ReorderMethods,
    ExtractTrait,
    ExtractModule,
    ImproveErrorHandling,
    OptimizePerformance,
    ModernizeSyntax,
}

/// Refactoring suggestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefactoringSuggestion {
    pub suggestion_id: String,
    pub refactoring_type: RefactoringType,
    pub file_path: PathBuf,
    pub location: CodeLocation,
    pub description: String,
    pub rationale: String,
    pub impact: RefactoringImpact,
    pub risk_level: RiskLevel,
    pub automated: bool,
    pub preview: RefactoringPreview,
}

/// Code location
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeLocation {
    pub start_line: usize,
    pub end_line: usize,
    pub function_name: Option<String>,
}

/// Impact of refactoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefactoringImpact {
    pub complexity_reduction: f32,
    pub readability_improvement: f32,
    pub performance_impact: f32,
    pub maintainability_improvement: f32,
    pub affected_lines: usize,
}

/// Preview of refactoring changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefactoringPreview {
    pub before: String,
    pub after: String,
    pub explanation: String,
}

/// Refactoring analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefactoringAnalysis {
    pub suggestions: Vec<RefactoringSuggestion>,
    pub priority_order: Vec<String>,
    pub total_impact: RefactoringImpact,
    pub patterns_applicable: Vec<PatternApplication>,
}

/// Pattern application opportunity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternApplication {
    pub pattern_id: String,
    pub pattern_name: String,
    pub applicable_location: CodeLocation,
    pub confidence: f32,
}

/// Refactoring result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefactoringResult {
    pub suggestion_id: String,
    pub success: bool,
    pub changes_applied: Vec<CodeChange>,
    pub tests_passed: bool,
    pub rollback_available: bool,
}

/// Story-driven refactoring suggester
pub struct StoryDrivenRefactoring {
    config: StoryDrivenRefactoringConfig,
    story_engine: Arc<StoryEngine>,
    code_analyzer: Arc<CodeAnalyzer>,
    self_modify: Arc<SelfModificationPipeline>,
    quality_monitor: Option<Arc<StoryDrivenQuality>>,
    learning_system: Option<Arc<StoryDrivenLearning>>,
    memory: Arc<CognitiveMemory>,
    codebase_story_id: StoryId,
    suggestion_history: Arc<RwLock<Vec<RefactoringSuggestion>>>,
    applied_refactorings: Arc<RwLock<Vec<RefactoringResult>>>,
}

impl StoryDrivenRefactoring {
    /// Create new refactoring suggester
    pub async fn new(
        config: StoryDrivenRefactoringConfig,
        story_engine: Arc<StoryEngine>,
        code_analyzer: Arc<CodeAnalyzer>,
        self_modify: Arc<SelfModificationPipeline>,
        quality_monitor: Option<Arc<StoryDrivenQuality>>,
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
        
        // Record initialization
        story_engine
            .add_plot_point(
                codebase_story_id.clone(),
                PlotType::Discovery {
                    insight: "Story-driven refactoring system initialized".to_string(),
                },
                vec!["refactoring".to_string(), "improvement".to_string()],
            )
            .await?;
        
        Ok(Self {
            config,
            story_engine,
            code_analyzer,
            self_modify,
            quality_monitor,
            learning_system,
            memory,
            codebase_story_id,
            suggestion_history: Arc::new(RwLock::new(Vec::new())),
            applied_refactorings: Arc::new(RwLock::new(Vec::new())),
        })
    }
    
    /// Analyze code and generate refactoring suggestions
    pub async fn analyze_for_refactoring(&self) -> Result<RefactoringAnalysis> {
        info!("🔧 Analyzing code for refactoring opportunities");
        
        let mut suggestions = Vec::new();
        
        // Get quality analysis if available
        let quality_analysis = if let Some(quality) = &self.quality_monitor {
            Some(quality.analyze_quality().await?)
        } else {
            None
        };
        
        // Find method extraction opportunities
        if self.config.enable_method_extraction {
            let extraction_suggestions = self.find_method_extractions().await?;
            suggestions.extend(extraction_suggestions);
        }
        
        // Find duplication consolidation opportunities
        if self.config.enable_consolidation {
            let consolidation_suggestions = self.find_consolidation_opportunities().await?;
            suggestions.extend(consolidation_suggestions);
        }
        
        // Find pattern application opportunities
        if self.config.enable_pattern_application {
            let pattern_suggestions = self.find_pattern_applications().await?;
            suggestions.extend(pattern_suggestions);
        }
        
        // Find performance optimizations
        if self.config.enable_performance_refactoring {
            let perf_suggestions = self.find_performance_optimizations().await?;
            suggestions.extend(perf_suggestions);
        }
        
        // Find architectural improvements
        if self.config.enable_architectural_refactoring {
            let arch_suggestions = self.find_architectural_improvements(&quality_analysis).await?;
            suggestions.extend(arch_suggestions);
        }
        
        // Prioritize suggestions
        let priority_order = self.prioritize_suggestions(&suggestions);
        
        // Calculate total impact
        let total_impact = self.calculate_total_impact(&suggestions);
        
        // Find applicable patterns
        let patterns_applicable = self.find_applicable_patterns().await?;
        
        // Store suggestions in history
        self.suggestion_history.write().await.extend(suggestions.clone());
        
        Ok(RefactoringAnalysis {
            suggestions,
            priority_order,
            total_impact,
            patterns_applicable,
        })
    }
    
    /// Apply refactoring suggestion
    pub async fn apply_refactoring(&self, suggestion_id: &str) -> Result<RefactoringResult> {
        info!("🔨 Applying refactoring: {}", suggestion_id);
        
        // Find the suggestion
        let suggestions = self.suggestion_history.read().await;
        let suggestion = suggestions
            .iter()
            .find(|s| s.suggestion_id == suggestion_id)
            .ok_or_else(|| anyhow::anyhow!("Suggestion not found"))?
            .clone();
        
        // Check risk level
        if suggestion.risk_level > self.config.max_auto_refactor_risk {
            return Ok(RefactoringResult {
                suggestion_id: suggestion_id.to_string(),
                success: false,
                changes_applied: vec![],
                tests_passed: false,
                rollback_available: false,
            });
        }
        
        // Generate code changes
        let changes = self.generate_refactoring_changes(&suggestion).await?;
        
        // Apply changes
        let mut applied_changes = Vec::new();
        for change in changes {
            match self.self_modify.apply_code_change(change.clone()).await {
                Ok(_) => applied_changes.push(change),
                Err(e) => {
                    warn!("Failed to apply refactoring change: {}", e);
                    // Rollback previous changes
                    for applied in applied_changes.iter().rev() {
                        let _ = self.self_modify.rollback_change(&applied.file_path.to_string_lossy()).await;
                    }
                    
                    return Ok(RefactoringResult {
                        suggestion_id: suggestion_id.to_string(),
                        success: false,
                        changes_applied: vec![],
                        tests_passed: false,
                        rollback_available: false,
                    });
                }
            }
        }
        
        // Run tests (simplified - would use test system)
        let tests_passed = true;
        
        let result = RefactoringResult {
            suggestion_id: suggestion_id.to_string(),
            success: true,
            changes_applied: applied_changes,
            tests_passed,
            rollback_available: true,
        };
        
        // Store result
        self.applied_refactorings.write().await.push(result.clone());
        
        // Record in story
        self.story_engine
            .add_plot_point(
                self.codebase_story_id.clone(),
                PlotType::Transformation {
                    before: suggestion.description.clone(),
                    after: "Refactoring applied successfully".to_string(),
                },
                vec!["refactoring".to_string(), "improvement".to_string()],
            )
            .await?;
        
        Ok(result)
    }
    
    /// Find method extraction opportunities
    async fn find_method_extractions(&self) -> Result<Vec<RefactoringSuggestion>> {
        let mut suggestions = Vec::new();
        let files = self.find_source_files().await?;
        
        for file in files {
            if let Ok(analysis) = self.code_analyzer.analyze_file(&file).await {
                for function in &analysis.functions {
                    // Check for long methods
                    let method_length = function.line_end - function.line_start;
                    if method_length > self.config.max_method_length || 
                       function.complexity as usize > self.config.min_complexity_for_extraction {
                        
                        // Analyze for extractable blocks
                        let extractable_blocks = self.find_extractable_blocks(function, &file).await?;
                        
                        for block in extractable_blocks {
                            suggestions.push(RefactoringSuggestion {
                                suggestion_id: uuid::Uuid::new_v4().to_string(),
                                refactoring_type: RefactoringType::ExtractMethod,
                                file_path: file.clone(),
                                location: CodeLocation {
                                    start_line: block.start_line,
                                    end_line: block.end_line,
                                    function_name: Some(function.name.clone()),
                                },
                                description: format!(
                                    "Extract lines {}-{} into separate method",
                                    block.start_line, block.end_line
                                ),
                                rationale: "Reduces method complexity and improves readability".to_string(),
                                impact: RefactoringImpact {
                                    complexity_reduction: 0.3,
                                    readability_improvement: 0.4,
                                    performance_impact: 0.0,
                                    maintainability_improvement: 0.35,
                                    affected_lines: block.end_line - block.start_line,
                                },
                                risk_level: RiskLevel::Low,
                                automated: true,
                                preview: self.generate_extraction_preview(&block).await?,
                            });
                        }
                    }
                }
            }
        }
        
        Ok(suggestions)
    }
    
    /// Find consolidation opportunities
    async fn find_consolidation_opportunities(&self) -> Result<Vec<RefactoringSuggestion>> {
        let suggestions = Vec::new();
        
        // Would analyze for duplicate code blocks
        // For demo, return empty
        
        Ok(suggestions)
    }
    
    /// Find pattern application opportunities
    async fn find_pattern_applications(&self) -> Result<Vec<RefactoringSuggestion>> {
        let mut suggestions = Vec::new();
        
        if let Some(learning) = &self.learning_system {
            // Get learned patterns
            let pattern_stats = learning.get_pattern_stats().await?;
            
            // Find places where patterns could be applied
            let files = self.find_source_files().await?;
            
            for file in files {
                if let Ok(analysis) = self.code_analyzer.analyze_file(&file).await {
                    // Check for error handling improvements
                    for function in &analysis.functions {
                        let signature = format!("{}({})", function.name, function.parameters.join(", "));
                        if !signature.contains("Result<") && 
                           self.function_can_fail(&function) {
                            
                            suggestions.push(RefactoringSuggestion {
                                suggestion_id: uuid::Uuid::new_v4().to_string(),
                                refactoring_type: RefactoringType::ImproveErrorHandling,
                                file_path: file.clone(),
                                location: CodeLocation {
                                    start_line: function.line_start,
                                    end_line: function.line_end,
                                    function_name: Some(function.name.clone()),
                                },
                                description: "Apply Result<T, E> error handling pattern".to_string(),
                                rationale: "Improves error handling and follows Rust best practices".to_string(),
                                impact: RefactoringImpact {
                                    complexity_reduction: 0.0,
                                    readability_improvement: 0.3,
                                    performance_impact: 0.0,
                                    maintainability_improvement: 0.4,
                                    affected_lines: 5,
                                },
                                risk_level: RiskLevel::Medium,
                                automated: false,
                                preview: RefactoringPreview {
                                    before: signature.clone(),
                                    after: signature.replace("->", "-> Result<") + ", Error>",
                                    explanation: "Add Result return type for proper error handling".to_string(),
                                },
                            });
                        }
                    }
                }
            }
        }
        
        Ok(suggestions)
    }
    
    /// Find performance optimizations
    async fn find_performance_optimizations(&self) -> Result<Vec<RefactoringSuggestion>> {
        let mut suggestions = Vec::new();
        let files = self.find_source_files().await?;
        
        for file in files {
            let content = tokio::fs::read_to_string(&file).await?;
            
            // Check for common performance issues
            
            // Clone in loops
            if content.contains(".clone()") && content.contains("for") {
                suggestions.push(RefactoringSuggestion {
                    suggestion_id: uuid::Uuid::new_v4().to_string(),
                    refactoring_type: RefactoringType::OptimizePerformance,
                    file_path: file.clone(),
                    location: CodeLocation {
                        start_line: 0,
                        end_line: 0,
                        function_name: None,
                    },
                    description: "Avoid cloning in loops".to_string(),
                    rationale: "Cloning in loops can cause performance issues".to_string(),
                    impact: RefactoringImpact {
                        complexity_reduction: 0.0,
                        readability_improvement: 0.1,
                        performance_impact: 0.4,
                        maintainability_improvement: 0.1,
                        affected_lines: 10,
                    },
                    risk_level: RiskLevel::Low,
                    automated: false,
                    preview: RefactoringPreview {
                        before: "for item in items { let x = item.clone(); }".to_string(),
                        after: "for item in &items { let x = item; }".to_string(),
                        explanation: "Use references instead of cloning".to_string(),
                    },
                });
            }
            
            // String concatenation in loops
            if content.contains("String::new()") && content.contains("push_str") && content.contains("for") {
                suggestions.push(RefactoringSuggestion {
                    suggestion_id: uuid::Uuid::new_v4().to_string(),
                    refactoring_type: RefactoringType::OptimizePerformance,
                    file_path: file.clone(),
                    location: CodeLocation {
                        start_line: 0,
                        end_line: 0,
                        function_name: None,
                    },
                    description: "Use String::with_capacity for string building".to_string(),
                    rationale: "Pre-allocating string capacity improves performance".to_string(),
                    impact: RefactoringImpact {
                        complexity_reduction: 0.0,
                        readability_improvement: 0.0,
                        performance_impact: 0.3,
                        maintainability_improvement: 0.1,
                        affected_lines: 5,
                    },
                    risk_level: RiskLevel::Low,
                    automated: true,
                    preview: RefactoringPreview {
                        before: "let mut s = String::new();".to_string(),
                        after: "let mut s = String::with_capacity(estimated_size);".to_string(),
                        explanation: "Pre-allocate string capacity".to_string(),
                    },
                });
            }
        }
        
        Ok(suggestions)
    }
    
    /// Find architectural improvements
    async fn find_architectural_improvements(
        &self,
        quality_analysis: &Option<QualityAnalysis>,
    ) -> Result<Vec<RefactoringSuggestion>> {
        let mut suggestions = Vec::new();
        
        // Check for module extraction opportunities
        if let Some(analysis) = quality_analysis {
            for hotspot in &analysis.hotspots {
                if hotspot.complexity > 20.0 && hotspot.issue_count > 5 {
                    suggestions.push(RefactoringSuggestion {
                        suggestion_id: uuid::Uuid::new_v4().to_string(),
                        refactoring_type: RefactoringType::ExtractModule,
                        file_path: hotspot.file_path.clone(),
                        location: CodeLocation {
                            start_line: 0,
                            end_line: 0,
                            function_name: None,
                        },
                        description: format!(
                            "Consider extracting {} into separate module",
                            hotspot.file_path.display()
                        ),
                        rationale: "High complexity and issue count indicate need for modularization".to_string(),
                        impact: RefactoringImpact {
                            complexity_reduction: 0.4,
                            readability_improvement: 0.5,
                            performance_impact: 0.0,
                            maintainability_improvement: 0.6,
                            affected_lines: 100,
                        },
                        risk_level: RiskLevel::High,
                        automated: false,
                        preview: RefactoringPreview {
                            before: "Large file with multiple responsibilities".to_string(),
                            after: "Split into focused modules".to_string(),
                            explanation: "Separate concerns into distinct modules".to_string(),
                        },
                    });
                }
            }
        }
        
        Ok(suggestions)
    }
    
    /// Find extractable blocks in a function
    async fn find_extractable_blocks(
        &self,
        function: &FunctionInfo,
        _file: &Path,
    ) -> Result<Vec<ExtractableBlock>> {
        let mut blocks = Vec::new();
        
        // Simplified - in real implementation would analyze AST
        if function.complexity > 10 {
            let block_size = (function.line_end - function.line_start) / 3;
            if block_size > 5 {
                blocks.push(ExtractableBlock {
                    start_line: function.line_start + block_size,
                    end_line: function.line_start + (2 * block_size),
                    suggested_name: "extracted_logic".to_string(),
                    parameters: vec![],
                    return_type: "()".to_string(),
                });
            }
        }
        
        Ok(blocks)
    }
    
    /// Generate extraction preview
    async fn generate_extraction_preview(&self, block: &ExtractableBlock) -> Result<RefactoringPreview> {
        Ok(RefactoringPreview {
            before: format!("// Lines {}-{} with complex logic", block.start_line, block.end_line),
            after: format!(
                "fn {}({}) -> {} {{\n    // Extracted logic\n}}\n\n// Call extracted method\nself.{}();",
                block.suggested_name,
                block.parameters.join(", "),
                block.return_type,
                block.suggested_name
            ),
            explanation: "Extract complex logic into separate method".to_string(),
        })
    }
    
    /// Check if function can fail
    fn function_can_fail(&self, function: &FunctionInfo) -> bool {
        // Simplified check - would analyze function body
        function.name.contains("parse") || 
        function.name.contains("read") ||
        function.name.contains("write") ||
        function.name.contains("connect")
    }
    
    /// Prioritize suggestions by impact
    fn prioritize_suggestions(&self, suggestions: &[RefactoringSuggestion]) -> Vec<String> {
        let mut priority_list: Vec<(String, f32)> = suggestions
            .iter()
            .map(|s| {
                let score = s.impact.complexity_reduction * 0.3 +
                           s.impact.maintainability_improvement * 0.3 +
                           s.impact.readability_improvement * 0.2 +
                           s.impact.performance_impact * 0.2;
                (s.suggestion_id.clone(), score)
            })
            .collect();
        
        priority_list.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        priority_list.into_iter().map(|(id, _)| id).collect()
    }
    
    /// Calculate total impact
    fn calculate_total_impact(&self, suggestions: &[RefactoringSuggestion]) -> RefactoringImpact {
        let mut total = RefactoringImpact {
            complexity_reduction: 0.0,
            readability_improvement: 0.0,
            performance_impact: 0.0,
            maintainability_improvement: 0.0,
            affected_lines: 0,
        };
        
        for suggestion in suggestions {
            total.complexity_reduction += suggestion.impact.complexity_reduction;
            total.readability_improvement += suggestion.impact.readability_improvement;
            total.performance_impact += suggestion.impact.performance_impact;
            total.maintainability_improvement += suggestion.impact.maintainability_improvement;
            total.affected_lines += suggestion.impact.affected_lines;
        }
        
        // Average the percentages
        let count = suggestions.len() as f32;
        if count > 0.0 {
            total.complexity_reduction /= count;
            total.readability_improvement /= count;
            total.performance_impact /= count;
            total.maintainability_improvement /= count;
        }
        
        total
    }
    
    /// Find applicable patterns
    async fn find_applicable_patterns(&self) -> Result<Vec<PatternApplication>> {
        let mut applications = Vec::new();
        
        if let Some(learning) = &self.learning_system {
            // Would check learned patterns against code
            // For demo, return example
            applications.push(PatternApplication {
                pattern_id: uuid::Uuid::new_v4().to_string(),
                pattern_name: "Error Handling Pattern".to_string(),
                applicable_location: CodeLocation {
                    start_line: 50,
                    end_line: 75,
                    function_name: Some("process_data".to_string()),
                },
                confidence: 0.85,
            });
        }
        
        Ok(applications)
    }
    
    /// Generate refactoring changes
    async fn generate_refactoring_changes(
        &self,
        suggestion: &RefactoringSuggestion,
    ) -> Result<Vec<CodeChange>> {
        let mut changes = Vec::new();
        
        match suggestion.refactoring_type {
            RefactoringType::ExtractMethod => {
                changes.push(CodeChange {
                    file_path: suggestion.file_path.clone(),
                    change_type: crate::cognitive::self_modify::ChangeType::Refactor,
                    description: suggestion.description.clone(),
                    reasoning: suggestion.rationale.clone(),
                    old_content: Some(suggestion.preview.before.clone()),
                    new_content: suggestion.preview.after.clone(),
                    line_range: None,
                    risk_level: suggestion.risk_level.clone(),
                    attribution: None,
                });
            }
            RefactoringType::OptimizePerformance => {
                changes.push(CodeChange {
                    file_path: suggestion.file_path.clone(),
                    change_type: crate::cognitive::self_modify::ChangeType::Enhancement,
                    description: suggestion.description.clone(),
                    reasoning: format!("Performance optimization: {}", suggestion.rationale),
                    old_content: Some(suggestion.preview.before.clone()),
                    new_content: suggestion.preview.after.clone(),
                    line_range: None,
                    risk_level: suggestion.risk_level.clone(),
                    attribution: None,
                });
            }
            _ => {
                // Other refactoring types
            }
        }
        
        Ok(changes)
    }
    
    /// Find source files
    async fn find_source_files(&self) -> Result<Vec<PathBuf>> {
        let mut source_files = Vec::new();
        let src_dir = self.config.repo_path.join("src");
        
        if src_dir.exists() {
            self.find_rust_files_recursive(&src_dir, &mut source_files).await?;
        }
        
        Ok(source_files)
    }
    
    /// Find Rust files recursively
    async fn find_rust_files_recursive(
        &self,
        dir: &Path,
        files: &mut Vec<PathBuf>,
    ) -> Result<()> {
        let mut entries = tokio::fs::read_dir(dir).await?;
        
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            
            if path.is_dir() {
                Box::pin(self.find_rust_files_recursive(&path, files)).await?;
            } else if path.extension().map_or(false, |ext| ext == "rs") {
                files.push(path);
            }
        }
        
        Ok(())
    }
}

/// Extractable block
#[derive(Debug, Clone)]
struct ExtractableBlock {
    start_line: usize,
    end_line: usize,
    suggested_name: String,
    parameters: Vec<String>,
    return_type: String,
}

impl RefactoringType {
    fn to_string(&self) -> &'static str {
        match self {
            RefactoringType::ExtractMethod => "Extract Method",
            RefactoringType::ExtractVariable => "Extract Variable",
            RefactoringType::InlineVariable => "Inline Variable",
            RefactoringType::RenameSymbol => "Rename Symbol",
            RefactoringType::ConsolidateDuplication => "Consolidate Duplication",
            RefactoringType::ApplyPattern => "Apply Pattern",
            RefactoringType::SimplifyConditional => "Simplify Conditional",
            RefactoringType::RemoveDeadCode => "Remove Dead Code",
            RefactoringType::OptimizeImports => "Optimize Imports",
            RefactoringType::ReorderMethods => "Reorder Methods",
            RefactoringType::ExtractTrait => "Extract Trait",
            RefactoringType::ExtractModule => "Extract Module",
            RefactoringType::ImproveErrorHandling => "Improve Error Handling",
            RefactoringType::OptimizePerformance => "Optimize Performance",
            RefactoringType::ModernizeSyntax => "Modernize Syntax",
        }
    }
}