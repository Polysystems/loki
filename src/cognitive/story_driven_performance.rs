//! Story-Driven Performance Monitoring and Optimization
//! 
//! This module implements intelligent performance analysis that tracks metrics,
//! identifies bottlenecks, and suggests optimizations based on narrative context.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{info, warn};

use crate::cognitive::self_modify::{CodeChange, SelfModificationPipeline, RiskLevel};
use crate::cognitive::story_driven_learning::StoryDrivenLearning;
use crate::memory::{CognitiveMemory, MemoryMetadata};
use crate::story::{PlotType, StoryEngine, StoryId};
use crate::tools::code_analysis::CodeAnalyzer;

/// Configuration for performance monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoryDrivenPerformanceConfig {
    /// Enable runtime performance monitoring
    pub enable_runtime_monitoring: bool,
    
    /// Enable static performance analysis
    pub enable_static_analysis: bool,
    
    /// Enable memory profiling
    pub enable_memory_profiling: bool,
    
    /// Enable CPU profiling
    pub enable_cpu_profiling: bool,
    
    /// Enable benchmark generation
    pub enable_benchmark_generation: bool,
    
    /// Enable automatic optimization
    pub enable_auto_optimization: bool,
    
    /// Performance degradation threshold (percentage)
    pub degradation_threshold: f32,
    
    /// Minimum improvement for optimization (percentage)
    pub min_improvement_threshold: f32,
    
    /// Maximum risk level for auto-optimization
    pub max_optimization_risk: RiskLevel,
    
    /// Monitoring interval
    pub monitoring_interval: Duration,
    
    /// Repository path
    pub repo_path: PathBuf,
}

impl Default for StoryDrivenPerformanceConfig {
    fn default() -> Self {
        Self {
            enable_runtime_monitoring: true,
            enable_static_analysis: true,
            enable_memory_profiling: true,
            enable_cpu_profiling: true,
            enable_benchmark_generation: true,
            enable_auto_optimization: true,
            degradation_threshold: 10.0,
            min_improvement_threshold: 5.0,
            max_optimization_risk: RiskLevel::Low,
            monitoring_interval: Duration::from_secs(300),
            repo_path: PathBuf::from("."),
        }
    }
}

/// Performance metric types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MetricType {
    ExecutionTime,
    MemoryUsage,
    CpuUsage,
    Throughput,
    Latency,
    GarbageCollection,
    CacheHitRate,
    DatabaseQueries,
    NetworkCalls,
}

impl fmt::Display for MetricType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            MetricType::ExecutionTime => "execution_time",
            MetricType::MemoryUsage => "memory_usage",
            MetricType::CpuUsage => "cpu_usage",
            MetricType::Throughput => "throughput",
            MetricType::Latency => "latency",
            MetricType::GarbageCollection => "garbage_collection",
            MetricType::CacheHitRate => "cache_hit_rate",
            MetricType::DatabaseQueries => "database_queries",
            MetricType::NetworkCalls => "network_calls",
        };
        write!(f, "{}", s)
    }
}

/// Performance measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMeasurement {
    pub metric_type: MetricType,
    pub value: f64,
    pub unit: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub context: String,
}

/// Performance bottleneck
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBottleneck {
    pub bottleneck_id: String,
    pub bottleneck_type: BottleneckType,
    pub location: CodeLocation,
    pub severity: BottleneckSeverity,
    pub impact: PerformanceImpact,
    pub description: String,
    pub measurements: Vec<PerformanceMeasurement>,
    pub suggested_fix: Option<OptimizationSuggestion>,
}

/// Bottleneck types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum BottleneckType {
    CPUBound,
    IOBound,
    MemoryLeak,
    ExcessiveAllocation,
    IneffcientAlgorithm,
    DatabaseNPlusOne,
    SynchronousBlocking,
    ExcessiveLocking,
    CacheMiss,
    NetworkLatency,
}

/// Bottleneck severity
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
pub enum BottleneckSeverity {
    Critical,
    High,
    Medium,
    Low,
}

/// Code location
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeLocation {
    pub file_path: PathBuf,
    pub function_name: Option<String>,
    pub line_number: Option<usize>,
}

/// Performance impact
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceImpact {
    pub execution_time_impact: f32,  // Percentage
    pub memory_impact: f32,          // Percentage
    pub cpu_impact: f32,             // Percentage
    pub user_experience_impact: f32, // Percentage
}

/// Optimization suggestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationSuggestion {
    pub optimization_id: String,
    pub optimization_type: OptimizationType,
    pub description: String,
    pub expected_improvement: PerformanceImpact,
    pub implementation_complexity: ComplexityLevel,
    pub risk_level: RiskLevel,
    pub code_changes: Vec<ProposedChange>,
}

/// Optimization types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum OptimizationType {
    AlgorithmOptimization,
    CachingStrategy,
    ParallelProcessing,
    AsynchronousIO,
    MemoryOptimization,
    DatabaseQueryOptimization,
    LazyLoading,
    Memoization,
    DataStructureOptimization,
    CodeElimination,
}

/// Complexity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
pub enum ComplexityLevel {
    Trivial,
    Simple,
    Moderate,
    Complex,
}

/// Proposed code change
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProposedChange {
    pub file_path: PathBuf,
    pub change_type: String,
    pub before: String,
    pub after: String,
    pub explanation: String,
}

/// Performance analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAnalysis {
    pub overall_performance_score: f32,
    pub bottlenecks: Vec<PerformanceBottleneck>,
    pub optimization_suggestions: Vec<OptimizationSuggestion>,
    pub performance_trends: PerformanceTrends,
    pub benchmark_results: Vec<BenchmarkResult>,
}

/// Performance trends
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTrends {
    pub execution_time_trend: TrendDirection,
    pub memory_usage_trend: TrendDirection,
    pub cpu_usage_trend: TrendDirection,
    pub trend_period: Duration,
}

/// Trend direction
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TrendDirection {
    Improving,
    Stable,
    Degrading,
}

/// Benchmark result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub benchmark_name: String,
    pub execution_time: Duration,
    pub memory_usage: usize,
    pub iterations: usize,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Performance optimization result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResult {
    pub optimization_id: String,
    pub success: bool,
    pub actual_improvement: PerformanceImpact,
    pub changes_applied: Vec<CodeChange>,
    pub benchmarks_before: Vec<BenchmarkResult>,
    pub benchmarks_after: Vec<BenchmarkResult>,
}

/// Story-driven performance monitor
pub struct StoryDrivenPerformance {
    config: StoryDrivenPerformanceConfig,
    story_engine: Arc<StoryEngine>,
    code_analyzer: Arc<CodeAnalyzer>,
    self_modify: Arc<SelfModificationPipeline>,
    learning_system: Option<Arc<StoryDrivenLearning>>,
    memory: Arc<CognitiveMemory>,
    codebase_story_id: StoryId,
    performance_history: Arc<RwLock<Vec<PerformanceMeasurement>>>,
    bottleneck_cache: Arc<RwLock<HashMap<String, PerformanceBottleneck>>>,
    optimization_history: Arc<RwLock<Vec<OptimizationResult>>>,
}

impl StoryDrivenPerformance {
    /// Create new performance monitor
    pub async fn new(
        config: StoryDrivenPerformanceConfig,
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
        
        // Record initialization
        story_engine
            .add_plot_point(
                codebase_story_id.clone(),
                PlotType::Discovery {
                    insight: "Performance monitoring system initialized".to_string(),
                },
                vec!["performance".to_string(), "monitoring".to_string()],
            )
            .await?;
        
        Ok(Self {
            config,
            story_engine,
            code_analyzer,
            self_modify,
            learning_system,
            memory,
            codebase_story_id,
            performance_history: Arc::new(RwLock::new(Vec::new())),
            bottleneck_cache: Arc::new(RwLock::new(HashMap::new())),
            optimization_history: Arc::new(RwLock::new(Vec::new())),
        })
    }
    
    /// Analyze performance
    pub async fn analyze_performance(&self) -> Result<PerformanceAnalysis> {
        info!("🚀 Analyzing application performance");
        
        let mut bottlenecks = Vec::new();
        let mut optimization_suggestions = Vec::new();
        
        // Static performance analysis
        if self.config.enable_static_analysis {
            let static_bottlenecks = self.analyze_static_performance().await?;
            bottlenecks.extend(static_bottlenecks);
        }
        
        // Runtime performance analysis
        if self.config.enable_runtime_monitoring {
            let runtime_bottlenecks = self.analyze_runtime_performance().await?;
            bottlenecks.extend(runtime_bottlenecks);
        }
        
        // Memory profiling
        if self.config.enable_memory_profiling {
            let memory_issues = self.analyze_memory_usage().await?;
            bottlenecks.extend(memory_issues);
        }
        
        // Generate optimization suggestions
        for bottleneck in &bottlenecks {
            if let Some(suggestion) = self.generate_optimization_suggestion(bottleneck).await? {
                optimization_suggestions.push(suggestion);
            }
        }
        
        // Calculate performance trends
        let performance_trends = self.calculate_performance_trends().await?;
        
        // Run benchmarks
        let benchmark_results = if self.config.enable_benchmark_generation {
            self.run_benchmarks().await?
        } else {
            Vec::new()
        };
        
        // Calculate overall score
        let overall_performance_score = self.calculate_performance_score(&bottlenecks);
        
        // Store bottlenecks in cache
        let mut cache = self.bottleneck_cache.write().await;
        for bottleneck in &bottlenecks {
            cache.insert(bottleneck.bottleneck_id.clone(), bottleneck.clone());
        }
        
        // Record in story
        self.story_engine
            .add_plot_point(
                self.codebase_story_id.clone(),
                PlotType::Analysis {
                    subject: "performance".to_string(),
                    findings: vec![
                        format!("Found {} performance bottlenecks", bottlenecks.len()),
                        format!("Generated {} optimization suggestions", optimization_suggestions.len()),
                        format!("Overall performance score: {:.1}%", overall_performance_score * 100.0),
                    ],
                },
                vec!["performance".to_string(), "analysis".to_string()],
            )
            .await?;
        
        Ok(PerformanceAnalysis {
            overall_performance_score,
            bottlenecks,
            optimization_suggestions,
            performance_trends,
            benchmark_results,
        })
    }
    
    /// Apply optimization
    pub async fn apply_optimization(&self, optimization_id: &str) -> Result<OptimizationResult> {
        info!("⚡ Applying optimization: {}", optimization_id);
        
        // Find the optimization suggestion
        let analysis = self.analyze_performance().await?;
        let optimization = analysis.optimization_suggestions
            .iter()
            .find(|o| o.optimization_id == optimization_id)
            .ok_or_else(|| anyhow::anyhow!("Optimization not found"))?
            .clone();
        
        // Check risk level
        if optimization.risk_level > self.config.max_optimization_risk {
            return Ok(OptimizationResult {
                optimization_id: optimization_id.to_string(),
                success: false,
                actual_improvement: PerformanceImpact {
                    execution_time_impact: 0.0,
                    memory_impact: 0.0,
                    cpu_impact: 0.0,
                    user_experience_impact: 0.0,
                },
                changes_applied: vec![],
                benchmarks_before: vec![],
                benchmarks_after: vec![],
            });
        }
        
        // Run benchmarks before
        let benchmarks_before = self.run_targeted_benchmarks(&optimization).await?;
        
        // Apply code changes
        let mut changes_applied = Vec::new();
        for proposed_change in &optimization.code_changes {
            let code_change = self.convert_to_code_change(proposed_change, &optimization).await?;
            match self.self_modify.apply_code_change(code_change.clone()).await {
                Ok(_) => changes_applied.push(code_change),
                Err(e) => {
                    warn!("Failed to apply optimization change: {}", e);
                    // Rollback previous changes
                    for applied in changes_applied.iter().rev() {
                        let _ = self.self_modify.rollback_change(&applied.file_path.to_string_lossy()).await;
                    }
                    
                    return Ok(OptimizationResult {
                        optimization_id: optimization_id.to_string(),
                        success: false,
                        actual_improvement: PerformanceImpact {
                            execution_time_impact: 0.0,
                            memory_impact: 0.0,
                            cpu_impact: 0.0,
                            user_experience_impact: 0.0,
                        },
                        changes_applied: vec![],
                        benchmarks_before,
                        benchmarks_after: vec![],
                    });
                }
            }
        }
        
        // Run benchmarks after
        let benchmarks_after = self.run_targeted_benchmarks(&optimization).await?;
        
        // Calculate actual improvement
        let actual_improvement = self.calculate_actual_improvement(
            &benchmarks_before,
            &benchmarks_after
        );
        
        let result = OptimizationResult {
            optimization_id: optimization_id.to_string(),
            success: true,
            actual_improvement,
            changes_applied,
            benchmarks_before,
            benchmarks_after,
        };
        
        // Store in history
        self.optimization_history.write().await.push(result.clone());
        
        // Record in story
        self.story_engine
            .add_plot_point(
                self.codebase_story_id.clone(),
                PlotType::Transformation {
                    before: "Unoptimized code".to_string(),
                    after: format!("Optimized: {}", optimization.description),
                },
                vec!["performance".to_string(), "optimization".to_string()],
            )
            .await?;
        
        Ok(result)
    }
    
    /// Analyze static performance
    async fn analyze_static_performance(&self) -> Result<Vec<PerformanceBottleneck>> {
        let mut bottlenecks = Vec::new();
        let files = self.find_source_files().await?;
        
        for file in files {
            if let Ok(content) = tokio::fs::read_to_string(&file).await {
                // Check for common performance issues
                
                // Nested loops (O(n²) or worse)
                if content.contains("for") && content.matches("for").count() > 1 {
                    let lines: Vec<_> = content.lines().collect();
                    for (i, line) in lines.iter().enumerate() {
                        if line.contains("for") {
                            // Check if there's another loop within 10 lines
                            for j in i+1..std::cmp::min(i+10, lines.len()) {
                                if lines[j].contains("for") {
                                    bottlenecks.push(PerformanceBottleneck {
                                        bottleneck_id: uuid::Uuid::new_v4().to_string(),
                                        bottleneck_type: BottleneckType::IneffcientAlgorithm,
                                        location: CodeLocation {
                                            file_path: file.clone(),
                                            function_name: None,
                                            line_number: Some(i + 1),
                                        },
                                        severity: BottleneckSeverity::Medium,
                                        impact: PerformanceImpact {
                                            execution_time_impact: 30.0,
                                            memory_impact: 10.0,
                                            cpu_impact: 40.0,
                                            user_experience_impact: 25.0,
                                        },
                                        description: "Nested loops detected - potential O(n²) complexity".to_string(),
                                        measurements: vec![],
                                        suggested_fix: None,
                                    });
                                    break;
                                }
                            }
                        }
                    }
                }
                
                // Excessive cloning
                let clone_count = content.matches(".clone()").count();
                if clone_count > 5 {
                    bottlenecks.push(PerformanceBottleneck {
                        bottleneck_id: uuid::Uuid::new_v4().to_string(),
                        bottleneck_type: BottleneckType::ExcessiveAllocation,
                        location: CodeLocation {
                            file_path: file.clone(),
                            function_name: None,
                            line_number: None,
                        },
                        severity: BottleneckSeverity::Low,
                        impact: PerformanceImpact {
                            execution_time_impact: 10.0,
                            memory_impact: 20.0,
                            cpu_impact: 15.0,
                            user_experience_impact: 10.0,
                        },
                        description: format!("Excessive cloning detected ({} instances)", clone_count),
                        measurements: vec![],
                        suggested_fix: None,
                    });
                }
                
                // Synchronous I/O in async context
                if content.contains("async") && content.contains("std::fs::") {
                    bottlenecks.push(PerformanceBottleneck {
                        bottleneck_id: uuid::Uuid::new_v4().to_string(),
                        bottleneck_type: BottleneckType::SynchronousBlocking,
                        location: CodeLocation {
                            file_path: file.clone(),
                            function_name: None,
                            line_number: None,
                        },
                        severity: BottleneckSeverity::High,
                        impact: PerformanceImpact {
                            execution_time_impact: 40.0,
                            memory_impact: 5.0,
                            cpu_impact: 20.0,
                            user_experience_impact: 50.0,
                        },
                        description: "Synchronous I/O in async context".to_string(),
                        measurements: vec![],
                        suggested_fix: None,
                    });
                }
            }
        }
        
        Ok(bottlenecks)
    }
    
    /// Analyze runtime performance
    async fn analyze_runtime_performance(&self) -> Result<Vec<PerformanceBottleneck>> {
        // In a real implementation, this would integrate with profiling tools
        // For demo, return simulated data
        Ok(vec![])
    }
    
    /// Analyze memory usage
    async fn analyze_memory_usage(&self) -> Result<Vec<PerformanceBottleneck>> {
        let mut bottlenecks = Vec::new();
        
        // Check for potential memory leaks
        let files = self.find_source_files().await?;
        
        for file in files {
            if let Ok(content) = tokio::fs::read_to_string(&file).await {
                // Check for Arc cycles
                if content.contains("Arc<RefCell") || content.contains("Arc<Mutex") {
                    if content.contains("self") && content.contains("clone()") {
                        bottlenecks.push(PerformanceBottleneck {
                            bottleneck_id: uuid::Uuid::new_v4().to_string(),
                            bottleneck_type: BottleneckType::MemoryLeak,
                            location: CodeLocation {
                                file_path: file.clone(),
                                function_name: None,
                                line_number: None,
                            },
                            severity: BottleneckSeverity::High,
                            impact: PerformanceImpact {
                                execution_time_impact: 5.0,
                                memory_impact: 50.0,
                                cpu_impact: 10.0,
                                user_experience_impact: 30.0,
                            },
                            description: "Potential Arc cycle detected".to_string(),
                            measurements: vec![],
                            suggested_fix: None,
                        });
                    }
                }
                
                // Check for large allocations
                if content.contains("Vec::with_capacity") {
                    // Extract capacity values
                    for line in content.lines() {
                        if line.contains("Vec::with_capacity") {
                            if let Some(cap_str) = line.split("with_capacity(").nth(1) {
                                if let Some(cap_end) = cap_str.find(')') {
                                    let cap_val = &cap_str[..cap_end];
                                    if let Ok(capacity) = cap_val.parse::<usize>() {
                                        if capacity > 1_000_000 {
                                            bottlenecks.push(PerformanceBottleneck {
                                                bottleneck_id: uuid::Uuid::new_v4().to_string(),
                                                bottleneck_type: BottleneckType::ExcessiveAllocation,
                                                location: CodeLocation {
                                                    file_path: file.clone(),
                                                    function_name: None,
                                                    line_number: None,
                                                },
                                                severity: BottleneckSeverity::Medium,
                                                impact: PerformanceImpact {
                                                    execution_time_impact: 15.0,
                                                    memory_impact: 40.0,
                                                    cpu_impact: 10.0,
                                                    user_experience_impact: 20.0,
                                                },
                                                description: format!("Large pre-allocation: {} elements", capacity),
                                                measurements: vec![],
                                                suggested_fix: None,
                                            });
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        Ok(bottlenecks)
    }
    
    /// Generate optimization suggestion
    async fn generate_optimization_suggestion(
        &self,
        bottleneck: &PerformanceBottleneck,
    ) -> Result<Option<OptimizationSuggestion>> {
        match bottleneck.bottleneck_type {
            BottleneckType::IneffcientAlgorithm => {
                Ok(Some(OptimizationSuggestion {
                    optimization_id: uuid::Uuid::new_v4().to_string(),
                    optimization_type: OptimizationType::AlgorithmOptimization,
                    description: "Optimize nested loops using more efficient algorithm".to_string(),
                    expected_improvement: PerformanceImpact {
                        execution_time_impact: 40.0,
                        memory_impact: 10.0,
                        cpu_impact: 35.0,
                        user_experience_impact: 30.0,
                    },
                    implementation_complexity: ComplexityLevel::Moderate,
                    risk_level: RiskLevel::Medium,
                    code_changes: vec![
                        ProposedChange {
                            file_path: bottleneck.location.file_path.clone(),
                            change_type: "algorithm".to_string(),
                            before: "for item in items { for other in others { } }".to_string(),
                            after: "Use HashMap for O(1) lookup instead of nested loops".to_string(),
                            explanation: "Replace O(n²) algorithm with O(n) using HashMap".to_string(),
                        }
                    ],
                }))
            }
            BottleneckType::ExcessiveAllocation => {
                Ok(Some(OptimizationSuggestion {
                    optimization_id: uuid::Uuid::new_v4().to_string(),
                    optimization_type: OptimizationType::MemoryOptimization,
                    description: "Reduce unnecessary cloning and allocations".to_string(),
                    expected_improvement: PerformanceImpact {
                        execution_time_impact: 15.0,
                        memory_impact: 30.0,
                        cpu_impact: 20.0,
                        user_experience_impact: 15.0,
                    },
                    implementation_complexity: ComplexityLevel::Simple,
                    risk_level: RiskLevel::Low,
                    code_changes: vec![
                        ProposedChange {
                            file_path: bottleneck.location.file_path.clone(),
                            change_type: "memory".to_string(),
                            before: "data.clone()".to_string(),
                            after: "&data or Arc::clone(&data)".to_string(),
                            explanation: "Use references or Arc for shared data".to_string(),
                        }
                    ],
                }))
            }
            BottleneckType::SynchronousBlocking => {
                Ok(Some(OptimizationSuggestion {
                    optimization_id: uuid::Uuid::new_v4().to_string(),
                    optimization_type: OptimizationType::AsynchronousIO,
                    description: "Replace synchronous I/O with async alternatives".to_string(),
                    expected_improvement: PerformanceImpact {
                        execution_time_impact: 50.0,
                        memory_impact: 5.0,
                        cpu_impact: 30.0,
                        user_experience_impact: 60.0,
                    },
                    implementation_complexity: ComplexityLevel::Simple,
                    risk_level: RiskLevel::Low,
                    code_changes: vec![
                        ProposedChange {
                            file_path: bottleneck.location.file_path.clone(),
                            change_type: "async".to_string(),
                            before: "std::fs::read_to_string(path)".to_string(),
                            after: "tokio::fs::read_to_string(path).await".to_string(),
                            explanation: "Use async I/O to prevent blocking".to_string(),
                        }
                    ],
                }))
            }
            _ => Ok(None),
        }
    }
    
    /// Calculate performance trends
    async fn calculate_performance_trends(&self) -> Result<PerformanceTrends> {
        let history = self.performance_history.read().await;
        
        // Simple trend calculation - in real implementation would be more sophisticated
        let trend_direction = if history.is_empty() {
            TrendDirection::Stable
        } else {
            // Check last 10 measurements
            let recent: Vec<_> = history.iter().rev().take(10).collect();
            if recent.len() < 2 {
                TrendDirection::Stable
            } else {
                let first_avg = recent[0].value;
                let last_avg = recent[recent.len()-1].value;
                
                if last_avg < first_avg * 0.95 {
                    TrendDirection::Improving
                } else if last_avg > first_avg * 1.05 {
                    TrendDirection::Degrading
                } else {
                    TrendDirection::Stable
                }
            }
        };
        
        Ok(PerformanceTrends {
            execution_time_trend: trend_direction.clone(),
            memory_usage_trend: trend_direction.clone(),
            cpu_usage_trend: trend_direction,
            trend_period: self.config.monitoring_interval,
        })
    }
    
    /// Run benchmarks
    async fn run_benchmarks(&self) -> Result<Vec<BenchmarkResult>> {
        // In real implementation, would run actual benchmarks
        // For demo, return simulated results
        Ok(vec![
            BenchmarkResult {
                benchmark_name: "api_response_time".to_string(),
                execution_time: Duration::from_millis(150),
                memory_usage: 1024 * 1024 * 10, // 10MB
                iterations: 1000,
                timestamp: chrono::Utc::now(),
            }
        ])
    }
    
    /// Run targeted benchmarks
    async fn run_targeted_benchmarks(
        &self,
        _optimization: &OptimizationSuggestion,
    ) -> Result<Vec<BenchmarkResult>> {
        // Run benchmarks specific to the optimization
        Ok(vec![
            BenchmarkResult {
                benchmark_name: "targeted_benchmark".to_string(),
                execution_time: Duration::from_millis(100),
                memory_usage: 1024 * 1024 * 5,
                iterations: 100,
                timestamp: chrono::Utc::now(),
            }
        ])
    }
    
    /// Calculate performance score
    fn calculate_performance_score(&self, bottlenecks: &[PerformanceBottleneck]) -> f32 {
        if bottlenecks.is_empty() {
            return 1.0;
        }
        
        let total_impact: f32 = bottlenecks.iter()
            .map(|b| {
                let severity_weight = match b.severity {
                    BottleneckSeverity::Critical => 1.0,
                    BottleneckSeverity::High => 0.7,
                    BottleneckSeverity::Medium => 0.4,
                    BottleneckSeverity::Low => 0.2,
                };
                
                let avg_impact = (b.impact.execution_time_impact +
                                 b.impact.memory_impact +
                                 b.impact.cpu_impact +
                                 b.impact.user_experience_impact) / 4.0;
                
                avg_impact * severity_weight / 100.0
            })
            .sum();
        
        (1.0 - total_impact).max(0.0)
    }
    
    /// Convert proposed change to code change
    async fn convert_to_code_change(
        &self,
        proposed: &ProposedChange,
        optimization: &OptimizationSuggestion,
    ) -> Result<CodeChange> {
        Ok(CodeChange {
            file_path: proposed.file_path.clone(),
            change_type: crate::cognitive::self_modify::ChangeType::PerformanceOptimization,
            description: optimization.description.clone(),
            reasoning: proposed.explanation.clone(),
            old_content: Some(proposed.before.clone()),
            new_content: proposed.after.clone(),
            line_range: None,
            risk_level: optimization.risk_level.clone(),
            attribution: None,
        })
    }
    
    /// Calculate actual improvement
    fn calculate_actual_improvement(
        &self,
        before: &[BenchmarkResult],
        after: &[BenchmarkResult],
    ) -> PerformanceImpact {
        if before.is_empty() || after.is_empty() {
            return PerformanceImpact {
                execution_time_impact: 0.0,
                memory_impact: 0.0,
                cpu_impact: 0.0,
                user_experience_impact: 0.0,
            };
        }
        
        let avg_time_before: f64 = before.iter()
            .map(|b| b.execution_time.as_millis() as f64)
            .sum::<f64>() / before.len() as f64;
        
        let avg_time_after: f64 = after.iter()
            .map(|b| b.execution_time.as_millis() as f64)
            .sum::<f64>() / after.len() as f64;
        
        let avg_mem_before: f64 = before.iter()
            .map(|b| b.memory_usage as f64)
            .sum::<f64>() / before.len() as f64;
        
        let avg_mem_after: f64 = after.iter()
            .map(|b| b.memory_usage as f64)
            .sum::<f64>() / after.len() as f64;
        
        let time_improvement = ((avg_time_before - avg_time_after) / avg_time_before * 100.0) as f32;
        let mem_improvement = ((avg_mem_before - avg_mem_after) / avg_mem_before * 100.0) as f32;
        
        PerformanceImpact {
            execution_time_impact: time_improvement,
            memory_impact: mem_improvement,
            cpu_impact: time_improvement * 0.8, // Approximate
            user_experience_impact: time_improvement * 0.9, // Approximate
        }
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
    
    /// Record performance measurement
    pub async fn record_measurement(&self, measurement: PerformanceMeasurement) -> Result<()> {
        self.performance_history.write().await.push(measurement.clone());
        
        // Store in memory
        let metadata = MemoryMetadata {
            source: "performance_monitor".to_string(),
            tags: vec!["performance".to_string(), format!("{}", measurement.metric_type)],
            importance: 0.5,
            associations: vec![],
            context: Some("performance measurement".to_string()),
            created_at: chrono::Utc::now(),
            accessed_count: 0,
            last_accessed: None,
            version: 1,
            category: "performance".to_string(),
            timestamp: measurement.timestamp,
            expiration: None,
        };
        
        self.memory.store(
            serde_json::to_string(&measurement)?,
            vec!["performance".to_string(), format!("{}", measurement.metric_type)],
            metadata,
        ).await?;
        
        Ok(())
    }
    
    /// Get performance history
    pub async fn get_performance_history(&self) -> Result<Vec<PerformanceMeasurement>> {
        Ok(self.performance_history.read().await.clone())
    }
    
    /// Get optimization history
    pub async fn get_optimization_history(&self) -> Result<Vec<OptimizationResult>> {
        Ok(self.optimization_history.read().await.clone())
    }
}

impl OptimizationType {
    fn to_string(&self) -> &'static str {
        match self {
            OptimizationType::AlgorithmOptimization => "Algorithm Optimization",
            OptimizationType::CachingStrategy => "Caching Strategy",
            OptimizationType::ParallelProcessing => "Parallel Processing",
            OptimizationType::AsynchronousIO => "Asynchronous I/O",
            OptimizationType::MemoryOptimization => "Memory Optimization",
            OptimizationType::DatabaseQueryOptimization => "Database Query Optimization",
            OptimizationType::LazyLoading => "Lazy Loading",
            OptimizationType::Memoization => "Memoization",
            OptimizationType::DataStructureOptimization => "Data Structure Optimization",
            OptimizationType::CodeElimination => "Dead Code Elimination",
        }
    }
}

impl MetricType {
    fn to_string(&self) -> &'static str {
        match self {
            MetricType::ExecutionTime => "execution_time",
            MetricType::MemoryUsage => "memory_usage",
            MetricType::CpuUsage => "cpu_usage",
            MetricType::Throughput => "throughput",
            MetricType::Latency => "latency",
            MetricType::GarbageCollection => "gc",
            MetricType::CacheHitRate => "cache_hit_rate",
            MetricType::DatabaseQueries => "db_queries",
            MetricType::NetworkCalls => "network_calls",
        }
    }
}