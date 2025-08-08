//! Model Benchmarking System
//! 
//! Provides comprehensive benchmarking capabilities for AI models including
//! latency, throughput, accuracy, and cost analysis.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use chrono::{DateTime, Utc};
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};
use anyhow::Result;
use tracing::{info, debug, warn};

use crate::models::providers::{ModelProvider, CompletionRequest, Message, MessageRole};

/// Benchmark suite for testing models
pub struct BenchmarkSuite {
    /// Available benchmark tests
    tests: Vec<BenchmarkTest>,
    
    /// Benchmark results
    results: Arc<RwLock<HashMap<String, BenchmarkResult>>>,
    
    /// Benchmark configuration
    config: BenchmarkConfig,
    
    /// Test prompts
    prompts: TestPromptSet,
}

/// Benchmark configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// Number of iterations per test
    pub iterations: usize,
    
    /// Warm-up iterations
    pub warmup_iterations: usize,
    
    /// Test timeout
    pub timeout: Duration,
    
    /// Test in parallel
    pub parallel: bool,
    
    /// Save results to file
    pub save_results: bool,
    
    /// Results file path
    pub results_path: Option<String>,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            iterations: 10,
            warmup_iterations: 2,
            timeout: Duration::from_secs(30),
            parallel: false,
            save_results: true,
            results_path: Some("benchmark_results.json".to_string()),
        }
    }
}

/// Individual benchmark test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkTest {
    pub id: String,
    pub name: String,
    pub description: String,
    pub test_type: TestType,
    pub prompts: Vec<TestPrompt>,
    pub expected_outputs: Option<Vec<String>>,
    pub scoring_criteria: ScoringCriteria,
}

/// Types of benchmark tests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TestType {
    Latency,
    Throughput,
    Accuracy,
    Consistency,
    ContextRetention,
    CodeGeneration,
    Reasoning,
    CreativeWriting,
    Summarization,
    Translation,
}

/// Test prompt
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestPrompt {
    pub id: String,
    pub content: String,
    pub expected_tokens: Option<usize>,
    pub category: String,
    pub difficulty: Difficulty,
}

/// Difficulty levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Difficulty {
    Easy,
    Medium,
    Hard,
    Expert,
}

/// Scoring criteria for evaluating results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoringCriteria {
    pub measure_latency: bool,
    pub measure_tokens: bool,
    pub measure_accuracy: bool,
    pub accuracy_method: Option<AccuracyMethod>,
    pub custom_scorer: Option<String>,
    pub coherence: f32,
    pub completeness: f32,
    pub accuracy: f32,
    pub relevance: f32,
}

/// Accuracy measurement methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AccuracyMethod {
    ExactMatch,
    Contains,
    Similarity,
    CustomRegex(String),
    HumanEval,
}

/// Benchmark result for a model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub model_id: String,
    pub timestamp: DateTime<Utc>,
    pub test_results: HashMap<String, TestResult>,
    pub overall_score: f64,
    pub latency_stats: LatencyStats,
    pub throughput_stats: ThroughputStats,
    pub accuracy_stats: AccuracyStats,
    pub cost_analysis: CostAnalysis,
}

/// Individual test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    pub test_id: String,
    pub iterations: Vec<IterationResult>,
    pub avg_latency_ms: f64,
    pub avg_tokens_per_second: f64,
    pub accuracy_score: f64,
    pub consistency_score: f64,
}

/// Single iteration result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IterationResult {
    pub iteration: usize,
    pub latency_ms: f64,
    pub tokens_generated: usize,
    pub tokens_per_second: f64,
    pub output: String,
    pub score: f64,
}

/// Latency statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyStats {
    pub min_ms: f64,
    pub max_ms: f64,
    pub mean_ms: f64,
    pub median_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub std_dev_ms: f64,
}

/// Throughput statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputStats {
    pub min_tps: f64,
    pub max_tps: f64,
    pub mean_tps: f64,
    pub median_tps: f64,
    pub total_tokens: usize,
    pub total_time_s: f64,
}

/// Accuracy statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccuracyStats {
    pub overall_accuracy: f64,
    pub by_category: HashMap<String, f64>,
    pub by_difficulty: HashMap<String, f64>,
    pub consistency_score: f64,
}

/// Cost analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostAnalysis {
    pub total_input_tokens: usize,
    pub total_output_tokens: usize,
    pub estimated_input_cost: f64,
    pub estimated_output_cost: f64,
    pub total_cost: f64,
    pub cost_per_request: f64,
    pub cost_efficiency_score: f64,
}

/// Standard test prompt set
pub struct TestPromptSet {
    pub latency_tests: Vec<TestPrompt>,
    pub accuracy_tests: Vec<TestPrompt>,
    pub code_tests: Vec<TestPrompt>,
    pub reasoning_tests: Vec<TestPrompt>,
    pub creative_tests: Vec<TestPrompt>,
}

impl TestPromptSet {
    /// Create standard test prompts
    pub fn standard() -> Self {
        Self {
            latency_tests: vec![
                TestPrompt {
                    id: "lat_1".to_string(),
                    content: "Hello".to_string(),
                    expected_tokens: Some(10),
                    category: "greeting".to_string(),
                    difficulty: Difficulty::Easy,
                },
                TestPrompt {
                    id: "lat_2".to_string(),
                    content: "What is 2+2?".to_string(),
                    expected_tokens: Some(20),
                    category: "math".to_string(),
                    difficulty: Difficulty::Easy,
                },
            ],
            accuracy_tests: vec![
                TestPrompt {
                    id: "acc_1".to_string(),
                    content: "What is the capital of France?".to_string(),
                    expected_tokens: Some(10),
                    category: "geography".to_string(),
                    difficulty: Difficulty::Easy,
                },
                TestPrompt {
                    id: "acc_2".to_string(),
                    content: "Explain quantum entanglement in simple terms.".to_string(),
                    expected_tokens: Some(200),
                    category: "science".to_string(),
                    difficulty: Difficulty::Hard,
                },
            ],
            code_tests: vec![
                TestPrompt {
                    id: "code_1".to_string(),
                    content: "Write a Python function to calculate fibonacci numbers.".to_string(),
                    expected_tokens: Some(150),
                    category: "coding".to_string(),
                    difficulty: Difficulty::Medium,
                },
                TestPrompt {
                    id: "code_2".to_string(),
                    content: "Implement a binary search tree in Rust with insert and search methods.".to_string(),
                    expected_tokens: Some(300),
                    category: "coding".to_string(),
                    difficulty: Difficulty::Hard,
                },
            ],
            reasoning_tests: vec![
                TestPrompt {
                    id: "reason_1".to_string(),
                    content: "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?".to_string(),
                    expected_tokens: Some(100),
                    category: "logic".to_string(),
                    difficulty: Difficulty::Medium,
                },
            ],
            creative_tests: vec![
                TestPrompt {
                    id: "creative_1".to_string(),
                    content: "Write a haiku about artificial intelligence.".to_string(),
                    expected_tokens: Some(30),
                    category: "creative".to_string(),
                    difficulty: Difficulty::Medium,
                },
            ],
        }
    }
}

impl BenchmarkSuite {
    /// Create a new benchmark suite
    pub fn new() -> Self {
        Self {
            tests: Self::create_standard_tests(),
            results: Arc::new(RwLock::new(HashMap::new())),
            config: BenchmarkConfig::default(),
            prompts: TestPromptSet::standard(),
        }
    }
    
    /// Create standard benchmark tests
    fn create_standard_tests() -> Vec<BenchmarkTest> {
        let prompts = TestPromptSet::standard();
        
        vec![
            BenchmarkTest {
                id: "latency_test".to_string(),
                name: "Latency Test".to_string(),
                description: "Measure response latency".to_string(),
                test_type: TestType::Latency,
                prompts: prompts.latency_tests.clone(),
                expected_outputs: None,
                scoring_criteria: ScoringCriteria {
                    measure_latency: true,
                    measure_tokens: true,
                    measure_accuracy: false,
                    accuracy_method: None,
                    custom_scorer: None,
                    coherence: 0.0,
                    completeness: 0.0,
                    accuracy: 0.0,
                    relevance: 0.0,
                },
            },
            BenchmarkTest {
                id: "throughput_test".to_string(),
                name: "Throughput Test".to_string(),
                description: "Measure tokens per second".to_string(),
                test_type: TestType::Throughput,
                prompts: prompts.accuracy_tests.clone(),
                expected_outputs: None,
                scoring_criteria: ScoringCriteria {
                    measure_latency: true,
                    measure_tokens: true,
                    measure_accuracy: false,
                    accuracy_method: None,
                    custom_scorer: None,
                    coherence: 0.0,
                    completeness: 0.0,
                    accuracy: 0.0,
                    relevance: 0.0,
                },
            },
            BenchmarkTest {
                id: "code_gen_test".to_string(),
                name: "Code Generation Test".to_string(),
                description: "Test code generation capabilities".to_string(),
                test_type: TestType::CodeGeneration,
                prompts: prompts.code_tests.clone(),
                expected_outputs: None,
                scoring_criteria: ScoringCriteria {
                    measure_latency: true,
                    measure_tokens: true,
                    measure_accuracy: true,
                    accuracy_method: Some(AccuracyMethod::Contains),
                    custom_scorer: None,
                    coherence: 0.8,
                    completeness: 0.9,
                    accuracy: 0.7,
                    relevance: 0.8,
                },
            },
        ]
    }
    
    /// Benchmark a specific model
    pub async fn benchmark_model(
        &mut self,
        model_id: &str,
        provider: Arc<dyn ModelProvider>,
    ) -> Result<BenchmarkResult> {
        info!("Starting benchmark for model: {}", model_id);
        
        let mut test_results = HashMap::new();
        let start_time = std::time::Instant::now();
        
        // Run each test
        for test in &self.tests {
            debug!("Running test: {}", test.name);
            let result = self.run_test(test, provider.clone()).await?;
            test_results.insert(test.id.clone(), result);
        }
        
        // Calculate statistics
        let latency_stats = self.calculate_latency_stats(&test_results);
        let throughput_stats = self.calculate_throughput_stats(&test_results);
        let accuracy_stats = self.calculate_accuracy_stats(&test_results);
        let cost_analysis = self.calculate_cost_analysis(&test_results, 0.001, 0.002); // Default pricing
        
        // Calculate overall score
        let overall_score = self.calculate_overall_score(
            &latency_stats,
            &throughput_stats,
            &accuracy_stats,
            &cost_analysis,
        );
        
        let result = BenchmarkResult {
            model_id: model_id.to_string(),
            timestamp: Utc::now(),
            test_results,
            overall_score,
            latency_stats,
            throughput_stats,
            accuracy_stats,
            cost_analysis,
        };
        
        // Store result
        self.results.write().await.insert(model_id.to_string(), result.clone());
        
        info!("Benchmark complete for {}: overall score = {:.2}", model_id, overall_score);
        
        Ok(result)
    }
    
    /// Run a single test
    async fn run_test(
        &self,
        test: &BenchmarkTest,
        provider: Arc<dyn ModelProvider>,
    ) -> Result<TestResult> {
        let mut iterations = Vec::new();
        let mut latencies = Vec::new();
        let mut tokens_per_second_list = Vec::new();
        
        // Warmup
        for _ in 0..self.config.warmup_iterations {
            let prompt = &test.prompts[0];
            let request = self.create_request(&prompt.content);
            let _ = provider.complete(request).await;
        }
        
        // Actual test iterations
        let mut outputs = Vec::new();
        for i in 0..self.config.iterations {
            for prompt in &test.prompts {
                let request = self.create_request(&prompt.content);
                
                let start = std::time::Instant::now();
                let response = provider.complete(request).await?;
                let latency = start.elapsed();
                
                let latency_ms = latency.as_millis() as f64;
                let tokens_generated = self.count_tokens(&response.content);
                let tokens_per_second = tokens_generated as f64 / latency.as_secs_f64();
                
                latencies.push(latency_ms);
                tokens_per_second_list.push(tokens_per_second);
                outputs.push(response.content.clone());
                
                iterations.push(IterationResult {
                    iteration: i,
                    latency_ms,
                    tokens_generated,
                    tokens_per_second,
                    score: self.score_output(&prompt.content, &response.content, &test.scoring_criteria),
                    output: response.content,
                });
            }
        }
        
        Ok(TestResult {
            test_id: test.id.clone(),
            iterations,
            avg_latency_ms: latencies.iter().sum::<f64>() / latencies.len() as f64,
            avg_tokens_per_second: tokens_per_second_list.iter().sum::<f64>() / tokens_per_second_list.len() as f64,
            accuracy_score: self.calculate_accuracy_score(&outputs, &test.scoring_criteria),
            consistency_score: self.calculate_consistency_score(&outputs)
        })
    }
    
    /// Create a completion request
    fn create_request(&self, prompt: &str) -> CompletionRequest {
        CompletionRequest {
            model: String::new(), // Will be set by provider
            messages: vec![Message {
                role: MessageRole::User,
                content: prompt.to_string(),
            }],
            max_tokens: Some(500),
            temperature: Some(0.7),
            top_p: Some(0.9),
            stop: None,
            stream: false,
        }
    }
    
    /// Count tokens in text (simple approximation)
    fn count_tokens(&self, text: &str) -> usize {
        // Simple approximation: ~4 characters per token
        text.len() / 4
    }
    
    /// Score output based on criteria
    fn score_output(&self, prompt: &str, output: &str, criteria: &ScoringCriteria) -> f64 {
        let mut score: f64 = 0.0;
        let mut weight_sum: f64 = 0.0;
        
        // Evaluate coherence - does output relate to prompt?
        if criteria.coherence > 0.0 {
            let coherence_score = self.calculate_coherence(prompt, output);
            score += coherence_score * criteria.coherence as f64;
            weight_sum += criteria.coherence as f64;
        }
        
        // Evaluate completeness - is the response complete?
        if criteria.completeness > 0.0 {
            let completeness_score = self.calculate_completeness(output);
            score += completeness_score * criteria.completeness as f64;
            weight_sum += criteria.completeness as f64;
        }
        
        // Evaluate accuracy - for factual/code tasks
        if criteria.accuracy > 0.0 {
            let accuracy_score = self.calculate_accuracy(prompt, output);
            score += accuracy_score * criteria.accuracy as f64;
            weight_sum += criteria.accuracy as f64;
        }
        
        // Evaluate relevance
        if criteria.relevance > 0.0 {
            let relevance_score = self.calculate_relevance(prompt, output);
            score += relevance_score * criteria.relevance as f64;
            weight_sum += criteria.relevance as f64;
        }
        
        // Normalize by weight sum
        if weight_sum > 0.0 {
            score / weight_sum
        } else {
            0.5 // Default neutral score
        }
    }
    
    /// Calculate coherence score
    fn calculate_coherence(&self, prompt: &str, output: &str) -> f64 {
        // Check if output contains key terms from prompt
        let prompt_words: Vec<&str> = prompt.split_whitespace()
            .filter(|w| w.len() > 3)
            .collect();
        
        let mut matches = 0;
        for word in &prompt_words {
            if output.to_lowercase().contains(&word.to_lowercase()) {
                matches += 1;
            }
        }
        
        // Score based on term overlap
        if prompt_words.is_empty() {
            0.5
        } else {
            (matches as f64 / prompt_words.len() as f64).min(1.0)
        }
    }
    
    /// Calculate completeness score
    fn calculate_completeness(&self, output: &str) -> f64 {
        // Check for complete sentences and proper structure
        let has_punctuation = output.ends_with('.') || output.ends_with('!') || output.ends_with('?');
        let word_count = output.split_whitespace().count();
        let has_substance = word_count >= 5;
        
        let mut score: f64 = 0.0;
        if has_punctuation { score += 0.3; }
        if has_substance { score += 0.4; }
        if word_count >= 10 { score += 0.3; }
        
        score.min(1.0)
    }
    
    /// Calculate accuracy score (simplified)
    fn calculate_accuracy(&self, prompt: &str, output: &str) -> f64 {
        // For code-related prompts, check for code blocks
        if prompt.contains("code") || prompt.contains("function") || prompt.contains("implement") {
            if output.contains("```") || output.contains("fn ") || output.contains("def ") {
                return 0.9;
            }
        }
        
        // For math prompts, check for numbers
        if prompt.contains("calculate") || prompt.contains("math") || prompt.contains("number") {
            if output.chars().any(|c| c.is_numeric()) {
                return 0.8;
            }
        }
        
        // Default reasonable accuracy
        0.7
    }
    
    /// Calculate relevance score
    fn calculate_relevance(&self, prompt: &str, output: &str) -> f64 {
        // Simple relevance based on length ratio and keyword matching
        let prompt_len = prompt.len() as f64;
        let output_len = output.len() as f64;
        
        // Penalize very short or very long responses
        let length_ratio = if output_len < prompt_len * 0.5 {
            0.5 // Too short
        } else if output_len > prompt_len * 10.0 {
            0.6 // Too verbose
        } else {
            0.9 // Good length ratio
        };
        
        // Check for question-answer matching
        let is_question = prompt.contains('?');
        let has_answer = output.len() > 10;
        
        if is_question && has_answer {
            length_ratio
        } else if !is_question {
            length_ratio * 0.9 // Slightly lower for non-questions
        } else {
            0.4 // Question without proper answer
        }
    }
    
    /// Calculate average accuracy score across outputs
    fn calculate_accuracy_score(&self, outputs: &[String], criteria: &ScoringCriteria) -> f64 {
        if outputs.is_empty() {
            return 0.0;
        }
        
        // For accuracy, we look at the quality of each individual output
        let mut total_score = 0.0;
        for output in outputs {
            // Simple heuristics for accuracy
            let mut score: f64 = 0.0;
            
            // Check output is not empty
            if !output.is_empty() {
                score += 0.2;
            }
            
            // Check output has reasonable length
            if output.len() >= 10 {
                score += 0.3;
            }
            
            // Check for complete sentences
            if output.contains(". ") || output.ends_with('.') {
                score += 0.2;
            }
            
            // Apply criteria weights if provided
            if criteria.accuracy > 0.0 {
                score *= criteria.accuracy as f64;
            }
            
            // Additional quality checks
            if !output.contains("error") && !output.contains("failed") {
                score += 0.3;
            }
            
            total_score += score.min(1.0);
        }
        
        total_score / outputs.len() as f64
    }
    
    /// Calculate consistency score across outputs
    fn calculate_consistency_score(&self, outputs: &[String]) -> f64 {
        if outputs.len() <= 1 {
            return 1.0; // Single output is perfectly consistent
        }
        
        // Calculate similarity between outputs
        let mut similarity_sum = 0.0;
        let mut comparison_count = 0;
        
        for i in 0..outputs.len() {
            for j in i+1..outputs.len() {
                let similarity = self.calculate_output_similarity(&outputs[i], &outputs[j]);
                similarity_sum += similarity;
                comparison_count += 1;
            }
        }
        
        if comparison_count > 0 {
            similarity_sum / comparison_count as f64
        } else {
            0.5
        }
    }
    
    /// Calculate similarity between two outputs
    fn calculate_output_similarity(&self, output1: &str, output2: &str) -> f64 {
        // Simple word overlap similarity
        let words1: Vec<&str> = output1.split_whitespace().collect();
        let words2: Vec<&str> = output2.split_whitespace().collect();
        
        if words1.is_empty() || words2.is_empty() {
            return 0.0;
        }
        
        let mut matches = 0;
        for word in &words1 {
            if words2.contains(word) {
                matches += 1;
            }
        }
        
        // Jaccard similarity
        let union_size = (words1.len() + words2.len() - matches) as f64;
        if union_size > 0.0 {
            matches as f64 / union_size
        } else {
            0.0
        }
    }
    
    /// Calculate latency statistics
    fn calculate_latency_stats(&self, results: &HashMap<String, TestResult>) -> LatencyStats {
        let mut all_latencies: Vec<f64> = Vec::new();
        
        for result in results.values() {
            for iteration in &result.iterations {
                all_latencies.push(iteration.latency_ms);
            }
        }
        
        all_latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let len = all_latencies.len();
        if len == 0 {
            return LatencyStats {
                min_ms: 0.0,
                max_ms: 0.0,
                mean_ms: 0.0,
                median_ms: 0.0,
                p95_ms: 0.0,
                p99_ms: 0.0,
                std_dev_ms: 0.0,
            };
        }
        
        let mean = all_latencies.iter().sum::<f64>() / len as f64;
        let variance = all_latencies.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / len as f64;
        
        LatencyStats {
            min_ms: all_latencies[0],
            max_ms: all_latencies[len - 1],
            mean_ms: mean,
            median_ms: all_latencies[len / 2],
            p95_ms: all_latencies[(len as f64 * 0.95) as usize],
            p99_ms: all_latencies[(len as f64 * 0.99) as usize],
            std_dev_ms: variance.sqrt(),
        }
    }
    
    /// Calculate throughput statistics
    fn calculate_throughput_stats(&self, results: &HashMap<String, TestResult>) -> ThroughputStats {
        let mut all_tps: Vec<f64> = Vec::new();
        let mut total_tokens = 0;
        let mut total_time = 0.0;
        
        for result in results.values() {
            for iteration in &result.iterations {
                all_tps.push(iteration.tokens_per_second);
                total_tokens += iteration.tokens_generated;
                total_time += iteration.latency_ms / 1000.0;
            }
        }
        
        all_tps.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let len = all_tps.len();
        if len == 0 {
            return ThroughputStats {
                min_tps: 0.0,
                max_tps: 0.0,
                mean_tps: 0.0,
                median_tps: 0.0,
                total_tokens: 0,
                total_time_s: 0.0,
            };
        }
        
        ThroughputStats {
            min_tps: all_tps[0],
            max_tps: all_tps[len - 1],
            mean_tps: all_tps.iter().sum::<f64>() / len as f64,
            median_tps: all_tps[len / 2],
            total_tokens,
            total_time_s: total_time,
        }
    }
    
    /// Calculate accuracy statistics
    fn calculate_accuracy_stats(&self, results: &HashMap<String, TestResult>) -> AccuracyStats {
        let mut scores = Vec::new();
        
        for result in results.values() {
            scores.push(result.accuracy_score);
        }
        
        let overall = if scores.is_empty() { 0.0 } else { scores.iter().sum::<f64>() / scores.len() as f64 };
        
        // Calculate consistency score from results
        let consistency = if results.is_empty() {
            0.0
        } else {
            let consistency_scores: Vec<f64> = results.values()
                .map(|r| r.consistency_score)
                .collect();
            consistency_scores.iter().sum::<f64>() / consistency_scores.len() as f64
        };
        
        AccuracyStats {
            overall_accuracy: overall,
            by_category: HashMap::new(),
            by_difficulty: HashMap::new(),
            consistency_score: consistency,
        }
    }
    
    /// Calculate cost analysis
    fn calculate_cost_analysis(
        &self,
        results: &HashMap<String, TestResult>,
        input_price_per_1k: f64,
        output_price_per_1k: f64,
    ) -> CostAnalysis {
        let mut total_input_tokens = 0;
        let mut total_output_tokens = 0;
        
        for result in results.values() {
            for iteration in &result.iterations {
                total_input_tokens += 50; // Approximate input tokens
                total_output_tokens += iteration.tokens_generated;
            }
        }
        
        let input_cost = (total_input_tokens as f64 / 1000.0) * input_price_per_1k;
        let output_cost = (total_output_tokens as f64 / 1000.0) * output_price_per_1k;
        let total_cost = input_cost + output_cost;
        
        CostAnalysis {
            total_input_tokens,
            total_output_tokens,
            estimated_input_cost: input_cost,
            estimated_output_cost: output_cost,
            total_cost,
            cost_per_request: total_cost / results.len() as f64,
            cost_efficiency_score: 1.0 / (total_cost + 0.001), // Avoid division by zero
        }
    }
    
    /// Calculate overall score
    fn calculate_overall_score(
        &self,
        latency: &LatencyStats,
        throughput: &ThroughputStats,
        accuracy: &AccuracyStats,
        cost: &CostAnalysis,
    ) -> f64 {
        // Weighted scoring
        let latency_score = 100.0 / (latency.mean_ms + 1.0); // Lower is better
        let throughput_score = throughput.mean_tps;
        let accuracy_score = accuracy.overall_accuracy * 100.0;
        let cost_score = cost.cost_efficiency_score * 10.0;
        
        // Weighted average
        (latency_score * 0.25 + throughput_score * 0.25 + accuracy_score * 0.35 + cost_score * 0.15) / 100.0
    }
    
    /// Get benchmark results for a model
    pub async fn get_results(&self, model_id: &str) -> Option<BenchmarkResult> {
        self.results.read().await.get(model_id).cloned()
    }
    
    /// Get all benchmark results
    pub async fn get_all_results(&self) -> HashMap<String, BenchmarkResult> {
        self.results.read().await.clone()
    }
    
    /// Compare two models
    pub async fn compare_models(&self, model1_id: &str, model2_id: &str) -> Option<ModelBenchmarkComparison> {
        let results = self.results.read().await;
        let result1 = results.get(model1_id)?;
        let result2 = results.get(model2_id)?;
        
        Some(ModelBenchmarkComparison {
            model1_id: model1_id.to_string(),
            model2_id: model2_id.to_string(),
            latency_diff_ms: result1.latency_stats.mean_ms - result2.latency_stats.mean_ms,
            throughput_diff_tps: result1.throughput_stats.mean_tps - result2.throughput_stats.mean_tps,
            accuracy_diff: result1.accuracy_stats.overall_accuracy - result2.accuracy_stats.overall_accuracy,
            cost_diff: result1.cost_analysis.total_cost - result2.cost_analysis.total_cost,
            overall_score_diff: result1.overall_score - result2.overall_score,
            winner: if result1.overall_score > result2.overall_score { model1_id.to_string() } else { model2_id.to_string() },
        })
    }
}

/// Model benchmark comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelBenchmarkComparison {
    pub model1_id: String,
    pub model2_id: String,
    pub latency_diff_ms: f64,
    pub throughput_diff_tps: f64,
    pub accuracy_diff: f64,
    pub cost_diff: f64,
    pub overall_score_diff: f64,
    pub winner: String,
}