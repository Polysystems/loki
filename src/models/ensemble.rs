use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;
use tracing::{debug, info, warn};
use futures::stream::StreamExt;

use super::local_manager::LocalModelManager;
use super::orchestrator::{ModelSelection, TaskRequest, TaskResponse};
use super::providers::ModelProvider;

/// Multi-model ensemble system for improved response quality
pub struct ModelEnsemble {
    ensembleconfig: EnsembleConfig,
    local_manager: Arc<LocalModelManager>,
    api_providers: HashMap<String, Arc<dyn ModelProvider>>,
    quality_assessor: Arc<ResponseQualityAssessor>,
    parallel_executor: Arc<ParallelExecutor>,
}

impl std::fmt::Debug for ModelEnsemble {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelEnsemble")
            .field("ensembleconfig", &self.ensembleconfig)
            .field("local_manager", &self.local_manager)
            .field("api_providers", &self.api_providers.keys().collect::<Vec<_>>())
            .field("quality_assessor", &self.quality_assessor)
            .field("parallel_executor", &self.parallel_executor)
            .finish()
    }
}

/// Configuration for ensemble behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleConfig {
    pub voting_strategy: VotingStrategy,
    pub min_models: usize,
    pub max_models: usize,
    pub quality_threshold: f32,
    pub max_parallel_requests: usize,
    pub timeout_seconds: u64,
    pub consensus_threshold: f32,
    pub diversity_bonus: f32,
    pub cost_weight: f32,
    pub latency_weight: f32,
    pub quality_weight: f32,
}

/// Voting strategies for combining multiple model responses
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum VotingStrategy {
    /// Select the response with highest quality score
    BestQuality,

    /// Combine responses using weighted averaging
    WeightedAverage { weights: HashMap<String, f32> },

    /// Require consensus between models (similarity threshold)
    Consensus { similarity_threshold: f32 },

    /// Route different parts to different specialized models
    SpecializedRouting,

    /// Majority vote on discrete choices
    MajorityVote,

    /// Use multiple models for validation and correction
    ValidateAndCorrect,

    /// Generate multiple candidates and select best
    GenerateAndSelect { candidates: usize },
}

/// Result from ensemble execution
#[derive(Debug, Clone)]
pub struct EnsembleResponse {
    pub primary_response: TaskResponse,
    pub contributing_models: Vec<String>,
    pub consensus_score: f32,
    pub quality_score: f32,
    pub diversity_score: f32,
    pub individual_responses: Vec<IndividualResponse>,
    pub voting_details: VotingDetails,
    pub execution_metrics: EnsembleMetrics,
}

/// Individual model response within ensemble
#[derive(Debug, Clone)]
pub struct IndividualResponse {
    pub model_id: String,
    pub response: TaskResponse,
    pub quality_score: f32,
    pub confidence_score: f32,
    pub latency_ms: u32,
    pub tokens_generated: u32,
    pub cost_cents: Option<f32>,
}

/// Details about the voting process
#[derive(Debug, Clone)]
pub struct VotingDetails {
    pub strategy_used: VotingStrategy,
    pub models_participated: usize,
    pub models_agreed: usize,
    pub disagreement_areas: Vec<String>,
    pub confidence_distribution: HashMap<String, f32>,
}

/// Metrics for ensemble execution
#[derive(Debug, Clone)]
pub struct EnsembleMetrics {
    pub total_execution_time: Duration,
    pub parallel_efficiency: f32,
    pub resource_utilization: f32,
    pub cost_effectiveness: f32,
    pub quality_improvement: f32,
}

impl ModelEnsemble {
    /// Create a new model ensemble
    pub fn new(
        ensembleconfig: EnsembleConfig,
        local_manager: Arc<LocalModelManager>,
        api_providers: HashMap<String, Arc<dyn ModelProvider>>,
    ) -> Self {
        Self {
            ensembleconfig,
            local_manager,
            api_providers,
            quality_assessor: Arc::new(ResponseQualityAssessor::new()),
            parallel_executor: Arc::new(ParallelExecutor::new()),
        }
    }

    /// Execute task using ensemble of models
    pub async fn execute_ensemble(&self, task: TaskRequest) -> Result<EnsembleResponse> {
        let start_time = Instant::now();
        info!("Starting ensemble execution for task: {:?}", task.task_type);

        // Select models for ensemble based on task and configuration
        let selected_models = self.select_ensemble_models(&task).await?;

        if selected_models.len() < self.ensembleconfig.min_models {
            return Err(anyhow::anyhow!(
                "Insufficient models for ensemble: {} < {}",
                selected_models.len(),
                self.ensembleconfig.min_models
            ));
        }

        info!(
            "Selected {} models for ensemble: {:?}",
            selected_models.len(),
            selected_models.iter().map(|m| &m.model_id).collect::<Vec<_>>()
        );

        // Execute in parallel with timeout
        let individual_responses =
            self.execute_parallel_requests(task.clone(), selected_models).await?;

        if individual_responses.is_empty() {
            return Err(anyhow::anyhow!("No successful responses from ensemble models"));
        }

        // Apply voting strategy to combine responses
        let (primary_response, voting_details) =
            self.apply_voting_strategy(&individual_responses, &task).await?;

        // Calculate ensemble metrics
        let execution_time = start_time.elapsed();
        let metrics = self.calculate_ensemble_metrics(&individual_responses, execution_time);

        // Calculate quality and consensus scores
        let quality_score = self
            .quality_assessor
            .assess_ensemble_quality(&individual_responses, &primary_response)
            .await;
        let consensus_score = self.calculate_consensus_score(&individual_responses);
        let diversity_score = self.calculate_diversity_score(&individual_responses);

        let ensemble_response = EnsembleResponse {
            primary_response,
            contributing_models: individual_responses.iter().map(|r| r.model_id.clone()).collect(),
            consensus_score,
            quality_score,
            diversity_score,
            individual_responses,
            voting_details,
            execution_metrics: metrics,
        };

        info!(
            "Ensemble execution completed in {:.2?} with quality score {:.2}",
            execution_time, quality_score
        );

        Ok(ensemble_response)
    }

    /// Select models for ensemble based on task requirements
    async fn select_ensemble_models(&self, task: &TaskRequest) -> Result<Vec<EnsembleModel>> {
        let mut selected_models = Vec::new();

        // Get available local models
        let local_models = self.local_manager.get_available_models().await;
        for model_id in local_models {
            if self.local_manager.can_handle_request(&model_id).await {
                selected_models.push(EnsembleModel {
                    model_id: model_id.clone(),
                    model_type: EnsembleModelType::Local,
                    capability_score: self.estimate_capability_score(&model_id, task).await,
                    cost_estimate: 0.0, // Local models have no per-request cost
                    latency_estimate: 100.0, // ms, typically faster
                });
            }
        }

        // Get available API providers
        for (provider_name, provider) in &self.api_providers {
            if provider.is_available() {
                selected_models.push(EnsembleModel {
                    model_id: provider_name.clone(),
                    model_type: EnsembleModelType::API,
                    capability_score: self.estimate_api_capability_score(provider_name, task).await,
                    cost_estimate: self.estimate_api_cost(provider_name, task).await,
                    latency_estimate: 500.0, // ms, typically slower
                });
            }
        }

        // Sort by capability score and select top models
        selected_models
            .sort_by(|a, b| b.capability_score.partial_cmp(&a.capability_score).unwrap());

        // Apply diversity and cost constraints
        let filtered_models = self.apply_selection_constraints(selected_models, task).await;

        // Limit to max_models
        let final_models: Vec<_> =
            filtered_models.into_iter().take(self.ensembleconfig.max_models).collect();

        Ok(final_models)
    }

    /// Execute requests in parallel across selected models
    async fn execute_parallel_requests(
        &self,
        task: TaskRequest,
        models: Vec<EnsembleModel>,
    ) -> Result<Vec<IndividualResponse>> {
        let semaphore = Arc::new(Semaphore::new(self.ensembleconfig.max_parallel_requests));
        let timeout_duration = Duration::from_secs(self.ensembleconfig.timeout_seconds);

        let mut tasks = Vec::new();

        for model in models {
            let task_clone = task.clone();
            let semaphore_clone = semaphore.clone();
            let local_manager = self.local_manager.clone();
            let api_providers = self.api_providers.clone();

            let task_handle = tokio::spawn(async move {
                // Graceful semaphore acquisition with timeout handling
                let _permit = match tokio::time::timeout(
                    Duration::from_secs(30),
                    semaphore_clone.acquire()
                ).await {
                    Ok(Ok(permit)) => permit,
                    Ok(Err(_)) => {
                        warn!("Semaphore acquisition failed for model {}", model.model_id);
                        return None;
                    }
                    Err(_) => {
                        warn!("Semaphore acquisition timeout for model {}", model.model_id);
                        return None;
                    }
                };

                let start_time = Instant::now();

                let result = match model.model_type {
                    EnsembleModelType::Local => {
                        Self::execute_local_model(&local_manager, &model.model_id, task_clone).await
                    }
                    EnsembleModelType::API => {
                        Self::execute_api_model(&api_providers, &model.model_id, task_clone).await
                    }
                };

                let latency_ms = start_time.elapsed().as_millis() as u32;

                match result {
                    Ok(response) => {
                        Some(IndividualResponse {
                            model_id: model.model_id,
                            response,
                            quality_score: 0.0,    // Will be calculated later
                            confidence_score: 0.0, // Will be calculated later
                            latency_ms,
                            tokens_generated: 0, // Will be extracted from response
                            cost_cents: model.cost_estimate.into(),
                        })
                    }
                    Err(e) => {
                        warn!("Model {} failed: {}", model.model_id, e);
                        None
                    }
                }
            });

            tasks.push(task_handle);
        }

        // Enhanced streaming ensemble processing - process results as they arrive
        let mut responses = Vec::new();
        let mut completed_count = 0;
        let total_models = tasks.len();
        let min_required_responses = (total_models / 2).max(1); // Require at least half responses

        // Convert task handles to a FuturesUnordered for streaming processing
        let mut pending_tasks = futures::stream::FuturesUnordered::from_iter(tasks);
        let timeout_task = tokio::time::sleep(timeout_duration);

        tokio::pin!(timeout_task);

        let mut early_decision_possible = false;

        // Stream processing with early termination capability
        loop {
            tokio::select! {
                _ = &mut timeout_task => {
                    warn!("Ensemble execution timed out after {:?} with {}/{} responses",
                          timeout_duration, completed_count, total_models);
                    break;
                }

                Some(result) = pending_tasks.next() => {
                    completed_count += 1;

                    if let Ok(Some(response)) = result {
                        responses.push(response);

                        // Check for early decision possibility
                        if responses.len() >= min_required_responses {
                            // Quick quality assessment for early decision
                            let high_quality_count = responses.iter()
                                .filter(|r| r.quality_score > 0.8 || r.response.content.len() > 100)
                                .count();

                            if high_quality_count >= min_required_responses {
                                early_decision_possible = true;
                                debug!("Early decision possible with {}/{} high-quality responses",
                                       high_quality_count, responses.len());
                            }
                        }
                    }

                    // Check if we should terminate early
                    if completed_count >= total_models ||
                       (responses.len() >= min_required_responses && early_decision_possible) {
                        break;
                    }
                }

                else => break, // No more pending tasks
            }
        }

        info!("Ensemble processing completed: {}/{} responses received", responses.len(), total_models);

        // Assess quality scores for all responses
        for response in &mut responses {
            response.quality_score =
                self.quality_assessor.assess_response_quality(&response.response, &task).await;
            response.confidence_score =
                self.quality_assessor.assess_confidence(&response.response).await;
        }

        Ok(responses)
    }

    /// Execute task on local model
    async fn execute_local_model(
        local_manager: &Arc<LocalModelManager>,
        model_id: &str,
        task: TaskRequest,
    ) -> Result<TaskResponse> {
        let instance = local_manager
            .get_model(model_id)
            .await
            .ok_or_else(|| anyhow::anyhow!("Local model not found: {}", model_id))?;

        let request =
            super::local_manager::LocalGenerationRequest { prompt: task.content, options: None };

        let response = instance.generate(request).await?;

        Ok(TaskResponse {
            content: response.text,
            model_used: ModelSelection::Local(model_id.to_string()),
            tokens_generated: Some(response.tokens_generated),
            generation_time_ms: Some(response.generation_time_ms),
            cost_cents: None,
            quality_score: 0.8, // Default quality score for local models
            cost_info: Some("Local model - no cost".to_string()),
            model_info: Some(model_id.to_string()),
            error: None,
        })
    }

    /// Execute task on API model
    async fn execute_api_model(
        api_providers: &HashMap<String, Arc<dyn ModelProvider>>,
        provider_name: &str,
        task: TaskRequest,
    ) -> Result<TaskResponse> {
        let provider = api_providers
            .get(provider_name)
            .ok_or_else(|| anyhow::anyhow!("API provider not found: {}", provider_name))?;

        // Create proper API request
        let messages = vec![super::providers::Message {
            role: super::providers::MessageRole::User,
            content: task.content.clone(),
        }];

        let request = super::providers::CompletionRequest {
            model: Self::get_default_model_for_provider(provider_name),
            messages,
            max_tokens: Some(2048),
            temperature: Some(0.7),
            top_p: None,
            stop: None,
            stream: false,
        };

        let start_time = std::time::Instant::now();
        let response = provider.complete(request).await?;
        let generation_time = start_time.elapsed().as_millis() as u32;

        Ok(TaskResponse {
            content: response.content,
            model_used: ModelSelection::API(provider_name.to_string()),
            tokens_generated: Some(response.usage.completion_tokens.try_into().unwrap_or(0)),
            generation_time_ms: Some(generation_time),
            cost_cents: Some(Self::estimate_cost(
                provider_name,
                response.usage.completion_tokens.try_into().unwrap_or(0),
            )),
            quality_score: 0.7, // Default quality score for API models
            cost_info: Some(format!("API provider: {} - estimated cost", provider_name)),
            model_info: Some(provider_name.to_string()),
            error: None,
        })
    }

    /// Get default model for provider
    fn get_default_model_for_provider(provider_name: &str) -> String {
        match provider_name {
            "openai" => "gpt-4o-mini".to_string(),
            "anthropic" => "claude-3-5-haiku-20241022".to_string(),
            "mistral" => "mistral-small-latest".to_string(),
            "gemini" => "gemini-1.5-flash".to_string(),
            _ => "default".to_string(),
        }
    }

    /// Estimate API cost based on tokens
    fn estimate_cost(provider_name: &str, tokens: u32) -> f32 {
        let cost_per_1k_tokens = match provider_name {
            "openai" => 0.15,    // GPT-4o-mini approximate
            "anthropic" => 0.25, // Claude 3.5 Haiku
            "mistral" => 0.25,   // Mistral Small
            "gemini" => 0.075,   // Gemini 1.5 Flash
            _ => 0.1,
        };

        (tokens as f32 / 1000.0) * cost_per_1k_tokens
    }

    /// Apply voting strategy to combine responses
    async fn apply_voting_strategy(
        &self,
        responses: &[IndividualResponse],
        task: &TaskRequest,
    ) -> Result<(TaskResponse, VotingDetails)> {
        match &self.ensembleconfig.voting_strategy {
            VotingStrategy::BestQuality => self.best_quality_voting(responses).await,
            VotingStrategy::WeightedAverage { weights } => {
                self.weighted_average_voting(responses, weights).await
            }
            VotingStrategy::Consensus { similarity_threshold } => {
                self.consensus_voting(responses, *similarity_threshold).await
            }
            VotingStrategy::MajorityVote => self.majority_voting(responses).await,
            VotingStrategy::ValidateAndCorrect => {
                self.validate_and_correct_voting(responses, task).await
            }
            VotingStrategy::GenerateAndSelect { candidates } => {
                self.generate_and_select_voting(responses, *candidates).await
            }
            VotingStrategy::SpecializedRouting => {
                self.specialized_routing_voting(responses, task).await
            }
        }
    }

    /// Best quality voting strategy
    async fn best_quality_voting(
        &self,
        responses: &[IndividualResponse],
    ) -> Result<(TaskResponse, VotingDetails)> {
        let best_response = responses
            .iter()
            .max_by(|a, b| a.quality_score.partial_cmp(&b.quality_score).unwrap())
            .ok_or_else(|| anyhow::anyhow!("No responses to vote on"))?;

        let voting_details = VotingDetails {
            strategy_used: VotingStrategy::BestQuality,
            models_participated: responses.len(),
            models_agreed: 1, // Only one "winner"
            disagreement_areas: vec![],
            confidence_distribution: responses
                .iter()
                .map(|r| (r.model_id.clone(), r.quality_score))
                .collect(),
        };

        Ok((best_response.response.clone(), voting_details))
    }

    /// Weighted average voting strategy
    async fn weighted_average_voting(
        &self,
        responses: &[IndividualResponse],
        weights: &HashMap<String, f32>,
    ) -> Result<(TaskResponse, VotingDetails)> {
        // For text responses, weighted average is complex
        // For now, select the response with highest weighted score

        let mut best_response = None;
        let mut best_weighted_score = 0.0;

        for response in responses {
            let weight = weights.get(&response.model_id).unwrap_or(&1.0);
            let weighted_score = response.quality_score * weight;

            if weighted_score > best_weighted_score {
                best_weighted_score = weighted_score;
                best_response = Some(response);
            }
        }

        let selected_response =
            best_response.ok_or_else(|| anyhow::anyhow!("No responses to vote on"))?;

        // Create a basic similarity matrix for weighted average disagreement analysis
        let mut similarity_matrix = vec![vec![0.0; responses.len()]; responses.len()];
        for i in 0..responses.len() {
            for j in 0..responses.len() {
                if i == j {
                    similarity_matrix[i][j] = 1.0;
                } else {
                    similarity_matrix[i][j] = self.calculate_response_similarity(
                        &responses[i].response,
                        &responses[j].response,
                    ).await;
                }
            }
        }

        // Enhanced disagreement detection for weighted average
        let best_idx = responses.iter()
            .position(|r| std::ptr::eq(r, selected_response))
            .unwrap_or(0);

        let disagreement_areas = self.analyze_disagreement_areas(
            responses,
            &similarity_matrix,
            best_idx,
            0.7 // Standard threshold
        ).await;

        let voting_details = VotingDetails {
            strategy_used: VotingStrategy::WeightedAverage { weights: weights.clone() },
            models_participated: responses.len(),
            models_agreed: 1,
            disagreement_areas,
            confidence_distribution: responses
                .iter()
                .map(|r| {
                    (r.model_id.clone(), r.quality_score * weights.get(&r.model_id).unwrap_or(&1.0))
                })
                .collect(),
        };

        Ok((selected_response.response.clone(), voting_details))
    }

    /// Consensus voting strategy
    async fn consensus_voting(
        &self,
        responses: &[IndividualResponse],
        similarity_threshold: f32,
    ) -> Result<(TaskResponse, VotingDetails)> {
        // Calculate pairwise similarities between responses
        let mut similarity_matrix = Vec::new();

        for i in 0..responses.len() {
            let mut row = Vec::new();
            for j in 0..responses.len() {
                let similarity = if i == j {
                    1.0
                } else {
                    self.calculate_response_similarity(
                        &responses[i].response,
                        &responses[j].response,
                    )
                    .await
                };
                row.push(similarity);
            }
            similarity_matrix.push(row);
        }

        // Find the response with highest average similarity to others
        let mut best_consensus_score = 0.0;
        let mut best_response_idx = 0;
        let mut models_agreed = 0;

        for i in 0..responses.len() {
            let avg_similarity: f32 =
                similarity_matrix[i].iter().sum::<f32>() / similarity_matrix[i].len() as f32;
            let consensus_count =
                similarity_matrix[i].iter().filter(|&&sim| sim >= similarity_threshold).count();

            if avg_similarity > best_consensus_score {
                best_consensus_score = avg_similarity;
                best_response_idx = i;
                models_agreed = consensus_count;
            }
        }

        // Enhanced disagreement detection with sophisticated analysis
        let disagreement_areas = self.analyze_disagreement_areas(
            responses,
            &similarity_matrix,
            best_response_idx,
            similarity_threshold
        ).await;

        let voting_details = VotingDetails {
            strategy_used: VotingStrategy::Consensus { similarity_threshold },
            models_participated: responses.len(),
            models_agreed,
            disagreement_areas,
            confidence_distribution: responses
                .iter()
                .enumerate()
                .map(|(i, r)| (r.model_id.clone(), similarity_matrix[best_response_idx][i]))
                .collect(),
        };

        Ok((responses[best_response_idx].response.clone(), voting_details))
    }

    /// Majority voting based on response similarity clustering
    async fn majority_voting(
        &self,
        responses: &[IndividualResponse],
    ) -> Result<(TaskResponse, VotingDetails)> {
        if responses.len() < 3 {
            return self.best_quality_voting(responses).await;
        }

        // Calculate similarity matrix
        let mut similarity_matrix = vec![vec![0.0; responses.len()]; responses.len()];

        for i in 0..responses.len() {
            for j in i + 1..responses.len() {
                let similarity = self
                    .calculate_response_similarity(&responses[i].response, &responses[j].response)
                    .await;
                similarity_matrix[i][j] = similarity;
                similarity_matrix[j][i] = similarity;
            }
            similarity_matrix[i][i] = 1.0;
        }

        // Find clusters of similar responses
        let threshold = 0.6;
        let mut clusters = Vec::new();
        let mut assigned = vec![false; responses.len()];

        for i in 0..responses.len() {
            if assigned[i] {
                continue;
            }

            let mut cluster = vec![i];
            assigned[i] = true;

            for j in i + 1..responses.len() {
                if !assigned[j] && similarity_matrix[i][j] >= threshold {
                    cluster.push(j);
                    assigned[j] = true;
                }
            }

            clusters.push(cluster);
        }

        // Find the largest cluster (majority)
        let majority_cluster = clusters
            .iter()
            .max_by_key(|cluster| cluster.len())
            .ok_or_else(|| anyhow::anyhow!("No majority cluster found"))?;

        // Select best response from majority cluster
        let best_in_cluster = majority_cluster
            .iter()
            .max_by(|&&a, &&b| {
                responses[a].quality_score.partial_cmp(&responses[b].quality_score).unwrap()
            })
            .ok_or_else(|| anyhow::anyhow!("Empty majority cluster"))?;

        // Enhanced disagreement detection for majority voting
        let mut disagreement_areas = vec![format!(
            "Found {} clusters, majority has {} responses",
            clusters.len(),
            majority_cluster.len()
        )];

        // Add sophisticated disagreement analysis
        let additional_disagreements = self.analyze_disagreement_areas(
            responses,
            &similarity_matrix,
            *best_in_cluster,
            threshold
        ).await;
        disagreement_areas.extend(additional_disagreements);

        let voting_details = VotingDetails {
            strategy_used: VotingStrategy::MajorityVote,
            models_participated: responses.len(),
            models_agreed: majority_cluster.len(),
            disagreement_areas,
            confidence_distribution: responses
                .iter()
                .enumerate()
                .map(|(i, r)| {
                    (r.model_id.clone(), if majority_cluster.contains(&i) { 1.0 } else { 0.0 })
                })
                .collect(),
        };

        Ok((responses[*best_in_cluster].response.clone(), voting_details))
    }

    /// Validate and correct voting - uses one model to validate others
    async fn validate_and_correct_voting(
        &self,
        responses: &[IndividualResponse],
        _task: &TaskRequest,
    ) -> Result<(TaskResponse, VotingDetails)> {
        if responses.len() < 2 {
            return self.best_quality_voting(responses).await;
        }

        // Find the highest quality response to use as validator
        let validator_idx = responses
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.quality_score.partial_cmp(&b.quality_score).unwrap())
            .map(|(i, _)| i)
            .ok_or_else(|| anyhow::anyhow!("No responses to validate"))?;

        let validator_response = &responses[validator_idx];

        // Score other responses based on agreement with validator
        let mut validation_scores = Vec::new();

        for (i, response) in responses.iter().enumerate() {
            if i == validator_idx {
                validation_scores.push(1.0); // Validator agrees with itself
                continue;
            }

            let similarity = self
                .calculate_response_similarity(&validator_response.response, &response.response)
                .await;

            // Combine similarity with original quality score
            let validation_score = (similarity * 0.6) + (response.quality_score * 0.4);
            validation_scores.push(validation_score);
        }

        // Find the best validated response
        let best_validated_idx = validation_scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        let voting_details = VotingDetails {
            strategy_used: VotingStrategy::ValidateAndCorrect,
            models_participated: responses.len(),
            models_agreed: validation_scores.iter().filter(|&&score| score > 0.7).count(),
            disagreement_areas: vec![format!(
                "Validator: {}, Best validated: {}",
                validator_response.model_id, responses[best_validated_idx].model_id
            )],
            confidence_distribution: responses
                .iter()
                .enumerate()
                .map(|(i, r)| (r.model_id.clone(), validation_scores[i]))
                .collect(),
        };

        Ok((responses[best_validated_idx].response.clone(), voting_details))
    }

    /// Generate and select voting - analyzes multiple candidates and selects
    /// best
    async fn generate_and_select_voting(
        &self,
        responses: &[IndividualResponse],
        candidates: usize,
    ) -> Result<(TaskResponse, VotingDetails)> {
        if responses.is_empty() {
            return Err(anyhow::anyhow!("No responses to select from"));
        }

        // Take top candidates based on quality score
        let mut sorted_responses = responses.to_vec();
        sorted_responses.sort_by(|a, b| b.quality_score.partial_cmp(&a.quality_score).unwrap());

        let num_candidates = candidates.min(sorted_responses.len());
        let top_candidates = &sorted_responses[0..num_candidates];

        // Evaluate candidates based on multiple criteria
        let mut candidate_scores = Vec::new();

        for candidate in top_candidates {
            // Composite score: quality + diversity + efficiency
            let quality_score = candidate.quality_score;
            let efficiency_score = if candidate.latency_ms > 0 {
                1.0 / (candidate.latency_ms as f32 / 1000.0) // Favor faster responses
            } else {
                0.5
            };

            let cost_score = match candidate.cost_cents {
                Some(cost) if cost > 0.0 => 1.0 / cost, // Favor cheaper responses
                _ => 1.0,                               // Local models get perfect cost score
            };

            // Weighted composite score
            let composite_score = (quality_score * 0.6)
                + (efficiency_score.min(1.0) * 0.2)
                + (cost_score.min(1.0) * 0.2);

            candidate_scores.push(composite_score);
        }

        // Select the best candidate
        let best_candidate_idx = candidate_scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        let selected_candidate = &top_candidates[best_candidate_idx];

        let voting_details = VotingDetails {
            strategy_used: VotingStrategy::GenerateAndSelect { candidates },
            models_participated: responses.len(),
            models_agreed: 1, // Only one selected
            disagreement_areas: vec![format!("Selected from {} candidates", num_candidates)],
            confidence_distribution: top_candidates
                .iter()
                .enumerate()
                .map(|(i, r)| (r.model_id.clone(), candidate_scores[i]))
                .collect(),
        };

        Ok((selected_candidate.response.clone(), voting_details))
    }

    /// Specialized routing voting - routes different aspects to specialized
    /// models
    async fn specialized_routing_voting(
        &self,
        responses: &[IndividualResponse],
        task: &TaskRequest,
    ) -> Result<(TaskResponse, VotingDetails)> {
        if responses.is_empty() {
            return Err(anyhow::anyhow!("No responses for specialized routing"));
        }

        // Define specialization preferences based on task type
        let preferred_specialists = match &task.task_type {
            super::orchestrator::TaskType::CodeGeneration { language } => match language.as_str() {
                "python" | "javascript" | "rust" => vec!["codestral", "openai"],
                _ => vec!["openai", "codestral"],
            },
            super::orchestrator::TaskType::LogicalReasoning => {
                vec!["anthropic", "openai", "mistral"]
            }
            super::orchestrator::TaskType::CreativeWriting => {
                vec!["anthropic", "mistral", "openai"]
            }
            super::orchestrator::TaskType::DataAnalysis => {
                vec!["openai", "anthropic", "gemini"]
            }
            _ => vec!["anthropic", "openai"], // General tasks
        };

        // Score responses based on model specialization
        let mut specialization_scores = Vec::new();

        for response in responses {
            let base_score = response.quality_score;

            // Bonus for specialized models
            let specialization_bonus = preferred_specialists
                .iter()
                .position(|&specialist| response.model_id.to_lowercase().contains(specialist))
                .map(|pos| 0.2 * (1.0 - pos as f32 * 0.05)) // Decreasing bonus by preference order
                .unwrap_or(0.0);

            let final_score = base_score + specialization_bonus;
            specialization_scores.push(final_score);
        }

        // Select the best specialized response
        let best_specialist_idx = specialization_scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        let selected_response = &responses[best_specialist_idx];

        let voting_details = VotingDetails {
            strategy_used: VotingStrategy::SpecializedRouting,
            models_participated: responses.len(),
            models_agreed: 1, // One specialist selected
            disagreement_areas: vec![format!("Specialized for task: {:?}", task.task_type)],
            confidence_distribution: responses
                .iter()
                .enumerate()
                .map(|(i, r)| (r.model_id.clone(), specialization_scores[i]))
                .collect(),
        };

        Ok((selected_response.response.clone(), voting_details))
    }

    // Helper methods

    async fn estimate_capability_score(&self, model_id: &str, task: &TaskRequest) -> f32 {
        // Score based on model capabilities and task requirements
        let base_score: f32 = match model_id {
            id if id.contains("llama") => 0.7,
            id if id.contains("mistral") => 0.75,
            id if id.contains("codellama") => 0.8,
            id if id.contains("phi") => 0.6,
            _ => 0.65,
        };

        // Adjust based on task type
        let task_modifier: f32 =
            match &task.task_type {
                super::orchestrator::TaskType::CodeGeneration { .. } => {
                    if model_id.contains("code") { 0.2 } else { 0.0 }
                }
                super::orchestrator::TaskType::LogicalReasoning => {
                    if model_id.contains("llama") {
                        0.15
                    } else {
                        0.05
                    }
                }
                super::orchestrator::TaskType::CreativeWriting => {
                    if model_id.contains("mistral") {
                        0.1
                    } else {
                        0.0
                    }
                }
                _ => 0.0,
            };

        (base_score + task_modifier).min(1.0f32)
    }

    async fn estimate_api_capability_score(&self, provider_name: &str, task: &TaskRequest) -> f32 {
        // Base scores for different providers
        let base_score: f32 = match provider_name {
            "openai" => 0.95,
            "anthropic" => 0.93,
            "mistral" => 0.85,
            "gemini" => 0.88,
            "codestral" => 0.9,
            _ => 0.7,
        };

        // Task-specific adjustments
        let task_modifier: f32 = match &task.task_type {
            super::orchestrator::TaskType::CodeGeneration { .. } => match provider_name {
                "codestral" => 0.05,
                "openai" => 0.03,
                _ => 0.0,
            },
            super::orchestrator::TaskType::LogicalReasoning => match provider_name {
                "anthropic" => 0.04,
                "openai" => 0.03,
                _ => 0.0,
            },
            super::orchestrator::TaskType::CreativeWriting => match provider_name {
                "anthropic" => 0.05,
                "openai" => 0.02,
                _ => 0.0,
            },
            _ => 0.0,
        };

        (base_score + task_modifier).min(1.0f32)
    }

    async fn estimate_api_cost(&self, provider_name: &str, task: &TaskRequest) -> f32 {
        // Estimate tokens based on content length
        let estimated_tokens = (task.content.len() / 4) as u32 + 500; // Rough estimation + response buffer

        // Cost per 1k tokens (in cents)
        let cost_per_1k = match provider_name {
            "openai" => 15.0,    // GPT-4o-mini
            "anthropic" => 25.0, // Claude 3.5 Haiku
            "mistral" => 25.0,   // Mistral Small
            "gemini" => 7.5,     // Gemini 1.5 Flash
            "codestral" => 25.0, // Codestral
            _ => 10.0,
        };

        (estimated_tokens as f32 / 1000.0) * cost_per_1k
    }

    async fn apply_selection_constraints(
        &self,
        mut models: Vec<EnsembleModel>,
        task: &TaskRequest,
    ) -> Vec<EnsembleModel> {
        // Apply cost constraints
        if let Some(max_cost) = task.constraints.max_cost_cents {
            models.retain(|model| model.cost_estimate <= max_cost);
        }

        // Apply diversity constraints - ensure mix of local and API models
        let mut local_count = 0;
        let mut api_count = 0;

        for model in &models {
            match model.model_type {
                EnsembleModelType::Local => local_count += 1,
                EnsembleModelType::API => api_count += 1,
            }
        }

        // Prefer diverse model types for better ensemble performance
        if local_count > 0 && api_count > 0 {
            // Good diversity, keep as is
            models
        } else if local_count == 0 && api_count > 2 {
            // Too many API models, try to balance cost
            models.sort_by(|a, b| a.cost_estimate.partial_cmp(&b.cost_estimate).unwrap());
            models.truncate(3); // Limit expensive API calls
            models
        } else {
            // Good local/API balance or mostly local
            models
        }
    }

    async fn calculate_response_similarity(
        &self,
        response1: &TaskResponse,
        response2: &TaskResponse,
    ) -> f32 {
        // Simple text-based similarity using Jaccard coefficient
        let words1: std::collections::HashSet<&str> =
            response1.content.split_whitespace().collect();
        let words2: std::collections::HashSet<&str> =
            response2.content.split_whitespace().collect();

        if words1.is_empty() && words2.is_empty() {
            return 1.0;
        }

        let intersection_size = words1.intersection(&words2).count();
        let union_size = words1.union(&words2).count();

        if union_size == 0 { 0.0 } else { intersection_size as f32 / union_size as f32 }
    }

    /// Sophisticated disagreement analysis using Rust 2025 patterns
    /// Identifies specific areas where ensemble models disagree
    async fn analyze_disagreement_areas(
        &self,
        responses: &[IndividualResponse],
        similarity_matrix: &[Vec<f32>],
        _best_response_idx: usize,
        threshold: f32,
    ) -> Vec<String> {
        let mut disagreement_areas = Vec::new();

        // 1. Content-level disagreement analysis
        let content_disagreements = self.analyze_content_disagreements(responses, threshold).await;
        disagreement_areas.extend(content_disagreements);

        // 2. Quality score variance analysis
        let quality_disagreements = self.analyze_quality_disagreements(responses).await;
        disagreement_areas.extend(quality_disagreements);

        // 3. Model clustering analysis
        let clustering_disagreements = self.analyze_clustering_disagreements(
            similarity_matrix,
            responses,
            threshold
        ).await;
        disagreement_areas.extend(clustering_disagreements);

        // 4. Semantic disagreement detection
        let semantic_disagreements = self.analyze_semantic_disagreements(responses).await;
        disagreement_areas.extend(semantic_disagreements);

        // 5. Confidence distribution analysis
        let confidence_disagreements = self.analyze_confidence_disagreements(responses).await;
        disagreement_areas.extend(confidence_disagreements);

        // Remove duplicates and return sorted results
        disagreement_areas.sort();
        disagreement_areas.dedup();
        disagreement_areas
    }

    /// Analyze content-level disagreements using advanced text analysis
    async fn analyze_content_disagreements(
        &self,
        responses: &[IndividualResponse],
        threshold: f32,
    ) -> Vec<String> {
        let mut disagreements = Vec::new();

        // Content length variance analysis
        let lengths: Vec<usize> = responses.iter().map(|r| r.response.content.len()).collect();
        let avg_length = lengths.iter().sum::<usize>() as f32 / lengths.len() as f32;
        let length_variance = lengths.iter()
            .map(|&len| (len as f32 - avg_length).powi(2))
            .sum::<f32>() / lengths.len() as f32;

        if length_variance > (avg_length * threshold).powi(2) {
            let max_len = *lengths.iter().max().unwrap_or(&0);
            let min_len = *lengths.iter().min().unwrap_or(&0);
            disagreements.push(format!(
                "Content length variance: {:.0}% (range: {}-{} chars)",
                (length_variance.sqrt() / avg_length) * 100.0,
                min_len,
                max_len
            ));
        }

        // Keyword frequency disagreement
        let mut keyword_disagreements = self.analyze_keyword_disagreements(responses, threshold).await;
        disagreements.append(&mut keyword_disagreements);

        // Structural disagreement (bullet points, numbered lists, etc.)
        let mut structural_disagreements = self.analyze_structural_disagreements(responses).await;
        disagreements.append(&mut structural_disagreements);

        disagreements
    }

    /// Analyze keyword frequency disagreements between responses
    async fn analyze_keyword_disagreements(&self, responses: &[IndividualResponse], threshold: f32) -> Vec<String> {
        let mut disagreements = Vec::new();

        // Extract important keywords from each response
        let response_keywords: Vec<std::collections::HashMap<String, usize>> = responses.iter()
            .map(|r| self.extract_keywords(&r.response.content))
            .collect();

        // Find keywords that appear in some responses but not others
        let mut all_keywords = std::collections::HashSet::new();
        for keywords in &response_keywords {
            all_keywords.extend(keywords.keys().cloned());
        }

        let mut controversial_keywords = Vec::new();
        for keyword in all_keywords {
            let appearances = response_keywords.iter()
                .filter(|keywords| keywords.contains_key(&keyword))
                .count();

            // If keyword appears in less than threshold% of responses, it's controversial
            let threshold_count = (responses.len() as f32 * (1.0 - threshold)) as usize;
            if appearances < threshold_count && appearances > 0 {
                controversial_keywords.push((keyword, appearances));
            }
        }

        if !controversial_keywords.is_empty() {
            disagreements.push(format!(
                "Keyword disagreements: {} terms not consistently used ({})",
                controversial_keywords.len(),
                controversial_keywords.iter()
                    .take(3)
                    .map(|(k, count)| format!("'{}' in {}/{}", k, count, responses.len()))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }

        disagreements
    }

    /// Extract keywords from content using simple frequency analysis
    fn extract_keywords(&self, content: &str) -> std::collections::HashMap<String, usize> {
        let mut keywords = std::collections::HashMap::new();

        // Common stop words to ignore
        let stop_words = ["the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "can", "a", "an", "this", "that", "these", "those"];

        for word in content.split_whitespace() {
            let cleaned_word = word.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase();
            if cleaned_word.len() > 3 && !stop_words.contains(&cleaned_word.as_str()) {
                *keywords.entry(cleaned_word).or_insert(0) += 1;
            }
        }

        keywords
    }

    /// Analyze structural disagreements (formatting, organization)
    async fn analyze_structural_disagreements(&self, responses: &[IndividualResponse]) -> Vec<String> {
        let mut disagreements = Vec::new();

        // Check for bullet point usage disagreement
        let bullet_usage: Vec<bool> = responses.iter()
            .map(|r| r.response.content.contains("•") || r.response.content.contains("-") || r.response.content.contains("*"))
            .collect();

        let bullet_count = bullet_usage.iter().filter(|&&uses| uses).count();
        if bullet_count > 0 && bullet_count < responses.len() {
            disagreements.push(format!(
                "Structural disagreement: {}/{} models use bullet points/lists",
                bullet_count, responses.len()
            ));
        }

        // Check for numbered list disagreement
        let numbered_usage: Vec<bool> = responses.iter()
            .map(|r| {
                let content = &r.response.content;
                content.contains("1.") || content.contains("1)") ||
                content.contains("2.") || content.contains("2)")
            })
            .collect();

        let numbered_count = numbered_usage.iter().filter(|&&uses| uses).count();
        if numbered_count > 0 && numbered_count < responses.len() {
            disagreements.push(format!(
                "Structural disagreement: {}/{} models use numbered lists",
                numbered_count, responses.len()
            ));
        }

        disagreements
    }

    /// Analyze quality score disagreements
    async fn analyze_quality_disagreements(&self, responses: &[IndividualResponse]) -> Vec<String> {
        let mut disagreements = Vec::new();

        let quality_scores: Vec<f32> = responses.iter().map(|r| r.quality_score).collect();
        if quality_scores.len() < 2 {
            return disagreements;
        }

        let avg_quality = quality_scores.iter().sum::<f32>() / quality_scores.len() as f32;
        let quality_variance = quality_scores.iter()
            .map(|&score| (score - avg_quality).powi(2))
            .sum::<f32>() / quality_scores.len() as f32;

        if quality_variance > 0.05 { // Significant quality disagreement
            let max_quality = quality_scores.iter().fold(0.0f32, |acc, &x| acc.max(x));
            let min_quality = quality_scores.iter().fold(1.0f32, |acc, &x| acc.min(x));

            disagreements.push(format!(
                "Quality score variance: {:.1}% (range: {:.2}-{:.2})",
                (quality_variance.sqrt() / avg_quality) * 100.0,
                min_quality,
                max_quality
            ));
        }

        disagreements
    }

    /// Analyze clustering-based disagreements
    async fn analyze_clustering_disagreements(
        &self,
        similarity_matrix: &[Vec<f32>],
        responses: &[IndividualResponse],
        threshold: f32,
    ) -> Vec<String> {
        let mut disagreements = Vec::new();

        // Form clusters based on similarity threshold
        let mut clusters = Vec::new();
        let mut assigned = vec![false; responses.len()];

        for i in 0..responses.len() {
            if assigned[i] {
                continue;
            }

            let mut cluster = vec![i];
            assigned[i] = true;

            for j in (i + 1)..responses.len() {
                if !assigned[j] && similarity_matrix[i][j] >= threshold {
                    cluster.push(j);
                    assigned[j] = true;
                }
            }

            if cluster.len() > 1 {
                clusters.push(cluster);
            }
        }

        // Analyze cluster formation
        let total_clustered = clusters.iter().map(|c| c.len()).sum::<usize>();
        let unclustered = responses.len() - total_clustered;

        if clusters.len() > 1 {
            disagreements.push(format!(
                "Response clustering: {} distinct groups, {} unclustered models",
                clusters.len(),
                unclustered
            ));
        }

        if unclustered > 0 {
            disagreements.push(format!(
                "Outlier responses: {} models produced significantly different outputs",
                unclustered
            ));
        }

        disagreements
    }

    /// Analyze semantic disagreements using advanced text analysis
    async fn analyze_semantic_disagreements(&self, responses: &[IndividualResponse]) -> Vec<String> {
        let mut disagreements = Vec::new();

        // Sentiment analysis disagreement
        let sentiments: Vec<f32> = responses.iter()
            .map(|r| self.analyze_sentiment(&r.response.content))
            .collect();

        if sentiments.len() > 1 {
            let max_sentiment = sentiments.iter().fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
            let min_sentiment = sentiments.iter().fold(f32::INFINITY, |acc, &x| acc.min(x));

            if (max_sentiment - min_sentiment) > 0.5 {
                disagreements.push(format!(
                    "Sentiment disagreement: response tone varies from {:.2} to {:.2}",
                    min_sentiment,
                    max_sentiment
                ));
            }
        }

        // Technical depth disagreement
        let technical_depths: Vec<f32> = responses.iter()
            .map(|r| self.analyze_technical_depth(&r.response.content))
            .collect();

        if technical_depths.len() > 1 {
            let max_depth = technical_depths.iter().fold(0.0f32, |acc, &x| acc.max(x));
            let min_depth = technical_depths.iter().fold(1.0f32, |acc, &x| acc.min(x));

            if (max_depth - min_depth) > 0.3 {
                disagreements.push(format!(
                    "Technical depth disagreement: complexity varies from {:.2} to {:.2}",
                    min_depth,
                    max_depth
                ));
            }
        }

        disagreements
    }

    /// Simple sentiment analysis based on keyword presence
    fn analyze_sentiment(&self, content: &str) -> f32 {
        let positive_words = ["good", "great", "excellent", "effective", "beneficial", "optimal", "improved", "better", "successful", "positive"];
        let negative_words = ["bad", "poor", "ineffective", "problematic", "difficult", "challenging", "worse", "failed", "negative", "error"];

        let content_lower = content.to_lowercase();
        let positive_count = positive_words.iter().filter(|&&word| content_lower.contains(word)).count();
        let negative_count = negative_words.iter().filter(|&&word| content_lower.contains(word)).count();

        let total_sentiment_words = positive_count + negative_count;
        if total_sentiment_words == 0 {
            return 0.0; // Neutral
        }

        (positive_count as f32 - negative_count as f32) / total_sentiment_words as f32
    }

    /// Analyze technical depth based on technical term frequency
    fn analyze_technical_depth(&self, content: &str) -> f32 {
        let technical_indicators = [
            "algorithm", "implementation", "configuration", "optimization", "architecture",
            "framework", "protocol", "interface", "module", "function", "method", "class",
            "database", "server", "client", "API", "REST", "HTTP", "JSON", "XML",
            "memory", "CPU", "performance", "latency", "throughput", "scalability"
        ];

        let content_lower = content.to_lowercase();
        let technical_count = technical_indicators.iter()
            .filter(|&&term| content_lower.contains(term))
            .count();

        let word_count = content.split_whitespace().count().max(1);
        technical_count as f32 / word_count as f32
    }

    /// Analyze confidence distribution disagreements
    async fn analyze_confidence_disagreements(&self, responses: &[IndividualResponse]) -> Vec<String> {
        let mut disagreements = Vec::new();

        // Analyze hedging language (uncertainty indicators)
        let confidence_levels: Vec<f32> = responses.iter()
            .map(|r| self.analyze_confidence_level(&r.response.content))
            .collect();

        if confidence_levels.len() > 1 {
            let avg_confidence = confidence_levels.iter().sum::<f32>() / confidence_levels.len() as f32;
            let confidence_variance = confidence_levels.iter()
                .map(|&conf| (conf - avg_confidence).powi(2))
                .sum::<f32>() / confidence_levels.len() as f32;

            if confidence_variance > 0.05 {
                let max_conf = confidence_levels.iter().fold(0.0f32, |acc, &x| acc.max(x));
                let min_conf = confidence_levels.iter().fold(1.0f32, |acc, &x| acc.min(x));

                disagreements.push(format!(
                    "Confidence disagreement: certainty varies from {:.2} to {:.2}",
                    min_conf,
                    max_conf
                ));
            }
        }

        disagreements
    }

    /// Analyze confidence level based on hedging language
    fn analyze_confidence_level(&self, content: &str) -> f32 {
        let uncertainty_words = ["might", "could", "possibly", "perhaps", "maybe", "likely", "probably", "seems", "appears", "suggests"];
        let certainty_words = ["will", "must", "definitely", "certainly", "clearly", "obviously", "always", "never", "exactly", "precisely"];

        let content_lower = content.to_lowercase();
        let uncertainty_count = uncertainty_words.iter().filter(|&&word| content_lower.contains(word)).count();
        let certainty_count = certainty_words.iter().filter(|&&word| content_lower.contains(word)).count();

        let total_confidence_words = uncertainty_count + certainty_count;
        if total_confidence_words == 0 {
            return 0.7; // Default moderate confidence
        }

        // Scale from 0.0 (very uncertain) to 1.0 (very certain)
        certainty_count as f32 / total_confidence_words as f32
    }

    fn calculate_consensus_score(&self, responses: &[IndividualResponse]) -> f32 {
        if responses.len() <= 1 {
            return 1.0;
        }

        // Simple consensus based on quality score variance
        let scores: Vec<f32> = responses.iter().map(|r| r.quality_score).collect();
        let mean = scores.iter().sum::<f32>() / scores.len() as f32;
        let variance = scores.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / scores.len() as f32;

        // Lower variance = higher consensus
        1.0 - variance.min(1.0)
    }

    fn calculate_diversity_score(&self, responses: &[IndividualResponse]) -> f32 {
        // Diversity based on model types and response lengths
        let unique_models = responses
            .iter()
            .map(|r| r.model_id.as_str())
            .collect::<std::collections::HashSet<_>>()
            .len();

        unique_models as f32 / responses.len() as f32
    }

    fn calculate_ensemble_metrics(
        &self,
        responses: &[IndividualResponse],
        execution_time: Duration,
    ) -> EnsembleMetrics {
        let total_latency: u32 = responses.iter().map(|r| r.latency_ms).sum();
        let max_latency = responses.iter().map(|r| r.latency_ms).max().unwrap_or(0);

        let parallel_efficiency = if max_latency > 0 && !responses.is_empty() {
            // How well we utilized parallelism (1.0 = perfect parallel execution)
            max_latency as f32 / (total_latency as f32 / responses.len() as f32)
        } else {
            1.0
        };

        // Calculate cost effectiveness
        let total_cost: f32 = responses.iter().filter_map(|r| r.cost_cents).sum();

        let avg_quality: f32 =
            responses.iter().map(|r| r.quality_score).sum::<f32>() / responses.len().max(1) as f32;

        let cost_effectiveness = if total_cost > 0.0 {
            avg_quality / total_cost // Quality per cent
        } else {
            avg_quality // Free local models get perfect cost effectiveness
        };

        // Calculate quality improvement (compare to single best model)
        let best_individual_quality =
            responses.iter().map(|r| r.quality_score).fold(0.0f32, |acc, score| acc.max(score));

        let quality_improvement = if best_individual_quality > 0.0 {
            (avg_quality - best_individual_quality) / best_individual_quality
        } else {
            0.0
        };

        EnsembleMetrics {
            total_execution_time: execution_time,
            parallel_efficiency: parallel_efficiency.min(1.0),
            resource_utilization: responses.len() as f32 / self.ensembleconfig.max_models as f32,
            cost_effectiveness,
            quality_improvement,
        }
    }
}

/// Model selected for ensemble execution
#[derive(Debug, Clone)]
struct EnsembleModel {
    model_id: String,
    model_type: EnsembleModelType,
    capability_score: f32,
    cost_estimate: f32,
    latency_estimate: f32,
}

#[derive(Debug, Clone)]
enum EnsembleModelType {
    Local,
    API,
}

/// Response quality assessment system
#[derive(Debug)]
pub struct ResponseQualityAssessor;

impl ResponseQualityAssessor {
    pub fn new() -> Self {
        Self
    }

    pub async fn assess_response_quality(
        &self,
        response: &TaskResponse,
        task: &TaskRequest,
    ) -> f32 {
        let mut quality_score = 0.0;
        let content = &response.content;

        // Content length assessment (reasonable length indicates thoughtfulness)
        let length_score = if content.len() < 10 {
            0.2 // Too short
        } else if content.len() < 50 {
            0.5 // Short but acceptable
        } else if content.len() < 500 {
            0.8 // Good length
        } else if content.len() < 2000 {
            1.0 // Comprehensive
        } else {
            0.7 // Might be too verbose
        };
        quality_score += length_score * 0.2;

        // Task-specific quality assessment
        let task_score = match &task.task_type {
            super::orchestrator::TaskType::CodeGeneration { .. } => {
                self.assess_code_quality(content)
            }
            super::orchestrator::TaskType::LogicalReasoning => {
                self.assess_reasoning_quality(content)
            }
            super::orchestrator::TaskType::CreativeWriting => self.assess_creative_quality(content),
            _ => self.assess_general_quality(content),
        };
        quality_score += task_score * 0.5;

        // Coherence assessment (basic grammar and structure)
        let coherence_score = self.assess_coherence(content);
        quality_score += coherence_score * 0.2;

        // Completeness assessment (does it address the task?)
        let completeness_score = self.assess_completeness(content, &task.content);
        quality_score += completeness_score * 0.1;

        quality_score.min(1.0).max(0.0)
    }

    fn assess_code_quality(&self, content: &str) -> f32 {
        let mut score: f32 = 0.5; // Base score

        // Look for code indicators
        if content.contains("```") {
            score += 0.2;
        }
        if content.contains("fn ") || content.contains("def ") || content.contains("function") {
            score += 0.1;
        }
        if content.contains("//") || content.contains("#") {
            score += 0.1;
        } // Comments
        if content.contains("{") && content.contains("}") {
            score += 0.1;
        } // Structure

        score.min(1.0)
    }

    fn assess_reasoning_quality(&self, content: &str) -> f32 {
        let mut score: f32 = 0.5;

        // Look for reasoning indicators
        if content.contains("because") || content.contains("therefore") || content.contains("thus")
        {
            score += 0.15;
        }
        if content.contains("however") || content.contains("although") || content.contains("while")
        {
            score += 0.1;
        }
        if content.contains("first") || content.contains("second") || content.contains("finally") {
            score += 0.1;
        }
        if content.contains("?") {
            score += 0.05;
        } // Questions show thinking

        score.min(1.0)
    }

    fn assess_creative_quality(&self, content: &str) -> f32 {
        let mut score: f32 = 0.5;

        // Look for creative indicators
        let word_count = content.split_whitespace().count();
        if word_count > 50 {
            score += 0.1;
        }

        // Variety in sentence structure
        let sentences = content.split(['.', '!', '?']).count();
        if sentences > 3 {
            score += 0.1;
        }

        // Descriptive language
        if content.contains("like") || content.contains("as if") {
            score += 0.1;
        }

        score.min(1.0)
    }

    fn assess_general_quality(&self, content: &str) -> f32 {
        let mut score: f32 = 0.6; // Slightly higher base for general tasks

        let word_count = content.split_whitespace().count();
        if word_count > 20 {
            score += 0.1;
        }
        if word_count > 100 {
            score += 0.1;
        }

        // Check for structured response
        if content.contains(":") {
            score += 0.05;
        }
        if content.contains("\n") {
            score += 0.05;
        }

        score.min(1.0)
    }

    fn assess_coherence(&self, content: &str) -> f32 {
        let mut score: f32 = 0.8; // Start high, deduct for issues

        // Check for obvious issues
        if content.chars().filter(|c| c.is_uppercase()).count() as f32 / content.len() as f32 > 0.5
        {
            score -= 0.3; // Too much uppercase
        }

        if content.contains("ERROR") || content.contains("Failed") {
            score -= 0.5; // Contains error messages
        }

        score.max(0.0)
    }

    fn assess_completeness(&self, response: &str, task: &str) -> f32 {
        // Simple keyword overlap check
        let task_lower = task.to_lowercase();
        let task_words: std::collections::HashSet<&str> = task_lower
            .split_whitespace()
            .filter(|w| w.len() > 3) // Filter out short words
            .collect();

        let response_lower = response.to_lowercase();
        let response_words: std::collections::HashSet<&str> =
            response_lower.split_whitespace().collect();

        if task_words.is_empty() {
            return 0.8; // Can't assess, assume reasonable
        }

        let overlap = task_words.intersection(&response_words).count();
        (overlap as f32 / task_words.len() as f32).min(1.0)
    }

    pub async fn assess_confidence(&self, response: &TaskResponse) -> f32 {
        let content = &response.content;
        let mut confidence = 0.7; // Base confidence

        // Length indicates thoroughness
        if content.len() > 100 {
            confidence += 0.1;
        }
        if content.len() > 500 {
            confidence += 0.1;
        }

        // Look for uncertainty markers (reduce confidence)
        let uncertainty_phrases =
            ["might", "maybe", "possibly", "perhaps", "I think", "not sure", "unclear"];
        let uncertainty_count =
            uncertainty_phrases.iter().map(|phrase| content.matches(phrase).count()).sum::<usize>();

        confidence -= (uncertainty_count as f32 * 0.05).min(0.3);

        // Look for confidence markers (increase confidence)
        let confidence_phrases = ["definitely", "certainly", "clearly", "obviously", "indeed"];
        let confidence_count =
            confidence_phrases.iter().map(|phrase| content.matches(phrase).count()).sum::<usize>();

        confidence += (confidence_count as f32 * 0.05).min(0.2);

        // Structured responses tend to be more confident
        if content.contains("1.") || content.contains("-") || content.contains("*") {
            confidence += 0.05;
        }

        // Model-specific confidence adjustment
        match &response.model_used {
            ModelSelection::API(provider) => {
                match provider.as_str() {
                    "anthropic" => confidence += 0.05, // Claude tends to be well-calibrated
                    "openai" => confidence += 0.03,
                    _ => {}
                }
            }
            ModelSelection::Local(_) => {
                confidence -= 0.05; // Local models might be less reliable
            }
        }

        confidence.min(1.0).max(0.0)
    }

    pub async fn assess_ensemble_quality(
        &self,
        responses: &[IndividualResponse],
        primary: &TaskResponse,
    ) -> f32 {
        if responses.is_empty() {
            return 0.0;
        }

        // Find the best individual response quality
        let best_individual_quality =
            responses.iter().map(|r| r.quality_score).fold(0.0f32, |acc, score| acc.max(score));

        // Calculate consensus strength
        let consensus_bonus = if responses.len() > 1 {
            let avg_quality =
                responses.iter().map(|r| r.quality_score).sum::<f32>() / responses.len() as f32;
            let quality_variance =
                responses.iter().map(|r| (r.quality_score - avg_quality).powi(2)).sum::<f32>()
                    / responses.len() as f32;

            // Lower variance = higher consensus = bonus
            (1.0 - quality_variance.min(1.0)) * 0.1
        } else {
            0.0
        };

        // Diversity bonus (different model types)
        let diversity_bonus = {
            let unique_models = responses
                .iter()
                .map(|r| &r.model_id)
                .collect::<std::collections::HashSet<_>>()
                .len();

            if unique_models > 1 {
                (unique_models as f32 / responses.len() as f32) * 0.05
            } else {
                0.0
            }
        };

        // Calculate primary response quality independently
        let primary_length_factor = if primary.content.len() > 200 {
            0.05 // Bonus for comprehensive response
        } else {
            0.0
        };

        // Ensemble quality = best individual + bonuses
        let ensemble_quality =
            best_individual_quality + consensus_bonus + diversity_bonus + primary_length_factor;

        ensemble_quality.min(1.0).max(0.0)
    }

    /// Get comprehensive disagreement analytics for ensemble responses
    pub async fn get_disagreement_analytics(
        &self,
        responses: &[IndividualResponse],
    ) -> Result<DisagreementAnalytics> {
        if responses.len() < 2 {
            return Ok(DisagreementAnalytics::default());
        }

        // Calculate similarity matrix
        let mut similarity_matrix = vec![vec![0.0; responses.len()]; responses.len()];
        for i in 0..responses.len() {
            for j in 0..responses.len() {
                if i == j {
                    similarity_matrix[i][j] = 1.0;
                } else {
                    similarity_matrix[i][j] = self.calculate_response_similarity(
                        &responses[i].response,
                        &responses[j].response,
                    ).await;
                }
            }
        }

        // Find the best response for analysis
        let best_response_idx = responses.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.quality_score.partial_cmp(&b.quality_score).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        // Get comprehensive disagreement analysis
        let disagreement_areas = self.analyze_disagreement_areas(
            responses,
            &similarity_matrix,
            best_response_idx,
            0.7,
        ).await;

        // Calculate overall disagreement metrics
        let avg_similarity = similarity_matrix.iter()
            .flatten()
            .filter(|&&sim| sim < 1.0) // Exclude self-similarity
            .sum::<f32>() / ((responses.len() * responses.len() - responses.len()) as f32);

        let quality_variance = {
            let avg_quality = responses.iter().map(|r| r.quality_score).sum::<f32>() / responses.len() as f32;
            responses.iter()
                .map(|r| (r.quality_score - avg_quality).powi(2))
                .sum::<f32>() / responses.len() as f32
        };

        // Calculate consensus strength
        let consensus_threshold = 0.7;
        let consensus_pairs = similarity_matrix.iter()
            .enumerate()
            .flat_map(|(i, row)| {
                row.iter()
                    .enumerate()
                    .filter(move |(j, _)| *j > i)
                    .map(move |(j, &sim)| (i, j, sim))
            })
            .filter(|(_, _, sim)| *sim >= consensus_threshold)
            .count();

        let total_pairs = responses.len() * (responses.len() - 1) / 2;
        let consensus_strength = if total_pairs > 0 {
            consensus_pairs as f32 / total_pairs as f32
        } else {
            1.0
        };

        Ok(DisagreementAnalytics {
            total_models: responses.len(),
            avg_similarity,
            quality_variance,
            consensus_strength,
            disagreement_count: disagreement_areas.len(),
            disagreement_areas,
            most_similar_pair: self.find_most_similar_pair(&similarity_matrix, responses),
            most_different_pair: self.find_most_different_pair(&similarity_matrix, responses),
        })
    }

    /// Find the most similar pair of responses
    fn find_most_similar_pair(
        &self,
        similarity_matrix: &[Vec<f32>],
        responses: &[IndividualResponse],
    ) -> Option<(String, String, f32)> {
        let mut max_similarity = 0.0;
        let mut best_pair = None;

        for i in 0..responses.len() {
            for j in (i + 1)..responses.len() {
                if similarity_matrix[i][j] > max_similarity {
                    max_similarity = similarity_matrix[i][j];
                    best_pair = Some((
                        responses[i].model_id.clone(),
                        responses[j].model_id.clone(),
                        max_similarity,
                    ));
                }
            }
        }

        best_pair
    }

    /// Find the most different pair of responses
    fn find_most_different_pair(
        &self,
        similarity_matrix: &[Vec<f32>],
        responses: &[IndividualResponse],
    ) -> Option<(String, String, f32)> {
        let mut min_similarity = 1.0;
        let mut worst_pair = None;

        for i in 0..responses.len() {
            for j in (i + 1)..responses.len() {
                if similarity_matrix[i][j] < min_similarity {
                    min_similarity = similarity_matrix[i][j];
                    worst_pair = Some((
                        responses[i].model_id.clone(),
                        responses[j].model_id.clone(),
                        min_similarity,
                    ));
                }
            }
        }

        worst_pair
    }

    /// Calculate similarity between two task responses
    pub async fn calculate_response_similarity(
        &self,
        response1: &TaskResponse,
        response2: &TaskResponse,
    ) -> f32 {
        let content1 = &response1.content;
        let content2 = &response2.content;

        // Simple text similarity based on word overlap
        let words1: std::collections::HashSet<&str> = content1.split_whitespace().collect();
        let words2: std::collections::HashSet<&str> = content2.split_whitespace().collect();

        let intersection = words1.intersection(&words2).count();
        let union = words1.union(&words2).count();

        if union == 0 {
            return 0.0;
        }

        // Jaccard similarity
        let jaccard = intersection as f32 / union as f32;

        // Consider length similarity as well
        let len1 = content1.len() as f32;
        let len2 = content2.len() as f32;
        let length_similarity = 1.0 - ((len1 - len2).abs() / (len1.max(len2) + 1.0));

        // Combine metrics
        (jaccard * 0.7) + (length_similarity * 0.3)
    }

    /// Analyze specific areas of disagreement between responses
    pub async fn analyze_disagreement_areas(
        &self,
        responses: &[IndividualResponse],
        similarity_matrix: &[Vec<f32>],
        _best_response_idx: usize,
        threshold: f32,
    ) -> Vec<String> {
        let mut disagreement_areas = Vec::new();

        // Find pairs with low similarity and analyze differences
        for i in 0..responses.len() {
            for j in (i + 1)..responses.len() {
                if similarity_matrix[i][j] < threshold {
                    // Analyze specific areas of disagreement
                    let content1 = &responses[i].response.content;
                    let content2 = &responses[j].response.content;

                    // Check for different answer types
                    if content1.len() > content2.len() * 2 || content2.len() > content1.len() * 2 {
                        disagreement_areas.push("Response length varies significantly".to_string());
                    }

                    // Check for contradictory keywords
                    let has_yes_1 = content1.to_lowercase().contains("yes") || content1.to_lowercase().contains("true");
                    let has_no_1 = content1.to_lowercase().contains("no") || content1.to_lowercase().contains("false");
                    let has_yes_2 = content2.to_lowercase().contains("yes") || content2.to_lowercase().contains("true");
                    let has_no_2 = content2.to_lowercase().contains("no") || content2.to_lowercase().contains("false");

                    if (has_yes_1 && has_no_2) || (has_no_1 && has_yes_2) {
                        disagreement_areas.push("Contradictory yes/no positions".to_string());
                    }

                    // Check for different technical approaches
                    let tech_words1: std::collections::HashSet<&str> = content1
                        .split_whitespace()
                        .filter(|w| w.len() > 6 && (w.contains("_") || w.contains("::") || w.contains("()")))
                        .collect();
                    let tech_words2: std::collections::HashSet<&str> = content2
                        .split_whitespace()
                        .filter(|w| w.len() > 6 && (w.contains("_") || w.contains("::") || w.contains("()")))
                        .collect();

                    if tech_words1.intersection(&tech_words2).count() == 0 && !tech_words1.is_empty() && !tech_words2.is_empty() {
                        disagreement_areas.push("Different technical approaches suggested".to_string());
                    }
                }
            }
        }

        // Remove duplicates
        disagreement_areas.sort();
        disagreement_areas.dedup();

        if disagreement_areas.is_empty() {
            vec!["No significant disagreements detected".to_string()]
        } else {
            disagreement_areas
        }
    }
}

/// Parallel execution coordinator
#[derive(Debug)]
pub struct ParallelExecutor;

impl ParallelExecutor {
    pub fn new() -> Self {
        Self
    }

}

/// Comprehensive disagreement analytics for ensemble responses
#[derive(Debug, Clone)]
pub struct DisagreementAnalytics {
    pub total_models: usize,
    pub avg_similarity: f32,
    pub quality_variance: f32,
    pub consensus_strength: f32,
    pub disagreement_count: usize,
    pub disagreement_areas: Vec<String>,
    pub most_similar_pair: Option<(String, String, f32)>,
    pub most_different_pair: Option<(String, String, f32)>,
}

impl Default for DisagreementAnalytics {
    fn default() -> Self {
        Self {
            total_models: 0,
            avg_similarity: 1.0,
            quality_variance: 0.0,
            consensus_strength: 1.0,
            disagreement_count: 0,
            disagreement_areas: Vec::new(),
            most_similar_pair: None,
            most_different_pair: None,
        }
    }
}

impl Default for EnsembleConfig {
    fn default() -> Self {
        Self {
            voting_strategy: VotingStrategy::BestQuality,
            min_models: 2,
            max_models: 4,
            quality_threshold: 0.7,
            max_parallel_requests: 8,
            timeout_seconds: 30,
            consensus_threshold: 0.8,
            diversity_bonus: 0.1,
            cost_weight: 0.3,
            latency_weight: 0.2,
            quality_weight: 0.5,
        }
    }
}
