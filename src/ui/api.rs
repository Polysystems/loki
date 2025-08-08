// Simplified API Layer - Clean interface for the UI
//
// This provides a simplified, user-friendly API on top of our complex
// model orchestration system.

use std::sync::Arc;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio_stream::Stream;
use tracing::{debug, info, warn};

use super::{
    ActiveSession,
    OptimizationSuggestion,
    Recommendation,
    SessionId,
    SessionManager,
    SetupId,
    SetupTemplate,
    SetupTemplateManager,
    UserId,
    UserProfileManager,
};
use crate::models::{IntegratedModelSystem, TaskRequest, TaskResponse};

/// Simplified API for the UI layer
pub struct SimplifiedAPI {
    orchestrator: Arc<IntegratedModelSystem>,
    profile_manager: Arc<UserProfileManager>,
    template_manager: Arc<SetupTemplateManager>,
    session_manager: Arc<SessionManager>,
}

impl SimplifiedAPI {
    pub async fn new(
        orchestrator: Arc<IntegratedModelSystem>,
        profile_manager: Arc<UserProfileManager>,
        template_manager: Arc<SetupTemplateManager>,
        session_manager: Arc<SessionManager>,
    ) -> Result<Self> {
        Ok(Self { orchestrator, profile_manager, template_manager, session_manager })
    }

    /// Launch a pre-configured setup for a user
    pub async fn launch_setup(&self, setup_id: &SetupId, user_id: &UserId) -> Result<SessionId> {
        // Get the setup template
        let template = self
            .template_manager
            .get_template(setup_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Setup template not found: {}", setup_id))?;

        // Get user preferences
        let user_profile = self.profile_manager.get_or_create_profile(user_id).await?;

        // Create a new session
        let session_id =
            self.session_manager.create_session(template, user_profile.preferences.clone()).await?;

        // Initialize the models in the background
        tokio::spawn({
            let session_manager = self.session_manager.clone();
            let orchestrator = self.orchestrator.clone();
            let session_id = session_id.clone();

            async move {
                if let Err(e) = session_manager.initialize_session(&session_id, &orchestrator).await
                {
                    warn!("Failed to initialize session {}: {}", session_id, e);
                }
            }
        });

        // Update user's recent setups
        self.profile_manager.add_recent_setup(user_id, setup_id).await?;

        info!("Session {} created for setup {}", session_id, setup_id);
        Ok(session_id)
    }

    /// Send a message to an active session and get response
    pub async fn send_message(
        &self,
        session_id: &SessionId,
        message: &str,
    ) -> Result<TaskResponse> {
        debug!("Sending message to session {}: {}", session_id, message);

        // Get the active session
        let session = self
            .session_manager
            .get_session(session_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Session not found: {}", session_id))?;

        // Create task request
        let task_request = TaskRequest {
            task_type: self.infer_task_type(message),
            content: message.to_string(),
            constraints: session.get_request_constraints(),
            context_integration: true,
            memory_integration: true,
            cognitive_enhancement: true,
        };

        // Execute through orchestrator
        let response =
            self.orchestrator.get_orchestrator().execute_with_fallback(task_request).await?;

        // Update session stats
        self.session_manager.record_interaction(session_id, &response).await?;

        debug!("Response generated for session {}", session_id);
        Ok(response)
    }

    /// Stream responses for real-time UI updates
    pub async fn stream_response(
        &self,
        session_id: &SessionId,
        message: &str,
    ) -> Result<Box<dyn Stream<Item = Result<ResponseChunk>> + Send + Unpin>> {
        use tokio_stream::wrappers::ReceiverStream;
        use tokio::sync::mpsc;

        debug!("Starting streaming response for session {}: {}", session_id, message);

        // Get the active session
        let session = self
            .session_manager
            .get_session(session_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Session not found: {}", session_id))?;

        // Create task request for streaming
        let task_request = crate::models::TaskRequest {
            task_type: self.infer_task_type(message),
            content: message.to_string(),
            constraints: session.get_request_constraints(),
            context_integration: true,
            memory_integration: true,
            cognitive_enhancement: true,
        };

        // Try to execute with streaming first
        let orchestrator = self.orchestrator.get_orchestrator();
        match orchestrator.execute_streaming(task_request.clone()).await {
            Ok(streaming_response) => {
                // Stream available - process streaming response
                let (tx, rx) = mpsc::channel(32);
                let session_id_clone = session_id.clone();
                let session_manager = self.session_manager.clone();

                tokio::spawn(async move {
                    let mut content_buffer = String::new();
                    let mut chunk_count = 0u64;
                    let start_time = std::time::Instant::now();

                    // Process streaming events
                    let mut event_receiver = streaming_response.event_receiver;
                    while let Some(event) = event_receiver.recv().await {
                        match event {
                            crate::models::streaming::StreamEvent::TextChunk { content, .. } => {
                                content_buffer.push_str(&content);
                                chunk_count += 1;

                                let chunk = ResponseChunk {
                                    content,
                                    is_final: false,
                                    model_used: streaming_response.initial_metadata.model_id.clone(),
                                    generation_time_ms: Some(start_time.elapsed().as_millis() as u32),
                                };

                                if tx.send(Ok(chunk)).await.is_err() {
                                    warn!("Stream receiver dropped for session {}", session_id_clone);
                                    break;
                                }
                            },
                            crate::models::streaming::StreamEvent::Error { error_message, .. } => {
                                let _ = tx.send(Err(anyhow::anyhow!("Streaming error: {}", error_message))).await;
                                break;
                            },
                            crate::models::streaming::StreamEvent::Completed { .. } => {
                                // Send final chunk
                                let final_chunk = ResponseChunk {
                                    content: content_buffer.clone(),
                                    is_final: true,
                                    model_used: streaming_response.initial_metadata.model_id.clone(),
                                    generation_time_ms: Some(start_time.elapsed().as_millis() as u32),
                                };

                                let _ = tx.send(Ok(final_chunk)).await;

                                // Record interaction for session stats
                                let task_response = crate::models::TaskResponse {
                                    content: content_buffer,
                                    model_used: crate::models::ModelSelection::API(streaming_response.initial_metadata.model_id.clone()),
                                    tokens_generated: Some(streaming_response.initial_metadata.estimated_tokens.unwrap_or(0) as u32),
                                    generation_time_ms: Some(start_time.elapsed().as_millis() as u32),
                                    cost_cents: None,
                                    quality_score: 0.8,
                                    cost_info: None,
                                    model_info: Some(streaming_response.initial_metadata.model_id.clone()),
                                    error: None,
                                };

                                let _ = session_manager.record_interaction(&session_id_clone, &task_response).await;
                                
                                // Log streaming completion statistics
                                info!("Streaming completed for session {} - {} chunks processed in {:.2}s", 
                                     session_id_clone, chunk_count, start_time.elapsed().as_secs_f64());
                                break;
                            },
                            _ => {
                                // Handle other event types as needed
                                continue;
                            }
                        }
                    }
                });

                Ok(Box::new(ReceiverStream::new(rx)) as Box<dyn Stream<Item = Result<ResponseChunk>> + Send + Unpin>)
            },
            Err(_) => {
                // Streaming not available, fallback to single response
                info!("Streaming not supported for this model, falling back to single response");
                let response = orchestrator.execute_with_fallback(task_request).await?;

                // Update session stats
                self.session_manager.record_interaction(session_id, &response).await?;

                let chunks = vec![ResponseChunk {
                    content: response.content,
                    is_final: true,
                    model_used: response.model_used.model_id(),
                    generation_time_ms: response.generation_time_ms,
                }];

                Ok(Box::new(tokio_stream::iter(chunks.into_iter().map(Ok))) as Box<dyn Stream<Item = Result<ResponseChunk>> + Send + Unpin>)
            }
        }
    }

    /// Get current session status
    pub async fn get_session_status(&self, session_id: &SessionId) -> Result<SessionStatus> {
        let session = self
            .session_manager
            .get_session(session_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Session not found: {}", session_id))?;

        Ok(SessionStatus {
            session_id: session_id.clone(),
            setup_name: session.template.name.clone(),
            active_models: session.get_active_models().await,
            current_cost_cents: session.stats.total_cost_cents,
            message_count: session.stats.message_count,
            uptime_seconds: session.get_uptime_seconds(),
            status: session.get_health_status().await,
        })
    }

    /// Get all available setup templates
    pub async fn get_available_setups(&self) -> Result<Vec<SetupTemplate>> {
        self.template_manager.list_templates().await
    }

    /// Get setups filtered by category
    pub async fn get_setups_by_category(&self, category: &str) -> Result<Vec<SetupTemplate>> {
        self.template_manager.get_by_category(category).await
    }

    /// Save a setup as favorite for a user
    pub async fn save_favorite_setup(&self, user_id: &UserId, setup_id: &SetupId) -> Result<()> {
        info!("Saving setup {} as favorite for user {}", setup_id, user_id);
        self.profile_manager.add_favorite_setup(user_id, setup_id).await
    }

    /// Get personalized setup recommendations
    pub async fn get_recommended_setups(&self, user_id: &UserId) -> Result<Vec<Recommendation>> {
        let user_profile = self.profile_manager.get_or_create_profile(user_id).await?;
        let available_setups = self.template_manager.list_templates().await?;

        // Simple recommendation engine based on user preferences and history
        let mut recommendations = Vec::new();

        for template in available_setups {
            let score = self.calculate_recommendation_score(&template, &user_profile);

            if score > 0.6 {
                recommendations.push(Recommendation {
                    setup_template: template.clone(),
                    reason: self.generate_recommendation_reason(&template, &user_profile),
                    confidence_score: score,
                    expected_benefit: self.generate_expected_benefit(&template),
                });
            }
        }

        // Sort by confidence score
        recommendations
            .sort_by(|a, b| b.confidence_score.partial_cmp(&a.confidence_score).unwrap());
        recommendations.truncate(5); // Top 5 recommendations

        Ok(recommendations)
    }

    /// Get optimization suggestions for active session
    pub async fn get_optimization_suggestions(
        &self,
        session_id: &SessionId,
    ) -> Result<Vec<OptimizationSuggestion>> {
        let session = self
            .session_manager
            .get_session(session_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Session not found: {}", session_id))?;

        // Analyze current performance and suggest improvements
        let suggestions = self.analyze_session_performance(&session).await?;

        Ok(suggestions)
    }

    /// Stop an active session
    pub async fn stop_session(&self, session_id: &SessionId) -> Result<()> {
        info!("Stopping session {}", session_id);
        self.session_manager.stop_session(session_id).await
    }

    /// Get user's favorite setups
    pub async fn get_user_favorites(&self, user_id: &UserId) -> Result<Vec<SetupTemplate>> {
        let user_profile = self.profile_manager.get_or_create_profile(user_id).await?;

        let mut favorites = Vec::new();
        for setup_id in &user_profile.favorite_setups {
            if let Ok(Some(template)) = self.template_manager.get_template(setup_id).await {
                favorites.push(template);
            }
        }

        Ok(favorites)
    }

    /// Get user's recent setups
    pub async fn get_user_recents(&self, user_id: &UserId) -> Result<Vec<SetupTemplate>> {
        let user_profile = self.profile_manager.get_or_create_profile(user_id).await?;

        let mut recents = Vec::new();
        for usage_record in user_profile.usage_history.iter().rev().take(5) {
            if let Ok(Some(template)) =
                self.template_manager.get_template(&usage_record.setup_id).await
            {
                recents.push(template);
            }
        }

        Ok(recents)
    }

    // Helper methods

    fn infer_task_type(&self, message: &str) -> crate::models::TaskType {
        let message_lower = message.to_lowercase();

        // Simple heuristics to infer task type
        if message_lower.contains("write")
            || message_lower.contains("code")
            || message_lower.contains("function")
            || message_lower.contains("implement")
        {
            crate::models::TaskType::CodeGeneration { language: "unknown".to_string() }
        } else if message_lower.contains("explain")
            || message_lower.contains("analyze")
            || message_lower.contains("reason")
            || message_lower.contains("why")
        {
            crate::models::TaskType::LogicalReasoning
        } else if message_lower.contains("story")
            || message_lower.contains("creative")
            || message_lower.contains("write")
        {
            crate::models::TaskType::CreativeWriting
        } else {
            crate::models::TaskType::GeneralChat
        }
    }

    fn calculate_recommendation_score(
        &self,
        template: &SetupTemplate,
        user_profile: &super::UserProfile,
    ) -> f32 {
        let mut score = 0.5; // Base score

        // Check if user has used this category before
        for usage in &user_profile.usage_history {
            if usage.setup_id.as_str().contains(&template.category.to_string()) {
                score += 0.2;
                break;
            }
        }

        // Prefer setups within budget
        if let Some(budget) = &user_profile.preferences.budget_limit_cents_per_hour {
            if template.cost_estimate.hourly_cost_cents <= *budget {
                score += 0.2;
            } else {
                score -= 0.3;
            }
        }

        // Consider user's performance preferences
        match user_profile.preferences.preferred_response_time {
            crate::ui::profiles::ResponseTimePreference::Speed => {
                if matches!(
                    template.performance_profile.response_time,
                    super::ResponseTimeClass::Lightning | super::ResponseTimeClass::Fast
                ) {
                    score += 0.3;
                }
            }
            crate::ui::profiles::ResponseTimePreference::Quality => {
                if matches!(
                    template.performance_profile.quality_level,
                    super::QualityLevel::Premium | super::QualityLevel::Excellence
                ) {
                    score += 0.3;
                }
            }
            crate::ui::profiles::ResponseTimePreference::Balanced => {
                score += 0.1; // Neutral bonus
            }
        }

        (score as f32).min(1.0).max(0.0)
    }

    fn generate_recommendation_reason(
        &self,
        _template: &SetupTemplate,
        user_profile: &super::UserProfile,
    ) -> String {
        format!(
            "Based on your {} preference and usage history",
            user_profile.preferences.preferred_response_time.to_string()
        )
    }

    fn generate_expected_benefit(&self, template: &SetupTemplate) -> String {
        format!(
            "{:?} performance with {:?} cost",
            template.performance_profile.quality_level, template.cost_estimate.cost_class
        )
    }

    async fn analyze_session_performance(
        &self,
        session: &ActiveSession,
    ) -> Result<Vec<OptimizationSuggestion>> {
        // Simple performance analysis - could be much more sophisticated
        let mut suggestions = Vec::new();

        // Check if cost is high
        if session.stats.total_cost_cents > 100.0 {
            suggestions.push(OptimizationSuggestion {
                current_setup: session.template.id.clone(),
                suggested_changes: vec![super::SetupChange::AddModel {
                    model_id: "local-alternative".to_string(),
                    reason: "Reduce API costs by using more local models".to_string(),
                }],
                expected_improvement: super::ImprovementMetrics {
                    cost_change_percent: -30.0,
                    speed_change_percent: 10.0,
                    quality_change_percent: -5.0,
                },
                explanation: "Your session has high API costs. Adding local models could reduce \
                              costs."
                    .to_string(),
            });
        }

        Ok(suggestions)
    }
}

/// Response chunk for streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseChunk {
    pub content: String,
    pub is_final: bool,
    pub model_used: String,
    pub generation_time_ms: Option<u32>,
}

/// Session status for UI display
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionStatus {
    pub session_id: SessionId,
    pub setup_name: String,
    pub active_models: Vec<String>,
    pub current_cost_cents: f32,
    pub message_count: u32,
    pub uptime_seconds: u64,
    pub status: SessionHealth,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SessionHealth {
    Healthy,
    Warning(String),
    Error(String),
}
