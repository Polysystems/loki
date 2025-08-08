//! Cognitive Update Connector
//! 
//! Bridges cognitive systems to the UI data stream for real-time updates

use std::sync::Arc;
use std::time::Duration;
use anyhow::{Result, Context};
use tokio::sync::{mpsc, RwLock};
use tokio::task::JoinHandle;
use tracing::{info, warn};

use crate::tui::{
    cognitive::core::data_stream::{CognitiveDataStream, CognitiveDataUpdate},
    chat::integrations::cognitive::CognitiveChatEnhancement,
    cognitive::integration::background_processor::{BackgroundCognitiveProcessor, BackgroundEvent},
    cognitive::integration::main::{ CognitiveModality},
};

/// Connector configuration
pub struct ConnectorConfig {
    /// Update interval for polling
    pub update_interval: Duration,
    
    /// Enable reasoning updates
    pub enable_reasoning: bool,
    
    /// Enable consciousness updates
    pub enable_consciousness: bool,
    
    /// Enable background processing updates
    pub enable_background: bool,
    
    /// Maximum update rate per second
    pub max_updates_per_second: usize,
}

impl Default for ConnectorConfig {
    fn default() -> Self {
        Self {
            update_interval: Duration::from_millis(100),
            enable_reasoning: true,
            enable_consciousness: true,
            enable_background: true,
            max_updates_per_second: 20,
        }
    }
}

/// Cognitive update connector
pub struct CognitiveUpdateConnector {
    /// Data stream reference
    data_stream: Arc<CognitiveDataStream>,
    
    /// Cognitive enhancement reference
    cognitive_enhancement: Option<Arc<CognitiveChatEnhancement>>,
    
    /// Background processor reference
    background_processor: Option<Arc<BackgroundCognitiveProcessor>>,
    
    /// Configuration
    config: ConnectorConfig,
    
    /// Update sender
    update_tx: mpsc::Sender<CognitiveDataUpdate>,
    
    /// Active tasks
    active_tasks: Vec<JoinHandle<()>>,
    
    /// Active state
    is_active: Arc<RwLock<bool>>,
}

impl CognitiveUpdateConnector {
    /// Create new connector
    pub fn new(data_stream: Arc<CognitiveDataStream>, config: ConnectorConfig) -> Self {
        let update_tx = data_stream.get_update_sender();
        
        Self {
            data_stream,
            cognitive_enhancement: None,
            background_processor: None,
            config,
            update_tx,
            active_tasks: Vec::new(),
            is_active: Arc::new(RwLock::new(false)),
        }
    }
    
    /// Set cognitive enhancement
    pub fn set_cognitive_enhancement(&mut self, enhancement: Arc<CognitiveChatEnhancement>) {
        self.cognitive_enhancement = Some(enhancement);
    }
    
    /// Set background processor
    pub fn set_background_processor(&mut self, processor: Arc<BackgroundCognitiveProcessor>) {
        self.background_processor = Some(processor);
    }
    
    /// Start connector
    pub async fn start(&mut self) -> Result<()> {
        if *self.is_active.read().await {
            return Ok(());
        }
        
        info!("Starting cognitive update connector");
        *self.is_active.write().await = true;
        
        // Start data stream
        // Note: data_stream should be started before creating the connector
        // self.data_stream.clone().start().await?;
        
        // Start update tasks based on configuration
        if self.config.enable_consciousness {
            self.start_consciousness_updates().await?;
        }
        
        if self.config.enable_reasoning {
            self.start_reasoning_updates().await?;
        }
        
        if self.config.enable_background {
            self.start_background_updates().await?;
        }
        
        Ok(())
    }
    
    /// Stop connector
    pub async fn stop(&mut self) -> Result<()> {
        info!("Stopping cognitive update connector");
        *self.is_active.write().await = false;
        
        // Cancel all tasks
        for task in self.active_tasks.drain(..) {
            task.abort();
        }
        
        Ok(())
    }
    
    /// Start consciousness updates
    async fn start_consciousness_updates(&mut self) -> Result<()> {
        if let Some(enhancement) = &self.cognitive_enhancement {
            let update_tx = self.update_tx.clone();
            let enhancement = enhancement.clone();
            let is_active = self.is_active.clone();
            let interval = self.config.update_interval;
            
            let task = tokio::spawn(async move {
                let mut ticker = tokio::time::interval(interval);
                
                while *is_active.read().await {
                    ticker.tick().await;
                    
                    // Get consciousness activity
                    let activity = enhancement.get_cognitive_activity();
                    {
                        let update = CognitiveDataUpdate::CognitiveActivity {
                            awareness_level: activity.awareness_level,
                            coherence: activity.gradient_coherence,
                            free_energy: activity.free_energy,
                            current_focus: activity.current_focus,
                        };
                        
                        if let Err(e) = update_tx.send(update).await {
                            warn!("Failed to send consciousness update: {}", e);
                        }
                    }
                    
                    // Check for new insights (placeholder for now)
                    // Cognitive stream not available yet - will be added later
                    /*
                    if let Some(stream) = &enhancement.cognitive_stream {
                        let stream = stream.read().await;
                        let insights = stream.get_recent_insights(1).await;
                        for insight in insights {
                            let update = CognitiveDataUpdate::Insight {
                                insight,
                                relevance: 0.8,
                                novelty: 0.7,
                            };
                            
                            if let Err(e) = update_tx.send(update).await {
                                warn!("Failed to send insight update: {}", e);
                            }
                        }
                    }
                    */
                }
            });
            
            self.active_tasks.push(task);
        }
        
        Ok(())
    }
    
    /// Start reasoning updates
    async fn start_reasoning_updates(&mut self) -> Result<()> {
        if let Some(enhancement) = &self.cognitive_enhancement {
            let update_tx = self.update_tx.clone();
            let enhancement = enhancement.clone();
            let is_active = self.is_active.clone();
            
            // Monitor for reasoning chain updates
            let task = tokio::spawn(async move {
                let mut last_chain_id = String::new();
                let mut ticker = tokio::time::interval(Duration::from_millis(200));
                
                while *is_active.read().await {
                    ticker.tick().await;
                }
            });
            
            self.active_tasks.push(task);
        }
        
        Ok(())
    }
    
    /// Start background processing updates
    async fn start_background_updates(&mut self) -> Result<()> {
        if let Some(processor) = &self.background_processor {
            let update_tx = self.update_tx.clone();
            let event_rx = processor.subscribe_events();
            let is_active = self.is_active.clone();
            
            let task = tokio::spawn(async move {
                let mut rx = event_rx;
                
                while *is_active.read().await {
                    match rx.recv().await {
                        Ok(event) => {
                            match event {
                                BackgroundEvent::TaskStarted { task, .. } => {
                                    let update = CognitiveDataUpdate::ProcessingStatus {
                                        is_processing: true,
                                        processing_depth: 0.5,
                                        active_tasks: vec![format!("{:?}", task)],
                                        resource_usage: 0.3,
                                    };
                                    
                                    let _ = update_tx.send(update).await;
                                }
                                
                                BackgroundEvent::InsightGenerated { insight, .. } => {
                                    let update = CognitiveDataUpdate::BackgroundThought {
                                        content: insight.content,
                                        category: insight.category,
                                        importance: insight.relevance,
                                    };
                                    
                                    let _ = update_tx.send(update).await;
                                }
                                
                                BackgroundEvent::LearningAchieved { learning, .. } => {
                                    let update = CognitiveDataUpdate::LearningEvent {
                                        topic: learning.topic,
                                        understanding_delta: learning.understanding_after - learning.understanding_before,
                                        key_realization: learning.key_realization,
                                    };
                                    
                                    let _ = update_tx.send(update).await;
                                }
                                
                                _ => {}
                            }
                        }
                        Err(_) => {
                            // Channel closed
                            break;
                        }
                    }
                }
            });
            
            self.active_tasks.push(task);
        }
        
        Ok(())
    }
    
    /// Send manual update
    pub async fn send_update(&self, update: CognitiveDataUpdate) -> Result<()> {
        self.update_tx.send(update).await
            .context("Failed to send cognitive update")
    }

    
    /// Send modality activation
    pub async fn send_modality_activation(
        &self,
        modality: CognitiveModality,
        level: f64,
        context: String,
    ) -> Result<()> {
        let update = CognitiveDataUpdate::ModalityActivation {
            modality,
            activation_level: level,
            context,
        };
        
        self.update_tx.send(update).await
            .context("Failed to send modality update")
    }
    
    /// Send processing step
    pub async fn send_processing_step(
        &self,
        step_type: crate::tui::cognitive::core::data_stream::ProcessingStepType,
        description: String,
        input_summary: String,
        output_summary: String,
        duration_ms: u64,
    ) -> Result<()> {
        let update = CognitiveDataUpdate::ProcessingStep {
            step_type,
            description,
            input_summary,
            output_summary,
            duration_ms,
            timestamp: std::time::Instant::now(),
        };
        
        self.update_tx.send(update).await
            .context("Failed to send processing step")
    }
}
