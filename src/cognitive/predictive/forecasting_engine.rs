use std::collections::HashMap;

use anyhow::Result;
use chrono::{DateTime, Datelike, Duration, Timelike, Utc};
use tracing::{debug, info};

use super::predictive_architecture::{ForecastingResult, Prediction, PredictionData};

/// Advanced forecasting engine with multiple prediction algorithms
#[derive(Debug)]
pub struct ForecastingEngine {
    /// Engine identifier
    pub engine_id: String,

    /// Engine type
    pub engine_type: String,

    /// Forecasting models
    models: Vec<ForecastingModel>,

    /// Model selection algorithm
    model_selector: ModelSelector,

    /// Performance tracker
    performance_tracker: PerformanceTracker,

    /// Real-time adaptation engine
    adaptation_engine: AdaptationEngine,
}

/// Forecasting model
#[derive(Debug, Clone)]
pub struct ForecastingModel {
    /// Model identifier
    pub model_id: String,

    /// Model type
    pub model_type: ModelType,

    /// Model parameters
    pub parameters: ModelParameters,

    /// Model performance
    pub performance: ModelPerformance,

    /// Training data
    pub training_data: Vec<TrainingPoint>,

    /// Model state
    pub state: ModelState,
}

/// Types of forecasting models
#[derive(Debug, Clone, PartialEq)]
pub enum ModelType {
    LinearRegression,
    PolynomialRegression,
    ARIMA,
    Prophet,
    LSTM,
    GRU,
    Transformer,
    RandomForest,
    GradientBoosting,
    SupportVectorRegression,
    NeuralNetwork,
    EnsembleModel,
}

/// Model parameters
#[derive(Debug, Clone)]
pub struct ModelParameters {
    /// Learning rate
    pub learning_rate: f64,

    /// Regularization strength
    pub regularization: f64,

    /// Model complexity
    pub complexity: f64,

    /// Hyperparameters
    pub hyperparameters: HashMap<String, f64>,

    /// Feature engineering parameters
    pub feature_params: FeatureParameters,
}

/// Feature engineering parameters
#[derive(Debug, Clone)]
pub struct FeatureParameters {
    /// Window size for features
    pub window_size: usize,

    /// Lag features
    pub lag_features: Vec<usize>,

    /// Rolling statistics
    pub rolling_stats: Vec<RollingStatistic>,

    /// Seasonal features
    pub seasonal_features: bool,

    /// Trend features
    pub trend_features: bool,
}

/// Rolling statistics for features
#[derive(Debug, Clone, PartialEq)]
pub enum RollingStatistic {
    Mean,
    Median,
    StandardDeviation,
    Min,
    Max,
    Quantile(f64),
    Skewness,
    Kurtosis,
}

/// Model performance metrics
#[derive(Debug, Clone, Default)]
pub struct ModelPerformance {
    /// Mean Absolute Error
    pub mae: f64,

    /// Root Mean Squared Error
    pub rmse: f64,

    /// Mean Absolute Percentage Error
    pub mape: f64,

    /// R-squared score
    pub r2_score: f64,

    /// Prediction accuracy
    pub accuracy: f64,

    /// Model confidence
    pub confidence: f64,

    /// Training time
    pub training_time: std::time::Duration,

    /// Inference time
    pub inference_time: std::time::Duration,
}

/// Training data point
#[derive(Debug, Clone)]
pub struct TrainingPoint {
    /// Input features
    pub features: Vec<f64>,

    /// Target value
    pub target: f64,

    /// Timestamp
    pub timestamp: DateTime<Utc>,

    /// Data weight
    pub weight: f64,
}

/// Model state
#[derive(Debug, Clone, PartialEq)]
pub enum ModelState {
    Untrained,
    Training,
    Trained,
    Validating,
    Production,
    Retraining,
    Deprecated,
}

/// Model selection algorithm
#[derive(Debug)]
pub struct ModelSelector {
    /// Selection strategy
    selection_strategy: SelectionStrategy,

    /// Performance history
    performance_history: HashMap<String, Vec<f64>>,

    /// Model rankings
    model_rankings: Vec<ModelRanking>,

    /// Selection criteria
    criteria: Vec<SelectionCriterion>,
}

/// Model selection strategies
#[derive(Debug, Clone, PartialEq)]
pub enum SelectionStrategy {
    BestPerformance,   // Select model with best performance
    EnsembleWeighted,  // Ensemble based on performance weights
    AdaptiveSelection, // Adaptive model selection
    MetaLearning,      // Meta-learning based selection
    BayesianOptimal,   // Bayesian optimal selection
}

/// Model ranking
#[derive(Debug, Clone)]
pub struct ModelRanking {
    /// Model identifier
    pub model_id: String,

    /// Rank position
    pub rank: usize,

    /// Performance score
    pub score: f64,

    /// Confidence in ranking
    pub confidence: f64,
}

/// Selection criterion
#[derive(Debug, Clone)]
pub struct SelectionCriterion {
    /// Criterion name
    pub name: String,

    /// Criterion weight
    pub weight: f64,

    /// Optimization direction
    pub direction: OptimizationDirection,
}

/// Optimization direction
#[derive(Debug, Clone, PartialEq)]
pub enum OptimizationDirection {
    Maximize,
    Minimize,
}

/// Performance tracker
#[derive(Debug)]
pub struct PerformanceTracker {
    /// Performance metrics over time
    metrics_history: Vec<PerformanceSnapshot>,

    /// Model comparisons
    model_comparisons: Vec<ModelComparison>,

    /// Performance trends
    performance_trends: HashMap<String, PerformanceTrend>,
}

/// Performance snapshot
#[derive(Debug, Clone)]
pub struct PerformanceSnapshot {
    /// Timestamp
    pub timestamp: DateTime<Utc>,

    /// Model performance
    pub model_performance: HashMap<String, f64>,

    /// Context information
    pub context: String,
}

/// Model comparison
#[derive(Debug, Clone)]
pub struct ModelComparison {
    /// Model A
    pub model_a: String,

    /// Model B
    pub model_b: String,

    /// Performance difference
    pub performance_diff: f64,

    /// Statistical significance
    pub significance: f64,

    /// Comparison context
    pub context: String,
}

/// Performance trend
#[derive(Debug, Clone)]
pub struct PerformanceTrend {
    /// Trend direction
    pub direction: TrendDirection,

    /// Trend strength
    pub strength: f64,

    /// Trend significance
    pub significance: f64,

    /// Data points
    pub data_points: Vec<(DateTime<Utc>, f64)>,
}

/// Trend direction
#[derive(Debug, Clone, PartialEq)]
pub enum TrendDirection {
    Improving,
    Declining,
    Stable,
    Volatile,
}

/// Real-time adaptation engine
#[derive(Debug)]
pub struct AdaptationEngine {
    /// Adaptation strategies
    strategies: Vec<AdaptationStrategy>,

    /// Trigger conditions
    triggers: Vec<AdaptationTrigger>,

    /// Adaptation history
    adaptation_history: Vec<AdaptationEvent>,

    /// Performance thresholds
    thresholds: PerformanceThresholds,
}

/// Adaptation strategy
#[derive(Debug, Clone)]
pub struct AdaptationStrategy {
    /// Strategy identifier
    pub id: String,

    /// Strategy type
    pub strategy_type: AdaptationType,

    /// Strategy parameters
    pub parameters: HashMap<String, f64>,

    /// Strategy effectiveness
    pub effectiveness: f64,
}

/// Types of adaptations
#[derive(Debug, Clone, PartialEq)]
pub enum AdaptationType {
    ParameterTuning,    // Adjust model parameters
    ModelSelection,     // Switch to different model
    FeatureEngineering, // Modify feature engineering
    EnsembleWeighting,  // Adjust ensemble weights
    OnlineUpdate,       // Online model updating
    Retraining,         // Full model retraining
}

/// Adaptation trigger
#[derive(Debug, Clone)]
pub struct AdaptationTrigger {
    /// Trigger identifier
    pub id: String,

    /// Trigger condition
    pub condition: TriggerCondition,

    /// Trigger threshold
    pub threshold: f64,

    /// Associated strategy
    pub strategy: String,
}

/// Trigger conditions
#[derive(Debug, Clone, PartialEq)]
pub enum TriggerCondition {
    PerformanceDrop, // Performance drops below threshold
    AccuracyDecline, // Accuracy declines
    ConfidenceDrop,  // Confidence drops
    DataDrift,       // Data distribution drift
    ConceptDrift,    // Concept drift detected
    TimeElapsed,     // Time since last update
}

/// Adaptation event
#[derive(Debug, Clone)]
pub struct AdaptationEvent {
    /// Event identifier
    pub id: String,

    /// Event timestamp
    pub timestamp: DateTime<Utc>,

    /// Adaptation type
    pub adaptation_type: AdaptationType,

    /// Trigger that caused adaptation
    pub trigger: String,

    /// Performance before adaptation
    pub performance_before: f64,

    /// Performance after adaptation
    pub performance_after: f64,

    /// Adaptation success
    pub success: bool,
}

/// Performance thresholds
#[derive(Debug, Clone)]
pub struct PerformanceThresholds {
    /// Minimum accuracy threshold
    pub min_accuracy: f64,

    /// Minimum confidence threshold
    pub min_confidence: f64,

    /// Maximum error threshold
    pub max_error: f64,

    /// Performance degradation threshold
    pub degradation_threshold: f64,
}

impl ForecastingEngine {
    /// Create new forecasting engine
    pub async fn new(engine_type: &str) -> Result<Self> {
        info!("🚀 Creating forecasting engine: {}", engine_type);

        let engine_id = format!("forecast_engine_{}", uuid::Uuid::new_v4());

        let mut engine = Self {
            engine_id,
            engine_type: engine_type.to_string(),
            models: Vec::new(),
            model_selector: ModelSelector::new(),
            performance_tracker: PerformanceTracker::new(),
            adaptation_engine: AdaptationEngine::new(),
        };

        // Initialize models based on engine type
        engine.initialize_models(engine_type).await?;

        info!("✅ Forecasting engine created: {}", engine.engine_id);
        Ok(engine)
    }

    /// Generate forecast from prediction data
    pub async fn generate_forecast(&self, data: &PredictionData) -> Result<ForecastingResult> {
        debug!("📈 Generating forecast for target: {}", data.target);

        // Select optimal model for this prediction
        let selected_model = self.model_selector.select_model(data).await?;

        // Prepare features from data
        let features = self.prepare_features(data).await?;

        // Generate predictions using selected model
        let predictions = self.predict_with_model(&selected_model, &features).await?;

        // Calculate forecast metrics
        let accuracy = self.calculate_forecast_accuracy(&predictions).await?;
        let confidence = self.calculate_forecast_confidence(&predictions).await?;

        // Determine forecast horizon
        let forecast_horizon = self.determine_forecast_horizon(data).await?;

        // Check for adaptation triggers
        self.check_adaptation_triggers(accuracy, confidence).await?;

        let result = ForecastingResult { predictions, accuracy, confidence, forecast_horizon };

        debug!("✅ Forecast generated with {:.2} accuracy, {:.2} confidence", accuracy, confidence);
        Ok(result)
    }

    /// Initialize models for engine type
    async fn initialize_models(&mut self, engine_type: &str) -> Result<()> {
        match engine_type {
            "time_series_forecasting" => {
                self.add_model(ModelType::ARIMA, "ARIMA forecasting model").await?;
                self.add_model(ModelType::Prophet, "Prophet forecasting model").await?;
                self.add_model(ModelType::LSTM, "LSTM neural network").await?;
                self.add_model(ModelType::EnsembleModel, "Ensemble time series model").await?;
            }
            "regression_forecasting" => {
                self.add_model(ModelType::LinearRegression, "Linear regression model").await?;
                self.add_model(ModelType::PolynomialRegression, "Polynomial regression model")
                    .await?;
                self.add_model(ModelType::RandomForest, "Random forest regressor").await?;
                self.add_model(ModelType::GradientBoosting, "Gradient boosting regressor").await?;
            }
            "neural_forecasting" => {
                self.add_model(ModelType::NeuralNetwork, "Deep neural network").await?;
                self.add_model(ModelType::LSTM, "LSTM network").await?;
                self.add_model(ModelType::GRU, "GRU network").await?;
                self.add_model(ModelType::Transformer, "Transformer network").await?;
            }
            "ensemble_forecasting" => {
                self.add_model(ModelType::EnsembleModel, "Primary ensemble model").await?;
                self.add_model(ModelType::RandomForest, "Random forest component").await?;
                self.add_model(ModelType::GradientBoosting, "Boosting component").await?;
                self.add_model(ModelType::NeuralNetwork, "Neural component").await?;
            }
            "real_time_forecasting" => {
                self.add_model(ModelType::LinearRegression, "Fast linear model").await?;
                self.add_model(ModelType::NeuralNetwork, "Lightweight neural network").await?;
                self.add_model(ModelType::EnsembleModel, "Real-time ensemble").await?;
            }
            _ => {
                // Default models
                self.add_model(ModelType::LinearRegression, "Default linear model").await?;
                self.add_model(ModelType::RandomForest, "Default forest model").await?;
            }
        }

        debug!("🔧 Initialized {} models for engine type: {}", self.models.len(), engine_type);
        Ok(())
    }

    /// Add a forecasting model
    async fn add_model(&mut self, model_type: ModelType, _description: &str) -> Result<()> {
        let model = ForecastingModel {
            model_id: format!("model_{}", self.models.len()),
            model_type: model_type.clone(),
            parameters: ModelParameters::default_for_type(&model_type),
            performance: ModelPerformance::default(),
            training_data: Vec::new(),
            state: ModelState::Untrained,
        };

        self.models.push(model);
        Ok(())
    }

    /// Prepare features from prediction data
    async fn prepare_features(&self, data: &PredictionData) -> Result<Vec<Vec<f64>>> {
        let mut features = Vec::new();

        // Convert historical data to feature vectors
        for (i, point) in data.historical_data.iter().enumerate() {
            let mut feature_vector = vec![point.value];

            // Add lag features
            for lag in 1..=5 {
                if i >= lag {
                    feature_vector.push(data.historical_data[i - lag].value);
                } else {
                    feature_vector.push(0.0); // Padding for missing lags
                }
            }

            // Add rolling statistics
            if i >= 5 {
                let window: Vec<f64> =
                    data.historical_data[i - 4..=i].iter().map(|p| p.value).collect();
                feature_vector.push(window.iter().sum::<f64>() / window.len() as f64); // Mean
                let variance = window
                    .iter()
                    .map(|&x| (x - (window.iter().sum::<f64>() / window.len() as f64)).powi(2))
                    .sum::<f64>()
                    / window.len() as f64;
                feature_vector.push(variance.sqrt()); // Standard deviation
            } else {
                feature_vector.push(point.value); // Use current value as fallback
                feature_vector.push(0.0); // No std dev for small windows
            }

            // Add temporal features
            feature_vector.push(point.timestamp.hour() as f64 / 24.0); // Hour of day
            feature_vector.push(point.timestamp.weekday().num_days_from_monday() as f64 / 7.0); // Day of week

            features.push(feature_vector);
        }

        debug!(
            "🔧 Prepared {} feature vectors with {} features each",
            features.len(),
            features.first().map_or(0, |f| f.len())
        );
        Ok(features)
    }

    /// Predict using selected model
    async fn predict_with_model(
        &self,
        model_id: &str,
        features: &[Vec<f64>],
    ) -> Result<Vec<Prediction>> {
        let mut predictions = Vec::new();

        if let Some(model) = self.models.iter().find(|m| m.model_id == *model_id) {
            // Simulate predictions based on model type
            for (i, feature_vector) in features.iter().enumerate() {
                let predicted_value = self.simulate_model_prediction(model, feature_vector).await?;
                let confidence =
                    self.calculate_prediction_confidence(model, feature_vector).await?;

                let prediction = Prediction {
                    target: format!("target_{}", i),
                    predicted_value,
                    confidence,
                    horizon: Utc::now() + Duration::minutes((i + 1) as i64 * 5),
                    model_used: model_id.to_string(),
                };

                predictions.push(prediction);
            }
        }

        debug!("📊 Generated {} predictions using model: {}", predictions.len(), model_id);
        Ok(predictions)
    }

    /// Simulate model prediction (simplified implementation)
    async fn simulate_model_prediction(
        &self,
        model: &ForecastingModel,
        features: &[f64],
    ) -> Result<f64> {
        let prediction = match model.model_type {
            ModelType::LinearRegression => {
                // Simple linear combination of features
                features.iter().enumerate().map(|(i, &f)| f * (0.1 + i as f64 * 0.05)).sum::<f64>()
            }
            ModelType::ARIMA => {
                // ARIMA-like prediction (simplified)
                let latest_values: Vec<f64> = features.iter().take(3).cloned().collect();
                let trend = if latest_values.len() >= 2 {
                    latest_values[0] - latest_values[1]
                } else {
                    0.0
                };
                latest_values[0] + trend * 0.8
            }
            ModelType::LSTM | ModelType::GRU => {
                // Neural network prediction (simplified)
                let weighted_sum = features
                    .iter()
                    .enumerate()
                    .map(|(i, &f)| f * (0.8_f64).powi(i as i32))
                    .sum::<f64>();
                weighted_sum.tanh() * features[0] * 1.1
            }
            ModelType::RandomForest => {
                // Random forest prediction (simplified)
                let feature_sum = features.iter().sum::<f64>();
                let feature_mean = feature_sum / features.len() as f64;
                feature_mean * (1.0 + (feature_sum % 1.0) * 0.1)
            }
            ModelType::EnsembleModel => {
                // Ensemble prediction (combination of methods)
                let linear = features
                    .iter()
                    .enumerate()
                    .map(|(i, &f)| f * (0.1 + i as f64 * 0.05))
                    .sum::<f64>();
                let trend_based = if features.len() >= 2 {
                    features[0] + (features[0] - features[1]) * 0.8
                } else {
                    features[0]
                };
                linear * 0.4 + trend_based * 0.6
            }
            _ => {
                // Default prediction
                features[0] * 1.05 // Simple 5% growth assumption
            }
        };

        Ok(prediction)
    }

    /// Calculate prediction confidence
    async fn calculate_prediction_confidence(
        &self,
        model: &ForecastingModel,
        features: &[f64],
    ) -> Result<f64> {
        // Base confidence on model performance and feature quality
        let base_confidence = model.performance.confidence.max(0.5);

        // Adjust for feature quality
        let feature_quality = if features.is_empty() {
            0.0
        } else {
            let feature_variance = {
                let mean = features.iter().sum::<f64>() / features.len() as f64;
                features.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / features.len() as f64
            };
            (1.0 / (1.0 + feature_variance)).min(1.0)
        };

        let final_confidence = (base_confidence * 0.7 + feature_quality * 0.3).min(1.0);
        Ok(final_confidence)
    }

    /// Calculate forecast accuracy
    async fn calculate_forecast_accuracy(&self, predictions: &[Prediction]) -> Result<f64> {
        if predictions.is_empty() {
            return Ok(0.0);
        }

        // Calculate aggregate accuracy from individual prediction confidences
        let total_confidence: f64 = predictions.iter().map(|p| p.confidence).sum();
        let avg_confidence = total_confidence / predictions.len() as f64;

        // Adjust for prediction consistency
        let confidence_variance = {
            predictions.iter().map(|p| (p.confidence - avg_confidence).powi(2)).sum::<f64>()
                / predictions.len() as f64
        };

        let consistency_factor = 1.0 / (1.0 + confidence_variance * 10.0);
        let accuracy = avg_confidence * consistency_factor;

        Ok(accuracy.min(1.0))
    }

    /// Calculate forecast confidence
    async fn calculate_forecast_confidence(&self, predictions: &[Prediction]) -> Result<f64> {
        if predictions.is_empty() {
            return Ok(0.0);
        }

        // Weighted confidence based on prediction horizon
        let mut weighted_confidence = 0.0;
        let mut total_weight = 0.0;

        let now = Utc::now();

        for prediction in predictions {
            let time_diff = prediction.horizon.signed_duration_since(now);
            let hours_ahead = time_diff.num_hours().max(1) as f64;

            // Confidence decreases with time horizon
            let time_weight = 1.0 / (1.0 + hours_ahead * 0.1);
            weighted_confidence += prediction.confidence * time_weight;
            total_weight += time_weight;
        }

        let final_confidence =
            if total_weight > 0.0 { weighted_confidence / total_weight } else { 0.0 };

        Ok(final_confidence.min(1.0))
    }

    /// Determine forecast horizon
    async fn determine_forecast_horizon(&self, data: &PredictionData) -> Result<Duration> {
        // Determine horizon based on data frequency and quality
        let data_points = data.historical_data.len();
        let data_quality = data.quality_score;

        // Base horizon on data availability
        let base_hours = match data_points {
            0..=10 => 1,      // Very limited data: 1 hour
            11..=50 => 6,     // Limited data: 6 hours
            51..=200 => 24,   // Moderate data: 24 hours
            201..=1000 => 72, // Good data: 72 hours
            _ => 168,         // Extensive data: 1 week
        };

        // Adjust for data quality
        let quality_factor = data_quality;
        let adjusted_hours = (base_hours as f64 * quality_factor) as i64;

        Ok(Duration::hours(adjusted_hours.max(1)))
    }

    /// Check adaptation triggers
    async fn check_adaptation_triggers(&self, accuracy: f64, confidence: f64) -> Result<()> {
        for trigger in &self.adaptation_engine.triggers {
            let should_trigger = match trigger.condition {
                TriggerCondition::PerformanceDrop => accuracy < trigger.threshold,
                TriggerCondition::AccuracyDecline => accuracy < trigger.threshold,
                TriggerCondition::ConfidenceDrop => confidence < trigger.threshold,
                _ => false, // Other triggers would need more context
            };

            if should_trigger {
                debug!(
                    "🚨 Adaptation trigger activated: {} (threshold: {})",
                    trigger.id, trigger.threshold
                );
                self.execute_adaptation(&trigger.strategy).await?;
            }
        }

        Ok(())
    }

    /// Execute adaptation strategy
    async fn execute_adaptation(&self, strategy_id: &str) -> Result<()> {
        debug!("🔄 Executing adaptation strategy: {}", strategy_id);

        // Find and execute the adaptation strategy
        for strategy in &self.adaptation_engine.strategies {
            if strategy.id == strategy_id {
                match strategy.strategy_type {
                    AdaptationType::ParameterTuning => {
                        debug!("⚙️ Tuning model parameters");
                        // Implementation would adjust model parameters
                    }
                    AdaptationType::ModelSelection => {
                        debug!("🔀 Switching to different model");
                        // Implementation would select different model
                    }
                    AdaptationType::OnlineUpdate => {
                        debug!("📡 Performing online model update");
                        // Implementation would update model with new data
                    }
                    _ => {
                        debug!("🔧 Executing adaptation: {:?}", strategy.strategy_type);
                    }
                }
                break;
            }
        }

        Ok(())
    }
}

impl ModelSelector {
    fn new() -> Self {
        Self {
            selection_strategy: SelectionStrategy::BestPerformance,
            performance_history: HashMap::new(),
            model_rankings: Vec::new(),
            criteria: vec![
                SelectionCriterion {
                    name: "accuracy".to_string(),
                    weight: 0.5,
                    direction: OptimizationDirection::Maximize,
                },
                SelectionCriterion {
                    name: "speed".to_string(),
                    weight: 0.3,
                    direction: OptimizationDirection::Minimize,
                },
                SelectionCriterion {
                    name: "robustness".to_string(),
                    weight: 0.2,
                    direction: OptimizationDirection::Maximize,
                },
            ],
        }
    }

    async fn select_model(&self, _data: &PredictionData) -> Result<String> {
        // Simplified model selection - in practice would analyze data characteristics
        Ok("LinearRegression_default".to_string())
    }
}

impl PerformanceTracker {
    fn new() -> Self {
        Self {
            metrics_history: Vec::new(),
            model_comparisons: Vec::new(),
            performance_trends: HashMap::new(),
        }
    }
}

impl AdaptationEngine {
    fn new() -> Self {
        Self {
            strategies: vec![
                AdaptationStrategy {
                    id: "parameter_tuning".to_string(),
                    strategy_type: AdaptationType::ParameterTuning,
                    parameters: HashMap::new(),
                    effectiveness: 0.7,
                },
                AdaptationStrategy {
                    id: "model_selection".to_string(),
                    strategy_type: AdaptationType::ModelSelection,
                    parameters: HashMap::new(),
                    effectiveness: 0.8,
                },
            ],
            triggers: vec![
                AdaptationTrigger {
                    id: "accuracy_drop".to_string(),
                    condition: TriggerCondition::AccuracyDecline,
                    threshold: 0.7,
                    strategy: "parameter_tuning".to_string(),
                },
                AdaptationTrigger {
                    id: "confidence_drop".to_string(),
                    condition: TriggerCondition::ConfidenceDrop,
                    threshold: 0.6,
                    strategy: "model_selection".to_string(),
                },
            ],
            adaptation_history: Vec::new(),
            thresholds: PerformanceThresholds {
                min_accuracy: 0.7,
                min_confidence: 0.6,
                max_error: 0.3,
                degradation_threshold: 0.1,
            },
        }
    }
}

impl ModelParameters {
    fn default_for_type(model_type: &ModelType) -> Self {
        let mut hyperparameters = HashMap::new();

        match model_type {
            ModelType::LinearRegression => {
                hyperparameters.insert("alpha".to_string(), 1.0);
            }
            ModelType::RandomForest => {
                hyperparameters.insert("n_estimators".to_string(), 100.0);
                hyperparameters.insert("max_depth".to_string(), 10.0);
            }
            ModelType::NeuralNetwork => {
                hyperparameters.insert("hidden_layers".to_string(), 3.0);
                hyperparameters.insert("hidden_units".to_string(), 64.0);
            }
            ModelType::LSTM => {
                hyperparameters.insert("sequence_length".to_string(), 10.0);
                hyperparameters.insert("hidden_units".to_string(), 50.0);
            }
            _ => {
                hyperparameters.insert("default_param".to_string(), 1.0);
            }
        }

        Self {
            learning_rate: 0.001,
            regularization: 0.01,
            complexity: 1.0,
            hyperparameters,
            feature_params: FeatureParameters::default(),
        }
    }
}

impl Default for FeatureParameters {
    fn default() -> Self {
        Self {
            window_size: 10,
            lag_features: vec![1, 2, 3, 5, 10],
            rolling_stats: vec![RollingStatistic::Mean, RollingStatistic::StandardDeviation],
            seasonal_features: true,
            trend_features: true,
        }
    }
}
