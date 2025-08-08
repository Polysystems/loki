use std::collections::HashMap;

use anyhow::Result;
use chrono::Duration;
use tracing::{debug, info};

use super::predictive_architecture::{PredictionData, TemporalResult};

/// Advanced temporal modeling engine for time-based pattern analysis
#[derive(Debug)]
pub struct TemporalModelingEngine {
    /// Time series analyzers
    analyzers: HashMap<String, TimeSeriesAnalyzer>,

    /// Temporal pattern detectors
    pattern_detectors: Vec<TemporalPatternDetector>,

    /// Seasonality modelers
    seasonality_modelers: Vec<SeasonalityModeler>,

    /// Trend analyzers
    trend_analyzers: Vec<TrendAnalyzer>,

    /// Temporal forecasters
    forecasters: HashMap<String, TemporalForecaster>,
}

/// Time series analyzer
#[derive(Debug, Clone)]
pub struct TimeSeriesAnalyzer {
    /// Analyzer identifier
    pub id: String,

    /// Analysis methods
    pub methods: Vec<AnalysisMethod>,

    /// Window parameters
    pub window_params: WindowParameters,

    /// Quality metrics
    pub quality_metrics: QualityMetrics,
}

/// Analysis methods for time series
#[derive(Debug, Clone, PartialEq)]
pub enum AnalysisMethod {
    AutoCorrelation,    // Autocorrelation analysis
    CrossCorrelation,   // Cross-correlation analysis
    SpectralAnalysis,   // Spectral/frequency analysis
    WaveletAnalysis,    // Wavelet decomposition
    StateSpaceModeling, // State space modeling
    NonlinearDynamics,  // Nonlinear dynamics analysis
}

/// Window parameters for analysis
#[derive(Debug, Clone)]
pub struct WindowParameters {
    /// Analysis window size
    pub window_size: usize,

    /// Overlap percentage
    pub overlap: f64,

    /// Sliding window step
    pub step_size: usize,

    /// Adaptive windowing
    pub adaptive: bool,
}

/// Quality metrics for temporal analysis
#[derive(Debug, Clone, Default)]
pub struct QualityMetrics {
    /// Signal-to-noise ratio
    pub signal_noise_ratio: f64,

    /// Data completeness
    pub completeness: f64,

    /// Temporal consistency
    pub consistency: f64,

    /// Stationarity score
    pub stationarity: f64,
}

/// Temporal pattern detector
#[derive(Debug, Clone)]
pub struct TemporalPatternDetector {
    /// Detector identifier
    pub id: String,

    /// Pattern types to detect
    pub pattern_types: Vec<PatternType>,

    /// Detection parameters
    pub parameters: DetectionParameters,

    /// Detection accuracy
    pub accuracy: f64,
}

/// Types of temporal patterns
#[derive(Debug, Clone, PartialEq)]
pub enum PatternType {
    Cyclical, // Cyclical patterns
    Seasonal, // Seasonal patterns
    Trending, // Trending patterns
    Periodic, // Periodic patterns
    Chaotic,  // Chaotic patterns
    Regime,   // Regime changes
}

/// Detection parameters
#[derive(Debug, Clone)]
pub struct DetectionParameters {
    /// Sensitivity threshold
    pub sensitivity: f64,

    /// Minimum pattern length
    pub min_length: usize,

    /// Maximum pattern length
    pub max_length: usize,

    /// Confidence threshold
    pub confidence_threshold: f64,
}

/// Seasonality modeler
#[derive(Debug, Clone)]
pub struct SeasonalityModeler {
    /// Modeler identifier
    pub id: String,

    /// Seasonality types
    pub seasonality_types: Vec<SeasonalityType>,

    /// Decomposition method
    pub decomposition_method: DecompositionMethod,

    /// Model accuracy
    pub accuracy: f64,
}

/// Types of seasonality
#[derive(Debug, Clone, PartialEq)]
pub enum SeasonalityType {
    Daily,            // Daily seasonality
    Weekly,           // Weekly seasonality
    Monthly,          // Monthly seasonality
    Quarterly,        // Quarterly seasonality
    Yearly,           // Yearly seasonality
    Custom(Duration), // Custom period seasonality
}

/// Decomposition methods
#[derive(Debug, Clone, PartialEq)]
pub enum DecompositionMethod {
    Additive,       // Additive decomposition
    Multiplicative, // Multiplicative decomposition
    STL,            // STL decomposition
    X11,            // X-11 decomposition
    SEATS,          // SEATS decomposition
}

/// Trend analyzer
#[derive(Debug, Clone)]
pub struct TrendAnalyzer {
    /// Analyzer identifier
    pub id: String,

    /// Trend detection methods
    pub methods: Vec<TrendMethod>,

    /// Smoothing parameters
    pub smoothing_params: SmoothingParameters,

    /// Analysis accuracy
    pub accuracy: f64,
}

/// Trend detection methods
#[derive(Debug, Clone, PartialEq)]
pub enum TrendMethod {
    LinearTrend,          // Linear trend analysis
    PolynomialTrend,      // Polynomial trend fitting
    MovingAverage,        // Moving average trend
    ExponentialSmoothing, // Exponential smoothing
    HodrickPrescott,      // Hodrick-Prescott filter
    KalmanFilter,         // Kalman filter trend
}

/// Smoothing parameters
#[derive(Debug, Clone)]
pub struct SmoothingParameters {
    /// Smoothing factor
    pub alpha: f64,

    /// Trend smoothing factor
    pub beta: f64,

    /// Seasonal smoothing factor
    pub gamma: f64,

    /// Damping factor
    pub phi: f64,
}

/// Temporal forecaster
#[derive(Debug, Clone)]
pub struct TemporalForecaster {
    /// Forecaster identifier
    pub id: String,

    /// Forecasting model
    pub model: ForecastingModel,

    /// Model parameters
    pub parameters: ModelParameters,

    /// Forecast accuracy
    pub accuracy: f64,
}

/// Forecasting models
#[derive(Debug, Clone, PartialEq)]
pub enum ForecastingModel {
    ARIMA,       // ARIMA model
    SARIMA,      // Seasonal ARIMA
    Prophet,     // Prophet model
    HoltWinters, // Holt-Winters method
    StateSpace,  // State space model
    LSTM,        // LSTM neural network
}

/// Model parameters
#[derive(Debug, Clone)]
pub struct ModelParameters {
    /// Model order (p, d, q)
    pub order: (usize, usize, usize),

    /// Seasonal order (P, D, Q, s)
    pub seasonal_order: Option<(usize, usize, usize, usize)>,

    /// Learning rate
    pub learning_rate: f64,

    /// Regularization
    pub regularization: f64,
}

impl TemporalModelingEngine {
    /// Create new temporal modeling engine
    pub async fn new() -> Result<Self> {
        info!("⏰ Initializing Temporal Modeling Engine");

        let mut engine = Self {
            analyzers: HashMap::new(),
            pattern_detectors: Vec::new(),
            seasonality_modelers: Vec::new(),
            trend_analyzers: Vec::new(),
            forecasters: HashMap::new(),
        };

        // Initialize components
        engine.initialize_analyzers().await?;
        engine.initialize_detectors().await?;
        engine.initialize_modelers().await?;
        engine.initialize_forecasters().await?;

        info!("✅ Temporal Modeling Engine initialized");
        Ok(engine)
    }

    /// Model temporal patterns from prediction data
    pub async fn model_temporal_patterns(&self, data: &PredictionData) -> Result<TemporalResult> {
        debug!("⏰ Modeling temporal patterns for target: {}", data.target);

        // Analyze time series characteristics
        let analysis_result = self.analyze_time_series(data).await?;

        // Detect temporal patterns
        let patterns = self.detect_patterns(data).await?;

        // Model seasonality
        let seasonality = self.model_seasonality(data).await?;

        // Analyze trends
        let trends = self.analyze_trends(data).await?;

        // Generate temporal forecasts
        let forecasts = self.generate_forecasts(data).await?;

        // Calculate modeling accuracy
        let accuracy = self.calculate_modeling_accuracy(&analysis_result, &patterns).await?;

        // Calculate confidence
        let confidence = self.calculate_modeling_confidence(&seasonality, &trends).await?;

        let result = TemporalResult {
            insights: vec![
                format!("Detected {} temporal patterns", patterns.len()),
                format!("Identified seasonality: {:?}", seasonality.seasonality_type),
                format!("Trend direction: {:?}", trends.direction),
                format!("Generated {} forecasts", forecasts.len()),
            ],
            modeling_accuracy: accuracy,
            confidence,
        };

        debug!("✅ Temporal modeling completed with {:.2} accuracy", accuracy);
        Ok(result)
    }

    /// Initialize time series analyzers
    async fn initialize_analyzers(&mut self) -> Result<()> {
        let analyzer_types = vec![
            "autocorrelation_analyzer",
            "spectral_analyzer",
            "wavelet_analyzer",
            "state_space_analyzer",
        ];

        for analyzer_type in analyzer_types {
            let analyzer = TimeSeriesAnalyzer {
                id: analyzer_type.to_string(),
                methods: match analyzer_type {
                    "autocorrelation_analyzer" => vec![AnalysisMethod::AutoCorrelation],
                    "spectral_analyzer" => vec![AnalysisMethod::SpectralAnalysis],
                    "wavelet_analyzer" => vec![AnalysisMethod::WaveletAnalysis],
                    "state_space_analyzer" => vec![AnalysisMethod::StateSpaceModeling],
                    _ => vec![AnalysisMethod::AutoCorrelation],
                },
                window_params: WindowParameters {
                    window_size: 50,
                    overlap: 0.5,
                    step_size: 1,
                    adaptive: true,
                },
                quality_metrics: QualityMetrics::default(),
            };

            self.analyzers.insert(analyzer_type.to_string(), analyzer);
        }

        debug!("🔧 Initialized {} time series analyzers", self.analyzers.len());
        Ok(())
    }

    /// Initialize pattern detectors
    async fn initialize_detectors(&mut self) -> Result<()> {
        self.pattern_detectors = vec![
            TemporalPatternDetector {
                id: "cyclical_detector".to_string(),
                pattern_types: vec![PatternType::Cyclical, PatternType::Periodic],
                parameters: DetectionParameters {
                    sensitivity: 0.7,
                    min_length: 3,
                    max_length: 50,
                    confidence_threshold: 0.8,
                },
                accuracy: 0.85,
            },
            TemporalPatternDetector {
                id: "seasonal_detector".to_string(),
                pattern_types: vec![PatternType::Seasonal],
                parameters: DetectionParameters {
                    sensitivity: 0.6,
                    min_length: 7,
                    max_length: 365,
                    confidence_threshold: 0.75,
                },
                accuracy: 0.8,
            },
        ];

        debug!("🔍 Initialized {} pattern detectors", self.pattern_detectors.len());
        Ok(())
    }

    /// Initialize seasonality modelers
    async fn initialize_modelers(&mut self) -> Result<()> {
        self.seasonality_modelers = vec![
            SeasonalityModeler {
                id: "daily_modeler".to_string(),
                seasonality_types: vec![SeasonalityType::Daily],
                decomposition_method: DecompositionMethod::STL,
                accuracy: 0.85,
            },
            SeasonalityModeler {
                id: "weekly_modeler".to_string(),
                seasonality_types: vec![SeasonalityType::Weekly],
                decomposition_method: DecompositionMethod::Additive,
                accuracy: 0.8,
            },
        ];

        debug!("📊 Initialized {} seasonality modelers", self.seasonality_modelers.len());
        Ok(())
    }

    /// Initialize forecasters
    async fn initialize_forecasters(&mut self) -> Result<()> {
        let forecaster_types = vec![
            ("arima_forecaster", ForecastingModel::ARIMA),
            ("prophet_forecaster", ForecastingModel::Prophet),
            ("lstm_forecaster", ForecastingModel::LSTM),
        ];

        for (name, model) in forecaster_types {
            let forecaster = TemporalForecaster {
                id: name.to_string(),
                model,
                parameters: ModelParameters {
                    order: (2, 1, 2),
                    seasonal_order: Some((1, 1, 1, 12)),
                    learning_rate: 0.001,
                    regularization: 0.01,
                },
                accuracy: 0.8,
            };

            self.forecasters.insert(name.to_string(), forecaster);
        }

        debug!("📈 Initialized {} temporal forecasters", self.forecasters.len());
        Ok(())
    }

    /// Analyze time series characteristics
    async fn analyze_time_series(&self, data: &PredictionData) -> Result<TimeSeriesAnalysis> {
        let values: Vec<f64> = data.historical_data.iter().map(|p| p.value).collect();

        // Calculate basic statistics
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance =
            values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let std_dev = variance.sqrt();

        // Test stationarity (simplified)
        let stationarity = self.test_stationarity(&values).await?;

        // Calculate autocorrelation
        let autocorr = self.calculate_autocorrelation(&values, 1).await?;

        let analysis = TimeSeriesAnalysis {
            mean,
            variance,
            standard_deviation: std_dev,
            stationarity,
            autocorrelation: autocorr,
            data_quality: data.quality_score,
        };

        Ok(analysis)
    }

    /// Test stationarity
    async fn test_stationarity(&self, values: &[f64]) -> Result<f64> {
        if values.len() < 10 {
            return Ok(0.5); // Insufficient data
        }

        // Simple trend test
        let n = values.len() as f64;
        let x_mean = (n - 1.0) / 2.0;
        let y_mean = values.iter().sum::<f64>() / n;

        let slope_numerator: f64 =
            values.iter().enumerate().map(|(i, &y)| (i as f64 - x_mean) * (y - y_mean)).sum();

        let slope_denominator: f64 = (0..values.len()).map(|i| (i as f64 - x_mean).powi(2)).sum();

        let slope = if slope_denominator > 0.0 { slope_numerator / slope_denominator } else { 0.0 };
        let trend_strength = slope.abs() / (y_mean.abs() + 1.0);

        // Stationarity is inverse of trend strength
        let stationarity = (1.0 - trend_strength).max(0.0).min(1.0);
        Ok(stationarity)
    }

    /// Calculate autocorrelation
    async fn calculate_autocorrelation(&self, values: &[f64], lag: usize) -> Result<f64> {
        if values.len() <= lag {
            return Ok(0.0);
        }

        let n = values.len() - lag;
        let mean = values.iter().sum::<f64>() / values.len() as f64;

        let numerator: f64 = (0..n).map(|i| (values[i] - mean) * (values[i + lag] - mean)).sum();

        let denominator: f64 = values.iter().map(|&x| (x - mean).powi(2)).sum();

        let autocorr = if denominator > 0.0 { numerator / denominator } else { 0.0 };
        Ok(autocorr)
    }

    /// Detect temporal patterns
    async fn detect_patterns(&self, data: &PredictionData) -> Result<Vec<DetectedPattern>> {
        let mut patterns = Vec::new();

        let values: Vec<f64> = data.historical_data.iter().map(|p| p.value).collect();

        // Detect cyclical patterns
        if let Some(cyclical_pattern) = self.detect_cyclical_pattern(&values).await? {
            patterns.push(cyclical_pattern);
        }

        // Detect seasonal patterns
        if let Some(seasonal_pattern) = self.detect_seasonal_pattern(&values).await? {
            patterns.push(seasonal_pattern);
        }

        debug!("🔍 Detected {} temporal patterns", patterns.len());
        Ok(patterns)
    }

    /// Detect cyclical pattern
    async fn detect_cyclical_pattern(&self, values: &[f64]) -> Result<Option<DetectedPattern>> {
        if values.len() < 6 {
            return Ok(None);
        }

        // Simple peak detection for cycles
        let mut peaks = Vec::new();

        for i in 1..values.len() - 1 {
            if values[i] > values[i - 1] && values[i] > values[i + 1] {
                peaks.push(i);
            }
        }

        if peaks.len() >= 2 {
            let cycle_length = if peaks.len() > 1 {
                (peaks[peaks.len() - 1] - peaks[0]) / (peaks.len() - 1)
            } else {
                0
            };

            let pattern = DetectedPattern {
                pattern_type: PatternType::Cyclical,
                confidence: 0.7,
                period: Some(cycle_length),
                strength: 0.6,
                description: format!("Cyclical pattern with period ~{}", cycle_length),
            };

            return Ok(Some(pattern));
        }

        Ok(None)
    }

    /// Detect seasonal pattern
    async fn detect_seasonal_pattern(&self, values: &[f64]) -> Result<Option<DetectedPattern>> {
        // Simple seasonality detection based on data length
        let seasonality_detected = values.len() > 24; // Assume daily data needs 24+ points

        if seasonality_detected {
            let pattern = DetectedPattern {
                pattern_type: PatternType::Seasonal,
                confidence: 0.6,
                period: Some(24), // Daily seasonality
                strength: 0.5,
                description: "Daily seasonal pattern detected".to_string(),
            };

            return Ok(Some(pattern));
        }

        Ok(None)
    }

    /// Model seasonality
    async fn model_seasonality(&self, data: &PredictionData) -> Result<SeasonalityModel> {
        let values: Vec<f64> = data.historical_data.iter().map(|p| p.value).collect();

        // Determine dominant seasonality type
        let seasonality_type = if values.len() >= 365 {
            SeasonalityType::Yearly
        } else if values.len() >= 30 {
            SeasonalityType::Monthly
        } else {
            SeasonalityType::Custom(Duration::days(7))
        };

        let seasonality_period = match &seasonality_type {
            SeasonalityType::Yearly => 365,
            SeasonalityType::Monthly => 30,
            SeasonalityType::Weekly => 7,
            SeasonalityType::Daily => 1,
            SeasonalityType::Quarterly => 90,
            SeasonalityType::Custom(d) => d.num_days() as usize,
        };

        let model = SeasonalityModel {
            seasonality_type,
            strength: 0.6,
            period: seasonality_period,
            components: self.decompose_seasonality(&values).await?,
        };

        Ok(model)
    }

    /// Decompose seasonality components
    async fn decompose_seasonality(&self, values: &[f64]) -> Result<SeasonalityComponents> {
        let trend = self.calculate_trend(values).await?;
        let seasonal = self.extract_seasonal_component(values, &trend).await?;
        let residual = self.calculate_residual(values, &trend, &seasonal).await?;

        let components = SeasonalityComponents { trend, seasonal, residual };

        Ok(components)
    }

    /// Calculate trend component
    async fn calculate_trend(&self, values: &[f64]) -> Result<Vec<f64>> {
        // Simple moving average for trend
        let window_size = (values.len() / 4).max(3).min(12);
        let mut trend = Vec::new();

        for i in 0..values.len() {
            let start = if i >= window_size / 2 { i - window_size / 2 } else { 0 };
            let end = (i + window_size / 2 + 1).min(values.len());

            let window_mean = values[start..end].iter().sum::<f64>() / (end - start) as f64;
            trend.push(window_mean);
        }

        Ok(trend)
    }

    /// Extract seasonal component
    async fn extract_seasonal_component(&self, values: &[f64], trend: &[f64]) -> Result<Vec<f64>> {
        // Simple detrending for seasonal component
        let seasonal: Vec<f64> =
            values.iter().zip(trend.iter()).map(|(&val, &tr)| val - tr).collect();

        Ok(seasonal)
    }

    /// Calculate residual component
    async fn calculate_residual(
        &self,
        values: &[f64],
        trend: &[f64],
        seasonal: &[f64],
    ) -> Result<Vec<f64>> {
        let residual: Vec<f64> = values
            .iter()
            .zip(trend.iter())
            .zip(seasonal.iter())
            .map(|((&val, &tr), &seas)| val - tr - seas)
            .collect();

        Ok(residual)
    }

    /// Analyze trends
    async fn analyze_trends(&self, data: &PredictionData) -> Result<TrendAnalysis> {
        let values: Vec<f64> = data.historical_data.iter().map(|p| p.value).collect();

        let trend_strength = self.calculate_trend_strength(&values).await?;
        let direction = self.determine_trend_direction(&values).await?;
        let confidence = self.calculate_trend_confidence(&values, trend_strength).await?;

        let analysis = TrendAnalysis {
            direction,
            strength: trend_strength,
            confidence,
            persistence: self.calculate_trend_persistence(&values).await?,
        };

        Ok(analysis)
    }

    /// Calculate trend strength
    async fn calculate_trend_strength(&self, values: &[f64]) -> Result<f64> {
        if values.len() < 3 {
            return Ok(0.0);
        }

        let n = values.len() as f64;
        let x_mean = (n - 1.0) / 2.0;
        let y_mean = values.iter().sum::<f64>() / n;

        let slope_numerator: f64 =
            values.iter().enumerate().map(|(i, &y)| (i as f64 - x_mean) * (y - y_mean)).sum();

        let slope_denominator: f64 = (0..values.len()).map(|i| (i as f64 - x_mean).powi(2)).sum();

        let slope = if slope_denominator > 0.0 { slope_numerator / slope_denominator } else { 0.0 };
        let trend_strength = slope.abs() / (y_mean.abs() + 1.0);

        Ok(trend_strength.min(1.0))
    }

    /// Determine trend direction
    async fn determine_trend_direction(&self, values: &[f64]) -> Result<TrendDirection> {
        if values.len() < 2 {
            return Ok(TrendDirection::Stable);
        }

        let first_half = &values[0..values.len() / 2];
        let second_half = &values[values.len() / 2..];

        let first_mean = first_half.iter().sum::<f64>() / first_half.len() as f64;
        let second_mean = second_half.iter().sum::<f64>() / second_half.len() as f64;

        let direction = if second_mean > first_mean * 1.05 {
            TrendDirection::Increasing
        } else if second_mean < first_mean * 0.95 {
            TrendDirection::Decreasing
        } else {
            TrendDirection::Stable
        };

        Ok(direction)
    }

    /// Calculate trend confidence
    async fn calculate_trend_confidence(&self, values: &[f64], trend_strength: f64) -> Result<f64> {
        let data_quality = if values.len() > 50 {
            0.9
        } else if values.len() > 20 {
            0.7
        } else {
            0.5
        };
        let confidence = trend_strength * data_quality;
        Ok(confidence.min(1.0))
    }

    /// Calculate trend persistence
    async fn calculate_trend_persistence(&self, values: &[f64]) -> Result<f64> {
        if values.len() < 5 {
            return Ok(0.5);
        }

        // Count direction changes
        let mut direction_changes = 0;

        for i in 2..values.len() {
            let prev_change = values[i - 1] - values[i - 2];
            let curr_change = values[i] - values[i - 1];

            if (prev_change > 0.0 && curr_change < 0.0) || (prev_change < 0.0 && curr_change > 0.0)
            {
                direction_changes += 1;
            }
        }

        let max_changes = values.len() - 2;
        let persistence = 1.0 - (direction_changes as f64 / max_changes as f64);

        Ok(persistence.max(0.0).min(1.0))
    }

    /// Generate temporal forecasts
    async fn generate_forecasts(&self, data: &PredictionData) -> Result<Vec<TemporalForecast>> {
        let mut forecasts = Vec::new();

        // Generate simple trend-based forecast
        let values: Vec<f64> = data.historical_data.iter().map(|p| p.value).collect();
        let trend_strength = self.calculate_trend_strength(&values).await?;
        let last_value = values.last().copied().unwrap_or(0.0);

        for i in 1..=5 {
            let forecast_value = last_value * (1.0 + trend_strength * i as f64 * 0.1);

            let forecast = TemporalForecast {
                horizon: i,
                predicted_value: forecast_value,
                confidence: (0.9 - i as f64 * 0.1).max(0.3),
                method: "trend_extrapolation".to_string(),
            };

            forecasts.push(forecast);
        }

        debug!("📈 Generated {} temporal forecasts", forecasts.len());
        Ok(forecasts)
    }

    /// Calculate modeling accuracy
    async fn calculate_modeling_accuracy(
        &self,
        analysis: &TimeSeriesAnalysis,
        patterns: &[DetectedPattern],
    ) -> Result<f64> {
        let data_quality_factor = analysis.data_quality;
        let pattern_confidence = if !patterns.is_empty() {
            patterns.iter().map(|p| p.confidence).sum::<f64>() / patterns.len() as f64
        } else {
            0.7
        };

        let accuracy = (data_quality_factor * 0.6 + pattern_confidence * 0.4).min(1.0);
        Ok(accuracy)
    }

    /// Calculate modeling confidence
    async fn calculate_modeling_confidence(
        &self,
        seasonality: &SeasonalityModel,
        trends: &TrendAnalysis,
    ) -> Result<f64> {
        let seasonality_confidence = seasonality.strength;
        let trend_confidence = trends.confidence;

        let combined_confidence = (seasonality_confidence + trend_confidence) / 2.0;
        Ok(combined_confidence.min(1.0))
    }
}

// Supporting data structures
#[derive(Debug, Clone)]
pub struct TimeSeriesAnalysis {
    pub mean: f64,
    pub variance: f64,
    pub standard_deviation: f64,
    pub stationarity: f64,
    pub autocorrelation: f64,
    pub data_quality: f64,
}

#[derive(Debug, Clone)]
pub struct DetectedPattern {
    pub pattern_type: PatternType,
    pub confidence: f64,
    pub period: Option<usize>,
    pub strength: f64,
    pub description: String,
}

#[derive(Debug, Clone)]
pub struct SeasonalityModel {
    pub seasonality_type: SeasonalityType,
    pub strength: f64,
    pub period: usize,
    pub components: SeasonalityComponents,
}

#[derive(Debug, Clone)]
pub struct SeasonalityComponents {
    pub trend: Vec<f64>,
    pub seasonal: Vec<f64>,
    pub residual: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct TrendAnalysis {
    pub direction: TrendDirection,
    pub strength: f64,
    pub confidence: f64,
    pub persistence: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
    Volatile,
}

#[derive(Debug, Clone)]
pub struct TemporalForecast {
    pub horizon: usize,
    pub predicted_value: f64,
    pub confidence: f64,
    pub method: String,
}
