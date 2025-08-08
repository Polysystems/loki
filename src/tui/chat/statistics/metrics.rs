//! Chat metrics collection and calculation

use std::collections::HashMap;
use chrono::{DateTime, Utc, Duration};
use serde::{Serialize, Deserialize};

use crate::tui::chat::ChatState;
use crate::tui::run::AssistantResponseType;

/// Time range for metrics calculation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TimeRange {
    LastHour,
    Today,
    LastWeek,
    LastMonth,
    AllTime,
    Custom { start: DateTime<Utc>, end: DateTime<Utc> },
}

impl TimeRange {
    /// Get the start time for this range
    pub fn start_time(&self) -> DateTime<Utc> {
        match self {
            Self::LastHour => Utc::now() - Duration::hours(1),
            Self::Today => Utc::now().date_naive().and_hms_opt(0, 0, 0).unwrap().and_utc(),
            Self::LastWeek => Utc::now() - Duration::weeks(1),
            Self::LastMonth => Utc::now() - Duration::days(30),
            Self::AllTime => DateTime::from_timestamp(0, 0).unwrap(),
            Self::Custom { start, .. } => *start,
        }
    }
    
    /// Get the end time for this range
    pub fn end_time(&self) -> DateTime<Utc> {
        match self {
            Self::Custom { end, .. } => *end,
            _ => Utc::now(),
        }
    }
}

/// Types of metrics we can calculate
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum MetricType {
    MessageCount,
    TokenCount,
    ResponseTime,
    ErrorRate,
    ModelUsage,
    TopicDistribution,
    UserActivity,
    ConversationLength,
}

/// Collected chat metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMetrics {
    /// Total number of messages
    pub total_messages: usize,
    
    /// Messages by type
    pub messages_by_type: HashMap<String, usize>,
    
    /// Total tokens used
    pub total_tokens: usize,
    
    /// Average response time (ms)
    pub avg_response_time: f64,
    
    /// Error count
    pub error_count: usize,
    
    /// Model usage statistics
    pub model_usage: HashMap<String, usize>,
    
    /// Topic distribution
    pub topic_distribution: HashMap<String, usize>,
    
    /// Messages per hour
    pub hourly_activity: Vec<(String, usize)>,
    
    /// Average conversation length
    pub avg_conversation_length: f64,
    
    /// Most active times
    pub peak_hours: Vec<u32>,
    
    /// Response quality metrics
    pub quality_metrics: QualityMetrics,
    
    /// Time range for these metrics
    pub time_range: String,
}

/// Quality metrics for responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Average confidence score
    pub avg_confidence: f64,
    
    /// Successful completions
    pub success_rate: f64,
    
    /// Regeneration rate
    pub regeneration_rate: f64,
    
    /// Average satisfaction (if tracked)
    pub avg_satisfaction: Option<f64>,
}

/// Metrics calculator
pub struct MetricsCalculator;

impl MetricsCalculator {
    /// Calculate metrics for a given chat state and time range
    pub fn calculate(state: &ChatState, time_range: TimeRange) -> ChatMetrics {
        let start = time_range.start_time();
        let end = time_range.end_time();
        
        // Filter messages by time range
        let filtered_messages: Vec<_> = state.messages.iter()
            .filter(|msg| {
                if let AssistantResponseType::Message { timestamp, .. } = msg {
                    let msg_time = DateTime::parse_from_rfc3339(timestamp)
                        .map(|dt| dt.with_timezone(&Utc))
                        .unwrap_or_else(|_| Utc::now());
                    msg_time >= start && msg_time <= end
                } else {
                    false
                }
            })
            .collect();
        
        // Calculate various metrics
        let total_messages = filtered_messages.len();
        let messages_by_type = Self::count_messages_by_type(&filtered_messages);
        let total_tokens = Self::calculate_total_tokens(&filtered_messages);
        let avg_response_time = Self::calculate_avg_response_time(&filtered_messages);
        let error_count = Self::count_errors(&filtered_messages);
        let model_usage = Self::analyze_model_usage(&filtered_messages);
        let topic_distribution = Self::analyze_topics(&filtered_messages);
        let hourly_activity = Self::calculate_hourly_activity(&filtered_messages);
        let avg_conversation_length = Self::calculate_avg_conversation_length(&filtered_messages);
        let peak_hours = Self::find_peak_hours(&hourly_activity);
        let quality_metrics = Self::calculate_quality_metrics(&filtered_messages);
        
        ChatMetrics {
            total_messages,
            messages_by_type,
            total_tokens,
            avg_response_time,
            error_count,
            model_usage,
            topic_distribution,
            hourly_activity,
            avg_conversation_length,
            peak_hours,
            quality_metrics,
            time_range: format!("{:?}", time_range),
        }
    }
    
    /// Count messages by type
    fn count_messages_by_type(messages: &[&AssistantResponseType]) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        
        for msg in messages {
            let type_name = match msg {
                AssistantResponseType::Message { author, .. } => {
                    if author == "You" {
                        "User"
                    } else {
                        "Assistant"
                    }
                }
                AssistantResponseType::Action { .. } => "Action",
                AssistantResponseType::Code { .. } => "Code",
                AssistantResponseType::Error { .. } => "Error",
                _ => "Other",
            };
            
            *counts.entry(type_name.to_string()).or_insert(0) += 1;
        }
        
        counts
    }
    
    /// Calculate total tokens used
    fn calculate_total_tokens(messages: &[&AssistantResponseType]) -> usize {
        messages.iter()
            .filter_map(|msg| {
                if let AssistantResponseType::Message { metadata, .. } = msg {
                    metadata.tokens_used.map(|t| t as usize)
                } else {
                    None
                }
            })
            .sum()
    }
    
    /// Calculate average response time
    fn calculate_avg_response_time(messages: &[&AssistantResponseType]) -> f64 {
        let response_times: Vec<u64> = messages.iter()
            .filter_map(|msg| {
                if let AssistantResponseType::Message { metadata, .. } = msg {
                    metadata.generation_time_ms
                } else {
                    None
                }
            })
            .collect();
        
        if response_times.is_empty() {
            0.0
        } else {
            response_times.iter().sum::<u64>() as f64 / response_times.len() as f64
        }
    }
    
    /// Count error messages
    fn count_errors(messages: &[&AssistantResponseType]) -> usize {
        messages.iter()
            .filter(|msg| matches!(msg, AssistantResponseType::Error { .. }))
            .count()
    }
    
    /// Analyze model usage
    fn analyze_model_usage(messages: &[&AssistantResponseType]) -> HashMap<String, usize> {
        let mut usage = HashMap::new();
        
        for msg in messages {
            if let AssistantResponseType::Message { metadata, .. } = msg {
                if let Some(model) = metadata.model_used.as_ref() {
                    *usage.entry(model.clone()).or_insert(0) += 1;
                }
            }
        }
        
        usage
    }
    
    /// Analyze topic distribution
    fn analyze_topics(messages: &[&AssistantResponseType]) -> HashMap<String, usize> {
        let mut topics = HashMap::new();
        
        // Simple keyword-based topic detection
        let topic_keywords = vec![
            ("Code", vec!["code", "function", "class", "debug", "error"]),
            ("Data", vec!["data", "database", "query", "sql", "analysis"]),
            ("AI/ML", vec!["ai", "ml", "model", "training", "neural"]),
            ("System", vec!["system", "server", "deploy", "config", "setup"]),
            ("Design", vec!["design", "architecture", "pattern", "structure"]),
        ];
        
        for msg in messages {
            if let AssistantResponseType::Message { message, .. } = msg {
                let lower = message.to_lowercase();
                
                for (topic, keywords) in &topic_keywords {
                    if keywords.iter().any(|kw| lower.contains(kw)) {
                        *topics.entry(topic.to_string()).or_insert(0) += 1;
                    }
                }
            }
        }
        
        topics
    }
    
    /// Calculate hourly activity
    fn calculate_hourly_activity(messages: &[&AssistantResponseType]) -> Vec<(String, usize)> {
        let mut hourly = HashMap::new();
        
        for msg in messages {
            if let AssistantResponseType::Message { timestamp, .. } = msg {
                if let Ok(dt) = DateTime::parse_from_rfc3339(timestamp) {
                    use chrono::Timelike;
                    let hour = dt.hour();
                    *hourly.entry(hour).or_insert(0) += 1;
                }
            }
        }
        
        // Convert to sorted vec
        let mut activity: Vec<_> = hourly.into_iter()
            .map(|(hour, count)| (format!("{:02}:00", hour), count))
            .collect();
        activity.sort_by_key(|&(ref h, _)| h.clone());
        
        activity
    }
    
    /// Calculate average conversation length
    fn calculate_avg_conversation_length(messages: &[&AssistantResponseType]) -> f64 {
        if messages.is_empty() {
            return 0.0;
        }
        
        // Simple heuristic: count user-assistant exchanges
        let mut conversation_count = 0;
        let mut current_length = 0;
        let mut lengths = Vec::new();
        let mut last_was_user = false;
        
        for msg in messages {
            if let AssistantResponseType::Message { author, .. } = msg {
                let is_user = author == "You";
                
                if is_user && !last_was_user {
                    current_length += 1;
                } else if !is_user && last_was_user {
                    current_length += 1;
                } else if current_length > 0 {
                    // Same author twice, end conversation
                    lengths.push(current_length);
                    current_length = 1;
                    conversation_count += 1;
                }
                
                last_was_user = is_user;
            }
        }
        
        if current_length > 0 {
            lengths.push(current_length);
        }
        
        if lengths.is_empty() {
            0.0
        } else {
            lengths.iter().sum::<usize>() as f64 / lengths.len() as f64
        }
    }
    
    /// Find peak activity hours
    fn find_peak_hours(hourly_activity: &[(String, usize)]) -> Vec<u32> {
        let mut hours_with_counts: Vec<_> = hourly_activity.iter()
            .filter_map(|(hour_str, count)| {
                hour_str[..2].parse::<u32>().ok().map(|h| (h, *count))
            })
            .collect();
        
        hours_with_counts.sort_by_key(|&(_, count)| std::cmp::Reverse(count));
        
        hours_with_counts.into_iter()
            .take(3)
            .map(|(hour, _)| hour)
            .collect()
    }
    
    /// Calculate quality metrics
    fn calculate_quality_metrics(messages: &[&AssistantResponseType]) -> QualityMetrics {
        let mut total_confidence = 0.0;
        let mut confidence_count = 0;
        let mut success_count = 0;
        let mut total_responses = 0;
        
        for msg in messages {
            if let AssistantResponseType::Message { author, metadata, .. } = msg {
                if author != "You" {
                    total_responses += 1;
                    
                    if let Some(conf) = metadata.confidence_score {
                        total_confidence += conf;
                        confidence_count += 1;
                    }
                    
                    // Consider it successful if not followed by an error
                    success_count += 1;
                }
            } else if matches!(msg, AssistantResponseType::Error { .. }) {
                if success_count > 0 {
                    success_count -= 1;
                }
            }
        }
        
        QualityMetrics {
            avg_confidence: if confidence_count > 0 {
                total_confidence / confidence_count as f32
            } else {
                0.0
            } as f64,
            success_rate: if total_responses > 0 {
                success_count as f64 / total_responses as f64
            } else {
                0.0
            },
            regeneration_rate: 0.0, // Would need to track regenerations
            avg_satisfaction: None, // Would need user feedback
        }
    }
}