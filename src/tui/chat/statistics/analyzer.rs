//! Advanced chat analysis functionality

use std::collections::HashMap;
use chrono::{DateTime, Utc};
use serde::{Serialize, Deserialize};
use anyhow::Result;

use crate::tui::chat::ChatState;
use crate::tui::run::AssistantResponseType;

/// Chat analyzer for advanced insights
pub struct ChatAnalyzer;

/// Analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResult {
    /// Conversation patterns
    pub patterns: ConversationPatterns,
    
    /// User behavior insights
    pub user_behavior: UserBehavior,
    
    /// AI performance metrics
    pub ai_performance: AIPerformance,
    
    /// Recommendations
    pub recommendations: Vec<Recommendation>,
}

/// Conversation patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationPatterns {
    /// Average questions per conversation
    pub avg_questions_per_conversation: f64,
    
    /// Common conversation starters
    pub common_starters: Vec<(String, usize)>,
    
    /// Conversation flow patterns
    pub flow_patterns: Vec<FlowPattern>,
    
    /// Topic transitions
    pub topic_transitions: HashMap<String, Vec<String>>,
}

/// User behavior insights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserBehavior {
    /// Typing speed (chars per minute)
    pub avg_typing_speed: f64,
    
    /// Message length distribution
    pub message_length_stats: MessageLengthStats,
    
    /// Interaction patterns
    pub interaction_patterns: Vec<InteractionPattern>,
    
    /// Preferred communication style
    pub communication_style: CommunicationStyle,
}

/// AI performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIPerformance {
    /// Response quality score
    pub quality_score: f64,
    
    /// Helpfulness rating
    pub helpfulness_rating: f64,
    
    /// Context retention score
    pub context_retention: f64,
    
    /// Task completion rate
    pub task_completion_rate: f64,
}

/// Flow pattern in conversations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowPattern {
    pub pattern_type: String,
    pub frequency: usize,
    pub avg_duration: f64,
}

/// Message length statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageLengthStats {
    pub min: usize,
    pub max: usize,
    pub avg: f64,
    pub median: f64,
}

/// Interaction pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionPattern {
    pub pattern_name: String,
    pub occurrences: usize,
    pub description: String,
}

/// Communication style
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CommunicationStyle {
    Concise,
    Detailed,
    Technical,
    Conversational,
    Mixed,
}

/// Recommendation for improvement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    pub category: String,
    pub suggestion: String,
    pub priority: Priority,
    pub impact: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Priority {
    High,
    Medium,
    Low,
}

impl ChatAnalyzer {
    /// Perform comprehensive analysis
    pub fn analyze(state: &ChatState) -> Result<AnalysisResult> {
        let patterns = Self::analyze_patterns(state);
        let user_behavior = Self::analyze_user_behavior(state);
        let ai_performance = Self::analyze_ai_performance(state);
        let recommendations = Self::generate_recommendations(&patterns, &user_behavior, &ai_performance);
        
        Ok(AnalysisResult {
            patterns,
            user_behavior,
            ai_performance,
            recommendations,
        })
    }
    
    /// Analyze conversation patterns
    fn analyze_patterns(state: &ChatState) -> ConversationPatterns {
        let messages = &state.messages;
        
        // Find conversation boundaries
        let mut conversations = Vec::new();
        let mut current_conv = Vec::new();
        let mut last_time = None;
        
        for msg in messages {
            if let AssistantResponseType::Message { timestamp, .. } = msg {
                let msg_time = DateTime::parse_from_rfc3339(timestamp)
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now());
                
                // New conversation if gap > 30 minutes
                if let Some(last) = last_time {
                    if msg_time.signed_duration_since(last).num_minutes() > 30 {
                        if !current_conv.is_empty() {
                            conversations.push(current_conv);
                            current_conv = Vec::new();
                        }
                    }
                }
                
                current_conv.push(msg);
                last_time = Some(msg_time);
            }
        }
        
        if !current_conv.is_empty() {
            conversations.push(current_conv);
        }
        
        // Calculate metrics
        let total_questions = messages.iter()
            .filter(|msg| {
                if let AssistantResponseType::Message { author, message, .. } = msg {
                    author == "You" && message.contains('?')
                } else {
                    false
                }
            })
            .count();
        
        let avg_questions = if !conversations.is_empty() {
            total_questions as f64 / conversations.len() as f64
        } else {
            0.0
        };
        
        // Find common starters
        let mut starters = HashMap::new();
        for conv in &conversations {
            if let Some(AssistantResponseType::Message { message, author, .. }) = conv.first() {
                if author == "You" {
                    let words: Vec<_> = message.split_whitespace().take(3).collect();
                    let starter = words.join(" ");
                    *starters.entry(starter).or_insert(0) += 1;
                }
            }
        }
        
        let mut common_starters: Vec<_> = starters.into_iter().collect();
        common_starters.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
        common_starters.truncate(5);
        
        // Analyze flow patterns
        let flow_patterns = vec![
            FlowPattern {
                pattern_type: "Question-Answer".to_string(),
                frequency: conversations.iter()
                    .filter(|conv| conv.len() == 2)
                    .count(),
                avg_duration: 2.0,
            },
            FlowPattern {
                pattern_type: "Multi-turn Discussion".to_string(),
                frequency: conversations.iter()
                    .filter(|conv| conv.len() > 4)
                    .count(),
                avg_duration: conversations.iter()
                    .filter(|conv| conv.len() > 4)
                    .map(|conv| conv.len() as f64)
                    .sum::<f64>() / conversations.iter().filter(|conv| conv.len() > 4).count().max(1) as f64,
            },
        ];
        
        ConversationPatterns {
            avg_questions_per_conversation: avg_questions,
            common_starters,
            flow_patterns,
            topic_transitions: HashMap::new(), // Would need more sophisticated topic modeling
        }
    }
    
    /// Analyze user behavior
    fn analyze_user_behavior(state: &ChatState) -> UserBehavior {
        let user_messages: Vec<_> = state.messages.iter()
            .filter_map(|msg| {
                if let AssistantResponseType::Message { author, message, .. } = msg {
                    if author == "You" {
                        Some(message)
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect();
        
        // Message length statistics
        let lengths: Vec<_> = user_messages.iter().map(|m| m.len()).collect();
        let min = lengths.iter().min().copied().unwrap_or(0);
        let max = lengths.iter().max().copied().unwrap_or(0);
        let avg = if !lengths.is_empty() {
            lengths.iter().sum::<usize>() as f64 / lengths.len() as f64
        } else {
            0.0
        };
        
        let mut sorted_lengths = lengths.clone();
        sorted_lengths.sort_unstable();
        let median = if !sorted_lengths.is_empty() {
            sorted_lengths[sorted_lengths.len() / 2] as f64
        } else {
            0.0
        };
        
        let message_length_stats = MessageLengthStats {
            min,
            max,
            avg,
            median,
        };
        
        // Determine communication style
        let communication_style = if avg < 20.0 {
            CommunicationStyle::Concise
        } else if avg > 100.0 {
            CommunicationStyle::Detailed
        } else if user_messages.iter().any(|m| m.contains("function") || m.contains("code") || m.contains("error")) {
            CommunicationStyle::Technical
        } else {
            CommunicationStyle::Conversational
        };
        
        // Interaction patterns
        let mut patterns = Vec::new();
        
        if user_messages.iter().filter(|m| m.contains('?')).count() > user_messages.len() / 2 {
            patterns.push(InteractionPattern {
                pattern_name: "Question-Heavy".to_string(),
                occurrences: user_messages.iter().filter(|m| m.contains('?')).count(),
                description: "Frequently asks questions".to_string(),
            });
        }
        
        if user_messages.iter().any(|m| m.contains("please") || m.contains("thank")) {
            patterns.push(InteractionPattern {
                pattern_name: "Polite".to_string(),
                occurrences: user_messages.iter()
                    .filter(|m| m.contains("please") || m.contains("thank"))
                    .count(),
                description: "Uses polite language".to_string(),
            });
        }
        
        UserBehavior {
            avg_typing_speed: 60.0, // Would need timing data to calculate
            message_length_stats,
            interaction_patterns: patterns,
            communication_style,
        }
    }
    
    /// Analyze AI performance
    fn analyze_ai_performance(state: &ChatState) -> AIPerformance {
        let ai_messages: Vec<_> = state.messages.iter()
            .filter(|msg| {
                if let AssistantResponseType::Message { author, .. } = msg {
                    author != "You"
                } else {
                    false
                }
            })
            .collect();
        
        // Calculate metrics
        let total_responses = ai_messages.len();
        let errors = state.messages.iter()
            .filter(|msg| matches!(msg, AssistantResponseType::Error { .. }))
            .count();
        
        let quality_score = if total_responses > 0 {
            ((total_responses - errors) as f64 / total_responses as f64) * 100.0
        } else {
            0.0
        };
        
        // These would need more sophisticated analysis
        let helpfulness_rating = 85.0;
        let context_retention = 90.0;
        let task_completion_rate = 88.0;
        
        AIPerformance {
            quality_score,
            helpfulness_rating,
            context_retention,
            task_completion_rate,
        }
    }
    
    /// Generate recommendations
    fn generate_recommendations(
        patterns: &ConversationPatterns,
        behavior: &UserBehavior,
        performance: &AIPerformance,
    ) -> Vec<Recommendation> {
        let mut recommendations = Vec::new();
        
        // Based on patterns
        if patterns.avg_questions_per_conversation > 5.0 {
            recommendations.push(Recommendation {
                category: "Efficiency".to_string(),
                suggestion: "Consider asking compound questions to reduce back-and-forth".to_string(),
                priority: Priority::Medium,
                impact: "Faster problem resolution".to_string(),
            });
        }
        
        // Based on user behavior
        if let CommunicationStyle::Concise = behavior.communication_style {
            recommendations.push(Recommendation {
                category: "Communication".to_string(),
                suggestion: "Providing more context in questions can lead to better responses".to_string(),
                priority: Priority::Low,
                impact: "More accurate and relevant answers".to_string(),
            });
        }
        
        // Based on AI performance
        if performance.quality_score < 90.0 {
            recommendations.push(Recommendation {
                category: "Model Selection".to_string(),
                suggestion: "Consider using a more powerful model for complex tasks".to_string(),
                priority: Priority::High,
                impact: "Improved response quality".to_string(),
            });
        }
        
        recommendations
    }
}