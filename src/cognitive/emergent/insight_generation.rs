//! Spontaneous Insight Generation System
//!
//! This module implements the spontaneous insight generation capabilities that allow
//! novel insights to emerge from complex interactions between cognitive subsystems.

use anyhow::Result;
use chrono::{DateTime, Utc, Timelike};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::memory::CognitiveMemory;
use super::{
    EmergentPatternId, EmergenceType, EmergenceLevel, CognitiveDomain, ComplexityMetrics,
    EmergentPattern, EvidenceType, EmergentIntelligenceConfig,
    EmergentAnalysis,
    EmergenceTrigger, TriggerType, TriggerCondition, ComparisonOperator,
};

/// Spontaneous insight generation system
pub struct SpontaneousInsightGenerator {
    /// Configuration
    #[allow(dead_code)]
    config: EmergentIntelligenceConfig,

    /// Memory system reference
    memory: Arc<CognitiveMemory>,

    /// Active insight generation sessions
    active_sessions: Arc<RwLock<HashMap<String, InsightGenerationSession>>>,

    /// Generated insights history
    insights_history: Arc<RwLock<VecDeque<GeneratedInsight>>>,

    /// Cross-domain connection tracker
    connection_tracker: Arc<RwLock<CrossDomainConnectionTracker>>,

    /// Insight patterns discovered
    insight_patterns: Arc<RwLock<HashMap<EmergentPatternId, InsightPattern>>>,
}

/// A session of insight generation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InsightGenerationSession {
    /// Session identifier
    pub session_id: String,

    /// Start time
    pub start_time: DateTime<Utc>,

    /// Cognitive domains involved
    pub domains: HashSet<CognitiveDomain>,

    /// Input context for insight generation
    pub input_context: InsightContext,

    /// Generated insights in this session
    pub generated_insights: Vec<GeneratedInsight>,

    /// Session status
    pub status: InsightSessionStatus,

    /// Complexity level of session
    pub complexity_level: f64,
}

/// Context for insight generation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InsightContext {
    /// Problem or question being explored
    pub focus_question: String,

    /// Available information domains
    pub information_domains: Vec<String>,

    /// Current cognitive state
    pub cognitive_state: String,

    /// Active patterns or themes
    pub active_patterns: Vec<String>,

    /// Constraints or requirements
    pub constraints: Vec<String>,

    /// Expected insight types
    pub expected_types: Vec<InsightType>,
}

/// Status of insight generation session
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum InsightSessionStatus {
    /// Session is initializing
    Initializing,
    /// Actively generating insights
    Generating,
    /// Evaluating generated insights
    Evaluating,
    /// Session completed successfully
    Completed,
    /// Session failed to generate insights
    Failed,
    /// Session was interrupted
    Interrupted,
}

/// A generated insight
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GeneratedInsight {
    /// Insight identifier
    pub insight_id: String,

    /// Insight description
    pub description: String,

    /// Type of insight
    pub insight_type: InsightType,

    /// Cognitive domains that contributed
    pub contributing_domains: Vec<CognitiveDomain>,

    /// Novelty score (0.0 to 1.0)
    pub novelty: f64,

    /// Confidence in insight validity
    pub confidence: f64,

    /// Potential impact or importance
    pub impact: f64,

    /// Evidence supporting the insight
    pub supporting_evidence: Vec<InsightEvidence>,

    /// Connections to existing knowledge
    pub knowledge_connections: Vec<KnowledgeConnection>,

    /// Implications of the insight
    pub implications: Vec<InsightImplication>,

    /// Generation timestamp
    pub generated_at: DateTime<Utc>,

    /// Validation status
    pub validation_status: ValidationStatus,
}

/// Types of insights that can be generated
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum InsightType {
    /// Novel connections between concepts
    ConceptualConnection,
    /// New problem-solving approaches
    MethodologicalInnovation,
    /// Pattern recognition breakthrough
    PatternRecognition,
    /// Analogical reasoning insight
    AnalogicalInsight,
    /// Causal relationship discovery
    CausalDiscovery,
    /// Synthesis of disparate information
    SynthesisInsight,
    /// Meta-cognitive realization
    MetaCognitiveInsight,
    /// Creative breakthrough
    CreativeBreakthrough,
    /// Architectural understanding
    ArchitecturalInsight,
    /// Predictive insight
    PredictiveInsight,
}

/// Evidence supporting an insight
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InsightEvidence {
    /// Evidence description
    pub description: String,

    /// Source of evidence
    pub source: String,

    /// Evidence strength
    pub strength: f64,

    /// Type of evidence
    pub evidence_type: EvidenceType,

    /// Verification status
    pub verified: bool,
}

/// Connection to existing knowledge
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KnowledgeConnection {
    /// Connected knowledge item
    pub knowledge_item: String,

    /// Type of connection
    pub connection_type: ConnectionType,

    /// Strength of connection
    pub strength: f64,

    /// Explanation of connection
    pub explanation: String,
}

/// Types of knowledge connections
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ConnectionType {
    /// Direct logical connection
    Logical,
    /// Analogical connection
    Analogical,
    /// Causal relationship
    Causal,
    /// Temporal relationship
    Temporal,
    /// Structural similarity
    Structural,
    /// Functional similarity
    Functional,
    /// Thematic connection
    Thematic,
    /// Contradictory relationship
    Contradictory,
}

/// Implication of an insight
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InsightImplication {
    /// Implication description
    pub description: String,

    /// Domain affected
    pub affected_domain: CognitiveDomain,

    /// Likelihood of implication
    pub likelihood: f64,

    /// Potential impact
    pub impact: f64,

    /// Required actions
    pub required_actions: Vec<String>,
}

/// Validation status of insights
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ValidationStatus {
    /// Not yet validated
    Pending,
    /// Insight appears valid
    Validated,
    /// Insight needs refinement
    NeedsRefinement,
    /// Insight is invalid
    Invalid,
    /// Validation inconclusive
    Inconclusive,
}

/// Tracks cross-domain connections for insight generation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CrossDomainConnectionTracker {
    /// Active connections between domains
    pub active_connections: HashMap<DomainPair, ConnectionStrength>,

    /// Connection patterns discovered
    pub connection_patterns: Vec<ConnectionPattern>,

    /// Cross-domain interaction history
    pub interaction_history: VecDeque<CrossDomainInteraction>,

    /// Synchronization levels between domains
    pub synchronization_levels: HashMap<DomainPair, f64>,
}

/// Pair of cognitive domains
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct DomainPair {
    pub domain1: CognitiveDomain,
    pub domain2: CognitiveDomain,
}

/// Connection strength between domains
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConnectionStrength {
    /// Strength value (0.0 to 1.0)
    pub strength: f64,

    /// Frequency of connection
    pub frequency: f64,

    /// Last interaction time
    pub last_interaction: DateTime<Utc>,

    /// Connection quality
    pub quality: f64,
}

/// Pattern in cross-domain connections
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConnectionPattern {
    /// Pattern identifier
    pub pattern_id: String,

    /// Domains involved
    pub domains: Vec<CognitiveDomain>,

    /// Pattern description
    pub description: String,

    /// Pattern frequency
    pub frequency: f64,

    /// Effectiveness for insight generation
    pub effectiveness: f64,

    /// Trigger conditions
    pub triggers: Vec<String>,
}

/// Interaction between cognitive domains
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CrossDomainInteraction {
    /// Timestamp of interaction
    pub timestamp: DateTime<Utc>,

    /// Domains involved
    pub domains: Vec<CognitiveDomain>,

    /// Interaction type
    pub interaction_type: String,

    /// Data exchanged
    pub data_summary: String,

    /// Outcome of interaction
    pub outcome: InteractionOutcome,

    /// Generated insights from interaction
    pub insights_generated: Vec<String>,
}

/// Outcome of domain interactions
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum InteractionOutcome {
    /// New insight generated
    InsightGenerated,
    /// Knowledge synthesis occurred
    KnowledgeSynthesis,
    /// Pattern recognized
    PatternRecognized,
    /// Connection strengthened
    ConnectionStrengthened,
    /// No significant outcome
    NoSignificantOutcome,
    /// Unexpected result
    UnexpectedResult,
}

/// Insight pattern discovered
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InsightPattern {
    /// Base pattern information
    pub base_pattern: EmergentPattern,

    /// Insight generation triggers
    pub generation_triggers: Vec<EmergenceTrigger>,

    /// Success patterns for insight types
    pub success_patterns: HashMap<InsightType, f64>,

    /// Optimal domain combinations
    pub optimal_combinations: Vec<DomainCombination>,

    /// Pattern effectiveness metrics
    pub effectiveness_metrics: InsightPatternMetrics,
}

/// Combination of domains for insight generation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DomainCombination {
    /// Domains in combination
    pub domains: Vec<CognitiveDomain>,

    /// Synergy score
    pub synergy_score: f64,

    /// Best insight types for this combination
    pub best_insight_types: Vec<InsightType>,

    /// Optimal conditions
    pub optimal_conditions: Vec<String>,
}

/// State of a cognitive domain
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DomainState {
    /// Domain identifier
    pub domain: CognitiveDomain,

    /// Current activity level
    pub activity_level: f64,

    /// Current focus areas
    pub focus_areas: Vec<String>,

    /// Processing load
    pub processing_load: f64,

    /// State quality indicators
    pub quality_indicators: HashMap<String, f64>,
}

/// Metrics for insight pattern effectiveness
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InsightPatternMetrics {
    /// Average novelty of generated insights
    pub average_novelty: f64,

    /// Average confidence in insights
    pub average_confidence: f64,

    /// Success rate of insight validation
    pub validation_success_rate: f64,

    /// Impact score of insights
    pub average_impact: f64,

    /// Generation efficiency
    pub generation_efficiency: f64,
}

impl SpontaneousInsightGenerator {
    /// Create a new spontaneous insight generator
    pub async fn new(
        config: EmergentIntelligenceConfig,
        memory: Arc<CognitiveMemory>,
    ) -> Result<Self> {
        Ok(Self {
            config,
            memory,
            active_sessions: Arc::new(RwLock::new(HashMap::new())),
            insights_history: Arc::new(RwLock::new(VecDeque::new())),
            connection_tracker: Arc::new(RwLock::new(CrossDomainConnectionTracker {
                active_connections: HashMap::new(),
                connection_patterns: Vec::new(),
                interaction_history: VecDeque::new(),
                synchronization_levels: HashMap::new(),
            })),
            insight_patterns: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// **Generate insights from cross-domain interactions** - Core insight generation
    pub async fn generate_insights(
        &self,
        context: InsightContext,
    ) -> Result<Vec<GeneratedInsight>> {
        let session_id = uuid::Uuid::new_v4().to_string();

        let mut session = InsightGenerationSession {
            session_id: session_id.clone(),
            start_time: Utc::now(),
            domains: self.extract_domains_from_context(&context).await?,
            input_context: context.clone(),
            generated_insights: Vec::new(),
            status: InsightSessionStatus::Initializing,
            complexity_level: self.assess_context_complexity(&context).await?,
        };

        // Update session status
        session.status = InsightSessionStatus::Generating;

        // Generate insights based on different approaches
        let mut insights = Vec::new();

        // Cross-domain synthesis insights
        insights.extend(self.generate_cross_domain_insights(&context).await?);

        // Pattern recognition insights
        insights.extend(self.generate_pattern_insights(&context).await?);

        // Analogical insights
        insights.extend(self.generate_analogical_insights(&context).await?);

        // Meta-cognitive insights
        insights.extend(self.generate_meta_cognitive_insights(&context).await?);

        // Evaluate and rank insights
        session.status = InsightSessionStatus::Evaluating;
        let evaluated_insights = self.evaluate_insights(insights).await?;

        session.generated_insights = evaluated_insights.clone();
        session.status = InsightSessionStatus::Completed;

        // Store session
        {
            let mut sessions = self.active_sessions.write().await;
            sessions.insert(session_id, session);
        }

        // Update insights history
        {
            let mut history = self.insights_history.write().await;
            for insight in &evaluated_insights {
                history.push_back(insight.clone());
            }

            // Keep only recent insights
            while history.len() > 1000 {
                history.pop_front();
            }
        }

        tracing::info!("Generated {} insights from context: {}", evaluated_insights.len(), context.focus_question);

        Ok(evaluated_insights)
    }

    /// **Detect emergent insight patterns** - Pattern discovery in insights
    pub async fn detect_insight_patterns(&self) -> Result<Vec<InsightPattern>> {
        let history = self.insights_history.read().await;
        let mut patterns = Vec::new();

        // Analyze insight generation patterns
        patterns.extend(self.analyze_insight_type_patterns(&history).await?);
        patterns.extend(self.analyze_domain_combination_patterns(&history).await?);
        patterns.extend(self.analyze_temporal_patterns(&history).await?);

        Ok(patterns)
    }

    /// **Monitor cross-domain connections** - Track domain interaction patterns
    pub async fn monitor_cross_domain_connections(&self) -> Result<CrossDomainConnectionTracker> {
        Ok(self.connection_tracker.read().await.clone())
    }

    /// Get insights history
    pub async fn get_insights_history(&self) -> Result<Vec<GeneratedInsight>> {
        Ok(self.insights_history.read().await.iter().cloned().collect())
    }

    // Private helper methods...

    async fn extract_domains_from_context(&self, context: &InsightContext) -> Result<HashSet<CognitiveDomain>> {
        // Simplified domain extraction
        let mut domains = HashSet::new();

        // Extract domains based on context content
        if context.focus_question.to_lowercase().contains("creative") {
            domains.insert(CognitiveDomain::Creativity);
        }
        if context.focus_question.to_lowercase().contains("memory") {
            domains.insert(CognitiveDomain::Memory);
        }
        if context.focus_question.to_lowercase().contains("reasoning") {
            domains.insert(CognitiveDomain::Reasoning);
        }

        // Always include self-reflection for insight generation
        domains.insert(CognitiveDomain::SelfReflection);

        Ok(domains)
    }

    async fn assess_context_complexity(&self, context: &InsightContext) -> Result<f64> {
        let question_length = context.focus_question.len() as f64;
        let domain_count = context.information_domains.len() as f64;
        let pattern_count = context.active_patterns.len() as f64;
        let constraint_count = context.constraints.len() as f64;

        Ok((question_length / 100.0 + domain_count / 10.0 + pattern_count / 5.0 + constraint_count / 3.0).min(1.0))
    }

    async fn generate_cross_domain_insights(&self, context: &InsightContext) -> Result<Vec<GeneratedInsight>> {
        let mut insights = Vec::new();

        // Generate insights from domain combinations
        let insight = GeneratedInsight {
            insight_id: uuid::Uuid::new_v4().to_string(),
            description: format!("Cross-domain insight for: {}", context.focus_question),
            insight_type: InsightType::SynthesisInsight,
            contributing_domains: vec![CognitiveDomain::Reasoning, CognitiveDomain::Creativity],
            novelty: 0.8,
            confidence: 0.7,
            impact: 0.75,
            supporting_evidence: Vec::new(),
            knowledge_connections: Vec::new(),
            implications: Vec::new(),
            generated_at: Utc::now(),
            validation_status: ValidationStatus::Pending,
        };

        insights.push(insight);
        Ok(insights)
    }

    async fn generate_pattern_insights(&self, context: &InsightContext) -> Result<Vec<GeneratedInsight>> {
        let mut insights = Vec::new();

        if !context.active_patterns.is_empty() {
            let insight = GeneratedInsight {
                insight_id: uuid::Uuid::new_v4().to_string(),
                description: format!("Pattern recognition insight from active patterns: {:?}", context.active_patterns),
                insight_type: InsightType::PatternRecognition,
                contributing_domains: vec![CognitiveDomain::Reasoning],
                novelty: 0.6,
                confidence: 0.8,
                impact: 0.7,
                supporting_evidence: Vec::new(),
                knowledge_connections: Vec::new(),
                implications: Vec::new(),
                generated_at: Utc::now(),
                validation_status: ValidationStatus::Pending,
            };

            insights.push(insight);
        }

        Ok(insights)
    }

    async fn generate_analogical_insights(&self, context: &InsightContext) -> Result<Vec<GeneratedInsight>> {
        let mut insights = Vec::new();

        let insight = GeneratedInsight {
            insight_id: uuid::Uuid::new_v4().to_string(),
            description: format!("Analogical insight based on domain similarities for: {}", context.focus_question),
            insight_type: InsightType::AnalogicalInsight,
            contributing_domains: vec![CognitiveDomain::Reasoning, CognitiveDomain::Memory],
            novelty: 0.7,
            confidence: 0.6,
            impact: 0.65,
            supporting_evidence: Vec::new(),
            knowledge_connections: Vec::new(),
            implications: Vec::new(),
            generated_at: Utc::now(),
            validation_status: ValidationStatus::Pending,
        };

        insights.push(insight);
        Ok(insights)
    }

    async fn generate_meta_cognitive_insights(&self, context: &InsightContext) -> Result<Vec<GeneratedInsight>> {
        let mut insights = Vec::new();

        let insight = GeneratedInsight {
            insight_id: uuid::Uuid::new_v4().to_string(),
            description: format!("Meta-cognitive insight about the insight generation process for: {}", context.focus_question),
            insight_type: InsightType::MetaCognitiveInsight,
            contributing_domains: vec![CognitiveDomain::SelfReflection],
            novelty: 0.9,
            confidence: 0.8,
            impact: 0.85,
            supporting_evidence: Vec::new(),
            knowledge_connections: Vec::new(),
            implications: Vec::new(),
            generated_at: Utc::now(),
            validation_status: ValidationStatus::Pending,
        };

        insights.push(insight);
        Ok(insights)
    }

    async fn evaluate_insights(&self, insights: Vec<GeneratedInsight>) -> Result<Vec<GeneratedInsight>> {
        // Sort insights by a composite score
        let mut evaluated = insights;
        evaluated.sort_by(|a, b| {
            let score_a = a.novelty * 0.4 + a.confidence * 0.3 + a.impact * 0.3;
            let score_b = b.novelty * 0.4 + b.confidence * 0.3 + b.impact * 0.3;
            score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(evaluated)
    }

    async fn analyze_insight_type_patterns(&self, history: &VecDeque<GeneratedInsight>) -> Result<Vec<InsightPattern>> {
        use std::collections::HashMap;
        use rayon::prelude::*;

        if history.is_empty() {
            return Ok(Vec::new());
        }

        // Parallelize insight type frequency analysis
        let insight_types: Vec<_> = history.iter()
            .map(|insight| &insight.insight_type)
            .collect();

        let type_frequencies = insight_types.par_iter()
            .fold(HashMap::new, |mut acc, insight_type| {
                *acc.entry(insight_type).or_insert(0) += 1;
                acc
            })
            .reduce(HashMap::new, |mut acc1, acc2| {
                for (key, value) in acc2 {
                    *acc1.entry(key).or_insert(0) += value;
                }
                acc1
            });

        // Analyze temporal patterns
        let _temporal_windows = self.analyze_temporal_insight_windows(history).await?;

        // Generate patterns based on frequency and temporal analysis
        let mut patterns = Vec::new();

        for (insight_type, frequency) in type_frequencies {
            if frequency >= 3 { // Minimum threshold for pattern detection
                let effectiveness = self.calculate_type_effectiveness(insight_type, history).await?;
                let optimal_combinations = self.find_optimal_domain_combinations_for_type(insight_type, history).await?;

                let pattern = InsightPattern {
                    base_pattern: EmergentPattern {
                        id: EmergentPatternId::from_description(&format!("type_pattern_{:?}", insight_type)),
                        emergence_type: EmergenceType::SpontaneousInsight,
                        level: EmergenceLevel::Complex,
                        domains: HashSet::from([CognitiveDomain::Reasoning, CognitiveDomain::Creativity]),
                        description: format!("Pattern for {:?} insights based on {} occurrences", insight_type, frequency),
                        analysis: EmergentAnalysis {
                            contributing_components: vec!["insight_generator".to_string()],
                            interaction_patterns: vec![],
                            complexity_metrics: ComplexityMetrics {
                                component_count: 1,
                                interaction_complexity: effectiveness,
                                entropy: 0.8,
                                organizational_complexity: 0.7,
                                dynamic_complexity: 0.6,
                                computational_complexity: 0.5,
                            },
                            triggers: vec![],
                            synchronization_levels: HashMap::new(),
                            information_flows: vec![],
                        },
                        confidence: (frequency as f64 / history.len() as f64).min(1.0),
                        importance: effectiveness,
                        novelty: 0.7,
                        discovered_at: Utc::now(),
                        evidence: vec![],
                        related_patterns: vec![],
                        implications: vec![],
                    },
                    generation_triggers: self.extract_triggers_for_type(insight_type, history).await?,
                    success_patterns: HashMap::from([((*insight_type).clone(), effectiveness)]),
                    optimal_combinations,
                    effectiveness_metrics: InsightPatternMetrics {
                        average_novelty: self.calculate_average_novelty_for_type(insight_type, history).await?,
                        average_confidence: self.calculate_average_confidence_for_type(insight_type, history).await?,
                        validation_success_rate: self.calculate_validation_success_rate_for_type(insight_type, history).await?,
                        average_impact: self.calculate_average_impact_for_type(insight_type, history).await?,
                        generation_efficiency: frequency as f64 / history.len() as f64,
                    },
                };

                patterns.push(pattern);
            }
        }

        // Sort patterns by effectiveness
        patterns.sort_by(|a, b| {
            b.effectiveness_metrics.average_impact
                .partial_cmp(&a.effectiveness_metrics.average_impact)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        tracing::info!("Analyzed {} insight type patterns from {} historical insights", patterns.len(), history.len());
        Ok(patterns)
    }

    async fn analyze_domain_combination_patterns(&self, history: &VecDeque<GeneratedInsight>) -> Result<Vec<InsightPattern>> {
        use std::collections::HashMap;
        use rayon::prelude::*;

        if history.is_empty() {
            return Ok(Vec::new());
        }

        // Extract domain combinations with parallel processing
        let domain_combinations: Vec<_> = history.par_iter()
            .map(|insight| {
                let mut domains = insight.contributing_domains.clone();
                domains.sort(); // Ensure consistent ordering
                (domains, insight.novelty * insight.confidence * insight.impact)
            })
            .collect();

        // Analyze combination effectiveness
        let combination_stats = domain_combinations.par_iter()
            .fold(HashMap::new, |mut acc, (domains, score)| {
                let key = format!("{:?}", domains);
                let entry = acc.entry(key).or_insert_with(|| Vec::new());
                entry.push(*score);
                acc
            })
            .reduce(HashMap::new, |mut acc1, acc2| {
                for (key, mut values) in acc2 {
                    acc1.entry(key).or_insert_with(Vec::new).append(&mut values);
                }
                acc1
            });

        let mut patterns = Vec::new();

        for (combination_key, scores) in combination_stats {
            if scores.len() >= 2 { // Minimum occurrences for pattern
                let average_score = scores.iter().sum::<f64>() / scores.len() as f64;
                let std_dev = {
                    let variance = scores.iter()
                        .map(|score| (score - average_score).powi(2))
                        .sum::<f64>() / scores.len() as f64;
                    variance.sqrt()
                };

                let stability = 1.0 - (std_dev / average_score).min(1.0);

                if average_score > 0.5 && stability > 0.6 { // Quality thresholds
                    let pattern = InsightPattern {
                        base_pattern: EmergentPattern {
                            id: EmergentPatternId::from_description(&format!("domain_combo_{}", combination_key.chars().filter(|c| c.is_alphanumeric()).collect::<String>())),
                            emergence_type: EmergenceType::CrossDomain,
                            level: EmergenceLevel::Complex,
                            domains: HashSet::from([CognitiveDomain::Reasoning, CognitiveDomain::Creativity]),
                            description: format!("Effective domain combination pattern: {}", combination_key),
                            analysis: EmergentAnalysis {
                                contributing_components: vec!["insight_generator".to_string()],
                                interaction_patterns: vec![],
                                complexity_metrics: ComplexityMetrics {
                                    component_count: 1,
                                    interaction_complexity: average_score,
                                    entropy: 0.8,
                                    organizational_complexity: 0.7,
                                    dynamic_complexity: 0.6,
                                    computational_complexity: 0.5,
                                },
                                triggers: vec![
                                    EmergenceTrigger {
                                        trigger_type: TriggerType::CriticalMass,
                                        description: "Multi-domain activation".to_string(),
                                        condition: TriggerCondition {
                                            metric: "domain_activity".to_string(),
                                            threshold: 0.7,
                                            operator: ComparisonOperator::GreaterThan,
                                            duration_seconds: Some(10.0),
                                        },
                                        frequency: stability,
                                    }
                                ],
                                synchronization_levels: HashMap::new(),
                                information_flows: vec![],
                            },
                            confidence: stability,
                            importance: average_score,
                            novelty: average_score * 0.8,
                            discovered_at: Utc::now(),
                            evidence: vec![],
                            related_patterns: vec![],
                            implications: vec![],
                        },
                        generation_triggers: vec![
                            EmergenceTrigger {
                                trigger_type: TriggerType::CriticalMass,
                                description: "Multi-domain activation".to_string(),
                                condition: TriggerCondition {
                                    metric: "domain_activity".to_string(),
                                    threshold: 0.7,
                                    operator: ComparisonOperator::GreaterThan,
                                    duration_seconds: Some(10.0),
                                },
                                frequency: stability,
                            }
                        ],
                        success_patterns: HashMap::new(),
                        optimal_combinations: vec![
                            DomainCombination {
                                domains: vec![], // Would parse from combination_key in production
                                synergy_score: average_score,
                                best_insight_types: vec![InsightType::SynthesisInsight],
                                optimal_conditions: vec!["High domain activity".to_string()],
                            }
                        ],
                        effectiveness_metrics: InsightPatternMetrics {
                            average_novelty: average_score * 0.8, // Estimated based on score
                            average_confidence: average_score * 0.9,
                            validation_success_rate: stability,
                            average_impact: average_score,
                            generation_efficiency: scores.len() as f64 / history.len() as f64,
                        },
                    };

                    patterns.push(pattern);
                }
            }
        }

        tracing::info!("Analyzed {} domain combination patterns", patterns.len());
        Ok(patterns)
    }

    async fn analyze_temporal_patterns(&self, history: &VecDeque<GeneratedInsight>) -> Result<Vec<InsightPattern>> {
        use std::collections::HashMap;
        use chrono::Duration;

        if history.len() < 5 {
            return Ok(Vec::new());
        }

        // Sort insights by timestamp for temporal analysis
        let mut temporal_insights: Vec<_> = history.iter().collect();
        temporal_insights.sort_by(|a, b| a.generated_at.cmp(&b.generated_at));

        // Analyze time-based patterns
        let mut time_windows = HashMap::new();
        let _window_size = Duration::hours(1);

        for insight in &temporal_insights {
            let window_start = insight.generated_at
                .with_minute(0)
                .unwrap()
                .with_second(0)
                .unwrap()
                .with_nanosecond(0)
                .unwrap();

            let entry = time_windows.entry(window_start).or_insert_with(Vec::new);
            entry.push(insight);
        }

        // Find productive time patterns
        let mut patterns = Vec::new();

        for (window_start, insights_in_window) in time_windows {
            if insights_in_window.len() >= 3 { // Minimum for temporal pattern
                let avg_quality = insights_in_window.iter()
                    .map(|i| i.novelty * i.confidence * i.impact)
                    .sum::<f64>() / insights_in_window.len() as f64;

                let hour = window_start.hour();
                let pattern_description = match hour {
                    6..=11 => "Morning productive insights",
                    12..=17 => "Afternoon analytical insights",
                    18..=23 => "Evening creative insights",
                    _ => "Late night deep insights",
                };

                if avg_quality > 0.6 {
                    let pattern = InsightPattern {
                        base_pattern: EmergentPattern {
                            id: EmergentPatternId::from_description(&format!("temporal_hour_{}", hour)),
                            emergence_type: EmergenceType::BehavioralEmergence,
                            level: EmergenceLevel::Complex,
                            domains: HashSet::from([CognitiveDomain::Reasoning, CognitiveDomain::Creativity]),
                            description: pattern_description.to_string(),
                            analysis: EmergentAnalysis {
                                contributing_components: vec!["insight_generator".to_string()],
                                interaction_patterns: vec![],
                                complexity_metrics: ComplexityMetrics {
                                    component_count: 1,
                                    interaction_complexity: avg_quality,
                                    entropy: 0.8,
                                    organizational_complexity: 0.7,
                                    dynamic_complexity: 0.6,
                                    computational_complexity: 0.5,
                                },
                                triggers: vec![],
                                synchronization_levels: HashMap::new(),
                                information_flows: vec![],
                            },
                            confidence: (insights_in_window.len() as f64 / 10.0).min(1.0),
                            importance: avg_quality,
                            novelty: avg_quality * 0.8,
                            discovered_at: Utc::now(),
                            evidence: vec![],
                            related_patterns: vec![],
                            implications: vec![],
                        },
                        generation_triggers: vec![],
                        success_patterns: HashMap::new(),
                        optimal_combinations: vec![],
                        effectiveness_metrics: InsightPatternMetrics {
                            average_novelty: avg_quality * 0.8,
                            average_confidence: avg_quality * 0.9,
                            validation_success_rate: avg_quality,
                            average_impact: avg_quality,
                            generation_efficiency: insights_in_window.len() as f64 / history.len() as f64,
                        },
                    };

                    patterns.push(pattern);
                }
            }
        }

        tracing::info!("Analyzed {} temporal patterns", patterns.len());
        Ok(patterns)
    }

    // Helper methods for pattern analysis
    async fn analyze_temporal_insight_windows(&self, history: &VecDeque<GeneratedInsight>) -> Result<HashMap<String, Vec<f64>>> {
        use chrono::Duration;

        let mut windows = HashMap::new();
        let _window_size = Duration::minutes(30);

        for insight in history {
            let window_key = format!("{}_{}",
                insight.generated_at.hour(),
                insight.generated_at.minute() / 30 * 30
            );

            let scores = windows.entry(window_key).or_insert_with(Vec::new);
            scores.push(insight.novelty * insight.confidence * insight.impact);
        }

        Ok(windows)
    }

    async fn calculate_type_effectiveness(&self, insight_type: &InsightType, history: &VecDeque<GeneratedInsight>) -> Result<f64> {
        let type_insights: Vec<_> = history.iter()
            .filter(|i| &i.insight_type == insight_type)
            .collect();

        if type_insights.is_empty() {
            return Ok(0.0);
        }

        let total_score = type_insights.iter()
            .map(|i| i.novelty * i.confidence * i.impact)
            .sum::<f64>();

        Ok(total_score / type_insights.len() as f64)
    }

    #[allow(dead_code)]
    async fn calculate_pattern_stability(&self, _insight_type: &InsightType, temporal_windows: &HashMap<String, Vec<f64>>) -> Result<f64> {
        // Calculate coefficient of variation across temporal windows
        let mut window_scores = Vec::new();

        for scores in temporal_windows.values() {
            if !scores.is_empty() {
                window_scores.push(scores.iter().sum::<f64>() / scores.len() as f64);
            }
        }

        if window_scores.len() < 2 {
            return Ok(0.5); // Default stability for insufficient data
        }

        let mean = window_scores.iter().sum::<f64>() / window_scores.len() as f64;
        let variance = window_scores.iter()
            .map(|score| (score - mean).powi(2))
            .sum::<f64>() / window_scores.len() as f64;
        let std_dev = variance.sqrt();

        let coefficient_of_variation = if mean > 0.0 { std_dev / mean } else { 1.0 };
        Ok((1.0 - coefficient_of_variation).max(0.0).min(1.0))
    }

    async fn find_optimal_domain_combinations_for_type(&self, insight_type: &InsightType, history: &VecDeque<GeneratedInsight>) -> Result<Vec<DomainCombination>> {
        use std::collections::HashMap;

        let type_insights: Vec<_> = history.iter()
            .filter(|i| &i.insight_type == insight_type)
            .collect();

        let mut domain_combos = HashMap::new();

        for insight in type_insights {
            let mut domains = insight.contributing_domains.clone();
            domains.sort();
            let key = format!("{:?}", domains);
            let score = insight.novelty * insight.confidence * insight.impact;

            let entry = domain_combos.entry(key).or_insert_with(Vec::new);
            entry.push(score);
        }

        let mut combinations = Vec::new();
        for (_combo_key, scores) in domain_combos {
            if scores.len() >= 2 {
                let avg_score = scores.iter().sum::<f64>() / scores.len() as f64;
                if avg_score > 0.6 {
                    combinations.push(DomainCombination {
                        domains: vec![], // Would parse from combo_key in production
                        synergy_score: avg_score,
                        best_insight_types: vec![insight_type.clone()],
                        optimal_conditions: vec!["Multi-domain activation".to_string()],
                    });
                }
            }
        }

        Ok(combinations)
    }

    async fn extract_triggers_for_type(&self, insight_type: &InsightType, history: &VecDeque<GeneratedInsight>) -> Result<Vec<EmergenceTrigger>> {
        let type_insights: Vec<_> = history.iter()
            .filter(|i| &i.insight_type == insight_type)
            .collect();

        if type_insights.is_empty() {
            return Ok(Vec::new());
        }

        let success_rate = type_insights.iter()
            .filter(|i| i.validation_status == ValidationStatus::Validated)
            .count() as f64 / type_insights.len() as f64;

        let trigger = EmergenceTrigger {
            trigger_type: TriggerType::NovelInput,
            description: format!("Context suitable for {:?}", insight_type),
            condition: TriggerCondition {
                metric: "insight_context_suitability".to_string(),
                threshold: 0.7,
                operator: ComparisonOperator::GreaterThan,
                duration_seconds: None,
            },
            frequency: success_rate,
        };

        Ok(vec![trigger])
    }

    async fn calculate_average_novelty_for_type(&self, insight_type: &InsightType, history: &VecDeque<GeneratedInsight>) -> Result<f64> {
        let type_insights: Vec<_> = history.iter()
            .filter(|i| &i.insight_type == insight_type)
            .collect();

        if type_insights.is_empty() {
            return Ok(0.0);
        }

        Ok(type_insights.iter().map(|i| i.novelty).sum::<f64>() / type_insights.len() as f64)
    }

    async fn calculate_average_confidence_for_type(&self, insight_type: &InsightType, history: &VecDeque<GeneratedInsight>) -> Result<f64> {
        let type_insights: Vec<_> = history.iter()
            .filter(|i| &i.insight_type == insight_type)
            .collect();

        if type_insights.is_empty() {
            return Ok(0.0);
        }

        Ok(type_insights.iter().map(|i| i.confidence).sum::<f64>() / type_insights.len() as f64)
    }

    async fn calculate_validation_success_rate_for_type(&self, insight_type: &InsightType, history: &VecDeque<GeneratedInsight>) -> Result<f64> {
        let type_insights: Vec<_> = history.iter()
            .filter(|i| &i.insight_type == insight_type)
            .collect();

        if type_insights.is_empty() {
            return Ok(0.0);
        }

        let validated_count = type_insights.iter()
            .filter(|i| i.validation_status == ValidationStatus::Validated)
            .count();

        Ok(validated_count as f64 / type_insights.len() as f64)
    }

    async fn calculate_average_impact_for_type(&self, insight_type: &InsightType, history: &VecDeque<GeneratedInsight>) -> Result<f64> {
        let type_insights: Vec<_> = history.iter()
            .filter(|i| &i.insight_type == insight_type)
            .collect();

        if type_insights.is_empty() {
            return Ok(0.0);
        }

        Ok(type_insights.iter().map(|i| i.impact).sum::<f64>() / type_insights.len() as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insight_type_variants() {
        let insight_type = InsightType::ConceptualConnection;
        assert_eq!(insight_type, InsightType::ConceptualConnection);
        assert_ne!(insight_type, InsightType::CreativeBreakthrough);
    }

    #[test]
    fn test_domain_pair_creation() {
        let pair = DomainPair {
            domain1: CognitiveDomain::Memory,
            domain2: CognitiveDomain::Reasoning,
        };

        assert_eq!(pair.domain1, CognitiveDomain::Memory);
        assert_eq!(pair.domain2, CognitiveDomain::Reasoning);
    }

    #[test]
    fn test_validation_status() {
        let status = ValidationStatus::Validated;
        assert_eq!(status, ValidationStatus::Validated);
        assert_ne!(status, ValidationStatus::Invalid);
    }
}
