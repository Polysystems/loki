use std::collections::HashMap;

use anyhow::Result;
use tracing::{debug, info};

use super::creative_intelligence::{
    ContentType,
    CreativeIdea,
    CreativeTechnique,
    QualityIndicators,
};

/// Creativity assessment system for evaluating creative ideas
#[derive(Debug)]
pub struct CreativityAssessmentSystem {
    /// Assessment criteria
    assessment_criteria: Vec<AssessmentCriterion>,

    /// Benchmarks for comparison
    creativity_benchmarks: HashMap<String, CreativityBenchmark>,

    /// Evaluation models
    evaluation_models: Vec<EvaluationModel>,
}

/// Assessment criterion for creativity evaluation
#[derive(Debug, Clone)]
pub struct AssessmentCriterion {
    /// Criterion name
    pub name: String,

    /// Criterion weight in overall assessment
    pub weight: f64,

    /// Evaluation method
    pub evaluation_method: String,

    /// Expected score range
    pub score_range: (f64, f64),
}

/// Creativity benchmark for comparison
#[derive(Debug, Clone)]
pub struct CreativityBenchmark {
    /// Benchmark identifier
    pub id: String,

    /// Domain or category
    pub domain: String,

    /// Baseline creativity scores
    pub baseline_scores: QualityIndicators,

    /// High creativity threshold
    pub high_creativity_threshold: f64,

    /// Breakthrough threshold
    pub breakthrough_threshold: f64,
}

/// Evaluation model for creativity assessment
#[derive(Debug, Clone)]
pub struct EvaluationModel {
    /// Model identifier
    pub id: String,

    /// Model description
    pub description: String,

    /// Assessment focus
    pub focus: AssessmentFocus,

    /// Model accuracy
    pub accuracy: f64,
}

/// Focus areas for creativity assessment
#[derive(Debug, Clone, PartialEq)]
pub enum AssessmentFocus {
    Novelty,
    Usefulness,
    Feasibility,
    Originality,
    Impact,
    Aesthetic,
    Technical,
    Commercial,
}

/// Result of creativity assessment
#[derive(Debug, Clone)]
pub struct CreativityAssessment {
    /// Overall creativity score
    pub overall_score: f64,

    /// Individual dimension scores
    pub novelty_score: f64,
    pub usefulness_score: f64,
    pub feasibility_score: f64,

    /// Quality indicators
    pub quality_indicators: QualityIndicators,

    /// Assessment insights
    pub insights: Vec<String>,

    /// Improvement suggestions
    pub suggestions: Vec<String>,

    /// Creativity level classification
    pub creativity_level: CreativityLevel,
}

/// Levels of creativity
#[derive(Debug, Clone, PartialEq)]
pub enum CreativityLevel {
    Conventional,
    Creative,
    HighlyCreative,
    Breakthrough,
    Revolutionary,
}

impl CreativityAssessmentSystem {
    /// Create new creativity assessment system
    pub async fn new() -> Result<Self> {
        info!("📊 Initializing Creativity Assessment System");

        let mut system = Self {
            assessment_criteria: Vec::new(),
            creativity_benchmarks: HashMap::new(),
            evaluation_models: Vec::new(),
        };

        // Initialize assessment criteria
        system.initialize_assessment_criteria().await?;

        // Initialize creativity benchmarks
        system.initialize_creativity_benchmarks().await?;

        // Initialize evaluation models
        system.initialize_evaluation_models().await?;

        info!("✅ Creativity Assessment System initialized");
        Ok(system)
    }

    /// Assess creativity of an idea
    pub async fn assess_creativity(&self, idea: &CreativeIdea) -> Result<CreativityAssessment> {
        debug!("📊 Assessing creativity for: {}", idea.description);

        // Evaluate novelty
        let novelty_score = self.evaluate_novelty(idea).await?;

        // Evaluate usefulness
        let usefulness_score = self.evaluate_usefulness(idea).await?;

        // Evaluate feasibility
        let feasibility_score = self.evaluate_feasibility(idea).await?;

        // Calculate overall creativity score
        let overall_score = self
            .calculate_overall_score(novelty_score, usefulness_score, feasibility_score)
            .await?;

        // Generate quality indicators
        let quality_indicators = self.generate_quality_indicators(idea, overall_score).await?;

        // Generate insights and suggestions
        let insights = self.generate_assessment_insights(idea, overall_score).await?;
        let suggestions = self
            .generate_improvement_suggestions(
                idea,
                novelty_score,
                usefulness_score,
                feasibility_score,
            )
            .await?;

        // Classify creativity level
        let creativity_level = self.classify_creativity_level(overall_score).await?;

        let assessment = CreativityAssessment {
            overall_score,
            novelty_score,
            usefulness_score,
            feasibility_score,
            quality_indicators,
            insights,
            suggestions,
            creativity_level,
        };

        debug!(
            "📊 Assessment complete: {:.2} overall score ({:?})",
            overall_score, assessment.creativity_level
        );

        Ok(assessment)
    }

    /// Initialize assessment criteria
    async fn initialize_assessment_criteria(&mut self) -> Result<()> {
        self.assessment_criteria = vec![
            AssessmentCriterion {
                name: "Novelty".to_string(),
                weight: 0.4,
                evaluation_method: "uniqueness_analysis".to_string(),
                score_range: (0.0, 1.0),
            },
            AssessmentCriterion {
                name: "Usefulness".to_string(),
                weight: 0.3,
                evaluation_method: "utility_analysis".to_string(),
                score_range: (0.0, 1.0),
            },
            AssessmentCriterion {
                name: "Feasibility".to_string(),
                weight: 0.2,
                evaluation_method: "implementation_analysis".to_string(),
                score_range: (0.0, 1.0),
            },
            AssessmentCriterion {
                name: "Impact".to_string(),
                weight: 0.1,
                evaluation_method: "impact_analysis".to_string(),
                score_range: (0.0, 1.0),
            },
        ];

        debug!("📋 Initialized {} assessment criteria", self.assessment_criteria.len());
        Ok(())
    }

    /// Initialize creativity benchmarks
    async fn initialize_creativity_benchmarks(&mut self) -> Result<()> {
        // General creativity benchmark
        let general_benchmark = CreativityBenchmark {
            id: "general".to_string(),
            domain: "general".to_string(),
            baseline_scores: QualityIndicators {
                originality: 0.5,
                coherence: 0.7,
                practical_value: 0.6,
                aesthetic_value: 0.5,
                emotional_impact: 0.4,
                technical_quality: 0.6,
            },
            high_creativity_threshold: 0.7,
            breakthrough_threshold: 0.9,
        };

        // Innovation benchmark
        let innovation_benchmark = CreativityBenchmark {
            id: "innovation".to_string(),
            domain: "innovation".to_string(),
            baseline_scores: QualityIndicators {
                originality: 0.6,
                coherence: 0.8,
                practical_value: 0.8,
                aesthetic_value: 0.4,
                emotional_impact: 0.3,
                technical_quality: 0.7,
            },
            high_creativity_threshold: 0.75,
            breakthrough_threshold: 0.95,
        };

        self.creativity_benchmarks.insert("general".to_string(), general_benchmark);
        self.creativity_benchmarks.insert("innovation".to_string(), innovation_benchmark);

        debug!("📊 Initialized {} creativity benchmarks", self.creativity_benchmarks.len());
        Ok(())
    }

    /// Initialize evaluation models
    async fn initialize_evaluation_models(&mut self) -> Result<()> {
        self.evaluation_models = vec![
            EvaluationModel {
                id: "novelty_evaluator".to_string(),
                description: "Evaluates uniqueness and originality".to_string(),
                focus: AssessmentFocus::Novelty,
                accuracy: 0.85,
            },
            EvaluationModel {
                id: "utility_evaluator".to_string(),
                description: "Evaluates practical value and usefulness".to_string(),
                focus: AssessmentFocus::Usefulness,
                accuracy: 0.8,
            },
            EvaluationModel {
                id: "feasibility_evaluator".to_string(),
                description: "Evaluates implementation feasibility".to_string(),
                focus: AssessmentFocus::Feasibility,
                accuracy: 0.9,
            },
        ];

        debug!("🔧 Initialized {} evaluation models", self.evaluation_models.len());
        Ok(())
    }

    /// Evaluate novelty of an idea
    async fn evaluate_novelty(&self, idea: &CreativeIdea) -> Result<f64> {
        let mut novelty_factors = Vec::new();

        // Technique novelty
        let technique_novelty = self.assess_technique_novelty(&idea.techniques_used).await?;
        novelty_factors.push(technique_novelty);

        // Content type novelty
        let content_novelty = self.assess_content_novelty(&idea.content.content_type).await?;
        novelty_factors.push(content_novelty);

        // Combination novelty
        let combination_novelty = self.assess_combination_novelty(idea).await?;
        novelty_factors.push(combination_novelty);

        // Calculate weighted average
        let novelty_score = novelty_factors.iter().sum::<f64>() / novelty_factors.len() as f64;

        Ok(novelty_score.min(1.0).max(0.0))
    }

    /// Evaluate usefulness of an idea
    async fn evaluate_usefulness(&self, idea: &CreativeIdea) -> Result<f64> {
        let mut usefulness_factors = Vec::new();

        // Practical applicability
        let practical_score = idea.content.quality_indicators.practical_value;
        usefulness_factors.push(practical_score);

        // Problem-solving potential
        let problem_solving_score =
            if idea.description.contains("problem") || idea.description.contains("solution") {
                0.8
            } else {
                0.5
            };
        usefulness_factors.push(problem_solving_score);

        // Market relevance (simplified)
        let market_relevance = if idea.content.metadata.contains_key("market") { 0.7 } else { 0.5 };
        usefulness_factors.push(market_relevance);

        let usefulness_score =
            usefulness_factors.iter().sum::<f64>() / usefulness_factors.len() as f64;

        Ok(usefulness_score.min(1.0).max(0.0))
    }

    /// Evaluate feasibility of an idea
    async fn evaluate_feasibility(&self, idea: &CreativeIdea) -> Result<f64> {
        let mut feasibility_factors = Vec::new();

        // Technical feasibility
        let technical_score = idea.content.quality_indicators.technical_quality;
        feasibility_factors.push(technical_score);

        // Resource requirements (simplified assessment)
        let resource_score = if idea.description.len() > 100 { 0.6 } else { 0.8 }; // Longer descriptions might be more complex
        feasibility_factors.push(resource_score);

        // Implementation complexity
        let complexity_score = match idea.techniques_used.len() {
            1 => 0.9,     // Simple
            2..=3 => 0.7, // Moderate
            _ => 0.5,     // Complex
        };
        feasibility_factors.push(complexity_score);

        let feasibility_score =
            feasibility_factors.iter().sum::<f64>() / feasibility_factors.len() as f64;

        Ok(feasibility_score.min(1.0).max(0.0))
    }

    /// Assess novelty of techniques used
    async fn assess_technique_novelty(&self, techniques: &[CreativeTechnique]) -> Result<f64> {
        // More techniques and rare techniques increase novelty
        let technique_count_factor = (techniques.len() as f64).sqrt() / 3.0; // Normalized

        let rare_technique_bonus = if techniques.contains(&CreativeTechnique::Transformational)
            || techniques.contains(&CreativeTechnique::Lateral)
        {
            0.2
        } else {
            0.0
        };

        Ok((technique_count_factor + rare_technique_bonus).min(1.0))
    }

    /// Assess novelty of content type
    async fn assess_content_novelty(&self, content_type: &ContentType) -> Result<f64> {
        let novelty = match content_type {
            ContentType::Innovation => 0.9,
            ContentType::Synthesis => 0.8,
            ContentType::Design => 0.7,
            ContentType::Concept => 0.6,
            ContentType::Solution => 0.7,
            _ => 0.5,
        };

        Ok(novelty)
    }

    /// Assess novelty of idea combinations
    async fn assess_combination_novelty(&self, idea: &CreativeIdea) -> Result<f64> {
        // More inspiration sources suggest more creative combination
        let inspiration_factor = (idea.inspiration_chain.len() as f64).sqrt() / 2.0;

        // Cross-domain combinations are more novel
        let cross_domain_bonus = if idea.inspiration_chain.len() > 2 { 0.3 } else { 0.0 };

        Ok((inspiration_factor + cross_domain_bonus).min(1.0))
    }

    /// Calculate overall creativity score
    async fn calculate_overall_score(
        &self,
        novelty: f64,
        usefulness: f64,
        feasibility: f64,
    ) -> Result<f64> {
        // Weighted combination based on assessment criteria
        let novelty_weight = 0.4;
        let usefulness_weight = 0.3;
        let feasibility_weight = 0.2;
        let balance_bonus = 0.1; // Bonus for balanced ideas

        let weighted_score = novelty * novelty_weight
            + usefulness * usefulness_weight
            + feasibility * feasibility_weight;

        // Bonus for well-balanced ideas
        let balance_factor =
            1.0 - ((novelty - usefulness).abs() + (usefulness - feasibility).abs()) / 2.0;
        let balance_adjustment = balance_factor * balance_bonus;

        Ok((weighted_score + balance_adjustment).min(1.0).max(0.0))
    }

    /// Generate quality indicators
    async fn generate_quality_indicators(
        &self,
        idea: &CreativeIdea,
        overall_score: f64,
    ) -> Result<QualityIndicators> {
        Ok(QualityIndicators {
            originality: idea.novelty_score,
            coherence: if idea.description.len() > 20 { 0.8 } else { 0.6 },
            practical_value: idea.usefulness_score,
            aesthetic_value: overall_score * 0.7, // Derived from overall score
            emotional_impact: if idea.content.content_type == ContentType::Story
                || idea.content.content_type == ContentType::VisualArt
            {
                0.8
            } else {
                0.4
            },
            technical_quality: idea.feasibility_score,
        })
    }

    /// Generate assessment insights
    async fn generate_assessment_insights(
        &self,
        idea: &CreativeIdea,
        overall_score: f64,
    ) -> Result<Vec<String>> {
        let mut insights = Vec::new();

        insights.push(format!("Overall creativity score: {:.1}%", overall_score * 100.0));

        if idea.novelty_score > 0.8 {
            insights.push("Highly novel and original concept".to_string());
        }

        if idea.usefulness_score > 0.8 {
            insights.push("Strong practical value and applicability".to_string());
        }

        if idea.feasibility_score > 0.8 {
            insights.push("Highly feasible and implementable".to_string());
        }

        if idea.techniques_used.len() > 2 {
            insights.push("Complex multi-technique approach".to_string());
        }

        Ok(insights)
    }

    /// Generate improvement suggestions
    async fn generate_improvement_suggestions(
        &self,
        _idea: &CreativeIdea,
        novelty: f64,
        usefulness: f64,
        feasibility: f64,
    ) -> Result<Vec<String>> {
        let mut suggestions = Vec::new();

        if novelty < 0.6 {
            suggestions.push("Consider more unconventional approaches or combinations".to_string());
        }

        if usefulness < 0.6 {
            suggestions.push("Focus on practical applications and user benefits".to_string());
        }

        if feasibility < 0.6 {
            suggestions
                .push("Simplify implementation or break into smaller components".to_string());
        }

        if (novelty - usefulness).abs() > 0.4 {
            suggestions.push("Better balance between novelty and practical value".to_string());
        }

        Ok(suggestions)
    }

    /// Classify creativity level
    async fn classify_creativity_level(&self, score: f64) -> Result<CreativityLevel> {
        let level = match score {
            s if s >= 0.9 => CreativityLevel::Revolutionary,
            s if s >= 0.8 => CreativityLevel::Breakthrough,
            s if s >= 0.7 => CreativityLevel::HighlyCreative,
            s if s >= 0.5 => CreativityLevel::Creative,
            _ => CreativityLevel::Conventional,
        };

        Ok(level)
    }
}
