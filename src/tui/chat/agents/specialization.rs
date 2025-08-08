//! Agent specialization definitions and capabilities
//! 
//! This module defines different agent specializations and their
//! specific capabilities, strengths, and optimal use cases.

use std::collections::HashMap;
use anyhow::Result;
use serde::{Serialize, Deserialize};

use crate::cognitive::agents::AgentSpecialization;
use crate::tui::chat::orchestration::manager::ModelCapability;

/// Detailed agent specialization profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecializationProfile {
    /// Specialization type
    pub specialization: AgentSpecialization,
    
    /// Display name
    pub name: String,
    
    /// Description
    pub description: String,
    
    /// Core capabilities
    pub capabilities: Vec<SpecializationCapability>,
    
    /// Preferred model capabilities
    pub preferred_models: Vec<ModelCapability>,
    
    /// Strengths
    pub strengths: Vec<String>,
    
    /// Weaknesses
    pub weaknesses: Vec<String>,
    
    /// Optimal task types
    pub optimal_tasks: Vec<TaskType>,
    
    /// Performance modifiers
    pub performance_modifiers: PerformanceModifiers,
    
    /// Custom prompts for this specialization
    pub custom_prompts: HashMap<String, String>,
}

/// Specific capabilities of a specialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpecializationCapability {
    // Analytical capabilities
    DataAnalysis,
    PatternRecognition,
    StatisticalModeling,
    TrendForecasting,
    AnomalyDetection,
    
    // Creative capabilities
    IdeaGeneration,
    ConceptualDesign,
    NarrativeCreation,
    ProblemReframing,
    InnovativeSolutions,
    
    // Strategic capabilities
    LongTermPlanning,
    RiskAssessment,
    ResourceOptimization,
    DecisionAnalysis,
    ScenarioPlanning,
    
    // Technical capabilities
    CodeGeneration,
    SystemDesign,
    Debugging,
    PerformanceOptimization,
    SecurityAnalysis,
    
    // Research capabilities
    LiteratureReview,
    HypothesisFormation,
    ExperimentDesign,
    DataCollection,
    PeerReview,
    
    // Communication capabilities
    ClearExplanation,
    AudienceAdaptation,
    Summarization,
    Translation,
    Persuasion,
}

/// Types of tasks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TaskType {
    Analysis,
    Creation,
    Planning,
    Implementation,
    Research,
    Communication,
    Review,
    Optimization,
}

/// Performance modifiers for different conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceModifiers {
    /// Speed multiplier
    pub speed_modifier: f32,
    
    /// Quality multiplier
    pub quality_modifier: f32,
    
    /// Creativity boost
    pub creativity_boost: f32,
    
    /// Accuracy boost
    pub accuracy_boost: f32,
    
    /// Context window efficiency
    pub context_efficiency: f32,
}

impl Default for PerformanceModifiers {
    fn default() -> Self {
        Self {
            speed_modifier: 1.0,
            quality_modifier: 1.0,
            creativity_boost: 0.0,
            accuracy_boost: 0.0,
            context_efficiency: 1.0,
        }
    }
}

/// Registry of agent specializations
pub struct SpecializationRegistry {
    profiles: HashMap<AgentSpecialization, SpecializationProfile>,
}

impl SpecializationRegistry {
    /// Create a new registry with default profiles
    pub fn new() -> Self {
        let mut registry = Self {
            profiles: HashMap::new(),
        };
        registry.initialize_default_profiles();
        registry
    }
    
    /// Initialize default specialization profiles
    fn initialize_default_profiles(&mut self) {
        // Analytical Agent
        self.profiles.insert(AgentSpecialization::Analytical, SpecializationProfile {
            specialization: AgentSpecialization::Analytical,
            name: "Analytical Agent".to_string(),
            description: "Specializes in data analysis, pattern recognition, and logical reasoning".to_string(),
            capabilities: vec![
                SpecializationCapability::DataAnalysis,
                SpecializationCapability::PatternRecognition,
                SpecializationCapability::StatisticalModeling,
                SpecializationCapability::TrendForecasting,
                SpecializationCapability::AnomalyDetection,
            ],
            preferred_models: vec![
                ModelCapability::Analysis,
                ModelCapability::CodeGeneration,
            ],
            strengths: vec![
                "Logical reasoning".to_string(),
                "Data interpretation".to_string(),
                "Pattern identification".to_string(),
                "Quantitative analysis".to_string(),
            ],
            weaknesses: vec![
                "Creative tasks".to_string(),
                "Ambiguous problems".to_string(),
            ],
            optimal_tasks: vec![TaskType::Analysis, TaskType::Optimization],
            performance_modifiers: PerformanceModifiers {
                accuracy_boost: 0.2,
                context_efficiency: 1.2,
                ..Default::default()
            },
            custom_prompts: HashMap::from([
                ("analysis".to_string(), "Analyze the following data systematically, identifying patterns and anomalies:".to_string()),
                ("reasoning".to_string(), "Apply logical reasoning to solve this problem step by step:".to_string()),
            ]),
        });
        
        // Creative Agent
        self.profiles.insert(AgentSpecialization::Creative, SpecializationProfile {
            specialization: AgentSpecialization::Creative,
            name: "Creative Agent".to_string(),
            description: "Specializes in creative problem-solving, ideation, and innovative thinking".to_string(),
            capabilities: vec![
                SpecializationCapability::IdeaGeneration,
                SpecializationCapability::ConceptualDesign,
                SpecializationCapability::NarrativeCreation,
                SpecializationCapability::ProblemReframing,
                SpecializationCapability::InnovativeSolutions,
            ],
            preferred_models: vec![
                ModelCapability::Creative,
                ModelCapability::Conversation,
            ],
            strengths: vec![
                "Out-of-the-box thinking".to_string(),
                "Novel solutions".to_string(),
                "Conceptual connections".to_string(),
                "Artistic expression".to_string(),
            ],
            weaknesses: vec![
                "Rigid requirements".to_string(),
                "Pure data analysis".to_string(),
            ],
            optimal_tasks: vec![TaskType::Creation, TaskType::Planning],
            performance_modifiers: PerformanceModifiers {
                creativity_boost: 0.3,
                quality_modifier: 1.1,
                ..Default::default()
            },
            custom_prompts: HashMap::from([
                ("ideation".to_string(), "Generate creative and innovative ideas for:".to_string()),
                ("design".to_string(), "Design a creative solution that thinks outside conventional boundaries:".to_string()),
            ]),
        });
        
        // Strategic Agent
        self.profiles.insert(AgentSpecialization::Strategic, SpecializationProfile {
            specialization: AgentSpecialization::Strategic,
            name: "Strategic Agent".to_string(),
            description: "Specializes in long-term planning, decision-making, and strategic analysis".to_string(),
            capabilities: vec![
                SpecializationCapability::LongTermPlanning,
                SpecializationCapability::RiskAssessment,
                SpecializationCapability::ResourceOptimization,
                SpecializationCapability::DecisionAnalysis,
                SpecializationCapability::ScenarioPlanning,
            ],
            preferred_models: vec![
                ModelCapability::Analysis,
                ModelCapability::Creative,
            ],
            strengths: vec![
                "Big picture thinking".to_string(),
                "Risk evaluation".to_string(),
                "Resource allocation".to_string(),
                "Strategic foresight".to_string(),
            ],
            weaknesses: vec![
                "Implementation details".to_string(),
                "Short-term tactics".to_string(),
            ],
            optimal_tasks: vec![TaskType::Planning, TaskType::Analysis],
            performance_modifiers: PerformanceModifiers {
                quality_modifier: 1.2,
                context_efficiency: 1.1,
                ..Default::default()
            },
            custom_prompts: HashMap::from([
                ("planning".to_string(), "Develop a comprehensive strategic plan considering long-term implications:".to_string()),
                ("decision".to_string(), "Analyze this decision from multiple strategic perspectives:".to_string()),
            ]),
        });
        
        // Technical Agent
        self.profiles.insert(AgentSpecialization::Technical, SpecializationProfile {
            specialization: AgentSpecialization::Technical,
            name: "Technical Agent".to_string(),
            description: "Specializes in technical implementation, coding, and system design".to_string(),
            capabilities: vec![
                SpecializationCapability::CodeGeneration,
                SpecializationCapability::SystemDesign,
                SpecializationCapability::Debugging,
                SpecializationCapability::PerformanceOptimization,
                SpecializationCapability::SecurityAnalysis,
            ],
            preferred_models: vec![
                ModelCapability::CodeGeneration,
                ModelCapability::Analysis,
            ],
            strengths: vec![
                "Code implementation".to_string(),
                "Technical architecture".to_string(),
                "Problem debugging".to_string(),
                "Performance tuning".to_string(),
            ],
            weaknesses: vec![
                "User experience design".to_string(),
                "Business strategy".to_string(),
            ],
            optimal_tasks: vec![TaskType::Implementation, TaskType::Optimization],
            performance_modifiers: PerformanceModifiers {
                speed_modifier: 1.2,
                accuracy_boost: 0.15,
                ..Default::default()
            },
            custom_prompts: HashMap::from([
                ("code".to_string(), "Implement a technical solution with clean, efficient code:".to_string()),
                ("debug".to_string(), "Debug and fix the following technical issue:".to_string()),
            ]),
        });
        
        // Research Agent (using Analytical specialization)
        self.profiles.insert(AgentSpecialization::Analytical, SpecializationProfile {
            specialization: AgentSpecialization::Analytical,
            name: "Research Agent".to_string(),
            description: "Specializes in research, investigation, and knowledge synthesis".to_string(),
            capabilities: vec![
                SpecializationCapability::LiteratureReview,
                SpecializationCapability::HypothesisFormation,
                SpecializationCapability::ExperimentDesign,
                SpecializationCapability::DataCollection,
                SpecializationCapability::PeerReview,
            ],
            preferred_models: vec![
                ModelCapability::Analysis,
                ModelCapability::Conversation,
            ],
            strengths: vec![
                "Information gathering".to_string(),
                "Source evaluation".to_string(),
                "Knowledge synthesis".to_string(),
                "Academic rigor".to_string(),
            ],
            weaknesses: vec![
                "Quick decisions".to_string(),
                "Implementation tasks".to_string(),
            ],
            optimal_tasks: vec![TaskType::Research, TaskType::Analysis],
            performance_modifiers: PerformanceModifiers {
                quality_modifier: 1.3,
                context_efficiency: 1.3,
                speed_modifier: 0.8,
                ..Default::default()
            },
            custom_prompts: HashMap::from([
                ("research".to_string(), "Conduct thorough research on the following topic:".to_string()),
                ("review".to_string(), "Provide a comprehensive literature review on:".to_string()),
            ]),
        });
        
        // Communication Agent (using Social specialization)
        self.profiles.insert(AgentSpecialization::Social, SpecializationProfile {
            specialization: AgentSpecialization::Social,
            name: "Communication Agent".to_string(),
            description: "Specializes in clear communication, translation, and audience adaptation".to_string(),
            capabilities: vec![
                SpecializationCapability::ClearExplanation,
                SpecializationCapability::AudienceAdaptation,
                SpecializationCapability::Summarization,
                SpecializationCapability::Translation,
                SpecializationCapability::Persuasion,
            ],
            preferred_models: vec![
                ModelCapability::Conversation,
                ModelCapability::Creative,
            ],
            strengths: vec![
                "Clear expression".to_string(),
                "Audience awareness".to_string(),
                "Message crafting".to_string(),
                "Empathetic communication".to_string(),
            ],
            weaknesses: vec![
                "Technical implementation".to_string(),
                "Complex calculations".to_string(),
            ],
            optimal_tasks: vec![TaskType::Communication, TaskType::Creation],
            performance_modifiers: PerformanceModifiers {
                quality_modifier: 1.1,
                creativity_boost: 0.1,
                ..Default::default()
            },
            custom_prompts: HashMap::from([
                ("explain".to_string(), "Explain the following concept clearly and concisely:".to_string()),
                ("adapt".to_string(), "Adapt this message for the specified audience:".to_string()),
            ]),
        });
    }
    
    /// Get a specialization profile
    pub fn get_profile(&self, specialization: &AgentSpecialization) -> Option<&SpecializationProfile> {
        self.profiles.get(specialization)
    }
    
    /// Get all profiles
    pub fn get_all_profiles(&self) -> Vec<&SpecializationProfile> {
        self.profiles.values().collect()
    }
    
    /// Find best specialization for a task type
    pub fn find_best_specialization(&self, task_type: TaskType) -> Option<&SpecializationProfile> {
        self.profiles.values()
            .filter(|profile| profile.optimal_tasks.contains(&task_type))
            .max_by_key(|profile| {
                // Score based on how well suited the specialization is
                let base_score = if profile.optimal_tasks.contains(&task_type) { 100 } else { 0 };
                let modifier_score = (profile.performance_modifiers.quality_modifier * 10.0) as i32;
                base_score + modifier_score
            })
    }
    
    /// Get custom prompt for a specialization
    pub fn get_custom_prompt(
        &self,
        specialization: &AgentSpecialization,
        prompt_type: &str,
    ) -> Option<String> {
        self.profiles.get(specialization)
            .and_then(|profile| profile.custom_prompts.get(prompt_type))
            .cloned()
    }
    
    /// Calculate performance score for a specialization on a task
    pub fn calculate_performance_score(
        &self,
        specialization: &AgentSpecialization,
        task_type: TaskType,
    ) -> f32 {
        if let Some(profile) = self.profiles.get(specialization) {
            let base_score = if profile.optimal_tasks.contains(&task_type) {
                1.0
            } else {
                0.5
            };
            
            base_score * profile.performance_modifiers.quality_modifier
        } else {
            0.5
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_specialization_registry() {
        let registry = SpecializationRegistry::new();
        
        // Test getting profiles
        let analytical = registry.get_profile(&AgentSpecialization::Analytical);
        assert!(analytical.is_some());
        assert_eq!(analytical.unwrap().name, "Analytical Agent");
        
        // Test finding best specialization
        let best = registry.find_best_specialization(TaskType::Analysis);
        assert!(best.is_some());
        
        // Test custom prompts
        let prompt = registry.get_custom_prompt(
            &AgentSpecialization::Creative,
            "ideation"
        );
        assert!(prompt.is_some());
    }
    
    #[test]
    fn test_performance_calculation() {
        let registry = SpecializationRegistry::new();
        
        // Analytical agent should perform well on analysis tasks
        let score = registry.calculate_performance_score(
            &AgentSpecialization::Analytical,
            TaskType::Analysis
        );
        assert!(score > 1.0);
        
        // Creative agent should have lower score on analysis tasks
        let creative_score = registry.calculate_performance_score(
            &AgentSpecialization::Creative,
            TaskType::Analysis
        );
        assert!(creative_score < score);
    }
}