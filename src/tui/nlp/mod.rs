//\! Natural Language Processing module

pub mod core;
pub mod analysis;

// Re-export commonly used items
pub use core::{
    base::NaturalLanguageProcessor as NaturalLanguageInterface,
    processor::NaturalLanguageProcessor,
    orchestrator::NaturalLanguageOrchestrator,
};

pub use analysis::enhanced_analyzer::EnhancedNLPAnalyzer;
