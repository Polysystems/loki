//! Component synchronization between old and modular systems

use std::sync::Arc;
use anyhow::Result;

use crate::tui::chat::integration::SubtabManager;
use crate::models::ModelOrchestrator;
use crate::tui::chat::core::tool_executor::ChatToolExecutor;
use crate::tui::chat::integrations::cognitive::CognitiveChatEnhancement;
use crate::tui::nlp::core::orchestrator::NaturalLanguageOrchestrator;
use crate::tools::intelligent_manager::IntelligentToolManager;
use crate::tools::task_management::TaskManager;

/// Sync components from the old chat system to the modular system
pub fn sync_components_to_modular(
    subtab_manager: &mut SubtabManager,
    model_orchestrator: Option<Arc<ModelOrchestrator>>,
    tool_executor: Option<Arc<ChatToolExecutor>>,
    cognitive_enhancement: Option<Arc<CognitiveChatEnhancement>>,
    nlp_orchestrator: Option<Arc<NaturalLanguageOrchestrator>>,
    intelligent_tool_manager: Option<Arc<IntelligentToolManager>>,
    task_manager: Option<Arc<TaskManager>>,
) -> Result<()> {
    // Sync model orchestrator
    if let Some(orchestrator) = model_orchestrator {
        subtab_manager.set_model_orchestrator(orchestrator);
        tracing::info!("✅ Synced model orchestrator to modular system");
    }
    
    // Sync tool executor
    if let Some(executor) = tool_executor {
        subtab_manager.set_tool_executor(executor);
        tracing::info!("✅ Synced tool executor to modular system");
    }
    
    // Sync cognitive enhancement
    if let Some(enhancement) = cognitive_enhancement {
        subtab_manager.set_cognitive_enhancement(enhancement);
        tracing::info!("✅ Synced cognitive enhancement to modular system");
    }
    
    // Sync NLP orchestrator
    if let Some(orchestrator) = nlp_orchestrator {
        subtab_manager.set_nlp_orchestrator(orchestrator);
        tracing::info!("✅ Synced NLP orchestrator to modular system");
    }
    
    // Sync tool managers
    if let (Some(tool_mgr), Some(task_mgr)) = (intelligent_tool_manager, task_manager) {
        subtab_manager.set_tool_managers(tool_mgr, task_mgr);
        tracing::info!("✅ Synced tool managers to modular system");
    }
    
    tracing::info!("🎯 Component synchronization complete");
    Ok(())
}