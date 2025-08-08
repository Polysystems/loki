use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
// use tracing::info; // Unused import

use crate::models::InferenceEngine;

mod analyzer;
mod assistant;
pub mod panic_safe;
mod watcher;

pub use assistant::{Assistant, AssistantBuilder, DynAssistantBuilder};
pub use panic_safe::{
    CircuitBreaker,
    CircuitBreakerConfig,
    DeadLetterQueue,
    PanicSafeExecutor,
    catch_panic,
    catch_panic_or,
    catch_panic_with_msg,
};

use crate::cli::RunArgs;
use crate::config::{ApiKeysConfig, Config};
use crate::models::{ModelManager, ProviderFactory};

/// Main entry point for running the assistant
pub async fn run(args: RunArgs, config: Config) -> Result<()> {
    // Validate the path
    let path = args.path.canonicalize()?;
    if !path.exists() {
        anyhow::bail!("Path does not exist: {:?}", args.path);
    }


    // Initialize model manager
    let model_name = args.model.as_ref().unwrap_or(&config.default_model);
    let model_manager = ModelManager::new(config.clone()).await?;

    let model = model_manager.load_model(model_name).await?;

    let apiconfig = ApiKeysConfig::from_env()?;
    let _providers = ProviderFactory::create_providers(&apiconfig);

    // Create assistant  
    let boxed_model = Box::new(model) as Box<dyn InferenceEngine>;
    let assistant =
        DynAssistantBuilder::new().with_model(Arc::new(boxed_model)).withconfig(config.clone()).build()?;

    if args.watch {
        // Start file watcher
        watcher::watch_path(&path, assistant, args.interactive).await?;
    } else if args.interactive {
        // Start interactive mode
        assistant.run_interactive(&path).await?;
    } else {
        // Single analysis
        let analysis = assistant.analyze_path(&path).await?;
        assistant.present_analysis(analysis).await?;
    }

    Ok(())
}

/// Analyze a specific path
pub async fn analyze_path(path: &Path, config: &Config) -> Result<analyzer::Analysis> {
    let analyzer = analyzer::Analyzer::new(config.clone());
    analyzer.analyze(path).await
}
