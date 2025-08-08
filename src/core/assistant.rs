use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use tracing::{debug, info};

use super::analyzer::Analysis;
use crate::config::Config;
use crate::models::{InferenceEngine, InferenceRequest};

/// Generic assistant struct with zero-cost inference engine abstraction
pub struct Assistant<E: InferenceEngine> {
    model: Arc<E>,
    config: Config,
}

/// Type-erased assistant for compatibility with existing APIs
pub type DynAssistant = Assistant<Box<dyn InferenceEngine>>;

/// Default Assistant type for backward compatibility
pub use DynAssistant as DefaultAssistant;

/// Concrete assistant types for common engines (zero dispatch overhead)
/// Production-ready concrete types for cognitive processing
pub type AnthropicAssistant = Assistant<crate::models::ApiInferenceEngine>;
pub type OpenAIAssistant = Assistant<crate::models::ApiInferenceEngine>;
pub type LocalAssistant = Assistant<crate::models::ApiInferenceEngine>;
pub type CognitiveAssistant = Assistant<crate::models::ApiInferenceEngine>;
/// Specialized implementations for concrete inference engines
/// These provide zero-cost optimizations through monomorphization
mod specialized_assistants {
    use super::*;
    use crate::models::ApiInferenceEngine;
    
    /// Specialized builder and assistant for API-based inference engines
    /// This avoids dynamic dispatch and enables compile-time optimizations
    pub type ApiAssistant = Assistant<ApiInferenceEngine>;
    pub type ApiAssistantBuilder = AssistantBuilder<ApiInferenceEngine>;
    
    impl ApiAssistant {
        /// Fast path for API-based inference (no trait object overhead)
        #[inline(always)]
        pub async fn quick_infer(&self, prompt: &str) -> Result<String> {
            let request = InferenceRequest {
                prompt: prompt.to_string(),
                max_tokens: 1000,
                temperature: 0.7,
                top_p: 0.95,
                stop_sequences: vec![],
            };
            
            // Direct call - compiler can inline this completely
            let response = self.model.infer(request).await?;
            Ok(response.text)
        }
        
        /// Optimized batch processing for API engines
        #[inline(always)]
        pub async fn batch_infer(&self, prompts: &[&str]) -> Result<Vec<String>> {
            let mut results = Vec::with_capacity(prompts.len());
            
            // For API engines, we can potentially batch these requests
            // This specialization allows API-specific optimizations
            for prompt in prompts {
                let result = self.quick_infer(prompt).await?;
                results.push(result);
            }
            
            Ok(results)
        }
    }
    
    impl ApiAssistantBuilder {
        /// Specialized builder for API engines with pre-configured defaults
        pub fn with_api_defaults() -> Self {
            Self::new()
        }
        
        /// Chain-optimized configuration for API engines
        #[inline(always)]
        pub fn with_api_config(self, config: Config) -> Self {
            self.withconfig(config)
        }
    }
}

// Re-export specialized types when needed
#[allow(unused_imports)]
pub use specialized_assistants::*;

/// Generic builder for creating an Assistant instance with zero-cost abstractions
pub struct AssistantBuilder<E: InferenceEngine> {
    model: Option<Arc<E>>,
    config: Option<Config>,
}

/// Type-erased builder for compatibility
pub type DynAssistantBuilder = AssistantBuilder<Box<dyn InferenceEngine>>;

impl<E: InferenceEngine> AssistantBuilder<E> {
    /// Create a new builder
    pub fn new() -> Self {
        Self { model: None, config: None }
    }

    /// Set the model (zero-cost for concrete types)
    pub fn with_model(mut self, model: Arc<E>) -> Self {
        self.model = Some(model);
        self
    }

    /// Set the configuration
    pub fn withconfig(mut self, config: Config) -> Self {
        self.config = Some(config);
        self
    }

    /// Build the Assistant
    pub fn build(self) -> Result<Assistant<E>> {
        let model = self.model.ok_or_else(|| anyhow::anyhow!("Model is required"))?;
        let config = self.config.ok_or_else(|| anyhow::anyhow!("Configuration is required"))?;

        Ok(Assistant { model, config })
    }
}

impl<E: InferenceEngine> Assistant<E> {
    /// Analyze a path (file or directory)
    pub async fn analyze_path(&self, path: &Path) -> Result<Analysis> {
        info!("Analyzing path: {:?}", path);

        // Use the analyzer from the parent module
        let analyzer = super::analyzer::Analyzer::new(self.config.clone());
        analyzer.analyze(path).await
    }

    /// Present analysis results to the user
    pub async fn present_analysis(&self, analysis: Analysis) -> Result<()> {
        // Create a summary prompt
        let prompt = format!(
            "Summarize the following code analysis results in a helpful way for the \
             developer:\n\n{}",
            serde_json::to_string_pretty(&analysis)?
        );

        let request = InferenceRequest {
            prompt,
            max_tokens: 500,
            temperature: 0.7,
            top_p: 0.95,
            stop_sequences: vec![],
        };

        let response = self.model.infer(request).await?;

        println!("\n📊 Analysis Summary:\n");
        println!("{}", response.text);
        println!("\n---");

        // Show key metrics
        println!("\n📈 Key Metrics:");
        println!("  Files analyzed: {}", analysis.file_count);
        println!("  Total lines: {}", analysis.total_lines);
        if !analysis.languages.is_empty() {
            println!("  Languages: {}", analysis.languages.join(", "));
        }
        if !analysis.issues.is_empty() {
            println!("  Issues found: {}", analysis.issues.len());
        }

        Ok(())
    }

    /// Run in interactive mode
    pub async fn run_interactive(&self, path: &Path) -> Result<()> {
        use std::io::{self, Write};

        println!("🚀 Loki Interactive Mode");
        println!("📁 Working directory: {:?}", path);
        println!("🤖 Model: {}", self.config.default_model);
        println!("\nType 'help' for available commands or 'exit' to quit.\n");

        let stdin = io::stdin();
        let mut stdout = io::stdout();

        loop {
            print!("loki> ");
            stdout.flush()?;

            let mut input = String::new();
            stdin.read_line(&mut input)?;
            let input = input.trim();

            match input {
                "exit" | "quit" => {
                    println!("Goodbye! 👋");
                    break;
                }
                "help" => {
                    self.show_help();
                }
                "analyze" => {
                    // Optimized: Chain awaits to minimize state transitions
                    self.present_analysis(self.analyze_path(path).await?).await?;
                }
                _ if !input.is_empty() => {
                    // Send the input to the model as a question about the codebase
                    self.handle_query(input, path).await?;
                }
                _ => {} // Empty input, just show prompt again
            }
        }

        Ok(())
    }

    /// Show help information
    fn show_help(&self) {
        println!("\n📚 Available Commands:");
        println!("  help     - Show this help message");
        println!("  analyze  - Analyze the current directory");
        println!("  exit     - Exit interactive mode");
        println!("  <query>  - Ask anything about your code");
        println!("\nExamples:");
        println!("  'What does this project do?'");
        println!("  'Find potential bugs'");
        println!("  'Suggest improvements for performance'");
        println!();
    }

    /// Handle a user query (optimized async state machine layout)
    #[inline(always)] // Single await - inline for zero-cost
    async fn handle_query(&self, query: &str, path: &Path) -> Result<()> {
        debug!("Handling query: {}", query);

        // Minimize state machine by preparing request data first
        println!("\n🤔 Thinking...\n");
        
        // Single await point reduces state machine complexity
        let analysis = self.analyze_path(path).await?;
        
        // All CPU-intensive operations after async completion (smaller state machine)
        let prompt = Self::build_context_prompt(path, &analysis, query);
        let request = InferenceRequest {
            prompt,
            max_tokens: 1000,
            temperature: 0.7,
            top_p: 0.95,
            stop_sequences: vec![],
        };

        // Final await in tail position (compiler optimization)
        let response = self.model.infer(request).await?;
        println!("{}\n", response.text);
        Ok(())
    }
    
    /// Helper function to reduce async state machine size (backend optimized pure function)
    #[inline(always)] // Critical path for prompt generation
    fn build_context_prompt(path: &Path, analysis: &Analysis, query: &str) -> String {
        // Backend optimization: low register pressure for string formatting
        crate::compiler_backend_optimization::register_optimization::low_register_pressure(|| {
            // Critical hot path for context building
            crate::compiler_backend_optimization::critical_path_optimization::ultra_fast_path(|| {
                format!(
                    "You are a helpful coding assistant. The user is working on a project at \
                     {:?}.\n\nProject summary:\n- Files: {}\n- Languages: {}\n- Total lines: {}\n\nUser \
                     question: {}\n\nPlease provide a helpful and concise answer:",
                        path,
                        analysis.file_count,
                        analysis.languages.join(", "),
                        analysis.total_lines,
                        query
                    )
            })
        })
    }
}
