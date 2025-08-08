use anyhow::Result;
use async_trait::async_trait;
use serde_json::json;
use tracing::{debug, info};

use crate::tasks::{Task, TaskArgs, TaskContext, TaskResult};

pub struct CodeCompletionTask;

#[async_trait]
impl Task for CodeCompletionTask {
    fn name(&self) -> &str {
        "complete"
    }

    fn description(&self) -> &str {
        "Generate code completions using local models"
    }

    async fn execute(&self, args: TaskArgs, context: TaskContext) -> Result<TaskResult> {
        // Get input file
        let input_path =
            args.input.ok_or_else(|| anyhow::anyhow!("Input file required for code completion"))?;

        // Validate input file exists
        if !input_path.exists() {
            return Ok(TaskResult {
                success: false,
                message: format!("Input file not found: {:?}", input_path),
                data: None,
            });
        }

        info!("Generating completions for: {:?}", input_path);

        // Read the file content
        let content = tokio::fs::read_to_string(&input_path).await?;

        // Parse additional arguments
        let cursor_position =
            args.args.get(0).and_then(|s| s.parse::<usize>().ok()).unwrap_or(content.len());

        // Get the model
        // TODO: import default model name
        let model_name = context.config.default_model.clone();
        let model = context.model_manager.load_model(&model_name).await?;

        // Prepare context
        let (prefix, suffix) = if cursor_position <= content.len() {
            (&content[..cursor_position], &content[cursor_position..])
        } else {
            (content.as_str(), "")
        };

        // Create prompt
        let prompt =
            format!("Complete the following code:\n\n{}[CURSOR]{}\n\nCompletion:", prefix, suffix);

        debug!("Completion prompt: {}", prompt);

        // Run inference
        let request = crate::models::InferenceRequest {
            prompt,
            max_tokens: 150,
            temperature: 0.2,
            top_p: 0.95,
            stop_sequences: vec!["\n\n".to_string()],
        };

        let response = model.infer(request).await?;

        // Process completions
        let completions = vec![json!({
            "text": response.text,
            "confidence": 0.9,
            "cursor_position": cursor_position,
        })];

        // Save to output if specified
        if let Some(output_path) = args.output {
            let output_data = json!({
                "input_file": input_path,
                "completions": completions,
                "model": model_name,
            });
            let output_content = serde_json::to_string_pretty(&output_data)?;
            tokio::fs::write(output_path, output_content).await?;
        }

        Ok(TaskResult {
            success: true,
            message: format!("Generated {} completion(s)", completions.len()),
            data: Some(json!({
                "completions": completions,
                "model": model_name,
            })),
        })
    }
}
