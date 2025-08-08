use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Model information from Ollama
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub modified_at: String,
    pub size: u64,
    pub digest: String,
    pub details: Option<ModelDetails>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDetails {
    pub format: String,
    pub family: String,
    pub families: Option<Vec<String>>,
    pub parameter_size: String,
    pub quantization_level: String,
}

/// Model resource requirements
#[derive(Debug, Clone)]
pub struct ModelRequirements {
    pub name: String,
    pub memory_gb: f32,
    pub gpu_memory_gb: f32,
    pub context_window: usize,
    pub parameter_count: String,
    pub quantization: String,
}

impl ModelRequirements {
    /// Estimate requirements based on model name
    pub fn estimate(model_name: &str) -> Result<Self> {
        let (base_name, quant) = parse_model_name(model_name);

        let (memory_gb, gpu_memory_gb, context_window, param_count) = match base_name {
            // Llama 3.3 models (latest)
            name if name.contains("llama3.3:70b") => (70.0, 65.0, 128000, "70B"),
            name if name.contains("llama3.3:8b") => (8.0, 7.5, 128000, "8B"),
            name if name.contains("llama3.2:1b") => (1.5, 1.2, 128000, "1B"),
            name if name.contains("llama3.2:3b") => (3.0, 2.5, 128000, "3B"),
            name if name.contains("llama3.2:11b") => (11.0, 10.0, 128000, "11B"),
            name if name.contains("llama3.2:90b") => (90.0, 85.0, 128000, "90B"),
            
            // Phi 4 models (latest)
            name if name.contains("phi4:14b") => (14.0, 13.0, 16384, "14B"),
            name if name.contains("phi3.5") || name.contains("phi3:3.8b") => {
                (4.0, 3.5, 128000, "3.8B")
            }
            name if name.contains("phi3:mini") => (2.0, 1.8, 4096, "3.8B"),
            
            // Qwen 2.5 models (latest)
            name if name.contains("qwen2.5-coder:32b") => (32.0, 30.0, 128000, "32B"),
            name if name.contains("qwen2.5-coder:14b") => (14.0, 13.0, 128000, "14B"),
            name if name.contains("qwen2.5-coder:7b") => (7.5, 7.0, 128000, "7B"),
            name if name.contains("qwen2.5-coder:3b") => (3.5, 3.0, 32768, "3B"),
            name if name.contains("qwen2.5-coder:1.5b") => (2.0, 1.8, 32768, "1.5B"),
            name if name.contains("qwen2.5-coder:0.5b") => (0.8, 0.6, 32768, "0.5B"),
            
            // DeepSeek R1 local models
            name if name.contains("deepseek-r1:7b") => (7.5, 7.0, 64000, "7B"),
            name if name.contains("deepseek-r1:14b") => (14.0, 13.0, 64000, "14B"),
            name if name.contains("deepseek-coder-v3:7b") => (7.0, 6.5, 64000, "7B"),
            name if name.contains("deepseek-coder:6.7b") => (7.0, 6.5, 16384, "6.7B"),
            name if name.contains("deepseek-coder:1.3b") => (1.8, 1.5, 16384, "1.3B"),
            
            // Mistral/Mixtral models
            name if name.contains("mixtral:8x22b") => (140.0, 130.0, 65536, "176B"),
            name if name.contains("mixtral:8x7b") => (24.0, 22.0, 32768, "47B"),
            name if name.contains("mistral:7b") => (6.0, 5.5, 32768, "7B"),
            name if name.contains("mistral-nemo") => (12.0, 11.0, 128000, "12B"),
            
            // Codellama models
            name if name.contains("codellama:70b") => (70.0, 65.0, 100000, "70B"),
            name if name.contains("codellama:34b") => (34.0, 32.0, 100000, "34B"),
            name if name.contains("codellama:13b") => (12.0, 11.0, 100000, "13B"),
            name if name.contains("codellama:7b") => (6.5, 6.0, 100000, "7B"),
            
            // Gemma 2 models
            name if name.contains("gemma2:27b") => (27.0, 25.0, 8192, "27B"),
            name if name.contains("gemma2:9b") => (9.0, 8.5, 8192, "9B"),
            name if name.contains("gemma2:2b") => (2.5, 2.0, 8192, "2B"),
            
            // Command-R models
            name if name.contains("command-r:35b") => (35.0, 33.0, 128000, "35B"),
            name if name.contains("command-r-plus") => (104.0, 100.0, 128000, "104B"),
            
            _ => (4.0, 3.5, 8192, "Unknown"), // Default fallback
        };

        // Adjust for quantization
        let (memory_factor, gpu_memory_factor) = match quant.as_str() {
            "q2_k" => (0.3, 0.3),
            "q3_k" | "q3_k_m" => (0.4, 0.4),
            "q4_0" | "q4_k" | "q4_k_m" => (0.5, 0.5),
            "q5_0" | "q5_k" | "q5_k_m" => (0.6, 0.6),
            "q6_k" => (0.7, 0.7),
            "q8_0" => (0.9, 0.9),
            "f16" => (2.0, 2.0),
            "f32" => (4.0, 4.0),
            _ => (1.0, 1.0), // Default/unknown
        };

        Ok(ModelRequirements {
            name: model_name.to_string(),
            memory_gb: memory_gb * memory_factor,
            gpu_memory_gb: gpu_memory_gb * gpu_memory_factor,
            context_window,
            parameter_count: param_count.to_string(),
            quantization: quant,
        })
    }
}

/// Parse model name to extract base name and quantization
fn parse_model_name(model_name: &str) -> (String, String) {
    // Check for explicit quantization in name (e.g., "mistral:7b-q4_0")
    if let Some(pos) = model_name.rfind("-q") {
        let base = model_name[..pos].to_string();
        let quant = model_name[pos + 1..].to_string();
        return (base, quant);
    }

    // Check for GGUF-style quantization (e.g., "model.Q4_K_M.gguf")
    if model_name.to_lowercase().contains(".gguf") {
        for quant in &[
            "q2_k", "q3_k", "q3_k_m", "q4_0", "q4_k", "q4_k_m", "q5_0", "q5_k", "q5_k_m", "q6_k",
            "q8_0",
        ] {
            if model_name.to_lowercase().contains(quant) {
                return (model_name.to_string(), quant.to_string());
            }
        }
    }

    // Default quantization for common models
    let default_quant = if model_name.contains("mini") || model_name.contains("1b") {
        "q4_k_m"
    } else if model_name.contains("3b") || model_name.contains("7b") {
        "q4_0"
    } else {
        "q4_k_m"
    };

    (model_name.to_string(), default_quant.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_requirements() {
        let req = ModelRequirements::estimate("llama3.2:3b").unwrap();
        assert!(req.memory_gb > 0.0);
        assert!(req.gpu_memory_gb > 0.0);
        assert_eq!(req.parameter_count, "3B");

        let req2 = ModelRequirements::estimate("mistral:7b-q4_0").unwrap();
        assert_eq!(req2.quantization, "q4_0");
    }
}
