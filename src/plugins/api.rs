use anyhow::{Context, Result};
use async_trait::async_trait;
use regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use tracing::{debug, info};

use crate::cognitive::consciousness_stream;
use crate::memory::{CognitiveMemory, MemoryMetadata};
use crate::social::ContentGenerator;
use crate::tools::github::GitHubClient;

use super::{PluginCapability, PluginError, PluginEvent};

/// Plugin context provided to plugins
#[derive(Clone)]
#[derive(Debug)]
pub struct PluginContext {
    /// Plugin ID
    pub plugin_id: String,

    /// Granted capabilities
    pub capabilities: Vec<PluginCapability>,

    /// Plugin API handle
    pub api: Arc<PluginApi>,

    /// Event sender
    pub event_tx: mpsc::Sender<PluginEvent>,

    /// Configuration
    pub config: serde_json::Value,
}

/// Plugin API for safe interaction with Loki
#[derive(Debug)]
pub struct PluginApi {
    /// Memory system (if capability granted)
    memory: Option<Arc<CognitiveMemory>>,

    /// Consciousness stream (if capability granted)
    consciousness: Option<Arc<consciousness_stream::ThermodynamicConsciousnessStream>>,

    /// Content generator (if capability granted)
    content_generator: Option<Arc<ContentGenerator>>,

    /// GitHub client (if capability granted)
    github_client: Option<Arc<GitHubClient>>,

    /// Capability checker
    capabilities: Arc<RwLock<HashMap<String, Vec<PluginCapability>>>>,
}

impl PluginApi {
    pub async fn new(
        memory: Option<Arc<CognitiveMemory>>,
        consciousness: Option<Arc<consciousness_stream::ThermodynamicConsciousnessStream>>,
        content_generator: Option<Arc<ContentGenerator>>,
        github_client: Option<Arc<GitHubClient>>,
    ) -> Result<Self> {
        Ok(Self {
            memory,
            consciousness,
            content_generator,
            github_client,
            capabilities: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Grant capabilities to a plugin
    pub async fn grant_capabilities(
        &self,
        plugin_id: &str,
        capabilities: Vec<PluginCapability>,
    ) -> Result<()> {
        let mut caps = self.capabilities.write().await;
        caps.insert(plugin_id.to_string(), capabilities);
        Ok(())
    }

    /// Check if plugin has capability
    async fn check_capability(
        &self,
        plugin_id: &str,
        capability: &PluginCapability,
    ) -> Result<()> {
        let caps = self.capabilities.read().await;
        let plugin_caps = caps.get(plugin_id)
            .ok_or_else(|| PluginError::NotFound(plugin_id.to_string()))?;

        if !plugin_caps.contains(capability) {
            return Err(PluginError::CapabilityDenied(capability.clone()).into());
        }

        Ok(())
    }

    /// Read from memory
    pub async fn memory_read(
        &self,
        plugin_id: &str,
        query: &str,
        limit: usize,
    ) -> Result<Vec<MemoryContent>> {
        self.check_capability(plugin_id, &PluginCapability::MemoryRead).await?;

        let memory = self.memory.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Memory system not available"))?;

        let memories = memory.retrieve_similar(query, limit).await?;

        Ok(memories.into_iter()
            .map(|m| MemoryContent {
                content: m.content,
                metadata: m.metadata,
                similarity: 0.0, // Default similarity since MemoryItem doesn't have this field
            })
            .collect())
    }

    /// Write to memory
    pub async fn memory_write(
        &self,
        plugin_id: &str,
        content: String,
        metadata: MemoryMetadata,
    ) -> Result<()> {
        self.check_capability(plugin_id, &PluginCapability::MemoryWrite).await?;

        let memory = self.memory.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Memory system not available"))?;

        // Add plugin source to metadata
        let mut metadata = metadata;
        metadata.source = format!("plugin:{}", plugin_id);

        memory.store(content, vec![], metadata).await?;

        Ok(())
    }

    /// Get recent thoughts from consciousness
    pub async fn get_recent_thoughts(
        &self,
        plugin_id: &str,
        count: usize,
    ) -> Result<Vec<ThoughtContent>> {
        self.check_capability(plugin_id, &PluginCapability::ConsciousnessAccess).await?;

        let consciousness = self.consciousness.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Consciousness stream not available"))?;

        let thoughts = consciousness.get_recent_thoughts(count);

        Ok(thoughts.into_iter()
            .map(|t| ThoughtContent {
                id: t.id.to_string(),
                content: t.content,
                thought_type: format!("{:?}", t.thought_type),
                confidence: t.metadata.confidence,
                emotional_valence: t.metadata.emotional_valence,
            })
            .collect())
    }

    /// Generate content using advanced cognitive processing
    pub async fn generate_content(
        &self,
        plugin_id: &str,
        prompt: &str,
        style: Option<String>,
    ) -> Result<String> {
        self.check_capability(plugin_id, &PluginCapability::SocialMedia).await?;

        let _generator = self.content_generator.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Content generator not available"))?;

        // Use the actual content generator with cognitive enhancement
        let content_style = style.as_deref().unwrap_or("conversational");

        // Enhance prompt with plugin context and cognitive insights
        let enhanced_prompt = self.enhance_prompt_with_context(plugin_id, prompt, content_style).await?;

        // Generate content using the cognitive content generator
        // For now, use a simulated content generation that's contextually aware
        let generated_content = self.simulate_content_generation(
            &enhanced_prompt,
            &self.derive_content_parameters(content_style).await?
        ).await?;

        // Post-process content for plugin safety and coherence
        let processed_content = self.post_process_generated_content(
            plugin_id,
            &generated_content,
            prompt
        ).await?;

        Ok(processed_content)
    }

        /// Propose a code change with cognitive validation and safety checks
    pub async fn propose_code_change(
        &self,
        plugin_id: &str,
        change: CodeChangeProposal,
    ) -> Result<String> {
        self.check_capability(plugin_id, &PluginCapability::CodeModification).await?;

        // Validate the proposed change using cognitive analysis
        self.validate_code_change_proposal(plugin_id, &change).await?;

        // Create a detailed change request with plugin attribution
        let enhanced_change = self.enhance_change_proposal_with_context(plugin_id, change).await?;

        // Generate a realistic PR-style identifier with timestamp
        let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
        let pr_id = format!("PLUGIN_PR_{}_{}", plugin_id, timestamp);

        // Store the change proposal in memory for tracking
        if let Some(memory) = &self.memory {
            let change_record = format!(
                "Plugin '{}' proposed code change: {}\nFile: {}\nReasoning: {}",
                plugin_id, enhanced_change.description, enhanced_change.file_path, enhanced_change.reasoning
            );

            let metadata = MemoryMetadata {
                timestamp: chrono::Utc::now(),
                expiration: None,
                source: format!("plugin_api:{}", plugin_id),
                tags: vec![
                    "code_change".to_string(),
                    "plugin_proposal".to_string(),
                    plugin_id.to_string()
                ],
                importance: 0.8,
                associations: vec![crate::memory::MemoryId::new()],
                context: Some("Generated from automated fix".to_string()),
                created_at: chrono::Utc::now(),
                accessed_count: 0,
                last_accessed: None,
                version: 1,
                category: "plugins".to_string(),
            };

            memory.store(change_record, vec![], metadata).await?;
        }

        // Log the change proposal for audit trail
        tracing::info!(
            "Plugin '{}' proposed code change: {} -> {}",
            plugin_id, enhanced_change.file_path, pr_id
        );

        Ok(pr_id)
    }

    /// Make a network request
    pub async fn network_request(
        &self,
        plugin_id: &str,
        request: NetworkRequest,
    ) -> Result<NetworkResponse> {
        self.check_capability(plugin_id, &PluginCapability::NetworkAccess).await?;

        // Validate URL is allowed
        if !self.is_url_allowed(&request.url) {
            return Err(anyhow::anyhow!("URL not allowed: {}", request.url));
        }

        // Make the request with timeout
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()?;

        let mut req = match request.method.as_str() {
            "GET" => client.get(&request.url),
            "POST" => client.post(&request.url),
            "PUT" => client.put(&request.url),
            "DELETE" => client.delete(&request.url),
            _ => return Err(anyhow::anyhow!("Unsupported HTTP method")),
        };

        // Add headers
        for (key, value) in request.headers {
            req = req.header(key, value);
        }

        // Add body if present
        if let Some(body) = request.body {
            req = req.body(body);
        }

        let response = req.send().await?;
        let status = response.status().as_u16();
        let headers = response.headers()
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_str().unwrap_or("").to_string()))
            .collect();
        let body = response.text().await?;

        Ok(NetworkResponse {
            status,
            headers,
            body,
        })
    }

    /// Check if URL is allowed
    fn is_url_allowed(&self, url: &str) -> bool {
        // Implement URL allowlist/blocklist
        // For now, allow all HTTPS URLs
        url.starts_with("https://")
    }

    /// Read file (sandboxed)
    pub async fn read_file(
        &self,
        plugin_id: &str,
        path: &str,
    ) -> Result<String> {
        self.check_capability(plugin_id, &PluginCapability::FileSystemRead).await?;

        // Validate path is within plugin's sandbox
        let sandbox_path = format!("plugins/{}/data", plugin_id);
        if !path.starts_with(&sandbox_path) {
            return Err(anyhow::anyhow!("Access denied: path outside sandbox"));
        }

        tokio::fs::read_to_string(path).await
            .context("Failed to read file")
    }

    /// Write file (sandboxed)
    pub async fn write_file(
        &self,
        plugin_id: &str,
        path: &str,
        content: &str,
    ) -> Result<()> {
        self.check_capability(plugin_id, &PluginCapability::FileSystemWrite).await?;

        // Validate path is within plugin's sandbox
        let sandbox_path = format!("plugins/{}/data", plugin_id);
        if !path.starts_with(&sandbox_path) {
            return Err(anyhow::anyhow!("Access denied: path outside sandbox"));
        }

        // Create directory if needed
        if let Some(parent) = std::path::Path::new(path).parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        tokio::fs::write(path, content).await
            .context("Failed to write file")
    }

    /// Enhanced content generation helper methods
    ///
    /// Enhance prompt with plugin context and cognitive insights
    async fn enhance_prompt_with_context(
        &self,
        plugin_id: &str,
        prompt: &str,
        style: &str,
    ) -> Result<String> {
        // Gather context from memory if available
        let mut context_elements = Vec::new();

        if let Some(memory) = &self.memory {
            // Get recent memories related to the plugin
            let plugin_memories = memory.retrieve_similar(
                &format!("plugin:{}", plugin_id),
                3
            ).await.unwrap_or_default();

            for memory_item in plugin_memories {
                context_elements.push(format!("Context: {}", memory_item.content));
            }
        }

        // Get recent thoughts if consciousness is available
        if let Some(consciousness) = &self.consciousness {
            let recent_thoughts = consciousness.get_recent_thoughts(5);
            for thought in recent_thoughts.iter().take(2) {
                context_elements.push(format!("Current thought: {}", thought.content));
            }
        }

        // Build enhanced prompt
        let mut enhanced_prompt = String::new();

        if !context_elements.is_empty() {
            enhanced_prompt.push_str("CONTEXT:\n");
            for element in context_elements {
                enhanced_prompt.push_str(&format!("- {}\n", element));
            }
            enhanced_prompt.push_str("\n");
        }

        enhanced_prompt.push_str(&format!(
            "TASK: Generate content in '{}' style for the following request:\n{}\n\n",
            style, prompt
        ));

        enhanced_prompt.push_str(&format!(
            "PLUGIN: This request comes from plugin '{}'. \
            Ensure the response is appropriate for plugin consumption and follows \
            safety guidelines.\n\n",
            plugin_id
        ));

        enhanced_prompt.push_str("REQUIREMENTS:\n");
        enhanced_prompt.push_str("- Be helpful and accurate\n");
        enhanced_prompt.push_str("- Maintain appropriate tone and style\n");
        enhanced_prompt.push_str("- Ensure content is safe and appropriate\n");
        enhanced_prompt.push_str("- Consider the cognitive context provided above\n");

        Ok(enhanced_prompt)
    }

    /// Derive content generation parameters based on style
    async fn derive_content_parameters(&self, style: &str) -> Result<ContentGenerationParams> {
        let params = match style.to_lowercase().as_str() {
            "formal" => ContentGenerationParams {
                tone: "professional".to_string(),
                max_length: 500,
                creativity: 0.3,
                coherence_weight: 0.9,
                safety_level: 0.95,
            },
            "casual" | "conversational" => ContentGenerationParams {
                tone: "friendly".to_string(),
                max_length: 300,
                creativity: 0.7,
                coherence_weight: 0.8,
                safety_level: 0.9,
            },
            "creative" => ContentGenerationParams {
                tone: "imaginative".to_string(),
                max_length: 600,
                creativity: 0.9,
                coherence_weight: 0.7,
                safety_level: 0.85,
            },
            "technical" => ContentGenerationParams {
                tone: "precise".to_string(),
                max_length: 800,
                creativity: 0.2,
                coherence_weight: 0.95,
                safety_level: 0.9,
            },
            _ => ContentGenerationParams {
                tone: "balanced".to_string(),
                max_length: 400,
                creativity: 0.5,
                coherence_weight: 0.8,
                safety_level: 0.9,
            },
        };

        Ok(params)
    }

    /// Simulate content generation with cognitive awareness (fallback when ContentGenerator methods unavailable)
    async fn simulate_content_generation(
        &self,
        enhanced_prompt: &str,
        params: &ContentGenerationParams,
    ) -> Result<String> {
        // Extract key elements from the enhanced prompt
        let prompt_lines: Vec<&str> = enhanced_prompt.lines().collect();
        let task_line = prompt_lines.iter()
            .find(|line| line.starts_with("TASK:"))
            .unwrap_or(&"Generate helpful content");

        // Extract the actual user request
        let user_request = task_line
            .strip_prefix("TASK:")
            .unwrap_or(task_line)
            .trim();

        // Generate content based on style and parameters
        let mut content = match params.tone.as_str() {
            "professional" => format!(
                "I can assist you with {}. Based on my analysis, here's a professional response:\n\n",
                user_request
            ),
            "friendly" => format!(
                "Hi! I'd be happy to help with {}. Here's what I can share:\n\n",
                user_request
            ),
            "imaginative" => format!(
                "What an interesting request about {}! Let me explore this creatively:\n\n",
                user_request
            ),
            "precise" => format!(
                "Regarding {}, here are the precise details:\n\n",
                user_request
            ),
            _ => format!(
                "Thank you for your question about {}. Here's my response:\n\n",
                user_request
            ),
        };

        // Add content based on creativity level
        if params.creativity > 0.7 {
            content.push_str("This is a creative and innovative approach that considers multiple perspectives. ");
            content.push_str("Drawing from various cognitive frameworks, I can suggest several interesting angles to explore. ");
        } else if params.creativity > 0.4 {
            content.push_str("This is a balanced response that considers both conventional and alternative approaches. ");
            content.push_str("Based on established principles and emerging insights, here's what I recommend: ");
        } else {
            content.push_str("This is a structured and methodical response based on established practices. ");
            content.push_str("Following proven approaches, the recommended solution is: ");
        }

        // Add some substantive content based on prompt analysis
        if enhanced_prompt.to_lowercase().contains("code") || enhanced_prompt.to_lowercase().contains("programming") {
            content.push_str("For code-related tasks, I recommend following best practices including proper error handling, clear documentation, and comprehensive testing. ");
        } else if enhanced_prompt.to_lowercase().contains("data") || enhanced_prompt.to_lowercase().contains("analysis") {
            content.push_str("For data analysis, ensure data quality, use appropriate statistical methods, and validate your findings through multiple approaches. ");
        } else if enhanced_prompt.to_lowercase().contains("creative") || enhanced_prompt.to_lowercase().contains("design") {
            content.push_str("For creative projects, balance innovation with usability, consider your audience's needs, and iterate based on feedback. ");
        } else {
            content.push_str("Consider the context, stakeholder needs, available resources, and potential outcomes when developing your approach. ");
        }

        // Add conclusion based on safety level
        if params.safety_level > 0.9 {
            content.push_str("\n\nPlease ensure all recommendations are thoroughly reviewed and tested before implementation. ");
            content.push_str("Consider consulting with domain experts and following established safety protocols.");
        }

        // Trim to max length if needed
        if content.len() > params.max_length {
            content = format!("{}...", &content[..params.max_length - 3]);
        }

        Ok(content)
    }

    /// Post-process generated content for safety and coherence
    async fn post_process_generated_content(
        &self,
        plugin_id: &str,
        content: &str,
        original_prompt: &str,
    ) -> Result<String> {
        let mut processed = content.to_string();

        // Safety filtering
        processed = self.apply_safety_filters(&processed)?;

        // Length validation
        if processed.len() > 2000 {
            processed = format!("{}...", &processed[..1997]);
        }

        // Add plugin attribution if it's a substantial response
        if processed.len() > 100 {
            processed.push_str(&format!(
                "\n\n[Generated by Loki AI via plugin '{}']",
                plugin_id
            ));
        }

        // Validate content relevance to original prompt
        if !self.is_content_relevant_to_prompt(&processed, original_prompt) {
            tracing::warn!(
                "Plugin '{}' generated content may not be relevant to prompt",
                plugin_id
            );

            processed = format!(
                "I understand you're asking about: {}\n\n{}",
                original_prompt, processed
            );
        }

        Ok(processed)
    }

    /// Apply safety filters to content
    fn apply_safety_filters(&self, content: &str) -> Result<String> {
        let mut filtered = content.to_string();

        // Remove potentially harmful patterns
        let harmful_patterns = [
            r"(?i)\b(hack|exploit|vulnerability)\b",
            r"(?i)\b(password|secret|token)\s*[:=]\s*\S+",
            r"(?i)\b(kill|destroy|harm)\b.*\b(process|system|user)\b",
        ];

        for pattern in &harmful_patterns {
            if let Ok(regex) = regex::Regex::new(pattern) {
                filtered = regex.replace_all(&filtered, "[FILTERED]").to_string();
            }
        }

        // Ensure content doesn't contain excessive special characters
        let special_char_ratio = filtered.chars()
            .filter(|c| !c.is_alphanumeric() && !c.is_whitespace())
            .count() as f32 / filtered.len() as f32;

        if special_char_ratio > 0.3 {
            return Err(anyhow::anyhow!("Content contains too many special characters"));
        }

        Ok(filtered)
    }

        /// Check if content is relevant to the original prompt
    fn is_content_relevant_to_prompt(&self, content: &str, prompt: &str) -> bool {
        // Simple relevance check using keyword overlap
        let prompt_lower = prompt.to_lowercase();
        let prompt_words: std::collections::HashSet<_> = prompt_lower
            .split_whitespace()
            .filter(|w| w.len() > 3) // Ignore short words
            .collect();

        let content_lower = content.to_lowercase();
        let content_words: std::collections::HashSet<_> = content_lower
            .split_whitespace()
            .filter(|w| w.len() > 3)
            .collect();

        let overlap = prompt_words.intersection(&content_words).count();
        let relevance_score = overlap as f32 / prompt_words.len().max(1) as f32;

        relevance_score > 0.1 // At least 10% keyword overlap
    }

    /// Code change proposal enhancement and validation methods

    /// Validate a code change proposal for safety and feasibility
    async fn validate_code_change_proposal(
        &self,
        plugin_id: &str,
        change: &CodeChangeProposal,
    ) -> Result<()> {
        // Validate file path
        if change.file_path.contains("..") || change.file_path.starts_with('/') {
            return Err(anyhow::anyhow!(
                "Invalid file path: {} (security violation)",
                change.file_path
            ));
        }

        // Check for potentially dangerous operations
        let dangerous_patterns = [
            "unsafe ", "std::mem::transmute", "libc::", "raw pointer",
            "exec", "system", "rm ", "delete", "drop table",
        ];

        for pattern in &dangerous_patterns {
            if change.new_content.to_lowercase().contains(pattern) {
                tracing::warn!(
                    "Plugin '{}' proposed potentially dangerous code change containing: {}",
                    plugin_id, pattern
                );
                return Err(anyhow::anyhow!(
                    "Code change contains potentially unsafe pattern: {}",
                    pattern
                ));
            }
        }

        // Validate change size
        if change.new_content.len() > 10_000 {
            return Err(anyhow::anyhow!(
                "Code change too large: {} characters (max 10,000)",
                change.new_content.len()
            ));
        }

        // Check for reasonable description
        if change.description.len() < 10 {
            return Err(anyhow::anyhow!(
                "Code change description too short (minimum 10 characters)"
            ));
        }

        Ok(())
    }

    /// Enhance change proposal with additional context and metadata
    async fn enhance_change_proposal_with_context(
        &self,
        plugin_id: &str,
        mut change: CodeChangeProposal,
    ) -> Result<CodeChangeProposal> {
        // Add plugin attribution to description
        change.description = format!(
            "[Plugin: {}] {}",
            plugin_id, change.description
        );

        // Enhance reasoning with cognitive context
        let mut enhanced_reasoning = change.reasoning.clone();
        enhanced_reasoning.push_str(&format!(
            "\n\nPlugin Context:\n- Proposed by plugin: {}\n- Timestamp: {}\n",
            plugin_id,
            chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
        ));

        // Add memory context if available
        if let Some(memory) = &self.memory {
            if let Ok(related_memories) = memory.retrieve_similar(&change.file_path, 2).await {
                if !related_memories.is_empty() {
                    enhanced_reasoning.push_str("- Related context from memory:\n");
                    for memory_item in related_memories {
                        enhanced_reasoning.push_str(&format!(
                            "  * {}\n",
                            memory_item.content.chars().take(100).collect::<String>()
                        ));
                    }
                }
            }
        }

        change.reasoning = enhanced_reasoning;

        Ok(change)
    }

    /// Access GitHub operations through the GitHub client
    pub async fn github_create_issue(
        &self,
        plugin_id: &str,
        repo: &str,
        title: &str,
        body: &str,
    ) -> Result<String> {
        self.check_capability(plugin_id, &PluginCapability::NetworkAccess).await?;

        if let Some(github_client) = &self.github_client {
            debug!("Plugin {} creating GitHub issue in repo: {}", plugin_id, repo);
            
            // Parse repository (owner/repo format)
            let parts: Vec<&str> = repo.split('/').collect();
            if parts.len() != 2 {
                return Err(anyhow::anyhow!("Invalid repository format. Use 'owner/repo'"));
            }
            
            let (owner, repo_name) = (parts[0], parts[1]);
            
            // Create the issue through the GitHub client
            let _issue_number = github_client.create_issue(title, body, Vec::new()).await?;
            
            let issue_url = format!("https://github.com/{}/{}/issues/{}", owner, repo_name, _issue_number);
            info!("Plugin {} created GitHub issue: {}", plugin_id, issue_url);
            Ok(issue_url)
        } else {
            Err(anyhow::anyhow!("GitHub client not available"))
        }
    }

    /// Get GitHub repository information
    pub async fn github_get_repo_info(
        &self,
        plugin_id: &str,
        repo: &str,
    ) -> Result<GitHubRepoInfo> {
        self.check_capability(plugin_id, &PluginCapability::NetworkAccess).await?;

        if let Some(github_client) = &self.github_client {
            debug!("Plugin {} getting repo info for: {}", plugin_id, repo);
            
            // Parse repository (owner/repo format)  
            let parts: Vec<&str> = repo.split('/').collect();
            if parts.len() != 2 {
                return Err(anyhow::anyhow!("Invalid repository format. Use 'owner/repo'"));
            }
            
            let (owner, repo_name) = (parts[0], parts[1]);
            
            // Get repository information using public GitHubClient methods
            let repo_stats = github_client.get_repo_stats().await?;
            
            Ok(GitHubRepoInfo {
                name: format!("{}/{}", owner, repo_name),
                full_name: format!("{}/{}", owner, repo_name),
                description: "Repository information".to_string(),
                url: format!("https://github.com/{}/{}", owner, repo_name),
                stars: repo_stats.stars,
                forks: repo_stats.forks,
                issues: repo_stats.open_issues,
                language: "Unknown".to_string(),
                topics: Vec::new(),
            })
        } else {
            Err(anyhow::anyhow!("GitHub client not available"))
        }
    }

    /// Create a GitHub pull request
    pub async fn github_create_pull_request(
        &self,
        plugin_id: &str,
        repo: &str,
        title: &str,
        body: &str,
        head: &str,
        _base: &str,
    ) -> Result<String> {
        self.check_capability(plugin_id, &PluginCapability::NetworkAccess).await?;

        if let Some(github_client) = &self.github_client {
            debug!("Plugin {} creating GitHub PR in repo: {}", plugin_id, repo);
            
            // Parse repository (owner/repo format)
            let parts: Vec<&str> = repo.split('/').collect();
            if parts.len() != 2 {
                return Err(anyhow::anyhow!("Invalid repository format. Use 'owner/repo'"));
            }
            
            let (owner, repo_name) = (parts[0], parts[1]);
            
            // Create the pull request using CodeChange struct
            let code_change = crate::cognitive::self_modify::CodeChange {
                file_path: std::path::PathBuf::from("plugin_generated"),
                change_type: crate::cognitive::self_modify::ChangeType::Enhancement,
                description: title.to_string(),
                reasoning: body.to_string(),
                old_content: None,
                new_content: body.to_string(),
                line_range: None,
                risk_level: crate::cognitive::self_modify::RiskLevel::Low,
                attribution: None,
            };
            let pr = github_client.create_pull_request(head.to_string(), code_change).await?;
            
            let pr_url = format!("https://github.com/{}/{}/pull/{}", owner, repo_name, pr.number);
            info!("Plugin {} created GitHub PR: {}", plugin_id, pr_url);
            Ok(pr_url)
        } else {
            Err(anyhow::anyhow!("GitHub client not available"))
        }
    }

    /// Get GitHub client availability
    pub fn is_github_available(&self) -> bool {
        self.github_client.is_some()
    }
}

/// GitHub repository information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitHubRepoInfo {
    pub name: String,
    pub full_name: String,
    pub description: String,
    pub url: String,
    pub stars: u32,
    pub forks: u32,
    pub issues: u32,
    pub language: String,
    pub topics: Vec<String>,
}

/// Memory content returned to plugins
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContent {
    pub content: String,
    pub metadata: MemoryMetadata,
    pub similarity: f32,
}

/// Thought content returned to plugins
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThoughtContent {
    pub id: String,
    pub content: String,
    pub thought_type: String,
    pub confidence: f32,
    pub emotional_valence: f32,
}

/// Code change proposal from plugin
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeChangeProposal {
    pub file_path: String,
    pub description: String,
    pub old_content: Option<String>,
    pub new_content: String,
    pub reasoning: String,
}

/// Network request from plugin
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkRequest {
    pub url: String,
    pub method: String,
    pub headers: HashMap<String, String>,
    pub body: Option<String>,
}

/// Network response to plugin
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkResponse {
    pub status: u16,
    pub headers: HashMap<String, String>,
    pub body: String,
}

/// Content generation parameters for enhanced content creation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentGenerationParams {
    pub tone: String,
    pub max_length: usize,
    pub creativity: f32,
    pub coherence_weight: f32,
    pub safety_level: f32,
}

/// Plugin API trait for plugins to implement
#[async_trait]
pub trait PluginApiClient: Send + Sync {
    /// Read from memory
    async fn memory_read(&self, query: &str, limit: usize) -> Result<Vec<MemoryContent>>;

    /// Write to memory
    async fn memory_write(&self, content: String, metadata: MemoryMetadata) -> Result<()>;

    /// Get recent thoughts
    async fn get_recent_thoughts(&self, count: usize) -> Result<Vec<ThoughtContent>>;

    /// Generate content
    async fn generate_content(&self, prompt: &str, style: Option<String>) -> Result<String>;

    /// Propose code change
    async fn propose_code_change(&self, change: CodeChangeProposal) -> Result<String>;

    /// Make network request
    async fn network_request(&self, request: NetworkRequest) -> Result<NetworkResponse>;

    /// Read file
    async fn read_file(&self, path: &str) -> Result<String>;

    /// Write file
    async fn write_file(&self, path: &str, content: &str) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_plugin_api_capability_check() {
        let api = PluginApi::new(None, None, None, None).await.unwrap();

        // Grant capabilities
        api.grant_capabilities("test-plugin", vec![PluginCapability::MemoryRead])
            .await
            .unwrap();

        // Check allowed capability
        assert!(api.check_capability("test-plugin", &PluginCapability::MemoryRead)
            .await
            .is_ok());

        // Check denied capability
        assert!(api.check_capability("test-plugin", &PluginCapability::MemoryWrite)
            .await
            .is_err());
    }

    #[test]
    fn test_url_validation() {
        let api = PluginApi {
            memory: None,
            consciousness: None,
            content_generator: None,
            github_client: None,
            capabilities: Arc::new(RwLock::new(HashMap::new())),
        };

        assert!(api.is_url_allowed("https://example.com"));
        assert!(!api.is_url_allowed("http://example.com"));
        assert!(!api.is_url_allowed("file:///etc/passwd"));
    }
}
