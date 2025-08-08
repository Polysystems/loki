use std::time::Instant;
use std::path::Path;

use anyhow::Result;
use base64::Engine as _;
use base64::engine::general_purpose;
use chrono::Utc;
use reqwest::Client;
use tokio::fs;
use tracing::{info, warn};
use uuid::Uuid;

use crate::tools::creative_media::{
    AudioGenre,
    CreativeMediaConfig,
    GeneratedMedia,
    ImageStyle,
    MediaMetadata,
    VideoStyle,
    VoiceEmotion,
    VoiceStyle,
};

/// Video generator using Pika Labs and Runway APIs
#[derive(Debug)]
pub struct VideoGenerator {
    client: Client,
    pika_api_key: Option<String>,
    runway_api_key: Option<String>,
}

impl VideoGenerator {
    pub async fn new(config: &CreativeMediaConfig, client: Client) -> Result<Self> {
        Ok(Self {
            client,
            pika_api_key: config.pika_api_key.clone(),
            runway_api_key: config.runway_api_key.clone(),
        })
    }

    pub async fn generate(
        &self,
        task_id: &str,
        prompt: &str,
        duration: u32,
        style: VideoStyle,
        config: &CreativeMediaConfig,
    ) -> Result<GeneratedMedia> {
        let start_time = Instant::now();
        info!("Generating video with style {:?}: {}", style, prompt);

        // Enhanced prompt with style
        let enhanced_prompt = self.enhance_prompt_for_style(prompt, style);

        // Generate video content
        let file_path = config.media_storage_path.join(format!("{}_video.mp4", task_id));
        
        // Check if we have API keys for video generation services
        if let Some(api_key) = &self.runway_api_key {
            // Use Runway ML Gen-2 for video generation
            match self.generate_with_runway(&enhanced_prompt, duration, api_key).await {
                Ok(video_data) => {
                    fs::write(&file_path, video_data).await?;
                }
                Err(e) => {
                    warn!("Runway generation failed: {}. Creating placeholder.", e);
                    // Fallback to placeholder
                    let placeholder_path = file_path.with_extension("txt");
                    fs::write(&placeholder_path, format!("Video placeholder: {}\nError: {}", enhanced_prompt, e)).await?;
                    return Ok(self.create_placeholder_media(task_id, prompt, style, duration, start_time));
                }
            }
        } else if let Some(api_key) = &self.pika_api_key {
            // Use Pika Labs for video generation
            match self.generate_with_pika(&enhanced_prompt, duration, api_key).await {
                Ok(video_data) => {
                    fs::write(&file_path, video_data).await?;
                }
                Err(e) => {
                    warn!("Pika generation failed: {}. Creating placeholder.", e);
                    // Fallback to placeholder
                    let placeholder_path = file_path.with_extension("txt");
                    fs::write(&placeholder_path, format!("Video placeholder: {}\nError: {}", enhanced_prompt, e)).await?;
                    return Ok(self.create_placeholder_media(task_id, prompt, style, duration, start_time));
                }
            }
        } else {
            // No API keys available, create a demo video file
            info!("No video generation API keys found. Creating demo video.");
            
            // Create a simple test pattern video using FFmpeg (if available)
            let demo_created = self.create_demo_video(&file_path, duration).await;
            
            if !demo_created {
                // Final fallback to text placeholder
                let placeholder_path = file_path.with_extension("txt");
                fs::write(&placeholder_path, format!("Video placeholder: {}\nNote: Configure RUNWAY_API_KEY or PIKA_API_KEY for actual video generation", enhanced_prompt)).await?;
                return Ok(self.create_placeholder_media(task_id, prompt, style, duration, start_time));
            }
        }

        let media = GeneratedMedia {
            id: Uuid::new_v4().to_string(),
            media_type: "video".to_string(),
            file_path: file_path.to_string_lossy().to_string(),
            prompt: prompt.to_string(),
            metadata: MediaMetadata {
                provider: "Video Generator".to_string(),
                model: "video-gen-v1".to_string(),
                quality_score: self.estimate_quality_score(style),
                creative_score: self.estimate_creativity_score(prompt),
                file_size_bytes: enhanced_prompt.len() as u64,
                dimensions: Some((1920, 1080)),
                duration_seconds: Some(duration),
                tags: vec![format!("{:?}", style), "video".to_string()],
            },
            created_at: Utc::now(),
            cognitive_context: Some(format!("Generated {} style video", format!("{:?}", style))),
            generation_time_ms: start_time.elapsed().as_millis() as u64,
        };

        Ok(media)
    }

    fn enhance_prompt_for_style(&self, prompt: &str, style: VideoStyle) -> String {
        let style_modifier = match style {
            VideoStyle::Cinematic => "cinematic, film quality, dramatic lighting",
            VideoStyle::Documentary => "documentary style, realistic, natural lighting",
            VideoStyle::Artistic => "artistic, creative, stylized",
            VideoStyle::Technical => "technical demonstration, clear, informative",
            VideoStyle::Presentation => "presentation style, clean, professional",
            VideoStyle::Abstract => "abstract, surreal, artistic interpretation",
            VideoStyle::Anime => "anime style, japanese animation",
            VideoStyle::Photorealistic => "photorealistic, hyperrealistic, detailed",
        };

        format!("{}, {}", prompt, style_modifier)
    }

    fn estimate_quality_score(&self, style: VideoStyle) -> f32 {
        match style {
            VideoStyle::Cinematic => 0.9,
            VideoStyle::Photorealistic => 0.85,
            VideoStyle::Documentary => 0.8,
            VideoStyle::Technical => 0.75,
            VideoStyle::Artistic => 0.8,
            VideoStyle::Presentation => 0.7,
            VideoStyle::Abstract => 0.7,
            VideoStyle::Anime => 0.75,
        }
    }

    fn estimate_creativity_score(&self, prompt: &str) -> f32 {
        let creative_words = ["artistic", "abstract", "creative", "unique", "experimental"];
        let creative_count =
            creative_words.iter().filter(|&word| prompt.to_lowercase().contains(word)).count()
                as f32;

        (0.5 + (creative_count / creative_words.len() as f32) * 0.5).min(1.0)
    }
    
    async fn generate_with_runway(&self, prompt: &str, duration: u32, api_key: &str) -> Result<Vec<u8>> {
        // Runway ML Gen-2 API integration
        let response = self.client
            .post("https://api.runwayml.com/v1/generate")
            .bearer_auth(api_key)
            .json(&serde_json::json!({
                "prompt": prompt,
                "duration_seconds": duration,
                "model": "gen-2",
                "resolution": "1920x1080",
                "fps": 24
            }))
            .send()
            .await?;
            
        if !response.status().is_success() {
            return Err(anyhow::anyhow!("Runway API error: {}", response.status()));
        }
        
        // Get generation ID and poll for completion
        let gen_response: serde_json::Value = response.json().await?;
        let generation_id = gen_response["id"].as_str()
            .ok_or_else(|| anyhow::anyhow!("No generation ID in response"))?;
            
        // Poll for completion (simplified - in production would need proper polling)
        tokio::time::sleep(tokio::time::Duration::from_secs(30)).await;
        
        // Download the generated video
        let video_response = self.client
            .get(format!("https://api.runwayml.com/v1/generations/{}/download", generation_id))
            .bearer_auth(api_key)
            .send()
            .await?;
            
        Ok(video_response.bytes().await?.to_vec())
    }
    
    async fn generate_with_pika(&self, prompt: &str, duration: u32, api_key: &str) -> Result<Vec<u8>> {
        // Pika Labs API integration
        let response = self.client
            .post("https://api.pika.art/generate/video")
            .header("Authorization", format!("Bearer {}", api_key))
            .json(&serde_json::json!({
                "prompt": prompt,
                "duration": duration,
                "width": 1920,
                "height": 1080,
                "guidance_scale": 7.5
            }))
            .send()
            .await?;
            
        if !response.status().is_success() {
            return Err(anyhow::anyhow!("Pika API error: {}", response.status()));
        }
        
        let result: serde_json::Value = response.json().await?;
        let video_url = result["video_url"].as_str()
            .ok_or_else(|| anyhow::anyhow!("No video URL in response"))?;
            
        // Download the video
        let video_data = self.client.get(video_url).send().await?.bytes().await?;
        Ok(video_data.to_vec())
    }
    
    async fn create_demo_video(&self, file_path: &Path, duration: u32) -> bool {
        // Try to create a simple demo video using FFmpeg
        match tokio::process::Command::new("ffmpeg")
            .args(&[
                "-f", "lavfi",
                "-i", &format!("testsrc=duration={}:size=1920x1080:rate=24", duration),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-y",
                file_path.to_str().unwrap_or("output.mp4")
            ])
            .output()
            .await
        {
            Ok(output) => output.status.success(),
            Err(_) => false,
        }
    }
    
    fn create_placeholder_media(
        &self,
        task_id: &str,
        prompt: &str,
        style: VideoStyle,
        duration: u32,
        start_time: Instant,
    ) -> GeneratedMedia {
        GeneratedMedia {
            id: Uuid::new_v4().to_string(),
            media_type: "video_placeholder".to_string(),
            file_path: format!("{}_video_placeholder.txt", task_id),
            prompt: prompt.to_string(),
            metadata: MediaMetadata {
                provider: "Placeholder".to_string(),
                model: "none".to_string(),
                quality_score: 0.0,
                creative_score: 0.0,
                file_size_bytes: 0,
                dimensions: Some((1920, 1080)),
                duration_seconds: Some(duration),
                tags: vec![format!("{:?}", style), "placeholder".to_string()],
            },
            created_at: Utc::now(),
            cognitive_context: Some("Placeholder video - API keys not configured".to_string()),
            generation_time_ms: start_time.elapsed().as_millis() as u64,
        }
    }
}

/// Image generator using OpenAI DALL-E, Midjourney, and Stability AI
#[derive(Debug)]
pub struct ImageGenerator {
    client: Client,
    openai_api_key: Option<String>,
    midjourney_api_key: Option<String>,
    stability_api_key: Option<String>,
}

impl ImageGenerator {
    pub async fn new(config: &CreativeMediaConfig, client: Client) -> Result<Self> {
        Ok(Self {
            client,
            openai_api_key: config.openai_api_key.clone(),
            midjourney_api_key: config.midjourney_api_key.clone(),
            stability_api_key: config.stability_api_key.clone(),
        })
    }

    pub async fn generate(
        &self,
        task_id: &str,
        prompt: &str,
        style: ImageStyle,
        _reference_image: Option<&str>,
        config: &CreativeMediaConfig,
    ) -> Result<GeneratedMedia> {
        let start_time = Instant::now();
        info!("Generating image with style {:?}: {}", style, prompt);

        let enhanced_prompt = self.enhance_prompt_for_style(prompt, style);

        // Try actual API generation first, fallback to intelligent simulation
        let (file_path, actual_file_size, generation_method) = if let Some(_api_key) =
            &self.openai_api_key
        {
            match self.generate_with_openai(task_id, &enhanced_prompt, config).await {
                Ok((path, size)) => (path, size, "OpenAI DALL-E"),
                Err(e) => {
                    tracing::warn!("OpenAI generation failed: {}, using simulation", e);
                    self.generate_simulated_image(task_id, &enhanced_prompt, style, config).await?
                }
            }
        } else if let Some(_api_key) = &self.stability_api_key {
            match self.generate_with_stability_ai(task_id, &enhanced_prompt, config).await {
                Ok((path, size)) => (path, size, "Stability AI"),
                Err(e) => {
                    tracing::warn!("Stability AI generation failed: {}, using simulation", e);
                    self.generate_simulated_image(task_id, &enhanced_prompt, style, config).await?
                }
            }
        } else {
            // No API keys available, use intelligent simulation
            self.generate_simulated_image(task_id, &enhanced_prompt, style, config).await?
        };

        let media = GeneratedMedia {
            id: Uuid::new_v4().to_string(),
            media_type: "image".to_string(),
            file_path: file_path.to_string_lossy().to_string(),
            prompt: prompt.to_string(),
            metadata: MediaMetadata {
                provider: generation_method.to_string(),
                model: "image-gen-v1".to_string(),
                quality_score: self.estimate_quality_score(style),
                creative_score: self.estimate_creativity_score(prompt),
                file_size_bytes: actual_file_size,
                dimensions: Some((1024, 1024)),
                duration_seconds: None,
                tags: vec![format!("{:?}", style), "image".to_string()],
            },
            created_at: Utc::now(),
            cognitive_context: Some(format!(
                "Generated {} style image using {}",
                format!("{:?}", style),
                generation_method
            )),
            generation_time_ms: start_time.elapsed().as_millis() as u64,
        };

        Ok(media)
    }

    /// Generate image using OpenAI DALL-E API
    async fn generate_with_openai(
        &self,
        task_id: &str,
        prompt: &str,
        config: &CreativeMediaConfig,
    ) -> Result<(std::path::PathBuf, u64)> {
        use serde_json::json;

        let api_key = self.openai_api_key.as_ref().unwrap();

        let request_body = json!({
            "prompt": prompt,
            "n": 1,
            "size": "1024x1024",
            "response_format": "url"
        });

        let response = self
            .client
            .post("https://api.openai.com/v1/images/generations")
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(anyhow::anyhow!("OpenAI API error: {}", error_text));
        }

        let response_json: serde_json::Value = response.json().await?;
        let image_url = response_json["data"][0]["url"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("No image URL in response"))?;

        // Download the generated image
        let image_response = self.client.get(image_url).send().await?;
        let image_bytes = image_response.bytes().await?;

        let file_path = config.media_storage_path.join(format!("{}_image_dalle.png", task_id));

        fs::write(&file_path, &image_bytes).await?;

        Ok((file_path, image_bytes.len() as u64))
    }

    /// Generate image using Stability AI API
    async fn generate_with_stability_ai(
        &self,
        task_id: &str,
        prompt: &str,
        config: &CreativeMediaConfig,
    ) -> Result<(std::path::PathBuf, u64)> {
        use serde_json::json;

        let api_key = self.stability_api_key.as_ref().unwrap();

        let request_body = json!({
            "text_prompts": [
                {
                    "text": prompt,
                    "weight": 1.0
                }
            ],
            "cfg_scale": 7,
            "height": 1024,
            "width": 1024,
            "samples": 1,
            "steps": 30
        });

        let response = self
            .client
            .post("https://api.stability.ai/v1/generation/stable-diffusion-v1-6/text-to-image")
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(anyhow::anyhow!("Stability AI API error: {}", error_text));
        }

        let response_json: serde_json::Value = response.json().await?;
        let image_base64 = response_json["artifacts"][0]["base64"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("No image data in response"))?;

        // Decode base64 image
        let image_bytes = general_purpose::STANDARD.decode(image_base64)?;

        let file_path = config.media_storage_path.join(format!("{}_image_stability.png", task_id));

        fs::write(&file_path, &image_bytes).await?;

        Ok((file_path, image_bytes.len() as u64))
    }

    /// Generate simulated image with intelligent content
    async fn generate_simulated_image(
        &self,
        task_id: &str,
        prompt: &str,
        style: ImageStyle,
        config: &CreativeMediaConfig,
    ) -> Result<(std::path::PathBuf, u64, &'static str)> {
        // Create an SVG image with the prompt and style information
        let svg_content = self.generate_svg_representation(prompt, style);

        let file_path = config.media_storage_path.join(format!("{}_image_simulated.svg", task_id));

        fs::write(&file_path, &svg_content).await?;

        Ok((file_path, svg_content.len() as u64, "Simulated Generator"))
    }

    /// Generate SVG representation of the image concept
    fn generate_svg_representation(&self, prompt: &str, style: ImageStyle) -> String {
        // Analyze prompt for visual elements
        let (primary_color, secondary_color) = self.extract_colors_from_prompt(prompt);
        let shapes = self.generate_shapes_from_prompt(prompt, style);
        let title = prompt.chars().take(50).collect::<String>();

        format!(
            r#"<svg viewBox="0 0 400 400" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:{};stop-opacity:1" />
      <stop offset="100%" style="stop-color:{};stop-opacity:1" />
    </linearGradient>
  </defs>

  <!-- Background -->
  <rect width="400" height="400" fill="url(#bg)" />

  <!-- Generated shapes based on prompt -->
  {}

  <!-- Style indicator -->
  <text x="20" y="30" font-family="Arial" font-size="14" fill="white" opacity="0.8">{:?}</text>

  <!-- Prompt text -->
  <text x="20" y="370" font-family="Arial" font-size="12" fill="white" opacity="0.9">{}</text>

  <!-- AI generation indicator -->
  <circle cx="370" cy="30" r="15" fill="rgba(255,255,255,0.3)" />
  <text x="370" y="35" font-family="Arial" font-size="10" fill="white" text-anchor="middle">AI</text>
</svg>"#,
            primary_color, secondary_color, shapes, style, title
        )
    }

    /// Extract colors based on prompt content
    fn extract_colors_from_prompt(&self, prompt: &str) -> (String, String) {
        let prompt_lower = prompt.to_lowercase();

        // Simple color extraction based on keywords
        let primary_color = if prompt_lower.contains("red") {
            "#e74c3c"
        } else if prompt_lower.contains("blue") {
            "#3498db"
        } else if prompt_lower.contains("green") {
            "#2ecc71"
        } else if prompt_lower.contains("yellow") {
            "#f1c40f"
        } else if prompt_lower.contains("purple") {
            "#9b59b6"
        } else if prompt_lower.contains("orange") {
            "#e67e22"
        } else if prompt_lower.contains("dark") {
            "#2c3e50"
        } else if prompt_lower.contains("light") {
            "#ecf0f1"
        } else {
            "#34495e"
        }; // Default

        let secondary_color = if primary_color == "#e74c3c" {
            "#c0392b"
        } else if primary_color == "#3498db" {
            "#2980b9"
        } else if primary_color == "#2ecc71" {
            "#27ae60"
        } else {
            "#2c3e50"
        };

        (primary_color.to_string(), secondary_color.to_string())
    }

    /// Generate SVG shapes based on prompt analysis
    fn generate_shapes_from_prompt(&self, prompt: &str, style: ImageStyle) -> String {
        let prompt_lower = prompt.to_lowercase();
        let mut shapes = Vec::new();

        // Shape generation based on content and style
        match style {
            ImageStyle::Abstract => {
                shapes.push(r#"<circle cx="150" cy="150" r="50" fill="rgba(255,255,255,0.6)" />"#);
                shapes.push(r#"<polygon points="200,100 300,150 250,250 150,200" fill="rgba(255,255,255,0.4)" />"#);
            }
            ImageStyle::Technical | ImageStyle::Diagram => {
                shapes.push(r#"<rect x="100" y="100" width="200" height="100" fill="none" stroke="white" stroke-width="2" />"#);
                shapes.push(r#"<line x1="50" y1="150" x2="350" y2="150" stroke="white" stroke-width="1" />"#);
                shapes.push(r#"<circle cx="200" cy="150" r="30" fill="none" stroke="white" stroke-width="2" />"#);
            }
            ImageStyle::Logo => {
                shapes.push(r#"<circle cx="200" cy="200" r="80" fill="rgba(255,255,255,0.8)" />"#);
                shapes.push(r#"<text x="200" y="210" font-family="Arial" font-size="24" fill="black" text-anchor="middle">LOGO</text>"#);
            }
            _ => {
                // Add contextual shapes based on prompt
                if prompt_lower.contains("mountain") || prompt_lower.contains("peak") {
                    shapes.push(r#"<polygon points="100,300 200,100 300,300" fill="rgba(255,255,255,0.7)" />"#);
                }
                if prompt_lower.contains("circle") || prompt_lower.contains("sun") {
                    shapes.push(
                        r#"<circle cx="300" cy="100" r="40" fill="rgba(255,255,255,0.8)" />"#,
                    );
                }
                if prompt_lower.contains("tree") || prompt_lower.contains("forest") {
                    shapes.push(r#"<rect x="180" y="250" width="40" height="80" fill="rgba(139,69,19,0.8)" />"#);
                    shapes
                        .push(r#"<circle cx="200" cy="220" r="50" fill="rgba(34,139,34,0.8)" />"#);
                }
                if prompt_lower.contains("building") || prompt_lower.contains("city") {
                    shapes.push(r#"<rect x="120" y="200" width="60" height="120" fill="rgba(255,255,255,0.6)" />"#);
                    shapes.push(r#"<rect x="200" y="180" width="80" height="140" fill="rgba(255,255,255,0.7)" />"#);
                }
            }
        }

        shapes.join("\n  ")
    }

    fn estimate_quality_score(&self, style: ImageStyle) -> f32 {
        match style {
            ImageStyle::Photorealistic => 0.9,
            ImageStyle::Concept => 0.85,
            ImageStyle::Artistic => 0.8,
            ImageStyle::Portrait => 0.85,
            ImageStyle::Landscape => 0.8,
            ImageStyle::Technical => 0.75,
            ImageStyle::Logo => 0.8,
            ImageStyle::Diagram => 0.7,
            ImageStyle::Wireframe => 0.6,
            ImageStyle::Abstract => 0.75,
        }
    }

    fn enhance_prompt_for_style(&self, prompt: &str, style: ImageStyle) -> String {
        let style_modifier = match style {
            ImageStyle::Photorealistic => "photorealistic, high detail",
            ImageStyle::Artistic => "artistic, creative interpretation",
            ImageStyle::Concept => "concept art, detailed illustration",
            ImageStyle::Technical => "technical diagram, clear, precise",
            ImageStyle::Abstract => "abstract art, creative",
            ImageStyle::Diagram => "diagram, schematic, clean lines",
            ImageStyle::Wireframe => "wireframe, technical sketch",
            ImageStyle::Logo => "logo design, clean, memorable",
            ImageStyle::Portrait => "portrait, detailed facial features",
            ImageStyle::Landscape => "landscape, scenic",
        };

        format!("{}, {}", prompt, style_modifier)
    }

    fn estimate_creativity_score(&self, prompt: &str) -> f32 {
        let creative_words = ["artistic", "abstract", "creative", "unique", "experimental"];
        let creative_count =
            creative_words.iter().filter(|&word| prompt.to_lowercase().contains(word)).count()
                as f32;

        (0.6 + (creative_count / creative_words.len() as f32) * 0.4).min(1.0)
    }
}

/// Voice generator using ElevenLabs and OpenAI Whisper
#[derive(Debug)]
pub struct VoiceGenerator {
    client: Client,
    elevenlabs_api_key: Option<String>,
    openai_api_key: Option<String>,
}

impl VoiceGenerator {
    pub async fn new(config: &CreativeMediaConfig, client: Client) -> Result<Self> {
        Ok(Self {
            client,
            elevenlabs_api_key: config.elevenlabs_api_key.clone(),
            openai_api_key: config.openai_api_key.clone(),
        })
    }

    pub async fn generate(
        &self,
        task_id: &str,
        text: &str,
        emotion: VoiceEmotion,
        style: VoiceStyle,
        config: &CreativeMediaConfig,
    ) -> Result<GeneratedMedia> {
        let start_time = Instant::now();
        info!("Generating voice with emotion {:?} and style {:?}", emotion, style);

        // Generate voice audio
        let file_path = config.media_storage_path.join(format!("{}_voice.mp3", task_id));
        
        // Select voice based on emotion and style
        let voice_id = self.select_voice_id(emotion, style);
        
        // Try ElevenLabs first
        if let Some(api_key) = &self.elevenlabs_api_key {
            match self.generate_with_elevenlabs(text, &voice_id, emotion, api_key).await {
                Ok(audio_data) => {
                    fs::write(&file_path, audio_data).await?;
                }
                Err(e) => {
                    warn!("ElevenLabs generation failed: {}. Trying OpenAI.", e);
                    
                    // Try OpenAI TTS as fallback
                    if let Some(openai_key) = &self.openai_api_key {
                        match self.generate_with_openai(text, &voice_id, openai_key).await {
                            Ok(audio_data) => {
                                fs::write(&file_path, audio_data).await?;
                            }
                            Err(e2) => {
                                warn!("OpenAI TTS also failed: {}. Creating placeholder.", e2);
                                return Ok(self.create_placeholder_voice(task_id, text, emotion, style, config, start_time).await?);
                            }
                        }
                    } else {
                        return Ok(self.create_placeholder_voice(task_id, text, emotion, style, config, start_time).await?);
                    }
                }
            }
        } else if let Some(api_key) = &self.openai_api_key {
            // No ElevenLabs key, try OpenAI directly
            match self.generate_with_openai(text, &voice_id, api_key).await {
                Ok(audio_data) => {
                    fs::write(&file_path, audio_data).await?;
                }
                Err(e) => {
                    warn!("OpenAI TTS failed: {}. Creating placeholder.", e);
                    return Ok(self.create_placeholder_voice(task_id, text, emotion, style, config, start_time).await?);
                }
            }
        } else {
            // No API keys available, create demo audio
            info!("No voice generation API keys found. Creating demo audio.");
            
            // Try to generate with espeak or similar TTS engine
            if !self.create_demo_audio(&file_path, text).await {
                return Ok(self.create_placeholder_voice(task_id, text, emotion, style, config, start_time).await?);
            }
        }

        let duration = self.estimate_audio_duration(text);

        let media = GeneratedMedia {
            id: Uuid::new_v4().to_string(),
            media_type: "voice".to_string(),
            file_path: file_path.to_string_lossy().to_string(),
            prompt: text.to_string(),
            metadata: MediaMetadata {
                provider: "Voice Generator".to_string(),
                model: config.default_voice_model.clone(),
                quality_score: 0.9,
                creative_score: self.estimate_voice_creativity(emotion, style),
                file_size_bytes: text.len() as u64,
                dimensions: None,
                duration_seconds: Some(duration),
                tags: vec![format!("{:?}", emotion), format!("{:?}", style), "voice".to_string()],
            },
            created_at: Utc::now(),
            cognitive_context: Some(format!("Generated {:?} {:?} voice", emotion, style)),
            generation_time_ms: start_time.elapsed().as_millis() as u64,
        };

        Ok(media)
    }

    pub async fn generate_audio(
        &self,
        task_id: &str,
        prompt: &str,
        duration: u32,
        genre: AudioGenre,
        config: &CreativeMediaConfig,
    ) -> Result<GeneratedMedia> {
        let audio_description = format!(
            "Audio: {} | Genre: {:?} | Duration: {}s | Prompt: {}",
            task_id, genre, duration, prompt
        );

        let file_path =
            config.media_storage_path.join(format!("{}_audio_placeholder.txt", task_id));

        fs::write(&file_path, &audio_description).await?;

        let media = GeneratedMedia {
            id: Uuid::new_v4().to_string(),
            media_type: "audio".to_string(),
            file_path: file_path.to_string_lossy().to_string(),
            prompt: prompt.to_string(),
            metadata: MediaMetadata {
                provider: "Audio Generator".to_string(),
                model: "audio-gen-v1".to_string(),
                quality_score: 0.7,
                creative_score: 0.8,
                file_size_bytes: audio_description.len() as u64,
                dimensions: None,
                duration_seconds: Some(duration),
                tags: vec![format!("{:?}", genre), "audio".to_string()],
            },
            created_at: Utc::now(),
            cognitive_context: Some(format!("Generated {:?} audio", genre)),
            generation_time_ms: 0,
        };

        Ok(media)
    }

    fn estimate_audio_duration(&self, text: &str) -> u32 {
        // Rough estimation: ~150 words per minute, ~5 chars per word
        let char_count = text.len();
        let estimated_seconds = (char_count as f32 / 5.0 / 150.0 * 60.0) as u32;
        estimated_seconds.max(1)
    }

    fn estimate_voice_creativity(&self, emotion: VoiceEmotion, style: VoiceStyle) -> f32 {
        let emotion_creativity = match emotion {
            VoiceEmotion::Mysterious | VoiceEmotion::Mischievous => 0.9,
            VoiceEmotion::Playful | VoiceEmotion::Excited => 0.8,
            VoiceEmotion::Contemplative => 0.7,
            _ => 0.6,
        };

        let style_creativity = match style {
            VoiceStyle::Dramatic | VoiceStyle::Narrative => 0.8,
            VoiceStyle::Whisper => 0.9,
            _ => 0.6,
        };

        (emotion_creativity + style_creativity) / 2.0
    }

    fn select_voice_id(&self, emotion: VoiceEmotion, style: VoiceStyle) -> String {
        // Map emotion and style to specific voice IDs
        match (emotion, style) {
            (VoiceEmotion::Happy, VoiceStyle::Conversational) => "voice_happy_conversational",
            (VoiceEmotion::Sad, VoiceStyle::Narrative) => "voice_sad_narrative",
            (VoiceEmotion::Angry, VoiceStyle::Dramatic) => "voice_angry_dramatic",
            (VoiceEmotion::Calm, VoiceStyle::Whisper) => "voice_calm_whisper",
            (VoiceEmotion::Excited, VoiceStyle::Dramatic) => "voice_excited_dramatic",
            (VoiceEmotion::Mysterious, VoiceStyle::Whisper) => "voice_mysterious_whisper",
            (VoiceEmotion::Playful, VoiceStyle::Conversational) => "voice_playful_casual",
            (VoiceEmotion::Serious, VoiceStyle::Professional) => "voice_serious_professional",
            (VoiceEmotion::Mischievous, VoiceStyle::Narrative) => "voice_mischievous_storyteller",
            (VoiceEmotion::Contemplative, VoiceStyle::Professional) => "voice_contemplative_thoughtful",
            _ => "voice_default", // Default voice
        }.to_string()
    }

    async fn generate_with_elevenlabs(
        &self,
        text: &str,
        voice_id: &str,
        emotion: VoiceEmotion,
        api_key: &str,
    ) -> Result<Vec<u8>> {
        // ElevenLabs API integration
        let response = self.client
            .post("https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM")
            .header("xi-api-key", api_key)
            .header("Content-Type", "application/json")
            .json(&serde_json::json!({
                "text": text,
                "model_id": "eleven_monolingual_v1",
                "voice_settings": {
                    "stability": self.get_stability_for_emotion(emotion),
                    "similarity_boost": 0.75,
                    "style": 0.5,
                    "use_speaker_boost": true
                }
            }))
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(anyhow::anyhow!("ElevenLabs API error: {}", response.status()));
        }

        Ok(response.bytes().await?.to_vec())
    }

    async fn generate_with_openai(
        &self,
        text: &str,
        voice_id: &str,
        api_key: &str,
    ) -> Result<Vec<u8>> {
        // Map voice_id to OpenAI voice names
        let openai_voice = match voice_id {
            "voice_happy_conversational" => "nova",
            "voice_sad_narrative" => "onyx",
            "voice_angry_dramatic" => "fable",
            "voice_calm_whisper" => "shimmer",
            "voice_excited_dramatic" => "alloy",
            "voice_mysterious_whisper" => "echo",
            _ => "nova", // Default voice
        };

        let response = self.client
            .post("https://api.openai.com/v1/audio/speech")
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(&serde_json::json!({
                "model": "tts-1",
                "input": text,
                "voice": openai_voice,
                "response_format": "mp3",
                "speed": 1.0
            }))
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(anyhow::anyhow!("OpenAI TTS API error: {}", response.status()));
        }

        Ok(response.bytes().await?.to_vec())
    }

    async fn create_demo_audio(&self, file_path: &Path, text: &str) -> bool {
        // Try to generate audio using espeak or say command
        #[cfg(target_os = "macos")]
        {
            match tokio::process::Command::new("say")
                .args(&["-o", file_path.to_str().unwrap_or("output.aiff"), text])
                .output()
                .await
            {
                Ok(output) => return output.status.success(),
                Err(_) => {}
            }
        }

        #[cfg(not(target_os = "macos"))]
        {
            // Try espeak on Linux/Unix systems
            match tokio::process::Command::new("espeak")
                .args(&["-w", file_path.to_str().unwrap_or("output.wav"), text])
                .output()
                .await
            {
                Ok(output) => return output.status.success(),
                Err(_) => {}
            }
        }

        false
    }

    async fn create_placeholder_voice(
        &self,
        task_id: &str,
        text: &str,
        emotion: VoiceEmotion,
        style: VoiceStyle,
        config: &CreativeMediaConfig,
        start_time: Instant,
    ) -> Result<GeneratedMedia> {
        // Create a text file with voice generation details
        let placeholder_path = config.media_storage_path.join(format!("{}_voice_placeholder.txt", task_id));
        let placeholder_content = format!(
            "Voice Placeholder\n\nText: {}\nEmotion: {:?}\nStyle: {:?}\n\nNote: Configure ELEVENLABS_API_KEY or OPENAI_API_KEY for actual voice generation",
            text, emotion, style
        );
        
        fs::write(&placeholder_path, &placeholder_content).await?;

        Ok(GeneratedMedia {
            id: Uuid::new_v4().to_string(),
            media_type: "voice_placeholder".to_string(),
            file_path: placeholder_path.to_string_lossy().to_string(),
            prompt: text.to_string(),
            metadata: MediaMetadata {
                provider: "Placeholder".to_string(),
                model: "none".to_string(),
                quality_score: 0.0,
                creative_score: 0.0,
                file_size_bytes: placeholder_content.len() as u64,
                dimensions: None,
                duration_seconds: Some(self.estimate_audio_duration(text)),
                tags: vec![format!("{:?}", emotion), format!("{:?}", style), "placeholder".to_string()],
            },
            created_at: Utc::now(),
            cognitive_context: Some("Placeholder voice - API keys not configured".to_string()),
            generation_time_ms: start_time.elapsed().as_millis() as u64,
        })
    }

    fn get_stability_for_emotion(&self, emotion: VoiceEmotion) -> f32 {
        match emotion {
            VoiceEmotion::Angry | VoiceEmotion::Excited => 0.3,
            VoiceEmotion::Happy | VoiceEmotion::Playful => 0.5,
            VoiceEmotion::Calm | VoiceEmotion::Contemplative => 0.8,
            VoiceEmotion::Sad | VoiceEmotion::Serious => 0.7,
            VoiceEmotion::Mysterious | VoiceEmotion::Mischievous => 0.6,
            VoiceEmotion::Neutral => 0.7,
        }
    }
}
