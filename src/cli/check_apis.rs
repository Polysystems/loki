//! CLI command to check API configuration status

use anyhow::Result;
use colored::*;
use std::io::{self, Write};

use crate::config::{ApiKeysConfig, SetupWizard};

/// Check and display API configuration status
pub async fn check_api_status() -> Result<()> {
    println!("{}", "🔍 Checking API Configuration...".cyan().bold());
    println!();
    
    // Load API keys
    let apiconfig = match ApiKeysConfig::from_env() {
        Ok(config) => config,
        Err(e) => {
            eprintln!("{} Failed to load API configuration: {}", "❌".red(), e);
            return Err(e);
        }
    };
    
    // Get summary
    let summary = apiconfig.summary();
    
    // Display status
    println!("{}", "Core APIs:".yellow().bold());
    print_status("GitHub", summary.get("GitHub").copied().unwrap_or(false));
    print_status("X/Twitter", summary.get("X/Twitter").copied().unwrap_or(false));
    
    println!();
    println!("{}", "AI Model APIs:".yellow().bold());
    print_status("OpenAI", summary.get("OpenAI").copied().unwrap_or(false));
    print_status("Anthropic", summary.get("Anthropic").copied().unwrap_or(false));
    print_status("DeepSeek", summary.get("DeepSeek").copied().unwrap_or(false));
    print_status("Mistral", summary.get("Mistral").copied().unwrap_or(false));
    print_status("Codestral", summary.get("Codestral").copied().unwrap_or(false));
    print_status("Gemini", summary.get("Gemini").copied().unwrap_or(false));
    print_status("Grok", summary.get("Grok").copied().unwrap_or(false));
    
    println!();
    println!("{}", "Search APIs:".yellow().bold());
    print_status("Brave Search", summary.get("Brave Search").copied().unwrap_or(false));
    print_status("Google Search", summary.get("Google Search").copied().unwrap_or(false));
    print_status("Bing Search", summary.get("Bing Search").copied().unwrap_or(false));
    
    println!();
    println!("{}", "Data Storage:".yellow().bold());
    print_status("Vector Database", summary.get("Vector Database").copied().unwrap_or(false));
    
    println!();
    println!("{}", "Optional Services:".yellow().bold());
    print_status("Stack Exchange", summary.get("Stack Exchange").copied().unwrap_or(false));
    print_status("Replicate", summary.get("Replicate").copied().unwrap_or(false));
    print_status("Hugging Face", summary.get("Hugging Face").copied().unwrap_or(false));

    println!();
    println!("{}", "Local Models (Ollama):".yellow().bold());
    let mut wizzard = SetupWizard::new()?;
    wizzard.scan_local_models(false).await?;
    if wizzard.config.modelconfig.local_models.is_empty() {
        println!("  ❌ No local models found. Run 'Scan for Local Models' first.");
    } else {
        for model in &wizzard.config.modelconfig.local_models {
            println!("  ✅ {}", model);
        }
    }

    // Check if we have minimum requirements
    println!();
    if apiconfig.has_ai_model() {
        println!("{} At least one AI model API is configured", "✅".green());
    } else {
        println!("{} No AI model APIs configured - will use Ollama", "⚠️ ".yellow());
    }
    
    if let Some((provider, _)) = apiconfig.get_preferred_search_api() {
        println!("{} Search API available: {}", "✅".green(), provider);
    } else {
        println!("{} No search APIs configured", "⚠️ ".yellow());
    }
    
    // Show preferred models
    println!();
    if let Some((provider, _)) = apiconfig.get_preferred_ai_model() {
        println!("{}", format!("Preferred AI model: {}", provider).green());
    }
    
    // Check critical services
    println!();
    let mut missing_critical = Vec::new();
    
    if apiconfig.github.is_none() {
        missing_critical.push("GitHub (required for self-modification)");
    }
    
    if apiconfig.x_twitter.is_none() {
        missing_critical.push("X/Twitter (required for social engagement)");
    }
    
    if !missing_critical.is_empty() {
        println!("{}", "Missing critical services:".red().bold());
        for service in missing_critical {
            println!("  {} {}", "•".red(), service);
        }
    } else {
        println!("{} All critical services configured!", "✅".green().bold());
    }
    
    Ok(())
}

fn print_status(name: &str, configured: bool) {
    let status = if configured {
        "✅ Configured".green()
    } else {
        "❌ Not configured".red()
    };
    println!("  {:<20} {}", name, status);
}

/// Interactive API key setup
pub async fn setup_api_keys() -> Result<()> {
    println!("{}", "🔧 API Key Setup Wizard".cyan().bold());
    println!();
    println!("This wizard will help you set up your API keys.");
    println!("Press Enter to skip any service you don't want to configure.");
    println!();
    
    let mut env_lines = Vec::new();
    
    // GitHub
    println!("{}", "GitHub API:".yellow().bold());
    if let Some(token) = prompt("GitHub Personal Access Token")? {
        env_lines.push(format!("GITHUB_TOKEN={}", token));
        
        if let Some(owner) = prompt("GitHub Username/Organization")? {
            env_lines.push(format!("GITHUB_OWNER={}", owner));
        }
        
        if let Some(repo) = prompt("Repository Name")? {
            env_lines.push(format!("GITHUB_REPO={}", repo));
        }
        
        env_lines.push("GITHUB_DEFAULT_BRANCH=main".to_string());
    }
    
    println!();
    
    // X/Twitter
    println!("{}", "X (Twitter) API:".yellow().bold());
    if let Some(key) = prompt("X API Key")? {
        env_lines.push(format!("X_API_KEY={}", key));
        
        if let Some(secret) = prompt("X API Secret")? {
            env_lines.push(format!("X_API_SECRET={}", secret));
        }
        
        if let Some(token) = prompt("X Access Token")? {
            env_lines.push(format!("X_ACCESS_TOKEN={}", token));
        }
        
        if let Some(secret) = prompt("X Access Token Secret")? {
            env_lines.push(format!("X_ACCESS_TOKEN_SECRET={}", secret));
        }
    }
    
    println!();
    
    // AI Models
    println!("{}", "AI Model APIs:".yellow().bold());
    
    if let Some(key) = prompt("OpenAI API Key")? {
        env_lines.push(format!("OPENAI_API_KEY={}", key));
    }
    
    if let Some(key) = prompt("Anthropic API Key")? {
        env_lines.push(format!("ANTHROPIC_API_KEY={}", key));
    }
    
    if let Some(key) = prompt("DeepSeek API Key")? {
        env_lines.push(format!("DEEPSEEK_API_KEY={}", key));
    }
    
    // Add more as needed...
    
    println!();
    
    // Search APIs
    println!("{}", "Search APIs:".yellow().bold());
    
    if let Some(key) = prompt("Brave Search API Key")? {
        env_lines.push(format!("BRAVE_SEARCH_API_KEY={}", key));
    }
    
    // Save to .env.example
    if !env_lines.is_empty() {
        println!();
        println!("{}", "Generated .env configuration:".green().bold());
        println!();
        for line in &env_lines {
            println!("{}", line);
        }
        
        println!();
        println!("Copy the above configuration to your .env file.");
    }
    
    Ok(())
}

fn prompt(message: &str) -> Result<Option<String>> {
    print!("{}: ", message);
    io::stdout().flush()?;
    
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    
    let trimmed = input.trim();
    if trimmed.is_empty() {
        Ok(None)
    } else {
        Ok(Some(trimmed.to_string()))
    }
} 