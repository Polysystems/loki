//! Story-Driven Dependency Management
//!
//! This module implements intelligent dependency management that keeps dependencies
//! up-to-date, secure, and optimized based on project needs and narrative context.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn};
use semver::{Version, VersionReq};

use crate::cognitive::self_modify::{CodeChange, SelfModificationPipeline, RiskLevel};
use crate::memory::CognitiveMemory;
use crate::story::{PlotType, StoryEngine, StoryId};
use crate::tools::code_analysis::CodeAnalyzer;

/// Configuration for dependency management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoryDrivenDependencyConfig {
    /// Enable automatic dependency updates
    pub enable_auto_update: bool,

    /// Enable security vulnerability checking
    pub enable_security_check: bool,

    /// Enable unused dependency detection
    pub enable_unused_detection: bool,

    /// Enable dependency optimization
    pub enable_optimization: bool,

    /// Enable license compliance checking
    pub enable_license_check: bool,

    /// Update strategy
    pub update_strategy: UpdateStrategy,

    /// Maximum risk level for auto-updates
    pub max_auto_update_risk: RiskLevel,

    /// Repository path
    pub repo_path: PathBuf,
}

impl Default for StoryDrivenDependencyConfig {
    fn default() -> Self {
        Self {
            enable_auto_update: true,
            enable_security_check: true,
            enable_unused_detection: true,
            enable_optimization: true,
            enable_license_check: true,
            update_strategy: UpdateStrategy::Conservative,
            max_auto_update_risk: RiskLevel::Low,
            repo_path: PathBuf::from("."),
        }
    }
}

/// Dependency update strategy
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum UpdateStrategy {
    /// Only patch updates (1.2.3 -> 1.2.4)
    Conservative,
    /// Minor updates allowed (1.2.3 -> 1.3.0)
    Balanced,
    /// Major updates allowed (1.2.3 -> 2.0.0)
    Aggressive,
}

/// Dependency information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyInfo {
    pub name: String,
    pub current_version: Version,
    pub latest_version: Option<Version>,
    pub update_available: bool,
    pub update_type: Option<UpdateType>,
    pub is_dev_dependency: bool,
    pub features: Vec<String>,
    pub usage_count: usize,
    pub last_updated: Option<chrono::DateTime<chrono::Utc>>,
}

/// Type of version update
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum UpdateType {
    Patch,
    Minor,
    Major,
}

/// Security vulnerability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityVulnerability {
    pub dependency: String,
    pub severity: VulnerabilitySeverity,
    pub description: String,
    pub cve: Option<String>,
    pub patched_versions: Vec<VersionReq>,
    pub published_date: chrono::DateTime<chrono::Utc>,
}

/// Vulnerability severity
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum VulnerabilitySeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// License information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LicenseInfo {
    pub dependency: String,
    pub license: String,
    pub is_compatible: bool,
    pub compatibility_notes: Option<String>,
}

/// Dependency analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyAnalysis {
    pub total_dependencies: usize,
    pub outdated_dependencies: Vec<DependencyInfo>,
    pub security_vulnerabilities: Vec<SecurityVulnerability>,
    pub unused_dependencies: Vec<String>,
    pub license_issues: Vec<LicenseInfo>,
    pub optimization_opportunities: Vec<OptimizationOpportunity>,
}

/// Optimization opportunity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationOpportunity {
    pub opportunity_type: OptimizationType,
    pub description: String,
    pub impact: String,
    pub recommendation: String,
}

/// Type of optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationType {
    DuplicateFunctionality,
    BundleSize,
    CompileTime,
    FeatureFlags,
    AlternativeCrate,
}

/// Update result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateResult {
    pub dependency: String,
    pub old_version: Version,
    pub new_version: Version,
    pub update_type: UpdateType,
    pub success: bool,
    pub tests_passed: bool,
    pub breaking_changes: Vec<String>,
}

/// Story-driven dependency manager
pub struct StoryDrivenDependencies {
    config: StoryDrivenDependencyConfig,
    story_engine: Arc<StoryEngine>,
    code_analyzer: Arc<CodeAnalyzer>,
    self_modify: Arc<SelfModificationPipeline>,
    memory: Arc<CognitiveMemory>,
    codebase_story_id: StoryId,
    dependency_cache: Arc<RwLock<HashMap<String, DependencyInfo>>>,
    vulnerability_db: Arc<RwLock<Vec<SecurityVulnerability>>>,
    update_history: Arc<RwLock<Vec<UpdateResult>>>,
}

impl StoryDrivenDependencies {
    /// Create new dependency manager
    pub async fn new(
        config: StoryDrivenDependencyConfig,
        story_engine: Arc<StoryEngine>,
        code_analyzer: Arc<CodeAnalyzer>,
        self_modify: Arc<SelfModificationPipeline>,
        memory: Arc<CognitiveMemory>,
    ) -> Result<Self> {
        // Create or get codebase story
        let codebase_story_id = story_engine
            .create_codebase_story(
                config.repo_path.clone(),
                "rust".to_string()
            )
            .await?;

        // Record initialization
        story_engine
            .add_plot_point(
                codebase_story_id.clone(),
                PlotType::Discovery {
                    insight: "Story-driven dependency management initialized".to_string(),
                },
                vec!["dependencies".to_string(), "security".to_string()],
            )
            .await?;

        Ok(Self {
            config,
            story_engine,
            code_analyzer,
            self_modify,
            memory,
            codebase_story_id,
            dependency_cache: Arc::new(RwLock::new(HashMap::new())),
            vulnerability_db: Arc::new(RwLock::new(Vec::new())),
            update_history: Arc::new(RwLock::new(Vec::new())),
        })
    }

    /// Analyze all dependencies
    pub async fn analyze_dependencies(&self) -> Result<DependencyAnalysis> {
        info!("🔍 Analyzing project dependencies");

        // Load Cargo.toml
        let cargo_toml = self.load_cargo_toml().await?;

        // Extract dependencies
        let dependencies = self.extract_dependencies(&cargo_toml).await?;
        let total_dependencies = dependencies.len();

        // Check for updates
        let outdated_dependencies = if self.config.enable_auto_update {
            self.check_for_updates(&dependencies).await?
        } else {
            vec![]
        };

        // Check security vulnerabilities
        let security_vulnerabilities = if self.config.enable_security_check {
            self.check_security_vulnerabilities(&dependencies).await?
        } else {
            vec![]
        };

        // Find unused dependencies
        let unused_dependencies = if self.config.enable_unused_detection {
            self.find_unused_dependencies(&dependencies).await?
        } else {
            vec![]
        };

        // Check licenses
        let license_issues = if self.config.enable_license_check {
            self.check_license_compliance(&dependencies).await?
        } else {
            vec![]
        };

        // Find optimization opportunities
        let optimization_opportunities = if self.config.enable_optimization {
            self.find_optimization_opportunities(&dependencies).await?
        } else {
            vec![]
        };

        Ok(DependencyAnalysis {
            total_dependencies,
            outdated_dependencies,
            security_vulnerabilities,
            unused_dependencies,
            license_issues,
            optimization_opportunities,
        })
    }

    /// Update dependencies based on analysis
    pub async fn update_dependencies(&self) -> Result<Vec<UpdateResult>> {
        info!("🔄 Updating dependencies");

        let analysis = self.analyze_dependencies().await?;
        let mut update_results = Vec::new();

        // Handle security updates first (highest priority)
        for vuln in &analysis.security_vulnerabilities {
            if let Some(update_result) = self.update_for_security(&vuln).await? {
                update_results.push(update_result);
            }
        }

        // Handle regular updates based on strategy
        for dep in &analysis.outdated_dependencies {
            if self.should_update(dep)? {
                if let Some(update_result) = self.update_dependency(dep).await? {
                    update_results.push(update_result);
                }
            }
        }

        // Remove unused dependencies
        for unused in &analysis.unused_dependencies {
            if let Err(e) = self.remove_dependency(unused).await {
                warn!("Failed to remove unused dependency {}: {}", unused, e);
            }
        }

        // Store update history
        self.update_history.write().await.extend(update_results.clone());

        // Record in story
        if !update_results.is_empty() {
            self.story_engine
                .add_plot_point(
                    self.codebase_story_id.clone(),
                    PlotType::Action {
                        action_type: "dependency_update".to_string(),
                        parameters: vec![format!("{} dependencies", update_results.len())],
                        outcome: "successful".to_string(),
                    },
                    vec![],
                )
                .await?;
        }

        Ok(update_results)
    }

    /// Generate dependency report
    pub async fn generate_dependency_report(&self) -> Result<DependencyReport> {
        let analysis = self.analyze_dependencies().await?;
        let history = self.update_history.read().await;

        Ok(DependencyReport {
            total_dependencies: analysis.total_dependencies,
            outdated_count: analysis.outdated_dependencies.len(),
            security_issues: analysis.security_vulnerabilities.len(),
            unused_count: analysis.unused_dependencies.len(),
            license_issues: analysis.license_issues.len(),
            recent_updates: history.clone(),
            recommendations: self.generate_recommendations(&analysis).await?,
        })
    }

    /// Load Cargo.toml
    async fn load_cargo_toml(&self) -> Result<toml::Value> {
        let cargo_path = self.config.repo_path.join("Cargo.toml");
        let content = tokio::fs::read_to_string(&cargo_path)
            .await
            .context("Failed to read Cargo.toml")?;

        toml::from_str(&content).context("Failed to parse Cargo.toml")
    }

    /// Extract dependencies from Cargo.toml
    async fn extract_dependencies(&self, cargo_toml: &toml::Value) -> Result<Vec<DependencyInfo>> {
        let mut dependencies = Vec::new();

        // Regular dependencies
        if let Some(deps) = cargo_toml.get("dependencies").and_then(|d| d.as_table()) {
            for (name, value) in deps {
                let dep_info = self.parse_dependency(name, value, false).await?;
                dependencies.push(dep_info);
            }
        }

        // Dev dependencies
        if let Some(deps) = cargo_toml.get("dev-dependencies").and_then(|d| d.as_table()) {
            for (name, value) in deps {
                let dep_info = self.parse_dependency(name, value, true).await?;
                dependencies.push(dep_info);
            }
        }

        // Cache dependencies
        let mut cache = self.dependency_cache.write().await;
        for dep in &dependencies {
            cache.insert(dep.name.clone(), dep.clone());
        }

        Ok(dependencies)
    }

    /// Parse individual dependency
    async fn parse_dependency(
        &self,
        name: &str,
        value: &toml::Value,
        is_dev: bool,
    ) -> Result<DependencyInfo> {
        let version = match value {
            toml::Value::String(v) => Version::parse(v.trim_start_matches('^'))?,
            toml::Value::Table(t) => {
                if let Some(v) = t.get("version").and_then(|v| v.as_str()) {
                    Version::parse(v.trim_start_matches('^'))?
                } else {
                    Version::new(0, 0, 0) // Git dependencies
                }
            }
            _ => Version::new(0, 0, 0),
        };

        let features = if let toml::Value::Table(t) = value {
            t.get("features")
                .and_then(|f| f.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default()
        } else {
            vec![]
        };

        Ok(DependencyInfo {
            name: name.to_string(),
            current_version: version,
            latest_version: None, // Will be populated by update check
            update_available: false,
            update_type: None,
            is_dev_dependency: is_dev,
            features,
            usage_count: 0, // Will be populated by usage analysis
            last_updated: None,
        })
    }

    /// Check for available updates
    async fn check_for_updates(&self, dependencies: &[DependencyInfo]) -> Result<Vec<DependencyInfo>> {
        let mut outdated = Vec::new();

        for dep in dependencies {
            // In real implementation, would query crates.io API
            // For demo, simulate some updates
            let mut dep_with_update = dep.clone();

            // Simulate finding newer version
            if dep.name.contains("serde") {
                dep_with_update.latest_version = Some(Version::new(
                    dep.current_version.major,
                    dep.current_version.minor,
                    dep.current_version.patch + 1,
                ));
                dep_with_update.update_available = true;
                dep_with_update.update_type = Some(UpdateType::Patch);
                outdated.push(dep_with_update);
            }
        }

        Ok(outdated)
    }

    /// Check security vulnerabilities
    async fn check_security_vulnerabilities(
        &self,
        dependencies: &[DependencyInfo],
    ) -> Result<Vec<SecurityVulnerability>> {
        let mut vulnerabilities = Vec::new();

        // In real implementation, would check RustSec advisory database
        // For demo, simulate some vulnerabilities
        for dep in dependencies {
            if dep.name == "openssl" && dep.current_version < Version::new(1, 1, 1) {
                vulnerabilities.push(SecurityVulnerability {
                    dependency: dep.name.clone(),
                    severity: VulnerabilitySeverity::High,
                    description: "Multiple vulnerabilities in OpenSSL".to_string(),
                    cve: Some("CVE-2021-XXXX".to_string()),
                    patched_versions: vec![VersionReq::parse(">=1.1.1")?, VersionReq::parse(">=0.10.48,<1.0.0")?],
                    published_date: chrono::Utc::now() - chrono::Duration::days(30),
                });
            }
        }

        Ok(vulnerabilities)
    }

    /// Find unused dependencies
    async fn find_unused_dependencies(&self, dependencies: &[DependencyInfo]) -> Result<Vec<String>> {
        let mut unused = Vec::new();

        // Analyze usage in source files
        let src_files = self.find_source_files().await?;
        let mut usage_map: HashMap<String, usize> = HashMap::new();

        for file in src_files {
            let content = tokio::fs::read_to_string(&file).await?;

            for dep in dependencies {
                let crate_name = dep.name.replace('-', "_");
                if content.contains(&format!("use {};", crate_name))
                    || content.contains(&format!("use {}::", crate_name))
                    || content.contains(&format!("extern crate {};", crate_name))
                {
                    *usage_map.entry(dep.name.clone()).or_insert(0) += 1;
                }
            }
        }

        // Find dependencies with no usage
        for dep in dependencies {
            if !usage_map.contains_key(&dep.name) && !is_essential_dependency(&dep.name) {
                unused.push(dep.name.clone());
            }
        }

        Ok(unused)
    }

    /// Check license compliance
    async fn check_license_compliance(
        &self,
        dependencies: &[DependencyInfo],
    ) -> Result<Vec<LicenseInfo>> {
        let mut issues = Vec::new();

        // In real implementation, would check actual licenses
        // For demo, simulate some license checks
        let allowed_licenses = vec!["MIT", "Apache-2.0", "BSD-3-Clause"];

        for dep in dependencies {
            // Simulate license check
            if dep.name.contains("gpl") {
                issues.push(LicenseInfo {
                    dependency: dep.name.clone(),
                    license: "GPL-3.0".to_string(),
                    is_compatible: false,
                    compatibility_notes: Some("GPL license may not be compatible with your project".to_string()),
                });
            }
        }

        Ok(issues)
    }

    /// Find optimization opportunities
    async fn find_optimization_opportunities(
        &self,
        dependencies: &[DependencyInfo],
    ) -> Result<Vec<OptimizationOpportunity>> {
        let mut opportunities = Vec::new();

        // Check for duplicate functionality
        let mut functionality_map: HashMap<&str, Vec<&str>> = HashMap::new();
        functionality_map.insert("serialization", vec!["serde", "json", "bincode"]);
        functionality_map.insert("async_runtime", vec!["tokio", "async-std", "smol"]);
        functionality_map.insert("http_client", vec!["reqwest", "surf", "ureq"]);

        for (functionality, crates) in functionality_map {
            let found: Vec<_> = dependencies
                .iter()
                .filter(|d| crates.contains(&d.name.as_str()))
                .collect();

            if found.len() > 1 {
                opportunities.push(OptimizationOpportunity {
                    opportunity_type: OptimizationType::DuplicateFunctionality,
                    description: format!("Multiple {} crates found", functionality),
                    impact: "Increased binary size and compilation time".to_string(),
                    recommendation: format!("Consider using only one {} crate", functionality),
                });
            }
        }

        // Check for heavy dependencies
        for dep in dependencies {
            if is_heavy_dependency(&dep.name) {
                opportunities.push(OptimizationOpportunity {
                    opportunity_type: OptimizationType::BundleSize,
                    description: format!("{} is a heavy dependency", dep.name),
                    impact: "Increased binary size and compilation time".to_string(),
                    recommendation: format!("Consider lighter alternatives to {}", dep.name),
                });
            }
        }

        Ok(opportunities)
    }

    /// Update dependency for security
    async fn update_for_security(&self, vuln: &SecurityVulnerability) -> Result<Option<UpdateResult>> {
        info!("🔒 Updating {} for security vulnerability", vuln.dependency);

        // Find the dependency
        let cache = self.dependency_cache.read().await;
        let dep = cache.get(&vuln.dependency)
            .ok_or_else(|| anyhow::anyhow!("Dependency not found"))?;

        // Find appropriate version
        let target_version = self.find_secure_version(dep, vuln)?;

        // Create update
        let update_result = UpdateResult {
            dependency: dep.name.clone(),
            old_version: dep.current_version.clone(),
            new_version: target_version.clone(),
            update_type: self.determine_update_type(&dep.current_version, &target_version),
            success: false,
            tests_passed: false,
            breaking_changes: vec![],
        };

        // Apply update
        self.apply_dependency_update(&update_result).await?;

        Ok(Some(update_result))
    }

    /// Update a dependency
    async fn update_dependency(&self, dep: &DependencyInfo) -> Result<Option<UpdateResult>> {
        if let Some(latest) = &dep.latest_version {
            let update_result = UpdateResult {
                dependency: dep.name.clone(),
                old_version: dep.current_version.clone(),
                new_version: latest.clone(),
                update_type: dep.update_type.clone().unwrap_or(UpdateType::Patch),
                success: false,
                tests_passed: false,
                breaking_changes: vec![],
            };

            // Check risk level
            let risk = self.assess_update_risk(&update_result);
            if risk <= self.config.max_auto_update_risk {
                self.apply_dependency_update(&update_result).await?;
                Ok(Some(update_result))
            } else {
                info!("Update for {} skipped due to risk level {:?}", dep.name, risk);
                Ok(None)
            }
        } else {
            Ok(None)
        }
    }

    /// Remove unused dependency
    async fn remove_dependency(&self, name: &str) -> Result<()> {
        info!("🗑️  Removing unused dependency: {}", name);

        let cargo_path = self.config.repo_path.join("Cargo.toml");
        let mut cargo_toml = self.load_cargo_toml().await?;

        // Remove from dependencies
        if let Some(deps) = cargo_toml.get_mut("dependencies").and_then(|d| d.as_table_mut()) {
            deps.remove(name);
        }

        // Remove from dev-dependencies
        if let Some(deps) = cargo_toml.get_mut("dev-dependencies").and_then(|d| d.as_table_mut()) {
            deps.remove(name);
        }

        // Write back
        let content = toml::to_string_pretty(&cargo_toml)?;
        tokio::fs::write(&cargo_path, content).await?;

        Ok(())
    }

    /// Apply dependency update
    async fn apply_dependency_update(&self, update: &UpdateResult) -> Result<()> {
        let change = CodeChange {
            file_path: self.config.repo_path.join("Cargo.toml"),
            change_type: crate::cognitive::self_modify::ChangeType::Enhancement,
            description: format!(
                "Update {} from {} to {}",
                update.dependency, update.old_version, update.new_version
            ),
            reasoning: format!(
                "Update {} dependency for improved features and bug fixes",
                update.dependency
            ),
            old_content: Some(format!("{} = \"{}\"", update.dependency, update.old_version)),
            new_content: format!("{} = \"{}\"", update.dependency, update.new_version),
            line_range: None,
            risk_level: self.assess_update_risk(update),
            attribution: None,
        };

        self.self_modify.apply_code_change(change).await?;

        Ok(())
    }

    /// Should update based on strategy
    fn should_update(&self, dep: &DependencyInfo) -> Result<bool> {
        if let Some(update_type) = &dep.update_type {
            match (&self.config.update_strategy, update_type) {
                (UpdateStrategy::Conservative, UpdateType::Patch) => Ok(true),
                (UpdateStrategy::Conservative, _) => Ok(false),
                (UpdateStrategy::Balanced, UpdateType::Major) => Ok(false),
                (UpdateStrategy::Balanced, _) => Ok(true),
                (UpdateStrategy::Aggressive, _) => Ok(true),
            }
        } else {
            Ok(false)
        }
    }

    /// Assess update risk
    fn assess_update_risk(&self, update: &UpdateResult) -> RiskLevel {
        match update.update_type {
            UpdateType::Patch => RiskLevel::Low,
            UpdateType::Minor => RiskLevel::Medium,
            UpdateType::Major => RiskLevel::High,
        }
    }

    /// Find secure version
    fn find_secure_version(
        &self,
        dep: &DependencyInfo,
        vuln: &SecurityVulnerability,
    ) -> Result<Version> {
        // In real implementation, would find the best matching version
        // For demo, increment patch version
        Ok(Version::new(
            dep.current_version.major,
            dep.current_version.minor,
            dep.current_version.patch + 1,
        ))
    }

    /// Determine update type
    fn determine_update_type(&self, old: &Version, new: &Version) -> UpdateType {
        if old.major != new.major {
            UpdateType::Major
        } else if old.minor != new.minor {
            UpdateType::Minor
        } else {
            UpdateType::Patch
        }
    }

    /// Find source files
    async fn find_source_files(&self) -> Result<Vec<PathBuf>> {
        let mut source_files = Vec::new();
        let src_dir = self.config.repo_path.join("src");

        if src_dir.exists() {
            self.find_rust_files_recursive(&src_dir, &mut source_files).await?;
        }

        Ok(source_files)
    }

    /// Find Rust files recursively
    async fn find_rust_files_recursive(
        &self,
        dir: &Path,
        files: &mut Vec<PathBuf>,
    ) -> Result<()> {
        let mut entries = tokio::fs::read_dir(dir).await?;

        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();

            if path.is_dir() {
                Box::pin(self.find_rust_files_recursive(&path, files)).await?;
            } else if path.extension().map_or(false, |ext| ext == "rs") {
                files.push(path);
            }
        }

        Ok(())
    }

    /// Generate recommendations
    async fn generate_recommendations(&self, analysis: &DependencyAnalysis) -> Result<Vec<String>> {
        let mut recommendations = Vec::new();

        if !analysis.security_vulnerabilities.is_empty() {
            recommendations.push(format!(
                "🔒 Update {} dependencies with security vulnerabilities immediately",
                analysis.security_vulnerabilities.len()
            ));
        }

        if analysis.outdated_dependencies.len() > 5 {
            recommendations.push(
                "📦 Consider updating dependencies more frequently to avoid large batches".to_string()
            );
        }

        if !analysis.unused_dependencies.is_empty() {
            recommendations.push(format!(
                "🧹 Remove {} unused dependencies to reduce build time",
                analysis.unused_dependencies.len()
            ));
        }

        if !analysis.license_issues.is_empty() {
            recommendations.push(
                "⚖️ Review license compatibility issues before distribution".to_string()
            );
        }

        for opportunity in &analysis.optimization_opportunities {
            recommendations.push(format!("💡 {}", opportunity.recommendation));
        }

        Ok(recommendations)
    }
}

/// Dependency report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyReport {
    pub total_dependencies: usize,
    pub outdated_count: usize,
    pub security_issues: usize,
    pub unused_count: usize,
    pub license_issues: usize,
    pub recent_updates: Vec<UpdateResult>,
    pub recommendations: Vec<String>,
}

/// Check if dependency is essential
fn is_essential_dependency(name: &str) -> bool {
    matches!(
        name,
        "std" | "core" | "alloc" | "proc_macro" | "test" | "criterion" | "serde_derive"
    )
}

/// Check if dependency is heavy
fn is_heavy_dependency(name: &str) -> bool {
    matches!(
        name,
        "tokio" | "actix-web" | "diesel" | "rocket" | "tensorflow" | "opencv"
    )
}
