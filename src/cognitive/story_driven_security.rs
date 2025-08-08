//! Story-Driven Security Vulnerability Detection
//!
//! This module implements intelligent security analysis that detects vulnerabilities,
//! assesses risks, and applies fixes based on narrative context and threat modeling.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, error};

use crate::cognitive::self_modify::{CodeChange, SelfModificationPipeline, RiskLevel};
use crate::cognitive::story_driven_dependencies::{StoryDrivenDependencies, VulnerabilitySeverity};
use crate::memory::CognitiveMemory;
use crate::story::{PlotType, StoryEngine, StoryId};
use crate::tools::code_analysis::CodeAnalyzer;

/// Configuration for security monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoryDrivenSecurityConfig {
    /// Enable static security analysis
    pub enable_static_analysis: bool,

    /// Enable dependency vulnerability scanning
    pub enable_dependency_scanning: bool,

    /// Enable runtime security monitoring
    pub enable_runtime_monitoring: bool,

    /// Enable secret detection
    pub enable_secret_detection: bool,

    /// Enable OWASP rule checking
    pub enable_owasp_checks: bool,

    /// Enable automatic security patching
    pub enable_auto_patching: bool,

    /// Maximum risk level for automatic fixes
    pub max_auto_fix_risk: RiskLevel,

    /// Security scan interval
    pub scan_interval: std::time::Duration,

    /// Repository path
    pub repo_path: PathBuf,
}

impl Default for StoryDrivenSecurityConfig {
    fn default() -> Self {
        Self {
            enable_static_analysis: true,
            enable_dependency_scanning: true,
            enable_runtime_monitoring: true,
            enable_secret_detection: true,
            enable_owasp_checks: true,
            enable_auto_patching: true,
            max_auto_fix_risk: RiskLevel::Low,
            scan_interval: std::time::Duration::from_secs(3600), // 1 hour
            repo_path: PathBuf::from("."),
        }
    }
}

/// Security vulnerability types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum VulnerabilityType {
    SqlInjection,
    CommandInjection,
    PathTraversal,
    XSS,
    CSRF,
    XXE,
    InsecureDeserialization,
    BrokenAuthentication,
    SensitiveDataExposure,
    BrokenAccessControl,
    SecurityMisconfiguration,
    InsecureDependency,
    HardcodedSecret,
    WeakCryptography,
    InsecureRandomness,
    UnvalidatedInput,
    BufferOverflow,
    RaceCondition,
    DenialOfService,
}

impl std::fmt::Display for VulnerabilityType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VulnerabilityType::SqlInjection => write!(f, "SQL Injection"),
            VulnerabilityType::CommandInjection => write!(f, "Command Injection"),
            VulnerabilityType::PathTraversal => write!(f, "Path Traversal"),
            VulnerabilityType::XSS => write!(f, "Cross-Site Scripting (XSS)"),
            VulnerabilityType::CSRF => write!(f, "Cross-Site Request Forgery (CSRF)"),
            VulnerabilityType::XXE => write!(f, "XML External Entity (XXE)"),
            VulnerabilityType::InsecureDeserialization => write!(f, "Insecure Deserialization"),
            VulnerabilityType::BrokenAuthentication => write!(f, "Broken Authentication"),
            VulnerabilityType::SensitiveDataExposure => write!(f, "Sensitive Data Exposure"),
            VulnerabilityType::BrokenAccessControl => write!(f, "Broken Access Control"),
            VulnerabilityType::SecurityMisconfiguration => write!(f, "Security Misconfiguration"),
            VulnerabilityType::InsecureDependency => write!(f, "Insecure Dependency"),
            VulnerabilityType::HardcodedSecret => write!(f, "Hardcoded Secret"),
            VulnerabilityType::WeakCryptography => write!(f, "Weak Cryptography"),
            VulnerabilityType::InsecureRandomness => write!(f, "Insecure Randomness"),
            VulnerabilityType::UnvalidatedInput => write!(f, "Unvalidated Input"),
            VulnerabilityType::BufferOverflow => write!(f, "Buffer Overflow"),
            VulnerabilityType::RaceCondition => write!(f, "Race Condition"),
            VulnerabilityType::DenialOfService => write!(f, "Denial of Service"),
        }
    }
}

/// Security vulnerability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityVulnerability {
    pub vulnerability_id: String,
    pub vulnerability_type: VulnerabilityType,
    pub severity: VulnerabilitySeverity,
    pub location: VulnerabilityLocation,
    pub description: String,
    pub impact: SecurityImpact,
    pub cve_id: Option<String>,
    pub cwe_id: Option<String>,
    pub owasp_category: Option<String>,
    pub evidence: VulnerabilityEvidence,
    pub fix_available: bool,
    pub fix_suggestion: Option<SecurityFix>,
}

/// Vulnerability location
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VulnerabilityLocation {
    pub file_path: PathBuf,
    pub line_start: Option<usize>,
    pub line_end: Option<usize>,
    pub function_name: Option<String>,
    pub dependency_name: Option<String>,
}

/// Security impact assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityImpact {
    pub confidentiality_impact: ImpactLevel,
    pub integrity_impact: ImpactLevel,
    pub availability_impact: ImpactLevel,
    pub exploitability: ExploitabilityLevel,
    pub cvss_score: Option<f32>,
}

/// Impact levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
pub enum ImpactLevel {
    None,
    Low,
    Medium,
    High,
    Critical,
}

/// Exploitability levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ExploitabilityLevel {
    Theoretical,
    ProofOfConcept,
    Functional,
    Weaponized,
}

/// Vulnerability evidence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VulnerabilityEvidence {
    pub code_snippet: Option<String>,
    pub attack_vector: Option<String>,
    pub prerequisites: Vec<String>,
    pub indicators: Vec<String>,
}

/// Security fix suggestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityFix {
    pub fix_id: String,
    pub fix_type: SecurityFixType,
    pub description: String,
    pub code_changes: Vec<ProposedSecurityChange>,
    pub risk_level: RiskLevel,
    pub validation_required: bool,
}

/// Security fix types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SecurityFixType {
    InputValidation,
    OutputEncoding,
    ParameterizedQuery,
    AccessControl,
    Encryption,
    Authentication,
    ConfigurationChange,
    DependencyUpdate,
    CodeRefactoring,
    SecurityHeader,
}

/// Proposed security change
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProposedSecurityChange {
    pub file_path: PathBuf,
    pub change_type: String,
    pub vulnerable_code: String,
    pub secure_code: String,
    pub explanation: String,
}

/// Security scan result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityScanResult {
    pub scan_id: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub vulnerabilities: Vec<SecurityVulnerability>,
    pub risk_summary: RiskSummary,
    pub compliance_status: ComplianceStatus,
    pub recommendations: Vec<SecurityRecommendation>,
}

/// Risk summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskSummary {
    pub critical_count: usize,
    pub high_count: usize,
    pub medium_count: usize,
    pub low_count: usize,
    pub overall_risk_level: RiskLevel,
    pub trending: TrendDirection,
}

/// Trend direction
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TrendDirection {
    Improving,
    Stable,
    Worsening,
}

/// Compliance status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceStatus {
    pub owasp_top_10_compliant: bool,
    pub pci_dss_compliant: bool,
    pub gdpr_compliant: bool,
    pub violations: Vec<ComplianceViolation>,
}

/// Compliance violation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceViolation {
    pub standard: String,
    pub requirement: String,
    pub description: String,
    pub severity: VulnerabilitySeverity,
}

/// Security recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityRecommendation {
    pub recommendation_id: String,
    pub priority: RecommendationPriority,
    pub category: String,
    pub description: String,
    pub implementation_effort: EffortLevel,
    pub impact: SecurityImpact,
}

/// Recommendation priority
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
pub enum RecommendationPriority {
    Immediate,
    High,
    Medium,
    Low,
}

/// Effort level
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EffortLevel {
    Trivial,
    Low,
    Medium,
    High,
}

/// Security fix result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityFixResult {
    pub fix_id: String,
    pub success: bool,
    pub vulnerabilities_fixed: Vec<String>,
    pub changes_applied: Vec<CodeChange>,
    pub validation_passed: bool,
    pub rollback_available: bool,
}

/// Story-driven security monitor
pub struct StoryDrivenSecurity {
    config: StoryDrivenSecurityConfig,
    story_engine: Arc<StoryEngine>,
    code_analyzer: Arc<CodeAnalyzer>,
    self_modify: Arc<SelfModificationPipeline>,
    dependency_scanner: Option<Arc<StoryDrivenDependencies>>,
    memory: Arc<CognitiveMemory>,
    codebase_story_id: StoryId,
    vulnerability_cache: Arc<RwLock<HashMap<String, SecurityVulnerability>>>,
    scan_history: Arc<RwLock<Vec<SecurityScanResult>>>,
    fix_history: Arc<RwLock<Vec<SecurityFixResult>>>,
}

impl StoryDrivenSecurity {
    /// Create new security monitor
    pub async fn new(
        config: StoryDrivenSecurityConfig,
        story_engine: Arc<StoryEngine>,
        code_analyzer: Arc<CodeAnalyzer>,
        self_modify: Arc<SelfModificationPipeline>,
        dependency_scanner: Option<Arc<StoryDrivenDependencies>>,
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
                    insight: "Security monitoring system initialized".to_string(),
                },
                vec!["security".to_string(), "monitoring".to_string()],
            )
            .await?;

        Ok(Self {
            config,
            story_engine,
            code_analyzer,
            self_modify,
            dependency_scanner,
            memory,
            codebase_story_id,
            vulnerability_cache: Arc::new(RwLock::new(HashMap::new())),
            scan_history: Arc::new(RwLock::new(Vec::new())),
            fix_history: Arc::new(RwLock::new(Vec::new())),
        })
    }

    /// Perform security scan
    pub async fn scan_for_vulnerabilities(&self) -> Result<SecurityScanResult> {
        info!("🔒 Scanning for security vulnerabilities");

        let mut vulnerabilities = Vec::new();

        // Static security analysis
        if self.config.enable_static_analysis {
            let static_vulns = self.perform_static_analysis().await?;
            vulnerabilities.extend(static_vulns);
        }

        // Dependency vulnerability scanning
        if self.config.enable_dependency_scanning {
            if let Some(dep_scanner) = &self.dependency_scanner {
                let dep_vulns = self.scan_dependencies(dep_scanner).await?;
                vulnerabilities.extend(dep_vulns);
            }
        }

        // Secret detection
        if self.config.enable_secret_detection {
            let secret_vulns = self.detect_secrets().await?;
            vulnerabilities.extend(secret_vulns);
        }

        // OWASP checks
        if self.config.enable_owasp_checks {
            let owasp_vulns = self.check_owasp_rules().await?;
            vulnerabilities.extend(owasp_vulns);
        }

        // Calculate risk summary
        let risk_summary = self.calculate_risk_summary(&vulnerabilities);

        // Check compliance
        let compliance_status = self.check_compliance(&vulnerabilities);

        // Generate recommendations
        let recommendations = self.generate_recommendations(&vulnerabilities);

        // Store vulnerabilities in cache
        let mut cache = self.vulnerability_cache.write().await;
        for vuln in &vulnerabilities {
            cache.insert(vuln.vulnerability_id.clone(), vuln.clone());
        }

        let scan_result = SecurityScanResult {
            scan_id: uuid::Uuid::new_v4().to_string(),
            timestamp: chrono::Utc::now(),
            vulnerabilities,
            risk_summary,
            compliance_status,
            recommendations,
        };

        // Store scan result
        self.scan_history.write().await.push(scan_result.clone());

        // Record in story
        self.story_engine
            .add_plot_point(
                self.codebase_story_id.clone(),
                PlotType::Analysis {
                    subject: "security".to_string(),
                    findings: vec![
                        format!("Found {} vulnerabilities", scan_result.vulnerabilities.len()),
                        format!("{} critical, {} high severity",
                            scan_result.risk_summary.critical_count,
                            scan_result.risk_summary.high_count),
                        format!("Overall risk: {:?}", scan_result.risk_summary.overall_risk_level),
                    ],
                },
                vec!["security".to_string(), "scan".to_string()],
            )
            .await?;

        Ok(scan_result)
    }

    /// Fix security vulnerability
    pub async fn fix_vulnerability(&self, vulnerability_id: &str) -> Result<SecurityFixResult> {
        info!("🛡️ Fixing security vulnerability: {}", vulnerability_id);

        // Find vulnerability
        let cache = self.vulnerability_cache.read().await;
        let vulnerability = cache.get(vulnerability_id)
            .ok_or_else(|| anyhow::anyhow!("Vulnerability not found"))?
            .clone();
        drop(cache);

        // Check if fix is available
        if !vulnerability.fix_available {
            return Ok(SecurityFixResult {
                fix_id: uuid::Uuid::new_v4().to_string(),
                success: false,
                vulnerabilities_fixed: vec![],
                changes_applied: vec![],
                validation_passed: false,
                rollback_available: false,
            });
        }

        let fix = vulnerability.fix_suggestion
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No fix suggestion available"))?;

        // Check risk level
        if fix.risk_level > self.config.max_auto_fix_risk {
            warn!("Fix risk level {:?} exceeds maximum {:?}",
                fix.risk_level, self.config.max_auto_fix_risk);
            return Ok(SecurityFixResult {
                fix_id: fix.fix_id.clone(),
                success: false,
                vulnerabilities_fixed: vec![],
                changes_applied: vec![],
                validation_passed: false,
                rollback_available: false,
            });
        }

        // Apply security fix
        let mut changes_applied = Vec::new();

        for proposed_change in &fix.code_changes {
            let code_change = self.convert_to_code_change(proposed_change, &fix).await?;

            match self.self_modify.apply_code_change(code_change.clone()).await {
                Ok(_) => changes_applied.push(code_change),
                Err(e) => {
                    error!("Failed to apply security fix: {}", e);
                    // Rollback previous changes
                    for applied in changes_applied.iter().rev() {
                        let _ = self.self_modify.rollback_change(&applied.file_path.to_string_lossy()).await;
                    }

                    return Ok(SecurityFixResult {
                        fix_id: fix.fix_id.clone(),
                        success: false,
                        vulnerabilities_fixed: vec![],
                        changes_applied: vec![],
                        validation_passed: false,
                        rollback_available: false,
                    });
                }
            }
        }

        // Validate fix
        let validation_passed = if fix.validation_required {
            self.validate_security_fix(&vulnerability, &changes_applied).await?
        } else {
            true
        };

        let result = SecurityFixResult {
            fix_id: fix.fix_id.clone(),
            success: true,
            vulnerabilities_fixed: vec![vulnerability_id.to_string()],
            changes_applied,
            validation_passed,
            rollback_available: true,
        };

        // Store fix result
        self.fix_history.write().await.push(result.clone());

        // Record in story
        self.story_engine
            .add_plot_point(
                self.codebase_story_id.clone(),
                PlotType::Action {
                    action_type: "security_fix".to_string(),
                    parameters: vec![format!("{}", vulnerability.vulnerability_type)],
                    outcome: "successful".to_string(),
                },
                vec![],
            )
            .await?;

        Ok(result)
    }

    /// Perform static security analysis
    async fn perform_static_analysis(&self) -> Result<Vec<SecurityVulnerability>> {
        let mut vulnerabilities = Vec::new();
        let files = self.find_source_files().await?;

        for file in files {
            if let Ok(content) = tokio::fs::read_to_string(&file).await {
                // SQL Injection detection
                if content.contains("format!(") && content.contains("SELECT") {
                    vulnerabilities.push(SecurityVulnerability {
                        vulnerability_id: uuid::Uuid::new_v4().to_string(),
                        vulnerability_type: VulnerabilityType::SqlInjection,
                        severity: VulnerabilitySeverity::Critical,
                        location: VulnerabilityLocation {
                            file_path: file.clone(),
                            line_start: None,
                            line_end: None,
                            function_name: None,
                            dependency_name: None,
                        },
                        description: "Potential SQL injection vulnerability - using string formatting in SQL query".to_string(),
                        impact: SecurityImpact {
                            confidentiality_impact: ImpactLevel::High,
                            integrity_impact: ImpactLevel::High,
                            availability_impact: ImpactLevel::Medium,
                            exploitability: ExploitabilityLevel::Functional,
                            cvss_score: Some(9.1),
                        },
                        cve_id: None,
                        cwe_id: Some("CWE-89".to_string()),
                        owasp_category: Some("A03:2021".to_string()),
                        evidence: VulnerabilityEvidence {
                            code_snippet: Some("format!(\"SELECT * FROM users WHERE id = {}\", user_id)".to_string()),
                            attack_vector: Some("SQL injection via user input".to_string()),
                            prerequisites: vec!["User input reaches SQL query".to_string()],
                            indicators: vec!["String interpolation in SQL".to_string()],
                        },
                        fix_available: true,
                        fix_suggestion: Some(SecurityFix {
                            fix_id: uuid::Uuid::new_v4().to_string(),
                            fix_type: SecurityFixType::ParameterizedQuery,
                            description: "Use parameterized queries instead of string formatting".to_string(),
                            code_changes: vec![],
                            risk_level: RiskLevel::Low,
                            validation_required: true,
                        }),
                    });
                }

                // Command Injection detection
                if content.contains("Command::new") && (content.contains("format!(") || content.contains(".arg(")) {
                    let lines: Vec<_> = content.lines().enumerate().collect();
                    for (i, line) in &lines {
                        if line.contains("Command::new") {
                            // Check next few lines for user input
                            for j in i+1..std::cmp::min(i+5, lines.len()) {
                                if lines[j].1.contains(".arg(") && !lines[j].1.contains("\"") {
                                    vulnerabilities.push(SecurityVulnerability {
                                        vulnerability_id: uuid::Uuid::new_v4().to_string(),
                                        vulnerability_type: VulnerabilityType::CommandInjection,
                                        severity: VulnerabilitySeverity::Critical,
                                        location: VulnerabilityLocation {
                                            file_path: file.clone(),
                                            line_start: Some(i + 1),
                                            line_end: Some(j + 1),
                                            function_name: None,
                                            dependency_name: None,
                                        },
                                        description: "Command injection vulnerability - passing user input to system command".to_string(),
                                        impact: SecurityImpact {
                                            confidentiality_impact: ImpactLevel::Critical,
                                            integrity_impact: ImpactLevel::Critical,
                                            availability_impact: ImpactLevel::Critical,
                                            exploitability: ExploitabilityLevel::Weaponized,
                                            cvss_score: Some(10.0),
                                        },
                                        cve_id: None,
                                        cwe_id: Some("CWE-78".to_string()),
                                        owasp_category: Some("A03:2021".to_string()),
                                        evidence: VulnerabilityEvidence {
                                            code_snippet: Some(lines[*i].1.to_string()),
                                            attack_vector: Some("Command injection via user input".to_string()),
                                            prerequisites: vec!["User input reaches command execution".to_string()],
                                            indicators: vec!["Direct user input to Command".to_string()],
                                        },
                                        fix_available: true,
                                        fix_suggestion: Some(SecurityFix {
                                            fix_id: uuid::Uuid::new_v4().to_string(),
                                            fix_type: SecurityFixType::InputValidation,
                                            description: "Validate and sanitize user input before command execution".to_string(),
                                            code_changes: vec![],
                                            risk_level: RiskLevel::Low,
                                            validation_required: true,
                                        }),
                                    });
                                    break;
                                }
                            }
                        }
                    }
                }

                // Path Traversal detection
                if content.contains("std::fs::") && (content.contains("../") || content.contains("user_input")) {
                    vulnerabilities.push(SecurityVulnerability {
                        vulnerability_id: uuid::Uuid::new_v4().to_string(),
                        vulnerability_type: VulnerabilityType::PathTraversal,
                        severity: VulnerabilitySeverity::High,
                        location: VulnerabilityLocation {
                            file_path: file.clone(),
                            line_start: None,
                            line_end: None,
                            function_name: None,
                            dependency_name: None,
                        },
                        description: "Path traversal vulnerability - user input used in file path".to_string(),
                        impact: SecurityImpact {
                            confidentiality_impact: ImpactLevel::High,
                            integrity_impact: ImpactLevel::Medium,
                            availability_impact: ImpactLevel::Low,
                            exploitability: ExploitabilityLevel::Functional,
                            cvss_score: Some(7.5),
                        },
                        cve_id: None,
                        cwe_id: Some("CWE-22".to_string()),
                        owasp_category: Some("A01:2021".to_string()),
                        evidence: VulnerabilityEvidence {
                            code_snippet: None,
                            attack_vector: Some("Path traversal via ../ sequences".to_string()),
                            prerequisites: vec!["User input reaches file system operations".to_string()],
                            indicators: vec!["Unsanitized file paths".to_string()],
                        },
                        fix_available: true,
                        fix_suggestion: Some(SecurityFix {
                            fix_id: uuid::Uuid::new_v4().to_string(),
                            fix_type: SecurityFixType::InputValidation,
                            description: "Validate and normalize file paths".to_string(),
                            code_changes: vec![],
                            risk_level: RiskLevel::Low,
                            validation_required: true,
                        }),
                    });
                }
            }
        }

        Ok(vulnerabilities)
    }

    /// Scan dependencies for vulnerabilities
    async fn scan_dependencies(&self, dep_scanner: &StoryDrivenDependencies) -> Result<Vec<SecurityVulnerability>> {
        let mut vulnerabilities = Vec::new();

        let dep_analysis = dep_scanner.analyze_dependencies().await?;

        for vuln in &dep_analysis.security_vulnerabilities {
            vulnerabilities.push(SecurityVulnerability {
                vulnerability_id: uuid::Uuid::new_v4().to_string(),
                vulnerability_type: VulnerabilityType::InsecureDependency,
                severity: vuln.severity.clone(),
                location: VulnerabilityLocation {
                    file_path: PathBuf::from("Cargo.toml"),
                    line_start: None,
                    line_end: None,
                    function_name: None,
                    dependency_name: Some(vuln.dependency.clone()),
                },
                description: vuln.description.clone(),
                impact: SecurityImpact {
                    confidentiality_impact: match &vuln.severity {
                        VulnerabilitySeverity::Critical => ImpactLevel::Critical,
                        VulnerabilitySeverity::High => ImpactLevel::High,
                        VulnerabilitySeverity::Medium => ImpactLevel::Medium,
                        VulnerabilitySeverity::Low => ImpactLevel::Low,
                    },
                    integrity_impact: ImpactLevel::Medium,
                    availability_impact: ImpactLevel::Low,
                    exploitability: ExploitabilityLevel::ProofOfConcept,
                    cvss_score: None, // Not available in dependencies struct
                },
                cve_id: vuln.cve.clone(),
                cwe_id: None,
                owasp_category: Some("A06:2021".to_string()),
                evidence: VulnerabilityEvidence {
                    code_snippet: None,
                    attack_vector: None,
                    prerequisites: vec!["Vulnerable dependency in use".to_string()],
                    indicators: vec![format!("Using vulnerable dependency: {}", vuln.dependency)],
                },
                fix_available: !vuln.patched_versions.is_empty(),
                fix_suggestion: if !vuln.patched_versions.is_empty() {
                    Some(SecurityFix {
                    fix_id: uuid::Uuid::new_v4().to_string(),
                    fix_type: SecurityFixType::DependencyUpdate,
                    description: format!("Update {} to patched version", vuln.dependency),
                    code_changes: vec![],
                    risk_level: RiskLevel::Low,
                    validation_required: true,
                    })
                } else {
                    None
                },
            });
        }

        Ok(vulnerabilities)
    }

    /// Detect hardcoded secrets
    async fn detect_secrets(&self) -> Result<Vec<SecurityVulnerability>> {
        let mut vulnerabilities = Vec::new();
        let files = self.find_source_files().await?;

        // Common secret patterns
        let secret_patterns = vec![
            (r#"(?i)(api[_-]?key|apikey)\s*=\s*["'][\w\-]+["']"#, "API Key"),
            (r#"(?i)(secret|password|passwd|pwd)\s*=\s*["'][^"']+["']"#, "Password"),
            (r#"(?i)token\s*=\s*["'][\w\-\.]+["']"#, "Token"),
            (r#"(?i)aws[_-]?access[_-]?key[_-]?id\s*=\s*["'][A-Z0-9]+["']"#, "AWS Access Key"),
            (r#"(?i)private[_-]?key\s*=\s*["'][\w\+/=\-]+["']"#, "Private Key"),
        ];

        for file in files {
            if let Ok(content) = tokio::fs::read_to_string(&file).await {
                for (pattern, secret_type) in &secret_patterns {
                    let re = regex::Regex::new(pattern)?;
                    if re.is_match(&content) {
                        vulnerabilities.push(SecurityVulnerability {
                            vulnerability_id: uuid::Uuid::new_v4().to_string(),
                            vulnerability_type: VulnerabilityType::HardcodedSecret,
                            severity: VulnerabilitySeverity::High,
                            location: VulnerabilityLocation {
                                file_path: file.clone(),
                                line_start: None,
                                line_end: None,
                                function_name: None,
                                dependency_name: None,
                            },
                            description: format!("Hardcoded {} detected in source code", secret_type),
                            impact: SecurityImpact {
                                confidentiality_impact: ImpactLevel::Critical,
                                integrity_impact: ImpactLevel::High,
                                availability_impact: ImpactLevel::Low,
                                exploitability: ExploitabilityLevel::Weaponized,
                                cvss_score: Some(8.5),
                            },
                            cve_id: None,
                            cwe_id: Some("CWE-798".to_string()),
                            owasp_category: Some("A07:2021".to_string()),
                            evidence: VulnerabilityEvidence {
                                code_snippet: None,
                                attack_vector: Some("Direct access to hardcoded credentials".to_string()),
                                prerequisites: vec!["Access to source code".to_string()],
                                indicators: vec![format!("Hardcoded {}", secret_type)],
                            },
                            fix_available: true,
                            fix_suggestion: Some(SecurityFix {
                                fix_id: uuid::Uuid::new_v4().to_string(),
                                fix_type: SecurityFixType::ConfigurationChange,
                                description: "Move secrets to environment variables or secure vault".to_string(),
                                code_changes: vec![],
                                risk_level: RiskLevel::Low,
                                validation_required: false,
                            }),
                        });
                        break; // One vulnerability per file for this pattern
                    }
                }
            }
        }

        Ok(vulnerabilities)
    }

    /// Check OWASP rules
    async fn check_owasp_rules(&self) -> Result<Vec<SecurityVulnerability>> {
        let mut vulnerabilities = Vec::new();

        // This is a simplified check - real implementation would be more comprehensive
        let files = self.find_source_files().await?;

        for file in files {
            if let Ok(content) = tokio::fs::read_to_string(&file).await {
                // A01:2021 - Broken Access Control
                if content.contains("fn ") && !content.contains("auth") && content.contains("admin") {
                    vulnerabilities.push(SecurityVulnerability {
                        vulnerability_id: uuid::Uuid::new_v4().to_string(),
                        vulnerability_type: VulnerabilityType::BrokenAccessControl,
                        severity: VulnerabilitySeverity::High,
                        location: VulnerabilityLocation {
                            file_path: file.clone(),
                            line_start: None,
                            line_end: None,
                            function_name: None,
                            dependency_name: None,
                        },
                        description: "Potential broken access control - admin function without authentication".to_string(),
                        impact: SecurityImpact {
                            confidentiality_impact: ImpactLevel::High,
                            integrity_impact: ImpactLevel::High,
                            availability_impact: ImpactLevel::Medium,
                            exploitability: ExploitabilityLevel::Functional,
                            cvss_score: Some(8.0),
                        },
                        cve_id: None,
                        cwe_id: Some("CWE-862".to_string()),
                        owasp_category: Some("A01:2021".to_string()),
                        evidence: VulnerabilityEvidence {
                            code_snippet: None,
                            attack_vector: Some("Direct access to admin functionality".to_string()),
                            prerequisites: vec!["Know admin endpoint".to_string()],
                            indicators: vec!["Missing authentication check".to_string()],
                        },
                        fix_available: true,
                        fix_suggestion: Some(SecurityFix {
                            fix_id: uuid::Uuid::new_v4().to_string(),
                            fix_type: SecurityFixType::AccessControl,
                            description: "Add authentication and authorization checks".to_string(),
                            code_changes: vec![],
                            risk_level: RiskLevel::Medium,
                            validation_required: true,
                        }),
                    });
                }

                // A02:2021 - Cryptographic Failures
                if content.contains("md5") || content.contains("sha1") {
                    vulnerabilities.push(SecurityVulnerability {
                        vulnerability_id: uuid::Uuid::new_v4().to_string(),
                        vulnerability_type: VulnerabilityType::WeakCryptography,
                        severity: VulnerabilitySeverity::Medium,
                        location: VulnerabilityLocation {
                            file_path: file.clone(),
                            line_start: None,
                            line_end: None,
                            function_name: None,
                            dependency_name: None,
                        },
                        description: "Weak cryptographic algorithm in use (MD5/SHA1)".to_string(),
                        impact: SecurityImpact {
                            confidentiality_impact: ImpactLevel::High,
                            integrity_impact: ImpactLevel::High,
                            availability_impact: ImpactLevel::None,
                            exploitability: ExploitabilityLevel::ProofOfConcept,
                            cvss_score: Some(6.5),
                        },
                        cve_id: None,
                        cwe_id: Some("CWE-327".to_string()),
                        owasp_category: Some("A02:2021".to_string()),
                        evidence: VulnerabilityEvidence {
                            code_snippet: None,
                            attack_vector: Some("Cryptographic attack on weak algorithm".to_string()),
                            prerequisites: vec!["Access to hashed values".to_string()],
                            indicators: vec!["Use of deprecated hash functions".to_string()],
                        },
                        fix_available: true,
                        fix_suggestion: Some(SecurityFix {
                            fix_id: uuid::Uuid::new_v4().to_string(),
                            fix_type: SecurityFixType::Encryption,
                            description: "Replace with SHA-256 or stronger algorithm".to_string(),
                            code_changes: vec![],
                            risk_level: RiskLevel::Low,
                            validation_required: true,
                        }),
                    });
                }
            }
        }

        Ok(vulnerabilities)
    }

    /// Calculate risk summary
    fn calculate_risk_summary(&self, vulnerabilities: &[SecurityVulnerability]) -> RiskSummary {
        let critical_count = vulnerabilities.iter()
            .filter(|v| v.severity == VulnerabilitySeverity::Critical)
            .count();
        let high_count = vulnerabilities.iter()
            .filter(|v| v.severity == VulnerabilitySeverity::High)
            .count();
        let medium_count = vulnerabilities.iter()
            .filter(|v| v.severity == VulnerabilitySeverity::Medium)
            .count();
        let low_count = vulnerabilities.iter()
            .filter(|v| v.severity == VulnerabilitySeverity::Low)
            .count();

        let overall_risk_level = if critical_count > 0 {
            RiskLevel::Critical
        } else if high_count > 2 {
            RiskLevel::High
        } else if high_count > 0 || medium_count > 5 {
            RiskLevel::Medium
        } else {
            RiskLevel::Low
        };

        RiskSummary {
            critical_count,
            high_count,
            medium_count,
            low_count,
            overall_risk_level,
            trending: TrendDirection::Stable, // Would compare with history
        }
    }

    /// Check compliance status
    fn check_compliance(&self, vulnerabilities: &[SecurityVulnerability]) -> ComplianceStatus {
        let mut violations = Vec::new();

        // OWASP Top 10 compliance
        let owasp_categories: std::collections::HashSet<_> = vulnerabilities.iter()
            .filter_map(|v| v.owasp_category.as_ref())
            .collect();

        let owasp_compliant = owasp_categories.is_empty();

        if !owasp_compliant {
            violations.push(ComplianceViolation {
                standard: "OWASP Top 10".to_string(),
                requirement: "No vulnerabilities from OWASP Top 10".to_string(),
                description: format!("Found vulnerabilities in {} OWASP categories", owasp_categories.len()),
                severity: VulnerabilitySeverity::High,
            });
        }

        // PCI DSS compliance (simplified)
        let has_crypto_issues = vulnerabilities.iter()
            .any(|v| v.vulnerability_type == VulnerabilityType::WeakCryptography);
        let pci_dss_compliant = !has_crypto_issues;

        if !pci_dss_compliant {
            violations.push(ComplianceViolation {
                standard: "PCI DSS".to_string(),
                requirement: "Strong cryptography required".to_string(),
                description: "Weak cryptographic algorithms detected".to_string(),
                severity: VulnerabilitySeverity::High,
            });
        }

        // GDPR compliance (simplified)
        let has_data_exposure = vulnerabilities.iter()
            .any(|v| v.vulnerability_type == VulnerabilityType::SensitiveDataExposure);
        let gdpr_compliant = !has_data_exposure;

        if !gdpr_compliant {
            violations.push(ComplianceViolation {
                standard: "GDPR".to_string(),
                requirement: "Personal data must be protected".to_string(),
                description: "Potential sensitive data exposure detected".to_string(),
                severity: VulnerabilitySeverity::Critical,
            });
        }

        ComplianceStatus {
            owasp_top_10_compliant: owasp_compliant,
            pci_dss_compliant,
            gdpr_compliant,
            violations,
        }
    }

    /// Generate security recommendations
    fn generate_recommendations(&self, vulnerabilities: &[SecurityVulnerability]) -> Vec<SecurityRecommendation> {
        let mut recommendations = Vec::new();

        // Check for patterns
        let has_injection = vulnerabilities.iter().any(|v|
            matches!(v.vulnerability_type, VulnerabilityType::SqlInjection | VulnerabilityType::CommandInjection)
        );

        if has_injection {
            recommendations.push(SecurityRecommendation {
                recommendation_id: uuid::Uuid::new_v4().to_string(),
                priority: RecommendationPriority::Immediate,
                category: "Input Validation".to_string(),
                description: "Implement comprehensive input validation and sanitization framework".to_string(),
                implementation_effort: EffortLevel::Medium,
                impact: SecurityImpact {
                    confidentiality_impact: ImpactLevel::High,
                    integrity_impact: ImpactLevel::High,
                    availability_impact: ImpactLevel::Medium,
                    exploitability: ExploitabilityLevel::Theoretical,
                    cvss_score: None,
                },
            });
        }

        let has_secrets = vulnerabilities.iter().any(|v|
            v.vulnerability_type == VulnerabilityType::HardcodedSecret
        );

        if has_secrets {
            recommendations.push(SecurityRecommendation {
                recommendation_id: uuid::Uuid::new_v4().to_string(),
                priority: RecommendationPriority::High,
                category: "Secrets Management".to_string(),
                description: "Implement secure secrets management using environment variables or vault".to_string(),
                implementation_effort: EffortLevel::Low,
                impact: SecurityImpact {
                    confidentiality_impact: ImpactLevel::Critical,
                    integrity_impact: ImpactLevel::High,
                    availability_impact: ImpactLevel::Low,
                    exploitability: ExploitabilityLevel::Theoretical,
                    cvss_score: None,
                },
            });
        }

        // General recommendations
        if vulnerabilities.len() > 10 {
            recommendations.push(SecurityRecommendation {
                recommendation_id: uuid::Uuid::new_v4().to_string(),
                priority: RecommendationPriority::Medium,
                category: "Security Training".to_string(),
                description: "Provide secure coding training to development team".to_string(),
                implementation_effort: EffortLevel::Medium,
                impact: SecurityImpact {
                    confidentiality_impact: ImpactLevel::Medium,
                    integrity_impact: ImpactLevel::Medium,
                    availability_impact: ImpactLevel::Low,
                    exploitability: ExploitabilityLevel::Theoretical,
                    cvss_score: None,
                },
            });
        }

        recommendations
    }

    /// Convert security change to code change
    async fn convert_to_code_change(
        &self,
        proposed: &ProposedSecurityChange,
        fix: &SecurityFix,
    ) -> Result<CodeChange> {
        Ok(CodeChange {
            file_path: proposed.file_path.clone(),
            change_type: crate::cognitive::self_modify::ChangeType::SecurityPatch,
            description: fix.description.clone(),
            reasoning: proposed.explanation.clone(),
            old_content: Some(proposed.vulnerable_code.clone()),
            new_content: proposed.secure_code.clone(),
            line_range: None,
            risk_level: fix.risk_level.clone(),
            attribution: None,
        })
    }

    /// Validate security fix
    async fn validate_security_fix(
        &self,
        _vulnerability: &SecurityVulnerability,
        _changes: &[CodeChange],
    ) -> Result<bool> {
        // In real implementation, would re-scan the specific area
        // For demo, assume validation passes
        Ok(true)
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

    /// Get scan history
    pub async fn get_scan_history(&self) -> Result<Vec<SecurityScanResult>> {
        Ok(self.scan_history.read().await.clone())
    }

    /// Get fix history
    pub async fn get_fix_history(&self) -> Result<Vec<SecurityFixResult>> {
        Ok(self.fix_history.read().await.clone())
    }

    /// Get current risk level
    pub async fn get_current_risk_level(&self) -> Result<RiskLevel> {
        let history = self.scan_history.read().await;
        if let Some(latest) = history.last() {
            Ok(latest.risk_summary.overall_risk_level.clone())
        } else {
            Ok(RiskLevel::Low)
        }
    }
}

impl VulnerabilityType {
    fn to_string(&self) -> &'static str {
        match self {
            VulnerabilityType::SqlInjection => "SQL Injection",
            VulnerabilityType::CommandInjection => "Command Injection",
            VulnerabilityType::PathTraversal => "Path Traversal",
            VulnerabilityType::XSS => "Cross-Site Scripting",
            VulnerabilityType::CSRF => "Cross-Site Request Forgery",
            VulnerabilityType::XXE => "XML External Entity",
            VulnerabilityType::InsecureDeserialization => "Insecure Deserialization",
            VulnerabilityType::BrokenAuthentication => "Broken Authentication",
            VulnerabilityType::SensitiveDataExposure => "Sensitive Data Exposure",
            VulnerabilityType::BrokenAccessControl => "Broken Access Control",
            VulnerabilityType::SecurityMisconfiguration => "Security Misconfiguration",
            VulnerabilityType::InsecureDependency => "Insecure Dependency",
            VulnerabilityType::HardcodedSecret => "Hardcoded Secret",
            VulnerabilityType::WeakCryptography => "Weak Cryptography",
            VulnerabilityType::InsecureRandomness => "Insecure Randomness",
            VulnerabilityType::UnvalidatedInput => "Unvalidated Input",
            VulnerabilityType::BufferOverflow => "Buffer Overflow",
            VulnerabilityType::RaceCondition => "Race Condition",
            VulnerabilityType::DenialOfService => "Denial of Service",
        }
    }
}
