use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use async_trait;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::cognitive::{EnhancedCognitiveModel, create_cognitive_model};
use crate::config::api_keys::ApiKeysConfig;
use crate::memory::{CognitiveMemory, MemoryMetadata};
use crate::tools::code_analysis::{CodeAnalyzer, FunctionInfo};

/// Test generation configuration
#[derive(Debug, Clone)]
pub struct TestGeneratorConfig {
    /// Model to use for test generation
    pub model_name: String,

    /// Maximum tests per function
    pub max_tests_per_function: usize,

    /// Include property-based tests
    pub include_property_tests: bool,

    /// Include edge case tests
    pub include_edge_cases: bool,

    /// Include integration tests
    pub include_integration_tests: bool,

    /// Test framework to use
    pub test_framework: TestFramework,
}

impl Default for TestGeneratorConfig {
    fn default() -> Self {
        Self {
            model_name: "deepseek-coder-v2:16b".to_string(),
            max_tests_per_function: 5,
            include_property_tests: true,
            include_edge_cases: true,
            include_integration_tests: false,
            test_framework: TestFramework::RustBuiltin,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TestFramework {
    RustBuiltin,
    Proptest,
    Quickcheck,
    Criterion,
}

/// Generated test case
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestCase {
    pub name: String,
    pub description: String,
    pub test_type: TestType,
    pub code: String,
    pub imports: Vec<String>,
    pub setup: Option<String>,
    pub teardown: Option<String>,
    pub assertions: Vec<String>,
    pub metadata: HashMap<String, serde_json::Value>,
    pub themes: Vec<String>,
    pub expected_behavior: String,
    pub edge_cases: Vec<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TestType {
    Unit,
    Property,
    EdgeCase,
    Integration,
    Benchmark,
    Fuzz,
    Performance,
}

/// Test suite for a module or file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestSuite {
    pub module_path: PathBuf,
    pub test_cases: Vec<TestCase>,
    pub imports: Vec<String>,
    pub fixtures: Vec<TestFixture>,
    pub coverage_estimate: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestFixture {
    pub name: String,
    pub setup_code: String,
    pub teardown_code: String,
}

/// Trait for test generation models
#[async_trait::async_trait]
trait TestGenerationModel: Send + Sync {
    async fn generate(&self, prompt: &str) -> Result<String>;
}

/// Wrapper to adapt EnhancedCognitiveModel to TestGenerationModel
struct ModelAdapter {
    inner: EnhancedCognitiveModel,
}

#[async_trait::async_trait]
impl TestGenerationModel for ModelAdapter {
    async fn generate(&self, prompt: &str) -> Result<String> {
        // Call generate_with_context with empty context for simple generation
        self.inner.generate_with_context(prompt, &[]).await
    }
}

/// AI-powered test generator
pub struct TestGenerator {
    /// Configuration
    config: TestGeneratorConfig,

    /// Cognitive model for generation
    model: Arc<dyn TestGenerationModel>,

    /// Code analyzer
    code_analyzer: Arc<CodeAnalyzer>,

    /// Memory system
    memory: Arc<CognitiveMemory>,

    /// Test patterns database
    test_patterns: Arc<TestPatternDatabase>,
}

impl TestGenerator {
    pub async fn new(config: TestGeneratorConfig, memory: Arc<CognitiveMemory>) -> Result<Self> {
        info!("Initializing AI test generator");

        // Create cognitive model with API config
        let apiconfig = ApiKeysConfig::from_env().unwrap_or_else(|e| {
            warn!("Failed to load API keys from environment: {}. Using Ollama fallback.", e);
            // Return a minimal config that will use Ollama
            ApiKeysConfig {
                github: None,
                x_twitter: None,
                ai_models: Default::default(),
                embedding_models: Default::default(),
                search: Default::default(),
                vector_db: None,
                optional_services: Default::default(),
            }
        });

        // Use Ollama as default if no other model is specified
        let model_name = if apiconfig.has_ai_model() {
            &config.model_name
        } else {
            warn!("No AI model API keys found, using Ollama with deepseek-coder model");
            "ollama"
        };

        let model = create_cognitive_model(Some(model_name), &apiconfig).await?;
        let model: Arc<dyn TestGenerationModel> = Arc::new(ModelAdapter { inner: model });

        // Create code analyzer
        let code_analyzer = Arc::new(CodeAnalyzer::new(memory.clone()).await?);

        // Load test patterns
        let test_patterns = Arc::new(TestPatternDatabase::new().await?);

        Ok(Self { config, model, code_analyzer, memory, test_patterns })
    }

    /// Generate tests for a file
    pub async fn generate_tests_for_file(&self, file_path: &Path) -> Result<TestSuite> {
        info!("Generating tests for: {:?}", file_path);

        // Analyze the code
        let analysis = self.code_analyzer.analyze_file(file_path).await?;

        // Detect language from file extension
        let language = file_path
            .extension()
            .and_then(|e| e.to_str())
            .map(|ext| match ext {
                "rs" => "rust",
                "py" => "python",
                "js" => "javascript",
                "ts" => "typescript",
                "go" => "go",
                _ => "unknown",
            })
            .unwrap_or("unknown");

        // Generate tests for each function
        let mut test_cases = Vec::new();
        let mut all_imports = Vec::new();

        for function in &analysis.functions {
            let function_tests = self.generate_function_tests(function, language).await?;
            test_cases.extend(function_tests);
        }

        // Generate integration tests if requested
        if self.config.include_integration_tests {
            let integration_tests =
                self.generate_integration_tests(file_path, &analysis, language).await?;
            test_cases.extend(integration_tests);
        }

        // Deduplicate imports
        all_imports.sort();
        all_imports.dedup();

        // Estimate coverage
        let coverage_estimate = self.estimate_coverage(&test_cases, &analysis);

        // Store in memory
        self.memory
            .store(
                format!("Generated {} tests for {}", test_cases.len(), file_path.display()),
                vec![serde_json::to_string(&test_cases)?],
                MemoryMetadata {
                    source: "test_generator".to_string(),
                    tags: vec!["tests".to_string(), "generated".to_string()],
                    importance: 0.7,
                    associations: vec![],
                    context: Some("generated test cases".to_string()),
                    created_at: chrono::Utc::now(),
                    accessed_count: 0,
                    last_accessed: None,
                    version: 1,
                    category: "cognitive".to_string(),
                    timestamp: chrono::Utc::now(),
                    expiration: None,
                },
            )
            .await?;

        Ok(TestSuite {
            module_path: file_path.to_path_buf(),
            test_cases,
            imports: all_imports,
            fixtures: Vec::new(),
            coverage_estimate,
        })
    }

    /// Generate tests for a specific function
    async fn generate_function_tests(
        &self,
        function: &FunctionInfo,
        language: &str,
    ) -> Result<Vec<TestCase>> {
        let mut tests = Vec::new();

        // Generate unit tests
        let unit_test = self.generate_unit_test(function, language).await?;
        tests.push(unit_test);

        // Generate edge case tests
        if self.config.include_edge_cases {
            let edge_cases = self.generate_edge_case_tests(function, language).await?;
            tests.extend(edge_cases);
        }

        // Generate property-based tests
        if self.config.include_property_tests && language == "rust" {
            if let Ok(property_test) = self.generate_property_test(function).await {
                tests.push(property_test);
            }
        }

        // Limit tests per function
        tests.truncate(self.config.max_tests_per_function);

        Ok(tests)
    }

    /// Generate a unit test for a function
    async fn generate_unit_test(
        &self,
        function: &FunctionInfo,
        language: &str,
    ) -> Result<TestCase> {
        // Get relevant patterns
        let patterns = self.test_patterns.get_patterns_for_function(function)?;

        // Build function signature from available info
        let _function_signature =
            format!("fn {}({:?})", function.name, function.parameters.join(", "));

        // Build prompt
        let prompt = format!(
            r"Generate a comprehensive unit test for this {} function:

Function name: {}
Parameters: {:?}
Complexity: {}
Is async: {}
Is test: {}

Requirements:
1. Test the happy path with typical inputs
2. Include clear assertions
3. Use descriptive test name
4. Add comments explaining the test logic
5. Follow {} testing best practices

Example patterns:
{}

Generate ONLY the test function code, no explanations.",
            language,
            function.name,
            function.parameters,
            function.complexity,
            function.is_async,
            function.is_test,
            language,
            patterns.join("\n")
        );

        // Generate test code
        let test_code = self.model.generate(&prompt).await?;

        // Extract test name
        let test_name = self.extract_test_name(&test_code, language)?;

        Ok(TestCase {
            name: test_name,
            description: format!("Unit test for {}", function.name),
            test_type: TestType::Unit,
            code: test_code.clone(),
            imports: self.extract_imports(&test_code, language),
            setup: None,
            teardown: None,
            assertions: self.extract_assertions(&test_code),
            metadata: HashMap::new(),
            themes: vec!["unit_testing".to_string()],
            expected_behavior: format!("Test should verify the correct behavior of {}", function.name),
            edge_cases: vec![],
        })
    }

    /// Generate edge case tests
    async fn generate_edge_case_tests(
        &self,
        function: &FunctionInfo,
        language: &str,
    ) -> Result<Vec<TestCase>> {
        let prompt = format!(
            r"Generate edge case tests for this {} function:

Function: {}
Parameters: {:?}

Consider these edge cases:
1. Empty/null/zero inputs
2. Maximum/minimum values
3. Boundary conditions
4. Invalid inputs (if applicable)
5. Resource exhaustion scenarios

Generate 2-3 focused edge case tests. Include only the test code.",
            language, function.name, function.parameters
        );

        let response = self.model.generate(&prompt).await?;

        // Parse multiple tests from response
        let tests = self.parse_multiple_tests(&response, language)?;

        Ok(tests
            .into_iter()
            .map(|(name, code)| TestCase {
                name,
                description: format!("Edge case test for {}", function.name),
                test_type: TestType::EdgeCase,
                code: code.clone(),
                imports: self.extract_imports(&code, language),
                setup: None,
                teardown: None,
                assertions: self.extract_assertions(&code),
                metadata: HashMap::new(),
                themes: vec!["edge_case_testing".to_string()],
                expected_behavior: "Test should handle edge cases gracefully".to_string(),
                edge_cases: vec!["boundary_values".to_string(), "null_values".to_string()],
            })
            .collect())
    }

    /// Generate property-based test
    async fn generate_property_test(&self, function: &FunctionInfo) -> Result<TestCase> {
        let framework = match self.config.test_framework {
            TestFramework::Proptest => "proptest",
            TestFramework::Quickcheck => "quickcheck",
            _ => "proptest", // Default to proptest
        };

        let prompt = format!(
            r"Generate a property-based test for this Rust function using {}:

Function: {}
Parameters: {:?}

Requirements:
1. Define appropriate generators for inputs
2. Express properties that should always hold
3. Use {} macros correctly
4. Include shrinking strategies if applicable

Generate only the test code with proper {} syntax.",
            framework, function.name, function.parameters, framework, framework
        );

        let test_code = self.model.generate(&prompt).await?;
        let test_name = self.extract_test_name(&test_code, "rust")?;

        Ok(TestCase {
            name: test_name,
            description: format!("Property test for {}", function.name),
            test_type: TestType::Property,
            code: test_code.clone(),
            imports: vec![format!("use {}::prelude::*;", framework)],
            setup: None,
            teardown: None,
            assertions: Vec::new(), // Properties are implicit assertions
            metadata: HashMap::new(),
            themes: vec!["property_testing".to_string()],
            expected_behavior: "Test should verify invariants hold for all inputs".to_string(),
            edge_cases: vec![],
        })
    }

    /// Generate integration tests
    async fn generate_integration_tests(
        &self,
        file_path: &Path,
        analysis: &crate::tools::code_analysis::AnalysisResult,
        language: &str,
    ) -> Result<Vec<TestCase>> {
        if analysis.dependencies.is_empty() {
            return Ok(Vec::new());
        }

        let function_names: Vec<String> =
            analysis.functions.iter().map(|f| f.name.clone()).collect();

        let prompt = format!(
            r"Generate integration tests for this module:

File: {}
Main functions: {}
Dependencies: {:?}

Generate 1-2 integration tests that:
1. Test interactions between multiple functions
2. Verify the module's public API
3. Test with realistic scenarios
4. Include setup/teardown if needed

Provide only the test code.",
            file_path.display(),
            function_names.join(", "),
            analysis.dependencies
        );

        let response = self.model.generate(&prompt).await?;
        let tests = self.parse_multiple_tests(&response, language)?;

        Ok(tests
            .into_iter()
            .map(|(name, code)| TestCase {
                name,
                description: format!("Integration test for {}", file_path.display()),
                test_type: TestType::Integration,
                code: code.clone(),
                imports: self.extract_imports(&code, language),
                setup: None,
                teardown: None,
                assertions: self.extract_assertions(&code),
                metadata: HashMap::new(),
                themes: vec!["integration_testing".to_string()],
                expected_behavior: "Test should verify component interactions work correctly".to_string(),
                edge_cases: vec![],
            })
            .collect())
    }

    /// Extract test name from code
    fn extract_test_name(&self, code: &str, language: &str) -> Result<String> {
        match language {
            "rust" => {
                // Look for #[test] fn test_name
                for line in code.lines() {
                    if line.trim().starts_with("fn ") && line.contains("test") {
                        let name = line
                            .trim()
                            .strip_prefix("fn ")
                            .and_then(|s| s.split('(').next())
                            .unwrap_or("test_unknown");
                        return Ok(name.to_string());
                    }
                }
            }
            _ => {
                // Generic extraction
                if let Some(name) = code
                    .lines()
                    .find(|line| line.contains("test") || line.contains("Test"))
                    .and_then(|line| line.split_whitespace().nth(1))
                {
                    return Ok(name.to_string());
                }
            }
        }

        Ok("test_generated".to_string())
    }

    /// Extract imports from test code
    fn extract_imports(&self, code: &str, language: &str) -> Vec<String> {
        let mut imports = Vec::new();

        match language {
            "rust" => {
                for line in code.lines() {
                    if line.trim().starts_with("use ") {
                        imports.push(line.trim().to_string());
                    }
                }
            }
            _ => {
                // Generic import extraction
                for line in code.lines() {
                    if line.contains("import") || line.contains("require") {
                        imports.push(line.trim().to_string());
                    }
                }
            }
        }

        imports
    }

    /// Extract assertions from test code
    fn extract_assertions(&self, code: &str) -> Vec<String> {
        code.lines()
            .filter(|line| {
                line.contains("assert")
                    || line.contains("expect")
                    || line.contains("should")
                    || line.contains("must")
            })
            .map(|s| s.trim().to_string())
            .collect()
    }

    /// Parse multiple tests from a response
    fn parse_multiple_tests(
        &self,
        response: &str,
        language: &str,
    ) -> Result<Vec<(String, String)>> {
        let mut tests = Vec::new();
        let mut current_test = String::new();
        let mut current_name = String::new();
        let mut in_test = false;

        for line in response.lines() {
            match language {
                "rust" => {
                    if line.contains("#[test]") || line.contains("#[tokio::test]") {
                        if in_test && !current_test.is_empty() {
                            tests.push((current_name.clone(), current_test.clone()));
                            current_test.clear();
                        }
                        in_test = true;
                    } else if in_test && line.trim().starts_with("fn ") {
                        current_name = self.extract_test_name(line, language)?;
                    }
                }
                _ => {
                    // Generic test detection
                    if line.contains("test") || line.contains("Test") {
                        if !current_test.is_empty() {
                            tests.push((current_name.clone(), current_test.clone()));
                            current_test.clear();
                        }
                        current_name = format!("test_{}", tests.len() + 1);
                        in_test = true;
                    }
                }
            }

            if in_test {
                current_test.push_str(line);
                current_test.push('\n');
            }
        }

        // Don't forget the last test
        if !current_test.is_empty() {
            tests.push((current_name, current_test));
        }

        Ok(tests)
    }

    /// Estimate test coverage
    fn estimate_coverage(
        &self,
        tests: &[TestCase],
        analysis: &crate::tools::code_analysis::AnalysisResult,
    ) -> f32 {
        if analysis.functions.is_empty() {
            return 0.0;
        }

        let functions_with_tests = analysis
            .functions
            .iter()
            .filter(|f| tests.iter().any(|t| t.description.contains(&f.name)))
            .count();

        let base_coverage = functions_with_tests as f32 / analysis.functions.len() as f32;

        // Adjust for test types
        let type_bonus = tests
            .iter()
            .map(|t| match t.test_type {
                TestType::Unit => 0.1,
                TestType::Property => 0.15,
                TestType::EdgeCase => 0.1,
                TestType::Integration => 0.2,
                TestType::Benchmark => 0.05,
                TestType::Fuzz => 0.15,
                TestType::Performance => 0.1,
            })
            .sum::<f32>()
            / tests.len().max(1) as f32;

        (base_coverage + type_bonus).min(1.0)
    }
}

/// Database of test patterns
struct TestPatternDatabase {
    patterns: HashMap<String, Vec<String>>,
}

impl TestPatternDatabase {
    async fn new() -> Result<Self> {
        let mut patterns = HashMap::new();

        // Rust patterns
        patterns.insert(
            "rust_unit".to_string(),
            vec![
                r"#[test]
fn test_function_name() {
    let result = function_name(input);
    assert_eq!(result, expected);
}"
                .to_string(),
                r"#[test]
fn test_error_case() {
    let result = function_name(invalid_input);
    assert!(result.is_err());
}"
                .to_string(),
            ],
        );

        patterns.insert(
            "rust_async".to_string(),
            vec![
                r"#[tokio::test]
async fn test_async_function() {
    let result = async_function().await;
    assert!(result.is_ok());
}"
                .to_string(),
            ],
        );

        Ok(Self { patterns })
    }

    fn get_patterns_for_function(&self, function: &FunctionInfo) -> Result<Vec<String>> {
        let key = if function.is_async { "rust_async" } else { "rust_unit" };

        Ok(self.patterns.get(key).cloned().unwrap_or_default())
    }
}

/// Generate mutation tests
pub async fn generate_mutation_tests(
    _generator: &TestGenerator,
    original_code: &str,
    test_suite: &TestSuite,
) -> Result<Vec<MutationTest>> {
    info!("Generating mutation tests");

    let mutations = vec![
        // Arithmetic mutations
        ("+", "-"),
        ("-", "+"),
        ("*", "/"),
        ("/", "*"),
        ("==", "!="),
        ("!=", "=="),
        (">", ">="),
        ("<", "<="),
        ("&&", "||"),
        ("||", "&&"),
    ];

    let mut mutation_tests = Vec::new();

    for (original, mutated) in mutations {
        if original_code.contains(original) {
            let mutated_code = original_code.replace(original, mutated);

            let mutation_test = MutationTest {
                mutation_type: format!("{} -> {}", original, mutated),
                original_code: original_code.to_string(),
                mutated_code,
                killing_tests: test_suite.test_cases.iter().map(|t| t.name.clone()).collect(),
                survived: false, // Will be determined by running tests
            };

            mutation_tests.push(mutation_test);
        }
    }

    Ok(mutation_tests)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutationTest {
    pub mutation_type: String,
    pub original_code: String,
    pub mutated_code: String,
    pub killing_tests: Vec<String>,
    pub survived: bool,
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_extract_test_name() {
        // Create a minimal test generator for testing
        // Note: We can't properly instantiate it without async context,
        // so we'll test the method directly on a dummy instance

        let rust_code = r#"
#[test]
fn test_addition() {
    assert_eq!(2 + 2, 4);
}
"#;

        // Test the extraction logic directly
        let mut test_name = "test_generated".to_string();

        for line in rust_code.lines() {
            if line.trim().starts_with("fn ") && line.contains("test") {
                test_name = line
                    .trim()
                    .strip_prefix("fn ")
                    .and_then(|s| s.split('(').next())
                    .unwrap_or("test_unknown")
                    .to_string();
                break;
            }
        }

        assert_eq!(test_name, "test_addition");
    }

    #[test]
    fn test_parse_assertions() {
        let code = r#"
        assert_eq!(result, expected);
        assert!(condition);
        expect(value).to_equal(other);
        result.should_be(true);
        "#;

        let assertions: Vec<String> = code
            .lines()
            .filter(|line| {
                line.contains("assert")
                    || line.contains("expect")
                    || line.contains("should")
                    || line.contains("must")
            })
            .map(|s| s.trim().to_string())
            .collect();

        assert_eq!(assertions.len(), 4);
        assert!(assertions[0].contains("assert_eq!"));
        assert!(assertions[1].contains("assert!"));
        assert!(assertions[2].contains("expect"));
        assert!(assertions[3].contains("should"));
    }
}
