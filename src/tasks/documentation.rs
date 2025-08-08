use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::Result;
use async_trait::async_trait;
use regex::Regex;
use serde::Serialize;
use serde_json::json;
use tokio::fs;
use tracing::{debug, info, warn};

use crate::tasks::{Task, TaskArgs, TaskContext, TaskResult};

pub struct DocumentationTask;

/// Documentation configuration
#[derive(Debug, Clone, Serialize)]
struct DocConfig {
    output_format: DocumentationFormat,
    include_private: bool,
    include_tests: bool,
    generate_examples: bool,
    include_dependencies: bool,
    max_depth: usize,
}

/// Supported documentation formats
#[derive(Debug, Clone, Serialize)]
enum DocumentationFormat {
    Markdown,
    Html,
    Json,
    RustDoc,
}

/// Rust code analysis result
#[derive(Debug, Clone, Serialize)]
struct CodeAnalysis {
    modules: Vec<ModuleInfo>,
    functions: Vec<FunctionInfo>,
    structs: Vec<StructInfo>,
    enums: Vec<EnumInfo>,
    traits: Vec<TraitInfo>,
    constants: Vec<ConstantInfo>,
    dependencies: HashMap<String, String>,
}

/// Module information
#[derive(Debug, Clone, Serialize)]
struct ModuleInfo {
    name: String,
    path: PathBuf,
    doc_comment: Option<String>,
    visibility: Visibility,
    submodules: Vec<String>,
}

/// Function information
#[derive(Debug, Clone, Serialize)]
struct FunctionInfo {
    name: String,
    module: String,
    signature: String,
    doc_comment: Option<String>,
    visibility: Visibility,
    is_async: bool,
    parameters: Vec<Parameter>,
    return_type: Option<String>,
    examples: Vec<String>,
}

/// Struct information
#[derive(Debug, Clone, Serialize)]
struct StructInfo {
    name: String,
    module: String,
    doc_comment: Option<String>,
    visibility: Visibility,
    fields: Vec<FieldInfo>,
    derives: Vec<String>,
    generics: Vec<String>,
}

/// Enum information
#[derive(Debug, Clone, Serialize)]
struct EnumInfo {
    name: String,
    module: String,
    doc_comment: Option<String>,
    visibility: Visibility,
    variants: Vec<VariantInfo>,
    derives: Vec<String>,
}

/// Trait information
#[derive(Debug, Clone, Serialize)]
struct TraitInfo {
    name: String,
    module: String,
    doc_comment: Option<String>,
    visibility: Visibility,
    methods: Vec<FunctionInfo>,
    associated_types: Vec<String>,
}

/// Constant information
#[derive(Debug, Clone, Serialize)]
struct ConstantInfo {
    name: String,
    module: String,
    doc_comment: Option<String>,
    visibility: Visibility,
    value_type: String,
}

/// Field information for structs
#[derive(Debug, Clone, Serialize)]
struct FieldInfo {
    name: String,
    field_type: String,
    doc_comment: Option<String>,
    visibility: Visibility,
}

/// Enum variant information
#[derive(Debug, Clone, Serialize)]
struct VariantInfo {
    name: String,
    doc_comment: Option<String>,
    fields: Vec<FieldInfo>,
}

/// Function parameter information
#[derive(Debug, Clone, Serialize)]
struct Parameter {
    name: String,
    param_type: String,
    is_mutable: bool,
    is_reference: bool,
}

/// Visibility levels
#[derive(Debug, Clone, PartialEq, Serialize)]
enum Visibility {
    Public,
    Crate,
    Super,
    Private,
}

impl Default for DocConfig {
    fn default() -> Self {
        Self {
            output_format: DocumentationFormat::Markdown,
            include_private: false,
            include_tests: false,
            generate_examples: true,
            include_dependencies: true,
            max_depth: 10,
        }
    }
}

#[async_trait]
impl Task for DocumentationTask {
    fn name(&self) -> &str {
        "document"
    }

    fn description(&self) -> &str {
        "Generate comprehensive documentation for Rust code with intelligent analysis"
    }

    async fn execute(&self, args: TaskArgs, _context: TaskContext) -> Result<TaskResult> {
        let input_path =
            args.input.ok_or_else(|| anyhow::anyhow!("Input path required for documentation"))?;

        info!("Generating documentation for: {:?}", input_path);

        let config = DocConfig::default();
        let path = Path::new(&input_path);

        // Analyze the code structure
        let analysis = self.analyze_code_structure(path, &config).await?;

        // Generate documentation based on format
        let documentation = self.generate_documentation(&analysis, &config).await?;

        // Write documentation to output
        let output_path = self.determine_output_path(path, &config).await?;
        fs::write(&output_path, documentation).await?;

        info!("Documentation generated at: {:?}", output_path);

        Ok(TaskResult {
            success: true,
            message: format!("Documentation generated successfully at {:?}", output_path),
            data: Some(json!({
                "documented_path": input_path,
                "output_path": output_path,
                "format": format!("{:?}", config.output_format),
                "modules_count": analysis.modules.len(),
                "functions_count": analysis.functions.len(),
                "structs_count": analysis.structs.len(),
                "enums_count": analysis.enums.len(),
                "traits_count": analysis.traits.len(),
                "dependencies_count": analysis.dependencies.len(),
            })),
        })
    }
}

impl DocumentationTask {
    /// Analyze code structure recursively
    async fn analyze_code_structure(
        &self,
        path: &Path,
        _config: &DocConfig,
    ) -> Result<CodeAnalysis> {
        debug!("Analyzing code structure at: {:?}", path);

        let mut analysis = CodeAnalysis {
            modules: Vec::new(),
            functions: Vec::new(),
            structs: Vec::new(),
            enums: Vec::new(),
            traits: Vec::new(),
            constants: Vec::new(),
            dependencies: HashMap::new(),
        };

        if path.is_file() && path.extension().map_or(false, |ext| ext == "rs") {
            self.analyze_rust_file(path, &mut analysis, _config).await?;
        } else if path.is_dir() {
            self.analyze_directory(path, &mut analysis, _config, 0).await?;
        }

        Ok(analysis)
    }

    /// Analyze a single Rust file
    async fn analyze_rust_file(
        &self,
        file_path: &Path,
        analysis: &mut CodeAnalysis,
        _config: &DocConfig,
    ) -> Result<()> {
        let content = fs::read_to_string(file_path).await?;
        let module_name =
            file_path.file_stem().and_then(|s| s.to_str()).unwrap_or("unknown").to_string();

        debug!("Analyzing Rust file: {:?}", file_path);

        // Extract module information
        let doc_comment = self.extract_module_doc_comment(&content);
        analysis.modules.push(ModuleInfo {
            name: module_name.clone(),
            path: file_path.to_path_buf(),
            doc_comment,
            visibility: Visibility::Public, // Simplified
            submodules: self.extract_submodules(&content),
        });

        // Extract dependencies from use statements and Cargo.toml
        self.extract_dependencies(&content, &mut analysis.dependencies);

        // Extract functions
        analysis.functions.extend(self.extract_functions(&content, &module_name));

        // Extract structs
        analysis.structs.extend(self.extract_structs(&content, &module_name));

        // Extract enums
        analysis.enums.extend(self.extract_enums(&content, &module_name));

        // Extract traits
        analysis.traits.extend(self.extract_traits(&content, &module_name));

        // Extract constants
        analysis.constants.extend(self.extract_constants(&content, &module_name));

        Ok(())
    }

    /// Analyze directory recursively
    async fn analyze_directory(
        &self,
        dir_path: &Path,
        analysis: &mut CodeAnalysis,
        _config: &DocConfig,
        depth: usize,
    ) -> Result<()> {
        if depth >= _config.max_depth {
            warn!("Maximum analysis depth reached at: {:?}", dir_path);
            return Ok(());
        }

        let mut entries = fs::read_dir(dir_path).await?;
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();

            if path.is_file() && path.extension().map_or(false, |ext| ext == "rs") {
                // Skip test files if not including tests
                if !_config.include_tests && path.to_string_lossy().contains("test") {
                    continue;
                }

                self.analyze_rust_file(&path, analysis, _config).await?;
            } else if path.is_dir()
                && !path.file_name().unwrap_or_default().to_string_lossy().starts_with('.')
            {
                Box::pin(self.analyze_directory(&path, analysis, _config, depth + 1)).await?;
            }
        }

        Ok(())
    }

    /// Extract module-level doc comment
    fn extract_module_doc_comment(&self, content: &str) -> Option<String> {
        let doc_regex = Regex::new(r"^//!\s*(.*)$").unwrap();
        let mut doc_lines = Vec::new();

        for line in content.lines() {
            if let Some(caps) = doc_regex.captures(line) {
                doc_lines.push(caps.get(1).unwrap().as_str().to_string());
            } else if !line.trim().is_empty() && !line.trim().starts_with("//") {
                break; // Stop at first non-doc content
            }
        }

        if doc_lines.is_empty() { None } else { Some(doc_lines.join("\n")) }
    }

    /// Extract submodule declarations
    fn extract_submodules(&self, content: &str) -> Vec<String> {
        let mod_regex = Regex::new(r"mod\s+(\w+)").unwrap();
        mod_regex
            .captures_iter(content)
            .filter_map(|caps| caps.get(1).map(|m| m.as_str().to_string()))
            .collect()
    }

    /// Extract dependencies from use statements
    fn extract_dependencies(&self, content: &str, dependencies: &mut HashMap<String, String>) {
        let use_regex = Regex::new(r"use\s+(\w+)").unwrap();
        for caps in use_regex.captures_iter(content) {
            if let Some(crate_name) = caps.get(1) {
                let name = crate_name.as_str().to_string();
                if !["std", "core", "alloc", "crate", "self", "super"].contains(&name.as_str()) {
                    dependencies.insert(name.clone(), "unknown".to_string()); // Version would require Cargo.toml parsing
                }
            }
        }
    }

    /// Extract function information
    fn extract_functions(&self, content: &str, module: &str) -> Vec<FunctionInfo> {
        let fn_regex = Regex::new(r"(?m)^(?:\s*)(?:(pub(?:\([^)]*\))?)\s+)?(async\s+)?fn\s+(\w+)\s*\([^)]*\)(?:\s*->\s*([^{]+))?").unwrap();
        let mut functions = Vec::new();

        for caps in fn_regex.captures_iter(content) {
            let visibility = match caps.get(1) {
                Some(_) => Visibility::Public,
                None => Visibility::Private,
            };

            let is_async = caps.get(2).is_some();
            let name = caps.get(3).unwrap().as_str().to_string();
            let return_type = caps.get(4).map(|m| m.as_str().trim().to_string());

            // Extract doc comment (simplified - would need more sophisticated parsing)
            let doc_comment = self.extract_doc_comment_for_item(content, &name);

            functions.push(FunctionInfo {
                name: name.clone(),
                module: module.to_string(),
                signature: caps.get(0).unwrap().as_str().to_string(),
                doc_comment,
                visibility,
                is_async,
                parameters: Vec::new(), // Simplified - would need proper parsing
                return_type,
                examples: Vec::new(), // Would extract from doc comments
            });
        }

        functions
    }

    /// Extract struct information
    fn extract_structs(&self, content: &str, module: &str) -> Vec<StructInfo> {
        let struct_regex =
            Regex::new(r"(?m)^(?:\s*)(?:(pub(?:\([^)]*\))?)\s+)?struct\s+(\w+)").unwrap();
        let mut structs = Vec::new();

        for caps in struct_regex.captures_iter(content) {
            let visibility = match caps.get(1) {
                Some(_) => Visibility::Public,
                None => Visibility::Private,
            };

            let name = caps.get(2).unwrap().as_str().to_string();
            let doc_comment = self.extract_doc_comment_for_item(content, &name);

            structs.push(StructInfo {
                name: name.clone(),
                module: module.to_string(),
                doc_comment,
                visibility,
                fields: Vec::new(),   // Simplified - would need field parsing
                derives: Vec::new(),  // Would extract from #[derive(...)]
                generics: Vec::new(), // Would extract generic parameters
            });
        }

        structs
    }

    /// Extract enum information
    fn extract_enums(&self, content: &str, module: &str) -> Vec<EnumInfo> {
        let enum_regex =
            Regex::new(r"(?m)^(?:\s*)(?:(pub(?:\([^)]*\))?)\s+)?enum\s+(\w+)").unwrap();
        let mut enums = Vec::new();

        for caps in enum_regex.captures_iter(content) {
            let visibility = match caps.get(1) {
                Some(_) => Visibility::Public,
                None => Visibility::Private,
            };

            let name = caps.get(2).unwrap().as_str().to_string();
            let doc_comment = self.extract_doc_comment_for_item(content, &name);

            enums.push(EnumInfo {
                name: name.clone(),
                module: module.to_string(),
                doc_comment,
                visibility,
                variants: Vec::new(), // Simplified - would need variant parsing
                derives: Vec::new(),
            });
        }

        enums
    }

    /// Extract trait information
    fn extract_traits(&self, content: &str, module: &str) -> Vec<TraitInfo> {
        let trait_regex =
            Regex::new(r"(?m)^(?:\s*)(?:(pub(?:\([^)]*\))?)\s+)?trait\s+(\w+)").unwrap();
        let mut traits = Vec::new();

        for caps in trait_regex.captures_iter(content) {
            let visibility = match caps.get(1) {
                Some(_) => Visibility::Public,
                None => Visibility::Private,
            };

            let name = caps.get(2).unwrap().as_str().to_string();
            let doc_comment = self.extract_doc_comment_for_item(content, &name);

            traits.push(TraitInfo {
                name: name.clone(),
                module: module.to_string(),
                doc_comment,
                visibility,
                methods: Vec::new(),          // Would extract trait methods
                associated_types: Vec::new(), // Would extract associated types
            });
        }

        traits
    }

    /// Extract constant information
    fn extract_constants(&self, content: &str, module: &str) -> Vec<ConstantInfo> {
        let const_regex =
            Regex::new(r"(?m)^(?:\s*)(?:(pub(?:\([^)]*\))?)\s+)?const\s+(\w+):\s*([^=]+)").unwrap();
        let mut constants = Vec::new();

        for caps in const_regex.captures_iter(content) {
            let visibility = match caps.get(1) {
                Some(_) => Visibility::Public,
                None => Visibility::Private,
            };

            let name = caps.get(2).unwrap().as_str().to_string();
            let value_type = caps.get(3).unwrap().as_str().trim().to_string();
            let doc_comment = self.extract_doc_comment_for_item(content, &name);

            constants.push(ConstantInfo {
                name: name.clone(),
                module: module.to_string(),
                doc_comment,
                visibility,
                value_type,
            });
        }

        constants
    }

    /// Extract doc comment for a specific item (simplified)
    fn extract_doc_comment_for_item(&self, content: &str, item_name: &str) -> Option<String> {
        let lines: Vec<&str> = content.lines().collect();

        for (i, line) in lines.iter().enumerate() {
            if line.contains(item_name) {
                // Look backwards for doc comments
                let mut doc_lines = Vec::new();
                for j in (0..i).rev() {
                    let prev_line = lines[j].trim();
                    if prev_line.starts_with("///") {
                        let doc_content = prev_line.trim_start_matches("///").trim();
                        doc_lines.insert(0, doc_content.to_string());
                    } else if !prev_line.is_empty() && !prev_line.starts_with("//") {
                        break;
                    }
                }

                if !doc_lines.is_empty() {
                    return Some(doc_lines.join("\n"));
                }
            }
        }

        None
    }

    /// Generate documentation in the specified format
    async fn generate_documentation(
        &self,
        analysis: &CodeAnalysis,
        _config: &DocConfig,
    ) -> Result<String> {
        match _config.output_format {
            DocumentationFormat::Markdown => self.generate_markdown_docs(analysis, _config).await,
            DocumentationFormat::Html => self.generate_html_docs(analysis, _config).await,
            DocumentationFormat::Json => self.generate_json_docs(analysis, _config).await,
            DocumentationFormat::RustDoc => self.generate_rustdoc_style(analysis, _config).await,
        }
    }

    /// Generate Markdown documentation
    async fn generate_markdown_docs(
        &self,
        analysis: &CodeAnalysis,
        _config: &DocConfig,
    ) -> Result<String> {
        let mut doc = String::new();

        // Title and overview
        doc.push_str("# Code Documentation\n\n");
        doc.push_str(&format!(
            "Generated on: {}\n\n",
            chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
        ));

        // Table of contents
        doc.push_str("## Table of Contents\n\n");
        if !analysis.modules.is_empty() {
            doc.push_str("- [Modules](#modules)\n");
        }
        if !analysis.functions.is_empty() {
            doc.push_str("- [Functions](#functions)\n");
        }
        if !analysis.structs.is_empty() {
            doc.push_str("- [Structs](#structs)\n");
        }
        if !analysis.enums.is_empty() {
            doc.push_str("- [Enums](#enums)\n");
        }
        if !analysis.traits.is_empty() {
            doc.push_str("- [Traits](#traits)\n");
        }
        if !analysis.constants.is_empty() {
            doc.push_str("- [Constants](#constants)\n");
        }
        if !analysis.dependencies.is_empty() {
            doc.push_str("- [Dependencies](#dependencies)\n");
        }
        doc.push_str("\n");

        // Modules section
        if !analysis.modules.is_empty() {
            doc.push_str("## Modules\n\n");
            for module in &analysis.modules {
                doc.push_str(&format!("### {}\n\n", module.name));
                doc.push_str(&format!("**Path:** `{}`\n\n", module.path.display()));

                if let Some(ref doc_comment) = module.doc_comment {
                    doc.push_str(&format!("{}\n\n", doc_comment));
                }

                if !module.submodules.is_empty() {
                    doc.push_str("**Submodules:**\n");
                    for submodule in &module.submodules {
                        doc.push_str(&format!("- `{}`\n", submodule));
                    }
                    doc.push_str("\n");
                }
            }
        }

        // Functions section
        if !analysis.functions.is_empty() {
            doc.push_str("## Functions\n\n");
            for function in &analysis.functions {
                if function.visibility == Visibility::Public {
                    doc.push_str(&format!("### {}\n\n", function.name));
                    doc.push_str(&format!("**Module:** `{}`\n\n", function.module));
                    let params_str = function.parameters
                        .iter()
                        .map(|p| format!("{}{}{}", 
                            if p.is_mutable { "mut " } else { "" },
                            p.name,
                            if !p.param_type.is_empty() { format!(": {}", p.param_type) } else { String::new() }
                        ))
                        .collect::<Vec<_>>()
                        .join(", ");
                    let signature = format!("{}({})", function.name, params_str);
                    doc.push_str(&format!(
                        "**Signature:**\n```rust\n{}\n```\n\n",
                        signature
                    ));

                    if function.is_async {
                        doc.push_str("**Async:** ✓\n\n");
                    }

                    if let Some(ref doc_comment) = function.doc_comment {
                        doc.push_str(&format!("{}\n\n", doc_comment));
                    }

                    if let Some(ref return_type) = function.return_type {
                        doc.push_str(&format!("**Returns:** `{}`\n\n", return_type));
                    }
                }
            }
        }

        // Structs section
        if !analysis.structs.is_empty() {
            doc.push_str("## Structs\n\n");
            for struct_info in &analysis.structs {
                if struct_info.visibility == Visibility::Public {
                    doc.push_str(&format!("### {}\n\n", struct_info.name));
                    doc.push_str(&format!("**Module:** `{}`\n\n", struct_info.module));

                    if let Some(ref doc_comment) = struct_info.doc_comment {
                        doc.push_str(&format!("{}\n\n", doc_comment));
                    }
                }
            }
        }

        // Enums section
        if !analysis.enums.is_empty() {
            doc.push_str("## Enums\n\n");
            for enum_info in &analysis.enums {
                if enum_info.visibility == Visibility::Public {
                    doc.push_str(&format!("### {}\n\n", enum_info.name));
                    doc.push_str(&format!("**Module:** `{}`\n\n", enum_info.module));

                    if let Some(ref doc_comment) = enum_info.doc_comment {
                        doc.push_str(&format!("{}\n\n", doc_comment));
                    }
                }
            }
        }

        // Traits section
        if !analysis.traits.is_empty() {
            doc.push_str("## Traits\n\n");
            for trait_info in &analysis.traits {
                if trait_info.visibility == Visibility::Public {
                    doc.push_str(&format!("### {}\n\n", trait_info.name));
                    doc.push_str(&format!("**Module:** `{}`\n\n", trait_info.module));

                    if let Some(ref doc_comment) = trait_info.doc_comment {
                        doc.push_str(&format!("{}\n\n", doc_comment));
                    }
                }
            }
        }

        // Constants section
        if !analysis.constants.is_empty() {
            doc.push_str("## Constants\n\n");
            for constant in &analysis.constants {
                if constant.visibility == Visibility::Public {
                    doc.push_str(&format!("### {}\n\n", constant.name));
                    doc.push_str(&format!("**Module:** `{}`\n\n", constant.module));
                    doc.push_str(&format!("**Type:** `{}`\n\n", constant.value_type));

                    if let Some(ref doc_comment) = constant.doc_comment {
                        doc.push_str(&format!("{}\n\n", doc_comment));
                    }
                }
            }
        }

        // Dependencies section
        if !analysis.dependencies.is_empty() {
            doc.push_str("## Dependencies\n\n");
            doc.push_str("| Crate | Version |\n");
            doc.push_str("|-------|--------|\n");
            for (name, version) in &analysis.dependencies {
                doc.push_str(&format!("| `{}` | {} |\n", name, version));
            }
            doc.push_str("\n");
        }

        Ok(doc)
    }

    /// Generate HTML documentation (simplified)
    async fn generate_html_docs(
        &self,
        analysis: &CodeAnalysis,
        _config: &DocConfig,
    ) -> Result<String> {
        let markdown = self.generate_markdown_docs(analysis, _config).await?;
        // In a real implementation, you'd use a markdown-to-HTML converter like
        // pulldown-cmark
        Ok(format!("<html><body><pre>{}</pre></body></html>", markdown))
    }

    /// Generate JSON documentation
    async fn generate_json_docs(
        &self,
        analysis: &CodeAnalysis,
        _config: &DocConfig,
    ) -> Result<String> {
        let json_data = json!({
            "modules": analysis.modules,
            "functions": analysis.functions,
            "structs": analysis.structs,
            "enums": analysis.enums,
            "traits": analysis.traits,
            "constants": analysis.constants,
            "dependencies": analysis.dependencies,
            "generated_at": chrono::Utc::now().to_rfc3339(),
        });

        Ok(serde_json::to_string_pretty(&json_data)?)
    }

    /// Generate Rust-doc style documentation
    async fn generate_rustdoc_style(
        &self,
        analysis: &CodeAnalysis,
        _config: &DocConfig,
    ) -> Result<String> {
        // For now, generate markdown in rustdoc style
        self.generate_markdown_docs(analysis, _config).await
    }

    /// Determine output path based on input and configuration
    async fn determine_output_path(
        &self,
        input_path: &Path,
        _config: &DocConfig,
    ) -> Result<PathBuf> {
        let extension = match _config.output_format {
            DocumentationFormat::Markdown => "md",
            DocumentationFormat::Html => "html",
            DocumentationFormat::Json => "json",
            DocumentationFormat::RustDoc => "md",
        };

        let output_name = if input_path.is_file() {
            format!(
                "{}_docs.{}",
                input_path.file_stem().unwrap_or_default().to_string_lossy(),
                extension
            )
        } else {
            format!(
                "{}_docs.{}",
                input_path.file_name().unwrap_or_default().to_string_lossy(),
                extension
            )
        };

        let output_dir = input_path.parent().unwrap_or(Path::new("."));
        Ok(output_dir.join(output_name))
    }
}
