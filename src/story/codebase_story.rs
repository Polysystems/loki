//! Codebase-specific story functionality

use super::types::*;
use anyhow::Result;
use std::path::{Path, PathBuf};
use std::collections::HashMap;
use tracing::{debug, info};
use walkdir::WalkDir;
use uuid::Uuid;

/// Represents a story for a codebase
pub struct CodebaseStory {
    pub story: Story,
    pub file_stories: HashMap<PathBuf, StoryId>,
    pub directory_stories: HashMap<PathBuf, StoryId>,
}

impl CodebaseStory {
    /// Analyze a codebase and create stories
    pub async fn analyze_codebase(
        root_path: &Path,
        story_engine: &crate::story::StoryEngine,
    ) -> Result<Self> {
        // Create root story
        let root_story_id = story_engine
            .create_codebase_story(
                root_path.to_path_buf(),
                detect_language(root_path)?,
            )
            .await?;

        let root_story = story_engine
            .get_story(&root_story_id)
            .ok_or_else(|| anyhow::anyhow!("Failed to get root story"))?;

        let mut file_stories = HashMap::new();
        let mut directory_stories = HashMap::new();

        // Walk the directory tree
        for entry in WalkDir::new(root_path)
            .follow_links(false)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            let path = entry.path();

            // Skip hidden files and common ignore patterns
            if should_ignore(path) {
                continue;
            }

            if path.is_dir() && path != root_path {
                // Create directory story
                let dir_story_id = create_directory_story(
                    path,
                    root_story_id,
                    story_engine,
                ).await?;

                directory_stories.insert(path.to_path_buf(), dir_story_id);

                // Add plot point to root story
                story_engine.add_plot_point(
                    root_story_id,
                    PlotType::Discovery {
                        insight: format!("Found directory: {}", path.display()),
                    },
                    vec![],
                ).await?;
            } else if path.is_file() {
                // Create file story
                let file_story_id = create_file_story(
                    path,
                    root_story_id,
                    story_engine,
                ).await?;

                file_stories.insert(path.to_path_buf(), file_story_id);

                // Analyze file content for plot points
                if let Ok(content) = tokio::fs::read_to_string(path).await {
                    analyze_file_content(
                        path,
                        &content,
                        file_story_id,
                        story_engine,
                    ).await?;
                }
            }
        }

        info!(
            "Analyzed codebase at {} - {} files, {} directories",
            root_path.display(),
            file_stories.len(),
            directory_stories.len()
        );

        Ok(CodebaseStory {
            story: root_story,
            file_stories,
            directory_stories,
        })
    }

    /// Get the story for a specific file
    pub fn get_file_story(&self, path: &Path) -> Option<StoryId> {
        self.file_stories.get(path).copied()
    }

    /// Get the story for a specific directory
    pub fn get_directory_story(&self, path: &Path) -> Option<StoryId> {
        self.directory_stories.get(path).copied()
    }

    /// Find all TODOs in the codebase
    pub async fn find_todos(&self, story_engine: &crate::story::StoryEngine) -> Result<Vec<MappedTask>> {
        let mut all_tasks = Vec::new();

        // Check all file stories for TODO plot points
        for (path, story_id) in &self.file_stories {
            if let Some(story) = story_engine.get_story(story_id) {
                for arc in &story.arcs {
                    for plot_point in &arc.plot_points {
                        if let PlotType::Task { description, completed } = &plot_point.plot_type {
                            if description.contains("TODO") || description.contains("FIXME") {
                                all_tasks.push(MappedTask {
                                    id: format!("todo-{}", plot_point.id.0),
                                    description: description.clone(),
                                    story_context: format!("In file: {}", path.display()),
                                    status: if *completed {
                                        TaskStatus::Completed
                                    } else {
                                        TaskStatus::Pending
                                    },
                                    assigned_to: None,
                                    created_at: plot_point.timestamp,
                                    updated_at: plot_point.timestamp,
                                    plot_point: Some(plot_point.id),
                                });
                            }
                        }
                    }
                }
            }
        }

        Ok(all_tasks)
    }
}

/// Detect the primary language of a codebase
fn detect_language(root_path: &Path) -> Result<String> {
    // Check for language-specific files
    let indicators = vec![
        ("Cargo.toml", "Rust"),
        ("package.json", "JavaScript/TypeScript"),
        ("requirements.txt", "Python"),
        ("go.mod", "Go"),
        ("pom.xml", "Java"),
        ("*.sln", "C#"),
        ("Gemfile", "Ruby"),
    ];

    for (file, lang) in indicators {
        if root_path.join(file).exists() {
            return Ok(lang.to_string());
        }
    }

    // Count file extensions
    let mut ext_count: HashMap<String, usize> = HashMap::new();

    for entry in WalkDir::new(root_path)
        .max_depth(3)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        if let Some(ext) = entry.path().extension() {
            *ext_count.entry(ext.to_string_lossy().to_string())
                .or_insert(0) += 1;
        }
    }

    // Map extensions to languages
    let ext_to_lang = HashMap::from([
        ("rs", "Rust"),
        ("js", "JavaScript"),
        ("ts", "TypeScript"),
        ("py", "Python"),
        ("go", "Go"),
        ("java", "Java"),
        ("cs", "C#"),
        ("rb", "Ruby"),
        ("cpp", "C++"),
        ("c", "C"),
    ]);

    let mut lang_count: HashMap<&str, usize> = HashMap::new();
    for (ext, count) in ext_count {
        if let Some(lang) = ext_to_lang.get(ext.as_str()) {
            *lang_count.entry(lang).or_insert(0) += count;
        }
    }

    Ok(lang_count
        .into_iter()
        .max_by_key(|(_, count)| *count)
        .map(|(lang, _)| lang.to_string())
        .unwrap_or_else(|| "Unknown".to_string()))
}

/// Check if a path should be ignored
fn should_ignore(path: &Path) -> bool {
    let ignore_patterns = vec![
        ".git",
        "node_modules",
        "target",
        ".idea",
        ".vscode",
        "__pycache__",
        ".pytest_cache",
    ];

    for component in path.components() {
        if let Some(name) = component.as_os_str().to_str() {
            if ignore_patterns.contains(&name) || name.starts_with('.') {
                return true;
            }
        }
    }

    false
}

/// Create a story for a directory
async fn create_directory_story(
    path: &Path,
    parent_story: StoryId,
    story_engine: &crate::story::StoryEngine,
) -> Result<StoryId> {
    let story_id = StoryId::new();
    let chain_id = ChainId(Uuid::new_v4());

    let story = Story {
        id: story_id,
        story_type: StoryType::Directory {
            path: path.to_path_buf(),
            parent_story: Some(parent_story),
        },
        title: format!("Directory: {}", path.file_name()
            .unwrap_or_default()
            .to_string_lossy()),
        description: format!("Story for directory at {}", path.display()),
        summary: format!("Story for directory at {}", path.display()),
        status: StoryStatus::Active,
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
        arcs: vec![],
        current_arc: None,
        metadata: StoryMetadata {
            tags: vec!["directory".to_string()],
            dependencies: vec![parent_story],
            related_stories: vec![],
            priority: Priority::Medium,
            complexity: 0.0,
            custom_data: HashMap::new(),
        },
        context_chain: chain_id,
        segments: vec![],
        context: HashMap::new(),
    };

    // Note: In a real implementation, we'd add this to the story engine
    debug!("Created directory story for {}", path.display());

    Ok(story_id)
}

/// Create a story for a file
async fn create_file_story(
    path: &Path,
    parent_story: StoryId,
    story_engine: &crate::story::StoryEngine,
) -> Result<StoryId> {
    let file_type = path.extension()
        .and_then(|e| e.to_str())
        .unwrap_or("unknown")
        .to_string();

    let story_id = StoryId::new();
    let chain_id = ChainId(Uuid::new_v4());

    let story = Story {
        id: story_id,
        story_type: StoryType::File {
            path: path.to_path_buf(),
            file_type: file_type.clone(),
            parent_story,
        },
        title: format!("File: {}", path.file_name()
            .unwrap_or_default()
            .to_string_lossy()),
        description: format!("Story for {} file at {}", file_type, path.display()),
        summary: format!("Story for {} file at {}", file_type, path.display()),
        status: StoryStatus::Active,
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
        arcs: vec![],
        current_arc: None,
        metadata: StoryMetadata {
            tags: vec!["file".to_string(), file_type],
            dependencies: vec![parent_story],
            related_stories: vec![],
            priority: Priority::Low,
            complexity: 0.0,
            custom_data: HashMap::new(),
        },
        context_chain: chain_id,
        segments: vec![],
        context: HashMap::new(),
    };

    debug!("Created file story for {}", path.display());

    Ok(story_id)
}

/// Analyze file content and create plot points
async fn analyze_file_content(
    path: &Path,
    content: &str,
    story_id: StoryId,
    story_engine: &crate::story::StoryEngine,
) -> Result<()> {
    // Look for TODOs and FIXMEs
    for (line_num, line) in content.lines().enumerate() {
        if line.contains("TODO") || line.contains("FIXME") {
            let task_desc = line.trim().to_string();
            story_engine.add_plot_point(
                story_id,
                PlotType::Task {
                    description: task_desc,
                    completed: false,
                },
                vec![format!("Line {}", line_num + 1)],
            ).await?;
        }
    }

    // Look for function/class definitions (simple heuristic)
    let code_patterns = vec![
        (r"^\s*fn\s+(\w+)", "function"),
        (r"^\s*class\s+(\w+)", "class"),
        (r"^\s*struct\s+(\w+)", "struct"),
        (r"^\s*impl\s+", "implementation"),
    ];

    for (pattern, element_type) in code_patterns {
        if let Ok(re) = regex::Regex::new(pattern) {
            for cap in re.captures_iter(content) {
                if let Some(name) = cap.get(1) {
                    story_engine.add_plot_point(
                        story_id,
                        PlotType::Discovery {
                            insight: format!("Found {}: {}", element_type, name.as_str()),
                        },
                        vec![],
                    ).await?;
                }
            }
        }
    }

    Ok(())
}
