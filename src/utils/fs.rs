use std::path::{Path, PathBuf};

use anyhow::Result;
use tokio::fs;

/// Read a file with size limit
pub async fn read_file_limited(path: &Path, max_size: usize) -> Result<String> {
    let metadata = fs::metadata(path).await?;

    if metadata.len() > max_size as u64 {
        anyhow::bail!("File too large: {} bytes (max: {} bytes)", metadata.len(), max_size);
    }

    Ok(fs::read_to_string(path).await?)
}

/// Find files matching a pattern
pub async fn find_files(root: &Path, pattern: &str) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    let mut dir = fs::read_dir(root).await?;

    while let Some(entry) = dir.next_entry().await? {
        let path = entry.path();

        if path.is_dir() {
            let mut subfiles = Box::pin(find_files(&path, pattern)).await?;
            files.append(&mut subfiles);
        } else if let Some(name) = path.file_name() {
            if name.to_string_lossy().contains(pattern) {
                files.push(path);
            }
        }
    }

    Ok(files)
}

/// Get relative path from base
pub fn relative_path(path: &Path, base: &Path) -> Option<PathBuf> {
    path.strip_prefix(base).ok().map(|p| p.to_path_buf())
}
