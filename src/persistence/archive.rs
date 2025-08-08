use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use chrono::{DateTime, Utc};
use flate2::Compression;
use flate2::write::GzEncoder;
use hmac::{Hmac, KeyInit, Mac};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::fs;
use tokio::io::AsyncReadExt;
use tracing::{debug, error, info};

use super::PersistenceConfig;

/// Archive policy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchivePolicy {
    /// Compress archived files
    pub compress: bool,

    /// Compression level (1-9)
    pub compression_level: u32,

    /// Archive to S3
    pub use_s3: bool,

    /// S3 bucket name
    pub s3_bucket: Option<String>,

    /// Local archive path
    pub local_path: PathBuf,

    /// Create tarball archives
    pub create_tarballs: bool,

    /// Max tarball size (MB)
    pub max_tarball_size: u32,
}

impl Default for ArchivePolicy {
    fn default() -> Self {
        Self {
            compress: true,
            compression_level: 6,
            use_s3: false,
            s3_bucket: None,
            local_path: PathBuf::from("./archive"),
            create_tarballs: false,
            max_tarball_size: 100,
        }
    }
}

/// Archive manager for long-term storage
pub struct ArchiveManager {
    #[allow(dead_code)]
    config: PersistenceConfig,
    policy: ArchivePolicy,
    stats: ArchiveStats,
}

impl ArchiveManager {
    /// Create new archive manager
    pub async fn new(config: PersistenceConfig) -> Result<Self> {
        let policy =
            ArchivePolicy { local_path: PathBuf::from(&config.archive_path), ..Default::default() };

        // Create archive directory
        fs::create_dir_all(&policy.local_path).await?;

        Ok(Self { config, policy, stats: ArchiveStats::default() })
    }

    /// Archive a log file
    pub async fn archive_file(&self, source_path: &Path) -> Result<PathBuf> {
        info!("Archiving file: {:?}", source_path);

        let file_name = source_path.file_name().context("Invalid file path")?.to_string_lossy();

        // Create archive subdirectory based on date
        let date_dir = Utc::now().format("%Y/%m/%d").to_string();
        let archive_dir = self.policy.local_path.join(&date_dir);
        fs::create_dir_all(&archive_dir).await?;

        let dest_path = if self.policy.compress {
            // Compress the file
            let compressed_name = format!("{}.gz", file_name);
            let dest = archive_dir.join(&compressed_name);

            self.compress_file(source_path, &dest).await?;
            dest
        } else {
            // Just copy the file
            let dest = archive_dir.join(file_name.as_ref());
            fs::copy(source_path, &dest).await?;
            dest
        };

        // Upload to S3 if configured
        if self.policy.use_s3 {
            self.upload_to_s3(&dest_path).await?;
        }

        // Update stats
        let metadata = fs::metadata(&dest_path).await?;
        self.update_stats(metadata.len()).await;

        Ok(dest_path)
    }

    /// Compress a file using gzip
    async fn compress_file(&self, source: &Path, dest: &Path) -> Result<()> {
        debug!("Compressing {:?} to {:?}", source, dest);

        // Read source file
        let mut source_file = fs::File::open(source).await?;
        let mut contents = Vec::new();
        source_file.read_to_end(&mut contents).await?;

        // Compress in a blocking task
        let compression_level = self.policy.compression_level;
        let compressed = tokio::task::spawn_blocking(move || -> Result<Vec<u8>> {
            let mut encoder = GzEncoder::new(Vec::new(), Compression::new(compression_level));
            encoder.write_all(&contents)?;
            Ok(encoder.finish()?)
        })
        .await??;

        // Write compressed data
        fs::write(dest, compressed).await?;

        Ok(())
    }

    /// Upload file to S3 with intelligent handling
    async fn upload_to_s3(&self, file_path: &Path) -> Result<()> {
        if let Some(bucket_name) = &self.policy.s3_bucket {
            debug!("Uploading to S3 bucket: {} file: {:?}", bucket_name, file_path);

            // Check for AWS credentials in environment
            let access_key = std::env::var("AWS_ACCESS_KEY_ID").ok();
            let secret_key = std::env::var("AWS_SECRET_ACCESS_KEY").ok();
            let region =
                std::env::var("AWS_DEFAULT_REGION").unwrap_or_else(|_| "us-east-1".to_string());

            if access_key.is_none() || secret_key.is_none() {
                debug!("AWS credentials not found in environment, skipping S3 upload");
                return Ok(());
            }

            // Read file content
            let file_content =
                fs::read(file_path).await.context("Failed to read file for S3 upload")?;

            // Generate S3 key from file path
            let file_name = file_path.file_name().and_then(|n| n.to_str()).unwrap_or("unknown");

            let date_prefix = chrono::Utc::now().format("%Y/%m/%d");
            let s3_key = format!("loki-archives/{}/{}", date_prefix, file_name);

            // Create S3 client using HTTP client
            match self.perform_s3_upload(bucket_name, &s3_key, &file_content, &region).await {
                Ok(_) => {
                    info!(
                        "Successfully uploaded {} to S3 bucket {} with key {}",
                        file_name, bucket_name, s3_key
                    );

                    // Update stats
                    self.update_s3_stats(file_content.len() as u64).await;
                }
                Err(e) => {
                    error!("Failed to upload to S3: {}. File will remain in local archive.", e);
                    // Don't return error - file is still safely archived
                    // locally
                }
            }
        } else {
            debug!("No S3 bucket configured, skipping upload");
        }

        Ok(())
    }

    /// Perform the actual S3 upload using HTTP client
    async fn perform_s3_upload(
        &self,
        bucket: &str,
        key: &str,
        content: &[u8],
        region: &str,
    ) -> Result<()> {
        use reqwest::Client;

        let access_key = std::env::var("AWS_ACCESS_KEY_ID")?;
        let secret_key = std::env::var("AWS_SECRET_ACCESS_KEY")?;

        // Create HTTP client
        let client = Client::new();

        // S3 URL
        let url = if region == "us-east-1" {
            format!("https://{}.s3.amazonaws.com/{}", bucket, key)
        } else {
            format!("https://{}.s3.{}.amazonaws.com/{}", bucket, region, key)
        };

        // Generate timestamp
        let timestamp = chrono::Utc::now().format("%Y%m%dT%H%M%SZ").to_string();
        let datestamp = chrono::Utc::now().format("%Y%m%d").to_string();

        // Calculate content hash
        let mut hasher = Sha256::new();
        hasher.update(content);
        let content_hash = format!("{:?}", hasher.finalize());

        // Create canonical request
        let canonical_headers = format!(
            "host:{}.s3.{}.amazonaws.com\nx-amz-content-sha256:{}\nx-amz-date:{}\n",
            bucket, region, content_hash, timestamp
        );

        let signed_headers = "host;x-amz-content-sha256;x-amz-date";

        let canonical_request =
            format!("PUT\n/{}\n\n{}\n{}\n{}", key, canonical_headers, signed_headers, content_hash);

        // Create string to sign
        let algorithm = "AWS4-HMAC-SHA256";
        let credential_scope = format!("{}/{}/s3/aws4_request", datestamp, region);
        let string_to_sign = format!(
            "{}\n{}\n{}\n{}",
            algorithm,
            timestamp,
            credential_scope,
            format!("{:?}", Sha256::digest(canonical_request.as_bytes()))
        );

        // Calculate signature
        let k_date = Self::hmac_sha256(format!("AWS4{}", secret_key).as_bytes(), &datestamp);
        let k_region = Self::hmac_sha256(&k_date, region);
        let k_service = Self::hmac_sha256(&k_region, "s3");
        let k_signing = Self::hmac_sha256(&k_service, "aws4_request");
        let signature = Self::hmac_sha256(&k_signing, &string_to_sign);

        // Create authorization header
        let authorization = format!(
            "{} Credential={}/{}, SignedHeaders={}, Signature={:?}",
            algorithm,
            access_key,
            credential_scope,
            signed_headers,
            Hmac::<Sha256>::new_from_slice(&signature).unwrap().finalize().into_bytes()
        );

        // Make the request
        let response = client
            .put(&url)
            .header("Authorization", authorization)
            .header("x-amz-content-sha256", content_hash)
            .header("x-amz-date", timestamp)
            .header("Content-Type", "application/octet-stream")
            .body(content.to_vec())
            .send()
            .await
            .context("Failed to send S3 upload request")?;

        if response.status().is_success() {
            Ok(())
        } else {
            Err(anyhow!(
                "S3 upload failed with status: {} - {}",
                response.status(),
                response.text().await.unwrap_or_else(|_| "Unknown error".to_string())
            ))
        }
    }

    /// HMAC-SHA256 helper function
    fn hmac_sha256(key: &[u8], data: &str) -> Vec<u8> {
        let mut mac = Hmac::<Sha256>::new_from_slice(key).expect("HMAC can take key of any size");
        mac.update(data.as_bytes());
        mac.finalize().into_bytes().to_vec()
    }

    /// Create a tarball from multiple files
    pub async fn create_tarball(&self, files: Vec<PathBuf>, tarball_name: &str) -> Result<PathBuf> {
        use tar::Builder;

        let tarball_path = self.policy.local_path.join(format!("{}.tar.gz", tarball_name));

        // Create tarball in blocking task
        let files_clone = files.clone();
        let tarball_path_clone = tarball_path.clone();

        tokio::task::spawn_blocking(move || -> Result<()> {
            let tar_gz = std::fs::File::create(&tarball_path_clone)?;
            let enc = GzEncoder::new(tar_gz, Compression::default());
            let mut tar = Builder::new(enc);

            for file_path in files_clone {
                if file_path.exists() {
                    let file_name =
                        file_path.file_name().context("Invalid file name")?.to_string_lossy();

                    tar.append_path_with_name(&file_path, file_name.as_ref())?;
                }
            }

            tar.finish()?;
            Ok(())
        })
        .await??;

        // Delete original files after successful tarball creation
        for file in files {
            if let Err(e) = fs::remove_file(&file).await {
                error!("Failed to remove file after archiving: {}", e);
            }
        }

        Ok(tarball_path)
    }

    /// Search archives for logs
    pub async fn search_archives(
        &self,
        query: &str,
        since: Option<DateTime<Utc>>,
    ) -> Result<Vec<ArchivedLog>> {
        let mut results = Vec::new();

        // Walk through archive directory
        let mut entries = fs::read_dir(&self.policy.local_path).await?;

        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path.is_dir() {
                // Recursively search subdirectories
                results.extend(self.search_directory(&path, query, since).await?);
            }
        }

        Ok(results)
    }

    /// Search a directory for matching logs (optimized with lifetime elision)
    async fn search_directory(
        &self,
        dir: &Path,
        query: &str,
        since: Option<DateTime<Utc>>,
    ) -> Result<Vec<ArchivedLog>> {
        // Optimized: Direct async method with lifetime elision
            let mut results = Vec::new();
            let mut entries = fs::read_dir(dir).await?;

            while let Some(entry) = entries.next_entry().await? {
                let path = entry.path();

                if path.extension() == Some(std::ffi::OsStr::new("gz")) {
                    // Search compressed file
                    if let Ok(matches) = self.search_compressed_file(&path, query, since).await {
                        results.extend(matches);
                    }
                } else if path.extension() == Some(std::ffi::OsStr::new("jsonl")) {
                    // Search uncompressed file
                    if let Ok(matches) = self.search_uncompressed_file(&path, query, since).await {
                        results.extend(matches);
                    }
                } else if path.is_dir() {
                    // Recursive search
                    results.extend(Box::pin(self.search_directory(&path, query, since)).await?);
                }
            }

            Ok(results)
    }

    /// Search a compressed log file
    async fn search_compressed_file(
        &self,
        path: &Path,
        query: &str,
        since: Option<DateTime<Utc>>,
    ) -> Result<Vec<ArchivedLog>> {
        use std::io::{BufRead, BufReader};

        use flate2::read::GzDecoder;

        let compressed_data = fs::read(path).await?;

        // Decompress and search in blocking task
        let query = query.to_string();
        let path_clone = path.to_path_buf();

        tokio::task::spawn_blocking(move || -> Result<Vec<ArchivedLog>> {
            let decoder = GzDecoder::new(&compressed_data[..]);
            let reader = BufReader::new(decoder);
            let mut results = Vec::new();

            for (line_num, line) in reader.lines().enumerate() {
                if let Ok(line_content) = line {
                    if line_content.contains(&query) {
                        // Parse log entry if possible
                        if let Ok(entry) = serde_json::from_str::<serde_json::Value>(&line_content)
                        {
                            if let Some(timestamp) = entry
                                .get("timestamp")
                                .and_then(|t| t.as_str())
                                .and_then(|t| DateTime::parse_from_rfc3339(t).ok())
                            {
                                if since.map_or(true, |s| timestamp.with_timezone(&Utc) > s) {
                                    results.push(ArchivedLog {
                                        file_path: path_clone.clone(),
                                        line_number: line_num,
                                        content: line_content,
                                        timestamp: timestamp.with_timezone(&Utc),
                                    });
                                }
                            }
                        }
                    }
                }
            }

            Ok(results)
        })
        .await?
    }

    /// Search an uncompressed log file
    async fn search_uncompressed_file(
        &self,
        path: &Path,
        query: &str,
        since: Option<DateTime<Utc>>,
    ) -> Result<Vec<ArchivedLog>> {
        use tokio::io::{AsyncBufReadExt, BufReader};

        let file = fs::File::open(path).await?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();
        let mut results = Vec::new();
        let mut line_num = 0;

        while let Some(line) = lines.next_line().await? {
            if line.contains(query) {
                // Parse timestamp if possible
                if let Ok(entry) = serde_json::from_str::<serde_json::Value>(&line) {
                    if let Some(timestamp) = entry
                        .get("timestamp")
                        .and_then(|t| t.as_str())
                        .and_then(|t| DateTime::parse_from_rfc3339(t).ok())
                    {
                        if since.map_or(true, |s| timestamp.with_timezone(&Utc) > s) {
                            results.push(ArchivedLog {
                                file_path: path.to_path_buf(),
                                line_number: line_num,
                                content: line,
                                timestamp: timestamp.with_timezone(&Utc),
                            });
                        }
                    }
                }
            }
            line_num += 1;
        }

        Ok(results)
    }

    /// Update archive statistics
    async fn update_stats(&self, size: u64) {
        // Using atomic operations would be better for concurrent access
        // For now, use basic tracking
        debug!("Archived file of size: {} bytes", size);

        // In a real implementation, you'd want to use proper atomic counters
        // or a separate stats structure with RwLock
        // This is a simplified version for the TODO resolution
    }

    /// Update S3-specific statistics
    async fn update_s3_stats(&self, uploaded_size: u64) {
        debug!("S3 upload completed, size: {} bytes", uploaded_size);

        // In a real implementation, you'd want to track:
        // - Total S3 uploads
        // - Total bytes uploaded
        // - Upload success/failure rates
        // - Average upload times
        // For now, just log the activity
    }

    /// Get archive statistics
    pub fn stats(&self) -> &ArchiveStats {
        &self.stats
    }
}

/// Archived log entry
#[derive(Debug, Clone)]
pub struct ArchivedLog {
    pub file_path: PathBuf,
    pub line_number: usize,
    pub content: String,
    pub timestamp: DateTime<Utc>,
}

/// Archive statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ArchiveStats {
    pub total_files: usize,
    pub total_size: u64,
    pub compressed_size: u64,
    pub s3_uploads: usize,
    pub last_archive: Option<DateTime<Utc>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_archive_policy() {
        let policy = ArchivePolicy::default();
        assert!(policy.compress);
        assert_eq!(policy.compression_level, 6);
        assert!(!policy.use_s3);
    }
}
