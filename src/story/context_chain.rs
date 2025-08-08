//! Context chaining implementation for story continuity

use super::types::*;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::path::PathBuf;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Maximum number of segments in a chain
const MAX_CHAIN_SEGMENTS: usize = 100;

/// A context chain maintains narrative continuity across story elements
#[derive(Debug)]
pub struct ContextChain {
    pub id: ChainId,
    pub story_id: StoryId,
    pub segments: RwLock<VecDeque<ContextSegment>>,
    pub links: RwLock<HashMap<SegmentId, Vec<ContextLink>>>,
    pub metadata: RwLock<ChainMetadata>,
}

/// Unique identifier for a context segment
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SegmentId(pub Uuid);

impl fmt::Display for SegmentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A segment in the context chain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextSegment {
    pub id: SegmentId,
    pub content: String,
    pub tokens: Vec<String>,
    pub embedding: Option<Vec<f32>>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub segment_type: SegmentType,
    pub importance: f32,
}

/// Types of context segments
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SegmentType {
    PlotPoint(PlotPointId),
    Summary,
    Transition,
    Reference(StoryId),
    Agent(String),
    Code(PathBuf),
}

impl fmt::Display for SegmentType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SegmentType::PlotPoint(id) => write!(f, "PlotPoint({})", id),
            SegmentType::Summary => write!(f, "Summary"),
            SegmentType::Transition => write!(f, "Transition"),
            SegmentType::Reference(id) => write!(f, "Reference({})", id),
            SegmentType::Agent(name) => write!(f, "Agent({})", name),
            SegmentType::Code(path) => write!(f, "Code({})", path.display()),
        }
    }
}

/// A link between context segments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextLink {
    pub from: SegmentId,
    pub to: SegmentId,
    pub link_type: LinkType,
    pub strength: f32,
}

/// Types of links between segments
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LinkType {
    Sequential,
    Causal,
    Reference,
    Parallel,
    Conflict,
}

impl fmt::Display for LinkType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LinkType::Sequential => write!(f, "Sequential"),
            LinkType::Causal => write!(f, "Causal"),
            LinkType::Reference => write!(f, "Reference"),
            LinkType::Parallel => write!(f, "Parallel"),
            LinkType::Conflict => write!(f, "Conflict"),
        }
    }
}

/// Metadata for the context chain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainMetadata {
    pub total_segments: usize,
    pub compressed_segments: usize,
    pub last_compression: Option<chrono::DateTime<chrono::Utc>>,
    pub average_importance: f32,
}

impl ContextChain {
    /// Create a new context chain
    pub fn new(id: ChainId, story_id: StoryId) -> Self {
        Self {
            id,
            story_id,
            segments: RwLock::new(VecDeque::new()),
            links: RwLock::new(HashMap::new()),
            metadata: RwLock::new(ChainMetadata {
                total_segments: 0,
                compressed_segments: 0,
                last_compression: None,
                average_importance: 0.0,
            }),
        }
    }
    
    /// Add a plot point to the context chain
    pub async fn add_plot_point(&self, plot_point: &PlotPoint) -> Result<SegmentId> {
        let segment_id = SegmentId(Uuid::new_v4());
        
        let segment = ContextSegment {
            id: segment_id,
            content: plot_point.description.clone(),
            tokens: plot_point.context_tokens.clone(),
            embedding: None, // Would be computed by embedding service
            created_at: chrono::Utc::now(),
            segment_type: SegmentType::PlotPoint(plot_point.id),
            importance: plot_point.importance,
        };
        
        self.add_segment(segment).await?;
        
        Ok(segment_id)
    }
    
    /// Add a summary segment
    pub async fn add_summary(&self, summary: String, importance: f32) -> Result<SegmentId> {
        let segment_id = SegmentId(Uuid::new_v4());
        
        let segment = ContextSegment {
            id: segment_id,
            content: summary.clone(),
            tokens: self.tokenize(&summary),
            embedding: None,
            created_at: chrono::Utc::now(),
            segment_type: SegmentType::Summary,
            importance,
        };
        
        self.add_segment(segment).await?;
        
        Ok(segment_id)
    }
    
    /// Add a code reference segment
    pub async fn add_code_reference(
        &self,
        path: PathBuf,
        description: String,
        importance: f32,
    ) -> Result<SegmentId> {
        let segment_id = SegmentId(Uuid::new_v4());
        
        let segment = ContextSegment {
            id: segment_id,
            content: format!("Code at {}: {}", path.display(), description),
            tokens: self.tokenize(&description),
            embedding: None,
            created_at: chrono::Utc::now(),
            segment_type: SegmentType::Code(path),
            importance,
        };
        
        self.add_segment(segment).await?;
        
        Ok(segment_id)
    }
    
    /// Add a segment to the chain
    async fn add_segment(&self, segment: ContextSegment) -> Result<()> {
        let mut segments = self.segments.write().await;
        let mut metadata = self.metadata.write().await;
        
        // Add sequential link to previous segment
        if let Some(last_segment) = segments.back() {
            let link = ContextLink {
                from: last_segment.id,
                to: segment.id,
                link_type: LinkType::Sequential,
                strength: 1.0,
            };
            
            let mut links = self.links.write().await;
            links.entry(last_segment.id)
                .or_insert_with(Vec::new)
                .push(link);
        }
        
        // Add the segment
        segments.push_back(segment.clone());
        
        // Update metadata
        metadata.total_segments += 1;
        metadata.average_importance = 
            (metadata.average_importance * (metadata.total_segments - 1) as f32 
             + segment.importance) / metadata.total_segments as f32;
        
        // Compress if needed
        if segments.len() > MAX_CHAIN_SEGMENTS {
            self.compress_chain(&mut segments, &mut metadata).await?;
        }
        
        Ok(())
    }
    
    /// Create a causal link between segments
    pub async fn link_segments(
        &self,
        from: SegmentId,
        to: SegmentId,
        link_type: LinkType,
        strength: f32,
    ) -> Result<()> {
        let link = ContextLink {
            from,
            to,
            link_type,
            strength,
        };
        
        let mut links = self.links.write().await;
        links.entry(from)
            .or_insert_with(Vec::new)
            .push(link);
        
        Ok(())
    }
    
    /// Get recent context as a string
    pub async fn get_recent_context(&self, max_tokens: usize) -> Result<String> {
        let segments = self.segments.read().await;
        let mut context = Vec::new();
        let mut token_count = 0;
        
        // Iterate from most recent
        for segment in segments.iter().rev() {
            if token_count + segment.tokens.len() > max_tokens {
                break;
            }
            
            context.push(segment.content.clone());
            token_count += segment.tokens.len();
        }
        
        context.reverse();
        Ok(context.join("\n\n"))
    }
    
    /// Get segments by type
    pub async fn get_segments_by_type(&self, segment_type: SegmentType) -> Vec<ContextSegment> {
        let segments = self.segments.read().await;
        segments
            .iter()
            .filter(|s| std::mem::discriminant(&s.segment_type) == std::mem::discriminant(&segment_type))
            .cloned()
            .collect()
    }
    
    /// Find related segments using links
    pub async fn find_related_segments(
        &self,
        segment_id: SegmentId,
        max_depth: usize,
    ) -> Result<Vec<(ContextSegment, f32)>> {
        let segments = self.segments.read().await;
        let links = self.links.read().await;
        
        let mut visited = std::collections::HashSet::new();
        let mut to_visit = VecDeque::new();
        let mut related = Vec::new();
        
        // Start with the given segment
        to_visit.push_back((segment_id, 1.0, 0));
        
        while let Some((current_id, strength, depth)) = to_visit.pop_front() {
            if depth > max_depth || visited.contains(&current_id) {
                continue;
            }
            
            visited.insert(current_id);
            
            // Find the segment
            if let Some(segment) = segments.iter().find(|s| s.id == current_id) {
                if current_id != segment_id {
                    related.push((segment.clone(), strength));
                }
                
                // Add linked segments
                if let Some(segment_links) = links.get(&current_id) {
                    for link in segment_links {
                        if !visited.contains(&link.to) {
                            to_visit.push_back((
                                link.to,
                                strength * link.strength,
                                depth + 1,
                            ));
                        }
                    }
                }
            }
        }
        
        // Sort by relevance (strength)
        related.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        Ok(related)
    }
    
    /// Compress the chain by summarizing old segments
    async fn compress_chain(
        &self,
        segments: &mut VecDeque<ContextSegment>,
        metadata: &mut ChainMetadata,
    ) -> Result<()> {
        let compression_target = MAX_CHAIN_SEGMENTS / 2;
        let to_compress = segments.len() - compression_target;
        
        // Extract segments to compress
        let mut compressed_content = Vec::new();
        for _ in 0..to_compress {
            if let Some(segment) = segments.pop_front() {
                compressed_content.push(segment.content);
            }
        }
        
        // Create summary segment
        let summary = format!(
            "Compressed {} segments: {}",
            to_compress,
            compressed_content.join(" → ")
        );
        
        let summary_segment = ContextSegment {
            id: SegmentId(Uuid::new_v4()),
            content: summary.clone(),
            tokens: self.tokenize(&summary),
            embedding: None,
            created_at: chrono::Utc::now(),
            segment_type: SegmentType::Summary,
            importance: 0.5, // Medium importance for summaries
        };
        
        segments.push_front(summary_segment);
        
        metadata.compressed_segments += to_compress;
        metadata.last_compression = Some(chrono::Utc::now());
        
        Ok(())
    }
    
    /// Simple tokenization (would use proper tokenizer in production)
    fn tokenize(&self, text: &str) -> Vec<String> {
        text.split_whitespace()
            .map(|s| s.to_string())
            .collect()
    }
}