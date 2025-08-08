//! arXiv Research Paper Integration
//!
//! Advanced arXiv client for searching, downloading, and indexing academic
//! papers with full-text processing and semantic understanding.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use anyhow::{Result, anyhow};
use reqwest::Client;
use scraper::{Html, Selector};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, info};

use crate::memory::{CognitiveMemory, MemoryId, MemoryMetadata};
use crate::safety::{ActionType, ActionValidator};

/// arXiv API configuration
#[derive(Debug, Clone)]
pub struct ArxivConfig {
    pub api_base: String,
    pub max_results_per_query: usize,
    pub request_timeout: Duration,
    pub download_pdfs: bool,
    pub cache_duration: Duration,
    pub user_agent: String,
}

impl Default for ArxivConfig {
    fn default() -> Self {
        Self {
            api_base: "http://export.arxiv.org/api/query".to_string(),
            max_results_per_query: 100,
            request_timeout: Duration::from_secs(30),
            download_pdfs: false, // Due to size considerations
            cache_duration: Duration::from_secs(3600 * 24), // 24 hours
            user_agent: "Loki arXiv Client/1.0".to_string(),
        }
    }
}

/// arXiv paper metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArxivPaper {
    pub id: String,
    pub title: String,
    pub authors: Vec<String>,
    pub summary: String,
    pub categories: Vec<String>,
    pub primary_category: String,
    pub published: String,
    pub updated: String,
    pub doi: Option<String>,
    pub pdf_url: String,
    pub abs_url: String,
    pub comment: Option<String>,
    pub journal_ref: Option<String>,
    pub subjects: Vec<String>,
    pub msc_classes: Vec<String>,
    pub acm_classes: Vec<String>,
    pub extracted_text: Option<String>,
    pub extracted_at: Option<SystemTime>,
    pub relevance_score: f32,
}

/// Search query for arXiv
#[derive(Debug, Clone)]
pub struct ArxivQuery {
    pub search_query: String,
    pub id_list: Vec<String>,
    pub start: usize,
    pub max_results: usize,
    pub sort_by: ArxivSortBy,
    pub sort_order: ArxivSortOrder,
}

impl Default for ArxivQuery {
    fn default() -> Self {
        Self {
            search_query: String::new(),
            id_list: Vec::new(),
            start: 0,
            max_results: 10,
            sort_by: ArxivSortBy::Relevance,
            sort_order: ArxivSortOrder::Descending,
        }
    }
}

/// Sort options for arXiv search
#[derive(Debug, Clone)]
pub enum ArxivSortBy {
    Relevance,
    LastUpdatedDate,
    SubmittedDate,
}

impl ArxivSortBy {
    fn to_string(&self) -> &'static str {
        match self {
            ArxivSortBy::Relevance => "relevance",
            ArxivSortBy::LastUpdatedDate => "lastUpdatedDate",
            ArxivSortBy::SubmittedDate => "submittedDate",
        }
    }
}

/// Sort order
#[derive(Debug, Clone)]
pub enum ArxivSortOrder {
    Ascending,
    Descending,
}

impl ArxivSortOrder {
    fn to_string(&self) -> &'static str {
        match self {
            ArxivSortOrder::Ascending => "ascending",
            ArxivSortOrder::Descending => "descending",
        }
    }
}

/// Search statistics
#[derive(Debug, Clone, Default)]
pub struct ArxivStats {
    pub total_searches: usize,
    pub total_papers_found: usize,
    pub total_papers_cached: usize,
    pub total_pdfs_downloaded: usize,
    pub average_relevance_score: f32,
    pub last_search_time: Option<SystemTime>,
    pub popular_categories: HashMap<String, usize>,
    pub popular_authors: HashMap<String, usize>,
}

/// Knowledge extraction from papers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaperKnowledge {
    pub paper_id: String,
    pub key_concepts: Vec<String>,
    pub methodologies: Vec<String>,
    pub technologies: Vec<String>,
    pub findings: Vec<String>,
    pub related_work: Vec<String>,
    pub future_work: Vec<String>,
    pub citations: Vec<String>,
    pub difficulty_level: String,
    pub practical_applications: Vec<String>,
}

/// arXiv category mapping
#[derive(Debug, Clone)]
pub struct ArxivCategories {
    categories: HashMap<String, String>,
}

impl ArxivCategories {
    pub fn new() -> Self {
        let mut categories = HashMap::new();

        // Computer Science categories
        categories.insert("cs.AI".to_string(), "Artificial Intelligence".to_string());
        categories.insert("cs.CL".to_string(), "Computation and Language".to_string());
        categories.insert("cs.CR".to_string(), "Cryptography and Security".to_string());
        categories
            .insert("cs.CV".to_string(), "Computer Vision and Pattern Recognition".to_string());
        categories.insert("cs.DB".to_string(), "Databases".to_string());
        categories.insert(
            "cs.DC".to_string(),
            "Distributed, Parallel, and Cluster Computing".to_string(),
        );
        categories.insert("cs.DS".to_string(), "Data Structures and Algorithms".to_string());
        categories.insert("cs.ET".to_string(), "Emerging Technologies".to_string());
        categories.insert("cs.FL".to_string(), "Formal Languages and Automata Theory".to_string());
        categories.insert("cs.GL".to_string(), "General Literature".to_string());
        categories.insert("cs.GR".to_string(), "Graphics".to_string());
        categories.insert("cs.GT".to_string(), "Computer Science and Game Theory".to_string());
        categories.insert("cs.HC".to_string(), "Human-Computer Interaction".to_string());
        categories.insert("cs.IR".to_string(), "Information Retrieval".to_string());
        categories.insert("cs.IT".to_string(), "Information Theory".to_string());
        categories.insert("cs.LG".to_string(), "Machine Learning".to_string());
        categories.insert("cs.LO".to_string(), "Logic in Computer Science".to_string());
        categories.insert("cs.MA".to_string(), "Multiagent Systems".to_string());
        categories.insert("cs.MM".to_string(), "Multimedia".to_string());
        categories.insert("cs.MS".to_string(), "Mathematical Software".to_string());
        categories.insert("cs.NA".to_string(), "Numerical Analysis".to_string());
        categories.insert("cs.NE".to_string(), "Neural and Evolutionary Computing".to_string());
        categories.insert("cs.NI".to_string(), "Networking and Internet Architecture".to_string());
        categories.insert("cs.OH".to_string(), "Other Computer Science".to_string());
        categories.insert("cs.OS".to_string(), "Operating Systems".to_string());
        categories.insert("cs.PF".to_string(), "Performance".to_string());
        categories.insert("cs.PL".to_string(), "Programming Languages".to_string());
        categories.insert("cs.RO".to_string(), "Robotics".to_string());
        categories.insert("cs.SC".to_string(), "Symbolic Computation".to_string());
        categories.insert("cs.SD".to_string(), "Sound".to_string());
        categories.insert("cs.SE".to_string(), "Software Engineering".to_string());
        categories.insert("cs.SI".to_string(), "Social and Information Networks".to_string());
        categories.insert("cs.SY".to_string(), "Systems and Control".to_string());

        // Mathematics categories
        categories.insert("math.AG".to_string(), "Algebraic Geometry".to_string());
        categories.insert("math.AT".to_string(), "Algebraic Topology".to_string());
        categories.insert("math.AP".to_string(), "Analysis of PDEs".to_string());
        categories.insert("math.CT".to_string(), "Category Theory".to_string());
        categories.insert("math.CA".to_string(), "Classical Analysis and ODEs".to_string());
        categories.insert("math.CO".to_string(), "Combinatorics".to_string());
        categories.insert("math.AC".to_string(), "Commutative Algebra".to_string());
        categories.insert("math.CV".to_string(), "Complex Variables".to_string());
        categories.insert("math.DG".to_string(), "Differential Geometry".to_string());
        categories.insert("math.DS".to_string(), "Dynamical Systems".to_string());
        categories.insert("math.FA".to_string(), "Functional Analysis".to_string());
        categories.insert("math.GM".to_string(), "General Mathematics".to_string());
        categories.insert("math.GN".to_string(), "General Topology".to_string());
        categories.insert("math.GT".to_string(), "Geometric Topology".to_string());
        categories.insert("math.GR".to_string(), "Group Theory".to_string());
        categories.insert("math.HO".to_string(), "History and Overview".to_string());
        categories.insert("math.IT".to_string(), "Information Theory".to_string());
        categories.insert("math.KT".to_string(), "K-Theory and Homology".to_string());
        categories.insert("math.LO".to_string(), "Logic".to_string());
        categories.insert("math.MP".to_string(), "Mathematical Physics".to_string());
        categories.insert("math.MG".to_string(), "Metric Geometry".to_string());
        categories.insert("math.NT".to_string(), "Number Theory".to_string());
        categories.insert("math.NA".to_string(), "Numerical Analysis".to_string());
        categories.insert("math.OA".to_string(), "Operator Algebras".to_string());
        categories.insert("math.OC".to_string(), "Optimization and Control".to_string());
        categories.insert("math.PR".to_string(), "Probability".to_string());
        categories.insert("math.QA".to_string(), "Quantum Algebra".to_string());
        categories.insert("math.RT".to_string(), "Representation Theory".to_string());
        categories.insert("math.RA".to_string(), "Rings and Algebras".to_string());
        categories.insert("math.SP".to_string(), "Spectral Theory".to_string());
        categories.insert("math.ST".to_string(), "Statistics Theory".to_string());
        categories.insert("math.SG".to_string(), "Symplectic Geometry".to_string());

        // Physics categories
        categories.insert("physics.acc-ph".to_string(), "Accelerator Physics".to_string());
        categories
            .insert("physics.ao-ph".to_string(), "Atmospheric and Oceanic Physics".to_string());
        categories.insert("physics.atom-ph".to_string(), "Atomic Physics".to_string());
        categories
            .insert("physics.atm-clus".to_string(), "Atomic and Molecular Clusters".to_string());
        categories.insert("physics.bio-ph".to_string(), "Biological Physics".to_string());
        categories.insert("physics.chem-ph".to_string(), "Chemical Physics".to_string());
        categories.insert("physics.class-ph".to_string(), "Classical Physics".to_string());
        categories.insert("physics.comp-ph".to_string(), "Computational Physics".to_string());
        categories.insert(
            "physics.data-an".to_string(),
            "Data Analysis, Statistics and Probability".to_string(),
        );
        categories.insert("physics.flu-dyn".to_string(), "Fluid Dynamics".to_string());
        categories.insert("physics.gen-ph".to_string(), "General Physics".to_string());
        categories.insert("physics.geo-ph".to_string(), "Geophysics".to_string());
        categories
            .insert("physics.hist-ph".to_string(), "History and Philosophy of Physics".to_string());
        categories
            .insert("physics.ins-det".to_string(), "Instrumentation and Detectors".to_string());
        categories.insert("physics.med-ph".to_string(), "Medical Physics".to_string());
        categories.insert("physics.optics".to_string(), "Optics".to_string());
        categories.insert("physics.ed-ph".to_string(), "Physics Education".to_string());
        categories.insert("physics.soc-ph".to_string(), "Physics and Society".to_string());
        categories.insert("physics.plasm-ph".to_string(), "Plasma Physics".to_string());
        categories.insert("physics.pop-ph".to_string(), "Popular Physics".to_string());
        categories.insert("physics.space-ph".to_string(), "Space Physics".to_string());

        Self { categories }
    }

    pub fn get_description(&self, category: &str) -> Option<&String> {
        self.categories.get(category)
    }

    pub fn all_categories(&self) -> &HashMap<String, String> {
        &self.categories
    }
}

/// Advanced arXiv client
pub struct ArxivClient {
    /// HTTP client
    client: Client,

    /// Configuration
    config: ArxivConfig,

    /// Memory system
    memory: Arc<CognitiveMemory>,

    /// Safety validator
    validator: Option<Arc<ActionValidator>>,

    /// Paper cache
    paper_cache: Arc<RwLock<HashMap<String, ArxivPaper>>>,

    /// Knowledge cache
    knowledge_cache: Arc<RwLock<HashMap<String, PaperKnowledge>>>,

    /// Statistics
    stats: Arc<RwLock<ArxivStats>>,

    /// Category mapping
    categories: ArxivCategories,
}

impl ArxivClient {
    /// Create a new arXiv client
    pub async fn new(
        config: ArxivConfig,
        memory: Arc<CognitiveMemory>,
        validator: Option<Arc<ActionValidator>>,
    ) -> Result<Self> {
        info!("Initializing arXiv client");

        let client = Client::builder()
            .timeout(config.request_timeout)
            .user_agent(&config.user_agent)
            .build()?;

        Ok(Self {
            client,
            config,
            memory,
            validator,
            paper_cache: Arc::new(RwLock::new(HashMap::new())),
            knowledge_cache: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(ArxivStats::default())),
            categories: ArxivCategories::new(),
        })
    }

    /// Search for papers on arXiv
    pub async fn search(&self, query: ArxivQuery) -> Result<Vec<ArxivPaper>> {
        // Validate search through safety system
        if let Some(validator) = &self.validator {
            validator
                .validate_action(
                    ActionType::ApiCall {
                        provider: "arxiv".to_string(),
                        endpoint: "search".to_string(),
                    },
                    format!("arXiv search: {}", query.search_query),
                    vec!["Searching academic papers for research".to_string()],
                )
                .await?;
        }

        info!("Searching arXiv: {}", query.search_query);

        // Build query URL
        let mut url = format!("{}?", self.config.api_base);

        if !query.search_query.is_empty() {
            url.push_str(&format!("search_query={}&", urlencoding::encode(&query.search_query)));
        }

        if !query.id_list.is_empty() {
            url.push_str(&format!("id_list={}&", query.id_list.join(",")));
        }

        url.push_str(&format!(
            "start={}&max_results={}&sortBy={}&sortOrder={}",
            query.start,
            query.max_results.min(self.config.max_results_per_query),
            query.sort_by.to_string(),
            query.sort_order.to_string()
        ));

        debug!("arXiv API URL: {}", url);

        // Make the request
        let response = self.client.get(&url).send().await?;

        if !response.status().is_success() {
            return Err(anyhow!("arXiv API error: {}", response.status()));
        }

        let xml_content = response.text().await?;

        // Parse the XML response
        let papers = self.parse_arxiv_response(&xml_content)?;

        // Update statistics
        {
            let mut stats = self.stats.write().await;
            stats.total_searches += 1;
            stats.total_papers_found += papers.len();
            stats.last_search_time = Some(SystemTime::now());

            // Update popular categories and authors
            for paper in &papers {
                for category in &paper.categories {
                    *stats.popular_categories.entry(category.clone()).or_insert(0) += 1;
                }
                for author in &paper.authors {
                    *stats.popular_authors.entry(author.clone()).or_insert(0) += 1;
                }
            }

            // Calculate average relevance score
            if !papers.is_empty() {
                let total_score: f32 = papers.iter().map(|p| p.relevance_score).sum();
                stats.average_relevance_score = total_score / papers.len() as f32;
            }
        }

        // Cache papers and store in memory
        for paper in &papers {
            self.cache_paper(paper.clone()).await?;
            self.store_paper_in_memory(paper).await?;
        }

        info!("Found {} papers for query: {}", papers.len(), query.search_query);

        Ok(papers)
    }

    /// Parse arXiv XML response
    fn parse_arxiv_response(&self, xml_content: &str) -> Result<Vec<ArxivPaper>> {
        // Parse XML using a simple approach (in production, use a proper XML parser)
        let document = Html::parse_document(xml_content);

        let entry_selector = Selector::parse("entry").unwrap();
        let id_selector = Selector::parse("id").unwrap();
        let title_selector = Selector::parse("title").unwrap();
        let summary_selector = Selector::parse("summary").unwrap();
        let author_selector = Selector::parse("author name").unwrap();
        let category_selector = Selector::parse("category").unwrap();
        let published_selector = Selector::parse("published").unwrap();
        let updated_selector = Selector::parse("updated").unwrap();
        let link_selector = Selector::parse("link").unwrap();
        let comment_selector = Selector::parse("arxiv\\:comment").unwrap();
        let journal_selector = Selector::parse("arxiv\\:journal_ref").unwrap();
        let doi_selector = Selector::parse("arxiv\\:doi").unwrap();

        let mut papers = Vec::new();

        for entry in document.select(&entry_selector) {
            // Extract ID
            let id = entry
                .select(&id_selector)
                .next()
                .map(|e| e.text().collect::<String>())
                .unwrap_or_default()
                .trim()
                .to_string();

            if id.is_empty() {
                continue;
            }

            // Extract paper ID from full URL
            let paper_id = id.split('/').last().unwrap_or(&id).to_string();

            // Extract title
            let title = entry
                .select(&title_selector)
                .next()
                .map(|e| e.text().collect::<String>())
                .unwrap_or_default()
                .trim()
                .to_string();

            // Extract summary
            let summary = entry
                .select(&summary_selector)
                .next()
                .map(|e| e.text().collect::<String>())
                .unwrap_or_default()
                .trim()
                .to_string();

            // Extract authors
            let authors: Vec<String> = entry
                .select(&author_selector)
                .map(|e| e.text().collect::<String>().trim().to_string())
                .collect();

            // Extract categories
            let categories: Vec<String> = entry
                .select(&category_selector)
                .filter_map(|e| e.value().attr("term"))
                .map(|s| s.to_string())
                .collect();

            let primary_category = categories.first().cloned().unwrap_or_default();

            // Extract dates
            let published = entry
                .select(&published_selector)
                .next()
                .map(|e| e.text().collect::<String>())
                .unwrap_or_default()
                .trim()
                .to_string();

            let updated = entry
                .select(&updated_selector)
                .next()
                .map(|e| e.text().collect::<String>())
                .unwrap_or_default()
                .trim()
                .to_string();

            // Extract links
            let mut pdf_url = String::new();
            let mut abs_url = String::new();

            for link in entry.select(&link_selector) {
                if let Some(href) = link.value().attr("href") {
                    if let Some(rel) = link.value().attr("rel") {
                        match rel {
                            "related" => abs_url = href.to_string(),
                            _ => {}
                        }
                    }
                    if let Some(title) = link.value().attr("title") {
                        if title == "pdf" {
                            pdf_url = href.to_string();
                        }
                    }
                }
            }

            // If PDF URL not found, construct it
            if pdf_url.is_empty() && !paper_id.is_empty() {
                pdf_url = format!("http://arxiv.org/pdf/{}.pdf", paper_id);
            }

            // If abstract URL not found, construct it
            if abs_url.is_empty() && !paper_id.is_empty() {
                abs_url = format!("http://arxiv.org/abs/{}", paper_id);
            }

            // Extract optional fields
            let comment = entry
                .select(&comment_selector)
                .next()
                .map(|e| e.text().collect::<String>().trim().to_string());

            let journal_ref = entry
                .select(&journal_selector)
                .next()
                .map(|e| e.text().collect::<String>().trim().to_string());

            let doi = entry
                .select(&doi_selector)
                .next()
                .map(|e| e.text().collect::<String>().trim().to_string());

            // Generate subjects from categories
            let subjects: Vec<String> = categories
                .iter()
                .filter_map(|cat| self.categories.get_description(cat))
                .cloned()
                .collect();

            // Calculate relevance score (simplified)
            let relevance_score = self.calculate_relevance_score(&title, &summary, &categories);

            let paper = ArxivPaper {
                id: paper_id,
                title,
                authors,
                summary,
                categories,
                primary_category,
                published,
                updated,
                doi,
                pdf_url,
                abs_url,
                comment,
                journal_ref,
                subjects,
                msc_classes: Vec::new(), // Would extract from metadata
                acm_classes: Vec::new(), // Would extract from metadata
                extracted_text: None,
                extracted_at: None,
                relevance_score,
            };

            papers.push(paper);
        }

        Ok(papers)
    }

    /// Calculate relevance score for a paper
    fn calculate_relevance_score(&self, title: &str, summary: &str, categories: &[String]) -> f32 {
        let mut score = 0.0;

        // Base score from title and summary length (longer = more comprehensive)
        score += (title.len() + summary.len()) as f32 * 0.001;

        // Boost for AI/ML related categories
        let ai_categories = ["cs.AI", "cs.LG", "cs.NE", "cs.CL", "cs.CV"];
        for category in categories {
            if ai_categories.contains(&category.as_str()) {
                score += 0.5;
            }
        }

        // Boost for recent papers (simplified - would use actual dates)
        score += 0.3;

        // Normalize to 0-1 range
        score.min(1.0).max(0.0)
    }

    /// Cache a paper
    async fn cache_paper(&self, paper: ArxivPaper) -> Result<()> {
        let mut cache = self.paper_cache.write().await;
        cache.insert(paper.id.clone(), paper);

        let mut stats = self.stats.write().await;
        stats.total_papers_cached += 1;

        Ok(())
    }

    /// Store paper in cognitive memory
    async fn store_paper_in_memory(&self, paper: &ArxivPaper) -> Result<()> {
        let importance = match paper.relevance_score {
            score if score > 0.8 => 0.9,
            score if score > 0.6 => 0.7,
            score if score > 0.4 => 0.5,
            _ => 0.3,
        };

        let mut tags = vec!["research".to_string(), "arxiv".to_string(), "academic".to_string()];

        // Add subject tags
        tags.extend(paper.subjects.clone());

        // Add category tags
        tags.extend(paper.categories.clone());

        // Add author tags (first few authors)
        for author in paper.authors.iter().take(3) {
            tags.push(format!("author:{}", author));
        }

        // Create content with title, authors, and summary
        let content = format!(
            "Title: {}\n\nAuthors: {}\n\nCategories: {}\n\nAbstract:\n{}",
            paper.title,
            paper.authors.join(", "),
            paper.subjects.join(", "),
            paper.summary
        );

        self.memory
            .store(
                content.clone(),
                vec![], // No code examples in papers typically
                MemoryMetadata {
                    source: "arxiv".to_string(),
                    tags: tags.clone(),
                    importance,
                    associations: self
                        .find_related_memories(&content, &tags)
                        .await
                        .unwrap_or_default(),
                    context: Some("ArXiv research paper".to_string()),
                    created_at: chrono::Utc::now(),
                    accessed_count: 0,
                    last_accessed: None,
                    version: 1,
                    category: "tool_usage".to_string(),
                    timestamp: chrono::Utc::now(),
                    expiration: None,
                },
            )
            .await?;

        debug!("Stored paper in memory: {}", paper.title);

        Ok(())
    }

    /// Extract knowledge from a paper's text
    pub async fn extract_knowledge(&self, paper_id: &str) -> Result<PaperKnowledge> {
        // Check cache first
        {
            let cache = self.knowledge_cache.read().await;
            if let Some(knowledge) = cache.get(paper_id) {
                return Ok(knowledge.clone());
            }
        }

        // Get paper from cache
        let paper = {
            let cache = self.paper_cache.read().await;
            cache.get(paper_id).cloned()
        };

        let paper = paper.ok_or_else(|| anyhow!("Paper not found in cache: {}", paper_id))?;

        info!("Extracting knowledge from paper: {}", paper.title);

        // For now, extract knowledge from title and abstract
        // In a full implementation, you'd download and parse the PDF
        let knowledge = self.analyze_paper_content(&paper).await?;

        // Cache the knowledge
        {
            let mut cache = self.knowledge_cache.write().await;
            cache.insert(paper_id.to_string(), knowledge.clone());
        }

        // Store knowledge in memory
        self.store_knowledge_in_memory(&knowledge).await?;

        Ok(knowledge)
    }

    /// Analyze paper content to extract knowledge
    async fn analyze_paper_content(&self, paper: &ArxivPaper) -> Result<PaperKnowledge> {
        let text = format!("{} {}", paper.title, paper.summary);
        let text_lower = text.to_lowercase();

        // Extract key concepts using keyword matching
        let mut key_concepts = HashSet::new();
        let concept_keywords = [
            "neural network",
            "machine learning",
            "deep learning",
            "artificial intelligence",
            "natural language processing",
            "computer vision",
            "reinforcement learning",
            "transformer",
            "attention",
            "convolution",
            "recurrent",
            "lstm",
            "gru",
            "optimization",
            "gradient descent",
            "backpropagation",
            "training",
            "classification",
            "regression",
            "clustering",
            "dimensionality reduction",
            "supervised",
            "unsupervised",
            "semi-supervised",
            "self-supervised",
            "generative",
            "discriminative",
            "autoencoder",
            "gan",
            "variational",
            "bayesian",
            "probabilistic",
            "stochastic",
            "deterministic",
        ];

        for keyword in &concept_keywords {
            if text_lower.contains(keyword) {
                key_concepts.insert(keyword.to_string());
            }
        }

        // Extract methodologies
        let mut methodologies = HashSet::new();
        let method_keywords = [
            "algorithm",
            "method",
            "approach",
            "technique",
            "framework",
            "model",
            "architecture",
            "system",
            "protocol",
            "procedure",
            "experiment",
            "evaluation",
            "benchmark",
            "comparison",
            "analysis",
        ];

        for keyword in &method_keywords {
            if text_lower.contains(keyword) {
                methodologies.insert(keyword.to_string());
            }
        }

        // Extract technologies
        let mut technologies = HashSet::new();
        let tech_keywords = [
            "python",
            "pytorch",
            "tensorflow",
            "keras",
            "scikit-learn",
            "cuda",
            "gpu",
            "cpu",
            "distributed",
            "parallel",
            "docker",
            "kubernetes",
            "cloud",
            "aws",
            "azure",
            "gcp",
            "api",
            "rest",
            "graphql",
            "database",
            "sql",
            "nosql",
        ];

        for keyword in &tech_keywords {
            if text_lower.contains(keyword) {
                technologies.insert(keyword.to_string());
            }
        }

        // Assess difficulty level
        let difficulty_level = if text_lower.contains("survey") || text_lower.contains("review") {
            "intermediate".to_string()
        } else if text_lower.contains("novel")
            || text_lower.contains("advanced")
            || text_lower.contains("theoretical")
        {
            "advanced".to_string()
        } else if text_lower.contains("introduction") || text_lower.contains("tutorial") {
            "beginner".to_string()
        } else {
            "intermediate".to_string()
        };

        // Extract practical applications
        let mut applications = HashSet::new();
        let app_keywords = [
            "application",
            "real-world",
            "practical",
            "deployment",
            "production",
            "industry",
            "business",
            "commercial",
            "use case",
            "implementation",
            "healthcare",
            "finance",
            "education",
            "transportation",
            "security",
            "recommendation",
            "search",
            "translation",
            "generation",
            "prediction",
        ];

        for keyword in &app_keywords {
            if text_lower.contains(keyword) {
                applications.insert(keyword.to_string());
            }
        }

        Ok(PaperKnowledge {
            paper_id: paper.id.clone(),
            key_concepts: key_concepts.into_iter().collect(),
            methodologies: methodologies.into_iter().collect(),
            technologies: technologies.into_iter().collect(),
            findings: vec![],     // Would extract from results section
            related_work: vec![], // Would extract from related work section
            future_work: vec![],  // Would extract from conclusion/future work
            citations: vec![],    // Would extract from references
            difficulty_level,
            practical_applications: applications.into_iter().collect(),
        })
    }

    /// Store extracted knowledge in memory
    async fn store_knowledge_in_memory(&self, knowledge: &PaperKnowledge) -> Result<()> {
        let content = format!(
            "Paper Knowledge Extraction:\n\nKey Concepts: {}\n\nMethodologies: \
             {}\n\nTechnologies: {}\n\nDifficulty: {}\n\nApplications: {}",
            knowledge.key_concepts.join(", "),
            knowledge.methodologies.join(", "),
            knowledge.technologies.join(", "),
            knowledge.difficulty_level,
            knowledge.practical_applications.join(", ")
        );

        let mut tags = vec![
            "knowledge".to_string(),
            "extracted".to_string(),
            "research".to_string(),
            knowledge.difficulty_level.clone(),
        ];

        tags.extend(knowledge.key_concepts.clone());
        tags.extend(knowledge.technologies.clone());

        self.memory
            .store(
                content,
                vec![], // No code in knowledge extraction
                MemoryMetadata {
                    source: "arxiv_knowledge".to_string(),
                    tags,
                    importance: 0.8,
                    associations: vec![],

                    context: Some("Generated from automated fix".to_string()),
                    created_at: chrono::Utc::now(),
                    accessed_count: 0,
                    last_accessed: None,
                    version: 1,
                    category: "tool_usage".to_string(),
                    timestamp: chrono::Utc::now(),
                    expiration: None,
                },
            )
            .await?;

        Ok(())
    }

    /// Search papers by category
    pub async fn search_by_category(
        &self,
        category: &str,
        max_results: usize,
    ) -> Result<Vec<ArxivPaper>> {
        let query = ArxivQuery {
            search_query: format!("cat:{}", category),
            max_results,
            ..Default::default()
        };

        self.search(query).await
    }

    /// Search papers by author
    pub async fn search_by_author(
        &self,
        author: &str,
        max_results: usize,
    ) -> Result<Vec<ArxivPaper>> {
        let query = ArxivQuery {
            search_query: format!("au:{}", author),
            max_results,
            ..Default::default()
        };

        self.search(query).await
    }

    /// Get trending papers (most recent in popular categories)
    pub async fn get_trending_papers(&self, max_results: usize) -> Result<Vec<ArxivPaper>> {
        let trending_categories = ["cs.AI", "cs.LG", "cs.CL", "cs.CV", "cs.RO"];
        let mut all_papers = Vec::new();

        for category in &trending_categories {
            let papers =
                self.search_by_category(category, max_results / trending_categories.len()).await?;
            all_papers.extend(papers);
        }

        // Sort by relevance score
        all_papers.sort_by(|a, b| {
            b.relevance_score.partial_cmp(&a.relevance_score).unwrap_or(std::cmp::Ordering::Equal)
        });
        all_papers.truncate(max_results);

        Ok(all_papers)
    }

    /// Get statistics
    pub async fn stats(&self) -> ArxivStats {
        self.stats.read().await.clone()
    }

    /// Get all available categories
    pub fn get_categories(&self) -> &HashMap<String, String> {
        self.categories.all_categories()
    }

    /// Search cached papers
    pub async fn search_cache(&self, query: &str, limit: usize) -> Vec<ArxivPaper> {
        let cache = self.paper_cache.read().await;
        let query_lower = query.to_lowercase();

        let mut results: Vec<_> = cache
            .values()
            .filter(|paper| {
                paper.title.to_lowercase().contains(&query_lower)
                    || paper.summary.to_lowercase().contains(&query_lower)
                    || paper
                        .authors
                        .iter()
                        .any(|author| author.to_lowercase().contains(&query_lower))
                    || paper
                        .subjects
                        .iter()
                        .any(|subject| subject.to_lowercase().contains(&query_lower))
            })
            .cloned()
            .collect();

        // Sort by relevance score
        results.sort_by(|a, b| {
            b.relevance_score.partial_cmp(&a.relevance_score).unwrap_or(std::cmp::Ordering::Equal)
        });
        results.into_iter().take(limit).collect()
    }

    /// Find related memories based on content and tags for ArXiv papers
    async fn find_related_memories(&self, content: &str, tags: &[String]) -> Result<Vec<MemoryId>> {
        let mut related_memories = Vec::new();

        // 1. Find memories with similar research topics
        let topic_related = self.find_memories_by_research_topics(tags).await?;
        related_memories.extend(topic_related);

        // 2. Find memories by authors
        let author_related = self.find_memories_by_authors(tags).await?;
        related_memories.extend(author_related);

        // 3. Find semantically similar research content
        let semantic_related = self.find_semantically_similar_research(content).await?;
        related_memories.extend(semantic_related);

        // 4. Find memories with similar academic categories
        let category_related = self.find_memories_by_categories(tags).await?;
        related_memories.extend(category_related);

        // Remove duplicates and limit results
        related_memories.sort();
        related_memories.dedup();
        related_memories.truncate(8); // Limit to top 8 associations for research papers

        Ok(related_memories)
    }

    /// Find memories related to research topics
    async fn find_memories_by_research_topics(&self, tags: &[String]) -> Result<Vec<MemoryId>> {
        let mut related = Vec::new();

        // Look for research topic keywords
        let research_topics: Vec<_> = tags
            .iter()
            .filter(|tag| {
                let tag_lower = tag.to_lowercase();
                tag_lower.contains("learning")
                    || tag_lower.contains("neural")
                    || tag_lower.contains("network")
                    || tag_lower.contains("algorithm")
                    || tag_lower.contains("intelligence")
                    || tag_lower.contains("vision")
                    || tag_lower.contains("language")
                    || tag_lower.contains("optimization")
                    || tag_lower.contains("deep")
                    || tag_lower.contains("machine")
            })
            .collect();

        for topic in research_topics {
            if topic.len() > 4 {
                match self.memory.retrieve_similar(topic, 3).await {
                    Ok(memories) => {
                        for memory in memories {
                            // Check if it's research-related content
                            if memory.metadata.tags.iter().any(|t| {
                                t.contains("research")
                                    || t.contains("academic")
                                    || t.contains("paper")
                                    || t.contains("arxiv")
                            }) {
                                related.push(memory.id);
                            }
                        }
                    }
                    Err(_) => continue,
                }
            }
        }

        Ok(related)
    }

    /// Find memories by research authors
    async fn find_memories_by_authors(&self, tags: &[String]) -> Result<Vec<MemoryId>> {
        let mut related = Vec::new();

        // Look for author tags
        let author_tags: Vec<_> = tags.iter().filter(|tag| tag.starts_with("author:")).collect();

        for author_tag in author_tags {
            let author_name = author_tag.strip_prefix("author:").unwrap_or(author_tag);
            if author_name.len() > 3 {
                match self.memory.retrieve_similar(author_name, 2).await {
                    Ok(memories) => {
                        for memory in memories {
                            // Only include if it's from academic sources
                            if memory.metadata.source == "arxiv"
                                || memory.metadata.tags.contains(&"research".to_string())
                            {
                                related.push(memory.id);
                            }
                        }
                    }
                    Err(_) => continue,
                }
            }
        }

        Ok(related)
    }

    /// Find semantically similar research content
    async fn find_semantically_similar_research(&self, content: &str) -> Result<Vec<MemoryId>> {
        // Extract key research terms
        let research_terms = self.extract_research_terms(content);
        let mut related = Vec::new();

        for term in research_terms.iter().take(3) {
            // Top 3 research terms
            if term.len() > 6 {
                match self.memory.retrieve_similar(term, 2).await {
                    Ok(memories) => {
                        for memory in memories {
                            // Calculate content similarity for research papers
                            let similarity =
                                self.calculate_research_similarity(content, &memory.content);
                            if similarity > 0.25 {
                                // Lower threshold for academic content
                                related.push(memory.id);
                            }
                        }
                    }
                    Err(_) => continue,
                }
            }
        }

        Ok(related)
    }

    /// Find memories with similar academic categories
    async fn find_memories_by_categories(&self, tags: &[String]) -> Result<Vec<MemoryId>> {
        let mut related = Vec::new();

        // Look for ArXiv category codes (cs.AI, cs.LG, etc.)
        let category_tags: Vec<_> = tags
            .iter()
            .filter(|tag| tag.contains('.') && tag.len() < 10) // ArXiv category format
            .collect();

        for category in category_tags {
            match self.memory.retrieve_similar(category, 2).await {
                Ok(memories) => {
                    for memory in memories {
                        // Only include academic content
                        if memory.metadata.source == "arxiv"
                            || memory.metadata.tags.contains(&"academic".to_string())
                        {
                            related.push(memory.id);
                        }
                    }
                }
                Err(_) => continue,
            }
        }

        Ok(related)
    }

    /// Extract research-specific terms from content
    fn extract_research_terms(&self, content: &str) -> Vec<String> {
        let mut terms = Vec::new();
        let content_lower = content.to_lowercase();

        // Common research terms and phrases
        let research_patterns = [
            "neural network",
            "machine learning",
            "deep learning",
            "artificial intelligence",
            "computer vision",
            "natural language processing",
            "reinforcement learning",
            "convolutional",
            "transformer",
            "attention mechanism",
            "gradient descent",
            "optimization",
            "classification",
            "regression",
            "clustering",
            "supervised learning",
            "unsupervised learning",
            "feature extraction",
            "dimensionality reduction",
            "cross-validation",
            "hyperparameter",
            "training set",
            "test set",
            "validation",
            "accuracy",
            "precision",
            "recall",
            "f1-score",
            "loss function",
            "backpropagation",
            "overfitting",
            "regularization",
            "dropout",
            "batch normalization",
        ];

        for pattern in &research_patterns {
            if content_lower.contains(pattern) {
                terms.push(pattern.to_string());
            }
        }

        // Also extract technical terms (capitalized words that might be
        // algorithms/methods)
        for word in content.split_whitespace() {
            if word.len() > 5 && word.chars().next().unwrap().is_uppercase() {
                let cleaned = word.trim_matches(|c: char| !c.is_alphanumeric());
                if cleaned.len() > 3 && !cleaned.chars().any(|c| c.is_lowercase()) {
                    terms.push(cleaned.to_string());
                }
            }
        }

        terms.truncate(8); // Limit to top 8 terms
        terms
    }

    /// Calculate similarity between research contents
    fn calculate_research_similarity(&self, content1: &str, content2: &str) -> f32 {
        // Extract technical terms from both contents
        let terms1 = self.extract_research_terms(content1);
        let terms2 = self.extract_research_terms(content2);

        if terms1.is_empty() || terms2.is_empty() {
            return 0.0;
        }

        let set1: std::collections::HashSet<String> = terms1.into_iter().collect();
        let set2: std::collections::HashSet<String> = terms2.into_iter().collect();

        let intersection = set1.intersection(&set2).count();
        let union = set1.union(&set2).count();

        if union == 0 { 0.0 } else { intersection as f32 / union as f32 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arxiv_categories() {
        let categories = ArxivCategories::new();
        assert_eq!(
            categories.get_description("cs.AI"),
            Some(&"Artificial Intelligence".to_string())
        );
        assert_eq!(categories.get_description("cs.LG"), Some(&"Machine Learning".to_string()));
        assert!(categories.get_description("invalid.category").is_none());
    }

    #[test]
    fn test_arxiv_query_default() {
        let query = ArxivQuery::default();
        assert_eq!(query.max_results, 10);
        assert_eq!(query.start, 0);
        assert!(query.search_query.is_empty());
    }

    #[test]
    fn test_sort_by_string() {
        assert_eq!(ArxivSortBy::Relevance.to_string(), "relevance");
        assert_eq!(ArxivSortBy::LastUpdatedDate.to_string(), "lastUpdatedDate");
        assert_eq!(ArxivSortOrder::Descending.to_string(), "descending");
    }
}
