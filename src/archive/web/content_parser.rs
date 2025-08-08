use std::time::Duration;

use anyhow::Result;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::info;

/// Extracted web content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebContent {
    pub url: String,
    pub title: Option<String>,
    pub text: String,
    pub summary: Option<String>,
    pub author: Option<String>,
    pub published_date: Option<chrono::DateTime<chrono::Utc>>,
    pub images: Vec<String>,
    pub links: Vec<String>,
    pub tags: Vec<String>,
    pub content_type: ContentType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContentType {
    Article,
    BlogPost,
    Documentation,
    ForumPost,
    SocialMedia,
    Video,
    Unknown,
}

/// Content extractor for web pages
pub struct ContentExtractor {
    client: Client,
}

impl ContentExtractor {
    /// Create a new content extractor
    pub fn new() -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .user_agent("Loki/1.0 ContentExtractor")
            .build()
            .unwrap();

        Self { client }
    }

    /// Extract content from a URL
    pub async fn extract_from_url(&self, url: &str) -> Result<WebContent> {
        info!("Extracting content from: {}", url);

        let response = self.client.get(url).send().await?;
        let html = response.text().await?;

        self.extract_from_html(url, &html)
    }

    /// Extract content from HTML
    pub fn extract_from_html(&self, url: &str, html: &str) -> Result<WebContent> {
        use scraper::Html;

        let document = Html::parse_document(html);

        // Extract title
        let title = self.extract_title(&document);

        // Extract main content
        let text = self.extract_main_content(&document);

        // Extract metadata
        let author = self.extract_meta(&document, "author");
        let published_date = self.extract_published_date(&document);

        // Extract images
        let images = self.extract_images(&document, url);

        // Extract links
        let links = self.extract_links(&document, url);

        // Extract tags/keywords
        let tags = self.extract_tags(&document);

        // Determine content type
        let content_type = self.determine_content_type(url, &document);

        // Generate summary (first 200 chars of main content)
        let summary = Some(text.chars().take(200).collect::<String>().trim().to_string() + "...");

        Ok(WebContent {
            url: url.to_string(),
            title,
            text,
            summary,
            author,
            published_date,
            images,
            links,
            tags,
            content_type,
        })
    }

    /// Extract title from HTML
    fn extract_title(&self, document: &scraper::Html) -> Option<String> {
        use scraper::Selector;

        // Try different title selectors
        let selectors =
            vec!["h1", "title", "meta[property='og:title']", "meta[name='twitter:title']"];

        for selector_str in selectors {
            if let Ok(selector) = Selector::parse(selector_str) {
                if let Some(element) = document.select(&selector).next() {
                    let title = if selector_str.starts_with("meta") {
                        element.value().attr("content").map(|s| s.to_string())
                    } else {
                        Some(element.text().collect::<String>().trim().to_string())
                    };

                    if let Some(t) = title {
                        if !t.is_empty() {
                            return Some(t);
                        }
                    }
                }
            }
        }

        None
    }

    /// Extract main content from HTML
    fn extract_main_content(&self, document: &scraper::Html) -> String {
        use scraper::Selector;

        // Try common content selectors
        let content_selectors = vec![
            "main",
            "article",
            "[role='main']",
            ".content",
            "#content",
            ".post-content",
            ".entry-content",
        ];

        for selector_str in content_selectors {
            if let Ok(selector) = Selector::parse(selector_str) {
                if let Some(element) = document.select(&selector).next() {
                    let text = self.extract_text_from_element(element);
                    if text.len() > 100 {
                        // Minimum content length
                        return text;
                    }
                }
            }
        }

        // Fallback: extract all paragraphs
        if let Ok(p_selector) = Selector::parse("p") {
            let paragraphs: Vec<String> = document
                .select(&p_selector)
                .map(|el| self.extract_text_from_element(el))
                .filter(|text| text.len() > 20)
                .collect();

            return paragraphs.join("\n\n");
        }

        // Last resort: all text
        document.root_element().text().collect::<String>().trim().to_string()
    }

    /// Extract text from an element, cleaning it up
    fn extract_text_from_element(&self, element: scraper::ElementRef) -> String {
        element
            .text()
            .collect::<String>()
            .lines()
            .map(|line| line.trim())
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Extract meta tag content
    fn extract_meta(&self, document: &scraper::Html, name: &str) -> Option<String> {
        use scraper::Selector;

        let selectors = vec![
            format!("meta[name='{}']", name),
            format!("meta[property='{}']", name),
            format!("meta[property='og:{}']", name),
        ];

        for selector_str in selectors {
            if let Ok(selector) = Selector::parse(&selector_str) {
                if let Some(element) = document.select(&selector).next() {
                    if let Some(content) = element.value().attr("content") {
                        return Some(content.to_string());
                    }
                }
            }
        }

        None
    }

    /// Extract published date
    fn extract_published_date(
        &self,
        document: &scraper::Html,
    ) -> Option<chrono::DateTime<chrono::Utc>> {
        use scraper::Selector;

        let date_selectors = vec![
            "meta[property='article:published_time']",
            "meta[name='publish_date']",
            "time[datetime]",
            "time",
        ];

        for selector_str in date_selectors {
            if let Ok(selector) = Selector::parse(selector_str) {
                if let Some(element) = document.select(&selector).next() {
                    let date_str = if selector_str.contains("meta") {
                        element.value().attr("content").map(|s| s.to_string())
                    } else if let Some(datetime) = element.value().attr("datetime") {
                        Some(datetime.to_string())
                    } else {
                        Some(element.text().collect::<String>())
                    };

                    if let Some(date_str) = date_str {
                        // Try to parse various date formats
                        if let Ok(date) = chrono::DateTime::parse_from_rfc3339(&date_str) {
                            return Some(date.with_timezone(&chrono::Utc));
                        }
                        if let Ok(date) = chrono::DateTime::parse_from_rfc2822(&date_str) {
                            return Some(date.with_timezone(&chrono::Utc));
                        }
                    }
                }
            }
        }

        None
    }

    /// Extract images
    fn extract_images(&self, document: &scraper::Html, base_url: &str) -> Vec<String> {
        use scraper::Selector;

        let mut images = Vec::new();

        if let Ok(img_selector) = Selector::parse("img[src]") {
            for element in document.select(&img_selector) {
                if let Some(src) = element.value().attr("src") {
                    let full_url = self.resolve_url(base_url, src);
                    if !images.contains(&full_url) {
                        images.push(full_url);
                    }
                }
            }
        }

        images
    }

    /// Extract links
    fn extract_links(&self, document: &scraper::Html, base_url: &str) -> Vec<String> {
        use scraper::Selector;

        let mut links = Vec::new();

        if let Ok(link_selector) = Selector::parse("a[href]") {
            for element in document.select(&link_selector) {
                if let Some(href) = element.value().attr("href") {
                    if !href.starts_with('#') && !href.starts_with("javascript:") {
                        let full_url = self.resolve_url(base_url, href);
                        if !links.contains(&full_url) {
                            links.push(full_url);
                        }
                    }
                }
            }
        }

        links
    }

    /// Extract tags/keywords
    fn extract_tags(&self, document: &scraper::Html) -> Vec<String> {
        use scraper::Selector;

        let mut tags = Vec::new();

        // Extract from meta keywords
        if let Some(keywords) = self.extract_meta(document, "keywords") {
            tags.extend(
                keywords.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()),
            );
        }

        // Extract from article tags
        if let Ok(tag_selector) = Selector::parse("a[rel='tag'], .tag, .tags a") {
            for element in document.select(&tag_selector) {
                let tag = element.text().collect::<String>().trim().to_string();
                if !tag.is_empty() && !tags.contains(&tag) {
                    tags.push(tag);
                }
            }
        }

        tags
    }

    /// Determine content type
    fn determine_content_type(&self, url: &str, document: &scraper::Html) -> ContentType {
        use scraper::Selector;

        // Check URL patterns
        if url.contains("/blog/") || url.contains("/post/") {
            return ContentType::BlogPost;
        }
        if url.contains("/docs/") || url.contains("/documentation/") {
            return ContentType::Documentation;
        }
        if url.contains("youtube.com") || url.contains("vimeo.com") {
            return ContentType::Video;
        }
        if url.contains("twitter.com") || url.contains("x.com") {
            return ContentType::SocialMedia;
        }

        // Check meta type
        if let Some(og_type) = self.extract_meta(document, "og:type") {
            match og_type.as_str() {
                "article" => return ContentType::Article,
                "blog" => return ContentType::BlogPost,
                "video" | "video.movie" | "video.episode" => return ContentType::Video,
                _ => {}
            }
        }

        // Check for article tag
        if let Ok(article_selector) = Selector::parse("article") {
            if document.select(&article_selector).next().is_some() {
                return ContentType::Article;
            }
        }

        ContentType::Unknown
    }

    /// Resolve relative URLs to absolute
    fn resolve_url(&self, base_url: &str, relative_url: &str) -> String {
        if relative_url.starts_with("http://") || relative_url.starts_with("https://") {
            return relative_url.to_string();
        }

        if relative_url.starts_with("//") {
            return format!("https:{}", relative_url);
        }

        if relative_url.starts_with('/') {
            if let Ok(base) = url::Url::parse(base_url) {
                if let Ok(resolved) = base.join(relative_url) {
                    return resolved.to_string();
                }
            }
        }

        relative_url.to_string()
    }

    /// Extract readable content using readability algorithm
    pub async fn extract_readable(&self, url: &str) -> Result<String> {
        let content = self.extract_from_url(url).await?;

        // Simple readability: combine title and main content
        let mut readable = String::new();

        if let Some(title) = content.title {
            readable.push_str(&title);
            readable.push_str("\n\n");
        }

        if let Some(author) = content.author {
            readable.push_str(&format!("By {}\n", author));
        }

        if let Some(date) = content.published_date {
            readable.push_str(&format!("Published: {}\n\n", date.format("%B %d, %Y")));
        }

        readable.push_str(&content.text);

        Ok(readable)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_url() {
        let extractor = ContentExtractor::new();

        assert_eq!(
            extractor.resolve_url("https://example.com/page", "https://other.com/file"),
            "https://other.com/file"
        );

        assert_eq!(
            extractor.resolve_url("https://example.com/page", "/file"),
            "https://example.com/file"
        );

        assert_eq!(
            extractor.resolve_url("https://example.com/page", "//cdn.com/file"),
            "https://cdn.com/file"
        );
    }
}
