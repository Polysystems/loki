//! GraphQL Client and Query Builder
//!
//! Advanced GraphQL client with schema introspection, dynamic query building,
//! and integration with Loki's safety and memory systems.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, error, info};

use crate::memory::{CognitiveMemory, MemoryMetadata};
use crate::safety::{ActionType, ActionValidator};

/// GraphQL endpoint configuration
#[derive(Debug, Clone)]
pub struct GraphQLConfig {
    pub endpoint: String,
    pub headers: HashMap<String, String>,
    pub timeout_seconds: u64,
    pub max_query_depth: usize,
    pub cache_ttl_seconds: u64,
}

impl Default for GraphQLConfig {
    fn default() -> Self {
        Self {
            endpoint: String::new(),
            headers: HashMap::new(),
            timeout_seconds: 30,
            max_query_depth: 10,
            cache_ttl_seconds: 300, // 5 minutes
        }
    }
}

/// GraphQL query request
#[derive(Debug, Clone, Serialize)]
pub struct GraphQLRequest {
    pub query: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub variables: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_name: Option<String>,
}

/// GraphQL response
#[derive(Debug, Clone, Deserialize)]
pub struct GraphQLResponse {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub errors: Vec<GraphQLError>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<serde_json::Value>,
}

/// GraphQL error
#[derive(Debug, Clone, Deserialize)]
pub struct GraphQLError {
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub locations: Option<Vec<GraphQLLocation>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<Vec<serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<serde_json::Value>,
}

/// GraphQL error location
#[derive(Debug, Clone, Deserialize)]
pub struct GraphQLLocation {
    pub line: u32,
    pub column: u32,
}

/// GraphQL schema introspection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphQLSchema {
    pub types: Vec<GraphQLType>,
    pub query_type: Option<String>,
    pub mutation_type: Option<String>,
    pub subscription_type: Option<String>,
    pub directives: Vec<GraphQLDirective>,
}

/// GraphQL type definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphQLType {
    pub name: String,
    pub description: Option<String>,
    pub kind: String,
    pub fields: Option<Vec<GraphQLField>>,
    pub possible_types: Option<Vec<String>>,
}

/// GraphQL field definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphQLField {
    pub name: String,
    pub description: Option<String>,
    pub r#type: String,
    pub args: Vec<GraphQLArgument>,
}

/// GraphQL argument definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphQLArgument {
    pub name: String,
    pub description: Option<String>,
    pub r#type: String,
    pub default_value: Option<String>,
}

/// GraphQL directive definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphQLDirective {
    pub name: String,
    pub description: Option<String>,
    pub locations: Vec<String>,
    pub args: Vec<GraphQLArgument>,
}

/// Query builder for constructing GraphQL queries dynamically
#[derive(Debug, Clone)]
pub struct QueryBuilder {
    operation_type: String,
    selections: Vec<String>,
    variables: HashMap<String, serde_json::Value>,
    fragments: Vec<String>,
}

impl QueryBuilder {
    /// Create a new query builder
    pub fn new(operation_type: &str) -> Self {
        Self {
            operation_type: operation_type.to_string(),
            selections: Vec::new(),
            variables: HashMap::new(),
            fragments: Vec::new(),
        }
    }

    /// Add a field selection
    pub fn field(mut self, name: &str) -> Self {
        self.selections.push(name.to_string());
        self
    }

    /// Add a field with arguments
    pub fn field_with_args(mut self, name: &str, args: &[(&str, serde_json::Value)]) -> Self {
        let args_str = args
            .iter()
            .map(|(k, v)| format!("{}: {}", k, serde_json::to_string(v).unwrap_or_default()))
            .collect::<Vec<_>>()
            .join(", ");
        self.selections.push(format!("{}({})", name, args_str));
        self
    }

    /// Add a nested selection
    pub fn nested(mut self, field: &str, selections: &[&str]) -> Self {
        let nested = format!("{} {{ {} }}", field, selections.join(" "));
        self.selections.push(nested);
        self
    }

    /// Add a variable
    pub fn variable(mut self, name: &str, value: serde_json::Value) -> Self {
        self.variables.insert(name.to_string(), value);
        self
    }

    /// Add a fragment
    pub fn fragment(mut self, fragment: &str) -> Self {
        self.fragments.push(fragment.to_string());
        self
    }

    /// Build the final query string
    pub fn build(self) -> (String, Option<serde_json::Value>) {
        let mut query = String::new();

        // Add fragments
        for fragment in &self.fragments {
            query.push_str(fragment);
            query.push('\n');
        }

        // Add operation
        query.push_str(&format!("{} {{\n", self.operation_type));

        // Add selections
        for selection in &self.selections {
            query.push_str(&format!("  {}\n", selection));
        }

        query.push_str("}\n");

        let variables = if self.variables.is_empty() {
            None
        } else {
            Some(serde_json::Value::Object(
                self.variables.into_iter().map(|(k, v)| (k, v)).collect(),
            ))
        };

        (query, variables)
    }
}

/// Advanced GraphQL client with caching and safety integration
pub struct GraphQLClient {
    /// HTTP client
    client: Client,

    /// Configuration
    config: GraphQLConfig,

    /// Memory system for caching
    memory: Arc<CognitiveMemory>,

    /// Safety validator
    validator: Option<Arc<ActionValidator>>,

    /// Schema cache
    schema_cache: Arc<RwLock<Option<GraphQLSchema>>>,

    /// Response cache
    response_cache: Arc<RwLock<HashMap<String, (serde_json::Value, std::time::SystemTime)>>>,
}

impl GraphQLClient {
    /// Create a new GraphQL client
    pub async fn new(
        config: GraphQLConfig,
        memory: Arc<CognitiveMemory>,
        validator: Option<Arc<ActionValidator>>,
    ) -> Result<Self> {
        info!("Initializing GraphQL client for endpoint: {}", config.endpoint);

        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("Content-Type", "application/json".parse()?);
        headers.insert("Accept", "application/json".parse()?);

        // Add custom headers
        for (key, value) in &config.headers {
            headers
                .insert(reqwest::header::HeaderName::from_bytes(key.as_bytes())?, value.parse()?);
        }

        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_seconds))
            .default_headers(headers)
            .build()?;

        Ok(Self {
            client,
            config,
            memory,
            validator,
            schema_cache: Arc::new(RwLock::new(None)),
            response_cache: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Execute a GraphQL query
    pub async fn query(
        &self,
        query: &str,
        variables: Option<serde_json::Value>,
        operation_name: Option<String>,
    ) -> Result<GraphQLResponse> {
        // Validate the operation through safety system
        if let Some(validator) = &self.validator {
            validator
                .validate_action(
                    ActionType::ApiCall {
                        provider: "graphql".to_string(),
                        endpoint: self.config.endpoint.clone(),
                    },
                    format!("GraphQL query: {}", query.chars().take(100).collect::<String>()),
                    vec!["GraphQL API call".to_string()],
                )
                .await?;
        }

        // Check cache first
        let cache_key = self.generate_cache_key(query, &variables);
        if let Some(cached) = self.get_cached_response(&cache_key).await {
            debug!("Returning cached GraphQL response");
            return Ok(serde_json::from_value(cached)?);
        }

        // Prepare request
        let request = GraphQLRequest { query: query.to_string(), variables, operation_name };

        debug!("Executing GraphQL query: {}", query.chars().take(200).collect::<String>());

        // Make the request
        let response = self.client.post(&self.config.endpoint).json(&request).send().await?;

        if !response.status().is_success() {
            return Err(anyhow!("GraphQL request failed with status: {}", response.status()));
        }

        let graphql_response: GraphQLResponse = response.json().await?;

        // Cache successful responses
        if graphql_response.errors.is_empty() {
            if let Some(data) = &graphql_response.data {
                self.cache_response(&cache_key, data.clone()).await;
            }
        }

        // Log errors
        for error in &graphql_response.errors {
            error!("GraphQL error: {}", error.message);
        }

        // Store query in memory for learning
        self.memory
            .store(
                format!("GraphQL query executed: {}", query.chars().take(100).collect::<String>()),
                vec![format!("Endpoint: {}", self.config.endpoint)],
                MemoryMetadata {
                    source: "graphql".to_string(),
                    tags: vec!["api".to_string(), "graphql".to_string()],
                    importance: 0.6,
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

        Ok(graphql_response)
    }

    /// Introspect the GraphQL schema
    pub async fn introspect_schema(&self) -> Result<GraphQLSchema> {
        // Check cache first
        {
            let cache = self.schema_cache.read().await;
            if let Some(schema) = cache.as_ref() {
                return Ok(schema.clone());
            }
        }

        info!("Introspecting GraphQL schema");

        let introspection_query = r#"
            query IntrospectionQuery {
                __schema {
                    queryType { name }
                    mutationType { name }
                    subscriptionType { name }
                    types {
                        ...FullType
                    }
                    directives {
                        name
                        description
                        locations
                        args {
                            ...InputValue
                        }
                    }
                }
            }
            
            fragment FullType on __Type {
                kind
                name
                description
                fields(includeDeprecated: true) {
                    name
                    description
                    args {
                        ...InputValue
                    }
                    type {
                        ...TypeRef
                    }
                }
                inputFields {
                    ...InputValue
                }
                interfaces {
                    ...TypeRef
                }
                enumValues(includeDeprecated: true) {
                    name
                    description
                }
                possibleTypes {
                    ...TypeRef
                }
            }
            
            fragment InputValue on __InputValue {
                name
                description
                type { ...TypeRef }
                defaultValue
            }
            
            fragment TypeRef on __Type {
                kind
                name
                ofType {
                    kind
                    name
                    ofType {
                        kind
                        name
                        ofType {
                            kind
                            name
                            ofType {
                                kind
                                name
                                ofType {
                                    kind
                                    name
                                    ofType {
                                        kind
                                        name
                                        ofType {
                                            kind
                                            name
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        "#;

        let response = self.query(introspection_query, None, None).await?;

        if !response.errors.is_empty() {
            return Err(anyhow!("Schema introspection failed: {:?}", response.errors));
        }

        let schema_data =
            response.data.ok_or_else(|| anyhow!("No schema data in introspection response"))?;

        // Parse the introspection result into our schema format
        // This is a simplified version - in practice you'd want more robust parsing
        let schema = self.parse_introspection_result(schema_data)?;

        // Cache the schema
        {
            let mut cache = self.schema_cache.write().await;
            *cache = Some(schema.clone());
        }

        info!("Schema introspection complete: {} types found", schema.types.len());

        Ok(schema)
    }

    /// Parse introspection result into schema structure
    fn parse_introspection_result(&self, data: serde_json::Value) -> Result<GraphQLSchema> {
        // This is a simplified parser - you'd want more robust parsing in practice
        let schema_obj =
            data.get("__schema").ok_or_else(|| anyhow!("No __schema in introspection result"))?;

        Ok(GraphQLSchema {
            types: Vec::new(), // Would parse types here
            query_type: schema_obj
                .get("queryType")
                .and_then(|t| t.get("name"))
                .and_then(|n| n.as_str())
                .map(|s| s.to_string()),
            mutation_type: schema_obj
                .get("mutationType")
                .and_then(|t| t.get("name"))
                .and_then(|n| n.as_str())
                .map(|s| s.to_string()),
            subscription_type: schema_obj
                .get("subscriptionType")
                .and_then(|t| t.get("name"))
                .and_then(|n| n.as_str())
                .map(|s| s.to_string()),
            directives: Vec::new(), // Would parse directives here
        })
    }

    /// Create a query builder
    pub fn query_builder(&self, operation_type: &str) -> QueryBuilder {
        QueryBuilder::new(operation_type)
    }

    /// Generate cache key for a query
    fn generate_cache_key(&self, query: &str, variables: &Option<serde_json::Value>) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(query.as_bytes());
        if let Some(vars) = variables {
            hasher.update(vars.to_string().as_bytes());
        }
        format!("{:?}", hasher.finalize())
    }

    /// Get cached response
    async fn get_cached_response(&self, key: &str) -> Option<serde_json::Value> {
        let cache = self.response_cache.read().await;
        if let Some((value, timestamp)) = cache.get(key) {
            let age = std::time::SystemTime::now().duration_since(*timestamp).unwrap_or_default();

            if age.as_secs() < self.config.cache_ttl_seconds {
                return Some(value.clone());
            }
        }
        None
    }

    /// Cache a response
    async fn cache_response(&self, key: &str, value: serde_json::Value) {
        let mut cache = self.response_cache.write().await;
        cache.insert(key.to_string(), (value, std::time::SystemTime::now()));

        // Clean old entries if cache is getting large
        if cache.len() > 1000 {
            let cutoff = std::time::SystemTime::now()
                - std::time::Duration::from_secs(self.config.cache_ttl_seconds);
            cache.retain(|_, (_, timestamp)| *timestamp > cutoff);
        }
    }

    /// Get schema information
    pub async fn get_schema(&self) -> Option<GraphQLSchema> {
        self.schema_cache.read().await.clone()
    }

    /// Clear all caches
    pub async fn clear_cache(&self) {
        let mut response_cache = self.response_cache.write().await;
        response_cache.clear();

        let mut schema_cache = self.schema_cache.write().await;
        *schema_cache = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_builder() {
        let (query, variables) = QueryBuilder::new("query")
            .field("user")
            .nested("posts", &["title", "content"])
            .field_with_args("comments", &[("limit", serde_json::json!(10))])
            .variable("userId", serde_json::json!("123"))
            .build();

        assert!(query.contains("query"));
        assert!(query.contains("user"));
        assert!(query.contains("posts { title content }"));
        assert!(variables.is_some());
    }
}
