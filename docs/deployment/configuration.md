# ⚙️ Configuration Guide

## Overview

Loki's configuration system provides flexible control over all aspects of the system, from API keys and model selection to cognitive features and performance tuning. Configuration can be managed through environment variables, configuration files, and runtime commands.

## Configuration Hierarchy

Configuration sources are evaluated in the following order (later sources override earlier ones):

1. **Default Values** - Built-in defaults
2. **System Config** - `/etc/loki/config.yaml`
3. **User Config** - `~/.loki/config.yaml`
4. **Project Config** - `./loki.yaml`
5. **Environment Variables** - `LOKI_*` variables
6. **Command Line** - Runtime flags

## Environment Variables

### API Keys

Essential API keys for LLM providers:

```bash
# Primary LLM Providers
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GEMINI_API_KEY="..."
export MISTRAL_API_KEY="..."
export DEEPSEEK_API_KEY="..."

# Local Models
export OLLAMA_HOST="http://localhost:11434"

# Development Tools
export GITHUB_TOKEN="ghp_..."
export GITLAB_TOKEN="glpat-..."

# Social Platforms
export X_CONSUMER_KEY="..."
export X_CONSUMER_SECRET="..."
export X_ACCESS_TOKEN="..."
export X_ACCESS_TOKEN_SECRET="..."
export SLACK_TOKEN="xoxb-..."
export DISCORD_TOKEN="..."

# Databases
export DATABASE_URL="postgresql://user:pass@localhost/loki"
export REDIS_URL="redis://localhost:6379"

# Cloud Services
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export GCP_PROJECT_ID="..."
export AZURE_CLIENT_ID="..."
```

### System Settings

```bash
# Core Settings
export LOKI_HOME="~/.loki"                    # Home directory
export LOKI_DATA_DIR="~/.loki/data"           # Data storage
export LOKI_LOG_DIR="~/.loki/logs"            # Log files
export LOKI_PLUGIN_DIR="~/.loki/plugins"      # Plugins
export LOKI_CONFIG="~/.loki/config.yaml"      # Config file

# Logging
export RUST_LOG="info"                         # Log level
export LOKI_LOG_LEVEL="info"                   # Loki-specific
export LOKI_LOG_FORMAT="json"                  # json|text
export LOKI_LOG_COLOR="auto"                   # auto|always|never

# Performance
export LOKI_MAX_WORKERS="8"                    # Worker threads
export LOKI_CACHE_SIZE="2GB"                   # Memory cache
export LOKI_PARALLEL_TOOLS="5"                 # Concurrent tools
export LOKI_REQUEST_TIMEOUT="30"               # Seconds
export LOKI_BATCH_SIZE="100"                   # Batch operations

# Features
export LOKI_COGNITIVE_ENABLED="true"           # Cognitive features
export LOKI_MEMORY_PERSISTENT="true"           # Persistent memory
export LOKI_TOOLS_ENABLED="true"               # Tool usage
export LOKI_SAFETY_ENABLED="true"              # Safety checks
export LOKI_TELEMETRY_ENABLED="false"          # Anonymous telemetry
```

## Configuration File

### Main Configuration (`config.yaml`)

```yaml
# Loki Configuration
version: "1.0"

# System Settings
system:
  name: "Loki"
  instance_id: "${HOSTNAME}-${USER}"
  timezone: "UTC"
  locale: "en_US"

# LLM Configuration
llm:
  default_provider: "openai"  # openai|anthropic|gemini|mistral|ollama
  default_model: "gpt-4-turbo-preview"
  
  providers:
    openai:
      api_key: "${OPENAI_API_KEY}"
      base_url: "https://api.openai.com/v1"
      models:
        - "gpt-4-turbo-preview"
        - "gpt-4"
        - "gpt-3.5-turbo"
      rate_limit: 10000  # requests per hour
      
    anthropic:
      api_key: "${ANTHROPIC_API_KEY}"
      base_url: "https://api.anthropic.com"
      models:
        - "claude-3-opus-20240229"
        - "claude-3-sonnet-20240229"
        - "claude-3-haiku-20240307"
      rate_limit: 5000
      
    gemini:
      api_key: "${GEMINI_API_KEY}"
      models:
        - "gemini-1.5-pro"
        - "gemini-1.5-flash"
      
    ollama:
      host: "${OLLAMA_HOST:-http://localhost:11434}"
      models:
        - "llama3"
        - "mistral"
        - "deepseek-coder"

  # Model Selection Strategy
  selection:
    strategy: "adaptive"  # fixed|round-robin|adaptive|cost-optimized
    fallback_enabled: true
    retry_attempts: 3
    retry_delay: "1s"

# Cognitive Configuration
cognitive:
  enabled: true
  
  consciousness:
    enabled: true
    stream_interval: "100ms"
    attention_capacity: 7  # Miller's 7±2
    meta_awareness: true
    
  reasoning:
    enabled: true
    max_depth: 10
    timeout: "30s"
    parallel_paths: 3
    
    engines:
      logical: true
      analogical: true
      causal: true
      probabilistic: true
      abductive: true
      
  creativity:
    enabled: true
    temperature: 0.8
    divergence_factor: 0.6
    novelty_threshold: 0.7
    
  empathy:
    enabled: true
    emotion_recognition: true
    response_adaptation: true
    social_awareness: true
    
  learning:
    enabled: true
    learning_rate: 0.01
    consolidation_interval: "1h"
    reinforcement: true
    
  theory_of_mind:
    enabled: true
    max_mental_models: 10
    belief_tracking: true
    intention_prediction: true

# Memory Configuration
memory:
  enabled: true
  persistent: true
  path: "${LOKI_DATA_DIR}/memory"
  
  hierarchy:
    sensory:
      duration: "1s"
      capacity: "unlimited"
      
    working:
      capacity: 9  # 7±2 items
      rehearsal: true
      rehearsal_interval: "100ms"
      
    short_term:
      capacity: 1000
      retention: "24h"
      consolidation_threshold: 0.7
      
    long_term:
      storage: "rocksdb"
      compression: true
      index_type: "hnsw"  # hnsw|annoy|faiss
      embedding_dim: 768
      
  cache:
    enabled: true
    size: "${LOKI_CACHE_SIZE:-1GB}"
    ttl: "1h"
    strategy: "lru"  # lru|lfu|arc
    
  forgetting:
    enabled: true
    decay_rate: 0.5
    interference_threshold: 0.8
    min_strength: 0.1
    cleanup_interval: "24h"

# Tool Configuration
tools:
  enabled: true
  
  execution:
    parallel: true
    max_concurrent: "${LOKI_PARALLEL_TOOLS:-5}"
    timeout: "30s"
    retry_on_failure: true
    
  categories:
    development:
      enabled: true
      tools:
        - github
        - gitlab
        - code_analysis
        - testing
        
    web:
      enabled: true
      tools:
        - search
        - browse
        - scrape
        - api_client
        
    social:
      enabled: true
      tools:
        - x_twitter
        - slack
        - discord
        - email
        
    data:
      enabled: true
      tools:
        - database
        - vector_store
        - graph_db
        - cache
        
    creative:
      enabled: true
      tools:
        - image_gen
        - audio_gen
        - video_gen
        - 3d_modeling
        
    system:
      enabled: true
      tools:
        - file_system
        - process
        - network
        - monitor
        
  rate_limits:
    github: 5000  # per hour
    openai: 60    # per minute
    web_search: 10000  # per day
    
  sandboxing:
    enabled: true
    runtime: "wasm"  # wasm|docker|firecracker
    memory_limit: "512MB"
    cpu_limit: "50%"
    network: "restricted"

# Multi-Agent Configuration
agents:
  enabled: true
  max_agents: 10
  
  coordination:
    strategy: "hierarchical"  # hierarchical|peer-to-peer|swarm
    communication: "message-passing"
    consensus: "majority"  # majority|unanimous|weighted
    
  specializations:
    - type: "researcher"
      model: "gpt-4"
      tools: ["web_search", "arxiv", "wikipedia"]
      
    - type: "coder"
      model: "deepseek-coder"
      tools: ["github", "code_analysis", "testing"]
      
    - type: "writer"
      model: "claude-3-opus"
      tools: ["grammar", "style", "research"]
      
    - type: "analyst"
      model: "gemini-1.5-pro"
      tools: ["database", "visualization", "statistics"]

# Story-Driven Configuration
story:
  enabled: true
  
  narrative:
    coherence_check: true
    context_window: 10000
    chapter_size: 500
    
  persistence:
    save_stories: true
    path: "${LOKI_DATA_DIR}/stories"
    format: "markdown"  # markdown|json|yaml
    
  templates:
    - "hero_journey"
    - "problem_solution"
    - "discovery"
    - "transformation"

# Safety Configuration
safety:
  enabled: true
  
  limits:
    max_tokens: 4000
    max_requests_per_minute: 100
    max_memory_usage: "4GB"
    max_cpu_usage: "80%"
    max_execution_time: "5m"
    
  validation:
    input_sanitization: true
    output_filtering: true
    code_review: true
    
  audit:
    enabled: true
    log_level: "info"  # debug|info|warn|error
    path: "${LOKI_LOG_DIR}/audit.log"
    retention: "90d"
    
  policies:
    - "no_harmful_content"
    - "no_personal_data"
    - "no_unauthorized_access"
    - "rate_limiting"

# Network Configuration
network:
  server:
    enabled: false
    host: "0.0.0.0"
    port: 8080
    ssl: false
    
  client:
    timeout: "30s"
    retry: true
    proxy: "${HTTP_PROXY}"
    
  security:
    cors:
      enabled: true
      origins: ["http://localhost:3000"]
    
    auth:
      enabled: false
      providers:
        - "jwt"
        - "oauth2"

# Performance Configuration
performance:
  optimization:
    simd: true
    gpu: "auto"  # auto|cuda|metal|rocm|none
    compile_flags: "-C target-cpu=native"
    
  threading:
    workers: "${LOKI_MAX_WORKERS:-8}"
    async_runtime: "tokio"
    thread_pool: "rayon"
    
  caching:
    query_cache: true
    result_cache: true
    embedding_cache: true
    
  profiling:
    enabled: false
    flamegraph: false
    traces: false

# Monitoring Configuration
monitoring:
  metrics:
    enabled: true
    provider: "prometheus"
    endpoint: "/metrics"
    interval: "10s"
    
  tracing:
    enabled: false
    provider: "jaeger"
    endpoint: "http://localhost:14268"
    
  health:
    enabled: true
    endpoint: "/health"
    checks:
      - "memory"
      - "cpu"
      - "disk"
      - "api_connections"

# Plugin Configuration
plugins:
  enabled: true
  directory: "${LOKI_PLUGIN_DIR}"
  auto_load: true
  
  security:
    sandboxed: true
    permissions:
      - "read_files"
      - "network_access"
      - "tool_execution"
    
  marketplace:
    enabled: true
    url: "https://plugins.loki.ai"
    auto_update: false

# Development Configuration
development:
  debug: false
  hot_reload: false
  mock_apis: false
  
  testing:
    enabled: false
    fixtures: "./tests/fixtures"
    coverage: false
    
  benchmarking:
    enabled: false
    baseline: "main"
    iterations: 100
```

### Model-Specific Configuration

```yaml
# models.yaml
models:
  gpt-4-turbo-preview:
    temperature: 0.7
    max_tokens: 4000
    top_p: 0.9
    frequency_penalty: 0.0
    presence_penalty: 0.0
    
  claude-3-opus:
    temperature: 0.6
    max_tokens: 4000
    top_k: 40
    
  gemini-1.5-pro:
    temperature: 0.8
    max_output_tokens: 8192
    top_k: 40
    top_p: 0.95
    
  llama3:
    temperature: 0.7
    max_tokens: 2048
    repeat_penalty: 1.1
```

## Configuration Profiles

### Development Profile

```yaml
# config.dev.yaml
profile: development

system:
  debug: true
  
llm:
  default_model: "gpt-3.5-turbo"  # Cheaper for development
  
cognitive:
  reasoning:
    max_depth: 5  # Faster responses
    
safety:
  limits:
    max_tokens: 1000  # Smaller for testing
    
monitoring:
  metrics:
    enabled: true
    interval: "1s"  # More frequent
```

### Production Profile

```yaml
# config.prod.yaml
profile: production

system:
  debug: false
  
llm:
  default_model: "gpt-4-turbo-preview"
  selection:
    strategy: "cost-optimized"
    
performance:
  optimization:
    simd: true
    gpu: "cuda"
    
safety:
  audit:
    enabled: true
    log_level: "warn"
    
monitoring:
  metrics:
    enabled: true
  tracing:
    enabled: true
```

### Minimal Profile

```yaml
# config.minimal.yaml
profile: minimal

llm:
  default_provider: "ollama"
  default_model: "llama3"
  
cognitive:
  enabled: false
  
tools:
  enabled: false
  
memory:
  persistent: false
```

## Runtime Configuration

### Dynamic Configuration Updates

```rust
// Programmatic configuration
use loki::config::Config;

let mut config = Config::load()?;
config.set("llm.default_model", "claude-3-opus")?;
config.set("cognitive.reasoning.max_depth", 15)?;
config.save()?;
```

### CLI Configuration

```bash
# Set configuration values
loki config set llm.default_model "gpt-4"
loki config set memory.cache.size "2GB"

# Get configuration values
loki config get llm.default_model
loki config get --all

# Use specific profile
loki --profile production chat "Hello"

# Override with environment
LOKI_LLM_DEFAULT_MODEL=claude-3 loki chat "Hello"
```

## Configuration Validation

### Schema Validation

```yaml
# config-schema.yaml
type: object
required:
  - version
  - llm
properties:
  version:
    type: string
    pattern: "^\\d+\\.\\d+$"
  llm:
    type: object
    required:
      - default_provider
    properties:
      default_provider:
        type: string
        enum: [openai, anthropic, gemini, mistral, ollama]
```

### Validation Command

```bash
# Validate configuration
loki config validate

# Validate specific file
loki config validate --file custom-config.yaml

# Check for missing API keys
loki config check-keys
```

## Secret Management

### Using Secret Managers

```yaml
# Using environment variables
api_key: "${OPENAI_API_KEY}"

# Using file reference
api_key: "file:///etc/secrets/openai.key"

# Using secret manager
api_key: "vault://secret/openai/key"
api_key: "aws-secrets://loki/openai-key"
api_key: "gcp-secrets://projects/myproject/secrets/openai"
```

### Encryption

```bash
# Encrypt sensitive config
loki config encrypt --key $ENCRYPTION_KEY

# Decrypt for editing
loki config decrypt --key $ENCRYPTION_KEY
```

## Best Practices

### Configuration Management

1. **Use Profiles**: Separate dev/staging/production configs
2. **Environment Variables**: Keep secrets in environment
3. **Version Control**: Track config changes (exclude secrets)
4. **Validation**: Always validate before deployment
5. **Documentation**: Document custom settings

### Security

1. **Never Commit Secrets**: Use environment variables
2. **Encrypt Sensitive Data**: Use encryption for stored configs
3. **Rotate Keys**: Regularly rotate API keys
4. **Audit Access**: Log configuration changes
5. **Least Privilege**: Minimize permissions

### Performance

1. **Profile Settings**: Optimize for your use case
2. **Cache Appropriately**: Balance memory vs performance
3. **Monitor Metrics**: Track resource usage
4. **Tune Workers**: Adjust based on CPU cores
5. **Batch Operations**: Configure batch sizes

## Troubleshooting

### Common Issues

**Configuration not loading:**
```bash
# Check config path
loki config path

# Validate syntax
loki config validate

# Check permissions
ls -la ~/.loki/config.yaml
```

**API key errors:**
```bash
# Check environment
env | grep API_KEY

# Test specific provider
loki check-apis --provider openai
```

**Performance issues:**
```bash
# Check current settings
loki config get performance

# Monitor resources
loki diagnostic --performance
```

---

Next: [Installation Guide](installation_guide.md) | [Performance Tuning](performance_tuning.md)