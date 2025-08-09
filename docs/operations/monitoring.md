# 📊 Monitoring Documentation

## Overview

Comprehensive monitoring ensures Loki operates efficiently and reliably. This guide covers metrics collection, alerting, visualization, and performance tracking.

## Metrics Collection

### Built-in Metrics

Loki exposes metrics in Prometheus format:

```bash
# Enable metrics endpoint
loki config set monitoring.metrics.enabled true

# Access metrics
curl http://localhost:8080/metrics
```

### Key Metrics

#### System Metrics
- `loki_cpu_usage_percent` - CPU utilization
- `loki_memory_usage_bytes` - Memory consumption
- `loki_disk_usage_bytes` - Disk space used
- `loki_uptime_seconds` - System uptime

#### Cognitive Metrics
- `loki_thoughts_generated_total` - Total thoughts
- `loki_reasoning_depth_histogram` - Reasoning depth distribution
- `loki_cognitive_latency_seconds` - Processing time
- `loki_decision_confidence_gauge` - Decision confidence

#### Memory Metrics
- `loki_memory_items_total` - Total memories
- `loki_memory_cache_hits_ratio` - Cache hit rate
- `loki_memory_retrieval_latency` - Retrieval time
- `loki_memory_storage_bytes` - Storage size

#### Tool Metrics
- `loki_tool_executions_total` - Tool executions
- `loki_tool_failures_total` - Failed executions
- `loki_tool_latency_seconds` - Execution time
- `loki_tool_rate_limit_remaining` - API limits

## Prometheus Setup

### Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'loki'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: '/metrics'
```

### Run Prometheus

```bash
# Docker
docker run -d \
  -p 9090:9090 \
  -v ./prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus

# Access UI
open http://localhost:9090
```

## Grafana Dashboards

### Setup Grafana

```bash
# Docker
docker run -d \
  -p 3000:3000 \
  grafana/grafana

# Default login: admin/admin
```

### Import Dashboard

```json
{
  "dashboard": {
    "title": "Loki AI Monitoring",
    "panels": [
      {
        "title": "CPU Usage",
        "targets": [
          {
            "expr": "loki_cpu_usage_percent"
          }
        ]
      },
      {
        "title": "Memory Usage",
        "targets": [
          {
            "expr": "loki_memory_usage_bytes / 1024 / 1024"
          }
        ]
      },
      {
        "title": "Cognitive Performance",
        "targets": [
          {
            "expr": "rate(loki_thoughts_generated_total[5m])"
          }
        ]
      }
    ]
  }
}
```

## Real-time Monitoring

### CLI Monitoring

```bash
# Real-time stats
loki monitor

# Specific component
loki monitor cognitive
loki monitor memory
loki monitor tools

# JSON output
loki monitor --json
```

### TUI Monitoring

In the TUI, press `F12` for monitoring view:

```
┌─ System Metrics ──────────────────────┐
│ CPU:     ████░░░░░░  42%             │
│ Memory:  ██████░░░░  63% (2.1GB)     │
│ Disk:    ███░░░░░░░  31% (15GB)      │
│ Network: ↓ 1.2MB/s  ↑ 0.3MB/s        │
└───────────────────────────────────────┘

┌─ Cognitive Metrics ───────────────────┐
│ Thoughts/sec:     12.3                │
│ Reasoning Depth:  7                   │
│ Confidence:       0.89                │
│ Active Agents:    3                   │
└───────────────────────────────────────┘
```

## Logging

### Log Configuration

```yaml
logging:
  level: info  # trace, debug, info, warn, error
  output: file
  file: ~/.loki/logs/loki.log
  rotation:
    max_size: 100MB
    max_files: 10
  format: json  # json or text
```

### Structured Logging

```json
{
  "timestamp": "2024-01-10T10:30:45Z",
  "level": "INFO",
  "module": "cognitive::reasoning",
  "message": "Reasoning completed",
  "fields": {
    "depth": 7,
    "duration_ms": 234,
    "confidence": 0.89
  }
}
```

### Log Aggregation

```bash
# Using Loki (Grafana Loki, not our AI)
docker run -d \
  -p 3100:3100 \
  grafana/loki

# Configure Promtail
# promtail.yml
clients:
  - url: http://localhost:3100/loki/api/v1/push
scrape_configs:
  - job_name: loki-ai
    static_configs:
      - targets:
          - localhost
        labels:
          job: loki-ai
          __path__: /home/user/.loki/logs/*.log
```

## Health Checks

### HTTP Health Endpoint

```bash
# Check health
curl http://localhost:8080/health

# Response
{
  "status": "healthy",
  "components": {
    "cognitive": "healthy",
    "memory": "healthy",
    "tools": "healthy"
  },
  "metrics": {
    "uptime": 3600,
    "memory_usage": 2147483648,
    "cpu_usage": 0.42
  }
}
```

### Automated Health Monitoring

```bash
# Health check script
#!/bin/bash
while true; do
  if ! curl -f http://localhost:8080/health; then
    echo "Loki unhealthy!" | mail -s "Alert" admin@example.com
  fi
  sleep 30
done
```

## Alerting

### Alert Rules

```yaml
# alerts.yml
groups:
  - name: loki_alerts
    rules:
      - alert: HighCPU
        expr: loki_cpu_usage_percent > 80
        for: 5m
        annotations:
          summary: "High CPU usage"
          
      - alert: MemoryLeak
        expr: rate(loki_memory_usage_bytes[1h]) > 100000000
        for: 10m
        annotations:
          summary: "Possible memory leak"
          
      - alert: ToolFailures
        expr: rate(loki_tool_failures_total[5m]) > 0.1
        for: 5m
        annotations:
          summary: "High tool failure rate"
```

### Alert Manager

```yaml
# alertmanager.yml
route:
  receiver: 'team-notifications'
  
receivers:
  - name: 'team-notifications'
    email_configs:
      - to: 'team@example.com'
    slack_configs:
      - api_url: 'YOUR_SLACK_WEBHOOK'
        channel: '#alerts'
```

## Performance Profiling

### CPU Profiling

```bash
# Enable profiling
loki --profile cpu

# Generate flamegraph
cargo flamegraph --bin loki

# View results
open flamegraph.svg
```

### Memory Profiling

```bash
# Using Valgrind
valgrind --tool=massif loki

# Analyze results
ms_print massif.out.*
```

### Tracing

```yaml
# Enable tracing
tracing:
  enabled: true
  provider: jaeger
  endpoint: http://localhost:14268
  sampling_rate: 0.1
```

## Custom Metrics

### Adding Custom Metrics

```rust
use prometheus::{Counter, Histogram, register_counter, register_histogram};

lazy_static! {
    static ref CUSTOM_COUNTER: Counter = register_counter!(
        "loki_custom_events_total",
        "Total custom events"
    ).unwrap();
    
    static ref CUSTOM_HISTOGRAM: Histogram = register_histogram!(
        "loki_custom_duration_seconds",
        "Custom operation duration"
    ).unwrap();
}

// Use in code
CUSTOM_COUNTER.inc();
CUSTOM_HISTOGRAM.observe(duration.as_secs_f64());
```

## Monitoring Best Practices

### Key Metrics to Watch
1. **Response Time**: P50, P95, P99 latencies
2. **Error Rate**: Failures per minute
3. **Resource Usage**: CPU, memory, disk
4. **Throughput**: Requests/thoughts per second
5. **Saturation**: Queue depths, thread pool usage

### Alert Thresholds
- CPU: Alert at 80%, critical at 95%
- Memory: Alert at 80%, critical at 95%
- Disk: Alert at 80%, critical at 90%
- Error rate: Alert at 1%, critical at 5%
- Latency: Alert at 2x baseline, critical at 5x

### Dashboard Organization
1. **Overview**: System health at a glance
2. **Cognitive**: Reasoning and thinking metrics
3. **Memory**: Storage and retrieval metrics
4. **Tools**: External integration metrics
5. **Agents**: Multi-agent coordination metrics

## Troubleshooting Monitoring

### Missing Metrics
```bash
# Check metrics endpoint
curl -v http://localhost:8080/metrics

# Verify configuration
loki config get monitoring.metrics.enabled
```

### High Cardinality
```bash
# Identify high cardinality metrics
curl http://localhost:8080/metrics | grep -c "^loki_"

# Reduce labels
loki config set monitoring.metrics.labels minimal
```

### Performance Impact
```bash
# Disable expensive metrics
loki config set monitoring.metrics.detailed false

# Reduce scrape frequency
# In prometheus.yml
scrape_interval: 60s
```

---

Next: [Troubleshooting](troubleshooting.md) | [Operations](../README.md)