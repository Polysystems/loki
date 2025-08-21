# 🗺️ Module Map

## Overview

This document provides a comprehensive map of Loki's 590+ modules, organized by category and function. Each module listing includes its purpose and key dependencies.

## Module Organization

```
src/
├── cognitive/       (100+ modules) - Cognitive processing
├── memory/          (45+ modules)  - Memory systems
├── tools/           (50+ modules)  - External integrations
├── tui/             (80+ modules)  - Terminal interface
├── models/          (25+ modules)  - LLM management
├── safety/          (35+ modules)  - Safety systems
├── social/          (30+ modules)  - Social features
├── story/           (20+ modules)  - Story-driven processing
├── cluster/         (15+ modules)  - Distributed systems
├── monitoring/      (20+ modules)  - Observability
└── [others]         (170+ modules) - Core functionality
```

## Core Modules

### Main Entry Points

| Module | Path | Purpose |
|--------|------|---------|
| **main.rs** | `src/main.rs` | Application entry point |
| **lib.rs** | `src/lib.rs` | Library interface |
| **error.rs** | `src/error.rs` | Error types and handling |

## Cognitive Modules (100+)

### Consciousness & Awareness

| Module | Purpose |
|--------|---------|
| **consciousness_stream.rs** | Main consciousness implementation |
| **consciousness_bridge.rs** | Bridge to consciousness systems |
| **consciousness_integration.rs** | Integration with other systems |
| **consciousness_orchestration_bridge.rs** | Orchestration interface |
| **meta_awareness.rs** | Self-reflection and introspection |
| **temporal_consciousness.rs** | Time-aware processing |
| **distributed_consciousness.rs** | Multi-node consciousness |

### Reasoning & Decision Making

| Module | Purpose |
|--------|---------|
| **decision_engine.rs** | Complex decision making |
| **decision_engine_integration.rs** | Decision system integration |
| **decision_learner.rs** | Learning from decisions |
| **decision_tracking.rs** | Decision history and metrics |
| **enhanced_processor.rs** | Advanced cognitive processing |
| **pathway_tracer.rs** | Reasoning path tracking |

### Theory of Mind & Empathy

| Module | Purpose |
|--------|---------|
| **theory_of_mind.rs** | Mental state modeling |
| **empathy_system.rs** | Emotional understanding |
| **social_context.rs** | Social situation analysis |
| **emotional_core.rs** | Core emotion processing |

### Learning & Adaptation

| Module | Purpose |
|--------|---------|
| **neuroplasticity.rs** | Adaptive learning mechanisms |
| **autonomous_evolution.rs** | Self-improvement |
| **self_modify.rs** | Code self-modification |
| **self_reflection.rs** | Performance self-analysis |

### Creative & Story Systems

| Module | Purpose |
|--------|---------|
| **story_driven_autonomy.rs** | Narrative-based processing |
| **story_driven_code_generation.rs** | Code as narrative |
| **story_driven_testing.rs** | Test generation stories |
| **story_driven_refactoring.rs** | Refactoring narratives |
| **story_driven_documentation.rs** | Documentation stories |

### Specialized Cognitive

| Module | Purpose |
|--------|---------|
| **thermodynamic_cognition.rs** | Energy-based cognition |
| **thermodynamic_optimization.rs** | Thermodynamic optimization |
| **three_gradient_coordinator.rs** | Multi-gradient optimization |
| **value_gradients.rs** | Value-based gradients |
| **subconscious.rs** | Background processing |

### Multi-Agent Coordination

| Module | Purpose |
|--------|---------|
| **multi_agent_coordinator.rs** | Agent coordination |
| **autonomous_loop.rs** | Autonomous operation loop |
| **unified_controller.rs** | Unified control interface |
| **orchestrator.rs** | System orchestration |

## Memory Modules (45+)

### Core Memory

| Module | Purpose |
|--------|---------|
| **mod.rs** | Memory module interface |
| **cache.rs** | Basic caching system |
| **cache_controller.rs** | Cache management |
| **simd_cache.rs** | SIMD-optimized cache |
| **persistence.rs** | Persistent storage |

### Memory Types

| Module | Purpose |
|--------|---------|
| **layers.rs** | Hierarchical memory layers |
| **associations.rs** | Memory associations |
| **embeddings.rs** | Vector embeddings |
| **pattern_learning.rs** | Pattern extraction |

### Advanced Memory

| Module | Purpose |
|--------|---------|
| **fractal_activation.rs** | Fractal memory patterns |
| **fractal_interface.rs** | Fractal system interface |
| **prefetch_engine.rs** | Predictive prefetching |

## Tool Modules (50+)

### Development Tools

| Module | Purpose |
|--------|---------|
| **github.rs** | GitHub integration |
| **code_analysis.rs** | Code analysis tools |
| **python_executor.rs** | Python code execution |
| **database_cognitive.rs** | Database operations |

### Web Tools

| Module | Purpose |
|--------|---------|
| **web_search.rs** | Web search integration |
| **autonomous_browser.rs** | Browser automation |
| **doc_crawler.rs** | Documentation crawler |
| **websocket.rs** | WebSocket connections |

### Communication Tools

| Module | Purpose |
|--------|---------|
| **slack.rs** | Slack integration |
| **discord.rs** | Discord integration |
| **email.rs** | Email functionality |
| **calendar.rs** | Calendar management |

### Creative Tools

| Module | Purpose |
|--------|---------|
| **creative_generators.rs** | Creative content generation |
| **creative_media.rs** | Media creation |
| **blender_integration.rs** | 3D modeling with Blender |
| **vision_system.rs** | Computer vision |

### System Tools

| Module | Purpose |
|--------|---------|
| **file_system.rs** | File operations |
| **computer_use.rs** | Computer automation |
| **task_management.rs** | Task tracking |
| **metrics_collector.rs** | Metrics collection |

### Advanced Tools

| Module | Purpose |
|--------|---------|
| **mcp_client.rs** | MCP server integration |
| **parallel_execution.rs** | Parallel tool execution |
| **intelligent_manager.rs** | Smart tool management |
| **api_connector.rs** | API integrations |
| **arxiv.rs** | Academic paper access |
| **graphql.rs** | GraphQL client |
| **vector_memory.rs** | Vector database |
| **emergent_types.rs** | Dynamic type generation |

## TUI Modules (80+)

### Core TUI

| Module | Purpose |
|--------|---------|
| **app.rs** | Main TUI application |
| **run.rs** | TUI runtime |
| **state.rs** | TUI state management |
| **components.rs** | UI components |
| **widgets.rs** | Custom widgets |

### Views & Interfaces

| Module | Purpose |
|--------|---------|
| **orchestration_view.rs** | System orchestration view |
| **collaborative_view.rs** | Multi-user collaboration |
| **plugin_view.rs** | Plugin management UI |
| **cost_optimization_view.rs** | Cost analysis UI |
| **agent_specialization_view.rs** | Agent management UI |

### Integration & Bridging

| Module | Purpose |
|--------|---------|
| **ui_bridge.rs** | UI system bridge |
| **system_connector.rs** | System integration |
| **tool_connector.rs** | Tool integration |
| **cognitive_stream_integration.rs** | Cognitive stream UI |
| **story_memory_integration.rs** | Story memory UI |
| **real_time_integration.rs** | Real-time updates |

### Advanced TUI Features

| Module | Purpose |
|--------|---------|
| **multiplexer.rs** | Multiple view management |
| **event_bus.rs** | Event handling |
| **session_manager.rs** | Session management |
| **state_sync.rs** | State synchronization |
| **task_decomposer.rs** | Task breakdown UI |
| **visual_components.rs** | Visual elements |

## Model Management (25+)

| Module | Purpose |
|--------|---------|
| **mod.rs** | Model module interface |
| **loader.rs** | Model loading |
| **config.rs** | Model configuration |
| **registry.rs** | Model registry |
| **orchestrator.rs** | Model orchestration |
| **ensemble.rs** | Model ensembles |
| **integration.rs** | LLM integration |
| **local_manager.rs** | Local model management |
| **distributed_serving.rs** | Distributed inference |
| **multi_agent_orchestrator.rs** | Multi-agent models |
| **performance_analytics.rs** | Performance tracking |
| **cost_manager.rs** | Cost optimization |
| **intelligent_cost_optimizer.rs** | Smart cost management |

## Safety & Security (35+)

| Module | Purpose |
|--------|---------|
| **mod.rs** | Safety module interface |
| **validator.rs** | Input validation |
| **limits.rs** | Resource limits |
| **audit.rs** | Audit logging |
| **encryption.rs** | Data encryption |
| **security_audit.rs** | Security scanning |
| **multi_agent_safety.rs** | Multi-agent safety |
| **integration_test.rs** | Safety testing |

## Social Integration (30+)

| Module | Purpose |
|--------|---------|
| **mod.rs** | Social module interface |
| **x_client.rs** | X/Twitter client |
| **x_consciousness.rs** | X integration with consciousness |
| **x_safety_wrapper.rs** | X safety wrapper |
| **oauth2.rs** | OAuth2 authentication |
| **attribution.rs** | Content attribution |
| **content_gen.rs** | Social content generation |

## Story System (20+)

| Module | Purpose |
|--------|---------|
| **engine.rs** | Story processing engine |
| **types.rs** | Story data types |
| **context_chain.rs** | Story context management |
| **context_retrieval.rs** | Context retrieval |
| **task_mapper.rs** | Task to story mapping |
| **templates.rs** | Story templates |
| **learning.rs** | Story-based learning |
| **visualization.rs** | Story visualization |
| **export_import.rs** | Story persistence |
| **story_sync.rs** | Story synchronization |
| **file_watcher.rs** | File change stories |
| **agent_story.rs** | Agent narratives |
| **agent_coordination.rs** | Multi-agent stories |
| **codebase_story.rs** | Code narratives |

## Infrastructure Modules

### Cluster & Distribution (15+)

| Module | Purpose |
|--------|---------|
| **coordinator.rs** | Cluster coordination |
| **discovery.rs** | Service discovery |
| **health_monitor.rs** | Health monitoring |
| **load_balancer.rs** | Load balancing |
| **intelligent_load_balancer.rs** | Smart load balancing |

### Monitoring (20+)

| Module | Purpose |
|--------|---------|
| **health.rs** | Health checks |
| **performance.rs** | Performance metrics |
| **real_time.rs** | Real-time monitoring |
| **recovery.rs** | Error recovery |
| **cost_analytics.rs** | Cost analysis |
| **distributed_safety.rs** | Distributed safety monitoring |
| **production_optimizer.rs** | Production optimization |

### Storage & Persistence

| Module | Purpose |
|--------|---------|
| **chat_history.rs** | Chat history storage |
| **secrets.rs** | Secret management |
| **story_persistence.rs** | Story storage |

### Streaming & Processing

| Module | Purpose |
|--------|---------|
| **buffer.rs** | Stream buffering |
| **pipeline.rs** | Processing pipeline |
| **processor.rs** | Stream processor |
| **consciousness_bridge.rs** | Consciousness streaming |
| **enhanced_context_processor.rs** | Context processing |

## Utility Modules

| Module | Purpose |
|--------|---------|
| **async_optimization.rs** | Async performance |
| **fs.rs** | File system utilities |
| **progress.rs** | Progress tracking |
| **stub_tracking.rs** | Stub management |
| **syntax.rs** | Syntax utilities |

## CLI Modules

| Module | Purpose |
|--------|---------|
| **mod.rs** | CLI interface |
| **check_apis.rs** | API verification |
| **plugin_commands.rs** | Plugin management |
| **safety_commands.rs** | Safety controls |
| **tui_commands.rs** | TUI commands |
| **ui_commands.rs** | UI commands |
| **x_commands.rs** | X/Twitter commands |

## Configuration Modules

| Module | Purpose |
|--------|---------|
| **mod.rs** | Config interface |
| **api_keys.rs** | API key management |
| **network.rs** | Network configuration |
| **secure_env.rs** | Secure environment |
| **setup.rs** | System setup |

## Database Modules

| Module | Purpose |
|--------|---------|
| **mod.rs** | Database interface |

## Authentication Modules

| Module | Purpose |
|--------|---------|
| **mod.rs** | Auth interface |
| **providers.rs** | Auth providers |
| **session.rs** | Session management |
| **storage.rs** | Auth storage |
| **tokens.rs** | Token management |

## Plugin System

| Module | Purpose |
|--------|---------|
| **api.rs** | Plugin API |
| **loader.rs** | Plugin loading |
| **manager.rs** | Plugin management |
| **marketplace.rs** | Plugin marketplace |
| **registry.rs** | Plugin registry |
| **sandbox.rs** | Plugin sandboxing |
| **wasm_engine.rs** | WebAssembly runtime |

## Testing & Development

| Module | Purpose |
|--------|---------|
| **tests.rs** | Test utilities |
| **benchmarking.rs** | Performance benchmarks |
| **integration_test.rs** | Integration tests |

## Module Statistics

### By Category
- **Cognitive**: 100+ modules
- **TUI**: 80+ modules
- **Tools**: 50+ modules
- **Memory**: 45+ modules
- **Safety**: 35+ modules
- **Social**: 30+ modules
- **Models**: 25+ modules
- **Story**: 20+ modules
- **Monitoring**: 20+ modules
- **Others**: 185+ modules

### Total: 590+ modules

## Module Dependencies

### Core Dependencies
Most modules depend on:
- `tokio` - Async runtime
- `serde` - Serialization
- `anyhow`/`thiserror` - Error handling
- `tracing` - Logging and tracing

### Inter-module Dependencies
- Cognitive modules → Memory modules
- TUI modules → Bridge modules
- Tool modules → Safety modules
- All modules → Error module

## Finding Modules

### By Functionality
1. **Cognitive Processing**: `src/cognitive/`
2. **Memory Operations**: `src/memory/`
3. **External Tools**: `src/tools/`
4. **User Interface**: `src/tui/`, `src/ui/`
5. **Safety & Security**: `src/safety/`

### By Layer
1. **Application Layer**: `src/tui/`, `src/cli/`, `src/ui/`
2. **Consciousness Layer**: `src/cognitive/`
3. **Bridge Layer**: `*_bridge.rs` files
4. **Memory Layer**: `src/memory/`
5. **Tool Layer**: `src/tools/`
6. **Safety Layer**: `src/safety/`

---

Next: [Cognitive Architecture](cognitive_architecture.md) | [Data Flow](data_flow.md)