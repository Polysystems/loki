# 🔄 Data Flow Architecture

## Overview

Understanding how data flows through Loki's 590+ modules is crucial for grasping the system's behavior. This document traces the journey of information from user input to response generation, showing how data transforms and enriches at each stage.

## High-Level Data Flow

```
User Input → Parsing → Enrichment → Cognitive Processing → 
Memory Integration → Tool Execution → Response Generation → Output
```

## Detailed Flow Diagram

```
┌──────────────┐
│  User Input  │
└──────┬───────┘
       ▼
┌──────────────────────────────────────┐
│         Input Processing              │
│  • Tokenization                       │
│  • Intent Recognition                 │
│  • Context Extraction                 │
└──────────────┬───────────────────────┘
               ▼
┌──────────────────────────────────────┐
│         Context Enrichment            │
│  • Memory Retrieval                   │
│  • Session History                    │
│  • User Preferences                   │
└──────────────┬───────────────────────┘
               ▼
┌──────────────────────────────────────┐
│      Cognitive Processing             │
│  ┌─────────────┐  ┌─────────────┐   │
│  │  Reasoning  │  │  Creativity │   │
│  └─────────────┘  └─────────────┘   │
│  ┌─────────────┐  ┌─────────────┐   │
│  │   Empathy   │  │   Planning  │   │
│  └─────────────┘  └─────────────┘   │
└──────────────┬───────────────────────┘
               ▼
┌──────────────────────────────────────┐
│         Decision Making               │
│  • Strategy Selection                 │
│  • Action Planning                    │
│  • Risk Assessment                    │
└──────────────┬───────────────────────┘
               ▼
┌──────────────────────────────────────┐
│         Tool Execution                │
│  • Tool Selection                     │
│  • Parallel Execution                 │
│  • Result Aggregation                 │
└──────────────┬───────────────────────┘
               ▼
┌──────────────────────────────────────┐
│       Response Generation             │
│  • Content Synthesis                  │
│  • Formatting                         │
│  • Safety Validation                  │
└──────────────┬───────────────────────┘
               ▼
┌──────────────────────────────────────┐
│         Memory Storage                │
│  • Experience Recording               │
│  • Knowledge Update                   │
│  • Pattern Learning                   │
└──────────────┬───────────────────────┘
               ▼
┌──────────────┐
│    Output    │
└──────────────┘
```

## Data Types and Structures

### Input Data Types

```rust
pub enum Input {
    Text(String),
    Command(Command),
    Query(Query),
    Task(Task),
    Feedback(Feedback),
}

pub struct EnrichedInput {
    raw: Input,
    intent: Intent,
    context: Context,
    metadata: Metadata,
    timestamp: DateTime<Utc>,
}
```

### Intermediate Data Structures

```rust
// Cognitive Processing Data
pub struct CognitiveState {
    thoughts: Vec<Thought>,
    reasoning_chains: Vec<ReasoningChain>,
    emotional_state: EmotionalState,
    attention_focus: AttentionFocus,
}

// Memory Data
pub struct MemoryContext {
    working_memory: WorkingMemory,
    relevant_memories: Vec<Memory>,
    knowledge_graph: GraphView,
}

// Decision Data
pub struct Decision {
    action: Action,
    confidence: f64,
    alternatives: Vec<Alternative>,
    rationale: String,
}
```

### Output Data Types

```rust
pub struct Response {
    content: Content,
    metadata: ResponseMetadata,
    side_effects: Vec<SideEffect>,
}

pub enum Content {
    Text(String),
    Code(CodeBlock),
    Structured(Value),
    Multimodal(Vec<MediaItem>),
}
```

## Processing Stages

### Stage 1: Input Processing

```rust
async fn process_input(raw: String) -> EnrichedInput {
    // Tokenization
    let tokens = tokenizer.tokenize(&raw);
    
    // Intent recognition
    let intent = intent_classifier.classify(&tokens).await;
    
    // Entity extraction
    let entities = entity_extractor.extract(&tokens).await;
    
    // Context building
    let context = Context {
        session_id: current_session(),
        user_id: current_user(),
        timestamp: Utc::now(),
        entities,
        history: get_recent_history(),
    };
    
    EnrichedInput {
        raw: Input::Text(raw),
        intent,
        context,
        metadata: extract_metadata(),
        timestamp: Utc::now(),
    }
}
```

### Stage 2: Context Enrichment

```rust
async fn enrich_context(input: EnrichedInput) -> EnrichedContext {
    // Retrieve relevant memories
    let memories = memory_bridge
        .retrieve_relevant(&input.context)
        .await;
    
    // Get user preferences
    let preferences = user_profile
        .get_preferences(input.context.user_id)
        .await;
    
    // Fetch session history
    let history = session_manager
        .get_history(input.context.session_id)
        .await;
    
    EnrichedContext {
        input,
        memories,
        preferences,
        history,
        knowledge: knowledge_graph.subgraph(&input.entities),
    }
}
```

### Stage 3: Cognitive Processing

```rust
async fn cognitive_processing(context: EnrichedContext) -> CognitiveResult {
    // Initialize cognitive state
    let mut state = CognitiveState::new();
    
    // Parallel cognitive processing
    let (reasoning, creativity, empathy) = tokio::join!(
        reasoning_engine.process(&context),
        creativity_engine.generate(&context),
        empathy_system.analyze(&context)
    );
    
    // Theory of mind modeling
    let mental_model = theory_of_mind
        .model_user_state(&context)
        .await;
    
    // Consciousness integration
    let integrated = consciousness_stream
        .integrate(reasoning, creativity, empathy, mental_model)
        .await;
    
    CognitiveResult {
        thoughts: integrated.thoughts,
        insights: integrated.insights,
        emotional_response: integrated.emotions,
        proposed_actions: integrated.actions,
    }
}
```

### Stage 4: Decision Making

```rust
async fn make_decision(cognitive: CognitiveResult) -> Decision {
    // Evaluate options
    let options = cognitive.proposed_actions;
    
    // Apply decision criteria
    let evaluated = decision_engine
        .evaluate(options, &DecisionCriteria {
            safety: 1.0,
            effectiveness: 0.8,
            efficiency: 0.6,
            user_satisfaction: 0.9,
        })
        .await;
    
    // Select best action
    let selected = evaluated
        .iter()
        .max_by_key(|e| e.score)
        .unwrap();
    
    Decision {
        action: selected.action.clone(),
        confidence: selected.confidence,
        alternatives: evaluated.into_iter()
            .filter(|e| e != selected)
            .take(3)
            .collect(),
        rationale: generate_rationale(selected),
    }
}
```

### Stage 5: Tool Execution

```rust
async fn execute_tools(decision: Decision) -> ExecutionResult {
    let tools_needed = identify_tools(&decision.action);
    
    // Parallel tool execution
    let futures = tools_needed
        .into_iter()
        .map(|tool| {
            tool_bridge.execute(tool, decision.params.clone())
        })
        .collect::<Vec<_>>();
    
    let results = futures::future::join_all(futures).await;
    
    // Aggregate results
    ExecutionResult {
        primary: results[0].clone(),
        supplementary: results[1..].to_vec(),
        metrics: collect_metrics(&results),
    }
}
```

### Stage 6: Response Generation

```rust
async fn generate_response(
    execution: ExecutionResult,
    cognitive: CognitiveResult,
) -> Response {
    // Synthesize content
    let content = response_generator
        .synthesize(&execution, &cognitive)
        .await;
    
    // Apply formatting
    let formatted = formatter
        .format(content, output_preferences())
        .await;
    
    // Safety validation
    let validated = safety_layer
        .validate(formatted)
        .await?;
    
    Response {
        content: validated,
        metadata: ResponseMetadata {
            processing_time: elapsed(),
            confidence: cognitive.confidence,
            tools_used: execution.tools,
        },
        side_effects: execution.side_effects,
    }
}
```

## Data Transformation Pipeline

### Transformation Chain

```
Raw Text → Tokens → Intent → Context → Thoughts → 
Decisions → Actions → Results → Response → Output
```

### Data Enrichment Points

1. **Input Enrichment**
   - Add timestamp
   - Extract metadata
   - Identify entities

2. **Context Enrichment**
   - Add memories
   - Include history
   - Attach preferences

3. **Cognitive Enrichment**
   - Add reasoning chains
   - Include insights
   - Attach emotions

4. **Response Enrichment**
   - Add explanations
   - Include confidence
   - Attach sources

## Async Data Flow

### Concurrent Processing

```rust
pub struct DataFlowOrchestrator {
    channels: HashMap<StageId, Channel>,
    workers: Vec<JoinHandle<()>>,
}

impl DataFlowOrchestrator {
    pub async fn process(&self, input: Input) -> Response {
        // Create processing pipeline
        let (tx1, rx1) = channel(100);
        let (tx2, rx2) = channel(100);
        let (tx3, rx3) = channel(100);
        
        // Spawn stage workers
        tokio::spawn(input_stage(rx1, tx2));
        tokio::spawn(cognitive_stage(rx2, tx3));
        tokio::spawn(response_stage(rx3));
        
        // Send input
        tx1.send(input).await?;
        
        // Await response
        response_receiver.recv().await
    }
}
```

### Stream Processing

```rust
pub async fn stream_processing(input_stream: impl Stream<Item = Input>) {
    input_stream
        .map(|input| process_input(input))
        .buffer_unordered(10)
        .map(|enriched| cognitive_processing(enriched))
        .buffer_unordered(5)
        .map(|cognitive| generate_response(cognitive))
        .for_each(|response| async {
            output_handler.send(response).await;
        })
        .await;
}
```

## Memory Data Flow

### Memory Write Path

```
Experience → Encoding → Importance Scoring → 
Storage Layer Selection → Persistence → Indexing
```

### Memory Read Path

```
Query → Embedding → Similarity Search → 
Retrieval → Filtering → Ranking → Return
```

## Error Handling Flow

```rust
pub enum DataFlowError {
    InputError(String),
    ProcessingError(String),
    ToolError(String),
    SafetyViolation(String),
}

impl DataFlow {
    async fn handle_error(&self, error: DataFlowError) -> Response {
        match error {
            DataFlowError::InputError(e) => {
                Response::error(format!("Invalid input: {}", e))
            },
            DataFlowError::ProcessingError(e) => {
                // Fallback to simpler processing
                self.fallback_processing().await
            },
            DataFlowError::ToolError(e) => {
                // Retry or use alternative tool
                self.retry_with_alternative().await
            },
            DataFlowError::SafetyViolation(e) => {
                Response::safety_error(e)
            },
        }
    }
}
```

## Performance Optimization

### Data Flow Optimizations

1. **Caching**
   - Cache frequent queries
   - Memoize expensive computations
   - Share immutable data

2. **Batching**
   - Batch memory queries
   - Group tool executions
   - Aggregate similar requests

3. **Streaming**
   - Stream large responses
   - Progressive rendering
   - Incremental processing

4. **Parallelization**
   - Parallel cognitive processing
   - Concurrent tool execution
   - Distributed memory search

## Monitoring Data Flow

### Metrics Collection

```rust
pub struct DataFlowMetrics {
    stage_latencies: HashMap<Stage, Duration>,
    data_sizes: HashMap<Stage, usize>,
    throughput: f64,
    error_rates: HashMap<Stage, f64>,
}
```

### Tracing

```rust
#[instrument]
async fn trace_data_flow(input: Input) -> Response {
    let span = span!(Level::INFO, "data_flow");
    let _enter = span.enter();
    
    span.record("input_size", input.size());
    
    let result = process(input).await;
    
    span.record("output_size", result.size());
    span.record("duration", elapsed());
    
    result
}
```

## Configuration

```yaml
data_flow:
  pipeline:
    buffer_size: 100
    max_concurrent: 10
    timeout: 30s
    
  stages:
    input:
      max_size: 10KB
      timeout: 1s
      
    cognitive:
      max_thoughts: 100
      timeout: 10s
      
    tools:
      max_parallel: 5
      timeout: 20s
      
    response:
      max_size: 50KB
      streaming: true
      
  optimization:
    caching: true
    batching: true
    compression: true
```

## Best Practices

1. **Data Validation**: Validate at each stage boundary
2. **Error Recovery**: Implement fallback paths
3. **Monitoring**: Track metrics at each stage
4. **Optimization**: Profile and optimize bottlenecks
5. **Documentation**: Document data schemas

---

Next: [Module Map](module_map.md) | [Layered Design](layered_design.md)