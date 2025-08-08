# Task Priority Mapping in Message Processor

## Overview
The message processor now includes sophisticated task priority mapping that intelligently converts between NLP-detected priorities and the task management system priorities, with additional urgency detection from natural language.

## Features

### 1. Priority Mapping
Direct mapping between NLP and task management priorities:
- `NlpTaskPriority::Critical` → `ToolTaskPriority::Critical`
- `NlpTaskPriority::High` → `ToolTaskPriority::High`
- `NlpTaskPriority::Medium` → `ToolTaskPriority::Medium`
- `NlpTaskPriority::Low` → `ToolTaskPriority::Low`

### 2. Urgency Detection
The processor analyzes message content for urgency indicators:

**High Priority Indicators:**
- "urgent", "asap", "immediately", "critical"
- Applied to task constraints with priority: "high"

**Medium Priority Indicators:**
- "important", "soon", "priority"
- Applied to task constraints with priority: "medium"

**Low Priority Indicators:**
- "whenever", "eventually", "low priority"
- Applied to task constraints with priority: "low"

**Default:**
- Messages without urgency indicators get priority: "normal"

### 3. Enhanced Task Context
When creating tasks from NLP extraction, the processor builds rich context including:

- **Task Type**: The detected type (CodingTask, DocumentationTask, etc.)
- **Confidence Level**: 
  - ✅ High confidence (>90%)
  - ⚠️ Low confidence (<70%)
- **Estimated Effort**: Converted to human-readable format
  - Minutes for tasks < 1 hour
  - Hours for longer tasks
- **Dependencies**: Listed if present

### 4. Due Date Calculation
Automatically calculates due dates based on:
- Estimated effort from NLP extraction
- Current time + effort duration
- Defaults to 1 day if no effort estimate

### 5. Task Creation Notifications
When tasks are created, the processor:
- Logs detailed information with priority and confidence
- Sends user-visible notifications through the response channel
- Shows abbreviated task IDs for easy reference

## Usage Example

```rust
// User input with urgency
"urgent: fix the authentication bug in the login system"

// Results in:
// - Task priority: High (from "urgent")
// - Task type: CodingTask (from "fix")
// - Notification: "📋 Task created: Fix the authentication bug in the login system (ID: abc12345)"
// - Context includes: "Type: CodingTask\n✅ High confidence: 92%\nEstimated time: 45 minutes"
```

## Integration Points

1. **NLP Orchestrator**: Provides ExtractedTask with initial priority
2. **Message Processor**: Maps priorities and adds urgency detection
3. **Task Manager**: Receives properly prioritized tasks with context
4. **Response Channel**: Notifies users of task creation

## Benefits

1. **Intelligent Prioritization**: Combines NLP understanding with keyword detection
2. **Rich Context**: Provides task manager with comprehensive metadata
3. **User Feedback**: Immediate notification of task creation
4. **Flexible Mapping**: Easy to extend with new priority schemes
5. **Time Awareness**: Automatic due date calculation based on effort

## Future Enhancements

1. **Custom Priority Rules**: User-defined keywords for priority
2. **Context-Aware Priority**: Consider conversation history
3. **Dynamic Effort Estimation**: Learn from completed tasks
4. **Priority Escalation**: Automatic escalation for overdue tasks
5. **Team Priority Mapping**: Different priorities for different teams