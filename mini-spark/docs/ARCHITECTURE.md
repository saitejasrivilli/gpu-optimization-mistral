# Mini Spark Architecture

## Overview

Mini Spark is a distributed batch processing engine demonstrating core distributed systems concepts.

## Components

### Master
- Job and task state management
- Distributed state storage (Redis)
- Task scheduling coordination

### Worker
- Task execution
- Dependency resolution
- Result storage

### Driver
- DAG construction
- Load balancing
- Job orchestration

## Execution Flow

1. Driver builds task DAG
2. Master tracks task state
3. Workers execute tasks in parallel
4. Results stored in distributed state
5. Driver collects final output

## Data Flow

```
Data → Map Stage → Shuffle → Reduce Stage → Output
```

## Fault Tolerance

- Task status tracking in Redis
- Retry mechanism for failed tasks
- Checkpoint state for recovery
