# GPUForge — agent instructions

Infra software, not ML demo. Go primary lang. Phased build — do not skip ahead, do not implement production code outside current phase.

## Boundary (do not duplicate sibling projects)

- Distributed Object Store owns: replication, quorum, node failure recovery, health monitoring patterns.
- LedgerDB owns: Raft, consensus, TCP fault injection, durable storage.
- Fleet Telemetry Monitor owns: telemetry ingestion, dashboards, diagnostics UI.

GPUForge owns: GPU resource model, GPU-aware scheduling, topology-aware placement, admission control, worker/workload lifecycle, fragmentation/utilization optimization, preemption. Telemetry is consumed here, not re-implemented.

## Engineering rules

- explicit interfaces, context.Context, bounded timeouts, structured errors/logging, explicit state machines, idempotent lifecycle ops, safe bounded-backoff retries, concurrency-safe state, race detector on, unit+integration+e2e tests, real benchmarks only.
- no global mutable state, no giant files/functions, no premature abstraction/microservices, no magic constants, no hidden retries, no swallowed errors, no fake benchmark numbers, no TODO-driven architecture.
- every number in README/docs traces to a file under benchmark/results/ — never estimated.
- simulated hardware never represented as real hardware.
- local dev never requires cloud credentials.

## Phase discipline

Each phase: build on prior phase, explicit acceptance criteria, tests, doc updates, format/lint/test run, diff inspection, one focused commit. Do not start next phase until current passes.
