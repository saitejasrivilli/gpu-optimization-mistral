# Requirements

## 1. Product scope

GPUForge manages a pool of GPU workers (simulated or real) and schedules AI workloads onto them. In scope:

- worker discovery + capability validation
- worker lifecycle management (provisioning through retirement)
- workload submission with resource requirements
- GPU-aware scheduling (multiple pluggable policies)
- topology-aware placement (NVLink/PCIe/multi-node awareness)
- allocation, release, retry, cancellation of workloads
- draining/maintenance workflows for workers
- REST/gRPC API for clients
- Prometheus metrics exposition (consumer of GPU/telemetry data, not a monitoring platform)
- simulated GPU fleets for local dev (8-16 simulated workers, deterministic seeds)
- real CUDA-backed worker support where hardware exists, with real measurements
- benchmarking of scheduling policies
- (future) controlled autonomous diagnosis/remediation

## 2. Non-goals

- Not a general-purpose distributed storage system. No replication/quorum design here — see Distributed Object Store project.
- Not a consensus system. No Raft/leader-election research here — see LedgerDB.
- Not a telemetry/monitoring platform. No new dashboard, no new metrics-storage backend — GPUForge consumes Prometheus data already produced elsewhere and exposes its own scheduling-specific counters only.
- Not a Kubernetes scheduler replacement (yet) — Phase-0-era design keeps a K8s integration path as future work, not a Phase-1 deliverable.
- Not multi-tenant billing/quota system.
- Not a general failure-detection research project — failure handling is scoped strictly to "must the scheduler treat this worker/workload differently."

## 3. Functional requirements

- FR1: discover workers and their GPU inventory (identity + capability)
- FR2: validate GPU capability (driver/runtime/CUDA compat) before marking a worker schedulable
- FR3: accept workload submissions with declared resource requirements (GPU count, memory, CUDA version, topology, priority, preemptibility, estimated duration, workload type)
- FR4: schedule queued workloads onto suitable GPUs via a pluggable policy
- FR5: support topology-aware placement decisions (co-locate multi-GPU workloads on NVLink-connected GPUs when required)
- FR6: allocate/release GPU resources; support retry of failed placements and cancellation of queued/running workloads
- FR7: support draining a worker (stop new allocations, let running workloads finish or be migrated) and maintenance mode
- FR8: detect worker failure (heartbeat loss) and stop scheduling onto it; do not attempt data-replication-style recovery — that's out of scope
- FR9: expose REST/gRPC APIs for submit/query/cancel workload and worker inspection
- FR10: expose Prometheus metrics for scheduling latency, queue depth, allocation success rate, utilization, fragmentation
- FR11: run against a simulated fleet with deterministic seed, or a real CUDA fleet, via the same scheduler code path
- FR12: benchmark each scheduling policy under reproducible simulated load, write results to files, never invent numbers

## 4. Non-functional requirements

- Scheduling latency must be measured (P50/P95/P99), not assumed.
- All state transitions auditable: worker ID/workload ID, prev state, new state, reason, timestamp, source.
- Deterministic simulation given a fixed seed (same seed → same schedule).
- No cloud credentials required for local development.
- Concurrency-safe under the Go race detector.
- Explicit, bounded timeouts on every RPC and external call.

## 5. Out-of-scope hardware assumption

Author has no guaranteed access to multiple physical GPUs. System must be fully exercisable via simulation; real-GPU code paths are additive and never required for demoing scheduling logic.
