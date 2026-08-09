# Domain model (Phase 1)

Package: `internal/domain`. Pure Go, no infrastructure dependencies (no HTTP/gRPC/K8s/Docker/CUDA/DB/Prometheus). Usable standalone from unit tests.

## Entities

- **Worker** — a GPU host. Holds `ID`, `HardwareMode`, current `WorkerState`, attached `GPU`s, and in-memory transition history. All state transitions go through `Worker.Transition`, the single authoritative mechanism.
- **GPU** — one device, composed of four separate concerns per docs/architecture.md: identity (`ID`, `WorkerID`, `Model`, `HardwareMode`), `GPUCapability` (static: compute capability, driver/runtime version, memory capacity), `GPUState` (runtime: utilization, available memory, last heartbeat), and `ValidationResult` (validation state). These are never merged into one blob.
- **Workload** — a submitted unit of work. Holds `ID`, current `WorkloadState`, and transition history. Deliberately has no resource-requirement fields yet — those belong to the scheduler's input type in a later phase, not to the lifecycle-only aggregate built here.
- **Allocation** — a workload's claim on a specific worker's specific GPUs. Records `WorkloadID`, `WorkerID`, `GPUIDs`, `CreatedAt`, `AllocationState` (ACTIVE/RELEASED), and release info. Never selects GPUs itself — only represents a claim that was already decided.
- **ValidationResult** — PENDING/PASSED/FAILED outcome of a (not-yet-implemented) capability check, with a mandatory reason on failure.
- **ClusterSnapshot** — an immutable, mutex-free read-only copy of all workers/GPUs at a point in time, matching the `Scheduler.Place(ctx, req, snapshot)` shape defined in docs/scheduler.md. Phase 1 only builds this type (`Worker.Snapshot`, `NewClusterSnapshot`); nothing consumes it yet.

## Relationships

```
Worker 1---* GPU
Workload 1---1 Allocation---1 Worker
Allocation *---* GPU (by ID, must belong to the Allocation's Worker)
Worker.Snapshot() -> WorkerSnapshot -> ClusterSnapshot
```

## Lifecycle boundaries

- Worker and Workload each own exactly one transition table (`workerTransitions`, `workloadTransitions` in worker.go/workload.go) — the sole source of truth for valid transitions. No other file re-implements or overrides these rules.
- Every transition requires a non-empty `reason`, a `TransitionSource`, and a timestamp; failed transitions never mutate state (verified by tests asserting state is unchanged after a rejected transition).
- Transition history lives in memory on the aggregate (`[]WorkerTransition` / `[]WorkloadTransition`) — no database required, per docs/lifecycle.md.

## Hardware-mode invariant

`HardwareMode` is a closed two-value enum (`simulated`, `real`). A GPU's mode is fixed at construction and checked against its owning worker's mode the only time a GPU becomes attached (`Worker.AddGPU`) — a simulated worker can never end up hosting a GPU that claims to be real, and vice versa. This is enforced structurally, not by convention; see `TestInvariant_SimulatedGPUCannotBeAddedToRealWorker` / `TestInvariant_RealGPUCannotBeAddedToSimulatedWorker`.

## Important invariants (see tests for exhaustive coverage)

- Every one of the 10x10 worker state pairs and 8x8 workload state pairs is checked against docs/lifecycle.md's transition tables (`worker_test.go`, `workload_test.go`).
- RETIRED worker cannot become READY; COMPLETED/CANCELLED workload cannot become RUNNING.
- Only READY/ALLOCATED workers are allocatable (`WorkerAllocatable`); QUARANTINED and RETIRED workers reject `NewAllocation`.
- An allocation cannot contain duplicate GPU IDs, and every GPU ID must belong to the target worker.
- A GPU's hardware mode must match its worker's hardware mode.
- A failed transition or a failed `Allocation.Release` leaves state completely unchanged (no partial mutation).
- All aggregates (`Worker`, `Workload`, `Allocation`) are safe for concurrent use (mutex-guarded; `-race` clean).

## Phase 2 addition

`Worker.UpdateGPUState` and `Worker.UpdateGPUValidation` (added in Phase 2, worker.go) are the only sanctioned way to mutate a GPU's runtime/validation state after attachment — both mutex-guarded, so the agent layer (`internal/agent`, see docs/agent.md) never reaches into a `*GPU`'s fields directly. This is additive; it does not change any transition table or invariant from Phase 1.

## Phase 3 additions

Phase 3 (the scheduler, docs/scheduling-engine.md) needed two pieces of GPU data the domain model didn't yet carry, both additive — no existing transition table, invariant, or exported behavior changed:

- **`GPU.AllocationState`** (`FREE`/`ALLOCATED`) — the fifth concept docs/architecture.md's data model always called out (identity/capability/runtime-state/validation-state/**allocation-state**) but that had no consumer until the scheduler needed to know which GPUs are actually selectable. Mutated only via `Worker.MarkGPUsAllocated`/`Worker.MarkGPUsReleased` (atomic: all-or-nothing across a GPU set, mutex-guarded). `Allocation.NewAllocation` now calls `MarkGPUsAllocated` and `Allocation.Release` calls `MarkGPUsReleased`, closing a real gap: before this, two `Allocation`s could have been constructed over the same GPU with nothing to stop them (`TestInvariant_CannotDoubleBookGPU` guards against regression).
- **`GPU.Topology`** (`GPUTopology{NodeID, NVLinkGroup}`) — static, discovered-only placement data consumed by topology-aware scheduling. Empty means unknown; never invented. Populated by the agent layer (`internal/agent`) at discovery time, same as capability.

`ClusterSnapshot`/`GPUSnapshot` (snapshot.go) now carry both fields, since the scheduler operates on snapshots, not live `*Worker`/`*GPU`.

`domain.WorkloadRequirements` (workload_requirements.go) is new: the scheduler's second input alongside `ClusterSnapshot`. See docs/scheduling-engine.md for field-by-field rationale.

## Phase 4 note

No domain package changes were needed for Phase 4 (the orchestration control loop, `internal/orchestrator`, see docs/orchestration.md). The existing `Worker`/`Workload`/`Allocation` transition tables and `MarkGPUsAllocated`/`MarkGPUsReleased` atomicity were sufficient as-is; cancellation of a `RETRYING` or `SCHEDULED` workload is expressed as two valid transitions in sequence rather than requiring a new direct edge — see docs/orchestration.md's cancellation section for the reasoning. The single-worker-per-`Allocation` constraint (noted in the Phase 3 section above) was inspected again for Phase 4 and still required no change.

## Phase 5 note

No domain package changes were needed for Phase 5 either (the Kubernetes execution backend, `internal/k8sexec`, see docs/kubernetes-execution.md). `KubernetesExecutor` implements `orchestrator.Executor` using only `orchestrator.ExecutionRequest`/`ExecutionStatus` — it never imports `internal/domain`, and `internal/domain` never imports any Kubernetes package.

## Deferred to later phases

gRPC/HTTP transport, Prometheus metrics, Terraform/Ansible integration, distributed consensus, telemetry storage, autonomous agents, preemption/eviction logic (fields exist on `WorkloadRequirements`, no behavior yet), multi-node/multi-worker allocations, automatic worker health monitoring — none of this package's business yet.
