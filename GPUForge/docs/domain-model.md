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

## Deferred to later phases

Scheduling algorithms, gRPC/HTTP transport, Prometheus metrics, Kubernetes/Terraform/Ansible integration, distributed consensus, telemetry storage, autonomous agents — none of this package's business. GPU discovery/capability/state collection and CUDA-adjacent probing (via nvidia-smi) moved from "deferred" to "implemented" in Phase 2 — see docs/agent.md.
