# Orchestration control loop (Phase 4)

Package: `internal/orchestrator`. Depends on `internal/domain` and `internal/scheduler` only. No HTTP/gRPC/Kubernetes/Docker/Prometheus — this is the in-memory control plane, not an infrastructure integration (those are later phases).

```
Workload Queue -> Scheduler.Place (pure) -> Allocation (atomic) -> Executor -> workload state -> release
```

`scheduler.Scheduler` stays pure exactly as Phase 3 left it — the orchestrator is the only place side effects happen (calling `Place`, constructing `Allocation`s, calling the executor, driving `domain.Workload`/`domain.Worker` transitions).

## Queue

`Queue` (queue.go) is in-memory, priority-ordered (higher `WorkloadRequirements.Priority` first), FIFO within equal priority (`EnqueuedAt`, then `WorkloadID` for full determinism), mutex-guarded. `Dequeue` only ever removes what it returns — no workload is silently dropped by the queue itself. Deliberately not distributed: a single-process control plane doesn't need a replicated queue, and building one would duplicate LedgerDB's consensus work.

## Executor

```go
type Executor interface {
    Start(ctx, ExecutionRequest, now) error
    Status(ctx, workloadID, now) (ExecutionStatus, error)
    Cancel(ctx, workloadID, now) error
}
```

Keyed by `WorkloadID` (there's always exactly one execution per workload — no need for a separate handle type). `Start` is **not** implicitly idempotent: a second `Start` for the same workload returns `ErrAlreadyStarted`, because a real executor can't safely restart an already-running process either — the orchestrator's own state machine (only calling `Start` once per `SCHEDULED` transition) is what actually prevents duplicates; the executor's rejection is defense in depth, not the primary guard.

`SimulatedExecutor` (simulated_executor.go) resolves deterministically: each workload's outcome is configured in advance via `Plan(workloadID, ExecutionPlan{Outcome, FailureReason, Retryable, Delay})`. `Status` takes `now` explicitly and only resolves a delayed plan once `now >= startedAt+Delay` — no sleeps anywhere, tests advance a `time.Time` value instead. It never claims to be real GPU/CUDA/NCCL execution; it exists solely to drive the orchestration state machine deterministically.

## Allocation (atomicity)

The orchestrator reuses Phase 1/3's domain mechanisms unchanged: `domain.NewAllocation` calls `Worker.MarkGPUsAllocated`, which is atomic under the worker's own mutex — either every requested GPU flips to `ALLOCATED` or none do. Two `ScheduleNext` calls racing for the same GPU (see Concurrency below) will have one succeed and one get a `*domain.AllocationError` back, which the orchestrator treats as ordinary scheduling contention (see Retry) rather than a bug.

## Control loop

`ScheduleNext(ctx, now)`: dequeue -> verify the workload is still `QUEUED` (else drop as stale, see Failure handling) -> build a fresh `domain.ClusterSnapshot` -> `scheduler.Place` -> `domain.NewAllocation` -> transition `QUEUED -> SCHEDULED` -> `executor.Start` -> transition `SCHEDULED -> RUNNING`. Any failure along the way releases whatever was allocated and leaves the workload in a valid, documented state (usually back in the queue, occasionally `CANCELLED` — never a partial allocation).

`Tick(ctx, now)`: the only place time-driven bookkeeping happens — promotes `RETRYING` workloads whose backoff has elapsed back to `QUEUED`, and polls `executor.Status` for every `RUNNING` workload, applying the resulting transition and releasing the allocation on any terminal outcome. There is no background goroutine; callers drive `Tick` (and `now`) explicitly, matching Phase 4's "avoid sleeps, use deterministic clocks" requirement.

## Concurrency model

One coarse `sync.Mutex` guards every state-mutating `Orchestrator` method (`Submit`, `ScheduleNext`, `Tick`, `Cancel`, `DrainWorker`, `CompleteDraining`). This is a deliberate simplicity choice, not an oversight: GPUForge's control plane is explicitly single-process and in-memory (per the portfolio boundary — a distributed/replicated control plane is LedgerDB's problem, not this project's), so there is no correctness reason to shard the lock, and doing so would only add risk of subtle ordering bugs for no throughput benefit at this scope. Real concurrency is still exercised and meaningful in tests: goroutines genuinely race to *acquire* the lock, and domain-level atomicity (`MarkGPUsAllocated`) still matters as defense in depth for the moment the lock is briefly released between a snapshot read and an allocation attempt in two back-to-back `ScheduleNext` calls.

Tested under `-race` (`TestConcurrentScheduling_NoDoubleAllocation`, `TestConcurrentSchedulingAcrossWorkers`, `TestCancelDuringConcurrentScheduling`, `TestCompletionDuringCancellation`, `TestRetryDuringConcurrentScheduling`): no double allocation, no corrupted state, every observed outcome is one of the well-defined states the state machine allows.

## Retry

One `RetryPolicy{MaxAttempts, BaseDelay, Factor, MaxDelay}` (retry.go) governs two distinct situations with the same shape ("try again later, up to a limit"):

1. **Scheduling failure** (no compatible GPU, or lost a race for one) — tracked as `QueueItem.Attempts`, incremented in `requeueOrGiveUp`. Since `domain.WorkloadQueued`'s only valid transitions are to `SCHEDULED` or `CANCELLED` (docs/lifecycle.md — there is no `QUEUED -> FAILED` edge), "permanently unschedulable" is represented as `CANCELLED`, not `FAILED`. This is a domain constraint, not an orchestrator shortcut, and it's the correct one: a workload that never got a GPU never failed at anything.
2. **Execution failure** (`ExecutionStatus.Retryable == true`) — tracked in `Orchestrator.attempts`, checked in `Tick`. `RUNNING -> FAILED -> RETRYING -> QUEUED` per docs/lifecycle.md; `RETRYING`'s backoff delay is computed once (`RetryPolicy.NextDelay`, deterministic exponential backoff, no jitter) and stored as an absolute `NextAttemptAt`, checked by `Tick` — never a real sleep.

A retry always releases the failed allocation first (`Tick`'s `releaseAllocation` call happens before the `RETRYING` transition) and always re-enters through the queue rather than re-attempting the same worker directly, so a retried workload can never double-allocate the GPU it just gave up.

## Cancellation

Idempotent by design. Per-state behavior:

| State | Behavior |
|---|---|
| `QUEUED` | Removed from queue, `-> CANCELLED` |
| `RUNNING` | `executor.Cancel` called, allocation released, `-> CANCELLED` |
| `FAILED` | `-> CANCELLED` directly (valid domain edge) |
| `RETRYING` | Backoff cancelled, routed `RETRYING -> QUEUED -> CANCELLED` (two valid hops — see below) |
| `SCHEDULED` | Routed `SCHEDULED -> QUEUED -> CANCELLED` (two valid hops; see below — not externally reachable in practice) |
| `COMPLETED` / `CANCELLED` | No-op, returns `nil` — idempotent success, not an error |
| `SUBMITTED` | `ErrNotCancellable` — transient state, never observed externally since `Submit` synchronously advances to `QUEUED` |

**Why `RETRYING`/`SCHEDULED` route through an intermediate `QUEUED` hop instead of a direct edge:** docs/lifecycle.md's transition table (from Phase 1, before any scheduler or orchestrator existed) has no `RETRYING -> CANCELLED` or `SCHEDULED -> CANCELLED` edge — only `RETRYING -> QUEUED` and `SCHEDULED -> {RUNNING, QUEUED}`. Rather than treating this as a domain bug needing a Phase 1 redesign, the orchestrator drives two valid transitions in sequence (`RETRYING -> QUEUED` then `QUEUED -> CANCELLED`), both recorded in the workload's transition history. This was inspected deliberately (per Phase 4's instruction to check before redesigning): the existing table is internally consistent and sufficient once cancellation is allowed to take an extra, still-fully-valid hop. No domain change was needed or made.

**Why `SCHEDULED` is not actually reachable from `Cancel` in practice:** `ScheduleNext` holds the orchestrator's single mutex for its entire allocate-then-start sequence, so no `Cancel` call can observe a workload sitting in `SCHEDULED` — by the time `Cancel` can acquire the lock, `ScheduleNext` has already moved it to `RUNNING` or back to `QUEUED`. The `SCHEDULED` case in `Cancel` exists purely as defense in depth against a future change to the locking model, not because it's exercised today.

## Worker draining

`DrainWorker` transitions a worker to `DRAINING` via the existing domain state machine — no scheduler change was needed, because `scheduler.eligibleWorkers` already gates on `domain.WorkerAllocatable` (`READY`/`ALLOCATED` only), so a `DRAINING` worker is automatically excluded from every policy's candidate set.

**Explicit limitation:** workloads already running on a draining worker are left to run to completion. GPUForge does not implement workload migration or checkpointing — there is no mechanism to move a running workload off a draining worker onto another one. `CompleteDraining` (worker `DRAINING -> MAINTENANCE`) only succeeds once no active allocation references that worker, so an operator always gets an explicit, checkable signal (`ErrDrainIncomplete`) rather than a silent partial drain.

## Failure handling

| Situation | Handling |
|---|---|
| No compatible GPU | `requeueOrGiveUp`: requeue with incremented attempt count, or `CANCELLED` once `RetryPolicy.MaxAttempts` is exhausted |
| Executor start failure | Allocation released, `SCHEDULED -> QUEUED`, requeued with incremented attempt count |
| Executor runtime failure | `RUNNING -> FAILED`, allocation released; `RETRYING` if `Retryable` and budget remains, else `CANCELLED` |
| Cancellation | See Cancellation above |
| Allocation failure (lost a race) | Treated identically to "no compatible GPU" — requeued, not an error condition |
| Release failure | `domain.ErrAlreadyReleased` is treated as a benign no-op (see Idempotency) |
| Duplicate start | `Executor.ErrAlreadyStarted`; the state machine itself prevents this from being reachable in normal operation |
| Duplicate cancellation | Idempotent no-op on terminal states (see Cancellation) |
| Stale workload state | `ScheduleNext` checks `wl.State() == QUEUED` after dequeuing; a mismatch (e.g. cancelled out-of-band without going through `Cancel`) drops the item with `ErrStaleWorkload` instead of acting on it |
| Worker unavailable before execution | Two windows, both covered: (a) between snapshot and allocation — covered by atomic `MarkGPUsAllocated` + requeue-on-conflict; (b) between allocation and execution start — covered by "executor start failure" above (a real executor would fail `Start` against an unreachable worker) |

Per Phase 4's explicit scope, there is no automatic worker health monitoring here — heartbeat-driven quarantine (Phase 2's `agent.RunValidation`/heartbeat abstraction) is a separate, earlier mechanism this phase consumes but does not extend.

## Idempotency

| Operation | Twice |
|---|---|
| `Start` | Rejected (`ErrAlreadyStarted`) |
| `Cancel` | Second call is a no-op on a terminal state |
| `Release` | Second call rejected (`domain.ErrAlreadyReleased`); orchestrator's `releaseAllocation` swallows that specific error rather than treating it as fatal |
| `Complete` (executor Succeeded observed twice) | `Tick` only acts on workloads still tracked as `running`; once completed, the workload is removed from that set, so a repeat `Status` observation (if it somehow occurred) would be a no-op |

## Single-worker limitation (inspected, not changed)

Phase 3 already documented (docs/scheduling-engine.md) that `domain.Allocation.WorkerID` is a single string, not per-GPU — every `Allocation`, and therefore every `Placement`, is scoped to exactly one worker. Phase 4's orchestration loop was checked against this constraint specifically, per this phase's instruction to inspect how deeply it's embedded before touching it:

- `ScheduleNext` calls `domain.NewAllocation(wl, worker, placement.GPUIDs, now)` with a single `*domain.Worker` — there is no code path anywhere in the orchestrator that could construct a multi-worker allocation even by accident.
- `ExecutionRequest` carries one `WorkerID` and a `GPUIDs` list assumed to belong to that worker.
- `CompleteDraining` checks `alloc.WorkerID == workerID`, a single-worker-scoped check throughout.

No orchestration behavior in this phase required multi-worker placement, so no domain change was made and no ADR is being proposed here. If a future phase needs multi-node workloads (matching docs/architecture.md's already-documented future integration point), that will require a genuine `Allocation`/`Placement` schema change (per-GPU worker references instead of one `WorkerID`) and should go through an ADR at that time — not retrofitted silently into this phase.
