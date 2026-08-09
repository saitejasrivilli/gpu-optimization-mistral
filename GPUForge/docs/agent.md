# Worker agent (Phase 2)

Package: `internal/agent`. Depends on `internal/domain` only (plus stdlib `os/exec` for the real agent). No HTTP/gRPC/K8s/Prometheus — those are later-phase transport/consumption concerns, not part of the hardware boundary itself.

## The contract

```go
type WorkerAgent interface {
    HardwareMode() domain.HardwareMode
    Discover(ctx) (DiscoveryResult, error)
    CollectState(ctx) ([]StateSample, error)
    Validate(ctx) ([]ValidationSample, error)
    Heartbeat(ctx) (HeartbeatResult, error)
}
```

One interface, two implementations, matching the boundary in docs/architecture.md:

```
Controller
    |
    | WorkerAgent interface
    |
+---+------------------+
|                      |
SimulatedAgent      NVIDIAAgent
|                      |
seeded RNG model    nvidia-smi (shelled out)
```

Controller code (a later phase) never branches on which implementation it holds.

## SimulatedAgent

Deterministic: constructed from a `SimulatedConfig{WorkerID, Seed, GPUs []GPUSpec, ...}`. `Discover` is pure (doesn't touch the RNG) and always returns the same result for a given config. `CollectState` draws from a `math/rand` source seeded once at construction — two agents built from identical configs produce an identical sequence of samples call-for-call, which is what benchmark reproducibility (docs/benchmark-plan.md) depends on. `FailValidationReason` and `SimulateUnreachable` are test/demo knobs for exercising the QUARANTINED and heartbeat-failure paths without needing a health monitor yet.

Always reports `domain.HardwareModeSimulated`.

## NVIDIAAgent

Shells out to `nvidia-smi --query-gpu=...` (via an injectable `CommandRunner`, so unit tests don't need real hardware). Parses CSV for identity/capability (`Discover`) and utilization/free-memory (`CollectState`). `Validate` is a presence/liveness check against nvidia-smi — real CUDA/NCCL correctness testing is out of scope for this phase, per Phase 2's explicit exclusion list. `RuntimeVersion` in `GPUCapability` is left empty: `nvidia-smi --query-gpu` doesn't expose it; not guessed.

Always reports `domain.HardwareModeReal`. The one integration test that touches a real binary (`TestNVIDIAAgent_RealHardware_Integration`) skips outright when `nvidia-smi` isn't on `PATH`, rather than faking a result — a simulated outcome must never stand in for a real-hardware assertion (ADR-003).

## Lifecycle wiring

`internal/agent/lifecycle.go` is the sole bridge from the agent boundary to the domain lifecycle:

- `Register(ctx, agent, now)` — calls `Discover`, constructs a `domain.Worker` + its `domain.GPU`s, transitions PROVISIONING -> DISCOVERING.
- `RunValidation(ctx, worker, agent, now)` — transitions DISCOVERING -> VALIDATING, calls `Validate`, applies each result via `Worker.UpdateGPUValidation`, then transitions to READY (all passed) or QUARANTINED (first failure's reason attached to the transition record).
- `CollectAndApplyState(ctx, worker, agent)` — calls `CollectState`, applies via `Worker.UpdateGPUState`. Never triggers a lifecycle transition on its own; runtime-state drift alone doesn't move a worker between states (only validation/heartbeat outcomes do, per docs/lifecycle.md).

All three functions return errors rather than swallowing them; a failed `Validate` call itself (not a failed validation *result*) also quarantines the worker, since an unreachable/broken validation path is itself a reason not to trust the worker.

## Deferred to later phases

Scheduler algorithms and workload placement, HTTP/gRPC transport, Kubernetes/Terraform/Ansible, Prometheus/Grafana, consensus/replication, telemetry storage, autonomous agents, and the consecutive-miss-count health-monitor logic that turns a `HeartbeatResult` into a QUARANTINED transition (Phase 2 only defines the heartbeat *abstraction*, not that policy).
