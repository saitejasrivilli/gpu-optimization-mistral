# GPUForge

Production-oriented GPU cluster lifecycle & workload orchestrator. Manages pool of GPU workers + AI workloads: discovery, capability validation, lifecycle, GPU-aware scheduling, topology-aware placement, allocation/release/retry/cancel, draining/maintenance, metrics, simulated + real CUDA fleets.

## What this is NOT

Not a distributed storage system (see [Distributed Object Store] — replication/quorum/node-failure-recovery already covered there). Not a consensus/networking project (see LedgerDB — Raft, TCP fault injection). Not a telemetry platform (see Fleet Telemetry Monitor — ingestion/dashboards/diagnostics). GPUForge consumes telemetry, it does not produce a second monitoring stack.

This project's unique surface: **GPU resource modeling, GPU-aware scheduling policies, topology-aware placement, workload admission/queueing/preemption, worker lifecycle state machine.** Failure handling here exists only insofar as it affects scheduling/lifecycle correctness (e.g. a dead worker must stop receiving allocations) — it is not a general failure-detection research project.

## Docs

- [Architecture](docs/architecture.md)
- [Requirements](docs/requirements.md)
- [Failure model](docs/failure-model.md)
- [Scheduler](docs/scheduler.md)
- [Lifecycle](docs/lifecycle.md)
- [Benchmark plan](docs/benchmark-plan.md)
- [Domain model](docs/domain-model.md)
- [Worker agent](docs/agent.md)
- [Scheduling engine](docs/scheduling-engine.md)
- [Orchestration control loop](docs/orchestration.md)
- [Kubernetes execution backend](docs/kubernetes-execution.md)
- [ADRs](docs/decisions/)

## Status

Phase 5 — Kubernetes execution backend (`KubernetesExecutor`, one `batchv1.Job` per workload) implemented behind the same `Executor` interface as Phase 4's `SimulatedExecutor`.

## Architecture (through Phase 5)

```
Orchestrator (queue, retry, cancellation, draining)
    |
    | scheduler.Scheduler   (pure — FirstFit/BestFit/UtilizationAware/TopologyAware)
    | orchestrator.Executor (Start/Status/Cancel)
    |
    +-- SimulatedExecutor    (deterministic, in-memory)
    +-- KubernetesExecutor   (one Job per workload, real cluster)
```

## Example workload flow

```go
o := orchestrator.New(scheduler.FirstFit{}, k8sexec.New(client, "gpuforge", "gpuforge-workload:dev", nil), orchestrator.DefaultRetryPolicy)
o.RegisterWorker(readyWorker) // from internal/agent, Phase 2

o.Submit(domain.WorkloadRequirements{WorkloadID: "wl-1", GPUCount: 1}, now)
o.ScheduleNext(ctx, now)      // -> places, allocates, creates a Kubernetes Job
o.Tick(ctx, now)              // -> polls the Job, transitions on completion/failure
o.Cancel("wl-1", "user requested", now) // idempotent, deletes the Job
```

## Local Kubernetes setup

See docs/kubernetes-execution.md's "Local development" section for the full `kind`-based walkthrough (cluster creation, RBAC, building/loading the workload image, running the integration tests). Summary:

```sh
kind create cluster --name gpuforge
kubectl apply -f deploy/kubernetes/namespace.yaml -f deploy/kubernetes/rbac.yaml
docker build -t gpuforge-workload:dev . && kind load docker-image gpuforge-workload:dev --name gpuforge
GPUFORGE_K8S_INTEGRATION=1 go test -tags=integration ./internal/k8sexec/...
```

`go test ./...` (default, no `integration` tag) never requires Kubernetes or Docker.

## Stack

Go (core), Python/CUDA/PyTorch/NCCL (real-GPU validation + workload simulation), Docker (local sim fleet), Kubernetes/Terraform/Ansible (future), Prometheus (metrics consumption). No cloud credentials required locally.
