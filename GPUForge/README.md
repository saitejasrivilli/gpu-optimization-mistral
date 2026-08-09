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
- [ADRs](docs/decisions/)

## Status

Phase 3 — GPU-aware scheduling engine (FirstFit, BestFit, UtilizationAware, TopologyAware) implemented on top of the Phase 1 domain kernel and Phase 2 worker agent boundary.

## Stack

Go (core), Python/CUDA/PyTorch/NCCL (real-GPU validation + workload simulation), Docker (local sim fleet), Kubernetes/Terraform/Ansible (future), Prometheus (metrics consumption). No cloud credentials required locally.
