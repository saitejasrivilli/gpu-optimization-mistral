# ADR-003: Worker simulation

## Status
Accepted

## Context
Author has no guaranteed access to a multi-GPU cluster. System must be developable, testable, and benchmarkable entirely on a single machine.

## Decision
GPU agents implement a common interface; a simulated agent implementation generates deterministic synthetic GPU state (identity, capability, utilization, topology, availability, validation status) from a configurable seed. Simulated agents run as regular Docker Compose containers/processes, same gRPC contract as real agents. A worker's registry record always carries `hardware_mode: simulated|real`, surfaced in every API response and metrics label.

## Tradeoffs
Determinism (fixed seed => identical fleet state => reproducible scheduling benchmarks) is prioritized over "realism knobs" — the simulator models utilization/memory pressure with simple, documented distributions rather than trying to emulate real workload traces precisely, since the target here is scheduler-logic validation, not GPU-performance modeling. Explicit risk: simulated numbers must never be presented as real hardware results — enforced by the `hardware_mode` field being mandatory and checked before any benchmark result is written to `benchmark/results/`.
