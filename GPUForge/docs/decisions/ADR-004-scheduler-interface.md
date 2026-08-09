# ADR-004: Scheduler interface

## Status
Accepted

## Context
Need multiple scheduling policies (First Fit, Best Fit, Topology Aware, Utilization Aware) comparable via benchmarks, swappable without controller code changes, and each decision explainable.

## Decision
A single `Scheduler` interface (`Place(ctx, requirements, snapshot) (PlacementResult, error)`) that is pure with respect to controller state — it reads a registry snapshot and returns a decision, never mutates state itself. Policy is chosen by controller config, not compile-time wiring. Every `PlacementResult` carries policy name, score/reason, and rejected alternatives.

## Tradeoffs
Purity (no side effects in `Place`) makes policies trivially unit-testable and benchmarkable in isolation, at the cost of the controller needing a snapshot-then-apply two-step for every placement (slightly more code than a scheduler that mutates state directly). Accepted because it also makes scheduling fully deterministic and replayable, which the benchmark plan depends on. Explainability fields add a small overhead per decision but are required by the product spec and are load-bearing for the benchmark plan's "topology-aware improvement" metric.
