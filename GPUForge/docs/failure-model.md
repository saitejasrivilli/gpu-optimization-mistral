# Failure model

Scope note: failure handling exists here only where it affects workload scheduling/lifecycle correctness. This is not a general failure-detection/recovery research project (that ground is covered by the Distributed Object Store's replication/quorum work and LedgerDB's consensus work).

## What GPUForge detects

- **Worker heartbeat loss**: controller heartbeats every registered worker on a fixed interval (default 2s). N consecutive missed beats (default 3) => worker transitions to `QUARANTINED`.
- **Agent-reported workload failure**: agent reports a workload process exited non-zero, or crashed.
- **Placement failure at execution time**: agent rejects a SCHEDULED workload (e.g. race where GPU became unavailable) => workload requeued, not silently dropped.

## What GPUForge does NOT do

- No chunk replication, no quorum reads/writes, no re-replication scans — there is no durable data being sharded across workers here. A GPU worker holds running compute, not durable object data.
- No gossip protocol, no distributed consensus for failure agreement — failure detection is centralized at the controller by design (tradeoff documented in ADR-002).
- No network partition simulation/fault injection framework — that's LedgerDB's job.

## Consequences of a worker failure

1. Worker marked `QUARANTINED` (see docs/lifecycle.md), immediately excluded from scheduling candidate sets.
2. Any workload the worker was running for transitions `RUNNING -> FAILED` with reason `worker-unreachable`.
3. Failed workload enters the workload's own retry policy (`FAILED -> RETRYING -> QUEUED`) if retry budget remains, else `CANCELLED`.
4. No re-replication step exists because there is nothing to re-replicate — the workload is simply rescheduled onto a different healthy worker from QUEUED.

## Failure of the controller itself

Single controller process is a documented single point of failure (ADR-002). Phase 0 does not attempt controller HA (no Raft-based leader election — that's LedgerDB's domain, and duplicating it here would violate the portfolio boundary). Acceptable for a local/simulated/portfolio-scope system; documented explicitly as a known limitation rather than hidden or half-solved.

## Bounded retry policy

Workload retries use bounded exponential backoff (default: 3 attempts, base 1s, factor 2, max 30s). Retry budget exhaustion is a terminal `CANCELLED`, always with a reason string, never a silent drop.

## Measuring failure impact

Any claim about "how fast a preempted/failed workload gets rescheduled" must come from an actual benchmark run under benchmark/results/ (see docs/benchmark-plan.md) — never estimated.
