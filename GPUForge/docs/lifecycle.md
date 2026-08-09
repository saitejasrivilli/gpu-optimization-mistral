# Lifecycle

## Worker lifecycle

States: `PROVISIONING`, `DISCOVERING`, `VALIDATING`, `READY`, `ALLOCATED`, `DRAINING`, `MAINTENANCE`, `QUARANTINED`, `RETIRING`, `RETIRED`.

### Valid transitions

| From | To | Trigger |
|---|---|---|
| PROVISIONING | DISCOVERING | agent process started, registered with controller |
| DISCOVERING | VALIDATING | GPU inventory reported |
| DISCOVERING | QUARANTINED | discovery failed / malformed inventory |
| VALIDATING | READY | capability checks passed |
| VALIDATING | QUARANTINED | capability checks failed (bad driver/CUDA mismatch) |
| READY | ALLOCATED | scheduler placed >=1 workload on this worker |
| ALLOCATED | READY | all workloads on worker completed/released |
| READY | DRAINING | operator/health-triggered drain request |
| ALLOCATED | DRAINING | operator/health-triggered drain request (running workloads finish, no new allocations) |
| DRAINING | MAINTENANCE | drain completed, operator takes worker offline |
| DRAINING | READY | drain cancelled before completion |
| MAINTENANCE | DISCOVERING | operator returns worker to service (re-run discovery/validation) |
| READY, ALLOCATED, VALIDATING | QUARANTINED | heartbeat failure threshold exceeded |
| QUARANTINED | DISCOVERING | operator-initiated recovery attempt |
| QUARANTINED | RETIRING | recovery abandoned |
| MAINTENANCE | RETIRING | operator decommission decision |
| RETIRING | RETIRED | all references removed, safe to delete record |

### Invalid transitions (must be rejected by the state machine)

- PROVISIONING -> READY (must pass discovery+validation)
- ALLOCATED -> RETIRED (must drain first — running workloads cannot be dropped silently)
- RETIRED -> anything (terminal)
- QUARANTINED -> ALLOCATED (must re-validate first)

### Transition record (every transition emits one)

```
worker_id, previous_state, new_state, reason, timestamp, source
```

`source` is one of: `agent-report`, `health-monitor`, `operator`, `scheduler`.

## Workload lifecycle

States: `SUBMITTED`, `QUEUED`, `SCHEDULED`, `RUNNING`, `COMPLETED`, `FAILED`, `RETRYING`, `CANCELLED`.

### Valid transitions

| From | To | Trigger |
|---|---|---|
| SUBMITTED | QUEUED | passed admission control (well-formed requirements) |
| SUBMITTED | FAILED | admission control rejected (unsatisfiable requirements) |
| QUEUED | SCHEDULED | scheduler found a placement |
| QUEUED | CANCELLED | client cancel request while waiting |
| SCHEDULED | RUNNING | agent confirmed workload start |
| SCHEDULED | QUEUED | placement failed at execution time (agent rejected) — requeue |
| RUNNING | COMPLETED | agent reported successful exit |
| RUNNING | FAILED | agent reported error exit, or worker died mid-run |
| RUNNING | CANCELLED | client cancel request while running |
| FAILED | RETRYING | retry policy permits another attempt (bounded, backoff) |
| RETRYING | QUEUED | retry attempt re-enters the queue |
| FAILED | CANCELLED | retry budget exhausted, no further attempts |

Terminal states: `COMPLETED`, `CANCELLED`, and `FAILED` once retry budget is exhausted.

### Transition record

```
workload_id, previous_state, new_state, reason, timestamp, source
```

`source` is one of: `admission-control`, `scheduler`, `agent-report`, `health-monitor`, `client`.

## Idempotency

Every lifecycle-mutating RPC (allocate, release, cancel, drain) is idempotent keyed on `(entity_id, requested_state, request_id)` — replaying the same request must not double-apply a transition.
