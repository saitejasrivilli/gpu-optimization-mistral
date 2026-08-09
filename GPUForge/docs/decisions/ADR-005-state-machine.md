# ADR-005: State machine design

## Status
Accepted

## Context
Worker and workload lifecycles both need explicit, auditable state, with invalid transitions rejected rather than silently allowed.

## Decision
Two independent explicit state machines (worker: 10 states, workload: 8 states, see docs/lifecycle.md) enforced in code as a transition table (`map[fromState][toState]bool` plus a reason/source requirement), not as free-form string fields. Every transition emits an immutable record: entity ID, previous state, new state, reason, timestamp, source. Lifecycle-mutating RPCs are idempotent keyed on `(entity_id, requested_state, request_id)`.

## Tradeoffs
An explicit transition table is more upfront code than "just set a status field," but it is what makes invalid-state bugs (e.g. scheduling onto a QUARANTINED worker) a compile/test-time-catchable class of error rather than a runtime surprise. Cost: every new state or transition requires updating the table and its tests — accepted as a deliberate friction point, since lifecycle correctness is core project value, not incidental plumbing.
