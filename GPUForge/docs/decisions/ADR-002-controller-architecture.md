# ADR-002: Controller architecture

## Status
Accepted

## Context
Need to decide whether the controller is a single centralized process or a replicated/HA control plane from day one.

## Decision
Single centralized controller process for the scope of this project. It owns the registry, scheduler, lifecycle state machines, health monitor, and metrics in one process (internally modular, not split into microservices).

## Tradeoffs
Single point of failure, honestly: if the controller dies, no new scheduling happens and heartbeat monitoring stops until it restarts. This is an explicit, accepted tradeoff — building controller HA would mean re-deriving Raft-style leader election and replicated state, which is exactly LedgerDB's domain and would duplicate that project rather than add value here. Premature microservice-splitting of the controller (separate scheduler service, separate lifecycle service) was also rejected: it adds RPC hops and operational complexity with no scheduling-quality benefit at this scale (8-16 simulated workers). Revisit only if this ever needs to run at a scale where controller restart time becomes materially costly.
