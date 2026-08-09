# Architecture

## 1. System overview

```
                 +-------------------+
   clients ----->|   REST API (api/) |
                 +---------+---------+
                           |
                           v  gRPC
                 +-------------------+
                 |    Controller     |
                 | (coordinator proc)|
                 |-------------------|
                 | - Registry        |  worker identity/capability/state
                 | - Scheduler       |  pluggable placement policies
                 | - Lifecycle mgr   |  worker + workload state machines
                 | - Health monitor  |  heartbeat -> failure detection
                 | - Metrics         |  Prometheus exposition
                 +----+----+----+----+
                      |    |    |
                gRPC heartbeat + control
                      |    |    |
            +---------+  +-+--+  +---------+
            | GPU Agent |  ...  | GPU Agent |   (one per worker; simulated or real)
            +-----------+       +-----------+
            | simulated GPU set  |  real CUDA device set (nvidia-smi / NVML)
            +---------------------------------+
```

Single controller process (coordinator), N GPU-agent processes (one per worker, simulated or backed by real hardware). This mirrors a control-plane/data-plane split without inventing a second consensus or storage system — controller state is in-memory + periodically snapshotted, not a replicated store (that's explicitly the Distributed Object Store's job, not this one's).

## 2. Controller responsibilities

- own the worker registry (identity, capability, current state, allocation state, validation state — kept as separate structs, see docs data model below)
- run the scheduler against the queue of pending workloads
- run the workload/worker lifecycle state machines and emit transition records
- heartbeat every registered worker on a fixed interval; mark dead after N missed beats
- expose gRPC (agent-facing + client-facing) and REST (client-facing) APIs
- expose Prometheus metrics

The controller is a single logical component for Phase 0-era scope. It is explicitly a single point of failure — documented honestly in ADR-002, not hidden.

## 3. GPU agent responsibilities

- report GPU inventory + capability at startup (identity + capability)
- respond to heartbeat RPCs
- execute allocate/release chunk-of-work operations on request (start/stop a workload process, report exit status)
- (simulated agent) generate synthetic GPU state (utilization, memory pressure) deterministically from a seed
- (real agent) query actual hardware via NVML/nvidia-smi/CUDA runtime; never fabricate a number when reporting real-mode stats

## 4. Scheduler responsibilities

Pure function-ish component: given current registry state + a workload's requirements, produce a placement decision (or a rejection with reason). Pluggable via an interface (see docs/scheduler.md). No side effects beyond producing a `PlacementResult`; the controller applies the result.

## 5. API boundaries

- **Client-facing REST/gRPC**: submit workload, get workload status, cancel workload, list workers, drain worker, view scheduling explanation for a placement.
- **Agent-facing gRPC**: register worker, heartbeat, report GPU state, execute allocate/release.
- No client ever talks to a GPU agent directly — controller is the sole mediator, keeping the trust boundary simple (relevant to security considerations below).

## 6. Data model (see docs/lifecycle.md and scheduler.md for full detail)

Kept deliberately separated per the product spec:

- **GPU identity**: worker ID, GPU ID, GPU model, serial/UUID
- **GPU capability**: CUDA compute capability, driver version, runtime version, memory capacity (static)
- **GPU current state**: utilization, available memory, temperature (if real), last-heartbeat time
- **GPU allocation state**: which workload(s) currently hold this GPU, reserved memory
- **GPU validation state**: pass/fail of capability checks, last validated timestamp

These are separate structs/tables, never merged into one "GPU status" blob — avoids the mixing the spec explicitly warns against.

## 7. Simulation vs real boundary

Both simulated and real GPU agents implement the same `GPUAgent` gRPC-shaped interface. Controller code has zero branching on "is this simulated" — the distinction lives entirely in the agent binary/config. A worker's registry record carries an explicit `hardware_mode: simulated|real` field, always surfaced in API responses and metrics labels, so simulated results are never presentable as real ones.

## 8. Failure model

See docs/failure-model.md. Summary: heartbeat-based detection at the controller; failure only affects scheduling eligibility and running-workload disposition (fail/retry/migrate), not data durability.

## 9. Future integration points

- Kubernetes: GPU agent becomes a device-plugin-aware DaemonSet; controller's scheduler interface is designed so a K8s scheduler-extender shim could call it without a rewrite.
- Terraform/Ansible: real-worker provisioning (Phase >5) — infra-as-code drives GPU agent installation onto real hosts; out of scope until real-hardware phases.
- Autonomous diagnosis/remediation: a future controller module that watches metrics + lifecycle transitions and proposes (initially human-approved, later bounded-autonomy) remediation actions like drain-and-replace. Explicitly deferred; not designed in Phase 0 beyond this note.

## 10. Security considerations

- Agent-to-controller and client-to-controller channels: mutual auth deferred to a later phase, documented as a known gap (no cloud IAM available locally) — Phase 0 assumes a trusted local network; must be revisited before any non-local deployment.
- Controller is the only component with the authority to allocate; agents never accept direct client connections, shrinking the attack surface.
- No secrets/credentials required for local dev — simulation mode has no external dependencies.
