# Scheduler

## Interface

```go
type Scheduler interface {
    // Place attempts to find a placement for a workload given current registry snapshot.
    // Returns a PlacementResult (success or explained rejection). Never mutates state.
    Place(ctx context.Context, req WorkloadRequirements, snapshot RegistrySnapshot) (PlacementResult, error)
}

type WorkloadRequirements struct {
    WorkloadID        string
    GPUCount          int
    MinGPUMemoryBytes uint64
    CUDARequirement   string // e.g. "sm_80+"
    TopologyRequirement TopologyRequirement // None | SameNode | NVLinkGroup
    Priority          int
    Preemptible       bool
    EstimatedDuration time.Duration
    WorkloadType      string // training | inference | batch
}

type PlacementResult struct {
    WorkloadID        string
    SelectedGPUs      []GPURef
    Policy            string
    Score             float64
    Reason            string
    RejectedAlternatives []RejectedAlternative
    Timestamp         time.Time
}

type RejectedAlternative struct {
    GPURef GPURef
    Reason string
}
```

Every placement decision — success or failure — is explainable: policy name, score/reason, and (where useful) the alternatives considered and why they lost.

## Initial policies

1. **First Fit** — scan workers in registration order, take the first GPU set satisfying requirements. Baseline; fastest, worst fragmentation.
2. **Best Fit** — among all satisfying GPU sets, pick the one with least leftover capacity (minimizes fragmentation).
3. **Topology Aware** — for multi-GPU workloads with `NVLinkGroup` requirement, restrict candidate sets to GPUs sharing an NVLink domain; falls back to same-node, then rejects if unsatisfiable.
4. **Utilization Aware** — prefer GPUs/workers with lowest recent utilization to spread load; used to reduce hotspotting under mixed inference/training load.

Policy selection is a controller config value, not compiled-in — swapping policy requires no code change to the controller, only config.

## Admission control

Before a workload reaches the queue, admission control performs a cheap satisfiability check against the full fleet capability (not current availability): if no worker in the fleet could ever satisfy the requirement (e.g. requests `sm_90` but fleet caps at `sm_80`), reject immediately as `FAILED` rather than queueing forever.

## Preemption

Workloads marked `Preemptible: true` may be evicted by the scheduler to make room for a higher-priority non-preemptible workload. Eviction: workload transitions RUNNING -> FAILED with reason `preempted`, then RETRYING per its retry policy. Preemption decisions are logged with the same explainability fields as placements (which workload preempted which, and why).

## Topology model

Minimum topology info needed for placement:

- node ID a GPU belongs to
- NVLink group ID (GPUs sharing a fast interconnect), if any
- inter-node network tier (for future multi-node topology-aware scheduling)

Simulated fleets generate topology deterministically from the seed (e.g. workers grouped into NVLink pairs/quads per config). Real fleets derive topology from `nvidia-smi topo -m` / NVML.

## Determinism & explainability requirements

- Same registry snapshot + same workload requirements + same policy => same PlacementResult, always (no randomness in policies; ties broken by stable GPU ID ordering).
- Rejections always carry a human-readable reason, never a bare `false`.
