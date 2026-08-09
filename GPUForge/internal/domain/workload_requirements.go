package domain

// TopologyRequirement expresses what GPU connectivity a workload needs.
// See docs/scheduling-engine.md for the compatibility rules each value
// implies.
type TopologyRequirement string

const (
	// TopologyNone means the workload has no connectivity requirement.
	TopologyNone TopologyRequirement = "NONE"
	// TopologySameNode means all selected GPUs must be on the same worker.
	// In the current domain model an Allocation is always scoped to a
	// single Worker (see Allocation.WorkerID), so this requirement is
	// always satisfied by construction — it exists as an explicit,
	// checkable value now so a future multi-node Allocation model doesn't
	// have to invent it retroactively.
	TopologySameNode TopologyRequirement = "SAME_NODE"
	// TopologyNVLinkGroup means every selected GPU must share the same
	// non-empty GPUTopology.NVLinkGroup value. A single-GPU request
	// trivially satisfies this (nothing to link). Unknown topology
	// (empty NVLinkGroup) never satisfies this for GPUCount > 1 — see
	// docs/scheduling-engine.md's fallback-behavior section.
	TopologyNVLinkGroup TopologyRequirement = "NVLINK_GROUP"
)

// WorkloadRequirements is the scheduler's input alongside a
// ClusterSnapshot. Every field exists because a Phase 3 policy or
// compatibility check actually consumes it; see the per-field comments.
type WorkloadRequirements struct {
	// WorkloadID correlates a Placement (or a scheduling failure) back to
	// the workload that requested it.
	WorkloadID string

	// GPUCount is how many GPUs the workload needs simultaneously, all on
	// one worker (see TopologySameNode). Drives candidate subset size.
	GPUCount int

	// MinGPUMemoryBytes is the minimum capacity each *individual* selected
	// GPU must have. Requirements are per-GPU, not an aggregate sum,
	// because this domain model grants exclusive whole-GPU allocations
	// (see GPUAllocationState), not fractional/shared memory.
	MinGPUMemoryBytes uint64

	// CUDARequirement is the minimum CUDA compute capability the
	// workload's kernels require (e.g. "sm_80" or "8.0"). Empty means no
	// requirement. See docs/scheduling-engine.md for the comparison rules.
	CUDARequirement string

	// TopologyRequirement constrains which GPU combinations are
	// compatible at all (a hard gate), independent of which compatible
	// combination a policy then prefers.
	TopologyRequirement TopologyRequirement

	// Priority and Preemptible are not consumed by any Phase 3 policy —
	// no preemption/eviction logic exists yet. They are carried now
	// because Phase 0/2's design explicitly calls them out as required
	// workload-model fields for the scheduler's future preemption phase;
	// omitting them here would mean re-deriving them later against
	// already-placed workloads.
	Priority    int
	Preemptible bool

	// WorkloadType is a free-form classification (e.g. "training",
	// "inference", "batch"). Not consumed by policy logic in Phase 3;
	// carried for future workload-type-aware routing and for placement
	// explanations/metrics labeling.
	WorkloadType string
}

// Validate checks the structural minimum a WorkloadRequirements value must
// satisfy before a scheduler can reason about it at all. It does not check
// satisfiability against any cluster — that's the scheduler's job.
func (r WorkloadRequirements) Validate() error {
	if r.WorkloadID == "" {
		return &RequirementsError{Reason: "workload id required"}
	}
	if r.GPUCount < 1 {
		return &RequirementsError{Reason: "GPU count must be at least 1"}
	}
	switch r.TopologyRequirement {
	case TopologyNone, TopologySameNode, TopologyNVLinkGroup, "":
	default:
		return &RequirementsError{Reason: "unknown topology requirement " + string(r.TopologyRequirement)}
	}
	return nil
}
