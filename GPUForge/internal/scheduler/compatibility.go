package scheduler

import (
	"fmt"
	"sort"
	"strconv"
	"strings"

	"gpuforge/internal/domain"
)

// WorkerCandidate is a worker that passed the worker-level gate (allocatable
// state) together with the subset of its GPUs that individually satisfy
// req, sorted by GPU ID for deterministic downstream selection.
type WorkerCandidate struct {
	Worker   domain.WorkerSnapshot
	Eligible []domain.GPUSnapshot
}

// eligibleWorkers evaluates compatibility only — never ranking. It returns,
// in deterministic WorkerID order, every worker with at least req.GPUCount
// individually-eligible GPUs (and, if req requires an NVLink group, at
// least one group with enough members), plus every rejection reason found
// along the way, worker- and GPU-level alike.
func eligibleWorkers(req domain.WorkloadRequirements, snapshot domain.ClusterSnapshot) ([]WorkerCandidate, []RejectedAlternative) {
	workers := make([]domain.WorkerSnapshot, len(snapshot.Workers))
	copy(workers, snapshot.Workers)
	sort.Slice(workers, func(i, j int) bool { return workers[i].ID < workers[j].ID })

	var candidates []WorkerCandidate
	var rejected []RejectedAlternative

	for _, w := range workers {
		if !domain.WorkerAllocatable(w.State) {
			rejected = append(rejected, RejectedAlternative{
				WorkerID: w.ID,
				Reason:   fmt.Sprintf("worker not allocatable in state %s", w.State),
			})
			continue
		}

		gpus := make([]domain.GPUSnapshot, len(w.GPUs))
		copy(gpus, w.GPUs)
		sort.Slice(gpus, func(i, j int) bool { return gpus[i].ID < gpus[j].ID })

		var eligible []domain.GPUSnapshot
		for _, g := range gpus {
			if reason := ineligibleReason(req, g); reason != "" {
				rejected = append(rejected, RejectedAlternative{WorkerID: w.ID, GPUID: g.ID, Reason: reason})
				continue
			}
			eligible = append(eligible, g)
		}

		if len(eligible) < req.GPUCount {
			rejected = append(rejected, RejectedAlternative{
				WorkerID: w.ID,
				Reason:   fmt.Sprintf("only %d of %d required eligible GPUs available", len(eligible), req.GPUCount),
			})
			continue
		}

		if req.TopologyRequirement == domain.TopologyNVLinkGroup && req.GPUCount > 1 {
			if !hasNVLinkGroupOfSize(eligible, req.GPUCount) {
				rejected = append(rejected, RejectedAlternative{
					WorkerID: w.ID,
					Reason:   fmt.Sprintf("no NVLink group with %d eligible GPUs (topology requirement)", req.GPUCount),
				})
				continue
			}
		}

		candidates = append(candidates, WorkerCandidate{Worker: w, Eligible: eligible})
	}

	return candidates, rejected
}

// ineligibleReason returns why g cannot participate in req at all, or ""
// if it's individually eligible. Compatibility only — never scoring.
func ineligibleReason(req domain.WorkloadRequirements, g domain.GPUSnapshot) string {
	if g.AllocationState != domain.GPUFree {
		return fmt.Sprintf("GPU %s is not FREE (allocation state %s)", g.ID, g.AllocationState)
	}
	if g.Validation.Status != domain.ValidationPassed {
		return fmt.Sprintf("GPU %s has not passed validation (status %s)", g.ID, g.Validation.Status)
	}
	if g.Capability.MemoryBytes < req.MinGPUMemoryBytes {
		return fmt.Sprintf("GPU %s has insufficient memory (%d < %d required)", g.ID, g.Capability.MemoryBytes, req.MinGPUMemoryBytes)
	}
	if req.CUDARequirement != "" && !cudaSatisfies(req.CUDARequirement, g.Capability.ComputeCapability) {
		return fmt.Sprintf("GPU %s does not meet CUDA requirement %q (has %q)", g.ID, req.CUDARequirement, g.Capability.ComputeCapability)
	}
	return ""
}

// groupsFor returns the usable pools to select req.GPUCount GPUs from,
// given a worker's eligible set. When req hard-requires an NVLink group
// (and needs more than one GPU), only non-empty NVLink groups with enough
// members are usable pools, in deterministic group-key order. Otherwise
// the whole eligible set is the single usable pool.
func groupsFor(req domain.WorkloadRequirements, eligible []domain.GPUSnapshot) [][]domain.GPUSnapshot {
	if req.TopologyRequirement != domain.TopologyNVLinkGroup || req.GPUCount <= 1 {
		return [][]domain.GPUSnapshot{eligible}
	}
	byGroup := map[string][]domain.GPUSnapshot{}
	for _, g := range eligible {
		if g.Topology.NVLinkGroup == "" {
			continue
		}
		byGroup[g.Topology.NVLinkGroup] = append(byGroup[g.Topology.NVLinkGroup], g)
	}
	keys := make([]string, 0, len(byGroup))
	for k, v := range byGroup {
		if len(v) >= req.GPUCount {
			keys = append(keys, k)
		}
	}
	sort.Strings(keys)
	pools := make([][]domain.GPUSnapshot, 0, len(keys))
	for _, k := range keys {
		pools = append(pools, byGroup[k])
	}
	return pools
}

// hasNVLinkGroupOfSize reports whether at least one non-empty NVLinkGroup
// among the given GPUs has size >= n.
func hasNVLinkGroupOfSize(gpus []domain.GPUSnapshot, n int) bool {
	counts := map[string]int{}
	for _, g := range gpus {
		if g.Topology.NVLinkGroup == "" {
			continue
		}
		counts[g.Topology.NVLinkGroup]++
	}
	for _, c := range counts {
		if c >= n {
			return true
		}
	}
	return false
}

// cudaSatisfies reports whether `have` meets minimum requirement `want`.
// Both are normalized to a major.minor float when possible (accepting
// "sm_80" and "8.0" forms interchangeably, since simulated and real agents
// report capability in different conventions — see docs/agent.md).
// If either side fails to parse, it falls back to exact string equality
// rather than guessing a numeric comparison.
func cudaSatisfies(want, have string) bool {
	wf, werr := parseComputeCapability(want)
	hf, herr := parseComputeCapability(have)
	if werr == nil && herr == nil {
		return hf >= wf
	}
	return want == have
}

func parseComputeCapability(s string) (float64, error) {
	s = strings.ToLower(strings.TrimSpace(s))
	s = strings.TrimPrefix(s, "sm_")
	s = strings.TrimPrefix(s, "sm")
	if !strings.Contains(s, ".") && len(s) >= 2 {
		// "80" -> "8.0"
		s = s[:len(s)-1] + "." + s[len(s)-1:]
	}
	return strconv.ParseFloat(s, 64)
}
