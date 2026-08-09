package scheduler

import (
	"context"
	"fmt"
	"sort"
	"time"

	"gpuforge/internal/domain"
)

// TopologyAware prefers GPU combinations with better communication
// topology: a subset entirely within one NVLink group scores 1.0; a
// fallback subset (chosen when no single group has enough members, or
// topology is simply unknown) scores by how much of the selection happens
// to share a group. It never assumes NVLink/NVSwitch/InfiniBand/RoCE exists
// — cohesion is computed purely from GPUTopology data actually present in
// the snapshot. A single-GPU workload always scores 1.0: there is nothing
// to connect.
type TopologyAware struct{}

func (TopologyAware) Name() string { return "topology-aware" }

func (TopologyAware) Place(ctx context.Context, req domain.WorkloadRequirements, snapshot domain.ClusterSnapshot, now time.Time) (Placement, error) {
	if err := ctx.Err(); err != nil {
		return Placement{}, err
	}
	if err := req.Validate(); err != nil {
		return Placement{}, err
	}

	candidates, rejected := eligibleWorkers(req, snapshot)

	var (
		haveBest     bool
		bestWorker   string
		bestGPUs     []domain.GPUSnapshot
		bestCohesion float64
		bestLeftover uint64
		bestGrouped  bool
	)

	consider := func(workerID string, gpus []domain.GPUSnapshot, cohesion float64) {
		leftover := leftoverMemory(gpus, req.MinGPUMemoryBytes)
		better := !haveBest ||
			cohesion > bestCohesion ||
			(cohesion == bestCohesion && leftover < bestLeftover) ||
			(cohesion == bestCohesion && leftover == bestLeftover &&
				idSliceLess(workerID, gpuIDs(gpus), bestWorker, gpuIDs(bestGPUs)))
		if better {
			haveBest = true
			bestWorker = workerID
			bestGPUs = gpus
			bestCohesion = cohesion
			bestLeftover = leftover
			bestGrouped = cohesion == 1
		}
	}

	for _, c := range candidates {
		if req.GPUCount == 1 {
			for _, pool := range groupsFor(req, c.Eligible) {
				if len(pool) < 1 {
					continue
				}
				consider(c.Worker.ID, smallestKByMemory(pool, 1), 1.0)
			}
			continue
		}

		// Prefer a subset entirely within one non-empty NVLink group.
		groupPools := groupedByNVLink(c.Eligible)
		placedInGroup := false
		for _, pool := range groupPools {
			if len(pool) < req.GPUCount {
				continue
			}
			consider(c.Worker.ID, smallestKByMemory(pool, req.GPUCount), 1.0)
			placedInGroup = true
		}

		// Fallback: no single group has enough members (or none of the
		// eligible GPUs report topology at all). Pick the tightest-fitting
		// subset from the whole eligible pool and score its cohesion
		// honestly rather than assuming connectivity.
		if !placedInGroup && len(c.Eligible) >= req.GPUCount {
			fallback := smallestKByMemory(c.Eligible, req.GPUCount)
			consider(c.Worker.ID, fallback, cohesionOf(fallback))
		}
	}

	if !haveBest {
		return Placement{}, classifyFailure(req.WorkloadID, rejected)
	}

	reason := fmt.Sprintf("selected worker %s's GPU(s) with topology cohesion score %.2f among %d compatible option(s)",
		bestWorker, bestCohesion, len(candidates))
	if bestGrouped {
		reason = fmt.Sprintf("selected worker %s's GPU(s), all within the same NVLink group, the best topology fit among %d compatible option(s)",
			bestWorker, len(candidates))
	} else if bestCohesion == 0 {
		reason = fmt.Sprintf("no NVLink group information connected the eligible GPUs on worker %s; fell back to the tightest-fitting selection (topology unknown, not assumed)", bestWorker)
	}

	return Placement{
		WorkloadID:           req.WorkloadID,
		WorkerID:             bestWorker,
		GPUIDs:               gpuIDs(bestGPUs),
		Policy:               TopologyAware{}.Name(),
		Score:                bestCohesion,
		Reason:               reason,
		RejectedAlternatives: rejected,
		Timestamp:            now,
	}, nil
}

// groupedByNVLink partitions gpus by non-empty NVLinkGroup, in
// deterministic group-key order. GPUs with unknown topology (empty
// NVLinkGroup) never appear in any pool here — unknown topology is never
// treated as "connected."
func groupedByNVLink(gpus []domain.GPUSnapshot) [][]domain.GPUSnapshot {
	byGroup := map[string][]domain.GPUSnapshot{}
	for _, g := range gpus {
		if g.Topology.NVLinkGroup == "" {
			continue
		}
		byGroup[g.Topology.NVLinkGroup] = append(byGroup[g.Topology.NVLinkGroup], g)
	}
	keys := make([]string, 0, len(byGroup))
	for k := range byGroup {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	pools := make([][]domain.GPUSnapshot, 0, len(keys))
	for _, k := range keys {
		pools = append(pools, byGroup[k])
	}
	return pools
}

// cohesionOf scores how much of a selected subset shares one NVLink group:
// the size of the largest same-group cluster within the selection, divided
// by the selection size. A selection with no shared group at all scores 0.
func cohesionOf(gpus []domain.GPUSnapshot) float64 {
	if len(gpus) == 0 {
		return 0
	}
	counts := map[string]int{}
	for _, g := range gpus {
		if g.Topology.NVLinkGroup == "" {
			continue
		}
		counts[g.Topology.NVLinkGroup]++
	}
	max := 0
	for _, c := range counts {
		if c > max {
			max = c
		}
	}
	return float64(max) / float64(len(gpus))
}

var _ Scheduler = TopologyAware{}
