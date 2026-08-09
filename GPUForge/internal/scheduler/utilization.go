package scheduler

import (
	"context"
	"fmt"
	"time"

	"gpuforge/internal/domain"
)

// UtilizationAware prefers the GPU subset with the lowest current
// utilization (as already reported in ClusterSnapshot — this policy
// creates no new telemetry), among compatible options. Ties are broken
// deterministically by worker ID then GPU ID.
type UtilizationAware struct{}

func (UtilizationAware) Name() string { return "utilization-aware" }

func (UtilizationAware) Place(ctx context.Context, req domain.WorkloadRequirements, snapshot domain.ClusterSnapshot, now time.Time) (Placement, error) {
	if err := ctx.Err(); err != nil {
		return Placement{}, err
	}
	if err := req.Validate(); err != nil {
		return Placement{}, err
	}

	candidates, rejected := eligibleWorkers(req, snapshot)

	var (
		haveBest   bool
		bestWorker string
		bestGPUs   []domain.GPUSnapshot
		bestAvg    float64
	)
	for _, c := range candidates {
		for _, pool := range groupsFor(req, c.Eligible) {
			if len(pool) < req.GPUCount {
				continue
			}
			selected := smallestKByUtilization(pool, req.GPUCount)
			avg := avgUtilization(selected)
			if !haveBest ||
				avg < bestAvg ||
				(avg == bestAvg && idSliceLess(c.Worker.ID, gpuIDs(selected), bestWorker, gpuIDs(bestGPUs))) {
				haveBest = true
				bestWorker = c.Worker.ID
				bestGPUs = selected
				bestAvg = avg
			}
		}
	}

	if !haveBest {
		return Placement{}, classifyFailure(req.WorkloadID, rejected)
	}

	return Placement{
		WorkloadID: req.WorkloadID,
		WorkerID:   bestWorker,
		GPUIDs:     gpuIDs(bestGPUs),
		Policy:     UtilizationAware{}.Name(),
		Score:      bestAvg,
		Reason: fmt.Sprintf("selected worker %s's GPU(s) with the lowest average utilization (%.2f%%) among %d compatible option(s)",
			bestWorker, bestAvg, len(candidates)),
		RejectedAlternatives: rejected,
		Timestamp:            now,
	}, nil
}

var _ Scheduler = UtilizationAware{}
