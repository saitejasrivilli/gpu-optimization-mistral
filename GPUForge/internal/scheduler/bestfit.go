package scheduler

import (
	"context"
	"fmt"
	"time"

	"gpuforge/internal/domain"
)

// BestFit prefers the GPU subset that minimizes unused capacity after
// placement (tightest fit), across all compatible workers/pools. Ties are
// broken deterministically by worker ID then GPU ID.
type BestFit struct{}

func (BestFit) Name() string { return "best-fit" }

func (BestFit) Place(ctx context.Context, req domain.WorkloadRequirements, snapshot domain.ClusterSnapshot, now time.Time) (Placement, error) {
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
		bestLeftover uint64
	)
	for _, c := range candidates {
		for _, pool := range groupsFor(req, c.Eligible) {
			if len(pool) < req.GPUCount {
				continue
			}
			selected := smallestKByMemory(pool, req.GPUCount)
			leftover := leftoverMemory(selected, req.MinGPUMemoryBytes)
			if !haveBest ||
				leftover < bestLeftover ||
				(leftover == bestLeftover && idSliceLess(c.Worker.ID, gpuIDs(selected), bestWorker, gpuIDs(bestGPUs))) {
				haveBest = true
				bestWorker = c.Worker.ID
				bestGPUs = selected
				bestLeftover = leftover
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
		Policy:     BestFit{}.Name(),
		Score:      float64(bestLeftover),
		Reason: fmt.Sprintf("selected worker %s's tightest-fitting GPU(s) (%d bytes of unused capacity above the %d-byte minimum, the lowest among %d compatible option(s))",
			bestWorker, bestLeftover, req.MinGPUMemoryBytes, len(candidates)),
		RejectedAlternatives: rejected,
		Timestamp:            now,
	}, nil
}

var _ Scheduler = BestFit{}
