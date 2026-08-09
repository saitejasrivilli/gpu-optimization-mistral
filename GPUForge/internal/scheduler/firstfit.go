package scheduler

import (
	"context"
	"fmt"
	"time"

	"gpuforge/internal/domain"
)

// FirstFit selects the first compatible resources under a deterministic
// (worker ID, then GPU ID) ordering. It is the baseline policy: fastest,
// worst fragmentation.
type FirstFit struct{}

func (FirstFit) Name() string { return "first-fit" }

func (FirstFit) Place(ctx context.Context, req domain.WorkloadRequirements, snapshot domain.ClusterSnapshot, now time.Time) (Placement, error) {
	if err := ctx.Err(); err != nil {
		return Placement{}, err
	}
	if err := req.Validate(); err != nil {
		return Placement{}, err
	}

	candidates, rejected := eligibleWorkers(req, snapshot)
	for _, c := range candidates {
		pools := groupsFor(req, c.Eligible)
		for _, pool := range pools {
			if len(pool) < req.GPUCount {
				continue
			}
			sorted := sortedByID(pool)
			selected := sorted[:req.GPUCount]
			return Placement{
				WorkloadID: req.WorkloadID,
				WorkerID:   c.Worker.ID,
				GPUIDs:     gpuIDs(selected),
				Policy:     FirstFit{}.Name(),
				Score:      0,
				Reason: fmt.Sprintf("selected the first %d compatible GPU(s) on worker %s in deterministic ID order",
					req.GPUCount, c.Worker.ID),
				RejectedAlternatives: rejected,
				Timestamp:            now,
			}, nil
		}
	}
	return Placement{}, classifyFailure(req.WorkloadID, rejected)
}

var _ Scheduler = FirstFit{}
