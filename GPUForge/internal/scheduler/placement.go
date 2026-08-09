// Package scheduler implements pure, deterministic GPU placement policies
// on top of the domain package's ClusterSnapshot and WorkloadRequirements.
// It never mutates its inputs, never performs infrastructure operations,
// and never depends on NVIDIA/CUDA/Kubernetes/Docker/HTTP/Prometheus — see
// docs/scheduling-engine.md.
package scheduler

import (
	"context"
	"time"

	"gpuforge/internal/domain"
)

// RejectedAlternative explains why one candidate (a whole worker, or one
// GPU on a worker) was not selected. GPUID is empty when the rejection
// applies to the worker as a whole (e.g. not enough eligible GPUs).
type RejectedAlternative struct {
	WorkerID string
	GPUID    string
	Reason   string
}

// Placement is a successful scheduling decision. It is intentionally small:
// only what's needed to apply the decision (WorkerID, GPUIDs) and to
// explain it (Policy, Score, Reason, RejectedAlternatives).
type Placement struct {
	WorkloadID           string
	WorkerID             string
	GPUIDs               []string
	Policy               string
	Score                float64
	Reason               string
	RejectedAlternatives []RejectedAlternative
	Timestamp            time.Time
}

// Scheduler is a pluggable placement policy. Implementations must be
// deterministic (same inputs -> same output), side-effect free (never
// mutate snapshot or requirements), and safe for concurrent use.
type Scheduler interface {
	// Name identifies the policy, echoed into Placement.Policy.
	Name() string

	// Place selects GPUs for req from snapshot, or returns a
	// *SchedulingError explaining why no placement exists. It must not
	// mutate snapshot or req.
	Place(ctx context.Context, req domain.WorkloadRequirements, snapshot domain.ClusterSnapshot, now time.Time) (Placement, error)
}
