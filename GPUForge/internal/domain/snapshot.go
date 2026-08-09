package domain

import "time"

// GPUSnapshot is a read-only copy of a GPU's state at snapshot time. Unlike
// *GPU it carries no mutex and is safe to pass across goroutines/policies
// without synchronization.
type GPUSnapshot struct {
	ID           string
	WorkerID     string
	Model        string
	HardwareMode HardwareMode
	Capability   GPUCapability
	State        GPUState
	Validation   ValidationResult
}

// WorkerSnapshot is a read-only copy of a worker's state at snapshot time.
type WorkerSnapshot struct {
	ID           string
	HardwareMode HardwareMode
	State        WorkerState
	GPUs         []GPUSnapshot
}

// ClusterSnapshot is an immutable, point-in-time view of the cluster's
// workers and their GPUs. It exists in Phase 1 purely as the input type a
// later scheduler will consume (per docs/scheduler.md's
// Place(ctx, req, snapshot)); Phase 1 does not implement anything that
// produces or scores against it beyond construction.
type ClusterSnapshot struct {
	Workers   []WorkerSnapshot
	Timestamp time.Time
}

// Snapshot produces a read-only copy of the worker's current state.
func (w *Worker) Snapshot() WorkerSnapshot {
	gpus := w.GPUs()
	out := WorkerSnapshot{
		ID:           w.ID(),
		HardwareMode: w.HardwareMode(),
		State:        w.State(),
		GPUs:         make([]GPUSnapshot, 0, len(gpus)),
	}
	for _, g := range gpus {
		out.GPUs = append(out.GPUs, GPUSnapshot{
			ID:           g.ID,
			WorkerID:     g.WorkerID,
			Model:        g.Model,
			HardwareMode: g.HardwareMode,
			Capability:   g.Capability,
			State:        g.State,
			Validation:   g.Validation,
		})
	}
	return out
}

// NewClusterSnapshot builds a ClusterSnapshot from the given workers at now.
func NewClusterSnapshot(workers []*Worker, now time.Time) ClusterSnapshot {
	out := ClusterSnapshot{
		Workers:   make([]WorkerSnapshot, 0, len(workers)),
		Timestamp: now,
	}
	for _, w := range workers {
		out.Workers = append(out.Workers, w.Snapshot())
	}
	return out
}
