package domain

import (
	"sync"
	"time"
)

// AllocationState is deliberately small: Phase 1 only needs to represent
// that an allocation is currently holding resources or has released them.
// Scheduling (deciding placements, retries, preemption bookkeeping) is a
// later phase's concern.
type AllocationState string

const (
	AllocationActive   AllocationState = "ACTIVE"
	AllocationReleased AllocationState = "RELEASED"
)

// Allocation represents a workload's claim on a specific worker's specific
// GPUs. It never selects which GPUs to use (that's the scheduler's job in a
// later phase) — it only represents a claim that has already been decided.
type Allocation struct {
	mu sync.Mutex

	WorkloadID string
	WorkerID   string
	GPUIDs     []string
	CreatedAt  time.Time

	worker        *Worker
	state         AllocationState
	releasedAt    time.Time
	releaseReason string
}

// NewAllocation validates and constructs an allocation of worker's gpuIDs
// to workload. It enforces the invariants that matter at this layer:
//   - workload and worker must be non-nil and identified
//   - gpuIDs must be non-empty and free of duplicates
//   - every gpuID must actually belong to worker
//   - worker must be in an allocatable state (READY or ALLOCATED) — a
//     QUARANTINED or RETIRED worker can never receive an allocation
//   - every gpuID must currently be FREE — constructing two allocations
//     over the same GPU is rejected, atomically, by Worker.MarkGPUsAllocated
//
// Selecting *which* GPUs to request is the caller's (the scheduler's) job;
// this constructor only guards against constructing an invalid claim.
func NewAllocation(workload *Workload, worker *Worker, gpuIDs []string, now time.Time) (*Allocation, error) {
	if workload == nil || workload.ID() == "" {
		return nil, &AllocationError{Reason: "workload is required"}
	}
	if worker == nil || worker.ID() == "" {
		return nil, &AllocationError{Reason: "worker is required"}
	}
	if len(gpuIDs) == 0 {
		return nil, &AllocationError{Reason: "at least one GPU is required"}
	}

	seen := make(map[string]bool, len(gpuIDs))
	for _, id := range gpuIDs {
		if seen[id] {
			return nil, &AllocationError{Reason: "duplicate GPU ID in allocation: " + id}
		}
		seen[id] = true
		if _, ok := worker.GPU(id); !ok {
			return nil, &AllocationError{Reason: "GPU " + id + " does not belong to worker " + worker.ID()}
		}
	}

	if !WorkerAllocatable(worker.State()) {
		return nil, &AllocationError{Reason: "worker " + worker.ID() + " is not allocatable in state " + string(worker.State())}
	}

	ids := make([]string, len(gpuIDs))
	copy(ids, gpuIDs)

	if err := worker.MarkGPUsAllocated(ids); err != nil {
		return nil, err
	}

	return &Allocation{
		WorkloadID: workload.ID(),
		WorkerID:   worker.ID(),
		GPUIDs:     ids,
		CreatedAt:  now,
		worker:     worker,
		state:      AllocationActive,
	}, nil
}

// State returns the allocation's current state.
func (a *Allocation) State() AllocationState {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.state
}

// Release marks the allocation released. It fails if the allocation was
// already released — release is not idempotent-by-silent-success, callers
// must not call it twice on the same claim.
func (a *Allocation) Release(reason string, now time.Time) error {
	if reason == "" {
		return ErrReasonRequired
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.state == AllocationReleased {
		return ErrAlreadyReleased
	}
	if err := a.worker.MarkGPUsReleased(a.GPUIDs); err != nil {
		return err
	}
	a.state = AllocationReleased
	a.releasedAt = now
	a.releaseReason = reason
	return nil
}

// ReleaseInfo returns the release timestamp and reason, and whether the
// allocation has in fact been released.
func (a *Allocation) ReleaseInfo() (releasedAt time.Time, reason string, released bool) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.state != AllocationReleased {
		return time.Time{}, "", false
	}
	return a.releasedAt, a.releaseReason, true
}
