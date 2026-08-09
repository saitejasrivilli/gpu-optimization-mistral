package orchestrator

import "errors"

var (
	ErrDuplicateWorkload = errors.New("orchestrator: workload already submitted")
	ErrWorkloadNotFound  = errors.New("orchestrator: workload not found")
	ErrWorkerNotFound    = errors.New("orchestrator: worker not found")
	ErrNotCancellable    = errors.New("orchestrator: workload cannot be cancelled from its current state")
	ErrDrainIncomplete   = errors.New("orchestrator: worker still has active allocations, drain not complete")
	ErrQueueEmpty        = errors.New("orchestrator: queue is empty")
	// ErrStaleWorkload is returned by ScheduleNext when a dequeued item's
	// workload is no longer in QUEUED state (e.g. it was cancelled
	// out-of-band). The item is dropped, not requeued or errored loudly.
	ErrStaleWorkload = errors.New("orchestrator: dequeued workload is no longer QUEUED, dropping stale item")
)
