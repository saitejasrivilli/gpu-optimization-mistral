package domain

import (
	"sync"
	"time"
)

// WorkloadState is the explicit workload lifecycle state defined in
// docs/lifecycle.md. No states beyond these eight exist.
type WorkloadState string

const (
	WorkloadSubmitted WorkloadState = "SUBMITTED"
	WorkloadQueued    WorkloadState = "QUEUED"
	WorkloadScheduled WorkloadState = "SCHEDULED"
	WorkloadRunning   WorkloadState = "RUNNING"
	WorkloadCompleted WorkloadState = "COMPLETED"
	WorkloadFailed    WorkloadState = "FAILED"
	WorkloadRetrying  WorkloadState = "RETRYING"
	WorkloadCancelled WorkloadState = "CANCELLED"
)

// workloadTransitions is the single authoritative table of valid workload
// transitions, taken directly from docs/lifecycle.md.
var workloadTransitions = map[WorkloadState]map[WorkloadState]bool{
	WorkloadSubmitted: {WorkloadQueued: true, WorkloadFailed: true},
	WorkloadQueued:    {WorkloadScheduled: true, WorkloadCancelled: true},
	WorkloadScheduled: {WorkloadRunning: true, WorkloadQueued: true},
	WorkloadRunning:   {WorkloadCompleted: true, WorkloadFailed: true, WorkloadCancelled: true},
	WorkloadFailed:    {WorkloadRetrying: true, WorkloadCancelled: true},
	WorkloadRetrying:  {WorkloadQueued: true},
	WorkloadCompleted: {},
	WorkloadCancelled: {},
}

// IsValidWorkloadTransition reports whether from -> to is a permitted
// workload lifecycle transition.
func IsValidWorkloadTransition(from, to WorkloadState) bool {
	return workloadTransitions[from][to]
}

// WorkloadTransition is an immutable transition record, kept in memory on
// the Workload per docs/lifecycle.md.
type WorkloadTransition struct {
	WorkloadID string
	From       WorkloadState
	To         WorkloadState
	Reason     string
	Timestamp  time.Time
	Source     TransitionSource
}

// Workload is the domain aggregate for a submitted unit of work. Phase 1
// keeps it deliberately minimal: resource requirements and scheduling
// concerns belong to a later phase's scheduler package, not here — this
// type only needs enough identity to drive the lifecycle state machine and
// to be referenced by an Allocation.
type Workload struct {
	mu      sync.Mutex
	id      string
	state   WorkloadState
	history []WorkloadTransition
}

// NewWorkload constructs a Workload in its initial SUBMITTED state.
func NewWorkload(id string) (*Workload, error) {
	if id == "" {
		return nil, ErrEmptyID
	}
	return &Workload{id: id, state: WorkloadSubmitted}, nil
}

func (w *Workload) ID() string { return w.id }

// State returns the workload's current lifecycle state.
func (w *Workload) State() WorkloadState {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.state
}

// History returns a copy of the workload's transition history in order.
func (w *Workload) History() []WorkloadTransition {
	w.mu.Lock()
	defer w.mu.Unlock()
	out := make([]WorkloadTransition, len(w.history))
	copy(out, w.history)
	return out
}

// Transition attempts to move the workload to state `to`. On failure the
// workload's state is left completely unchanged and a
// *WorkloadTransitionError is returned.
func (w *Workload) Transition(to WorkloadState, reason string, source TransitionSource, now time.Time) error {
	if reason == "" {
		return ErrReasonRequired
	}
	w.mu.Lock()
	defer w.mu.Unlock()

	from := w.state
	if !IsValidWorkloadTransition(from, to) {
		return &WorkloadTransitionError{WorkloadID: w.id, From: from, To: to}
	}
	w.state = to
	w.history = append(w.history, WorkloadTransition{
		WorkloadID: w.id,
		From:       from,
		To:         to,
		Reason:     reason,
		Timestamp:  now,
		Source:     source,
	})
	return nil
}
