// Package orchestrator implements the control-plane workflow that turns a
// submitted WorkloadRequirements into a scheduled, executed, and released
// workload — the side-effecting layer that sits on top of the pure
// scheduler.Scheduler and the domain lifecycle state machines. See
// docs/orchestration.md.
package orchestrator

import (
	"context"
	"errors"
	"time"
)

// ExecutionPhase is the executor's own view of a running workload. It is
// deliberately smaller than domain.WorkloadState: the executor only knows
// about the process it started, not about queueing/scheduling/retry
// bookkeeping, which belong to the orchestrator.
type ExecutionPhase string

const (
	ExecutionRunning   ExecutionPhase = "RUNNING"
	ExecutionSucceeded ExecutionPhase = "SUCCEEDED"
	ExecutionFailed    ExecutionPhase = "FAILED"
	ExecutionCancelled ExecutionPhase = "CANCELLED"
)

// ExecutionRequest is what the orchestrator hands an executor after a
// scheduling decision has been made and an Allocation constructed. It is
// intentionally minimal: an executor needs to know what to run and where,
// nothing about queueing or retry state.
type ExecutionRequest struct {
	WorkloadID string
	WorkerID   string
	GPUIDs     []string
}

// ExecutionStatus is the result of polling an execution. Retryable is only
// meaningful when Phase == ExecutionFailed: it lets an executor distinguish
// a transient failure (worth retrying) from one that never will be (e.g. a
// user's kernel is simply broken) — the orchestrator's retry policy reads
// this rather than guessing from a reason string.
type ExecutionStatus struct {
	Phase     ExecutionPhase
	Reason    string
	Retryable bool
}

// Sentinel executor errors. Never swallowed: every Executor implementation
// must return one of these (or wrap it) rather than silently no-op'ing.
var (
	// ErrAlreadyStarted is returned by Start when an execution already
	// exists for the given workload ID. Executors are not implicitly
	// idempotent on Start — a real executor cannot safely "restart" a
	// process that's already running, so the orchestrator (via the domain
	// workload state machine) is responsible for never calling Start twice
	// for the same workload; this is the executor's own defense in depth.
	ErrAlreadyStarted = errors.New("orchestrator: execution already started for this workload")
	// ErrUnknownExecution is returned by Status/Cancel when no execution
	// exists for the given workload ID.
	ErrUnknownExecution = errors.New("orchestrator: no execution found for this workload")
	// ErrCannotCancelTerminal is returned by Cancel when the execution has
	// already reached a terminal phase other than Cancelled.
	ErrCannotCancelTerminal = errors.New("orchestrator: cannot cancel an execution that already finished")
)

// Executor is the minimal abstraction the orchestrator uses to actually run
// a workload, real or simulated. Keyed by WorkloadID rather than an opaque
// handle, since there is always exactly one execution per workload.
type Executor interface {
	// Start begins execution. Returns ErrAlreadyStarted if called twice for
	// the same workload.
	Start(ctx context.Context, req ExecutionRequest, now time.Time) error

	// Status reports the current execution state. Returns
	// ErrUnknownExecution if Start was never called (or the execution was
	// never recorded) for workloadID.
	Status(ctx context.Context, workloadID string, now time.Time) (ExecutionStatus, error)

	// Cancel requests the execution stop. Idempotent when the execution is
	// already Cancelled (returns nil); returns ErrCannotCancelTerminal if
	// it already succeeded or failed; returns ErrUnknownExecution if Start
	// was never called.
	Cancel(ctx context.Context, workloadID string, now time.Time) error
}
