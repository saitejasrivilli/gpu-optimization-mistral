package domain

import (
	"errors"
	"fmt"
)

// Sentinel errors. Use errors.Is against these; use errors.As against the
// concrete *TransitionError / *AllocationError / *HardwareModeError types
// below to recover details.
var (
	ErrReasonRequired              = errors.New("domain: transition reason required")
	ErrEmptyID                     = errors.New("domain: id required")
	ErrInvalidHardwareMode         = errors.New("domain: invalid hardware mode")
	ErrInvalidWorkerTransition     = errors.New("domain: invalid worker state transition")
	ErrInvalidWorkloadTransition   = errors.New("domain: invalid workload state transition")
	ErrInvalidAllocation           = errors.New("domain: invalid allocation")
	ErrAlreadyReleased             = errors.New("domain: allocation already released")
	ErrValidationReasonRequired    = errors.New("domain: validation failure reason required")
	ErrInvalidWorkloadRequirements = errors.New("domain: invalid workload requirements")
)

// WorkerTransitionError carries the detail of a rejected worker transition.
type WorkerTransitionError struct {
	WorkerID string
	From     WorkerState
	To       WorkerState
}

func (e *WorkerTransitionError) Error() string {
	return fmt.Sprintf("domain: worker %q cannot transition %s -> %s", e.WorkerID, e.From, e.To)
}

func (e *WorkerTransitionError) Unwrap() error { return ErrInvalidWorkerTransition }

// WorkloadTransitionError carries the detail of a rejected workload transition.
type WorkloadTransitionError struct {
	WorkloadID string
	From       WorkloadState
	To         WorkloadState
}

func (e *WorkloadTransitionError) Error() string {
	return fmt.Sprintf("domain: workload %q cannot transition %s -> %s", e.WorkloadID, e.From, e.To)
}

func (e *WorkloadTransitionError) Unwrap() error { return ErrInvalidWorkloadTransition }

// AllocationError carries the detail of a rejected allocation operation.
type AllocationError struct {
	Reason string
}

func (e *AllocationError) Error() string {
	return fmt.Sprintf("domain: invalid allocation: %s", e.Reason)
}

func (e *AllocationError) Unwrap() error { return ErrInvalidAllocation }

// HardwareModeError carries the detail of a rejected hardware-mode value or
// mismatch (e.g. a GPU claiming a different mode than its owning worker).
type HardwareModeError struct {
	Reason string
}

func (e *HardwareModeError) Error() string {
	return fmt.Sprintf("domain: invalid hardware mode: %s", e.Reason)
}

func (e *HardwareModeError) Unwrap() error { return ErrInvalidHardwareMode }

// RequirementsError carries the detail of a structurally invalid
// WorkloadRequirements value (as opposed to one that's merely unsatisfiable
// against a given cluster — that's a scheduler-package error, not this one).
type RequirementsError struct {
	Reason string
}

func (e *RequirementsError) Error() string {
	return fmt.Sprintf("domain: invalid workload requirements: %s", e.Reason)
}

func (e *RequirementsError) Unwrap() error { return ErrInvalidWorkloadRequirements }
