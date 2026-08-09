package scheduler

import (
	"errors"
	"fmt"
	"strings"
)

// Sentinel scheduling-failure errors. Use errors.Is against these; use
// errors.As against *SchedulingError to recover the full diagnostic list of
// rejected alternatives.
var (
	ErrNoEligibleWorkers    = errors.New("scheduler: no eligible worker found")
	ErrInsufficientGPUs     = errors.New("scheduler: insufficient compatible GPUs")
	ErrInsufficientMemory   = errors.New("scheduler: insufficient GPU memory")
	ErrIncompatibleCUDA     = errors.New("scheduler: incompatible CUDA requirement")
	ErrIncompatibleTopology = errors.New("scheduler: incompatible topology requirement")
)

// SchedulingError is returned when Place cannot produce a placement. It
// always carries the full set of rejected alternatives considered, so a
// caller (or a test) never has to re-derive why scheduling failed.
type SchedulingError struct {
	WorkloadID string
	Reason     string
	Rejected   []RejectedAlternative
	sentinel   error
}

func (e *SchedulingError) Error() string {
	return fmt.Sprintf("scheduler: workload %q: %s (%d alternatives rejected)", e.WorkloadID, e.Reason, len(e.Rejected))
}

func (e *SchedulingError) Unwrap() error { return e.sentinel }

// classifyFailure picks the most specific sentinel error the full set of
// rejections supports. It is a diagnostic convenience only — the complete,
// authoritative reason set is always in SchedulingError.Rejected.
func classifyFailure(workloadID string, rejected []RejectedAlternative) error {
	if len(rejected) == 0 {
		return &SchedulingError{WorkloadID: workloadID, Reason: "no workers present in cluster snapshot", Rejected: rejected, sentinel: ErrNoEligibleWorkers}
	}

	// Classify using only per-GPU-specific rejections (GPUID != ""); the
	// worker-level "N of M eligible" summary is a consequence of those, not
	// an independent reason, and would otherwise dilute the tally.
	specific := make([]RejectedAlternative, 0, len(rejected))
	for _, r := range rejected {
		if r.GPUID != "" {
			specific = append(specific, r)
		}
	}
	if len(specific) == 0 {
		specific = rejected
	}

	var memory, cuda, topology, gpuCount int
	for _, r := range specific {
		switch {
		case strings.Contains(r.Reason, "memory"):
			memory++
		case strings.Contains(r.Reason, "CUDA"):
			cuda++
		case strings.Contains(r.Reason, "topology") || strings.Contains(r.Reason, "NVLink"):
			topology++
		case strings.Contains(r.Reason, "eligible GPU"):
			gpuCount++
		}
	}

	total := len(specific)
	switch {
	case memory == total:
		return &SchedulingError{WorkloadID: workloadID, Reason: "no GPU met the minimum memory requirement", Rejected: rejected, sentinel: ErrInsufficientMemory}
	case cuda == total:
		return &SchedulingError{WorkloadID: workloadID, Reason: "no GPU met the CUDA requirement", Rejected: rejected, sentinel: ErrIncompatibleCUDA}
	case topology == total:
		return &SchedulingError{WorkloadID: workloadID, Reason: "no GPU combination satisfied the topology requirement", Rejected: rejected, sentinel: ErrIncompatibleTopology}
	case gpuCount == total:
		return &SchedulingError{WorkloadID: workloadID, Reason: "no worker had enough eligible GPUs", Rejected: rejected, sentinel: ErrInsufficientGPUs}
	default:
		return &SchedulingError{WorkloadID: workloadID, Reason: "no compatible placement found", Rejected: rejected, sentinel: ErrInsufficientGPUs}
	}
}
