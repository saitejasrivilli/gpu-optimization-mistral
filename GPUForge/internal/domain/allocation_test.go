package domain

import (
	"errors"
	"testing"
	"time"
)

func readyWorkerWithGPUs(t *testing.T, ids ...string) *Worker {
	t.Helper()
	w, err := NewWorker("w1", HardwareModeSimulated)
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range ids {
		g, err := NewGPU(id, "w1", "A100", HardwareModeSimulated, GPUCapability{MemoryBytes: 80 << 30})
		if err != nil {
			t.Fatal(err)
		}
		if err := w.AddGPU(g); err != nil {
			t.Fatal(err)
		}
	}
	now := time.Now()
	must := func(err error) {
		if err != nil {
			t.Fatal(err)
		}
	}
	must(w.Transition(WorkerDiscovering, "discovered", SourceAgentReport, now))
	must(w.Transition(WorkerValidating, "validating", SourceHealthMonitor, now))
	must(w.Transition(WorkerReady, "validated", SourceHealthMonitor, now))
	return w
}

func TestNewAllocation_Success(t *testing.T) {
	wl, _ := NewWorkload("wl1")
	w := readyWorkerWithGPUs(t, "g1", "g2")
	a, err := NewAllocation(wl, w, []string{"g1", "g2"}, time.Now())
	if err != nil {
		t.Fatalf("expected success, got %v", err)
	}
	if a.State() != AllocationActive {
		t.Fatalf("expected ACTIVE, got %s", a.State())
	}
}

func TestNewAllocation_DuplicateGPUIDsRejected(t *testing.T) {
	wl, _ := NewWorkload("wl1")
	w := readyWorkerWithGPUs(t, "g1")
	_, err := NewAllocation(wl, w, []string{"g1", "g1"}, time.Now())
	if !errors.Is(err, ErrInvalidAllocation) {
		t.Fatalf("expected ErrInvalidAllocation for duplicate GPU IDs, got %v", err)
	}
}

func TestNewAllocation_UnknownGPURejected(t *testing.T) {
	wl, _ := NewWorkload("wl1")
	w := readyWorkerWithGPUs(t, "g1")
	_, err := NewAllocation(wl, w, []string{"does-not-exist"}, time.Now())
	if !errors.Is(err, ErrInvalidAllocation) {
		t.Fatalf("expected ErrInvalidAllocation for unknown GPU, got %v", err)
	}
}

func TestNewAllocation_EmptyGPUListRejected(t *testing.T) {
	wl, _ := NewWorkload("wl1")
	w := readyWorkerWithGPUs(t, "g1")
	_, err := NewAllocation(wl, w, nil, time.Now())
	if !errors.Is(err, ErrInvalidAllocation) {
		t.Fatalf("expected ErrInvalidAllocation for empty GPU list, got %v", err)
	}
}

// TestInvariant_QuarantinedWorkerCannotReceiveAllocation is an explicit
// invariant test called out in the Phase 1 spec.
func TestInvariant_QuarantinedWorkerCannotReceiveAllocation(t *testing.T) {
	wl, _ := NewWorkload("wl1")
	w := readyWorkerWithGPUs(t, "g1")
	if err := w.Transition(WorkerQuarantined, "heartbeat lost", SourceHealthMonitor, time.Now()); err != nil {
		t.Fatal(err)
	}
	_, err := NewAllocation(wl, w, []string{"g1"}, time.Now())
	if !errors.Is(err, ErrInvalidAllocation) {
		t.Fatalf("expected quarantined worker to reject allocation, got %v", err)
	}
}

// TestInvariant_RetiredWorkerCannotReceiveAllocation is an explicit
// invariant test called out in the Phase 1 spec.
func TestInvariant_RetiredWorkerCannotReceiveAllocation(t *testing.T) {
	wl, _ := NewWorkload("wl1")
	w := readyWorkerWithGPUs(t, "g1")
	now := time.Now()
	must := func(err error) {
		if err != nil {
			t.Fatal(err)
		}
	}
	must(w.Transition(WorkerQuarantined, "heartbeat lost", SourceHealthMonitor, now))
	must(w.Transition(WorkerRetiring, "decommission", SourceOperator, now))
	must(w.Transition(WorkerRetired, "removed", SourceOperator, now))
	_, err := NewAllocation(wl, w, []string{"g1"}, now)
	if !errors.Is(err, ErrInvalidAllocation) {
		t.Fatalf("expected retired worker to reject allocation, got %v", err)
	}
}

func TestAllocation_ReleaseTwiceRejected(t *testing.T) {
	wl, _ := NewWorkload("wl1")
	w := readyWorkerWithGPUs(t, "g1")
	a, err := NewAllocation(wl, w, []string{"g1"}, time.Now())
	if err != nil {
		t.Fatal(err)
	}
	if err := a.Release("workload completed", time.Now()); err != nil {
		t.Fatalf("expected first release to succeed, got %v", err)
	}
	if err := a.Release("second attempt", time.Now()); !errors.Is(err, ErrAlreadyReleased) {
		t.Fatalf("expected ErrAlreadyReleased, got %v", err)
	}
	if a.State() != AllocationReleased {
		t.Fatalf("expected state to remain RELEASED after failed double-release")
	}
}

func TestAllocation_ReleaseReasonRequired(t *testing.T) {
	wl, _ := NewWorkload("wl1")
	w := readyWorkerWithGPUs(t, "g1")
	a, _ := NewAllocation(wl, w, []string{"g1"}, time.Now())
	if err := a.Release("", time.Now()); !errors.Is(err, ErrReasonRequired) {
		t.Fatalf("expected ErrReasonRequired, got %v", err)
	}
}
