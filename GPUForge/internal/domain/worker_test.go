package domain

import (
	"errors"
	"testing"
	"time"
)

var allWorkerStates = []WorkerState{
	WorkerProvisioning, WorkerDiscovering, WorkerValidating, WorkerReady,
	WorkerAllocated, WorkerDraining, WorkerMaintenance, WorkerQuarantined,
	WorkerRetiring, WorkerRetired,
}

// TestWorkerTransitions_AllPairs exhaustively checks every (from, to) pair
// against docs/lifecycle.md's transition table, for both valid and invalid
// combinations.
func TestWorkerTransitions_AllPairs(t *testing.T) {
	now := time.Now()
	for _, from := range allWorkerStates {
		for _, to := range allWorkerStates {
			from, to := from, to
			t.Run(string(from)+"->"+string(to), func(t *testing.T) {
				w := &Worker{id: "w1", hardwareMode: HardwareModeSimulated, state: from, gpus: map[string]*GPU{}}
				err := w.Transition(to, "test", SourceOperator, now)
				want := IsValidWorkerTransition(from, to)
				if want {
					if err != nil {
						t.Fatalf("expected valid transition %s->%s to succeed, got %v", from, to, err)
					}
					if w.State() != to {
						t.Fatalf("expected state %s, got %s", to, w.State())
					}
				} else {
					if err == nil {
						t.Fatalf("expected invalid transition %s->%s to fail", from, to)
					}
					var terr *WorkerTransitionError
					if !errors.As(err, &terr) {
						t.Fatalf("expected *WorkerTransitionError, got %T", err)
					}
					if !errors.Is(err, ErrInvalidWorkerTransition) {
						t.Fatalf("expected errors.Is(err, ErrInvalidWorkerTransition)")
					}
					if w.State() != from {
						t.Fatalf("state must not mutate on failed transition: expected %s, got %s", from, w.State())
					}
				}
			})
		}
	}
}

func TestWorkerTransition_ReasonRequired(t *testing.T) {
	w, err := NewWorker("w1", HardwareModeSimulated)
	if err != nil {
		t.Fatal(err)
	}
	if err := w.Transition(WorkerDiscovering, "", SourceOperator, time.Now()); !errors.Is(err, ErrReasonRequired) {
		t.Fatalf("expected ErrReasonRequired, got %v", err)
	}
	if w.State() != WorkerProvisioning {
		t.Fatalf("state must not mutate when reason missing")
	}
}

func TestWorkerTransition_HistoryRecorded(t *testing.T) {
	w, _ := NewWorker("w1", HardwareModeSimulated)
	now := time.Now()
	if err := w.Transition(WorkerDiscovering, "agent started", SourceAgentReport, now); err != nil {
		t.Fatal(err)
	}
	hist := w.History()
	if len(hist) != 1 {
		t.Fatalf("expected 1 history entry, got %d", len(hist))
	}
	got := hist[0]
	if got.WorkerID != "w1" || got.From != WorkerProvisioning || got.To != WorkerDiscovering ||
		got.Reason != "agent started" || got.Source != SourceAgentReport || !got.Timestamp.Equal(now) {
		t.Fatalf("unexpected transition record: %+v", got)
	}
}

func TestNewWorker_Validation(t *testing.T) {
	if _, err := NewWorker("", HardwareModeSimulated); !errors.Is(err, ErrEmptyID) {
		t.Fatalf("expected ErrEmptyID, got %v", err)
	}
	if _, err := NewWorker("w1", HardwareMode("quantum")); !errors.Is(err, ErrInvalidHardwareMode) {
		t.Fatalf("expected ErrInvalidHardwareMode, got %v", err)
	}
}

// TestInvariant_RetiredWorkerCannotBecomeReady is an explicit invariant test
// called out in the Phase 1 spec.
func TestInvariant_RetiredWorkerCannotBecomeReady(t *testing.T) {
	w := &Worker{id: "w1", hardwareMode: HardwareModeSimulated, state: WorkerRetired, gpus: map[string]*GPU{}}
	err := w.Transition(WorkerReady, "attempt", SourceOperator, time.Now())
	if !errors.Is(err, ErrInvalidWorkerTransition) {
		t.Fatalf("expected retired worker to reject -> READY, got %v", err)
	}
}

func TestWorker_AddGPU_HardwareModeMismatchRejected(t *testing.T) {
	w, _ := NewWorker("w1", HardwareModeSimulated)
	g, _ := NewGPU("g1", "w1", "A100", HardwareModeReal, GPUCapability{})
	err := w.AddGPU(g)
	if !errors.Is(err, ErrInvalidHardwareMode) {
		t.Fatalf("expected hardware mode mismatch to be rejected, got %v", err)
	}
}

func TestWorker_AddGPU_MatchingModeAccepted(t *testing.T) {
	w, _ := NewWorker("w1", HardwareModeSimulated)
	g, _ := NewGPU("g1", "w1", "A100", HardwareModeSimulated, GPUCapability{})
	if err := w.AddGPU(g); err != nil {
		t.Fatalf("expected matching hardware mode to be accepted, got %v", err)
	}
	got, ok := w.GPU("g1")
	if !ok || got.ID != "g1" {
		t.Fatalf("expected to retrieve attached GPU")
	}
}

// TestConcurrentTransitions exercises the worker under concurrent access;
// run with -race.
func TestConcurrentTransitions(t *testing.T) {
	w, _ := NewWorker("w1", HardwareModeSimulated)
	done := make(chan struct{})
	for i := 0; i < 20; i++ {
		go func() {
			_ = w.Transition(WorkerDiscovering, "race", SourceOperator, time.Now())
			_ = w.State()
			_ = w.History()
			done <- struct{}{}
		}()
	}
	for i := 0; i < 20; i++ {
		<-done
	}
}
