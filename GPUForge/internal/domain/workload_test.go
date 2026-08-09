package domain

import (
	"errors"
	"testing"
	"time"
)

var allWorkloadStates = []WorkloadState{
	WorkloadSubmitted, WorkloadQueued, WorkloadScheduled, WorkloadRunning,
	WorkloadCompleted, WorkloadFailed, WorkloadRetrying, WorkloadCancelled,
}

func TestWorkloadTransitions_AllPairs(t *testing.T) {
	now := time.Now()
	for _, from := range allWorkloadStates {
		for _, to := range allWorkloadStates {
			from, to := from, to
			t.Run(string(from)+"->"+string(to), func(t *testing.T) {
				w := &Workload{id: "wl1", state: from}
				err := w.Transition(to, "test", SourceScheduler, now)
				want := IsValidWorkloadTransition(from, to)
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
					var terr *WorkloadTransitionError
					if !errors.As(err, &terr) {
						t.Fatalf("expected *WorkloadTransitionError, got %T", err)
					}
					if !errors.Is(err, ErrInvalidWorkloadTransition) {
						t.Fatalf("expected errors.Is(err, ErrInvalidWorkloadTransition)")
					}
					if w.State() != from {
						t.Fatalf("state must not mutate on failed transition: expected %s, got %s", from, w.State())
					}
				}
			})
		}
	}
}

// TestInvariant_CompletedWorkloadCannotRun is an explicit invariant test
// called out in the Phase 1 spec.
func TestInvariant_CompletedWorkloadCannotRun(t *testing.T) {
	w := &Workload{id: "wl1", state: WorkloadCompleted}
	err := w.Transition(WorkloadRunning, "attempt", SourceScheduler, time.Now())
	if !errors.Is(err, ErrInvalidWorkloadTransition) {
		t.Fatalf("expected COMPLETED workload to reject -> RUNNING, got %v", err)
	}
}

// TestInvariant_CancelledWorkloadCannotRun is an explicit invariant test
// called out in the Phase 1 spec.
func TestInvariant_CancelledWorkloadCannotRun(t *testing.T) {
	w := &Workload{id: "wl1", state: WorkloadCancelled}
	err := w.Transition(WorkloadRunning, "attempt", SourceScheduler, time.Now())
	if !errors.Is(err, ErrInvalidWorkloadTransition) {
		t.Fatalf("expected CANCELLED workload to reject -> RUNNING, got %v", err)
	}
}

func TestWorkload_ReasonRequired(t *testing.T) {
	w, _ := NewWorkload("wl1")
	if err := w.Transition(WorkloadQueued, "", SourceScheduler, time.Now()); !errors.Is(err, ErrReasonRequired) {
		t.Fatalf("expected ErrReasonRequired, got %v", err)
	}
	if w.State() != WorkloadSubmitted {
		t.Fatalf("state must not mutate when reason missing")
	}
}

func TestNewWorkload_Validation(t *testing.T) {
	if _, err := NewWorkload(""); !errors.Is(err, ErrEmptyID) {
		t.Fatalf("expected ErrEmptyID, got %v", err)
	}
}

func TestWorkload_HistoryRecorded(t *testing.T) {
	w, _ := NewWorkload("wl1")
	now := time.Now()
	if err := w.Transition(WorkloadQueued, "admitted", SourceAdmissionControl, now); err != nil {
		t.Fatal(err)
	}
	hist := w.History()
	if len(hist) != 1 {
		t.Fatalf("expected 1 history entry, got %d", len(hist))
	}
	got := hist[0]
	if got.WorkloadID != "wl1" || got.From != WorkloadSubmitted || got.To != WorkloadQueued ||
		got.Reason != "admitted" || got.Source != SourceAdmissionControl || !got.Timestamp.Equal(now) {
		t.Fatalf("unexpected transition record: %+v", got)
	}
}

func TestConcurrentWorkloadTransitions(t *testing.T) {
	w, _ := NewWorkload("wl1")
	done := make(chan struct{})
	for i := 0; i < 20; i++ {
		go func() {
			_ = w.Transition(WorkloadQueued, "race", SourceScheduler, time.Now())
			_ = w.State()
			_ = w.History()
			done <- struct{}{}
		}()
	}
	for i := 0; i < 20; i++ {
		<-done
	}
}
