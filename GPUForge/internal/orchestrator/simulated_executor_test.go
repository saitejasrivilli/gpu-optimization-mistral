package orchestrator

import (
	"context"
	"errors"
	"testing"
	"time"
)

func TestSimulatedExecutor_ImmediateSuccess(t *testing.T) {
	e := NewSimulatedExecutor()
	now := time.Now()
	if err := e.Start(context.Background(), ExecutionRequest{WorkloadID: "wl1"}, now); err != nil {
		t.Fatal(err)
	}
	status, err := e.Status(context.Background(), "wl1", now)
	if err != nil {
		t.Fatal(err)
	}
	if status.Phase != ExecutionSucceeded {
		t.Fatalf("expected immediate success under DefaultPlan, got %s", status.Phase)
	}
}

func TestSimulatedExecutor_ConfiguredFailure(t *testing.T) {
	e := NewSimulatedExecutor()
	now := time.Now()
	e.Plan("wl1", ExecutionPlan{Outcome: OutcomeFail, FailureReason: "kernel crashed", Retryable: true})
	if err := e.Start(context.Background(), ExecutionRequest{WorkloadID: "wl1"}, now); err != nil {
		t.Fatal(err)
	}
	status, err := e.Status(context.Background(), "wl1", now)
	if err != nil {
		t.Fatal(err)
	}
	if status.Phase != ExecutionFailed || status.Reason != "kernel crashed" || !status.Retryable {
		t.Fatalf("unexpected status: %+v", status)
	}
}

func TestSimulatedExecutor_DelayedCompletion(t *testing.T) {
	e := NewSimulatedExecutor()
	now := time.Now()
	e.Plan("wl1", ExecutionPlan{Outcome: OutcomeSucceed, Delay: 10 * time.Second})
	if err := e.Start(context.Background(), ExecutionRequest{WorkloadID: "wl1"}, now); err != nil {
		t.Fatal(err)
	}

	status, err := e.Status(context.Background(), "wl1", now.Add(5*time.Second))
	if err != nil {
		t.Fatal(err)
	}
	if status.Phase != ExecutionRunning {
		t.Fatalf("expected still RUNNING before delay elapses, got %s", status.Phase)
	}

	status, err = e.Status(context.Background(), "wl1", now.Add(10*time.Second))
	if err != nil {
		t.Fatal(err)
	}
	if status.Phase != ExecutionSucceeded {
		t.Fatalf("expected SUCCEEDED once delay elapses, got %s", status.Phase)
	}
}

func TestSimulatedExecutor_DuplicateStartRejected(t *testing.T) {
	e := NewSimulatedExecutor()
	now := time.Now()
	if err := e.Start(context.Background(), ExecutionRequest{WorkloadID: "wl1"}, now); err != nil {
		t.Fatal(err)
	}
	if err := e.Start(context.Background(), ExecutionRequest{WorkloadID: "wl1"}, now); !errors.Is(err, ErrAlreadyStarted) {
		t.Fatalf("expected ErrAlreadyStarted, got %v", err)
	}
}

func TestSimulatedExecutor_Cancel(t *testing.T) {
	e := NewSimulatedExecutor()
	now := time.Now()
	e.Plan("wl1", ExecutionPlan{Outcome: OutcomeSucceed, Delay: time.Minute})
	_ = e.Start(context.Background(), ExecutionRequest{WorkloadID: "wl1"}, now)

	if err := e.Cancel(context.Background(), "wl1", now); err != nil {
		t.Fatal(err)
	}
	status, _ := e.Status(context.Background(), "wl1", now)
	if status.Phase != ExecutionCancelled {
		t.Fatalf("expected CANCELLED, got %s", status.Phase)
	}
}

func TestSimulatedExecutor_DuplicateCancelIdempotent(t *testing.T) {
	e := NewSimulatedExecutor()
	now := time.Now()
	e.Plan("wl1", ExecutionPlan{Outcome: OutcomeSucceed, Delay: time.Minute})
	_ = e.Start(context.Background(), ExecutionRequest{WorkloadID: "wl1"}, now)
	_ = e.Cancel(context.Background(), "wl1", now)

	if err := e.Cancel(context.Background(), "wl1", now); err != nil {
		t.Fatalf("expected duplicate cancel to be idempotent, got %v", err)
	}
}

func TestSimulatedExecutor_CancelAfterTerminalRejected(t *testing.T) {
	e := NewSimulatedExecutor()
	now := time.Now()
	_ = e.Start(context.Background(), ExecutionRequest{WorkloadID: "wl1"}, now) // DefaultPlan, immediate success
	_, _ = e.Status(context.Background(), "wl1", now)                           // resolve it

	if err := e.Cancel(context.Background(), "wl1", now); !errors.Is(err, ErrCannotCancelTerminal) {
		t.Fatalf("expected ErrCannotCancelTerminal, got %v", err)
	}
}

func TestSimulatedExecutor_UnknownExecution(t *testing.T) {
	e := NewSimulatedExecutor()
	now := time.Now()
	if _, err := e.Status(context.Background(), "nope", now); !errors.Is(err, ErrUnknownExecution) {
		t.Fatalf("expected ErrUnknownExecution, got %v", err)
	}
	if err := e.Cancel(context.Background(), "nope", now); !errors.Is(err, ErrUnknownExecution) {
		t.Fatalf("expected ErrUnknownExecution, got %v", err)
	}
}
