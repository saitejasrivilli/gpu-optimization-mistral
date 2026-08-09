package orchestrator

import (
	"context"
	"sync"
	"time"
)

// Outcome is what a SimulatedExecutor's execution resolves to once its
// configured delay elapses. It never stands in for real GPU/CUDA/NCCL
// execution — see docs/orchestration.md's executor section.
type Outcome string

const (
	OutcomeSucceed Outcome = "SUCCEED"
	OutcomeFail    Outcome = "FAIL"
)

// ExecutionPlan configures how a SimulatedExecutor resolves one workload's
// execution, set in advance via SimulatedExecutor.Plan. If no plan is set
// for a workload, Start uses DefaultPlan (immediate success).
type ExecutionPlan struct {
	Outcome       Outcome
	FailureReason string // required when Outcome == OutcomeFail
	Retryable     bool   // only meaningful when Outcome == OutcomeFail
	Delay         time.Duration
}

// DefaultPlan succeeds immediately (Delay 0).
var DefaultPlan = ExecutionPlan{Outcome: OutcomeSucceed}

type simulatedExecution struct {
	req       ExecutionRequest
	plan      ExecutionPlan
	startedAt time.Time
	// resolved is set once Status has observed a terminal outcome (or Cancel
	// was called), so a terminal phase is reported consistently forever
	// after — Status is not re-derived from `now` once terminal.
	resolved  bool
	phase     ExecutionPhase
	reason    string
	retryable bool
}

// SimulatedExecutor is a deterministic, in-memory Executor. Every workload's
// outcome is configured explicitly via Plan before (or in place of) Start,
// so tests never depend on wall-clock sleeps: Status takes `now` explicitly
// and resolves a delayed plan only once now >= startedAt+Delay.
type SimulatedExecutor struct {
	mu    sync.Mutex
	plans map[string]ExecutionPlan
	execs map[string]*simulatedExecution
}

func NewSimulatedExecutor() *SimulatedExecutor {
	return &SimulatedExecutor{
		plans: make(map[string]ExecutionPlan),
		execs: make(map[string]*simulatedExecution),
	}
}

// Plan configures the outcome workloadID's execution will resolve to. Must
// be called before Start for the plan to take effect (Start captures the
// currently configured plan, or DefaultPlan if none was set).
func (s *SimulatedExecutor) Plan(workloadID string, plan ExecutionPlan) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.plans[workloadID] = plan
}

func (s *SimulatedExecutor) Start(ctx context.Context, req ExecutionRequest, now time.Time) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, exists := s.execs[req.WorkloadID]; exists {
		return ErrAlreadyStarted
	}
	plan, ok := s.plans[req.WorkloadID]
	if !ok {
		plan = DefaultPlan
	}
	s.execs[req.WorkloadID] = &simulatedExecution{
		req:       req,
		plan:      plan,
		startedAt: now,
		phase:     ExecutionRunning,
	}
	return nil
}

func (s *SimulatedExecutor) Status(ctx context.Context, workloadID string, now time.Time) (ExecutionStatus, error) {
	if err := ctx.Err(); err != nil {
		return ExecutionStatus{}, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	e, ok := s.execs[workloadID]
	if !ok {
		return ExecutionStatus{}, ErrUnknownExecution
	}
	if !e.resolved && !now.Before(e.startedAt.Add(e.plan.Delay)) {
		e.resolved = true
		switch e.plan.Outcome {
		case OutcomeFail:
			e.phase = ExecutionFailed
			e.reason = e.plan.FailureReason
			e.retryable = e.plan.Retryable
		default:
			e.phase = ExecutionSucceeded
		}
	}
	return ExecutionStatus{Phase: e.phase, Reason: e.reason, Retryable: e.retryable}, nil
}

func (s *SimulatedExecutor) Cancel(ctx context.Context, workloadID string, now time.Time) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	e, ok := s.execs[workloadID]
	if !ok {
		return ErrUnknownExecution
	}
	if e.resolved {
		if e.phase == ExecutionCancelled {
			return nil // idempotent
		}
		return ErrCannotCancelTerminal
	}
	e.resolved = true
	e.phase = ExecutionCancelled
	return nil
}

var _ Executor = (*SimulatedExecutor)(nil)
