package orchestrator

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"

	"gpuforge/internal/domain"
	"gpuforge/internal/scheduler"
)

// readyWorker builds a READY worker with n FREE, validated, 80GB GPUs.
func readyWorker(t *testing.T, id string, n int, now time.Time) *domain.Worker {
	t.Helper()
	w, err := domain.NewWorker(id, domain.HardwareModeSimulated)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < n; i++ {
		g, err := domain.NewGPU(fmt.Sprintf("%s-gpu-%d", id, i), id, "A100", domain.HardwareModeSimulated,
			domain.GPUCapability{ComputeCapability: "sm_80", MemoryBytes: 80 << 30})
		if err != nil {
			t.Fatal(err)
		}
		if err := w.AddGPU(g); err != nil {
			t.Fatal(err)
		}
		if err := w.UpdateGPUValidation(g.ID, domain.Pass(now)); err != nil {
			t.Fatal(err)
		}
	}
	must(t, w.Transition(domain.WorkerDiscovering, "discovered", domain.SourceAgentReport, now))
	must(t, w.Transition(domain.WorkerValidating, "validating", domain.SourceHealthMonitor, now))
	must(t, w.Transition(domain.WorkerReady, "validated", domain.SourceHealthMonitor, now))
	return w
}

func must(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}

func newTestOrchestrator(policy RetryPolicy) (*Orchestrator, *SimulatedExecutor) {
	exec := NewSimulatedExecutor()
	o := New(scheduler.FirstFit{}, exec, policy)
	return o, exec
}

// --- 1/2/4: successful & queued workload, successful allocation ---

func TestSuccessfulWorkload_EndToEnd(t *testing.T) {
	now := time.Now()
	o, _ := newTestOrchestrator(DefaultRetryPolicy)
	o.RegisterWorker(readyWorker(t, "w1", 1, now))

	if err := o.Submit(domain.WorkloadRequirements{WorkloadID: "wl1", GPUCount: 1}, now); err != nil {
		t.Fatal(err)
	}
	if o.QueueLen() != 1 {
		t.Fatalf("expected 1 queued workload, got %d", o.QueueLen())
	}

	scheduled, err := o.ScheduleNext(context.Background(), now)
	if err != nil || !scheduled {
		t.Fatalf("expected successful scheduling, got scheduled=%v err=%v", scheduled, err)
	}
	wl, _ := o.Workload("wl1")
	if wl.State() != domain.WorkloadRunning {
		t.Fatalf("expected RUNNING, got %s", wl.State())
	}
	if _, ok := o.Allocation("wl1"); !ok {
		t.Fatal("expected an active allocation")
	}

	o.Tick(context.Background(), now) // DefaultPlan resolves immediately
	if wl.State() != domain.WorkloadCompleted {
		t.Fatalf("expected COMPLETED, got %s", wl.State())
	}
	if _, ok := o.Allocation("wl1"); ok {
		t.Fatal("expected allocation released after completion")
	}
	g, _ := o.workers["w1"].GPU("w1-gpu-0")
	if g.AllocationState != domain.GPUFree {
		t.Fatalf("expected GPU freed after completion, got %s", g.AllocationState)
	}
}

// --- 3: insufficient resources ---

func TestInsufficientResources_RequeuedThenCancelledAfterMaxAttempts(t *testing.T) {
	now := time.Now()
	policy := RetryPolicy{MaxAttempts: 2, BaseDelay: time.Second, Factor: 2, MaxDelay: time.Minute}
	o, _ := newTestOrchestrator(policy)
	o.RegisterWorker(readyWorker(t, "w1", 1, now)) // only 1 GPU, workload needs 2

	must(t, o.Submit(domain.WorkloadRequirements{WorkloadID: "wl1", GPUCount: 2}, now))

	scheduled, err := o.ScheduleNext(context.Background(), now)
	if scheduled || err == nil {
		t.Fatalf("expected scheduling to fail, got scheduled=%v err=%v", scheduled, err)
	}
	wl, _ := o.Workload("wl1")
	if wl.State() != domain.WorkloadQueued {
		t.Fatalf("expected still QUEUED after first failed attempt, got %s", wl.State())
	}
	if o.QueueLen() != 1 {
		t.Fatal("expected workload requeued")
	}

	// Second attempt also fails -> attempts (2) reaches MaxAttempts -> CANCELLED.
	scheduled, err = o.ScheduleNext(context.Background(), now)
	if scheduled || err == nil {
		t.Fatal("expected second scheduling attempt to also fail")
	}
	if wl.State() != domain.WorkloadCancelled {
		t.Fatalf("expected CANCELLED after exhausting attempts, got %s", wl.State())
	}
	if o.QueueLen() != 0 {
		t.Fatal("expected workload not requeued after giving up")
	}
}

// --- 6/7: executor failure and retry ---

func TestExecutorFailure_RetryEligible(t *testing.T) {
	now := time.Now()
	o, exec := newTestOrchestrator(DefaultRetryPolicy)
	o.RegisterWorker(readyWorker(t, "w1", 1, now))
	exec.Plan("wl1", ExecutionPlan{Outcome: OutcomeFail, FailureReason: "transient node issue", Retryable: true})

	must(t, o.Submit(domain.WorkloadRequirements{WorkloadID: "wl1", GPUCount: 1}, now))
	scheduled, err := o.ScheduleNext(context.Background(), now)
	if err != nil || !scheduled {
		t.Fatalf("expected scheduling to succeed (executor Start always succeeds), got %v %v", scheduled, err)
	}

	o.Tick(context.Background(), now)
	wl, _ := o.Workload("wl1")
	if wl.State() != domain.WorkloadRetrying {
		t.Fatalf("expected RETRYING after retryable failure, got %s", wl.State())
	}
	if _, ok := o.Allocation("wl1"); ok {
		t.Fatal("expected allocation released after failure")
	}

	// Advance time past backoff; Tick should promote it back to QUEUED.
	later := now.Add(DefaultRetryPolicy.NextDelay(1) + time.Millisecond)
	o.Tick(context.Background(), later)
	if wl.State() != domain.WorkloadQueued {
		t.Fatalf("expected QUEUED after backoff elapsed, got %s", wl.State())
	}
	if o.QueueLen() != 1 {
		t.Fatal("expected retried workload back in queue")
	}
}

func TestExecutorFailure_NonRetryableCancelled(t *testing.T) {
	now := time.Now()
	o, exec := newTestOrchestrator(DefaultRetryPolicy)
	o.RegisterWorker(readyWorker(t, "w1", 1, now))
	exec.Plan("wl1", ExecutionPlan{Outcome: OutcomeFail, FailureReason: "bad user kernel", Retryable: false})

	must(t, o.Submit(domain.WorkloadRequirements{WorkloadID: "wl1", GPUCount: 1}, now))
	scheduled, err := o.ScheduleNext(context.Background(), now)
	if err != nil || !scheduled {
		t.Fatalf("expected scheduling to succeed, got scheduled=%v err=%v", scheduled, err)
	}

	o.Tick(context.Background(), now)
	wl, _ := o.Workload("wl1")
	if wl.State() != domain.WorkloadCancelled {
		t.Fatalf("expected CANCELLED for non-retryable failure, got %s", wl.State())
	}
}

func TestRetryExhaustion_EndsCancelled(t *testing.T) {
	now := time.Now()
	policy := RetryPolicy{MaxAttempts: 1, BaseDelay: time.Second, Factor: 2, MaxDelay: time.Minute}
	o, exec := newTestOrchestrator(policy)
	o.RegisterWorker(readyWorker(t, "w1", 1, now))
	exec.Plan("wl1", ExecutionPlan{Outcome: OutcomeFail, FailureReason: "always fails", Retryable: true})

	must(t, o.Submit(domain.WorkloadRequirements{WorkloadID: "wl1", GPUCount: 1}, now))
	_, _ = o.ScheduleNext(context.Background(), now)
	o.Tick(context.Background(), now)

	wl, _ := o.Workload("wl1")
	if wl.State() != domain.WorkloadCancelled {
		t.Fatalf("expected CANCELLED once MaxAttempts=1 is exhausted after first failure, got %s", wl.State())
	}
}

// --- 8/9: cancellation and duplicate cancellation ---

func TestCancel_Queued(t *testing.T) {
	now := time.Now()
	o, _ := newTestOrchestrator(DefaultRetryPolicy)
	must(t, o.Submit(domain.WorkloadRequirements{WorkloadID: "wl1", GPUCount: 1}, now))

	if err := o.Cancel("wl1", "user requested", now); err != nil {
		t.Fatal(err)
	}
	wl, _ := o.Workload("wl1")
	if wl.State() != domain.WorkloadCancelled {
		t.Fatalf("expected CANCELLED, got %s", wl.State())
	}
	if o.QueueLen() != 0 {
		t.Fatal("expected removal from queue")
	}
}

func TestCancel_Running(t *testing.T) {
	now := time.Now()
	o, exec := newTestOrchestrator(DefaultRetryPolicy)
	o.RegisterWorker(readyWorker(t, "w1", 1, now))
	exec.Plan("wl1", ExecutionPlan{Outcome: OutcomeSucceed, Delay: time.Hour})
	must(t, o.Submit(domain.WorkloadRequirements{WorkloadID: "wl1", GPUCount: 1}, now))
	_, err := o.ScheduleNext(context.Background(), now)
	must(t, err)

	if err := o.Cancel("wl1", "user requested", now); err != nil {
		t.Fatal(err)
	}
	wl, _ := o.Workload("wl1")
	if wl.State() != domain.WorkloadCancelled {
		t.Fatalf("expected CANCELLED, got %s", wl.State())
	}
	if _, ok := o.Allocation("wl1"); ok {
		t.Fatal("expected allocation released on cancel")
	}
}

func TestCancel_DuplicateIsIdempotent(t *testing.T) {
	now := time.Now()
	o, _ := newTestOrchestrator(DefaultRetryPolicy)
	must(t, o.Submit(domain.WorkloadRequirements{WorkloadID: "wl1", GPUCount: 1}, now))
	must(t, o.Cancel("wl1", "first", now))

	if err := o.Cancel("wl1", "second", now); err != nil {
		t.Fatalf("expected duplicate cancel to be idempotent, got %v", err)
	}
}

func TestCancel_AlreadyCompletedIsIdempotent(t *testing.T) {
	now := time.Now()
	o, _ := newTestOrchestrator(DefaultRetryPolicy)
	o.RegisterWorker(readyWorker(t, "w1", 1, now))
	must(t, o.Submit(domain.WorkloadRequirements{WorkloadID: "wl1", GPUCount: 1}, now))
	_, err := o.ScheduleNext(context.Background(), now)
	must(t, err)
	o.Tick(context.Background(), now) // completes under DefaultPlan

	if err := o.Cancel("wl1", "too late", now); err != nil {
		t.Fatalf("expected cancel of completed workload to be a no-op, got %v", err)
	}
}

func TestCancel_UnknownWorkload(t *testing.T) {
	o, _ := newTestOrchestrator(DefaultRetryPolicy)
	if err := o.Cancel("nope", "x", time.Now()); !errors.Is(err, ErrWorkloadNotFound) {
		t.Fatalf("expected ErrWorkloadNotFound, got %v", err)
	}
}

// --- 10/11: duplicate release, duplicate start ---

func TestDuplicateRelease_IsSafe(t *testing.T) {
	now := time.Now()
	o, _ := newTestOrchestrator(DefaultRetryPolicy)
	o.RegisterWorker(readyWorker(t, "w1", 1, now))
	must(t, o.Submit(domain.WorkloadRequirements{WorkloadID: "wl1", GPUCount: 1}, now))
	_, err := o.ScheduleNext(context.Background(), now)
	must(t, err)

	alloc, _ := o.Allocation("wl1")
	// Release once via Tick (completion), then attempt again directly:
	o.Tick(context.Background(), now)
	if err := alloc.Release("second attempt", now); !errors.Is(err, domain.ErrAlreadyReleased) {
		t.Fatalf("expected domain-level double release to be rejected, got %v", err)
	}
	// And the orchestrator's own release helper must not panic/corrupt state either.
	o.releaseAllocation("wl1", "third attempt", now)
	if _, ok := o.Allocation("wl1"); ok {
		t.Fatal("expected allocation to remain released/absent")
	}
}

func TestDuplicateStart_RejectedByExecutor(t *testing.T) {
	e := NewSimulatedExecutor()
	now := time.Now()
	must(t, e.Start(context.Background(), ExecutionRequest{WorkloadID: "wl1"}, now))
	if err := e.Start(context.Background(), ExecutionRequest{WorkloadID: "wl1"}, now); !errors.Is(err, ErrAlreadyStarted) {
		t.Fatalf("expected ErrAlreadyStarted, got %v", err)
	}
}

// --- 12: worker draining ---

func TestWorkerDraining_NoNewScheduling(t *testing.T) {
	now := time.Now()
	o, _ := newTestOrchestrator(DefaultRetryPolicy)
	o.RegisterWorker(readyWorker(t, "w1", 1, now))
	must(t, o.DrainWorker("w1", "maintenance window", now))

	must(t, o.Submit(domain.WorkloadRequirements{WorkloadID: "wl1", GPUCount: 1}, now))
	scheduled, err := o.ScheduleNext(context.Background(), now)
	if scheduled || err == nil {
		t.Fatal("expected scheduling to fail: draining worker must not be selected")
	}
	wl, _ := o.Workload("wl1")
	if wl.State() != domain.WorkloadQueued {
		t.Fatalf("expected workload to remain QUEUED (retryable), got %s", wl.State())
	}
}

func TestWorkerDraining_ExistingWorkloadRunsToCompletion(t *testing.T) {
	now := time.Now()
	o, _ := newTestOrchestrator(DefaultRetryPolicy)
	o.RegisterWorker(readyWorker(t, "w1", 1, now))
	must(t, o.Submit(domain.WorkloadRequirements{WorkloadID: "wl1", GPUCount: 1}, now))
	_, err := o.ScheduleNext(context.Background(), now)
	must(t, err)

	must(t, o.DrainWorker("w1", "maintenance window", now))
	if err := o.CompleteDraining("w1", "done", now); !errors.Is(err, ErrDrainIncomplete) {
		t.Fatalf("expected ErrDrainIncomplete while wl1 still running, got %v", err)
	}

	o.Tick(context.Background(), now) // completes
	if err := o.CompleteDraining("w1", "done", now); err != nil {
		t.Fatalf("expected drain to complete once no allocations remain, got %v", err)
	}
	w, _ := o.Worker("w1")
	if w.State() != domain.WorkerMaintenance {
		t.Fatalf("expected MAINTENANCE, got %s", w.State())
	}
}

// --- 16: stale state ---

func TestStaleWorkload_DroppedNotActedOn(t *testing.T) {
	now := time.Now()
	o, _ := newTestOrchestrator(DefaultRetryPolicy)
	o.RegisterWorker(readyWorker(t, "w1", 1, now))
	must(t, o.Submit(domain.WorkloadRequirements{WorkloadID: "wl1", GPUCount: 1}, now))

	// Simulate out-of-band state divergence: mutate the workload directly
	// without going through Cancel's queue.Remove, to prove ScheduleNext
	// defends against a stale queue entry rather than blindly scheduling it.
	wl, _ := o.Workload("wl1")
	must(t, wl.Transition(domain.WorkloadCancelled, "external cancel", domain.SourceOperator, now))

	scheduled, err := o.ScheduleNext(context.Background(), now)
	if scheduled || !errors.Is(err, ErrStaleWorkload) {
		t.Fatalf("expected ErrStaleWorkload, got scheduled=%v err=%v", scheduled, err)
	}
	if wl.State() != domain.WorkloadCancelled {
		t.Fatalf("expected state to remain CANCELLED, got %s", wl.State())
	}
}

// --- 17/18: delayed completion, failure after allocation ---

func TestExecutorDelayedCompletion_TicksUntilDone(t *testing.T) {
	now := time.Now()
	o, exec := newTestOrchestrator(DefaultRetryPolicy)
	o.RegisterWorker(readyWorker(t, "w1", 1, now))
	exec.Plan("wl1", ExecutionPlan{Outcome: OutcomeSucceed, Delay: 10 * time.Second})
	must(t, o.Submit(domain.WorkloadRequirements{WorkloadID: "wl1", GPUCount: 1}, now))
	_, err := o.ScheduleNext(context.Background(), now)
	must(t, err)

	o.Tick(context.Background(), now.Add(5*time.Second))
	wl, _ := o.Workload("wl1")
	if wl.State() != domain.WorkloadRunning {
		t.Fatalf("expected still RUNNING before delay elapses, got %s", wl.State())
	}

	o.Tick(context.Background(), now.Add(10*time.Second))
	if wl.State() != domain.WorkloadCompleted {
		t.Fatalf("expected COMPLETED once delay elapses, got %s", wl.State())
	}
}

func TestExecutorFailureAfterAllocation_ReleasesGPU(t *testing.T) {
	now := time.Now()
	o, exec := newTestOrchestrator(RetryPolicy{MaxAttempts: 1, BaseDelay: time.Second, Factor: 2, MaxDelay: time.Minute})
	w := readyWorker(t, "w1", 1, now)
	o.RegisterWorker(w)
	exec.Plan("wl1", ExecutionPlan{Outcome: OutcomeFail, FailureReason: "GPU ECC error", Retryable: false})
	must(t, o.Submit(domain.WorkloadRequirements{WorkloadID: "wl1", GPUCount: 1}, now))
	_, err := o.ScheduleNext(context.Background(), now)
	must(t, err)

	if _, ok := o.Allocation("wl1"); !ok {
		t.Fatal("expected allocation to exist while RUNNING")
	}
	o.Tick(context.Background(), now)

	if _, ok := o.Allocation("wl1"); ok {
		t.Fatal("expected allocation released after failure")
	}
	g, _ := w.GPU("w1-gpu-0")
	if g.AllocationState != domain.GPUFree {
		t.Fatalf("expected GPU freed after failure, got %s", g.AllocationState)
	}
}

// --- 13/14/15: concurrency and no double allocation ---

func TestConcurrentScheduling_NoDoubleAllocation(t *testing.T) {
	now := time.Now()
	o, _ := newTestOrchestrator(DefaultRetryPolicy)
	o.RegisterWorker(readyWorker(t, "w1", 4, now)) // exactly 4 GPUs

	for i := 0; i < 8; i++ {
		must(t, o.Submit(domain.WorkloadRequirements{WorkloadID: fmt.Sprintf("wl-%d", i), GPUCount: 1}, now))
	}

	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, _ = o.ScheduleNext(context.Background(), now)
		}()
	}
	wg.Wait()

	// Drain any that failed only due to transient contention re-queue.
	for tries := 0; tries < 20 && o.QueueLen() > 0; tries++ {
		_, _ = o.ScheduleNext(context.Background(), now)
	}

	if o.RunningCount() > 4 {
		t.Fatalf("expected at most 4 concurrently running workloads (4 GPUs), got %d", o.RunningCount())
	}

	seen := map[string]bool{}
	for i := 0; i < 8; i++ {
		wl, _ := o.Workload(fmt.Sprintf("wl-%d", i))
		if wl.State() != domain.WorkloadRunning {
			continue
		}
		alloc, ok := o.Allocation(wl.ID())
		if !ok {
			t.Fatalf("running workload %s has no allocation", wl.ID())
		}
		for _, gid := range alloc.GPUIDs {
			if seen[gid] {
				t.Fatalf("GPU %s double-allocated", gid)
			}
			seen[gid] = true
		}
	}
}

func TestConcurrentSchedulingAcrossWorkers(t *testing.T) {
	now := time.Now()
	o, _ := newTestOrchestrator(DefaultRetryPolicy)
	o.RegisterWorker(readyWorker(t, "w1", 2, now))
	o.RegisterWorker(readyWorker(t, "w2", 2, now))

	for i := 0; i < 4; i++ {
		must(t, o.Submit(domain.WorkloadRequirements{WorkloadID: fmt.Sprintf("wl-%d", i), GPUCount: 1}, now))
	}

	var wg sync.WaitGroup
	results := make([]bool, 4)
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			scheduled, _ := o.ScheduleNext(context.Background(), now)
			results[i] = scheduled
		}(i)
	}
	wg.Wait()

	for _, r := range results {
		if !r {
			t.Fatal("expected all 4 workloads to schedule successfully across 2 workers with 2 GPUs each")
		}
	}
}

func TestCancelDuringConcurrentScheduling(t *testing.T) {
	now := time.Now()
	o, _ := newTestOrchestrator(DefaultRetryPolicy)
	o.RegisterWorker(readyWorker(t, "w1", 1, now))
	must(t, o.Submit(domain.WorkloadRequirements{WorkloadID: "wl1", GPUCount: 1}, now))

	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		_, _ = o.ScheduleNext(context.Background(), now)
	}()
	go func() {
		defer wg.Done()
		_ = o.Cancel("wl1", "racing cancel", now)
	}()
	wg.Wait()

	wl, _ := o.Workload("wl1")
	switch wl.State() {
	case domain.WorkloadCancelled, domain.WorkloadRunning:
		// Either outcome is valid depending on which goroutine won the
		// coarse lock first; both are well-defined terminal-ish states,
		// never a corrupted/partial one.
	default:
		t.Fatalf("expected CANCELLED or RUNNING, got %s", wl.State())
	}
	if wl.State() == domain.WorkloadRunning {
		if _, ok := o.Allocation("wl1"); !ok {
			t.Fatal("RUNNING workload must have an active allocation")
		}
	} else if _, ok := o.Allocation("wl1"); ok {
		t.Fatal("CANCELLED workload must not hold an allocation")
	}
}

func TestCompletionDuringCancellation(t *testing.T) {
	now := time.Now()
	o, exec := newTestOrchestrator(DefaultRetryPolicy)
	o.RegisterWorker(readyWorker(t, "w1", 1, now))
	exec.Plan("wl1", ExecutionPlan{Outcome: OutcomeSucceed, Delay: 0})
	must(t, o.Submit(domain.WorkloadRequirements{WorkloadID: "wl1", GPUCount: 1}, now))
	_, err := o.ScheduleNext(context.Background(), now)
	must(t, err)

	// Executor has already resolved to Succeeded internally (Delay=0) by
	// the time Cancel races Tick.
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		o.Tick(context.Background(), now)
	}()
	go func() {
		defer wg.Done()
		_ = o.Cancel("wl1", "racing cancel", now)
	}()
	wg.Wait()

	wl, _ := o.Workload("wl1")
	if wl.State() != domain.WorkloadCompleted && wl.State() != domain.WorkloadCancelled {
		t.Fatalf("expected a well-defined terminal state, got %s", wl.State())
	}
	if _, ok := o.Allocation("wl1"); ok {
		t.Fatal("expected allocation released regardless of which terminal state won")
	}
}

func TestRetryDuringConcurrentScheduling(t *testing.T) {
	now := time.Now()
	o, exec := newTestOrchestrator(RetryPolicy{MaxAttempts: 5, BaseDelay: time.Second, Factor: 1, MaxDelay: time.Minute})
	o.RegisterWorker(readyWorker(t, "w1", 1, now))
	exec.Plan("wl1", ExecutionPlan{Outcome: OutcomeFail, FailureReason: "flaky", Retryable: true})
	must(t, o.Submit(domain.WorkloadRequirements{WorkloadID: "wl1", GPUCount: 1}, now))
	_, err := o.ScheduleNext(context.Background(), now)
	must(t, err)
	o.Tick(context.Background(), now) // -> RETRYING

	must(t, o.Submit(domain.WorkloadRequirements{WorkloadID: "wl2", GPUCount: 1}, now))

	later := now.Add(2 * time.Second)
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		o.Tick(context.Background(), later) // promotes wl1 RETRYING -> QUEUED
	}()
	go func() {
		defer wg.Done()
		_, _ = o.ScheduleNext(context.Background(), later) // races to schedule wl2 (or wl1 if already promoted)
	}()
	wg.Wait()

	// Regardless of interleaving, no GPU can be double-allocated: at most
	// one of wl1/wl2 can be RUNNING on the single-GPU worker at a time.
	running := 0
	for _, id := range []string{"wl1", "wl2"} {
		wl, _ := o.Workload(id)
		if wl.State() == domain.WorkloadRunning {
			running++
		}
	}
	if running > 1 {
		t.Fatalf("expected at most 1 running workload on a 1-GPU worker, got %d", running)
	}
}
