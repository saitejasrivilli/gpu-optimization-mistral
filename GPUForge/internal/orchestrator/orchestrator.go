package orchestrator

import (
	"context"
	"errors"
	"sync"
	"time"

	"gpuforge/internal/domain"
	"gpuforge/internal/scheduler"
)

// Orchestrator is the control-plane workflow: Queue -> Scheduler.Place ->
// Allocation -> Executor -> workload state -> release. It owns all side
// effects; scheduler.Scheduler stays pure (see docs/orchestration.md).
//
// Concurrency model: one coarse mutex guards every state-mutating method
// (Submit, ScheduleNext, Tick, Cancel, DrainWorker, CompleteDraining). This
// is a deliberate simplicity choice for an in-memory, single-process
// control plane (explicitly not a distributed queue/consensus system, per
// the portfolio boundary) — see docs/orchestration.md for the tradeoff and
// why it's still safe to exercise real concurrency in tests (goroutines
// still race to acquire the lock; domain-level invariants like atomic GPU
// allocation provide defense in depth underneath it).
type Orchestrator struct {
	mu sync.Mutex

	scheduler   scheduler.Scheduler
	executor    Executor
	retryPolicy RetryPolicy

	workers      map[string]*domain.Worker
	workloads    map[string]*domain.Workload
	requirements map[string]domain.WorkloadRequirements
	allocations  map[string]*domain.Allocation
	queue        *Queue
	running      map[string]struct{}
	retrying     map[string]time.Time
	// attempts counts execution attempts (Failed outcomes), used against
	// retryPolicy for RETRYING eligibility. Scheduling-attempt counts live
	// on the QueueItem itself (reset on every fresh retry pass) since
	// they're a different budget: "how many times have we tried to find a
	// GPU for this queue entry", not "how many times has it run and failed".
	attempts map[string]int
}

// New constructs an Orchestrator. sched and exec must be non-nil.
func New(sched scheduler.Scheduler, exec Executor, policy RetryPolicy) *Orchestrator {
	return &Orchestrator{
		scheduler:    sched,
		executor:     exec,
		retryPolicy:  policy,
		workers:      make(map[string]*domain.Worker),
		workloads:    make(map[string]*domain.Workload),
		requirements: make(map[string]domain.WorkloadRequirements),
		allocations:  make(map[string]*domain.Allocation),
		queue:        NewQueue(),
		running:      make(map[string]struct{}),
		retrying:     make(map[string]time.Time),
		attempts:     make(map[string]int),
	}
}

// RegisterWorker adds w to the pool the scheduler considers. Re-registering
// the same ID overwrites the previous entry (idempotent by design — the
// caller, typically after agent.Register+RunValidation from Phase 2, always
// hands over the current worker object).
func (o *Orchestrator) RegisterWorker(w *domain.Worker) {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.workers[w.ID()] = w
}

// Worker returns the registered worker for id, if any.
func (o *Orchestrator) Worker(id string) (*domain.Worker, bool) {
	o.mu.Lock()
	defer o.mu.Unlock()
	w, ok := o.workers[id]
	return w, ok
}

// Workload returns the tracked workload for id, if any.
func (o *Orchestrator) Workload(id string) (*domain.Workload, bool) {
	o.mu.Lock()
	defer o.mu.Unlock()
	w, ok := o.workloads[id]
	return w, ok
}

// Allocation returns the active allocation record for a workload, if any
// (nil after release).
func (o *Orchestrator) Allocation(workloadID string) (*domain.Allocation, bool) {
	o.mu.Lock()
	defer o.mu.Unlock()
	a, ok := o.allocations[workloadID]
	return a, ok
}

// QueueLen reports how many workloads are currently queued.
func (o *Orchestrator) QueueLen() int { return o.queue.Len() }

// RunningCount reports how many workloads are currently RUNNING.
func (o *Orchestrator) RunningCount() int {
	o.mu.Lock()
	defer o.mu.Unlock()
	return len(o.running)
}

// Submit validates req, creates its Workload (SUBMITTED -> QUEUED), and
// enqueues it. Returns ErrDuplicateWorkload if req.WorkloadID was already
// submitted — Submit is not implicitly idempotent, since a second Submit
// with different requirements would otherwise silently overwrite the first.
func (o *Orchestrator) Submit(req domain.WorkloadRequirements, now time.Time) error {
	if err := req.Validate(); err != nil {
		return err
	}
	o.mu.Lock()
	defer o.mu.Unlock()

	if _, exists := o.workloads[req.WorkloadID]; exists {
		return ErrDuplicateWorkload
	}
	wl, err := domain.NewWorkload(req.WorkloadID)
	if err != nil {
		return err
	}
	if err := wl.Transition(domain.WorkloadQueued, "admitted", domain.SourceAdmissionControl, now); err != nil {
		return err
	}
	o.workloads[req.WorkloadID] = wl
	o.requirements[req.WorkloadID] = req
	o.queue.Enqueue(QueueItem{Requirements: req, EnqueuedAt: now})
	return nil
}

// ScheduleNext dequeues the highest-priority workload and attempts to place
// it: builds a fresh ClusterSnapshot, calls the pure scheduler, constructs
// an Allocation (atomically reserving the GPUs — see
// domain.Worker.MarkGPUsAllocated), and starts the executor. Returns
// (true, nil) on a successful start. Returns (false, err) if the queue was
// empty, the item was stale, or scheduling/allocation/start failed — in
// every failure case the workload is left in a valid, well-defined state
// (usually back in the queue with an incremented attempt count, or
// CANCELLED if attempts are exhausted) and no partial allocation survives.
func (o *Orchestrator) ScheduleNext(ctx context.Context, now time.Time) (bool, error) {
	o.mu.Lock()
	defer o.mu.Unlock()

	item, ok := o.queue.Dequeue()
	if !ok {
		return false, ErrQueueEmpty
	}

	wl, exists := o.workloads[item.Requirements.WorkloadID]
	if !exists {
		return false, ErrWorkloadNotFound
	}
	if wl.State() != domain.WorkloadQueued {
		// Stale: the workload moved on (e.g. cancelled) without going
		// through queue.Remove — drop the item rather than act on it.
		return false, ErrStaleWorkload
	}

	snapshot := domain.NewClusterSnapshot(o.workerSlice(), now)
	placement, err := o.scheduler.Place(ctx, item.Requirements, snapshot, now)
	if err != nil {
		return false, o.requeueOrGiveUp(item, wl, now, err)
	}

	worker, ok := o.workers[placement.WorkerID]
	if !ok {
		return false, o.requeueOrGiveUp(item, wl, now, ErrWorkerNotFound)
	}

	alloc, err := domain.NewAllocation(wl, worker, placement.GPUIDs, now)
	if err != nil {
		// Concurrent contention: another ScheduleNext call (or a direct
		// caller) claimed one of these GPUs between the snapshot and this
		// allocation attempt. Requeue rather than fail the workload — the
		// GPU it wanted was real, just taken; try again.
		return false, o.requeueOrGiveUp(item, wl, now, err)
	}
	o.allocations[wl.ID()] = alloc

	if err := wl.Transition(domain.WorkloadScheduled, placement.Reason, domain.SourceScheduler, now); err != nil {
		_ = alloc.Release("workload state transition failed after allocation", now)
		delete(o.allocations, wl.ID())
		return false, err
	}

	execReq := ExecutionRequest{WorkloadID: wl.ID(), WorkerID: worker.ID(), GPUIDs: placement.GPUIDs}
	if err := o.executor.Start(ctx, execReq, now); err != nil {
		_ = alloc.Release("executor start failed: "+err.Error(), now)
		delete(o.allocations, wl.ID())
		_ = wl.Transition(domain.WorkloadQueued, "executor start failed: "+err.Error(), domain.SourceAgentReport, now)
		item.Attempts++
		o.queue.Enqueue(item)
		return false, err
	}

	if err := wl.Transition(domain.WorkloadRunning, "execution started", domain.SourceAgentReport, now); err != nil {
		// Executor already started; we cannot un-start it (Start has no
		// undo). Record the inconsistency rather than lose track of the
		// running execution — the workload stays SCHEDULED and Tick will
		// still observe and act on the executor's real status.
		return false, err
	}
	o.running[wl.ID()] = struct{}{}
	return true, nil
}

// requeueOrGiveUp increments the item's scheduling-attempt count and either
// requeues it or, once retryPolicy's budget is exhausted, cancels the
// workload. QUEUED's only valid domain transitions are to SCHEDULED or
// CANCELLED (see docs/lifecycle.md) — there is no QUEUED -> FAILED edge, so
// "permanently unschedulable" is represented as CANCELLED, not FAILED.
func (o *Orchestrator) requeueOrGiveUp(item QueueItem, wl *domain.Workload, now time.Time, cause error) error {
	item.Attempts++
	if o.retryPolicy.Exhausted(item.Attempts) {
		_ = wl.Transition(domain.WorkloadCancelled, "unschedulable after max attempts: "+cause.Error(), domain.SourceScheduler, now)
		return cause
	}
	o.queue.Enqueue(item)
	return cause
}

// Tick advances time-dependent bookkeeping: promotes RETRYING workloads
// whose backoff has elapsed back to QUEUED, and polls the executor for
// every RUNNING workload, applying the resulting state transition and
// releasing the allocation on any terminal outcome. Call it periodically
// (or after every simulated time advance in tests) — there is no
// background goroutine; the caller drives time explicitly, per Phase 4's
// "avoid sleeps, use deterministic clocks" requirement.
func (o *Orchestrator) Tick(ctx context.Context, now time.Time) {
	o.mu.Lock()
	defer o.mu.Unlock()

	for id, at := range o.retrying {
		if now.Before(at) {
			continue
		}
		delete(o.retrying, id)
		wl, ok := o.workloads[id]
		if !ok || wl.State() != domain.WorkloadRetrying {
			continue
		}
		req := o.requirements[id]
		if err := wl.Transition(domain.WorkloadQueued, "retry backoff elapsed", domain.SourceScheduler, now); err != nil {
			continue
		}
		o.queue.Enqueue(QueueItem{Requirements: req, EnqueuedAt: now})
	}

	for id := range o.running {
		wl, ok := o.workloads[id]
		if !ok {
			delete(o.running, id)
			continue
		}
		status, err := o.executor.Status(ctx, id, now)
		if err != nil {
			continue
		}
		switch status.Phase {
		case ExecutionRunning:
			continue
		case ExecutionSucceeded:
			o.releaseAllocation(id, "workload completed", now)
			_ = wl.Transition(domain.WorkloadCompleted, "execution succeeded", domain.SourceAgentReport, now)
			delete(o.running, id)
		case ExecutionFailed:
			o.releaseAllocation(id, "execution failed: "+status.Reason, now)
			_ = wl.Transition(domain.WorkloadFailed, status.Reason, domain.SourceAgentReport, now)
			delete(o.running, id)
			o.attempts[id]++
			if status.Retryable && !o.retryPolicy.Exhausted(o.attempts[id]) {
				_ = wl.Transition(domain.WorkloadRetrying, "retry scheduled", domain.SourceScheduler, now)
				o.retrying[id] = now.Add(o.retryPolicy.NextDelay(o.attempts[id]))
			} else {
				_ = wl.Transition(domain.WorkloadCancelled, "retry budget exhausted or non-retryable failure", domain.SourceScheduler, now)
			}
		case ExecutionCancelled:
			o.releaseAllocation(id, "execution cancelled", now)
			if wl.State() == domain.WorkloadRunning {
				_ = wl.Transition(domain.WorkloadCancelled, "execution cancelled", domain.SourceClient, now)
			}
			delete(o.running, id)
		}
	}
}

// releaseAllocation releases the workload's allocation, if any, tolerating
// domain.ErrAlreadyReleased (a benign no-op — the allocation was already
// released by another code path, e.g. Cancel racing Tick under the same
// lock in sequential order) rather than treating it as a fatal error.
func (o *Orchestrator) releaseAllocation(workloadID, reason string, now time.Time) {
	alloc, ok := o.allocations[workloadID]
	if !ok {
		return
	}
	if err := alloc.Release(reason, now); err != nil && !errors.Is(err, domain.ErrAlreadyReleased) {
		return
	}
	delete(o.allocations, workloadID)
}

// Cancel requests a workload stop, from whatever state it's in.
// Idempotent: cancelling an already-COMPLETED or already-CANCELLED
// workload is a no-op success, not an error, since the caller's intent
// ("this should not be running") is already satisfied. See
// docs/orchestration.md for the full per-state table, including why
// SCHEDULED/RETRYING route through an extra valid intermediate transition
// rather than needing a direct-to-CANCELLED domain edge.
func (o *Orchestrator) Cancel(workloadID, reason string, now time.Time) error {
	o.mu.Lock()
	defer o.mu.Unlock()

	wl, ok := o.workloads[workloadID]
	if !ok {
		return ErrWorkloadNotFound
	}

	switch wl.State() {
	case domain.WorkloadCompleted, domain.WorkloadCancelled:
		return nil // idempotent no-op

	case domain.WorkloadQueued:
		o.queue.Remove(workloadID)
		return wl.Transition(domain.WorkloadCancelled, reason, domain.SourceClient, now)

	case domain.WorkloadScheduled:
		// Not observable externally given ScheduleNext holds o.mu for its
		// entire allocate+start sequence — included for defense in depth.
		o.releaseAllocation(workloadID, reason, now)
		o.queue.Remove(workloadID)
		if err := wl.Transition(domain.WorkloadQueued, "cancelled while scheduled", domain.SourceClient, now); err != nil {
			return err
		}
		return wl.Transition(domain.WorkloadCancelled, reason, domain.SourceClient, now)

	case domain.WorkloadRunning:
		if err := o.executor.Cancel(context.Background(), workloadID, now); err != nil &&
			!errors.Is(err, ErrCannotCancelTerminal) {
			return err
		}
		o.releaseAllocation(workloadID, reason, now)
		delete(o.running, workloadID)
		if err := wl.Transition(domain.WorkloadCancelled, reason, domain.SourceClient, now); err != nil {
			// The execution finished (succeeded/failed) in the same instant
			// as the cancel request; a later Tick already will have (or
			// will) resolve the workload's real terminal state. Treat as
			// idempotent rather than surfacing a race as an error.
			return nil
		}
		return nil

	case domain.WorkloadFailed:
		return wl.Transition(domain.WorkloadCancelled, reason, domain.SourceClient, now)

	case domain.WorkloadRetrying:
		delete(o.retrying, workloadID)
		o.queue.Remove(workloadID)
		if err := wl.Transition(domain.WorkloadQueued, "cancelled during retry backoff", domain.SourceClient, now); err != nil {
			return err
		}
		return wl.Transition(domain.WorkloadCancelled, reason, domain.SourceClient, now)

	default: // SUBMITTED: transient, never observed externally in this design
		return ErrNotCancellable
	}
}

// DrainWorker transitions a worker into DRAINING. Once draining, the
// scheduler's existing compatibility gate (domain.WorkerAllocatable) simply
// stops considering it — no scheduler change was needed. Workloads already
// running on it are left to run to completion; GPUForge does not implement
// migration or checkpointing (see docs/orchestration.md's explicit
// limitation).
func (o *Orchestrator) DrainWorker(workerID, reason string, now time.Time) error {
	o.mu.Lock()
	defer o.mu.Unlock()
	w, ok := o.workers[workerID]
	if !ok {
		return ErrWorkerNotFound
	}
	return w.Transition(domain.WorkerDraining, reason, domain.SourceOperator, now)
}

// CompleteDraining transitions a DRAINING worker to MAINTENANCE, but only
// once it holds no active allocations — otherwise ErrDrainIncomplete.
func (o *Orchestrator) CompleteDraining(workerID, reason string, now time.Time) error {
	o.mu.Lock()
	defer o.mu.Unlock()
	w, ok := o.workers[workerID]
	if !ok {
		return ErrWorkerNotFound
	}
	for _, alloc := range o.allocations {
		if alloc.WorkerID == workerID && alloc.State() == domain.AllocationActive {
			return ErrDrainIncomplete
		}
	}
	return w.Transition(domain.WorkerMaintenance, reason, domain.SourceOperator, now)
}

func (o *Orchestrator) workerSlice() []*domain.Worker {
	out := make([]*domain.Worker, 0, len(o.workers))
	for _, w := range o.workers {
		out = append(out, w)
	}
	return out
}
