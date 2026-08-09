package domain

import (
	"sync"
	"time"
)

// WorkerState is the explicit worker lifecycle state defined in
// docs/lifecycle.md. No states beyond these ten exist.
type WorkerState string

const (
	WorkerProvisioning WorkerState = "PROVISIONING"
	WorkerDiscovering  WorkerState = "DISCOVERING"
	WorkerValidating   WorkerState = "VALIDATING"
	WorkerReady        WorkerState = "READY"
	WorkerAllocated    WorkerState = "ALLOCATED"
	WorkerDraining     WorkerState = "DRAINING"
	WorkerMaintenance  WorkerState = "MAINTENANCE"
	WorkerQuarantined  WorkerState = "QUARANTINED"
	WorkerRetiring     WorkerState = "RETIRING"
	WorkerRetired      WorkerState = "RETIRED"
)

// workerTransitions is the single authoritative table of valid worker
// transitions, taken directly from docs/lifecycle.md. All transition
// validation goes through this table — no scattered rules elsewhere.
var workerTransitions = map[WorkerState]map[WorkerState]bool{
	WorkerProvisioning: {WorkerDiscovering: true},
	WorkerDiscovering:  {WorkerValidating: true, WorkerQuarantined: true},
	WorkerValidating:   {WorkerReady: true, WorkerQuarantined: true},
	WorkerReady:        {WorkerAllocated: true, WorkerDraining: true, WorkerQuarantined: true},
	WorkerAllocated:    {WorkerReady: true, WorkerDraining: true, WorkerQuarantined: true},
	WorkerDraining:     {WorkerMaintenance: true, WorkerReady: true},
	WorkerMaintenance:  {WorkerDiscovering: true, WorkerRetiring: true},
	WorkerQuarantined:  {WorkerDiscovering: true, WorkerRetiring: true},
	WorkerRetiring:     {WorkerRetired: true},
	WorkerRetired:      {},
}

// IsValidWorkerTransition reports whether from -> to is a permitted worker
// lifecycle transition.
func IsValidWorkerTransition(from, to WorkerState) bool {
	return workerTransitions[from][to]
}

// WorkerAllocatable reports whether a worker in the given state may be
// selected to receive a new allocation. Only READY and ALLOCATED workers
// may receive allocations (a worker already holding one workload may hold
// more, subject to capacity checks performed by the scheduler in a later
// phase).
func WorkerAllocatable(s WorkerState) bool {
	return s == WorkerReady || s == WorkerAllocated
}

// WorkerTransition is an immutable transition record. Kept in memory on the
// Worker; no database required per docs/lifecycle.md.
type WorkerTransition struct {
	WorkerID  string
	From      WorkerState
	To        WorkerState
	Reason    string
	Timestamp time.Time
	Source    TransitionSource
}

// Worker is the domain aggregate for a GPU host. GPUs attached to it all
// share its HardwareMode (see hardware.go) — a simulated worker cannot host
// a GPU claiming to be real, enforced at construction.
type Worker struct {
	mu           sync.Mutex
	id           string
	hardwareMode HardwareMode
	state        WorkerState
	gpus         map[string]*GPU
	history      []WorkerTransition
}

// NewWorker constructs a Worker in its initial PROVISIONING state.
func NewWorker(id string, mode HardwareMode) (*Worker, error) {
	if id == "" {
		return nil, ErrEmptyID
	}
	if !mode.Valid() {
		return nil, &HardwareModeError{Reason: "unknown hardware mode: " + string(mode)}
	}
	return &Worker{
		id:           id,
		hardwareMode: mode,
		state:        WorkerProvisioning,
		gpus:         make(map[string]*GPU),
	}, nil
}

func (w *Worker) ID() string { return w.id }

func (w *Worker) HardwareMode() HardwareMode { return w.hardwareMode }

// State returns the worker's current lifecycle state.
func (w *Worker) State() WorkerState {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.state
}

// History returns a copy of the worker's transition history in order.
func (w *Worker) History() []WorkerTransition {
	w.mu.Lock()
	defer w.mu.Unlock()
	out := make([]WorkerTransition, len(w.history))
	copy(out, w.history)
	return out
}

// Transition attempts to move the worker to state `to`. On failure the
// worker's state is left completely unchanged (no partial mutation) and a
// *WorkerTransitionError is returned.
func (w *Worker) Transition(to WorkerState, reason string, source TransitionSource, now time.Time) error {
	if reason == "" {
		return ErrReasonRequired
	}
	w.mu.Lock()
	defer w.mu.Unlock()

	from := w.state
	if !IsValidWorkerTransition(from, to) {
		return &WorkerTransitionError{WorkerID: w.id, From: from, To: to}
	}
	w.state = to
	w.history = append(w.history, WorkerTransition{
		WorkerID:  w.id,
		From:      from,
		To:        to,
		Reason:    reason,
		Timestamp: now,
		Source:    source,
	})
	return nil
}

// AddGPU attaches a GPU to the worker. The GPU's hardware mode must match
// the worker's hardware mode — this is the enforcement point for the
// "simulated hardware never reports as real" invariant.
func (w *Worker) AddGPU(g *GPU) error {
	if g == nil {
		return &AllocationError{Reason: "nil GPU"}
	}
	if g.HardwareMode != w.hardwareMode {
		return &HardwareModeError{Reason: "GPU hardware mode " + string(g.HardwareMode) +
			" does not match worker hardware mode " + string(w.hardwareMode)}
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	w.gpus[g.ID] = g
	return nil
}

// GPU returns the GPU with the given ID, and whether it was found.
func (w *Worker) GPU(id string) (*GPU, bool) {
	w.mu.Lock()
	defer w.mu.Unlock()
	g, ok := w.gpus[id]
	return g, ok
}

// GPUs returns a copy of the worker's attached GPUs.
func (w *Worker) GPUs() []*GPU {
	w.mu.Lock()
	defer w.mu.Unlock()
	out := make([]*GPU, 0, len(w.gpus))
	for _, g := range w.gpus {
		out = append(out, g)
	}
	return out
}

// UpdateGPUState applies a freshly collected runtime state sample to the
// named GPU. This is the only sanctioned way to mutate a GPU's runtime
// state after attachment — callers (the agent layer, in a later phase the
// health monitor) must not reach into a *GPU's fields directly, since that
// would bypass the worker's mutex and race with concurrent readers.
func (w *Worker) UpdateGPUState(gpuID string, state GPUState) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	g, ok := w.gpus[gpuID]
	if !ok {
		return &AllocationError{Reason: "GPU " + gpuID + " does not belong to worker " + w.id}
	}
	g.State = state
	return nil
}

// UpdateGPUValidation applies a validation result to the named GPU. Same
// mutex-guarded-mutation rationale as UpdateGPUState.
func (w *Worker) UpdateGPUValidation(gpuID string, result ValidationResult) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	g, ok := w.gpus[gpuID]
	if !ok {
		return &AllocationError{Reason: "GPU " + gpuID + " does not belong to worker " + w.id}
	}
	g.Validation = result
	return nil
}

// MarkGPUsAllocated marks every named GPU ALLOCATED, atomically: if any GPU
// is missing or already ALLOCATED, no GPU's state is changed and an error
// is returned. Used by NewAllocation so a scheduler decision can never
// double-book a GPU that's already held by another allocation.
func (w *Worker) MarkGPUsAllocated(gpuIDs []string) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	for _, id := range gpuIDs {
		g, ok := w.gpus[id]
		if !ok {
			return &AllocationError{Reason: "GPU " + id + " does not belong to worker " + w.id}
		}
		if g.AllocationState == GPUAllocated {
			return &AllocationError{Reason: "GPU " + id + " is already allocated"}
		}
	}
	for _, id := range gpuIDs {
		w.gpus[id].AllocationState = GPUAllocated
	}
	return nil
}

// MarkGPUsReleased marks every named GPU FREE, atomically. Used by
// Allocation.Release.
func (w *Worker) MarkGPUsReleased(gpuIDs []string) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	for _, id := range gpuIDs {
		if _, ok := w.gpus[id]; !ok {
			return &AllocationError{Reason: "GPU " + id + " does not belong to worker " + w.id}
		}
	}
	for _, id := range gpuIDs {
		w.gpus[id].AllocationState = GPUFree
	}
	return nil
}
