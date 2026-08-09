package domain

import "time"

// GPUCapability is static hardware capability — the things a validation
// pass checks and that do not change while the GPU is attached. Kept
// separate from runtime state per docs/architecture.md's data model.
type GPUCapability struct {
	ComputeCapability string // e.g. "sm_80"
	DriverVersion     string
	RuntimeVersion    string
	MemoryBytes       uint64 // total capacity
}

// GPUState is runtime state that changes continuously. Kept separate from
// GPUCapability (static) and from allocation state (see Allocation) per
// docs/architecture.md's explicit separation of identity/capability/
// runtime-state/validation-state/allocation-state.
type GPUState struct {
	UtilizationPercent   float64
	AvailableMemoryBytes uint64
	LastHeartbeat        time.Time
}

// ValidationStatus distinguishes the phases of a capability validation pass.
type ValidationStatus string

const (
	ValidationPending ValidationStatus = "PENDING"
	ValidationPassed  ValidationStatus = "PASSED"
	ValidationFailed  ValidationStatus = "FAILED"
)

// ValidationResult records the outcome of validating a GPU's capability.
// A Failed result must always carry a Reason explaining what failed;
// actually running CUDA/NCCL checks is deferred to a later phase — this
// type only represents the outcome.
type ValidationResult struct {
	Status    ValidationStatus
	Reason    string
	Timestamp time.Time
}

// NewPendingValidation returns the initial validation state for a newly
// discovered GPU.
func NewPendingValidation() ValidationResult {
	return ValidationResult{Status: ValidationPending}
}

// Pass returns a Passed validation result at the given time.
func Pass(now time.Time) ValidationResult {
	return ValidationResult{Status: ValidationPassed, Timestamp: now}
}

// Fail returns a Failed validation result. reason is required so failures
// are always explainable (per docs/lifecycle.md, no bare rejections).
func Fail(reason string, now time.Time) (ValidationResult, error) {
	if reason == "" {
		return ValidationResult{}, ErrValidationReasonRequired
	}
	return ValidationResult{Status: ValidationFailed, Reason: reason, Timestamp: now}, nil
}

// GPUTopology is static placement/connectivity information used by
// topology-aware scheduling (docs/scheduling-engine.md). It is discovered
// alongside capability, never invented: an empty NodeID/NVLinkGroup means
// "unknown," and scheduling policies must treat unknown topology as a
// defined fallback rather than assuming connectivity.
type GPUTopology struct {
	// NodeID identifies the physical/virtual host the GPU lives on. In the
	// current single-worker-per-allocation model this is redundant with
	// GPU.WorkerID (kept as its own field so multi-node topology, a
	// documented future integration point, doesn't require a schema change).
	NodeID string
	// NVLinkGroup identifies a set of GPUs on the same worker connected by a
	// fast interconnect (e.g. NVLink). GPUs with the same non-empty
	// NVLinkGroup value are assumed directly connected; empty means unknown.
	NVLinkGroup string
}

// GPUAllocationState distinguishes a GPU currently held by an Allocation
// from one that is free to be selected by the scheduler. This is the fifth
// concept docs/architecture.md's data model calls out (identity/capability/
// runtime-state/validation-state/allocation-state) — added here in Phase 3
// because it has no consumer until the scheduler needs to know which GPUs
// are actually selectable.
type GPUAllocationState string

const (
	GPUFree      GPUAllocationState = "FREE"
	GPUAllocated GPUAllocationState = "ALLOCATED"
)

// GPU is the aggregate of a single device's identity, capability, runtime
// state, validation state, topology, and allocation state.
type GPU struct {
	// Identity
	ID           string // unique within the cluster
	WorkerID     string
	Model        string
	HardwareMode HardwareMode

	// Capability (static)
	Capability GPUCapability

	// Topology (static, discovered — never invented; see GPUTopology)
	Topology GPUTopology

	// Runtime state
	State GPUState

	// Validation state
	Validation ValidationResult

	// Allocation state — mutate only via Worker.MarkGPUsAllocated /
	// Worker.MarkGPUsReleased, which hold the worker's mutex; never set this
	// field directly on an attached GPU.
	AllocationState GPUAllocationState
}

// NewGPU constructs a GPU in PENDING validation state and FREE allocation
// state. mode must match the owning worker's hardware mode (enforced by
// Worker.AddGPU); passing it here explicitly (rather than letting a GPU
// pick its own mode independently) is what makes "simulated GPU claims to
// be real" a construction-time, testable invariant rather than a runtime
// bug class.
func NewGPU(id, workerID, model string, mode HardwareMode, cap GPUCapability) (*GPU, error) {
	if id == "" || workerID == "" {
		return nil, ErrEmptyID
	}
	if !mode.Valid() {
		return nil, &HardwareModeError{Reason: "unknown hardware mode: " + string(mode)}
	}
	return &GPU{
		ID:              id,
		WorkerID:        workerID,
		Model:           model,
		HardwareMode:    mode,
		Capability:      cap,
		Validation:      NewPendingValidation(),
		AllocationState: GPUFree,
	}, nil
}
