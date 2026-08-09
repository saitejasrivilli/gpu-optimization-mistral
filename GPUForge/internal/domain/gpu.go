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

// GPU is the aggregate of a single device's identity, capability, runtime
// state, and validation state. Allocation state is represented separately
// by Allocation, not embedded here, so a GPU never needs to know about the
// workload occupying it — that link lives in the Allocation.
type GPU struct {
	// Identity
	ID           string // unique within the cluster
	WorkerID     string
	Model        string
	HardwareMode HardwareMode

	// Capability (static)
	Capability GPUCapability

	// Runtime state
	State GPUState

	// Validation state
	Validation ValidationResult
}

// NewGPU constructs a GPU in PENDING validation state. mode must match the
// owning worker's hardware mode (enforced by Worker.AddGPU); passing it here
// explicitly (rather than letting a GPU pick its own mode independently)
// is what makes "simulated GPU claims to be real" a construction-time,
// testable invariant rather than a runtime bug class.
func NewGPU(id, workerID, model string, mode HardwareMode, cap GPUCapability) (*GPU, error) {
	if id == "" || workerID == "" {
		return nil, ErrEmptyID
	}
	if !mode.Valid() {
		return nil, &HardwareModeError{Reason: "unknown hardware mode: " + string(mode)}
	}
	return &GPU{
		ID:           id,
		WorkerID:     workerID,
		Model:        model,
		HardwareMode: mode,
		Capability:   cap,
		Validation:   NewPendingValidation(),
	}, nil
}
