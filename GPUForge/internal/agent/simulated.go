package agent

import (
	"context"
	"errors"
	"fmt"
	"math/rand"
	"sync"
	"time"

	"gpuforge/internal/domain"
)

// ErrNoGPUs is returned by NewSimulatedAgent when configured with zero GPUs.
var ErrNoGPUs = errors.New("agent: simulated worker requires at least one GPU")

// ErrEmptyWorkerID is returned when a simulated agent is configured without
// a worker ID.
var ErrEmptyWorkerID = errors.New("agent: worker id required")

// GPUSpec is the configured shape of one simulated GPU. Capability is fixed
// for the agent's lifetime; only runtime state varies call-to-call, driven
// by the agent's seeded RNG.
type GPUSpec struct {
	Model      string
	Capability domain.GPUCapability
}

// SimulatedConfig configures a deterministic simulated worker. Two agents
// built from the same config (same Seed, same GPUs) produce byte-identical
// Discover results and identical CollectState sequences call-for-call —
// this is what makes benchmark runs reproducible per docs/benchmark-plan.md.
type SimulatedConfig struct {
	WorkerID string
	Seed     int64
	GPUs     []GPUSpec

	// FailValidationReason, if non-empty, makes Validate report every GPU as
	// FAILED with this reason. Lets tests (and later, the health monitor)
	// exercise the QUARANTINED path deterministically.
	FailValidationReason string

	// SimulateUnreachable, if true, makes Heartbeat report the worker as not
	// alive. Lets tests exercise failure-detection wiring deterministically.
	SimulateUnreachable bool
}

// SimulatedAgent is a deterministic, seeded stand-in for a real GPU worker.
// It never touches real hardware and always reports
// domain.HardwareModeSimulated — see hardware_mode.go for the invariant
// this feeds into.
type SimulatedAgent struct {
	cfg    SimulatedConfig
	gpuIDs []string

	mu  sync.Mutex
	rng *rand.Rand
}

// NewSimulatedAgent constructs a simulated worker agent. GPU IDs are derived
// deterministically from the worker ID and index, so repeated construction
// with the same config yields the same GPU IDs.
func NewSimulatedAgent(cfg SimulatedConfig) (*SimulatedAgent, error) {
	if cfg.WorkerID == "" {
		return nil, ErrEmptyWorkerID
	}
	if len(cfg.GPUs) == 0 {
		return nil, ErrNoGPUs
	}
	ids := make([]string, len(cfg.GPUs))
	for i := range cfg.GPUs {
		ids[i] = fmt.Sprintf("%s-gpu-%d", cfg.WorkerID, i)
	}
	return &SimulatedAgent{
		cfg:    cfg,
		gpuIDs: ids,
		rng:    rand.New(rand.NewSource(cfg.Seed)),
	}, nil
}

func (a *SimulatedAgent) HardwareMode() domain.HardwareMode { return domain.HardwareModeSimulated }

// Discover returns the agent's fixed identity+capability set. Deterministic:
// does not consume RNG state, so calling it any number of times, in any
// order relative to CollectState, never changes.
func (a *SimulatedAgent) Discover(ctx context.Context) (DiscoveryResult, error) {
	if err := ctx.Err(); err != nil {
		return DiscoveryResult{}, err
	}
	gpus := make([]GPUDiscovery, len(a.cfg.GPUs))
	for i, spec := range a.cfg.GPUs {
		gpus[i] = GPUDiscovery{
			ID:         a.gpuIDs[i],
			Model:      spec.Model,
			Capability: spec.Capability,
		}
	}
	return DiscoveryResult{
		WorkerID:     a.cfg.WorkerID,
		HardwareMode: domain.HardwareModeSimulated,
		GPUs:         gpus,
		Timestamp:    time.Now(),
	}, nil
}

// CollectState generates the next deterministic runtime-state sample for
// each GPU, drawn from the agent's seeded RNG. Two agents built from
// identical configs produce an identical sequence of samples across
// successive calls.
func (a *SimulatedAgent) CollectState(ctx context.Context) ([]StateSample, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	a.mu.Lock()
	defer a.mu.Unlock()

	now := time.Now()
	samples := make([]StateSample, len(a.cfg.GPUs))
	for i, spec := range a.cfg.GPUs {
		util := a.rng.Float64() * 100
		freeFraction := a.rng.Float64()
		available := uint64(freeFraction * float64(spec.Capability.MemoryBytes))
		samples[i] = StateSample{
			GPUID: a.gpuIDs[i],
			State: domain.GPUState{
				UtilizationPercent:   util,
				AvailableMemoryBytes: available,
				LastHeartbeat:        now,
			},
		}
	}
	return samples, nil
}

// Validate reports PASSED for every GPU unless the agent is configured with
// FailValidationReason, in which case every GPU reports FAILED with that
// reason. Real capability checking (CUDA/NCCL) is deferred to a later phase
// — this only represents the outcome.
func (a *SimulatedAgent) Validate(ctx context.Context) ([]ValidationSample, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	now := time.Now()
	samples := make([]ValidationSample, len(a.gpuIDs))
	for i, id := range a.gpuIDs {
		var result domain.ValidationResult
		if a.cfg.FailValidationReason != "" {
			var err error
			result, err = domain.Fail(a.cfg.FailValidationReason, now)
			if err != nil {
				return nil, err
			}
		} else {
			result = domain.Pass(now)
		}
		samples[i] = ValidationSample{GPUID: id, Result: result}
	}
	return samples, nil
}

// Heartbeat reports Alive: true unless the agent is configured with
// SimulateUnreachable, letting tests exercise failure-detection wiring
// deterministically without a real network partition.
func (a *SimulatedAgent) Heartbeat(ctx context.Context) (HeartbeatResult, error) {
	if err := ctx.Err(); err != nil {
		return HeartbeatResult{}, err
	}
	return HeartbeatResult{
		WorkerID:  a.cfg.WorkerID,
		Alive:     !a.cfg.SimulateUnreachable,
		Timestamp: time.Now(),
	}, nil
}

var _ WorkerAgent = (*SimulatedAgent)(nil)
