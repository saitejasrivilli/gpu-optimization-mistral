package agent

import (
	"context"
	"fmt"
	"time"

	"gpuforge/internal/domain"
)

// Register discovers a worker through its agent and constructs the
// corresponding domain.Worker (PROVISIONING -> DISCOVERING, with GPUs
// attached). It is the sole bridge between the agent boundary and the
// domain lifecycle for the discovery step — nothing else in this package
// or the domain package duplicates this wiring.
func Register(ctx context.Context, ag WorkerAgent, now time.Time) (*domain.Worker, error) {
	result, err := ag.Discover(ctx)
	if err != nil {
		return nil, fmt.Errorf("agent: register: discovery failed: %w", err)
	}
	if result.HardwareMode != ag.HardwareMode() {
		return nil, &domain.HardwareModeError{Reason: fmt.Sprintf(
			"agent reported hardware mode %q in discovery result but agent itself is %q",
			result.HardwareMode, ag.HardwareMode())}
	}

	w, err := domain.NewWorker(result.WorkerID, result.HardwareMode)
	if err != nil {
		return nil, fmt.Errorf("agent: register: %w", err)
	}

	for _, gd := range result.GPUs {
		g, err := domain.NewGPU(gd.ID, result.WorkerID, gd.Model, result.HardwareMode, gd.Capability)
		if err != nil {
			return nil, fmt.Errorf("agent: register: constructing GPU %s: %w", gd.ID, err)
		}
		if err := w.AddGPU(g); err != nil {
			return nil, fmt.Errorf("agent: register: attaching GPU %s: %w", gd.ID, err)
		}
	}

	if err := w.Transition(domain.WorkerDiscovering, "agent reported discovery result", domain.SourceAgentReport, now); err != nil {
		return nil, fmt.Errorf("agent: register: %w", err)
	}
	return w, nil
}

// RunValidation runs the agent's capability validation, applies each
// result to the corresponding GPU, and drives the worker's lifecycle
// accordingly: VALIDATING -> READY if every GPU passed, VALIDATING ->
// QUARANTINED (with the first failure's reason) otherwise. The worker must
// already be in DISCOVERING state (Register leaves it there); this
// function performs the DISCOVERING -> VALIDATING step itself.
func RunValidation(ctx context.Context, w *domain.Worker, ag WorkerAgent, now time.Time) error {
	if err := w.Transition(domain.WorkerValidating, "validation started", domain.SourceAgentReport, now); err != nil {
		return fmt.Errorf("agent: run validation: %w", err)
	}

	samples, err := ag.Validate(ctx)
	if err != nil {
		if qerr := w.Transition(domain.WorkerQuarantined, "validation could not be performed: "+err.Error(), domain.SourceAgentReport, now); qerr != nil {
			return fmt.Errorf("agent: run validation: validate call failed (%v) and quarantine transition also failed: %w", err, qerr)
		}
		return fmt.Errorf("agent: run validation: %w", err)
	}

	firstFailure := ""
	for _, s := range samples {
		if err := w.UpdateGPUValidation(s.GPUID, s.Result); err != nil {
			return fmt.Errorf("agent: run validation: %w", err)
		}
		if s.Result.Status == domain.ValidationFailed && firstFailure == "" {
			firstFailure = s.Result.Reason
		}
	}

	if firstFailure != "" {
		return w.Transition(domain.WorkerQuarantined, "capability validation failed: "+firstFailure, domain.SourceAgentReport, now)
	}
	return w.Transition(domain.WorkerReady, "capability validation passed", domain.SourceAgentReport, now)
}

// CollectAndApplyState pulls a fresh runtime-state sample from the agent
// and applies it to the worker's GPUs. It does not perform any lifecycle
// transition — runtime state changes alone never move a worker between
// lifecycle states; only validation and heartbeat outcomes do.
func CollectAndApplyState(ctx context.Context, w *domain.Worker, ag WorkerAgent) error {
	samples, err := ag.CollectState(ctx)
	if err != nil {
		return fmt.Errorf("agent: collect state: %w", err)
	}
	for _, s := range samples {
		if err := w.UpdateGPUState(s.GPUID, s.State); err != nil {
			return fmt.Errorf("agent: collect state: %w", err)
		}
	}
	return nil
}
