// Package agent defines the worker-agent boundary: the contract the
// controller uses to talk to a GPU worker, real or simulated, without
// caring which. See docs/architecture.md section 3 and 7.
package agent

import (
	"context"
	"time"

	"gpuforge/internal/domain"
)

// GPUDiscovery is the identity+capability information an agent reports for
// one GPU during discovery. It deliberately excludes runtime state and
// validation state — those are collected separately (CollectState,
// Validate), matching the Phase 0/1 separation of GPU concerns.
type GPUDiscovery struct {
	ID         string
	Model      string
	Capability domain.GPUCapability
}

// DiscoveryResult is what an agent reports about the worker it runs on.
type DiscoveryResult struct {
	WorkerID     string
	HardwareMode domain.HardwareMode
	GPUs         []GPUDiscovery
	Timestamp    time.Time
}

// StateSample is one GPU's freshly collected runtime state.
type StateSample struct {
	GPUID string
	State domain.GPUState
}

// ValidationSample is one GPU's freshly collected validation outcome.
type ValidationSample struct {
	GPUID  string
	Result domain.ValidationResult
}

// HeartbeatResult is the outcome of a liveness check against the agent.
// Consecutive-miss counting and quarantine decisions belong to a later
// phase's health monitor — this type only reports one liveness sample.
type HeartbeatResult struct {
	WorkerID  string
	Alive     bool
	Timestamp time.Time
}

// WorkerAgent is the single contract the controller uses to talk to a
// worker's GPUs, regardless of whether they are real or simulated. Every
// method must be safe to call repeatedly and must not silently swallow
// errors — an agent that cannot answer a question returns an error rather
// than a guessed/zero value.
type WorkerAgent interface {
	// HardwareMode reports whether this agent represents real or simulated
	// hardware. Must never change over the agent's lifetime.
	HardwareMode() domain.HardwareMode

	// Discover reports the worker's identity and its GPUs' identity+capability.
	Discover(ctx context.Context) (DiscoveryResult, error)

	// CollectState reports current runtime state for every GPU the agent owns.
	CollectState(ctx context.Context) ([]StateSample, error)

	// Validate runs capability validation for every GPU the agent owns.
	Validate(ctx context.Context) ([]ValidationSample, error)

	// Heartbeat reports whether the agent is currently reachable/alive.
	Heartbeat(ctx context.Context) (HeartbeatResult, error)
}
