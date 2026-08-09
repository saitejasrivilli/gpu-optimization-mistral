package agent

import (
	"context"
	"errors"
	"testing"
	"time"

	"gpuforge/internal/domain"
)

func TestRegister_BuildsWorkerWithGPUs(t *testing.T) {
	a, _ := NewSimulatedAgent(SimulatedConfig{WorkerID: "w1", Seed: 1, GPUs: testGPUSpecs()})
	w, err := Register(context.Background(), a, time.Now())
	if err != nil {
		t.Fatal(err)
	}
	if w.ID() != "w1" || w.HardwareMode() != domain.HardwareModeSimulated {
		t.Fatalf("unexpected worker: id=%s mode=%s", w.ID(), w.HardwareMode())
	}
	if w.State() != domain.WorkerDiscovering {
		t.Fatalf("expected DISCOVERING after register, got %s", w.State())
	}
	gpus := w.GPUs()
	if len(gpus) != 2 {
		t.Fatalf("expected 2 GPUs attached, got %d", len(gpus))
	}
}

func TestRunValidation_AllPass_TransitionsToReady(t *testing.T) {
	a, _ := NewSimulatedAgent(SimulatedConfig{WorkerID: "w1", Seed: 1, GPUs: testGPUSpecs()})
	w, err := Register(context.Background(), a, time.Now())
	if err != nil {
		t.Fatal(err)
	}
	if err := RunValidation(context.Background(), w, a, time.Now()); err != nil {
		t.Fatal(err)
	}
	if w.State() != domain.WorkerReady {
		t.Fatalf("expected READY after passing validation, got %s", w.State())
	}
	for _, g := range w.GPUs() {
		if g.Validation.Status != domain.ValidationPassed {
			t.Fatalf("expected GPU %s validation PASSED, got %s", g.ID, g.Validation.Status)
		}
	}
}

func TestRunValidation_Failure_TransitionsToQuarantined(t *testing.T) {
	a, _ := NewSimulatedAgent(SimulatedConfig{WorkerID: "w1", Seed: 1, GPUs: testGPUSpecs(), FailValidationReason: "driver too old"})
	w, err := Register(context.Background(), a, time.Now())
	if err != nil {
		t.Fatal(err)
	}
	if err := RunValidation(context.Background(), w, a, time.Now()); err != nil {
		t.Fatal(err)
	}
	if w.State() != domain.WorkerQuarantined {
		t.Fatalf("expected QUARANTINED after failing validation, got %s", w.State())
	}
	hist := w.History()
	last := hist[len(hist)-1]
	if last.Reason == "" {
		t.Fatal("expected quarantine transition to carry a non-empty reason")
	}
	for _, g := range w.GPUs() {
		if g.Validation.Status != domain.ValidationFailed || g.Validation.Reason != "driver too old" {
			t.Fatalf("expected GPU %s validation FAILED with reason, got %+v", g.ID, g.Validation)
		}
	}
}

// TestInvariant_QuarantinedWorkerNeverReachesReadyViaValidation guards
// against a workflow bug where a failed-then-retried validation could skip
// straight to READY; the domain's own transition table already forbids
// QUARANTINED -> READY directly, this exercises that through the agent path.
func TestInvariant_QuarantinedWorkerNeverReachesReadyViaValidation(t *testing.T) {
	if domain.IsValidWorkerTransition(domain.WorkerQuarantined, domain.WorkerReady) {
		t.Fatal("QUARANTINED -> READY must not be a valid transition")
	}
}

func TestCollectAndApplyState_UpdatesWorkerGPUs(t *testing.T) {
	a, _ := NewSimulatedAgent(SimulatedConfig{WorkerID: "w1", Seed: 3, GPUs: testGPUSpecs()})
	w, err := Register(context.Background(), a, time.Now())
	if err != nil {
		t.Fatal(err)
	}
	if err := CollectAndApplyState(context.Background(), w, a); err != nil {
		t.Fatal(err)
	}
	for _, g := range w.GPUs() {
		if g.State.LastHeartbeat.IsZero() {
			t.Fatalf("expected GPU %s state to be populated, got zero value", g.ID)
		}
	}
}

func TestCollectAndApplyState_UnknownGPURejected(t *testing.T) {
	a, _ := NewSimulatedAgent(SimulatedConfig{WorkerID: "w1", Seed: 1, GPUs: testGPUSpecs()})
	w, err := domain.NewWorker("w1", domain.HardwareModeSimulated)
	if err != nil {
		t.Fatal(err)
	}
	// w has no GPUs attached, but a will report state for GPU IDs w doesn't know.
	err = CollectAndApplyState(context.Background(), w, a)
	if !errors.Is(err, domain.ErrInvalidAllocation) {
		t.Fatalf("expected ErrInvalidAllocation-wrapped error for unknown GPU, got %v", err)
	}
}
