package agent

import (
	"context"
	"errors"
	"reflect"
	"testing"

	"gpuforge/internal/domain"
)

func testGPUSpecs() []GPUSpec {
	return []GPUSpec{
		{Model: "A100", Capability: domain.GPUCapability{ComputeCapability: "sm_80", DriverVersion: "535.104", MemoryBytes: 80 << 30}},
		{Model: "A100", Capability: domain.GPUCapability{ComputeCapability: "sm_80", DriverVersion: "535.104", MemoryBytes: 80 << 30}},
	}
}

func TestNewSimulatedAgent_Validation(t *testing.T) {
	if _, err := NewSimulatedAgent(SimulatedConfig{WorkerID: "", GPUs: testGPUSpecs()}); !errors.Is(err, ErrEmptyWorkerID) {
		t.Fatalf("expected ErrEmptyWorkerID, got %v", err)
	}
	if _, err := NewSimulatedAgent(SimulatedConfig{WorkerID: "w1", GPUs: nil}); !errors.Is(err, ErrNoGPUs) {
		t.Fatalf("expected ErrNoGPUs, got %v", err)
	}
}

func TestSimulatedAgent_HardwareMode(t *testing.T) {
	a, err := NewSimulatedAgent(SimulatedConfig{WorkerID: "w1", Seed: 1, GPUs: testGPUSpecs()})
	if err != nil {
		t.Fatal(err)
	}
	if a.HardwareMode() != domain.HardwareModeSimulated {
		t.Fatalf("expected simulated hardware mode, got %s", a.HardwareMode())
	}
}

func TestSimulatedAgent_Discover_Deterministic(t *testing.T) {
	cfg := SimulatedConfig{WorkerID: "w1", Seed: 42, GPUs: testGPUSpecs()}
	a1, _ := NewSimulatedAgent(cfg)
	a2, _ := NewSimulatedAgent(cfg)

	d1, err := a1.Discover(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	d2, err := a2.Discover(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(d1.GPUs, d2.GPUs) {
		t.Fatalf("expected identical discovery GPUs from identical config, got %+v vs %+v", d1.GPUs, d2.GPUs)
	}
	if d1.WorkerID != "w1" || d1.HardwareMode != domain.HardwareModeSimulated {
		t.Fatalf("unexpected discovery result: %+v", d1)
	}
	if len(d1.GPUs) != 2 || d1.GPUs[0].ID != "w1-gpu-0" || d1.GPUs[1].ID != "w1-gpu-1" {
		t.Fatalf("unexpected GPU IDs: %+v", d1.GPUs)
	}
}

func TestSimulatedAgent_CollectState_DeterministicSequence(t *testing.T) {
	cfg := SimulatedConfig{WorkerID: "w1", Seed: 7, GPUs: testGPUSpecs()}
	a1, _ := NewSimulatedAgent(cfg)
	a2, _ := NewSimulatedAgent(cfg)

	for call := 0; call < 3; call++ {
		s1, err := a1.CollectState(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		s2, err := a2.CollectState(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		for i := range s1 {
			if s1[i].GPUID != s2[i].GPUID ||
				s1[i].State.UtilizationPercent != s2[i].State.UtilizationPercent ||
				s1[i].State.AvailableMemoryBytes != s2[i].State.AvailableMemoryBytes {
				t.Fatalf("call %d: expected identical sample sequence, got %+v vs %+v", call, s1[i], s2[i])
			}
		}
	}
}

func TestSimulatedAgent_CollectState_DifferentSeedsDiverge(t *testing.T) {
	a1, _ := NewSimulatedAgent(SimulatedConfig{WorkerID: "w1", Seed: 1, GPUs: testGPUSpecs()})
	a2, _ := NewSimulatedAgent(SimulatedConfig{WorkerID: "w1", Seed: 2, GPUs: testGPUSpecs()})

	s1, _ := a1.CollectState(context.Background())
	s2, _ := a2.CollectState(context.Background())
	if s1[0].State.UtilizationPercent == s2[0].State.UtilizationPercent {
		t.Fatalf("expected different seeds to (almost certainly) diverge, got identical utilization %v", s1[0].State.UtilizationPercent)
	}
}

func TestSimulatedAgent_Validate_PassByDefault(t *testing.T) {
	a, _ := NewSimulatedAgent(SimulatedConfig{WorkerID: "w1", Seed: 1, GPUs: testGPUSpecs()})
	samples, err := a.Validate(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	for _, s := range samples {
		if s.Result.Status != domain.ValidationPassed {
			t.Fatalf("expected PASSED, got %+v", s)
		}
	}
}

func TestSimulatedAgent_Validate_ConfiguredFailure(t *testing.T) {
	a, _ := NewSimulatedAgent(SimulatedConfig{WorkerID: "w1", Seed: 1, GPUs: testGPUSpecs(), FailValidationReason: "driver too old"})
	samples, err := a.Validate(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	for _, s := range samples {
		if s.Result.Status != domain.ValidationFailed || s.Result.Reason != "driver too old" {
			t.Fatalf("expected FAILED with configured reason, got %+v", s)
		}
	}
}

func TestSimulatedAgent_Heartbeat(t *testing.T) {
	alive, _ := NewSimulatedAgent(SimulatedConfig{WorkerID: "w1", Seed: 1, GPUs: testGPUSpecs()})
	hb, err := alive.Heartbeat(context.Background())
	if err != nil || !hb.Alive {
		t.Fatalf("expected alive heartbeat, got %+v, err=%v", hb, err)
	}

	down, _ := NewSimulatedAgent(SimulatedConfig{WorkerID: "w1", Seed: 1, GPUs: testGPUSpecs(), SimulateUnreachable: true})
	hb, err = down.Heartbeat(context.Background())
	if err != nil || hb.Alive {
		t.Fatalf("expected unreachable heartbeat, got %+v, err=%v", hb, err)
	}
}

func TestSimulatedAgent_RespectsContextCancellation(t *testing.T) {
	a, _ := NewSimulatedAgent(SimulatedConfig{WorkerID: "w1", Seed: 1, GPUs: testGPUSpecs()})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	if _, err := a.Discover(ctx); err == nil {
		t.Fatal("expected Discover to respect cancelled context")
	}
	if _, err := a.CollectState(ctx); err == nil {
		t.Fatal("expected CollectState to respect cancelled context")
	}
	if _, err := a.Validate(ctx); err == nil {
		t.Fatal("expected Validate to respect cancelled context")
	}
	if _, err := a.Heartbeat(ctx); err == nil {
		t.Fatal("expected Heartbeat to respect cancelled context")
	}
}
