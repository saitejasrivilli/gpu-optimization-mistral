package agent

import (
	"context"
	"errors"
	"os/exec"
	"testing"

	"gpuforge/internal/domain"
)

type fakeRunner struct {
	out map[string][]byte
	err map[string]error
}

func (f *fakeRunner) key(name string, args ...string) string {
	k := name
	for _, a := range args {
		k += " " + a
	}
	return k
}

func (f *fakeRunner) Run(ctx context.Context, name string, args ...string) ([]byte, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	k := f.key(name, args...)
	if err, ok := f.err[k]; ok {
		return nil, err
	}
	if out, ok := f.out[k]; ok {
		return out, nil
	}
	return nil, errors.New("fakeRunner: no stub for " + k)
}

func TestNewNVIDIAAgent_Validation(t *testing.T) {
	if _, err := NewNVIDIAAgent(""); !errors.Is(err, ErrEmptyWorkerID) {
		t.Fatalf("expected ErrEmptyWorkerID, got %v", err)
	}
}

func TestNVIDIAAgent_HardwareMode(t *testing.T) {
	a, _ := NewNVIDIAAgent("w1")
	if a.HardwareMode() != domain.HardwareModeReal {
		t.Fatalf("expected real hardware mode, got %s", a.HardwareMode())
	}
}

func TestNVIDIAAgent_Discover_ParsesCSV(t *testing.T) {
	fr := &fakeRunner{out: map[string][]byte{
		"nvidia-smi --query-gpu=" + discoverQuery + " --format=csv,noheader,nounits": []byte(
			"GPU-aaaa,NVIDIA A100,81920,535.104.05,8.0\nGPU-bbbb,NVIDIA A100,81920,535.104.05,8.0\n"),
	}}
	a := newNVIDIAAgentWithRunner("w1", fr)
	result, err := a.Discover(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if result.WorkerID != "w1" || result.HardwareMode != domain.HardwareModeReal {
		t.Fatalf("unexpected discovery result: %+v", result)
	}
	if len(result.GPUs) != 2 {
		t.Fatalf("expected 2 GPUs, got %d", len(result.GPUs))
	}
	g := result.GPUs[0]
	if g.ID != "GPU-aaaa" || g.Model != "NVIDIA A100" || g.Capability.DriverVersion != "535.104.05" ||
		g.Capability.ComputeCapability != "8.0" || g.Capability.MemoryBytes != 81920*1024*1024 {
		t.Fatalf("unexpected GPU: %+v", g)
	}
}

func TestNVIDIAAgent_Discover_MalformedRowRejected(t *testing.T) {
	fr := &fakeRunner{out: map[string][]byte{
		"nvidia-smi --query-gpu=" + discoverQuery + " --format=csv,noheader,nounits": []byte("GPU-aaaa,NVIDIA A100,81920\n"),
	}}
	a := newNVIDIAAgentWithRunner("w1", fr)
	if _, err := a.Discover(context.Background()); err == nil {
		t.Fatal("expected malformed row to be rejected")
	}
}

func TestNVIDIAAgent_Discover_CommandFailurePropagates(t *testing.T) {
	fr := &fakeRunner{err: map[string]error{
		"nvidia-smi --query-gpu=" + discoverQuery + " --format=csv,noheader,nounits": &exec.Error{Name: "nvidia-smi", Err: exec.ErrNotFound},
	}}
	a := newNVIDIAAgentWithRunner("w1", fr)
	if _, err := a.Discover(context.Background()); err == nil {
		t.Fatal("expected command failure to propagate, not be swallowed")
	}
}

func TestNVIDIAAgent_CollectState_ParsesCSV(t *testing.T) {
	fr := &fakeRunner{out: map[string][]byte{
		"nvidia-smi --query-gpu=" + stateQuery + " --format=csv,noheader,nounits": []byte("GPU-aaaa,42,10000\n"),
	}}
	a := newNVIDIAAgentWithRunner("w1", fr)
	samples, err := a.CollectState(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(samples) != 1 || samples[0].GPUID != "GPU-aaaa" || samples[0].State.UtilizationPercent != 42 ||
		samples[0].State.AvailableMemoryBytes != 10000*1024*1024 {
		t.Fatalf("unexpected state sample: %+v", samples)
	}
}

func TestNVIDIAAgent_Heartbeat(t *testing.T) {
	okRunner := &fakeRunner{out: map[string][]byte{
		"nvidia-smi --query-gpu=uuid --format=csv,noheader": []byte("GPU-aaaa\n"),
	}}
	a := newNVIDIAAgentWithRunner("w1", okRunner)
	hb, err := a.Heartbeat(context.Background())
	if err != nil || !hb.Alive {
		t.Fatalf("expected alive heartbeat, got %+v err=%v", hb, err)
	}

	downRunner := &fakeRunner{err: map[string]error{
		"nvidia-smi --query-gpu=uuid --format=csv,noheader": errors.New("no devices found"),
	}}
	a = newNVIDIAAgentWithRunner("w1", downRunner)
	hb, err = a.Heartbeat(context.Background())
	if err != nil {
		t.Fatalf("Heartbeat itself should not error on unreachable device, got %v", err)
	}
	if hb.Alive {
		t.Fatalf("expected not-alive heartbeat when command fails, got %+v", hb)
	}
}

// TestNVIDIAAgent_RealHardware_Integration only runs where nvidia-smi is
// actually installed. On machines without NVIDIA hardware (e.g. this
// project's usual dev machine) it skips rather than faking success —
// simulated results must never stand in for a real-hardware assertion.
func TestNVIDIAAgent_RealHardware_Integration(t *testing.T) {
	if _, err := exec.LookPath("nvidia-smi"); err != nil {
		t.Skip("nvidia-smi not present on this machine; skipping real-hardware integration test")
	}
	a, err := NewNVIDIAAgent("real-worker-1")
	if err != nil {
		t.Fatal(err)
	}
	result, err := a.Discover(context.Background())
	if err != nil {
		t.Fatalf("Discover against real hardware failed: %v", err)
	}
	if result.HardwareMode != domain.HardwareModeReal {
		t.Fatalf("expected real hardware mode, got %s", result.HardwareMode)
	}
	if len(result.GPUs) == 0 {
		t.Fatal("expected at least one real GPU to be discovered")
	}
}
