package domain

import "testing"

func TestHardwareMode_Valid(t *testing.T) {
	cases := []struct {
		mode HardwareMode
		want bool
	}{
		{HardwareModeSimulated, true},
		{HardwareModeReal, true},
		{HardwareMode(""), false},
		{HardwareMode("fake"), false},
		{HardwareMode("REAL"), false}, // case-sensitive: no accidental match
	}
	for _, c := range cases {
		if got := c.mode.Valid(); got != c.want {
			t.Errorf("HardwareMode(%q).Valid() = %v, want %v", c.mode, got, c.want)
		}
	}
}

// TestInvariant_SimulatedGPUCannotBeAddedToRealWorker and its inverse make
// the "simulated hardware never reports as real" invariant structurally
// hard to violate: GPU hardware mode is checked against the owning worker's
// mode at the only point a GPU becomes attached.
func TestInvariant_SimulatedGPUCannotBeAddedToRealWorker(t *testing.T) {
	w, err := NewWorker("w1", HardwareModeReal)
	if err != nil {
		t.Fatal(err)
	}
	g, err := NewGPU("g1", "w1", "A100", HardwareModeSimulated, GPUCapability{})
	if err != nil {
		t.Fatal(err)
	}
	if err := w.AddGPU(g); err == nil {
		t.Fatal("expected simulated GPU on real worker to be rejected")
	}
}

func TestInvariant_RealGPUCannotBeAddedToSimulatedWorker(t *testing.T) {
	w, err := NewWorker("w1", HardwareModeSimulated)
	if err != nil {
		t.Fatal(err)
	}
	g, err := NewGPU("g1", "w1", "A100", HardwareModeReal, GPUCapability{})
	if err != nil {
		t.Fatal(err)
	}
	if err := w.AddGPU(g); err == nil {
		t.Fatal("expected real GPU on simulated worker to be rejected")
	}
}
