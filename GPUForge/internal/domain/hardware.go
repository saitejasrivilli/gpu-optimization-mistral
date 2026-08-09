package domain

// HardwareMode distinguishes simulated from real GPU hardware. It is a
// closed enum: nothing else may claim to be a valid mode. Every entity that
// wraps hardware (GPU, Worker) carries this field explicitly, per the Phase 0
// invariant that simulated hardware must never be represented as real.
type HardwareMode string

const (
	HardwareModeSimulated HardwareMode = "simulated"
	HardwareModeReal      HardwareMode = "real"
)

// Valid reports whether m is one of the closed set of known modes.
func (m HardwareMode) Valid() bool {
	switch m {
	case HardwareModeSimulated, HardwareModeReal:
		return true
	default:
		return false
	}
}

func (m HardwareMode) String() string { return string(m) }
