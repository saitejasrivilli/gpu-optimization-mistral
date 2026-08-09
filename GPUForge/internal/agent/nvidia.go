package agent

import (
	"context"
	"encoding/csv"
	"fmt"
	"os/exec"
	"strconv"
	"strings"
	"time"

	"gpuforge/internal/domain"
)

// CommandRunner abstracts shelling out to nvidia-smi so NVIDIAAgent is
// testable without real hardware: tests inject a fake runner returning
// canned CSV, the real binary uses execRunner. Never add hidden retries
// here — a failed command is a failed call, surfaced to the caller.
type CommandRunner interface {
	Run(ctx context.Context, name string, args ...string) ([]byte, error)
}

type execRunner struct{}

func (execRunner) Run(ctx context.Context, name string, args ...string) ([]byte, error) {
	cmd := exec.CommandContext(ctx, name, args...)
	out, err := cmd.Output()
	if err != nil {
		if ee, ok := err.(*exec.ExitError); ok {
			return nil, fmt.Errorf("agent: %s exited: %w: %s", name, err, strings.TrimSpace(string(ee.Stderr)))
		}
		return nil, fmt.Errorf("agent: failed to run %s: %w", name, err)
	}
	return out, nil
}

// NVIDIAAgent is the real-hardware WorkerAgent implementation. It shells out
// to nvidia-smi rather than binding NVML/CUDA directly (see
// docs/decisions/ADR-001-language-and-runtime.md) because scheduling logic
// only needs GPU metadata, not direct device access. It always reports
// domain.HardwareModeReal and never fabricates a value it could not query.
type NVIDIAAgent struct {
	workerID string
	runner   CommandRunner
}

// NewNVIDIAAgent constructs a real-hardware agent for workerID. Discovery is
// not performed at construction time — call Discover explicitly so
// construction never fails due to transient hardware/driver issues.
func NewNVIDIAAgent(workerID string) (*NVIDIAAgent, error) {
	if workerID == "" {
		return nil, ErrEmptyWorkerID
	}
	return &NVIDIAAgent{workerID: workerID, runner: execRunner{}}, nil
}

// newNVIDIAAgentWithRunner is used by tests to inject a fake CommandRunner.
func newNVIDIAAgentWithRunner(workerID string, r CommandRunner) *NVIDIAAgent {
	return &NVIDIAAgent{workerID: workerID, runner: r}
}

func (a *NVIDIAAgent) HardwareMode() domain.HardwareMode { return domain.HardwareModeReal }

const discoverQuery = "uuid,name,memory.total,driver_version,compute_cap"

// Discover shells out to `nvidia-smi --query-gpu=... --format=csv,noheader,nounits`
// and parses each row into a GPUDiscovery. RuntimeVersion is left empty:
// nvidia-smi's --query-gpu form does not expose the CUDA runtime version
// (only driver_version); wiring that up is deferred rather than guessed.
func (a *NVIDIAAgent) Discover(ctx context.Context) (DiscoveryResult, error) {
	out, err := a.runner.Run(ctx, "nvidia-smi", "--query-gpu="+discoverQuery, "--format=csv,noheader,nounits")
	if err != nil {
		return DiscoveryResult{}, err
	}
	rows, err := parseCSV(out, 5)
	if err != nil {
		return DiscoveryResult{}, err
	}
	gpus := make([]GPUDiscovery, 0, len(rows))
	for _, row := range rows {
		uuid, name, memTotalMiB, driverVersion, computeCap := row[0], row[1], row[2], row[3], row[4]
		memBytes, err := parseMiBToBytes(memTotalMiB)
		if err != nil {
			return DiscoveryResult{}, fmt.Errorf("agent: parsing memory.total for GPU %s: %w", uuid, err)
		}
		gpus = append(gpus, GPUDiscovery{
			ID:    uuid,
			Model: name,
			Capability: domain.GPUCapability{
				ComputeCapability: computeCap,
				DriverVersion:     driverVersion,
				RuntimeVersion:    "",
				MemoryBytes:       memBytes,
			},
		})
	}
	return DiscoveryResult{
		WorkerID:     a.workerID,
		HardwareMode: domain.HardwareModeReal,
		GPUs:         gpus,
		Timestamp:    time.Now(),
	}, nil
}

const stateQuery = "uuid,utilization.gpu,memory.free"

// CollectState shells out for current utilization and free memory per GPU.
func (a *NVIDIAAgent) CollectState(ctx context.Context) ([]StateSample, error) {
	out, err := a.runner.Run(ctx, "nvidia-smi", "--query-gpu="+stateQuery, "--format=csv,noheader,nounits")
	if err != nil {
		return nil, err
	}
	rows, err := parseCSV(out, 3)
	if err != nil {
		return nil, err
	}
	now := time.Now()
	samples := make([]StateSample, 0, len(rows))
	for _, row := range rows {
		uuid, utilStr, freeMiB := row[0], row[1], row[2]
		util, err := strconv.ParseFloat(strings.TrimSpace(utilStr), 64)
		if err != nil {
			return nil, fmt.Errorf("agent: parsing utilization.gpu for GPU %s: %w", uuid, err)
		}
		freeBytes, err := parseMiBToBytes(freeMiB)
		if err != nil {
			return nil, fmt.Errorf("agent: parsing memory.free for GPU %s: %w", uuid, err)
		}
		samples = append(samples, StateSample{
			GPUID: uuid,
			State: domain.GPUState{
				UtilizationPercent:   util,
				AvailableMemoryBytes: freeBytes,
				LastHeartbeat:        now,
			},
		})
	}
	return samples, nil
}

// Validate checks that every GPU discoverable at construction time is still
// visible to nvidia-smi. This is a presence/liveness check, not a CUDA/NCCL
// correctness test — real compute validation is out of scope for this phase.
func (a *NVIDIAAgent) Validate(ctx context.Context) ([]ValidationSample, error) {
	discovered, err := a.Discover(ctx)
	if err != nil {
		return nil, err
	}
	now := time.Now()
	samples := make([]ValidationSample, len(discovered.GPUs))
	for i, g := range discovered.GPUs {
		samples[i] = ValidationSample{GPUID: g.ID, Result: domain.Pass(now)}
	}
	return samples, nil
}

// Heartbeat treats a successful, fast nvidia-smi query as liveness.
func (a *NVIDIAAgent) Heartbeat(ctx context.Context) (HeartbeatResult, error) {
	_, err := a.runner.Run(ctx, "nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader")
	return HeartbeatResult{
		WorkerID:  a.workerID,
		Alive:     err == nil,
		Timestamp: time.Now(),
	}, nil
}

func parseCSV(data []byte, wantFields int) ([][]string, error) {
	r := csv.NewReader(strings.NewReader(string(data)))
	r.TrimLeadingSpace = true
	rows, err := r.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("agent: parsing nvidia-smi CSV output: %w", err)
	}
	for i, row := range rows {
		if len(row) != wantFields {
			return nil, fmt.Errorf("agent: row %d: expected %d fields, got %d: %q", i, wantFields, len(row), row)
		}
	}
	return rows, nil
}

func parseMiBToBytes(s string) (uint64, error) {
	mib, err := strconv.ParseUint(strings.TrimSpace(s), 10, 64)
	if err != nil {
		return 0, err
	}
	return mib * 1024 * 1024, nil
}

var _ WorkerAgent = (*NVIDIAAgent)(nil)
