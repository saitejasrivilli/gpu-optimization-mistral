package scheduler

import (
	"time"

	"gpuforge/internal/domain"
)

// gpuOpt customizes a test GPU snapshot beyond its defaults.
type gpuOpt func(*domain.GPUSnapshot)

func withMemory(bytes uint64) gpuOpt {
	return func(g *domain.GPUSnapshot) { g.Capability.MemoryBytes = bytes }
}

func withCUDA(cap string) gpuOpt {
	return func(g *domain.GPUSnapshot) { g.Capability.ComputeCapability = cap }
}

func withUtilization(pct float64) gpuOpt {
	return func(g *domain.GPUSnapshot) { g.State.UtilizationPercent = pct }
}

func withAllocationState(s domain.GPUAllocationState) gpuOpt {
	return func(g *domain.GPUSnapshot) { g.AllocationState = s }
}

func withValidation(s domain.ValidationStatus) gpuOpt {
	return func(g *domain.GPUSnapshot) { g.Validation.Status = s }
}

func withNVLinkGroup(group string) gpuOpt {
	return func(g *domain.GPUSnapshot) { g.Topology.NVLinkGroup = group }
}

// testGPU builds a GPU snapshot that is eligible by default (FREE,
// PASSED, 80GB, sm_80, 0% utilization) and applies opts on top.
func testGPU(id string, opts ...gpuOpt) domain.GPUSnapshot {
	g := domain.GPUSnapshot{
		ID:              id,
		Model:           "A100",
		HardwareMode:    domain.HardwareModeSimulated,
		Capability:      domain.GPUCapability{ComputeCapability: "sm_80", MemoryBytes: 80 << 30},
		AllocationState: domain.GPUFree,
		Validation:      domain.ValidationResult{Status: domain.ValidationPassed},
	}
	for _, o := range opts {
		o(&g)
	}
	return g
}

// testWorker builds a READY worker snapshot with the given GPUs, setting
// each GPU's WorkerID to match.
func testWorker(id string, state domain.WorkerState, gpus ...domain.GPUSnapshot) domain.WorkerSnapshot {
	for i := range gpus {
		gpus[i].WorkerID = id
	}
	return domain.WorkerSnapshot{
		ID:           id,
		HardwareMode: domain.HardwareModeSimulated,
		State:        state,
		GPUs:         gpus,
	}
}

func testSnapshot(workers ...domain.WorkerSnapshot) domain.ClusterSnapshot {
	return domain.ClusterSnapshot{Workers: workers, Timestamp: time.Now()}
}

func req(id string, gpuCount int, opts ...func(*domain.WorkloadRequirements)) domain.WorkloadRequirements {
	r := domain.WorkloadRequirements{WorkloadID: id, GPUCount: gpuCount}
	for _, o := range opts {
		o(&r)
	}
	return r
}

func withMinMemory(bytes uint64) func(*domain.WorkloadRequirements) {
	return func(r *domain.WorkloadRequirements) { r.MinGPUMemoryBytes = bytes }
}

func withCUDARequirement(cap string) func(*domain.WorkloadRequirements) {
	return func(r *domain.WorkloadRequirements) { r.CUDARequirement = cap }
}

func withTopologyRequirement(t domain.TopologyRequirement) func(*domain.WorkloadRequirements) {
	return func(r *domain.WorkloadRequirements) { r.TopologyRequirement = t }
}

var allPolicies = []Scheduler{FirstFit{}, BestFit{}, UtilizationAware{}, TopologyAware{}}
