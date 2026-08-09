package scheduler

import (
	"context"
	"fmt"
	"math/rand"
	"testing"
	"time"

	"gpuforge/internal/domain"
)

// genFleet builds a deterministic simulated fleet of gpuCount GPUs spread
// across ceil(gpuCount/8) 8-GPU workers (a typical single-node GPU host),
// with each worker's GPUs split into two 4-GPU NVLink groups. Utilization
// and remaining memory vary deterministically by seed so BestFit/
// UtilizationAware have real work to do, not a trivially uniform fleet.
func genFleet(gpuCount int, seed int64) domain.ClusterSnapshot {
	rng := rand.New(rand.NewSource(seed))
	const perWorker = 8
	numWorkers := (gpuCount + perWorker - 1) / perWorker

	workers := make([]domain.WorkerSnapshot, 0, numWorkers)
	remaining := gpuCount
	for wi := 0; wi < numWorkers; wi++ {
		workerID := fmt.Sprintf("worker-%03d", wi)
		n := perWorker
		if remaining < perWorker {
			n = remaining
		}
		gpus := make([]domain.GPUSnapshot, 0, n)
		for gi := 0; gi < n; gi++ {
			group := fmt.Sprintf("%s-nvlink-%d", workerID, gi/4)
			gpus = append(gpus, domain.GPUSnapshot{
				ID:           fmt.Sprintf("%s-gpu-%d", workerID, gi),
				WorkerID:     workerID,
				Model:        "A100",
				HardwareMode: domain.HardwareModeSimulated,
				Capability: domain.GPUCapability{
					ComputeCapability: "sm_80",
					MemoryBytes:       80 << 30,
				},
				Topology:        domain.GPUTopology{NodeID: workerID, NVLinkGroup: group},
				State:           domain.GPUState{UtilizationPercent: rng.Float64() * 100},
				Validation:      domain.ValidationResult{Status: domain.ValidationPassed},
				AllocationState: domain.GPUFree,
			})
		}
		workers = append(workers, domain.WorkerSnapshot{
			ID:           workerID,
			HardwareMode: domain.HardwareModeSimulated,
			State:        domain.WorkerReady,
			GPUs:         gpus,
		})
		remaining -= n
	}
	return domain.ClusterSnapshot{Workers: workers, Timestamp: time.Now()}
}

var benchFleetSizes = []int{8, 16, 32, 64}
var benchBatchSizes = []int{100, 1000, 10000}
var benchPolicies = []Scheduler{FirstFit{}, BestFit{}, UtilizationAware{}, TopologyAware{}}

// BenchmarkScheduler measures per-placement latency and derives throughput
// (placements/sec) for every (policy, fleet size, batch size) combination
// required by docs/benchmark-plan.md's scheduling section. Every workload
// in a batch requests 1 GPU with no requirements beyond that, so every
// combination is guaranteed schedulable regardless of fleet size — the
// point is to measure scheduler cost, not to simulate exhaustion.
func BenchmarkScheduler(b *testing.B) {
	for _, policy := range benchPolicies {
		for _, fleetSize := range benchFleetSizes {
			snapshot := genFleet(fleetSize, 42)
			for _, batchSize := range benchBatchSizes {
				name := fmt.Sprintf("%s/fleet-%dgpu/batch-%d", policy.Name(), fleetSize, batchSize)
				b.Run(name, func(b *testing.B) {
					ctx := context.Background()
					now := time.Now()
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						for j := 0; j < batchSize; j++ {
							r := domain.WorkloadRequirements{
								WorkloadID: fmt.Sprintf("bench-wl-%d-%d", i, j),
								GPUCount:   1,
							}
							if _, err := policy.Place(ctx, r, snapshot, now); err != nil {
								b.Fatalf("unexpected scheduling failure: %v", err)
							}
						}
					}
					opsPerSec := float64(batchSize) / (float64(b.Elapsed().Nanoseconds()) / float64(b.N) / 1e9)
					b.ReportMetric(opsPerSec, "placements/sec")
				})
			}
		}
	}
}
