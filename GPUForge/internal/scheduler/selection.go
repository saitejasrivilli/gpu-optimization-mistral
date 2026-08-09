package scheduler

import (
	"sort"

	"gpuforge/internal/domain"
)

// leftoverMemory is the total wasted capacity across a selected GPU subset:
// the sum, per selected GPU, of its capacity above the per-GPU minimum the
// workload asked for. Lower is a tighter fit.
func leftoverMemory(gpus []domain.GPUSnapshot, minPerGPU uint64) uint64 {
	var total uint64
	for _, g := range gpus {
		total += g.Capability.MemoryBytes - minPerGPU
	}
	return total
}

func avgUtilization(gpus []domain.GPUSnapshot) float64 {
	if len(gpus) == 0 {
		return 0
	}
	var total float64
	for _, g := range gpus {
		total += g.State.UtilizationPercent
	}
	return total / float64(len(gpus))
}

func gpuIDs(gpus []domain.GPUSnapshot) []string {
	ids := make([]string, len(gpus))
	for i, g := range gpus {
		ids[i] = g.ID
	}
	return ids
}

// sortedByID returns a copy of gpus sorted by GPU ID ascending — the base
// deterministic ordering every policy starts from.
func sortedByID(gpus []domain.GPUSnapshot) []domain.GPUSnapshot {
	out := make([]domain.GPUSnapshot, len(gpus))
	copy(out, gpus)
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
}

// smallestKByMemory returns the k GPUs with smallest capacity (ties broken
// by GPU ID), from a pool already known to be individually eligible.
func smallestKByMemory(pool []domain.GPUSnapshot, k int) []domain.GPUSnapshot {
	out := sortedByID(pool)
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Capability.MemoryBytes != out[j].Capability.MemoryBytes {
			return out[i].Capability.MemoryBytes < out[j].Capability.MemoryBytes
		}
		return out[i].ID < out[j].ID
	})
	return out[:k]
}

// smallestKByUtilization returns the k GPUs with lowest current
// utilization (ties broken by GPU ID).
func smallestKByUtilization(pool []domain.GPUSnapshot, k int) []domain.GPUSnapshot {
	out := sortedByID(pool)
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].State.UtilizationPercent != out[j].State.UtilizationPercent {
			return out[i].State.UtilizationPercent < out[j].State.UtilizationPercent
		}
		return out[i].ID < out[j].ID
	})
	return out[:k]
}

// idSliceLess gives a deterministic total order over equally-scored
// (workerID, gpuIDs) choices: compare worker IDs, then GPU IDs
// lexicographically. Used as the final tie-break by every policy so ties
// never depend on map/slice iteration order.
func idSliceLess(aWorker string, aIDs []string, bWorker string, bIDs []string) bool {
	if aWorker != bWorker {
		return aWorker < bWorker
	}
	for i := 0; i < len(aIDs) && i < len(bIDs); i++ {
		if aIDs[i] != bIDs[i] {
			return aIDs[i] < bIDs[i]
		}
	}
	return len(aIDs) < len(bIDs)
}
