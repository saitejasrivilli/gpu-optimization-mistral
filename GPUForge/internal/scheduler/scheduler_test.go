package scheduler

import (
	"context"
	"errors"
	"reflect"
	"sync"
	"testing"
	"time"

	"gpuforge/internal/domain"
)

func TestAllPolicies_OneAvailableGPU(t *testing.T) {
	snap := testSnapshot(testWorker("w1", domain.WorkerReady, testGPU("g1")))
	r := req("wl1", 1)
	for _, p := range allPolicies {
		t.Run(p.Name(), func(t *testing.T) {
			pl, err := p.Place(context.Background(), r, snap, time.Now())
			if err != nil {
				t.Fatal(err)
			}
			if pl.WorkerID != "w1" || !reflect.DeepEqual(pl.GPUIDs, []string{"g1"}) {
				t.Fatalf("unexpected placement: %+v", pl)
			}
			if pl.Reason == "" {
				t.Fatal("expected non-empty placement reason")
			}
		})
	}
}

func TestAllPolicies_MultipleGPUs(t *testing.T) {
	snap := testSnapshot(testWorker("w1", domain.WorkerReady, testGPU("g1"), testGPU("g2"), testGPU("g3")))
	r := req("wl1", 2)
	for _, p := range allPolicies {
		t.Run(p.Name(), func(t *testing.T) {
			pl, err := p.Place(context.Background(), r, snap, time.Now())
			if err != nil {
				t.Fatal(err)
			}
			if len(pl.GPUIDs) != 2 {
				t.Fatalf("expected 2 GPUs, got %v", pl.GPUIDs)
			}
		})
	}
}

func TestAllPolicies_InsufficientGPUs(t *testing.T) {
	snap := testSnapshot(testWorker("w1", domain.WorkerReady, testGPU("g1")))
	r := req("wl1", 2)
	for _, p := range allPolicies {
		t.Run(p.Name(), func(t *testing.T) {
			_, err := p.Place(context.Background(), r, snap, time.Now())
			var serr *SchedulingError
			if !errors.As(err, &serr) {
				t.Fatalf("expected *SchedulingError, got %v", err)
			}
			if !errors.Is(err, ErrInsufficientGPUs) {
				t.Fatalf("expected ErrInsufficientGPUs, got %v", err)
			}
		})
	}
}

func TestAllPolicies_InsufficientMemory(t *testing.T) {
	snap := testSnapshot(testWorker("w1", domain.WorkerReady, testGPU("g1", withMemory(8<<30))))
	r := req("wl1", 1, withMinMemory(80<<30))
	for _, p := range allPolicies {
		t.Run(p.Name(), func(t *testing.T) {
			_, err := p.Place(context.Background(), r, snap, time.Now())
			if !errors.Is(err, ErrInsufficientMemory) {
				t.Fatalf("expected ErrInsufficientMemory, got %v", err)
			}
		})
	}
}

func TestAllPolicies_UnavailableGPUsSkipped(t *testing.T) {
	snap := testSnapshot(testWorker("w1", domain.WorkerReady,
		testGPU("g1", withAllocationState(domain.GPUAllocated)),
		testGPU("g2"),
	))
	r := req("wl1", 1)
	for _, p := range allPolicies {
		t.Run(p.Name(), func(t *testing.T) {
			pl, err := p.Place(context.Background(), r, snap, time.Now())
			if err != nil {
				t.Fatal(err)
			}
			if pl.GPUIDs[0] != "g2" {
				t.Fatalf("expected allocated GPU g1 to be skipped, got %v", pl.GPUIDs)
			}
		})
	}
}

func TestAllPolicies_IncompatibleCUDA(t *testing.T) {
	snap := testSnapshot(testWorker("w1", domain.WorkerReady, testGPU("g1", withCUDA("sm_70"))))
	r := req("wl1", 1, withCUDARequirement("sm_90"))
	for _, p := range allPolicies {
		t.Run(p.Name(), func(t *testing.T) {
			_, err := p.Place(context.Background(), r, snap, time.Now())
			if !errors.Is(err, ErrIncompatibleCUDA) {
				t.Fatalf("expected ErrIncompatibleCUDA, got %v", err)
			}
		})
	}
}

func TestAllPolicies_MultipleWorkers(t *testing.T) {
	snap := testSnapshot(
		testWorker("w1", domain.WorkerReady, testGPU("g1")),
		testWorker("w2", domain.WorkerReady, testGPU("g2")),
	)
	r := req("wl1", 1)
	for _, p := range allPolicies {
		t.Run(p.Name(), func(t *testing.T) {
			pl, err := p.Place(context.Background(), r, snap, time.Now())
			if err != nil {
				t.Fatal(err)
			}
			if pl.WorkerID != "w1" {
				t.Fatalf("expected deterministic pick of lowest-ID worker w1, got %s", pl.WorkerID)
			}
		})
	}
}

// TestInvariant_QuarantinedAndRetiredWorkersNeverSelected covers two
// explicit Phase 3 invariants at once.
func TestInvariant_QuarantinedAndRetiredWorkersNeverSelected(t *testing.T) {
	snap := testSnapshot(
		testWorker("w1", domain.WorkerQuarantined, testGPU("g1")),
		testWorker("w2", domain.WorkerRetired, testGPU("g2")),
		testWorker("w3", domain.WorkerReady, testGPU("g3")),
	)
	r := req("wl1", 1)
	for _, p := range allPolicies {
		t.Run(p.Name(), func(t *testing.T) {
			pl, err := p.Place(context.Background(), r, snap, time.Now())
			if err != nil {
				t.Fatal(err)
			}
			if pl.WorkerID != "w3" {
				t.Fatalf("expected only READY worker w3 to be selectable, got %s", pl.WorkerID)
			}
		})
	}
}

func TestInvariant_UnvalidatedGPUNeverSelected(t *testing.T) {
	snap := testSnapshot(testWorker("w1", domain.WorkerReady,
		testGPU("g1", withValidation(domain.ValidationPending)),
		testGPU("g2"),
	))
	r := req("wl1", 1)
	for _, p := range allPolicies {
		t.Run(p.Name(), func(t *testing.T) {
			pl, err := p.Place(context.Background(), r, snap, time.Now())
			if err != nil {
				t.Fatal(err)
			}
			if pl.GPUIDs[0] != "g2" {
				t.Fatalf("expected unvalidated GPU g1 to be skipped, got %v", pl.GPUIDs)
			}
		})
	}
}

func TestAllPolicies_FragmentedCapacity(t *testing.T) {
	// Neither worker alone has 2 free GPUs; only w3 does.
	snap := testSnapshot(
		testWorker("w1", domain.WorkerReady, testGPU("g1", withAllocationState(domain.GPUAllocated)), testGPU("g2")),
		testWorker("w2", domain.WorkerReady, testGPU("g3"), testGPU("g4", withAllocationState(domain.GPUAllocated))),
		testWorker("w3", domain.WorkerReady, testGPU("g5"), testGPU("g6")),
	)
	r := req("wl1", 2)
	for _, p := range allPolicies {
		t.Run(p.Name(), func(t *testing.T) {
			pl, err := p.Place(context.Background(), r, snap, time.Now())
			if err != nil {
				t.Fatal(err)
			}
			if pl.WorkerID != "w3" {
				t.Fatalf("expected w3 (only worker with 2 free GPUs), got %s", pl.WorkerID)
			}
		})
	}
}

func TestAllPolicies_DeterministicOrdering(t *testing.T) {
	snap := testSnapshot(
		testWorker("w2", domain.WorkerReady, testGPU("g3"), testGPU("g4")),
		testWorker("w1", domain.WorkerReady, testGPU("g1"), testGPU("g2")),
	)
	r := req("wl1", 1)
	for _, p := range allPolicies {
		t.Run(p.Name(), func(t *testing.T) {
			var first Placement
			for i := 0; i < 20; i++ {
				pl, err := p.Place(context.Background(), r, snap, time.Now())
				if err != nil {
					t.Fatal(err)
				}
				if i == 0 {
					first = pl
					continue
				}
				if pl.WorkerID != first.WorkerID || !reflect.DeepEqual(pl.GPUIDs, first.GPUIDs) {
					t.Fatalf("non-deterministic placement: %+v vs %+v", pl, first)
				}
			}
		})
	}
}

func TestAllPolicies_EqualScores_StableTieBreak(t *testing.T) {
	// Two workers, identically eligible GPUs -> every policy must pick the
	// lower-ID worker/GPU deterministically, not "whichever happened first".
	snap := testSnapshot(
		testWorker("w2", domain.WorkerReady, testGPU("g9")),
		testWorker("w1", domain.WorkerReady, testGPU("g8")),
	)
	r := req("wl1", 1)
	for _, p := range allPolicies {
		t.Run(p.Name(), func(t *testing.T) {
			pl, err := p.Place(context.Background(), r, snap, time.Now())
			if err != nil {
				t.Fatal(err)
			}
			if pl.WorkerID != "w1" {
				t.Fatalf("expected tie-break to favor lower worker ID w1, got %s", pl.WorkerID)
			}
		})
	}
}

func TestAllPolicies_NoEligibleWorkers_EmptySnapshot(t *testing.T) {
	snap := testSnapshot()
	r := req("wl1", 1)
	for _, p := range allPolicies {
		t.Run(p.Name(), func(t *testing.T) {
			_, err := p.Place(context.Background(), r, snap, time.Now())
			if !errors.Is(err, ErrNoEligibleWorkers) {
				t.Fatalf("expected ErrNoEligibleWorkers, got %v", err)
			}
		})
	}
}

func TestAllPolicies_FailedSchedulingProducesNoPartialAllocation(t *testing.T) {
	snap := testSnapshot(testWorker("w1", domain.WorkerReady, testGPU("g1")))
	r := req("wl1", 5)
	for _, p := range allPolicies {
		t.Run(p.Name(), func(t *testing.T) {
			pl, err := p.Place(context.Background(), r, snap, time.Now())
			if err == nil {
				t.Fatal("expected error")
			}
			if pl.WorkerID != "" || pl.GPUIDs != nil {
				t.Fatalf("expected zero-value Placement on failure, got %+v", pl)
			}
		})
	}
}

// TestInvariant_SchedulerNeverMutatesSnapshot builds a snapshot, takes a
// deep-equal snapshot of it, runs every policy against it repeatedly
// (including concurrently), and asserts the original is untouched.
func TestInvariant_SchedulerNeverMutatesSnapshot(t *testing.T) {
	snap := testSnapshot(
		testWorker("w1", domain.WorkerReady, testGPU("g1"), testGPU("g2"), testGPU("g3")),
		testWorker("w2", domain.WorkerReady, testGPU("g4"), testGPU("g5")),
	)
	before := deepCopySnapshot(snap)
	r := req("wl1", 2)

	var wg sync.WaitGroup
	for _, p := range allPolicies {
		for i := 0; i < 10; i++ {
			wg.Add(1)
			go func(p Scheduler) {
				defer wg.Done()
				_, _ = p.Place(context.Background(), r, snap, time.Now())
			}(p)
		}
	}
	wg.Wait()

	if !reflect.DeepEqual(snap, before) {
		t.Fatalf("scheduler mutated its input snapshot:\nbefore=%+v\nafter=%+v", before, snap)
	}
}

// TestInvariant_SelectedGPUIDsUnique guards against a policy accidentally
// selecting the same GPU twice for a multi-GPU workload.
func TestInvariant_SelectedGPUIDsUnique(t *testing.T) {
	snap := testSnapshot(testWorker("w1", domain.WorkerReady, testGPU("g1"), testGPU("g2"), testGPU("g3")))
	r := req("wl1", 3)
	for _, p := range allPolicies {
		t.Run(p.Name(), func(t *testing.T) {
			pl, err := p.Place(context.Background(), r, snap, time.Now())
			if err != nil {
				t.Fatal(err)
			}
			seen := map[string]bool{}
			for _, id := range pl.GPUIDs {
				if seen[id] {
					t.Fatalf("duplicate GPU ID in placement: %v", pl.GPUIDs)
				}
				seen[id] = true
			}
		})
	}
}

// TestInvariant_SelectedGPUsBelongToNamedWorker checks every selected GPU
// ID actually exists on the placement's WorkerID in the snapshot.
func TestInvariant_SelectedGPUsBelongToNamedWorker(t *testing.T) {
	snap := testSnapshot(
		testWorker("w1", domain.WorkerReady, testGPU("g1"), testGPU("g2")),
		testWorker("w2", domain.WorkerReady, testGPU("g3")),
	)
	r := req("wl1", 2)
	for _, p := range allPolicies {
		t.Run(p.Name(), func(t *testing.T) {
			pl, err := p.Place(context.Background(), r, snap, time.Now())
			if err != nil {
				t.Fatal(err)
			}
			var worker domain.WorkerSnapshot
			for _, w := range snap.Workers {
				if w.ID == pl.WorkerID {
					worker = w
				}
			}
			ids := map[string]bool{}
			for _, g := range worker.GPUs {
				ids[g.ID] = true
			}
			for _, id := range pl.GPUIDs {
				if !ids[id] {
					t.Fatalf("selected GPU %s does not belong to worker %s", id, pl.WorkerID)
				}
			}
		})
	}
}

func deepCopySnapshot(s domain.ClusterSnapshot) domain.ClusterSnapshot {
	out := domain.ClusterSnapshot{Timestamp: s.Timestamp}
	for _, w := range s.Workers {
		wc := w
		wc.GPUs = append([]domain.GPUSnapshot(nil), w.GPUs...)
		out.Workers = append(out.Workers, wc)
	}
	return out
}
