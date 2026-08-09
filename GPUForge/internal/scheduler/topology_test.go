package scheduler

import (
	"context"
	"errors"
	"testing"
	"time"

	"gpuforge/internal/domain"
)

func TestTopologyAware_PrefersSameNVLinkGroup(t *testing.T) {
	// g1+g2 share a group; g3 does not. Requesting 2 GPUs with no hard
	// topology requirement should still prefer the connected pair.
	snap := testSnapshot(testWorker("w1", domain.WorkerReady,
		testGPU("g1", withNVLinkGroup("A")),
		testGPU("g2", withNVLinkGroup("A")),
		testGPU("g3"),
	))
	pl, err := TopologyAware{}.Place(context.Background(), req("wl1", 2), snap, time.Now())
	if err != nil {
		t.Fatal(err)
	}
	if pl.Score != 1.0 {
		t.Fatalf("expected cohesion score 1.0 for a same-group pair, got %v", pl.Score)
	}
	got := map[string]bool{pl.GPUIDs[0]: true, pl.GPUIDs[1]: true}
	if !got["g1"] || !got["g2"] {
		t.Fatalf("expected topology-aware to select the connected pair g1+g2, got %v", pl.GPUIDs)
	}
}

func TestTopologyAware_MissingTopologyInfoFallsBackHonestly(t *testing.T) {
	// No GPU reports any NVLink group at all -> fallback path, cohesion 0,
	// but scheduling still succeeds rather than refusing outright.
	snap := testSnapshot(testWorker("w1", domain.WorkerReady, testGPU("g1"), testGPU("g2")))
	pl, err := TopologyAware{}.Place(context.Background(), req("wl1", 2), snap, time.Now())
	if err != nil {
		t.Fatal(err)
	}
	if pl.Score != 0 {
		t.Fatalf("expected cohesion score 0 when topology is entirely unknown, got %v", pl.Score)
	}
	if len(pl.GPUIDs) != 2 {
		t.Fatalf("expected fallback placement to still select 2 GPUs, got %v", pl.GPUIDs)
	}
}

func TestTopologyAware_SingleGPUAlwaysScoresFull(t *testing.T) {
	snap := testSnapshot(testWorker("w1", domain.WorkerReady, testGPU("g1")))
	pl, err := TopologyAware{}.Place(context.Background(), req("wl1", 1), snap, time.Now())
	if err != nil {
		t.Fatal(err)
	}
	if pl.Score != 1.0 {
		t.Fatalf("expected single-GPU request to score 1.0 (nothing to connect), got %v", pl.Score)
	}
}

func TestHardTopologyRequirement_NVLinkGroup_Satisfied(t *testing.T) {
	snap := testSnapshot(testWorker("w1", domain.WorkerReady,
		testGPU("g1", withNVLinkGroup("A")),
		testGPU("g2", withNVLinkGroup("A")),
		testGPU("g3"),
	))
	r := req("wl1", 2, withTopologyRequirement(domain.TopologyNVLinkGroup))
	for _, p := range allPolicies {
		t.Run(p.Name(), func(t *testing.T) {
			pl, err := p.Place(context.Background(), r, snap, time.Now())
			if err != nil {
				t.Fatal(err)
			}
			got := map[string]bool{}
			for _, id := range pl.GPUIDs {
				got[id] = true
			}
			if !got["g1"] || !got["g2"] {
				t.Fatalf("expected NVLink-group requirement to force selection of g1+g2, got %v", pl.GPUIDs)
			}
		})
	}
}

func TestHardTopologyRequirement_NVLinkGroup_RejectedWhenNoGroupBigEnough(t *testing.T) {
	// Two GPUs, neither sharing a group with the other (or with anything).
	snap := testSnapshot(testWorker("w1", domain.WorkerReady, testGPU("g1"), testGPU("g2")))
	r := req("wl1", 2, withTopologyRequirement(domain.TopologyNVLinkGroup))
	for _, p := range allPolicies {
		t.Run(p.Name(), func(t *testing.T) {
			_, err := p.Place(context.Background(), r, snap, time.Now())
			if !errors.Is(err, ErrIncompatibleTopology) {
				t.Fatalf("expected ErrIncompatibleTopology, got %v", err)
			}
		})
	}
}

func TestHardTopologyRequirement_NVLinkGroup_SingleGPUAlwaysOK(t *testing.T) {
	snap := testSnapshot(testWorker("w1", domain.WorkerReady, testGPU("g1")))
	r := req("wl1", 1, withTopologyRequirement(domain.TopologyNVLinkGroup))
	for _, p := range allPolicies {
		t.Run(p.Name(), func(t *testing.T) {
			if _, err := p.Place(context.Background(), r, snap, time.Now()); err != nil {
				t.Fatalf("expected single-GPU NVLink requirement to trivially succeed, got %v", err)
			}
		})
	}
}

func TestHardTopologyRequirement_SameNode_AlwaysSatisfied(t *testing.T) {
	// SameNode is always satisfied in this domain model since an Allocation
	// is always scoped to one worker; this locks in that documented
	// behavior against regression.
	snap := testSnapshot(testWorker("w1", domain.WorkerReady, testGPU("g1"), testGPU("g2")))
	r := req("wl1", 2, withTopologyRequirement(domain.TopologySameNode))
	for _, p := range allPolicies {
		t.Run(p.Name(), func(t *testing.T) {
			if _, err := p.Place(context.Background(), r, snap, time.Now()); err != nil {
				t.Fatalf("expected SameNode requirement to be trivially satisfied, got %v", err)
			}
		})
	}
}
