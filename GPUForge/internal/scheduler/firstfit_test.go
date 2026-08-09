package scheduler

import (
	"context"
	"testing"
	"time"

	"gpuforge/internal/domain"
)

func TestFirstFit_PicksFirstEligibleRegardlessOfFitOrUtilization(t *testing.T) {
	// g1 is a worse fit and more utilized than g2; FirstFit should still
	// take g1 because it comes first in ID order — that's the whole point
	// of the baseline policy.
	snap := testSnapshot(testWorker("w1", domain.WorkerReady,
		testGPU("g1", withMemory(80<<30), withUtilization(90)),
		testGPU("g2", withMemory(16<<30), withUtilization(0)),
	))
	pl, err := FirstFit{}.Place(context.Background(), req("wl1", 1, withMinMemory(8<<30)), snap, time.Now())
	if err != nil {
		t.Fatal(err)
	}
	if pl.GPUIDs[0] != "g1" {
		t.Fatalf("expected FirstFit to take g1, got %v", pl.GPUIDs)
	}
	if pl.Policy != "first-fit" {
		t.Fatalf("unexpected policy name %q", pl.Policy)
	}
}
