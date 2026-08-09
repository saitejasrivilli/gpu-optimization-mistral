package scheduler

import (
	"context"
	"testing"
	"time"

	"gpuforge/internal/domain"
)

func TestBestFit_PicksTightestFit(t *testing.T) {
	snap := testSnapshot(testWorker("w1", domain.WorkerReady,
		testGPU("g1", withMemory(80<<30)), // large, wasteful
		testGPU("g2", withMemory(16<<30)), // tightest fit for a 8GB min
	))
	pl, err := BestFit{}.Place(context.Background(), req("wl1", 1, withMinMemory(8<<30)), snap, time.Now())
	if err != nil {
		t.Fatal(err)
	}
	if pl.GPUIDs[0] != "g2" {
		t.Fatalf("expected BestFit to prefer tightest-fitting g2, got %v", pl.GPUIDs)
	}
	wantLeftover := float64(16<<30 - 8<<30)
	if pl.Score != wantLeftover {
		t.Fatalf("expected score (leftover bytes) %v, got %v", wantLeftover, pl.Score)
	}
}

func TestBestFit_AcrossWorkers(t *testing.T) {
	snap := testSnapshot(
		testWorker("w1", domain.WorkerReady, testGPU("g1", withMemory(80<<30))),
		testWorker("w2", domain.WorkerReady, testGPU("g2", withMemory(16<<30))),
	)
	pl, err := BestFit{}.Place(context.Background(), req("wl1", 1, withMinMemory(8<<30)), snap, time.Now())
	if err != nil {
		t.Fatal(err)
	}
	if pl.WorkerID != "w2" {
		t.Fatalf("expected BestFit to prefer worker w2's tighter fit, got %s", pl.WorkerID)
	}
}
