package scheduler

import (
	"context"
	"testing"
	"time"

	"gpuforge/internal/domain"
)

func TestUtilizationAware_PicksLowestUtilization(t *testing.T) {
	snap := testSnapshot(testWorker("w1", domain.WorkerReady,
		testGPU("g1", withUtilization(80)),
		testGPU("g2", withUtilization(5)),
	))
	pl, err := UtilizationAware{}.Place(context.Background(), req("wl1", 1), snap, time.Now())
	if err != nil {
		t.Fatal(err)
	}
	if pl.GPUIDs[0] != "g2" {
		t.Fatalf("expected UtilizationAware to prefer least-utilized g2, got %v", pl.GPUIDs)
	}
	if pl.Score != 5 {
		t.Fatalf("expected score 5 (avg utilization), got %v", pl.Score)
	}
}

func TestUtilizationAware_AcrossWorkers(t *testing.T) {
	snap := testSnapshot(
		testWorker("w1", domain.WorkerReady, testGPU("g1", withUtilization(90))),
		testWorker("w2", domain.WorkerReady, testGPU("g2", withUtilization(10))),
	)
	pl, err := UtilizationAware{}.Place(context.Background(), req("wl1", 1), snap, time.Now())
	if err != nil {
		t.Fatal(err)
	}
	if pl.WorkerID != "w2" {
		t.Fatalf("expected UtilizationAware to prefer worker w2's idle GPU, got %s", pl.WorkerID)
	}
}
