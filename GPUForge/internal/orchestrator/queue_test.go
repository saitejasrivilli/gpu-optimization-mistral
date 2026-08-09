package orchestrator

import (
	"testing"
	"time"

	"gpuforge/internal/domain"
)

func TestQueue_FIFOWithinEqualPriority(t *testing.T) {
	q := NewQueue()
	base := time.Now()
	q.Enqueue(QueueItem{Requirements: domain.WorkloadRequirements{WorkloadID: "a"}, EnqueuedAt: base})
	q.Enqueue(QueueItem{Requirements: domain.WorkloadRequirements{WorkloadID: "b"}, EnqueuedAt: base.Add(time.Second)})
	q.Enqueue(QueueItem{Requirements: domain.WorkloadRequirements{WorkloadID: "c"}, EnqueuedAt: base.Add(2 * time.Second)})

	for _, want := range []string{"a", "b", "c"} {
		item, ok := q.Dequeue()
		if !ok || item.Requirements.WorkloadID != want {
			t.Fatalf("expected %s, got %+v (ok=%v)", want, item, ok)
		}
	}
	if _, ok := q.Dequeue(); ok {
		t.Fatal("expected empty queue")
	}
}

func TestQueue_PriorityOrdering(t *testing.T) {
	q := NewQueue()
	base := time.Now()
	q.Enqueue(QueueItem{Requirements: domain.WorkloadRequirements{WorkloadID: "low", Priority: 1}, EnqueuedAt: base})
	q.Enqueue(QueueItem{Requirements: domain.WorkloadRequirements{WorkloadID: "high", Priority: 10}, EnqueuedAt: base.Add(time.Second)})

	item, ok := q.Dequeue()
	if !ok || item.Requirements.WorkloadID != "high" {
		t.Fatalf("expected higher priority item first, got %+v", item)
	}
}

func TestQueue_Remove(t *testing.T) {
	q := NewQueue()
	q.Enqueue(QueueItem{Requirements: domain.WorkloadRequirements{WorkloadID: "a"}})
	q.Enqueue(QueueItem{Requirements: domain.WorkloadRequirements{WorkloadID: "b"}})

	if !q.Remove("a") {
		t.Fatal("expected removal of a to succeed")
	}
	if q.Remove("a") {
		t.Fatal("expected second removal of a to report not-found")
	}
	if q.Len() != 1 {
		t.Fatalf("expected 1 remaining item, got %d", q.Len())
	}
}

func TestQueue_DeterministicTieBreakByWorkloadID(t *testing.T) {
	q := NewQueue()
	same := time.Now()
	q.Enqueue(QueueItem{Requirements: domain.WorkloadRequirements{WorkloadID: "z"}, EnqueuedAt: same})
	q.Enqueue(QueueItem{Requirements: domain.WorkloadRequirements{WorkloadID: "a"}, EnqueuedAt: same})

	item, _ := q.Dequeue()
	if item.Requirements.WorkloadID != "a" {
		t.Fatalf("expected deterministic tie-break by WorkloadID (a before z), got %s", item.Requirements.WorkloadID)
	}
}

func TestQueue_NoLostWorkloads(t *testing.T) {
	q := NewQueue()
	for i := 0; i < 50; i++ {
		q.Enqueue(QueueItem{Requirements: domain.WorkloadRequirements{WorkloadID: string(rune('a'+i%26)) + "-" + string(rune(i))}})
	}
	count := 0
	for {
		if _, ok := q.Dequeue(); !ok {
			break
		}
		count++
	}
	if count != 50 {
		t.Fatalf("expected all 50 items dequeued, got %d", count)
	}
}
