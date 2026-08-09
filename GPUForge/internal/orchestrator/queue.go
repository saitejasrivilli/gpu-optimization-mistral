package orchestrator

import (
	"sort"
	"sync"
	"time"

	"gpuforge/internal/domain"
)

// QueueItem is one pending workload, along with when it was enqueued
// (breaks ties within equal priority) and how many scheduling attempts it
// has already exhausted (unschedulable-after-max-attempts and retry
// bookkeeping both live in the orchestrator, not here — the queue only
// orders and stores).
type QueueItem struct {
	Requirements domain.WorkloadRequirements
	EnqueuedAt   time.Time
	Attempts     int
}

// Queue is an in-memory, priority-ordered, FIFO-within-priority workload
// queue. Not distributed — see docs/orchestration.md for why that's out of
// scope. Safe for concurrent use.
type Queue struct {
	mu    sync.Mutex
	items []QueueItem
}

func NewQueue() *Queue {
	return &Queue{}
}

// Enqueue adds item to the queue. It does not deduplicate — callers
// (Orchestrator) are responsible for ensuring a given WorkloadID is never
// enqueued twice at once.
func (q *Queue) Enqueue(item QueueItem) {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.items = append(q.items, item)
}

// Dequeue removes and returns the highest-priority item (ties broken by
// earliest EnqueuedAt, then WorkloadID for full determinism), or ok=false
// if the queue is empty. No workload is ever silently dropped: Dequeue
// only removes what it returns.
func (q *Queue) Dequeue() (QueueItem, bool) {
	q.mu.Lock()
	defer q.mu.Unlock()
	if len(q.items) == 0 {
		return QueueItem{}, false
	}
	best := 0
	for i := 1; i < len(q.items); i++ {
		if lessQueueItem(q.items[i], q.items[best]) {
			best = i
		}
	}
	item := q.items[best]
	q.items = append(q.items[:best], q.items[best+1:]...)
	return item, true
}

func lessQueueItem(a, b QueueItem) bool {
	if a.Requirements.Priority != b.Requirements.Priority {
		return a.Requirements.Priority > b.Requirements.Priority // higher priority first
	}
	if !a.EnqueuedAt.Equal(b.EnqueuedAt) {
		return a.EnqueuedAt.Before(b.EnqueuedAt)
	}
	return a.Requirements.WorkloadID < b.Requirements.WorkloadID
}

// Remove deletes the item for workloadID, if present, and reports whether
// it was found — used by cancellation of a still-queued workload.
func (q *Queue) Remove(workloadID string) bool {
	q.mu.Lock()
	defer q.mu.Unlock()
	for i, item := range q.items {
		if item.Requirements.WorkloadID == workloadID {
			q.items = append(q.items[:i], q.items[i+1:]...)
			return true
		}
	}
	return false
}

// Len reports the number of queued items.
func (q *Queue) Len() int {
	q.mu.Lock()
	defer q.mu.Unlock()
	return len(q.items)
}

// Snapshot returns a deterministically ordered (by the same priority/FIFO
// rule as Dequeue) copy of the queue's contents, for inspection/tests.
func (q *Queue) Snapshot() []QueueItem {
	q.mu.Lock()
	defer q.mu.Unlock()
	out := make([]QueueItem, len(q.items))
	copy(out, q.items)
	sort.Slice(out, func(i, j int) bool { return lessQueueItem(out[i], out[j]) })
	return out
}
