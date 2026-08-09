package orchestrator

import "time"

// RetryPolicy bounds how many times, and how quickly, the orchestrator
// retries. The same policy governs both kinds of retry this phase handles:
// an execution failure (RETRYING state) and a scheduling failure (a QUEUED
// workload that found no compatible GPU) — both are "try again later, up
// to a limit" situations, and giving them one shared, explicit knob is
// simpler than inventing two independent policies with no behavioral
// difference. Defaults match docs/failure-model.md's bounded-backoff spec.
type RetryPolicy struct {
	MaxAttempts int
	BaseDelay   time.Duration
	Factor      float64
	MaxDelay    time.Duration
}

// DefaultRetryPolicy is 3 attempts, base 1s, factor 2, max 30s — the values
// docs/failure-model.md already specifies.
var DefaultRetryPolicy = RetryPolicy{
	MaxAttempts: 3,
	BaseDelay:   time.Second,
	Factor:      2,
	MaxDelay:    30 * time.Second,
}

// NextDelay returns how long to wait before attempt number `attempt`
// (1-indexed: the delay before the *first* retry, i.e. attempt=1, is
// BaseDelay). Deterministic: no jitter, so tests never need randomness or
// real sleeps — callers advance a test clock instead.
func (p RetryPolicy) NextDelay(attempt int) time.Duration {
	if attempt < 1 {
		attempt = 1
	}
	delay := p.BaseDelay
	for i := 1; i < attempt; i++ {
		delay = time.Duration(float64(delay) * p.Factor)
		if delay > p.MaxDelay {
			return p.MaxDelay
		}
	}
	if delay > p.MaxDelay {
		return p.MaxDelay
	}
	return delay
}

// Exhausted reports whether `attempts` already-made attempts have used up
// the policy's budget.
func (p RetryPolicy) Exhausted(attempts int) bool {
	return attempts >= p.MaxAttempts
}
