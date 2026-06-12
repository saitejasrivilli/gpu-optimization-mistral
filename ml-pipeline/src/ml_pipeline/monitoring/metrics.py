"""Prometheus metrics for monitoring + SLO enforcement"""

from prometheus_client import Counter, Histogram, Gauge
import time
from typing import Optional

# ============================================================
# BASIC METRICS
# ============================================================

prediction_counter = Counter(
    'predictions_total',
    'Total predictions',
    ['model', 'success']
)

prediction_latency = Histogram(
    'prediction_latency_ms',
    'Prediction latency in milliseconds',
    buckets=[10, 25, 50, 100, 250, 500, 1000]
)

# ============================================================
# SLO METRICS & ENFORCEMENT
# ============================================================

# SLO targets (in milliseconds)
SLO_P99_LATENCY_MS = 100  # 99th percentile must be < 100ms
SLO_P95_LATENCY_MS = 50   # 95th percentile must be < 50ms
SLO_ERROR_RATE = 0.01     # Error rate must be < 1%

# SLO compliance gauges
slo_p99_latency = Histogram(
    'prediction_latency_p99_ms',
    'P99 prediction latency (SLO: < 100ms)',
    buckets=[25, 50, 75, 100, 150, 200]
)

slo_p95_latency = Histogram(
    'prediction_latency_p95_ms',
    'P95 prediction latency (SLO: < 50ms)',
    buckets=[10, 25, 50, 75, 100]
)

slo_error_rate = Gauge(
    'prediction_error_rate',
    'Prediction error rate (SLO: < 1%)'
)

# Model drift detection
model_accuracy = Gauge(
    'model_accuracy',
    'Current model accuracy',
    ['model_version']
)

model_accuracy_drift = Gauge(
    'model_accuracy_drift_pct',
    'Accuracy degradation from baseline (%)',
    ['model_version']
)

inference_timeout_count = Counter(
    'inference_timeout_count',
    'Inference requests exceeding SLO deadline',
    ['model']
)


# ============================================================
# SLO ENFORCEMENT CONTEXT MANAGER
# ============================================================

class SLOEnforcer:
    """Context manager for SLO-aware inference with deadline enforcement."""

    def __init__(self, model_name: str, deadline_ms: Optional[float] = None):
        """
        Args:
            model_name: Model identifier for metrics
            deadline_ms: Latency deadline (default: SLO_P99_LATENCY_MS)
        """
        self.model_name = model_name
        self.deadline_ms = deadline_ms or SLO_P99_LATENCY_MS
        self.start_time = None
        self.elapsed_ms = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_ms = (time.time() - self.start_time) * 1000

        # Record latency metric
        prediction_latency.observe(self.elapsed_ms)

        # Check SLO violation
        if self.elapsed_ms > self.deadline_ms:
            inference_timeout_count.labels(model=self.model_name).inc()
            # Log warning in production (omitted here)
            return False  # Could raise exception in strict mode

        return True

    def within_slo(self) -> bool:
        """Check if inference completed within SLO."""
        return self.elapsed_ms <= self.deadline_ms


# ============================================================
# MODEL PERFORMANCE TRACKING
# ============================================================

class ModelPerformanceTracker:
    """Track model accuracy, drift, and SLO compliance."""

    def __init__(self, baseline_accuracy: float = 0.95):
        self.baseline_accuracy = baseline_accuracy
        self.current_accuracy = baseline_accuracy

    def update_accuracy(self, model_version: str, accuracy: float):
        """Update model accuracy and detect drift."""
        self.current_accuracy = accuracy
        model_accuracy.labels(model_version=model_version).set(accuracy)

        # Calculate drift
        drift_pct = 100 * (self.baseline_accuracy - accuracy) / self.baseline_accuracy
        model_accuracy_drift.labels(model_version=model_version).set(drift_pct)

        if drift_pct > 5.0:
            # Alert in production: accuracy degraded >5%
            pass

    def update_error_rate(self, error_count: int, total_count: int):
        """Update error rate metric."""
        rate = error_count / max(total_count, 1)
        slo_error_rate.set(rate)
