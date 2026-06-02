"""Prometheus metrics for monitoring"""

from prometheus_client import Counter, Histogram

prediction_counter = Counter(
    'predictions_total',
    'Total predictions',
    ['model', 'success']
)

prediction_latency = Histogram(
    'prediction_latency_ms',
    'Prediction latency in milliseconds'
)
