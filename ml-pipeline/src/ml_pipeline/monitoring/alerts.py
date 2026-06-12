"""Alert System: Automated alerting for SLO violations, drift, and anomalies"""

from enum import Enum
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = 0
    WARNING = 1
    CRITICAL = 2


class AlertType(Enum):
    """Types of alerts"""
    SLO_VIOLATION = "slo_violation"
    ACCURACY_DRIFT = "accuracy_drift"
    INFERENCE_TIMEOUT = "inference_timeout"
    ERROR_SPIKE = "error_spike"
    LATENCY_SPIKE = "latency_spike"
    MEMORY_PRESSURE = "memory_pressure"


@dataclass
class Alert:
    """Alert event"""
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    message: str
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    tags: Dict[str, str] = field(default_factory=dict)
    resolved_at: Optional[float] = None

    def is_resolved(self) -> bool:
        return self.resolved_at is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "alert_type": self.alert_type.value,
            "severity": self.severity.name,
            "message": self.message,
            "timestamp": self.timestamp,
            "tags": self.tags,
            "resolved": self.is_resolved()
        }


class AlertHandler:
    """Base class for alert handlers"""

    def handle(self, alert: Alert) -> bool:
        """Handle alert, return True if processed"""
        raise NotImplementedError


class LogAlertHandler(AlertHandler):
    """Log alerts to console/file"""

    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file
        self.alerts_logged = []

    def handle(self, alert: Alert) -> bool:
        message = f"[{alert.severity.name}] {alert.alert_type.value}: {alert.message}"

        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(f"{datetime.fromtimestamp(alert.timestamp)} {message}\n")

        print(message)
        self.alerts_logged.append(alert)
        return True


class MetricsAlertHandler(AlertHandler):
    """Emit metrics for alerting systems (Prometheus, etc)"""

    def __init__(self):
        self.alerts_by_type = {}

    def handle(self, alert: Alert) -> bool:
        alert_type = alert.alert_type.value
        if alert_type not in self.alerts_by_type:
            self.alerts_by_type[alert_type] = 0
        self.alerts_by_type[alert_type] += 1
        return True


class ThresholdMonitor:
    """Monitor metric thresholds and trigger alerts"""

    def __init__(self, alert_manager: 'AlertManager'):
        self.alert_manager = alert_manager
        self.thresholds = {
            "p99_latency_ms": 100,
            "p95_latency_ms": 50,
            "error_rate": 0.01,
            "accuracy_drift_pct": 5.0,
            "memory_usage_pct": 90.0,
        }
        self.violation_history = {}

    def set_threshold(self, metric_name: str, threshold: float):
        """Set threshold for metric"""
        self.thresholds[metric_name] = threshold

    def check_metric(self, metric_name: str, value: float) -> Optional[Alert]:
        """Check metric against threshold, return alert if violated"""
        if metric_name not in self.thresholds:
            return None

        threshold = self.thresholds[metric_name]

        # Determine alert type
        alert_type_map = {
            "p99_latency_ms": AlertType.LATENCY_SPIKE,
            "p95_latency_ms": AlertType.LATENCY_SPIKE,
            "error_rate": AlertType.ERROR_SPIKE,
            "accuracy_drift_pct": AlertType.ACCURACY_DRIFT,
            "memory_usage_pct": AlertType.MEMORY_PRESSURE,
        }

        if value > threshold:
            alert_type = alert_type_map.get(metric_name, AlertType.SLO_VIOLATION)

            # Determine severity
            severity_threshold = threshold * 1.5
            severity = AlertSeverity.CRITICAL if value > severity_threshold else AlertSeverity.WARNING

            alert = Alert(
                alert_id=f"{metric_name}_{int(datetime.now().timestamp() * 1000)}",
                alert_type=alert_type,
                severity=severity,
                message=f"{metric_name} exceeded threshold: {value:.2f} > {threshold}",
                tags={"metric": metric_name, "value": str(value), "threshold": str(threshold)}
            )

            # Record violation
            if metric_name not in self.violation_history:
                self.violation_history[metric_name] = []
            self.violation_history[metric_name].append(value)

            return alert

        return None

    def get_violation_stats(self, metric_name: str) -> Dict[str, float]:
        """Get violation statistics"""
        if metric_name not in self.violation_history:
            return {}

        violations = self.violation_history[metric_name]
        return {
            "total_violations": len(violations),
            "max_value": max(violations),
            "avg_value": sum(violations) / len(violations),
            "threshold": self.thresholds.get(metric_name, 0)
        }


class AlertManager:
    """Central alert management system"""

    def __init__(self):
        self.handlers: List[AlertHandler] = []
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.threshold_monitor = ThresholdMonitor(self)

    def register_handler(self, handler: AlertHandler):
        """Register alert handler"""
        self.handlers.append(handler)

    def emit_alert(self, alert: Alert) -> bool:
        """Emit alert to all handlers"""
        if alert.alert_id in self.active_alerts and not alert.is_resolved():
            return False  # Duplicate alert, suppress

        self.active_alerts[alert.alert_id] = alert
        self.alert_history.append(alert)

        # Notify all handlers
        for handler in self.handlers:
            try:
                handler.handle(alert)
            except Exception as e:
                print(f"Handler error: {e}")

        return True

    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve active alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved_at = datetime.now().timestamp()
            return True
        return False

    def check_slo_violation(self, metric_name: str, value: float) -> bool:
        """Check metric and emit SLO violation alert if needed"""
        alert = self.threshold_monitor.check_metric(metric_name, value)
        if alert:
            self.emit_alert(alert)
            return True
        return False

    def check_accuracy_drift(self, current_accuracy: float, baseline_accuracy: float, model_version: str) -> bool:
        """Check accuracy drift and emit alert if significant"""
        drift_pct = 100 * (baseline_accuracy - current_accuracy) / baseline_accuracy

        if drift_pct > self.threshold_monitor.thresholds["accuracy_drift_pct"]:
            alert = Alert(
                alert_id=f"drift_{model_version}_{int(datetime.now().timestamp() * 1000)}",
                alert_type=AlertType.ACCURACY_DRIFT,
                severity=AlertSeverity.WARNING,
                message=f"Model {model_version} accuracy degraded by {drift_pct:.1f}%",
                tags={"model": model_version, "drift_pct": str(drift_pct), "current": str(current_accuracy)}
            )
            self.emit_alert(alert)
            return True

        return False

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active alerts"""
        return [alert.to_dict() for alert in self.active_alerts.values() if not alert.is_resolved()]

    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary"""
        active = self.get_active_alerts()
        critical_count = sum(1 for a in active if a["severity"] == AlertSeverity.CRITICAL.name)

        return {
            "total_alerts_generated": len(self.alert_history),
            "active_alerts": len(active),
            "critical_alerts": critical_count,
            "alerts_by_type": self._count_alerts_by_type(),
            "recent_alerts": [a.to_dict() for a in self.alert_history[-5:]]
        }

    def _count_alerts_by_type(self) -> Dict[str, int]:
        """Count alerts by type"""
        counts = {}
        for alert in self.alert_history:
            alert_type = alert.alert_type.value
            counts[alert_type] = counts.get(alert_type, 0) + 1
        return counts


# Global alert manager instance
_global_alert_manager = None


def get_alert_manager() -> AlertManager:
    """Get global alert manager instance"""
    global _global_alert_manager
    if _global_alert_manager is None:
        _global_alert_manager = AlertManager()
        _global_alert_manager.register_handler(LogAlertHandler())
        _global_alert_manager.register_handler(MetricsAlertHandler())
    return _global_alert_manager


def initialize_alerting(log_file: Optional[str] = None) -> AlertManager:
    """Initialize alert system"""
    global _global_alert_manager
    _global_alert_manager = AlertManager()
    _global_alert_manager.register_handler(LogAlertHandler(log_file))
    _global_alert_manager.register_handler(MetricsAlertHandler())
    return _global_alert_manager
