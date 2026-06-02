"""Load Balancer: Distributed model serving with load balancing"""

import torch
import torch.nn as nn
from typing import List, Dict, Any
import numpy as np
from datetime import datetime
import threading

class InferenceReplica:
    """Single inference replica"""

    def __init__(self, replica_id: int, model: nn.Module):
        self.replica_id = replica_id
        self.model = model
        self.model.eval()
        self.request_count = 0
        self.total_latency = 0
        self.errors = 0

    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        """Single replica prediction"""
        start = datetime.now()
        try:
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                predictions = self.model(X_tensor).numpy()

            latency_ms = (datetime.now() - start).total_seconds() * 1000
            self.request_count += 1
            self.total_latency += latency_ms

            return {
                "predictions": predictions.flatten().tolist(),
                "latency_ms": latency_ms,
                "replica_id": self.replica_id,
                "success": True
            }
        except Exception as e:
            self.errors += 1
            return {
                "error": str(e),
                "replica_id": self.replica_id,
                "success": False
            }

class LoadBalancer:
    """Load balancer for distributed inference"""

    def __init__(self, num_replicas: int, model: nn.Module):
        self.replicas = [InferenceReplica(i, model) for i in range(num_replicas)]
        self.replica_index = 0
        self.lock = threading.Lock()

    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        """Predict with load balancing (round-robin)"""
        with self.lock:
            replica = self.replicas[self.replica_index % len(self.replicas)]
            self.replica_index += 1

        return replica.predict(X)

    def get_cluster_health(self) -> Dict:
        """Get overall cluster health"""
        replica_health = []
        for r in self.replicas:
            avg_latency = r.total_latency / max(r.request_count, 1)
            replica_health.append({
                "replica_id": r.replica_id,
                "request_count": r.request_count,
                "avg_latency_ms": round(avg_latency, 2),
                "error_rate": r.errors / max(r.request_count, 1)
            })

        total_requests = sum(r["request_count"] for r in replica_health)
        avg_latency = sum(r["avg_latency_ms"] for r in replica_health) / len(replica_health) if replica_health else 0
        return {
            "num_replicas": len(self.replicas),
            "total_requests": total_requests,
            "avg_latency_ms": round(avg_latency, 2),
            "replicas": replica_health
        }
