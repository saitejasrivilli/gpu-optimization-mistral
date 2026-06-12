"""Batch Scheduler: Deadline-aware batch scheduling and routing"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import heapq
import numpy as np
from enum import Enum


class Priority(Enum):
    """Request priority levels"""
    LOW = 3
    MEDIUM = 2
    HIGH = 1
    CRITICAL = 0


@dataclass
class InferenceRequest:
    """Inference request with deadline"""
    request_id: str
    data: np.ndarray
    deadline_ms: float
    priority: Priority = Priority.MEDIUM
    created_at: float = field(default_factory=lambda: datetime.now().timestamp() * 1000)
    assigned_replica: Optional[int] = None
    completed_at: Optional[float] = None

    def time_until_deadline(self) -> float:
        """Milliseconds until deadline"""
        now = datetime.now().timestamp() * 1000
        return self.deadline_ms - (now - self.created_at)

    def is_overdue(self) -> bool:
        """Check if request exceeded deadline"""
        return self.time_until_deadline() < 0

    def urgency_score(self) -> Tuple[int, float]:
        """Priority queue: (priority_value, -time_to_deadline)"""
        return (self.priority.value, -self.time_until_deadline())


@dataclass
class ReplicaCapacity:
    """Track replica capacity and load"""
    replica_id: int
    model: nn.Module
    max_batch_size: int = 32
    current_load: int = 0
    pending_requests: List[InferenceRequest] = field(default_factory=list)
    total_processed: int = 0
    missed_deadlines: int = 0

    def available_capacity(self) -> int:
        """Available slots in batch"""
        return self.max_batch_size - self.current_load

    def can_accept(self, batch_size: int) -> bool:
        """Check if replica can accept batch"""
        return self.available_capacity() >= batch_size

    def utilization(self) -> float:
        """Current utilization percentage"""
        return self.current_load / self.max_batch_size

    def deadline_miss_rate(self) -> float:
        """Percentage of deadline misses"""
        return self.missed_deadlines / max(self.total_processed, 1)


class BatchScheduler:
    """Deadline-aware batch scheduler"""

    def __init__(self, num_replicas: int, model: nn.Module, max_batch_size: int = 32):
        self.replicas = [
            ReplicaCapacity(i, model, max_batch_size)
            for i in range(num_replicas)
        ]
        self.request_queue: List[InferenceRequest] = []
        self.pending_batches: Dict[int, List[InferenceRequest]] = {i: [] for i in range(num_replicas)}
        self.completed_requests: List[InferenceRequest] = []

    def enqueue_request(self, request: InferenceRequest):
        """Add request to priority queue"""
        heapq.heappush(self.request_queue, (request.urgency_score(), request))

    def schedule_batch(self) -> Dict[str, Any]:
        """Schedule next batch considering deadlines"""
        if not self.request_queue:
            return {"scheduled": False, "reason": "No pending requests"}

        # Pull high-priority requests
        batch = []
        batch_deadline = float('inf')

        while self.request_queue and len(batch) < 32:  # Max batch
            priority_score, request = heapq.heappop(self.request_queue)

            if request.is_overdue():
                # Overdue: skip to next
                continue

            batch.append(request)
            batch_deadline = min(batch_deadline, request.deadline_ms)

        if not batch:
            return {"scheduled": False, "reason": "All requests overdue"}

        # Find best replica (deadline-aware)
        best_replica = self._select_replica_for_batch(batch)
        if best_replica is None:
            return {"scheduled": False, "reason": "No available replica"}

        # Assign batch to replica
        replica = self.replicas[best_replica]
        for request in batch:
            request.assigned_replica = best_replica
            replica.pending_requests.append(request)
            self.pending_batches[best_replica].append(request)

        return {
            "scheduled": True,
            "batch_size": len(batch),
            "assigned_replica": best_replica,
            "batch_deadline_ms": batch_deadline,
            "replicas_available": sum(1 for r in self.replicas if r.available_capacity() > 0)
        }

    def _select_replica_for_batch(self, batch: List[InferenceRequest]) -> Optional[int]:
        """Select replica using deadline-aware routing"""
        batch_size = len(batch)
        candidates = []

        for replica in self.replicas:
            if not replica.can_accept(batch_size):
                continue

            # Score: prefer low utilization + low deadline miss rate
            score = (
                replica.utilization(),
                replica.deadline_miss_rate(),
                replica.replica_id  # Tie-breaker
            )
            candidates.append((score, replica.replica_id))

        if not candidates:
            return None

        # Sort by score (ascending)
        candidates.sort()
        return candidates[0][1]

    def process_batch(self, replica_id: int) -> Dict[str, Any]:
        """Process batch on replica and check deadline compliance"""
        replica = self.replicas[replica_id]

        if not replica.pending_requests:
            return {"processed": 0, "deadline_met": 0, "deadline_missed": 0}

        batch_data = np.array([r.data for r in replica.pending_requests])

        # Simulate inference
        try:
            with torch.no_grad():
                X = torch.FloatTensor(batch_data)
                outputs = replica.model(X).detach().numpy()
        except Exception as e:
            return {"error": str(e), "processed": 0}

        # Check deadline compliance
        deadline_met = 0
        deadline_missed = 0

        for request in replica.pending_requests:
            request.completed_at = datetime.now().timestamp() * 1000

            if not request.is_overdue():
                deadline_met += 1
            else:
                deadline_missed += 1
                replica.missed_deadlines += 1

            replica.total_processed += 1
            self.completed_requests.append(request)

        replica.current_load = 0
        replica.pending_requests.clear()
        self.pending_batches[replica_id].clear()

        return {
            "processed": len(batch_data),
            "deadline_met": deadline_met,
            "deadline_missed": deadline_missed,
            "batch_outputs": outputs
        }

    def get_scheduler_stats(self) -> Dict[str, Any]:
        """Get scheduling statistics"""
        total_requests = len(self.completed_requests)
        total_deadline_met = sum(
            1 for r in self.completed_requests
            if not r.is_overdue()
        )

        replica_stats = []
        for replica in self.replicas:
            replica_stats.append({
                "replica_id": replica.replica_id,
                "utilization": replica.utilization(),
                "total_processed": replica.total_processed,
                "deadline_miss_rate": replica.deadline_miss_rate(),
                "pending_requests": len(replica.pending_requests)
            })

        return {
            "total_requests": total_requests,
            "deadline_compliance_rate": total_deadline_met / max(total_requests, 1),
            "pending_in_queue": len(self.request_queue),
            "replicas": replica_stats,
            "avg_batch_utilization": np.mean([r.utilization() for r in self.replicas])
        }

    def get_replica_recommendations(self) -> Dict[str, str]:
        """Get recommendations for load balancing"""
        recommendations = {}

        max_utilization = max((r.utilization() for r in self.replicas), default=0)
        min_utilization = min((r.utilization() for r in self.replicas), default=0)

        if max_utilization - min_utilization > 0.3:
            recommendations["load_imbalance"] = "High load imbalance detected, consider scaling"

        for replica in self.replicas:
            if replica.deadline_miss_rate() > 0.05:
                recommendations[f"replica_{replica.replica_id}"] = "High deadline miss rate, reduce batch size or add replica"

        if len(self.request_queue) > 100:
            recommendations["queue_buildup"] = "Excessive request queue, add more replicas"

        return recommendations if recommendations else {"status": "All systems normal"}


class DeadlineAwareRouter:
    """Route requests to replicas based on deadlines"""

    def __init__(self, scheduler: BatchScheduler):
        self.scheduler = scheduler
        self.routing_decisions = []

    def route_request(self, request: InferenceRequest) -> int:
        """Route single request considering its deadline"""
        # Add to queue
        self.scheduler.enqueue_request(request)

        # Trigger scheduling
        schedule_result = self.scheduler.schedule_batch()

        if schedule_result.get("scheduled"):
            replica_id = schedule_result["assigned_replica"]
            self.routing_decisions.append({
                "request_id": request.request_id,
                "replica_id": replica_id,
                "deadline_ms": request.deadline_ms,
                "routed_at": datetime.now().timestamp() * 1000
            })
            return replica_id

        return -1  # Failed to route

    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get routing statistics"""
        return {
            "total_routed": len(self.routing_decisions),
            "scheduler_stats": self.scheduler.get_scheduler_stats(),
            "recommendations": self.scheduler.get_replica_recommendations()
        }
