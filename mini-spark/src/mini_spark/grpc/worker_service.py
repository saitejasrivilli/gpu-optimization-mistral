"""gRPC Worker Service - production-grade distributed execution"""

import logging
import pickle
import time
from datetime import datetime
from typing import Dict, Any, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExecuteTaskRequest:
    def __init__(self, task_id: str, stage_id: int, function_bytes: bytes,
                 args: List[bytes], dependencies: List[str]):
        self.task_id = task_id
        self.stage_id = stage_id
        self.function_bytes = function_bytes
        self.args = args
        self.dependencies = dependencies

class ExecuteTaskResponse:
    def __init__(self, task_id: str, success: bool, result: bytes = None,
                 error: str = None, latency_ms: float = 0.0):
        self.task_id = task_id
        self.success = success
        self.result = result
        self.error = error
        self.latency_ms = latency_ms

class StatusRequest:
    def __init__(self, worker_id: str):
        self.worker_id = worker_id

class StatusResponse:
    def __init__(self, worker_id: str, total_tasks: int, completed_tasks: int,
                 failed_tasks: int, avg_latency_ms: float):
        self.worker_id = worker_id
        self.total_tasks = total_tasks
        self.completed_tasks = completed_tasks
        self.failed_tasks = failed_tasks
        self.avg_latency_ms = avg_latency_ms

class ShutdownRequest:
    pass

class ShutdownResponse:
    def __init__(self, success: bool):
        self.success = success

class WorkerServicer:
    """Production gRPC Worker service"""

    def __init__(self, worker_id: str, state_store: Any):
        self.worker_id = worker_id
        self.state_store = state_store
        self.total_tasks = 0
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.task_latencies: List[float] = []
        logger.info(f"Worker {worker_id} initialized")

    def ExecuteTask(self, request: ExecuteTaskRequest, context) -> ExecuteTaskResponse:
        """Execute task via gRPC with real computation"""
        start = time.time()
        task_id = request.task_id

        try:
            logger.info(f"[{self.worker_id}] Executing task {task_id}")

            # Deserialize function
            func = pickle.loads(request.function_bytes)

            # Deserialize dependencies from state store
            dep_data = []
            if request.dependencies:
                for dep_id in request.dependencies:
                    dep_bytes = self.state_store.get_partition(dep_id, 0)
                    if dep_bytes:
                        dep_data.append(pickle.loads(dep_bytes))

            # Deserialize args
            args = [pickle.loads(arg_bytes) for arg_bytes in request.args]

            # Execute task with dependencies and args
            if dep_data:
                result = func(*dep_data, *args)
            else:
                result = func(*args)

            # Store result in distributed state
            result_bytes = pickle.dumps(result)
            self.state_store.store_partition(task_id, 0, result_bytes)

            latency_ms = (time.time() - start) * 1000
            self.task_latencies.append(latency_ms)
            self.completed_tasks += 1
            self.total_tasks += 1

            logger.info(f"[{self.worker_id}] Task {task_id} completed in {latency_ms:.2f}ms")

            return ExecuteTaskResponse(
                task_id=task_id,
                success=True,
                result=result_bytes,
                latency_ms=latency_ms
            )

        except Exception as e:
            self.failed_tasks += 1
            self.total_tasks += 1
            latency_ms = (time.time() - start) * 1000
            error_msg = f"Task {task_id} failed: {str(e)}"
            logger.error(error_msg)

            return ExecuteTaskResponse(
                task_id=task_id,
                success=False,
                error=error_msg,
                latency_ms=latency_ms
            )

    def GetStatus(self, request: StatusRequest, context) -> StatusResponse:
        """Get worker status (health check)"""
        avg_latency = sum(self.task_latencies) / len(self.task_latencies) if self.task_latencies else 0.0

        return StatusResponse(
            worker_id=self.worker_id,
            total_tasks=self.total_tasks,
            completed_tasks=self.completed_tasks,
            failed_tasks=self.failed_tasks,
            avg_latency_ms=avg_latency
        )

    def Shutdown(self, request: ShutdownRequest, context) -> ShutdownResponse:
        """Graceful shutdown"""
        logger.info(f"[{self.worker_id}] Shutting down. Tasks: {self.completed_tasks} completed, {self.failed_tasks} failed")
        return ShutdownResponse(success=True)
