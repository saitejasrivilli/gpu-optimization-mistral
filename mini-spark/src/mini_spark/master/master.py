import redis
import json
from enum import Enum
from typing import Dict, List
from datetime import datetime

class TaskStatus(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class Master:
    def __init__(self, redis_host='localhost', redis_port=6379):
        self.redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.job_counter = 0

    def create_job(self, job_name: str) -> str:
        job_id = f"job_{self.job_counter}"
        self.job_counter += 1
        self.redis.hset(f"job:{job_id}", mapping={
            "name": job_name,
            "created_at": datetime.now().isoformat(),
            "status": "RUNNING"
        })
        return job_id

    def register_task(self, job_id: str, task_id: str, stage_id: int):
        self.redis.hset(f"task:{task_id}", mapping={
            "job_id": job_id,
            "stage_id": str(stage_id),
            "status": TaskStatus.PENDING.value,
            "attempts": "0",
            "worker_id": ""
        })

    def update_task_status(self, task_id: str, status: TaskStatus, worker_id: str = ""):
        data = {"status": status.value}
        if worker_id:
            data["worker_id"] = worker_id
        if status == TaskStatus.FAILED:
            attempts = int(self.redis.hget(f"task:{task_id}", "attempts") or 0)
            data["attempts"] = str(attempts + 1)
        self.redis.hset(f"task:{task_id}", mapping=data)

    def get_task_status(self, task_id: str) -> Dict:
        return self.redis.hgetall(f"task:{task_id}")

    def get_job_tasks(self, job_id: str) -> List[Dict]:
        keys = self.redis.keys("task:*")
        tasks = []
        for key in keys:
            task_data = self.redis.hgetall(key)
            if task_data.get("job_id") == job_id:
                tasks.append(task_data)
        return tasks

    def store_partition(self, task_id: str, partition_id: int, data: bytes):
        key = f"partition:{task_id}:{partition_id}"
        self.redis.set(key, data)

    def get_partition(self, task_id: str, partition_id: int) -> bytes:
        key = f"partition:{task_id}:{partition_id}"
        return self.redis.get(key)

    def complete_job(self, job_id: str):
        self.redis.hset(f"job:{job_id}", "status", "COMPLETED")
