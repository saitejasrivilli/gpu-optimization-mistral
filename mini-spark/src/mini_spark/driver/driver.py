from ..core.task import TaskGraph, Task
from ..worker.worker import Worker
from ..master.master import Master, TaskStatus
from typing import Callable, List, Any
import pickle
import time

class Driver:
    def __init__(self, master: Master):
        self.master = master
        self.graph = TaskGraph()
        self.workers = {}
        self.job_id = None

    def create_job(self, job_name: str) -> str:
        self.job_id = self.master.create_job(job_name)
        return self.job_id

    def add_worker(self, worker_id: str):
        """Add worker to driver"""
        worker = Worker(worker_id, self.master)
        self.workers[worker_id] = worker

    def add_stage(self, func: Callable, partitions: int = 1, dependencies: List[str] = None) -> List[str]:
        """Add computation stage"""
        task_ids = self.graph.add_stage(func, partitions, dependencies)
        for task_id in task_ids:
            self.master.register_task(self.job_id, task_id, self.graph.stage_count - 1)
        return task_ids

    def execute(self) -> Any:
        """Execute DAG with load balancing"""
        tasks = self.graph.get_all_tasks()
        tasks_by_stage = {}

        for task in tasks:
            if task.stage_id not in tasks_by_stage:
                tasks_by_stage[task.stage_id] = []
            tasks_by_stage[task.stage_id].append(task)

        results = {}
        worker_list = list(self.workers.values())
        worker_idx = 0

        for stage_id in sorted(tasks_by_stage.keys()):
            stage_tasks = tasks_by_stage[stage_id]
            print(f"Executing stage {stage_id} ({len(stage_tasks)} tasks)")

            for task in stage_tasks:
                worker = worker_list[worker_idx % len(worker_list)]
                worker_idx += 1

                try:
                    result = worker.execute_task(
                        task.task_id,
                        task.serialize_func(),
                        task.args,
                        task.dependencies
                    )
                    results[task.task_id] = result
                    print(f"  Task {task.task_id} completed on {worker.worker_id}")

                except Exception as e:
                    print(f"  Task {task.task_id} failed: {e}")
                    raise

        self.master.complete_job(self.job_id)
        return results

    def get_result(self, task_id: str) -> Any:
        """Fetch final result"""
        worker = list(self.workers.values())[0]
        return worker.get_task_result(task_id)
