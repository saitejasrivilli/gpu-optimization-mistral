from dataclasses import dataclass
from typing import Callable, List, Any
import pickle
import uuid

@dataclass
class Task:
    task_id: str
    stage_id: int
    partition_id: int
    func: Callable
    args: tuple
    dependencies: List[str]

    def __init__(self, stage_id: int, partition_id: int, func: Callable, args: tuple = (), dependencies: List[str] = None):
        self.task_id = str(uuid.uuid4())[:8]
        self.stage_id = stage_id
        self.partition_id = partition_id
        self.func = func
        self.args = args
        self.dependencies = dependencies or []

    def serialize_func(self):
        return pickle.dumps(self.func)

    @staticmethod
    def deserialize_func(serialized):
        return pickle.loads(serialized)

class TaskGraph:
    def __init__(self):
        self.tasks = {}
        self.stage_count = 0

    def add_stage(self, func: Callable, partitions: int, dependencies: List[str] = None):
        stage_id = self.stage_count
        tasks = []
        for p in range(partitions):
            task = Task(stage_id, p, func, dependencies=dependencies or [])
            self.tasks[task.task_id] = task
            tasks.append(task)
        self.stage_count += 1
        return [t.task_id for t in tasks]

    def get_task(self, task_id: str) -> Task:
        return self.tasks.get(task_id)

    def get_all_tasks(self) -> List[Task]:
        return list(self.tasks.values())
