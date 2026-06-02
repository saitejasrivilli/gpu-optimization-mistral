import pickle
from typing import Callable, Any, List
from ..master.master import Master, TaskStatus

class Worker:
    def __init__(self, worker_id: str, master: Master):
        self.worker_id = worker_id
        self.master = master
        self.task_results = {}

    def execute_task(self, task_id: str, func: bytes, args: tuple = (), dependencies: List[str] = None) -> Any:
        """Execute task with fault tolerance"""
        try:
            self.master.update_task_status(task_id, TaskStatus.RUNNING, self.worker_id)

            # Deserialize function
            func_obj = pickle.loads(func)

            # Fetch dependency results if any
            dep_data = []
            if dependencies:
                for dep_id in dependencies:
                    data = self.master.get_partition(dep_id, 0)
                    if data:
                        dep_data.append(pickle.loads(data))

            # Execute with dependencies
            if dep_data:
                result = func_obj(*dep_data, *args)
            else:
                result = func_obj(*args)

            # Store result
            serialized = pickle.dumps(result)
            self.master.store_partition(task_id, 0, serialized)
            self.master.update_task_status(task_id, TaskStatus.COMPLETED, self.worker_id)

            return result

        except Exception as e:
            self.master.update_task_status(task_id, TaskStatus.FAILED, self.worker_id)
            print(f"Task {task_id} failed: {e}")
            raise

    def get_task_result(self, task_id: str) -> Any:
        """Retrieve task result from distributed storage"""
        data = self.master.get_partition(task_id, 0)
        if data:
            return pickle.loads(data)
        return None
