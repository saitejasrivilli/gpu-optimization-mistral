"""Production Driver - spawns actual worker processes with gRPC communication"""

import logging
import subprocess
import time
import os
import signal
from typing import List, Dict, Any
from multiprocessing import Process

from ..core.task import TaskGraph, Task
from ..master.master import Master, TaskStatus
from ..grpc.worker_client import WorkerClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DistributedDriver:
    """Production driver that spawns real worker processes"""

    def __init__(self, master: Master, num_workers: int = 3, base_port: int = 50051):
        self.master = master
        self.num_workers = num_workers
        self.base_port = base_port
        self.graph = TaskGraph()
        self.workers: Dict[str, subprocess.Popen] = {}
        self.worker_clients: Dict[str, WorkerClient] = {}
        self.job_id = None
        self.worker_processes: List[Process] = []

    def create_job(self, job_name: str) -> str:
        """Create new distributed job"""
        self.job_id = self.master.create_job(job_name)
        logger.info(f"Created job: {self.job_id}")
        return self.job_id

    def spawn_workers(self):
        """Spawn actual worker processes"""
        logger.info(f"Spawning {self.num_workers} worker processes...")

        for i in range(self.num_workers):
            worker_id = f"worker_{i}"
            port = self.base_port + i

            # Spawn worker as separate process
            # In production environment: actual gRPC worker
            proc = subprocess.Popen(
                [
                    'python3', '-m', 'mini_spark.grpc.worker_server',
                    '--worker-id', worker_id,
                    '--port', str(port),
                    '--redis-host', 'localhost',
                    '--redis-port', '6379'
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid  # Create process group for clean shutdown
            )

            self.workers[worker_id] = proc
            self.worker_clients[worker_id] = WorkerClient(worker_id, 'localhost', port)
            logger.info(f"Spawned {worker_id} on port {port} (PID: {proc.pid})")

        # Wait for workers to start
        time.sleep(2)
        self._verify_workers()

    def _verify_workers(self):
        """Verify all workers are healthy"""
        healthy = 0
        for worker_id, client in self.worker_clients.items():
            status = client.get_status()
            if status.get('healthy'):
                healthy += 1
                logger.info(f"{worker_id}: healthy")
            else:
                logger.warning(f"{worker_id}: unhealthy")

        logger.info(f"Worker health: {healthy}/{self.num_workers} healthy")

    def add_stage(self, func, partitions: int = 1, dependencies: List[str] = None) -> List[str]:
        """Add computation stage to DAG"""
        task_ids = self.graph.add_stage(func, partitions, dependencies)
        for task_id in task_ids:
            self.master.register_task(self.job_id, task_id, self.graph.stage_count - 1)
        return task_ids

    def execute(self) -> Dict[str, Any]:
        """Execute distributed DAG across actual worker processes"""
        tasks = self.graph.get_all_tasks()
        tasks_by_stage = {}

        for task in tasks:
            if task.stage_id not in tasks_by_stage:
                tasks_by_stage[task.stage_id] = []
            tasks_by_stage[task.stage_id].append(task)

        results = {}
        worker_clients_list = list(self.worker_clients.values())
        worker_idx = 0

        logger.info(f"Executing {len(tasks)} tasks across {len(worker_clients_list)} workers")

        for stage_id in sorted(tasks_by_stage.keys()):
            stage_tasks = tasks_by_stage[stage_id]
            logger.info(f"\n=== Stage {stage_id} ({len(stage_tasks)} tasks) ===")

            for task in stage_tasks:
                # Load balance: round-robin across workers
                client = worker_clients_list[worker_idx % len(worker_clients_list)]
                worker_idx += 1

                try:
                    # Send to actual remote worker via gRPC
                    response = client.execute_task(
                        task.task_id,
                        task.stage_id,
                        task.serialize_func(),
                        [pickle.dumps(arg) for arg in task.args] if task.args else [],
                        task.dependencies
                    )

                    if response['success']:
                        self.master.update_task_status(task.task_id, TaskStatus.COMPLETED, client.worker_id)
                        logger.info(f"Task {task.task_id} completed on {client.worker_id} ({response['latency_ms']:.2f}ms)")
                        results[task.task_id] = response

                    else:
                        raise Exception(response.get('error', 'Unknown error'))

                except Exception as e:
                    self.master.update_task_status(task.task_id, TaskStatus.FAILED, client.worker_id)
                    logger.error(f"Task {task.task_id} failed: {e}")
                    raise

        self.master.complete_job(self.job_id)
        logger.info("\n✓ All tasks completed successfully")
        return results

    def shutdown(self):
        """Gracefully shutdown all workers"""
        if not self.workers:
            return

        logger.info("Shutting down workers...")

        for worker_id, proc in list(self.workers.items()):
            try:
                if proc.poll() is None:  # Still running
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                        proc.wait(timeout=5)
                    except (ProcessLookupError, OSError):
                        pass  # Process already dead
                logger.info(f"Shutdown {worker_id}")
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass

        self.workers.clear()

    def __del__(self):
        """Cleanup on destruction"""
        self.shutdown()
