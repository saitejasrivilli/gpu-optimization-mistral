"""Unit tests for Task module"""
import sys
sys.path.insert(0, '../src')

from mini_spark.core.task import Task, TaskGraph

def test_task_creation():
    def dummy_func(x):
        return x * 2

    task = Task(stage_id=0, partition_id=0, func=dummy_func, args=(5,))
    assert task.stage_id == 0
    assert task.partition_id == 0
    assert len(task.task_id) > 0

def test_task_serialization():
    def dummy_func(x):
        return x * 2

    task = Task(stage_id=0, partition_id=0, func=dummy_func)
    serialized = task.serialize_func()
    deserialized = Task.deserialize_func(serialized)
    assert deserialized(5) == 10

def test_task_graph():
    def stage1_func(x):
        return x * 2

    def stage2_func(x):
        return x + 5

    graph = TaskGraph()
    s1_tasks = graph.add_stage(stage1_func, partitions=3)
    s2_tasks = graph.add_stage(stage2_func, partitions=1, dependencies=s1_tasks)

    assert len(s1_tasks) == 3
    assert len(s2_tasks) == 1
    assert graph.stage_count == 2

if __name__ == "__main__":
    test_task_creation()
    test_task_serialization()
    test_task_graph()
    print("✓ All tests passed")
