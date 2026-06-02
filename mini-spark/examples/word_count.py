#!/usr/bin/env python3
"""Word Count Example using Mini Spark"""

import sys
sys.path.insert(0, '../src')

from mini_spark.driver.driver import Driver
from mini_spark.master.master import Master
from typing import List
import time

def map_words(line: str) -> List[tuple]:
    """Map: split line into (word, 1) pairs"""
    words = line.split()
    return [(w, 1) for w in words]

def reduce_words(pairs: List[tuple]) -> dict:
    """Reduce: aggregate word counts"""
    counts = {}
    for word, count in pairs:
        counts[word] = counts.get(word, 0) + count
    return counts

def main():
    master = Master(redis_host='localhost', redis_port=6379)
    driver = Driver(master)
    job_id = driver.create_job("word_count_job")
    print(f"Created job: {job_id}\n")

    num_workers = 3
    for i in range(num_workers):
        driver.add_worker(f"worker_{i}")
    print(f"Registered {num_workers} workers\n")

    data = ["hello world", "world of big data", "spark is great"]

    print("Building computation DAG:")
    print(f"  Stage 0 (Map): Process {len(data)} partitions")
    map_tasks = driver.add_stage(map_words, partitions=len(data))

    print(f"  Stage 1 (Reduce): Aggregate counts")
    reduce_tasks = driver.add_stage(reduce_words, partitions=1, dependencies=map_tasks)

    print(f"\nExecuting distributed computation...")
    start = time.time()

    try:
        results = driver.execute()
        elapsed = time.time() - start

        final_result = driver.get_result(reduce_tasks[0])

        print(f"\n✓ Job completed in {elapsed:.2f}s")
        print(f"\nWord counts:")
        for word, count in sorted(final_result.items(), key=lambda x: -x[1]):
            print(f"  {word}: {count}")

    except Exception as e:
        print(f"✗ Job failed: {e}")

if __name__ == "__main__":
    main()
