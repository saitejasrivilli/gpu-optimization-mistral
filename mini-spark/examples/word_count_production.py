#!/usr/bin/env python3
"""Mini Spark Production Example - Actual distributed workers with gRPC"""

import sys
sys.path.insert(0, '../src')

from mini_spark.driver.distributed_driver import DistributedDriver
from mini_spark.master.master import Master
from typing import List
import pickle
import urllib.request

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
    print("=" * 70)
    print("Mini Spark - Production Version (Actual Distributed Workers)")
    print("=" * 70)

    try:
        # Initialize master (Redis backend)
        print("\n1. MASTER INITIALIZATION")
        print("-" * 70)
        master = Master(redis_host='localhost', redis_port=6379)

        # Create distributed driver
        print("\n2. SPAWNING WORKER PROCESSES")
        print("-" * 70)
        driver = DistributedDriver(master, num_workers=3, base_port=50051)
        driver.spawn_workers()

        # Create job
        print("\n3. JOB CREATION")
        print("-" * 70)
        job_id = driver.create_job("word_count_production")
        print(f"Created job: {job_id}")

        # Build DAG
        print("\n4. BUILDING COMPUTATION DAG")
        print("-" * 70)

        # Load real text data (Shakespeare)
        print("Loading Shakespeare text...")
        try:
            url = "https://www.gutenberg.org/cache/epub/100/pg100.txt"
            with urllib.request.urlopen(url, timeout=10) as response:
                text = response.read().decode('utf-8', errors='ignore')
                # Split into lines and take first 100 for demo
                data = [line.strip() for line in text.split('\n') if line.strip()][:100]
        except:
            # Fallback to synthetic data
            print("Could not load Shakespeare, using synthetic data")
            data = ["hello world spark distributed computing"] * 10

        print(f"  Loaded {len(data)} text lines for processing")
        print(f"  Stage 0 (Map): {len(data)} partitions")
        map_tasks = driver.add_stage(map_words, partitions=len(data))

        print(f"  Stage 1 (Reduce): Aggregate across workers")
        reduce_tasks = driver.add_stage(reduce_words, partitions=1, dependencies=map_tasks)

        # Execute distributed computation
        print("\n5. DISTRIBUTED EXECUTION (Actual RPC to Workers)")
        print("-" * 70)
        print("Executing tasks across actual worker processes...\n")

        try:
            results = driver.execute()

            # Fetch final result
            print("\n6. RESULTS")
            print("-" * 70)
            final_result = master.get_partition(reduce_tasks[0], 0)
            if final_result:
                word_counts = pickle.loads(final_result)
                print("Word counts (distributed computation result):")
                for word, count in sorted(word_counts.items(), key=lambda x: -x[1]):
                    print(f"  {word}: {count}")

            print("\n✓ Production distributed computation complete!")

        finally:
            driver.shutdown()

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nSetup required:")
        print("  Redis: redis-server (or docker run -d -p 6379:6379 redis:7-alpine)")
        sys.exit(1)

if __name__ == "__main__":
    main()
