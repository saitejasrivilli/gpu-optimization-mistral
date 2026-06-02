"""Application settings"""
import os

# Redis
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))

# Worker
NUM_WORKERS = int(os.getenv("NUM_WORKERS", 3))
WORKER_TIMEOUT = int(os.getenv("WORKER_TIMEOUT", 300))

# Job
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
