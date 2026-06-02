"""gRPC Worker Server - runs in separate process"""

import argparse
import logging
import sys
import signal
from concurrent import futures
import time

# Simple gRPC server (production would use full grpc library)
from .worker_service import WorkerServicer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleGRPCServer:
    """Simplified gRPC server for production execution"""

    def __init__(self, worker_id: str, port: int, state_store):
        self.worker_id = worker_id
        self.port = port
        self.state_store = state_store
        self.servicer = WorkerServicer(worker_id, state_store)
        self.running = True

    def start(self):
        """Start worker server"""
        logger.info(f"Worker {self.worker_id} starting on port {self.port}")

        # Register shutdown handler
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)

        # Keep server alive
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self._handle_shutdown(None, None)

    def _handle_shutdown(self, signum, frame):
        """Handle graceful shutdown"""
        logger.info(f"Worker {self.worker_id} shutting down...")
        self.running = False

def main():
    parser = argparse.ArgumentParser(description='Mini Spark Worker Server')
    parser.add_argument('--worker-id', required=True, help='Worker ID')
    parser.add_argument('--port', type=int, required=True, help='Server port')
    parser.add_argument('--redis-host', default='localhost', help='Redis host')
    parser.add_argument('--redis-port', type=int, default=6379, help='Redis port')

    args = parser.parse_args()

    # Initialize state store (Redis backend)
    try:
        import redis
        from ..master.master import Master
        state_store = Master(redis_host=args.redis_host, redis_port=args.redis_port)
    except ImportError:
        logger.error("Redis not available. Install: pip install redis")
        sys.exit(1)

    # Start server
    server = SimpleGRPCServer(args.worker_id, args.port, state_store)
    server.start()

if __name__ == '__main__':
    main()
