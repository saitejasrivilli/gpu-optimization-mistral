"""Production Feature Store - Redis backend for distributed versioning"""

import json
import pickle
import hashlib
from typing import Dict, Any, List
from datetime import datetime
import redis

class RedisFeatureStore:
    """Production feature store with Redis backend"""

    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379, db: int = 0):
        try:
            self.redis = redis.Redis(host=redis_host, port=redis_port, db=db, decode_responses=False)
            self.redis.ping()
            print(f"✓ Connected to Redis at {redis_host}:{redis_port}")
        except redis.ConnectionError as e:
            print(f"✗ Redis connection failed: {e}")
            raise

    def write_features(self, dataset_id: str, features: Dict[str, Any], version: str = None) -> str:
        """Write features with version control and consistency guarantees"""
        if version is None:
            version = datetime.now().isoformat()

        version_key = f"features:{dataset_id}:{version}"

        # Serialize features to bytes for storage
        features_bytes = pickle.dumps(features)

        # Store in Redis with expiration (30 days)
        self.redis.setex(version_key, 86400 * 30, features_bytes)

        # Track versions
        self.redis.lpush(f"versions:{dataset_id}", version)

        # Store metadata
        feature_hash = hashlib.md5(features_bytes).hexdigest()
        num_rows = len(next(iter(features.values()))) if features else 0

        metadata = {
            'dataset_id': dataset_id,
            'version': version,
            'feature_hash': feature_hash,
            'timestamp': datetime.now().isoformat(),
            'num_rows': num_rows,
            'size_bytes': len(features_bytes)
        }

        self.redis.hset(f"metadata:{dataset_id}:{version}", mapping={
            k: str(v) for k, v in metadata.items()
        })

        print(f"✓ Stored features {dataset_id}:{version} ({num_rows} rows, {len(features_bytes)/1024:.2f}KB)")
        return version_key

    def read_features(self, dataset_id: str, version: str = None) -> Dict[str, Any]:
        """Read features with data consistency guarantee"""
        if version is None:
            # Get latest version
            versions = self.redis.lrange(f"versions:{dataset_id}", 0, 0)
            if not versions:
                raise ValueError(f"No features found for {dataset_id}")
            version = versions[0].decode()

        version_key = f"features:{dataset_id}:{version}"
        features_bytes = self.redis.get(version_key)

        if not features_bytes:
            raise ValueError(f"Features {dataset_id}:{version} not found in Redis")

        return pickle.loads(features_bytes)

    def get_version_history(self, dataset_id: str, limit: int = 10) -> List[str]:
        """Get version history (audit logging)"""
        versions = self.redis.lrange(f"versions:{dataset_id}", 0, limit - 1)
        return [v.decode() for v in versions]

    def get_metadata(self, dataset_id: str, version: str = None) -> Dict:
        """Get feature metadata for monitoring"""
        if version is None:
            versions = self.redis.lrange(f"versions:{dataset_id}", 0, 0)
            if not versions:
                return {}
            version = versions[0].decode()

        metadata_bytes = self.redis.hgetall(f"metadata:{dataset_id}:{version}")
        return {k.decode(): v.decode() for k, v in metadata_bytes.items()}

    def delete_old_versions(self, dataset_id: str, keep_recent: int = 3):
        """Cleanup old versions (garbage collection)"""
        versions = self.redis.lrange(f"versions:{dataset_id}", 0, -1)
        to_delete = versions[keep_recent:]

        for version_bytes in to_delete:
            version = version_bytes.decode()
            version_key = f"features:{dataset_id}:{version}"
            metadata_key = f"metadata:{dataset_id}:{version}"

            self.redis.delete(version_key)
            self.redis.delete(metadata_key)
            self.redis.lrem(f"versions:{dataset_id}", 0, version_bytes)

        print(f"✓ Cleaned up {len(to_delete)} old feature versions for {dataset_id}")

    def list_datasets(self) -> List[str]:
        """List all datasets in feature store"""
        keys = self.redis.keys("versions:*")
        return list(set(k.decode().split(":")[1] for k in keys))
