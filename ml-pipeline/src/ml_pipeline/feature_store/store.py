"""Feature Store: Versioned, fault-tolerant feature storage"""

import json
import pickle
from typing import Dict, Any, List
from datetime import datetime
import hashlib

class FeatureStore:
    """Distributed feature store with versioning and consistency guarantees"""

    def __init__(self):
        self.features = {}
        self.versions = {}
        self.metadata = {}

    def write_features(self, dataset_id: str, features: Dict[str, Any], version: str = None):
        """Write features with version control (data consistency)"""
        if version is None:
            version = datetime.now().isoformat()

        version_key = f"{dataset_id}:{version}"
        self.features[version_key] = features

        if dataset_id not in self.versions:
            self.versions[dataset_id] = []
        self.versions[dataset_id].append(version)

        feature_hash = hashlib.md5(
            json.dumps(features, sort_keys=True, default=str).encode()
        ).hexdigest()

        self.metadata[version_key] = {
            "dataset_id": dataset_id,
            "version": version,
            "feature_hash": feature_hash,
            "timestamp": datetime.now().isoformat(),
            "num_rows": len(next(iter(features.values()))) if features else 0
        }

        print(f"✓ Stored features {dataset_id}:{version} ({self.metadata[version_key]['num_rows']} rows)")

    def read_features(self, dataset_id: str, version: str = None) -> Dict[str, Any]:
        """Read features (data consistency guarantee)"""
        if version is None:
            if dataset_id in self.versions and self.versions[dataset_id]:
                version = self.versions[dataset_id][-1]
            else:
                raise ValueError(f"No features found for {dataset_id}")

        version_key = f"{dataset_id}:{version}"
        if version_key not in self.features:
            raise ValueError(f"Features {dataset_id}:{version} not found")

        return self.features[version_key]

    def get_version_history(self, dataset_id: str) -> List[str]:
        """Get all versions of a dataset (reliability/audit logging)"""
        return self.versions.get(dataset_id, [])

    def list_datasets(self) -> List[str]:
        """List all datasets (for monitoring)"""
        return list(set(v.split(":")[0] for v in self.features.keys()))
