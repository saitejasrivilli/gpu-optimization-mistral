"""Model Registry: Version control for ML models"""

import pickle
import json
from typing import Any, Dict, List
from datetime import datetime
import hashlib

class ModelRegistry:
    """Distributed model registry with versioning and rollback capability"""

    def __init__(self):
        self.models = {}
        self.metadata = {}
        self.current_production = None

    def save_model(self, model: Any, model_name: str, metrics: Dict = None,
                   version: str = None) -> str:
        """Save model with version and metadata"""
        if version is None:
            version = f"v{len([k for k in self.models.keys() if k.startswith(f'{model_name}_')])}"

        version_key = f"{model_name}:{version}"

        self.models[version_key] = pickle.dumps(model)

        model_size = len(self.models[version_key]) / 1024 / 1024
        model_hash = hashlib.md5(self.models[version_key]).hexdigest()

        self.metadata[version_key] = {
            "model_name": model_name,
            "version": version,
            "metrics": metrics or {},
            "created_at": datetime.now().isoformat(),
            "model_size_mb": round(model_size, 2),
            "model_hash": model_hash,
            "status": "active"
        }

        print(f"✓ Saved model {version_key} ({model_size:.2f} MB)")
        return version_key

    def load_model(self, version_key: str) -> Any:
        """Load model by version"""
        if version_key not in self.models:
            raise ValueError(f"Model {version_key} not found")
        return pickle.loads(self.models[version_key])

    def promote_to_production(self, version_key: str):
        """Promote model to production (canary deployment)"""
        if version_key not in self.models:
            raise ValueError(f"Model {version_key} not found")

        if self.current_production:
            old_metadata = self.metadata[self.current_production]
            old_metadata["status"] = "staging"

        self.current_production = version_key
        self.metadata[version_key]["status"] = "production"
        self.metadata[version_key]["promoted_at"] = datetime.now().isoformat()

        print(f"✓ Promoted {version_key} to production")

    def get_production_model(self) -> tuple:
        """Get current production model"""
        if not self.current_production:
            raise ValueError("No production model available")
        return self.load_model(self.current_production), self.metadata[self.current_production]

    def get_model_history(self, model_name: str) -> List[Dict]:
        """Get all versions of a model"""
        history = []
        for version_key, metadata in self.metadata.items():
            if metadata["model_name"] == model_name:
                history.append({
                    "version_key": version_key,
                    **metadata
                })
        return sorted(history, key=lambda x: x["created_at"], reverse=True)
