"""ML Pipeline Platform: End-to-end ML infrastructure"""

__version__ = "1.0.0"
__author__ = "ML Engineer"

from .feature_store.store import FeatureStore
from .model_registry.registry import ModelRegistry
from .training.trainer import DistributedTrainer
from .inference.load_balancer import LoadBalancer

__all__ = ["FeatureStore", "ModelRegistry", "DistributedTrainer", "LoadBalancer"]
