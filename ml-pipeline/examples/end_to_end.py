#!/usr/bin/env python3
"""End-to-End ML Pipeline Example"""

import sys
sys.path.insert(0, '../src')

from ml_pipeline.feature_store.store import FeatureStore
from ml_pipeline.model_registry.registry import ModelRegistry
from ml_pipeline.training.trainer import train_model
from ml_pipeline.inference.load_balancer import LoadBalancer
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def main():
    print("=" * 70)
    print("ML Pipeline Platform - End-to-End Example")
    print("=" * 70)

    # Initialize components
    feature_store = FeatureStore()
    model_registry = ModelRegistry()

    print("\n1. FEATURE STORE")
    print("-" * 70)
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    feature_store.write_features("training_data", {
        "X_train": X_train.tolist(),
        "y_train": y_train.tolist()
    })

    print("\n2. DISTRIBUTED TRAINING")
    print("-" * 70)
    model, metrics = train_model(num_epochs=10)
    print(f"\nTraining metrics: {metrics['final_metrics']}")

    print("\n3. MODEL REGISTRY")
    print("-" * 70)
    v1_key = model_registry.save_model(
        model, "fraud_detector",
        metrics=metrics['final_metrics'],
        version="v1.0"
    )
    model_registry.promote_to_production(v1_key)

    print("\n4. DISTRIBUTED INFERENCE")
    print("-" * 70)
    lb = LoadBalancer(num_replicas=2, model=model)

    print("Simulating inference requests...")
    for i in range(3):
        sample = X_val[i:i+1]
        result = lb.predict(sample)
        print(f"  Request {i+1}: replica {result['replica_id']}, latency {result['latency_ms']:.2f}ms")

    print(f"\nCluster status: {lb.get_cluster_health()}")
    print("\n✓ Pipeline complete!")

if __name__ == "__main__":
    main()
