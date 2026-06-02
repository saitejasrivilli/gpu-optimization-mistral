#!/usr/bin/env python3
"""Production ML Pipeline Example - Redis + PostgreSQL"""

import sys
sys.path.insert(0, '../src')

from ml_pipeline.feature_store.redis_store import RedisFeatureStore
from ml_pipeline.model_registry.postgres_registry import PostgresModelRegistry
from ml_pipeline.training.trainer import train_model
from ml_pipeline.inference.load_balancer import LoadBalancer
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    print("=" * 70)
    print("ML Pipeline - Production Version (Redis + PostgreSQL)")
    print("=" * 70)

    try:
        # 1. Feature Store (Redis backend)
        print("\n1. FEATURE STORE (Redis)")
        print("-" * 70)
        feature_store = RedisFeatureStore(redis_host='localhost', redis_port=6379)

        # Load synthetic fraud dataset (Kaggle URL unreliable)
        from sklearn.datasets import make_classification
        print("Generating synthetic fraud dataset...")
        X, y = make_classification(n_samples=5000, n_features=30, n_informative=20,
                                   weights=[0.997, 0.003], random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        print(f"Dataset: {X.shape[0]} transactions, {X.shape[1]} features")
        print(f"Fraud rate: {y.sum() / len(y) * 100:.2f}%")

        feature_store.write_features("training_data", {
            "X_train": X_train.tolist(),
            "y_train": y_train.tolist()
        })

        datasets = feature_store.list_datasets()
        print(f"Available datasets: {datasets}")

        # 2. Training (distributed)
        print("\n2. DISTRIBUTED TRAINING")
        print("-" * 70)
        model, metrics = train_model(num_epochs=10)
        print(f"Training complete. F1 score: {metrics['final_metrics']['f1']:.4f}")

        # 3. Model Registry (PostgreSQL backend)
        print("\n3. MODEL REGISTRY (PostgreSQL)")
        print("-" * 70)
        model_registry = PostgresModelRegistry(
            host='localhost',
            port=5432,
            database='ml_pipeline'
        )

        v1_key = model_registry.save_model(
            model, "fraud_detector",
            metrics=metrics['final_metrics'],
            version="v1.0"
        )
        model_registry.promote_to_production(v1_key)

        # Simulate improved model
        print("\nSimulating improved model...")
        improved_model, improved_metrics = train_model(num_epochs=15)

        v2_key = model_registry.save_model(
            improved_model, "fraud_detector",
            metrics=improved_metrics['final_metrics'],
            version="v2.0"
        )

        # Get history
        history = model_registry.get_model_history("fraud_detector")
        print(f"\nModel versions in PostgreSQL:")
        for h in history:
            print(f"  - {h['version']} ({h['status']})")

        # 4. Distributed Inference
        print("\n4. DISTRIBUTED INFERENCE (Load Balanced)")
        print("-" * 70)
        lb = LoadBalancer(num_replicas=3, model=model)

        print("Simulating inference requests...")
        for i in range(5):
            sample = X_val[i:i+1]
            result = lb.predict(sample)
            latency = result.get('latency_ms', 0.03)
            print(f"  Request {i+1}: replica {result['replica_id']}, latency {latency:.2f}ms")

        # Cluster status
        cluster_health = lb.get_cluster_health()
        print(f"\nCluster status:")
        print(f"  Total requests: {cluster_health['total_requests']}")
        print(f"  Avg latency: {cluster_health['avg_latency_ms']:.2f}ms")
        print(f"  Replicas: {cluster_health['num_replicas']}")

        model_registry.close()
        print("\n✓ Production pipeline complete!")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nSetup required:")
        print("  PostgreSQL: brew install postgresql && brew services start postgresql")
        print("  Create DB: createdb ml_pipeline")
        print("  Redis: docker run -d -p 6379:6379 redis:7-alpine")
        sys.exit(1)

if __name__ == "__main__":
    main()
