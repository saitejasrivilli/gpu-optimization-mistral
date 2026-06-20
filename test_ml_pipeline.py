#!/usr/bin/env python3
"""Test ml-pipeline end-to-end"""

import sys
sys.path.insert(0, '/Users/saitejasrivillibhutturu/Downloads/ml-pipeline/src')

from ml_pipeline.training.trainer import DistributedTrainer, SimpleNN
from ml_pipeline.model_registry.registry import ModelRegistry
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import torch

print("=" * 70)
print("ML-PIPELINE END-TO-END TEST")
print("=" * 70)

# Step 1: Data
print("\n[1/6] GENERATE DATA")
X, y = make_classification(n_samples=500, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

# Step 2: Train
print("\n[2/6] TRAIN MODEL (3 epochs)")
model = SimpleNN(20)
trainer = DistributedTrainer(model)
for e in range(3):
    m = trainer.train_epoch(X_train, y_train, epoch=e)
    v = trainer.evaluate(X_test, y_test, epoch=e)
    print(f"  Epoch {e+1}: loss={m['epoch_loss']:.4f}, acc={v['accuracy']:.4f}, f1={v['f1']:.4f}")

# Step 3: Save
print("\n[3/6] SAVE TO REGISTRY")
registry = ModelRegistry(use_gcs=False)
version_key = registry.save_model(model, "prod-classifier", metrics=v, version="1.0.0")
print(f"  Saved: {version_key}")

# Step 4: Load
print("\n[4/6] LOAD FROM REGISTRY")
loaded = registry.load_model(version_key)
print(f"  Loaded: {type(loaded).__name__}")

# Step 5: Predict
print("\n[5/6] MAKE PREDICTIONS")
X_pred = torch.FloatTensor(X_test[:3])
with torch.no_grad():
    preds = loaded(X_pred).numpy()
for i, p in enumerate(preds):
    print(f"  Sample {i+1}: {p[0]:.4f}")

# Step 6: Promote
print("\n[6/6] PROMOTE TO PRODUCTION")
registry.promote_to_production(version_key)
print(f"  Promoted: {version_key}")

print("\n" + "=" * 70)
print("✓ ML-PIPELINE TEST COMPLETE")
print("=" * 70)
