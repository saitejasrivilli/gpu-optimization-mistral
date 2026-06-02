"""Application settings"""
import os

# API
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))

# Inference
NUM_INFERENCE_REPLICAS = int(os.getenv("NUM_INFERENCE_REPLICAS", 3))
REPLICA_TIMEOUT = int(os.getenv("REPLICA_TIMEOUT", 30))

# Training
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", 10))

# Model
MODEL_NAME = os.getenv("MODEL_NAME", "fraud_detector")
