"""Distributed Training: ML model training using distributed patterns"""

import torch
import torch.nn as nn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, Any, Tuple
import numpy as np

class SimpleNN(nn.Module):
    """Simple neural network for classification"""
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

class DistributedTrainer:
    """Trainer demonstrating distributed ML training patterns"""

    def __init__(self, model: nn.Module, learning_rate: float = 0.01):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()
        self.training_history = []

    def train_batch(self, X_batch: np.ndarray, y_batch: np.ndarray) -> float:
        """Train on single batch (simulates worker task)"""
        X = torch.FloatTensor(X_batch)
        y = torch.FloatTensor(y_batch).unsqueeze(1)

        predictions = self.model(X)
        loss = self.criterion(predictions, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train_epoch(self, X_train: np.ndarray, y_train: np.ndarray,
                   batch_size: int = 32) -> Dict[str, float]:
        """Train one epoch (distributed across batches)"""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            batch_loss = self.train_batch(X_batch, y_batch)
            epoch_loss += batch_loss
            num_batches += 1

        return {"epoch_loss": epoch_loss / num_batches}

    def evaluate(self, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        """Evaluate model (compute distributed metrics)"""
        self.model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(X_val)
            y_pred = self.model(X).numpy().flatten()
            y_pred = (y_pred > 0.5).astype(int)

        return {
            "accuracy": accuracy_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred, zero_division=0),
            "recall": recall_score(y_val, y_pred, zero_division=0),
            "f1": f1_score(y_val, y_pred, zero_division=0)
        }

def train_model(num_epochs: int = 10) -> Tuple[nn.Module, Dict]:
    """Train ML model"""
    print("Generating training data...")
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15,
        n_redundant=5, random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = SimpleNN(input_dim=X_train.shape[1])
    trainer = DistributedTrainer(model, learning_rate=0.01)

    print(f"Training for {num_epochs} epochs...\n")
    best_f1 = 0
    best_model_state = None

    for epoch in range(num_epochs):
        train_metrics = trainer.train_epoch(X_train, y_train, batch_size=32)
        val_metrics = trainer.evaluate(X_val, y_val)

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_model_state = model.state_dict().copy()

        trainer.training_history.append({
            "epoch": epoch + 1,
            **train_metrics,
            **{f"val_{k}": v for k, v in val_metrics.items()}
        })

        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch+1:2d} | Loss: {train_metrics['epoch_loss']:.4f} | Val F1: {val_metrics['f1']:.4f}")

    if best_model_state:
        model.load_state_dict(best_model_state)

    final_metrics = trainer.evaluate(X_val, y_val)
    return model, {
        "final_metrics": final_metrics,
        "best_f1": best_f1,
        "num_parameters": sum(p.numel() for p in model.parameters()),
    }
