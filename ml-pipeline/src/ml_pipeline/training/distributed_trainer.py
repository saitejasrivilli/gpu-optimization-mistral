"""Distributed Training: PyTorch DDP/FSDP with multi-GPU support"""

import torch
import torch.nn as nn
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
import torch.optim as optim
from typing import Dict, Any, Tuple, Optional
import numpy as np
import os
import time
from datetime import datetime


class DistributedModel(nn.Module):
    """Scalable model for distributed training"""
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class DDPTrainer:
    """PyTorch Distributed Data Parallel trainer"""

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.01,
        use_fsdp: bool = False,
        world_size: Optional[int] = None,
        rank: Optional[int] = None
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.use_fsdp = use_fsdp
        self.world_size = world_size or 1
        self.rank = rank or 0
        self.device = torch.device(f"cuda:{self.rank}" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

        # Wrap model for distributed training
        if self.use_fsdp and self.world_size > 1:
            self.model = FSDP(
                model,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                device_id=self.device,
                auto_wrap_policy=None
            )
        elif self.world_size > 1:
            self.model = DDP(model, device_ids=[self.rank])

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()
        self.training_history = []
        self.start_time = time.time()

    def train_batch(self, X_batch: np.ndarray, y_batch: np.ndarray) -> float:
        """Train on single batch"""
        X = torch.FloatTensor(X_batch).to(self.device)
        y = torch.FloatTensor(y_batch).unsqueeze(1).to(self.device)

        predictions = self.model(X)
        loss = self.criterion(predictions, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train_epoch(self, X_train: np.ndarray, y_train: np.ndarray,
                   batch_size: int = 32) -> Dict[str, float]:
        """Train one epoch"""
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
        """Evaluate model"""
        self.model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(X_val).to(self.device)
            y_pred = self.model(X).cpu().numpy().flatten()
            y_pred = (y_pred > 0.5).astype(int)

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        return {
            "accuracy": accuracy_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred, zero_division=0),
            "recall": recall_score(y_val, y_pred, zero_division=0),
            "f1": f1_score(y_val, y_pred, zero_division=0)
        }

    def get_timing_stats(self) -> Dict[str, float]:
        """Get training timing stats for speedup calculation"""
        elapsed = time.time() - self.start_time
        return {
            "elapsed_seconds": elapsed,
            "num_gpus": self.world_size if torch.cuda.is_available() else 1,
        }


def train_distributed_model(
    num_epochs: int = 10,
    num_gpus: int = 1,
    use_fsdp: bool = False
) -> Tuple[nn.Module, Dict]:
    """Train model with distributed training"""
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    # Generate data
    print("Generating training data...")
    X, y = make_classification(
        n_samples=2000, n_features=30, n_informative=20,
        n_redundant=10, random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create trainer
    model = DistributedModel(input_dim=X_train.shape[1])
    trainer = DDPTrainer(
        model,
        learning_rate=0.01,
        use_fsdp=use_fsdp,
        world_size=num_gpus,
        rank=0  # Rank 0 for single process
    )

    print(f"Training for {num_epochs} epochs on {num_gpus} GPU(s)...\n")
    best_f1 = 0
    best_model_state = None

    for epoch in range(num_epochs):
        train_metrics = trainer.train_epoch(X_train, y_train, batch_size=32)
        val_metrics = trainer.evaluate(X_val, y_val)

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_model_state = {
                k: v.cpu() if isinstance(v, torch.Tensor) else v
                for k, v in trainer.model.state_dict().items()
            }

        trainer.training_history.append({
            "epoch": epoch + 1,
            **train_metrics,
            **{f"val_{k}": v for k, v in val_metrics.items()}
        })

        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch+1:2d} | Loss: {train_metrics['epoch_loss']:.4f} | Val F1: {val_metrics['f1']:.4f}")

    if best_model_state:
        trainer.model.load_state_dict(best_model_state)

    timing = trainer.get_timing_stats()
    final_metrics = trainer.evaluate(X_val, y_val)

    return trainer.model, {
        "final_metrics": final_metrics,
        "best_f1": best_f1,
        "num_parameters": sum(p.numel() for p in trainer.model.parameters()),
        "timing": timing,
    }
