"""Model definitions: classic ML, boosting, and simple NN."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset


@dataclass
class ModelSpec:
    """Container for a model and its parameter grid."""

    pipeline: Pipeline
    param_grid: Dict


class FeedForwardNN(nn.Module):
    """Simple feedforward neural network for tabular data."""

    def __init__(self, in_dim: int, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x)


def make_classifiers() -> Dict[str, Tuple[Pipeline, Dict]]:
    """Return dictionary of classifier pipelines and param grids."""
    models: Dict[str, Tuple[Pipeline, Dict]] = {}

    models["knn"] = (
        KNeighborsClassifier(),
        {"n_neighbors": [3, 5, 11, 21], "weights": ["uniform", "distance"]},
    )

    models["logreg"] = (
        CalibratedClassifierCV(
            base_estimator=LogisticRegression(max_iter=2000, solver="saga", class_weight="balanced"),
            method="isotonic",
            cv=3,
        ),
        {"base_estimator__C": [0.01, 0.1, 1.0, 10.0], "base_estimator__penalty": ["l1", "l2"]},
    )

    models["dtree"] = (
        DecisionTreeClassifier(random_state=42, class_weight="balanced"),
        {"max_depth": [3, 5, 10, None], "min_samples_split": [2, 10, 50]},
    )

    models["xgb"] = (
        XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
            tree_method="hist",
            scale_pos_weight=1.0,
        ),
        {
            "max_depth": [3, 5, 7],
            "min_child_weight": [1, 5, 10],
            "gamma": [0.0, 0.1, 0.2],
        },
    )

    models["lgbm"] = (
        LGBMClassifier(
            n_estimators=600,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            class_weight="balanced",
            random_state=42,
        ),
        {"num_leaves": [31, 63, 127], "max_depth": [-1, 5, 10]},
    )

    # Optional Torch NN; to be used after preprocessing (dense numeric array)
    try:
        models["nn"] = (
            TorchTabularClassifier(in_dim=None, hidden=64, dropout=0.2, epochs=30, batch_size=64, focal_loss=True),
            {"epochs": [20, 30, 50], "hidden": [32, 64, 128], "dropout": [0.0, 0.2, 0.4]},
        )
    except Exception:
        pass

    return models


def make_regressors() -> Dict[str, Tuple[object, Dict]]:
    """Return regression models and param grids."""
    return {
        "linreg": (LinearRegression(), {}),
    }


def make_kmeans() -> Tuple[KMeans, Dict]:
    """Return KMeans model and param grid for exploration."""
    return KMeans(n_clusters=3, random_state=42), {"n_clusters": [2, 3, 4, 5, 8]}


class TorchTabularClassifier:
    """PyTorch classifier wrapper with sklearn-like API."""

    def __init__(self, in_dim: int | None = None, hidden: int = 64, dropout: float = 0.2, lr: float = 1e-3, epochs: int = 50, batch_size: int = 64, focal_loss: bool = False):
        self.in_dim = in_dim
        self.model = None
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.focal = focal_loss
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _criterion(self, logits, y):
        bce = nn.BCEWithLogitsLoss()
        if not self.focal:
            return bce(logits, y)
        # simple focal loss
        p = torch.sigmoid(logits)
        pt = p * y + (1 - p) * (1 - y)
        gamma = 2.0
        alpha = 0.25
        return (-(alpha * (1 - pt) ** gamma * torch.log(pt + 1e-8))).mean()

    def fit(self, X, y):
        if self.in_dim is None:
            self.in_dim = X.shape[1]
        if self.model is None:
            self.model = FeedForwardNN(self.in_dim, hidden=self.hidden if hasattr(self, 'hidden') else 64, dropout=self.dropout if hasattr(self, 'dropout') else 0.2)
        # cache hyperparams on self for get_params
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y.values if hasattr(y, "values") else y, dtype=torch.float32).view(-1, 1)
        ds = TensorDataset(X_t, y_t)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        best_loss = float("inf")
        best_state = None

        for _ in range(self.epochs):
            self.model.train()
            running = 0.0
            for xb, yb in dl:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                opt.zero_grad()
                logits = self.model(xb)
                loss = self._criterion(logits, yb)
                loss.backward()
                opt.step()
                running += loss.item() * xb.size(0)
            epoch_loss = running / len(ds)
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_state = self.model.state_dict()
        if best_state:
            self.model.load_state_dict(best_state)
        return self

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            logits = self.model(X_t)
            probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
        return np.vstack([1 - probs, probs]).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    # sklearn compatibility
    def get_params(self, deep: bool = True):
        return {
            "in_dim": self.in_dim,
            "hidden": getattr(self, 'hidden', 64),
            "dropout": getattr(self, 'dropout', 0.2),
            "lr": self.lr,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "focal_loss": self.focal,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        # reset model to rebuild with new dims on fit
        self.model = None
        return self
