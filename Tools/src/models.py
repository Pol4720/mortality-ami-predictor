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
import inspect
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier  # type: ignore
except Exception:  # optional
    XGBClassifier = None  # type: ignore
try:
    from lightgbm import LGBMClassifier  # type: ignore
except Exception:
    LGBMClassifier = None  # type: ignore



@dataclass
class ModelSpec:
    """Container for a model and its parameter grid."""

    pipeline: Pipeline
    param_grid: Dict


# Note: PyTorch imports are intentionally avoided at module import-time
# to prevent Streamlit's file watcher from scanning torch.classes. The
# TorchTabularClassifier performs lazy imports inside its methods.


def make_classifiers() -> Dict[str, Tuple[Pipeline, Dict]]:
    """Return dictionary of classifier pipelines and param grids."""
    models: Dict[str, Tuple[Pipeline, Dict]] = {}

    models["knn"] = (
        KNeighborsClassifier(),
        {"n_neighbors": [3, 7], "weights": ["uniform", "distance"]},
    )

    # Calibrated Logistic Regression with faster settings
    logreg = LogisticRegression(max_iter=500, tol=1e-3, solver="lbfgs", class_weight="balanced")
    sig = inspect.signature(CalibratedClassifierCV)
    if "estimator" in sig.parameters:
        calib = CalibratedClassifierCV(estimator=logreg, method="sigmoid", cv=2)
        grid = {"estimator__C": [0.1, 1.0]}
    else:
        calib = CalibratedClassifierCV(base_estimator=logreg, method="sigmoid", cv=2)
        grid = {"base_estimator__C": [0.1, 1.0]}
    models["logreg"] = (calib, grid)

    models["dtree"] = (
        DecisionTreeClassifier(random_state=42, class_weight="balanced"),
        {"max_depth": [3, 5, None], "min_samples_split": [2, 20]},
    )

    if XGBClassifier is not None:
        models["xgb"] = (
            XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                random_state=42,
                tree_method="hist",
                scale_pos_weight=1.0,
                n_jobs=-1,
            ),
            {
                "max_depth": [3, 5],
                "min_child_weight": [1, 5],
            },
        )

    if LGBMClassifier is not None:
        models["lgbm"] = (
            LGBMClassifier(
                n_estimators=400,
                learning_rate=0.03,
                subsample=0.9,
                colsample_bytree=0.9,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
            {"num_leaves": [31, 63], "max_depth": [-1, 6]},
        )

    # Optional Torch NN; to be used after preprocessing (dense numeric array)
    # Keep NN optional; enable by a flag if needed (excluded from default grid to save time)

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
        # device will be set lazily when torch is imported in fit/predict
        self.device = None

    def _criterion(self, logits, y):
        import torch
        import torch.nn as nn
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
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        if self.in_dim is None:
            self.in_dim = X.shape[1]
        if self.model is None:
            h = getattr(self, 'hidden', 64)
            d = getattr(self, 'dropout', 0.2)
            self.model = nn.Sequential(
                nn.Linear(self.in_dim, h), nn.ReLU(), nn.Dropout(d),
                nn.Linear(h, h), nn.ReLU(), nn.Dropout(d),
                nn.Linear(h, 1)
            )
        # cache hyperparams on self for get_params
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y.values if hasattr(y, "values") else y, dtype=torch.float32).view(-1, 1)
        ds = TensorDataset(X_t, y_t)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        best_loss = float("inf")
        best_state = None
        # set device lazily
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

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
        import torch
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device if self.device is not None else 'cpu')
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
