"""FLAML Integration Module.

Provides cross-platform AutoML using FLAML (Fast and Lightweight AutoML).
This is the recommended option for Windows users as auto-sklearn
requires Linux.

FLAML advantages:
- Cross-platform (Windows, Linux, macOS)
- Faster than auto-sklearn for many tasks
- Lower memory requirements
- Supports more estimators out of the box
"""
from __future__ import annotations

import logging
import sys
import time
import io
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from contextlib import redirect_stdout, redirect_stderr

import numpy as np
import pandas as pd

from ..models.custom_base import BaseCustomClassifier, BaseCustomRegressor
from ..config import RANDOM_SEED
from .config import AutoMLConfig, AutoMLPreset

logger = logging.getLogger(__name__)


class AutoMLLogger:
    """Custom logger to capture FLAML output and route to callback."""
    
    def __init__(self, callback: Optional[Callable[[str, float], None]] = None, 
                 start_time: Optional[float] = None,
                 time_budget: int = 3600):
        self.callback = callback
        self.logs: List[str] = []
        self.start_time = start_time or time.time()
        self.time_budget = time_budget
        self.last_update = time.time()
        self.models_evaluated = 0
        self.best_score = float('inf')
        self.best_estimator = None
    
    def write(self, message: str) -> None:
        """Capture log messages."""
        if message.strip():
            self.logs.append(message.strip())
            self._parse_and_update(message)
    
    def flush(self) -> None:
        """Flush buffer."""
        pass
    
    def _parse_and_update(self, message: str) -> None:
        """Parse FLAML output and update callback."""
        now = time.time()
        elapsed = now - self.start_time
        progress = min(elapsed / self.time_budget, 0.99)
        
        # Parse different types of messages
        msg_lower = message.lower()
        
        if 'best' in msg_lower and 'loss' in msg_lower:
            # Extract best model info
            try:
                parts = message.split()
                for i, part in enumerate(parts):
                    if 'loss' in part.lower() and i + 1 < len(parts):
                        try:
                            self.best_score = float(parts[i + 1].strip(':,'))
                        except:
                            pass
            except:
                pass
            
            self.models_evaluated += 1
            status = f"🔍 Modelos evaluados: {self.models_evaluated} | Mejor score: {-self.best_score:.4f}"
            
        elif 'trial' in msg_lower or 'iteration' in msg_lower:
            self.models_evaluated += 1
            status = f"🔄 Evaluando modelo #{self.models_evaluated}..."
            
        elif 'fitting' in msg_lower or 'training' in msg_lower:
            status = f"⚙️ Entrenando: {message[:50]}..."
            
        elif any(est in msg_lower for est in ['lgbm', 'xgboost', 'rf', 'catboost', 'extra']):
            # Estimator being evaluated
            for est in ['lgbm', 'xgboost', 'rf', 'catboost', 'extra_tree', 'kneighbor', 'lrl1', 'lrl2']:
                if est in msg_lower:
                    status = f"🧪 Probando: {est.upper()}"
                    break
            else:
                status = f"🔬 {message[:60]}"
        else:
            status = f"📊 {message[:60]}"
        
        # Only update callback every 0.5 seconds to avoid flooding
        if self.callback and (now - self.last_update > 0.5):
            self.callback(status, progress)
            self.last_update = now
    
    def get_logs(self) -> List[str]:
        """Get all captured logs."""
        return self.logs.copy()


def is_flaml_available() -> bool:
    """Check if FLAML is available.
    
    Returns:
        True if FLAML can be imported
    """
    try:
        import flaml
        return True
    except ImportError:
        return False


class FLAMLClassifier(BaseCustomClassifier):
    """FLAML-based AutoML Classifier.
    
    A cross-platform AutoML classifier using Microsoft's FLAML library.
    Works on Windows, Linux, and macOS.
    
    FLAML is faster and more lightweight than auto-sklearn while
    achieving competitive accuracy.
    
    Attributes:
        config: AutoML configuration
        automl_: Fitted FLAML AutoML object
        best_estimator_: Name of best estimator
        best_config_: Best hyperparameter configuration
        best_model_: Best fitted model
    
    Example:
        >>> clf = FLAMLClassifier(time_budget=300)  # 5 minutes
        >>> clf.fit(X_train, y_train)
        >>> proba = clf.predict_proba(X_test)
    """
    
    def __init__(
        self,
        time_budget: int = 3600,
        metric: str = "roc_auc",
        estimator_list: Optional[List[str]] = None,
        n_jobs: int = -1,
        ensemble: bool = True,
        max_iter: Optional[int] = None,
        early_stop: bool = True,
        random_state: int = RANDOM_SEED,
        verbose: int = 2,  # Default verbose = 2 for detailed output
        log_file_name: Optional[str] = None,
        name: str = "FLAMLClassifier",
        progress_callback: Optional[Callable[[str, float], None]] = None,
        **kwargs
    ):
        """
        Initialize FLAML Classifier.
        
        Args:
            time_budget: Time budget in seconds
            metric: Optimization metric (roc_auc, accuracy, f1, log_loss)
            estimator_list: List of estimators to try. Options:
                - lgbm: LightGBM
                - xgboost: XGBoost
                - xgb_limitdepth: XGBoost with limited depth
                - catboost: CatBoost
                - rf: Random Forest
                - extra_tree: Extra Trees
                - kneighbor: K-Neighbors
                - lrl1: L1 Logistic Regression
                - lrl2: L2 Logistic Regression
            n_jobs: Number of parallel jobs
            ensemble: Whether to use ensemble
            max_iter: Maximum iterations (None = unlimited)
            early_stop: Enable early stopping
            random_state: Random seed
            verbose: Verbosity level (0=silent, 1=progress, 2=detailed, 3=debug)
            log_file_name: Optional log file path
            name: Model name
            progress_callback: Optional progress callback(message, progress)
            **kwargs: Additional FLAML arguments
        """
        super().__init__(name=name)
        
        self.time_budget = time_budget
        self.metric = metric
        self.estimator_list = estimator_list
        self.n_jobs = n_jobs
        self.ensemble = ensemble
        self.max_iter = max_iter
        self.early_stop = early_stop
        self.random_state = random_state
        self.verbose = verbose
        self.log_file_name = log_file_name
        self.extra_kwargs = kwargs
        self.progress_callback = progress_callback
        
        # Runtime attributes
        self.automl_: Optional[Any] = None
        self.best_estimator_: Optional[str] = None
        self.best_config_: Optional[Dict[str, Any]] = None
        self.best_model_: Optional[Any] = None
        self.best_loss_: Optional[float] = None
        self.fit_time_: float = 0.0
        self.feature_importances_: Optional[np.ndarray] = None
        self.training_logs_: List[str] = []
    
    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        **kwargs
    ) -> "FLAMLClassifier":
        """
        Fit the FLAML classifier.
        
        Args:
            X: Training features
            y: Training labels
            **kwargs: Additional FLAML fit arguments
            
        Returns:
            Self (fitted classifier)
        """
        if not is_flaml_available():
            raise ImportError(
                "FLAML is not installed. Install with: pip install flaml[automl]"
            )
        
        from flaml import AutoML
        
        self._validate_input(X, training=True)
        y = self._validate_targets(y, training=True)
        
        # Convert to arrays/DataFrames as FLAML prefers
        if isinstance(X, np.ndarray):
            X_train = pd.DataFrame(X)
        else:
            X_train = X.copy()
        
        y_train = y if isinstance(y, np.ndarray) else y.values
        
        start_time = time.time()
        
        if self.progress_callback:
            self.progress_callback("🚀 Iniciando FLAML AutoML...", 0.0)
        
        # Build FLAML settings
        flaml_settings = {
            "time_budget": self.time_budget,
            "metric": self.metric,
            "task": "classification",
            "n_jobs": self.n_jobs,
            "ensemble": self.ensemble,
            "seed": self.random_state,
            "verbose": self.verbose,
            "early_stop": self.early_stop,
        }
        
        if self.estimator_list is not None:
            flaml_settings["estimator_list"] = self.estimator_list
        
        if self.max_iter is not None:
            flaml_settings["max_iter"] = self.max_iter
        
        if self.log_file_name is not None:
            flaml_settings["log_file_name"] = self.log_file_name
        
        # Add extra kwargs
        flaml_settings.update(self.extra_kwargs)
        flaml_settings.update(kwargs)
        
        # Initialize and fit with progress tracking
        self.automl_ = AutoML()
        
        # Create custom logger to capture output
        automl_logger = AutoMLLogger(
            callback=self.progress_callback,
            start_time=start_time,
            time_budget=self.time_budget
        )
        
        # Configure FLAML logging
        import logging as std_logging
        flaml_logger = std_logging.getLogger('flaml.automl')
        
        # Add custom handler
        class CallbackHandler(std_logging.Handler):
            def __init__(self, automl_log: AutoMLLogger):
                super().__init__()
                self.automl_log = automl_log
                
            def emit(self, record):
                msg = self.format(record)
                self.automl_log.write(msg)
        
        handler = CallbackHandler(automl_logger)
        handler.setLevel(std_logging.INFO)
        flaml_logger.addHandler(handler)
        
        # Set FLAML verbosity
        if self.verbose >= 2:
            flaml_logger.setLevel(std_logging.INFO)
        elif self.verbose >= 1:
            flaml_logger.setLevel(std_logging.WARNING)
        else:
            flaml_logger.setLevel(std_logging.ERROR)
        
        try:
            # Fit with output capture for additional logging
            if self.verbose >= 2 and self.progress_callback:
                # Capture stdout for verbose output
                old_stdout = sys.stdout
                sys.stdout = automl_logger
                
                try:
                    self.automl_.fit(X_train, y_train, **flaml_settings)
                finally:
                    sys.stdout = old_stdout
            else:
                self.automl_.fit(X_train, y_train, **flaml_settings)
        
        finally:
            # Remove handler
            flaml_logger.removeHandler(handler)
        
        # Store results
        self.best_estimator_ = self.automl_.best_estimator
        self.best_config_ = self.automl_.best_config
        self.best_model_ = self.automl_.model
        self.best_loss_ = self.automl_.best_loss
        self.fit_time_ = time.time() - start_time
        self.training_logs_ = automl_logger.get_logs()
        
        # Extract feature importances if available
        self._extract_feature_importances()
        
        self.is_fitted_ = True
        
        if self.progress_callback:
            self.progress_callback(
                f"✅ FLAML completado: {self.best_estimator_} "
                f"(score={-self.best_loss_:.4f}, time={self.fit_time_:.1f}s)",
                1.0
            )
        
        logger.info(
            f"FLAML training completed: best={self.best_estimator_}, "
            f"loss={self.best_loss_:.4f}, time={self.fit_time_:.1f}s"
        )
        
        return self
    
    def get_training_logs(self) -> List[str]:
        """Get the training logs captured during fit.
        
        Returns:
            List of log messages
        """
        return self.training_logs_.copy()
    
    def _extract_feature_importances(self) -> None:
        """Extract feature importances from best model."""
        try:
            if hasattr(self.best_model_, 'feature_importances_'):
                self.feature_importances_ = self.best_model_.feature_importances_
            elif hasattr(self.best_model_, 'coef_'):
                self.feature_importances_ = np.abs(self.best_model_.coef_).flatten()
        except Exception as e:
            logger.debug(f"Could not extract feature importances: {e}")
    
    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        self._validate_input(X, training=False)
        
        if isinstance(X, np.ndarray):
            X_pred = pd.DataFrame(X)
        else:
            X_pred = X
        
        return self.automl_.predict_proba(X_pred)
    
    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Predict class labels."""
        self._validate_input(X, training=False)
        
        if isinstance(X, np.ndarray):
            X_pred = pd.DataFrame(X)
        else:
            X_pred = X
        
        return self.automl_.predict(X_pred)
    
    def get_best_model(self) -> Any:
        """Get the best fitted model."""
        return self.best_model_
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importances if available."""
        return self.feature_importances_
    
    def get_leaderboard(self) -> pd.DataFrame:
        """Get leaderboard of evaluated models.
        
        Returns:
            DataFrame with estimator performance info
        """
        if self.automl_ is None:
            return pd.DataFrame()
        
        try:
            rows = []
            
            # Best config per estimator
            if hasattr(self.automl_, 'best_config_per_estimator'):
                for est, config in self.automl_.best_config_per_estimator.items():
                    if config is not None:
                        # Try to get loss for this estimator
                        loss = None
                        if hasattr(self.automl_, 'best_loss_per_estimator'):
                            loss = self.automl_.best_loss_per_estimator.get(est)
                        
                        rows.append({
                            'estimator': est,
                            'loss': loss if loss else 'N/A',
                            'score': -loss if loss else 'N/A',
                            'config': str(config)[:100],
                            'is_best': est == self.best_estimator_,
                        })
            
            # Sort by loss
            df = pd.DataFrame(rows)
            if len(df) > 0 and 'loss' in df.columns:
                df = df.sort_values('loss', ascending=True)
            
            return df
        except Exception as e:
            logger.warning(f"Could not generate leaderboard: {e}")
            return pd.DataFrame()
    
    def get_search_history(self) -> pd.DataFrame:
        """Get the search history as a DataFrame.
        
        Returns:
            DataFrame with columns: estimator, config, loss, time
        """
        return self.get_leaderboard()
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator."""
        return {
            "time_budget": self.time_budget,
            "metric": self.metric,
            "estimator_list": self.estimator_list,
            "n_jobs": self.n_jobs,
            "ensemble": self.ensemble,
            "max_iter": self.max_iter,
            "early_stop": self.early_stop,
            "random_state": self.random_state,
            "verbose": self.verbose,
            "name": self.name,
        }
    
    def set_params(self, **params) -> "FLAMLClassifier":
        """Set parameters for this estimator."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get extended metadata."""
        metadata = super().get_metadata()
        metadata.update({
            'backend': 'flaml',
            'best_estimator': self.best_estimator_,
            'best_loss': self.best_loss_,
            'fit_time_seconds': self.fit_time_,
            'time_budget': self.time_budget,
            'metric': self.metric,
            'n_logs': len(self.training_logs_),
        })
        return metadata


class FLAMLRegressor(BaseCustomRegressor):
    """FLAML-based AutoML Regressor.
    
    Cross-platform AutoML regressor using FLAML.
    """
    
    def __init__(
        self,
        time_budget: int = 3600,
        metric: str = "r2",
        estimator_list: Optional[List[str]] = None,
        n_jobs: int = -1,
        ensemble: bool = True,
        random_state: int = RANDOM_SEED,
        verbose: int = 2,  # Default verbose = 2
        log_file_name: Optional[str] = None,
        name: str = "FLAMLRegressor",
        progress_callback: Optional[Callable[[str, float], None]] = None,
        **kwargs
    ):
        """Initialize FLAML Regressor."""
        super().__init__(name=name)
        
        self.time_budget = time_budget
        self.metric = metric
        self.estimator_list = estimator_list
        self.n_jobs = n_jobs
        self.ensemble = ensemble
        self.random_state = random_state
        self.verbose = verbose
        self.log_file_name = log_file_name
        self.extra_kwargs = kwargs
        self.progress_callback = progress_callback
        
        self.automl_: Optional[Any] = None
        self.best_estimator_: Optional[str] = None
        self.best_model_: Optional[Any] = None
        self.best_loss_: Optional[float] = None
        self.fit_time_: float = 0.0
        self.training_logs_: List[str] = []
    
    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        **kwargs
    ) -> "FLAMLRegressor":
        """Fit the FLAML regressor."""
        if not is_flaml_available():
            raise ImportError("FLAML is not installed")
        
        from flaml import AutoML
        import logging as std_logging
        
        self._validate_input(X, training=True)
        
        if isinstance(X, np.ndarray):
            X_train = pd.DataFrame(X)
        else:
            X_train = X.copy()
        
        y_train = y.values if isinstance(y, pd.Series) else y
        
        start_time = time.time()
        
        if self.progress_callback:
            self.progress_callback("🚀 Iniciando FLAML Regressor...", 0.0)
        
        settings = {
            "time_budget": self.time_budget,
            "metric": self.metric,
            "task": "regression",
            "n_jobs": self.n_jobs,
            "ensemble": self.ensemble,
            "seed": self.random_state,
            "verbose": self.verbose,
        }
        
        if self.estimator_list:
            settings["estimator_list"] = self.estimator_list
        
        if self.log_file_name:
            settings["log_file_name"] = self.log_file_name
        
        settings.update(self.extra_kwargs)
        settings.update(kwargs)
        
        # Create logger
        automl_logger = AutoMLLogger(
            callback=self.progress_callback,
            start_time=start_time,
            time_budget=self.time_budget
        )
        
        # Configure logging
        flaml_logger = std_logging.getLogger('flaml.automl')
        
        class CallbackHandler(std_logging.Handler):
            def __init__(self, automl_log: AutoMLLogger):
                super().__init__()
                self.automl_log = automl_log
                
            def emit(self, record):
                msg = self.format(record)
                self.automl_log.write(msg)
        
        handler = CallbackHandler(automl_logger)
        handler.setLevel(std_logging.INFO)
        flaml_logger.addHandler(handler)
        
        if self.verbose >= 2:
            flaml_logger.setLevel(std_logging.INFO)
        elif self.verbose >= 1:
            flaml_logger.setLevel(std_logging.WARNING)
        else:
            flaml_logger.setLevel(std_logging.ERROR)
        
        try:
            self.automl_ = AutoML()
            self.automl_.fit(X_train, y_train, **settings)
        finally:
            flaml_logger.removeHandler(handler)
        
        self.best_estimator_ = self.automl_.best_estimator
        self.best_model_ = self.automl_.model
        self.best_loss_ = self.automl_.best_loss
        self.fit_time_ = time.time() - start_time
        self.training_logs_ = automl_logger.get_logs()
        self.is_fitted_ = True
        
        if self.progress_callback:
            self.progress_callback(
                f"✅ FLAML Regressor completado: {self.best_estimator_} "
                f"(loss={self.best_loss_:.4f}, time={self.fit_time_:.1f}s)",
                1.0
            )
        
        return self
    
    def get_training_logs(self) -> List[str]:
        """Get the training logs captured during fit."""
        return self.training_logs_.copy()
    
    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        self._validate_input(X, training=False)
        
        if isinstance(X, np.ndarray):
            X_pred = pd.DataFrame(X)
        else:
            X_pred = X
        
        return self.automl_.predict(X_pred)
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters."""
        return {
            "time_budget": self.time_budget,
            "metric": self.metric,
            "estimator_list": self.estimator_list,
            "n_jobs": self.n_jobs,
            "ensemble": self.ensemble,
            "random_state": self.random_state,
            "verbose": self.verbose,
            "name": self.name,
        }
    
    def set_params(self, **params) -> "FLAMLRegressor":
        """Set parameters."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get extended metadata."""
        metadata = super().get_metadata()
        metadata.update({
            'backend': 'flaml',
            'best_estimator': self.best_estimator_,
            'best_loss': self.best_loss_,
            'fit_time_seconds': self.fit_time_,
            'time_budget': self.time_budget,
            'metric': self.metric,
            'n_logs': len(self.training_logs_),
        })
        return metadata
