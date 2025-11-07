"""Advanced checkpointing system for granular training state persistence."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import joblib
import pandas as pd
import numpy as np


class TrainingCheckpoint:
    """Granular checkpoint for training state."""
    
    def __init__(
        self,
        checkpoint_dir: str | Path,
        experiment_id: str,
        models: Dict[str, Tuple[object, Dict]],
    ):
        """Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to store checkpoints
            experiment_id: Unique identifier for this experiment
            models: Dictionary of models to train
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_id = experiment_id
        self.models = models
        
        # State tracking
        self.state = {
            'experiment_id': experiment_id,
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'models': list(models.keys()),
            'completed_models': [],
            'current_model': None,
            'current_model_results': {},
            'global_results': {},
            'metadata': {}
        }
        
        self._load_or_create_state()
    
    def _get_state_path(self) -> Path:
        """Get path to state file."""
        return self.checkpoint_dir / f"{self.experiment_id}_state.json"
    
    def _get_checkpoint_path(self, model_name: str, fold_idx: int) -> Path:
        """Get path to checkpoint file for specific fold."""
        return self.checkpoint_dir / f"{self.experiment_id}_{model_name}_fold{fold_idx}.joblib"
    
    def _get_model_results_path(self, model_name: str) -> Path:
        """Get path to model results file."""
        return self.checkpoint_dir / f"{self.experiment_id}_{model_name}_results.joblib"
    
    def _load_or_create_state(self):
        """Load existing state or create new."""
        state_path = self._get_state_path()
        
        if state_path.exists():
            try:
                with open(state_path, 'r') as f:
                    loaded_state = json.load(f)
                
                # Merge with current state (preserve structure)
                self.state.update(loaded_state)
                self.state['last_updated'] = datetime.now().isoformat()
                
                print(f"✅ Checkpoint cargado: {len(self.state['completed_models'])}/{len(self.state['models'])} modelos completados")
            except Exception as e:
                print(f"⚠️ Error cargando checkpoint: {e}. Creando nuevo...")
                self._save_state()
        else:
            self._save_state()
    
    def _save_state(self):
        """Save current state to disk."""
        self.state['last_updated'] = datetime.now().isoformat()
        
        state_path = self._get_state_path()
        with open(state_path, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def should_skip_model(self, model_name: str) -> bool:
        """Check if model was already completed."""
        return model_name in self.state['completed_models']
    
    def get_completed_folds(self, model_name: str) -> List[int]:
        """Get list of completed fold indices for a model."""
        if model_name not in self.state['current_model_results']:
            return []
        
        return list(self.state['current_model_results'][model_name].get('completed_folds', []))
    
    def save_fold_result(
        self,
        model_name: str,
        fold_idx: int,
        score: float,
        fitted_pipeline: object,
        metadata: Optional[Dict] = None
    ):
        """Save result for a single fold.
        
        Args:
            model_name: Name of the model
            fold_idx: Index of the fold
            score: Score achieved on this fold
            fitted_pipeline: Fitted pipeline for this fold
            metadata: Additional metadata (train/val indices, etc.)
        """
        # Update state
        if model_name not in self.state['current_model_results']:
            self.state['current_model_results'][model_name] = {
                'scores': [],
                'completed_folds': [],
                'metadata': []
            }
        
        self.state['current_model_results'][model_name]['scores'].append(score)
        self.state['current_model_results'][model_name]['completed_folds'].append(fold_idx)
        if metadata:
            self.state['current_model_results'][model_name]['metadata'].append(metadata)
        
        self.state['current_model'] = model_name
        
        # Save fold checkpoint (pipeline)
        checkpoint_path = self._get_checkpoint_path(model_name, fold_idx)
        joblib.dump(fitted_pipeline, checkpoint_path)
        
        # Save state
        self._save_state()
    
    def complete_model(self, model_name: str, final_results: Dict):
        """Mark model as completed and save final results.
        
        Args:
            model_name: Name of the model
            final_results: Final aggregated results (mean, std, all_scores, etc.)
        """
        # Save final model results
        results_path = self._get_model_results_path(model_name)
        joblib.dump(final_results, results_path)
        
        # Update state
        self.state['completed_models'].append(model_name)
        self.state['global_results'][model_name] = {
            'mean_score': final_results['mean_score'],
            'std_score': final_results['std_score'],
            'n_runs': final_results['n_runs']
        }
        
        # Clean up fold-level results (optional - keep for now for debugging)
        # self.state['current_model_results'].pop(model_name, None)
        
        self._save_state()
        
        print(f"✅ Modelo '{model_name}' completado: μ={final_results['mean_score']:.4f}, σ={final_results['std_score']:.4f}")
    
    def get_model_results(self, model_name: str) -> Optional[Dict]:
        """Load completed model results from disk.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model results dictionary or None if not completed
        """
        if model_name not in self.state['completed_models']:
            return None
        
        results_path = self._get_model_results_path(model_name)
        if not results_path.exists():
            return None
        
        return joblib.load(results_path)
    
    def load_fold_checkpoint(self, model_name: str, fold_idx: int) -> Optional[object]:
        """Load fitted pipeline for a specific fold.
        
        Args:
            model_name: Name of the model
            fold_idx: Index of the fold
            
        Returns:
            Fitted pipeline or None if not found
        """
        checkpoint_path = self._get_checkpoint_path(model_name, fold_idx)
        if not checkpoint_path.exists():
            return None
        
        return joblib.load(checkpoint_path)
    
    def is_complete(self) -> bool:
        """Check if all models have been completed."""
        return len(self.state['completed_models']) == len(self.state['models'])
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current progress statistics.
        
        Returns:
            Dictionary with progress information
        """
        total_models = len(self.state['models'])
        completed_models = len(self.state['completed_models'])
        
        current_model = self.state.get('current_model')
        current_progress = {}
        
        if current_model and current_model in self.state['current_model_results']:
            results = self.state['current_model_results'][current_model]
            current_progress = {
                'model': current_model,
                'completed_folds': len(results['completed_folds']),
                'scores': results['scores']
            }
        
        return {
            'total_models': total_models,
            'completed_models': completed_models,
            'progress_pct': (completed_models / total_models * 100) if total_models > 0 else 0,
            'current_model_progress': current_progress,
            'completed_model_names': self.state['completed_models']
        }
    
    def cleanup_checkpoints(self, keep_final: bool = True):
        """Clean up checkpoint files.
        
        Args:
            keep_final: If True, keep final model results
        """
        # Delete all fold-level checkpoints
        for model_name in self.state['models']:
            # Get all fold checkpoints for this model
            pattern = f"{self.experiment_id}_{model_name}_fold*.joblib"
            for checkpoint_file in self.checkpoint_dir.glob(pattern):
                try:
                    checkpoint_file.unlink()
                except Exception as e:
                    print(f"⚠️ Error eliminando checkpoint {checkpoint_file}: {e}")
        
        # Optionally delete final results
        if not keep_final:
            for model_name in self.state['completed_models']:
                results_path = self._get_model_results_path(model_name)
                try:
                    if results_path.exists():
                        results_path.unlink()
                except Exception as e:
                    print(f"⚠️ Error eliminando resultados {results_path}: {e}")
        
        # Delete state file
        try:
            state_path = self._get_state_path()
            if state_path.exists():
                state_path.unlink()
        except Exception as e:
            print(f"⚠️ Error eliminando estado: {e}")
        
        print(f"✅ Checkpoints limpiados (mantener finales: {keep_final})")


def create_experiment_id(task: str, prefix: str = "exp") -> str:
    """Create unique experiment ID.
    
    Args:
        task: Task name (e.g., 'mortality')
        prefix: Prefix for experiment ID
        
    Returns:
        Unique experiment ID
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{task}_{timestamp}"