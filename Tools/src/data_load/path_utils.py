"""Utility functions for managing processed data paths and file operations.

This module provides helper functions for:
- Getting latest files by timestamp
- Managing model-specific directories
- Handling plot file overwriting
- Cleaning up old files
"""
from __future__ import annotations

import re
from pathlib import Path
from datetime import datetime
from typing import Optional, List
import glob


def get_timestamp() -> str:
    """Get current timestamp in standard format YYYYMMDD_HHMMSS.
    
    Returns:
        Formatted timestamp string
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def extract_timestamp_from_filename(filename: str) -> Optional[datetime]:
    """Extract timestamp from filename with format *_YYYYMMDD_HHMMSS.*.
    
    Args:
        filename: Filename containing timestamp
        
    Returns:
        datetime object if timestamp found, None otherwise
    """
    pattern = r"(\d{8}_\d{6})"
    match = re.search(pattern, filename)
    if match:
        try:
            return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
        except ValueError:
            return None
    return None


def get_latest_file(directory: Path, pattern: str) -> Optional[Path]:
    """Get the most recent file matching pattern based on timestamp in filename.
    
    Args:
        directory: Directory to search
        pattern: Glob pattern for files (e.g., "model_dtree_*.joblib")
        
    Returns:
        Path to most recent file, or None if no files found
    """
    if not directory.exists():
        return None
    
    files = list(directory.glob(pattern))
    if not files:
        return None
    
    # Sort by timestamp extracted from filename
    files_with_timestamps = []
    for file in files:
        timestamp = extract_timestamp_from_filename(file.name)
        if timestamp:
            files_with_timestamps.append((file, timestamp))
    
    if not files_with_timestamps:
        # Fallback to modification time
        return max(files, key=lambda f: f.stat().st_mtime)
    
    # Return file with latest timestamp
    return max(files_with_timestamps, key=lambda x: x[1])[0]


def get_latest_model(model_type: str, models_dir: Path) -> Optional[Path]:
    """Get the most recent model file for a specific model type.
    
    Args:
        model_type: Model type (e.g., "dtree", "knn", "xgb")
        models_dir: Base models directory (e.g., Tools/processed/models)
        
    Returns:
        Path to latest model file, or None if not found
    """
    model_dir = models_dir / model_type
    if not model_dir.exists():
        return None
    
    return get_latest_file(model_dir, f"model_{model_type}_*.joblib")


def get_latest_testset(model_type: str, testsets_dir: Path) -> Optional[Path]:
    """Get the most recent testset for a specific model type or task.
    
    Args:
        model_type: Model type (e.g., "dtree", "knn", "xgb") or task name (e.g., "mortality")
        testsets_dir: Testsets directory (e.g., Tools/processed/models/testsets)
        
    Returns:
        Path to latest testset file, or None if not found
    """
    # Try exact match first (model_type or task)
    result = get_latest_file(testsets_dir, f"testset_{model_type}_*.parquet")
    if result:
        return result
    
    # If not found, return the most recent testset file regardless of type
    if testsets_dir.exists():
        all_testsets = sorted(
            testsets_dir.glob("testset_*.parquet"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        if all_testsets:
            return all_testsets[0]
    
    return None


def get_latest_trainset(model_type: str, testsets_dir: Path) -> Optional[Path]:
    """Get the most recent trainset for a specific model type.
    
    Args:
        model_type: Model type (e.g., "dtree", "knn", "xgb")
        testsets_dir: Testsets directory (e.g., Tools/processed/models/testsets)
        
    Returns:
        Path to latest trainset file, or None if not found
    """
    return get_latest_file(testsets_dir, f"trainset_{model_type}_*.parquet")


def get_latest_model_by_task(task: str, models_dir: Path) -> Optional[Path]:
    """Get the most recent model for any model type trained on a specific task.
    
    Searches across all model type subdirectories (dtree, knn, xgb, etc.) and
    returns the most recently saved model. Since models are saved without task name,
    it returns the most recent model file found across all types.
    
    Args:
        task: Task name (e.g., "mortality", "arrhythmia") - currently not used in filename
        models_dir: Base models directory (e.g., Tools/processed/models)
        
    Returns:
        Path to latest model file for the task, or None if not found
    """
    all_models = []
    
    # Search in all model type subdirectories
    if models_dir.exists():
        for model_dir in models_dir.iterdir():
            if model_dir.is_dir() and not model_dir.name.startswith('.'):
                # Look for all model files (format: model_{type}_{timestamp}.joblib)
                for model_file in model_dir.glob("model_*.joblib"):
                    all_models.append(model_file)
    
    if not all_models:
        return None
    
    # Return the most recent one based on modification time
    return max(all_models, key=lambda p: p.stat().st_mtime)


def get_model_metadata(model_path: Path) -> Optional[Path]:
    """Get the metadata file associated with a model.
    
    Args:
        model_path: Path to model file (.joblib)
        
    Returns:
        Path to metadata file if exists, None otherwise
    """
    metadata_path = model_path.with_suffix('.metadata.json')
    return metadata_path if metadata_path.exists() else None


def load_model_with_metadata(model_path: Path):
    """Load a model and its metadata.
    
    Args:
        model_path: Path to model file
        
    Returns:
        Tuple of (model, metadata) where metadata is ModelMetadata or None
    """
    import joblib
    from ..models.metadata import ModelMetadata
    
    # Load model
    model = joblib.load(model_path)
    
    # Try to load metadata
    metadata_path = get_model_metadata(model_path)
    metadata = None
    if metadata_path:
        try:
            metadata = ModelMetadata.load(metadata_path)
        except Exception as e:
            print(f"Warning: Could not load metadata: {e}")
    
    return model, metadata


def get_latest_plot(plots_dir: Path, base_name: str) -> Optional[Path]:
    """Get the most recent plot file (PNG or HTML) matching a base name.
    
    Args:
        plots_dir: Directory containing plots
        base_name: Base name of the plot (e.g., "comparison_matrix", "roc_curve")
        
    Returns:
        Path to latest plot file (.png or .html), or None if not found
    """
    if not plots_dir.exists():
        return None
    
    # Try PNG first
    png_file = get_latest_file(plots_dir, f"{base_name}_*.png")
    if png_file:
        return png_file
    
    # Try exact match PNG
    png_exact = plots_dir / f"{base_name}.png"
    if png_exact.exists():
        return png_exact
    
    # Try HTML
    html_file = get_latest_file(plots_dir, f"{base_name}_*.html")
    if html_file:
        return html_file
    
    # Try exact match HTML
    html_exact = plots_dir / f"{base_name}.html"
    if html_exact.exists():
        return html_exact
    
    return None


def get_latest_cleaned_dataset(cleaned_datasets_dir: Path) -> Optional[Path]:
    """Get the most recent cleaned dataset.
    
    Args:
        cleaned_datasets_dir: Cleaned datasets directory
        
    Returns:
        Path to latest cleaned dataset, or None if not found
    """
    return get_latest_file(cleaned_datasets_dir, "cleaned_dataset_*.csv")


def save_plot_with_overwrite(
    fig,
    plot_dir: Path,
    plot_name: str,
    model_type: Optional[str] = None,
    formats: List[str] = None
) -> List[Path]:
    """Save a plot, overwriting previous version for the same model type.
    
    This function ensures only the latest plot of each type is kept for each model.
    
    Args:
        fig: Plotly or matplotlib figure object
        plot_dir: Base plots directory (e.g., Tools/processed/plots/evaluation)
        plot_name: Base name for the plot (e.g., "roc_curve", "confusion_matrix")
        model_type: Model type (e.g., "dtree"). If None, saves without model subdirectory
        formats: List of formats to save (default: ["png", "html"])
        
    Returns:
        List of paths where plots were saved
    """
    if formats is None:
        formats = ["png", "html"]
    
    # Determine save directory
    if model_type:
        save_dir = plot_dir / model_type
    else:
        save_dir = plot_dir
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Remove old versions of this plot for this model
    old_files = list(save_dir.glob(f"{plot_name}.*"))
    for old_file in old_files:
        try:
            old_file.unlink()
        except Exception:
            pass  # Ignore errors if file is in use
    
    # Save new plot
    saved_paths = []
    for fmt in formats:
        save_path = save_dir / f"{plot_name}.{fmt}"
        
        try:
            if fmt == "html":
                # Plotly figure
                if hasattr(fig, 'write_html'):
                    fig.write_html(str(save_path))
                    saved_paths.append(save_path)
            elif fmt == "png":
                # Try plotly first, then matplotlib
                if hasattr(fig, 'write_image'):
                    fig.write_image(str(save_path))
                    saved_paths.append(save_path)
                elif hasattr(fig, 'savefig'):
                    fig.savefig(str(save_path), dpi=300, bbox_inches='tight')
                    saved_paths.append(save_path)
            elif fmt in ["jpg", "jpeg", "svg", "pdf"]:
                if hasattr(fig, 'write_image'):
                    fig.write_image(str(save_path))
                    saved_paths.append(save_path)
                elif hasattr(fig, 'savefig'):
                    fig.savefig(str(save_path), dpi=300, bbox_inches='tight')
                    saved_paths.append(save_path)
        except Exception as e:
            print(f"Warning: Could not save plot in {fmt} format: {e}")
    
    return saved_paths


def save_model_with_cleanup(
    model,
    model_type: str,
    models_dir: Path,
    keep_n_latest: int = 1,
    metadata = None
) -> Path:
    """Save model with metadata and optionally clean up old versions.
    
    Args:
        model: Model object to save (must be joblib-serializable)
        model_type: Model type (e.g., "dtree", "knn", "xgb")
        models_dir: Base models directory (e.g., Tools/processed/models)
        keep_n_latest: Number of latest models to keep (default: 1)
        metadata: Optional ModelMetadata object to save alongside the model
        
    Returns:
        Path where model was saved
    """
    import joblib
    
    model_dir = models_dir / model_type
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save new model
    timestamp = get_timestamp()
    save_path = model_dir / f"model_{model_type}_{timestamp}.joblib"
    joblib.dump(model, save_path)
    
    # Save metadata if provided
    if metadata is not None:
        metadata_path = model_dir / f"model_{model_type}_{timestamp}.metadata.json"
        metadata.model_file_path = str(save_path)
        metadata.save(metadata_path, format="json")
    
    # Clean up old models and their metadata
    if keep_n_latest > 0:
        old_models = sorted(
            model_dir.glob(f"model_{model_type}_*.joblib"),
            key=lambda f: extract_timestamp_from_filename(f.name) or datetime.min,
            reverse=True
        )
        
        # Keep only the N most recent
        for old_model in old_models[keep_n_latest:]:
            try:
                old_model.unlink()
                # Also delete associated metadata file
                metadata_file = old_model.with_suffix('.metadata.json')
                if metadata_file.exists():
                    metadata_file.unlink()
            except Exception:
                pass  # Ignore errors
    
    return save_path


def save_dataset_with_timestamp(
    df,
    directory: Path,
    prefix: str = "dataset",
    format: str = "csv"
) -> Path:
    """Save dataset with timestamp.
    
    Args:
        df: pandas DataFrame
        directory: Directory to save to
        prefix: Filename prefix (e.g., "cleaned_dataset", "testset_dtree")
        format: File format ("csv" or "parquet")
        
    Returns:
        Path where dataset was saved
    """
    directory.mkdir(parents=True, exist_ok=True)
    
    timestamp = get_timestamp()
    filename = f"{prefix}_{timestamp}.{format}"
    save_path = directory / filename
    
    if format == "csv":
        df.to_csv(save_path, index=False)
    elif format == "parquet":
        df.to_parquet(save_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    return save_path


def cleanup_old_testsets(
    model_type: str,
    testsets_dir: Path,
    keep_n_latest: int = 1
) -> None:
    """Clean up old testsets for a model type, keeping only the N most recent.
    
    Args:
        model_type: Model type (e.g., "dtree", "knn", "xgb")
        testsets_dir: Testsets directory
        keep_n_latest: Number of latest testsets to keep
    """
    for prefix in ["testset", "trainset"]:
        old_files = sorted(
            testsets_dir.glob(f"{prefix}_{model_type}_*.parquet"),
            key=lambda f: extract_timestamp_from_filename(f.name) or datetime.min,
            reverse=True
        )
        
        for old_file in old_files[keep_n_latest:]:
            try:
                old_file.unlink()
            except Exception:
                pass


def get_all_model_types(models_dir: Path) -> List[str]:
    """Get list of all model types that have saved models.
    
    Args:
        models_dir: Base models directory
        
    Returns:
        List of model type names
    """
    if not models_dir.exists():
        return []
    
    model_types = []
    for subdir in models_dir.iterdir():
        if subdir.is_dir() and subdir.name != "testsets":
            # Check if directory has any model files
            if list(subdir.glob("model_*.joblib")):
                model_types.append(subdir.name)
    
    return sorted(model_types)
