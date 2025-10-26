"""Risk score registry."""
from __future__ import annotations

from typing import List, Optional

from .grace import GraceScore
from .timi import TIMIScore


# Global score registry
_SCORE_REGISTRY = {
    "grace": GraceScore,
    "timi": TIMIScore,
}


def get_score(name: str):
    """Get a risk score calculator by name.
    
    Args:
        name: Score name ('grace' or 'timi')
        
    Returns:
        Risk score calculator instance
        
    Raises:
        ValueError: If score name not recognized
    """
    name_lower = name.lower()
    
    if name_lower not in _SCORE_REGISTRY:
        raise ValueError(
            f"Unknown score: {name}. Available: {list(_SCORE_REGISTRY.keys())}"
        )
    
    return _SCORE_REGISTRY[name_lower]()


def list_scores() -> List[str]:
    """List available risk scores.
    
    Returns:
        List of score names
    """
    return list(_SCORE_REGISTRY.keys())
