"""Classic risk scores (educational approximations) and interfaces.

DISCLAIMER: The implementations provided here are simplified and intended
for research/educational use ONLY. They are NOT clinically validated and must
NOT be used for patient care decisions. For clinical use, rely on original
validated calculators and official implementations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional


@dataclass
class ScoreResult:
    name: str
    points: float
    risk_category: str
    details: Dict[str, float]


def _band(points: float, bands: Tuple[Tuple[float, str], ...]) -> str:
    for thr, label in bands:
        if points <= thr:
            return label
    return bands[-1][1]


def compute_grace_approx(inputs: Dict[str, Any]) -> ScoreResult:
    """Approximate, non-clinical GRACE-like score.

    Expects keys:
    - age (years)
    - heart_rate (bpm)
    - sbp (mmHg)
    - creatinine_mg_dl (mg/dL)
    - killip (I/II/III/IV or 1..4)
    - st_deviation (bool)
    - enzymes_elevated (bool)
    - cardiac_arrest (bool)
    """
    age = float(inputs.get("age", 0) or 0)
    hr = float(inputs.get("heart_rate", 0) or 0)
    sbp = float(inputs.get("sbp", 120) or 120)
    crea = float(inputs.get("creatinine_mg_dl", 1.0) or 1.0)
    killip_raw = inputs.get("killip", "I")
    killip_map = {"I": 1, "II": 2, "III": 3, "IV": 4, 1: 1, 2: 2, 3: 3, 4: 4}
    killip = int(killip_map.get(killip_raw, 1))
    st_dev = bool(inputs.get("st_deviation", False))
    enz = bool(inputs.get("enzymes_elevated", False))
    arrest = bool(inputs.get("cardiac_arrest", False))

    details: Dict[str, float] = {}
    # Very rough point scheme (educational):
    details["age"] = max(0.0, (age // 10) * 2.0)
    details["heart_rate"] = max(0.0, (hr - 60.0) / 10.0)
    if sbp < 80:
        details["sbp"] = 8.0
    elif sbp < 100:
        details["sbp"] = 6.0
    elif sbp < 120:
        details["sbp"] = 4.0
    elif sbp < 140:
        details["sbp"] = 2.0
    else:
        details["sbp"] = 0.0
    if crea >= 2.0:
        details["creatinine_mg_dl"] = 6.0
    elif crea >= 1.5:
        details["creatinine_mg_dl"] = 4.0
    elif crea >= 1.2:
        details["creatinine_mg_dl"] = 2.0
    else:
        details["creatinine_mg_dl"] = 0.0
    details["killip"] = {1: 0.0, 2: 3.0, 3: 6.0, 4: 10.0}[killip]
    details["st_deviation"] = 3.0 if st_dev else 0.0
    details["enzymes_elevated"] = 2.0 if enz else 0.0
    details["cardiac_arrest"] = 7.0 if arrest else 0.0

    points = float(sum(details.values()))
    # Rough risk banding
    band = _band(points, ((30, "bajo"), (60, "intermedio"), (9999, "alto")))
    return ScoreResult(name="GRACE (aprox)", points=points, risk_category=band, details=details)


def compute_timi_placeholder(inputs: Dict[str, Any]) -> ScoreResult:
    """Placeholder for TIMI score; returns NotImplemented style result."""
    return ScoreResult(name="TIMI (no implementado)", points=0.0, risk_category="N/A", details={})


def compute_action_gwtg_placeholder(inputs: Dict[str, Any]) -> ScoreResult:
    return ScoreResult(name="ACTION-GWTG (no implementado)", points=0.0, risk_category="N/A", details={})


def compute_proacs_placeholder(inputs: Dict[str, Any]) -> ScoreResult:
    return ScoreResult(name="ProACS (no implementado)", points=0.0, risk_category="N/A", details={})


SCORE_REGISTRY = {
    "grace": compute_grace_approx,
    "timi": compute_timi_placeholder,
    "action_gwtg": compute_action_gwtg_placeholder,
    "proacs": compute_proacs_placeholder,
}


def available_scores() -> Dict[str, str]:
    return {
        "grace": "GRACE (aprox. educativa)",
        "timi": "TIMI (no implementado)",
        "action_gwtg": "ACTION-GWTG (no implementado)",
        "proacs": "ProACS (no implementado)",
    }


def compute_score(name: str, inputs: Dict[str, Any]) -> ScoreResult:
    key = name.lower()
    if key not in SCORE_REGISTRY:
        raise KeyError(f"Score '{name}' no disponible")
    return SCORE_REGISTRY[key](inputs)
