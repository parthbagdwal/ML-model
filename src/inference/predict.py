"""Stage 8: Production Inference Engine & Model Integration for mineRakshak-ai.

This module provides a production-grade inference layer for real-time risk assessment
of obstacle observations detected by haul truck LiDAR and perception pipelines.

Core Capabilities:
1. Loads frozen primary model (HistGradientBoostingClassifier) and preprocessor.
2. Accepts single observations (dict/Series) or batches (list of dicts/DataFrame).
3. Strictly validates input types, physical limits, and schema requirements.
4. Preserves deterministic 17-feature ordering without refitting transformers.
5. Returns predicted risk tier (SAFE, CAUTION, WARNING, CRITICAL), 4-class probabilities,
   confidence score (argmax probability), and deterministic human-readable response actions.
6. Evaluates secondary model (XGBoost) for arbitration/verification if requested.

Safety Notice:
These outputs are generated from synthetic prototype models and do NOT certify
physical haul truck safety without real-world sensor calibration and HIL testing.
"""

from __future__ import annotations

import json
import os
import sys
import time
import types
from pathlib import Path
from typing import Any, Sequence

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Decoupled loading for sklearn HistGradientBoostingClassifier on Python 3.14 (Windows SAC safe)
import sklearn
if "sklearn.ensemble" not in sys.modules:
    _pkg = types.ModuleType("sklearn.ensemble")
    _pkg.__path__ = [os.path.join(os.path.dirname(sklearn.__file__), "ensemble")]
    sys.modules["sklearn.ensemble"] = _pkg

from sklearn.ensemble._hist_gradient_boosting.gradient_boosting import (
    HistGradientBoostingClassifier,
)

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from src.preprocessing.preprocess import MineRakshakPreprocessor
from src.preprocessing.schema import (
    MAX_SAFE_TTC_S,
    NUMERICAL_FEATURES,
    OBJECT_TYPES,
    RISK_CLASSES,
    calculate_ttc,
)

# File Paths
MODELS_DIR = PROJECT_ROOT / "models"
PRIMARY_MODEL_PATH = MODELS_DIR / "hgb_model.joblib"
SECONDARY_MODEL_PATH = MODELS_DIR / "xgboost_model.json"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
LABEL_MAPPING_PATH = MODELS_DIR / "label_mapping.json"
FEATURE_SCHEMA_PATH = MODELS_DIR / "feature_schema.json"
MANIFEST_PATH = MODELS_DIR / "manifest.json"

# Safety-oriented human-readable risk response mapping
RESPONSE_MAPPINGS: dict[str, str] = {
    "SAFE": "no immediate hazard",
    "CAUTION": "increased awareness / monitor",
    "WARNING": "active warning / prepare intervention",
    "CRITICAL": "immediate hazard / urgent intervention",
}

SAFETY_DISCLAIMER: str = (
    "Prototype decision support output based on synthetic LiDAR features. "
    "Does NOT certify physical haul truck safety."
)


class InferenceValidationError(ValueError):
    """Raised when an obstacle observation violates the schema or physical bounds."""
    pass


class MineRakshakInferenceEngine:
    """Production inference engine for haul truck obstacle collision risk assessment.

    Manages artifact lifecycle, input validation, deterministic feature engineering,
    and class probability calculation.
    """

    def __init__(
        self,
        primary_model_path: Path | str = PRIMARY_MODEL_PATH,
        preprocessor_path: Path | str = PREPROCESSOR_PATH,
        label_mapping_path: Path | str = LABEL_MAPPING_PATH,
        feature_schema_path: Path | str = FEATURE_SCHEMA_PATH,
        secondary_model_path: Path | str | None = SECONDARY_MODEL_PATH,
        load_secondary: bool = False,
    ) -> None:
        self.primary_model_path = Path(primary_model_path)
        self.preprocessor_path = Path(preprocessor_path)
        self.label_mapping_path = Path(label_mapping_path)
        self.feature_schema_path = Path(feature_schema_path)
        self.secondary_model_path = Path(secondary_model_path) if secondary_model_path else None
        self.load_secondary = load_secondary

        self.primary_model: HistGradientBoostingClassifier | None = None
        self.secondary_model: xgb.XGBClassifier | None = None
        self.preprocessor: MineRakshakPreprocessor | None = None
        self.label_mapping: dict[str, Any] = {}
        self.feature_schema: dict[str, Any] = {}
        self.int_to_class: dict[int, str] = {}
        self.class_to_int: dict[str, int] = {}

        self.is_loaded: bool = False
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load all frozen model and preprocessing artifacts into memory."""
        # 1. Feature schema & Label mapping
        if not self.feature_schema_path.exists():
            raise FileNotFoundError(f"Missing feature schema at {self.feature_schema_path}")
        with open(self.feature_schema_path, "r", encoding="utf-8") as f:
            self.feature_schema = json.load(f)

        if not self.label_mapping_path.exists():
            raise FileNotFoundError(f"Missing label mapping at {self.label_mapping_path}")
        with open(self.label_mapping_path, "r", encoding="utf-8") as f:
            self.label_mapping = json.load(f)

        self.class_to_int = {k: int(v) for k, v in self.label_mapping["class_to_int"].items()}
        self.int_to_class = {int(k): v for k, v in self.label_mapping["int_to_class"].items()}

        # 2. Frozen Preprocessor
        if not self.preprocessor_path.exists():
            raise FileNotFoundError(f"Missing preprocessor at {self.preprocessor_path}")
        self.preprocessor = joblib.load(self.preprocessor_path)
        if not getattr(self.preprocessor, "is_fitted", False):
            raise RuntimeError("Loaded preprocessor is not fitted!")

        # 3. Primary Model (HistGradientBoosting)
        if not self.primary_model_path.exists():
            raise FileNotFoundError(f"Missing primary model at {self.primary_model_path}")
        self.primary_model = joblib.load(self.primary_model_path)

        # 4. Optional Secondary Model (XGBoost)
        if self.load_secondary and self.secondary_model_path and self.secondary_model_path.exists():
            self.secondary_model = xgb.XGBClassifier()
            self.secondary_model.load_model(str(self.secondary_model_path))

        self.is_loaded = True

    def validate_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Strictly validate and sanitize a single raw obstacle observation dictionary.

        Raises:
            InferenceValidationError: If required features are missing, non-numeric,
                                     or outside physically allowable sensor bounds.
        """
        if not isinstance(observation, dict):
            raise InferenceValidationError(f"Observation must be a dictionary, got {type(observation).__name__}")

        # Check required numerical features
        required_numerical = [
            "distance_m",
            "object_x_m",
            "object_y_m",
            "object_z_m",
            "object_width_m",
            "object_height_m",
            "object_length_m",
            "point_count",
            "relative_velocity_mps",
            "truck_speed_kmph",
        ]

        cleaned: dict[str, Any] = {}

        # Validate categorical object_type
        if "object_type" not in observation:
            raise InferenceValidationError("Missing required feature: 'object_type'")
        obj_type_val = observation["object_type"]
        if obj_type_val is None or (isinstance(obj_type_val, float) and np.isnan(obj_type_val)):
            raise InferenceValidationError("Invalid 'object_type': cannot be null or NaN")
        cleaned["object_type"] = str(obj_type_val).strip()

        # Validate numerical features
        for key in required_numerical:
            if key not in observation:
                raise InferenceValidationError(f"Missing required numerical feature: '{key}'")
            val = observation[key]

            # Reject None or NaN
            if val is None or (isinstance(val, float) and np.isnan(val)):
                raise InferenceValidationError(f"Feature '{key}' cannot be null or NaN")

            # Try numeric cast
            try:
                numeric_val = float(val)
            except (ValueError, TypeError):
                raise InferenceValidationError(
                    f"Feature '{key}' must be numeric, got '{val}' ({type(val).__name__})"
                )

            # Check for infinities
            if np.isinf(numeric_val):
                raise InferenceValidationError(f"Feature '{key}' cannot be infinite")

            # Physical domain sanity checks
            if key == "distance_m" and numeric_val < 0:
                raise InferenceValidationError(f"'distance_m' cannot be negative: {numeric_val}")
            if key in ["object_width_m", "object_height_m", "object_length_m"] and numeric_val <= 0:
                raise InferenceValidationError(f"Physical dimension '{key}' must be positive: {numeric_val}")
            if key == "point_count" and numeric_val < 0:
                raise InferenceValidationError(f"'point_count' cannot be negative: {numeric_val}")
            if key == "truck_speed_kmph" and numeric_val < 0:
                raise InferenceValidationError(f"'truck_speed_kmph' cannot be negative: {numeric_val}")

            cleaned[key] = int(numeric_val) if key == "point_count" else numeric_val

        # Handle time_to_collision_s
        if "time_to_collision_s" in observation and observation["time_to_collision_s"] is not None:
            ttc_val = observation["time_to_collision_s"]
            try:
                ttc_float = float(ttc_val)
            except (ValueError, TypeError):
                raise InferenceValidationError(f"'time_to_collision_s' must be numeric, got '{ttc_val}'")
            if np.isnan(ttc_float) or np.isinf(ttc_float) or ttc_float < 0:
                raise InferenceValidationError(f"Invalid 'time_to_collision_s': {ttc_float}")
            cleaned["time_to_collision_s"] = min(ttc_float, MAX_SAFE_TTC_S)
        else:
            # Dynamically derive TTC if not provided
            cleaned["time_to_collision_s"] = float(
                calculate_ttc(cleaned["distance_m"], cleaned["relative_velocity_mps"])
            )

        return cleaned

    def predict_single(
        self,
        observation: dict[str, Any] | pd.Series,
        include_secondary: bool = False,
    ) -> dict[str, Any]:
        """Perform risk prediction on a single obstacle observation.

        Args:
            observation: Raw observation dictionary or pandas Series.
            include_secondary: If True and XGBoost is loaded, returns secondary model vote.

        Returns:
            Dictionary containing predicted risk level, probabilities, confidence,
            recommended safety action, and metadata.
        """
        if isinstance(observation, pd.Series):
            obs_dict = observation.to_dict()
        else:
            obs_dict = observation

        cleaned_dict = self.validate_observation(obs_dict)
        df_input = pd.DataFrame([cleaned_dict])

        # Preprocess features (never refits)
        X_trans = self.preprocessor.transform(df_input)

        # Predict with Primary Model (HistGradientBoosting)
        probs = self.primary_model.predict_proba(X_trans)[0]
        pred_idx = int(np.argmax(probs))
        pred_label = self.int_to_class[pred_idx]
        confidence = float(probs[pred_idx])

        prob_dict = {
            self.int_to_class[i]: round(float(probs[i]), 4) for i in range(len(probs))
        }

        result: dict[str, Any] = {
            "risk_level": pred_label,
            "predicted_risk_level": pred_label,
            "class_id": pred_idx,
            "risk_code": pred_idx,
            "confidence": round(confidence, 4),
            "probabilities": prob_dict,
            "model": "HistGradientBoostingClassifier",
            "primary_model": "HistGradientBoostingClassifier",
            "status": "valid",
            "recommendation": RESPONSE_MAPPINGS[pred_label],
            "recommended_action": RESPONSE_MAPPINGS[pred_label],
            "disclaimer": SAFETY_DISCLAIMER,
        }

        # Optional secondary model vote
        if include_secondary:
            if self.secondary_model is None:
                self.secondary_model = xgb.XGBClassifier()
                self.secondary_model.load_model(str(self.secondary_model_path))
            xgb_probs = self.secondary_model.predict_proba(X_trans)[0]
            xgb_idx = int(np.argmax(xgb_probs))
            xgb_label = self.int_to_class[xgb_idx]
            result["secondary_model_vote"] = {
                "model_name": "XGBClassifier",
                "predicted_risk_level": xgb_label,
                "confidence": round(float(xgb_probs[xgb_idx]), 4),
                "agreement_with_primary": (xgb_label == pred_label),
            }

        return result

    def predict_batch(
        self,
        observations: Sequence[dict[str, Any]] | pd.DataFrame,
        include_secondary: bool = False,
    ) -> list[dict[str, Any]]:
        """Perform vectorized risk prediction on a batch of obstacle observations.

        Args:
            observations: List of observation dicts or pandas DataFrame.
            include_secondary: If True and secondary model is available, includes voter results.

        Returns:
            List of result dictionaries.
        """
        if isinstance(observations, pd.DataFrame):
            raw_list = observations.to_dict(orient="records")
        elif isinstance(observations, (list, tuple)):
            raw_list = list(observations)
        else:
            raise InferenceValidationError(
                f"Batch input must be DataFrame or Sequence of dicts, got {type(observations).__name__}"
            )

        if len(raw_list) == 0:
            return []

        # Validate all observations
        cleaned_list = [self.validate_observation(obs) for obs in raw_list]
        df_batch = pd.DataFrame(cleaned_list)

        # Batch transform
        X_trans = self.preprocessor.transform(df_batch)

        # Batch inference
        batch_probs = self.primary_model.predict_proba(X_trans)
        batch_preds = np.argmax(batch_probs, axis=1)

        # Optional secondary inference
        batch_xgb_probs = None
        if include_secondary:
            if self.secondary_model is None:
                self.secondary_model = xgb.XGBClassifier()
                self.secondary_model.load_model(str(self.secondary_model_path))
            batch_xgb_probs = self.secondary_model.predict_proba(X_trans)

        results: list[dict[str, Any]] = []
        for i in range(len(raw_list)):
            probs = batch_probs[i]
            pred_idx = int(batch_preds[i])
            pred_label = self.int_to_class[pred_idx]
            conf = float(probs[pred_idx])

            prob_dict = {
                self.int_to_class[j]: round(float(probs[j]), 4) for j in range(len(probs))
            }

            res: dict[str, Any] = {
                "risk_level": pred_label,
                "predicted_risk_level": pred_label,
                "class_id": pred_idx,
                "risk_code": pred_idx,
                "confidence": round(conf, 4),
                "probabilities": prob_dict,
                "model": "HistGradientBoostingClassifier",
                "primary_model": "HistGradientBoostingClassifier",
                "status": "valid",
                "recommendation": RESPONSE_MAPPINGS[pred_label],
                "recommended_action": RESPONSE_MAPPINGS[pred_label],
                "disclaimer": SAFETY_DISCLAIMER,
            }

            if include_secondary and batch_xgb_probs is not None:
                xgb_p = batch_xgb_probs[i]
                xgb_idx = int(np.argmax(xgb_p))
                xgb_label = self.int_to_class[xgb_idx]
                res["secondary_model_vote"] = {
                    "model_name": "XGBClassifier",
                    "predicted_risk_level": xgb_label,
                    "confidence": round(float(xgb_p[xgb_idx]), 4),
                    "agreement_with_primary": (xgb_label == pred_label),
                }

            results.append(res)

        return results


# Module-level convenience functions
_DEFAULT_ENGINE: MineRakshakInferenceEngine | None = None


def get_inference_engine() -> MineRakshakInferenceEngine:
    """Obtain or initialize the global singleton inference engine."""
    global _DEFAULT_ENGINE
    if _DEFAULT_ENGINE is None:
        _DEFAULT_ENGINE = MineRakshakInferenceEngine()
    return _DEFAULT_ENGINE


def predict(
    observation: dict[str, Any] | pd.Series | Sequence[dict[str, Any]] | pd.DataFrame,
    include_secondary: bool = False,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Predict safety risk tier for a single observation or a batch.

    Args:
        observation: Single observation (dict/Series) or batch (list of dicts/DataFrame).
        include_secondary: If True, includes secondary XGBoost verification vote.

    Returns:
        Prediction dictionary (single) or list of prediction dictionaries (batch).
    """
    engine = get_inference_engine()
    if isinstance(observation, (list, tuple)) or (isinstance(observation, pd.DataFrame) and len(observation) > 1):
        return engine.predict_batch(observation, include_secondary=include_secondary)
    elif isinstance(observation, pd.DataFrame) and len(observation) == 1:
        return engine.predict_single(observation.iloc[0].to_dict(), include_secondary=include_secondary)
    else:
        return engine.predict_single(observation, include_secondary=include_secondary)
