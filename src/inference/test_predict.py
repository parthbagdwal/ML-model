"""Stage 8: Production Inference Test Suite for mineRakshak-ai.

Tests:
1. Valid SAFE-like input
2. Valid CAUTION-like input
3. Valid WARNING-like input
4. Valid CRITICAL-like input
5. Missing field validation
6. NaN input validation
7. Infinite input validation
8. Invalid negative physical value validation
9. Unknown object type handling
10. Repeated prediction determinism
11. Probability vector sum ≈ 1.0
12. Returned class matches argmax probability
13. Output labels match label_mapping.json
14. Preprocessing feature order matches feature_schema.json
15. Serialized model reload produces identical predictions
"""

from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.inference.predict import (
    InferenceValidationError,
    MineRakshakInferenceEngine,
    RESPONSE_MAPPINGS,
    get_inference_engine,
    predict,
)
from src.preprocessing.schema import RISK_CLASSES


class TestProductionPredict(unittest.TestCase):
    """Test suite for production inference pipeline and model packaging."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize the inference engine and standard test observations."""
        cls.engine = MineRakshakInferenceEngine(load_secondary=True)

        # 1. SAFE observation: distant stationary vehicle
        cls.safe_input = {
            "distance_m": 75.0,
            "object_x_m": 8.0,
            "object_y_m": 74.5,
            "object_z_m": 0.0,
            "object_width_m": 2.0,
            "object_height_m": 1.6,
            "object_length_m": 4.5,
            "point_count": 85,
            "relative_velocity_mps": 0.0,
            "truck_speed_kmph": 18.0,
            "time_to_collision_s": 99.9,
            "object_type": "car",
        }

        # 2. CAUTION observation: excavator working at moderate distance
        cls.caution_input = {
            "distance_m": 35.0,
            "object_x_m": 6.5,
            "object_y_m": 34.3,
            "object_z_m": 1.0,
            "object_width_m": 4.2,
            "object_height_m": 3.8,
            "object_length_m": 6.5,
            "point_count": 450,
            "relative_velocity_mps": -1.2,
            "truck_speed_kmph": 22.0,
            "time_to_collision_s": 29.17,
            "object_type": "excavator",
        }

        # 3. WARNING observation: approaching mobile crane
        cls.warning_input = {
            "distance_m": 20.0,
            "object_x_m": -3.0,
            "object_y_m": 19.8,
            "object_z_m": 0.5,
            "object_width_m": 3.0,
            "object_height_m": 3.5,
            "object_length_m": 8.5,
            "point_count": 480,
            "relative_velocity_mps": -4.5,
            "truck_speed_kmph": 25.0,
            "time_to_collision_s": 4.44,
            "object_type": "crane",
        }

        # 4. CRITICAL observation: close approaching haul truck
        cls.critical_input = {
            "distance_m": 5.5,
            "object_x_m": 0.8,
            "object_y_m": 5.4,
            "object_z_m": 0.3,
            "object_width_m": 3.4,
            "object_height_m": 3.6,
            "object_length_m": 7.5,
            "point_count": 750,
            "relative_velocity_mps": -9.0,
            "truck_speed_kmph": 32.0,
            "time_to_collision_s": 0.61,
            "object_type": "truck",
        }

        cls.models_dir = PROJECT_ROOT / "models"
        cls.metrics_dir = PROJECT_ROOT / "results" / "metrics"

    def test_01_valid_safe_like_input(self) -> None:
        """1. Verify valid SAFE-like input yields SAFE prediction and expected schema keys."""
        res = self.engine.predict_single(self.safe_input)
        self.assertEqual(res["risk_level"], "SAFE")
        self.assertEqual(res["class_id"], 0)
        self.assertEqual(res["status"], "valid")
        self.assertEqual(res["model"], "HistGradientBoostingClassifier")
        self.assertGreater(res["confidence"], 0.60)
        self.assertIn("SAFE", res["probabilities"])
        self.assertIn("recommendation", res)

    def test_02_valid_caution_like_input(self) -> None:
        """2. Verify valid CAUTION-like input yields valid risk prediction."""
        res = self.engine.predict_single(self.caution_input)
        self.assertIn(res["risk_level"], ["SAFE", "CAUTION"])
        self.assertEqual(res["status"], "valid")
        self.assertIn(res["class_id"], [0, 1])

    def test_03_valid_warning_like_input(self) -> None:
        """3. Verify valid WARNING-like input yields valid risk prediction."""
        res = self.engine.predict_single(self.warning_input)
        self.assertIn(res["risk_level"], ["SAFE", "CAUTION", "WARNING"])
        self.assertEqual(res["status"], "valid")

    def test_04_valid_critical_like_input(self) -> None:
        """4. Verify valid CRITICAL-like input yields CRITICAL prediction with high confidence."""
        res = self.engine.predict_single(self.critical_input)
        self.assertEqual(res["risk_level"], "CRITICAL")
        self.assertEqual(res["class_id"], 3)
        self.assertEqual(res["status"], "valid")
        self.assertGreater(res["confidence"], 0.80)
        self.assertEqual(res["recommendation"], "immediate hazard / urgent intervention")

    def test_05_missing_field(self) -> None:
        """5. Verify omitting required field raises clear InferenceValidationError."""
        bad_sample = {k: v for k, v in self.safe_input.items() if k != "distance_m"}
        with self.assertRaises(InferenceValidationError) as ctx:
            self.engine.predict_single(bad_sample)
        self.assertIn("distance_m", str(ctx.exception))

    def test_06_nan_input(self) -> None:
        """6. Verify NaN input values raise InferenceValidationError."""
        bad_sample = {**self.safe_input, "truck_speed_kmph": float("nan")}
        with self.assertRaises(InferenceValidationError) as ctx:
            self.engine.predict_single(bad_sample)
        self.assertIn("cannot be null or NaN", str(ctx.exception))

    def test_07_infinite_input(self) -> None:
        """7. Verify infinite input values raise InferenceValidationError."""
        bad_sample = {**self.safe_input, "distance_m": float("inf")}
        with self.assertRaises(InferenceValidationError) as ctx:
            self.engine.predict_single(bad_sample)
        self.assertIn("cannot be infinite", str(ctx.exception))

    def test_08_invalid_negative_physical_value(self) -> None:
        """8. Verify negative physical distance, point count, or speed raises error."""
        # Negative distance
        with self.assertRaises(InferenceValidationError) as ctx1:
            self.engine.predict_single({**self.safe_input, "distance_m": -5.0})
        self.assertIn("cannot be negative", str(ctx1.exception))

        # Negative point count
        with self.assertRaises(InferenceValidationError) as ctx2:
            self.engine.predict_single({**self.safe_input, "point_count": -10})
        self.assertIn("cannot be negative", str(ctx2.exception))

        # Negative truck speed
        with self.assertRaises(InferenceValidationError) as ctx3:
            self.engine.predict_single({**self.safe_input, "truck_speed_kmph": -20.0})
        self.assertIn("cannot be negative", str(ctx3.exception))

        # Negative bounding box dimension
        with self.assertRaises(InferenceValidationError) as ctx4:
            self.engine.predict_single({**self.safe_input, "object_width_m": -1.0})
        self.assertIn("must be positive", str(ctx4.exception))

    def test_09_unknown_object_type(self) -> None:
        """9. Verify unknown object type is handled safely via OneHotEncoder fallback."""
        novel_sample = {**self.safe_input, "object_type": "novel_drilling_drone"}
        res = self.engine.predict_single(novel_sample)
        self.assertEqual(res["status"], "valid")
        self.assertIn(res["risk_level"], RISK_CLASSES)
        self.assertAlmostEqual(sum(res["probabilities"].values()), 1.0, places=4)

    def test_10_repeated_prediction_produces_deterministic_output(self) -> None:
        """10. Verify repeated prediction produces bitwise deterministic output."""
        ref = self.engine.predict_single(self.critical_input)
        for _ in range(50):
            current = self.engine.predict_single(self.critical_input)
            self.assertEqual(current["risk_level"], ref["risk_level"])
            self.assertEqual(current["class_id"], ref["class_id"])
            self.assertEqual(current["confidence"], ref["confidence"])
            self.assertEqual(current["probabilities"], ref["probabilities"])

    def test_11_probability_sum_approximately_one(self) -> None:
        """11. Verify class probabilities sum to approximately 1.0."""
        res = self.engine.predict_single(self.critical_input)
        prob_sum = sum(res["probabilities"].values())
        self.assertAlmostEqual(prob_sum, 1.0, places=4)

    def test_12_returned_class_matches_argmax_probability(self) -> None:
        """12. Verify returned class strictly matches the key with argmax probability."""
        res = self.engine.predict_single(self.critical_input)
        argmax_key = max(res["probabilities"], key=res["probabilities"].get)
        self.assertEqual(res["risk_level"], argmax_key)
        self.assertEqual(res["confidence"], res["probabilities"][argmax_key])

    def test_13_output_labels_match_label_mapping(self) -> None:
        """13. Verify output labels and class IDs strictly match label_mapping.json."""
        with open(self.models_dir / "label_mapping.json", "r", encoding="utf-8") as f:
            mapping = json.load(f)

        res = self.engine.predict_single(self.critical_input)
        self.assertEqual(list(res["probabilities"].keys()), list(mapping["class_to_int"].keys()))
        self.assertEqual(res["class_id"], mapping["class_to_int"][res["risk_level"]])

    def test_14_preprocessing_feature_order_matches_schema(self) -> None:
        """14. Verify transformed features strictly preserve the order from feature_schema.json."""
        with open(self.models_dir / "feature_schema.json", "r", encoding="utf-8") as f:
            schema = json.load(f)
        expected_cols = schema["transformed_feature_names"]

        cleaned = self.engine.validate_observation(self.safe_input)
        df_in = pd.DataFrame([cleaned])
        X_trans = self.engine.preprocessor.transform(df_in)

        self.assertEqual(list(X_trans.columns), expected_cols)
        self.assertEqual(len(X_trans.columns), 17)

    def test_15_serialized_model_reload_produces_identical_predictions(self) -> None:
        """15. Verify serialized model reload from disk produces identical predictions."""
        fresh_engine = MineRakshakInferenceEngine()
        pred_original = self.engine.predict_single(self.critical_input)
        pred_reloaded = fresh_engine.predict_single(self.critical_input)

        self.assertEqual(pred_original["risk_level"], pred_reloaded["risk_level"])
        self.assertEqual(pred_original["class_id"], pred_reloaded["class_id"])
        self.assertEqual(pred_original["confidence"], pred_reloaded["confidence"])
        self.assertEqual(pred_original["probabilities"], pred_reloaded["probabilities"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
