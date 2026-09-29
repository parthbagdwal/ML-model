"""Stage 8: Comprehensive Unit Tests for MineRakshak Inference Pipeline.

Tests:
1. Valid SAFE-like input.
2. Valid high-risk/CRITICAL-like input.
3. Unknown object_type.
4. Missing required field.
5. Invalid negative distance.
6. Invalid point count.
7. Invalid bounding-box dimensions.
8. Invalid TTC.
9. Probability vector sums to approximately 1.0.
10. Returned class is one of the four valid risk levels.
11. Confidence equals the maximum returned class probability.
12. Repeated identical input produces identical output.
13. Batch and DataFrame inference functionality.
14. Exact reproduction of known Stage 7 test set prediction.
15. Module-level convenience function predict().
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
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


class TestInferencePipeline(unittest.TestCase):
    """Test suite for MineRakshak production inference pipeline."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize the inference engine and standard test samples."""
        cls.engine = MineRakshakInferenceEngine(load_secondary=True)

        # Baseline sample (Caution-like)
        cls.raw_sample = {
            "distance_m": 25.0,
            "object_x_m": 2.0,
            "object_y_m": 24.9,
            "object_z_m": 0.5,
            "object_width_m": 3.0,
            "object_height_m": 3.2,
            "object_length_m": 6.0,
            "point_count": 420,
            "relative_velocity_mps": -2.5,
            "truck_speed_kmph": 25.0,
            "time_to_collision_s": 10.0,
            "object_type": "truck",
        }

        # Clear SAFE-like sample: distant obstacle, zero closing velocity, long TTC
        cls.safe_sample = {
            "distance_m": 75.0,
            "object_x_m": 8.0,
            "object_y_m": 74.5,
            "object_z_m": 0.0,
            "object_width_m": 2.0,
            "object_height_m": 1.6,
            "object_length_m": 4.5,
            "point_count": 85,
            "relative_velocity_mps": 0.0,
            "truck_speed_kmph": 15.0,
            "time_to_collision_s": 99.9,
            "object_type": "car",
        }

        # Clear CRITICAL-like sample: very close, high closing velocity, critical TTC
        cls.critical_sample = {
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

        cls.data_dir = PROJECT_ROOT / "data" / "processed"
        cls.metrics_dir = PROJECT_ROOT / "results" / "metrics"
        cls.raw_data_dir = PROJECT_ROOT / "data" / "raw"

    def test_01_valid_safe_like_input(self) -> None:
        """1. Verify valid SAFE-like input yields SAFE prediction with high confidence."""
        result = self.engine.predict_single(self.safe_sample)
        self.assertEqual(result["predicted_risk_level"], "SAFE")
        self.assertEqual(result["risk_code"], 0)
        self.assertGreater(result["confidence"], 0.60)
        self.assertEqual(result["recommended_action"], "no immediate hazard")
        self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0, places=4)

    def test_02_valid_high_risk_critical_like_input(self) -> None:
        """2. Verify valid high-risk/CRITICAL-like input yields CRITICAL prediction."""
        result = self.engine.predict_single(self.critical_sample)
        self.assertEqual(result["predicted_risk_level"], "CRITICAL")
        self.assertEqual(result["risk_code"], 3)
        self.assertGreater(result["confidence"], 0.80)
        self.assertEqual(result["recommended_action"], "immediate hazard / urgent intervention")
        self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0, places=4)

    def test_03_unknown_object_type(self) -> None:
        """3. Verify unknown object_type is handled gracefully via OneHotEncoder."""
        unknown_sample = {**self.raw_sample, "object_type": "novel_autonomous_drone"}
        result = self.engine.predict_single(unknown_sample)
        self.assertIn(result["predicted_risk_level"], RISK_CLASSES)
        self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0, places=4)

    def test_04_missing_required_field(self) -> None:
        """4. Verify missing required fields raise InferenceValidationError."""
        # Missing distance_m
        invalid_sample_1 = {k: v for k, v in self.raw_sample.items() if k != "distance_m"}
        with self.assertRaises(InferenceValidationError) as ctx1:
            self.engine.predict_single(invalid_sample_1)
        self.assertIn("distance_m", str(ctx1.exception))

        # Missing object_type
        invalid_sample_2 = {k: v for k, v in self.raw_sample.items() if k != "object_type"}
        with self.assertRaises(InferenceValidationError) as ctx2:
            self.engine.predict_single(invalid_sample_2)
        self.assertIn("object_type", str(ctx2.exception))

    def test_05_invalid_negative_distance(self) -> None:
        """5. Verify invalid negative distance raises InferenceValidationError."""
        invalid_sample = {**self.raw_sample, "distance_m": -12.5}
        with self.assertRaises(InferenceValidationError) as ctx:
            self.engine.predict_single(invalid_sample)
        self.assertIn("distance_m", str(ctx.exception))
        self.assertIn("cannot be negative", str(ctx.exception))

    def test_06_invalid_point_count(self) -> None:
        """6. Verify invalid negative point count raises InferenceValidationError."""
        invalid_sample = {**self.raw_sample, "point_count": -25}
        with self.assertRaises(InferenceValidationError) as ctx:
            self.engine.predict_single(invalid_sample)
        self.assertIn("point_count", str(ctx.exception))
        self.assertIn("cannot be negative", str(ctx.exception))

    def test_07_invalid_bounding_box_dimensions(self) -> None:
        """7. Verify non-positive bounding box dimensions raise InferenceValidationError."""
        # Negative width
        with self.assertRaises(InferenceValidationError):
            self.engine.predict_single({**self.raw_sample, "object_width_m": -2.0})

        # Zero height
        with self.assertRaises(InferenceValidationError):
            self.engine.predict_single({**self.raw_sample, "object_height_m": 0.0})

        # Negative length
        with self.assertRaises(InferenceValidationError):
            self.engine.predict_single({**self.raw_sample, "object_length_m": -0.5})

    def test_08_invalid_ttc(self) -> None:
        """8. Verify invalid negative TTC raises InferenceValidationError."""
        invalid_sample = {**self.raw_sample, "time_to_collision_s": -3.5}
        with self.assertRaises(InferenceValidationError) as ctx:
            self.engine.predict_single(invalid_sample)
        self.assertIn("time_to_collision_s", str(ctx.exception))

    def test_09_probability_vector_sums_to_approximately_one(self) -> None:
        """9. Verify probability vector sums to approximately 1.0 (Softmax distribution)."""
        result = self.engine.predict_single(self.raw_sample)
        probs = result["probabilities"]
        prob_sum = sum(probs.values())
        self.assertAlmostEqual(prob_sum, 1.0, places=4)

    def test_10_returned_class_is_one_of_four_valid_risk_levels(self) -> None:
        """10. Verify returned class is strictly one of SAFE, CAUTION, WARNING, CRITICAL."""
        result = self.engine.predict_single(self.raw_sample)
        self.assertIn(result["predicted_risk_level"], RISK_CLASSES)
        self.assertIn(result["risk_code"], [0, 1, 2, 3])
        self.assertEqual(list(result["probabilities"].keys()), RISK_CLASSES)

    def test_11_confidence_equals_max_returned_class_probability(self) -> None:
        """11. Verify confidence score strictly equals the maximum returned class probability."""
        result = self.engine.predict_single(self.raw_sample)
        probs = result["probabilities"]
        max_prob = max(probs.values())
        self.assertAlmostEqual(result["confidence"], max_prob, places=4)
        argmax_class = max(probs, key=probs.get)
        self.assertEqual(result["predicted_risk_level"], argmax_class)

    def test_12_repeated_identical_input_produces_identical_output(self) -> None:
        """12. Verify repeated identical input produces bitwise deterministic output."""
        first_run = self.engine.predict_single(self.raw_sample)
        for _ in range(50):
            repeated_run = self.engine.predict_single(self.raw_sample)
            self.assertEqual(repeated_run["predicted_risk_level"], first_run["predicted_risk_level"])
            self.assertEqual(repeated_run["risk_code"], first_run["risk_code"])
            self.assertEqual(repeated_run["confidence"], first_run["confidence"])
            self.assertEqual(repeated_run["probabilities"], first_run["probabilities"])

    def test_13_batch_and_dataframe_inference(self) -> None:
        """13. Verify batch inference over list of dicts and pandas DataFrame."""
        batch_list = [self.safe_sample, self.raw_sample, self.critical_sample]

        # List of dicts
        results_list = self.engine.predict_batch(batch_list)
        self.assertEqual(len(results_list), 3)
        self.assertEqual(results_list[0]["predicted_risk_level"], "SAFE")
        self.assertEqual(results_list[2]["predicted_risk_level"], "CRITICAL")

        # Pandas DataFrame
        df_batch = pd.DataFrame(batch_list)
        results_df = self.engine.predict_batch(df_batch)
        self.assertEqual(len(results_df), 3)
        self.assertEqual(
            [r["predicted_risk_level"] for r in results_list],
            [r["predicted_risk_level"] for r in results_df],
        )

    def test_14_reproduces_known_test_sample(self) -> None:
        """14. Verify exact reproduction of Stage 7 test set prediction for sample #0."""
        saved_preds_path = self.metrics_dir / "hgb_test_predictions.csv"
        x_test_path = self.data_dir / "X_test.csv"

        self.assertTrue(saved_preds_path.exists())
        self.assertTrue(x_test_path.exists())

        X_test = pd.read_csv(x_test_path)
        saved_df = pd.read_csv(saved_preds_path)

        sample_features = X_test.iloc[[0]]
        expected_label = saved_df.iloc[0]["predicted_risk_level"]
        expected_conf = float(saved_df.iloc[0]["predicted_confidence"])

        probs = self.engine.primary_model.predict_proba(sample_features)[0]
        pred_idx = int(np.argmax(probs))
        pred_label = self.engine.int_to_class[pred_idx]

        self.assertEqual(pred_label, expected_label)
        self.assertAlmostEqual(float(probs[pred_idx]), expected_conf, places=4)

    def test_15_convenience_predict_function(self) -> None:
        """15. Verify module-level convenience function predict() operates correctly."""
        res_single = predict(self.safe_sample)
        self.assertIsInstance(res_single, dict)
        self.assertEqual(res_single["predicted_risk_level"], "SAFE")

        res_batch = predict([self.safe_sample, self.critical_sample])
        self.assertIsInstance(res_batch, list)
        self.assertEqual(len(res_batch), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
