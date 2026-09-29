"""Stage 9: ROS 2 Integration Test Suite for MineRakshak Perception Node.

Validates the required ROS 2 perception-to-risk behaviors:
1. Valid SAFE-like observation.
2. Valid CAUTION-like observation.
3. Valid WARNING-like observation.
4. Valid CRITICAL-like observation.
5. Missing required input.
6. NaN input.
7. Infinite input.
8. Invalid negative physical value.
9. Unknown object type.
10. Published output contains all required fields.
11. Published probabilities sum approximately to 1.
12. Published class ID matches the maximum probability.
13. Node can load the serialized model and preprocessor successfully.
14. Multiple consecutive messages produce deterministic predictions.
15. Node handles malformed input without crashing.
Supplementary:
16. Fast-path emergency threat selection prioritizes threats correctly.
17. Stage 8 test set sample #0 prediction reproduces identically.
18. Clean node resource cleanup and shutdown.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

# Ensure project root and ros2 package are in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROS2_PKG_ROOT = PROJECT_ROOT / "ros2_ws" / "src" / "minerakshak_risk"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(ROS2_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(ROS2_PKG_ROOT))

import numpy as np
import pandas as pd

from minerakshak_risk.risk_node import (
    MineRakshakRiskNode,
    PerceptionToRiskProcessor,
    ROS2_AVAILABLE,
)
from minerakshak_risk.schemas import (
    DetectedObject,
    PerceptionFrame,
    RiskAssessmentFrame,
)
from src.preprocessing.schema import RISK_CLASSES


class TestROS2PerceptionIntegration(unittest.TestCase):
    """Integration test suite for ROS 2 perception-to-risk processing."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize the processor and test sample objects."""
        cls.processor = PerceptionToRiskProcessor()
        cls.node = MineRakshakRiskNode()

        # 1. SAFE observation: distant stationary vehicle
        cls.safe_object_dict = {
            "object_id": "safe_car_001",
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

        # 2. CAUTION observation: excavator working at moderate distance
        cls.caution_object_dict = {
            "object_id": "caution_excavator_002",
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
        cls.warning_object_dict = {
            "object_id": "warning_crane_003",
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

        # 4. CRITICAL observation: close rapid approaching haul truck
        cls.critical_object_dict = {
            "object_id": "crit_truck_004",
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

    def test_01_valid_safe_like_observation(self) -> None:
        """1. Verify valid SAFE-like observation produces SAFE risk prediction and schema fields."""
        res = self.processor.process_observation(self.safe_object_dict)
        self.assertEqual(res["risk_level"], "SAFE")
        self.assertEqual(res["class_id"], 0)
        self.assertEqual(res["status"], "valid")
        self.assertGreater(res["confidence"], 0.60)
        self.assertEqual(res["recommendation"], "no immediate hazard")

    def test_02_valid_caution_like_observation(self) -> None:
        """2. Verify valid CAUTION-like observation produces low/moderate risk evaluation."""
        res = self.processor.process_observation(self.caution_object_dict)
        self.assertIn(res["risk_level"], ["SAFE", "CAUTION"])
        self.assertIn(res["class_id"], [0, 1])
        self.assertEqual(res["status"], "valid")

    def test_03_valid_warning_like_observation(self) -> None:
        """3. Verify valid WARNING-like observation produces valid risk assessment."""
        res = self.processor.process_observation(self.warning_object_dict)
        self.assertIn(res["risk_level"], ["SAFE", "CAUTION", "WARNING"])
        self.assertEqual(res["status"], "valid")

    def test_04_valid_critical_like_observation(self) -> None:
        """4. Verify valid CRITICAL-like observation produces CRITICAL output with urgent action."""
        res = self.processor.process_observation(self.critical_object_dict)
        self.assertEqual(res["risk_level"], "CRITICAL")
        self.assertEqual(res["class_id"], 3)
        self.assertEqual(res["status"], "valid")
        self.assertGreater(res["confidence"], 0.80)
        self.assertEqual(res["recommendation"], "immediate hazard / urgent intervention")

    def test_05_missing_required_input(self) -> None:
        """5. Verify omitting required field raises validation error and is handled safely."""
        bad_sample = {k: v for k, v in self.safe_object_dict.items() if k != "distance_m"}
        # Single observation path returns error status
        res = self.processor.process_observation(bad_sample)
        self.assertEqual(res["status"], "error")
        self.assertEqual(res["class_id"], -1)
        self.assertIn("distance_m", res["error"])

        # Frame path isolates the error without terminating
        frame = PerceptionFrame(
            frame_id="frame_missing_05",
            timestamp_ns=1727568000000000000,
            objects=[DetectedObject(**bad_sample), DetectedObject(**self.safe_object_dict)],
        )
        frame_res = self.processor.process_frame(frame)
        self.assertEqual(frame_res.objects_evaluated, 1)
        self.assertEqual(len(frame_res.errors), 1)

    def test_06_nan_input(self) -> None:
        """6. Verify NaN sensor values are rejected with error status."""
        bad_sample = {**self.safe_object_dict, "truck_speed_kmph": float("nan")}
        res = self.processor.process_observation(bad_sample)
        self.assertEqual(res["status"], "error")
        self.assertIn("cannot be null or NaN", res["error"])

    def test_07_infinite_input(self) -> None:
        """7. Verify infinite sensor values are rejected with error status."""
        bad_sample = {**self.safe_object_dict, "distance_m": float("inf")}
        res = self.processor.process_observation(bad_sample)
        self.assertEqual(res["status"], "error")
        self.assertIn("cannot be infinite", res["error"])

    def test_08_invalid_negative_physical_value(self) -> None:
        """8. Verify negative physical distance, speed, point count, or dimension are rejected."""
        # Negative distance
        res_neg_dist = self.processor.process_observation({**self.safe_object_dict, "distance_m": -5.0})
        self.assertEqual(res_neg_dist["status"], "error")
        self.assertIn("cannot be negative", res_neg_dist["error"])

        # Negative point count
        res_neg_pts = self.processor.process_observation({**self.safe_object_dict, "point_count": -10})
        self.assertEqual(res_neg_pts["status"], "error")
        self.assertIn("cannot be negative", res_neg_pts["error"])

        # Negative bounding box dimension
        res_neg_dim = self.processor.process_observation({**self.safe_object_dict, "object_width_m": -2.0})
        self.assertEqual(res_neg_dim["status"], "error")
        self.assertIn("must be positive", res_neg_dim["error"])

    def test_09_unknown_object_type(self) -> None:
        """9. Verify unknown object type is handled safely without crashing the node."""
        novel_sample = {**self.safe_object_dict, "object_type": "novel_autonomous_drone"}
        res = self.processor.process_observation(novel_sample)
        self.assertEqual(res["status"], "valid")
        self.assertIn(res["risk_level"], RISK_CLASSES)
        self.assertAlmostEqual(sum(res["probabilities"].values()), 1.0, places=3)

    def test_10_published_output_contains_all_required_fields(self) -> None:
        """10. Verify published output contains all required Stage 8 fields."""
        res = self.processor.process_observation(self.critical_object_dict)
        required_fields = [
            "risk_level",
            "class_id",
            "confidence",
            "probabilities",
            "model",
            "status",
            "recommendation",
        ]
        for field in required_fields:
            self.assertIn(field, res, f"Missing required output field: {field}")

        for tier in RISK_CLASSES:
            self.assertIn(tier, res["probabilities"])

    def test_11_published_probabilities_sum_approximately_one(self) -> None:
        """11. Verify predicted risk probabilities sum to approximately 1.0."""
        res = self.processor.process_observation(self.critical_object_dict)
        self.assertAlmostEqual(sum(res["probabilities"].values()), 1.0, places=3)

    def test_12_published_class_id_matches_maximum_probability(self) -> None:
        """12. Verify published class ID strictly matches the key with maximum probability."""
        res = self.processor.process_observation(self.critical_object_dict)
        argmax_key = max(res["probabilities"], key=res["probabilities"].get)
        self.assertEqual(res["risk_level"], argmax_key)
        self.assertEqual(res["confidence"], res["probabilities"][argmax_key])

    def test_13_node_loads_model_and_preprocessor_successfully(self) -> None:
        """13. Verify node loads serialized model and preprocessor successfully into memory."""
        self.assertTrue(self.processor.engine.is_loaded)
        self.assertIsNotNone(self.processor.engine.primary_model)
        self.assertIsNotNone(self.processor.engine.preprocessor)
        self.assertEqual(len(self.processor.engine.feature_schema["transformed_feature_names"]), 17)
        self.assertEqual(len(self.processor.engine.int_to_class), 4)

    def test_14_multiple_consecutive_messages_produce_deterministic_predictions(self) -> None:
        """14. Verify repeated consecutive messages produce bitwise deterministic output."""
        ref = self.processor.process_observation(self.critical_object_dict)
        for _ in range(50):
            cur = self.processor.process_observation(self.critical_object_dict)
            self.assertEqual(cur["risk_level"], ref["risk_level"])
            self.assertEqual(cur["class_id"], ref["class_id"])
            self.assertEqual(cur["confidence"], ref["confidence"])
            self.assertEqual(cur["probabilities"], ref["probabilities"])

    def test_15_node_handles_malformed_input_without_crashing(self) -> None:
        """15. Verify malformed inputs are safely isolated without terminating the node."""
        # 1. Invalid object among valid objects in a frame
        bad_obj = DetectedObject(
            object_id="bad_001",
            distance_m=-10.0,
            object_x_m=0.0,
            object_y_m=0.0,
            object_z_m=0.0,
            object_width_m=2.0,
            object_height_m=2.0,
            object_length_m=2.0,
            point_count=100,
            relative_velocity_mps=-5.0,
            object_type="truck",
        )
        frame = PerceptionFrame(
            frame_id="frame_fault_15",
            timestamp_ns=1727568000000000000,
            truck_speed_kmph=25.0,
            objects=[DetectedObject(**self.safe_object_dict), bad_obj],
        )
        res = self.processor.process_frame(frame)
        self.assertEqual(res.objects_evaluated, 1)
        self.assertEqual(len(res.errors), 1)

        # 2. Corrupt JSON string
        bad_json_res = self.processor.process_frame("corrupt_json_{{")
        self.assertIsInstance(bad_json_res, RiskAssessmentFrame)
        self.assertEqual(len(bad_json_res.errors), 1)

    def test_16_highest_threat_selection(self) -> None:
        """16. Verify fast-path threat selection prioritizes CRITICAL > WARNING > CAUTION > SAFE."""
        frame = PerceptionFrame(
            frame_id="frame_threat_16",
            timestamp_ns=1727568000000000000,
            truck_speed_kmph=28.0,
            objects=[
                DetectedObject(**self.safe_object_dict),
                DetectedObject(**self.critical_object_dict),
            ],
        )
        res = self.processor.process_frame(frame)
        self.assertEqual(res.highest_threat_level, "CRITICAL")
        self.assertEqual(res.highest_threat_object_id, "crit_truck_004")

    def test_17_stage8_known_sample_reproduction(self) -> None:
        """17. Verify Stage 8 test set sample #0 prediction reproduces identically through ROS wrapper."""
        x_test_path = self.data_dir / "X_test.csv"
        saved_preds_path = self.metrics_dir / "hgb_test_predictions.csv"

        self.assertTrue(x_test_path.exists())
        self.assertTrue(saved_preds_path.exists())

        X_test = pd.read_csv(x_test_path)
        saved_df = pd.read_csv(saved_preds_path)

        sample_0 = X_test.iloc[[0]]
        probs_0 = self.processor.engine.primary_model.predict_proba(sample_0)[0]
        pred_idx = int(np.argmax(probs_0))
        pred_label = self.processor.engine.int_to_class[pred_idx]

        expected_label = saved_df.iloc[0]["predicted_risk_level"]
        expected_conf = float(saved_df.iloc[0]["predicted_confidence"])

        self.assertEqual(pred_label, expected_label)
        self.assertAlmostEqual(float(probs_0[pred_idx]), expected_conf, places=4)

    def test_18_node_clean_shutdown(self) -> None:
        """18. Verify node resource cleanup and shutdown execute cleanly."""
        test_node = MineRakshakRiskNode(node_name="shutdown_test_node")
        self.assertTrue(test_node.destroy_node())


if __name__ == "__main__":
    unittest.main(verbosity=2)
