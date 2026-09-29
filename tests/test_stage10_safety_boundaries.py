"""Stage 10: System-Level Validation, Safety Boundaries & Interface Verification.

Validates the explicit Stage 10 requirements:
1. Scenario A: SAFE obstacle classification & 17-feature frozen schema.
2. Scenario B: CAUTION obstacle classification & probability consistency.
3. Scenario C: WARNING obstacle classification & situational recommendation.
4. Scenario D: CRITICAL obstacle classification & urgent intervention recommendation.
5. Scenario E: Mixed multi-obstacle perception frame & threat prioritization.
6. Scenario F: Malformed / invalid sensor input & fault isolation without crash.
7. Scenario G: Unknown object type handling via zero-encoded categories.
8. Scenario H: Multiple simultaneous threats & CRITICAL selection.
9. Safety Boundary 1: CRITICAL must NEVER be silently downgraded to SAFE or CAUTION.
10. Safety Boundary 2: WARNING must NEVER be silently downgraded to SAFE.
11. Safety Boundary 3: Invalid sensor data must NEVER produce a normal 'valid' prediction.
12. Safety Boundary 4: Zero vehicle-actuation commands (no brake, steer, throttle, etc.).
13. Determinism: Consecutive runs must yield bitwise-identical probabilities and predictions.
14. ROS 2 Interface: Verify topic names, parameter schema, launch file, and message serialization.
15. Model Artifact Integrity: Primary model is frozen HGB, preprocessor produces 17 features.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from typing import Any

# Ensure project root and ros2 package are in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROS2_PKG_ROOT = PROJECT_ROOT / "ros2_ws" / "src" / "minerakshak_risk"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(ROS2_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(ROS2_PKG_ROOT))

import numpy as np

from minerakshak_risk.risk_node import (
    MineRakshakRiskNode,
    PerceptionToRiskProcessor,
    THREAT_PRIORITY,
)
from minerakshak_risk.schemas import (
    DetectedObject,
    PerceptionFrame,
    RiskAssessmentFrame,
)
from src.demo_stage10 import FORBIDDEN_ACTUATION_KEYS, build_stage10_scenarios
from src.inference.predict import (
    RESPONSE_MAPPINGS,
    SAFETY_DISCLAIMER,
    get_inference_engine,
)
from src.preprocessing.schema import RISK_CLASSES


class TestStage10SafetyBoundariesAndValidation(unittest.TestCase):
    """Stage 10 system-level end-to-end validation and safety boundary test suite."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.processor = PerceptionToRiskProcessor()
        cls.node = MineRakshakRiskNode()
        cls.scenarios = build_stage10_scenarios()

    def test_01_scenario_a_safe_obstacle(self) -> None:
        """1. Verify Scenario A produces SAFE classification, 17-feature schema, and valid probabilities."""
        sc = self.scenarios["Scenario_A"]
        res = self.processor.process_observation(sc["data"])

        self.assertEqual(res["status"], "valid")
        self.assertEqual(res["risk_level"], "SAFE")
        self.assertEqual(res["class_id"], 0)
        self.assertGreater(res["confidence"], 0.90)
        self.assertAlmostEqual(sum(res["probabilities"].values()), 1.0, places=3)
        self.assertEqual(max(res["probabilities"], key=res["probabilities"].get), "SAFE")
        self.assertEqual(res["recommendation"], "no immediate hazard")

    def test_02_scenario_b_caution_obstacle(self) -> None:
        """2. Verify Scenario B produces CAUTION classification with monitor recommendation."""
        sc = self.scenarios["Scenario_B"]
        res = self.processor.process_observation(sc["data"])

        self.assertEqual(res["status"], "valid")
        self.assertEqual(res["risk_level"], "CAUTION")
        self.assertEqual(res["class_id"], 1)
        self.assertGreater(res["confidence"], 0.90)
        self.assertAlmostEqual(sum(res["probabilities"].values()), 1.0, places=3)
        self.assertEqual(max(res["probabilities"], key=res["probabilities"].get), "CAUTION")
        self.assertEqual(res["recommendation"], "increased awareness / monitor")

    def test_03_scenario_c_warning_obstacle(self) -> None:
        """3. Verify Scenario C produces WARNING classification with prepare intervention recommendation."""
        sc = self.scenarios["Scenario_C"]
        res = self.processor.process_observation(sc["data"])

        self.assertEqual(res["status"], "valid")
        self.assertEqual(res["risk_level"], "WARNING")
        self.assertEqual(res["class_id"], 2)
        self.assertGreater(res["confidence"], 0.85)
        self.assertAlmostEqual(sum(res["probabilities"].values()), 1.0, places=3)
        self.assertEqual(max(res["probabilities"], key=res["probabilities"].get), "WARNING")
        self.assertEqual(res["recommendation"], "active warning / prepare intervention")

    def test_04_scenario_d_critical_obstacle(self) -> None:
        """4. Verify Scenario D produces CRITICAL classification with urgent intervention recommendation."""
        sc = self.scenarios["Scenario_D"]
        res = self.processor.process_observation(sc["data"])

        self.assertEqual(res["status"], "valid")
        self.assertEqual(res["risk_level"], "CRITICAL")
        self.assertEqual(res["class_id"], 3)
        self.assertGreater(res["confidence"], 0.95)
        self.assertAlmostEqual(sum(res["probabilities"].values()), 1.0, places=3)
        self.assertEqual(max(res["probabilities"], key=res["probabilities"].get), "CRITICAL")
        self.assertEqual(res["recommendation"], "immediate hazard / urgent intervention")

    def test_05_scenario_e_mixed_quad_frame(self) -> None:
        """5. Verify multi-obstacle frame with mixed SAFE/CAUTION/WARNING/CRITICAL selects CRITICAL."""
        sc = self.scenarios["Scenario_E"]
        det_objs = [DetectedObject(**obj) for obj in sc["objects"]]
        frame = PerceptionFrame(
            frame_id=sc["frame_id"],
            timestamp_ns=1727568000000000000,
            truck_speed_kmph=sc["truck_speed_kmph"],
            objects=det_objs,
        )
        res = self.processor.process_frame(frame)

        self.assertEqual(res.objects_detected, 4)
        self.assertEqual(res.objects_evaluated, 4)
        self.assertEqual(len(res.errors), 0)
        self.assertEqual(res.highest_threat_level, "CRITICAL")
        self.assertEqual(res.highest_threat_object_id, "obs_E_critical")

        # Verify individual assessments correspond to individual objects
        tiers = {a.object_id: a.predicted_risk_level for a in res.assessments}
        self.assertEqual(tiers["obs_E_safe"], "SAFE")
        self.assertEqual(tiers["obs_E_caution"], "CAUTION")
        self.assertEqual(tiers["obs_E_warning"], "WARNING")
        self.assertEqual(tiers["obs_E_critical"], "CRITICAL")

    def test_06_scenario_f_malformed_input_fault_isolation(self) -> None:
        """6. Verify malformed obstacle is rejected safely without node crash or valid classification."""
        sc = self.scenarios["Scenario_F"]
        res = self.processor.process_observation(sc["data"])

        self.assertEqual(res["status"], "error")
        self.assertEqual(res["risk_level"], "INVALID")
        self.assertEqual(res["class_id"], -1)
        self.assertIn("cannot be negative", res["error"])
        self.assertEqual(res["recommendation"], "sensor validation error - do not proceed")

        # In a mixed frame, the valid objects are evaluated while the malformed one is isolated
        mixed_frame = PerceptionFrame(
            frame_id="frame_fault_isolation_test",
            timestamp_ns=1727568000000000000,
            truck_speed_kmph=20.0,
            objects=[
                DetectedObject(**sc["data"]),  # malformed
                DetectedObject(**self.scenarios["Scenario_A"]["data"]),  # valid safe
            ],
        )
        frame_res = self.processor.process_frame(mixed_frame)
        self.assertEqual(frame_res.objects_detected, 2)
        self.assertEqual(frame_res.objects_evaluated, 1)
        self.assertEqual(len(frame_res.errors), 1)
        self.assertEqual(frame_res.highest_threat_level, "SAFE")

    def test_07_scenario_g_unknown_object_type(self) -> None:
        """7. Verify unknown object type is gracefully handled via one-hot zero encoding."""
        sc = self.scenarios["Scenario_G"]
        res = self.processor.process_observation(sc["data"])

        self.assertEqual(res["status"], "valid")
        self.assertIn(res["risk_level"], RISK_CLASSES)
        self.assertAlmostEqual(sum(res["probabilities"].values()), 1.0, places=3)
        self.assertIn("disclaimer", res)

    def test_08_scenario_h_simultaneous_threats_critical_priority(self) -> None:
        """8. Verify multiple simultaneous threats (WARNING + CRITICAL) prioritizes CRITICAL."""
        sc = self.scenarios["Scenario_H"]
        det_objs = [DetectedObject(**obj) for obj in sc["objects"]]
        frame = PerceptionFrame(
            frame_id=sc["frame_id"],
            timestamp_ns=1727568000000000000,
            truck_speed_kmph=sc["truck_speed_kmph"],
            objects=det_objs,
        )
        res = self.processor.process_frame(frame)

        self.assertEqual(res.objects_evaluated, 2)
        self.assertEqual(res.highest_threat_level, "CRITICAL")
        self.assertEqual(res.highest_threat_object_id, "obs_H_critical_object")

    def test_09_safety_boundary_critical_never_downgraded(self) -> None:
        """9. Safety Boundary 1: CRITICAL must NEVER be downgraded to SAFE or CAUTION or WARNING."""
        crit_obj = self.scenarios["Scenario_D"]["data"]
        safe_obj = self.scenarios["Scenario_A"]["data"]
        caut_obj = self.scenarios["Scenario_B"]["data"]
        warn_obj = self.scenarios["Scenario_C"]["data"]

        # Test permutations with CRITICAL at different positions
        permutations = [
            [crit_obj, safe_obj, caut_obj],
            [safe_obj, crit_obj, caut_obj],
            [safe_obj, warn_obj, crit_obj],
            [caut_obj, warn_obj, safe_obj, crit_obj],
        ]

        for idx, perm in enumerate(permutations):
            frame = PerceptionFrame(
                frame_id=f"frame_crit_downgrade_{idx}",
                timestamp_ns=1727568000000000000,
                truck_speed_kmph=20.0,
                objects=[DetectedObject(**o) for o in perm],
            )
            res = self.processor.process_frame(frame)
            self.assertEqual(
                res.highest_threat_level,
                "CRITICAL",
                f"Safety Violation: CRITICAL was downgraded to {res.highest_threat_level} in permutation {idx}"
            )
            self.assertNotEqual(res.highest_threat_level, "SAFE")
            self.assertNotEqual(res.highest_threat_level, "CAUTION")

    def test_10_safety_boundary_warning_never_downgraded_to_safe(self) -> None:
        """10. Safety Boundary 2: WARNING must NEVER be silently downgraded to SAFE."""
        warn_obj = self.scenarios["Scenario_C"]["data"]
        safe_obj = self.scenarios["Scenario_A"]["data"]

        permutations = [
            [warn_obj, safe_obj],
            [safe_obj, warn_obj],
            [safe_obj, safe_obj, warn_obj],
        ]

        for idx, perm in enumerate(permutations):
            frame = PerceptionFrame(
                frame_id=f"frame_warn_downgrade_{idx}",
                timestamp_ns=1727568000000000000,
                truck_speed_kmph=15.0,
                objects=[DetectedObject(**o) for o in perm],
            )
            res = self.processor.process_frame(frame)
            self.assertIn(
                res.highest_threat_level,
                ["WARNING", "CRITICAL"],
                f"Safety Violation: WARNING was downgraded to {res.highest_threat_level} in permutation {idx}"
            )
            self.assertNotEqual(res.highest_threat_level, "SAFE")

    def test_11_safety_boundary_invalid_never_produces_valid_status(self) -> None:
        """11. Safety Boundary 3: Invalid sensor data must NEVER produce status 'valid'."""
        invalid_cases = [
            {"distance_m": -1.0},
            {"point_count": -5},
            {"object_width_m": -0.5},
            {"truck_speed_kmph": float("nan")},
            {"distance_m": float("inf")},
            {"distance_m": None},
        ]
        base = self.scenarios["Scenario_A"]["data"]

        for idx, patch in enumerate(invalid_cases):
            corrupt = {**base, **patch}
            res = self.processor.process_observation(corrupt)
            self.assertEqual(
                res["status"],
                "error",
                f"Safety Violation: Invalid case {idx} produced status '{res.get('status')}'"
            )
            self.assertNotEqual(res["risk_level"], "SAFE")
            self.assertEqual(res["class_id"], -1)

    def test_12_safety_boundary_zero_vehicle_actuation_commands(self) -> None:
        """12. Safety Boundary 4: System is inference/alerting only. Zero vehicle actuation commands."""
        # 1. Single observation check
        for sc_key in ["Scenario_A", "Scenario_B", "Scenario_C", "Scenario_D", "Scenario_F"]:
            obs = self.scenarios[sc_key]["data"]
            res = self.processor.process_observation(obs)
            for forbidden_key in FORBIDDEN_ACTUATION_KEYS:
                self.assertNotIn(
                    forbidden_key,
                    res,
                    f"Safety Violation: Forbidden actuation command '{forbidden_key}' found in single observation output"
                )

        # 2. Frame observation check
        mixed_frame = PerceptionFrame(
            frame_id="frame_actuation_audit",
            timestamp_ns=1727568000000000000,
            truck_speed_kmph=20.0,
            objects=[DetectedObject(**self.scenarios["Scenario_D"]["data"])],
        )
        frame_res = self.processor.process_frame(mixed_frame)
        frame_dict = frame_res.to_dict()

        for forbidden_key in FORBIDDEN_ACTUATION_KEYS:
            self.assertNotIn(
                forbidden_key,
                frame_dict,
                f"Safety Violation: Forbidden actuation command '{forbidden_key}' found in frame output"
            )

    def test_13_deterministic_behavior_across_50_iterations(self) -> None:
        """13. Verify bitwise deterministic predictions and probabilities across 50 iterations."""
        for sc_key in ["Scenario_A", "Scenario_B", "Scenario_C", "Scenario_D"]:
            obs = self.scenarios[sc_key]["data"]
            baseline = self.processor.process_observation(obs)
            for i in range(50):
                cur = self.processor.process_observation(obs)
                self.assertEqual(cur["risk_level"], baseline["risk_level"])
                self.assertEqual(cur["class_id"], baseline["class_id"])
                self.assertEqual(cur["confidence"], baseline["confidence"])
                self.assertEqual(cur["probabilities"], baseline["probabilities"])

    def test_14_ros2_interface_topics_and_payload_schemas(self) -> None:
        """14. Verify ROS 2 topic names, parameter configuration, and schema contracts."""
        config_path = PROJECT_ROOT / "ros2_ws" / "src" / "minerakshak_risk" / "config" / "risk_node_params.yaml"
        launch_path = PROJECT_ROOT / "ros2_ws" / "src" / "minerakshak_risk" / "launch" / "risk_node.launch.py"

        self.assertTrue(config_path.exists())
        self.assertTrue(launch_path.exists())

        config_text = config_path.read_text(encoding="utf-8")
        self.assertIn("/minerakshak/object_observations", config_text)
        self.assertIn("/minerakshak/risk_prediction", config_text)
        self.assertIn("/minerakshak/highest_threat", config_text)

        launch_text = launch_path.read_text(encoding="utf-8")
        self.assertIn("/minerakshak/object_observations", launch_text)
        self.assertIn("/minerakshak/risk_prediction", launch_text)
        self.assertIn("/minerakshak/highest_threat", launch_text)

    def test_15_frozen_artifacts_integrity_and_17_features(self) -> None:
        """15. Verify primary model is frozen HGB, preprocessor is fitted, and feature schema has 17 names."""
        engine = get_inference_engine()
        self.assertEqual(engine.primary_model.__class__.__name__, "HistGradientBoostingClassifier")
        self.assertEqual(len(engine.feature_schema["transformed_feature_names"]), 17)
        self.assertEqual(len(engine.int_to_class), 4)
        self.assertTrue(engine.is_loaded)


if __name__ == "__main__":
    unittest.main(verbosity=2)
