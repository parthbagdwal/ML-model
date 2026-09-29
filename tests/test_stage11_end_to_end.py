"""Stage 11: End-to-End System Integration Test Suite.

Validates the full perception-to-telemetry pipeline:
3D LiDAR / perception adapter
  -> ROS 2 /minerakshak/object_observations
  -> Frozen HGB ML inference (PerceptionToRiskProcessor)
  -> ROS 2 /minerakshak/risk_prediction + /minerakshak/highest_threat
  -> FastAPI backend endpoints (/api/minerakshak/*, /api/dashboard)
  -> Dashboard WebSocket broadcast (/ws/dashboard, /ws/minerakshak/risk).

Verifies:
1. SAFE obstacle full chain.
2. CAUTION obstacle full chain.
3. WARNING obstacle full chain.
4. CRITICAL obstacle full chain.
5. Mixed multi-obstacle frame (SAFE, CAUTION, WARNING, CRITICAL).
6. Simultaneous threats (WARNING + CRITICAL) prioritizes CRITICAL.
7. Invalid sensor data fault isolation (negative range/dimension).
8. NaN and infinite sensor data fault isolation.
9. Unknown object type handling via zero-encoding.
10. Malformed message isolation without node crash.
11. Empty perception frame handling.
12. Repeated identical frames (strict determinism).
13. Probability sum ≈ 1.0 and class_id == argmax(probability).
14. Strict zero vehicle actuation commands across all layers.
15. Backend unavailable / network resilience.
16. WebSocket client disconnect resilience.
17. Stale and duplicate timestamp handling.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from typing import Any

# Ensure project root, workspace root, and ROS 2 package are on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
ROS2_PKG_ROOT = PROJECT_ROOT / "ros2_ws" / "src" / "minerakshak_risk"

for p in [str(PROJECT_ROOT), str(WORKSPACE_ROOT), str(ROS2_PKG_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from fastapi.testclient import TestClient
import main as fastapi_app_module

from src.bridge.ros2_fastapi_bridge import MineRakshakSystemBridge, get_system_bridge
from src.demo_stage10 import FORBIDDEN_ACTUATION_KEYS
from src.inference.predict import get_inference_engine
from src.perception.mock_lidar_adapter import MockLiDARPerceptionAdapter
from minerakshak_risk.schemas import DetectedObject, PerceptionFrame, RiskAssessmentFrame


class TestStage11EndToEndIntegration(unittest.TestCase):
    """Stage 11 Full-System End-to-End Integration Test Suite."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(fastapi_app_module.app)
        cls.adapter = MockLiDARPerceptionAdapter()
        cls.bridge = MineRakshakSystemBridge(use_in_process_fastapi=True)

    def test_01_full_chain_safe_obstacle(self) -> None:
        """1. Verify SAFE obstacle flows end-to-end from perception to FastAPI and WebSocket."""
        obj = self.adapter.create_safe_obstacle("safe_ped_01")
        frame = self.adapter.create_perception_frame([obj], frame_id="frame_stg11_safe")

        # Process through bridge
        res = self.bridge.process_perception_to_dashboard(frame)

        self.assertEqual(res["frame_id"], "frame_stg11_safe")
        self.assertEqual(res["highest_threat_level"], "SAFE")
        self.assertEqual(res["objects_evaluated"], 1)
        self.assertEqual(len(res["assessments"]), 1)
        self.assertEqual(res["assessments"][0]["risk_level"], "SAFE")
        self.assertGreater(res["assessments"][0]["confidence"], 0.90)

        # Verify FastAPI endpoint reflects latest risk
        api_res = self.client.get("/api/minerakshak/latest_risk")
        self.assertEqual(api_res.status_code, 200)
        latest_data = api_res.json()
        self.assertEqual(latest_data["frame_id"], "frame_stg11_safe")
        self.assertEqual(latest_data["highest_threat_level"], "SAFE")

        # Verify dashboard state has updated minerakshak_risk
        dash_res = self.client.get("/api/dashboard")
        self.assertEqual(dash_res.status_code, 200)
        self.assertIn("minerakshak_risk", dash_res.json())
        self.assertEqual(dash_res.json()["minerakshak_risk"]["highest_threat_level"], "SAFE")

    def test_02_full_chain_caution_obstacle(self) -> None:
        """2. Verify CAUTION obstacle flows end-to-end with monitor recommendation."""
        obj = self.adapter.create_caution_obstacle("caut_car_02")
        frame = self.adapter.create_perception_frame([obj], frame_id="frame_stg11_caution")

        res = self.bridge.process_perception_to_dashboard(frame)

        self.assertEqual(res["highest_threat_level"], "CAUTION")
        self.assertEqual(res["assessments"][0]["risk_level"], "CAUTION")
        self.assertEqual(res["assessments"][0]["class_id"], 1)
        self.assertEqual(res["assessments"][0]["recommendation"], "increased awareness / monitor")

        # Verify FastAPI endpoint
        api_res = self.client.get("/api/minerakshak/latest_risk")
        self.assertEqual(api_res.json()["highest_threat_level"], "CAUTION")

    def test_03_full_chain_warning_obstacle(self) -> None:
        """3. Verify WARNING obstacle flows end-to-end with prepare intervention recommendation."""
        obj = self.adapter.create_warning_obstacle("warn_worker_03")
        frame = self.adapter.create_perception_frame([obj], frame_id="frame_stg11_warning")

        res = self.bridge.process_perception_to_dashboard(frame)

        self.assertEqual(res["highest_threat_level"], "WARNING")
        self.assertEqual(res["assessments"][0]["risk_level"], "WARNING")
        self.assertEqual(res["assessments"][0]["class_id"], 2)
        self.assertEqual(res["assessments"][0]["recommendation"], "active warning / prepare intervention")

        api_res = self.client.get("/api/minerakshak/latest_risk")
        self.assertEqual(api_res.json()["highest_threat_level"], "WARNING")

    def test_04_full_chain_critical_obstacle(self) -> None:
        """4. Verify CRITICAL obstacle flows end-to-end with urgent intervention recommendation."""
        obj = self.adapter.create_critical_obstacle("crit_truck_04")
        frame = self.adapter.create_perception_frame([obj], frame_id="frame_stg11_critical")

        res = self.bridge.process_perception_to_dashboard(frame)

        self.assertEqual(res["highest_threat_level"], "CRITICAL")
        self.assertEqual(res["highest_threat_object_id"], "crit_truck_04")
        self.assertEqual(res["assessments"][0]["risk_level"], "CRITICAL")
        self.assertEqual(res["assessments"][0]["class_id"], 3)
        self.assertGreater(res["assessments"][0]["confidence"], 0.95)
        self.assertEqual(res["assessments"][0]["recommendation"], "immediate hazard / urgent intervention")

        api_res = self.client.get("/api/minerakshak/latest_risk")
        self.assertEqual(api_res.json()["highest_threat_level"], "CRITICAL")
        self.assertEqual(api_res.json()["highest_threat_object_id"], "crit_truck_04")

    def test_05_full_chain_mixed_quad_frame(self) -> None:
        """5. Verify mixed frame containing SAFE, CAUTION, WARNING, CRITICAL prioritizes CRITICAL."""
        frame = self.adapter.create_mixed_threat_frame(frame_id="frame_stg11_quad")
        res = self.bridge.process_perception_to_dashboard(frame)

        self.assertEqual(res["objects_detected"], 4)
        self.assertEqual(res["objects_evaluated"], 4)
        self.assertEqual(res["highest_threat_level"], "CRITICAL")
        self.assertEqual(res["highest_threat_object_id"], "obj_critical_04")

        # Verify individual object classifications
        assm_map = {a["object_id"]: a["risk_level"] for a in res["assessments"]}
        self.assertEqual(assm_map["obj_safe_01"], "SAFE")
        self.assertEqual(assm_map["obj_caution_02"], "CAUTION")
        self.assertEqual(assm_map["obj_warning_03"], "WARNING")
        self.assertEqual(assm_map["obj_critical_04"], "CRITICAL")

    def test_06_simultaneous_warning_and_critical(self) -> None:
        """6. Verify simultaneous WARNING + CRITICAL prioritizes CRITICAL and never downgrades."""
        warn_obj = self.adapter.create_warning_obstacle("sim_warn_01")
        crit_obj = self.adapter.create_critical_obstacle("sim_crit_02")
        frame = self.adapter.create_perception_frame([warn_obj, crit_obj], frame_id="frame_stg11_dual_threat")

        res = self.bridge.process_perception_to_dashboard(frame)

        self.assertEqual(res["objects_evaluated"], 2)
        self.assertEqual(res["highest_threat_level"], "CRITICAL")
        self.assertEqual(res["highest_threat_object_id"], "sim_crit_02")

    def test_07_invalid_sensor_data_fault_isolation(self) -> None:
        """7. Verify invalid sensor data (negative distance) is safely isolated without crashing."""
        bad_obj = self.adapter.create_invalid_obstacle("negative_distance", "bad_range_01")
        safe_obj = self.adapter.create_safe_obstacle("good_safe_02")
        frame = self.adapter.create_perception_frame([bad_obj, safe_obj], frame_id="frame_stg11_fault_range")

        res = self.bridge.process_perception_to_dashboard(frame)

        self.assertEqual(res["objects_detected"], 2)
        self.assertEqual(res["objects_evaluated"], 1)  # Only valid object evaluated
        self.assertEqual(len(res["errors"]), 1)
        self.assertEqual(res["errors"][0]["object_id"], "bad_range_01")
        self.assertIn("cannot be negative", res["errors"][0]["error"])
        self.assertEqual(res["highest_threat_level"], "SAFE")

        # Verify FastAPI exposed errors
        api_res = self.client.get("/api/minerakshak/latest_risk").json()
        self.assertEqual(len(api_res["errors"]), 1)

    def test_08_nan_and_infinite_sensor_data(self) -> None:
        """8. Verify NaN velocity and infinite distance inputs are trapped with error status."""
        nan_obj = self.adapter.create_invalid_obstacle("nan_velocity", "nan_obj_01")
        inf_obj = self.adapter.create_invalid_obstacle("inf_distance", "inf_obj_02")
        frame = self.adapter.create_perception_frame([nan_obj, inf_obj], frame_id="frame_stg11_nan_inf")

        res = self.bridge.process_perception_to_dashboard(frame)

        self.assertEqual(res["objects_detected"], 2)
        self.assertEqual(res["objects_evaluated"], 0)
        self.assertEqual(len(res["errors"]), 2)

    def test_09_unknown_object_type_graceful_fallback(self) -> None:
        """9. Verify unmodeled categorical label defaults gracefully via zero-encoding without crash."""
        novel_obj = self.adapter.create_unknown_type_obstacle("novel_drone_01")
        frame = self.adapter.create_perception_frame([novel_obj], frame_id="frame_stg11_novel")

        res = self.bridge.process_perception_to_dashboard(frame)

        self.assertEqual(res["objects_evaluated"], 1)
        self.assertEqual(res["assessments"][0]["status"], "valid")
        self.assertIn(res["assessments"][0]["risk_level"], ["SAFE", "CAUTION", "WARNING", "CRITICAL"])
        self.assertAlmostEqual(sum(res["assessments"][0]["probabilities"].values()), 1.0, places=3)

    def test_10_malformed_raw_json_message(self) -> None:
        """10. Verify corrupt JSON payload is safely trapped without terminating the node or backend."""
        corrupt_payload = "{corrupt_perception_json: true, unexpected_eof..."
        res = self.bridge.process_perception_to_dashboard(corrupt_payload)

        self.assertEqual(res["frame_id"], "invalid_frame")
        self.assertEqual(res["objects_evaluated"], 0)
        self.assertGreater(len(res["errors"]), 0)

    def test_11_empty_perception_frame(self) -> None:
        """11. Verify perception frame with 0 objects evaluates safely with 0 assessments."""
        empty_frame = self.adapter.create_perception_frame([], frame_id="frame_stg11_empty")
        res = self.bridge.process_perception_to_dashboard(empty_frame)

        self.assertEqual(res["objects_detected"], 0)
        self.assertEqual(res["objects_evaluated"], 0)
        self.assertEqual(res["highest_threat_level"], "SAFE")
        self.assertIsNone(res["highest_threat_object_id"])

    def test_12_repeated_identical_frames_determinism(self) -> None:
        """12. Verify repeated consecutive identical frames produce strictly identical predictions."""
        frame = self.adapter.create_mixed_threat_frame(frame_id="frame_stg11_deterministic")
        base = self.bridge.process_perception_to_dashboard(frame)

        for _ in range(25):
            cur = self.bridge.process_perception_to_dashboard(frame)
            self.assertEqual(cur["highest_threat_level"], base["highest_threat_level"])
            self.assertEqual(cur["highest_threat_object_id"], base["highest_threat_object_id"])
            for a_cur, a_base in zip(cur["assessments"], base["assessments"]):
                self.assertEqual(a_cur["risk_level"], a_base["risk_level"])
                self.assertEqual(a_cur["confidence"], a_base["confidence"])
                self.assertEqual(a_cur["probabilities"], a_base["probabilities"])

    def test_13_probability_sum_and_argmax_consistency(self) -> None:
        """13. Verify four-class probabilities sum to ~1.0 and class_id matches argmax(prob)."""
        frame = self.adapter.create_mixed_threat_frame(frame_id="frame_stg11_probs")
        res = self.bridge.process_perception_to_dashboard(frame)

        for a in res["assessments"]:
            probs = a["probabilities"]
            self.assertAlmostEqual(sum(probs.values()), 1.0, places=3)
            argmax_tier = max(probs, key=probs.get)
            self.assertEqual(a["risk_level"], argmax_tier)
            self.assertEqual(a["confidence"], round(probs[argmax_tier], 4))

    def test_14_zero_vehicle_actuation_invariants(self) -> None:
        """14. Strict safety invariant: Zero vehicle actuation commands across all layers."""
        frame = self.adapter.create_mixed_threat_frame(frame_id="frame_stg11_actuation_audit")
        res = self.bridge.process_perception_to_dashboard(frame)

        # Check bridge output
        for forbidden in FORBIDDEN_ACTUATION_KEYS:
            self.assertNotIn(forbidden, res)
            for a in res["assessments"]:
                self.assertNotIn(forbidden, a)

        # Check FastAPI state
        api_data = self.client.get("/api/minerakshak/latest_risk").json()
        for forbidden in FORBIDDEN_ACTUATION_KEYS:
            self.assertNotIn(forbidden, api_data)

    def test_15_backend_unavailable_resilience(self) -> None:
        """15. Verify bridge remains fully operational even if remote backend HTTP endpoint fails."""
        remote_bridge = MineRakshakSystemBridge(
            fastapi_base_url="http://127.0.0.1:9999",  # Non-existent port
            use_in_process_fastapi=False,
        )
        frame = self.adapter.create_safe_obstacle("resilience_obj_01")
        # Process frame should complete ML evaluation and trap HTTP error gracefully
        pframe = self.adapter.create_perception_frame([frame], frame_id="frame_stg11_resilience")
        res = remote_bridge.process_perception_to_dashboard(pframe)

        self.assertEqual(res["objects_evaluated"], 1)
        self.assertEqual(res["highest_threat_level"], "SAFE")
        self.assertIn("error", res["fastapi_forward_status"].lower())

    def test_16_websocket_client_disconnect_resilience(self) -> None:
        """16. Verify WebSocket connection, message reception, and clean disconnect handling."""
        # Connect to /ws/minerakshak/risk
        with self.client.websocket_connect("/ws/minerakshak/risk") as ws:
            init_msg = ws.receive_json()
            self.assertEqual(init_msg["type"], "risk_assessment_frame")
            self.assertIn("highest_threat_level", init_msg["data"])

        # Connect to /ws/dashboard
        with self.client.websocket_connect("/ws/dashboard") as ws:
            dash_msg = ws.receive_json()
            self.assertEqual(dash_msg["type"], "dashboard_update")
            self.assertIn("minerakshak_risk", dash_msg["data"])

    def test_17_stale_and_duplicate_timestamp_handling(self) -> None:
        """17. Verify frames with duplicate and stale timestamps are ingested without failure."""
        t_stale = 1600000000000000000  # Stale timestamp in nanoseconds
        frame1 = PerceptionFrame(
            frame_id="frame_dup_001",
            timestamp_ns=t_stale,
            objects=[self.adapter.create_safe_obstacle()],
        )
        frame2 = PerceptionFrame(
            frame_id="frame_dup_002",
            timestamp_ns=t_stale,  # Duplicate timestamp
            objects=[self.adapter.create_critical_obstacle()],
        )

        res1 = self.bridge.process_perception_to_dashboard(frame1)
        res2 = self.bridge.process_perception_to_dashboard(frame2)

        self.assertEqual(res1["highest_threat_level"], "SAFE")
        self.assertEqual(res2["highest_threat_level"], "CRITICAL")


if __name__ == "__main__":
    unittest.main(verbosity=2)
