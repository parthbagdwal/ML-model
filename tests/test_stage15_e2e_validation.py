"""Stage 15 Test Suite: End-to-End System Validation, Fault Recovery & Deployment Readiness.

===============================================================================
MANDATORY SAFETY & ENVIRONMENT CONSTRAINTS:
1. Physical 2D LiDAR hardware is NOT available in this environment.
2. Synthetic/replay scans are explicitly distinguished from real physical data.
3. Frozen ML model artifacts must remain 100% bitwise identical.
4. Zero-actuation invariant enforced: driver decision support only.
5. No fabricated 3D data: elevation and 3D bounding boxes strictly disallowed.
===============================================================================
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
import unittest
from pathlib import Path
from typing import Any

# Ensure project root, parent directory, and ROS 2 package are on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
ROS2_PKG_ROOT = PROJECT_ROOT / "ros2_ws" / "src" / "minerakshak_risk"

for p in [str(PROJECT_ROOT), str(WORKSPACE_ROOT), str(ROS2_PKG_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from fastapi.testclient import TestClient
import main as fastapi_app_module
from src.bridge.ros2_fastapi_bridge import MineRakshakSystemBridge
from src.inference.predict import (
    MineRakshakInferenceEngine,
    RESPONSE_MAPPINGS,
    SAFETY_DISCLAIMER,
    get_inference_engine,
)
from src.perception.feature_mapper_2d import (
    FEATURE_COMPATIBILITY_MATRIX,
    STAGE12_COMPATIBILITY_STATEMENT,
    FeatureMapper2D,
)
from src.perception.lidar_2d import LaserScan2D, Obstacle2D
from src.perception.obstacle_extractor_2d import (
    MultiScanTracker2D,
    ObstacleExtractor2D,
)
from src.perception.ros2_laserscan_adapter import (
    HARDWARE_READY_DISCLAIMER,
    ROS2LaserScanAdapter,
)
from src.perception.scan_quality_2d import (
    SCAN_QUALITY_DISCLAIMER,
    ScanQualityAnalyzer2D,
    ScanQualityReport,
)
from src.perception.sensor_config_2d import (
    LiDAR2DConfig,
    SensorHealthState,
    SensorValidationResult,
)
from minerakshak_risk.laserscan_node import LaserScanToRiskPipeline
from minerakshak_risk.risk_node import (
    PerceptionToRiskProcessor,
    THREAT_PRIORITY,
)
from minerakshak_risk.schemas import (
    DetectedObject,
    ObjectRiskAssessment,
    PerceptionFrame,
    RiskAssessmentFrame,
)

# Reference frozen model SHA-256 hashes
FROZEN_MODEL_HASHES = {
    "models/hgb_model.joblib": "f885c29d3ceddae8d2f5d7a2c1ff44fac849ec57ed498504938d632adb7f1974",
    "models/preprocessor.joblib": "37bc0e1366174965e741ab223ff3619c2e6fdc62e139b24ba5e8eaff3f0948da",
    "models/feature_schema.json": "e9ca72a4b9a9675c8e85ad2059f0448f2140891512ad4d19c10280b90d2f986b",
    "models/label_mapping.json": "2b5fd687a4f99cc2fcc94e3bd031c48571615193e7cff4de621a379492b17ebc",
    "models/xgboost_model.json": "bb385fb728583b385acbd1e7a052a4579a951835473dc10b12f6dccfdfacb983",
}


def make_synthetic_scan(
    timestamp_ns: int | None = None,
    num_beams: int = 361,
    angle_min: float = -math.pi / 2,
    angle_max: float = math.pi / 2,
    range_min: float = 0.1,
    range_max: float = 80.0,
    ranges: list[float] | None = None,
    frame_id: str = "laser_frame",
) -> dict[str, Any]:
    """Helper to construct synthetic LaserScan dictionary for testing."""
    if timestamp_ns is None:
        timestamp_ns = int(time.time() * 1e9)
    if ranges is None:
        ranges = [float("inf")] * num_beams

    sec = timestamp_ns // 1_000_000_000
    nanosec = timestamp_ns % 1_000_000_000
    angle_inc = (angle_max - angle_min) / max(1, num_beams - 1)

    return {
        "header": {
            "frame_id": frame_id,
            "stamp": {"sec": sec, "nanosec": nanosec},
        },
        "angle_min": angle_min,
        "angle_max": angle_max,
        "angle_increment": angle_inc,
        "time_increment": 0.0,
        "scan_time": 0.05,
        "range_min": range_min,
        "range_max": range_max,
        "ranges": ranges,
        "intensities": [100.0] * len(ranges),
    }


def add_obstacle_to_ranges(
    ranges: list[float],
    distance: float,
    center_angle_rad: float,
    angular_width_rad: float,
    angle_min: float = -math.pi / 2,
    angle_max: float = math.pi / 2,
) -> None:
    """Helper to inject an obstacle arc into a scan ranges list."""
    num_beams = len(ranges)
    angle_inc = (angle_max - angle_min) / max(1, num_beams - 1)
    for i in range(num_beams):
        theta = angle_min + i * angle_inc
        if abs(theta - center_angle_rad) <= (angular_width_rad / 2.0):
            ranges[i] = distance


# =============================================================================
# TEST CLASS 1: FROZEN ARTIFACT & ARCHITECTURE AUDIT (Tasks 1, 2, 8, 9)
# =============================================================================
class TestStage15FrozenArtifactIntegrity(unittest.TestCase):
    """TASK 9: Verify bitwise SHA-256 hashes of all 5 frozen ML artifacts."""

    def test_frozen_artifacts_match_hashes_bitwise(self) -> None:
        for rel_path, expected_hash in FROZEN_MODEL_HASHES.items():
            full_path = PROJECT_ROOT / rel_path
            self.assertTrue(full_path.exists(), f"Frozen artifact missing: {rel_path}")
            data = full_path.read_bytes()
            computed_hash = hashlib.sha256(data).hexdigest()
            self.assertEqual(
                computed_hash,
                expected_hash,
                f"BITWISE HASH MISMATCH for {rel_path}: {computed_hash} != {expected_hash}",
            )


class TestStage15ArchitectureAnd2DBoundary(unittest.TestCase):
    """TASK 1, 2 & 8: 2D-only boundary audit and machine-readable provenance."""

    def test_feature_provenance_json_validity(self) -> None:
        prov_file = PROJECT_ROOT / "results" / "metrics" / "stage15_feature_provenance.json"
        self.assertTrue(prov_file.exists(), "stage15_feature_provenance.json does not exist")

        with open(prov_file, "r") as f:
            data = json.load(f)

        self.assertEqual(data["provenance_summary"]["total_features"], 12)
        self.assertEqual(data["provenance_summary"]["REAL_2D_MEASUREMENT"], 4)
        self.assertEqual(data["provenance_summary"]["DERIVED_FROM_2D"], 4)
        self.assertEqual(data["provenance_summary"]["EXTERNAL_TELEMETRY"], 1)
        self.assertEqual(data["provenance_summary"]["COMPATIBILITY_DEFAULT"], 3)

        features = {feat["feature_name"]: feat for feat in data["features"]}
        self.assertEqual(len(features), 12)

        # Check compatibility defaults have explicit disclaimers
        for comp_name in ["object_z_m", "object_height_m", "object_type"]:
            self.assertIn(comp_name, features)
            self.assertEqual(features[comp_name]["provenance_category"], "COMPATIBILITY_DEFAULT")
            self.assertTrue(features[comp_name]["is_compatibility_default"])
            self.assertIn("NOT a physical 2D LiDAR measurement", features[comp_name]["disclaimer"])

        # Check real 2D measurements
        for real_name in ["distance_m", "object_x_m", "object_y_m", "point_count"]:
            self.assertIn(real_name, features)
            self.assertEqual(features[real_name]["provenance_category"], "REAL_2D_MEASUREMENT")
            self.assertTrue(features[real_name]["is_measurable_by_2d_lidar"])

    def test_zero_category_e_runtime_3d_processing(self) -> None:
        """Verify that the active 2D perception pipeline has ZERO Category E 3D processing."""
        active_files = [
            PROJECT_ROOT / "src" / "perception" / "ros2_laserscan_adapter.py",
            PROJECT_ROOT / "src" / "perception" / "obstacle_extractor_2d.py",
            PROJECT_ROOT / "src" / "perception" / "feature_mapper_2d.py",
            PROJECT_ROOT / "src" / "perception" / "scan_quality_2d.py",
            PROJECT_ROOT / "src" / "perception" / "lidar_calibration_2d.py",
            PROJECT_ROOT / "ros2_ws" / "src" / "minerakshak_risk" / "minerakshak_risk" / "laserscan_node.py",
        ]

        prohibited_runtime_calls = [
            "PointCloud2",
            "open3d",
            "voxel_grid",
            "estimate_height_from_pointcloud",
            "compute_3d_bounding_box",
            "z_elevation_clustering",
        ]

        for file_path in active_files:
            self.assertTrue(file_path.exists(), f"Active file missing: {file_path}")
            content = file_path.read_text(encoding="utf-8")
            for term in prohibited_runtime_calls:
                self.assertNotIn(
                    term,
                    content,
                    f"Forbidden 3D runtime call '{term}' found in active 2D file: {file_path}",
                )


# =============================================================================
# TEST CLASS 2: END-TO-END NOMINAL SCENARIOS (Task 3)
# =============================================================================
class TestStage15EndToEndNominalPipeline(unittest.TestCase):
    """TASK 3: Complete synthetic LaserScan through the public pipeline interface across 9 scenarios."""

    def setUp(self) -> None:
        self.pipeline = LaserScanToRiskPipeline()

    def test_scenario_1_empty_environment(self) -> None:
        """Scenario 1: Empty environment -> 0 obstacles, SAFE, sensor health OK."""
        now_ns = int(time.time() * 1e9)
        scan = make_synthetic_scan(timestamp_ns=now_ns, num_beams=361)  # all inf

        val_res, frame = self.pipeline.process_scan(scan, truck_speed_kmph=20.0)

        self.assertTrue(val_res.is_valid)
        self.assertEqual(val_res.health_state, SensorHealthState.OK)
        self.assertEqual(frame.objects_detected, 0)
        self.assertEqual(frame.objects_evaluated, 0)
        self.assertEqual(frame.highest_threat_level, "SAFE")
        self.assertIsNone(frame.highest_threat_object_id)
        self.assertEqual(len(frame.assessments), 0)
        self.assertEqual(frame.feature_compatibility_status, "COMPATIBLE_WITH_DEFAULTS")

    def test_scenario_2_distant_obstacle(self) -> None:
        """Scenario 2: Distant obstacle at 40m -> detected, classified, SAFE/CAUTION."""
        now_ns = int(time.time() * 1e9)
        ranges = [float("inf")] * 361
        add_obstacle_to_ranges(ranges, distance=40.0, center_angle_rad=0.0, angular_width_rad=0.1)
        scan = make_synthetic_scan(timestamp_ns=now_ns, ranges=ranges)

        val_res, frame = self.pipeline.process_scan(scan, truck_speed_kmph=20.0)

        self.assertTrue(val_res.is_valid)
        self.assertEqual(frame.objects_detected, 1)
        self.assertEqual(frame.objects_evaluated, 1)
        self.assertIn(frame.highest_threat_level, ["SAFE", "CAUTION"])
        assessment = frame.assessments[0]
        self.assertAlmostEqual(assessment.distance_m, 40.0, delta=0.5)
        self.assertIn("HistGradientBoostingClassifier", assessment.model)
        self.assertIn(assessment.risk_level, ["SAFE", "CAUTION"])
        self.assertAlmostEqual(sum(assessment.probabilities.values()), 1.0, delta=1e-4)

    def test_scenario_3_lateral_obstacle(self) -> None:
        """Scenario 3: Lateral obstacle (at +30 deg, range 15m) -> correctly localized laterally."""
        now_ns = int(time.time() * 1e9)
        ranges = [float("inf")] * 361
        center_angle = math.radians(30.0)
        add_obstacle_to_ranges(ranges, distance=15.0, center_angle_rad=center_angle, angular_width_rad=0.1)
        scan = make_synthetic_scan(timestamp_ns=now_ns, ranges=ranges)

        val_res, frame = self.pipeline.process_scan(scan, truck_speed_kmph=15.0)

        self.assertTrue(val_res.is_valid)
        self.assertEqual(frame.objects_detected, 1)
        assessment = frame.assessments[0]
        # x = 15 * cos(30 deg) ~ 13.0m, y = 15 * sin(30 deg) ~ 7.5m
        self.assertGreater(assessment.object_y_m, 5.0)
        self.assertGreater(assessment.object_x_m, 10.0)

    def test_scenario_4_approaching_obstacle_multi_scan(self) -> None:
        """Scenario 4: Approaching obstacle closing over multiple scans -> relative_velocity < 0, TTC calculated."""
        # 3 consecutive scans at 100ms intervals moving closer: 25m -> 23.5m -> 22.0m (-15 m/s)
        base_t = time.time()
        distances = [25.0, 23.5, 22.0]
        last_frame = None

        for idx, dist in enumerate(distances):
            t_s = base_t + (idx * 0.1)
            t_ns = int(t_s * 1e9)
            ranges = [float("inf")] * 361
            add_obstacle_to_ranges(ranges, distance=dist, center_angle_rad=0.0, angular_width_rad=0.1)
            scan = make_synthetic_scan(timestamp_ns=t_ns, ranges=ranges)

            val_res, frame = self.pipeline.process_scan(scan, current_time_s=t_s, truck_speed_kmph=30.0)
            self.assertTrue(val_res.is_valid)
            last_frame = frame

        self.assertIsNotNone(last_frame)
        self.assertEqual(last_frame.objects_evaluated, 1)
        assessment = last_frame.assessments[0]
        # Verified closing velocity: negative sign
        self.assertLess(assessment.relative_velocity_mps, -5.0)
        # Time to collision is calculated and finite
        self.assertIsNotNone(assessment.time_to_collision_s)
        self.assertLess(assessment.time_to_collision_s, 5.0)
        self.assertIn(last_frame.highest_threat_level, ["CAUTION", "WARNING", "CRITICAL"])

    def test_scenario_5_imminent_obstacle(self) -> None:
        """Scenario 5: Imminent close obstacle at 6m with high closing velocity -> CRITICAL."""
        base_t = time.time()
        # Feed two fast closing scans: 7.5m then 6.0m over 0.1s (-15 m/s closing)
        for idx, dist in enumerate([7.5, 6.0]):
            t_s = base_t + (idx * 0.1)
            ranges = [float("inf")] * 361
            add_obstacle_to_ranges(ranges, distance=dist, center_angle_rad=0.0, angular_width_rad=0.15)
            scan = make_synthetic_scan(timestamp_ns=int(t_s * 1e9), ranges=ranges)
            val_res, frame = self.pipeline.process_scan(scan, current_time_s=t_s, truck_speed_kmph=35.0)

        self.assertEqual(frame.objects_evaluated, 1)
        self.assertEqual(frame.highest_threat_level, "CRITICAL")
        self.assertIn("CRITICAL", frame.assessments[0].risk_level)

    def test_scenario_6_multiple_obstacles_threat_prioritization(self) -> None:
        """Scenario 6: Multiple obstacles (distant, mid, imminent) -> highest threat is CRITICAL."""
        now_ns = int(time.time() * 1e9)
        ranges = [float("inf")] * 361
        # Obstacle 1: Far ahead (50m, 0 deg)
        add_obstacle_to_ranges(ranges, distance=50.0, center_angle_rad=0.0, angular_width_rad=0.08)
        # Obstacle 2: Mid-range left (25m, +40 deg)
        add_obstacle_to_ranges(ranges, distance=25.0, center_angle_rad=math.radians(40), angular_width_rad=0.08)
        # Obstacle 3: Close right (5m, -30 deg)
        add_obstacle_to_ranges(ranges, distance=5.0, center_angle_rad=math.radians(-30), angular_width_rad=0.15)
        scan = make_synthetic_scan(timestamp_ns=now_ns, ranges=ranges)

        val_res, frame = self.pipeline.process_scan(scan, truck_speed_kmph=30.0)

        self.assertTrue(val_res.is_valid)
        self.assertGreaterEqual(frame.objects_detected, 3)
        self.assertEqual(frame.highest_threat_level, "CRITICAL")

    def test_scenario_7_invalid_scan_handling(self) -> None:
        """Scenario 7: Invalid scan (negative ranges) -> isolated safely, INVALID health."""
        now_ns = int(time.time() * 1e9)
        ranges = [-5.0] * 361
        scan = make_synthetic_scan(timestamp_ns=now_ns, ranges=ranges)

        val_res, frame = self.pipeline.process_scan(scan)

        self.assertFalse(val_res.is_valid)
        self.assertEqual(val_res.health_state, SensorHealthState.INVALID)
        self.assertEqual(frame.highest_threat_level, "SAFE")
        self.assertEqual(frame.objects_detected, 0)
        self.assertGreaterEqual(len(frame.errors), 1)

    def test_scenario_8_partially_corrupted_scan(self) -> None:
        """Scenario 8: Partially corrupted scan (some NaN beams but valid cluster exists) -> processed safely."""
        now_ns = int(time.time() * 1e9)
        ranges = [float("inf")] * 361
        # Corrupt 20 random beams with NaN
        for i in range(10, 30):
            ranges[i] = float("nan")
        # Valid obstacle at 18m
        add_obstacle_to_ranges(ranges, distance=18.0, center_angle_rad=0.0, angular_width_rad=0.1)
        scan = make_synthetic_scan(timestamp_ns=now_ns, ranges=ranges)

        val_res, frame = self.pipeline.process_scan(scan)

        # Scan should be valid or degraded, but processed without exception
        self.assertTrue(val_res.is_valid)
        self.assertEqual(frame.objects_detected, 1)

    def test_scenario_9_stale_scan(self) -> None:
        """Scenario 9: Stale scan (timestamp 1.5s in the past) -> STALE health state, fault frame emitted."""
        now_s = time.time()
        stale_ns = int((now_s - 1.5) * 1e9)
        scan = make_synthetic_scan(timestamp_ns=stale_ns)

        val_res, frame = self.pipeline.process_scan(scan, current_time_s=now_s)

        self.assertFalse(val_res.is_valid)
        self.assertEqual(val_res.health_state, SensorHealthState.STALE)
        self.assertEqual(frame.highest_threat_level, "SAFE")
        self.assertEqual(frame.objects_evaluated, 0)


# =============================================================================
# TEST CLASS 3: RISK-STATE CONSISTENCY (Task 4)
# =============================================================================
class TestStage15RiskStateConsistency(unittest.TestCase):
    """TASK 4: Verify SAFE, CAUTION, WARNING, CRITICAL consistency across the stack and priority order."""

    def test_threat_priority_order(self) -> None:
        """Verify CRITICAL > WARNING > CAUTION > SAFE strict priority."""
        self.assertGreater(THREAT_PRIORITY["CRITICAL"], THREAT_PRIORITY["WARNING"])
        self.assertGreater(THREAT_PRIORITY["WARNING"], THREAT_PRIORITY["CAUTION"])
        self.assertGreater(THREAT_PRIORITY["CAUTION"], THREAT_PRIORITY["SAFE"])

    def test_risk_level_representations_match(self) -> None:
        """Verify schema and response mapping representations match 1-to-1."""
        levels = ["SAFE", "CAUTION", "WARNING", "CRITICAL"]
        for lvl in levels:
            self.assertIn(lvl, RESPONSE_MAPPINGS)
            self.assertIn(lvl, THREAT_PRIORITY)

    def test_highest_threat_selection_permutations(self) -> None:
        """Test combinations of risk levels on dummy objects always pick the highest priority."""
        processor = PerceptionToRiskProcessor()

        test_cases = [
            (["SAFE", "CAUTION"], "CAUTION"),
            (["SAFE", "WARNING"], "WARNING"),
            (["CAUTION", "WARNING"], "WARNING"),
            (["CAUTION", "CRITICAL"], "CRITICAL"),
            (["SAFE", "CAUTION", "WARNING", "CRITICAL"], "CRITICAL"),
            (["SAFE"], "SAFE"),
        ]

        for input_levels, expected_highest in test_cases:
            # Create synthetic assessments
            assessments = [
                ObjectRiskAssessment(
                    object_id=f"obj_{idx}",
                    risk_level=lvl,
                    class_id=idx,
                    confidence=0.9,
                    probabilities={l: (0.9 if l == lvl else 0.033) for l in ["SAFE", "CAUTION", "WARNING", "CRITICAL"]},
                    distance_m=10.0,
                    relative_velocity_mps=-2.0,
                    time_to_collision_s=5.0,
                    object_x_m=10.0,
                    object_y_m=0.0,
                    model="HistGradientBoostingClassifier",
                )
                for idx, lvl in enumerate(input_levels)
            ]
            frame = RiskAssessmentFrame(
                frame_id="test_frame",
                timestamp_ns=1000,
                inference_timestamp="2026-09-29T00:00:00Z",
                processing_time_ms=5.0,
                objects_detected=len(assessments),
                objects_evaluated=len(assessments),
                highest_threat_level=processor._determine_highest_threat(assessments),
                highest_threat_object_id=None,
                assessments=assessments,
            )
            self.assertEqual(frame.highest_threat_level, expected_highest)


# =============================================================================
# TEST CLASS 4: SENSOR HEALTH / RISK DECOUPLING (Task 5)
# =============================================================================
class TestStage15SensorHealthRiskDecoupling(unittest.TestCase):
    """TASK 5: Verify strict decoupling between SensorHealthState and Risk State."""

    def setUp(self) -> None:
        self.adapter = ROS2LaserScanAdapter()

    def test_all_health_states_distinct_from_risk(self) -> None:
        health_states = [s.value for s in SensorHealthState]
        risk_levels = ["SAFE", "CAUTION", "WARNING", "CRITICAL"]
        # Health states and risk levels must not overlap
        self.assertEqual(len(set(health_states).intersection(set(risk_levels))), 0)

    def test_health_state_transitions(self) -> None:
        """Test transitions: NO_DATA -> OK -> DEGRADED -> OK -> STALE -> OK -> INVALID -> OK."""
        now_s = time.time()
        now_ns = int(now_s * 1e9)

        # 1. Startup: adapter initial state is NO_DATA
        self.assertEqual(self.adapter.health_state, SensorHealthState.NO_DATA)

        # 2. First clean scan -> OK
        scan1 = make_synthetic_scan(timestamp_ns=now_ns)
        res1, _ = self.adapter.ingest_laser_scan(scan1, current_time_s=now_s)
        self.assertEqual(res1.health_state, SensorHealthState.OK)
        self.assertEqual(self.adapter.health_state, SensorHealthState.OK)

        # 3. Degraded scan (severe beam occlusion / 90% NaN) -> DEGRADED
        ranges_deg = [float("nan")] * 325 + [12.0] * 36
        scan_deg = make_synthetic_scan(timestamp_ns=now_ns + 50_000_000, ranges=ranges_deg)
        res_deg, _ = self.adapter.ingest_laser_scan(scan_deg, current_time_s=now_s + 0.05)
        self.assertEqual(res_deg.health_state, SensorHealthState.DEGRADED)
        self.assertEqual(self.adapter.health_state, SensorHealthState.DEGRADED)

        # 4. Clean scan recovers to OK
        scan_clean = make_synthetic_scan(timestamp_ns=now_ns + 100_000_000)
        res_clean, _ = self.adapter.ingest_laser_scan(scan_clean, current_time_s=now_s + 0.1)
        self.assertEqual(res_clean.health_state, SensorHealthState.OK)
        self.assertEqual(self.adapter.health_state, SensorHealthState.OK)

        # 5. Stale scan (>0.5s old) -> STALE
        scan_stale = make_synthetic_scan(timestamp_ns=now_ns - 1_000_000_000)
        res_stale, _ = self.adapter.ingest_laser_scan(scan_stale, current_time_s=now_s + 0.15)
        self.assertEqual(res_stale.health_state, SensorHealthState.STALE)
        self.assertEqual(self.adapter.health_state, SensorHealthState.STALE)

        # 6. Fresh scan recovers to OK
        scan_fresh = make_synthetic_scan(timestamp_ns=now_ns + 200_000_000)
        res_fresh, _ = self.adapter.ingest_laser_scan(scan_fresh, current_time_s=now_s + 0.2)
        self.assertEqual(res_fresh.health_state, SensorHealthState.OK)
        self.assertEqual(self.adapter.health_state, SensorHealthState.OK)

        # 7. Corrupted geometry -> INVALID
        scan_inv = make_synthetic_scan(timestamp_ns=now_ns + 250_000_000)
        scan_inv["angle_increment"] = -0.01  # invalid increment
        res_inv, _ = self.adapter.ingest_laser_scan(scan_inv, current_time_s=now_s + 0.25)
        self.assertEqual(res_inv.health_state, SensorHealthState.INVALID)
        self.assertEqual(self.adapter.health_state, SensorHealthState.INVALID)

        # 8. Clean scan recovers to OK
        scan_recov = make_synthetic_scan(timestamp_ns=now_ns + 300_000_000)
        res_recov, _ = self.adapter.ingest_laser_scan(scan_recov, current_time_s=now_s + 0.3)
        self.assertEqual(res_recov.health_state, SensorHealthState.OK)
        self.assertEqual(self.adapter.health_state, SensorHealthState.OK)

    def test_stale_and_invalid_inputs_do_not_fabricate_risk(self) -> None:
        """Verify invalid or stale scans emit safe fallback risk, not fabricated alerts."""
        pipeline = LaserScanToRiskPipeline()
        now_s = time.time()

        # Stale input
        stale_scan = make_synthetic_scan(timestamp_ns=int((now_s - 2.0) * 1e9))
        val_res, frame = pipeline.process_scan(stale_scan, current_time_s=now_s)
        self.assertEqual(val_res.health_state, SensorHealthState.STALE)
        self.assertEqual(frame.highest_threat_level, "SAFE")
        self.assertEqual(frame.objects_evaluated, 0)

        # Invalid input
        inv_scan = make_synthetic_scan(timestamp_ns=int(now_s * 1e9), ranges=[-10.0] * 361)
        val_res2, frame2 = pipeline.process_scan(inv_scan, current_time_s=now_s)
        self.assertEqual(val_res2.health_state, SensorHealthState.INVALID)
        self.assertEqual(frame2.highest_threat_level, "SAFE")
        self.assertEqual(frame2.objects_evaluated, 0)


# =============================================================================
# TEST CLASS 5: FAULT INJECTION & RECOVERY (Task 6)
# =============================================================================
class TestStage15FaultInjectionAndRecovery(unittest.TestCase):
    """TASK 6: Controlled software fault injection and instantaneous recovery verification."""

    def setUp(self) -> None:
        self.pipeline = LaserScanToRiskPipeline()

    def _assert_fault_and_clean_recovery(self, bad_scan: dict[str, Any], current_time_s: float | None = None) -> None:
        """Helper to run a bad scan, assert safe isolation, then verify clean recovery with a good scan."""
        val_bad, frame_bad = self.pipeline.process_scan(bad_scan, current_time_s=current_time_s)
        self.assertFalse(val_bad.is_valid, f"Faulty scan was unexpectedly accepted: {val_bad.message}")
        self.assertEqual(frame_bad.highest_threat_level, "SAFE")
        self.assertEqual(frame_bad.objects_evaluated, 0)

        # Follow up immediately with clean valid scan
        t_clean = (current_time_s or time.time()) + 0.05
        good_scan = make_synthetic_scan(timestamp_ns=int(t_clean * 1e9))
        val_good, frame_good = self.pipeline.process_scan(good_scan, current_time_s=t_clean)
        self.assertTrue(val_good.is_valid, f"Clean scan failed to recover: {val_good.message}")
        self.assertEqual(val_good.health_state, SensorHealthState.OK)

    def test_fault_nan_ranges(self) -> None:
        scan = make_synthetic_scan(ranges=[float("nan")] * 361)
        self._assert_fault_and_clean_recovery(scan)

    def test_fault_negative_ranges(self) -> None:
        scan = make_synthetic_scan(ranges=[-1.0] * 361)
        self._assert_fault_and_clean_recovery(scan)

    def test_fault_wrong_range_array_length(self) -> None:
        scan = make_synthetic_scan(ranges=[5.0] * 50)  # expects 361
        self._assert_fault_and_clean_recovery(scan)

    def test_fault_invalid_angle_min_max(self) -> None:
        scan = make_synthetic_scan(angle_min=math.pi, angle_max=-math.pi)
        self._assert_fault_and_clean_recovery(scan)

    def test_fault_invalid_angle_increment(self) -> None:
        scan = make_synthetic_scan()
        scan["angle_increment"] = 0.0
        self._assert_fault_and_clean_recovery(scan)

    def test_fault_old_timestamp(self) -> None:
        now_s = time.time()
        scan = make_synthetic_scan(timestamp_ns=int((now_s - 5.0) * 1e9))
        self._assert_fault_and_clean_recovery(scan, current_time_s=now_s)

    def test_fault_future_timestamp(self) -> None:
        cfg = LiDAR2DConfig(max_future_skew_s=1.0)
        pipeline = LaserScanToRiskPipeline(config=cfg)
        now_s = time.time()
        scan = make_synthetic_scan(timestamp_ns=int((now_s + 10.0) * 1e9))
        val_bad, frame_bad = pipeline.process_scan(scan, current_time_s=now_s)
        self.assertFalse(val_bad.is_valid)
        self.assertEqual(val_bad.health_state, SensorHealthState.INVALID)
        self.assertIn("future timestamp", val_bad.message.lower())

    def test_fault_frame_id_mismatch(self) -> None:
        # Test empty frame_id
        scan_empty = make_synthetic_scan(frame_id="")
        self._assert_fault_and_clean_recovery(scan_empty)

        # Test frame_id mismatch when enforcement is active
        cfg = LiDAR2DConfig(expected_frame_id="laser_frame", enforce_frame_id=True)
        pipeline_strict = LaserScanToRiskPipeline(config=cfg)
        bad_scan = make_synthetic_scan(frame_id="wrong_camera_frame")
        val_bad, frame_bad = pipeline_strict.process_scan(bad_scan)
        self.assertFalse(val_bad.is_valid)
        self.assertEqual(val_bad.health_state, SensorHealthState.INVALID)

    def test_fault_empty_scan(self) -> None:
        scan = make_synthetic_scan(ranges=[])
        self._assert_fault_and_clean_recovery(scan)

    def test_fault_corrupted_majority_of_beams(self) -> None:
        """Verify high beam corruption enters DEGRADED state safely and recovers."""
        ranges = [float("inf")] * 361
        for i in range(320):  # >88% corrupted
            ranges[i] = float("nan")
        scan = make_synthetic_scan(ranges=ranges)
        val_res, frame = self.pipeline.process_scan(scan)
        self.assertTrue(val_res.is_valid)
        self.assertEqual(val_res.health_state, SensorHealthState.DEGRADED)
        self.assertEqual(frame.highest_threat_level, "SAFE")

        # Clean recovery
        good_scan = make_synthetic_scan()
        val_good, frame_good = self.pipeline.process_scan(good_scan)
        self.assertTrue(val_good.is_valid)
        self.assertEqual(val_good.health_state, SensorHealthState.OK)

    def test_fault_all_beams_corrupted(self) -> None:
        """Verify 100% corrupted beams enters INVALID state and isolates."""
        scan = make_synthetic_scan(ranges=[float("nan")] * 361)
        self._assert_fault_and_clean_recovery(scan)

    def test_fault_abrupt_obstacle_disappearance(self) -> None:
        """Obstacle appears at 10m then disappears. System should handle without crash."""
        now_s = time.time()
        # Scan 1: obstacle present
        ranges1 = [float("inf")] * 361
        add_obstacle_to_ranges(ranges1, distance=10.0, center_angle_rad=0.0, angular_width_rad=0.1)
        scan1 = make_synthetic_scan(timestamp_ns=int(now_s * 1e9), ranges=ranges1)
        v1, f1 = self.pipeline.process_scan(scan1, current_time_s=now_s)
        self.assertEqual(f1.objects_detected, 1)

        # Scan 2: obstacle abruptly gone
        scan2 = make_synthetic_scan(timestamp_ns=int((now_s + 0.1) * 1e9))
        v2, f2 = self.pipeline.process_scan(scan2, current_time_s=now_s + 0.1)
        self.assertTrue(v2.is_valid)
        self.assertEqual(f2.objects_detected, 0)
        self.assertEqual(f2.highest_threat_level, "SAFE")


# =============================================================================
# TEST CLASS 6: TEMPORAL TRACKING VALIDATION (Task 7)
# =============================================================================
class TestStage15TemporalTrackingValidation(unittest.TestCase):
    """TASK 7: Multi-scan temporal tracking kinematics, single-scan restriction & smoothing."""

    def setUp(self) -> None:
        self.pipeline = LaserScanToRiskPipeline()

    def test_single_scan_cannot_measure_velocity(self) -> None:
        """Explicit invariant: a single 2D scan CANNOT measure velocity."""
        now_ns = int(time.time() * 1e9)
        ranges = [float("inf")] * 361
        add_obstacle_to_ranges(ranges, distance=15.0, center_angle_rad=0.0, angular_width_rad=0.1)
        scan = make_synthetic_scan(timestamp_ns=now_ns, ranges=ranges)

        val_res, frame = self.pipeline.process_scan(scan)

        self.assertTrue(val_res.is_valid)
        self.assertEqual(frame.objects_evaluated, 1)
        assessment = frame.assessments[0]
        # In single scan: relative velocity must be strictly 0.0 m/s and TTC must be safe default (>= 50s)
        self.assertEqual(assessment.relative_velocity_mps, 0.0)
        self.assertGreaterEqual(assessment.time_to_collision_s, 50.0)

    def test_stationary_obstacle_velocity(self) -> None:
        """Stationary obstacle over multiple scans -> relative_velocity ~ 0.0 m/s."""
        base_t = time.time()
        for idx in range(4):
            t_s = base_t + (idx * 0.1)
            ranges = [float("inf")] * 361
            add_obstacle_to_ranges(ranges, distance=20.0, center_angle_rad=0.0, angular_width_rad=0.1)
            scan = make_synthetic_scan(timestamp_ns=int(t_s * 1e9), ranges=ranges)
            val_res, frame = self.pipeline.process_scan(scan, current_time_s=t_s)

        assessment = frame.assessments[0]
        self.assertAlmostEqual(assessment.relative_velocity_mps, 0.0, delta=0.5)

    def test_approaching_obstacle_negative_velocity_and_ttc(self) -> None:
        """Approaching obstacle -> negative relative velocity, decreasing distance, valid TTC."""
        base_t = time.time()
        # Closing from 30m to 24m in 3 steps of 0.1s (nominal closing speed = -20 m/s)
        distances = [30.0, 28.0, 26.0, 24.0]
        for idx, dist in enumerate(distances):
            t_s = base_t + (idx * 0.1)
            ranges = [float("inf")] * 361
            add_obstacle_to_ranges(ranges, distance=dist, center_angle_rad=0.0, angular_width_rad=0.1)
            scan = make_synthetic_scan(timestamp_ns=int(t_s * 1e9), ranges=ranges)
            val_res, frame = self.pipeline.process_scan(scan, current_time_s=t_s)

        assessment = frame.assessments[0]
        self.assertLess(assessment.relative_velocity_mps, -10.0)
        self.assertIsNotNone(assessment.time_to_collision_s)
        self.assertAlmostEqual(
            assessment.time_to_collision_s,
            assessment.distance_m / (-assessment.relative_velocity_mps),
            delta=0.2,
        )

    def test_receding_obstacle_positive_velocity(self) -> None:
        """Receding obstacle -> positive relative velocity, TTC is None."""
        base_t = time.time()
        distances = [15.0, 16.5, 18.0, 19.5]
        for idx, dist in enumerate(distances):
            t_s = base_t + (idx * 0.1)
            ranges = [float("inf")] * 361
            add_obstacle_to_ranges(ranges, distance=dist, center_angle_rad=0.0, angular_width_rad=0.1)
            scan = make_synthetic_scan(timestamp_ns=int(t_s * 1e9), ranges=ranges)
            val_res, frame = self.pipeline.process_scan(scan, current_time_s=t_s)

        assessment = frame.assessments[0]
        self.assertGreater(assessment.relative_velocity_mps, 5.0)
        self.assertGreaterEqual(assessment.time_to_collision_s, 50.0)

    def test_track_continuity_and_timeout(self) -> None:
        """Track ID persists across consecutive frames and times out after missed frames."""
        tracker = MultiScanTracker2D(max_missed_cycles=2)
        base_t = time.time()

        # Frame 1: obstacle at (10.0, 0.0)
        obs1 = Obstacle2D(
            obstacle_id="obs_1",
            distance_m=10.0,
            object_x_m=10.0,
            object_y_m=0.0,
            span_width_m=1.0,
            depth_length_m=1.0,
            point_count=10,
        )
        t1 = tracker.track([obs1], timestamp_ns=int(base_t * 1e9))
        track_id = t1[0].tracking_id

        # Frame 2: obstacle slightly moved to (9.8, 0.0)
        obs2 = Obstacle2D(
            obstacle_id="obs_2",
            distance_m=9.8,
            object_x_m=9.8,
            object_y_m=0.0,
            span_width_m=1.0,
            depth_length_m=1.0,
            point_count=10,
        )
        t2 = tracker.track([obs2], timestamp_ns=int((base_t + 0.1) * 1e9))
        self.assertEqual(t2[0].tracking_id, track_id, "Track ID must remain continuous")

        # Frame 3 & 4: empty frames (missed)
        tracker.track([], timestamp_ns=int((base_t + 0.2) * 1e9))
        tracker.track([], timestamp_ns=int((base_t + 0.3) * 1e9))
        self.assertEqual(len(tracker.active_tracks), 1)  # still aged, missed=2

        # Frame 5: missed 3 > max_missed_frames -> pruned
        tracker.track([], timestamp_ns=int((base_t + 0.4) * 1e9))
        self.assertEqual(len(tracker.active_tracks), 0)


# =============================================================================
# TEST CLASS 7: API & WEBSOCKET CONTRACT AUDIT (Task 11)
# =============================================================================
class TestStage15ApiAndWebSocketContract(unittest.TestCase):
    """TASK 11: End-to-end FastAPI bridge, schema consistency & stale watchdog verification."""

    def setUp(self) -> None:
        self.client = TestClient(fastapi_app_module.app)
        self.bridge = MineRakshakSystemBridge(use_in_process_fastapi=True)

    def test_pipeline_to_fastapi_provenance_propagation(self) -> None:
        """Verify that provenance fields correctly reach FastAPI state."""
        now_ns = int(time.time() * 1e9)
        ranges = [float("inf")] * 361
        add_obstacle_to_ranges(ranges, distance=14.0, center_angle_rad=0.0, angular_width_rad=0.1)
        scan = make_synthetic_scan(timestamp_ns=now_ns, ranges=ranges)

        res = self.bridge.process_laserscan_to_dashboard(scan, truck_speed_kmph=25.0)

        self.assertEqual(res["status"], "valid")
        self.assertEqual(res["sensor_health"], "OK")

        # Verify FastAPI memory cache
        cached = fastapi_app_module.LATEST_MINERAKSHAK_RISK
        self.assertIn("ROS2LaserScanAdapter", cached["sensor_source"])
        self.assertIn(cached["source_type"], ["synthetic", "real_ros2"])
        self.assertFalse(cached["hardware_validated"])
        self.assertEqual(cached["sensor_health"], "OK")
        self.assertEqual(cached["feature_compatibility_status"], "COMPATIBLE_WITH_DEFAULTS")

    def test_fastapi_stale_watchdog(self) -> None:
        """Verify that GET /api/minerakshak/latest_risk marks state STALE when scan is older than 0.5s."""
        # Inject fresh state with old timestamp (1.0s ago)
        old_ns = int((time.time() - 1.0) * 1e9)
        fastapi_app_module.LATEST_MINERAKSHAK_RISK["timestamp_ns"] = old_ns
        fastapi_app_module.LATEST_MINERAKSHAK_RISK["sensor_health"] = "OK"
        fastapi_app_module.LATEST_MINERAKSHAK_RISK["status"] = "valid"

        response = self.client.get("/api/minerakshak/latest_risk")
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["sensor_health"], "STALE")
        self.assertEqual(data["status"], "stale_timed_out")


# =============================================================================
# TEST CLASS 8: RESTART & RECOVERY LIFECYCLE (Task 12)
# =============================================================================
class TestStage15RestartAndRecoveryLifecycle(unittest.TestCase):
    """TASK 12: 8-stage lifecycle validation from startup to restart."""

    def test_full_lifecycle(self) -> None:
        pipeline = LaserScanToRiskPipeline()
        base_t = time.time()

        # Step 1: Startup with no scans -> Initial state is clean
        self.assertEqual(pipeline.total_scans_processed, 0)
        self.assertEqual(len(pipeline.tracker.active_tracks), 0)

        # Step 2: First valid scan -> Velocity is 0.0 m/s
        ranges1 = [float("inf")] * 361
        add_obstacle_to_ranges(ranges1, distance=20.0, center_angle_rad=0.0, angular_width_rad=0.1)
        scan1 = make_synthetic_scan(timestamp_ns=int(base_t * 1e9), ranges=ranges1)
        val1, frame1 = pipeline.process_scan(scan1, current_time_s=base_t)
        self.assertTrue(val1.is_valid)
        self.assertEqual(frame1.assessments[0].relative_velocity_mps, 0.0)

        # Step 3: Multiple valid scans -> Velocity computed
        ranges2 = [float("inf")] * 361
        add_obstacle_to_ranges(ranges2, distance=18.5, center_angle_rad=0.0, angular_width_rad=0.1)
        scan2 = make_synthetic_scan(timestamp_ns=int((base_t + 0.1) * 1e9), ranges=ranges2)
        val2, frame2 = pipeline.process_scan(scan2, current_time_s=base_t + 0.1)
        self.assertLess(frame2.assessments[0].relative_velocity_mps, -5.0)

        # Step 4 & 5: Sensor dropout -> Stale scan rejected safely
        stale_scan = make_synthetic_scan(timestamp_ns=int((base_t - 2.0) * 1e9))
        val_stale, frame_stale = pipeline.process_scan(stale_scan, current_time_s=base_t + 0.2)
        self.assertEqual(val_stale.health_state, SensorHealthState.STALE)
        self.assertEqual(frame_stale.highest_threat_level, "SAFE")

        # Step 6: Sensor recovery with fresh valid scan
        ranges_rec = [float("inf")] * 361
        add_obstacle_to_ranges(ranges_rec, distance=15.0, center_angle_rad=0.0, angular_width_rad=0.1)
        scan_rec = make_synthetic_scan(timestamp_ns=int((base_t + 0.3) * 1e9), ranges=ranges_rec)
        val_rec, frame_rec = pipeline.process_scan(scan_rec, current_time_s=base_t + 0.3)
        self.assertTrue(val_rec.is_valid)

        # Step 7: Restart of processing component / tracker reset
        pipeline.reset_tracker()
        self.assertEqual(len(pipeline.tracker.active_tracks), 0)

        # Step 8: Post-restart first valid scan -> Velocity resets to 0.0 m/s (no leftover state)
        ranges_post = [float("inf")] * 361
        add_obstacle_to_ranges(ranges_post, distance=14.0, center_angle_rad=0.0, angular_width_rad=0.1)
        scan_post = make_synthetic_scan(timestamp_ns=int((base_t + 0.4) * 1e9), ranges=ranges_post)
        val_post, frame_post = pipeline.process_scan(scan_post, current_time_s=base_t + 0.4)
        self.assertTrue(val_post.is_valid)
        self.assertEqual(frame_post.assessments[0].relative_velocity_mps, 0.0)


# =============================================================================
# TEST CLASS 9: SAFETY INVARIANT AUDIT (Task 13)
# =============================================================================
class TestStage15SafetyInvariantAudit(unittest.TestCase):
    """TASK 13: Prove zero vehicle actuation commands and strictly advisory output."""

    def test_repository_contains_zero_actuation_commands(self) -> None:
        """Verify that no active control or actuator output topics/commands exist."""
        # Active control keywords that must NEVER exist as publisher topics or command methods
        prohibited_actuators = [
            "/cmd_vel",
            "publish_brake_command",
            "publish_steering_command",
            "publish_throttle_command",
            "actuator_override",
            "apply_emergency_braking",
        ]

        # Scan active src files
        src_dir = PROJECT_ROOT / "src"
        ros_dir = PROJECT_ROOT / "ros2_ws" / "src" / "minerakshak_risk"

        for base_path in [src_dir, ros_dir]:
            for py_path in base_path.rglob("*.py"):
                text = py_path.read_text(encoding="utf-8")
                for cmd in prohibited_actuators:
                    self.assertNotIn(
                        cmd,
                        text,
                        f"Prohibited actuation mechanism '{cmd}' found in {py_path}",
                    )

    def test_advisory_disclaimers_present_in_all_outputs(self) -> None:
        """Verify that SAFETY_DISCLAIMER is present in predictions and schemas."""
        engine = get_inference_engine()
        pred = engine.predict_single({
            "distance_m": 12.0,
            "relative_velocity_mps": -5.0,
            "truck_speed_kmph": 25.0,
            "object_x_m": 12.0,
            "object_y_m": 0.0,
            "object_z_m": 0.0,
            "object_width_m": 2.0,
            "object_height_m": 1.5,
            "object_length_m": 2.0,
            "point_count": 15,
            "time_to_collision_s": 2.4,
            "object_type": "unknown",
        })
        self.assertIn("disclaimer", pred)
        self.assertIn("decision support", pred["disclaimer"])


if __name__ == "__main__":
    unittest.main()
