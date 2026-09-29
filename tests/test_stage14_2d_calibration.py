"""Stage 14 Test Suite: Real-World 2D LiDAR Calibration, Sensor Health & Feature-Mapping Validation.

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

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
ROS2_PKG_ROOT = PROJECT_ROOT / "ros2_ws" / "src" / "minerakshak_risk"

for p in [str(PROJECT_ROOT), str(ROS2_PKG_ROOT), str(WORKSPACE_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from fastapi.testclient import TestClient
import main as fastapi_app_module
from src.bridge.ros2_fastapi_bridge import MineRakshakSystemBridge
from src.perception.feature_mapper_2d import (
    FEATURE_COMPATIBILITY_MATRIX,
    STAGE12_COMPATIBILITY_STATEMENT,
    FeatureMapper2D,
)
from src.perception.lidar_2d import LaserScan2D, Obstacle2D
from src.perception.lidar_calibration_2d import (
    CALIBRATION_DISCLAIMER,
    LiDARCalibration2D,
)
from src.perception.offline_validator_2d import (
    OfflineValidator2D,
    ValidationSessionSummary,
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
from minerakshak_risk.risk_node import PerceptionToRiskProcessor
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


def make_synthetic_scan_dict(
    timestamp_ns: int | None = None,
    num_beams: int = 361,
    angle_min: float = -math.pi / 2,
    angle_max: float = math.pi / 2,
    range_min: float = 0.1,
    range_max: float = 80.0,
    ranges: list[float] | None = None,
    frame_id: str = "laser_frame",
) -> dict[str, Any]:
    """Helper to construct synthetic LaserScan dictionary."""
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
        "intensities": [100.0] * num_beams,
    }


class TestStage14SensorConfiguration(unittest.TestCase):
    """TASK 1 & 2: Comprehensive tests for 2D sensor configuration & calibration."""

    def test_default_config_is_valid(self) -> None:
        cfg = LiDAR2DConfig()
        self.assertTrue(cfg.is_valid)
        self.assertEqual(len(cfg.validate_config()), 0)
        self.assertEqual(cfg.heading_convention, "+X_FORWARD")
        self.assertEqual(cfg.lateral_convention, "+Y_LEFT")
        self.assertEqual(cfg.elevation_convention, "Z_UNAVAILABLE")

    def test_invalid_angle_configurations(self) -> None:
        # Invalid FOV <= 0
        cfg1 = LiDAR2DConfig(expected_fov_deg=-10.0)
        self.assertFalse(cfg1.is_valid)
        self.assertIn("expected_fov_deg must be in (0, 360]", cfg1.validate_config())

        # Invalid resolution <= 0
        cfg2 = LiDAR2DConfig(angular_resolution_deg=0.0)
        self.assertFalse(cfg2.is_valid)
        self.assertIn("angular_resolution_deg must be in (0, 45]", cfg2.validate_config())

    def test_invalid_range_configurations(self) -> None:
        # Negative range_min
        cfg1 = LiDAR2DConfig(range_min_m=-0.5)
        self.assertFalse(cfg1.is_valid)
        self.assertIn("range_min_m cannot be negative", cfg1.validate_config())

        # range_min >= range_max
        cfg2 = LiDAR2DConfig(range_min_m=20.0, range_max_m=10.0)
        self.assertFalse(cfg2.is_valid)
        self.assertIn("range_min_m must be strictly less than range_max_m", cfg2.validate_config())

    def test_invalid_scan_frequencies(self) -> None:
        # Non-positive expected frequency
        cfg1 = LiDAR2DConfig(expected_scan_frequency_hz=0.0)
        self.assertFalse(cfg1.is_valid)

        # Inverted min/max scan frequency
        cfg2 = LiDAR2DConfig(min_scan_frequency_hz=30.0, max_scan_frequency_hz=10.0)
        self.assertFalse(cfg2.is_valid)

    def test_preset_profiles(self) -> None:
        # SICK TiM781
        sick = LiDAR2DConfig.create_sick_tim781_preset()
        self.assertTrue(sick.is_valid)
        self.assertEqual(sick.sensor_model_name, "SICK_TiM781")
        self.assertEqual(sick.expected_fov_deg, 270.0)
        self.assertEqual(sick.range_max_m, 25.0)

        # Hokuyo UST-20LX
        hokuyo = LiDAR2DConfig.create_hokuyo_ust20lx_preset()
        self.assertTrue(hokuyo.is_valid)
        self.assertEqual(hokuyo.expected_scan_frequency_hz, 40.0)

        # RPLIDAR S2
        rplidar = LiDAR2DConfig.create_rplidar_s2_preset()
        self.assertTrue(rplidar.is_valid)
        self.assertEqual(rplidar.expected_fov_deg, 360.0)

    def test_serialization_round_trip(self) -> None:
        cfg = LiDAR2DConfig(sensor_model_name="Custom_Test_LiDAR", mounting_yaw_offset_rad=0.1)
        d = cfg.to_dict()
        cfg_loaded = LiDAR2DConfig.from_dict(d)
        self.assertEqual(cfg_loaded.sensor_model_name, "Custom_Test_LiDAR")
        self.assertAlmostEqual(cfg_loaded.mounting_yaw_offset_rad, 0.1)


class TestStage14CalibrationFramework(unittest.TestCase):
    """TASK 2: Coordinate transformation, mounting offsets, and verification."""

    def setUp(self) -> None:
        self.config = LiDAR2DConfig(
            mounting_yaw_offset_rad=0.0,
            mounting_x_offset_m=0.0,
            mounting_y_offset_m=0.0,
        )
        self.calibrator = LiDARCalibration2D(config=self.config)

    def test_forward_obstacle_transformation(self) -> None:
        # Obstacle directly ahead at 10.0 m (theta = 0)
        x, y, valid = self.calibrator.transform_polar_to_vehicle_cartesian(range_m=10.0, angle_rad=0.0)
        self.assertTrue(valid)
        self.assertAlmostEqual(x, 10.0, places=3)
        self.assertAlmostEqual(y, 0.0, places=3)

    def test_lateral_obstacle_transformations(self) -> None:
        # Left obstacle at 5.0 m (theta = +pi/2)
        x_left, y_left, valid_l = self.calibrator.transform_polar_to_vehicle_cartesian(range_m=5.0, angle_rad=math.pi / 2)
        self.assertTrue(valid_l)
        self.assertAlmostEqual(x_left, 0.0, places=3)
        self.assertAlmostEqual(y_left, 5.0, places=3)

        # Right obstacle at 5.0 m (theta = -pi/2)
        x_right, y_right, valid_r = self.calibrator.transform_polar_to_vehicle_cartesian(range_m=5.0, angle_rad=-math.pi / 2)
        self.assertTrue(valid_r)
        self.assertAlmostEqual(x_right, 0.0, places=3)
        self.assertAlmostEqual(y_right, -5.0, places=3)

    def test_mounting_yaw_offset(self) -> None:
        # Sensor rotated +90 deg (+pi/2) counter-clockwise on vehicle
        # A beam fired at sensor 0 rad actually points to vehicle +Y (left)
        cal = LiDARCalibration2D(LiDAR2DConfig(mounting_yaw_offset_rad=math.pi / 2))
        x, y, valid = cal.transform_polar_to_vehicle_cartesian(range_m=10.0, angle_rad=0.0)
        self.assertTrue(valid)
        self.assertAlmostEqual(x, 0.0, places=3)
        self.assertAlmostEqual(y, 10.0, places=3)

    def test_mounting_xy_translation_offsets(self) -> None:
        # Sensor mounted on front bumper: +3.0m forward (+X), +0.5m lateral left (+Y)
        cal = LiDARCalibration2D(LiDAR2DConfig(mounting_x_offset_m=3.0, mounting_y_offset_m=0.5))
        x, y, valid = cal.transform_polar_to_vehicle_cartesian(range_m=10.0, angle_rad=0.0)
        self.assertTrue(valid)
        self.assertAlmostEqual(x, 13.0, places=3)
        self.assertAlmostEqual(y, 0.5, places=3)

    def test_out_of_bounds_filtering(self) -> None:
        # Returns below range_min or above range_max are marked invalid
        _, _, valid1 = self.calibrator.transform_polar_to_vehicle_cartesian(range_m=0.05, angle_rad=0.0)
        self.assertFalse(valid1)
        _, _, valid2 = self.calibrator.transform_polar_to_vehicle_cartesian(range_m=95.0, angle_rad=0.0)
        self.assertFalse(valid2)
        _, _, valid3 = self.calibrator.transform_polar_to_vehicle_cartesian(range_m=float("nan"), angle_rad=0.0)
        self.assertFalse(valid3)
        _, _, valid4 = self.calibrator.transform_polar_to_vehicle_cartesian(range_m=float("inf"), angle_rad=0.0)
        self.assertFalse(valid4)

    def test_calibration_verification_report(self) -> None:
        scan = LaserScan2D(
            frame_id="laser_frame",
            timestamp_ns=int(time.time() * 1e9),
            angle_min=-math.pi / 2,
            angle_max=math.pi / 2,
            angle_increment=math.pi / 360,
            time_increment=0.0,
            scan_time=0.05,
            range_min=0.1,
            range_max=80.0,
            ranges=[10.0] * 361,
            intensities=[100.0] * 361,
        )
        report = self.calibrator.verify_scan_calibration(scan)
        self.assertTrue(report.is_calibrated)
        self.assertEqual(len(report.errors), 0)


class TestStage14ScanQualityMetrics(unittest.TestCase):
    """TASK 3: Diagnostic metrics per scan (beam counts, percentages, coverage, frequency)."""

    def setUp(self) -> None:
        self.config = LiDAR2DConfig()
        self.analyzer = ScanQualityAnalyzer2D(config=self.config)

    def test_healthy_scan_quality(self) -> None:
        ranges = [20.0] * 361
        scan = LaserScan2D(
            frame_id="laser_frame",
            timestamp_ns=int(time.time() * 1e9),
            angle_min=-math.pi / 2,
            angle_max=math.pi / 2,
            angle_increment=math.pi / 360,
            time_increment=0.0,
            scan_time=0.05,
            range_min=0.1,
            range_max=80.0,
            ranges=ranges,
            intensities=[100.0] * 361,
        )
        q = self.analyzer.analyze_scan(scan)
        self.assertEqual(q.quality_assessment, "HEALTHY")
        self.assertEqual(q.valid_beam_count, 361)
        self.assertEqual(q.valid_beam_percentage, 100.0)
        self.assertEqual(q.nan_beam_count, 0)
        self.assertAlmostEqual(q.min_valid_range_m, 20.0)
        self.assertAlmostEqual(q.max_valid_range_m, 20.0)
        self.assertAlmostEqual(q.scan_coverage_deg, 180.0, places=1)

    def test_nan_heavy_scan(self) -> None:
        # 95% NaN beams (corrupted / severe sensor fault)
        ranges = [float("nan")] * 345 + [15.0] * 16
        scan = LaserScan2D(
            frame_id="laser_frame",
            timestamp_ns=int(time.time() * 1e9),
            angle_min=-math.pi / 2,
            angle_max=math.pi / 2,
            angle_increment=math.pi / 360,
            time_increment=0.0,
            scan_time=0.05,
            range_min=0.1,
            range_max=80.0,
            ranges=ranges,
            intensities=[0.0] * 361,
        )
        q = self.analyzer.analyze_scan(scan)
        self.assertEqual(q.quality_assessment, "DEGRADED")
        self.assertGreater(q.nan_beam_percentage, 90.0)
        self.assertEqual(q.valid_beam_count, 16)

    def test_infinite_free_space_scan(self) -> None:
        # 100% inf beams (empty open pit / open haul road)
        ranges = [float("inf")] * 361
        scan = LaserScan2D(
            frame_id="laser_frame",
            timestamp_ns=int(time.time() * 1e9),
            angle_min=-math.pi / 2,
            angle_max=math.pi / 2,
            angle_increment=math.pi / 360,
            time_increment=0.0,
            scan_time=0.05,
            range_min=0.1,
            range_max=80.0,
            ranges=ranges,
            intensities=[0.0] * 361,
        )
        q = self.analyzer.analyze_scan(scan)
        self.assertEqual(q.quality_assessment, "EMPTY_FIELD")
        self.assertEqual(q.infinite_beam_count, 361)
        self.assertEqual(q.valid_beam_count, 0)

    def test_frequency_estimation_from_consecutive_timestamps(self) -> None:
        t0_ns = 1_000_000_000_000
        # 50 ms gap = 20 Hz
        t1_ns = t0_ns + 50_000_000

        scan1 = LaserScan2D(
            frame_id="laser_frame", timestamp_ns=t0_ns, angle_min=-1.0, angle_max=1.0,
            angle_increment=0.1, time_increment=0.0, scan_time=0.05, range_min=0.1, range_max=80.0,
            ranges=[10.0] * 21, intensities=[100.0] * 21,
        )
        scan2 = LaserScan2D(
            frame_id="laser_frame", timestamp_ns=t1_ns, angle_min=-1.0, angle_max=1.0,
            angle_increment=0.1, time_increment=0.0, scan_time=0.05, range_min=0.1, range_max=80.0,
            ranges=[10.0] * 21, intensities=[100.0] * 21,
        )

        self.analyzer.reset()
        self.analyzer.analyze_scan(scan1)
        q2 = self.analyzer.analyze_scan(scan2)
        self.assertAlmostEqual(q2.scan_frequency_hz_estimate, 20.0, delta=0.5)


class TestStage14SensorHealthClassification(unittest.TestCase):
    """TASK 4: Explicit, auditable criteria for OK, NO_DATA, STALE, INVALID, DEGRADED."""

    def setUp(self) -> None:
        self.adapter = ROS2LaserScanAdapter(
            config=LiDAR2DConfig(stale_timeout_s=0.5, min_valid_beam_ratio=0.15),
            is_synthetic_source=True,
        )

    def test_initial_state_is_no_data(self) -> None:
        self.assertEqual(self.adapter.current_health, SensorHealthState.NO_DATA)

    def test_none_input_transitions_to_invalid_or_no_data(self) -> None:
        val_res, scan = self.adapter.ingest_laser_scan(None)
        self.assertFalse(val_res.is_valid)
        self.assertIsNone(scan)
        self.assertIn(val_res.health_state, [SensorHealthState.NO_DATA, SensorHealthState.INVALID])

    def test_stale_scan_detection(self) -> None:
        # Scan timestamp is 1.0 second old (exceeds 0.5s stale threshold)
        ref_time = time.time()
        old_time_ns = int((ref_time - 1.0) * 1e9)
        msg = make_synthetic_scan_dict(timestamp_ns=old_time_ns)

        val_res, scan = self.adapter.ingest_laser_scan(msg, current_time_s=ref_time)
        self.assertFalse(val_res.is_valid)
        self.assertEqual(val_res.health_state, SensorHealthState.STALE)
        self.assertIsNone(scan)

    def test_corrupted_scan_geometry_transitions_to_invalid(self) -> None:
        # Inverted angles (angle_min >= angle_max)
        bad_msg = make_synthetic_scan_dict()
        bad_msg["angle_min"] = 1.5
        bad_msg["angle_max"] = -1.5

        val_res, scan = self.adapter.ingest_laser_scan(bad_msg)
        self.assertFalse(val_res.is_valid)
        self.assertEqual(val_res.health_state, SensorHealthState.INVALID)
        self.assertIsNone(scan)

    def test_degraded_state_on_severe_occlusion(self) -> None:
        # 90% NaN beams (dust/slurry obscuration)
        ranges = [float("nan")] * 325 + [12.0] * 36
        now = time.time()
        msg = make_synthetic_scan_dict(timestamp_ns=int(now * 1e9), ranges=ranges)

        val_res, scan = self.adapter.ingest_laser_scan(msg, current_time_s=now)
        self.assertTrue(val_res.is_valid)  # Valid enough to process remaining beams
        self.assertEqual(val_res.health_state, SensorHealthState.DEGRADED)
        self.assertIsNotNone(scan)


class TestStage14FeatureMappingAndCompatibilityMatrix(unittest.TestCase):
    """TASK 5 & 6: Feature audit, nominal defaults, and frozen model compatibility."""

    def setUp(self) -> None:
        self.mapper = FeatureMapper2D(nominal_z_m=0.0, nominal_height_m=1.5)

    def test_feature_matrix_file_exists_and_covers_all_12_features(self) -> None:
        matrix_file = PROJECT_ROOT / "results" / "metrics" / "stage14_feature_compatibility.json"
        self.assertTrue(matrix_file.exists(), "Feature compatibility matrix JSON must exist")

        with open(matrix_file, encoding="utf-8") as f:
            data = json.load(f)

        features = data.get("features", [])
        self.assertEqual(len(features), 12)

        feat_names = {f["feature_name"] for f in features}
        expected_names = {
            "distance_m", "object_x_m", "object_y_m", "object_z_m",
            "object_width_m", "object_height_m", "object_length_m",
            "point_count", "relative_velocity_mps", "truck_speed_kmph",
            "time_to_collision_s", "object_type",
        }
        self.assertEqual(feat_names, expected_names)

    def test_compatibility_defaults_have_mandatory_disclaimer(self) -> None:
        matrix_file = PROJECT_ROOT / "results" / "metrics" / "stage14_feature_compatibility.json"
        with open(matrix_file, encoding="utf-8") as f:
            data = json.load(f)

        defaults_found = 0
        for f in data.get("features", []):
            if f.get("is_compatibility_default"):
                defaults_found += 1
                self.assertEqual(
                    f.get("compatibility_disclaimer"),
                    "This value is NOT a physical 2D LiDAR measurement.",
                )
        self.assertEqual(defaults_found, 3, "Exactly 3 features must be compatibility defaults (z, height, type)")

    def test_mapper_nominal_defaults_honesty(self) -> None:
        obs = Obstacle2D(
            obstacle_id="obs_001",
            distance_m=math.hypot(15.0, -2.0),
            object_x_m=15.0,
            object_y_m=-2.0,
            span_width_m=2.4,
            depth_length_m=3.0,
            point_count=10,
            relative_velocity_mps=-4.0,
            time_to_collision_s=3.75,
        )
        inf_d, meta = self.mapper.map_obstacle(obs, truck_speed_kmph=30.0)

        # Check values
        self.assertAlmostEqual(inf_d["distance_m"], math.hypot(15.0, -2.0))
        self.assertEqual(inf_d["object_z_m"], 0.0)
        self.assertEqual(inf_d["object_height_m"], 1.5)
        self.assertEqual(inf_d["object_type"], "unknown")

        # Check honest metadata flags
        self.assertFalse(meta.z_is_real_measurement)
        self.assertFalse(meta.height_is_real_measurement)
        self.assertFalse(meta.classification_is_real_measurement)


class TestStage14PredictionProvenance(unittest.TestCase):
    """TASK 7: End-to-end traceability of sensor provenance and health."""

    def test_provenance_propagated_to_risk_frames(self) -> None:
        adapter = ROS2LaserScanAdapter(is_synthetic_source=True)
        pipeline = LaserScanToRiskPipeline(adapter=adapter)
        now = time.time()
        # Create scan with 1 cluster ahead at 12m
        ranges = [float("inf")] * 361
        for b in range(175, 186):
            ranges[b] = 12.0

        msg = make_synthetic_scan_dict(timestamp_ns=int(now * 1e9), ranges=ranges)
        val_res, risk_frame = pipeline.process_scan(msg, current_time_s=now)

        self.assertTrue(val_res.is_valid)
        self.assertEqual(risk_frame.sensor_health, "OK")
        self.assertIn("ROS2LaserScanAdapter", risk_frame.sensor_source)
        self.assertEqual(risk_frame.source_type, "synthetic")
        self.assertFalse(risk_frame.hardware_validated)
        self.assertEqual(risk_frame.feature_compatibility_status, "COMPATIBLE_WITH_DEFAULTS")

        frame_dict = risk_frame.to_dict()
        self.assertIn("sensor_source", frame_dict)
        self.assertIn("source_type", frame_dict)
        self.assertIn("hardware_validated", frame_dict)
        self.assertIn("sensor_health", frame_dict)
        self.assertIn("feature_compatibility_status", frame_dict)

        if risk_frame.assessments:
            ass = risk_frame.assessments[0]
            self.assertEqual(ass.sensor_health, "OK")
            self.assertFalse(ass.hardware_validated)
            self.assertEqual(ass.feature_compatibility_status, "COMPATIBLE_WITH_DEFAULTS")


class TestStage14OfflineTooling(unittest.TestCase):
    """TASK 9: Command-line / programmatic offline validation tool."""

    def test_offline_validator_executes_on_records(self) -> None:
        validator = OfflineValidator2D()
        now = time.time()
        records: list[dict[str, Any]] = []

        for i in range(5):
            t = now + (i * 0.05)
            ranges = [float("inf")] * 361
            for b in range(178, 183):
                ranges[b] = 20.0 - (i * 0.5)
            records.append(make_synthetic_scan_dict(timestamp_ns=int(t * 1e9), ranges=ranges))

        summary = validator.run_validation_on_records(records, source_label="test_suite_replay")
        self.assertIsInstance(summary, ValidationSessionSummary)
        self.assertEqual(summary.total_frames_processed, 5)
        self.assertEqual(summary.hardware_status, "NOT VALIDATED — HARDWARE NOT AVAILABLE")
        self.assertIn(summary.data_source, "test_suite_replay")
        self.assertGreater(summary.total_obstacles_detected, 0)
        self.assertIn("OK", summary.health_state_counts)
        self.assertIn("total_cycle_mean_ms", summary.latency_ms)


class TestStage14SafetyInvariants(unittest.TestCase):
    """SAFETY AUDIT: Zero actuation, frozen ML artifacts, zero elevation fabrication."""

    def test_frozen_model_hashes_are_bitwise_identical(self) -> None:
        for rel_path, exp_hash in FROZEN_MODEL_HASHES.items():
            p = PROJECT_ROOT / rel_path
            self.assertTrue(p.exists(), f"Model artifact {rel_path} must exist")
            calc_hash = hashlib.sha256(p.read_bytes()).hexdigest()
            self.assertEqual(
                calc_hash,
                exp_hash,
                f"FROZEN ARTIFACT BREACH: {rel_path} was modified! Calculated: {calc_hash}",
            )

    def test_zero_actuation_invariants(self) -> None:
        pipeline = LaserScanToRiskPipeline()
        now = time.time()
        # Scan with close critical obstacle
        ranges = [float("inf")] * 361
        for b in range(178, 183):
            ranges[b] = 3.5  # Critical close range

        msg = make_synthetic_scan_dict(timestamp_ns=int(now * 1e9), ranges=ranges)
        _, risk_frame = pipeline.process_scan(msg, current_time_s=now)

        # Output MUST be strictly advisory; no vehicle actuation fields
        d = risk_frame.to_dict()
        for forbidden in ["brake_cmd", "steering_cmd", "throttle_cmd", "actuation"]:
            self.assertNotIn(forbidden, d)

    def test_stale_scan_produces_fault_frame_not_prediction(self) -> None:
        pipeline = LaserScanToRiskPipeline()
        now = time.time()
        old_t = now - 2.0  # 2.0s stale
        msg = make_synthetic_scan_dict(timestamp_ns=int(old_t * 1e9))

        val_res, risk_frame = pipeline.process_scan(msg, current_time_s=now)
        self.assertFalse(val_res.is_valid)
        self.assertEqual(val_res.health_state, SensorHealthState.STALE)
        self.assertEqual(len(risk_frame.assessments), 0)
        self.assertGreater(len(risk_frame.errors), 0)


class TestStage14EndToEndIntegration(unittest.TestCase):
    """TASK 10: End-to-End integration across full stack (LaserScan -> HGB -> FastAPI -> WebSocket)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(fastapi_app_module.app)
        cls.bridge = MineRakshakSystemBridge(use_in_process_fastapi=True)

    def test_laserscan_to_fastapi_and_websocket(self) -> None:
        now = time.time()
        # Scan with close multi-beam obstacle
        ranges = [float("inf")] * 361
        for b in range(178, 183):
            ranges[b] = 5.0

        scan_dict = make_synthetic_scan_dict(
            timestamp_ns=int(now * 1e9),
            ranges=ranges,
            frame_id="frame_stg14_e2e",
        )

        with self.client.websocket_connect("/ws/dashboard") as ws:
            # 1. Connect packet
            init_msg = ws.receive_json()
            self.assertEqual(init_msg["type"], "dashboard_update")

            # 2. Ingest LaserScan through bridge
            bridge_res = self.bridge.process_laserscan_to_dashboard(scan_dict, current_time_s=now)
            self.assertEqual(bridge_res["sensor_health"], "OK")
            self.assertEqual(bridge_res["frame_id"], "frame_stg14_e2e")

            # 3. Query FastAPI REST endpoint
            resp = self.client.get("/api/minerakshak/latest_risk")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["frame_id"], "frame_stg14_e2e")
            self.assertIn(data["highest_threat_level"], ["WARNING", "CRITICAL"])


if __name__ == "__main__":
    unittest.main()
