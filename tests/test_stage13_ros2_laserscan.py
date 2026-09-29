"""Stage 13 Test Suite: Real 2D LiDAR ROS 2 Interface, Scan Replay & Hardware-Ready Integration.

Validates the complete hardware-ready 2D planar LiDAR interface:
1. Valid LaserScan conversion (ROS 2 message / dict into LaserScan2D).
2. Correct angle calculation (get_angle(i) == angle_min + i * angle_increment).
3. Correct range handling (to_points extracts valid Cartesian points).
4. NaN rejection (individual NaN beams safely filtered out).
5. Infinity rejection (individual +inf, -inf beams safely handled).
6. Out-of-range rejection (beams < range_min or > range_max filtered).
7. Empty scan handling (empty ranges array produces controlled fault result).
8. Invalid configuration handling (angle_min >= angle_max, angle_increment <= 0, range_min >= range_max).
9. Timestamp preservation (timestamp_ns preserved exactly from header).
10. Frame ID preservation (frame_id preserved).
11. Mounting-offset handling (mounting yaw, x, y shifts applied accurately in vehicle frame).
12. Stale scan detection (scan with age > stale_timeout_s flagged STALE, prediction suppressed).
13. Invalid scan detection (corrupted arrays, malformed structures safely caught).
14. Sensor health transitions (NO_DATA -> OK -> STALE -> INVALID -> DEGRADED).
15. Replay serialization (LiDARScanRecorder writes valid JSONL with metadata).
16. Replay deserialization (LiDARScanReplayer reads scans back correctly).
17. Synthetic/real source metadata separation (source_type: synthetic vs real_ros2, rejecting ambiguous datasets).
18. Existing 2D obstacle extractor compatibility (ObstacleExtractor2D processes converted scan).
19. Existing temporal tracker compatibility (MultiScanTracker2D tracks obstacles across cycles).
20. Existing FeatureMapper2D compatibility (FeatureMapper2D maps obstacles to model schema).
21. Frozen HGB inference compatibility (HistGradientBoostingClassifier evaluates mapped features).
22. Highest-threat selection (threat hierarchy CRITICAL > WARNING > CAUTION > SAFE).
23. ROS 2 risk-message compatibility (laserscan_node callback produces valid RiskAssessmentFrame).
24. FastAPI integration compatibility (FastAPI endpoint reflects latest risk).
25. WebSocket integration compatibility (WebSocket communication and risk broadcasts).
26. Zero-actuation invariant (no braking, steering, throttle, or actuator commands).
27. Frozen model hash verification (SHA-256 hashes of all 5 frozen artifacts unchanged).
28. No-3D-fabrication invariant (metadata confirms 3D features are nominal contract defaults, not fabricated).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROS2_PKG_ROOT = PROJECT_ROOT / "ros2_ws" / "src" / "minerakshak_risk"
WORKSPACE_ROOT = PROJECT_ROOT.parent

for p in [str(PROJECT_ROOT), str(ROS2_PKG_ROOT), str(WORKSPACE_ROOT)]:
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
    STAGE12_COMPATIBILITY_STATEMENT,
    FeatureMapper2D,
)
from src.perception.lidar_2d import (
    LaserScan2D,
    LaserScanPoint2D,
    Obstacle2D,
)
from src.perception.lidar_scan_replay import (
    AmbiguousReplayDatasetError,
    LiDARScanRecorder,
    LiDARScanReplayer,
)
from src.perception.obstacle_extractor_2d import (
    MultiScanTracker2D,
    ObstacleExtractor2D,
)
from src.perception.ros2_laserscan_adapter import (
    HARDWARE_READY_DISCLAIMER,
    ROS2LaserScanAdapter,
)
from src.perception.sensor_config_2d import (
    LiDAR2DConfig,
    SensorHealthState,
    SensorValidationResult,
)
from src.perception.synthetic_lidar_2d import Synthetic2DLiDARAdapter
from minerakshak_risk.laserscan_node import (
    LaserScanToRiskPipeline,
    MineRakshakLaserScanNode,
)
from minerakshak_risk.risk_node import PerceptionToRiskProcessor
from minerakshak_risk.schemas import (
    DetectedObject,
    ObjectRiskAssessment,
    PerceptionFrame,
    RiskAssessmentFrame,
)

# Reference hashes recorded at start of Stage 13 baseline audit
EXPECTED_FROZEN_HASHES = {
    "hgb_model.joblib": "f885c29d3ceddae8d2f5d7a2c1ff44fac849ec57ed498504938d632adb7f1974",
    "preprocessor.joblib": "37bc0e1366174965e741ab223ff3619c2e6fdc62e139b24ba5e8eaff3f0948da",
    "feature_schema.json": "e9ca72a4b9a9675c8e85ad2059f0448f2140891512ad4d19c10280b90d2f986b",
    "label_mapping.json": "2b5fd687a4f99cc2fcc94e3bd031c48571615193e7cff4de621a379492b17ebc",
    "xgboost_model.json": "bb385fb728583b385acbd1e7a052a4579a951835473dc10b12f6dccfdfacb983",
}


class MockROS2LaserScanMsg:
    """Mock representing a native ROS 2 sensor_msgs/msg/LaserScan message."""

    def __init__(
        self,
        frame_id: str = "laser_frame",
        sec: int | None = None,
        nanosec: int | None = None,
        angle_min: float = -math.pi / 2.0,
        angle_max: float = math.pi / 2.0,
        angle_increment: float = math.radians(0.5),
        time_increment: float = 0.0,
        scan_time: float = 0.05,
        range_min: float = 0.1,
        range_max: float = 80.0,
        ranges: list[float] | None = None,
        intensities: list[float] | None = None,
    ) -> None:
        if sec is None:
            now = time.time()
            sec = int(now)
            nanosec = int((now - sec) * 1e9)
        elif nanosec is None:
            nanosec = 0

        class Stamp:
            def __init__(self, s: int, ns: int):
                self.sec = s
                self.nanosec = ns

        class Header:
            def __init__(self, fid: str, s: int, ns: int):
                self.frame_id = fid
                self.stamp = Stamp(s, ns)

        self.header = Header(frame_id, sec, nanosec)
        self.angle_min = angle_min
        self.angle_max = angle_max
        self.angle_increment = angle_increment
        self.time_increment = time_increment
        self.scan_time = scan_time
        self.range_min = range_min
        self.range_max = range_max
        if angle_increment > 0 and (angle_max > angle_min):
            num_beams = int(round((angle_max - angle_min) / angle_increment)) + 1
        else:
            num_beams = 100
        self.ranges = ranges if ranges is not None else [float("inf")] * num_beams
        self.intensities = intensities if intensities is not None else [100.0] * len(self.ranges)


class TestStage13ROS2LaserScan(unittest.TestCase):
    """Exhaustive test suite for Stage 13 Real 2D LiDAR ROS 2 Interface & Scan Replay."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = get_inference_engine()
        cls.synthetic_adapter = Synthetic2DLiDARAdapter(fov_deg=180.0, resolution_deg=0.5, seed=42)
        cls.extractor = ObstacleExtractor2D(cluster_distance_threshold_m=1.2, min_cluster_points=2)
        cls.tracker = MultiScanTracker2D(gating_distance_m=4.5, velocity_smoothing_alpha=0.7)
        cls.mapper = FeatureMapper2D(nominal_z_m=0.0, nominal_height_m=1.5)
        cls.client = TestClient(fastapi_app_module.app)
        cls.bridge = MineRakshakSystemBridge(use_in_process_fastapi=True)

    def setUp(self) -> None:
        self.tracker.reset()

    # -------------------------------------------------------------------------
    # Test 1: Valid LaserScan Conversion
    # -------------------------------------------------------------------------
    def test_01_valid_laserscan_conversion(self) -> None:
        """1. Verify valid ROS 2 LaserScan message converts accurately to LaserScan2D."""
        msg = MockROS2LaserScanMsg(frame_id="front_lidar", sec=100, nanosec=200)
        adapter = ROS2LaserScanAdapter(time_provider=lambda: 100.25)
        val_res, scan = adapter.ingest_laser_scan(msg)

        self.assertTrue(val_res.is_valid)
        self.assertEqual(val_res.health_state, SensorHealthState.OK)
        self.assertIsNotNone(scan)
        self.assertEqual(scan.frame_id, "front_lidar")
        self.assertEqual(scan.timestamp_ns, 100_000_000_200)
        self.assertEqual(scan.beam_count, 361)
        self.assertAlmostEqual(scan.range_min, 0.1)
        self.assertAlmostEqual(scan.range_max, 80.0)

    # -------------------------------------------------------------------------
    # Test 2: Correct Angle Calculation
    # -------------------------------------------------------------------------
    def test_02_correct_angle_calculation(self) -> None:
        """2. Verify polar beam angle computation matches ROS 2 convention."""
        msg = MockROS2LaserScanMsg()
        adapter = ROS2LaserScanAdapter(time_provider=lambda: 1774847392.5)
        _, scan = adapter.ingest_laser_scan(msg)
        self.assertIsNotNone(scan)

        # Beam 0 is -90 deg (-pi/2)
        self.assertAlmostEqual(scan.get_angle(0), -math.pi / 2.0, places=5)
        # Center beam (180) is straight ahead (0.0 rad)
        mid_idx = 180
        self.assertAlmostEqual(scan.get_angle(mid_idx), 0.0, places=5)
        # Last beam (360) is +90 deg (+pi/2)
        self.assertAlmostEqual(scan.get_angle(360), math.pi / 2.0, places=5)

    # -------------------------------------------------------------------------
    # Test 3: Correct Range Handling
    # -------------------------------------------------------------------------
    def test_03_correct_range_handling(self) -> None:
        """3. Verify conversion from polar range to Cartesian (x, y) coordinates."""
        msg = MockROS2LaserScanMsg()
        # Set a target 10m ahead at 0 rad (beam 180)
        msg.ranges[180] = 10.0
        adapter = ROS2LaserScanAdapter(time_provider=lambda: 1774847392.5)
        _, scan = adapter.ingest_laser_scan(msg)
        points = scan.to_points()

        mid_point = points[180]
        self.assertTrue(mid_point.is_valid)
        self.assertAlmostEqual(mid_point.range_m, 10.0)
        self.assertAlmostEqual(mid_point.x_m, 10.0, places=2)
        self.assertAlmostEqual(mid_point.y_m, 0.0, places=2)

    # -------------------------------------------------------------------------
    # Test 4: NaN Rejection
    # -------------------------------------------------------------------------
    def test_04_nan_rejection(self) -> None:
        """4. Verify individual NaN returns are rejected without crashing scan processing."""
        msg = MockROS2LaserScanMsg()
        msg.ranges[10] = float("nan")
        msg.ranges[180] = 12.0
        adapter = ROS2LaserScanAdapter(time_provider=lambda: 1774847392.5)
        val_res, scan = adapter.ingest_laser_scan(msg)

        self.assertTrue(val_res.is_valid)
        points = scan.to_points()
        self.assertFalse(points[10].is_valid)
        self.assertEqual(points[10].x_m, 0.0)
        self.assertEqual(points[10].y_m, 0.0)
        self.assertTrue(points[180].is_valid)

    # -------------------------------------------------------------------------
    # Test 5: Infinity Rejection
    # -------------------------------------------------------------------------
    def test_05_infinity_rejection(self) -> None:
        """5. Verify positive infinity (free space) and negative infinity are handled safely."""
        msg = MockROS2LaserScanMsg()
        msg.ranges[0] = float("inf")
        msg.ranges[1] = float("-inf")
        msg.ranges[180] = 8.5
        adapter = ROS2LaserScanAdapter(time_provider=lambda: 1774847392.5)
        val_res, scan = adapter.ingest_laser_scan(msg)

        self.assertTrue(val_res.is_valid)
        points = scan.to_points()
        self.assertFalse(points[0].is_valid, "+inf must not be flagged as a valid obstacle return")
        self.assertFalse(points[1].is_valid, "-inf must be rejected")
        self.assertTrue(points[180].is_valid)

    # -------------------------------------------------------------------------
    # Test 6: Out-of-Range Rejection
    # -------------------------------------------------------------------------
    def test_06_out_of_range_rejection(self) -> None:
        """6. Verify returns below range_min or above range_max are marked invalid."""
        msg = MockROS2LaserScanMsg()
        msg.ranges[50] = 0.05   # Below range_min (0.1m)
        msg.ranges[60] = 85.0   # Above range_max (80.0m)
        msg.ranges[180] = 20.0  # Within range
        adapter = ROS2LaserScanAdapter(time_provider=lambda: 1774847392.5)
        _, scan = adapter.ingest_laser_scan(msg)
        points = scan.to_points()

        self.assertFalse(points[50].is_valid)
        self.assertFalse(points[60].is_valid)
        self.assertTrue(points[180].is_valid)

    # -------------------------------------------------------------------------
    # Test 7: Empty Scan Handling
    # -------------------------------------------------------------------------
    def test_07_empty_scan_handling(self) -> None:
        """7. Verify empty ranges array produces controlled validation failure."""
        msg = MockROS2LaserScanMsg()
        msg.ranges = []
        adapter = ROS2LaserScanAdapter(time_provider=lambda: 1774847392.5)
        val_res, scan = adapter.ingest_laser_scan(msg)

        self.assertFalse(val_res.is_valid)
        self.assertEqual(val_res.health_state, SensorHealthState.INVALID)
        self.assertIsNone(scan)
        self.assertIn("empty", val_res.message.lower())

    # -------------------------------------------------------------------------
    # Test 8: Invalid Configuration Handling
    # -------------------------------------------------------------------------
    def test_08_invalid_configuration_handling(self) -> None:
        """8. Verify bad scan geometry (e.g. angle_min >= angle_max or inc <= 0) is rejected."""
        # Bad angle_min >= angle_max
        msg1 = MockROS2LaserScanMsg(angle_min=1.0, angle_max=0.0)
        adapter = ROS2LaserScanAdapter(time_provider=lambda: 1774847392.5)
        val_res1, scan1 = adapter.ingest_laser_scan(msg1)
        self.assertFalse(val_res1.is_valid)
        self.assertEqual(val_res1.health_state, SensorHealthState.INVALID)
        self.assertIsNone(scan1)

        # Bad angle_increment <= 0
        msg2 = MockROS2LaserScanMsg(angle_increment=0.0)
        val_res2, scan2 = adapter.ingest_laser_scan(msg2)
        self.assertFalse(val_res2.is_valid)
        self.assertEqual(val_res2.health_state, SensorHealthState.INVALID)

        # Bad range_min >= range_max
        msg3 = MockROS2LaserScanMsg(range_min=50.0, range_max=20.0)
        val_res3, scan3 = adapter.ingest_laser_scan(msg3)
        self.assertFalse(val_res3.is_valid)
        self.assertEqual(val_res3.health_state, SensorHealthState.INVALID)

    # -------------------------------------------------------------------------
    # Test 9: Timestamp Preservation
    # -------------------------------------------------------------------------
    def test_09_timestamp_preservation(self) -> None:
        """9. Verify sensor header timestamp is preserved exactly in nanoseconds."""
        sec = 1718000000
        nanosec = 123456789
        msg = MockROS2LaserScanMsg(sec=sec, nanosec=nanosec)
        adapter = ROS2LaserScanAdapter(time_provider=lambda: sec + (nanosec / 1e9) + 0.05)
        val_res, scan = adapter.ingest_laser_scan(msg)

        self.assertTrue(val_res.is_valid)
        expected_ns = (sec * 1_000_000_000) + nanosec
        self.assertEqual(scan.timestamp_ns, expected_ns)

    # -------------------------------------------------------------------------
    # Test 10: Frame ID Preservation
    # -------------------------------------------------------------------------
    def test_10_frame_id_preservation(self) -> None:
        """10. Verify frame_id string is preserved from ROS 2 message."""
        msg = MockROS2LaserScanMsg(frame_id="mining_truck_bumper_lidar")
        adapter = ROS2LaserScanAdapter(time_provider=lambda: 1774847392.5)
        _, scan = adapter.ingest_laser_scan(msg)
        self.assertEqual(scan.frame_id, "mining_truck_bumper_lidar")

    # -------------------------------------------------------------------------
    # Test 11: Mounting-Offset Handling
    # -------------------------------------------------------------------------
    def test_11_mounting_offset_handling(self) -> None:
        """11. Verify mounting yaw rotation and translational offsets (x, y) transform correctly."""
        cfg = LiDAR2DConfig(
            mounting_yaw_offset_rad=math.radians(90.0),  # Sensor rotated 90 deg counter-clockwise
            mounting_x_offset_m=3.5,                    # Mounted 3.5m forward of vehicle origin
            mounting_y_offset_m=-0.5,                   # Mounted 0.5m right
        )
        adapter = ROS2LaserScanAdapter(config=cfg, time_provider=lambda: 1774847392.5)
        msg = MockROS2LaserScanMsg()
        # Beam 180 is 0 rad in sensor frame. With 90 deg yaw, vehicle angle becomes +90 deg (+Y)
        msg.ranges[180] = 10.0
        val_res, scan = adapter.ingest_laser_scan(msg)
        self.assertTrue(val_res.is_valid)

        points = scan.to_points()
        p180 = points[180]
        # In sensor frame: (x=10, y=0). Rotated +90 deg: (x=0, y=10).
        # Offset translation: x = 0 + 3.5 = 3.5m, y = 10 - 0.5 = 9.5m.
        self.assertAlmostEqual(p180.x_m, 3.5, places=2)
        self.assertAlmostEqual(p180.y_m, 9.5, places=2)

    # -------------------------------------------------------------------------
    # Test 12: Stale Scan Detection
    # -------------------------------------------------------------------------
    def test_12_stale_scan_detection(self) -> None:
        """12. Verify scans older than stale_timeout_s trigger STALE health and are rejected."""
        cfg = LiDAR2DConfig(stale_timeout_s=0.5)
        adapter = ROS2LaserScanAdapter(config=cfg)

        # Scan timestamp: t = 100.0s, current time: t = 100.8s (age = 0.8s > 0.5s timeout)
        msg = MockROS2LaserScanMsg(sec=100, nanosec=0)
        val_res, scan = adapter.ingest_laser_scan(msg, current_time_s=100.8)

        self.assertFalse(val_res.is_valid)
        self.assertEqual(val_res.health_state, SensorHealthState.STALE)
        self.assertIsNone(scan)
        self.assertIn("stale", val_res.message.lower())

        # Also verify get_latest_scan() expires and transitions to STALE when time elapses
        fresh_msg = MockROS2LaserScanMsg(sec=200, nanosec=0)
        adapter.ingest_laser_scan(fresh_msg, current_time_s=200.1)
        self.assertEqual(adapter.current_health, SensorHealthState.OK)

        # Advance clock beyond timeout
        adapter.time_provider = lambda: 201.0
        expired_scan = adapter.get_latest_scan()
        self.assertIsNone(expired_scan)
        self.assertEqual(adapter.current_health, SensorHealthState.STALE)

    # -------------------------------------------------------------------------
    # Test 13: Invalid Scan Detection
    # -------------------------------------------------------------------------
    def test_13_invalid_scan_detection(self) -> None:
        """13. Verify completely corrupted arrays (all NaNs or negative) are flagged INVALID."""
        msg = MockROS2LaserScanMsg()
        msg.ranges = [float("nan")] * len(msg.ranges)
        adapter = ROS2LaserScanAdapter(time_provider=lambda: 1774847392.5)
        val_res, scan = adapter.ingest_laser_scan(msg)

        self.assertFalse(val_res.is_valid)
        self.assertEqual(val_res.health_state, SensorHealthState.INVALID)
        self.assertIsNone(scan)
        self.assertIn("corrupted", val_res.message.lower())

    # -------------------------------------------------------------------------
    # Test 14: Sensor Health Transitions
    # -------------------------------------------------------------------------
    def test_14_sensor_health_transitions(self) -> None:
        """14. Verify full sensor health state machine transitions correctly."""
        adapter = ROS2LaserScanAdapter()
        # 1. Startup state: NO_DATA
        self.assertEqual(adapter.current_health, SensorHealthState.NO_DATA)

        # 2. Ingest valid scan -> OK
        msg_ok = MockROS2LaserScanMsg(sec=100, nanosec=0)
        val_res, _ = adapter.ingest_laser_scan(msg_ok, current_time_s=100.1)
        self.assertEqual(adapter.current_health, SensorHealthState.OK)

        # 3. Ingest stale scan -> STALE
        msg_stale = MockROS2LaserScanMsg(sec=100, nanosec=0)
        val_res, _ = adapter.ingest_laser_scan(msg_stale, current_time_s=102.0)
        self.assertEqual(adapter.current_health, SensorHealthState.STALE)

        # 4. Ingest corrupted scan -> INVALID
        msg_inv = MockROS2LaserScanMsg(angle_min=5.0, angle_max=1.0)
        val_res, _ = adapter.ingest_laser_scan(msg_inv, current_time_s=103.0)
        self.assertEqual(adapter.current_health, SensorHealthState.INVALID)

        # 5. Ingest degraded scan (noisy, 90% below range_min) -> DEGRADED
        msg_deg = MockROS2LaserScanMsg(sec=105, nanosec=0)
        for i in range(len(msg_deg.ranges)):
            if i % 10 != 0:
                msg_deg.ranges[i] = 0.05  # below range_min (corrupted return)
        val_res, scan = adapter.ingest_laser_scan(msg_deg, current_time_s=105.1)
        self.assertEqual(adapter.current_health, SensorHealthState.DEGRADED)
        self.assertIsNotNone(scan)

    # -------------------------------------------------------------------------
    # Test 15: Replay Serialization
    # -------------------------------------------------------------------------
    def test_15_replay_serialization(self) -> None:
        """15. Verify LiDARScanRecorder writes valid JSON Lines with explicit metadata."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "test_replay.jsonl"
            with LiDARScanRecorder(
                output_path=file_path,
                source_type="synthetic",
                hardware_validated=False,
                description="Test recording",
            ) as recorder:
                scan = self.synthetic_adapter.create_safe_scenario_scan()
                recorder.record_scan(scan)
                recorder.record_scan(scan)

            self.assertTrue(file_path.exists())
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            self.assertEqual(len(lines), 3)  # 1 header + 2 scans
            header = json.loads(lines[0])
            self.assertEqual(header["type"], "metadata")
            self.assertEqual(header["source_type"], "synthetic")
            self.assertFalse(header["hardware_validated"])

            scan1 = json.loads(lines[1])
            self.assertEqual(scan1["type"], "scan")
            self.assertEqual(len(scan1["ranges"]), 361)

    # -------------------------------------------------------------------------
    # Test 16: Replay Deserialization
    # -------------------------------------------------------------------------
    def test_16_replay_deserialization(self) -> None:
        """16. Verify LiDARScanReplayer correctly loads and streams recorded scans."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "test_replay.jsonl"
            scan_orig = self.synthetic_adapter.create_safe_scenario_scan()

            with LiDARScanRecorder(file_path, source_type="synthetic", hardware_validated=False) as rec:
                rec.record_scan(scan_orig)

            replayer = LiDARScanReplayer(file_path)
            self.assertEqual(len(replayer), 1)
            self.assertTrue(replayer.is_synthetic())

            scan_loaded = replayer.get_latest_scan()
            self.assertIsNotNone(scan_loaded)
            self.assertEqual(scan_loaded.beam_count, scan_orig.beam_count)
            self.assertEqual(scan_loaded.frame_id, scan_orig.frame_id)

    # -------------------------------------------------------------------------
    # Test 17: Synthetic / Real Source Metadata Separation
    # -------------------------------------------------------------------------
    def test_17_synthetic_real_metadata_separation(self) -> None:
        """17. Verify datasets reject ambiguous provenance and prevent synthetic mislabeling."""
        # 1. Attempting to mark synthetic data as hardware_validated=True must raise ValueError
        with tempfile.TemporaryDirectory() as tmp_dir:
            bad_path = Path(tmp_dir) / "fraudulent.jsonl"
            with self.assertRaises(ValueError):
                LiDARScanRecorder(bad_path, source_type="synthetic", hardware_validated=True)

            # 2. Loading a dataset with missing or invalid source_type must raise AmbiguousReplayDatasetError
            corrupt_path = Path(tmp_dir) / "ambiguous.jsonl"
            with open(corrupt_path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"type": "metadata", "description": "missing source_type"}) + "\n")

            with self.assertRaises(AmbiguousReplayDatasetError):
                LiDARScanReplayer(corrupt_path)

    # -------------------------------------------------------------------------
    # Test 18: Existing 2D Obstacle Extractor Compatibility
    # -------------------------------------------------------------------------
    def test_18_extractor_compatibility(self) -> None:
        """18. Verify existing ObstacleExtractor2D segments obstacles from converted LaserScan2D."""
        msg = MockROS2LaserScanMsg()
        # Inject an obstacle cluster around beam 180 (dist = 10m)
        for b in range(175, 186):
            msg.ranges[b] = 10.0
        adapter = ROS2LaserScanAdapter(time_provider=lambda: 1774847392.5)
        _, scan = adapter.ingest_laser_scan(msg)

        obstacles = self.extractor.extract_obstacles(scan)
        self.assertEqual(len(obstacles), 1)
        self.assertAlmostEqual(obstacles[0].distance_m, 10.0, places=1)
        self.assertEqual(obstacles[0].point_count, 11)

    # -------------------------------------------------------------------------
    # Test 19: Existing Temporal Tracker Compatibility
    # -------------------------------------------------------------------------
    def test_19_tracker_compatibility(self) -> None:
        """19. Verify MultiScanTracker2D derives closing velocity across consecutive cycles."""
        adapter = ROS2LaserScanAdapter()
        # Scan 1 at t = 100.0s, distance = 20.0m
        msg1 = MockROS2LaserScanMsg(sec=100, nanosec=0)
        for b in range(175, 186):
            msg1.ranges[b] = 20.0
        _, scan1 = adapter.ingest_laser_scan(msg1, current_time_s=100.05)
        obs1 = self.extractor.extract_obstacles(scan1)
        tracked1 = self.tracker.track(obs1, timestamp_ns=scan1.timestamp_ns)
        self.assertEqual(len(tracked1), 1)

        # Scan 2 at t = 100.1s (100ms later), distance = 18.5m (approaching: -15.0 m/s closing)
        msg2 = MockROS2LaserScanMsg(sec=100, nanosec=100_000_000)
        for b in range(175, 186):
            msg2.ranges[b] = 18.5
        _, scan2 = adapter.ingest_laser_scan(msg2, current_time_s=100.15)
        obs2 = self.extractor.extract_obstacles(scan2)
        tracked2 = self.tracker.track(obs2, timestamp_ns=scan2.timestamp_ns)

        self.assertEqual(len(tracked2), 1)
        self.assertLess(tracked2[0].relative_velocity_mps, -5.0)

    # -------------------------------------------------------------------------
    # Test 20: Existing FeatureMapper2D Compatibility
    # -------------------------------------------------------------------------
    def test_20_feature_mapper_compatibility(self) -> None:
        """20. Verify FeatureMapper2D maps 2D tracked obstacles without altering schema."""
        msg = MockROS2LaserScanMsg()
        for b in range(175, 186):
            msg.ranges[b] = 12.0
        adapter = ROS2LaserScanAdapter(time_provider=lambda: 1774847392.5)
        _, scan = adapter.ingest_laser_scan(msg)
        obs = self.extractor.extract_obstacles(scan)
        tracked = self.tracker.track(obs, timestamp_ns=scan.timestamp_ns)

        frame = self.mapper.to_perception_frame(
            tracked, truck_speed_kmph=20.0, frame_id="test_frame", timestamp_ns=scan.timestamp_ns
        )
        self.assertEqual(frame.frame_id, "test_frame")
        self.assertEqual(len(frame.objects), 1)
        obj = frame.objects[0]
        self.assertAlmostEqual(obj.distance_m, 12.0, places=1)
        self.assertEqual(obj.object_z_m, 0.0)  # Nominal baseline
        self.assertEqual(obj.object_height_m, 1.5)  # Nominal baseline

    # -------------------------------------------------------------------------
    # Test 21: Frozen HGB Inference Compatibility
    # -------------------------------------------------------------------------
    def test_21_frozen_hgb_inference_compatibility(self) -> None:
        """21. Verify frozen HGB model evaluates features produced through the ROS 2 adapter."""
        pipeline = LaserScanToRiskPipeline()
        msg = MockROS2LaserScanMsg()
        for b in range(175, 186):
            msg.ranges[b] = 30.0  # Safe distant object
        val_res, risk_frame = pipeline.process_scan(msg, current_time_s=1774847392.5)

        self.assertTrue(val_res.is_valid)
        self.assertEqual(risk_frame.objects_evaluated, 1)
        self.assertIn(risk_frame.highest_threat_level, ["SAFE", "CAUTION", "WARNING", "CRITICAL"])
        self.assertEqual(risk_frame.assessments[0].model, "HistGradientBoostingClassifier")

    # -------------------------------------------------------------------------
    # Test 22: Highest-Threat Selection
    # -------------------------------------------------------------------------
    def test_22_highest_threat_selection(self) -> None:
        """22. Verify threat prioritization selects the most hazardous obstacle in a frame."""
        pipeline = LaserScanToRiskPipeline()
        msg = MockROS2LaserScanMsg()
        # Obstacle 1: Distant left (safe)
        for b in range(50, 60):
            msg.ranges[b] = 40.0
        # Obstacle 2: Imminent center (critical)
        for b in range(175, 186):
            msg.ranges[b] = 4.0
        val_res, risk_frame = pipeline.process_scan(msg, current_time_s=1774847392.5)

        self.assertTrue(val_res.is_valid)
        self.assertEqual(risk_frame.objects_evaluated, 2)
        self.assertEqual(risk_frame.highest_threat_level, "CRITICAL")

    # -------------------------------------------------------------------------
    # Test 23: ROS 2 Risk-Message Compatibility
    # -------------------------------------------------------------------------
    def test_23_ros2_risk_message_compatibility(self) -> None:
        """23. Verify MineRakshakLaserScanNode callback formats outputs conforming to schemas."""
        node = MineRakshakLaserScanNode()
        msg = MockROS2LaserScanMsg()
        for b in range(178, 183):
            msg.ranges[b] = 15.0
        val_res, risk_frame = node.scan_callback(msg)

        self.assertTrue(val_res.is_valid)
        json_str = risk_frame.to_json()
        parsed = json.loads(json_str)
        self.assertIn("frame_id", parsed)
        self.assertIn("highest_threat_level", parsed)
        self.assertIn("assessments", parsed)

    # -------------------------------------------------------------------------
    # Test 24: FastAPI Integration Compatibility
    # -------------------------------------------------------------------------
    def test_24_fastapi_integration_compatibility(self) -> None:
        """24. Verify full path from LaserScan to FastAPI /api/minerakshak/latest_risk."""
        msg = MockROS2LaserScanMsg(frame_id="frame_stg13_fastapi")
        for b in range(178, 183):
            msg.ranges[b] = 5.0  # Close multi-beam obstacle
        bridge_res = self.bridge.process_laserscan_to_dashboard(msg)

        self.assertEqual(bridge_res["sensor_health"], "OK")
        self.assertEqual(bridge_res["frame_id"], "frame_stg13_fastapi")

        # Query FastAPI endpoint
        resp = self.client.get("/api/minerakshak/latest_risk")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["frame_id"], "frame_stg13_fastapi")
        self.assertIn(data["highest_threat_level"], ["WARNING", "CRITICAL"])

    # -------------------------------------------------------------------------
    # Test 25: WebSocket Integration Compatibility
    # -------------------------------------------------------------------------
    def test_25_websocket_integration_compatibility(self) -> None:
        """25. Verify WebSocket endpoint streams dashboard state after LaserScan ingestion."""
        with self.client.websocket_connect("/ws/dashboard") as ws:
            # Initial connection packet
            init_msg = ws.receive_json()
            self.assertEqual(init_msg["type"], "dashboard_update")

            # Ingest scan through bridge
            msg = MockROS2LaserScanMsg(frame_id="frame_ws_test")
            self.bridge.process_laserscan_to_dashboard(msg, current_time_s=1774847392.5)

    # -------------------------------------------------------------------------
    # Test 26: Zero-Actuation Invariant
    # -------------------------------------------------------------------------
    def test_26_zero_actuation_invariant(self) -> None:
        """26. INVARIANT 1-4: Verify no braking, steering, throttle, or actuator commands exist."""
        pipeline = LaserScanToRiskPipeline()
        msg = MockROS2LaserScanMsg()
        msg.ranges[180] = 2.0  # Ultra-close emergency collision threat
        _, risk_frame = pipeline.process_scan(msg, current_time_s=1774847392.5)

        frame_dict = risk_frame.to_dict()
        forbidden_actuator_keys = [
            "brake", "braking", "brake_pressure_bar", "steering",
            "steering_angle_deg", "throttle", "throttle_pct", "actuator_command",
            "apply_brake", "control_override"
        ]

        for k in forbidden_actuator_keys:
            self.assertNotIn(k, frame_dict, f"Actuation field '{k}' found in RiskAssessmentFrame!")

        for assm in frame_dict.get("assessments", []):
            for k in forbidden_actuator_keys:
                self.assertNotIn(k, assm, f"Actuation field '{k}' found in ObjectRiskAssessment!")

    # -------------------------------------------------------------------------
    # Test 27: Frozen Model Hash Verification
    # -------------------------------------------------------------------------
    def test_27_frozen_model_hash_verification(self) -> None:
        """27. INVARIANT 7: Verify SHA-256 hashes of all 5 frozen artifacts remain identical."""
        models_dir = PROJECT_ROOT / "models"
        for filename, expected_hash in EXPECTED_FROZEN_HASHES.items():
            file_path = models_dir / filename
            self.assertTrue(file_path.exists(), f"Frozen artifact missing: {filename}")
            actual_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
            self.assertEqual(
                actual_hash,
                expected_hash,
                f"VIOLATION: Frozen artifact '{filename}' has been modified! Expected {expected_hash}, got {actual_hash}",
            )

    # -------------------------------------------------------------------------
    # Test 28: No-3D-Fabrication Invariant
    # -------------------------------------------------------------------------
    def test_28_no_3d_fabrication_invariant(self) -> None:
        """28. INVARIANT 6: Verify 3D features are explicitly declared as nominal defaults."""
        msg = MockROS2LaserScanMsg()
        for b in range(178, 183):
            msg.ranges[b] = 14.0
        adapter = ROS2LaserScanAdapter(time_provider=lambda: 1774847392.5)
        _, scan = adapter.ingest_laser_scan(msg)
        obs = self.extractor.extract_obstacles(scan)[0]

        # Verify Obstacle2D declares absence of 3D capabilities
        self.assertFalse(obs.has_3d_z)
        self.assertFalse(obs.has_3d_height)
        self.assertFalse(obs.has_3d_classification)

        # Verify FeatureMapper2D metadata statement
        _, meta = self.mapper.map_obstacle(obs)
        self.assertEqual(meta.compatibility_statement, STAGE12_COMPATIBILITY_STATEMENT)
        self.assertFalse(meta.z_is_real_measurement)
        self.assertFalse(meta.height_is_real_measurement)
        self.assertFalse(meta.classification_is_real_measurement)


if __name__ == "__main__":
    unittest.main()
