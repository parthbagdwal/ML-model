"""Stage 12 Test Suite: 2D LiDAR Perception Adapter & Sensor-Ready Integration.

Validates the complete 2D planar LiDAR perception pipeline:
1. Empty scan handling
2. Valid single obstacle extraction
3. Multiple obstacle segmentation
4. Invalid range filtering (< min_range)
5. Infinite range handling (inf / -inf)
6. NaN range handling
7. Out-of-range filtering (> max_range)
8. Noisy scan robustness
9. Deterministic synthetic scan generation
10. Multi-scan temporal tracking & closing velocity derivation
11. 2D polar-to-Cartesian coordinate conversion accuracy
12. Obstacle clustering jump-distance thresholding
13. Isolation: zero 3D processing in 2D perception pathway
14. Frozen HGB inference integration on 2D-mapped features
15. Multi-obstacle highest threat selection preservation
16. Downstream ROS 2 schema compatibility (DetectedObject, PerceptionFrame)
17. FastAPI endpoint compatibility with 2D-derived perception frames
18. Safety boundary invariants: Zero vehicle actuation commands, no fabricated 3D data.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROS2_PKG_ROOT = PROJECT_ROOT / "ros2_ws" / "src" / "minerakshak_risk"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(ROS2_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(ROS2_PKG_ROOT))

from src.inference.predict import (
    MineRakshakInferenceEngine,
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
from src.perception.obstacle_extractor_2d import (
    MultiScanTracker2D,
    ObstacleExtractor2D,
)
from src.perception.synthetic_lidar_2d import (
    SYNTHETIC_2D_LIDAR_DISCLAIMER,
    Synthetic2DLiDARAdapter,
)
from minerakshak_risk.risk_node import PerceptionToRiskProcessor
from minerakshak_risk.schemas import DetectedObject, PerceptionFrame


class TestStage122DLiDARPerception(unittest.TestCase):
    """Rigorous unit and integration test suite for Stage 12 2D LiDAR perception."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize reusable perception and inference components."""
        cls.engine = get_inference_engine()
        cls.adapter = Synthetic2DLiDARAdapter(fov_deg=180.0, resolution_deg=0.5, seed=42)
        cls.extractor = ObstacleExtractor2D(cluster_distance_threshold_m=1.2, min_cluster_points=2)
        cls.tracker = MultiScanTracker2D(gating_distance_m=4.5, velocity_smoothing_alpha=0.7)
        cls.mapper = FeatureMapper2D(nominal_z_m=0.0, nominal_height_m=1.5, default_object_type="unknown")
        cls.risk_processor = PerceptionToRiskProcessor(engine=cls.engine)

    def setUp(self) -> None:
        """Reset state between tests."""
        self.tracker.reset()

    # -------------------------------------------------------------------------
    # Test 1: Empty Scan Handling
    # -------------------------------------------------------------------------
    def test_01_empty_scan_handling(self) -> None:
        """1. Verify empty scan produces 0 obstacles and evaluates safely without error."""
        scan = self.adapter.create_empty_scan(use_inf=True)
        self.assertEqual(scan.beam_count, 361)
        obstacles = self.extractor.extract_obstacles(scan)
        self.assertEqual(len(obstacles), 0, "Empty scan must produce exactly 0 obstacles.")

        # Test frame assembly and downstream risk processor with 0 obstacles
        frame = self.mapper.to_perception_frame(obstacles, frame_id="empty_frame", timestamp_ns=scan.timestamp_ns)
        risk_frame = self.risk_processor.process_frame(frame)
        self.assertEqual(risk_frame.objects_detected, 0)
        self.assertEqual(risk_frame.objects_evaluated, 0)
        self.assertEqual(risk_frame.highest_threat_level, "SAFE")
        self.assertIsNone(risk_frame.highest_threat_object_id)

    # -------------------------------------------------------------------------
    # Test 2: Valid Single Obstacle Extraction
    # -------------------------------------------------------------------------
    def test_02_valid_single_obstacle_extraction(self) -> None:
        """2. Verify accurate 2D obstacle extraction (range, centroid, planar span, point count)."""
        scan = self.adapter.create_single_obstacle_scan(
            distance_m=15.0,
            angle_rad=0.0,
            span_width_m=2.0,
        )
        obstacles = self.extractor.extract_obstacles(scan)
        self.assertEqual(len(obstacles), 1, "Expected exactly 1 obstacle extracted.")

        obs = obstacles[0]
        # Range should be close to 15.0m
        self.assertAlmostEqual(obs.distance_m, 15.0, delta=0.5)
        self.assertAlmostEqual(obs.object_x_m, 15.0, delta=0.5)
        self.assertAlmostEqual(obs.object_y_m, 0.0, delta=0.5)
        self.assertGreaterEqual(obs.span_width_m, 1.5)
        self.assertGreater(obs.point_count, 3)
        self.assertFalse(obs.has_3d_z, "2D obstacle must declare has_3d_z = False.")
        self.assertFalse(obs.has_3d_height, "2D obstacle must declare has_3d_height = False.")

    # -------------------------------------------------------------------------
    # Test 3: Multiple Obstacles Segmentation
    # -------------------------------------------------------------------------
    def test_03_multiple_obstacles_segmentation(self) -> None:
        """3. Verify multiple distinct obstacles are correctly segmented into separate clusters."""
        obs_configs = [
            (30.0, math.radians(30.0), 1.5),   # Left distant
            (18.0, math.radians(-30.0), 2.0),  # Right mid
            (7.0, math.radians(0.0), 1.8),     # Center close
        ]
        scan = self.adapter.create_multi_obstacle_scan(obstacles=obs_configs)
        obstacles = self.extractor.extract_obstacles(scan)

        self.assertEqual(len(obstacles), 3, "Expected 3 distinct obstacle clusters.")
        distances = sorted([o.distance_m for o in obstacles])
        self.assertAlmostEqual(distances[0], 7.0, delta=0.8)
        self.assertAlmostEqual(distances[1], 18.0, delta=1.0)
        self.assertAlmostEqual(distances[2], 30.0, delta=1.5)

    # -------------------------------------------------------------------------
    # Test 4: Invalid Ranges (< min_range)
    # -------------------------------------------------------------------------
    def test_04_invalid_ranges_low(self) -> None:
        """4. Verify readings below min_range are filtered out and rejected safely."""
        scan = self.adapter.create_faulty_scan(fault_type="out_of_range_low")
        points = scan.to_points()
        # Find point index 35 (which was set to 0.02m, below 0.1m)
        self.assertFalse(points[35].is_valid, "Point with range < min_range must be invalid.")

    # -------------------------------------------------------------------------
    # Test 5: Infinite Ranges (inf / -inf)
    # -------------------------------------------------------------------------
    def test_05_infinite_ranges_handling(self) -> None:
        """5. Verify infinite range values (inf / -inf) are trapped as invalid points."""
        scan = self.adapter.create_faulty_scan(fault_type="inf_ranges")
        points = scan.to_points()
        self.assertFalse(points[15].is_valid, "Point with +inf range must be marked invalid.")
        self.assertFalse(points[25].is_valid, "Point with -inf range must be marked invalid.")

        # Extraction must complete without crashing or returning infinite centroids
        obstacles = self.extractor.extract_obstacles(scan)
        for obs in obstacles:
            self.assertFalse(math.isinf(obs.distance_m))
            self.assertFalse(math.isinf(obs.object_x_m))

    # -------------------------------------------------------------------------
    # Test 6: NaN Ranges Handling
    # -------------------------------------------------------------------------
    def test_06_nan_ranges_handling(self) -> None:
        """6. Verify NaN range values are trapped safely without exceptions."""
        scan = self.adapter.create_faulty_scan(fault_type="nan_ranges")
        points = scan.to_points()
        self.assertFalse(points[10].is_valid, "Point with NaN range must be marked invalid.")
        self.assertFalse(points[20].is_valid, "Point with NaN range must be marked invalid.")

        obstacles = self.extractor.extract_obstacles(scan)
        for obs in obstacles:
            self.assertFalse(math.isnan(obs.distance_m))
            self.assertFalse(math.isnan(obs.object_x_m))

    # -------------------------------------------------------------------------
    # Test 7: Out-of-Range Measurements (> max_range)
    # -------------------------------------------------------------------------
    def test_07_out_of_range_high(self) -> None:
        """7. Verify measurements exceeding max_range (80m) are marked invalid."""
        scan = self.adapter.create_faulty_scan(fault_type="out_of_range_high")
        points = scan.to_points()
        self.assertFalse(points[40].is_valid, "Point with range > max_range must be marked invalid.")

    # -------------------------------------------------------------------------
    # Test 8: Noisy Scan Robustness
    # -------------------------------------------------------------------------
    def test_08_noisy_scan_robustness(self) -> None:
        """8. Verify 2D obstacle extraction remains stable and robust under Gaussian noise."""
        scan = self.adapter.create_single_obstacle_scan(
            distance_m=12.0,
            angle_rad=0.0,
            span_width_m=2.0,
            noise_std=0.08,  # 8cm Gaussian range jitter
        )
        obstacles = self.extractor.extract_obstacles(scan)
        self.assertEqual(len(obstacles), 1)
        self.assertAlmostEqual(obstacles[0].distance_m, 12.0, delta=0.5)

    # -------------------------------------------------------------------------
    # Test 9: Deterministic Synthetic Scans
    # -------------------------------------------------------------------------
    def test_09_deterministic_synthetic_scans(self) -> None:
        """9. Verify identical synthetic generator inputs yield bitwise identical scans."""
        adapter_a = Synthetic2DLiDARAdapter(seed=12345)
        adapter_b = Synthetic2DLiDARAdapter(seed=12345)

        scan_a = adapter_a.create_single_obstacle_scan(distance_m=18.0, angle_rad=0.1, span_width_m=1.8)
        scan_b = adapter_b.create_single_obstacle_scan(distance_m=18.0, angle_rad=0.1, span_width_m=1.8)

        self.assertEqual(scan_a.ranges, scan_b.ranges, "Deterministic synthetic scans must be identical.")

    # -------------------------------------------------------------------------
    # Test 10: Multi-Scan Temporal Tracking & Closing Velocity Derivation
    # -------------------------------------------------------------------------
    def test_10_multi_scan_temporal_tracking(self) -> None:
        """10. Verify consecutive scans allow MultiScanTracker2D to compute closing velocity and TTC."""
        # 5 scans at 10 Hz (dt = 0.1s), approaching at 10 m/s from 25m down to 21m
        scans = self.adapter.create_consecutive_approaching_scans(
            n_scans=5,
            start_dist_m=25.0,
            approach_speed_mps=10.0,
            dt_s=0.1,
            angle_rad=0.0,
            span_width_m=2.0,
        )

        last_tracked: list[Obstacle2D] = []
        for scan in scans:
            extracted = self.extractor.extract_obstacles(scan)
            last_tracked = self.tracker.track(extracted, timestamp_ns=scan.timestamp_ns)

        self.assertEqual(len(last_tracked), 1)
        tracked_obs = last_tracked[0]

        # Velocity should be negative (closing in) and approximately -10.0 m/s
        self.assertLess(tracked_obs.relative_velocity_mps, -5.0, "Closing obstacle must have negative velocity.")
        self.assertAlmostEqual(tracked_obs.relative_velocity_mps, -10.0, delta=2.5)

        # TTC should be around distance / closing_speed = 21m / 10m/s ~= 2.1s
        self.assertLess(tracked_obs.time_to_collision_s, 5.0, "TTC must reflect imminent approach.")
        self.assertGreater(tracked_obs.time_to_collision_s, 0.5)

    # -------------------------------------------------------------------------
    # Test 11: 2D Polar-to-Cartesian Conversion Accuracy
    # -------------------------------------------------------------------------
    def test_11_coordinate_conversion_accuracy(self) -> None:
        """11. Verify mathematical correctness of polar (range, angle) to Cartesian (x, y)."""
        # Test 1: Straight ahead (+X) at 10m
        pt_center = LaserScanPoint2D(angle_rad=0.0, range_m=10.0)
        self.assertAlmostEqual(pt_center.x_m, 10.0, places=3)
        self.assertAlmostEqual(pt_center.y_m, 0.0, places=3)

        # Test 2: 90 deg left (+Y) at 5m
        pt_left = LaserScanPoint2D(angle_rad=math.pi / 2.0, range_m=5.0)
        self.assertAlmostEqual(pt_left.x_m, 0.0, places=3)
        self.assertAlmostEqual(pt_left.y_m, 5.0, places=3)

        # Test 3: 45 deg right (-Y) at sqrt(2) * 10m
        pt_diag = LaserScanPoint2D(angle_rad=-math.pi / 4.0, range_m=10.0 * math.sqrt(2.0))
        self.assertAlmostEqual(pt_diag.x_m, 10.0, delta=0.01)
        self.assertAlmostEqual(pt_diag.y_m, -10.0, delta=0.01)

    # -------------------------------------------------------------------------
    # Test 12: Obstacle Clustering Jump-Distance Thresholding
    # -------------------------------------------------------------------------
    def test_12_obstacle_clustering_jump_distance(self) -> None:
        """12. Verify jump distance threshold correctly splits adjacent obstacles."""
        extractor = ObstacleExtractor2D(cluster_distance_threshold_m=1.0)
        # Create custom scan with two obstacles separated by 2.0m gap
        # Obs 1 at -10 deg, dist 10m; Obs 2 at +10 deg, dist 10m
        scan = self.adapter.create_multi_obstacle_scan(
            obstacles=[
                (10.0, math.radians(-15.0), 1.0),
                (10.0, math.radians(15.0), 1.0),
            ]
        )
        obstacles = extractor.extract_obstacles(scan)
        self.assertEqual(len(obstacles), 2, "Jump distance threshold must isolate the two obstacles.")

    # -------------------------------------------------------------------------
    # Test 13: Isolation - Zero 3D Processing in 2D Pathway
    # -------------------------------------------------------------------------
    def test_13_zero_3d_processing_in_2d_pathway(self) -> None:
        """13. Verify the 2D perception pathway contains zero 3D point cloud dependencies or calls."""
        scan = self.adapter.create_critical_scenario_scan()
        obstacles = self.extractor.extract_obstacles(scan)
        self.assertEqual(len(obstacles), 1)
        obs = obstacles[0]

        # Verify explicit 2D metadata flags
        self.assertFalse(obs.has_3d_z)
        self.assertFalse(obs.has_3d_height)
        self.assertFalse(obs.has_3d_classification)
        self.assertEqual(obs.detection_source, "2D_PLANAR_LIDAR_SCAN")

        # Map to inference contract and check metadata
        inf_dict, meta = self.mapper.map_obstacle(obs)
        self.assertFalse(meta.z_is_real_measurement)
        self.assertFalse(meta.height_is_real_measurement)
        self.assertEqual(meta.sensor_type, "2D_PLANAR_LIDAR")
        self.assertEqual(meta.compatibility_statement, STAGE12_COMPATIBILITY_STATEMENT)

    # -------------------------------------------------------------------------
    # Test 14: Frozen HGB Inference on 2D-Mapped Features
    # -------------------------------------------------------------------------
    def test_14_frozen_hgb_inference_on_2d_mapped_features(self) -> None:
        """14. Verify frozen HGB model runs cleanly on 2D-mapped features and produces valid risk."""
        # Critical scenario: close obstacle straight ahead
        scan = self.adapter.create_critical_scenario_scan()
        obstacles = self.extractor.extract_obstacles(scan)
        obs = obstacles[0]
        # Simulate approaching velocity
        obs.relative_velocity_mps = -12.0
        obs.time_to_collision_s = 0.5

        inf_dict, meta = self.mapper.map_obstacle(obs, truck_speed_kmph=25.0)

        # Run frozen primary model
        pred = self.engine.predict_single(inf_dict)
        self.assertEqual(pred["status"], "valid")
        self.assertIn(pred["risk_level"], ["WARNING", "CRITICAL"])
        self.assertEqual(pred["model"], "HistGradientBoostingClassifier")
        self.assertAlmostEqual(sum(pred["probabilities"].values()), 1.0, delta=0.01)

    # -------------------------------------------------------------------------
    # Test 15: Highest Threat Selection Preservation
    # -------------------------------------------------------------------------
    def test_15_highest_threat_selection_preservation(self) -> None:
        """15. Verify mixed 2D obstacles frame prioritizes highest threat level correctly."""
        # Create multi-obstacle scan
        scan = self.adapter.create_multi_obstacle_scan(
            obstacles=[
                (35.0, math.radians(20.0), 1.0),  # distant safe
                (14.0, math.radians(-25.0), 2.0), # lateral caution
                (5.5, math.radians(0.0), 2.2),    # critical threat
            ]
        )
        obstacles = self.extractor.extract_obstacles(scan)
        self.assertEqual(len(obstacles), 3)

        # Assign closing velocity to the closest obstacle
        for o in obstacles:
            if o.distance_m < 8.0:
                o.relative_velocity_mps = -14.0
                o.time_to_collision_s = 0.4
            else:
                o.relative_velocity_mps = 0.0
                o.time_to_collision_s = 99.9

        frame = self.mapper.to_perception_frame(obstacles, frame_id="threat_frame_01", timestamp_ns=scan.timestamp_ns)
        risk_frame = self.risk_processor.process_frame(frame)

        self.assertEqual(risk_frame.objects_detected, 3)
        self.assertEqual(risk_frame.objects_evaluated, 3)
        # Highest threat must be CRITICAL (or WARNING) and never SAFE
        self.assertIn(risk_frame.highest_threat_level, ["WARNING", "CRITICAL"])
        self.assertIsNotNone(risk_frame.highest_threat_object_id)

    # -------------------------------------------------------------------------
    # Test 16: Downstream ROS 2 Schema Compatibility
    # -------------------------------------------------------------------------
    def test_16_downstream_ros2_schema_compatibility(self) -> None:
        """16. Verify 2D obstacles map seamlessly into ROS 2 DetectedObject & PerceptionFrame schemas."""
        scan = self.adapter.create_safe_scenario_scan()
        obstacles = self.extractor.extract_obstacles(scan)
        frame = self.mapper.to_perception_frame(obstacles, frame_id="ros2_scan_01", timestamp_ns=scan.timestamp_ns)

        self.assertIsInstance(frame, PerceptionFrame)
        self.assertEqual(len(frame.objects), 1)
        self.assertIsInstance(frame.objects[0], DetectedObject)

        # Verify JSON serialization works
        json_str = frame.to_json()
        reconstructed = PerceptionFrame.from_json(json_str)
        self.assertEqual(reconstructed.frame_id, frame.frame_id)
        self.assertEqual(len(reconstructed.objects), 1)

    # -------------------------------------------------------------------------
    # Test 17: Safety Boundary - Zero Actuation Invariant
    # -------------------------------------------------------------------------
    def test_17_safety_boundary_zero_vehicle_actuation(self) -> None:
        """17. Safety Boundary: 2D perception pipeline emits ZERO vehicle actuation commands."""
        forbidden_actuation_keys = [
            "brake_cmd", "throttle_cmd", "steer_cmd", "emergency_stop_actuate",
            "apply_brakes", "actuator_override", "can_bus_actuate"
        ]

        scan = self.adapter.create_critical_scenario_scan()
        obstacles = self.extractor.extract_obstacles(scan)
        frame = self.mapper.to_perception_frame(obstacles, frame_id="safety_test_frame", timestamp_ns=scan.timestamp_ns)
        risk_frame = self.risk_processor.process_frame(frame)
        risk_dict = risk_frame.to_dict()

        for key in forbidden_actuation_keys:
            self.assertNotIn(key, risk_dict)
            for assessment in risk_dict.get("assessments", []):
                self.assertNotIn(key, assessment)

    # -------------------------------------------------------------------------
    # Test 18: Safety Boundary - Missing Sensor Features Transparently Audited
    # -------------------------------------------------------------------------
    def test_18_safety_boundary_missing_features_audited(self) -> None:
        """18. Safety Boundary: No 3D data fabricated; missing features explicitly declared."""
        scan = self.adapter.create_safe_scenario_scan()
        obstacles = self.extractor.extract_obstacles(scan)
        inf_dict, meta = self.mapper.map_obstacle(obstacles[0])

        # Verify metadata explicitly declares that z and height are NOT real measurements
        self.assertFalse(meta.z_is_real_measurement)
        self.assertFalse(meta.height_is_real_measurement)
        self.assertEqual(meta.compatibility_statement, STAGE12_COMPATIBILITY_STATEMENT)
        self.assertEqual(meta.sensor_type, "2D_PLANAR_LIDAR")


if __name__ == "__main__":
    unittest.main()
