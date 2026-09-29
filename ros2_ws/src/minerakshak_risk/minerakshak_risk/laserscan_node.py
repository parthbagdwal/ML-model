"""Stage 13: ROS 2 2D LiDAR LaserScan Subscription & Risk Processing Node.

Subscribes directly to physical or simulated 2D planar LiDAR via ROS 2 `/scan`
(`sensor_msgs/msg/LaserScan`), validates scan integrity and timing, runs the 2D
perception pipeline (clustering -> tracking -> feature mapping), executes frozen
HGB risk inference, and publishes structured risk evaluations to `/minerakshak/risk_prediction`
and `/minerakshak/highest_threat`.

Architecture Flow:
REAL 2D LIDAR / ROS 2 /scan (sensor_msgs/msg/LaserScan)
  ↓
ROS2LaserScanAdapter (geometry, bounds, timing, health validation)
  ↓
ObstacleExtractor2D (planar Euclidean jump-distance clustering)
  ↓
MultiScanTracker2D (temporal tracking, delta_d/delta_t closing velocity)
  ↓
FeatureMapper2D (maps 2D features + nominal baselines to frozen contract)
  ↓
Frozen HistGradientBoostingClassifier (models/hgb_model.joblib)
  ↓
RiskAssessmentFrame (published to ROS 2 topics & forwarded to FastAPI/WebSockets)

CRITICAL SAFETY & ENVIRONMENT SEPARATION:
- If native ROS 2 (rclpy) is available: instantiates Node and registers actual /scan subscription.
- If native ROS 2 is NOT installed: provides dependency-isolated test adapter and pipeline
  processor for deterministic unit testing and benchmarking without faking ROS runtime.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

# Ensure project root is accessible
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predict import (
    MineRakshakInferenceEngine,
    RESPONSE_MAPPINGS,
    SAFETY_DISCLAIMER,
    get_inference_engine,
)
from src.perception.feature_mapper_2d import FeatureMapper2D
from src.perception.lidar_2d import LaserScan2D, Obstacle2D
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

# Optional ROS 2 imports with safe environment isolation
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import LaserScan
    from std_msgs.msg import String
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    Node = object
    LaserScan = None
    String = None


class LaserScanToRiskPipeline:
    """Decoupled perception-to-risk pipeline processor consuming 2D LaserScans.

    Can be executed and rigorously benchmarked in standard Python environments
    independent of the ROS 2 middleware daemon.
    """

    def __init__(
        self,
        config: LiDAR2DConfig | None = None,
        adapter: ROS2LaserScanAdapter | None = None,
        extractor: ObstacleExtractor2D | None = None,
        tracker: MultiScanTracker2D | None = None,
        mapper: FeatureMapper2D | None = None,
        risk_processor: PerceptionToRiskProcessor | None = None,
        default_truck_speed_kmph: float = 25.0,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config or LiDAR2DConfig()
        self.adapter = adapter or ROS2LaserScanAdapter(config=self.config)
        self.extractor = extractor or ObstacleExtractor2D(cluster_distance_threshold_m=1.2, min_cluster_points=2)
        self.tracker = tracker or MultiScanTracker2D(gating_distance_m=4.5, velocity_smoothing_alpha=0.7)
        self.mapper = mapper or FeatureMapper2D(nominal_z_m=0.0, nominal_height_m=1.5)
        self.risk_processor = risk_processor or PerceptionToRiskProcessor()
        self.default_truck_speed_kmph = default_truck_speed_kmph
        self.logger = logger or logging.getLogger("LaserScanToRiskPipeline")

        self.total_scans_processed = 0
        self.total_faults_isolated = 0

    def process_scan(
        self,
        scan_msg: Any,
        truck_speed_kmph: float | None = None,
        current_time_s: float | None = None,
        include_secondary: bool = False,
    ) -> tuple[SensorValidationResult, RiskAssessmentFrame]:
        """Ingest a 2D LaserScan message and execute the full risk prediction pipeline.

        Args:
            scan_msg: ROS 2 LaserScan message, dictionary, or LaserScan2D object.
            truck_speed_kmph: External vehicle speed signal (CAN/odometry).
            current_time_s: Reference evaluation timestamp (s).
            include_secondary: Whether to include secondary XGBoost voter.

        Returns:
            Tuple of (SensorValidationResult, RiskAssessmentFrame).
        """
        t_start = time.perf_counter()
        speed = truck_speed_kmph if truck_speed_kmph is not None else self.default_truck_speed_kmph

        # 1. Validation & Conversion
        val_res, scan2d = self.adapter.ingest_laser_scan(scan_msg, current_time_s=current_time_s)

        # Handle validation failures (STALE, INVALID, NO_DATA)
        if not val_res.is_valid or scan2d is None:
            self.total_faults_isolated += 1
            frame_id = getattr(scan_msg, "frame_id", "invalid_frame") if scan_msg else "no_scan"
            if isinstance(scan_msg, dict):
                frame_id = scan_msg.get("frame_id", scan_msg.get("header", {}).get("frame_id", "invalid_frame"))

            fault_frame = RiskAssessmentFrame(
                frame_id=str(frame_id),
                timestamp_ns=int(getattr(scan_msg, "timestamp_ns", 0)) if scan_msg else 0,
                inference_timestamp=datetime.now(timezone.utc).isoformat(),
                processing_time_ms=(time.perf_counter() - t_start) * 1000.0,
                objects_detected=0,
                objects_evaluated=0,
                highest_threat_level="SAFE",
                highest_threat_object_id=None,
                assessments=[],
                errors=[{
                    "error": val_res.message,
                    "sensor_health_state": val_res.health_state.value,
                    "age_s": str(val_res.age_s),
                }],
                sensor_source=self.adapter.get_source_name(),
                source_type="synthetic" if self.adapter.is_synthetic() else "real_ros2",
                hardware_validated=False,
                sensor_health=val_res.health_state.value,
                feature_compatibility_status="COMPATIBLE_WITH_DEFAULTS",
            )
            return val_res, fault_frame

        # 2. Extract Obstacle Clusters
        extracted_obstacles = self.extractor.extract_obstacles(scan2d)

        # 3. Multi-Scan Temporal Tracking
        tracked_obstacles = self.tracker.track(extracted_obstacles, timestamp_ns=scan2d.timestamp_ns)

        # 4. Feature Mapping to Model Schema Contract
        perception_frame = self.mapper.to_perception_frame(
            obstacles=tracked_obstacles,
            truck_speed_kmph=speed,
            frame_id=scan2d.frame_id,
            timestamp_ns=scan2d.timestamp_ns,
            sensor_source=self.adapter.get_source_name(),
            source_type="synthetic" if self.adapter.is_synthetic() else "real_ros2",
            hardware_validated=False,
            sensor_health=val_res.health_state.value,
            feature_compatibility_status="COMPATIBLE_WITH_DEFAULTS",
        )

        # 5. Frozen ML Risk Inference
        risk_frame = self.risk_processor.process_frame(perception_frame, include_secondary=include_secondary)

        self.total_scans_processed += 1
        return val_res, risk_frame

    def reset_tracker(self) -> None:
        """Reset temporal tracking state."""
        self.tracker.reset()


class MineRakshakLaserScanNode(Node):
    """ROS 2 Node subscribing to `/scan` (sensor_msgs/msg/LaserScan)."""

    def __init__(self, node_name: str = "minerakshak_laserscan_node") -> None:
        self.node_name = node_name
        self.pipeline = LaserScanToRiskPipeline()

        if not ROS2_AVAILABLE:
            return

        super().__init__(node_name)

        # Declare parameters
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("risk_topic", "/minerakshak/risk_prediction")
        self.declare_parameter("threat_topic", "/minerakshak/highest_threat")
        self.declare_parameter("health_topic", "/minerakshak/sensor_health")
        self.declare_parameter("stale_timeout_s", 0.5)
        self.declare_parameter("default_truck_speed_kmph", 25.0)

        scan_topic = self.get_parameter("scan_topic").get_parameter_value().string_value
        risk_topic = self.get_parameter("risk_topic").get_parameter_value().string_value
        threat_topic = self.get_parameter("threat_topic").get_parameter_value().string_value
        health_topic = self.get_parameter("health_topic").get_parameter_value().string_value

        self.subscription = self.create_subscription(
            LaserScan,
            scan_topic,
            self.scan_callback,
            10,
        )
        self.risk_publisher = self.create_publisher(String, risk_topic, 10)
        self.threat_publisher = self.create_publisher(String, threat_topic, 10)
        self.health_publisher = self.create_publisher(String, health_topic, 10)

        self.get_logger().info(
            f"MineRakshakLaserScanNode started. Subscribed to '{scan_topic}', "
            f"publishing risk to '{risk_topic}'."
        )

    def scan_callback(self, msg: Any) -> tuple[SensorValidationResult, RiskAssessmentFrame]:
        """Callback invoked when a LaserScan is received from the ROS 2 driver."""
        val_res, risk_frame = self.pipeline.process_scan(msg)

        if ROS2_AVAILABLE and hasattr(self, "risk_publisher") and self.risk_publisher is not None:
            # 1. Publish Risk Assessment Frame
            risk_msg = String()
            risk_msg.data = risk_frame.to_json()
            self.risk_publisher.publish(risk_msg)

            # 2. Publish Threat Summary
            threat_msg = String()
            threat_msg.data = json.dumps({
                "frame_id": risk_frame.frame_id,
                "highest_threat_level": risk_frame.highest_threat_level,
                "highest_threat_object_id": risk_frame.highest_threat_object_id,
                "action": RESPONSE_MAPPINGS.get(risk_frame.highest_threat_level, "no immediate hazard"),
                "sensor_health": val_res.health_state.value,
                "timestamp": risk_frame.inference_timestamp,
            })
            self.threat_publisher.publish(threat_msg)

            # 3. Publish Sensor Health
            health_msg = String()
            health_msg.data = json.dumps(val_res.to_dict())
            self.health_publisher.publish(health_msg)

        return val_res, risk_frame

    def destroy_node(self) -> bool:
        if ROS2_AVAILABLE and hasattr(super(), "destroy_node"):
            return super().destroy_node()
        return True


def main(args: list[str] | None = None) -> None:
    """ROS 2 entry point."""
    if not ROS2_AVAILABLE:
        print("[ERROR] ROS 2 (rclpy) is not installed in the current Python environment.")
        print("To execute this node inside a ROS 2 workspace:")
        print("  1. Ensure ROS 2 (Humble / Iron / Rolling) is installed.")
        print("  2. In your workspace: colcon build --packages-select minerakshak_risk")
        print("  3. source install/setup.bash (or call install/setup.bat)")
        print("  4. ros2 run minerakshak_risk laserscan_node")
        sys.exit(1)

    rclpy.init(args=args)
    node = MineRakshakLaserScanNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
