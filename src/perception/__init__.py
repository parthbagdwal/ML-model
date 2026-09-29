"""Perception package for MineRakshak AI.

Provides modular interfaces for:
1. Stage 13: Real 2D LiDAR ROS 2 interface, scan validation, replay & health monitoring.
2. Stage 12: 2D LiDAR planar perception (LaserScan2D, Obstacle2D, Synthetic2DLiDARAdapter,
   ObstacleExtractor2D, MultiScanTracker2D, FeatureMapper2D).
3. Stage 11: Legacy 3D LiDAR synthetic reference adapter (MockLiDARPerceptionAdapter)
   retained solely for regression testing.
"""

from src.perception.feature_mapper_2d import (
    FEATURE_COMPATIBILITY_MATRIX,
    STAGE12_COMPATIBILITY_STATEMENT,
    FeatureCompatibilityMetadata,
    FeatureMapper2D,
)
from src.perception.lidar_2d import (
    LaserScan2D,
    LaserScanPoint2D,
    Obstacle2D,
    PerceptionSource,
)
from src.perception.lidar_scan_replay import (
    AmbiguousReplayDatasetError,
    LiDARScanRecorder,
    LiDARScanReplayer,
)
from src.perception.mock_lidar_adapter import (
    PERCEPTION_ADAPTER_DISCLAIMER,
    MockLiDARPerceptionAdapter,
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
from src.perception.synthetic_lidar_2d import (
    SYNTHETIC_2D_LIDAR_DISCLAIMER,
    Synthetic2DLiDARAdapter,
)

from src.perception.lidar_calibration_2d import (
    CALIBRATION_DISCLAIMER,
    CalibrationVerificationReport,
    LiDARCalibration2D,
)
from src.perception.offline_validator_2d import (
    OFFLINE_TOOL_DISCLAIMER,
    OfflineValidator2D,
    ValidationSessionSummary,
)
from src.perception.scan_quality_2d import (
    SCAN_QUALITY_DISCLAIMER,
    ScanQualityAnalyzer2D,
    ScanQualityReport,
)

__all__ = [
    # Stage 14: Real-World 2D Calibration, Quality & Validation Framework
    "LiDARCalibration2D",
    "CalibrationVerificationReport",
    "CALIBRATION_DISCLAIMER",
    "ScanQualityAnalyzer2D",
    "ScanQualityReport",
    "SCAN_QUALITY_DISCLAIMER",
    "OfflineValidator2D",
    "ValidationSessionSummary",
    "OFFLINE_TOOL_DISCLAIMER",
    # Stage 13: Hardware-Ready ROS 2 & Replay Architecture
    "ROS2LaserScanAdapter",
    "HARDWARE_READY_DISCLAIMER",
    "LiDAR2DConfig",
    "SensorHealthState",
    "SensorValidationResult",
    "LiDARScanRecorder",
    "LiDARScanReplayer",
    "AmbiguousReplayDatasetError",
    # Stage 12: 2D Perception Architecture
    "LaserScanPoint2D",
    "LaserScan2D",
    "Obstacle2D",
    "PerceptionSource",
    "Synthetic2DLiDARAdapter",
    "SYNTHETIC_2D_LIDAR_DISCLAIMER",
    "ObstacleExtractor2D",
    "MultiScanTracker2D",
    "FeatureMapper2D",
    "FeatureCompatibilityMetadata",
    "STAGE12_COMPATIBILITY_STATEMENT",
    "FEATURE_COMPATIBILITY_MATRIX",
    # Legacy 3D Reference Architecture (Stages 10-11 Regression)
    "MockLiDARPerceptionAdapter",
    "PERCEPTION_ADAPTER_DISCLAIMER",
]
