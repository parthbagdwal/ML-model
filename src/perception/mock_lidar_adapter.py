"""Stage 11: Legacy 3D LiDAR Synthetic Reference Adapter.

===============================================================================
LEGACY 3D REFERENCE NOTICE:
This adapter provides an isolated synthetic simulation of an upstream 3D LiDAR
point-cloud processing pipeline (ground removal + 3D bounding-box clustering).

It is retained SOLELY for regression and reference testing of Stage 10 and 11
validation suites. It is NOT the actual hardware configuration of this project
(which is a 2D planar LiDAR).

Distinction:
  - Actual Target Sensor: 2D Planar LiDAR (no height, no z, no 3D boxes)
  - Synthetic 2D Test Source: Synthetic2DLiDARAdapter (Stage 12)
  - Legacy 3D Reference: MockLiDARPerceptionAdapter (Retained for regression only)
===============================================================================

Conforms to the Stage 9/10 PerceptionFrame and DetectedObject schemas, generating
ROS 2-ready payload dictionaries and JSON structures for:
  - Standard operational mining scenarios (SAFE, CAUTION, WARNING, CRITICAL)
  - Multi-obstacle mixed frames
  - Variable obstacle density (1, 5, 10, 20, 50, 100 objects)
  - Controlled fault injection (negative distances, NaN/Inf, malformed fields, novel categories).
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure project root and ROS 2 package are on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROS2_PKG_ROOT = PROJECT_ROOT / "ros2_ws" / "src" / "minerakshak_risk"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(ROS2_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(ROS2_PKG_ROOT))

from minerakshak_risk.schemas import DetectedObject, PerceptionFrame

# Explicit disclaimer for this mock component
PERCEPTION_ADAPTER_DISCLAIMER = (
    "LEGACY 3D SYNTHETIC REFERENCE ADAPTER: Retained solely for regression testing of "
    "Stage 10/11 components. Does NOT represent actual 2D LiDAR hardware target or live sensor streams."
)


class MockLiDARPerceptionAdapter:
    """Synthetic perception adapter simulating 3D LiDAR clustering pipeline."""

    def __init__(self, default_truck_speed: float = 25.0) -> None:
        self.default_truck_speed = default_truck_speed
        self.frame_counter = 0

    def generate_timestamp_ns(self) -> int:
        """Return current monotonic UTC timestamp in nanoseconds."""
        return int(time.time() * 1e9)

    def next_frame_id(self, prefix: str = "lidar_frame") -> str:
        """Generate sequential frame identifier."""
        self.frame_counter += 1
        return f"{prefix}_{self.frame_counter:06d}"

    # ---------------------------------------------------------
    # Primitive Scenarios
    # ---------------------------------------------------------

    def create_safe_obstacle(self, object_id: str = "obs_safe_001") -> DetectedObject:
        """Distant pedestrian/object outside immediate collision path."""
        return DetectedObject(
            object_id=object_id,
            distance_m=36.2,
            object_x_m=35.77,
            object_y_m=5.45,
            object_z_m=-1.65,
            object_width_m=0.61,
            object_height_m=1.66,
            object_length_m=0.44,
            point_count=29,
            relative_velocity_mps=13.45,
            object_type="person",
            truck_speed_kmph=11.02,
            time_to_collision_s=99.9,
        )

    def create_caution_obstacle(self, object_id: str = "obs_caution_001") -> DetectedObject:
        """Support vehicle in lateral corridor requiring situational monitoring."""
        return DetectedObject(
            object_id=object_id,
            distance_m=12.8,
            object_x_m=3.11,
            object_y_m=-12.36,
            object_z_m=-1.86,
            object_width_m=1.66,
            object_height_m=1.36,
            object_length_m=4.57,
            point_count=1027,
            relative_velocity_mps=-12.66,
            object_type="car",
            truck_speed_kmph=10.89,
            time_to_collision_s=1.01,
        )

    def create_warning_obstacle(self, object_id: str = "obs_warning_001") -> DetectedObject:
        """Obstacle/person located inside truck travel lane with closing dynamics."""
        return DetectedObject(
            object_id=object_id,
            distance_m=15.68,
            object_x_m=15.49,
            object_y_m=-2.05,
            object_z_m=-1.62,
            object_width_m=0.48,
            object_height_m=1.95,
            object_length_m=0.53,
            point_count=68,
            relative_velocity_mps=-0.68,
            object_type="person",
            truck_speed_kmph=2.46,
            time_to_collision_s=22.94,
        )

    def create_critical_obstacle(self, object_id: str = "obs_critical_001") -> DetectedObject:
        """Imminent head-on collision threat at short range and high closing velocity."""
        return DetectedObject(
            object_id=object_id,
            distance_m=6.3,
            object_x_m=5.87,
            object_y_m=0.91,
            object_z_m=-2.12,
            object_width_m=1.83,
            object_height_m=1.48,
            object_length_m=5.0,
            point_count=789,
            relative_velocity_mps=-14.45,
            object_type="car",
            truck_speed_kmph=26.90,
            time_to_collision_s=0.44,
        )

    # ---------------------------------------------------------
    # Fault Injection Scenarios
    # ---------------------------------------------------------

    def create_invalid_obstacle(
        self,
        fault_type: str = "negative_distance",
        object_id: str = "obs_malformed_001",
    ) -> DetectedObject:
        """Generate physically impossible or corrupt obstacle observation."""
        base = self.create_safe_obstacle(object_id=object_id)
        if fault_type == "negative_distance":
            base.distance_m = -15.0
        elif fault_type == "negative_dimensions":
            base.object_width_m = -2.0
            base.object_height_m = -1.0
        elif fault_type == "negative_point_count":
            base.point_count = -50
        elif fault_type == "nan_velocity":
            base.relative_velocity_mps = float("nan")
        elif fault_type == "inf_distance":
            base.distance_m = float("inf")
        elif fault_type == "missing_required":
            base.distance_m = None
        else:
            base.distance_m = -999.0
        return base

    def create_unknown_type_obstacle(self, object_id: str = "obs_novel_drone_001") -> DetectedObject:
        """Generate obstacle with unmodeled categorical label."""
        base = self.create_safe_obstacle(object_id=object_id)
        base.object_type = "autonomous_inspection_drone"
        return base

    # ---------------------------------------------------------
    # Frame Assembly
    # ---------------------------------------------------------

    def create_perception_frame(
        self,
        objects: list[DetectedObject],
        frame_id: str | None = None,
        truck_speed_kmph: float | None = None,
    ) -> PerceptionFrame:
        """Assemble a complete PerceptionFrame from detected objects."""
        fid = frame_id or self.next_frame_id()
        speed = truck_speed_kmph if truck_speed_kmph is not None else self.default_truck_speed
        return PerceptionFrame(
            frame_id=fid,
            timestamp_ns=self.generate_timestamp_ns(),
            objects=objects,
            truck_speed_kmph=speed,
        )

    def create_mixed_threat_frame(self, frame_id: str | None = None) -> PerceptionFrame:
        """Perception frame containing SAFE, CAUTION, WARNING, and CRITICAL obstacles."""
        objs = [
            self.create_safe_obstacle("obj_safe_01"),
            self.create_caution_obstacle("obj_caution_02"),
            self.create_warning_obstacle("obj_warning_03"),
            self.create_critical_obstacle("obj_critical_04"),
        ]
        return self.create_perception_frame(objs, frame_id=frame_id)

    def create_density_frame(
        self,
        object_count: int,
        frame_id: str | None = None,
        include_threat: bool = True,
    ) -> PerceptionFrame:
        """Generate frame with N obstacles to benchmark perception-to-risk scaling."""
        fid = frame_id or f"density_frame_{object_count:03d}"
        objs: list[DetectedObject] = []

        for i in range(object_count):
            obj_id = f"density_obj_{i:03d}"
            # Scatter obstacles across realistic mining haul road geometry
            lateral = (i % 7 - 3) * 3.5  # lanes between -10.5m and +10.5m
            dist = 10.0 + (i * 1.5)  # distance staggered from 10m to 160m
            rel_vel = -2.0 if (lateral > -4.0 and lateral < 4.0) else 0.0

            objs.append(DetectedObject(
                object_id=obj_id,
                distance_m=round(dist, 2),
                object_x_m=round(math.sqrt(max(0.1, dist**2 - lateral**2)), 2),
                object_y_m=round(lateral, 2),
                object_z_m=-1.5,
                object_width_m=2.2,
                object_height_m=2.0,
                object_length_m=4.8,
                point_count=max(20, int(800.0 / math.sqrt(dist))),
                relative_velocity_mps=rel_vel,
                object_type="truck" if (i % 2 == 0) else "car",
                truck_speed_kmph=self.default_truck_speed,
                time_to_collision_s=round(dist / max(0.1, -rel_vel), 2) if rel_vel < 0 else 99.9,
            ))

        # Ensure at least one critical threat exists if requested
        if include_threat and objs:
            objs[0] = self.create_critical_obstacle("density_obj_critical_lead")

        return self.create_perception_frame(objs, frame_id=fid)

    def to_ros2_json(self, frame: PerceptionFrame) -> str:
        """Serialize PerceptionFrame into ROS 2 String JSON payload."""
        return frame.to_json() if hasattr(frame, "to_json") else json.dumps(asdict(frame))
