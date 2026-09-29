"""Stage 12: 2D LiDAR Perception Data Structures & Sensor Interface.

Defines sensor-independent data models and abstract interfaces for 2D planar LiDAR
systems. This module explicitly separates planar 2D scan data from 3D point-cloud
assumptions.

Architecture highlights:
- `LaserScanPoint2D`: Individual planar beam measurement (angle, range, intensity, Cartesian x, y).
- `LaserScan2D`: Standard 2D planar scan matching the ROS 2 sensor_msgs/msg/LaserScan definition.
- `Obstacle2D`: Planar obstacle cluster extracted from a 2D scan (centroid, planar span, return count).
- `PerceptionSource`: Sensor-independent abstract base class for perception adapters.

CRITICAL SENSOR CONVENTION:
A 2D LiDAR provides only planar range and angle measurements. It does NOT provide
elevation (z), 3D bounding boxes, vertical height, 3D length, or object classification.
"""

from __future__ import annotations

import json
import math
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence


@dataclass
class LaserScanPoint2D:
    """Represents a single range measurement from a 2D planar LiDAR beam."""

    angle_rad: float
    range_m: float
    intensity: float = 0.0
    x_m: float = 0.0
    y_m: float = 0.0
    is_valid: bool = True

    def __post_init__(self) -> None:
        """Convert polar beam measurement (range, angle) to Cartesian (x, y) in vehicle frame.

        Vehicle Frame Convention:
        +X: Forward travel heading
        +Y: Lateral left (+Y) / right (-Y)
        Angle 0.0 rad points along +X (straight ahead).
        """
        if self.is_valid and not math.isnan(self.range_m) and not math.isinf(self.range_m) and self.range_m > 0:
            self.x_m = round(self.range_m * math.cos(self.angle_rad), 4)
            self.y_m = round(self.range_m * math.sin(self.angle_rad), 4)
        else:
            self.x_m = 0.0
            self.y_m = 0.0
            self.is_valid = False

    def to_dict(self) -> dict[str, Any]:
        """Convert point to dictionary representation."""
        return {
            "angle_rad": round(self.angle_rad, 4),
            "range_m": round(self.range_m, 4),
            "intensity": round(self.intensity, 2),
            "x_m": self.x_m,
            "y_m": self.y_m,
            "is_valid": self.is_valid,
        }


@dataclass
class LaserScan2D:
    """Standard 2D planar laser scan matching ROS 2 sensor_msgs/msg/LaserScan.

    Compatible with standard 2D LiDAR drivers (e.g. SICK, Hokuyo, RPLIDAR, Velodyne 2D mode).
    """

    frame_id: str = "laser_frame"
    timestamp_ns: int = 0
    angle_min: float = -math.pi / 2.0  # -90 degrees
    angle_max: float = math.pi / 2.0   # +90 degrees
    angle_increment: float = math.radians(0.5)  # 0.5 deg resolution
    time_increment: float = 0.0
    scan_time: float = 0.05  # 20 Hz
    range_min: float = 0.1   # meters
    range_max: float = 80.0  # meters
    ranges: list[float] = field(default_factory=list)
    intensities: list[float] = field(default_factory=list)
    mounting_yaw_offset_rad: float = 0.0  # Rotation relative to +X vehicle forward
    mounting_x_offset_m: float = 0.0      # Longitudinal offset (m)
    mounting_y_offset_m: float = 0.0      # Lateral offset (m)

    def __post_init__(self) -> None:
        if self.timestamp_ns == 0:
            self.timestamp_ns = int(time.time() * 1e9)
        if not self.intensities and self.ranges:
            self.intensities = [0.0] * len(self.ranges)

    @property
    def beam_count(self) -> int:
        """Total number of beams in this scan."""
        return len(self.ranges)

    def get_angle(self, index: int) -> float:
        """Compute the polar angle for beam at index."""
        return self.angle_min + (index * self.angle_increment)

    def to_points(self) -> list[LaserScanPoint2D]:
        """Convert all scan beams into Cartesian LaserScanPoint2D objects with validation.

        Transforms measurements into the vehicle coordinate frame:
        +X: Forward travel direction
        +Y: Lateral left
        Z: NOT measured (2D planar sensor)
        """
        points: list[LaserScanPoint2D] = []
        has_offsets = (
            self.mounting_yaw_offset_rad != 0.0
            or self.mounting_x_offset_m != 0.0
            or self.mounting_y_offset_m != 0.0
        )
        for i, r in enumerate(self.ranges):
            angle = self.get_angle(i)
            intensity = self.intensities[i] if i < len(self.intensities) else 0.0

            # Validate range bounds
            is_valid = True
            if math.isnan(r) or math.isinf(r) or r < self.range_min or r > self.range_max:
                is_valid = False

            if has_offsets:
                veh_angle = angle + self.mounting_yaw_offset_rad
                pt = LaserScanPoint2D(
                    angle_rad=veh_angle,
                    range_m=r,
                    intensity=intensity,
                    is_valid=is_valid,
                )
                if pt.is_valid:
                    pt.x_m = round(pt.x_m + self.mounting_x_offset_m, 4)
                    pt.y_m = round(pt.y_m + self.mounting_y_offset_m, 4)
                points.append(pt)
            else:
                points.append(
                    LaserScanPoint2D(
                        angle_rad=angle,
                        range_m=r,
                        intensity=intensity,
                        is_valid=is_valid,
                    )
                )
        return points

    def to_dict(self) -> dict[str, Any]:
        """Serialize scan metadata and ranges to dictionary."""
        return {
            "frame_id": self.frame_id,
            "timestamp_ns": self.timestamp_ns,
            "angle_min": round(self.angle_min, 4),
            "angle_max": round(self.angle_max, 4),
            "angle_increment": round(self.angle_increment, 6),
            "time_increment": self.time_increment,
            "scan_time": self.scan_time,
            "range_min": self.range_min,
            "range_max": self.range_max,
            "beam_count": len(self.ranges),
            "mounting_yaw_offset_rad": round(self.mounting_yaw_offset_rad, 4),
            "mounting_x_offset_m": round(self.mounting_x_offset_m, 4),
            "mounting_y_offset_m": round(self.mounting_y_offset_m, 4),
            "ranges": [round(r, 4) if not (math.isnan(r) or math.isinf(r)) else None for r in self.ranges],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LaserScan2D:
        """Construct LaserScan2D from dictionary."""
        ranges_raw = data.get("ranges", [])
        clean_ranges: list[float] = []
        for r in ranges_raw:
            if r is None:
                clean_ranges.append(float("nan"))
            else:
                clean_ranges.append(float(r))

        return cls(
            frame_id=str(data.get("frame_id", "laser_frame")),
            timestamp_ns=int(data.get("timestamp_ns", 0)),
            angle_min=float(data.get("angle_min", -math.pi / 2.0)),
            angle_max=float(data.get("angle_max", math.pi / 2.0)),
            angle_increment=float(data.get("angle_increment", math.radians(0.5))),
            time_increment=float(data.get("time_increment", 0.0)),
            scan_time=float(data.get("scan_time", 0.05)),
            range_min=float(data.get("range_min", 0.1)),
            range_max=float(data.get("range_max", 80.0)),
            ranges=clean_ranges,
            intensities=[float(x) for x in data.get("intensities", [])],
            mounting_yaw_offset_rad=float(data.get("mounting_yaw_offset_rad", 0.0)),
            mounting_x_offset_m=float(data.get("mounting_x_offset_m", 0.0)),
            mounting_y_offset_m=float(data.get("mounting_y_offset_m", 0.0)),
        )

    @classmethod
    def from_ros2_laser_scan_dict(
        cls,
        ros2_dict: dict[str, Any],
        mounting_yaw_offset_rad: float = 0.0,
        mounting_x_offset_m: float = 0.0,
        mounting_y_offset_m: float = 0.0,
    ) -> LaserScan2D:
        """Construct from ROS 2 sensor_msgs/msg/LaserScan serialized dictionary."""
        header = ros2_dict.get("header", {})
        stamp = header.get("stamp", {})
        sec = stamp.get("sec", 0) if isinstance(stamp, dict) else getattr(stamp, "sec", 0)
        nanosec = stamp.get("nanosec", 0) if isinstance(stamp, dict) else getattr(stamp, "nanosec", 0)
        timestamp_ns = (sec * 1_000_000_000) + nanosec
        if timestamp_ns == 0:
            timestamp_ns = int(time.time() * 1e9)

        frame_id = header.get("frame_id", "laser_frame") if isinstance(header, dict) else getattr(header, "frame_id", "laser_frame")

        return cls(
            frame_id=str(frame_id),
            timestamp_ns=timestamp_ns,
            angle_min=float(ros2_dict.get("angle_min", -math.pi / 2.0)),
            angle_max=float(ros2_dict.get("angle_max", math.pi / 2.0)),
            angle_increment=float(ros2_dict.get("angle_increment", 0.0087266)),
            time_increment=float(ros2_dict.get("time_increment", 0.0)),
            scan_time=float(ros2_dict.get("scan_time", 0.05)),
            range_min=float(ros2_dict.get("range_min", 0.1)),
            range_max=float(ros2_dict.get("range_max", 80.0)),
            ranges=[float(r) for r in ros2_dict.get("ranges", [])],
            intensities=[float(i) for i in ros2_dict.get("intensities", [])],
            mounting_yaw_offset_rad=mounting_yaw_offset_rad,
            mounting_x_offset_m=mounting_x_offset_m,
            mounting_y_offset_m=mounting_y_offset_m,
        )


@dataclass
class Obstacle2D:
    """Represents a 2D planar obstacle cluster extracted from a 2D LiDAR scan.

    Contains exclusively planar measurements and explicitly notes the absence of 3D data.
    """

    obstacle_id: str
    distance_m: float
    object_x_m: float
    object_y_m: float
    span_width_m: float
    depth_length_m: float
    point_count: int
    relative_velocity_mps: float = 0.0
    time_to_collision_s: float = 99.9
    tracking_id: int | None = None
    timestamp_ns: int = 0
    is_valid: bool = True
    detection_source: str = "2D_LIDAR_SCAN"

    # Explicit sensor capability declarations
    has_3d_z: bool = False
    has_3d_height: bool = False
    has_3d_classification: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize 2D obstacle to dictionary."""
        return {
            "obstacle_id": self.obstacle_id,
            "distance_m": round(self.distance_m, 2),
            "object_x_m": round(self.object_x_m, 2),
            "object_y_m": round(self.object_y_m, 2),
            "span_width_m": round(self.span_width_m, 2),
            "depth_length_m": round(self.depth_length_m, 2),
            "point_count": self.point_count,
            "relative_velocity_mps": round(self.relative_velocity_mps, 2),
            "time_to_collision_s": round(self.time_to_collision_s, 2),
            "tracking_id": self.tracking_id,
            "timestamp_ns": self.timestamp_ns,
            "is_valid": self.is_valid,
            "detection_source": self.detection_source,
            "has_3d_z": self.has_3d_z,
            "has_3d_height": self.has_3d_height,
            "has_3d_classification": self.has_3d_classification,
        }


class PerceptionSource(ABC):
    """Abstract base class defining the contract for perception input sources."""

    @abstractmethod
    def get_source_name(self) -> str:
        """Return human-readable identifier for this perception source."""
        pass

    @abstractmethod
    def is_synthetic(self) -> bool:
        """Return True if this source produces synthetic/simulated data."""
        pass

    @abstractmethod
    def get_sensor_dimensionality(self) -> str:
        """Return '2D' or '3D'."""
        pass

    @abstractmethod
    def get_latest_scan(self) -> LaserScan2D | None:
        """Retrieve the latest available 2D planar scan."""
        pass
