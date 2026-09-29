"""Stage 14: 2D LiDAR Sensor Configuration, Profiles & Health State Definitions.

Defines configurable hardware, operational parameters, and calibration presets for
2D planar LiDAR sensors, along with explicit sensor-health states to decouple data
integrity from downstream ML risk classification.

CRITICAL SENSOR CONVENTIONS & CONSTRAINTS:
- Coordinate Convention:
    +X: Forward travel direction of haul truck
    +Y: Lateral left relative to truck centerline
    Z: NOT MEASURED (2D planar LiDAR; elevation is physically unavailable)
- Sensor Health State (OK, NO_DATA, STALE, INVALID, DEGRADED) represents physical /
  telemetry data stream integrity.
- ML Risk Level (SAFE, CAUTION, WARNING, CRITICAL) represents situational hazard.
- These two state machines must never be conflated.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class SensorHealthState(str, Enum):
    """Explicit health and integrity states for 2D LiDAR data streams."""

    OK = "OK"                  # Fresh, physically valid scan within nominal parameters
    NO_DATA = "NO_DATA"        # No scan has been received yet (startup / uninitialized)
    STALE = "STALE"            # Scan received but timestamp exceeds acceptable latency threshold
    INVALID = "INVALID"        # Malformed scan geometry, corrupted arrays, or impossible parameters
    DEGRADED = "DEGRADED"      # High ratio of invalid beams (dust/slurry backscatter or partial occlusion)


@dataclass
class SensorValidationResult:
    """Detailed result of a LaserScan validation pass."""

    is_valid: bool
    health_state: SensorHealthState
    message: str = ""
    valid_beam_count: int = 0
    total_beam_count: int = 0
    valid_beam_ratio: float = 0.0
    age_s: float = 0.0
    timestamp_gap_s: float = 0.0
    quality_report: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert validation result to dictionary."""
        data: dict[str, Any] = {
            "is_valid": self.is_valid,
            "health_state": self.health_state.value,
            "message": self.message,
            "valid_beam_count": self.valid_beam_count,
            "total_beam_count": self.total_beam_count,
            "valid_beam_ratio": round(self.valid_beam_ratio, 4),
            "age_s": round(self.age_s, 4),
            "timestamp_gap_s": round(self.timestamp_gap_s, 4),
        }
        if self.quality_report is not None:
            data["quality_report"] = self.quality_report
        return data


@dataclass
class LiDAR2DConfig:
    """Configurable sensor parameters for physical or simulated 2D planar LiDAR."""

    # Sensor Identification & Model Profile
    sensor_model_name: str = "Generic_2D_Planar_LiDAR"
    calibration_status: str = "CONFIGURATION_DEFAULT"  # CONFIGURATION_DEFAULT or FIELD_CALIBRATED

    # ROS 2 Interface Parameters
    topic_name: str = "/scan"
    expected_frame_id: str = "laser_frame"
    enforce_frame_id: bool = False

    # Sensor Physical Bounds
    range_min_m: float = 0.1
    range_max_m: float = 80.0
    expected_fov_deg: float = 180.0
    angular_resolution_deg: float = 0.5

    # Scan Timing & Frequency Parameters
    expected_scan_frequency_hz: float = 20.0
    min_scan_frequency_hz: float = 5.0
    max_scan_frequency_hz: float = 100.0

    # Mounting Offsets relative to Truck Base Center / Front Bumper
    # Coordinate Convention:
    # +X: Forward travel direction
    # +Y: Lateral left
    # Z: NOT measured (2D planar sensor)
    mounting_yaw_offset_rad: float = 0.0  # Counter-clockwise rotation of sensor 0 rad relative to +X
    mounting_x_offset_m: float = 0.0      # Longitudinal offset (m)
    mounting_y_offset_m: float = 0.0      # Lateral offset (m)

    # Coordinate Convention Declarations
    heading_convention: str = "+X_FORWARD"
    lateral_convention: str = "+Y_LEFT"
    elevation_convention: str = "Z_UNAVAILABLE"

    # Timing & Health Thresholds
    stale_timeout_s: float = 0.5          # Scans older than this are flagged STALE (500 ms)
    max_timestamp_gap_s: float = 1.0      # Scans separated by more than this trigger timing warning
    min_valid_beam_ratio: float = 0.15    # If valid beams fall below 15%, state becomes DEGRADED / INVALID
    max_future_skew_s: float | None = None  # If set, scans ahead by more than this threshold are flagged INVALID

    # Vehicle Speed Fallback (when external vehicle odometry is temporarily missing)
    default_truck_speed_kmph: float = 25.0

    def validate_config(self) -> list[str]:
        """Validate configuration parameters and return list of error descriptions."""
        errors: list[str] = []
        if self.range_min_m < 0.0:
            errors.append("range_min_m cannot be negative")
        if self.range_max_m <= 0.0:
            errors.append("range_max_m must be strictly positive")
        if self.range_min_m >= self.range_max_m:
            errors.append("range_min_m must be strictly less than range_max_m")
        if self.expected_fov_deg <= 0.0 or self.expected_fov_deg > 360.0:
            errors.append("expected_fov_deg must be in (0, 360]")
        if self.angular_resolution_deg <= 0.0 or self.angular_resolution_deg > 45.0:
            errors.append("angular_resolution_deg must be in (0, 45]")
        if self.expected_scan_frequency_hz <= 0.0:
            errors.append("expected_scan_frequency_hz must be strictly positive")
        if self.min_scan_frequency_hz <= 0.0 or self.max_scan_frequency_hz <= self.min_scan_frequency_hz:
            errors.append("Invalid scan frequency bounds (min_scan_frequency_hz must be > 0 and < max_scan_frequency_hz)")
        if self.stale_timeout_s <= 0.0:
            errors.append("stale_timeout_s must be strictly positive")
        if self.max_timestamp_gap_s <= 0.0:
            errors.append("max_timestamp_gap_s must be strictly positive")
        if self.min_valid_beam_ratio < 0.0 or self.min_valid_beam_ratio > 1.0:
            errors.append("min_valid_beam_ratio must be between 0.0 and 1.0")
        return errors

    @property
    def is_valid(self) -> bool:
        """Return True if configuration is valid."""
        return len(self.validate_config()) == 0

    @property
    def expected_beam_count(self) -> int:
        """Calculate expected beam count based on FOV and angular resolution."""
        return int(round(self.expected_fov_deg / self.angular_resolution_deg)) + 1

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LiDAR2DConfig:
        """Create configuration from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    # -------------------------------------------------------------------------
    # Commercial Sensor Preset Profiles
    # -------------------------------------------------------------------------

    @classmethod
    def create_sick_tim781_preset(cls) -> LiDAR2DConfig:
        """Preset configuration for SICK TiM781 industrial outdoor 2D LiDAR."""
        return cls(
            sensor_model_name="SICK_TiM781",
            expected_frame_id="sick_laser",
            range_min_m=0.05,
            range_max_m=25.0,
            expected_fov_deg=270.0,
            angular_resolution_deg=0.33,
            expected_scan_frequency_hz=15.0,
            min_scan_frequency_hz=10.0,
            max_scan_frequency_hz=20.0,
        )

    @classmethod
    def create_hokuyo_ust20lx_preset(cls) -> LiDAR2DConfig:
        """Preset configuration for Hokuyo UST-20LX high-speed 2D LiDAR."""
        return cls(
            sensor_model_name="Hokuyo_UST_20LX",
            expected_frame_id="hokuyo_laser",
            range_min_m=0.06,
            range_max_m=20.0,
            expected_fov_deg=270.0,
            angular_resolution_deg=0.25,
            expected_scan_frequency_hz=40.0,
            min_scan_frequency_hz=25.0,
            max_scan_frequency_hz=50.0,
        )

    @classmethod
    def create_rplidar_s2_preset(cls) -> LiDAR2DConfig:
        """Preset configuration for RPLIDAR S2 360-degree planar scanner."""
        return cls(
            sensor_model_name="RPLIDAR_S2",
            expected_frame_id="rplidar_laser",
            range_min_m=0.05,
            range_max_m=30.0,
            expected_fov_deg=360.0,
            angular_resolution_deg=0.12,
            expected_scan_frequency_hz=10.0,
            min_scan_frequency_hz=5.0,
            max_scan_frequency_hz=15.0,
        )

    @classmethod
    def create_generic_planar_preset(cls) -> LiDAR2DConfig:
        """Default generic 180-degree 20Hz mining forward scanner."""
        return cls()
