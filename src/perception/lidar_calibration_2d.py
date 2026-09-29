"""Stage 14: 2D LiDAR Calibration, Coordinate Verification & Geometry Framework.

===============================================================================
MANDATORY SENSOR CALIBRATION PRINCIPLES:
1. Zero 3D Elevation Fabrication:
   A 2D planar LiDAR measures ONLY in its scan plane. The Z-coordinate (elevation)
   is physically unmeasured. It must NOT be fabricated.
2. Coordinate Orientation Convention:
   +X: Forward travel direction along haul truck heading
   +Y: Lateral left relative to centerline
   Z: EXPLICITLY UNAVAILABLE from sensor measurements
3. Calibration Provenance:
   Calibration parameters default to nominal engineering specifications
   ("CONFIGURATION_DEFAULT"). When empirical survey or target calibration is
   performed on the physical haul truck, parameters transition to "FIELD_CALIBRATED".
===============================================================================
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from src.perception.lidar_2d import LaserScan2D, LaserScanPoint2D
from src.perception.sensor_config_2d import LiDAR2DConfig

CALIBRATION_DISCLAIMER: str = (
    "CALIBRATION FRAMEWORK — CONFIGURATION SPECIFICATIONS: "
    "Default calibration values represent nominal configuration defaults, not measured "
    "hardware values. Field calibration is required upon physical sensor installation."
)


@dataclass
class CalibrationVerificationReport:
    """Report evaluating whether an observed scan matches configured calibration parameters."""

    is_calibrated: bool
    sensor_model: str
    calibration_status: str
    frame_id: str
    checks_passed: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    mounting_yaw_deg: float = 0.0
    mounting_x_offset_m: float = 0.0
    mounting_y_offset_m: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "is_calibrated": self.is_calibrated,
            "sensor_model": self.sensor_model,
            "calibration_status": self.calibration_status,
            "frame_id": self.frame_id,
            "checks_passed": self.checks_passed,
            "warnings": self.warnings,
            "errors": self.errors,
            "mounting_yaw_deg": round(self.mounting_yaw_deg, 3),
            "mounting_x_offset_m": round(self.mounting_x_offset_m, 3),
            "mounting_y_offset_m": round(self.mounting_y_offset_m, 3),
        }


class LiDARCalibration2D:
    """Rigorous mathematical calibration and coordinate transformation engine for 2D LiDAR."""

    def __init__(self, config: LiDAR2DConfig | None = None) -> None:
        """Initialize calibration engine with sensor configuration."""
        self.config = config or LiDAR2DConfig()

    @property
    def calibration_status(self) -> str:
        """Return provenance of calibration values ('CONFIGURATION_DEFAULT' or 'FIELD_CALIBRATED')."""
        return self.config.calibration_status

    def transform_polar_to_vehicle_cartesian(
        self,
        range_m: float,
        angle_rad: float,
    ) -> tuple[float, float, bool]:
        """Transform polar beam return to vehicle frame Cartesian coordinates (+X forward, +Y left).

        Applies configured mounting yaw offset and positional displacements:
            theta_veh = angle_rad + mounting_yaw_offset_rad
            x_veh = range_m * cos(theta_veh) + mounting_x_offset_m
            y_veh = range_m * sin(theta_veh) + mounting_y_offset_m

        Returns:
            Tuple of (x_veh_m, y_veh_m, is_valid).
            If range is NaN, Inf, or outside physical sensor bounds, is_valid is False.
        """
        # Validate range physical integrity
        if (
            math.isnan(range_m)
            or math.isinf(range_m)
            or range_m < self.config.range_min_m
            or range_m > self.config.range_max_m
        ):
            return 0.0, 0.0, False

        theta_veh = angle_rad + self.config.mounting_yaw_offset_rad
        x_veh = round(range_m * math.cos(theta_veh) + self.config.mounting_x_offset_m, 4)
        y_veh = round(range_m * math.sin(theta_veh) + self.config.mounting_y_offset_m, 4)
        return x_veh, y_veh, True

    def verify_scan_calibration(self, scan: LaserScan2D) -> CalibrationVerificationReport:
        """Validate whether a received LaserScan2D conforms to the configured hardware profile."""
        checks_passed: list[str] = []
        warnings: list[str] = []
        errors: list[str] = []

        # 1. Frame ID check
        if scan.frame_id == self.config.expected_frame_id:
            checks_passed.append(f"Frame ID matches expected '{self.config.expected_frame_id}'")
        else:
            warnings.append(
                f"Frame ID mismatch: received '{scan.frame_id}', expected '{self.config.expected_frame_id}'"
            )

        # 2. Angle Bounds Check
        expected_fov_rad = math.radians(self.config.expected_fov_deg)
        observed_fov_rad = scan.angle_max - scan.angle_min
        if math.isclose(observed_fov_rad, expected_fov_rad, rel_tol=0.05):
            checks_passed.append(f"Observed FOV ({math.degrees(observed_fov_rad):.1f} deg) matches expected {self.config.expected_fov_deg:.1f} deg")
        else:
            warnings.append(
                f"Observed FOV ({math.degrees(observed_fov_rad):.1f} deg) deviates from configured {self.config.expected_fov_deg:.1f} deg"
            )

        # 3. Angular Resolution Check
        expected_inc = math.radians(self.config.angular_resolution_deg)
        if math.isclose(scan.angle_increment, expected_inc, rel_tol=0.05):
            checks_passed.append(f"Angular resolution ({math.degrees(scan.angle_increment):.3f} deg) matches expected")
        else:
            warnings.append(
                f"Angular resolution mismatch: scan={math.degrees(scan.angle_increment):.3f} deg, config={self.config.angular_resolution_deg:.3f} deg"
            )

        # 4. Range Limits Check
        if scan.range_min < 0.0 or scan.range_max <= scan.range_min:
            errors.append(f"Corrupt sensor range limits: min={scan.range_min}, max={scan.range_max}")
        else:
            checks_passed.append(f"Sensor range limits physically valid [{scan.range_min}m, {scan.range_max}m]")

        # 5. Beam Count Check
        if scan.beam_count == 0:
            errors.append("Scan contains zero beam measurements")
        else:
            checks_passed.append(f"Scan contains {scan.beam_count} beam measurements")

        is_calibrated = (len(errors) == 0)

        return CalibrationVerificationReport(
            is_calibrated=is_calibrated,
            sensor_model=self.config.sensor_model_name,
            calibration_status=self.config.calibration_status,
            frame_id=scan.frame_id,
            checks_passed=checks_passed,
            warnings=warnings,
            errors=errors,
            mounting_yaw_deg=math.degrees(self.config.mounting_yaw_offset_rad),
            mounting_x_offset_m=self.config.mounting_x_offset_m,
            mounting_y_offset_m=self.config.mounting_y_offset_m,
        )

    def set_field_calibration(
        self,
        mounting_yaw_offset_rad: float,
        mounting_x_offset_m: float,
        mounting_y_offset_m: float,
    ) -> None:
        """Apply measured empirical mounting calibration values from physical truck surveying."""
        self.config.mounting_yaw_offset_rad = mounting_yaw_offset_rad
        self.config.mounting_x_offset_m = mounting_x_offset_m
        self.config.mounting_y_offset_m = mounting_y_offset_m
        self.config.calibration_status = "FIELD_CALIBRATED"
