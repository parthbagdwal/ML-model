"""Stage 12: Synthetic 2D LiDAR Adapter.

===============================================================================
MANDATORY SENSOR DISCLAIMER:
SYNTHETIC 2D LIDAR DATA — SOFTWARE TEST HARNESS ONLY
This adapter generates synthetic planar laser scan readings strictly for
software verification, unit testing, and benchmarking of 2D perception logic.
It does NOT connect to physical sensor hardware. Real 2D LiDAR hardware is
NOT currently available, and this synthetic data MUST NOT be described or
treated as real-sensor validation.
===============================================================================

Provides deterministic 2D planar range scans matching standard LiDAR angular
resolutions (e.g. 180 deg field-of-view, 0.5 deg beam increment = 361 beams).
"""

from __future__ import annotations

import math
import random
import time
from typing import Any, Sequence

from src.perception.lidar_2d import LaserScan2D, LaserScanPoint2D, PerceptionSource

SYNTHETIC_2D_LIDAR_DISCLAIMER = (
    "SYNTHETIC 2D LIDAR DATA: Generated strictly for software test verification. "
    "Does NOT connect to physical hardware. Real 2D LiDAR sensor readings are "
    "not currently available, and this data is NOT real sensor validation."
)


class Synthetic2DLiDARAdapter(PerceptionSource):
    """Deterministic synthetic 2D LiDAR adapter for planar scan simulation."""

    def __init__(
        self,
        fov_deg: float = 180.0,
        resolution_deg: float = 0.5,
        range_min: float = 0.1,
        range_max: float = 80.0,
        default_intensity: float = 100.0,
        seed: int | None = 42,
    ) -> None:
        """Initialize synthetic planar scanner.

        Args:
            fov_deg: Total field-of-view in degrees (default 180 degrees, from -90 to +90).
            resolution_deg: Angular increment between beams in degrees (default 0.5 degrees).
            range_min: Minimum reliable sensor range in meters.
            range_max: Maximum detectable sensor range in meters.
            default_intensity: Nominal reflectivity return intensity.
            seed: Optional random seed for reproducible noise generation.
        """
        self.fov_rad = math.radians(fov_deg)
        self.angle_min = -self.fov_rad / 2.0
        self.angle_max = self.fov_rad / 2.0
        self.angle_increment = math.radians(resolution_deg)
        self.range_min = range_min
        self.range_max = range_max
        self.default_intensity = default_intensity

        # Compute number of beams
        self.num_beams = int(round((self.angle_max - self.angle_min) / self.angle_increment)) + 1
        self.rng = random.Random(seed)
        self.frame_counter = 0

    def get_source_name(self) -> str:
        return "Synthetic2DLiDARAdapter"

    def is_synthetic(self) -> bool:
        return True

    def get_sensor_dimensionality(self) -> str:
        return "2D"

    def get_latest_scan(self) -> LaserScan2D | None:
        """Generate and return default scan with an obstacle."""
        return self.create_safe_scenario_scan()

    def next_frame_id(self, prefix: str = "scan2d") -> str:
        self.frame_counter += 1
        return f"{prefix}_{self.frame_counter:06d}"

    def _base_empty_ranges(self, free_space_val: float = float("inf")) -> list[float]:
        """Generate a baseline empty scan where all beams register free space."""
        return [free_space_val] * self.num_beams

    def _inject_obstacle(
        self,
        ranges: list[float],
        intensities: list[float],
        center_distance_m: float,
        center_angle_rad: float,
        span_width_m: float,
        noise_std: float = 0.0,
    ) -> None:
        """Inject a planar obstacle signature into the beam array.

        Computes which angular rays intersect an obstacle of given planar width
        at a given distance and angle, and fills the range values accordingly.
        """
        # Tangential angular half-span: theta_half = atan2(span/2, distance)
        half_width = max(0.1, span_width_m / 2.0)
        angular_half_span = math.atan2(half_width, center_distance_m)

        for i in range(self.num_beams):
            beam_angle = self.angle_min + (i * self.angle_increment)
            angle_diff = abs(beam_angle - center_angle_rad)

            if angle_diff <= angular_half_span:
                # Radial distance to chord profile
                # Projected range for flat or curved obstacle profile
                base_r = center_distance_m / max(0.001, math.cos(angle_diff))
                if noise_std > 0.0:
                    base_r += self.rng.gauss(0.0, noise_std)

                # Clamp to sensor limits
                clamped_r = max(self.range_min, min(self.range_max, base_r))

                # If this beam is closer than existing reading, update it
                if clamped_r < ranges[i]:
                    ranges[i] = round(clamped_r, 4)
                    intensities[i] = self.default_intensity

    # -------------------------------------------------------------------------
    # Scenario Generation Methods
    # -------------------------------------------------------------------------

    def create_empty_scan(
        self,
        frame_id: str | None = None,
        timestamp_ns: int | None = None,
        use_inf: bool = True,
    ) -> LaserScan2D:
        """Generate scan with zero obstacles (all beams report max range or inf)."""
        fid = frame_id or self.next_frame_id("empty_scan")
        t_ns = timestamp_ns if timestamp_ns is not None else int(time.time() * 1e9)
        free_val = float("inf") if use_inf else self.range_max

        ranges = self._base_empty_ranges(free_space_val=free_val)
        intensities = [0.0] * self.num_beams

        return LaserScan2D(
            frame_id=fid,
            timestamp_ns=t_ns,
            angle_min=self.angle_min,
            angle_max=self.angle_max,
            angle_increment=self.angle_increment,
            range_min=self.range_min,
            range_max=self.range_max,
            ranges=ranges,
            intensities=intensities,
        )

    def create_single_obstacle_scan(
        self,
        distance_m: float = 20.0,
        angle_rad: float = 0.0,
        span_width_m: float = 2.0,
        noise_std: float = 0.0,
        frame_id: str | None = None,
        timestamp_ns: int | None = None,
    ) -> LaserScan2D:
        """Generate scan containing exactly one planar obstacle."""
        fid = frame_id or self.next_frame_id("single_obs")
        t_ns = timestamp_ns if timestamp_ns is not None else int(time.time() * 1e9)

        ranges = self._base_empty_ranges()
        intensities = [0.0] * self.num_beams

        self._inject_obstacle(
            ranges=ranges,
            intensities=intensities,
            center_distance_m=distance_m,
            center_angle_rad=angle_rad,
            span_width_m=span_width_m,
            noise_std=noise_std,
        )

        return LaserScan2D(
            frame_id=fid,
            timestamp_ns=t_ns,
            angle_min=self.angle_min,
            angle_max=self.angle_max,
            angle_increment=self.angle_increment,
            range_min=self.range_min,
            range_max=self.range_max,
            ranges=ranges,
            intensities=intensities,
        )

    def create_safe_scenario_scan(self, frame_id: str | None = None) -> LaserScan2D:
        """Distant obstacle outside collision trajectory (approx 36m, +8 deg angle)."""
        return self.create_single_obstacle_scan(
            distance_m=36.0,
            angle_rad=math.radians(8.0),
            span_width_m=0.8,
            frame_id=frame_id or self.next_frame_id("safe_scan"),
        )

    def create_caution_scenario_scan(self, frame_id: str | None = None) -> LaserScan2D:
        """Lateral obstacle in adjacent lane/corridor (approx 14m, -40 deg angle)."""
        return self.create_single_obstacle_scan(
            distance_m=14.0,
            angle_rad=math.radians(-40.0),
            span_width_m=2.2,
            frame_id=frame_id or self.next_frame_id("caution_scan"),
        )

    def create_warning_scenario_scan(self, frame_id: str | None = None) -> LaserScan2D:
        """Obstacle in travel lane ahead at medium distance (approx 15m, -7 deg angle)."""
        return self.create_single_obstacle_scan(
            distance_m=15.0,
            angle_rad=math.radians(-7.0),
            span_width_m=1.0,
            frame_id=frame_id or self.next_frame_id("warning_scan"),
        )

    def create_critical_scenario_scan(self, frame_id: str | None = None) -> LaserScan2D:
        """Imminent collision threat straight ahead at short distance (approx 6m, 0 deg angle)."""
        return self.create_single_obstacle_scan(
            distance_m=6.0,
            angle_rad=math.radians(2.0),
            span_width_m=2.4,
            frame_id=frame_id or self.next_frame_id("critical_scan"),
        )

    def create_multi_obstacle_scan(
        self,
        obstacles: Sequence[tuple[float, float, float]] | None = None,
        noise_std: float = 0.0,
        frame_id: str | None = None,
    ) -> LaserScan2D:
        """Generate scan with multiple distinct planar obstacles.

        Args:
            obstacles: Sequence of (distance_m, angle_rad, span_width_m) tuples.
        """
        fid = frame_id or self.next_frame_id("multi_obs")
        ranges = self._base_empty_ranges()
        intensities = [0.0] * self.num_beams

        obs_list = obstacles or [
            (35.0, math.radians(25.0), 1.5),   # Distant left
            (18.0, math.radians(-30.0), 2.2),  # Mid right
            (8.5, math.radians(0.0), 2.0),     # Close center
        ]

        for dist, ang, width in obs_list:
            self._inject_obstacle(
                ranges=ranges,
                intensities=intensities,
                center_distance_m=dist,
                center_angle_rad=ang,
                span_width_m=width,
                noise_std=noise_std,
            )

        return LaserScan2D(
            frame_id=fid,
            timestamp_ns=int(time.time() * 1e9),
            angle_min=self.angle_min,
            angle_max=self.angle_max,
            angle_increment=self.angle_increment,
            range_min=self.range_min,
            range_max=self.range_max,
            ranges=ranges,
            intensities=intensities,
        )

    def create_faulty_scan(
        self,
        fault_type: str = "nan_ranges",
        frame_id: str | None = None,
    ) -> LaserScan2D:
        """Inject faulty sensor readings to test robustness.

        Fault types:
        - 'nan_ranges': Injects NaN values into range array.
        - 'inf_ranges': Injects Infinite values.
        - 'negative_ranges': Injects negative range readings (< 0.0).
        - 'out_of_range_low': Injects readings below range_min.
        - 'out_of_range_high': Injects readings far exceeding range_max.
        """
        scan = self.create_single_obstacle_scan(
            distance_m=12.0,
            angle_rad=0.0,
            span_width_m=2.0,
            frame_id=frame_id or self.next_frame_id("fault_scan"),
        )
        ranges = list(scan.ranges)

        if fault_type == "nan_ranges":
            ranges[10] = float("nan")
            ranges[20] = float("nan")
        elif fault_type == "inf_ranges":
            ranges[15] = float("inf")
            ranges[25] = float("-inf")
        elif fault_type == "negative_ranges":
            ranges[30] = -5.0
        elif fault_type == "out_of_range_low":
            ranges[35] = 0.02  # below range_min (0.1m)
        elif fault_type == "out_of_range_high":
            ranges[40] = 250.0  # above range_max (80m)

        scan.ranges = ranges
        return scan

    def create_consecutive_approaching_scans(
        self,
        n_scans: int = 5,
        start_dist_m: float = 30.0,
        approach_speed_mps: float = 10.0,
        dt_s: float = 0.1,
        angle_rad: float = 0.0,
        span_width_m: float = 2.0,
    ) -> list[LaserScan2D]:
        """Generate a temporal sequence of consecutive scans for multi-scan velocity tracking tests.

        Args:
            n_scans: Number of consecutive scan frames to generate.
            start_dist_m: Initial obstacle distance.
            approach_speed_mps: Closing speed in m/s (approaching).
            dt_s: Time step between consecutive scans (e.g. 0.1s for 10 Hz).
            angle_rad: Bearing angle of obstacle.
            span_width_m: Planar obstacle width.

        Returns:
            List of LaserScan2D instances with increasing timestamps and decreasing distances.
        """
        base_t_ns = int(time.time() * 1e9)
        dt_ns = int(dt_s * 1e9)
        scans: list[LaserScan2D] = []

        for k in range(n_scans):
            dist_k = max(1.0, start_dist_m - (k * approach_speed_mps * dt_s))
            timestamp_k = base_t_ns + (k * dt_ns)
            fid = f"seq_scan_{k:03d}"

            scan = self.create_single_obstacle_scan(
                distance_m=dist_k,
                angle_rad=angle_rad,
                span_width_m=span_width_m,
                frame_id=fid,
                timestamp_ns=timestamp_k,
            )
            scans.append(scan)

        return scans
