"""Stage 14: 2D LiDAR Scan Quality Metrics & Diagnostic Analyzer.

===============================================================================
MANDATORY DIAGNOSTIC NOTICE:
DIAGNOSTIC METRICS ONLY — DOES NOT ALTER FROZEN ML MODEL
The metrics computed by this module provide telemetry and sensor health
diagnostics for haul truck operators, maintenance engineers, and ROS 2 health nodes.
They evaluate physical signal quality (dust backscatter, sensor face fouling,
dropped beams, out-of-bounds readings, and timing gaps).
They do NOT alter, retrain, or dynamically rescale the frozen ML model.
===============================================================================
"""

from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from src.perception.lidar_2d import LaserScan2D
from src.perception.sensor_config_2d import LiDAR2DConfig

SCAN_QUALITY_DISCLAIMER = "DIAGNOSTIC METRICS ONLY — DOES NOT ALTER FROZEN ML MODEL"


@dataclass
class ScanQualityReport:
    """Detailed diagnostic metrics computed for a single 2D planar laser scan."""

    frame_id: str
    timestamp_ns: int
    timestamp_age_s: float
    total_beam_count: int
    valid_beam_count: int
    invalid_beam_count: int
    nan_beam_count: int
    infinite_beam_count: int
    negative_beam_count: int
    below_min_beam_count: int
    above_max_beam_count: int
    valid_beam_percentage: float
    nan_beam_percentage: float
    infinite_beam_percentage: float
    invalid_beam_percentage: float
    min_valid_range_m: float | None
    max_valid_range_m: float | None
    median_valid_range_m: float | None
    mean_valid_range_m: float | None
    scan_coverage_deg: float
    angular_coverage_ratio: float
    scan_frequency_hz_estimate: float
    quality_assessment: str
    disclaimer: str = SCAN_QUALITY_DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        """Convert diagnostic report to dictionary representation."""
        return {
            "frame_id": self.frame_id,
            "timestamp_ns": self.timestamp_ns,
            "timestamp_age_s": round(self.timestamp_age_s, 4),
            "total_beam_count": self.total_beam_count,
            "valid_beam_count": self.valid_beam_count,
            "invalid_beam_count": self.invalid_beam_count,
            "nan_beam_count": self.nan_beam_count,
            "infinite_beam_count": self.infinite_beam_count,
            "negative_beam_count": self.negative_beam_count,
            "below_min_beam_count": self.below_min_beam_count,
            "above_max_beam_count": self.above_max_beam_count,
            "valid_beam_percentage": round(self.valid_beam_percentage, 2),
            "nan_beam_percentage": round(self.nan_beam_percentage, 2),
            "infinite_beam_percentage": round(self.infinite_beam_percentage, 2),
            "invalid_beam_percentage": round(self.invalid_beam_percentage, 2),
            "min_valid_range_m": round(self.min_valid_range_m, 3) if self.min_valid_range_m is not None else None,
            "max_valid_range_m": round(self.max_valid_range_m, 3) if self.max_valid_range_m is not None else None,
            "median_valid_range_m": round(self.median_valid_range_m, 3) if self.median_valid_range_m is not None else None,
            "mean_valid_range_m": round(self.mean_valid_range_m, 3) if self.mean_valid_range_m is not None else None,
            "scan_coverage_deg": round(self.scan_coverage_deg, 2),
            "angular_coverage_ratio": round(self.angular_coverage_ratio, 4),
            "scan_frequency_hz_estimate": round(self.scan_frequency_hz_estimate, 2),
            "quality_assessment": self.quality_assessment,
            "disclaimer": self.disclaimer,
        }


class ScanQualityAnalyzer2D:
    """Computes comprehensive signal and geometry quality diagnostics on 2D planar laser scans."""

    def __init__(self, config: LiDAR2DConfig | None = None) -> None:
        self.config = config or LiDAR2DConfig()
        self._last_timestamp_ns: int = 0

    def analyze_scan(
        self,
        scan: LaserScan2D,
        current_time_s: float | None = None,
    ) -> ScanQualityReport:
        """Evaluate physical and timing quality metrics for a given scan.

        Args:
            scan: LaserScan2D planar scan object.
            current_time_s: Optional reference time in seconds for age calculation.

        Returns:
            ScanQualityReport containing full diagnostic metrics.
        """
        now_s = current_time_s if current_time_s is not None else (scan.timestamp_ns / 1e9)
        scan_time_s = scan.timestamp_ns / 1e9 if scan.timestamp_ns > 0 else now_s
        age_s = max(0.0, now_s - scan_time_s)

        # Estimate scan frequency from timestamp delta
        freq_est = self.config.expected_scan_frequency_hz
        if self._last_timestamp_ns > 0 and scan.timestamp_ns > self._last_timestamp_ns:
            dt = (scan.timestamp_ns - self._last_timestamp_ns) / 1e9
            if dt > 0:
                freq_est = 1.0 / dt
        self._last_timestamp_ns = scan.timestamp_ns

        total_beams = scan.beam_count
        if total_beams == 0:
            return ScanQualityReport(
                frame_id=scan.frame_id,
                timestamp_ns=scan.timestamp_ns,
                timestamp_age_s=age_s,
                total_beam_count=0,
                valid_beam_count=0,
                invalid_beam_count=0,
                nan_beam_count=0,
                infinite_beam_count=0,
                negative_beam_count=0,
                below_min_beam_count=0,
                above_max_beam_count=0,
                valid_beam_percentage=0.0,
                nan_beam_percentage=0.0,
                infinite_beam_percentage=0.0,
                invalid_beam_percentage=100.0,
                min_valid_range_m=None,
                max_valid_range_m=None,
                median_valid_range_m=None,
                mean_valid_range_m=None,
                scan_coverage_deg=0.0,
                angular_coverage_ratio=0.0,
                scan_frequency_hz_estimate=freq_est,
                quality_assessment="CORRUPTED",
            )

        valid_ranges: list[float] = []
        valid_angles: list[float] = []

        nan_count = 0
        inf_count = 0
        negative_count = 0
        below_min_count = 0
        above_max_count = 0

        for i, r in enumerate(scan.ranges):
            angle = scan.get_angle(i)
            if math.isnan(r):
                nan_count += 1
            elif math.isinf(r):
                inf_count += 1
                if r < 0:
                    negative_count += 1
            elif r < 0.0:
                negative_count += 1
            elif r < scan.range_min:
                below_min_count += 1
            elif r > scan.range_max:
                above_max_count += 1
            else:
                # Physically valid obstacle return
                valid_ranges.append(r)
                valid_angles.append(angle)

        valid_count = len(valid_ranges)
        invalid_count = total_beams - valid_count

        valid_pct = (valid_count / total_beams) * 100.0
        nan_pct = (nan_count / total_beams) * 100.0
        inf_pct = (inf_count / total_beams) * 100.0
        invalid_pct = (invalid_count / total_beams) * 100.0

        min_r = min(valid_ranges) if valid_ranges else None
        max_r = max(valid_ranges) if valid_ranges else None
        median_r = statistics.median(valid_ranges) if valid_ranges else None
        mean_r = statistics.mean(valid_ranges) if valid_ranges else None

        # Angular coverage (span of valid returns)
        coverage_deg = 0.0
        coverage_ratio = 0.0
        total_fov_rad = scan.angle_max - scan.angle_min
        if valid_angles and total_fov_rad > 0:
            span_rad = max(valid_angles) - min(valid_angles)
            coverage_deg = math.degrees(span_rad)
            coverage_ratio = span_rad / total_fov_rad

        # Classify overall scan quality
        corrupted_beams = nan_count + negative_count + below_min_count
        corrupted_ratio = corrupted_beams / total_beams

        if nan_count == total_beams or negative_count == total_beams:
            assessment = "CORRUPTED"
        elif corrupted_ratio > (1.0 - self.config.min_valid_beam_ratio):
            assessment = "DEGRADED"
        elif valid_count == 0 and inf_count > 0:
            assessment = "EMPTY_FIELD"
        elif age_s > self.config.stale_timeout_s:
            assessment = "STALE"
        else:
            assessment = "HEALTHY"

        return ScanQualityReport(
            frame_id=scan.frame_id,
            timestamp_ns=scan.timestamp_ns,
            timestamp_age_s=age_s,
            total_beam_count=total_beams,
            valid_beam_count=valid_count,
            invalid_beam_count=invalid_count,
            nan_beam_count=nan_count,
            infinite_beam_count=inf_count,
            negative_beam_count=negative_count,
            below_min_beam_count=below_min_count,
            above_max_beam_count=above_max_count,
            valid_beam_percentage=valid_pct,
            nan_beam_percentage=nan_pct,
            infinite_beam_percentage=inf_pct,
            invalid_beam_percentage=invalid_pct,
            min_valid_range_m=min_r,
            max_valid_range_m=max_r,
            median_valid_range_m=median_r,
            mean_valid_range_m=mean_r,
            scan_coverage_deg=coverage_deg,
            angular_coverage_ratio=coverage_ratio,
            scan_frequency_hz_estimate=freq_est,
            quality_assessment=assessment,
        )

    def reset(self) -> None:
        """Reset historical timing state."""
        self._last_timestamp_ns = 0
